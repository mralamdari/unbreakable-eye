"""
main.py — FastAPI application entry point.

Manages:
  - Auth (session signing via itsdangerous, bcrypt via passlib)
  - Camera registration and pipeline lifecycle
  - MJPEG video streaming
  - JSON API for the frontend dashboard

Database backend: PostgreSQL via psycopg2.
All blocking DB calls are offloaded via run_db() to avoid blocking the event loop.
"""

import os
import cv2
import json
import asyncio
import uvicorn
from loguru import logger
import multiprocessing as mp
from contextlib import asynccontextmanager

import psycopg2
import psycopg2.errors

from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from passlib.context import CryptContext

from fastapi import FastAPI, HTTPException, Request, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.core.config import settings
from src.core.logging import setup_logging
from src.core.db_writer import start_db_writer
from src.core.database import init_db, load_cache, get_connection, write_connection
from src.engine.pipeline import VisionPipeline, batched_detector_worker


# ─────────────────────────────────────────────────────────────────────────────
# Security
# ─────────────────────────────────────────────────────────────────────────────
pwd_ctx         = CryptContext(schemes=["bcrypt"], deprecated="auto")
_signer         = URLSafeTimedSerializer(settings.SECRET_KEY)
SESSION_COOKIE  = "session"
SESSION_MAX_AGE = 60 * 60 * 8   # 8 hours


def _sign_session(user_id: int) -> str:
    return _signer.dumps({"uid": user_id})


def _verify_session(token: str) -> int | None:
    try:
        data = _signer.loads(token, max_age=SESSION_MAX_AGE)
        return int(data["uid"])
    except (BadSignature, SignatureExpired, KeyError):
        return None


def get_current_user(request: Request) -> int | None:
    token = request.cookies.get(SESSION_COOKIE)
    return _verify_session(token) if token else None


def require_login(request: Request) -> int:
    uid = get_current_user(request)
    if uid is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return uid


# ─────────────────────────────────────────────────────────────────────────────
# Flash messages
# ─────────────────────────────────────────────────────────────────────────────
def flash_message(response: Response, message: str, category: str = "info"):
    import datetime
    expires = datetime.datetime.now() + datetime.timedelta(seconds=20)
    payload = json.dumps({"message": message, "category": category})
    response.set_cookie(
        key="flash_msg", value=payload, max_age=10,
        expires=expires.strftime("%a, %d %b %Y %H:%M:%S GMT"),
        samesite="lax",
    )


def get_flash_message(request: Request) -> dict | None:
    cookie = request.cookies.get("flash_msg")
    if cookie:
        try:
            return json.loads(cookie)
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def clear_flash_message(response: Response):
    response.delete_cookie("flash_msg")


# ─────────────────────────────────────────────────────────────────────────────
# Async DB helper
# Blocking psycopg2 calls must NOT run on the event loop thread.
# run_in_executor offloads them to the thread pool.
# ─────────────────────────────────────────────────────────────────────────────
async def run_db(fn, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fn, *args)


# ─────────────────────────────────────────────────────────────────────────────
# Global pipeline state
# ─────────────────────────────────────────────────────────────────────────────
processors        = {}   # cam_id -> VisionPipeline (running pipelines only)
batched_det_proc  = None
frame_ready_queue = None
det_queues        = {}
free_slots_queues = {}
shm_names         = {}
response_queues   = {}
stop_events       = {}   # cam_id -> per-camera CTX.Event
CTX               = None
db_queue          = None
db_writer_proc    = None
global_stop_event = None  # signals batched_detector + final app shutdown only
cameras_dirty     = False  # True when DB camera list changed since last apply
_pipelines_booted = False  # set True once apply_changes has run since process start
analytics_queue       = None
analytics_writer_proc = None


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline management
# ─────────────────────────────────────────────────────────────────────────────
def _start_batched_detector() -> mp.Process:
    p = CTX.Process(
        target=batched_detector_worker,
        args=(
            frame_ready_queue,
            det_queues,
            free_slots_queues,
            shm_names,
            settings.FRAME_SHAPE,
            settings.FRAME_BYTES,
            global_stop_event,
        ),
        daemon=True,
    )
    p.start()
    logger.info(f"Batched detector started (pid={p.pid})")
    return p


def _stop_batched_detector():
    global batched_det_proc
    if batched_det_proc is not None and batched_det_proc.is_alive():
        batched_det_proc.terminate()
        batched_det_proc.join(timeout=3)
        logger.info("Batched detector stopped")
    batched_det_proc = None


def create_pipeline(stream_url: str, cam_id: int) -> VisionPipeline:
    free_slots     = CTX.Queue(maxsize=4)
    det_queue      = CTX.Queue(maxsize=4)
    response_queue = CTX.Queue()
    cam_stop_event = CTX.Event()   # per-camera — stopping one never affects others

    det_queues[cam_id]        = det_queue
    free_slots_queues[cam_id] = free_slots
    response_queues[cam_id]   = response_queue
    stop_events[cam_id]       = cam_stop_event

    processor = VisionPipeline(
        RTSP_URL=stream_url,
        CAM_ID=cam_id,
        ctx=CTX,
        free_slots=free_slots,
        det_queue=det_queue,
        stop_event=cam_stop_event,
        db_queue=db_queue,
        response_queue=response_queue,
        frame_ready_queue=frame_ready_queue,
        analytics_queue=analytics_queue,
    )
    shm_names[cam_id] = processor.input_shm_name
    return processor


# ─────────────────────────────────────────────────────────────────────────────
# Three-phase pipeline restart engine
#
# The fundamental rule: teardown and startup never overlap, and the
# event loop is never blocked while processes are being joined.
#
# Phase A — _disconnect_streams()
#   Runs on the event loop. Snapshots and clears processors{} instantly.
#   Every live generate() loop sees cam_id not in processors on its next
#   asyncio.sleep() tick (≤33 ms) and breaks, closing the StreamingResponse.
#   No process is touched. Returns snapshot for Phase B.
#
# Phase B — _teardown_sync(snapshot)
#   Runs in executor thread. Correct dependency order:
#     1. batched_detector terminated FIRST — immediately unblocks any reader
#        that is stuck trying to put() into a full frame_ready_queue.
#     2. Per-camera readers + embedders joined — they exit cleanly now because
#        the consumer of their output queue is gone.
#     3. db_writer shut down — safe now because all embedders have stopped.
#   Clears the remaining module-level dicts.
#
# Phase C — _startup_sync(user_cameras)
#   Runs in the same executor thread, immediately after B.
#   Rebuilds everything from scratch in correct dependency order:
#     1. Fresh frame_ready_queue — old one discarded (may have stale items).
#     2. create_pipeline() per camera — fresh SHM, queues, per-camera event.
#     3. db_writer started with fully populated response_queues.
#     4. batched_detector started with fully populated shm_names + det_queues.
#     5. p_reader + p_embedder started last — begin producing frames.
#
# apply_changes(user_cameras)
#   Async orchestrator. Runs A on event loop, then B+C together in one thread.
#   Called by: login POST (auto-start on login), /monitor (auto-restart when cameras_dirty).
# ─────────────────────────────────────────────────────────────────────────────

async def _disconnect_streams() -> dict:
    """Phase A — instant, on the event loop. Clears processors{}."""
    snapshot = dict(processors)
    processors.clear()
    return snapshot


def _teardown_sync(snapshot: dict) -> None:
    """Phase B — blocking, in executor thread. Correct dependency order."""
    global db_writer_proc, batched_det_proc

    # 1. Kill batched_detector first — unblocks readers stuck on full queue
    _stop_batched_detector()

    # 2. Stop per-camera readers + embedders
    for cam_id, proc in snapshot.items():
        try:
            proc.stop()
        except Exception as e:
            logger.error(f"Teardown error cam {cam_id}: {e}")

    # 3. Shut down db_writer — safe now that all embedders are dead
    if db_writer_proc is not None:
        try:
            db_queue.put(None, timeout=2)
            db_writer_proc.join(timeout=5)
        except Exception as e:
            logger.error(f"db_writer shutdown error: {e}")
            if db_writer_proc.is_alive():
                db_writer_proc.terminate()
        db_writer_proc = None

    # Clear remaining dicts (processors already cleared in Phase A)
    det_queues.clear()
    free_slots_queues.clear()
    shm_names.clear()
    response_queues.clear()
    stop_events.clear()
    logger.info("Teardown complete")


def _startup_sync(user_cameras: list) -> None:
    """Phase C — blocking, in executor thread. Rebuilds from scratch."""
    global frame_ready_queue, db_writer_proc, batched_det_proc

    if not user_cameras:
        logger.info("No cameras configured — startup skipped")
        return

    # 1. Fresh frame_ready_queue — guaranteed empty, no stale items
    frame_ready_queue = CTX.Queue(maxsize=64)

    # 2. create_pipeline() for every camera
    for cam in user_cameras:
        try:
            proc = create_pipeline(cam["stream_url"], cam["id"])
            processors[cam["id"]] = proc
        except Exception as e:
            logger.error(f"Failed to create pipeline cam {cam['id']}: {e}")

    if not processors:
        logger.error("No pipelines created — startup aborted")
        return

    # 3. db_writer with fully populated response_queues
    db_writer_proc = start_db_writer(CTX, db_queue, response_queues)

    # 4. ONE batched_detector — single YOLO session for all cameras
    batched_det_proc = _start_batched_detector()

    # 5. Start all readers + embedders simultaneously
    for proc in processors.values():
        try:
            proc.start()
        except Exception as e:
            logger.error(f"Failed to start pipeline: {e}")

    logger.info(
        f"Startup complete — {len(processors)} camera(s) live, "
        f"1 YOLO session shared across all"
    )


async def apply_changes(user_cameras: list) -> None:
    """
    Orchestrates the full three-phase restart.

    Phase A runs on the event loop (instant) — HTTP response is never delayed.
    Phases B+C run together in one executor thread — blocking joins are hidden
    from the event loop so streaming connections can close naturally.

    user_cameras: list of {id, stream_url} dicts fetched from DB by the caller.
    """
    snapshot = await _disconnect_streams()        # Phase A — instant

    # Give generate() loops one tick to observe the empty processors dict
    await asyncio.sleep(0.05)

    def _teardown_then_startup():                 # Phases B+C — sequential, one thread
        _teardown_sync(snapshot)
        _startup_sync(user_cameras)

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _teardown_then_startup)


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — startup / shutdown
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global CTX, db_queue, db_writer_proc, frame_ready_queue, global_stop_event, batched_det_proc

    CTX               = mp.get_context("spawn")
    global_stop_event = CTX.Event()
    db_queue          = CTX.Queue(maxsize=1024)
    frame_ready_queue = CTX.Queue(maxsize=64)
    analytics_queue   = CTX.Queue(maxsize=2048)

    init_db()

    with get_connection() as conn:
        load_cache(conn)

    # db_writer starts at app boot — owns the embedding cache.
    # Pipelines start on first login via apply_changes().
    db_writer_proc = start_db_writer(CTX, db_queue, response_queues)

    # Analytics writer — batches detection events, runs retention cleanup
    from src.core.analytics_writer import start_analytics_writer
    analytics_writer_proc = start_analytics_writer(CTX, analytics_queue)

    logger.info("App ready — pipelines will start automatically on login")

    yield

    # ── Shutdown — same dependency order as _teardown_sync ────────────────────
    logger.info("Shutting down...")
    global_stop_event.set()

    snapshot = dict(processors)
    processors.clear()

    _stop_batched_detector()                          # 1. detector first
    for cam_id, proc in snapshot.items():             # 2. per-camera workers
        try:
            proc.stop()
        except Exception as e:
            logger.error(f"Shutdown error cam {cam_id}: {e}")
    if db_writer_proc is not None:                    # 3. db_writer last
        try:
            db_queue.put(None, timeout=2)
            db_writer_proc.join(timeout=5)
        except Exception:
            pass
    if analytics_writer_proc is not None:             # 4. analytics writer
        try:
            analytics_queue.put("SHUTDOWN", timeout=2)
            analytics_writer_proc.join(timeout=5)
        except Exception:
            pass

    logger.info("Shutdown complete")


# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────
setup_logging()
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app       = FastAPI(lifespan=lifespan, title="ShopVision API", version="2.0.0")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "web", "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "web", "static")), name="static")

# CORS — restrict origins in production via CORS_ORIGINS env var
_cors_origins = os.environ.get("CORS_ORIGINS", "").split(",")
_cors_origins = [o.strip() for o in _cors_origins if o.strip()]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "cameras": len(processors)}

# Load offline image once — served by generate() until has_frame.value is set.
_offline_path = os.path.join(BASE_DIR, "web", "static", "camera_offline.jpg")
with open(_offline_path, "rb") as _f:
    _OFFLINE_JPEG: bytes = _f.read()
logger.info(f"Offline image loaded ({len(_OFFLINE_JPEG)} bytes): {_offline_path}")


@app.middleware("http")
async def auto_resume_on_restart(request: Request, call_next):
    """
    If the app process was restarted (e.g. uvicorn restart, crash recovery,
    deploy) while the browser still holds a valid session cookie, the user
    should NOT have to logout/login again to get monitoring running.

    This runs once: the first incoming request after boot that carries a
    valid session triggers apply_changes() in the background, exactly like
    do_login() does. Every later request is a no-op pass-through.

    The response is returned IMMEDIATELY — the DB fetch and pipeline startup
    happen in a background task so the user never sees a frozen page.
    """
    global _pipelines_booted

    if not _pipelines_booted:
        uid = get_current_user(request)
        if uid is not None:
            _pipelines_booted = True  # set before spawn — never trigger twice

            async def _background_resume(user_id: int):
                try:
                    def _fetch_cams():
                        with get_connection() as conn:
                            with conn.cursor() as cur:
                                cur.execute(
                                    "SELECT id, stream_url FROM cameras WHERE user_id = %s",
                                    (user_id,),
                                )
                                return [{"id": r["id"], "stream_url": r["stream_url"]}
                                        for r in cur.fetchall()]

                    user_cameras = await run_db(_fetch_cams)
                    await apply_changes(user_cameras)
                    logger.info(
                        f"Resumed session (uid={user_id}) — {len(user_cameras)} camera(s) live"
                    )
                except Exception as e:
                    logger.error(f"Auto-resume failed: {e}")

            asyncio.create_task(_background_resume(uid))

    return await call_next(request)


# ─────────────────────────────────────────────────────────────────────────────
# Auth endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    flash    = get_flash_message(request)
    response = templates.TemplateResponse("home.html", {
        "request": request,
        "logged_in": get_current_user(request) is not None,
        "flash_message": flash,
    })
    clear_flash_message(response)
    return response


@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    flash    = get_flash_message(request)
    response = templates.TemplateResponse("about.html", {
        "request": request,
        "logged_in": get_current_user(request) is not None,
        "flash_message": flash,
    })
    clear_flash_message(response)
    return response


@app.get("/register_user", response_class=HTMLResponse)
async def register_page(request: Request):
    flash    = get_flash_message(request)
    response = templates.TemplateResponse("register_user.html", {
        "request": request,
        "logged_in": get_current_user(request) is not None,
        "flash_message": flash,
    })
    clear_flash_message(response)
    return response


@app.post("/register_user")
async def register_user(
    username: str = Form(...),
    password: str = Form(...),
):
    hashed = pwd_ctx.hash(password)

    def _insert():
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
                    (username, hashed),
                )
            conn.commit()

    try:
        await run_db(_insert)
        resp = RedirectResponse(url="/login", status_code=302)
        flash_message(resp, f"{username} registered successfully", "success")
        logger.info(f"User registered: {username}")
        return resp
    except psycopg2.errors.UniqueViolation:
        resp = RedirectResponse(url="/register_user", status_code=302)
        flash_message(resp, f"Username '{username}' is already taken", "danger")
        return resp


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    flash    = get_flash_message(request)
    response = templates.TemplateResponse("login.html", {
        "request": request,
        "logged_in": get_current_user(request) is not None,
        "flash_message": flash,
    })
    clear_flash_message(response)
    return response


@app.post("/login")
async def do_login(
    username: str = Form(...),
    password: str = Form(...),
):
    def _fetch():
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, password_hash FROM users WHERE username = %s",
                    (username,),
                )
                return cur.fetchone()

    row   = await run_db(_fetch)
    valid = await run_db(lambda: row is not None and pwd_ctx.verify(password, row["password_hash"]))

    if valid:
        global _pipelines_booted
        uid   = row["id"]
        token = _sign_session(uid)

        # Fetch this user's cameras then fire apply_changes as a background
        # task so monitoring starts immediately without blocking the response.
        def _fetch_cams():
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id, stream_url FROM cameras WHERE user_id = %s",
                        (uid,),
                    )
                    return [{"id": r["id"], "stream_url": r["stream_url"]}
                            for r in cur.fetchall()]

        user_cameras = await run_db(_fetch_cams)
        _pipelines_booted = True  # prevent the auto-resume middleware from re-triggering
        asyncio.create_task(apply_changes(user_cameras))
        logger.info(
            f"Login: {username} — pipeline startup triggered "
            f"for {len(user_cameras)} camera(s)"
        )

        resp = RedirectResponse(url="/", status_code=302)
        resp.set_cookie(
            SESSION_COOKIE, token,
            max_age=SESSION_MAX_AGE,
            httponly=True,
            samesite="lax",
            secure=os.environ.get("SECURE_COOKIES", "false").lower() == "true",
        )
        flash_message(resp, f"Welcome {username}", "success")
        return resp

    resp = RedirectResponse(url="/login", status_code=302)
    flash_message(resp, "Invalid credentials", "danger")
    return resp


@app.get("/logout")
async def logout():
    global _pipelines_booted
    _pipelines_booted = False
    asyncio.create_task(apply_changes([]))  # empty list = teardown only, no startup
    logger.info("Logout — pipeline teardown triggered")

    resp = RedirectResponse(url="/", status_code=302)
    resp.delete_cookie(SESSION_COOKIE)
    flash_message(resp, "Logged out", "success")
    return resp


# ─────────────────────────────────────────────────────────────────────────────
# Camera management
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/register", response_class=HTMLResponse)
async def camera_page(request: Request):
    uid = get_current_user(request)
    if uid is None:
        resp = RedirectResponse(url="/", status_code=302)
        flash_message(resp, "You need to log in first.", "danger")
        return resp

    def _fetch():
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, name, stream_url FROM cameras WHERE user_id = %s",
                    (uid,),
                )
                return cur.fetchall()

    rows    = await run_db(_fetch)
    cameras = [{"id": r["id"], "name": r["name"], "url": r["stream_url"]} for r in rows]
    flash   = get_flash_message(request)
    resp    = templates.TemplateResponse("register.html", {
        "request": request,
        "logged_in": True,
        "cameras": cameras,
        "flash_message": flash,
    })
    clear_flash_message(resp)
    return resp


@app.post("/register")
async def register_camera(
    request: Request,
    name: str = Form(...),
    url:  str = Form(...),
):
    uid = get_current_user(request)
    if uid is None:
        resp = RedirectResponse(url="/", status_code=302)
        flash_message(resp, "You need to log in first.", "danger")
        return resp

    if not name or not url:
        resp = RedirectResponse(url="/register", status_code=302)
        flash_message(resp, "Camera name and URL are required.", "danger")
        return resp

    def _insert():
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO cameras (name, stream_url, user_id) VALUES (%s, %s, %s) RETURNING id",
                    (name.strip(), url.strip(), uid),
                )
                cam_id = cur.fetchone()["id"]
            conn.commit()
            return cam_id

    cam_id = await run_db(_insert)
    global cameras_dirty
    cameras_dirty = True
    logger.info(f"Camera {cam_id} '{name.strip()}' added — changes staged")

    resp = RedirectResponse(url="/register", status_code=302)
    flash_message(resp, f"Camera '{name.strip()}' added. Go to Monitor to activate.", "success")
    return resp


@app.post("/delete-camera/{cam_id}")
async def delete_camera(cam_id: int, request: Request):
    uid = get_current_user(request)
    if uid is None:
        resp = RedirectResponse(url="/", status_code=302)
        flash_message(resp, "You need to log in first.", "danger")
        return resp

    def _delete():
        # write_connection() commits on success, rolls back on exception.
        # The old code used get_connection() which never commits in PostgreSQL —
        # that is why the deleted camera always reappeared after restart.
        with write_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM cameras WHERE id = %s AND user_id = %s",
                    (cam_id, uid),
                )
                if not cur.fetchone():
                    return False
                cur.execute("DELETE FROM detections WHERE camera_id = %s", (cam_id,))
                cur.execute(
                    "DELETE FROM cameras WHERE id = %s AND user_id = %s",
                    (cam_id, uid),
                )
        return True

    try:
        deleted = await run_db(_delete)
    except Exception as e:
        logger.error(f"Camera deletion failed: {e}")
        resp = RedirectResponse(url="/register", status_code=302)
        flash_message(resp, f"Deletion failed: {e}", "danger")
        return resp

    if not deleted:
        resp = RedirectResponse(url="/register", status_code=302)
        flash_message(resp, "Camera not found or access denied.", "danger")
        return resp

    global cameras_dirty
    cameras_dirty = True
    logger.info(f"Camera {cam_id} deleted — changes staged")

    resp = RedirectResponse(url="/register", status_code=302)
    flash_message(resp, f"Camera {cam_id} deleted. Go to Monitor to activate.", "success")
    return resp


# ─────────────────────────────────────────────────────────────────────────────
# Video feed
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/video_feed/{cam_id}")
async def video_feed(cam_id: int, request: Request):
    uid = get_current_user(request)
    if uid is None:
        return RedirectResponse(url="/", status_code=302)

    async def generate():
        last_buf:     bytes | None = None
        interval = 1.0 / 30
        processor = None

        try:
            while True:
                # Re-fetch processor each iteration so we pick up pipelines
                # that start after the initial request (background resume).
                processor = processors.get(cam_id)

                if processor is None:
                    # Pipeline not started yet — serve offline image so the
                    # <img> tag gets a valid MJPEG frame instead of a 404.
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + _OFFLINE_JPEG
                        + b"\r\n"
                    )
                    await asyncio.sleep(1.0)
                    continue

                # has_frame.value is set to 1 by reader_worker the first time it
                # successfully reads a frame from the camera. Until then, serve
                # the offline image so the tile is never black or empty.
                # After the camera connects, switch seamlessly to the live stream.
                if not processor.has_frame.value:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + _OFFLINE_JPEG
                        + b"\r\n"
                    )
                    await asyncio.sleep(1.0)   # poll once per second while offline
                    continue

                frame = processor.get_latest_frame()
                ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_bytes = buf.tobytes()
                    if frame_bytes != last_buf:
                        last_buf = frame_bytes
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n"
                            + frame_bytes
                            + b"\r\n"
                        )

                await asyncio.sleep(interval)
        except Exception as e:
            logger.error(f"Video feed error cam {cam_id}: {e}")

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/monitor", response_class=HTMLResponse)
async def monitor(request: Request):
    uid = get_current_user(request)
    if uid is None:
        resp = RedirectResponse(url="/", status_code=302)
        flash_message(resp, "You need to log in first.", "danger")
        return resp

    global cameras_dirty

    def _fetch():
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, name, stream_url FROM cameras WHERE user_id = %s",
                    (uid,),
                )
                return cur.fetchall()

    rows = await run_db(_fetch)

    if cameras_dirty:
        # Camera list changed since last apply — restart pipelines automatically.
        # Fire as background task so the page renders immediately with the
        # spinner; pipelines come up while the user is already looking at monitor.
        cameras_dirty    = False   # clear before the task starts — not after
        user_cameras     = [{"id": r["id"], "stream_url": r["stream_url"]} for r in rows]
        asyncio.create_task(apply_changes(user_cameras))
        logger.info(f"Monitor: cameras_dirty — auto-restart triggered "
                    f"for {len(user_cameras)} camera(s)")
        pipelines_running = False   # show spinner while they boot
    else:
        pipelines_running = len(processors) > 0

    cam_list = [{"id": r["id"], "name": r["name"], "path": r["stream_url"]} for r in rows]

    return templates.TemplateResponse("monitor.html", {
        "request": request,
        "cameras": cam_list,
        "logged_in": True,
        "pipelines_running": pipelines_running,
    }, headers={"Cache-Control": "no-store"})


@app.get("/analysis", response_class=HTMLResponse)
async def analysis(request: Request):
    flash = get_flash_message(request)
    resp  = templates.TemplateResponse("analysis.html", {
        "request": request,
        "logged_in": get_current_user(request) is not None,
        "flash_message": flash,
    })
    clear_flash_message(resp)
    return resp


# ─────────────────────────────────────────────────────────────────────────────
# JSON API
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/state")
async def api_state(request: Request):
    uid = get_current_user(request)
    if uid is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    def _fetch():
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, name, stream_url FROM cameras WHERE user_id = %s",
                    (uid,),
                )
                cameras = cur.fetchall()

                cur.execute("""
                    SELECT id, first_seen, last_seen
                    FROM customers
                    ORDER BY last_seen DESC
                    LIMIT 50
                """)
                customers = cur.fetchall()

                cur.execute("""
                    SELECT d.id, d.camera_id, d.timestamp, d.bbox,
                           e.customer_id
                    FROM detections d
                    LEFT JOIN embeddings e ON e.id = d.embedding_id
                    WHERE d.camera_id IN (
                        SELECT id FROM cameras WHERE user_id = %s
                    )
                    ORDER BY d.timestamp DESC
                    LIMIT 100
                """, (uid,))
                detections = cur.fetchall()

            return cameras, customers, detections

    cameras_rows, customer_rows, detection_rows = await run_db(_fetch)

    cam_list = []
    for r in cameras_rows:
        proc   = processors.get(r["id"])
        alive = proc is not None and getattr(getattr(proc, "p_reader", None), "is_alive", lambda: False)()
        status = "online" if alive else "offline"
        cam_list.append({
            "id": r["id"], "name": r["name"], "streamUrl": r["stream_url"], "status": status,
        })

    return {
        "cameras": cam_list,
        "customers": [{"id": r["id"], "firstSeen": r["first_seen"], "lastSeen": r["last_seen"]}
                      for r in customer_rows],
        "detections": [{
            "id": r["id"], "cameraId": r["camera_id"], "timestamp": r["timestamp"],
            "bbox": json.loads(r["bbox"]) if r["bbox"] else None,
            "customerId": r["customer_id"],
        } for r in detection_rows],
        "alerts": [],
    }


@app.get("/api/cameras/{cam_id}/status")
async def camera_status(cam_id: int, request: Request):
    uid = get_current_user(request)
    if uid is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    proc = processors.get(cam_id)
    alive = proc is not None and getattr(getattr(proc, "p_reader", None), "is_alive", lambda: False)()
    return {
        "cam_id":  cam_id,
        "running": alive,
        "pid":     proc.p_reader.pid if proc and hasattr(proc, "p_reader") else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Analytics API
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/analysis/overview")
async def analysis_overview(request: Request):
    uid = get_current_user(request)
    if uid is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    def _fetch():
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS cnt FROM customers")
                total_customers = cur.fetchone()["cnt"]

                cur.execute("SELECT COUNT(*) AS cnt FROM detections")
                total_detections = cur.fetchone()["cnt"]

                # Active now = distinct tracker_ids seen in last 5 minutes
                import time
                cutoff = time.time() - 300
                cur.execute(
                    "SELECT COUNT(DISTINCT tracker_id) AS cnt FROM detection_events WHERE timestamp > %s",
                    (cutoff,),
                )
                active_now = cur.fetchone()["cnt"]

            return {
                "total_customers": total_customers,
                "total_detections": total_detections,
                "active_now": active_now,
            }

    return await run_db(_fetch)


@app.get("/api/analysis/footfall")
async def analysis_footfall(request: Request, interval: str = "hour", days: int = 7):
    uid = get_current_user(request)
    if uid is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    def _fetch():
        with get_connection() as conn:
            # Get first camera for the user
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM cameras WHERE user_id = %s LIMIT 1",
                    (uid,),
                )
                row = cur.fetchone()
                if not row:
                    return []
                cam_id = row["id"]

            from src.core.database import get_footfall
            return get_footfall(conn, cam_id, interval, days)

    data = await run_db(_fetch)
    return [
        {"bucket": r["hour_bucket"], "visitors": r["unique_visitors"], "detections": r["total_detections"]}
        for r in data
    ]


@app.get("/api/analysis/occupancy")
async def analysis_occupancy(request: Request, hours: int = 24):
    uid = get_current_user(request)
    if uid is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    def _fetch():
        import time
        cutoff = time.time() - hours * 3600
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM cameras WHERE user_id = %s LIMIT 1",
                    (uid,),
                )
                row = cur.fetchone()
                if not row:
                    return []
                cam_id = row["id"]

                cur.execute(
                    """SELECT hour_bucket, peak_occupancy
                       FROM analytics_hourly
                       WHERE camera_id = %s AND hour_bucket > %s
                       ORDER BY hour_bucket""",
                    (cam_id, cutoff),
                )
                return [dict(r) for r in cur.fetchall()]

    data = await run_db(_fetch)
    return [{"time": r["hour_bucket"], "occupancy": r["peak_occupancy"]} for r in data]


@app.get("/api/analysis/dwell")
async def analysis_dwell(request: Request, zone_id: int = None):
    uid = get_current_user(request)
    if uid is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    def _fetch():
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM cameras WHERE user_id = %s LIMIT 1",
                    (uid,),
                )
                row = cur.fetchone()
                if not row:
                    return {}
                cam_id = row["id"]

                if zone_id:
                    cur.execute(
                        """SELECT dwell_seconds FROM zone_events
                           WHERE zone_id = %s AND event_type = 'exit' AND dwell_seconds > 0""",
                        (zone_id,),
                    )
                else:
                    cur.execute(
                        """SELECT ze.dwell_seconds FROM zone_events ze
                           JOIN zones z ON z.id = ze.zone_id
                           WHERE z.camera_id = %s AND ze.event_type = 'exit' AND ze.dwell_seconds > 0""",
                        (cam_id,),
                    )
                rows = cur.fetchall()

        # Build histogram buckets
        if not rows:
            return {}
        buckets = {"0-10s": 0, "10-30s": 0, "30-60s": 0, "1-3min": 0, "3-10min": 0, "10min+": 0}
        for r in rows:
            d = r["dwell_seconds"]
            if d < 10: buckets["0-10s"] += 1
            elif d < 30: buckets["10-30s"] += 1
            elif d < 60: buckets["30-60s"] += 1
            elif d < 180: buckets["1-3min"] += 1
            elif d < 600: buckets["3-10min"] += 1
            else: buckets["10min+"] += 1
        return buckets

    return await run_db(_fetch)


@app.get("/api/analysis/peak-hours")
async def analysis_peak_hours(request: Request):
    uid = get_current_user(request)
    if uid is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    def _fetch():
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM cameras WHERE user_id = %s LIMIT 1",
                    (uid,),
                )
                row = cur.fetchone()
                if not row:
                    return []
                cam_id = row["id"]

            from src.core.database import get_peak_hours
            return get_peak_hours(conn, cam_id)

    data = await run_db(_fetch)
    return [{"dow": int(r["dow"]), "hour": int(r["hour"]), "total": r["total"]} for r in data]


@app.get("/api/analysis/repeat-visitors")
async def analysis_repeat_visitors(request: Request, days: int = 30):
    uid = get_current_user(request)
    if uid is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    def _fetch():
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM cameras WHERE user_id = %s LIMIT 1",
                    (uid,),
                )
                row = cur.fetchone()
                if not row:
                    return {"new_visitors": 0, "return_visitors": 0}
                cam_id = row["id"]

            from src.core.database import get_repeat_visitors
            return get_repeat_visitors(conn, cam_id, days)

    return await run_db(_fetch)


@app.get("/api/analysis/zone-stats")
async def analysis_zone_stats(request: Request, cam_id: int = None):
    uid = get_current_user(request)
    if uid is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    def _fetch():
        with get_connection() as conn:
            if cam_id is None:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id FROM cameras WHERE user_id = %s LIMIT 1",
                        (uid,),
                    )
                    row = cur.fetchone()
                    if not row:
                        return []
                    target_cam = row["id"]
            else:
                target_cam = cam_id

            from src.core.database import get_zone_stats
            return get_zone_stats(conn, target_cam)

    return await run_db(_fetch)


# ─────────────────────────────────────────────────────────────────────────────
# Zone management API
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/zones/{cam_id}")
async def list_zones(cam_id: int, request: Request):
    uid = get_current_user(request)
    if uid is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    def _fetch():
        from src.core.database import get_zones_for_camera
        with get_connection() as conn:
            return get_zones_for_camera(conn, cam_id)

    zones = await run_db(_fetch)
    return {"zones": zones}


@app.post("/api/zones/{cam_id}")
async def create_zone_endpoint(cam_id: int, request: Request):
    uid = get_current_user(request)
    if uid is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    body = await request.json()
    name = body.get("name", "Zone")
    polygon = body.get("polygon")  # [[x1,y1],[x2,y2],...]
    zone_type = body.get("zone_type", "area")
    color = body.get("color", "#4f8cff")

    if not polygon or len(polygon) < 3:
        raise HTTPException(status_code=400, detail="Polygon must have at least 3 points")

    def _insert():
        from src.core.database import create_zone
        with write_connection() as conn:
            return create_zone(conn, cam_id, name, polygon, zone_type, color)

    zone_id = await run_db(_insert)
    return {"id": zone_id, "name": name, "zone_type": zone_type}


@app.delete("/api/zones/{zone_id}")
async def delete_zone_endpoint(zone_id: int, request: Request):
    uid = get_current_user(request)
    if uid is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    def _delete():
        from src.core.database import delete_zone
        with write_connection() as conn:
            return delete_zone(conn, zone_id)

    deleted = await run_db(_delete)
    if not deleted:
        raise HTTPException(status_code=404, detail="Zone not found")
    return {"deleted": True}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,      # MUST be 1 — multiprocessing state is not fork-safe
        reload=False,   # MUST be False — reloader kills child processes
    )