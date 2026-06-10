import os
import cv2
import time
import json
import asyncio
import sqlite3
import uvicorn
from loguru import logger
import multiprocessing as mp
from functools import partial
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from passlib.context import CryptContext
from src.core.config import settings
from starlette.requests import Request
from src.core.logging import setup_logging
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from src.core.db_writer import start_db_writer
from fastapi import FastAPI, HTTPException, Request, Form, Response
from src.core.database import init_db, load_cache, read_connection, write_connection
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from src.engine.pipeline import VisionPipeline, batched_detector_worker


# ```
# ┌─────────────────────────────────┐
# │  inference engine (your code)   │  Pure Python, no web framework
# │  pipeline.py, db_writer.py      │  Writes output to shared state
# └──────────────┬──────────────────┘
#                │ reads
# ┌──────────────▼──────────────────┐
# │  FastAPI — JSON API only        │  Thin, async, no templates
# │  /api/cameras  /api/state       │  Serves React/Vue SPA
# │  /api/stream/{cam_id}           │
# └──────────────┬──────────────────┘
#                │ served to
# ┌──────────────▼──────────────────┐
# │  Frontend — React or plain HTML │  Static files, CDN or nginx
# │  Fetches /api/state every 2s    │  WebSocket for live alerts
# └─────────────────────────────────┘
# ```

# On an edge device, run nginx in front. Nginx serves static files at zero CPU cost, proxies `/api/*` to FastAPI, and handles MJPEG streams efficiently.


# New code uses `itsdangerous.URLSafeTimedSerializer` with your `settings.SECRET_KEY`. The session cookie is **HMAC-signed** — if anyone tampers with it, `_verify_session` raises `BadSignature` and returns `None`. Tokens also expire after 8 hours via `SignatureExpired`. The cookie is `httponly=True` so JavaScript can't read it at all.

# You need to add `SECRET_KEY` to your config. Generate it once: `python -c "import secrets; print(secrets.token_hex(32))"` and put it in your `.env`.

# `werkzeug` is gone. `passlib` with bcrypt replaces `generate_password_hash`/`check_password_hash`. Same security, no Flask dependency.

# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------
pwd_ctx    = CryptContext(schemes=["bcrypt"], deprecated="auto")
_signer    = URLSafeTimedSerializer(settings.SECRET_KEY)   # add SECRET_KEY to config
SESSION_COOKIE = "session"
SESSION_MAX_AGE = 60 * 60 * 8   # 8 hours


def _sign_session(user_id: int) -> str:
    """Create a tamper-proof signed session token."""
    return _signer.dumps({"uid": user_id})


def _verify_session(token: str) -> int | None:
    """
    Verify and decode the session token.
    Returns user_id or None if invalid/expired.
    Tokens expire after SESSION_MAX_AGE seconds.
    """
    try:
        data = _signer.loads(token, max_age=SESSION_MAX_AGE)
        return int(data["uid"])
    except (BadSignature, SignatureExpired, KeyError):
        return None


def get_current_user(request: Request) -> int | None:
    """Return user_id from signed cookie, or None if not authenticated."""
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    return _verify_session(token)


def require_login(request: Request) -> int:
    """
    Dependency for endpoints that need auth.
    Raises HTTPException(401) if not logged in.
    Use as: user_id = require_login(request)
    For redirect-based pages, call get_current_user() and redirect manually.
    """
    uid = get_current_user(request)
    if uid is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return uid


# ---------------------------------------------------------------------------
# Flash message helpers (unchanged logic, kept as-is)
# ---------------------------------------------------------------------------
def flash_message(response: Response, message: str, category: str = "info"):
    import datetime
    expires = datetime.datetime.now() + datetime.timedelta(seconds=20)
    payload = json.dumps({"message": message, "category": category})
    response.set_cookie(
        key="flash_msg",
        value=payload,
        max_age=10,
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


# ---------------------------------------------------------------------------
# Async DB helper
# Blocking SQLite calls must NOT run on the event loop thread.
# run_in_executor offloads them to a thread pool.
# ---------------------------------------------------------------------------
async def run_db(fn, *args):
    """
    Run a blocking DB function in a thread pool so FastAPI's event loop
    is not blocked.

    Usage:
        rows = await run_db(lambda: conn.execute(...).fetchall())

    For write operations:
        await run_db(lambda: _do_write())
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fn)

# ---------------------------------------------------------------------------
# Global pipeline state
# ---------------------------------------------------------------------------
processors        = {}
batched_det_proc  = None
frame_ready_queue = None
det_queues        = {}
free_slots_queues = {}
shm_names         = {}
response_queues   = {}
CTX               = None
db_queue          = None
db_writer_proc    = None
stop_event        = None

# ---------------------------------------------------------------------------
# Pipeline management
# ---------------------------------------------------------------------------
def _start_batched_detector() -> mp.Process:
    """
    Start the shared YOLO detector process.
    MUST be called only after all cameras are registered in
    det_queues, free_slots_queues, and shm_names.
    """
    p = CTX.Process(
        target=batched_detector_worker,
        args=(
            frame_ready_queue,
            det_queues,
            free_slots_queues,
            shm_names,
            settings.FRAME_SHAPE,
            settings.FRAME_BYTES,
            stop_event,
        ),
        daemon=True,
    )
    p.start()
    logger.info(f"Batched detector started (pid={p.pid})")
    return p


def _stop_batched_detector():
    """Signal and join the current batched detector process."""
    global batched_det_proc
    if batched_det_proc is not None and batched_det_proc.is_alive():
        batched_det_proc.terminate()
        batched_det_proc.join(timeout=3)
        logger.info("Batched detector stopped")
    batched_det_proc = None


# def create_pipeline(stream_url: str, cam_id: int) -> VisionPipeline:
#     """
#     Build per-camera queues, register them globally, create VisionPipeline.
#     Does NOT start worker processes — call proc.start() separately.

#     Order matters:
#         1. Create queues
#         2. Register in global dicts   ← batched detector reads these at spawn time
#         3. Create VisionPipeline      ← allocates shared memory
#         4. Register shm_name          ← available after step 3
#     """
#     free_slots     = CTX.Queue(maxsize=4)
#     det_queue      = CTX.Queue(maxsize=4)
#     response_queue = CTX.Queue()

#     det_queues[cam_id]        = det_queue
#     free_slots_queues[cam_id] = free_slots

#     processor = VisionPipeline(
#         RTSP_URL=stream_url,
#         CAM_ID=cam_id,
#         ctx=CTX,
#         free_slots=free_slots,
#         det_queue=det_queue,
#         stop_event=stop_event,
#         db_queue=db_queue,
#         response_queue=response_queue,
#         frame_ready_queue=frame_ready_queue,
#     )
#     shm_names[cam_id] = processor.input_shm_name
#     return processor


# Add this global alongside the others
response_queues: dict[int, mp.Queue] = {}

def create_pipeline(stream_url: str, cam_id: int) -> VisionPipeline:
    free_slots     = CTX.Queue(maxsize=4)
    det_queue      = CTX.Queue(maxsize=4)
    response_queue = CTX.Queue()

    # Register ALL global dicts before anything starts
    det_queues[cam_id]        = det_queue
    free_slots_queues[cam_id] = free_slots
    response_queues[cam_id]   = response_queue   # ← register here

    processor = VisionPipeline(
        RTSP_URL=stream_url,
        CAM_ID=cam_id,
        ctx=CTX,
        free_slots=free_slots,
        det_queue=det_queue,
        stop_event=stop_event,
        db_queue=db_queue,
        response_queue=response_queue,
        frame_ready_queue=frame_ready_queue,
    )
    shm_names[cam_id] = processor.input_shm_name
    return processor

# def _restart_batched_detector()():
#     """
#     Option A: stop existing detector, start fresh with current dict state.
#     Call after adding or removing a camera from det_queues/shm_names.

#     Why re-create: mp.Process with spawn context pickles args at creation
#     time. Dict mutations after spawn are invisible to the subprocess.
#     Restarting gives it a fresh pickle of the updated dicts.
#     """
#     _stop_batched_detector()
#     global batched_det_proc
#     batched_det_proc = _start_batched_detector()

def _restart_all_workers():
    """
    Restart both the batched detector and db_writer with updated dicts.
    Call after adding or removing a camera.
    """
    global db_writer_proc, batched_det_proc

    # Stop both
    _stop_batched_detector()
    db_queue.put(None)           # poison pill stops db_writer cleanly
    db_writer_proc.join(timeout=5)

    # Restart with updated dicts
    db_writer_proc   = start_db_writer(CTX, db_queue, response_queues)
    batched_det_proc = _start_batched_detector()


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global CTX, db_queue, db_writer_proc, frame_ready_queue, stop_event, batched_det_proc

#     CTX               = mp.get_context("spawn")
#     stop_event        = CTX.Event()
#     db_queue          = CTX.Queue(maxsize=1024)
#     frame_ready_queue = CTX.Queue(maxsize=64)

#     db_writer_proc = start_db_writer(CTX, db_queue)

#     # DB init is blocking — do it before the event loop is critical
#     init_db()
#     with read_connection() as conn:
#         load_cache(conn)

#     # Load cameras from DB and register them all before starting detector
#     with read_connection() as conn:
#         rows = conn.execute("SELECT id, stream_url FROM cameras").fetchall()

#     for cam_id, stream_url in rows:
#         try:
#             proc = create_pipeline(stream_url, cam_id)
#             processors[cam_id] = proc
#             logger.info(f"Registered camera {cam_id}: {stream_url}")
#         except Exception as e:
#             logger.error(f"Failed to create pipeline for camera {cam_id}: {e}")

#     if processors:
#         batched_det_proc = _start_batched_detector()
#         for proc in processors.values():
#             proc.start()
#     else:
#         logger.warning("No cameras registered — batched detector not started")

#     yield

#     # ── Shutdown ──────────────────────────────────────────────────────────
#     logger.info("Shutting down...")
#     stop_event.set()

#     for proc in processors.values():
#         try:
#             proc.stop()
#         except Exception as e:
#             logger.error(f"Error stopping pipeline: {e}")

#     _stop_batched_detector()

#     db_queue.put(None)           # poison pill for db_writer
#     db_writer_proc.join(timeout=5)
#     logger.info("Shutdown complete")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global CTX, db_queue, db_writer_proc, frame_ready_queue, stop_event, batched_det_proc

    CTX               = mp.get_context("spawn")
    stop_event        = CTX.Event()
    db_queue          = CTX.Queue(maxsize=1024)
    frame_ready_queue = CTX.Queue(maxsize=64)

    # Load cameras from DB first — needed to build response_queues
    init_db()
    with read_connection() as conn:
        rows = conn.execute(
            "SELECT id, stream_url FROM cameras"
        ).fetchall()

    # Register all cameras BEFORE starting db_writer
    for cam_id, stream_url in rows:
        try:
            proc = create_pipeline(stream_url, cam_id)
            processors[cam_id] = proc
        except Exception as e:
            logger.error(f"Failed to create pipeline for camera {cam_id}: {e}")

    # Now response_queues is fully populated — safe to start db_writer
    db_writer_proc = start_db_writer(CTX, db_queue, response_queues)

    with read_connection() as conn:
        load_cache(conn)

    if processors:
        batched_det_proc = _start_batched_detector()
        for proc in processors.values():
            proc.start()

    yield

    stop_event.set()
    for proc in processors.values():
        try:
            proc.stop()
        except Exception as e:
            logger.error(f"Error stopping pipeline: {e}")
    _stop_batched_detector()
    db_queue.put(None)
    db_writer_proc.join(timeout=5)


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
setup_logging()
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app       = FastAPI(lifespan=lifespan, title="ShopVision API", version="2.0.0")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "web", "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "web", "static")), name="static")


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    flash = get_flash_message(request)
    response = templates.TemplateResponse("home.html", {
        "request":      request,
        "logged_in":    get_current_user(request) is not None,
        "flash_message": flash,
    })
    clear_flash_message(response)
    return response


@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    flash = get_flash_message(request)
    response = templates.TemplateResponse("about.html", {
        "request":      request,
        "logged_in":    get_current_user(request) is not None,
        "flash_message": flash,
    })
    clear_flash_message(response)
    return response


@app.get("/register_user", response_class=HTMLResponse)
async def register_page(request: Request):
    flash = get_flash_message(request)
    response = templates.TemplateResponse("register_user.html", {
        "request":      request,
        "logged_in":    get_current_user(request) is not None,
        "flash_message": flash,
    })
    clear_flash_message(response)
    return response


@app.post("/register_user")
async def register_user(
    username: str = Form(...),
    password: str = Form(...),
):
    hashed = pwd_ctx.hash(password)   # bcrypt, not werkzeug

    def _insert():
        with write_connection() as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, hashed),
            )

    try:
        await run_db(_insert)
        resp = RedirectResponse(url="/login", status_code=302)
        flash_message(resp, f"{username} registered successfully", "success")
        logger.info(f"User registered: {username}")
        return resp
    except sqlite3.IntegrityError:
        resp = RedirectResponse(url="/register_user", status_code=302)
        flash_message(resp, f"Username '{username}' is already taken", "danger")
        return resp


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    flash = get_flash_message(request)
    response = templates.TemplateResponse("login.html", {
        "request":      request,
        "logged_in":    get_current_user(request) is not None,
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
        with read_connection() as conn:
            return conn.execute(
                "SELECT id, password_hash FROM users WHERE username=?",
                (username,),
            ).fetchone()

    row = await run_db(_fetch)

    # pwd_ctx.verify is CPU-bound (bcrypt) — offload it too
    valid = await run_db(lambda: row is not None and pwd_ctx.verify(password, row[1]))

    if valid:
        token = _sign_session(row[0])
        resp  = RedirectResponse(url="/", status_code=302)
        resp.set_cookie(
            SESSION_COOKIE,
            token,
            max_age=SESSION_MAX_AGE,
            httponly=True,    # JS cannot read this cookie
            samesite="lax",
        )
        flash_message(resp, f"Welcome {username}", "success")
        logger.info(f"Login: {username}")
        return resp

    resp = RedirectResponse(url="/login", status_code=302)
    flash_message(resp, "Invalid credentials", "danger")
    return resp


@app.get("/logout")
async def logout():
    resp = RedirectResponse(url="/", status_code=302)
    resp.delete_cookie(SESSION_COOKIE)
    flash_message(resp, "Logged out", "success")
    return resp


# ---------------------------------------------------------------------------
# Camera management
# ---------------------------------------------------------------------------
@app.get("/register", response_class=HTMLResponse)
async def camera_page(request: Request):
    uid = get_current_user(request)
    if uid is None:
        resp = RedirectResponse(url="/", status_code=302)
        flash_message(resp, "You need to log in first.", "danger")
        return resp

    def _fetch():
        with read_connection() as conn:
            return conn.execute(
                "SELECT id, name, stream_url FROM cameras WHERE user_id=?",
                (uid,),
            ).fetchall()

    rows    = await run_db(_fetch)
    cameras = [{"id": r[0], "name": r[1], "url": r[2]} for r in rows]
    flash   = get_flash_message(request)
    resp    = templates.TemplateResponse("register.html", {
        "request":      request,
        "logged_in":    True,
        "cameras":      cameras,
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
        with write_connection() as conn:
            cur = conn.execute(
                "INSERT INTO cameras (name, stream_url, user_id) VALUES (?, ?, ?)",
                (name.strip(), url.strip(), uid),
            )
            return cur.lastrowid

    cam_id = await run_db(_insert)

    # Register pipeline + restart detector (Option A)
    # This runs in the main thread — it's fast (queue creation, shm alloc)
    try:
        proc = create_pipeline(url.strip(), cam_id)
        processors[cam_id] = proc
        _restart_all_workers()()   # ← fresh detector snapshot with new camera
        proc.start()
        logger.info(f"Camera {cam_id} '{name.strip()}' added and pipeline started")
    except Exception as e:
        logger.error(f"Pipeline start failed for camera {cam_id}: {e}")
        resp = RedirectResponse(url="/register", status_code=302)
        flash_message(resp, f"Camera registered but pipeline failed: {e}", "danger")
        return resp

    resp = RedirectResponse(url="/register", status_code=302)
    flash_message(resp, f"Camera '{name.strip()}' registered and started.", "success")
    return resp


@app.post("/delete-camera/{cam_id}")
async def delete_camera(cam_id: int, request: Request):
    uid = get_current_user(request)
    if uid is None:
        resp = RedirectResponse(url="/", status_code=302)
        flash_message(resp, "You need to log in first.", "danger")
        return resp

    # Stop the pipeline first
    proc = processors.pop(cam_id, None)
    if proc is not None:
        try:
            proc.stop()
        except Exception as e:
            logger.error(f"Error stopping pipeline for cam {cam_id}: {e}")

    # Remove from global tracking dicts
    det_queues.pop(cam_id, None)
    free_slots_queues.pop(cam_id, None)
    shm_names.pop(cam_id, None)

    # Restart detector without this camera
    if processors:
        # _restart_all_workers()()
        _restart_all_workers()
    else:
        _stop_batched_detector()
        logger.info("No cameras remaining — batched detector stopped")

    # Delete from DB
    def _delete():
        with write_connection() as conn:
            cam = conn.execute(
                "SELECT id FROM cameras WHERE id=? AND user_id=?",
                (cam_id, uid),
            ).fetchone()
            if not cam:
                return False
            conn.execute("DELETE FROM detections WHERE camera_id=?", (cam_id,))
            emb_ids = [
                r[0] for r in conn.execute(
                    "SELECT id FROM embedding_meta WHERE camera_id=?",
                    (cam_id,),
                ).fetchall()
            ]
            conn.execute("DELETE FROM embedding_meta WHERE camera_id=?", (cam_id,))
            if emb_ids:
                conn.execute(
                    f"DELETE FROM embeddings WHERE rowid IN ({','.join('?'*len(emb_ids))})",
                    emb_ids,
                )
            conn.execute("DELETE FROM cameras WHERE id=? AND user_id=?", (cam_id, uid))
            return True

    try:
        deleted = await run_db(_delete)
        if not deleted:
            resp = RedirectResponse(url="/register", status_code=302)
            flash_message(resp, "Camera not found or access denied.", "danger")
            return resp
        resp = RedirectResponse(url="/register", status_code=302)
        flash_message(resp, f"Camera {cam_id} deleted.", "success")
        return resp
    except Exception as e:
        resp = RedirectResponse(url="/register", status_code=302)
        flash_message(resp, f"Deletion failed: {e}", "danger")
        return resp


# ---------------------------------------------------------------------------
# Video feed
# ---------------------------------------------------------------------------
@app.get("/video_feed/{cam_id}")
async def video_feed(cam_id: int, request: Request):
    uid = get_current_user(request)
    if uid is None:
        return RedirectResponse(url="/", status_code=302)

    processor = processors.get(cam_id)

    if processor is None:
        # Camera registered in DB but not running — start it
        def _fetch():
            with read_connection() as conn:
                return conn.execute(
                    "SELECT stream_url FROM cameras WHERE id=? AND user_id=?",
                    (cam_id, uid),
                ).fetchone()

        row = await run_db(_fetch)
        if not row:
            raise HTTPException(status_code=404, detail="Camera not found")

        stream_url = row[0]

        # Only create if genuinely missing — race condition guard
        if cam_id not in processors:
            proc = create_pipeline(stream_url, cam_id)
            processors[cam_id] = proc
            _restart_all_workers()()   # ← Option A restart
            proc.start()
            logger.info(f"Late-start pipeline for camera {cam_id}")

        processor = processors[cam_id]

    async def generate():
        # Display resolution — separate from inference resolution
        # Change these to whatever your cameras' native resolution is
        # or add DISPLAY_WIDTH/DISPLAY_HEIGHT to settings
        display_w = settings.DISPLAY_WIDTH   # e.g. 1280
        display_h = settings.DISPLAY_HEIGHT  # e.g. 720

        while True:
            frame = processor.get_latest_frame()
            if frame is not None:
                # Upscale from inference resolution (512×512) to display resolution
                if (frame.shape[1], frame.shape[0]) != (display_w, display_h):
                    frame = cv2.resize(
                        frame,
                        (display_w, display_h),
                        interpolation=cv2.INTER_LINEAR
                    )
                ret, buf = cv2.imencode(
                    ".jpg", frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 75]
                )
                if ret:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + buf.tobytes()
                        + b"\r\n"
                    )
            await asyncio.sleep(0.033)

    #### OPTION 2: Full Res Image through the Pipeline
    # async def generate():
    #     while True:
    #         frame = processor.get_latest_frame()   # already native res
    #         if frame is not None:
    #             ret, buf = cv2.imencode(
    #                 ".jpg", frame,
    #                 [cv2.IMWRITE_JPEG_QUALITY, 75])
    #             if ret:
    #                 yield (
    #                     b"--frame\r\n"
    #                     b"Content-Type: image/jpeg\r\n\r\n"
    #                     + buf.tobytes()
    #                     + b"\r\n"
    #                 )
    #         await asyncio.sleep(0.033)


    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.get("/monitor", response_class=HTMLResponse)
async def monitor(request: Request):
    uid = get_current_user(request)
    if uid is None:
        resp = RedirectResponse(url="/", status_code=302)
        flash_message(resp, "You need to log in first.", "danger")
        return resp

    def _fetch():
        with read_connection() as conn:
            return conn.execute(
                "SELECT id, name, stream_url FROM cameras WHERE user_id=?",
                (uid,),
            ).fetchall()

    rows    = await run_db(_fetch)
    cam_list = [{"id": r[0], "name": r[1], "path": r[2]} for r in rows]
    return templates.TemplateResponse("monitor.html", {
        "request":   request,
        "cameras":   cam_list,
        "logged_in": True,
    })


@app.get("/analysis", response_class=HTMLResponse)
async def analysis(request: Request):
    flash = get_flash_message(request)
    resp  = templates.TemplateResponse("analysis.html", {
        "request":      request,
        "logged_in":    get_current_user(request) is not None,
        "flash_message": flash,
    })
    clear_flash_message(resp)
    return resp


# ---------------------------------------------------------------------------
# JSON API  (the commented schema you had — now real)
# ---------------------------------------------------------------------------
@app.get("/api/state")
async def api_state(request: Request):
    """
    Live system state for the frontend dashboard.
    Returns cameras, recent customers, detections, and alerts.
    """
    uid = get_current_user(request)
    if uid is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    def _fetch():
        with read_connection() as conn:
            cameras = conn.execute(
                "SELECT id, name, stream_url FROM cameras WHERE user_id=?",
                (uid,),
            ).fetchall()

            customers = conn.execute(
                """
                SELECT id, first_seen, last_seen
                FROM customers
                ORDER BY last_seen DESC
                LIMIT 50
                """,
            ).fetchall()

            detections = conn.execute(
                """
                SELECT d.id, d.camera_id, d.timestamp, d.bbox,
                       em.customer_id
                FROM detections d
                JOIN embedding_meta em ON em.id = d.embedding_meta_id
                WHERE d.camera_id IN (
                    SELECT id FROM cameras WHERE user_id=?
                )
                ORDER BY d.timestamp DESC
                LIMIT 100
                """,
                (uid,),
            ).fetchall()

        return cameras, customers, detections

    cameras_rows, customer_rows, detection_rows = await run_db(_fetch)

    # Enrich cameras with live pipeline status
    cam_list = []
    for r in cameras_rows:
        proc   = processors.get(r[0])
        status = "online" if (proc and proc.is_alive()) else "offline"
        cam_list.append({
            "id":        r[0],
            "name":      r[1],
            "streamUrl": r[2],
            "status":    status,
        })

    cust_list = [{
        "id":        r[0],
        "firstSeen": r[1],
        "lastSeen":  r[2],
    } for r in customer_rows]

    det_list = [{
        "id":         r[0],
        "cameraId":   r[1],
        "timestamp":  r[2],
        "bbox":       json.loads(r[3]) if r[3] else None,
        "customerId": r[4],
    } for r in detection_rows]

    return {
        "cameras":    cam_list,
        "customers":  cust_list,
        "detections": det_list,
        "alerts":     [],   # wire loitering alerts here when ready
    }


@app.get("/api/cameras/{cam_id}/status")
async def camera_status(cam_id: int, request: Request):
    """Quick health check for a single camera pipeline."""
    uid = get_current_user(request)
    if uid is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    proc = processors.get(cam_id)
    return {
        "cam_id":  cam_id,
        "running": proc is not None and proc.is_alive(),
        "pid":     proc.p_reader.pid if proc and hasattr(proc, "p_reader") else None,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,           # MUST be 1 — multiprocessing state is not fork-safe
        reload=False,        # MUST be False in production — reloader kills child procs
    )
