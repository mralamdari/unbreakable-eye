# # import os
# # import sqlite3
# # from fastapi import FastAPI, HTTPException, Body
# # from fastapi.staticfiles import StaticFiles
# # from fastapi.responses import HTMLResponse, FileResponse
# # from fastapi.middleware.cors import CORSMiddleware

# # app = FastAPI(title="Shop Vision API", version="1.0.0")

# # # 1. Enable CORS for local development testing
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # # 2. Match the exact endpoints your React Dashboard expects:

# # @app.get("/api/state")
# # def get_dashboard_state():
# #     """
# #     Combines SQL and active feed metrics into a unified JSON structure 
# #     that feeds the React dashboard cards and tables.
# #     """
# #     try:
# #         # Connect to your WAL SQLite-vec database
# #         conn = get_connection(readonly=True) # or read_connection context
        
# #         # Pull latest camera settings
# #         cameras = conn.execute(
# #             "SELECT id, name, stream_url, 'online' as status, 30 as fps FROM cameras"
# #         ).fetchall()
        
# #         # Pull registered Re-ID identities
# #         customers_query = """
# #             SELECT c.id, c.first_seen, c.last_seen, c.total_visits, 
# #                    COUNT(m.id) as embeddingCount,
# #                    'returning' as status 
# #             FROM customers c
# #             LEFT JOIN embedding_meta m ON c.id = m.customer_id
# #             GROUP BY c.id
# #         """
# #         raw_custs = conn.execute(customers_query).fetchall()
        
# #         # Pull latest detection logs to populate timeline matches
# #         detections_query = """
# #             SELECT d.id, d.camera_id, c.name as cameraName, d.timestamp, d.bbox,
# #                    d.embedding_meta_id as customerId, 'Main Zone' as currentZone
# #             FROM detections d
# #             JOIN cameras c ON d.camera_id = c.id
# #             ORDER BY d.timestamp DESC LIMIT 100
# #         """
# #         raw_dets = conn.execute(detections_query).fetchall()
# #         conn.close()

# #         # Format rows cleanly to match React schemas (camelCase keys)
# #         return {
# #             "cameras": [{"id": r[0], "name": r[1], "streamUrl": r[2], "status": r[3], "fps": r[4], "resolution": "1920x1080", "activeZoneCount": 1} for r in cameras],
# #             "customers": [{
# #                 "id": r[0], 
# #                 "firstSeen": r[1], 
# #                 "lastSeen": r[2], 
# #                 "totalVisits": r[3], 
# #                 "avgDwellTime": 420,  # calculated or static placeholder
# #                 "status": "loitering" if r[3] == 1 else "returning", # your matching status logic
# #                 "avatarSeed": f"cust_{r[0]}",
# #                 "embeddingCount": r[4],
# #                 "confidenceScore": 0.92
# #             } for r in raw_custs],
# #             "detections": [{
# #                 "id": str(r[0]),
# #                 "cameraId": r[1],
# #                 "cameraName": r[2],
# #                 "timestamp": r[3],
# #                 "bbox": [int(x) for x in r[4].split(",")] if r[4] else [0,0,0,0],
# #                 "customerId": r[5],
# #                 "dwellTime": 45,
# #                 "currentZone": r[6]
# #             } for r in raw_dets],
# #             "alerts": [
# #                 # Optional: Send live alert dictionaries
# #                 {"id": "alt_1", "timestamp": "2026-05-24T15:00:00Z", "level": "warning", "message": "High Cosmetics linger detected", "cameraName": "Pharmacy Camera", "resolved": False}
# #             ],
# #             "stats": {
# #                 "onlineCameras": len(cameras),
# #                 "totalCameras": len(cameras),
# #                 "totalDetectionsToday": len(raw_dets),
# #                 "avgDwellTimeSeconds": 480,
# #                 "loiteringActiveCount": 1,
# #                 "reidMatchCount": len(raw_custs),
# #                 "activeAlarmsCount": 1
# #             }
# #         }
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=str(e))

# # @app.post("/api/cameras")
# # def add_camera(payload: dict = Body(...)):
# #     """Inserts a new RTSP camera stream directly into SQLite."""
# #     # Run insert statement
# #     return {"status": "success"}

# # @app.delete("/api/cameras/{cam_id}")
# # def delete_camera(cam_id: int):
# #     """Deletes camera and dependent vector embeddings mapping details."""
# #     # Run SQL delete cascades
# #     return {"status": "deleted"}

# # # 3. Serve the Built React App directly from FastAPI static directory
# # # Place static files in a directory named `/dist` relative to main.py
# # app.mount("/assets", StaticFiles(directory="dist/assets"), name="assets")

# # @app.get("/{rest_of_path:path}")
# # def serve_spa(rest_of_path: str):
# #     """Fallback router to ensure React HTML5 pushState routing is preserved."""
# #     # Serve index.html globally so React's internal routes render correctly
# #     return FileResponse("dist/index.html")  

































# from pydantic import BaseModel
# from typing import Optional, List
# from google import genai

# # -- Pydantic Schemas for Dashboard Mapping --
# class CameraAdd(BaseModel):
#     name: str
#     streamUrl: str
#     activeZoneCount: int = 1

# class CustomerUpdate(BaseModel):
#     notes: Optional[str] = None
#     status: Optional[str] = None

# # 1. State Sync Endpoint (Dashboard loads metrics, camera statuses, and customer matches here)
# @app.get("/api/state")
# def get_state():
#     with read_connection() as conn:
#         # A. Fetch Camera lists
#         cameras_rows = conn.execute("SELECT id, name, stream_url FROM cameras").fetchall()
#         cameras = [
#             {
#                 "id": r["id"],
#                 "name": r["name"],
#                 "streamUrl": r["stream_url"],
#                 "status": "online" if r["id"] in processors else "offline",
#                 "fps": 30 if r["id"] in processors else 0,
#                 "resolution": "1920x1080",
#                 "activeZoneCount": 2,
#                 "totalDetections24h": conn.execute(
#                     "SELECT COUNT(*) FROM detections WHERE camera_id=?", (r["id"],)
#                 ).fetchone()[0]
#             } for r in cameras_rows
#         ]
        
#         # B. Fetch Unique Customer Profiles matched via OSNet
#         customer_rows = conn.execute(
#             "SELECT id, first_seen, last_seen, total_visits FROM customers"
#         ).fetchall()
        
#         customers_list = []
#         for r in customer_rows:
#             # Get count of stored embeddings for this customer
#             emb_count = conn.execute(
#                 "SELECT COUNT(*) FROM embedding_meta WHERE customer_id=?", (r["id"],)
#             ).fetchone()[0]
            
#             # Fetch latest custom notes/flags (You can add these columns to your SQLite table)
#             notes_row = conn.execute("SELECT notes, status FROM customer_meta WHERE customer_id=?", (r["id"],)).fetchone()
#             notes = notes_row["notes"] if notes_row else "No custom notes."
#             status = notes_row["status"] if notes_row else "returning" if r["total_visits"] > 1 else "new"

#             customers_list.append({
#                 "id": r["id"],
#                 "firstSeen": r["first_seen"],
#                 "lastSeen": r["last_seen"],
#                 "totalVisits": r["total_visits"],
#                 "avgDwellTime": 450,  # calculated via first_seen / last_seen deltas
#                 "status": status,
#                 "avatarSeed": f"cust_{r['id']}",
#                 "notes": notes,
#                 "embeddingCount": emb_count,
#                 "confidenceScore": 0.94
#             })

#         # C. Return dynamic anomalies (e.g., loitering alerts)
#         alerts = []
#         # Query anyone tagged as active "loitering" or custom triggers
#         for c in customers_list:
#             if c["status"] == "loitering":
#                 alerts.append({
#                     "id": f"alt_{c['id']}",
#                     "timestamp": c["lastSeen"],
#                     "level": "warning",
#                     "message": f"Customer Re-ID #{c['id']} triggered Loitering Alert",
#                     "cameraName": "Pharmacy Display Aisle",
#                     "customerId": c["id"],
#                     "resolved": False
#                 })

#     return {
#         "cameras": cameras,
#         "customers": customers_list,
#         "detections": [], # Fetch recent spatial coordinates from detections table
#         "alerts": alerts,
#         "stats": {
#             "onlineCameras": len([c for c in cameras if c["status"] == "online"]),
#             "totalCameras": len(cameras),
#             "totalDetectionsToday": sum(c["totalDetections24h"] for c in cameras),
#             "avgDwellTimeSeconds": 450,
#             "loiteringActiveCount": len([c for c in customers_list if c["status"] == "loitering"]),
#             "reidMatchCount": len([c for c in customers_list if c["totalVisits"] > 1]),
#             "activeAlarmsCount": len(alerts)
#         }
#     }

# # 2. Camera Deletion Router
# @app.delete("/api/cameras/{cam_id}")
# def delete_camera_endpoint(cam_id: int):
#     # Call your existing clean/purge methods...
#     return {"success": True}

# # 3. Customer Profile Updater
# @app.post("/api/customers/{customer_id}/update")
# def update_profile(customer_id: int, updates: CustomerUpdate):
#     with write_connection() as conn:
#         # Save custom supervisor notations/statuses to a local table
#         conn.execute(
#             "INSERT OR REPLACE INTO customer_meta (customer_id, notes, status) VALUES (?, ?, ?)",
#             (customer_id, updates.notes, updates.status)
#         )
#     return {"success": True}




































































# # main.py
# import os
# import cv2
# import time
# import json
# import sqlite3
# import uvicorn
# from loguru import logger
# import multiprocessing as mp
# from src.core.config import settings
# from starlette.requests import Request
# from src.core.logging import setup_logging
# from contextlib import asynccontextmanager
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from src.core.db_writer import start_db_writer
# from werkzeug.security import generate_password_hash, check_password_hash
# from fastapi import FastAPI, Depends, HTTPException, Request, Form, Response
# from src.core.database import init_db,load_cache, read_connection, write_connection
# from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, Response
# from src.engine.pipeline import VisionPipeline

# processors = {}
# CTX = None
# db_queue = None
# response_queue = None
# db_writer_thread=None
# det_queues = {}
# shm_names = {}

# # @asynccontextmanager
# # async def lifespan(app: FastAPI):

# #     # global CTX
# #     global CTX
# #     global db_queue
# #     global response_queue
# #     global db_writer_thread
    
# #     mp.set_start_method("spawn", force=True)
# #     CTX = mp.get_context("spawn")

# #     db_queue = CTX.Queue(maxsize=1024)
# #     response_queue = CTX.Queue()          # one reply pipe for all processes
# #     db_writer_thread = start_db_writer(CTX,db_queue,response_queue)
# #     frame_ready_queue = CTX.Queue(maxsize=64)

# #     init_db()
    
# #     with read_connection() as conn:
# #         load_cache(conn)
# #     yield

# #     for proc in processors.values():
# #         try:
# #             proc.stop()
# #         except Exception as e:
# #             print(f"[PROCESS STOP ERROR] {e}")
# #     try:
# #         db_queue.put(None)
# #     except Exception as e:
# #         print(f"[DB QUEUE STOP ERROR] {e}")
# #     try:
# #         db_writer_thread.join(timeout=5)
# #     except Exception as e:
# #         print(f"[DB WRITER JOIN ERROR] {e}")
    

# # def create_pipeline(stream_url, cam_id, response_queue):
# #     free_slots = CTX.Queue(maxsize=4)
# #     ready_slots = CTX.Queue(maxsize=4)
# #     det_queue = CTX.Queue(maxsize=4)
# #     stop_event = CTX.Event()

# #     processor = VisionPipeline(
# #         RTSP_URL=stream_url,
# #         CAM_ID=cam_id,
# #         ctx=CTX,
# #         free_slots=free_slots,
# #         ready_slots=ready_slots,
# #         det_queue=det_queue,
# #         stop_event=stop_event,
# #         db_queue=db_queue,
# #         response_queue=response_queue
# #     )
# #     return processor  


# # @asynccontextmanager
# # async def lifespan(app: FastAPI):
# #     global CTX, db_queue, db_writer_proc

# #     CTX = mp.get_context("spawn")
# #     db_queue = CTX.Queue(maxsize=1024)
# #     db_writer_proc = start_db_writer(CTX, db_queue)  # no response_queue

# #     init_db()
# #     with read_connection() as conn:
# #         load_cache(conn)
# #     yield

# #     for proc in processors.values():
# #         proc.stop()
# #     db_queue.put(None)
# #     db_writer_proc.join(timeout=5)


# # def create_pipeline(stream_url, cam_id):
# #     free_slots  = CTX.Queue(maxsize=4)
# #     ready_slots = CTX.Queue(maxsize=4)
# #     det_queue   = CTX.Queue(maxsize=4)
# #     response_queue = CTX.Queue()       # ← one per camera, created here
# #     stop_event  = CTX.Event()

# #     processor = VisionPipeline(
# #         RTSP_URL=stream_url,
# #         CAM_ID=cam_id,
# #         ctx=CTX,
# #         free_slots=free_slots,
# #         ready_slots=ready_slots,
# #         det_queue=det_queue,
# #         stop_event=stop_event,
# #         db_queue=db_queue,
# #         response_queue=response_queue  # camera owns its queue
# #     )
# #     return processor
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
# from src.engine.pipeline import VisionPipeline, batched_detector_worker


# processors        = {}
# batched_det_proc  = None
# frame_ready_queue = None
# det_queues        = {}
# free_slots_queues = {}
# shm_names         = {}
# CTX               = None
# db_queue          = None
# db_writer_proc    = None
# stop_event        = None
# # No batched_det_proc as a lazy global — start it explicitly


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global CTX, db_queue, db_writer_proc
#     global frame_ready_queue, stop_event, batched_det_proc

#     CTX        = mp.get_context("spawn")
#     stop_event = CTX.Event()
#     db_queue   = CTX.Queue(maxsize=1024)
#     frame_ready_queue = CTX.Queue(maxsize=64)

#     db_writer_proc = start_db_writer(CTX, db_queue)
#     init_db()
#     with read_connection() as conn:
#         load_cache(conn)

#     # Register all known cameras BEFORE starting the detector
#     for cam_id, stream_url in settings.CAMERAS.items():
#         proc = create_pipeline(stream_url, cam_id)
#         processors[cam_id] = proc

#     # Now dicts are fully populated — start detector once
#     batched_det_proc = _start_batched_detector()

#     # Start camera pipelines
#     for proc in processors.values():
#         proc.start()

#     yield

#     # Shutdown
#     stop_event.set()
#     for proc in processors.values():
#         proc.stop()
#     batched_det_proc.join(timeout=3)
#     db_queue.put(None)
#     db_writer_proc.join(timeout=5)


# def _start_batched_detector() -> mp.Process:
#     """
#     Call ONLY after all cameras are registered in det_queues,
#     free_slots_queues, and shm_names.
#     """
#     p = CTX.Process(
#         target=batched_detector_worker,
#         args=(
#             frame_ready_queue,
#             det_queues,           # fully populated at this point
#             free_slots_queues,    # fully populated
#             shm_names,            # fully populated
#             settings.FRAME_SHAPE,
#             settings.FRAME_BYTES,
#             stop_event,
#         ),
#         daemon=True
#     )
#     p.start()
#     return p


# def create_pipeline(stream_url: str, cam_id: int) -> VisionPipeline:
#     """
#     Register a camera and create its pipeline.
#     Does NOT start processes — call proc.start() separately.
#     """
#     free_slots     = CTX.Queue(maxsize=4)
#     det_queue      = CTX.Queue(maxsize=4)
#     response_queue = CTX.Queue()

#     # Register in global dicts BEFORE starting detector
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

#     # Register shm name after VisionPipeline creates shared memory
#     shm_names[cam_id] = processor.input_shm_name
#     return processor  
  
  
  
  
  
  
  

# # processors         = {}
# # batched_det_proc   = None
# # frame_ready_queue  = None   # cameras push (cam_id, idx) here
# # det_queues         = {}     # cam_id -> Queue, embedder reads from here
# # shm_names          = {}     # cam_id -> shared memory name
# # CTX                = None
# # db_queue           = None
# # db_writer_proc     = None
# # stop_event         = None
# # free_slots_queues = {}   # cam_id -> Queue




# # @asynccontextmanager
# # async def lifespan(app: FastAPI):
# #     global CTX, db_queue, db_writer_proc, frame_ready_queue, stop_event

# #     CTX = mp.get_context("spawn")

# #     stop_event        = CTX.Event()           # ← initialize here
# #     db_queue          = CTX.Queue(maxsize=1024)
# #     frame_ready_queue = CTX.Queue(maxsize=64)
# #     db_writer_proc    = start_db_writer(CTX, db_queue)

# #     init_db()
# #     with read_connection() as conn:
# #         load_cache(conn)
# #     yield

# #     # Shutdown
# #     stop_event.set()                          # ← signals all workers to exit
# #     for proc in processors.values():
# #         proc.stop()
# #     if batched_det_proc is not None:
# #         batched_det_proc.join(timeout=3)
# #     db_queue.put(None)
# #     db_writer_proc.join(timeout=5)


# # def _ensure_batched_detector(stop_event):
# #     """Start the shared detector process once the first camera connects."""
# #     global batched_det_proc
# #     if batched_det_proc is not None and batched_det_proc.is_alive():
# #         return
# #     batched_det_proc = CTX.Process(
# #     target=batched_detector_worker,
# #     args=(
# #         frame_ready_queue,
# #         det_queues,
# #         free_slots_queues,   # ← added
# #         shm_names,
# #         settings.FRAME_SHAPE,
# #         settings.FRAME_BYTES,
# #         stop_event,
# #     ),
# #     daemon=True
# # )
# #     batched_det_proc.start()



# # def create_pipeline(stream_url: str, cam_id: int) -> VisionPipeline:
# #     free_slots     = CTX.Queue(maxsize=4)
# #     frame_ready_queue    = CTX.Queue(maxsize=4)   # reader → frame_ready_queue (not detector_worker)
# #     det_queue      = CTX.Queue(maxsize=4)   # batched_detector → embedder
# #     response_queue = CTX.Queue()
# #     stop_event  = CTX.Event()

# #     # Register this camera's queues and shm name globally
# #     # (batched_detector_worker needs them)
# #     det_queues[cam_id] = det_queue
# #     free_slots_queues[cam_id] = free_slots   # register globally

# #     processor = VisionPipeline(
# #         RTSP_URL=stream_url,
# #         CAM_ID=cam_id,
# #         ctx=CTX,
# #         free_slots=free_slots,
# #         det_queue=det_queue,
# #         stop_event=stop_event,
# #         db_queue=db_queue,
# #         response_queue=response_queue,
# #         frame_ready_queue=frame_ready_queue,
# #     )
# #     # After VisionPipeline creates its shm, register the name
# #     shm_names[cam_id] = processor.input_shm_name
# #     return processor
  

# setup_logging()
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# app = FastAPI(lifespan=lifespan, title="Shop Vision API", version="1.0.0")
# # Mount Jinja2 templates
# templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "server", "templates"))
# # Mount static files
# app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "server", "static")), name="static")

# def is_logged_in(request: Request) -> bool:
#     return "user_id" in request.cookies  # Add logic here to verify authenticity later

# def flash_message(response: Response, message: str, category: str = "info"):
#     """
#     Set a cookie containing a flash message.
#     The cookie expires in 10 seconds – just enough for the next request.
#     """
#     import datetime
#     expires = datetime.datetime.now() + datetime.timedelta(seconds=20)
#     payload = json.dumps({"message": message, "category": category})
#     response.set_cookie(
#         key="flash_msg",
#         value=payload,
#         max_age=10,
#         expires=expires.strftime("%a, %d %b %Y %H:%M:%S GMT"),
#         samesite="lax",
#     )

# def get_flash_message(request: Request) -> dict | None:
#     """Read the flash cookie and return its content as a dict."""
#     cookie = request.cookies.get("flash_msg")
#     if cookie:
#         try:
#             return json.loads(cookie)
#         except (json.JSONDecodeError, TypeError):
#             pass
#     return None

# def clear_flash_message(response: Response):
#     """Remove the flash cookie (send it to the browser with empty value)."""
#     response.delete_cookie("flash_msg")

# @app.get("/", response_class=HTMLResponse)
# async def home(request: Request):
#     flash = get_flash_message(request)
    
#     response = templates.TemplateResponse("home.html", {
#         "request": request,
#         "logged_in": is_logged_in(request),
#         "flash_message": flash
#     })
#     clear_flash_message(response)
#     return response

# @app.get("/about", response_class=HTMLResponse)
# async def about(request: Request):
#     flash = get_flash_message(request)

#     response = templates.TemplateResponse("about.html", {
#         "request": request,
#         "logged_in": is_logged_in(request),
#         "flash_message": flash
#     })
#     clear_flash_message(response)
#     return response

# @app.get("/register_user", response_class=HTMLResponse)
# async def register_page(request: Request):
#     flash = get_flash_message(request)

#     response = templates.TemplateResponse("register_user.html", {
#         "request": request,
#         "logged_in": is_logged_in(request),
#         "flash_message": flash
#     })
#     clear_flash_message(response)
#     return response

# @app.post("/register_user")
# async def register_user(username: str = Form(...), password: str = Form(...)):
#     try:
#         with write_connection() as conn:
#             conn.execute(
#                 "INSERT INTO users (username, password_hash) VALUES (?, ?)",
#                 (username, generate_password_hash(password))
#             )

#         resp = RedirectResponse(url="/login", status_code=302)
#         flash_message(resp, f"{username} Registered Successfully", "success")
#         logger.info(f"{username} Registered Successfully")
#         return resp

#     except sqlite3.IntegrityError:
#         resp = RedirectResponse(url="/register_user", status_code=302)
#         flash_message(resp, f"Username: {username} already taken", "danger")
#         return resp

# @app.get("/login", response_class=HTMLResponse)
# async def login_page(request: Request):
#     flash = get_flash_message(request)

#     response = templates.TemplateResponse("login.html", {
#         "request": request,
#         "logged_in": is_logged_in(request),
#         "flash_message": flash
#     })
#     clear_flash_message(response)
#     return response
    

# @app.post("/login")
# async def do_login(username: str = Form(...), password: str = Form(...)):
#     with write_connection() as conn:
#         row = conn.execute(
#             "SELECT id, password_hash FROM users WHERE username=?", 
#             (username,)
#         ).fetchone()
#     if row and check_password_hash(row[1], password):
#         resp = RedirectResponse(url="/", status_code=302)
#         flash_message(resp, f"Welcome {username}", "success")
#         resp.set_cookie("user_id", str(row[0]), secure=False)
#         logger.info(f"User: {username} Logged In")
#         return resp
#     else:
#         resp = RedirectResponse(url="/login", status_code=302)
#         flash_message(resp, "Invalid credentials", "danger")
#         return resp
    
# @app.get("/logout")
# async def logout():
#     """Logout endpoint"""
#     # Clear session cookie and server-side session
#     response = RedirectResponse(url="/", status_code=302)
#     response.delete_cookie("user_id")
#     logger.info(f"The User Logged Out")
#     flash_message(response, f"The User Logged Out", "success")
#     return response

# @app.get("/register", response_class=HTMLResponse)
# async def camera_page(request: Request):
#     if not is_logged_in(request):
#         response = RedirectResponse(url="/", status_code=302)
#         flash_message(response, "You need to log in first.", "danger")
#         return response

#     user_id = request.cookies.get("user_id")
#     with read_connection() as conn:
#         row_cameras = conn.execute(
#             "SELECT id, name, stream_url FROM cameras WHERE user_id=?", 
#             (user_id,)
#         ).fetchall()
#         cameras = [{"id": row[0], "name": row[1], "url": row[2]} for row in row_cameras]

#     flash = get_flash_message(request)
#     response = templates.TemplateResponse("register.html", {
#         "request": request,
#         "logged_in": True,
#         "cameras": cameras,
#         "flash_message": flash
#     })
#     clear_flash_message(response)   # remove cookie after use
#     return response

# # --- Camera Management ---
# @app.post("/register")
# async def register_camera(request: Request, name: str = Form(...), url: str = Form(...)):
#     if not is_logged_in(request):
#         response = RedirectResponse(url="/", status_code=302)
#         flash_message(response, "You need to log in first.", "danger")
#         return response
#     # try:
#     user_id = request.cookies.get("user_id")
#     if not name or not url:
#         response = RedirectResponse(url="/register", status_code=302)
#         flash_message(response, "Camera name and URL are required.", "danger")
#         return response
#     with write_connection() as conn:
#         conn.execute(
#             "INSERT INTO cameras (name, stream_url, user_id) VALUES (?, ?, ?)",
#             (name.strip(), url.strip(), user_id))

#     resp = RedirectResponse(url="/register", status_code=302)
#     flash_message(resp, f"Camera '{name.strip()}' registered successfully.", "success")
#     logger.info(f"Camera '{name.strip()}' registered successfully.")
#     return resp

# @app.post("/delete-camera/{cam_id}")
# async def delete_camera(cam_id: int, request: Request):
#     if not is_logged_in(request):
#         response = RedirectResponse(url="/", status_code=302)
#         flash_message(response, "You need to log in first.", "danger")
#         return response

#     user_id = request.cookies.get("user_id")
#     try:
#         with write_connection() as conn:
#             # 1. Verify ownership
#             cam = conn.execute(
#                 "SELECT id FROM cameras WHERE id = ? AND user_id = ?",
#                 (cam_id, int(user_id))
#             ).fetchone()
#             if not cam:
#                 resp = RedirectResponse(url="/register", status_code=302)
#                 flash_message(resp, "Camera not found or access denied.", "danger")
#                 return resp

#             conn.execute("DELETE FROM detections WHERE camera_id = ?", (cam_id,))
#             emb_ids = [
#                 row[0] for row in conn.execute(
#                     "SELECT id FROM embedding_meta WHERE camera_id = ?",
#                     (cam_id,)).fetchall()]
#             conn.execute("DELETE FROM embedding_meta WHERE camera_id = ?", (cam_id,))
#             if emb_ids:
#                 conn.execute(
#                     "DELETE FROM embeddings WHERE rowid IN ({})".format(
#                         ",".join("?" * len(emb_ids))), emb_ids)

#             conn.execute(
#                 "DELETE FROM cameras WHERE id = ? AND user_id = ?",
#                 (cam_id, int(user_id)))

#         resp = RedirectResponse(url="/register", status_code=302)
#         flash_message(resp, f"Camera {cam_id} and all related data deleted.", "success")
#         return resp

#     except Exception as e:
#         resp = RedirectResponse(url="/register", status_code=302)
#         flash_message(resp, f"Deletion failed: {e}", "danger")
#         return resp


# @app.get("/video_feed/{cam_id}")
# async def video_feed(cam_id: int, request: Request):
#     if not is_logged_in(request):
#         return RedirectResponse(url="/", status_code=302)
#     processor = processors.get(cam_id)
#     if processor is None:
#         # Fetch camera URL from database
#         user_id = request.cookies.get("user_id")
#         with read_connection() as conn:
#             row = conn.execute(
#                 "SELECT stream_url FROM cameras WHERE id=? AND user_id=?",
#                 (cam_id, user_id)).fetchone()
#         if not row:
#             raise HTTPException(status_code=404, detail="Camera not found")

#         stream_url = row[0]
#         processor = processors.get(cam_id)
#         if processor is None:
#             processor = create_pipeline(stream_url, cam_id)
#             processors[cam_id] = processor
#             processor.start()
#             # if processor.online:
#             #     processors[cam_id] = processor
#             #     processor.start()
#             # else:
#             #     response = RedirectResponse(url=f"/video_feed/{cam_id}", status_code=302)
#             #     flash_message(response, f"Could not open stream for Camera with ID: {cam_id}. Check the URL.", "danger")
#         else:
#             response = RedirectResponse(url=f"/video_feed/{cam_id}", status_code=302)
#             flash_message(response, f"Could not open stream for Camera with ID: {cam_id}. Check the URL.", "danger")

#     def generate():
#         while True:
#             frame = processor.get_latest_frame()
#             if frame is not None:
#                 ret, buffer = cv2.imencode('.jpg', frame)
#                 if ret:
#                     yield (b'--frame\r\n'
#                         b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
#             time.sleep(0.03)   
#     return StreamingResponse(
#         generate(),
#         media_type="multipart/x-mixed-replace; boundary=frame")


# @app.get("/monitor", response_class=HTMLResponse)
# async def monitor(request: Request):
#     """
#     Renders the monitor template with the list of cameras.
#     """
#     if not is_logged_in(request):
#         response = RedirectResponse(url="/", status_code=302)
#         flash_message(response, "You need to log in first.", "danger")
#         return response
    
#     user_id = request.cookies.get("user_id")
#     with read_connection() as conn:
#         conn.row_factory = sqlite3.Row
#         cameras = conn.execute(
#             "SELECT id, name, stream_url FROM cameras WHERE user_id=?",
#             (user_id,)
#         ).fetchall()

#     # Convert the cameras dict to a list of dicts (id and name)
#     cam_list = [{'id': tup[0], 'name': tup[1], 'path': tup[2]} for tup in cameras]
#     return templates.TemplateResponse("monitor.html", {
#             "request": request,
#             "cameras": cam_list,
#             "logged_in": is_logged_in(request)
#         })
    
# # --- Analysis ---
# @app.get("/analysis", response_class=HTMLResponse)
# async def analysis(request: Request):
#     flash = get_flash_message(request)

#     response = templates.TemplateResponse("analysis.html", {
#         "request": request,
#         "logged_in": is_logged_in(request),
#         "flash_message": flash
#     })
#     clear_flash_message(response)
#     return response
    
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)    












































# # # Python FastAPI Expected JSON Payload Schema /api/state:



# # from pydantic import BaseModel
# # from typing import Optional, List
# # # from google import genai


# # import json
# # import sqlite3
# # import uvicorn
# # from loguru import logger
# # import multiprocessing as mp
# # from src.core.config import settings
# # from starlette.requests import Request
# # from src.core.logging import setup_logging
# # from contextlib import asynccontextmanager
# # from fastapi.staticfiles import StaticFiles
# # from fastapi.templating import Jinja2Templates
# # from src.core.db_writer import start_db_writer
# # from werkzeug.security import generate_password_hash, check_password_hash
# # from fastapi import FastAPI, Depends, HTTPException, Request, Form, Response
# # from src.core.database import init_db,load_cache, read_connection, write_connection
# # from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, Response
# # from src.engine.pipeline import VisionPipeline

# # from fastapi import FastAPI
# # from pydantic import BaseModel
# # from typing import List, Optional

# # app = FastAPI()

# # @app.get("/api/state")
# # async def get_state():
# #     return {
# #         # 1. Registered Cameras List
# #         "cameras": [
# #             {
# #                 "id": 1,
# #                 "name": "Entrance CCTV",
# #                 "streamUrl": "rtsp://192.168.1.100:554/h264",
# #                 "status": "online",  # "online" or "offline"
# #                 "fps": 28,
# #                 "resolution": "1920x1080",
# #                 "activeZoneCount": 1,
# #                 "totalDetections24h": 342
# #             }
# #         ],
# #         # 2. Registered unique identities
# #         "customers": [
# #             {
# #                 "id": 101,
# #                 "firstSeen": "2026-05-24T10:15:00Z",
# #                 "lastSeen": "2026-05-24T14:10:00Z",
# #                 "totalVisits": 4,
# #                 "avgDwellTime": 410,       # in seconds
# #                 "status": "returning",     # "new" | "returning" | "loitering" | "flagged"
# #                 "avatarSeed": "cust_101",  # seeds high-contrast SVG placeholders
# #                 "notes": "Regular morning buyer.",
# #                 "embeddingCount": 5,       # current exemplar count
# #                 "latestCameraId": 1,
# #                 "confidenceScore": 0.94
# #             }
# #         ],
# #         # 3. Raw matching detections feed
# #         "detections": [
# #             {
# #                 "id": "det_1",
# #                 "customerId": 101,
# #                 "cameraId": 1,
# #                 "cameraName": "Entrance CCTV",
# #                 "timestamp": "2026-05-24T14:02:00Z",
# #                 "dwellTime": 45,
# #                 "confidence": 0.95,
# #                 "bbox": [250, 150, 410, 480],  # bounding box cords [x1, y1, x2, y2]
# #                 "loitering": False,
# #                 "currentZone": "Entrance Vestibule"
# #             }
# #         ],
# #         # 4. Critical Warning Alerts
# #         "alerts": [
# #             {
# #                 "id": "alt_1",
# #                 "timestamp": "2026-05-24T11:35:00Z",
# #                 "level": "warning",    # "info" | "warning" | "danger"
# #                 "message": "Customer ID #102 loitering in Fragrance section > 15 mins",
# #                 "cameraName": "Cosmetics Area",
# #                 "customerId": 102,
# #                 "resolved": False
# #             }
# #         ]
# #     }










































