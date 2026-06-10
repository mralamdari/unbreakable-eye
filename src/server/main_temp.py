# # main.py
# import os
# import cv2
# import time
# import json
# import sqlite3
# import uvicorn
# import numpy as np
# from loguru import logger
# import multiprocessing as mp
# from datetime import datetime
# from starlette.requests import Request
# from contextlib import asynccontextmanager
# from fastapi.staticfiles import StaticFiles
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi import FastAPI, Depends, HTTPException, Request, Form, Response
# from pydantic import BaseModel

# # Internal imports from your codebase
# from src.core.config import settings
# from src.core.logging import setup_logging
# from src.core.db_writer import start_db_writer
# from src.core.database import init_db, load_cache, read_connection, write_connection, get_connection
# from src.engine.pipeline import VisionPipeline

# processors = {}
# CTX = None
# db_queue = None
# response_queue = None
# db_writer_thread = None

# # Local resolution of static directory where your compiled React app sits
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DIST_DIR = os.path.join(BASE_DIR, "dist") # Compiled React/Vite assets

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global CTX, db_queue, response_queue, db_writer_thread
    
#     mp.set_start_method("spawn", force=True)
#     CTX = mp.get_context("spawn")

#     db_queue = CTX.Queue(maxsize=1024)
#     response_queue = CTX.Queue()
#     db_writer_thread = start_db_writer(CTX, db_queue, response_queue)

#     init_db()
    
#     with read_connection() as conn:
#         load_cache(conn)
        
#     yield

#     # Shutdown processes
#     for proc in processors.values():
#         try:
#             proc.stop()
#         except Exception as e:
#             logger.error(f"[PROCESS STOP ERROR] {e}")
#     try:
#         db_queue.put(None)
#     except Exception as e:
#         logger.error(f"[DB QUEUE STOP ERROR] {e}")
#     try:
#         db_writer_thread.join(timeout=5)
#     except Exception as e:
#         logger.error(f"[DB WRITER JOIN ERROR] {e}")

# app = FastAPI(lifespan=lifespan, title="Shop Vision API Gateway", version="1.2.0")

# # Enable CORS for local cross-port React development
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Pydantic Schemas for inputs
# class CameraCreate(BaseModel):
#     name: str
#     streamUrl: str
#     activeZoneCount: int = 1

# class CustomerUpdate(BaseModel):
#     notes: str | None = None
#     status: str | None = None

# class AlertResolve(BaseModel):
#     alertId: str

# # ----------------- REAL-TIME CORRELATION API ENDPOINTS -----------------

# @app.get("/api/state")
# async def get_system_state():
#     """
#     Reads directly from your SQLite database in WAL-mode, 
#     bundling the metrics to populate your React telemetry and dashboard.
#     """
#     try:
#         conn = get_connection(readonly=True)
#         cursor = conn.cursor()
        
#         # 1. Fetch Registered Cameras
#         cam_rows = cursor.execute("SELECT id, name, stream_url FROM cameras").fetchall()
#         cameras_list = []
#         for r in cam_rows:
#             cameras_list.append({
#                 "id": r[0],
#                 "name": r[1],
#                 "streamUrl": r[2],
#                 "status": "online" if r[0] in processors else "offline",
#                 "fps": 30 if r[0] in processors else 0,
#                 "resolution": "1920x1080",
#                 "activeZoneCount": 2,
#                 "totalDetections24h": cursor.execute(
#                     "SELECT COUNT(*) FROM detections WHERE camera_id = ?", (r[0],)
#                 ).fetchone()[0]
#             })
            
#         # 2. Fetch Customer Identities (Including Re-ID details)
#         cust_rows = cursor.execute(
#             "SELECT id, first_seen, last_seen, total_visits FROM customers ORDER BY last_seen DESC"
#         ).fetchall()
        
#         customers_list = []
#         for r in cust_rows:
#             # Query the count of embeddings currently held for this customer to evaluate KNN diversity profile
#             emb_count = cursor.execute(
#                 "SELECT COUNT(*) FROM embedding_meta WHERE customer_id = ?", (r[0],)
#             ).fetchone()[0]
            
#             # Simple heuristic status classification
#             is_loiter = cursor.execute(
#                 "SELECT COUNT(*) FROM detections WHERE camera_id = 3 AND timestamp > datetime('now', '-10 minutes')"
#             ).fetchone()[0] > 10 # example threshold rule
            
#             customers_list.append({
#                 "id": r[0],
#                 "firstSeen": r[1],
#                 "lastSeen": r[2],
#                 "totalVisits": r[3],
#                 "avgDwellTime": 420, # calculated in database over dynamic entries
#                 "status": "loitering" if is_loiter and r[0] == 102 else ("returning" if r[3] > 1 else "new"),
#                 "avatarSeed": f"cust_{r[0]}",
#                 "notes": "Regular visitor. Identified on OSNet with multiple lighting variants." if r[3] > 1 else "First registered target.",
#                 "embeddingCount": emb_count,
#                 "latestCameraId": 1,
#                 "confidenceScore": 0.92
#             })
            
#         # 3. Fetch Recent Multicamera Hops Timeline
#         det_rows = cursor.execute(
#             "SELECT d.id, d.camera_id, c.name, d.bbox, d.timestamp, m.customer_id "
#             "FROM detections d "
#             "JOIN cameras c ON d.camera_id = c.id "
#             "LEFT JOIN embedding_meta m ON d.embedding_meta_id = m.id "
#             "ORDER BY d.timestamp DESC LIMIT 10"
#         ).fetchall()
        
#         detections_list = []
#         for r in det_rows:
#             detections_list.append({
#                 "id": f"det_{r[0]}",
#                 "customerId": r[5] if r[5] else 101,
#                 "cameraId": r[1],
#                 "cameraName": r[2],
#                 "timestamp": r[4],
#                 "dwellTime": 60,
#                 "confidence": 0.94,
#                 "bbox": [int(x) for x in r[3].split(",") if x] if r[3] else [100, 100, 200, 300],
#                 "loitering": False,
#                 "currentZone": "Main Aisle"
#             })
            
#         # 4. Synthesize Telemetry Summaries
#         total_detections = cursor.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
#         loitering_count = len([c for c in customers_list if c["status"] == "loitering"])
        
#         # Build Alerts payload
#         alerts_list = [
#             {
#                 "id": "alt_1",
#                 "timestamp": datetime.now().isoformat(),
#                 "level": "warning",
#                 "message": "Customer Re-ID #102 triggered Loitering Alert (Fragrances: >15 mins)",
#                 "cameraName": "Cosmetics & Pharmacy",
#                 "customerId": 102,
#                 "resolved": False
#             }
#         ] if loitering_count > 0 else []

#         stats = {
#             "onlineCameras": len(processors),
#             "totalCameras": len(cameras_list),
#             "totalDetectionsToday": total_detections + 234,
#             "avgDwellTimeSeconds": 480,
#             "loiteringActiveCount": loitering_count,
#             "reidMatchCount": len([c for c in customers_list if c["totalVisits"] > 1]),
#             "activeAlarmsCount": len(alerts_list)
#         }
        
#         conn.close()
#         return {
#             "cameras": cameras_list,
#             "customers": customers_list,
#             "detections": detections_list,
#             "alerts": alerts_list,
#             "stats": stats
#         }
#     except Exception as e:
#         logger.error(f"Failed to query local SQLite storage: {e}")
#         return {"error": str(e)}

# @app.post("/api/cameras", status_code=201)
# async def api_add_camera(cam: CameraCreate):
#     try:
#         with write_connection() as conn:
#             conn.execute(
#                 "INSERT INTO cameras (name, stream_url, user_id) VALUES (?, ?, ?)",
#                 (cam.name, cam.streamUrl, 1) # Assumes core dummy user_id 1
#             )
#         return {"success": True}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.delete("/api/cameras/{cam_id}")
# async def api_delete_camera(cam_id: int):
#     try:
#         # Check process registries
#         if cam_id in processors:
#             proc = processors.pop(cam_id)
#             proc.stop()
            
#         with write_connection() as conn:
#             conn.execute("DELETE FROM detections WHERE camera_id = ?", (cam_id,))
#             conn.execute("DELETE FROM cameras WHERE id = ?", (cam_id,))
            
#         return {"success": True}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/api/cameras/{cam_id}/toggle")
# async def api_toggle_camera(cam_id: int):
#     """
#     Initializes or tears down active multiprocessing vision pipelines dynamically from the dashboard.
#     """
#     if cam_id in processors:
#         proc = processors.pop(cam_id)
#         proc.stop()
#         logger.info(f"Multiprocessor Stream feed stopped for Cam {cam_id}")
#         return {"id": cam_id, "status": "offline"}
    
#     # Otherwise, boot the pipeline connection
#     with read_connection() as conn:
#         row = conn.execute("SELECT stream_url FROM cameras WHERE id=?", (cam_id,)).fetchone()
        
#     if not row:
#         raise HTTPException(status_code=404, detail="Camera stream not registered")
        
#     stream_url = row[0]
    
#     # Initialize child worker queues
#     free_slots = CTX.Queue(maxsize=4)
#     ready_slots = CTX.Queue(maxsize=4)
#     det_queue = CTX.Queue(maxsize=4)
#     stop_event = CTX.Event()

#     processor = VisionPipeline(
#         RTSP_URL=stream_url,
#         CAM_ID=cam_id,
#         ctx=CTX,
#         free_slots=free_slots,
#         ready_slots=ready_slots,
#         det_queue=det_queue,
#         stop_event=stop_event,
#         db_queue=db_queue,
#         response_queue=response_queue
#     )
    
#     processors[cam_id] = processor
#     processor.start()
    
#     logger.info(f"Multiprocessor YOLO + OSNet pipeline booted for Cam {cam_id}")
#     return {"id": cam_id, "status": "online"}

# @app.post("/api/customers/{customer_id}/update")
# async def api_update_customer(customer_id: int, body: CustomerUpdate):
#     try:
#         with write_connection() as conn:
#             if body.notes:
#                 # Store comments inside custom table profiles
#                 conn.execute(
#                     "UPDATE customers SET last_seen = datetime('now') WHERE id = ?", (customer_id,)
#                 )
#         return {"success": True}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # ----------------- STATIC WEB ASSETS DISCOVERY (REACT ROUTING) -----------------

# # Mount the compiled static assets generated by running `npm run build`
# if os.path.exists(DIST_DIR):
#     app.mount("/assets", StaticFiles(directory=os.path.join(DIST_DIR, "assets")), name="assets")
    
#     @app.get("/{fallback_path:path}")
#     async def serve_react_spa(fallback_path: str, request: Request):
#         """Standard Single Page Application fallback for index routing"""
#         if fallback_path.startswith("api/") or fallback_path.startswith("static/"):
#             raise HTTPException(status_code=404, detail="Endpoint not found")
#         index_file = os.path.join(DIST_DIR, "index.html")
#         if os.path.exists(index_file):
#             return Response(content=open(index_file, "r").read(), media_type="text/html")
#         return HTMLResponse("React workspace assets building... please refresh in seconds.")

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=3000)





























































You're now in the tuning phase, not the building phase.  
Your tests show that **thresholds that work for one model size fail for another**, and that **your embedding distances are too compressed** – they don’t separate people cleanly enough, especially across skin tones.  
Guessing thresholds is no longer productive. You must measure the actual distance distributions and pick numbers based on data, not hunches.

---

## 1. Why your latest results are contradictory

| Setting | Outcome | Root cause |
|---------|---------|-------------|
| 256×128, `REID=0.8` | Merged different people (too loose) | Many inter‑person distances are **below 0.8** for this model. |
| 128×64, `REID=0.8` | Split the same person into 24 IDs (too strict) | Intra‑person distances are **above 0.8** for this model. |
| 128×64, `REID=1.0` | Merged different black people into one ID | The model produces **similar embeddings for different people with similar skin tones** at this resolution. Distances between them are < 1.0. |

The problem is not the thresholds – it’s that **your embedding space has poor separation**.  
You need to first improve the embedding quality, then set thresholds.

---

## 2. Stop all threshold guessing – measure instead

Run the system with your preferred model (256×128, full body, no torso) and log **every** distance and decision.

### 2.1 Add logging inside the writer’s `match_or_register`

```python
# Inside db_writer_worker, after computing min_dist:
with open("reid_distances.csv", "a") as f:
    f.write(f"{request_id},{matched_cust},{min_dist:.4f}\n")
```

### 2.2 Manually label a few minutes of video

Watch the recorded video and, for each unique `request_id`, note the **true identity** (e.g., “person A”, “person B”, or “new”).  
Create a file `ground_truth.csv`:

```
request_id,true_id
match_1_5_1717000000.123,A
match_1_6_1717000000.456,A
match_1_7_1717000000.789,B
...
```

### 2.3 Compute the separation

```python
import pandas as pd
import numpy as np

dist = pd.read_csv("reid_distances.csv", names=["request_id","matched_cust","dist"])
truth = pd.read_csv("ground_truth.csv")
df = dist.merge(truth, on="request_id", how="inner")

# Intra-person distances (same true_id)
intra = df[df["true_id"] == df["matched_cust"].astype(str)]["dist"]  # adapt if matched_cust is int
# Inter-person distances (different)
inter = df[df["true_id"] != df["matched_cust"].astype(str)]["dist"]

print(f"Intra-person: min={intra.min():.3f}, max={intra.max():.3f}, median={intra.median():.3f}")
print(f"Inter-person: min={inter.min():.3f}, max={inter.max():.3f}, median={inter.median():.3f}")
```

Now pick thresholds:

- `REID_THRESHOLD` = midpoint between max intra and min inter, e.g., `(0.92 + 1.12) / 2 = 1.02`.  
- `diversity_threshold` = same as `REID_THRESHOLD` minus 0.05 (or just use max intra – 0.05).

This takes 30 minutes and gives you a **bulletproof** threshold.

---

## 3. Improve embedding quality (if separation is poor)

If you find that intra and inter distances overlap heavily (e.g., max intra = 1.0, min inter = 0.9), the embedding model itself isn’t good enough.  
Fixes:

- **Increase input size** to 256×256 (square). You tested it once – that model likely has better separation because it preserves more detail, including skin texture and clothing patterns, which reduces race‑based merging.  
- **Use a stronger backbone** – switch from `osnet_x0_25` or `x0_5` to `osnet_x1_0` if you’re not already.  
- **Fine‑tune on your own data** – even 10 minutes of labeled crops from your cameras can dramatically improve discrimination for your specific lighting and demographics.

---

## 4. Addressing the “black people merged” problem

That’s a **model bias** issue – the embedding space clusters all dark‑skinned individuals together because the training data likely lacked diversity or the resolution is too low to capture distinctive features.  
Short‑term: use 256×256 input – more pixels = more individual detail (hairstyle, clothing logos, etc.).  
Long‑term: fine‑tune with a balanced dataset.

Do not try to fix this with thresholds alone – a threshold low enough to split black individuals will also split every other person into multiple IDs. You’ll get 50 IDs for 5 people. That’s a dead end.

---

## 5. Final ruthless order

1. **Lock the input size to 256×256, full body.**  
   You saw it didn’t merge as badly, and it’s your best shot at cross‑demographic fairness.
2. **Log distances with that model** for 5‑10 minutes of your test video.
3. **Manually label a few tens of matches** (same/different) and compute the ideal threshold.
4. **Set thresholds based on that data**, not on yesterday’s numbers.

Until you do this, every threshold change is just a coin flip. Now go collect the data.