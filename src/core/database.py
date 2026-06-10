# import os
# import sqlite3
# import threading
# import sqlite_vec
# import numpy as np
# from loguru import logger
# from datetime import datetime
# from src.core.config import settings
# from contextlib import contextmanager
# from typing import Dict, List, Optional, Tuple

# class EmbeddingCache:
#     def __init__(self):
#         self.embeddings = np.empty((0, settings.EMBEDDING_DIM), dtype=np.float32)
#         self.customer_ids = np.empty((0,), dtype=np.int64)
#         self._lock = threading.Lock()   # if you ever go multi‑threaded

#     def load_all(self, conn):
#         rows = conn.execute("""
#             SELECT e.embedding, m.customer_id
#             FROM embeddings e
#             JOIN embedding_meta m ON e.rowid = m.id
#         """).fetchall()
#         if rows:
#             emb_list = [np.frombuffer(r[0], dtype=np.float32) for r in rows]
#             self.embeddings = np.vstack(emb_list).astype(np.float32)
#             self.customer_ids = np.array([r[1] for r in rows], dtype=np.int64)
#         else:
#             self.embeddings = np.empty((0, settings.EMBEDDING_DIM), dtype=np.float32)
#             self.customer_ids = np.empty((0,), dtype=np.int64)

#     # def add(self, embedding: np.ndarray, customer_id: int):
#     #     """Call this immediately after INSERT into SQLite (from the same thread)."""
#     #     embedding = embedding.astype(np.float32).flatten()
#     #     with self._lock:
#     #         # Append to the array efficiently
#     #         self.embeddings = np.vstack([self.embeddings, embedding])
#     #         self.customer_ids = np.append(self.customer_ids, customer_id)

#     def add(self, embedding: np.ndarray, customer_id: int):
#         """Thread‑safe append of a new embedding."""
#         embedding = embedding.astype(np.float32).flatten()
#         with self._lock:
#             if self.embeddings.shape[0] == 0:
#                 self.embeddings = embedding.reshape(1, -1).astype(np.float32)
#                 self.customer_ids = np.array([customer_id], dtype=np.int64)
#             else:
#                 self.embeddings = np.vstack([self.embeddings, embedding])
#                 self.customer_ids = np.append(self.customer_ids, customer_id)
                
#     def get_snapshot(self):
#         """Return a consistent (embeddings, customer_ids) tuple."""
#         with self._lock:
#             # return self.embeddings.copy(), self.customer_ids.copy()
#             return self.embeddings, self.customer_ids # no copy – read‑only arrays are safe because writer creates new arrays via vstack
        
        
# # Global instance – used everywhere
# embedding_cache = EmbeddingCache()

# def load_cache(conn):
#     embedding_cache.load_all(conn)


# # 2. Specific Code Tactics to Fix Re-ID Swaps
# # Tactic A: Spatial-Temporal Transition Calibration
# # Do not perform flat KNN vector checks in isolation. A customer cannot teleport from Entrance CCTV to Checkout Main in 1 second.
# # Introduce a geographical routing adjacency matrix inside src/core/database.py. If the elapsed time (
# # ) is physically impossible for the transit, mathematically inflate the Euclidean distance inside your fast_match logic so they are registered separately:
    
    
    
#     # Create a relative travel-time lookup index (seconds)
# TransitMatrix = {
#     (1, 4): 25.0,  # Cam 1 (Entrance) to Cam 4 (Checkout) takes at least 25s
# }

# def fast_match_calibrated(query_emb, current_cam_id, clock_now):
#     embs, cids = embedding_cache.get_snapshot()
#     if embs.shape[0] == 0:
#         return None, float('inf')
        
#     query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
#     dists = np.linalg.norm(embs - query_emb, axis=1) # standard Euclidean mapping
    
#     for idx, cid in enumerate(cids):
#         # Query the database/cache for where this candidate ID was last seen
#         last_cam = last_known_cam.get(cid)
#         last_time = last_seen_time.get(cid, 0)
#         dt = clock_now - last_time
        
#         # Calculate expected route boundaries
#         min_seconds = TransitMatrix.get((last_cam, current_cam_id), 0.0)
#         if dt < min_seconds:
#             # Artificially penalize the distance so the crop registers as a different identity
#             dists[idx] *= 1.8 
            
#     min_idx = np.argmin(dists)
#     min_dist = dists[min_idx]
    
#     if min_dist < settings.REID_THRESHOLD:
#         return int(cids[min_idx]), min_dist
#     return None, min_dist
    
    
    
    

# # as you can see I have bytetracker and it is useless, 
# # maybe I should delete it and just use the embedders?
# # I mean in my code,I use the embedders and sqlite and strored
# # embeddings on a numpy arracy cache to make the embedding search faster
# # I assume we can go with the:
# #     Tactic A: Spatial-Temporal Transition Calibration
# # as you mentioned but finding the geographical distance is difficult, 
# # maybe I can caculate the center point diff btw a new ID that 
# # the tracker finds and the list of the embeddings and center_points I have already stored
# # so there would be no need to calculate the embeddings for every single embeddnig?
# # and I don't understand the Tactic B: Maintain a Multi-View Exemplar Pool (Rolling Cluster)
# # you gave me as well, so what is the best approach to this problem?











# # [Camera Feed Ingest]
# #           │
# #           ▼
# #    [YOLOv8 Detector] ────► [BBoxes Detected?]
# #           │                      │
# #           ▼ Yes                  ▼ No
# #    [ByteTrack Update] ───► [Render Empty Frame]
# #           │
# #           ▼
# #    [For Each Active target_id in ByteTrack]
# #    ┌─────────────────────────────────────────────────────────────┐
# #    │ If target_id NOT YET MAPPED to a global Customer ID:        │
# #    │   1. Crop box, preprocess, and run through OSNet.           │
# #    │   2. Match minimum distance across global Exemplar Pool.    │
# #    │   3. Map local target_id ──► global Customer ID.            │
# #    │                                                             │
# #    │ If target_id IS ALREADY MAPPED to global Customer ID:        │
# #    │   1. SKIP OSNet entirely.                                   │
# #    │   2. Trust ByteTrack coordinates, increment target dwell.   │
# #    │   3. Every N frames (e.g., 150 frames): run a sparse        │
# #    │      diversity check to add new pose exemplars if needed.    │
# #    └─────────────────────────────────────────────────────────────┘
# #           │
# #           ▼
# # [Update SQLite DB via DB_Queue & Render Dashboard Stats]





# import time


# # In src/core/database.py
# # In real retail situations:
# # Different shoppers wearing denim shirts or red uniforms will register nearly identical OSNet features, causing massive ID swaps.
# # Transition Swaps: A user seen on Cam-2 near high-luxury counters might trigger a match with a user who entered on Cam-1 only 2 seconds ago, ignoring physically impossible transit times.
# # The Code Patch: Spatial-Temporal Transitions
# # Improve your matching logic in fast_match by multiplying Euclidean distance by a time/routing transition factor. If the spatial transfer between cam_previous and cam_current is physically impossible within dt seconds, artificially scale up the Euclidean distance so the model registers them as a separate individual:

# def fast_match_calibrated(query_emb, current_cam_id, last_known_coords=None):
#     """
#     Calibrates raw Cosine/Euclidean distance by route probability and transit timers
#     """
#     embs, cids = embedding_cache.get_snapshot()
#     if embs.shape[0] == 0:
#         return None, float('inf')
        
#     query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
#     dists = np.linalg.norm(embs - query_emb, axis=1)
    
#     # ── HEURISTIC OVERLAY ──
#     for idx, cid in enumerate(cids):
#         # Query where this specific identity ID was spotted last
#         # If timestamp is dt < 10 seconds and camera is geographically separated, scale dist flag
#         dt = time.time() - last_seen_timestamps.get(cid, 0)
#         if dt < 15.0 and last_known_cam.get(cid) != current_cam_id:
#             dists[idx] *= 1.8 # Double the matching penalty for physically impossible transfers!
            
#     min_idx = np.argmin(dists)
#     min_dist = dists[min_idx]
    
#     if min_dist < settings.REID_THRESHOLD:
#         return int(cids[min_idx]), min_dist
#     return None, min_dist



# def fast_match(query_emb):
#     embs, cids = embedding_cache.get_snapshot()
#     if embs.shape[0] == 0:
#         return None, float('inf')
#     query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
#     dists = np.linalg.norm(embs - query_emb, axis=1)
#     min_idx = np.argmin(dists)
#     min_dist = dists[min_idx]
#     if min_dist < settings.REID_THRESHOLD:
#         return int(cids[min_idx]), min_dist
#     return None, min_dist

# def fast_min_dist_to_customer(query_emb, customer_id):
#     embs, cids = embedding_cache.get_snapshot()
#     mask = cids == customer_id
#     if not np.any(mask):
#         return float('inf')
#     dists = np.linalg.norm(embs[mask] - query_emb, axis=1)
#     return float(dists.min())

# def get_connection(readonly: bool = False) -> sqlite3.Connection:
#     """
#     Returns a connection with the sqlite-vec extension loaded.
#     Use `readonly=True` for read‑only dashboard queries
#     (allows concurrent reads in WAL mode).
#     """
#     os.makedirs(os.path.dirname(settings.DB_PATH), exist_ok=True)
#     if readonly:
#         uri = f"file:{settings.DB_PATH}?mode=ro"
#         conn = sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=10)
#     else:
#         conn = sqlite3.connect(settings.DB_PATH, check_same_thread=False, timeout=10)

#     conn.enable_load_extension(True)
#     sqlite_vec.load(conn)
#     conn.enable_load_extension(False)
#     conn.row_factory = sqlite3.Row
#     conn.execute("PRAGMA journal_mode=WAL")
#     conn.execute("PRAGMA synchronous=NORMAL")
#     conn.execute("PRAGMA mmap_size=268435456")
#     conn.execute("PRAGMA foreign_keys=ON")
#     conn.execute("PRAGMA busy_timeout = 5000")   # 5 seconds – more than enough
#     return conn

# @contextmanager
# def write_connection():
#     """Context manager for the single writer connection."""
#     conn = get_connection()
#     try:
#         yield conn
#         conn.commit()
#     except Exception:
#         conn.rollback()
#         raise
#     finally:
#         conn.close()

# @contextmanager
# def read_connection():
#     """Context manager for concurrent readers."""
#     conn = get_connection(readonly=True)
#     try:
#         yield conn
#     finally:
#         conn.close()

# def init_db():
#     """
#     Creates all tables if they don't exist.
#     Run once at application startup.
#     """
#     with write_connection() as conn:
#         # --- User & camera tables (already exist in your code) ---
#         conn.execute("""
#             CREATE TABLE IF NOT EXISTS users (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 username TEXT UNIQUE NOT NULL,
#                 password_hash TEXT NOT NULL
#             )
#         """)
#         conn.execute("""
#             CREATE TABLE IF NOT EXISTS cameras (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 user_id INTEGER NOT NULL,
#                 name TEXT NOT NULL,
#                 stream_url TEXT NOT NULL,
#                 FOREIGN KEY (user_id) REFERENCES users(id)
#             )
#         """)

#         # --- Vector store for Re‑ID ---
#         # The vec0 virtual table holds the actual embedding arrays.
#         conn.execute(f"""
#             CREATE VIRTUAL TABLE IF NOT EXISTS embeddings USING vec0(
#                 embedding float[{settings.EMBEDDING_DIM}]
#             )
#         """)

#         # --- Metadata linked to each embedding ---
#         conn.execute("""
#             CREATE TABLE IF NOT EXISTS embedding_meta (
#                 id INTEGER PRIMARY KEY,
#                 customer_id INTEGER NOT NULL,
#                 camera_id INTEGER NOT NULL,
#                 timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
#                 FOREIGN KEY (customer_id) REFERENCES customers(id),
#                 FOREIGN KEY (camera_id) REFERENCES cameras(id)
#             )
#         """)

#         # --- Unified customer / person identity ---
#         conn.execute("""
#             CREATE TABLE IF NOT EXISTS customers (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
#                 last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
#                 total_visits INTEGER DEFAULT 1
#             )
#         """)
#         # This records every detection with its bounding box, timestamp, and a pointer to the embedding that was (optionally) stored.
#         conn.execute("""
#             CREATE TABLE IF NOT EXISTS detections (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 camera_id INTEGER NOT NULL,
#                 bbox TEXT NOT NULL,               -- stored as "x1,y1,x2,y2"
#                 center_point TEXT NOT NULL,       -- stored as "x,y"
#                 timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
#                 embedding_meta_id INTEGER,
#                 FOREIGN KEY (camera_id) REFERENCES cameras(id),
#                 FOREIGN KEY (embedding_meta_id) REFERENCES embedding_meta(id)
#             )
#         """)
#     logger.info("Initializing the Database...")

# # def store_embedding(conn, customer_id, camera_id, embedding, timestamp=None) -> int:
# #     if timestamp is None:
# #         timestamp = datetime.now()

# #     # Insert into vec0 table
# #     cursor = conn.execute(
# #         "INSERT INTO embeddings (embedding) VALUES (?)",
# #         (embedding.astype(np.float32).tobytes(),)
# #     )
# #     emb_rowid = cursor.lastrowid

# #     # Insert meta -> get the id we need for detections
# #     cursor = conn.execute(
# #         "INSERT INTO embedding_meta (id, customer_id, camera_id, timestamp) "
# #         "VALUES (?, ?, ?, ?)",
# #         (emb_rowid, customer_id, camera_id, timestamp)
# #     )
# #     meta_id = cursor.lastrowid   # <-- this is the embedding_meta.id

# #     # Update numpy cache
# #     global embedding_cache
    
# #     if embedding_cache.embeddings.shape[0] == 0:
# #         embedding_cache.embeddings = embedding.reshape(1, -1).astype(np.float32)
# #         embedding_cache.customer_ids = np.array([customer_id], dtype=np.int64)
# #     else:
# #         embedding_cache.embeddings = np.vstack([embedding_cache.embeddings, embedding])
# #         embedding_cache.customer_ids = np.append(embedding_cache.customer_ids, customer_id)

# #     return meta_id   # return the embedding_meta id

# def store_embedding(conn, customer_id, camera_id, embedding, timestamp=None) -> int:
#     if timestamp is None:
#         timestamp = datetime.now()

#     # Insert into vec0 table
#     cursor = conn.execute(
#         "INSERT INTO embeddings (embedding) VALUES (?)",
#         (embedding.astype(np.float32).tobytes(),)
#     )
#     emb_rowid = cursor.lastrowid

#     # Insert meta
#     cursor = conn.execute(
#         "INSERT INTO embedding_meta (id, customer_id, camera_id, timestamp) "
#         "VALUES (?, ?, ?, ?)",
#         (emb_rowid, customer_id, camera_id, timestamp)
#     )
#     meta_id = cursor.lastrowid

#     # ---- Keep the in‑memory cache instantly up‑to‑date ----
#     embedding_cache.add(embedding, customer_id)   # ← thread‑safe, handles empty cache
#     # ---------------------------------------------------------
#     return meta_id

# def find_matching_person(conn: sqlite3.Connection,
#                          query_embedding: np.ndarray) -> Optional[int]:
#     q = query_embedding.astype(np.float32).tobytes()
#     sql = """
#         SELECT
#             embeddings.rowid,
#             distance,
#             embedding_meta.customer_id
#         FROM embeddings
#         JOIN embedding_meta ON embeddings.rowid = embedding_meta.id
#         WHERE embedding MATCH ? AND k = 1
#     """
#     result = conn.execute(sql, (q,)).fetchone()
#     if result and result['distance'] < settings.REID_THRESHOLD:
#         return result['customer_id']
#     return None

# def get_customer_analytics(conn: sqlite3.Connection,
#                         customer_id: int) -> Dict:
#     """Quick summary for dashboard."""
#     info = conn.execute(
#         "SELECT * FROM customers WHERE id=?", (customer_id,)
#     ).fetchone()
#     visits = conn.execute(
#         "SELECT COUNT(*) as count FROM visits WHERE customer_id=?",
#         (customer_id,)
#     ).fetchone()
#     return {**dict(info), 'total_visits': visits['count']}

# def purge_old_embeddings(conn: sqlite3.Connection, days=30):
#     """Deletes embeddings older than `days` to keep the search set tight."""
#     cutoff = datetime.now() - datetime.timedelta(days=days)
#     conn.execute(
#         "DELETE FROM embeddings WHERE rowid IN ("
#         "  SELECT id FROM embedding_meta WHERE timestamp < ?"
#         ")", (cutoff,)
#     )
#     conn.execute(
#         "DELETE FROM embedding_meta WHERE timestamp < ?", (cutoff,)
#     )

# # This tells you how similar the new embedding is to what you already have.
# # If the distance is large (e.g. > 0.3), the new pose adds valuable diversity.
# def min_distance_to_customer(conn: sqlite3.Connection,
#                              customer_id: int,
#                              query_embedding: np.ndarray) -> float:
#     """
#     Returns the minimum Euclidean distance between `query_embedding`
#     and any existing embedding of the given customer.
#     Returns float('inf') if no embeddings exist.
#     """
#     q = query_embedding.astype(np.float32).tobytes()
#     sql = f"""
#         SELECT MIN(distance) as min_dist
#         FROM (
#             SELECT distance, embedding_meta.customer_id
#             FROM embeddings
#             JOIN embedding_meta ON embeddings.rowid = embedding_meta.id
#             WHERE embedding MATCH ? AND k = {settings.KNN_N}
#         )
#         WHERE customer_id = ?
#     """
#     row = conn.execute(sql, (q, customer_id)).fetchone()
#     if row and row['min_dist'] is not None:
#         return row['min_dist']
#     return float('inf')

# # def process_detection(conn: sqlite3.Connection,
# #                       camera_id: int,
# #                       embedding: np.ndarray,
# #                       bbox: Tuple[int, int, int, int],   # (x1, y1, x2, y2)
# #                       center_point: Tuple[float, float],
# #                       timestamp: Optional[datetime] = None,
# #                     #   reid_threshold: float = settings.REID_THRESHOLD,
# #                       diversity_threshold: float = 0.2
# #                       ) -> Dict:
# #     """
# #     Full pipeline for one detected person:

# #     1. Match embedding against the gallery.
# #     2. If match found:
# #          - If the new embedding is sufficiently different from the
# #            existing gallery (distance > diversity_threshold), store it.
# #     3. If no match:
# #          - Create a new customer and store this embedding as the first.
# #     4. Always log a detection record (with bbox).

# #     Returns a dict with:
# #       - customer_id
# #       - is_new_customer
# #       - embedding_stored (bool)
# #       - detection_id
# #     """
# #     if timestamp is None:
# #         timestamp = datetime.now()

# #     # ---- 1. Try to re-identify ----
# #     existing_customer = find_matching_person(conn, embedding)
# #     is_new_customer = (existing_customer is None)
# #     embedding_stored = False

# #     if is_new_customer:
# #         # Create customer and immediately store the first embedding
# #         cur = conn.execute(
# #             "INSERT INTO customers (first_seen, last_seen) VALUES (?, ?)",
# #             (timestamp, timestamp)
# #         )
        
# #         customer_id = cur.lastrowid
# #         emb_meta_id = store_embedding(conn, customer_id, camera_id, embedding, timestamp)
# #         embedding_stored = True
# #     else:
# #         customer_id = existing_customer
# #         # Optionally update last_seen
# #         conn.execute("UPDATE customers SET last_seen = ? WHERE id = ?", (timestamp, customer_id))

# #         # Check diversity: how far is this embedding from the customer's closest one?
# #         min_dist = min_distance_to_customer(conn, customer_id, embedding)
# #         if min_dist > diversity_threshold:
# #             emb_meta_id = store_embedding(conn, customer_id, camera_id, embedding, timestamp)
# #             embedding_stored = True
# #         else:
# #             emb_meta_id = None   # no new embedding stored

# #     ###### DO I NEED IT?
# #     # ---- 4. Log the detection ----
# #     bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
# #     center_str = f"{center_point[0]:.2f},{center_point[1]:.2f}"
# #     cur = conn.execute(
# #         "INSERT INTO detections (camera_id, bbox, center_point, timestamp, embedding_meta_id) VALUES (?, ?, ?, ?, ?)",
# #         (camera_id, bbox_str, center_str, timestamp, emb_meta_id)
# #     )
# #     detection_id = cur.lastrowid

# #     return {
# #         "customer_id": customer_id,
# #         "is_new_customer": is_new_customer,
# #         "embedding_stored": embedding_stored,
# #         "detection_id": detection_id
# #     }
















































import os
import sqlite3
import threading
import sqlite_vec
import numpy as np
from loguru import logger
from datetime import datetime
from src.core.config import settings
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple
# from src.core.database import embedding_cache, last_seen_time, last_seen_camera, last_seen_center


class EmbeddingCache:
    def __init__(self):
        self.embeddings = np.empty((0, settings.EMBEDDING_DIM), dtype=np.float32)
        self.customer_ids = np.empty((0,), dtype=np.int64)
        self._lock = threading.Lock()   # if you ever go multi‑threaded
        self.box_sizes = np.empty((0, 4), dtype=np.float32)   # (w, h) per embedding


    def load_all(self, conn):
        rows = conn.execute("""
            SELECT e.embedding, m.customer_id, m.bbox_w, m.bbox_h, m.center_x, m.center_y
            FROM embeddings e
            JOIN embedding_meta m ON e.rowid = m.id
        """).fetchall()
        if rows:
            emb_list = [np.frombuffer(r[0], dtype=np.float32) for r in rows]
            self.embeddings = np.vstack(emb_list).astype(np.float32)
            self.customer_ids = np.array([r[1] for r in rows], dtype=np.int64)
            self.box_sizes = np.array([(r[2], r[3], r[4], r[5]) for r in rows], dtype=np.float32)
        else:
            self.embeddings   = np.empty((0, settings.EMBEDDING_DIM), dtype=np.float32)
            self.customer_ids = np.empty((0,), dtype=np.int64)
            self.box_sizes    = np.empty((0, 4), dtype=np.float32)

    def add(self, embedding: np.ndarray, customer_id: int, bbox_w: float, bbox_h: float, center_x:float, center_y:float):
        """Thread‑safe append of a new embedding."""
        embedding = embedding.astype(np.float32).flatten()
        with self._lock:
            if self.embeddings.shape[0] == 0:
                self.embeddings = embedding.reshape(1, -1).astype(np.float32)
                self.customer_ids = np.array([customer_id], dtype=np.int64)
                self.box_sizes = np.array([[bbox_w, bbox_h, center_x, center_y]], dtype=np.float32)
            else:
                self.embeddings = np.vstack([self.embeddings, embedding])
                self.customer_ids = np.append(self.customer_ids, customer_id)
                self.box_sizes = np.vstack([self.box_sizes, [bbox_w, bbox_h, center_x, center_y]])
                
    def get_snapshot(self):
        """Return a consistent (embeddings, customer_ids) tuple."""
        with self._lock:
            # return self.embeddings.copy(), self.customer_ids.copy()
            return self.embeddings, self.customer_ids, self.box_sizes # no copy – read‑only arrays are safe because writer creates new arrays via vstack
        
        
# Global instance – used everywhere
embedding_cache = EmbeddingCache()

# Runtime identity state cache
last_seen_time = {}
last_seen_camera = {}
last_seen_center = {}

MAX_EXEMPLARS_PER_CUSTOMER = 12
SPATIAL_TEMPORAL_PENALTY = 1.8
MAX_PIXEL_SPEED = 250.0


def load_cache(conn):
    embedding_cache.load_all(conn)
    
# def fast_match(
#     query_emb,
#     current_cam_id=None,
#     center_point=None,
#     now=None,
#     bbox_w=None,
#     bbox_h=None,
#     ):

#     embs, cids, box_sizes = embedding_cache.get_snapshot()
#     if embs.shape[0] == 0:
#         return None, float('inf')

#     query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
#     dists = np.linalg.norm(embs - query_emb,axis=1)
#     # Spatial-temporal calibration
#     for idx, cid in enumerate(cids):
#         if cid not in last_seen_time:
#             continue

#         prev_time = last_seen_time[cid]
#         dt = max(now - prev_time, 1e-3)
#         prev_cam = last_seen_camera.get(cid)
#         prev_center = last_seen_center.get(cid)

#         if (
#             prev_cam != current_cam_id
#             and prev_center is not None
#             and center_point is not None
#         ):

#             spatial_dist = np.linalg.norm(
#                 np.array(center_point) - np.array(prev_center))

#             speed = spatial_dist / dt
#             if speed > MAX_PIXEL_SPEED:
#                 dists[idx] *= SPATIAL_TEMPORAL_PENALTY

#          # Size consistency penalty
#         if bbox_w is not None and bbox_h is not None:
#             stored_w, stored_h = box_sizes[idx]
#             if stored_w > 0 and stored_h > 0:
#                 ratio_w = max(bbox_w, stored_w) / min(bbox_w, stored_w)
#                 ratio_h = max(bbox_h, stored_h) / min(bbox_h, stored_h)
#                 if ratio_w > 2.0 or ratio_h > 2.0:   # dimensions differ by more than 2x
#                     dists[idx] *= 1.5   # penalty factor
                    
#     min_idx = np.argmin(dists)
#     min_dist = dists[min_idx]

#     if min_dist < settings.REID_THRESHOLD:
#         customer_id = int(cids[min_idx])
#         last_seen_time[customer_id] = now
#         last_seen_camera[customer_id] = current_cam_id
#         last_seen_center[customer_id] = center_point
#         return customer_id, min_dist

#     return None, min_dist




# def fast_match(
#     query_emb,
#     current_cam_id=None,
#     center_point=None,
#     now=None,
#     bbox_w=None,
#     bbox_h=None,
# ):
#     embs, cids, box_sizes = embedding_cache.get_snapshot()
#     if embs.shape[0] == 0:
#         return None, float('inf')

#     query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
#     raw_dists = np.linalg.norm(embs - query_emb, axis=1)   # (N,)

#     # --- Compute spatial cost for every candidate ---
#     spatial_cost = np.zeros_like(raw_dists)
#     if current_cam_id is not None and center_point is not None and now is not None:
#         for idx, cid in enumerate(cids):
#             if cid not in last_seen_time:
#                 continue
#             prev_time = last_seen_time[cid]
#             dt = max(now - prev_time, 1e-3)
#             prev_cam = last_seen_camera.get(cid)
#             prev_center = last_seen_center.get(cid)
#             if (prev_cam != current_cam_id
#                 and prev_center is not None
#                 and center_point is not None):
#                 spatial_dist = np.linalg.norm(
#                     np.array(center_point) - np.array(prev_center))
#                 speed = spatial_dist / dt
#                 if speed > SOFT_SPEED_THRESHOLD:       # e.g., 300 px/s
#                     excess = speed - SOFT_SPEED_THRESHOLD
#                     spatial_cost[idx] = LAMBDA_SPATIAL * excess   # LAMBDA ~ 0.001

#     # --- Size consistency gate (still binary – identity breaker) ---
#     size_penalty_mask = np.zeros(len(cids), dtype=bool)
#     if bbox_w is not None and bbox_h is not None:
#         for idx in range(len(cids)):
#             stored_w, stored_h = box_sizes[idx]
#             if stored_w > 0 and stored_h > 0:
#                 ratio_w = max(bbox_w, stored_w) / min(bbox_w, stored_w)
#                 ratio_h = max(bbox_h, stored_h) / min(bbox_h, stored_h)
#                 if ratio_w > 2.0 or ratio_h > 2.0:
#                     size_penalty_mask[idx] = True

#     # --- Combined score ---
#     total_scores = raw_dists + spatial_cost
#     # Sort by total_score
#     sorted_indices = np.argsort(total_scores)

#     for idx in sorted_indices:
#         cid = cids[idx]
#         # Skip size‑mismatch candidates entirely
#         if size_penalty_mask[idx]:
#             continue

#         final_score = total_scores[idx]
#         if final_score < settings.REID_THRESHOLD:
#             last_seen_time[cid] = now
#             last_seen_camera[cid] = current_cam_id
#             last_seen_center[cid] = center_point
#             return int(cid), final_score
#         else:
#             # Best valid candidate still too far → no match
#             return None, final_score

#     return None, float('inf')


def fast_match(
    query_emb,
    current_cam_id=None,
    center_point=None,
    now=None,
    bbox_w=None,
    bbox_h=None,
):
    embs, cids, box_sizes = embedding_cache.get_snapshot()
    N = embs.shape[0]
    if N == 0:
        return None, float('inf')

    # ── 1. Embedding distance (already vectorized) ──────────────────────────
    query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    raw_dists = np.linalg.norm(embs - query_emb, axis=1)          # (N,)

    # ── 2. Spatial cost — fully vectorized ──────────────────────────────────
    spatial_cost = np.zeros(N, dtype=np.float32)

    if current_cam_id is not None and center_point is not None and now is not None:

        # Build lookup arrays aligned with cids (one value per embedding row).
        # dict.get is called in a list comprehension — still O(N) but in C,
        # not a Python for-loop with branch logic inside.
        ### Insteading of looping through all the Ids, and check lots of conditions, 
        ### we use numpy arrays with default values to get the arrays that really sent values
        prev_times   = np.fromiter(
            (last_seen_time.get(int(c), now) for c in cids),
            dtype=np.float64, count=N)                             # (N,)

        prev_cams    = np.fromiter(
            (last_seen_camera.get(int(c), current_cam_id) for c in cids),
            dtype=np.int64, count=N)                               # (N,)

        # Centers: shape (N, 2). Use (0,0) sentinel for "never seen".
        sentinel = np.array([0.0, 0.0], dtype=np.float32)
        prev_centers = np.stack(
            [last_seen_center.get(int(c), sentinel) for c in cids]
        )                                                          # (N, 2)

        # Masks
        has_history   = prev_times < now                           # seen before
        diff_cam_mask = prev_cams != current_cam_id                # different camera
        apply_mask    = has_history & diff_cam_mask                # (N,) bool

        if np.any(apply_mask):
            dt = np.maximum(now - prev_times[apply_mask], 1e-3)    # (M,)

            cp_arr = np.array(center_point, dtype=np.float32)
            spatial_dist = np.linalg.norm(
                prev_centers[apply_mask] - cp_arr, axis=1)         # (M,)

            # Speed penalty
            speed          = spatial_dist / dt
            excess         = np.maximum(speed - settings.SOFT_SPEED_THRESHOLD, 0.0)
            speed_penalty  = settings.LAMBDA_SPATIAL * excess               # (M,)

            # Distance penalty (time-weighted)
            time_weight    = 1.0 / (dt + 1.0)
            dist_penalty   = settings.LAMBDA_DISTANCE * spatial_dist * time_weight

            spatial_cost[apply_mask] = speed_penalty + dist_penalty

    # ── 3. Size consistency gate — vectorized ───────────────────────────────
    size_penalty_mask = np.zeros(N, dtype=bool)

    if bbox_w is not None and bbox_h is not None and box_sizes.shape[1] >= 2:
        stored_w = box_sizes[:, 0]                                 # (N,)
        stored_h = box_sizes[:, 1]                                 # (N,)

        valid = (stored_w > 0) & (stored_h > 0)                   # (N,) bool

        # Safe ratio: max/min, but only where valid
        ratio_w = np.ones(N, dtype=np.float32)
        ratio_h = np.ones(N, dtype=np.float32)

        ratio_w[valid] = np.maximum(bbox_w, stored_w[valid]) / np.minimum(bbox_w, stored_w[valid])
        ratio_h[valid] = np.maximum(bbox_h, stored_h[valid]) / np.minimum(bbox_h, stored_h[valid])

        size_penalty_mask = valid & ((ratio_w > settings.SIZE_RATIO_GATE) | (ratio_h > settings.SIZE_RATIO_GATE))

    # ── 4. Combine and select ────────────────────────────────────────────────
    total_scores = raw_dists + spatial_cost                        # (N,)

    # Mask out size-rejected candidates with inf so argsort pushes them last
    total_scores_gated = np.where(size_penalty_mask, np.inf, total_scores)
    sorted_indices     = np.argsort(total_scores_gated)

    best_idx   = sorted_indices[0]
    best_score = total_scores_gated[best_idx]

    if best_score < settings.REID_THRESHOLD:
        cid = int(cids[best_idx])
        last_seen_time[cid]   = now
        last_seen_camera[cid] = current_cam_id
        last_seen_center[cid] = np.array(center_point, dtype=np.float32)
        return cid, float(best_score)

    return None, float(best_score)

def fast_min_dist_to_customer(query_emb, customer_id):
    embs, cids, box_sizes = embedding_cache.get_snapshot()
    mask = cids == customer_id
    if not np.any(mask):
        return float('inf')
    dists = np.linalg.norm(embs[mask] - query_emb, axis=1)
    return float(dists.min())

def get_connection(readonly: bool = False) -> sqlite3.Connection:
    """
    Returns a connection with the sqlite-vec extension loaded.
    Use `readonly=True` for read‑only dashboard queries
    (allows concurrent reads in WAL mode).
    """
    os.makedirs(os.path.dirname(settings.DB_PATH), exist_ok=True)
    if readonly:
        uri = f"file:{settings.DB_PATH}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=10)
    else:
        conn = sqlite3.connect(settings.DB_PATH, check_same_thread=False, timeout=10)

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA mmap_size=268435456")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout = 5000")   # 5 seconds – more than enough
    return conn

@contextmanager
def write_connection():
    """Context manager for the single writer connection."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

@contextmanager
def read_connection():
    """Context manager for concurrent readers."""
    conn = get_connection(readonly=True)
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    """
    Creates all tables if they don't exist.
    Run once at application startup.
    """
    with write_connection() as conn:
        # --- User & camera tables (already exist in your code) ---
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cameras (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                stream_url TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        # --- Vector store for Re‑ID ---
        # The vec0 virtual table holds the actual embedding arrays.
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS embeddings USING vec0(
                embedding float[{settings.EMBEDDING_DIM}]
            )
        """)

        # --- Metadata linked to each embedding ---
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embedding_meta (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            camera_id INTEGER NOT NULL,

            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,

            center_x REAL DEFAULT 0,
            center_y REAL DEFAULT 0,
            bbox_w REAL DEFAULT 0,
            bbox_h REAL DEFAULT 0,

            quality_score REAL DEFAULT 0,

            track_id INTEGER DEFAULT -1,

            FOREIGN KEY (customer_id) REFERENCES customers(id),
            FOREIGN KEY (camera_id) REFERENCES cameras(id)
            )
        """)

        # --- Unified customer / person identity ---
        conn.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                total_visits INTEGER DEFAULT 1
            )
        """)
        # This records every detection with its bounding box, timestamp, and a pointer to the embedding that was (optionally) stored.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id INTEGER NOT NULL,
                bbox TEXT NOT NULL,               -- stored as "x1,y1,x2,y2"
                center_point TEXT NOT NULL,       -- stored as "x,y"
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                embedding_meta_id INTEGER,
                FOREIGN KEY (camera_id) REFERENCES cameras(id),
                FOREIGN KEY (embedding_meta_id) REFERENCES embedding_meta(id)
            )
        """)
    logger.info("Initializing the Database...")

def store_embedding(
    conn,
    customer_id,
    camera_id,
    embedding,
    timestamp=None,
    center_point=None,
    bbox_w=0.0, 
    bbox_h=0.0,
    quality_score=0.0,
    track_id=-1
):
    if timestamp is None:
        timestamp = datetime.now()

    # Insert into vec0 table
    cursor = conn.execute(
        "INSERT INTO embeddings (embedding) VALUES (?)",
        (embedding.astype(np.float32).tobytes(),)
    )
    emb_rowid = cursor.lastrowid

    # Insert meta
    if center_point is None:
        center_x, center_y = 0.0, 0.0
    else:
        center_x, center_y = center_point
    
    cursor = conn.execute(
        '''
        INSERT INTO embedding_meta (
            id,
            customer_id,
            camera_id,
            timestamp,
            center_x,
            center_y,
            bbox_w,
            bbox_h,
            quality_score,
            track_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        (
            emb_rowid,
            customer_id,
            camera_id,
            timestamp,
            float(center_x),
            float(center_y),
            float(bbox_w),
            float(bbox_h),
            float(quality_score),
            int(track_id)
        )
    )
    meta_id = cursor.lastrowid

    # ---- Keep the in‑memory cache instantly up‑to‑date ----
    embedding_cache.add(embedding, customer_id, bbox_w, bbox_h, center_x, center_y)   # ← thread‑safe, handles empty cache
    
    ########################### THIS is the rolling cluster.    
    # Keep only best exemplars
    rows = conn.execute(
        """
        SELECT id, quality_score
        FROM embedding_meta
        WHERE customer_id = ?
        ORDER BY quality_score DESC
        """,
        (customer_id,)).fetchall()

    if len(rows) > MAX_EXEMPLARS_PER_CUSTOMER:
        delete_ids = [r["id"] for r in rows[MAX_EXEMPLARS_PER_CUSTOMER:]]
        for did in delete_ids:
            conn.execute("DELETE FROM embeddings WHERE rowid = ?", (did,))
            conn.execute("DELETE FROM embedding_meta WHERE id = ?",(did,))
    return meta_id

def find_matching_person(conn: sqlite3.Connection,
                         query_embedding: np.ndarray) -> Optional[int]:
    q = query_embedding.astype(np.float32).tobytes()
    sql = """
        SELECT
            embeddings.rowid,
            distance,
            embedding_meta.customer_id
        FROM embeddings
        JOIN embedding_meta ON embeddings.rowid = embedding_meta.id
        WHERE embedding MATCH ? AND k = 1
    """
    result = conn.execute(sql, (q,)).fetchone()
    if result and result['distance'] < settings.REID_THRESHOLD:
        return result['customer_id']
    return None

def get_customer_analytics(conn: sqlite3.Connection,
                        customer_id: int) -> Dict:
    """Quick summary for dashboard."""
    info = conn.execute(
        "SELECT * FROM customers WHERE id=?", (customer_id,)
    ).fetchone()
    visits = conn.execute(
        "SELECT COUNT(*) as count FROM visits WHERE customer_id=?",
        (customer_id,)
    ).fetchone()
    return {**dict(info), 'total_visits': visits['count']}

def purge_old_embeddings(conn: sqlite3.Connection, days=30):
    """Deletes embeddings older than `days` to keep the search set tight."""
    cutoff = datetime.now() - datetime.timedelta(days=days)
    conn.execute(
        "DELETE FROM embeddings WHERE rowid IN ("
        "  SELECT id FROM embedding_meta WHERE timestamp < ?"
        ")", (cutoff,)
    )
    conn.execute(
        "DELETE FROM embedding_meta WHERE timestamp < ?", (cutoff,)
    )

# This tells you how similar the new embedding is to what you already have.
# If the distance is large (e.g. > 0.3), the new pose adds valuable diversity.
def min_distance_to_customer(conn: sqlite3.Connection,
                             customer_id: int,
                             query_embedding: np.ndarray) -> float:
    """
    Returns the minimum Euclidean distance between `query_embedding`
    and any existing embedding of the given customer.
    Returns float('inf') if no embeddings exist.
    """
    q = query_embedding.astype(np.float32).tobytes()
    sql = f"""
        SELECT MIN(distance) as min_dist
        FROM (
            SELECT distance, embedding_meta.customer_id
            FROM embeddings
            JOIN embedding_meta ON embeddings.rowid = embedding_meta.id
            WHERE embedding MATCH ? AND k = {settings.KNN_N}
        )
        WHERE customer_id = ?
    """
    row = conn.execute(sql, (q, customer_id)).fetchone()
    if row and row['min_dist'] is not None:
        return row['min_dist']
    return float('inf')
