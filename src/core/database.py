"""
Database layer — PostgreSQL + pgvector.

Architecture decisions:
  - Connection pooling via psycopg2.pool.ThreadedConnectionPool
    One pool shared by the db_writer process and the FastAPI process.
    Each process creates its own pool at startup — pools are not shared
    across OS processes (they can't be; sockets are per-process).

  - EmbeddingCache stays 100% numpy — zero DB dependency.
    fast_match() and fast_min_dist_to_customer() run entirely in-process
    against the numpy cache. pgvector is used for:
      (a) Populating the cache at startup via load_cache()
      (b) Persistent storage of all embeddings
      (c) Dashboard / analytics queries (find_matching_person, etc.)

  - Placeholder syntax: PostgreSQL uses %s (not ? like SQLite).

  - Row access: RealDictCursor returns dict-like rows (same as sqlite3.Row).

  - pgvector storage: embedding column is type vector(DIM).
    Insert with list, query with <-> (cosine) or <#> (inner product).

  - SERIAL / BIGSERIAL replaces SQLite's INTEGER PRIMARY KEY AUTOINCREMENT.
    lastrowid → cursor.fetchone()["id"] after RETURNING id.
"""

import os
import threading
import numpy as np
from datetime import datetime
from contextlib import contextmanager
from typing import Optional

import psycopg2
import psycopg2.pool
import psycopg2.extras
from loguru import logger

# pgvector type adapter for psycopg2 ─────────────────────────────────────────
# Without this, psycopg2 returns vector columns as raw strings like
# '[0.1,0.2,...]' instead of Python lists, causing the ValueError in load_cache.
# We register a lightweight string-parser rather than pulling in the pgvector
# Python package, so there is no extra dependency.
def _parse_vector(s, cur=None):   # cur is passed by psycopg2 but unused
    """Convert a pgvector string '[x,y,...]' to a Python list of floats."""
    if s is None:
        return None
    return [float(v) for v in s.strip("[]").split(",")]

def _register_vector_type(conn) -> None:
    """
    Register the pgvector OID and type adapter on *conn* so that every
    SELECT returns Python lists instead of raw strings.

    Called once per new connection, right after getconn().
    Gracefully skips if the extension hasn't been created yet (init_db() handles that).
    """
    with conn.cursor() as cur:
        cur.execute("SELECT oid FROM pg_type WHERE typname = 'vector'")
        row = cur.fetchone()
    if row is None:
        # Extension not created yet — init_db() will create it.
        # Return silently so the first connection during startup doesn't crash.
        return
    oid = row[0] if isinstance(row, (list, tuple)) else row["oid"]
    vector_type = psycopg2.extensions.new_type(
        (oid,), "VECTOR", _parse_vector
    )
    psycopg2.extensions.register_type(vector_type, conn)

from src.core.config import settings
from src.core.exceptions import DatabaseError

# ─────────────────────────────────────────────────────────────────────────────
# Connection Pool
# ─────────────────────────────────────────────────────────────────────────────

# Module-level pool — created once per process by _get_pool() on first use.
_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
_pool_lock = threading.Lock()


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    """
    Return the process-level connection pool, creating it if necessary.

    ThreadedConnectionPool is safe to share across threads within one process.
    It is NOT safe to share across OS processes — each spawned subprocess
    (db_writer_worker, FastAPI) calls this independently and gets its own pool.
    """
    global _pool
    if _pool is not None:
        return _pool

    with _pool_lock:
        if _pool is not None:     # double-checked locking
            return _pool

        dsn = (
            f"host={settings.POSTGRES_HOST} "
            f"port={settings.POSTGRES_PORT} "
            f"dbname={settings.POSTGRES_DB} "
            f"user={settings.POSTGRES_USER} "
            f"password={settings.POSTGRES_PASSWORD} "
            f"application_name=unbreakable_eye "
            f"connect_timeout=10"
        )
        try:
            _pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=getattr(settings, "POSTGRES_POOL_MIN", 2),
                maxconn=getattr(settings, "POSTGRES_POOL_MAX", 10),
                dsn=dsn,
            )
            logger.info(
                f"PostgreSQL pool created | "
                f"host={settings.POSTGRES_HOST} "
                f"db={settings.POSTGRES_DB} "
                f"pool={getattr(settings, 'POSTGRES_POOL_MIN', 2)}-"
                f"{getattr(settings, 'POSTGRES_POOL_MAX', 10)}"
            )
        except psycopg2.Error as e:
            raise DatabaseError(
                "Failed to create PostgreSQL connection pool",
                context={
                    "host": settings.POSTGRES_HOST,
                    "db":   settings.POSTGRES_DB,
                    "error": str(e),
                },
            ) from e

    return _pool


@contextmanager
def get_connection():
    """
    Yield a psycopg2 connection from the pool with RealDictCursor factory.

    Usage:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT ...")

    The connection is automatically returned to the pool on exit.
    Transactions are NOT auto-committed — caller must call conn.commit()
    or rely on write_connection() which commits/rolls back automatically.
    """
    pool = _get_pool()
    conn = pool.getconn()
    conn.cursor_factory = psycopg2.extras.RealDictCursor
    _register_vector_type(conn)
    try:
        yield conn
    finally:
        pool.putconn(conn)


@contextmanager
def write_connection():
    """
    Connection context manager that commits on success, rolls back on error.

    Use for INSERT / UPDATE / DELETE operations.
    """
    with get_connection() as conn:
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

@contextmanager
def read_connection():
    """
    Connection context manager for read-only queries.

    Identical to get_connection() but signals intent clearly.
    In the future this can be pointed at a read replica.
    """
    with get_connection() as conn:
        yield conn


# ─────────────────────────────────────────────────────────────────────────────
# Schema Initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """
    Create all tables and the pgvector extension if they don't exist.

    Safe to call on every startup — all statements use IF NOT EXISTS.
    Requires the connected PostgreSQL user to have CREATE privilege.
    """
    dim = settings.EMBEDDING_DIM

    with write_connection() as conn:
        with conn.cursor() as cur:

            # pgvector extension — must be done before any vector columns
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id            BIGSERIAL PRIMARY KEY,
                    username      TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS cameras (
                    id         BIGSERIAL PRIMARY KEY,
                    user_id    BIGINT NOT NULL REFERENCES users(id),
                    name       TEXT NOT NULL,
                    stream_url TEXT NOT NULL
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS customers (
                    id           BIGSERIAL PRIMARY KEY,
                    first_seen   DOUBLE PRECISION NOT NULL,
                    last_seen    DOUBLE PRECISION NOT NULL,
                    total_visits INTEGER DEFAULT 1
                )
            """)

            # Main embedding store — vector column for pgvector
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id          BIGSERIAL PRIMARY KEY,
                    customer_id BIGINT NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
                    camera_id   BIGINT NOT NULL REFERENCES cameras(id)   ON DELETE CASCADE,
                    embedding   vector({dim}) NOT NULL,
                    timestamp   DOUBLE PRECISION NOT NULL,
                    center_x    REAL DEFAULT 0,
                    center_y    REAL DEFAULT 0,
                    bbox_w      REAL DEFAULT 0,
                    bbox_h      REAL DEFAULT 0,
                    quality_score REAL DEFAULT 0,
                    track_id    INTEGER DEFAULT -1
                )
            """)

            # HNSW index — fast approximate nearest-neighbour search at query time.
            # cosine distance (<->) is correct for L2-normalised Re-ID embeddings.
            # Build after the table exists; IF NOT EXISTS prevents duplicate creation.
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw
                ON embeddings
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """)

            # Index for per-customer embedding lookups (rolling eviction, analytics)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_customer_id
                ON embeddings (customer_id)
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id              BIGSERIAL PRIMARY KEY,
                    camera_id       BIGINT NOT NULL REFERENCES cameras(id),
                    bbox            TEXT NOT NULL,
                    center_point    TEXT NOT NULL,
                    timestamp       DOUBLE PRECISION NOT NULL,
                    embedding_id    BIGINT REFERENCES embeddings(id)
                )
            """)

            # Zone definitions per camera
            cur.execute("""
                CREATE TABLE IF NOT EXISTS zones (
                    id          BIGSERIAL PRIMARY KEY,
                    camera_id   BIGINT NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
                    name        TEXT NOT NULL,
                    polygon     JSONB NOT NULL,
                    zone_type   TEXT DEFAULT 'area',
                    color       TEXT DEFAULT '#4f8cff',
                    created_at  DOUBLE PRECISION NOT NULL
                )
            """)

            # Zone entry/exit events
            cur.execute("""
                CREATE TABLE IF NOT EXISTS zone_events (
                    id            BIGSERIAL PRIMARY KEY,
                    zone_id       BIGINT NOT NULL REFERENCES zones(id) ON DELETE CASCADE,
                    camera_id     BIGINT NOT NULL REFERENCES cameras(id),
                    customer_id   BIGINT REFERENCES customers(id),
                    tracker_id    INTEGER,
                    event_type    TEXT NOT NULL,
                    timestamp     DOUBLE PRECISION NOT NULL,
                    dwell_seconds REAL DEFAULT 0
                )
            """)

            # Aggregated analytics (hourly snapshots per camera)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS analytics_hourly (
                    id               BIGSERIAL PRIMARY KEY,
                    camera_id        BIGINT NOT NULL REFERENCES cameras(id),
                    hour_bucket      DOUBLE PRECISION NOT NULL,
                    unique_visitors  INTEGER DEFAULT 0,
                    total_detections INTEGER DEFAULT 0,
                    avg_dwell_secs   REAL DEFAULT 0,
                    peak_occupancy   INTEGER DEFAULT 0,
                    new_visitors     INTEGER DEFAULT 0,
                    return_visitors  INTEGER DEFAULT 0
                )
            """)

            # Raw detection events (retained for RAW_RETENTION_DAYS)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS detection_events (
                    id          BIGSERIAL PRIMARY KEY,
                    camera_id   BIGINT NOT NULL REFERENCES cameras(id),
                    tracker_id  INTEGER,
                    customer_id BIGINT REFERENCES customers(id),
                    timestamp   DOUBLE PRECISION NOT NULL,
                    bbox_x1     REAL,
                    bbox_y1     REAL,
                    bbox_x2     REAL,
                    bbox_y2     REAL,
                    confidence  REAL,
                    center_x    REAL,
                    center_y    REAL,
                    zone_id     BIGINT REFERENCES zones(id),
                    velocity_x  REAL DEFAULT 0,
                    velocity_y  REAL DEFAULT 0
                )
            """)

            # Indexes for analytics queries
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_zone_events_camera_time
                ON zone_events (camera_id, timestamp)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_detection_events_camera_time
                ON detection_events (camera_id, timestamp)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_analytics_hourly_bucket
                ON analytics_hourly (camera_id, hour_bucket)
            """)

            # Upgrade FK constraints to ON DELETE CASCADE for proper camera deletion
            # These ALTER TABLE statements are safe to run on every startup
            _cascade_fks = [
                ("detections", "detections_camera_id_fkey"),
                ("zone_events", "zone_events_camera_id_fkey"),
                ("analytics_hourly", "analytics_hourly_camera_id_fkey"),
                ("detection_events", "detection_events_camera_id_fkey"),
            ]
            for table, fk_name in _cascade_fks:
                try:
                    cur.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {fk_name}")
                    cur.execute(
                        f"ALTER TABLE {table} ADD CONSTRAINT {fk_name} "
                        f"FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE CASCADE"
                    )
                except Exception:
                    pass  # constraint may not exist yet or naming differs

    logger.info("Database schema initialised (PostgreSQL + pgvector)")


# ─────────────────────────────────────────────────────────────────────────────
# In-Memory Embedding Cache
# ─────────────────────────────────────────────────────────────────────────────
# This is unchanged from SQLite — it's pure numpy with no DB dependency.
# fast_match() runs entirely against this cache; the DB is only touched for
# persistence and startup hydration.

MAX_EXEMPLARS_PER_CUSTOMER = 12
SPATIAL_TEMPORAL_PENALTY   = 1.8
MAX_PIXEL_SPEED            = 250.0


class EmbeddingCache:
    def __init__(self):
        self.embeddings   = np.empty((0, settings.EMBEDDING_DIM), dtype=np.float32)
        self.customer_ids = np.empty((0,),                        dtype=np.int64)
        self.box_sizes    = np.empty((0, 4),                      dtype=np.float32)
        self._lock        = threading.Lock()

    def load_all(self, conn) -> None:
        """
        Hydrate the cache from the database at startup.
        Must be called once in the db_writer process after init_db().
        """
        with conn.cursor() as cur:
            cur.execute("""
                SELECT embedding, customer_id, bbox_w, bbox_h, center_x, center_y
                FROM embeddings
                ORDER BY id ASC
            """)
            rows = cur.fetchall()

        if rows:
            # pgvector returns embeddings as Python lists — convert to numpy
            emb_list = [np.array(r["embedding"], dtype=np.float32) for r in rows]
            self.embeddings   = np.vstack(emb_list).astype(np.float32)
            self.customer_ids = np.array([r["customer_id"] for r in rows], dtype=np.int64)
            self.box_sizes    = np.array(
                [(r["bbox_w"], r["bbox_h"], r["center_x"], r["center_y"]) for r in rows],
                dtype=np.float32
            )
            logger.info(f"Embedding cache loaded: {len(rows)} embeddings")
        else:
            self.embeddings   = np.empty((0, settings.EMBEDDING_DIM), dtype=np.float32)
            self.customer_ids = np.empty((0,),                        dtype=np.int64)
            self.box_sizes    = np.empty((0, 4),                      dtype=np.float32)
            logger.info("Embedding cache empty — no embeddings in DB yet")

    def add(
        self,
        embedding:  np.ndarray,
        customer_id: int,
        bbox_w:     float,
        bbox_h:     float,
        center_x:   float,
        center_y:   float,
    ) -> None:
        """Thread-safe append of a new embedding row."""
        emb = embedding.astype(np.float32).flatten()
        with self._lock:
            if self.embeddings.shape[0] == 0:
                self.embeddings   = emb.reshape(1, -1)
                self.customer_ids = np.array([customer_id], dtype=np.int64)
                self.box_sizes    = np.array([[bbox_w, bbox_h, center_x, center_y]],
                                            dtype=np.float32)
            else:
                self.embeddings   = np.vstack([self.embeddings, emb])
                self.customer_ids = np.append(self.customer_ids, customer_id)
                self.box_sizes    = np.vstack(
                    [self.box_sizes, [bbox_w, bbox_h, center_x, center_y]]
                )

    def get_snapshot(self):
        """Return (embeddings, customer_ids, box_sizes) — consistent view."""
        with self._lock:
            return self.embeddings, self.customer_ids, self.box_sizes


# Module-level singletons — unchanged from SQLite version
embedding_cache  = EmbeddingCache()
last_seen_time   = {}
last_seen_camera = {}
last_seen_center = {}


def load_cache(conn) -> None:
    embedding_cache.load_all(conn)


# ─────────────────────────────────────────────────────────────────────────────
# Real-Time Matching — unchanged from SQLite (pure numpy, no DB)
# ─────────────────────────────────────────────────────────────────────────────

def fast_match(
    query_emb,
    current_cam_id = None,
    center_point   = None,
    now            = None,
    bbox_w         = None,
    bbox_h         = None,
):
    """
    Match a query embedding against the in-memory cache.

    Identical logic to the SQLite version — this function never touches the DB.
    pgvector is not used here because the numpy cache is faster than any
    network round-trip and the matching runs at 30+ fps per camera.
    """
    embs, cids, box_sizes = embedding_cache.get_snapshot()
    N = embs.shape[0]
    if N == 0:
        return None, float("inf")

    # ── 1. Embedding distance ─────────────────────────────────────────────────
    query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    raw_dists = np.linalg.norm(embs - query_emb, axis=1)  # (N,)

    # ── 2. Spatial-temporal cost ──────────────────────────────────────────────
    spatial_cost = np.zeros(N, dtype=np.float32)

    if current_cam_id is not None and center_point is not None and now is not None:
        prev_times = np.fromiter(
            (last_seen_time.get(int(c), now) for c in cids),
            dtype=np.float64, count=N,
        )
        prev_cams = np.fromiter(
            (last_seen_camera.get(int(c), current_cam_id) for c in cids),
            dtype=np.int64, count=N,
        )
        sentinel    = np.array([0.0, 0.0], dtype=np.float32)
        prev_centers = np.stack(
            [last_seen_center.get(int(c), sentinel) for c in cids]
        )

        has_history  = prev_times < now
        diff_cam     = prev_cams != current_cam_id
        apply_mask   = has_history & diff_cam

        if np.any(apply_mask):
            dt = np.maximum(now - prev_times[apply_mask], 1e-3)
            cp = np.array(center_point, dtype=np.float32)
            sd = np.linalg.norm(prev_centers[apply_mask] - cp, axis=1)

            excess        = np.maximum(sd / dt - settings.SOFT_SPEED_THRESHOLD, 0.0)
            speed_penalty = settings.LAMBDA_SPATIAL * excess
            time_weight   = 1.0 / (dt + 1.0)
            dist_penalty  = settings.LAMBDA_DISTANCE * sd * time_weight
            spatial_cost[apply_mask] = speed_penalty + dist_penalty

    # ── 3. Size consistency gate ──────────────────────────────────────────────
    size_penalty_mask = np.zeros(N, dtype=bool)

    if bbox_w is not None and bbox_h is not None and box_sizes.shape[1] >= 2:
        stored_w = box_sizes[:, 0]
        stored_h = box_sizes[:, 1]
        valid    = (stored_w > 0) & (stored_h > 0)

        ratio_w = np.ones(N, dtype=np.float32)
        ratio_h = np.ones(N, dtype=np.float32)
        ratio_w[valid] = (
            np.maximum(bbox_w, stored_w[valid]) / np.minimum(bbox_w, stored_w[valid])
        )
        ratio_h[valid] = (
            np.maximum(bbox_h, stored_h[valid]) / np.minimum(bbox_h, stored_h[valid])
        )
        gate = settings.SIZE_RATIO_GATE
        size_penalty_mask = valid & ((ratio_w > gate) | (ratio_h > gate))

    # ── 4. Select best candidate ──────────────────────────────────────────────
    total = raw_dists + spatial_cost
    total_gated = np.where(size_penalty_mask, np.inf, total)
    best_idx    = np.argmin(total_gated)
    best_score  = float(total_gated[best_idx])

    if best_score < settings.REID_THRESHOLD:
        cid = int(cids[best_idx])
        last_seen_time[cid]   = now
        last_seen_camera[cid] = current_cam_id
        last_seen_center[cid] = np.array(center_point, dtype=np.float32)
        return cid, best_score

    return None, best_score


def fast_min_dist_to_customer(query_emb: np.ndarray, customer_id: int) -> float:
    """Minimum L2 distance from query_emb to any stored embedding for customer_id."""
    embs, cids, _ = embedding_cache.get_snapshot()
    mask = cids == customer_id
    if not np.any(mask):
        return float("inf")
    return float(np.linalg.norm(embs[mask] - query_emb, axis=1).min())


# ─────────────────────────────────────────────────────────────────────────────
# Persistence — store_embedding
# ─────────────────────────────────────────────────────────────────────────────

def store_embedding(
    conn,
    customer_id:   int,
    camera_id:     int,
    embedding:     np.ndarray,
    timestamp:     float  = None,
    center_point          = None,
    bbox_w:        float  = 0.0,
    bbox_h:        float  = 0.0,
    quality_score: float  = 0.0,
    track_id:      int    = -1,
) -> int:
    """
    Persist one embedding to PostgreSQL and update the in-memory cache.

    Args:
        conn:          Active psycopg2 connection (write_connection context).
        customer_id:   Customer this embedding belongs to.
        camera_id:     Camera that captured the detection.
        embedding:     (EMBEDDING_DIM,) float32 numpy array.
        timestamp:     Unix timestamp (float). Defaults to now.
        center_point:  (cx, cy) of the bounding box in original image pixels.
        bbox_w/bbox_h: Bounding box dimensions for size-consistency gating.
        quality_score: Detection confidence or crop quality estimate.
        track_id:      ByteTrack tracker ID for this detection.

    Returns:
        Database row ID of the inserted embedding.
    """
    if timestamp is None:
        timestamp = datetime.now().timestamp()

    center_x, center_y = (center_point if center_point is not None else (0.0, 0.0))

    # pgvector accepts Python lists directly — no tobytes() needed
    emb_list = embedding.astype(np.float32).tolist()

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO embeddings
                (customer_id, camera_id, embedding, timestamp,
                 center_x, center_y, bbox_w, bbox_h, quality_score, track_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                customer_id, camera_id, emb_list, timestamp,
                float(center_x), float(center_y),
                float(bbox_w), float(bbox_h),
                float(quality_score), int(track_id),
            ),
        )
        embedding_id = cur.fetchone()["id"]

    # Keep in-memory cache in sync immediately — no separate DB read needed
    embedding_cache.add(embedding, customer_id, bbox_w, bbox_h, center_x, center_y)

    # Rolling exemplar eviction — keep only top MAX_EXEMPLARS per customer
    _evict_excess_embeddings(conn, customer_id)

    return embedding_id


def _evict_excess_embeddings(conn, customer_id: int) -> None:
    """
    Delete embeddings beyond MAX_EXEMPLARS_PER_CUSTOMER for this customer,
    keeping the highest quality_score ones.

    Called inside store_embedding() — conn is already in a transaction.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id FROM embeddings
            WHERE customer_id = %s
            ORDER BY quality_score DESC
            """,
            (customer_id,),
        )
        rows = cur.fetchall()

    if len(rows) <= MAX_EXEMPLARS_PER_CUSTOMER:
        return

    delete_ids = [r["id"] for r in rows[MAX_EXEMPLARS_PER_CUSTOMER:]]
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM embeddings WHERE id = ANY(%s)",
            (delete_ids,),
        )
    logger.debug(
        f"Evicted {len(delete_ids)} embeddings for customer {customer_id} "
        f"(kept {MAX_EXEMPLARS_PER_CUSTOMER})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Analytics queries — use pgvector for KNN search
# ─────────────────────────────────────────────────────────────────────────────

def find_matching_person(
    conn,
    query_embedding: np.ndarray,
    top_k: int = 5,
) -> Optional[int]:
    """
    Find the closest customer to query_embedding using pgvector HNSW index.

    Used by analytics and dashboard endpoints — NOT used in the real-time
    pipeline (fast_match() handles that via the in-memory cache).

    Args:
        conn:            Read connection.
        query_embedding: (EMBEDDING_DIM,) float32 array.
        top_k:           Number of nearest neighbours to consider.

    Returns:
        customer_id of the best match, or None if no match above threshold.
    """
    emb_list = query_embedding.astype(np.float32).tolist()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT customer_id,
                   embedding <-> %s::vector AS distance
            FROM embeddings
            ORDER BY embedding <-> %s::vector
            LIMIT %s
            """,
            (emb_list, emb_list, top_k),
        )
        rows = cur.fetchall()

    if not rows:
        return None

    best = rows[0]
    if best["distance"] < settings.REID_THRESHOLD:
        return best["customer_id"]
    return None


def get_customer_analytics(conn, cam_id: int, since: float) -> list:
    """Return recent customer activity for a camera since a Unix timestamp."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id, c.first_seen, c.last_seen, c.total_visits,
                   COUNT(e.id) AS embedding_count
            FROM customers c
            JOIN embeddings e ON e.customer_id = c.id
            WHERE e.camera_id = %s AND c.last_seen > %s
            GROUP BY c.id, c.first_seen, c.last_seen, c.total_visits
            ORDER BY c.last_seen DESC
            """,
            (cam_id, since),
        )
        return [dict(r) for r in cur.fetchall()]


def purge_old_embeddings(conn, days: int = 30) -> int:
    """Delete embeddings older than *days* days. Returns count deleted."""
    cutoff = datetime.now().timestamp() - days * 86_400
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM embeddings WHERE timestamp < %s",
            (cutoff,),
        )
        deleted = cur.rowcount
    logger.info(f"Purged {deleted} embeddings older than {days} days")
    return deleted


# ─────────────────────────────────────────────────────────────────────────────
# Analytics retention
# ─────────────────────────────────────────────────────────────────────────────

def purge_old_detection_events(conn, days: int = 7) -> int:
    """Delete raw detection events older than *days* days."""
    cutoff = datetime.now().timestamp() - days * 86_400
    with conn.cursor() as cur:
        cur.execute("DELETE FROM detection_events WHERE timestamp < %s", (cutoff,))
        deleted = cur.rowcount
    if deleted:
        logger.info(f"Purged {deleted} detection events older than {days} days")
    return deleted


def purge_old_analytics(conn, days: int = 30) -> int:
    """Delete aggregated analytics older than *days* days."""
    cutoff = datetime.now().timestamp() - days * 86_400
    with conn.cursor() as cur:
        cur.execute("DELETE FROM analytics_hourly WHERE hour_bucket < %s", (cutoff,))
        deleted = cur.rowcount
    if deleted:
        logger.info(f"Purged {deleted} analytics_hourly rows older than {days} days")
    return deleted


def purge_old_zone_events(conn, days: int = 30) -> int:
    """Delete zone events older than *days* days."""
    cutoff = datetime.now().timestamp() - days * 86_400
    with conn.cursor() as cur:
        cur.execute("DELETE FROM zone_events WHERE timestamp < %s", (cutoff,))
        deleted = cur.rowcount
    if deleted:
        logger.info(f"Purged {deleted} zone events older than {days} days")
    return deleted


# ─────────────────────────────────────────────────────────────────────────────
# Zone CRUD
# ─────────────────────────────────────────────────────────────────────────────

def get_zones_for_camera(conn, camera_id: int) -> list:
    """Return all zones for a camera."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, camera_id, name, polygon, zone_type, color FROM zones WHERE camera_id = %s ORDER BY id",
            (camera_id,),
        )
        return [dict(r) for r in cur.fetchall()]


def create_zone(conn, camera_id: int, name: str, polygon: list, zone_type: str = "area", color: str = "#4f8cff") -> int:
    """Insert a new zone and return its id."""
    import json
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO zones (camera_id, name, polygon, zone_type, color, created_at)
               VALUES (%s, %s, %s::jsonb, %s, %s, %s) RETURNING id""",
            (camera_id, name, json.dumps(polygon), zone_type, color, datetime.now().timestamp()),
        )
        return cur.fetchone()["id"]


def delete_zone(conn, zone_id: int) -> bool:
    """Delete a zone by id. Returns True if it existed."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM zones WHERE id = %s", (zone_id,))
        return cur.rowcount > 0


# ─────────────────────────────────────────────────────────────────────────────
# Analytics queries (implementing the stubs in analysis.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_footfall(conn, camera_id: int, interval: str = "hour", days: int = 7) -> list:
    """Time-bucketed footfall counts from analytics_hourly."""
    cutoff = datetime.now().timestamp() - days * 86_400
    with conn.cursor() as cur:
        cur.execute(
            """SELECT hour_bucket, unique_visitors, total_detections
               FROM analytics_hourly
               WHERE camera_id = %s AND hour_bucket > %s
               ORDER BY hour_bucket""",
            (camera_id, cutoff),
        )
        return [dict(r) for r in cur.fetchall()]


def get_peak_hours(conn, camera_id: int) -> list:
    """Aggregate hourly visitors into a 7x24 grid (day-of-week x hour)."""
    with conn.cursor() as cur:
        cur.execute(
            """SELECT EXTRACT(DOW FROM TO_TIMESTAMP(hour_bucket)) AS dow,
                      EXTRACT(HOUR FROM TO_TIMESTAMP(hour_bucket)) AS hour,
                      SUM(unique_visitors) AS total
               FROM analytics_hourly
               WHERE camera_id = %s
               GROUP BY dow, hour
               ORDER BY dow, hour""",
            (camera_id,),
        )
        return [dict(r) for r in cur.fetchall()]


def get_zone_stats(conn, camera_id: int) -> list:
    """Per-zone entry/exit/dwell metrics."""
    with conn.cursor() as cur:
        cur.execute(
            """SELECT z.id, z.name, z.zone_type, z.color,
                      COUNT(ze.id) FILTER (WHERE ze.event_type='enter') AS entries,
                      COUNT(ze.id) FILTER (WHERE ze.event_type='exit') AS exits,
                      AVG(ze.dwell_seconds) FILTER (WHERE ze.event_type='exit') AS avg_dwell,
                      COUNT(DISTINCT ze.customer_id) AS unique_visitors
               FROM zones z
               LEFT JOIN zone_events ze ON ze.zone_id = z.id
               WHERE z.camera_id = %s
               GROUP BY z.id, z.name, z.zone_type, z.color
               ORDER BY z.id""",
            (camera_id,),
        )
        return [dict(r) for r in cur.fetchall()]


def get_repeat_visitors(conn, camera_id: int, days: int = 30) -> dict:
    """New vs return visitor counts."""
    cutoff = datetime.now().timestamp() - days * 86_400
    with conn.cursor() as cur:
        cur.execute(
            """SELECT
                 COUNT(DISTINCT e.customer_id) FILTER (WHERE c.total_visits = 1) AS new_visitors,
                 COUNT(DISTINCT e.customer_id) FILTER (WHERE c.total_visits > 1) AS return_visitors
               FROM embeddings e
               JOIN customers c ON c.id = e.customer_id
               WHERE e.camera_id = %s AND e.timestamp > %s""",
            (camera_id, cutoff),
        )
        row = cur.fetchone()
        return dict(row) if row else {"new_visitors": 0, "return_visitors": 0}


def min_distance_to_customer(
    conn,
    customer_id:     int,
    query_embedding: np.ndarray,
) -> float:
    """
    Minimum cosine distance between query_embedding and all stored exemplars
    for customer_id, using pgvector — for analytics use only.
    """
    emb_list = query_embedding.astype(np.float32).tolist()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT MIN(embedding <-> %s::vector) AS min_dist
            FROM embeddings
            WHERE customer_id = %s
            """,
            (emb_list, customer_id),
        )
        row = cur.fetchone()
    return float(row["min_dist"]) if row and row["min_dist"] is not None else float("inf")


# ─────────────────────────────────────────────────────────────────────────────
# Telegram report helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_summary_report(conn, camera_id: int) -> dict:
    """Build summary dict for Telegram daily/weekly reports."""
    import time as _time
    from datetime import datetime
    now = _time.time()
    # Use LOCAL timezone for "today" boundary, not UTC
    local_now = datetime.now()
    local_midnight = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_start = local_midnight.timestamp()
    week_start = today_start - 7 * 86400

    with conn.cursor() as cur:
        cur.execute("""
            SELECT COALESCE(SUM(unique_visitors), 0) AS visitors,
                   COALESCE(SUM(total_detections), 0) AS detections
            FROM analytics_hourly
            WHERE camera_id = %s AND hour_bucket >= %s
        """, (camera_id, today_start))
        today = dict(cur.fetchone())

        # Fallback: if analytics_hourly has no today data, count from detection_events
        if today["visitors"] == 0:
            cur.execute("""
                SELECT COUNT(DISTINCT tracker_id) AS visitors,
                       COUNT(*) AS detections
                FROM detection_events
                WHERE camera_id = %s AND timestamp >= %s
            """, (camera_id, today_start))
            fallback = dict(cur.fetchone())
            if fallback["visitors"] > 0:
                today = fallback

        cur.execute("""
            SELECT COUNT(DISTINCT tracker_id) AS cnt
            FROM detection_events
            WHERE camera_id = %s AND timestamp > %s
        """, (camera_id, now - 300))
        active_now = cur.fetchone()["cnt"]

        cur.execute("""
            SELECT COALESCE(SUM(unique_visitors), 0) AS visitors,
                   COALESCE(SUM(total_detections), 0) AS detections,
                   COALESCE(AVG(avg_dwell_secs), 0) AS avg_dwell,
                   COALESCE(MAX(peak_occupancy), 0) AS max_occupancy
            FROM analytics_hourly
            WHERE camera_id = %s AND hour_bucket >= %s
        """, (camera_id, week_start))
        week = dict(cur.fetchone())

    return {
        "today": today,
        "week": week,
        "active_now": active_now,
        "zones": get_zone_stats(conn, camera_id),
        "peak_hours": get_peak_hours(conn, camera_id),
    }


def get_all_cameras(conn) -> list:
    """Return all cameras with IDs and names."""
    with conn.cursor() as cur:
        cur.execute("SELECT id, name FROM cameras ORDER BY id")
        return [dict(r) for r in cur.fetchall()]


def get_multi_camera_summary(conn) -> dict:
    """Build per-camera + totals summary for Telegram reports.

    Returns:
        {
            "cameras": [{"id", "name", "today", "week", "active_now", "zones"}, ...],
            "totals": {"today": {...}, "week": {...}, "active_now": int}
        }
    """
    cameras = get_all_cameras(conn)
    if not cameras:
        return {"cameras": [], "totals": {"today": {}, "week": {}, "active_now": 0}}

    result_cameras = []
    totals_today = {"visitors": 0, "detections": 0}
    totals_week = {"visitors": 0, "detections": 0, "avg_dwell": 0, "max_occupancy": 0}
    total_active = 0
    week_count = 0

    for cam in cameras:
        summary = get_summary_report(conn, cam["id"])
        result_cameras.append({
            "id": cam["id"],
            "name": cam["name"],
            "today": summary.get("today", {}),
            "week": summary.get("week", {}),
            "active_now": summary.get("active_now", 0),
            "zones": summary.get("zones", []),
        })
        totals_today["visitors"] += summary.get("today", {}).get("visitors", 0) or 0
        totals_today["detections"] += summary.get("today", {}).get("detections", 0) or 0
        totals_week["visitors"] += summary.get("week", {}).get("visitors", 0) or 0
        totals_week["detections"] += summary.get("week", {}).get("detections", 0) or 0
        totals_week["max_occupancy"] = max(
            totals_week["max_occupancy"],
            summary.get("week", {}).get("max_occupancy", 0) or 0,
        )
        avg_d = summary.get("week", {}).get("avg_dwell", 0) or 0
        if avg_d > 0:
            totals_week["avg_dwell"] += avg_d
            week_count += 1
        total_active += summary.get("active_now", 0) or 0

    if week_count > 0:
        totals_week["avg_dwell"] = totals_week["avg_dwell"] / week_count

    return {
        "cameras": result_cameras,
        "totals": {"today": totals_today, "week": totals_week, "active_now": total_active},
    }


def get_heatmap_history(conn, camera_id: int, hours: int = 24) -> list:
    """Get detection center points for heatmap generation."""
    import time as _time
    cutoff = _time.time() - hours * 3600
    with conn.cursor() as cur:
        cur.execute("""
            SELECT center_x, center_y, COUNT(*) AS weight
            FROM detection_events
            WHERE camera_id = %s AND timestamp > %s
              AND center_x IS NOT NULL AND center_y IS NOT NULL
            GROUP BY center_x, center_y
        """, (camera_id, cutoff))
        return [dict(r) for r in cur.fetchall()]    