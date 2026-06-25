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
