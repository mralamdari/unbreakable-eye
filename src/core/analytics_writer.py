"""
Analytics writer process — batches and persists analytics events.

Receives events from the pipeline via analytics_queue:
  - ("detection", {...})     — per-detection metadata
  - ("zone_event", {...})    — zone entry/exit events

Responsibilities:
  1. Batch INSERT detection_events (high volume)
  2. INSERT zone_events immediately (low volume)
  3. Aggregate into analytics_hourly periodically
  4. Run retention cleanup (raw 7 days, aggregate 30 days)
"""

import time
import math
from datetime import datetime
from loguru import logger

from src.core.database import (
    write_connection,
    purge_old_detection_events,
    purge_old_analytics,
    purge_old_zone_events,
)


def start_analytics_writer(ctx, analytics_queue):
    """Spawn the analytics_writer_worker subprocess."""
    p = ctx.Process(
        target=analytics_writer_worker,
        args=(analytics_queue,),
        daemon=True,
        name="analytics_writer",
    )
    p.start()
    logger.info(f"Analytics writer process started | pid={p.pid}")
    return p


def analytics_writer_worker(analytics_queue) -> None:
    """Main loop — batches detection events, flushes periodically."""
    from src.core.config import settings

    batch = []
    last_flush = time.time()
    last_cleanup = time.time()
    last_aggregation = time.time()
    CLEANUP_INTERVAL = 3600.0  # run retention every hour
    AGGREGATION_INTERVAL = 300.0  # run aggregation every 5 minutes (was 1 hour for faster testing)

    logger.info("Analytics writer worker started")

    # Run aggregation on startup to catch any missed hours
    try:
        _aggregate_hourly()
    except Exception as e:
        logger.error(f"Startup aggregation failed: {e}", exc_info=True)

    events_received = 0
    while True:
        try:
            cmd = analytics_queue.get(timeout=1.0)
        except Exception:
            # Timeout — flush if batch is non-empty
            cmd = None

        if cmd is None:
            now = time.time()
            if batch and (now - last_flush >= settings.ANALYTICS_FLUSH_INTERVAL):
                try:
                    _flush_batch(batch)
                    batch.clear()
                    last_flush = now
                except Exception as e:
                    logger.error(f"Flush failed (timeout path): {e}", exc_info=True)

            # Periodic retention cleanup
            if now - last_cleanup >= CLEANUP_INTERVAL:
                _run_retention()
                last_cleanup = now

            # Periodic hourly aggregation
            if now - last_aggregation >= AGGREGATION_INTERVAL:
                logger.info(f"Analytics stats: {events_received} events received, {len(batch)} in batch")
                _aggregate_hourly()
                last_aggregation = now
            continue

        if cmd == "SHUTDOWN":
            logger.info("Analytics writer received shutdown signal")
            break

        typ = cmd[0]

        try:
            if typ == "detection":
                batch.append(cmd[1])
                events_received += 1
                if len(batch) >= settings.ANALYTICS_BATCH_SIZE:
                    _flush_batch(batch)
                    batch.clear()
                    last_flush = time.time()

            elif typ == "zone_event":
                _write_zone_event(cmd[1])

        except Exception as e:
            logger.error(f"Analytics writer error | cmd={typ} | error={e}", exc_info=True)

    # Final flush
    if batch:
        try:
            _flush_batch(batch)
        except Exception as e:
            logger.error(f"Final flush failed: {e}", exc_info=True)

    logger.info("Analytics writer worker exiting")


def _aggregate_hourly() -> None:
    """Aggregate detection_events and zone_events into analytics_hourly."""
    now = time.time()
    hour_start = math.floor(now / 3600) * 3600 - 3600
    hour_end = hour_start + 3600

    try:
        with write_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM cameras")
                cameras = [r["id"] for r in cur.fetchall()]

                for cam_id in cameras:
                    # Skip if already aggregated
                    cur.execute(
                        "SELECT 1 FROM analytics_hourly WHERE camera_id = %s AND hour_bucket = %s",
                        (cam_id, hour_start),
                    )
                    if cur.fetchone():
                        continue

                    # Unique visitors
                    cur.execute("""
                        SELECT COUNT(DISTINCT tracker_id) AS uv
                        FROM detection_events
                        WHERE camera_id = %s AND timestamp >= %s AND timestamp < %s
                    """, (cam_id, hour_start, hour_end))
                    uv = cur.fetchone()["uv"] or 0

                    # Total detections
                    cur.execute("""
                        SELECT COUNT(*) AS td
                        FROM detection_events
                        WHERE camera_id = %s AND timestamp >= %s AND timestamp < %s
                    """, (cam_id, hour_start, hour_end))
                    td = cur.fetchone()["td"] or 0

                    # Avg dwell
                    cur.execute("""
                        SELECT AVG(dwell_seconds) AS ad
                        FROM zone_events
                        WHERE camera_id = %s AND event_type = 'exit'
                          AND timestamp >= %s AND timestamp < %s AND dwell_seconds > 0
                    """, (cam_id, hour_start, hour_end))
                    ad = cur.fetchone()["ad"] or 0

                    # Peak occupancy
                    cur.execute("""
                        SELECT MAX(cnt) AS po FROM (
                            SELECT COUNT(DISTINCT tracker_id) AS cnt
                            FROM detection_events
                            WHERE camera_id = %s AND timestamp >= %s AND timestamp < %s
                            GROUP BY FLOOR(timestamp)
                        ) sub
                    """, (cam_id, hour_start, hour_end))
                    po = cur.fetchone()["po"] or 0

                    # New vs returning
                    cur.execute("""
                        SELECT
                            COUNT(DISTINCT de.customer_id) FILTER (
                                WHERE c.total_visits = 1 OR c.total_visits IS NULL
                            ) AS nv,
                            COUNT(DISTINCT de.customer_id) FILTER (
                                WHERE c.total_visits > 1
                            ) AS rv
                        FROM detection_events de
                        LEFT JOIN customers c ON c.id = de.customer_id
                        WHERE de.camera_id = %s AND de.timestamp >= %s AND de.timestamp < %s
                    """, (cam_id, hour_start, hour_end))
                    vr = cur.fetchone()
                    nv = vr["nv"] or 0
                    rv = vr["rv"] or 0

                    cur.execute("""
                        INSERT INTO analytics_hourly
                        (camera_id, hour_bucket, unique_visitors, total_detections,
                         avg_dwell_secs, peak_occupancy, new_visitors, return_visitors)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (cam_id, hour_start, uv, td, ad, po, nv, rv))

                logger.info(f"Hourly aggregation complete | hour={datetime.fromtimestamp(hour_start).isoformat()} | cameras={len(cameras)}")
    except Exception as e:
        logger.error(f"Hourly aggregation failed: {e}", exc_info=True)


def _flush_batch(batch: list) -> None:
    """Bulk INSERT a batch of detection events."""
    if not batch:
        return

    logger.debug(f"Flushing {len(batch)} detection events (cam={batch[0].get('camera_id')})")
    with write_connection() as conn:
        with conn.cursor() as cur:
            args_str = ",".join(
                cur.mogrify(
                    "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (
                        ev["camera_id"],
                        ev.get("tracker_id"),
                        ev.get("customer_id"),
                        ev["timestamp"],
                        ev.get("bbox_x1"), ev.get("bbox_y1"),
                        ev.get("bbox_x2"), ev.get("bbox_y2"),
                        ev.get("confidence"),
                        ev.get("center_x"), ev.get("center_y"),
                        ev.get("zone_id"),
                        ev.get("velocity_x", 0),
                        ev.get("velocity_y", 0),
                    ),
                ).decode()
                for ev in batch
            )
            cur.execute(
                f"""INSERT INTO detection_events
                    (camera_id, tracker_id, customer_id, timestamp,
                     bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence,
                     center_x, center_y, zone_id, velocity_x, velocity_y)
                    VALUES {args_str}"""
            )

    logger.debug(f"Flushed {len(batch)} detection events")
    batch.clear()


def _write_zone_event(ev: dict) -> None:
    """Write a single zone entry/exit event."""
    with write_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO zone_events
                   (zone_id, camera_id, customer_id, tracker_id,
                    event_type, timestamp, dwell_seconds)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (
                    ev["zone_id"],
                    ev["camera_id"],
                    ev.get("customer_id"),
                    ev.get("tracker_id"),
                    ev["event_type"],
                    ev["timestamp"],
                    ev.get("dwell_seconds", 0),
                ),
            )


def _run_retention() -> None:
    """Delete old data according to retention policies."""
    from src.core.config import settings

    with write_connection() as conn:
        purge_old_detection_events(conn, days=settings.RAW_RETENTION_DAYS)
        purge_old_zone_events(conn, days=settings.AGGREGATE_RETENTION_DAYS)
        purge_old_analytics(conn, days=settings.AGGREGATE_RETENTION_DAYS)
