"""
DB Writer process — PostgreSQL version.

Single process that owns all database writes for the entire application.
Reads commands from db_queue and routes replies to per-camera response_queues.

Queue contract (unchanged from SQLite version):
    update_customer_last_seen → (typ, customer_id, now)
    store_embedding           → (typ, customer_id, cam_id, emb, now,
                                  center_point, bbox_w, bbox_h)
    min_dist_to_customer      → (typ, emb, customer_id, request_id, cam_id)
    match_or_register         → (typ, emb, cam_id, now, request_id,
                                  center_point, bbox_w, bbox_h, track_id,
                                  quality_score)
"""

from loguru import logger

from src.core.database import (
    write_connection,
    fast_match,
    store_embedding,
    fast_min_dist_to_customer,
)


def start_db_writer(ctx, db_queue, response_queues: dict):
    """
    Spawn the db_writer_worker subprocess.

    Args:
        ctx:             multiprocessing context (spawn).
        db_queue:        Queue that pipeline workers put commands into.
        response_queues: dict[cam_id -> Queue] — pre-built at startup,
                         passed by inheritance at spawn time (not in messages).

    Returns:
        The started Process object.
    """
    p = ctx.Process(
        target=db_writer_worker,
        args=(db_queue, response_queues),
        daemon=True,
        name="db_writer",
    )
    p.start()
    logger.info(f"DB writer process started | pid={p.pid}")
    return p


def db_writer_worker(db_queue, response_queues: dict) -> None:
    """
    Main loop of the db_writer subprocess.

    Runs until it receives a None sentinel on db_queue.
    Each command is processed in its own transaction — a failure in one
    command does not affect subsequent commands.
    """
    logger.info("DB writer worker started")

    while True:
        cmd = db_queue.get()

        if cmd is None:
            logger.info("DB writer received shutdown signal")
            break

        typ = cmd[0]

        try:
            _dispatch(typ, cmd, response_queues)
        except Exception as e:
            logger.error(
                f"DB writer error | cmd={typ} | error={e}",
                exc_info=True,
            )

    logger.info("DB writer worker exiting")


def _dispatch(typ: str, cmd: tuple, response_queues: dict) -> None:
    """Route one command to the correct handler."""

    if typ == "update_customer_last_seen":
        _, customer_id, now = cmd
        logger.debug(f"DB writer: update_customer_last_seen {customer_id}")
        with write_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE customers SET last_seen = %s WHERE id = %s",
                    (now, customer_id),
                )

    elif typ == "store_embedding":
        _, customer_id, cam_id, emb, now, center_point, bbox_w, bbox_h = cmd
        logger.debug(f"DB writer: store_embedding for customer {customer_id}, cam {cam_id}")
        with write_connection() as conn:
            # Camera may have been deleted while its pipeline was still running
            # (between DB delete and the next apply_changes() teardown cycle).
            # Skip silently — the FK violation error is avoided and the pipeline
            # will be stopped cleanly on the next /monitor visit.
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM cameras WHERE id = %s", (cam_id,))
                if cur.fetchone() is None:
                    logger.debug(
                        f"store_embedding skipped — camera {cam_id} no longer in DB"
                    )
                    return
            store_embedding(
                conn,
                customer_id=customer_id,
                camera_id=cam_id,
                embedding=emb,
                timestamp=now,
                center_point=center_point,
                bbox_w=bbox_w,
                bbox_h=bbox_h,
            )

    elif typ == "min_dist_to_customer":
        _, emb, customer_id, request_id, cam_id = cmd
        dist        = fast_min_dist_to_customer(emb, customer_id)
        reply_queue = response_queues[cam_id]
        reply_queue.put((request_id, dist))

    elif typ == "match_or_register":
        _, emb, cam_id, now, request_id, center_point, \
            bbox_w, bbox_h, track_id, quality_score = cmd
        logger.debug(f"DB writer: match_or_register for cam {cam_id}, tracker {track_id}")

        matched_cust, dist = fast_match(
            emb,
            current_cam_id=cam_id,
            center_point=center_point,
            now=now,
            bbox_w=bbox_w,
            bbox_h=bbox_h,
        )

        with write_connection() as conn:
            # Same guard as store_embedding — camera may be gone between
            # DB delete and pipeline teardown. Return a safe reply so the
            # embedder worker doesn't block forever on response_queue.get().
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM cameras WHERE id = %s", (cam_id,))
                if cur.fetchone() is None:
                    logger.debug(
                        f"match_or_register skipped — camera {cam_id} no longer in DB"
                    )
                    reply_queue = response_queues.get(cam_id)
                    if reply_queue:
                        reply_queue.put((request_id, None, False, None))
                    return

            if matched_cust is not None:
                customer_id = matched_cust
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE customers SET last_seen = %s WHERE id = %s",
                        (now, customer_id),
                    )
                is_new = False
            else:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO customers (first_seen, last_seen)
                        VALUES (%s, %s)
                        RETURNING id
                        """,
                        (now, now),
                    )
                    customer_id = cur.fetchone()["id"]

                store_embedding(
                    conn,
                    customer_id=customer_id,
                    camera_id=cam_id,
                    embedding=emb,
                    timestamp=now,
                    center_point=center_point,
                    bbox_w=bbox_w,
                    bbox_h=bbox_h,
                    quality_score=quality_score,
                    track_id=track_id,
                )
                is_new = True

        reply_queue = response_queues[cam_id]
        reply_queue.put((request_id, customer_id, is_new, dist))

    else:
        logger.warning(f"DB writer received unknown command type: {typ!r}")