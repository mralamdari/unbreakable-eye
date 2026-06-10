import os
import numpy as np
from src.core.config import settings
from src.core.database import get_connection, fast_match, store_embedding, fast_min_dist_to_customer


def start_db_writer(ctx, db_queue, response_queues: dict):
    """
    response_queues: dict[cam_id -> mp.Queue]
    Passed at spawn time — queues are inherited, not pickled mid-flight.
    """
    p = ctx.Process(
        target=db_writer_worker,
        args=(db_queue, response_queues),
        daemon=True
    )
    p.start()
    return p


def db_writer_worker(db_queue, response_queues: dict):
    """
    response_queues: dict[cam_id -> Queue]
    db_writer uses cam_id from each message to route replies back.
    """
    conn = get_connection()
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA mmap_size=268435456")
    conn.execute("PRAGMA busy_timeout=5000")

    while True:
        cmd = db_queue.get()
        if cmd is None:
            break
        typ = cmd[0]

        try:
            if typ == "update_customer_last_seen":
                _, customer_id, now = cmd
                conn.execute(
                    "UPDATE customers SET last_seen = ? WHERE id = ?",
                    (now, customer_id))

            elif typ == "store_embedding":
                _, customer_id, cam_id, emb, now, center_point, bbox_w, bbox_h = cmd
                store_embedding(
                    conn, customer_id, cam_id, emb,
                    timestamp=now,
                    center_point=center_point,
                    bbox_w=bbox_w,
                    bbox_h=bbox_h,
                )
                

            elif typ == "min_dist_to_customer":
                # No reply_queue in message — look it up by cam_id
                _, emb, customer_id, request_id, cam_id = cmd
                dist = fast_min_dist_to_customer(emb, customer_id)
                reply_queue = response_queues[cam_id]
                reply_queue.put((request_id, dist))
                continue  # no commit needed

            elif typ == "match_or_register":
                _, emb, cam_id, now, request_id, center_point, \
                    bbox_w, bbox_h, track_id, quality_score = cmd

                matched_cust, dist = fast_match(
                    emb,
                    current_cam_id=cam_id,
                    center_point=center_point,
                    now=now,
                    bbox_w=bbox_w,
                    bbox_h=bbox_h,
                )

                if matched_cust is not None:
                    customer_id = matched_cust
                    conn.execute(
                        "UPDATE customers SET last_seen = ? WHERE id = ?",
                        (now, customer_id))
                    is_new = False
                else:
                    cur = conn.execute(
                        "INSERT INTO customers (first_seen, last_seen) VALUES (?, ?)",
                        (now, now))
                    customer_id = cur.lastrowid
                    store_embedding(
                    conn, customer_id, cam_id, emb,
                    timestamp=now,
                    center_point=center_point,
                    bbox_w=bbox_w,
                    bbox_h=bbox_h,
                )
                    is_new = True

                reply_queue = response_queues[cam_id]
                reply_queue.put((request_id, customer_id, is_new, dist))

            conn.commit()
        except Exception as e:
            print(f"[DB WRITER ERROR] {e}")
            conn.rollback()

    conn.close()