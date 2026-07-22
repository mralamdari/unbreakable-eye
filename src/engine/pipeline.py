import os
import cv2
import time
import queue
import psutil
import functools
import numpy as np
import supervision as sv
from loguru import logger
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
from src.core.config import settings
from src.vision.factory import get_detector
from src.vision.utils import create_session, preprocess_crop
from src.engine.heatmap import HeatmapAccumulator
# fast_min_dist_to_customer is called inside db_writer, not pipeline
from multiprocessing import shared_memory

SKIP_FRAME = (None, None, None, None)

# ─────────────────────────────────────────────────────────────────────────────
# SHARED EMBEDDING WORKER
#
# Single process that owns ONE OSNet session for ALL cameras.
# Each embedder_worker sends crops here, gets embeddings back.
# This eliminates the per-camera ONNX session that was causing OOM crashes.
# ─────────────────────────────────────────────────────────────────────────────
def shared_embedder_worker(
    embed_input_queue,    # mp.Queue: (request_id, cam_id, crops_onnx, crop_meta)
    embed_output_queues,  # dict[cam_id -> mp.Queue]: results back to each embedder
    stop_event,
):
    """
    Single process that owns ONE OSNet session for ALL cameras.
    Receives crops from all embedder_workers, runs batched inference,
    routes embeddings back to each camera's embedder_worker.
    """
    pin_process([4,5])
    embedder_session = create_session(settings.FEATURE_EXTRACTOR_MODEL, num_threads=2)
    logger.info("Shared embedding worker started with single OSNet session")

    while not stop_event.is_set():
        try:
            request_id, cam_id, crops_onnx, crop_meta = embed_input_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if request_id is None:
            break  # shutdown signal

        # Run batched inference on the crops
        if crops_onnx:
            batch_input = np.stack(crops_onnx, axis=0)
            embeddings = embedder_session.run(None, {"input": batch_input})[0]
        else:
            embeddings = np.array([])

        # Send results back to the requesting camera's embedder
        try:
            embed_output_queues[cam_id].put_nowait((request_id, embeddings, crop_meta))
        except queue.Full:
            logger.warning(f"Embed output queue full for camera {cam_id}, dropping results")
            continue

    logger.info("Shared embedding worker stopped")

# ─────────────────────────────────────────────────────────────────────────
# CLIENT-FACING OVERLAY CONFIG
#
# SHOW_DEBUG_INFO=False renders the clean, branded overlay meant for
# clients/demos: no tracker ids, no confidence scores, no raw FPS counter,
# no customer db ids on screen.
#
# SHOW_DEBUG_INFO=True renders the dev/ops view instead: tracker id +
# confidence + customer id + live per-track dwell seconds on every box, plus
# a stats panel (raw FPS, people in frame, cumulative unique visitors,
# active loitering count).
#
# Driven by settings so you can run a dev instance and a client-demo
# instance side by side without editing code — add
# `SHOW_DEBUG_OVERLAY: bool = False` to your Settings class and set
# SHOW_DEBUG_OVERLAY=true in that instance's env/.env. Falls back to the
# clean client view if the setting doesn't exist yet.
# ─────────────────────────────────────────────────────────────────────────
SHOW_DEBUG_INFO = getattr(settings, "SHOW_DEBUG_OVERLAY", False)

# Modern, muted palette (not supervision's default neon) — one color per
# tracked person via color_lookup=TRACK, so two people on screen never look
# identical the way they did with the default class-based coloring.
OVERLAY_PALETTE_HEX = [
    "#2EC4B6",  # teal
    "#5C6BC0",  # indigo
    "#FF7F66",  # coral
    "#66BB6A",  # sage green
    "#AB47BC",  # plum
    "#26C6DA",  # cyan
    "#EC7063",  # rose
    "#42A5F5",  # sky blue
]
LOITER_COLOR_HEX = "#FFB627"   # amber — extended-dwell flag, deliberately not alarming red

# Bundle a real font with the repo — headless Docker images have no system
# fonts, and cv2's built-in Hershey font is what was making labels look
# "robotic"/blurry. Get e.g. Inter or Poppins SemiBold (Google Fonts, free)
# and drop it at this path. If it's missing we fall back to a basic PIL
# bitmap font and log a warning instead of crashing the pipeline.
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "fonts")
FONT_PATH = os.path.join(FONT_DIR, "Inter-SemiBold.ttf")


@functools.lru_cache(maxsize=8)
def _load_font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        logger.warning(
            f"Overlay font not found at {path} — using low-quality fallback font. "
            f"Add a .ttf there for production-quality text rendering."
        )
        return ImageFont.load_default()

def pin_process(cores):
    try:
        psutil.Process(os.getpid()).cpu_affinity(list(cores))
    except Exception:
        pass
    
def frame_view(shm: shared_memory.SharedMemory, frame_shape, frame_bytes: int, idx: int):
    offset = idx * frame_bytes
    return np.ndarray(frame_shape, dtype=np.uint8, buffer=shm.buf, offset=offset)
    
# shared memory write helper
def _write_frame_to_slot(shm, idx, frame, frame_shape, frame_bytes):
    dst = frame_view(shm, frame_shape, frame_bytes, idx)
    if frame.shape != frame_shape:
        # INTER_AREA is the standard choice for shrinking (less aliasing than
        # linear); only matters when frame is larger than the target slot.
        interp = cv2.INTER_AREA if frame.shape[0] > frame_shape[0] else cv2.INTER_LINEAR
        frame = cv2.resize(frame, (frame_shape[1], frame_shape[0]), interpolation=interp)
    np.copyto(dst, frame)


class VisionPipeline:
    def __init__(self,
                RTSP_URL,
                CAM_ID,
                ctx,
                free_slots,
                frame_ready_queue,
                det_queue,
                stop_event,
                db_queue,
                response_queue,
                analytics_queue=None,
                alert_queue=None,
                embed_input_queue=None,
                embed_output_queue=None,
                buffer_slots=4):
                
        #         embedder_worker
        #       │
        #       │  puts messages like:
        #       │  ("match_or_register", emb, cam_id, now, request_id, ...)
        #       │  ("store_embedding", customer_id, ...)
        #       │  ("update_customer_last_seen", customer_id, now)
        #       │
        #       ▼
        #    db_queue  (mp.Queue)
        #       │
        #       ▼
        # db_writer_worker  (separate process)
        #       │
        #       ├── runs fast_match()         → reads embedding_cache (in-memory numpy)
        #       ├── runs store_embedding()    → writes SQLite + updates embedding_cache
        #       ├── runs UPDATE customers     → writes SQLite
        #       │
        #       └── puts reply into reply_queue (the per-camera response_queue)
        #                │
        #                ▼
        #          embedder_worker.response_queue.get()
        #          → gets back (request_id, customer_id, is_new, dist)

        # Reasons to use the db_queue:
        #     Reason 1: SQLite only allows one writer.
        #     Reason 2: Writes are slow, inference can't wait. Pushing to a queue is microseconds.


        
        
        # Queue contracts — do not break these
        # frame_ready_queue : (cam_id: int, idx: int)
        #   reader → batched_detector_worker
        #
        # det_queue         : (idx: int | None, xyxy: np.ndarray | None,
        #                      confidence: np.ndarray | None, class_id: np.ndarray | None)
        #   batched_detector_worker → embedder_worker
        #   None tuple = skipped frame, reuse last detections
        #
        # db_queue          : ("match_or_register", emb, cam_id, now, request_id,
        #                       center_point, bbox_w, bbox_h, track_id, quality_score, reply_queue)
        #                   | ("store_embedding", customer_id, cam_id, emb, now,
        #                       center_point, bbox_w, bbox_h, quality_score, track_id)
        #                   | ("update_customer_last_seen", customer_id, now)
        #                   | ("min_dist_to_customer", emb, customer_id, request_id, reply_queue)
        #   embedder_worker → db_writer_worker
        #
        # response_queue    : (request_id, customer_id, is_new, dist)  ← match_or_register reply
        #                   | (request_id, dist)                        ← min_dist reply
        #   db_writer_worker → embedder_worker (per-camera queue)
        
        # So the flow of a slot index is:
        #     free_slots → reader → frame_ready_queue → detector → det_queue → embedder → free_slots
        #     It's a ring. The slot number travels through the pipeline tracking which memory slot contains the current frame.
        
        
        
        
        
        
        
        
        
        # SLOT RING INVARIANTS — never violate these:
        #
        # 1. Every idx that leaves free_slots MUST eventually return to free_slots.
        #    Leak one slot and the ring drains. After buffer_slots frames, the
        #    reader blocks forever on free_slots.get_nowait().
        #
        # 2. Only real frame indices (integers) enter the slot ring.
        #    None is never a slot index. Skipped frames use SKIP_FRAME sentinel.
        #
        # 3. The embedder is the normal slot returner.
        #    The batched detector is the emergency slot returner (when embedder backed up).
        #    The reader is the emergency slot returner (when detector backed up).
        #    No one else touches free_slots.
        #
        # 4. SKIP_FRAME = (None, None, None, None) never enters frame_ready_queue.
        #    It only goes into det_queue directly from reader_worker.
        self.model_input_size = (256, 128)
        self.ctx = ctx
        self.RTSP_URL = RTSP_URL
        self.cam_id = CAM_ID
        self.torso_ratio = 2/3
        self.free_slots = free_slots
        self.frame_ready_queue = frame_ready_queue
        self.det_queue  = det_queue
        self.stop_event = stop_event
        self.db_queue   = db_queue
        self.response_queue = response_queue
        self.analytics_queue = analytics_queue
        self.alert_queue = alert_queue
        self.buffer_slots = buffer_slots

        # Inference resolution — input ring shared with the detector
        self.frame_shape = settings.FRAME_SHAPE   # e.g. (512, 512, 3)
        self.frame_bytes = settings.FRAME_BYTES

        # Display resolution — output SHM holds one full-res annotated frame
        # The embedder upscales annotated frames to this shape before writing.
        self.display_shape = settings.DISPLAY_SHAPE   # (1080, 1920, 3)
        self.display_bytes = settings.DISPLAY_BYTES   # 1920 * 1080 * 3

        self.online = True   # reader handles offline state

        # INPUT SHM: ring buffer at inference resolution
        self.input_shm = shared_memory.SharedMemory(
            create=True,
            size=self.frame_bytes * buffer_slots)
        self.input_shm_name = self.input_shm.name

        # NATIVE SHM: ring buffer at display resolution, written by the
        # reader from the same cap.read() frame it downsamples for
        # inference. The detector never touches this — only the embedder
        # reads it, to draw on real captured detail instead of upscaling
        # the tiny inference frame. Costs one extra resize in the reader
        # and a bit of shared memory (display_bytes * buffer_slots); zero
        # extra cost to the detector, so it doesn't touch FPS.
        self.native_shm = shared_memory.SharedMemory(
            create=True,
            size=self.display_bytes * buffer_slots)
        self.native_shm_name = self.native_shm.name

        # OUTPUT SHM: single frame at display resolution (1920×1080)
        self.output_shm = shared_memory.SharedMemory(
            create=True,
            size=self.display_bytes)
        self.output_shm_name = self.output_shm.name

        self._shm_cleaned_up = False   # guard: prevent double close/unlink
        for i in range(buffer_slots):
            self.free_slots.put(i)

        # Shared flag: set to 1 by reader_worker the first time a real frame
        # is successfully read from the camera. Readable from the main process
        # via processor.has_frame.value so generate() knows whether to show
        # the offline image or the live stream.
        self.has_frame = self.ctx.Value('b', 0)

        # Shared embedding worker queues
        self.embed_input_queue = embed_input_queue
        self.embed_output_queue = embed_output_queue

    def _cleanup_shm(self) -> None:
        """Close and unlink both SHM blocks exactly once (idempotent)."""
        if self._shm_cleaned_up:
            return
        self._shm_cleaned_up = True
        for shm in (self.input_shm, self.native_shm, self.output_shm):
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except Exception:
                pass

    def __del__(self) -> None:
        """Fallback cleanup so shm is never leaked if stop() was not called."""
        try:
            self._cleanup_shm()
        except Exception:
            pass

    def start(self):
        self.p_embedder = self.ctx.Process(
            target=embedder_worker,
            args=(
                self.input_shm_name,
                self.native_shm_name,
                self.output_shm_name,
                self.frame_shape,
                self.frame_bytes,
                self.display_shape,
                self.display_bytes,
                self.det_queue,
                self.free_slots,
                self.stop_event,
                self.db_queue,
                self.response_queue,
                self.cam_id,
                self.has_frame,
                self.analytics_queue,
                self.alert_queue,
                self.embed_input_queue,
                self.embed_output_queue,
            ),
            daemon=True
        )
        
        self.p_reader = self.ctx.Process(
            target=reader_worker,
            args=(
                self.RTSP_URL,
                self.cam_id,
                self.input_shm_name,
                self.native_shm_name,
                self.frame_shape,
                self.frame_bytes,
                self.display_shape,
                self.display_bytes,
                self.free_slots,
                self.frame_ready_queue,
                self.det_queue,
                self.stop_event,
                self.has_frame,
                self.alert_queue,
            ),
            daemon=True
        )
        self.p_reader.start()
        self.p_embedder.start()

        
    def stop(self):
        # Signal THIS camera's reader and embedder only.
        # stop_event is per-camera — calling this never affects other cameras.
        self.stop_event.set()

        # Unblock embedder_worker if it is waiting on det_queue.get().
        # Do NOT touch frame_ready_queue — it is shared across ALL cameras
        # and is consumed by batched_detector_worker. _teardown_sync() kills
        # the batched_detector first (before calling stop() on any camera)
        # which drains frame_ready_queue automatically.
        try:
            self.det_queue.put_nowait(None)
        except Exception:
            pass

        for proc in (getattr(self, "p_reader", None),
                     getattr(self, "p_embedder", None)):
            if proc is not None and proc.is_alive():
                proc.join(timeout=3)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=3)
                if proc.is_alive():
                    proc.kill()          # SIGKILL — last resort
                    proc.join(timeout=2)

        self._cleanup_shm()

    def get_latest_frame(self):
        """Return a copy of the latest annotated frame at display resolution."""
        frame = np.ndarray(
            self.display_shape,
            dtype=np.uint8,
            buffer=self.output_shm.buf
        )
        return frame.copy()
    
#####################3.2 Loitering detection
def check_loitering(tracker_id,
                    current_time,
                    centroid,
                    track_positions,
                    loiter_threshold_pixels = 50,
                    loiter_time_threshold = 10.0):
    positions = track_positions[tracker_id]
    # remove old entries older than loiter_time_threshold
    while positions and positions[0][0] < current_time - loiter_time_threshold:
        positions.pop(0)
    positions.append((current_time, centroid))
    
    if len(positions) >= 2:
        first_pos = positions[0][1]
        displacement = np.linalg.norm(np.array(centroid) - np.array(first_pos))
        duration = current_time - positions[0][0]
        if displacement < loiter_threshold_pixels and duration >= loiter_time_threshold:
            return True
    return False    

def _pill(draw, xy, radius, fill):
    draw.rounded_rectangle(xy, radius=radius, fill=fill)


def draw_hud(frame_bgr, cam_label, in_store_count, alert_count):
    """
    Client-facing HUD: two small rounded pills instead of raw debug text
    floating at fixed pixel coordinates. Sized relative to frame width so
    it holds up if display resolution ever changes.

    Left pill  : live indicator + camera/store label
    Right pill : how many people are in frame right now
    Below-right: amber alert pill, only shown when someone trips the
                 extended-dwell check (loitering_tracker_ids)
    """
    h, w = frame_bgr.shape[:2]
    font_size = max(16, int(h * 0.02))
    font = _load_font(FONT_PATH, font_size)

    pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    pad_x, pad_y = int(font_size * 0.9), int(font_size * 0.55)
    margin = int(w * 0.015)
    dot_r = max(4, font_size // 5)

    # ── Left pill: live dot + camera label ──
    left_text = cam_label
    tb = draw.textbbox((0, 0), left_text, font=font)
    text_w, text_h = tb[2] - tb[0], tb[3] - tb[1]
    pill_w = pad_x * 3 + dot_r * 2 + text_w
    pill_h = text_h + pad_y * 2
    x0, y0 = margin, margin
    x1, y1 = x0 + pill_w, y0 + pill_h
    _pill(draw, [x0, y0, x1, y1], radius=pill_h // 2, fill=(18, 18, 22, 150))
    dot_cx = x0 + pad_x + dot_r
    dot_cy = y0 + pill_h // 2
    draw.ellipse(
        [dot_cx - dot_r, dot_cy - dot_r, dot_cx + dot_r, dot_cy + dot_r],
        fill=(52, 199, 89, 255),   # green
    )
    draw.text(
        (dot_cx + dot_r + pad_x * 0.6, y0 + pill_h // 2 - text_h // 2 - tb[1]),
        left_text, font=font, fill=(255, 255, 255, 255),
    )

    # ── Right pill: live in-store count ──
    right_text = f"{in_store_count} IN STORE" if in_store_count != 1 else "1 IN STORE"
    tb2 = draw.textbbox((0, 0), right_text, font=font)
    text_w2, text_h2 = tb2[2] - tb2[0], tb2[3] - tb2[1]
    pill_w2 = pad_x * 2 + text_w2
    pill_h2 = text_h2 + pad_y * 2
    rx1 = w - margin
    rx0 = rx1 - pill_w2
    ry0 = margin
    ry1 = ry0 + pill_h2
    _pill(draw, [rx0, ry0, rx1, ry1], radius=pill_h2 // 2, fill=(18, 18, 22, 150))
    draw.text(
        (rx0 + pad_x, ry0 + pill_h2 // 2 - text_h2 // 2 - tb2[1]),
        right_text, font=font, fill=(255, 255, 255, 255),
    )

    # ── Optional alert pill, directly under the count pill ──
    if alert_count > 0:
        alert_text = (
            f"{alert_count} EXTENDED DWELL" if alert_count != 1 else "EXTENDED DWELL"
        )
        tb3 = draw.textbbox((0, 0), alert_text, font=font)
        text_w3, text_h3 = tb3[2] - tb3[0], tb3[3] - tb3[1]
        pill_w3 = pad_x * 2 + text_w3
        pill_h3 = text_h3 + pad_y * 2
        ax1 = w - margin
        ax0 = ax1 - pill_w3
        ay0 = ry1 + int(pad_y * 0.7)
        ay1 = ay0 + pill_h3
        r, g, b = tuple(int(LOITER_COLOR_HEX.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        _pill(draw, [ax0, ay0, ax1, ay1], radius=pill_h3 // 2, fill=(r, g, b, 210))
        draw.text(
            (ax0 + pad_x, ay0 + pill_h3 // 2 - text_h3 // 2 - tb3[1]),
            alert_text, font=font, fill=(20, 20, 20, 255),
        )

    composed = Image.alpha_composite(pil_img, overlay).convert("RGB")
    return cv2.cvtColor(np.array(composed), cv2.COLOR_RGB2BGR)


def draw_debug_hud(frame_bgr, cam_id, fps, in_frame_count, unique_visitor_count, loiter_count):
    """
    Dev/ops stats panel (SHOW_DEBUG_INFO=True) — dense on purpose, this is
    for you, not a client. Per-box tracker id / confidence / customer id /
    dwell seconds are drawn separately by label_annotator (see process_frame
    step 5); this is just the global numbers in one place.
    """
    h, w = frame_bgr.shape[:2]
    font_size = max(16, int(h * 0.02))
    font = _load_font(FONT_PATH, font_size)

    pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    lines = [
        f"CAM {cam_id}  |  FPS {fps:.1f}",
        f"In frame: {in_frame_count}",
        f"Unique visitors (cumulative): {unique_visitor_count}",
        f"Loitering now: {loiter_count}",
    ]

    pad_x, pad_y = int(font_size * 0.8), int(font_size * 0.5)
    line_gap = int(font_size * 0.35)
    margin = int(w * 0.015)

    line_sizes = []
    max_w = 0
    total_h = 0
    for line in lines:
        tb = draw.textbbox((0, 0), line, font=font)
        lw, lh = tb[2] - tb[0], tb[3] - tb[1]
        line_sizes.append((lh, tb))
        max_w = max(max_w, lw)
        total_h += lh + line_gap
    total_h -= line_gap

    panel_w = max_w + pad_x * 2
    panel_h = total_h + pad_y * 2
    x0, y0 = margin, margin
    x1, y1 = x0 + panel_w, y0 + panel_h
    _pill(draw, [x0, y0, x1, y1], radius=int(font_size * 0.4), fill=(15, 15, 18, 165))

    cy = y0 + pad_y
    for line, (lh, tb) in zip(lines, line_sizes):
        draw.text((x0 + pad_x, cy - tb[1]), line, font=font, fill=(255, 255, 255, 255))
        cy += lh + line_gap

    composed = Image.alpha_composite(pil_img, overlay).convert("RGB")
    return cv2.cvtColor(np.array(composed), cv2.COLOR_RGB2BGR)


def process_frame(
    frame,
    detections,
    tracker,
    embedder_session,
    track_to_customer,
    db_queue,
    cam_id,
    box_annotator,
    fps_monitor,
    trace_annotator,
    label_annotator,
    response_queue,
    zones,
    zone_annotators,
    zone_box_annotators,
    dwell_start,
    dwell_total,
    track_positions,
    unique_visitors,
    run_embedding,
    corner_annotator=None,
    loiter_corner_annotator=None,
    loiter_trace_annotator=None,
    cam_label=None,
    display_frame=None,
    scale=(1.0, 1.0),
    person_class_id=0,
    blur_annotator=None,
    heatmap_accumulator=None,
    analytics_queue=None,
    zone_ids=None,
    alert_queue=None,
    alert_limiter=None,
    precomputed_embeddings=None,
    precomputed_crop_meta=None,
):
    # `frame` stays at native inference resolution end-to-end — it's what
    # embedding crops and zone/dwell logic are computed against, exactly as
    # before (zero change to re-id quality). `display_frame`, if given, is
    # the same frame pre-upscaled to display resolution; all DRAWING happens
    # on that, with box coordinates scaled to match, so lines/text render
    # crisp instead of being drawn small and then blown up afterward.
    render_base = display_frame if display_frame is not None else frame
    cam_label = cam_label or f"CAM {cam_id}"

    # ── 0. Guard — nobody in frame ──
    if detections is None or len(detections) == 0:
        fps_monitor.tick()
        if SHOW_DEBUG_INFO:
            annotated = draw_debug_hud(
                render_base.copy(), cam_id, fps_monitor.fps,
                in_frame_count=0, unique_visitor_count=len(unique_visitors), loiter_count=0,
            )
        else:
            annotated = draw_hud(render_base.copy(), cam_label, in_store_count=0, alert_count=0)
        return annotated

    # ── 1. Filter & Track ──
    if detections.class_id is not None:
        detections = detections[detections.class_id == person_class_id]

    detections = tracker.update_with_detections(detections)

    # ── 2. Zone analytics ──
    for zone_idx, (zone, zone_annotator, zone_box_ann) in enumerate(
            zip(zones, zone_annotators, zone_box_annotators)):
        mask = zone.trigger(detections=detections)
        detections_in_zone = detections[mask] if isinstance(mask, np.ndarray) else mask
        inside_now = set(
            tid for tid in detections_in_zone.tracker_id if tid is not None)

        zone_dwell_start = dwell_start.setdefault(zone_idx, {})
        zone_dwell_total = dwell_total.setdefault(zone_idx, defaultdict(float))

        for tid in inside_now:
            if tid not in zone_dwell_start:
                zone_dwell_start[tid] = time.time()
                # Emit zone enter event
                if analytics_queue is not None:
                    try:
                        analytics_queue.put_nowait(("zone_event", {
                            "zone_id": zone_ids[zone_idx] if zone_ids and zone_idx < len(zone_ids) else zone_idx,
                            "camera_id": cam_id,
                            "tracker_id": int(tid),
                            "customer_id": track_to_customer.get(tid),
                            "event_type": "enter",
                            "timestamp": time.time(),
                            "dwell_seconds": 0,
                        }))
                    except Exception:
                        pass
        for tid in list(zone_dwell_start.keys()):
            if tid not in inside_now:
                elapsed = time.time() - zone_dwell_start.pop(tid)
                zone_dwell_total[tid] += elapsed
                # Emit zone exit event
                if analytics_queue is not None:
                    try:
                        analytics_queue.put_nowait(("zone_event", {
                            "zone_id": zone_ids[zone_idx] if zone_ids and zone_idx < len(zone_ids) else zone_idx,
                            "camera_id": cam_id,
                            "tracker_id": int(tid),
                            "customer_id": track_to_customer.get(tid),
                            "event_type": "exit",
                            "timestamp": time.time(),
                            "dwell_seconds": elapsed,
                        }))
                    except Exception:
                        pass
                logger.debug(f"Tracker {tid} left zone {zone_idx} after {elapsed:.1f}s")

        # Dwell-time bookkeeping above always runs — it's real data feeding the
        # HUD alert pill. The visual outline below is skipped in client mode:
        # right now `zones` is a single polygon covering the entire frame, so
        # drawing its border is just a colored rectangle around the whole
        # picture with no informational value. Worth defining real sub-regions
        # (entrance, checkout lanes, etc.) before turning this back on.
        if SHOW_DEBUG_INFO:
            frame = zone_box_ann.annotate(scene=frame, detections=detections_in_zone)
            frame = zone_annotator.annotate(scene=frame)

    # ── 3. Crop collection ──
    crops_onnx, crop_meta = [], []
    loitering_tracker_ids = set()

    # Heatmap: add positions scaled to display resolution
    # Analytics: emit detection events (throttled to ~1/sec per track)
    _last_det_event = getattr(process_frame, "_last_det_event", {})
    now = time.time()
    for det in detections:
        det_box, _, det_conf, _, tracker_id, _ = det
        if tracker_id is None:
            continue
        cx = (det_box[0] + det_box[2]) / 2
        cy = (det_box[1] + det_box[3]) / 2
        if heatmap_accumulator is not None:
            heatmap_accumulator.add_position(cx * scale[0], cy * scale[1], now)
        # Emit detection event at most once per tracker per second
        if analytics_queue is not None and _last_det_event.get(tracker_id, 0) < now - 1.0:
            _last_det_event[tracker_id] = now
            try:
                analytics_queue.put_nowait(("detection", {
                    "camera_id": cam_id,
                    "tracker_id": int(tracker_id),
                    "customer_id": track_to_customer.get(tracker_id),
                    "timestamp": now,
                    "bbox_x1": float(det_box[0]), "bbox_y1": float(det_box[1]),
                    "bbox_x2": float(det_box[2]), "bbox_y2": float(det_box[3]),
                    "confidence": float(det_conf),
                    "center_x": float(cx), "center_y": float(cy),
                    "zone_id": None,
                    "velocity_x": 0, "velocity_y": 0,
                }))
            except Exception:
                pass
    process_frame._last_det_event = _last_det_event

    for det in detections:
        det_box, det_mask, det_conf, class_id, tracker_id, data = det
        if tracker_id is None:
            continue

        cx = (det_box[0] + det_box[2]) / 2
        cy = (det_box[1] + det_box[3]) / 2
        center_point = (cx, cy)

        if check_loitering(tracker_id, time.time(), center_point, track_positions):
            loitering_tracker_ids.add(tracker_id)
            # Emit loitering alert
            if alert_queue is not None and alert_limiter is not None:
                from src.engine.alerts import emit_loitering_alert
                duration = time.time() - track_positions[tracker_id][0][0]
                emit_loitering_alert(
                    alert_queue, cam_id, tracker_id,
                    track_to_customer.get(tracker_id),
                    duration, alert_limiter,
                )

        if run_embedding:
            input_tensor, crop_box, center_point, bbox_w, bbox_h, crop_flag = preprocess_crop(
                frame, det_box,
                model_input_size=(256, 128),
                torso_ratio=1)
            if not crop_flag:
                unique_visitors.add(tracker_id)
                crops_onnx.append(input_tensor)
                crop_meta.append((
                    crop_box, center_point, bbox_w, bbox_h,
                    int(tracker_id), det_conf))

    # ── 4. Embedding & Re-ID ──
    # Build as a dict: tracker_id -> label string
    # This avoids any index mismatch — we join to detections later by tracker_id
    label_map = {}   # tracker_id -> label string

    # Use pre-computed embeddings from shared worker, or compute locally
    if precomputed_embeddings is not None and precomputed_crop_meta is not None:
        embeddings = precomputed_embeddings
        crop_meta = precomputed_crop_meta
    elif crops_onnx and run_embedding and embedder_session is not None:
        batch_input = np.stack(crops_onnx, axis=0)
        embeddings = embedder_session.run(None, {"input": batch_input})[0]
    else:
        embeddings = None

    if embeddings is not None and len(embeddings) > 0:
        for i, emb in enumerate(embeddings):
            emb = emb.flatten()
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            bbox, center_point, bbox_w, bbox_h, tracker_id, confidence_score = crop_meta[i]

            if tracker_id in track_to_customer:
                customer_id = track_to_customer[tracker_id]
                request_id = f"dist_{cam_id}_{tracker_id}_{time.time()}"
                db_queue.put((
                    "min_dist_to_customer",
                    emb, customer_id, request_id, cam_id
                ))
                rid, min_dist = response_queue.get()
                for _ in range(10):
                    if rid == request_id:
                        break
                    logger.warning(f"Stale min_dist response drained: {rid} != {request_id}")
                    rid, min_dist = response_queue.get()

                if min_dist > settings.DIVERSITY_THRESHOLD:
                    db_queue.put(("store_embedding", customer_id, cam_id, emb,
                                  time.time(), center_point, bbox_w, bbox_h))
                else:
                    db_queue.put(("update_customer_last_seen", customer_id, time.time()))
            else:
                request_id = f"match_{cam_id}_{tracker_id}_{time.time()}"
                db_queue.put((
                    "match_or_register",
                    emb, cam_id, time.time(), request_id,
                    center_point, bbox_w, bbox_h, tracker_id, confidence_score
                ))
                # Drain stale responses left in the queue from a previous
                # pipeline run. The assert that was here killed the embedder
                # process on restart, crashing the container with exit 135.
                rid, customer_id, is_new, match_dist = response_queue.get()
                for _ in range(10):
                    if rid == request_id:
                        break
                    logger.warning(f"Stale response drained: {rid} != {request_id}")
                    rid, customer_id, is_new, match_dist = response_queue.get()
                else:
                    logger.error(f"No correct response after draining for {request_id}")
                    continue

                track_to_customer[tracker_id] = customer_id
                if not is_new and match_dist > settings.DIVERSITY_THRESHOLD:
                    db_queue.put(("store_embedding", customer_id, cam_id, emb,
                                  time.time(), center_point, bbox_w, bbox_h))
                else:
                    db_queue.put(("update_customer_last_seen", customer_id, time.time()))

            label_map[tracker_id] = (
                f"#Track:{int(tracker_id)} {confidence_score:.2f} ID:{customer_id}")

    # ── 5. Build labels aligned to detections ──
    # One label per detection, in the same order as detections.
    # This is the ONLY place labels list is built — guarantees len(labels) == len(detections).
    labels = []
    for det in detections:
        _, _, det_conf, _, tracker_id, _ = det

        if tracker_id is None:
            labels.append("?")
            continue

        if tracker_id in label_map:
            # Got a fresh embedding this frame
            labels.append(label_map[tracker_id])
        elif tracker_id in track_to_customer:
            # Known from a previous frame, no new embedding this frame
            customer_id = track_to_customer[tracker_id]
            labels.append(f"#Track:{int(tracker_id)} ID:{customer_id}")
        else:
            # Tracker exists but not yet identified (first appearance, crop was bad)
            labels.append(f"#Track:{int(tracker_id)} ?")

    if SHOW_DEBUG_INFO:
        # Append live dwell time + loiter flag to each label. zone 0 is the
        # only zone right now (see note above about it covering the whole
        # frame) so dwell here effectively means "time visible in frame."
        zone0_dwell_start = dwell_start.get(0, {})
        now_ts = time.time()
        augmented = []
        for det, label in zip(detections, labels):
            _, _, _, _, tracker_id, _ = det
            suffix = ""
            if tracker_id is not None:
                if tracker_id in zone0_dwell_start:
                    suffix += f" {now_ts - zone0_dwell_start[tracker_id]:.0f}s"
                if tracker_id in loitering_tracker_ids:
                    suffix += " LOITER"   # plain ASCII — cv2's Hershey font can't render symbols
            augmented.append(label + suffix)
        labels = augmented

    # ── 6. Annotate ──
    # Build a display-resolution copy of detections for drawing only — crop
    # and zone/dwell logic above already ran against the original native-
    # resolution `detections` and is untouched by this.
    scale_x, scale_y = scale
    if len(detections) > 0 and (scale_x != 1.0 or scale_y != 1.0):
        display_xyxy = detections.xyxy.astype(np.float32) * np.array(
            [scale_x, scale_y, scale_x, scale_y], dtype=np.float32)
    else:
        display_xyxy = detections.xyxy
    display_detections = sv.Detections(
        xyxy=display_xyxy,
        confidence=detections.confidence,
        class_id=detections.class_id,
        tracker_id=detections.tracker_id,
    )

    annotated = render_base.copy()

    if len(detections) > 0:
        if SHOW_DEBUG_INFO:
            # Old ops view: solid boxes + tracker id / confidence / customer id
            annotated = trace_annotator.annotate(annotated, display_detections)
            annotated = box_annotator.annotate(annotated, display_detections)
            annotated = label_annotator.annotate(annotated, display_detections, labels)
        else:
            # Client view: corner-bracket boxes, one color per person via
            # color_lookup=TRACK (set on corner_annotator/trace_annotator in
            # embedder_worker), amber for anyone flagged as loitering. No
            # tracker ids, no confidence scores, no db customer ids on screen.
            tracker_ids = display_detections.tracker_id
            is_loitering = np.array(
                [tid is not None and tid in loitering_tracker_ids for tid in tracker_ids]
            )

            normal_dets = display_detections[~is_loitering]
            loiter_dets = display_detections[is_loitering]

            if len(normal_dets) > 0:
                annotated = trace_annotator.annotate(annotated, normal_dets)
                annotated = corner_annotator.annotate(annotated, normal_dets)
            if len(loiter_dets) > 0:
                annotated = loiter_trace_annotator.annotate(annotated, loiter_dets)
                annotated = loiter_corner_annotator.annotate(annotated, loiter_dets)

    fps_monitor.tick()

    if SHOW_DEBUG_INFO:
        annotated = draw_debug_hud(
            annotated, cam_id, fps_monitor.fps,
            in_frame_count=len(detections),
            unique_visitor_count=len(unique_visitors),
            loiter_count=len(loitering_tracker_ids),
        )
    else:
        annotated = draw_hud(
            annotated,
            cam_label,
            in_store_count=len(detections),
            alert_count=len(loitering_tracker_ids),
        )

    # Heatmap overlay — rendered after HUD, before blur
    if heatmap_accumulator is not None and settings.HEATMAP_ENABLED:
        annotated = heatmap_accumulator.render_overlay(annotated)

    # Privacy blur — applied last so it covers all annotations on person regions
    if blur_annotator is not None and settings.PRIVACY_BLUR and len(display_detections) > 0:
        annotated = blur_annotator.annotate(scene=annotated, detections=display_detections)

    return annotated

# How the backoff works:
# First failure → wait 1 s, then retry.
# Second failure → wait 2 s, then retry.
# Third failure → wait 4 s, then retry.
# … up to a maximum of 60 s.
# As soon as a frame is read successfully, consecutive_failures resets to 0.
# During the wait, the thread is sleeping – it consumes almost zero CPU.
def reader_worker(rtsp_url,
                  cam_id,
                  shm_name,
                  native_shm_name,
                  frame_shape,
                  frame_bytes,
                  display_shape,
                  display_bytes,
                  free_slots,
                  frame_ready_queue,
                  det_queue,
                  stop_event,
                  has_frame,
                  alert_queue=None):
    pin_process([0])
    shm = shared_memory.SharedMemory(name=shm_name)
    native_shm = shared_memory.SharedMemory(name=native_shm_name)

    # frame_shape is (H, W, C) — e.g. (1080, 1920, 3)
    target_h, target_w = frame_shape[0], frame_shape[1]

    cap = None
    first_attempt = True
    online = False
    consecutive_failures = 0
    max_backoff = 60.0
    FREEZE_TIMEOUT = 5.0   # seconds without a new frame → treat as failure
    last_frame_time = time.time()

    def _open_capture(url):
        """Open VideoCapture and request target resolution + minimal buffering."""
        c = cv2.VideoCapture(url)
        if not c.isOpened():
            return None
        # Ask driver for target resolution — note this is frequently a no-op
        # for RTSP streams: OpenCV's FFmpeg backend generally decodes
        # whatever resolution the stream is actually encoded at and just
        # ignores this request (unlike USB/V4L2 cameras, which usually do
        # honor it). Don't assume `frame.shape` below matches (target_h,
        # target_w) — _write_frame_to_slot resizes down regardless, and
        # that's exactly the frame we also capture at full detail for
        # native_shm below.
        c.set(cv2.CAP_PROP_FRAME_WIDTH,  target_w)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
        # Keep OpenCV internal buffer at 1 frame so we always read the newest frame,
        # not a frame that was decoded 200 ms ago sitting in a 10-frame buffer.
        c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return c

    while not stop_event.is_set():
        if not online:
            if cap is not None:
                cap.release()
                cap = None

            if first_attempt and consecutive_failures < 6:
                first_attempt = False
                sleep_time = 0
            else:
                sleep_time = min(1.0 * (2 ** consecutive_failures), max_backoff)

            stop_event.wait(sleep_time)
            if stop_event.is_set():
                break

            cap = _open_capture(rtsp_url)
            if cap is None:
                consecutive_failures += 1
                try:
                    det_queue.put_nowait((None, None, None, None))
                except queue.Full:
                    pass
                continue

            online = True
            consecutive_failures = 0
            last_frame_time = time.time()

        # Stall guard — if cap.read() blocks or the stream freezes, reconnect.
        if time.time() - last_frame_time > FREEZE_TIMEOUT:
            logger.warning(f"Camera {cam_id}: stream frozen for {FREEZE_TIMEOUT}s, reconnecting")
            online = False
            consecutive_failures += 1
            # Emit camera offline alert
            if consecutive_failures == 3 and alert_queue is not None:
                from src.engine.alerts import AlertRateLimiter, emit_camera_offline_alert
                _offline_limiter = AlertRateLimiter(cooldown_seconds=300)
                emit_camera_offline_alert(alert_queue, cam_id, 15, _offline_limiter)
            try:
                det_queue.put_nowait((None, None, None, None))
            except queue.Full:
                pass
            continue

        ret, frame = cap.read()
        if not ret:
            online = False
            consecutive_failures += 1
            try:
                det_queue.put_nowait((None, None, None, None))
            except queue.Full:
                pass
            continue

        last_frame_time = time.time()
        consecutive_failures = 0

        # Signal to generate() in main.py that at least one real frame
        # arrived — switches the MJPEG stream from the offline image to live.
        if not has_frame.value:
            has_frame.value = 1

        # Every decoded frame goes to the detector.
        # If no slot is free, the detector is backed up — drop this frame and
        # keep reading so the reader never blocks the capture loop.
        try:
            idx = free_slots.get_nowait()
        except queue.Empty:
            continue   # drop frame; don't stall

        _write_frame_to_slot(shm, idx, frame, frame_shape, frame_bytes)
        # Same source frame, resized (usually downscaled, since `frame` is
        # most likely the stream's real resolution — see note above) to
        # display resolution instead of inference resolution. This is what
        # lets the embedder draw on real captured detail instead of
        # upscaling the tiny inference copy. Adds one resize + one shm copy
        # here in the reader; costs the detector nothing.
        _write_frame_to_slot(native_shm, idx, frame, display_shape, display_bytes)
        try:
            frame_ready_queue.put_nowait((cam_id, idx))
        except queue.Full:
            # Detector input queue full — return slot immediately.
            free_slots.put_nowait(idx)

    if cap is not None:
        cap.release()
    shm.close()
    native_shm.close()

def embedder_worker(
    input_shm_name,
    native_shm_name,
    output_shm_name,
    frame_shape,
    frame_bytes,
    display_shape,
    display_bytes,
    det_queue,
    free_slots,
    stop_event,
    db_queue,
    response_queue,
    cam_id,
    has_frame,
    analytics_queue=None,
    alert_queue=None,
    embed_input_queue=None,
    embed_output_queue=None,
):
    pin_process([4,5])
    input_shm = shared_memory.SharedMemory(name=input_shm_name)
    native_shm = shared_memory.SharedMemory(name=native_shm_name)
    output_shm = shared_memory.SharedMemory(name=output_shm_name)
    # NOTE: OSNet session is now in shared_embedder_worker — no local session needed
    # TODO: swap for a real store/location name once that's configurable
    # per camera (e.g. settings.CAMERA_LABELS[cam_id]) — "CAM 1" is a
    # placeholder that's still better than nothing on a client-facing feed.
    cam_label = f"CAM {cam_id}"
    track_to_customer = {}
    unique_visitors = set()   # persists across frames — counts unique tracker IDs seen
    track_positions = defaultdict(list)   # local to this process — not shared
    tracker = sv.ByteTrack(lost_track_buffer=120) # 120 frames ==> This stops the tracker from killing a track when a person is briefly occluded or missed by the detector.

    # Heatmap accumulator — one per camera, accumulates person positions over time
    heatmap_accumulator = HeatmapAccumulator(
        width=settings.DISPLAY_SHAPE[1],
        height=settings.DISPLAY_SHAPE[0]
    )
    fps_monitor = sv.FPSMonitor()

    # Line/text weight scaled to display resolution — thickness=1 is fine at
    # 512×512 but reads as a hairline once the frame is at 1920×1080.
    dh_px, dw_px, _ = display_shape
    line_thickness = max(2, round(dh_px * 0.0028))
    corner_len = round(dh_px * 0.025)

    palette = sv.ColorPalette.from_hex(OVERLAY_PALETTE_HEX)
    loiter_color = sv.Color.from_hex(LOITER_COLOR_HEX)

    # Old ops/debug annotators (SHOW_DEBUG_INFO=True) — now colored by track
    # too instead of by class, so even the debug view doesn't paint every
    # person the same magenta.
    color = palette
    box_annotator = sv.BoxAnnotator(color=color, color_lookup=sv.ColorLookup.TRACK,
                                     thickness=line_thickness)
    trace_annotator = sv.TraceAnnotator(color=color, color_lookup=sv.ColorLookup.TRACK,
                                         trace_length=30, thickness=line_thickness)
    label_annotator = sv.LabelAnnotator(color=color, color_lookup=sv.ColorLookup.TRACK,
                                         text_color=sv.Color.BLACK)

    # Client-facing annotators — corner brackets instead of solid boxes read
    # as a modern "tracking" indicator rather than a debug bounding box, and
    # they don't visually collide with each other in crowded frames the way
    # filled/labeled boxes did in your screenshots.
    corner_annotator = sv.BoxCornerAnnotator(
        color=palette, color_lookup=sv.ColorLookup.TRACK,
        thickness=line_thickness, corner_length=corner_len,
    )
    loiter_corner_annotator = sv.BoxCornerAnnotator(
        color=loiter_color, thickness=line_thickness + 1, corner_length=corner_len,
    )
    loiter_trace_annotator = sv.TraceAnnotator(
        color=loiter_color, trace_length=30, thickness=line_thickness,
    )

    # Privacy blur annotator — blurs faces when PRIVACY_BLUR is enabled
    blur_annotator = sv.BlurAnnotator(
        kernel_size=settings.PRIVACY_BLUR_KERNEL if settings.PRIVACY_BLUR else None
    )

    colors = sv.ColorPalette.DEFAULT
    # Zones must be defined in DISPLAY resolution coordinates — the annotated
    # frame written to output_shm is at display_shape, not frame_shape.
    # Using frame_shape (512×512) here caused zones and box annotations to
    # appear only in the top-left corner of the 1920×1080 output frame.
    dh, dw, _ = display_shape

    # Load zones from database
    from src.engine.zones import ZoneManager
    from src.core.database import get_connection
    zone_mgr = ZoneManager()
    with get_connection() as conn:
        db_zones = zone_mgr.load_from_db(conn, cam_id)
    zone_mgr.init_zones_for_camera(cam_id, dw, dh)

    if db_zones:
        zones = [z._sv_zone for z in db_zones if z._sv_zone is not None]
        zone_ids = [z.zone_id for z in db_zones if z._sv_zone is not None]
        logger.info(f"Loaded {len(zones)} zones for camera {cam_id}")
    else:
        # Fallback: full-frame polygon (current behavior)
        polygons = [np.array([[0,0],[dw,0],[dw,dh],[0,dh]], dtype=np.int32)]
        zones = [sv.PolygonZone(polygon=polygon) for polygon in polygons]
        zone_ids = [0]
        logger.info(f"No zones for camera {cam_id}, using full-frame fallback")

    zone_annotators = [
        sv.PolygonZoneAnnotator(
            zone=zone,
            color=colors.by_idx(index),
            thickness=1,
            text_thickness=2,
            text_scale=1
        ) for index, zone in enumerate(zones)]

    zone_box_annotators = [
        sv.BoxAnnotator(
            color=colors.by_idx(index),
            thickness=1,
        ) for index in range(len(zones))]

    # Zone reload tracking
    last_zone_reload = time.time()
    ZONE_RELOAD_INTERVAL = 30  # seconds
    _frame_count = 0

    # Alert rate limiter
    from src.engine.alerts import AlertRateLimiter
    alert_limiter = AlertRateLimiter(cooldown_seconds=300)

    # dwell time tracking
    dwell_start = {}   # tracker_id -> enter_time
    dwell_total = defaultdict(float)  # tracker_id -> total accumulated seconds
    _last_valid_detections = sv.Detections.empty()
    while not stop_event.is_set():
        try:
            item = det_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if item is None:
            break  # shutdown signal

        idx, xyxy, confidence, class_id = item  # always 4 values now

        # Skipped frame path — output_shm already holds the last annotated frame.
        # Do NOT re-run process_frame: that caused the 120fps spin (embedder racing
        # through skipped frames, re-annotating the same pixels repeatedly) which
        # made FPSMonitor report inflated numbers while the browser received a
        # frozen image that only jumped when a real detection batch arrived.
        # Simply skip — the MJPEG generator reads output_shm directly and will
        # keep streaming the last good frame until a new one arrives.
        if idx is None:
            continue

        else:
            # Real frame path
            frame = frame_view(input_shm, frame_shape, frame_bytes, idx).copy()
            # Real captured detail for display — written by the reader from
            # the same source frame, just resized to display resolution
            # instead of inference resolution. This is what replaced the old
            # "upscale the tiny inference frame" step and is why boxes/text
            # look sharp now instead of blurry.
            native_frame = frame_view(native_shm, display_shape, display_bytes, idx).copy()
            if xyxy is not None:
                detections = sv.Detections(
                    xyxy=xyxy, confidence=confidence, class_id=class_id)
                _last_valid_detections = detections
                run_embedding = True
            else:
                # Real frame, but detector found nothing
                detections    = sv.Detections.empty()
                _last_valid_detections = detections
                run_embedding = False

        # Free the slot ONLY if it was a real frame — both frame and
        # native_frame have already been copied out above, so it's safe to
        # let the reader reuse this slot's memory in both rings now.
        if idx is not None:
            try:
                free_slots.put_nowait(idx)
            except queue.Full:
                pass  # should never happen but don't crash

        # Detections still come from inference on the small frame_shape
        # frame, so their coordinates need scaling to line up with
        # native_frame (display resolution) for drawing.
        render_frame = native_frame
        if (frame_shape[0], frame_shape[1]) != (display_shape[0], display_shape[1]):
            render_scale = (
                display_shape[1] / frame_shape[1],
                display_shape[0] / frame_shape[0],
            )
        else:
            render_scale = (1.0, 1.0)

        # Extract crops for embedding (before calling process_frame)
        precomputed_embeddings = None
        precomputed_crop_meta = None
        if run_embedding and embed_input_queue is not None and embed_output_queue is not None:
            crops_onnx = []
            crop_meta_list = []
            for det in detections:
                det_box, det_mask, det_conf, class_id, tracker_id, data = det
                if tracker_id is None:
                    continue
                input_tensor, crop_box, center_point, bbox_w, bbox_h, crop_flag = preprocess_crop(
                    frame, det_box,
                    model_input_size=(256, 128),
                    torso_ratio=1)
                if not crop_flag:
                    unique_visitors.add(tracker_id)
                    crops_onnx.append(input_tensor)
                    crop_meta_list.append((
                        crop_box, center_point, bbox_w, bbox_h,
                        int(tracker_id), det_conf))

            if crops_onnx:
                # Send crops to shared embedding worker
                request_id = f"emb_{cam_id}_{time.time()}"
                try:
                    embed_input_queue.put_nowait((request_id, cam_id, crops_onnx, crop_meta_list))
                    # Wait for embeddings from shared worker
                    result_id, embeddings, crop_meta = embed_output_queue.get(timeout=1.0)
                    if result_id == request_id:
                        precomputed_embeddings = embeddings
                        precomputed_crop_meta = crop_meta
                except (queue.Empty, queue.Full) as e:
                    logger.warning(f"Shared embedder timeout/error for camera {cam_id}: {e}")

        annotated = process_frame(
            frame=frame,
            display_frame=render_frame,
            scale=render_scale,
            detections=detections,
            tracker=tracker,
            embedder_session=None,  # No local session — using shared worker
            track_to_customer=track_to_customer,
            db_queue=db_queue,
            cam_id=cam_id,
            cam_label=cam_label,
            unique_visitors=unique_visitors,
            zones=zones,
            zone_annotators=zone_annotators,
            dwell_start=dwell_start,
            dwell_total=dwell_total,
            fps_monitor=fps_monitor,
            box_annotator=box_annotator,
            corner_annotator=corner_annotator,
            loiter_corner_annotator=loiter_corner_annotator,
            loiter_trace_annotator=loiter_trace_annotator,
            zone_box_annotators=zone_box_annotators,
            track_positions=track_positions,
            response_queue=response_queue,
            trace_annotator=trace_annotator,
            label_annotator=label_annotator,
            run_embedding=run_embedding,
            blur_annotator=blur_annotator,
            heatmap_accumulator=heatmap_accumulator,
            analytics_queue=analytics_queue,
            zone_ids=zone_ids,
            alert_queue=alert_queue,
            alert_limiter=alert_limiter,
            precomputed_embeddings=precomputed_embeddings,
            precomputed_crop_meta=precomputed_crop_meta,
        )

        # annotated is already at display resolution — process_frame drew
        # directly onto render_frame, no second resize needed here.
        output_buf = np.ndarray(display_shape, dtype=np.uint8, buffer=output_shm.buf)
        np.copyto(output_buf, annotated)

        # Signal generate() in main.py that a real annotated frame exists in
        # output_shm — switches the MJPEG stream from offline image to live.
        if not has_frame.value:
            has_frame.value = 1

        # Periodic zone reload (every 30 seconds)
        _frame_count += 1
        if _frame_count % 300 == 0:  # ~300 frames at 10fps = 30 seconds
            now = time.time()
            if now - last_zone_reload > ZONE_RELOAD_INTERVAL:
                try:
                    with get_connection() as conn:
                        new_db_zones = zone_mgr.load_from_db(conn, cam_id)
                    zone_mgr.init_zones_for_camera(cam_id, dw, dh)
                    new_zones = [z._sv_zone for z in new_db_zones if z._sv_zone is not None]
                    new_zone_ids = [z.zone_id for z in new_db_zones if z._sv_zone is not None]
                    if new_zone_ids != zone_ids:
                        zones = new_zones
                        zone_ids = new_zone_ids
                        # Recreate annotators
                        zone_annotators = [
                            sv.PolygonZoneAnnotator(
                                zone=zone,
                                color=colors.by_idx(index),
                                thickness=1,
                                text_thickness=2,
                                text_scale=1
                            ) for index, zone in enumerate(zones)]
                        zone_box_annotators = [
                            sv.BoxAnnotator(
                                color=colors.by_idx(index),
                                thickness=1,
                            ) for index in range(len(zones))]
                        logger.info(f"Reloaded {len(zones)} zones for camera {cam_id}")
                except Exception as e:
                    logger.error(f"Zone reload failed for camera {cam_id}: {e}")
                last_zone_reload = now

    input_shm.close()
    native_shm.close()
    output_shm.close()

def batched_detector_worker(
    frame_ready_queue,   # mp.Queue of (cam_id, idx)  — frames ready for detection
    det_queues,          # dict[cam_id -> mp.Queue]   — where to send results
    free_slots_queues,
    shm_names,           # dict[cam_id -> str]        — shared memory names
    frame_shape,         # (H, W, C) — same for all cameras
    frame_bytes,         # int
    stop_event,
    batch_timeout=0.15,  # 150ms — enough for 3 unsynchronized RTSP cameras
):
    """
    Single process that owns ONE YOLO session.
    Collects frames from all cameras, runs one batched inference,
    routes detections back to each camera's det_queue.
    """
    # Attach to all shared memory blocks
    shm_blocks = {
        cam_id: shared_memory.SharedMemory(name=name)
        for cam_id, name in shm_names.items()
    }
    # n_cams is read at spawn time from the dict snapshot — this is the correct
    # number of cameras this detector instance was started for. Do NOT use
    # len(shm_blocks) inside the loop as a wait target; use it only as the
    # max batch size. Waiting for exactly n_cams frames would deadlock if a
    # camera is removed and its reader stops producing — but since we restart
    # the batched detector on every add/remove, n_cams is always accurate for
    # the lifetime of this process.
    n_cams     = len(shm_blocks)
    detector   = get_detector()   # one YOLO session for ALL cameras
    max_batch  = min(n_cams, int(os.getenv("MAX_BATCH_SIZE", "8")))

    while not stop_event.is_set():

        # ── Collect a batch: wait up to batch_timeout for frames ──────────
        pending = {}   # cam_id -> slot_idx
        deadline = time.time() + batch_timeout

        while len(pending) < max_batch:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                cam_id, idx = frame_ready_queue.get(timeout=remaining)
                pending[cam_id] = idx
            except queue.Empty:
                break

        if not pending:
            continue

        # ── Build frame list in a stable order ───────────────────────────
        cam_order = list(pending.keys())
        frames    = []
        for cam_id in cam_order:
            idx  = pending[cam_id]
            shm  = shm_blocks[cam_id]
            frame = frame_view(shm, frame_shape, frame_bytes, idx)
            frames.append(frame.copy())   # copy out of shm before inference

        # ── Single batched inference ──────────────────────────────────────
        detections_list = detector.predict_batch(frames)   # list[sv.Detections]

        # ── Route results back to each camera's det_queue ─────────────────
        for cam_id, det, frame in zip(cam_order, detections_list, frames):
            idx = pending[cam_id]
            xyxy       = det.xyxy        if len(det) > 0 else None
            confidence = det.confidence  if len(det) > 0 else None
            class_id   = det.class_id   if len(det) > 0 else None
            try:
                # idx is always a real int here — batched detector only
                # receives real frames from frame_ready_queue
                det_queues[cam_id].put_nowait((idx, xyxy, confidence, class_id))
            except queue.Full:
                # Embedder backed up — we must free the slot here
                # or it leaks and free_slots empties permanently
                # We need the free_slots queue for this camera
                # (see Step 4 below for how to pass it)
                free_slots_queues[cam_id].put_nowait(idx)

    for shm in shm_blocks.values():
        shm.close() 