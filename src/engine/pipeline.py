import os
import cv2
import time
import queue
import psutil
import numpy as np
import supervision as sv
from loguru import logger
from collections import defaultdict
from src.core.config import settings
from src.vision.factory import get_detector
from src.vision.utils import create_session, preprocess_crop
from src.core.database import  fast_min_dist_to_customer
from multiprocessing import shared_memory

SKIP_FRAME = (None, None, None, None)

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
        frame = cv2.resize(frame, (frame_shape[1], frame_shape[0]))
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
        self.buffer_slots = buffer_slots
        
        # Frame shape is fixed system-wide — no need to probe the stream
        self.frame_shape = settings.FRAME_SHAPE   # (512, 512, 3) — fixed
        self.frame_bytes = settings.FRAME_BYTES   # fixed
        self.online = True                        # reader handles offline state
    
        # INPUT SHM
        self.input_shm = shared_memory.SharedMemory(
            create=True,
            size=self.frame_bytes * buffer_slots)
        self.input_shm_name = self.input_shm.name

        # OUTPUT SHM (ONLY ONE FRAME)
        self.output_shm = shared_memory.SharedMemory(
            create=True,
            size=self.frame_bytes)

        self.output_shm_name = self.output_shm.name
        for i in range(buffer_slots):
            self.free_slots.put(i)

    def start(self):
        self.p_embedder = self.ctx.Process(
            target=embedder_worker,
            args=(
                self.input_shm_name,
                self.output_shm_name,
                self.frame_shape,
                self.frame_bytes,
                self.det_queue,
                self.free_slots,
                self.stop_event,
                self.db_queue,
                self.response_queue,
                self.cam_id
            ),
            daemon=True
        )
        
        self.p_reader = self.ctx.Process(
            target=reader_worker,
            args=(
                self.RTSP_URL,
                self.cam_id,
                self.input_shm_name,
                self.frame_shape,
                self.frame_bytes,
                self.free_slots,
                self.frame_ready_queue,
                self.det_queue,
                self.stop_event,
            ),
            daemon=True
        )
        self.p_reader.start()
        self.p_embedder.start()

        
    def stop(self):
        self.stop_event.set()
        # Do NOT put None into frame_ready_queue — it is shared across all cameras.
        # stop_event.set() is the shutdown signal for batched_detector_worker.
        try:
            self.det_queue.put_nowait(None)   # unblock embedder_worker
        except:
            pass

        self.p_reader.join(timeout=2)
        self.p_embedder.join(timeout=2)

        self.input_shm.close()
        self.input_shm.unlink()

        self.output_shm.close()
        self.output_shm.unlink()

    def get_latest_frame(self):
        frame = np.ndarray(
            self.frame_shape,
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
    person_class_id=0
):
    # ── 0. Guard ──
    if detections is None or len(detections) == 0:
        fps_monitor.tick()
        annotated = sv.draw_text(
            scene=frame.copy(),
            text=f"FPS: {fps_monitor.fps:.1f}",
            text_anchor=sv.Point(x=20, y=40),
            text_color=sv.Color.WHITE,
            text_scale=0.7,
            text_thickness=2,
            background_color=sv.Color.BLACK
        )
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
        for tid in list(zone_dwell_start.keys()):
            if tid not in inside_now:
                elapsed = time.time() - zone_dwell_start.pop(tid)
                zone_dwell_total[tid] += elapsed
                logger.debug(f"Tracker {tid} left zone {zone_idx} after {elapsed:.1f}s")

        frame = zone_box_ann.annotate(scene=frame, detections=detections_in_zone)
        frame = zone_annotator.annotate(scene=frame)

    # ── 3. Crop collection ──
    crops_onnx, crop_meta = [], []
    loitering_tracker_ids = set()

    for det in detections:
        det_box, det_mask, det_conf, class_id, tracker_id, data = det
        if tracker_id is None:
            continue

        cx = (det_box[0] + det_box[2]) / 2
        cy = (det_box[1] + det_box[3]) / 2
        center_point = (cx, cy)

        if check_loitering(tracker_id, time.time(), center_point, track_positions):
            loitering_tracker_ids.add(tracker_id)

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

    if crops_onnx and run_embedding:
        batch_input = np.stack(crops_onnx, axis=0)
        embeddings = embedder_session.run(None, {"input": batch_input})[0]

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
                rid, customer_id, is_new, match_dist = response_queue.get()
                assert rid == request_id, f"Response ID mismatch: {rid} != {request_id}"

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

    # ── 6. Annotate ──
    annotated = frame.copy()
    if len(detections) > 0:
        annotated = trace_annotator.annotate(annotated, detections)
        annotated = box_annotator.annotate(annotated, detections)
        annotated = label_annotator.annotate(annotated, detections, labels)

    fps_monitor.tick()
    annotated = sv.draw_text(
        scene=annotated,
        text=f"FPS: {fps_monitor.fps:.1f}",
        text_anchor=sv.Point(x=20, y=40),
        text_color=sv.Color.WHITE,
        text_scale=0.7,
        text_thickness=2,
        background_color=sv.Color.BLACK
    )
    return annotated

def detector_worker(
    shm_name,
    frame_shape,
    frame_bytes,
    frame_ready_queue,
    det_queue,
    free_slots,
    stop_event
):
    pin_process([1,2,3])
    shm = shared_memory.SharedMemory(name=shm_name)
    detector_model = get_detector()
    while not stop_event.is_set():
        try:
            idx = frame_ready_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if idx is None:
            break

        frame = frame_view(shm,frame_shape,frame_bytes,idx)
        detections = detector_model.predict(frame)
        try:
            det_queue.put_nowait((
            idx,
            detections.xyxy,
            detections.confidence,
            detections.class_id
        ))
        except queue.Full:
            try:
                free_slots.put_nowait(idx)
            except:
                pass
    shm.close()

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
                  frame_shape,
                  frame_bytes,
                  free_slots,
                  frame_ready_queue,
                  det_queue,
                  stop_event):
    pin_process([0])
    shm = shared_memory.SharedMemory(name=shm_name)

    cap = None
    first_attempt = True
    online = False
    consecutive_failures = 0
    max_backoff = 60.0
    offline_frame = np.zeros(frame_shape, dtype=np.uint8)
    cv2.putText(offline_frame, "Stream offline", (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    frame_counter = 0

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

            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                consecutive_failures += 1
                # Signal embedder that stream is offline
                try:
                    det_queue.put_nowait((None, None, None, None))
                except queue.Full:
                    pass
                continue

            online = True
            consecutive_failures = 0

        ret, frame = cap.read()
        if not ret:
            online = False
            consecutive_failures += 1
            try:
                det_queue.put_nowait((None, None, None, None))
            except queue.Full:
                pass
            continue

        frame_counter += 1
        consecutive_failures = 0

        if frame_counter % 3 == 0:
            # Keyframe — needs a real slot
            try:
                idx = free_slots.get_nowait()
            except queue.Empty:
                # No slot available, detector is backed up — drop this keyframe
                # Don't crash, don't block, just skip
                continue
            _write_frame_to_slot(shm, idx, frame, frame_shape, frame_bytes)
            try:
                frame_ready_queue.put_nowait((cam_id, idx))
            except queue.Full:
                # Detector queue full — return slot immediately
                free_slots.put_nowait(idx)
        else:
            # Skipped frame — no slot needed at all
            try:
                det_queue.put_nowait(SKIP_FRAME)
            except queue.Full:
                pass  # embedder backed up, drop the signal

    if cap is not None:
        cap.release()
    shm.close()

def embedder_worker(
    input_shm_name,
    output_shm_name,
    frame_shape,
    frame_bytes,
    det_queue,
    free_slots,
    stop_event,
    db_queue,
    response_queue,
    cam_id
):
    pin_process([4,5])
    input_shm = shared_memory.SharedMemory(name=input_shm_name)
    output_shm = shared_memory.SharedMemory(name=output_shm_name)
    embedder_session = create_session(settings.FEATURE_EXTRACTOR_MODEL, num_threads=1)
    track_to_customer = {}
    unique_visitors = set()   # persists across frames
    track_positions = defaultdict(list)   # local to this process — not shared
    unique_visitors = set()   # persists across frames — counts unique tracker IDs seen
    tracker = sv.ByteTrack(lost_track_buffer=120) # 120 frames ==> This stops the tracker from killing a track when a person is briefly occluded or missed by the detector.
    fps_monitor = sv.FPSMonitor()
    color = sv.ColorPalette.DEFAULT
    box_annotator = sv.EllipseAnnotator(color=color)
    trace_annotator = sv.TraceAnnotator(color=color, trace_length=30)
    label_annotator = sv.LabelAnnotator(color=color, text_color=sv.Color.BLACK)
    
    colors = sv.ColorPalette.DEFAULT
    h,w,c = frame_shape
    # Build the full‑frame polygon (top‑left → top‑right → bottom‑right → bottom‑left)

    polygons = [
            np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ], dtype=np.int32)
    ]


    zones = [sv.PolygonZone(polygon=polygon) for polygon in polygons]
    
    zone_annotators = [
        sv.PolygonZoneAnnotator(
            zone=zone,
            color=colors.by_idx(index),
            thickness=4,
            text_thickness=8,
            text_scale=4
        ) for index, zone in enumerate(zones)]
     
    zone_box_annotators = [
        sv.BoxAnnotator(
            color=colors.by_idx(index),
            thickness=4,
        ) for index in range(len(polygons))]
    
    # dwell time tracking
    dwell_start = {}   # tracker_id -> enter_time
    dwell_total = defaultdict(float)  # tracker_id -> total accumulated seconds
    last_valid_detections = sv.Detections.empty()
    while not stop_event.is_set():
        try:
            item = det_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if item is None:
            break  # shutdown signal

        idx, xyxy, confidence, class_id = item  # always 4 values now

        # Skipped frame path
        if idx is None:
            # No shared memory involved — reuse last state
            detections    = last_valid_detections
            run_embedding = False
            # Read last annotated frame from output shm for re-display
            frame = np.ndarray(
                frame_shape, dtype=np.uint8, buffer=output_shm.buf
            ).copy()
        else:
            # Real frame path
            frame = frame_view(input_shm, frame_shape, frame_bytes, idx).copy()
            if xyxy is not None:
                detections = sv.Detections(
                    xyxy=xyxy, confidence=confidence, class_id=class_id)
                last_valid_detections = detections
                run_embedding = True
            else:
                # Real frame, but detector found nothing
                detections    = sv.Detections.empty()
                last_valid_detections = detections
                run_embedding = False

        # Free the slot ONLY if it was a real frame
        if idx is not None:
            try:
                free_slots.put_nowait(idx)
            except queue.Full:
                pass  # should never happen but don't crash

        annotated = process_frame(
            frame=frame,
            detections=detections,
            tracker=tracker,
            embedder_session=embedder_session,
            track_to_customer=track_to_customer,
            db_queue=db_queue,
            cam_id=cam_id,
            unique_visitors=unique_visitors,
            zones=zones,
            zone_annotators=zone_annotators,
            dwell_start=dwell_start,
            dwell_total=dwell_total,
            fps_monitor=fps_monitor,
            box_annotator=box_annotator,
            zone_box_annotators=zone_box_annotators,
            track_positions=track_positions,
            response_queue=response_queue,
            trace_annotator=trace_annotator,
            label_annotator=label_annotator,
            run_embedding=run_embedding
        )

        # Write annotated frame to output shared memory
        output_frame = np.ndarray(frame_shape, dtype=np.uint8, buffer=output_shm.buf)
        np.copyto(output_frame, annotated)

    input_shm.close()
    output_shm.close()

def batched_detector_worker(
    frame_ready_queue,   # mp.Queue of (cam_id, idx)  — frames ready for detection
    det_queues,          # dict[cam_id -> mp.Queue]   — where to send results
    free_slots_queues,
    shm_names,           # dict[cam_id -> str]        — shared memory names
    frame_shape,         # (H, W, C) — same for all cameras
    frame_bytes,         # int
    stop_event,
    batch_timeout=0.02,  # seconds to wait collecting a full batch
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
    n_cams     = len(shm_blocks)
    detector   = get_detector()   # one YOLO session for ALL cameras

    while not stop_event.is_set():

        # ── Collect a batch: wait up to batch_timeout for frames ──────────
        pending = {}   # cam_id -> slot_idx 
        deadline = time.time() + batch_timeout

        while len(pending) < n_cams:
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