# import os
# import cv2
# import time
# import queue
# import psutil
# import numpy as np
# import supervision as sv
# import onnxruntime as ort
# from loguru import logger
# from datetime import datetime
# from collections import defaultdict
# from src.core.config import settings
# from src.vision.factory import get_detector
# from src.core.database import  fast_min_dist_to_customer
# from multiprocessing import shared_memory


# def pin_process(cores):
#     try:
#         psutil.Process(os.getpid()).cpu_affinity(list(cores))
#     except Exception:
#         pass

# def frame_view(shm: shared_memory.SharedMemory, frame_shape, frame_bytes: int, idx: int):
#     offset = idx * frame_bytes
#     return np.ndarray(frame_shape, dtype=np.uint8, buffer=shm.buf, offset=offset)

# def create_session(model_path, num_threads=2):
#     sess_options = ort.SessionOptions()
#     sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
#     sess_options.enable_mem_pattern = True
#     sess_options.enable_cpu_mem_arena = True
#     sess_options.intra_op_num_threads = num_threads
#     sess_options.inter_op_num_threads = 2
#     sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
#     sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
#     available = ort.get_available_providers()
#     providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
#     return ort.InferenceSession(model_path, 
#                                 sess_options=sess_options,
#                                 providers=providers)
    
# # shared memory write helper
# def _write_frame_to_slot(shm, idx, frame, frame_shape, frame_bytes):
#     dst = frame_view(shm, frame_shape, frame_bytes, idx)
#     if frame.shape != frame_shape:
#         frame = cv2.resize(frame, (frame_shape[1], frame_shape[0]))
#     np.copyto(dst, frame)
    
# def preprocess_crop(frame, bbox, torso_ratio, model_input_size):
#     x1, y1, x2, y2 = map(int, bbox)
#     w, h = x2 - x1, y2 - y1
#     cx, cy = (x1+x2)/2, (y1+y2)/2 
#     crop_y2 = int(y1 + h * torso_ratio)
#     crop = frame[y1:crop_y2, x1:x2]
#     flag =  crop.size == 0 or w < 20 or h < 20

#     # Preprocess
#     resized = cv2.resize(crop, (model_input_size[1], model_input_size[0]),
#                         interpolation=cv2.INTER_AREA)
#     # Normalize: (img / 255 - mean) / std
#     mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
#     std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
#     normalized = (resized / 255.0 - mean) / std
    
#     # Convert to CHW
#     input_tensor = np.transpose(normalized, (2, 0, 1)).astype(np.float32)
#     return input_tensor, (x1, y1, x2, y2), (cx, cy), flag

# class VisionPipeline:
#     def __init__(self,
#                 RTSP_URL,
#                 CAM_ID,
#                 ctx,
#                 free_slots,
#                 ready_slots,
#                 det_queue,
#                 stop_event,
#                 db_queue,
#                 response_queue,
#                 buffer_slots=4):
        
#         self.model_input_size = (256, 128)
#         self.ctx = ctx
#         self.RTSP_URL = RTSP_URL
#         self.cam_id = CAM_ID
#         self.torso_ratio = 2/3
#         self.free_slots = free_slots
#         self.ready_slots= ready_slots
#         self.det_queue  = det_queue
#         self.stop_event = stop_event
#         self.db_queue   = db_queue
#         self.response_queue = response_queue
#         self.buffer_slots = buffer_slots
        
#         probe = cv2.VideoCapture(self.RTSP_URL)
#         ok, frame = probe.read()
#         probe.release()
#         # Class‑level constant – adjust as you like
#         WORKING_HEIGHT = 512
#         WORKING_WIDTH  = 512
#         WORKING_CHANNELS = 3
        
#         if not ok:
#             self.online = False
#             # Fixed frame shape – never changes
#             self.frame_shape = (WORKING_HEIGHT, WORKING_WIDTH, WORKING_CHANNELS)
#             self.frame_bytes = WORKING_HEIGHT * WORKING_WIDTH * WORKING_CHANNELS
#         else:
#             self.frame_shape = frame.shape
#             self.frame_bytes = frame.nbytes
#             self.online = True
    
#         # INPUT SHM
#         self.input_shm = shared_memory.SharedMemory(
#             create=True,
#             size=self.frame_bytes * buffer_slots)
#         self.input_shm_name = self.input_shm.name

#         # OUTPUT SHM (ONLY ONE FRAME)
#         self.output_shm = shared_memory.SharedMemory(
#             create=True,
#             size=self.frame_bytes)

#         self.output_shm_name = self.output_shm.name
#         for i in range(buffer_slots):
#             self.free_slots.put(i)

#     def start(self):
#         self.p_reader = self.ctx.Process(
#             target=reader_worker,
#             args=(
#                 self.RTSP_URL,
#                 self.input_shm_name,
#                 self.frame_shape,
#                 self.frame_bytes,
#                 self.free_slots,
#                 self.ready_slots,
#                 self.stop_event
#             ),
#             daemon=True
#         )

#         self.p_detector = self.ctx.Process(
#             target=detector_worker,
#             args=(
#                 self.input_shm_name,
#                 self.frame_shape,
#                 self.frame_bytes,
#                 self.ready_slots,
#                 self.det_queue,
#                 self.free_slots,
#                 self.stop_event
#             ),
#             daemon=True
#         )

#         self.p_embedder = self.ctx.Process(
#             target=embedder_worker,
#             args=(
#                 self.input_shm_name,
#                 self.output_shm_name,
#                 self.frame_shape,
#                 self.frame_bytes,
#                 self.det_queue,
#                 self.free_slots,
#                 self.stop_event,
#                 self.db_queue,
#                 self.response_queue,
#                 self.cam_id
#             ),
#             daemon=True
#         )

#         self.p_reader.start()
#         self.p_detector.start()
#         self.p_embedder.start()
        
#     def stop(self):
#         self.stop_event.set()
#         try:
#             self.ready_slots.put_nowait(None)
#         except:
#             pass

#         try:
#             self.det_queue.put_nowait(None)
#         except:
#             pass

#         self.p_reader.join(timeout=2)
#         self.p_detector.join(timeout=2)
#         self.p_embedder.join(timeout=2)

#         self.input_shm.close()
#         self.input_shm.unlink()

#         self.output_shm.close()
#         self.output_shm.unlink()

#     def get_latest_frame(self):
#         frame = np.ndarray(
#             self.frame_shape,
#             dtype=np.uint8,
#             buffer=self.output_shm.buf
#         )
#         return frame.copy()
    
# def process_frame(
#     frame,
#     detections,
#     tracker,
#     embedder_session,
#     track_to_customer,
#     db_queue,
#     cam_id,
#     box_annotator,
#     fps_monitor,
#     trace_annotator,
#     label_annotator,
#     response_queue,
#     diversity_threshold=0.2,
#     person_class_id=0
# ):

#     if detections.class_id is not None:
#         detections = detections[
#             detections.class_id == person_class_id
#         ]

#     detections = tracker.update_with_detections(detections)

#     crops_onnx = []
#     crop_meta = []
#     labels = []

#     for det in detections:
#         det_box, mask, confidence, class_id, tracker_id, data = det
#         if tracker_id is None:
#             continue
        
#         input_tensor, crop_box, center_point, crop_flag = preprocess_crop(
#             frame,
#             det_box,
#             model_input_size=(128, 64),
#             torso_ratio=2/3,
#         )

#         if crop_flag:
#             continue

#         crops_onnx.append(input_tensor)
#         crop_meta.append((crop_box, int(tracker_id), confidence))

#     if not crops_onnx:
#         return frame

#     batch_input = np.stack(crops_onnx, axis=0)
#     embeddings = embedder_session.run(None, {"input": batch_input})[0]

#     for i, emb in enumerate(embeddings):
#         emb = emb.flatten()
#         emb /= (np.linalg.norm(emb) + 1e-8)
#         bbox, tracker_id, conf = crop_meta[i]
#         if tracker_id in track_to_customer:
#             customer_id = track_to_customer[tracker_id]
#             if fast_min_dist_to_customer(emb, customer_id) > diversity_threshold:
#                 db_queue.put(("store_embedding", customer_id, cam_id, emb, datetime.now()))
#         else:
#             # Ask the writer to match or create – atomic, no race conditions
#             request_id = f"{cam_id}_{tracker_id}_{time.time()}"
#             db_queue.put((
#                 "match_or_register",
#                 emb,
#                 cam_id,
#                 datetime.now(),
#                 request_id      
#             ))
            
#             # Wait for the reply that carries our request_id
#             while True:
#                 rid, customer_id, is_new = response_queue.get()
#                 if rid == request_id:
#                     break
#                 # Not ours – put it back for other consumers (if any)
#                 response_queue.put((rid, customer_id, is_new))

#             track_to_customer[tracker_id] = customer_id

#             # Optionally store a diverse embedding if the person already existed
#             if not is_new:
#                 if fast_min_dist_to_customer(emb, customer_id) > diversity_threshold:
#                     db_queue.put(("store_embedding", customer_id, cam_id, emb, datetime.now()))

#         labels.append(
#             f"#{int(tracker_id)} {confidence:.2f} "
#             f"#{tracker_id} ID:{customer_id}"
#         )
#     annotated = frame.copy()
#     annotated = trace_annotator.annotate(annotated,detections)
#     annotated = box_annotator.annotate(annotated,detections)
#     annotated = label_annotator.annotate(annotated,detections,labels)
    
#     # Tick every frame
#     fps_monitor.tick()

#     # Draw the FPS value on the frame
#     annotated = sv.draw_text(
#         scene=annotated,
#         text=f"FPS: {fps_monitor.fps:.1f}",
#         text_anchor=sv.Point(x=20, y=40),
#         text_color=sv.Color.WHITE,
#         text_scale=0.7,
#         text_thickness=2,
#         background_color=sv.Color.BLACK
#     )
#     return annotated

# def detector_worker(
#     shm_name,
#     frame_shape,
#     frame_bytes,
#     ready_slots,
#     det_queue,
#     free_slots,
#     stop_event
# ):
#     pin_process([1,2,3])
#     shm = shared_memory.SharedMemory(name=shm_name)
#     detector_model = get_detector()
#     while not stop_event.is_set():
#         try:
#             idx = ready_slots.get(timeout=0.1)
#         except queue.Empty:
#             continue

#         if idx is None:
#             break

#         frame = frame_view(shm,frame_shape,frame_bytes,idx)
#         detections = detector_model.predict(frame)
#         try:
#             det_queue.put_nowait((
#             idx,
#             detections.xyxy,
#             detections.confidence,
#             detections.class_id
#         ))
#         except queue.Full:
#             try:
#                 free_slots.put_nowait(idx)
#             except:
#                 pass
#     shm.close()

# # How the backoff works:
# # First failure → wait 1 s, then retry.
# # Second failure → wait 2 s, then retry.
# # Third failure → wait 4 s, then retry.
# # … up to a maximum of 60 s.
# # As soon as a frame is read successfully, consecutive_failures resets to 0.
# # During the wait, the thread is sleeping – it consumes almost zero CPU.
# def reader_worker(rtsp_url,
#                   shm_name,
#                   frame_shape, 
#                   frame_bytes,
#                   free_slots,
#                   ready_slots, 
#                   stop_event):
#     pin_process([0])
#     shm = shared_memory.SharedMemory(name=shm_name)

#     cap = None
#     first_attempt = True          # <-- ADD THIS
#     online = False
#     consecutive_failures = 0
#     max_backoff = 60.0            # never sleep longer than 60 seconds
#     offline_frame = np.zeros(frame_shape, dtype=np.uint8)
#     # Draw the offline message on the placeholder (optional)
#     cv2.putText(offline_frame, "Stream offline", (50, 240),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     while not stop_event.is_set():
#         if not online:
#             # Release any old capture and wait before reconnecting
#             if cap is not None:
#                 cap.release()
#                 cap = None
#             # Exponential backoff: 1s, 2s, 4s, 8s, ... capped at max_backoff
#             if first_attempt and consecutive_failures < 6:  # <-- no sleep on the first try
#                 first_attempt = False
#                 sleep_time = 0
#             else:
#                 sleep_time = min(1.0 * (2 ** consecutive_failures), max_backoff)

#             stop_event.wait(sleep_time)   # wait but break if stop_event is set
#             if stop_event.is_set():
#                 break
#             # Attempt to open the stream
#             cap = cv2.VideoCapture(rtsp_url)
#             if not cap.isOpened():
#                 consecutive_failures += 1
#                 # Push an offline placeholder into one slot so the user sees it
#                 _push_placeholder_frame(shm, frame_shape, frame_bytes,
#                                         free_slots, ready_slots, offline_frame)
#                 continue
#             online = True
#             consecutive_failures = 0

#         # Stream is open – try to read
#         ret, frame = cap.read()
#         if not ret:
#             # Stream died – switch back to offline mode
#             online = False
#             consecutive_failures += 1
#             _push_placeholder_frame(shm, frame_shape, frame_bytes,
#                                     free_slots, ready_slots, offline_frame)
#             continue

#         # Live frame received – push it into the ring buffer
#         consecutive_failures = 0
#         try:
#             idx = free_slots.get_nowait()
#         except queue.Empty:
#             continue   # ring is full, drop frame (normal back‑pressure)
#         _write_frame_to_slot(shm, idx, frame, frame_shape, frame_bytes)
#         try:
#             ready_slots.put_nowait(idx)
#         except queue.Full:
#             try:
#                 free_slots.put_nowait(idx)
#             except:
#                 pass

#     # Cleanup
#     if cap is not None:
#         cap.release()
#     shm.close()

# def _push_placeholder_frame(shm, frame_shape, frame_bytes,
#                             free_slots, ready_slots, placeholder):
#     """Push a single offline frame into the ring if a slot is available."""
#     try:
#         idx = free_slots.get_nowait()
#     except queue.Empty:
#         return   # ring is full – nothing to do
#     _write_frame_to_slot(shm, idx, placeholder, frame_shape, frame_bytes)
#     try:
#         ready_slots.put_nowait(idx)
#     except queue.Full:
#         try:
#             free_slots.put_nowait(idx)
#         except:
#             pass

# def embedder_worker(
#     input_shm_name,
#     output_shm_name,
#     frame_shape,
#     frame_bytes,
#     det_queue,
#     free_slots,
#     stop_event,
#     db_queue,
#     response_queue,
#     cam_id
# ):
#     pin_process([4,5])
#     input_shm = shared_memory.SharedMemory(name=input_shm_name)
#     output_shm = shared_memory.SharedMemory(name=output_shm_name)
#     embedder_session = create_session(
#         settings.FEATURE_EXTRACTOR_MODEL, num_threads=1)

#     track_to_customer = {}
#     tracker = sv.ByteTrack()
#     fps_monitor = sv.FPSMonitor()
#     color = sv.ColorPalette.DEFAULT
#     box_annotator = sv.BoxAnnotator(color=color)
#     trace_annotator = sv.TraceAnnotator(color=color,trace_length=30)
#     label_annotator = sv.LabelAnnotator(color=color,text_color=sv.Color.BLACK)
#     zone_polygon = np.array([[100, 200], [400, 200], [400, 500], [100, 500]])
#     zone = sv.PolygonZone(polygon=zone_polygon)
#     zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.GREEN)
#     # dwell time tracking
#     dwell_start = {}   # tracker_id -> enter_time
#     dwell_total = defaultdict(float)  # tracker_id -> total accumulated seconds
#     while not stop_event.is_set():
#         try:
#             item = det_queue.get(timeout=0.1)
#         except queue.Empty:
#             continue
        
#         if item is None:
#             break
        
#         idx, xyxy, confidence, class_id = item
#         detections = sv.Detections(
#             xyxy=xyxy,
#             confidence=confidence,
#             class_id=class_id
#         )
#         # --- Tick the FPS counter ---
#         fps_monitor.tick()

#         frame = frame_view(input_shm,frame_shape,frame_bytes,idx)
#         annotated = process_frame(
#             frame=frame,
#             detections=detections,
#             tracker=tracker,
#             embedder_session=embedder_session,
#             track_to_customer=track_to_customer,
#             db_queue=db_queue,
#             cam_id=cam_id,
#             zone_annotator=zone_annotator,
#             dwell_start=dwell_start,
#             dwell_total=dwell_total,
#             fps_monitor = fps_monitor,
#             box_annotator=box_annotator,
#             response_queue=response_queue,
#             trace_annotator=trace_annotator,
#             label_annotator=label_annotator
#         )

#         # WRITE DIRECTLY TO OUTPUT SHM
#         output_frame = np.ndarray(
#             frame_shape,
#             dtype=np.uint8,
#             buffer=output_shm.buf
#         )
#         np.copyto(output_frame, annotated)
#         try:
#             free_slots.put_nowait(idx)
#         except:
#             pass

#     input_shm.close()
#     output_shm.close()











































# import os
# import cv2
# import time
# import queue
# import psutil
# import numpy as np
# import supervision as sv
# import onnxruntime as ort
# from loguru import logger
# from datetime import datetime
# from collections import defaultdict
# from src.core.config import settings
# from src.vision.factory import get_detector
# from src.core.database import  fast_min_dist_to_customer
# from multiprocessing import shared_memory


# def pin_process(cores):
#     try:
#         psutil.Process(os.getpid()).cpu_affinity(list(cores))
#     except Exception:
#         pass

# def frame_view(shm: shared_memory.SharedMemory, frame_shape, frame_bytes: int, idx: int):
#     offset = idx * frame_bytes
#     return np.ndarray(frame_shape, dtype=np.uint8, buffer=shm.buf, offset=offset)

# def create_session(model_path, num_threads=2):
#     sess_options = ort.SessionOptions()
#     sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
#     sess_options.enable_mem_pattern = True
#     sess_options.enable_cpu_mem_arena = True
#     sess_options.intra_op_num_threads = num_threads
#     sess_options.inter_op_num_threads = 2
#     sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
#     sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
#     available = ort.get_available_providers()
#     providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
#     return ort.InferenceSession(model_path, 
#                                 sess_options=sess_options,
#                                 providers=providers)
    
# # shared memory write helper
# def _write_frame_to_slot(shm, idx, frame, frame_shape, frame_bytes):
#     dst = frame_view(shm, frame_shape, frame_bytes, idx)
#     if frame.shape != frame_shape:
#         frame = cv2.resize(frame, (frame_shape[1], frame_shape[0]))
#     np.copyto(dst, frame)

# def letterbox(img, input_size) -> tuple[np.ndarray, tuple[int, int]]:
#     """
#     Resize and pad image to target size while preserving aspect ratio.
#     Returns:
#         - padded image (np.ndarray) of shape (target_h, target_w, 3)
#         - pad (top, left) amounts used for padding (needed to map boxes back).
#     """
#     shape = img.shape[:2]                     # original (height, width)
#     target_h, target_w = input_size
#     r = min(target_h / shape[0], target_w / shape[1])
#     new_unpad = (round(shape[1] * r), round(shape[0] * r))  # (width, height)
#     dw, dh = (target_w - new_unpad[0]) / 2, (target_h - new_unpad[1]) / 2

#     # Resize only if needed
#     if shape[::-1] != new_unpad:
#         img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

#     # Pad with constant gray (114,114,114)
#     top, bottom = round(dh - 0.1), round(dh + 0.1)
#     left, right = round(dw - 0.1), round(dw + 0.1)
#     img = cv2.copyMakeBorder(img, top, bottom, left, right,
#                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
#     return img, (top, left)

# def preprocess_crop(frame, bbox, torso_ratio, model_input_size):
#     """
#     Preprocess the input image: BGR->RGB, letterbox, normalize, CHW, batch.
#     Returns:
#         - image_data: (1, 3, H, W) float32 numpy array (values 0..1)
#         - pad: (top, left) used for padding
#     """
#     x1, y1, x2, y2 = map(int, bbox)
#     w, h = x2 - x1, y2 - y1
#     cx, cy = (x1+x2)/2, (y1+y2)/2 
#     # crop_y2 = int(y1 + h * torso_ratio)
#     # frame = frame[y1:crop_y2, x1:x2]
#     # flag =  frame.size == 0 or w < 20 or h < 20
#     flag = False
#     img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     img_padded, pad = letterbox(img_rgb, model_input_size)
#     # img_norm = img_padded.astype(np.float32) / 255.0
#     mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
#     std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
#     img_norm = (img_padded / 255.0 - mean) / std
#     img_chw = np.transpose(img_norm, (2, 0, 1)).astype(np.float32)        # (C, H, W)
#     return img_chw, (x1, y1, x2, y2), (cx, cy), flag


# class VisionPipeline:
#     def __init__(self,
#                 RTSP_URL,
#                 CAM_ID,
#                 ctx,
#                 free_slots,
#                 ready_slots,
#                 det_queue,
#                 stop_event,
#                 db_queue,
#                 response_queue,
#                 buffer_slots=4):
        
#         self.model_input_size = (256, 128)
#         self.ctx = ctx
#         self.RTSP_URL = RTSP_URL
#         self.cam_id = CAM_ID
#         self.torso_ratio = 2/3
#         self.free_slots = free_slots
#         self.ready_slots= ready_slots
#         self.det_queue  = det_queue
#         self.stop_event = stop_event
#         self.db_queue   = db_queue
#         self.response_queue = response_queue
#         self.buffer_slots = buffer_slots
        
#         probe = cv2.VideoCapture(self.RTSP_URL)
#         ok, frame = probe.read()
#         probe.release()
#         # Class‑level constant – adjust as you like
#         WORKING_HEIGHT = 512
#         WORKING_WIDTH  = 512
#         WORKING_CHANNELS = 3
        
#         if not ok:
#             self.online = False
#             # Fixed frame shape – never changes
#             self.frame_shape = (WORKING_HEIGHT, WORKING_WIDTH, WORKING_CHANNELS)
#             self.frame_bytes = WORKING_HEIGHT * WORKING_WIDTH * WORKING_CHANNELS
#         else:
#             self.frame_shape = frame.shape
#             self.frame_bytes = frame.nbytes
#             self.online = True
    
#         # INPUT SHM
#         self.input_shm = shared_memory.SharedMemory(
#             create=True,
#             size=self.frame_bytes * buffer_slots)
#         self.input_shm_name = self.input_shm.name

#         # OUTPUT SHM (ONLY ONE FRAME)
#         self.output_shm = shared_memory.SharedMemory(
#             create=True,
#             size=self.frame_bytes)

#         self.output_shm_name = self.output_shm.name
#         for i in range(buffer_slots):
#             self.free_slots.put(i)

#     def start(self):
#         self.p_reader = self.ctx.Process(
#             target=reader_worker,
#             args=(
#                 self.RTSP_URL,
#                 self.input_shm_name,
#                 self.frame_shape,
#                 self.frame_bytes,
#                 self.free_slots,
#                 self.ready_slots,
#                 self.stop_event
#             ),
#             daemon=True
#         )

#         self.p_detector = self.ctx.Process(
#             target=detector_worker,
#             args=(
#                 self.input_shm_name,
#                 self.frame_shape,
#                 self.frame_bytes,
#                 self.ready_slots,
#                 self.det_queue,
#                 self.free_slots,
#                 self.stop_event
#             ),
#             daemon=True
#         )

#         self.p_embedder = self.ctx.Process(
#             target=embedder_worker,
#             args=(
#                 self.input_shm_name,
#                 self.output_shm_name,
#                 self.frame_shape,
#                 self.frame_bytes,
#                 self.det_queue,
#                 self.free_slots,
#                 self.stop_event,
#                 self.db_queue,
#                 self.response_queue,
#                 self.cam_id
#             ),
#             daemon=True
#         )

#         self.p_reader.start()
#         self.p_detector.start()
#         self.p_embedder.start()
        
#     def stop(self):
#         self.stop_event.set()
#         try:
#             self.ready_slots.put_nowait(None)
#         except:
#             pass

#         try:
#             self.det_queue.put_nowait(None)
#         except:
#             pass

#         self.p_reader.join(timeout=2)
#         self.p_detector.join(timeout=2)
#         self.p_embedder.join(timeout=2)

#         self.input_shm.close()
#         self.input_shm.unlink()

#         self.output_shm.close()
#         self.output_shm.unlink()

#     def get_latest_frame(self):
#         frame = np.ndarray(
#             self.frame_shape,
#             dtype=np.uint8,
#             buffer=self.output_shm.buf
#         )
#         return frame.copy()
    
# #####################3.2 Loitering detection
# def check_loitering(tracker_id,
#                     current_time,
#                     centroid,
#                     track_positions,
#                     loiter_threshold_pixels = 50,
#                     loiter_time_threshold = 10.0):
#     positions = track_positions[tracker_id]
#     # remove old entries older than loiter_time_threshold
#     while positions and positions[0][0] < current_time - loiter_time_threshold:
#         positions.pop(0)
#     positions.append((current_time, centroid))
    
#     if len(positions) >= 2:
#         first_pos = positions[0][1]
#         displacement = np.linalg.norm(np.array(centroid) - np.array(first_pos))
#         duration = current_time - positions[0][0]
#         if displacement < loiter_threshold_pixels and duration >= loiter_time_threshold:
#             return True
#     return False    

# def process_frame(
#     frame,
#     detections,
#     tracker,
#     embedder_session,
#     track_to_customer,
#     db_queue,
#     cam_id,
#     box_annotator,
#     fps_monitor,
#     trace_annotator,
#     label_annotator,
#     response_queue,
#     zones,
#     zone_annotators,
#     zone_box_annotators,
#     dwell_start,
#     dwell_total,
#     track_positions,
#     unique_visitors,
#     person_class_id=0
# ):
#     # ── 0. Guard against None or completely empty detections ──
#     if detections is None or len(detections) == 0:
#         # Still draw the zone outline and FPS on a clean copy
#         fps_monitor.tick()
#         annotated = sv.draw_text(
#             scene=frame.copy(),
#             text=f"FPS: {fps_monitor.fps:.1f}",
#             text_anchor=sv.Point(x=20, y=40),
#             text_color=sv.Color.WHITE,
#             text_scale=0.7,
#             text_thickness=2,
#             background_color=sv.Color.BLACK
#         )
#         return annotated

#     # ── 1. Filter & Track ──
#     if detections.class_id is not None:
#         detections = detections[detections.class_id == person_class_id]

#     detections = tracker.update_with_detections(detections)

#     # ── Zone analytics (once per frame, per zone) ──
#     # (Make sure dwell_start, dwell_total are dicts keyed by zone_id, each holding a dict tracker_id -> value)
#     for zone_idx, (zone, zone_annotator, box_annotator) in enumerate(zip(zones, zone_annotators, zone_box_annotators)):
#         mask = zone.trigger(detections=detections)           # boolean mask (supervision < 0.20) or Detections
#         if isinstance(mask, np.ndarray):
#             detections_in_zone = detections[mask]
#         else:
#             detections_in_zone = mask

#         inside_now = set(tid for tid in detections_in_zone.tracker_id if tid is not None)

#         # Update dwell start times
#         zone_dwell_start = dwell_start.setdefault(zone_idx, {})
#         zone_dwell_total = dwell_total.setdefault(zone_idx, defaultdict(float))

#         for tid in inside_now:
#             if tid not in zone_dwell_start:
#                 zone_dwell_start[tid] = time.time()

#         # Check who left
#         for tid in list(zone_dwell_start.keys()):
#             if tid not in inside_now:
#                 elapsed = time.time() - zone_dwell_start.pop(tid)
#                 zone_dwell_total[tid] += elapsed
#                 # print(f"Customer {tid} left zone {zone_idx} after {elapsed:.1f}s")

#         # Draw zone and its boxes
#         frame = box_annotator.annotate(scene=frame, detections=detections_in_zone)
#         frame = zone_annotator.annotate(scene=frame)

#     # ── Loitering, unique visitors, crop collection ──
#     crops_onnx, crop_meta, labels = [], [], []
#     loitering_tracker_ids = set()

#     for det in detections:
#         det_box, mask, confidence, class_id, tracker_id, data = det
#         if tracker_id is None:
#             continue

#         # Unique visitor counting
#         unique_visitors.add(tracker_id)

#         # Loitering check
#         x1, y1, x2, y2 = det_box
#         center = ((x1 + x2) / 2, (y1 + y2) / 2)
#         if check_loitering(tracker_id, time.time(), center, track_positions):
#             loitering_tracker_ids.add(tracker_id)

#         # Crop preprocessing
#         input_tensor, crop_box, center_point, crop_flag = preprocess_crop(
#             frame, det_box,
#             # model_input_size=(128, 64),
#             model_input_size=(256, 128),
#             # torso_ratio=2/3
#             torso_ratio=1
            
#         )
#         if crop_flag:
#             continue

#         crops_onnx.append(input_tensor)
#         crop_meta.append((crop_box, int(tracker_id), confidence))

#     # ── 4. Embedding & Re‑ID (only if crops exist) ──
#     if crops_onnx:
#         batch_input = np.stack(crops_onnx, axis=0)
#         embeddings = embedder_session.run(None, {"input": batch_input})[0]

#         # for i, emb in enumerate(embeddings):
#         #     emb = emb.flatten()
#         #     emb /= (np.linalg.norm(emb) + 1e-8)
#         #     bbox, tracker_id, conf = crop_meta[i]

#         #     if tracker_id in track_to_customer:
#         #         customer_id = track_to_customer[tracker_id]
#         #         if fast_min_dist_to_customer(emb, customer_id) > settings.DIVERSITY_THRESHOLD:
#         #             db_queue.put(("store_embedding", customer_id, cam_id, emb, datetime.now()))
#         #     else:
#         #         # Atomic match‑or‑create via the writer
#         #         request_id = f"{cam_id}_{tracker_id}_{time.time()}"
#         #         db_queue.put(("match_or_register", emb, cam_id, datetime.now(), request_id))
#         #         while True:
#         #             rid, customer_id, is_new = response_queue.get()
#         #             if rid == request_id:
#         #                 break
#         #             response_queue.put((rid, customer_id, is_new))

#         #         track_to_customer[tracker_id] = customer_id
#         #         if not is_new and fast_min_dist_to_customer(emb, customer_id) > settings.DIVERSITY_THRESHOLD:
#         #             db_queue.put(("store_embedding", customer_id, cam_id, emb, datetime.now()))

#         #     labels.append(f"#{int(tracker_id)} {conf:.2f} ID:{customer_id}")












#         # for i, emb in enumerate(embeddings):
#         #     emb = emb.flatten()
#         #     emb = emb / (np.linalg.norm(emb) + 1e-8)
#         #     bbox, tracker_id, conf = crop_meta[i]
#         #     with open ('data/HHHHHHHHHHHH.txt', 'a+') as f:
#         #         if tracker_id in track_to_customer:
#         #             customer_id = track_to_customer[tracker_id]
#         #             f.write(f"{tracker_id} ===> {customer_id} ===> {fast_min_dist_to_customer(emb, customer_id)}\n")
#         #             if fast_min_dist_to_customer(emb, customer_id) > settings.DIVERSITY_THRESHOLD:
#         #                 db_queue.put(("store_embedding", customer_id, cam_id, emb, datetime.now()))
#         #                 f.write(f"11111  store_embedding: TrackerId:{tracker_id} ===> CustomerId:{customer_id}  @@@ {fast_min_dist_to_customer(emb, customer_id)}\n")
#         #             else:
#         #                 f.write(f"22222  store_embedding: TrackerId:{tracker_id} ===> CustomerId:{customer_id}  @@@ {fast_min_dist_to_customer(emb, customer_id)}\n")
#         #         else:
#         #             # Atomic match‑or‑create via the writer
#         #             request_id = f"{cam_id}_{tracker_id}_{time.time()}"
#         #             f.write(f"match_or_register:  CamID: {cam_id} ===> TrackerId:{tracker_id}\n")
#         #             db_queue.put(("match_or_register", emb, cam_id, datetime.now(), request_id))
#         #             while True:
#         #                 rid, customer_id, is_new, match_dist = response_queue.get()
#         #                 if rid == request_id:
#         #                     break
#         #                 response_queue.put((rid, customer_id, is_new, match_dist))

#         #             track_to_customer[tracker_id] = customer_id
#         #             # if not is_new and fast_min_dist_to_customer(emb, customer_id) > settings.DIVERSITY_THRESHOLD:
#         #             if not is_new and match_dist > 0.68:
#         #                 db_queue.put(("store_embedding", customer_id, cam_id, emb, datetime.now()))

#         #         labels.append(f"#{int(tracker_id)} {conf:.2f} ID:{customer_id}")
#         #     f.close()
        
        
        
        
        
#         for i, emb in enumerate(embeddings):
#             emb = emb.flatten()
#             emb = emb / (np.linalg.norm(emb) + 1e-8)
#             bbox, tracker_id, conf = crop_meta[i]

#             if tracker_id in track_to_customer:
#                 customer_id = track_to_customer[tracker_id]
#                 # Ask the writer for the minimum distance to this customer
#                 request_id = f"dist_{cam_id}_{tracker_id}_{time.time()}"
#                 db_queue.put(("min_dist_to_customer", emb, customer_id, request_id))
#                 while True:
#                     rid, min_dist = response_queue.get()
#                     if rid == request_id:
#                         break
#                     response_queue.put((rid, min_dist))

#                 # Store only if the embedding is diverse
#                 if min_dist > settings.DIVERSITY_THRESHOLD:
#                     db_queue.put(("store_embedding", customer_id, cam_id, emb, datetime.now()))
#                 else:
#                     db_queue.put(("update_customer_last_seen", customer_id, datetime.now()))
                
#             else:
#                 # Unknown tracker – match or register atomically
#                 request_id = f"match_{cam_id}_{tracker_id}_{time.time()}"
#                 db_queue.put(("match_or_register", emb, cam_id, datetime.now(), request_id))
#                 while True:
#                     rid, customer_id, is_new, match_dist = response_queue.get()
#                     if rid == request_id:
#                         break
#                     response_queue.put((rid, customer_id, is_new, match_dist))

#                 track_to_customer[tracker_id] = customer_id
#                 # For existing matches, store diverse embeddings
#                 if not is_new and match_dist > settings.DIVERSITY_THRESHOLD:
#                     db_queue.put(("store_embedding", customer_id, cam_id, emb, datetime.now()))
#                 else:
#                     db_queue.put(("update_customer_last_seen", customer_id, datetime.now()))

#             labels.append(f"#{int(tracker_id)} {conf:.2f} ID:{customer_id}")


#     # ── 5. Annotate a clean copy ──
#     annotated = frame.copy()
#     # Traces, boxes, labels – only if we have detections
#     if len(detections) > 0:
#         annotated = trace_annotator.annotate(annotated, detections)
#         annotated = box_annotator.annotate(annotated, detections)
#         annotated = label_annotator.annotate(annotated, detections, labels)

#     # FPS counter
#     fps_monitor.tick()
#     annotated = sv.draw_text(
#         scene=annotated,
#         text=f"FPS: {fps_monitor.fps:.1f}",
#         text_anchor=sv.Point(x=20, y=40),
#         text_color=sv.Color.WHITE,
#         text_scale=0.7,
#         text_thickness=2,
#         background_color=sv.Color.BLACK
#     )

#     return annotated

# def detector_worker(
#     shm_name,
#     frame_shape,
#     frame_bytes,
#     ready_slots,
#     det_queue,
#     free_slots,
#     stop_event
# ):
#     pin_process([1,2,3])
#     shm = shared_memory.SharedMemory(name=shm_name)
#     detector_model = get_detector()
#     while not stop_event.is_set():
#         try:
#             idx = ready_slots.get(timeout=0.1)
#         except queue.Empty:
#             continue

#         if idx is None:
#             break

#         frame = frame_view(shm,frame_shape,frame_bytes,idx)
#         detections = detector_model.predict(frame)
#         try:
#             det_queue.put_nowait((
#             idx,
#             detections.xyxy,
#             detections.confidence,
#             detections.class_id
#         ))
#         except queue.Full:
#             try:
#                 free_slots.put_nowait(idx)
#             except:
#                 pass
#     shm.close()

# # How the backoff works:
# # First failure → wait 1 s, then retry.
# # Second failure → wait 2 s, then retry.
# # Third failure → wait 4 s, then retry.
# # … up to a maximum of 60 s.
# # As soon as a frame is read successfully, consecutive_failures resets to 0.
# # During the wait, the thread is sleeping – it consumes almost zero CPU.
# def reader_worker(rtsp_url,
#                   shm_name,
#                   frame_shape, 
#                   frame_bytes,
#                   free_slots,
#                   ready_slots, 
#                   stop_event):
#     pin_process([0])
#     shm = shared_memory.SharedMemory(name=shm_name)

#     cap = None
#     first_attempt = True          # <-- ADD THIS
#     online = False
#     consecutive_failures = 0
#     max_backoff = 60.0            # never sleep longer than 60 seconds
#     offline_frame = np.zeros(frame_shape, dtype=np.uint8)
#     # Draw the offline message on the placeholder (optional)
#     cv2.putText(offline_frame, "Stream offline", (50, 240),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     while not stop_event.is_set():
#         if not online:
#             # Release any old capture and wait before reconnecting
#             if cap is not None:
#                 cap.release()
#                 cap = None
#             # Exponential backoff: 1s, 2s, 4s, 8s, ... capped at max_backoff
#             if first_attempt and consecutive_failures < 6:  # <-- no sleep on the first try
#                 first_attempt = False
#                 sleep_time = 0
#             else:
#                 sleep_time = min(1.0 * (2 ** consecutive_failures), max_backoff)

#             stop_event.wait(sleep_time)   # wait but break if stop_event is set
#             if stop_event.is_set():
#                 break
#             # Attempt to open the stream
#             cap = cv2.VideoCapture(rtsp_url)
#             if not cap.isOpened():
#                 consecutive_failures += 1
#                 # Push an offline placeholder into one slot so the user sees it
#                 _push_placeholder_frame(shm, frame_shape, frame_bytes,
#                                         free_slots, ready_slots, offline_frame)
#                 continue
#             online = True
#             consecutive_failures = 0

#         # Stream is open – try to read
#         ret, frame = cap.read()
#         if not ret:
#             # Stream died – switch back to offline mode
#             online = False
#             consecutive_failures += 1
#             _push_placeholder_frame(shm, frame_shape, frame_bytes,
#                                     free_slots, ready_slots, offline_frame)
#             continue

#         # Live frame received – push it into the ring buffer
#         consecutive_failures = 0
#         try:
#             idx = free_slots.get_nowait()
#         except queue.Empty:
#             continue   # ring is full, drop frame (normal back‑pressure)
#         _write_frame_to_slot(shm, idx, frame, frame_shape, frame_bytes)
#         try:
#             ready_slots.put_nowait(idx)
#         except queue.Full:
#             try:
#                 free_slots.put_nowait(idx)
#             except:
#                 pass

#     # Cleanup
#     if cap is not None:
#         cap.release()
#     shm.close()

# def _push_placeholder_frame(shm, frame_shape, frame_bytes,
#                             free_slots, ready_slots, placeholder):
#     """Push a single offline frame into the ring if a slot is available."""
#     try:
#         idx = free_slots.get_nowait()
#     except queue.Empty:
#         return   # ring is full – nothing to do
#     _write_frame_to_slot(shm, idx, placeholder, frame_shape, frame_bytes)
#     try:
#         ready_slots.put_nowait(idx)
#     except queue.Full:
#         try:
#             free_slots.put_nowait(idx)
#         except:
#             pass

# def embedder_worker(
#     input_shm_name,
#     output_shm_name,
#     frame_shape,
#     frame_bytes,
#     det_queue,
#     free_slots,
#     stop_event,
#     db_queue,
#     response_queue,
#     cam_id
# ):
#     pin_process([4,5])
#     input_shm = shared_memory.SharedMemory(name=input_shm_name)
#     output_shm = shared_memory.SharedMemory(name=output_shm_name)
#     embedder_session = create_session(
#         settings.FEATURE_EXTRACTOR_MODEL, num_threads=1)

#     track_to_customer = {}
#     tracker = sv.ByteTrack()
#     fps_monitor = sv.FPSMonitor()
#     color = sv.ColorPalette.DEFAULT
#     box_annotator = sv.EllipseAnnotator(color=color)
#     trace_annotator = sv.TraceAnnotator(color=color, trace_length=30)
#     label_annotator = sv.LabelAnnotator(color=color, text_color=sv.Color.BLACK)
    
#     colors = sv.ColorPalette.DEFAULT
#     polygons = [
#         np.array([
#             [0, 0],
#             [1800 - 5, 0],
#             [1800 - 5, 1800 - 5],
#             [1080 + 5, 1800 - 5]
#         ], np.int32),
#         # np.array([
#         #     [1080 + 5, 0],
#         #     [2160, 0],
#         #     [2160, 1300 - 5],
#         #     [1080 + 5, 1300 - 5]
#         # ], np.int32),
#     ]

#     zones = [sv.PolygonZone(polygon=polygon) for polygon in polygons]
    
#     zone_annotators = [
#         sv.PolygonZoneAnnotator(
#             zone=zone,
#             color=colors.by_idx(index),
#             thickness=4,
#             text_thickness=8,
#             text_scale=4
#         ) for index, zone in enumerate(zones)]
     
#     zone_box_annotators = [
#         sv.BoxAnnotator(
#             color=colors.by_idx(index),
#             thickness=4,
#         ) for index in range(len(polygons))]
    
#     # dwell time tracking
#     dwell_start = {}   # tracker_id -> enter_time
#     dwell_total = defaultdict(float)  # tracker_id -> total accumulated seconds
#     while not stop_event.is_set():
#         try:
#             item = det_queue.get(timeout=0.1)
#         except queue.Empty:
#             continue
        
#         if item is None:
#             break
        
#         idx, xyxy, confidence, class_id = item
#         detections = sv.Detections(
#             xyxy=xyxy,
#             confidence=confidence,
#             class_id=class_id
#         )
#         # --- Tick the FPS counter ---
#         # fps_monitor.tick()
#         track_positions = defaultdict(list)   # tracker_id -> list of (time, centroid)
#         frame = frame_view(input_shm,frame_shape,frame_bytes,idx)
#         unique_visitors = set()
#         annotated = process_frame(
#             frame=frame,
#             detections=detections,
#             tracker=tracker,
#             embedder_session=embedder_session,
#             track_to_customer=track_to_customer,
#             db_queue=db_queue,
#             cam_id=cam_id,
#             unique_visitors=unique_visitors,
#             zones=zones,
#             zone_annotators=zone_annotators,
#             dwell_start=dwell_start,
#             dwell_total=dwell_total,
#             fps_monitor = fps_monitor,
#             box_annotator=box_annotator,
#             zone_box_annotators=zone_box_annotators,
#             track_positions=track_positions,
#             response_queue=response_queue,
#             trace_annotator=trace_annotator,
#             label_annotator=label_annotator
#         )

#         # WRITE DIRECTLY TO OUTPUT SHM
#         output_frame = np.ndarray(
#             frame_shape,
#             dtype=np.uint8,
#             buffer=output_shm.buf
#         )
#         np.copyto(output_frame, annotated)
#         try:
#             free_slots.put_nowait(idx)
#         except:
#             pass

#     input_shm.close()
#     output_shm.close()



















































# import os
# import cv2
# import time
# import queue
# import psutil
# import numpy as np
# import supervision as sv
# import onnxruntime as ort
# from loguru import logger
# from datetime import datetime
# from collections import defaultdict
# from src.core.config import settings
# from src.vision.factory import get_detector
# from src.core.database import  fast_min_dist_to_customer
# from multiprocessing import shared_memory


# def pin_process(cores):
#     try:
#         psutil.Process(os.getpid()).cpu_affinity(list(cores))
#     except Exception:
#         pass

# def frame_view(shm: shared_memory.SharedMemory, frame_shape, frame_bytes: int, idx: int):
#     offset = idx * frame_bytes
#     return np.ndarray(frame_shape, dtype=np.uint8, buffer=shm.buf, offset=offset)

# def create_session(model_path, num_threads=2):
#     sess_options = ort.SessionOptions()
#     sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
#     sess_options.enable_mem_pattern = True
#     sess_options.enable_cpu_mem_arena = True
#     sess_options.intra_op_num_threads = num_threads
#     sess_options.inter_op_num_threads = 2
#     sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
#     sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
#     available = ort.get_available_providers()
#     providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
#     return ort.InferenceSession(model_path, 
#                                 sess_options=sess_options,
#                                 providers=providers)
    
# # shared memory write helper
# def _write_frame_to_slot(shm, idx, frame, frame_shape, frame_bytes):
#     dst = frame_view(shm, frame_shape, frame_bytes, idx)
#     if frame.shape != frame_shape:
#         frame = cv2.resize(frame, (frame_shape[1], frame_shape[0]))
#     np.copyto(dst, frame)
    
# def preprocess_crop(frame, bbox, torso_ratio, model_input_size):
#     x1, y1, x2, y2 = map(int, bbox)
#     w, h = x2 - x1, y2 - y1
#     cx, cy = (x1+x2)/2, (y1+y2)/2 
#     crop_y2 = int(y1 + h * torso_ratio)
#     crop = frame[y1:crop_y2, x1:x2]
#     flag =  (crop.size == 0) or (w < 20) or (h < 20) or (h*w < 12000)

#     # Preprocess
#     resized = cv2.resize(crop, (model_input_size[1], model_input_size[0]),
#                         interpolation=cv2.INTER_AREA)
#     # Normalize: (img / 255 - mean) / std
#     mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
#     std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
#     normalized = (resized / 255.0 - mean) / std
    
#     # Convert to CHW
#     input_tensor = np.transpose(normalized, (2, 0, 1)).astype(np.float32)
#     return input_tensor, (x1, y1, x2, y2), (cx, cy), flag

# class VisionPipeline:
#     def __init__(self,
#                 RTSP_URL,
#                 CAM_ID,
#                 ctx,
#                 free_slots,
#                 ready_slots,
#                 det_queue,
#                 stop_event,
#                 db_queue,
#                 response_queue,
#                 buffer_slots=4):
        
#         self.model_input_size = (256, 128)
#         self.ctx = ctx
#         self.RTSP_URL = RTSP_URL
#         self.cam_id = CAM_ID
#         self.torso_ratio = 2/3
#         self.free_slots = free_slots
#         self.ready_slots= ready_slots
#         self.det_queue  = det_queue
#         self.stop_event = stop_event
#         self.db_queue   = db_queue
#         self.response_queue = response_queue
#         self.buffer_slots = buffer_slots
        
#         probe = cv2.VideoCapture(self.RTSP_URL)
#         ok, frame = probe.read()
#         probe.release()
#         # Class‑level constant – adjust as you like
#         WORKING_HEIGHT = 512
#         WORKING_WIDTH  = 512
#         WORKING_CHANNELS = 3
        
#         if not ok:
#             self.online = False
#             # Fixed frame shape – never changes
#             self.frame_shape = (WORKING_HEIGHT, WORKING_WIDTH, WORKING_CHANNELS)
#             self.frame_bytes = WORKING_HEIGHT * WORKING_WIDTH * WORKING_CHANNELS
#         else:
#             self.frame_shape = frame.shape
#             self.frame_bytes = frame.nbytes
#             self.online = True
    
#         # INPUT SHM
#         self.input_shm = shared_memory.SharedMemory(
#             create=True,
#             size=self.frame_bytes * buffer_slots)
#         self.input_shm_name = self.input_shm.name

#         # OUTPUT SHM (ONLY ONE FRAME)
#         self.output_shm = shared_memory.SharedMemory(
#             create=True,
#             size=self.frame_bytes)

#         self.output_shm_name = self.output_shm.name
#         for i in range(buffer_slots):
#             self.free_slots.put(i)

#     def start(self):
#         self.p_reader = self.ctx.Process(
#             target=reader_worker,
#             args=(
#                 self.RTSP_URL,
#                 self.input_shm_name,
#                 self.frame_shape,
#                 self.frame_bytes,
#                 self.free_slots,
#                 self.ready_slots,
#                 self.stop_event
#             ),
#             daemon=True
#         )

#         self.p_detector = self.ctx.Process(
#             target=detector_worker,
#             args=(
#                 self.input_shm_name,
#                 self.frame_shape,
#                 self.frame_bytes,
#                 self.ready_slots,
#                 self.det_queue,
#                 self.free_slots,
#                 self.stop_event
#             ),
#             daemon=True
#         )

#         self.p_embedder = self.ctx.Process(
#             target=embedder_worker,
#             args=(
#                 self.input_shm_name,
#                 self.output_shm_name,
#                 self.frame_shape,
#                 self.frame_bytes,
#                 self.det_queue,
#                 self.free_slots,
#                 self.stop_event,
#                 self.db_queue,
#                 self.response_queue,
#                 self.cam_id
#             ),
#             daemon=True
#         )

#         self.p_reader.start()
#         self.p_detector.start()
#         self.p_embedder.start()
        
#     def stop(self):
#         self.stop_event.set()
#         try:
#             self.ready_slots.put_nowait(None)
#         except:
#             pass

#         try:
#             self.det_queue.put_nowait(None)
#         except:
#             pass

#         self.p_reader.join(timeout=2)
#         self.p_detector.join(timeout=2)
#         self.p_embedder.join(timeout=2)

#         self.input_shm.close()
#         self.input_shm.unlink()

#         self.output_shm.close()
#         self.output_shm.unlink()

#     def get_latest_frame(self):
#         frame = np.ndarray(
#             self.frame_shape,
#             dtype=np.uint8,
#             buffer=self.output_shm.buf
#         )
#         return frame.copy()
    
# def process_frame(
#     frame,
#     detections,
#     tracker,
#     embedder_session,
#     track_to_customer,
#     db_queue,
#     cam_id,
#     box_annotator,
#     fps_monitor,
#     trace_annotator,
#     label_annotator,
#     response_queue,
#     diversity_threshold=0.2,
#     person_class_id=0
# ):

#     if detections.class_id is not None:
#         detections = detections[
#             detections.class_id == person_class_id
#         ]

#     detections = tracker.update_with_detections(detections)

#     crops_onnx = []
#     crop_meta = []
#     labels = []

#     for det in detections:
#         det_box, mask, confidence, class_id, tracker_id, data = det
#         if tracker_id is None:
#             continue
        
#         input_tensor, crop_box, center_point, crop_flag = preprocess_crop(
#             frame,
#             det_box,
#             model_input_size=(128, 64),
#             torso_ratio=2/3,
#         )

#         if crop_flag:
#             continue

#         crops_onnx.append(input_tensor)
#         crop_meta.append((crop_box, int(tracker_id), confidence))

#     if not crops_onnx:
#         return frame

#     batch_input = np.stack(crops_onnx, axis=0)
#     embeddings = embedder_session.run(None, {"input": batch_input})[0]

#     for i, emb in enumerate(embeddings):
#         emb = emb.flatten()
#         emb /= (np.linalg.norm(emb) + 1e-8)
#         bbox, tracker_id, confidence_score = crop_meta[i]
#         if tracker_id in track_to_customer:
#             customer_id = track_to_customer[tracker_id]
#             if fast_min_dist_to_customer(emb, customer_id) > diversity_threshold:
#                 db_queue.put(("store_embedding", customer_id, cam_id, emb, datetime.now()))
#         else:
#             # Ask the writer to match or create – atomic, no race conditions
#             request_id = f"{cam_id}_{tracker_id}_{time.time()}"
#             db_queue.put((
#                 "match_or_register",
#                 emb,
#                 cam_id,
#                 datetime.now(),
#                 request_id,
#                 center_point, 
#                 tracker_id, 
#                 confidence_score      
#             ))
            
#             # Wait for the reply that carries our request_id
#             while True:
#                 rid, customer_id, is_new = response_queue.get()
#                 if rid == request_id:
#                     break
#                 # Not ours – put it back for other consumers (if any)
#                 response_queue.put((rid, customer_id, is_new))

#             track_to_customer[tracker_id] = customer_id

#             # Optionally store a diverse embedding if the person already existed
#             if not is_new:
#                 if fast_min_dist_to_customer(emb, customer_id) > diversity_threshold:
#                     db_queue.put(("store_embedding", customer_id, cam_id, emb, datetime.now()))

#         labels.append(
#             f"#{int(tracker_id)} {confidence:.2f} "
#             f"#{tracker_id} ID:{customer_id}"
#         )
#     annotated = frame.copy()
#     annotated = trace_annotator.annotate(annotated,detections)
#     annotated = box_annotator.annotate(annotated,detections)
#     annotated = label_annotator.annotate(annotated,detections,labels)
    
#     # Tick every frame
#     fps_monitor.tick()

#     # Draw the FPS value on the frame
#     annotated = sv.draw_text(
#         scene=annotated,
#         text=f"FPS: {fps_monitor.fps:.1f}",
#         text_anchor=sv.Point(x=20, y=40),
#         text_color=sv.Color.WHITE,
#         text_scale=0.7,
#         text_thickness=2,
#         background_color=sv.Color.BLACK
#     )
#     return annotated

# def detector_worker(
#     shm_name,
#     frame_shape,
#     frame_bytes,
#     ready_slots,
#     det_queue,
#     free_slots,
#     stop_event
# ):
#     pin_process([1,2,3])
#     shm = shared_memory.SharedMemory(name=shm_name)
#     detector_model = get_detector()
#     while not stop_event.is_set():
#         try:
#             idx = ready_slots.get(timeout=0.1)
#         except queue.Empty:
#             continue

#         if idx is None:
#             break

#         frame = frame_view(shm,frame_shape,frame_bytes,idx)
#         detections = detector_model.predict(frame)
#         try:
#             det_queue.put_nowait((
#             idx,
#             detections.xyxy,
#             detections.confidence,
#             detections.class_id
#         ))
#         except queue.Full:
#             try:
#                 free_slots.put_nowait(idx)
#             except:
#                 pass
#     shm.close()

# # How the backoff works:
# # First failure → wait 1 s, then retry.
# # Second failure → wait 2 s, then retry.
# # Third failure → wait 4 s, then retry.
# # … up to a maximum of 60 s.
# # As soon as a frame is read successfully, consecutive_failures resets to 0.
# # During the wait, the thread is sleeping – it consumes almost zero CPU.
# def reader_worker(rtsp_url,
#                   shm_name,
#                   frame_shape, 
#                   frame_bytes,
#                   free_slots,
#                   ready_slots, 
#                   stop_event):
#     pin_process([0])
#     shm = shared_memory.SharedMemory(name=shm_name)

#     cap = None
#     first_attempt = True          # <-- ADD THIS
#     online = False
#     consecutive_failures = 0
#     max_backoff = 60.0            # never sleep longer than 60 seconds
#     offline_frame = np.zeros(frame_shape, dtype=np.uint8)
#     # Draw the offline message on the placeholder (optional)
#     cv2.putText(offline_frame, "Stream offline", (50, 240),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     while not stop_event.is_set():
#         if not online:
#             # Release any old capture and wait before reconnecting
#             if cap is not None:
#                 cap.release()
#                 cap = None
#             # Exponential backoff: 1s, 2s, 4s, 8s, ... capped at max_backoff
#             if first_attempt and consecutive_failures < 6:  # <-- no sleep on the first try
#                 first_attempt = False
#                 sleep_time = 0
#             else:
#                 sleep_time = min(1.0 * (2 ** consecutive_failures), max_backoff)

#             stop_event.wait(sleep_time)   # wait but break if stop_event is set
#             if stop_event.is_set():
#                 break
#             # Attempt to open the stream
#             cap = cv2.VideoCapture(rtsp_url)
#             if not cap.isOpened():
#                 consecutive_failures += 1
#                 # Push an offline placeholder into one slot so the user sees it
#                 _push_placeholder_frame(shm, frame_shape, frame_bytes,
#                                         free_slots, ready_slots, offline_frame)
#                 continue
#             online = True
#             consecutive_failures = 0

#         # Stream is open – try to read
#         ret, frame = cap.read()
#         if not ret:
#             # Stream died – switch back to offline mode
#             online = False
#             consecutive_failures += 1
#             _push_placeholder_frame(shm, frame_shape, frame_bytes,
#                                     free_slots, ready_slots, offline_frame)
#             continue

#         # Live frame received – push it into the ring buffer
#         consecutive_failures = 0
#         try:
#             idx = free_slots.get_nowait()
#         except queue.Empty:
#             continue   # ring is full, drop frame (normal back‑pressure)
#         _write_frame_to_slot(shm, idx, frame, frame_shape, frame_bytes)
#         try:
#             ready_slots.put_nowait(idx)
#         except queue.Full:
#             try:
#                 free_slots.put_nowait(idx)
#             except:
#                 pass

#     # Cleanup
#     if cap is not None:
#         cap.release()
#     shm.close()

# def _push_placeholder_frame(shm, frame_shape, frame_bytes,
#                             free_slots, ready_slots, placeholder):
#     """Push a single offline frame into the ring if a slot is available."""
#     try:
#         idx = free_slots.get_nowait()
#     except queue.Empty:
#         return   # ring is full – nothing to do
#     _write_frame_to_slot(shm, idx, placeholder, frame_shape, frame_bytes)
#     try:
#         ready_slots.put_nowait(idx)
#     except queue.Full:
#         try:
#             free_slots.put_nowait(idx)
#         except:
#             pass

# def embedder_worker(
#     input_shm_name,
#     output_shm_name,
#     frame_shape,
#     frame_bytes,
#     det_queue,
#     free_slots,
#     stop_event,
#     db_queue,
#     response_queue,
#     cam_id
# ):
#     pin_process([4,5])
#     input_shm = shared_memory.SharedMemory(name=input_shm_name)
#     output_shm = shared_memory.SharedMemory(name=output_shm_name)
#     embedder_session = create_session(
#         settings.FEATURE_EXTRACTOR_MODEL, num_threads=1)

#     track_to_customer = {}
#     tracker = sv.ByteTrack()
#     fps_monitor = sv.FPSMonitor()
#     color = sv.ColorPalette.DEFAULT
#     box_annotator = sv.BoxAnnotator(color=color)
#     trace_annotator = sv.TraceAnnotator(color=color,trace_length=30)
#     label_annotator = sv.LabelAnnotator(color=color,text_color=sv.Color.BLACK)
#     zone_polygon = np.array([[100, 200], [400, 200], [400, 500], [100, 500]])
#     zone = sv.PolygonZone(polygon=zone_polygon)
#     zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.GREEN)
#     # dwell time tracking
#     dwell_start = {}   # tracker_id -> enter_time
#     dwell_total = defaultdict(float)  # tracker_id -> total accumulated seconds
#     while not stop_event.is_set():
#         try:
#             item = det_queue.get(timeout=0.1)
#         except queue.Empty:
#             continue
        
#         if item is None:
#             break
        
#         idx, xyxy, confidence, class_id = item
#         detections = sv.Detections(
#             xyxy=xyxy,
#             confidence=confidence,
#             class_id=class_id
#         )
#         # --- Tick the FPS counter ---
#         fps_monitor.tick()

#         frame = frame_view(input_shm,frame_shape,frame_bytes,idx)
#         annotated = process_frame(
#             frame=frame,
#             detections=detections,
#             tracker=tracker,
#             embedder_session=embedder_session,
#             track_to_customer=track_to_customer,
#             db_queue=db_queue,
#             cam_id=cam_id,
#             zone_annotator=zone_annotator,
#             dwell_start=dwell_start,
#             dwell_total=dwell_total,
#             fps_monitor = fps_monitor,
#             box_annotator=box_annotator,
#             response_queue=response_queue,
#             trace_annotator=trace_annotator,
#             label_annotator=label_annotator
#         )

#         # WRITE DIRECTLY TO OUTPUT SHM
#         output_frame = np.ndarray(
#             frame_shape,
#             dtype=np.uint8,
#             buffer=output_shm.buf
#         )
#         np.copyto(output_frame, annotated)
#         try:
#             free_slots.put_nowait(idx)
#         except:
#             pass

#     input_shm.close()
#     output_shm.close()











































import os
import cv2
import time
import queue
import psutil
import numpy as np
import supervision as sv
import onnxruntime as ort
from loguru import logger
from datetime import datetime
from collections import defaultdict
from src.core.config import settings
from src.vision.factory import get_detector
from src.core.database import  fast_min_dist_to_customer
from multiprocessing import shared_memory


def pin_process(cores):
    try:
        psutil.Process(os.getpid()).cpu_affinity(list(cores))
    except Exception:
        pass

def frame_view(shm: shared_memory.SharedMemory, frame_shape, frame_bytes: int, idx: int):
    offset = idx * frame_bytes
    return np.ndarray(frame_shape, dtype=np.uint8, buffer=shm.buf, offset=offset)

def create_session(model_path, num_threads=2):
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.enable_mem_pattern = True
    sess_options.enable_cpu_mem_arena = True
    sess_options.intra_op_num_threads = num_threads
    sess_options.inter_op_num_threads = 2
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    available = ort.get_available_providers()
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
    return ort.InferenceSession(model_path, 
                                sess_options=sess_options,
                                providers=providers)
    
# shared memory write helper
def _write_frame_to_slot(shm, idx, frame, frame_shape, frame_bytes):
    dst = frame_view(shm, frame_shape, frame_bytes, idx)
    if frame.shape != frame_shape:
        frame = cv2.resize(frame, (frame_shape[1], frame_shape[0]))
    np.copyto(dst, frame)

def letterbox(img, input_size) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Resize and pad image to target size while preserving aspect ratio.
    Returns:
        - padded image (np.ndarray) of shape (target_h, target_w, 3)
        - pad (top, left) amounts used for padding (needed to map boxes back).
    """
    shape = img.shape[:2]                     # original (height, width)
    target_h, target_w = input_size
    r = min(target_h / shape[0], target_w / shape[1])
    new_unpad = (round(shape[1] * r), round(shape[0] * r))  # (width, height)
    dw, dh = (target_w - new_unpad[0]) / 2, (target_h - new_unpad[1]) / 2

    # Resize only if needed
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Pad with constant gray (114,114,114)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, (top, left)

# def preprocess_crop(frame, bbox, torso_ratio, model_input_size):
#     """
#     Preprocess the input image: BGR->RGB, letterbox, normalize, CHW, batch.
#     Returns:
#         - image_data: (1, 3, H, W) float32 numpy array (values 0..1)
#         - pad: (top, left) used for padding
#     """
#     x1, y1, x2, y2 = map(int, bbox)
#     w, h = x2 - x1, y2 - y1
#     cx, cy = (x1+x2)/2, (y1+y2)/2 
#     # crop_y2 = int(y1 + h * torso_ratio)
#     # frame = frame[y1:crop_y2, x1:x2]
#     flag =  frame.size == 0 or w < 20 or h < 20
#     img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     img_padded, pad = letterbox(img_rgb, model_input_size)
#     # img_norm = img_padded.astype(np.float32) / 255.0
#     mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
#     std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
#     img_norm = (img_padded / 255.0 - mean) / std
#     img_chw = np.transpose(img_norm, (2, 0, 1)).astype(np.float32)        # (C, H, W)
#     return img_chw, (x1, y1, x2, y2), (cx, cy), w, h, flag


def preprocess_crop(frame, bbox, torso_ratio, model_input_size):
    x1, y1, x2, y2 = map(int, bbox)
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1+x2)/2, (y1+y2)/2 
    crop_y2 = int(y1 + h * torso_ratio)
    crop = frame[y1:crop_y2, x1:x2]
    flag =  (crop.size == 0) or (w < 20) or (h < 20)
    # or (h*w < 12000)

    # Preprocess
    resized = cv2.resize(crop, (model_input_size[1], model_input_size[0]),
                        interpolation=cv2.INTER_AREA)
    # Normalize: (img / 255 - mean) / std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (resized / 255.0 - mean) / std
    
    # Convert to CHW
    input_tensor = np.transpose(normalized, (2, 0, 1)).astype(np.float32)
    return input_tensor, (x1, y1, x2, y2), (cx, cy), w, h, flag



class VisionPipeline:
    def __init__(self,
                RTSP_URL,
                CAM_ID,
                ctx,
                free_slots,
                ready_slots,
                det_queue,
                stop_event,
                db_queue,
                response_queue,
                buffer_slots=4):
        
        self.model_input_size = (256, 128)
        self.ctx = ctx
        self.RTSP_URL = RTSP_URL
        self.cam_id = CAM_ID
        self.torso_ratio = 2/3
        self.free_slots = free_slots
        self.ready_slots= ready_slots
        self.det_queue  = det_queue
        self.stop_event = stop_event
        self.db_queue   = db_queue
        self.response_queue = response_queue
        self.buffer_slots = buffer_slots
        self.track_positions = defaultdict(list)   # tracker_id -> list of (time, centroid)
        
        probe = cv2.VideoCapture(self.RTSP_URL)
        ok, frame = probe.read()
        probe.release()
        # Class‑level constant – adjust as you like
        WORKING_HEIGHT = 512
        WORKING_WIDTH  = 512
        WORKING_CHANNELS = 3
        
        if not ok:
            self.online = False
            # Fixed frame shape – never changes
            self.frame_shape = (WORKING_HEIGHT, WORKING_WIDTH, WORKING_CHANNELS)
            self.frame_bytes = WORKING_HEIGHT * WORKING_WIDTH * WORKING_CHANNELS
        else:
            self.frame_shape = frame.shape
            self.frame_bytes = frame.nbytes
            self.online = True
    
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
        self.p_reader = self.ctx.Process(
            target=reader_worker,
            args=(
                self.RTSP_URL,
                self.input_shm_name,
                self.frame_shape,
                self.frame_bytes,
                self.free_slots,
                self.ready_slots,
                self.det_queue,
                self.stop_event
            ),
            daemon=True
        )

        self.p_detector = self.ctx.Process(
            target=detector_worker,
            args=(
                self.input_shm_name,
                self.frame_shape,
                self.frame_bytes,
                self.ready_slots,
                self.det_queue,
                self.free_slots,
                self.stop_event
            ),
            daemon=True
        )

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
                self.track_positions,                
                self.cam_id
            ),
            daemon=True
        )

        self.p_reader.start()
        self.p_detector.start()
        self.p_embedder.start()
        
    def stop(self):
        self.stop_event.set()
        try:
            self.ready_slots.put_nowait(None)
        except:
            pass

        try:
            self.det_queue.put_nowait(None)
        except:
            pass

        self.p_reader.join(timeout=2)
        self.p_detector.join(timeout=2)
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
    # ── 0. Guard against None or completely empty detections ──
    if detections is None or len(detections) == 0:
        print('# Still draw the zone outline and FPS on a clean copy')
        # Still draw the zone outline and FPS on a clean copy
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

    # ── Zone analytics (once per frame, per zone) ──
    # (Make sure dwell_start, dwell_total are dicts keyed by zone_id, each holding a dict tracker_id -> value)
    for zone_idx, (zone, zone_annotator, box_annotator) in enumerate(zip(zones, zone_annotators, zone_box_annotators)):
        mask = zone.trigger(detections=detections)
        if isinstance(mask, np.ndarray):
            detections_in_zone = detections[mask]
        else:
            detections_in_zone = mask

        inside_now = set(tid for tid in detections_in_zone.tracker_id if tid is not None)

        # Update dwell start times
        zone_dwell_start = dwell_start.setdefault(zone_idx, {})
        zone_dwell_total = dwell_total.setdefault(zone_idx, defaultdict(float))

        for tid in inside_now:
            if tid not in zone_dwell_start:
                zone_dwell_start[tid] = time.time()

        # Check who left
        for tid in list(zone_dwell_start.keys()):
            if tid not in inside_now:
                elapsed = time.time() - zone_dwell_start.pop(tid)
                zone_dwell_total[tid] += elapsed
                print(f"Customer {tid} left zone {zone_idx} after {elapsed:.1f}s")

        # Draw zone and its boxes
        frame = box_annotator.annotate(scene=frame, detections=detections_in_zone)
        frame = zone_annotator.annotate(scene=frame)

    # ── Loitering, unique visitors, crop collection ──
    crops_onnx, crop_meta, labels = [], [], []
    loitering_tracker_ids = set()
    if run_embedding:
        for det in detections:
            det_box, det_mask, det_conf, class_id, tracker_id, data = det
            if tracker_id is None:
                continue

            # Crop preprocessing
            input_tensor, crop_box, center_point, bbox_w, bbox_h, crop_flag = preprocess_crop(
                frame, det_box,
                # model_input_size=(128, 64),
                model_input_size=(256, 128),
                # torso_ratio=2/3
                torso_ratio=1
            )
            
            # Loitering check
            if check_loitering(tracker_id, time.time(), center_point, track_positions):
                loitering_tracker_ids.add(tracker_id)
            

            if not crop_flag:
                # Unique visitor counting
                unique_visitors.add(tracker_id)
                crops_onnx.append(input_tensor)
                crop_meta.append((crop_box, center_point, bbox_w, bbox_h, int(tracker_id), det_conf))
    else:
        # Skipped frame – loitering check still runs, but no new crops
        for det in detections:
            det_box, det_mask, det_conf, class_id, tracker_id, data = det
            if tracker_id is None:
                continue
            x1, y1, x2, y2 = det_box
            center_point = ((x1 + x2) / 2, (y1 + y2) / 2)
            if check_loitering(tracker_id, time.time(), center_point, track_positions):
                loitering_tracker_ids.add(tracker_id)
        # labels will be filled from track_to_customer below

    # ── 4. Embedding & Re‑ID (only if crops exist) ──
    if crops_onnx and run_embedding:
        with open('data/embeddings_WTF.txt', 'a+') as f:
            batch_input = np.stack(crops_onnx, axis=0)
            embeddings = embedder_session.run(None, {"input": batch_input})[0]
            for i, emb in enumerate(embeddings):
                emb = emb.flatten()
                emb = emb / (np.linalg.norm(emb) + 1e-8)
                bbox, center_point, bbox_w, bbox_h, tracker_id, confidence_score = crop_meta[i]
                # print(111111111111,tracker_id)
                temp_f_text = f'{i+1}_tracker_id: {tracker_id} '
                if tracker_id in track_to_customer:
                    customer_id = track_to_customer[tracker_id]
                    # Ask the writer for the minimum distance to this customer
                    ########################### What is the use for this code?
                    # I send my request with a unique ID.
                    # I start listening.
                    # I pull the next available message.
                    # If the ID matches, I keep it.
                    # If not, I immediately push it back so the real owner can later pull it.
                    # I repeat until my own message appears.
                    # Because there are only a few active requests at any moment, the loop rarely spins more than once or twice, so it’s fast.
                    request_id = f"dist_{cam_id}_{tracker_id}_{time.time()}"
                    db_queue.put(("min_dist_to_customer", emb, customer_id, request_id))
                    while True:
                        rid, min_dist = response_queue.get()   # take one message out
                        if rid == request_id:                  # is it mine?
                            break                              # yes – I'm done
                        response_queue.put((rid, min_dist))    # no – return it for someone else

                    # print(2222222222, min_dist, f'tracker_id in track_to_customer: {tracker_id}')
                    # Store only if the embedding is diverse
                    if min_dist > settings.DIVERSITY_THRESHOLD:
                        db_queue.put(("store_embedding", customer_id, cam_id, emb, time.time(), bbox_w, bbox_h))
                        new_temp_f_text = f'✅: customer_id: {customer_id}  min_dist: {min_dist:.4f} ===> "store_embedding"\n'
                        f.write(temp_f_text + new_temp_f_text)
                    else:
                        new_temp_f_text = f'✅: customer_id: {customer_id}  min_dist: {min_dist:.4f} ===> "update_customer_last_seen"\n'
                        f.write(temp_f_text + new_temp_f_text)
                        db_queue.put(("update_customer_last_seen", customer_id, time.time()))
                    
                else:
                    # Unknown tracker – match or register atomically
                    request_id = f"match_{cam_id}_{tracker_id}_{time.time()}"
                    # print(center_point, type(center_point))
                    db_queue.put(("match_or_register", emb, cam_id, time.time(), request_id, center_point, bbox_w, bbox_h, tracker_id, confidence_score))
                    
                    while True:
                        rid, customer_id, is_new, match_dist = response_queue.get()
                        if rid == request_id:
                            break
                        response_queue.put((rid, customer_id, is_new, match_dist))

                    track_to_customer[tracker_id] = customer_id
                    # For existing matches, store diverse embeddings
                    if not is_new and match_dist > settings.DIVERSITY_THRESHOLD:
                        db_queue.put(("store_embedding", customer_id, cam_id, emb, time.time(), bbox_w, bbox_h))
                        new_temp_f_text = f'❌: customer_id: {customer_id}  match_dist: {match_dist} ===> "store_embedding"\n'
                        f.write(temp_f_text + new_temp_f_text)
                    else:
                        db_queue.put(("update_customer_last_seen", customer_id, time.time()))
                        new_temp_f_text = f'❌: customer_id: {customer_id}  match_dist: {match_dist} ===> "update_customer_last_seen"\n'
                        f.write(temp_f_text + new_temp_f_text)
                labels.append(f"#Track: {int(tracker_id)} {confidence_score:.2f} ID:{customer_id}")
                # labels.append(f"#{confidence_score:.2f} ID:{customer_id}")
                # print(track_to_customer)
            
            f.write("\n\n==================================================================================================================="+'\n\n')
    else:
        # Build labels from existing track_to_customer (no new embeddings)
        labels = []
        for det in detections:
            _, _, _, _, tracker_id, _ = det
            if tracker_id is None:
                continue
            customer_id = track_to_customer.get(tracker_id)
            if customer_id is not None:
                labels.append(f"#Track: {int(tracker_id)} ID:{customer_id}")
            else:
                labels.append(f"#Track: {int(tracker_id)} ?")    


    # ── 5. Annotate a clean copy ──
    annotated = frame.copy()
    # Traces, boxes, labels – only if we have detections
    if len(detections) > 0:
        annotated = trace_annotator.annotate(annotated, detections)
        annotated = box_annotator.annotate(annotated, detections)
        annotated = label_annotator.annotate(annotated, detections, labels)

    # FPS counter
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
    ready_slots,
    det_queue,
    free_slots,
    stop_event
):
    pin_process([1,2,3])
    shm = shared_memory.SharedMemory(name=shm_name)
    detector_model = get_detector()
    while not stop_event.is_set():
        try:
            idx = ready_slots.get(timeout=0.1)
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
                  shm_name,
                  frame_shape, 
                  frame_bytes,
                  free_slots,
                  ready_slots, 
                  det_queue,
                  stop_event):
    pin_process([0])
    shm = shared_memory.SharedMemory(name=shm_name)

    cap = None
    first_attempt = True          # <-- ADD THIS
    online = False
    consecutive_failures = 0
    max_backoff = 60.0            # never sleep longer than 60 seconds
    offline_frame = np.zeros(frame_shape, dtype=np.uint8)
    # Draw the offline message on the placeholder (optional)
    cv2.putText(offline_frame, "Stream offline", (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    frame_counter = 0
    while not stop_event.is_set():
        if not online:
            # Release any old capture and wait before reconnecting
            if cap is not None:
                cap.release()
                cap = None
            # Exponential backoff: 1s, 2s, 4s, 8s, ... capped at max_backoff
            if first_attempt and consecutive_failures < 6:  # <-- no sleep on the first try
                first_attempt = False
                sleep_time = 0
            else:
                sleep_time = min(1.0 * (2 ** consecutive_failures), max_backoff)

            stop_event.wait(sleep_time)   # wait but break if stop_event is set
            if stop_event.is_set():
                break
            # Attempt to open the stream
            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                consecutive_failures += 1
                # Push an offline placeholder into one slot so the user sees it
                _push_placeholder_frame(shm, frame_shape, frame_bytes,
                                        free_slots, ready_slots, offline_frame)
                continue
            online = True
            consecutive_failures = 0

        # Stream is open – try to read
        ret, frame = cap.read()
        if not ret:
            # Stream died – switch back to offline mode
            online = False
            consecutive_failures += 1
            _push_placeholder_frame(shm, frame_shape, frame_bytes,
                                    free_slots, ready_slots, offline_frame)
            continue

        # Live frame received – push it into the ring buffer
                # Live frame received
        frame_counter += 1
        consecutive_failures = 0
        # --- Frame‑skip: only process every 3rd frame ---
        if frame_counter % 3 == 0:
            # Keyframe → send to detector via ring buffer #(ready_slots)
            _push_placeholder_frame(shm, frame_shape, frame_bytes,
                                    free_slots, ready_slots, offline_frame)
        else:
            # Skipped frame → write to shared memory + send placeholder directly to #(det_queue)
            _push_placeholder_frame(shm, frame_shape, frame_bytes,
                                    free_slots, det_queue, offline_frame)
            
    # Cleanup
    if cap is not None:
        cap.release()
    shm.close()

def _push_placeholder_frame(shm, frame_shape, frame_bytes,
                            free_slots, ready_slots, placeholder):
    """Push a single offline frame into the ring if a slot is available."""
    try:
        idx = free_slots.get_nowait()
    except queue.Empty:
        return   # ring is full – nothing to do
    _write_frame_to_slot(shm, idx, placeholder, frame_shape, frame_bytes)
    try:
        ready_slots.put_nowait(idx)
    except queue.Full:
        try:
            free_slots.put_nowait(idx)
        except:
            pass

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
    track_positions,
    cam_id
):
    pin_process([4,5])
    input_shm = shared_memory.SharedMemory(name=input_shm_name)
    output_shm = shared_memory.SharedMemory(name=output_shm_name)
    embedder_session = create_session(settings.FEATURE_EXTRACTOR_MODEL, num_threads=1)
    track_to_customer = {}
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
        # np.array([
        #     [0, 0],
        #     [1800 - 5, 0],
        #     [1800 - 5, 1800 - 5],
        #     [1080 + 5, 1800 - 5]
        # ], np.int32),
            np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ], dtype=np.int32)
        # np.array([
        #     [1080 + 5, 0],
        #     [2160, 0],
        #     [2160, 1300 - 5],
        #     [1080 + 5, 1300 - 5]
        # ], np.int32),
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
            if item is None:
                break

            idx, xyxy, confidence, class_id = item
            # Build detections object (empty if it's a skipped frame)
            if xyxy is None:
                # detections = sv.Detections.empty()
                detections = last_valid_detections   # To make the Tracker_model predict's the person's next BBox so the frames will remain annotated
                run_embedding = False
            else:
                detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
                last_valid_detections = detections
                run_embedding = True
        except queue.Empty:
            continue
        
        if item is None:
            break
        
        # --- Tick the FPS counter ---
        # fps_monitor.tick()
        
        frame = frame_view(input_shm,frame_shape,frame_bytes,idx)
        unique_visitors = set()
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
            fps_monitor = fps_monitor,
            box_annotator=box_annotator,
            zone_box_annotators=zone_box_annotators,
            track_positions=track_positions,
            response_queue=response_queue,
            trace_annotator=trace_annotator,
            label_annotator=label_annotator,
            run_embedding=run_embedding
        )

        # WRITE DIRECTLY TO OUTPUT SHM
        output_frame = np.ndarray(
            frame_shape,
            dtype=np.uint8,
            buffer=output_shm.buf
        )
        np.copyto(output_frame, annotated)
        try:
            free_slots.put_nowait(idx)
        except:
            pass

    input_shm.close()
    output_shm.close()











