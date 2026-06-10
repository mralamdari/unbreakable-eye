# import cv2
# import time
# from datetime import datetime
# # print(f"\nStarted at: {start.strftime('%H:%M:%S.%f')[:-3]}")
# # print(f"Ended at: {end.strftime('%H:%M:%S.%f')[:-3]}")
# # print(f"Duration: {duration.total_seconds():.2f} seconds")
# # print(f"Duration: {duration}")
# import threading
# import numpy as np
# import pandas as pd
# import supervision as sv
# import onnxruntime as ort
# from loguru import logger
# import matplotlib.pyplot as plt
# from src.core.config import settings
# from src.vision.factory import get_detector
# from src.core.database import  (fast_match,
#                                 fast_min_dist_to_customer,
#                                 load_cache,
#                                 store_embedding,
#                                 # store_numpy_embedding,
#                                 get_connection,
#                                 # process_detection, 
#                                 write_connection)


# class VisionPipeline:
#     def __init__(self, RTSP_URL: str, CAM_ID: str):
#         logger.info("⚙️ Initializing Vision Pipeline components...")

#         # Configuration
#         self.REID_THRESHOLD = 0.5         # cosine distance / Euclidean threshold
#         self.cam_id = CAM_ID
#         # --- RAM CACHE ---
#         # Holds (person_id, embedding_vector)
#         # This is the "Active Tracks" in RAM
#         self.TIME_WINDOW_SECONDS = 300  # 5 minutes
#         self.ram_cache = {} 
#         self.RTSP_URL = RTSP_URL
#         self.frame_counter = 0
#         self.tracked_paths = {}
#         self.tracker = sv.ByteTrack() 
#         self.person_class_id = 0 
#         # At the start of your pipeline class (or in main.py lifespan)
#         # self.db_conn = 
#         get_connection()          # persistent write connection
#         # load_cache(self.db_conn)                 # load existing embeddings into memory
#         self.model_input_size = (256, 128)
#         self.torso_ratio = 2/3
#         self.reid_threshold = 0.5                # adjust
#         self.diversity_threshold = 0.2           # adjust
#          # Tracker mapping
#         self.track_to_customer = {}   # tracker_id -> customer_id


#         # Setup UI
#         self.fps_monitor = sv.FPSMonitor()
#         color = sv.ColorPalette.DEFAULT 
#         self.box_annotator = sv.BoxAnnotator(color=color)
#         self.trace_annotator = sv.TraceAnnotator(color=color, trace_length=30) # Reduced length for speed
#         self.label_annotator = sv.LabelAnnotator(color=color, text_color=sv.Color.BLACK)

        
#         self.door_orientation = "horizontal"   # "horizontal" or "vertical"
#         self.door_position_ratio = 0.75        # 55% down the frame if horizontal
#         self.door_margin_ratio = 0.01          # keep line inside image edges
#         self.trigger_anchor = sv.Position.BOTTOM_CENTER
#         self.width = 1280
#         self.height = 1280
#         # Auto-config state
#         margin_x = int(self.width * self.door_margin_ratio)
#         margin_y = int(self.height * self.door_margin_ratio)
#         if self.door_orientation == "horizontal":
#             y = int(self.height * self.door_position_ratio)
#             start = sv.Point(margin_x, y)
#             end = sv.Point(self.width - margin_x, y)
#         else:
#             x = int(self.width * self.door_position_ratio)
#             start = sv.Point(x, margin_y)
#             end = sv.Point(x, self.height - margin_y)

#         self.line_zone = sv.LineZone(
#             start=start,
#             end=end,
#             triggering_anchors=[self.trigger_anchor])

#         self.line_zone_annotator = sv.LineZoneAnnotator(
#             thickness=2,
#             text_thickness=1,
#             text_scale=0.7)
        
#         # Track how long each person stays visible for dwell/queue analytics.
#         # self.track_presence = {}
#         # self.completed_presence = {}
#         # self.track_timeout_seconds = 2.0
#         # self.stall_threshold_seconds = 15.0
#         # self.queue_threshold_seconds = 30.0
#         # Change these depending on your entrance geometry
#         logger.info(f"🔌 Connecting to Video Source: {self.RTSP_URL}")
        
#         # Camera
#         self.cap = cv2.VideoCapture(self.RTSP_URL)
#         self.lock = threading.Lock()
#         self.latest_frame = None           # annotated frame ready for streaming
#         if not self.cap.isOpened():
#             logger.warning(f"❌ COULD NOT OPEN VIDEO SOURCE: {self.RTSP_URL}")
#             # Optional: Retry logic could be logged here
#             self.running = False
#         else:
#             logger.info("✅ Video Source Connected.")
#             self.running = True
#             try: 
#                 self.model = get_detector()
#                 logger.info("✅ Detector Model Loaded Successfully.")
#             except:
#                 logger.warning("❌ COULD NOT OPEN Dectector Model")
            
#             # start background thread for detection
#             self.thread = threading.Thread(target=self._detection_loop, daemon=True)
#             self.thread.start()

#             try:
#                 fe_model = settings.FEATURE_EXTRACTOR_MODEL
#                 print(f"🔌 Loading Feature Extractor model: {fe_model}...")
#                 self.session = ort.InferenceSession(fe_model, providers=['CPUExecutionProvider'])
#                 print(f"🔍 Model input size: {self.session.get_inputs()[0].shape}")
#                 logger.info(f"✅ Loaded Feature Extractor model: {fe_model} successfully")
#             except:
#                 logger.warning(f"❌ COULD NOT LOAD Feature Extractor model")
        
#     def normalize_vector(self, vec):
#         """FAISS needs normalized vectors for Cosine Similarity."""
#         return vec / np.linalg.norm(vec)
    
#     def cleanup_ram(self):
#         current_time = datetime.now()
#         for pid in list(self.ram_cache.keys()):
#             self.ram_cache[pid]['seen_at'] # Access to check time
#             if current_time - self.ram_cache[pid]['seen_at'] > self.TIME_WINDOW_SECONDS:
#                 del self.ram_cache[pid]
#                 # Also remove from FAISS (optional, but saves memory)
#                 # FAISS doesn't support efficient removal, so you rebuild or use a separateself. index.

#     @staticmethod
#     def _format_duration(duration_seconds):
#         total_seconds = max(0, int(duration_seconds))
#         minutes, seconds = divmod(total_seconds, 60)
#         hours, minutes = divmod(minutes, 60)
#         if hours:
#             return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
#         return f"{minutes:02d}:{seconds:02d}"

#     def _update_presence_timers(self, now, active_track_ids, track_durations):
#         stale_track_ids = []
#         for tracker_id, presence in self.track_presence.items():
#             if tracker_id in active_track_ids:
#                 continue

#             if now - presence["last_seen"] >= self.track_timeout_seconds:
#                 self.completed_presence[tracker_id] = presence["last_seen"] - presence["first_seen"]
#                 stale_track_ids.append(tracker_id)

#         for tracker_id in stale_track_ids:
#             self.track_presence.pop(tracker_id, None)

#         active_durations = list(track_durations.values())
#         completed_durations = list(self.completed_presence.values())

#         longest_active = max(active_durations, default=0.0)
#         average_active = (
#             sum(active_durations) / len(active_durations)
#             if active_durations else 0.0
#         )
#         average_completed = (
#             sum(completed_durations) / len(completed_durations)
#             if completed_durations else 0.0
#         )
#         stall_count = sum(
#             1 for duration in active_durations if duration >= self.stall_threshold_seconds
#         )
#         queue_count = sum(
#             1 for duration in active_durations if duration >= self.queue_threshold_seconds
#         )

#         return {
#             "active_count": len(active_durations),
#             "stall_count": stall_count,
#             "queue_count": queue_count,
#             "longest_active": longest_active,
#             "average_active": average_active,
#             "average_completed": average_completed,
#         }

#     def export_path_data(self, file_path='data/customers_paths.csv'):
#         """
#         Export all tracked paths to a CSV database.
#         """
#         all_rows = []
#         for track_id, path in self.tracked_paths.items():
#             df_track = pd.DataFrame(path)
#             df_track['track_id'] = track_id
#             all_rows.append(df_track)
        
#         if not all_rows:
#             print("No path data to export.")
#             return

#         combined_df = pd.concat(all_rows, ignore_index=True)
#         combined_df.to_csv(file_path, index=False)
#         print(f"Data saved to {file_path}")
#         return combined_df

#     def generate_heatmap(self, output_file='data/heatmap.png'):
#         """
#         Create a density heatmap based on tracked positions.
#         """
#         # Initialize empty heatmap grid (same shape as input frame)
#         # h, w = self.frame_shape
#         heatmap_grid = np.zeros((self.height, self.width), dtype=np.float32)
        
#         # Accumulate density
#         for track_id, path in self.tracked_paths.items():
#             for point in path:
#                 # Round to nearest pixel (or use direct mapping if coordinates match)
#                 x = int(point['x'])
#                 y = int(point['y'])
                
#                 if 0 <= x < self.width and 0 <= y < self.height:
#                     # Increment pixel value
#                     heatmap_grid[y, x] += 1

#         # Normalize (optional)
#         max_val = np.max(heatmap_grid)
#         if max_val > 0:
#             heatmap_grid = (heatmap_grid / max_val) * 255

#         # Create visualization
#         heat_map = cv2.applyColorMap(np.uint8(heatmap_grid), cv2.COLORMAP_JET)
        
#         # Display result
#         cv2.imwrite(output_file, heat_map)
#         plt.figure(figsize=(15, 10))
#         plt.imshow(heat_map, cmap='hot')
#         plt.title(f"Customer Movement Heatmap (Max visits: {int(max_val)})")
#         plt.colorbar()
#         plt.tight_layout()
#         plt.savefig(output_file.replace('.png', '.jpg')) # Save figure separately
#         plt.close()
        
#         print(f"Heatmap generated and saved to {output_file}")
#         return heat_map

#     def generate_lines(self, output_file='data/customer_paths.png'):
#         """
#         Draw customer paths with different colors on a single image.
#         """
#         # Initialize empty figure
#         fig, ax = plt.subplots(figsize=(15, 10))
        
#         # Plot each path with a unique color
#         for track_id, path in self.tracked_paths.items():
#             x_coords = [point['x'] for point in path]
#             y_coords = [point['y'] for point in path]
#             ax.plot(x_coords, y_coords, label=f'Track {track_id}', linewidth=2)
        
#         # Set the aspect ratio to 'equal' to preserve the scale
#         ax.set_aspect('equal')
        
#         # Add labels and title
#         plt.title("Customer Movement Paths")
#         plt.xlabel("X Position")
#         plt.ylabel("Y Position")
#         plt.legend()
        
#         # Save the figure
#         plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
#         print(f"Paths saved to {output_file}")     
#         return fig

#     def preprocess_crop(self, frame, bbox):
#         x1, y1, x2, y2 = map(int, bbox)
#         w, h = x2 - x1, y2 - y1
#         cx, cy = (x1+x2)/2, (y1+y2)/2 
#         crop_y2 = int(y1 + h * self.torso_ratio)
#         crop = frame[y1:crop_y2, x1:x2]
#         flag =  crop.size == 0 or w < 20 or h < 20

#         # Preprocess
#         resized = cv2.resize(crop, 
#                                 (self.model_input_size[1], self.model_input_size[0]),
#                             interpolation=cv2.INTER_AREA)
#         # Normalize: (img / 255 - mean) / std
#         mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
#         std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
#         normalized = (resized / 255.0 - mean) / std
        
#         # Convert to CHW
#         input_tensor = np.transpose(normalized, (2, 0, 1)).astype(np.float32)
#         return input_tensor, (x1, y1, x2, y2), (cx, cy), flag

    # def process_frame(self, frame):
    #     current_frame_idx = self.frame_counter
    #     self.fps_monitor.tick()
    #     fps = self.fps_monitor.fps
    #     # ── 1. Detection & Filtering ──
    #     detections = self.model.predict(frame)
    #     if detections.class_id is not None:
    #         detections = detections[detections.class_id == self.person_class_id]

    #     # ── 2. Tracking (ByteTrack) ──
    #     detections = self.tracker.update_with_detections(detections)

    #     # ── 3. Collect crops and tracker metadata ──
    #     crops_onnx = []
    #     crop_meta = []          # (det_box, tracker_id, confidence)
    #     labels = []

    #     for det in detections:
    #         det_box, mask, confidence, class_id, tracker_id, data = det
    #         if tracker_id is None:
    #             continue
                        
    #         # Preprocess the crop (torso, resize, normalise)
    #         input_tensor, crop_box, center_point, crop_flag = self.preprocess_crop(frame, det_box)
    #         if crop_flag:
    #             continue

    #         crops_onnx.append(input_tensor)
    #         crop_meta.append((crop_box, int(tracker_id), confidence))

    #         cust = self.track_to_customer.get(int(tracker_id))
    #         labels.append(
    #             f"#{int(tracker_id)} {confidence:.2f} "
    #             f"ID:{cust if cust else '?'} ({confidence:.2f})"
    #         )

    #     if not crops_onnx:
    #         return frame   # nothing to process

    #     # ── 4. Batch ONNX embedding extraction ──
    #     batch_input = np.stack(crops_onnx, axis=0)          # (N, 3, 128, 64)
    #     embeddings = self.session.run(None, {'input': batch_input})[0]   # (N, 512)

    #     # ── 5. Re‑identification loop ──
    #     emb_meta_id = None
    #     with write_connection() as conn:
    #         load_cache(conn)   # ensure numpy cache is current (only first time it's needed)
    #         for i, emb in enumerate(embeddings):
    #             emb = emb.flatten()
    #             emb = emb / (np.linalg.norm(emb) + 1e-8)   # normalise

    #             bbox, tracker_id, conf = crop_meta[i]
    #             bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    #             center_str = f"{center_point[0]:.2f},{center_point[1]:.2f}"
    #             now = datetime.now()

    #             # ── 5a. If we already know this tracker → use the stored customer_id ──
    #             if tracker_id in self.track_to_customer:
    #                 customer_id = self.track_to_customer[tracker_id]
    #                 # Optionally store a diverse embedding
    #                 if fast_min_dist_to_customer(emb, customer_id) > self.diversity_threshold:
    #                     emb_meta_id = store_embedding(conn, customer_id, self.cam_id, emb, now)
    #                 else:
    #                     emb_meta_id = None
    #             else:
    #                 matched_cust, dist = fast_match(emb, self.reid_threshold)
    #                 if matched_cust is not None:
    #                     customer_id = matched_cust
    #                     self.track_to_customer[tracker_id] = customer_id
    #                     # Update last_seen
    #                     conn.execute("UPDATE customers SET last_seen = ? WHERE id = ?", (now, customer_id))
    #                     if fast_min_dist_to_customer(emb, customer_id) > self.diversity_threshold:
    #                         emb_meta_id = store_embedding(conn, customer_id, self.cam_id, emb, now)
    #                     else:
    #                         emb_meta_id = None
    #                 else:
    #                     # New customer
    #                     cur = conn.execute(
    #                         "INSERT INTO customers (first_seen, last_seen) VALUES (?, ?)", (now, now)
    #                     )
    #                     customer_id = cur.lastrowid
    #                     self.track_to_customer[tracker_id] = customer_id
    #                     # First embedding -> always store
    #                     emb_meta_id = store_embedding(conn, customer_id, self.cam_id, emb, now)

    #             # Log detection (only if emb_meta_id is meaningful)
    #             conn.execute(
    #                 "INSERT INTO detections (camera_id, bbox, center_point, timestamp, embedding_meta_id) "
    #                 "VALUES (?, ?, ?, ?, ?)",
    #                 (self.cam_id, bbox_str, center_str, now, emb_meta_id)
    #             )

    #     # ── 6. Annotate frame with customer_id (optional) ──
    #     annotated_frame = frame.copy()
    #     annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
    #     annotated_frame = self.box_annotator.annotate(annotated_frame, detections)
    #     annotated_frame = self.label_annotator.annotate(annotated_frame, detections, labels)
        
    #     # 3. Trigger line counter
    #     # No manual in/out values are passed here.
    #     if len(detections) > 0:
    #         self.line_zone.trigger(detections=detections)
        
    #     # Draw line + in/out counts
    #     annotated_frame = self.line_zone_annotator.annotate(
    #         frame=annotated_frame,
    #         line_counter=self.line_zone)
    #     # Extra text
    #     annotated_frame = sv.draw_text(
    #         scene=annotated_frame,
    #         text=f"FPS: {fps:.1f}",
    #         text_anchor=sv.Point(40, 30),
    #         background_color=sv.Color.RED,
    #         text_color=sv.Color.WHITE)

    #     occupancy = self.line_zone.in_count - self.line_zone.out_count
    #     annotated_frame = sv.draw_text(
    #         scene=annotated_frame,
    #         text=(
    #             f"IN: {self.line_zone.in_count}  "
    #             f"OUT: {self.line_zone.out_count}  "
    #             f"INSIDE: {occupancy}"
    #         ),
    #         text_anchor=sv.Point(220, 30),
    #         background_color=sv.Color.BLACK,
    #         text_color=sv.Color.WHITE)
    #     return annotated_frame
    
    # #· Detection loop runs in a background thread (or process in multiprocessing) – it uses short‑lived with write_connection() blocks only when writing to the DB.
    # def _detection_loop(self):
    #     frame_interval = 0.0   # process 2 FPS
    #     last_process = time.time()
    #     try:
    #         while self.running:
    #             success, frame = self.cap.read()
    #             if not success:
    #                 self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    #                 continue
    #             # resized_frame = frame
    #             resized_frame = cv2.resize(frame, (512, 256))
    #             self.height, self.width = resized_frame.shape[:2]
    #             # heavy processing here (detection, embedding, DB)
    #             now = time.time()
    #             if now - last_process >= frame_interval:
    #                 last_process = now
    #                 annotated = self.process_frame(frame)
    #                 with self.lock:
    #                     self.latest_frame = annotated.copy()   # update shared variable
    #             # Optional: sleep a short time to limit CPU (e.g., 0.1 s)
    #             # time.sleep(0.05)
    #     except Exception as e:
    #         logger.error(f"Pipeline thread crashed: {e}", exc_info=True)
    #     finally:    
    #         self.cap.release()
    
    # def get_latest_frame(self):
    #     """Thread‑safe access to the latest annotated frame."""
    #     with self.lock:
    #         return self.latest_frame.copy() if self.latest_frame is not None else None

    # def stop(self):
    #     self.running = False
    #     self.thread.join()
    # #· generate_frames – a generator that simply reads the latest frame from a shared variable (threading) or queue (multiprocessing). It never touches the database.
    # def generate_frames(self):
    #     """Generator for MJPEG – lightweight, just yields the latest frame."""
    #     while self.running:
    #         with self.lock:
    #             frame_to_yield = self.latest_frame
    #         if frame_to_yield is not None:
    #             ret, buffer = cv2.imencode('.jpg', frame_to_yield)
    #             if ret:
    #                 yield (b'--frame\r\n'
    #                        b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    #         # time.sleep(0.03)   # ~30 FPS streaming, throttle CPU
        
        
        
        































# import cv2
# import time
# from datetime import datetime
# import threading
# import numpy as np
# import pandas as pd
# import supervision as sv
# import onnxruntime as ort
# from loguru import logger
# import matplotlib.pyplot as plt
# from src.core.config import settings
# from src.vision.factory import get_detector
# from src.core.database import  (fast_match,
#                                 fast_min_dist_to_customer,
#                                 load_cache,
#                                 embedding_cache,
#                                 store_embedding,
#                                 get_connection,
#                                 write_connection)


# class VisionPipeline:
#     def __init__(self, RTSP_URL: str, CAM_ID: str):
#         logger.info("⚙️ Initializing Vision Pipeline components...")

#         # Configuration
#         self.cam_id = CAM_ID
#         # --- RAM CACHE ---
#         self.ram_cache = {} 
#         self.RTSP_URL = RTSP_URL
#         self.frame_counter = 0
#         self.tracked_paths = {}
#         self.tracker = sv.ByteTrack() 
#         self.person_class_id = 0 
#         # At the start of your pipeline class (or in main.py lifespan)
#         self.db_conn = get_connection()          # persistent write connection
#         load_cache(self.db_conn)                 # load existing embeddings into memory
        
#         # At the very beginning of your pipeline (e.g., main.py or in __init__)
#         self.model_input_size = (256, 128)
#         self.torso_ratio = 2/3
#         self.reid_threshold = settings.REID_THRESHOLD
#         self.diversity_threshold = 0.2 
#          # Tracker mapping
#         self.track_to_customer = {}   # tracker_id -> customer_id
        

#         # Setup UI
#         self.fps_monitor = sv.FPSMonitor()
#         color = sv.ColorPalette.DEFAULT 
#         self.box_annotator = sv.BoxAnnotator(color=color)
#         self.trace_annotator = sv.TraceAnnotator(color=color, trace_length=30) # Reduced length for speed
#         self.label_annotator = sv.LabelAnnotator(color=color, text_color=sv.Color.BLACK)

        
#         self.door_orientation = "horizontal"   # "horizontal" or "vertical"
#         self.door_position_ratio = 0.75        # 55% down the frame if horizontal
#         self.door_margin_ratio = 0.01          # keep line inside image edges
#         self.trigger_anchor = sv.Position.BOTTOM_CENTER
#         self.width = 8280
#         self.height = 8280
#         # Auto-config state
#         margin_x = int(self.width * self.door_margin_ratio)
#         margin_y = int(self.height * self.door_margin_ratio)
#         if self.door_orientation == "horizontal":
#             y = int(self.height * self.door_position_ratio)
#             start = sv.Point(margin_x, y)
#             end = sv.Point(self.width - margin_x, y)
#         else:
#             x = int(self.width * self.door_position_ratio)
#             start = sv.Point(x, margin_y)
#             end = sv.Point(x, self.height - margin_y)

#         self.line_zone = sv.LineZone(
#             start=start,
#             end=end,
#             triggering_anchors=[self.trigger_anchor])

#         self.line_zone_annotator = sv.LineZoneAnnotator(
#             thickness=2,
#             text_thickness=1,
#             text_scale=0.7)
        
#         logger.info(f"🔌 Connecting to Video Source: {self.RTSP_URL}")
        
#         # Camera
#         self.cap = cv2.VideoCapture(self.RTSP_URL)
#         self.lock = threading.Lock()
#         self.latest_frame = None           # annotated frame ready for streaming
#         if not self.cap.isOpened():
#             logger.warning(f"❌ COULD NOT OPEN VIDEO SOURCE: {self.RTSP_URL}")
#             # Optional: Retry logic could be logged here
#             self.running = False
#         else:
#             logger.info("✅ Video Source Connected.")
#             self.running = True
#             try: 
#                 self.model = get_detector()
#                 logger.info("✅ Detector Model Loaded Successfully.")
#             except:
#                 logger.warning("❌ COULD NOT OPEN Dectector Model")
            
#             # start background thread for detection
#             self.thread = threading.Thread(target=self._detection_loop, daemon=True)
#             self.thread.start()

#             try:
#                 fe_model = settings.FEATURE_EXTRACTOR_MODEL
#                 print(f"🔌 Loading Feature Extractor model: {fe_model}...")
#                 self.session = ort.InferenceSession(fe_model,
#                                                     providers=['CPUExecutionProvider'],
#                                                     graph_optimization_level=True)
#                 print(f"🔍 Model input size: {self.session.get_inputs()[0].shape}")
#                 logger.info(f"✅ Loaded Feature Extractor model: {fe_model} successfully")
#             except:
#                 logger.warning(f"❌ COULD NOT LOAD Feature Extractor model")

#     def preprocess_crop(self, frame, bbox):
#         x1, y1, x2, y2 = map(int, bbox)
#         w, h = x2 - x1, y2 - y1
#         cx, cy = (x1+x2)/2, (y1+y2)/2 
#         crop_y2 = int(y1 + h * self.torso_ratio)
#         crop = frame[y1:crop_y2, x1:x2]
#         flag =  crop.size == 0 or w < 20 or h < 20

#         # Preprocess
#         resized = cv2.resize(crop, 
#                                 (self.model_input_size[1], self.model_input_size[0]),
#                             interpolation=cv2.INTER_AREA)
#         # Normalize: (img / 255 - mean) / std
#         mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
#         std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
#         normalized = (resized / 255.0 - mean) / std
        
#         # Convert to CHW
#         input_tensor = np.transpose(normalized, (2, 0, 1)).astype(np.float32)
#         return input_tensor, (x1, y1, x2, y2), (cx, cy), flag


#     def process_frame(self, frame):
#         current_frame_idx = self.frame_counter
#         self.fps_monitor.tick()
#         fps = self.fps_monitor.fps
#         # ── 1. Detection & Filtering ──
#         t0 = time.time()
#         detections = self.model.predict(frame)
#         t1 = time.time()
#         print('# ── 1. Detection & Filtering ──', t1-t0)
#         if detections.class_id is not None:
#             detections = detections[detections.class_id == self.person_class_id]

#         # ── 2. Tracking (ByteTrack) ──
#         t0 = time.time()
#         detections = self.tracker.update_with_detections(detections)
#         t1 = time.time()
#         print('# ── 2. Tracking (ByteTrack) ──',t1-t0)
#         # ── 3. Collect crops and tracker metadata ──
#         crops_onnx = []
#         crop_meta = []          # (det_box, tracker_id, confidence)
#         labels = []
#         t0 = time.time()

#         for det in detections:
#             det_box, mask, confidence, class_id, tracker_id, data = det
#             if tracker_id is None:
#                 continue
                        
#             # Preprocess the crop (torso, resize, normalise)
#             input_tensor, crop_box, center_point, crop_flag = self.preprocess_crop(frame, det_box)
#             if crop_flag:
#                 continue

#             crops_onnx.append(input_tensor)
#             crop_meta.append((crop_box, int(tracker_id), confidence))

#             cust = self.track_to_customer.get(int(tracker_id))
#             labels.append(
#                 f"#{int(tracker_id)} {confidence:.2f} "
#                 f"ID:{cust if cust else '?'} ({confidence:.2f})"
#             )

#         if not crops_onnx:
#             return frame   # nothing to process
#         t1 = time.time()
#         print('# ── 3. Collect crops and tracker metadata + Embedding Generation ──', t1-t0)
        
#         t0 = time.time()
#         # ── 4. Batch ONNX embedding extraction ──
#         batch_input = np.stack(crops_onnx, axis=0)          # (N, 3, 256, 128)
#         embeddings = self.session.run(None, {'input': batch_input})[0]   # (N, 512)
#         t1 = time.time()
#         print('# ── 4. Batch ONNX embedding extraction ──', t1-t0)
        
        
#         # ── 5. Re‑identification loop ──
#         # Your current store_embedding already takes conn as the first argument, so it will work with self.db_conn without changes. Perfect.
#         conn = self.db_conn   # persistent, already open
#         for i, emb in enumerate(embeddings):
#             emb = emb.flatten()
#             emb = emb / (np.linalg.norm(emb) + 1e-8)

#             bbox, tracker_id, conf = crop_meta[i]
#             bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
#             center_str = f"{center_point[0]:.2f},{center_point[1]:.2f}"
#             now = datetime.now()

#             if tracker_id in self.track_to_customer:
#                 customer_id = self.track_to_customer[tracker_id]
#                 if fast_min_dist_to_customer(emb, customer_id) > self.diversity_threshold:
#                     emb_meta_id = store_embedding(conn, customer_id, self.cam_id, emb, now)
#                 else:
#                     emb_meta_id = None
#             else:
#                 matched_cust, dist = fast_match(emb, self.reid_threshold)
#                 if matched_cust is not None:
#                     customer_id = matched_cust
#                     self.track_to_customer[tracker_id] = customer_id
#                     conn.execute("UPDATE customers SET last_seen = ? WHERE id = ?", (now, customer_id))
#                     if fast_min_dist_to_customer(emb, customer_id) > self.diversity_threshold:
#                         emb_meta_id = store_embedding(conn, customer_id, self.cam_id, emb, now)
#                     else:
#                         emb_meta_id = None
#                 else:
#                     cur = conn.execute(
#                         "INSERT INTO customers (first_seen, last_seen) VALUES (?, ?)", (now, now)
#                     )
#                     customer_id = cur.lastrowid
#                     self.track_to_customer[tracker_id] = customer_id
#                     emb_meta_id = store_embedding(conn, customer_id, self.cam_id, emb, now)

#             conn.execute(
#                 "INSERT INTO detections (camera_id, bbox, center_point, timestamp, embedding_meta_id) "
#                 "VALUES (?, ?, ?, ?, ?)",
#                 (self.cam_id, bbox_str, center_str, now, emb_meta_id)
#             )
#         # Commit once per frame (still fast, but you may batch this later)
#         conn.commit()
#         t1 = time.time()
#         print('# ── 5. Re‑identification loop ──', t1-t0)
#         t0 = time.time()
#         # ── 6. Annotate frame with customer_id (optional) ──
#         annotated_frame = frame.copy()
#         annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
#         annotated_frame = self.box_annotator.annotate(annotated_frame, detections)
#         annotated_frame = self.label_annotator.annotate(annotated_frame, detections, labels)
        
#         # 3. Trigger line counter
#         # No manual in/out values are passed here.
#         if len(detections) > 0:
#             self.line_zone.trigger(detections=detections)
        
#         # Draw line + in/out counts
#         annotated_frame = self.line_zone_annotator.annotate(
#             frame=annotated_frame,
#             line_counter=self.line_zone)
#         # Extra text
#         annotated_frame = sv.draw_text(
#             scene=annotated_frame,
#             text=f"FPS: {fps:.1f}",
#             text_anchor=sv.Point(40, 30),
#             background_color=sv.Color.RED,
#             text_color=sv.Color.WHITE)

#         occupancy = self.line_zone.in_count - self.line_zone.out_count
#         annotated_frame = sv.draw_text(
#             scene=annotated_frame,
#             text=(
#                 f"IN: {self.line_zone.in_count}  "
#                 f"OUT: {self.line_zone.out_count}  "
#                 f"INSIDE: {occupancy}"
#             ),
#             text_anchor=sv.Point(220, 30),
#             background_color=sv.Color.BLACK,
#             text_color=sv.Color.WHITE)
#         t1 = time.time()
#         print('# ── 6. Annotate frame with customer_id (optional) ──', t1-t0)
#         return annotated_frame
    

#     def _detection_loop(self):
#         frame_interval = 0.0   # process 2 FPS
#         last_process = time.time()
#         try:
#             while self.running:
#                 success, frame = self.cap.read()
#                 if not success:
#                     self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#                     continue
#                 now = time.time()
#                 if now - last_process >= frame_interval:
#                     last_process = now
#                     annotated = self.process_frame(frame)
#                     with self.lock:
#                         self.latest_frame = annotated.copy()   # update shared variable
#         except Exception as e:
#             logger.error(f"Pipeline thread crashed: {e}", exc_info=True)
#         finally:  
#            self.cap.release() 
    
    
    
#     def get_latest_frame(self):
#         """Thread‑safe access to the latest annotated frame."""
#         with self.lock:
#             return self.latest_frame.copy() if self.latest_frame is not None else None

#     def stop(self):
#         self.running = False
#         self.thread.join()
        
#     #· generate_frames – a generator that simply reads the latest frame from a shared variable (threading) or queue (multiprocessing). It never touches the database.
#     def generate_frames(self):
#         """Generator for MJPEG – lightweight, just yields the latest frame."""
#         while self.running:
#             with self.lock:
#                 frame_to_yield = self.latest_frame
#             if frame_to_yield is not None:
#                 ret, buffer = cv2.imencode('.jpg', frame_to_yield)
#                 if ret:
#                     yield (b'--frame\r\n'
#                            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
#             # time.sleep(0.03)   # ~30 FPS streaming, throttle CPU
          
          
          
          
          
          
          
          
          














































# import os
# import cv2
# import time
# import threading
# import numpy as np
# import pandas as pd
# from queue import Queue
# import supervision as sv
# import onnxruntime as ort
# from loguru import logger
# from datetime import datetime
# import matplotlib.pyplot as plt
# from src.core.config import settings
# from src.core.db_writer import db_queue
# from src.vision.factory import get_detector
# from src.core.database import  (fast_match,
#                                 fast_min_dist_to_customer,
#                                 load_cache,
#                                 get_connection,
#                                 )

# os.environ["OMP_NUM_THREADS"] = "4"
# os.environ["OMP_WAIT_POLICY"] = "ACTIVE"
# os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
# os.environ["KMP_BLOCKTIME"] = "0"
# os.sched_setaffinity(0, {0,1,2,3})  # main thread, but you need the thread's native id; easier to use psutil

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


# class VisionPipeline:
#     def __init__(self, RTSP_URL: str, CAM_ID: str):
#         logger.info("⚙️ Initializing Vision Pipeline components...")
#         self.new_cust_count = 0
#         # Configuration
#         self.cam_id = CAM_ID
#         # --- RAM CACHE ---
#         self.ram_cache = {} 
#         self.RTSP_URL = RTSP_URL
#         self.frame_counter = 0
#         self.tracked_paths = {}
#         self.tracker = sv.ByteTrack() 
#         self.person_class_id = 0 
#         # Startup: self.db_conn = get_connection() → load_cache(self.db_conn).
#         # Embedder thread: for the “new customer” case, use self.db_conn directly (it’s the only thread that touches it, so no locking needed). All other writes → db_queue.put(...).
#         # Writer thread: its own connection, created inside start_db_writer.
#         self.db_conn = get_connection()          # persistent write connection
#         load_cache(self.db_conn)                 # load existing embeddings into memory
        
        
#         # ---- NEW: Threading infrastructure ----
#         self.db_queue = db_queue                 # reference to the global queue
#         self.frame_queue = Queue(maxsize=4)
#         self.det_queue = Queue(maxsize=4)
#         self.stop_event = threading.Event()
#         # At the very beginning of your pipeline (e.g., main.py or in __init__)
#         self.model_input_size = (256, 128)
#         self.torso_ratio = 2/3
#         self.reid_threshold = settings.REID_THRESHOLD
#         self.diversity_threshold = 0.2 
#          # Tracker mapping
#         self.track_to_customer = {}   # tracker_id -> customer_id
        

#         # Setup UI
#         self.fps_monitor = sv.FPSMonitor()
#         color = sv.ColorPalette.DEFAULT 
#         self.box_annotator = sv.BoxAnnotator(color=color)
#         self.trace_annotator = sv.TraceAnnotator(color=color, trace_length=30) # Reduced length for speed
#         self.label_annotator = sv.LabelAnnotator(color=color, text_color=sv.Color.BLACK)
#         # self.door_orientation = "horizontal"   # "horizontal" or "vertical"
#         # self.door_position_ratio = 0.75        # 55% down the frame if horizontal
#         # self.door_margin_ratio = 0.01          # keep line inside image edges
#         # self.trigger_anchor = sv.Position.BOTTOM_CENTER
#         # self.width = 8280
#         # self.height = 8280
#         # # Auto-config state
#         # margin_x = int(self.width * self.door_margin_ratio)
#         # margin_y = int(self.height * self.door_margin_ratio)
#         # if self.door_orientation == "horizontal":
#         #     y = int(self.height * self.door_position_ratio)
#         #     start = sv.Point(margin_x, y)
#         #     end = sv.Point(self.width - margin_x, y)
#         # else:
#         #     x = int(self.width * self.door_position_ratio)
#         #     start = sv.Point(x, margin_y)
#         #     end = sv.Point(x, self.height - margin_y)

#         # self.line_zone = sv.LineZone(
#         #     start=start,
#         #     end=end,
#         #     triggering_anchors=[self.trigger_anchor])

#         # self.line_zone_annotator = sv.LineZoneAnnotator(
#         #     thickness=2,
#         #     text_thickness=1,
#         #     text_scale=0.7)
        
#         logger.info(f"🔌 Connecting to Video Source: {self.RTSP_URL}")

#         cv2.setNumThreads(0)   # disables FFmpeg's own thread pool
#         # Camera
#         self.cap = cv2.VideoCapture(self.RTSP_URL)
#         self.lock = threading.Lock()
#         self.latest_frame = None           # annotated frame ready for streaming
#         if not self.cap.isOpened():
#             logger.warning(f"❌ COULD NOT OPEN VIDEO SOURCE: {self.RTSP_URL}")
#             # Optional: Retry logic could be logged here
#             self.running = False
#         else:
#             logger.info("✅ Video Source Connected.")
#             self.running = True
#             try: 
#                 self.model = get_detector()
#                 logger.info("✅ Detector Model Loaded Successfully.")
#             except:
#                 logger.warning("❌ COULD NOT OPEN Dectector Model")
            
#             # start background thread for detection
#             # Spawn the three threads that overlap detection, embedding, and I/O
#             self.thread_reader = threading.Thread(target=self.reader, daemon=True)
#             self.thread_detector = threading.Thread(target=self.detector_worker, daemon=True)
#             self.thread_embedder = threading.Thread(target=self.embedder_worker, daemon=True)

#             self.thread_reader.start()
#             self.thread_detector.start()
#             self.thread_embedder.start()
#             try:
#                 fe_model = settings.FEATURE_EXTRACTOR_MODEL
#                 print(f"🔌 Loading Feature Extractor model: {fe_model}...")
#                 self.session = create_session(fe_model, num_threads=(os.cpu_count())//2-2)
#                 print(f"🔍 Model input size: {self.session.get_inputs()[0].shape}")
#                 logger.info(f"✅ Loaded Feature Extractor model: {fe_model} successfully")
#             except:
#                 logger.warning(f"❌ COULD NOT LOAD Feature Extractor model")

#     def preprocess_crop(self, frame, bbox):
#         x1, y1, x2, y2 = map(int, bbox)
#         w, h = x2 - x1, y2 - y1
#         cx, cy = (x1+x2)/2, (y1+y2)/2 
#         crop_y2 = int(y1 + h * self.torso_ratio)
#         crop = frame[y1:crop_y2, x1:x2]
#         flag =  crop.size == 0 or w < 20 or h < 20

#         # Preprocess
#         resized = cv2.resize(crop, 
#                                 (self.model_input_size[1], self.model_input_size[0]),
#                             interpolation=cv2.INTER_AREA)
#         # Normalize: (img / 255 - mean) / std
#         mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
#         std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
#         normalized = (resized / 255.0 - mean) / std
        
#         # Convert to CHW
#         input_tensor = np.transpose(normalized, (2, 0, 1)).astype(np.float32)
#         return input_tensor, (x1, y1, x2, y2), (cx, cy), flag

#     def process_frame(self, frame, detections):
#         # current_frame_idx = self.frame_counter
#         self.fps_monitor.tick()
#         fps = self.fps_monitor.fps
#         zzcount_1 = 0
#         zzcount_2 = 0
        
#         # ── 1. Detection & Filtering ──
#         # You are ignoring the detections argument that was already computed by the detector thread and replacing it with a fresh (duplicate) prediction.
#         if detections.class_id is not None:
#             detections = detections[detections.class_id == self.person_class_id]

#         # ── 2. Tracking (ByteTrack) ──
#         t0 = time.time()
#         detections = self.tracker.update_with_detections(detections)
#         t1 = time.time()
#         print('# ── 2. Tracking (ByteTrack) ──',t1-t0)
#         # ── 3. Collect crops and tracker metadata ──
#         crops_onnx = []
#         crop_meta = []          # (det_box, tracker_id, confidence)
#         labels = []
#         t0 = time.time()
#         idd = 0
#         for det in detections:
#             idd += 1
#             det_box, mask, confidence, class_id, tracker_id, data = det
#             if tracker_id is None:
#                 continue
            
#             zz0 = time.time()
#             # Preprocess the crop (torso, resize, normalise)
#             input_tensor, crop_box, center_point, crop_flag = self.preprocess_crop(frame, det_box)
#             zz1 = time.time()
#             print(f'#  ── 3.{idd} Preprocess the crop (torso, resize, normalise)', zz1-zz0)                        
#             if crop_flag:
#                 continue

#             crops_onnx.append(input_tensor)
#             crop_meta.append((crop_box, int(tracker_id), confidence))

#             cust = self.track_to_customer.get(int(tracker_id))
#             labels.append(
#                 f"#{int(tracker_id)} {confidence:.2f} "
#                 f"ID:{cust if cust else '?'} ({confidence:.2f})"
#             )

#         if not crops_onnx:
#             return frame   # nothing to process
#         t1 = time.time()
#         print(f'# ── 3. Collect crops and tracker metadata + Embedding Generation ── FOR {idd} Detections', t1-t0)
        
#         t0 = time.time()
#         # ── 4. Batch ONNX embedding extraction ──
#         batch_input = np.stack(crops_onnx, axis=0)          # (N, 3, 256, 128)
#         embeddings = self.session.run(None, {'input': batch_input})[0]   # (N, 512)
#         t1 = time.time()
#         print('# ── 4. Batch ONNX embedding extraction ──', t1-t0)
        
        
#         # ── 5. Re‑identification loop (no DB writes) ──
#         for i, emb in enumerate(embeddings):
#             tA = time.time()
#             zz0 = time.time()
#             emb = emb.flatten()
#             emb = emb / (np.linalg.norm(emb) + 1e-8)

#             bbox, tracker_id, conf = crop_meta[i]
#             bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
#             center_str = f"{center_point[0]:.2f},{center_point[1]:.2f}"
#             now = datetime.now()

#             if tracker_id in self.track_to_customer:
#                 customer_id = self.track_to_customer[tracker_id]
#                 if fast_min_dist_to_customer(emb, customer_id) > self.diversity_threshold:
#                     self.db_queue.put(('store_embedding', customer_id, self.cam_id, emb, now))
#                     emb_meta_id = None
#                 else:
#                     emb_meta_id = None
#             else:
#                 zzcount_2 += 1
#                 print('tracker_id is NOT NOT NOT NOT NOT NOT NOT NOT NOT in self.track_to_customer', zzcount_2)
#                 tB = time.time()
#                 matched_cust, dist = fast_match(emb)
#                 tC = time.time()
                
#                 if matched_cust is not None:
#                     customer_id = matched_cust
#                     self.track_to_customer[tracker_id] = customer_id
#                     self.db_queue.put(('update_customer_last_seen', customer_id, now))
#                     if fast_min_dist_to_customer(emb, customer_id) > self.diversity_threshold:
#                         self.db_queue.put(('store_embedding', customer_id, self.cam_id, emb, now))
#                         emb_meta_id = None
#                     else:
#                         emb_meta_id = None
#                 else:
                    
#                     self.new_cust_count += 1
#                     print('# New customer – synchronous, rare (we need the ID immediately)',self.new_cust_count)
#                     # New customer – synchronous, rare (we need the ID immediately)
#                     # Uses self.db_conn (the persistent connection)
#                     cur = self.db_conn.execute(
#                         "INSERT INTO customers (first_seen, last_seen) VALUES (?, ?)", (now, now)
#                     )
#                     customer_id = cur.lastrowid
#                     self.track_to_customer[tracker_id] = customer_id
#                     self.db_queue.put(('store_embedding', customer_id, self.cam_id, emb, now))
#                     emb_meta_id = None
#             tEnd = time.time()
#             # Print per-detection times for the two functions
#             if not tracker_id in self.track_to_customer:
#                 print(f"fast_match took {tC-tB:.4f}s")
#             print(f"Total this detection: {tEnd - now.timestamp():.4f}s")  # now is datetime, better use time.time()
#             # Queue detection log
#             self.db_queue.put(('insert_detection', self.cam_id, bbox_str, center_str, now, emb_meta_id))
#             zz1 = time.time()
#             print(f'# ── 5.{i} Re‑identification loop (no DB writes) ──', zz1-zz0)
        
#         t1 = time.time()
#         print('# ── 5. Re‑identification loop ──', t1-t0)
#         t0 = time.time()
#         # ── 6. Annotate frame with customer_id (optional) ──
#         annotated_frame = frame.copy()
#         annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
#         annotated_frame = self.box_annotator.annotate(annotated_frame, detections)
#         annotated_frame = self.label_annotator.annotate(annotated_frame, detections, labels)
        
#         # # 3. Trigger line counter
#         # # No manual in/out values are passed here.
#         # if len(detections) > 0:
#         #     self.line_zone.trigger(detections=detections)
        
#         # # Draw line + in/out counts
#         # annotated_frame = self.line_zone_annotator.annotate(
#         #     frame=annotated_frame,
#         #     line_counter=self.line_zone)
        
#         # occupancy = self.line_zone.in_count - self.line_zone.out_count
#         # annotated_frame = sv.draw_text(
#         #     scene=annotated_frame,
#         #     text=(
#         #         f"IN: {self.line_zone.in_count}  "
#         #         f"OUT: {self.line_zone.out_count}  "
#         #         f"INSIDE: {occupancy}"
#         #     ),
#         #     text_anchor=sv.Point(220, 30),
#         #     background_color=sv.Color.BLACK,
#         #     text_color=sv.Color.WHITE)
        
#         # Extra text
#         annotated_frame = sv.draw_text(
#             scene=annotated_frame,
#             text=f"FPS: {fps:.1f}",
#             text_anchor=sv.Point(40, 30),
#             background_color=sv.Color.RED,
#             text_color=sv.Color.WHITE)

#         t1 = time.time()
#         print('# ── 6. Annotate frame with customer_id (optional) ──', t1-t0)
#         return annotated_frame


#     # def reader(self):
#     #     while not self.stop_event.is_set():
#     #         ret, frame = self.cap.read()
#     #         if not ret:
#     #             self.stop_event.set()
#     #             break
#     #         self.frame_queue.put(frame)
#     #     self.frame_queue.put(None)
    
#     def reader(self):
#         while not self.stop_event.is_set():
#             try:
#                 ret, frame = self.cap.read()
#                 if not ret:
#                     print("Stream ended or broken – reconnecting...")
#                     self.cap.release()
#                     time.sleep(2)
#                     self.cap = cv2.VideoCapture(self.RTSP_URL)   # store the path as self.RTSP_URL
#                     continue
#                 self.frame_queue.put(frame)
#             except Exception as e:
#                 print(f"Reader caught error: {e}")
#                 time.sleep(2)
#                 self.cap = cv2.VideoCapture(self.RTSP_URL)


#     def detector_worker(self):
#         frame_counter = 0
#         zt0 = time.time()
#         while not self.stop_event.is_set():
#             frame = self.frame_queue.get()
#             if frame is None: break
#             frame_counter += 1
#             detections = self.model.predict(frame)
#             # if frame_counter % 2 == 0:
#             # else:
#             #     # Push empty detections – tracker will propagate
#             #     detections = sv.Detections.empty()
#             # t0 = time.time()
#             # detections = self.model.predict(frame)
#             # t1 = time.time()
#             # print('# ── 1.Detection Time  ──', t1 - t0)
#             if frame_counter % 30 == 0:
#                 elapsed = time.time() - zt0
#                 zt0 = time.time()
#                 print(f"----------------------------------------Detector FPS: {30/elapsed:.1f}")
#             if detections.class_id is not None:
#                 detections = detections[detections.class_id == self.person_class_id]
#             self.det_queue.put((frame, detections))
#         self.det_queue.put(None)

#     def embedder_worker(self):
#         while not self.stop_event.is_set():
#             item = self.det_queue.get()
#             if item is None:
#                 break
#             frame, detections = item
#             # Call your existing process_frame logic, but modified to not write to DB
#             annotated = self.process_frame(frame, detections)
#             with self.lock:
#                 self.latest_frame = annotated
                
#     def start(self):
#         self.stop_event.clear()
#         self.thread_reader = threading.Thread(target=self.reader, daemon=True)
#         self.thread_detector = threading.Thread(target=self.detector_worker, daemon=True)
#         self.thread_embedder = threading.Thread(target=self.embedder_worker, daemon=True)
#         self.thread_reader.start()
#         self.thread_detector.start()
#         self.thread_embedder.start()
        
#     def stop(self):
#         self.stop_event.set()
#         self.thread_reader.join(timeout=2)
#         self.thread_detector.join(timeout=2)
#         self.thread_embedder.join(timeout=2)
        
    
#     def get_latest_frame(self):
#         """Thread‑safe access to the latest annotated frame."""
#         with self.lock:
#             return self.latest_frame.copy() if self.latest_frame is not None else None
        
#     #· generate_frames – a generator that simply reads the latest frame from a shared variable (threading) or queue (multiprocessing). It never touches the database.
#     def generate_frames(self):
#         """Generator for MJPEG – lightweight, just yields the latest frame."""
#         while self.running:
#             with self.lock:
#                 frame_to_yield = self.latest_frame
#             if frame_to_yield is not None:
#                 ret, buffer = cv2.imencode('.jpg', frame_to_yield)
#                 if ret:
#                     yield (b'--frame\r\n'
#                            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
#             # time.sleep(0.03)   # ~30 FPS streaming, throttle CPU
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          





# import os
# import cv2
# import time
# import threading
# import numpy as np
# import pandas as pd
# import multiprocessing
# from queue import Queue
# import supervision as sv
# import onnxruntime as ort
# from loguru import logger
# from datetime import datetime
# import matplotlib.pyplot as plt
# from src.core.config import settings
# from src.core.db_writer import db_queue
# from src.vision.factory import get_detector
# from src.core.database import  (fast_match,
#                                 fast_min_dist_to_customer,
#                                 load_cache,
#                                 get_connection,
#                                 )

# os.environ["OMP_NUM_THREADS"] = "4"
# os.environ["OMP_WAIT_POLICY"] = "ACTIVE"
# os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
# os.environ["KMP_BLOCKTIME"] = "0"
# os.sched_setaffinity(0, {0,1,2,3})  # main thread, but you need the thread's native id; easier to use psutil

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


# class VisionPipeline:
#     def __init__(self, RTSP_URL: str, CAM_ID: str):
#         logger.info("⚙️ Initializing Vision Pipeline components...")
#         self.new_cust_count = 0
#         # Configuration
#         self.cam_id = CAM_ID
#         # --- RAM CACHE ---
#         self.ram_cache = {} 
#         self.RTSP_URL = RTSP_URL
#         self.is_live_stream = self.RTSP_URL.startswith('rtsp')
#         self.frame_counter = 0
#         self.tracked_paths = {}
#         self.tracker = sv.ByteTrack() 
#         self.person_class_id = 0 
#         # Startup: self.db_conn = get_connection() → load_cache(self.db_conn).
#         # Embedder thread: for the “new customer” case, use self.db_conn directly (it’s the only thread that touches it, so no locking needed). All other writes → db_queue.put(...).
#         # Writer thread: its own connection, created inside start_db_writer.
#         # self.db_conn = get_connection()          # persistent write connection
#         # load_cache(self.db_conn)                 # load existing embeddings into memory
#         # Load the embedding cache once at startup – read‑only, then close
#         conn = get_connection(readonly=True)     # ensure your get_connection supports readonly
#         load_cache(conn)
#         conn.close()
        
        
        
#         # --- Threading & process infrastructure ---
#         self.db_queue = db_queue                 # reference to the global queue
#         self.frame_queue = Queue(maxsize=4)
#         self.det_queue = Queue(maxsize=4)
#         self.stop_event = threading.Event()
#         # At the very beginning of your pipeline (e.g., main.py or in __init__)
#         # self.model_input_size = (256, 128)
#         self.model_input_size = (128, 64)
        
#         self.torso_ratio = 2/3
#         self.reid_threshold = settings.REID_THRESHOLD
#         self.diversity_threshold = 0.2 
#          # Tracker mapping
#         self.track_to_customer = {}   # tracker_id -> customer_id
        

#         # Setup UI
#         self.fps_monitor = sv.FPSMonitor()
#         color = sv.ColorPalette.DEFAULT 
#         self.box_annotator = sv.BoxAnnotator(color=color)
#         self.trace_annotator = sv.TraceAnnotator(color=color, trace_length=30) # Reduced length for speed
#         self.label_annotator = sv.LabelAnnotator(color=color, text_color=sv.Color.BLACK)

#         # --- Multiprocessing pipes ---
#         # self.ctx = multiprocessing.get_context('spawn')   # safer on Linux
#         # self.frame_pipe, child_frame_pipe = self.ctx.Pipe(duplex=True)
#         # self.det_pipe, child_det_pipe = self.ctx.Pipe(duplex=True)
        
#         logger.info(f"🔌 Connecting to Video Source: {self.RTSP_URL}")

#         cv2.setNumThreads(0)   # disables FFmpeg's own thread pool
#         # Camera
#         self.cap = cv2.VideoCapture(self.RTSP_URL)
#         self.lock = threading.Lock()
#         self.latest_frame = None           # annotated frame ready for streaming
#         if not self.cap.isOpened():
#             logger.warning(f"❌ COULD NOT OPEN VIDEO SOURCE: {self.RTSP_URL}")
#             # Optional: Retry logic could be logged here
#             self.running = False
#         else:
#             logger.info("✅ Video Source Connected.")
#             self.running = True
            
#             # self.detector_process = self.ctx.Process(
#             #     target=run_detector,
#             #     args=(child_frame_pipe, child_det_pipe, self.person_class_id),
#             #     daemon=True)
#             # self.detector_process.start()
#             try: 
#                 self.detector_model = get_detector()
#                 logger.info("✅ Detector Model Loaded Successfully.")
#             except:
#                 logger.warning("❌ COULD NOT OPEN Dectector Model")
            
#             # start background thread for detection
#             # Spawn the three threads that overlap detection, embedding, and I/O
#             try:
#                 fe_model = settings.FEATURE_EXTRACTOR_MODEL
#                 print(f"🔌 Loading Feature Extractor model: {fe_model}...")
#                 # self.embedder_session = create_session(fe_model, num_threads=(os.cpu_count())//2-2)
#                 # self.embedder_session = create_session(fe_model, num_threads=2)
#                 self.embedder_session = create_session(fe_model, num_threads=1)
                
                
#                 print(f"🔍 Model input size: {self.embedder_session.get_inputs()[0].shape}")
#                 logger.info(f"✅ Loaded Feature Extractor model: {fe_model} successfully")
#             except:
#                 logger.warning(f"❌ COULD NOT LOAD Feature Extractor model")

#     def preprocess_crop(self, frame, bbox):
#         x1, y1, x2, y2 = map(int, bbox)
#         w, h = x2 - x1, y2 - y1
#         cx, cy = (x1+x2)/2, (y1+y2)/2 
#         crop_y2 = int(y1 + h * self.torso_ratio)
#         crop = frame[y1:crop_y2, x1:x2]
#         flag =  crop.size == 0 or w < 20 or h < 20

#         # Preprocess
#         resized = cv2.resize(crop, 
#                                 (self.model_input_size[1], self.model_input_size[0]),
#                             interpolation=cv2.INTER_AREA)
#         # Normalize: (img / 255 - mean) / std
#         mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
#         std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
#         normalized = (resized / 255.0 - mean) / std
        
#         # Convert to CHW
#         input_tensor = np.transpose(normalized, (2, 0, 1)).astype(np.float32)
#         return input_tensor, (x1, y1, x2, y2), (cx, cy), flag

#     def process_frame(self, frame, detections):
#         # current_frame_idx = self.frame_counter
#         self.fps_monitor.tick()
#         fps = self.fps_monitor.fps
#         zzcount_1 = 0
#         zzcount_2 = 0
        
#         # ── 1. Detection & Filtering ──
#         # You are ignoring the detections argument that was already computed by the detector thread and replacing it with a fresh (duplicate) prediction.
#         if detections.class_id is not None:
#             detections = detections[detections.class_id == self.person_class_id]

#         # ── 2. Tracking (ByteTrack) ──
#         t0 = time.time()
#         detections = self.tracker.update_with_detections(detections)
#         t1 = time.time()
#         # print('# ── 2. Tracking (ByteTrack) ──',t1-t0)
#         # ── 3. Collect crops and tracker metadata ──
#         crops_onnx = []
#         crop_meta = []          # (det_box, tracker_id, confidence)
#         labels = []
#         t0 = time.time()
#         idd = 0
#         for det in detections:
#             idd += 1
#             det_box, mask, confidence, class_id, tracker_id, data = det
#             if tracker_id is None:
#                 continue
            
#             zz0 = time.time()
#             # Preprocess the crop (torso, resize, normalise)
#             input_tensor, crop_box, center_point, crop_flag = self.preprocess_crop(frame, det_box)
#             zz1 = time.time()
#             # print(f'#  ── 3.{idd} Preprocess the crop (torso, resize, normalise)', zz1-zz0)                        
#             if crop_flag:
#                 continue

#             crops_onnx.append(input_tensor)
#             crop_meta.append((crop_box, int(tracker_id), confidence))

#             cust = self.track_to_customer.get(int(tracker_id))
#             labels.append(
#                 f"#{int(tracker_id)} {confidence:.2f} "
#                 f"ID:{cust if cust else '?'} ({confidence:.2f})"
#             )

#         if not crops_onnx:
#             return frame   # nothing to process
#         t1 = time.time()
#         # print(f'# ── 3. Collect crops and tracker metadata + Embedding Generation ── FOR {idd} Detections', t1-t0)
        
#         t0 = time.time()
#         # ── 4. Batch ONNX embedding extraction ──
#         batch_input = np.stack(crops_onnx, axis=0)          # (N, 3, 128, 64)
#         embeddings = self.embedder_session.run(None, {'input': batch_input})[0]   # (N, 512)
#         t1 = time.time()
#         # print('# ── 4. Batch ONNX embedding extraction ──', t1-t0)
        
        
#         # ── 5. Re‑identification loop (no DB writes) ──
#         for i, emb in enumerate(embeddings):
#             tA = time.time()
#             zz0 = time.time()
#             emb = emb.flatten()
#             emb = emb / (np.linalg.norm(emb) + 1e-8)

#             bbox, tracker_id, conf = crop_meta[i]
#             bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
#             center_str = f"{center_point[0]:.2f},{center_point[1]:.2f}"
#             now = datetime.now()

#             if tracker_id in self.track_to_customer:
#                 customer_id = self.track_to_customer[tracker_id]
#                 if fast_min_dist_to_customer(emb, customer_id) > self.diversity_threshold:
#                     self.db_queue.put(('store_embedding', customer_id, self.cam_id, emb, now))
#                     emb_meta_id = None
#                 else:
#                     emb_meta_id = None
                    
#             else:
#                 # Unknown tracker – ask the writer to match or register atomically
#                 req_q = Queue(maxsize=1)
#                 self.db_queue.put(('match_or_register', emb, self.cam_id, now, req_q))
#                 customer_id, is_new = req_q.get()   # (customer_id, is_new)
#                 self.track_to_customer[tracker_id] = customer_id
#                 print(customer_id, is_new)
#                 if not is_new:
#                     # Existing customer – maybe store a diverse embedding
#                     if fast_min_dist_to_customer(emb, customer_id) > self.diversity_threshold:
#                         self.db_queue.put(('store_embedding', customer_id, self.cam_id, emb, now))
#                     emb_meta_id = None
#                 else:
#                     # New customer – first embedding already stored by the writer
#                     self.new_cust_count += 1
#                     emb_meta_id = None

            
            
#             # else:
#             #     zzcount_2 += 1
#             #     # print('tracker_id is NOT NOT NOT NOT NOT NOT NOT NOT NOT in self.track_to_customer', zzcount_2)
#             #     tB = time.time()
#             #     matched_cust, dist = fast_match(emb)
#             #     tC = time.time()
                
#             #     if matched_cust is not None:
#             #         customer_id = matched_cust
#             #         self.track_to_customer[tracker_id] = customer_id
#             #         self.db_queue.put(('update_customer_last_seen', customer_id, now))
#             #         if fast_min_dist_to_customer(emb, customer_id) > self.diversity_threshold:
#             #             self.db_queue.put(('store_embedding', customer_id, self.cam_id, emb, now))
#             #             emb_meta_id = None
#             #         else:
#             #             emb_meta_id = None
#             #     else:
                    
#             #         self.new_cust_count += 1
#             #         emb_meta_id = None
#             #         req_q = Queue(maxsize=1)
#             #         self.db_queue.put(('create_customer', now, req_q))
#             #         customer_id = req_q.get()                 # waits microseconds for the reply
#             #         self.track_to_customer[tracker_id] = customer_id
#             #         self.db_queue.put(('store_embedding', customer_id, self.cam_id, emb, now))
     
                    
#             tEnd = time.time()
#             # Print per-detection times for the two functions
#             # if not tracker_id in self.track_to_customer:
#                 # print(f"fast_match took {tC-tB:.4f}s")
#             # print(f"Total this detection: {tEnd - now.timestamp():.4f}s")  # now is datetime, better use time.time()
#             # Queue detection log
#             self.db_queue.put(('insert_detection', self.cam_id, bbox_str, center_str, now, emb_meta_id))
#             zz1 = time.time()
#             # print(f'# ── 5.{i} Re‑identification loop (no DB writes) ──', zz1-zz0)
        
#         t1 = time.time()
#         # print('# ── 5. Re‑identification loop ──', t1-t0)
#         t0 = time.time()
#         # ── 6. Annotate frame with customer_id (optional) ──
#         annotated_frame = frame.copy()
#         annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
#         annotated_frame = self.box_annotator.annotate(annotated_frame, detections)
#         annotated_frame = self.label_annotator.annotate(annotated_frame, detections, labels)
        
#         # Extra text
#         annotated_frame = sv.draw_text(
#             scene=annotated_frame,
#             text=f"FPS: {fps:.1f}",
#             text_anchor=sv.Point(40, 30),
#             background_color=sv.Color.RED,
#             text_color=sv.Color.WHITE)

#         t1 = time.time()
#         # print('# ── 6. Annotate frame with customer_id (optional) ──', t1-t0)
#         return annotated_frame

#     def reader(self):
#         frame_counter = 0
#         while not self.stop_event.is_set():
#             ret, frame = self.cap.read()
#             if not ret:
#                 if self.is_live_stream:
#                     print("Stream ended – reconnecting...")
#                     self.cap.release()
#                     time.sleep(2)
#                     self.cap = cv2.VideoCapture(self.RTSP_URL)
#                     continue
#                 else:
#                     self.stop_event.set()
#                     break
#             frame_counter += 1
#             if frame_counter % 2 == 0:
#                 self.frame_queue.put(frame)      # detector will receive this
#             # Do nothing on skipped frames – no det_queue.put
#         self.frame_queue.put(None)
#         # Poison pill for embedder will come from detector_worker as before
        
#     def embedder_worker(self):
#         last_valid_detections = sv.Detections.empty()
#         while not self.stop_event.is_set():
#             item = self.det_queue.get()
#             if item is None: break
#             frame, detections = item
#             if detections is None:
#                 detections = last_valid_detections
#             else:
#                 last_valid_detections = detections
#             annotated = self.process_frame(frame, detections)
#             with self.lock:
#                 self.latest_frame = annotated
    

#     def detector_worker(self):
#         frame_counter = 0
#         zt0 = time.time()
#         while not self.stop_event.is_set():
#             frame = self.frame_queue.get()
#             if frame is None: break
#             frame_counter += 1
#             detections = self.detector_model.predict(frame)
#             if frame_counter % 30 == 0:
#                 elapsed = time.time() - zt0
#                 zt0 = time.time()
#                 print(f"----------------------------------------Detector FPS: {30/elapsed:.1f}")
#             if detections.class_id is not None:
#                 detections = detections[detections.class_id == self.person_class_id]
#             self.det_queue.put((frame, detections))
#         self.det_queue.put(None)
    
#     def start(self):
#         self.stop_event.clear()
#         self.thread_reader = threading.Thread(target=self.reader, daemon=True)
#         self.thread_detector = threading.Thread(target=self.detector_worker, daemon=True)
#         self.thread_embedder = threading.Thread(target=self.embedder_worker, daemon=True)
#         self.thread_reader.start()
#         self.thread_detector.start()
#         self.thread_embedder.start()
        
#     def stop(self):
#         self.stop_event.set()
#         self.thread_reader.join(timeout=2)
#         self.thread_detector.join(timeout=2)
#         self.thread_embedder.join(timeout=2)
        
    
#     def get_latest_frame(self):
#         """Thread‑safe access to the latest annotated frame."""
#         with self.lock:
#             return self.latest_frame.copy() if self.latest_frame is not None else None
        
#     #· generate_frames – a generator that simply reads the latest frame from a shared variable (threading) or queue (multiprocessing). It never touches the database.
#     def generate_frames(self):
#         """Generator for MJPEG – lightweight, just yields the latest frame."""
#         while self.running:
#             with self.lock:
#                 frame_to_yield = self.latest_frame
#             if frame_to_yield is not None:
#                 ret, buffer = cv2.imencode('.jpg', frame_to_yield)
#                 if ret:
#                     yield (b'--frame\r\n'
#                            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
#             # time.sleep(0.03)   # ~30 FPS streaming, throttle CPU
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            

# import os
# import cv2
# import psutil
# import time
# import threading
# import numpy as np
# import pandas as pd
# import multiprocessing
# from queue import Queue
# import supervision as sv
# import onnxruntime as ort
# from loguru import logger
# from datetime import datetime
# import matplotlib.pyplot as plt
# from src.core.config import settings
# from src.core.db_writer import db_queue
# from src.vision.factory import get_detector
# from src.core.database import  (fast_match,
#                                 fast_min_dist_to_customer,
#                                 load_cache,
#                                 get_connection,
#                                 )

# os.environ["OMP_NUM_THREADS"] = "4"
# os.environ["OMP_WAIT_POLICY"] = "ACTIVE"
# os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
# os.environ["KMP_BLOCKTIME"] = "0"

# proc = psutil.Process(multiprocessing.process.pid)
# proc.cpu_affinity([0,1,2,3])

# os.sched_setaffinity(0, {0,1,2,3})  # main thread, but you need the thread's native id; easier to use psutil

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


# class VisionPipeline:
#     def __init__(self, RTSP_URL: str, CAM_ID: str):
#         logger.info("⚙️ Initializing Vision Pipeline components...")
#         self.new_cust_count = 0
#         # Configuration
#         self.cam_id = CAM_ID
#         # --- RAM CACHE ---
#         self.ram_cache = {} 
#         self.RTSP_URL = RTSP_URL
#         self.is_live_stream = self.RTSP_URL.startswith('rtsp')
#         self.frame_counter = 0
#         self.tracked_paths = {}
#         self.tracker = sv.ByteTrack() 
#         self.person_class_id = 0 
#         # Startup: self.db_conn = get_connection() → load_cache(self.db_conn).
#         # Embedder thread: for the “new customer” case, use self.db_conn directly (it’s the only thread that touches it, so no locking needed). All other writes → db_queue.put(...).
#         # Writer thread: its own connection, created inside start_db_writer.
#         # self.db_conn = get_connection()          # persistent write connection
#         # load_cache(self.db_conn)                 # load existing embeddings into memory
#         # Load the embedding cache once at startup – read‑only, then close
#         conn = get_connection(readonly=True)     # ensure your get_connection supports readonly
#         load_cache(conn)
#         conn.close()
        
        
        
#         # --- Threading & process infrastructure ---
#         self.db_queue = db_queue                 # reference to the global queue
#         self.frame_queue = Queue(maxsize=4)
#         self.det_queue = Queue(maxsize=4)
#         self.stop_event = threading.Event()
#         # At the very beginning of your pipeline (e.g., main.py or in __init__)
#         # self.model_input_size = (256, 128)
#         self.model_input_size = (128, 64)
        
#         self.torso_ratio = 2/3
#         self.reid_threshold = settings.REID_THRESHOLD
#         self.diversity_threshold = 0.2 
#          # Tracker mapping
#         self.track_to_customer = {}   # tracker_id -> customer_id
        

#         # Setup UI
#         self.fps_monitor = sv.FPSMonitor()
#         color = sv.ColorPalette.DEFAULT 
#         self.box_annotator = sv.BoxAnnotator(color=color)
#         self.trace_annotator = sv.TraceAnnotator(color=color, trace_length=30) # Reduced length for speed
#         self.label_annotator = sv.LabelAnnotator(color=color, text_color=sv.Color.BLACK)

#         # --- Multiprocessing pipes ---
#         # self.ctx = multiprocessing.get_context('spawn')   # safer on Linux
#         # self.frame_pipe, child_frame_pipe = self.ctx.Pipe(duplex=True)
#         # self.det_pipe, child_det_pipe = self.ctx.Pipe(duplex=True)
        
#         logger.info(f"🔌 Connecting to Video Source: {self.RTSP_URL}")

#         cv2.setNumThreads(0)   # disables FFmpeg's own thread pool
#         # Camera
#         self.cap = cv2.VideoCapture(self.RTSP_URL)
#         self.lock = threading.Lock()
#         self.latest_frame = None           # annotated frame ready for streaming
#         if not self.cap.isOpened():
#             logger.warning(f"❌ COULD NOT OPEN VIDEO SOURCE: {self.RTSP_URL}")
#             # Optional: Retry logic could be logged here
#             self.running = False
#         else:
#             logger.info("✅ Video Source Connected.")
#             self.running = True
            
#             # self.detector_process = self.ctx.Process(
#             #     target=run_detector,
#             #     args=(child_frame_pipe, child_det_pipe, self.person_class_id),
#             #     daemon=True)
#             # self.detector_process.start()
#             try: 
#                 self.detector_model = get_detector()
#                 logger.info("✅ Detector Model Loaded Successfully.")
#             except:
#                 logger.warning("❌ COULD NOT OPEN Dectector Model")
            
#             # start background thread for detection
#             # Spawn the three threads that overlap detection, embedding, and I/O
#             try:
#                 fe_model = settings.FEATURE_EXTRACTOR_MODEL
#                 print(f"🔌 Loading Feature Extractor model: {fe_model}...")
#                 self.embedder_session = create_session(fe_model, num_threads=(os.cpu_count())//2-2)
#                 # self.embedder_session = create_session(fe_model, num_threads=2)
#                 # self.embedder_session = create_session(fe_model, num_threads=1)
                
                
#                 print(f"🔍 Model input size: {self.embedder_session.get_inputs()[0].shape}")
#                 logger.info(f"✅ Loaded Feature Extractor model: {fe_model} successfully")
#             except:
#                 logger.warning(f"❌ COULD NOT LOAD Feature Extractor model")

#     def preprocess_crop(self, frame, bbox):
#         x1, y1, x2, y2 = map(int, bbox)
#         w, h = x2 - x1, y2 - y1
#         cx, cy = (x1+x2)/2, (y1+y2)/2 
#         crop_y2 = int(y1 + h * self.torso_ratio)
#         crop = frame[y1:crop_y2, x1:x2]
#         flag =  crop.size == 0 or w < 20 or h < 20

#         # Preprocess
#         resized = cv2.resize(crop, 
#                                 (self.model_input_size[1], self.model_input_size[0]),
#                             interpolation=cv2.INTER_AREA)
#         # Normalize: (img / 255 - mean) / std
#         mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
#         std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
#         normalized = (resized / 255.0 - mean) / std
        
#         # Convert to CHW
#         input_tensor = np.transpose(normalized, (2, 0, 1)).astype(np.float32)
#         return input_tensor, (x1, y1, x2, y2), (cx, cy), flag

#     def process_frame(self, frame, detections):
#         # current_frame_idx = self.frame_counter
#         self.fps_monitor.tick()
#         fps = self.fps_monitor.fps
#         zzcount_1 = 0
#         zzcount_2 = 0
        
#         # ── 1. Detection & Filtering ──
#         # You are ignoring the detections argument that was already computed by the detector thread and replacing it with a fresh (duplicate) prediction.
#         if detections.class_id is not None:
#             detections = detections[detections.class_id == self.person_class_id]

#         # ── 2. Tracking (ByteTrack) ──
#         t0 = time.time()
#         detections = self.tracker.update_with_detections(detections)
#         t1 = time.time()
#         # print('# ── 2. Tracking (ByteTrack) ──',t1-t0)
#         # ── 3. Collect crops and tracker metadata ──
#         crops_onnx = []
#         crop_meta = []          # (det_box, tracker_id, confidence)
#         labels = []
#         t0 = time.time()
#         idd = 0
#         for det in detections:
#             idd += 1
#             det_box, mask, confidence, class_id, tracker_id, data = det
#             if tracker_id is None:
#                 continue
            
#             zz0 = time.time()
#             # Preprocess the crop (torso, resize, normalise)
#             input_tensor, crop_box, center_point, crop_flag = self.preprocess_crop(frame, det_box)
#             zz1 = time.time()
#             # print(f'#  ── 3.{idd} Preprocess the crop (torso, resize, normalise)', zz1-zz0)                        
#             if crop_flag:
#                 continue

#             crops_onnx.append(input_tensor)
#             crop_meta.append((crop_box, int(tracker_id), confidence))

#             cust = self.track_to_customer.get(int(tracker_id))
#             labels.append(
#                 f"#{int(tracker_id)} {confidence:.2f} "
#                 f"ID:{cust if cust else '?'} ({confidence:.2f})"
#             )

#         if not crops_onnx:
#             return frame   # nothing to process
#         t1 = time.time()
#         # print(f'# ── 3. Collect crops and tracker metadata + Embedding Generation ── FOR {idd} Detections', t1-t0)
        
#         t0 = time.time()
#         # ── 4. Batch ONNX embedding extraction ──
#         batch_input = np.stack(crops_onnx, axis=0)          # (N, 3, 128, 64)
#         embeddings = self.embedder_session.run(None, {'input': batch_input})[0]   # (N, 512)
#         t1 = time.time()
#         # print('# ── 4. Batch ONNX embedding extraction ──', t1-t0)
        
        
#         # ── 5. Re‑identification loop (no DB writes) ──
#         for i, emb in enumerate(embeddings):
#             tA = time.time()
#             zz0 = time.time()
#             emb = emb.flatten()
#             emb = emb / (np.linalg.norm(emb) + 1e-8)

#             bbox, tracker_id, conf = crop_meta[i]
#             bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
#             center_str = f"{center_point[0]:.2f},{center_point[1]:.2f}"
#             now = datetime.now()

#             if tracker_id in self.track_to_customer:
#                 customer_id = self.track_to_customer[tracker_id]
#                 if fast_min_dist_to_customer(emb, customer_id) > self.diversity_threshold:
#                     self.db_queue.put(('store_embedding', customer_id, self.cam_id, emb, now))
#                     emb_meta_id = None
#                 else:
#                     emb_meta_id = None
                    
#             else:
#                 # Unknown tracker – ask the writer to match or register atomically
#                 req_q = Queue(maxsize=1)
#                 self.db_queue.put(('match_or_register', emb, self.cam_id, now, req_q))
#                 customer_id, is_new = req_q.get()   # (customer_id, is_new)
#                 self.track_to_customer[tracker_id] = customer_id
#                 print(customer_id, is_new)
#                 if not is_new:
#                     # Existing customer – maybe store a diverse embedding
#                     if fast_min_dist_to_customer(emb, customer_id) > self.diversity_threshold:
#                         self.db_queue.put(('store_embedding', customer_id, self.cam_id, emb, now))
#                     emb_meta_id = None
#                 else:
#                     # New customer – first embedding already stored by the writer
#                     self.new_cust_count += 1
#                     emb_meta_id = None

#             tEnd = time.time()
#             # Print per-detection times for the two functions
#             # if not tracker_id in self.track_to_customer:
#                 # print(f"fast_match took {tC-tB:.4f}s")
#             # print(f"Total this detection: {tEnd - now.timestamp():.4f}s")  # now is datetime, better use time.time()
#             # Queue detection log
#             self.db_queue.put(('insert_detection', self.cam_id, bbox_str, center_str, now, emb_meta_id))
#             zz1 = time.time()
#             # print(f'# ── 5.{i} Re‑identification loop (no DB writes) ──', zz1-zz0)
        
#         t1 = time.time()
#         # print('# ── 5. Re‑identification loop ──', t1-t0)
#         t0 = time.time()
#         # ── 6. Annotate frame with customer_id (optional) ──
#         annotated_frame = frame.copy()
#         annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
#         annotated_frame = self.box_annotator.annotate(annotated_frame, detections)
#         annotated_frame = self.label_annotator.annotate(annotated_frame, detections, labels)
        
#         # Extra text
#         annotated_frame = sv.draw_text(
#             scene=annotated_frame,
#             text=f"FPS: {fps:.1f}",
#             text_anchor=sv.Point(40, 30),
#             background_color=sv.Color.RED,
#             text_color=sv.Color.WHITE)

#         t1 = time.time()
#         # print('# ── 6. Annotate frame with customer_id (optional) ──', t1-t0)
#         return annotated_frame

#     def reader(self):
#         frame_counter = 0
#         while not self.stop_event.is_set():
#             ret, frame = self.cap.read()
#             if not ret:
#                 if self.is_live_stream:
#                     print("Stream ended – reconnecting...")
#                     self.cap.release()
#                     time.sleep(2)
#                     self.cap = cv2.VideoCapture(self.RTSP_URL)
#                     continue
#                 else:
#                     self.stop_event.set()
#                     break
#             frame_counter += 1
#             if frame_counter % 2 == 0:
#                 # DROP FRAMES.
#                 # Real-time systems ALWAYS drop frames.
#                 # Never queue infinitely.
#                 try:
#                     self.frame_queue.put_nowait(idx)
#                 except queue.Full:
#                     pass
                
                
#             # Do nothing on skipped frames – no det_queue.put
#         self.frame_queue.put(None)
#         # Poison pill for embedder will come from detector_worker as before
        
#     def embedder_worker(self):
#         last_valid_detections = sv.Detections.empty()
#         while not self.stop_event.is_set():
#             item = self.det_queue.get()
#             if item is None: break
#             frame, detections = item
#             if detections is None:
#                 detections = last_valid_detections
#             else:
#                 last_valid_detections = detections
#             annotated = self.process_frame(frame, detections)
#             with self.lock:
#                 self.latest_frame = annotated
    

#     def detector_worker(self):
#         frame_counter = 0
#         zt0 = time.time()
#         while not self.stop_event.is_set():
#             frame = self.frame_queue.get()
#             if frame is None: break
#             frame_counter += 1
#             detections = self.detector_model.predict(frame)
#             if frame_counter % 30 == 0:
#                 elapsed = time.time() - zt0
#                 zt0 = time.time()
#                 print(f"----------------------------------------Detector FPS: {30/elapsed:.1f}")
#             if detections.class_id is not None:
#                 detections = detections[detections.class_id == self.person_class_id]
#             self.det_queue.put((frame, detections))
#         self.det_queue.put(None)
    
#     def start(self):
#         self.stop_event.clear()
#         self.thread_reader = multiprocessing.Process(target=self.reader, daemon=True)
#         self.thread_detector = multiprocessing.Process(target=self.detector_worker, daemon=True)
#         self.thread_embedder = multiprocessing.Process(target=self.embedder_worker, daemon=True)
#         self.thread_reader.start()
#         self.thread_detector.start()
#         self.thread_embedder.start()
        
#     def stop(self):
#         self.stop_event.set()
#         self.thread_reader.join(timeout=2)
#         self.thread_detector.join(timeout=2)
#         self.thread_embedder.join(timeout=2)
        
    
#     def get_latest_frame(self):
#         """Thread‑safe access to the latest annotated frame."""
#         with self.lock:
#             return self.latest_frame.copy() if self.latest_frame is not None else None
        
#     #· generate_frames – a generator that simply reads the latest frame from a shared variable (threading) or queue (multiprocessing). It never touches the database.
#     def generate_frames(self):
#         """Generator for MJPEG – lightweight, just yields the latest frame."""
#         while self.running:
#             with self.lock:
#                 frame_to_yield = self.latest_frame
#             if frame_to_yield is not None:
#                 ret, buffer = cv2.imencode('.jpg', frame_to_yield)
#                 if ret:
#                     yield (b'--frame\r\n'
#                            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
#             # time.sleep(0.03)   # ~30 FPS streaming, throttle CPU
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
      
            





























































































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

# SKIP_FRAME = (None, None, None, None)

# def pin_process(cores):
#     try:
#         psutil.Process(os.getpid()).cpu_affinity(list(cores))
#     except Exception:
#         pass

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
    
# def frame_view(shm: shared_memory.SharedMemory, frame_shape, frame_bytes: int, idx: int):
#     offset = idx * frame_bytes
#     return np.ndarray(frame_shape, dtype=np.uint8, buffer=shm.buf, offset=offset)
    
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
#     flag =  (crop.size == 0) or (w < 20) or (h < 20)
#     resized = cv2.resize(crop, (model_input_size[1], model_input_size[0]),
#                         interpolation=cv2.INTER_AREA)
#     # Normalize: (img / 255 - mean) / std
#     mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
#     std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
#     normalized = (resized / 255.0 - mean) / std
    
#     # Convert to CHW
#     input_tensor = np.transpose(normalized, (2, 0, 1)).astype(np.float32)
#     return input_tensor, (x1, y1, x2, y2), (cx, cy), w, h, flag

# class VisionPipeline:
#     def __init__(self,
#                 RTSP_URL,
#                 CAM_ID,
#                 ctx,
#                 free_slots,
#                 frame_ready_queue,
#                 det_queue,
#                 stop_event,
#                 db_queue,
#                 response_queue,
#                 buffer_slots=4):
                
#         #         embedder_worker
#         #       │
#         #       │  puts messages like:
#         #       │  ("match_or_register", emb, cam_id, now, request_id, ...)
#         #       │  ("store_embedding", customer_id, ...)
#         #       │  ("update_customer_last_seen", customer_id, now)
#         #       │
#         #       ▼
#         #    db_queue  (mp.Queue)
#         #       │
#         #       ▼
#         # db_writer_worker  (separate process)
#         #       │
#         #       ├── runs fast_match()         → reads embedding_cache (in-memory numpy)
#         #       ├── runs store_embedding()    → writes SQLite + updates embedding_cache
#         #       ├── runs UPDATE customers     → writes SQLite
#         #       │
#         #       └── puts reply into reply_queue (the per-camera response_queue)
#         #                │
#         #                ▼
#         #          embedder_worker.response_queue.get()
#         #          → gets back (request_id, customer_id, is_new, dist)

#         # Reasons to use the db_queue:
#         #     Reason 1: SQLite only allows one writer.
#         #     Reason 2: Writes are slow, inference can't wait. Pushing to a queue is microseconds.


        
        
#         # Queue contracts — do not break these
#         # frame_ready_queue : (cam_id: int, idx: int)
#         #   reader → batched_detector_worker
#         #
#         # det_queue         : (idx: int | None, xyxy: np.ndarray | None,
#         #                      confidence: np.ndarray | None, class_id: np.ndarray | None)
#         #   batched_detector_worker → embedder_worker
#         #   None tuple = skipped frame, reuse last detections
#         #
#         # db_queue          : ("match_or_register", emb, cam_id, now, request_id,
#         #                       center_point, bbox_w, bbox_h, track_id, quality_score, reply_queue)
#         #                   | ("store_embedding", customer_id, cam_id, emb, now,
#         #                       center_point, bbox_w, bbox_h, quality_score, track_id)
#         #                   | ("update_customer_last_seen", customer_id, now)
#         #                   | ("min_dist_to_customer", emb, customer_id, request_id, reply_queue)
#         #   embedder_worker → db_writer_worker
#         #
#         # response_queue    : (request_id, customer_id, is_new, dist)  ← match_or_register reply
#         #                   | (request_id, dist)                        ← min_dist reply
#         #   db_writer_worker → embedder_worker (per-camera queue)
        
#         # So the flow of a slot index is:
#         #     free_slots → reader → frame_ready_queue → detector → det_queue → embedder → free_slots
#         #     It's a ring. The slot number travels through the pipeline tracking which memory slot contains the current frame.
        
        
        
        
        
        
        
        
        
#         # SLOT RING INVARIANTS — never violate these:
#         #
#         # 1. Every idx that leaves free_slots MUST eventually return to free_slots.
#         #    Leak one slot and the ring drains. After buffer_slots frames, the
#         #    reader blocks forever on free_slots.get_nowait().
#         #
#         # 2. Only real frame indices (integers) enter the slot ring.
#         #    None is never a slot index. Skipped frames use SKIP_FRAME sentinel.
#         #
#         # 3. The embedder is the normal slot returner.
#         #    The batched detector is the emergency slot returner (when embedder backed up).
#         #    The reader is the emergency slot returner (when detector backed up).
#         #    No one else touches free_slots.
#         #
#         # 4. SKIP_FRAME = (None, None, None, None) never enters frame_ready_queue.
#         #    It only goes into det_queue directly from reader_worker.
#         self.model_input_size = (256, 128)
#         self.ctx = ctx
#         self.RTSP_URL = RTSP_URL
#         self.cam_id = CAM_ID
#         self.torso_ratio = 2/3
#         self.free_slots = free_slots
#         self.frame_ready_queue= frame_ready_queue
#         self.det_queue  = det_queue
#         self.stop_event = stop_event
#         self.db_queue   = db_queue
#         self.response_queue = response_queue
#         self.buffer_slots = buffer_slots
#         self.track_positions = defaultdict(list)   # tracker_id -> list of (time, centroid)
#         self.frame_ready_queue = frame_ready_queue
        
#         probe = cv2.VideoCapture(self.RTSP_URL)
#         ok, frame = probe.read()
#         probe.release()
#         # Class‑level constant – adjust as you like
#         self.frame_shape = settings.FRAME_SHAPE   # (512, 512, 3) — fixed
#         self.frame_bytes = settings.FRAME_BYTES   # fixed
#         self.online = True                        # reader handles offline state
    
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
#                 self.track_positions,                
#                 self.cam_id
#             ),
#             daemon=True
#         )
        
#         self.p_reader = self.ctx.Process(
#             target=reader_worker,
#             args=(
#                 self.RTSP_URL,
#                 self.cam_id,
#                 self.input_shm_name,
#                 self.frame_shape,
#                 self.frame_bytes,
#                 self.free_slots,
#                 self.frame_ready_queue,
#                 self.det_queue,
#                 self.stop_event,
#             ),
#             daemon=True
#         )
#         self.p_reader.start()
#         self.p_embedder.start()

        
#     def stop(self):
#         self.stop_event.set()
#         try:
#             self.frame_ready_queue.put_nowait(None)
#         except:
#             pass

#         try:
#             self.det_queue.put_nowait(None)
#         except:
#             pass

#         self.p_reader.join(timeout=2)
#         # self.p_detector.join(timeout=2)
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
#     run_embedding,
#     person_class_id=0
# ):
#     # ── 0. Guard against None or completely empty detections ──
#     if detections is None or len(detections) == 0:
#         print('# Still draw the zone outline and FPS on a clean copy')
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
#         mask = zone.trigger(detections=detections)
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
#                 print(f"Customer {tid} left zone {zone_idx} after {elapsed:.1f}s")

#         # Draw zone and its boxes
#         frame = box_annotator.annotate(scene=frame, detections=detections_in_zone)
#         frame = zone_annotator.annotate(scene=frame)

#     # ── Loitering, unique visitors, crop collection ──
#     crops_onnx, crop_meta, labels = [], [], []
#     loitering_tracker_ids = set()
#     if run_embedding:
#         for det in detections:
#             det_box, det_mask, det_conf, class_id, tracker_id, data = det
#             if tracker_id is None:
#                 continue

#             # Crop preprocessing
#             input_tensor, crop_box, center_point, bbox_w, bbox_h, crop_flag = preprocess_crop(
#                 frame, 
#                 det_box,
#                 model_input_size=(256, 128),
#                 torso_ratio=1)
            
#             # Loitering check
#             if check_loitering(tracker_id, time.time(), center_point, track_positions):
#                 loitering_tracker_ids.add(tracker_id)

#             if not crop_flag:
#                 # Unique visitor counting
#                 unique_visitors.add(tracker_id)
#                 crops_onnx.append(input_tensor)
#                 crop_meta.append((crop_box, center_point, bbox_w, bbox_h, int(tracker_id), det_conf))
#     else:
#         # Skipped frame – loitering check still runs, but no new crops
#         for det in detections:
#             det_box, det_mask, det_conf, class_id, tracker_id, data = det
#             if tracker_id is None:
#                 continue
#             x1, y1, x2, y2 = det_box
#             center_point = ((x1 + x2) / 2, (y1 + y2) / 2)
#             if check_loitering(tracker_id, time.time(), center_point, track_positions):
#                 loitering_tracker_ids.add(tracker_id)
#         # labels will be filled from track_to_customer below

#     # ── 4. Embedding & Re‑ID (only if crops exist) ──
#     if crops_onnx and run_embedding:
#         with open('data/embeddings_WTF.txt', 'a+') as f:
#             batch_input = np.stack(crops_onnx, axis=0)
#             embeddings = embedder_session.run(None, {"input": batch_input})[0]
#             for i, emb in enumerate(embeddings):
#                 emb = emb.flatten()
#                 emb = emb / (np.linalg.norm(emb) + 1e-8)
#                 bbox, center_point, bbox_w, bbox_h, tracker_id, confidence_score = crop_meta[i]
#                 # print(111111111111,tracker_id)
#                 temp_f_text = f'{i+1}_tracker_id: {tracker_id} '
#                 if tracker_id in track_to_customer:
#                     customer_id = track_to_customer[tracker_id]
#                     # Ask the writer for the minimum distance to this customer
#                     ########################### What is the use for this code?
#                     # I send my request with a unique ID.
#                     # I start listening.
#                     # I pull the next available message.
#                     # If the ID matches, I keep it.
#                     # If not, I immediately push it back so the real owner can later pull it.
#                     # I repeat until my own message appears.
#                     # Because there are only a few active requests at any moment, the loop rarely spins more than once or twice, so it’s fast.
#                     request_id = f"dist_{cam_id}_{tracker_id}_{time.time()}"
#                     # db_queue.put(("min_dist_to_customer", emb, customer_id, request_id))
#                     db_queue.put((
#                         "min_dist_to_customer",
#                         emb, customer_id, request_id,
#                         response_queue        # ← same pattern
#                     ))
#                     rid, min_dist = response_queue.get()
#                     # while True:
#                     #     rid, min_dist = response_queue.get()   # take one message out
#                     #     if rid == request_id:                  # is it mine?
#                     #         break                              # yes – I'm done
#                     #     response_queue.put((rid, min_dist))    # no – return it for someone else

#                     # print(2222222222, min_dist, f'tracker_id in track_to_customer: {tracker_id}')
#                     # Store only if the embedding is diverse
#                     if min_dist > settings.DIVERSITY_THRESHOLD:
#                         db_queue.put(("store_embedding", customer_id, cam_id, emb, time.time(), bbox_w, bbox_h))
#                         new_temp_f_text = f'✅: customer_id: {customer_id}  min_dist: {min_dist:.4f} ===> "store_embedding"\n'
#                         f.write(temp_f_text + new_temp_f_text)
#                     else:
#                         new_temp_f_text = f'✅: customer_id: {customer_id}  min_dist: {min_dist:.4f} ===> "update_customer_last_seen"\n'
#                         f.write(temp_f_text + new_temp_f_text)
#                         db_queue.put(("update_customer_last_seen", customer_id, time.time()))
                    
#                 else:
#                     # Unknown tracker – match or register atomically
#                     request_id = f"match_{cam_id}_{tracker_id}_{time.time()}"
#                     # print(center_point, type(center_point))
#                     # db_queue.put(("match_or_register", emb, cam_id, time.time(), request_id, center_point, bbox_w, bbox_h, tracker_id, confidence_score))
#                     request_id = f"match_{cam_id}_{tracker_id}_{time.time()}"
#                     db_queue.put((
#                         "match_or_register",
#                         emb, cam_id, time.time(), request_id,
#                         center_point, bbox_w, bbox_h, tracker_id, confidence_score,
#                         response_queue        # ← pass the camera's own queue
#                     ))
#                     rid, customer_id, is_new, match_dist = response_queue.get()  # no spin, just block
#                     # request_id check is now optional — only this camera uses this queue
#                     # but keep it for safety during debugging:
#                     assert rid == request_id
#                     # while True:
#                     #     rid, customer_id, is_new, match_dist = response_queue.get()
#                     #     if rid == request_id:
#                     #         break
#                     #     response_queue.put((rid, customer_id, is_new, match_dist))

#                     track_to_customer[tracker_id] = customer_id
#                     # For existing matches, store diverse embeddings
#                     if not is_new and match_dist > settings.DIVERSITY_THRESHOLD:
#                         #Those commands are fire‑and‑forget. They don’t need a reply, so you do not pass a response_queue in their tuples.
#                         db_queue.put(("store_embedding", customer_id, cam_id, emb, time.time(), bbox_w, bbox_h))
#                         new_temp_f_text = f'❌: customer_id: {customer_id}  match_dist: {match_dist} ===> "store_embedding"\n'
#                         f.write(temp_f_text + new_temp_f_text)
#                     else:
#                         #Those commands are fire‑and‑forget. They don’t need a reply, so you do not pass a response_queue in their tuples.
#                         db_queue.put(("update_customer_last_seen", customer_id, time.time()))
#                         new_temp_f_text = f'❌: customer_id: {customer_id}  match_dist: {match_dist} ===> "update_customer_last_seen"\n'
#                         f.write(temp_f_text + new_temp_f_text)
#                 labels.append(f"#Track: {int(tracker_id)} {confidence_score:.2f} ID:{customer_id}")
#                 # labels.append(f"#{confidence_score:.2f} ID:{customer_id}")
#                 # print(track_to_customer)
            
#             f.write("\n\n==================================================================================================================="+'\n\n')
#     else:
#         # Build labels from existing track_to_customer (no new embeddings)
#         labels = []
#         for det in detections:
#             _, _, _, _, tracker_id, _ = det
#             if tracker_id is None:
#                 continue
#             customer_id = track_to_customer.get(tracker_id)
#             if customer_id is not None:
#                 labels.append(f"#Track: {int(tracker_id)} ID:{customer_id}")
#             else:
#                 labels.append(f"#Track: {int(tracker_id)} ?")    


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
#     frame_ready_queue,
#     det_queue,
#     free_slots,
#     stop_event
# ):
#     pin_process([1,2,3])
#     shm = shared_memory.SharedMemory(name=shm_name)
#     detector_model = get_detector()
#     while not stop_event.is_set():
#         try:
#             idx = frame_ready_queue.get(timeout=0.1)
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
#                   cam_id,
#                   shm_name,
#                   frame_shape,
#                   frame_bytes,
#                   free_slots,
#                   frame_ready_queue,
#                   det_queue,
#                   stop_event):
#     pin_process([0])
#     shm = shared_memory.SharedMemory(name=shm_name)

#     cap = None
#     first_attempt = True
#     online = False
#     consecutive_failures = 0
#     max_backoff = 60.0
#     offline_frame = np.zeros(frame_shape, dtype=np.uint8)
#     cv2.putText(offline_frame, "Stream offline", (50, 240),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#     frame_counter = 0

#     while not stop_event.is_set():
#         if not online:
#             if cap is not None:
#                 cap.release()
#                 cap = None

#             if first_attempt and consecutive_failures < 6:
#                 first_attempt = False
#                 sleep_time = 0
#             else:
#                 sleep_time = min(1.0 * (2 ** consecutive_failures), max_backoff)

#             stop_event.wait(sleep_time)
#             if stop_event.is_set():
#                 break

#             cap = cv2.VideoCapture(rtsp_url)
#             if not cap.isOpened():
#                 consecutive_failures += 1
#                 # Signal embedder that stream is offline
#                 try:
#                     det_queue.put_nowait((None, None, None, None))
#                 except queue.Full:
#                     pass
#                 continue

#             online = True
#             consecutive_failures = 0

#         ret, frame = cap.read()
#         if not ret:
#             online = False
#             consecutive_failures += 1
#             try:
#                 det_queue.put_nowait((None, None, None, None))
#             except queue.Full:
#                 pass
#             continue

#         frame_counter += 1
#         consecutive_failures = 0

#         if frame_counter % 3 == 0:
#             # Keyframe — needs a real slot
#             try:
#                 idx = free_slots.get_nowait()
#             except queue.Empty:
#                 # No slot available, detector is backed up — drop this keyframe
#                 # Don't crash, don't block, just skip
#                 continue
#             _write_frame_to_slot(shm, idx, frame, frame_shape, frame_bytes)
#             try:
#                 frame_ready_queue.put_nowait((cam_id, idx))
#             except queue.Full:
#                 # Detector queue full — return slot immediately
#                 free_slots.put_nowait(idx)
#         else:
#             # Skipped frame — no slot needed at all
#             try:
#                 det_queue.put_nowait(SKIP_FRAME)
#             except queue.Full:
#                 pass  # embedder backed up, drop the signal

#     if cap is not None:
#         cap.release()
#     shm.close()

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
#     track_positions,
#     cam_id
# ):
#     pin_process([4,5])
#     input_shm = shared_memory.SharedMemory(name=input_shm_name)
#     output_shm = shared_memory.SharedMemory(name=output_shm_name)
#     embedder_session = create_session(settings.FEATURE_EXTRACTOR_MODEL, num_threads=1)
#     track_to_customer = {}
#     tracker = sv.ByteTrack(lost_track_buffer=120) # 120 frames ==> This stops the tracker from killing a track when a person is briefly occluded or missed by the detector.
#     fps_monitor = sv.FPSMonitor()
#     color = sv.ColorPalette.DEFAULT
#     box_annotator = sv.EllipseAnnotator(color=color)
#     trace_annotator = sv.TraceAnnotator(color=color, trace_length=30)
#     label_annotator = sv.LabelAnnotator(color=color, text_color=sv.Color.BLACK)
    
#     colors = sv.ColorPalette.DEFAULT
#     h,w,c = frame_shape
#     # Build the full‑frame polygon (top‑left → top‑right → bottom‑right → bottom‑left)

#     polygons = [
#             np.array([
#             [0, 0],
#             [w, 0],
#             [w, h],
#             [0, h]
#         ], dtype=np.int32)
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
#     last_valid_detections = sv.Detections.empty()
#     while not stop_event.is_set():
#         try:
#             item = det_queue.get(timeout=0.1)
#         except queue.Empty:
#             continue

#         if item is None:
#             break  # shutdown signal

#         idx, xyxy, confidence, class_id = item  # always 4 values now

#         # Skipped frame path
#         if idx is None:
#             # No shared memory involved — reuse last state
#             detections    = last_valid_detections
#             run_embedding = False
#             # Read last annotated frame from output shm for re-display
#             frame = np.ndarray(
#                 frame_shape, dtype=np.uint8, buffer=output_shm.buf
#             ).copy()
#         else:
#             # Real frame path
#             frame = frame_view(input_shm, frame_shape, frame_bytes, idx).copy()
#             if xyxy is not None:
#                 detections = sv.Detections(
#                     xyxy=xyxy, confidence=confidence, class_id=class_id)
#                 last_valid_detections = detections
#                 run_embedding = True
#             else:
#                 # Real frame, but detector found nothing
#                 detections    = sv.Detections.empty()
#                 last_valid_detections = detections
#                 run_embedding = False

        
#         # Free the slot ONLY if it was a real frame
#         if idx is not None:
#             try:
#                 free_slots.put_nowait(idx)
#             except queue.Full:
#                 pass  # should never happen but don't crash
        
#         if item is None:
#             break
        
#         # --- Tick the FPS counter ---
#         # fps_monitor.tick()
#         # print(frame_shape,frame_bytes,idx)
#         # print(input_shm)
#         # frame = frame_view(input_shm,frame_shape,frame_bytes,idx)
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
#             label_annotator=label_annotator,
#             run_embedding=run_embedding
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

# def batched_detector_worker(
#     frame_ready_queue,   # mp.Queue of (cam_id, idx)  — frames ready for detection
#     det_queues,          # dict[cam_id -> mp.Queue]   — where to send results
#     free_slots_queues,
#     shm_names,           # dict[cam_id -> str]        — shared memory names
#     frame_shape,         # (H, W, C) — same for all cameras
#     frame_bytes,         # int
#     stop_event,
#     batch_timeout=0.02,  # seconds to wait collecting a full batch
# ):
#     """
#     Single process that owns ONE YOLO session.
#     Collects frames from all cameras, runs one batched inference,
#     routes detections back to each camera's det_queue.
#     """
#     # Attach to all shared memory blocks
#     shm_blocks = {
#         cam_id: shared_memory.SharedMemory(name=name)
#         for cam_id, name in shm_names.items()
#     }
#     n_cams     = len(shm_blocks)
#     detector   = get_detector()   # one YOLO session for ALL cameras

#     while not stop_event.is_set():

#         # ── Collect a batch: wait up to batch_timeout for frames ──────────
#         pending = {}   # cam_id -> slot_idx
#         deadline = time.time() + batch_timeout

#         while len(pending) < n_cams:
#             remaining = deadline - time.time()
#             if remaining <= 0:
#                 break
#             try:
#                 cam_id, idx = frame_ready_queue.get(timeout=remaining)
#                 pending[cam_id] = idx
#             except queue.Empty:
#                 break

#         if not pending:
#             continue

#         # ── Build frame list in a stable order ───────────────────────────
#         cam_order = list(pending.keys())
#         frames    = []
#         for cam_id in cam_order:
#             idx  = pending[cam_id]
#             shm  = shm_blocks[cam_id]
#             frame = frame_view(shm, frame_shape, frame_bytes, idx)
#             frames.append(frame.copy())   # copy out of shm before inference

#         # ── Single batched inference ──────────────────────────────────────
#         detections_list = detector.predict_batch(frames)   # list[sv.Detections]

#         # ── Route results back to each camera's det_queue ─────────────────
#         for cam_id, det, frame in zip(cam_order, detections_list, frames):
#             idx = pending[cam_id]
#             xyxy       = det.xyxy        if len(det) > 0 else None
#             confidence = det.confidence  if len(det) > 0 else None
#             class_id   = det.class_id   if len(det) > 0 else None
#             try:
#                 # idx is always a real int here — batched detector only
#                 # receives real frames from frame_ready_queue
#                 det_queues[cam_id].put_nowait((idx, xyxy, confidence, class_id))
#             except queue.Full:
#                 # Embedder backed up — we must free the slot here
#                 # or it leaks and free_slots empties permanently
#                 # We need the free_slots queue for this camera
#                 # (see Step 4 below for how to pass it)
#                 free_slots_queues[cam_id].put_nowait(idx)

#     for shm in shm_blocks.values():
#         shm.close()




























