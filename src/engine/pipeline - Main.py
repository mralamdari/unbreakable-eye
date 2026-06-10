# import cv2
# import supervision as sv
# from src.vision.factory import get_detector
# from src.core.config import settings
# from loguru import logger  # <--- Import


# class VisionPipeline:
#     def __init__(self):
#         logger.info("⚙️ Initializing Vision Pipeline components...")
#         self.model = get_detector()

#         # --- THE FIX IS HERE ---
#         # Instead of your broken SORTTracker, use this:

#         self.tracker = sv.ByteTrack() 
#         self.person_class_id = 0  # YOLOv8 COCO person class
#         # -----------------------

#         # Setup UI
#         self.fps_monitor = sv.FPSMonitor()
#         color = sv.ColorPalette.DEFAULT 
#         self.box_annotator = sv.BoxAnnotator(color=color)
#         self.trace_annotator = sv.TraceAnnotator(color=color, trace_length=30) # Reduced length for speed
#         self.label_annotator = sv.LabelAnnotator(color=color, text_color=sv.Color.BLACK)

        
#         # Auto-config state
#         self.line_zone = None
#         self.line_zone_annotator = None
#         self.line_initialized = False

#         # Change these depending on your entrance geometry
#         self.door_orientation = "horizontal"   # "horizontal" or "vertical"
#         self.door_position_ratio = 0.75        # 55% down the frame if horizontal
#         self.door_margin_ratio = 0.01          # keep line inside image edges
#         self.trigger_anchor = sv.Position.BOTTOM_CENTER
        
#         logger.info(f"🔌 Connecting to Video Source: {settings.RTSP_URL}")
        
#         # Camera
#         self.cap = cv2.VideoCapture(settings.RTSP_URL)
#         if not self.cap.isOpened():
#             logger.error(f"❌ COULD NOT OPEN VIDEO SOURCE: {settings.RTSP_URL}")
#             # Optional: Retry logic could be logged here
#         else:
#             logger.info("✅ Video Source Connected.")

#     # def process_frame(self, frame):
#     #     self.fps_monitor.tick()
#     #     fps = self.fps_monitor.fps
        
#     #     # 1. Inference
#     #     detections = self.model.predict(frame)

#     #     # 2. Tracking (The syntax changes slightly for ByteTrack)
#     #     detections = self.tracker.update_with_detections(detections)
        
#     #     # 3. Annotation
#     #     labels = []
#     #     annotated_frame = frame.copy()
#     #     if detections.tracker_id is not None and detections.confidence is not None \
#     #        and len(detections.tracker_id) == len(detections.confidence): # Check lengths match
#     #         labels = [
#     #             f"#{tracker_id} {conf:.2f}" 
#     #             for tracker_id, conf in zip(detections.tracker_id, detections.confidence)
#     #         ]

#     #     annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
#     #     annotated_frame = self.box_annotator.annotate(annotated_frame, detections)
#     #     annotated_frame = self.label_annotator.annotate(annotated_frame, detections, labels)
        
#     #     # FPS
#     #     annotated_frame = sv.draw_text(
#     #         scene=annotated_frame,
#     #         text=f"FPS: {fps:.1f}",
#     #         text_anchor=sv.Point(40, 30),
#     #         background_color=sv.Color.RED,
#     #         text_color=sv.Color.WHITE
#     #     )

#     #     return annotated_frame

#     def _setup_doorway_line(self, frame):
#         height, width = frame.shape[:2]

#         margin_x = int(width * self.door_margin_ratio)
#         margin_y = int(height * self.door_margin_ratio)

#         if self.door_orientation == "horizontal":
#             y = int(height * self.door_position_ratio)

#             start = sv.Point(margin_x, y)
#             end = sv.Point(width - margin_x, y)

#         else:
#             x = int(width * self.door_position_ratio)

#             start = sv.Point(x, margin_y)
#             end = sv.Point(x, height - margin_y)

#         self.line_zone = sv.LineZone(
#             start=start,
#             end=end,
#             triggering_anchors=[self.trigger_anchor]
#         )

#         self.line_zone_annotator = sv.LineZoneAnnotator(
#             thickness=2,
#             text_thickness=1,
#             text_scale=0.7
#         )

#         self.line_initialized = True

#     def process_frame(self, frame):
#         self.fps_monitor.tick()
#         fps = self.fps_monitor.fps

#         if not self.line_initialized:
#             self._setup_doorway_line(frame)

#         # 1. Inference
#         detections = self.model.predict(frame)

#         # Keep only people
#         if detections.class_id is not None:
#             detections = detections[detections.class_id == self.person_class_id]

#         # 2. Tracking
#         detections = self.tracker.update_with_detections(detections)

#         # 3. Trigger line counter
#         # No manual in/out values are passed here.
#         if len(detections) > 0:
#             self.line_zone.trigger(detections=detections)

#         # 4. Annotation
#         labels = []
#         annotated_frame = frame.copy()

#         if (
#             detections.tracker_id is not None
#             and detections.confidence is not None
#             and len(detections.tracker_id) == len(detections.confidence)
#         ):
#             labels = [
#                 f"#{tracker_id} {conf:.2f}"
#                 for tracker_id, conf in zip(detections.tracker_id, detections.confidence)
#             ]
#         elif detections.tracker_id is not None:
#             labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]

#         annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
#         annotated_frame = self.box_annotator.annotate(annotated_frame, detections)
#         annotated_frame = self.label_annotator.annotate(annotated_frame, detections, labels)

#         # Draw line + in/out counts
#         annotated_frame = self.line_zone_annotator.annotate(
#             frame=annotated_frame,
#             line_counter=self.line_zone
#         )

#         # Extra text
#         occupancy = self.line_zone.in_count - self.line_zone.out_count

#         annotated_frame = sv.draw_text(
#             scene=annotated_frame,
#             text=f"FPS: {fps:.1f}",
#             text_anchor=sv.Point(40, 30),
#             background_color=sv.Color.RED,
#             text_color=sv.Color.WHITE
#         )

#         annotated_frame = sv.draw_text(
#             scene=annotated_frame,
#             text=(
#                 f"IN: {self.line_zone.in_count}  "
#                 f"OUT: {self.line_zone.out_count}  "
#                 f"INSIDE: {occupancy}"
#             ),
#             text_anchor=sv.Point(220, 30),
#             background_color=sv.Color.BLACK,
#             text_color=sv.Color.WHITE
#         )

#         return annotated_frame
   
   
   
   
   
   
   
   
   
    
#     def generate_frames(self):
#         # OPTIONAL: Setup Video Writer if you want to save
#         # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         # out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (640, 480))
#         frame_count = 0
#         while True:
#             success, frame = self.cap.read()
#             if not success:
#                 logger.warning("⚠️ Frame dropped or video ended. Rewinding...")
#                 self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#                 continue    
            
#             output_frame = self.process_frame(frame)
            
#             # OPTIONAL: Write to disk
#             # out.write(output_frame)

#             # Stream to browser
#             ret, buffer = cv2.imencode('.jpg', output_frame)
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
#             # TRICK: Log a "Heartbeat" every 1000 frames so you know it's alive
#             frame_count += 1
#             if frame_count % 1000 == 0:
#                 frame_count = 0  #set zero after 1000 frames
#                 logger.info(f"💓 System Alive. Processed {frame_count} frames.")















# import cv2
# import supervision as sv
# import time
# from src.vision.factory import get_detector
# from src.core.config import settings
# from loguru import logger  # <--- Import


# class VisionPipeline:
#     def __init__(self):
#         logger.info("⚙️ Initializing Vision Pipeline components...")
#         self.model = get_detector()

#         # --- THE FIX IS HERE ---
#         # Instead of your broken SORTTracker, use this:

#         self.tracker = sv.ByteTrack() 
#         self.person_class_id = 0  # YOLOv8 COCO person class
#         # -----------------------

#         # Setup UI
#         self.fps_monitor = sv.FPSMonitor()
#         color = sv.ColorPalette.DEFAULT 
#         self.box_annotator = sv.BoxAnnotator(color=color)
#         self.trace_annotator = sv.TraceAnnotator(color=color, trace_length=30) # Reduced length for speed
#         self.label_annotator = sv.LabelAnnotator(color=color, text_color=sv.Color.BLACK)

#         # Track how long each person stays visible for dwell/queue analytics.
#         self.track_presence = {}
#         self.completed_presence = {}
#         self.track_timeout_seconds = 2.0
#         self.stall_threshold_seconds = 15.0
#         self.queue_threshold_seconds = 30.0
        
        
#         # Auto-config state
#         self.line_zone = None
#         self.line_zone_annotator = None
#         self.line_initialized = False

#         # Change these depending on your entrance geometry
#         self.door_orientation = "horizontal"   # "horizontal" or "vertical"
#         self.door_position_ratio = 0.75        # 55% down the frame if horizontal
#         self.door_margin_ratio = 0.01          # keep line inside image edges
#         self.trigger_anchor = sv.Position.BOTTOM_CENTER
        
#         logger.info(f"🔌 Connecting to Video Source: {settings.RTSP_URL}")
        
#         # Camera
#         self.cap = cv2.VideoCapture(settings.RTSP_URL)
#         if not self.cap.isOpened():
#             logger.error(f"❌ COULD NOT OPEN VIDEO SOURCE: {settings.RTSP_URL}")
#             # Optional: Retry logic could be logged here
#         else:
#             logger.info("✅ Video Source Connected.")

#     # def process_frame(self, frame):
#     #     self.fps_monitor.tick()
#     #     fps = self.fps_monitor.fps
        
#     #     # 1. Inference
#     #     detections = self.model.predict(frame)

#     #     # 2. Tracking (The syntax changes slightly for ByteTrack)
#     #     detections = self.tracker.update_with_detections(detections)
        
#     #     # 3. Annotation
#     #     labels = []
#     #     annotated_frame = frame.copy()
#     #     if detections.tracker_id is not None and detections.confidence is not None \
#     #        and len(detections.tracker_id) == len(detections.confidence): # Check lengths match
#     #         labels = [
#     #             f"#{tracker_id} {conf:.2f}" 
#     #             for tracker_id, conf in zip(detections.tracker_id, detections.confidence)
#     #         ]

#     #     annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
#     #     annotated_frame = self.box_annotator.annotate(annotated_frame, detections)
#     #     annotated_frame = self.label_annotator.annotate(annotated_frame, detections, labels)
        
#     #     # FPS
#     #     annotated_frame = sv.draw_text(
#     #         scene=annotated_frame,
#     #         text=f"FPS: {fps:.1f}",
#     #         text_anchor=sv.Point(40, 30),
#     #         background_color=sv.Color.RED,
#     #         text_color=sv.Color.WHITE
#     #     )

#     #     return annotated_frame

#     def _setup_doorway_line(self, frame):
#         height, width = frame.shape[:2]

#         margin_x = int(width * self.door_margin_ratio)
#         margin_y = int(height * self.door_margin_ratio)

#         if self.door_orientation == "horizontal":
#             y = int(height * self.door_position_ratio)

#             start = sv.Point(margin_x, y)
#             end = sv.Point(width - margin_x, y)

#         else:
#             x = int(width * self.door_position_ratio)

#             start = sv.Point(x, margin_y)
#             end = sv.Point(x, height - margin_y)

#         self.line_zone = sv.LineZone(
#             start=start,
#             end=end,
#             triggering_anchors=[self.trigger_anchor]
#         )

#         self.line_zone_annotator = sv.LineZoneAnnotator(
#             thickness=2,
#             text_thickness=1,
#             text_scale=0.7
#         )

#         self.line_initialized = True

#     @staticmethod
#     def _format_duration(duration_seconds):
#         total_seconds = max(0, int(duration_seconds))
#         minutes, seconds = divmod(total_seconds, 60)
#         hours, minutes = divmod(minutes, 60)
#         if hours:
#             return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
#         return f"{minutes:02d}:{seconds:02d}"

#     def _update_presence_timers(self, detections):
#         now = time.monotonic()
#         active_track_ids = set()
#         track_durations = {}

#         tracker_ids = detections.tracker_id if detections.tracker_id is not None else []

#         for tracker_id in tracker_ids:
#             if tracker_id is None:
#                 continue

#             normalized_id = int(tracker_id)
#             active_track_ids.add(normalized_id)

#             if normalized_id not in self.track_presence:
#                 self.track_presence[normalized_id] = {
#                     "first_seen": now,
#                     "last_seen": now,
#                 }
#             else:
#                 self.track_presence[normalized_id]["last_seen"] = now

#             presence = self.track_presence[normalized_id]
#             track_durations[normalized_id] = now - presence["first_seen"]

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

#         return track_durations, {
#             "active_count": len(active_durations),
#             "stall_count": stall_count,
#             "queue_count": queue_count,
#             "longest_active": longest_active,
#             "average_active": average_active,
#             "average_completed": average_completed,
#         }

#     def process_frame(self, frame):
#         self.fps_monitor.tick()
#         fps = self.fps_monitor.fps

#         if not self.line_initialized:
#             self._setup_doorway_line(frame)

#         # 1. Inference
#         detections = self.model.predict(frame)

#         # Keep only people
#         if detections.class_id is not None:
#             detections = detections[detections.class_id == self.person_class_id]

#         # 2. Tracking
#         detections = self.tracker.update_with_detections(detections)
#         track_durations, dwell_metrics = self._update_presence_timers(detections)

#         # 3. Trigger line counter
#         # No manual in/out values are passed here.
#         if len(detections) > 0:
#             self.line_zone.trigger(detections=detections)

#         # 4. Annotation
#         labels = []
#         annotated_frame = frame.copy()

#         if (
#             detections.tracker_id is not None
#             and detections.confidence is not None
#             and len(detections.tracker_id) == len(detections.confidence)
#         ):
#             labels = [
#                 (
#                     f"#{int(tracker_id)} "
#                     f"{self._format_duration(track_durations.get(int(tracker_id), 0.0))} "
#                     f"{conf:.2f}"
#                 )
#                 for tracker_id, conf in zip(detections.tracker_id, detections.confidence)
#             ]
#         elif detections.tracker_id is not None:
#             labels = [
#                 (
#                     f"#{int(tracker_id)} "
#                     f"{self._format_duration(track_durations.get(int(tracker_id), 0.0))}"
#                 )
#                 for tracker_id in detections.tracker_id
#             ]

#         annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
#         annotated_frame = self.box_annotator.annotate(annotated_frame, detections)
#         annotated_frame = self.label_annotator.annotate(annotated_frame, detections, labels)

#         # Draw line + in/out counts
#         annotated_frame = self.line_zone_annotator.annotate(
#             frame=annotated_frame,
#             line_counter=self.line_zone
#         )

#         # Extra text
#         occupancy = self.line_zone.in_count - self.line_zone.out_count

#         annotated_frame = sv.draw_text(
#             scene=annotated_frame,
#             text=f"FPS: {fps:.1f}",
#             text_anchor=sv.Point(40, 30),
#             background_color=sv.Color.RED,
#             text_color=sv.Color.WHITE
#         )

#         annotated_frame = sv.draw_text(
#             scene=annotated_frame,
#             text=(
#                 f"IN: {self.line_zone.in_count}  "
#                 f"OUT: {self.line_zone.out_count}  "
#                 f"INSIDE: {occupancy}"
#             ),
#             text_anchor=sv.Point(220, 30),
#             background_color=sv.Color.BLACK,
#             text_color=sv.Color.WHITE
#         )

#         longest_active = self._format_duration(dwell_metrics["longest_active"])
#         average_active = self._format_duration(dwell_metrics["average_active"])
#         average_completed = self._format_duration(dwell_metrics["average_completed"])

#         annotated_frame = sv.draw_text(
#             scene=annotated_frame,
#             text=(
#                 f"VISIBLE: {dwell_metrics['active_count']}  "
#                 f"STALL>= {int(self.stall_threshold_seconds)}s: {dwell_metrics['stall_count']}  "
#                 f"QUEUE>= {int(self.queue_threshold_seconds)}s: {dwell_metrics['queue_count']}"
#             ),
#             text_anchor=sv.Point(200, 65),
#             background_color=sv.Color.BLUE,
#             text_color=sv.Color.WHITE
#         )

#         annotated_frame = sv.draw_text(
#             scene=annotated_frame,
#             text=(
#                 f"LONGEST: {longest_active}  "
#                 f"AVG LIVE: {average_active}  "
#                 f"AVG COMPLETED: {average_completed}"
#             ),
#             text_anchor=sv.Point(200, 100),
#             background_color=sv.Color.BLACK,
#             text_color=sv.Color.WHITE
#         )

#         return annotated_frame
   
   
   
   
    
#     def generate_frames(self):
#         # OPTIONAL: Setup Video Writer if you want to save
#         # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         # out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (640, 480))
#         frame_count = 0
#         while True:
#             success, frame = self.cap.read()
#             if not success:
#                 logger.warning("⚠️ Frame dropped or video ended. Rewinding...")
#                 self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#                 continue    
            
#             output_frame = self.process_frame(frame)
            
#             # OPTIONAL: Write to disk
#             # out.write(output_frame)

#             # Stream to browser
#             ret, buffer = cv2.imencode('.jpg', output_frame)
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
#             # TRICK: Log a "Heartbeat" every 1000 frames so you know it's alive
#             frame_count += 1
#             if frame_count % 1000 == 0:
#                 frame_count = 0  #set zero after 1000 frames
#                 logger.info(f"💓 System Alive. Processed {frame_count} frames.")















































































































import cv2
import supervision as sv
import time
from src.vision.factory import get_detector
from src.core.config import settings
from loguru import logger

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class VisionPipeline:
    def __init__(self, RTSP_URL):
        logger.info("⚙️ Initializing Vision Pipeline components...")
        self.model = get_detector()
        self.RTSP_URL = RTSP_URL
        self.frame_counter = 0
        self.tracked_paths = {}
        self.height = 8224
        self.width  = 8224

        self.tracker = sv.ByteTrack() 
        self.person_class_id = 0 
        # -----------------------

        # Setup UI
        self.fps_monitor = sv.FPSMonitor()
        color = sv.ColorPalette.DEFAULT 
        self.box_annotator = sv.BoxAnnotator(color=color)
        self.trace_annotator = sv.TraceAnnotator(color=color, trace_length=30) # Reduced length for speed
        self.label_annotator = sv.LabelAnnotator(color=color, text_color=sv.Color.BLACK)

        # Track how long each person stays visible for dwell/queue analytics.
        self.track_presence = {}
        self.completed_presence = {}
        self.track_timeout_seconds = 2.0
        self.stall_threshold_seconds = 15.0
        self.queue_threshold_seconds = 30.0
        
        
        # Auto-config state
        self.line_zone = None
        self.line_zone_annotator = None
        self.line_initialized = False

        # Change these depending on your entrance geometry
        self.door_orientation = "horizontal"   # "horizontal" or "vertical"
        self.door_position_ratio = 0.75        # 55% down the frame if horizontal
        self.door_margin_ratio = 0.01          # keep line inside image edges
        self.trigger_anchor = sv.Position.BOTTOM_CENTER
        
        logger.info(f"🔌 Connecting to Video Source: {self.RTSP_URL}")
        
        # Camera
        self.cap = cv2.VideoCapture(self.RTSP_URL)
        if not self.cap.isOpened():
            logger.error(f"❌ COULD NOT OPEN VIDEO SOURCE: {self.RTSP_URL}")
            # Optional: Retry logic could be logged here
        else:
            logger.info("✅ Video Source Connected.")

    def _setup_doorway_line(self, frame):
        # height, width = frame.shape[:2]

        margin_x = int(self.width * self.door_margin_ratio)
        margin_y = int(self.height * self.door_margin_ratio)

        if self.door_orientation == "horizontal":
            y = int(self.height * self.door_position_ratio)

            start = sv.Point(margin_x, y)
            end = sv.Point(self.width - margin_x, y)

        else:
            x = int(self.width * self.door_position_ratio)

            start = sv.Point(x, margin_y)
            end = sv.Point(x, self.height - margin_y)

        self.line_zone = sv.LineZone(
            start=start,
            end=end,
            triggering_anchors=[self.trigger_anchor]
        )

        self.line_zone_annotator = sv.LineZoneAnnotator(
            thickness=2,
            text_thickness=1,
            text_scale=0.7
        )

        self.line_initialized = True

    @staticmethod
    def _format_duration(duration_seconds):
        total_seconds = max(0, int(duration_seconds))
        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def _update_presence_timers(self, detections):
        now = time.monotonic()
        active_track_ids = set()
        track_durations = {}

        tracker_ids = detections.tracker_id if detections.tracker_id is not None else []

        for tracker_id in tracker_ids:
            if tracker_id is None:
                continue

            normalized_id = int(tracker_id)
            active_track_ids.add(normalized_id)

            if normalized_id not in self.track_presence:
                self.track_presence[normalized_id] = {
                    "first_seen": now,
                    "last_seen": now,
                }
            else:
                self.track_presence[normalized_id]["last_seen"] = now

            presence = self.track_presence[normalized_id]
            track_durations[normalized_id] = now - presence["first_seen"]

        stale_track_ids = []
        for tracker_id, presence in self.track_presence.items():
            if tracker_id in active_track_ids:
                continue

            if now - presence["last_seen"] >= self.track_timeout_seconds:
                self.completed_presence[tracker_id] = presence["last_seen"] - presence["first_seen"]
                stale_track_ids.append(tracker_id)

        for tracker_id in stale_track_ids:
            self.track_presence.pop(tracker_id, None)

        active_durations = list(track_durations.values())
        completed_durations = list(self.completed_presence.values())

        longest_active = max(active_durations, default=0.0)
        average_active = (
            sum(active_durations) / len(active_durations)
            if active_durations else 0.0
        )
        average_completed = (
            sum(completed_durations) / len(completed_durations)
            if completed_durations else 0.0
        )
        stall_count = sum(
            1 for duration in active_durations if duration >= self.stall_threshold_seconds
        )
        queue_count = sum(
            1 for duration in active_durations if duration >= self.queue_threshold_seconds
        )

        return track_durations, {
            "active_count": len(active_durations),
            "stall_count": stall_count,
            "queue_count": queue_count,
            "longest_active": longest_active,
            "average_active": average_active,
            "average_completed": average_completed,
        }

    # def process_frame(self, frame):
    #     self.fps_monitor.tick()
    #     fps = self.fps_monitor.fps

    #     if not self.line_initialized:
    #         self._setup_doorway_line(frame)

    #     # 1. Inference
    #     detections = self.model.predict(frame)

    #     # Keep only people
    #     if detections.class_id is not None:
    #         detections = detections[detections.class_id == self.person_class_id]

    #     # 2. Tracking
    #     detections = self.tracker.update_with_detections(detections)
    #     track_durations, dwell_metrics = self._update_presence_timers(detections)

    #     # 3. Trigger line counter
    #     # No manual in/out values are passed here.
    #     if len(detections) > 0:
    #         self.line_zone.trigger(detections=detections)

    #     # 4. Annotation
    #     labels = []
    #     annotated_frame = frame.copy()

    #     if (
    #         detections.tracker_id is not None
    #         and detections.confidence is not None
    #         and len(detections.tracker_id) == len(detections.confidence)
    #     ):
    #         labels = [
    #             (
    #                 f"#{int(tracker_id)} "
    #                 f"{self._format_duration(track_durations.get(int(tracker_id), 0.0))} "
    #                 f"{conf:.2f}"
    #             )
    #             for tracker_id, conf in zip(detections.tracker_id, detections.confidence)
    #         ]
    #     elif detections.tracker_id is not None:
    #         labels = [
    #             (
    #                 f"#{int(tracker_id)} "
    #                 f"{self._format_duration(track_durations.get(int(tracker_id), 0.0))}"
    #             )
    #             for tracker_id in detections.tracker_id
    #         ]

    #     annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
    #     annotated_frame = self.box_annotator.annotate(annotated_frame, detections)
    #     annotated_frame = self.label_annotator.annotate(annotated_frame, detections, labels)

    #     # Draw line + in/out counts
    #     annotated_frame = self.line_zone_annotator.annotate(
    #         frame=annotated_frame,
    #         line_counter=self.line_zone
    #     )

    #     # Extra text
    #     occupancy = self.line_zone.in_count - self.line_zone.out_count

    #     annotated_frame = sv.draw_text(
    #         scene=annotated_frame,
    #         text=f"FPS: {fps:.1f}",
    #         text_anchor=sv.Point(40, 30),
    #         background_color=sv.Color.RED,
    #         text_color=sv.Color.WHITE
    #     )

    #     annotated_frame = sv.draw_text(
    #         scene=annotated_frame,
    #         text=(
    #             f"IN: {self.line_zone.in_count}  "
    #             f"OUT: {self.line_zone.out_count}  "
    #             f"INSIDE: {occupancy}"
    #         ),
    #         text_anchor=sv.Point(220, 30),
    #         background_color=sv.Color.BLACK,
    #         text_color=sv.Color.WHITE
    #     )

    #     longest_active = self._format_duration(dwell_metrics["longest_active"])
    #     average_active = self._format_duration(dwell_metrics["average_active"])
    #     average_completed = self._format_duration(dwell_metrics["average_completed"])

    #     annotated_frame = sv.draw_text(
    #         scene=annotated_frame,
    #         text=(
    #             f"VISIBLE: {dwell_metrics['active_count']}  "
    #             f"STALL>= {int(self.stall_threshold_seconds)}s: {dwell_metrics['stall_count']}  "
    #             f"QUEUE>= {int(self.queue_threshold_seconds)}s: {dwell_metrics['queue_count']}"
    #         ),
    #         text_anchor=sv.Point(200, 65),
    #         background_color=sv.Color.BLUE,
    #         text_color=sv.Color.WHITE
    #     )

    #     annotated_frame = sv.draw_text(
    #         scene=annotated_frame,
    #         text=(
    #             f"LONGEST: {longest_active}  "
    #             f"AVG LIVE: {average_active}  "
    #             f"AVG COMPLETED: {average_completed}"
    #         ),
    #         text_anchor=sv.Point(200, 100),
    #         background_color=sv.Color.BLACK,
    #         text_color=sv.Color.WHITE
    #     )

    #     return annotated_frame
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
    def export_path_data(self, file_path='data/customers_paths.csv'):
        """
        Export all tracked paths to a CSV database.
        """
        all_rows = []
        for track_id, path in self.tracked_paths.items():
            df_track = pd.DataFrame(path)
            df_track['track_id'] = track_id
            all_rows.append(df_track)
        
        if not all_rows:
            print("No path data to export.")
            return

        combined_df = pd.concat(all_rows, ignore_index=True)
        combined_df.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
        return combined_df

    def generate_heatmap(self, output_file='data/heatmap.png'):
        """
        Create a density heatmap based on tracked positions.
        """
        # Initialize empty heatmap grid (same shape as input frame)
        # h, w = self.frame_shape
        heatmap_grid = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Accumulate density
        for track_id, path in self.tracked_paths.items():
            for point in path:
                # Round to nearest pixel (or use direct mapping if coordinates match)
                x = int(point['x'])
                y = int(point['y'])
                
                if 0 <= x < self.width and 0 <= y < self.height:
                    # Increment pixel value
                    heatmap_grid[y, x] += 1

        # Normalize (optional)
        max_val = np.max(heatmap_grid)
        if max_val > 0:
            heatmap_grid = (heatmap_grid / max_val) * 255

        # Create visualization
        heat_map = cv2.applyColorMap(np.uint8(heatmap_grid), cv2.COLORMAP_JET)
        
        # Display result
        cv2.imwrite(output_file, heat_map)
        plt.figure(figsize=(15, 10))
        plt.imshow(heat_map, cmap='hot')
        plt.title(f"Customer Movement Heatmap (Max visits: {int(max_val)})")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(output_file.replace('.png', '.jpg')) # Save figure separately
        plt.close()
        
        print(f"Heatmap generated and saved to {output_file}")
        return heat_map



    def generate_lines(self, output_file='data/customer_paths.png'):
        """
        Draw customer paths with different colors on a single image.
        """
        # Initialize empty figure
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Plot each path with a unique color
        for track_id, path in self.tracked_paths.items():
            x_coords = [point['x'] for point in path]
            y_coords = [point['y'] for point in path]
            ax.plot(x_coords, y_coords, label=f'Track {track_id}', linewidth=2)
        
        # Set the aspect ratio to 'equal' to preserve the scale
        ax.set_aspect('equal')
        
        # Add labels and title
        plt.title("Customer Movement Paths")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        
        # Save the figure
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
        print(f"Paths saved to {output_file}")
        
        return fig
   
    def process_frame(self, frame):
        # self.frame_counter += 1
        current_frame_idx = self.frame_counter
        
        self.fps_monitor.tick()
        fps = self.fps_monitor.fps
        # height, width = frame.shape[:2]
        if not self.line_initialized:
            self._setup_doorway_line(frame)

        # 1. Inference
        detections = self.model.predict(frame)

        # Keep only people
        if detections.class_id is not None:
            detections = detections[detections.class_id == self.person_class_id]

        # 2. Tracking
        detections = self.tracker.update_with_detections(detections)
        track_durations, dwell_metrics = self._update_presence_timers(detections)
        
        # 3. Path Tracking (STORAGE LOGIC)
        for track_id, det_box in zip(detections.tracker_id, detections.xyxy):

            if not track_id:
                continue

            # Extract box coordinates
            x1, y1, x2, y2 = det_box
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # Initialize list for new track
            if track_id not in self.tracked_paths:
                self.tracked_paths[track_id] = []
            
            # Store point
            self.tracked_paths[track_id].append({
                'frame': current_frame_idx,
                'x': cx,
                'y': cy
            })

        # 3. Trigger line counter
        # No manual in/out values are passed here.
        if len(detections) > 0:
            self.line_zone.trigger(detections=detections)

        # 4. Annotation
        labels = []
        annotated_frame = frame.copy()

        if (
            detections.tracker_id is not None
            and detections.confidence is not None
            and len(detections.tracker_id) == len(detections.confidence)
        ):
            labels = [
                (
                    f"#{int(tracker_id)} "
                    f"{self._format_duration(track_durations.get(int(tracker_id), 0.0))} "
                    f"{conf:.2f}"
                )
                for tracker_id, conf in zip(detections.tracker_id, detections.confidence)
            ]
        elif detections.tracker_id is not None:
            labels = [
                (
                    f"#{int(tracker_id)} "
                    f"{self._format_duration(track_durations.get(int(tracker_id), 0.0))}"
                )
                for tracker_id in detections.tracker_id
            ]

        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.box_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.label_annotator.annotate(annotated_frame, detections, labels)

        # Draw line + in/out counts
        annotated_frame = self.line_zone_annotator.annotate(
            frame=annotated_frame,
            line_counter=self.line_zone
        )

        # Extra text
        occupancy = self.line_zone.in_count - self.line_zone.out_count

        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=f"FPS: {fps:.1f}",
            text_anchor=sv.Point(40, 30),
            background_color=sv.Color.RED,
            text_color=sv.Color.WHITE
        )

        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=(
                f"IN: {self.line_zone.in_count}  "
                f"OUT: {self.line_zone.out_count}  "
                f"INSIDE: {occupancy}"
            ),
            text_anchor=sv.Point(220, 30),
            background_color=sv.Color.BLACK,
            text_color=sv.Color.WHITE
        )

        longest_active = self._format_duration(dwell_metrics["longest_active"])
        average_active = self._format_duration(dwell_metrics["average_active"])
        average_completed = self._format_duration(dwell_metrics["average_completed"])

        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=(
                f"VISIBLE: {dwell_metrics['active_count']}  "
                f"STALL>= {int(self.stall_threshold_seconds)}s: {dwell_metrics['stall_count']}  "
                f"QUEUE>= {int(self.queue_threshold_seconds)}s: {dwell_metrics['queue_count']}"
            ),
            text_anchor=sv.Point(200, 65),
            background_color=sv.Color.BLUE,
            text_color=sv.Color.WHITE
        )

        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=(
                f"LONGEST: {longest_active}  "
                f"AVG LIVE: {average_active}  "
                f"AVG COMPLETED: {average_completed}"
            ),
            text_anchor=sv.Point(200, 100),
            background_color=sv.Color.BLACK,
            text_color=sv.Color.WHITE
        )

        return annotated_frame
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
    
    def generate_frames(self):
        reading_video = True # Temporary for testing videos
        
        # OPTIONAL: Setup Video Writer if you want to save
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('data/output.mp4', fourcc, 30.0, (640, 480))
        
        # while True:
        while reading_video:
            success, frame = self.cap.read()
            if not success:
                reading_video = False
                self.export_path_data()
                self.generate_heatmap()
                self.generate_lines()
                logger.warning("⚠️ Frame dropped or video ended. Rewinding...")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                # continue
                break    
            
            self.height, self.width = frame.shape[:2]
            output_frame = self.process_frame(frame)
            
            # OPTIONAL: Write to disk
            out.write(output_frame)

            # Stream to browser
            ret, buffer = cv2.imencode('.jpg', output_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            # TRICK: Log a "Heartbeat" every 1000 frames so you know it's alive
            self.frame_counter += 1
            if self.frame_counter % 1000 == 0:
                self.export_path_data()
                self.generate_heatmap()
                self.generate_lines()
                logger.info(f"💓 System Alive. Processed {self.frame_counter} frames.")
                self.frame_counter = 0  #set zero after 1000 frames



















































import cv2
import onnx
import hashlib
import supervision as sv
import onnxruntime as ort
from src.vision.factory import get_detector
from src.core.config import settings
from loguru import logger
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import psycopg2
from psycopg2 import sql, extras
import time
import numpy as np
# import sqlite3
from pathlib import Path
from onnxruntime import InferenceSession


class VisionPipeline:
    def __init__(self, RTSP_URL):
        logger.info("⚙️ Initializing Vision Pipeline components...")
        self.model = get_detector()
        onnx_path: str = 'models/osnet_x1_0_imagenet.onnx'
        print(f"🔌 Loading ONNX model: {onnx_path}...")
        
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        print(f"🔍 Model input size: {self.session.get_inputs()[0].shape}")
        # model_input_size: tuple = (256, 128),
        # expected_input_shape = [1, 3] + list(model_input_size)  # batch, channels, height, width
        
        # Configuration
        DB_USER = "your_username"
        DB_PASSWORD = "your_password"
        DB_NAME = "your_database_name"
        DB_HOST = "db"  # Connect to Docker service name 'db'
        connection_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:5432/{DB_NAME}"
        
        self.conn = psycopg2.connect(connection_string)
        self.cur = self.conn.cursor()
        # --- CONFIGURATION ---
        DB_PATH = 'instance/personality.db'
        FAISS_INDEX_PATH = 'faissself._index.idx'
        FAISS_DIMENSION = 512  # Match your model's output size
        self.TIME_WINDOW_SECONDS = 300  # 5 minutes

        # --- INITIALIZATION ---
        # self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        # self.cursor = self.conn.cursor()
        # self.cursor.execute('''
        #     CREATE TABLE IF NOT EXISTS people (
        #         id INTEGER PRIMARY KEY,
        #         embedding BLOB,
        #         updated_at REAL
        #     )
        # ''')
        # self.conn.commit()
        # self.conn.close()

        # self.index = faiss.IndexFlatL2(FAISS_DIMENSION)
        # self.index.write_index(FAISS_INDEX_PATH) # Initial load

        # --- RAM CACHE ---
        # Holds (person_id, embedding_vector)
        # This is the "Active Tracks" in RAM
        self.ram_cache = {} 
        
        # Configuration
        self.CONFIG = {
            'db_path': 'person_history.db',
            'embedding_size': 128, # Assuming YOLOv8-SE or similar ReID model
            'distance_threshold': 0.5, # If < 0.5, same person
            'camera': 'Cam1',
        }

        self.RTSP_URL = RTSP_URL
        self.frame_counter = 0
        self.tracked_paths = {}
        self.height = 8224
        self.width  = 8224

        self.tracker = sv.ByteTrack() 
        self.person_class_id = 0 
        # -----------------------

        # Setup UI
        self.fps_monitor = sv.FPSMonitor()
        color = sv.ColorPalette.DEFAULT 
        self.box_annotator = sv.BoxAnnotator(color=color)
        self.trace_annotator = sv.TraceAnnotator(color=color, trace_length=30) # Reduced length for speed
        self.label_annotator = sv.LabelAnnotator(color=color, text_color=sv.Color.BLACK)

        # Track how long each person stays visible for dwell/queue analytics.
        self.track_presence = {}
        self.completed_presence = {}
        self.track_timeout_seconds = 2.0
        self.stall_threshold_seconds = 15.0
        self.queue_threshold_seconds = 30.0
        
        
        # Auto-config state
        self.line_zone = None
        self.line_zone_annotator = None
        self.line_initialized = False

        # Change these depending on your entrance geometry
        self.door_orientation = "horizontal"   # "horizontal" or "vertical"
        self.door_position_ratio = 0.75        # 55% down the frame if horizontal
        self.door_margin_ratio = 0.01          # keep line inside image edges
        self.trigger_anchor = sv.Position.BOTTOM_CENTER
        
        logger.info(f"🔌 Connecting to Video Source: {self.RTSP_URL}")
        
        # Camera
        self.cap = cv2.VideoCapture(self.RTSP_URL)
        if not self.cap.isOpened():
            logger.error(f"❌ COULD NOT OPEN VIDEO SOURCE: {self.RTSP_URL}")
            # Optional: Retry logic could be logged here
        else:
            logger.info("✅ Video Source Connected.")

    def normalize_vector(self, vec):
        """FAISS needs normalized vectors for Cosine Similarity."""
        return vec / np.linalg.norm(vec)

    def update_database(self, embedding, timestamp, confidence):
        # """Saves a new detection to the history DB."""
        # self.CONFIG = {
        #     'db_path': 'person_history.db',
        #     'embedding_size': 512, # Assuming YOLOv8-SE or similar ReID model
        #     'distance_threshold': 0.5, # If < 0.5, same person
        # }
        # conn = sqlite3.connect(self.CONFIG['db_path'])
        # cur = conn.cursor()
        
        # # Simple schema: id, embedding (raw bytes), path, cam, ts, conf
        # # Note: SQLite doesn't support vector math natively without extension.
        # # Using 'blob' column to store numpy array as hex/bytes for simplicity in pure python
        # # For production, use PostgreSQL with pgvector or SQLite with sqlite-vec
        
        # # Convert vector to bytes to save in DB
        # emb_bytes = embedding.tobytes() 
        
        # cur.execute('''
        #     INSERT INTO people_history (embedding_blob, timestamp, confidence, is_missing) 
        #     VALUES (?, ?, ?, ?, ?, ?)
        # ''', (emb_bytes, timestamp, confidence, 0))
        # # ''', (emb_bytes, image_path, cam_id, timestamp, conf, 0))
        
        
        # conn.commit()
        # conn.close()
        
        
        try:
            # 1. Enable extension (Good practice, though usually pre-loaded in image)
            self.cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # 2. Create Table
            # Using VECTOR(384) as standard embedding dimension
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS embeddings (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                embedding vector(384),
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
            self.cur.execute(create_table_sql)
            
            # 3. Create Index (HNSW is required for searching)
            # The index creates a pgvector index on the vector column
            create_index_sql = """
            CREATE INDEX IF NOT EXISTS embeddings_idx ON embeddings (embedding)
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 100);
            """
            self.cur.execute(create_index_sql)
            
            # 4. Insert Dummy Data (Placeholder vector for now)
            # In production, you would replace this with your actual embedding model output
            dummy_vector = [0.0] * 384 # Placeholder vector
            insert_sql = """
            INSERT INTO embeddings (content, embedding) 
            VALUES (%s, %s)
            ON CONFLICT DO NOTHING;
            """
            self.cur.execute(insert_sql, ( "Hello world initial data", tuple(dummy_vector) ))
            
            self.conn.commit()
            print("Database schema and data initialized successfully.")
            
        except Exception as e:
            print(f"Error initializing DB: {e}")
        finally:
            self.cur.close()
            self.conn.close()   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

    def normalize_vector(self, vec):
        """FAISS needs normalized vectors for Cosine Similarity."""
        return vec / np.linalg.norm(vec)

    def get_embeddings_from_detections(
        self,
        frame: np.ndarray,
        det_box: tuple,
        # onnx_path: str = 'osnet_x1_0.onnx',
        model_input_size: tuple = (256, 128),
        torso_ratio: float = 2/3
    ):
        """
        Extract embeddings from detected objects in a frame.

        Args:
            frame: OpenCV frame (np.array) with detections overlaid or not.
            detections: List of dicts in format {'bbox': [x1, y1, x2, y2], 'class': 'person'} etc.
            onnx_path: Path to your ONNX model file.
            model_input_size: (height, width) of the ONNX model input.
            torso_ratio: Fraction of box height to crop from top (e.g., 2/3).

        Returns:
            List of embeddings (normalized float vectors) with corresponding labels.
        """
        print(f"👁️ Processing detections...")
        
        try:
            # 1. Extract bbox coordinates
            x1, y1, x2, y2 = det_box
            
            # 2. Basic sanity checks
            w = x2 - x1
            h = y2 - y1

            # 3. Crop the torso (Top 2/3)
            crop_y1 = int(y1)
            crop_y2 = int(y1 + h * torso_ratio)
            crop_h = crop_y2 - crop_y1
            crop_w = w  # Keep full width of torso region
            
            # Handle edge cases where crop_y2 might exceed frame bounds
            if crop_y2 > y2:
                crop_y2 = y2
            
            # 4. Crop and Prepare for resizing
            crop_box = frame[int(y1):int(crop_y2), int(x1):int(x2)]
            cv2.imwrite(f'data/{x1}.png', crop_box)
            print(crop_box.shape)
            # 5. Resize to model input size (256x128) - using INTER_AREA for better quality
            resized_crop = cv2.resize(crop_box, (model_input_size[1], model_input_size[0]), interpolation=cv2.INTER_AREA)
            print(resized_crop.shape)
            cv2.imwrite(f'data/{x1}_2.png', resized_crop)
            # 6. Normalize (0-255 -> -1 to 1) OR use ImageNet stats if your model is ImageNet trained
            # Assuming standard ONNX conversion which usually keeps [0, 1] or [-1, 1]
            normalized_crop = resized_crop.astype(np.float32) / 255.0
            print(normalized_crop.shape, model_input_size, crop_h, crop_w)
            # 7. Pad to model height if necessary (Optional step for aspect ratio preservation)
            # If we crop the height, we might end up with a smaller height. 
            # If model requires 256x128 and we have a smaller crop, we need padding.
            # Let's pad with black or replicate borders:
            if normalized_crop.shape[0] < model_input_size[0]:
                pad_h = model_input_size[0] - crop_h
                pad_top = pad_h // 2
                pad_bot = pad_h - pad_top
                pad_top_crop = np.pad(normalized_crop, ((pad_top, pad_bot), (0, 0), (0, 0)), mode='constant', constant_values=0.0)
                padded_crop = pad_top_crop
            else:
                padded_crop = normalized_crop
                
            # 8. Add Batch Dimension [N, C, H, W]
            padded_crop = np.expand_dims(padded_crop, axis=0)
            print(33333333333333333333333, padded_crop.shape)
            input_tensor = np.transpose(padded_crop, (0, 3, 1, 2))  # H, W -> C, H, W
            print(44444444444444444,input_tensor.shape)
            # 9. Run Inference
            outputs = self.session.run(None, {'input': input_tensor})
            print(1111111111111111111, outputs[0].shape)
            # 10. Extract Embedding
            embedding = outputs[0].flatten()
            return embedding, crop_y1
        except Exception as e:
            print(f"❌ Error processing detection: {str(e)}")
    
    def insert_new_person(self, vector, bbox):
        vector_b = vector.tobytes()
        self.cursor.execute("INSERT OR REPLACE INTO people (embedding, updated_at) VALUES (?, ?)", 
                    (vector_b, time.time()))
        self.conn.commit()
        # Get last inserted id
        self.cursor.execute("SELECT last_insert_rowid()")
        new_id = self.cursor.fetchone()[0]
        return new_id

    def update_db_person(self, pid, vector, current_time):
        vector_b = vector.tobytes()
        self.cursor.execute("UPDATE people SET embedding=?, updated_at=? WHERE id=?", 
                    (vector_b, current_time, pid))
        self.conn.commit()
    # --- CLEANUP OLD RAM ---
    def cleanup_ram(self):
        current_time = time.time()
        for pid in list(self.ram_cache.keys()):
            self.ram_cache[pid]['seen_at'] # Access to check time
            if current_time - self.ram_cache[pid]['seen_at'] > self.TIME_WINDOW_SECONDS:
                del self.ram_cache[pid]
                # Also remove from FAISS (optional, but saves memory)
                # FAISS doesn't support efficient removal, so you rebuild or use a separateself. index.

    def _setup_doorway_line(self, frame):
        # height, width = frame.shape[:2]

        margin_x = int(self.width * self.door_margin_ratio)
        margin_y = int(self.height * self.door_margin_ratio)

        if self.door_orientation == "horizontal":
            y = int(self.height * self.door_position_ratio)

            start = sv.Point(margin_x, y)
            end = sv.Point(self.width - margin_x, y)

        else:
            x = int(self.width * self.door_position_ratio)

            start = sv.Point(x, margin_y)
            end = sv.Point(x, self.height - margin_y)

        self.line_zone = sv.LineZone(
            start=start,
            end=end,
            triggering_anchors=[self.trigger_anchor]
        )

        self.line_zone_annotator = sv.LineZoneAnnotator(
            thickness=2,
            text_thickness=1,
            text_scale=0.7
        )

        self.line_initialized = True

    @staticmethod
    def _format_duration(duration_seconds):
        total_seconds = max(0, int(duration_seconds))
        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def _update_presence_timers(self, now, active_track_ids, track_durations):
        stale_track_ids = []
        for tracker_id, presence in self.track_presence.items():
            if tracker_id in active_track_ids:
                continue

            if now - presence["last_seen"] >= self.track_timeout_seconds:
                self.completed_presence[tracker_id] = presence["last_seen"] - presence["first_seen"]
                stale_track_ids.append(tracker_id)

        for tracker_id in stale_track_ids:
            self.track_presence.pop(tracker_id, None)

        active_durations = list(track_durations.values())
        completed_durations = list(self.completed_presence.values())

        longest_active = max(active_durations, default=0.0)
        average_active = (
            sum(active_durations) / len(active_durations)
            if active_durations else 0.0
        )
        average_completed = (
            sum(completed_durations) / len(completed_durations)
            if completed_durations else 0.0
        )
        stall_count = sum(
            1 for duration in active_durations if duration >= self.stall_threshold_seconds
        )
        queue_count = sum(
            1 for duration in active_durations if duration >= self.queue_threshold_seconds
        )

        return {
            "active_count": len(active_durations),
            "stall_count": stall_count,
            "queue_count": queue_count,
            "longest_active": longest_active,
            "average_active": average_active,
            "average_completed": average_completed,
        }

    def export_path_data(self, file_path='data/customers_paths.csv'):
        """
        Export all tracked paths to a CSV database.
        """
        all_rows = []
        for track_id, path in self.tracked_paths.items():
            df_track = pd.DataFrame(path)
            df_track['track_id'] = track_id
            all_rows.append(df_track)
        
        if not all_rows:
            print("No path data to export.")
            return

        combined_df = pd.concat(all_rows, ignore_index=True)
        combined_df.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
        return combined_df

    def generate_heatmap(self, output_file='data/heatmap.png'):
        """
        Create a density heatmap based on tracked positions.
        """
        # Initialize empty heatmap grid (same shape as input frame)
        # h, w = self.frame_shape
        heatmap_grid = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Accumulate density
        for track_id, path in self.tracked_paths.items():
            for point in path:
                # Round to nearest pixel (or use direct mapping if coordinates match)
                x = int(point['x'])
                y = int(point['y'])
                
                if 0 <= x < self.width and 0 <= y < self.height:
                    # Increment pixel value
                    heatmap_grid[y, x] += 1

        # Normalize (optional)
        max_val = np.max(heatmap_grid)
        if max_val > 0:
            heatmap_grid = (heatmap_grid / max_val) * 255

        # Create visualization
        heat_map = cv2.applyColorMap(np.uint8(heatmap_grid), cv2.COLORMAP_JET)
        
        # Display result
        cv2.imwrite(output_file, heat_map)
        plt.figure(figsize=(15, 10))
        plt.imshow(heat_map, cmap='hot')
        plt.title(f"Customer Movement Heatmap (Max visits: {int(max_val)})")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(output_file.replace('.png', '.jpg')) # Save figure separately
        plt.close()
        
        print(f"Heatmap generated and saved to {output_file}")
        return heat_map

    def generate_lines(self, output_file='data/customer_paths.png'):
        """
        Draw customer paths with different colors on a single image.
        """
        # Initialize empty figure
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Plot each path with a unique color
        for track_id, path in self.tracked_paths.items():
            x_coords = [point['x'] for point in path]
            y_coords = [point['y'] for point in path]
            ax.plot(x_coords, y_coords, label=f'Track {track_id}', linewidth=2)
        
        # Set the aspect ratio to 'equal' to preserve the scale
        ax.set_aspect('equal')
        
        # Add labels and title
        plt.title("Customer Movement Paths")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        
        # Save the figure
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
        print(f"Paths saved to {output_file}")
        
        return fig
   
    def process_frame(self, frame):
        # self.frame_counter += 1
        detected_info = []
        embeddings = []
        current_frame_idx = self.frame_counter
        
        self.fps_monitor.tick()
        fps = self.fps_monitor.fps
        # height, width = frame.shape[:2]
        if not self.line_initialized:
            self._setup_doorway_line(frame)

        # 1. Inference
        detections = self.model.predict(frame)

        # Keep only people
        if detections.class_id is not None:
            detections = detections[detections.class_id == self.person_class_id]

        # 2. Tracking
        detections = self.tracker.update_with_detections(detections)
        now = time.monotonic()
        active_track_ids = set()
        track_durations = {}
        labels = []
        for obj_i, det in enumerate(detections):
            det_box, mask, confidence, class_id, tracker_id, data = det
            if tracker_id is None:
                continue

            # 3. Path Tracking (STORAGE LOGIC)
            # Extract box coordinates
            x1, y1, x2, y2 = det_box
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            
            # 2. Basic sanity checks
            w = x2 - x1
            h = y2 - y1
            if w < 20 or h < 20:
                print(f"⚠️  Detection {obj_i} is too small to process (w={w}, h={h}). Skipping.")
                # continue
            else:
                # 11. Store
                embedding, crop_y1 =  self.get_embeddings_from_detections(frame, det_box)
                # embeddings.append(embedding)
                embeddings.append(self.normalize_vector(embedding))
                detected_info.append({
                    "label": class_id,
                    "bbox": (x1, y1, x2, y2),
                    "crop_y": crop_y1
                })
                
            if len(embeddings) == 0:
                print("⚠️  No detections processed.")
            
            print(f"✅ Extracted {len(embeddings)} embeddings.")
                

            # Initialize list for new track
            if tracker_id not in self.tracked_paths:
                self.tracked_paths[tracker_id] = []
            
            # Store point
            self.tracked_paths[tracker_id].append({
                'frame': current_frame_idx,
                'x': cx,
                'y': cy
            })

            
            normalized_id = int(tracker_id)
            active_track_ids.add(normalized_id)

            if normalized_id not in self.track_presence:
                self.track_presence[normalized_id] = {
                    "first_seen": now,
                    "last_seen": now,
                }
            else:
                self.track_presence[normalized_id]["last_seen"] = now

            presence = self.track_presence[normalized_id]
            track_durations[normalized_id] = now - presence["first_seen"]
            # 4. Annotation

            # self.update_database(embedding, image_path, cam_id, timestamp, confidence)  
            self.update_database(embedding, track_durations[normalized_id], confidence)    
              
                
            labels.append(
                (
                    f"#{int(tracker_id)} "
                    f"{self._format_duration(track_durations.get(int(tracker_id), 0.0))} "
                    f"{confidence:.2f}"
                )
                )

           
        # 3. Trigger line counter
        # No manual in/out values are passed here.
        if len(detections) > 0:
            self.line_zone.trigger(detections=detections)


        dwell_metrics = self._update_presence_timers(now, active_track_ids, track_durations)
        
        annotated_frame = frame.copy()
        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.box_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.label_annotator.annotate(annotated_frame, detections, labels)

        # Draw line + in/out counts
        annotated_frame = self.line_zone_annotator.annotate(
            frame=annotated_frame,
            line_counter=self.line_zone
        )

        # Extra text
        occupancy = self.line_zone.in_count - self.line_zone.out_count

        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=f"FPS: {fps:.1f}",
            text_anchor=sv.Point(40, 30),
            background_color=sv.Color.RED,
            text_color=sv.Color.WHITE
        )

        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=(
                f"IN: {self.line_zone.in_count}  "
                f"OUT: {self.line_zone.out_count}  "
                f"INSIDE: {occupancy}"
            ),
            text_anchor=sv.Point(220, 30),
            background_color=sv.Color.BLACK,
            text_color=sv.Color.WHITE
        )

        longest_active = self._format_duration(dwell_metrics["longest_active"])
        average_active = self._format_duration(dwell_metrics["average_active"])
        average_completed = self._format_duration(dwell_metrics["average_completed"])

        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=(
                f"VISIBLE: {dwell_metrics['active_count']}  "
                f"STALL>= {int(self.stall_threshold_seconds)}s: {dwell_metrics['stall_count']}  "
                f"QUEUE>= {int(self.queue_threshold_seconds)}s: {dwell_metrics['queue_count']}"
            ),
            text_anchor=sv.Point(200, 65),
            background_color=sv.Color.BLUE,
            text_color=sv.Color.WHITE
        )

        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=(
                f"LONGEST: {longest_active}  "
                f"AVG LIVE: {average_active}  "
                f"AVG COMPLETED: {average_completed}"
            ),
            text_anchor=sv.Point(200, 100),
            background_color=sv.Color.BLACK,
            text_color=sv.Color.WHITE
        )

        return annotated_frame
   
    def generate_frames(self):
        reading_video = True # Temporary for testing videos
        
        # OPTIONAL: Setup Video Writer if you want to save
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('data/output.mp4', fourcc, 30.0, (640, 480))
        
        # while True:
        while reading_video:
            success, frame = self.cap.read()
            if not success:
                reading_video = False
                self.export_path_data()
                self.generate_heatmap()
                self.generate_lines()
                logger.warning("⚠️ Frame dropped or video ended. Rewinding...")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                # continue
                break    
            
            self.height, self.width = frame.shape[:2]
            output_frame = self.process_frame(frame)
            
            # OPTIONAL: Write to disk
            out.write(output_frame)

            # Stream to browser
            ret, buffer = cv2.imencode('.jpg', output_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            # TRICK: Log a "Heartbeat" every 1000 frames so you know it's alive
            self.frame_counter += 1
            if self.frame_counter % 1000 == 0:
                self.export_path_data()
                self.generate_heatmap()
                self.generate_lines()
                logger.info(f"💓 System Alive. Processed {self.frame_counter} frames.")
                self.frame_counter = 0  #set zero after 1000 frames





























# import cv2
# import onnx
# import hashlib
# import supervision as sv
# import onnxruntime as ort
# from src.vision.factory import get_detector
# from src.core.config import settings
# from loguru import logger
# import pandas as pd
# import matplotlib.pyplot as plt
# from datetime import datetime
# import psycopg2
# from psycopg2 import sql, extras
# import time
# import numpy as np
# # import sqlite3
# from pathlib import Path
# from onnxruntime import InferenceSession


# class VisionPipeline:
#     def __init__(self, RTSP_URL):
#         logger.info("⚙️ Initializing Vision Pipeline components...")
#         self.model = get_detector()
#         onnx_path: str = 'models/osnet_x1_0_imagenet.onnx'
#         print(f"🔌 Loading ONNX model: {onnx_path}...")
        
#         self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
#         print(f"🔍 Model input size: {self.session.get_inputs()[0].shape}")
#         # model_input_size: tuple = (256, 128),
#         # expected_input_shape = [1, 3] + list(model_input_size)  # batch, channels, height, width
        
#         # Configuration
#         DB_USER = "your_username"
#         DB_PASSWORD = "your_password"
#         DB_NAME = "your_database_name"
#         DB_HOST = "db"  # Connect to Docker service name 'db'
#         connection_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:5432/{DB_NAME}"
        
#         self.conn = psycopg2.connect(connection_string)
#         self.cur = self.conn.cursor()
#         # --- CONFIGURATION ---
#         DB_PATH = 'instance/personality.db'
#         FAISS_INDEX_PATH = 'faissself._index.idx'
#         FAISS_DIMENSION = 512  # Match your model's output size
#         self.TIME_WINDOW_SECONDS = 300  # 5 minutes

#         # --- INITIALIZATION ---
#         # self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
#         # self.cursor = self.conn.cursor()
#         # self.cursor.execute('''
#         #     CREATE TABLE IF NOT EXISTS people (
#         #         id INTEGER PRIMARY KEY,
#         #         embedding BLOB,
#         #         updated_at REAL
#         #     )
#         # ''')
#         # self.conn.commit()
#         # self.conn.close()

#         # self.index = faiss.IndexFlatL2(FAISS_DIMENSION)
#         # self.index.write_index(FAISS_INDEX_PATH) # Initial load

#         # --- RAM CACHE ---
#         # Holds (person_id, embedding_vector)
#         # This is the "Active Tracks" in RAM
#         self.ram_cache = {} 
        
#         # Configuration
#         self.CONFIG = {
#             'db_path': 'person_history.db',
#             'embedding_size': 512, # Assuming YOLOv8-SE or similar ReID model
#             'distance_threshold': 0.5, # If < 0.5, same person
#             'camera': 'Cam1',
#         }

#         self.RTSP_URL = RTSP_URL
#         self.frame_counter = 0
#         self.tracked_paths = {}
#         self.height = 8224
#         self.width  = 8224

#         self.tracker = sv.ByteTrack() 
#         self.person_class_id = 0 
#         # -----------------------

#         # Setup UI
#         self.fps_monitor = sv.FPSMonitor()
#         color = sv.ColorPalette.DEFAULT 
#         self.box_annotator = sv.BoxAnnotator(color=color)
#         self.trace_annotator = sv.TraceAnnotator(color=color, trace_length=30) # Reduced length for speed
#         self.label_annotator = sv.LabelAnnotator(color=color, text_color=sv.Color.BLACK)

#         # Track how long each person stays visible for dwell/queue analytics.
#         self.track_presence = {}
#         self.completed_presence = {}
#         self.track_timeout_seconds = 2.0
#         self.stall_threshold_seconds = 15.0
#         self.queue_threshold_seconds = 30.0
        
        
#         # Auto-config state
#         self.line_zone = None
#         self.line_zone_annotator = None
#         self.line_initialized = False

#         # Change these depending on your entrance geometry
#         self.door_orientation = "horizontal"   # "horizontal" or "vertical"
#         self.door_position_ratio = 0.75        # 55% down the frame if horizontal
#         self.door_margin_ratio = 0.01          # keep line inside image edges
#         self.trigger_anchor = sv.Position.BOTTOM_CENTER
        
#         logger.info(f"🔌 Connecting to Video Source: {self.RTSP_URL}")
        
#         # Camera
#         self.cap = cv2.VideoCapture(self.RTSP_URL)
#         if not self.cap.isOpened():
#             logger.error(f"❌ COULD NOT OPEN VIDEO SOURCE: {self.RTSP_URL}")
#             # Optional: Retry logic could be logged here
#         else:
#             logger.info("✅ Video Source Connected.")

#     def normalize_vector(self, vec):
#         """FAISS needs normalized vectors for Cosine Similarity."""
#         return vec / np.linalg.norm(vec)

#     def update_database(self, embedding, timestamp, confidence):
#         # """Saves a new detection to the history DB."""
#         # self.CONFIG = {
#         #     'db_path': 'person_history.db',
#         #     'embedding_size': 512, # Assuming YOLOv8-SE or similar ReID model
#         #     'distance_threshold': 0.5, # If < 0.5, same person
#         # }
#         # conn = sqlite3.connect(self.CONFIG['db_path'])
#         # cur = conn.cursor()
        
#         # # Simple schema: id, embedding (raw bytes), path, cam, ts, conf
#         # # Note: SQLite doesn't support vector math natively without extension.
#         # # Using 'blob' column to store numpy array as hex/bytes for simplicity in pure python
#         # # For production, use PostgreSQL with pgvector or SQLite with sqlite-vec
        
#         # # Convert vector to bytes to save in DB
#         # emb_bytes = embedding.tobytes() 
        
#         # cur.execute('''
#         #     INSERT INTO people_history (embedding_blob, timestamp, confidence, is_missing) 
#         #     VALUES (?, ?, ?, ?, ?, ?)
#         # ''', (emb_bytes, timestamp, confidence, 0))
#         # # ''', (emb_bytes, image_path, cam_id, timestamp, conf, 0))
        
        
#         # conn.commit()
#         # conn.close()
        
        
#         try:
#             # 1. Enable extension (Good practice, though usually pre-loaded in image)
#             self.cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
#             # 2. Create Table
#             # Using VECTOR(384) as standard embedding dimension
#             create_table_sql = """
#             CREATE TABLE IF NOT EXISTS embeddings (
#                 id SERIAL PRIMARY KEY,
#                 content TEXT NOT NULL,
#                 embedding vector(384),
#                 created_at TIMESTAMPTZ DEFAULT NOW()
#             );
#             """
#             self.cur.execute(create_table_sql)
            
#             # 3. Create Index (HNSW is required for searching)
#             # The index creates a pgvector index on the vector column
#             create_index_sql = """
#             CREATE INDEX IF NOT EXISTS embeddings_idx ON embeddings (embedding)
#             USING hnsw (embedding vector_cosine_ops)
#             WITH (m = 16, ef_construction = 100);
#             """
#             self.cur.execute(create_index_sql)
            
#             # 4. Insert Dummy Data (Placeholder vector for now)
#             # In production, you would replace this with your actual embedding model output
#             dummy_vector = [0.0] * 384 # Placeholder vector
#             insert_sql = """
#             INSERT INTO embeddings (content, embedding) 
#             VALUES (%s, %s)
#             ON CONFLICT DO NOTHING;
#             """
#             self.cur.execute(insert_sql, ( "Hello world initial data", tuple(dummy_vector) ))
            
#             self.conn.commit()
#             print("Database schema and data initialized successfully.")
            
#         except Exception as e:
#             print(f"Error initializing DB: {e}")
#         finally:
#             self.cur.close()
#             self.conn.close()   
        
        
        
        
        
        
        
        
        
        
   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

#     def normalize_vector(self, vec):
#         """FAISS needs normalized vectors for Cosine Similarity."""
#         return vec / np.linalg.norm(vec)

#     def get_embeddings_from_detections(
#         self,
#         frame: np.ndarray,
#         det_box: tuple,
#         # onnx_path: str = 'osnet_x1_0.onnx',
#         model_input_size: tuple = (256, 128),
#         torso_ratio: float = 2/3
#     ):
#         """
#         Extract embeddings from detected objects in a frame.

#         Args:
#             frame: OpenCV frame (np.array) with detections overlaid or not.
#             detections: List of dicts in format {'bbox': [x1, y1, x2, y2], 'class': 'person'} etc.
#             onnx_path: Path to your ONNX model file.
#             model_input_size: (height, width) of the ONNX model input.
#             torso_ratio: Fraction of box height to crop from top (e.g., 2/3).

#         Returns:
#             List of embeddings (normalized float vectors) with corresponding labels.
#         """
#         print(f"👁️ Processing detections...")
        
#         try:
#             # 1. Extract bbox coordinates
#             x1, y1, x2, y2 = det_box
            
#             # 2. Basic sanity checks
#             w = x2 - x1
#             h = y2 - y1

#             # 3. Crop the torso (Top 2/3)
#             crop_y1 = int(y1)
#             crop_y2 = int(y1 + h * torso_ratio)
#             crop_h = crop_y2 - crop_y1
#             crop_w = w  # Keep full width of torso region
            
#             # Handle edge cases where crop_y2 might exceed frame bounds
#             if crop_y2 > y2:
#                 crop_y2 = y2
            
#             # 4. Crop and Prepare for resizing
#             crop_box = frame[int(y1):int(crop_y2), int(x1):int(x2)]
#             cv2.imwrite(f'data/{x1}.png', crop_box)
#             print(crop_box.shape)
#             # 5. Resize to model input size (256x128) - using INTER_AREA for better quality
#             resized_crop = cv2.resize(crop_box, (model_input_size[1], model_input_size[0]), interpolation=cv2.INTER_AREA)
#             print(resized_crop.shape)
#             cv2.imwrite(f'data/{x1}_2.png', resized_crop)
#             # 6. Normalize (0-255 -> -1 to 1) OR use ImageNet stats if your model is ImageNet trained
#             # Assuming standard ONNX conversion which usually keeps [0, 1] or [-1, 1]
#             normalized_crop = resized_crop.astype(np.float32) / 255.0
#             print(normalized_crop.shape, model_input_size, crop_h, crop_w)
#             # 7. Pad to model height if necessary (Optional step for aspect ratio preservation)
#             # If we crop the height, we might end up with a smaller height. 
#             # If model requires 256x128 and we have a smaller crop, we need padding.
#             # Let's pad with black or replicate borders:
#             if normalized_crop.shape[0] < model_input_size[0]:
#                 pad_h = model_input_size[0] - crop_h
#                 pad_top = pad_h // 2
#                 pad_bot = pad_h - pad_top
#                 pad_top_crop = np.pad(normalized_crop, ((pad_top, pad_bot), (0, 0), (0, 0)), mode='constant', constant_values=0.0)
#                 padded_crop = pad_top_crop
#             else:
#                 padded_crop = normalized_crop
                
#             # 8. Add Batch Dimension [N, C, H, W]
#             padded_crop = np.expand_dims(padded_crop, axis=0)
#             print(33333333333333333333333, padded_crop.shape)
#             input_tensor = np.transpose(padded_crop, (0, 3, 1, 2))  # H, W -> C, H, W
#             print(44444444444444444,input_tensor.shape)
#             # 9. Run Inference
#             outputs = self.session.run(None, {'input': input_tensor})
#             print(1111111111111111111, outputs[0].shape)
#             # 10. Extract Embedding
#             embedding = outputs[0].flatten()
#             return embedding, crop_y1
#         except Exception as e:
#             print(f"❌ Error processing detection: {str(e)}")
    
#     def insert_new_person(self, vector, bbox):
#         vector_b = vector.tobytes()
#         self.cursor.execute("INSERT OR REPLACE INTO people (embedding, updated_at) VALUES (?, ?)", 
#                     (vector_b, time.time()))
#         self.conn.commit()
#         # Get last inserted id
#         self.cursor.execute("SELECT last_insert_rowid()")
#         new_id = self.cursor.fetchone()[0]
#         return new_id

#     def update_db_person(self, pid, vector, current_time):
#         vector_b = vector.tobytes()
#         self.cursor.execute("UPDATE people SET embedding=?, updated_at=? WHERE id=?", 
#                     (vector_b, current_time, pid))
#         self.conn.commit()
#     # --- CLEANUP OLD RAM ---
#     def cleanup_ram(self):
#         current_time = time.time()
#         for pid in list(self.ram_cache.keys()):
#             self.ram_cache[pid]['seen_at'] # Access to check time
#             if current_time - self.ram_cache[pid]['seen_at'] > self.TIME_WINDOW_SECONDS:
#                 del self.ram_cache[pid]
#                 # Also remove from FAISS (optional, but saves memory)
#                 # FAISS doesn't support efficient removal, so you rebuild or use a separateself. index.

#     def _setup_doorway_line(self, frame):
#         # height, width = frame.shape[:2]

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
#             triggering_anchors=[self.trigger_anchor]
#         )

#         self.line_zone_annotator = sv.LineZoneAnnotator(
#             thickness=2,
#             text_thickness=1,
#             text_scale=0.7
#         )

#         self.line_initialized = True

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
   
#     def process_frame(self, frame):
#         # self.frame_counter += 1
#         detected_info = []
#         embeddings = []
#         current_frame_idx = self.frame_counter
        
#         self.fps_monitor.tick()
#         fps = self.fps_monitor.fps
#         # height, width = frame.shape[:2]
#         if not self.line_initialized:
#             self._setup_doorway_line(frame)

#         # 1. Inference
#         detections = self.model.predict(frame)

#         # Keep only people
#         if detections.class_id is not None:
#             detections = detections[detections.class_id == self.person_class_id]

#         # 2. Tracking
#         detections = self.tracker.update_with_detections(detections)
#         now = time.monotonic()
#         active_track_ids = set()
#         track_durations = {}
#         labels = []
#         for obj_i, det in enumerate(detections):
#             det_box, mask, confidence, class_id, tracker_id, data = det
#             if tracker_id is None:
#                 continue

#             # 3. Path Tracking (STORAGE LOGIC)
#             # Extract box coordinates
#             x1, y1, x2, y2 = det_box
#             cx = (x1 + x2) / 2.0
#             cy = (y1 + y2) / 2.0
            
#             # 2. Basic sanity checks
#             w = x2 - x1
#             h = y2 - y1
#             if w < 20 or h < 20:
#                 print(f"⚠️  Detection {obj_i} is too small to process (w={w}, h={h}). Skipping.")
#                 # continue
#             else:
#                 # 11. Store
#                 embedding, crop_y1 =  self.get_embeddings_from_detections(frame, det_box)
#                 # embeddings.append(embedding)
#                 embeddings.append(self.normalize_vector(embedding))
#                 detected_info.append({
#                     "label": class_id,
#                     "bbox": (x1, y1, x2, y2),
#                     "crop_y": crop_y1
#                 })
                
#             if len(embeddings) == 0:
#                 print("⚠️  No detections processed.")
            
#             print(f"✅ Extracted {len(embeddings)} embeddings.")
                

#             # Initialize list for new track
#             if tracker_id not in self.tracked_paths:
#                 self.tracked_paths[tracker_id] = []
            
#             # Store point
#             self.tracked_paths[tracker_id].append({
#                 'frame': current_frame_idx,
#                 'x': cx,
#                 'y': cy
#             })

            
#             normalized_id = int(tracker_id)
#             active_track_ids.add(normalized_id)

#             if normalized_id not in self.track_presence:
#                 self.track_presence[normalized_id] = {
#                     "first_seen": now,
#                     "last_seen": now,
#                 }
#             else:
#                 self.track_presence[normalized_id]["last_seen"] = now

#             presence = self.track_presence[normalized_id]
#             track_durations[normalized_id] = now - presence["first_seen"]
#             # 4. Annotation

#             # self.update_database(embedding, image_path, cam_id, timestamp, confidence)  
#             self.update_database(embedding, track_durations[normalized_id], confidence)    
              
                
#             labels.append(
#                 (
#                     f"#{int(tracker_id)} "
#                     f"{self._format_duration(track_durations.get(int(tracker_id), 0.0))} "
#                     f"{confidence:.2f}"
#                 )
#                 )

           
#         # 3. Trigger line counter
#         # No manual in/out values are passed here.
#         if len(detections) > 0:
#             self.line_zone.trigger(detections=detections)


#         dwell_metrics = self._update_presence_timers(now, active_track_ids, track_durations)
        
#         annotated_frame = frame.copy()
#         annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
#         annotated_frame = self.box_annotator.annotate(annotated_frame, detections)
#         annotated_frame = self.label_annotator.annotate(annotated_frame, detections, labels)

#         # Draw line + in/out counts
#         annotated_frame = self.line_zone_annotator.annotate(
#             frame=annotated_frame,
#             line_counter=self.line_zone
#         )

#         # Extra text
#         occupancy = self.line_zone.in_count - self.line_zone.out_count

#         annotated_frame = sv.draw_text(
#             scene=annotated_frame,
#             text=f"FPS: {fps:.1f}",
#             text_anchor=sv.Point(40, 30),
#             background_color=sv.Color.RED,
#             text_color=sv.Color.WHITE
#         )

#         annotated_frame = sv.draw_text(
#             scene=annotated_frame,
#             text=(
#                 f"IN: {self.line_zone.in_count}  "
#                 f"OUT: {self.line_zone.out_count}  "
#                 f"INSIDE: {occupancy}"
#             ),
#             text_anchor=sv.Point(220, 30),
#             background_color=sv.Color.BLACK,
#             text_color=sv.Color.WHITE
#         )

#         longest_active = self._format_duration(dwell_metrics["longest_active"])
#         average_active = self._format_duration(dwell_metrics["average_active"])
#         average_completed = self._format_duration(dwell_metrics["average_completed"])

#         annotated_frame = sv.draw_text(
#             scene=annotated_frame,
#             text=(
#                 f"VISIBLE: {dwell_metrics['active_count']}  "
#                 f"STALL>= {int(self.stall_threshold_seconds)}s: {dwell_metrics['stall_count']}  "
#                 f"QUEUE>= {int(self.queue_threshold_seconds)}s: {dwell_metrics['queue_count']}"
#             ),
#             text_anchor=sv.Point(200, 65),
#             background_color=sv.Color.BLUE,
#             text_color=sv.Color.WHITE
#         )

#         annotated_frame = sv.draw_text(
#             scene=annotated_frame,
#             text=(
#                 f"LONGEST: {longest_active}  "
#                 f"AVG LIVE: {average_active}  "
#                 f"AVG COMPLETED: {average_completed}"
#             ),
#             text_anchor=sv.Point(200, 100),
#             background_color=sv.Color.BLACK,
#             text_color=sv.Color.WHITE
#         )

#         return annotated_frame
   
#     def generate_frames(self):
#         reading_video = True # Temporary for testing videos
        
#         # OPTIONAL: Setup Video Writer if you want to save
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         out = cv2.VideoWriter('data/output.mp4', fourcc, 30.0, (640, 480))
        
#         # while True:
#         while reading_video:
#             success, frame = self.cap.read()
#             if not success:
#                 reading_video = False
#                 self.export_path_data()
#                 self.generate_heatmap()
#                 self.generate_lines()
#                 logger.warning("⚠️ Frame dropped or video ended. Rewinding...")
#                 self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#                 # continue
#                 break    
            
#             self.height, self.width = frame.shape[:2]
#             output_frame = self.process_frame(frame)
            
#             # OPTIONAL: Write to disk
#             out.write(output_frame)

#             # Stream to browser
#             ret, buffer = cv2.imencode('.jpg', output_frame)
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
#             # TRICK: Log a "Heartbeat" every 1000 frames so you know it's alive
#             self.frame_counter += 1
#             if self.frame_counter % 1000 == 0:
#                 self.export_path_data()
#                 self.generate_heatmap()
#                 self.generate_lines()
#                 logger.info(f"💓 System Alive. Processed {self.frame_counter} frames.")
#                 self.frame_counter = 0  #set zero after 1000 frames



























































