import cv2
import supervision as sv
from src.vision.factory import get_detector
from src.core.config import settings
from loguru import logger  # <--- Import


class VisionPipeline:
    def __init__(self):
        logger.info("⚙️ Initializing Vision Pipeline components...")
        self.model = get_detector()

        # --- THE FIX IS HERE ---
        # Instead of your broken SORTTracker, use this:

        self.tracker = sv.ByteTrack() 
        # -----------------------

        # Setup UI
        self.fps_monitor = sv.FPSMonitor()
        color = sv.ColorPalette.DEFAULT 
        self.box_annotator = sv.BoxAnnotator(color=color)
        self.trace_annotator = sv.TraceAnnotator(color=color, trace_length=30) # Reduced length for speed
        self.label_annotator = sv.LabelAnnotator(color=color, text_color=sv.Color.BLACK)

        logger.info(f"🔌 Connecting to Video Source: {settings.RTSP_URL}")
        
        # Camera
        self.cap = cv2.VideoCapture(settings.RTSP_URL)
        if not self.cap.isOpened():
            logger.error(f"❌ COULD NOT OPEN VIDEO SOURCE: {settings.RTSP_URL}")
            # Optional: Retry logic could be logged here
        else:
            logger.info("✅ Video Source Connected.")

    def process_frame(self, frame):
        self.fps_monitor.tick()
        fps = self.fps_monitor.fps
        
        # 1. Inference
        detections = self.model.predict(frame)

        # 2. Tracking (The syntax changes slightly for ByteTrack)
        detections = self.tracker.update_with_detections(detections)
        
        # 3. Annotation
        labels = []
        annotated_frame = frame.copy()
        if detections.tracker_id is not None and detections.confidence is not None \
           and len(detections.tracker_id) == len(detections.confidence): # Check lengths match
        # if detections.tracker_id is not None:
            labels = [
                f"#{tracker_id} {conf:.2f}" 
                for tracker_id, conf in zip(detections.tracker_id, detections.confidence)
            ]

        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.box_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.label_annotator.annotate(annotated_frame, detections, labels)
        
        # FPS
        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=f"FPS: {fps:.1f}",
            text_anchor=sv.Point(40, 30),
            background_color=sv.Color.RED,
            text_color=sv.Color.WHITE
        )

        return annotated_frame
    
    def generate_frames(self):
        # OPTIONAL: Setup Video Writer if you want to save
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (640, 480))
        frame_count = 0
        while True:
            success, frame = self.cap.read()
            if not success:
                logger.warning("⚠️ Frame dropped or video ended. Rewinding...")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue    
            
            output_frame = self.process_frame(frame)
            
            # OPTIONAL: Write to disk
            # out.write(output_frame)

            # Stream to browser
            ret, buffer = cv2.imencode('.jpg', output_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            # TRICK: Log a "Heartbeat" every 1000 frames so you know it's alive
            frame_count += 1
            if frame_count % 1000 == 0:
                frame_count = 0  #set zero after 1000 frames
                logger.info(f"💓 System Alive. Processed {frame_count} frames.")