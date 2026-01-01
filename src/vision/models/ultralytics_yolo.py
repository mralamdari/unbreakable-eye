import numpy as np
from ultralytics import YOLO
import supervision as sv
from src.vision.base import BaseDetector

class UltralyticsDetector(BaseDetector):
    def __init__(self, model_path: str, conf_thresh: float, device: str):
        self.confidence_threshold = conf_thresh
        self.load_model(model_path)

        self.device = 0 if device.upper() in ["CUDA", "GPU"] else 'cpu'
        # Warmup (Optional, but good for production)
        # self.model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)

    def load_model(self, model_path: str):
        self.model = YOLO(model_path)
    
    def predict(self, frame: np.ndarray) -> sv.Detections:
        results = self.model(
            frame, 
            conf=self.confidence_threshold, 
            verbose=False,
            device=self.device
        )[0]

        return sv.Detections.from_ultralytics(results)
