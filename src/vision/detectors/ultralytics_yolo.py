"""
Ultralytics YOLO detector wrapper using native PyTorch implementation.
Best for: Real-time inference when GPU/CUDA is available.
"""

import numpy as np
import supervision as sv
from loguru import logger

from src.vision.base import BaseDetector
from src.core.exceptions import ModelLoadError, InferenceError

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # loaded lazily — only fails when UltralyticsDetector is instantiated


class UltralyticsDetector(BaseDetector):
    """
    YOLO detector using Ultralytics native implementation.
    Supports YOLO, YOLOv10, and other ultralytics models.
    """

    def __init__(self, model_path: str, conf_thresh: float = 0.45, device: str = "cuda"):
        """
        Args:
            model_path: Path to .pt model file (e.g., yolov8n.pt)
            conf_thresh: Confidence threshold
            device: Device to use ("cuda" or "cpu")

        Raises:
            ModelLoadError: If model cannot be loaded
        """
        if YOLO is None:
            raise ModelLoadError(
                "ultralytics is not installed. Run: pip install ultralytics",
                context={"model_path": model_path},
            )
        self.confidence_threshold = conf_thresh
        self.device = device.lower()

        try:
            logger.info(f"Loading YOLO model from {model_path} on {self.device}")
            self.model = YOLO(model_path)
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise ModelLoadError(
                f"Failed to load YOLO model from {model_path}",
                context={"path": model_path, "error": str(e)}
            ) from e

    def predict(self, frame: np.ndarray) -> sv.Detections:
        """
        Run inference on a single frame.

        Args:
            frame: Input BGR image

        Returns:
            sv.Detections object with boxes, confidence, class_id

        Raises:
            InferenceError: If inference fails
        """
        try:
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                verbose=False,
                device=self.device
            )[0]
            
            detections = sv.Detections.from_ultralytics(results)
            logger.debug(f"YOLO inference: {len(detections)} detections")
            return detections

        except Exception as e:
            logger.error(f"YOLO inference failed: {e}")
            raise InferenceError(
                "YOLO inference failed",
                context={"frame_shape": frame.shape, "error": str(e)}
            ) from e

    def predict_batch(self, frames: list[np.ndarray]) -> list[sv.Detections]:
        """
        Run inference on multiple frames.

        Args:
            frames: List of BGR images

        Returns:
            List of sv.Detections objects

        Raises:
            InferenceError: If any inference fails
        """
        try:
            results = self.model(
                frames,
                conf=self.confidence_threshold,
                verbose=False,
                device=self.device
            )
            
            detections_list = [sv.Detections.from_ultralytics(r) for r in results]
            logger.debug(f"YOLO batch inference: {len(frames)} frames, "
                        f"{sum(len(d) for d in detections_list)} total detections")
            return detections_list

        except Exception as e:
            logger.error(f"YOLO batch inference failed: {e}")
            raise InferenceError(
                "YOLO batch inference failed",
                context={"num_frames": len(frames), "error": str(e)}
            ) from e
