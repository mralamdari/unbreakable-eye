"""
HuggingFace Transformer-based detectors using ONNX Runtime.
Supports: RT-DETR, D-FINE, YOLO-NAS
Best for: Transformer-based detection models with strong accuracy.
"""

import os
import cv2
import numpy as np
import supervision as sv
from loguru import logger
from typing import Tuple

from src.vision.base import BaseDetector
from src.vision.utils import (
    create_session, letterbox, normalize_imagenet, to_chw_float32,
    cxcywh_to_xyxy, scale_boxes
)
from src.core.exceptions import ModelLoadError, InferenceError, PreprocessError, PostprocessError


class HFTransformerDetector(BaseDetector):
    """
    HuggingFace Transformer-based detector using ONNX Runtime.
    Works with RT-DETR, D-FINE, and similar transformer detection models.
    """

    def __init__(
        self,
        model_path: str,
        conf_thresh: float = 0.45,
        device: str = "cpu"
    ):
        """
        Args:
            model_path: Path to ONNX model file
            conf_thresh: Confidence threshold
            device: Device hint ("cuda" or "cpu")

        Raises:
            ModelLoadError: If model cannot be loaded
        """
        self.confidence_threshold = conf_thresh
        self.device = device.lower()

        try:
            num_threads = max(1, (os.cpu_count() or 4) // 2 - 1)
            logger.info(f"Creating Transformer ONNX session with {num_threads} threads")
            self.session = create_session(model_path, num_threads=num_threads)
            self.input_name = self.session.get_inputs()[0].name

            # Get expected input shape
            shape = self.session.get_inputs()[0].shape
            self.input_h = shape[2] if isinstance(shape[2], int) else 640
            self.input_w = shape[3] if isinstance(shape[3], int) else 640

            logger.info(f"Transformer model initialized: input {self.input_h}x{self.input_w}")

        except Exception as e:
            logger.error(f"Failed to initialize Transformer detector: {e}")
            raise ModelLoadError(
                "Failed to initialize Transformer detector",
                context={"model_path": model_path, "error": str(e)}
            ) from e

    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Preprocess frame for Transformer inference.
        
        Transformers typically require ImageNet normalization.

        Args:
            frame: Input BGR image

        Returns:
            (input_tensor, ratio) where input_tensor is (1, 3, H, W) and
            ratio is the scale factor for box rescaling

        Raises:
            PreprocessError: If preprocessing fails
        """
        try:
            # Letterbox with aspect ratio
            img_padded, _ = letterbox(
                frame, self.input_h, self.input_w, pad_value=0  # Transformers use black padding
            )

            # ImageNet normalization (required by most transformers)
            img_norm = normalize_imagenet(img_padded)

            # HWC → CHW
            input_tensor = np.transpose(img_norm, (2, 0, 1))

            # Add batch dimension
            batch = np.expand_dims(input_tensor, axis=0).astype(np.float32)

            # Calculate ratio for rescaling
            h, w = frame.shape[:2]
            scale = min(self.input_h / h, self.input_w / w)
            ratio = 1.0 / scale

            return np.ascontiguousarray(batch), ratio

        except Exception as e:
            logger.error(f"Transformer preprocessing failed: {e}")
            raise PreprocessError(
                "Transformer preprocessing failed",
                context={"image_shape": frame.shape, "error": str(e)}
            ) from e

    def postprocess(
        self,
        scores: np.ndarray,
        boxes: np.ndarray,
        ratio: float,
        img_h: int,
        img_w: int
    ) -> sv.Detections:
        """
        Decode Transformer detector output.

        Args:
            scores: Class scores, shape (N, num_classes) or (N,)
            boxes: Bounding boxes in normalized [0, 1] CXCYWH format, shape (N, 4)
            ratio: Scale ratio from preprocessing
            img_h: Original image height
            img_w: Original image width

        Returns:
            sv.Detections with detected objects

        Raises:
            PostprocessError: If postprocessing fails
        """
        try:
            # Ensure scores is 2D
            if scores.ndim == 1:
                scores = scores[:, np.newaxis]

            # Apply sigmoid if logits (some models export logits)
            if np.min(scores) < 0:  # Likely logits
                scores = 1 / (1 + np.exp(-scores))

            # Get max score and class for each box
            if scores.shape[1] > 1:
                class_ids = np.argmax(scores, axis=1)
                max_scores = np.max(scores, axis=1)
            else:
                class_ids = np.zeros(scores.shape[0], dtype=int)
                max_scores = scores.squeeze()

            # Filter by confidence
            mask = max_scores > self.confidence_threshold
            if not np.any(mask):
                return sv.Detections.empty()

            boxes = boxes[mask]
            max_scores = max_scores[mask]
            class_ids = class_ids[mask]

            # Convert from normalized CXCYWH to XYXY in model input space
            # boxes are in [0, 1] range (normalized by model input size)
            boxes_abs = boxes.copy()
            boxes_abs[:, 0] *= self.input_w  # cx
            boxes_abs[:, 1] *= self.input_h  # cy
            boxes_abs[:, 2] *= self.input_w  # w
            boxes_abs[:, 3] *= self.input_h  # h

            boxes_xyxy = cxcywh_to_xyxy(boxes_abs)

            # Rescale to original image coordinates
            boxes_xyxy = scale_boxes(
                boxes_xyxy, ratio, ratio,
                clip_h=img_h,
                clip_w=img_w
            )

            return sv.Detections(
                xyxy=boxes_xyxy,
                confidence=max_scores,
                class_id=class_ids.astype(int)
            )

        except Exception as e:
            logger.error(f"Transformer postprocessing failed: {e}")
            raise PostprocessError(
                "Transformer postprocessing failed",
                context={"scores_shape": scores.shape, "boxes_shape": boxes.shape, "error": str(e)}
            ) from e

    def predict(self, frame: np.ndarray) -> sv.Detections:
        """
        Run inference on a single frame.

        Args:
            frame: Input BGR image

        Returns:
            sv.Detections with detected objects

        Raises:
            InferenceError: If inference fails
        """
        try:
            img_h, img_w = frame.shape[:2]
            input_tensor, ratio = self.preprocess(frame)

            # Run inference
            outputs = self.session.run(None, {self.input_name: input_tensor})

            # Identify output structure
            # Most transformer detectors output: [logits, boxes] or [scores, boxes]
            # Logits/scores usually have last dim = num_classes
            # Boxes always have last dim = 4
            if len(outputs) == 2:
                out1, out2 = outputs[0][0], outputs[1][0]
            else:
                # Single output, might need to split
                logger.warning("Unexpected output count. Assuming first output is scores, second is boxes.")
                out1, out2 = outputs[0][0], outputs[0][0]

            # Determine which is which by shape
            if out1.shape[-1] == 4:
                boxes, scores = out1, out2
            else:
                boxes, scores = out2, out1

            detections = self.postprocess(scores, boxes, ratio, img_h, img_w)
            logger.debug(f"Transformer inference: {len(detections)} detections")
            return detections

        except Exception as e:
            logger.error(f"Transformer inference failed: {e}")
            raise InferenceError(
                "Transformer inference failed",
                context={"image_shape": frame.shape, "error": str(e)}
            ) from e
