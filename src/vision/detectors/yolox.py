"""
YOLOX detector using ONNX Runtime.
Best for: Edge devices, models with pre-calculated anchor grids.
"""

import os
import cv2
import numpy as np
import supervision as sv
from loguru import logger
from typing import Tuple

from src.vision.base import BaseDetector
from src.vision.utils import create_session, letterbox, normalize_simple, to_chw_float32, apply_nms_opencv
from src.core.exceptions import ModelLoadError, InferenceError, PreprocessError


class YOLOXDetector(BaseDetector):
    """
    YOLOX detector with optimized anchor grid pre-calculation.
    Avoids recalculating grid positions every frame.
    """

    def __init__(
        self,
        model_path: str,
        conf_thresh: float = 0.45,
        nms_thresh: float = 0.45,
        class_agnostic: bool = True
    ):
        """
        Args:
            model_path: Path to YOLOX .onnx model
            conf_thresh: Confidence threshold
            nms_thresh: NMS IOU threshold
            class_agnostic: Use class-agnostic NMS

        Raises:
            ModelLoadError: If model cannot be loaded
        """
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.class_agnostic = class_agnostic

        try:
            num_threads = max(1, (os.cpu_count() or 4) // 2 - 1)
            logger.info(f"Creating YOLOX ONNX session with {num_threads} threads")
            self.session = create_session(model_path, num_threads=num_threads)
            self.input_name = self.session.get_inputs()[0].name

            # Get model input shape
            # self.input_size = settings.FRAME_SHAPE  # (H, W, C)
            # self.input_h, self.input_w, _ = self.input_size
            shape = self.session.get_inputs()[0].shape
            self.input_h = shape[2] if isinstance(shape[2], int) else 640
            self.input_w = shape[3] if isinstance(shape[3], int) else 640

            logger.info(f"YOLOX model initialized: input {self.input_h}x{self.input_w}")

            # Pre-calculate grids (expensive operation, do once)
            self._generate_grids()

        except Exception as e:
            logger.error(f"Failed to initialize YOLOX detector: {e}")
            raise ModelLoadError(
                f"Failed to initialize YOLOX detector",
                context={"model_path": model_path, "error": str(e)}
            ) from e

    def _generate_grids(self):
        """
        Pre-calculate anchor grids for all scales.
        YOLOX optimization: avoid grid recalculation every frame.
        
        Grids are used to decode center offsets into absolute coordinates.
        """
        strides = [8, 16, 32]  # Multi-scale pyramid
        self.grids = []
        self.expanded_strides = []

        for stride in strides:
            # Create a grid of positions at this scale
            hsize = self.input_h // stride
            wsize = self.input_w // stride
            
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            
            self.grids.append(grid)
            self.expanded_strides.append(np.full((*grid.shape[:2], 1), stride))

        self.grids = np.concatenate(self.grids, 1)
        self.expanded_strides = np.concatenate(self.expanded_strides, 1)
        logger.debug(f"YOLOX grids pre-calculated: {self.grids.shape}")

    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Preprocess frame for YOLOX inference.

        Args:
            frame: Input BGR image

        Returns:
            (input_blob, ratio) where input_blob is (1, 3, H, W) and
            ratio is the scale factor for rescaling boxes

        Raises:
            PreprocessError: If preprocessing fails
        """
        try:
            # Letterbox with aspect ratio preservation
            img_padded, _ = letterbox(
                frame, self.input_h, self.input_w, pad_value=0  # YOLOX uses black padding
            )

            # YOLOX normalization: simple 0-1 scaling
            img_norm = normalize_simple(img_padded)

            # HWC → CHW
            img_chw = np.transpose(img_norm, (2, 0, 1))

            # Add batch dimension
            blob = np.expand_dims(img_chw, axis=0).astype(np.float32)

            # Calculate ratio for rescaling boxes back to original space
            h, w = frame.shape[:2]
            scale = min(self.input_h / h, self.input_w / w)
            ratio = 1.0 / scale

            return np.ascontiguousarray(blob), ratio

        except Exception as e:
            logger.error(f"YOLOX preprocessing failed: {e}")
            raise PreprocessError(
                "YOLOX preprocessing failed",
                context={"image_shape": frame.shape, "error": str(e)}
            ) from e

    def postprocess(
        self,
        outputs: np.ndarray,
        ratio: float
    ) -> sv.Detections:
        """
        Decode YOLOX raw output and apply NMS.

        Args:
            outputs: Model output shape (1, 8400, 85) — batch, anchors, xywh+obj+classes
            ratio: Scale ratio for rescaling boxes

        Returns:
            sv.Detections with detected objects

        Raises:
            PostprocessError: If postprocessing fails
        """
        try:
            # outputs is (batch=1, num_anchors, features=85)
            outputs = outputs[0]  # Remove batch dimension → (num_anchors, 85)

            # Decode boxes using pre-calculated grids
            # YOLOX output is: [dx, dy, dw, dh, objectness, class_scores...]
            # xy = (raw_xy + grid) * stride
            outputs[:, :2] = (outputs[:, :2] + self.grids[0]) * self.expanded_strides[0]
            # wh = exp(raw_wh) * stride
            outputs[:, 2:4] = np.exp(outputs[:, 2:4]) * self.expanded_strides[0]

            # Extract boxes and scores
            boxes = outputs[:, :4]  # [cx, cy, w, h]
            obj_conf = outputs[:, 4:5]  # Objectness
            cls_scores = outputs[:, 5:]  # Class scores

            # Combine objectness + class scores
            scores = obj_conf * cls_scores

            # Get class ID and max score for each anchor
            class_ids = np.argmax(scores, axis=1)
            max_scores = np.max(scores, axis=1)

            # Filter by confidence
            mask = max_scores > self.conf_thresh
            if not np.any(mask):
                return sv.Detections.empty()

            boxes = boxes[mask]
            max_scores = max_scores[mask]
            class_ids = class_ids[mask]

            # Convert CXCYWH → XYXY
            x1 = boxes[:, 0] - boxes[:, 2] / 2
            y1 = boxes[:, 1] - boxes[:, 3] / 2
            x2 = boxes[:, 0] + boxes[:, 2] / 2
            y2 = boxes[:, 1] + boxes[:, 3] / 2

            boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1) / ratio

            # Apply NMS using OpenCV (YOLOX tradition)
            indices = apply_nms_opencv(boxes_xyxy, max_scores, self.conf_thresh, self.nms_thresh)

            if len(indices) == 0:
                return sv.Detections.empty()

            return sv.Detections(
                xyxy=boxes_xyxy[indices],
                confidence=max_scores[indices],
                class_id=class_ids[indices].astype(int)
            )

        except Exception as e:
            logger.error(f"YOLOX postprocessing failed: {e}")
            raise PostprocessError(
                "YOLOX postprocessing failed",
                context={"output_shape": outputs.shape if outputs.ndim else "?", "error": str(e)}
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
            blob, ratio = self.preprocess(frame)

            # Inference
            outputs = self.session.run(None, {self.input_name: blob})

            detections = self.postprocess(outputs[0], ratio)
            logger.debug(f"YOLOX inference: {len(detections)} detections")
            return detections

        except Exception as e:
            logger.error(f"YOLOX inference failed: {e}")
            raise InferenceError(
                "YOLOX inference failed",
                context={"image_shape": frame.shape, "error": str(e)}
            ) from e
