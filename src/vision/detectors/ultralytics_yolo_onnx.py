"""
Ultralytics YOLOv8 detector using ONNX Runtime.
Best for: Edge deployment, consistent cross-platform performance.
"""

import os
import numpy as np
import supervision as sv
from loguru import logger
from typing import Tuple

from src.core.config import settings
from src.vision.base import BaseDetector
from src.vision.utils import (
        create_session, letterbox,
        to_chw_float32, scale_boxes)
from src.core.exceptions import ModelLoadError, InferenceError, PreprocessError


class UltralyticsONNXDetector(BaseDetector):
    """
    YOLOv8 detector using ONNX Runtime backend.
    Lightweight and cross-platform alternative to native PyTorch.
    """

    def __init__(
        self,
        model_path: str,
        conf_thresh: float = 0.45,
        iou_thres: float = 0.45,
        device: str = "cpu"
    ):
        """
        Args:
            model_path: Path to .onnx model file
            conf_thresh: Confidence threshold for detections
            iou_thres: IOU threshold for NMS
            device: Device hint ("cuda" or "cpu") — used only for context

        Raises:
            ModelLoadError: If model cannot be loaded
        """
        self.confidence_thres = conf_thresh
        self.iou_thres = iou_thres
        self.device = device.lower()

        try:
            num_threads = max(1, (os.cpu_count() or 4) // 2 - 1)
            logger.info(f"Creating YOLO ONNX session with {num_threads} threads")
            self.session = create_session(model_path, num_threads=num_threads)
            self.input_name = self.session.get_inputs()[0].name
            self.input_size = settings.FRAME_SHAPE  # (H, W, C)
            logger.info(f"YOLO ONNX model initialized: input shape {self.input_size}")
        except Exception as e:
            logger.error(f"Failed to initialize YOLO ONNX detector: {e}")
            raise

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Preprocess image for YOLO ONNX inference.
        
        Applies: BGR→RGB, letterbox padding, normalization, HWC→CHW.

        Args:
            image: Input BGR image (uint8)

        Returns:
            (input_tensor, (top_pad, left_pad))
            where input_tensor is (3, H, W) float32 ready for inference

        Raises:
            PreprocessError: If preprocessing fails
        """
        try:
            # Letterbox: preserve aspect ratio, pad to target size
            img_padded, (top_pad, left_pad) = letterbox(
                image,
                self.input_size[0],  # target_h = 512
                self.input_size[1],  # target_w = 512
                pad_value=114  # YOLO uses gray padding
            )

            # Normalize and convert to CHW float32
            input_tensor = to_chw_float32(img_padded, normalize=True)
            
            return input_tensor, (top_pad, left_pad)

        except Exception as e:
            logger.error(f"YOLO preprocessing failed: {e}")
            raise PreprocessError(
                "YOLO preprocessing failed",
                context={"image_shape": image.shape, "error": str(e)}
            ) from e

    def postprocess(
        self,
        output: np.ndarray,
        pad: Tuple[int, int],
        original_shape: Tuple[int, int]
    ) -> sv.Detections:
        """
        Decode raw YOLO ONNX output and apply NMS.

        Args:
            output: Raw model output, shape (1, 84, N) or (N, 84)
            pad: (top_pad, left_pad) from preprocessing
            original_shape: (H, W) of the original image

        Returns:
            sv.Detections with xyxy boxes, confidence, class_id

        Raises:
            PostprocessError: If postprocessing fails
        """
        try:
            # Squeeze batch dimension if present
            if output.ndim == 3:
                output = np.squeeze(output, axis=0)
            
            # Transpose if needed: (84, N) → (N, 84)
            if output.shape[0] == 84:
                output = output.T

            # Extract boxes and class scores
            boxes_xywh = output[:, 0:4]
            scores = output[:, 4:]

            # Confidence filtering
            max_scores = np.max(scores, axis=1)
            mask = max_scores >= self.confidence_thres
            
            if not np.any(mask):
                return sv.Detections.empty()

            boxes_xywh = boxes_xywh[mask]
            max_scores = max_scores[mask]
            class_ids = np.argmax(scores[mask], axis=1)
            # Convert CXCYWH (model output) → XYXY (supervision format)
            cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            
            boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

            # Remove padding and scale back to original image
            top_pad, left_pad = pad
            gain = min(self.input_size[0] / original_shape[0],
                      self.input_size[1] / original_shape[1])
            
            boxes_xyxy[:, 0] = (boxes_xyxy[:, 0] - left_pad) / gain
            boxes_xyxy[:, 1] = (boxes_xyxy[:, 1] - top_pad) / gain
            boxes_xyxy[:, 2] = (boxes_xyxy[:, 2] - left_pad) / gain
            boxes_xyxy[:, 3] = (boxes_xyxy[:, 3] - top_pad) / gain

            # Clip to original image boundaries
            boxes_xyxy = scale_boxes(
                boxes_xyxy, 1.0, 1.0,
                clip_h=original_shape[0],
                clip_w=original_shape[1]
            )

            # Apply NMS
            nms_values = np.stack([boxes_xyxy[:, 0], boxes_xyxy[:, 1],
                                   boxes_xyxy[:, 2], boxes_xyxy[:, 3],
                                   max_scores], axis=1).astype(np.float32)
            indices = sv.box_non_max_suppression(nms_values, self.iou_thres)

            return sv.Detections(
                xyxy=boxes_xyxy[indices],
                confidence=max_scores[indices],
                class_id=class_ids[indices]
            )

        except Exception as e:
            logger.error(f"YOLO postprocessing failed: {e}")
            raise PostprocessError(
                "YOLO postprocessing failed",
                context={"output_shape": output.shape, "error": str(e)}
            ) from e

    def predict(self, image: np.ndarray) -> sv.Detections:
        """
        Run inference on a single image.

        Args:
            image: Input BGR image

        Returns:
            sv.Detections with detected objects

        Raises:
            InferenceError: If inference fails
        """
        try:
            img_data, pad = self.preprocess(image)
            batch = img_data[np.newaxis]  # Add batch dimension: (1, 3, H, W)

            outputs = self.session.run(None, {self.input_name: batch})
            detections = self.postprocess(outputs[0], pad, image.shape[:2])

            logger.debug(f"YOLO inference: {len(detections)} detections")
            return detections

        except Exception as e:
            logger.error(f"YOLO inference failed: {e}")
            raise InferenceError(
                "YOLO inference failed",
                context={"image_shape": image.shape, "error": str(e)}
            ) from e

    def predict_batch(self, frames: list[np.ndarray]) -> list[sv.Detections]:
        """
        Run inference on multiple frames (requires dynamic=True export).

        Args:
            frames: List of BGR images

        Returns:
            List of sv.Detections objects

        Raises:
            InferenceError: If batch inference fails
        """
        try:
            tensors = []
            pads = []
            orig_shapes = []

            for frame in frames:
                t, p = self.preprocess(frame)
                tensors.append(t)
                pads.append(p)
                orig_shapes.append(frame.shape[:2])

            # Stack into batch: (N, 3, H, W)
            batch = np.stack(tensors, axis=0)

            # Batch inference
            raw = self.session.run(None, {self.input_name: batch})[0]

            # Postprocess each frame
            detections_list = [
                self.postprocess(raw[i], pads[i], orig_shapes[i])
                for i in range(len(frames))
            ]

            logger.debug(f"YOLO batch inference: {len(frames)} frames, "
                        f"{sum(len(d) for d in detections_list)} total detections")
            return detections_list

        except Exception as e:
            logger.error(f"YOLO batch inference failed: {e}")
            raise InferenceError(
                "YOLO batch inference failed",
                context={"num_frames": len(frames), "error": str(e)}
            ) from e
