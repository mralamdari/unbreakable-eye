"""
YOLOX detector using ONNX Runtime.

YOLOX-specific characteristics:
  - Input: raw 0-255 BGR pixels (NOT normalized, NOT divided by 255)
  - Preprocessing: aspect-ratio-preserving resize + top-left padding (pad=114)
  - Output: (1, N, 5+num_classes) where N = Σ (H/s * W/s) for s in [8,16,32]
  - Decoding: grid-cell offsets + strides computed per-call (can't pre-cache
    because grid depends on input shape which is always the same, but the
    decode uses mutable in-place ops on the output tensor)
  - NMS: class-agnostic via nms_numpy() in utils

Reference: https://github.com/Megvii-BaseDetection/YOLOX
"""

import os
import cv2
import numpy as np
import supervision as sv
from loguru import logger
from typing import Optional, Tuple

from src.core.config import settings
from src.vision.base import BaseDetector
from src.vision.utils import create_session, nms_numpy
from src.core.exceptions import ModelLoadError, InferenceError, PreprocessError, PostprocessError


def _multiclass_nms_class_agnostic(
    boxes: np.ndarray,
    scores: np.ndarray,
    nms_thresh: float,
    score_thresh: float,
) -> Optional[np.ndarray]:
    """
    Class-agnostic NMS matching YOLOX's official demo_utils.py behaviour.

    Args:
        boxes:        (N, 4) float32 [x1, y1, x2, y2] in original image pixels.
        scores:       (N, num_classes) float32 — objectness * class probabilities.
        nms_thresh:   IoU suppression threshold.
        score_thresh: Minimum confidence to even enter NMS.

    Returns:
        (K, 6) array [x1, y1, x2, y2, score, class_id], or None if no survivors.
    """
    cls_inds   = scores.argmax(axis=1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid      = cls_scores > score_thresh
    if not valid.any():
        return None

    valid_boxes  = boxes[valid]
    valid_scores = cls_scores[valid]
    valid_cls    = cls_inds[valid]

    keep = nms_numpy(valid_boxes, valid_scores, nms_thresh)
    if len(keep) == 0:
        return None

    return np.concatenate([
        valid_boxes[keep],
        valid_scores[keep, np.newaxis],
        valid_cls[keep, np.newaxis],
    ], axis=1)


class YOLOXDetector(BaseDetector):
    """YOLOX object detector via ONNX Runtime."""

    def __init__(
        self,
        model_path: str,
        conf_thresh: float = 0.45,
        nms_thresh:  float = 0.45,
        class_agnostic: bool = True,
    ):
        """
        Args:
            model_path:     Path to YOLOX .onnx model file.
            conf_thresh:    Minimum confidence score to keep a detection.
            nms_thresh:     IoU threshold for NMS suppression.
            class_agnostic: If True, use class-agnostic NMS (YOLOX default).

        Raises:
            ModelLoadError: If the ONNX session cannot be created.
        """
        self.conf_thresh    = conf_thresh
        self.nms_thresh     = nms_thresh
        self.class_agnostic = class_agnostic

        try:
            num_threads = max(1, (os.cpu_count() or 4) // 2 - 1)
            logger.info(f"Loading YOLOX ONNX | path={model_path} | threads={num_threads}")
            self.session    = create_session(model_path, num_threads=num_threads)
            self.input_name = self.session.get_inputs()[0].name

            shape        = self.session.get_inputs()[0].shape  # [1, 3, H, W]
            self.input_h = int(shape[2]) if isinstance(shape[2], int) else 640
            self.input_w = int(shape[3]) if isinstance(shape[3], int) else 640
            self.input_size = (self.input_h, self.input_w)

            logger.info(
                f"YOLOX ready | input={self.input_h}x{self.input_w} "
                f"| conf={conf_thresh} | nms={nms_thresh}"
            )

        except ModelLoadError:
            raise
        except Exception as e:
            raise ModelLoadError(
                "Failed to initialise YOLOX detector",
                context={"model_path": model_path, "error": str(e)},
            ) from e

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Resize with aspect-ratio preservation + top-left padding.

        YOLOX expects raw 0-255 BGR pixel values — no /255, no mean/std.

        Args:
            img: BGR uint8 image (any resolution).

        Returns:
            blob:  (1, 3, input_h, input_w) float32 — raw pixel values.
            ratio: scale factor used; divide decoded box coords by this.

        Raises:
            PreprocessError: If preprocessing fails.
        """
        try:
            ih, iw = img.shape[:2]
            ratio   = min(self.input_h / ih, self.input_w / iw)
            new_h   = int(ih * ratio)
            new_w   = int(iw * ratio)

            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Top-left padding with gray (114) — matches YOLOX training aug
            canvas = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
            canvas[:new_h, :new_w] = resized

            # CHW, float32, raw 0-255 — NO normalization
            blob = canvas.transpose(2, 0, 1).astype(np.float32)
            blob = np.ascontiguousarray(blob[np.newaxis])  # (1, 3, H, W)
            return blob, ratio

        except Exception as e:
            raise PreprocessError(
                "YOLOX preprocessing failed",
                context={"image_shape": list(img.shape), "error": str(e)},
            ) from e

    # ── Grid decoding ─────────────────────────────────────────────────────────

    def _decode_outputs(self, outputs: np.ndarray) -> np.ndarray:
        """
        Decode raw YOLOX network output from grid-cell space to image pixels.

        YOLOX outputs center offsets relative to grid cells and log-scale
        width/height. This function maps them to absolute coordinates on the
        (input_h, input_w) canvas — caller then divides by ratio to get
        original-image coordinates.

        Args:
            outputs: (1, N, 5+num_classes) raw ONNX output.

        Returns:
            (N, 5+num_classes) float32 with xy decoded to absolute canvas pixels.
        """
        grids, strides_exp = [], []
        for stride in ([8, 16, 32]):
            hs = self.input_h // stride
            ws = self.input_w // stride
            xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
            grid   = np.stack((xv, yv), axis=2).reshape(1, -1, 2)
            grids.append(grid)
            strides_exp.append(np.full((*grid.shape[:2], 1), stride))

        grids       = np.concatenate(grids, axis=1)
        strides_exp = np.concatenate(strides_exp, axis=1)

        out = outputs.copy()
        out[..., :2] = (out[..., :2] + grids) * strides_exp   # cx, cy
        out[..., 2:4] = np.exp(out[..., 2:4]) * strides_exp   # w, h
        return out[0]  # remove batch dim → (N, 5+C)

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, frame: np.ndarray) -> sv.Detections:
        """
        Run YOLOX inference on a single BGR frame.

        Args:
            frame: BGR uint8 image (any resolution).

        Returns:
            sv.Detections in original-image pixel coordinates.

        Raises:
            InferenceError: If inference fails.
        """
        try:
            blob, ratio = self.preprocess(frame)
            raw_outputs = self.session.run(None, {self.input_name: blob})

            predictions = self._decode_outputs(raw_outputs[0])  # (N, 5+C)

            boxes      = predictions[:, :4]          # cx, cy, w, h (canvas pixels)
            obj_conf   = predictions[:, 4:5]         # objectness
            cls_scores = predictions[:, 5:]          # per-class probabilities
            scores     = obj_conf * cls_scores       # combined score

            # Convert cxcywh → xyxy (still on letterbox canvas)
            boxes_xyxy = np.stack([
                boxes[:, 0] - boxes[:, 2] / 2,
                boxes[:, 1] - boxes[:, 3] / 2,
                boxes[:, 0] + boxes[:, 2] / 2,
                boxes[:, 1] + boxes[:, 3] / 2,
            ], axis=1)

            # Undo letterbox scaling → original image pixels
            boxes_xyxy /= ratio

            dets = _multiclass_nms_class_agnostic(
                boxes_xyxy, scores,
                nms_thresh=self.nms_thresh,
                score_thresh=self.conf_thresh,
            )

            if dets is None:
                return sv.Detections.empty()

            logger.debug(f"YOLOX | detections={len(dets)} | conf>{self.conf_thresh}")

            return sv.Detections(
                xyxy=dets[:, :4].astype(np.float32),
                confidence=dets[:, 4].astype(np.float32),
                class_id=dets[:, 5].astype(int),
            )

        except (PreprocessError, PostprocessError, InferenceError):
            raise
        except Exception as e:
            raise InferenceError(
                "YOLOX inference failed",
                context={"frame_shape": list(frame.shape), "error": str(e)},
            ) from e
