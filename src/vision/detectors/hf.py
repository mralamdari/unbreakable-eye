"""
RF-DETR / D-FINE detector using ONNX Runtime.

Key facts about these models' ONNX exports:
  - Input:   (1, 3, H, W) float32, ImageNet-normalized, RGB channel order
  - Resize:  plain cv2.resize — NO letterboxing (model handles coord mapping)
  - Outputs: two tensors — boxes (N,4) normalized [0,1] XYXY + labels (N, num_classes)
  - Decode:  boxes * [img_w, img_h, img_w, img_h] → absolute pixel coords

Reference: https://rfdetr.roboflow.com/develop/learn/export/#onnx-runtime
"""

import os
import cv2
import numpy as np
import supervision as sv
from loguru import logger
from typing import Tuple

from src.vision.base import BaseDetector
from src.vision.utils import create_session, bgr_to_rgb, normalize_imagenet, nms_numpy, cxcywh_to_xyxy
from src.core.exceptions import ModelLoadError, InferenceError, PreprocessError, PostprocessError


class HFTransformerDetector(BaseDetector):
    """RF-DETR / D-FINE detector via ONNX Runtime."""

    def __init__(
        self,
        model_path: str,
        conf_thresh: float = 0.45,
        device: str = "cpu",
    ):
        """
        Args:
            model_path:  Path to the exported .onnx file.
            conf_thresh: Confidence threshold (0.3–0.5 recommended for RF-DETR).
            device:      "cpu" or "cuda" (informational — session auto-selects).

        Raises:
            ModelLoadError: If the ONNX session cannot be created.
        """
        self.confidence_threshold = conf_thresh

        try:
            num_threads = max(1, (os.cpu_count() or 4) // 2 - 1)
            logger.info(f"Loading RF-DETR ONNX | path={model_path} | threads={num_threads}")
            self.session    = create_session(model_path, num_threads=num_threads)
            self.input_name = self.session.get_inputs()[0].name

            # Read input resolution from the ONNX graph — never hardcode
            shape        = self.session.get_inputs()[0].shape  # [1, 3, H, W]
            self.input_h = int(shape[2]) if isinstance(shape[2], int) else 560
            self.input_w = int(shape[3]) if isinstance(shape[3], int) else 560

            # Identify output indices by name (boxes / labels)
            self._boxes_idx  = None
            self._labels_idx = None
            for i, out in enumerate(self.session.get_outputs()):
                name = out.name.lower()
                if "box" in name:
                    self._boxes_idx = i
                elif "label" in name or "score" in name or "logit" in name:
                    self._labels_idx = i

            output_names = [o.name for o in self.session.get_outputs()]
            logger.info(
                f"RF-DETR ready | input={self.input_h}x{self.input_w} "
                f"| outputs={output_names} | conf={conf_thresh}"
            )

        except ModelLoadError:
            raise
        except Exception as e:
            raise ModelLoadError(
                "Failed to initialise RF-DETR ONNX detector",
                context={"model_path": model_path, "error": str(e)},
            ) from e

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Prepare a BGR frame for RF-DETR ONNX inference.

        Pipeline (matches Roboflow official example exactly):
          BGR → RGB → plain resize (no letterbox) → /255 → ImageNet norm → NCHW

        Args:
            frame: BGR uint8 image (any resolution).

        Returns:
            (1, 3, input_h, input_w) float32 contiguous array.

        Raises:
            PreprocessError: If any step fails.
        """
        try:
            img = bgr_to_rgb(frame)
            img = cv2.resize(img, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
            img = normalize_imagenet(img)                         # float32, ImageNet norm
            img = np.transpose(img, (2, 0, 1))[np.newaxis]       # HWC → NCHW
            return np.ascontiguousarray(img, dtype=np.float32)

        except Exception as e:
            raise PreprocessError(
                "RF-DETR preprocessing failed",
                context={"frame_shape": frame.shape, "error": str(e)},
            ) from e

    # ── Output routing ────────────────────────────────────────────────────────

    def _split_outputs(
        self, outputs: list
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (boxes, scores) with batch dimension removed.

        Tries name-based index matching first (reliable for Roboflow exports),
        then falls back to shape-based detection (last dim == 4 → boxes).

        Args:
            outputs: Raw list from session.run().

        Returns:
            boxes:  (N, 4) — normalized [0,1] XYXY.
            scores: (N, num_classes) — logits or probabilities.

        Raises:
            InferenceError: If outputs cannot be identified.
        """
        if self._boxes_idx is not None and self._labels_idx is not None:
            return (
                outputs[self._boxes_idx][0],
                outputs[self._labels_idx][0],
            )

        if len(outputs) == 2:
            a, b = outputs[0][0], outputs[1][0]
            return (a, b) if a.shape[-1] == 4 else (b, a)

        raise InferenceError(
            f"Unexpected ONNX output count: {len(outputs)} (expected 2)",
            context={"shapes": [list(o.shape) for o in outputs]},
        )

    # ── Postprocessing ────────────────────────────────────────────────────────

    def postprocess(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        original_shape: Tuple[int, int],
    ) -> sv.Detections:
        """
        Convert raw ONNX outputs to sv.Detections.

        RF-DETR exports boxes as normalized [0,1] XYXY relative to the
        *original* image — the transformer decoder handles coordinate mapping
        internally.  Multiply by [img_w, img_h, img_w, img_h] to get pixels.
        This matches Roboflow's official PostProcess exactly:
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
            boxes = boxes * scale_fct

        Args:
            boxes:          (N, 4) normalized [0,1] XYXY.
            scores:         (N, num_classes) logits or probabilities.
            original_shape: (H, W) of the frame passed to predict().

        Returns:
            sv.Detections in original-image pixel coordinates.

        Raises:
            PostprocessError: If decoding fails.
        """
        try:
            img_h, img_w = original_shape

            # Sigmoid if model outputs raw logits (min < 0 indicates logits)
            if np.min(scores) < 0:
                scores = 1.0 / (1.0 + np.exp(-scores))

            # Per-query best class and confidence
            max_scores = np.max(scores, axis=1)     # (N,)
            class_ids  = np.argmax(scores, axis=1)  # (N,)

            # Confidence filter — same mask applied to ALL three arrays
            mask = max_scores > self.confidence_threshold
            if not np.any(mask):
                return sv.Detections.empty()

            boxes      = boxes[mask]
            max_scores = max_scores[mask]
            class_ids  = class_ids[mask]

            # Scale normalized XYXY → absolute pixel coordinates
            scale    = np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
            boxes_px = (boxes * scale).astype(np.float32)

            # Clip to frame boundaries
            boxes_px[:, 0] = np.clip(boxes_px[:, 0], 0.0, img_w)  # x1
            boxes_px[:, 1] = np.clip(boxes_px[:, 1], 0.0, img_h)  # y1
            boxes_px[:, 2] = np.clip(boxes_px[:, 2], 0.0, img_w)  # x2
            boxes_px[:, 3] = np.clip(boxes_px[:, 3], 0.0, img_h)  # y2

            return sv.Detections(
                xyxy=boxes_px,
                confidence=max_scores,
                class_id=class_ids.astype(int),
            )

        except PostprocessError:
            raise
        except Exception as e:
            raise PostprocessError(
                "RF-DETR postprocessing failed",
                context={
                    "boxes_shape": list(boxes.shape),
                    "scores_shape": list(scores.shape),
                    "original_shape": original_shape,
                    "error": str(e),
                },
            ) from e

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, frame: np.ndarray) -> sv.Detections:
        """
        Run RF-DETR inference on a single BGR frame.

        Args:
            frame: BGR uint8 image (any resolution).

        Returns:
            sv.Detections in original-image pixel coordinates.

        Raises:
            InferenceError: If inference or output parsing fails.
        """
        try:
            tensor       = self.preprocess(frame)
            outputs      = self.session.run(None, {self.input_name: tensor})
            boxes, scores = self._split_outputs(outputs)
            detections   = self.postprocess(boxes, scores, frame.shape[:2])

            logger.debug(
                f"RF-DETR | detections={len(detections)} "
                f"| conf>{self.confidence_threshold}"
            )
            return detections

        except (PreprocessError, PostprocessError, InferenceError):
            raise
        except Exception as e:
            raise InferenceError(
                "RF-DETR inference failed",
                context={"frame_shape": list(frame.shape), "error": str(e)},
            ) from e
