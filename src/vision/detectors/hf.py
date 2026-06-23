"""
RF-DETR / D-FINE detector using ONNX Runtime.

──────────────────────────────────────────────────────────────────────────
These two model families share the same DETR architecture and sigmoid
scoring but differ in THREE ways that ALL must be correct simultaneously:
──────────────────────────────────────────────────────────────────────────

  RF-DETR  (onnx-community/rfdetr_*-ONNX, exported by Roboflow):
    Preprocessing : BGR→RGB → resize to input_h×input_w → /255 →
                   SUBTRACT ImageNet mean, DIVIDE ImageNet std → NCHW
                   (do_normalize=true)
    Box format    : already XYXY, normalized [0,1] relative to ORIGINAL
                   image. Roboflow bakes cxcywh→xyxy INSIDE the graph.
    Decode        : boxes * [img_w, img_h, img_w, img_h] → pixel coords.

  D-FINE   (onnx-community/dfine_*-ONNX, exported by Transformers.js):
    Preprocessing : BGR→RGB → resize to 640×640 → /255 ONLY — NO
                   mean/std subtraction (do_normalize=false confirmed in
                   ustc-community/dfine-*-coco preprocessor_config.json).
                   Applying ImageNet norm here shifts every pixel ~0.45
                   outside the training distribution → near-zero detections.
    Box format    : CXCYWH, normalized [0,1] relative to input canvas.
                   post_process_object_detection runs OUTSIDE the graph.
    Decode        : cxcywh→xyxy (normalized) →
                   x cols * img_w, y cols * img_h → pixel coords.

Family is auto-detected from output tensor names:
  "pred_boxes" in name → D-FINE  (cxcywh_input_normalized)
  "dets" / bare "boxes" → RF-DETR (xyxy_original_normalized)
"""

import os
import cv2
import numpy as np
import supervision as sv
from loguru import logger
from typing import Tuple

from src.vision.base import BaseDetector
from src.vision.utils import create_session, bgr_to_rgb, normalize_imagenet, cxcywh_to_xyxy
from src.core.exceptions import ModelLoadError, InferenceError, PreprocessError, PostprocessError

_RFDETR_DEFAULT_SIZE = 560   # 560 / 14 = 40 — valid DINOv2 patch-divisible size
_DFINE_DEFAULT_SIZE  = 640   # D-FINE fixed training/export resolution


class HFTransformerDetector(BaseDetector):
    """RF-DETR / D-FINE detector via ONNX Runtime.

    Auto-detects which family a given .onnx file belongs to from output
    tensor names, then applies the correct preprocessing AND postprocessing
    for that family. Using the wrong preprocessing OR the wrong decode is
    individually enough to produce garbage — both must match simultaneously.
    """

    def __init__(
        self,
        model_path: str,
        conf_thresh: float = 0.45,
        device: str = "cpu",
    ):
        self.confidence_threshold = conf_thresh

        try:
            num_threads = max(1, (os.cpu_count() or 4) // 2 - 1)
            logger.info(
                f"Loading HF-transformer ONNX | path={model_path} | threads={num_threads}"
            )
            self.session    = create_session(model_path, num_threads=num_threads)
            self.input_name = self.session.get_inputs()[0].name

            if len(self.session.get_outputs()) != 2:
                raise ModelLoadError(
                    "Expected exactly 2 ONNX outputs (boxes + class scores)",
                    context={
                        "model_path": model_path,
                        "outputs": [o.name for o in self.session.get_outputs()],
                    },
                )

            # ── Family detection via output tensor names ──────────────────
            # "pred_boxes" is the native HF/Transformers.js export name for
            # D-FINE; "dets"/"boxes" is Roboflow's RF-DETR export name.
            # These are the actual, distinct conventions each exporter uses —
            # not guesses.
            boxes_name = None
            for out in self.session.get_outputs():
                name = out.name.lower()
                if "box" in name or "det" in name:
                    boxes_name = name
                    break

            if boxes_name and "pred_box" in boxes_name:
                self.box_format  = "cxcywh_input_normalized"   # D-FINE
                default_size     = _DFINE_DEFAULT_SIZE
            else:
                self.box_format  = "xyxy_original_normalized"  # RF-DETR
                default_size     = _RFDETR_DEFAULT_SIZE

            # ── Input size — read from graph, never hardcode ──────────────
            shape = self.session.get_inputs()[0].shape   # [1, 3, H, W]
            self.input_h, self.input_w = self._resolve_input_size(shape, default_size)

            output_names = [o.name for o in self.session.get_outputs()]
            logger.info(
                f"HF-transformer ready | family={self.box_format} "
                f"| input={self.input_h}×{self.input_w} "
                f"| outputs={output_names} | conf={conf_thresh}"
            )

        except ModelLoadError:
            raise
        except Exception as e:
            raise ModelLoadError(
                "Failed to initialise HF-transformer ONNX detector",
                context={"model_path": model_path, "error": str(e)},
            ) from e

    # ── Input-size resolution ─────────────────────────────────────────────

    def _resolve_input_size(self, shape: list, default_size: int) -> Tuple[int, int]:
        """
        Read (H, W) from the ONNX graph's declared input shape [1, 3, H, W].

        onnxruntime reports dynamic/symbolic dims as strings (e.g. "height")
        rather than ints — this detects that and falls back to the
        family-appropriate default instead of crashing or silently producing 0.
        """
        if len(shape) != 4:
            raise ModelLoadError(
                "Unexpected input rank", context={"shape": shape, "expected_rank": 4}
            )
        raw_h, raw_w = shape[2], shape[3]
        h = raw_h if isinstance(raw_h, int) and raw_h > 0 else None
        w = raw_w if isinstance(raw_w, int) and raw_w > 0 else None

        if h is None or w is None:
            logger.warning(
                f"ONNX graph has dynamic/symbolic input dims (H={raw_h}, W={raw_w}). "
                f"Falling back to {default_size}×{default_size} "
                f"for family={self.box_format}. Re-export with a fixed shape to avoid this."
            )
            h = h or default_size
            w = w or default_size

        return h, w

    # ── Preprocessing ─────────────────────────────────────────────────────

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        BGR frame → (1, 3, H, W) float32 blob.

        Shared: BGR→RGB, plain square resize (no letterbox), NCHW layout.
        Differs per family in the normalization step only:

          RF-DETR → /255 then ImageNet mean/std  (do_normalize=true)
          D-FINE  → /255 ONLY                    (do_normalize=false)

        Raises:
            PreprocessError
        """
        try:
            img = bgr_to_rgb(frame)
            img = cv2.resize(
                img, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR
            )

            if self.box_format == "cxcywh_input_normalized":
                # D-FINE: rescale by 1/255 ONLY.
                # Confirmed from preprocessor_config.json:
                #   "do_normalize": false, "rescale_factor": 0.00392156862745098
                img = img.astype(np.float32) / 255.0
            else:
                # RF-DETR: /255 + ImageNet mean/std subtraction.
                img = normalize_imagenet(img)

            img = np.transpose(img, (2, 0, 1))[np.newaxis]
            return np.ascontiguousarray(img, dtype=np.float32)

        except Exception as e:
            raise PreprocessError(
                "HF-transformer preprocessing failed",
                context={"frame_shape": frame.shape, "error": str(e)},
            ) from e

    # ── Output routing ────────────────────────────────────────────────────

    def _split_outputs(self, outputs: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identify (boxes, logits) from session.run() output list and strip
        the batch dimension.

        Uses shape-based detection on still-batched tensors — the only
        unambiguous signal regardless of export naming conventions:
            out.shape[-1] == 4  →  boxes tensor
        Returns (boxes, logits) both with batch dim removed → (N,4), (N,C).

        Raises:
            InferenceError: If output count ≠ 2 or no tensor has last-dim 4.
        """
        if len(outputs) != 2:
            raise InferenceError(
                f"Expected exactly 2 ONNX outputs, got {len(outputs)}",
                context={"shapes": [list(o.shape) for o in outputs]},
            )

        out0, out1 = outputs[0], outputs[1]

        if out0.shape[-1] == 4:
            boxes, logits = out0, out1
        elif out1.shape[-1] == 4:
            logits, boxes = out0, out1
        else:
            raise InferenceError(
                "Cannot identify boxes tensor: neither output has last-dim == 4",
                context={
                    "out0_shape": list(out0.shape),
                    "out1_shape": list(out1.shape),
                },
            )

        return boxes[0], logits[0]   # strip batch → (N,4), (N,C)

    # ── Postprocessing ────────────────────────────────────────────────────

    def postprocess(
        self,
        boxes: np.ndarray,
        logits: np.ndarray,
        original_shape: Tuple[int, int],
    ) -> sv.Detections:
        """
        Decode raw ONNX outputs into sv.Detections in original-image pixels.

        Both families:
          1. sigmoid on logits → probs                    (always unconditional)
          2. argmax per query  → class_id + confidence
          3. confidence filter → self.confidence_threshold

        Then diverge:
          D-FINE  (cxcywh_input_normalized):
            4. cxcywh → xyxy  (still normalized [0,1])
            5. x cols × img_w,  y cols × img_h  → pixels

          RF-DETR (xyxy_original_normalized):
            4. boxes already xyxy, normalized to original image
            5. × [img_w, img_h, img_w, img_h]  → pixels

        NOTE: sigmoid is applied UNCONDITIONALLY — not gated on
        `np.min(logits) < 0`. If all logits happen to be positive (e.g.
        very confident frame), the gated version skips sigmoid entirely
        and compares raw logits against a 0-1 threshold, which either
        passes everything or nothing. Always sigmoid.

        Raises:
            PostprocessError
        """
        try:
            img_h, img_w = original_shape

            # Step 1-2: sigmoid → best class per query
            probs     = 1.0 / (1.0 + np.exp(-logits))          # (N, C)
            class_ids = np.argmax(probs, axis=1)                 # (N,)
            scores    = probs[np.arange(len(class_ids)), class_ids]  # (N,)

            # Step 3: confidence filter
            # CRITICAL: use self.confidence_threshold, NOT a bare `conf_thresh`
            # variable. A bare name resolves from the enclosing scope at runtime
            # and silently produces threshold=0, passing all 300 queries through
            # and returning random garbage class IDs as detections.
            keep = scores > self.confidence_threshold
            if not np.any(keep):
                return sv.Detections.empty()

            scores    = scores[keep]
            class_ids = class_ids[keep]
            boxes     = boxes[keep]

            # Step 4-5: box decode, family-specific
            if self.box_format == "cxcywh_input_normalized":
                # D-FINE path: cxcywh → xyxy → scale x/y independently
                cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                x1 = (cx - w / 2) * img_w
                y1 = (cy - h / 2) * img_h
                x2 = (cx + w / 2) * img_w
                y2 = (cy + h / 2) * img_h
                boxes_px = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

            else:
                # RF-DETR path: already xyxy, normalized to original image
                scale    = np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
                boxes_px = (boxes * scale).astype(np.float32)

            # Clip to frame boundaries
            boxes_px[:, [0, 2]] = np.clip(boxes_px[:, [0, 2]], 0.0, img_w)
            boxes_px[:, [1, 3]] = np.clip(boxes_px[:, [1, 3]], 0.0, img_h)

            return sv.Detections(
                xyxy=boxes_px.reshape(-1, 4),
                confidence=scores.astype(np.float32),
                class_id=class_ids.astype(int),
            )

        except PostprocessError:
            raise
        except Exception as e:
            raise PostprocessError(
                "HF-transformer postprocessing failed",
                context={
                    "boxes_shape":    list(boxes.shape),
                    "logits_shape":   list(logits.shape),
                    "original_shape": original_shape,
                    "box_format":     self.box_format,
                    "error":          str(e),
                },
            ) from e

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, frame: np.ndarray) -> sv.Detections:
        """
        Run inference on a single BGR frame.

        Args:
            frame: BGR uint8 image, any resolution.

        Returns:
            sv.Detections in original-image pixel coordinates.

        Raises:
            PreprocessError / InferenceError / PostprocessError
        """
        try:
            tensor        = self.preprocess(frame)
            outputs       = self.session.run(None, {self.input_name: tensor})
            boxes, logits = self._split_outputs(outputs)
            detections    = self.postprocess(boxes, logits, frame.shape[:2])

            logger.debug(
                f"{self.box_format} | detections={len(detections)} "
                f"| conf>{self.confidence_threshold}"
            )
            return detections

        except (PreprocessError, PostprocessError, InferenceError):
            raise
        except Exception as e:
            raise InferenceError(
                "HF-transformer inference failed",
                context={"frame_shape": list(frame.shape), "error": str(e)},
            ) from e
