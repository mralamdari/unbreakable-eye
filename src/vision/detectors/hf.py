"""
HF Transformer object detector using ONNX Runtime.

Model family: onnx-community/*-ONNX  (Transformers.js / Optimum export)
Best for: High-accuracy indoor/retail object detection on CPU.

Preprocessing facts (from preprocessor_config.json — ustc-community/*-coco):
  - Resize:     640×640, BILINEAR, plain square resize (no letterbox)
  - Rescale:    pixel / 255.0  ONLY
  - Normalize:  NONE — do_normalize=false in RTDetrImageProcessor config
                The image_mean/std fields are present but never applied.
                Applying ImageNet norm shifts every pixel ~0.45 outside
                the training distribution → near-zero detections.
  - Layout:     RGB, NCHW, float32

Output format (native HF export — NOT Roboflow convention):
  - pred_boxes: (1, N, 4) — cx,cy,w,h normalized [0,1] relative to input canvas
  - logits:     (1, N, num_classes) — raw class logits, decoded with sigmoid
"""

import os
import numpy as np
import cv2
import supervision as sv
from loguru import logger
from typing import List, Tuple
from src.vision.base import BaseDetector
from src.vision.utils import create_session, bgr_to_rgb
from src.core.exceptions import ModelLoadError, InferenceError, PreprocessError, PostprocessError

class HFTransformerDetector(BaseDetector):
    """
    Transformer detector using ONNX Runtime backend.

    Accepts any onnx-community/*-ONNX model file directly —
    no torch or transformers dependency at inference time.
    """

    def __init__(
        self,
        model_path: str,
        model_name: str,
        input_dim: tuple[int, int],
        conf_thresh: float = 0.4,
        device: str = "cpu",
    ):
        """
        Args:
            model_path:  Path to the .onnx file.
            conf_thresh: Confidence threshold (0.3–0.5 recommended).
            device:      Device hint ("cpu" / "cuda") — informational only,
                         actual provider selection is handled by create_session.

        Raises:
            ModelLoadError: If the ONNX session cannot be created or the
                             graph shape/output count is unexpected.
        """
        self.device = device.lower()
        self.model_name = model_name
        self.confidence_thres = conf_thresh

        try:
            num_threads = max(1, (os.cpu_count() or 4) // 2 - 1)
            logger.info(
                f"Loading Transformer ONNX | path={model_path} | threads={num_threads}"
            )
            self.session    = create_session(model_path, num_threads=num_threads)
            self.input_name = self.session.get_inputs()[0].name

            if len(self.session.get_outputs()) != 2:
                raise ModelLoadError(
                    "Transformer ONNX: expected exactly 2 outputs (pred_boxes + logits)",
                    context={
                        "model_path": model_path,
                        "outputs":    [o.name for o in self.session.get_outputs()],
                    },
                )

            # Read input (H, W) from the ONNX graph — never hardcode.
            # onnxruntime reports dynamic/symbolic dims as strings (e.g.
            # "height") rather than ints — detect and fall back safely.
            self.input_h, self.input_w = input_dim[0], input_dim[1]
            # self.input_h, self.input_w = settings.HF_INPUT_SIZE, settings.HF_INPUT_SIZE

            logger.info(
                f"Transformer ready | input={self.input_h}×{self.input_w} "
                f"| outputs={[o.name for o in self.session.get_outputs()]} "
                f"| conf_thresh={conf_thresh}"
            )

        except ModelLoadError:
            raise
        except Exception as e:
            logger.error(f"Failed to initialise Transformer ONNX detector: {e}")
            raise ModelLoadError(
                "Failed to initialise Transformer ONNX detector",
                context={"model_path": model_path, "error": str(e)},
            ) from e

    # ── Preprocessing ─────────────────────────────────────────────────────

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Prepare a BGR frame for Transformer ONNX inference.

        Pipeline (matches AutoImageProcessor("ustc-community/*") exactly,
        confirmed from preprocessor_config.json):
          BGR → RGB → plain square resize (no letterbox) →
          /255.0 ONLY (do_normalize=false) → NCHW float32

        Args:
            image: BGR uint8 image, any resolution.

        Returns:
            (1, 3, input_h, input_w) float32 contiguous array.

        Raises:
            PreprocessError: If any step fails.
        """
        try:
            img = bgr_to_rgb(image)

            # Plain square resize — no letterbox (Transformer was trained this way).
            img = cv2.resize(
                img, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR
            )

            # Rescale by 1/255 ONLY — do_normalize=false means NO mean/std.
            img = img.astype(np.float32) / 255.0
            if self.model_name == 'rfdetr':
                _IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                _IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
                        

            # HWC → NCHW, add batch dim
            img = np.transpose(img, (2, 0, 1))[np.newaxis]
            return np.ascontiguousarray(img, dtype=np.float32)

        except Exception as e:
            logger.error(f"Transformer preprocessing failed: {e}")
            raise PreprocessError(
                "Transformer preprocessing failed",
                context={"image_shape": image.shape, "error": str(e)},
            ) from e

    # ── Output routing ────────────────────────────────────────────────────

    def _split_outputs(self, outputs: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identify (boxes, logits) tensors and strip the batch dimension.

        Uses shape-based detection on still-batched tensors — unambiguous
        regardless of export naming conventions:
            out.shape[-1] == 4  →  boxes tensor  (pred_boxes)
            otherwise           →  logits tensor

        Args:
            outputs: Raw list from session.run(),
                     shapes (1, N, 4) and (1, N, num_classes).

        Returns:
            boxes:  (N, 4)  cxcywh normalized [0,1], batch dim removed.
            logits: (N, C)  raw class logits,         batch dim removed.

        Raises:
            InferenceError: If neither tensor has last-dim == 4.
        """
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
        Decode raw Transformer ONNX outputs into sv.Detections.

        Steps:
          1. sigmoid on logits → per-class probabilities
          2. argmax per query  → class_id + confidence score
          3. confidence filter → self.confidence_thres
          4. cxcywh → xyxy:  x cols × orig_w,  y cols × orig_h  → pixels
          5. clip to original image boundaries

        Args:
            boxes:          (N, 4) cxcywh normalized [0,1], batch dim removed.
            logits:         (N, num_classes) raw logits, batch dim removed.
            original_shape: (H, W) of the frame passed to predict().

        Returns:
            sv.Detections in original-image pixel coordinates.

        Raises:
            PostprocessError: If any decode step fails.
        """
        try:
            img_h, img_w = original_shape

            # Step 1-2: sigmoid (unconditional — never gate on np.min) → argmax
            probs     = 1.0 / (1.0 + np.exp(-logits))              # (N, C)
            class_ids = np.argmax(probs, axis=1)                    # (N,)
            scores    = probs[np.arange(len(class_ids)), class_ids] # (N,)

            # Step 3: confidence filter
            keep = scores > self.confidence_thres
            if not np.any(keep):
                return sv.Detections.empty()

            scores    = scores[keep]
            class_ids = class_ids[keep]
            boxes     = boxes[keep]

            # Step 4: cxcywh → xyxy, scale x/y columns independently
            cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes_px = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

            # Step 5: clip to frame boundaries
            boxes_px[:, [0, 2]] = np.clip(boxes_px[:, [0, 2]], 0.0, img_w)
            boxes_px[:, [1, 3]] = np.clip(boxes_px[:, [1, 3]], 0.0, img_h)

            logger.debug(
                f"Transformer postprocess | kept={len(scores)} / total={len(probs)} "
                f"| conf_thresh={self.confidence_thres}"
            )

            return sv.Detections(
                xyxy=boxes_px.reshape(-1, 4),
                confidence=scores.astype(np.float32),
                class_id=class_ids.astype(int),
            )

        except PostprocessError:
            raise
        except Exception as e:
            logger.error(f"Transformer postprocessing failed: {e}")
            raise PostprocessError(
                "Transformer postprocessing failed",
                context={
                    "boxes_shape":    list(boxes.shape),
                    "logits_shape":   list(logits.shape),
                    "original_shape": original_shape,
                    "error":          str(e),
                },
            ) from e

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, image: np.ndarray) -> sv.Detections:
        """
        Run Transformer inference on a single BGR frame.

        Args:
            image: BGR uint8 image, any resolution.

        Returns:
            sv.Detections in original-image pixel coordinates.

        Raises:
            PreprocessError / InferenceError / PostprocessError
        """
        try:
            blob          = self.preprocess(image)
            outputs       = self.session.run(None, {self.input_name: blob})
            boxes, logits = self._split_outputs(outputs)
            detections    = self.postprocess(boxes, logits, image.shape[:2])

            logger.debug(f"Transformer predict | detections={len(detections)}")
            return detections

        except (PreprocessError, PostprocessError, InferenceError):
            raise
        except Exception as e:
            logger.error(f"Transformer inference failed: {e}")
            raise InferenceError(
                "Transformer inference failed",
                context={"image_shape": list(image.shape), "error": str(e)},
            ) from e

    def predict_batch(self, frames: List[np.ndarray]) -> List[sv.Detections]:
        """
        Run Transformer inference on a batch of BGR frames in a single forward pass.

        Each frame is preprocessed independently (may differ in original
        resolution), stacked into one batch tensor for a single session.run(),
        then postprocessed back per-frame with its own original shape.

        Args:
            frames: List of BGR uint8 images, any resolution (may differ).

        Returns:
            List of sv.Detections, one per input frame, same order.

        Raises:
            PreprocessError / InferenceError / PostprocessError
        """
        if not frames:
            return []

        try:
            blobs        = []
            orig_shapes  = []

            for frame in frames:
                blobs.append(self.preprocess(frame))        # each (1,3,H,W)
                orig_shapes.append(frame.shape[:2])

            # Stack into batch (B, 3, H, W) — preprocess already adds batch dim,
            # so concatenate on axis 0 rather than np.stack.
            batch = np.concatenate(blobs, axis=0)

            outputs_raw = self.session.run(None, {self.input_name: batch})

            # outputs_raw[0]: (B, N, 4) boxes
            # outputs_raw[1]: (B, N, C) logits
            # _split_outputs works on a 2-element list with shape[-1]==4 detection,
            # but returns only one frame's slice — handle batch manually here.
            out0, out1 = outputs_raw[0], outputs_raw[1]
            if out0.shape[-1] == 4:
                all_boxes, all_logits = out0, out1
            else:
                all_logits, all_boxes = out0, out1

            detections_list = [
                self.postprocess(all_boxes[i], all_logits[i], orig_shapes[i])
                for i in range(len(frames))
            ]

            total = sum(len(d) for d in detections_list)
            logger.debug(
                f"Transformer batch predict | frames={len(frames)} | total_detections={total}"
            )
            return detections_list

        except (PreprocessError, PostprocessError, InferenceError):
            raise
        except Exception as e:
            logger.error(f"Transformer batch inference failed: {e}")
            raise InferenceError(
                "Transformer batch inference failed",
                context={"num_frames": len(frames), "error": str(e)},
            ) from e