"""
RF-DETR / D-FINE detector using ONNX Runtime.

Preprocessing and postprocessing follow Roboflow's official documentation exactly:
https://rfdetr.roboflow.com/develop/learn/export/#onnx-runtime

Key facts about RF-DETR ONNX exports:
  - Input:  (1, 3, H, W) float32, ImageNet-normalized, RGB channel order
  - Resize: plain cv2.resize to (input_w, input_h) — NO letterboxing
  - Output: two tensors named "boxes" and "labels"
      boxes:  (1, N, 4) — normalized [0,1] XYXY relative to the ORIGINAL image
      labels: (1, N, num_classes) — class logits or probabilities
  - Postprocess: multiply boxes by [img_w, img_h, img_w, img_h] to get pixels
"""
# from src.core.config import settings
import os
import cv2
import numpy as np
import supervision as sv
from loguru import logger
from typing import Tuple
from src.core.config import settings
from src.vision.base import BaseDetector
from src.vision.utils import create_session
from src.core.exceptions import ModelLoadError, InferenceError, PreprocessError, PostprocessError

# ImageNet normalization — required by DINOv2 backbone
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
def cxcywh_to_xyxy(boxes_norm: np.ndarray) -> np.ndarray:
    """
    Convert [cx, cy, w, h] in [0,1] → [x1, y1, x2, y2] in [0,1].
    RF-DETR (like RT-DETR) always outputs normalized cx/cy/w/h.
    """
    cx, cy, w, h = boxes_norm[:, 0], boxes_norm[:, 1], boxes_norm[:, 2], boxes_norm[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)

def nms(boxes, scores, class_ids, iou_thresh=0.4):
    """Simple class-agnostic NMS."""
    if len(boxes) == 0:
        return boxes, scores, class_ids

    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thresh]

    keep = np.array(keep)
    return boxes[keep], scores[keep], class_ids[keep]

class HFTransformerDetector(BaseDetector):
    """
    RF-DETR / D-FINE detector via ONNX Runtime.

    Use scripts/export_rfdetr.py to obtain the .onnx file from pretrained
    COCO weights before running this detector.
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
            logger.info(f"Loading RF-DETR ONNX model: {model_path} ({num_threads} threads)")
            self.session = create_session(model_path, num_threads=num_threads)

            # Read input shape from ONNX graph — never hardcode
            inp = self.session.get_inputs()[0]
            shape = inp.shape  # [1, 3, H, W]
            # self.input_h = 512
            # self.input_w = 512
            self.input_h = 640
            self.input_w = 640
            
            # self.input_h = int(shape[2]) if isinstance(shape[2], int) else 560
            # self.input_w = int(shape[3]) if isinstance(shape[3], int) else 560
            self.input_name = inp.name
            # self.input_size = settings.FRAME_SHAPE
            # Map output names — Roboflow exports use "boxes" and "labels"
            self._boxes_idx  = None
            self._labels_idx = None
            for i, o in enumerate(self.session.get_outputs()):
                name = o.name.lower()
                if "box" in name:
                    self._boxes_idx = i
                elif "label" in name or "score" in name or "logit" in name:
                    self._labels_idx = i

            output_names = [o.name for o in self.session.get_outputs()]
            logger.info(f"RF-DETR initialized: input {self.input_h}x{self.input_w}, "
                       f"outputs={output_names}, conf={conf_thresh}")

        except Exception as e:
            raise ModelLoadError(
                "Failed to load RF-DETR ONNX model",
                context={"model_path": model_path, "error": str(e)}
            ) from e

    # def preprocess(self, frame: np.ndarray) -> np.ndarray:
    #     """
    #     Preprocess BGR frame for RF-DETR ONNX inference.

    #     Follows Roboflow official example:
    #       BGR -> RGB -> resize (no letterbox) -> /255 -> ImageNet norm -> NCHW
    #     """
    #     try:
    #         img_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         img_resized = cv2.resize(img_rgb, (self.input_w, self.input_h),
    #                                  interpolation=cv2.INTER_LINEAR)
    #         img_f32     = img_resized.astype(np.float32) / 255.0
    #         img_norm    = (img_f32 - _IMAGENET_MEAN) / _IMAGENET_STD
    #         img_nchw    = np.transpose(img_norm, (2, 0, 1))[np.newaxis]
    #         return np.ascontiguousarray(img_nchw, dtype=np.float32)

    #     except Exception as e:
    #         raise PreprocessError(
    #             "RF-DETR preprocessing failed",
    #             context={"frame_shape": frame.shape, "error": str(e)}
    #         ) from e
    
    
    
    def preprocess(self, frame, orig_h, orig_w) -> np.ndarray:
        """
        Preprocess BGR frame for RF-DETR ONNX inference.

        Follows Roboflow official example:
          BGR -> RGB -> resize (no letterbox) -> /255 -> ImageNet norm -> NCHW
        """
        try:
            scale  = self.input_w / max(orig_h, orig_w)
            new_w  = int(round(orig_w * scale))
            new_h  = int(round(orig_h * scale))
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            canvas = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
            pad_top  = (self.input_h - new_h) // 2
            pad_left = (self.input_w - new_w) // 2
            canvas[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
            rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
            rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
            blob = rgb.transpose(2, 0, 1)[np.newaxis, ...]
            return blob, scale, pad_top, pad_left
        except Exception as e:
            raise PreprocessError(
                "RF-DETR preprocessing failed",
                context={"frame_shape": frame.shape, "error": str(e)}
            ) from e
            
    def postprocess(
        self,
        outputs, 
        orig_h, orig_w, 
        scale,
        pad_top, 
        pad_left,
    ) -> sv.Detections:
        
        # try:
        """
        Decode raw ONNX outputs into (boxes_pixel, scores, class_ids).

        The model returns two tensors:
        logits  [1, num_queries, num_classes]  – raw class logits
        boxes   [1, num_queries, 4]            – cx,cy,w,h in [0,1]
        """
        # ── locate the two output tensors ────────────────────────────────────────
        # Different exports may swap the order, so we identify by shape
        o0, o1 = outputs[0], outputs[1]  # shapes like (1,300,91) and (1,300,4)
                
        if o0.shape[-1] == 4:
            raw_boxes, raw_logits = o0, o1
        else:
            raw_logits, raw_boxes = o0, o1

        raw_logits = raw_logits[0]   # (num_queries, num_classes)
        raw_boxes  = raw_boxes[0]    # (num_queries, 4)

        # Sigmoid → probabilities (RF-DETR uses sigmoid, not softmax)
        probs = 1.0 / (1.0 + np.exp(-raw_logits))  # (Q, C)

        # Best class per query
        class_ids  = np.argmax(probs, axis=1)       # (Q,)
        scores     = probs[np.arange(len(class_ids)), class_ids]  # (Q,)

        # Filter by threshold
        keep = scores > settings.CONF_THRESHOLD
        if not keep.any():
            return [], [], []

        scores     = scores[keep]
        class_ids  = class_ids[keep]
        boxes_norm = raw_boxes[keep]

        # cx,cy,w,h → x1,y1,x2,y2  (still normalized 0-1 relative to 512×512 canvas)
        boxes_xyxy = cxcywh_to_xyxy(boxes_norm)
        # Unpad: remove letterbox offset and undo scale to get original pixel coords
        # canvas coords (0-1) → pixel in 512×512 canvas
        boxes_px = boxes_xyxy * self.input_w

        # Remove padding
        boxes_px[:, [0, 2]] -= pad_left
        boxes_px[:, [1, 3]] -= pad_top

        # Rescale to original image size
        boxes_px /= scale

        # Clip to image boundaries
        boxes_px[:, [0, 2]] = np.clip(boxes_px[:, [0, 2]], 0, orig_w)
        boxes_px[:, [1, 3]] = np.clip(boxes_px[:, [1, 3]], 0, orig_h)

        # NMS
        boxes_px, scores, class_ids = nms(boxes_px, scores, class_ids)
        return sv.Detections(
            xyxy=boxes_px.astype(np.float32),
            confidence=scores,
            class_id=class_ids.astype(int),
        )

        # except Exception as e:
        #     raise PostprocessError(
        #         "RF-DETR postprocessing failed",
        #         context={"boxes_shape": raw_boxes.shape, "scores_shape": scores.shape, "error": str(e)}
        #     ) from e



    # def postprocess(
    #     self,
    #     boxes: np.ndarray,
    #     scores: np.ndarray,
    #     original_shape: Tuple[int, int],
    # ) -> sv.Detections:
    #     """
    #     Convert RF-DETR ONNX outputs to sv.Detections.

    #     boxes are normalized [0,1] XYXY in ORIGINAL image space.
    #     Scale: boxes * [img_w, img_h, img_w, img_h]
    #     This matches Roboflow's official PostProcess exactly.
    #     """
    #     try:
    #         img_h, img_w = original_shape

    #         if np.min(scores) < 0:         # logits → probabilities
    #             scores = 1.0 / (1.0 + np.exp(-scores))

    #         max_scores = np.max(scores, axis=1)    # (N,)
    #         class_ids  = np.argmax(scores, axis=1) # (N,)

    #         mask = max_scores > self.confidence_threshold
    #         if not np.any(mask):
    #             return sv.Detections.empty()

    #         boxes      = boxes[mask]
    #         max_scores = max_scores[mask]
    #         class_ids  = class_ids[mask]   # same mask — arrays stay aligned

    #         scale    = np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
    #         boxes_px = boxes * scale

    #         boxes_px[:, 0] = np.clip(boxes_px[:, 0], 0.0, img_w)  # x1
    #         boxes_px[:, 1] = np.clip(boxes_px[:, 1], 0.0, img_h)  # y1
    #         boxes_px[:, 2] = np.clip(boxes_px[:, 2], 0.0, img_w)  # x2
    #         boxes_px[:, 3] = np.clip(boxes_px[:, 3], 0.0, img_h)  # y2

    #         return sv.Detections(
    #             xyxy=boxes_px.astype(np.float32),
    #             confidence=max_scores,
    #             class_id=class_ids.astype(int),
    #         )

    #     except Exception as e:
    #         raise PostprocessError(
    #             "RF-DETR postprocessing failed",
    #             context={"boxes_shape": boxes.shape, "scores_shape": scores.shape,
    #                      "original_shape": original_shape, "error": str(e)}
    #         ) from e

    # def _split_outputs(self, outputs: list) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     Extract (boxes, scores) from ONNX outputs.
    #     Uses name-based matching first, shape-based fallback second.
    #     """
    #     if self._boxes_idx is not None and self._labels_idx is not None:
    #         return outputs[self._boxes_idx][0], outputs[self._labels_idx][0]

    #     # Shape-based fallback: boxes have last dim == 4
    #     if len(outputs) == 2:
    #         out0, out1 = outputs[0][0], outputs[1][0]
    #         if out0.shape[-1] == 4:
    #             return out0, out1
    #         return out1, out0

    #     raise InferenceError(
    #         f"Expected 2 ONNX outputs (boxes + labels), got {len(outputs)}",
    #         context={"num_outputs": len(outputs),
    #                  "shapes": [o.shape for o in outputs]}
    #     )

    def predict(self, frame: np.ndarray) -> sv.Detections:
        
        orig_h, orig_w = frame.shape[:2]
        input_tensor, scale, pad_top, pad_left = self.preprocess(frame, orig_h, orig_w)
        # ── Inference ─────────────────────────────────────────────────────────────
        outputs = self.session.run(None, {self.input_name: input_tensor})
        # ── Decode outputs ────────────────────────────────────────────────────────
        detections = self.postprocess(
            outputs, orig_h, orig_w, scale, pad_top, pad_left)
        return detections
    
        # """Run RF-DETR on a single BGR frame."""
        # try:
        #     input_tensor = self.preprocess(frame)
        #     outputs      = self.session.run(None, {self.input_name: input_tensor})
        #     boxes, scores = self._split_outputs(outputs)
        #     detections    = self.postprocess(boxes, scores, frame.shape[:2])
        #     logger.debug(f"RF-DETR: {len(detections)} detections")
        #     return detections

        # except (PreprocessError, PostprocessError, InferenceError):
        #     raise
        # except Exception as e:
        #     raise InferenceError(
        #         "RF-DETR inference failed",
        #         context={"frame_shape": frame.shape, "error": str(e)}
        #     ) from e




































































































# """
# RF-DETR / D-FINE detector using ONNX Runtime.

# Key facts about these models' ONNX exports:
#   - Input:   (1, 3, H, W) float32, ImageNet-normalized, RGB channel order
#   - Resize:  plain cv2.resize — NO letterboxing (model handles coord mapping)
#   - Outputs: two tensors — boxes (N,4) normalized [0,1] XYXY + labels (N, num_classes)
#   - Decode:  boxes * [img_w, img_h, img_w, img_h] → absolute pixel coords

# Reference: https://rfdetr.roboflow.com/develop/learn/export/#onnx-runtime
# """

# import os
# import cv2
# import numpy as np
# import supervision as sv
# from loguru import logger
# from typing import Tuple

# from src.vision.base import BaseDetector
# from src.vision.utils import create_session, bgr_to_rgb, normalize_imagenet, nms_numpy, cxcywh_to_xyxy
# from src.core.exceptions import ModelLoadError, InferenceError, PreprocessError, PostprocessError


# class HFTransformerDetector(BaseDetector):
#     """RF-DETR / D-FINE detector via ONNX Runtime."""

#     def __init__(
#         self,
#         model_path: str,
#         conf_thresh: float = 0.45,
#         device: str = "cpu",
#     ):
#         """
#         Args:
#             model_path:  Path to the exported .onnx file.
#             conf_thresh: Confidence threshold (0.3–0.5 recommended for RF-DETR).
#             device:      "cpu" or "cuda" (informational — session auto-selects).

#         Raises:
#             ModelLoadError: If the ONNX session cannot be created.
#         """
#         self.confidence_threshold = conf_thresh

#         try:
#             num_threads = max(1, (os.cpu_count() or 4) // 2 - 1)
#             logger.info(f"Loading RF-DETR ONNX | path={model_path} | threads={num_threads}")
#             self.session    = create_session(model_path, num_threads=num_threads)
#             self.input_name = self.session.get_inputs()[0].name

#             # Read input resolution from the ONNX graph — never hardcode
#             shape        = self.session.get_inputs()[0].shape  # [1, 3, H, W]
#             # self.input_h = int(shape[2]) if isinstance(shape[2], int) else 560
#             # self.input_w = int(shape[3]) if isinstance(shape[3], int) else 560

#             self.input_h = 512
#             self.input_w = 512
#             # Identify output indices by name (boxes / labels)
#             self._boxes_idx  = None
#             self._labels_idx = None
#             for i, out in enumerate(self.session.get_outputs()):
#                 name = out.name.lower()
#                 if "box" in name:
#                     self._boxes_idx = i
#                 elif "label" in name or "score" in name or "logit" in name:
#                     self._labels_idx = i

#             output_names = [o.name for o in self.session.get_outputs()]
#             logger.info(
#                 f"RF-DETR ready | input={self.input_h}x{self.input_w} "
#                 f"| outputs={output_names} | conf={conf_thresh}"
#             )

#         except ModelLoadError:
#             raise
#         except Exception as e:
#             raise ModelLoadError(
#                 "Failed to initialise RF-DETR ONNX detector",
#                 context={"model_path": model_path, "error": str(e)},
#             ) from e

#     # ── Preprocessing ─────────────────────────────────────────────────────────

#     def preprocess(self, frame: np.ndarray) -> np.ndarray:
#         """
#         Prepare a BGR frame for RF-DETR ONNX inference.

#         Pipeline (matches Roboflow official example exactly):
#           BGR → RGB → plain resize (no letterbox) → /255 → ImageNet norm → NCHW

#         Args:
#             frame: BGR uint8 image (any resolution).

#         Returns:
#             (1, 3, input_h, input_w) float32 contiguous array.

#         Raises:
#             PreprocessError: If any step fails.
#         """
#         try:
#             img = bgr_to_rgb(frame)
#             img = cv2.resize(img, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
#             img = normalize_imagenet(img)                         # float32, ImageNet norm
#             img = np.transpose(img, (2, 0, 1))[np.newaxis]       # HWC → NCHW
#             return np.ascontiguousarray(img, dtype=np.float32)

#         except Exception as e:
#             raise PreprocessError(
#                 "RF-DETR preprocessing failed",
#                 context={"frame_shape": frame.shape, "error": str(e)},
#             ) from e

#     # ── Output routing ────────────────────────────────────────────────────────

#     def _split_outputs(
#         self, outputs: list
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Return (boxes, scores) with batch dimension removed.

#         Tries name-based index matching first (reliable for Roboflow exports),
#         then falls back to shape-based detection (last dim == 4 → boxes).

#         Args:
#             outputs: Raw list from session.run().

#         Returns:
#             boxes:  (N, 4) — normalized [0,1] XYXY.
#             scores: (N, num_classes) — logits or probabilities.

#         Raises:
#             InferenceError: If outputs cannot be identified.
#         """
#         if self._boxes_idx is not None and self._labels_idx is not None:
#             return (
#                 outputs[self._boxes_idx][0],
#                 outputs[self._labels_idx][0],
#             )

#         if len(outputs) == 2:
#             a, b = outputs[0][0], outputs[1][0]
#             return (a, b) if a.shape[-1] == 4 else (b, a)

#         raise InferenceError(
#             f"Unexpected ONNX output count: {len(outputs)} (expected 2)",
#             context={"shapes": [list(o.shape) for o in outputs]},
#         )

#     # ── Postprocessing ────────────────────────────────────────────────────────

#     def postprocess(
#         self,
#         boxes: np.ndarray,
#         scores: np.ndarray,
#         original_shape: Tuple[int, int],
#     ) -> sv.Detections:
#         """
#         Convert raw ONNX outputs to sv.Detections.

#         RF-DETR exports boxes as normalized [0,1] XYXY relative to the
#         *original* image — the transformer decoder handles coordinate mapping
#         internally.  Multiply by [img_w, img_h, img_w, img_h] to get pixels.
#         This matches Roboflow's official PostProcess exactly:
#             scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
#             boxes = boxes * scale_fct

#         Args:
#             boxes:          (N, 4) normalized [0,1] XYXY.
#             scores:         (N, num_classes) logits or probabilities.
#             original_shape: (H, W) of the frame passed to predict().

#         Returns:
#             sv.Detections in original-image pixel coordinates.

#         Raises:
#             PostprocessError: If decoding fails.
#         """
#         try:
#             img_h, img_w = original_shape

#             # Sigmoid if model outputs raw logits (min < 0 indicates logits)
#             if np.min(scores) < 0:
#                 scores = 1.0 / (1.0 + np.exp(-scores))

#             # Per-query best class and confidence
#             max_scores = np.max(scores, axis=1)     # (N,)
#             class_ids  = np.argmax(scores, axis=1)  # (N,)

#             # Confidence filter — same mask applied to ALL three arrays
#             mask = max_scores > self.confidence_threshold
#             if not np.any(mask):
#                 return sv.Detections.empty()

#             boxes      = boxes[mask]
#             max_scores = max_scores[mask]
#             class_ids  = class_ids[mask]

#             # Scale normalized XYXY → absolute pixel coordinates
#             scale    = np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
#             boxes_px = (boxes * scale).astype(np.float32)

#             # Clip to frame boundaries
#             boxes_px[:, 0] = np.clip(boxes_px[:, 0], 0.0, img_w)  # x1
#             boxes_px[:, 1] = np.clip(boxes_px[:, 1], 0.0, img_h)  # y1
#             boxes_px[:, 2] = np.clip(boxes_px[:, 2], 0.0, img_w)  # x2
#             boxes_px[:, 3] = np.clip(boxes_px[:, 3], 0.0, img_h)  # y2

#             return sv.Detections(
#                 xyxy=boxes_px,
#                 confidence=max_scores,
#                 class_id=class_ids.astype(int),
#             )

#         except PostprocessError:
#             raise
#         except Exception as e:
#             raise PostprocessError(
#                 "RF-DETR postprocessing failed",
#                 context={
#                     "boxes_shape": list(boxes.shape),
#                     "scores_shape": list(scores.shape),
#                     "original_shape": original_shape,
#                     "error": str(e),
#                 },
#             ) from e

#     # ── Inference ─────────────────────────────────────────────────────────────

#     def predict(self, frame: np.ndarray) -> sv.Detections:
#         """
#         Run RF-DETR inference on a single BGR frame.

#         Args:
#             frame: BGR uint8 image (any resolution).

#         Returns:
#             sv.Detections in original-image pixel coordinates.

#         Raises:
#             InferenceError: If inference or output parsing fails.
#         """
#         try:
#             tensor       = self.preprocess(frame)
#             outputs      = self.session.run(None, {self.input_name: tensor})
#             boxes, scores = self._split_outputs(outputs)
#             detections   = self.postprocess(boxes, scores, frame.shape[:2])

#             logger.debug(
#                 f"RF-DETR | detections={len(detections)} "
#                 f"| conf>{self.confidence_threshold}"
#             )
#             return detections

#         except (PreprocessError, PostprocessError, InferenceError):
#             raise
#         except Exception as e:
#             raise InferenceError(
#                 "RF-DETR inference failed",
#                 context={"frame_shape": list(frame.shape), "error": str(e)},
#             ) from e





































































# """
# RF-DETR / D-FINE detector using ONNX Runtime.

# ──────────────────────────────────────────────────────────────────────────
# IMPORTANT: these two model families do NOT share an output box format,
# despite both being DETR-style transformers with sigmoid scoring.
# ──────────────────────────────────────────────────────────────────────────

#   RF-DETR (Roboflow's own ONNX export, e.g. onnx-community/rfdetr_*-ONNX
#   exported via Roboflow's exporter):
#     - Outputs:  "dets" (N,4) and "labels" (N,num_classes)
#     - Box format: already XYXY, normalized [0,1] relative to the
#       ORIGINAL image. Roboflow's own PostProcess does the cxcywh→xyxy
#       conversion INSIDE the exported graph, so by the time you read the
#       ONNX output, it's already xyxy. This is Roboflow-specific behavior,
#       not a general DETR-family convention.
#     - Decode: boxes * [img_w, img_h, img_w, img_h] → pixel coords.
#     Reference: https://rfdetr.roboflow.com/develop/learn/export/#onnx-runtime

#   D-FINE (onnx-community/dfine_*-ONNX — a Transformers.js/Optimum-style
#   export of the RAW HuggingFace model, NOT Roboflow-style):
#     - Outputs:  "logits" (N,num_classes) and "pred_boxes" (N,4)
#     - Box format: CXCYWH, normalized [0,1] relative to the INPUT canvas
#       (the 640x640 resized image, not the original). This is the native
#       HF DFineForObjectDetection.forward() output — post_process_object_
#       detection (the function that does cxcywh→xyxy) runs OUTSIDE the
#       exported graph, in Python/JS, not baked into the ONNX file.
#     - Decode: cxcywh → xyxy (still normalized) → scale by input canvas
#       size → scale by original/input ratio → pixel coords.
#     Reference: https://huggingface.co/docs/transformers/model_doc/d_fine
#                https://huggingface.co/onnx-community/dfine_n_coco-ONNX

# Treating D-FINE's pred_boxes as if they were already-xyxy (the bug in the
# previous version of this file) produces boxes that are in roughly the
# right region but badly malformed in shape/size — exactly the symptom of
# "zero or garbage detections" you were chasing. This version detects
# which export format you actually have and decodes accordingly, instead
# of assuming RF-DETR's convention applies to both.
# """

# import os
# import cv2
# import numpy as np
# import supervision as sv
# from loguru import logger
# from typing import Tuple

# from src.vision.base import BaseDetector
# from src.vision.utils import create_session, bgr_to_rgb, normalize_imagenet, nms_numpy, cxcywh_to_xyxy
# from src.core.exceptions import ModelLoadError, InferenceError, PreprocessError, PostprocessError

# # DINOv2 patch size — RF-DETR's input resolution must be divisible by this.
# _DINOV2_PATCH_SIZE = 14
# _RFDETR_DEFAULT_SIZE = 560   # valid RF-DETR default (560 / 14 = 40)
# _DFINE_DEFAULT_SIZE = 640    # D-FINE's fixed training/export resolution


# class HFTransformerDetector(BaseDetector):
#     """RF-DETR / D-FINE detector via ONNX Runtime.

#     Auto-detects which family a given .onnx file belongs to (by output
#     tensor names) and applies the correct box-format decode for each,
#     rather than assuming one convention fits both.
#     """

#     def __init__(
#         self,
#         model_path: str,
#         conf_thresh: float = 0.45,
#         device: str = "cpu",
#     ):
#         """
#         Args:
#             model_path:  Path to the exported .onnx file.
#             conf_thresh: Confidence threshold (0.3-0.5 recommended).
#             device:      "cpu" or "cuda" (informational — session auto-selects).

#         Raises:
#             ModelLoadError: If the ONNX session cannot be created.
#         """
#         self.confidence_threshold = conf_thresh

#         try:
#             num_threads = max(1, (os.cpu_count() or 4) // 2 - 1)
#             logger.info(f"Loading HF-transformer ONNX | path={model_path} | threads={num_threads}")
#             self.session    = create_session(model_path, num_threads=num_threads)
#             self.input_name = self.session.get_inputs()[0].name

#             # Identify output indices AND, critically, which export family
#             # this is — that determines the box format we must decode.
#             self._boxes_idx  = None
#             self._labels_idx = None
#             self._boxes_name = None
#             for i, out in enumerate(self.session.get_outputs()):
#                 name = out.name.lower()
#                 if "box" in name or "det" in name:
#                     self._boxes_idx = i
#                     self._boxes_name = name
#                 elif "label" in name or "score" in name or "logit" in name:
#                     self._labels_idx = i

#             if len(self.session.get_outputs()) != 2:
#                 raise ModelLoadError(
#                     "Expected exactly 2 outputs (boxes + class scores)",
#                     context={
#                         "model_path": model_path,
#                         "outputs": [o.name for o in self.session.get_outputs()],
#                     },
#                 )

#             # Family detection:
#             #   - "pred_boxes" -> raw HF/Optimum export (D-FINE-style):
#             #     cxcywh, normalized to the INPUT canvas.
#             #   - "dets" / bare "boxes" -> Roboflow RF-DETR export:
#             #     already xyxy, normalized to the ORIGINAL image.
#             # This is a name-based heuristic, not a guess — these are the
#             # actual, distinct naming conventions each exporter uses.
#             if self._boxes_name and "pred_box" in self._boxes_name:
#                 self.box_format = "cxcywh_input_normalized"
#                 default_size = _DFINE_DEFAULT_SIZE
#             else:
#                 self.box_format = "xyxy_original_normalized"
#                 default_size = _RFDETR_DEFAULT_SIZE

#             # Read input resolution from the ONNX graph — never hardcode.
#             # onnxruntime reports dynamic/symbolic dims as strings (e.g.
#             # "height"), not ints — detect that explicitly rather than
#             # blindly casting, which is what silently broke this before.
#             shape = self.session.get_inputs()[0].shape  # [1, 3, H, W]
#             self.input_h, self.input_w = self._resolve_input_size(shape, default_size)

#             output_names = [o.name for o in self.session.get_outputs()]
#             logger.info(
#                 f"HF-transformer ready | format={self.box_format} "
#                 f"| input={self.input_h}x{self.input_w} "
#                 f"| outputs={output_names} | conf={conf_thresh}"
#             )

#         except ModelLoadError:
#             raise
#         except Exception as e:
#             raise ModelLoadError(
#                 "Failed to initialise HF-transformer ONNX detector",
#                 context={"model_path": model_path, "error": str(e)},
#             ) from e

#     # ── Input-size resolution ────────────────────────────────────────────

#     def _resolve_input_size(self, shape: list, default_size: int) -> Tuple[int, int]:
#         """
#         Read (H, W) from the ONNX graph's declared input shape.

#         Falls back to *default_size* (family-specific, NOT a single
#         hardcoded value shared across both architectures) only if the
#         graph reports a dynamic/symbolic dimension.

#         Args:
#             shape:        session.get_inputs()[0].shape, e.g. [1,3,640,640]
#                           or [1,3,'height','width'] for dynamic exports.
#             default_size: fallback square size for THIS detected family
#                           (640 for D-FINE, 560 for RF-DETR).

#         Returns:
#             (input_h, input_w) as concrete ints.
#         """
#         if len(shape) != 4:
#             raise ModelLoadError(
#                 "Unexpected input rank", context={"shape": shape, "expected_rank": 4}
#             )

#         raw_h, raw_w = shape[2], shape[3]
#         h = raw_h if isinstance(raw_h, int) and raw_h > 0 else None
#         w = raw_w if isinstance(raw_w, int) and raw_w > 0 else None

#         if h is None or w is None:
#             logger.warning(
#                 f"ONNX graph reports dynamic/symbolic input shape "
#                 f"(H={raw_h}, W={raw_w}) — falling back to "
#                 f"{default_size}x{default_size} for box_format={self.box_format}. "
#                 f"If this is wrong, re-export with a fixed input shape."
#             )
#             h = h or default_size
#             w = w or default_size

#         return h, w

#     # ── Preprocessing ─────────────────────────────────────────────────────

#     def preprocess(self, frame: np.ndarray) -> np.ndarray:
#         """
#         Prepare a BGR frame for ONNX inference.

#         Pipeline (shared by both families — both use plain square resize,
#         no letterboxing — only the OUTPUT decode differs):
#           BGR → RGB → plain resize (no letterbox) → /255 → ImageNet norm → NCHW

#         Args:
#             frame: BGR uint8 image (any resolution).

#         Returns:
#             (1, 3, input_h, input_w) float32 contiguous array.

#         Raises:
#             PreprocessError: If any step fails.
#         """
#         try:
#             img = bgr_to_rgb(frame)
#             img = cv2.resize(img, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
#             img = normalize_imagenet(img)                         # float32, ImageNet norm
#             img = np.transpose(img, (2, 0, 1))[np.newaxis]       # HWC → NCHW
#             return np.ascontiguousarray(img, dtype=np.float32)

#         except Exception as e:
#             raise PreprocessError(
#                 "HF-transformer preprocessing failed",
#                 context={"frame_shape": frame.shape, "error": str(e)},
#             ) from e

#     # ── Output routing ────────────────────────────────────────────────────

#     def _split_outputs(
#         self, outputs: list
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Return (boxes, scores) with batch dimension removed.

#         Args:
#             outputs: Raw list from session.run().

#         Returns:
#             boxes:  (N, 4) — format depends on self.box_format.
#             scores: (N, num_classes) — logits or probabilities.

#         Raises:
#             InferenceError: If outputs cannot be identified.
#         """
#         if self._boxes_idx is not None and self._labels_idx is not None:
#             return (
#                 outputs[self._boxes_idx][0],
#                 outputs[self._labels_idx][0],
#             )

#         if len(outputs) == 2:
#             a, b = outputs[0][0], outputs[1][0]
#             return (a, b) if a.shape[-1] == 4 else (b, a)

#         raise InferenceError(
#             f"Unexpected ONNX output count: {len(outputs)} (expected 2)",
#             context={"shapes": [list(o.shape) for o in outputs]},
#         )

#     # ── Postprocessing ────────────────────────────────────────────────────

#     def postprocess(
#         self,
#         boxes: np.ndarray,
#         scores: np.ndarray,
#         original_shape: Tuple[int, int],
#     ) -> sv.Detections:
#         """
#         Convert raw ONNX outputs to sv.Detections, using the decode path
#         appropriate to this model's detected export family.

#         Args:
#             boxes:          (N, 4) — format depends on self.box_format.
#             scores:         (N, num_classes) logits or probabilities.
#             original_shape: (H, W) of the frame passed to predict().

#         Returns:
#             sv.Detections in original-image pixel coordinates.

#         Raises:
#             PostprocessError: If decoding fails.
#         """
#         try:
#             img_h, img_w = original_shape

#             # Sigmoid if model outputs raw logits (min < 0 indicates logits).
#             # Both RF-DETR and D-FINE use focal-loss-style sigmoid scoring,
#             # so this step is shared.
#             if np.min(scores) < 0:
#                 scores = 1.0 / (1.0 + np.exp(-scores))

#             max_scores = np.max(scores, axis=1)     # (N,)
#             class_ids  = np.argmax(scores, axis=1)  # (N,)

#             mask = max_scores > self.confidence_threshold
#             if not np.any(mask):
#                 return sv.Detections.empty()

#             boxes      = boxes[mask]
#             max_scores = max_scores[mask]
#             class_ids  = class_ids[mask]

#             if self.box_format == "xyxy_original_normalized":
#                 # RF-DETR (Roboflow export): boxes are already xyxy,
#                 # normalized relative to the ORIGINAL image directly.
#                 scale    = np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
#                 boxes_px = (boxes * scale).astype(np.float32)

#             elif self.box_format == "cxcywh_input_normalized":
#                 # D-FINE (raw HF/Optimum export): boxes are cxcywh,
#                 # normalized relative to the INPUT CANVAS (e.g. 640x640),
#                 # not the original image. Decode in two stages:
#                 #   1. cxcywh -> xyxy, still normalized [0,1]
#                 #   2. normalized -> original-image pixels directly
#                 #      (plain resize means no letterbox padding to undo;
#                 #      both x and y are still fractions of the FULL image
#                 #      extent regardless of input canvas size, so scaling
#                 #      by [img_w, img_h] directly is correct here too)
#                 boxes_xyxy = cxcywh_to_xyxy(boxes)  # still normalized [0,1]
#                 scale      = np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
#                 boxes_px   = (boxes_xyxy * scale).astype(np.float32)

#             else:
#                 raise PostprocessError(
#                     f"Unknown box_format: {self.box_format}",
#                     context={"box_format": self.box_format},
#                 )

#             # Clip to frame boundaries
#             boxes_px[:, 0] = np.clip(boxes_px[:, 0], 0.0, img_w)  # x1
#             boxes_px[:, 1] = np.clip(boxes_px[:, 1], 0.0, img_h)  # y1
#             boxes_px[:, 2] = np.clip(boxes_px[:, 2], 0.0, img_w)  # x2
#             boxes_px[:, 3] = np.clip(boxes_px[:, 3], 0.0, img_h)  # y2

#             return sv.Detections(
#                 xyxy=boxes_px,
#                 confidence=max_scores,
#                 class_id=class_ids.astype(int),
#             )

#         except PostprocessError:
#             raise
#         except Exception as e:
#             raise PostprocessError(
#                 "HF-transformer postprocessing failed",
#                 context={
#                     "boxes_shape": list(boxes.shape),
#                     "scores_shape": list(scores.shape),
#                     "original_shape": original_shape,
#                     "box_format": self.box_format,
#                     "error": str(e),
#                 },
#             ) from e

#     # ── Inference ─────────────────────────────────────────────────────────

#     def predict(self, frame: np.ndarray) -> sv.Detections:
#         """
#         Run inference on a single BGR frame.

#         Args:
#             frame: BGR uint8 image (any resolution).

#         Returns:
#             sv.Detections in original-image pixel coordinates.

#         Raises:
#             InferenceError: If inference or output parsing fails.
#         """
#         try:
#             tensor        = self.preprocess(frame)
#             outputs       = self.session.run(None, {self.input_name: tensor})
#             boxes, scores = self._split_outputs(outputs)
#             detections    = self.postprocess(boxes, scores, frame.shape[:2])

#             logger.debug(
#                 f"{self.box_format} | detections={len(detections)} "
#                 f"| conf>{self.confidence_threshold}"
#             )
#             return detections

#         except (PreprocessError, PostprocessError, InferenceError):
#             raise
#         except Exception as e:
#             raise InferenceError(
#                 "HF-transformer inference failed",
#                 context={"frame_shape": list(frame.shape), "error": str(e)},
#             ) from e

























































# """
# RF-DETR / D-FINE detector using ONNX Runtime.

# ──────────────────────────────────────────────────────────────────────────
# IMPORTANT: these two model families do NOT share an output box format,
# despite both being DETR-style transformers with sigmoid scoring.
# ──────────────────────────────────────────────────────────────────────────

#   RF-DETR (Roboflow's own ONNX export, e.g. onnx-community/rfdetr_*-ONNX
#   exported via Roboflow's exporter):
#     - Outputs:  "dets" (N,4) and "labels" (N,num_classes)
#     - Box format: already XYXY, normalized [0,1] relative to the
#       ORIGINAL image. Roboflow's own PostProcess does the cxcywh→xyxy
#       conversion INSIDE the exported graph, so by the time you read the
#       ONNX output, it's already xyxy. This is Roboflow-specific behavior,
#       not a general DETR-family convention.
#     - Decode: boxes * [img_w, img_h, img_w, img_h] → pixel coords.
#     Reference: https://rfdetr.roboflow.com/develop/learn/export/#onnx-runtime

#   D-FINE (onnx-community/dfine_*-ONNX — a Transformers.js/Optimum-style
#   export of the RAW HuggingFace model, NOT Roboflow-style):
#     - Outputs:  "logits" (N,num_classes) and "pred_boxes" (N,4)
#     - Box format: CXCYWH, normalized [0,1] — scale by [img_w,img_h] → pixels.
#     - Preprocessing (from preprocessor_config.json, confirmed on HF):
#         do_resize:    true  — 640x640, BILINEAR (PIL resample=2)
#         do_rescale:   true  — pixel * (1/255) ONLY
#         do_normalize: FALSE — NO ImageNet mean/std subtraction at all
#         do_pad:       false — plain resize, no letterbox
#       The image_mean/image_std fields exist in the config but are UNUSED
#       because do_normalize=false. Applying ImageNet normalization here
#       (the bug in all previous versions of this file) shifts every pixel
#       far outside the training distribution — root cause of near-zero
#       detections despite correct box decoding.
#     Reference: https://huggingface.co/docs/transformers/model_doc/d_fine
#                preprocessor_config.json: RTDetrImageProcessor, do_normalize=false

# Both the wrong box format AND the wrong normalization must be fixed
# simultaneously — fixing only one still produces garbage. This version
# auto-detects the family by output tensor name and applies the correct
# preprocessing AND decode path for each.
# """

# import os
# import cv2
# import numpy as np
# import supervision as sv
# from loguru import logger
# from typing import Tuple

# from src.vision.base import BaseDetector
# from src.vision.utils import create_session, bgr_to_rgb, normalize_imagenet, nms_numpy, cxcywh_to_xyxy
# from src.core.exceptions import ModelLoadError, InferenceError, PreprocessError, PostprocessError

# # DINOv2 patch size — RF-DETR's input resolution must be divisible by this.
# _DINOV2_PATCH_SIZE = 14
# _RFDETR_DEFAULT_SIZE = 560   # valid RF-DETR default (560 / 14 = 40)
# _DFINE_DEFAULT_SIZE = 640    # D-FINE's fixed training/export resolution


# class HFTransformerDetector(BaseDetector):
#     """RF-DETR / D-FINE detector via ONNX Runtime.

#     Auto-detects which family a given .onnx file belongs to (by output
#     tensor names) and applies the correct box-format decode for each,
#     rather than assuming one convention fits both.
#     """

#     def __init__(
#         self,
#         model_path: str,
#         conf_thresh: float = 0.45,
#         device: str = "cpu",
#     ):
#         """
#         Args:
#             model_path:  Path to the exported .onnx file.
#             conf_thresh: Confidence threshold (0.3-0.5 recommended).
#             device:      "cpu" or "cuda" (informational — session auto-selects).

#         Raises:
#             ModelLoadError: If the ONNX session cannot be created.
#         """
#         self.confidence_threshold = conf_thresh

#         try:
#             num_threads = max(1, (os.cpu_count() or 4) // 2 - 1)
#             logger.info(f"Loading HF-transformer ONNX | path={model_path} | threads={num_threads}")
#             self.session    = create_session(model_path, num_threads=num_threads)
#             self.input_name = self.session.get_inputs()[0].name

#             # Identify output indices AND, critically, which export family
#             # this is — that determines the box format we must decode.
#             self._boxes_idx  = None
#             self._labels_idx = None
#             self._boxes_name = None
#             for i, out in enumerate(self.session.get_outputs()):
#                 name = out.name.lower()
#                 if "box" in name or "det" in name:
#                     self._boxes_idx = i
#                     self._boxes_name = name
#                 elif "label" in name or "score" in name or "logit" in name:
#                     self._labels_idx = i

#             if len(self.session.get_outputs()) != 2:
#                 raise ModelLoadError(
#                     "Expected exactly 2 outputs (boxes + class scores)",
#                     context={
#                         "model_path": model_path,
#                         "outputs": [o.name for o in self.session.get_outputs()],
#                     },
#                 )

#             # Family detection:
#             #   - "pred_boxes" -> raw HF/Optimum export (D-FINE-style):
#             #     cxcywh, normalized to the INPUT canvas.
#             #   - "dets" / bare "boxes" -> Roboflow RF-DETR export:
#             #     already xyxy, normalized to the ORIGINAL image.
#             # This is a name-based heuristic, not a guess — these are the
#             # actual, distinct naming conventions each exporter uses.
#             if self._boxes_name and "pred_box" in self._boxes_name:
#                 self.box_format = "cxcywh_input_normalized"
#                 default_size = _DFINE_DEFAULT_SIZE
#             else:
#                 self.box_format = "xyxy_original_normalized"
#                 default_size = _RFDETR_DEFAULT_SIZE

#             # Read input resolution from the ONNX graph — never hardcode.
#             # onnxruntime reports dynamic/symbolic dims as strings (e.g.
#             # "height"), not ints — detect that explicitly rather than
#             # blindly casting, which is what silently broke this before.
#             shape = self.session.get_inputs()[0].shape  # [1, 3, H, W]
#             self.input_h, self.input_w = self._resolve_input_size(shape, default_size)

#             output_names = [o.name for o in self.session.get_outputs()]
#             logger.info(
#                 f"HF-transformer ready | format={self.box_format} "
#                 f"| input={self.input_h}x{self.input_w} "
#                 f"| outputs={output_names} | conf={conf_thresh}"
#             )

#         except ModelLoadError:
#             raise
#         except Exception as e:
#             raise ModelLoadError(
#                 "Failed to initialise HF-transformer ONNX detector",
#                 context={"model_path": model_path, "error": str(e)},
#             ) from e

#     # ── Input-size resolution ────────────────────────────────────────────

#     def _resolve_input_size(self, shape: list, default_size: int) -> Tuple[int, int]:
#         """
#         Read (H, W) from the ONNX graph's declared input shape.

#         Falls back to *default_size* (family-specific, NOT a single
#         hardcoded value shared across both architectures) only if the
#         graph reports a dynamic/symbolic dimension.

#         Args:
#             shape:        session.get_inputs()[0].shape, e.g. [1,3,640,640]
#                           or [1,3,'height','width'] for dynamic exports.
#             default_size: fallback square size for THIS detected family
#                           (640 for D-FINE, 560 for RF-DETR).

#         Returns:
#             (input_h, input_w) as concrete ints.
#         """
#         if len(shape) != 4:
#             raise ModelLoadError(
#                 "Unexpected input rank", context={"shape": shape, "expected_rank": 4}
#             )

#         raw_h, raw_w = shape[2], shape[3]
#         h = raw_h if isinstance(raw_h, int) and raw_h > 0 else None
#         w = raw_w if isinstance(raw_w, int) and raw_w > 0 else None

#         if h is None or w is None:
#             logger.warning(
#                 f"ONNX graph reports dynamic/symbolic input shape "
#                 f"(H={raw_h}, W={raw_w}) — falling back to "
#                 f"{default_size}x{default_size} for box_format={self.box_format}. "
#                 f"If this is wrong, re-export with a fixed input shape."
#             )
#             h = h or default_size
#             w = w or default_size

#         return h, w

#     # ── Preprocessing ─────────────────────────────────────────────────────

#     def preprocess(self, frame: np.ndarray) -> np.ndarray:
#         """
#         Prepare a BGR frame for ONNX inference.

#         Both families share: BGR→RGB, plain square resize (no letterbox), NCHW.
#         They DIFFER in pixel normalization — confirmed from each family's
#         preprocessor_config.json on HuggingFace:

#           RF-DETR (xyxy_original_normalized):
#             /255  then  subtract ImageNet mean, divide ImageNet std
#             (do_normalize=true, standard ImageNet normalization)

#           D-FINE (cxcywh_input_normalized):
#             /255  ONLY — no mean/std subtraction at all
#             (do_normalize=false in RTDetrImageProcessor config)
#             rescale_factor = 1/255 = 0.00392156862745098

#         Applying ImageNet normalization to D-FINE is the root cause of
#         near-zero detections — it shifts pixel values ~0.45 below the
#         training distribution the model has never seen.

#         Args:
#             frame: BGR uint8 image (any resolution).

#         Returns:
#             (1, 3, input_h, input_w) float32 contiguous array.

#         Raises:
#             PreprocessError: If any step fails.
#         """
#         try:
#             img = bgr_to_rgb(frame)
#             img = cv2.resize(img, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)

#             if self.box_format == "cxcywh_input_normalized":
#                 # D-FINE: rescale_factor=1/255, do_normalize=FALSE
#                 # Matches AutoImageProcessor("ustc-community/dfine-*") exactly.
#                 img = img.astype(np.float32) / 255.0
#             else:
#                 # RF-DETR: /255 + ImageNet mean/std (do_normalize=true)
#                 img = normalize_imagenet(img)

#             img = np.transpose(img, (2, 0, 1))[np.newaxis]       # HWC → NCHW
#             return np.ascontiguousarray(img, dtype=np.float32)

#         except Exception as e:
#             raise PreprocessError(
#                 "HF-transformer preprocessing failed",
#                 context={"frame_shape": frame.shape, "error": str(e)},
#             ) from e

#     # ── Output routing ────────────────────────────────────────────────────

#     def _split_outputs(
#         self, outputs: list
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Return (boxes, scores) with batch dimension removed.

#         Args:
#             outputs: Raw list from session.run().

#         Returns:
#             boxes:  (N, 4) — format depends on self.box_format.
#             scores: (N, num_classes) — logits or probabilities.

#         Raises:
#             InferenceError: If outputs cannot be identified.
#         """
#         if self._boxes_idx is not None and self._labels_idx is not None:
#             return (
#                 outputs[self._boxes_idx][0],
#                 outputs[self._labels_idx][0],
#             )

#         if len(outputs) == 2:
#             a, b = outputs[0][0], outputs[1][0]
#             return (a, b) if a.shape[-1] == 4 else (b, a)

#         raise InferenceError(
#             f"Unexpected ONNX output count: {len(outputs)} (expected 2)",
#             context={"shapes": [list(o.shape) for o in outputs]},
#         )

#     # ── Postprocessing ────────────────────────────────────────────────────

#     def postprocess(
#         self,
#         boxes: np.ndarray,
#         scores: np.ndarray,
#         original_shape: Tuple[int, int],
#     ) -> sv.Detections:
#         """
#         Convert raw ONNX outputs to sv.Detections, using the decode path
#         appropriate to this model's detected export family.

#         Args:
#             boxes:          (N, 4) — format depends on self.box_format.
#             scores:         (N, num_classes) logits or probabilities.
#             original_shape: (H, W) of the frame passed to predict().

#         Returns:
#             sv.Detections in original-image pixel coordinates.

#         Raises:
#             PostprocessError: If decoding fails.
#         """
#         try:
#             img_h, img_w = original_shape

#             # Sigmoid if model outputs raw logits (min < 0 indicates logits).
#             # Both RF-DETR and D-FINE use focal-loss-style sigmoid scoring,
#             # so this step is shared.
#             if np.min(scores) < 0:
#                 scores = 1.0 / (1.0 + np.exp(-scores))

#             max_scores = np.max(scores, axis=1)     # (N,)
#             class_ids  = np.argmax(scores, axis=1)  # (N,)

#             mask = max_scores > self.confidence_threshold
#             if not np.any(mask):
#                 return sv.Detections.empty()

#             boxes      = boxes[mask]
#             max_scores = max_scores[mask]
#             class_ids  = class_ids[mask]

#             if self.box_format == "xyxy_original_normalized":
#                 # RF-DETR (Roboflow export): boxes are already xyxy,
#                 # normalized relative to the ORIGINAL image directly.
#                 scale    = np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
#                 boxes_px = (boxes * scale).astype(np.float32)

#             elif self.box_format == "cxcywh_input_normalized":
#                 # D-FINE (raw HF/Optimum export): boxes are cxcywh,
#                 # normalized relative to the INPUT CANVAS (e.g. 640x640),
#                 # not the original image. Decode in two stages:
#                 #   1. cxcywh -> xyxy, still normalized [0,1]
#                 #   2. normalized -> original-image pixels directly
#                 #      (plain resize means no letterbox padding to undo;
#                 #      both x and y are still fractions of the FULL image
#                 #      extent regardless of input canvas size, so scaling
#                 #      by [img_w, img_h] directly is correct here too)
#                 boxes_xyxy = cxcywh_to_xyxy(boxes)  # still normalized [0,1]
#                 scale      = np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
#                 boxes_px   = (boxes_xyxy * scale).astype(np.float32)

#             else:
#                 raise PostprocessError(
#                     f"Unknown box_format: {self.box_format}",
#                     context={"box_format": self.box_format},
#                 )

#             # Clip to frame boundaries
#             boxes_px[:, 0] = np.clip(boxes_px[:, 0], 0.0, img_w)  # x1
#             boxes_px[:, 1] = np.clip(boxes_px[:, 1], 0.0, img_h)  # y1
#             boxes_px[:, 2] = np.clip(boxes_px[:, 2], 0.0, img_w)  # x2
#             boxes_px[:, 3] = np.clip(boxes_px[:, 3], 0.0, img_h)  # y2

#             return sv.Detections(
#                 xyxy=boxes_px,
#                 confidence=max_scores,
#                 class_id=class_ids.astype(int),
#             )

#         except PostprocessError:
#             raise
#         except Exception as e:
#             raise PostprocessError(
#                 "HF-transformer postprocessing failed",
#                 context={
#                     "boxes_shape": list(boxes.shape),
#                     "scores_shape": list(scores.shape),
#                     "original_shape": original_shape,
#                     "box_format": self.box_format,
#                     "error": str(e),
#                 },
#             ) from e

#     # ── Inference ─────────────────────────────────────────────────────────

#     def predict(self, frame: np.ndarray) -> sv.Detections:
#         """
#         Run inference on a single BGR frame.

#         Args:
#             frame: BGR uint8 image (any resolution).

#         Returns:
#             sv.Detections in original-image pixel coordinates.

#         Raises:
#             InferenceError: If inference or output parsing fails.
#         """
#         try:
#             tensor        = self.preprocess(frame)
#             outputs       = self.session.run(None, {self.input_name: tensor})
#             boxes, scores = self._split_outputs(outputs)
#             detections    = self.postprocess(boxes, scores, frame.shape[:2])

#             logger.debug(
#                 f"{self.box_format} | detections={len(detections)} "
#                 f"| conf>{self.confidence_threshold}"
#             )
#             return detections

#         except (PreprocessError, PostprocessError, InferenceError):
#             raise
#         except Exception as e:
#             raise InferenceError(
#                 "HF-transformer inference failed",
#                 context={"frame_shape": list(frame.shape), "error": str(e)},
#             ) from e






















































































# """
# RF-DETR / D-FINE detector using ONNX Runtime.

# ──────────────────────────────────────────────────────────────────────────
# IMPORTANT: these two model families do NOT share an output box format,
# despite both being DETR-style transformers with sigmoid scoring.
# ──────────────────────────────────────────────────────────────────────────

#   RF-DETR (Roboflow's own ONNX export, e.g. onnx-community/rfdetr_*-ONNX
#   exported via Roboflow's exporter):
#     - Outputs:  "dets" (N,4) and "labels" (N,num_classes)
#     - Box format: already XYXY, normalized [0,1] relative to the
#       ORIGINAL image. Roboflow's own PostProcess does the cxcywh→xyxy
#       conversion INSIDE the exported graph, so by the time you read the
#       ONNX output, it's already xyxy. This is Roboflow-specific behavior,
#       not a general DETR-family convention.
#     - Decode: boxes * [img_w, img_h, img_w, img_h] → pixel coords.
#     Reference: https://rfdetr.roboflow.com/develop/learn/export/#onnx-runtime

#   D-FINE (onnx-community/dfine_*-ONNX — a Transformers.js/Optimum-style
#   export of the RAW HuggingFace model, NOT Roboflow-style):
#     - Outputs:  "logits" (N,num_classes) and "pred_boxes" (N,4)
#     - Box format: CXCYWH, normalized [0,1] — scale by [img_w,img_h] → pixels.
#     - Preprocessing (from preprocessor_config.json, confirmed on HF):
#         do_resize:    true  — 640x640, BILINEAR (PIL resample=2)
#         do_rescale:   true  — pixel * (1/255) ONLY
#         do_normalize: FALSE — NO ImageNet mean/std subtraction at all
#         do_pad:       false — plain resize, no letterbox
#       The image_mean/image_std fields exist in the config but are UNUSED
#       because do_normalize=false. Applying ImageNet normalization here
#       (the bug in all previous versions of this file) shifts every pixel
#       far outside the training distribution — root cause of near-zero
#       detections despite correct box decoding.
#     Reference: https://huggingface.co/docs/transformers/model_doc/d_fine
#                preprocessor_config.json: RTDetrImageProcessor, do_normalize=false

# Both the wrong box format AND the wrong normalization must be fixed
# simultaneously — fixing only one still produces garbage. This version
# auto-detects the family by output tensor name and applies the correct
# preprocessing AND decode path for each.
# """

# import os
# import cv2
# import numpy as np
# import supervision as sv
# from loguru import logger
# from typing import Tuple

# from src.vision.base import BaseDetector
# from src.vision.utils import create_session, bgr_to_rgb, normalize_imagenet, nms_numpy, cxcywh_to_xyxy
# from src.core.exceptions import ModelLoadError, InferenceError, PreprocessError, PostprocessError

# # DINOv2 patch size — RF-DETR's input resolution must be divisible by this.
# _DINOV2_PATCH_SIZE = 14
# _RFDETR_DEFAULT_SIZE = 560   # valid RF-DETR default (560 / 14 = 40)
# _DFINE_DEFAULT_SIZE = 640    # D-FINE's fixed training/export resolution


# class HFTransformerDetector(BaseDetector):
#     """RF-DETR / D-FINE detector via ONNX Runtime.

#     Auto-detects which family a given .onnx file belongs to (by output
#     tensor names) and applies the correct box-format decode for each,
#     rather than assuming one convention fits both.
#     """

#     def __init__(
#         self,
#         model_path: str,
#         conf_thresh: float = 0.45,
#         device: str = "cpu",
#     ):
#         """
#         Args:
#             model_path:  Path to the exported .onnx file.
#             conf_thresh: Confidence threshold (0.3-0.5 recommended).
#             device:      "cpu" or "cuda" (informational — session auto-selects).

#         Raises:
#             ModelLoadError: If the ONNX session cannot be created.
#         """
#         self.confidence_threshold = conf_thresh

#         try:
#             num_threads = max(1, (os.cpu_count() or 4) // 2 - 1)
#             logger.info(f"Loading HF-transformer ONNX | path={model_path} | threads={num_threads}")
#             self.session    = create_session(model_path, num_threads=num_threads)
#             self.input_name = self.session.get_inputs()[0].name

#             # Identify output indices AND, critically, which export family
#             # this is — that determines the box format we must decode.
#             self._boxes_idx  = None
#             self._labels_idx = None
#             self._boxes_name = None
#             for i, out in enumerate(self.session.get_outputs()):
#                 name = out.name.lower()
#                 if "box" in name or "det" in name:
#                     self._boxes_idx = i
#                     self._boxes_name = name
#                 elif "label" in name or "score" in name or "logit" in name:
#                     self._labels_idx = i

#             if len(self.session.get_outputs()) != 2:
#                 raise ModelLoadError(
#                     "Expected exactly 2 outputs (boxes + class scores)",
#                     context={
#                         "model_path": model_path,
#                         "outputs": [o.name for o in self.session.get_outputs()],
#                     },
#                 )

#             # Family detection:
#             #   - "pred_boxes" -> raw HF/Optimum export (D-FINE-style):
#             #     cxcywh, normalized to the INPUT canvas.
#             #   - "dets" / bare "boxes" -> Roboflow RF-DETR export:
#             #     already xyxy, normalized to the ORIGINAL image.
#             # This is a name-based heuristic, not a guess — these are the
#             # actual, distinct naming conventions each exporter uses.
#             if self._boxes_name and "pred_box" in self._boxes_name:
#                 self.box_format = "cxcywh_input_normalized"
#                 default_size = _DFINE_DEFAULT_SIZE
#             else:
#                 self.box_format = "xyxy_original_normalized"
#                 default_size = _RFDETR_DEFAULT_SIZE

#             # Read input resolution from the ONNX graph — never hardcode.
#             # onnxruntime reports dynamic/symbolic dims as strings (e.g.
#             # "height"), not ints — detect that explicitly rather than
#             # blindly casting, which is what silently broke this before.
#             shape = self.session.get_inputs()[0].shape  # [1, 3, H, W]
#             self.input_h, self.input_w = self._resolve_input_size(shape, default_size)

#             output_names = [o.name for o in self.session.get_outputs()]
#             logger.info(
#                 f"HF-transformer ready | format={self.box_format} "
#                 f"| input={self.input_h}x{self.input_w} "
#                 f"| outputs={output_names} | conf={conf_thresh}"
#             )

#         except ModelLoadError:
#             raise
#         except Exception as e:
#             raise ModelLoadError(
#                 "Failed to initialise HF-transformer ONNX detector",
#                 context={"model_path": model_path, "error": str(e)},
#             ) from e

#     # ── Input-size resolution ────────────────────────────────────────────

#     def _resolve_input_size(self, shape: list, default_size: int) -> Tuple[int, int]:
#         """
#         Read (H, W) from the ONNX graph's declared input shape.

#         Falls back to *default_size* (family-specific, NOT a single
#         hardcoded value shared across both architectures) only if the
#         graph reports a dynamic/symbolic dimension.

#         Args:
#             shape:        session.get_inputs()[0].shape, e.g. [1,3,640,640]
#                           or [1,3,'height','width'] for dynamic exports.
#             default_size: fallback square size for THIS detected family
#                           (640 for D-FINE, 560 for RF-DETR).

#         Returns:
#             (input_h, input_w) as concrete ints.
#         """
#         if len(shape) != 4:
#             raise ModelLoadError(
#                 "Unexpected input rank", context={"shape": shape, "expected_rank": 4}
#             )

#         raw_h, raw_w = shape[2], shape[3]
#         h = raw_h if isinstance(raw_h, int) and raw_h > 0 else None
#         w = raw_w if isinstance(raw_w, int) and raw_w > 0 else None

#         if h is None or w is None:
#             logger.warning(
#                 f"ONNX graph reports dynamic/symbolic input shape "
#                 f"(H={raw_h}, W={raw_w}) — falling back to "
#                 f"{default_size}x{default_size} for box_format={self.box_format}. "
#                 f"If this is wrong, re-export with a fixed input shape."
#             )
#             h = h or default_size
#             w = w or default_size

#         return h, w

#     # ── Preprocessing ─────────────────────────────────────────────────────

#     def preprocess(self, frame: np.ndarray) -> np.ndarray:
#         """
#         Prepare a BGR frame for ONNX inference.

#         Both families share: BGR→RGB, plain square resize (no letterbox), NCHW.
#         They DIFFER in pixel normalization — confirmed from each family's
#         preprocessor_config.json on HuggingFace:

#           RF-DETR (xyxy_original_normalized):
#             /255  then  subtract ImageNet mean, divide ImageNet std
#             (do_normalize=true, standard ImageNet normalization)

#           D-FINE (cxcywh_input_normalized):
#             /255  ONLY — no mean/std subtraction at all
#             (do_normalize=false in RTDetrImageProcessor config)
#             rescale_factor = 1/255 = 0.00392156862745098

#         Applying ImageNet normalization to D-FINE is the root cause of
#         near-zero detections — it shifts pixel values ~0.45 below the
#         training distribution the model has never seen.

#         Args:
#             frame: BGR uint8 image (any resolution).

#         Returns:
#             (1, 3, input_h, input_w) float32 contiguous array.

#         Raises:
#             PreprocessError: If any step fails.
#         """
#         try:
#             img = bgr_to_rgb(frame)
#             img = cv2.resize(img, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)

#             if self.box_format == "cxcywh_input_normalized":
#                 # D-FINE: rescale_factor=1/255, do_normalize=FALSE
#                 # Matches AutoImageProcessor("ustc-community/dfine-*") exactly.
#                 img = img.astype(np.float32) / 255.0
#             else:
#                 # RF-DETR: /255 + ImageNet mean/std (do_normalize=true)
#                 img = normalize_imagenet(img)

#             img = np.transpose(img, (2, 0, 1))[np.newaxis]       # HWC → NCHW
#             return np.ascontiguousarray(img, dtype=np.float32)

#         except Exception as e:
#             raise PreprocessError(
#                 "HF-transformer preprocessing failed",
#                 context={"frame_shape": frame.shape, "error": str(e)},
#             ) from e

#     # ── Output routing ────────────────────────────────────────────────────

#     def _split_outputs(
#         self, outputs: list
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Return (boxes, logits) with batch dimension removed.

#         Uses shape-based detection (last dim == 4 → boxes) on the
#         still-batched tensors, matching the working reference code exactly:

#             out0, out1 = outputs[0], outputs[1]          # (1, N, 4) / (1, N, C)
#             if out0.shape[-1] == 4:
#                 boxes, logits = out0, out1
#             else:
#                 logits, boxes = out0, out1
#             # caller receives boxes[0], logits[0]  →  (N, 4), (N, C)

#         Name-based matching is intentionally NOT used here because it can
#         silently assign the wrong tensor if both output names happen to
#         contain a matched keyword (e.g. "pred_boxes" and "logits" both
#         match, but so does any export that names outputs differently).
#         Shape is unambiguous: there is exactly one tensor with last-dim 4.

#         Args:
#             outputs: Raw list from session.run(), shapes (1, N, 4)/(1, N, C).

#         Returns:
#             boxes:  (N, 4) — batch dim removed, format per self.box_format.
#             logits: (N, num_classes) — raw logits, batch dim removed.

#         Raises:
#             InferenceError: If output count != 2 or no tensor has last-dim 4.
#         """
#         if len(outputs) != 2:
#             raise InferenceError(
#                 f"Expected exactly 2 ONNX outputs, got {len(outputs)}",
#                 context={"shapes": [list(o.shape) for o in outputs]},
#             )

#         out0, out1 = outputs[0], outputs[1]

#         if out0.shape[-1] == 4:
#             boxes, logits = out0, out1
#         elif out1.shape[-1] == 4:
#             logits, boxes = out0, out1
#         else:
#             raise InferenceError(
#                 "Cannot identify boxes output: neither tensor has last-dim == 4",
#                 context={"out0_shape": list(out0.shape), "out1_shape": list(out1.shape)},
#             )

#         return boxes[0], logits[0]   # strip batch dim → (N,4), (N,C)

#     # ── Postprocessing ────────────────────────────────────────────────────

#     # def postprocess(
#     #     self,
#     #     boxes: np.ndarray,
#     #     logits: np.ndarray,
#     #     original_shape: Tuple[int, int],
#     # ) -> sv.Detections:
#     #     """
#     #     Convert raw ONNX outputs to sv.Detections.

#     #     Mirrors the working reference postprocess() exactly:

#     #       D-FINE  (cxcywh_input_normalized):
#     #         1. sigmoid on logits → probs
#     #         2. best class per query (argmax) + confidence
#     #         3. confidence filter
#     #         4. cxcywh_to_xyxy  (normalized [0,1] → normalized [0,1])
#     #         5. scale x cols by orig_w, y cols by orig_h  → pixel coords
#     #         6. clip + return sv.Detections

#     #       RF-DETR  (xyxy_original_normalized):
#     #         Steps 1-3 identical.
#     #         4. boxes are already xyxy, normalized to original image.
#     #         5. scale all cols by [img_w, img_h, img_w, img_h].
#     #         6. clip + return sv.Detections.

#     #     Args:
#     #         boxes:          (N, 4) — cxcywh (D-FINE) or xyxy (RF-DETR),
#     #                          normalized [0,1], batch dim already removed.
#     #         logits:         (N, num_classes) raw logits, batch dim removed.
#     #         original_shape: (H, W) of the frame passed to predict().

#     #     Returns:
#     #         sv.Detections in original-image pixel coordinates.

#     #     Raises:
#     #         PostprocessError: If any decode step fails.
#     #     """
#     #     try:
#     #         img_h, img_w = original_shape

#     #         # sigmoid — both families use focal-loss (sigmoid) scoring
#     #         probs     = 1.0 / (1.0 + np.exp(-logits))   # (N, C)
#     #         class_ids = np.argmax(probs, axis=1)          # (N,)
#     #         scores    = probs[np.arange(len(class_ids)), class_ids]  # (N,)

#     #         mask = scores > self.confidence_threshold
#     #         if not np.any(mask):
#     #             return sv.Detections.empty()

#     #         scores    = scores[mask]
#     #         class_ids = class_ids[mask]
#     #         boxes     = boxes[mask]

#     #         if self.box_format == "cxcywh_input_normalized":
#     #             # D-FINE: cxcywh normalized → xyxy normalized → pixel coords.
#     #             # x and y columns are scaled independently, matching the
#     #             # reference implementation's explicit column indexing:
#     #             #   boxes_xyxy[:, [0,2]] *= orig_w
#     #             #   boxes_xyxy[:, [1,3]] *= orig_h
#     #             boxes_px = cxcywh_to_xyxy(boxes).astype(np.float32)
#     #             boxes_px[:, [0, 2]] *= img_w
#     #             boxes_px[:, [1, 3]] *= img_h

#     #         else:
#     #             # RF-DETR: already xyxy, normalized to original image.
#     #             boxes_px = (boxes * np.array(
#     #                 [img_w, img_h, img_w, img_h], dtype=np.float32
#     #             )).astype(np.float32)

#     #         boxes_px[:, 0] = np.clip(boxes_px[:, 0], 0.0, img_w)
#     #         boxes_px[:, 1] = np.clip(boxes_px[:, 1], 0.0, img_h)
#     #         boxes_px[:, 2] = np.clip(boxes_px[:, 2], 0.0, img_w)
#     #         boxes_px[:, 3] = np.clip(boxes_px[:, 3], 0.0, img_h)

#     #         return sv.Detections(
#     #             xyxy=boxes_px.reshape(-1, 4),
#     #             confidence=scores.astype(np.float32),
#     #             class_id=class_ids.astype(int),
#     #         )

#     #     except PostprocessError:
#     #         raise
#     #     except Exception as e:
#     #         raise PostprocessError(
#     #             "HF-transformer postprocessing failed",
#     #             context={
#     #                 "boxes_shape":  list(boxes.shape),
#     #                 "logits_shape": list(logits.shape),
#     #                 "original_shape": original_shape,
#     #                 "box_format":   self.box_format,
#     #                 "error": str(e),
#     #             },
#     #         ) from e




















#     def postprocess(
#         self,
#         boxes: np.ndarray,
#         logits: np.ndarray,
#         original_shape: Tuple[int, int],
#     ) -> sv.Detections:
#         img_h, img_w = original_shape

#         probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid — use_focal_loss=True default
#         class_ids = np.argmax(probs, axis=1)
#         scores = probs[np.arange(len(class_ids)), class_ids]

#         keep = scores > conf_thresh
#         if not np.any(keep):
#             return np.zeros((0, 4), dtype=np.float32), np.array([]), np.array([], dtype=int)

#         scores = scores[keep]
#         class_ids = class_ids[keep]
#         boxes = boxes[keep]

#         cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
#         x1 = (cx - w / 2) * img_w
#         y1 = (cy - h / 2) * img_h
#         x2 = (cx + w / 2) * img_w
#         y2 = (cy + h / 2) * img_h

#         boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
#         boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, img_w)
#         boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, img_h)


#         return sv.Detections(
#             xyxy=boxes_xyxy.reshape(-1, 4),
#             confidence=scores.astype(np.float32),
#             class_id=class_ids.astype(int),
#         )

#     # ── Inference ─────────────────────────────────────────────────────────

#     def predict(self, frame: np.ndarray) -> sv.Detections:
#         """
#         Run inference on a single BGR frame.

#         Args:
#             frame: BGR uint8 image (any resolution).

#         Returns:
#             sv.Detections in original-image pixel coordinates.

#         Raises:
#             InferenceError: If inference or output parsing fails.
#         """
#         try:
#             tensor         = self.preprocess(frame)
#             outputs        = self.session.run(None, {self.input_name: tensor})
#             boxes, logits  = self._split_outputs(outputs)
#             detections     = self.postprocess(boxes, logits, frame.shape[:2])

#             logger.debug(
#                 f"{self.box_format} | detections={len(detections)} "
#                 f"| conf>{self.confidence_threshold}"
#             )
#             return detections

#         except (PreprocessError, PostprocessError, InferenceError):
#             raise
#         except Exception as e:
#             raise InferenceError(
#                 "HF-transformer inference failed",
#                 context={"frame_shape": list(frame.shape), "error": str(e)},
#             ) from e































































