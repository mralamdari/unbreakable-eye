# # from src.core.config import settings
# import os
# import cv2
# import numpy as np
# import supervision as sv
# from loguru import logger
# from typing import Tuple
# from src.core.config import settings
# from src.vision.base import BaseDetector
# from src.vision.utils import create_session
# from src.core.exceptions import ModelLoadError, InferenceError, PreprocessError, PostprocessError

# # ImageNet normalization — required by DINOv2 backbone
# _IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
# _IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# def cxcywh_to_xyxy(boxes_norm: np.ndarray) -> np.ndarray:
#     """
#     Convert [cx, cy, w, h] in [0,1] → [x1, y1, x2, y2] in [0,1].
#     RF-DETR (like RT-DETR) always outputs normalized cx/cy/w/h.
#     """
#     cx, cy, w, h = boxes_norm[:, 0], boxes_norm[:, 1], boxes_norm[:, 2], boxes_norm[:, 3]
#     x1 = cx - w / 2
#     y1 = cy - h / 2
#     x2 = cx + w / 2
#     y2 = cy + h / 2
#     return np.stack([x1, y1, x2, y2], axis=1)

# def nms(boxes, scores, class_ids, iou_thresh=0.4):
#     """Simple class-agnostic NMS."""
#     if len(boxes) == 0:
#         return boxes, scores, class_ids

#     x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
#     areas = (x2 - x1) * (y2 - y1)
#     order = scores.argsort()[::-1]

#     keep = []
#     while order.size > 0:
#         i = order[0]
#         keep.append(i)
#         xx1 = np.maximum(x1[i], x1[order[1:]])
#         yy1 = np.maximum(y1[i], y1[order[1:]])
#         xx2 = np.minimum(x2[i], x2[order[1:]])
#         yy2 = np.minimum(y2[i], y2[order[1:]])
#         inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
#         iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
#         order = order[1:][iou <= iou_thresh]

#     keep = np.array(keep)
#     return boxes[keep], scores[keep], class_ids[keep]

# class HFTransformerDetector(BaseDetector):
#     """
#     RF-DETR / D-FINE detector via ONNX Runtime.

#     Use scripts/export_rfdetr.py to obtain the .onnx file from pretrained
#     COCO weights before running this detector.
#     """
#     def __init__(
#         self,
#         model_path: str,
#         model_name: str,
#         input_dim: tuple[int, int],
#         conf_thresh: float = 0.4,
#         device: str = "cpu",
#     ):
#         self.confidence_threshold = conf_thresh

#         try:
#             num_threads = max(1, (os.cpu_count() or 4) // 2 - 1)
#             logger.info(f"Loading RF-DETR ONNX model: {model_path} ({num_threads} threads)")
#             self.session = create_session(model_path, num_threads=num_threads)

#             # Read input shape from ONNX graph — never hardcode
#             inp = self.session.get_inputs()[0]
#             shape = inp.shape  # [1, 3, H, W]
#             # self.input_h = 512
#             # self.input_w = 512
#             self.input_h = 640
#             self.input_w = 640
            
#             # self.input_h = int(shape[2]) if isinstance(shape[2], int) else 560
#             # self.input_w = int(shape[3]) if isinstance(shape[3], int) else 560
#             self.input_name = inp.name
#             # self.input_size = settings.FRAME_SHAPE
#             # Map output names — Roboflow exports use "boxes" and "labels"
#             self._boxes_idx  = None
#             self._labels_idx = None
#             for i, o in enumerate(self.session.get_outputs()):
#                 name = o.name.lower()
#                 if "box" in name:
#                     self._boxes_idx = i
#                 elif "label" in name or "score" in name or "logit" in name:
#                     self._labels_idx = i

#             output_names = [o.name for o in self.session.get_outputs()]
#             logger.info(f"RF-DETR initialized: input {self.input_h}x{self.input_w}, "
#                        f"outputs={output_names}, conf={conf_thresh}")

#         except Exception as e:
#             raise ModelLoadError(
#                 "Failed to load RF-DETR ONNX model",
#                 context={"model_path": model_path, "error": str(e)}
#             ) from e

#     def preprocess(self, frame, orig_h, orig_w) -> np.ndarray:
#         """
#         Preprocess BGR frame for RF-DETR ONNX inference.

#         Follows Roboflow official example:
#           BGR -> RGB -> resize (no letterbox) -> /255 -> ImageNet norm -> NCHW
#         """
#         try:
#             scale  = self.input_w / max(orig_h, orig_w)
#             new_w  = int(round(orig_w * scale))
#             new_h  = int(round(orig_h * scale))
#             resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
#             canvas = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
#             pad_top  = (self.input_h - new_h) // 2
#             pad_left = (self.input_w - new_w) // 2
#             canvas[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
#             rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
#             rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
#             blob = rgb.transpose(2, 0, 1)[np.newaxis, ...]
#             return blob, scale, pad_top, pad_left
#         except Exception as e:
#             raise PreprocessError(
#                 "RF-DETR preprocessing failed",
#                 context={"frame_shape": frame.shape, "error": str(e)}
#             ) from e
            
#     def postprocess(
#         self,
#         outputs, 
#         orig_h, orig_w, 
#         scale,
#         pad_top, 
#         pad_left,
#     ) -> sv.Detections:
        
#         # try:
#         """
#         Decode raw ONNX outputs into (boxes_pixel, scores, class_ids).

#         The model returns two tensors:
#         logits  [1, num_queries, num_classes]  – raw class logits
#         boxes   [1, num_queries, 4]            – cx,cy,w,h in [0,1]
#         """
#         # ── locate the two output tensors ────────────────────────────────────────
#         # Different exports may swap the order, so we identify by shape
#         o0, o1 = outputs[0], outputs[1]  # shapes like (1,300,91) and (1,300,4)
                
#         if o0.shape[-1] == 4:
#             raw_boxes, raw_logits = o0, o1
#         else:
#             raw_logits, raw_boxes = o0, o1

#         raw_logits = raw_logits[0]   # (num_queries, num_classes)
#         raw_boxes  = raw_boxes[0]    # (num_queries, 4)

#         # Sigmoid → probabilities (RF-DETR uses sigmoid, not softmax)
#         probs = 1.0 / (1.0 + np.exp(-raw_logits))  # (Q, C)

#         # Best class per query
#         class_ids  = np.argmax(probs, axis=1)       # (Q,)
#         scores     = probs[np.arange(len(class_ids)), class_ids]  # (Q,)

#         # Filter by threshold
#         keep = scores > settings.CONF_THRESHOLD
#         if not keep.any():
#             return [], [], []

#         scores     = scores[keep]
#         class_ids  = class_ids[keep]
#         boxes_norm = raw_boxes[keep]

#         # cx,cy,w,h → x1,y1,x2,y2  (still normalized 0-1 relative to 512×512 canvas)
#         boxes_xyxy = cxcywh_to_xyxy(boxes_norm)
#         # Unpad: remove letterbox offset and undo scale to get original pixel coords
#         # canvas coords (0-1) → pixel in 512×512 canvas
#         boxes_px = boxes_xyxy * self.input_w

#         # Remove padding
#         boxes_px[:, [0, 2]] -= pad_left
#         boxes_px[:, [1, 3]] -= pad_top

#         # Rescale to original image size
#         boxes_px /= scale

#         # Clip to image boundaries
#         boxes_px[:, [0, 2]] = np.clip(boxes_px[:, [0, 2]], 0, orig_w)
#         boxes_px[:, [1, 3]] = np.clip(boxes_px[:, [1, 3]], 0, orig_h)

#         # NMS
#         boxes_px, scores, class_ids = nms(boxes_px, scores, class_ids)
#         return sv.Detections(
#             xyxy=boxes_px.astype(np.float32),
#             confidence=scores,
#             class_id=class_ids.astype(int),
#         )

#     def predict(self, frame: np.ndarray) -> sv.Detections:
        
#         orig_h, orig_w = frame.shape[:2]
#         input_tensor, scale, pad_top, pad_left = self.preprocess(frame, orig_h, orig_w)
#         # ── Inference ─────────────────────────────────────────────────────────────
#         outputs = self.session.run(None, {self.input_name: input_tensor})
#         # ── Decode outputs ────────────────────────────────────────────────────────
#         detections = self.postprocess(
#             outputs, orig_h, orig_w, scale, pad_top, pad_left)
#         return detections


































































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
    def preprocess(self, frame, orig_h, orig_w) -> np.ndarray:
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
            
    def postprocess():
        boxes_px /= scale
        # NMS
        boxes_px, scores, class_ids = nms(boxes_px, scores, class_ids)
