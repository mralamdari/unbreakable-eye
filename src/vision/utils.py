"""
Shared preprocessing and inference utilities for all vision detectors.

What belongs here:
  - ONNX Runtime session creation
  - Image transforms used by MORE THAN ONE detector (letterbox, normalize, etc.)
  - Bounding box math (format conversions, scaling, clipping)
  - NMS implementations
  - Re-ID crop preprocessing (preprocess_crop) — logically belongs in
    reid_preprocessing.py but kept here until that module is created

What does NOT belong here:
  - Model-specific postprocessing (stays in each detector file)
  - Settings or config imports (utils must have zero app-layer dependencies)
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
from loguru import logger
from typing import Optional, Tuple

from src.core.exceptions import ModelLoadError, PreprocessError

# ── OMP / MKL thread tuning ──────────────────────────────────────────────────
# Set before importing numpy or onnxruntime to take effect.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OMP_WAIT_POLICY", "ACTIVE")
os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")
os.environ.setdefault("KMP_BLOCKTIME", "0")

# ── Re-ID normalization constants (kept here until reid_preprocessing.py exists)
_REID_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_REID_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_MIN_BBOX_DIM = 20


# ─────────────────────────────────────────────────────────────────────────────
# ONNX Session
# ─────────────────────────────────────────────────────────────────────────────

def create_session(model_path: str, num_threads: int = 2) -> ort.InferenceSession:
    """
    Create an optimized ONNX Runtime InferenceSession.

    Automatically selects CUDA if available, falls back to CPU.

    Args:
        model_path:   Absolute or relative path to the .onnx file.
        num_threads:  intra-op thread count (rule of thumb: CPU_COUNT // 2 - 1).

    Returns:
        Ready-to-use ort.InferenceSession.

    Raises:
        ModelLoadError: If the file is missing or the session cannot be created.
    """
    if not os.path.exists(model_path):
        raise ModelLoadError(
            f"ONNX model file not found: {model_path}",
            context={"path": model_path},
        )
    try:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_mem_pattern       = True
        opts.enable_cpu_mem_arena     = True
        opts.intra_op_num_threads     = num_threads
        opts.inter_op_num_threads     = 2
        opts.execution_mode           = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")

        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]

        session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
        logger.info(f"ONNX session created | model={model_path} | providers={providers}")
        return session

    except ModelLoadError:
        raise
    except Exception as e:
        raise ModelLoadError(
            f"Failed to create ONNX session for {model_path}",
            context={"path": model_path, "error": str(e)},
        ) from e


# ─────────────────────────────────────────────────────────────────────────────
# Image Resizing / Padding
# ─────────────────────────────────────────────────────────────────────────────

def letterbox(
    img: np.ndarray,
    target_h: int,
    target_w: int,
    pad_value: int = 114,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Resize *img* preserving aspect ratio then pad to (target_h, target_w).

    Used by: YOLO ONNX, YOLOX (and any model trained with letterbox augmentation).
    NOT used by: RF-DETR / D-FINE (plain resize, no padding).

    Args:
        img:        BGR uint8 image.
        target_h:   Canvas height after padding.
        target_w:   Canvas width  after padding.
        pad_value:  Pixel fill value — 114 (gray) for YOLO, 0 (black) for others.

    Returns:
        (padded_image, (top_pad, left_pad))
        top_pad / left_pad are needed by postprocess to undo the offset.
    """
    h, w = img.shape[:2]
    scale = min(target_h / h, target_w / w)
    new_w, new_h = round(w * scale), round(h * scale)

    if (w, h) != (new_w, new_h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    dh = (target_h - new_h) / 2
    dw = (target_w - new_w) / 2
    top,    bottom = round(dh - 0.1), round(dh + 0.1)
    left,   right  = round(dw - 0.1), round(dw + 0.1)

    img = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=(pad_value, pad_value, pad_value),
    )
    return img, (top, left)


def plain_resize(
    img: np.ndarray,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """
    Simple resize WITHOUT aspect ratio preservation or padding.

    Used by: RF-DETR, D-FINE, OpenVINO models.
    The model's transformer decoder handles coordinate mapping internally.

    Args:
        img:      BGR uint8 image (any size).
        target_h: Target height.
        target_w: Target width.

    Returns:
        Resized image of shape (target_h, target_w, 3).
    """
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


# ─────────────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize_imagenet(img: np.ndarray) -> np.ndarray:
    """
    Scale [0,255] → [0,1] then subtract ImageNet mean and divide by std.

    Used by: RF-DETR, D-FINE, OpenVINO models (DINOv2 backbone).

    Args:
        img: HxWx3 float32 or uint8 image.

    Returns:
        HxWx3 float32 normalized image.
    """
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (img - mean) / std


def normalize_simple(img: np.ndarray) -> np.ndarray:
    """
    Scale [0,255] → [0,1] only (no mean/std subtraction).

    Used by: YOLO variants — they were trained without ImageNet normalization.

    Args:
        img: HxWx3 uint8 or float image.

    Returns:
        HxWx3 float32 image in [0, 1].
    """
    return img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Format Conversions
# ─────────────────────────────────────────────────────────────────────────────

def to_chw_float32(img: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Convert HWC uint8 image to CHW float32 for ONNX inference.

    Args:
        img:       HxWx3 image.
        normalize: If True, scale to [0, 1].

    Returns:
        (3, H, W) float32 contiguous array.
    """
    arr = img.astype(np.float32) / 255.0 if (normalize and img.dtype == np.uint8) \
          else img.astype(np.float32)
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Flip channel order BGR → RGB. Required by DINOv2-based models."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    Convert (cx, cy, w, h) → (x1, y1, x2, y2).

    Works with both normalized [0,1] and absolute pixel coordinates.

    Args:
        boxes: (N, 4) array in center format.

    Returns:
        (N, 4) array in corner format.
    """
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def xyxy_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
    """
    Convert (x1, y1, x2, y2) → (cx, cy, w, h).

    Args:
        boxes: (N, 4) array in corner format.

    Returns:
        (N, 4) array in center format.
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Bounding Box Geometry
# ─────────────────────────────────────────────────────────────────────────────

def scale_boxes(
    boxes: np.ndarray,
    scale_x: float,
    scale_y: float,
    clip_h: Optional[int] = None,
    clip_w: Optional[int] = None,
) -> np.ndarray:
    """
    Scale (x1,y1,x2,y2) boxes and optionally clip to image boundaries.

    Args:
        boxes:   (N, 4) float32 in [x1, y1, x2, y2] format.
        scale_x: Horizontal multiplier.
        scale_y: Vertical multiplier.
        clip_h:  Image height — clips y1,y2 when provided.
        clip_w:  Image width  — clips x1,x2 when provided.

    Returns:
        (N, 4) float32 scaled and clipped boxes.
    """
    out = boxes.copy()
    out[:, 0::2] *= scale_x   # x1, x2
    out[:, 1::2] *= scale_y   # y1, y2
    if clip_h is not None and clip_w is not None:
        out[:, 0] = np.clip(out[:, 0], 0, clip_w)
        out[:, 1] = np.clip(out[:, 1], 0, clip_h)
        out[:, 2] = np.clip(out[:, 2], 0, clip_w)
        out[:, 3] = np.clip(out[:, 3], 0, clip_h)
    return out


def remove_letterbox_padding(
    boxes: np.ndarray,
    top_pad: int,
    left_pad: int,
    scale: float,
) -> np.ndarray:
    """
    Map boxes from letterbox-canvas space back to original-image pixel space.

    Args:
        boxes:    (N, 4) float32 [x1,y1,x2,y2] in canvas coordinates.
        top_pad:  Pixels of top padding added by letterbox().
        left_pad: Pixels of left padding added by letterbox().
        scale:    The scale factor used during letterbox resize.

    Returns:
        (N, 4) boxes in original image pixel space.
    """
    out = boxes.copy()
    out[:, 0] = (out[:, 0] - left_pad) / scale   # x1
    out[:, 1] = (out[:, 1] - top_pad)  / scale   # y1
    out[:, 2] = (out[:, 2] - left_pad) / scale   # x2
    out[:, 3] = (out[:, 3] - top_pad)  / scale   # y2
    return out


# ─────────────────────────────────────────────────────────────────────────────
# NMS
# ─────────────────────────────────────────────────────────────────────────────

def nms_numpy(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """
    Pure-NumPy class-agnostic NMS — no OpenCV or supervision dependency.

    Used by: YOLOX (via multiclass_nms_class_agnostic), RF-DETR postprocess.

    Args:
        boxes:         (N, 4) float32 [x1, y1, x2, y2] in any pixel unit.
        scores:        (N,)   float32 confidence scores.
        iou_threshold: Boxes with IoU > this value are suppressed.

    Returns:
        (K,) int array of indices to keep, ordered by descending score.
    """
    if boxes.shape[0] == 0:
        return np.array([], dtype=int)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas  = (x2 - x1 + 1) * (y2 - y1 + 1)
    order  = scores.argsort()[::-1]
    keep   = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_threshold]

    return np.array(keep, dtype=int)


def apply_nms_supervision(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """
    NMS via supervision.box_non_max_suppression.

    Used by: YOLO ONNX (UltralyticsONNXDetector).

    Args:
        boxes:         (N, 4) float32 [x1, y1, x2, y2].
        scores:        (N,)   float32.
        iou_threshold: Suppression threshold.

    Returns:
        (K,) int array of kept indices.
    """
    import supervision as sv
    nms_input = np.concatenate([boxes, scores[:, np.newaxis]], axis=1).astype(np.float32)
    return sv.box_non_max_suppression(nms_input, iou_threshold)


def apply_nms_opencv(
    boxes: np.ndarray,
    scores: np.ndarray,
    score_threshold: float,
    iou_threshold: float,
) -> np.ndarray:
    """
    NMS via cv2.dnn.NMSBoxes — expects [x, y, w, h] internally.

    Args:
        boxes:           (N, 4) float32 [x1, y1, x2, y2].
        scores:          (N,)   float32.
        score_threshold: Pre-filter confidence threshold.
        iou_threshold:   Suppression threshold.

    Returns:
        (K,) int array of kept indices.
    """
    xywh = [[int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])] for b in boxes]
    indices = cv2.dnn.NMSBoxes(xywh, scores.tolist(), score_threshold, iou_threshold)
    return np.array(indices, dtype=int).flatten() if len(indices) > 0 else np.array([], dtype=int)


# ─────────────────────────────────────────────────────────────────────────────
# Re-ID Crop Preprocessing
# TODO: move to src/vision/reid_preprocessing.py once that module is created
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_crop(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    model_input_size: Tuple[int, int],
    torso_ratio: float = 1.0,
) -> Tuple[np.ndarray, Tuple[int, int, int, int], Tuple[float, float], int, int, bool]:
    """
    Crop a detection box from a frame and prepare it for the Re-ID model (OSNet).

    Args:
        frame:            Full BGR frame (HxWx3 uint8).
        bbox:             (x1, y1, x2, y2) detection box in frame pixels.
        model_input_size: (H, W) expected by Re-ID model — e.g. (256, 128).
        torso_ratio:      Fraction of bbox height to keep from the top.
                          1.0 = full body, 0.5 = upper half only.

    Returns:
        input_tensor:  (3, H, W) float32 ImageNet-normalized — feed to OSNet.
        crop_box:      (x1, y1, x2, y2) ints — the exact region cropped.
        center_point:  (cx, cy) of the original bbox.
        bbox_w:        Width of the original bbox.
        bbox_h:        Height of the original bbox.
        flag:          True  → crop is unusable (too small / empty), skip embedding.
                       False → crop is valid.

    Raises:
        PreprocessError: If the resize or normalization step fails.
    """
    x1, y1, x2, y2 = map(int, bbox)
    bbox_w      = x2 - x1
    bbox_h      = y2 - y1
    center_point = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    crop_y2 = int(y1 + bbox_h * torso_ratio)
    crop    = frame[y1:crop_y2, x1:x2]

    flag = (bbox_w < _MIN_BBOX_DIM) or (bbox_h < _MIN_BBOX_DIM) or (crop.size == 0)

    if flag:
        logger.debug(
            f"Re-ID crop skipped | bbox=({x1},{y1},{x2},{y2}) "
            f"size={bbox_w}x{bbox_h} empty={crop.size == 0}"
        )
        placeholder = np.zeros(
            (3, model_input_size[0], model_input_size[1]), dtype=np.float32
        )
        return placeholder, (x1, y1, x2, y2), center_point, bbox_w, bbox_h, flag

    try:
        resized    = cv2.resize(crop, (model_input_size[1], model_input_size[0]),
                                interpolation=cv2.INTER_AREA)
        normalized = (resized.astype(np.float32) / 255.0 - _REID_MEAN) / _REID_STD
        tensor     = np.ascontiguousarray(normalized.transpose(2, 0, 1))
        return tensor, (x1, y1, x2, y2), center_point, bbox_w, bbox_h, flag

    except Exception as e:
        raise PreprocessError(
            "Re-ID crop preprocessing failed",
            context={"bbox": bbox, "crop_shape": crop.shape, "error": str(e)},
        ) from e
