"""
Shared preprocessing and inference utilities for all vision models.
These functions are model-agnostic and used by multiple detector backends.
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
from loguru import logger
from typing import Tuple
from src.core.exceptions import PreprocessError

# ONNX Runtime optimization settings
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OMP_WAIT_POLICY"] = "ACTIVE"
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
os.environ["KMP_BLOCKTIME"] = "0"


# ─────────────────────────────────────────────────────────────────────────────
# ONNX Runtime Session Management
# ─────────────────────────────────────────────────────────────────────────────

def create_session(model_path: str, num_threads: int = 2) -> ort.InferenceSession:
    """
    Create an optimized ONNX Runtime session with tuned performance settings.

    Args:
        model_path: Path to .onnx model file
        num_threads: Number of threads for intra-op parallelism (usually CPU_COUNT // 2 - 1)

    Returns:
        Configured ort.InferenceSession

    Raises:
        FileNotFoundError: If model_path doesn't exist
        ort.InvalidProtobuf: If model is corrupted
    """
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = 2
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")

        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        
        if not providers:
            logger.warning("No execution providers available, falling back to CPU")
            providers = ["CPUExecutionProvider"]

        logger.debug(f"Creating ONNX session with providers: {providers}")
        session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        logger.info(f"ONNX model loaded successfully: {model_path}")
        return session

    except Exception as e:
        logger.error(f"Failed to create ONNX session: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Image Preprocessing — Letterbox / Padding
# ─────────────────────────────────────────────────────────────────────────────

def letterbox(
    img: np.ndarray,
    target_h: int,
    target_w: int,
    pad_value: int = 114
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Resize image with aspect ratio preservation + padding (letterbox).
    Used by: YOLO, YOLOX, some Transformers.

    Args:
        img: Input BGR image (HxWx3)
        target_h: Target height
        target_w: Target width
        pad_value: Padding color (default 114 = gray for YOLO, 0 = black for some transformers)

    Returns:
        (padded_image, (top_pad, left_pad)) where padded_image is target_h x target_w
    """
    shape = img.shape[:2]  # (H, W)
    r = min(target_h / shape[0], target_w / shape[1])
    new_unpad = (round(shape[1] * r), round(shape[0] * r))  # (W, H)
    dw, dh = (target_w - new_unpad[0]) / 2, (target_h - new_unpad[1]) / 2

    # Resize
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Pad
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(pad_value, pad_value, pad_value))
    return img, (top, left)


def aspect_ratio_resize(
    img: np.ndarray,
    target_h: int,
    target_w: int
) -> Tuple[np.ndarray, float]:
    """
    Resize with aspect ratio preservation (no padding, just scaling).
    Used by: Some OpenVINO models, custom pipelines.

    Args:
        img: Input BGR image
        target_h: Target height
        target_w: Target width

    Returns:
        (resized_image, scale_ratio) where scale_ratio is the scaling factor
    """
    h, w = img.shape[:2]
    scale = min(target_h / h, target_w / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, scale


# ─────────────────────────────────────────────────────────────────────────────
# Normalization Functions
# ─────────────────────────────────────────────────────────────────────────────

def normalize_imagenet(img: np.ndarray) -> np.ndarray:
    """
    ImageNet normalization: scale to [0, 1], then apply mean/std.
    Used by: Vision Transformers, DETR models, some YOLO variants.
    
    Formula: (img / 255.0 - mean) / std
    where mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]

    Args:
        img: Input image in [0, 255] range, float32 or uint8

    Returns:
        Normalized image as float32
    """
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (img - mean) / std


def normalize_simple(img: np.ndarray) -> np.ndarray:
    """
    Simple 0-1 normalization: just scale to [0, 1].
    Used by: YOLO, YOLOX, most CNN detectors.

    Args:
        img: Input image uint8 or float in [0, 255]

    Returns:
        Normalized image as float32 in [0, 1]
    """
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Image Format Conversions
# ─────────────────────────────────────────────────────────────────────────────

def to_chw_float32(img: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Convert HWC uint8 image to CHW float32 format for inference.

    Args:
        img: Input HxWx3 image (uint8 or float)
        normalize: If True, scale to [0, 1]

    Returns:
        CxHxW float32 tensor ready for ONNX/TensorRT
    """
    if img.dtype == np.uint8 and normalize:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
    
    chw = np.transpose(img, (2, 0, 1))
    return np.ascontiguousarray(chw)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert BGR image to RGB (important for transformers)."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────────────────────────────────
# Bounding Box Operations
# ─────────────────────────────────────────────────────────────────────────────

def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    Convert bounding boxes from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2).
    
    Args:
        boxes: Nx4 array in [cx, cy, w, h] format

    Returns:
        Nx4 array in [x1, y1, x2, y2] format
    """
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def xyxy_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
    """
    Convert bounding boxes from corner format (x1, y1, x2, y2) to center format (cx, cy, w, h).
    
    Args:
        boxes: Nx4 array in [x1, y1, x2, y2] format

    Returns:
        Nx4 array in [cx, cy, w, h] format
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return np.stack([cx, cy, w, h], axis=1)


def scale_boxes(
    boxes: np.ndarray,
    scale_x: float,
    scale_y: float,
    clip_h: int = None,
    clip_w: int = None
) -> np.ndarray:
    """
    Scale bounding boxes by given factors and optionally clip to image boundaries.

    Args:
        boxes: Nx4 array in [x1, y1, x2, y2] format
        scale_x: Horizontal scale factor
        scale_y: Vertical scale factor
        clip_h: Image height for clipping (optional)
        clip_w: Image width for clipping (optional)

    Returns:
        Scaled and optionally clipped boxes
    """
    scaled = boxes.copy()
    scaled[:, 0::2] *= scale_x  # x1, x2
    scaled[:, 1::2] *= scale_y  # y1, y2

    if clip_h is not None and clip_w is not None:
        scaled[:, 0] = np.clip(scaled[:, 0], 0, clip_w)  # x1
        scaled[:, 1] = np.clip(scaled[:, 1], 0, clip_h)  # y1
        scaled[:, 2] = np.clip(scaled[:, 2], 0, clip_w)  # x2
        scaled[:, 3] = np.clip(scaled[:, 3], 0, clip_h)  # y2

    return scaled


def remove_letterbox_padding(
    boxes: np.ndarray,
    top_pad: int,
    left_pad: int,
    scale: float
) -> np.ndarray:
    """
    Convert box coordinates from letterbox space back to original image space.

    Args:
        boxes: Nx4 array in [x1, y1, x2, y2] format (in letterbox space)
        top_pad: Padding added to top
        left_pad: Padding added to left
        scale: Scale factor used during letterbox

    Returns:
        Boxes in original image space
    """
    unpadded = boxes.copy()
    unpadded[:, 0] = (unpadded[:, 0] - left_pad) / scale  # x1
    unpadded[:, 1] = (unpadded[:, 1] - top_pad) / scale   # y1
    unpadded[:, 2] = (unpadded[:, 2] - left_pad) / scale  # x2
    unpadded[:, 3] = (unpadded[:, 3] - top_pad) / scale   # y3
    return unpadded


# ─────────────────────────────────────────────────────────────────────────────
# NMS Utilities
# ─────────────────────────────────────────────────────────────────────────────

def apply_nms_supervision(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float
) -> np.ndarray:
    """
    Apply NMS using supervision library.

    Args:
        boxes: Nx4 array in [x1, y1, x2, y2]
        scores: N array of confidence scores
        iou_threshold: NMS threshold

    Returns:
        Array of indices to keep
    """
    import supervision as sv
    nms_boxes = np.stack([boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], scores], axis=1)
    indices = sv.box_non_max_suppression(nms_boxes, iou_threshold)
    return indices


def apply_nms_opencv(
    boxes: np.ndarray,
    scores: np.ndarray,
    score_threshold: float,
    iou_threshold: float
) -> np.ndarray:
    """
    Apply NMS using OpenCV (cv2.dnn.NMSBoxes).

    Args:
        boxes: Nx4 array in [x1, y1, x2, y2]
        scores: N array of confidence scores
        score_threshold: Confidence threshold
        iou_threshold: NMS threshold

    Returns:
        Array of indices to keep
    """
    # cv2.dnn.NMSBoxes expects [x, y, w, h] format
    xywh_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        xywh_boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])

    indices = cv2.dnn.NMSBoxes(xywh_boxes, scores.tolist(), score_threshold, iou_threshold)
    return np.array(indices, dtype=int).flatten() if len(indices) > 0 else np.array([], dtype=int)


"""
Re-ID crop preprocessing.

Converts a detection bounding box into a normalized crop tensor ready
for the Re-ID embedding model (OSNet). This is a distinct concern from
detector preprocessing (src/vision/utils.py) — it operates on crops of
already-detected people, not full frames, and always uses ImageNet
normalization regardless of which detector produced the box.

Used by: src/engine/pipeline.py (embedder_worker)
"""

# OSNet / ImageNet normalization constants
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Minimum bbox dimensions (pixels) below which a crop is considered too
# small/unreliable for Re-ID — these people get flagged via `flag=True`
# and skipped from embedding, but still tracked.
_MIN_BBOX_DIM = 20


def preprocess_crop(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    model_input_size: tuple[int, int],
    torso_ratio: float = 1.0,
) -> Tuple[np.ndarray, Tuple[int, int, int, int], Tuple[float, float], int, int, bool]:
    """
    Crop a detection bounding box from a frame and preprocess it for Re-ID.

    Args:
        frame: Full BGR frame (HxWx3)
        bbox: (x1, y1, x2, y2) detection box in frame coordinates
        model_input_size: (H, W) expected by the Re-ID model, e.g. (256, 128)
        torso_ratio: Fraction of the bbox height to keep, measured from the
            top. 1.0 = full body. 0.5 = upper half only (useful when legs
            are frequently occluded by shelves/counters).

    Returns:
        Tuple of:
            input_tensor: (3, H, W) float32, ImageNet-normalized, ready for OSNet
            crop_box: (x1, y1, x2, y2) as ints — the actual region cropped
            center_point: (cx, cy) center of the original bbox
            bbox_w: width of the original bbox
            bbox_h: height of the original bbox
            flag: True if this crop should be SKIPPED for embedding
                  (too small or empty) — caller checks `if not flag`

    Raises:
        PreprocessError: If the crop produces an invalid tensor shape
    """
    x1, y1, x2, y2 = map(int, bbox)
    bbox_w, bbox_h = x2 - x1, y2 - y1
    center_point = ((x1 + x2) / 2, (y1 + y2) / 2)

    # Apply torso ratio — keep only the top portion of the box
    crop_y2 = int(y1 + bbox_h * torso_ratio)
    crop = frame[y1:crop_y2, x1:x2]

    # Flag unusable crops: empty array or below minimum size.
    # Caller must check this flag BEFORE using input_tensor — when flag is
    # True, input_tensor is a zero-filled placeholder, not a real crop.
    too_small = bbox_w < _MIN_BBOX_DIM or bbox_h < _MIN_BBOX_DIM
    is_empty = crop.size == 0
    flag = too_small or is_empty

    if flag:
        logger.debug(
            f"Skipping crop: bbox=({x1},{y1},{x2},{y2}) "
            f"size={bbox_w}x{bbox_h} empty={is_empty}"
        )
        # Return a correctly-shaped placeholder so callers that build a
        # batch array don't need special-case shape handling.
        placeholder = np.zeros((3, model_input_size[0], model_input_size[1]), dtype=np.float32)
        return placeholder, (x1, y1, x2, y2), center_point, bbox_w, bbox_h, flag

    try:
        resized = cv2.resize(
            crop, (model_input_size[1], model_input_size[0]),  # cv2 wants (W, H)
            interpolation=cv2.INTER_AREA
        )

        normalized = (resized.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        input_tensor = np.transpose(normalized, (2, 0, 1)).astype(np.float32)  # HWC -> CHW

        return input_tensor, (x1, y1, x2, y2), center_point, bbox_w, bbox_h, flag

    except Exception as e:
        raise PreprocessError(
            "Re-ID crop preprocessing failed",
            context={"bbox": bbox, "crop_shape": crop.shape, "error": str(e)}
        ) from e