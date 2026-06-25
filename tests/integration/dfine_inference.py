import argparse
import cv2
import numpy as np
import numpy as np
import cv2
from PIL import Image
from transformers import AutoImageProcessor
import torch


from src.vision.utils import create_session, bgr_to_rgb, normalize_imagenet, nms_numpy, cxcywh_to_xyxy

def preprocess(frame, input_h, input_w) -> np.ndarray:
    img = bgr_to_rgb(frame)
    img = cv2.resize(
        img, (input_h, input_w), interpolation=cv2.INTER_LINEAR
    )
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[np.newaxis]
    return np.ascontiguousarray(img, dtype=np.float32)

def _split_outputs(outputs: list):
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
    out0, out1 = outputs[0], outputs[1]
    if out0.shape[-1] == 4:
        boxes, logits = out0, out1
    elif out1.shape[-1] == 4:
        logits, boxes = out0, out1
    return boxes[0], logits[0]   # strip batch → (N,4), (N,C)

import supervision as sv
def postprocess(
    boxes,
    logits,
    img_h, img_w,
) -> sv.Detections:

    # Step 1-2: sigmoid → best class per query
    probs     = 1.0 / (1.0 + np.exp(-logits))          # (N, C)
    class_ids = np.argmax(probs, axis=1)                 # (N,)
    scores    = probs[np.arange(len(class_ids)), class_ids]  # (N,)
    keep = scores > 0.4
    if not np.any(keep):
        return sv.Detections.empty()

    scores    = scores[keep]
    class_ids = class_ids[keep]
    boxes     = boxes[keep]

    # Step 4-5: box decode, family-specific
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    boxes_px = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    # Clip to frame boundaries
    boxes_px[:, [0, 2]] = np.clip(boxes_px[:, [0, 2]], 0.0, img_w)
    boxes_px[:, [1, 3]] = np.clip(boxes_px[:, [1, 3]], 0.0, img_h)

    return sv.Detections(
        xyxy=boxes_px.reshape(-1, 4),
        confidence=scores.astype(np.float32),
        class_id=class_ids.astype(int),
    )
def run_onnx_inference(onnx_path: str, image_path: str, conf_thresh: float = 0.4,
                         save_path: str = "result_onnx.jpg"):
    """
    Pure ONNX Runtime inference — no torch/transformers dependency needed
    at inference time. Use this once PATH A has confirmed the model and
    your export are both correct.
    """
    import onnxruntime as ort
    import os
    
    print(f"Loading ONNX model: {onnx_path}")
    num_threads = max(1, (os.cpu_count() or 4) // 2 - 1)
    session = create_session(onnx_path, num_threads=num_threads)
    
    input_name = session.get_inputs()[0].name

    if len(session.get_outputs()) != 2:
        raise ModelLoadError(
            "Expected exactly 2 ONNX outputs (boxes + class scores)",
            context={
                "model_path": model_path,
                "outputs": [o.name for o in session.get_outputs()],
            },
        )
    # ── Input size — read from graph, never hardcode ──────────────
    shape = session.get_inputs()[0].shape   # [1, 3, H, W]
    output_names = [o.name for o in session.get_outputs()]
    
    
    inp = session.get_inputs()[0]
    print(f"  Input  : {inp.name}  {inp.shape}")
    for o in session.get_outputs():
        print(f"  Output : {o.name}  {o.shape}")

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")

    # blob, orig_h, orig_w = preprocess(img)
    orig_h, orig_w = img.shape[:2]
    blob= preprocess(img, 640,640)
    outputs = session.run(None, {inp.name: blob})
    boxes, logits = _split_outputs(outputs)
    detections    = postprocess(boxes, logits, orig_h, orig_w)

run_onnx_inference('models/onnx-community/rfdetr_small-ONNX/onnx/model_quantized.onnx', 'data/test.png', 0.4, save_path='data/RFDetr_Small_onnx.png')    
run_onnx_inference('models/onnx-community/dfine_s_coco-ONNX/onnx/model_quantized.onnx', 'data/test.png', 0.4, save_path='data/Dfine_Small_onnx.png', )    
