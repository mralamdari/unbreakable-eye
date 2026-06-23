"""
D-FINE ONNX export + inference (official HuggingFace source of truth)
========================================================================
Official model family: ustc-community/dfine-{nano,small,medium,large,xlarge}-coco
Official repo: https://github.com/Peterande/D-FINE
HF docs: https://huggingface.co/docs/transformers/model_doc/d_fine

WHY YOUR PREVIOUS ATTEMPT LIKELY FAILED
─────────────────────────────────────────────────────────────────────────
D-FINE is part of the same HF model family lineage as RT-DETR (same
image-processor conventions, same use_focal_loss=True default → sigmoid
scoring, NOT softmax). On the surface this looks like RF-DETR's format
too. But three things commonly differ between RF-DETR's actual ONNX
export and D-FINE's, and ANY single mismatch silently produces near-zero
detections:

  1. Input resolution: D-FINE uses 640x640 (not RF-DETR's 504-560-ish,
     14-divisible DINOv2 sizing). Wrong input size alone can crater
     recall on a transformer detector trained at a fixed resolution.
  2. Output tensor ORDER and NAMES differ between exporters. The HF
     model returns (logits, pred_boxes) via .from_pretrained() + forward
     pass, but a raw .onnx export's output order depends entirely on
     HOW it was exported — there's no universal "dets"/"labels" naming
     convention here like Roboflow established for RF-DETR.
  3. Some community ONNX conversions of D-FINE bake in different
     postprocessing assumptions (e.g. softmax instead of sigmoid scoring) —
     see the Node.js D-FINE ONNX runtime example which explicitly uses
     softmax, which would be WRONG for the official HF/Peterande weights.

THE FIX: this script gives you TWO paths so you have a verified ground
truth to validate any future ONNX conversion against.

  PATH A (recommended first): Export ONNX yourself from the official HF
  PyTorch checkpoint, using HF's OWN post_process_object_detection logic
  as the ground truth for what "correct" looks like. This guarantees the
  export and the postprocessing agree, because you control both ends.

  PATH B: Pure ONNX Runtime inference against an already-exported .onnx
  file, with manual sigmoid + cxcywh decoding mirroring HF's
  post_process_object_detection exactly.

─────────────────────────────────────────────────────────────────────────
USAGE
─────────────────────────────────────────────────────────────────────────
    pip install transformers torch onnx onnxruntime pillow opencv-python numpy

    # Step 1: sanity-check with the PyTorch model directly (no ONNX yet)
    # — confirms detections work BEFORE introducing any export risk.
    python dfine_inference.py --mode torch --image shop.jpg --checkpoint ustc-community/dfine-small-coco

    # Step 2: export to ONNX from that same verified checkpoint
    python dfine_inference.py --mode export --checkpoint ustc-community/dfine-small-coco --onnx-out dfine_small.onnx

    # Step 3: run pure ONNX Runtime inference on the exported file
    python dfine_inference.py --mode onnx --image shop.jpg --onnx dfine_small.onnx
"""

import argparse

import cv2
import numpy as np

import numpy as np
import cv2
from PIL import Image
from transformers import AutoImageProcessor
import torch


# ── COCO 80-class labels (same set D-FINE/RT-DETR/YOLOX all use) ────────────
COCO_CLASSES = (
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
    "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","wine glass","cup",
    "fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush",
)

INPUT_SIZE = 640  # D-FINE's standard training/export resolution
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def run_torch_reference(checkpoint: str, image_path: str, conf_thresh: float = 0.4,
                          save_path: str = "result_torch.jpg"):
    """
    Run D-FINE directly via transformers, using HF's OWN
    post_process_object_detection — this is the ground-truth decode logic,
    guaranteed correct because it's maintained by the model's own authors.

    Use this FIRST to confirm the model itself detects people in your
    image before introducing any ONNX export risk.
    """
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, DFineForObjectDetection

    print(f"Loading {checkpoint} ...")
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = DFineForObjectDetection.from_pretrained(checkpoint)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    print(f"logits shape: {tuple(outputs.logits.shape)}")
    print(f"pred_boxes shape: {tuple(outputs.pred_boxes.shape)}")

    target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
    results = image_processor.post_process_object_detection(
        outputs, threshold=conf_thresh, target_sizes=target_sizes
    )[0]

    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    print(f"\nDetections (threshold={conf_thresh}):")
    for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
        score = score.item()
        label_id = label_id.item()
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        label = model.config.id2label.get(label_id, f"cls{label_id}")
        print(f"  {label:<16s}  score={score:.3f}  box=[{x1},{y1},{x2},{y2}]")

        color = (60, 200, 60)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {score:.2f}"
        cv2.putText(img_cv, text, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    cv2.imwrite(save_path, img_cv)
    print(f"\nSaved -> {save_path}")
    print(
        "\nIf this PATH A run shows correct detections but your ONNX run "
        "(PATH B) doesn't, the bug is specifically in the ONNX export or "
        "its postprocessing — not in the model or your input image."
    )

def postprocess(logits: np.ndarray, boxes: np.ndarray, img_h: int, img_w: int,
                 conf_thresh: float = 0.4):
    """
    Mirrors RTDetrImageProcessor.post_process_object_detection with
    use_focal_loss=True (D-FINE's default): SIGMOID scoring (not softmax),
    cxcywh -> xyxy, scale by the ORIGINAL image size directly (no
    letterbox unpadding needed since preprocessing was a plain resize).

    Args:
        logits: (num_queries, num_classes) raw logits for ONE image
                 (already indexed out of the batch dimension).
        boxes:  (num_queries, 4) cx,cy,w,h normalized [0,1].
        img_h, img_w: original image dimensions (NOT the 640x640 input).
        conf_thresh: confidence threshold.

    Returns:
        (boxes_xyxy_px, scores, class_ids) — all numpy arrays, possibly empty.
    """
    probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid — use_focal_loss=True default
    class_ids = np.argmax(probs, axis=1)
    scores = probs[np.arange(len(class_ids)), class_ids]

    keep = scores > conf_thresh
    if not np.any(keep):
        return np.zeros((0, 4), dtype=np.float32), np.array([]), np.array([], dtype=int)

    scores = scores[keep]
    class_ids = class_ids[keep]
    boxes = boxes[keep]

    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, img_w)
    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, img_h)

    return boxes_xyxy, scores, class_ids

from src.vision.utils import create_session, bgr_to_rgb, normalize_imagenet, nms_numpy, cxcywh_to_xyxy

def preprocess(frame: np.ndarray, box_format='') -> np.ndarray:
    img = bgr_to_rgb(frame)
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)

    if box_format == "cxcywh_input_normalized":
        img = img.astype(np.float32) / 255.0
        # D-FINE: rescale_factor=1/255, do_normalize=FALSE
        # Matches AutoImageProcessor("ustc-community/dfine-*") exactly.
    else:
        # RF-DETR: /255 + ImageNet mean/std (do_normalize=true)
        img = normalize_imagenet(img)

    img = np.transpose(img, (2, 0, 1))[np.newaxis]       # HWC → NCHW
    return np.ascontiguousarray(img, dtype=np.float32)


def run_onnx_inference(onnx_path: str, image_path: str, conf_thresh: float = 0.4,
                         save_path: str = "result_onnx.jpg", box_format=''):
    """
    Pure ONNX Runtime inference — no torch/transformers dependency needed
    at inference time. Use this once PATH A has confirmed the model and
    your export are both correct.
    """
    import onnxruntime as ort

    print(f"Loading ONNX model: {onnx_path}")
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3
    session = ort.InferenceSession(onnx_path, sess_options=sess_opts,
                                    providers=["CPUExecutionProvider"])

    inp = session.get_inputs()[0]
    print(f"  Input  : {inp.name}  {inp.shape}")
    for o in session.get_outputs():
        print(f"  Output : {o.name}  {o.shape}")

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")

    # blob, orig_h, orig_w = preprocess(img)
    orig_h, orig_w = img.shape[:2]
    blob= preprocess(img, box_format=box_format)
    
    # ---- Config ----
    # checkpoint = "ustc-community/dfine-small-coco"
    # # ---- Reference: HuggingFace pipeline ----
    # image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    # pil_image = Image.open(image_path).convert("RGB")
    # inputs = image_processor(images=pil_image, return_tensors="pt")
    # blob = inputs["pixel_values"].numpy().astype(np.float32)   # shape (1, 3, 640, 640)

    outputs = session.run(None, {inp.name: blob})

    # Identify logits vs boxes by last-dim shape (4 = boxes)
    out0, out1 = outputs[0], outputs[1]
    if out0.shape[-1] == 4:
        boxes, logits = out0, out1
    else:
        logits, boxes = out0, out1

    boxes_xyxy, scores, class_ids = postprocess(
        logits[0], boxes[0], orig_h, orig_w, conf_thresh
    )

    print(f"\nDetections (threshold={conf_thresh}): {len(boxes_xyxy)}")
    for box, score, cid in zip(boxes_xyxy, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        label = COCO_CLASSES[int(cid)] if int(cid) < len(COCO_CLASSES) else f"cls{cid}"
        print(f"  {label:<16s}  score={score:.3f}  box=[{x1},{y1},{x2},{y2}]")

    for box, score, cid in zip(boxes_xyxy, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        label = COCO_CLASSES[int(cid)] if int(cid) < len(COCO_CLASSES) else f"cls{cid}"
        color = (60, 200, 60)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{label} {score:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    cv2.imwrite(save_path, img)
    print(f"\nSaved -> {save_path}")

    if len(boxes_xyxy) == 0:
        print(
            "\nZero detections. If PATH A (torch reference) DID find "
            "people on this same image, compare:\n"
            "  1. Input size used at export time (must be 640x640)\n"
            "  2. Output order — check the printed Output shapes above; "
            "if logits/boxes got swapped, scores will be garbage\n"
            "  3. Whether your earlier RF-DETR-based code used a "
            "DIFFERENT input size or normalization than this script"
        )


run_onnx_inference('models/onnx-community/rfdetr_small-ONNX/onnx/model_quantized.onnx', 'data/test.png', 0.4, save_path='data/RFDetr_Small_onnx.png',box_format='')    
run_onnx_inference('models/onnx-community/dfine_s_coco-ONNX/onnx/model_quantized.onnx', 'data/test.png', 0.4, save_path='data/Dfine_Small_onnx.png', box_format='cxcywh_input_normalized')    
# run_torch_reference('ustc-community/dfine-small-coco', 'data/test.png', 0.2)    
