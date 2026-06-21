
# ## Production-Ready Checklist
# # - ✅ All method signatures correct and type-hinted
# # - ✅ All imports present
# # - ✅ Custom exceptions for all failure modes
# # - ✅ Comprehensive logging at INFO, DEBUG, WARNING, ERROR levels
# # - ✅ No dead code or commented sections
# # - ✅ Shared preprocessing utilities in utils.py
# # - ✅ Model-specific logic isolated in detector classes
# # - ✅ Proper error handling with context
# # - ✅ Docstrings with Args/Returns/Raises
# # - ✅ Consistent NMS implementation (with fallback options)
# # - ✅ Unit testable (no side effects, pure functions where possible)
# # - ✅ Performance optimizations (grid pre-calculation in YOLOX, etc.)


# # ## Key Improvements for Your Portfolio

# # When showing this work to potential employers or clients:

# # 1. **Professional Structure**: Each detector follows clean architecture patterns
# # 2. **Error Handling**: Production-grade exception handling with context
# # 3. **Logging**: Structured logging enables debugging in production
# # 4. **Type Safety**: Full type hints across all methods
# # 5. **Documentation**: Complete docstrings for every class and method
# # 6. **Testability**: Code is structured to be unit testable
# # 7. **Extensibility**: Easy to add new backends (OpenVINO was example)
# # 8. **Performance**: Optimizations like YOLOX grid pre-calculation
# # 9. **Code Reuse**: 400+ lines of duplication eliminated
# # 10. **Standards Compliance**: Follows PEP 8, Google Python style guide

# # This is **real production code** that would be acceptable at senior engineer level.




# # ## Testing
# # Each detector should be tested with:
# # 1. Single frame inference
# # 2. Batch inference (if supported)
# # 3. Edge cases (empty frames, small objects, etc.)

# ## Migration Path
# # To switch detectors in production:
# # 1. Change `DETECTOR_BACKEND` in `.env`
# # 2. Ensure model file exists
# # 3. Run unit tests for new backend
# # 4. Deploy with canary rollout (10% traffic → 50% → 100%)
# # 5. Monitor accuracy/latency metrics









# from loguru import logger
# from src.core.config import settings, ModelType
# from src.vision.detectors.yolox import YOLOXDetector
# from src.vision.model_resolver import resolve_model_path
# from src.vision.detectors.hf import HFTransformerDetector
# from src.vision.detectors.openvino import OpenVinoDetector
# from src.vision.detectors.ultralytics_yolo import UltralyticsDetector
# from src.vision.detectors.ultralytics_yolo_onnx import UltralyticsONNXDetector
# import time

# arch = settings.MODEL_ARCH
# conf = settings.CONF_THRESHOLD
# device = settings.DEVICE.value # Get the string value from Enum
# iou_thres = settings.IOU_THRES
# nms_thres = settings.NMS_THRESHOLD
# t0 = time.time()
# model_local_path = resolve_model_path()
# t1 = time.time()
# print(f"1: Model Path Accuaired in : {t1- t0:.4f} Seconds")
# # model = YOLOXDetector(
# #         model_path=model_local_path, 
# #         conf_thresh=conf,
# #         nms_thresh=nms_thres,
# #         class_agnostic=settings.CLASS_AGNOSTIC)
# print(2020202020, model_local_path)
# model = HFTransformerDetector(
#         model_path=model_local_path, 
#         conf_thresh=conf
#     )

# # model =  OpenVinoDetector(
# #         model_path=model_local_path, 
# #         conf_thresh=conf, 
# #         device=device.upper()
# #     )
# # model =  UltralyticsDetector(
# #         model_path=model_local_path, 
# #         conf_thresh=conf, 
# #         device=device
# #     )

# # model = UltralyticsONNXDetector(
# #         model_path=model_local_path, 
# #         conf_thresh=conf,
# #         iou_thres=iou_thres, 
# #         device=device
# #     )
# t2 = time.time()
# print(f"2: Model Loaded in : {t2- t1:.4f} Seconds")

# import cv2
# import numpy as np
# import supervision as sv
# color = sv.ColorPalette.DEFAULT
# box_annotator = sv.BoxAnnotator(color=color)
# label_annotator = sv.LabelAnnotator(color=color, text_color=sv.Color.BLACK)

# frame = cv2.imread('data/test.png')
# detections = model.predict(frame)

# # # print(333333333333, frame.shape)
# # detections = model.predict(frame)
# # # print(detections[0])
# # # print(detections[1])

# # t2 = time.time()
# # # detections = detections[detections.class_id == 0]

# # labels = []
# # for det in detections:
# #     # print(det)
# #     _, _, det_conf, _, tracker_id, _ = det
# #     labels.append(f"#Confidence: {float(det_conf):.3f}")
            
            
# # annotated = frame.copy()
# # print('aaaaaaaaaaaaaaaaa')
# # annotated = box_annotator.annotate(annotated, detections)
# # annotated = label_annotator.annotate(annotated, detections, labels)
# # t3 = time.time()
# # print(f"3: Model Predicted and Annotated in : {t3- t2:.4f} Seconds")

# # cv2.imwrite('data/output.png', annotated)
# # t4 = time.time()
# # print(f"4: Output.png is Written in : {t4-t3:.4f} Seconds")





# https://huggingface.co/onnx-community/rfdetr_base-ONNX/resolve/main/onnx/model_uint8.onnx
# https://huggingface.co/onnx-community/rfdetr_small-ONNX/resolve/main/onnx/model_bnb4.onnx
# https://huggingface.co/onnx-community/rfdetr_nano-ONNX/resolve/main/onnx/model_int8.onnx
# https://huggingface.co/onnx-community/rfdetr_medium-ONNX/resolve/main/onnx/model_q4.onnx

# https://huggingface.co/onnx-community/dfine_n_coco-ONNX/resolve/main/onnx/model_fp16.onnx
# https://huggingface.co/onnx-community/dfine_s_coco-ONNX/resolve/main/onnx/model_uint8.onnx
# https://huggingface.co/onnx-community/dfine_m_coco-ONNX/resolve/main/onnx/model_q4f16.onnx
# https://huggingface.co/onnx-community/dfine_l_coco-ONNX/resolve/main/onnx/model_q4f16.onnx




# # so I used this model to get the latest RFDETRNano
# import cv2
# import supervision as sv
# from rfdetr import RFDETRSmall 
# from rfdetr.assets.coco_classes import COCO_CLASSES

# model = RFDETRSmall()

# frame = cv2.imread('data/test.png')
# detections = model.predict(frame, threshold=0.5)

# labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]

# annotated_image = sv.BoxAnnotator().annotate(detections.metadata["source_image"], detections)
# annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
# cv2.imwrite('data/output.png', annotated_image)

















import argparse
import cv2
import os
from src.core.config import settings
import numpy as np
import onnxruntime as ort
from src.vision.utils import (
        create_session, letterbox,
        to_chw_float32, scale_boxes)

# ── Model constants (from preprocessor_config.json) ──────────────────────────
INPUT_SIZE   = 512          # model expects 512×512
IMG_MEAN     = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD      = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SCORE_THRESH = 0.4          # confidence threshold – lower for more detections
NMS_IOU      = 0.5          # IoU threshold for NMS

# ── COCO 90-class labels (id2label from config.json) ─────────────────────────
COCO_LABELS = {
    1:"person",2:"bicycle",3:"car",4:"motorcycle",5:"airplane",
    6:"bus",7:"train",8:"truck",9:"boat",10:"traffic light",
    11:"fire hydrant",13:"stop sign",14:"parking meter",15:"bench",
    16:"bird",17:"cat",18:"dog",19:"horse",20:"sheep",21:"cow",
    22:"elephant",23:"bear",24:"zebra",25:"giraffe",27:"backpack",
    28:"umbrella",31:"handbag",32:"tie",33:"suitcase",34:"frisbee",
    35:"skis",36:"snowboard",37:"sports ball",38:"kite",39:"baseball bat",
    40:"baseball glove",41:"skateboard",42:"surfboard",43:"tennis racket",
    44:"bottle",46:"wine glass",47:"cup",48:"fork",49:"knife",50:"spoon",
    51:"bowl",52:"banana",53:"apple",54:"sandwich",55:"orange",56:"broccoli",
    57:"carrot",58:"hot dog",59:"pizza",60:"donut",61:"cake",62:"chair",
    63:"couch",64:"potted plant",65:"bed",67:"dining table",70:"toilet",
    72:"tv",73:"laptop",74:"mouse",75:"remote",76:"keyboard",77:"cell phone",
    78:"microwave",79:"oven",80:"toaster",81:"sink",82:"refrigerator",
    84:"book",85:"clock",86:"vase",87:"scissors",88:"teddy bear",
    89:"hair drier",90:"toothbrush",
}

# ── Random stable colors per class ───────────────────────────────────────────
rng = np.random.default_rng(42)
CLASS_COLORS = {cid: tuple(int(c) for c in rng.integers(80, 230, 3)) for cid in COCO_LABELS}


def preprocess(image_bgr: np.ndarray):
    """
    Resize to 512×512, convert BGR→RGB, rescale [0,1], normalize with
    ImageNet mean/std. Returns (blob, orig_h, orig_w).
    """
    orig_h, orig_w = image_bgr.shape[:2]

    # Resize keeping aspect ratio with letterboxing  ← key fix for correct bboxes
    scale  = INPUT_SIZE / max(orig_h, orig_w)
    new_w  = int(round(orig_w * scale))
    new_h  = int(round(orig_h * scale))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Letterbox: pad to 512×512 with grey (value 114)
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    pad_top  = (INPUT_SIZE - new_h) // 2
    pad_left = (INPUT_SIZE - new_w) // 2
    canvas[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized

    # BGR → RGB → float32 → [0,1] → normalize
    rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
    rgb = (rgb - IMG_MEAN) / IMG_STD

    # HWC → NCHW (batch of 1)
    blob = rgb.transpose(2, 0, 1)[np.newaxis, ...]
    return blob, orig_h, orig_w, scale, pad_top, pad_left


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


def postprocess(outputs, orig_h, orig_w, scale, pad_top, pad_left,
                score_thresh=SCORE_THRESH):
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
    keep = scores > score_thresh
    if not keep.any():
        return [], [], []

    scores     = scores[keep]
    class_ids  = class_ids[keep]
    boxes_norm = raw_boxes[keep]

    # cx,cy,w,h → x1,y1,x2,y2  (still normalized 0-1 relative to 512×512 canvas)
    boxes_xyxy = cxcywh_to_xyxy(boxes_norm)

    # Unpad: remove letterbox offset and undo scale to get original pixel coords
    # canvas coords (0-1) → pixel in 512×512 canvas
    boxes_px = boxes_xyxy * INPUT_SIZE

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

    return boxes_px, scores, class_ids


def nms(boxes, scores, class_ids, iou_thresh=NMS_IOU):
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


def draw_detections(image_bgr, boxes, scores, class_ids):
    out = image_bgr.copy()
    for box, score, cid in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        label = COCO_LABELS.get(int(cid), f"cls{cid}")
        color = CLASS_COLORS.get(int(cid), (0, 200, 0))

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        text   = f"{label} {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - baseline - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(out, text, (x1 + 1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def run(model_path: str, image_path: str, score_thresh: float = SCORE_THRESH,
        save_path: str = "result.jpg", show: bool = False):

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading ONNX model: {model_path}")
    num_threads = max(1, (os.cpu_count() or 4) // 2 - 1)
    session = create_session(model_path, num_threads=num_threads)
    input_name = session.get_inputs()[0].name
    input_size = settings.FRAME_SHAPE  # (H, W, C)
    
    
    # sess_opts = ort.SessionOptions()
    # sess_opts.log_severity_level = 3   # suppress verbose ONNX logs
    # providers = ["CPUExecutionProvider"]
    # session   = ort.InferenceSession(model_path,
    #                                  providers=providers)

    inp_name = session.get_inputs()[0].name
    print(f"  Input  : {inp_name}  {session.get_inputs()[0].shape}")
    for o in session.get_outputs():
        print(f"  Output : {o.name}  {o.shape}")

    # ── Load & preprocess image ───────────────────────────────────────────────
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")

    blob, orig_h, orig_w, scale, pad_top, pad_left = preprocess(img_bgr)

    # ── Inference ─────────────────────────────────────────────────────────────
    outputs = session.run(None, {inp_name: blob})

    # ── Decode outputs ────────────────────────────────────────────────────────
    boxes, scores, class_ids = postprocess(
        outputs, orig_h, orig_w, scale, pad_top, pad_left, score_thresh
    )

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\nDetections ({len(boxes)} above threshold {score_thresh}):")
    for box, score, cid in zip(boxes, scores, class_ids):
        label = COCO_LABELS.get(int(cid), f"cls{cid}")
        x1,y1,x2,y2 = map(int,box)
        print(f"  {label:<20s}  score={score:.3f}  box=[{x1},{y1},{x2},{y2}]")

    # ── Draw & save ───────────────────────────────────────────────────────────
    vis = draw_detections(img_bgr, boxes, scores, class_ids)
    cv2.imwrite(save_path, vis)
    print(f"\nSaved result → {save_path}")

    if show:
        cv2.imshow("RF-DETR detections", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return boxes, scores, class_ids


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RF-DETR ONNX inference")
    parser.add_argument("--model",  required=True,       help="Path to .onnx file")
    parser.add_argument("--image",  required=True,       help="Path to input image")

# run('models/onnx-community/dfine_n_coco-ONNX/onnx/model_quantized.onnx', 'data/test.png', 0.5, 'data/output.png')
# run('models/onnx-community/rfdetr_small-ONNX/onnx/model_quantized.onnx', 'data/test.png', 0.5, 'data/output.png')
run('models/onnx-community/rfdetr_small-ONNX/onnx/model_quantized.onnx', 'data/test.png', 0.2, 'data/output.png')
