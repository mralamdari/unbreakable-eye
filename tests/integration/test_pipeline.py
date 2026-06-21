from loguru import logger
from src.core.config import settings, ModelType
from src.vision.detectors.yolox import YOLOXDetector
from src.vision.model_resolver import resolve_model_path
from src.vision.detectors.hf import HFTransformerDetector
from src.vision.detectors.openvino import OpenVinoDetector
from src.vision.detectors.ultralytics_yolo import UltralyticsDetector
from src.vision.detectors.ultralytics_yolo_onnx import UltralyticsONNXDetector
import time

arch = settings.MODEL_ARCH
conf = settings.CONF_THRESHOLD
device = settings.DEVICE.value # Get the string value from Enum
iou_thres = settings.IOU_THRES
nms_thres = settings.NMS_THRESHOLD
t0 = time.time()
model_local_path = resolve_model_path()
t1 = time.time()
print(f"1: Model Path Accuaired in : {t1- t0:.4f} Seconds")
print(2020202020, model_local_path)
# model = YOLOXDetector(
#         model_path=model_local_path, 
#         conf_thresh=conf,
#         nms_thresh=nms_thres,
#         class_agnostic=settings.CLASS_AGNOSTIC)
# model = HFTransformerDetector(
#         model_path=model_local_path, 
#         conf_thresh=conf
#     )

model =  OpenVinoDetector(
        model_path=model_local_path, 
        conf_thresh=conf, 
        device=device.upper()
    )
# model =  UltralyticsDetector(
#         model_path=model_local_path, 
#         conf_thresh=conf, 
#         device=device
#     )

# model = UltralyticsONNXDetector(
#         model_path=model_local_path, 
#         conf_thresh=conf,
#         iou_thres=iou_thres, 
#         device=device
#     )
t2 = time.time()
print(f"2: Model Loaded in : {t2- t1:.4f} Seconds")

import cv2
import numpy as np
import supervision as sv
color = sv.ColorPalette.DEFAULT
box_annotator = sv.BoxAnnotator(color=color)
label_annotator = sv.LabelAnnotator(color=color, text_color=sv.Color.BLACK)

frame = cv2.imread('data/test.png')
detections = model.predict(frame)

# print(333333333333, frame.shape)
detections = model.predict(frame)
# print(detections[0])
# print(detections[1])

t2 = time.time()
# detections = detections[detections.class_id == 0]

labels = []
for det in detections:
    print(det)
    _, _, det_conf, _, tracker_id, _ = det
    labels.append(f"#Confidence: {float(det_conf):.3f}")
            
            
annotated = frame.copy()
print('aaaaaaaaaaaaaaaaa')
annotated = box_annotator.annotate(annotated, detections)
annotated = label_annotator.annotate(annotated, detections, labels)
t3 = time.time()
print(f"3: Model Predicted and Annotated in : {t3- t2:.4f} Seconds")

out_path = 'data/output.png'
cv2.imwrite(out_path, annotated)
t4 = time.time()
print(f"4: Output.png is Written in : {t4-t3:.4f} Seconds")

# ── Print results ─────────────────────────────────────────────────────────
for det in detections:
    xyxy, _, det_conf, tracker_id, _, _ = det
    print(f"tracker_id: {tracker_id}  score={det_conf:.3f}")

print(f"\nSaved result → {out_path}")



