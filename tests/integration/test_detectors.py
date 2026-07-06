
# # ## Production-Ready Checklist
# # # - ✅ All method signatures correct and type-hinted
# # # - ✅ All imports present
# # # - ✅ Custom exceptions for all failure modes
# # # - ✅ Comprehensive logging at INFO, DEBUG, WARNING, ERROR levels
# # # - ✅ No dead code or commented sections
# # # - ✅ Shared preprocessing utilities in utils.py
# # # - ✅ Model-specific logic isolated in detector classes
# # # - ✅ Proper error handling with context
# # # - ✅ Docstrings with Args/Returns/Raises
# # # - ✅ Consistent NMS implementation (with fallback options)
# # # - ✅ Unit testable (no side effects, pure functions where possible)
# # # - ✅ Performance optimizations (grid pre-calculation in YOLOX, etc.)


# # # ## Key Improvements for Your Portfolio

# # # When showing this work to potential employers or clients:

# # # 1. **Professional Structure**: Each detector follows clean architecture patterns
# # # 2. **Error Handling**: Production-grade exception handling with context
# # # 3. **Logging**: Structured logging enables debugging in production
# # # 4. **Type Safety**: Full type hints across all methods
# # # 5. **Documentation**: Complete docstrings for every class and method
# # # 6. **Testability**: Code is structured to be unit testable
# # # 7. **Extensibility**: Easy to add new backends (OpenVINO was example)
# # # 8. **Performance**: Optimizations like YOLOX grid pre-calculation
# # # 9. **Code Reuse**: 400+ lines of duplication eliminated
# # # 10. **Standards Compliance**: Follows PEP 8, Google Python style guide

# # # This is **real production code** that would be acceptable at senior engineer level.




# # # ## Testing
# # # Each detector should be tested with:
# # # 1. Single frame inference
# # # 2. Batch inference (if supported)
# # # 3. Edge cases (empty frames, small objects, etc.)

# # ## Migration Path
# # # To switch detectors in production:
# # # 1. Change `DETECTOR_BACKEND` in `.env`
# # # 2. Ensure model file exists
# # # 3. Run unit tests for new backend
# # # 4. Deploy with canary rollout (10% traffic → 50% → 100%)
# # # 5. Monitor accuracy/latency metrics









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
model = YOLOXDetector(model_path=model_local_path,
                      conf_thresh=conf,
                      nms_thresh=0.45)
# model = HFTransformerDetector(
#     model_path=model_local_path,
#     conf_thresh=conf,
#     # model_name='rfdetr',
#     model_name='dfine',
#     # input_dim=(640,640),
#     input_dim=(540,540),
#     )

# model =  OpenVinoDetector(
#         model_path=model_local_path, 
#         conf_thresh=conf, 
#         device=device.upper()
#     )
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
print(detections)
# detections = detections[detections.class_id == 1]
labels = []
for det in detections:
    _, _, det_conf, _, tracker_id, _ = det
    labels.append(f"#Confidence: {float(det_conf):.3f}")
            
annotated = frame.copy()
annotated = box_annotator.annotate(annotated, detections)
annotated = label_annotator.annotate(annotated, detections, labels)
cv2.imwrite('data/output_4.png', annotated)
print(f"\nSaved result → data/output_4.png")






# ## What Actually Works for a Product Like Yours
# You ship a Docker image. The client installs Docker Desktop (one click on Windows/Mac), then runs your image. 
# This is increasingly standard for professional CV and analytics products — it solves the dependency problem completely.
# Your GPU/CPU variants are just different image tags: `your-product:latest-cpu`, `your-product:latest-cuda12`.
# This is actually what most serious B2B computer vision products do.
