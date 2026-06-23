# import os
# from src.core.config import settings

# import os
# import re
# import shutil
# from functools import lru_cache

# import requests
# from loguru import logger
# from huggingface_hub import hf_hub_download

# from src.core.config import settings, ModelType
# from src.core.exceptions import ModelResolutionError, ModelDownloadError, ModelConfigError


# def resolve_model_path() -> str:
#     """
#     Resolve settings.MODEL_ID / settings.MODEL_ARCH into an absolute local
#     path to a usable model file, downloading it if necessary.

#     Cached with lru_cache — settings don't change at runtime, and this
#     touches the filesystem/network, so repeated calls (e.g. one per camera
#     using the same model) should not re-resolve from scratch.

#     Returns:
#         Absolute path to the model file (guaranteed to exist on success)

#     Raises:
#         ModelConfigError: If MODEL_ID/MODEL_ARCH settings are invalid
#         ModelDownloadError: If the model needs downloading and that fails
#         ModelResolutionError: For any other unexpected resolution failure
#     """
#     model_id = settings.MODEL_ID
#     model_arch = settings.MODEL_ARCH
#     filename = settings.HF_MODEL_FILENAME or "onnx/model_quantized.onnx"

#     try:
#         relative_path, model_arch = model_id_provider(model_id, model_arch)
#     except ModelConfigError:
#         raise  # already has a clear message, just propagate
#     # HuggingFace models (DFINE/RFDETR) live under a shared repo directory
#     repo_id = relative_path.rsplit("/", 1)[0]
#     repo_name = settings.HF_MODEL_REPONAME
#     print(11111111111111111111, relative_path, model_arch, repo_id, repo_name, model_id, model_arch, filename)
#     # 11111111111111111111 onnx-community/rfdetr_nano-ONNX ModelType.DFINE onnx-community onnx-community rfdetr_nano-ONNX ModelType.DFINE onnx/model_quantized.onnx
#     if repo_name and model_arch in (ModelType.DFINE, ModelType.RFDETR):
#         relative_path = relative_path.replace(repo_id, repo_name)
#         repo_id = repo_name

#     main_local_dir = os.path.join(settings.BASE_DIR, "models", relative_path)
#     final_local_dir = os.path.join(settings.BASE_DIR, "models", relative_path, filename.split('/')[0])
#     model_file_path = os.path.join(settings.BASE_DIR, "models", relative_path, filename)
#     print(333333333333333333, final_local_dir, model_file_path)
#     # 333333333333333333 /home/esi/unbreakable-eye/models/onnx-community/rfdetr_nano-ONNX/onnx /home/esi/unbreakable-eye/models/onnx-community/rfdetr_nano-ONNX/onnx/model_quantized.onnx
#     #CHECK IF the model exists first:
#     if os.path.exists(model_file_path):
#         print(111111)
#         return model_file_path
#     else:
#         print(222222222222)
#         os.makedirs(final_local_dir, exist_ok=True)
#         return download_from_hf(relative_path, main_local_dir, model_file_path, filename)

# def download_from_hf(repo_id: str, final_local_dir: str, model_file_path: str, full_filename: str) -> str:
#     subfolder, filename = full_filename.split('/')
#     print(final_local_dir) # /home/esi/unbreakable-eye/models/onnx-community/rfdetr_nano-ONNX/onnx
#     downloaded_path = hf_hub_download(
#                 repo_id=repo_id,
#                 filename=filename,
#                 subfolder=subfolder,
#                 cache_dir=os.path.join(settings.BASE_DIR, "hf_cache"),
#                 local_dir=final_local_dir,
#                 local_dir_use_symlinks=False,
#         )
#     return model_file_path # /home/esi/unbreakable-eye/models/onnx-community/rfdetr_nano-ONNX/onnx/model_quantized.onnx

# # 11111111111111111111
# # onnx-community/rfdetr_small-ONNX
# # ModelType.DFINE
# # onnx-community
# # onnx-community
# # rfdetr_small-ONNX
# # ModelType.DFINE
# # onnx/model_quantized.onnx

# # 333333333333333333
# # /home/esi/unbreakable-eye/models/onnx-community/rfdetr_small-ONNX/onnx
# # /home/esi/unbreakable-eye/models/onnx-community/rfdetr_small-ONNX/onnx/model_quantized.onnx
# # 222222222222
# # models/onnx-community/rfdetr_nano-ONNX/onnx/model_quantized.onnx/model_quantized.onnx


# # models/onnx-community/rfdetr_small-ONNX/onnx/model_quantized.onnx

# # HF_MODEL_FILENAME: str =  "onnx/model_quantized.onnx"
# # HF_MODEL_REPONAME: str =  "onnx-community"

# # os.path.join(final_local_dir, model_id, filename)

# # 11111111111111111111
# # onnx-community/rfdetr_small-ONNX
# # ModelType.DFINE
# # onnx-community
# # onnx-community
# # rfdetr_small-ONNX
# # ModelType.DFINE
# # onnx/model_quantized.onnx

# # 333333333333333
# # /home/esi/unbreakable-eye/models/onnx-community/rfdetr_small-ONNX/
# # /home/esi/unbreakable-eye/models/onnx-community/rfdetr_small-ONNX/onnx/model_quantized.onnx



# # 11111111111111111111 
# # onnx-community/rfdetr_small-ONNX
# # ModelType.DFINE
# # onnx-community
# # onnx-community
# # rfdetr_small-ONNX
# # ModelType.DFINE
# # 333333333333333
# # /home/esi/unbreakable-eye/models/onnx-community
# # /home/esi/unbreakable-eye/models/onnx-community/rfdetr_small-ONNX


# def model_id_provider(model_id: str, model_arch: ModelType | None) -> tuple[str, ModelType]:
#     """
#     Resolve (MODEL_ID, MODEL_ARCH) settings into a relative model path
#     of the form "{arch}/{filename}".

#     This function exists because users configure models in .env in several
#     different ways depending on what they already know:
#       - Only MODEL_ARCH set        -> use the default filename for that arch
#       - MODEL_ID is just a filename + MODEL_ARCH set -> combine them
#       - MODEL_ID is "arch/filename" -> use as-is (or override with MODEL_ARCH)
#       - MODEL_ID is just an arch name (e.g. "ultralytics") -> treat as arch selector

#     Args:
#         model_id: Raw MODEL_ID string from settings (may be empty)
#         model_arch: MODEL_ARCH enum from settings (may be None)

#     Returns:
#         (relative_path, resolved_arch) e.g. ("ultralytics/yolov8n.pt", ModelType.ULTRALYTICS)

#     Raises:
#         ModelConfigError: If the combination of settings is ambiguous or invalid
#     """
#     mid = model_id.strip() if model_id else ""
#     valid_arch_dirs = {m.value for m in ModelType}

#     # ── Case 1: No MODEL_ID at all ──
#     if not mid:
#         if model_arch is None:
#             raise ModelConfigError(
#                 "Both MODEL_ID and MODEL_ARCH are empty in settings. "
#                 "Set at least MODEL_ARCH (e.g. MODEL_ARCH=ultralytics)."
#             )
#         filename = _infer_default_filename(model_arch)
#         return f"{model_arch.value}/{filename}", model_arch

#     parts = mid.rsplit("/", 1)

#     # ── Case 2: MODEL_ID has no "/" — it's either a bare filename or an arch name ──
#     if len(parts) == 1:
#         if mid in valid_arch_dirs:
#             # MODEL_ID is itself an arch name, e.g. MODEL_ID="ultralytics"
#             final_arch = model_arch if model_arch is not None else ModelType(mid)
#             filename = _infer_default_filename(final_arch)
#             return f"{final_arch.value}/{filename}", final_arch

#         # MODEL_ID is a bare filename, e.g. MODEL_ID="yolov8n.pt"
#         if model_arch is None:
#             raise ModelConfigError(
#                 f"MODEL_ID='{mid}' looks like a filename, but MODEL_ARCH is not set "
#                 f"so the model folder cannot be determined. "
#                 f"Either set MODEL_ARCH, or use MODEL_ID='<arch>/{mid}'.",
#                 context={"model_id": mid}
#             )
#         return f"{model_arch.value}/{mid}", model_arch

#     # ── Case 3: MODEL_ID contains "/" — "folder/filename" or "org/repo" ──
#     current_dir, current_file = parts[0], parts[1]

#     if model_arch is not None:
#         # Explicit MODEL_ARCH always wins over whatever folder is in MODEL_ID
#         return f"{model_arch.value}/{current_file}", model_arch

#     if current_dir in valid_arch_dirs:
#         return mid, ModelType(current_dir)

#     # Unknown folder and no MODEL_ARCH — this is a configuration error, not
#     # something we should silently paper over by defaulting to OpenVINO.
#     raise ModelConfigError(
#         f"MODEL_ID='{mid}' has folder '{current_dir}' which is not a known "
#         f"architecture, and MODEL_ARCH is not set. "
#         f"Known architectures: {sorted(valid_arch_dirs)}",
#         context={"model_id": mid, "unknown_folder": current_dir}
#     )

# # def download_from_hf(repo_id: str, final_local_dir: str, model_file_path: str) -> str:
# #     # filename = settings.HF_MODEL_FILENAME or "model_quantized.onnx"
# #     downloaded_path = hf_hub_download(
# #                 repo_id=repo_id,
# #                 filename=filename,
# #                 subfolder="onnx",          # the file lives inside the 'onnx' folder
# #                 cache_dir=os.path.join(settings.BASE_DIR, "hf_cache"),
# #                 local_dir=os.path.join(final_local_dir, repo_id),
# #                 local_dir_use_symlinks=False,
# #         )

# # # model_quantized.onnx
# # # onnx/model_quantized.onnx
# # # /home/esi/unbreakable-eye/models/onnx-community
    
# #     # model_path = hf_hub_download(
# #     #     repo_id="onnx-community/dfine_n_coco-ONNX",
# #     #     filename="model_fp16.onnx",
# #     #     subfolder="onnx",          # the file lives inside the 'onnx' folder
# #     #     local_dir="./models/onnx-community/dfine_n_coco-ONNX/",      # where to save it (optional)
# #     #     local_dir_use_symlinks=False
# #     # )
    
# #     # Move into the expected final location, if different
# #     if os.path.abspath(downloaded_path) != os.path.abspath(model_file_path):
# #         os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
# #         shutil.move(downloaded_path, model_file_path)
# #     return model_file_path


# ss = resolve_model_path()
# print(ss)

# # from huggingface_hub import hf_hub_download

# # model_path = hf_hub_download(
# #     repo_id="onnx-community/dfine_n_coco-ONNX",
# #     filename="model_fp16.onnx",
# #     subfolder="onnx",          # the file lives inside the 'onnx' folder
# #     local_dir="./models/onnx-community/dfine_n_coco-ONNX/",      # where to save it (optional)
# #     local_dir_use_symlinks=False
# # )

# # print(f"Model downloaded to: {model_path}")


# # model_quantized.onnx
# # onnx/model_quantized.onnx
# # /home/esi/unbreakable-eye/models/onnx-community

















































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

model = HFTransformerDetector(
        model_path=model_local_path, 
        conf_thresh=conf
    )

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

# print(333333333333, frame.shape)
detections = model.predict(frame)
# print(detections[0])
# print(detections[1])

t2 = time.time()
# detections = detections[detections.class_id == 0]

labels = []
for det in detections:
    if len(det) > 0:
        _, _, det_conf, _, tracker_id, _ = det
        labels.append(f"#Confidence: {float(det_conf):.3f}")
            
if len(labels)>0:            
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



