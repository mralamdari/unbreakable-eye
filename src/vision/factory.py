# from loguru import logger
# from src.core.config import settings
# from src.vision.models.yolox import YoloXDetector
# from src.vision.models.hf import HFTransformerDetector
# from src.vision.models.openvino import OpenVinoDetector
# from src.vision.models.ultralytics_yolo import UltralyticsDetector

# def get_detector():
#     """
#     Factory: Returns the instantiated Model Class.
#     """
#     arch  = settings.MODEL_ARCH
#     path  = settings.absolute_model_path
#     conf  = settings.CONF_THRESHOLD
#     device = settings.DEVICE
    
#     # Log the Decision
#     logger.info(f"🏭 Factory Request: Arch={arch}, Device={device}")
#     logger.debug(f"📂 Loading Model from: {path}")
    
#     try:
#         if arch == "yolox":
#             return YoloXDetector(
#             model_path=path, 
#             conf_thresh=conf,
#             nms_thresh=settings.NMS_THRESHOLD,
#             class_agnostic=settings.CLASS_AGNOSTIC
#         )
        
#         elif arch in ["rtdetr", "dfine"]:
#             return HFTransformerDetector(
#             model_path=path, 
#             conf_thresh=conf
#         )
        
#         elif arch == "openvino":
#             return OpenVinoDetector(
#             model_path=path, 
#             conf_thresh=conf, 
#             device=device
#         )

#         elif 'yolo' in arch:
#             return UltralyticsDetector(
#             model_path=path, 
#             conf_thresh=conf, 
#             device=device
#         )
        
#         else:
#             raise ValueError(f"Unknown Architecture: {arch}")

#     except Exception as e:
#         logger.error(f"❌ Failed to load model {arch}: {e}")
#         raise e
    
    
    
    



# from loguru import logger
# from src.core.config import settings
# from src.vision.models.yolox import YoloXDetector
# from src.vision.models.hf import HFTransformerDetector
# from src.vision.models.openvino import OpenVinoDetector
# from src.vision.models.ultralytics_yolo import UltralyticsDetector
# from src.vision.utils import resolve_model_path # <--- NEW IMPORT


# def get_detector():
#     """
#     Factory: Returns the instantiated Model Class.
#     """
#     arch  = settings.MODEL_ARCH
#     # path  = settings.absolute_model_path
#     conf  = settings.CONF_THRESHOLD
#     device = settings.DEVICE
#     model_id = settings.MODEL_ID
    
#     # --- The Centralized Model Resolution ---
#     model_local_path = resolve_model_path(model_id, arch)
#     # ----------------------------------------
    
#     # Log the Decision
#     logger.info(f"🏭 Factory Request: Arch={arch}, Device={device}")
#     logger.debug(f"📂 Loading Model from: {model_local_path}")
    
#     try:
#         if arch == "yolox":
#             return YoloXDetector(
#             model_path=model_local_path, 
#             conf_thresh=conf,
#             nms_thresh=settings.NMS_THRESHOLD,
#             class_agnostic=settings.CLASS_AGNOSTIC
#         )
        
#         elif arch in ["rtdetr", "dfine"]:
#             return HFTransformerDetector(
#             model_path=model_local_path, 
#             conf_thresh=conf
#         )
        
#         elif arch == "openvino":
#             return OpenVinoDetector(
#             model_path=model_local_path, 
#             conf_thresh=conf, 
#             device=device
#         )

#         elif 'yolo' in arch:
#             return UltralyticsDetector(
#             model_path=model_local_path, 
#             conf_thresh=conf, 
#             device=device
#         )
        
#         else:
#             raise ValueError(f"Unknown Architecture: {arch}")

#     except Exception as e:
#         logger.error(f"❌ Failed to load model {arch}: {e}")
#         raise e
        
        





















from loguru import logger
from src.core.config import settings
from src.vision.models.yolox import YoloXDetector
from src.vision.models.hf import HFTransformerDetector
from src.vision.models.openvino import OpenVinoDetector
from src.vision.models.ultralytics_yolo import UltralyticsDetector
from src.vision.utils import resolve_model_path

def get_detector():
    """
    Factory: Returns the instantiated Model Class based on settings.
    Handles model path resolution (local/download) before instantiation.
    """
    model_id = settings.MODEL_ID
    conf = settings.CONF_THRESHOLD
    device = settings.DEVICE.value # Get the string value from Enum
    
    # --- Resolve the model path (download if needed) ---
    model_local_path = resolve_model_path(
        model_id=model_id,
    )
    logger.info(f"Resolved model path: {model_local_path}")
    # ----------------------------------------------------

    if 'yolox' in model_id.lower():
        return YoloXDetector(
            model_path=model_local_path, 
            conf_thresh=conf,
            nms_thresh=settings.NMS_THRESHOLD,
            class_agnostic=settings.CLASS_AGNOSTIC
        )
    elif 'onnx-community' in model_id.lower():
        return HFTransformerDetector(
            model_path=model_local_path, 
            conf_thresh=conf
        )    
    elif 'openvino' in model_id.lower():        
        return OpenVinoDetector(
            model_path=model_local_path, 
            conf_thresh=conf, 
            device=device
        )

    elif 'ultralytics' in model_id.lower():
        return UltralyticsDetector(
            model_path=model_local_path, 
            conf_thresh=conf, 
            device=device
        )
    
    else:
        # This case should ideally not be reached if ModelType enum is comprehensive
        raise ValueError(f"Unknown or unsupported Model Architecture: {model_id}")