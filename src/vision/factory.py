from loguru import logger
from src.core.config import settings, ModelType
from src.vision.models.yolox import YoloXDetector
from src.vision.models.hf import HFTransformerDetector
from src.vision.models.openvino import OpenVinoDetector
from src.vision.models.ultralytics_yolo import UltralyticsDetector
from src.vision.models.ultralytics_yolo_onnx import UltralyticsONNXDetector

from src.vision.utils import resolve_model_path
from src.core.exceptions import ModelResolutionError # Import our custom exception


def get_detector():
    """
    Factory: Returns the instantiated Model Class based on settings.
    Handles model path resolution (local/download) before instantiation.
    """
    # model_id = settings.MODEL_ID
    arch = settings.MODEL_ARCH
    conf = settings.CONF_THRESHOLD
    device = settings.DEVICE.value # Get the string value from Enum
    iou_thres = settings.IOU_THRES
    nms_thres = settings.NMS_THRESHOLD
    try:
        model_local_path = resolve_model_path()
        logger.info(f"Resolved model path: {model_local_path}")
    except ModelResolutionError as e:
        logger.critical(f"🔥 FATAL MODEL CONFIG ERROR: {e}")
        raise # Re-raise the error to stop application startup
    
    if arch == ModelType.YOLOX:
    # if 'yolox' in model_id.lower():
        return YoloXDetector(
            model_path=model_local_path, 
            conf_thresh=conf,
            nms_thresh=nms_thres,
            class_agnostic=settings.CLASS_AGNOSTIC
        )
    elif arch in [ModelType.RFDETR, ModelType.DFINE]:
    # elif 'onnx-community' in model_id.lower():
        return HFTransformerDetector(
            model_path=model_local_path, 
            conf_thresh=conf
        )
    elif arch == ModelType.OPENVINO:
    # elif 'openvino' in model_id.lower():        
        return OpenVinoDetector(
            model_path=model_local_path, 
            conf_thresh=conf, 
            device=device.upper()
        )
    elif arch == ModelType.ULTRALYTICS:
    # elif 'ultralytics' in model_id.lower():
        return UltralyticsDetector(
            model_path=model_local_path, 
            conf_thresh=conf, 
            device=device
        )
    
    elif arch == ModelType.YOLO_ONNX:
    # elif 'ultralytics' in model_id.lower():
        return UltralyticsONNXDetector(
            model_path=model_local_path, 
            conf_thresh=conf,
            iou_thres=iou_thres, 
            device=device
        )
    
    else:
        raise ModelResolutionError(f"Internal Error: Unknown or unsupported Model Architecture: '{arch.value}'. ")