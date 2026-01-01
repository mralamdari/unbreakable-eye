from loguru import logger
from src.core.config import settings
from src.vision.models.yolox import YoloXDetector
from src.vision.models.hf import HFTransformerDetector
from src.vision.models.openvino import OpenVinoDetector
from src.vision.models.ultralytics_yolo import UltralyticsDetector

def get_detector():
    """
    Factory: Returns the instantiated Model Class.
    """
    arch  = settings.MODEL_ARCH
    path  = settings.absolute_model_path
    conf  = settings.CONF_THRESHOLD
    device = settings.DEVICE
    
    # Log the Decision
    logger.info(f"🏭 Factory Request: Arch={arch}, Device={device}")
    logger.debug(f"📂 Loading Model from: {path}")
    
    # Common args
    kwargs = {"model_path": path, "conf_thresh": conf}
    try:
        if arch == "yolox":
            return YoloXDetector(
                **kwargs,
                nms_thresh=settings.NMS_THRESHOLD,
                class_agnostic=settings.CLASS_AGNOSTIC
            )
        
        elif arch in ["rtdetr", "dfine"]:
            return HFTransformerDetector(**kwargs)
        
        elif arch == "openvino":
            return OpenVinoDetector(**kwargs, device=device)

        elif 'yolo' in arch:
            return UltralyticsDetector(**kwargs, device=device)
        
        else:
            raise ValueError(f"Unknown Architecture: {arch}")

    except Exception as e:
        logger.error(f"❌ Failed to load model {arch}: {e}")
        raise e