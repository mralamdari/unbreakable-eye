"""
Detector factory.

Reads MODEL_ARCH from settings, resolves the model file path (downloading
if necessary), and returns the corresponding detector instance.

This is called once per process at startup (e.g. once in batched_detector_worker).
Failures here are FATAL — if the configured model can't be loaded, the
application cannot do its job, so we log critically and re-raise to stop
startup rather than limping along with no detector.
"""

from loguru import logger

from src.core.config import settings, ModelType
from src.core.exceptions import ModelConfigError, ModelResolutionError, ModelDownloadError
from src.vision.model_resolver import resolve_model_path
from src.vision.base import BaseDetector
from src.vision.detectors.yolox import YOLOXDetector
from src.vision.detectors.hf import HFTransformerDetector
from src.vision.detectors.openvino import OpenVinoDetector
from src.vision.detectors.ultralytics_yolo import UltralyticsDetector
from src.vision.detectors.ultralytics_yolo_onnx import UltralyticsONNXDetector


def get_detector() -> BaseDetector:
    """
    Build and return the detector configured via settings.MODEL_ARCH.

    Returns:
        A BaseDetector subclass instance, ready for predict()/predict_batch()

    Raises:
        ModelConfigError: MODEL_ID/MODEL_ARCH settings are invalid or
            MODEL_ARCH is not a supported value
        ModelDownloadError: Model needed downloading and that failed
        ModelResolutionError: Model resolution failed for any other reason
    """
    arch = settings.MODEL_ARCH
    conf = settings.CONF_THRESHOLD
    device = settings.DEVICE.value
    iou_thres = settings.IOU_THRES
    nms_thres = settings.NMS_THRESHOLD

    try:
        model_path = resolve_model_path()
        logger.info(f"Resolved model path: {model_path} (arch={arch.value})")
    except (ModelConfigError, ModelDownloadError, ModelResolutionError) as e:
        logger.critical(f"FATAL MODEL CONFIG ERROR: {e}")
        raise
    except Exception as e:
        logger.critical(f"FATAL: Unexpected error resolving model path: {e}")
        raise ModelResolutionError("Unexpected error during model resolution") from e

    if arch == ModelType.YOLOX:
        return YOLOXDetector(
            model_path=model_path,
            conf_thresh=conf,
            nms_thresh=nms_thres,
            class_agnostic=settings.CLASS_AGNOSTIC,
        )

    elif arch in (ModelType.RFDETR, ModelType.DFINE):
        _INPUT_DIM = (544, 544)
        _MODEL_NAME = 'rfdetr'
        if 'dfine' in model_path:
            _MODEL_NAME = 'dfine'
            _INPUT_DIM = (640, 640)
        return HFTransformerDetector(
            model_path=model_path,
            model_name=_MODEL_NAME,
            input_dim=_INPUT_DIM,
            conf_thresh=conf,
            device=device,
        )

    elif arch == ModelType.OPENVINO:
        return OpenVinoDetector(
            model_path=model_path,
            conf_thresh=conf,
            device=device.upper(),
        )

    elif arch == ModelType.ULTRALYTICS:
        return UltralyticsDetector(
            model_path=model_path,
            conf_thresh=conf,
            device=device,
        )

    elif arch == ModelType.YOLO_ONNX:
        return UltralyticsONNXDetector(
            model_path=model_path,
            conf_thresh=conf,
            iou_thres=iou_thres,
            device=device,
        )

    else:
        raise ModelConfigError(
            f"Unknown or unsupported MODEL_ARCH: '{arch}'. "
            f"Supported values: {[m.value for m in ModelType]}",
            context={"model_arch": str(arch)}
        )
