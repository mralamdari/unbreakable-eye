"""
OpenVINO detector using Intel OpenVINO Runtime.
Best for: Intel hardware, edge inference optimization.
"""

import cv2
import numpy as np
import supervision as sv
from loguru import logger
from typing import Tuple

from src.vision.base import BaseDetector
from src.vision.utils import letterbox, normalize_imagenet, cxcywh_to_xyxy, scale_boxes
from src.core.exceptions import ModelLoadError, InferenceError, PreprocessError, PostprocessError

try:
    from openvino.runtime import Core
except ImportError:
    raise ImportError("openvino not installed. Install with: pip install openvino")


class OpenVinoDetector(BaseDetector):
    """
    OpenVINO detector with native OpenVINO Runtime.
    Optimized for Intel hardware acceleration.
    """

    def __init__(
        self,
        model_path: str,
        conf_thresh: float = 0.45,
        device: str = "CPU"
    ):
        """
        Args:
            model_path: Path to .onnx or .xml model file
            conf_thresh: Confidence threshold
            device: OpenVINO device ("CPU", "GPU", "HETERO:GPU,CPU", etc.)

        Raises:
            ModelLoadError: If model cannot be loaded
        """
        self.confidence_threshold = conf_thresh
        self.device = device.upper()

        try:
            logger.info(f"Loading OpenVINO model from {model_path} on device {self.device}")
            core = Core()
            model = core.read_model(model=model_path)
            self.model = core.compile_model(model=model, device_name=self.device)

            # Create inference request
            self.infer_request = self.model.create_infer_request()
            self.input_layer = self.model.input(0)
            self.output_layers = {output.get_any_name(): idx for idx, output in enumerate(self.model.outputs)}

            # Get input shape
            shape = self.input_layer.shape
            self.input_h = shape[2] if len(shape) > 2 and isinstance(shape[2], int) else 640
            self.input_w = shape[3] if len(shape) > 3 and isinstance(shape[3], int) else 640

            logger.info(f"OpenVINO model loaded: input {self.input_h}x{self.input_w}, "
                       f"outputs: {len(self.output_layers)}")

        except Exception as e:
            logger.error(f"Failed to load OpenVINO model: {e}")
            raise ModelLoadError(
                "Failed to load OpenVINO model",
                context={"model_path": model_path, "device": device, "error": str(e)}
            ) from e

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Preprocess frame for OpenVINO inference.

        Args:
            img: Input BGR image

        Returns:
            (input_blob, ratio) where input_blob is (1, 3, H, W) and
            ratio is the scale factor for box rescaling

        Raises:
            PreprocessError: If preprocessing fails
        """
        try:
            # Letterbox with aspect ratio preservation
            img_padded, _ = letterbox(img, self.input_h, self.input_w, pad_value=0)

            # ImageNet normalization (standard for OpenVINO models)
            img_norm = normalize_imagenet(img_padded)

            # HWC → CHW
            img_chw = np.transpose(img_norm, (2, 0, 1))

            # Add batch dimension
            blob = np.expand_dims(img_chw, axis=0).astype(np.float32)

            # Calculate ratio for rescaling boxes
            h, w = img.shape[:2]
            scale = min(self.input_h / h, self.input_w / w)
            ratio = 1.0 / scale

            return np.ascontiguousarray(blob), ratio

        except Exception as e:
            logger.error(f"OpenVINO preprocessing failed: {e}")
            raise PreprocessError(
                "OpenVINO preprocessing failed",
                context={"image_shape": img.shape, "error": str(e)}
            ) from e

    def postprocess(
        self,
        outputs_dict: dict,
        ratio: float,
        img_h: int,
        img_w: int
    ) -> sv.Detections:
        """
        Decode OpenVINO model outputs.

        Args:
            outputs_dict: Dict of {output_name: array} from inference
            ratio: Scale ratio from preprocessing
            img_h: Original image height
            img_w: Original image width

        Returns:
            sv.Detections with detected objects

        Raises:
            PostprocessError: If postprocessing fails
        """
        try:
            # Extract outputs by searching for shape patterns
            # Typically: boxes are (N, 4) and scores are (N, num_classes) or (N,)
            boxes = None
            scores = None

            for name, tensor in outputs_dict.items():
                data = tensor.data
                if data.ndim == 2:
                    if data.shape[1] == 4:
                        boxes = data
                    elif data.shape[1] > 1:
                        scores = data
                    elif data.shape[1] == 1:
                        if scores is None:
                            scores = data
                elif data.ndim == 1:
                    if scores is None:
                        scores = data

            if boxes is None or scores is None:
                logger.warning("Could not identify boxes or scores in output")
                return sv.Detections.empty()

            # Ensure 2D shapes
            if scores.ndim == 1:
                scores = scores[:, np.newaxis]

            # Get max score and class
            if scores.shape[1] > 1:
                class_ids = np.argmax(scores, axis=1)
                max_scores = np.max(scores, axis=1)
            else:
                class_ids = np.zeros(scores.shape[0], dtype=int)
                max_scores = scores.squeeze()

            # Filter by confidence
            mask = max_scores > self.confidence_threshold
            if not np.any(mask):
                return sv.Detections.empty()

            boxes = boxes[mask]
            max_scores = max_scores[mask]
            class_ids = class_ids[mask]

            # Boxes format detection and conversion
            # Assume normalized [0, 1] CXCYWH format (common in OpenVINO exports)
            boxes_abs = boxes.copy()
            if np.max(boxes) <= 1.0:  # Normalized format
                boxes_abs[:, 0] *= self.input_w
                boxes_abs[:, 1] *= self.input_h
                boxes_abs[:, 2] *= self.input_w
                boxes_abs[:, 3] *= self.input_h

            # CXCYWH → XYXY
            boxes_xyxy = cxcywh_to_xyxy(boxes_abs)

            # Rescale to original image coordinates
            boxes_xyxy = scale_boxes(
                boxes_xyxy, ratio, ratio,
                clip_h=img_h,
                clip_w=img_w
            )

            return sv.Detections(
                xyxy=boxes_xyxy,
                confidence=max_scores,
                class_id=class_ids.astype(int)
            )

        except Exception as e:
            logger.error(f"OpenVINO postprocessing failed: {e}")
            raise PostprocessError(
                "OpenVINO postprocessing failed",
                context={"outputs_keys": list(outputs_dict.keys()), "error": str(e)}
            ) from e

    def predict(self, frame: np.ndarray) -> sv.Detections:
        """
        Run inference on a single frame.

        Args:
            frame: Input BGR image

        Returns:
            sv.Detections with detected objects

        Raises:
            InferenceError: If inference fails
        """
        try:
            img_h, img_w = frame.shape[:2]
            pre_frame, ratio = self.preprocess(frame)

            # Set input tensor
            self.infer_request.set_input_tensor(pre_frame)

            # Run inference
            self.infer_request.infer()

            # Extract outputs as dict
            outputs_dict = {}
            for output in self.model.outputs:
                name = output.get_any_name()
                outputs_dict[name] = self.infer_request.get_output_tensor(output)

            detections = self.postprocess(outputs_dict, ratio, img_h, img_w)
            logger.debug(f"OpenVINO inference: {len(detections)} detections")
            return detections

        except Exception as e:
            logger.error(f"OpenVINO inference failed: {e}")
            raise InferenceError(
                "OpenVINO inference failed",
                context={"image_shape": frame.shape, "device": self.device, "error": str(e)}
            ) from e
