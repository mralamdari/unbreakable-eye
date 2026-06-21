"""
OpenVINO detector using Intel OpenVINO Runtime.

Tested with: person-detection-retail-0013 (SSD-based, output shape [1,1,N,7])
Output format per detection: [image_id, label, conf, x_min, y_min, x_max, y_max]
  where coordinates are NORMALIZED [0,1] relative to the MODEL INPUT size.

Preprocessing: plain BGR resize (no letterbox, no normalization) — this model
was trained on raw BGR 0-255 values at a fixed 320×544 resolution.
"""

import cv2
import numpy as np
import supervision as sv
from loguru import logger
from typing import Tuple

from src.vision.base import BaseDetector
from src.core.exceptions import ModelLoadError, InferenceError, PreprocessError, PostprocessError

try:
    import openvino as ov
except ImportError:
    raise ImportError(
        "openvino not installed. Run: pip install openvino"
    )


class OpenVinoDetector(BaseDetector):
    """SSD-style object detector via Intel OpenVINO Runtime."""

    def __init__(
        self,
        model_path: str,
        conf_thresh: float = 0.45,
        device: str = "CPU",
    ):
        """
        Args:
            model_path:  Path to .xml model file (companion .bin must be alongside it).
            conf_thresh: Minimum confidence to keep a detection.
            device:      OpenVINO device string: "CPU", "GPU", "AUTO", etc.

        Raises:
            ModelLoadError: If the model cannot be loaded or compiled.
        """
        self.confidence_threshold = conf_thresh
        self.device               = device.upper()

        try:
            logger.info(
                f"Loading OpenVINO model | path={model_path} | device={self.device}"
            )
            core = ov.Core()
            logger.debug(f"Available OpenVINO devices: {core.available_devices}")

            model               = core.read_model(model_path)
            self.compiled_model = core.compile_model(model=model, device_name=self.device)
            self.input_layer    = self.compiled_model.input(0)
            self.output_layer   = self.compiled_model.output(0)

            # Input layout is [B, C, H, W]
            _, _, self.input_h, self.input_w = self.input_layer.shape

            logger.info(
                f"OpenVINO ready | input={self.input_h}x{self.input_w} "
                f"| output_shape={list(self.output_layer.shape)}"
            )

        except ModelLoadError:
            raise
        except Exception as e:
            raise ModelLoadError(
                "Failed to load OpenVINO model",
                context={"model_path": model_path, "device": device, "error": str(e)},
            ) from e

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Resize frame to model input size and convert to NCHW float32.

        This model (person-detection-retail-0013) expects:
          - BGR channel order (OpenCV default — no conversion needed)
          - Raw 0-255 pixel values (no /255, no mean/std)
          - Shape (1, 3, input_h, input_w)

        Args:
            img: BGR uint8 frame (any resolution).

        Returns:
            (1, 3, input_h, input_w) float32 contiguous array.

        Raises:
            PreprocessError: If resizing fails.
        """
        try:
            resized = cv2.resize(img, (self.input_w, self.input_h),
                                 interpolation=cv2.INTER_LINEAR)
            blob = resized.transpose(2, 0, 1)               # HWC → CHW
            blob = np.expand_dims(blob, axis=0).astype(np.float32)
            return np.ascontiguousarray(blob)

        except Exception as e:
            raise PreprocessError(
                "OpenVINO preprocessing failed",
                context={"image_shape": list(img.shape), "error": str(e)},
            ) from e

    # ── Postprocessing ────────────────────────────────────────────────────────

    def postprocess(
        self,
        result: np.ndarray,
        orig_h: int,
        orig_w: int,
    ) -> sv.Detections:
        """
        Decode SSD-style output into sv.Detections.

        person-detection-retail-0013 output shape: [1, 1, N, 7]
        Each row: [image_id, label, conf, x_min, y_min, x_max, y_max]
          where x/y values are normalized [0,1] relative to model input size.

        Args:
            result: Raw output tensor from compiled_model inference.
            orig_h: Height of the original (pre-resize) frame.
            orig_w: Width  of the original (pre-resize) frame.

        Returns:
            sv.Detections in original-image pixel coordinates.

        Raises:
            PostprocessError: If decoding fails.
        """
        try:
            # Flatten to (N, 7) regardless of leading batch/channel dims
            detections_raw = np.asarray(result).reshape(-1, 7)

            boxes, scores, class_ids = [], [], []
            for det in detections_raw:
                _, label, conf, x_min, y_min, x_max, y_max = det

                if conf < self.confidence_threshold:
                    continue

                # Scale normalized coords to original image pixels
                x1 = max(0,      int(x_min * orig_w))
                y1 = max(0,      int(y_min * orig_h))
                x2 = min(orig_w, int(x_max * orig_w))
                y2 = min(orig_h, int(y_max * orig_h))

                if x2 <= x1 or y2 <= y1:
                    continue   # degenerate box — skip

                boxes.append([x1, y1, x2, y2])
                scores.append(float(conf))
                class_ids.append(int(label))

            # supervision requires correctly shaped arrays even when empty
            return sv.Detections(
                xyxy=np.array(boxes,     dtype=np.float32).reshape(-1, 4),
                confidence=np.array(scores,    dtype=np.float32),
                class_id=np.array(class_ids, dtype=int),
            )

        except Exception as e:
            raise PostprocessError(
                "OpenVINO postprocessing failed",
                context={"result_shape": list(np.asarray(result).shape), "error": str(e)},
            ) from e

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, frame: np.ndarray) -> sv.Detections:
        """
        Run OpenVINO inference on a single BGR frame.

        Args:
            frame: BGR uint8 image (any resolution).

        Returns:
            sv.Detections in original-image pixel coordinates.

        Raises:
            InferenceError: If inference fails.
        """
        try:
            orig_h, orig_w = frame.shape[:2]
            blob           = self.preprocess(frame)
            result         = self.compiled_model([blob])[self.output_layer]
            detections     = self.postprocess(result, orig_h, orig_w)

            logger.debug(
                f"OpenVINO | detections={len(detections)} "
                f"| conf>{self.confidence_threshold}"
            )
            return detections

        except (PreprocessError, PostprocessError):
            raise
        except Exception as e:
            raise InferenceError(
                "OpenVINO inference failed",
                context={
                    "frame_shape": list(frame.shape),
                    "device": self.device,
                    "error": str(e),
                },
            ) from e
