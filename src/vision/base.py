"""
Abstract base class for all detector backends.

Every concrete detector (YOLO, YOLOX, RT-DETR, OpenVINO, ...) must
implement this interface so that pipeline.py and factory.py can treat
them interchangeably.
"""

import abc
import numpy as np
import supervision as sv


class BaseDetector(abc.ABC):
    """
    Common interface for all object detectors.

    Implementations are responsible for their OWN preprocessing and
    postprocessing — callers pass a raw BGR frame and receive ready-to-use
    sv.Detections in original-image coordinates.
    """

    @abc.abstractmethod
    def predict(self, frame: np.ndarray) -> sv.Detections:
        """
        Run inference on a single frame.

        Args:
            frame: Raw BGR image, shape (H, W, 3), dtype uint8

        Returns:
            sv.Detections with xyxy boxes (in `frame`'s coordinate space),
            confidence scores, and class_id — already filtered by
            confidence threshold and NMS.
        """
        raise NotImplementedError

    def predict_batch(self, frames: list[np.ndarray]) -> list[sv.Detections]:
        """
        Run inference on multiple frames.

        Default implementation calls predict() in a loop — correct but
        not batched. Backends that support true batched inference
        (e.g. ONNX models exported with dynamic=True) should override
        this for better throughput.

        Args:
            frames: List of raw BGR images

        Returns:
            List of sv.Detections, one per input frame, in the same order
        """
        return [self.predict(frame) for frame in frames]
