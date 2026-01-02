import abc
import numpy as np
import supervision as sv

class BaseDetector(abc.ABC):

    @abc.abstractmethod
    def predict(self, frame: np.ndarray) -> sv.Detections:
        """
        Input: Raw BGR Image (numpy)
        Output: List of [x1, y1, x2, y2, class_id, score]
        
        CRITICAL: This function must handle its own Pre/Post processing.
        """
        pass