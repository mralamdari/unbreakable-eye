import os
import cv2
import numpy as np
import supervision as sv
import onnxruntime as ort
from src.core.config import settings
from src.vision.base import BaseDetector
from src.vision.utils import create_session


class UltralyticsONNXDetector(BaseDetector):
    def __init__(self, 
                 model_path: str,
                 conf_thresh: float,
                 iou_thres: float,
                 device: str):

        self.device = 0 if device.upper() in ["CUDA", "GPU"] else 'cpu'
        self.confidence_thres = conf_thresh
        self.iou_thres = iou_thres
        self.class_names = [f"class_{i}" for i in range(80)]
        # self.model = ort.InferenceSession(model_path, providers=providers or available)
        self.model = create_session(model_path, num_threads=(os.cpu_count())//2-1)
        self.input_size = settings.FRAME_SHAPE
        self.input_name = self.model.get_inputs()[0].name


    def letterbox(self, img: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """
        Resize and pad image to target size while preserving aspect ratio.
        Returns:
            - padded image (np.ndarray) of shape (target_h, target_w, 3)
            - pad (top, left) amounts used for padding (needed to map boxes back).
        """
        shape = img.shape[:2]                     # original (height, width)
        target_h, target_w, _ = self.input_size
        r = min(target_h / shape[0], target_w / shape[1])
        new_unpad = (round(shape[1] * r), round(shape[0] * r))  # (width, height)
        dw, dh = (target_w - new_unpad[0]) / 2, (target_h - new_unpad[1]) / 2

        # Resize only if needed
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        # Pad with constant gray (114,114,114)
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, (top, left)

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_padded, pad = self.letterbox(img_rgb)
        img_norm = img_padded.astype(np.float32) / 255.0
        img_chw = np.transpose(img_norm, (2, 0, 1))   # (3, H, W) — NO batch dim
        return img_chw, pad

    def postprocess(self, output: np.ndarray, pad: tuple[int, int],
                    original_shape: tuple[int, int]) -> sv.Detections:
        """
        Decode raw ONNX output, apply NMS, and return Supervision Detections.
        Args:
            output: raw output from ONNX model, shape (1, 84, N) or (N, 84)
            pad: (top, left) padding applied during letterbox
            original_shape: (height, width) of the original image
        Returns:
            sv.Detections containing xyxy boxes, confidence, class_id
        """
        # Squeeze batch dimension and transpose to (N, 84)
        if output.ndim == 3:
            output = np.squeeze(output, axis=0)   # (84, N) or (N, 84) depending on exporter
        if output.shape[0] == 84:                 # shape (84, N) -> transpose
            output = output.T                     # now (N, 84)
        # output is now (num_detections, 84) where 4 = box, 80 = class scores

        # Extract boxes (x, y, w, h) and class scores
        boxes_xywh = output[:, 0:4]               # (N, 4)
        scores = output[:, 4:]                    # (N, 80)

        # Apply confidence threshold
        max_scores = np.max(scores, axis=1)
        mask = max_scores >= self.confidence_thres
        if not np.any(mask):
            return sv.Detections.empty()

        boxes_xywh = boxes_xywh[mask]
        max_scores = max_scores[mask]
        class_ids = np.argmax(scores[mask], axis=1)

        # Remove padding and scale back to original image coordinates
        top, left = pad
        gain = min(self.input_size[0] / original_shape[0],
                   self.input_size[1] / original_shape[1])

        # boxes_xywh are in padded coordinates (relative to padded input)
        # Convert to absolute xywh in padded image, then subtract padding, then scale by gain
        x_center = boxes_xywh[:, 0] - left
        y_center = boxes_xywh[:, 1] - top
        width = boxes_xywh[:, 2]
        height = boxes_xywh[:, 3]

        x1 = (x_center - width / 2) / gain
        y1 = (y_center - height / 2) / gain
        x2 = (x_center + width / 2) / gain
        y2 = (y_center + height / 2) / gain

        # Clip to original image boundaries
        x1 = np.clip(x1, 0, original_shape[1])
        y1 = np.clip(y1, 0, original_shape[0])
        x2 = np.clip(x2, 0, original_shape[1])
        y2 = np.clip(y2, 0, original_shape[0])

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        nms_values = np.stack([x1, y1, x2, y2, max_scores], axis=1).astype(np.float32)

        # Apply NMS (using Supervision's built-in NMS)
        detections = sv.Detections(
            xyxy=boxes_xyxy,
            confidence=max_scores,
            class_id=class_ids
        )
        # In-place NMS
        indices = sv.box_non_max_suppression(nms_values, self.iou_thres)
        detections = detections[indices]
        return detections

    def predict(self, image: np.ndarray) -> sv.Detections:
        img_data, pad = self.preprocess(image)
        batch = img_data[np.newaxis]                   # (1, 3, H, W)
        outputs = self.model.run(None, {self.input_name: batch})
        return self.postprocess(outputs[0], pad, image.shape[:2])

    def predict_batch(self, frames: list[np.ndarray]) -> list[sv.Detections]:
        tensors, pads, orig_shapes = [], [], []
        for frame in frames:
            t, p = self.preprocess(frame)              # (3, H, W)
            tensors.append(t)
            pads.append(p)
            orig_shapes.append(frame.shape[:2])

        batch = np.stack(tensors, axis=0)              # (N, 3, H, W) — correct rank 4
        raw = self.model.run(None, {self.input_name: batch})[0]  # (N, 84, A)

        return [
            self.postprocess(raw[i], pads[i], orig_shapes[i])
            for i in range(len(frames))
        ]







































# import os
# import cv2
# import numpy as np
# import supervision as sv
# import onnxruntime as ort
# # from ultralytics import YOLO
# from src.vision.base import BaseDetector


# os.environ["OMP_NUM_THREADS"] = "4"
# os.environ["OMP_WAIT_POLICY"] = "ACTIVE"
# os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
# os.environ["KMP_BLOCKTIME"] = "0"

# def create_session(model_path, num_threads=2):
#     sess_options = ort.SessionOptions()
#     sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
#     sess_options.enable_mem_pattern = True
#     sess_options.enable_cpu_mem_arena = True
#     sess_options.intra_op_num_threads = num_threads
#     sess_options.inter_op_num_threads = 2
#     sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
#     sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
#     available = ort.get_available_providers()
#     providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
#     return ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)

# class UltralyticsONNXDetector(BaseDetector):
#     def __init__(self, 
#                  model_path: str,
#                  conf_thresh: float,
#                  iou_thres: float,
#                  device: str):

#         self.device = 0 if device.upper() in ["CUDA", "GPU"] else 'cpu'
#         self.confidence_thres = conf_thresh
#         self.iou_thres = iou_thres
#         self.class_names = [f"class_{i}" for i in range(80)]
#         self.model = create_session(model_path, num_threads=(os.cpu_count())//2-1)
#         self.model_inputs = self.model.get_inputs()
#         self.input_name = self.model_inputs[0].name
#         self.model_in_size = self.model_inputs[0].shape
#         self.input_size = (self.model_in_size[2], self.model_in_size[3])

#     def predict(self, img_data) -> sv.Detections:
#         """Run inference on a single image and return Supervision Detections."""
#         return self.model.run(None, {self.input_name: img_data})
    
    
    
    
    
    
    
    
    
    
    
# "The rolling cluster eviction in store_embedding fires a DB read on every write:"
# and how to solve "The rolling cluster eviction in store_embedding fires a DB read on every write" to make the other processes caches to get updated as well ?
# and make the "But last_seen_time/camera/center are updated inside fast_match which runs in the db_writer's process " a flawless 


