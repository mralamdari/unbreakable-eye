import cv2
import numpy as np
import onnxruntime as ort
import supervision as sv
from src.vision.base import BaseDetector
from src.core.config import settings

class YoloXDetector(BaseDetector):
    def __init__(self, model_path: str, conf_thresh: float, nms_thresh: float, class_agnostic: bool = True):
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.class_agnostic = class_agnostic

        # 1. Load Model with Priority
        if settings.DEVICE == "CUDA":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
            
        print(f"Loading YOLOX from {model_path} on {providers[0]}...")
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        
        # 2. Determine Input Shape
        shape = self.session.get_inputs()[0].shape
        self.input_h = shape[2] if isinstance(shape[2], int) else 640
        self.input_w = shape[3] if isinstance(shape[3], int) else 640

        # 3. SENIOR MOVE: Pre-calculate the Grid ONCE.
        # We don't want to do this math every frame.
        self._generate_grids()

    def _generate_grids(self):
        """Pre-calculates the decoding grid to save CPU cycles during inference."""
        strides = [8, 16, 32]
        self.grids = []
        self.expanded_strides = []

        for stride in strides:
            hsize, wsize = self.input_h // stride, self.input_w // stride
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            self.grids.append(grid)
            self.expanded_strides.append(np.full((*grid.shape[:2], 1), stride))

        self.grids = np.concatenate(self.grids, 1)
        self.expanded_strides = np.concatenate(self.expanded_strides, 1)

    def preprocess(self, img: np.ndarray):
    # def preprocess(self, img: np.ndarray)-> np.ndarray:
        
        # 1. Letterbox Resize
        img_h, img_w = img.shape[:2]
        r = min(self.input_h / img_h, self.input_w / img_w)
        
        # Resize
        new_unpad = (int(img_w * r), int(img_h * r))
        resized_img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR).astype(np.uint8)

        # Pad with 114 (Grey)
        padded_img = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8) # type: ignore [assignment]
        padded_img[:new_unpad[1], :new_unpad[0]] = resized_img

        # 2. HWC -> CHW and BGR (YOLOX uses BGR, no swap needed usually)
        blob = padded_img.transpose((2, 0, 1))
        blob = np.ascontiguousarray(blob, dtype=np.float32) # type: ignore [assignment]
        return blob, r

    def postprocess(self, outputs: np.ndarray, ratio: float) -> sv.Detections:
        # outputs shape: [1, 8400, 85] (batch, anchors, xywh+obj+classes)
        outputs = outputs[0]

        # 1. Decode Boxes (Using Cached Grids)
        # xy = (raw_xy + grid) * stride
        outputs[:, :2] = (outputs[:, :2] + self.grids[0]) * self.expanded_strides[0]
        # wh = exp(raw_wh) * stride
        outputs[:, 2:4] = np.exp(outputs[:, 2:4]) * self.expanded_strides[0]

        # 2. Extract Data
        boxes = outputs[:, :4]
        obj_conf = outputs[:, 4:5]
        cls_scores = outputs[:, 5:]
        
        # Final scores = obj_conf * cls_score
        scores = obj_conf * cls_scores
        
        # 3. Filter by Confidence (Vectorized)
        # Get max score and class ID for each anchor
        class_ids = np.argmax(scores, axis=1)
        max_scores = np.max(scores, axis=1)
        
        mask = max_scores > self.conf_thresh
        if not np.any(mask):
            return sv.Detections.empty()
            
        detections = boxes[mask]
        scores = max_scores[mask]
        class_ids = class_ids[mask]

        # 4. Convert CXCYWH -> XYXY
        # (cx, cy, w, h) -> (x1, y1, x2, y2)
        x1 = detections[:, 0] - detections[:, 2] / 2
        y1 = detections[:, 1] - detections[:, 3] / 2
        x2 = detections[:, 0] + detections[:, 2] / 2
        y2 = detections[:, 1] + detections[:, 3] / 2
        
        # 5. Rescale to Original Image
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1) / ratio

        # 6. NMS (The Optimized C++ Way)
        # cv2.dnn.NMSBoxes(boxes, scores, score_threshold, nms_threshold)
        # We need to pass [x, y, w, h] to OpenCV NMS, not xyxy
        # But we already have xyxy. Let's use supervision or just standard cv2 logic.
        
        # Converting to list for OpenCV NMS
        # box format for cv2 NMS is [x, y, w, h]
        nms_boxes = []
        for b in boxes_xyxy:
            nms_boxes.append([int(b[0]), int(b[1]), int(b[2]-b[0]), int(b[3]-b[1])])
            
        indices = cv2.dnn.NMSBoxes(
            nms_boxes, 
            scores.tolist(), 
            self.conf_thresh, 
            self.nms_thresh
        )

        if len(indices) == 0:
            return sv.Detections.empty()

        # # OpenCV returns a weird tuple structure, flatten it
        # indices = np.array(indices, dtype=int).flatten() 
        
        # Ensure indices is a 1D numpy array of integers for sv.Detections
        # MyPy can be pedantic about the exact type of flattened array assigned to Sequence[int]
        indices = np.array(indices, dtype=int).flatten() # type: ignore[assignment]


        return sv.Detections(
            xyxy=boxes_xyxy[indices],
            confidence=scores[indices],
            class_id=class_ids[indices].astype(int)
        )

    def predict(self, frame: np.ndarray) -> sv.Detections:
        # img_h, img_w = frame.shape[:2]
        
        # Preprocess
        blob, ratio = self.preprocess(frame)
        
        # Inference
        outputs = self.session.run(None, {self.input_name: blob[None, :, :, :]})
        
        # Postprocess
        return self.postprocess(outputs[0], ratio)
