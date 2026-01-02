import numpy as np
import cv2
import onnxruntime as ort
import supervision as sv
from src.vision.base import BaseDetector
from src.core.config import settings

class HFTransformerDetector(BaseDetector):
    def __init__(self, model_path: str, conf_thresh: float):
        self.confidence_threshold = conf_thresh
        
        # 1. Load Model with Correct Provider Priority
        # (Assuming settings.DEVICE is "CUDA" or "CPU")
        if settings.DEVICE == "CUDA":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
            
        print(f"Loading RT-DETR/DFine from {model_path} on {providers[0]}...")
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # 2. Cache Input/Output Names (The Bulletproof Way)
        self.input_name = self.session.get_inputs()[0].name
        
        # Find which output is which by checking shapes or names
        # Usually: 'logits' (scores) and 'boxes' (coordinates)
        outputs = self.session.get_outputs()
        self.output_names = [o.name for o in outputs]
        
        # 3. Cache Input Shape
        shape = self.session.get_inputs()[0].shape
        # Handle dynamic axes safely
        self.input_h = shape[2] if isinstance(shape[2], int) else 640
        self.input_w = shape[3] if isinstance(shape[3], int) else 640

        # 4. Pre-calc constants (Optimization)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(self, img: np.ndarray):
        img_h, img_w = img.shape[:2]
        
        # 1. BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. Resize (Letterbox)
        r = min(self.input_h / img_h, self.input_w / img_w)
        new_unpad = (int(img_w * r), int(img_h * r))
        resized_img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        # 3. Canvas (Padding) - Use 0 or 114 depending on model training
        canvas = np.full((self.input_h, self.input_w, 3), 0, dtype=np.float32)  # type: ignore [assignment]
        canvas[:new_unpad[1], :new_unpad[0]] = resized_img

        # 4. Normalize
        canvas /= 255.0
        canvas -= self.mean
        canvas /= self.std

        # 5. HWC -> CHW -> Batch
        canvas = canvas.transpose((2, 0, 1))
        
        # Fix: MyPy sees np.full returning generic ndarray, expects specific type
        canvas = np.expand_dims(canvas, axis=0) # type: ignore [assignment]

        return np.ascontiguousarray(canvas, dtype=np.float32), r

    def postprocess(self, scores: np.ndarray, boxes: np.ndarray, ratio: float, img_h: int, img_w: int) -> sv.Detections:
        # 1. Sigmoid (if needed) - RT-DETR usually exports logits
        scores = 1 / (1 + np.exp(-scores))

        # 2. Filter by Confidence (Vectorized - Fast)
        max_scores = np.max(scores, axis=1)
        mask = max_scores > self.confidence_threshold
        
        if not np.any(mask):
            return sv.Detections.empty()

        pred_boxes = boxes[mask]
        pred_scores = max_scores[mask]
        pred_classes = np.argmax(scores[mask], axis=1)

        # 3. Convert CXCYWH (Normalized) -> XYXY (Absolute)
        cx, cy, w, h = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
        
        x1 = (cx - w / 2) * self.input_w
        y1 = (cy - h / 2) * self.input_h
        x2 = (cx + w / 2) * self.input_w
        y2 = (cy + h / 2) * self.input_h

        # 4. Scale back to original image size
        # Divide by ratio to undo resizing
        coords = np.stack([x1, y1, x2, y2], axis=1) / ratio

        # 5. Clip to image boundaries
        coords[:, 0::2] = np.clip(coords[:, 0::2], 0, img_w) # x1, x2
        coords[:, 1::2] = np.clip(coords[:, 1::2], 0, img_h) # y1, y2

        return sv.Detections(
            xyxy=coords,
            confidence=pred_scores,
            class_id=pred_classes.astype(int)
        )

    def predict(self, frame: np.ndarray) -> sv.Detections:
        img_h, img_w = frame.shape[:2]
        
        # Preprocess
        input_tensor, ratio = self.preprocess(frame)

        # Inference
        # We request outputs by name, or get all of them
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        # CRITICAL: Identify which output is which.
        # RT-DETR usually: outputs[0] = logits, outputs[1] = boxes
        # Check shape to be sure. Boxes last dim is usually 4.
        out1, out2 = outputs[0][0], outputs[1][0]
        
        if out1.shape[-1] == 4:
            raw_boxes, raw_scores = out1, out2
        else:
            raw_boxes, raw_scores = out2, out1
            
        return self.postprocess(raw_scores, raw_boxes, ratio, img_h, img_w)
