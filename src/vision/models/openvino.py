import cv2
import numpy as np
import supervision as sv
# from openvino import Core
from src.vision.base import BaseDetector
from openvino.runtime import Core


class OpenVinoDetector(BaseDetector):
    def __init__(self, model_path: str, conf_thresh: float, device: str):
        self.confidence_threshold = conf_thresh
        
        # ie = Core() # type: ignore [has-type]
        # self.model = ie.compile_model(model=model_path, device_name=device) # type: ignore [has-type]
        
        core = Core()
        # MyPy cannot determine type of "model" from core.read_model
        model = core.read_model(model=model_path) # type: ignore[has-type]
        # MyPy cannot determine type of "compiled_model"
        self.model = core.compile_model(model=model, device_name=device) # type: ignore[has-type]
        
        self.infer_request = self.model.create_infer_request()
        self.input_layer = self.model.input(0)
        self.output_layer = self.model.output(0)
        
        shape = self.input_layer.shape
        self.input_h = shape[2] if isinstance(shape[2], int) else 640
        self.input_w = shape[3] if isinstance(shape[3], int) else 640
        
        # self.input_layer = self.model.input(0)
        # self.output_layer = self.model.output(0)
        # self.input_h = self.input_layer.shape[2]
        # self.input_w = self.input_layer.shape[3]

        
    def preprocess(self, img: np.ndarray, img_h: int, img_w: int):
        # 1. Convert BGR to RGB (Critical for Transformers)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. Resize with Aspect Ratio (Letterbox)
        r = min(self.input_h / img_h, self.input_w / img_w)
        new_unpad = (int(img_w * r), int(img_h * r))
        
        # Resize
        resized_img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        # 3. Create Canvas (Padding)
        # Using 0 (black) or 114 (gray) depends on training. 
        # RT-DETR/DFine usually prefer 0 for padding.
        canvas = np.full((self.input_h, self.input_w, 3), 0, dtype=np.float32)
        
        # Paste resized image onto canvas
        canvas[:new_unpad[1], :new_unpad[0]] = resized_img

        # 4. Normalize (Scale 0-1, then Mean/Std)
        # Your params said: "do_rescale=True", "rescale_factor=0.00392" (1/255)
        canvas /= 255.0 
        canvas -= np.array([0.485, 0.456, 0.406])
        canvas /= np.array([0.229, 0.224, 0.225])

        # 5. Transpose to CHW and Add Batch Dimension
        canvas = canvas.transpose((2, 0, 1)) # HWC -> CHW
        # Add batch dim -> (1, 3, 640, 640)
        canvas = np.expand_dims(canvas, axis=0) # type: ignore [assignment]
        
        # Ensure contiguous memory for ONNX Runtime
        return np.ascontiguousarray(canvas, dtype=np.float32), r

    def postprocess(self, results:np.ndarray, ratio:int, original_h:int, original_w:int) -> sv.Detections:
        # 1. Extract Raw Data
        # Shape: [1, 1, N, 7] -> [N, 7]
        detections_raw = results[0]
        
        # 2. Filter by Confidence
        mask = detections_raw[:, 2] > self.confidence_threshold
        filtered_dets = detections_raw[mask]
        
        if len(filtered_dets) == 0:
            return sv.Detections.empty()

        # 3. Extract Coordinates (Normalized 0-1)
        # columns 3,4,5,6 correspond to xmin, ymin, xmax, ymax
        norm_boxes = filtered_dets[:, 3:7]

        # 4. Scale to Model Input Size (Canvas Pixels)
        # We use self.input_w and self.input_h which are the model's expected dims
        canvas_boxes = norm_boxes * np.array([self.input_w, self.input_h, self.input_w, self.input_h])

        # 5. Remove Padding & Scale to Original Image (Divide by Ratio)
        # Since you used top-left padding, we just divide by r.
        # If you used center padding, we would subtract padding offsets first.
        real_boxes = canvas_boxes / ratio

        # 6. Clip boxes to ensure they stay within original image boundaries
        # This removes any potential "detection" inside the black padding bars
        real_boxes[:, 0] = np.clip(real_boxes[:, 0], 0, original_w)
        real_boxes[:, 1] = np.clip(real_boxes[:, 1], 0, original_h)
        real_boxes[:, 2] = np.clip(real_boxes[:, 2], 0, original_w)
        real_boxes[:, 3] = np.clip(real_boxes[:, 3], 0, original_h)

        return sv.Detections(
            xyxy=real_boxes,
            confidence=filtered_dets[:, 2],
            class_id=filtered_dets[:, 1].astype(int)
        )


    def predict(self, frame: np.ndarray) -> sv.Detections:
        img_h, img_w, _ = frame.shape
        pre_frame, ratio = self.preprocess(frame, img_h, img_w)
        results = self.model([pre_frame])[self.output_layer]
        detections = self.postprocess(results[0], ratio, img_h, img_w)
        return detections