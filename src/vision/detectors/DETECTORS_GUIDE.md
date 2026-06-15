# Vision Detectors — Professional Documentation

## Overview

All detector implementations follow a common interface (`BaseDetector`) with consistent method signatures. 


## Detector Selection Guide

### 1. UltralyticsDetector (PyTorch Native)
- **Best for**: Development, when GPU with CUDA is available
- **Pros**: Simplest API, full debugging support, hot reload
- **Cons**: Heavy dependencies, slower on CPU, requires GPU for best performance
- **Use case**: Local testing, high-accuracy requirements
```python
detector = UltralyticsDetector(
    model_path="models/yolov8n.pt",
    conf_thresh=0.45,
    device="cuda"
)
```

### 2. UltralyticsONNXDetector (ONNX Runtime)
- **Best for**: Edge deployment, cross-platform consistency
- **Pros**: Lightweight, fast on CPU, batching support, no PyTorch dependency
- **Cons**: Model export required, slightly lower accuracy (quantization)
- **Use case**: Production edge devices, 24/7 monitoring
```python
detector = UltralyticsONNXDetector(
    model_path="models/yolov8n.onnx",
    conf_thresh=0.45,
    iou_thres=0.45,
    device="cpu"
)
```

### 3. YOLOXDetector (ONNX Runtime)
- **Best for**: Optimized anchor-based detection, custom training
- **Pros**: Pre-calculated grids (fast decoding), good accuracy-speed tradeoff
- **Cons**: Requires YOLOX-specific models, less common than YOLO
- **Use case**: When you've trained YOLOX models, edge devices
```python
detector = YOLOXDetector(
    model_path="models/yolox.onnx",
    conf_thresh=0.45,
    nms_thresh=0.45
)
```

### 4. HFTransformerDetector (ONNX Runtime)
- **Best for**: Highest accuracy, transformer-based models
- **Pros**: State-of-the-art accuracy, end-to-end learning, no NMS needed
- **Cons**: Slower inference, larger models, more memory
- **Use case**: When accuracy > speed, small object detection
```python
detector = HFTransformerDetector(
    model_path="models/rtdetr.onnx",
    conf_thresh=0.45,
    device="cpu"
)
```

### 5. OpenVinoDetector (Intel OpenVINO)
- **Best for**: Intel hardware optimization, Intel Core/Xeon processors
- **Pros**: Hardware-accelerated on Intel, excellent for CPU inference
- **Cons**: Intel-specific, less portable, different optimization pipeline
- **Use case**: Enterprise Intel deployments, data center inference
```python
detector = OpenVinoDetector(
    model_path="models/detector.xml",
    conf_thresh=0.45,
    device="CPU"
)
```

## Performance Benchmarks (Indicative)

```
Model              Backend      FPS (512×512)   Memory (MB)   Accuracy (mAP)
────────────────────────────────────────────────────────────────────────
YOLOv8n ONNX       CPU          60              120           0.37
YOLOv8n PyTorch    GPU (CUDA)   250             2000          0.37
YOLOX ONNX         CPU          50              100           0.32
RT-DETR ONNX       CPU          20              400           0.54
OpenVINO (YOLO)    CPU+VPU      120             150           0.37
```