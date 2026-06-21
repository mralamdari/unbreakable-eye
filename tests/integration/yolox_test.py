import argparse
import cv2
import numpy as np
import openvino as ov

CLASS_NAME = "person"  # this model is single-class: person only


def preprocess(img: np.ndarray, input_h: int, input_w: int):
    """
    Official spec: input shape [1,3,320,544] BGR, simple resize
    (NOT letterboxed — the model was trained on direct resize).
    """
    resized = cv2.resize(img, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    blob = resized.transpose(2, 0, 1)            # HWC -> CHW, stays BGR, stays 0-255
    blob = np.expand_dims(blob, axis=0).astype(np.float32)
    return blob


def postprocess(output: np.ndarray, orig_h: int, orig_w: int, conf_thresh: float):
    """
    Output shape: [1, 1, 200, 7]
    Each row: [image_id, label, conf, x_min, y_min, x_max, y_max]
    x_min/y_min/x_max/y_max are normalized 0-1 relative to the ORIGINAL
    image (this model's output is already normalized to input image,
    no letterbox math needed — just multiply by orig_w/orig_h).
    """
    detections = output.reshape(-1, 7)
    boxes, scores = [], []

    for det in detections:
        image_id, label, conf, x_min, y_min, x_max, y_max = det
        if conf < conf_thresh:
            continue
        x1 = max(0, int(x_min * orig_w))
        y1 = max(0, int(y_min * orig_h))
        x2 = min(orig_w, int(x_max * orig_w))
        y2 = min(orig_h, int(y_max * orig_h))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2, y2])
        scores.append(float(conf))

    return boxes, scores


def draw_detections(img, boxes, scores):
    out = img.copy()
    for (x1, y1, x2, y2), score in zip(boxes, scores):
        color = (60, 200, 60)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"{CLASS_NAME} {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - baseline - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(out, text, (x1 + 1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def run(model_xml: str, image_path: str, conf_thresh: float = 0.5,
        save_path: str = "result.jpg", device: str = "CPU", show: bool = False):

    print(f"Loading OpenVINO model: {model_xml}")
    core = ov.Core()
    print(f"  Available devices: {core.available_devices}")

    model = core.read_model(model_xml)         # auto-loads matching .bin
    compiled_model = core.compile_model(model, device)

    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    # Input layout is [B, C, H, W]
    _, _, input_h, input_w = input_layer.shape
    print(f"  Input  : {input_layer.any_name}  shape={list(input_layer.shape)}")
    print(f"  Output : {output_layer.any_name}  shape={list(output_layer.shape)}")

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")
    orig_h, orig_w = img.shape[:2]
    print(f"\nImage  : {image_path}  ({orig_w}x{orig_h})")

    blob = preprocess(img, input_h, input_w)
    print(f"Blob   : {blob.shape}  dtype={blob.dtype}")

    result = compiled_model([blob])[output_layer]

    boxes, scores = postprocess(result, orig_h, orig_w, conf_thresh)

    print(f"\nDetections ({len(boxes)} above threshold {conf_thresh}):")
    for (x1, y1, x2, y2), score in zip(boxes, scores):
        print(f"  {CLASS_NAME:<10s}  score={score:.3f}  box=[{x1},{y1},{x2},{y2}]")

    vis = draw_detections(img, boxes, scores)
    cv2.imwrite(save_path, vis)
    print(f"\nSaved result -> {save_path}")

    if show:
        cv2.imshow("Person detections", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return boxes, scores

run('models/openvino/FP16-INT8/person-detection-retail-0013.xml', 'data/test.png', 0.2, 'data/output.png')


    
    
    
    
    
    
    
    
    