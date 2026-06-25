"""
Multi-model detector benchmark harness.

Runs every configured detector architecture against the SAME video,
frame-by-frame, and reports:
    - speed:   ms/frame (mean, median, p95, fps) per model
    - "accuracy" proxy: cross-model IoU agreement (since no ground-truth
      labels exist yet — see note below)
    - a side-by-side annotated frame sample for manual spot-checking

──────────────────────────────────────────────────────────────────────────
IMPORTANT — what this CAN and CANNOT tell you without ground truth
──────────────────────────────────────────────────────────────────────────
With no labeled boxes, there is no objective "this model is right" signal.
What this harness gives you instead:

  1. Hard numbers on SPEED — fully objective, no labels needed.
  2. A cross-model "agreement score" — for each frame, how many models
     detected a person in roughly the same place (IoU-matched). Detections
     most models agree on are very likely real people. Detections only
     ONE model sees are either that model's unique strength (good recall)
     or a false positive — agreement alone can't tell you which.
  3. An auto-exported folder of side-by-side annotated frames so you can
     eyeball disagreements in minutes instead of hours of full labeling.

Use (2) + (3) together: sort frames by *lowest* agreement first when
spot-checking — that's where the actual differences between models live.
A model that's frequently the "odd one out" by MISSING boxes the others
agree on is under-detecting; a model that's frequently the odd one out by
ADDING boxes no one else sees is likely hallucinating false positives —
but only your eyes on the spot-check frames can confirm which is which.

──────────────────────────────────────────────────────────────────────────
USAGE
──────────────────────────────────────────────────────────────────────────
    python benchmark_models.py --video data/shop_cam1.mp4 --every-nth 5

    # Limit to specific architectures:
    python benchmark_models.py --video data/shop_cam1.mp4 --archs yolox openvino

    # Cap total frames processed (useful for a quick first pass):
    python benchmark_models.py --video data/shop_cam1.mp4 --max-frames 300
"""

import argparse
import itertools
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv
from loguru import logger

from src.core.config import settings, ModelType
from src.vision.model_resolver import resolve_model_path
from src.vision.detectors.hf import HFTransformerDetector
from src.vision.detectors.yolox import YOLOXDetector
from src.vision.detectors.openvino import OpenVinoDetector
from src.vision.detectors.ultralytics_yolo import UltralyticsDetector
from src.vision.detectors.ultralytics_yolo_onnx import UltralyticsONNXDetector

# ── Architectures included in the benchmark by default ──────────────────────
# Edit this if you want to add/remove a candidate. Each entry maps a
# friendly label -> (ModelType, constructor_kwargs_extra).

_ARCH_NAME_TO_TYPE = {
    "yolox":      ModelType.YOLOX,
    "rfdetr":     ModelType.RFDETR,
    "dfine":      ModelType.DFINE,
    "openvino":   ModelType.OPENVINO,
    "ultralytics":ModelType.ULTRALYTICS,
    "yolo_onnx":  ModelType.YOLO_ONNX,
}


_ARCH_NAME_TO_ID = {
    "yolox":      'yolox_s.onnx',
    "rfdetr":     'rfdetr_small-ONNX',
    "dfine":      'dfine_s_coco-ONNX',
    "openvino":   'FP16-INT8',
    "ultralytics":'yolo12s.pt',
    "yolo_onnx":  'yolo12s.onnx',
}


DEFAULT_ARCHS: List[str] = ["dfine_n", 
                            "dfine_m", 
                            "dfine_s", "rfdetr_medium", 
                            'rfdetr_small', 'rfdetr_base', 'rfdetr_nano',
                            'openvino', 'yolo12s', 'yolo12n', 'yolox_l',
                            'yolox_m', 'yolox_s', 'yolox_tiny', 'yolox_nano',
                            'yolo12n', 'yolo12s', 'yolo12m', 'yolo12l'
                            ]



model_paths = [
    'models/onnx-community/dfine_n_coco-ONNX/onnx/model_quantized.onnx',
    'models/onnx-community/dfine_m_coco-ONNX/onnx/model_quantized.onnx',
    'models/onnx-community/dfine_s_coco-ONNX/onnx/model_quantized.onnx',
    'models/onnx-community/rfdetr_medium-ONNX/onnx/model_quantized.onnx',
    'models/onnx-community/rfdetr_small-ONNX/onnx/model_quantized.onnx',
    'models/onnx-community/rfdetr_small-ONNX/onnx/model_quantized.onnx',
    'models/onnx-community/rfdetr_nano-ONNX/onnx/model_quantized.onnx',
    
    'models/openvino/FP16-INT8/person-detection-retail-0013.xml',
    
    'models/ultralytics/yolo12s.pt',
    'models/ultralytics/yolo12n.pt',
    
    'models/yolox/yolox_l.onnx',
    'models/yolox/yolox_m.onnx',
    'models/yolox/yolox_s.onnx',
    'models/yolox/yolox_tiny.onnx',
    'models/yolox/yolox_nano.onnx',
    
    'models/yolo_onnx/yolo12n.onnx',
    'models/yolo_onnx/yolo12s.onnx',
    'models/yolo_onnx/yolo12m.onnx',
    'models/yolo_onnx/yolo12l.onnx'
    ]


# ── Data classes ──────────────────────────────────────────────────────────

@dataclass
class FrameResult:
    frame_idx: int
    detections: sv.Detections
    latency_ms: float


@dataclass
class ModelBenchmarkResult:
    name: str
    arch: str
    model_path: str
    load_time_s: float
    frame_results: List[FrameResult] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def latencies_ms(self) -> List[float]:
        return [f.latency_ms for f in self.frame_results]

    @property
    def detection_counts(self) -> List[int]:
        return [len(f.detections) for f in self.frame_results]

    def speed_summary(self) -> dict:
        lat = self.latencies_ms
        if not lat:
            return {"mean_ms": None, "median_ms": None, "p95_ms": None, "fps": None}
        lat_sorted = sorted(lat)
        p95_idx = max(0, int(len(lat_sorted) * 0.95) - 1)
        mean_ms = statistics.mean(lat)
        return {
            "mean_ms":   round(mean_ms, 2),
            "median_ms": round(statistics.median(lat), 2),
            "p95_ms":    round(lat_sorted[p95_idx], 2),
            "fps":       round(1000.0 / mean_ms, 2) if mean_ms > 0 else None,
        }

    def detection_summary(self) -> dict:
        counts = self.detection_counts
        if not counts:
            return {"mean_detections": None, "min": None, "max": None, "zero_detection_frames": 0}
        return {
            "mean_detections":      round(statistics.mean(counts), 2),
            "min":                  min(counts),
            "max":                  max(counts),
            "zero_detection_frames": sum(1 for c in counts if c == 0),
        }


def build_detector(arch_id: str):
    """
    Construct a detector instance for the given friendly arch name,
    using the exact constructor signatures from your factory script.

    Returns:
        (detector_instance, model_path, load_time_seconds)

    Raises:
        ValueError: Unknown arch_name.
        Exception:  Propagated from model resolution/loading — caller
                    should catch and record as a benchmark failure rather
                    than letting one bad model abort the whole run.
    """

    conf = settings.CONF_THRESHOLD
    device = settings.DEVICE.value
    iou_thres = settings.IOU_THRES
    nms_thres = settings.NMS_THRESHOLD

    t0 = time.perf_counter()
    model_path = model_paths[arch_id]
    t1 = time.perf_counter()
    logger.info(f"[{arch_id}] model path resolved in {t1 - t0:.2f}s -> {model_path}")
    
    if arch_id <= 6:
        detector = HFTransformerDetector(model_path=model_path, conf_thresh=0.2)
    elif arch_id in [10, 11,12,13,14]:
        detector = YOLOXDetector(
            model_path=model_path,
            conf_thresh=0.4,
            nms_thresh=nms_thres,
            class_agnostic=settings.CLASS_AGNOSTIC,
        )
    elif arch_id == 7:
        detector = OpenVinoDetector(model_path=model_path, conf_thresh=0.98, device=device.upper())
    elif arch_id in [8, 9]:
        detector = UltralyticsDetector(model_path=model_path, conf_thresh=0.4, device=device)
    elif arch_id >= 15:
        detector = UltralyticsONNXDetector(
            model_path=model_path, conf_thresh=0.4, iou_thres=iou_thres, device=device
        )
    else:
        raise ValueError(f"No constructor wired for '{arch_id}'")

    t2 = time.perf_counter()
    logger.info(f"[{arch_id}] detector loaded in {t2 - t1:.2f}s")
    return detector, model_path, (t2 - t0)


# ── IoU / cross-model agreement ──────────────────────────────────────────

def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Pairwise IoU between two sets of xyxy boxes.

    Args:
        boxes_a: (N, 4) xyxy.
        boxes_b: (M, 4) xyxy.

    Returns:
        (N, M) IoU matrix.
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    a = boxes_a[:, None, :]  # (N,1,4)
    b = boxes_b[None, :, :]  # (1,M,4)

    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])
    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter = inter_w * inter_h

    area_a = np.clip((a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1]), 0, None)
    area_b = np.clip((b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1]), 0, None)
    union = area_a + area_b - inter

    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def cross_model_agreement(
    per_model_dets: Dict[str, sv.Detections], iou_thresh: float = 0.5
) -> dict:
    """
    For one frame, compute how many models agree on each detection.

    A detection from model X "agrees" with model Y if some box in Y's
    output has IoU >= iou_thresh with it.

    Returns:
        {
          "model_agreement_rate": {model_name: fraction of that model's
              boxes which at least one OTHER model also detected},
          "consensus_count": number of boxes 3+ models agree on (only
              meaningful with 4 models loaded; adjust threshold as needed),
        }
    """
    names = list(per_model_dets.keys())
    agreement_rate = {}
    for name in names:
        boxes = per_model_dets[name].xyxy
        if len(boxes) == 0:
            agreement_rate[name] = None
            continue
        agreed = np.zeros(len(boxes), dtype=bool)
        for other in names:
            if other == name:
                continue
            # print(len(per_model_dets[other]))
            # print(per_model_dets[other][0])
            # print(per_model_dets[other][1])
            # print(111111111111111111111111111, per_model_dets[other].class_id)
            other_boxes = per_model_dets[other].xyxy
            if len(other_boxes) == 0:
                continue
            ious = iou_matrix(boxes, other_boxes)
            agreed |= (ious.max(axis=1) >= iou_thresh)
        agreement_rate[name] = float(agreed.mean())

    return {"model_agreement_rate": agreement_rate}

# ── Main benchmark loop ───────────────────────────────────────────────────
# ── Main benchmark loop ───────────────────────────────────────────────────

def _coerce_to_detections(raw_result, arch_name: str) -> sv.Detections:
    """
    Normalize a detector's predict() return value to a bare sv.Detections.

    Not every detector class in this project returns the same shape from
    .predict(): some return sv.Detections directly, others return a tuple
    like (detections, raw_outputs) or (detections, tracker_state). This
    function makes that difference transparent to all benchmark code
    downstream (FrameResult, per_frame_per_model, cross_model_agreement),
    which all assume a bare sv.Detections.

    Args:
        raw_result: Whatever detector.predict(frame) returned.
        arch_name:  Friendly arch name, used only for logging context.

    Returns:
        sv.Detections

    Raises:
        TypeError: If no sv.Detections could be found in raw_result —
                   surfaced loudly rather than silently passing through
                   a tuple that would break downstream IoU/annotation code.
    """
    if isinstance(raw_result, sv.Detections):
        return raw_result

    if isinstance(raw_result, (tuple, list)):
        for item in raw_result:
            if isinstance(item, sv.Detections):
                return item
        raise TypeError(
            f"[{arch_name}] predict() returned a {type(raw_result).__name__} "
            f"but none of its elements is an sv.Detections. "
            f"Got element types: {[type(x).__name__ for x in raw_result]}"
        )

    raise TypeError(
        f"[{arch_name}] predict() returned unexpected type "
        f"{type(raw_result).__name__}, expected sv.Detections (or a "
        f"tuple/list containing one)."
    )


def load_frames(video_path: str, every_nth: int, max_frames: Optional[int]) -> List[np.ndarray]:
    """
    Decode every Nth frame from a video into memory.

    Loading frames once and reusing the same in-memory list across all
    models guarantees every model sees byte-identical input — critical
    for a fair comparison (re-decoding per model could introduce subtle
    differences from codec frame-seeking).

    Args:
        video_path: Path to the video file.
        every_nth:  Keep 1 out of every N frames (use >1 for long videos).
        max_frames: Stop after collecting this many frames (None = no cap).

    Returns:
        List of BGR uint8 frames.

    Raises:
        FileNotFoundError: If the video can't be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frames = []
    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % every_nth == 0:
                frames.append(frame)
                if max_frames and len(frames) >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()

    logger.info(f"Loaded {len(frames)} frames from {video_path} (every_nth={every_nth})")
    return frames


def run_benchmark(
    video_path: str,
    arch_names: List[str],
    every_nth: int = 5,
    max_frames: Optional[int] = None,
    sample_export_dir: Optional[str] = None,
    sample_export_count: int = 30,
) -> Dict[str, ModelBenchmarkResult]:
    """
    Run all requested architectures against the same frame set.

    Args:
        video_path:          Path to the test video.
        arch_names:           List of friendly arch names, e.g. ["yolox", "openvino"].
        every_nth:            Sample every Nth frame (reduces total runtime).
        max_frames:           Optional cap on total frames processed.
        sample_export_dir:    If set, write side-by-side annotated frames here
                              for manual spot-checking (lowest-agreement frames first).
        sample_export_count:  How many sample frames to export.

    Returns:
        {arch_name: ModelBenchmarkResult}
    """
    frames = load_frames(video_path, every_nth, max_frames)
    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")

    results: Dict[str, ModelBenchmarkResult] = {}
    per_frame_per_model: List[Dict[str, sv.Detections]] = [dict() for _ in frames]

    for arch_id, arch_name in enumerate(arch_names):
        logger.info(f"=== Benchmarking '{arch_name}' ===")
        
        # try:
        detector, model_path, load_time = build_detector(arch_id)
        # except Exception as e:
        #     logger.error(f"[{arch_name}] failed to load: {e}")
        #     results[arch_name] = ModelBenchmarkResult(
        #         name=arch_name, arch=arch_name, model_path="", load_time_s=0.0, error=str(e)
        #     )
        #     continue

        result = ModelBenchmarkResult(
            name=arch_name, arch=arch_name, model_path=model_path, load_time_s=load_time
        )

        for i, frame in enumerate(frames):
            # try:
            t0 = time.perf_counter()
            raw_result = detector.predict(frame)
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0
            
            detections = _coerce_to_detections(raw_result, arch_name)
            
            result.frame_results.append(
                FrameResult(frame_idx=i, detections=detections, latency_ms=latency_ms)
            )
            per_frame_per_model[i][arch_name] = detections

            # except Exception as e:
            #     logger.warning(f"[{arch_name}] frame {i} failed: {e}")
            #     # Record a failed frame as zero detections / NaN latency
            #     # rather than aborting the whole model's run on one bad frame.
            #     result.frame_results.append(
            #         FrameResult(frame_idx=i, detections=sv.Detections.empty(), latency_ms=float("nan"))
            #     )

            if (i + 1) % 50 == 0:
                logger.debug(f"[{arch_name}] processed {i + 1}/{len(frames)} frames")

        results[arch_name] = result
        # Free GPU/CPU resources between models if the detector supports it
        del detector

    # ── Cross-model agreement, per frame ─────────────────────────────────
    agreement_per_frame = []
    for i in range(len(frames)):
        dets_this_frame = per_frame_per_model[i]
        if len(dets_this_frame) >= 2:
            agreement_per_frame.append(cross_model_agreement(dets_this_frame))
        else:
            agreement_per_frame.append(None)

    # ── Export lowest-agreement frames for manual spot-check ────────────
    if sample_export_dir and len(arch_names) >= 2:
        _export_disagreement_samples(
            frames, per_frame_per_model, agreement_per_frame,
            sample_export_dir, sample_export_count,
        )

    return results


def _frame_agreement_score(agreement: Optional[dict]) -> float:
    """Lower = more disagreement among models on this frame."""
    if not agreement:
        return 1.0
    rates = [v for v in agreement["model_agreement_rate"].values() if v is not None]
    return statistics.mean(rates) if rates else 1.0


def _export_disagreement_samples(
    frames: List[np.ndarray],
    per_frame_per_model: List[Dict[str, sv.Detections]],
    agreement_per_frame: List[Optional[dict]],
    out_dir: str,
    count: int,
) -> None:
    """
    Save side-by-side annotated frames, prioritizing the frames where
    models disagree MOST — that's where manual inspection is most useful.
    """
    os.makedirs(out_dir, exist_ok=True)

    scored = sorted(
        range(len(frames)),
        key=lambda i: _frame_agreement_score(agreement_per_frame[i]),
    )
    chosen = scored[:count]

    color_cycle = [sv.Color.RED, sv.Color.GREEN, sv.Color.BLUE, sv.Color(255, 165, 0)]
    box_annotators = {}

    for frame_idx in chosen:
        frame = frames[frame_idx]
        model_names = list(per_frame_per_model[frame_idx].keys())
        tiles = []

        for j, name in enumerate(model_names):
            dets = per_frame_per_model[frame_idx][name]
            annotator = box_annotators.setdefault(
                name, sv.BoxAnnotator(color=color_cycle[j % len(color_cycle)])
            )
            tile = frame.copy()
            tile = annotator.annotate(tile, dets)
            cv2.putText(
                tile, f"{name} ({len(dets)})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA,
            )
            tiles.append(tile)

        if not tiles:
            continue

        # Arrange tiles in a grid (2 columns)
        h, w = tiles[0].shape[:2]
        cols = 2
        rows = (len(tiles) + cols - 1) // cols
        grid = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)
        for idx, tile in enumerate(tiles):
            r, c = divmod(idx, cols)
            grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = tile

        out_path = os.path.join(out_dir, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(out_path, grid)

    logger.info(f"Exported {len(chosen)} disagreement-sample frames -> {out_dir}")


# ── Report ─────────────────────────────────────────────────────────────────

def print_report(results: Dict[str, ModelBenchmarkResult]) -> dict:
    """Print a human-readable comparison table and return a JSON-able summary dict."""
    summary = {}

    print("\n" + "=" * 78)
    print(f"{'Model':<12} {'Load(s)':<9} {'Mean(ms)':<10} {'Median(ms)':<12} "
          f"{'P95(ms)':<9} {'FPS':<7} {'AvgDets':<9} {'0-det frames':<13}")
    print("-" * 78)

    for name, r in results.items():
        if r.error:
            print(f"{name:<12} FAILED: {r.error}")
            summary[name] = {"error": r.error}
            continue

        speed = r.speed_summary()
        dets = r.detection_summary()
        print(
            f"{name:<12} {r.load_time_s:<9.2f} "
            f"{speed['mean_ms'] or 0:<10.2f} {speed['median_ms'] or 0:<12.2f} "
            f"{speed['p95_ms'] or 0:<9.2f} {speed['fps'] or 0:<7.2f} "
            f"{dets['mean_detections'] or 0:<9.2f} {dets['zero_detection_frames']:<13}"
        )
        summary[name] = {
            "model_path": r.model_path,
            "load_time_s": round(r.load_time_s, 2),
            "speed": speed,
            "detections": dets,
        }

    print("=" * 78)
    print(
        "\nNote: 'AvgDets' alone does NOT tell you which model is most accurate —\n"
        "a model that over-detects (false positives) will show a HIGHER average.\n"
        "Check the exported disagreement-sample frames to judge quality by eye."
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Benchmark multiple detector architectures on one video")
    parser.add_argument("--video", required=True, help="Path to test video")
    parser.add_argument(
        "--archs", nargs="+", default=DEFAULT_ARCHS,
        choices=sorted(_ARCH_NAME_TO_TYPE), help="Architectures to benchmark"
    )
    parser.add_argument("--every-nth", type=int, default=1, help="Sample every Nth frame")
    parser.add_argument("--max-frames", type=int, default=None, help="Cap total frames processed")
    parser.add_argument(
        "--sample-dir", default="data/benchmark_samples",
        help="Where to export disagreement-sample frames"
    )
    parser.add_argument("--sample-count", type=int, default=30, help="Number of sample frames to export")
    parser.add_argument("--report-json", default="data/benchmark_report.json", help="Where to save the JSON report")
    args = parser.parse_args()

    results = run_benchmark(
        video_path=args.video,
        arch_names=args.archs,
        every_nth=args.every_nth,
        max_frames=args.max_frames,
        sample_export_dir=args.sample_dir,
        sample_export_count=args.sample_count,
    )

    summary = print_report(results)

    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w") as f:
        json.dump(summary, f, indent=2)
    logger.success(f"Report saved -> {args.report_json}")


if __name__ == "__main__":
    main()



























==============================================================================
Model        Load(s)   Mean(ms)   Median(ms)   P95(ms)   FPS     AvgDets   0-det frames 
------------------------------------------------------------------------------
dfine_n      0.34      110.81     107.27       133.36    9.02    13.49     0            
dfine_m      0.74      328.07     324.01       365.66    3.05    71.06     0            
dfine_s      0.45      202.34     197.86       230.37    4.94    80.64     0            
rfdetr_medium 0.84      369.96     362.07       406.93    2.70    8.97      0            
rfdetr_small 0.82      358.52     352.73       402.73    2.79    4.13      0            
rfdetr_base  0.19      358.21     352.16       389.95    2.79    4.13      0            
rfdetr_nano  0.78      342.79     337.94       379.19    2.92    1.33      76           
openvino     4.12      10.01      9.54         13.09     99.86   5.42      0            
yolo12s      0.99      165.50     157.42       213.30    6.04    4.01      5            
yolo12n      0.34      75.03      70.50        99.01     13.33   3.29      1            
yolox_l      4.31      1101.13    1067.70      1348.79   0.91    7.16      0            
yolox_m      2.18      541.08     525.00       678.04    1.85    6.40      0            
yolox_s      0.77      229.67     219.75       298.31    4.35    4.45      0            
yolox_tiny   0.40      65.54      63.92        76.79     15.26   4.82      0            
yolox_nano   0.13      32.31      31.19        39.16     30.95   4.67      1            
yolo12m      1.88      399.41     385.52       499.02    2.50    4.03      0            
yolo12l      2.47      533.11     519.48       652.28    1.88    4.06      0            
==============================================================================
