"""
Model path resolution and downloading.

Resolves a model file path from settings (MODEL_ID, MODEL_ARCH) by:
1. Checking if the file already exists locally under models/
2. If not, downloading it from the appropriate source:
   - Ultralytics: download .pt and export to .onnx
   - YOLOX: download .onnx from GitHub releases
   - DFINE/RFDETR: download .onnx from HuggingFace Hub
   - OpenVINO: must already exist locally (no auto-download source)

This module is imported ONCE at startup by factory.py — it is not
part of the per-frame inference hot path, so correctness and clear
error messages matter more than raw speed here.
"""

import os
import re
import shutil
from functools import lru_cache

import requests
from loguru import logger
from huggingface_hub import hf_hub_download

from src.core.config import settings, ModelType
from src.core.exceptions import ModelResolutionError, ModelDownloadError, ModelConfigError

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# --- Constants ---
YOLOX_GITHUB_BASE_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"

# Regex for "org/repo-name" style HuggingFace repo IDs
HF_REPO_ID_PATTERN = re.compile(r"^[a-zA-Z0-9-]+/[a-zA-Z0-9-.]+$")


# ─────────────────────────────────────────────────────────────────────────────
# Default filenames per architecture
# ─────────────────────────────────────────────────────────────────────────────

def _infer_default_filename(model_arch: ModelType) -> str:
    """
    Return the default filename for a given model architecture.

    Used when MODEL_ID is empty and only MODEL_ARCH is set in settings —
    lets a user write `MODEL_ARCH=yolox` in .env without specifying a
    filename and get a sensible default.

    Args:
        model_arch: The model architecture enum value

    Returns:
        Default filename for that architecture (e.g. "yolov8n.pt")

    Raises:
        ModelConfigError: If model_arch has no known default
    """
    defaults = {
        ModelType.ULTRALYTICS: "yolov8n.pt",
        ModelType.YOLO_ONNX: "yolov8n.onnx",
        ModelType.YOLOX: "yolox_nano.onnx",
        ModelType.RFDETR: "rfdetr_r18vd.onnx",
        ModelType.DFINE: "dfine_n_coco.onnx",
        ModelType.OPENVINO: "person-detection-retail-0013.xml",
    }

    if model_arch not in defaults:
        raise ModelConfigError(
            f"Cannot infer default filename for architecture '{model_arch}'. "
            f"Set MODEL_ID explicitly in .env, or use one of: "
            f"{[m.value for m in ModelType]}",
            context={"model_arch": str(model_arch)}
        )

    return defaults[model_arch]


# ─────────────────────────────────────────────────────────────────────────────
# Model ID parsing — turns "MODEL_ID + MODEL_ARCH" settings into a relative path
# ─────────────────────────────────────────────────────────────────────────────

def model_id_provider(model_id: str, model_arch: ModelType | None) -> tuple[str, ModelType]:
    """
    Resolve (MODEL_ID, MODEL_ARCH) settings into a relative model path
    of the form "{arch}/{filename}".

    This function exists because users configure models in .env in several
    different ways depending on what they already know:
      - Only MODEL_ARCH set        -> use the default filename for that arch
      - MODEL_ID is just a filename + MODEL_ARCH set -> combine them
      - MODEL_ID is "arch/filename" -> use as-is (or override with MODEL_ARCH)
      - MODEL_ID is just an arch name (e.g. "ultralytics") -> treat as arch selector

    Args:
        model_id: Raw MODEL_ID string from settings (may be empty)
        model_arch: MODEL_ARCH enum from settings (may be None)

    Returns:
        (relative_path, resolved_arch) e.g. ("ultralytics/yolov8n.pt", ModelType.ULTRALYTICS)

    Raises:
        ModelConfigError: If the combination of settings is ambiguous or invalid
    """
    mid = model_id.strip() if model_id else ""
    valid_arch_dirs = {m.value for m in ModelType}

    # ── Case 1: No MODEL_ID at all ──
    if not mid:
        if model_arch is None:
            raise ModelConfigError(
                "Both MODEL_ID and MODEL_ARCH are empty in settings. "
                "Set at least MODEL_ARCH (e.g. MODEL_ARCH=ultralytics)."
            )
        filename = _infer_default_filename(model_arch)
        return f"{model_arch.value}/{filename}", model_arch

    parts = mid.rsplit("/", 1)

    # ── Case 2: MODEL_ID has no "/" — it's either a bare filename or an arch name ──
    if len(parts) == 1:
        if mid in valid_arch_dirs:
            # MODEL_ID is itself an arch name, e.g. MODEL_ID="ultralytics"
            final_arch = model_arch if model_arch is not None else ModelType(mid)
            filename = _infer_default_filename(final_arch)
            return f"{final_arch.value}/{filename}", final_arch

        # MODEL_ID is a bare filename, e.g. MODEL_ID="yolov8n.pt"
        if model_arch is None:
            raise ModelConfigError(
                f"MODEL_ID='{mid}' looks like a filename, but MODEL_ARCH is not set "
                f"so the model folder cannot be determined. "
                f"Either set MODEL_ARCH, or use MODEL_ID='<arch>/{mid}'.",
                context={"model_id": mid}
            )
        return f"{model_arch.value}/{mid}", model_arch

    # ── Case 3: MODEL_ID contains "/" — "folder/filename" or "org/repo" ──
    current_dir, current_file = parts[0], parts[1]

    if model_arch is not None:
        # Explicit MODEL_ARCH always wins over whatever folder is in MODEL_ID
        return f"{model_arch.value}/{current_file}", model_arch

    if current_dir in valid_arch_dirs:
        return mid, ModelType(current_dir)

    # Unknown folder and no MODEL_ARCH — this is a configuration error, not
    # something we should silently paper over by defaulting to OpenVINO.
    raise ModelConfigError(
        f"MODEL_ID='{mid}' has folder '{current_dir}' which is not a known "
        f"architecture, and MODEL_ARCH is not set. "
        f"Known architectures: {sorted(valid_arch_dirs)}",
        context={"model_id": mid, "unknown_folder": current_dir}
    )


# ─────────────────────────────────────────────────────────────────────────────
# Download helpers — one per source
# ─────────────────────────────────────────────────────────────────────────────

def _process_ultralytics_model(model_id: str, target_onnx_path: str) -> str:
    """
    Download an Ultralytics .pt model (via the ultralytics package, which
    handles its own caching/downloading) and export it to ONNX.

    Args:
        model_id: Ultralytics model name, e.g. "yolov8n.pt" or "yolov8n"
        target_onnx_path: Where the final .onnx file should end up

    Returns:
        target_onnx_path (the .onnx file now exists at this path)

    Raises:
        ModelDownloadError: If download or export fails
    """
    from ultralytics import YOLO

    pt_path = target_onnx_path.replace(".onnx", ".pt")
    logger.info(f"Downloading Ultralytics model '{model_id}' (will export to ONNX)")

    try:
        os.makedirs(os.path.dirname(target_onnx_path), exist_ok=True)

        # YOLO(...) downloads the .pt weights via ultralytics' own cache if not present
        model = YOLO(model_id, task="detect")

        export_args = {
            "format": "onnx",
            "imgsz": settings.FRAME_SHAPE[:2],  # (H, W) — match the pipeline's working resolution
            "dynamic": True,                     # required for batched inference
            "simplify": True,
            "opset": 13,
        }
        exported_path = model.export(**export_args)

        if not exported_path or not os.path.exists(exported_path):
            raise ModelDownloadError(
                "Ultralytics export() did not produce an output file",
                context={"model_id": model_id, "export_args": export_args}
            )

        # Move the exported .onnx to where resolve_model_path() expects it
        if os.path.abspath(exported_path) != os.path.abspath(target_onnx_path):
            shutil.move(exported_path, target_onnx_path)

        # Optionally keep the .pt for future re-exports
        pt_source = str(model.ckpt_path) if hasattr(model, "ckpt_path") and model.ckpt_path else None
        if pt_source and os.path.exists(pt_source):
            try:
                shutil.copy(pt_source, pt_path)
            except OSError as e:
                logger.warning(f"Could not cache .pt file alongside .onnx: {e}")

        del model
        logger.success(f"Ultralytics model exported to ONNX: {target_onnx_path}")
        return target_onnx_path

    except ModelDownloadError:
        raise
    except Exception as e:
        raise ModelDownloadError(
            f"Failed to download/export Ultralytics model '{model_id}'",
            context={"model_id": model_id, "error": str(e)}
        ) from e


def download_yolox_from_github(url: str, destination_path: str, timeout: int = 30) -> str:
    """
    Download a YOLOX ONNX model from its GitHub releases page.

    Args:
        url: Full download URL
        destination_path: Local path to save the file
        timeout: Request timeout in seconds

    Returns:
        destination_path (the file now exists at this path)

    Raises:
        ModelDownloadError: If the download fails
    """
    logger.info(f"Downloading YOLOX model from {url}")

    try:
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        tmp_dest = destination_path + ".part"

        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with open(tmp_dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        shutil.move(tmp_dest, destination_path)
        logger.success(f"Downloaded YOLOX model to {destination_path}")
        return destination_path

    except requests.RequestException as e:
        raise ModelDownloadError(
            f"Failed to download YOLOX model from GitHub",
            context={"url": url, "error": str(e)}
        ) from e


def download_from_hf(repo_id: str, final_local_dir: str, model_file_path: str) -> str:
    """
    Download a model file from a HuggingFace Hub repository.

    Args:
        repo_id: HuggingFace repo ID, e.g. "onnx-community/rfdetr"
        final_local_dir: Directory to download into
        model_file_path: Final path the model file should end up at

    Returns:
        model_file_path (the file now exists at this path)

    Raises:
        ModelDownloadError: If the download or move fails
    """
    filename = settings.HF_MODEL_FILENAME or "onnx/model_quantized.onnx"
    logger.info(f"Downloading '{filename}' from HuggingFace Hub repo '{repo_id}'")

    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=os.path.join(settings.BASE_DIR, "hf_cache"),
            local_dir=final_local_dir,
            local_dir_use_symlinks=False,
        )
        logger.success(f"Downloaded from HF Hub to {downloaded_path}")

    except Exception as e:
        raise ModelDownloadError(
            f"Failed to download '{filename}' from HF Hub repo '{repo_id}'",
            context={"repo_id": repo_id, "filename": filename, "error": str(e)}
        ) from e

    # Move into the expected final location, if different
    if os.path.abspath(downloaded_path) != os.path.abspath(model_file_path):
        try:
            os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
            shutil.move(downloaded_path, model_file_path)
            logger.success(f"Moved model to {model_file_path}")
        except OSError as e:
            raise ModelDownloadError(
                f"Downloaded model but failed to move it into place",
                context={"from": downloaded_path, "to": model_file_path, "error": str(e)}
            ) from e

    return model_file_path


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=8)
def resolve_model_path() -> str:
    """
    Resolve settings.MODEL_ID / settings.MODEL_ARCH into an absolute local
    path to a usable model file, downloading it if necessary.

    Cached with lru_cache — settings don't change at runtime, and this
    touches the filesystem/network, so repeated calls (e.g. one per camera
    using the same model) should not re-resolve from scratch.

    Returns:
        Absolute path to the model file (guaranteed to exist on success)

    Raises:
        ModelConfigError: If MODEL_ID/MODEL_ARCH settings are invalid
        ModelDownloadError: If the model needs downloading and that fails
        ModelResolutionError: For any other unexpected resolution failure
    """
    model_id = settings.MODEL_ID
    model_arch = settings.MODEL_ARCH

    try:
        relative_path, model_arch = model_id_provider(model_id, model_arch)
    except ModelConfigError:
        raise  # already has a clear message, just propagate

    # HuggingFace models (DFINE/RFDETR) live under a shared repo directory
    repo_id = relative_path.rsplit("/", 1)[0]
    repo_name = settings.HF_MODEL_REPONAME
    if repo_name and model_arch in (ModelType.DFINE, ModelType.RFDETR):
        relative_path = relative_path.replace(repo_id, repo_name)
        repo_id = repo_name

    final_local_dir = os.path.join(settings.BASE_DIR, "models", repo_id)
    model_file_path = os.path.join(settings.BASE_DIR, "models", relative_path)
    os.makedirs(final_local_dir, exist_ok=True)

    # ── Already exists locally — done ──
    if os.path.exists(model_file_path):
        logger.info(f"Using local model: {model_file_path}")
        return model_file_path

    logger.info(f"Model '{model_id}' (arch={model_arch.value}) not found locally "
               f"at {model_file_path} — attempting to download")

    if model_arch == ModelType.ULTRALYTICS or model_arch == ModelType.YOLO_ONNX:
        return _process_ultralytics_model(model_id, model_file_path)

    elif model_arch == ModelType.YOLOX:
        yolox_url = YOLOX_GITHUB_BASE_URL + os.path.basename(relative_path)
        return download_yolox_from_github(yolox_url, model_file_path)

    elif model_arch in (ModelType.DFINE, ModelType.RFDETR):
        return download_from_hf(model_id, final_local_dir, model_file_path)

    elif model_arch == ModelType.OPENVINO:
        raise ModelResolutionError(
            f"OpenVINO model not found at {model_file_path} and OpenVINO "
            f"models have no auto-download source. Place the .xml/.bin "
            f"files there manually.",
            context={"expected_path": model_file_path}
        )

    else:
        raise ModelResolutionError(
            f"No download strategy implemented for architecture '{model_arch}'",
            context={"model_arch": str(model_arch), "model_id": model_id}
        )
