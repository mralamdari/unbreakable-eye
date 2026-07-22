"""
Model path resolution and downloading.

Called ONCE at startup by factory.py — not on the per-frame hot path.
Correctness and clear error messages matter more than speed here.

Resolution order for every architecture:
  1. Check if the model file already exists locally under models/
  2. If not, download from the appropriate source:
       ULTRALYTICS / YOLO_ONNX → ultralytics package (.pt export to .onnx)
       YOLOX               → GitHub releases (.onnx direct download)
       RFDETR / DFINE      → HuggingFace Hub (.onnx via hf_hub_download)
       OPENVINO            → Intel Open Model Zoo (.xml + .bin via urllib)
"""

import os
import shutil
import urllib.request
from functools import lru_cache
from typing import Optional

import requests
from loguru import logger
from huggingface_hub import hf_hub_download

from src.core.config import settings, ModelType
from src.core.exceptions import ModelConfigError, ModelDownloadError, ModelResolutionError

# ── Environment ───────────────────────────────────────────────────────────────
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

# ── Constants ─────────────────────────────────────────────────────────────────
_YOLOX_GITHUB_BASE = (
    "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"
)
_OPENVINO_OMZ_BASE = (
    "https://storage.openvinotoolkit.org/repositories/open_model_zoo/"
    "2023.0/models_bin/1/person-detection-retail-0013"
)

# Default filename for each architecture — used when MODEL_ID is empty
_ARCH_DEFAULTS: dict[ModelType, str] = {
    ModelType.ULTRALYTICS: "yolov8n.pt",
    ModelType.YOLO_ONNX:   "yolov8n.onnx",
    ModelType.YOLOX:       "yolox_nano.onnx",
    ModelType.RFDETR:      "rfdetr_r18vd.onnx",
    ModelType.DFINE:       "dfine_n_coco.onnx",
    ModelType.OPENVINO:    "person-detection-retail-0013.xml",
}


# ─────────────────────────────────────────────────────────────────────────────
# Config parsing — MODEL_ID + MODEL_ARCH → relative local path
# ─────────────────────────────────────────────────────────────────────────────

def _default_filename(model_arch: ModelType) -> str:
    """
    Return the canonical default filename for *model_arch*.

    Lets users set only MODEL_ARCH in .env and get a working default.

    Raises:
        ModelConfigError: If no default exists for this architecture.
    """
    try:
        return _ARCH_DEFAULTS[model_arch]
    except KeyError:
        raise ModelConfigError(
            f"No default filename for architecture '{model_arch.value}'. "
            f"Set MODEL_ID explicitly in .env.",
            context={"model_arch": model_arch.value,
                     "known": [m.value for m in ModelType]},
        )


def model_id_provider(
    model_id: str,
    model_arch: Optional[ModelType],
) -> tuple[str, ModelType]:
    """
    Resolve (MODEL_ID, MODEL_ARCH) into a relative model path and concrete arch.

    For HF-hosted architectures (DFINE, RFDETR) the local folder prefix is
    HF_MODEL_REPONAME (e.g. "onnx-community") to match the on-disk layout:
        models/onnx-community/rfdetr_nano-ONNX/onnx/model_quantized.onnx

    For all other architectures the folder prefix is the arch value itself:
        models/ultralytics/yolov8n.pt

    Handles four .env patterns users actually write:
      1. MODEL_ARCH only          → "<prefix>/<default_file>"
      2. MODEL_ID=<filename>      → "<prefix>/<filename>"  (MODEL_ARCH required)
      3. MODEL_ID=<arch>          → "<prefix>/<default_file>"
      4. MODEL_ID=<arch>/<file>   → "<prefix>/<file>"

    Returns:
        (relative_path, resolved_arch)
        e.g. ("onnx-community/rfdetr_nano-ONNX", ModelType.RFDETR)
             ("ultralytics/yolov8n.pt",           ModelType.ULTRALYTICS)

    Raises:
        ModelConfigError: If the combination is ambiguous or invalid.
    """
    _HF_ARCHS = {ModelType.DFINE, ModelType.RFDETR}

    def _folder_prefix(arch: ModelType) -> str:
        """Local folder prefix — HF archs use HF_MODEL_REPONAME, others use arch value."""
        return settings.HF_MODEL_REPONAME if arch in _HF_ARCHS else arch.value

    mid           = (model_id or "").strip()
    valid_arch_values = {m.value: m for m in ModelType}

    # ── Case 1: No MODEL_ID ───────────────────────────────────────────────────
    if not mid:
        if model_arch is None:
            raise ModelConfigError(
                "Both MODEL_ID and MODEL_ARCH are unset. "
                "Set at least MODEL_ARCH in .env (e.g. MODEL_ARCH=yolo_onnx)."
            )
        prefix = _folder_prefix(model_arch)
        return f"{prefix}/{_default_filename(model_arch)}", model_arch

    # ── Case 2 / 3: No slash — bare filename or bare arch name ───────────────
    if "/" not in mid:
        if mid in valid_arch_values:
            # MODEL_ID="rfdetr" — treat as arch selector
            arch   = model_arch or valid_arch_values[mid]
            prefix = _folder_prefix(arch)
            return f"{prefix}/{_default_filename(arch)}", arch

        # MODEL_ID="rfdetr_nano-ONNX" — bare model name, MODEL_ARCH must be set
        if model_arch is None:
            raise ModelConfigError(
                f"MODEL_ID='{mid}' looks like a filename but MODEL_ARCH is not set. "
                f"Either add MODEL_ARCH=<arch> or use MODEL_ID=<arch>/{mid}.",
                context={"model_id": mid},
            )
        prefix = _folder_prefix(model_arch)
        return f"{prefix}/{mid}", model_arch

    # ── Case 4: Contains slash — "folder/filename" ────────────────────────────
    folder, filename = mid.rsplit("/", 1)

    if model_arch is not None:
        # Explicit MODEL_ARCH always wins — rewrite folder to correct prefix
        prefix = _folder_prefix(model_arch)
        return f"{prefix}/{filename}", model_arch

    if folder in valid_arch_values:
        arch   = valid_arch_values[folder]
        prefix = _folder_prefix(arch)
        return f"{prefix}/{filename}", arch

    # folder might already be the HF namespace (e.g. "onnx-community")
    if folder == settings.HF_MODEL_REPONAME:
        # Can't determine arch from HF namespace alone — MODEL_ARCH required
        raise ModelConfigError(
            f"MODEL_ID='{mid}' uses the HF namespace as folder but MODEL_ARCH is not set. "
            f"Add MODEL_ARCH=dfine or MODEL_ARCH=rfdetr to .env.",
            context={"model_id": mid},
        )

    raise ModelConfigError(
        f"MODEL_ID='{mid}' has unknown folder '{folder}' and MODEL_ARCH is not set. "
        f"Known architectures: {sorted(valid_arch_values)}",
        context={"model_id": mid, "unknown_folder": folder},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Download helpers — one per source
# ─────────────────────────────────────────────────────────────────────────────

def _download_ultralytics(model_id: str, target_onnx_path: str) -> str:
    """
    Download an Ultralytics .pt model and export it to ONNX.

    The ultralytics package handles its own weight caching/downloading.
    The resulting .onnx is moved to *target_onnx_path*.

    Args:
        model_id:        Ultralytics model name, e.g. "yolov8n" or "yolov8n.pt".
        target_onnx_path: Absolute path where the .onnx file should land.

    Returns:
        target_onnx_path

    Raises:
        ModelDownloadError: If download or export fails.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ModelDownloadError(
            "ultralytics package not installed — cannot download/export YOLO model. "
            "Run: pip install ultralytics",
            context={"model_id": model_id},
        )

    logger.info(f"Downloading Ultralytics model '{model_id}' and exporting to ONNX")
    try:
        os.makedirs(os.path.dirname(target_onnx_path), exist_ok=True)

        model       = YOLO(model_id, task="detect")
        export_args = {
            "format":   "onnx",
            "imgsz":    settings.FRAME_SHAPE[:2],  # (H, W) — match pipeline resolution
            "dynamic":  True,                       # required for predict_batch()
            "simplify": True,
            "opset":    13,
        }
        exported = model.export(**export_args)

        if not exported or not os.path.exists(str(exported)):
            raise ModelDownloadError(
                "Ultralytics export() did not produce an output file",
                context={"model_id": model_id, "export_args": export_args},
            )

        # Move to the expected location if ultralytics placed it elsewhere
        if os.path.abspath(str(exported)) != os.path.abspath(target_onnx_path):
            shutil.move(str(exported), target_onnx_path)

        # Cache the .pt alongside the .onnx for future re-exports
        pt_path = target_onnx_path.replace(".onnx", ".pt")
        ckpt    = getattr(model, "ckpt_path", None)
        if ckpt and os.path.exists(str(ckpt)):
            try:
                shutil.copy(str(ckpt), pt_path)
            except OSError as err:
                logger.warning(f"Could not cache .pt alongside .onnx: {err}")

        del model
        logger.success(f"Ultralytics model exported → {target_onnx_path}")
        return target_onnx_path

    except ModelDownloadError:
        raise
    except Exception as e:
        raise ModelDownloadError(
            f"Failed to download/export Ultralytics model '{model_id}'",
            context={"model_id": model_id, "error": str(e)},
        ) from e


def _download_yolox_github(url: str, destination: str, timeout: int = 60) -> str:
    """
    Download a YOLOX ONNX file from its GitHub releases page.

    Uses a .part temporary file so a failed download never leaves a
    corrupted file at *destination*.

    Args:
        url:         Full download URL.
        destination: Local path for the final .onnx file.
        timeout:     HTTP request timeout in seconds.

    Returns:
        destination

    Raises:
        ModelDownloadError: If the HTTP request fails.
    """
    logger.info(f"Downloading YOLOX model | url={url}")
    tmp = destination + ".part"
    try:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with requests.get(url, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            done  = 0
            counter = 0
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=65_536):
                    if chunk:
                        fh.write(chunk)
                        done += len(chunk)
                        counter += 1
                        if total and counter%50==0:
                            counter = 0
                            logger.debug(
                                f"YOLOX download: {done/1024/1024:.1f} MB "
                                f"/ {total/1024/1024:.1f} MB"
                            )
        shutil.move(tmp, destination)
        logger.success(f"YOLOX model downloaded → {destination}")
        return destination

    except requests.RequestException as e:
        raise ModelDownloadError(
            "Failed to download YOLOX model from GitHub",
            context={"url": url, "error": str(e)},
        ) from e
    finally:
        # Always clean up the .part file on failure
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _download_from_hf(
    repo_id: str,
    hf_filename: str,
    final_path: str,
    local_dir: str,
) -> str:
    """
    Download a model file from HuggingFace Hub.

    Args:
        repo_id:    HF repo, e.g. "onnx-community/rfdetr_small-ONNX".
        hf_filename: Path inside the repo, e.g. "onnx/model_quantized.onnx".
        final_path: Absolute local path where the file should end up.
        local_dir:  Directory for hf_hub_download to place the file.

    Returns:
        final_path

    Raises:
        ModelDownloadError: If the download or move fails.
    """
    # Split "subfolder/filename" if present
    if "/" in hf_filename:
        subfolder, filename = hf_filename.rsplit("/", 1)
    else:
        subfolder, filename = None, hf_filename

    logger.info(
        f"Downloading from HF Hub | repo={repo_id} "
        f"| file={hf_filename}"
    )
    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            subfolder=subfolder,
            cache_dir=os.path.join(settings.BASE_DIR, "hf_cache"),
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        logger.success(f"HF download complete → {downloaded}")

    except Exception as e:
        raise ModelDownloadError(
            f"Failed to download '{hf_filename}' from HF Hub repo '{repo_id}'",
            context={"repo_id": repo_id, "filename": hf_filename, "error": str(e)},
        ) from e

    # Move to the canonical final path if hf_hub placed it elsewhere
    if os.path.abspath(downloaded) != os.path.abspath(final_path):
        try:
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
            shutil.move(downloaded, final_path)
            logger.success(f"Model moved → {final_path}")
        except OSError as e:
            raise ModelDownloadError(
                "Downloaded from HF Hub but failed to move file into place",
                context={"from": downloaded, "to": final_path, "error": str(e)},
            ) from e

    return final_path


def _download_openvino(precision: str, destination_dir: str) -> str:
    """
    Download person-detection-retail-0013 (.xml + .bin) from Intel Open Model Zoo.

    Both files MUST be present for OpenVINO to load the model — the function
    raises ModelDownloadError if either file fails.

    Args:
        precision:       Model precision variant: "FP32", "FP16", or "FP16-INT8".
        destination_dir: Directory where .xml and .bin will be saved.

    Returns:
        Absolute path to the .xml file (the model entry point).

    Raises:
        ModelDownloadError: If either file cannot be downloaded.
    """
    model_name = "person-detection-retail-0013"
    files      = [f"{model_name}.xml", f"{model_name}.bin"]
    base_url   = f"{_OPENVINO_OMZ_BASE}/{precision}"

    logger.info(
        f"Downloading OpenVINO model | precision={precision} "
        f"| destination={destination_dir}"
    )
    os.makedirs(destination_dir, exist_ok=True)

    for filename in files:
        url       = f"{base_url}/{filename}"
        dest_path = os.path.join(destination_dir, filename)
        tmp_path  = dest_path + ".part"

        logger.info(f"Fetching {url}")
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp, \
                 open(tmp_path, "wb") as fh:
                fh.write(resp.read())
            shutil.move(tmp_path, dest_path)
            size_kb = os.path.getsize(dest_path) / 1024
            logger.success(f"Downloaded {filename} ({size_kb:.0f} KB)")

        except Exception as e:
            # Clean up partial file before raising
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise ModelDownloadError(
                f"Failed to download OpenVINO model file '{filename}'",
                context={"url": url, "error": str(e)},
            ) from e

    xml_path = os.path.join(destination_dir, f"{model_name}.xml")
    logger.success(f"OpenVINO model ready → {xml_path}")
    return xml_path


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=8)
def resolve_model_path() -> str:
    """
    Resolve settings into an absolute path to a usable model file.

    Checks for a local copy first; downloads if missing.
    Cached with lru_cache so repeated calls (e.g. one per camera subprocess
    using the same model) skip redundant filesystem/network operations.

    Returns:
        Absolute path to the model file (guaranteed to exist on return).

    Raises:
        ModelConfigError:     MODEL_ID / MODEL_ARCH settings are invalid.
        ModelDownloadError:   Download or export failed.
        ModelResolutionError: No download strategy for this architecture,
                              or any other unexpected resolution failure.
    """
    model_id   = settings.MODEL_ID
    model_arch = settings.MODEL_ARCH

    # ── 1. Parse settings into a relative path ──────────────────────────────
    try:
        relative_path, model_arch = model_id_provider(model_id, model_arch)
    except ModelConfigError:
        raise

    logger.debug(
        f"Resolving model | id={model_id!r} | arch={model_arch.value} "
        f"| relative={relative_path}"
    )

    # ── 2. Build absolute paths ──────────────────────────────────────────────
    models_root   = os.path.join(settings.BASE_DIR, "models")
    model_dir     = os.path.join(models_root, relative_path.rsplit("/", 1)[0])
    model_file    = os.path.join(models_root, relative_path)

    # HF models nest an extra subfolder (e.g. "onnx/model_quantized.onnx")
    hf_filename   = settings.HF_MODEL_FILENAME or "onnx/model_quantized.onnx"
    hf_model_file = os.path.join(models_root, relative_path, hf_filename)

    # OpenVINO needs .xml at a predictable path inside the model dir
    ov_xml_file   = os.path.join(
        models_root, relative_path, "person-detection-retail-0013.xml"
    )

    os.makedirs(model_dir, exist_ok=True)

    # ── 3. Return immediately if already local ───────────────────────────────
    if model_arch == ModelType.OPENVINO:
        if os.path.exists(ov_xml_file):
            logger.info(f"Using local OpenVINO model: {ov_xml_file}")
            return ov_xml_file

    elif model_arch in (ModelType.DFINE, ModelType.RFDETR):
        if os.path.exists(hf_model_file):
            logger.info(f"Using local HF model: {hf_model_file}")
            return hf_model_file

    else:
        # YOLO ONNX, YOLOX, Ultralytics — single file at model_file
        if os.path.exists(model_file):
            logger.info(f"Using local model: {model_file}")
            return model_file

    # ── 4. Download ──────────────────────────────────────────────────────────
    logger.info(
        f"Model not found locally — downloading | "
        f"arch={model_arch.value} | id={model_id!r}"
    )

    if model_arch in (ModelType.ULTRALYTICS, ModelType.YOLO_ONNX):
        return _download_ultralytics(model_id, model_file)

    elif model_arch == ModelType.YOLOX:
        filename  = os.path.basename(relative_path)
        yolox_url = _YOLOX_GITHUB_BASE + filename
        return _download_yolox_github(yolox_url, model_file)

    elif model_arch in (ModelType.DFINE, ModelType.RFDETR):
        # relative_path is e.g. "onnx-community/rfdetr_nano-ONNX"
        # The HF repo_id is exactly that: "onnx-community/rfdetr_nano-ONNX"
        # (HF_MODEL_REPONAME / model_name) — already correct because
        # model_id_provider now uses HF_MODEL_REPONAME as the folder prefix.
        hf_repo_id   = relative_path          # "onnx-community/rfdetr_nano-ONNX"
        hf_local_dir = os.path.join(models_root, relative_path)
        os.makedirs(
            os.path.join(hf_local_dir, hf_filename.split("/")[0]),
            exist_ok=True
        )
        return _download_from_hf(
            repo_id=hf_repo_id,
            hf_filename=hf_filename,
            final_path=hf_model_file,
            local_dir=hf_local_dir,
        )

    elif model_arch == ModelType.OPENVINO:
        # precision comes from MODEL_ID when OpenVINO arch is selected
        # e.g. MODEL_ID=FP16  or  MODEL_ID=FP32
        precision    = model_id.upper() if model_id.upper() in ("FP32", "FP16", "FP16-INT8") \
                       else "FP16"
        ov_model_dir = os.path.join(models_root, relative_path)
        return _download_openvino(precision, ov_model_dir)

    else:
        raise ModelResolutionError(
            f"No download strategy for architecture '{model_arch.value}'. "
            f"Place the model file manually at: {model_file}",
            context={"model_arch": model_arch.value, "expected_path": model_file},
        )