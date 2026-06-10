import os
import re
import shutil
import requests
from loguru import logger
# from huggingface_hub import hf_hub_download, HfHubDisabledCache, HfHubError
from huggingface_hub import hf_hub_download
from src.core.config import settings, ModelType # Import ModelType for explicit routing
from src.core.exceptions import ModelResolutionError # Our custom exception

os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

# --- Constants ---
YOLOX_GITHUB_BASE_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"
# Regex to check if a string looks like a Hugging Face repo_id (e.g., "org/repo-name")
HF_REPO_ID_PATTERN = re.compile(r"^[a-zA-Z0-9-]+\/[a-zA-Z0-9-.]+$")


def _infer_default_filename(model_arch: ModelType) -> str:
    """
    Returns ONLY the default filename (not the full path) 
    based on the ModelType enum.
    """
    # Ultralytics / YoloV8
    if model_arch == ModelType.ULTRALYTICS:
        return "yolov8n.pt"
    
    # YoloX
    elif model_arch == ModelType.YOLOX:
        return "yolox_nano.onnx"
    
    # Onnx Community (DFINE / RFDETR)
    elif model_arch == ModelType.RFDETR:
        return "rfdetr_r18vd.onnx"

    elif model_arch == ModelType.DFINE:
        return "dfine_n_coco.onnx"
    
    # OpenVINO
    elif model_arch == ModelType.OPENVINO:
        return "person-detection-retail-0013.xml"
        
    # Fallback
    else:
        logger.warning(f"No specific default file for {model_arch}")
        raise ModelResolutionError(f"Cannot infer default filename for architecture '{model_arch}'. "
                           "Please ensure MODEL_ARCH in the .env is from this list: [yolox, dfine, rfdetr, ultralytics, openvino]")

def _process_ultralytics_model(repo_id_or_model_name: str, target_file_path: str):
    """
    Downloads an Ultralytics .pt model (or uses existing) and exports it to ONNX.
    """
    from ultralytics import YOLO
    logger.info(f"🔄 Processing Ultralytics model: {repo_id_or_model_name}")
    try:
        pt_model_path = target_file_path.replace('.onnx', '.pt') #if .onnx provided, The YOLO needs the .pt file first
        logger.info(f"🌐 Downloading Ultralytics model: {repo_id_or_model_name}")
        model = YOLO(pt_model_path, task='detect') 
        if target_file_path.lower().endswith(".onnx"):
            export_args = {
            "format": "onnx",
            "imgsz": 640, # Common input size, ensure your model is trained for this or adaptable
            "simplify": True,
            "opset": 13
            }
            # Ultralytics export creates a file in the same directory as the .pt model by default.
            # We need to ensure it lands in target_file_path.
            exported_onnx_file = model.export(**export_args)
            # exported_onnx_file = model.export(format="onnx", simplify=True, opset=13, imgsz=640)
            # Check if the exported file needs to be moved/renamed
            if not os.path.samefile(exported_onnx_file, pt_model_path):
                os.makedirs(os.path.dirname(pt_model_path), exist_ok=True)
                shutil.move(exported_onnx_file, pt_model_path)
                
            if not exported_onnx_file:
                raise Exception("Ultralytics export returned no results path.")
            
            logger.success(f"✅ Ultralytics model exported to ONNX: {pt_model_path}")
        
        del model
        return pt_model_path
        
    except Exception as e:
        logger.error(f"❌ Model not found locally and Failed to download model '{repo_id_or_model_name}' from Ultralytics: {e}")
        raise ModelResolutionError(f"Failed to process Ultralytics model '{repo_id_or_model_name}': {e}") from e

def download_yolox_from_github(url: str, destination_path: str, timeout: int = 30) -> None:
    """Safely downloads a file from a given URL to a destination."""
    logger.info(f"🌐 Downloading yolox model from URL: {url}")
    try:
        os.makedirs(os.path.dirname(destination_path), exist_ok=True) # Ensure destination dir exists
        tmp_dest = destination_path + ".part" # Download to temp file first
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status() # Raise an exception for bad status codes
            with open(tmp_dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        shutil.move(tmp_dest, destination_path) # Atomically move to final destination
        logger.success(f"✅ {url.rsplit('/',1)} Downloaded to: {destination_path} Successfully")
        return destination_path
    except requests.RequestException as exc:
        logger.error(f"❌ Failed to download the {url.rsplit('/',1)} model from GitHub: {exc}")
        raise ModelResolutionError(f"Failed to download from GitHub URL {url}: {exc}") from exc

def download_from_hf(repo_id: str, final_local_dir: str, model_file_path: str) -> str:
    # from huggingface_hub import hf_hub_download
    """Downloads a file from Hugging Face Hub."""
    filename = settings.HF_MODEL_FILENAME if settings.HF_MODEL_FILENAME else "onnx/model_quantized.onnx"
    logger.info(f"🌐 Downloading '{filename}' from Hugging Face Hub ({repo_id})...")
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1' 
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=os.path.join(settings.BASE_DIR, "hf_cache"), # Dedicated cache dir
            local_dir=final_local_dir, 
            local_dir_use_symlinks=False
        )
        logger.success(f"✅ Downloaded from HF Hub to: {downloaded_path}")
        try:
            shutil.move(downloaded_path, model_file_path)
            logger.success(f"✅ Moved {downloaded_path} -> {model_file_path}")
        except Exception as exc:
            logger.error(f"❌ Failed moving {downloaded_path} to {model_file_path}: {exc}")
            raise ModelResolutionError(f"An unexpected error occurred during moving {final_local_dir}/onnx/model_quantized.onnx to {model_file_path}: {e}") from e
        return downloaded_path
    # except HfHubError as e:
    #     raise ModelResolutionError(f"Failed to download '{filename}' from HF Hub repo '{repo_id}': {e}") from e
    except Exception as e:
        raise ModelResolutionError(f"Failed to download '{filename}' from HF Hub repo '{repo_id}': {e}") from e

def model_id_provider(model_id: str, model_arch: ModelType):
        """
        Resolves the model defined in settings to an absolute local file path,
        downloading from appropriate sources (local, Hugging Face, GitHub) if necessary.
        
        Returns:
            The absolute local path to the model file.
            
        Raises:
            FileNotFoundError: If the model cannot be found or downloaded.
            
        It can even handle the cases where we only have Model_ID with no Model_Arch 
        """
        # Clean input (handle None or whitespace)
        mid = model_id.strip() if model_id else ""
        
        # Get a set of valid directory names (values of the Enum)
        valid_arch_dirs = {m.value for m in ModelType}
        # ---------------------------------------------------------
        # SCENARIOS 1 & 2: No model_id provided
        # ---------------------------------------------------------
        if not mid:
            if not model_arch:
                # Scenario 1: No ID, No Arch ==> Error
                raise ValueError("Error: Both model_id and model_arch are missing. Please specify at least one.")
            else:
                # Scenario 2: No ID, Arch exists ==> Use Default
                filename = _infer_default_filename(model_arch)
                return f"{model_arch.value}/{filename}", model_arch

        # Analyze structure of input ID
        parts = mid.rsplit('/', 1)
        
        # ---------------------------------------------------------
        # HANDLING NO SLASH (Could be a Filename OR an Arch/Dir name)
        # ---------------------------------------------------------
        if len(parts) == 1:
            # Check if the single string is actually a known architecture directory (e.g., 'ultralytics')
            is_directory_name = mid in valid_arch_dirs

            if is_directory_name:
                # SCENARIO 5 & 6: Input is just the Arch name (e.g. 'ultralytics')
                # If model_arch is provided (Scenario 6), it technically overrides, 
                # but usually, they match. If not (Scenario 5), we infer arch from the string.
                
                # If explicit arch is given, use it; otherwise, look up Enum by value
                final_arch = model_arch if model_arch else ModelType(mid)
                filename = _infer_default_filename(final_arch)
                return f"{final_arch.value}/{filename}", final_arch
                
            else:
                # SCENARIO 3 & 4: Input is a Filename (e.g. 'yolov8n.pt')
                if not model_arch:
                    # Scenario 3: Filename given, but we don't know the folder ==> Error
                    raise ValueError(f"Error: model_id '{mid}' looks like a filename, but no model_arch was specified to provide the folder path.")
                else:
                    # Scenario 4: Filename + Arch ==> Combine
                    return f"{model_arch.value}/{mid}", model_arch

        # ---------------------------------------------------------
        # HANDLING WITH SLASH (Full Path Input)
        # ---------------------------------------------------------
        else: 
            # model_id looks like "folder/file.pt" or "wrong/file.pt"
            # We assume the first part is the folder, the rest is the file
            current_dir = parts[0]
            current_file = "/".join(parts[1:]) # Joins rest in case file has sub-slashes
            
            if model_arch:
                # SCENARIO 8: Full path + Arch ==> Override/Correction
                # The user provided a path, but ALSO an explicit architecture.
                # We trust the architecture (ModelType) and force that directory,
                # ignoring the directory inside model_id.
                return f"{model_arch.value}/{current_file}", model_arch
                
            else:
                # SCENARIO 7: Full path + No Arch ==> Validate
                # We must check if the folder provided in model_id is valid.
                if current_dir in valid_arch_dirs:
                    return mid, ModelType(current_dir)
                else:
                    # Fallback / Error if the folder in string is unknown and no Arch provided
                    logger.warning(f"The provided directory '{current_dir}' is not a known ModelType. defaulting to OpenVINO")
                    return  'openvino/'+_infer_default_filename(ModelType('openvino')), ModelType('openvino')
                
        
def resolve_model_path() -> str:
    model_id = settings.MODEL_ID
    model_arch = settings.MODEL_ARCH 
    filename_to_use, model_arch =  model_id_provider(model_id, model_arch)   

    # 2. Determine the target local directory for this model
    # Hierarchical storage: models/{MODEL_ARCH_NAME}/{MODEL_ID_NORMALIZED}/filename
    # E.g., models/yolov8/ultralytics_yolov8n-pt/yolov8n.pt
    repo_id = filename_to_use.rsplit('/', 1)[0]
    repo_name = settings.HF_MODEL_REPONAME
    if repo_name and model_arch in [ModelType.DFINE, ModelType.RFDETR]:  # Huggingace files are from a repositoy with the name: HF_MODEL_REPONAME
        filename_to_use = filename_to_use.replace(repo_id, repo_name)
        repo_id = repo_name
    final_local_dir = os.path.join(settings.BASE_DIR, "models", repo_id)
    model_file_path = os.path.join(settings.BASE_DIR, "models", filename_to_use)
    os.makedirs(final_local_dir, exist_ok=True)
    
    # 3. If the resolved file already exists locally, use it
    if os.path.exists(model_file_path):
        logger.info(f"💾 Using local model: {model_file_path}")
        return model_file_path
    
    # 4. If not local, attempt download/processing based on MODEL_ARCH and MODEL_ID format
    logger.info(f"⏳ Model '{model_id}' (arch: {model_arch}) not found locally. Attempting to resolve...")
        
    # Download ultralytics (can download the onnx file too)
    if model_arch == ModelType.ULTRALYTICS:
        model_file_path = _process_ultralytics_model(model_id, model_file_path)
    
    # Download the yolox model from the Github
    elif model_arch == ModelType.YOLOX:
        yolox_url = os.path.join(YOLOX_GITHUB_BASE_URL, filename_to_use)  #???????????????????????? YOLOX_GITHUB_BASE_URL/yolox/yolox_nano.onnx    or YOLOX_GITHUB_BASE_URL/yolox_nano.onnx
        model_file_path = download_yolox_from_github(yolox_url, model_file_path)
    
    # 4. If not local, and it looks like a Hugging Face repo_id, download it
    elif model_arch in [ModelType.DFINE, ModelType.RFDETR]:
        model_file_path = download_from_hf(model_id, final_local_dir, model_file_path)
    else:
        logger.error(f"❌ An UnExpected Error happend while choosing the correct model path for {model_arch} based on the ModelArch")
    return model_file_path    