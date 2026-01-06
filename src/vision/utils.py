import os
import requests
from loguru import logger
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from src.core.config import settings

# Regex to check if a string looks like a Hugging Face repo_id (e.g., "org/repo-name")
YOLOX_BASE_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"

def download_file(url, destination):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        raise Exception(f"Failed to download from {url}")

def resolve_model_path(model_id: str) -> str:
    """
    Resolves the model ID to an absolute local file path, downloading from HF Hub if necessary.
    
    Args:
        model_id: The identifier for the model (can be a local path, or HF repo_id).
        
    Returns:
        The absolute local path to the model file.
        
    Raises:
        FileNotFoundError: If the model cannot be found or downloaded.
    """
    repo_id, file_name = model_id.split('/')
    final_local_dir = os.path.join(settings.BASE_DIR, "models", repo_id)
    model_dir = os.path.join(settings.BASE_DIR, "models", model_id)
    os.makedirs(final_local_dir, exist_ok=True) # Ensure path exists
    
    # 3. Check if the file already exists locally
    if os.path.exists(model_dir):
        logger.info(f"💾 Using local model: {model_dir}")
        return model_dir

    # Download ultralytics (can download the onnx file too)
    if 'ultralytics' in model_id.lower():
        try:
            if '.onnx' in model_id:
                model_dir = model_dir.replace('.onnx', '.pt')
            logger.info(f"🌐 Downloading Ultralytics model: {file_name}")
            model = YOLO(model_dir, task='detect') 
            if '.onnx' in model_id:
                model.export(format="onnx")
            del model
            logger.info(f"✅ Model downloaded to: {model_dir}")
            return model_dir
        except Exception as e:
            logger.error(f"❌ Model not found locally and Failed to download model '{model_id}' from Hub: {e}'")
            raise FileNotFoundError(f"Model '{model_id}' not found locally and couldn't download it.")
    
    # Download the yolox model from the Github
    if 'yolox' in model_id.lower():
        try:
            model_url = YOLOX_BASE_URL+file_name
            logger.info(f"🌐 Downloading yolox model: {model_dir}")
            download_file(model_url, model_dir)
            logger.info(f"✅ Model downloaded to: {model_dir}")
            return model_dir
        except Exception as e:
            logger.error(f"❌ Model not found locally and Failed to download model '{model_id}' from Hub: {e}'")
            raise FileNotFoundError(f"Model '{model_id}' not found locally and couldn't download it.")

    # 4. If not local, and it looks like a Hugging Face repo_id, download it
    try:
        logger.info(f"🌐 Downloading '{model_id}' from Hugging Face Hub ({repo_id})...")
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1' 
        downloaded_path = hf_hub_download(
            repo_id=model_id,
            filename="onnx/model_quantized.onnx",
            local_dir=final_local_dir,
            local_dir_use_symlinks=False
        )
        os.rename(downloaded_path, model_dir)
        logger.success(f"✅ Model downloaded to: {model_dir}")
        os.rmdir(os.path.join(settings.BASE_DIR, 'models/onnx-community/onnx'))
        return model_dir
    except Exception as e:
        # If not local, and not a valid HF ID pattern, then it's an error
        logger.error(f"❌ Model not found locally and Failed to download model '{model_id}' from Hub: {e}'")
        raise FileNotFoundError(f"Model '{model_id}' not found locally or on Hugging Face Hub.")


# MODEL_ID='openvino/person-detection-retail-0013.xml'

# MODEL_ID='onnx-community/dfine_x_obj365-ONNX'
# MODEL_ID='onnx-community/dfine_n_coco-ONNX'
# MODEL_ID='onnx-community/dfine_s_coco-ONNX'
# MODEL_ID='onnx-community/dfine_x_coco-ONNX'
# MODEL_ID='onnx-community/dfine_l_coco-ONNX'
# MODEL_ID='onnx-community/dfine_m_coco-ONNX'
# MODEL_ID='onnx-community/dfine_s_obj365-ONNX'
# MODEL_ID='onnx-community/dfine_m_obj365-ONNX'
# MODEL_ID='onnx-community/dfine_s_obj2coco-ONNX'
# MODEL_ID='onnx-community/dfine_m_obj2coco-ONNX'
# MODEL_ID='onnx-community/rfdetr_base-ONNX'
# MODEL_ID='onnx-community/rfdetr_nano-ONNX'
# MODEL_ID='onnx-community/rfdetr_large-ONNX'
# MODEL_ID='onnx-community/rfdetr_small-ONNX'
# MODEL_ID='onnx-community/rfdetr_medium-ONNX'

# MODEL_ID='yolox/yolox_nano.onnx'
# MODEL_ID='yolox/yolox_s.onnx'
# MODEL_ID='yolox/yolox_tiny.onnx'
# MODEL_ID='yolox/yolox_m.onnx'

# MODEL_ID='ultralytics/yolov8n.onnx'
# MODEL_ID='ultralytics/yolov8n.pt'
# MODEL_ID='ultralytics/yolo11n.onnx'
# MODEL_ID='ultralytics/yolo11n.pt'


# RTSP_URL=0
# DEVICE='cpu'
# CONF_THRESHOLD=0.5
# a = resolve_model_path(MODEL_ID)
# print(a)
    

