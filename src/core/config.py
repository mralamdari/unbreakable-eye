import os
from enum import Enum
from pydantic_settings import BaseSettings, SettingsConfigDict

class Device(str, Enum):
    CPU = "cpu"
    CUDA = "cuda" # Standard PyTorch name for GPU
    GPU = "gpu"   # Common alias or for OpenVINO

class Settings(BaseSettings):
    # --- APP INFO ---
    PROJECT_NAME: str = "Unbreakable Eye"
    
    # --- LOGGING ---
    LOG_LEVEL: str = "INFO" # DEBUG, INFO, WARNING, ERROR
    LOG_JSON: bool = False  # Set to True for structured logs in production

    # --- INPUT ---
    RTSP_URL: str = "0" # "0" for webcam, or actual RTSP link

    # --- MODEL IDENTIFIER & ARCHITECTURE ---
    # MODEL_ID: str = "ultralytics/yolov8n-pt"
    MODEL_ID: str = "openvino/person-detection-retail-0013.xml"
    
    # --- HARDWARE ---
    DEVICE: Device = Device.CPU

    # --- THRESHOLDS ---
    CONF_THRESHOLD: float = 0.5
    NMS_THRESHOLD: float = 0.4
    CLASS_AGNOSTIC: bool = True # YOLOX specific

    # --- SERVER ---
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # --- COMPUTED PROPERTIES ---
    # BASE_DIR is fundamental for resolving local paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Pydantic Settings configuration (essential for .env loading)
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_ignore_empty=True,
        extra="ignore" # Ignore unknown variables in .env without crashing
    )

# Instantiate settings once at startup
settings = Settings()