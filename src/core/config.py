import os
from enum import Enum
from pydantic_settings import BaseSettings, SettingsConfigDict

# 1. Enums
class ModelType(str, Enum):
    YOLOX = "yolox"
    DFINE = "dfine"
    RTDETR = "rtdetr"
    YOLOV8 = "yolov8"
    OPENVINO = "openvino"

class Device(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    GPU = "gpu"

# 2. The Settings Class
class Settings(BaseSettings):
    # --- APP INFO (Matches PROJECT_NAME in .env) ---
    PROJECT_NAME: str = "Unbreakable Eye"
    
    # --- LOGGING (Matches LOG_LEVEL in .env) ---
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = False

    # --- INPUT (Matches RTSP_URL in .env) ---
    # We allow str because it could be "0" (webcam) or "rtsp://..."
    RTSP_URL: str = "0"

    # --- MODEL CONFIG (Matches MODEL_ARCH, MODEL_PATH in .env) ---
    MODEL_ARCH: ModelType = ModelType.YOLOX
    MODEL_PATH: str = "models/yolov8n.pt"
    
    # --- HARDWARE (Matches DEVICE in .env) ---
    DEVICE: Device = Device.CPU

    # --- THRESHOLDS (Matches CONF_THRESHOLD in .env) ---
    CONF_THRESHOLD: float = 0.5
    NMS_THRESHOLD: float = 0.4
    CLASS_AGNOSTIC: bool = True

    # --- SERVER ---
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # --- COMPUTED PROPERTIES ---
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    @property
    def absolute_model_path(self) -> str:
        # If the path is already absolute, return it. Otherwise join with base.
        if os.path.isabs(self.MODEL_PATH):
            return self.MODEL_PATH
        return os.path.join(self.BASE_DIR, self.MODEL_PATH)

    # 3. Config Rules
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_ignore_empty=True,
        extra="ignore"  # <--- THIS IS THE FIX. It ignores unknown variables instead of crashing.
    )

settings = Settings()
