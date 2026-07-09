import os
from enum import Enum
from pydantic_settings import BaseSettings, SettingsConfigDict

# 1. Enums: Provide strict types for model architecture and device
class ModelType(str, Enum):
    YOLOX       = "yolox"
    DFINE       = "dfine"        # type-safe discriminator — HF repo stays in HF_MODEL_REPONAME
    RFDETR      = "rfdetr"       # type-safe discriminator — HF repo stays in HF_MODEL_REPONAME
    ULTRALYTICS = "ultralytics"
    OPENVINO    = "openvino"
    YOLO_ONNX   = "yolo_onnx"
    

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

    SECRET_KEY: str = ""  # Must be set in .env — no default for security

    WORKING_HEIGHT: int = 512
    WORKING_WIDTH:  int = 512
    WORKING_CHANNELS: int = 3
    DISPLAY_SHAPE: tuple = (1080, 1920, 3)          # H, W, C  — output SHM resolution
    DISPLAY_BYTES: int   = 1920 * 1080 * 3          # = 6_220_800
    
    
    
    DISPLAY_WIDTH:   int = 1920   # native camera resolution or desired display size
    DISPLAY_HEIGHT:  int = 1080
    
    
    # Display/native resolution — used for display SHM and crop extraction
    NATIVE_WIDTH:     int = 1920
    NATIVE_HEIGHT:    int = 1080

    @property
    def NATIVE_SHAPE(self) -> tuple[int, int, int]:
        return (self.NATIVE_HEIGHT, self.NATIVE_WIDTH, self.WORKING_CHANNELS)

    @property
    def NATIVE_BYTES(self) -> int:
        return self.NATIVE_HEIGHT * self.NATIVE_WIDTH * self.WORKING_CHANNELS


    @property
    def FRAME_SHAPE(self) -> tuple[int, int, int]:
        return (self.WORKING_HEIGHT, self.WORKING_WIDTH, self.WORKING_CHANNELS)

    @property
    def FRAME_BYTES(self) -> int:
        return self.WORKING_HEIGHT * self.WORKING_WIDTH * self.WORKING_CHANNELS

    # --- Dataset ---
    POSTGRES_HOST:     str = "localhost"
    POSTGRES_PORT:     int = 5432
    POSTGRES_DB:       str = "unbreakable_eye"
    POSTGRES_USER:     str = "app"
    POSTGRES_PASSWORD: str = ""
    POSTGRES_POOL_MIN: int = 2
    POSTGRES_POOL_MAX: int = 10

    # --- HARDWARE ---
    DEVICE: Device = Device.CPU
    HF_MODEL_FILENAME: str =  "onnx/model_quantized.onnx"
    HF_MODEL_REPONAME: str =  "onnx-community"
    LAMBDA_DISTANCE: float      = 0.005
    SOFT_SPEED_THRESHOLD: float = 300.0
    LAMBDA_SPATIAL: float       = 0.001
    SIZE_RATIO_GATE: float      = 2.0

    # model.onnx
    # model_bnb4.onnx
    # model_fp16.onnx
    # model_int8.onnx
    # model_q4.onnx
    # model_q4f16.onnx    #ODD VERSION
    # model_quantized.onnx
    # model_uint8.onnx
    
    
    # --- MODEL IDENTIFIER & ARCHITECTURE ---
    MODEL_ID: str = "ultralytics/yolov8n.pt"
    # --- FEATURE EXTRACTOR MODELS ---
    FEATURE_EXTRACTOR_MODEL: str = 'models/osnet_x1_0_imagenet.onnx'
    # FEATURE_EXTRACTOR_MODEL: str = 'models/osnet_x1_0_imagenet_128x64.onnx'
    EMBEDDING_DIM: int = 512
    # KNN_N: int = 1000
    KNN_N: int = 200
    HF_INPUT_SIZE: int = 640
    # ALLOWED_ARCH = {"yolox", "ultralytics", "openvino", "onnx-community"}
    # This explicitly tells the system WHAT TYPE of model it is.
    # CRITICAL: This is used to route to the correct model handler and infer default filenames.
    MODEL_ARCH: ModelType = ModelType.ULTRALYTICS 
    
    # --- THRESHOLDS ---
    REID_THRESHOLD: float = 0.7
    DIVERSITY_THRESHOLD: float = 0.6
    CONF_THRESHOLD: float = 0.25  #0.5  this lets two overlapping boxes survive if they don’t cover exactly the same area.
    IOU_THRES: float = 0.6 
    NMS_THRESHOLD: float = 0.4  #0.5  YOLO’s NMS is too aggressive. The box that remains after NMS covers both people, so the second never gets its own ID.
    
    CLASS_AGNOSTIC: bool = True # YOLOX specific

    # --- PRIVACY ---
    PRIVACY_BLUR: bool = False
    PRIVACY_BLUR_KERNEL: int = 51

    # --- HEATMAP ---
    HEATMAP_ENABLED: bool = True
    HEATMAP_RETENTION_SECONDS: int = 3600
    HEATMAP_OPACITY: float = 0.25
    HEATMAP_RADIUS: int = 40
    HEATMAP_DECAY_RATE: float = 0.95

    # --- ANALYTICS RETENTION ---
    RAW_RETENTION_DAYS: int = 7
    AGGREGATE_RETENTION_DAYS: int = 30
    ANALYTICS_BATCH_SIZE: int = 100
    ANALYTICS_FLUSH_INTERVAL: float = 5.0

    # --- ZONES ---
    DEFAULT_ZONE_NAME: str = "Full Frame"

    # --- WEB ---
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # --- COMPUTED PROPERTIES ---
    # BASE_DIR is fundamental for resolving local paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DB_PATH: str = "data/db/surveillance.db"
    # Pydantic Settings configuration (essential for .env loading)
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_ignore_empty=True,
        extra="ignore" # Ignore unknown variables in .env without crashing
    )

# Instantiate settings once at startup
settings = Settings()