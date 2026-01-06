# import os
# from enum import Enum
# from pydantic_settings import BaseSettings, SettingsConfigDict

# # 1. Enums
# class ModelType(str, Enum):
#     YOLOX = "yolox"
#     DFINE = "dfine"
#     RTDETR = "rtdetr"
#     YOLOV8 = "yolov8"
#     OPENVINO = "openvino"

# class Device(str, Enum):
#     CPU = "cpu"
#     CUDA = "cuda"
#     GPU = "gpu"

# # 2. The Settings Class
# class Settings(BaseSettings):
#     # --- APP INFO (Matches PROJECT_NAME in .env) ---
#     PROJECT_NAME: str = "Unbreakable Eye"
    
#     # --- LOGGING (Matches LOG_LEVEL in .env) ---
#     LOG_LEVEL: str = "INFO"
#     LOG_JSON: bool = False

#     # --- INPUT (Matches RTSP_URL in .env) ---
#     # We allow str because it could be "0" (webcam) or "rtsp://..."
#     RTSP_URL: str = "0"

#     # --- MODEL CONFIG (Matches MODEL_ARCH, MODEL_PATH in .env) ---
#     MODEL_ARCH: ModelType = ModelType.YOLOX
#     MODEL_PATH: str = "models/yolov8n.pt"
#     # MODEL_PATH: str = "onnx-community/dfine_n_coco-ONNX"
    
#     # --- HARDWARE (Matches DEVICE in .env) ---
#     DEVICE: Device = Device.CPU

#     # --- THRESHOLDS (Matches CONF_THRESHOLD in .env) ---
#     CONF_THRESHOLD: float = 0.5
#     NMS_THRESHOLD: float = 0.4
#     CLASS_AGNOSTIC: bool = True

#     # --- SERVER ---
#     HOST: str = "0.0.0.0"
#     PORT: int = 8000

#     # --- COMPUTED PROPERTIES ---
#     BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#     # --- HUGGING FACE HUB SETTINGS ---
#     # Example: "ultralytics/yolov8-n-coco" for yolov8n.pt
#     HF_MODEL_REPO_ID: str = "" 
#     # Example: "yolov8n.pt" if HF_MODEL_REPO_ID is set
#     HF_MODEL_FILENAME: str = ""
#     # Example: "yolov8/n" if the model is in a subfolder in the repo
#     HF_MODEL_SUBFOLDER: str = ""
    
#     onnx-community/dfine_n_coco-ONNX
    
#     @property
#     def absolute_model_path(self) -> str:
#         # If HF Hub details are provided, return path where it *would* be downloaded
#         if self.HF_MODEL_REPO_ID and self.HF_MODEL_FILENAME:
#             return os.path.join(self.BASE_DIR, "models", self.HF_MODEL_SUBFOLDER, self.HF_MODEL_FILENAME)
        
#         # Otherwise, fall back to explicit MODEL_PATH
#         # If the path is already absolute, return it. Otherwise join with base.
#         if os.path.isabs(self.MODEL_PATH):
#             return self.MODEL_PATH
#         return os.path.join(self.BASE_DIR, self.MODEL_PATH)

#     # 3. Config Rules
#     model_config = SettingsConfigDict(
#         env_file=".env", 
#         env_ignore_empty=True,
#         extra="ignore"  # <--- THIS IS THE FIX. It ignores unknown variables instead of crashing.
#     )

# settings = Settings()



# # # .env file (to download yolov8n.pt)
# # MODEL_ARCH=yolov8
# # HF_MODEL_REPO_ID=ultralytics/yolov8n-pt
# # HF_MODEL_FILENAME=yolov8n.pt
# # # You can leave MODEL_PATH empty or point to a default, as HF will override
# # MODEL_PATH=
# # DEVICE=cpu
# # CONF_THRESHOLD=0.5



















# import os
# from enum import Enum
# from pydantic_settings import BaseSettings, SettingsConfigDict

# # 1. Enums
# class ModelType(str, Enum):
#     YOLOX = "yolox"
#     DFINE = "dfine"
#     RTDETR = "rtdetr"
#     YOLOV8 = "yolov8"
#     OPENVINO = "openvino"

# class Device(str, Enum):
#     CPU = "cpu"
#     CUDA = "cuda"
#     GPU = "gpu"

# # # 2. The Settings Class
# class Settings(BaseSettings):
#     # --- APP INFO (Matches PROJECT_NAME in .env) ---
#     PROJECT_NAME: str = "Unbreakable Eye"
    
#     # --- LOGGING (Matches LOG_LEVEL in .env) ---
#     LOG_LEVEL: str = "INFO"
#     LOG_JSON: bool = False

#     # --- INPUT (Matches RTSP_URL in .env) ---
#     # We allow str because it could be "0" (webcam) or "rtsp://..."
#     RTSP_URL: str = "0"

#     # --- MODEL CONFIG (Matches MODEL_ARCH, MODEL_PATH in .env) ---
#     # MODEL_ID: str = "ultralytics/yolov8n-pt" # Example of a HF repo_id
#     # MODEL_ARCH: ModelType = ModelType.YOLOV8 # Must specify arch for correct handler
    
#      # --- MODEL IDENTIFIER ---
#     # This string should be the Hugging Face repo_id for remote models,
#     # or a relative/absolute path for local models.
#     MODEL_ID: str = "ultralytics/yolov8n-pt" # Example of a HF repo_id
#     MODEL_ARCH: ModelType = ModelType.YOLOV8 # Must specify arch for correct handler
   
#     # --- MODEL FILENAME (if different from default inference for arch) ---
#     # Use this ONLY if the model filename in the repo is NOT the default for MODEL_ARCH.
#     # E.g., MODEL_ARCH=YOLOV8 but the file is not yolov8n.pt, it's custom_v8.pt
#     # Or if a HF repo has multiple models and you need a specific one.
#     MODEL_FILENAME_OVERRIDE: str = "" # e.g., "dfine_n_coco-ONNX"

    
    
    
#     # MODEL_ID: str = "models/yolov8n.pt" 
#     # MODEL_ARCH: ModelType = ModelType.YOLOV8 # Must specify arch for correct handler
#     # MODEL_PATH: str = "models/yolov8n.pt"
#     # MODEL_PATH: str = "onnx-community/dfine_n_coco-ONNX"
    
#     # --- COMPUTED PROPERTIES ---
#     BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#     # if not os.path.isabs(MODEL_PATH):
#     #     MODEL_PATH=os.path.join(BASE_DIR, MODEL_PATH)
    
#     # --- HARDWARE (Matches DEVICE in .env) ---
#     DEVICE: str = "CPU"

#     # --- THRESHOLDS (Matches CONF_THRESHOLD in .env) ---
#     CONF_THRESHOLD: float = 0.5
#     NMS_THRESHOLD: float = 0.4
#     CLASS_AGNOSTIC: bool = True

#     # --- SERVER ---
#     HOST: str = "0.0.0.0"
#     PORT: int = 8000

#     # --- COMPUTED PROPERTIES ---
#     # BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#     # @property
#     # def absolute_model_path(self) -> str:
#     #     # If HF Hub details are provided, return path where it *would* be downloaded
#     #     if self.HF_MODEL_REPO_ID and self.HF_MODEL_FILENAME:
#     #         return os.path.join(self.BASE_DIR, "models", self.HF_MODEL_SUBFOLDER, self.HF_MODEL_FILENAME)
        
#     #     # Otherwise, fall back to explicit MODEL_PATH
#     #     # If the path is already absolute, return it. Otherwise join with base.
#     #     if os.path.isabs(self.MODEL_PATH):
#     #         return self.MODEL_PATH
#     #     return os.path.join(self.BASE_DIR, self.MODEL_PATH)

#     # 3. Config Rules
#     model_config = SettingsConfigDict(
#         env_file=".env", 
#         env_ignore_empty=True,
#         extra="ignore"  # <--- THIS IS THE FIX. It ignores unknown variables instead of crashing.
#     )

# settings = Settings()



# # .env file (to download yolov8n.pt)
# MODEL_ARCH=yolov8
# HF_MODEL_REPO_ID=ultralytics/yolov8n-pt
# HF_MODEL_FILENAME=yolov8n.pt
# # You can leave MODEL_PATH empty or point to a default, as HF will override
# MODEL_PATH=
# DEVICE=cpu
# CONF_THRESHOLD=0.5





















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








## Scenario A: Local Model (e.g., you already have yolov8n.pt in models/yolov8/yolov8n.pt)
# MODEL_ID=yolov8n.pt  # Or full relative path like "yolov8/yolov8n.pt"
# MODEL_ARCH=yolov8
# RTSP_URL=0
# DEVICE=cpu
# CONF_THRESHOLD=0.5


##Scenario B: Hugging Face Model (e.g., ultralytics/yolov8n-pt for yolov8n.pt)
# MODEL_ID=ultralytics/yolov8n-pt # The HF repo ID
# MODEL_ARCH=yolov8
# RTSP_URL=0
# DEVICE=cpu
# CONF_THRESHOLD=0.5



##Scenario C: Hugging Face Model with Filename Override (if repo has many files)
# MODEL_ID=onnx-community/dfine_n_coco-ONNX
# MODEL_ARCH=dfine
# MODEL_FILENAME_OVERRIDE=dfine_n_coco.onnx # Must specify if not inferable
# RTSP_URL=0
# DEVICE=cpu
# CONF_THRESHOLD=0.5