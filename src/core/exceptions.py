"""
Custom exceptions for vision pipeline with proper error hierarchy.
Enables specific exception handling and better error diagnostics.
"""

from loguru import logger


class VisionError(RuntimeError):
    """Base exception for all vision pipeline errors."""

    def __init__(self, message: str, context: dict | None = None):
        """
        Args:
            message: Error description
            context: Optional dict with additional context (model_path, cam_id, etc.)
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}
        
        if self.context:
            logger.error(f"{self.__class__.__name__}: {message} | Context: {self.context}")
        else:
            logger.error(f"{self.__class__.__name__}: {message}")


class ModelLoadError(VisionError):
    """Raised when a model file cannot be loaded, is missing, or is corrupted."""
    pass


class ModelResolutionError(VisionError):
    """Raised when a model path cannot be resolved from MODEL_ID/MODEL_ARCH settings."""
    pass


class ModelDownloadError(VisionError):
    """Raised when a model download (GitHub, HuggingFace, Ultralytics) fails."""
    pass


class ModelConfigError(VisionError):
    """Raised when model configuration is invalid or incomplete."""
    pass


class PreprocessError(VisionError):
    """Raised when image preprocessing fails."""
    pass


class InferenceError(VisionError):
    """Raised when model inference fails."""
    pass


class PostprocessError(VisionError):
    """Raised when output postprocessing fails."""
    pass


class StreamError(VisionError):
    """Raised when RTSP stream cannot be opened or read."""
    pass


class SharedMemoryError(VisionError):
    """Raised when shared memory allocation or access fails."""
    pass


class DatabaseError(VisionError):
    """Raised when database operations fail."""
    pass


class ReIDError(VisionError):
    """Raised when Re-ID matching produces invalid results."""
    pass