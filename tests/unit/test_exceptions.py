"""
Tests for the custom exception hierarchy (src/core/exceptions.py).

All exceptions are pure Python — no models, cameras, or GPU needed.
"""
import pytest
from src.core.exceptions import (
    VisionError,
    ModelLoadError,
    ModelResolutionError,
    ModelDownloadError,
    ModelConfigError,
    PreprocessError,
    InferenceError,
    PostprocessError,
    StreamError,
    SharedMemoryError,
    DatabaseError,
    ReIDError,
)


class TestExceptionHierarchy:
    """All custom exceptions should inherit from VisionError."""

    def test_vision_error_is_base(self):
        err = VisionError("test error")
        assert isinstance(err, RuntimeError)
        assert err.message == "test error"
        assert err.context == {}

    def test_vision_error_with_context(self):
        err = VisionError("failed", context={"cam_id": 1})
        assert err.context == {"cam_id": 1}

    def test_all_exceptions_inherit_vision_error(self):
        exceptions = [
            ModelLoadError(""),
            ModelResolutionError(""),
            ModelDownloadError(""),
            ModelConfigError(""),
            PreprocessError(""),
            InferenceError(""),
            PostprocessError(""),
            StreamError(""),
            SharedMemoryError(""),
            DatabaseError(""),
            ReIDError(""),
        ]
        for exc in exceptions:
            assert isinstance(exc, VisionError), f"{type(exc).__name__} is not a VisionError"
            assert isinstance(exc, RuntimeError)

    def test_exception_chain(self):
        """Test that exceptions can be raised and caught at the base level."""
        with pytest.raises(VisionError):
            raise ModelLoadError("model not found")

    def test_str_representation(self):
        err = ModelConfigError("invalid config", context={"key": "MODEL_ARCH"})
        assert "invalid config" in str(err)


class TestSpecificExceptions:
    """Check each exception carries correct type info."""

    @pytest.mark.parametrize("exc_cls,expected_name", [
        (ModelLoadError, "ModelLoadError"),
        (ModelResolutionError, "ModelResolutionError"),
        (ModelDownloadError, "ModelDownloadError"),
        (ModelConfigError, "ModelConfigError"),
        (PreprocessError, "PreprocessError"),
        (InferenceError, "InferenceError"),
        (PostprocessError, "PostprocessError"),
        (StreamError, "StreamError"),
        (SharedMemoryError, "SharedMemoryError"),
        (DatabaseError, "DatabaseError"),
        (ReIDError, "ReIDError"),
    ])
    def test_exception_name(self, exc_cls, expected_name):
        err = exc_cls("something went wrong")
        assert type(err).__name__ == expected_name
