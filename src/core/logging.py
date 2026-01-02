import sys
import logging
from loguru import logger
from src.core.config import settings
from types import FrameType

class InterceptHandler(logging.Handler):
    """
    Redirects standard logging messages (like uvicorn/fastapi) to Loguru
    so we have ONE unified log stream.
    """
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            # FIX: Convert levelno to its string name for consistency
            level = record.levelname # Use the string name, not the number
        
        # ... (rest of the code for frame handling) ...

        _current_frame: FrameType | None = logging.currentframe()
        depth = 2

        if _current_frame is not None:
            _iter_frame: FrameType = _current_frame
            while _iter_frame.f_code.co_filename == logging.__file__:
                _iter_frame = _iter_frame.f_back # type: ignore[assignment]
                depth += 1
                if _iter_frame is None:
                    break

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, # This 'level' variable must be a string
            record.getMessage()
        )
        
def setup_logging():
    """
    Configures the logging system. Call this once at startup.
    """
    # 1. Remove default handlers
    logging.getLogger().handlers = [InterceptHandler()]
    logger.remove()

    # 2. Determine Format (JSON for Cloud, Text for Dev)
    if settings.LOG_JSON:
        # JSON format for DataDog/CloudWatch/ELK
        log_format = "{message}"
        serialize = True
    else:
        # Beautiful colored text for local dev
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
        serialize = False

    # 3. Add Console Sink (Stdout)
    logger.add(
        sys.stderr,
        level=settings.LOG_LEVEL,
        format=log_format,
        serialize=serialize,
        backtrace=True,
        diagnose=True,
    )

    # 4. [...](asc_slot://start-slot-5)Add File Sink (Optional: Rotating File)
    # Automatically rotates file every 10 MB, keeps for 1 week
    logger.add(
        "logs/app.log",
        rotation="10 MB",
        retention="1 week",
        level="WARNING",  # Only save warnings/errors to disk
        compression="zip"
    )

    # 5. Hijack Uvicorn & FastAPI logs
    logging.getLogger("uvicorn.access").handlers = [InterceptHandler()]
    logging.getLogger("uvicorn.error").handlers = [InterceptHandler()]
    logging.getLogger("fastapi").handlers = [InterceptHandler()]
