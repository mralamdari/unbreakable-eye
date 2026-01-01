from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from src.core.config import settings
from loguru import logger
from src.core.logging import setup_logging
from src.engine.pipeline import VisionPipeline
import threading
import uvicorn
import cv2

# 1. Setup Logging
setup_logging()

# 2. Global State
pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Log Startup
    logger.info("🚀 Unbreakable Eye is Waking Up...")
    
    global pipeline
    try:
        pipeline = VisionPipeline()
        logger.success("✅ Vision Pipeline initialized successfully")
    except Exception as e:
        logger.critical(f"🔥 FATAL ERROR: Pipeline failed to start! {e}")
        raise e
        
    yield
    
    # Log Shutdown
    logger.warning("🛑 System Shutting Down...")
    if pipeline.cap.isOpened():
        pipeline.cap.release()

app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

@app.get("/")
def health_check():
    return {
        "status": "online",
        "model": settings.MODEL_ARCH,
        "device": settings.DEVICE
    }

@app.get("/stream/video")
def video_feed():
    """
    Streams the annotated video to the browser.
    Usage: <img src="http://localhost:8000/stream/video" />
    """
    if not pipeline:
        return Response("System initializing...", status_code=503)

    return StreamingResponse(
        pipeline.generate_frames(), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    # Local Development Entrypoint
    uvicorn.run("src.server.main:app", host="0.0.0.0", port=8000, reload=True)
