"""Telegram bot configuration."""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))


@dataclass
class TelegramConfig:
    """Telegram bot settings loaded from environment variables."""

    # Bot credentials
    bot_token: str = ""
    chat_id: str = ""

    # Enable/disable
    enabled: bool = False

    # Cloudflare Worker relay
    worker_url: str = ""
    worker_secret: str = ""

    # URLs for inline buttons
    dashboard_url: str = "http://localhost"
    camera_url: str = "http://localhost/camera/1"
    analytics_url: str = "http://localhost/analysis"

    # Report schedule
    daily_report_hour: int = 9
    daily_report_minute: int = 0
    weekly_report_day: int = 1  # Monday

    # Alert thresholds
    alert_occupancy_limit: int = 50
    alert_loiter_seconds: int = 30
    alert_inactivity_minutes: int = 30
    alert_camera_offline_seconds: int = 30

    # Heatmap settings
    heatmap_width: int = 640
    heatmap_height: int = 480

    @classmethod
    def from_env(cls) -> "TelegramConfig":
        return cls(
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
            enabled=os.getenv("TELEGRAM_ENABLED", "false").lower() == "true",
            worker_url=os.getenv("CLOUDFLARE_WORKER_URL", ""),
            worker_secret=os.getenv("WORKER_SECRET", ""),
            dashboard_url=os.getenv("DASHBOARD_URL", "http://localhost"),
            camera_url=os.getenv("CAMERA_URL", "http://localhost/camera/1"),
            analytics_url=os.getenv("ANALYTICS_URL", "http://localhost/analysis"),
            daily_report_hour=int(os.getenv("TELEGRAM_DAILY_REPORT_HOUR", "9")),
            daily_report_minute=int(os.getenv("TELEGRAM_DAILY_REPORT_MINUTE", "0")),
            weekly_report_day=int(os.getenv("TELEGRAM_WEEKLY_REPORT_DAY", "1")),
            alert_occupancy_limit=int(os.getenv("TELEGRAM_ALERT_OCCUPANCY_LIMIT", "50")),
            alert_loiter_seconds=int(os.getenv("TELEGRAM_ALERT_LOITER_SECONDS", "30")),
            alert_inactivity_minutes=int(os.getenv("TELEGRAM_ALERT_INACTIVITY_MINUTES", "30")),
            alert_camera_offline_seconds=int(os.getenv("TELEGRAM_ALERT_CAMERA_OFFLINE_SECONDS", "30")),
            heatmap_width=int(os.getenv("TELEGRAM_HEATMAP_WIDTH", "640")),
            heatmap_height=int(os.getenv("TELEGRAM_HEATMAP_HEIGHT", "480")),
        )


telegram_config = TelegramConfig.from_env()
