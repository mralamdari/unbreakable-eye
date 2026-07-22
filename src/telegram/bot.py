"""
Telegram bot for Unbreakable Eye shop monitoring.

Sends alerts and reports to Telegram via Cloudflare Worker relay.
No python-telegram-bot library needed — just simple HTTP requests.
"""

import asyncio
import time
from datetime import datetime
from loguru import logger
import requests

from src.telegram.config import telegram_config
from src.telegram.reports import (
    format_daily_report,
    format_weekly_report,
    format_status_message,
    format_zone_report,
    format_peak_hours,
)
from src.telegram.heatmap import generate_heatmap_image
from src.core.database import get_connection, write_connection


def _is_valid_telegram_url(url: str) -> bool:
    """Telegram requires publicly-accessible URLs for inline buttons."""
    if not url:
        return False
    lower = url.lower()
    if any(host in lower for host in ("localhost", "127.0.0.1", "0.0.0.0")):
        return False
    return lower.startswith("http://") or lower.startswith("https://")


def start_telegram_bot(ctx, alert_queue):
    """Spawn the telegram bot subprocess."""
    if not telegram_config.enabled:
        logger.info("Telegram bot disabled (TELEGRAM_ENABLED=false)")
        return None

    if not telegram_config.worker_url:
        logger.warning("CLOUDFLARE_WORKER_URL not set, Telegram bot disabled")
        return None

    p = ctx.Process(
        target=telegram_bot_worker,
        args=(alert_queue,),
        daemon=True,
        name="telegram_bot",
    )
    p.start()
    logger.info(f"Telegram bot process started | pid={p.pid}")
    return p


def send_to_telegram(message, photo_url=None, buttons=None):
    """Send a message to Telegram via Cloudflare Worker."""
    if not telegram_config.worker_url:
        logger.warning("No worker URL configured, cannot send to Telegram")
        return False

    payload = {
        "message": message,
        "token": telegram_config.worker_secret,
    }

    if photo_url:
        payload["photo"] = photo_url

    if buttons:
        payload["buttons"] = buttons

    try:
        resp = requests.post(
            telegram_config.worker_url,
            json=payload,
            timeout=10,
        )
        if resp.status_code == 200:
            result = resp.json()
            if result.get("ok"):
                return True
            else:
                logger.error(f"Telegram API error: {result}")
                return False
        else:
            logger.error(f"Worker returned {resp.status_code}: {resp.text}")
            return False
    except Exception as e:
        logger.error(f"Failed to send to Telegram: {e}")
        return False


def telegram_bot_worker(alert_queue):
    """Main loop for the Telegram bot — sends alerts and scheduled reports."""

    async def alert_listener():
        """Listen for alerts from the alert queue and send to Telegram."""
        while True:
            try:
                if not alert_queue.empty():
                    logger.info(f"Alert received from queue (size: {alert_queue.qsize()})")
                    alert = alert_queue.get_nowait()
                    severity = alert.get("severity", "INFO")
                    message = alert.get("message", "")
                    alert_type = alert.get("type", "info")
                    logger.info(f"Processing alert: {alert_type} - {severity}")

                    # Format alert message
                    icons = {"CRITICAL": "\u26a0\ufe0f", "WARNING": "\u26a0\ufe0f", "INFO": "\u2139\ufe0f"}
                    icon = icons.get(severity, "\u2139\ufe0f")
                    text = f"{icon} *{severity}*\n\n{message}"

                    # Add inline buttons based on alert type
                    buttons = _get_alert_buttons(alert_type)

                    logger.info(f"Sending alert to Telegram: {message[:50]}...")
                    success = send_to_telegram(text, buttons=buttons)
                    logger.info(f"Telegram send result: {'success' if success else 'failed'}")

                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Alert listener error: {e}", exc_info=True)
                await asyncio.sleep(5)

    _last_daily_date: str | None = None
    _last_weekly_date: str | None = None

    async def daily_report_scheduler():
        """Send daily report at configured time (once per day)."""
        nonlocal _last_daily_date
        while True:
            try:
                now = datetime.now()
                today_key = now.strftime("%Y-%m-%d")
                if (now.hour == telegram_config.daily_report_hour
                        and now.minute == telegram_config.daily_report_minute
                        and _last_daily_date != today_key):
                    _last_daily_date = today_key
                    summary = _get_summary_report()
                    text = format_daily_report(summary)
                    logger.info(f"Sending daily report for {today_key}")
                    send_to_telegram(text)
                    # Sleep a full minute to avoid double-fire on the same match
                    await asyncio.sleep(61)
            except Exception as e:
                logger.error(f"Daily report error: {e}")
            await asyncio.sleep(30)

    async def weekly_report_scheduler():
        """Send weekly report on configured day (once per week)."""
        nonlocal _last_weekly_date
        while True:
            try:
                now = datetime.now()
                week_key = now.strftime("%Y-%W")
                if (now.weekday() == telegram_config.weekly_report_day
                        and now.hour == telegram_config.daily_report_hour
                        and now.minute == telegram_config.daily_report_minute + 5
                        and _last_weekly_date != week_key):
                    _last_weekly_date = week_key
                    summary = _get_summary_report()
                    text = format_weekly_report(summary)
                    logger.info(f"Sending weekly report for week {week_key}")
                    send_to_telegram(text)
                    await asyncio.sleep(61)
            except Exception as e:
                logger.error(f"Weekly report error: {e}")
            await asyncio.sleep(60)

    async def run():
        # Send startup message
        send_to_telegram("*Unbreakable Eye Bot Started*\n\nMonitoring is active.")

        # Run listeners
        await asyncio.gather(
            alert_listener(),
            daily_report_scheduler(),
            weekly_report_scheduler(),
        )

    try:
        asyncio.run(run())
    except Exception as e:
        logger.error(f"Telegram bot crashed: {e}", exc_info=True)


def _get_alert_buttons(alert_type):
    """Get inline keyboard buttons based on alert type."""
    buttons = []

    if alert_type == "loitering":
        row = []
        if _is_valid_telegram_url(telegram_config.camera_url):
            row.append({"text": "View Camera", "url": telegram_config.camera_url})
        row.append({"text": "Dismiss", "callback_data": "dismiss"})
        buttons.append(row)
    elif alert_type in ("zone_entry", "zone_exit"):
        row = []
        if _is_valid_telegram_url(telegram_config.camera_url):
            row.append({"text": "View Camera", "url": telegram_config.camera_url})
        row.append({"text": "Dismiss", "callback_data": "dismiss"})
        buttons.append(row)
    elif alert_type == "occupancy":
        row = []
        if _is_valid_telegram_url(telegram_config.analytics_url):
            row.append({"text": "View Analytics", "url": telegram_config.analytics_url})
        row.append({"text": "Dismiss", "callback_data": "dismiss"})
        buttons.append(row)
    elif alert_type == "camera_offline":
        if _is_valid_telegram_url(telegram_config.camera_url):
            buttons.append([{"text": "View Camera", "url": telegram_config.camera_url}])
    else:
        row = []
        if _is_valid_telegram_url(telegram_config.dashboard_url):
            row.append({"text": "View Dashboard", "url": telegram_config.dashboard_url})
        row.append({"text": "Dismiss", "callback_data": "dismiss"})
        buttons.append(row)

    return buttons


# ─────────────────────────────────────────────────────────────────────────────
# Database helpers for reports
# ─────────────────────────────────────────────────────────────────────────────

def _get_summary_report() -> dict:
    """Get multi-camera summary report from database."""
    from src.core.database import get_multi_camera_summary

    with get_connection() as conn:
        return get_multi_camera_summary(conn)


def _get_zone_stats() -> dict:
    """Get zone stats for all cameras from database."""
    from src.core.database import get_all_cameras, get_zone_stats

    with get_connection() as conn:
        cameras = get_all_cameras(conn)
        result = {}
        for cam in cameras:
            zones = get_zone_stats(conn, cam["id"])
            if zones:
                result[cam["name"]] = zones
        return result


def _get_peak_hours() -> dict:
    """Get peak hours for all cameras from database."""
    from src.core.database import get_all_cameras, get_peak_hours

    with get_connection() as conn:
        cameras = get_all_cameras(conn)
        result = {}
        for cam in cameras:
            hours = get_peak_hours(conn, cam["id"])
            if hours:
                result[cam["name"]] = hours
        return result


def _get_heatmap_points(hours: int = 1) -> dict:
    """Get heatmap points for all cameras from database."""
    from src.core.database import get_all_cameras, get_heatmap_history

    with get_connection() as conn:
        cameras = get_all_cameras(conn)
        result = {}
        for cam in cameras:
            points = get_heatmap_history(conn, cam["id"], hours)
            if points:
                result[cam["name"]] = points
        return result
