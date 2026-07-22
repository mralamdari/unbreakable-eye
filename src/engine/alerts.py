"""
Alert generation for the pipeline.

Detects loitering, camera offline, and other events, then emits alerts
to the alert_queue for the Telegram bot to consume.
"""

import time
import queue
from loguru import logger


class AlertRateLimiter:
    """Rate-limits alerts to prevent spam — max 1 alert per (type, camera) per cooldown."""

    def __init__(self, cooldown_seconds=300):
        self.cooldown = cooldown_seconds
        self._last_alerts = {}  # (type, cam_id) -> timestamp

    def should_alert(self, alert_type, cam_id):
        key = (alert_type, cam_id)
        now = time.time()
        last = self._last_alerts.get(key, 0)
        if now - last >= self.cooldown:
            self._last_alerts[key] = now
            return True
        logger.debug(f"Alert rate-limited: {alert_type} for camera {cam_id}")
        return False


def _emit(alert_queue, alert_type, severity, message, cam_id, extra=None):
    """Helper to put an alert on the queue."""
    alert = {
        "type": alert_type,
        "severity": severity,
        "message": message,
        "camera_id": cam_id,
        "timestamp": time.time(),
    }
    if extra:
        alert.update(extra)
    try:
        alert_queue.put_nowait(alert)
        logger.debug(f"{alert_type} alert added to queue (size: {alert_queue.qsize()})")
    except queue.Full:
        logger.warning(f"Alert queue full — dropping {alert_type} alert")


def emit_loitering_alert(alert_queue, cam_id, tracker_id, customer_id,
                         duration_seconds, limiter):
    """Emit loitering alert if rate limit allows."""
    if not limiter.should_alert("loitering", cam_id):
        return

    msg = (f"Person loitering for {duration_seconds:.0f}s in CAM {cam_id}"
           f" (tracker #{tracker_id})")
    logger.info(f"Emitting loitering alert: {msg}")
    _emit(alert_queue, "loitering", "WARNING", msg, cam_id,
          {"tracker_id": tracker_id, "customer_id": customer_id})


def emit_camera_offline_alert(alert_queue, cam_id, offline_seconds, limiter):
    """Emit camera offline alert if rate limit allows."""
    if not limiter.should_alert("camera_offline", cam_id):
        return

    msg = f"CAM {cam_id} has been offline for {offline_seconds:.0f}s"
    logger.info(f"Emitting camera offline alert: {msg}")
    _emit(alert_queue, "camera_offline", "CRITICAL", msg, cam_id)


def emit_zone_entry_alert(alert_queue, cam_id, zone_name, customer_id, limiter):
    """Emit zone entry alert when a person enters a monitored zone."""
    if not limiter.should_alert("zone_entry", cam_id):
        return

    msg = f"Person entered zone '{zone_name}' on CAM {cam_id}"
    logger.info(f"Emitting zone entry alert: {msg}")
    _emit(alert_queue, "zone_entry", "INFO", msg, cam_id,
          {"zone_name": zone_name, "customer_id": customer_id})


def emit_zone_exit_alert(alert_queue, cam_id, zone_name, dwell_seconds, customer_id, limiter):
    """Emit zone exit alert with dwell time when a person leaves a zone."""
    if not limiter.should_alert("zone_exit", cam_id):
        return

    dwell_str = f"{int(dwell_seconds // 60)}m {int(dwell_seconds % 60)}s" if dwell_seconds > 60 else f"{dwell_seconds:.0f}s"
    msg = f"Person left zone '{zone_name}' on CAM {cam_id} (dwell: {dwell_str})"
    logger.info(f"Emitting zone exit alert: {msg}")
    _emit(alert_queue, "zone_exit", "INFO", msg, cam_id,
          {"zone_name": zone_name, "dwell_seconds": dwell_seconds, "customer_id": customer_id})


def emit_occupancy_alert(alert_queue, cam_id, current_count, limit, limiter):
    """Emit occupancy limit alert when too many people are in the store."""
    if not limiter.should_alert("occupancy", cam_id):
        return

    msg = f"CAM {cam_id}: {current_count} people in store (limit: {limit})"
    logger.info(f"Emitting occupancy alert: {msg}")
    _emit(alert_queue, "occupancy", "WARNING", msg, cam_id,
          {"current_count": current_count, "limit": limit})


def emit_inactivity_alert(alert_queue, cam_id, inactive_minutes, limiter):
    """Emit inactivity alert when no detections for a period of time."""
    if not limiter.should_alert("inactivity", cam_id):
        return

    msg = f"CAM {cam_id}: No activity for {inactive_minutes} minutes"
    logger.info(f"Emitting inactivity alert: {msg}")
    _emit(alert_queue, "inactivity", "WARNING", msg, cam_id,
          {"inactive_minutes": inactive_minutes})
