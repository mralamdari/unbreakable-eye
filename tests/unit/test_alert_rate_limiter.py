"""
Tests for the AlertRateLimiter (src/engine/alerts.py).

Pure logic — no models, cameras, or GPU needed.
"""
import time
from src.engine.alerts import AlertRateLimiter


class TestAlertRateLimiter:
    """AlertRateLimiter prevents alert spam by enforcing a cooldown."""

    def test_default_cooldown(self):
        limiter = AlertRateLimiter()
        assert limiter.cooldown == 300  # 5 minutes default

    def test_custom_cooldown(self):
        limiter = AlertRateLimiter(cooldown_seconds=10)
        assert limiter.cooldown == 10

    def test_first_alert_always_passes(self):
        limiter = AlertRateLimiter(cooldown_seconds=300)
        assert limiter.should_alert("loitering", 1) is True

    def test_alert_rate_limited_within_cooldown(self):
        limiter = AlertRateLimiter(cooldown_seconds=10)
        # First call passes
        assert limiter.should_alert("loitering", 1) is True
        # Second call immediately after should be blocked
        assert limiter.should_alert("loitering", 1) is False

    def test_alert_passes_after_cooldown(self):
        limiter = AlertRateLimiter(cooldown_seconds=0.05)
        assert limiter.should_alert("loitering", 1) is True
        assert limiter.should_alert("loitering", 1) is False
        time.sleep(0.06)
        assert limiter.should_alert("loitering", 1) is True

    def test_different_types_have_independent_cooldowns(self):
        limiter = AlertRateLimiter(cooldown_seconds=10)
        assert limiter.should_alert("loitering", 1) is True
        assert limiter.should_alert("camera_offline", 1) is True  # different type

    def test_different_cameras_have_independent_cooldowns(self):
        limiter = AlertRateLimiter(cooldown_seconds=10)
        assert limiter.should_alert("loitering", 1) is True
        assert limiter.should_alert("loitering", 2) is True  # different camera

    def test_same_type_different_cameras_after_cooldown(self):
        limiter = AlertRateLimiter(cooldown_seconds=0.05)
        assert limiter.should_alert("loitering", 1) is True
        assert limiter.should_alert("loitering", 1) is False
        time.sleep(0.06)
        assert limiter.should_alert("loitering", 1) is True

    def test_cameras_cooldown_independently(self):
        limiter = AlertRateLimiter(cooldown_seconds=10)
        assert limiter.should_alert("loitering", 1) is True
        assert limiter.should_alert("loitering", 2) is True
        assert limiter.should_alert("loitering", 3) is True
        # Cam 1 should be rate limited
        assert limiter.should_alert("loitering", 1) is False
        # Cam 2 and 3 should also be rate limited
        assert limiter.should_alert("loitering", 2) is False
        assert limiter.should_alert("loitering", 3) is False
