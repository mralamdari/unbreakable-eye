from src.core.database import read_connection


def get_dwell_times(cam_id: int, since: float) -> list[dict]:
    """Return average dwell time per zone for a camera since timestamp."""
    with read_connection() as conn:
        rows = conn.execute("""
            SELECT zone_id, AVG(dwell_seconds) as avg_dwell,
                   COUNT(DISTINCT customer_id) as unique_visitors
            FROM zone_events
            WHERE camera_id = ? AND timestamp > ?
            GROUP BY zone_id
        """, (cam_id, since)).fetchall()
    return [dict(r) for r in rows]


def get_repeat_visitors(cam_id: int, days: int = 7) -> list[dict]:
    """Customers seen more than once in the last N days."""
    with read_connection() as conn:
        rows = conn.execute("""
            SELECT customer_id, COUNT(*) as visit_count
            FROM zone_events
            WHERE camera_id = ? AND timestamp > ?
            GROUP BY customer_id
            HAVING COUNT(*) > 1
        """, (cam_id, days * 86400)).fetchall()
    return [dict(r) for r in rows]


def get_peak_hours(cam_id: int) -> list[dict]:
    """Hourly visitor counts to find busy periods."""
    with read_connection() as conn:
        rows = conn.execute("""
            SELECT EXTRACT(HOUR FROM timestamp) as hour,
                   COUNT(*) as visitor_count
            FROM zone_events
            WHERE camera_id = ?
            GROUP BY hour
            ORDER BY hour
        """, (cam_id,)).fetchall()
    return [dict(r) for r in rows]


__all__ = ["get_dwell_times", "get_repeat_visitors", "get_peak_hours"]