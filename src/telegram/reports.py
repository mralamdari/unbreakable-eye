"""Report generation for Telegram bot."""

from datetime import datetime


def _fmt_dwell(seconds):
    """Format dwell time as 'Xm Ys' or 'Xs'."""
    if not seconds:
        return "0s"
    s = float(seconds)
    if s > 60:
        return f"{int(s // 60)}m {int(s % 60)}s"
    return f"{s:.1f}s"


def _format_multi_camera_report(summary: dict, period: str = "Daily") -> str:
    """Format multi-camera breakdown report."""
    cameras = summary.get("cameras", [])
    totals = summary.get("totals", {})

    if not cameras:
        return f"*{period} Shop Report*\n\nNo cameras configured."

    t_today = totals.get("today", {})
    t_week = totals.get("week", {})
    t_active = totals.get("active_now", 0)

    lines = [
        f"*{period} Shop Report*",
        f"*{datetime.now().strftime('%B %d, %Y')}*",
        "",
        "*Shop Totals*",
        f"\u2022 Total visitors: *{t_today.get('visitors', 0)}*",
        f"\u2022 Total detections: *{t_today.get('detections', 0)}*",
        f"\u2022 Currently in store: *{t_active}*",
        f"\u2022 Avg dwell time: *{_fmt_dwell(t_week.get('avg_dwell', 0))}*",
        f"\u2022 Peak occupancy: *{t_week.get('max_occupancy', 0)}*",
        "",
    ]

    for cam in cameras:
        name = cam.get("name", "Unknown")
        cam_today = cam.get("today", {})
        cam_active = cam.get("active_now", 0)
        zones = cam.get("zones", [])

        lines.append(f"*{name}*")
        lines.append(
            f"\u2022 Visitors: *{cam_today.get('visitors', 0)}* "
            f"| Detections: *{cam_today.get('detections', 0)}* "
            f"| Active: *{cam_active}*"
        )

        if zones:
            zone_parts = []
            for z in zones[:5]:
                entries = z.get("entries", 0) or 0
                zone_parts.append(f"{z['name']} ({entries})")
            lines.append(f"\u2022 Zones: {', '.join(zone_parts)}")
        lines.append("")

    lines.append("_Powered by Unbreakable Eye_")
    return "\n".join(lines)


def format_daily_report(summary: dict) -> str:
    """Format daily summary report as Telegram message."""
    if "cameras" in summary:
        return _format_multi_camera_report(summary, "Daily")

    # Legacy single-camera fallback
    today = summary.get("today", {})
    week = summary.get("week", {})
    active = summary.get("active_now", 0)
    zones = summary.get("zones", [])
    peak_hours = summary.get("peak_hours", [])

    lines = [
        "*Daily Shop Report*",
        f"*{datetime.now().strftime('%A, %B %d, %Y')}*",
        "",
        "*Today's Stats*",
        f"\u2022 Visitors: *{today.get('visitors', 0)}*",
        f"\u2022 Detections: *{today.get('detections', 0)}*",
        f"\u2022 Currently in store: *{active}*",
        "",
    ]

    if week.get("visitors"):
        avg_dwell = week.get("avg_dwell", 0)
        dwell_str = _fmt_dwell(avg_dwell)
        lines.extend([
            "*This Week*",
            f"\u2022 Total visitors: *{week.get('visitors', 0)}*",
            f"\u2022 Avg dwell time: *{dwell_str}*",
            f"\u2022 Peak occupancy: *{week.get('max_occupancy', 0)}*",
            "",
        ])

    if zones:
        lines.append("*Zone Performance*")
        for z in zones[:5]:
            entries = z.get("entries", 0) or 0
            avg_d = z.get("avg_dwell", 0) or 0
            lines.append(f"\u2022 {z['name']}: {entries} entries, {avg_d:.0f}s avg")
        lines.append("")

    if peak_hours:
        lines.append("*Peak Hours Today*")
        for ph in peak_hours[:3]:
            hour = int(ph.get("hour", 0))
            visitors = ph.get("visitors", 0)
            lines.append(f"\u2022 {hour:02d}:00 - *{visitors}* visitors")
        lines.append("")

    lines.append("_Powered by Unbreakable Eye_")

    return "\n".join(lines)


def format_weekly_report(summary: dict, comparison: dict = None) -> str:
    """Format weekly summary report as Telegram message."""
    if "cameras" in summary:
        return _format_multi_camera_report(summary, "Weekly")

    # Legacy single-camera fallback
    week = summary.get("week", {})
    zones = summary.get("zones", [])

    lines = [
        "*Weekly Shop Report*",
        f"*Week of {datetime.now().strftime('%B %d, %Y')}*",
        "",
        "*Weekly Stats*",
        f"\u2022 Total visitors: *{week.get('visitors', 0)}*",
        f"\u2022 Total detections: *{week.get('detections', 0)}*",
    ]

    avg_dwell = week.get("avg_dwell", 0)
    dwell_str = _fmt_dwell(avg_dwell)
    lines.append(f"\u2022 Avg dwell time: *{dwell_str}*")
    lines.append(f"\u2022 Peak occupancy: *{week.get('max_occupancy', 0)}*")
    lines.append("")

    if comparison and "changes" in comparison:
        ch = comparison["changes"]
        lines.append("*vs Last Week*")
        for key, label in [("visitors_pct", "Visitors"), ("avg_dwell_pct", "Dwell time")]:
            val = ch.get(key, 0)
            arrow = "\u25b2" if val > 0 else "\u25bc" if val < 0 else "--"
            lines.append(f"\u2022 {label}: {arrow} {abs(val)}%")
        lines.append("")

    if zones:
        lines.append("*Top Zones*")
        sorted_zones = sorted(zones, key=lambda z: z.get("entries", 0) or 0, reverse=True)
        for z in sorted_zones[:3]:
            entries = z.get("entries", 0) or 0
            lines.append(f"\u2022 {z['name']}: *{entries}* entries")
        lines.append("")

    lines.append("_Powered by Unbreakable Eye_")

    return "\n".join(lines)


def format_status_message(summary: dict) -> str:
    """Format current status as Telegram message."""
    if "cameras" in summary:
        cameras = summary.get("cameras", [])
        totals = summary.get("totals", {})
        t_active = totals.get("active_now", 0)
        t_today = totals.get("today", {})

        lines = [
            "*Current Status*",
            f"\u2022 People in store: *{t_active}*",
            f"\u2022 Today's visitors: *{t_today.get('visitors', 0)}*",
            f"\u2022 Today's detections: *{t_today.get('detections', 0)}*",
            "",
        ]
        for cam in cameras:
            cam_active = cam.get("active_now", 0)
            cam_today = cam.get("today", {})
            lines.append(
                f"\u2022 *{cam.get('name', 'Unknown')}*: "
                f"{cam_active} active, "
                f"{cam_today.get('visitors', 0)} visitors"
            )
        return "\n".join(lines)

    active = summary.get("active_now", 0)
    today = summary.get("today", {})

    lines = [
        "*Current Status*",
        f"\u2022 People in store: *{active}*",
        f"\u2022 Today's visitors: *{today.get('visitors', 0)}*",
        f"\u2022 Today's detections: *{today.get('detections', 0)}*",
    ]

    return "\n".join(lines)


def format_zone_report(zones: dict) -> str:
    """Format zone performance as Telegram message.

    Args:
        zones: dict of {camera_name: [zone_dicts]} for multi-camera,
               or list of zone_dicts for legacy single-camera.
    """
    if isinstance(zones, dict):
        if not zones:
            return "*Zone Performance*\n\nNo zones configured yet."
        lines = ["*Zone Performance*", ""]
        for cam_name, cam_zones in zones.items():
            lines.append(f"*{cam_name}*")
            for z in cam_zones:
                name = z.get("name", "Unnamed")
                entries = z.get("entries", 0) or 0
                exits = z.get("exits", 0) or 0
                avg_dwell = z.get("avg_dwell", 0) or 0
                unique = z.get("unique_visitors", 0) or 0
                dwell_str = _fmt_dwell(avg_dwell)
                lines.extend([
                    f"\u2022 *{name}*: {entries} in, {exits} out, "
                    f"{dwell_str} avg, {unique} unique",
                ])
            lines.append("")
        return "\n".join(lines)

    # Legacy single-camera fallback
    if not zones:
        return "*Zone Performance*\n\nNo zones configured yet."

    lines = ["*Zone Performance*", ""]

    for z in zones:
        name = z.get("name", "Unnamed")
        entries = z.get("entries", 0) or 0
        exits = z.get("exits", 0) or 0
        avg_dwell = z.get("avg_dwell", 0) or 0
        unique = z.get("unique_visitors", 0) or 0

        dwell_str = _fmt_dwell(avg_dwell)

        lines.extend([
            f"*{name}*",
            f"\u2022 Entries: {entries} | Exits: {exits}",
            f"\u2022 Avg dwell: {dwell_str} | Unique: {unique}",
            "",
        ])

    return "\n".join(lines)


def format_peak_hours(peak_data) -> str:
    """Format peak hours as Telegram message.

    Args:
        peak_data: dict of {camera_name: [hour_dicts]} for multi-camera,
                   or list of hour_dicts for legacy single-camera.
    """
    if isinstance(peak_data, dict):
        if not peak_data:
            return "*Peak Hours*\n\nNo data yet."
        lines = ["*Peak Hours*", ""]
        for cam_name, hours in peak_data.items():
            sorted_hours = sorted(hours, key=lambda x: x.get("total", 0), reverse=True)[:3]
            if sorted_hours:
                lines.append(f"*{cam_name}*")
                for ph in sorted_hours:
                    dow = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][int(ph.get("dow", 0))]
                    hour = int(ph.get("hour", 0))
                    total = ph.get("total", 0)
                    lines.append(f"\u2022 {dow} {hour:02d}:00 - *{total}* visitors")
                lines.append("")
        return "\n".join(lines)

    # Legacy single-camera fallback
    if not peak_data:
        return "*Peak Hours*\n\nNo data yet."

    lines = ["*Peak Hours*", ""]

    sorted_hours = sorted(peak_data, key=lambda x: x.get("total", 0), reverse=True)[:5]

    for ph in sorted_hours:
        dow = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][int(ph.get("dow", 0))]
        hour = int(ph.get("hour", 0))
        total = ph.get("total", 0)
        lines.append(f"\u2022 {dow} {hour:02d}:00 - *{total}* visitors")

    return "\n".join(lines)
