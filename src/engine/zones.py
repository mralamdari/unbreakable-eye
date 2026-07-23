"""
Zone management — configurable named regions per camera.

Zones are defined as polygons in normalized coordinates (0-1) and scaled
to inference resolution at runtime. Supports two types:
  - 'area': PolygonZone for occupancy/dwell tracking
  - 'line': LineZone for entry/exit counting
"""

import numpy as np
import supervision as sv
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Zone:
    """A named region on a camera's field of view."""
    zone_id: int
    camera_id: int
    name: str
    polygon: np.ndarray       # normalized (0-1) coordinates
    zone_type: str            # 'area' | 'line'
    color: str = "#4f8cff"

    # Runtime state (not persisted)
    _sv_zone: Optional[sv.PolygonZone] = field(default=None, repr=False)
    _line_zone: Optional[sv.LineZone] = field(default=None, repr=False)
    _dwell_start: dict = field(default_factory=dict, repr=False)
    _inside_now: set = field(default_factory=set, repr=False)

    def init_sv_zones(self, frame_width: int, frame_height: int) -> None:
        """Scale normalized polygon to pixel coordinates and create supervision zones."""
        pixel_poly = self.polygon.copy().astype(np.float32)
        pixel_poly[:, 0] *= frame_width
        pixel_poly[:, 1] *= frame_height
        pixel_poly = pixel_poly.astype(np.int32)

        if self.zone_type == "line" and len(pixel_poly) >= 2:
            # Line zone: first two points define the line, rest is the counting direction
            start = sv.Point(int(pixel_poly[0, 0]), int(pixel_poly[0, 1]))
            end = sv.Point(int(pixel_poly[1, 0]), int(pixel_poly[1, 1]))
            # Default counting direction: left-to-right
            if len(pixel_poly) >= 4:
                anchor = sv.Point(int(pixel_poly[2, 0]), int(pixel_poly[2, 1]))
            else:
                anchor = sv.Point(int(pixel_poly[1, 0]) + 100, int(pixel_poly[1, 1]))
            self._line_zone = sv.LineZone(start=start, end=end, anchor=anchor)  # type: ignore[call-arg]
        else:
            self._sv_zone = sv.PolygonZone(polygon=pixel_poly)

    def trigger(self, detections: sv.Detections) -> np.ndarray:
        """Return boolean mask of detections inside this zone."""
        if self._sv_zone is not None:
            return self._sv_zone.trigger(detections=detections)
        return np.zeros(len(detections), dtype=bool)

    def trigger_line(self, detections: sv.Detections) -> tuple:
        """For line zones: return (in_count, out_count, in_dets, out_dets)."""
        if self._line_zone is not None:
            return self._line_zone.trigger(detections=detections)
        return 0, 0, np.array([], dtype=int), np.array([], dtype=int)

    def track_dwell(self, tracker_ids: set, now: float) -> list:
        """Update dwell tracking. Returns list of (tracker_id, dwell_seconds) for exits."""
        exits = []
        entered = tracker_ids - self._inside_now
        left = self._inside_now - tracker_ids

        for tid in entered:
            self._dwell_start[tid] = now

        for tid in left:
            if tid in self._dwell_start:
                dwell = now - self._dwell_start.pop(tid)
                exits.append((tid, dwell))

        self._inside_now = tracker_ids
        return exits


class ZoneManager:
    """Manages zones for all cameras. Loads from DB, caches in memory."""

    def __init__(self):
        self._zones: dict[int, list[Zone]] = {}  # camera_id -> list of Zone
        self._initialized = False

    def load_from_db(self, conn, camera_id: int) -> list[Zone]:
        """Load zones for a camera from the database."""
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, camera_id, name, polygon, zone_type, color FROM zones WHERE camera_id = %s",
                (camera_id,),
            )
            rows = cur.fetchall()

        zones = []
        for r in rows:
            poly = np.array(r["polygon"], dtype=np.float32)
            zone = Zone(
                zone_id=r["id"],
                camera_id=r["camera_id"],
                name=r["name"],
                polygon=poly,
                zone_type=r["zone_type"],
                color=r["color"],
            )
            zones.append(zone)

        self._zones[camera_id] = zones
        return zones

    def init_zones_for_camera(self, camera_id: int, frame_width: int, frame_height: int) -> list[Zone]:
        """Initialize supervision zones with pixel coordinates for a camera."""
        zones = self._zones.get(camera_id, [])
        for zone in zones:
            zone.init_sv_zones(frame_width, frame_height)
        return zones

    def get_zones(self, camera_id: int) -> list[Zone]:
        return self._zones.get(camera_id, [])

    def remove_camera(self, camera_id: int):
        self._zones.pop(camera_id, None)


# Module-level singleton
zone_manager = ZoneManager()
