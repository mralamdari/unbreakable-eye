"""
Tests for Zone dwell tracking logic (src/engine/zones.py).

Zone.track_dwell uses pure set operations — no models, cameras, or GPU needed.

Note: Zone.__init__ requires supervision (sv) which needs the full package,
but we can test in isolation by creating a Zone manually with minimal setup.
"""
import numpy as np
from src.engine.zones import Zone, ZoneManager


def _make_zone(zone_id=1, camera_id=1, name="test"):
    """Helper: create a Zone with a tiny default polygon."""
    poly = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]], dtype=np.float32)
    return Zone(
        zone_id=zone_id,
        camera_id=camera_id,
        name=name,
        polygon=poly,
        zone_type="area",
        color="#ff0000",
    )


class TestZoneInitialization:
    def test_zone_default_state(self):
        zone = _make_zone()
        assert zone.zone_id == 1
        assert zone.camera_id == 1
        assert zone.name == "test"
        assert zone._inside_now == set()
        assert zone._dwell_start == {}

    def test_zone_polygon_preserved(self):
        poly = np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]], dtype=np.float32)
        zone = Zone(
            zone_id=1, camera_id=1, name="full",
            polygon=poly, zone_type="area",
        )
        np.testing.assert_array_equal(zone.polygon, poly)


class TestZoneDwellTracking:
    """track_dwell should correctly track who enters and leaves."""

    def test_no_trackers_empty_sets(self):
        zone = _make_zone()
        now = 1000.0
        exits = zone.track_dwell(set(), now)
        assert exits == []
        assert zone._inside_now == set()

    def test_first_person_enters(self):
        zone = _make_zone()
        now = 1000.0
        exits = zone.track_dwell({1}, now)
        assert exits == []
        assert zone._inside_now == {1}
        assert zone._dwell_start[1] == 1000.0

    def test_multiple_people_enter(self):
        zone = _make_zone()
        exits = zone.track_dwell({1, 2, 3}, 1000.0)
        assert exits == []
        assert zone._inside_now == {1, 2, 3}
        assert len(zone._dwell_start) == 3

    def test_person_leaves_records_dwell(self):
        zone = _make_zone()
        zone.track_dwell({1}, 1000.0)
        # Person 1 leaves after 5 seconds
        exits = zone.track_dwell(set(), 1005.0)
        assert len(exits) == 1
        tid, dwell = exits[0]
        assert tid == 1
        assert dwell == 5.0  # 1005 - 1000

    def test_dwell_time_is_accurate(self):
        zone = _make_zone()
        zone.track_dwell({1}, 100.0)
        zone.track_dwell({1}, 105.0)  # still inside
        exits = zone.track_dwell(set(), 112.0)  # leaves
        assert len(exits) == 1
        _, dwell = exits[0]
        assert dwell == 12.0  # 112 - 100

    def test_multiple_exits_recorded_in_sequence(self):
        """Track dwelling for multiple people across sequential calls."""
        zone = _make_zone()
        exits = zone.track_dwell({1, 2, 3}, 1000.0)  # all enter
        assert exits == []

        exits = zone.track_dwell({1}, 1010.0)  # 2 and 3 leave
        assert len(exits) == 2
        # 2 dwelled 10s, 3 dwelled 10s
        assert set(tid for tid, _ in exits) == {2, 3}

        exits = zone.track_dwell(set(), 1020.0)  # 1 leaves
        assert len(exits) == 1
        assert exits[0][0] == 1
        assert exits[0][1] == 20.0


    def test_re_entry_restarts_dwell(self):
        """A person who leaves and re-enters should restart their dwell timer."""
        zone = _make_zone()
        zone.track_dwell({1}, 1000.0)  # enters
        zone.track_dwell(set(), 1010.0)  # leaves (dwell=10s)
        zone.track_dwell({1}, 1020.0)  # re-enters
        assert zone._dwell_start[1] == 1020.0  # timer reset

    def test_stay_inside_no_exit(self):
        """If the same set of trackers is inside, no exits."""
        zone = _make_zone()
        zone.track_dwell({1, 2}, 1000.0)
        exits = zone.track_dwell({1, 2}, 1005.0)
        assert exits == []
        assert zone._dwell_start[1] == 1000.0
        assert zone._dwell_start[2] == 1000.0


class TestZoneManager:
    """ZoneManager should manage zones per camera."""

    def test_empty_initial_state(self):
        zm = ZoneManager()
        assert zm.get_zones(1) == []
        assert zm._initialized is False

    def test_remove_camera_clears_zones(self):
        zm = ZoneManager()
        zone = _make_zone()
        zm._zones[1] = [zone]
        assert len(zm.get_zones(1)) == 1
        zm.remove_camera(1)
        assert zm.get_zones(1) == []

    def test_remove_nonexistent_camera_does_not_crash(self):
        zm = ZoneManager()
        zm.remove_camera(99)  # should not raise
