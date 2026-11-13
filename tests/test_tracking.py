import numpy as np
import pytest

from lidar_vehicle_perception.config import TrackerConfig
from lidar_vehicle_perception.models import Detection
from lidar_vehicle_perception.tracking import MultiObjectTracker, gated_assignment


def box(x):
    return Detection(np.array([x, -5.0, 0.7]), np.array([3.7, 1.5, 1.3]), np.eye(3), "car", 100)


def test_impossible_pair_cannot_steal_two_valid_matches():
    # Ungated Hungarian picks (0,0)+(1,1), then drops the second pair.
    assert gated_assignment([[0.1, 2.5], [2.6, 3.6]], 3.0) == [(0, 1), (1, 0)]


def test_ids_survive_detection_order_change_and_brief_occlusion():
    tracker = MultiObjectTracker(TrackerConfig(min_hits=2, max_age=2, distance_threshold=2))
    assert tracker.update([box(-3), box(3)], 0.1) == []
    observed = tracker.update([box(3.1), box(-2.9)], 0.1)
    assert [(t.track_id, t.observed) for t in observed] == [(0, True), (1, True)]
    coast = tracker.update([], 0.1)
    assert all(not t.observed and t.missed_frames == 1 for t in coast)
    restored = tracker.update([box(-2.7), box(3.3)], 0.1)
    assert [t.track_id for t in restored] == [0, 1]
    assert all(t.observed for t in restored)
    tracker.update([], 0.1)
    tracker.update([], 0.1)
    assert tracker.update([], 0.1) == []


@pytest.mark.parametrize("dt", [0, -1, float("nan"), float("inf")])
def test_invalid_time_step_rejected(dt):
    with pytest.raises(ValueError):
        MultiObjectTracker().update([], dt)
