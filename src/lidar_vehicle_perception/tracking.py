"""Constant-velocity Kalman tracks with distance-gated Hungarian association."""

from dataclasses import dataclass

from filterpy.kalman import KalmanFilter
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .config import TrackerConfig
from .models import Detection


def gated_assignment(costs, threshold):
    """Gate *before* assignment so an impossible pair cannot consume a valid match."""
    costs = np.asarray(costs, dtype=float)
    if costs.ndim != 2 or not np.isfinite(costs).all() or np.any(costs < 0):
        raise ValueError("costs must be a finite nonnegative matrix")
    if costs.size == 0:
        return []
    valid = costs < threshold
    penalty = (min(costs.shape) + 1) * max(threshold, 1.0)
    rows, columns = linear_sum_assignment(np.where(valid, costs, penalty))
    return [(int(r), int(c)) for r, c in zip(rows, columns) if valid[r, c]]


@dataclass
class _Track:
    filter: KalmanFilter
    detection: Detection
    hits: int = 1
    missed: int = 0


class MultiObjectTracker:
    def __init__(self, config: TrackerConfig | None = None):
        self.config = config or TrackerConfig()
        self._tracks = {}
        self._next_id = 0

    @staticmethod
    def _filter(center):
        kalman = KalmanFilter(dim_x=6, dim_z=3)
        kalman.x = np.r_[center, np.zeros(3)]
        kalman.F = np.eye(6)
        kalman.H = np.c_[np.eye(3), np.zeros((3, 3))]
        kalman.P *= 10.0
        kalman.R = np.eye(3) * 0.5
        kalman.Q = np.eye(6) * 0.1
        return kalman

    def update(self, detections: list[Detection], dt: float) -> list[Detection]:
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        ids = list(self._tracks)
        transition = np.eye(6)
        transition[:3, 3:] = np.eye(3) * dt
        for track in self._tracks.values():
            track.filter.F = transition
            track.filter.predict()
            track.missed += 1
        if ids and detections:
            costs = cdist(
                [self._tracks[i].filter.x[:3] for i in ids], [d.center for d in detections]
            )
            pairs = gated_assignment(costs, self.config.distance_threshold)
        else:
            pairs = []
        matched_detections = set()
        for row, column in pairs:
            track = self._tracks[ids[row]]
            track.filter.update(detections[column].center)
            track.detection = detections[column]
            track.hits += 1
            track.missed = 0
            matched_detections.add(column)
        for index, detection in enumerate(detections):
            if index not in matched_detections:
                self._tracks[self._next_id] = _Track(self._filter(detection.center), detection)
                self._next_id += 1
        self._tracks = {i: t for i, t in self._tracks.items() if t.missed <= self.config.max_age}
        return [
            t.detection.moved(
                t.filter.x[:3], track_id=i, observed=t.missed == 0, missed_frames=t.missed
            )
            for i, t in self._tracks.items()
            if t.hits >= self.config.min_hits
        ]
