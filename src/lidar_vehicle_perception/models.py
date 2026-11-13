from dataclasses import dataclass, replace
from itertools import product

import numpy as np


BOX_SIGNS = np.array(list(product((-1.0, 1.0), repeat=3)))
BOX_EDGES = tuple((i, j) for i in range(8) for j in range(i + 1, 8) if (i ^ j) in (1, 2, 4))


@dataclass(frozen=True)
class Detection:
    center: np.ndarray
    extent: np.ndarray
    rotation: np.ndarray
    class_name: str
    point_count: int
    track_id: int | None = None
    observed: bool = True
    missed_frames: int = 0

    @property
    def corners(self):
        return self.center + (BOX_SIGNS * self.extent / 2) @ self.rotation.T

    def moved(self, center, **kwargs):
        return replace(self, center=np.asarray(center, dtype=float).copy(), **kwargs)

    def as_dict(self):
        return {
            "center": self.center.tolist(),
            "extent": self.extent.tolist(),
            "rotation": self.rotation.tolist(),
            "corners": self.corners.tolist(),
            "class_name": self.class_name,
            "point_count": int(self.point_count),
            "track_id": self.track_id,
            "observed": self.observed,
            "missed_frames": self.missed_frames,
        }
