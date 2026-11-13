"""Explicit settings replacing UI globals in the original desktop prototype."""

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class DetectorConfig:
    roi: tuple = ((-7.0, 9.0), (-24.0, 0.0), (-4.0, 5.0))
    ground_distance: float = 0.15
    ground_iterations: int = 300
    ground_normal_z_min: float = 0.7
    min_cluster_size: int = 30
    min_samples: int = 3
    cluster_selection_epsilon: float = 0.5
    max_bottom_height: float = 0.50
    object_layer_height: float = 3.5
    dimensions_min: tuple = (1.5, 0.7, 0.8)
    dimensions_max: tuple = (8.0, 4.0, 3.5)
    min_density: float = 5.0
    dimension_tolerance: float = 0.1
    car_upper: tuple = (4.0, 1.7, 1.4)
    minivan_upper: tuple = (4.8, 2.1, 1.7)
    truck_upper: tuple = (8.0, 2.5, 2.3)
    reconstruct_shadow: bool = False
    shadow_step: float = 0.30
    max_shadow: float = 5.0
    random_seed: int = 42

    def __post_init__(self):
        if len(self.roi) != 3 or any(len(r) != 2 for r in self.roi):
            raise ValueError("roi must contain three [minimum, maximum] pairs")
        if any(not all(math.isfinite(v) for v in r) or r[0] >= r[1] for r in self.roi):
            raise ValueError("roi bounds must be finite and increasing")
        for name in (
            "dimensions_min",
            "dimensions_max",
            "car_upper",
            "minivan_upper",
            "truck_upper",
        ):
            value = getattr(self, name)
            if len(value) != 3 or any(not math.isfinite(v) or v <= 0 for v in value):
                raise ValueError(f"{name} must contain three positive finite dimensions")
        if any(lo >= hi for lo, hi in zip(self.dimensions_min, self.dimensions_max)):
            raise ValueError("minimum dimensions must be smaller than maximum dimensions")
        if self.min_cluster_size < 3 or self.min_samples < 1 or self.ground_iterations < 1:
            raise ValueError("cluster sizes and iteration count must be positive")
        for name in ("ground_distance", "object_layer_height", "shadow_step", "max_shadow"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0 < self.ground_normal_z_min <= 1:
            raise ValueError("ground_normal_z_min must be in (0, 1]")
        for name in (
            "max_bottom_height",
            "min_density",
            "dimension_tolerance",
            "cluster_selection_epsilon",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class TrackerConfig:
    distance_threshold: float = 3.5
    max_age: int = 8
    min_hits: int = 3

    def __post_init__(self):
        if not math.isfinite(self.distance_threshold) or self.distance_threshold <= 0:
            raise ValueError("distance_threshold must be positive and finite")
        if self.max_age < 0 or self.min_hits < 1:
            raise ValueError("max_age must be nonnegative and min_hits at least one")


@dataclass(frozen=True)
class Settings:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)

    def as_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path | None):
        if path is None:
            return cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(data) - {"detector", "tracker"}
        if unknown:
            raise ValueError(f"unknown settings sections: {sorted(unknown)}")
        return cls(
            DetectorConfig(**data.get("detector", {})), TrackerConfig(**data.get("tracker", {}))
        )
