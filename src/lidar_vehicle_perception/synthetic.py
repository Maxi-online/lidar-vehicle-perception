"""Procedural two-vehicle sequence with exact generating boxes; no external data."""

from pathlib import Path

import numpy as np

from .io import write_jsonl
from .models import Detection


def cuboid_surface(rng, center, extent, yaw, count=850):
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    local = rng.uniform(-0.5, 0.5, (count, 3)) * extent
    faces = rng.integers(0, 6, count)
    for face in range(6):
        mask = faces == face
        local[mask, face // 2] = extent[face // 2] / 2 * (1 if face % 2 else -1)
    points = local @ rotation.T + center + rng.normal(0, 0.008, local.shape)
    return points, rotation


def generate_sequence(output, frames=12, seed=42):
    if frames < 1 or frames > 1000:
        raise ValueError("frames must be between 1 and 1000")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Avoid silently mixing earlier, longer sequences with the requested demo.
    if any(output.iterdir()):
        raise ValueError("synthetic output directory must be empty")
    rng = np.random.default_rng(seed)
    records = []
    for index in range(frames):
        ground = np.column_stack(
            [rng.uniform(-7, 9, 2600), rng.uniform(-24, 0, 2600), rng.normal(0, 0.006, 2600)]
        )
        objects, boxes = [], []
        fraction = index / max(frames - 1, 1)
        for track_id, center, extent, yaw, label in [
            (
                0,
                np.array([-3.3 + fraction * 1.1, -7.0, 0.70]),
                np.array([3.7, 1.55, 1.30]),
                0.10,
                "car",
            ),
            (
                1,
                np.array([3.2 - fraction * 0.6, -16.0 + fraction, 1.15]),
                np.array([5.5, 2.15, 2.20]),
                -0.08,
                "truck",
            ),
        ]:
            points, rotation = cuboid_surface(rng, center, extent, yaw)
            objects.append(points)
            boxes.append(
                Detection(center, extent, rotation, label, len(points), track_id).as_dict()
            )
        cloud = np.vstack([ground, *objects]).astype(np.float32)
        frame_id = f"frame_{index:04d}.npy"
        np.save(output / frame_id, cloud, allow_pickle=False)
        records.append({"frame_id": frame_id, "timestamp_s": index * 0.1, "detections": boxes})
    write_jsonl(output / "ground_truth.jsonl", records)
    return records
