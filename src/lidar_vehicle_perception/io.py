"""PCD/NPY/NPZ readers and line-delimited JSON output, without pickle loading."""

import json
from pathlib import Path

import numpy as np
import open3d as o3d

from .detector import finite_xyz


def read_points(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".pcd":
        points = np.asarray(o3d.io.read_point_cloud(str(path)).points)
        if len(points) == 0:
            raise ValueError(f"PCD has no readable points: {path.name}")
    elif path.suffix.lower() == ".npy":
        points = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            points = data["points"].copy()
    else:
        raise ValueError("supported inputs: .pcd, .npy, .npz")
    return finite_xyz(points)


def discover_frames(path):
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(p for p in path.rglob("*") if p.suffix.lower() in {".pcd", ".npy", ".npz"})
    if not files:
        raise ValueError("no .pcd, .npy or .npz frames found")
    return files


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False) + "\n")


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
