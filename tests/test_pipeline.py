from dataclasses import replace

import numpy as np
import pytest

from lidar_vehicle_perception.config import DetectorConfig, Settings
from lidar_vehicle_perception.detector import VehicleDetector, dimensions_lwh, finite_xyz
from lidar_vehicle_perception.io import read_points
from lidar_vehicle_perception.metrics import evaluate
from lidar_vehicle_perception.synthetic import generate_sequence


def test_nonfinite_points_removed_and_bad_shape_rejected():
    clean = finite_xyz([[1, 2, 3], [float("nan"), 0, 0], [0, float("inf"), 0]])
    np.testing.assert_array_equal(clean, [[1, 2, 3]])
    with pytest.raises(ValueError, match="shape"):
        finite_xyz(np.zeros((3, 4)))


def test_empty_cloud_and_outside_roi_produce_no_proposals():
    detector = VehicleDetector()
    assert detector.detect(np.empty((0, 3)))[0] == []
    assert detector.detect(np.ones((100, 3)) * 100)[0] == []


def test_vertical_wall_is_not_accepted_as_ground():
    y, z = np.meshgrid(np.linspace(-20, -2, 25), np.linspace(0.2, 3.0, 25))
    wall = np.column_stack([np.zeros(y.size), y.ravel(), z.ravel()])
    detections, stats = VehicleDetector().detect(wall)
    assert detections == []
    assert stats["ground_found"] is False


def test_world_vertical_axis_is_used_for_height():
    rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    np.testing.assert_allclose(dimensions_lwh(rotation, [2, 1.5, 4]), [4, 2, 1.5])


def test_zero_shadow_step_is_rejected():
    with pytest.raises(ValueError):
        replace(DetectorConfig(), shadow_step=0)


def test_unknown_config_section_is_not_silently_ignored(tmp_path):
    config = tmp_path / "config.json"
    config.write_text('{"detecotr": {}}', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        Settings.load(config)


@pytest.mark.parametrize("seed", [7, 19])
def test_actual_detector_on_procedural_clouds(tmp_path, seed):
    labels = generate_sequence(tmp_path / "inputs", frames=1, seed=seed)
    detector = VehicleDetector()
    points = read_points(tmp_path / "inputs/frame_0000.npy")
    detections, stats = detector.detect(points)
    assert stats["ground_found"]
    assert len(detections) == 2
    metrics = evaluate(
        [{"frame_id": "frame_0000.npy", "detections": [d.as_dict() for d in detections]}], labels
    )
    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0


def test_npz_reader_and_no_pickle(tmp_path):
    path = tmp_path / "points.npz"
    np.savez(path, points=np.array([[1, 2, 3]]))
    np.testing.assert_array_equal(read_points(path), [[1, 2, 3]])
    bad = tmp_path / "object.npy"
    np.save(bad, np.array([{"x": 1}], dtype=object))
    with pytest.raises(ValueError):
        read_points(bad)
