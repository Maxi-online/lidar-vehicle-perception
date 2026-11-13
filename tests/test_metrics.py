from copy import deepcopy

import numpy as np
import pytest

from lidar_vehicle_perception.metrics import evaluate
from lidar_vehicle_perception.models import Detection


def target(x=0, identity=10):
    return Detection(
        np.array([x, 0, 1]), np.array([4, 2, 2]), np.eye(3), "car", 100, identity
    ).as_dict()


def test_missing_and_extra_frames_affect_recall_and_precision():
    predictions = [
        {"frame_id": "0", "detections": [target(), target(20)]},
        {"frame_id": "2", "detections": [target()]},
    ]
    truth = [
        {"frame_id": "0", "detections": [target()]},
        {"frame_id": "1", "detections": [target()]},
    ]
    result = evaluate(predictions, truth, iou_threshold=1)
    assert (result["true_positives"], result["false_positives"], result["false_negatives"]) == (
        1,
        2,
        1,
    )
    assert result["precision"] == pytest.approx(1 / 3)
    assert result["recall"] == 0.5


def test_observed_identity_switch_count_and_dimension_error():
    truth = [{"frame_id": str(i), "detections": [target()]} for i in range(2)]
    predictions = deepcopy(truth)
    for i, record in enumerate(predictions):
        record["tracks"] = [target(identity=i)]
    result = evaluate(predictions, truth)
    assert result["identity_switches_on_matched_tracks"] == 1
    assert result["dimension_mae_m_lwh"] == [0, 0, 0]


def test_duplicate_frame_ids_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        evaluate([{"frame_id": "0"}, {"frame_id": "0"}], [])


def test_invalid_ground_truth_geometry_rejected():
    box = target()
    box["extent"] = [4, 0, 2]
    with pytest.raises(ValueError, match="positive extents"):
        evaluate([], [{"frame_id": "0", "detections": [box]}])
