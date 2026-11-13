"""Auditable baseline: axis-aligned BEV IoU, Hungarian matches and dimension MAE.

Projected AABBs intentionally make no oriented-3D-IoU claim. Predicted-only
coasting tracks are excluded; detector proposals are scored before track confirmation.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detector import dimensions_lwh
from .models import BOX_SIGNS


def box_corners(box):
    center = np.asarray(box["center"], dtype=float)
    extent = np.asarray(box["extent"], dtype=float)
    rotation = np.asarray(box["rotation"], dtype=float)
    if center.shape != (3,) or extent.shape != (3,) or rotation.shape != (3, 3):
        raise ValueError("box needs center[3], extent[3] and rotation[3,3]")
    if not all(np.isfinite(v).all() for v in (center, extent, rotation)) or np.any(extent <= 0):
        raise ValueError("box geometry must be finite with positive extents")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1, atol=1e-5
    ):
        raise ValueError("box rotation must be orthonormal and right-handed")
    return center + (BOX_SIGNS * extent / 2) @ rotation.T


def bev_aabb_iou(first, second):
    a, b = box_corners(first)[:, :2], box_corners(second)[:, :2]
    lo = np.maximum(a.min(axis=0), b.min(axis=0))
    hi = np.minimum(a.max(axis=0), b.max(axis=0))
    intersection = float(np.prod(np.maximum(hi - lo, 0)))
    union = float(
        np.prod(a.max(axis=0) - a.min(axis=0))
        + np.prod(b.max(axis=0) - b.min(axis=0))
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def match_boxes(predictions, truth, threshold):
    if not predictions or not truth:
        return []
    overlaps = np.array([[bev_aabb_iou(p, t) for t in truth] for p in predictions])
    valid = overlaps >= threshold
    penalty = min(overlaps.shape) + 1.0
    rows, cols = linear_sum_assignment(np.where(valid, 1 - overlaps, penalty))
    return [(int(r), int(c)) for r, c in zip(rows, cols) if valid[r, c]]


def _by_frame(records):
    result = {}
    for record in records:
        frame_id = record["frame_id"]
        if not isinstance(frame_id, str) or not frame_id:
            raise ValueError("frame_id must be a nonempty string")
        if frame_id in result:
            raise ValueError(f"duplicate frame_id: {frame_id}")
        for box in record.get("detections", []) + record.get("tracks", []):
            box_corners(box)
        result[frame_id] = record
    return result


def evaluate(predictions, ground_truth, iou_threshold=0.3):
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    pred, truth = _by_frame(predictions), _by_frame(ground_truth)
    tp = fp = fn = switches = tracked_matches = 0
    errors, identities = [], {}
    for frame in sorted(set(pred) | set(truth)):
        boxes = pred.get(frame, {}).get("detections", [])
        targets = truth.get(frame, {}).get("detections", [])
        matches = match_boxes(boxes, targets, iou_threshold)
        tp += len(matches)
        fp += len(boxes) - len(matches)
        fn += len(targets) - len(matches)
        for pi, ti in matches:
            box, target = boxes[pi], targets[ti]
            measured = dimensions_lwh(box["rotation"], box["extent"])
            expected = dimensions_lwh(target["rotation"], target["extent"])
            errors.append(np.abs(measured - expected))
        tracks = [t for t in pred.get(frame, {}).get("tracks", []) if t.get("observed", True)]
        for pi, ti in match_boxes(tracks, targets, iou_threshold):
            target_id, track_id = targets[ti].get("track_id"), tracks[pi].get("track_id")
            if target_id is None or track_id is None:
                continue
            if target_id in identities and identities[target_id] != track_id:
                switches += 1
            identities[target_id] = track_id
            tracked_matches += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "metric": "axis-aligned BEV IoU",
        "iou_threshold": iou_threshold,
        "frames": len(set(pred) | set(truth)),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "dimension_mae_m_lwh": np.mean(errors, axis=0).tolist() if errors else None,
        "observed_track_matches": tracked_matches,
        "identity_switches_on_matched_tracks": switches,
        "limitations": "Synthetic demo or caller-supplied labels only. AABB-BEV is not oriented 3D IoU; "
        "ID-switch count is not a complete MOT benchmark. Class labels are not scored.",
    }
