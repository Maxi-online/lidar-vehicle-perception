"""ROI -> RANSAC ground -> XY HDBSCAN -> geometric vehicle proposals.

The optional shadow completion is a heuristic inherited from the prototype.
It is disabled by default because unseen object geometry is not observed evidence.
"""

from threading import Lock

import hdbscan
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull, QhullError, cKDTree

from .config import DetectorConfig
from .models import Detection

_RANSAC_LOCK = Lock()  # Open3D's seed is process-global.


def finite_xyz(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("point cloud must have shape (N, 3), with XYZ in metres")
    return points[np.isfinite(points).all(axis=1)]


def dimensions_lwh(rotation, extent):
    """Map OBB axes to L/W/H using the axis most aligned with world Z."""
    height_index = int(np.argmax(np.abs(np.asarray(rotation)[2])))
    horizontal = [float(extent[i]) for i in range(3) if i != height_index]
    return np.array([max(horizontal), min(horizontal), float(extent[height_index])])


def classify_dimensions(dimensions, config):
    for label, upper in (
        ("car", config.car_upper),
        ("minivan", config.minivan_upper),
        ("truck", config.truck_upper),
    ):
        if np.all(dimensions <= np.asarray(upper) + config.dimension_tolerance):
            return label
    return "unknown"


def upright_obb(points):
    """Minimum-area XY hull rectangle with world-Z height; stable vehicle dimensions.

    An unconstrained 3D PCA box can tilt height into length on sparsely sampled
    vehicle sides. Here roll/pitch are constrained by the declared Z-up frame.
    """
    hull = ConvexHull(points[:, :2])
    contour = points[hull.vertices, :2]
    edges = np.roll(contour, -1, axis=0) - contour
    angles = np.unique(np.round(np.mod(np.arctan2(edges[:, 1], edges[:, 0]), np.pi / 2), 10))
    best = None
    for angle in angles:
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        local = contour @ rotation
        lower, upper = local.min(axis=0), local.max(axis=0)
        area = float(np.prod(upper - lower))
        if best is None or area < best[0]:
            best = (area, rotation, lower, upper)
    _, rotation_xy, lower, upper = best
    center_xy = ((lower + upper) / 2) @ rotation_xy.T
    extent = np.r_[upper - lower, np.ptp(points[:, 2])]
    center = np.r_[center_xy, (points[:, 2].min() + points[:, 2].max()) / 2]
    rotation = np.eye(3)
    rotation[:2, :2] = rotation_xy
    return center, extent, rotation


def shadow_completion(points, lidar_xy, occlusion_tree, ground_z, config):
    """Extend visible contour rays on the ground; not a learned reconstruction."""
    try:
        hull = ConvexHull(points[:, :2])
    except QhullError:
        return np.empty((0, 3))
    shadow = []
    for point in points[hull.vertices]:
        direction = point[:2] - lidar_xy
        length = np.linalg.norm(direction)
        if length < 1e-6:
            continue
        direction /= length
        for step in np.arange(config.shadow_step, config.max_shadow + 1e-9, config.shadow_step):
            xy = point[:2] + direction * step
            if occlusion_tree is not None and occlusion_tree.query_ball_point(xy, 0.35):
                break
            shadow.append([xy[0], xy[1], float(ground_z(xy[0], xy[1]))])
    return np.asarray(shadow).reshape(-1, 3)


class VehicleDetector:
    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()

    def detect(self, points: np.ndarray) -> tuple[list[Detection], dict[str, int | bool]]:
        config = self.config
        clean = finite_xyz(points)
        roi = np.asarray(config.roi)
        points_roi = clean[((clean >= roi[:, 0]) & (clean <= roi[:, 1])).all(axis=1)]
        stats = {
            "input_points": len(points),
            "finite_points": len(clean),
            "roi_points": len(points_roi),
            "object_points": 0,
            "ground_found": False,
        }
        if len(points_roi) < max(50, config.min_cluster_size):
            return [], stats
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_roi))
        ground = None
        # Reject dominant walls before accepting a sufficiently horizontal ground plane.
        with _RANSAC_LOCK:
            o3d.utility.random.seed(config.random_seed)
            for _ in range(2):
                if len(cloud.points) < 3:
                    break
                try:
                    plane, inliers = cloud.segment_plane(
                        distance_threshold=config.ground_distance,
                        ransac_n=3,
                        num_iterations=config.ground_iterations,
                    )
                except RuntimeError:
                    break
                if not inliers:
                    break
                if abs(plane[2]) >= config.ground_normal_z_min:
                    ground = plane
                    cloud = cloud.select_by_index(inliers, invert=True)
                    break
                cloud = cloud.select_by_index(inliers, invert=True)
        if ground is None:
            return [], stats
        stats["ground_found"] = True
        a, b, c, d = ground

        def ground_z(x, y):
            return (-a * x - b * y - d) / c

        objects = np.asarray(cloud.points)
        height = objects[:, 2] - ground_z(objects[:, 0], objects[:, 1])
        objects = objects[(height > 0) & (height < config.object_layer_height)]
        stats["object_points"] = len(objects)
        if len(objects) < config.min_cluster_size:
            return [], stats
        labels = hdbscan.HDBSCAN(
            min_cluster_size=config.min_cluster_size,
            min_samples=config.min_samples,
            allow_single_cluster=True,
            metric="euclidean",
            cluster_selection_epsilon=float(config.cluster_selection_epsilon),
        ).fit_predict(objects[:, :2])
        detections = []
        for label in sorted(set(labels) - {-1}):
            cluster = objects[labels == label]
            if len(cluster) < config.min_cluster_size:
                continue
            xy = cluster[:, :2].mean(axis=0)
            if cluster[:, 2].min() - ground_z(*xy) > config.max_bottom_height:
                continue
            augmented = cluster
            if config.reconstruct_shadow:
                others = objects[(labels != label) & (labels != -1)]
                tree = cKDTree(others[:, :2]) if len(others) else None
                shadow = shadow_completion(cluster, np.zeros(2), tree, ground_z, config)
                if len(shadow):
                    augmented = np.vstack([cluster, shadow])
            try:
                center, extent, rotation = upright_obb(augmented)
            except QhullError:
                continue
            dimensions = dimensions_lwh(rotation, extent)
            if np.any(dimensions < config.dimensions_min) or np.any(
                dimensions > config.dimensions_max
            ):
                continue
            if len(cluster) / max(float(np.prod(extent)), 1e-9) < config.min_density:
                continue
            detections.append(
                Detection(
                    center=center,
                    extent=extent,
                    rotation=rotation,
                    class_name=classify_dimensions(dimensions, config),
                    point_count=len(cluster),
                )
            )
        return detections, stats
