"""Optional ROS 2 adapter; ROS libraries are imported only by main()."""

import argparse
import json

from .config import Settings
from .detector import VehicleDetector
from .pointcloud2 import decode_pointcloud2
from .tracking import MultiObjectTracker


def main():
    parser = argparse.ArgumentParser(description="Subscribe to PointCloud2 and publish JSON tracks")
    parser.add_argument("--topic", default="/nanohomer_points")
    parser.add_argument("--output-topic", default="/vehicle_perception/tracks")
    parser.add_argument("--config")
    args, ros_args = parser.parse_known_args()
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import PointCloud2
        from std_msgs.msg import String
    except ImportError as exc:
        raise SystemExit(
            "ROS 2 environment required; source its setup.bash first. See docs/ros2.md"
        ) from exc
    settings = Settings.load(args.config)

    class PerceptionNode(Node):
        def __init__(self):
            super().__init__("lidar_vehicle_perception")
            self.detector = VehicleDetector(settings.detector)
            self.tracker = MultiObjectTracker(settings.tracker)
            self.previous_stamp = None
            self.publisher = self.create_publisher(String, args.output_topic, 10)
            qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.BEST_EFFORT
            )
            self.subscription = self.create_subscription(
                PointCloud2, args.topic, self.on_cloud, qos
            )

        def on_cloud(self, message):
            stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
            if self.previous_stamp is not None and stamp <= self.previous_stamp:
                self.get_logger().warning("Ignoring non-increasing PointCloud2 timestamp")
                return
            try:
                points = decode_pointcloud2(message)
                dt = stamp - self.previous_stamp if self.previous_stamp is not None else 0.1
                detections, stats = self.detector.detect(points)
                tracks = self.tracker.update(detections, dt)
                self.previous_stamp = stamp
                output = String()
                output.data = json.dumps(
                    {
                        "timestamp_s": stamp,
                        "frame_id": f"{message.header.stamp.sec}.{message.header.stamp.nanosec:09d}",
                        "coordinate_frame": message.header.frame_id,
                        "detections": [d.as_dict() for d in detections],
                        "tracks": [t.as_dict() for t in tracks],
                        "stats": stats,
                    },
                    allow_nan=False,
                )
                self.publisher.publish(output)
            except (ValueError, RuntimeError) as exc:
                self.get_logger().error(f"Point cloud rejected: {exc}")

    rclpy.init(args=ros_args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
