# Optional ROS 2 adapter

The offline package does not install or import `rclpy` by default. Use a sourced
ROS 2 environment with `rclpy`, `sensor_msgs` and `std_msgs` and a Python version
supported by both that ROS distribution and this package (3.10–3.12).

Example for an existing Ubuntu/ROS 2 Humble installation:

```bash
source /opt/ros/humble/setup.bash
python3 -m venv --system-site-packages .venv-ros
. .venv-ros/bin/activate
python -m pip install .
python -m lidar_vehicle_perception.ros2 --topic /nanohomer_points --output-topic /vehicle_perception/tracks --config configs/default.json
```

`/vehicle_perception/tracks` carries `std_msgs/String` JSON with timestamp,
unique timestamp-derived `frame_id`, source `coordinate_frame`, detections,
tracks and point-count diagnostics. Binary XYZ decoding is unit tested for
organized XYZI, row padding and both byte orders. Live subscription/publication,
sensor QoS compatibility and hardware latency have not been tested in this release.

The input must already be in metres and Z-up. No TF/ego-motion transform is applied.
Depth-one best-effort input intentionally allows frame loss under load. The adapter
does not publish RViz markers or standard autonomous-driving object messages.
