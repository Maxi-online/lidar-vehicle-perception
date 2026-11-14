# LiDAR Vehicle Perception

Geometric conveyer for transportation and support of objects in clouds
Points. Based on its own system `detect_vehicles`; public version
separates processing, tracking, data entry and visualization.

![Synthetic scene result](docs/demo.png))

Realized ROI, land disposal RANSAC, HDBSCAN clustering in XY
Vertical Boxes, Dimensions, Filtering and Kalman/Hungarian-tracking. Got a launch
without ROS2 on PCD/NPY/NPZ, synthetic marking sequence, interactive
TML and optional ROS2-adapter.

## PowerShel launch

I do.Python 3.10–3.12; Local audit was performed onPython 3.11.

```powershell
git clone https://github.com/Maxi-online/lidar-vehicle-perception.git
cd lidar-vehicle-perception
py -3.11 -m venv .venv
& .venv/Scripts/python.exe -m pip install -e .
$env:OMP_NUM_THREADS='2'
$env:OPENBLAS_NUM_THREADS='1'
& .venv/Scripts/python.exe -m lidar_vehicle_perception demo --output artifacts/demo
```

Open `artifacts/demo/replay.html`: The Play button and the slider show the cloud,
Detectives and confirmed IDs. TML contains Plutly and does not require CDN.
JSONL Saves Boxes and Tracks separately. `evaluation.json` — Comparison with
`summary.json` - configuration, version of relationships and processing time.

Synthetic demo: 12 frames, 2 sites each. Saved startup is compared
All 24 boxes at AABB-BEV Iou ≥ 0.3. It's a test of the software conveyor on
Process cubes, **not an estimate of accuracy at real traffic flow**.
Classes are defined by position rules; trained neural classification
It's not part of this release. Dimensions and ROI need to be adjusted for installation.

Synthetic cloud treatment, I.D. preservation, short missing
Detectives, rejection of impossible associations, PointCloud2 formats and full launch
CLI. The live integration of ROS2, container and hardware has not been confirmed.

Details: [Main README](README.md), [architecture](docs/architecture.md),
[Assessment methodology](docs/evaluation.md), [ Origin and modification](docs/ provenance.md).

MIT © 2026 Maxi-online. Reference records and weights are not included in the repository.
