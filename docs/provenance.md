# Project provenance

This is the public modular release of Maxi-online's local `detect_vehicles`
prototype. The selected source was `detect_vehicles_engine.py` (453 lines), not
the separate experimental neural helper or the duplicate dated UI scripts.

Source SHA-256 before packaging:

```text
6307aed1a4cbe8b500b6c6df7c92ad9d0202d138970360b7aa0c7d73434840c8
```

The original pipeline supplied ROI filtering, RANSAC ground removal, XY HDBSCAN,
geometric dimension rules, shadow completion, Kalman/Hungarian tracking and a
Panel/ROS application. The public release retains those engineering ideas and
separates them into import-safe modules.

Release changes:

- explicit validated configuration instead of widget/global detector state;
- gravity-aligned minimum-area XY boxes rather than unconstrained 3D PCA OBB;
- HDBSCAN selection epsilon for fragmentation control;
- the declared density filter is now applied;
- shadow completion is an explicit experimental option, off by default;
- association gates distances before solving the assignment;
- coasting boxes follow the predicted center and expose observation state;
- PointCloud2 decoding respects layout instead of `reshape(-1, 3)`;
- standalone replay, procedural labels, an actual matching evaluator, tests and CI;
- source datasets, thermal images, dated copies and unrelated model downloads are
  excluded from the repository.

RANSAC, HDBSCAN, Kalman filtering, Hungarian assignment and minimum-area rectangles
are established algorithms implemented using third-party libraries. This project
does not claim to invent them, train a novel neural model or establish state-of-the-art
performance. MIT applies to this project's code; dependencies retain their own licenses.
