# Recorded validation — 2026-11-13

Local environment: Windows, CPython 3.11.4, Open3D 0.19.0. Exact versions of all
installed packages are in `requirements-lock-py311-windows.txt`.

- Ruff checks passed.
- 24 tests passed, including the real Open3D/HDBSCAN detector and the CLI demo.
- The default 12-frame synthetic demo completed; full results are checked into
  `demo-evaluation.json` and `demo-summary.json`.
- One local real PCD recording was processed successfully. The recording and its
  derived visualization are not distributed; no real-data accuracy follows from
  this smoke check. Available thermal boxes were not treated as 3D labels.
- The workflow builds and installs the wheel and checks the installed CLI on Linux
  Python 3.10–3.12. Its actual run status is available in the repository's Actions tab.

The shipped demo fixture uses a fixed procedural scene and seed 42. Noise-sampling
fixtures with seeds 7 and 19 are used by detector tests; the CLI test uses seed 101.
They do not cover the variety of real traffic. Synthetic thresholds and results
must not be represented as independent real-world validation.

Package import does not start ROS, UI servers or the prototype's background workers.
PointCloud2 decoding is tested using message-shaped inputs without a ROS runtime.
Live ROS graphs, sensor capture, calibrated tracking quality, Docker execution and
safety qualification remain unverified.
