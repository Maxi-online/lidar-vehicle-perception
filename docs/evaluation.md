# Evaluation protocol and limits

`lidar-perception evaluate` joins predictions and ground truth by unique `frame_id`.
Missing prediction frames count as false negatives; additional prediction frames
count as false positives. Duplicate frame IDs and invalid box geometry are errors.

Each box needs `center` (XYZ), `extent` (three positive local-axis lengths) and a
right-handed orthonormal `rotation` (3x3). `track_id` is needed only for identity
matching. `corners` in exported records is convenient visualization data; the
evaluator recomputes corners from the box geometry.

Matching uses **axis-aligned BEV IoU of projected boxes**, default threshold 0.3,
with gating before Hungarian assignment. This is an intentionally simple baseline;
it is not oriented 3D IoU, AP/mAP, HOTA or a complete MOT evaluation. AABB overlap
can overstate overlap for rotated boxes.

Reported fields:

- TP / FP / FN, precision, recall and F1 for detector proposals before track confirmation;
- matched-box L/W/H MAE in metres, with H along world Z;
- observed confirmed-track matches and identity changes for matched ground-truth IDs;
- detector+tracker p50/p95 time, including the first frame, excluding disk IO,
  process imports and rendering. `summary.json` records dependency versions,
  platform, nominal replay rate and OpenMP setting.

The generated demo contains a flat road and two moving cuboid vehicles. Exact
generating boxes are provided as labels. Surfaces are sampled with noise; this is
a software/geometry fixture, not a physical LiDAR simulation. It has no realistic
beam pattern, material response, thermal calibration, pedestrians or traffic
occlusion. Configuration development used synthetic fixtures; their results do
not establish generalization or real-world classification accuracy.

For a real assessment, collect independently annotated sequences across distance,
orientation, density, weather, object type and occlusion. Split by recording/site
before adjusting thresholds. Freeze configuration, compare against a simple
Euclidean clustering baseline, report uncertainty and failure slices. Real-data
3D/BEV labels must use the same coordinate frame and units. Thermal image boxes
alone cannot be treated as 3D ground truth without calibration and association.

Known failure cases: sloped/unlevel sensor frames, vegetation or street furniture
with vehicle-like dimensions, nearby vehicle clusters merging, sparse/partial
observations, missed ground, severe occlusion and long tracking gaps. The system
is an engineering prototype, not a safety-qualified perception stack.
