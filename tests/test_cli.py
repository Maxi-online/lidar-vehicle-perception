import json
import os
import subprocess
import sys


def test_demo_command_produces_readable_artifacts(tmp_path):
    output = tmp_path / "demo"
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "lidar_vehicle_perception",
            "demo",
            "--output",
            str(output),
            "--frames",
            "3",
            "--seed",
            "101",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    evaluation = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    assert summary["frames"] == 3
    assert evaluation["dataset"]["real_world_validation"] is False
    assert len((output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    assert (output / "preview.png").stat().st_size > 1000
    assert "plotly" in (output / "replay.html").read_text(encoding="utf-8").lower()
