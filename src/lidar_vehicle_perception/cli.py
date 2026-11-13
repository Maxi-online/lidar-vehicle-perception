"""Reproducible offline entry points. No services or model downloads on import."""

import argparse
import importlib.metadata
import os
from pathlib import Path
import platform
import time

import numpy as np

from .config import Settings
from .detector import VehicleDetector
from .io import discover_frames, read_jsonl, read_points, write_json, write_jsonl
from .metrics import evaluate
from .synthetic import generate_sequence
from .tracking import MultiObjectTracker


def replay(input_path, output, settings, fps=10.0, max_frames=60, html=False):
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    if max_frames < 0:
        raise ValueError("max_frames must be nonnegative; zero means all")
    files = discover_frames(input_path)
    if max_frames:
        files = files[:max_frames]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    detector = VehicleDetector(settings.detector)
    tracker = MultiObjectTracker(settings.tracker)
    records, durations, cloud_preview, record_preview = [], [], [], []
    last_points = None
    base = Path(input_path) if Path(input_path).is_dir() else Path(input_path).parent
    for index, path in enumerate(files):
        points = read_points(path)
        started = time.perf_counter()
        detections, stats = detector.detect(points)
        tracks = tracker.update(detections, 1.0 / fps)
        elapsed = (time.perf_counter() - started) * 1000
        record = {
            "frame_id": path.relative_to(base).as_posix(),
            "timestamp_s": index / fps,
            "detections": [d.as_dict() for d in detections],
            "tracks": [t.as_dict() for t in tracks],
            "processing_ms": elapsed,
            "stats": stats,
        }
        records.append(record)
        durations.append(elapsed)
        last_points = points
        if html and len(cloud_preview) < 60:
            cloud_preview.append(points)
            record_preview.append(record)
    write_jsonl(output / "predictions.jsonl", records)
    summary = {
        "frames": len(records),
        "detections_per_frame": [len(r["detections"]) for r in records],
        "observed_confirmed_tracks_per_frame": [
            sum(t["observed"] for t in r["tracks"]) for r in records
        ],
        "processing_ms": {
            "p50": float(np.percentile(durations, 50)),
            "p95": float(np.percentile(durations, 95)),
            "mean": float(np.mean(durations)),
        },
        "timing_scope": "Detector + tracker, includes first frame; excludes disk IO, imports and rendering.",
        "nominal_replay_fps": fps,
        "visualized_frames": len(cloud_preview),
        "settings": settings.as_dict(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.system(),
            "machine": platform.machine(),
            "logical_cpus": os.cpu_count(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset"),
            "versions": {
                p: importlib.metadata.version(p)
                for p in ("numpy", "scipy", "open3d", "hdbscan", "filterpy")
            },
        },
    }
    from .visualization import write_html, write_preview

    write_preview(output / "preview.png", last_points, records[-1])
    if html:
        write_html(output / "replay.html", cloud_preview, record_preview)
    write_json(output / "summary.json", summary)
    return records, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Geometric LiDAR vehicle proposals and tracking")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser(
        "demo", help="generate a labelled procedural scene and run the pipeline"
    )
    demo.add_argument("--output", type=Path, default=Path("artifacts/demo"))
    demo.add_argument("--frames", type=int, default=12)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--config", type=Path)
    run = subparsers.add_parser(
        "run", help="process local PCD/NPY/NPZ frames in lexical filename order"
    )
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--output", type=Path, default=Path("artifacts/run"))
    run.add_argument("--config", type=Path)
    run.add_argument("--fps", type=float, default=10.0)
    run.add_argument("--max-frames", type=int, default=60)
    run.add_argument("--html", action="store_true", help="visualize at most the first 60 frames")
    run.add_argument("--ground-truth", type=Path)
    score = subparsers.add_parser("evaluate", help="score predictions against 3D-box JSONL labels")
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--ground-truth", type=Path, required=True)
    score.add_argument("--output", type=Path, default=Path("artifacts/evaluation.json"))
    score.add_argument("--iou-threshold", type=float, default=0.3)
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate":
            metrics = evaluate(
                read_jsonl(args.predictions), read_jsonl(args.ground_truth), args.iou_threshold
            )
            write_json(args.output, metrics)
        elif args.command == "demo":
            labels = generate_sequence(args.output / "inputs", args.frames, args.seed)
            predictions, summary = replay(
                args.output / "inputs",
                args.output,
                Settings.load(args.config),
                max_frames=args.frames,
                html=True,
            )
            metrics = evaluate(predictions, labels)
            metrics["dataset"] = {
                "kind": "procedural synthetic cuboids",
                "seed": args.seed,
                "real_world_validation": False,
            }
            write_json(args.output / "evaluation.json", metrics)
            print(
                f"Processed {summary['frames']} synthetic frames; results: {args.output.resolve()}"
            )
        else:
            predictions, summary = replay(
                args.input,
                args.output,
                Settings.load(args.config),
                args.fps,
                args.max_frames,
                args.html,
            )
            if args.ground_truth:
                write_json(
                    args.output / "evaluation.json",
                    evaluate(predictions, read_jsonl(args.ground_truth)),
                )
            print(f"Processed {summary['frames']} frames; results: {args.output.resolve()}")
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
