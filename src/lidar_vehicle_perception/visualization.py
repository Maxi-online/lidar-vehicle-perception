"""Artifacts generated from actual point arrays and pipeline outputs."""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from .models import BOX_EDGES


def _traces(points, record):
    stride = max(len(points) // 1600, 1)
    sample = points[::stride]
    traces = [
        go.Scatter3d(
            x=sample[:, 0],
            y=sample[:, 1],
            z=sample[:, 2],
            mode="markers",
            marker={"size": 1.8, "color": "#8b9bb4", "opacity": 0.55},
            name="Point cloud",
        )
    ]
    # A single trace holds all box edges, keeping slider frames consistent.
    xs, ys, zs = [], [], []
    for box in record["detections"]:
        corners = np.asarray(box["corners"])
        for start, end in BOX_EDGES:
            xs += [corners[start, 0], corners[end, 0], None]
            ys += [corners[start, 1], corners[end, 1], None]
            zs += [corners[start, 2], corners[end, 2], None]
    traces.append(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line={"color": "#1dd3b0", "width": 5},
            name="Detector OBB",
        )
    )
    observed = [t for t in record["tracks"] if t["observed"]]
    traces.append(
        go.Scatter3d(
            x=[t["center"][0] for t in observed],
            y=[t["center"][1] for t in observed],
            z=[t["center"][2] + 2.0 for t in observed],
            mode="text",
            text=[f"ID {t['track_id']} · {t['class_name']}" for t in observed],
            textfont={"color": "#ffd166", "size": 14},
            name="Confirmed tracks",
        )
    )
    return traces


def write_html(path, clouds, records):
    frames = [
        go.Frame(name=str(i), data=_traces(points, record))
        for i, (points, record) in enumerate(zip(clouds, records))
    ]
    figure = go.Figure(data=frames[0].data, frames=frames)
    figure.update_layout(
        title="LiDAR vehicle perception · synthetic engineering demo",
        template="plotly_dark",
        height=760,
        margin={"l": 0, "r": 0, "b": 0, "t": 60},
        scene={
            "xaxis_title": "X [m]",
            "yaxis_title": "Y [m]",
            "zaxis_title": "Z [m]",
            "aspectmode": "data",
            "uirevision": "keep-view",
            "camera": {"eye": {"x": 1.25, "y": 1.5, "z": 0.9}},
        },
        sliders=[
            {
                "steps": [
                    {
                        "label": str(i),
                        "method": "animate",
                        "args": [
                            [str(i)],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0},
                                "transition": {"duration": 0},
                            },
                        ],
                    }
                    for i in range(len(frames))
                ]
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "bgcolor": "#203047",
                "font": {"color": "#ffffff"},
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": True,
                                "frame": {"duration": 150, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(path), include_plotlyjs=True, auto_play=False)


def write_preview(path, points, record):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(13, 7), facecolor="#101c30")
    axis = figure.add_subplot(111, projection="3d", facecolor="#101c30")
    sample = points[:: max(len(points) // 2600, 1)]
    axis.scatter(*sample.T, s=2, c="#90a5c0", alpha=0.45)
    for box in record["detections"]:
        corners = np.asarray(box["corners"])
        for start, end in BOX_EDGES:
            axis.plot(*corners[[start, end]].T, color="#1dd3b0", linewidth=1.8)
    for track in record["tracks"]:
        if track["observed"]:
            x, y, z = track["center"]
            axis.text(
                x,
                y,
                z + 2,
                f"ID {track['track_id']} · {track['class_name']}",
                color="#ffd166",
                fontsize=11,
            )
    axis.set(xlabel="X [m]", ylabel="Y [m]", zlabel="Z [m]", zlim=(-0.2, 4))
    axis.set_box_aspect((16, 24, 8))
    axis.view_init(elev=28, azim=-42)
    axis.tick_params(colors="#d9e4f5")
    for label in (axis.xaxis.label, axis.yaxis.label, axis.zaxis.label):
        label.set_color("#d9e4f5")
    figure.suptitle("LiDAR Vehicle Perception", color="white", fontsize=23, x=0.08, ha="left")
    figure.text(
        0.08,
        0.87,
        "RANSAC ground · XY HDBSCAN · OBB · Kalman / Hungarian",
        color="#b9cce6",
        fontsize=12,
    )
    figure.text(
        0.08,
        0.05,
        "Procedural synthetic scene. Engineering demonstration; no real-world accuracy claim.",
        color="#b9cce6",
        fontsize=10,
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)
