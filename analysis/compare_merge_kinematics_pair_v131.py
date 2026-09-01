#!/usr/bin/env python3
"""Compare the paired v0.31 merger-kinematics checkpoint branches."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from diagnose_stagnant_sector_v131 import analyze


RULE_LABELS = {
    "area_weighted": "Area-weighted control",
    "inertia_tensor": "Inertia-tensor merger",
}


def event_times(case: Path) -> tuple[list[float], list[float]]:
    checkpoints = sorted(case.glob("gui_checkpoint_*_Myr"))
    metadata = json.loads((checkpoints[-1] / "meta.json").read_text(encoding="utf-8"))
    mergers = [
        float(event["time_myr"])
        for event in metadata.get("events", [])
        if event.get("kind") == "merge" and float(event["time_myr"]) >= 300.0
    ]
    disconnects = [
        float(event["time_myr"])
        for event in metadata.get("events", [])
        if event.get("kind") == "disconnect_split" and float(event["time_myr"]) >= 300.0
    ]
    return mergers, disconnects


def row_at(rows: list[dict[str, float | int]], time_myr: float) -> dict[str, float | int]:
    return min(rows, key=lambda row: abs(float(row["time_myr"]) - time_myr))


def map_panel(
    axis,
    snapshot: dict[str, np.ndarray],
    row: dict[str, float | int],
    title: str,
    bounds: tuple[float, float, float, float],
    vmax: float,
):
    artist = axis.scatter(
        snapshot["longitude"],
        snapshot["latitude"],
        c=snapshot["plate_speed"],
        s=6,
        linewidths=0,
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
        rasterized=True,
    )
    lon_min, lon_max, lat_min, lat_max = bounds
    axis.add_patch(
        Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            fill=False,
            edgecolor="crimson",
            linewidth=1.8,
        )
    )
    axis.scatter(
        [row["nearest_euler_pole_lon_deg"]],
        [row["nearest_euler_pole_lat_deg"]],
        marker="X",
        s=120,
        c="white",
        edgecolors="black",
        linewidths=1.0,
        zorder=5,
    )
    axis.set(
        title=title,
        xlabel="Longitude, degrees",
        ylabel="Latitude, degrees",
        xlim=(-180, 180),
        ylim=(-90, 90),
    )
    axis.set_xticks(np.arange(-180, 181, 60))
    axis.set_yticks(np.arange(-90, 91, 30))
    axis.grid(alpha=0.2)
    return artist


def write_comparison_csv(
    rows_by_rule: dict[str, list[dict[str, float | int]]],
    path: Path,
) -> None:
    times = sorted(
        set.intersection(
            *(set(float(row["time_myr"]) for row in rows) for rows in rows_by_rule.values())
        )
    )
    output_rows = []
    for time_myr in times:
        item: dict[str, float | int] = {"time_myr": time_myr}
        for rule, rows in rows_by_rule.items():
            row = row_at(rows, time_myr)
            for key in (
                "plate_count",
                "dominant_sector_fraction",
                "sector_plate_speed_mean_km_myr",
                "sector_plate_speed_p10_km_myr",
                "sector_plate_speed_p90_km_myr",
                "sector_area_below_2_km_myr",
                "dominant_angular_speed_deg_myr",
                "nearest_euler_pole_lon_deg",
                "nearest_euler_pole_lat_deg",
            ):
                item[f"{rule}_{key}"] = row[key]
        output_rows.append(item)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-root", required=True, type=Path)
    parser.add_argument("--initial-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--map-time", type=float, default=380.0)
    parser.add_argument("--subdivisions", type=int, default=5)
    parser.add_argument("--radius-km", type=float, default=5287.0)
    args = parser.parse_args()

    bounds = (0.0, 90.0, -60.0, 0.0)
    initial_rows, initial_snapshots, _ = analyze(
        args.initial_root.resolve(), args.subdivisions, args.radius_km, bounds
    )
    initial_row = row_at(initial_rows, 300.0)
    initial_snapshot = initial_snapshots[float(initial_row["time_myr"])]
    rows_by_rule: dict[str, list[dict[str, float | int]]] = {}
    snapshots_by_rule: dict[str, dict[float, dict[str, np.ndarray]]] = {}
    events_by_rule: dict[str, tuple[list[float], list[float]]] = {}
    for rule in RULE_LABELS:
        case = args.pair_root.resolve() / rule
        rows, snapshots, _ = analyze(case, args.subdivisions, args.radius_km, bounds)
        rows_by_rule[rule] = [dict(initial_row), *rows]
        snapshots_by_rule[rule] = {300.0: initial_snapshot, **snapshots}
        events_by_rule[rule] = event_times(case)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_comparison_csv(rows_by_rule, output / "merge_kinematics_sector_comparison.csv")

    map_time = min(
        snapshots_by_rule["area_weighted"],
        key=lambda value: abs(value - args.map_time),
    )
    combined_speed = np.concatenate(
        [snapshots_by_rule[rule][map_time]["plate_speed"] for rule in RULE_LABELS]
    )
    vmax = max(20.0, float(np.quantile(combined_speed, 0.95)))
    figure = plt.figure(figsize=(15.5, 10), constrained_layout=True)
    grid = figure.add_gridspec(2, 2)
    map_axes = [figure.add_subplot(grid[0, index]) for index in range(2)]
    speed_axis = figure.add_subplot(grid[1, 0])
    topology_axis = figure.add_subplot(grid[1, 1])
    map_artist = None
    colors = {"area_weighted": "tab:blue", "inertia_tensor": "tab:orange"}
    for axis, rule in zip(map_axes, RULE_LABELS):
        row = row_at(rows_by_rule[rule], map_time)
        map_artist = map_panel(
            axis,
            snapshots_by_rule[rule][map_time],
            row,
            f"{RULE_LABELS[rule]} at {map_time:.0f} Myr\n"
            f"sector mean {float(row['sector_plate_speed_mean_km_myr']):.2f} km/Myr",
            bounds,
            vmax,
        )
    figure.colorbar(map_artist, ax=map_axes, orientation="horizontal", shrink=0.75, pad=0.08).set_label(
        "Rigid-plate surface speed, km/Myr"
    )

    for rule, rows in rows_by_rule.items():
        time = np.asarray([row["time_myr"] for row in rows], dtype=float)
        mean = np.asarray([row["sector_plate_speed_mean_km_myr"] for row in rows], dtype=float)
        low = np.asarray([row["sector_plate_speed_p10_km_myr"] for row in rows], dtype=float)
        high = np.asarray([row["sector_plate_speed_p90_km_myr"] for row in rows], dtype=float)
        speed_axis.fill_between(time, low, high, color=colors[rule], alpha=0.14)
        speed_axis.plot(time, mean, marker="o", color=colors[rule], label=RULE_LABELS[rule])
    speed_axis.set(
        title="Sector speed: inertia weighting weakens but does not remove the quiet zone",
        xlabel="Time, Myr",
        ylabel="Speed, km/Myr",
        xlim=(300, 400),
    )
    speed_axis.grid(alpha=0.25)
    speed_axis.legend()

    plate_axis = topology_axis.twinx()
    for rule, rows in rows_by_rule.items():
        time = np.asarray([row["time_myr"] for row in rows], dtype=float)
        low_area = np.asarray([row["sector_area_below_2_km_myr"] for row in rows], dtype=float) * 100.0
        plates = np.asarray([row["plate_count"] for row in rows], dtype=float)
        topology_axis.plot(time, low_area, marker="o", color=colors[rule], label=f"{RULE_LABELS[rule]}: <2 km/Myr")
        plate_axis.step(time, plates, where="post", linestyle="--", color=colors[rule], alpha=0.8, label=f"{RULE_LABELS[rule]}: plates")
    for split_time in events_by_rule["inertia_tensor"][1]:
        topology_axis.axvline(split_time, color="black", linewidth=1.0, alpha=0.6)
        topology_axis.text(split_time + 1.0, 37.0, "disconnect split", rotation=90, va="top")
    topology_axis.set(
        title="The inertia branch creates a macroscopic seventh plate at 372 Myr",
        xlabel="Time, Myr",
        ylabel="Sector area below 2 km/Myr, %",
        xlim=(300, 400),
        ylim=(0, 40),
    )
    plate_axis.set_ylabel("Plate count")
    plate_axis.set_ylim(5.5, 9.5)
    topology_axis.grid(alpha=0.25)
    lines, labels = topology_axis.get_legend_handles_labels()
    lines2, labels2 = plate_axis.get_legend_handles_labels()
    topology_axis.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=8)

    figure.suptitle(
        "v0.31 paired replay from the same 300 Myr checkpoint",
        fontsize=15,
    )
    figure.savefig(output / "merge_kinematics_sector_comparison.png", dpi=180)
    plt.close(figure)
    print(output / "merge_kinematics_sector_comparison.csv")
    print(output / "merge_kinematics_sector_comparison.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
