#!/usr/bin/env python3
"""Diagnose a low-motion geographic sector from v0.31 GUI checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tectonics.mesh import build_icosphere


def lon_lat(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    lat = np.degrees(np.arcsin(np.clip(points[:, 2], -1.0, 1.0)))
    return lon, lat


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    target = float(q) * float(cumulative[-1])
    return float(sorted_values[min(int(np.searchsorted(cumulative, target)), len(values) - 1)])


def checkpoint_records(output: Path) -> list[tuple[float, Path]]:
    records: list[tuple[float, Path]] = []
    for checkpoint in output.glob("gui_checkpoint_*_Myr"):
        meta_path = checkpoint / "meta.json"
        state_path = checkpoint / "state.npz"
        if not meta_path.is_file() or not state_path.is_file():
            continue
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        records.append((float(metadata["time_myr"]), checkpoint))
    return sorted(records)


def nearest_euler_pole(
    omega: np.ndarray,
    target_unit: np.ndarray,
) -> tuple[float, float, float]:
    axis = omega / max(float(np.linalg.norm(omega)), 1.0e-30)
    candidates = np.stack((axis, -axis))
    distances = np.degrees(
        np.arccos(np.clip(candidates @ target_unit, -1.0, 1.0))
    )
    pole = candidates[int(np.argmin(distances))]
    pole_lon, pole_lat = lon_lat(pole[None, :])
    return float(pole_lon[0]), float(pole_lat[0]), float(np.min(distances))


def analyze(
    output: Path,
    subdivisions: int,
    radius_km: float,
    bounds: tuple[float, float, float, float],
) -> tuple[list[dict[str, float | int]], dict[float, dict[str, np.ndarray]], list[float]]:
    mesh = build_icosphere(subdivisions)
    points = np.asarray(mesh.centroids, dtype=np.float64)
    area = np.asarray(mesh.areas_unit_sphere, dtype=np.float64)
    longitude, latitude = lon_lat(points)
    lon_min, lon_max, lat_min, lat_max = bounds
    sector = (
        (longitude >= lon_min)
        & (longitude <= lon_max)
        & (latitude >= lat_min)
        & (latitude <= lat_max)
    )
    if not np.any(sector):
        raise ValueError("Selected sector contains no mesh cells")

    center_lon = np.radians(0.5 * (lon_min + lon_max))
    center_lat = np.radians(0.5 * (lat_min + lat_max))
    target = np.array(
        [
            np.cos(center_lat) * np.cos(center_lon),
            np.cos(center_lat) * np.sin(center_lon),
            np.sin(center_lat),
        ]
    )
    rows: list[dict[str, float | int]] = []
    snapshots: dict[float, dict[str, np.ndarray]] = {}
    latest_events: list[float] = []

    for time_myr, checkpoint in checkpoint_records(output):
        metadata = json.loads((checkpoint / "meta.json").read_text(encoding="utf-8"))
        with np.load(checkpoint / "state.npz", allow_pickle=False) as state:
            owner = np.asarray(state["system_cell_plate"], dtype=np.int32)
            plates = metadata["system_plates"]
            omega = np.asarray(
                [
                    np.asarray(item["euler_axis"], dtype=np.float64)
                    * float(item["angular_speed_rad_per_myr"])
                    for item in plates
                ]
            )
            plate_velocity = np.cross(omega[owner], points) * radius_km
            plate_speed = np.linalg.norm(plate_velocity, axis=1)
            continental_fraction = np.clip(
                np.asarray(state["continental_fraction"], dtype=np.float64),
                0.0,
                1.0,
            )
            continental_weights = area * continental_fraction
            sector_continental_weights = continental_weights[sector]
            mantle_velocity = (
                np.cross(state["mantle_cell_omega_rad_per_myr"], points) * radius_km
            )
            mantle_speed = np.linalg.norm(mantle_velocity, axis=1)
            ids = np.unique(owner[sector])
            sector_areas = np.asarray(
                [np.sum(area[sector & (owner == plate_id)]) for plate_id in ids]
            )
            dominant = int(ids[int(np.argmax(sector_areas))])
            sector_weights = area[sector]
            sector_speed = plate_speed[sector]
            pole_lon, pole_lat, pole_distance = nearest_euler_pole(
                omega[dominant], target
            )
            hold_ages = np.asarray(state["transport_hold_age_myr"], dtype=float)

            rows.append(
                {
                    "time_myr": time_myr,
                    "plate_count": len(plates),
                    "dominant_plate": dominant,
                    "dominant_sector_fraction": float(
                        np.max(sector_areas) / np.sum(sector_areas)
                    ),
                    "dominant_global_area_fraction": float(
                        np.sum(area[owner == dominant]) / np.sum(area)
                    ),
                    "sector_plate_speed_mean_km_myr": float(
                        np.average(sector_speed, weights=sector_weights)
                    ),
                    "sector_plate_speed_p10_km_myr": weighted_quantile(
                        sector_speed, sector_weights, 0.10
                    ),
                    "sector_plate_speed_p90_km_myr": weighted_quantile(
                        sector_speed, sector_weights, 0.90
                    ),
                    "sector_area_below_2_km_myr": float(
                        np.sum(sector_weights[sector_speed < 2.0])
                        / np.sum(sector_weights)
                    ),
                    "sector_continental_material_fraction": float(
                        np.sum(sector_continental_weights) / np.sum(sector_weights)
                    ),
                    "sector_share_of_global_continental_material": float(
                        np.sum(sector_continental_weights)
                        / max(float(np.sum(continental_weights)), 1.0e-30)
                    ),
                    "sector_continental_plate_speed_mean_km_myr": float(
                        np.sum(sector_speed * sector_continental_weights)
                        / max(float(np.sum(sector_continental_weights)), 1.0e-30)
                    ),
                    "global_plate_speed_mean_km_myr": float(
                        np.average(plate_speed, weights=area)
                    ),
                    "global_continental_plate_speed_mean_km_myr": float(
                        np.sum(plate_speed * continental_weights)
                        / max(float(np.sum(continental_weights)), 1.0e-30)
                    ),
                    "sector_mantle_speed_mean_km_myr": float(
                        np.average(mantle_speed[sector], weights=sector_weights)
                    ),
                    "dominant_angular_speed_deg_myr": float(
                        np.degrees(np.linalg.norm(omega[dominant]))
                    ),
                    "nearest_euler_pole_lon_deg": pole_lon,
                    "nearest_euler_pole_lat_deg": pole_lat,
                    "pole_distance_from_sector_center_deg": pole_distance,
                    "dominant_transport_hold_age_myr": float(hold_ages[dominant]),
                    "transport_cumulative_commits": int(
                        metadata["transport_state"]["cumulative_commit_count"]
                    ),
                }
            )
            snapshots[time_myr] = {
                "owner": owner.copy(),
                "plate_speed": plate_speed.copy(),
                "continental_fraction": continental_fraction.copy(),
                "longitude": longitude,
                "latitude": latitude,
                "sector": sector,
            }
        latest_events = [
            float(event["time_myr"])
            for event in metadata.get("events", [])
            if event.get("kind") == "merge"
        ]
    return rows, snapshots, latest_events


def write_csv(rows: list[dict[str, float | int]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_diagnostics(
    rows: list[dict[str, float | int]],
    snapshot: dict[str, np.ndarray],
    map_row: dict[str, float | int],
    merge_times: list[float],
    bounds: tuple[float, float, float, float],
    path: Path,
) -> None:
    time = np.asarray([row["time_myr"] for row in rows], dtype=float)
    sector_mean = np.asarray(
        [row["sector_plate_speed_mean_km_myr"] for row in rows], dtype=float
    )
    sector_p10 = np.asarray(
        [row["sector_plate_speed_p10_km_myr"] for row in rows], dtype=float
    )
    sector_p90 = np.asarray(
        [row["sector_plate_speed_p90_km_myr"] for row in rows], dtype=float
    )
    global_mean = np.asarray(
        [row["global_plate_speed_mean_km_myr"] for row in rows], dtype=float
    )
    mantle_mean = np.asarray(
        [row["sector_mantle_speed_mean_km_myr"] for row in rows], dtype=float
    )
    dominant_fraction = np.asarray(
        [row["dominant_sector_fraction"] for row in rows], dtype=float
    )
    low_fraction = np.asarray(
        [row["sector_area_below_2_km_myr"] for row in rows], dtype=float
    )
    angular_speed = np.asarray(
        [row["dominant_angular_speed_deg_myr"] for row in rows], dtype=float
    )

    fig = plt.figure(figsize=(15.5, 8.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.35, 1.0))
    map_ax = fig.add_subplot(grid[:, 0])
    speed_ax = fig.add_subplot(grid[0, 1])
    cause_ax = fig.add_subplot(grid[1, 1])

    vmax = max(20.0, float(np.quantile(snapshot["plate_speed"], 0.95)))
    scatter = map_ax.scatter(
        snapshot["longitude"],
        snapshot["latitude"],
        c=snapshot["plate_speed"],
        s=7,
        linewidths=0,
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
        rasterized=True,
    )
    lon_min, lon_max, lat_min, lat_max = bounds
    map_ax.add_patch(
        Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            fill=False,
            edgecolor="crimson",
            linewidth=2.0,
            label="Investigated sector",
        )
    )
    map_ax.scatter(
        [map_row["nearest_euler_pole_lon_deg"]],
        [map_row["nearest_euler_pole_lat_deg"]],
        marker="X",
        s=160,
        c="white",
        edgecolors="black",
        linewidths=1.2,
        label="Nearest Euler pole",
        zorder=5,
    )
    map_ax.set(
        title=f"Rigid-plate speed at {map_row['time_myr']:.0f} Myr",
        xlabel="Longitude, degrees",
        ylabel="Latitude, degrees",
        xlim=(-180, 180),
        ylim=(-90, 90),
    )
    map_ax.set_xticks(np.arange(-180, 181, 45))
    map_ax.set_yticks(np.arange(-90, 91, 30))
    map_ax.grid(alpha=0.2)
    map_ax.legend(loc="upper left")
    colorbar = fig.colorbar(scatter, ax=map_ax, orientation="horizontal", pad=0.07)
    colorbar.set_label("Surface speed, km/Myr")

    speed_ax.fill_between(time, sector_p10, sector_p90, alpha=0.2, label="Sector p10–p90")
    speed_ax.plot(time, sector_mean, marker="o", markersize=3, label="Sector plate mean")
    speed_ax.plot(time, global_mean, label="Global plate mean")
    speed_ax.plot(time, mantle_mean, label="Sector mantle mean")
    speed_ax.set(
        title="The low-motion interval is local to the rigid plate",
        xlabel="Time, Myr",
        ylabel="Speed, km/Myr",
        xlim=(max(0.0, float(np.min(time))), float(np.max(time))),
    )
    speed_ax.grid(alpha=0.25)
    speed_ax.legend(loc="best")

    cause_ax.plot(time, dominant_fraction * 100.0, label="Dominant plate in sector, %")
    cause_ax.plot(time, low_fraction * 100.0, label="Area below 2 km/Myr, %")
    cause_ax.set(
        title="Mergers place one large plate and its Euler pole in the sector",
        xlabel="Time, Myr",
        ylabel="Sector area, %",
        ylim=(0, 105),
    )
    cause_ax.grid(alpha=0.25)
    omega_ax = cause_ax.twinx()
    omega_ax.plot(
        time,
        angular_speed,
        color="black",
        linestyle="--",
        label="Dominant angular speed",
    )
    omega_ax.set_ylabel("Angular speed, degrees/Myr")
    for event_time in merge_times:
        if float(np.min(time)) <= event_time <= float(np.max(time)):
            cause_ax.axvline(event_time, color="0.55", linewidth=0.8, alpha=0.7)
    lines, labels = cause_ax.get_legend_handles_labels()
    lines2, labels2 = omega_ax.get_legend_handles_labels()
    cause_ax.legend(lines + lines2, labels + labels2, loc="best")

    fig.suptitle(
        "v0.31 diagnostic: 0–90°E, 60°S–0° low-motion sector",
        fontsize=15,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--subdivisions", type=int, default=5)
    parser.add_argument("--radius-km", type=float, default=5287.0)
    parser.add_argument("--map-time", type=float, default=380.0)
    parser.add_argument("--lon-min", type=float, default=0.0)
    parser.add_argument("--lon-max", type=float, default=90.0)
    parser.add_argument("--lat-min", type=float, default=-60.0)
    parser.add_argument("--lat-max", type=float, default=0.0)
    args = parser.parse_args()

    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bounds = (args.lon_min, args.lon_max, args.lat_min, args.lat_max)
    rows, snapshots, merge_times = analyze(
        input_dir, args.subdivisions, args.radius_km, bounds
    )
    if not rows:
        raise SystemExit("No complete GUI checkpoints found")
    map_time = min(snapshots, key=lambda value: abs(value - args.map_time))
    map_row = min(rows, key=lambda row: abs(float(row["time_myr"]) - map_time))
    csv_path = output_dir / "stagnant_sector_timeseries.csv"
    figure_path = output_dir / "stagnant_sector_diagnostics.png"
    write_csv(rows, csv_path)
    plot_diagnostics(
        rows, snapshots[map_time], map_row, merge_times, bounds, figure_path
    )
    print(csv_path)
    print(figure_path)
    print(json.dumps(map_row, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
