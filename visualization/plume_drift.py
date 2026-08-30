"""Plots for v0.30 mobile plume sources and relative source/plate motion."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _group_paths(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row["plume_id"]), []).append(row)
    return {
        plume_id: sorted(items, key=lambda item: float(item["time_myr"]))
        for plume_id, items in grouped.items()
    }


def _seam_safe_path(rows):
    lon = np.deg2rad(np.asarray([row["longitude_deg"] for row in rows], dtype=float))
    lat = np.deg2rad(np.asarray([row["latitude_deg"] for row in rows], dtype=float))
    if len(lon) > 1:
        for jump in np.flatnonzero(np.abs(np.diff(lon)) > np.pi)[::-1]:
            lon = np.insert(lon, jump + 1, np.nan)
            lat = np.insert(lat, jump + 1, np.nan)
    return lon, lat


def save_plume_source_paths(rows, path: Path, dpi: int = 180) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 6.8))
    ax = fig.add_subplot(111, projection="mollweide")
    colors = plt.get_cmap("tab10")
    for index, (plume_id, items) in enumerate(_group_paths(rows).items()):
        lon, lat = _seam_safe_path(items)
        color = colors(index % 10)
        ax.plot(lon, lat, color=color, linewidth=2.0, label=f"source {plume_id}")
        start = items[0]
        end = items[-1]
        ax.scatter(
            np.deg2rad(start["longitude_deg"]), np.deg2rad(start["latitude_deg"]),
            marker="o", s=28, facecolors="none", edgecolors=[color],
        )
        ax.scatter(
            np.deg2rad(end["longitude_deg"]), np.deg2rad(end["latitude_deg"]),
            marker="*", s=75, color=color, edgecolors="black", linewidths=0.4,
        )
    ax.grid(True, alpha=0.3)
    ax.set_title("v0.30 deep plume-source trajectories (circle: birth, star: latest)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.20), ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_plume_drift_history(rows, path: Path, dpi: int = 170) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.asarray([row["time_myr"] for row in rows], dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(time, [r["mean_source_speed_km_per_myr"] for r in rows], label="deep source")
    axes[0].plot(time, [r["mean_overlying_plate_speed_km_per_myr"] for r in rows], label="overlying plate")
    axes[0].plot(time, [r["mean_relative_track_speed_km_per_myr"] for r in rows], label="plate − source")
    axes[0].set_ylabel("Mean speed, km/Myr")
    axes[0].legend()
    axes[1].plot(time, [r["mean_source_to_plate_speed_ratio"] for r in rows], label="source / plate speed")
    axes[1].plot(time, [r["mean_source_motion_deflection_deg"] for r in rows], label="track-direction deflection, deg")
    axes[1].set_ylabel("Relative kinematics")
    axes[1].legend()
    axes[2].plot(time, [r["population_source_path_length_km"] for r in rows], label="cumulative source path, km")
    axes[2].plot(time, [r["population_source_bend_angle_deg"] for r in rows], label="cumulative conduit bend, deg")
    axes[2].set_ylabel("Population cumulative")
    axes[2].set_xlabel("Time, Myr")
    axes[2].legend()
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle("v0.30 mobile plume sources: source motion versus plate motion")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


__all__ = ["save_plume_source_paths", "save_plume_drift_history"]
