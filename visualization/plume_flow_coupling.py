"""v0.31 resolved-flow/residual plume-source diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_plume_flow_coupling_history(rows, path: Path, dpi: int = 170) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.asarray([row["time_myr"] for row in rows], dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    axes[0].plot(time, [r["mean_resolved_flow_speed_km_per_myr"] for r in rows], label="resolved mantle-flow component")
    axes[0].plot(time, [r["mean_residual_speed_km_per_myr"] for r in rows], label="unresolved residual")
    axes[0].plot(time, [r["mean_effective_source_speed_km_per_myr"] for r in rows], label="effective source")
    axes[0].set_ylabel("Mean speed, km/Myr")
    axes[0].legend()
    axes[1].plot(time, [r["mean_effective_flow_alignment"] for r in rows], label="effective/flow alignment")
    axes[1].plot(time, [r["mean_flow_velocity_fraction_of_effective_speed"] for r in rows], label="flow speed / effective speed")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Dimensionless")
    axes[1].set_xlabel("Time, Myr")
    axes[1].legend()
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle("v0.31 mantle-flow-coupled plume sources")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


__all__ = ["save_plume_flow_coupling_history"]
