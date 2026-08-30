#!/usr/bin/env python3
"""Check initial plume-flow interpolation across mesh resolutions."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tectonics.dynamics import center_net_rotation
from tectonics.lithosphere import initialize_lithosphere
from tectonics.mantle import initialize_mantle_flow
from tectonics.plumes import (
    MantlePlumeParameters,
    initialize_mantle_plumes,
    update_plume_source_flow,
)
from tectonics.simulation import build_prototype, load_config


RADII_KM = (400.0, 550.0, 800.0, 1100.0, 1500.0)


def _initial_state(subdivisions: int):
    config = load_config(ROOT / "configs" / "canonical_moon.yaml")
    config["mesh"]["subdivisions"] = subdivisions
    proto = build_prototype(config)
    lithosphere = config["lithosphere"]
    radius_km = float(config["moon"]["radius_km"])
    state = initialize_lithosphere(
        proto.mesh,
        proto.plates,
        float(lithosphere["initial_continental_fraction"]),
        int(lithosphere["continental_nuclei"]),
        float(lithosphere["oceanic_thickness_km"]),
        float(lithosphere["continental_thickness_km"]),
        float(lithosphere["initial_continental_age_myr"]),
        radius_km=radius_km,
    )
    centered = center_net_rotation(proto.mesh, state, proto.plates, radius_km)
    return proto.mesh, initialize_mantle_flow(proto.mesh, centered), radius_km


def _sample(subdivisions: int, radius: float) -> np.ndarray:
    mesh, mantle, moon_radius = _initial_state(subdivisions)
    params = replace(
        MantlePlumeParameters(),
        source_drift_enabled=True,
        source_flow_coupling_enabled=True,
        source_flow_sampling_radius_km=radius,
    )
    plumes = initialize_mantle_plumes(mesh, 0.0, params)
    update_plume_source_flow(
        mesh,
        plumes,
        mantle.cell_omega_rad_per_myr,
        moon_radius,
        params,
        initialize_effective_velocity=True,
    )
    return np.asarray(plumes.source_flow_omega_rad_per_myr) * moon_radius


def main() -> None:
    print("radius_km,relative_vector_difference,mean_speed_sub3,mean_speed_sub4")
    for radius in RADII_KM:
        coarse = _sample(3, radius)
        fine = _sample(4, radius)
        relative = float(
            np.linalg.norm(coarse - fine) / max(np.linalg.norm(fine), 1.0e-30)
        )
        print(
            f"{radius:.0f},{relative:.9f},"
            f"{np.mean(np.linalg.norm(coarse, axis=1)):.9f},"
            f"{np.mean(np.linalg.norm(fine, axis=1)):.9f}"
        )


if __name__ == "__main__":
    main()
