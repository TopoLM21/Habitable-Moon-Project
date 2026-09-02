"""Configuration loading and one-shot prototype build."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .kinematics import BoundaryRecord, classify_boundaries, rigid_motion_residual
from .mesh import SphereMesh, build_icosphere
from .plates import PlateSystem, random_plate_system


@dataclass(slots=True)
class PrototypeResult:
    mesh: SphereMesh
    plates: PlateSystem
    boundaries: list[BoundaryRecord]
    rigid_residual: float
    config: dict[str, Any]


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a mapping")
    return data


def build_initial_mesh(config: dict[str, Any]) -> SphereMesh:
    """Build fixed geometry once per optimized execution, uncached otherwise."""
    from .cpu_runtime import current_execution

    subdivisions = int(config["mesh"]["subdivisions"])
    execution = current_execution()
    if execution is None:
        return build_icosphere(subdivisions)
    return execution.initial_mesh(subdivisions, build_icosphere)


def build_prototype(config: dict[str, Any]) -> PrototypeResult:
    moon = config["moon"]
    plate_cfg = config["plates"]
    class_cfg = config["classification"]

    mesh = build_initial_mesh(config)
    plates = random_plate_system(
        mesh=mesh,
        plate_count=int(plate_cfg["count"]),
        seed=int(plate_cfg["seed"]),
        boundary_roughness=float(plate_cfg["boundary_roughness"]),
        min_speed_deg_per_myr=float(plate_cfg["min_speed_deg_per_myr"]),
        max_speed_deg_per_myr=float(plate_cfg["max_speed_deg_per_myr"]),
    )
    boundaries = classify_boundaries(
        mesh=mesh,
        system=plates,
        radius_km=float(moon["radius_km"]),
        normal_threshold_km_per_myr=float(class_cfg["normal_threshold_km_per_myr"]),
        inactive_speed_km_per_myr=float(class_cfg["inactive_speed_km_per_myr"]),
    )
    residual = rigid_motion_residual(mesh, plates)
    return PrototypeResult(mesh, plates, boundaries, residual, config)
