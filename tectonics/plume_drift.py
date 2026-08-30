"""v0.30 plume-source mobility and source/plate relative kinematics.

The source trajectory belongs to the deep plume state, while volcanic material
belongs to and is transported by the lithosphere.  Keeping those two velocities
explicit makes it possible to distinguish a bend caused by plate motion from a
bend introduced by a migrating or reorienting mantle conduit.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .mesh import SphereMesh
from .plates import PlateSystem
from .plumes import MantlePlumeParameters, MantlePlumeState

Array = np.ndarray


@dataclass(slots=True)
class PlumeDriftParameters:
    enabled: bool = True
    minimum_speed_km_per_myr: float = 8.0
    maximum_speed_km_per_myr: float = 30.0
    direction_persistence_myr: float = 80.0
    direction_memory: float = 0.65
    path_sample_interval_myr: float = 4.0
    area_normalization_exponent: float = 1.4


@dataclass(slots=True)
class PlumeDriftDiagnostics:
    time_myr: float
    enabled: bool
    active_source_count: int
    mean_source_speed_km_per_myr: float
    maximum_source_speed_km_per_myr: float
    mean_overlying_plate_speed_km_per_myr: float
    maximum_overlying_plate_speed_km_per_myr: float
    mean_relative_track_speed_km_per_myr: float
    maximum_relative_track_speed_km_per_myr: float
    mean_source_to_plate_speed_ratio: float
    mean_source_motion_deflection_deg: float
    maximum_source_motion_deflection_deg: float
    active_source_path_length_km: float
    maximum_active_source_path_length_km: float
    population_source_path_length_km: float
    active_source_bend_angle_deg: float
    maximum_active_source_bend_angle_deg: float
    population_source_bend_angle_deg: float


def validate_plume_drift_parameters(params: PlumeDriftParameters) -> None:
    if not (
        0.0 <= params.minimum_speed_km_per_myr <= params.maximum_speed_km_per_myr
    ):
        raise ValueError("plume drift speed bounds must be non-negative and ordered")
    if params.direction_persistence_myr <= 0.0:
        raise ValueError("direction_persistence_myr must be positive")
    if not (0.0 <= params.direction_memory <= 1.0):
        raise ValueError("direction_memory must be in [0, 1]")
    if params.path_sample_interval_myr <= 0.0:
        raise ValueError("path_sample_interval_myr must be positive")
    if params.area_normalization_exponent <= 0.0:
        raise ValueError("area_normalization_exponent must be positive")


def plume_parameters_with_drift(
    plume_params: MantlePlumeParameters,
    params: PlumeDriftParameters,
) -> MantlePlumeParameters:
    """Opt the shared plume population into v0.30 source mobility."""

    validate_plume_drift_parameters(params)
    return replace(
        plume_params,
        source_drift_enabled=bool(params.enabled),
        minimum_source_drift_km_per_myr=float(params.minimum_speed_km_per_myr),
        maximum_source_drift_km_per_myr=float(params.maximum_speed_km_per_myr),
        source_drift_persistence_myr=float(params.direction_persistence_myr),
        source_drift_direction_memory=float(params.direction_memory),
        component_flux_area_normalization_exponent=float(
            params.area_normalization_exponent
        ),
    )


def _active_arrays(state: MantlePlumeState) -> tuple[Array, Array, Array, Array, Array, Array]:
    count = len(state.ages_myr)

    def vector_or_zeros(values, width: int | None = None, *, dtype=np.float64):
        shape = (count, width) if width is not None else (count,)
        if values is None or np.asarray(values).shape != shape:
            return np.zeros(shape, dtype=dtype)
        return np.asarray(values, dtype=dtype)

    plume_ids = vector_or_zeros(state.plume_ids, dtype=np.int64)
    axes = vector_or_zeros(
        state.last_effective_source_axes_unit
        if state.last_effective_source_axes_unit is not None
        else state.source_drift_axes_unit,
        3,
    )
    speeds = vector_or_zeros(
        state.last_effective_source_speeds_km_per_myr
        if state.last_effective_source_speeds_km_per_myr is not None
        else state.source_drift_speeds_km_per_myr
    )
    distances = vector_or_zeros(state.cumulative_source_distance_km)
    bends = vector_or_zeros(state.cumulative_source_bend_deg)
    centers = np.asarray(state.centers_unit, dtype=np.float64)
    return plume_ids, centers, axes, speeds, distances, bends


def source_plate_kinematics(
    mesh: SphereMesh,
    plume_state: MantlePlumeState,
    plate_system: PlateSystem | None,
    radius_km: float,
) -> dict[str, Array]:
    """Return source, plate and plate-relative velocity at every active source."""

    plume_ids, centers, axes, speeds, distances, bends = _active_arrays(plume_state)
    count = len(centers)
    source_vectors = np.zeros((count, 3), dtype=np.float64)
    plate_vectors = np.zeros((count, 3), dtype=np.float64)
    plate_ids = np.full(count, -1, dtype=np.int64)
    for i in range(count):
        source_vectors[i] = (
            np.cross(axes[i], centers[i]) * float(speeds[i])
        )
    if plate_system is not None and count:
        plate_by_id = {int(plate.plate_id): plate for plate in plate_system.plates}
        nearest = np.argmax(np.asarray(mesh.centroids) @ centers.T, axis=0)
        for i, cell in enumerate(nearest):
            plate_id = int(plate_system.cell_plate[int(cell)])
            plate_ids[i] = plate_id
            plate = plate_by_id.get(plate_id)
            if plate is None:
                continue
            omega = (
                np.asarray(plate.euler_axis, dtype=np.float64)
                * float(plate.angular_speed_rad_per_myr)
            )
            plate_vectors[i] = np.cross(omega, centers[i]) * float(radius_km)
    relative_vectors = plate_vectors - source_vectors
    source_norm = np.linalg.norm(source_vectors, axis=1)
    plate_norm = np.linalg.norm(plate_vectors, axis=1)
    relative_norm = np.linalg.norm(relative_vectors, axis=1)
    ratios = np.divide(
        source_norm,
        plate_norm,
        out=np.zeros_like(source_norm),
        where=plate_norm > 1.0e-12,
    )
    deflections = np.zeros(count, dtype=np.float64)
    valid = (plate_norm > 1.0e-12) & (relative_norm > 1.0e-12)
    if np.any(valid):
        cosine = np.sum(
            plate_vectors[valid] * relative_vectors[valid], axis=1
        ) / (plate_norm[valid] * relative_norm[valid])
        deflections[valid] = np.rad2deg(
            np.arccos(np.clip(cosine, -1.0, 1.0))
        )
    return {
        "plume_ids": plume_ids,
        "centers_unit": centers,
        "source_velocity_km_per_myr": source_vectors,
        "plate_velocity_km_per_myr": plate_vectors,
        "relative_velocity_km_per_myr": relative_vectors,
        "source_speed_km_per_myr": source_norm,
        "plate_speed_km_per_myr": plate_norm,
        "relative_speed_km_per_myr": relative_norm,
        "source_to_plate_speed_ratio": ratios,
        "source_motion_deflection_deg": deflections,
        "plate_ids": plate_ids,
        "cumulative_source_distance_km": distances,
        "cumulative_source_bend_deg": bends,
    }


def diagnose_plume_drift(
    mesh: SphereMesh,
    plume_state: MantlePlumeState,
    plate_system: PlateSystem | None,
    radius_km: float,
    params: PlumeDriftParameters,
) -> PlumeDriftDiagnostics:
    validate_plume_drift_parameters(params)
    fields = source_plate_kinematics(mesh, plume_state, plate_system, radius_km)

    def mean(name: str) -> float:
        values = fields[name]
        return float(np.mean(values)) if len(values) else 0.0

    def maximum(name: str) -> float:
        values = fields[name]
        return float(np.max(values)) if len(values) else 0.0

    distances = fields["cumulative_source_distance_km"]
    bends = fields["cumulative_source_bend_deg"]
    return PlumeDriftDiagnostics(
        time_myr=float(plume_state.time_myr),
        enabled=bool(params.enabled),
        active_source_count=int(len(fields["plume_ids"])),
        mean_source_speed_km_per_myr=mean("source_speed_km_per_myr"),
        maximum_source_speed_km_per_myr=maximum("source_speed_km_per_myr"),
        mean_overlying_plate_speed_km_per_myr=mean("plate_speed_km_per_myr"),
        maximum_overlying_plate_speed_km_per_myr=maximum("plate_speed_km_per_myr"),
        mean_relative_track_speed_km_per_myr=mean("relative_speed_km_per_myr"),
        maximum_relative_track_speed_km_per_myr=maximum("relative_speed_km_per_myr"),
        mean_source_to_plate_speed_ratio=mean("source_to_plate_speed_ratio"),
        mean_source_motion_deflection_deg=mean("source_motion_deflection_deg"),
        maximum_source_motion_deflection_deg=maximum(
            "source_motion_deflection_deg"
        ),
        active_source_path_length_km=float(np.sum(distances)),
        maximum_active_source_path_length_km=(
            float(np.max(distances)) if len(distances) else 0.0
        ),
        population_source_path_length_km=float(
            plume_state.population_source_distance_km
        ),
        active_source_bend_angle_deg=float(np.sum(bends)),
        maximum_active_source_bend_angle_deg=(
            float(np.max(bends)) if len(bends) else 0.0
        ),
        population_source_bend_angle_deg=float(plume_state.population_source_bend_deg),
    )


def source_path_rows(
    mesh: SphereMesh,
    plume_state: MantlePlumeState,
    plate_system: PlateSystem | None,
    radius_km: float,
) -> list[dict]:
    """Create scalar checkpoint-safe samples of all current source positions."""

    fields = source_plate_kinematics(mesh, plume_state, plate_system, radius_km)
    rows: list[dict] = []
    for i, center in enumerate(fields["centers_unit"]):
        lon = float(np.rad2deg(np.arctan2(center[1], center[0])))
        lat = float(np.rad2deg(np.arcsin(np.clip(center[2], -1.0, 1.0))))
        rows.append(
            {
                "time_myr": float(plume_state.time_myr),
                "plume_id": int(fields["plume_ids"][i]),
                "age_myr": float(plume_state.ages_myr[i]),
                "longitude_deg": lon,
                "latitude_deg": lat,
                "overlying_plate_id": int(fields["plate_ids"][i]),
                "source_speed_km_per_myr": float(
                    fields["source_speed_km_per_myr"][i]
                ),
                "overlying_plate_speed_km_per_myr": float(
                    fields["plate_speed_km_per_myr"][i]
                ),
                "relative_track_speed_km_per_myr": float(
                    fields["relative_speed_km_per_myr"][i]
                ),
                "source_to_plate_speed_ratio": float(
                    fields["source_to_plate_speed_ratio"][i]
                ),
                "source_motion_deflection_deg": float(
                    fields["source_motion_deflection_deg"][i]
                ),
                "cumulative_source_distance_km": float(
                    fields["cumulative_source_distance_km"][i]
                ),
                "cumulative_source_bend_deg": float(
                    fields["cumulative_source_bend_deg"][i]
                ),
            }
        )
    return rows


__all__ = [
    "PlumeDriftParameters",
    "PlumeDriftDiagnostics",
    "validate_plume_drift_parameters",
    "plume_parameters_with_drift",
    "source_plate_kinematics",
    "diagnose_plume_drift",
    "source_path_rows",
]
