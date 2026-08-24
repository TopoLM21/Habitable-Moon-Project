"""Independent fixed-grid mantle-flow memory for the v0.9.8+ reconstruction.

The lost v0.9.8 source is not available, but its preserved diagnostics specify
one essential architectural rule: mantle motion is an Eulerian field attached
to the fixed spherical mesh, not to mutable plate IDs.  Topology changes must
therefore never replace mantle memory with the current plate velocity.

This reconstruction keeps that invariant explicit.  The initial field is built
from the centred formation-era plate motion.  Its *spatial pattern* remains on
the fixed mesh while its amplitude relaxes slowly with thermal activity.  The
amplitude-evolution coefficients below are reconstruction assumptions, not
recovered historical calibration constants.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .mesh import SphereMesh
from .plates import PlateSystem

Array = np.ndarray


@dataclass(slots=True, frozen=True)
class MantleFlowParameters:
    # Reconstructed, deliberately slow thermal modulation.  Keeping a sizeable
    # floor prevents late-time mantle memory from collapsing merely because the
    # scalar tectonic-activity proxy declines.
    amplitude_relaxation_myr: float = 600.0
    thermal_speed_floor_fraction: float = 0.72
    thermal_activity_exponent: float = 0.35
    spatial_smoothing_fraction_per_myr: float = 0.00035


@dataclass(slots=True)
class MantleFlowState:
    time_myr: float
    cell_omega_rad_per_myr: Array       # (N,3), fixed-grid Eulerian field
    formation_rms_rad_per_myr: float


@dataclass(slots=True)
class MantleFlowDiagnostics:
    time_myr: float
    rms_speed_deg_per_myr: float
    target_amplitude_fraction: float
    realised_amplitude_fraction: float


def _plate_omega(system: PlateSystem) -> Array:
    return np.asarray(
        [p.euler_axis * p.angular_speed_rad_per_myr for p in system.plates],
        dtype=np.float64,
    )


def initialize_mantle_flow(mesh: SphereMesh, formation_system: PlateSystem) -> MantleFlowState:
    """Seed the fixed-grid field from the centred formation-era plate motion."""
    omega = _plate_omega(formation_system)
    owner = np.asarray(formation_system.cell_plate, dtype=np.int32)
    if owner.shape != (mesh.cell_count,):
        raise ValueError("formation system does not match mesh")
    field = omega[owner].copy()
    rms = float(np.sqrt(np.mean(np.sum(field * field, axis=1)))) if len(field) else 0.0
    return MantleFlowState(time_myr=0.0, cell_omega_rad_per_myr=field, formation_rms_rad_per_myr=rms)


def mantle_flow_rms_rad_per_myr(state: MantleFlowState) -> float:
    field = np.asarray(state.cell_omega_rad_per_myr, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(field * field, axis=1)))) if len(field) else 0.0


def advance_mantle_flow(
    mesh: SphereMesh,
    state: MantleFlowState,
    dt_myr: float,
    tectonic_activity_factor: float,
    params: MantleFlowParameters | None = None,
) -> tuple[MantleFlowState, MantleFlowDiagnostics]:
    """Evolve mantle memory without consulting plate IDs.

    This is intentionally weak evolution: neighbour diffusion makes the old
    plate-shaped seed field gradually smoother, while a slow scalar relaxation
    follows thermal activity.  Crucially, no topology event can overwrite it.
    """
    if params is None:
        params = MantleFlowParameters()
    if dt_myr <= 0.0:
        raise ValueError("dt_myr must be positive")
    field = np.asarray(state.cell_omega_rad_per_myr, dtype=np.float64).copy()
    if field.shape != (mesh.cell_count, 3):
        raise ValueError("mantle field does not match mesh")

    # Small explicit graph diffusion on the *fixed* mesh.
    smooth = float(np.clip(params.spatial_smoothing_fraction_per_myr * dt_myr, 0.0, 0.20))
    if smooth > 0.0:
        neighbour_mean = np.empty_like(field)
        for i, nbs in enumerate(mesh.neighbors):
            if nbs:
                neighbour_mean[i] = np.mean(field[np.asarray(nbs, dtype=np.int32)], axis=0)
            else:
                neighbour_mean[i] = field[i]
        field += smooth * (neighbour_mean - field)

    activity = max(float(tectonic_activity_factor), 0.0)
    floor = float(np.clip(params.thermal_speed_floor_fraction, 0.0, 1.0))
    target_fraction = floor + (1.0 - floor) * activity ** float(params.thermal_activity_exponent)
    target_rms = float(state.formation_rms_rad_per_myr) * target_fraction
    current_rms = float(np.sqrt(np.mean(np.sum(field * field, axis=1)))) if len(field) else 0.0
    if current_rms > 1e-15:
        alpha = 1.0 - np.exp(-float(dt_myr) / max(float(params.amplitude_relaxation_myr), 1e-9))
        scale = 1.0 + alpha * (target_rms / current_rms - 1.0)
        field *= scale

    new_state = MantleFlowState(
        time_myr=float(state.time_myr + dt_myr),
        cell_omega_rad_per_myr=field,
        formation_rms_rad_per_myr=float(state.formation_rms_rad_per_myr),
    )
    rms = mantle_flow_rms_rad_per_myr(new_state)
    diag = MantleFlowDiagnostics(
        time_myr=float(new_state.time_myr),
        rms_speed_deg_per_myr=float(np.rad2deg(rms)),
        target_amplitude_fraction=float(target_fraction),
        realised_amplitude_fraction=float(rms / max(state.formation_rms_rad_per_myr, 1e-30)),
    )
    return new_state, diag


def plate_mean_mantle_omega(
    mesh: SphereMesh,
    cell_plate: Array,
    plate_count: int,
    radius_km: float,
    mantle: MantleFlowState,
) -> Array:
    """Area-weighted fixed-grid mantle omega underneath each current plate."""
    owner = np.asarray(cell_plate, dtype=np.int32)
    field = np.asarray(mantle.cell_omega_rad_per_myr, dtype=np.float64)
    if owner.shape != (mesh.cell_count,) or field.shape != (mesh.cell_count, 3):
        raise ValueError("mantle/current plate field does not match mesh")
    areas = mesh.physical_cell_areas_km2(radius_km)
    out = np.zeros((int(plate_count), 3), dtype=np.float64)
    weight = np.bincount(owner, weights=areas, minlength=int(plate_count)).astype(np.float64)
    for axis in range(3):
        out[:, axis] = np.bincount(
            owner, weights=areas * field[:, axis], minlength=int(plate_count)
        )
    nz = weight > 0.0
    out[nz] /= weight[nz, None]
    return out


__all__ = [
    "MantleFlowParameters",
    "MantleFlowState",
    "MantleFlowDiagnostics",
    "initialize_mantle_flow",
    "advance_mantle_flow",
    "mantle_flow_rms_rad_per_myr",
    "plate_mean_mantle_omega",
]
