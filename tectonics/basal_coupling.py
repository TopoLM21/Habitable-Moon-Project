"""Prescribed mantle traction and linear basal drag, in explicit SI units.

The source is an external traction amplitude, not a solved free-mantle velocity.
The retained Genesis constitutive choice is

    tau_b = c(H) tau_source - beta u_plate
          = beta (u_equilibrium - u_plate).

Here c transmits the prescribed driving load; beta is the unchanged independent
drag coefficient. This differs from beta*c*(u_free-u_plate), which would cancel
c in the force-free terminal speed and change the original starter experiment.
Nothing here solves plate torque balance or supplies the mature memory factor.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np

from .genesis import SECONDS_PER_MYR
from .mesh import SphereMesh

PRESCRIBED_BASAL_LAW = "prescribed_traction_linear_drag_v1"
PRESCRIBED_PATTERN = "symmetric_traceless_quadratic_potential_v1"


@dataclass(frozen=True)
class PrescribedBasalParameters:
    seed: int
    convective_traction_pa: float
    basal_drag_pa_s_m: float
    traction_coupling_depth_km: float

    def __post_init__(self):
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral) or self.seed < 0:
            raise ValueError("Basal source seed must be a nonnegative integer")
        for name in ("convective_traction_pa", "basal_drag_pa_s_m", "traction_coupling_depth_km"):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or not math.isfinite(value)
                    or value < 0 or (name != "convective_traction_pa" and value == 0)):
                raise ValueError(f"Invalid basal source {name}")

    def metadata(self):
        """Persist interpretation as well as numbers; no formation thickness."""
        return {"law": PRESCRIBED_BASAL_LAW, "pattern": PRESCRIBED_PATTERN,
            "source_kind": "uncoupled_prescribed_traction_velocity_equivalent",
            "frame": "fixed_mesh", "source_velocity_units": "m/s",
            "stored_omega_units": "rad/Myr", "seed": int(self.seed),
            "convective_traction_pa": float(self.convective_traction_pa),
            "basal_drag_pa_s_m": float(self.basal_drag_pa_s_m),
            "traction_coupling_depth_km": float(self.traction_coupling_depth_km)}


@dataclass(frozen=True)
class BasalInteraction:
    source_traction_pa: np.ndarray
    transmitted_traction_pa: np.ndarray
    equilibrium_velocity_m_s: np.ndarray
    plate_velocity_m_s: np.ndarray
    slip_traction_pa: np.ndarray
    drag_dissipation_w_m2: np.ndarray
    source_power_w_m2: np.ndarray
    net_plate_power_w_m2: np.ndarray


def basal_coupling_fraction(thickness_km, coupling_depth_km):
    thickness = np.asarray(thickness_km, dtype=np.float64)
    depth = float(coupling_depth_km)
    if (not math.isfinite(depth) or depth <= 0 or not np.isfinite(thickness).all()
            or np.any(thickness < 0)):
        raise ValueError("Basal coupling needs nonnegative finite thickness and positive depth")
    # The starter's scalar stress path historically uses libm expm1; retain
    # that rounding while array traction quadrature keeps NumPy's operation.
    if thickness.ndim == 0:
        return -math.expm1(-float(thickness) / depth)
    return -np.expm1(-thickness / depth)


def mantle_source_pattern(mesh: SphereMesh, seed: int):
    """Dimensionless vertex tangent pattern, preserving Genesis normalization."""
    rng = np.random.default_rng(int(seed))
    matrix = rng.normal(size=(3, 3))
    matrix = (matrix + matrix.T) / 2
    matrix -= np.eye(3) * np.trace(matrix) / 3
    x = mesh.vertices
    gradient = 2 * (x @ matrix.T)
    gradient -= x * np.sum(gradient * x, axis=1)[:, None]
    gradient /= max(2 * np.linalg.norm(matrix, 2), 1e-12)
    return gradient


def vertex_coupling_fraction(mesh: SphereMesh, thickness_km, coupling_depth_km):
    """Area-weight cell transmission onto vertices before traction quadrature."""
    thickness = np.asarray(thickness_km, dtype=np.float64)
    if thickness.ndim == 0:
        thickness = np.full(mesh.cell_count, float(thickness))
    if thickness.shape != (mesh.cell_count,):
        raise ValueError("Basal thickness must be scalar or one value per cell")
    coupling = basal_coupling_fraction(thickness, coupling_depth_km)
    vertex_coupling = np.zeros(mesh.vertex_count)
    vertex_area = np.zeros(mesh.vertex_count)
    np.add.at(vertex_coupling, mesh.faces.ravel(),
              np.repeat(coupling * mesh.areas_unit_sphere, 3))
    np.add.at(vertex_area, mesh.faces.ravel(), np.repeat(mesh.areas_unit_sphere, 3))
    return vertex_coupling / vertex_area


def prescribed_vertex_traction(mesh: SphereMesh, *, seed, convective_traction_pa,
                               thickness_km=None, traction_coupling_depth_km=2.0):
    """Source (H=None) or currently transmitted prescribed traction, in Pa."""
    amplitude = float(convective_traction_pa)
    if not math.isfinite(amplitude) or amplitude < 0:
        raise ValueError("Prescribed traction must be nonnegative and finite")
    pattern = mantle_source_pattern(mesh, seed)
    if thickness_km is None:
        return amplitude * pattern
    coupling = vertex_coupling_fraction(mesh, thickness_km, traction_coupling_depth_km)
    # Keep multiplication order identical to the existing starter/shell path.
    return amplitude * coupling[:, None] * pattern


def cell_tangential_vectors(mesh: SphereMesh, vertex_vectors):
    vertices = np.asarray(vertex_vectors, dtype=np.float64)
    if vertices.shape != (mesh.vertex_count, 3) or not np.isfinite(vertices).all():
        raise ValueError("Expected finite vectors at every mesh vertex")
    cells = vertices[mesh.faces].mean(axis=1)
    x = mesh.centroids
    cells -= x * np.sum(cells * x, axis=1)[:, None]
    return cells


def velocity_to_local_omega(positions, velocity_m_s, radius_km):
    """Tangential SI velocity -> minimum-norm local omega in rad/Myr."""
    x, velocity = np.asarray(positions), np.asarray(velocity_m_s)
    radius = float(radius_km)
    if (not math.isfinite(radius) or radius <= 0 or x.ndim != 2 or x.shape[1] != 3
            or velocity.shape != x.shape or not np.isfinite(x).all()
            or not np.isfinite(velocity).all()):
        raise ValueError("Invalid positions, velocity, or radius for basal omega")
    return np.cross(x, velocity) * SECONDS_PER_MYR / (radius * 1000.)


def basal_interaction_from_traction(source_traction_pa, transmitted_traction_pa,
                                    basal_drag_pa_s_m, plate_velocity_m_s=None):
    """Evaluate driving, drag, slip and power without prescribing plate motion."""
    source = np.asarray(source_traction_pa, dtype=np.float64)
    driving = np.asarray(transmitted_traction_pa, dtype=np.float64)
    beta = float(basal_drag_pa_s_m)
    plate = np.zeros_like(source) if plate_velocity_m_s is None else np.asarray(plate_velocity_m_s, dtype=np.float64)
    if (source.ndim != 2 or source.shape[1] != 3 or driving.shape != source.shape
            or plate.shape != source.shape or not np.isfinite(source).all()
            or not np.isfinite(driving).all() or not np.isfinite(plate).all()
            or not math.isfinite(beta) or beta <= 0):
        raise ValueError("Invalid basal tractions, velocity, or drag coefficient")
    equilibrium = driving / beta
    slip_traction = beta * (equilibrium - plate)
    dissipation = beta * np.sum(plate * plate, axis=1)
    source_power = np.sum(driving * plate, axis=1)
    net_power = np.sum(slip_traction * plate, axis=1)
    return BasalInteraction(source.copy(), driving.copy(), equilibrium, plate.copy(),
        slip_traction, dissipation, source_power, net_power)


def prescribed_cell_basal_state(mesh: SphereMesh, parameters: PrescribedBasalParameters,
                                thickness_km, plate_velocity_m_s=None):
    """Exact vertex loading -> cell quadrature, including nonuniform lid H.

    Averaging c*tau is essential: average(c)*average(tau) is generally a
    different vector. Coupled and uncoupled cell fields are returned separately.
    """
    arguments = dict(seed=parameters.seed, convective_traction_pa=parameters.convective_traction_pa,
                     traction_coupling_depth_km=parameters.traction_coupling_depth_km)
    source = cell_tangential_vectors(mesh, prescribed_vertex_traction(mesh, **arguments))
    transmitted = cell_tangential_vectors(mesh,
        prescribed_vertex_traction(mesh, thickness_km=thickness_km, **arguments))
    return basal_interaction_from_traction(source, transmitted, parameters.basal_drag_pa_s_m,
                                          plate_velocity_m_s)
