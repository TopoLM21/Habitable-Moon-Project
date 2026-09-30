"""Quasistatic SI basal mechanics on an explicitly fractional oceanic surface.

Every material component contributes its own area, owner and nonlinear lid
transmission. The prescribed mantle source and independent linear drag are
unchanged. This is a basal-only experiment: there is no inferred subcell ridge
or trench, slab force, velocity relaxation, or post-solve rotation removal.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np

from .basal_coupling import (
    PrescribedBasalParameters, basal_coupling_fraction,
    cell_tangential_vectors, prescribed_vertex_traction,
)
from .fractional_surface import FractionalSurfaceState
from .genesis import SECONDS_PER_MYR


MODEL = "fractional_prescribed_basal_v1"
QUADRATURE = "conditional_component_vertex_transmission_v1"


@dataclass(frozen=True)
class FractionalBasalDynamics:
    omega_rad_per_myr: np.ndarray
    plate_areas_km2: np.ndarray
    basal_drag_tensor_nm_s: np.ndarray
    basal_driving_torque_nm: np.ndarray
    torque_residual_nm: np.ndarray
    torque_relative_residual: float
    parcel_source_traction_pa: np.ndarray
    parcel_transmitted_traction_pa: np.ndarray
    parcel_velocity_m_s: np.ndarray
    parcel_total_lid_thickness_km: np.ndarray
    basal_source_power_w: float
    basal_drag_dissipation_w: float
    power_residual_w: float
    power_relative_residual: float
    mean_speed_mm_per_year: float
    max_speed_mm_per_year: float
    plate_mean_speed_mm_per_year: np.ndarray
    area_weighted_mean_omega_rad_per_myr: np.ndarray
    model: str = MODEL
    quadrature: str = QUADRATURE
    source_frame: str = "fixed_mesh"


def _geometry(mesh, surface, radius_km):
    if not isinstance(surface, FractionalSurfaceState):
        raise ValueError("Fractional basal mechanics requires a complete parcel surface")
    if isinstance(radius_km, (bool, np.bool_)) or not math.isfinite(radius_km) or radius_km <= 0:
        raise ValueError("Fractional basal radius must be positive and finite")
    expected = mesh.physical_cell_areas_km2(radius_km)
    if (surface.cell_count != mesh.cell_count or
            not np.allclose(surface.cell_areas_km2, expected, rtol=2e-13, atol=0.)):
        raise ValueError("Fractional surface capacities do not match the basal mesh/radius")
    x = np.asarray(mesh.centroids, dtype=np.float64)
    if (x.shape != (mesh.cell_count, 3) or not np.isfinite(x).all() or
            not np.allclose(np.sum(x*x, axis=1), 1., rtol=0., atol=2e-14)):
        raise ValueError("Fractional basal quadrature requires unit cell positions")
    return x


def _lid_thickness(surface, thickness):
    if thickness is None:
        try:
            thickness = [dict(parcel.material_fields)["thermal_total_lid_thickness_km"]
                         for parcel in surface.parcels]
        except KeyError as exc:
            raise ValueError("Explicit per-parcel total thermal lid thickness is required") from exc
    h = np.asarray(thickness)
    if (h.dtype.kind not in "fiu" or h.shape != (len(surface.parcels),) or
            not np.isfinite(h).all() or np.any(h < 0.)):
        raise ValueError("Total lid thickness must be a finite nonnegative value per parcel")
    # Chemical crust need not be cold. Never reconstruct this diagnostic by
    # adding chemical crust thickness to the transported cold mantle inventory.
    return h.astype(np.float64, copy=True)


def fractional_prescribed_tractions(mesh, surface, parameters, total_lid_thickness_km):
    """Return source and transmitted traction for each component in input order.

    The legacy quadrature first averages c(H) from incident cells onto each
    vertex, multiplies the vertex source, then averages vectors to the cell.
    Its extension here conditions that operation on the component being
    evaluated: its own cell contributes c(H_component); neighboring cells
    contribute their area-weighted mean *transmission*, not mean thickness.

    This is the expectation of the original quadrature over unresolved
    neighboring components. It assumes no subcell spatial correlation and is
    not a resolved component footprint. It reproduces the original quadrature
    exactly in the one-component-per-cell limit, and splitting an identical
    component changes neither its loading nor the total force. As in the old
    quadrature, vertex smoothing transmits some neighboring-cell load even to
    a zero-thickness component; this existing spatial closure stays explicit.
    """
    if not isinstance(parameters, PrescribedBasalParameters):
        raise ValueError("Fractional basal mechanics requires explicit prescribed-source parameters")
    h = _lid_thickness(surface, total_lid_thickness_km)
    cell = np.asarray([parcel.cell for parcel in surface.parcels], dtype=np.int64)
    area = np.asarray([parcel.area_km2 for parcel in surface.parcels])
    coupling = basal_coupling_fraction(h, parameters.traction_coupling_depth_km)
    mean_c = np.bincount(cell, weights=area*coupling, minlength=mesh.cell_count)
    mean_c /= np.asarray(surface.cell_areas_km2)
    vertex_area = np.zeros(mesh.vertex_count)
    vertex_c = np.zeros(mesh.vertex_count)
    weights = np.asarray(mesh.areas_unit_sphere)
    np.add.at(vertex_area, mesh.faces.ravel(), np.repeat(weights, 3))
    np.add.at(vertex_c, mesh.faces.ravel(), np.repeat(weights*mean_c, 3))
    vertex_c /= vertex_area
    source_vertex = prescribed_vertex_traction(mesh, seed=parameters.seed,
        convective_traction_pa=parameters.convective_traction_pa)
    vertices = mesh.faces[cell]
    self_weight = weights[cell, None]/vertex_area[vertices]
    conditioned_c = vertex_c[vertices]+self_weight*(coupling-mean_c[cell])[:, None]
    transmitted = np.mean(conditioned_c[:, :, None]*source_vertex[vertices], axis=1)
    x = mesh.centroids[cell]
    transmitted -= x*np.sum(transmitted*x, axis=1)[:, None]
    source = cell_tangential_vectors(mesh, source_vertex)[cell]
    return source, transmitted


def solve_fractional_basal_dynamics(mesh, surface, radius_km,
        parameters: PrescribedBasalParameters, total_lid_thickness_km=None, *,
        plate_count=None, source_frame="fixed_mesh", remove_net_rotation=False):
    """Balance each owner's component-weighted torque against its SI drag.

    The only velocity is the quasistatic solution D omega = T. An owner with
    no area returns zero velocity; a geometrically rank-deficient owner gets
    the minimum-norm solution. Power is independently integrated over parcels.
    Thickness is explicitly aligned with ``surface.parcels`` or read from each
    parcel's ``thermal_total_lid_thickness_km`` field. Imported chemical crust
    and cold mantle volumes alone cannot determine the total cold lid.
    """
    x_cell = _geometry(mesh, surface, radius_km)
    if source_frame != "fixed_mesh" or remove_net_rotation is not False:
        raise ValueError("Prescribed-source mechanics requires the shared fixed_mesh frame without rotation removal")
    owner = np.asarray([parcel.plate for parcel in surface.parcels], dtype=np.int64)
    count = int(np.max(owner))+1 if plate_count is None else plate_count
    if (isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral) or
            count <= int(np.max(owner))):
        raise ValueError("Plate count must include every fractional material owner")
    count = int(count)
    h = _lid_thickness(surface, total_lid_thickness_km)
    source, transmitted = fractional_prescribed_tractions(mesh, surface, parameters, h)
    cell = np.asarray([parcel.cell for parcel in surface.parcels], dtype=np.int64)
    area_km2 = np.asarray([parcel.area_km2 for parcel in surface.parcels])
    area = area_km2*1e6
    x, r, beta = x_cell[cell], float(radius_km)*1000., parameters.basal_drag_pa_s_m
    drag = np.zeros((count, 3, 3))
    torque = np.zeros((count, 3))
    rhs = np.cross(x, transmitted)
    for a in range(3):
        torque[:, a] = np.bincount(owner, weights=r*area*rhs[:, a], minlength=count)
        for b in range(3):
            drag[:, a, b] = np.bincount(owner,
                weights=beta*r*r*area*((a == b)-x[:, a]*x[:, b]), minlength=count)
    omega_si = np.zeros((count, 3))
    for pid in range(count):
        scale = float(np.linalg.norm(drag[pid]))
        if scale > 0.:
            omega_si[pid] = np.linalg.lstsq(drag[pid]/scale, torque[pid]/scale, rcond=None)[0]
    resistive = np.einsum("pij,pj->pi", drag, omega_si)
    residual = torque-resistive
    torque_scale = max(float(np.linalg.norm(torque)), float(np.linalg.norm(resistive)))
    relative_torque = float(np.linalg.norm(residual))/torque_scale if torque_scale else 0.
    omega = omega_si*SECONDS_PER_MYR
    velocity = np.cross(omega_si[owner], x)*r
    source_power = float(area@np.sum(transmitted*velocity, axis=1))
    dissipation = float(area@(beta*np.sum(velocity*velocity, axis=1)))
    net_power = source_power-dissipation
    power_scale = max(abs(source_power), abs(dissipation))
    relative_power = abs(net_power)/power_scale if power_scale else 0.
    speed = np.linalg.norm(velocity, axis=1)*SECONDS_PER_MYR/1000.
    plate_areas = np.bincount(owner, weights=area_km2, minlength=count)
    weighted_speeds = np.bincount(owner, weights=area_km2*speed, minlength=count)
    plate_speed = np.divide(weighted_speeds, plate_areas, out=np.zeros(count), where=plate_areas > 0.)
    mean_omega = np.sum(omega*plate_areas[:, None], axis=0)/plate_areas.sum()
    return FractionalBasalDynamics(omega, plate_areas, drag, torque, residual, relative_torque,
        source, transmitted, velocity, h, source_power, dissipation, net_power, relative_power,
        float(area_km2@speed/area_km2.sum()), float(np.max(speed)), plate_speed, mean_omega)


__all__ = ["MODEL", "QUADRATURE", "FractionalBasalDynamics",
           "fractional_prescribed_tractions", "solve_fractional_basal_dynamics"]
