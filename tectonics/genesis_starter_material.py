"""Explicit primary-material view of the starter's existing thermal inventory.

The global model already reserves a surface silicate mass distinct from its
mantle mass. Here its solid part is *assumed* to be primary mafic material,
representable by the mature oceanic endmember. This is a coarse composition
closure, not a prediction of differentiation, melt extraction, or crust age.
No continental material is introduced. Chemical thickness is calculated from
this reserved mass and its stated density, never from mechanical lid depth.

All phase energies below partition the existing global thermal reservoirs.
They are diagnostic views, not additional heat stores. In particular the
passive conductive reference column is excluded: it overlaps those reservoirs
and must not be counted a second time. The genesis heat/orbit/water owner must
continue to evolve its own budget during any young-world continuation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .genesis import ENERGY_SCALE, SECONDS_PER_MYR
from .genesis_shell import mantle_traction


@dataclass(frozen=True)
class PrimaryMaterialInventory:
    primary_mass_kg: np.ndarray
    surface_remaining_mass_kg: np.ndarray
    mantle_mass_kg: np.ndarray
    primary_thickness_km: np.ndarray
    surface_solid_fraction: float
    liquid_water_mass_kg: np.ndarray
    vapor_water_mass_kg: np.ndarray
    primary_enthalpy_j: np.ndarray
    surface_remaining_enthalpy_j: np.ndarray
    mantle_enthalpy_j: np.ndarray
    water_enthalpy_j: np.ndarray
    surface_temperature_k: float
    mantle_temperature_k: float
    primary_density_kg_m3: float
    mechanical_lid_thickness_km: float
    fully_solid_surface: bool
    silicate_mass_relative_residual: float
    water_mass_relative_residual: float
    thermal_energy_relative_residual: float

    @property
    def source_silicate_mass_kg(self):
        """Entire modeled silicate inventory; primary material is a subset."""
        return self.primary_mass_kg+self.surface_remaining_mass_kg+self.mantle_mass_kg

    @property
    def source_enthalpy_j(self):
        """Existing global silicate+water enthalpy, without passive columns."""
        return (self.primary_enthalpy_j+self.surface_remaining_enthalpy_j
                +self.mantle_enthalpy_j+self.water_enthalpy_j)


def _areas(model):
    areas = model.mesh.physical_cell_areas_km2(model.thermal.radius_km)*1e6
    if (areas.shape != (model.mesh.cell_count,) or not np.isfinite(areas).all()
            or np.any(areas <= 0)
            or not math.isclose(float(areas.sum()), model.thermal.area_m2, rel_tol=2e-12)):
        raise ValueError("Primary material requires a complete positive-area spherical mesh")
    return areas


def primary_material_inventory(model, starter_state):
    """Partition actual thermal mass/enthalpy without changing either owner.

    Molten and partial states are valid inventories, so callers can audit the
    full cooling history. A mature import requiring solid crust must explicitly
    require ``fully_solid_surface``; this flag alone does not certify mechanical
    plate rigidity, a handoff, or complete solidification of the mantle.
    A solid primary layer can be chemically distinct from the mechanical lid;
    differences between the lumped surface reservoir and the passive column
    remain visible in ``mechanical_lid_thickness_km``.
    """
    p = model.thermal
    sample = model.loading.sample(starter_state.thermal_context)
    row = sample.thermal
    areas = _areas(model)
    solid = float(1.-row["surface_melt_fraction"])
    ts, tm = float(row["surface_temperature_k"]), float(row["mantle_temperature_k"])
    surface_mass = areas*p.surface_column_kg_m2
    primary = surface_mass*solid
    remaining = surface_mass-primary
    mantle = areas*p.mantle_column_kg_m2
    thickness = primary/(p.surface_layer_density_kg_m3*areas)/1000.
    liquid = areas*(row["ocean_mass_kg"]/p.area_m2)
    vapor = areas*(row["vapor_mass_kg"]/p.area_m2)
    primary_energy = primary*p.silicate_heat_capacity_j_kg_k*ts
    remaining_energy = remaining*(p.silicate_heat_capacity_j_kg_k*ts+p.silicate_latent_heat_j_kg)
    mantle_energy = areas*starter_state.thermal_context.thermal.energy[0]*ENERGY_SCALE
    water_energy = ((liquid+vapor)*p.water_heat_capacity_j_kg_k*ts
                    +vapor*p.water_latent_heat_j_kg)
    silicate_total = p.area_m2*p.silicate_column_kg_m2
    water_total = row["total_water_mass_kg"]
    thermal_total = p.area_m2*sum(starter_state.thermal_context.thermal.energy[:2])*ENERGY_SCALE
    silicate_residual = (float((primary+remaining+mantle).sum())-silicate_total)/silicate_total
    water_residual = (float((liquid+vapor).sum())-water_total)/max(water_total, 1.)
    energy_residual = (float((primary_energy+remaining_energy+mantle_energy+water_energy).sum())
                       -thermal_total)/max(abs(thermal_total), 1.)
    return PrimaryMaterialInventory(primary, remaining, mantle, thickness, solid,
        liquid, vapor, primary_energy, remaining_energy, mantle_energy, water_energy,
        ts, tm, p.surface_layer_density_kg_m3, sample.lid_thickness_km,
        bool(solid == 1.), silicate_residual, water_residual, energy_residual)


def independent_mantle_omega(model, starter_state):
    """Per-cell angular flow [rad/Myr] from the starter's prescribed mantle.

    Cell tangential speed is the mean vertex traction divided by basal drag,
    exactly as in the starter's candidate Euler fit. omega=cross(r,v)/R is the
    minimum-norm local angular vector whose cross product with R*r gives v.
    This is an independent prescribed field: no domain labels, fitted plate
    rotations, or arbitrary post-split velocity impulses are used.
    """
    parameters = model.parameters
    drag = float(parameters.basal_drag_pa_s_m)
    if not math.isfinite(drag) or drag <= 0:
        raise ValueError("Basal drag must be positive and finite")
    sample = model.loading.sample(starter_state.thermal_context)
    shell = replace(model.shell, seed=parameters.seed)
    traction = mantle_traction(model.mesh, shell,
        np.full(model.mesh.cell_count, sample.lid_thickness_km))
    velocity = traction[model.mesh.faces].mean(axis=1)/drag
    positions = model.mesh.centroids
    velocity -= positions*np.sum(velocity*positions, axis=1)[:, None]
    return np.cross(positions, velocity)*SECONDS_PER_MYR/(model.thermal.radius_km*1000.)
