"""Passive age-resolved cooling for the experimental fractional ocean surface.

Genesis remains the owner of heat and its passive reference column. This
adapter evaluates the existing young mechanical closure for each material
history, without a mean-age raster or an additional heat sink. Cold mantle
volume and density excess are mechanical state, so their thermal changes are
reported separately from the conservative transport transaction.
"""
from __future__ import annotations

from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np

from .fractional_surface import EXTENSIVE_FIELDS, FractionalSurfaceState, SurfaceParcel
from .genesis_local_mechanics import refresh_young_material_mechanics


MODEL_VERSION = "fractional-young-thermal-1"


def refresh_fractional_mechanics(state, sample, model, *, origin_time_myr,
                                 thermal_diffusivity_m2_s=1e-6):
    """Return a new sparse state and a JSON-safe passive thermal source ledger.

    The state and sample must have the same clock. The inherited intrinsic
    basalt thickness, rather than a repeatedly divided extensive volume, is
    passed to the unchanged 0.5 young thermal closure. Primordial material
    retains the reference column under that closure's existing age criterion.

    Area, chemical basalt volume, age, ownership, lineage and damage do not
    change. Signed changes of cold volume and excess mass are caused by this
    mechanical cooling refresh; they are not transport production or global
    thermal energy withdrawals. Only the finite, unexhausted young column is
    supported. A matched mature closure must be implemented explicitly before
    allowing that transition in the fractional experiment.
    """
    if not isinstance(state, FractionalSurfaceState):
        raise ValueError("Fractional cooling requires a FractionalSurfaceState")
    parcels, diagnostics = refresh_parcel_mechanics(state.parcels, state.time_myr,
        sample, model, origin_time_myr=origin_time_myr,
        thermal_diffusivity_m2_s=thermal_diffusivity_m2_s)
    return replace(state, parcels=parcels), diagnostics


def refresh_parcel_mechanics(parcels, time_myr, sample, model, *, origin_time_myr,
                             thermal_diffusivity_m2_s=1e-6):
    """Refresh an arbitrary nonempty set of parcels at one physical clock.

    This applies the same passive young thermal closure as a complete surface
    without requiring its input to tile any cells. In particular, a removed
    fraction may be evaluated at its removal time without constructing an
    artificial full surface or changing its area, lineage, age or damage.
    Distinct age histories remain distinct. The caller owns event timing and
    includes the returned signed source delta in its material ledger.

    Empty groups are rejected explicitly; callers with no event parcels skip
    the refresh. The input sequence and thermal context are never mutated.
    """
    original = tuple(parcels)
    if not original or any(not isinstance(p, SurfaceParcel) for p in original):
        raise ValueError("Fractional cooling requires a nonempty sequence of SurfaceParcel values")
    clock = float(time_myr)
    now = float(sample.time_myr)
    if (not math.isfinite(clock) or not math.isfinite(now) or not math.isclose(now, clock,
            rel_tol=0., abs_tol=64.*math.ulp(max(1., abs(clock))))):
        raise ValueError("Fractional surface and thermal sample clocks disagree")
    depth, lid = float(model.shell.column_depth_km), float(sample.lid_thickness_km)
    mean = float(sample.mean_lid_temperature_k)
    surface = float(sample.thermal["surface_temperature_k"])
    mantle = float(sample.thermal["mantle_temperature_k"])
    column = np.asarray(sample.state.column_enthalpy, dtype=float)
    if (not all(math.isfinite(v) for v in (depth, lid, mean, surface, mantle))
            or depth <= 0. or lid < 0. or min(mean, surface, mantle) <= 0.
            or column.shape != (model.shell.column_layers,)
            or not np.isfinite(column).all() or np.any(column <= 0.)):
        raise ValueError("Fractional cooling requires a finite positive thermal column")
    if sample.column_depth_limit_reached or lid >= depth*(1.-1e-12):
        raise ValueError("Fractional cooling does not yet support an exhausted young column")

    view = SimpleNamespace(time_myr=clock,
        crust_age_myr=np.asarray([p.age_myr for p in original]),
        crust_thickness_km=np.asarray([p.specific_properties[0] for p in original]))
    local = refresh_young_material_mechanics(view, sample, model,
        origin_time_myr=origin_time_myr,
        thermal_diffusivity_m2_s=thermal_diffusivity_m2_s)
    parcels = []
    for parcel, thickness, density in zip(original,
            view.mantle_lithosphere_thickness_km,
            view.mantle_lithosphere_density_anomaly_kg_m3):
        thickness, density = float(thickness), float(density)
        volume = parcel.area_km2*thickness
        mass = volume*density*1e9
        # Replace the two derived thermal specific properties. Retain the
        # inherited chemical thickness exactly so nested splits cannot change
        # its last bit and prevent otherwise identical histories from merging.
        specific = (parcel.specific_properties[0], thickness, thickness*density*1e9)
        parcels.append(replace(parcel, cold_mantle_volume_km3=volume,
            density_excess_mass_kg=mass, specific_properties=specific))
    refreshed = tuple(parcels)
    before, after = ({name: math.fsum(getattr(p, name) for p in sequence)
                     for name in EXTENSIVE_FIELDS} for sequence in (original, refreshed))
    source = {name: math.fsum(getattr(new, name)-getattr(old, name)
                            for old, new in zip(original, refreshed))
              for name in ("cold_mantle_volume_km3", "density_excess_mass_kg")}
    total_lid = local["local_total_lid_thickness_km"]
    diagnostics = {
        "model_version": MODEL_VERSION,
        "thermal_closure_version": local["model_version"],
        "time_myr": clock,
        "origin_time_myr": float(origin_time_myr),
        "thermal_diffusivity_m2_s": float(thermal_diffusivity_m2_s),
        "primordial_parcel_count": local["primordial_cell_count"],
        "rejuvenated_parcel_count": local["rejuvenated_cell_count"],
        "before": before,
        "after": after,
        "thermal_source_delta": source,
        "thermal_source_ledger_residual": {
            name: math.fsum((before[name], source[name], -after[name])) for name in source},
        # Aligned with the returned parcels, including a cold layer that ends
        # within chemical basalt. Cold mantle plus full crust cannot recover
        # this depth and must not be substituted by force consumers.
        "local_total_lid_thickness_km": total_lid.tolist(),
        "minimum_total_lid_thickness_km": float(np.min(total_lid)),
        "maximum_total_lid_thickness_km": float(np.max(total_lid)),
        "rejuvenated_heat_content_deficit_j": math.fsum(
            p.area_km2*1e6*float(deficit) for p, deficit in zip(original,
                local["rejuvenated_heat_content_deficit_j_m2"])),
        "thermal_owner": local["thermal_owner"],
        "boundary_temperature_approximation": local["boundary_temperature_approximation"],
        "source_interpretation": "Signed passive mechanical cooling change; not transport creation",
        "additional_global_heat_sink_j": 0.,
    }
    return refreshed, diagnostics


__all__ = ["MODEL_VERSION", "refresh_fractional_mechanics", "refresh_parcel_mechanics"]
