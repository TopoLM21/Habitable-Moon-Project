"""Continue mechanical support when the young solidus column is exhausted.

The passive column still evolves and keeps its own heat ledger. Once its whole
depth is solid, that depth is a lower bound, not the bottom of the lithosphere.
We join the existing mature half-space cooling approximation at the attained
thickness. This effective mechanical closure does not alter the global heat,
water or orbital reservoirs and does not certify long-term plate dynamics.
"""
from dataclasses import replace
import math

from .genesis import SECONDS_PER_MYR
from .lithosphere import oceanic_thermal_lithosphere_total_thickness_km


def begin_mechanical_transition(sample, model, mechanical_config):
    depth = model.shell.column_depth_km
    if not math.isclose(sample.lid_thickness_km, depth, rel_tol=0., abs_tol=1e-8):
        raise ValueError("Young thermal support limit flag disagrees with the column depth")
    diffusivity = float(mechanical_config.get("thermal_diffusivity_m2_s", 1e-6))
    coefficient = float(mechanical_config.get("cooling_coefficient", 2.))
    cap = float(mechanical_config.get("oceanic_max_total_thickness_km", 155.))
    if (not all(math.isfinite(x) for x in (diffusivity, coefficient, cap))
            or diffusivity <= 0 or coefficient <= 0 or cap < depth):
        raise ValueError("Mature cooling parameters cannot continue the attained young shell thickness")
    row = sample.thermal
    contrast = row["mantle_temperature_k"] - row["surface_temperature_k"]
    fraction = ((sample.mean_lid_temperature_k-row["surface_temperature_k"])/contrast
                if contrast > 0 else .5)
    return {"law": "matched_mature_half_space_v1", "time_myr": sample.time_myr,
            "initial_thickness_km": depth,
            "equivalent_cooling_age_myr": (depth*1000/coefficient)**2/diffusivity/SECONDS_PER_MYR,
            "thermal_diffusivity_m2_s": diffusivity, "cooling_coefficient": coefficient,
            "max_total_thickness_km": cap,
            "mean_temperature_fraction": max(0., min(1., fraction))}


def mechanical_sample(sample, transition):
    if transition is None or sample.time_myr < transition["time_myr"]:
        return sample
    elapsed = max(0., sample.time_myr-transition["time_myr"])
    thickness = float(oceanic_thermal_lithosphere_total_thickness_km(
        transition["equivalent_cooling_age_myr"]+elapsed,
        thermal_diffusivity_m2_s=transition["thermal_diffusivity_m2_s"],
        cooling_coefficient=transition["cooling_coefficient"],
        max_total_thickness_km=transition["max_total_thickness_km"]))
    row = sample.thermal
    mean = row["surface_temperature_k"]+transition["mean_temperature_fraction"]*(
        row["mantle_temperature_k"]-row["surface_temperature_k"])
    return replace(sample, lid_thickness_km=thickness, mean_lid_temperature_k=mean,
                   column_depth_limit_reached=False)


def refresh_matched_mechanics(state, dt_myr, age_cap, previous_contrast_k, **options):
    """Mature local targets with time-consistent relaxation of mixed roots.

    Oceanic cooling responds to material age. Only the residual continental
    contribution relaxes; a zero-time refresh cannot grow a mixed root. The
    existing mature target laws, including rift thinning and cratons, are used.
    """
    from copy import copy
    import numpy as np
    from .lithosphere import target_mantle_lithosphere_fields

    dt = max(float(dt_myr), 0.)
    tau = max(float(options.pop("continental_relaxation_myr", 250.)), 1e-9)
    view = copy(state)
    view.crust_age_myr = np.minimum(state.crust_age_myr, age_cap)
    target_h, target_rho = target_mantle_lithosphere_fields(view, **options)
    if state.mantle_lithosphere_thickness_km is None or state.mantle_lithosphere_density_anomaly_kg_m3 is None:
        state.mantle_lithosphere_thickness_km = target_h
        state.mantle_lithosphere_density_anomaly_kg_m3 = target_rho
        return state
    fraction = (np.clip(state.continental_fraction, 0., 1.)
                if state.continental_fraction is not None else (state.crust_type == 1).astype(float))
    if dt == 0.:
        state.mantle_lithosphere_thickness_km = np.where(fraction == 0., target_h, state.mantle_lithosphere_thickness_km)
        state.mantle_lithosphere_density_anomaly_kg_m3 = np.where(fraction == 0., target_rho, state.mantle_lithosphere_density_anomaly_kg_m3)
        return state
    view.continental_fraction = np.zeros(len(state.crust_age_myr))
    ocean_h, ocean_rho = target_mantle_lithosphere_fields(view, **options)
    view.crust_age_myr = np.maximum(0., view.crust_age_myr-dt)
    previous_options = dict(options, mantle_temperature_contrast_k=previous_contrast_k)
    old_ocean_h, old_ocean_rho = target_mantle_lithosphere_fields(view, **previous_options)
    ocean_delta_h = (1.-fraction)*(ocean_h-old_ocean_h)
    ocean_delta_rho = ((1.-fraction)*(ocean_rho-old_ocean_rho) if dt > 0 else 0.)
    decay = math.exp(-dt/tau)
    mixed_h = target_h+(state.mantle_lithosphere_thickness_km+ocean_delta_h-target_h)*decay
    mixed_rho = target_rho+(state.mantle_lithosphere_density_anomaly_kg_m3+ocean_delta_rho-target_rho)*decay
    state.mantle_lithosphere_thickness_km = np.where(fraction == 0., target_h, mixed_h)
    state.mantle_lithosphere_density_anomaly_kg_m3 = np.where(fraction == 0., target_rho, mixed_rho)
    return state
