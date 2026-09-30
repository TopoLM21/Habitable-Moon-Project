"""Passive material-age cooling before the primordial column is exhausted.

The Genesis reservoirs and reference enthalpy column remain authoritative.
This is a quasi-static local mechanical closure, not another global heat sink:
newborn oceanic material cools by a half-space temperature profile evaluated at
the current boundary temperatures. The reference column supplies its warm-side
bound. Thus local cooling cannot make a parcel colder than primordial material.
Crust age already follows the material donor and is saved/remeshed; no hidden
per-process clock or new energy reservoir is introduced here.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.special import erf, erfinv

from .genesis import SECONDS_PER_MYR
from .genesis_shell import rock_temperature, rock_enthalpy


MECHANICS_MODEL_VERSION = "young-mechanics-0.3"
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(8)


def _profile_integrals(age_myr, lid_km, depth_km, reference_z, reference_t,
                       surface_k, mantle_k, diffusivity, thermal):
    """Temperature over the lid and heat-content deficit over the reference.

    Quadrature is split at every reference-column interface and the lid base.
    The heat-content number is diagnostic only; it is not a flux or a ledger
    of heat exported to the globally owned reservoirs.
    """
    if age_myr == 0.:
        return mantle_k, 0.
    length_km = 2.*math.sqrt(diffusivity*age_myr*SECONDS_PER_MYR)/1000.
    edges = np.unique(np.r_[reference_z, lid_km])
    edges = edges[(edges >= 0.) & (edges <= depth_km)]
    centers = .5*(edges[:-1]+edges[1:])
    widths = .5*np.diff(edges)
    z = centers[:, None]+widths[:, None]*_NODES
    weight = widths[:, None]*_WEIGHTS
    reference = np.interp(z, reference_z, reference_t)
    cooling = surface_k+(mantle_k-surface_k)*erf(z/length_km)
    local = np.maximum(reference, cooling)
    lid_segments = edges[1:] <= lid_km
    mean = float(np.sum(local[lid_segments]*weight[lid_segments])/lid_km) if lid_km > 0 else mantle_k
    deficit_j_kg_km = float(np.sum((rock_enthalpy(mantle_k, thermal)
        -rock_enthalpy(local, thermal))*weight))
    return mean, max(deficit_j_kg_km, 0.)


def refresh_young_material_mechanics(state, sample, model, *, origin_time_myr,
                                     thermal_diffusivity_m2_s=1e-6):
    """Set local young oceanic support and return passive diagnostics.

    Primordial cells are identified by their original, unreset material age.
    They retain the reference column exactly, including at the initial handoff.
    Younger cells retain their transported age; zero-time repeated calls and
    checkpoint/resume therefore give identical mechanical fields.

    The local mantle density follows the same whole-lid mean-temperature
    convention as the existing Genesis adapter. This does not replace its
    coarse mantle/crust thermal partition with a resolved petrological model.
    """
    origin = float(origin_time_myr)
    kappa = float(thermal_diffusivity_m2_s)
    now = float(state.time_myr)
    if (not all(math.isfinite(v) for v in (origin, now, kappa)) or origin < 0.
            or now < origin-1e-9 or kappa <= 0.):
        raise ValueError("Local young mechanics needs a valid origin, clock, and diffusivity")
    age = np.asarray(state.crust_age_myr, dtype=float)
    crust = np.asarray(state.crust_thickness_km, dtype=float)
    if age.shape != crust.shape or not np.isfinite(age).all() or np.any(age < 0.):
        raise ValueError("Local young mechanics needs finite nonnegative material ages")
    elapsed = max(now-origin, 0.)
    # The tolerance covers floating-point sums of accepted dt, not a physical
    # minimum birth interval or a tunable rheological threshold.
    primordial = age >= elapsed-64.*np.finfo(float).eps*max(1., abs(now))
    row = sample.thermal
    surface, mantle = float(row["surface_temperature_k"]), float(row["mantle_temperature_k"])
    ref_h = float(sample.lid_thickness_km)
    total = np.full(age.shape, ref_h)
    mean = np.full(age.shape, float(sample.mean_lid_temperature_k))
    heat_deficit = np.zeros(age.shape)
    local = ~primordial
    reference_z = np.r_[0., (np.arange(model.shell.column_layers)+.5)
                        *model.shell.column_depth_km/model.shell.column_layers,
                        model.shell.column_depth_km]
    reference_t = np.r_[surface, rock_temperature(sample.state.column_enthalpy, model.thermal), mantle]
    contrast = max(mantle-surface, 0.)
    # Actual solidus crossing of the half-space profile, rather than a tuned
    # multiple of sqrt(age) or the finite reference-column depth as a switch.
    solid_fraction = ((model.thermal.solidus_k-surface)/contrast if contrast else 0.)
    inverse = float(erfinv(np.clip(solid_fraction, 0., 1.-np.finfo(float).eps)))
    for value in np.unique(age[local]):
        selected = local & (age == value)
        thickness = min(ref_h, 2.*math.sqrt(kappa*float(value)*SECONDS_PER_MYR)/1000.*inverse)
        local_mean, deficit = _profile_integrals(float(value), thickness,
            model.shell.column_depth_km, reference_z, reference_t, surface, mantle,
            kappa, model.thermal)
        total[selected], mean[selected] = thickness, local_mean
        heat_deficit[selected] = deficit*1000.*model.shell.density_kg_m3
    thickness = np.maximum(total-np.maximum(crust, 0.), 0.)
    density = model.shell.density_kg_m3*3e-5*np.maximum(mantle-mean, 0.)
    # A newborn column has no finite cold mantle support or cold mantle mass.
    density[local & (total == 0.)] = 0.
    state.mantle_lithosphere_thickness_km = thickness
    state.mantle_lithosphere_density_anomaly_kg_m3 = density
    return {"model_version": MECHANICS_MODEL_VERSION,
        "primordial_cell_count": int(primordial.sum()), "rejuvenated_cell_count": int(local.sum()),
        "local_total_lid_thickness_km": total,
        "local_mean_lid_temperature_k": mean,
        "rejuvenated_heat_content_deficit_j_m2": heat_deficit,
        "thermal_owner": "Genesis; local deficit is diagnostic, not an additional global heat sink",
        "boundary_temperature_approximation": "quasi-static current Ts/Tm; transported material cooling age"}
