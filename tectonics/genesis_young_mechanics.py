"""Versioned prescribed loading for Genesis-origin continuations.

The stored mantle field represents uncoupled prescribed traction divided by
the configured linear drag, not a solved free-convection velocity. Transmission
through the current lid is evaluated separately. Old saves retain their legacy
attenuated, thermally modulated field until rebuilt from the original Starter.
"""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np

from .basal_coupling import prescribed_cell_basal_state, velocity_to_local_omega
from .genesis_starter_material import mantle_source_parameters
from .mantle import MantleFlowState, MantleFlowDiagnostics, mantle_flow_rms_rad_per_myr

MECHANICS_MODEL_VERSION = "young-mechanics-0.5"
UNIFORM_SINKING_MECHANICS = "young-mechanics-0.4"
BASAL_RIDGE_MECHANICS = "young-mechanics-0.3"
LEGACY_MECHANICS = "legacy-young-0.2"


def mechanics_version(config):
    version = config.get("young_shell", {}).get("mechanics_model_version", LEGACY_MECHANICS)
    if version not in (MECHANICS_MODEL_VERSION, UNIFORM_SINKING_MECHANICS,
                       BASAL_RIDGE_MECHANICS, LEGACY_MECHANICS):
        raise ValueError(f"Unsupported young mechanics model: {version}")
    return version


def corrected_mechanics(config):
    return mechanics_version(config) != LEGACY_MECHANICS


def sinking_mechanics(config):
    """Both saved sinking versions use endpoint clocks and a quasistatic law."""
    return mechanics_version(config) in (MECHANICS_MODEL_VERSION, UNIFORM_SINKING_MECHANICS)


def transmitted_mantle_flow(model, source_flow, thickness_km):
    """Current prescribed traction / beta, with the original vertex quadrature.

    Re-evaluate transmission before averaging: mean(c*tau) is not generally
    mean(c)*mean(tau). Neither stored source arrays nor the material are edited.
    """
    interaction = prescribed_cell_basal_state(model.mesh, mantle_source_parameters(model), thickness_km)
    positions = model.mesh.centroids
    omega = velocity_to_local_omega(positions, interaction.equilibrium_velocity_m_s, model.thermal.radius_km)
    rms = float(np.sqrt(np.mean(np.sum(omega * omega, axis=1))))
    return MantleFlowState(source_flow.time_myr, omega, rms)


def advance_prescribed_source(mesh, state, dt_myr, tectonic_activity_factor, params=None):
    """Advance the clock of a prescribed source, without evolving its forcing.

    A constant configured traction is also the Starter/damage assumption.
    Applying the mature flux-ratio amplitude law here would make two different
    forcings and reintroduce partition-time dependence. No thermal energy is
    read or modified by this mechanical operation.
    """
    if not math.isfinite(dt_myr) or dt_myr <= 0:
        raise ValueError("dt_myr must be positive and finite")
    field = np.asarray(state.cell_omega_rad_per_myr)
    if field.shape != (mesh.cell_count, 3) or not np.isfinite(field).all():
        raise ValueError("Prescribed mantle source does not match mesh")
    result = replace(state, time_myr=state.time_myr + dt_myr,
                     cell_omega_rad_per_myr=field.copy())
    rms = mantle_flow_rms_rad_per_myr(result)
    fraction = rms / state.formation_rms_rad_per_myr if state.formation_rms_rad_per_myr > 0 else 0.
    return result, MantleFlowDiagnostics(result.time_myr, float(np.rad2deg(rms)), 1., fraction)
