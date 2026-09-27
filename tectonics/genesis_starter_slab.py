"""Finite slab development for the experimental young-world continuation.

The mature surface force assumes an established slab. A young domain must
first accumulate one through the existing convergence-integrated subduction
memory. Here the existing slab-length cap normalizes that stored length: this
is an explicit model closure, not a new calibrated slab-initiation law.

The context is deliberately confined to one dynamics call in the dedicated
continuation process. Ordinary mature runs keep their existing calibration.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import copy
import math

import numpy as np

from .subduction_memory import SlabZone, SubductionMemoryParameters, SubductionMemoryState


def slab_development_fraction(zone: SlabZone | None, reference_length_km: float) -> float:
    """Zero for no attached slab; otherwise its bounded developed-length ratio."""
    if not math.isfinite(reference_length_km) or reference_length_km <= 0.0:
        raise ValueError("Slab reference length must be finite and positive")
    if zone is None or zone.broken_off:
        return 0.0
    if not math.isfinite(zone.slab_length_km) or zone.slab_length_km < 0.0:
        raise ValueError("Stored slab length must be finite and nonnegative")
    return min(zone.slab_length_km / reference_length_km, 1.0)


def young_slab_pull_multiplier(
    memory: SubductionMemoryState | None,
    subducting_plate: int,
    overriding_plate: int,
    parameters: SubductionMemoryParameters,
) -> float:
    """Resolve an oriented pair without inventing a slab on first contact."""
    zone = None if memory is None or not parameters.enabled else memory.zones.get(
        (int(subducting_plate), int(overriding_plate)))
    return slab_development_fraction(zone, parameters.slab_length_cap_km)


def _developed_residual_pull(original, memory, plate_count, parameters, weight, gain):
    """Retain mature decay/breakoff rules, scaling each residual slab once.

    The original calculation is evaluated one zone at a time so both force
    and diagnostic fractions can be scaled before its trench-length average.
    This also preserves normalization when a zero-length residual coexists
    with a developed slab. The checkpoint memory is never mutated.
    """
    vec = np.zeros((int(plate_count), 3), dtype=np.float64)
    fraction = np.zeros(int(plate_count), dtype=np.float64)
    lengths = np.zeros(int(plate_count), dtype=np.float64)
    if memory is None or not parameters.enabled or parameters.residual_pull_gain <= 0.0:
        return vec, fraction
    one = copy(memory)
    for key in sorted(memory.zones):
        zone = memory.zones[key]
        sub = int(zone.subducting_plate)
        if (zone.active or zone.broken_off or zone.inactive_age_myr < 0.0
                or sub < 0 or sub >= int(plate_count)):
            continue
        developed = slab_development_fraction(zone, parameters.slab_length_cap_km)
        one.zones = {key: zone}
        local, local_fraction = original(one, plate_count, parameters, weight, gain)
        length = max(float(zone.trench_length_km), 1e-9)
        vec[sub] += length * developed * local[sub]
        fraction[sub] += length * developed * local_fraction[sub]
        lengths[sub] += length
    valid = lengths > 0.0
    vec[valid] /= lengths[valid, None]
    fraction[valid] /= lengths[valid]
    return vec, fraction


@contextmanager
def young_slab_pull(parameters: SubductionMemoryParameters):
    """Apply finite development to active and residual pull for one call.

    Both CPU boundary-force implementations consult these same dynamics
    aliases. Their previous functions are restored even if dynamics raises.
    Call only in the existing serial, dedicated continuation worker.
    """
    from . import dynamics

    slab_development_fraction(None, parameters.slab_length_cap_km)
    old_multiplier = dynamics.slab_pull_multiplier_for_pair
    old_residual = dynamics.residual_pull_by_plate

    def multiplier(memory, sub, over):
        return old_multiplier(memory, sub, over) * young_slab_pull_multiplier(
            memory, sub, over, parameters)

    def residual(memory, plate_count, actual_parameters, weight, gain):
        # The runtime parameters also own residual decay and enabled state.
        return _developed_residual_pull(old_residual, memory, plate_count,
                                        actual_parameters, weight, gain)

    dynamics.slab_pull_multiplier_for_pair = multiplier
    dynamics.residual_pull_by_plate = residual
    try:
        yield
    finally:
        dynamics.slab_pull_multiplier_for_pair = old_multiplier
        dynamics.residual_pull_by_plate = old_residual


__all__ = ["slab_development_fraction", "young_slab_pull_multiplier", "young_slab_pull"]
