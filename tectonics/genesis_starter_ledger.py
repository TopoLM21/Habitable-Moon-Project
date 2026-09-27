"""Independent reference-material ledger around the existing mature engine.

The mature model conserves several separate volume reservoirs. This audit joins
their *recorded process counters*, then compares the predicted change against
the independently measured surface reservoirs. It never balances an error by
defining mantle transfer as the observed surface difference.

All volumes use one declared reference density, so this is an equivalent-rock
mass budget. Density contrasts used for buoyancy, porosity, and plume support
are not physical species-mass accounting. Thermal enthalpy remains exclusively
owned by the genesis thermal context; no heat is created by this material audit.
"""
from __future__ import annotations

import math
from numbers import Real

import numpy as np


def _nonnegative(value, label):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(value) or value < 0.0):
        raise ValueError(f"{label} must be a finite nonnegative number")
    return float(value)


def _array_total(values, count, label, *, allow_missing=False):
    if values is None and allow_missing:
        return 0.0
    array = np.asarray(values, dtype=np.float64)
    if (array.shape != (count,) or not np.isfinite(array).all() or np.any(array < 0.0)):
        raise ValueError(f"{label} must be a finite nonnegative per-cell volume field")
    total = float(np.sum(array))
    if not math.isfinite(total):
        raise ValueError(f"{label} total overflows")
    return total


def _upper_components(checkpoint, label):
    state = checkpoint.state
    owner = np.asarray(state.cell_plate)
    if owner.ndim != 1 or not len(owner):
        raise ValueError(f"{label} must contain nonempty cell ownership")
    count = len(owner)
    result = {
        "oceanic_bedrock": _array_total(state.oceanic_volume_km3, count, f"{label}.oceanic"),
        "continental_bedrock": _array_total(state.continental_volume_km3, count, f"{label}.continental"),
        "surface_sediment": _array_total(state.sediment_volume_km3, count, f"{label}.sediment", allow_missing=True),
    }
    magmatic = checkpoint.plume_magmatism_state
    for reservoir in ("extrusive", "dyke", "underplate"):
        result[f"plume_{reservoir}"] = (0.0 if magmatic is None else _array_total(
            getattr(magmatic, f"{reservoir}_volume_km3"), count, f"{label}.plume_{reservoir}"))
    if checkpoint.sediment_budget is None and result["surface_sediment"] != 0.0:
        raise ValueError(f"{label} has sediment without its process budget")
    return count, result


def _counter_delta(initial, final, field, label):
    first = 0.0 if initial is None else _nonnegative(getattr(initial, field), f"initial.{label}.{field}")
    last = 0.0 if final is None else _nonnegative(getattr(final, field), f"final.{label}.{field}")
    if last < first:
        raise ValueError(f"{label}.{field} decreased or its history was discarded")
    return last - first


def _new_rows(initial_cp, final_cp, name, first_time, last_time):
    initial = getattr(initial_cp, name)
    final = getattr(final_cp, name)
    if not isinstance(initial, list) or not isinstance(final, list):
        raise ValueError(f"{name} must be a list of process records")
    if len(final) < len(initial) or final[:len(initial)] != initial:
        raise ValueError(f"{name} must preserve the initial history as an exact prefix")
    result = final[len(initial):]
    previous = first_time
    for row in result:
        if not isinstance(row, dict) or "time_myr" not in row:
            raise ValueError(f"{name} process record has no time")
        time = _nonnegative(row["time_myr"], f"{name}.time_myr")
        if time <= previous or time > last_time:
            raise ValueError(f"{name} process records do not follow the checkpoint interval")
        previous = time
    return result


def _recorded_sum(rows, field, label):
    values = []
    for row in rows:
        if field not in row:
            raise ValueError(f"{label} lacks required material counter {field}")
        values.append(_nonnegative(row[field], f"{label}.{field}"))
    return math.fsum(values)


def continuation_material_ledger(initial_cp, final_cp, density_kg_m3=3000.0) -> dict:
    """Audit mature-process volume exchange and return its mantle mass transfer.

    Both checkpoints must retain independent basalt and continental material
    fields. New lithosphere and continental rows are selected by the length of
    the initial append-only history, so resumed runs do not count old exchanges
    again. Cumulative continental, sediment and igneous counters are differenced.

    A positive ``mantle_reference_mass_change_kg`` means material returned to
    the interior. A negative value means extraction into the upper reservoirs.
    The caller must apply that transfer to a separately bounded interior mass
    reservoir. ``balanced`` audits this predicted exchange against material
    arrays; it does not assert adequate interior mass or thermal closure.
    """
    density = _nonnegative(density_kg_m3, "density_kg_m3")
    if density == 0.0:
        raise ValueError("density_kg_m3 must be positive")
    first_time = _nonnegative(initial_cp.state.time_myr, "initial.time_myr")
    last_time = _nonnegative(final_cp.state.time_myr, "final.time_myr")
    if last_time < first_time:
        raise ValueError("Final checkpoint precedes the initial checkpoint")
    initial_count, initial_upper = _upper_components(initial_cp, "initial")
    final_count, final_upper = _upper_components(final_cp, "final")
    if initial_count != final_count:
        raise ValueError("Material audit requires the same canonical cell count")
    lithosphere = _new_rows(initial_cp, final_cp, "lithosphere_rows", first_time, last_time)
    continental = _new_rows(initial_cp, final_cp, "cycle_rows", first_time, last_time)
    if initial_cp.cycle is None or final_cp.cycle is None:
        raise ValueError("Material audit requires the continental process budget")

    sources = {
        "lithosphere_oceanic_creation": _recorded_sum(lithosphere, "oceanic_created_volume_km3", "lithosphere"),
        "continental_cycle_oceanic_creation": _recorded_sum(continental, "oceanic_generated_volume_km3", "continental cycle"),
        "continental_generation": _counter_delta(initial_cp.cycle, final_cp.cycle,
            "cumulative_generated_volume_km3", "continental cycle"),
    }
    sinks = {
        "oceanic_subduction": _recorded_sum(lithosphere, "oceanic_subducted_volume_km3", "lithosphere"),
        "oceanic_rift_recycling": _recorded_sum(lithosphere, "oceanic_rift_recycled_volume_km3", "lithosphere"),
        "continental_rift_recycling": _recorded_sum(lithosphere, "rift_recycled_volume_km3", "lithosphere"),
        "continental_cycle_oceanic_recycling": _recorded_sum(continental, "oceanic_recycled_volume_km3", "continental cycle"),
        "oceanic_replacement_by_juvenile_crust": _recorded_sum(continental,
            "oceanic_replaced_by_juvenile_volume_km3", "continental cycle"),
        "continental_cycle_recycling": _counter_delta(initial_cp.cycle, final_cp.cycle,
            "cumulative_recycled_volume_km3", "continental cycle"),
        "deep_sediment_recycling": _counter_delta(initial_cp.sediment_budget, final_cp.sediment_budget,
            "deep_recycled_sediment_volume_km3", "sediment budget"),
    }
    for reservoir in ("extrusive", "dyke", "underplate"):
        sources[f"plume_{reservoir}_generation"] = _counter_delta(
            initial_cp.plume_magmatism_state, final_cp.plume_magmatism_state,
            f"cumulative_generated_{reservoir}_volume_km3", "plume magmatism")
        sinks[f"plume_{reservoir}_recycling"] = _counter_delta(
            initial_cp.plume_magmatism_state, final_cp.plume_magmatism_state,
            f"deep_recycled_{reservoir}_volume_km3", "plume magmatism")

    initial_volume = math.fsum(initial_upper.values())
    final_volume = math.fsum(final_upper.values())
    generated, recycled = math.fsum(sources.values()), math.fsum(sinks.values())
    predicted = generated - recycled
    actual = final_volume - initial_volume
    residual = actual - predicted
    scale = max(initial_volume, final_volume, generated + recycled, 1.0)
    tolerance = float(max(1e-6, 128 * np.finfo(float).eps * scale))
    conversion = density * 1e9
    numbers = (initial_volume, final_volume, generated, recycled, predicted, actual,
               residual, conversion, initial_volume * conversion, final_volume * conversion)
    if not all(math.isfinite(value) for value in numbers):
        raise ValueError("Reference-material ledger overflows")
    return {
        "format": "genesis-starter-material-ledger-0.1",
        "initial_time_myr": first_time, "final_time_myr": last_time,
        "reference_density_kg_m3": density,
        "new_lithosphere_records": len(lithosphere), "new_continental_records": len(continental),
        "initial_upper_components_km3": initial_upper,
        "final_upper_components_km3": final_upper,
        "source_components_km3": sources, "sink_components_km3": sinks,
        "generated_volume_km3": generated, "recycled_volume_km3": recycled,
        "initial_upper_volume_km3": initial_volume, "final_upper_volume_km3": final_volume,
        "observed_upper_change_km3": actual, "predicted_upper_change_km3": predicted,
        "volume_residual_km3": residual, "volume_tolerance_km3": tolerance,
        "relative_volume_residual": residual / scale,
        "initial_upper_reference_mass_kg": initial_volume * conversion,
        "final_upper_reference_mass_kg": final_volume * conversion,
        "mantle_reference_mass_change_kg": -predicted * conversion,
        "reference_mass_residual_kg": residual * conversion,
        "balanced": bool(abs(residual) <= tolerance),
        "density_convention": "One reference density for equivalent-rock volume; not a species-density mass budget.",
        "thermal_energy_changed_by_audit_j": 0.0,
    }


__all__ = ["continuation_material_ledger"]
