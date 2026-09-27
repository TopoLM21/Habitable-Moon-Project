from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tectonics.genesis_starter_ledger import continuation_material_ledger


def lithosphere_row(time, **changes):
    row = dict(time_myr=time, oceanic_created_volume_km3=0.0,
               oceanic_subducted_volume_km3=0.0, oceanic_rift_recycled_volume_km3=0.0,
               rift_recycled_volume_km3=0.0)
    row.update(changes)
    return row


def cycle_row(time, **changes):
    row = dict(time_myr=time, oceanic_generated_volume_km3=0.0,
               oceanic_recycled_volume_km3=0.0, oceanic_replaced_by_juvenile_volume_km3=0.0)
    row.update(changes)
    return row


@pytest.fixture
def interval():
    # Nonzero baseline counters and retained old records exercise continuation
    # accounting instead of relying on a special all-zero initial state.
    first = SimpleNamespace(
        state=SimpleNamespace(time_myr=1.0, cell_plate=np.zeros(4, dtype=np.int32),
            oceanic_volume_km3=np.full(4, 250.0), continental_volume_km3=np.full(4, 50.0),
            sediment_volume_km3=np.full(4, 10.0)),
        cycle=SimpleNamespace(cumulative_generated_volume_km3=300.0, cumulative_recycled_volume_km3=10.0),
        sediment_budget=SimpleNamespace(deep_recycled_sediment_volume_km3=20.0,
            cumulative_rift_recycled_volume_km3=2.0, cumulative_eroded_bedrock_volume_km3=60.0),
        plume_magmatism_state=SimpleNamespace(
            extrusive_volume_km3=np.full(4, 3.0), dyke_volume_km3=np.full(4, 2.0),
            underplate_volume_km3=np.full(4, 5.0),
            cumulative_generated_extrusive_volume_km3=20.0,
            cumulative_generated_dyke_volume_km3=12.0,
            cumulative_generated_underplate_volume_km3=40.0,
            deep_recycled_extrusive_volume_km3=8.0,
            deep_recycled_dyke_volume_km3=4.0,
            deep_recycled_underplate_volume_km3=20.0),
        lithosphere_rows=[lithosphere_row(1.0, oceanic_created_volume_km3=1000.0)],
        cycle_rows=[cycle_row(1.0, oceanic_generated_volume_km3=500.0)],
    )
    last = deepcopy(first)
    last.state.time_myr = 2.0
    last.lithosphere_rows.append(lithosphere_row(2.0))
    last.cycle_rows.append(cycle_row(2.0))
    return first, last


def test_created_and_subducted_basalt_are_independent_signed_mantle_exchanges(interval):
    first, last = interval
    last.lithosphere_rows[-1].update(oceanic_created_volume_km3=40.0, oceanic_subducted_volume_km3=10.0)
    last.state.oceanic_volume_km3[0] += 30.0
    result = continuation_material_ledger(first, last)
    assert result["balanced"]
    assert result["generated_volume_km3"] == 40.0
    assert result["recycled_volume_km3"] == 10.0
    assert result["predicted_upper_change_km3"] == 30.0
    assert result["mantle_reference_mass_change_kg"] == -30.0 * 3000.0 * 1e9
    assert result["volume_residual_km3"] == 0.0
    assert result["thermal_energy_changed_by_audit_j"] == 0.0


@pytest.mark.parametrize("field,material,sign", [
    ("oceanic_created_volume_km3", "oceanic", 1),
    ("oceanic_subducted_volume_km3", "oceanic", -1),
    ("oceanic_rift_recycled_volume_km3", "oceanic", -1),
    ("rift_recycled_volume_km3", "continental", -1),
])
def test_individual_lithosphere_process_counters(interval, field, material, sign):
    first, last = interval
    last.lithosphere_rows[-1][field] = 3.0
    getattr(last.state, material + "_volume_km3")[0] += sign * 3.0
    report = continuation_material_ledger(first, last)
    assert report["balanced"] and report["predicted_upper_change_km3"] == sign * 3.0


def test_juvenile_crust_accounts_for_both_new_felsic_and_replaced_basalt(interval):
    first, last = interval
    last.cycle_rows[-1]["oceanic_replaced_by_juvenile_volume_km3"] = 7.0
    last.cycle.cumulative_generated_volume_km3 += 26.0
    last.state.oceanic_volume_km3[0] -= 7.0
    last.state.continental_volume_km3[0] += 26.0
    report = continuation_material_ledger(first, last)
    assert report["balanced"]
    assert report["generated_volume_km3"] == 26.0
    assert report["recycled_volume_km3"] == 7.0
    assert report["predicted_upper_change_km3"] == 19.0


@pytest.mark.parametrize("process", ["ocean_generation", "ocean_recycling", "felsic_generation", "felsic_recycling"])
def test_individual_continental_cycle_fluxes(interval, process):
    first, last = interval
    sign = 1 if process.endswith("generation") else -1
    if process.startswith("ocean"):
        field = "oceanic_generated_volume_km3" if sign > 0 else "oceanic_recycled_volume_km3"
        last.cycle_rows[-1][field] = 3.0
        last.state.oceanic_volume_km3[0] += sign * 3.0
    else:
        field = "cumulative_generated_volume_km3" if sign > 0 else "cumulative_recycled_volume_km3"
        setattr(last.cycle, field, getattr(last.cycle, field) + 3.0)
        last.state.continental_volume_km3[0] += sign * 3.0
    report = continuation_material_ledger(first, last)
    assert report["balanced"] and report["predicted_upper_change_km3"] == sign * 3.0


def test_erosion_is_internal_but_deep_sediment_is_a_mantle_return(interval):
    first, last = interval
    last.state.continental_volume_km3[0] -= 8.0
    last.state.sediment_volume_km3[0] += 5.0
    last.sediment_budget.cumulative_eroded_bedrock_volume_km3 += 8.0
    last.sediment_budget.deep_recycled_sediment_volume_km3 += 3.0
    report = continuation_material_ledger(first, last)
    assert report["balanced"] and report["generated_volume_km3"] == 0.0
    assert report["recycled_volume_km3"] == 3.0
    assert report["mantle_reference_mass_change_kg"] == 9e12


def test_sediment_rift_counter_does_not_double_count_lithosphere_recycling(interval):
    first, last = interval
    last.lithosphere_rows[-1]["rift_recycled_volume_km3"] = 4.0
    last.sediment_budget.cumulative_rift_recycled_volume_km3 += 4.0
    last.state.continental_volume_km3[0] -= 4.0
    report = continuation_material_ledger(first, last)
    assert report["balanced"] and report["recycled_volume_km3"] == 4.0


@pytest.mark.parametrize("reservoir", ["extrusive", "dyke", "underplate"])
@pytest.mark.parametrize("sign", [-1, 1])
def test_all_plume_reservoirs_have_independent_source_and_sink_counters(interval, reservoir, sign):
    first, last = interval
    process = "cumulative_generated" if sign > 0 else "deep_recycled"
    counter = f"{process}_{reservoir}_volume_km3"
    setattr(last.plume_magmatism_state, counter, getattr(last.plume_magmatism_state, counter) + 1.0)
    getattr(last.plume_magmatism_state, f"{reservoir}_volume_km3")[0] += sign * 1.0
    # Hotspot head/tail counters classify this same generated material, not a
    # second reservoir/source, and deliberately do not enter the ledger.
    last.hotspot_track_state = SimpleNamespace(cumulative_head_generated_volume_km3=10000.0)
    report = continuation_material_ledger(first, last)
    assert report["balanced"] and report["predicted_upper_change_km3"] == sign * 1.0


def test_internal_underplate_to_dyke_transfer_has_zero_mantle_exchange(interval):
    first, last = interval
    last.plume_magmatism_state.underplate_volume_km3[0] -= 2.0
    last.plume_magmatism_state.dyke_volume_km3[0] += 2.0
    report = continuation_material_ledger(first, last)
    assert report["balanced"] and report["mantle_reference_mass_change_kg"] == 0.0


def test_unrecorded_material_error_is_reported_not_absorbed_into_mantle(interval):
    first, last = interval
    last.lithosphere_rows[-1]["oceanic_created_volume_km3"] = 10.0
    last.state.oceanic_volume_km3[0] += 13.0
    report = continuation_material_ledger(first, last)
    assert not report["balanced"]
    assert report["volume_residual_km3"] == 3.0
    assert report["mantle_reference_mass_change_kg"] == -10.0 * 3e12


def test_recorded_exchange_without_array_change_also_fails(interval):
    first, last = interval
    last.lithosphere_rows[-1]["oceanic_subducted_volume_km3"] = 5.0
    report = continuation_material_ledger(first, last)
    assert not report["balanced"] and report["volume_residual_km3"] == 5.0


def test_resumed_interval_counts_only_new_records_and_new_cumulative_fluxes(interval):
    first, middle = interval
    middle.lithosphere_rows[-1]["oceanic_created_volume_km3"] = 7.0
    middle.state.oceanic_volume_km3[0] += 7.0
    middle.cycle.cumulative_generated_volume_km3 += 4.0
    middle.state.continental_volume_km3[0] += 4.0
    final = deepcopy(middle)
    final.state.time_myr = 3.0
    final.lithosphere_rows.append(lithosphere_row(3.0, oceanic_subducted_volume_km3=3.0))
    final.cycle_rows.append(cycle_row(3.0))
    final.state.oceanic_volume_km3[0] -= 3.0
    full = continuation_material_ledger(first, final)
    part_a = continuation_material_ledger(first, middle)
    part_b = continuation_material_ledger(middle, final)
    assert full["balanced"] and part_a["balanced"] and part_b["balanced"]
    assert part_b["new_lithosphere_records"] == 1
    assert part_b["generated_volume_km3"] == 0.0
    assert full["mantle_reference_mass_change_kg"] == (
        part_a["mantle_reference_mass_change_kg"] + part_b["mantle_reference_mass_change_kg"])


@pytest.mark.parametrize("density", [0.0, -1.0, np.nan, np.inf, True])
def test_invalid_reference_density_is_rejected(interval, density):
    with pytest.raises(ValueError):
        continuation_material_ledger(*interval, density)


@pytest.mark.parametrize("case", ["prefix_changed", "history_truncated", "missing_counter", "decreasing_counter",
                                   "negative_volume", "nonfinite_volume", "wrong_volume_shape", "past_record",
                                   "future_record", "past_checkpoint"])
def test_corrupt_material_or_history_is_rejected(interval, case):
    first, last = interval
    if case == "prefix_changed":
        last.lithosphere_rows[0]["oceanic_created_volume_km3"] += 1.0
    elif case == "history_truncated":
        last.lithosphere_rows.clear()
    elif case == "missing_counter":
        del last.lithosphere_rows[-1]["oceanic_created_volume_km3"]
    elif case == "decreasing_counter":
        last.cycle.cumulative_generated_volume_km3 -= 1.0
    elif case == "negative_volume":
        last.state.oceanic_volume_km3[0] = -1.0
    elif case == "nonfinite_volume":
        last.state.oceanic_volume_km3[0] = np.inf
    elif case == "wrong_volume_shape":
        last.state.oceanic_volume_km3 = np.zeros(3)
    elif case == "past_record":
        last.cycle_rows[-1]["time_myr"] = 1.0
    elif case == "future_record":
        last.cycle_rows[-1]["time_myr"] = 3.0
    else:
        last.state.time_myr = 0.5
    with pytest.raises(ValueError):
        continuation_material_ledger(first, last)


def test_missing_empty_optional_reservoirs_are_not_fabricated(interval):
    first, last = interval
    for checkpoint in (first, last):
        checkpoint.plume_magmatism_state = None
        checkpoint.sediment_budget = None
        checkpoint.state.sediment_volume_km3 = None
    report = continuation_material_ledger(first, last)
    assert report["balanced"]
    assert report["initial_upper_components_km3"]["plume_extrusive"] == 0.0
    assert report["initial_upper_components_km3"]["surface_sediment"] == 0.0


def test_reference_density_only_scales_mass_and_never_changes_volume_audit(interval):
    first, last = interval
    last.lithosphere_rows[-1]["oceanic_created_volume_km3"] = 2.0
    last.state.oceanic_volume_km3[0] += 2.0
    a = continuation_material_ledger(first, last, 3000.0)
    b = continuation_material_ledger(first, last, 3300.0)
    assert a["balanced"] and b["balanced"]
    assert a["predicted_upper_change_km3"] == b["predicted_upper_change_km3"]
    assert b["mantle_reference_mass_change_kg"] == pytest.approx(1.1 * a["mantle_reference_mass_change_kg"])


def test_planet_scale_ledger_serializes_without_numpy_scalar_fallback(interval):
    first, last = interval
    for checkpoint in (first, last):
        checkpoint.state.oceanic_volume_km3 *= 1e6
    report = continuation_material_ledger(first, last)
    assert type(report["balanced"]) is bool
    assert type(report["volume_tolerance_km3"]) is float
    assert json.loads(json.dumps(report, allow_nan=False)) == report
