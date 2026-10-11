"""Resealed first-entry forgeries must fail independently of producer replay.

Production supplies one original eight-octant fixture.  The independent
auditor reconstructs the continuous envelope, its complete quadrature and
material closure using its own standard-library mathematics.  Checksums are
recalculated for every attack so they cannot be the rejection mechanism.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import pytest

from analysis.geometric_native_first_entry_validation import independent_first_entry_audit as oracle
from tectonics.geometric_native_domain_journal import (
    initialize_native_domain_journal, native_domain_journal_from_dict,
    native_domain_journal_to_dict,
)
from tectonics.geometric_native_material_history import make_native_material_memory_reference
from tectonics.geometric_underthrust_motion import HotBirthParameters


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "analysis/geometric_native_full_domain_validation/control_release_01"


def reseal(raw, field):
    raw[field] = oracle.digest({key: value for key, value in raw.items() if key != field})
    assert raw[field] == oracle.digest({key: value for key, value in raw.items() if key != field})
    return raw


def fraction(raw):
    return Fraction(*raw)


def rational(value):
    value = Fraction(value)
    return [value.numerator, value.denominator]


def shift_interval(raw, amount):
    return {key: rational(fraction(raw[key]) + amount) for key in ("lo", "hi")}


@pytest.fixture(scope="module")
def actual():
    paths = {name: CONTROL / name for name in (
        "full_domain_journal.json", "original_force/initial.json",
        "original_force/force_candidate.json", "original_force/constitutive_time_support.json")}
    original_bytes = {name: path.read_bytes() for name, path in paths.items()}
    saved_raw, initial, candidate, support = (json.loads(original_bytes[name]) for name in paths)
    saved = native_domain_journal_from_dict(saved_raw)
    before = dict(json.loads(saved.memory_reference.before_payload_json),
        original_mechanics_payload=deepcopy(initial["native_underthrust_motion"]["initial_mechanics"]),
        original_surface_payload=deepcopy(initial["current_surface"]),
        original_force_checkpoint_sha256=hashlib.sha256(
            original_bytes["original_force/initial.json"]).hexdigest())
    memory = make_native_material_memory_reference(before,
        json.loads(saved.memory_reference.candidate_after_payload_json),
        constitutive_path_sha256=saved.memory_reference.constitutive_path_sha256)
    typed_journal = initialize_native_domain_journal(saved.source_surface,
        dict(saved.omega_rad_per_myr), saved.dt_myr, memory_reference=memory)
    journal = native_domain_journal_to_dict(typed_journal)
    binding = oracle.audit_source_binding(journal, initial["current_surface"],
        initial, candidate, support, initial_checkpoint_sha256=hashlib.sha256(
            original_bytes["original_force/initial.json"]).hexdigest())
    yield dict(journal=journal, typed_journal=typed_journal,
               initial=initial, candidate=candidate, support=support, binding=binding)
    assert all(path.read_bytes() == original_bytes[name] for name, path in paths.items())


@pytest.fixture(scope="module")
def sweep(actual):
    from tectonics.geometric_native_first_entry_domains import (
        certify_native_first_entry_domains, native_first_entry_domains_to_dict,
    )
    typed = certify_native_first_entry_domains(actual["typed_journal"], donor_owner=1,
        receiver_owner=0, integration_cells=16)
    raw = native_first_entry_domains_to_dict(typed)
    receipt = oracle.audit_sweep(raw, actual["binding"], actual["journal"])
    return dict(typed=typed, raw=raw, receipt=receipt)


@pytest.fixture(scope="module")
def material(actual, sweep):
    from tectonics.geometric_native_first_entry_material import (
        initialize_native_first_entry_material, native_first_entry_material_to_dict,
    )
    original = actual["initial"]["native_underthrust_motion"]
    state = initialize_native_first_entry_material(actual["typed_journal"], sweep["typed"],
        original_mechanics_payload=original["initial_mechanics"],
        birth_parameters=HotBirthParameters(**original["birth_parameters"]),
        provenance="Independent full first-entry oracle fixture from original physical sources")
    raw = native_first_entry_material_to_dict(state)
    receipt = oracle.audit_material(raw, sweep["receipt"], actual["binding"],
        actual["journal"], actual["initial"], actual["candidate"])
    return dict(raw=raw, receipt=receipt)


def reject_sweep(actual, raw):
    reseal(raw, "payload_sha256")
    with pytest.raises(ValueError):
        oracle.audit_sweep(raw, actual["binding"], actual["journal"])


def reject_material(actual, sweep, raw):
    reseal(raw, "state_sha256")
    with pytest.raises(ValueError):
        oracle.audit_material(raw, sweep["receipt"], actual["binding"],
            actual["journal"], actual["initial"], actual["candidate"])


def test_independent_whole_curved_sweep_keeps_all_four_original_donor_domains(sweep):
    raw = sweep["raw"]
    receipt = sweep["receipt"]
    assert receipt["whole_selected_owner_envelope_independently_proved"]
    assert receipt["whole_curved_excess_measure_independently_enclosed"]
    assert receipt["unique_ratio_maximum_independently_proved"]
    assert receipt["all_native_endpoint_clip_solid_angles_independently_proved"]
    assert receipt["physical_force_or_global_inventory_adopted"] is False
    assert {row["fragment_id"] for row in raw["donor_domains"]} == {
        "octant:4", "octant:5", "octant:6", "octant:7"}
    assert len(raw["quadrature"]) == raw["integration_cells"] == 16
    assert receipt["quadrature_cell_count"] == 16
    assert receipt["complete_donor_count"] == 4
    extra = raw["integration"]["extra_swept_area_bounds_sr"]
    assert 0 < fraction(extra["lo"]) <= fraction(extra["hi"]) < Fraction(1, 10**60)
    assert raw["extra_endpoint_fixed_parent_id"] == "octant:5"
    assert all(fraction(row["bounds"]["lo"]) > 0
               for row in raw["extra_endpoint_parent_support_bounds"])


def test_independent_material_audit_keeps_all_parents_and_once_only_cohorts(material):
    raw = material["raw"]
    receipt = material["receipt"]
    assert receipt["full_original_parent_count"] == 8
    assert receipt["once_only_archive_cohort_count"] == 3
    assert receipt["original_extensive_reservoir_count"] == 32
    assert receipt["all_original_reservoir_complements_authenticated"]
    assert receipt["common_exact_geometric_measure_symbols_preserved"]
    assert receipt["whole_curved_only_actual_vacancy_retained"]
    assert receipt["original_seed_trace_direction_authenticated"]
    assert receipt["endpoint_hot_birth_law_applied"]
    assert receipt["geometric_vacancy_equals_archived_swept_measure"]
    assert receipt["accepted_loaded_native_intervals"] == 0
    assert receipt["candidate_memory_adopted"] is False
    assert len(raw["parent_allocations"]) == len(raw["endpoint_vacancies"]) == 8
    assert {row["original_parent_id"] for row in raw["archive_cohorts"]} == {
        "octant:5", "octant:6", "octant:7"}
    assert raw["material_closure_applied"] is True
    assert raw["candidate_mechanics_adopted"] is False
    assert raw["native_force_intervals_accepted"] == 0


def test_removed_interior_quadrature_cell_cannot_be_hidden_by_resealing(actual, sweep):
    changed = deepcopy(sweep["raw"])
    changed["quadrature"].pop(len(changed["quadrature"]) // 2)
    reject_sweep(actual, changed)


def test_gap_inside_complete_time_partition_is_rejected_even_with_all_cells(actual, sweep):
    changed = deepcopy(sweep["raw"])
    cell = changed["quadrature"][len(changed["quadrature"]) // 2]
    low, high = map(fraction, (cell["time_fraction"]["lo"], cell["time_fraction"]["hi"]))
    cell["time_fraction"]["lo"] = rational(low + (high - low) / 4)
    reject_sweep(actual, changed)


def test_saved_zero_integrand_cannot_erase_a_positive_whole_area_cell(actual, sweep):
    changed = deepcopy(sweep["raw"])
    cell = next(row for row in reversed(changed["quadrature"])
                if fraction(row["integrand_bounds"]["lo"]) > 0)
    cell["integrand_bounds"] = cell["integral_bounds"] = {"lo": [0, 1], "hi": [0, 1]}
    reject_sweep(actual, changed)


def test_removed_transverse_series_is_not_authenticated_by_a_saved_polynomial(actual, sweep):
    changed = deepcopy(sweep["raw"])
    polynomial = next(row for row in changed["analytic_polynomials"] if row["name"] == "Z")
    assert any(fraction(coefficient) for coefficient in polynomial["coefficients"])
    polynomial["coefficients"] = [[0, 1] for _ in polynomial["coefficients"]]
    reject_sweep(actual, changed)


def test_false_whole_curvature_sign_is_rejected_beyond_the_receipt_hash(actual, sweep):
    changed = deepcopy(sweep["raw"])
    changed["proof_bounds"]["curvature_after_maximum_ratio_negative"] = {
        "lo": [0, 1], "hi": [1, 1]}
    reject_sweep(actual, changed)


def test_positive_curved_extra_cannot_be_replaced_with_endpoint_only_loss(actual, sweep):
    changed = deepcopy(sweep["raw"])
    zero = {"lo": [0, 1], "hi": [0, 1]}
    changed["integration"]["extra_swept_area_bounds_sr"] = deepcopy(zero)
    domain = next(row for row in changed["donor_domains"] if row["fragment_id"] == "octant:5")
    domain["extra_swept_area_bounds_sr"] = deepcopy(zero)
    domain["swept_area_bounds_sr"] = deepcopy(domain["endpoint_area_bounds_sr"])
    domain["domain_case"] = "endpoint_equals_whole_swept"
    reject_sweep(actual, changed)


def test_curved_extra_cannot_claim_a_foreign_fixed_world_parent(actual, sweep):
    changed = deepcopy(sweep["raw"])
    changed["extra_endpoint_fixed_parent_id"] = "octant:4"
    for row in changed["extra_endpoint_parent_support_bounds"]:
        row["normal"][2] *= -1
    reject_sweep(actual, changed)


def test_rebalanced_counterfeit_parent_reservoir_still_fails_original_source_binding(actual, sweep, material):
    changed = deepcopy(material["raw"])
    allocation = next(row for row in changed["parent_allocations"] if row["fragment_id"] == "octant:0")
    reservoir = next(row for row in allocation["reservoirs"] if row["reservoir"] == "oceanic_volume_km3")
    reservoir["parent_amount"] = rational(fraction(reservoir["parent_amount"]) + 1)
    reservoir["live_amount_bounds"] = shift_interval(reservoir["live_amount_bounds"], Fraction(1))
    reject_material(actual, sweep, changed)


def test_independent_reservoirs_cannot_use_different_archive_measure_symbols(actual, sweep, material):
    changed = deepcopy(material["raw"])
    allocation = next(row for row in changed["parent_allocations"] if row["fragment_id"] == "octant:5")
    reservoir = next(row for row in allocation["reservoirs"] if row["reservoir"] == "cold_mantle_volume_km3")
    reservoir["measure_reference"] = "native_area(counterfeit_endpoint_only_domain)/native_area(octant:5)"
    reservoir["archived_expression"] = "original_parent_amount * " + reservoir["measure_reference"]
    reject_material(actual, sweep, changed)


def test_material_vacancy_must_include_curved_extra_absent_from_raw_endpoint_gaps(actual, sweep, material):
    changed = deepcopy(material["raw"])
    vacancy = next(row for row in changed["endpoint_vacancies"]
                   if row["fixed_original_parent_id"] == "octant:5")
    assert vacancy["curved_extra_donor_ids"] == ["octant:5"]
    vacancy["curved_extra_donor_ids"] = []
    reject_material(actual, sweep, changed)


def test_newborn_cannot_reuse_an_original_material_identity(actual, sweep, material):
    changed = deepcopy(material["raw"])
    newborn = next(row["newborn"] for row in changed["endpoint_vacancies"] if row["newborn"])
    newborn["material_id"] = actual["initial"]["current_surface"]["fragments"][0]["parcel"]["material_id"]
    reject_material(actual, sweep, changed)


def test_hot_birth_cannot_inherit_a_cold_mantle_reservoir(actual, sweep, material):
    changed = deepcopy(material["raw"])
    newborn = next(row["newborn"] for row in changed["endpoint_vacancies"] if row["newborn"])
    newborn["cold_mantle_volume_km3"] = [1, 10**100]
    reject_material(actual, sweep, changed)


def test_endpoint_birth_cannot_be_backdated_to_an_unimplemented_continuous_birth_law(actual, sweep, material):
    changed = deepcopy(material["raw"])
    newborn = next(row["newborn"] for row in changed["endpoint_vacancies"] if row["newborn"])
    newborn["birth_time_myr"] = rational(fraction(newborn["birth_time_myr"]) / 2)
    reject_material(actual, sweep, changed)


def test_resealed_original_trace_direction_cannot_choose_the_opposite_donor(actual, sweep, material):
    changed = deepcopy(material["raw"])
    trace = changed["original_mechanics_payload"]["fault_traces"]["traces"][0]
    trace["subducting_plate"], trace["overriding_plate"] = trace["overriding_plate"], trace["subducting_plate"]
    changed["original_mechanics_sha256"] = oracle.digest(changed["original_mechanics_payload"])
    reject_material(actual, sweep, changed)


@pytest.mark.parametrize("field, value", [
    ("candidate_mechanics_adopted", True),
    ("physical_pair_admission_authorized", True),
    ("native_force_intervals_accepted", 1),
    ("mechanical_work_assigned_to_material_volume", True),
])
def test_material_closure_cannot_promote_refused_force_memory_or_allocate_mechanical_work(
        actual, sweep, material, field, value):
    changed = deepcopy(material["raw"])
    changed[field] = value
    reject_material(actual, sweep, changed)
