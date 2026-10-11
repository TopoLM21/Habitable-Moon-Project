"""Complete first-entry material closure retains source and force authority.

The historical eight-octant control contains both an internal receiving
material switch and a positive swept-only donor loss.  These tests exercise
those distinct cases rather than treating endpoint overlap as a swept domain.
No finite reservoir is assigned to a point, and material closure cannot promote
the unaccepted native force candidate into the old resolved surface registry.
"""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import pytest

from tectonics.fractional_surface import EXTENSIVE_FIELDS
from tectonics.geometric_native_domain_journal import (
    initialize_native_domain_journal, native_domain_journal_from_dict,
)
from tectonics.geometric_native_material_history import make_native_material_memory_reference
from tectonics.geometric_native_pose import FractionInterval as I
from tectonics.geometric_native_surface import native_surface_to_dict
from tectonics.geometric_underthrust_motion import HotBirthParameters


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "analysis/geometric_native_full_domain_validation/control_release_01"
PROVENANCE = "Declared once-only passive first-entry material control"


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")


def _seal(raw):
    """An attacker can calculate a checksum; replay must supply authority."""
    raw["state_sha256"] = hashlib.sha256(_encoded({key: value
        for key, value in raw.items() if key != "state_sha256"})).hexdigest()
    return raw


def _fraction(pair):
    return Fraction(*pair)


@pytest.fixture(scope="module")
def actual():
    paths = tuple(CONTROL / name for name in (
        "full_domain_journal.json", "point_certificates.json", "point_queries.json",
        "swept_only_material_patch.json", "transient_receiver_patch.json",
        "original_force/initial.json", "original_force/after_refusal.json",
        "original_force/force_candidate.json"))
    originals = {path: path.read_bytes() for path in paths}
    journal_raw, certificates, queries, swept_only, transient, initial, after, candidate = (
        json.loads(originals[path]) for path in paths)
    saved_journal = native_domain_journal_from_dict(journal_raw)
    memory = make_native_material_memory_reference(
        dict(json.loads(saved_journal.memory_reference.before_payload_json),
            original_mechanics_payload=deepcopy(initial["native_underthrust_motion"]["initial_mechanics"]),
            original_surface_payload=deepcopy(initial["current_surface"]),
            original_force_checkpoint_sha256=hashlib.sha256(
                originals[CONTROL / "original_force/initial.json"]).hexdigest()),
        json.loads(saved_journal.memory_reference.candidate_after_payload_json),
        constitutive_path_sha256=saved_journal.memory_reference.constitutive_path_sha256)
    journal = initialize_native_domain_journal(saved_journal.source_surface,
        dict(saved_journal.omega_rad_per_myr), saved_journal.dt_myr, memory_reference=memory)
    mechanics = initial["native_underthrust_motion"]["initial_mechanics"]
    birth = HotBirthParameters(**initial["native_underthrust_motion"]["birth_parameters"])
    return dict(journal=journal, saved_journal=saved_journal, mechanics=mechanics, birth=birth,
        original_files=originals, initial=initial, after=after, candidate=candidate,
        point_certificates={record["label"]: record["certificate"]
            for record in certificates if record["kind"] != "fixed_world_receiver"},
        point_queries=queries, swept_only=swept_only, transient=transient,
        parents={f.fragment_id: f for f in journal.source_surface.fragments})


@pytest.fixture(scope="module")
def sweep(actual):
    from tectonics.geometric_native_first_entry_domains import certify_native_first_entry_domains
    return certify_native_first_entry_domains(actual["journal"],
        donor_owner=1, receiver_owner=0, integration_cells=16)


@pytest.fixture(scope="module")
def state(actual, sweep):
    from tectonics.geometric_native_first_entry_material import initialize_native_first_entry_material
    return initialize_native_first_entry_material(actual["journal"], sweep,
        original_mechanics_payload=actual["mechanics"], birth_parameters=actual["birth"],
        provenance=PROVENANCE)


@pytest.fixture(scope="module")
def point_contexts(actual, state):
    from tectonics.geometric_native_first_entry_material import certify_native_first_entry_material_point
    parents = dict((("main_plane_reentry", "octant:6"),
        ("transverse_masked_main_entry", "octant:6"),
        ("ordinary_entry", "octant:6"), ("no_entry", "octant:6"),
        ("swept_only_owner_entry_exit", "octant:5")))

    class Contexts(dict):
        def __missing__(self, label):
            original = actual["point_certificates"][label]
            point = tuple(_fraction(pair) for pair in original["trajectory"]["point"])
            context = certify_native_first_entry_material_point(state, parents[label], point)
            self[label] = context
            return context

    return Contexts()


def test_all_eight_original_material_payloads_and_physical_force_refs_survive(actual, state):
    assert len(state.parent_allocations) == 8
    assert state.journal == actual["journal"]
    assert _encoded(native_surface_to_dict(state.journal.source_surface)) == _encoded(
        actual["initial"]["current_surface"])
    assert state.original_mechanics_payload_json == _encoded(actual["mechanics"])
    assert state.birth_parameters == actual["birth"]
    assert state.provenance == PROVENANCE
    assert actual["original_files"][CONTROL / "original_force/initial.json"] == (
        actual["original_files"][CONTROL / "original_force/after_refusal.json"])
    assert all(path.read_bytes() == before for path, before in actual["original_files"].items())


def test_material_closure_authenticates_four_original_physical_banks_without_force_adoption(actual, state):
    from tectonics.geometric_native_first_entry_material import native_first_entry_material_to_dict
    traces = actual["mechanics"]["fault_traces"]["traces"]
    assert len(traces) == 4
    assert {(trace["subducting_plate"], trace["overriding_plate"]) for trace in traces} == {(1, 0)}
    raw = native_first_entry_material_to_dict(state)
    assert raw["physical_pair_admission_authorized"] is False
    assert raw["candidate_mechanics_adopted"] is False
    assert raw["native_force_intervals_accepted"] == 0
    assert raw["endpoint_polygon_registry_compatible"] is False


def test_archive_cohorts_are_three_nonempty_original_parents_and_not_receiver_pairs(state):
    assert len(state.archive_cohorts) == 3
    assert {cohort.original_parent_id for cohort in state.archive_cohorts} == {
        "octant:5", "octant:6", "octant:7"}
    assert len({cohort.cohort_id for cohort in state.archive_cohorts}) == 3


@pytest.mark.parametrize("parent_id", [f"octant:{index}" for index in range(8)])
def test_parent_reservoirs_keep_exact_shared_native_measure_and_complement(actual, state, parent_id):
    allocation, = [item for item in state.parent_allocations if item.fragment_id == parent_id]
    parent = actual["parents"][parent_id]
    assert allocation.owner == parent.parcel.plate
    assert allocation.material_id == parent.parcel.material_id
    assert tuple(item.reservoir for item in allocation.reservoirs) == EXTENSIVE_FIELDS
    for item in allocation.reservoirs:
        original = Fraction.from_float(getattr(parent.parcel, item.reservoir))
        assert type(item.parent_amount) is Fraction and item.parent_amount == original
        assert item.archived_amount_bounds.lo >= 0
        assert item.archived_amount_bounds.hi <= original
        assert item.live_amount_bounds == I(original) - item.archived_amount_bounds
        assert item.archived_amount_bounds.lo + item.live_amount_bounds.hi == original
        assert item.archived_amount_bounds.hi + item.live_amount_bounds.lo == original
        assert "archive" in item.conservation_identity and "live" in item.conservation_identity
        assert item.measure_reference == allocation.measure_reference
    if parent.parcel.plate == 0 or parent_id == "octant:4":
        assert allocation.archived_area_bounds_sr == I(0)
        assert all(item.archived_amount_bounds == I(0) for item in allocation.reservoirs)
    else:
        assert allocation.archived_area_bounds_sr.lo > 0
        assert all(item.archived_amount_bounds.lo > 0 for item in allocation.reservoirs)


@pytest.mark.parametrize("parent_id", [f"octant:{index}" for index in range(8)])
def test_live_and_archive_keep_original_intensives_and_exact_source_age(actual, state, parent_id):
    allocation, = [item for item in state.parent_allocations if item.fragment_id == parent_id]
    parent = actual["parents"][parent_id]
    assert allocation.material_fields == parent.parcel.material_fields
    assert allocation.specific_properties == parent.parcel.specific_properties
    assert type(allocation.original_age_myr) is Fraction
    assert allocation.original_age_myr == Fraction.from_float(parent.parcel.age_myr)


def test_checkpoint_roundtrip_replays_full_material_closure_and_does_not_mutate(actual, state):
    from tectonics.geometric_native_first_entry_material import (
        native_first_entry_material_from_dict, native_first_entry_material_to_dict,
    )
    raw = native_first_entry_material_to_dict(state)
    copied = deepcopy(raw)
    old = _encoded(copied)
    rebuilt = native_first_entry_material_from_dict(copied,
        expected_journal=actual["journal"], expected_original_mechanics_payload=actual["mechanics"])
    assert rebuilt == state
    assert _encoded(native_first_entry_material_to_dict(rebuilt)) == old
    assert _encoded(copied) == old
    assert all(path.read_bytes() == before for path, before in actual["original_files"].items())


def test_material_checkpoint_cannot_be_relabelled_as_the_old_polygon_partition_or_registry(state):
    from tectonics.geometric_native_buried_support import native_buried_registry_to_dict
    with pytest.raises(ValueError):
        native_surface_to_dict(state)
    with pytest.raises(ValueError):
        native_buried_registry_to_dict(state)


def _historical_times(actual, label):
    return tuple(_fraction(record["time_fraction"]) for record in actual["point_queries"]
        if record["label"] == label)


def test_true_reentry_archives_once_and_never_unburies_after_contact_exit(actual, state, point_contexts):
    from tectonics.geometric_native_first_entry_material import evaluate_native_first_entry_material_point
    chronology = point_contexts["main_plane_reentry"]
    times = _historical_times(actual, "main_plane_reentry")
    queries = [evaluate_native_first_entry_material_point(chronology, time) for time in times]
    assert [query.material_allocation for query in queries] == [
        "live", "archived", "archived", "archived", "archived"]
    assert len({query.cohort_id for query in queries[1:]}) == 1
    cohort, = [cohort for cohort in state.archive_cohorts if cohort.original_parent_id == "octant:6"]
    assert queries[-1].cohort_id == cohort.cohort_id
    assert all(query.finite_material_reservoir_assigned_to_point is False for query in queries)
    assert all(query.physical_pair_ancestry_authenticated is False for query in queries)


def test_internal_receiver_switch_cannot_create_second_archive_or_new_fault(actual, point_contexts):
    from tectonics.geometric_native_first_entry_material import evaluate_native_first_entry_material_point
    from tectonics.geometric_native_domain_journal import evaluate_native_domain_journal_owner_point
    chronology = point_contexts["transverse_masked_main_entry"]
    times = _historical_times(actual, "transverse_masked_main_entry")
    queries = [evaluate_native_first_entry_material_point(chronology, time) for time in times]
    assert [query.material_allocation for query in queries] == [
        "live", "live", "archived", "archived", "archived"]
    assert queries[2].receiver_original_fragment_ids == ("octant:3",)
    assert queries[3].receiver_original_fragment_ids == ("octant:3",)
    assert queries[4].receiver_original_fragment_ids == ("octant:3",)
    current = evaluate_native_domain_journal_owner_point(chronology.receiver_chronology, times[3])
    assert tuple(identity for identity, query in current.receiver_queries if query.occupancy == "inside") == ("octant:2",)
    assert queries[2].cohort_id == queries[3].cohort_id == queries[4].cohort_id
    assert queries[2].original_trace_ids == queries[3].original_trace_ids == queries[4].original_trace_ids


def test_no_entry_point_stays_original_live_material(actual, point_contexts):
    from tectonics.geometric_native_first_entry_material import evaluate_native_first_entry_material_point
    chronology = point_contexts["no_entry"]
    for time in _historical_times(actual, "no_entry"):
        query = evaluate_native_first_entry_material_point(chronology, time)
        assert query.material_allocation == "live"
        assert query.archive_debits == 0
        assert query.receiver_original_fragment_ids == ()


def test_material_age_uses_exact_original_time_even_below_float_ulp(actual, point_contexts):
    from tectonics.geometric_native_first_entry_material import evaluate_native_first_entry_material_point
    chronology = point_contexts["transverse_masked_main_entry"]
    time = _historical_times(actual, "transverse_masked_main_entry")[2]
    age = Fraction.from_float(actual["parents"]["octant:6"].parcel.age_myr)
    increment = time * Fraction.from_float(actual["journal"].dt_myr)
    assert 0 < increment < Fraction(1, 10**17)
    query = evaluate_native_first_entry_material_point(chronology, time)
    assert type(query.material_age_myr) is Fraction
    assert query.material_age_myr == age + increment > age
    assert float(query.material_age_myr) == float(age)


@pytest.mark.parametrize("time", [Fraction(-1, 10), Fraction(11, 10), True, float("nan"), float("inf")])
def test_invalid_time_cannot_authorize_material_allocation(point_contexts, time):
    from tectonics.geometric_native_first_entry_material import evaluate_native_first_entry_material_point
    with pytest.raises(ValueError):
        evaluate_native_first_entry_material_point(point_contexts["ordinary_entry"], time)


def test_swept_only_donor_five_remains_archived_after_real_owner_exit(actual, point_contexts):
    from tectonics.geometric_native_first_entry_material import evaluate_native_first_entry_material_point
    chronology = point_contexts["swept_only_owner_entry_exit"]
    times = _historical_times(actual, "swept_only_owner_entry_exit")
    queries = [evaluate_native_first_entry_material_point(chronology, time) for time in times]
    assert [query.material_allocation for query in queries] == [
        "live", "archived", "archived", "archived", "archived"]
    assert queries[2].receiver_original_fragment_ids == ("octant:1",)
    assert queries[-1].receiver_original_fragment_ids == ("octant:1",)
    assert queries[-1].contact_occupancy == "outside"
    assert queries[2].cohort_id == queries[-1].cohort_id


def test_raw_covered_swept_only_point_is_a_real_endpoint_material_vacancy_with_hot_fill(point_contexts):
    from tectonics.geometric_native_first_entry_material import evaluate_native_first_entry_material_endpoint_point
    query = evaluate_native_first_entry_material_endpoint_point(point_contexts["swept_only_owner_entry_exit"])
    assert query.raw_coverage is True
    assert query.retained_material is False
    assert query.actual_material_vacancy is True
    assert query.birth_material_ids
    assert query.fixed_original_parent_ids
    assert query.physical_pair_admission_authorized is False


def test_no_entry_endpoint_trajectory_keeps_original_live_material_without_birth(point_contexts):
    from tectonics.geometric_native_first_entry_material import evaluate_native_first_entry_material_endpoint_point
    query = evaluate_native_first_entry_material_endpoint_point(point_contexts["no_entry"])
    assert query.raw_coverage is True
    assert query.retained_material is True
    assert query.actual_material_vacancy is False
    assert query.birth_material_ids == ()


def test_archived_donor_still_inside_receiver_has_retained_receiver_material_and_no_hot_fill(point_contexts):
    from tectonics.geometric_native_first_entry_material import evaluate_native_first_entry_material_endpoint_point
    query = evaluate_native_first_entry_material_endpoint_point(point_contexts["main_plane_reentry"])
    assert query.point_query.material_allocation == "archived"
    assert query.point_query.contact_occupancy == "inside"
    assert query.raw_coverage is True and query.retained_material is True
    assert query.actual_material_vacancy is False
    assert query.birth_material_ids == ()


def test_first_entry_material_state_is_immutable(state):
    with pytest.raises(AttributeError):
        state.provenance = "forged replacement"


@pytest.mark.parametrize("parameters", [None, {},
    HotBirthParameters(-1., 0., 0., "invalid thickness"),
    HotBirthParameters(2., float("nan"), 0., "invalid damage"),
    HotBirthParameters(2., 0., float("inf"), "invalid water"),
    HotBirthParameters(2., 0., 0., "")])
def test_hot_fill_requires_explicit_valid_material_parameters(actual, sweep, parameters):
    from tectonics.geometric_native_first_entry_material import initialize_native_first_entry_material
    with pytest.raises(ValueError):
        initialize_native_first_entry_material(actual["journal"], sweep,
            original_mechanics_payload=actual["mechanics"], birth_parameters=parameters,
            provenance=PROVENANCE)


@pytest.mark.parametrize("provenance", [None, "", True, 123])
def test_first_entry_closure_requires_declared_provenance(actual, sweep, provenance):
    from tectonics.geometric_native_first_entry_material import initialize_native_first_entry_material
    with pytest.raises(ValueError):
        initialize_native_first_entry_material(actual["journal"], sweep,
            original_mechanics_payload=actual["mechanics"], birth_parameters=actual["birth"],
            provenance=provenance)


def test_mechanical_reference_is_frozen_against_later_caller_mutation(actual, sweep):
    from tectonics.geometric_native_first_entry_material import initialize_native_first_entry_material
    supplied = deepcopy(actual["mechanics"])
    original = _encoded(supplied)
    closure = initialize_native_first_entry_material(actual["journal"], sweep,
        original_mechanics_payload=supplied, birth_parameters=actual["birth"], provenance=PROVENANCE)
    supplied["cumulative_source_work_j"] += 1.
    supplied["fault_traces"]["traces"][0]["subducting_plate"] = 0
    assert closure.original_mechanics_payload_json == original


@pytest.mark.parametrize("damage", ["missing_trace", "reversed_polarity", "foreign_surface", "reset_work"])
def test_unrelated_or_modified_original_mechanics_cannot_authenticate_material_transfer(actual, sweep, damage):
    from tectonics.geometric_native_first_entry_material import initialize_native_first_entry_material
    mechanics = deepcopy(actual["mechanics"])
    if damage == "missing_trace":
        mechanics["fault_traces"]["traces"].pop()
    elif damage == "reversed_polarity":
        mechanics["fault_traces"]["traces"][0]["subducting_plate"] = 0
        mechanics["fault_traces"]["traces"][0]["overriding_plate"] = 1
    elif damage == "foreign_surface":
        mechanics["surface_sha256"] = "0" * 64
    else:
        mechanics["preparation_work_j"] = 0.
    with pytest.raises(ValueError):
        initialize_native_first_entry_material(actual["journal"], sweep,
            original_mechanics_payload=mechanics, birth_parameters=actual["birth"], provenance=PROVENANCE)


def test_external_original_mechanics_mismatch_is_rejected_even_for_valid_checkpoint(actual, state):
    from tectonics.geometric_native_first_entry_material import (
        native_first_entry_material_from_dict, native_first_entry_material_to_dict,
    )
    different = deepcopy(actual["mechanics"])
    different["cumulative_source_work_j"] = 1.
    with pytest.raises(ValueError):
        native_first_entry_material_from_dict(native_first_entry_material_to_dict(state),
            expected_journal=actual["journal"], expected_original_mechanics_payload=different)


def test_old_geometry_diagnostic_without_authenticated_source_cannot_apply_material_law(actual):
    from tectonics.geometric_native_first_entry_domains import certify_native_first_entry_domains
    from tectonics.geometric_native_first_entry_material import initialize_native_first_entry_material
    old = actual["saved_journal"]
    geometric_proof = certify_native_first_entry_domains(old,
        donor_owner=1, receiver_owner=0, integration_cells=16)
    with pytest.raises(ValueError):
        initialize_native_first_entry_material(old, geometric_proof,
            original_mechanics_payload=actual["mechanics"], birth_parameters=actual["birth"],
            provenance=PROVENANCE)


@pytest.mark.parametrize("field", ["physical_pair_admission_authorized", "candidate_mechanics_adopted", "native_force_intervals_accepted"])
def test_resealed_checkpoint_cannot_promote_material_closure_to_force_admission(actual, state, field):
    from tectonics.geometric_native_first_entry_material import (
        native_first_entry_material_from_dict, native_first_entry_material_to_dict,
    )
    raw = deepcopy(native_first_entry_material_to_dict(state))
    raw[field] = 1 if field == "native_force_intervals_accepted" else True
    _seal(raw)
    with pytest.raises(ValueError):
        native_first_entry_material_from_dict(raw, expected_journal=actual["journal"],
            expected_original_mechanics_payload=actual["mechanics"])


def test_arbitrary_resealed_material_schema_extension_cannot_adopt_material(actual, state):
    from tectonics.geometric_native_first_entry_material import (
        native_first_entry_material_from_dict, native_first_entry_material_to_dict,
    )
    raw = deepcopy(native_first_entry_material_to_dict(state))
    raw["resurrect_archived_material"] = True
    _seal(raw)
    with pytest.raises(ValueError):
        native_first_entry_material_from_dict(raw, expected_journal=actual["journal"],
            expected_original_mechanics_payload=actual["mechanics"])


def test_all_eight_fixed_domains_reserve_actual_vacancies_and_five_nonzero_hot_histories(actual, state):
    assert len(state.endpoint_vacancies) == 8
    assert {vacancy.fixed_original_parent_id for vacancy in state.endpoint_vacancies} == set(actual["parents"])
    nonzero = [vacancy for vacancy in state.endpoint_vacancies if vacancy.newborn is not None]
    assert {vacancy.fixed_original_parent_id for vacancy in nonzero} == {
        "octant:0", "octant:2", "octant:4", "octant:5", "octant:6"}
    assert len({vacancy.newborn.material_id for vacancy in nonzero}) == 5
    assert not ({vacancy.newborn.material_id for vacancy in nonzero}
        & set(state.journal.source_surface.known_material_ids))
    for vacancy in state.endpoint_vacancies:
        assert type(vacancy.area_bounds_sr) is I
        if vacancy.newborn is None:
            assert vacancy.area_bounds_sr == I(0)
        else:
            assert vacancy.area_bounds_sr.lo > 0
            assert vacancy.newborn.vacancy_domain_id == vacancy.vacancy_domain_id


def test_hot_newborns_have_endpoint_clock_declared_crust_and_zero_cold_or_excess_mass(actual, state):
    radius_squared = Fraction.from_float(state.journal.source_surface.radius_km)**2
    thickness = Fraction.from_float(actual["birth"].crust_thickness_km)
    endpoint = Fraction.from_float(state.journal.source_surface.time_myr) + Fraction.from_float(state.journal.dt_myr)
    for vacancy in state.endpoint_vacancies:
        newborn = vacancy.newborn
        if newborn is None:
            continue
        parent = actual["parents"][vacancy.fixed_original_parent_id]
        assert newborn.owner == parent.parcel.plate
        assert newborn.cell == parent.parcel.cell
        assert newborn.birth_time_myr == endpoint
        assert newborn.area_km2_bounds == vacancy.area_bounds_sr * radius_squared
        assert newborn.oceanic_volume_km3_bounds == newborn.area_km2_bounds * thickness
        assert type(newborn.age_myr) is Fraction and newborn.age_myr == 0
        assert type(newborn.cold_mantle_volume_km3) is Fraction and newborn.cold_mantle_volume_km3 == 0
        assert type(newborn.density_excess_mass_kg) is Fraction and newborn.density_excess_mass_kg == 0
        assert newborn.material_fields == (("damage", actual["birth"].damage), ("water", actual["birth"].water))
        assert newborn.specific_properties == (thickness, Fraction(0), Fraction(0))


def test_full_positive_swept_only_excess_survives_inside_raw_covered_parent_five(state):
    vacancy, = [item for item in state.endpoint_vacancies if item.fixed_original_parent_id == "octant:5"]
    raw = sum((item.area_bounds_sr for item in state.journal.raw_vacancies
        if item.fixed_original_parent_id == "octant:5"), I(0))
    assert vacancy.curved_extra_donor_ids == ("octant:5",)
    assert vacancy.area_bounds_sr.lo > raw.lo
    assert vacancy.area_bounds_sr.hi > raw.hi
    # The extra is physically below binary64 resolution of the large raw gap;
    # retaining it requires the exact measure provenance and its enclosure.
    assert float(vacancy.area_bounds_sr.lo) == float(raw.lo)
    assert float(vacancy.area_bounds_sr.hi) == float(raw.hi)
    donor_five, = [item for item in state.parent_allocations if item.fragment_id == "octant:5"]
    endpoint, = [item for item in state.sweep_certificate.donor_domains if item.fragment_id == "octant:5"]
    assert donor_five.archived_area_bounds_sr.lo > endpoint.endpoint_area_bounds_sr.hi


@pytest.mark.parametrize("damage", ["double_cohort", "zero_debit", "new_receiver_cohort",
    "missing_tiny_birth", "newborn_cold_mass", "reset_original_age", "false_midpoint_debit"])
def test_resealed_derived_material_ledger_cannot_override_conservation_or_once_law(actual, state, damage):
    from tectonics.geometric_native_first_entry_material import (
        native_first_entry_material_from_dict, native_first_entry_material_to_dict,
    )
    raw = deepcopy(native_first_entry_material_to_dict(state))
    if damage == "double_cohort":
        raw["archive_cohorts"].append(deepcopy(raw["archive_cohorts"][0]))
    elif damage == "zero_debit":
        donor_five, = [item for item in raw["parent_allocations"] if item["fragment_id"] == "octant:5"]
        donor_five["reservoirs"][0]["archived_amount_bounds"] = {"lo": [0, 1], "hi": [0, 1]}
    elif damage == "new_receiver_cohort":
        raw["archive_cohorts"][0]["cohort_id"] = "forged-separate-receiver-cohort"
    elif damage == "missing_tiny_birth":
        vacancy, = [item for item in raw["endpoint_vacancies"] if item["fixed_original_parent_id"] == "octant:0"]
        vacancy["newborn"] = None
    elif damage == "newborn_cold_mass":
        vacancy, = [item for item in raw["endpoint_vacancies"] if item["fixed_original_parent_id"] == "octant:5"]
        vacancy["newborn"]["cold_mantle_volume_km3"] = [1, 1]
        vacancy["newborn"]["density_excess_mass_kg"] = [1, 1]
    elif damage == "reset_original_age":
        raw["parent_allocations"][0]["original_age_myr"] = [0, 1]
    else:
        raw["parent_allocations"][0]["reservoirs"][0]["interval_midpoint_used_as_debit"] = True
    _seal(raw)
    with pytest.raises(ValueError):
        native_first_entry_material_from_dict(raw, expected_journal=actual["journal"],
            expected_original_mechanics_payload=actual["mechanics"])


@pytest.mark.parametrize("damage", ["float_amount", "float_age", "boolean_owner", "float_birth_age"])
def test_equal_numeric_types_cannot_alias_trusted_exact_material_state(state, damage):
    from tectonics.geometric_native_first_entry_material import (
        native_first_entry_material_to_dict, validate_native_first_entry_material,
    )
    validate_native_first_entry_material(state)
    if damage == "float_amount":
        allocation = state.parent_allocations[0]
        item = replace(allocation.reservoirs[0], parent_amount=float(allocation.reservoirs[0].parent_amount))
        modified = replace(allocation, reservoirs=(item, *allocation.reservoirs[1:]))
        forged = replace(state, parent_allocations=(modified, *state.parent_allocations[1:]))
    elif damage == "float_age":
        modified = replace(state.parent_allocations[0], original_age_myr=float(state.parent_allocations[0].original_age_myr))
        forged = replace(state, parent_allocations=(modified, *state.parent_allocations[1:]))
    elif damage == "boolean_owner":
        modified = replace(state.parent_allocations[0], owner=False)
        forged = replace(state, parent_allocations=(modified, *state.parent_allocations[1:]))
    else:
        vacancy = state.endpoint_vacancies[0]
        newborn = replace(vacancy.newborn, age_myr=0.)
        forged = replace(state, endpoint_vacancies=(replace(vacancy, newborn=newborn), *state.endpoint_vacancies[1:]))
    with pytest.raises(ValueError):
        native_first_entry_material_to_dict(forged)


def test_first_entry_root_window_keeps_allocation_unresolved(point_contexts):
    from tectonics.geometric_native_first_entry_material import evaluate_native_first_entry_material_point
    chronology = point_contexts["main_plane_reentry"]
    window = chronology.first_entry_certificate.first_entry_time_fraction
    query = evaluate_native_first_entry_material_point(chronology, (window.lo + window.hi) / 2)
    assert query.material_allocation == "event_window"
    assert query.contact_occupancy == "event_window"
    assert query.first_entry_time_fraction == window
    assert query.archive_debits == 0


@pytest.mark.parametrize("index", [1, 2])
def test_exit_or_reentry_window_preserves_the_single_earlier_archive(point_contexts, index):
    from tectonics.geometric_native_first_entry_material import evaluate_native_first_entry_material_point
    chronology = point_contexts["main_plane_reentry"]
    window = chronology.first_entry_certificate.plane_chronology.events[index].time_fraction
    query = evaluate_native_first_entry_material_point(chronology, (window.lo + window.hi) / 2)
    assert query.contact_occupancy == "event_window"
    assert query.material_allocation == "archived"
    assert query.archive_debits == 1


def test_first_receiving_material_keeps_both_original_banks_without_inventing_their_spring(actual, state, point_contexts):
    from tectonics.geometric_native_first_entry_material import (
        evaluate_native_first_entry_material_point, native_first_entry_material_point_query_to_dict,
    )
    query = evaluate_native_first_entry_material_point(point_contexts["transverse_masked_main_entry"], 1)
    donor_trace, = [binding for binding in state.trace_bindings if binding.donor_fragment_id == "octant:6"]
    receiver_trace, = [binding for binding in state.trace_bindings if binding.receiver_fragment_id == "octant:3"]
    assert query.original_trace_ids == (donor_trace.trace_id,)
    assert query.receiver_trace_ids == (receiver_trace.trace_id,)
    assert donor_trace.receiver_fragment_id == "octant:2"
    assert receiver_trace.donor_fragment_id == "octant:7"
    assert query.original_trace_ids != query.receiver_trace_ids
    assert query.first_entry_receiver_ancestry_status == "proved"
    assert query.physical_pair_ancestry_authenticated is False
    assert native_first_entry_material_point_query_to_dict(query)["donor_receiver_trace_difference_creates_new_fault"] is False


def test_nonselected_original_receiver_material_is_never_archived(state):
    from tectonics.geometric_native_first_entry_material import (
        certify_native_first_entry_material_point, evaluate_native_first_entry_material_point,
    )
    chronology = certify_native_first_entry_material_point(state, "octant:2",
        (Fraction(-1), Fraction(1), Fraction(-1)))
    assert chronology.first_entry_certificate is None
    assert chronology.receiver_chronology is None
    query = evaluate_native_first_entry_material_point(chronology, 1)
    assert query.material_allocation == "live" and query.archive_debits == 0
    assert query.cohort_id is None
    assert query.material_owner == 0 and query.material_id == "octant-material:2"


def test_point_context_and_queries_roundtrip_against_their_authoritative_material_state(state, point_contexts):
    from tectonics.geometric_native_first_entry_material import (
        native_first_entry_material_point_chronology_from_dict,
        native_first_entry_material_point_chronology_to_dict,
        evaluate_native_first_entry_material_point,
        native_first_entry_material_point_query_from_dict,
        native_first_entry_material_point_query_to_dict,
    )
    chronology = point_contexts["swept_only_owner_entry_exit"]
    raw = native_first_entry_material_point_chronology_to_dict(chronology)
    loaded = native_first_entry_material_point_chronology_from_dict(deepcopy(raw), state=state)
    assert loaded == chronology
    assert _encoded(native_first_entry_material_point_chronology_to_dict(loaded)) == _encoded(raw)
    query = evaluate_native_first_entry_material_point(loaded, 1)
    raw_query = native_first_entry_material_point_query_to_dict(query)
    rebuilt = native_first_entry_material_point_query_from_dict(deepcopy(raw_query), chronology=loaded)
    assert rebuilt == query
    assert _encoded(native_first_entry_material_point_query_to_dict(rebuilt)) == _encoded(raw_query)


@pytest.mark.parametrize("damage", ["revive", "receiver_replacement", "new_cohort", "second_debit", "physical_pair_adopted", "point_volume"])
def test_saved_point_claims_cannot_replace_first_entry_ancestry_or_once_law(point_contexts, damage):
    from tectonics.geometric_native_first_entry_material import (
        evaluate_native_first_entry_material_point,
        native_first_entry_material_point_query_from_dict,
        native_first_entry_material_point_query_to_dict,
    )
    chronology = point_contexts["transverse_masked_main_entry"]
    raw = deepcopy(native_first_entry_material_point_query_to_dict(
        evaluate_native_first_entry_material_point(chronology, 1)))
    if damage == "revive":
        raw["material_allocation"] = "live"
    elif damage == "receiver_replacement":
        raw["receiver_original_fragment_ids"] = ["octant:2"]
    elif damage == "new_cohort":
        raw["cohort_id"] = "forged-endpoint-receiver-cohort"
    elif damage == "second_debit":
        raw["archive_debits"] = 2
    elif damage == "physical_pair_adopted":
        raw["physical_pair_ancestry_authenticated"] = True
    else:
        raw["finite_material_reservoir_assigned_to_point"] = True
    with pytest.raises(ValueError):
        native_first_entry_material_point_query_from_dict(raw, chronology=chronology)


@pytest.mark.parametrize("damage", ["float_age", "boolean_owner"])
def test_equal_numeric_point_types_cannot_alias_authoritative_query(point_contexts, damage):
    from tectonics.geometric_native_first_entry_material import (
        evaluate_native_first_entry_material_point, native_first_entry_material_point_query_to_dict,
    )
    query = evaluate_native_first_entry_material_point(point_contexts["ordinary_entry"], 0)
    native_first_entry_material_point_query_to_dict(query)
    forged = (replace(query, material_age_myr=float(query.material_age_myr))
        if damage == "float_age" else replace(query, material_owner=True))
    with pytest.raises(ValueError):
        native_first_entry_material_point_query_to_dict(forged)


def test_point_query_cannot_reuse_another_original_material_context(point_contexts):
    from tectonics.geometric_native_first_entry_material import (
        evaluate_native_first_entry_material_point,
        native_first_entry_material_point_query_to_dict,
        native_first_entry_material_point_query_from_dict,
    )
    raw = native_first_entry_material_point_query_to_dict(
        evaluate_native_first_entry_material_point(point_contexts["swept_only_owner_entry_exit"], 1))
    with pytest.raises(ValueError):
        native_first_entry_material_point_query_from_dict(raw, chronology=point_contexts["ordinary_entry"])


def test_endpoint_vacancy_query_roundtrips_with_exact_rotated_ray_and_source_context(point_contexts):
    from tectonics.geometric_intrinsic_geometry import RotationIncrement
    from tectonics.geometric_native_polygons import NativeRay, native_ray_to_dict
    from tectonics.geometric_native_first_entry_material import (
        evaluate_native_first_entry_material_endpoint_point,
        native_first_entry_material_endpoint_query_to_dict,
        native_first_entry_material_endpoint_query_from_dict,
    )
    chronology = point_contexts["swept_only_owner_entry_exit"]
    query = evaluate_native_first_entry_material_endpoint_point(chronology)
    raw = native_first_entry_material_endpoint_query_to_dict(query)
    loaded = native_first_entry_material_endpoint_query_from_dict(deepcopy(raw), chronology=chronology)
    assert loaded == query
    assert _encoded(native_first_entry_material_endpoint_query_to_dict(loaded)) == _encoded(raw)
    assert len(query.endpoint_ray.rotations) == 1
    parent = next(fragment for fragment in chronology.state.journal.source_surface.fragments
                  if fragment.fragment_id == chronology.original_parent_id)
    step = RotationIncrement(
        dict(chronology.state.journal.omega_rad_per_myr)[parent.parcel.plate],
        chronology.state.journal.dt_myr)
    expected = NativeRay("seed", chronology.original_point).rotated((step,))
    assert query.endpoint_ray == expected
    assert _encoded(native_ray_to_dict(query.endpoint_ray)) == _encoded(native_ray_to_dict(expected))


@pytest.mark.parametrize("damage", ["revive", "suppress_vacancy", "drop_birth", "native_pose", "force_adopted"])
def test_saved_endpoint_query_cannot_hide_true_swept_only_vacancy_or_birth(point_contexts, damage):
    from tectonics.geometric_native_first_entry_material import (
        evaluate_native_first_entry_material_endpoint_point,
        native_first_entry_material_endpoint_query_to_dict,
        native_first_entry_material_endpoint_query_from_dict,
    )
    chronology = point_contexts["swept_only_owner_entry_exit"]
    raw = deepcopy(native_first_entry_material_endpoint_query_to_dict(
        evaluate_native_first_entry_material_endpoint_point(chronology)))
    if damage == "revive":
        raw["retained_material"] = True
    elif damage == "suppress_vacancy":
        raw["actual_material_vacancy"] = False
    elif damage == "drop_birth":
        raw["birth_material_ids"] = []
    elif damage == "native_pose":
        raw["endpoint_ray"]["rotations"] = []
    else:
        raw["candidate_mechanics_adopted"] = True
    with pytest.raises(ValueError):
        native_first_entry_material_endpoint_query_from_dict(raw, chronology=chronology)


def test_endpoint_nested_point_owner_cannot_use_boolean_alias(point_contexts):
    from tectonics.geometric_native_first_entry_material import (
        evaluate_native_first_entry_material_endpoint_point,
        native_first_entry_material_endpoint_query_to_dict,
    )
    query = evaluate_native_first_entry_material_endpoint_point(point_contexts["swept_only_owner_entry_exit"])
    native_first_entry_material_endpoint_query_to_dict(query)
    forged = replace(query, point_query=replace(query.point_query, material_owner=True))
    with pytest.raises(ValueError):
        native_first_entry_material_endpoint_query_to_dict(forged)


def test_warm_material_cache_rechecks_all_fields_after_same_object_reservoir_mutation(state):
    from tectonics.geometric_native_first_entry_material import validate_native_first_entry_material
    allocation = state.parent_allocations[5]
    reservoir = replace(allocation.reservoirs[0])
    copied = replace(state, parent_allocations=(*state.parent_allocations[:5],
        replace(allocation, reservoirs=(reservoir, *allocation.reservoirs[1:])),
        *state.parent_allocations[6:]))
    validate_native_first_entry_material(copied)
    object.__setattr__(reservoir, "parent_amount", Fraction(0))
    with pytest.raises(ValueError):
        validate_native_first_entry_material(copied)
    assert state.parent_allocations[5].reservoirs[0].parent_amount > 0


def test_warm_point_cache_rechecks_nested_original_source_epoch_after_same_object_mutation(state, point_contexts):
    from tectonics.geometric_native_first_entry_material import validate_native_first_entry_material_point
    from tectonics.geometric_native_surface import native_surface_from_dict
    source = native_surface_from_dict(native_surface_to_dict(state.journal.source_surface))
    private_state = replace(state, journal=replace(state.journal, source_surface=source))
    copied = replace(point_contexts["ordinary_entry"], state=private_state)
    validate_native_first_entry_material_point(copied)
    object.__setattr__(source, "time_myr", 1.)
    with pytest.raises(ValueError):
        validate_native_first_entry_material_point(copied)
    assert state.journal.source_surface.time_myr == 0.
