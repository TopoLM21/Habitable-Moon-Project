"""Full endpoint domains retain the actual unaccepted native force source.

The reference checks use the native kernel on the immutable source fragments,
not the journal's saved geometry, areas, or claimed conservation fields.  Raw
kinematic holes describe coverage by moved footprints; they do not establish a
live-material vacancy, a continuous birth law, or a globally accepted step.
"""
from copy import deepcopy
from dataclasses import asdict, replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import pytest

from tectonics.fractional_surface import EXTENSIVE_FIELDS
from tectonics.geometric_intrinsic_geometry import RotationIncrement
from tectonics.geometric_native_buried_support import native_pullback
from tectonics.geometric_native_material_history import (
    make_native_material_memory_reference,
)
from tectonics.geometric_native_polygons import (
    NativePolygon, NativeRay, _primitive, _ray_values, _side,
    native_intersection, native_polygon_from_dict,
    native_polygon_to_dict, native_subtract,
)
from tectonics.geometric_native_pose import FractionInterval as I
from tectonics.geometric_native_surface import (
    native_surface_from_dict, native_surface_to_dict,
)


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "analysis/geometric_native_contact_event_validation/time_support_release_01"
PROBE = ROOT / "analysis/geometric_native_underthrust_validation/force_control_release_02/force_candidate_refusal_probe.json"
EXPECTED_PAIRS = {
    ("octant:1", "octant:5"), ("octant:2", "octant:6"),
    ("octant:2", "octant:7"), ("octant:3", "octant:5"),
    ("octant:3", "octant:7"),
}
EXPECTED_GAP_COUNTS = {"octant:0": 1, "octant:1": 0, "octant:2": 1,
                       "octant:3": 0, "octant:4": 2, "octant:5": 2,
                       "octant:6": 1, "octant:7": 0}


def _encoded(raw):
    return json.dumps(raw, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash(raw):
    return hashlib.sha256(_encoded(raw)).hexdigest()


def _subtract_all(parent, covers):
    residual = (parent,)
    for cover in covers:
        residual = tuple(piece for geometry in residual
                         for piece in native_subtract(geometry, cover))
    return residual


def _same_native_set(left, right):
    assert not native_subtract(left, right)
    assert not native_subtract(right, left)


@pytest.fixture(scope="module")
def actual():
    paths = (CONTROL / "initial.json", CONTROL / "force_candidate.json",
             CONTROL / "constitutive_time_support.json", PROBE)
    files = {path: path.read_bytes() for path in paths}
    initial, candidate, support, probe = (json.loads(files[path]) for path in paths)
    surface = native_surface_from_dict(initial["current_surface"])
    omega = {owner: tuple(vector) for owner, vector in candidate["omega_rad_per_myr"]}
    reference = make_native_material_memory_reference(
        initial["native_underthrust_motion"]["snapshot"],
        dict(candidate=deepcopy(candidate), whole_time_support=deepcopy(support),
             adopted=False), constitutive_path_sha256=support["path_sha256"])
    moved = {f.fragment_id: f.geometry.rotated(omega[f.parcel.plate],
        candidate["admission"]["dt_myr"]) for f in surface.fragments}
    return dict(initial=initial, candidate=candidate, support=support, probe=probe,
        surface=surface, omega=omega, dt=candidate["admission"]["dt_myr"],
        reference=reference, files=files, moved=moved,
        parents={f.fragment_id: f for f in surface.fragments})


@pytest.fixture(scope="module")
def journal(actual):
    from tectonics.geometric_native_domain_journal import initialize_native_domain_journal
    return initialize_native_domain_journal(actual["surface"], actual["omega"],
        actual["dt"], memory_reference=actual["reference"])


def test_all_eight_native_parents_full_motion_and_original_source_are_preserved(actual, journal):
    assert len(journal.source_surface.fragments) == len(journal.moved_fragments) == 8
    assert _encoded(native_surface_to_dict(journal.source_surface)) == _encoded(
        actual["initial"]["current_surface"])
    assert dict(journal.omega_rad_per_myr) == actual["omega"]
    assert all(value != 0 for vector in actual["omega"].values() for value in vector)
    assert journal.dt_myr == actual["dt"]
    assert journal.source_surface_sha256 == _hash(actual["initial"]["current_surface"])
    assert journal.memory_reference == actual["reference"]
    assert actual["probe"]["accepted_loaded_native_intervals"] == 0
    assert actual["probe"]["input_unchanged"] is True
    assert all(path.read_bytes() == old for path, old in actual["files"].items())


@pytest.mark.parametrize("identity", [f"octant:{number}" for number in range(8)])
def test_each_moved_parent_has_its_complete_owner_rotation_and_exact_age(actual, journal, identity):
    moved, = [f for f in journal.moved_fragments if f.fragment_id == identity]
    parent = actual["parents"][identity]
    assert moved.owner == parent.parcel.plate
    assert moved.material_id == parent.parcel.material_id
    assert _encoded(native_polygon_to_dict(moved.geometry)) == _encoded(
        native_polygon_to_dict(actual["moved"][identity]))
    assert moved.geometry.pose[-1] == RotationIncrement(
        actual["omega"][parent.parcel.plate], actual["dt"])
    assert type(moved.material_age_at_endpoint_myr) is Fraction
    assert moved.material_age_at_endpoint_myr == (
        Fraction.from_float(parent.parcel.age_myr) + Fraction.from_float(actual["dt"]))


def test_all_five_actual_overlap_pairs_are_enumerated_without_an_area_floor(actual, journal):
    actual_pairs = {(p.fragment_a, p.fragment_b) for p in journal.endpoint_pairs}
    assert actual_pairs == EXPECTED_PAIRS
    assert len(journal.endpoint_pairs) == 5
    assert actual_pairs == {(p["fragment_a"], p["fragment_b"])
                            for p in actual["probe"]["admission"]["actual_overlap_assessments"]}
    tiny, = [p for p in journal.endpoint_pairs
             if (p.fragment_a, p.fragment_b) == ("octant:3", "octant:5")]
    assert 0 < tiny.area_bounds_sr.lo <= tiny.area_bounds_sr.hi < Fraction(1, 10**58)


@pytest.mark.parametrize("identities", sorted(EXPECTED_PAIRS))
def test_overlap_is_the_independent_full_native_intersection_and_both_original_pullbacks(actual, journal, identities):
    left, right = identities
    pair, = [p for p in journal.endpoint_pairs
             if (p.fragment_a, p.fragment_b) == identities]
    independent = native_intersection(actual["moved"][left], actual["moved"][right])
    assert independent is not None
    _same_native_set(pair.overlap_geometry, independent)
    evidence, = [p for p in actual["probe"]["admission"]["actual_overlap_assessments"]
                 if (p["fragment_a"], p["fragment_b"]) == identities]
    _same_native_set(pair.overlap_geometry, native_polygon_from_dict(evidence["native_overlap"]))
    assert pair.area_bounds_sr.lo > 0
    assert pair.area_bounds_sr.lo <= independent.area_bounds().hi
    assert independent.area_bounds().lo <= pair.area_bounds_sr.hi
    for identity, owner, domain in (
        (left, pair.owner_a, pair.donor_a_original_domain),
        (right, pair.owner_b, pair.donor_b_original_domain)):
        parent = actual["parents"][identity]
        assert owner == parent.parcel.plate
        step = RotationIncrement(actual["omega"][owner], actual["dt"])
        pulled_back = native_pullback(independent, (step,))
        _same_native_set(domain, pulled_back)
        assert domain.pose[-1] == RotationIncrement(
            tuple(-value for value in actual["omega"][owner]), actual["dt"])
        assert not native_subtract(domain, parent.geometry)
        _same_native_set(domain.rotated(actual["omega"][owner], actual["dt"]), independent)


def test_seven_raw_vacancies_include_the_positive_transverse_fringes(journal):
    assert len(journal.raw_vacancies) == 7
    assert {identity: sum(v.fixed_original_parent_id == identity
                         for v in journal.raw_vacancies)
            for identity in EXPECTED_GAP_COUNTS} == EXPECTED_GAP_COUNTS
    gaps = [v for v in journal.raw_vacancies
            if v.fixed_original_parent_id in ("octant:0", "octant:2", "octant:6")]
    assert len(gaps) == 3
    assert all(0 < gap.area_bounds_sr.lo <= gap.area_bounds_sr.hi < Fraction(1, 10**45)
               for gap in gaps)


@pytest.mark.parametrize("identity", [f"octant:{number}" for number in range(8)])
def test_raw_vacancy_is_subtraction_of_all_eight_moved_footprints_from_fixed_old_parent(actual, journal, identity):
    parent = actual["parents"][identity].geometry
    expected = _subtract_all(parent, actual["moved"].values())
    saved = tuple(v.geometry for v in journal.raw_vacancies
                  if v.fixed_original_parent_id == identity)
    assert len(expected) == len(saved) == EXPECTED_GAP_COUNTS[identity]
    assert all(not _subtract_all(part, saved) for part in expected)
    assert all(not _subtract_all(part, expected) for part in saved)
    for index, gap in enumerate(saved):
        assert gap.area_bounds().lo > 0
        assert not native_subtract(gap, parent)
        assert all(native_intersection(gap, moved) is None
                   for moved in actual["moved"].values())
        assert all(native_intersection(gap, other) is None for other in saved[index+1:])
    covered = tuple(hit for moved in actual["moved"].values()
                    if (hit := native_intersection(parent, moved)) is not None)
    # Coverage is geometric, even when a positive hole is below the binary64
    # area tolerance. Overlap duplication must not be counted as new material.
    assert not _subtract_all(parent, (*covered, *saved))
    for hit in covered:
        assert all(native_intersection(hit, gap) is None for gap in saved)


def test_journal_preserves_four_parent_reservoirs_and_material_properties(actual, journal):
    original = actual["parents"]
    saved = {f.fragment_id: f for f in journal.source_surface.fragments}
    for identity, parent in original.items():
        assert asdict(saved[identity].parcel) == asdict(parent.parcel)
        for reservoir in EXTENSIVE_FIELDS:
            assert Fraction.from_float(getattr(saved[identity].parcel, reservoir)) == (
                Fraction.from_float(getattr(parent.parcel, reservoir)))
        assert saved[identity].parcel.material_fields == parent.parcel.material_fields
        assert saved[identity].parcel.specific_properties == parent.parcel.specific_properties
    assert len(journal.source_surface.known_material_ids) == 8


def test_journal_dto_complete_replay_is_byte_identical_and_does_not_mutate_inputs(actual, journal):
    from tectonics.geometric_native_domain_journal import (
        native_domain_journal_from_dict, native_domain_journal_to_dict,
    )
    raw = native_domain_journal_to_dict(journal)
    copied = deepcopy(raw)
    before = _encoded(copied)
    loaded = native_domain_journal_from_dict(copied,
        expected_memory_reference=actual["reference"])
    assert _encoded(native_domain_journal_to_dict(loaded)) == _encoded(raw)
    assert _encoded(copied) == before
    assert loaded == journal
    assert all(path.read_bytes() == old for path, old in actual["files"].items())


def test_journal_is_immutable(journal):
    with pytest.raises(AttributeError):
        journal.dt_myr = 1.


@pytest.mark.parametrize("duration", [0., -1., True, float("nan"), float("inf"), "1"])
def test_invalid_physical_duration_is_rejected_before_a_domain_claim(actual, duration):
    from tectonics.geometric_native_domain_journal import initialize_native_domain_journal
    with pytest.raises((TypeError, ValueError)):
        initialize_native_domain_journal(actual["surface"], actual["omega"], duration,
            memory_reference=actual["reference"])


@pytest.mark.parametrize("damage", ["missing_owner", "extra_owner", "bool_owner", "float_owner",
                                  "short_vector", "bool_component", "nan_component", "inf_component"])
def test_owner_mapping_requires_complete_canonical_finite_physical_vectors(actual, damage):
    from tectonics.geometric_native_domain_journal import initialize_native_domain_journal
    omega = dict(actual["omega"])
    if damage == "missing_owner":
        del omega[1]
    elif damage == "extra_owner":
        omega[2] = omega[1]
    elif damage in ("bool_owner", "float_owner"):
        omega = {0: omega[0], (True if damage == "bool_owner" else 1.): omega[1]}
    else:
        vector = list(omega[1])
        if damage == "short_vector":
            vector.pop()
        else:
            vector[0] = {"bool_component": True, "nan_component": float("nan"),
                         "inf_component": float("inf")}[damage]
        omega[1] = tuple(vector)
    with pytest.raises((TypeError, ValueError)):
        initialize_native_domain_journal(actual["surface"], omega, actual["dt"],
            memory_reference=actual["reference"])


def test_replay_refuses_a_different_authenticated_mechanical_reference(actual, journal):
    from tectonics.geometric_native_domain_journal import (
        native_domain_journal_from_dict, native_domain_journal_to_dict,
    )
    reference = make_native_material_memory_reference(dict(unrelated_before=True),
        dict(unrelated_after=True, adopted=False), constitutive_path_sha256="a"*64)
    with pytest.raises(ValueError):
        native_domain_journal_from_dict(native_domain_journal_to_dict(journal),
            expected_memory_reference=reference)


def test_whole_parent_material_domains_bind_all_receivers_and_four_conserved_reservoirs(actual, journal):
    from tectonics.geometric_native_domain_journal import native_domain_journal_to_dict
    raw = native_domain_journal_to_dict(journal)
    parents = raw["whole_original_parent_domains"]
    assert {p["fragment_id"] for p in parents} == set(actual["parents"])
    for saved in parents:
        parent = actual["parents"][saved["fragment_id"]]
        assert saved["material_id"] == parent.parcel.material_id
        assert saved["owner"] == parent.parcel.plate
        assert saved["original_geometry_sha256"] == _hash(native_polygon_to_dict(parent.geometry))
        assert saved["original_age_myr"] == [Fraction.from_float(parent.parcel.age_myr).numerator,
                                             Fraction.from_float(parent.parcel.age_myr).denominator]
        contacts = saved["potential_contact_domains"]
        expected_receivers = {f.fragment_id for f in actual["surface"].fragments
                              if f.parcel.plate != parent.parcel.plate}
        assert {c["receiver_fragment_id"] for c in contacts} == expected_receivers
        assert len(contacts) == len(expected_receivers) == 4
        for contact in contacts:
            receiver = actual["parents"][contact["receiver_fragment_id"]]
            assert contact["receiver_owner"] == receiver.parcel.plate
            assert contact["receiver_payload_sha256"] == _hash(dict(
                fragment_id=receiver.fragment_id, geometry=native_polygon_to_dict(receiver.geometry),
                parcel=asdict(receiver.parcel), parent_fragment_id=receiver.parent_fragment_id))
            assert contact["physical_pair_and_direction_authenticated"] is False
            # In original parent coordinates the receiver normal is Rp^-1 Rr n.
            # The inverse operators belong on this normal before its dot q.
            assert "(exp(-Omega_parent*u*D) exp(Omega_receiver*u*D) receiver_normal) dot q" in contact["predicate"]
        assert tuple(r["reservoir"] for r in saved["original_reservoirs"]) == EXTENSIVE_FIELDS
        for reservoir in saved["original_reservoirs"]:
            amount = Fraction.from_float(getattr(parent.parcel, reservoir["reservoir"]))
            assert reservoir["original_amount"] == [amount.numerator, amount.denominator]
            assert reservoir["each_conditional_amount_bounds"] == {
                "lo": [0, 1], "hi": [amount.numerator, amount.denominator]}
            assert reservoir["conditional_live_expression"] == "original_amount - conditional_archive_amount"
            assert "conditional_live_amount + conditional_archive_amount = original_amount" == (
                reservoir["conservation_identity"])
            assert "physically_authenticated_first_contact" in reservoir["conditional_archive_expression"]
            assert reservoir["material_allocation_adopted"] is False
        assert saved["contact_exit_does_not_unbury_under_conditional_law"] is True


def test_raw_gap_and_hot_birth_reservations_cannot_be_adopted_as_actual_material_vacancies(actual, journal):
    from tectonics.geometric_native_domain_journal import native_domain_journal_to_dict
    raw = native_domain_journal_to_dict(journal)
    for flag in ("raw_gap_equals_passive_archive_material_vacancy_proved",
                 "endpoint_pair_direction_resolved", "material_loss_decided",
                 "original_material_allocation_adopted", "newborn_material_created",
                 "candidate_mechanics_adopted", "global_inventory_updated", "admission_authorized"):
        assert raw[flag] is False
    assert raw["raw_gap_domain"] != raw["passive_archive_material_vacancy_domain"]
    reservations = raw["endpoint_hot_birth_reservations"]
    assert len(reservations) == 7
    gaps = {gap.vacancy_sha256: gap for gap in journal.raw_vacancies}
    endpoint = Fraction.from_float(actual["surface"].time_myr) + Fraction.from_float(actual["dt"])
    radius = Fraction.from_float(actual["surface"].radius_km)
    for reservation in reservations:
        vacancy = gaps[reservation["vacancy_sha256"]]
        assert reservation["fixed_original_parent_id"] == vacancy.fixed_original_parent_id
        assert reservation["possible_birth_time_myr"] == [endpoint.numerator, endpoint.denominator]
        area = vacancy.area_bounds_sr * radius * radius
        assert reservation["raw_area_km2_bounds"] == {
            "lo": [area.lo.numerator, area.lo.denominator],
            "hi": [area.hi.numerator, area.hi.denominator]}
        assert reservation["original_kinematic_gap_is_only_a_reservation"] is True
        assert reservation["actual_retained_surface_available"] is False
        assert reservation["newborn_parcel_created"] is False
        assert reservation["continuous_birth_or_newborn_motion_defined"] is False
        assert reservation["damage_water_and_material_identity_require_explicit_birth_factory"] is True
        assert reservation["prospective_cold_mantle_volume"] == [0, 1]
        assert reservation["prospective_density_excess_mass"] == [0, 1]
        assert reservation["prospective_age_at_birth"] == [0, 1]
        assert "parcel" not in reservation


def test_external_mechanical_payloads_and_work_are_preserved_without_volume_assignment(actual, journal):
    from tectonics.geometric_native_domain_journal import native_domain_journal_to_dict
    memory = native_domain_journal_to_dict(journal)["memory_reference"]
    assert _encoded(memory["before_payload"]) == actual["reference"].before_payload_json
    assert _encoded(memory["candidate_after_payload"]) == actual["reference"].candidate_after_payload_json
    assert memory["before_sha256"] == _hash(memory["before_payload"])
    assert memory["candidate_after_sha256"] == _hash(memory["candidate_after_payload"])
    assert memory["constitutive_path_sha256"] == actual["support"]["path_sha256"]
    assert memory["candidate_after_payload"]["adopted"] is False
    assert _encoded(memory["candidate_after_payload"]["candidate"]) == _encoded(actual["candidate"])
    assert _encoded(memory["candidate_after_payload"]["whole_time_support"]) == _encoded(actual["support"])


@pytest.mark.parametrize("damage", [
    "top_hash", "source_hash", "missing_pair", "missing_vacancy", "duplicate_pair",
    "wrong_pair_owner", "wrong_pullback_rotation", "lost_overlap_plane", "fake_area_floor",
    "wrong_vacancy_parent", "wrong_vacancy_geometry", "wrong_reservoir", "double_reservoir",
    "false_archive_cohort", "allocation_adopted", "newborn_created", "global_adoption",
    "raw_gap_equated_to_material_gap", "fake_continuous_birth", "parent_source_amount",
    "source_epsilon", "float_rational", "noncanonical_rational", "bool_owner",
    "integer_component", "float_direction_flag", "extra_field", "mechanical_payload",
])
def test_resealed_journal_cannot_change_native_domains_material_provenance_or_authority(actual, journal, damage):
    from tectonics.geometric_native_domain_journal import (
        native_domain_journal_from_dict, native_domain_journal_to_dict,
    )
    raw = deepcopy(native_domain_journal_to_dict(journal))
    pair = raw["endpoint_pairs"][0]
    gap = raw["raw_vacancies"][0]
    parent = raw["whole_original_parent_domains"][0]
    if damage == "top_hash":
        raw["journal_sha256"] = "0"*64
    elif damage == "source_hash":
        raw["source_surface_sha256"] = "0"*64
    elif damage == "missing_pair":
        raw["endpoint_pairs"].pop()
    elif damage == "missing_vacancy":
        raw["raw_vacancies"].pop()
    elif damage == "duplicate_pair":
        raw["endpoint_pairs"].append(deepcopy(pair))
    elif damage == "wrong_pair_owner":
        pair["owner_b"] = pair["owner_a"]
    elif damage == "wrong_pullback_rotation":
        pair["donor_a_original_domain"]["pose"][-1]["omega_rad_per_myr"][0] = 0.
    elif damage == "lost_overlap_plane":
        pair["overlap_geometry"]["supports"].pop()
    elif damage == "fake_area_floor":
        pair["area_bounds_sr"]["lo"] = [0, 1]
    elif damage == "wrong_vacancy_parent":
        gap["fixed_original_parent_id"] = "octant:1"
    elif damage == "wrong_vacancy_geometry":
        gap["geometry"] = deepcopy(raw["source_surface"]["fragments"][0]["geometry"])
    elif damage == "wrong_reservoir":
        parent["original_reservoirs"][0]["original_amount"][0] += 1
    elif damage == "double_reservoir":
        parent["original_reservoirs"].append(deepcopy(parent["original_reservoirs"][0]))
    elif damage == "false_archive_cohort":
        parent["archive_cohort_id"] = "invented-whole-parent-cohort"
    elif damage == "allocation_adopted":
        parent["original_reservoirs"][0]["material_allocation_adopted"] = True
    elif damage == "newborn_created":
        raw["endpoint_hot_birth_reservations"][0]["newborn_parcel_created"] = True
    elif damage == "global_adoption":
        raw["admission_authorized"] = True
        raw["global_inventory_updated"] = True
    elif damage == "raw_gap_equated_to_material_gap":
        raw["raw_gap_equals_passive_archive_material_vacancy_proved"] = True
    elif damage == "fake_continuous_birth":
        raw["endpoint_hot_birth_reservations"][0]["continuous_birth_or_newborn_motion_defined"] = True
    elif damage == "parent_source_amount":
        raw["source_surface"]["fragments"][0]["parcel"]["cold_mantle_volume_km3"] *= 2
    elif damage == "source_epsilon":
        raw["source_surface"]["fragments"][0]["geometry"]["area_epsilon"] = 1e-10
    elif damage == "float_rational":
        raw["moved_fragments"][0]["material_age_at_endpoint_myr"][0] = float(
            raw["moved_fragments"][0]["material_age_at_endpoint_myr"][0])
    elif damage == "noncanonical_rational":
        value = raw["moved_fragments"][0]["material_age_at_endpoint_myr"]
        value[:] = [2*value[0], 2*value[1]]
    elif damage == "bool_owner":
        raw["omega_rad_per_myr"][1][0] = True
    elif damage == "integer_component":
        raw["omega_rad_per_myr"][0][1][0] = 0
    elif damage == "float_direction_flag":
        raw["endpoint_pair_direction_resolved"] = 0.
    elif damage == "extra_field":
        raw["accepted_native_force_intervals"] = 1
    elif damage == "mechanical_payload":
        raw["memory_reference"]["candidate_after_payload"]["adopted"] = True
        raw["memory_reference"]["candidate_after_sha256"] = _hash(
            raw["memory_reference"]["candidate_after_payload"])
    else:
        raise AssertionError(damage)
    # Even a freshly sealed, otherwise well-formed body must be rebuilt from
    # authoritative source geometry and external memory before it is trusted.
    for item in raw["endpoint_pairs"]:
        item.pop("pair_sha256")
        item["pair_sha256"] = _hash(item)
    for item in raw["raw_vacancies"]:
        item.pop("vacancy_sha256")
        item["vacancy_sha256"] = _hash(item)
    if damage != "top_hash":
        del raw["journal_sha256"]
        raw["journal_sha256"] = _hash(raw)
    with pytest.raises((TypeError, ValueError)):
        native_domain_journal_from_dict(raw, expected_memory_reference=actual["reference"])


@pytest.mark.parametrize("damage", ["float_age", "bool_owner", "zero_pair_area",
                                  "missing_pair", "missing_vacancy", "wrong_source_hash",
                                  "float_owner", "integer_motion", "wrong_parent_hash"])
def test_warmed_replay_cache_cannot_authorize_forged_frozen_journal(journal, damage):
    from tectonics.geometric_native_domain_journal import validate_native_domain_journal
    validate_native_domain_journal(journal)
    if damage in ("float_age", "bool_owner", "wrong_parent_hash"):
        first = journal.moved_fragments[0]
        field, value = {
            "float_age": ("material_age_at_endpoint_myr", float(first.material_age_at_endpoint_myr)),
            "bool_owner": ("owner", False), "wrong_parent_hash": ("parent_payload_sha256", "0"*64),
        }[damage]
        forged = replace(journal, moved_fragments=(replace(first, **{field: value}),
                                                  *journal.moved_fragments[1:]))
    elif damage == "zero_pair_area":
        forged = replace(journal, endpoint_pairs=(replace(journal.endpoint_pairs[0],
            area_bounds_sr=I(0)), *journal.endpoint_pairs[1:]))
    elif damage == "missing_pair":
        forged = replace(journal, endpoint_pairs=journal.endpoint_pairs[:-1])
    elif damage == "missing_vacancy":
        forged = replace(journal, raw_vacancies=journal.raw_vacancies[:-1])
    elif damage == "wrong_source_hash":
        forged = replace(journal, source_surface_sha256="0"*64)
    elif damage == "float_owner":
        rows = journal.omega_rad_per_myr
        forged = replace(journal, omega_rad_per_myr=((float(rows[0][0]), rows[0][1]), *rows[1:]))
    elif damage == "integer_motion":
        rows = journal.omega_rad_per_myr
        forged = replace(journal, omega_rad_per_myr=((rows[0][0], (0, *rows[0][1][1:])), *rows[1:]))
    else:
        raise AssertionError(damage)
    with pytest.raises((TypeError, ValueError)):
        validate_native_domain_journal(forged)


def test_native_domain_journal_never_uses_rounded_world_geometry(actual, monkeypatch):
    from tectonics.geometric_native_domain_journal import initialize_native_domain_journal
    def forbidden(*args, **kwargs):
        raise AssertionError("Native domains cannot be inferred from display vertices")
    monkeypatch.setattr(NativePolygon, "world_polygon", forbidden)
    # A distinct exact time prevents an endpoint-geometry cache hit from
    # making this control vacuous.
    copied = initialize_native_domain_journal(actual["surface"], actual["omega"],
        actual["dt"]*0.5, memory_reference=actual["reference"])
    assert len(copied.endpoint_pairs) == 5
    assert len(copied.raw_vacancies) == 7


@pytest.fixture(scope="module")
def pair_point(actual, journal):
    from tectonics.geometric_native_domain_journal import certify_native_domain_journal_pair_point
    from tectonics.geometric_native_donor_sweep import (
        certify_native_donor_patch_sweep, isolate_native_donor_patch_point_events,
    )
    patch = certify_native_donor_patch_sweep(actual["parents"]["octant:6"].geometry,
        actual["parents"]["octant:2"].geometry, actual["omega"][1], actual["omega"][0], actual["dt"])
    point = (patch.patch_x_bounds.midpoint, patch.patch_y_bounds.midpoint, Fraction(-1))
    chronology = certify_native_domain_journal_pair_point(journal, "octant:6", "octant:2", point)
    independent_roots = isolate_native_donor_patch_point_events(patch, point[0], point[1])
    return chronology, independent_roots, patch


def test_generic_pair_point_has_the_same_three_events_as_the_prior_independent_donor_path(actual, journal, pair_point):
    from tectonics.geometric_native_domain_journal import evaluate_native_domain_journal_pair_point
    chronology, old_events, patch = pair_point
    certificate = chronology.chronology_certificate
    assert chronology.journal is journal
    assert certificate.donor_omega == actual["omega"][1]
    assert certificate.receiver_omega == actual["omega"][0]
    assert len(certificate.plane_chronologies) == 3
    assert tuple(event.kind for event in certificate.occupancy_events) == ("entry", "exit", "entry")
    for event, independent in zip(certificate.occupancy_events, old_events):
        assert event.time_fraction.lo <= independent.time_fraction.hi
        assert independent.time_fraction.lo <= event.time_fraction.hi
    queries = [evaluate_native_domain_journal_pair_point(chronology, u) for u in patch.witness_times]
    assert [q.contact_occupancy for q in queries] == ["outside", "inside", "outside", "inside", "inside"]
    assert [q.conditional_material_allocation for q in queries] == ["live", "archived", "archived", "archived", "archived"]
    assert len({q.conditional_cohort_id for q in queries}) == 1
    parent = actual["parents"]["octant:6"].parcel
    for u, query in zip(patch.witness_times, queries):
        assert query.material_age_myr == Fraction.from_float(parent.age_myr) + u*Fraction.from_float(actual["dt"])
        assert query.original_parent_reservoir_references == tuple(
            (field, Fraction.from_float(getattr(parent, field))) for field in EXTENSIVE_FIELDS)
        assert query.memory_reference == actual["reference"]
        assert query.pair_direction_authenticated is False
        assert query.finite_material_reservoir_assigned_to_point is False
        assert query.admission_authorized is False


def test_conditional_archive_does_not_use_a_common_midpoint_event_clock(pair_point):
    from tectonics.geometric_native_domain_journal import evaluate_native_domain_journal_pair_point
    chronology, _, _ = pair_point
    events = chronology.chronology_certificate.occupancy_events
    first = evaluate_native_domain_journal_pair_point(chronology, events[0].time_fraction.midpoint)
    assert first.contact_occupancy == "event_window"
    assert first.conditional_material_allocation == "first_entry_window"
    for event in events[1:]:
        query = evaluate_native_domain_journal_pair_point(chronology, event.time_fraction.midpoint)
        assert query.contact_occupancy == "event_window"
        assert query.conditional_material_allocation == "archived"
        assert query.conditional_cohort_id == first.conditional_cohort_id


def test_pair_point_and_queries_replay_byte_identically_without_adopting_material(journal, pair_point):
    from tectonics.geometric_native_domain_journal import (
        evaluate_native_domain_journal_pair_point, native_domain_pair_point_chronology_from_dict,
        native_domain_pair_point_chronology_to_dict, native_domain_pair_point_query_to_dict,
    )
    chronology, _, patch = pair_point
    raw = native_domain_pair_point_chronology_to_dict(chronology)
    loaded = native_domain_pair_point_chronology_from_dict(deepcopy(raw), journal=journal)
    assert _encoded(native_domain_pair_point_chronology_to_dict(loaded)) == _encoded(raw)
    for u in patch.witness_times:
        old = native_domain_pair_point_query_to_dict(evaluate_native_domain_journal_pair_point(chronology, u))
        new = native_domain_pair_point_query_to_dict(evaluate_native_domain_journal_pair_point(loaded, u))
        assert _encoded(old) == _encoded(new)
        assert old["conditional_archive_law"] == "once_only_passive_first_entry_if_pair_and_direction_are_authenticated"
        assert old["material_loss_decided"] is False
        assert old["finite_material_reservoir_assigned_to_point"] is False
        assert old["admission_authorized"] is False


def test_whole_parent_query_accepts_a_point_outside_the_previous_bounded_patch(journal):
    from tectonics.geometric_native_domain_journal import (
        certify_native_domain_journal_pair_point, evaluate_native_domain_journal_pair_point,
    )
    chronology = certify_native_domain_journal_pair_point(journal, "octant:6", "octant:2", (1, 1, -1))
    assert not chronology.chronology_certificate.occupancy_events
    for u in (Fraction(0), Fraction(1, 2), Fraction(1)):
        query = evaluate_native_domain_journal_pair_point(chronology, u)
        assert query.contact_occupancy == "outside"
        assert query.conditional_material_allocation == "live"
        assert query.conditional_first_entry_time_fraction is None


@pytest.mark.parametrize("damage", ["donor", "same_owner", "receiver", "point", "motion",
                                  "source_hash", "journal_hash", "global_adoption"])
def test_pair_point_context_cannot_be_resealed_to_another_source_or_material_direction(journal, pair_point, damage):
    from tectonics.geometric_native_domain_journal import (
        native_domain_pair_point_chronology_from_dict, native_domain_pair_point_chronology_to_dict,
    )
    raw = deepcopy(native_domain_pair_point_chronology_to_dict(pair_point[0]))
    if damage == "donor":
        raw["donor_fragment_id"] = "octant:7"
    elif damage == "same_owner":
        raw["receiver_fragment_id"] = "octant:5"
    elif damage == "receiver":
        raw["receiver_fragment_id"] = "octant:3"
    elif damage == "point":
        raw["original_point"][0] = [1, 1]
    elif damage == "motion":
        cert = raw["chronology_certificate"]
        cert["trajectory"]["donor_omega"][0] = 0.
        cert.pop("payload_sha256")
        cert["payload_sha256"] = _hash(cert)
    elif damage == "source_hash":
        raw["source_surface_sha256"] = "0"*64
    elif damage == "journal_hash":
        raw["journal_sha256"] = "0"*64
    elif damage == "global_adoption":
        raw["admission_authorized"] = True
        raw["pair_direction_authenticated"] = True
    raw.pop("point_chronology_sha256")
    raw["point_chronology_sha256"] = _hash(raw)
    with pytest.raises((TypeError, ValueError)):
        native_domain_pair_point_chronology_from_dict(raw, journal=journal)


@pytest.mark.parametrize("damage", ["allocation", "float_age", "cohort", "reservoir", "admission"])
def test_pair_point_query_serializer_replays_material_references_and_typed_age(pair_point, damage):
    from tectonics.geometric_native_domain_journal import (
        evaluate_native_domain_journal_pair_point, native_domain_pair_point_query_to_dict,
    )
    query = evaluate_native_domain_journal_pair_point(pair_point[0], Fraction(1))
    if damage == "allocation":
        forged = replace(query, conditional_material_allocation="live")
    elif damage == "float_age":
        forged = replace(query, material_age_myr=float(query.material_age_myr))
    elif damage == "cohort":
        forged = replace(query, conditional_cohort_id="false-second-entry-cohort")
    elif damage == "reservoir":
        forged = replace(query, original_parent_reservoir_references=query.original_parent_reservoir_references[:-1])
    elif damage == "admission":
        forged = replace(query, admission_authorized=True)
    with pytest.raises((TypeError, ValueError)):
        native_domain_pair_point_query_to_dict(forged)


@pytest.fixture(scope="module")
def world_gap_point(actual, journal):
    from tectonics.geometric_native_domain_journal import certify_native_domain_journal_world_point
    gap = max((v for v in journal.raw_vacancies if v.fixed_original_parent_id == "octant:4"),
              key=lambda item: item.area_bounds_sr.lo)
    rays = [_ray_values(ray.rotated(gap.geometry.pose), 256)[0] for ray in gap.geometry.vertices]
    point = tuple(sum((ray[i] for ray in rays), Fraction(0)) for i in range(3))
    exact_ray = NativeRay("seed", _primitive(point))
    assert all(_side(exact_ray, support) > 0 for support in gap.geometry.support_planes)
    assert all(any(_side(exact_ray, support) < 0 for support in moved.support_planes)
               for moved in actual["moved"].values())
    return certify_native_domain_journal_world_point(journal, point)


def test_fixed_world_query_uses_all_eight_native_original_domains_and_only_proves_a_raw_gap(journal, world_gap_point):
    from tectonics.geometric_native_domain_journal import (
        evaluate_native_domain_journal_world_point, native_domain_world_point_query_to_dict,
    )
    assert len(world_gap_point.receiver_certificates) == 8
    assert {identity for identity, _ in world_gap_point.receiver_certificates} == {
        f.fragment_id for f in journal.source_surface.fragments}
    assert all(c.donor_omega == (0., 0., 0.) for _, c in world_gap_point.receiver_certificates)
    start = evaluate_native_domain_journal_world_point(world_gap_point, Fraction(0))
    finish = evaluate_native_domain_journal_world_point(world_gap_point, Fraction(1))
    assert start.raw_gap_status == "occupied"
    assert finish.raw_gap_status == "raw_vacant"
    assert all(q.occupancy == "outside" for _, q in finish.receiver_queries)
    for query in (start, finish):
        saved = native_domain_world_point_query_to_dict(query)
        assert len(saved["receiver_queries"]) == 8
        assert saved["actual_passive_archive_vacancy_proved"] is False
        assert saved["newborn_material_created"] is False
        assert saved["admission_authorized"] is False
        assert saved["global_inventory_updated"] is False


@pytest.mark.parametrize("damage", ["occupied", "vacancy", "birth", "float_time", "missing_receiver"])
def test_fixed_world_query_serializer_cannot_grant_material_vacancy_or_hot_birth(world_gap_point, damage):
    from tectonics.geometric_native_domain_journal import (
        evaluate_native_domain_journal_world_point, native_domain_world_point_query_to_dict,
    )
    query = evaluate_native_domain_journal_world_point(world_gap_point, Fraction(1))
    if damage == "occupied":
        forged = replace(query, raw_gap_status="occupied")
    elif damage == "vacancy":
        forged = replace(query, actual_passive_archive_vacancy_proved=True)
    elif damage == "birth":
        forged = replace(query, newborn_material_created=True)
    elif damage == "float_time":
        forged = replace(query, time_fraction=1.)
    elif damage == "missing_receiver":
        forged = replace(query, receiver_queries=query.receiver_queries[:-1])
    with pytest.raises((TypeError, ValueError)):
        native_domain_world_point_query_to_dict(forged)


@pytest.fixture(scope="module")
def owner_handover_point(actual, journal):
    from tectonics.geometric_native_domain_journal import certify_native_domain_journal_owner_point
    # Derive the transverse turning scale from the literal first two Taylor
    # coefficients of Rp^-1 Rr n. This fixture uses no saved journal event.
    def cross(a, b):
        return tuple(a[j]*b[k]-a[k]*b[j] for j, k in ((1, 2), (2, 0), (0, 1)))
    duration = Fraction.from_float(actual["dt"])
    d = tuple(Fraction.from_float(value)*duration for value in actual["omega"][1])
    r = tuple(Fraction.from_float(value)*duration for value in actual["omega"][0])
    def coefficients(normal):
        first = cross(tuple(a-b for a, b in zip(r, d)), normal)
        dd, rr, dr = cross(d, cross(d, normal)), cross(r, cross(r, normal)), cross(d, cross(r, normal))
        second = tuple((a+b)/2-c for a, b, c in zip(dd, rr, dr))
        return first, second
    x_first, _ = coefficients((-1, 0, 0))
    z_first, z_second = coefficients((0, 0, -1))
    turning = -z_first[1]/(2*z_second[1])
    point = (x_first[1]*turning/2, Fraction(1), -z_first[1]**2/(8*z_second[1]))
    exact_ray = NativeRay("seed", _primitive(point))
    assert all(_side(exact_ray, support) > 0
               for support in actual["parents"]["octant:6"].geometry.support_planes)
    return certify_native_domain_journal_owner_point(journal, "octant:6", 0, point), turning


def test_receiver_material_boundary_does_not_create_a_false_owner_contact_exit(owner_handover_point):
    from tectonics.geometric_native_domain_journal import evaluate_native_domain_journal_owner_point
    chronology, turning = owner_handover_point
    certificates = dict(chronology.receiver_certificates)
    assert set(certificates) == {"octant:0", "octant:1", "octant:2", "octant:3"}
    assert len(certificates["octant:2"].occupancy_events) == 1
    returning = certificates["octant:2"].occupancy_events[0]
    assert returning.kind == "entry"
    assert turning < returning.time_fraction.lo
    query = evaluate_native_domain_journal_owner_point(chronology, returning.time_fraction.midpoint)
    individual = {identity: q.occupancy for identity, q in query.receiver_queries}
    assert individual["octant:2"] == individual["octant:3"] == "event_window"
    assert "inside" not in individual.values()
    # Both signs and exact zero of the common +/-Z plane leave the owner
    # covered by material 2 or 3; this is a logical union, not a chosen root.
    assert query.contact_occupancy == "inside"
    assert query.boolean_sign_assignments_checked == 3
    assert query.new_fault_inferred_from_internal_boundary is False
    assert query.material_loss_decided is False
    assert query.admission_authorized is False
    assert evaluate_native_domain_journal_owner_point(chronology, Fraction(0)).contact_occupancy == "outside"


def test_owner_query_serializer_replays_shared_sign_coverage_without_inventing_a_fault(owner_handover_point):
    from tectonics.geometric_native_domain_journal import (
        evaluate_native_domain_journal_owner_point, native_domain_owner_point_query_to_dict,
    )
    chronology, turning = owner_handover_point
    query = evaluate_native_domain_journal_owner_point(chronology, turning)
    raw = native_domain_owner_point_query_to_dict(query)
    assert raw["contact_occupancy"] == "inside"
    assert len(raw["receiver_queries"]) == 4
    assert raw["new_fault_inferred_from_internal_boundary"] is False
    for forged in (replace(query, contact_occupancy="outside"),
                   replace(query, boolean_sign_assignments_checked=float(query.boolean_sign_assignments_checked)),
                   replace(query, new_fault_inferred_from_internal_boundary=True)):
        with pytest.raises((TypeError, ValueError)):
            native_domain_owner_point_query_to_dict(forged)


def test_fixed_world_initial_internal_boundary_is_covered_without_birth(journal):
    from tectonics.geometric_native_domain_journal import (
        certify_native_domain_journal_world_point, evaluate_native_domain_journal_world_point,
        native_domain_world_point_query_to_dict,
    )
    chronology = certify_native_domain_journal_world_point(journal, (-1, 1, 0))
    query = evaluate_native_domain_journal_world_point(chronology, Fraction(0))
    individual = {identity: q.occupancy for identity, q in query.receiver_queries}
    assert individual["octant:2"] == individual["octant:3"] == "event_window"
    assert "inside" not in individual.values()
    assert query.raw_gap_status == "occupied"
    assert query.boolean_sign_assignments_checked == 1
    raw = native_domain_world_point_query_to_dict(query)
    assert raw["actual_passive_archive_vacancy_proved"] is False
    assert raw["newborn_material_created"] is False
    assert raw["admission_authorized"] is False
