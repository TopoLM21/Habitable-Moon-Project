"""Independent proof rejection beyond saved hashes and producer replay.

The auditor imports only standard-library mathematics. Production creates
fixtures here; all mutations are resealed and must still fail the independent
original-source, analytic root partition, or numerical geometry checks.
"""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path

import pytest

from analysis.geometric_native_full_domain_validation.independent_domain_audit import (
    _memory_payloads, audit_certificate, audit_endpoint_geometry, audit_query,
    audit_source_binding, audit_swept_domain, audit_swept_only_material_patch,
    digest, matrix_series, rational,
)
from tectonics.geometric_native_domain_journal import (
    initialize_native_domain_journal, native_domain_journal_to_dict,
)
from tectonics.geometric_native_material_history import make_native_material_memory_reference
from tectonics.geometric_native_point_chronology import (
    certify_native_point_chronology, evaluate_native_point_chronology,
    native_point_chronology_query_to_dict, native_point_chronology_to_dict,
)
from tectonics.geometric_native_surface import native_surface_from_dict

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "analysis/geometric_native_contact_event_validation/time_support_release_01"


def reseal(raw, field):
    raw[field] = digest({key: value for key, value in raw.items() if key != field})
    return raw


@pytest.fixture(scope="module")
def actual():
    initial, candidate, support = (json.loads((SOURCE/(name+".json")).read_bytes())
                                   for name in ("initial", "force_candidate", "constitutive_time_support"))
    original = initial["current_surface"]
    surface = native_surface_from_dict(original)
    omega = dict(candidate["omega_rad_per_myr"])
    duration = candidate["admission"]["dt_myr"]
    before, after = _memory_payloads(initial, candidate, support)
    reference = make_native_material_memory_reference(before, after, constitutive_path_sha256=support["path_sha256"])
    raw = native_domain_journal_to_dict(initialize_native_domain_journal(surface,
        {owner: tuple(vector) for owner, vector in omega.items()}, duration,
        memory_reference=reference))
    fragments = {row.fragment_id: row for row in surface.fragments}
    return dict(original=original, initial=initial, candidate=candidate, support=support,
                journal=raw, fragments=fragments, omega=omega, duration=duration)


def bind(actual, journal):
    return audit_source_binding(journal, actual["original"], actual["initial"],
                                actual["candidate"], actual["support"])


def point_certificate(actual, point, *, equal_motion=False):
    donor = tuple(actual["omega"][1])
    receiver = donor if equal_motion else tuple(actual["omega"][0])
    return native_point_chronology_to_dict(certify_native_point_chronology(
        actual["fragments"]["octant:6"].geometry, actual["fragments"]["octant:2"].geometry,
        donor, receiver, actual["duration"], point))


@pytest.fixture(scope="module")
def ordinary(actual):
    duration = rational(actual["duration"])
    d = tuple(rational(x)*duration for x in actual["omega"][1])
    r = tuple(rational(x)*duration for x in actual["omega"][0])
    first_y = -matrix_series(d, r, 1)[1][1][0]
    return point_certificate(actual, (first_y/2, Fraction(1), Fraction(-1)))


@pytest.fixture(scope="module")
def full_swept(actual):
    from tectonics.geometric_native_swept_domain import certify_native_swept_domain, native_swept_domain_to_dict
    typed = certify_native_swept_domain(actual["fragments"]["octant:6"].geometry,
        actual["fragments"]["octant:2"].geometry,
        tuple(actual["omega"][1]), tuple(actual["omega"][0]), actual["duration"])
    return typed, native_swept_domain_to_dict(typed)


@pytest.fixture(scope="module")
def swept_only(actual, full_swept):
    from analysis.geometric_native_full_domain_validation.run_full_domain_control import swept_only_material_patch
    return swept_only_material_patch(actual["fragments"]["octant:5"], actual["fragments"]["octant:1"],
        tuple(actual["omega"][1]), tuple(actual["omega"][0]), actual["duration"], full_swept[0])


def test_authentic_full_native_source_and_all_tiny_positive_endpoint_domains_pass(actual):
    journal = actual["journal"]
    binding = bind(actual, journal)
    result = audit_endpoint_geometry(journal, binding, actual["candidate"])
    assert len(binding["parents"]) == 8
    assert result["endpoint_pair_count"] == 5
    assert result["raw_vacancy_count"] == 7
    assert result["all_raw_vacancy_domains_numerically_reconstructed_against_all_original_moved_footprints"]
    assert result["exact_native_set_partition_independently_recomputed"] is False
    assert min(rational(p["area_bounds_sr"]["lo"]) for p in journal["endpoint_pairs"]) < Fraction(1, 10**58)


def test_resealed_projected_motion_is_rejected_against_the_authentic_force_candidate(actual):
    changed = deepcopy(actual["journal"])
    changed["omega_rad_per_myr"][0][1][0] = 0.
    reseal(changed, "journal_sha256")
    with pytest.raises(ValueError, match="projected|Euler"):
        bind(actual, changed)


def test_resealed_counterfeit_original_material_is_rejected_against_external_source(actual):
    changed = deepcopy(actual["journal"])
    changed["source_surface"]["fragments"][0]["parcel"]["material_id"] = "counterfeit-material"
    changed["source_surface_sha256"] = digest(changed["source_surface"])
    reseal(changed, "journal_sha256")
    with pytest.raises(ValueError, match="authentic original material"):
        bind(actual, changed)


def test_resealed_counterfeit_memory_is_rejected_against_external_before_candidate(actual):
    changed = deepcopy(actual["journal"])
    reference = changed["memory_reference"]
    reference["before_payload"]["cumulative_source_work_j"] += 1.
    reference["before_sha256"] = digest(reference["before_payload"])
    reseal(changed, "journal_sha256")
    with pytest.raises(ValueError, match="external mechanical memory"):
        bind(actual, changed)


def test_missing_tiny_endpoint_pair_cannot_be_hidden_by_resealing(actual):
    changed = deepcopy(actual["journal"])
    changed["endpoint_pairs"].pop()
    reseal(changed, "journal_sha256")
    with pytest.raises(ValueError, match="endpoint overlap"):
        audit_endpoint_geometry(changed, bind(actual, changed), actual["candidate"])


def test_missing_raw_vacancy_and_matching_reservation_fail_independent_all_cover_subtraction(actual):
    changed = deepcopy(actual["journal"])
    vacancy = changed["raw_vacancies"].pop()
    changed["endpoint_hot_birth_reservations"] = [row for row in changed["endpoint_hot_birth_reservations"]
                                                if row["vacancy_sha256"] != vacancy["vacancy_sha256"]]
    reseal(changed, "journal_sha256")
    with pytest.raises(ValueError, match="missing raw vacancy coverage"):
        audit_endpoint_geometry(changed, bind(actual, changed), actual["candidate"])


def test_resealed_pullback_with_one_projected_component_is_rejected(actual):
    changed = deepcopy(actual["journal"])
    changed["endpoint_pairs"][0]["donor_a_original_domain"]["pose"][-1]["omega_rad_per_myr"][0] = 0.
    reseal(changed["endpoint_pairs"][0], "pair_sha256")
    reseal(changed, "journal_sha256")
    with pytest.raises(ValueError, match="inverse owner rotation"):
        audit_endpoint_geometry(changed, bind(actual, changed), actual["candidate"])


def test_independent_ordinary_full_receiver_proof_has_one_authentic_entry(ordinary):
    proof = audit_certificate(ordinary)
    assert proof["whole_interval_root_count_complete"]
    assert len(proof["roots"]) == len(proof["occupancy_events"]) == 1
    assert proof["roots"][0]["kind"] == "entry"
    assert Fraction(49, 100) < proof["roots"][0]["low"] < proof["roots"][0]["high"] < Fraction(51, 100)


def test_resealed_removed_proof_leaf_is_rejected_without_producer_replay(ordinary):
    changed = deepcopy(ordinary)
    plane = next(p for p in changed["plane_chronologies"] if any(c["status"] == "unique_root" for c in p["cells"]))
    plane["cells"].pop(next(i for i, c in enumerate(plane["cells"]) if c["status"] == "unique_root"))
    reseal(changed, "payload_sha256")
    with pytest.raises(ValueError, match="partition|beginning|end"):
        audit_certificate(changed)


def test_resealed_hidden_root_cannot_be_relabelled_as_a_locked_no_root_leaf(ordinary):
    changed = deepcopy(ordinary)
    cell = next(c for p in changed["plane_chronologies"] for c in p["cells"] if c["status"] == "unique_root")
    cell.update(status="no_root", sign=1, root_fraction=None)
    reseal(changed, "payload_sha256")
    with pytest.raises(ValueError, match="hidden point root"):
        audit_certificate(changed)


def test_root_window_query_keeps_unknown_occupancy_without_sampled_grants(actual, ordinary):
    from tectonics.geometric_native_point_chronology import native_point_chronology_from_dict
    typed = native_point_chronology_from_dict(ordinary)
    event, = typed.occupancy_events
    instant = event.time_fraction.midpoint
    query = native_point_chronology_query_to_dict(evaluate_native_point_chronology(typed, instant))
    proof = audit_certificate(ordinary)
    assert audit_query(query, instant, proof, ordinary) == "event_window"
    changed = deepcopy(query)
    changed["occupancy"] = "inside"
    with pytest.raises(ValueError, match="unknown event window"):
        audit_query(changed, instant, proof, ordinary)


def test_masked_zero_time_receiver_crossing_is_retained_as_raw_evidence(actual):
    raw = point_certificate(actual, (Fraction(1), Fraction(1), Fraction(0)))
    proof = audit_certificate(raw)
    assert proof["initial_boundaries"]
    assert raw["initial_occupancy"] == "outside"
    assert raw["occupancy_events"] == []
    changed = deepcopy(raw)
    event = deepcopy(changed["plane_events"][0])
    event.update(sequence=1, plane_event_sequence=event["sequence"],
                 support_indices=[event["support_index"]], plane_event_sequences=[event["sequence"]])
    changed["occupancy_events"] = [event]
    reseal(changed, "payload_sha256")
    with pytest.raises(ValueError, match="masking receiver support"):
        audit_certificate(changed)


def test_ch9_identically_zero_support_is_verified_from_full_axes_not_a_boolean_claim(actual):
    raw = point_certificate(actual, (Fraction(1), Fraction(1), Fraction(0)), equal_motion=True)
    proof = audit_certificate(raw)
    assert proof["exact_coincident_support_count"] == 1
    changed = deepcopy(raw)
    changed["trajectory"]["point"][2] = [-1, 100]
    reseal(changed, "payload_sha256")
    with pytest.raises(ValueError, match="CH9 identity"):
        audit_certificate(changed)


def test_whole_blocking_pair_swept_cone_has_independent_analytic_and_exact_lineage_proofs(actual, full_swept):
    result = audit_swept_domain(full_swept[1], bind(actual, actual["journal"]), actual["journal"])
    assert result["complete_positive_time_swept_coefficient_inequalities_independently_proved"]
    assert result["native_cone_set_identity_independently_exact_recomputed"]
    assert result["physical_archive_mass_adopted"] is False


def test_resealed_counterfeit_terminal_cone_cannot_hide_behind_the_same_endpoint_area(actual, full_swept):
    changed = deepcopy(full_swept[1])
    changed["terminal_native_cone"]["supports"][0]["normal"][0] *= -1
    reseal(changed, "payload_sha256")
    with pytest.raises(ValueError, match="plane/ray lineage"):
        audit_swept_domain(changed, bind(actual, actual["journal"]), actual["journal"])


def test_independent_positive_owner_entry_exit_patch_is_missing_from_endpoint_material_loss(actual, swept_only):
    result = audit_swept_only_material_patch(swept_only, bind(actual, actual["journal"]), actual["journal"])
    assert result["whole_positive_original_material_rectangle_owner_entry_then_exit_independently_proved"]
    assert result["final_exclusion_from_every_original_receiver_owner_fragment_independently_proved"]
    assert result["full_rectangle_disjoint_from_actual_endpoint_material_loss"]
    assert result["global_endpoint_loss_equals_complete_swept_loss"] is False
    assert result["true_once_only_archive_material_vacancy_computed"] is False


def test_true_owner_entry_exit_witness_cannot_be_promoted_to_global_endpoint_equals_sweep(actual, swept_only):
    changed = deepcopy(swept_only)
    changed["global_endpoint_loss_equals_complete_swept_loss"] = True
    with pytest.raises(ValueError, match="adopted global archive"):
        audit_swept_only_material_patch(changed, bind(actual, actual["journal"]), actual["journal"])
