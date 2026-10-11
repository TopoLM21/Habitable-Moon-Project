"""Independent executable point proofs retain first-entry material ancestry.

The complete swept-area and material-ledger fixtures are independently audited
before these point checks.  A point does not stand in for a whole material
domain.  Saved checksums are recalculated for certificate attacks, while query
attacks must fail against the separately proved original trajectory.
"""
from copy import deepcopy
from fractions import Fraction
import json

import pytest

from analysis.geometric_native_first_entry_validation import independent_first_entry_audit as oracle
from analysis.geometric_native_first_entry_validation import independent_first_entry_point_audit as points
from test_geometric_native_first_entry_oracle import CONTROL, actual, material, sweep


def rational(value):
    value = Fraction(value)
    return [value.numerator, value.denominator]


def reseal(raw, field):
    raw[field] = points.digest({key: value for key, value in raw.items() if key != field})
    assert raw[field] == points.digest({key: value for key, value in raw.items() if key != field})
    return raw


@pytest.fixture(scope="module")
def contexts(actual, sweep, material):
    from tectonics.geometric_native_first_entry_material import (
        certify_native_first_entry_material_point,
        evaluate_native_first_entry_material_endpoint_point,
        evaluate_native_first_entry_material_point,
        native_first_entry_material_endpoint_query_to_dict,
        native_first_entry_material_from_dict,
        native_first_entry_material_point_chronology_to_dict,
        native_first_entry_material_point_query_to_dict,
    )
    source_files = (CONTROL / "point_certificates.json", CONTROL / "point_queries.json")
    original_bytes = {path: path.read_bytes() for path in source_files}
    saved_certificates = json.loads(original_bytes[source_files[0]])
    saved_queries = json.loads(original_bytes[source_files[1]])
    state = native_first_entry_material_from_dict(material["raw"],
        expected_journal=actual["typed_journal"],
        expected_original_mechanics_payload=actual["initial"]["native_underthrust_motion"]["initial_mechanics"])
    result = {}
    for label, parent in (("swept_only_owner_entry_exit", "octant:5"),
                          ("transverse_masked_main_entry", "octant:6")):
        original = next(row["certificate"] for row in saved_certificates if row["label"] == label)
        point = tuple(Fraction(*value) for value in original["trajectory"]["point"])
        context = certify_native_first_entry_material_point(state, parent, point)
        raw = native_first_entry_material_point_chronology_to_dict(context)
        proof = points.audit_point_context(raw, actual["journal"], sweep["raw"],
                                          material["raw"], actual["binding"])
        times = tuple(Fraction(*row["time_fraction"]) for row in saved_queries if row["label"] == label)
        queries = [native_first_entry_material_point_query_to_dict(
            evaluate_native_first_entry_material_point(context, time)) for time in times]
        for query in queries:
            assert points.audit_material_point_query(query, proof, actual["journal"], material["raw"]) == query
        endpoint = native_first_entry_material_endpoint_query_to_dict(
            evaluate_native_first_entry_material_endpoint_point(context))
        assert points.audit_endpoint_material_query(endpoint, proof, actual["journal"],
            sweep["raw"], material["raw"], actual["binding"]) == endpoint
        result[label] = dict(raw=raw, proof=proof, queries=queries, endpoint=endpoint)
    yield result
    assert all(path.read_bytes() == original_bytes[path] for path in source_files)


def reject_context(raw, actual, sweep, material):
    reseal(raw, "point_chronology_sha256")
    with pytest.raises(ValueError):
        points.audit_point_context(raw, actual["journal"], sweep["raw"],
                                  material["raw"], actual["binding"])


def reject_query(raw, context, actual, material):
    with pytest.raises(ValueError):
        points.audit_material_point_query(raw, context["proof"], actual["journal"], material["raw"])


def reject_endpoint(raw, context, actual, sweep, material):
    with pytest.raises(ValueError):
        points.audit_endpoint_material_query(raw, context["proof"], actual["journal"],
            sweep["raw"], material["raw"], actual["binding"])


def test_authentic_owner_exit_and_internal_receiver_switch_keep_once_only_archive(contexts):
    swept = contexts["swept_only_owner_entry_exit"]
    masked = contexts["transverse_masked_main_entry"]
    assert swept["proof"]["first_receivers"] == ["octant:1"]
    assert masked["proof"]["first_receivers"] == ["octant:3"]
    assert swept["proof"]["ancestry_status"] == masked["proof"]["ancestry_status"] == "proved"
    assert swept["proof"]["owner"]["first_window"] is not None
    assert masked["proof"]["owner"]["first_window"] is not None
    assert [row["material_allocation"] for row in swept["queries"]] == [
        "live", "archived", "archived", "archived", "archived"]
    assert [row["material_allocation"] for row in masked["queries"]] == [
        "live", "live", "archived", "archived", "archived"]
    assert swept["queries"][-1]["contact_occupancy"] == "outside"
    for context in (swept, masked):
        archived = [row for row in context["queries"] if row["material_allocation"] == "archived"]
        assert all(row["archive_debits"] == 1 for row in archived)
        assert len({row["cohort_id"] for row in archived}) == 1
        assert context["proof"]["leaf_count"] > 0 and context["proof"]["root_count"] > 0
    assert swept["endpoint"]["raw_coverage"] is True
    assert swept["endpoint"]["retained_material"] is False
    assert swept["endpoint"]["actual_material_vacancy"] is True
    assert swept["endpoint"]["fixed_original_parent_ids"] == ["octant:5"]
    assert swept["endpoint"]["birth_material_ids"]
    assert masked["endpoint"]["retained_material"] is True
    assert masked["endpoint"]["actual_material_vacancy"] is False


def test_resealed_owner_proof_cannot_omit_a_root_containing_time_cell(actual, sweep, material, contexts):
    changed = deepcopy(contexts["swept_only_owner_entry_exit"]["raw"])
    first = changed["first_entry_certificate"]
    cells = first["owner_plane"]["cells"]
    index = next(index for index, cell in enumerate(cells) if cell["status"] == "unique_root")
    cells.pop(index)
    reseal(first, "payload_sha256")
    reject_context(changed, actual, sweep, material)


def test_resealed_owner_archive_cannot_start_at_the_later_exit_root(actual, sweep, material, contexts):
    changed = deepcopy(contexts["swept_only_owner_entry_exit"]["raw"])
    first = changed["first_entry_certificate"]
    exit_event = next(row for row in first["owner_plane"]["events"] if row["kind"] == "exit")
    first["first_entry_time_fraction"] = deepcopy(exit_event["time_fraction"])
    reseal(first, "payload_sha256")
    reject_context(changed, actual, sweep, material)


def test_internal_receiver_reentry_cannot_debit_original_material_twice(actual, material, contexts):
    context = contexts["transverse_masked_main_entry"]
    changed = deepcopy(context["queries"][-1])
    changed["archive_debits"] = 2
    reject_query(changed, context, actual, material)


def test_owner_exit_cannot_restore_an_archived_point_to_live_material(actual, material, contexts):
    context = contexts["swept_only_owner_entry_exit"]
    changed = deepcopy(context["queries"][-1])
    assert changed["contact_occupancy"] == "outside"
    changed.update(material_allocation="live", archive_debits=0)
    reject_query(changed, context, actual, material)


def test_current_receiver_two_cannot_replace_first_receiving_material_three(actual, material, contexts):
    context = contexts["transverse_masked_main_entry"]
    changed = deepcopy(context["queries"][-1])
    assert changed["receiver_original_fragment_ids"] == ["octant:3"]
    changed["receiver_original_fragment_ids"] = ["octant:2"]
    changed["receiver_trace_ids"] = [row["trace_id"] for row in material["raw"]["trace_bindings"]
                                     if row["receiver_fragment_id"] == "octant:2"]
    reject_query(changed, context, actual, material)


def test_sub_ulp_material_age_cannot_be_rounded_back_to_original_age(actual, material, contexts):
    context = contexts["transverse_masked_main_entry"]
    changed = deepcopy(context["queries"][2])
    exact = Fraction(*changed["material_age_myr"])
    assert exact > 5 and float(exact) == 5.
    changed["material_age_myr"] = [5, 1]
    reject_query(changed, context, actual, material)


def test_point_query_cannot_promote_receiving_ancestry_to_physical_admission(actual, material, contexts):
    context = contexts["transverse_masked_main_entry"]
    changed = deepcopy(context["queries"][-1])
    changed.update(physical_pair_ancestry_authenticated=True,
                   physical_pair_admission_authorized=True, candidate_mechanics_adopted=True)
    reject_query(changed, context, actual, material)


def test_endpoint_query_cannot_project_away_a_retained_angular_component(actual, sweep, material, contexts):
    context = contexts["swept_only_owner_entry_exit"]
    changed = deepcopy(context["endpoint"])
    omega = changed["endpoint_ray"]["rotations"][0]["omega_rad_per_myr"]
    assert omega[0] != 0.
    omega[0] = 0.
    reject_endpoint(changed, context, actual, sweep, material)


def test_raw_covered_swept_only_point_cannot_erase_actual_vacancy_and_hot_fill(actual, sweep, material, contexts):
    context = contexts["swept_only_owner_entry_exit"]
    changed = deepcopy(context["endpoint"])
    assert changed["raw_coverage"] is True and changed["actual_material_vacancy"] is True
    changed.update(retained_material=True, actual_material_vacancy=False, birth_material_ids=[])
    reject_endpoint(changed, context, actual, sweep, material)


def test_globally_real_hot_material_cannot_fill_a_different_fixed_world_parent(actual, sweep, material, contexts):
    context = contexts["swept_only_owner_entry_exit"]
    changed = deepcopy(context["endpoint"])
    foreign = next(row for row in material["raw"]["endpoint_vacancies"]
                   if row["newborn"] is not None and row["fixed_original_parent_id"] != "octant:5")
    changed["fixed_original_parent_ids"] = [foreign["fixed_original_parent_id"]]
    changed["birth_material_ids"] = [foreign["newborn"]["material_id"]]
    reject_endpoint(changed, context, actual, sweep, material)


def test_positive_tiny_raw_vacancy_area_has_independent_native_ray_proof(actual):
    vacancy = min(actual["journal"]["raw_vacancies"], key=lambda row: Fraction(*row["area_bounds_sr"]["hi"]))
    proved = oracle.polygon_area_bounds(oracle.encoded(vacancy["geometry"]))
    oracle.enclosing(vacancy["area_bounds_sr"], proved, "authentic tiny raw vacancy")
    assert 0 < proved[0] <= proved[1] < Fraction(1, 10**40)


def test_counterfeit_positive_area_cannot_replace_a_true_tiny_world_vacancy(actual):
    vacancy = min(actual["journal"]["raw_vacancies"], key=lambda row: Fraction(*row["area_bounds_sr"]["hi"]))
    proved = oracle.polygon_area_bounds(oracle.encoded(vacancy["geometry"]))
    changed = {"lo": [1, 10**100], "hi": [1, 5 * 10**99]}
    with pytest.raises(ValueError, match="whole-domain enclosure"):
        oracle.enclosing(changed, proved, "counterfeit positive tiny vacancy")
