"""Independent full-vector checks of the genuine original donor rectangle."""
from copy import deepcopy
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
ORACLE_PATH = ROOT / "analysis/geometric_native_material_history_validation/independent_donor_sweep_audit.py"
SPEC = importlib.util.spec_from_file_location("independent_donor_sweep_test_oracle", ORACLE_PATH)
ORACLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ORACLE
SPEC.loader.exec_module(ORACLE)


@pytest.fixture(scope="module")
def probe():
    return json.loads((ROOT / "analysis/geometric_native_underthrust_validation/force_control_release_02/force_candidate_refusal_probe.json").read_bytes())


@pytest.fixture(scope="module")
def proof(probe):
    return ORACLE.certify_actual_rectangle(probe)


@pytest.fixture(scope="module")
def midpoint_report(proof):
    point = tuple(value.midpoint for value in proof["patch"])
    with localcontext() as context:
        context.prec = ORACLE.DECIMAL_PRECISION
        events = ORACLE.numerical_point_roots(proof, point)
    return dict(point_events_independent_rodrigues=[dict(events=events)])


@pytest.fixture(scope="module")
def production(probe):
    # This fixture alone imports production code. The independent auditor
    # below must reconstruct its evidence without any production imports.
    from tectonics.geometric_native_donor_sweep import (
        certify_native_donor_patch_sweep, native_donor_patch_sweep_to_dict,
    )
    from tectonics.geometric_native_polygons import native_polygon_from_dict
    initial = json.loads((ROOT / "analysis/geometric_native_contact_event_validation/time_support_release_01/initial.json").read_bytes())
    domains = {fragment["fragment_id"]: fragment["geometry"]
               for fragment in initial["current_surface"]["fragments"]}
    certificate = certify_native_donor_patch_sweep(
        native_polygon_from_dict(domains["octant:6"]),
        native_polygon_from_dict(domains["octant:2"]),
        tuple(probe["omega"]["1"]), tuple(probe["omega"]["0"]),
        probe["admission"]["dt_myr"])
    return native_donor_patch_sweep_to_dict(certificate)


@pytest.fixture(scope="module")
def material_bundle(production):
    from analysis.geometric_native_material_history_validation.run_material_history_control import memory_payloads
    from tectonics.geometric_native_donor_sweep import native_donor_patch_sweep_from_dict
    from tectonics.geometric_native_material_history import (
        make_native_material_memory_reference, initialize_native_material_history,
        native_material_history_to_dict,
    )
    from tectonics.geometric_native_surface import native_surface_from_dict
    base = ROOT / "analysis/geometric_native_contact_event_validation/time_support_release_01"
    initial, candidate, support = (json.loads((base/name).read_bytes()) for name in (
        "initial.json", "force_candidate.json", "constitutive_time_support.json"))
    before, after = memory_payloads(initial, candidate, support)
    memory = make_native_material_memory_reference(before, after,
        constitutive_path_sha256=support["path_sha256"])
    fragments = {item.fragment_id: item for item in native_surface_from_dict(initial["current_surface"]).fragments}
    history = initialize_native_material_history(fragments["octant:6"], fragments["octant:2"],
        native_donor_patch_sweep_from_dict(production),
        material_epoch_myr=initial["current_surface"]["time_myr"], memory_reference=memory)
    return native_material_history_to_dict(history), initial, candidate, support


def reseal(raw):
    raw.pop("payload_sha256", None)
    raw["payload_sha256"] = hashlib.sha256(ORACLE.encoded(raw)).hexdigest()
    return raw


def test_independent_proof_covers_positive_area_and_exactly_three_roots(proof):
    assert proof["area"].lo > 0
    assert proof["third"].lo > 0
    assert all(value.lo > 0 for value in proof["other_supports"].values())
    assert proof["three_roots_complete"]
    signs = tuple(1 if value.lo > 0 else -1 if value.hi < 0 else 0
                  for value in proof["signs"])
    assert signs == (-1, 1, -1, 1, 1)
    assert proof["patch"][0].lo > 0
    assert proof["patch"][1].lo > 0


def test_report_intervals_preserve_proofs_with_portable_json_integers(proof):
    raw = [ORACLE.interval_dto(value) for value in proof["signs"]]
    assert json.loads(ORACLE.encoded(raw)) == raw
    for dto, exact in zip(raw, proof["signs"]):
        saved = ORACLE.interval(dto)
        assert saved.lo <= exact.lo <= exact.hi <= saved.hi
        assert (saved.lo > 0) == (exact.lo > 0)
        assert (saved.hi < 0) == (exact.hi < 0)


def test_first_material_entry_is_strictly_after_initial_epoch(proof, midpoint_report):
    events = midpoint_report["point_events_independent_rodrigues"][0]["events"]
    roots = tuple(Decimal(event["fraction"]) for event in events)
    assert tuple(event["kind"] for event in events) == ("entry", "exit", "entry")
    assert 0 < roots[0] < roots[1] < roots[2] < 1
    assert all(abs(Decimal(event["residual"])) < Decimal("1e-430") for event in events)


def test_actual_production_windows_contain_independent_whole_patch_roots(proof, production, midpoint_report):
    report = deepcopy(midpoint_report)
    ORACLE.bind_production_certificate(production, proof, report)
    assert report["production_patch_exactly_matches_independent_rectangle"]
    assert report["production_complete_native_domains_match_original_force_candidate"]
    assert all(item["every_patch_point_bracketed"]
               for item in report["production_windows_independently_verified"])


def test_projecting_small_euler_component_loses_actual_material_proof(probe):
    altered = deepcopy(probe)
    for owner in ("0", "1"):
        altered["omega"][owner][0] = 0.0
    with pytest.raises(ValueError, match="original native rotation"):
        ORACLE.certify_actual_rectangle(altered)


def test_changing_source_plane_lineage_is_rejected_before_computing_a_patch(probe):
    altered = deepcopy(probe)
    candidate = next(item for item in altered["admission"]["actual_overlap_assessments"]
                     if (item["fragment_a"], item["fragment_b"]) == ("octant:2", "octant:6"))
    candidate["native_overlap"]["supports"][0]["rotations"][0]["dt_myr"] *= 2
    with pytest.raises(ValueError, match="original native rotation"):
        ORACLE.certify_actual_rectangle(altered)


@pytest.mark.parametrize("mutation", ("patch", "support", "area", "root_windows", "epoch"))
def test_resealed_geometric_or_chronology_forgery_fails_independent_audit(
        proof, production, midpoint_report, mutation):
    raw, report = deepcopy(production), deepcopy(midpoint_report)
    if mutation == "patch":
        raw["certificate"]["patch_x_bounds"]["lo"] = [0, 1]
    elif mutation == "support":
        raw["certificate"]["receiver_positive_bounds"][0] = dict(lo=[1, 1], hi=[2, 1])
    elif mutation == "area":
        raw["certificate"]["patch_area_bounds"] = dict(lo=[1, 1], hi=[2, 1])
    elif mutation == "root_windows":
        event = raw["certificate"]["event_windows"][0]
        event["time_fraction"]["hi"] = [1, 1]
        duration = Fraction.from_float(raw["trajectory"]["dt_myr"])
        event["time_myr"]["hi"] = [duration.numerator, duration.denominator]
    else:
        report["point_events_independent_rodrigues"][0]["events"][0]["fraction"] = "0"
    with pytest.raises(ValueError):
        ORACLE.bind_production_certificate(reseal(raw), proof, report)


def test_auditor_has_no_production_dependency():
    import ast
    tree = ast.parse(ORACLE_PATH.read_text())
    imports = tuple(node.module if isinstance(node, ast.ImportFrom)
                    else alias.name
                    for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
                    for alias in ([None] if isinstance(node, ast.ImportFrom) else node.names))
    assert not any(value and value.startswith("tectonics") for value in imports)


def test_independent_material_history_preserves_original_reservoirs_and_memory(
        material_bundle, production, proof):
    report = {}
    ORACLE.bind_material_history(*material_bundle, production, proof, report)
    result = report["local_history_independently_bound"]
    assert result["four_original_extensive_reservoirs_conserved_symbolically"]
    assert result["archive_transfer_count"] == 1
    assert result["mechanical_memory_and_work_are_exact_external_references"]
    assert not result["independent_force_or_constitutive_solve_performed"]


@pytest.mark.parametrize("mutation", ("work", "duplicate_transfer", "reservoir_ratio", "material_identity"))
def test_resealed_material_forgery_cannot_reset_or_duplicate_ancestry(
        material_bundle, production, proof, mutation):
    history, initial, candidate, support = deepcopy(material_bundle)
    if mutation == "work":
        history["memory_reference"]["before_payload"]["cumulative_source_work_j"] += 1
    elif mutation == "duplicate_transfer":
        history["events"][2]["operation"] = "live_to_passive_archive_once"
    elif mutation == "reservoir_ratio":
        history["shares"][0]["amount_bounds"] = dict(lo=[0, 1], hi=[0, 1])
    else:
        history["parent"]["parcel"]["material_id"] = "newborn"
    history.pop("history_sha256")
    history["history_sha256"] = hashlib.sha256(ORACLE.encoded(history)).hexdigest()
    with pytest.raises(ValueError):
        ORACLE.bind_material_history(history, initial, candidate, support, production, proof, {})
