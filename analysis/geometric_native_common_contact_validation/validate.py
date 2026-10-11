"""Fresh retained-cut/actual-body ports with the immutable 841-source ancestry.

This gate establishes an additive initial geometry and field component only.
It does not reinterpret old contact springs or admit a new physical interval.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import importlib
import importlib.util
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.geometric_native_common_reference_validation import validate as previous
from analysis.geometric_native_finite_equilibrium_validation.validate import _purge_added_source_modules

RELEASE = ROOT / "analysis/geometric_native_common_reference_validation/release_validation_01"
EVIDENCE = RELEASE.parent
EXPECTED_REPORT = "9898f31f796006a7791140bf1c290c0e1605555a12e476e1585ce3f5992a743f"
EXPECTED_REPORT_SEAL = "5b86eb947fd07789ea4db0b13419bad4f22b38a4cc8ef2ddb1b60ef2f5ae6997"
EXPECTED_NATIVE = "0a8880d58a849cfeabf3f80aa6694c282eff6c24ce646740fb89b70a30320c40"
EXPECTED_PACKET = "dc8f820dbd926c53223a5c141a10296d1c7135efc6bf1077763785866dda9275"
EXPECTED_SOURCE_MAP = "dc109894fb7fb219b38a617cb1212f31dd7f758316658b5bdaba9335afaa6c35"
PHYSICAL_SCOPE = "COMMON_CONTACT_GEOMETRY_FIELD_COMPONENT_ONLY"
ADDED_SOURCE_PATHS = frozenset((
    "tectonics/geometric_native_common_contact_cuts.py",
    "tectonics/geometric_native_common_contact_ports.py",
    "tests/test_geometric_native_common_contact_cuts.py",
    "tests/test_geometric_native_common_contact_ports.py",
    "analysis/geometric_native_common_contact_validation/native_control.py",
    "analysis/geometric_native_common_contact_validation/validate.py",
))
STANDALONE_EVIDENCE = {
    "frozen_source_map_841.json": "89beb81ef3d1e9bc10ad37e08d23f1134b5b8af8f6c0922f364649db31e32345",
    "independent_audit.json": "e28bcca1a1031b72b0561ff46e2006147886921926cc1972c294a723c5351eed",
    "independent_audit_source.txt": "17420e0c5d81fcd6ee5cc56355f1a1486793a5247d2b6acd6b71e35b40c47c99",
    "source_design_audit.json": "e86f8c2070fc852e51475fd0c63a7c813865e76d60229c45821e5204b34e1e46",
    "source_design_audit_source.txt": "eebde91cc9098d1c780dae011c2b4dbcef698932188e9adf65f95eaffef21ba7",
    "retained_original_contact_cut_audit.json": "7ac3cd74013a20d7a06e95eccec4e6ceb3db412fb197ec3ed559c5af43108200",
    "retained_original_contact_cut_audit_source.txt": "4f7647fe1e2957fb2d1a37cf923d175dc880c167fd960907555bf845902b1a48",
    "contact_history_audit.json": "6da68ccda2089921fbe197293a05e4f84c3ea83fd2abb648b98d20f9b1813901",
    "contact_history_audit_source.txt": "c02dcd2f0c62795aeb4e15a35a3d32da7190ddb71033b597d324b2c459651d6f",
    "hot_mean_load_dependency_audit.json": "ed5052e3d7a132651abc34de9226e4456cdf03ed64a77240e811c5c78ae15cbc",
    "hot_mean_load_dependency_audit_source.txt": "950e90e7564f653b2d1ad1f61d4d1e1ddb82b3b44723002a15fcfbcd3cbc20ea",
    "retained_c6_terminal_trim_math_review.txt": "5580f7d22fd4e52eac49fe9156e594a720d0346a287685941e53c473c44492c7",
    "retained_original_contact_cut_next_contract.txt": "453085e1f4fe177233bc1336c7832670584306ba004c4fbc178dd2dc22abd9a4",
}
CURRENT_SOURCE_REVIEW_EVIDENCE = {
    "baseline_841_audit_source.txt": "f3d55b7c1541567a59c013a38f7f89bc949436afaca64ca56d629f3e34a7fdca",
    "baseline_841_audit.json": "20f89fcd71b459dedd4b78b41b314749811c5a99e5ca4ad163beb7586a2fd0f6",
    "source_geometry_peer_source.txt": "0d885b3ab8a9ee9e46be02b5641d9bca0ddb641b98a10222b0aa0e9c39e492ce",
    "source_geometry_peer.json": "569093c5d72177a1a4914a73ce92863088cd41d4a3d49e1e08b0383a00e2609a",
    "contact_reference_source_review_source.txt": "331203e7077e272e3040777f8b6a8e3aa2cf643f7708ceb7ddfbdc32d2699f04",
    "contact_reference_source_review.json": "0ac66b47262bf4c051365f378006b15e51cb316c97cfe8fea1ef353975ab0f25",
}
FALSE_SCOPE = previous.FALSE_SCOPE + (
    "physical_contact_rest_reference_and_measure_certified",
    "original_node_states_assigned_to_new_cuts",
    "old_column_compliance_added_to_new_bulk",
    "finite_owner_whole_clock_motion_certified",
)
encoded, digest, value_digest = previous.encoded, previous.digest, previous.value_digest
read, sealed, source_map, parse_counts = previous.read, previous.sealed, previous.source_map, previous.parse_counts


def verify_prior():
    """Verify all immutable ancestors and the byte-pinned completed 841 stage."""
    inherited = previous.verify_prior()
    report = sealed(RELEASE / "report.json", "report_sha256")
    if (digest(RELEASE / "report.json") != EXPECTED_REPORT
            or read(RELEASE / "report.json").get("report_sha256") != EXPECTED_REPORT_SEAL
            or report.get("status") != "PASS"
            or report.get("pytest_counts") != {"passed": 36}
            or report.get("pytest_exit_code") != 0
            or report.get("physical_candidate_status") != previous.PHYSICAL_SCOPE
            or report.get("source_count") != 841
            or report.get("source_map_before_sha256") != EXPECTED_SOURCE_MAP
            or report.get("source_map_after_sha256") != EXPECTED_SOURCE_MAP
            or report.get("changed_python_paths") != []
            or report.get("prior_artifacts_and_python_sources_preserved_after") is not True
            or set(report.get("added_python_sources", {})) != previous.ADDED_SOURCE_PATHS
            or set(inherited) & set(report["added_python_sources"])):
        raise ValueError("The completed common-reference 841 source epoch differs")
    inherited.update(report["added_python_sources"])
    if len(inherited) != 841 or value_digest(inherited) != EXPECTED_SOURCE_MAP:
        raise ValueError("The immutable 841-source map differs")
    for name, expected in inherited.items():
        if digest(ROOT / name) != expected:
            raise ValueError("An original Python source changed: " + name)
    previous._verify_outputs(RELEASE, report, 4)
    if {str(p.relative_to(RELEASE)) for p in RELEASE.rglob("*") if p.is_file()} != set(report["outputs"]) | {"report.json"}:
        raise ValueError("A file was added to or removed from the previous sealed release")
    if (report["outputs"]["native_control/report.json"]["sha256"] != EXPECTED_NATIVE
            or report["outputs"]["native_control/source_packet.json"]["sha256"] != EXPECTED_PACKET
            or report.get("accepted_original_native_intervals") != 1
            or report.get("accepted_new_native_intervals") != 0
            or report.get("original_point_1_remaining_work_blocks") != 1
            or any(report.get(name) is not False for name in previous.FALSE_SCOPE)):
        raise ValueError("The prior common component scope gained physical acceptance")
    for name, expected in STANDALONE_EVIDENCE.items():
        if digest(EVIDENCE / name) != expected:
            raise ValueError("Immutable standalone common evidence changed: " + name)
    if read(EVIDENCE / "frozen_source_map_841.json") != inherited:
        raise ValueError("The recorded 841 source map differs from its ancestry")
    for name, expected in CURRENT_SOURCE_REVIEW_EVIDENCE.items():
        if digest(Path(__file__).resolve().parent / name) != expected:
            raise ValueError("The current standalone source review changed: " + name)
    audit = sealed(EVIDENCE / "independent_audit.json", "audit_packet_sha256")
    if (audit.get("status") != "PASS" or audit.get("fresh_tests") != 36
            or audit.get("final_python_sources") != 841
            or audit.get("source_map_canonical_sha256") != EXPECTED_SOURCE_MAP
            or audit.get("wrapper_raw_sha256") != EXPECTED_REPORT
            or audit.get("native_raw_sha256") != EXPECTED_NATIVE
            or audit.get("source_packet_raw_sha256") != EXPECTED_PACKET
            or audit.get("material_admission_authorized") is not False
            or audit.get("second_certified_tensor_quadrature_performed") is not False):
        raise ValueError("The independently sealed completed common audit differs")
    return inherited


def _check_native_control(out):
    from analysis.geometric_native_common_contact_validation import native_control as control
    from tectonics.geometric_native_pose import FractionInterval as I
    from tectonics.geometric_native_normal_support import load_native_normal_support_source
    from tectonics.geometric_native_common_reference import native_common_reference_from_dict, native_common_reference_to_dict
    from tectonics import geometric_native_common_contact_cuts as cuts_api
    from tectonics import geometric_native_common_contact_ports as ports_api

    native_path = out / "native_control/report.json"
    packet_path = out / "native_control/source_packet.json"
    native = sealed(native_path, "report_sha256")
    packet = sealed(packet_path, "packet_sha256")
    if (native.get("status") != "PASS" or native.get("physical_candidate_status") != PHYSICAL_SCOPE
            or native.get("source_packet_sha256") != read(packet_path)["packet_sha256"]
            or packet.get("authenticated_common_wrapper_raw_sha256") != EXPECTED_REPORT
            or packet.get("authenticated_common_source_packet_raw_sha256") != EXPECTED_PACKET
            or packet.get("accepted_input_unchanged") is not True):
        raise ValueError("The new contact component lacks its immutable 841 ancestry")
    source = load_native_normal_support_source()
    if (packet.get("complete_source_dto_json") != [list(x) for x in source.complete_source_dto_json]
            or encoded(packet.get("saved_joint_reference_field")) != source.saved_field_json.encode()
            or packet.get("source_snapshot_sha256") != source.source_snapshot_sha256):
        raise ValueError("A complete historical DTO, saved field or source snapshot changed")
    reference = native_common_reference_from_dict(packet["common_reference"], source=source)
    if native_common_reference_to_dict(reference) != packet["common_reference"]:
        raise ValueError("The complete common initial reference differs on source replay")
    cuts = cuts_api.common_contact_cuts_from_dict(packet["cut_set"], reference=reference)
    ports = ports_api.native_common_contact_ports_from_dict(packet["ports"], reference)
    if ports.cuts != cuts or len(cuts.cuts) != 6 or len(ports.ports) != 6:
        raise ValueError("The six ports substituted their true material boundary cuts")
    expected_pairs = {
        ("octant:5", "octant:1"), ("octant:5", "octant:3"),
        ("octant:6", "octant:2"), ("octant:7", "octant:2"), ("octant:7", "octant:3"),
    }
    cold = tuple(c for c in cuts.cuts if c.receiver_material_kind == "original_live")
    curve = tuple(c for c in cuts.cuts if c.kind == "true_curved_envelope")
    if ({(c.donor_original_parent_id, c.receiver_original_parent_id) for c in cold} != expected_pairs
            or len(cold) != 5 or len(curve) != 1
            or curve[0].receiving_trace_id is not None
            or curve[0].receiver_material_kind != "hot_birth"
            or sum(c.kind == "implicit_terminal_trim" for c in cuts.cuts) != 1):
        raise ValueError("The complete cold arrangement or separate true HOT curve differs")
    for cut in cuts.cuts:
        if (not cut.true_source_boundary_of_both_actual_masks_certified
                or cut.old_mechanical_atom_ids_assigned
                or cut.constitutive_rest_contact_reference_recovered
                or cut.physical_material_admission_authorized
                or cut.common_physical_radius_cubed_bounds_m3.lo <= 0):
            raise ValueError("A source boundary gained an unsupported physical contact claim")
    whole = tuple(cuts_api.evaluate_common_contact_cut(reference, cut, precision=160) for cut in cuts.cuts)
    if packet.get("whole_cut_enclosures") != [cuts_api.common_contact_cut_bounds_to_dict(row) for row in whole]:
        raise ValueError("A full cut/depth enclosure changed")
    if any(not g.reference_faces_match_at_every_common_physical_radius
           or g.area_density_bounds_m2_per_parameter_per_material_m.lo <= 0 for g in whole):
        raise ValueError("Common-radius gluing or true boundary regularity failed")
    expected_history = control.source_history_records(source)
    if any(packet.get(k) != v for k, v in expected_history.items()):
        raise ValueError("The original sixteen states, histories, areas, work or clocks changed")
    expected_probe_keys = {(p.cut.cut_id, sample) for p in ports.ports for sample in (1, 2)}
    rows = packet.get("strict_interior_body_port_probes", ())
    if len(rows) != 12 or {(r.get("cut_id"), r.get("sample")) for r in rows} != expected_probe_keys:
        raise ValueError("Exactly two true interior field probes per cut are required")
    by_id = {p.cut.cut_id: p for p in ports.ports}
    mode_columns = owner_columns = cross_columns = 0
    for row in rows:
        port = by_id[row["cut_id"]]
        arguments = control.probe_arguments(port.cut, row["sample"])
        if row.get("arguments") != control.payload(arguments) or row.get("nonzero_probe_is_a_mixed_force_solution") is not False:
            raise ValueError("A literal component probe was substituted for a force solution")
        for label, expected_d, expected_r in (
                ("response", arguments["donor_coordinates_m"], arguments["receiver_coordinates_m"]),
                ("zero_response", (F(0),)*7, (F(0),)*len(arguments["receiver_coordinates_m"]))):
            response = ports_api.native_common_contact_port_response_from_dict(row[label], reference)
            geometry = response.geometry
            if (response.port != port or response.donor_fields.coordinates_m != expected_d
                    or response.receiver_fields.coordinates_m != expected_r
                    or geometry.parameter_bounds != arguments["parameter_bounds"]
                    or geometry.common_physical_radius_cubed_bounds_m3 != arguments["physical_radius_cubed_bounds_m3"]
                    or not geometry.genuine_query_support_certified
                    or not geometry.every_point_of_parameter_enclosure_is_material
                    or not response.one_common_reference_point_used
                    or not response.owner_columns_are_infinitesimal_left_chart_at_displaced_sides
                    or not response.projection_frame_is_reference_geometry
                    or response.unresolved_parameter_bands_promoted_to_material
                    or response.contact_constitutive_law_inferred
                    or response.physical_material_admission_authorized):
                raise ValueError("An actual-body field response changed its source/coordinate/frame interpretation")
            for side in (response.donor_fields, response.receiver_fields):
                if (side.reference_snapshot_sha256 != reference.reference_snapshot_sha256
                        or len(side.mode_displacement_columns) != len(side.coordinate_order)
                        or len(side.mode_spatial_gradient_columns_per_m) != len(side.coordinate_order)):
                    raise ValueError("A field lost its authentic common reference or full modal derivative set")
            if label == "zero_response":
                if (response.displacement_jump_bounds_m != (I(0),)*3
                        or response.normal_jump_bounds_m != I(0)
                        or response.down_jump_bounds_m != I(0)
                        or response.angular_jump_bounds_m != I(0)):
                    raise ValueError("Independent interval reference points introduced an artificial rest jump")
            else:
                count = len(expected_d)+len(expected_r)
                if (len(response.mode_jump_columns) != count
                        or len(response.mode_jump_spatial_gradient_columns_per_m) != count
                        or len(response.mode_jump_parameter_derivative_columns) != count
                        or len(response.mode_jump_material_depth_derivative_columns) != count
                        or tuple(len(r) for r in response.projected_mode_jacobian_bounds) != (count,)*3):
                    raise ValueError("A mixed body-port modal Jacobian is incomplete")
                mode_columns += count
                owner_columns += len(response.owner_rotation_jump_columns_m)
                cross_columns += sum(len(r) for r in response.owner_body_mode_cross_derivative_columns)
    shared_keys = ("common_reference", "cut_set", "ports", "whole_cut_enclosures", "strict_interior_body_port_probes", *expected_history)
    if any(native.get(key) != packet.get(key) for key in shared_keys):
        raise ValueError("Native report and complete source packet differ")
    for key, expected in dict(actual_retained_cold_cut_count=5, actual_true_curved_cold_hot_cut_count=1,
            actual_common_body_port_count=6, strict_interior_body_port_probe_count=12,
            physical_full_radius_intersections_preserved=True, true_source_mask_boundary_lineage_preserved=True,
            old_sixteen_constitutive_histories_preserved=True, accepted_input_complete_typed_state_unchanged=True,
            accepted_original_native_intervals=1, accepted_new_native_intervals=0,
            original_point_1_remaining_work_blocks=1).items():
        if native.get(key) != expected:
            raise ValueError("Contact component declaration differs: " + key)
    if (len(native.get("remaining_physical_conditions", [])) != 3
            or any(native.get(name) is not False for name in FALSE_SCOPE)):
        raise ValueError("The component gained equilibrium, constitutive law or material admission")
    return dict(physical_candidate_status=PHYSICAL_SCOPE,
        native_control_report_raw_sha256=digest(native_path), native_control_source_packet_raw_sha256=digest(packet_path),
        actual_retained_cold_cut_count=5, actual_true_curved_cold_hot_cut_count=1,
        actual_common_body_port_count=6, strict_interior_body_port_probe_count=12,
        authenticated_mode_jump_column_replay_count=mode_columns,
        authenticated_owner_left_rotation_column_replay_count=owner_columns,
        authenticated_owner_body_cross_derivative_replay_count=cross_columns,
        complete_historical_dtos_saved_field_states_and_clocks_preserved=True,
        accepted_original_native_intervals=1, accepted_new_native_intervals=0,
        original_point_1_remaining_work_blocks=1, **{name: False for name in FALSE_SCOPE})


def run(out, test_paths, control):
    tests = tuple(test_paths)
    expected_tests = {name for name in ADDED_SOURCE_PATHS if name.startswith("tests/")}
    if (len(tests) != 2 or set(tests) != expected_tests
            or any(not (ROOT / name).is_file() for name in tests)
            or control != "analysis.geometric_native_common_contact_validation.native_control:run"):
        raise ValueError("Exactly the two new test modules and this stage's callback are required")
    inherited = verify_prior()
    before = source_map()
    added = set(before) - set(inherited)
    if len(before) != 847 or added != ADDED_SOURCE_PATHS:
        raise ValueError("Exactly six additive retained-cut/port Python sources are required")
    removed_caches = []
    for name in sorted(added):
        cache = Path(importlib.util.cache_from_source(str(ROOT / name)))
        if cache.is_file():
            cache.unlink()
            removed_caches.append(str(cache.relative_to(ROOT)))
    own = str(Path(__file__).resolve().relative_to(ROOT))
    purged = _purge_added_source_modules(added - {own})
    importlib.invalidate_caches()
    if source_map() != before:
        raise ValueError("A Python source changed while preparing imports")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(format="native-common-contact-same-process-validation-1", status="FAIL",
        regression_validation_status="FAIL", physical_candidate_status="NOT_CHECKED",
        started_utc=datetime.now(timezone.utc).isoformat(), verified_prior_python_sources=841,
        verified_prior_artifacts=1633, verified_prior_common_outputs=4,
        verified_prior_standalone_evidence=dict(STANDALONE_EVIDENCE),
        verified_current_source_review_evidence=dict(CURRENT_SOURCE_REVIEW_EVIDENCE),
        prior_common_report_raw_sha256=EXPECTED_REPORT, prior_common_packet_raw_sha256=EXPECTED_PACKET,
        prior_source_map_sha256=EXPECTED_SOURCE_MAP, source_count=len(before),
        source_map_before_sha256=value_digest(before), added_python_sources={name: before[name] for name in sorted(added)},
        removed_added_source_bytecode_caches=removed_caches, purged_previously_imported_added_source_modules=purged,
        historical_tests_reexecuted=False, test_paths=list(tests), control=control,
        execution_mode="same_python_process", pytest_import_mode="importlib",
        full_genuine_source_reconstructed=False, inherited_force_certificate_for_placement=False,
        whole_native_interval_admitted_by_test_gate=False)
    try:
        import pytest
        test_started = time.perf_counter()
        old_directory = Path.cwd()
        try:
            os.chdir(ROOT)
            with (out / "pytest.log").open("x", buffering=1) as stream:
                with redirect_stdout(stream), redirect_stderr(stream):
                    code = int(pytest.main(["-q", "-p", "no:cacheprovider", "--import-mode=importlib", *tests]))
        finally:
            os.chdir(old_directory)
        summary, counts = parse_counts(out / "pytest.log")
        report.update(pytest_exit_code=code, pytest_summary=summary, pytest_counts=counts,
                      pytest_seconds=time.perf_counter()-test_started)
        if (code or counts.get("passed", 0) < 1 or any(counts.get(name, 0)
                for name in ("failed", "skipped", "xfailed", "xpassed", "error", "errors"))):
            raise RuntimeError("Fresh retained-cut/port tests failed")
        if source_map() != before:
            raise ValueError("A source changed during tests")
        module, function = control.split(":", 1)
        callback = getattr(importlib.import_module(module), function)
        control_started = time.perf_counter()
        with (out / "control_console.log").open("x", buffering=1) as stream:
            with redirect_stdout(stream), redirect_stderr(stream):
                callback(out / "native_control")
        report["native_control_seconds"] = time.perf_counter()-control_started
        report.update(_check_native_control(out))
        expected_outputs = {"pytest.log", "control_console.log", "native_control/report.json", "native_control/source_packet.json"}
        if {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()} != expected_outputs:
            raise ValueError("Exactly four declared output files are required")
        report["status"] = report["regression_validation_status"] = "PASS"
    except BaseException as error:
        report.update(failure_type=type(error).__name__, failure_message=str(error), failure_traceback=traceback.format_exc())
    finally:
        after = source_map()
        report["source_map_after_sha256"] = value_digest(after)
        report["changed_python_paths"] = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
        try:
            verify_prior()
            report["prior_artifacts_and_python_sources_preserved_after"] = True
        except BaseException as error:
            report["prior_artifacts_and_python_sources_preserved_after"] = False
            report["prior_after_error"] = str(error)
        if report["changed_python_paths"] or not report["prior_artifacts_and_python_sources_preserved_after"]:
            report["status"] = report["regression_validation_status"] = "FAIL"
        report["seconds"] = time.perf_counter()-started
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["outputs"] = {str(p.relative_to(out)): dict(sha256=digest(p), bytes=p.stat().st_size)
            for p in sorted(out.rglob("*")) if p.is_file()}
        report["report_sha256"] = value_digest(report)
        with (out / "report.json").open("xb") as stream:
            stream.write(encoded(report)+b"\n")
        print(encoded({key: report.get(key) for key in ("status", "physical_candidate_status", "pytest_counts", "seconds", "source_count", "changed_python_paths")}).decode(), flush=True)
    if report["status"] != "PASS":
        raise RuntimeError("Validation refused; see " + str(out / "report.json"))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("tests", nargs="+")
    args = parser.parse_args()
    run(args.out, args.tests, args.control)
