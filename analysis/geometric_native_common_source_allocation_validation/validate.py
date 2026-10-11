"""Fresh four-source addition gate for whole-source accounting; old859 immutable."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import time
import traceback
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.geometric_native_common_contact_transfer_validation import validate as previous
from analysis.geometric_native_finite_equilibrium_validation.validate import _purge_added_source_modules
from analysis.geometric_native_common_source_allocation_validation import native_control as control_api

PHYSICAL_SCOPE = control_api.PHYSICAL_SCOPE
FALSE_SCOPE = control_api.FALSE_SCOPE
PARENT = Path(__file__).resolve().parent
OLD_PARENT = ROOT / "analysis/geometric_native_common_contact_transfer_validation"
OLD_MAP = "88f0d2e78a81712458beb61f21bbdfe0b7222467a0e3241e23757e3fac6bda7d"
ADDED_SOURCE_PATHS = frozenset((
    "tectonics/geometric_native_common_source_allocation.py",
    "tests/test_geometric_native_common_source_allocation.py",
    "analysis/geometric_native_common_source_allocation_validation/native_control.py",
    "analysis/geometric_native_common_source_allocation_validation/validate.py",
))
CURRENT_EVIDENCE = {
    "baseline_859_bootstrap_source.txt": "240127f21ee6077212795b512bdab2cb1cd83cf88f6f2293cfd8d1f44a91730d",
    "baseline_859_bootstrap.json": "e44d4f4512e090469fe2204f64f9f5d943e99447e57e1d99ec3c38bc0076e655",
    "baseline_859_reusable_source.txt": "778787b5231c6d03e1c87857038b52889077be0e37290a542fc88acda7c34836",
}
encoded, value_digest = previous.encoded, previous.value_digest
read, sealed, source_map, parse_counts = previous.read, previous.sealed, previous.source_map, previous.parse_counts


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def verify_prior():
    for name, expected in CURRENT_EVIDENCE.items():
        if digest(PARENT / name) != expected:
            raise ValueError("The immutable859 recovery evidence changed: " + name)
    name = "allocation_gate_stdlib859_readback"
    module = types.ModuleType(name); module.__file__ = str(PARENT / "baseline_859_reusable_source.txt")
    sys.modules[name] = module
    exec(compile(Path(module.__file__).read_bytes(), module.__file__, "exec"), module.__dict__)
    with redirect_stdout(io.StringIO()):
        baseline = module.run(ADDED_SOURCE_PATHS,
            output_path="/tmp/native_common_source_allocation_gate_prior_readback_20261006.json",
            bootstrap_source_path=PARENT / "baseline_859_bootstrap_source.txt",
            bootstrap_report_path=PARENT / "baseline_859_bootstrap.json")
    if (baseline.get("status") != "PASS_IMMUTABLE859_WITH_DECLARED863_SOURCE_ADDITIONS_ONLY"
            or baseline.get("baseline_source_count") != 859
            or baseline.get("all859_python_sources_unchanged") is not True
            or baseline.get("preserved_old1633_artifact_chain") is not True):
        raise ValueError("The complete immutable859 ancestry readback failed")
    inherited = read(OLD_PARENT / "frozen_source_map_859.json")
    if len(inherited) != 859 or value_digest(inherited) != OLD_MAP:
        raise ValueError("The preserved859 source map differs")
    return inherited


def _check_native_control(out):
    from fractions import Fraction as F
    from tectonics.geometric_native_pose import FractionInterval as I
    from tectonics.geometric_native_normal_support import load_native_normal_support_source
    from tectonics.geometric_native_common_reference import native_common_reference_from_dict, native_common_reference_to_dict
    from tectonics import geometric_native_common_contact_cuts as cuts_api
    from tectonics import geometric_native_common_source_allocation as api
    native = sealed(out / "native_control/report.json", "report_sha256")
    packet = sealed(out / "native_control/source_packet.json", "packet_sha256")
    if (native.get("status") != "PASS" or native.get("source_packet_sha256") != value_digest(packet)
            or packet.get("prior_859_wrapper_raw_sha256") != "1b5e1695f2ddfa8ea879619e91cde6b4cfb2c7c7c89e34db04148d8765f7f9ae"):
        raise ValueError("The whole-source output identities differ")
    source = load_native_normal_support_source()
    if (packet.get("complete_source_dto_json") != [list(item) for item in source.complete_source_dto_json]
            or encoded(packet.get("saved_joint_reference_field")) != source.saved_field_json.encode()
            or packet.get("source_snapshot_sha256") != source.source_snapshot_sha256):
        raise ValueError("The four original DTOs or saved128 source changed")
    reference = native_common_reference_from_dict(packet["common_reference"], source=source)
    if native_common_reference_to_dict(reference) != packet["common_reference"]:
        raise ValueError("The genuine common reference changed")
    cuts = cuts_api.common_contact_cuts_from_dict(packet["cut_set"], reference=reference)
    allocation = api.native_common_source_allocation_from_dict(packet["allocation"], reference=reference, cuts=cuts)
    if (allocation.provenance != control_api.PROVENANCE or allocation.precision != 96
            or {family.trace_id: family.depth_profile_coefficients for family in allocation.families}
                != control_api.declared_profiles(source)
            or allocation.source_current_node_epoch != "unaccepted_candidate"
            or (allocation.source_native_elapsed_time_s, allocation.source_memory_elapsed_time_s,
                allocation.source_contact_elapsed_time_s) != (F(1000000), F(1000000), F(2000000))):
        raise ValueError("A family depth profile, complete source clock or current epoch differs")
    if (len(allocation.families), len(allocation.boundaries), len(allocation.cells), len(allocation.original_atom_ledger)) != (4, 12, 8, 16):
        raise ValueError("All original source families, boundaries, supports and atom ledgers are required")
    memory_queries = packet.get("source_memory_queries", ())
    families = {family.trace_id: family for family in allocation.families}
    if (len(memory_queries) != 4 or {row.get("trace_id") for row in memory_queries} != set(families)
            or native.get("source_memory_queries_sha256") != value_digest(memory_queries)):
        raise ValueError("Every complete CURRENT source-memory family must be queried once")
    for row in memory_queries:
        family = families[row["trace_id"]]
        angle = I(0, family.source_computational_extent_radians)
        query = api.native_common_source_memory_query_from_dict(row["query"], allocation=allocation)
        if (row.get("computational_angle_bounds") != control_api.payload(angle)
                or query.trace_id != family.trace_id or query.computational_angle_bounds != angle
                or query.precision != 96 or query.source_current_memory_sha256 != allocation.source_current_memory_sha256
                or not query.literal_current_memory_used or not query.all_closed_current_piece_intersections_retained
                or query.current_memory_replaced_with_previous_or_dense_query
                or query.three_dimensional_material_memory_advection_certified
                or query.source_state_or_thermal_state_advanced or query.physical_material_admission_authorized):
            raise ValueError("A whole-family memory query lost actual current pieces, clocks or source scope")
    cold = {cut.cut_id: cut for cut in cuts.cuts if cut.receiving_trace_id is not None}
    rows = packet.get("density_queries", ())
    if (len(cold) != 5 or len(rows) != 10
            or {(row.get("cut_id"), row.get("sample")) for row in rows} != {(cut_id, sample) for cut_id in cold for sample in (1, 2)}
            or native.get("density_queries_sha256") != value_digest(rows)
            or native.get("allocation_binding_sha256") != allocation.binding_sha256):
        raise ValueError("The two genuine density queries for every allocated cold support differ")
    for row in rows:
        cut = cold[row["cut_id"]]
        arguments = control_api.query_arguments(cut, row["sample"])
        if row.get("arguments") != control_api.payload(arguments):
            raise ValueError("A source-labelled physical query changed")
        geometry = cuts_api.common_contact_cut_bounds_from_dict(row["geometry"], reference=reference)
        expected = cuts_api.evaluate_common_contact_cut(reference, cut, **arguments)
        if geometry != expected:
            raise ValueError("A density query escaped its actual geometry or physical depth")
        density = api.allocated_source_reference_density_from_dict(row["density"],
            allocation=allocation, reference=reference, cut_bounds=geometry)
        cell = api.native_common_source_allocation_cell_for_cut(allocation, cut)
        if (row.get("source_cell_id") != cell.cell_id or density.receiving_cell_id != cell.cell_id
                or density.precision != 96 or density.depth_profile_coefficients != cell.depth_profile_coefficients
                or density.reference_area_density_bounds_m2_per_parameter_per_physical_volume_depth_m.lo <= 0
                or density.physical_area_density_bounds_m2_per_parameter_per_physical_volume_depth_m
                    != geometry.area_density_bounds_m2_per_parameter_per_material_m
                or density.extra_support_fraction_multiplier_applied or density.finite_point_area_assigned
                or density.globally_integrated_force_solution or density.physical_material_admission_authorized):
            raise ValueError("An allocated local density acquired a duplicate weight or physical handover claim")
    hot = next(cut for cut in cuts.cuts if cut.receiving_trace_id is None)
    try:
        api.native_common_source_allocation_cell_for_cut(allocation, hot)
    except ValueError as error:
        rejection = str(error)
    else:
        raise ValueError("The HOT boundary received an original cold source budget")
    if packet.get("hot_original_source_budget_rejection") != rejection:
        raise ValueError("The authentic HOT exclusion differs")
    summary = control_api.allocation_summary(allocation, len(rows))
    if any(native.get(key) != value or packet.get(key) != value for key, value in summary.items()):
        raise ValueError("Whole-source counts, conservation or physical scope differs")
    if (summary["active_cold_support_cell_count"], summary["inactive_source_cell_count"],
            summary["active_source_atom_accounting_count"], summary["inactive_source_atom_accounting_count"],
            summary["unresolved_source_atom_accounting_count"], summary["exact_shared_projective_seam_count"]) != (5, 3, 8, 8, 0, 2):
        raise ValueError("The actual support/atom partition or shared projective endpoints differ")
    return summary


def run(out, test_paths, control):
    tests = tuple(test_paths)
    if (tests != ("tests/test_geometric_native_common_source_allocation.py",)
            or control != "analysis.geometric_native_common_source_allocation_validation.native_control:run"):
        raise ValueError("Exactly the new source-allocation test module and callback are required")
    inherited = verify_prior(); before = source_map(); added = set(before)-set(inherited)
    if len(before) != 863 or added != ADDED_SOURCE_PATHS:
        raise ValueError("Exactly four additive whole-source allocation Python sources are required")
    removed = []
    for name in sorted(added):
        cache = Path(importlib.util.cache_from_source(str(ROOT/name)))
        if cache.is_file(): cache.unlink(); removed.append(str(cache.relative_to(ROOT)))
    purged = _purge_added_source_modules(added-{str(Path(__file__).resolve().relative_to(ROOT))})
    importlib.invalidate_caches()
    if source_map() != before: raise ValueError("A source changed before fresh imports")
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False); started = time.perf_counter()
    report = dict(format="native-common-whole-source-allocation-same-process-validation-1", status="FAIL",
        physical_candidate_status="NOT_CHECKED", started_utc=datetime.now(timezone.utc).isoformat(),
        source_count=863, source_map_before_sha256=value_digest(before), prior_source_map_sha256=OLD_MAP,
        verified_prior_python_sources=859, verified_prior_artifacts=1633,
        added_python_sources={name: before[name] for name in sorted(added)}, execution_mode="same_python_process",
        pytest_import_mode="importlib", historical_tests_reexecuted=False, test_paths=list(tests), control=control,
        removed_added_source_bytecode_caches=removed, purged_previously_imported_added_source_modules=purged,
        verified_current_baseline_evidence=CURRENT_EVIDENCE)
    try:
        import pytest
        start = time.perf_counter(); old_dir = Path.cwd()
        try:
            os.chdir(ROOT)
            with (out/"pytest.log").open("x", buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream):
                code = int(pytest.main(["-q", "-p", "no:cacheprovider", "--import-mode=importlib", *tests]))
        finally: os.chdir(old_dir)
        summary, counts = parse_counts(out/"pytest.log")
        report.update(pytest_exit_code=code, pytest_counts=counts, pytest_summary=summary, pytest_seconds=time.perf_counter()-start)
        if code or counts != {"passed": 17}:
            raise RuntimeError("Fresh whole-source allocation tests failed")
        if source_map() != before: raise ValueError("A source changed during fresh tests")
        module, function = control.split(":", 1); callback = getattr(importlib.import_module(module), function)
        start = time.perf_counter()
        with (out/"control_console.log").open("x", buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream):
            callback(out/"native_control")
        report["native_control_seconds"] = time.perf_counter()-start
        report.update(_check_native_control(out))
        expected = {"pytest.log", "control_console.log", "native_control/report.json", "native_control/source_packet.json"}
        if {str(path.relative_to(out)) for path in out.rglob("*") if path.is_file()} != expected:
            raise ValueError("Exactly four closed outputs are required")
        report["status"] = "PASS"
    except BaseException as error:
        report.update(failure_type=type(error).__name__, failure_message=str(error), failure_traceback=traceback.format_exc())
    finally:
        after = source_map(); report["source_map_after_sha256"] = value_digest(after)
        report["changed_python_paths"] = sorted(name for name in set(before)|set(after) if before.get(name)!=after.get(name))
        try:
            verify_prior(); report["prior_artifacts_and_python_sources_preserved_after"] = True
        except BaseException as error:
            report.update(prior_artifacts_and_python_sources_preserved_after=False, prior_after_error=str(error))
        if report["changed_python_paths"] or not report["prior_artifacts_and_python_sources_preserved_after"]: report["status"]="FAIL"
        report["seconds"] = time.perf_counter()-started; report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["outputs"] = {str(path.relative_to(out)):dict(sha256=digest(path), bytes=path.stat().st_size)
                             for path in sorted(out.rglob("*")) if path.is_file()}
        report["report_sha256"] = value_digest(report)
        (out/"report.json").write_bytes(encoded(report)+b"\n")
        print(encoded({key:report.get(key) for key in ("status", "physical_candidate_status", "pytest_counts", "seconds", "source_count", "changed_python_paths")}).decode(), flush=True)
    if report["status"] != "PASS": raise RuntimeError("Validation refused; see " + str(out/"report.json"))
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--out", required=True); parser.add_argument("--control", required=True); parser.add_argument("tests", nargs="+")
    args=parser.parse_args(); run(args.out, args.tests, args.control)
