"""Fresh mechanical-carrier component gate; all old863 sources immutable."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PARENT = Path(__file__).resolve().parent
OLD_PARENT = ROOT / "analysis/geometric_native_common_source_allocation_validation"
OLD_MAP = "83ebcc92428688af96c4f104b5ce9bbd96f29625e8a46d2b08e77b2fbad99406"
PHYSICAL_SCOPE = "COMMON_WHOLE_SOURCE_MECHANICAL_CARRIER_COMPONENT_ONLY"
FRESH_TEST_COUNT = 18
ADDED_SOURCE_PATHS = frozenset((
    "tectonics/geometric_native_common_source_carrier.py",
    "tests/test_geometric_native_common_source_carrier.py",
    "analysis/geometric_native_common_source_carrier_validation/native_control.py",
    "analysis/geometric_native_common_source_carrier_validation/validate.py",
))
TEST_PATH = "tests/test_geometric_native_common_source_carrier.py"
CONTROL = "analysis.geometric_native_common_source_carrier_validation.native_control:run"
CURRENT_EVIDENCE = {
    "baseline_863_bootstrap_source.txt": "eafbd34a81e9a6ee1f6d06c368797e4cec6291f445c8efc5e024ac16ffda4572",
    "baseline_863_bootstrap.json": "bfd7d18e5f407083ade3aea48e7ed183952c00b01d7ad66ad2cbbe2a09811732",
    "baseline_863_reusable_source.txt": "2f9168e7691a6ad4cd56633323f60895abd71f502beb553b8a239f0e66c5a46c",
}


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def value_digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def sealed(path, key):
    # Retain the validated checksum key; callers never depend on a popped key.
    value = read(path)
    if value.get(key) != value_digest({k: v for k, v in value.items() if k != key}):
        raise ValueError("Evidence selfseal changed: " + str(path))
    return value


def source_map():
    return {str(path.relative_to(ROOT)): digest(path)
            for directory in ("tectonics", "tests", "scripts", "analysis")
            for path in sorted((ROOT/directory).rglob("*.py"))}


def verify_prior():
    for name, expected in CURRENT_EVIDENCE.items():
        if digest(PARENT/name) != expected:
            raise ValueError("The immutable863 recovery evidence changed: " + name)
    name = "carrier_gate_stdlib863_readback"
    module = types.ModuleType(name); module.__file__ = str(PARENT/"baseline_863_reusable_source.txt")
    sys.modules[name] = module
    exec(compile(Path(module.__file__).read_bytes(), module.__file__, "exec"), module.__dict__)
    with redirect_stdout(io.StringIO()):
        baseline = module.run(ADDED_SOURCE_PATHS,
            output_path="/tmp/native_common_source_carrier_gate_prior_readback_20261006.json",
            bootstrap_source_path=PARENT/"baseline_863_bootstrap_source.txt",
            bootstrap_report_path=PARENT/"baseline_863_bootstrap.json")
    if (baseline.get("status") != "PASS_IMMUTABLE863_WITH_DECLARED867_SOURCE_ADDITIONS_ONLY"
            or baseline.get("baseline_source_count") != 863
            or baseline.get("all863_python_sources_unchanged") is not True
            or baseline.get("preserved_old1633_artifact_chain") is not True):
        raise ValueError("The complete immutable863 ancestry readback failed")
    inherited = read(OLD_PARENT/"frozen_source_map_863.json")
    if len(inherited) != 863 or value_digest(inherited) != OLD_MAP:
        raise ValueError("The preserved863 source map differs")
    return inherited


def _check_native_control(out):
    from fractions import Fraction as F
    from tectonics.geometric_native_normal_support import load_native_normal_support_source
    from tectonics.geometric_native_common_reference import native_common_reference_from_dict, native_common_reference_to_dict
    from tectonics import geometric_native_common_contact_cuts as cuts_api
    from tectonics import geometric_native_common_source_allocation as allocation_api
    from tectonics import geometric_native_common_source_carrier as api
    from analysis.geometric_native_common_source_carrier_validation import native_control as control_api
    from analysis.geometric_native_common_source_allocation_validation.native_control import declared_profiles
    native = sealed(out/"native_control/report.json", "report_sha256")
    packet = sealed(out/"native_control/source_packet.json", "packet_sha256")
    if (native.get("status") != "PASS" or native.get("source_packet_sha256") != packet["packet_sha256"]
            or native.get("format") != "native-common-whole-source-carrier-control-1"
            or packet.get("format") != "native-common-whole-source-carrier-packet-1"
            or packet.get("prior_863_wrapper_raw_sha256") != control_api.PRIOR_WRAPPER_RAW):
        raise ValueError("The whole-source mechanical output identities differ")
    source = load_native_normal_support_source()
    if packet.get("source_snapshot_sha256") != source.source_snapshot_sha256:
        raise ValueError("The complete original source snapshot changed")
    reference = native_common_reference_from_dict(packet["common_reference"], source=source)
    if native_common_reference_to_dict(reference) != packet["common_reference"]:
        raise ValueError("The genuine common reference changed")
    cuts = cuts_api.common_contact_cuts_from_dict(packet["cut_set"], reference=reference)
    allocation = allocation_api.native_common_source_allocation_from_dict(packet["allocation"], reference=reference, cuts=cuts)
    if (allocation.precision != 96 or allocation.provenance != control_api.ALLOCATION_PROVENANCE
            or {family.trace_id: family.depth_profile_coefficients for family in allocation.families} != declared_profiles(source)
            or (allocation.source_native_elapsed_time_s, allocation.source_memory_elapsed_time_s,
         allocation.source_contact_elapsed_time_s) != (F(1000000), F(1000000), F(2000000))
            or allocation.source_current_node_epoch != "unaccepted_candidate"
            or len(allocation.original_atom_ledger) != 16):
        raise ValueError("The original sixteen histories or distinct clocks changed")
    system = api.native_common_source_carrier_system_from_dict(packet["carrier_system"],
        allocation=allocation, reference=reference, cuts=cuts)
    if (system.provenance != control_api.PROVENANCE or system.precision != 96
            or len(system.coordinate_order) != 14 or len(system.carriers) != 16
            or native.get("carrier_system_sha256") != system.binding_sha256):
        raise ValueError("Every original mechanical carrier and physical coordinate is required")
    rows = packet.get("carrier_queries", ())
    cases = control_api.query_cases(allocation, system)
    if len(rows) != 68 or len(cases) != 68 or native.get("carrier_queries_sha256") != value_digest(rows):
        raise ValueError("The full two-basis/two-branch carrier query list changed")
    for row, arguments in zip(rows, cases):
        if type(row) is not dict or set(row) != {"arguments", "response"} or row.get("arguments") != control_api.payload(arguments):
            raise ValueError("A source point, branch, mechanical basis or radial query changed")
        response = api.native_common_source_carrier_response_from_dict(row["response"], system=system)
        expected = api.evaluate_native_common_source_carrier(system, **arguments)
        if api.native_common_source_carrier_response_to_dict(response) != api.native_common_source_carrier_response_to_dict(expected):
            raise ValueError("A full carrier operator or variational potential changed")
    result = control_api.summary(allocation, system, len(rows))
    if any(native.get(key) != value or packet.get(key) != value for key, value in result.items()):
        raise ValueError("The whole-source work lineages, counts or physical scope differ")
    if (result["active_source_atom_accounting_count"], result["inactive_source_atom_accounting_count"]) != (8, 8):
        raise ValueError("The original source accounting classes changed")
    return result


def _purge_added_modules(paths):
    stems = {str(Path(name).with_suffix("")).replace("/", ".") for name in paths}
    removed = []
    for name, module in tuple(sys.modules.items()):
        location = getattr(module, "__file__", None)
        exact_file = location is not None and any(Path(location).resolve() == ROOT/path for path in paths)
        if name != "__main__" and (name in stems or exact_file):
            sys.modules.pop(name, None); removed.append(name)
    return sorted(removed)


def run(out, test_paths, control):
    tests = tuple(test_paths)
    if tests != (TEST_PATH,) or control != CONTROL:
        raise ValueError("Exactly the new mechanical-carrier test module and control are authorized")
    inherited = verify_prior(); before = source_map()
    if (len(before) != 867 or set(before)-set(inherited) != ADDED_SOURCE_PATHS
            or any(before.get(name) != pin for name, pin in inherited.items())):
        raise ValueError("Exactly four additive carrier Python sources are required")
    removed = []
    for name in sorted(ADDED_SOURCE_PATHS):
        cache = Path(importlib.util.cache_from_source(str(ROOT/name)))
        if cache.is_file():
            cache.unlink(); removed.append(str(cache.relative_to(ROOT)))
    purged = _purge_added_modules(ADDED_SOURCE_PATHS-{str(Path(__file__).resolve().relative_to(ROOT))})
    importlib.invalidate_caches()
    if source_map() != before:
        raise ValueError("A source changed before fresh imports")
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False); started = time.perf_counter()
    report = dict(format="native-common-whole-source-carrier-same-process-validation-1", status="FAIL",
        physical_candidate_status="NOT_CHECKED", started_utc=datetime.now(timezone.utc).isoformat(),
        source_count=867, source_map_before_sha256=value_digest(before), prior_source_map_sha256=OLD_MAP,
        verified_prior_python_sources=863, verified_prior_artifacts=1633,
        added_python_sources={name: before[name] for name in sorted(ADDED_SOURCE_PATHS)},
        execution_mode="same_python_process", pytest_import_mode="importlib", historical_tests_reexecuted=False,
        test_paths=list(tests), control=control, removed_added_source_bytecode_caches=removed,
        purged_previously_imported_added_source_modules=purged, verified_current_baseline_evidence=CURRENT_EVIDENCE)
    try:
        import pytest
        start = time.perf_counter(); old_dir = Path.cwd()
        try:
            os.chdir(ROOT)
            with (out/"pytest.log").open("x", buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream):
                code = int(pytest.main(["-q", "-p", "no:cacheprovider", "--import-mode=importlib", *tests]))
        finally:
            os.chdir(old_dir)
        lines = (out/"pytest.log").read_text().splitlines()
        summary_line = lines[-1] if lines else ""
        counts = {name: int(number) for number, name in re.findall(r"\b(\d+)\s+(passed|failed|skipped|xfailed|xpassed|errors?|deselected)\b", summary_line)}
        report.update(pytest_exit_code=code, pytest_counts=counts, pytest_summary=summary_line,
                      pytest_seconds=time.perf_counter()-start)
        if code or counts != {"passed": FRESH_TEST_COUNT}:
            raise RuntimeError("Fresh whole-source mechanical-carrier tests failed")
        if source_map() != before:
            raise ValueError("A source changed during fresh tests")
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
        report["changed_python_paths"] = sorted(name for name in set(before)|set(after) if before.get(name) != after.get(name))
        try:
            verify_prior(); report["prior_artifacts_and_python_sources_preserved_after"] = True
        except BaseException as error:
            report.update(prior_artifacts_and_python_sources_preserved_after=False, prior_after_error=str(error))
        if report["changed_python_paths"] or not report["prior_artifacts_and_python_sources_preserved_after"]:
            report["status"] = "FAIL"
        report["seconds"] = time.perf_counter()-started; report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["outputs"] = {str(path.relative_to(out)): dict(sha256=digest(path), bytes=path.stat().st_size)
                             for path in sorted(out.rglob("*")) if path.is_file()}
        report["report_sha256"] = value_digest(report)
        (out/"report.json").write_bytes(encoded(report)+b"\n")
        print(encoded({key: report.get(key) for key in ("status", "physical_candidate_status", "pytest_counts", "seconds", "source_count", "changed_python_paths")}).decode(), flush=True)
    if report["status"] != "PASS":
        raise RuntimeError("Validation refused; see " + str(out/"report.json"))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--out", required=True); parser.add_argument("--control", required=True); parser.add_argument("tests", nargs="+")
    args = parser.parse_args(); run(args.out, args.tests, args.control)
