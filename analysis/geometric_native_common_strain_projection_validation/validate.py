"""Fresh nonlinear strain-projection gate; immutable871 ancestry, one process."""
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
OLD_PARENT = ROOT / "analysis/geometric_native_common_material_transfer_validation"
OLD_MAP = "5ba59a6e8231e7cf2408b2614b19601cc631bd204eb4f436b4241cf4fa535915"
PHYSICAL_SCOPE = "COMMON_BANKWISE_COLD_STRAIN_PROJECTION_REDUCTION_COMPONENT_ONLY"
FRESH_TEST_COUNT = 14
ADDED_SOURCE_PATHS = frozenset((
    "tectonics/geometric_native_common_strain_projection.py",
    "tests/test_geometric_native_common_strain_projection.py",
    "analysis/geometric_native_common_strain_projection_validation/native_control.py",
    "analysis/geometric_native_common_strain_projection_validation/validate.py",
))
TEST_PATH = "tests/test_geometric_native_common_strain_projection.py"
CONTROL = "analysis.geometric_native_common_strain_projection_validation.native_control:run"
CURRENT_EVIDENCE = {
    "baseline_871_bootstrap_source.txt": "429397f48f08ce1fb9df2ef57b1c4c983b48bc6c6850c18b0400620a9a5a7752",
    "baseline_871_bootstrap.json": "116db75c6dd43da6404bc458d628dccd684eb65362e03ad583cd704a6eff68cb",
    "baseline_871_reusable_source.txt": "3ad31801a86dd4b3a0a00751a867c3fb0c77d49b50d15dd4435e17b1fb5966fc",
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
    return json.loads(Path(path).read_bytes())


def sealed(path, key):
    value = read(path)
    if value.get(key) != value_digest({k: v for k, v in value.items() if k != key}):
        raise ValueError("Evidence selfseal changed: " + str(path))
    return value


def source_map():
    return {str(path.relative_to(ROOT)): digest(path)
        for directory in ("tectonics", "tests", "scripts", "analysis")
        for path in sorted((ROOT / directory).rglob("*.py"))}


def verify_prior():
    for name, expected in CURRENT_EVIDENCE.items():
        if digest(PARENT / name) != expected:
            raise ValueError("The immutable871 recovery evidence changed: " + name)
    name = "strain_projection_gate_stdlib871_readback"
    module = types.ModuleType(name)
    module.__file__ = str(PARENT / "baseline_871_reusable_source.txt")
    sys.modules[name] = module
    exec(compile(Path(module.__file__).read_bytes(), module.__file__, "exec"), module.__dict__)
    with redirect_stdout(io.StringIO()):
        baseline = module.run(ADDED_SOURCE_PATHS,
            output_path="/tmp/native_common_strain_projection_gate_prior_readback_20261006.json",
            bootstrap_source_path=PARENT / "baseline_871_bootstrap_source.txt",
            bootstrap_report_path=PARENT / "baseline_871_bootstrap.json")
    if (baseline.get("status") != "PASS_IMMUTABLE871_WITH_DECLARED_SOURCE_ADDITIONS_ONLY"
            or baseline.get("baseline_source_count") != 871
            or baseline.get("current_python_source_count") != 875
            or baseline.get("all871_python_sources_unchanged") is not True
            or baseline.get("preserved_old1633_artifact_chain") is not True):
        raise ValueError("The complete immutable871 ancestry readback failed")
    inherited = read(OLD_PARENT / "frozen_source_map_871.json")
    if len(inherited) != 871 or value_digest(inherited) != OLD_MAP:
        raise ValueError("The preserved871 source map differs")
    return inherited


def _check_native_control(out):
    from analysis.geometric_native_common_strain_projection_validation import native_control as control
    native = sealed(out / "native_control/report.json", "report_sha256")
    packet = sealed(out / "native_control/source_packet.json", "packet_sha256")
    if (native.get("status") != "PASS"
            or native.get("format") != "native-common-bankwise-cold-strain-projection-control-1"
            or packet.get("format") != "native-common-bankwise-cold-strain-projection-packet-1"
            or native.get("source_packet_sha256") != packet["packet_sha256"]
            or native.get("global_queries_sha256") != control.digest(packet["global_queries"])
            or native.get("authentic_cold_contexts_sha256") != control.digest(packet["authentic_cold_contexts"])
            or native.get("reference_precision_equivalence_sha256") != packet["reference_precision_equivalence"]["equivalence_sha256"]
            or native.get("strain_projection_sha256") != packet["strain_projection"]["binding_sha256"]
            or native.get("material_transfer_sha256") != packet["material_transfer"]["binding_sha256"]):
        raise ValueError("The whole92 strain-projection control identities differ")
    result = control.replay_packet(packet)
    expected_native = set(result) | {"format", "status", "source_packet_sha256", "global_queries_sha256",
        "authentic_cold_contexts_sha256", "strain_projection_sha256", "material_transfer_sha256",
        "reference_precision_equivalence_sha256", "seconds", "report_sha256"}
    if set(native) != expected_native:
        raise ValueError("The complete strain-projection control schema changed")
    if result.get("physical_candidate_status") != PHYSICAL_SCOPE:
        raise ValueError("The strain-projection component scope changed")
    if encoded({k: native.get(k) for k in result}) != encoded(result):
        raise ValueError("The complete strain-projection metadata or query receipts changed")
    return result


def _purge_added_modules(paths):
    stems = {str(Path(name).with_suffix("")).replace("/", ".") for name in paths}
    removed = []
    for name, module in tuple(sys.modules.items()):
        location = getattr(module, "__file__", None)
        exact_file = location is not None and any(Path(location).resolve() == ROOT / path for path in paths)
        if name != "__main__" and (name in stems or exact_file):
            parent_name, _, child_name = name.rpartition(".")
            parent = sys.modules.get(parent_name)
            if parent is not None and getattr(parent, child_name, None) is module:
                delattr(parent, child_name)
            sys.modules.pop(name, None); removed.append(name)
    return sorted(removed)


def run(out, test_paths, control):
    tests = tuple(test_paths)
    if tests != (TEST_PATH,) or control != CONTROL:
        raise ValueError("Exactly the new strain-projection test module and control are authorized")
    started = time.perf_counter(); started_utc = datetime.now(timezone.utc).isoformat()
    inherited = verify_prior(); before = source_map()
    if (len(before) != 875 or set(before) - set(inherited) != ADDED_SOURCE_PATHS
            or any(before.get(name) != pin for name, pin in inherited.items())):
        raise ValueError("Exactly four additive strain-projection Python sources are required")
    frozen = PARENT / "frozen_source_map_875.json"
    if not frozen.is_file() or read(frozen) != before:
        raise ValueError("The complete875 source freeze is required and must match current sources")
    removed = []
    for name in sorted(ADDED_SOURCE_PATHS):
        cache = Path(importlib.util.cache_from_source(str(ROOT / name)))
        if cache.is_file():
            cache.unlink(); removed.append(str(cache.relative_to(ROOT)))
    purged = _purge_added_modules(ADDED_SOURCE_PATHS - {str(Path(__file__).resolve().relative_to(ROOT))})
    importlib.invalidate_caches()
    if source_map() != before:
        raise ValueError("A source changed before fresh imports")
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False)
    report = dict(format="native-common-bankwise-strain-projection-same-process-validation-1",
        status="FAIL", physical_candidate_status="NOT_CHECKED", started_utc=started_utc,
        source_count=875, source_map_before_sha256=value_digest(before), prior_source_map_sha256=OLD_MAP,
        verified_prior_python_sources=871, verified_prior_artifacts=1633,
        added_python_sources={name: before[name] for name in sorted(ADDED_SOURCE_PATHS)},
        execution_mode="same_python_process", pytest_import_mode="importlib", historical_tests_reexecuted=False,
        test_paths=list(tests), control=control, removed_added_source_bytecode_caches=removed,
        purged_previously_imported_added_source_modules=purged, verified_current_baseline_evidence=CURRENT_EVIDENCE)
    try:
        import pytest
        start = time.perf_counter(); old_dir = Path.cwd()
        try:
            os.chdir(ROOT)
            with (out / "pytest.log").open("x", buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream):
                code = int(pytest.main(["-q", "-p", "no:cacheprovider", "--import-mode=importlib", *tests]))
        finally:
            os.chdir(old_dir)
        lines = (out / "pytest.log").read_text().splitlines()
        summary_line = lines[-1] if lines else ""
        counts = {name: int(number) for number, name in re.findall(
            r"\b(\d+)\s+(passed|failed|skipped|xfailed|xpassed|errors?|deselected)\b", summary_line)}
        report.update(pytest_exit_code=code, pytest_counts=counts, pytest_summary=summary_line,
            pytest_seconds=time.perf_counter() - start)
        if code or counts != {"passed": FRESH_TEST_COUNT}:
            raise RuntimeError("Fresh whole92 nonlinear strain-projection tests failed")
        if source_map() != before:
            raise ValueError("A source changed during fresh tests")
        module, function = control.split(":", 1); callback = getattr(importlib.import_module(module), function)
        start = time.perf_counter()
        with (out / "control_console.log").open("x", buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream):
            callback(out / "native_control")
        report["native_control_seconds"] = time.perf_counter() - start
        report.update(_check_native_control(out))
        expected = {"pytest.log", "control_console.log", "native_control/report.json", "native_control/source_packet.json"}
        if {str(path.relative_to(out)) for path in out.rglob("*") if path.is_file()} != expected:
            raise ValueError("Exactly four closed outputs are required")
        report["status"] = "PASS"
    except BaseException as error:
        report.update(failure_type=type(error).__name__, failure_message=str(error), failure_traceback=traceback.format_exc())
    finally:
        after = source_map(); report["source_map_after_sha256"] = value_digest(after)
        report["changed_python_paths"] = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
        try:
            verify_prior(); report["prior_artifacts_and_python_sources_preserved_after"] = True
        except BaseException as error:
            report.update(prior_artifacts_and_python_sources_preserved_after=False, prior_after_error=str(error))
        if report["changed_python_paths"] or not report["prior_artifacts_and_python_sources_preserved_after"]:
            report["status"] = "FAIL"
        report["seconds"] = time.perf_counter() - started; report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["outputs"] = {str(path.relative_to(out)): dict(sha256=digest(path), bytes=path.stat().st_size)
            for path in sorted(out.rglob("*")) if path.is_file()}
        report["report_sha256"] = value_digest(report)
        (out / "report.json").write_bytes(encoded(report) + b"\n")
        print(encoded({key: report.get(key) for key in ("status", "physical_candidate_status", "pytest_counts",
            "seconds", "source_count", "changed_python_paths")}).decode(), flush=True)
    if report["status"] != "PASS":
        raise RuntimeError("Validation refused; see " + str(out / "report.json"))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--out", required=True)
    parser.add_argument("--control", required=True); parser.add_argument("tests", nargs="+")
    args = parser.parse_args(); run(args.out, args.tests, args.control)
