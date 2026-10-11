"""Fresh material-extension gate; immutable867 ancestry, one Python process."""
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
OLD_PARENT = ROOT / "analysis/geometric_native_common_source_carrier_validation"
OLD_MAP = "bfabfa58dbe49ce1102ef9ab72680a301f8cea6fb30093442150d8d4bda45ef8"
PHYSICAL_SCOPE = "COMMON_ORIGINAL_MATERIAL_EXTENDED_REDUCED_SOURCE_COMPONENT_ONLY"
FRESH_TEST_COUNT = 14
ADDED_SOURCE_PATHS = frozenset((
    "tectonics/geometric_native_common_material_transfer.py",
    "tests/test_geometric_native_common_material_transfer.py",
    "analysis/geometric_native_common_material_transfer_validation/native_control.py",
    "analysis/geometric_native_common_material_transfer_validation/validate.py",
))
TEST_PATH = "tests/test_geometric_native_common_material_transfer.py"
CONTROL = "analysis.geometric_native_common_material_transfer_validation.native_control:run"
CURRENT_EVIDENCE = {
    "baseline_867_bootstrap_source.txt": "8e86273fadec1d0246132f4ff2be0c33f497d1d617dc6ec9221c09a8b5e415d6",
    "baseline_867_bootstrap.json": "17d4094a5b3a16edaaa36519c29c0a96d1eeba741e8a826f7686fddd16dc1f83",
    "baseline_867_reusable_source.txt": "50eca1fa73aac54061f82f8104ecfc88c1d94ba696cd23ab2a573a2b66ba58cc",
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
            raise ValueError("The immutable867 recovery evidence changed: " + name)
    name = "material_gate_stdlib867_readback"
    module = types.ModuleType(name); module.__file__ = str(PARENT / "baseline_867_reusable_source.txt")
    sys.modules[name] = module
    exec(compile(Path(module.__file__).read_bytes(), module.__file__, "exec"), module.__dict__)
    with redirect_stdout(io.StringIO()):
        baseline = module.run(ADDED_SOURCE_PATHS,
            output_path="/tmp/native_common_material_transfer_gate_prior_readback_20261006.json",
            bootstrap_source_path=PARENT / "baseline_867_bootstrap_source.txt",
            bootstrap_report_path=PARENT / "baseline_867_bootstrap.json")
    if (baseline.get("status") != "PASS_IMMUTABLE867_WITH_DECLARED871_SOURCE_ADDITIONS_ONLY"
            or baseline.get("baseline_source_count") != 867
            or baseline.get("all867_python_sources_unchanged") is not True
            or baseline.get("preserved_old1633_artifact_chain") is not True):
        raise ValueError("The complete immutable867 ancestry readback failed")
    inherited = read(OLD_PARENT / "frozen_source_map_867.json")
    if len(inherited) != 867 or value_digest(inherited) != OLD_MAP:
        raise ValueError("The preserved867 source map differs")
    return inherited


def _check_native_control(out):
    from analysis.geometric_native_common_material_transfer_validation import native_control as control
    native = sealed(out / "native_control/report.json", "report_sha256")
    packet = sealed(out / "native_control/source_packet.json", "packet_sha256")
    if (native.get("status") != "PASS"
            or native.get("format") != "native-common-original-material-reduced-source-control-1"
            or native.get("source_packet_sha256") != packet["packet_sha256"]
            or native.get("global_queries_sha256") != control.digest(packet["global_queries"])
            or native.get("material_point_queries_sha256") != control.digest(packet["material_point_queries"])
            or native.get("material_transfer_sha256") != packet["material_transfer"]["binding_sha256"]):
        raise ValueError("The material extension control identities differ")
    result = control.replay_packet(packet)
    if any(native.get(k) != v for k, v in result.items()):
        raise ValueError("Material extension physical scope changed")
    return result


def _purge_added_modules(paths):
    stems = {str(Path(name).with_suffix("")).replace("/", ".") for name in paths}
    removed = []
    for name, module in tuple(sys.modules.items()):
        location = getattr(module, "__file__", None)
        exact_file = location is not None and any(Path(location).resolve() == ROOT / path for path in paths)
        if name != "__main__" and (name in stems or exact_file):
            sys.modules.pop(name, None); removed.append(name)
    return sorted(removed)


def run(out, test_paths, control):
    tests = tuple(test_paths)
    if tests != (TEST_PATH,) or control != CONTROL:
        raise ValueError("Exactly the new material-transfer test module and control are authorized")
    started = time.perf_counter(); started_utc = datetime.now(timezone.utc).isoformat()
    inherited = verify_prior(); before = source_map()
    if (len(before) != 871 or set(before) - set(inherited) != ADDED_SOURCE_PATHS
            or any(before.get(name) != pin for name, pin in inherited.items())):
        raise ValueError("Exactly four additive material-transfer Python sources are required")
    frozen = PARENT / "frozen_source_map_871.json"
    if not frozen.is_file() or read(frozen) != before:
        raise ValueError("The complete871 source freeze is required and must match current sources")
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
    report = dict(format="native-common-original-material-reduced-source-same-process-validation-1",
        status="FAIL", physical_candidate_status="NOT_CHECKED", started_utc=started_utc,
        source_count=871, source_map_before_sha256=value_digest(before), prior_source_map_sha256=OLD_MAP,
        verified_prior_python_sources=867, verified_prior_artifacts=1633,
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
            raise RuntimeError("Fresh original-material reduced-source tests failed")
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
