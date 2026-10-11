"""Run only agreed newborn-contact checks with a complete frozen source guard."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import importlib
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.geometric_native_birth_contact_validation.validation_support import (
    digest, encoded, guard_sources, write,
)


def parse_summary(path):
    for line in reversed(Path(path).read_text().splitlines()):
        counts = {kind: int(number) for number, kind in re.findall(
            r"\b(\d+) (passed|failed|skipped|xfailed|xpassed|errors?|deselected)\b", line)}
        if counts:
            return line, counts
    return "", {}


def run_gate(out, test_paths, *, control=None, control_out=None):
    paths = tuple(test_paths)
    if (not paths or len(set(paths)) != len(paths)
            or any(not name.startswith("tests/test_") or not name.endswith(".py")
                or not (ROOT / name).is_file() for name in paths)):
        raise ValueError("The scoped gate requires distinct existing test modules")
    before, original = guard_sources(require_frozen=True)
    out.mkdir(parents=True, exist_ok=False)
    write(out / "sources_before.json", before)
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
        "--import-mode=importlib", *paths]
    started, started_utc = time.perf_counter(), datetime.now(timezone.utc).isoformat()
    import pytest
    original_directory = Path.cwd()
    try:
        os.chdir(ROOT)
        with (out / "pytest.log").open("x", buffering=1) as stream:
            with redirect_stdout(stream), redirect_stderr(stream):
                exit_code = int(pytest.main(command[3:]))
    finally:
        os.chdir(original_directory)
    summary, counts = parse_summary(out / "pytest.log")
    after, original_after = guard_sources(require_frozen=True)
    write(out / "sources_after.json", after)
    changed = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
    if original != original_after:
        changed.append("inherited_732_source_map")
    passed = (exit_code == 0 and not changed and counts.get("passed", 0) > 0
        and not any(counts.get(key, 0) for key in ("failed", "skipped", "xfailed", "xpassed", "error", "errors")))
    receipt = dict(format="native-birth-contact-scoped-gate-1",
        status="PASS" if passed else "FAIL", is_full_profile_execution=False,
        historical_tests_reexecuted=False, command=command, test_paths=list(paths),
        execution_mode="single_pytest_process", canonical_pytest_import_mode="importlib",
        started_utc=started_utc, finished_utc=datetime.now(timezone.utc).isoformat(),
        seconds=time.perf_counter()-started, exit_code=exit_code, pytest_summary=summary,
        counts=counts, inherited_source_count=len(original), source_count=len(before),
        changed_paths=changed, sources_before_sha256=digest(out / "sources_before.json"),
        sources_after_sha256=digest(out / "sources_after.json"),
        log_sha256=digest(out / "pytest.log"), runner_sha256=digest(Path(__file__)),
        whole_native_interval_admitted_by_test_gate=False)
    saved = write(out / "receipt.json", receipt, "receipt_sha256")
    print(encoded({key: saved[key] for key in ("status", "counts", "seconds", "source_count")}).decode(), flush=True)
    if not passed:
        raise RuntimeError("Frozen scoped gate failed; complete receipt saved at " + str(out / "receipt.json"))
    if control is not None:
        if control_out is None:
            raise ValueError("A same-process control requires its distinct output directory")
        control_started = time.perf_counter()
        with (out / "control_console.log").open("x", buffering=1) as stream:
            with redirect_stdout(stream), redirect_stderr(stream):
                control(control_out)
        final, final_original = guard_sources(require_frozen=True)
        write(out / "suite_sources_after.json", final)
        if final != before or final_original != original:
            raise ValueError("Sources changed during the same-process control")
        combined = dict(format="native-birth-contact-same-process-suite-1", status="PASS",
            pytest_gate_receipt_sha256=digest(out / "receipt.json"),
            control_callable=control.__module__ + ":" + control.__name__,
            control_output=str(control_out.relative_to(ROOT)),
            control_seconds=time.perf_counter()-control_started,
            source_count=len(before), source_before_after_unchanged=True,
            sources_before_sha256=digest(out / "sources_before.json"),
            suite_sources_after_sha256=digest(out / "suite_sources_after.json"),
            control_console_sha256=digest(out / "control_console.log"))
        write(out / "suite_receipt.json", combined, "receipt_sha256")
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--control", help="Optional module:function called after pytest")
    parser.add_argument("--control-out", type=Path)
    parser.add_argument("test_paths", nargs="+")
    options = parser.parse_args()
    callback = None
    if options.control is not None:
        if ":" not in options.control or options.control_out is None:
            parser.error("--control needs module:function and --control-out")
        module, function = options.control.split(":", 1)
        callback = getattr(importlib.import_module(module), function)
    run_gate(options.out.resolve(), options.test_paths, control=callback,
        control_out=None if options.control_out is None else options.control_out.resolve())
