"""Run the agreed active-binding checks and only their direct regressions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.geometric_native_active_binding_validation.validation_support import (
    digest, encoded, guard_sources, write,
)


def parse_summary(path):
    lines = Path(path).read_text().splitlines()
    for line in reversed(lines):
        counts = {kind: int(number) for number, kind in re.findall(
            r"\b(\d+) (passed|failed|skipped|xfailed|xpassed|errors?|deselected)\b", line)}
        if counts:
            return line, counts
    return "", {}


def run_gate(out, test_paths):
    paths = tuple(test_paths)
    if (not paths or len(set(paths)) != len(paths)
            or any(not name.startswith("tests/test_") or not name.endswith(".py")
                or not (ROOT / name).is_file() for name in paths)):
        raise ValueError("The scoped gate requires distinct existing test files")
    before, original = guard_sources(require_frozen=True)
    out.mkdir(parents=True, exist_ok=False)
    write(out / "sources_before.json", before)
    command = [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
        "--import-mode=importlib", *paths]
    started, started_utc = time.perf_counter(), datetime.now(timezone.utc).isoformat()
    with (out / "pytest.log").open("x") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    summary, counts = parse_summary(out / "pytest.log")
    after, original_after = guard_sources(require_frozen=True)
    write(out / "sources_after.json", after)
    changed = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
    if original != original_after:
        changed.append("frozen_694_source_map")
    passed = result.returncode == 0 and not changed and counts.get("passed", 0) > 0
    receipt = dict(format="native-active-binding-scoped-regression-gate-1",
        status="PASS" if passed else "FAIL", scope="active_binding_and_agreed_direct_regressions",
        is_full_profile_execution=False, prior_341_full_scope_reexecuted=False,
        command=command, test_paths=list(paths),
        execution_mode="single_pytest_process_shared_actual_source_fixture",
        canonical_pytest_import_mode="importlib", started_utc=started_utc,
        finished_utc=datetime.now(timezone.utc).isoformat(), seconds=time.perf_counter() - started,
        exit_code=result.returncode, pytest_summary=summary, counts=counts,
        original_source_count=694, source_count=len(before), changed_paths=changed,
        sources_before_sha256=digest(out / "sources_before.json"),
        sources_after_sha256=digest(out / "sources_after.json"),
        log_sha256=digest(out / "pytest.log"), runner_sha256=digest(Path(__file__)))
    saved = write(out / "receipt.json", receipt, seal="receipt_sha256")
    print(encoded({name: saved[name] for name in ("status", "counts", "seconds",
        "source_count", "is_full_profile_execution")}).decode(), flush=True)
    if not passed:
        raise SystemExit(result.returncode or 1)
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tests", nargs="+", required=True)
    args = parser.parse_args()
    run_gate(args.out.resolve(), args.tests)
