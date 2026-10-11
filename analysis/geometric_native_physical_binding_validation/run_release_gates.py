"""Fresh scoped regression gate for original physical-history bindings.

The prior authenticated 770-test release is referenced separately.  This gate
runs all three new test files and the two directly affected dependencies; it
does not execute the full test profile or claim to repeat the earlier gate.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.geometric_native_physical_binding_validation.run_binding_control import (
    digest, guard_sources, write,
)

TEST_PATHS = (
    "tests/test_geometric_native_physical_pair_time.py",
    "tests/test_geometric_native_physical_history.py",
    "tests/test_geometric_native_first_entry_physical_binding.py",
    "tests/test_geometric_native_contact_time_support.py",
    "tests/test_geometric_native_first_entry_material.py",
)


def parse_summary(path):
    lines = path.read_text().splitlines()
    summary = next((line for line in reversed(lines)
        if re.search(r"\d+ (passed|failed|errors?|skipped)", line)), "")
    counts = {label: int(count) for count, label
        in re.findall(r"(\d+) (passed|failed|errors?|skipped)", summary)}
    return summary, counts


def run_gate(out, *, workers=3):
    if type(workers) is not int or workers < 1:
        raise ValueError("Worker count must be a positive integer")
    before, starting = guard_sources()
    if any(not (ROOT / path).is_file() for path in TEST_PATHS):
        raise ValueError("Scoped gate requires all three new and two dependency test files")
    out.mkdir(parents=True, exist_ok=False)
    write(out / "sources_before.json", before)
    logs = out / "per_file"
    logs.mkdir()
    started, started_utc = time.perf_counter(), datetime.now(timezone.utc).isoformat()

    def run_file(item):
        index, path = item
        command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", path]
        name = f"{index:02d}-{Path(path).stem}"
        log_path = logs / (name + ".log")
        file_started = time.perf_counter()
        with log_path.open("x") as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        summary, counts = parse_summary(log_path)
        record = dict(test_path=path, command=command, exit_code=result.returncode,
            seconds=time.perf_counter() - file_started, pytest_summary=summary, counts=counts,
            log_path=str(log_path.relative_to(out)), log_sha256=digest(log_path))
        write(logs / (name + ".json"), record)
        return record

    with ThreadPoolExecutor(max_workers=workers) as executor:
        per_file = list(executor.map(run_file, enumerate(TEST_PATHS)))
    if tuple(row["test_path"] for row in per_file) != TEST_PATHS:
        raise AssertionError("Scoped test-file coverage changed")
    exit_code = next((row["exit_code"] for row in per_file if row["exit_code"]), 0)
    counts = {}
    for row in per_file:
        for key, count in row["counts"].items():
            counts[key] = counts.get(key, 0) + count
    with (out / "pytest.log").open("x") as stream:
        for row in per_file:
            stream.write(row["test_path"] + "\n")
            stream.write((out / row["log_path"]).read_text())
    after, starting_after = guard_sources()
    write(out / "sources_after.json", after)
    changed = sorted(name for name in set(before) | set(after)
        if before.get(name) != after.get(name))
    if starting != starting_after:
        changed.append("frozen_654_starting_source_map")
    passed = exit_code == 0 and not changed and counts.get("passed", 0) > 0
    receipt = dict(format="native-physical-binding-scoped-regression-gate-1",
        status="PASS" if passed else "FAIL", scope="three_new_files_and_two_direct_dependency_files",
        original_770_tests_reexecuted=False, is_full_profile_execution=False,
        started_utc=started_utc, finished_utc=datetime.now(timezone.utc).isoformat(),
        seconds=time.perf_counter() - started, worker_count=workers,
        execution_mode="parallel_complete_test_files", test_paths=list(TEST_PATHS),
        per_file_results=per_file, exit_code=exit_code, counts=counts,
        pytest_summary=", ".join(f"{count} {key}" for key, count in sorted(counts.items()))
            + f" (sum of {len(per_file)} complete-file pytest processes)",
        original_source_count=654, source_count=len(before), changed_paths=changed,
        sources_before_sha256=digest(out / "sources_before.json"),
        sources_after_sha256=digest(out / "sources_after.json"),
        log_sha256=digest(out / "pytest.log"), runner_sha256=digest(Path(__file__)))
    saved = write(out / "receipt.json", receipt, seal="receipt_sha256")
    print(json.dumps({key:saved[key] for key in ("status", "counts", "seconds",
        "source_count", "is_full_profile_execution")}), flush=True)
    if not passed:
        raise SystemExit(exit_code or 1)
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    run_gate(args.out.resolve(), workers=args.workers)
