"""Fresh scoped handover tests and existing local-contact/beam regressions.

One pytest process shares the genuine physical source fixture across the new
test modules. Earlier 253- and 770-test receipts are authenticated ancestry;
this gate never claims to repeat either release or the full test profile.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.geometric_native_physical_handover_validation.run_handover_control import (
    digest, guard_sources, write,
)

TEST_PATHS = (
    "tests/test_geometric_native_beam_material_chart.py",
    "tests/test_geometric_native_contact_reference_transport.py",
    "tests/test_geometric_native_material_port_handover.py",
    "tests/test_geometric_contact_mechanics.py",
    "tests/test_geometric_bending.py",
)


def parse_summary(path):
    lines = path.read_text().splitlines()
    summary = next((line for line in reversed(lines)
        if re.search(r"\d+ (passed|failed|errors?|skipped)", line)), "")
    counts = {label:int(count) for count, label
        in re.findall(r"(\d+) (passed|failed|errors?|skipped)", summary)}
    return summary, counts


def run_gate(out):
    before, starting = guard_sources()
    if any(not (ROOT / path).is_file() for path in TEST_PATHS):
        raise ValueError("The handover gate requires all three new and two existing regression files")
    out.mkdir(parents=True, exist_ok=False)
    write(out / "sources_before.json", before)
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *TEST_PATHS]
    started, started_utc = time.perf_counter(), datetime.now(timezone.utc).isoformat()
    with (out / "pytest.log").open("x") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    summary, counts = parse_summary(out / "pytest.log")
    after, starting_after = guard_sources()
    write(out / "sources_after.json", after)
    changed = sorted(name for name in set(before) | set(after)
        if before.get(name) != after.get(name))
    if starting != starting_after:
        changed.append("frozen_664_source_map")
    passed = result.returncode == 0 and not changed and counts.get("passed", 0) > 0
    receipt = dict(format="native-material-port-handover-scoped-regression-gate-1",
        status="PASS" if passed else "FAIL", scope="three_new_files_and_two_local_contact_beam_regressions",
        original_253_tests_reexecuted=False, original_770_tests_reexecuted=False,
        is_full_profile_execution=False, command=command, test_paths=list(TEST_PATHS),
        execution_mode="single_pytest_process_shared_actual_source_fixture",
        started_utc=started_utc, finished_utc=datetime.now(timezone.utc).isoformat(),
        seconds=time.perf_counter() - started, exit_code=result.returncode,
        pytest_summary=summary, counts=counts, original_source_count=664,
        source_count=len(before), changed_paths=changed,
        sources_before_sha256=digest(out / "sources_before.json"),
        sources_after_sha256=digest(out / "sources_after.json"),
        log_sha256=digest(out / "pytest.log"), runner_sha256=digest(Path(__file__)))
    saved = write(out / "receipt.json", receipt, seal="receipt_sha256")
    print(json.dumps({name:saved[name] for name in
        ("status", "counts", "seconds", "source_count", "is_full_profile_execution")}), flush=True)
    if not passed:
        raise SystemExit(result.returncode or 1)
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run_gate(parser.parse_args().out.resolve())
