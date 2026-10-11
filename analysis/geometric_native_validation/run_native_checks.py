"""Preserve exclusive, source-bound receipts for the native geometry gates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = Path(__file__).resolve().parent
PROFILE = ROOT / "analysis/geometric_underthrust_validation/run_portable_checks.py"
spec = importlib.util.spec_from_file_location("native_profile_receipt", PROFILE)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def sources():
    paths = set(VALIDATION.rglob("*.py")) | set(VALIDATION.rglob("*_source.txt"))
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("focused", "control", "collection", "sources"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--control-out", type=Path)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.control_out is not None:
        args.control_out = args.control_out.resolve()
    if args.mode == "focused":
        command = [sys.executable, "-m", "pytest", "-q",
            "tests/test_geometric_native_pose.py", "tests/test_geometric_native_polygons.py",
            "tests/test_geometric_native_surface.py", "tests/test_geometric_intrinsic_geometry.py",
            "tests/test_geometric_intrinsic_surface.py", "tests/test_geometric_precision.py",
            "tests/test_geometric_surface.py", "tests/test_spherical_polygons.py",
            "tests/test_cloud_handoff.py"]
    elif args.mode == "control":
        if args.control_out is None:
            parser.error("control requires --control-out")
        command = [sys.executable, str(VALIDATION / "run_native_control.py"),
                   "--out", str(args.control_out)]
    elif args.mode == "collection":
        command = [sys.executable, "scripts/run_cloud_tests.py", "--collect-only", "-q"]
    else:
        command = [sys.executable, "scripts/cloud_handoff.py", "verify"]
    args.out.mkdir(parents=True, exist_ok=True)
    before, extra = profile.hashes(), sources()
    for name, value in (("code_and_tests_before", before), ("validation_sources_before", extra)):
        write(args.out / f"{args.mode}_{name}.json", value)
    started = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    log_path = args.out / f"{args.mode}.log"
    with log_path.open("x", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    seconds = time.perf_counter() - start
    after, extra_after = profile.hashes(), sources()
    for name, value in (("code_and_tests_after", after), ("validation_sources_after", extra_after)):
        write(args.out / f"{args.mode}_{name}.json", value)
    changed = sorted(name for name in before.keys() | after.keys() if before.get(name) != after.get(name))
    extra_changed = sorted(name for name in extra.keys() | extra_after.keys()
                           if extra.get(name) != extra_after.get(name))
    output = log_path.read_text(encoding="utf-8")
    summaries = [line.strip() for line in output.splitlines()
                 if re.search(r"\b\d+ (?:passed|failed|errors?|collected|tests collected)\b", line)
                 and " in " in line]
    summary = summaries[-1] if summaries else None
    counts = {name: int(count) for count, name in re.findall(
        r"\b(\d+) (passed|failed|errors?|skipped|collected|tests collected)\b", summary or "")}
    report = dict(status="passed" if result.returncode == 0 and not changed and not extra_changed else "failed",
        command=command, cwd=str(ROOT), started_utc=started,
        finished_utc=datetime.now(timezone.utc).isoformat(), seconds=seconds,
        exit_code=result.returncode, pytest_summary=summary, counts=counts,
        core_path_count=len(before), core_changed_paths=changed,
        validation_source_changed_paths=extra_changed,
        log=log_path.relative_to(ROOT).as_posix(), log_sha256=sha(log_path),
        gate_source_sha256=sha(__file__), is_full_profile_execution=False,
        is_collection_only=args.mode == "collection")
    write(args.out / f"{args.mode}_receipt.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
