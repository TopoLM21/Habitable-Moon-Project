"""Repeat the focused native geometry and force-integration release gate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
TESTS = (
    "test_geometric_native_pose.py", "test_geometric_native_polygons.py",
    "test_geometric_native_surface.py", "test_geometric_intrinsic_geometry.py",
    "test_geometric_intrinsic_surface.py", "test_geometric_precision.py",
    "test_geometric_surface.py", "test_spherical_polygons.py", "test_cloud_handoff.py",
    "test_geometric_native_buried_support.py", "test_geometric_native_mantle_loading.py",
    "test_geometric_native_spatial_underthrust.py", "test_geometric_native_underthrust_motion.py",
    "test_geometric_native_underthrust_motion_io.py",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot():
    paths = set()
    for folder in ("tectonics", "tests", "scripts", "analysis"):
        paths.update((ROOT/folder).rglob("*.py"))
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(paths)}


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-m", "pytest", "-q", *("tests/"+name for name in TESTS)]
    started, before = time.monotonic(), snapshot()
    write(args.out/"sources_before.json", before)
    receipt = dict(started_utc=datetime.now(timezone.utc).isoformat(), command=command,
                   is_full_profile_execution=False, is_collection_only=False)
    with (args.out/"pytest.log").open("x") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    after = snapshot()
    write(args.out/"sources_after.json", after)
    lines = (args.out/"pytest.log").read_text().splitlines()
    changed = sorted(path for path in set(before)|set(after) if before.get(path)!=after.get(path))
    summary = next((line for line in reversed(lines) if re.search(r"\d+ (passed|failed|error)", line)), "")
    receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(),
        seconds=time.monotonic()-started, exit_code=result.returncode,
        status="passed" if result.returncode==0 and not changed else "failed",
        source_count=len(before), changed_paths=changed, pytest_summary=summary,
        counts={label:int(count) for count,label in re.findall(r"(\d+) (passed|failed|errors?|skipped)", summary)},
        runner_sha256=sha(__file__), log_sha256=sha(args.out/"pytest.log"),
        before_sha256=sha(args.out/"sources_before.json"), after_sha256=sha(args.out/"sources_after.json"))
    write(args.out/"receipt.json", receipt)
    print(json.dumps(receipt), flush=True)
    if receipt["status"]!="passed":
        raise SystemExit(result.returncode or 1)


if __name__=="__main__":
    main()
