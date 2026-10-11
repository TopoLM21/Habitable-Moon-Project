"""Regression gate for whole first-entry domains and their material closure."""
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
from analysis.geometric_native_full_domain_validation.run_release_gates import ADDED_TESTS
from analysis.geometric_native_material_history_validation.run_release_gates import (
    TESTS, digest, snapshot, write,
)

FIRST_ENTRY_TESTS = ("native_first_entry_domains", "native_first_entry_material",
                     "native_first_entry_oracle", "native_first_entry_point_oracle")


def parse_summary(log_path):
    lines = log_path.read_text().splitlines()
    summary = next((line for line in reversed(lines)
                    if re.search(r"\d+ (passed|failed|error|skipped)", line)), "")
    counts = {label: int(count) for count, label
              in re.findall(r"(\d+) (passed|failed|errors?|skipped)", summary)}
    return summary, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1,
                        help="Independent pytest processes, each running a complete test file")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    test_paths = tuple(
        "tests/test_geometric_" + name + ".py"
        for name in (*TESTS, *ADDED_TESTS, *FIRST_ENTRY_TESTS))
    assert len(test_paths) == len(set(test_paths))
    command = [sys.executable, "-m", "pytest", "-q", *test_paths]
    before, started = snapshot(), time.monotonic()
    write(args.out / "sources_before.json", before)
    started_utc = datetime.now(timezone.utc).isoformat()
    per_file_results = []
    if args.workers == 1:
        with (args.out / "pytest.log").open("x") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        exit_code = result.returncode
        summary, counts = parse_summary(args.out / "pytest.log")
    else:
        log_dir = args.out / "per_file"
        log_dir.mkdir()

        def run_file(item):
            index, path = item
            file_started = time.monotonic()
            file_command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", path]
            name = f"{index:02d}-{Path(path).stem}"
            log_path = log_dir / (name + ".log")
            with log_path.open("x") as log:
                result = subprocess.run(file_command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            file_summary, file_counts = parse_summary(log_path)
            record = dict(test_path=path, command=file_command, exit_code=result.returncode,
                seconds=time.monotonic() - file_started, pytest_summary=file_summary,
                counts=file_counts, log_path=str(log_path.relative_to(args.out)),
                log_sha256=digest(log_path))
            write(log_dir / (name + ".json"), record)
            return record

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            per_file_results = list(executor.map(run_file, enumerate(test_paths)))
        assert tuple(record["test_path"] for record in per_file_results) == test_paths
        exit_code = next((record["exit_code"] for record in per_file_results
                          if record["exit_code"] != 0), 0)
        counts = {}
        for record in per_file_results:
            for label, count in record["counts"].items():
                counts[label] = counts.get(label, 0) + count
        summary = ", ".join(f"{count} {label}" for label, count in sorted(counts.items()))
        summary += f" (sum of {len(per_file_results)} complete-file pytest processes)"
        with (args.out / "pytest.log").open("x") as log:
            for record in per_file_results:
                log.write(record["test_path"] + "\n")
                log.write((args.out / record["log_path"]).read_text())
        command = None
    after = snapshot()
    write(args.out / "sources_after.json", after)
    changed = sorted(path for path in set(before) | set(after)
                     if before.get(path) != after.get(path))
    receipt = dict(
        status="PASS" if exit_code == 0 and not changed else "FAIL",
        command=command, started_utc=started_utc,
        execution_mode="single_pytest_process" if args.workers == 1 else "parallel_complete_test_files",
        worker_count=args.workers, test_paths=list(test_paths), per_file_results=per_file_results,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        seconds=time.monotonic() - started, exit_code=exit_code,
        is_full_profile_execution=False, pytest_summary=summary,
        counts=counts,
        source_count=len(before), changed_paths=changed,
        runner_sha256=digest(Path(__file__)),
        sources_before_sha256=digest(args.out / "sources_before.json"),
        sources_after_sha256=digest(args.out / "sources_after.json"),
        log_sha256=digest(args.out / "pytest.log"))
    write(args.out / "receipt.json", receipt)
    print(json.dumps(receipt), flush=True)
    if receipt["status"] != "PASS":
        raise SystemExit(exit_code or 1)


if __name__ == "__main__":
    main()
