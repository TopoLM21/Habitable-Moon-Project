"""Record the current portable profile, with immutable before/after manifests."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
COMMAND = [".venv/bin/python", "scripts/run_cloud_tests.py"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hashes() -> dict[str, str]:
    result = {}
    for directory in ("tectonics", "tests", "scripts"):
        for path in sorted((ROOT / directory).rglob("*")):
            if not path.is_file() or any(
                part in {"__pycache__", ".pytest_cache"}
                for part in path.relative_to(ROOT).parts
            ) or path.suffix in {".pyc", ".pyo"}:
                continue
            result[path.relative_to(ROOT).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    for path in (Path(__file__).resolve(), OUT / "underthrust_case.py"):
        result[path.relative_to(ROOT).as_posix()] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return dict(sorted(result.items()))


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


def main() -> int:
    before = hashes()
    before_path = OUT / "portable_code_and_tests_before.json"
    after_path = OUT / "portable_code_and_tests_after.json"
    log_path = OUT / "final_cloud_tests.log"
    receipt_path = OUT / "cloud_test_receipt.json"
    # Exclusive creation prevents a later invocation from replacing evidence.
    for path in (before_path, after_path, log_path, receipt_path):
        if path.exists():
            raise FileExistsError(path)
    write_json(before_path, before)
    started = utc_now()
    start = time.perf_counter()
    print(json.dumps({"status": "running", "started_utc": started,
                      "files_hashed": len(before), "command": COMMAND}), flush=True)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("x", encoding="utf-8") as log:
        completed = subprocess.run(COMMAND, cwd=ROOT, env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
    seconds = time.perf_counter() - start
    after = hashes()
    write_json(after_path, after)
    output = log_path.read_text(encoding="utf-8")
    summary_lines = [line.strip() for line in output.splitlines()
                     if re.search(r"\b\d+ (?:passed|failed|error|errors|skipped|deselected|xfailed|xpassed)\b", line)
                     and re.search(r"\bin [0-9.]+s\b", line)]
    summary = summary_lines[-1] if summary_lines else None
    counts = {name: int(count) for count, name in re.findall(
        r"\b(\d+) (passed|failed|error|errors|skipped|deselected|xfailed|xpassed)\b",
        summary or "")}
    changes = [name for name in sorted(before.keys() | after.keys())
               if before.get(name) != after.get(name)]
    receipt = {
        "command": COMMAND,
        "cwd": ROOT.as_posix(),
        "started_utc": started,
        "finished_utc": utc_now(),
        "seconds": seconds,
        "exit_code": completed.returncode,
        "status": "passed" if completed.returncode == 0 and not changes else "failed",
        "pytest_summary": summary,
        "counts": counts,
        "failed_test_summaries": [line.strip() for line in output.splitlines()
                                 if line.startswith("FAILED ") or line.startswith("ERROR ")],
        "log": log_path.relative_to(ROOT).as_posix(),
        "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
        "code_and_tests_unchanged": not changes,
        "changed_code_or_test_paths": changes,
        "before_sha256_manifest": before_path.relative_to(ROOT).as_posix(),
        "before_manifest_sha256": hashlib.sha256(before_path.read_bytes()).hexdigest(),
        "after_sha256_manifest": after_path.relative_to(ROOT).as_posix(),
        "after_manifest_sha256": hashlib.sha256(after_path.read_bytes()).hexdigest(),
        "code_tests_and_scripts_file_count": len(before),
        "manifest_scope": ["tectonics/**/*", "tests/**/*", "scripts/**/*",
                           "analysis/geometric_underthrust_validation/run_portable_checks.py",
                           "analysis/geometric_underthrust_validation/underthrust_case.py"],
        "manifest_exclusions": ["__pycache__", ".pytest_cache", "*.pyc", "*.pyo"],
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
    }
    write_json(receipt_path, receipt)
    print(json.dumps({key: receipt[key] for key in (
        "status", "seconds", "exit_code", "pytest_summary", "counts",
        "code_and_tests_unchanged", "changed_code_or_test_paths")}), flush=True)
    return completed.returncode or (1 if changes else 0)


if __name__ == "__main__":
    raise SystemExit(main())
