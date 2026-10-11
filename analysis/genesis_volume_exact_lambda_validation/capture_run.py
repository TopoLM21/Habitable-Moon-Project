"""Capture a real subprocess and the complete prior plus exact-lambda sources."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from analysis.genesis_volume_clipped_validation.capture_run import (
    canonical, digest, import_closure, THREAD_CONTROLS,
)

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / "analysis/genesis_volume_clipped_validation/evidence/package_manifest.json"


def source_paths(mode="tests", *, validate_prior=True):
    previous = json.loads(PRIOR.read_text())["source_sha256"]
    for name, expected in previous.items():
        if validate_prior and digest(ROOT/name) != expected:
            raise ValueError("Frozen prior source changed: "+name)
    paths = {ROOT/name for name in previous}
    family = ROOT/"analysis/genesis_volume_exact_lambda_validation"
    if mode == "tests":
        paths.update(family.glob("*.py"))
    else:
        names = ("actual_control.py", "angular_control.py", "source_control.py", "capture_run.py")
        if mode == "audit":
            names += ("actual_line_audit.py", "independent_line_audit.py")
        paths.update(family/name for name in names if (family/name).is_file())
    paths.update((ROOT/"tectonics").glob("genesis_volume_exact_lambda*.py"))
    if mode == "tests":
        paths.update((ROOT/"tests").glob("test_genesis_volume_exact_lambda*.py"))
    return import_closure({path for path in paths if path.is_file()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--source-mode", choices=("control", "tests", "audit"), default="tests")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    directory = args.directory.resolve()
    if not command or not directory.is_relative_to(Path("/workspace")) or directory.is_relative_to(ROOT):
        raise ValueError("Capture needs a subprocess and a fresh external /workspace directory")
    directory.mkdir(exist_ok=False)
    paths = source_paths(args.source_mode)
    if any(path.is_symlink() for path in paths):
        raise ValueError("Source capture forbids symlinks")
    before = {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)}
    for relative, expected in before.items():
        target = directory/"source_snapshots"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, target)
        if digest(target) != expected:
            raise ValueError("Source changed before execution: "+relative)
    environment = os.environ.copy()
    environment.update({name:"1" for name in THREAD_CONTROLS})
    started = time.monotonic()
    with (directory/"stdout.log").open("xb") as out, (directory/"stderr.log").open("xb") as err:
        child = subprocess.run(command, cwd=ROOT, env=environment, stdout=out, stderr=err)
    elapsed = time.monotonic()-started
    after, digest_errors = {}, {}
    for name in before:
        try:
            after[name] = digest(ROOT/name) if (ROOT/name).is_file() else None
        except OSError as error:
            after[name] = None
            digest_errors[name] = type(error).__name__+": "+str(error)
    inventory_error = None
    try:
        final_paths = {str(path.relative_to(ROOT)) for path in source_paths(args.source_mode, validate_prior=False)}
    except (OSError, ValueError, SyntaxError) as error:
        # The factual child exit and raw logs must survive an invalidated
        # post-run source inventory; this is an explicit capture refusal.
        final_paths = set(before)
        inventory_error = type(error).__name__+": "+str(error)
    facts = dict(format="genesis-exact-lambda-real-subprocess-capture-0.1",
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        command=command, source_mode=args.source_mode,
        source_inventory_policy="unchanged_prior_434_plus_declared_exact_lambda_mode_and_literal_import_closure",
        actual_exit_code=child.returncode, actual_seconds=elapsed,
        source_before_sha256=before, source_after_sha256=after,
        declared_source_paths=sorted(before), static_local_import_closure_paths=[],
        import_inventory_scope="literal_local_imports_and_ancestor_initializers_not_dynamic_trace",
        changed_sources=[name for name in before if before[name] != after[name]],
        newly_added_sources=sorted(final_paths-set(before)),
        post_run_source_inventory_error=inventory_error,
        post_run_source_digest_errors=digest_errors,
        stdout_raw_sha256=digest(directory/"stdout.log"), stderr_raw_sha256=digest(directory/"stderr.log"),
        stdout_bytes=(directory/"stdout.log").stat().st_size,
        stderr_bytes=(directory/"stderr.log").stat().st_size,
        public_numerical_thread_controls={name:1 for name in THREAD_CONTROLS},
        thread_control_method="Explicit public child environment overrides")
    facts["sha256"] = hashlib.sha256(canonical(facts)).hexdigest()
    (directory/"execution_facts.json").write_bytes(canonical(facts)+b"\n")
    print(json.dumps({key:facts[key] for key in ("actual_exit_code", "actual_seconds", "changed_sources",
          "newly_added_sources", "sha256")}, sort_keys=True))
    raise SystemExit(child.returncode or (2 if facts["changed_sources"] or facts["newly_added_sources"] or inventory_error or digest_errors else 0))


if __name__ == "__main__":
    main()
