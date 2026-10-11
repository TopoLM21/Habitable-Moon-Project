"""Retain real clipped-bulk subprocess outputs and literal local source closure.

This helper never imports mechanics producers. New captures are external and
fresh; numerical subprocesses use the three public single-thread controls.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
FAMILIES = (
    "genesis_origin_starter_validation", "genesis_bulk_birth_validation",
    "genesis_cohort_mechanics_validation",
    "genesis_volume_continuation_validation", "genesis_volume_fault_validation",
    "genesis_volume_slip_validation", "genesis_volume_bank_validation",
    "genesis_volume_creation_validation", "genesis_volume_insertion_validation",
    "genesis_volume_location_validation", "genesis_volume_nucleation_validation",
    "genesis_volume_seed_validation", "genesis_volume_memory_validation",
    "genesis_volume_release_validation", "genesis_volume_clipped_validation",
)
TEST_PATTERNS = ("test_genesis_volume_*.py",)
THREAD_CONTROLS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_paths(mode):
    paths = set((ROOT / "tectonics").glob("*.py"))
    for family in FAMILIES:
        paths.update((ROOT / "analysis" / family).glob("*.py"))
    if mode == "tests":
        for pattern in TEST_PATTERNS:
            paths.update((ROOT / "tests").glob(pattern))
        paths.update((ROOT / "tests").glob("conftest.py"))
    return paths


def import_closure(paths):
    """Inventory literal transitive imports without executing any source."""
    result, pending = set(paths), list(paths)
    while pending:
        path = pending.pop()
        package = list(path.relative_to(ROOT).parts[:-1])
        for count in range(1, len(package) + 1):
            initializer = ROOT.joinpath(*package[:count], "__init__.py")
            if initializer.is_file() and initializer not in result:
                result.add(initializer)
                pending.append(initializer)
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = package[:len(package) - node.level + 1] if node.level else []
                if node.module:
                    base += node.module.split(".")
                names = [".".join(base)] + [".".join(base + [alias.name])
                    for alias in node.names if alias.name != "*"]
            for name in names:
                if not name or name.split(".")[0] not in ("analysis", "tectonics", "tests"):
                    continue
                local = ROOT.joinpath(*name.split("."))
                for candidate in (local.with_suffix(".py"), local / "__init__.py"):
                    if candidate.is_file() and candidate not in result:
                        result.add(candidate)
                        pending.append(candidate)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--source-mode", choices=("control", "tests", "audit"), required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not directory.is_relative_to(Path("/workspace")) or directory.is_relative_to(ROOT):
        raise ValueError("Capture must be external, fresh and inside /workspace")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("Capture requires a real subprocess command")
    directory.mkdir(exist_ok=False)
    declared = source_paths(args.source_mode)
    paths = import_closure(declared)
    if any(p.is_symlink() for p in paths):
        raise ValueError("Literal source inventory must not contain symlinks")
    before = {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}
    for relative, expected in before.items():
        target = directory / "source_snapshots" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
        if digest(target) != expected:
            raise ValueError("Source changed before subprocess execution: " + relative)
    environment = os.environ.copy()
    environment.update({name: "1" for name in THREAD_CONTROLS})
    started = time.monotonic()
    with (directory / "stdout.log").open("xb") as out, (directory / "stderr.log").open("xb") as err:
        child = subprocess.run(command, cwd=ROOT, stdout=out, stderr=err, env=environment)
    elapsed = time.monotonic() - started
    after = {relative: digest(ROOT / relative) for relative in before}
    final_paths = {str(p.relative_to(ROOT)) for p in import_closure(source_paths(args.source_mode))}
    facts = dict(
        format="genesis-volume-clipped-bulk-real-subprocess-capture-0.1",
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        command=command, source_mode=args.source_mode, actual_exit_code=child.returncode,
        actual_seconds=elapsed, source_before_sha256=before, source_after_sha256=after,
        declared_source_paths=sorted(str(p.relative_to(ROOT)) for p in declared),
        static_local_import_closure_paths=sorted(str(p.relative_to(ROOT)) for p in paths - declared),
        import_inventory_scope="literal_transitive_local_imports_and_ancestor_initializers_not_dynamic_execution_trace",
        changed_sources=[relative for relative in before if before[relative] != after[relative]],
        newly_added_sources=sorted(final_paths - set(before)),
        stdout_raw_sha256=digest(directory / "stdout.log"),
        stderr_raw_sha256=digest(directory / "stderr.log"),
        stdout_bytes=(directory / "stdout.log").stat().st_size,
        stderr_bytes=(directory / "stderr.log").stat().st_size,
        public_numerical_thread_controls={name: int(environment[name]) for name in THREAD_CONTROLS},
        thread_control_method="Explicit public child environment overrides; no private numerical patches",
    )
    facts["sha256"] = hashlib.sha256(canonical(facts)).hexdigest()
    (directory / "execution_facts.json").write_bytes(canonical(facts) + b"\n")
    print(json.dumps({key: facts[key] for key in (
        "actual_exit_code", "actual_seconds", "changed_sources", "newly_added_sources", "sha256")}, sort_keys=True))
    raise SystemExit(child.returncode or (2 if facts["changed_sources"] or facts["newly_added_sources"] else 0))


if __name__ == "__main__":
    main()
