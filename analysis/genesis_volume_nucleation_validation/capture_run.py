"""Capture real subprocess evidence and declared local import source closure.

The child keeps its exact command and ordinary Python execution semantics.
Fresh captures reside outside the repository. Source imports are inventoried
statically, without importing archived producers or copying the environment.
"""
from pathlib import Path
import argparse
import ast
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def source_paths(mode):
    paths = set((ROOT/"tectonics").glob("*.py"))
    for family in ("genesis_volume_continuation_validation",
                   "genesis_volume_fault_validation",
                   "genesis_volume_slip_validation",
                   "genesis_volume_bank_validation",
                   "genesis_volume_creation_validation",
                   "genesis_volume_insertion_validation",
                   "genesis_volume_location_validation"):
        paths.update((ROOT/"analysis"/family).glob("*.py"))
    new_family = ROOT/"analysis"/"genesis_volume_nucleation_validation"
    if mode == "control":
        # The independent oracle may still be authored while the authentic
        # checkpoint replay runs. Only called control sources enter this
        # inventory; the literal dependency closure below remains exhaustive.
        paths.update((new_family/"capture_run.py", new_family/"actual_control.py"))
    else:
        paths.update(new_family.glob("*.py"))
    if mode == "tests":
        for pattern in ("test_genesis_volume_nucleation*.py",
                        "test_genesis_volume_embedded*.py"):
            paths.update((ROOT/"tests").glob(pattern))
        paths.update((ROOT/"tests").glob("conftest.py"))
    return paths


def import_closure(paths):
    """Add literal local imports without executing any imported module."""
    result, pending = set(paths), list(paths)
    while pending:
        path = pending.pop()
        package = list(path.relative_to(ROOT).parts[:-1])
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = package[:len(package)-node.level+1] if node.level else []
                if node.module:
                    base += node.module.split(".")
                names = [".".join(base)]
                names += [".".join(base+[alias.name]) for alias in node.names if alias.name != "*"]
            for name in names:
                if not name or name.split(".")[0] not in ("analysis", "tectonics", "tests"):
                    continue
                local = ROOT.joinpath(*name.split("."))
                candidates = (local.with_suffix(".py"), local/"__init__.py")
                for candidate in candidates:
                    if candidate.is_file() and candidate not in result:
                        result.add(candidate); pending.append(candidate)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--source-mode", choices=("control", "tests", "audit"), required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not directory.is_relative_to(Path("/workspace")) or directory.is_relative_to(ROOT):
        raise ValueError("Capture must be fresh and outside the repository")
    directory.mkdir(exist_ok=False)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("Capture requires a real subprocess command")
    declared = source_paths(args.source_mode)
    paths = import_closure(declared)
    before = {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}
    for relative in before:
        target = directory/"source_snapshots"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, target)
        if digest(target) != before[relative]:
            raise ValueError("Source changed before subprocess execution")
    start = time.monotonic()
    with (directory/"stdout.log").open("xb") as out, (directory/"stderr.log").open("xb") as err:
        child = subprocess.run(command, cwd=ROOT, stdout=out, stderr=err)
    elapsed = time.monotonic()-start
    after = {r: digest(ROOT/r) for r in before}
    final_paths = {str(p.relative_to(ROOT)) for p in import_closure(source_paths(args.source_mode))}
    facts = dict(format="genesis-volume-nucleation-real-subprocess-capture-0.1",
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        command=command, source_mode=args.source_mode,
        actual_exit_code=child.returncode, actual_seconds=elapsed,
        source_before_sha256=before, source_after_sha256=after,
        declared_source_paths=sorted(str(p.relative_to(ROOT)) for p in declared),
        static_local_import_closure_paths=sorted(str(p.relative_to(ROOT)) for p in paths-declared),
        import_inventory_scope="literal_transitive_local_imports_not_dynamic_execution_trace",
        changed_sources=[r for r in before if before[r] != after[r]],
        newly_added_sources=sorted(final_paths-set(before)),
        stdout_raw_sha256=digest(directory/"stdout.log"),
        stderr_raw_sha256=digest(directory/"stderr.log"),
        stdout_bytes=(directory/"stdout.log").stat().st_size,
        stderr_bytes=(directory/"stderr.log").stat().st_size)
    threads = {}
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        value = os.environ.get(name)
        threads[name] = int(value) if value is not None and value.isdecimal() else None
    facts["public_numerical_thread_controls"] = threads
    facts["sha256"] = hashlib.sha256(canonical(facts)).hexdigest()
    (directory/"execution_facts.json").write_bytes(canonical(facts)+b"\n")
    print(json.dumps({key: facts[key] for key in ("actual_exit_code", "actual_seconds",
        "changed_sources", "newly_added_sources", "sha256")}, sort_keys=True))
    raise SystemExit(child.returncode or
        (2 if facts["changed_sources"] or facts["newly_added_sources"] else 0))


if __name__ == "__main__":
    main()
