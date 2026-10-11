"""Capture a real subprocess with frozen live physics sources and raw logs."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not directory.is_relative_to(Path('/workspace')) or directory.is_relative_to(ROOT):
        raise ValueError("Capture must be fresh and outside repository evidence")
    directory.mkdir(exist_ok=False)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        raise ValueError("Actual subprocess command is required")
    paths = set((ROOT/'tectonics').glob('*.py'))
    paths.update((ROOT/'tests').glob('test_genesis_volume_*.py'))
    paths.update((ROOT/'tests').glob('conftest.py'))
    paths.update(Path(__file__).parent.glob('*.py'))
    before = {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}
    for relative in before:
        destination = directory/'source_snapshots'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, destination)
        if digest(destination) != before[relative]:
            raise ValueError("Capture snapshot differs before execution")
    start = time.monotonic()
    with (directory/'stdout.log').open('xb') as out, (directory/'stderr.log').open('xb') as err:
        child = subprocess.run(command, cwd=ROOT, stdout=out, stderr=err)
    after = {relative: digest(ROOT/relative) for relative in before}
    changed = [relative for relative in before if before[relative] != after[relative]]
    facts = dict(format='genesis-volume-real-subprocess-capture-0.1',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        command=command, actual_exit_code=child.returncode,
        actual_seconds=time.monotonic()-start, source_before_sha256=before,
        source_after_sha256=after, changed_sources=changed,
        stdout_raw_sha256=digest(directory/'stdout.log'),
        stderr_raw_sha256=digest(directory/'stderr.log'),
        stdout_bytes=(directory/'stdout.log').stat().st_size,
        stderr_bytes=(directory/'stderr.log').stat().st_size)
    facts['sha256'] = hashlib.sha256(canonical(facts)).hexdigest()
    (directory/'execution_facts.json').write_bytes(canonical(facts)+b'\n')
    print(json.dumps({key: facts[key] for key in ('actual_exit_code', 'actual_seconds',
        'changed_sources', 'stdout_bytes', 'stderr_bytes', 'sha256')}, sort_keys=True))
    raise SystemExit(child.returncode or (2 if changed else 0))


if __name__ == '__main__':
    main()
