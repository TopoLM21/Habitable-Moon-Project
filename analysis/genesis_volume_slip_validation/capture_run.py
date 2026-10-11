"""Capture a real child process and every declared executing source.

Only public thread-count controls are recorded. No general environment or
credential values are inspected. Fresh outputs never overwrite evidence.
"""
from pathlib import Path
import argparse
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
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--source-mode', choices=('control', 'tests', 'audit'), required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not directory.is_relative_to(Path('/workspace')) or directory.is_relative_to(ROOT):
        raise ValueError('Capture must be fresh and outside repository')
    directory.mkdir(exist_ok=False)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        raise ValueError('Capture requires a real subprocess command')
    paths = set((ROOT/'tectonics').glob('*.py'))
    for family in ('genesis_volume_continuation_validation', 'genesis_volume_fault_validation'):
        paths.update((ROOT/'analysis'/family).glob('*.py'))
    paths.add(Path(__file__))
    if args.source_mode == 'control':
        paths.add(Path(__file__).parent/'actual_control.py')
    else:
        paths.update(Path(__file__).parent.glob('*.py'))
    if args.source_mode == 'tests':
        paths.update((ROOT/'tests').glob('test_genesis_volume_*.py'))
        paths.update((ROOT/'tests').glob('conftest.py'))
    before = {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}
    for relative in before:
        target = directory/'source_snapshots'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, target)
        if digest(target) != before[relative]:
            raise ValueError('A source snapshot changed before execution')
    start = time.monotonic()
    with (directory/'stdout.log').open('xb') as out, (directory/'stderr.log').open('xb') as err:
        child = subprocess.run(command, cwd=ROOT, stdout=out, stderr=err)
    after = {r: digest(ROOT/r) for r in before}
    facts = dict(format='genesis-volume-slip-real-subprocess-capture-0.1',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        command=command, source_mode=args.source_mode,
        actual_exit_code=child.returncode, actual_seconds=time.monotonic()-start,
        source_before_sha256=before, source_after_sha256=after,
        changed_sources=[r for r in before if before[r] != after[r]],
        stdout_raw_sha256=digest(directory/'stdout.log'),
        stderr_raw_sha256=digest(directory/'stderr.log'),
        stdout_bytes=(directory/'stdout.log').stat().st_size,
        stderr_bytes=(directory/'stderr.log').stat().st_size)
    public_threads = {}
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        raw = os.environ.get(name)
        # Only public numeric controls are recorded; unexpected text is never
        # printed or copied from the environment.
        public_threads[name] = int(raw) if raw is not None and raw.isdecimal() else None
    facts['public_numerical_thread_controls'] = public_threads
    facts['sha256'] = hashlib.sha256(canonical(facts)).hexdigest()
    (directory/'execution_facts.json').write_bytes(canonical(facts)+b'\n')
    print(json.dumps({k:facts[k] for k in ('actual_exit_code','actual_seconds','changed_sources','sha256')}, sort_keys=True))
    raise SystemExit(child.returncode or (2 if facts['changed_sources'] else 0))


if __name__ == '__main__':
    main()
