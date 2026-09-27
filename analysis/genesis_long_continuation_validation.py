"""Replay the user's 107 Myr failure and exercise the same owner at 4.5 Gyr."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    original = ROOT / "results/gui_runs/genesis_20260928_000016_583453/gui_checkpoint_0000101p0984_Myr"
    manifest = json.loads((original / "continuation.json").read_text(encoding="utf-8"))
    before = {p: digest(original / p) for p in manifest["checkpoint_sha256"]}
    cases = {}

    def run(name, source, duration, dt, starter=False):
        target = output / name
        command = [sys.executable, str(ROOT / "run_genesis_starter_continuation.py"),
                   "--starter-checkpoint" if starter else "--resume", str(source),
                   "--output", str(target), "--duration-myr", str(duration), "--step-myr", str(dt)]
        began = perf_counter()
        print("RUN", name, flush=True)
        with (output / f"{name}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=900)
        if result.returncode:
            raise RuntimeError(f"{name} exited {result.returncode}; see {output / (name+'.log')}")
        report = json.loads((target / "continuation.json").read_text(encoding="utf-8"))
        assert report["status"] == "completed" and all(report["checks"].values())
        assert report["mechanical_transition"] is not None
        cases[name] = {"path": str(target), "time_myr": report["final_time_myr"],
            "wall_seconds": perf_counter()-began, "checks": report["checks"],
            "mechanical_transition": report["mechanical_transition"],
            "plate_count": report["final_plate_count"], "ocean_fraction": report["ocean_fraction"]}
        (output / "progress.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
        print("OK", name, cases[name]["time_myr"], cases[name]["wall_seconds"], flush=True)
        return target

    recovered = run("user_recovered_111myr", original, 110., 1.)
    resumed = run("user_resumed_131myr", recovered, 130., 1.)
    whole = run("user_whole_131myr", original, 130., 1.)
    equality = {}
    for relative in ("mature_checkpoint/state.npz", "young_context/starter_checkpoint.npz",
                     "young_context/fracture_memory.npz"):
        with np.load(resumed / relative) as a, np.load(whole / relative) as b:
            equality[relative] = a.files == b.files and all(np.array_equal(a[k], b[k]) for k in a.files)
    assert all(equality.values()), equality
    starter = ROOT / "results/genesis_runs/starter_20260927_234820_027216/starter_checkpoint.npz"
    run("coarse_4500myr", starter, 4500., 4., starter=True)
    unchanged = all(digest(original / p) == h for p, h in before.items())
    assert unchanged
    result = {"cases": cases, "resume_arrays_equal": equality, "user_source_unchanged": unchanged}
    (output / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
