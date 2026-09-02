"""Check optimized and stable continuation of an optimized checkpoint."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from benchmark_cpu_modes import compare_checkpoints

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stable-runner", type=Path, default=ROOT / "run_long_evolution_v131.py")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    initial_time = float(json.loads((args.resume / "meta.json").read_text())["time_myr"])
    final_time = float(json.loads((args.reference / "meta.json").read_text())["time_myr"])
    split_time = initial_time + 8.0
    if not split_time < final_time:
        parser.error("Reference must be later than resume + 8 Myr")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    environment["PYTHONUNBUFFERED"] = "1"
    cases = [
        ("optimized_first", ROOT / "run_long_evolution_v131_cpu.py", args.resume, split_time),
        ("optimized_resume", ROOT / "run_long_evolution_v131_cpu.py", output / "optimized_first" / "checkpoint", final_time),
        ("stable_resume", args.stable_runner.resolve(), output / "optimized_first" / "checkpoint", final_time),
    ]
    results = []
    for name, runner, resume, end in cases:
        case = output / name
        case.mkdir()
        command = [sys.executable, str(runner), "--config", str(args.config.resolve()),
                   "--resume", str(resume.resolve()), "--end-time", f"{end:g}", "--dt", "4",
                   "--output", str(case), "--checkpoint", str(case / "checkpoint")]
        if name.startswith("optimized"):
            command += ["--cpu-workers", str(args.workers)]
        # For the unmodified stable checkout, use its own module search root.
        case_env = environment.copy()
        case_env["PYTHONPATH"] = str(runner.parent)
        print("START", name, flush=True)
        with (case / "run.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=runner.parent, env=case_env, stdout=log, stderr=subprocess.STDOUT, check=True)
        if name != "optimized_first":
            result = {"case": name, **compare_checkpoints(args.reference, case / "checkpoint")}
            results.append(result)
            print(json.dumps(result), flush=True)
            if not result["exact"]:
                raise RuntimeError("Checkpoint continuation differs from stable reference")
    (output / "validation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
