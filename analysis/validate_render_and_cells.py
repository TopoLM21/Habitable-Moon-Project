"""Validate combined accelerations through canonical merger and split events."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.benchmark_cpu_modes import compare_checkpoints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT / "results/gui_runs/stable_copy_0_700"
    rows = []
    for name, start, end in [("merger", 320, 340), ("split", 620, 640)]:
        case = output / name
        case.mkdir()
        resume = source / f"gui_checkpoint_{start:06d}_Myr"
        reference = source / f"gui_checkpoint_{end:06d}_Myr"
        command = [sys.executable, str(ROOT / "run_long_evolution_v131_cpu.py"),
                   "--cpu-workers", "1", "--render-workers", "4", "--cell-kernels",
                   "--config", str(source / "gui_runtime_config.yaml"), "--resume", str(resume),
                   "--end-time", str(end), "--dt", "4", "--frame-interval", "20",
                   "--output", str(case), "--checkpoint", str(case / "checkpoint")]
        print("START", name, flush=True)
        with (case / "run.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=ROOT, env=dict(os.environ, MPLBACKEND="Agg", PYTHONPATH=str(ROOT)),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        comparison = compare_checkpoints(reference, case / "checkpoint")
        rows.append({"case": name, "start": start, "end": end, **comparison})
        (output / "validation.json").write_text(json.dumps(rows, indent=2))
        print(rows[-1], flush=True)
        if not comparison["exact"]:
            raise RuntimeError("Combined acceleration changed state/history")


if __name__ == "__main__":
    main()
