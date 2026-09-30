"""Small dynamic-component experiments, with immutable prior source files.

These cases validate the basal-only coupled foundation. They do not predict
slab-driven plate speeds or certify convergence of the complete plate model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

STARTER = ROOT / "results/genesis_runs/starter_20260928_192306_547551/starter_checkpoint.npz"
LATE = ROOT / "analysis/slab_sinking_followup/runs/ordered_sub4_dt1/elapsed_0050"
CASES = {
    "starter_dt0p25": (STARTER, .5, .25),
    "late_dt1": (LATE, 2., 1.),
    "late_dt0p5": (LATE, 2., .5),
    "late_dt0p25": (LATE, 2., .25),
    "late_restart_first": (LATE, 1., 1.),
    "late_restart_second": (LATE, 1., 1.),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.case == "late_restart_second" and args.resume is None:
        parser.error("late_restart_second requires the explicit saved fractional --resume checkpoint")
    from run_fractional_coupling_probe import execute_probe
    source, duration, step = CASES[args.case]
    output = args.output or HERE / "runs" / args.case
    result = execute_probe(source, output, duration, step, resume=args.resume, time_scheme="endpoint_v1")
    code = [ROOT / "run_fractional_coupling_probe.py", ROOT / "run_fractional_transport_probe.py"]
    code += sorted((ROOT / "tectonics").glob("fractional_*.py"))
    metadata = dict(case=args.case, source=str(source), duration_myr=duration, step_myr=step,
        output=str(output.resolve()), code_sha256={str(p.relative_to(ROOT)): sha(p) for p in code},
        report_sha256=sha(output / "report.json"), report=result)
    output.with_suffix(".validation.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(case=args.case, output=str(output.resolve()),
        report_sha256=metadata["report_sha256"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
