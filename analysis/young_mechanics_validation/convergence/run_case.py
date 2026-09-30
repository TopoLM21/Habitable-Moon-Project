"""Run one independent default-young timestep/grid sensitivity experiment."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
BASE = Path(__file__).resolve().parent
SOURCE = ROOT / "results/genesis_runs/starter_20260928_192306_547551/starter_checkpoint.npz"


def build_sub3_source():
    from tectonics.genesis_starter_continuation import load_starter_source
    from tectonics.genesis_starter import StarterModel
    from tectonics.mesh import build_icosphere
    original, _, metadata = load_starter_source(SOURCE)
    shell = replace(original.shell, subdivisions=3)
    model = StarterModel(build_icosphere(3), original.thermal, original.tides, shell, original.parameters)
    state = model.initial_state()
    step = float(metadata["step_myr"])
    while not state.stopped_reason and state.time_myr < 10.:
        state = model.advance(state, min(10., state.time_myr+step))
    if state.stopped_reason != "first_partition":
        raise RuntimeError("Subdivision-3 rebuild did not produce a first partition by 10 Myr")
    output = BASE / "source_sub3"
    output.mkdir(exist_ok=False)
    parameters = deepcopy(metadata)
    parameters["shell"]["subdivisions"] = 3
    parameters["provenance"] = {"rebuilt_from_parameters_of": str(SOURCE),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "changed_fields": {"shell.subdivisions": [4,3]}}
    (output/"parameters.json").write_text(json.dumps(parameters, indent=2)+"\n",encoding="utf-8")
    model.save_state(output/"starter_checkpoint.npz",state)
    (output/"summary.json").write_text(json.dumps(model.diagnose(state),indent=2)+"\n",encoding="utf-8")
    return output/"starter_checkpoint.npz"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("sub4_dt0p5","sub5_dt1","sub3_dt1"))
    args = parser.parse_args()
    output = BASE / args.case
    log = BASE / f"{args.case}.log"
    if output.exists() or log.exists():
        raise ValueError("Sensitivity output must be new; no existing result is overwritten")
    started = time.perf_counter()
    source = build_sub3_source() if args.case=="sub3_dt1" else SOURCE
    dt = .5 if args.case=="sub4_dt0p5" else 1.
    command = [sys.executable,str(ROOT/"run_genesis_starter_continuation.py"),
        "--starter-checkpoint",str(source),"--output",str(output),"--duration-myr","50",
        "--step-myr",str(dt),"--cpu-workers","1","--render-workers","1",
        "--cell-kernels","--no-boundary-forces","--frame-interval","1000000",
        "--surface-only-frames"]
    if args.case=="sub5_dt1":
        command += ["--subdivisions","5"]
    env = {**os.environ,"PYTHONIOENCODING":"utf-8","MPLBACKEND":"Agg",
        "CUPY_CACHE_DIR":str(ROOT/".venv/cupy-cache")}
    with log.open("w",encoding="utf-8") as stream:
        result = subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    summary = dict(case=args.case,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        command=command,exit_code=result.returncode,elapsed_seconds=time.perf_counter()-started,
        step_myr=dt,subdivisions=5 if args.case=="sub5_dt1" else 3 if args.case=="sub3_dt1" else 4)
    if result.returncode==0:
        report = json.loads((output/"continuation.json").read_text(encoding="utf-8"))
        summary.update({key:report[key] for key in ("status","final_time_myr","final_plate_count",
            "final_mean_surface_speed_km_myr","final_max_surface_speed_km_myr","transport_commits","checks")})
        summary["origin_time_myr"]=report["import"]["origin_time_myr"]
        summary["thermal_energy_relative_residual"]=report["history"][-1]["thermal_energy_relative_residual"]
        summary["material_relative_residual"]=report["material_ledger"]["relative_volume_residual"]
    (BASE/f"{args.case}.metrics.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)
    if result.returncode:
        print(log.read_text(encoding="utf-8")[-6000:])
        raise SystemExit(result.returncode)


if __name__=="__main__":
    main()
