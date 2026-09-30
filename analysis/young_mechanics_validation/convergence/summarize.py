"""Compact sensitivity evidence; differences are not a convergence proof."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import yaml

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
REFERENCE = BASE.parent / "final/corrected_mechanics/elapsed_0050"


def diff(a,b,path=""):
    if isinstance(a,dict) and isinstance(b,dict):
        out={}
        for key in sorted(a.keys()|b.keys()):
            out.update(diff(a.get(key),b.get(key),f"{path}.{key}" if path else key))
        return out
    return {} if a==b else {path:[a,b]}


def read_case(path,label):
    report=json.loads((path/"continuation.json").read_text(encoding="utf-8"))
    config=yaml.safe_load((path/"mature_config.yaml").read_text(encoding="utf-8"))
    parameters=json.loads((path/"young_context/parameters.json").read_text(encoding="utf-8"))
    meta=json.loads((path/"mature_checkpoint/meta.json").read_text(encoding="utf-8"))
    inventory=meta["subduction_memory"].get("young_boundary_state")
    return dict(case=label,path=str(path),origin_time_myr=report["import"]["origin_time_myr"],
        final_time_myr=report["final_time_myr"],step_myr=report["step_myr"],
        subdivisions=config["mesh"]["subdivisions"],plate_count=report["final_plate_count"],
        mean_speed_km_myr=report["final_mean_surface_speed_km_myr"],
        max_speed_km_myr=report["final_max_surface_speed_km_myr"],
        transport_commits=report["transport_commits"],checks=report["checks"],
        first_commit_elapsed_myr=next((r["time_myr"]-report["import"]["origin_time_myr"]
            for r in report["history"] if r["transport_commits"]>0),None),
        continued_fracture_elapsed_times_myr=[r["time_myr"]-report["import"]["origin_time_myr"]
            for r in report["young_fracture_events"]],
        thermal_energy_relative_residual=report["history"][-1]["thermal_energy_relative_residual"],
        material_relative_volume_residual=report["material_ledger"]["relative_volume_residual"],
        cumulative_accepted_oceanic_volume_km3=0. if inventory is None else inventory["cumulative_accepted_oceanic_volume_km3"],
        config=config,parameters={k:parameters[k] for k in ("thermal","tides","shell","starter")})


def main():
    cases=[read_case(REFERENCE,"sub4_dt1_reference")]
    for label in ("sub4_dt0p5","sub5_dt1","sub3_dt1"):
        path=BASE/label
        if (path/"continuation.json").exists():
            cases.append(read_case(path,label))
    reference=cases[0]
    config_keys=("plate_dynamics","young_shell","subduction_memory","slab_breakoff","rollback")
    for case in cases:
        case["relative_mean_speed_change"]=case["mean_speed_km_myr"]/reference["mean_speed_km_myr"]-1.
        case["source_parameter_differences"]=diff(reference["parameters"],case["parameters"])
        case["equation_config_differences"]=diff({k:reference["config"].get(k) for k in config_keys},
            {k:case["config"].get(k) for k in config_keys})
    # Strip full parameter bodies only after comparisons have been calculated.
    for case in cases:
        del case["config"],case["parameters"]
    equations=("basal_coupling.py","young_plate_dynamics.py","young_boundary.py",
               "genesis_young_mechanics.py","genesis_starter_continuation.py","young_local_thermal.py")
    hashes={name:hashlib.sha256((ROOT/"tectonics"/name).read_bytes()).hexdigest()
            for name in equations if (ROOT/"tectonics"/name).exists()}
    result=dict(note="Same-source temporal refinement and spatial subdivision are separate sensitivity checks. Subdivision 3 rebuilds the partition rather than coarsening an existing world. A difference in geometric/material feedback is not convergence proof.",
        equation_hashes_at_summary=hashes,cases=cases)
    (BASE/"comparison.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
