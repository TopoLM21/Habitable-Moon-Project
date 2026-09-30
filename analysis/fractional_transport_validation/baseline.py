"""Read-only mechanics0.5 transport baseline from existing validated saves.

No model imports, new evolution, or modifications to archived artifacts.
"""
from __future__ import annotations

import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PREVIOUS=ROOT/"analysis/slab_sinking_followup"
CASES={"sub4_dt1":"ordered_sub4_dt1", "sub4_dt0p5":"ordered_sub4_dt0p5",
    "sub5_dt1":"ordered_sub5_dt1_solver_fixed"}
TEST_FILES=("test_conservative_transport.py","test_continental_material_transport.py",
    "test_transport_cycle_memory.py","test_young_boundary_material.py","test_genesis_starter_ledger.py",
    "test_genesis_sinking_compatibility.py","test_genesis_starter_continuation_independent.py",
    "test_genesis_starter_continuation_cli.py","test_genesis_continuation_execution.py",
    "test_genesis_continuation_remesh.py","test_genesis_young_mechanics_regressions.py")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    values=np.asarray(list(values),dtype=float)
    if not values.size:
        return dict(count=0)
    return dict(count=int(values.size), minimum=float(values.min()), median=float(np.median(values)),
        mean=float(values.mean()), maximum=float(values.max()), p90=float(np.quantile(values,.9)))


def case(folder):
    path=PREVIOUS/"runs"/folder/"elapsed_0050"
    report=read(path/"continuation.json")
    meta=read(path/"mature_checkpoint/meta.json")
    metrics=read(path.with_suffix(".metrics.json"))
    origin=report["import"]["origin_time_myr"]
    dt=report["step_myr"]
    history={round(r["time_myr"]-origin,8):r for r in report["history"]}
    inventory=meta["subduction_memory"]["young_boundary_state"]
    events=defaultdict(lambda:dict(parcels=0,area_km2=0.,oceanic_volume_km3=0.,cold_volume_km3=0.,initial_density_excess_mass_kg=0.))
    parcels=[]
    for segment in inventory["segments"]:
        for cohort in segment["thermal_cohorts"]:
            time=round(cohort["acceptance_time_myr"]-origin,8)
            event=events[time]
            event["parcels"]+=1
            for target,source in (("area_km2","accepted_area_km2"),("oceanic_volume_km3","oceanic_volume_km3"),
                    ("cold_volume_km3","cold_mantle_volume_km3"),("initial_density_excess_mass_kg","initial_density_excess_mass_kg")):
                event[target]+=cohort[source]
            parcels.append(dict(elapsed_myr=time,**cohort))
    rows=[]
    for text in path.with_suffix(".dynamics.jsonl").read_text(encoding="utf-8").splitlines():
        raw=json.loads(text)
        time=round(raw["state_time_myr"]-origin,8)
        before=history.get(round(time-dt,8),{"transport_commits":0})
        hist=history[time]
        rows.append(dict(elapsed_myr=time,
            cumulative_raster_commits=hist["transport_commits"],new_raster_commits=hist["transport_commits"]-before["transport_commits"],
            mean_surface_speed_mm_yr=hist["mean_surface_speed_km_myr"],
            acceptance=events.get(time,dict(parcels=0,area_km2=0.,oceanic_volume_km3=0.,cold_volume_km3=0.,initial_density_excess_mass_kg=0.)),
            display_subduction_boundary_length_km=raw["dynamics_diagnostics"]["slab_boundary_length_km"],
            force_sections=len(raw["trace"].get("slab_sections",[])),
            neck_failures=len(raw["trace"].get("slab_neck_failures",[])),
            inventory_at_force_evaluation=raw["accepted_slab_inventory"]))
    first_commit=next((r["elapsed_myr"] for r in rows if r["new_raster_commits"]),None)
    first_accept=min(events,default=None)
    first_display_convergence=next((r["elapsed_myr"] for r in rows if r["display_subduction_boundary_length_km"]>0),None)
    active=[r for r in rows if first_accept is not None and r["elapsed_myr"]>=first_accept]
    accepting=[r for r in rows if r["acceptance"]["oceanic_volume_km3"]>0]
    summary=dict(path=str(path),version=report["mechanics_model_version"],dt_myr=dt,
        origin_time_myr=origin,mean_speed50_mm_yr=report["final_mean_surface_speed_km_myr"],
        first_raster_commit_myr=first_commit,first_acceptance_myr=first_accept,
        first_display_convergence_myr=first_display_convergence,
        first_force_section_myr=next((r["elapsed_myr"] for r in rows if r["force_sections"]),None),
        first_neck_failure_myr=next((r["elapsed_myr"] for r in rows if r["neck_failures"]),None),
        accepted_steps_without_same_step_raster_commit=sum(r["new_raster_commits"]==0 for r in accepting),
        zero_acceptance_steps_after_first_acceptance=sum(r["acceptance"]["oceanic_volume_km3"]==0 for r in active),
        timesteps_after_first_acceptance=len(active),raster_commits=report["transport_commits"],
        acceptance_burst_rate_km3_myr=stats(r["acceptance"]["oceanic_volume_km3"]/dt for r in accepting),
        accepted_parcel_area_km2=stats(p["accepted_area_km2"] for p in parcels),
        accepted_parcel_ocean_volume_km3=stats(p["oceanic_volume_km3"] for p in parcels),
        initial_parcel_cold_thickness_km=stats(p["initial_thickness_km"] for p in parcels),
        cumulative_ocean_volume_from_cohorts_km3=sum(p["oceanic_volume_km3"] for p in parcels),
        saved_inventory=report["accepted_slab_inventory"],checks=report["checks"],
        original_production_sha256=metrics["production_sha256"],
        original_source_sha256=metrics["source_sha256"],
        input_sha256={str(p):sha(p) for p in (path/"continuation.json",path/"mature_checkpoint/meta.json",
            path.with_suffix(".metrics.json"),path.with_suffix(".dynamics.jsonl"))})
    summary["cohort_sum_matches_cumulative_inventory"]=bool(np.isclose(summary["cumulative_ocean_volume_from_cohorts_km3"],
        inventory["cumulative_accepted_oceanic_volume_km3"],rtol=2e-14,atol=1e-7))
    return dict(summary=summary,rows=rows,acceptance_timeline=[dict(elapsed_myr=t,**e) for t,e in sorted(events.items())])


def main():
    HERE.mkdir(parents=True,exist_ok=True)
    cases={name:case(folder) for name,folder in CASES.items()}
    old=read(PREVIOUS/"validation_results.json")
    data=dict(scope="Read-only baseline from completed mechanics0.5 saves; no new simulations.",cases=cases,
        archived_main_endpoints=old["ordered05"],last100_myr_speed=old["last100_myr_speed_statistics"]["ordered05"],
        caveats=["Display-classified convergence is not the weak-boundary mechanical flux; it only demonstrates that existing transport delay coexists with resolved convergence.",
            "Trace inventory is sampled before the current material transaction; per-step acceptance is reconstructed independently from saved cohorts.",
            "No per-step donor budget or source identity was persisted in old accepted cohorts; donor-level double counting cannot be retrospectively ruled out from total ledgers alone."])
    (HERE/"baseline.json").write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    catalogue=[]
    for name in TEST_FILES:
        path=ROOT/"tests"/name
        tree=ast.parse(path.read_text(encoding="utf-8-sig"))
        tests=[dict(name=node.name,line=node.lineno) for node in ast.walk(tree)
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name.startswith("test_")]
        catalogue.append(dict(file=str(path),sha256=sha(path),tests=tests))
    (HERE/"test_catalogue.json").write_text(json.dumps(catalogue,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({name:{key:data["summary"][key] for key in ("mean_speed50_mm_yr","first_raster_commit_myr",
        "first_acceptance_myr","first_display_convergence_myr","accepted_steps_without_same_step_raster_commit",
        "zero_acceptance_steps_after_first_acceptance","timesteps_after_first_acceptance")} for name,data in cases.items()},indent=2))


if __name__=="__main__":
    main()
