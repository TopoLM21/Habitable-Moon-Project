"""Small actual-state fractional-transport probes; never coupled evolution.

Run only after the experimental transport API is frozen. Outputs are new and
source files are integrity checked. Per-material-id budgets are independent of
the kernel's global diagnostics.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from run_fractional_transport_probe import (DEFAULT_SOURCE,load_probe_source,make_birth_factory,
    source_hashes,execute_probe,digest)
from tectonics.fractional_surface import state_to_json,surface_totals
from tectonics.fractional_surface_io import (surface_from_lithosphere,save_fractional_checkpoint,
    load_fractional_checkpoint,refine_fractional_surface)
from tectonics.fractional_transport import advance_fractional_transport
from tectonics.mesh import build_icosphere
from tectonics.transport import build_transport_map,initialize_transport_state

EXTENSIVE=("area_km2","oceanic_volume_km3","cold_mantle_volume_km3","density_excess_mass_kg")
SAVED50=ROOT/"analysis/slab_sinking_followup/runs/ordered_sub4_dt1/elapsed_0050"


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def hashes():
    paths=list((ROOT/"tectonics").glob("fractional_*.py"))+[ROOT/"run_fractional_transport_probe.py"]
    return {str(p.relative_to(ROOT)):digest(p) for p in paths}


def state_digest(state):
    return hashlib.sha256(json.dumps(state_to_json(state),sort_keys=True,allow_nan=False).encode()).hexdigest()


def add(ledger,parcels):
    for parcel in parcels:
        ledger[parcel.material_id]+=np.array([getattr(parcel,key) for key in EXTENSIVE])


def audit_identity(initial,remaining,births,losses):
    left,right=defaultdict(lambda:np.zeros(4)),defaultdict(lambda:np.zeros(4))
    add(left,initial.parcels)
    add(left,births)
    add(right,remaining.parcels)
    add(right,(loss.parcel for loss in losses))
    maxima=np.zeros(4)
    absolute=np.zeros(4)
    worst=[None]*4
    for material_id in left.keys()|right.keys():
        raw=np.abs(left[material_id]-right[material_id])
        relative=raw/np.maximum(np.maximum(np.abs(left[material_id]),np.abs(right[material_id])),1.)
        for index in range(4):
            if relative[index]>maxima[index]:
                maxima[index],worst[index]=relative[index],material_id
        absolute=np.maximum(absolute,raw)
    if np.any(maxima>5e-12):
        raise AssertionError(f"Material-id donor budget failed: {maxima}")
    return dict(material_id_count=len(left.keys()|right.keys()),
        maximum_relative_residual=dict(zip(EXTENSIVE,maxima.tolist())),
        maximum_absolute_residual=dict(zip(EXTENSIVE,absolute.tolist())),
        worst_material_id=dict(zip(EXTENSIVE,worst)))


def totals(parcels):
    parcels=tuple(parcels)
    return {key:math.fsum(getattr(p,key) for p in parcels) for key in EXTENSIVE}


def run_case(path,mesh,initial,omega,model,provenance,dt,*,duration=1.,common=False):
    path.mkdir()
    frozen=state_digest(initial)
    before=hashes()
    state=initial
    losses,births,history=[],[],[]
    factory=make_birth_factory(initial,model,provenance)
    for index in range(round(duration/dt)):
        result=advance_fractional_transport(mesh,state,omega,provenance["radius_km"],dt,birth_factory=factory)
        state=result.state
        losses.extend(result.losses)
        births.extend(result.births)
        audit=audit_identity(initial,state,births,losses)
        history.append(dict(elapsed_myr=(index+1)*dt,parcel_count=len(state.parcels),
            cumulative_losses=totals(x.parcel for x in losses),cumulative_births=totals(births),
            material_id_budget=audit,kernel_diagnostics=result.diagnostics))
    if state_digest(initial)!=frozen:
        raise AssertionError("Input immutable surface changed")
    if hashes()!=before:
        raise AssertionError("Experimental production changed during trial")
    if source_hashes(Path(provenance["source"]))!=provenance["source_sha256"]:
        raise AssertionError("Archived source changed")
    loss_total,birth_total=totals(x.parcel for x in losses),totals(births)
    if common and (loss_total["area_km2"]!=0. or birth_total["area_km2"]!=0.):
        raise AssertionError("Shared rigid rotation generated a spurious sink/source")
    save_fractional_checkpoint(path/"fractional_checkpoint.json",mesh,state,provenance["radius_km"],
        provenance=dict(experiment=provenance,scope="independent frozen-velocity transport probe"))
    owners=defaultdict(set)
    plate_area=defaultdict(float)
    for parcel in state.parcels:
        owners[parcel.cell].add(parcel.plate)
        plate_area[(parcel.cell,parcel.plate)]+=parcel.area_km2
    minority_area=math.fsum(state.cell_areas_km2[cell]-max(plate_area[(cell,plate)] for plate in plates)
        for cell,plates in owners.items())
    report=dict(source=provenance,dt_myr=dt,duration_myr=duration,common_rotation=common,
        initial_totals=surface_totals(initial),final_totals=surface_totals(state),
        multi_owner_cell_count=sum(len(value)>1 for value in owners.values()),
        multi_owner_cell_fraction=sum(len(value)>1 for value in owners.values())/state.cell_count,
        minority_owner_area_fraction=minority_area/math.fsum(state.cell_areas_km2),
        cumulative_losses=loss_total,cumulative_births=birth_total,history=history,
        first_positive_acceptance_myr=next((r["elapsed_myr"] for r in history if r["cumulative_losses"]["area_km2"]>0),None),
        source_unchanged=True,input_state_unchanged=True,production_sha256=before,
        final_state_sha256=state_digest(state),scope="No heat/force/fracture/topology evolution; no speed prediction")
    write(path/"report.json",report)
    print(json.dumps(dict(case=path.name,dt=dt,loss=loss_total,parcels=len(state.parcels))),flush=True)
    return report,state


def raster_reference(mesh,checkpoint,dt):
    rows=[]
    for label,transport in (("saved_residual",deepcopy(checkpoint.transport_state)),
            ("zero_residual_control",initialize_transport_state(len(checkpoint.system.plates)))):
        mapping=build_transport_map(mesh,checkpoint.system,checkpoint.state,dt,transport)
        rows.append(dict(kind=label,dt_myr=dt,committed_plates=mapping.diagnostics.committed_plates,
            overlap_target_cells=int(np.sum(mapping.covered.sum(axis=0)>1)),
            gap_target_cells=int(np.sum(mapping.covered.sum(axis=0)==0)),
            note="Raster map only: no winner resolution or inferred subducted mass; zero-residual control is explicitly reset, not an evolved save."))
    return rows


def distribution(state,parent_factor=1):
    result=defaultdict(float)
    for parcel in state.parcels:
        result[(parcel.cell//parent_factor,parcel.plate)]+=parcel.area_km2
    return result


def distribution_difference(first,second,parent_factor=1):
    a,b=distribution(first),distribution(second,parent_factor)
    return math.fsum(abs(a[key]-b[key]) for key in a.keys()|b.keys())/math.fsum(first.cell_areas_km2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    if output.exists():
        raise ValueError("Validation output must be new")
    output.mkdir(parents=True)
    sources={"starter":DEFAULT_SOURCE,"saved50":SAVED50}
    imports={}
    cases={}
    references={}
    snapshots={}
    for label,source in sources.items():
        mesh,cp,fracture,model,provenance=load_probe_source(source)
        surface=surface_from_lithosphere(mesh,cp.state,provenance["radius_km"],fracture_memory=fracture.memory)
        omega=np.array([p.euler_axis*p.angular_speed_rad_per_myr for p in cp.system.plates])
        imports[label]=(mesh,cp,fracture,model,provenance,surface,omega)
        references[label]=raster_reference(mesh,cp,1.)
        name=label+"_dt1"
        cases[name],snapshots[name]=run_case(output/name,mesh,surface,omega,model,provenance,1.)
    mesh,cp,fracture,model,provenance,surface,omega=imports["saved50"]
    for step,name in ((.5,"saved50_dt0p5"),(.25,"saved50_dt0p25")):
        cases[name],snapshots[name]=run_case(output/name,mesh,surface,omega,model,provenance,step)
    shared=np.tile(np.array([.0007,-.0004,.0002]),(len(cp.system.plates),1))
    cases["saved50_common_rotation"],_=run_case(output/"saved50_common_rotation",mesh,surface,shared,model,
        {**provenance,"common_omega_rad_per_myr":shared[0].tolist()},1.,common=True)
    fine=build_icosphere(model.shell.subdivisions+1)
    refined=refine_fractional_surface(surface,mesh,fine,provenance["radius_km"])
    birth_model=SimpleNamespace(shell=model.shell,strength_factor=np.repeat(model.strength_factor,4))
    fine_provenance={**provenance,"refinement":"conservative children; newborn strength inherits each parent's original field"}
    cases["saved50_refined_dt1"],fine_final=run_case(output/"saved50_refined_dt1",fine,refined,omega,birth_model,fine_provenance,1.)
    snapshot=output/"saved50_halfway.json"
    first=advance_fractional_transport(mesh,surface,omega,provenance["radius_km"],.5,
        birth_factory=make_birth_factory(surface,model,provenance))
    save_fractional_checkpoint(snapshot,mesh,first.state,provenance["radius_km"],provenance=provenance)
    restored,restored_provenance=load_fractional_checkpoint(snapshot,mesh,provenance["radius_km"])
    second=advance_fractional_transport(mesh,restored,omega,provenance["radius_km"],.5,
        birth_factory=make_birth_factory(restored,model,provenance))
    restart_equal=state_to_json(second.state)==state_to_json(snapshots["saved50_dt0p5"])
    if not restart_equal or restored_provenance!=provenance:
        raise AssertionError("Snapshot restart changed the transport state or provenance")
    summary=dict(scope="Independent fractional material experiment: prescribed frozen velocities, no coupled speed prediction",
        cases=cases,raster_reference=references,snapshot_restart_bitwise_equal=restart_equal,
        time_refinement=dict(dt1_vs_dt0p5_plate_area_l1_fraction=distribution_difference(snapshots["saved50_dt1"],snapshots["saved50_dt0p5"]),
            dt0p5_vs_dt0p25_plate_area_l1_fraction=distribution_difference(snapshots["saved50_dt0p5"],snapshots["saved50_dt0p25"])),
        grid_refinement=dict(parent_aggregated_plate_area_l1_fraction=distribution_difference(snapshots["saved50_dt1"],fine_final,4),
            coarse_loss=cases["saved50_dt1"]["cumulative_losses"],fine_loss=cases["saved50_refined_dt1"]["cumulative_losses"]),
        limitations=["First-order upwind interface diffusion remains; convergence of the mature coupled model is untested.",
            "Cell overlap lacks a resolved within-cell trench and cannot yet drive the slab force law.",
            "No sparse parcel is silently discarded; exact histories grow with the number of steps."])
    write(output/"validation.json",summary)
    print(json.dumps(dict(snapshot_restart_bitwise_equal=restart_equal,time_refinement=summary["time_refinement"],
        grid_refinement=summary["grid_refinement"]),indent=2),flush=True)


if __name__=="__main__":
    main()
