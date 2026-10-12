"""Bounded BE chain: immutable root, exact predecessor link, separate ledgers."""
from dataclasses import replace
import hashlib,json
import numpy as np
from analysis.genesis_be_candidate.adapter import trial,no_remelt,state_payload,BEState,gate
from tectonics.genesis_volume_thermal import prepare_volume_thermal_step
from tectonics.genesis_volume_fault_dynamics import retained_live_fraction
from tectonics.genesis_volume_fault_evolution import evaluate_volume_fault_evolution
from tectonics.genesis_volume_slip_dynamics import _vertices
from tectonics.genesis import temperatures,diagnose
from tectonics.genesis_shell import principal_tensile
from tectonics.genesis_volume_checkpoint import _jsonable
from analysis.genesis_connected_step_validation.run import PARENT_SHA

FORMAT='genesis-be-chain-0.1'
SUMS=('external_work_j','internal_work_j','thermal_work_j','physical_drag_j','physical_maxwell_j',
      'numerical_kinetic_j','numerical_memory_j','birth_kinetic_inflow_j','damage_energy_change_j',
      'remelt_kinetic_outflow_j','remelt_elastic_outflow_j')
START=('kinetic_before_j','elastic_before_j')
END=('kinetic_after_j','elastic_after_j')


def accumulate(previous,interval):
    out={k:interval[k]+(0. if previous is None else previous[k]) for k in SUMS}
    out.update({k:interval[k] if previous is None else previous[k] for k in START})
    out.update({k:interval[k] for k in END})
    out['balance_residual_j']=balance(out)
    return out


def balance(L):
    return L['kinetic_after_j']+L['elastic_after_j']-L['kinetic_before_j']-L['elastic_before_j']-L['external_work_j']-L['thermal_work_j']-L['birth_kinetic_inflow_j']-L['damage_energy_change_j']+sum(L[k] for k in ['physical_drag_j','physical_maxwell_j','numerical_kinetic_j','numerical_memory_j','remelt_kinetic_outflow_j','remelt_elastic_outflow_j'])


def seal(packet):
    return {**packet,'sha256':hashlib.sha256(json.dumps(packet,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()}


def transition(model,parent,previous,dt):
    if dt not in (5e-5,2.5e-5):raise ValueError('Only bounded 50/25-year control')
    if previous is None:
        model.validate_state(parent);sizes=[];prefix=PARENT_SHA
    else:
        if not isinstance(parent,BEState) or state_payload(parent)!=previous['state']:raise ValueError('Parent state differs from chain')
        sizes=previous['step_sizes_myr'];prefix=previous['sha256']
    if sum(sizes)+dt>.0001+1e-14:raise ValueError('Outside original 100-year interval')
    gate(parent.fault_history,model.fault_parameters)
    thermal=prepare_volume_thermal_step(parent.thermal,parent.history,parent.geometry,dt,displacement_m=parent.displacement_m)
    no_remelt(parent.geometry,thermal.geometry);retained=retained_live_fraction(parent.geometry,thermal.geometry)
    d0=parent.fault_history.damage*retained
    h0=replace(parent.fault_history,time_myr=thermal.state.time_myr,accepted_steps=parent.fault_history.accepted_steps+1,damage=d0)
    predictor,pred_report=trial(model,parent,thermal,d0,h0,False)
    driver=model.face_response(predictor);vertices=_vertices(predictor)
    _,surface=temperatures(np.asarray(thermal.state.global_state.energy),thermal.state.source_boundary_parameters.thermal)
    ocean=diagnose(thermal.state.global_state,thermal.state.source_boundary_parameters.thermal)['ocean_fraction']
    faults,evolution=evaluate_volume_fault_evolution(parent.fault_history,model.source_packet,
        after_time_myr=thermal.state.time_myr,physical_vertices_m=vertices,face_stress_pa=driver['stress_pa'],face_mean_temperature_k=driver['mean_temperature_k'],
        retained_fraction=retained,active_material=driver['active_material'],surface_temperature_k=surface,ocean_fraction=ocean,
        orbit_before=parent.thermal.orbit,orbit_after=thermal.state.orbit,weak_plane_slip_available=True)
    gate(faults,model.fault_parameters)
    final,report=trial(model,parent,thermal,faults.damage,faults,True)
    tensile=principal_tensile(model.face_response(final)['stress_pa'])
    final=replace(final,fault_history=replace(faults,peak_tensile_pa=np.maximum(faults.peak_tensile_pa,tensile)),
        be_accumulated_ledger=accumulate(None if previous is None else previous['state']['be_accumulated_ledger'],final.interval_ledger))
    driver_payload=dict(physical_vertices_m=vertices,stress_pa=driver['stress_pa'],mean_temperature_k=driver['mean_temperature_k'],retained_fraction=retained,
                        active_material=driver['active_material'],surface_temperature_k=float(surface),ocean_fraction=float(ocean))
    detail=dict(final=report,predictor=pred_report,predictor_state=state_payload(predictor),predictor_driver=_jsonable(driver_payload),
                fault_evolution=_jsonable(evolution),final_face_tensile_pa=tensile.tolist())
    packet=seal(dict(format=FORMAT,parent_sha256=PARENT_SHA,predecessor_sha256=prefix,step_sizes_myr=sizes+[dt],
        source_packet=model.source_packet,state=state_payload(final),report=detail))
    return final,packet
