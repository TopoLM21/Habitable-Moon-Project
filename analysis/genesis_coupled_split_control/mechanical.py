"""Reusable fixed-catalogue bounded mechanical correction; no production hook."""
import argparse, hashlib, json, resource, time, traceback
from dataclasses import replace, asdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from tectonics.genesis import SECONDS_PER_MYR, GenesisParameters
from tectonics.genesis_cutcell_loading import prepare_saved_loading
from tectonics.genesis_cutcell_exchange import local_work, apply_exchange
from tectonics.genesis_implicit_enthalpy import invert
from tectonics.genesis_volume_loading import prepare_volume_loading, VolumeCohortMemory, RadialMoments
from tectonics.genesis_volume_slip_checkpoint import load_volume_slip_checkpoint
from tectonics.genesis_volume_operator import evaluate_genesis_volume_operator
from tectonics.genesis_volume_fault_dynamics import volume_face_response
from tectonics.genesis_volume_fault_evolution import evaluate_volume_fault_evolution, volume_fault_history_to_dict
from tectonics.genesis_volume_temporal import cone_homothetic_to_dict
from tectonics.genesis_shell import principal_tensile
from analysis.genesis_connected_step_validation.run import PARENT, PARENT_SHA, FORMATION_SHA, gate, deadline
from analysis.genesis_be_candidate.solver import solve_trial
from analysis.genesis_be_candidate.adapter import BEMemory, dumps
from analysis.genesis_be_candidate.checkpoint import seal
from analysis.genesis_be_candidate.audit import trial as independent_trial
from analysis.genesis_volume_fault_validation.independent_audit import independent_damage_law
from analysis.genesis_cutcell_mechanical_trial.audit import audit_feedback
from analysis.genesis_be_candidate.adapter import preparation_view


def advance(model,parent,root,packet,radial_moments,*,orbit_after=None,diagnostic_max_step_years=50):
    r=root['runtime_state'];report={'correctors':[]}
    template=SimpleNamespace(after_ids=tuple(c.id for c in parent.mechanics.cohorts),radial_moments=radial_moments)
    report['before_gate']=gate(parent.fault_history,model.fault_parameters)
    t=packet['time_myr'];end_orbit=replace(parent.thermal.orbit,time_myr=t) if orbit_after is None else orbit_after
    if end_orbit.time_myr!=t:raise ValueError('Endpoint orbit clock mismatch')
    dt=(t-parent.time_myr)*SECONDS_PER_MYR
    if diagnostic_max_step_years not in (50,100):raise ValueError('Unsupported diagnostic step cap')
    assert 0 < t-parent.time_myr <=diagnostic_max_step_years/1e6+1e-14
    ids=list(template.after_ids); index={x['material_id']:i for i,x in enumerate(packet['fixed_cut_geometry']) if x['material_id'] is not None};idx=np.array([index[i] for i in ids])
    m=np.array(packet['mass_kg']);base=np.array(packet['energy_j']);T=np.array(packet['temperature_k']);initial_T=T.copy()
    gp=GenesisParameters(**root['source_packet']['source']['parameters']['thermal']);n=len(T)
    cp=np.full(n,gp.silicate_heat_capacity_j_kg_k);latent=np.full(n,gp.silicate_latent_heat_j_kg);offset=np.zeros(n)
    cp[-1]=gp.water_heat_capacity_j_kg_k;latent[-1]=0;offset[-1]=gp.water_latent_heat_j_kg
    zero=RadialMoments((0.,)*800,(0.,)*800,(0.,)*800)
    assembly0,basal0=model._assembly(parent.geometry,parent.fault_history.damage)
    def mechanical(temperatures,history):
        assembly,basal=model._assembly(parent.geometry,history.damage)
        loading=prepare_volume_loading(preparation_view(parent.mechanics),after_ids=ids,after_temperature_k=temperatures[idx],
            before_temperature_k=[c.temperature_k for c in parent.mechanics.cohorts],birth_temperature_k=gp.solidus_k,
            radial_moments=template.radial_moments,before_radial_moments=template.radial_moments,removed_radial_moments=zero,
            before_damage=parent.fault_history.damage[assembly.face_indices],after_damage=history.damage[assembly.face_indices],parameters=model.shell_parameters,after_time_myr=t)
        du,v,e0,e1,L=solve_trial(basal.mass_matrix.toarray(),basal.basal_drag_matrix.toarray(),basal.external_force_n,parent.velocity_m_s,dt,assembly.dofs,assembly.B0,assembly.B1,loading)
        memory=BEMemory(parent.mechanics.source_identity,t,parent.mechanics.accepted_steps+1,tuple(VolumeCohortMemory(cid,float(temperatures[index[cid]]),e0[i],e1[i]) for i,cid in enumerate(ids)),L['elastic_after_j'])
        X=np.array(parent.geometry.vertices_m)+(parent.displacement_m+du).reshape(-1,6)[:,:3]
        driver=volume_face_response(assembly,memory,history.damage,X,model.shell_parameters)
        endpoint=evaluate_genesis_volume_operator(assembly,np.zeros(assembly.ndof),elastic_constant=e0,elastic_inverse_scale=e1)
        if endpoint.maximum_elastic_principal_strain>model.elastic_strain_limit:raise ValueError('Elastic strain gate')
        residual=basal.mass_matrix@(v-parent.velocity_m_s)/dt+basal.basal_drag_matrix@v+endpoint.force-basal.external_force_n
        rel=float(np.linalg.norm(residual)/max(np.linalg.norm(endpoint.force),np.linalg.norm(basal.external_force_n),1))
        if rel>model.contact_parameters.equilibrium_tolerance:raise ValueError('Momentum gate')
        L.update(kinetic_before_j=float(.5*parent.velocity_m_s@basal0.mass_matrix@parent.velocity_m_s),elastic_before_j=parent.mechanics.elastic_energy_j,birth_kinetic_inflow_j=0.,damage_energy_change_j=loading.damage_energy_change_j,remelt_kinetic_outflow_j=0.,remelt_elastic_outflow_j=0.)
        state=dict(geometry=r['geometry'],thermal={'time_myr':t},mechanics=asdict(memory),fault_history=volume_fault_history_to_dict(history),displacement_m=(parent.displacement_m+du).tolist(),velocity_m_s=v.tolist(),interval_ledger=L)
        inputs=model._audit_input(parent,assembly,loading,du,basal)
        return state,inputs,loading,e0,e1,driver,X,assembly
    previous=None
    for iteration in range(6):
        if np.any(T[idx]>=gp.solidus_k):raise ValueError('Mixed old material not admitted')
        print('corrector '+str(iteration+1)+' predictor/final',flush=True)
        pred=mechanical(T,parent.fault_history)
        driver,X=pred[5:7]
        faults,evolution=evaluate_volume_fault_evolution(parent.fault_history,model.source_packet,after_time_myr=t,physical_vertices_m=X,face_stress_pa=driver['stress_pa'],face_mean_temperature_k=driver['mean_temperature_k'],retained_fraction=np.ones(80),active_material=driver['active_material'],surface_temperature_k=float(T[-1]),ocean_fraction=0.,orbit_before=parent.thermal.orbit,orbit_after=end_orbit,weak_plane_slip_available=True)
        gate(faults,model.fault_parameters)
        healing_guard(parent.fault_history.damage,faults.damage)
        final=mechanical(T,faults);s,inputs,loading,e0,e1=final[:5]
        faults=replace(faults,peak_tensile_pa=np.maximum(faults.peak_tensile_pa,principal_tensile(final[5]['stress_pa'])))
        gate(faults,model.fault_parameters);s['fault_history']=volume_fault_history_to_dict(faults)
        Q,W=local_work(loading,e0,e1)
        E,exchanges=apply_exchange(base,[x['material_id'] for x in packet['fixed_cut_geometry']]+[None,None],ids,Q,W)
        nextT=invert(E/m,cp,latent,np.full(n,gp.solidus_k),np.full(n,gp.liquidus_k),offset)
        error=float(np.max(np.abs(nextT-T)))
        exchange_error=None if previous is None else float(max(np.max(np.abs(Q-previous[0])/(1e3+3e-8*np.abs(Q))),np.max(np.abs(W-previous[1])/(1e3+3e-8*np.abs(W)))))
        report['correctors'].append(dict(index=iteration+1,temperature_error_k=error,exchange_error_over_budget=exchange_error));print(json.dumps(report['correctors'][-1]),flush=True)
        if error<=1e-8 and exchange_error is not None and exchange_error<=1:break
        previous=(Q,W);T=nextT
    else:raise ValueError('Six-corrector budget exhausted')
    if np.any(nextT[idx]>=gp.solidus_k):raise ValueError('Feedback induces unsupported melting')
    report['intervals']=1
    certificate=model._certificate(parent,replace(final[7],time_myr=t,geometry=replace(parent.geometry,time_myr=t)),np.array(s['displacement_m']))
    s['last_certificate']=cone_homothetic_to_dict(certificate)
    driver_payload=dict(physical_vertices_m=pred[6].tolist(),stress_pa=pred[5]['stress_pa'].tolist(),mean_temperature_k=pred[5]['mean_temperature_k'].tolist(),retained_fraction=[1.]*80,active_material=pred[5]['active_material'].tolist(),surface_temperature_k=float(T[-1]),ocean_fraction=0.)
    L=s['interval_ledger']
    if L['damage_energy_change_j']>0:raise ValueError('Damage storage would need unknown source')
    result=dict(format='genesis-cutcell-BE-feedback-0.1',parent_sha256=PARENT_SHA,thermal_sha256=packet['sha256'],scheme='fixed_conduction_then_endpoint_mechanical_feedback',state=s,inputs=inputs,predictor=pred[0],predictor_inputs=pred[1],predictor_driver=driver_payload,
        thermal=dict(time_myr=t,mass_kg=m.tolist(),energy_j=E.tolist(),temperature_k=nextT.tolist(),mechanical_loading_temperature_k=T.tolist(),phase_fraction=np.clip((nextT[:-1]-gp.solidus_k)/(gp.liquidus_k-gp.solidus_k),0,1).tolist()),
        exchanges=exchanges,storage=dict(interface_drag_energy_j=L['physical_drag_j'],damage_energy_j=-L['damage_energy_change_j'],thermal_feedback='unknown; no heat applied',numerical_sink_j=L['numerical_kinetic_j']+L['numerical_memory_j']),correctors=report['correctors'],unresolved=['prescribed external work boundary','drag thermal partition','damage thermal partition','conduction/mechanics splitting accuracy','closure sensitivity','frozen orbital parameters in damage driver'])
    if orbit_after is not None:
        result['orbit_after']=asdict(end_orbit)
        result['unresolved'].remove('frozen orbital parameters in damage driver')
    return json.loads(dumps(seal(result)))


def healing_guard(before,after):
    delta=np.asarray(after)-np.asarray(before)
    if not np.isfinite(delta).all() or np.any(delta<0):
        raise ValueError('UNFUNDED_LOCAL_HEALING: no constitutive energy donor defined')
