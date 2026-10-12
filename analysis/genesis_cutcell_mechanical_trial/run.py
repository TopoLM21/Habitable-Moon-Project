"""One bounded BE50 interval with fixed conduction and local energy feedback."""
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
from .audit import audit_feedback

THERMAL=Path('/workspace/genesis-implicit-cutcell-01/after.json')
THERMAL_SHA='632354594ccda5aaffeb8933d9ed95be96cb53037a2ce67d18cf407562d6f2e8'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);out=parser.parse_args().output;out.mkdir(exist_ok=False)
    start=time.monotonic();report=dict(status='RUNNING',intervals=0,canonical_advanced=False,correctors=[])
    write=lambda name,x:(out/name).write_text(dumps(x))
    raw=PARENT.read_bytes(); thermal_raw=THERMAL.read_bytes()
    try:
      with deadline(300):
        resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
        assert hashlib.sha256(raw).hexdigest()==PARENT_SHA and hashlib.sha256(thermal_raw).hexdigest()==THERMAL_SHA
        root=json.loads(raw); packet=json.loads(thermal_raw);r=root['runtime_state']
        template=prepare_saved_loading(raw,packet)
        print('loading authenticated parent',flush=True)
        model,parent=load_volume_slip_checkpoint(PARENT,expected_source_checkpoint_sha256=FORMATION_SHA)
        report['before_gate']=gate(parent.fault_history,model.fault_parameters)
        t=packet['time_myr'];dt=(t-parent.time_myr)*SECONDS_PER_MYR
        assert abs(t-parent.time_myr-5e-5)<1e-14
        ids=list(template.after_ids); index={x['material_id']:i for i,x in enumerate(packet['fixed_cut_geometry']) if x['material_id'] is not None};idx=np.array([index[i] for i in ids])
        m=np.array(packet['mass_kg']);base=np.array(packet['energy_j']);T=np.array(packet['temperature_k']);initial_T=T.copy()
        gp=GenesisParameters(**root['source_packet']['source']['parameters']['thermal']);n=len(T)
        cp=np.full(n,gp.silicate_heat_capacity_j_kg_k);latent=np.full(n,gp.silicate_latent_heat_j_kg);offset=np.zeros(n)
        cp[-1]=gp.water_heat_capacity_j_kg_k;latent[-1]=0;offset[-1]=gp.water_latent_heat_j_kg
        zero=RadialMoments((0.,)*800,(0.,)*800,(0.,)*800)
        assembly0,basal0=model._assembly(parent.geometry,parent.fault_history.damage)
        def mechanical(temperatures,history):
            assembly,basal=model._assembly(parent.geometry,history.damage)
            loading=prepare_volume_loading(parent.mechanics,after_ids=ids,after_temperature_k=temperatures[idx],
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
            faults,evolution=evaluate_volume_fault_evolution(parent.fault_history,model.source_packet,after_time_myr=t,physical_vertices_m=X,face_stress_pa=driver['stress_pa'],face_mean_temperature_k=driver['mean_temperature_k'],retained_fraction=np.ones(80),active_material=driver['active_material'],surface_temperature_k=float(T[-1]),ocean_fraction=0.,orbit_before=parent.thermal.orbit,orbit_after=replace(parent.thermal.orbit,time_myr=t),weak_plane_slip_available=True)
            gate(faults,model.fault_parameters)
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
        result=dict(format='genesis-cutcell-BE-feedback-0.1',parent_sha256=PARENT_SHA,thermal_sha256=THERMAL_SHA,scheme='fixed_conduction_then_endpoint_mechanical_feedback',state=s,inputs=inputs,predictor=pred[0],predictor_inputs=pred[1],predictor_driver=driver_payload,
            thermal=dict(time_myr=t,mass_kg=m.tolist(),energy_j=E.tolist(),temperature_k=nextT.tolist(),mechanical_loading_temperature_k=T.tolist(),phase_fraction=np.clip((nextT[:-1]-gp.solidus_k)/(gp.liquidus_k-gp.solidus_k),0,1).tolist()),
            exchanges=exchanges,storage=dict(interface_drag_energy_j=L['physical_drag_j'],damage_energy_j=-L['damage_energy_change_j'],thermal_feedback='unknown; no heat applied',numerical_sink_j=L['numerical_kinetic_j']+L['numerical_memory_j']),correctors=report['correctors'],unresolved=['external traction donor','drag thermal partition','damage thermal partition','conduction/mechanics splitting accuracy','closure sensitivity','frozen orbital parameters in damage driver'])
        result=seal(result);write('candidate.json',result)
        report['predictor_audit']=independent_trial(r,pred[0],pred[1],root['source_packet'])
        report['mechanical_audit']=independent_trial(r,s,inputs,root['source_packet'])
        expected,_=independent_damage_law(r['fault_history'],driver_payload,root['source_packet'],r['thermal']['orbit'],r['thermal']['orbit'],t)
        for name,value in expected.items():
            if name=='peak_tensile_pa':value=np.maximum(value,principal_tensile(final[5]['stress_pa']))
            if not np.allclose(value,s['fault_history'][name],rtol=3e-10,atol=1e-12):raise ValueError('Damage law '+name)
        report['feedback_audit']=audit_feedback(root,packet,result)
        restored=json.loads((out/'candidate.json').read_bytes());body={k:v for k,v in restored.items() if k!='sha256'}
        assert seal(body)==restored
        typed=tuple(VolumeCohortMemory(**c) for c in restored['state']['mechanics']['cohorts'])
        restored['state']['mechanics']['cohorts']=[asdict(c) for c in typed]
        assert dumps(restored)==dumps(result)
        audit_feedback(root,packet,restored);write('restart.json',restored)
        assert (out/'restart.json').read_bytes()==(out/'candidate.json').read_bytes()
        report.update(status='PASS',restart_byte_identical=True,after_gate=gate(faults,model.fault_parameters),max_feedback_temperature_change_k=float(np.max(np.abs(nextT-initial_T))),phase_compatible=True,old_ids=800,births=0,retirements=0,ledger=L,storage=result['storage'])
    except Exception as exc:
        report.update(status='STOPPED',error=str(exc));(out/'failure.txt').write_text(traceback.format_exc())
    finally:
        report.update(parent_unchanged=PARENT.read_bytes()==raw,thermal_input_unchanged=THERMAL.read_bytes()==thermal_raw,seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write('report.json',report)
    print(json.dumps({k:report[k] for k in ('status','seconds','intervals')}|({'error':report['error']} if 'error' in report else {})),flush=True)


if __name__=='__main__':main()
