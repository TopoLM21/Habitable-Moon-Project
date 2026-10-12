"""Exactly two new 25-year coupled steps, then compare saved BE50; 600 s cap."""
import argparse,hashlib,json,resource,time,traceback
from dataclasses import replace,asdict
from pathlib import Path
import numpy as np
from tectonics.genesis import GenesisParameters
from tectonics.genesis_cutcell_loading import prepare_saved_loading
from tectonics.genesis_volume_slip_checkpoint import load_volume_slip_checkpoint
from tectonics.genesis_volume_loading import VolumeCohortMemory
from tectonics.genesis_volume_fault_evolution import volume_fault_history_from_dict
from analysis.genesis_be_candidate.adapter import BEMemory,dumps
from analysis.genesis_be_candidate.checkpoint import seal
from analysis.genesis_be_candidate.audit import face_driver
from .audit import trial as mechanical_audit
from analysis.genesis_volume_fault_validation.independent_audit import independent_damage_law,tensile_principal
from analysis.genesis_connected_step_validation.run import PARENT,PARENT_SHA,FORMATION_SHA,gate,deadline
from analysis.genesis_cutcell_mechanical_trial.audit import audit_feedback
from tectonics.genesis_energy_ports import boundary_budget
from .mechanical import advance
from .thermal import conduct,phase_guard


def pinned(path,sha):
    raw=Path(path).read_bytes();assert hashlib.sha256(raw).hexdigest()==sha
    return json.loads(raw)


def root_from_previous(root,p):
    r=dict(root['runtime_state']);s=p['state']
    r.update(displacement_m=s['displacement_m'],velocity_m_s=s['velocity_m_s'],fault_history=s['fault_history'],mechanics={'state':s['mechanics']})
    r['thermal']={**r['thermal'],'time_myr':p['thermal']['time_myr'],'orbit':{**r['thermal']['orbit'],'time_myr':p['thermal']['time_myr']}}
    return {**root,'runtime_state':r}


def decode(parent,p,source):
    s=p['state'];m=s['mechanics'];t=p['thermal']['time_myr']
    memory=BEMemory(m['source_identity'],m['time_myr'],m['accepted_steps'],tuple(VolumeCohortMemory(**c) for c in m['cohorts']),m['elastic_energy_j'])
    # Only this private trial view uses the legacy thermal carrier for clocks.
    # Authoritative enthalpy is always p['thermal'], never the old column fields.
    return replace(parent,mechanics=memory,fault_history=volume_fault_history_from_dict(s['fault_history'],source),displacement_m=np.array(s['displacement_m']),velocity_m_s=np.array(s['velocity_m_s']),thermal=replace(parent.thermal,time_myr=t,orbit=replace(parent.thermal.orbit,time_myr=t)))


def audit_step(root,thermal,p,source):
    assert seal({k:v for k,v in p.items() if k!='sha256'})==p
    r=root['runtime_state'];s=p['state'];old=r['fault_history']
    assert np.all(np.array(s['fault_history']['damage'])>=old['damage']), 'UNFUNDED_LOCAL_HEALING'
    for state in (p['predictor'],s):
        f=state['fault_history'];assert not any(f['fault_active']) and not any(f['fault_candidate_age_myr']) and max(f['damage'])<source['source']['parameters']['faults']['activation_damage']
    pred=mechanical_audit(r,p['predictor'],p['predictor_inputs'],source);final=mechanical_audit(r,s,p['inputs'],source)
    assert s['mechanics']['accepted_steps']==r['mechanics']['state']['accepted_steps']+1
    assert s['fault_history']['accepted_steps']==old['accepted_steps']+1
    expected,_=independent_damage_law(old,p['predictor_driver'],source,r['thermal']['orbit'],p.get('orbit_after',r['thermal']['orbit']),p['thermal']['time_myr'])
    tensile=tensile_principal(face_driver(s,source)['stress_pa'])
    for k,v in expected.items():
        if k=='peak_tensile_pa':v=np.maximum(v,tensile)
        assert np.allclose(v,s['fault_history'][k],rtol=3e-10,atol=1e-12),k
    feedback=audit_feedback(root,thermal,p)
    budget=boundary_budget(s['interval_ledger'],feedback['heat_increment_j'])
    assert abs(budget['balance_residual_j'])<=feedback['combined_allowance_j']
    return dict(status='PASS',predictor=pred,mechanical=final,feedback=feedback,open_boundary_budget=budget)


def compare(reference,result,cuts,p):
    a=np.array(reference['thermal']['temperature_k']);b=np.array(result['thermal']['temperature_k']);mass=np.array(result['thermal']['mass_kg'])
    ea=np.array(reference['thermal']['energy_j']);eb=np.array(result['thermal']['energy_j']);pa=np.clip((a[:-1]-p.solidus_k)/(p.liquidus_k-p.solidus_k),0,1);pb=np.clip((b[:-1]-p.solidus_k)/(p.liquidus_k-p.solidus_k),0,1)
    def metric(x,y):
        x=np.asarray(x);y=np.asarray(y);d=y-x
        return dict(max_abs=float(np.max(np.abs(d))),rms=float(np.sqrt(np.mean(d*d))),relative_l2=float(np.linalg.norm(d)/max(np.linalg.norm(x),np.linalg.norm(y),1e-300)))
    def front(T):
        value=[]
        for f in range(80):
            q=0.
            for i,c in enumerate(cuts):
                if c['face']!=f:continue
                if T[i]>=p.solidus_k:break
                q=c['q_hi']
            value.append(q)
        return value
    dr=face_driver(reference['state'],reference['source_packet'] if 'source_packet' in reference else SOURCE)
    ds=face_driver(result['state'],SOURCE)
    from analysis.genesis_volume_continuation_validation import independent_audit as q
    g=result['state']['geometry'];M=q.mass_matrix(np.array(g['vertices_m']),np.array(g['faces']),np.array(g['column_mass_kg']),g['density_kg_m3'],g['cohorts']);total_mass=sum(c['mass_kg'] for c in g['cohorts'])
    va=np.array(reference['state']['velocity_m_s']);vb=np.array(result['state']['velocity_m_s']);norm=lambda v:float(np.sqrt(v@M@v/total_mass))
    physical_velocity=dict(v50_rms_m_s=norm(va),v25_rms_m_s=norm(vb),difference_rms_m_s=norm(vb-va),relative_mass_norm=norm(vb-va)/max(norm(va),norm(vb)))
    return dict(physical_velocity=physical_velocity,temperature_k=metric(a,b),specific_enthalpy_j_kg=metric(ea/mass,eb/mass),melt_fraction=metric(pa,pb),column_liquid_mass_difference_kg=float(np.dot(mass[:3360],pb[:3360]-pa[:3360])),front_q=metric(front(a),front(b)),velocity_m_s=metric(reference['state']['velocity_m_s'],result['state']['velocity_m_s']),stress_pa=metric(dr['stress_pa'],ds['stress_pa']),damage=metric(reference['state']['fault_history']['damage'],result['state']['fault_history']['damage']),interpretation='Combined temporal/splitting sensitivity only; two resolutions do not isolate splitting error or prove convergence.')


def main():
    global SOURCE
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path);out=parser.parse_args().output;out.mkdir(exist_ok=False)
    start=time.monotonic();report=dict(status='RUNNING',steps_started=0,steps_passed=0,compute_limit_seconds=600,stages=[],canonical_advanced=False)
    def write(name,x):
        with (out/name).open('x') as f:f.write(dumps(x))
    raw=PARENT.read_bytes()
    try:
      with deadline(600):
        resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
        root=pinned(PARENT,PARENT_SHA);SOURCE=root['source_packet']
        cut=pinned('/workspace/genesis-cutcell-feasibility-02/candidate.json','6006521e7b3e22ce23894e63248d0145ee15d3065333c949701c730c6cab47ad')
        th50=pinned('/workspace/genesis-implicit-cutcell-01/after.json','632354594ccda5aaffeb8933d9ed95be96cb53037a2ce67d18cf407562d6f2e8')
        baseline=pinned('/workspace/genesis-cutcell-mechanical-02/candidate.json','5a0cfcd3edfd57f4f05b59e8b408b09c25ac2a704068609e22029ab865230f2e')
        proj=pinned('/workspace/genesis-coupled-thermal-projection-03/projected.json','d25fe1fd4e8b56b18ab8ec697b197c326d723e9421148e5d3726fb7affdd45a4')
        p=GenesisParameters(**SOURCE['source']['parameters']['thermal']);cuts=th50['fixed_cut_geometry'];assert len(cuts)==3360
        cells={c['identity']:c for c in proj['state']['cells']}
        before=dict(time_myr=root['runtime_state']['thermal']['time_myr'],mass_kg=th50['mass_kg'],energy_j=[c['mass_kg']*c['specific_h'] for c in cut['cells']]+[(cells[k]['energy_j']/cells[k]['mass_kg'])*cells[k]['mass_kg'] for k in ('deep_mantle','surface_water')])
        initial_thermal=dict(before)
        template=prepare_saved_loading(raw,th50)
        print('load parent',flush=True);model,parent=load_volume_slip_checkpoint(PARENT,expected_source_checkpoint_sha256=FORMATION_SHA)
        current_root=root;previous_sha=PARENT_SHA;cumulative={};packets=[]
        for step in (1,2):
            report['steps_started']+=1;print('coupled step '+str(step)+' thermal',flush=True)
            thermal,ta=conduct(before,cuts,current_root,p,2.5e-5);write(f'thermal_{step}.json',thermal);write(f'thermal_audit_{step}.json',ta)
            result=advance(model,parent,current_root,thermal,template.radial_moments)
            phase_guard(cuts,np.array(result['thermal']['temperature_k']),p,np.array(thermal['temperature_k']))
            result.pop('sha256');result.update(format='genesis-cutcell-coupled-chain-0.1',predecessor_sha256=previous_sha,step_index=step,thermal_packet=thermal)
            result=seal(result);write(f'step_{step}.json',result)
            print('step '+str(step)+' independent audit',flush=True);aud=audit_step(current_root,thermal,result,SOURCE);write(f'audit_{step}.json',aud)
            # Restore the private typed mechanical state from serialized bytes.
            decoded=json.loads((out/f'step_{step}.json').read_bytes());assert seal({k:v for k,v in decoded.items() if k!='sha256'})==decoded
            restored=decode(parent,decoded,SOURCE)
            assert json.loads(dumps(asdict(restored.mechanics)))==decoded['state']['mechanics']
            gate(restored.fault_history,model.fault_parameters)
            write(f'restart_{step}.json',decoded);assert (out/f'step_{step}.json').read_bytes()==(out/f'restart_{step}.json').read_bytes()
            report['steps_passed']+=1;report['stages'].append(dict(step=step,correctors=len(result['correctors']),thermal_audit=ta,feedback_audit=aud['feedback'],mechanical_audit=aud['mechanical'],restart_byte_identical=True))
            for k in ('external_work_j','physical_drag_j','physical_maxwell_j','thermal_work_j','damage_energy_change_j','numerical_kinetic_j','numerical_memory_j'):
                cumulative[k]=cumulative.get(k,0.)+result['state']['interval_ledger'][k]
            previous_sha=hashlib.sha256((out/f'step_{step}.json').read_bytes()).hexdigest();parent=restored;before=decoded['thermal'];current_root=root_from_previous(current_root,decoded);packets.append(decoded)
        assert abs(before['time_myr']-baseline['thermal']['time_myr'])<1e-14
        end=packets[-1]['state'];initial=root['runtime_state']
        combined={**cumulative,'kinetic_before_j':packets[0]['state']['interval_ledger']['kinetic_before_j'],'kinetic_after_j':end['interval_ledger']['kinetic_after_j'],'elastic_before_j':initial['mechanics']['state']['elastic_energy_j'],'elastic_after_j':end['mechanics']['elastic_energy_j']}
        heat_delta=float(np.sum(np.array(before['energy_j'],dtype=np.longdouble)-np.array(initial_thermal['energy_j'],dtype=np.longdouble)))
        whole=boundary_budget(combined,heat_delta)
        whole_allowance=sum(x['thermal_audit']['global_allowance_j']+x['feedback_audit']['combined_allowance_j'] for x in report['stages'])
        assert abs(whole['balance_residual_j'])<=whole_allowance
        report['cumulative_open_budget']=whole;report['cumulative_allowance_j']=whole_allowance
        report['feedback_temperature_lag_k']=[float(np.max(np.abs(np.array(p['thermal']['temperature_k'])-p['thermal_packet']['temperature_k']))) for p in packets]
        report.update(status='PASS',comparison=compare(baseline,packets[-1],cuts,p),cumulative_ledger=cumulative,final_sha256=previous_sha,unresolved_heat_return='zero only for this diagnostic window',sequential_runner_usable=True)
    except Exception as exc:
        report.update(status='STOPPED',error_type=type(exc).__name__,error=str(exc),sequential_runner_usable=False);(out/'failure.txt').write_text(traceback.format_exc())
    finally:
        report.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,canonical_unchanged=PARENT.read_bytes()==raw);write('report.json',report)
    print(json.dumps({k:report[k] for k in ('status','steps_started','steps_passed','seconds')}|({'error':report['error']} if 'error' in report else {})),flush=True)


if __name__=='__main__':main()
