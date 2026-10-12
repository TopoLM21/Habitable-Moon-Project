"""Exactly one exploratory 100 vs 2x50 diagnostic. Never admits a state."""
import argparse,copy,hashlib,json,os,platform,resource,shutil,time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import numpy as np
from tectonics.genesis import GenesisParameters
from analysis.genesis_adaptive_control import backend as b
from analysis.genesis_adaptive_control.controller import time_limit
from analysis.genesis_forced_pair_validation.run import PairBackend,Timings,TH50,TH50_SHA,THERMAL_AUDIT,THERMAL_AUDIT_SHA
from analysis.genesis_coupled_split_control.run import pinned
from analysis.genesis_coupled_split_control.mechanical import advance as original_mechanical
from analysis.genesis_volume_fault_validation.independent_audit import independent_damage_law
from analysis.genesis_invariant_fastpath.cache import FastPath,bindings_for_geometry,digest
from analysis.genesis_invariant_fastpath.replay import code_identity
from .checks import thermal_check,observe,assess


def configuration():
    c=json.loads(Path(__file__).with_name('config.json').read_text())
    v=json.loads(Path('analysis/genesis_adaptive_control/config.json').read_text())
    for k in ('global_work_atol_j_per_year','global_work_rtol_stored','local_work_atol_j_per_year','local_work_rtol_stored','damage_gate_margin','damage_error_fraction_of_margin'):
        if c[k]!=v[k]:raise ValueError('VERIFIED_GUARD_CHANGED: '+k)
    if c['horizon_years']!=100 or c['runtime_seconds']!=600 or c['max_attempts']!=1:raise ValueError('EXACT_ONE_PAIR_CONFIGURATION')
    return c


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume-from',type=Path);a=p.parse_args()
    if Path('/workspace').is_symlink() or not Path('/workspace').is_dir():raise ValueError('REAL_WORKSPACE_REQUIRED')
    if platform.system()!='Linux' or np.finfo(np.longdouble).nmant!=63:raise ValueError('PRECISION')
    if shutil.disk_usage('/workspace').free<2200000000:raise ValueError('DISK_RESERVE')
    prior=None;corrected=None;budget=600.;prior_bytes=0
    if a.resume_from:
        prior=json.loads((a.resume_from/'report.json').read_text())
        if prior['completed_trials']!=0 or prior['reason']!='ValueError: Source endpoint energy audit':raise ValueError('RESUME_ONLY_SOURCE_STOP')
        corrected=json.loads((a.resume_from/'corrected_source100.json').read_text())
        # Charge ten seconds conservatively for isolated diagnosis/tests as well.
        budget=600-prior['seconds']-10.
        prior_bytes=sum(x.stat().st_size for x in a.resume_from.iterdir() if x.is_file())
        if budget<=0:raise ValueError('NO_REMAINING_BUDGET')
    c=configuration();out=a.output;out.mkdir(exist_ok=False);used=0;start=time.perf_counter();backend=None;timings=None
    canonical=b.PARENT.read_bytes();result=dict(status='STOPPED',accepted=False,canonical_advanced=False,new_trials_started=0,completed_trials=0)
    def write(name,obj):
        nonlocal used
        data=json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        if prior_bytes+used+len(data)>199000000 or shutil.disk_usage(out).free-len(data)<2000000000:raise RuntimeError('ARTIFACT_OR_DISK_LIMIT')
        with (out/name).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        used+=len(data)
    try:
      with time_limit(budget):
        resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
        root=pinned(b.PARENT,b.PARENT_SHA);whole=pinned(b.SEED,b.SEED_SHA);th=pinned(TH50,TH50_SHA);ta=pinned(THERMAL_AUDIT,THERMAL_AUDIT_SHA)
        initial=pinned('/workspace/genesis-forced-pair-25-01/initial.json',json.loads(Path('/workspace/genesis-forced-pair-25-01/report.json').read_text())['output_raw_sha256']['initial.json'])
        check=thermal_check(initial['packet'],th['fixed_cut_geometry'],root,GenesisParameters(**root['source_packet']['source']['parameters']['thermal']))
        write('preflight.json',check);write('config.json',c);result['preflight']=check
        if check['unsupported']:raise ValueError('PREFLIGHT_UNSUPPORTED: '+','.join(check['unsupported']))
        backend=PairBackend(write,out,root,whole,th,ta)
        if backend.initial!=initial:raise ValueError('INITIAL_BINDING')
        write('initial.json',initial);result['setup_seconds']=time.perf_counter()-start
        code,manifest,runtime=code_identity()
        for path in Path(__file__).parent.glob('*.py'):manifest[str(path.resolve().relative_to(Path.cwd()))]=hashlib.sha256(path.read_bytes()).hexdigest()
        code=digest((manifest,runtime));write('provenance.json',dict(manifest=manifest,runtime=runtime,solver_sha256=code))
        timings=Timings(backend);cache=FastPath(bindings_for_geometry(backend.parent.geometry,code));snapshots={};observations={};events={}
        def mechanical(*args,**kw):return original_mechanical(*args,**kw,diagnostic_max_step_years=100)
        original_ports=b.apply_ports;source_reused=False
        def ports(before,gp,tp,orbit,target):
            nonlocal source_reused
            if corrected is not None and backend.label=='whole100':
                if source_reused or before!=corrected['source_input'] or corrected['ports']['orbit_before']!=__import__('dataclasses').asdict(orbit):raise ValueError('CORRECTED_SOURCE_BINDING')
                if corrected['orbit']['time_myr']!=target:raise ValueError('CORRECTED_SOURCE_CLOCK')
                b.audit_ports(before,corrected['source_endpoint'],corrected['ports'],gp,tp)
                source_reused=True
                return copy.deepcopy(corrected['source_endpoint']),b.TidalOrbitState(**corrected['orbit']),copy.deepcopy(corrected['ports'])
            return original_ports(before,gp,tp,orbit,target)
        with ExitStack() as stack:
            stack.enter_context(patch.object(b,'apply_ports',ports))
            stack.enter_context(cache)
            stack.enter_context(patch.object(b,'mechanical',mechanical))
            for name in ('apply_ports','conduct','mechanical','audit_forced','decode'):
                stack.enter_context(patch.object(b,name,timings.wrap(name,getattr(b,name))))
            for label,h in (('whole100',100.),('half1',50.),('half2',50.)):
                before=initial if label!='half2' else snapshots['half1']
                backend.label=label;result['new_trials_started']+=1
                remaining=budget-(time.perf_counter()-start)
                if remaining<=0:raise TimeoutError('COMPUTE_LIMIT')
                after=b.ForcedBackend.advance(backend,before,h,label,min(300.,remaining))
                after=json.loads((out/(label+'.json')).read_bytes())
                proof=backend.audit(before,after);write(label+'_audit.json',proof)
                packet=after['packet'];ev=thermal_check(packet,backend.cuts,root,backend.gp)
                pre=before['packet'];expected,driver=independent_damage_law(pre['state']['fault_history'],packet['predictor_driver'],backend.source,pre['orbit_after'],packet['orbit_after'],packet['state']['thermal']['time_myr'])
                d0=np.array(pre['state']['fault_history']['damage']);d1=np.array(expected['damage']);load=np.array(driver['nonlocal_loading_per_myr'])
                ev['damage_frozen_driver']=dict(max_loading_per_myr=float(load.max()),max_change=float(abs(d1-d0).max()),
                    maximum_along_frozen_driver=float(np.maximum(d0,d1).max()),minimum_gate_margin=float(.65-np.maximum(d0,d1).max()),
                    full_path_certified=False,reason='Exact monotone exponential for frozen stage driver only; stress/temperature driver between stages is not enclosed')
                ev['thermal_stages']=[]
                for name in ('source_endpoint','thermal_packet'):
                    stage={'state':pre['state'],'thermal':packet[name]}
                    ev['thermal_stages'].append(dict(stage=name,check=thermal_check(stage,backend.cuts,root,backend.gp)))
                write(label+'_events.json',ev);events[label]=ev
                if ev['unsupported']:raise ValueError('UNSUPPORTED_EVENT: '+str(ev['unsupported']))
                snapshots[label]=after;observations[label]=observe(packet,backend.source,initial['packet']);result['completed_trials']+=1
        assessment=assess(observations['whole100'],observations['half1'],observations['half2'],100.,c)
        result.update(status='PAIR_COMPLETE_DIAGNOSTIC_ONLY',assessment=assessment,output_budgets_pass=not assessment.get('fatal') and assessment['score']<=1,
            full_path_events_certified=False,reason='No complete physical interval enclosure; no acceptance even if output budgets pass',events=events,cache=cache.report())
    except Exception as exc:
        result.update(status='STOPPED',reason=f'{type(exc).__name__}: {exc}')
    finally:
        result.update(seconds=time.perf_counter()-start,artifact_bytes=used,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,canonical_unchanged=b.PARENT.read_bytes()==canonical)
        if timings:result['timings']=timings.entries
        result.update(remaining_budget_seconds=budget,prior_seconds=prior['seconds'] if prior else 0.,isolated_work_charge_seconds=10. if prior else 0.,aggregate_charged_seconds=(prior['seconds']+10. if prior else 0.)+result['seconds'])
        with (out/'report.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ('timings','events','cache')},indent=2),flush=True)


if __name__=='__main__':main()
