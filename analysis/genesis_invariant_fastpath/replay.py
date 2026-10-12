"""Profile uncached/cached saved algebra and independent audits, no time steps."""
import argparse,copy,hashlib,json,resource,subprocess,sys,time
from collections import defaultdict
from contextlib import ExitStack,nullcontext
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import scipy
from .cache import FastPath,bindings_for_geometry,digest
from analysis.genesis_adaptive_control.controller import time_limit
from analysis.genesis_adaptive_control import backend as b
from analysis.genesis_be_candidate import solver
from analysis.genesis_coupled_split_control.audit import trial as independent_trial
from tectonics.genesis_volume_loading import RadialMoments
from tectonics.genesis_volume_slip_checkpoint import load_volume_slip_checkpoint

REPO=Path(__file__).resolve().parents[2]
PAIRS=Path('/workspace/genesis-forced-pair-25-01')
PINS={'001_whole.json':'4ac5efaeb55948a226379bbc184f298248a2e09a67a423699748790e0d3a0539',
      '001_half1.json':'f9f0dff6dad2dedfd52529536939d2c8eacf3554dba1a552a96839091ef7c5fa',
      '001_half2.json':'5e8d430efa592c06be35ea967757ad81e63f0a0191706fa8f612173182e75fdb'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def code_identity():
    paths=subprocess.check_output(['git','ls-files','tectonics','analysis/genesis_be_candidate','analysis/genesis_backward_euler_prototype','analysis/genesis_coupled_split_control','analysis/genesis_cutcell_mechanical_trial','analysis/genesis_volume_continuation_validation','analysis/genesis_volume_fault_validation','analysis/genesis_forced_coupled_validation','analysis/genesis_adaptive_control'],cwd=REPO,text=True).splitlines()
    paths+=['analysis/genesis_invariant_fastpath/cache.py','analysis/genesis_invariant_fastpath/replay.py']
    manifest={p:sha(REPO/p) for p in paths if p.endswith('.py')}
    runtime=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__)
    return digest((manifest,runtime)),manifest,runtime


def saved_loading(inp,a):
    return SimpleNamespace(old0=np.array(inp['old0']),old1=np.array(inp['old1']),thermal0=np.array(inp['thermal0']),
        elasticity_pa=np.array(inp['elasticity_pa']),ratio=np.array(inp['maxwell_ratio']),
        radial_moments=RadialMoments(tuple(a.volume_m3),tuple(a.inverse_scale_mean),tuple(a.inverse_scale_variance)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    out=args.output;out.mkdir(exist_ok=False);start=time.perf_counter();report=dict(status='RUNNING',new_planetary_steps=0,nonlinear_correctors_executed=0)
    def write(name,value):
        with (out/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    try:
      with time_limit(600):
        resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
        assert np.finfo(np.longdouble).nmant==63
        io_start=time.perf_counter();root=b.pinned(b.PARENT,b.PARENT_SHA)
        packets=[]
        for name,s in PINS.items():
            assert sha(PAIRS/name)==s;packets.append(json.loads((PAIRS/name).read_bytes())['packet'])
        io_read=time.perf_counter()-io_start
        setup_start=time.perf_counter();model,parent=load_volume_slip_checkpoint(b.PARENT,expected_source_checkpoint_sha256=b.FORMATION_SHA)
        setup_seconds=time.perf_counter()-setup_start
        code_sha,manifest,runtime=code_identity();bindings=bindings_for_geometry(parent.geometry,code_sha)
        write('provenance.json',dict(code_sha256=code_sha,code_manifest=manifest,runtime=runtime,bindings=asdict(bindings),inputs=PINS,canonical_sha256=b.PARENT_SHA))
        before_roots=[root,root,b.continued_root(root,packets[1])]
        refs={};audit_refs={};results={}
        def forbidden(*args,**kw):raise AssertionError('PLANETARY_STEP_FORBIDDEN_IN_REPLAY')
        # Guard the public pipeline entries. Only saved linear algebra is replayed.
        import analysis.genesis_coupled_split_control.mechanical as mechanical
        import analysis.genesis_coupled_split_control.thermal as thermal
        with ExitStack() as guards:
            for module,name in [(b,'apply_ports'),(mechanical,'advance'),(thermal,'conduct')]:guards.enter_context(patch.object(module,name,forbidden))
            for mode in ('baseline','cached'):
                model._cache.clear() # Ephemeral implementation cache, no files/state removed.
                cache=FastPath(bindings);rows=[];audits=[];started=time.perf_counter();stats=defaultdict(float)
                print(mode+' replay start',flush=True)
                with (cache if mode=='cached' else nullcontext()):
                    for number,(p,previous) in enumerate(zip(packets,before_roots)):
                        t=time.perf_counter();model._assembly(parent.geometry,np.array(previous['runtime_state']['fault_history']['damage']));stats['initial_assembly_s']+=time.perf_counter()-t
                        dt=(p['thermal']['time_myr']-previous['runtime_state']['thermal']['time_myr'])*365.25*86400*1e6
                        # Four alternating predictor/final visits reproduce the
                        # assembly access pattern, using only SAVED converged inputs.
                        # They are NOT a new nonlinear correction or trajectory.
                        for visit in range(4):
                            for state_key,input_key in [('predictor','predictor_inputs'),('state','inputs')]:
                                inp=p[input_key];s=p[state_key];label=f'{number}_{state_key}'
                                t=time.perf_counter();a,basal=model._assembly(parent.geometry,np.array(s['fault_history']['damage']));assembly_seconds=time.perf_counter()-t
                                t=time.perf_counter();loading=saved_loading(inp,a);M=basal.mass_matrix.toarray();D=basal.basal_drag_matrix.toarray();v0=np.array(inp['before_velocity_m_s']);prep_seconds=time.perf_counter()-t
                                times={};original_solve=solver.solve
                                def linear(*args,**kw):
                                    times['start']=time.perf_counter()
                                    result=original_solve(*args,**kw)
                                    times['end']=time.perf_counter();return result
                                t=time.perf_counter()
                                with patch.object(solver,'solve',linear):r=solver.solve_trial(M,D,basal.external_force_n,v0,dt,a.dofs,a.B0,a.B1,loading)
                                end=time.perf_counter()
                                row=dict(packet=number,visit=visit,role=state_key,geometry_basal_assembly_s=assembly_seconds,saved_loading_s=prep_seconds,
                                         BE_matrix_rhs_s=times['start']-t,linear_solve_s=times['end']-times['start'],memory_ledger_s=end-times['end'])
                                for key,val in row.items():
                                    if key.endswith('_s'):stats[key]+=val
                                result_sha=digest(r);row['result_sha256']=result_sha
                                if mode=='baseline':
                                    if label in refs:assert refs[label]==result_sha
                                    refs[label]=result_sha
                                else:assert refs[label]==result_sha, 'CACHE_CHANGED_NUMERICAL_RESULT '+label
                                # Saved recorded values are also reconstructed.
                                du,v,e0,e1,L=r
                                target=[np.array(s['velocity_m_s']),np.array([c['strain0'] for c in s['mechanics']['cohorts']]),np.array([c['strain1'] for c in s['mechanics']['cohorts']])]
                                row['saved_max_abs_errors']=[float(np.max(abs(x-y))) for x,y in zip((v,e0,e1),target)]
                                for x,y in zip((v,e0,e1),target):np.testing.assert_allclose(x,y,rtol=3e-10,atol=1e-16)
                                row['saved_bitwise']=all(np.array_equal(x,y) for x,y in zip((v,e0,e1),target))
                                rows.append(row)
                        t=time.perf_counter();proof=b.audit_forced(previous,p);elapsed=time.perf_counter()-t;stats['independent_audits_s']+=elapsed
                        proof_sha=digest(proof)
                        if mode=='baseline':audit_refs[number]=proof_sha
                        else:assert audit_refs[number]==proof_sha,'CACHE_CHANGED_AUDIT'
                        audits.append(dict(packet=number,seconds=elapsed,sha256=proof_sha,status=proof['status']))
                        print(mode+' saved packet '+str(number)+' complete',flush=True)
                    if mode=='cached':
                        corrupt=copy.deepcopy(packets[0]['state']);corrupt['mechanics']['cohorts'][0]['strain0'][0][0]+=1e-5
                        try:independent_trial(root['runtime_state'],corrupt,packets[0]['inputs'],root['source_packet'])
                        except ValueError as exc:report['hot_cache_tamper_rejected']=str(exc)
                        else:raise AssertionError('Auditor accepted corrupted solution')
                results[mode]=dict(wall_seconds=time.perf_counter()-started,stages=dict(stats),rows=rows,audits=audits,
                                   cache=cache.report() if mode=='cached' else None)
                write(mode+'.json',results[mode])
        report.update(status='PASS_SAVED_REPLAY_EQUIVALENCE',runtime=runtime,setup_seconds=setup_seconds,input_io_seconds=io_read,
                      linear_replays_per_path=24,saved_packet_audits_per_path=3,correction_access_pattern='four visits to saved predictor/final inputs; no evolving intermediate iterates',
                      speedup=results['baseline']['wall_seconds']/results['cached']['wall_seconds'],
                      baseline_wall_seconds=results['baseline']['wall_seconds'],cached_wall_seconds=results['cached']['wall_seconds'],
                      baseline_stages=results['baseline']['stages'],cached_stages=results['cached']['stages'],
                      cache_vs_baseline_bitwise=True,independent_audit_vs_baseline_bitwise=True,
                      all_saved_outputs_bitwise=all(r['saved_bitwise'] for r in results['cached']['rows']),
                      inputs_unchanged=all(sha(PAIRS/p)==s for p,s in PINS.items()) and sha(b.PARENT)==b.PARENT_SHA)
    except Exception as exc:
        report.update(status='STOPPED',error=f'{type(exc).__name__}: {exc}')
        import traceback
        with (out/'failure.txt').open('x') as f:f.write(traceback.format_exc())
    finally:
        report.update(total_seconds=time.perf_counter()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        t=time.perf_counter();write('report.json',report);print('report IO seconds',time.perf_counter()-t,flush=True)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
