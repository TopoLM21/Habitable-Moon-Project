"""Saved-data integration validation: absolutely no runtime.advance calls."""
import argparse,json,time,hashlib,resource,gc
from pathlib import Path
from analysis.genesis_connected_step_validation.run import PARENT,PARENT_SHA,deadline
from tectonics.genesis_volume_runtime import load_runtime,metadata


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    start=time.monotonic();report=dict(status='RUNNING',new_Genesis_steps=0,stages={},modes={})
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    def stage(name,fn):
        remaining=600-(time.monotonic()-start)
        if remaining<=0:raise TimeoutError('600-second validation budget')
        t=time.monotonic();print(json.dumps(dict(stage=name,event='start',elapsed=t-start)),flush=True)
        with deadline(min(180,remaining)):result=fn()
        report['stages'][name]=time.monotonic()-t
        print(json.dumps(dict(stage=name,event='end',seconds=report['stages'][name])),flush=True)
        return result
    raw=PARENT.read_bytes();assert hashlib.sha256(raw).hexdigest()==PARENT_SHA
    legacy=stage('legacy_default_load',lambda:load_runtime(PARENT))
    legacy.save(a.output/'midpoint.json')
    assert (a.output/'midpoint.json').read_bytes()==raw
    report['default_byte_identical']=True
    for mode,n in [('be50',2),('be25',4)]:
        label=mode[2:];paths=[Path('/workspace/genesis-be-time-control-01')/label/f'step_{i}.json' for i in range(1,n+1)]
        packets=[json.loads(p.read_bytes()) for p in paths]
        runtime=legacy.model.configured_runtime(legacy.state,integrator=mode,root_bytes=raw)
        stage(mode+'_attach_saved_chain',lambda:runtime.attach_chain(packets))
        path=a.output/(mode+'.json');runtime.save(path);description=runtime.describe()
        del runtime,packets;gc.collect()
        loaded=stage(mode+'_load_bundle',lambda:load_runtime(path,integrator=mode))
        loaded.save(a.output/(mode+'_restart.json'))
        assert path.read_bytes()==(a.output/(mode+'_restart.json')).read_bytes()
        assert loaded.describe()==description
        for wrong in ['midpoint','be25' if mode=='be50' else 'be50']:
            try:load_runtime(path,integrator=wrong)
            except ValueError:pass
            else:raise AssertionError('Silent scheme mixing')
        L=loaded.state.be_accumulated_ledger
        elastic=L['elastic_before_j']-L['elastic_after_j'];kinetic=L['kinetic_before_j']-L['kinetic_after_j']
        sources=elastic+kinetic+L['external_work_j']+L['thermal_work_j']+L['birth_kinetic_inflow_j']+L['damage_energy_change_j']
        physical=L['physical_drag_j']+L['physical_maxwell_j'];numerical=L['numerical_kinetic_j']+L['numerical_memory_j']
        report['modes'][mode]=dict(description=description,restart_byte_identical=True,wrong_integrators_rejected=True,
            thermal_payload_unchanged=loaded.chain[-1]['state']['thermal']==json.loads(paths[-1].read_bytes())['state']['thermal'],
            reservoir=dict(elastic_release_j=elastic,kinetic_release_j=kinetic,external_work_j=L['external_work_j'],thermal_mechanical_work_j=L['thermal_work_j'],
                birth_kinetic_inflow_j=L['birth_kinetic_inflow_j'],damage_energy_change_j=L['damage_energy_change_j'],
                source_total_j=sources,physical_dissipation_j=physical,numerical_dissipation_j=numerical,
                source_minus_sinks_j=sources-physical-numerical,numerical_over_external=numerical/abs(L['external_work_j']),
                numerical_over_elastic_release=numerical/elastic,global_thermal_reservoir_closed=False))
        del loaded;gc.collect()
    assert PARENT.read_bytes()==raw
    report.update(status='PASS',seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,canonical_bytes_unchanged=True,
        artifacts={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in a.output.iterdir() if p.is_file()})
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','seconds','peak_rss_kib','new_Genesis_steps']}),flush=True)


if __name__=='__main__':main()
