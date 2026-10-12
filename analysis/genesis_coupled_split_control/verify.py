"""Single final read-only batch action for the saved two-step control."""
import argparse,json,hashlib
from pathlib import Path
from dataclasses import asdict
import numpy as np
from tectonics.genesis import GenesisParameters
from tectonics.genesis_volume_loading import VolumeCohortMemory
from tectonics.genesis_volume_fault_evolution import volume_fault_history_from_dict,volume_fault_history_to_dict
from analysis.genesis_connected_step_validation.run import PARENT,PARENT_SHA,deadline
from analysis.genesis_be_candidate.adapter import dumps
from analysis.genesis_be_candidate.checkpoint import seal
from . import run,mechanical,thermal


def forbidden(*args,**kwargs):raise AssertionError('Read-only verification cannot evolve a state')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True);out=parser.parse_args().directory
    mechanical.solve_trial=forbidden;thermal.solve=forbidden
    with deadline(150):
        root=run.pinned(PARENT,PARENT_SHA);run.SOURCE=root['source_packet'];previous=PARENT_SHA
        baseline=run.pinned('/workspace/genesis-cutcell-mechanical-02/candidate.json','5a0cfcd3edfd57f4f05b59e8b408b09c25ac2a704068609e22029ab865230f2e')
        completion=json.loads((out/'completion.json').read_bytes());summary=[]
        for i in (1,2):
            raw=(out/f'step_{i}.json').read_bytes();p=json.loads(raw);th=json.loads((out/f'thermal_{i}.json').read_bytes())
            assert p['predecessor_sha256']==previous and p['step_index']==i and p['thermal_packet']==th
            assert seal({k:v for k,v in th.items() if k!='sha256'})==th
            assert p['thermal_sha256']==th['sha256']
            assert seal({k:v for k,v in p.items() if k!='sha256'})==p
            audit=run.audit_step(root,th,p,run.SOURCE)
            memory=[asdict(VolumeCohortMemory(**c)) for c in p['state']['mechanics']['cohorts']];p['state']['mechanics']['cohorts']=memory
            faults=volume_fault_history_from_dict(p['state']['fault_history'],run.SOURCE);assert volume_fault_history_to_dict(faults)==p['state']['fault_history']
            assert dumps(p).encode()==raw==(out/f'restart_{i}.json').read_bytes()
            summary.append(dict(step=i,status='PASS',local_heat_balance_j=audit['feedback']['local_heat_max_residual_j'],momentum_relative=audit['mechanical']['coordinate_momentum_relative'],correctors=len(p['correctors'])))
            previous=hashlib.sha256(raw).hexdigest();root=run.root_from_previous(root,p)
        assert previous==completion['final_sha256']
        gp=GenesisParameters(**run.SOURCE['source']['parameters']['thermal']);comparison=run.compare(baseline,p,th['fixed_cut_geometry'],gp)
        for name,value in completion['comparison'].items():
            if isinstance(value,dict):
                for k,v in value.items():assert np.isclose(comparison[name][k],v,rtol=1e-12,atol=1e-20)
            elif isinstance(value,(int,float)):assert np.isclose(comparison[name],value,rtol=1e-12,atol=1e-20)
        print(dumps(dict(status='PASS',new_evolution_steps=0,steps=summary,final_sha256=previous,comparison=comparison)))


if __name__=='__main__':main()
