"""One short adaptive comparison against saved fixed trajectories."""
from pathlib import Path
from copy import deepcopy
import hashlib,json,time
import numpy as np
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.experimental_edge_adaptive import integrate,attempt,DEFAULT
from tectonics.experimental_edge_weakening import MODE
from tectonics.genesis_transfer_bridge import encoded
from .nodal_retirement_check import SOURCE,OPTION
ROOT=Path(__file__).parent
OUTPUT=Path('/workspace/genesis-edge-adaptive-01')


def main(output=OUTPUT,result_path=ROOT/'edge_adaptive_result.json'):
    assert not output.exists();output.mkdir();source=SimplifiedGenesis.load(SOURCE);c=deepcopy(source.config);c['experimental_geological_edge_fatigue']={**OPTION,'mode':MODE};m=SimplifiedGenesis(c,source.state)
    (output/'declaration.json').write_text(json.dumps(dict(horizon_years=25,settings=DEFAULT),indent=2)+'\n')
    before=encoded(m.state);clock=time.monotonic();trial,row=attempt(m,.25);benchmark=time.monotonic()-clock
    assert encoded(m.state)==before;print(json.dumps(dict(first_attempt_seconds=benchmark,result=row)),flush=True)
    final,report=integrate(m,25.);assert encoded(m.state)==before
    final.save(output/'checkpoint.json');comparisons=[]
    for dt in [.25,.125]:
        path=Path(f'/workspace/genesis-edge-short-accuracy-01/dt-{dt:g}-fields.npz');z=np.load(path,allow_pickle=False);records=[]
        for r in report['accepted']:
            j=round(r['years']/dt)-1
            if 0<=j<len(z['years']) and abs(z['years'][j]-r['years'])<1e-6:
                t=np.array(r['traction_pa']);d=np.array(r['damage']);fine=z['shear_vectors_pa'][j]
                records.append(dict(years=r['years'],traction=float(np.linalg.norm(t-fine,axis=1).max()/np.linalg.norm(fine,axis=1).max()),damage=float(np.max(abs(d-z['damage'][j]))),saturation_equal=bool(np.array_equal(d>=1,z['damage'][j]>=1))))
        comparisons.append(dict(reference_dt=dt,samples=len(records),max_traction=max(r['traction'] for r in records),max_damage=max(r['damage'] for r in records),rows=records))
    # Last accepted fine state is saved, trial rejection never mutates its caller.
    final_loaded=SimplifiedGenesis.load(output/'checkpoint.json');assert encoded(final_loaded.state)==encoded(final.state)
    report.update(first_attempt_seconds=benchmark,comparisons=comparisons,source_unchanged=encoded(m.state)==before,
        checkpoint_sha256=hashlib.sha256((output/'checkpoint.json').read_bytes()).hexdigest(),load_exact=True,
        rejection_count=sum(not r['accepted'] for r in report['attempts']),approximate_native_solves=3*len(report['attempts']))
    result_path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['attempts','accepted','comparisons']}),flush=True)
    print(json.dumps([{k:v for k,v in q.items() if k!='rows'} for q in comparisons]),flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--result',type=Path,default=ROOT/'edge_adaptive_result.json');args=p.parse_args();main(args.output,args.result)
