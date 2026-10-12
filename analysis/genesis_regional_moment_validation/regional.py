"""One fixed-partition regional replay using polynomial moments, capped at 1999 nodes."""
from fractions import Fraction as F
from pathlib import Path
import argparse,hashlib,json,time,shutil
from .moments import enclose,enclose_certificates
from .restrict import halve
from analysis.genesis_regional_work_validation.identity import expand


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('/workspace/genesis-regional-moment-region-01.json'));args=parser.parse_args()
    output=args.output
    if output.exists():raise ValueError('Fresh output required')
    if shutil.disk_usage('/workspace').free<128*1024**2:raise ValueError('128 MiB reserve')
    path=Path('/workspace/genesis-regional-work-centered-01.json');old=json.loads(path.read_bytes())
    assert old['processed_blocks']==1999 and old['maximum_depth']==11
    start=time.monotonic();polys,_=expand(old['graph'])
    root=tuple(tuple(map(F,b)) for b in old['root']);full=enclose(polys,root);root_cert=full['polynomials']
    certificates={0:root_cert};active={0};rows=[];splits=[];trace=[];total=[F(),F()]
    def record(i,full):
        assert i==len(rows)
        coeff=full.pop('polynomials');r=dict(id=i,**full,coefficient_sha256=digest(coeff))
        rows.append(r)
        for e in (0,1):total[e]+=F(r['bounds'][e])
    record(0,full)
    def snapshot():
        trace.append(dict(processed_blocks=len(rows),bounds=list(map(str,total)),seconds=time.monotonic()-start))
        print(json.dumps(dict(blocks=len(rows),bounds=[float(x) for x in total],seconds=trace[-1]['seconds'])),flush=True)
    snapshot();reason='fixed_partition_complete'
    for split in old['splits']:
        if time.monotonic()-start>=120:reason='time_cap';break
        if len(rows)+2>1999:reason='block_cap';break
        parent=split['parent'];axis=split['axis'];assert parent in active
        cert=certificates.pop(parent);pairs={k:halve(c,axis) for k,c in cert.items()}
        active.remove(parent)
        for e in (0,1):total[e]-=F(rows[parent]['bounds'][e])
        for side,i in enumerate(split['children']):
            child={k:p[side] for k,p in pairs.items()};certificates[i]=child;active.add(i)
            box=tuple(tuple(map(F,b)) for b in old['records'][i]['box']);record(i,enclose_certificates(child,box))
        splits.append(split)
        if len(rows) in (31,127,511,999,1499,1999):snapshot()
    if trace[-1]['processed_blocks']!=len(rows):snapshot()
    result=dict(source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),root_certificates=root_cert,
        root=old['root'],records=rows,splits=splits,leaves=sorted(active),bounds=list(map(str,total)),trace=trace,
        sign='positive' if total[0]>0 else 'negative' if total[1]<0 else 'unresolved',reason=reason,
        processed_blocks=len(rows),max_blocks=1999,max_compute_seconds=120,maximum_depth=max(old['records'][i]['depth'] for i in range(len(rows))),
        seconds=time.monotonic()-start,root_polynomial_transforms=3,half_restrictions=3*len(splits),
        radial_evaluations=0,mechanical_solves=0,new_partition_splits=0,physical_admission=False)
    with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print('saved',output,result['sign'],result['seconds'],flush=True)


if __name__=='__main__':main()
