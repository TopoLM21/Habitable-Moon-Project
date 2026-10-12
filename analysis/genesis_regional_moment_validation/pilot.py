"""Eight fixed saved leaves, no subdivisions and no radial calls."""
from pathlib import Path
from fractions import Fraction as F
import argparse,hashlib,json,time
from analysis.genesis_regional_work_validation.identity import expand
from .moments import enclose
from .audit import audit

IDS=(1339,1342,1225,1230,1007,1010,1176,1177)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('/workspace/genesis-regional-moment-pilot-01.json'));args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh output required')
    path=Path('/workspace/genesis-regional-work-centered-01.json');saved=json.loads(path.read_bytes())
    start=time.monotonic();polys,_=expand(saved['graph']);build=time.monotonic()-start
    rows=[];compute=build;audittime=0
    for i in IDS:
        if compute>=120:break
        source=saved['records'][i];assert i in saved['leaves']
        box=tuple(tuple(map(F,b)) for b in source['box']);start=time.monotonic();r=enclose(polys,box);elapsed=time.monotonic()-start;compute+=elapsed
        start=time.monotonic();a=audit(polys,r);audittime+=time.monotonic()-start
        old=tuple(map(F,source['bounds']));new=tuple(map(F,r['bounds']));intersection=(max(old[0],new[0]),min(old[1],new[1]));assert intersection[0]<=intersection[1]
        r.update(id=i,old_bounds=list(map(str,old)),intersection=list(map(str,intersection)),seconds=elapsed,audit=a)
        rows.append(r)
        print(json.dumps(dict(id=i,state=r['state'],old_width=float(old[1]-old[0]),new_bounds=[float(v) for v in new],seconds=elapsed)),flush=True)
    result=dict(source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),fixed_ids=IDS,records=rows,
        compute_seconds=compute,audit_seconds=audittime,polynomial_transforms=3*len(rows),max_blocks=8,max_compute_seconds=120,
        radial_evaluations=0,mechanical_solves=0,subdivisions=0,physical_admission=False)
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__=='__main__':main()
