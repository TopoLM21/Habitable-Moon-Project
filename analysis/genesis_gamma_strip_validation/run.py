"""One bounded symmetric check of exactly sixteen saved Gamma leaves."""
from fractions import Fraction as F
from pathlib import Path
import argparse,hashlib,json,time
from analysis.genesis_regional_work_validation.identity import expand
from analysis.genesis_regional_moment_validation.moments import enclose
from analysis.genesis_regional_moment_validation.regional import digest
from .strips import partition

IDS=(1258,1255,1270,1267,1257,1256,1269,1268,1301,1300,1305,1304,1230,1225,1229,1226)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh output required')
    path=Path('/workspace/genesis-regional-moment-region-01.json');old=json.loads(path.read_bytes())
    receiptpath=Path('/workspace/genesis-regional-moment-region-audit-01.json');receipt=json.loads(receiptpath.read_bytes())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['input_sha256']
    sourcepath=Path(old['source']);assert hashlib.sha256(sourcepath.read_bytes()).hexdigest()==old['source_sha256']
    source=json.loads(sourcepath.read_bytes());start=time.monotonic();polys,_=expand(source['graph'])
    rows=[];newblocks=0;transforms=0;reason='fixed_set_complete';total=list(map(F,old['bounds']))
    for identifier in IDS:
        if time.monotonic()-start>=120:reason='time_cap';break
        rechecks=len(rows)+1
        if newblocks+rechecks>256:reason='work_cap';break
        parent=old['records'][identifier];assert identifier in old['leaves'] and parent['state']=='gamma_uncertain_inside'
        box=tuple(tuple(map(F,b)) for b in parent['box']);parentcert=enclose(polys,box);transforms+=3
        assert digest(parentcert['polynomials'])==parent['coefficient_sha256']
        plan=partition(polys['gamma'],box);children=[];complete=True;subtotal=[F(),F()];quarters_started=0
        for quarter in plan['quarters']:
            if newblocks+rechecks+1>256:complete=False;reason='work_cap';break
            newblocks+=1;quarters_started+=1
            for childbox in quarter['slabs']:
                if time.monotonic()-start>=120:complete=False;reason='time_cap';break
                if newblocks+rechecks+1>256:complete=False;reason='work_cap';break
                cb=tuple(tuple(map(F,b)) for b in childbox);r=enclose(polys,cb);transforms+=3;newblocks+=1
                assert newblocks+rechecks<=256 and old['processed_blocks']+newblocks+rechecks<=1999
                children.append(r)
                for e in (0,1):subtotal[e]+=F(r['bounds'][e])
            if not complete:break
        if not plan['quarters']:complete=False
        prior=list(map(F,parent['bounds']))
        improved=[max(prior[0],subtotal[0]),min(prior[1],subtotal[1])] if complete else prior
        assert improved[0]<=improved[1]
        for e in (0,1):total[e]+=improved[e]-prior[e]
        row=dict(id=identifier,old_bounds=parent['bounds'],plan=plan,children=children,complete=complete,quarters_started=quarters_started,
                 summed_child_bounds=list(map(str,subtotal)) if complete else None,accepted_bounds=list(map(str,improved)),
                 sign='positive' if improved[0]>0 else 'negative' if improved[1]<0 else 'unresolved')
        rows.append(row)
        print(json.dumps(dict(id=identifier,axis=plan['axis'],complete=complete,slabs=len(children),bounds=[float(v) for v in improved],
                             sign=row['sign'],seconds=time.monotonic()-start)),flush=True)
        if not complete and reason in ('time_cap','work_cap'):break
    result=dict(source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),prior_audit=str(receiptpath),
        prior_audit_sha256=hashlib.sha256(receiptpath.read_bytes()).hexdigest(),ids=list(IDS),records=rows,
        old_regional_bounds=old['bounds'],regional_bounds=list(map(str,total)),
        regional_sign='positive' if total[0]>0 else 'negative' if total[1]<0 else 'unresolved',reason=reason,
        seconds=time.monotonic()-start,max_compute_seconds=120,new_local_blocks=newblocks,max_new_local_blocks=256,
        old_processed_blocks=old['processed_blocks'],combined_processed_blocks=old['processed_blocks']+newblocks,
        parent_rechecks=len(rows),new_block_work_with_parent_rechecks=newblocks+len(rows),
        combined_block_work_with_parent_rechecks=old['processed_blocks']+newblocks+len(rows),
        polynomial_transforms=transforms,radial_evaluations=0,mechanical_solves=0,tolerance_changed=False,physical_admission=False)
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','old_regional_bounds','regional_bounds')}),flush=True)


if __name__=='__main__':main()
