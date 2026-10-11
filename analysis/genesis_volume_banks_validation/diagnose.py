"""Recover the exact old final mesh from its node trace, without tensor reruns."""
import argparse,json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from analysis.genesis_volume_integral_validation.source_control import load_authenticated_inputs,support_source_inputs,require,pin,seal
from analysis.genesis_volume_precision_validation.control import write
from analysis.genesis_volume_mapped_validation.angular import CellGeometry
from tectonics.genesis_volume_bank_bounds import PreparedBankBounds,halfspace
from tectonics.genesis_volume_source_coverage import exact_node
from tectonics.genesis_volume_event_partition import literal_cell,split_cell,ratio
from tectonics.genesis_volume_rational_fields import fraction


def node(record):return tuple(F(*map(int,v)) for v in record['N'])


def recover_cell(records):
    rule=(np.polynomial.legendre.leggauss(3)[0]+1)/2
    s0,s1=map(fraction,rule[:2]);t0,t1=s0,s1
    p0,p1,p3=map(node,(records[0],records[1],records[3]))
    dt=tuple((b-a)/(s0*(t1-t0)) for a,b in zip(p0,p1))
    ds=tuple((b-a)/(s1-s0) for a,b in zip(p0,p3))
    a=tuple(x-s0*y for x,y in zip(p0,ds))
    b=tuple(x+y-t0*z for x,y,z in zip(a,ds,dt));c=tuple(x+y for x,y in zip(b,dt))
    cell=literal_cell((a,b,c));index=0
    for order in (3,5):
        nodes=tuple(map(fraction,(np.polynomial.legendre.leggauss(order)[0]+1)/2))
        for s in nodes:
            for t in nodes:
                require(exact_node(cell,s,t)==node(records[index]),'Exact mesh replay differs from recorded node')
                index+=1
    return cell


def positive_bank(geometry,N,side):
    N=np.array(N,object);u=N@geometry.C;d=N@geometry.D;v=d-geometry.center
    slope,intercept=side*(u@geometry.normal),side*(d@geometry.normal-geometry.offset)
    require(bool(slope or intercept),'Audited node lies wholly on Gamma')
    interval=halfspace(geometry.cohort,slope,intercept)
    if interval is None:return False
    A,B,E=u@u,u@v,v@v-geometry.radius**2
    require(A>0,'Collapsed line in diagnostic')
    lam=min(max(-B/A,interval[0]),interval[1])
    return A*lam*lam+2*B*lam+E<0


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    require(not args.output.exists(),'Fresh diagnostic directory required');args.output.mkdir()
    prior=Path('/workspace/genesis-mapped-full-03');summary=json.loads(Path('analysis/genesis_volume_mapped_validation/summary.json').read_bytes())
    rows=[]
    for i,item in enumerate(load_authenticated_inputs()['supports']):
        raw=support_source_inputs(item['case'],item['label']);old=CellGeometry(raw);new=PreparedBankBounds(raw)
        path=prior/f'case_{i:03d}'/'nodes.jsonl';require(pin(path)==summary['cases'][i]['node_file_pin'],'Prior node trace changed')
        trace=[json.loads(t) for t in path.read_text().splitlines()];require(len(trace)%34==0,'Partial cell in saved trace')
        evaluated={recover_cell(trace[j:j+34]):trace[j:j+34] for j in range(0,len(trace),34)}
        require(len(evaluated)*34==len(trace),'Duplicated evaluated triangle')
        active={c:r for c,r in evaluated.items() if not any(child in evaluated for child in split_cell(c))}
        angular=json.loads((prior/f'case_{i:03d}'/'angular.json').read_bytes())
        require(len(active)==angular['retained_angular_cells'],'Final mesh cannot be uniquely reconstructed by recorded children')
        missing=[]
        for cell,records in active.items():
            sides=[s for s in (-1,1) if old.bounds(cell)[s][1]>0 and not any(positive_bank(new,node(r),s) for r in records[9:])]
            if not sides:continue
            cert=new.certificate(cell);witnesses={}
            for side in sides:
                hi=F(*map(int,cert['bank_measure_bounds'][str(side)][1]))
                witness=None
                if hi:
                    for a in range(17):
                        for b in range(17-a):
                            n=tuple((a*cell[0][j]+b*cell[1][j]+(16-a-b)*cell[2][j])/16 for j in range(3))
                            if positive_bank(new,n,side):witness=n;break
                        if witness is not None:break
                witnesses[str(side)]=dict(status='PROVED_EMPTY' if hi==0 else 'POSITIVE_WITNESS' if witness else 'OPEN',
                    witness=None if witness is None else list(map(ratio,witness)))
            missing.append(dict(vertices=[[ratio(v) for v in row] for row in cell],missing_sides=sides,
                old_upper_bounds=[str(fraction(np.asarray(old.bounds(cell)[s][1]).item())) for s in sides],new_certificate=cert,classification=witnesses))
        require(len(missing)==angular['unresolved_positive_candidate_cells'],'Saved missing-cell count differs from exact bank positivity replay')
        classified=[v['status'] for row in missing for v in row['classification'].values()]
        report=seal(dict(case=i,node_trace_pin=pin(path),evaluated_cells=len(evaluated),active_cells=len(active),
            missing_cells=missing,missing_cell_count=len(missing),missing_bank_counts={s:classified.count(s) for s in ('PROVED_EMPTY','POSITIVE_WITNESS','OPEN')},
            source_budget_failures=angular['source_uncertainty_failed_component_count'],
            volume_failed_components=sum(e>a for e,a in zip(angular['observed_embedded_errors'][:2],angular['component_allowances'][:2])),
            tensor_failed_components=sum(e>a for e,a in zip(angular['observed_embedded_errors'][2:],angular['component_allowances'][2:])),
            all_recorded_node_coordinates_exactly_replayed=True))
        write(args.output/f'case_{i:03d}.json',report);rows.append(report)
        print(json.dumps({k:v for k,v in report.items() if k not in ('missing_cells','sha256')}),flush=True)
    write(args.output/'report.json',seal(dict(cases=rows,source_pins={str(p):pin(p) for p in
        (Path(__file__),Path('tectonics/genesis_volume_bank_bounds.py'),Path('analysis/genesis_volume_mapped_validation/angular.py'))})))


if __name__=='__main__':main()
