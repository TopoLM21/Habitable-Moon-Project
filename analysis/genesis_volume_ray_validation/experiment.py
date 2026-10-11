"""Bounded scalar experiment: exact event partitions along Duffy rays.

Not a production tensor integrator. Root boxes are numerically integrated,
not discarded. Exact fiberwise coverage does not certify angular remainders
or topology between sampled outer rays.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import time
import numpy as np

from tectonics.genesis_volume_ray_roots import partition,restrict_ray,evaluate
from tectonics.genesis_volume_event_partition import source_events,signed_double_area,ratio
from tectonics.genesis_volume_rational_fields import fraction
from tectonics.genesis_volume_operand_bounds import DirectedInterval,point,add,subtract,multiply,rational_enclosure
from analysis.genesis_volume_integral_validation.angular_control import _strict_allowance,_sum_enclosures,_abs_interval
from analysis.genesis_volume_banks_validation.independent_bank import fiber
from analysis.genesis_volume_banks_validation.partition_audit import check

LD=np.longdouble
EVENTS=('A','Delta','sphere_cohort_lower','sphere_cohort_upper','g',
        'Gamma_cohort_lower','Gamma_cohort_upper','Gamma_sphere_resultant')


def fixture(radius=.25,offset=9.):
    return dict(section=dict(triangle_m=[[0.,0.,10.],[4.,0.,10.],[0.,4.,10.]],
        slope_P_m=[[0.,0.,0.]]*3,displacement_A_m=[[0.,0.,0.]]*3,
        lambda_bounds=[.8,1.],normal_xyz=[0.,0.,1.],offset_m=offset),
        seed_center_world_m=[1.,1.,9.],seed_radius_m=radius)


def positive(polys,rho,side,cohort):
    A,B,E,g,h=(evaluate(polys[k],rho) for k in ('A','B','E','g','h'))
    if A<=0:raise ValueError('Collapsed source line is not covered by this experiment')
    g*=side;h*=side;lo,hi=cohort
    if not g and not h:raise ValueError('Entire source line on Gamma requires separate semantics')
    if g>0:lo=max(lo,-h/g)
    elif g<0:hi=min(hi,-h/g)
    elif h<0:return False
    if lo>=hi:return False
    lam=min(max(-B/A,lo),hi)
    return A*lam*lam+2*B*lam+E<0


class RayPlanner:
    def __init__(self,raw,triangles):
        self.raw=raw;self.triangles=triangles;self.events=source_events(raw)[0]
        self.cohort=tuple(map(fraction,raw['section']['lambda_bounds']))
        self.cache={};self.seconds=0.;self.root_boxes=0;self.zero_intervals=0

    def ray(self,index,t):
        key=(index,t)
        if key in self.cache:return self.cache[key]
        start=time.monotonic();a,b,c=self.triangles[index]
        end=tuple((1-t)*x+t*y for x,y in zip(b,c))
        polys={k:restrict_ray(p,a,end) for k,p in self.events.items() if k in set(EVENTS)|{'B','E','h'}}
        cert=partition({k:polys[k] for k in EVENTS})
        bands=[]
        for lo,hi,kind in cert['intervals']:
            signs=None if kind=='root_box' else tuple(positive(polys,(lo+hi)/2,s,self.cohort) for s in (-1,1))
            if signs is not None and not any(signs):self.zero_intervals+=1;continue
            bands.append((lo,hi,kind));self.root_boxes+=int(kind=='root_box')
        result=(a,end,tuple(bands),cert)
        self.cache[key]=result;self.seconds+=time.monotonic()-start
        return result


@dataclass
class Cell:
    triangle:int
    s0:F=F()
    s1:F=F(1)
    t0:F=F()
    t1:F=F(1)
    depth:int=0
    coarse:object=None
    fine:object=None
    error:object=None
    source_error:object=None
    observed:object=None
    coarse_enclosure:object=None
    fine_enclosure:object=None

    def children(self):
        sm=(self.s0+self.s1)/2;tm=(self.t0+self.t1)/2
        return [Cell(self.triangle,a,b,c,d,self.depth+1)
                for a,b in ((self.s0,sm),(sm,self.s1))
                for c,d in ((self.t0,tm),(tm,self.t1))]


def gauss(order):
    nodes,weights=np.polynomial.legendre.leggauss(order)
    return tuple(zip(map(fraction,(nodes+1)/2),map(fraction,weights/2)))


class Experiment:
    def __init__(self,raw):
        self.raw=raw
        root=tuple(tuple(F(int(i==j)) for j in range(3)) for i in range(3))
        center=(F(4,9),F(5,18),F(5,18))
        self.triangles=tuple((center,root[i],root[(i+1)%3]) for i in range(3))
        self.mesh_audit=check([tuple(n[1:] for n in root)],
            [tuple(n[1:] for n in c) for c in self.triangles])
        self.planner=RayPlanner(raw,self.triangles)
        self.rules={n:gauss(n) for n in (3,5)}
        self.nodes=0;self.evaluated_cells=0;self.kernel_seconds=0.;self.root_box_nodes=0

    def cost(self,cell):
        return sum(order*sum(len(self.planner.ray(cell.triangle,cell.t0+(cell.t1-cell.t0)*t)[2])
                             for t,_ in self.rules[order]) for order in (3,5))

    def rule(self,cell,order):
        value=np.zeros(2,LD);enclosure=point(value)
        area=signed_double_area(self.triangles[cell.triangle])
        for t,tw in self.rules[order]:
            t=cell.t0+(cell.t1-cell.t0)*t
            start,end,bands,_=self.planner.ray(cell.triangle,t)
            for lo,hi,kind in bands:
                for s,sw in self.rules[order]:
                    s=cell.s0+(cell.s1-cell.s0)*s
                    # Endpoint map keeps an exact rational node and Jacobian.
                    rho=lo+(hi-lo)*(2*s-s*s)
                    jac=area*rho*(hi-lo)*2*(1-s)*(cell.s1-cell.s0)*(cell.t1-cell.t0)
                    N=tuple(a+rho*(b-a) for a,b in zip(start,end))
                    clock=time.monotonic();bounds=[fiber(self.raw,N,side) for side in (-1,1)]
                    if any(b is None for b in bounds):raise ValueError('Unresolved Gamma fiber')
                    self.kernel_seconds+=time.monotonic()-clock
                    lower=np.array([rational_enclosure(b[0]).lower for b in bounds],LD)
                    upper=np.array([rational_enclosure(b[1]).upper for b in bounds],LD)
                    w=rational_enclosure(jac*sw*tw)
                    value+=(w.lower/2+w.upper/2)*(lower/2+upper/2)
                    enclosure=add(enclosure,multiply(w,DirectedInterval(lower,upper)))
                    self.nodes+=1;self.root_box_nodes+=int(kind=='root_box')
        return value,enclosure

    def compute(self,cell):
        coarse,cb=self.rule(cell,3);fine,fb=self.rule(cell,5)
        cell.coarse=coarse;cell.fine=fine
        cell.coarse_enclosure=cb;cell.fine_enclosure=fb
        cell.source_error=add(point(cb.absolute_error_for(coarse)),point(fb.absolute_error_for(fine))).upper
        cell.observed=multiply(_abs_interval(subtract(point(fine),point(coarse))),8).upper
        cell.error=add(point(cell.observed),point(cell.source_error)).upper
        self.evaluated_cells+=1
        return cell

    def run(self):
        cells=[Cell(i) for i in range(3)]
        if sum(map(self.cost,cells))>3400:raise ValueError('Initial ray coverage exceeds original work cap')
        cells=list(map(self.compute,cells));trace=[]
        while True:
            coarse=np.sum([c.coarse for c in cells],axis=0,dtype=LD)
            fine=np.sum([c.fine for c in cells],axis=0,dtype=LD)
            cb=_sum_enclosures((c.coarse_enclosure for c in cells),(2,)).absolute_error_for(coarse)
            fb=_sum_enclosures((c.fine_enclosure for c in cells),(2,)).absolute_error_for(fine)
            observed=_sum_enclosures((point(c.observed) for c in cells),(2,)).upper
            error=add(add(point(observed),point(cb)),point(fb)).upper
            allowance=_strict_allowance(2e-6,coarse,fine)
            trace.append(dict(nodes=self.nodes,values=list(map(float,fine)),
                embedded_errors=list(map(float,error)),allowances=list(map(float,allowance))))
            if np.all(error<=allowance):reason='Embedded observations fit; analytic and general certification separate';break
            def priority(c):
                ratio=np.divide(c.error,allowance,out=np.zeros(2,LD),where=allowance!=0)
                ratio[(allowance==0)&(c.error>0)]=np.inf
                return max(ratio)
            chosen=max(cells,key=priority);children=chosen.children()
            cost=sum(map(self.cost,children))
            if self.nodes+cost>3400 or self.evaluated_cells+len(children)>2000 or chosen.depth>=24:
                reason='Original work/depth cap reached';break
            cells.remove(chosen);cells.extend(map(self.compute,children))
        return dict(reason=reason,trace=trace,radial_line_evaluations=self.nodes,
            scalar_bank_fiber_calls=2*self.nodes,root_box_node_evaluations=self.root_box_nodes,
            ray_partitions_built=len(self.planner.cache),root_boxes_retained=self.planner.root_boxes,
            proved_zero_root_free_intervals_skipped=self.planner.zero_intervals,
            event_isolation_seconds=self.planner.seconds,kernel_seconds=self.kernel_seconds,
            evaluated_cells=self.evaluated_cells,final_cells=len(cells),
            maximum_depth=max(c.depth for c in cells),initial_mesh_audit=self.mesh_audit,
            all_ray_partitions_cover_unit_interval=True,
            root_boxes_discarded=False,general_outer_angular_coverage_certified=False,
            angular_accuracy_certified=False,source_bounds_added_to_allowance=False,
            unchanged_production_allowance_and_directed_aggregation=True,
            relative_tolerance=2e-6,angular_orders=[3,5],max_line_evaluations=3400,
            max_cells=2000,max_depth=24,production_tensor_integrator_changed=False)


def pi_bounds():
    terms=[]
    for x in (F(1,2),F(1,3)):
        n=128;s=sum(((-1)**k*x**(2*k+1)/F(2*k+1) for k in range(n)),F())
        t=s+(-1)**n*x**(2*n+1)/F(2*n+1);terms.append((min(s,t),max(s,t)))
    return tuple(4*sum(t[i] for t in terms) for i in (0,1))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case',choices=('half_ball','small_half_ball','tangent_Gamma'),default='half_ball')
    args=parser.parse_args();out=args.output.absolute()
    if out.exists() or not out.is_relative_to('/workspace'):raise ValueError('Fresh /workspace output required')
    free=shutil.disk_usage('/workspace').free
    if free<16*1024**2:raise ValueError('Insufficient 16 MiB experiment reserve')
    if np.finfo(LD).nmant!=63:raise ValueError('Requires longdouble nmant=63')
    raw=fixture(2**-12 if args.case=='small_half_ball' else .25,9.25 if args.case=='tangent_Gamma' else 9.)
    R=F(raw['seed_radius_m']);assert F(9)-10*F(.8)>R and 1>R
    assert (F(18,5)-2)**2>R**2*(2+F(4,25))
    start=time.monotonic();experiment=Experiment(raw);result=experiment.run()
    truth=[tuple(F(2,3)*R**3*v for v in pi_bounds())]*2
    if args.case=='tangent_Gamma':truth=[tuple(2*v for v in truth[0]),(F(),F())]
    errors=[];passed=[]
    for v,(lo,hi) in zip(result['trace'][-1]['values'],truth):
        error=max(abs(F(v)-lo),abs(F(v)-hi))
        errors.append(float(error/lo) if lo else float(error))
        passed.append(error<=F(2,10**6)*lo)
    result.update(case=args.case,source_geometry=raw,analytic_truth_bounds=[[ratio(v) for v in b] for b in truth],
        relative_errors_or_absolute_for_zero_truth=errors,analytic_volume_gate_passed=all(passed),
        seconds=time.monotonic()-start,free_bytes_before=free,
        history_advanced=False,physical_admission=False)
    out.mkdir();(out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    rays=[]
    for (index,t),(a,b,bands,cert) in experiment.planner.cache.items():
        polynomials={k:restrict_ray(experiment.planner.events[k],a,b) for k in EVENTS}
        rays.append(dict(triangle=index,t=ratio(t),start=list(map(ratio,a)),end=list(map(ratio,b)),
            polynomials={k:list(map(ratio,p)) for k,p in polynomials.items()},
            isolated={k:dict(identically_zero=v['identically_zero'],
                roots=[[ratio(lo),ratio(hi)] for lo,hi in v['roots']]) for k,v in cert['events'].items()},
            intervals=[dict(lo=ratio(lo),hi=ratio(hi),kind=kind,integrated=(lo,hi,kind) in bands)
                       for lo,hi,kind in cert['intervals']]))
    (out/'ray_certificates.json').write_text(json.dumps(rays)+'\n')
    paths=[Path(__file__),Path('tectonics/genesis_volume_ray_roots.py')]
    for p in paths:(out/p.name).write_bytes(p.read_bytes())
    pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (out/'source_pins.json').write_text(json.dumps(pins,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('case','radial_line_evaluations','relative_errors_or_absolute_for_zero_truth','analytic_volume_gate_passed','seconds','ray_partitions_built','event_isolation_seconds')}))


if __name__=='__main__':main()
