"""Budgeted mixed-rule diagnosis and directional refinement of analytic fixtures.

Acceptance remains the original 8*sum(abs(Q55-Q33)) plus source/root bounds.
Mixed rules only diagnose directions; analytic truth is never used to adapt.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import time
import numpy as np
from analysis.genesis_volume_ray_enclosure_validation.experiment import EnclosureExperiment,f
from analysis.genesis_volume_ray_validation.experiment import Cell,fixture,pi_bounds,EVENTS
from analysis.genesis_volume_banks_validation.independent_bank import fiber
from analysis.genesis_volume_integral_validation.angular_control import _strict_allowance,_sum_enclosures,_abs_interval
from tectonics.genesis_volume_operand_bounds import DirectedInterval,point,add,subtract,multiply,rational_enclosure
from tectonics.genesis_volume_event_partition import signed_double_area,ratio
from tectonics.genesis_volume_ray_roots import restrict_ray

LD=np.longdouble
BASE=((3,3),(5,5))
MIXED=((3,5),(5,3))


def exact(v):return F(*v.as_integer_ratio()) if hasattr(v,'as_integer_ratio') else F(v)
def decode(v):return F(*map(int,v))
def cell_key(c):return (c.triangle,c.s0,c.s1,c.t0,c.t1)


def split(c,axis):
    if axis=='s':
        m=(c.s0+c.s1)/2
        return [Cell(c.triangle,a,b,c.t0,c.t1,c.depth+1) for a,b in ((c.s0,m),(m,c.s1))]
    if axis=='t':
        m=(c.t0+c.t1)/2
        return [Cell(c.triangle,c.s0,c.s1,a,b,c.depth+1) for a,b in ((c.t0,m),(m,c.t1))]
    raise ValueError('Unknown refinement direction')


def indicators(q):
    # Use both paths across the order square: cancellation in Q55-Q33
    # cannot by itself hide a direction from the diagnosis.
    return dict(s=np.maximum(abs(q[5,5]-q[3,5]),abs(q[5,3]-q[3,3])),
                t=np.maximum(abs(q[5,5]-q[5,3]),abs(q[3,5]-q[3,3])))


def normalized(error,allowance):
    result=np.divide(error,allowance,out=np.zeros_like(error,dtype=LD),where=allowance!=0)
    result[(allowance==0)&(error>0)]=np.inf
    return max(result)


class DirectionalExperiment(EnclosureExperiment):
    def __init__(self,raw):
        super().__init__(raw)
        self.samples={};self.plans={};self.panels={};self.requests=0;self.mixed_new_nodes=0

    def plan(self,c,pair):
        key=(cell_key(c),pair)
        if key in self.plans:return self.plans[key]
        ns,nt=pair;result=[];area=signed_double_area(self.triangles[c.triangle])
        for t,tw in self.rules[nt]:
            t=c.t0+(c.t1-c.t0)*t
            a,b,bands,cert=self.planner.ray(c.triangle,t)
            for lo,hi,kind in bands:
                if kind=='root_box':continue
                # Exact names of events adjacent to this root-free band.
                left=tuple(k for k,r in cert['events'].items() if any(y==lo for x,y in r['roots'])) or ('domain_start',)
                right=tuple(k for k,r in cert['events'].items() if any(x==hi for x,y in r['roots'])) or ('domain_end',)
                label='|'.join(left)+' -> '+'|'.join(right)
                for s,sw in self.rules[ns]:
                    s=c.s0+(c.s1-c.s0)*s;rho=lo+(hi-lo)*f(s)
                    jac=area*rho*(hi-lo)*2*(1-s)*(c.s1-c.s0)*(c.t1-c.t0)
                    N=tuple(x+rho*(y-x) for x,y in zip(a,b))
                    result.append((N,jac*sw*tw,label))
        self.plans[key]=result
        return result

    def cost(self,cells,pairs=BASE+MIXED):
        return len({N for c in cells for pair in pairs for N,_,_ in self.plan(c,pair)}-self.samples.keys())

    def rule_pair(self,c,pair):
        value=np.zeros(2,LD);enclosure=point(value);by_band={}
        for N,weight,label in self.plan(c,pair):
            self.requests+=1
            if N not in self.samples:
                if self.nodes>=3400:raise ValueError('All radial calls, including diagnosis, must fit the original cap')
                clock=time.monotonic();bounds=[fiber(self.raw,N,side) for side in (-1,1)]
                if any(b is None for b in bounds):raise ValueError('Unresolved Gamma fiber')
                self.kernel_seconds+=time.monotonic()-clock
                self.samples[N]=DirectedInterval(np.array([rational_enclosure(b[0]).lower for b in bounds],LD),
                    np.array([rational_enclosure(b[1]).upper for b in bounds],LD))
                self.nodes+=1;self.mixed_new_nodes+=int(pair in MIXED)
            sample=self.samples[N];w=rational_enclosure(weight)
            contribution=(w.lower/2+w.upper/2)*(sample.lower/2+sample.upper/2)
            value+=contribution;by_band.setdefault(label,np.zeros(2,LD));by_band[label]+=contribution
            enclosure=add(enclosure,multiply(w,sample))
        upper=self.root_upper(c)
        mids=[rational_enclosure(v/2) for v in upper]
        value+=np.array([v.lower/2+v.upper/2 for v in mids],LD)
        enclosure=add(enclosure,DirectedInterval(np.zeros(2,LD),np.array([rational_enclosure(v).upper for v in upper],LD)))
        return value,enclosure,by_band

    def compute(self,c,pairs=BASE+MIXED):
        key=cell_key(c)
        if key not in self.panels:
            self.panels[key]={};self.evaluated_cells+=1
        data=self.panels[key]
        for pair in pairs:
            if pair not in data:data[pair]=self.rule_pair(c,pair)
        c.coarse,c.coarse_enclosure,_=data[3,3]
        c.fine,c.fine_enclosure,_=data[5,5]
        c.source_error=add(point(c.coarse_enclosure.absolute_error_for(c.coarse)),point(c.fine_enclosure.absolute_error_for(c.fine))).upper
        c.observed=multiply(_abs_interval(subtract(point(c.fine),point(c.coarse))),8).upper
        c.error=add(point(c.observed),point(c.source_error)).upper
        return c

    def reduce(self,cells):
        coarse=np.sum([c.coarse for c in cells],axis=0,dtype=LD);fine=np.sum([c.fine for c in cells],axis=0,dtype=LD)
        cb=_sum_enclosures((c.coarse_enclosure for c in cells),(2,)).absolute_error_for(coarse)
        fb=_sum_enclosures((c.fine_enclosure for c in cells),(2,)).absolute_error_for(fine)
        observed=_sum_enclosures((point(c.observed) for c in cells),(2,)).upper
        source=add(point(cb),point(fb)).upper;error=add(point(observed),point(source)).upper
        allowance=_strict_allowance(2e-6,coarse,fine)
        sums=tuple(sum(self.root_upper(c)[side] for c in cells) for side in (0,1))
        assert sums==tuple(self.global_upper if p else F() for p in self.bank_nonempty_possible)
        assert all(exact(v)>=u for v,u in zip(source,sums))
        record=dict(nodes=self.nodes,coarse_values=list(map(float,coarse)),values=list(map(float,fine)),
            embedded_observations=list(map(float,observed)),source_error=list(map(float,source)),
            combined_errors=list(map(float,error)),allowances=list(map(float,allowance)),
            root_integral_enclosure_widths=list(map(ratio,sums)))
        return record,allowance,bool(np.all(error<=allowance))

    def diagnostic(self,c):
        data=self.panels[cell_key(c)]
        if any(p not in data for p in MIXED):return None
        q={p:v[0] for p,v in data.items()};directions=indicators(q)
        labels=set(k for v in data.values() for k in v[2]);bands=[]
        for label in sorted(labels):
            d=indicators({p:v[2].get(label,np.zeros(2,LD)) for p,v in data.items()})
            bands.append(dict(band=label,s=list(map(float,d['s'])),t=list(map(float,d['t']))))
        return dict(s=list(map(float,directions['s'])),t=list(map(float,directions['t'])),
            diagonal=list(map(float,abs(c.fine-c.coarse))),band_indicators=bands,
            rules={str(p):list(map(float,v[0])) for p,v in data.items()})

    def run(self,baseline=None,lazy_mixed=False):
        refinements=[]
        if baseline is not None:
            saved=json.loads(Path(baseline).read_bytes())
            assert saved['source_geometry']==self.raw
            cells=[Cell(c['triangle'],*(decode(c[k]) for k in ('s0','s1','t0','t1')),c['depth']) for c in saved['cell_records']]
            if self.cost(cells,BASE)>3400:raise ValueError('Baseline replay exceeds diagnosis budget')
            cells=[self.compute(c,BASE) for c in cells];_,allowance,_=self.reduce(cells)
            for c in sorted(cells,key=lambda c:normalized(c.error,allowance),reverse=True):
                if self.nodes+self.cost([c],MIXED)>3400:break
                self.compute(c,MIXED)
            record,_,passed=self.reduce(cells);trace=[record]
            reason='Budgeted directional diagnosis of saved final mesh'
        else:
            pairs=BASE if lazy_mixed else BASE+MIXED
            cells=[Cell(i) for i in range(3)]
            if self.cost(cells,pairs)>3400:raise ValueError('Initial mixed-rule cost exceeds cap')
            cells=[self.compute(c,pairs) for c in cells];trace=[]
            while True:
                record,allowance,passed=self.reduce(cells);trace.append(record)
                if passed:reason='Original embedded/source criterion passed; general proof separate';break
                chosen=max(cells,key=lambda c:normalized(c.error,allowance))
                if lazy_mixed:
                    extra=self.cost([chosen],MIXED)
                    reserve=max(self.cost(split(chosen,d),BASE) for d in ('s','t'))
                    if self.nodes+extra+reserve>3400 or self.evaluated_cells+2>2000 or chosen.depth>=24:
                        reason='Original work/depth cap reached';break
                    self.compute(chosen,MIXED)
                directions=indicators({p:v[0] for p,v in self.panels[cell_key(chosen)].items()})
                axis=max(('s','t'),key=lambda d:normalized(directions[d],allowance))
                children=split(chosen,axis);cost=self.cost(children,pairs)
                if self.nodes+cost>3400 or self.evaluated_cells+2>2000 or chosen.depth>=24:
                    reason='Original work/depth cap reached';break
                refinements.append(dict(parent=list(map(str,cell_key(chosen))),axis=axis,new_nodes=cost,
                    parent_diagnostic=self.diagnostic(chosen)))
                cells.remove(chosen);cells.extend(self.compute(c,pairs) for c in children)
        records=[dict(triangle=c.triangle,s0=ratio(c.s0),s1=ratio(c.s1),t0=ratio(c.t0),t1=ratio(c.t1),depth=c.depth,
            root_integral_upper=list(map(ratio,self.root_upper(c))),observed=list(map(float,c.observed)),diagnostic=self.diagnostic(c)) for c in cells]
        total=np.sum([c.observed for c in cells],axis=0,dtype=LD)
        covered=np.sum([c.observed for c in cells if self.diagnostic(c) is not None],axis=0,dtype=LD)
        return dict(mode='diagnosis' if baseline else 'directional',reason=reason,trace=trace,
            mixed_rules_on_selected_parents_only=bool(lazy_mixed and not baseline),
            radial_line_evaluations=self.nodes,scalar_bank_fiber_calls=2*self.nodes,
            quadrature_node_requests=self.requests,exact_node_cache_hits=self.requests-self.nodes,
            mixed_rule_new_evaluations=self.mixed_new_nodes,root_box_node_evaluations=0,
            ray_partitions_built=len(self.planner.cache),root_boxes_retained=self.planner.root_boxes,
            event_isolation_seconds=self.planner.seconds,kernel_seconds=self.kernel_seconds,
            evaluated_cells=self.evaluated_cells,final_cells=len(cells),cell_records=records,refinements=refinements,
            diagnosed_cells=sum(c['diagnostic'] is not None for c in records),
            diagnosed_embedded_fraction=list(map(float,np.divide(covered,total,out=np.ones(2,LD),where=total!=0))),
            initial_mesh_audit=self.mesh_audit,root_boxes_discarded=False,exact_global_upper=ratio(self.global_upper),
            bank_nonempty_possible=self.bank_nonempty_possible,embedded_gate_passed=passed,
            source_budget_failed_components=sum(s>a for s,a in zip(trace[-1]['source_error'],trace[-1]['allowances'])),
            angular_accuracy_certified=False,source_bounds_added_to_allowance=False,
            relative_tolerance=2e-6,angular_orders=[3,5],max_line_evaluations=3400,max_cells=2000,max_depth=24,
            production_tensor_integrator_changed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',choices=('half_ball','small_half_ball','tangent_Gamma'),default='half_ball')
    p.add_argument('--diagnose-baseline',type=Path)
    p.add_argument('--lazy-mixed',action='store_true',help='Pay mixed-rule diagnosis only for selected parents')
    args=p.parse_args()
    out=args.output.absolute();free=shutil.disk_usage('/workspace').free
    if out.exists() or not out.is_relative_to('/workspace'):raise ValueError('Fresh /workspace output required')
    if free<16*1024**2 or np.finfo(LD).nmant!=63:raise ValueError('Need 16 MiB output reserve and nmant=63')
    raw=fixture(2**-12 if args.case=='small_half_ball' else .25,9.25 if args.case=='tangent_Gamma' else 9.)
    clock=time.monotonic();experiment=DirectionalExperiment(raw);result=experiment.run(args.diagnose_baseline,args.lazy_mixed)
    R=F(raw['seed_radius_m']);truth=[tuple(F(2,3)*R**3*v for v in pi_bounds())]*2
    if args.case=='tangent_Gamma':truth=[tuple(2*v for v in truth[0]),(F(),F())]
    errors=[];coarse_errors=[];passes=[]
    for v,c,(lo,hi) in zip(result['trace'][-1]['values'],result['trace'][-1]['coarse_values'],truth):
        e=max(abs(F(v)-lo),abs(F(v)-hi));ce=max(abs(F(c)-lo),abs(F(c)-hi))
        errors.append(float(e/lo) if lo else float(e));coarse_errors.append(float(ce/lo) if lo else float(ce));passes.append(e<=F(2,10**6)*lo)
    result.update(case=args.case,source_geometry=raw,analytic_truth_bounds=[[ratio(v) for v in t] for t in truth],
        relative_errors_or_absolute_for_zero_truth=errors,coarse_true_relative_errors=coarse_errors,
        analytic_volume_gate_passed=all(passes),all_addressed_gates_passed=all(passes) and result['embedded_gate_passed'],
        seconds=time.monotonic()-clock,free_bytes_before=free,history_advanced=False,physical_admission=False)
    out.mkdir();(out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    rays=[]
    for (index,t),(a,b,bands,cert) in experiment.planner.cache.items():
        rays.append(dict(triangle=index,t=ratio(t),start=list(map(ratio,a)),end=list(map(ratio,b)),
            polynomials={k:list(map(ratio,restrict_ray(experiment.planner.events[k],a,b))) for k in EVENTS},
            isolated={k:dict(identically_zero=v['identically_zero'],roots=[[ratio(lo),ratio(hi)] for lo,hi in v['roots']]) for k,v in cert['events'].items()},
            intervals=[dict(lo=ratio(lo),hi=ratio(hi),kind=kind,integrated=(lo,hi,kind) in bands,
                treatment='integral_enclosure' if kind=='root_box' else 'quadrature_or_proved_zero') for lo,hi,kind in cert['intervals']]))
    (out/'ray_certificates.json').write_text(json.dumps(rays)+'\n')
    paths=[Path(__file__),Path('analysis/genesis_volume_ray_enclosure_validation/experiment.py'),
        Path('analysis/genesis_volume_ray_validation/experiment.py'),Path('analysis/genesis_volume_ray_validation/root_box_bound.py'),
        Path('tectonics/genesis_volume_ray_roots.py')]
    snapshots=out/'source_snapshots';snapshots.mkdir()
    for i,path in enumerate(paths):(snapshots/f'{i}_{path.name}').write_bytes(path.read_bytes())
    (out/'source_pins.json').write_text(json.dumps({str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('case','mode','radial_line_evaluations','relative_errors_or_absolute_for_zero_truth','coarse_true_relative_errors','embedded_gate_passed','diagnosed_cells','diagnosed_embedded_fraction','seconds')}))


if __name__=='__main__':main()
