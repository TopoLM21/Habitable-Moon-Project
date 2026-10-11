"""Analytic-fixture experiment with rigorously enclosed root-box integrals.

The previous ray experiment is preserved. This adapter only supports its
documented undeformed analytic family. It is not a production tensor kernel.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import time
import numpy as np

from analysis.genesis_volume_ray_validation.experiment import (
    Experiment as PreviousExperiment,Cell,fixture,pi_bounds,EVENTS,
)
from analysis.genesis_volume_ray_validation.root_box_bound import fixture_root_box_mass_bound
from analysis.genesis_volume_banks_validation.independent_bank import fiber
from analysis.genesis_volume_integral_validation.angular_control import _strict_allowance,_sum_enclosures
from tectonics.genesis_volume_operand_bounds import DirectedInterval,point,add,multiply,rational_enclosure
from tectonics.genesis_volume_event_partition import signed_double_area,ratio
from tectonics.genesis_volume_ray_roots import restrict_ray

LD=np.longdouble


def f(s):return 2*s-s*s


class EnclosureExperiment(PreviousExperiment):
    def __init__(self,raw):
        radius=raw['seed_radius_m'];offset=raw['section']['offset_m']
        if raw!=fixture(radius,offset):raise ValueError('Analytic undeformed fixture only; no general tensor bound')
        self.global_upper=fixture_root_box_mass_bound(F(radius))
        # Exact sphere/plane separation, including tangency. An empty bank is
        # proved geometrically, never inferred from zero quadrature samples.
        self.bank_nonempty_possible=(F(offset)>9-F(radius),F(offset)<9+F(radius))
        super().__init__(raw)

    def root_upper(self,cell):
        factor=signed_double_area(self.triangles[cell.triangle])*(cell.t1-cell.t0)*(f(cell.s1)-f(cell.s0))
        return tuple(self.global_upper*factor if possible else F() for possible in self.bank_nonempty_possible)

    def cost(self,cell):
        return sum(order*sum(sum(kind!='root_box' for _,_,kind in
            self.planner.ray(cell.triangle,cell.t0+(cell.t1-cell.t0)*t)[2])
            for t,_ in self.rules[order]) for order in (3,5))

    def rule(self,cell,order):
        value=np.zeros(2,LD);enclosure=point(value)
        area=signed_double_area(self.triangles[cell.triangle])
        for t,tw in self.rules[order]:
            t=cell.t0+(cell.t1-cell.t0)*t
            start,end,bands,_=self.planner.ray(cell.triangle,t)
            for lo,hi,kind in bands:
                if kind=='root_box':continue  # Added below as [0,U], not zero.
                for s,sw in self.rules[order]:
                    s=cell.s0+(cell.s1-cell.s0)*s
                    rho=lo+(hi-lo)*f(s)
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
                    self.nodes+=1
        exact_upper=self.root_upper(cell)
        upper=np.array([rational_enclosure(v).upper for v in exact_upper],LD)
        midpoint=[rational_enclosure(v/2) for v in exact_upper]
        value+=np.array([v.lower/2+v.upper/2 for v in midpoint],LD)
        enclosure=add(enclosure,DirectedInterval(np.zeros(2,LD),upper))
        return value,enclosure

    def run(self):
        cells=[Cell(i) for i in range(3)]
        if sum(map(self.cost,cells))>3400:raise ValueError('Initial source coverage exceeds original work cap')
        cells=list(map(self.compute,cells));trace=[]
        while True:
            coarse=np.sum([c.coarse for c in cells],axis=0,dtype=LD)
            fine=np.sum([c.fine for c in cells],axis=0,dtype=LD)
            cb=_sum_enclosures((c.coarse_enclosure for c in cells),(2,)).absolute_error_for(coarse)
            fb=_sum_enclosures((c.fine_enclosure for c in cells),(2,)).absolute_error_for(fine)
            observed=_sum_enclosures((point(c.observed) for c in cells),(2,)).upper
            source=add(point(cb),point(fb)).upper
            error=add(point(observed),point(source)).upper
            allowance=_strict_allowance(2e-6,coarse,fine)
            sums=tuple(sum(self.root_upper(c)[side] for c in cells) for side in (0,1))
            assert sums==tuple(self.global_upper if p else F() for p in self.bank_nonempty_possible)
            # Both coarse and fine enclosures pay their half-width; no credit
            # is added to the tolerance. Include summation rounding as well.
            assert all(F(*v.as_integer_ratio())>=bound for v,bound in zip(source,sums))
            trace.append(dict(nodes=self.nodes,values=list(map(float,fine)),
                embedded_observations=list(map(float,observed)),source_error=list(map(float,source)),
                combined_errors=list(map(float,error)),allowances=list(map(float,allowance)),
                root_integral_enclosure_widths=list(map(ratio,sums))))
            if np.all(error<=allowance):reason='Embedded and enclosure criterion passed; angular proof separate';break
            def priority(c):
                r=np.divide(c.error,allowance,out=np.zeros(2,LD),where=allowance!=0)
                r[(allowance==0)&(c.error>0)]=np.inf
                return max(r)
            chosen=max(cells,key=priority);children=chosen.children()
            cost=sum(map(self.cost,children))
            if self.nodes+cost>3400 or self.evaluated_cells+4>2000 or chosen.depth>=24:
                reason='Original work/depth cap reached';break
            cells.remove(chosen);cells.extend(map(self.compute,children))
        return dict(reason=reason,trace=trace,radial_line_evaluations=self.nodes,
            scalar_bank_fiber_calls=2*self.nodes,root_box_node_evaluations=0,
            ray_partitions_built=len(self.planner.cache),root_boxes_retained=self.planner.root_boxes,
            event_isolation_seconds=self.planner.seconds,kernel_seconds=self.kernel_seconds,
            evaluated_cells=self.evaluated_cells,final_cells=len(cells),
            cell_records=[dict(triangle=c.triangle,s0=ratio(c.s0),s1=ratio(c.s1),t0=ratio(c.t0),t1=ratio(c.t1),
                depth=c.depth,root_integral_upper=list(map(ratio,self.root_upper(c)))) for c in cells],
            initial_mesh_audit=self.mesh_audit,root_boxes_discarded=False,
            root_integral_treatment='Strict integral [0,U] plus midpoint estimate; full uncertainty charged',
            exact_global_upper=ratio(self.global_upper),bank_nonempty_possible=self.bank_nonempty_possible,
            root_bound_geometry_scope='Only the explicitly validated analytic fixture family',
            embedded_gate_passed=bool(np.all(error<=allowance)),
            source_budget_failed_components=int(np.sum(source>allowance)),
            angular_accuracy_certified=False,source_bounds_added_to_allowance=False,
            relative_tolerance=2e-6,angular_orders=[3,5],max_line_evaluations=3400,
            max_cells=2000,max_depth=24,production_tensor_integrator_changed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',choices=('half_ball','small_half_ball','tangent_Gamma'),default='half_ball')
    args=p.parse_args();out=args.output.absolute();free=shutil.disk_usage('/workspace').free
    if out.exists() or not out.is_relative_to('/workspace'):raise ValueError('Fresh /workspace output required')
    if free<16*1024**2:raise ValueError('Insufficient 16 MiB experiment reserve')
    if np.finfo(LD).nmant!=63:raise ValueError('Requires longdouble nmant=63')
    raw=fixture(2**-12 if args.case=='small_half_ball' else .25,9.25 if args.case=='tangent_Gamma' else 9.)
    start=time.monotonic();experiment=EnclosureExperiment(raw);result=experiment.run()
    R=F(raw['seed_radius_m']);truth=[tuple(F(2,3)*R**3*v for v in pi_bounds())]*2
    if args.case=='tangent_Gamma':truth=[tuple(2*v for v in truth[0]),(F(),F())]
    errors=[];passes=[]
    for v,(lo,hi) in zip(result['trace'][-1]['values'],truth):
        e=max(abs(F(v)-lo),abs(F(v)-hi));errors.append(float(e/lo) if lo else float(e));passes.append(e<=F(2,10**6)*lo)
    result.update(case=args.case,source_geometry=raw,analytic_truth_bounds=[[ratio(v) for v in b] for b in truth],
        relative_errors_or_absolute_for_zero_truth=errors,analytic_volume_gate_passed=all(passes),
        all_addressed_gates_passed=all(passes) and result['embedded_gate_passed'],seconds=time.monotonic()-start,
        free_bytes_before=free,history_advanced=False,physical_admission=False)
    out.mkdir();(out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    rays=[]
    for (index,t),(a,b,bands,cert) in experiment.planner.cache.items():
        rays.append(dict(triangle=index,t=ratio(t),start=list(map(ratio,a)),end=list(map(ratio,b)),
            polynomials={k:list(map(ratio,restrict_ray(experiment.planner.events[k],a,b))) for k in EVENTS},
            isolated={k:dict(identically_zero=v['identically_zero'],roots=[[ratio(lo),ratio(hi)] for lo,hi in v['roots']]) for k,v in cert['events'].items()},
            intervals=[dict(lo=ratio(lo),hi=ratio(hi),kind=kind,integrated=(lo,hi,kind) in bands,
                treatment='integral_enclosure' if kind=='root_box' else 'quadrature_or_proved_zero') for lo,hi,kind in cert['intervals']]))
    (out/'ray_certificates.json').write_text(json.dumps(rays)+'\n')
    paths=[Path(__file__),Path('tectonics/genesis_volume_ray_roots.py'),
        Path('analysis/genesis_volume_ray_validation/experiment.py'),Path('analysis/genesis_volume_ray_validation/root_box_bound.py')]
    pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    snapshots=out/'source_snapshots';snapshots.mkdir()
    for index,p in enumerate(paths):(snapshots/f'{index}_{p.name}').write_bytes(p.read_bytes())
    (out/'source_pins.json').write_text(json.dumps(pins,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('case','radial_line_evaluations','relative_errors_or_absolute_for_zero_truth','analytic_volume_gate_passed','embedded_gate_passed','source_budget_failed_components','seconds')}))


if __name__=='__main__':main()
