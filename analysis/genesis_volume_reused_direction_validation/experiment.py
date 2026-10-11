"""Direction selection from existing 5x5 data; no mixed radial evaluations.

The fourth divided difference is a refinement heuristic only. It never
replaces the original Q33/Q55 acceptance observation or its uncertainty.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import time
import numpy as np

from analysis.genesis_volume_directional_validation.experiment import (
    DirectionalExperiment,Cell,BASE,cell_key,split,normalized,fixture,pi_bounds,EVENTS,
)
from tectonics.genesis_volume_operand_bounds import rational_enclosure
from tectonics.genesis_volume_event_partition import ratio
from tectonics.genesis_volume_ray_roots import restrict_ray

LD=np.longdouble


def fourth_stencil(nodes):
    result=[]
    for i,x in enumerate(nodes):
        denominator=F(1)
        for j,y in enumerate(nodes):
            if i!=j:denominator*=x-y
        result.append(1/denominator)
    return tuple(result)


def tensor_indicators(grid,stencil,weights):
    """Grid axes are t,s,component. Absolute values precede other-axis sums."""
    coefficients=np.array([float(c) for c in stencil],LD)
    weights=np.array([float(w) for w in weights],LD)
    ds=np.sum(grid*coefficients[None,:,None],axis=1,dtype=LD)
    dt=np.sum(grid*coefficients[:,None,None],axis=0,dtype=LD)
    return dict(s=np.sum(abs(ds)*weights[:,None],axis=0,dtype=LD),
                t=np.sum(abs(dt)*weights[:,None],axis=0,dtype=LD))


class ReusedDirectionExperiment(DirectionalExperiment):
    def existing_indicators(self,c):
        if (5,5) not in self.panels[cell_key(c)]:raise ValueError('Fine grid must already have been evaluated')
        before=(self.nodes,self.requests,len(self.planner.cache))
        groups={}
        for N,weight,label in self.plans[cell_key(c),(5,5)]:
            if N not in self.samples:raise AssertionError('Direction heuristic requested a new radial node')
            entries=groups.setdefault(label,[]);t_index,s_index=divmod(len(entries),5)
            if t_index>=5:raise ValueError('Changing band topology requires a different direction diagnostic')
            sw=self.rules[5][s_index][1];tw=self.rules[5][t_index][1]
            jac=rational_enclosure(weight/(sw*tw));sample=self.samples[N]
            entries.append((jac.lower/2+jac.upper/2)*(sample.lower/2+sample.upper/2))
        stencil=fourth_stencil(tuple(n for n,w in self.rules[5]))
        weights=tuple(w for n,w in self.rules[5]);total=dict(s=np.zeros(2,LD),t=np.zeros(2,LD));by_band=[]
        for label,entries in groups.items():
            if len(entries)!=25:raise ValueError('Incomplete stable 5x5 band; no fabricated missing values')
            d=tensor_indicators(np.asarray(entries,LD).reshape(5,5,2),stencil,weights)
            for axis in ('s','t'):total[axis]+=d[axis]
            by_band.append(dict(band=label,**{axis:list(map(float,d[axis])) for axis in ('s','t')}))
        assert before==(self.nodes,self.requests,len(self.planner.cache))
        return total,by_band

    def run(self):
        cells=[Cell(i) for i in range(3)]
        if self.cost(cells,BASE)>3400:raise ValueError('Initial source coverage exceeds cap')
        cells=[self.compute(c,BASE) for c in cells];trace=[];refinements=[]
        while True:
            record,allowance,passed=self.reduce(cells);trace.append(record)
            if passed:reason='Original embedded/source criterion passed; general proof separate';break
            chosen=max(cells,key=lambda c:normalized(c.error,allowance))
            directions,bands=self.existing_indicators(chosen)
            axis=max(('s','t'),key=lambda d:normalized(directions[d],allowance))
            children=split(chosen,axis);cost=self.cost(children,BASE)
            if self.nodes+cost>3400 or self.evaluated_cells+2>2000 or chosen.depth>=24:
                reason='Original work/depth cap reached';break
            parent_observed=np.array(chosen.observed,copy=True)
            cells.remove(chosen);children=[self.compute(c,BASE) for c in children];cells.extend(children)
            child_observed=np.sum([c.observed for c in children],axis=0,dtype=LD)
            before=normalized(parent_observed,allowance);after=normalized(child_observed,allowance)
            refinements.append(dict(parent=list(map(str,cell_key(chosen))),axis=axis,new_nodes=cost,
                directions={a:list(map(float,directions[a])) for a in ('s','t')},band_indicators=bands,
                parent_observed=list(map(float,parent_observed)),child_observed=list(map(float,child_observed)),
                observed_contraction=float(after/before) if before else None))
        records=[]
        for c in cells:
            directions,bands=self.existing_indicators(c)
            records.append(dict(triangle=c.triangle,s0=ratio(c.s0),s1=ratio(c.s1),t0=ratio(c.t0),t1=ratio(c.t1),
                depth=c.depth,root_integral_upper=list(map(ratio,self.root_upper(c))),observed=list(map(float,c.observed)),
                directions={a:list(map(float,directions[a])) for a in ('s','t')},band_indicators=bands))
        assert self.mixed_new_nodes==0
        return dict(mode='directional',directions_from_existing_5x5=True,
            indicator='Fourth divided difference of the already weighted-map density, per band; heuristic only',
            reason=reason,trace=trace,radial_line_evaluations=self.nodes,scalar_bank_fiber_calls=2*self.nodes,
            quadrature_node_requests=self.requests,exact_node_cache_hits=self.requests-self.nodes,
            mixed_rule_new_evaluations=0,direction_new_radial_evaluations=0,root_box_node_evaluations=0,
            ray_partitions_built=len(self.planner.cache),root_boxes_retained=self.planner.root_boxes,
            event_isolation_seconds=self.planner.seconds,kernel_seconds=self.kernel_seconds,
            evaluated_cells=self.evaluated_cells,final_cells=len(cells),cell_records=records,refinements=refinements,
            initial_mesh_audit=self.mesh_audit,root_boxes_discarded=False,exact_global_upper=ratio(self.global_upper),
            bank_nonempty_possible=self.bank_nonempty_possible,embedded_gate_passed=passed,
            source_budget_failed_components=sum(s>a for s,a in zip(trace[-1]['source_error'],trace[-1]['allowances'])),
            angular_accuracy_certified=False,source_bounds_added_to_allowance=False,
            relative_tolerance=2e-6,angular_orders=[3,5],max_line_evaluations=3400,max_cells=2000,max_depth=24,
            production_tensor_integrator_changed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',choices=('half_ball','small_half_ball','tangent_Gamma'),default='half_ball');args=p.parse_args()
    out=args.output.absolute();free=shutil.disk_usage('/workspace').free
    if out.exists() or not out.is_relative_to('/workspace'):raise ValueError('Fresh /workspace output required')
    if free<16*1024**2 or np.finfo(LD).nmant!=63:raise ValueError('Need 16 MiB reserve and nmant=63')
    raw=fixture(2**-12 if args.case=='small_half_ball' else .25,9.25 if args.case=='tangent_Gamma' else 9.)
    clock=time.monotonic();experiment=ReusedDirectionExperiment(raw);result=experiment.run()
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
    paths=[Path(__file__),Path('analysis/genesis_volume_directional_validation/experiment.py'),
        Path('analysis/genesis_volume_ray_enclosure_validation/experiment.py'),Path('analysis/genesis_volume_ray_validation/experiment.py'),
        Path('analysis/genesis_volume_ray_validation/root_box_bound.py'),Path('tectonics/genesis_volume_ray_roots.py')]
    snapshots=out/'source_snapshots';snapshots.mkdir()
    for i,path in enumerate(paths):(snapshots/f'{i}_{path.name}').write_bytes(path.read_bytes())
    (out/'source_pins.json').write_text(json.dumps({str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('case','radial_line_evaluations','direction_new_radial_evaluations','relative_errors_or_absolute_for_zero_truth','embedded_gate_passed','seconds')}))


if __name__=='__main__':main()
