"""Budgeted scalar trial using existing-grid interpolation residual marginals."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import time
import numpy as np
from analysis.genesis_volume_reused_direction_validation.experiment import ReusedDirectionExperiment
from analysis.genesis_volume_residual_diagnosis.target_check import diagnose
from analysis.genesis_volume_directional_validation.experiment import exact,fixture,pi_bounds,EVENTS
from tectonics.genesis_volume_ray_roots import restrict_ray
from tectonics.genesis_volume_event_partition import ratio


class ResidualExperiment(ReusedDirectionExperiment):
    def existing_indicators(self,c):
        directions,bands,_=diagnose(self,c)
        return {a:np.array([float(v) for v in values],np.longdouble) for a,values in directions.items()},bands

    def run(self):
        result=super().run()
        result['directions_from_existing_3x3_and_5x5']=True
        result['directions_from_existing_5x5']=False
        result['indicator']='Centered weighted marginals of coarse values minus tensor quartic interpolation of fine values; heuristic only'
        return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',choices=('half_ball','small_half_ball','tangent_Gamma'),required=True);args=p.parse_args()
    out=args.output.absolute()
    if out.exists() or not out.is_relative_to('/workspace'):raise ValueError('Fresh /workspace output required')
    if shutil.disk_usage('/workspace').free<32*1024**2 or np.finfo(np.longdouble).nmant!=63:
        raise ValueError('Need nmant=63 and 32 MiB reserve')
    e=ResidualExperiment(fixture(2**-12 if args.case=='small_half_ball' else .25,9.25 if args.case=='tangent_Gamma' else 9.))
    clock=time.monotonic();result=e.run();R=F(e.raw['seed_radius_m'])
    truth=[tuple(F(2,3)*R**3*v for v in pi_bounds())]*2
    if args.case=='tangent_Gamma':truth=[tuple(2*v for v in truth[0]),(F(),F())]
    errors=[];coarse_errors=[];passes=[]
    for v,c,(lo,hi) in zip(result['trace'][-1]['values'],result['trace'][-1]['coarse_values'],truth):
        error=max(abs(F(v)-lo),abs(F(v)-hi));ce=max(abs(F(c)-lo),abs(F(c)-hi))
        errors.append(float(error/lo) if lo else float(error));coarse_errors.append(float(ce/lo) if lo else float(ce))
        passes.append(error<=F(2,10**6)*lo)
    result.update(case=args.case,source_geometry=e.raw,analytic_truth_bounds=[[ratio(v) for v in t] for t in truth],
                  relative_errors_or_absolute_for_zero_truth=errors,coarse_true_relative_errors=coarse_errors,
                  analytic_volume_gate_passed=all(passes),all_addressed_gates_passed=all(passes) and result['embedded_gate_passed'],
                  seconds=time.monotonic()-clock,history_advanced=False,physical_admission=False,
                  prerequisite_diagnostic_calls_separate=33)
    out.mkdir();(out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    rays=[]
    for (index,t),(a,b,bands,cert) in e.planner.cache.items():
        rays.append(dict(triangle=index,t=ratio(t),start=list(map(ratio,a)),end=list(map(ratio,b)),
            polynomials={k:list(map(ratio,restrict_ray(e.planner.events[k],a,b))) for k in EVENTS},
            isolated={k:dict(identically_zero=v['identically_zero'],roots=[[ratio(lo),ratio(hi)] for lo,hi in v['roots']]) for k,v in cert['events'].items()},
            intervals=[dict(lo=ratio(lo),hi=ratio(hi),kind=kind,integrated=(lo,hi,kind) in bands,
                treatment='integral_enclosure' if kind=='root_box' else 'quadrature_or_proved_zero') for lo,hi,kind in cert['intervals']]))
    (out/'ray_certificates.json').write_text(json.dumps(rays)+'\n')
    samples=[dict(N=list(map(str,N)),lower=[str(exact(v)) for v in sample.lower],upper=[str(exact(v)) for v in sample.upper])
             for N,sample in e.samples.items()]
    (out/'samples.json').write_text(json.dumps(samples)+'\n')
    paths=[Path(__file__),Path('analysis/genesis_volume_residual_diagnosis/target_check.py'),
           Path('analysis/genesis_volume_residual_diagnosis/residual.py'),
           Path('analysis/genesis_volume_reused_direction_validation/experiment.py'),
           Path('analysis/genesis_volume_directional_validation/experiment.py'),
           Path('analysis/genesis_volume_ray_enclosure_validation/experiment.py')]
    snapshots=out/'source_snapshots';snapshots.mkdir()
    for i,path in enumerate(paths):(snapshots/f'{i}_{path.name}').write_bytes(path.read_bytes())
    (out/'source_pins.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('case','radial_line_evaluations','direction_new_radial_evaluations',
                       'relative_errors_or_absolute_for_zero_truth','embedded_gate_passed','seconds')}))


if __name__=='__main__':main()
