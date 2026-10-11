"""Read-only independent audit of existing analytic half-ball trial reports.

No quadrature, checkpoint restore, or evolution is run. The independent
triangle-intersection and bank-fiber auditors are reused. A different atan
identity recomputes pi instead of trusting the trial's stored truth or errors.
"""
import argparse
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path

from analysis.genesis_volume_banks_validation.partition_audit import check, decode
from analysis.genesis_volume_banks_validation.independent_bank import audit_certificates
from tectonics.genesis_volume_source_coverage import source_coverage


def pin(path):
    data=path.read_bytes()
    return dict(path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def atan_bounds(x):
    n=128
    s=sum(((-1)**k*x**(2*k+1)/F(2*k+1) for k in range(n)),F())
    t=s+(-1)**n*x**(2*n+1)/F(2*n+1)
    return min(s,t),max(s,t)


def audit(path):
    report=json.loads(path.read_bytes());a=report['angular'];raw=report['source_geometry']
    # Authenticate the analytic geometry, not merely a label in its report.
    assert raw['section']==dict(triangle_m=[[0.,0.,10.],[4.,0.,10.],[0.,4.,10.]],
        slope_P_m=[[0.,0.,0.]]*3,displacement_A_m=[[0.,0.,0.]]*3,
        lambda_bounds=[.8,1.],normal_xyz=[0.,0.,1.],offset_m=9.)
    assert raw['seed_center_world_m']==[1.,1.,9.] and raw['seed_radius_m']==.25
    radius=F(1,4)
    assert F(9)-10*F(.8)>radius and 10-9>radius and 1>radius
    assert (F(18,5)-2)**2>radius**2*(2+F(4,25))
    # atan(1/2)+atan(1/3)=pi/4, both angles in (0,pi/2).
    b,c=atan_bounds(F(1,2)),atan_bounds(F(1,3))
    truth=tuple(F(8,3)*radius**3*(b[i]+c[i]) for i in (0,1))
    error_bounds=[]
    for v in a['fine_values']:
        # Cover JSON binary64 rounding of the original long-double sum.
        lo=F(math.nextafter(v,-math.inf));hi=F(math.nextafter(v,math.inf))
        lower=max(F(),truth[0]-hi,lo-truth[1])/truth[1]
        upper=max(abs(lo-truth[1]),abs(hi-truth[0]))/truth[0]
        assert lower>F(2,10**6), 'This trial needs a new disposition'
        error_bounds.append([float(lower),float(upper)])
    assert a['relative_tolerance']==2e-6 and a['angular_orders']==[3,5]
    assert (a['max_cells'],a['max_depth'],a['max_line_evaluations'])==(2000,24,3400)
    roots,coverage=source_coverage(raw)
    assert coverage['sha256']==a['coverage_sha256']
    certificates=[c['bank_bound_certificate'] for c in a['final_cells']]+a['discarded_cells']
    partition=check([tuple(n[1:] for n in c) for c in roots],
                    [decode(c['vertices']) for c in certificates])
    fibers=audit_certificates(raw,certificates)
    return dict(report_pin=pin(path),nodes=a['radial_line_evaluations'],
        missing_cells=a['unresolved_positive_candidate_cells'],
        source_budget_failures=a['source_uncertainty_failed_component_count'],
        sphere_events=a['sphere_event_guided_refinement'],smoothing=a['endpoint_smoothing'],
        relative_error_bounds_from_independent_pi=error_bounds,
        both_banks_fail_original_tolerance=True,partition_audit=partition,
        sampled_fiber_audit=fibers,numerical_integration_repeated=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh output required')
    trials=[]
    for path in args.reports:
        result=audit(path);trials.append(result)
        print(json.dumps(dict(trial=str(path.parent),nodes=result['nodes'],
            errors=result['relative_error_bounds_from_independent_pi'],audit='PASS')),flush=True)
    result=dict(format='genesis-exact-events-sphere-trial-readback-1',
        independent_pi_identity='4*(atan(1/2)+atan(1/3)); alternating rational bounds',
        auditor_pin=pin(Path(__file__)),trials=trials,
        four_original_tensor_controls_rerun=False,
        disposition='Curved-event refinement and smoothing rejected; exact Gamma cuts retained separately')
    with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')


if __name__=='__main__':main()
