"""Independent audit of root-box integral bounds and their error-budget charge."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
import math
from analysis.genesis_volume_ray_validation.audit import audit_ray,decode


def upper_float(value):
    return F(math.nextafter(value,math.inf)) if value else F()


def audit_budget(report):
    raw=report['source_geometry'];R=F(raw['seed_radius_m']);s=raw['section']
    assert raw['seed_center_world_m']==[1.,1.,9.] and 0<R<=F(1,4)
    assert s['triangle_m']==[[0.,0.,10.],[4.,0.,10.],[0.,4.,10.]]
    assert s['slope_P_m']==s['displacement_A_m']==[[0.,0.,0.]]*3
    assert s['lambda_bounds']==[.8,1.] and s['normal_xyz']==[0.,0.,1.]
    # Independent derivation: coordinate deviation <=10R/(36 lambda_min),
    # max edge direction >=2/9, fiber <= det(X)*lambda_max^2*2R/10.
    lambda_min=(9-R)/10
    rho_bound=(9/F(2))*10*R/(36*lambda_min)
    density_bound=F(160)*F(s['lambda_bounds'][1])**2*(2*R/10)
    U=density_bound*17*rho_bound*rho_bound/2**40
    assert decode(report['exact_global_upper'])==U
    possible=(F(s['offset_m'])>9-R,F(s['offset_m'])<9+R)
    assert tuple(report['bank_nonempty_possible'])==possible
    totals=[F(),F()];areas=(F(5,18),F(4,9),F(5,18))
    # The triangle fan is (center,e0,e1), (center,e1,e2), (center,e2,e0).
    rectangles={i:[] for i in range(3)}
    for c in report['cell_records']:
        a,b,u,v=(decode(c[k]) for k in ('s0','s1','t0','t1'))
        assert 0<=a<b<=1 and 0<=u<v<=1 and 0<=c['depth']<=24
        index=c['triangle'];assert index in rectangles
        for x,y,p,q in rectangles[index]:assert min(b,y)<=max(a,x) or min(v,q)<=max(u,p)
        rectangles[index].append((a,b,u,v))
        measure=(2*(b-a)-(b*b-a*a))*(v-u)*areas[index]
        expected=[U*measure if p else F() for p in possible]
        assert list(map(decode,c['root_integral_upper']))==expected
        totals=[x+y for x,y in zip(totals,expected)]
    assert all(sum((b-a)*(v-u) for a,b,u,v in rect)==1 for rect in rectangles.values())
    assert totals==[U if p else F() for p in possible]
    assert report['relative_tolerance']==2e-6 and not report['source_bounds_added_to_allowance']
    assert (report['max_line_evaluations'],report['max_cells'],report['max_depth'])==(3400,2000,24)
    assert report['angular_orders']==[3,5] and report['root_box_node_evaluations']==0
    for row in report['trace']:
        assert list(map(decode,row['root_integral_enclosure_widths']))==totals
        for bound,source,observed,combined in zip(totals,row['source_error'],row['embedded_observations'],row['combined_errors']):
            # One outward binary64 step covers serialization of long doubles.
            assert upper_float(source)>=bound
            assert upper_float(combined)>=F(source)+F(observed)
    final=report['trace'][-1]
    return dict(final_cells=len(report['cell_records']),exact_parameter_cover_no_overlap=True,
        independent_uniform_integral_upper=str(U),uncertainty_conserved_under_refinement=True,
        full_root_uncertainty_charged_to_error=True,
        source_to_allowance=[s/a if a else (0 if not s else None) for s,a in zip(final['source_error'],final['allowances'])],
        embedded_to_allowance=[e/a if a else (0 if not e else None) for e,a in zip(final['embedded_observations'],final['allowances'])])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('Fresh output required')
    results=[]
    for directory in args.runs:
        report=json.loads((directory/'report.json').read_bytes());budget=audit_budget(report)
        path=directory/'ray_certificates.json';rows=json.loads(path.read_bytes());roots=zeros=0
        for row in rows:
            r,z=audit_ray(report['source_geometry'],row);roots+=r;zeros+=z
            for event in row['isolated'].values():
                for lo,hi in event['roots']:
                    lo,hi=decode(lo),decode(hi)
                    assert 0<=lo<=hi<=1
                    assert lo==hi or hi-lo<=min(lo,1-hi)/2**40
            assert all(r['integrated'] and r['treatment']=='integral_enclosure' for r in row['intervals'] if r['kind']=='root_box')
        results.append(dict(case=report['case'],rays=len(rows),independent_roots=roots,zero_fiber_checks=zeros,
            ray_records_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**budget))
    result=dict(cases=results,scope='Analytic fixtures only; root contributions enclosed, no global angular certification')
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
