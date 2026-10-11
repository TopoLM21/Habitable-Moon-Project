"""Analytic half-ball comparison at the unchanged four-control work cap."""
from fractions import Fraction as F
from pathlib import Path
import json,hashlib,argparse
import numpy as np
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--baseline',action='store_true')
args=parser.parse_args()
if args.baseline:
    from analysis.genesis_volume_mapped_validation.angular import integrate_angular
else:
    from analysis.genesis_volume_banks_validation.angular import integrate_angular
from analysis.genesis_volume_banks_validation.independent_bank import fiber
from tectonics.genesis_volume_operand_bounds import rational_enclosure
from tectonics.genesis_volume_rational_fields import fraction

def atan_bounds(x,terms):
    partial=sum(((-1)**k*x**(2*k+1)/F(2*k+1) for k in range(terms)),F())
    next_term=(-1)**terms*x**(2*terms+1)/F(2*terms+1)
    return min(partial,partial+next_term),max(partial,partial+next_term)
a,b=atan_bounds(F(1,5),120);c,d=atan_bounds(F(1,239),120)
pi=(16*a-4*d,16*b-4*c)
raw=dict(section=dict(triangle_m=[[0.,0.,10.],[4.,0.,10.],[0.,4.,10.]],
    slope_P_m=np.zeros((3,3)).tolist(),displacement_A_m=np.zeros((3,3)).tolist(),
    lambda_bounds=[.8,1.],normal_xyz=[0.,0.,1.],offset_m=9.),
    seed_center_world_m=[1.,1.,9.],seed_radius_m=.25)
# The ball is strictly inside z in [8,10], x>=0, y>=0, x+y<=0.4z.
# Distances to those five planes exceed R, proven by rational squared tests.
R=F(1,4)
assert F(9)-10*fraction(raw['section']['lambda_bounds'][0])>R
assert 10*fraction(raw['section']['lambda_bounds'][1])-9>R
assert R<1 and (F(18,5)-2)**2>R**2*(2+F(4,25))
truth=tuple(F(2,3)*R**3*v for v in pi)
def evaluate(n):
    targets=[fiber(raw,n,side) for side in (-1,1)]
    assert all(v is not None for v in targets)
    lows=[rational_enclosure(v[0]).lower for v in targets]
    highs=[rational_enclosure(v[1]).upper for v in targets]
    values=np.array([(a+b)/2 for a,b in zip(lows,highs)],np.longdouble)
    errors=[]
    for v,lo,hi in zip(values,lows,highs):
        errors.append(rational_enclosure(max(abs(fraction(v)-fraction(lo.item())),abs(fraction(v)-fraction(hi.item())))).upper)
    return values,np.array(errors,np.longdouble)
result=integrate_angular(raw,evaluate,('reference_volume_negative_m3','reference_volume_positive_m3'))
errors=[]
for value in result['fine_values']:
    v=fraction(value);errors.append(float(max(abs(v-truth[0]),abs(v-truth[1]))/truth[0]))
out=args.output;out.mkdir()
report=dict(analytic_reference='2*pi*R^3/3 per bank; Machin alternating rational bounds',
    exact_truth_bounds=[[str(v.numerator),str(v.denominator)] for v in truth],
    fine_relative_errors=errors,original_tolerance=2e-6,angular=result,
    production_sphere_geometry_used_for_reference=False,
    literal_cohort_halfspaces_contain_whole_ball_by_exact_rational_checks=True,
    baseline=args.baseline,
    source_geometry=raw,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
with (out/'report.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(dict(status=result['status'],nodes=result['radial_line_evaluations'],
    missing=result['unresolved_positive_candidate_cells'],relative_error=errors,
    embedded_error=result['observed_embedded_errors'],allowance=result['component_allowances'])))
