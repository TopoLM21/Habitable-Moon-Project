import sys,json,hashlib
from pathlib import Path
from fractions import Fraction
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import tectonics.spherical_polygons as sp

def exact_sign(a,b,v):
 a,b,v=[[Fraction(float(x)) for x in row] for row in (a,b,v)]
 d=sum(a[i]*(b[j]*v[k]-b[k]*v[j]) for i,j,k in ((0,1,2),(1,2,0),(2,0,1)))
 return (d>0)-(d<0)
rng=np.random.default_rng(56)
triples=[]
for _ in range(160):
 rows=rng.normal(size=(3,3));rows/=np.linalg.norm(rows,axis=1)[:,None];triples.append(rows)
for exponent in [-20,-30,-100,-150,-160,-310]:
 for _ in range(30):
  rows=rng.normal(size=(3,3));rows[:,:2]*=10.**exponent;rows[:,2]=rng.choice([-1.,1.],size=3);triples.append(rows)
for axis in range(3):
 a=np.zeros(3);a[axis]=1.;b=np.zeros(3);b[(axis+1)%3]=1.;triples.extend([[a,b,a],[a,b,b],[a,b,-a]])
original=sp._LD;side_results=[]
try:
 for dtype in [np.longdouble,np.float64]:
  sp._LD=dtype;bad=[]
  for index,(a,b,v) in enumerate(triples):
   expected=exact_sign(a,b,v);got=sp._saved_ray_side(a,b,v,{})
   if got!=expected:bad.append(dict(index=index,expected=expected,actual=got))
  side_results.append(dict(dtype=str(dtype),cases=len(triples),failures=bad))
finally:sp._LD=original
capture=json.loads(Path(__file__).resolve().parent.parent.joinpath('independent_capture.json').read_text())
records=[]
for index,r in enumerate(capture['overlaps']):
 a,b=[x['polygon'] for x in (r['fragment_a'],r['fragment_b'])]
 normalized=[sp.normalize_polygon(x) for x in (a,b)]
 actual=sp._saved_inputs_separated(*normalized)
 expected=index==1
 records.append(dict(case=f'original_restart_pair_{index}',expected_separated=expected,actual_separated=actual))
old=json.loads(Path(__file__).with_name('four_step_raw_false_pairs.json').read_text())
for index,r in enumerate(old):
 actual=sp._saved_inputs_separated(sp.normalize_polygon(r['a']['polygon']),sp.normalize_polygon(r['b']['polygon']))
 records.append(dict(case=f'four_step_false_pair_{index}',expected_separated=True,actual_separated=actual))
# The shared-plane scratch path is deliberately kept out of the saved-ray gate.
ordinary=np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
flags=dict(binary64=sp._saved_binary64_rays(ordinary),extended=sp._saved_binary64_rays(ordinary.astype(np.longdouble)),inherited=sp._saved_binary64_rays(sp.normalize_polygon(ordinary)))
invalids=[]
for p in [[[1.,0.,0.],[0.,1.,0.]], [[1.,0.,0.],[0.,1.,0.],[float('nan'),0.,1.]]]:
 try:sp.intersect_convex(p,ordinary)
 except ValueError:invalids.append(True)
 else:invalids.append(False)
passed=not any(row['failures'] for row in side_results) and all(r['expected_separated']==r['actual_separated'] for r in records) and flags==dict(binary64=True,extended=False,inherited=False) and all(invalids)
result=dict(status='passed' if passed else 'failed',method='independent Fraction signs versus adaptive/exact saved-ray sign; historical exact positive and separating witnesses',source_sha256=hashlib.sha256(Path(sp.__file__).read_bytes()).hexdigest(),side_results=side_results,separation_records=records,input_precision_flags=flags,invalid_input_validation_precedes_separation=invalids)
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);options=parser.parse_args()
with options.out.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
print(result['status'],result['source_sha256'],side_results,records,flags)
if not passed:sys.exit(1)
