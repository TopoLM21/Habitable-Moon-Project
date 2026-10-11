import argparse,sys,json,hashlib
from decimal import Decimal,localcontext
from fractions import Fraction
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import tectonics.spherical_polygons as sp

def F(x):
 n,d=x.as_integer_ratio();return Fraction(n,d)
def cross(a,b):
 return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def D(x):return Decimal(x.numerator)/Decimal(x.denominator)
def check_case(name,first,second):
 a,b=map(lambda x:np.asarray(x,dtype=sp._LD),first);c,d=map(lambda x:np.asarray(x,dtype=sp._LD),second)
 p,ep=sp._edge_cross_with_error(a,b);q,eq=sp._edge_cross_with_error(c,d)
 if ep is None or eq is None or not np.isfinite(ep).all() or not np.isfinite(eq).all():
  return dict(name=name,status='no_inner_arithmetic_certificate')
 pe=cross(list(map(F,a)),list(map(F,b)));qe=cross(list(map(F,c)),list(map(F,d)))
 err1=[abs(F(x)-y)<=F(e) for x,y,e in zip(p,pe,ep)]
 err2=[abs(F(x)-y)<=F(e) for x,y,e in zip(q,qe,eq)]
 line=sp._edge_cross(p,q);true_line=cross(pe,qe)
 if not any(line):return dict(name=name,status='zero_nominal_line',inner1=err1,inner2=err2)
 try:unit=sp._unit(line)
 except ValueError as error:return dict(name=name,status='normalizer_rejected',inner1=err1,inner2=err2,error=str(error))
 envelope=sp._plane_crossing_error((a,b),(c,d),unit)
 if envelope is None:return dict(name=name,status='no_finite_resolved_norm_certificate',inner1=err1,inner2=err2)
 norm=sum(x*x for x in true_line)
 with localcontext() as ctx:
  ctx.prec=200;n=D(norm).sqrt();exact=[D(x)/n for x in true_line]
  passed=[abs(D(F(u))-v)<=D(F(e)) for u,v,e in zip(unit,exact,envelope)]
  errors=[str(abs(D(F(u))-v)) for u,v in zip(unit,exact)]
 return dict(name=name,status='certified',inner1=err1,inner2=err2,unit_bound_contains_exact=passed,absolute_errors=errors,envelope=[str(x) for x in envelope])

p=argparse.ArgumentParser();p.add_argument('--trace',default=str(Path(__file__).with_name('construction_trace.json')));p.add_argument('--out',required=True);args=p.parse_args()
trace=json.load(open(args.trace))[11]
first=trace['parts'][0]['supports'][0];second=trace['parts'][0]['supports'][1]
cases=[('actual_saved_birth_top_crossing',first,second)]
for tiny in [1e-20,1e-30,1e-100]:
 cases.append((f'positive_tiny_shared_ray_{tiny}',[[1.,0.,0.],[tiny,0.,1.]],[[0.,1.,0.],[tiny,0.,1.]]))
 cases.append((f'resolved_tiny_planes_{tiny}',[[0.,0.,1.],[1.,0.,0.]],[[tiny,tiny,1.],[1.,0.,0.]]))
for tiny in [1e-70,1e-78,1e-80,1e-81,1e-100,1e-150]:
 cases.append((f'two_short_planes_norm_portability_{tiny}',[[0.,0.,1.],[tiny,0.,1.]],[[0.,0.,1.],[0.,tiny,1.]]))
for tiny in [1e-100,1e-150,1e-155]:
 cases.append((f'positive_normalized_components_after_short_plane_cross_{tiny}',
  [[0.,0.,1.],[tiny,2*tiny,1.]],[[tiny,tiny,1.],[3*tiny,4*tiny,1.]]))
rng=np.random.default_rng(1264)
for index in range(12):
 points=rng.normal(size=(4,3));points/=np.linalg.norm(points,axis=1)[:,None];cases.append((f'random_planes_{index}',points[:2],points[2:]))
original=sp._LD;records=[]
try:
 for dtype in [np.longdouble,np.float64]:
  sp._LD=dtype
  records.append(dict(dtype=str(dtype),cases=[check_case(*c) for c in cases]))
finally:sp._LD=original
all_ok=all(all(r.get('inner1',[])) and all(r.get('inner2',[])) and all(r.get('unit_bound_contains_exact',[True])) for group in records for r in group['cases'])
result=dict(status='passed' if all_ok else 'failed',method='exact Fraction plane cross / Decimal200 normalized ray; no assumed input-coordinate uncertainty',source_sha256=hashlib.sha256(Path(sp.__file__).read_bytes()).hexdigest(),records=records)
Path(args.out).write_text(json.dumps(result,indent=2)+'\n');print(result['status'],result['source_sha256'])
if not all_ok:sys.exit(1)
