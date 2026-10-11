"""Exploratory causal diagnosis only; this prototype is not the production repair.

The simplified normalization envelope lacks a resolved positive norm-lower
certificate. Production uses the separately audited `_plane_crossing_error`.
Only runtime monkeypatching is performed; no production/test file is edited.
The original failing source SHA is recorded in this directory's manifest.
"""
import sys,json,tempfile,inspect,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import pytest,numpy as np
import tests.test_geometric_slab_probe as reg
import tectonics.geometric_transport as tr
import tectonics.spherical_polygons as sp
LD=sp._LD;EPS=sp._EPS

def vc(a,b,ea=None,eb=None):
 a,b=np.asarray(a,dtype=LD),np.asarray(b,dtype=LD)
 delta,de=sp._difference_with_error(b,a)
 vals=[];errs=[]
 for j,k in ((1,2),(2,0),(0,1)):
  dirb=LD(8*EPS)*(abs(a[j]*b[k])+abs(a[k]*b[j]))
  difb=LD(8*EPS)*(abs(a[j]*delta[k])+abs(a[k]*delta[j]))+abs(a[j])*de[k]+abs(a[k])*de[j]
  v=(a[j]*delta[k]-a[k]*delta[j]) if difb<=dirb else (a[j]*b[k]-a[k]*b[j])
  e=min(dirb,difb)
  if ea is not None:
   e+=abs(a[j])*eb[k]+abs(a[k])*eb[j]+abs(b[j])*ea[k]+abs(b[k])*ea[j]+ea[j]*eb[k]+ea[k]*eb[j]
  vals.append(v);errs.append(e)
 return np.array(vals,dtype=LD),np.array(errs,dtype=LD)

def crossing_bound(a,b,c,d,crossing):
 p,ep=vc(a,b);q,eq=vc(c,d);v,ev=vc(p,q,ep,eq);n=np.sqrt(np.sum(v*v));en=np.sqrt(np.sum(ev*ev))
 return ev/n+np.abs(np.asarray(crossing,dtype=LD))*en/n+LD(4*EPS)*np.abs(crossing)

sp.crossing_bound=crossing_bound
src=inspect.getsource(sp._clip)
src=src.replace('crossing_error = _LD(4.*_EPS)*(absolute_terms/weighted_norm+np.abs(crossing))','crossing_error = _LD(4.*_EPS)*(absolute_terms/weighted_norm+np.abs(crossing))\n            if plane_points is not None:\n                crossing_error += crossing_bound(*subject_support, *plane_points, crossing)')
exec(src,sp.__dict__)
if '--vertex-only' in sys.argv:
 def ed(v,a,b,diag):
  if np.array_equal(v,a) or np.array_equal(v,b):return LD(0)
  a,b,v=map(lambda x:np.asarray(x,dtype=LD),(a,b,v));n=sp._edge_cross(a,b);norm=np.sqrt(np.dot(n,n));det,bound=sp._determinant_value_and_error(a,b,v);bound+=4*EPS*np.dot(np.abs(n),np.abs(v))
  return LD(0) if abs(det)<=bound else det/norm
 sp._edge_distance=ed
out=Path(tempfile.mkdtemp(prefix='clip-boundary-repair-trial-'))
mp=pytest.MonkeyPatch();config=reg.source.__wrapped__(mp,out)
real_audit=tr.audit_partition;captures=[]
def audit(s,**kw):
 try:return real_audit(s,**kw)
 except Exception as e:
  caps=[]
  for i,j in tr.candidate_pairs(s.fragments):
   a,b=s.fragments[i],s.fragments[j];p=sp.intersect_convex(a.polygon,b.polygon)
   if len(p):caps.append({'a':a.fragment_id,'b':b.fragment_id,'area':sp.polygon_area(p),'polygon_a':a.polygon,'polygon_b':b.polygon,'overlap':p.tolist()})
  captures.append({'time':s.time_myr,'error':str(e),'pairs':caps});raise
mp.setattr(tr,'audit_partition',audit)
try:
 reg.test_two_step_resume_exactly_reproduces_surface_inventory_registry_and_history(config,out,True)
 print('PASSED')
except Exception as e:print(type(e).__name__,str(e))
(out/'captures.json').write_text(json.dumps(captures,indent=2));print(out)
mp.undo()
