"""Exact scalar-volume inclusion on original (rho,t,lambda) boxes.

No radial evaluator or quadrature sampling. All certification arithmetic is
rational; decimal conversion is reserved for display. Not a tensor solver.
"""
from fractions import Fraction as F
from itertools import product
from math import comb
import heapq

ZERO=(0,0,0)


def constant(x): return {ZERO:F(x)} if x else {}
def add(a,b):
    out=dict(a)
    for k,v in b.items(): out[k]=out.get(k,F())+v
    return {k:v for k,v in out.items() if v}
def scale(a,v): return {k:x*v for k,x in a.items() if x*v}
def mul(a,b):
    out={}
    for k,x in a.items():
        for j,y in b.items():
            h=tuple(u+v for u,v in zip(k,j));out[h]=out.get(h,F())+x*y
    return {k:v for k,v in out.items() if v}
def evaluate(p,x): return sum(v*product_value(xi**ki for xi,ki in zip(x,k)) for k,v in p.items())
def product_value(values):
    result=F(1)
    for v in values: result*=v
    return result
def degree(p): return tuple(max((k[i] for k in p),default=0) for i in range(3))
def indices(d): return product(*(range(n+1) for n in d))


def coefficients(p,box):
    """Affine restriction in power basis, then exact tensor Bernstein map."""
    d=degree(p);power={}
    for k,v in p.items():
        for j in indices(k):
            term=v*product_value(F(comb(ki,ji))*lo**(ki-ji)*(hi-lo)**ji
                                  for ki,ji,(lo,hi) in zip(k,j,box))
            power[j]=power.get(j,F())+term
    return d,{i:sum(v*product_value(F(comb(ii,kk),comb(dd,kk)) for ii,kk,dd in zip(i,k,d))
                   for k,v in power.items() if all(kk<=ii for kk,ii in zip(k,i))) for i in indices(d)}


def integral(p,box):
    return sum(v*product_value((hi**(k+1)-lo**(k+1))/(k+1) for k,(lo,hi) in zip(powers,box))
               for powers,v in p.items())


def classify(qrange,grange,side):
    gl,gu=grange if side==1 else (-grange[1],-grange[0])
    if qrange[0]>0 or gu<0: return 'outside'
    if qrange[1]<=0 and gl>=0: return 'inside'
    return 'unknown'


def enclose(polys,box,empty_banks=()):
    if not polys['G']: raise ValueError('Identically zero Gamma requires separate ownership semantics')
    certificates={}
    for name,p in polys.items():
        d,b=coefficients(p,box)
        certificates[name]=dict(degree=d,coefficients=[str(b[i]) for i in indices(d)])
    ranges={name:(min(map(F,c['coefficients'])),max(map(F,c['coefficients']))) for name,c in certificates.items()}
    volume=product_value(hi-lo for lo,hi in box);result=[];states=[]
    for side in (-1,1):
        state='geometric_empty' if side in empty_banks else classify(ranges['Q'],ranges['G'],side)
        if state in ('outside','geometric_empty'): bounds=(F(),F())
        elif state=='inside':
            exact=integral(polys['K'],box);bounds=(exact,exact)
        else:
            l,u=ranges['K'];bounds=(volume*min(F(),l),volume*max(F(),u))
        states.append(state);result.append(bounds)
    return dict(box=[[str(lo),str(hi)] for lo,hi in box],polynomials=certificates,
                states=states,bounds=[[str(lo),str(hi)] for lo,hi in result])


def fixture_polynomials(raw):
    s=raw['section']
    if any(v for name in ('slope_P_m','displacement_A_m') for row in s[name] for v in row):
        raise ValueError('Only undeformed scalar geometry supported')
    X=[[F(v) for v in row] for row in s['triangle_m']]
    det=(X[0][0]*(X[1][1]*X[2][2]-X[1][2]*X[2][1])-
         X[0][1]*(X[1][0]*X[2][2]-X[1][2]*X[2][0])+X[0][2]*(X[1][0]*X[2][1]-X[1][1]*X[2][0]))
    if det<=0: raise ValueError('Positive reference orientation required')
    rho={(1,0,0):F(1)};t={(0,1,0):F(1)};lam={(0,0,1):F(1)}
    center=(F(4,9),F(5,18),F(5,18));vertices=[tuple(F(i==j) for j in range(3)) for i in range(3)]
    output=[]
    for index in range(3):
        b,c=vertices[index],vertices[(index+1)%3]
        N=[add(constant(center[j]),mul(rho,add(constant(b[j]-center[j]),scale(t,c[j]-b[j])))) for j in range(3)]
        xyz=[mul(lam,add(add(scale(N[0],X[0][j]),scale(N[1],X[1][j])),scale(N[2],X[2][j]))) for j in range(3)]
        Q=constant(-F(raw['seed_radius_m'])**2);G=constant(-F(s['offset_m']))
        for j,x in enumerate(xyz):
            dx=add(x,constant(-F(raw['seed_center_world_m'][j])));Q=add(Q,mul(dx,dx));G=add(G,scale(x,F(s['normal_xyz'][j])))
        area=(b[1]-center[1])*(c[2]-center[2])-(b[2]-center[2])*(c[1]-center[1])
        output.append(dict(Q=Q,G=G,K=scale(mul(rho,mul(lam,lam)),area*det)))
    normal=list(map(F,s['normal_xyz']));norm2=sum(x*x for x in normal)
    radius=F(raw['seed_radius_m']);offset=F(s['offset_m'])-sum(n*F(c) for n,c in zip(normal,raw['seed_center_world_m']))
    if norm2<=0 or radius<=0: raise ValueError('Nonzero normal and positive radius required')
    empty=[]
    if offset>=0 and offset**2>=radius**2*norm2:empty.append(1)
    if offset<=0 and offset**2>=radius**2*norm2:empty.append(-1)
    lo,hi=map(F,s['lambda_bounds'])
    if lo>=hi:raise ValueError('Ordered cohort bounds required')
    return output,((F(),F(1)),(F(),F(1)),(lo,hi)),tuple(empty)


def run(raw,reference,allowance,max_blocks=2000,max_depth=24):
    if not 3<=max_blocks<=2000 or not 0<=max_depth<=24: raise ValueError('Original hard caps cannot be raised')
    polys,initial,empty=fixture_polynomials(raw)
    records=[];leaves=set();heap=[];splits=[];total=[[F(),F()] for _ in range(2)]
    reference=list(map(F,reference));allowance=list(map(F,allowance))
    def create(triangle,box,depth):
        if len(records)>=max_blocks:raise AssertionError('Unreserved block work')
        r=enclose(polys[triangle],box,empty);r.update(id=len(records),triangle=triangle,depth=depth)
        records.append(r);leaves.add(r['id'])
        widths=[]
        for side,(lo,hi) in enumerate(r['bounds']):
            lo,hi=F(lo),F(hi);total[side][0]+=lo;total[side][1]+=hi
            widths.append((hi-lo)/allowance[side] if allowance[side] else (F(0) if hi==lo else None))
        # Exact priority; unresolved zero allowance precedes finite priorities.
        score=max((v for v in widths if v is not None),default=F())
        if depth<max_depth and any(F(hi)>F(lo) for lo,hi in r['bounds']):
            heapq.heappush(heap,(-int(None in widths),-score,r['id']))
        return r['id']
    for i in range(3):create(i,initial,0)
    while True:
        errors=[max(abs(v-lo),abs(hi-v)) for v,(lo,hi) in zip(reference,total)]
        if all(e<=a for e,a in zip(errors,allowance)):reason='certificate_fits_saved_allowance';break
        if not heap:reason='depth_or_unsplittable_limit';break
        if len(records)+2>max_blocks:reason='processed_block_cap';break
        _,_,identifier=heapq.heappop(heap);p=records[identifier];leaves.remove(identifier)
        for side,(lo,hi) in enumerate(p['bounds']):total[side][0]-=F(lo);total[side][1]-=F(hi)
        box=tuple(tuple(map(F,b)) for b in p['box']);axis=p['depth']%3;lo,hi=box[axis];mid=(lo+hi)/2
        children=[]
        for interval in ((lo,mid),(mid,hi)):
            child=list(box);child[axis]=interval;children.append(create(p['triangle'],tuple(child),p['depth']+1))
        splits.append(dict(parent=identifier,axis=axis,children=children))
    return dict(source_geometry=raw,records=records,splits=splits,leaves=sorted(leaves),
                reference=list(map(str,reference)),allowance=list(map(str,allowance)),
                integral_bounds=[[str(v) for v in row] for row in total],error_bounds=list(map(str,errors)),
                certificate_gate_passed=all(e<=a for e,a in zip(errors,allowance)),reason=reason,
                empty_banks=list(empty),processed_blocks=len(records),polynomial_transforms=3*len(records),
                stored_bernstein_coefficients=sum(len(c['coefficients']) for r in records for c in r['polynomials'].values()),
                maximum_depth=max(r['depth'] for r in records),max_blocks=max_blocks,max_depth=max_depth,
                radial_evaluations=0,relative_tolerance='1/500000',scalar_only=True,
                legacy_gate_overridden=False,physical_admission=False)
