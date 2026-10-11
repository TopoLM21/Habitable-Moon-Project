"""Experimental exact real-root isolation for polynomial events on Duffy rays.

Root boxes are retained as unresolved intervals, never labelled empty.
Sturm counts concern the exact rational polynomial, including even roots.
This proves a one-dimensional partition, not angular quadrature accuracy.
"""
from fractions import Fraction as F
from math import gcd, lcm


class RootIsolationRefusal(ValueError):
    pass


def trim(p):
    p=list(map(F,p))
    while p and not p[-1]:p.pop()
    return tuple(p)


def evaluate(p,x):
    result=F()
    for coefficient in reversed(p):result=result*x+coefficient
    return result


def multiply(p,q):
    out=[F()]*(len(p)+len(q)-1) if p and q else []
    for i,a in enumerate(p):
        for j,b in enumerate(q):out[i+j]+=a*b
    return trim(out)


def derivative(p):return trim(i*c for i,c in enumerate(p) if i)


def divide(p,q):
    p=list(trim(p));q=trim(q)
    if not q:raise ZeroDivisionError('Zero polynomial divisor')
    result=[F()]*max(0,len(p)-len(q)+1)
    while p and len(p)>=len(q):
        i=len(p)-len(q);factor=p[-1]/q[-1];result[i]=factor
        for j,c in enumerate(q):p[i+j]-=factor*c
        p=list(trim(p))
    return trim(result),trim(p)


def primitive(p):
    """Positive scaling only: Sturm signs must not be flipped."""
    p=trim(p)
    if not p:return p
    denominator=lcm(*(v.denominator for v in p))
    integers=[v.numerator*(denominator//v.denominator) for v in p]
    common=gcd(*integers)
    return tuple(F(v//common) for v in integers)


def squarefree(p):
    p=trim(p)
    if len(p)<2:return p
    a,b=p,derivative(p)
    while b:a,b=b,primitive(divide(a,b)[1])
    q,r=divide(p,a)
    assert not r
    return primitive(q)


def sturm(p):
    p=squarefree(p)
    if len(p)<2:return (p,)
    sequence=[p,primitive(derivative(p))]
    while sequence[-1]:
        remainder=divide(sequence[-2],sequence[-1])[1]
        if not remainder:break
        sequence.append(primitive(-v for v in remainder))
    return tuple(sequence)


def variations(sequence,x):
    signs=[1 if v>0 else -1 for p in sequence if (v:=evaluate(p,x))]
    return sum(a!=b for a,b in zip(signs,signs[1:]))


def count_open(sequence,lo,hi):
    if not lo<hi:return 0
    return variations(sequence,lo)-variations(sequence,hi)-int(evaluate(sequence[0],hi)==0)


def isolate(p,*,bits=40,max_depth=512):
    """Enclose all distinct roots on [0,1], exactly if encountered rationally.

    Each nonpoint box contains one root and has relative width bounded by
    2**-bits times its distance from the nearer domain endpoint. Multiple
    roots are counted once, after exact square-free factorization.
    """
    p=trim(p)
    if type(bits) is not int or bits<1 or type(max_depth) is not int or max_depth<1:
        raise ValueError('Invalid root isolation limits')
    if not p:return dict(identically_zero=True,roots=(),distinct_roots=None)
    if len(p)==1:return dict(identically_zero=False,roots=(),distinct_roots=0)
    if len(p)==2:
        root=-p[0]/p[1];roots=((root,root),) if 0<=root<=1 else ()
        return dict(identically_zero=False,roots=roots,distinct_roots=len(roots))
    sequence=sturm(p);q=sequence[0];roots=[]
    for endpoint in (F(),F(1)):
        if evaluate(q,endpoint)==0:roots.append((endpoint,endpoint))
    stack=[(F(),F(1),count_open(sequence,F(),F(1)),0,None)]
    while stack:
        lo,hi,count,depth,target=stack.pop()
        if not count:continue
        if count==1 and target is None:target=(hi-lo)/2**bits
        if (count==1 and evaluate(q,lo)!=0 and evaluate(q,hi)!=0
                and hi-lo<=min(target,min(lo,1-hi)/2**bits)):
            roots.append((lo,hi));continue
        if depth>=max_depth:
            raise RootIsolationRefusal('Unisolated roots retained; root depth cap reached')
        mid=(lo+hi)/2;at_root=evaluate(q,mid)==0
        left=count_open(sequence,lo,mid);right=count_open(sequence,mid,hi)
        assert left+right+int(at_root)==count
        if at_root:roots.append((mid,mid))
        stack.extend(((mid,hi,right,depth+1,target),(lo,mid,left,depth+1,target)))
    roots=tuple(sorted(roots))
    return dict(identically_zero=False,roots=roots,distinct_roots=len(roots))


def restrict_ray(polynomial,start,end):
    """Substitute N=start+rho*(end-start) with no floating coordinates."""
    if len(start)!=polynomial.variables or len(end)!=polynomial.variables:
        raise ValueError('Ray dimension mismatch')
    result=[F()]*(polynomial.degree+1)
    for powers,coefficient in polynomial.coefficients.items():
        term=(coefficient,)
        for a,b,power in zip(start,end,powers):
            for _ in range(power):term=multiply(term,(F(a),F(b)-F(a)))
        for i,c in enumerate(term):result[i]+=c
    return trim(result)


def partition(polynomials,*,bits=40,max_depth=512):
    """Cover [0,1] by disjoint interiors of root-free gaps and root boxes.

    Overlapping boxes of different events are united, not sorted by guessed
    root locations. Such a box may contain several roots and stays unresolved.
    Exact root points have zero measure and remain explicit in the record.
    """
    records={name:isolate(p,bits=bits,max_depth=max_depth) for name,p in polynomials.items()}
    boxes=sorted(box for r in records.values() for box in r['roots'])
    merged=[]
    for lo,hi in boxes:
        if merged and lo<=merged[-1][1]:merged[-1]=(merged[-1][0],max(hi,merged[-1][1]))
        else:merged.append((lo,hi))
    intervals=[];cursor=F()
    for lo,hi in merged:
        if cursor<lo:intervals.append((cursor,lo,'root_free'))
        if lo<hi:intervals.append((lo,hi,'root_box'))
        cursor=hi
    if cursor<1:intervals.append((cursor,F(1),'root_free'))
    assert intervals and intervals[0][0]==0 and intervals[-1][1]==1
    assert all(a<b for a,b,_ in intervals)
    assert all(a[1]==b[0] for a,b in zip(intervals,intervals[1:]))
    assert sum(b-a for a,b,_ in intervals)==1
    return dict(intervals=tuple(intervals),events=records,root_boxes=tuple(merged),
        identically_zero_events=tuple(k for k,v in records.items() if v['identically_zero']))
