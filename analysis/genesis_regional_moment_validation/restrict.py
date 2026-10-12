"""Exact de Casteljau half restriction; no new geometric selection."""
from fractions import Fraction as F
from itertools import product


def halve(cert,axis):
    d=cert['degree'];keys=list(product(*(range(n+1) for n in d)))
    values=dict(zip(keys,map(F,cert['coefficients'])));children=[{},{}]
    other=[i for i in range(3) if i!=axis]
    for pair in product(*(range(d[i]+1) for i in other)):
        k=[0,0,0]
        for a,v in zip(other,pair):k[a]=v
        line=[]
        for i in range(d[axis]+1):k[axis]=i;line.append(values[tuple(k)])
        left=[line[0]];right=[line[-1]]
        while len(line)>1:
            line=[(a+b)/2 for a,b in zip(line,line[1:])]
            left.append(line[0]);right.append(line[-1])
        for side,sequence in enumerate((left,list(reversed(right)))):
            for i,v in enumerate(sequence):k[axis]=i;children[side][tuple(k)]=v
    return [dict(degree=d,coefficients=[str(c[k]) for k in keys]) for c in children]
