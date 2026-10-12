"""Exact polynomial identity check for two serialized arithmetic graphs."""
from fractions import Fraction as F


def expand(graph):
    result=[];maximum=0
    for node in graph['nodes']:
        op=node[0]
        if op=='c':p={(0,0,0):F(node[1])} if F(node[1]) else {}
        elif op=='x':
            k=[0,0,0];k[node[1]]=1;p={tuple(k):F(1)}
        elif op=='+':
            p=dict(result[node[1]])
            for k,v in result[node[2]].items():p[k]=p.get(k,F())+v
        elif op=='*':
            p={}
            for a,x in result[node[1]].items():
                for b,y in result[node[2]].items():
                    k=tuple(i+j for i,j in zip(a,b));p[k]=p.get(k,F())+x*y
        else:raise ValueError('Unknown operation')
        p={k:v for k,v in p.items() if v}
        if len(p)>5000:raise ValueError('Fixed symbolic-term cap')
        maximum=max(maximum,len(p));result.append(p)
    return {key:result[value] for key,value in graph['outputs'].items()},maximum


def check(left,right):
    a,ma=expand(left);b,mb=expand(right)
    assert a==b
    return dict(all_output_polynomials_exactly_equal=True,maximum_intermediate_terms=max(ma,mb),
                output_terms={k:len(v) for k,v in a.items()},method='Exact coefficient equality, not point sampling')
