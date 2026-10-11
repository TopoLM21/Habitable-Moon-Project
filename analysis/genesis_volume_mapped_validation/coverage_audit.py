"""Independent Cramer-polynomial and polygon-vertex audit of initial coverage.

Cofactors use polynomial minors and determinant permutations, separate from
runtime cross products. Polygon vertices are enumerated as all intersections
of bounding lines in (N1,N2), separate from sequential polygon clipping.
"""
import argparse
from fractions import Fraction as F
from itertools import permutations,combinations
import json
from pathlib import Path
import numpy as np
from analysis.genesis_volume_integral_validation.source_control import load_authenticated_inputs,support_source_inputs,seal,pin,require
from analysis.genesis_volume_precision_validation.control import write


def q(value):
    if isinstance(value,(float,np.floating)):return F(*value.as_integer_ratio())
    return F(value)


def dec(pair):return F(*map(int,pair))


def plus(a,b):
    out=[F()]*max(len(a),len(b))
    for i,v in enumerate(a):out[i]+=v
    for i,v in enumerate(b):out[i]+=v
    return out


def times(a,b):
    out=[F()]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(b):out[i+j]+=x*y
    return out


def negative(a):return [-v for v in a]


def audit(raw,certificate):
    s=raw['section'];X,A,P=(np.asarray(s[k],float) for k in ('triangle_m','displacement_A_m','slope_P_m'))
    C,D=X+P,A-P;lo,hi=map(q,s['lambda_bounds']);mid,half=(lo+hi)/2,(hi-lo)/2
    Y=[[[mid*q(C[i,j])+q(D[i,j]),q(C[i,j])] for j in range(3)] for i in range(3)]
    determinant=[F()]*4
    for permutation in permutations(range(3)):
        inversions=sum(permutation[i]>permutation[j] for i in range(3) for j in range(i+1,3))
        term=[F((-1)**inversions)]
        for i,j in enumerate(permutation):term=times(term,Y[i][j])
        determinant=plus(determinant,term)
    require(determinant==list(map(dec,certificate['determinant_coefficients'])),'Independent determinant polynomial differs')
    variation=sum(abs(determinant[k])*half**k for k in range(1,4))
    den=(determinant[0]-variation,determinant[0]+variation)
    require(den==tuple(map(dec,certificate['determinant_bounds'])),'Determinant enclosure differs')
    bounds=[tuple(map(dec,b)) for b in certificate['coordinate_bounds']]
    center=list(map(q,raw['seed_center_world_m']));radius=q(raw['seed_radius_m'])
    witnesses=certificate['coordinate_witnesses']
    if not den[0]<=0<=den[1]:
        require(len(witnesses)==3,'Missing cofactor witnesses')
        for i in range(3):
            coefficients=[]
            rows=[r for r in range(3) if r!=i]
            for j in range(3):
                cols=[c for c in range(3) if c!=j]
                polynomial=plus(times(Y[rows[0]][cols[0]],Y[rows[1]][cols[1]]),
                    negative(times(Y[rows[0]][cols[1]],Y[rows[1]][cols[0]])))
                coefficients.append([(-1)**(i+j)*v for v in polynomial])
            stored=[[dec(v) for v in row] for row in witnesses[i]['cofactor']]
            require(stored==list(map(list,zip(*coefficients))),'Independent cofactor minor differs')
            norm2=sum(sum(abs(p[k])*half**k for k in range(3))**2 for p in coefficients)
            norm=dec(witnesses[i]['cofactor_norm_upper'])
            require(norm>=0 and norm*norm>=norm2,'Cofactor norm fails exact square inequality')
            numerator=[sum(coefficients[j][k]*center[j] for j in range(3)) for k in range(3)]
            spread=sum(abs(numerator[k])*half**k for k in (1,2))+radius*norm
            ends=(numerator[0]-spread,numerator[0]+spread)
            require(ends==tuple(map(dec,witnesses[i]['numerator_bounds'])),'Numerator bounds differ')
            quotients=[a/b for a in ends for b in den]
            require(bounds[i][0]<=max(F(),min(quotients)) and bounds[i][1]>=min(F(1),max(quotients)),
                'Coordinate clipping shrank the exact Cramer enclosure')
    else:require(bounds==[(F(),F(1))]*3,'Singular interval lost full simplex')
    # Each inequality is a*N1+b*N2+c >= 0.
    constraints=[]
    for (a,b,c),(lower,upper) in zip(((-1,-1,1),(1,0,0),(0,1,0)),bounds):
        constraints.extend(((F(a),F(b),F(c)-lower),(-F(a),-F(b),upper-F(c))))
    vertices=set()
    for (a,b,c),(d,e,f) in combinations(constraints,2):
        determinant=a*e-b*d
        if not determinant:continue
        u,v=(b*f-c*e)/determinant,(c*d-a*f)/determinant
        if all(x*u+y*v+z>=0 for x,y,z in constraints):vertices.add((1-u-v,u,v))
    polygon=[tuple(map(dec,row)) for row in certificate['polygon']]
    require(vertices==set(polygon),'Independent bounding-line intersection polygon differs')
    cells=[tuple(tuple(map(dec,row)) for row in cell) for cell in certificate['cells']]
    expected=[(polygon[0],polygon[i],polygon[i+1]) for i in range(1,len(polygon)-1)]
    require(cells==expected,'Polygon fan coverage differs')
    return dict(independent_determinant_and_minor_polynomials=True,
        cofactor_norm_exact_square_inequality=True,independent_halfplane_vertex_enumeration=True,
        initial_support_completeness_proof_verified=True,partition_coverage_is_separate=True,
        polygon_vertices=len(polygon),initial_triangles=len(cells))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--control',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();require(not args.output.exists(),'Fresh audit directory required');args.output.mkdir()
    rows=[]
    for i,item in enumerate(load_authenticated_inputs()['supports']):
        path=args.control/f'case_{i:03d}'/'coverage.json'
        raw=support_source_inputs(item['case'],item['label'])
        rows.append(dict(case=i,certificate_pin=pin(path),**audit(raw,json.loads(path.read_bytes()))))
    write(args.output/'report.json',seal(dict(cases=rows,auditor_pin=pin(Path(__file__)),angular_accuracy_certified=False)))
    print(json.dumps(rows))


if __name__=='__main__':main()
