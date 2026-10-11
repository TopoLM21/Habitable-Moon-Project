"""Exact outer angular support of the literal sphere over its whole cohort.

For x=N Y(lambda), Cramer's rule gives N_i=x.cofactor_i/det(Y).
Centered polynomial bounds hold for every lambda in the original slab;
Cauchy--Schwarz bounds every x in the closed source ball. Exact clipping of
those coordinate bounds therefore cannot remove positive source support.
A determinant interval containing zero retains the entire simplex.
"""
from fractions import Fraction as F
import numpy as np
from .genesis_volume_rational_fields import rational, fraction
from .genesis_volume_event_partition import ratio, digest, literal_cell
from analysis.genesis_volume_operand_validation.independent_geometry_bounds import sqrt_enclosure


def cross(a,b):
    return np.array([a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],object)


def clip(polygon, coordinate, threshold, sense):
    result=[]
    for a,b in zip(polygon,polygon[1:]+polygon[:1]):
        va,vb=sense*(a[coordinate]-threshold),sense*(b[coordinate]-threshold)
        if va>=0:
            result.append(a)
        if (va<0)!=(vb<0):
            result.append(tuple(x+(y-x)*va/(va-vb) for x,y in zip(a,b)))
    # Exact boundary intersections may repeat a vertex.
    return [p for i,p in enumerate(result) if p!=result[i-1]]


def source_coverage(raw):
    section=raw['section']
    X,A,P=(np.asarray(section[k],float) for k in ('triangle_m','displacement_A_m','slope_P_m'))
    C,D=rational(X+P),rational(A-P)
    center=rational(raw['seed_center_world_m']); radius=fraction(raw['seed_radius_m'])
    low,high=map(fraction,np.asarray(section['lambda_bounds'],float))
    if C.shape!=(3,3) or D.shape!=(3,3) or center.shape!=(3,) or not 0<low<high or radius<=0:
        raise ValueError('Invalid source support operands')
    mid,half=(low+high)/2,(high-low)/2
    Y=mid*C+D
    cof=[]
    for i,j in ((1,2),(2,0),(0,1)):
        cof.append(np.stack((cross(Y[i],Y[j]),cross(C[i],Y[j])+cross(Y[i],C[j]),cross(C[i],C[j]))))
    det=[F() for _ in range(4)]
    for k in range(3):
        det[k]+=Y[0]@cof[0][k]; det[k+1]+=C[0]@cof[0][k]
    spread=sum(abs(det[k])*half**k for k in range(1,4))
    denominator=(det[0]-spread,det[0]+spread)
    bounds=[(F(),F(1)) for _ in range(3)]
    witnesses=[]
    if denominator[0]>0 or denominator[1]<0:
        for i,c in enumerate(cof):
            numerator=c@center
            normbox=sum((abs(c[k])*half**k for k in range(3)),np.zeros(3,object))
            norm2=sum(v*v for v in normbox)
            norm=sqrt_enclosure(norm2,binary_places=128)[1]
            spread=sum(abs(numerator[k])*half**k for k in (1,2))+radius*norm
            ends=(numerator[0]-spread,numerator[0]+spread)
            quotients=[a/b for a in ends for b in denominator]
            # Outward dyadic coarsening limits rational node cost; never shrink.
            scale=2**80
            lo=F((min(quotients)*scale).__floor__(),scale)
            hi=F((max(quotients)*scale).__ceil__(),scale)
            bounds[i]=(max(F(),lo),min(F(1),hi))
            witnesses.append(dict(cofactor=[[ratio(v) for v in row] for row in c],
                cofactor_norm_squared_upper=ratio(norm2),cofactor_norm_upper=ratio(norm),
                numerator_bounds=list(map(ratio,ends)),coordinate_bounds=list(map(ratio,bounds[i]))))
    polygon=[tuple(row) for row in rational(np.eye(3))]
    for i,(lo,hi) in enumerate(bounds):
        for bound,sense in ((lo,1),(hi,-1)):
            if polygon:
                polygon=clip(polygon,i,bound,sense)
    cells=[]
    for i in range(1,len(polygon)-1):
        cell=(polygon[0],polygon[i],polygon[i+1])
        from .genesis_volume_event_partition import signed_double_area
        if signed_double_area(cell)>0:
            cells.append(literal_cell(cell))
    body=dict(format='genesis-initial-source-coverage-1',
        source=dict(C=[[ratio(v) for v in row] for row in C],D=[[ratio(v) for v in row] for row in D],
            center=list(map(ratio,center)),radius=ratio(radius),lambda_bounds=list(map(ratio,(low,high)))),
        determinant_coefficients=list(map(ratio,det)),determinant_bounds=list(map(ratio,denominator)),
        coordinate_bounds=[list(map(ratio,b)) for b in bounds],coordinate_witnesses=witnesses,
        singular_interval_retains_full_simplex=not bool(witnesses),
        polygon=[[ratio(v) for v in row] for row in polygon],
        cells=[[[ratio(v) for v in row] for row in cell] for cell in cells],
        initial_source_support_coverage_certified=True,
        proof='Cramer polynomial bounds on full lambda slab; ball Cauchy-Schwarz; outward dyadic bounds; exact halfplane clipping',
        partition_of_supplied_cells_is_separate=True,angular_accuracy_certified=False)
    body['sha256']=digest(body)
    return tuple(cells),body


def verify_coverage(raw,certificate):
    if source_coverage(raw)[1]!=certificate:
        raise ValueError('Initial source coverage certificate differs')
    return True


def exact_node(vertices,s,t):
    """Exact Duffy map of literal binary quadrature parameters, no native N."""
    vertices=literal_cell(vertices); s,t=map(fraction,(s,t))
    if any(sum(row)!=1 for row in vertices) or not 0<=s<=1 or not 0<=t<=1:
        raise ValueError('Exact angular map requires simplex vertices and unit parameters')
    weights=(1-s,s*(1-t),s*t)
    return tuple(sum(weights[i]*vertices[i][j] for i in range(3)) for j in range(3))
