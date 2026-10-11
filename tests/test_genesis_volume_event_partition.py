"""Exact polynomial identities, coverage, and no-discard event regressions."""
from copy import deepcopy
from fractions import Fraction as F
import math
import random

import numpy as np
import pytest

from tectonics.genesis_volume_event_partition import (
    Polynomial,source_events,bernstein_coefficients,sign_certificate,
    verify_sign_certificate,split_cell,signed_double_area,literal_cell,
    partition_events,verify_partition,digest,
)
from analysis.genesis_volume_operand_validation.independent_geometry_bounds import source_line, _source_interval

CELL=tuple(tuple(F(int(i==j)) for j in range(3)) for i in range(3))


def raw():
    return dict(section=dict(triangle_m=np.eye(3)*4,displacement_A_m=np.zeros((3,3)),
        slope_P_m=np.zeros((3,3)),normal_xyz=(1.,0.,0.),offset_m=1.,lambda_bounds=(.25,.75)),
        seed_center_world_m=(1.,1.,1.),seed_radius_m=.5)


@pytest.mark.parametrize('seed',range(6))
def test_bernstein_partition_unity_and_exact_polynomial_values(seed):
    rng=random.Random(seed)
    poly=Polynomial(3,{(i,j,k):F(rng.randrange(-11,12),16)
        for i in range(5) for j in range(5-i) for k in range(5-i-j)})
    vertices=split_cell(CELL)[seed%4]
    coefficients=bernstein_coefficients(poly,vertices)
    degree=poly.degree
    for i in range(9):
        for j in range(9-i):
            B=(F(8-i-j,8),F(i,8),F(j,8))
            N=tuple(sum(B[k]*vertices[k][axis] for k in range(3)) for axis in range(3))
            value=F();weight_sum=F()
            for powers,c in coefficients.items():
                weight=F(math.factorial(degree),math.prod(math.factorial(p) for p in powers))
                weight*=math.prod(x**p for x,p in zip(B,powers))
                assert weight>=0
                weight_sum+=weight;value+=weight*c
            assert weight_sum==1
            assert value==poly.evaluate(N)
            assert min(coefficients.values())<=value<=max(coefficients.values())


@pytest.mark.parametrize('N',[(1.,0.,0.),(0.,1.,0.),(.5,.25,.25),(.125,.375,.5)])
def test_literal_source_event_values_match_independent_geometry_algebra(N):
    source=raw();events,_=source_events(source);line=source_line(source,N)
    n=line['n'];A,B,E=line['uu'],line['uv'],line['vv']-line['radius']**2
    g,h=line['slope'],line['intercept']
    expected=dict(A=A,B=B,E=E,Delta=B*B-A*E,g=g,h=h,Gamma_sphere_resultant=A*h*h-2*B*h*g+E*g*g)
    for name,x in zip(('lower','upper'),line['bounds']):
        expected['sphere_cohort_'+name]=A*x*x+2*B*x+E
        expected['sphere_derivative_half_'+name]=A*x+B
        expected['Gamma_cohort_'+name]=g*x+h
    expected.update({f'N_{i}':v for i,v in enumerate(n)})
    assert set(events)==set(expected)
    assert all(poly.evaluate(n)==expected[name] for name,poly in events.items())


def test_exact_children_cover_parent_and_keep_literal_three_component_bits():
    vertices=np.array(((.7,.2,.1),(.2,.7,.1),(.2,.2,.6)),np.longdouble)
    vertices[0,0]=np.nextafter(vertices[0,0],np.longdouble(1))
    original=literal_cell(vertices)
    children=split_cell(vertices)
    assert children[0][0]==original[0]
    assert sum(children[0][0])==sum(original[0])!=1
    assert all(signed_double_area(child)==signed_double_area(original)/4 for child in children)
    # Every child vertex is an original vertex or exact edge midpoint.
    allowed=set(original)|{tuple((a+b)/2 for a,b in zip(original[i],original[j])) for i,j in ((0,1),(1,2),(2,0))}
    assert all(v in allowed for child in children for v in child)


def test_work_cap_retains_crossing_and_identically_zero_events():
    source=raw();source['section']['normal_xyz']=(0.,0.,0.);source['section']['offset_m']=0.
    report=partition_events(source,[CELL],max_depth=2,max_cells=1)
    assert report['leaf_count']==1 and report['split_count']==0
    assert report['discarded_cell_count']==0 and report['unresolved_leaf_count']==1
    assert 'g' in report['leaves'][0]['identically_zero_events']
    assert report['exact_partition_of_supplied_cells']
    assert not report['source_physical_support_coverage_certified']
    assert not report['native_node_mapping_accuracy_certified']
    assert not report['angular_accuracy_certified']


def test_canonical_resealed_partition_and_bound_inflation_are_rejected():
    source=raw();events,_=source_events(source)
    cert=sign_certificate(events['Delta'],CELL)
    cert['upper']=['999999','1'];cert['certificate_sha256']=digest({k:v for k,v in cert.items() if k!='certificate_sha256'})
    with pytest.raises(ValueError,match='Canonical'):
        verify_sign_certificate(events['Delta'],CELL,cert)
    report=partition_events(source,[CELL],max_depth=1,max_cells=4)
    assert verify_partition(source,[CELL],report)
    forged=deepcopy(report);forged['leaves'].pop();forged['sha256']=digest({k:v for k,v in forged.items() if k!='sha256'})
    with pytest.raises(ValueError,match='Canonical'):
        verify_partition(source,[CELL],forged)


@pytest.mark.parametrize('center,radius,expected',[
    ((0.,0.,0.),100.,'POSITIVE_CERTIFIED_ON_SUPPLIED_CELL'),
    ((100.,100.,100.),.5,'EMPTY_CERTIFIED_ON_SUPPLIED_CELL'),
])
def test_convex_radial_branch_certificates_match_independent_exact_clipping(center,radius,expected):
    source=raw();source.update(seed_center_world_m=center,seed_radius_m=radius)
    cell=tuple(tuple(F(1,2) if i==j else F(1,4) for j in range(3)) for i in range(3))
    report=partition_events(source,[cell],max_depth=0)
    assert report['leaves'][0]['radial_support_classification']==expected
    for i in range(9):
        for j in range(9-i):
            barycentric=(F(8-i-j,8),F(i,8),F(j,8))
            N=tuple(sum(barycentric[k]*cell[k][axis] for k in range(3)) for axis in range(3))
            truth=_source_interval(source_line(source,N),256)
            assert (truth is None)==expected.startswith('EMPTY')


def test_inconclusive_coarse_A_bound_stays_open_until_subdivision_proves_it():
    source=raw();source['seed_radius_m']=100.
    coarse=partition_events(source,[CELL],max_depth=0)
    assert coarse['leaves'][0]['radial_support_classification']=='OPEN'
    refined=partition_events(source,[CELL],max_depth=1,max_cells=4)
    assert all(leaf['radial_support_classification']=='POSITIVE_CERTIFIED_ON_SUPPLIED_CELL'
               for leaf in refined['leaves'])


@pytest.mark.parametrize('cells,kwargs',[((),{}),((CELL,),dict(max_cells=0)),((CELL,),dict(max_depth=13)),((CELL[::-1],),{})])
def test_invalid_or_unbounded_partition_is_refused(cells,kwargs):
    with pytest.raises(ValueError):
        partition_events(raw(),cells,**kwargs)
