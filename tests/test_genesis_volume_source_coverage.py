from fractions import Fraction as F
import copy
import numpy as np
import pytest
from tectonics.genesis_volume_source_coverage import source_coverage,verify_coverage,exact_node
from tectonics.genesis_volume_event_partition import signed_double_area,split_cell
from tectonics.genesis_volume_mapped_integrals import PreparedMappedIntegrals
from tectonics.genesis_volume_rational_fields import fraction
from analysis.genesis_volume_operand_validation.independent_coefficients import exact_source_coefficients
from analysis.genesis_volume_mapped_validation.independent import independent_targets
from test_genesis_volume_precision_integrals import fixture


def simple(X=None):
    return dict(section=dict(triangle_m=np.eye(3) if X is None else X,
        displacement_A_m=np.zeros((3,3)),slope_P_m=np.zeros((3,3)),lambda_bounds=(.5,1.)),
        seed_center_world_m=(.2,.3,.4),seed_radius_m=.04)


@pytest.mark.parametrize('sign',[1,-1])
def test_cramer_bounds_cover_sphere_points_with_both_determinant_signs(sign):
    raw=simple(np.diag([sign,1,1]));raw['seed_center_world_m']=(sign*.2,.3,.4)
    cells,cert=source_coverage(raw)
    assert verify_coverage(raw,cert)
    bounds=[tuple(F(*map(int,v)) for v in b) for b in cert['coordinate_bounds']]
    # Independent direct diagonal inversion for every lambda, ball displacement.
    for lam in (F(1,2),F(3,4),F(1)):
        for axis in range(3):
            for delta in (-F(1,50),F(),F(1,50)):
                point=list(map(fraction,raw['seed_center_world_m']));point[axis]+=delta
                N=[point[0]/(sign*lam),point[1]/lam,point[2]/lam]
                for q,(lo,hi) in zip(N,bounds):
                    if 0<=q<=1:
                        assert lo<=q<=hi
    assert all(sum(row)==1 for cell in cells for row in cell)
    assert all(signed_double_area(c)>0 for c in cells)


def test_singular_slab_keeps_full_simplex_and_forged_proof_is_rejected():
    cells,cert=source_coverage(simple(np.zeros((3,3))))
    assert cert['singular_interval_retains_full_simplex']
    assert len(cells)==1 and signed_double_area(cells[0])==1
    altered=copy.deepcopy(cert);altered['coordinate_bounds'][0][1]=['0','1']
    with pytest.raises(ValueError):verify_coverage(simple(np.zeros((3,3))),altered)


def test_exact_map_has_zero_mapping_residual_and_split_has_no_gap():
    cell=((F(1),F(),F()),(F(),F(1),F()),(F(),F(),F(1)))
    s,t=F(1,7),F(1,11)
    n=exact_node(cell,s,t)
    assert n==(1-s,s*(1-t),s*t) and sum(n)==1
    assert any(fraction(np.longdouble(float(q)))!=q for q in n)
    assert sum(signed_double_area(c) for c in split_cell(cell))==1
    with pytest.raises(ValueError):exact_node(cell,F(2),t)


@pytest.mark.parametrize('radius',[100.,2.])
def test_rational_node_integrals_enclose_separate_discriminant_contraction(radius):
    raw,source,options,_=fixture(radius)
    before=(source.parent_stiffness.tobytes(),source.parent_mass.tobytes())
    n=(F(2,7),F(1,5),F(18,35))
    runtime=PreparedMappedIntegrals(source,raw,**options)
    line=runtime.evaluate(n)
    chart=line.interval
    assert chart is not None and chart.barycentric==n
    fields=runtime.fields.evaluate(chart)
    oracle=exact_source_coefficients(raw,n,fraction(chart.lambda_mid),fraction(chart.lambda_halfwidth))
    assert all(np.array_equal(fields[k],oracle[k]) for k in fields)
    targets=independent_targets(raw,n,chart,**options)
    assert tuple(targets)==line.names
    for name,lo,hi,v,e in zip(line.names,line.enclosure.lower,line.enclosure.upper,line.values,line.errors):
        a,b=targets[name]
        assert fraction(lo)<=a<=b<=fraction(hi),name
        assert max(abs(fraction(v)-a),abs(fraction(v)-b))<=fraction(e),name
    assert before==(source.parent_stiffness.tobytes(),source.parent_mass.tobytes())
    assert not line.angular_accuracy_certified


def test_reject_nonunit_exact_node_instead_of_normalizing_it():
    raw,source,options,_=fixture()
    runtime=PreparedMappedIntegrals(source,raw,**options)
    with pytest.raises(ValueError,match='simplex'):
        runtime.evaluate((F(1,3),F(1,3),F(1,3)+F(1,2**90)))


def angular_fixture(radius=100.):
    raw=simple();raw['seed_radius_m']=radius
    raw['section'].update(normal_xyz=(0.,0.,1.),offset_m=-1.)
    return raw


def test_event_integrator_exact_map_constant_measure_and_strict_tolerance():
    from analysis.genesis_volume_mapped_validation.angular import integrate_angular,AngularControlRefusal
    seen=[]
    def evaluator(n):
        assert sum(n)==1 and all(isinstance(q,F) for q in n)
        seen.append(n)
        return np.array([0.,1.,1.],np.longdouble),np.zeros(3,np.longdouble)
    names=('reference_volume_negative_m3','reference_volume_positive_m3','constant')
    report=integrate_angular(angular_fixture(),evaluator,names)
    assert seen and report['initial_source_support_coverage_certified']
    assert report['angular_node_mapping_accuracy_certified']
    assert abs(report['fine_values'][2]-.5)<1e-14
    assert report['failed_component_count']==0
    assert not report['angular_accuracy_certified']
    with pytest.raises(AngularControlRefusal,match='2e-6'):
        integrate_angular(angular_fixture(),evaluator,names,relative_tolerance=3e-6)


def test_unsampled_positive_support_is_retained_as_open():
    from analysis.genesis_volume_mapped_validation.angular import integrate_angular
    names=('reference_volume_negative_m3','reference_volume_positive_m3')
    report=integrate_angular(angular_fixture(),lambda n:(np.zeros(2),np.zeros(2)),names,max_line_evaluations=544)
    assert report['status']=='OPEN'
    assert report['unresolved_positive_candidate_cells']>0
    assert report['original_own_relative_component_allowances']==[0.,0.]
