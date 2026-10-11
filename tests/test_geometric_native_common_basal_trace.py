"""Actual-source material association and independent position-jet controls."""
import copy
from dataclasses import replace
from fractions import Fraction as F
import json
import math
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

from tectonics.geometric_native_pose import FractionInterval as I
from tectonics.geometric_native_common_cold_depth_preparation import DeclaredColdDepthPreparation
from tectonics import geometric_native_common_basal_trace as api


def middle(value):
    return float((value.lo+value.hi)/2)


def diagnostic_preparation():
    return SimpleNamespace(declaration=DeclaredColdDepthPreparation(
        F(10),F(2),(F(1,100),F(-1,200),F(1,50)),"Independent finite Cartesian twist diagnostic"))


def diagnostic_columns():
    return ((I(1),I(F(1,3)),I(0)),(I(0),I(1),I(F(1,7))))


def independent_twist(c1,c2,lam):
    # Independent Rodrigues evaluation of the declared physical map.  It uses
    # no project rotation, interval, derivative or modal helper.
    x=(12+c1,7+c1/3+c2,3+c2/7)
    radius=math.sqrt(sum(v*v for v in x))
    tau=(1000-(radius/2)**3)/300
    omega=(.01,-.005,.02);om=math.sqrt(sum(v*v for v in omega));angle=lam*tau*om
    axis=tuple(v/om for v in omega);cross=(axis[1]*x[2]-axis[2]*x[1],axis[2]*x[0]-axis[0]*x[2],axis[0]*x[1]-axis[1]*x[0])
    dot=sum(a*b for a,b in zip(axis,x))
    return tuple(math.cos(angle)*x[i]+math.sin(angle)*cross[i]+(1-math.cos(angle))*axis[i]*dot for i in range(3))


def test_declared_cold_twist_all_first_derivatives_match_independent_map():
    x=(I(12),I(7),I(3));lam=F(2,5)
    position,first,second,receipt=api._prepared_cold_twojet(diagnostic_preparation(),x,diagnostic_columns(),lam,96)
    expected=independent_twist(0,0,float(lam))
    for i in range(3): assert middle(position[i])==pytest.approx(expected[i],rel=3e-15,abs=3e-15)
    h=1e-5
    for a in range(3):
        plus=[0.,0.,float(lam)];minus=list(plus);plus[a]+=h;minus[a]-=h
        yp,ym=independent_twist(*plus),independent_twist(*minus)
        for i in range(3): assert middle(first[a][i])==pytest.approx((yp[i]-ym[i])/(2*h),rel=3e-7,abs=3e-10)
    assert receipt['nonlinear_body_body_curvature_included']
    assert receipt['mixed_body_preparation_curvature_included']


def test_declared_cold_twist_full_mixed_second_derivatives_match_independent_map():
    _,_,second,_=api._prepared_cold_twojet(diagnostic_preparation(),(I(12),I(7),I(3)),diagnostic_columns(),F(2,5),96)
    h=2e-3;centre=[0.,0.,.4];yc=independent_twist(*centre)
    for a in range(3):
        for b in range(a,3):
            if a==b:
                plus=list(centre);minus=list(centre);plus[a]+=h;minus[a]-=h
                yp,ym=independent_twist(*plus),independent_twist(*minus)
                control=tuple((yp[i]-2*yc[i]+ym[i])/(h*h) for i in range(3))
            else:
                vals=[]
                for sa,sb in ((1,1),(1,-1),(-1,1),(-1,-1)):
                    c=list(centre);c[a]+=sa*h;c[b]+=sb*h;vals.append(independent_twist(*c))
                control=tuple((vals[0][i]-vals[1][i]-vals[2][i]+vals[3][i])/(4*h*h) for i in range(3))
            for i in range(3): assert middle(second[a][b][i])==pytest.approx(control[i],rel=8e-5,abs=5e-9)


def test_zero_preparation_has_exact_zero_modal_curvature_and_true_cross_lambda():
    x=(I(12),I(7),I(3));columns=diagnostic_columns()
    position,first,second,_=api._prepared_cold_twojet(diagnostic_preparation(),x,columns,F(0),96)
    for i in range(3): assert position[i].lo<=x[i].lo<=position[i].hi
    for a in range(2):
        for b in range(2): assert second[a][b]==(I(0),)*3
    assert any(v!=I(0) for v in second[0][2])
    assert any(v!=I(0) for v in second[2][2])


def test_owner_lift_contains_exact_spin_and_body_mixed_curvature():
    x=(I(2),I(3),I(5));f=((I(7),I(11),I(13)),);H=(((I(17),I(19),I(23)),),)
    ids,J,Q=api._owner_lift(x,(36,),f,H,1,96)
    assert ids==(3,4,5,36)
    expected=((I(0),I(5),I(-3),I(7)),(I(-5),I(0),I(2),I(11)),(I(3),I(-2),I(0),I(13)))
    assert J==expected
    for axis in Q:
        for i in range(4):
            for j in range(4): assert axis[i][j]==axis[j][i]
    assert tuple(axis[2][3] for axis in Q)==(I(-11),I(7),I(0))
    assert tuple(axis[3][3] for axis in Q)==(I(17),I(19),I(23))


def test_owner_lift_intersects_equivalent_outward_curvature_enclosures():
    H=(((I(0),)*3,(I(1,3),)*3),((I(2,4),)*3,(I(0),)*3))
    ids,J,Q=api._owner_lift((I(1),I(0),I(0)),(36,37),((I(1),I(0),I(0)),(I(0),I(1),I(0))),H,0,96)
    a,b=ids.index(36),ids.index(37)
    assert all(axis[a][b]==axis[b][a]==I(2,3) for axis in Q)


def independent_rotation(vector,axis,t=1.):
    norm=math.sqrt(sum(v*v for v in axis))
    if norm==0: return tuple(vector)
    direction=tuple(v/norm for v in axis);angle=t*norm
    cross=(direction[1]*vector[2]-direction[2]*vector[1],direction[2]*vector[0]-direction[0]*vector[2],direction[0]*vector[1]-direction[1]*vector[0])
    dot=sum(x*y for x,y in zip(direction,vector))
    return tuple(math.cos(angle)*vector[i]+math.sin(angle)*cross[i]+(1-math.cos(angle))*direction[i]*dot for i in range(3))


def independent_gap(ray,donor,receiver,time):
    x=independent_rotation(ray,donor,time);normal=independent_rotation((1.,0.,0.),receiver,time)
    return -sum(a*b for a,b in zip(normal,x))


def test_exact_all_clock_majorant_proves_true_nonentering_rotated_material():
    ray=(F(1),F(1,2),F(1,3));donor=(F(1,100),F(1,200),F(1,100));receiver=(F(0),F(1,200),F(0))
    proof=api._no_entry_gap_certificate(ray,donor,receiver)
    assert proof['method']=='EXACT_RELATIVE_ROTATION_L1_MAJORANT'
    assert proof['whole_clock_gap_upper']<0 and proof['endpoint_only_proof'] is False
    for time in (0.,.125,.5,.875,1.):
        value=independent_gap(tuple(map(float,ray)),tuple(map(float,donor)),tuple(map(float,receiver)),time)
        assert value<=float(proof['whole_clock_gap_upper'])+1e-14


def test_exact_bernstein_proves_thin_live_ray_that_simple_majorant_cannot_resolve():
    ray=(F(1,10**6),F(-1),F(1,5));donor=(F(0),F(0),F(1,100));receiver=(F(0),)*3
    proof=api._no_entry_gap_certificate(ray,donor,receiver)
    assert proof['method']=='EXACT_RATIONAL_TAYLOR_BERNSTEIN_WITH_PRODUCT_NORM_TAIL'
    assert proof['cells'][0]['clock_fraction_bounds'].lo==0
    assert proof['cells'][-1]['clock_fraction_bounds'].hi==1
    assert all(cell['whole_cell_gap_upper']<0 for cell in proof['cells'])
    assert proof['product_exponential_tail_upper']>0
    assert proof['historical_archive_activation_beta_identically_zero']
    for t in (0.,.1,.3,.7,1.): assert independent_gap(tuple(map(float,ray)),tuple(map(float,donor)),(0.,0.,0.),t)<0


def test_all_clock_proof_rejects_actual_entry_instead_of_loading_archived_material():
    assert api._no_entry_gap_certificate((F(1,10**6),F(1),F(0)),(F(0),F(0),F(1,100)),(F(0),)*3) is None


def test_negative_both_endpoints_do_not_authorize_an_interior_entry_path():
    ray=(F(1,10**7),F(-1),F(-1,10));donor=(F(1,10),F(1,100),F(0));receiver=(F(0),F(9,1000),F(0))
    numeric=tuple(tuple(map(float,v)) for v in (ray,donor,receiver))
    assert independent_gap(*numeric,0.)<0 and independent_gap(*numeric,1.)<0
    assert independent_gap(*numeric,.125)>0
    assert api._no_entry_gap_certificate(ray,donor,receiver) is None


def test_all_clock_receipt_rejects_axis_bound_and_kind_corruption():
    ray=(F(1,10**6),F(-1),F(1,5));donor=(F(0),F(0),F(1,100));receiver=(F(0),)*3
    proof=api._no_entry_gap_certificate(ray,donor,receiver)
    assert api._verify_no_entry_gap_certificate(proof,original_ray=ray,donor_axis=donor,receiver_axis=receiver)==proof
    for name,value in (('strict_whole_clock_gap_negative',1),('historical_archive_activation_beta_identically_zero',False),
            ('product_exponential_tail_upper',F(0)),('donor_axis',(F(0),F(0),F(1,200)))):
        bad=copy.deepcopy(proof);bad[name]=value
        with pytest.raises(ValueError,match='changed all-clock'):
            api._verify_no_entry_gap_certificate(bad,original_ray=ray,donor_axis=donor,receiver_axis=receiver)


def test_persisted_strict_gap_and_tail_survive_values_below_binary64_range():
    ray=(F(1,10**1000),F(1),F(-1));zero=(F(0),)*3
    proof=api._no_entry_gap_certificate(ray,zero,zero)
    assert -ray[0]<=proof['whole_clock_gap_upper']<0
    thin=(F(1,10**300),F(-1),F(0));axis=(F(0),F(0),F(1,10**100))
    b=api._no_entry_gap_certificate(thin,axis,zero)
    assert b['method']=='EXACT_RATIONAL_TAYLOR_BERNSTEIN_WITH_PRODUCT_NORM_TAIL'
    assert 0<b['product_exponential_tail_upper']<F(1,10**2000)
    assert all(c['whole_cell_gap_upper']<0 for c in b['cells'])
    with pytest.raises(ValueError,match='persisted upper bound'):
        api._strict_negative_upper(F(0),96)


@pytest.mark.parametrize('owner',[0,1])
def test_closed_live_source_pose_keeps_full_vector_history_and_true_parent_bottom(owner):
    ray=(F(2),F(3),F(5));axis=(F(1,100),F(-1,200),F(1,50));bank=SimpleNamespace(
        original_parent_id='original-parent',original_material_id='original-material',owner=owner,areal_material_thickness_m=F(1))
    state=SimpleNamespace(state_sha256='source-state');archive=SimpleNamespace(model_sha256='archive-model',radius_m=F(10),
        donor_owner=1,receiver_owner=0,donor_axis=axis,receiver_axis=axis,
        whole_domain_certificate=SimpleNamespace(original_parent_thicknesses_m=(('original-parent',F(1)),)))
    proof=dict(material_state_sha256='source-state',archive_model_sha256='archive-model',original_parent_id='original-parent',
        original_ray=ray,all_clock_live_material_allocation_proved=True,historical_archive_activation_beta_identically_zero=True)
    point=api._closed_live_source_point(state,archive,bank,ray,proof,96)
    norm=math.sqrt(38);expected=independent_rotation(tuple(float(x)/norm for x in ray),tuple(map(float,axis)))
    radius=700**(1/3) if owner==1 else 10.
    for i in range(3): assert middle(point.position_bounds_m[i])==pytest.approx(radius*expected[i],rel=3e-15,abs=3e-15)
    assert point.material_depth_bounds_m==I(1) and point.beta_bounds==I(0) and point.archive_debits==0
    if owner==0: assert point.radial_support_bounds_m==I(10)
    bad=dict(proof,historical_archive_activation_beta_identically_zero=False)
    with pytest.raises(ValueError,match='both genuine'):
        api._closed_live_source_point(state,archive,bank,ray,bad,96)


@pytest.mark.parametrize('precision',[True,95,513,F(96)])
def test_trace_precision_rejects_kind_alias_and_outside_budget(precision):
    with pytest.raises(ValueError,match='literal integer'):
        api._precision(precision)


@pytest.fixture(scope='module')
def context():
    path=Path(__file__).resolve().parents[1]/'analysis/geometric_native_common_basal_loading_validation/native_control.txt'
    return runpy.run_path(str(path))['context']()


@pytest.fixture(scope='module')
def actual(context):
    return api.evaluate_native_common_basal_trace(context['trace_system'],precision=96)


def test_all68_actual_positive_material_cell_weights_retained_once(context,actual):
    system=context['trace_system'];source=json.loads(system.complete_source903_load_json)
    assert actual.positive_source_material_cell_row_count==68
    assert actual.original_live_material_trace_count==59
    assert actual.hot_birth_material_trace_count==9
    assert system.paired_correction_metadata_count==1
    assert actual.all_positive_source_cell_weights_used_once
    assert not actual.paired_correction_added_again
    assert not actual.aggregate_owner6_load_reused
    records=api.native_common_basal_trace_loading_context(system,actual)['trace_records']
    assert tuple(r['source_material_cell_index'] for r in records)==tuple(range(68))
    for i,record in enumerate(records):
        assert record['source_weight_bounds_m2']==api._interval(source['material_cell_measures'][i]['area_bounds_m2'])
        assert record['source_weight_bounds_m2'].lo>0
        assert record['point_is_material_cell_centroid'] is False
        assert min(record['source_intersection_strict_support_signs'])==1
        assert record['source_reference_area_used_once'] is True
    assert min(r['source_weight_bounds_m2'].lo for r in records)<F(1,10**40)


def test_actual_cold_points_are_live_at_authentic_bottoms_and_hot_points_have_true_source_proxy(context,actual):
    rows=api.native_common_basal_trace_loading_context(context['trace_system'],actual)['trace_records']
    for r in rows:
        if r['material_kind']=='original_live':
            assert r['material_depth_m']==F(10000)
            assert r['endpoint_allocation']=='AUTHENTIC_ENDPOINT_LIVE_ORIGINAL_MATERIAL'
            saved=json.loads(r['geometry_receipt']['source_archive_point_json'])['response']
            assert saved['allocation']=='live' and saved['archive_debits']==0
            assert r['geometry_receipt']['actual_seam_blended_cold56_fields'] is True
        else:
            assert r['material_depth_m']==F(2000)
            assert r['endpoint_allocation']=='AUTHENTIC_HOT_SOURCE_PROXY_INTERIOR'
            assert r['geometry_receipt']['additional_historical_owner_pose_applied'] is False
    assert actual.archived_material_receives_basal_loading is False


def test_actual_body86_sampling_includes_genuine_active_cold_strip_and_hot_fields(actual):
    assert set(actual.active_cold_coordinate_indices)==set(range(36,92))
    assert set(actual.active_hot_coordinate_indices)==set(range(6,36))
    assert actual.internal_contact_arm_basal_derivatives_exactly_zero


def test_actual_position_jets_have_sparse_global101_indices_and_symmetric_curvature(context,actual):
    for r in api.native_common_basal_trace_loading_context(context['trace_system'],actual)['trace_records']:
        ids=r['global_coordinate_indices'];N=len(ids)
        assert tuple(sorted(ids))==ids and len(set(ids))==N
        assert all(0<=i<=100 and not 92<=i<100 for i in ids)
        assert len(r['position_first_bounds'])==3 and all(len(v)==N for v in r['position_first_bounds'])
        assert len(r['position_second_bounds'])==3
        for matrix in r['position_second_bounds']:
            assert len(matrix)==N and all(len(v)==N for v in matrix)
            assert all(matrix[i][j]==matrix[j][i] for i in range(N) for j in range(N))
        if r['material_kind']=='original_live':
            assert 100 in ids


def test_actual_complete_source_system_and_exact_query_restart(context,actual):
    raw=api.native_common_basal_trace_to_dict(context['trace_system'])
    restored=api.native_common_basal_trace_from_dict(copy.deepcopy(raw),context['source903'],context['source895'],
        context['source907'],cold_carrier=context['cold_carrier'],hot_system=context['hot_system'],preparations=context['preparations'])
    assert api.native_common_basal_trace_to_dict(restored)==raw
    dto=api.native_common_basal_trace_response_to_dict(actual)
    response=api.native_common_basal_trace_response_from_dict(copy.deepcopy(dto),system=restored,
        expected_coordinates101=(F(0),)*101,precision=96)
    assert api.native_common_basal_trace_response_to_dict(response)==dto


def test_actual_restarts_reject_resealed_witness_measure_and_false_scope(context,actual):
    dto=api.native_common_basal_trace_response_to_dict(actual)
    dto['continuum_basal_quadrature_certified']=True
    with pytest.raises(ValueError,match='exact expected query'):
        api.native_common_basal_trace_response_from_dict(dto,system=context['trace_system'])
    bad=replace(actual,binding_sha256='0'*64)
    with pytest.raises(ValueError,match='registered source'):
        api.native_common_basal_trace_loading_context(context['trace_system'],bad)
    raw=api.native_common_basal_trace_to_dict(context['trace_system'])
    w=json.loads(raw['source_row_witnesses_json'][0]);w['source_cell']=19
    raw['source_row_witnesses_json'][0]=json.dumps(w)
    with pytest.raises(ValueError,match='restart changed'):
        api.native_common_basal_trace_from_dict(raw,context['source903'],context['source895'],context['source907'],
            cold_carrier=context['cold_carrier'],hot_system=context['hot_system'],preparations=context['preparations'])


def test_actual_coordinates_fail_closed_without_zeroing_missing_or_unproved_fields(context):
    with pytest.raises(ValueError,match='101 literal Fraction'):
        api.evaluate_native_common_basal_trace(context['trace_system'],(F(0),)*100)
    owner=[F(0)]*101;owner[2]=F(1,1000)
    with pytest.raises(ValueError,match='restricted to zero rotations'):
        api.evaluate_native_common_basal_trace(context['trace_system'],tuple(owner))
    invalid=[F(0)]*101;invalid[100]=F(2)
    with pytest.raises(ValueError,match='restricted to'):
        api.evaluate_native_common_basal_trace(context['trace_system'],tuple(invalid))
    assert context['trace_system'].continuum_basal_quadrature_certified is False
    assert context['trace_system'].complete_actual_bottom_support_certified is False
    assert context['trace_system'].full_general_state_mixed_operator_certified is False
    assert context['trace_system'].original_point_1_completed is False
