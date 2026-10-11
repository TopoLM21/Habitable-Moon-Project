"""Independent spatial/state diagnostics and authenticated basal restart gates."""
import copy
from dataclasses import replace
from fractions import Fraction as F
import math
from pathlib import Path
import runpy

import pytest

from tectonics.geometric_native_pose import FractionInterval as I
from tectonics import geometric_native_common_basal_loading as api


def source():
    return dict(radius_m=F(7),basal_drag_pa_s_m=F(2),
        angular_velocity_rad_s=(F(0),F(0),F(3,5)),
        amplitude_constant=F(1),amplitude_gradient=(F(1,5),F(0),F(0)))


def record():
    return dict(position_bounds_m=(I(3),I(4),I(0)),global_coordinate_indices=(0,1),
        position_first_bounds=((I(1),I(0)),(I(0),I(1)),(I(0),I(0))),
        position_second_bounds=(((I(0),I(0)),(I(0),I(0))),)*3,
        source_weight_bounds_m2=I(3))


def encloses(bound,value):
    return bound.lo<=value<=bound.hi


def middle(bound):
    return float((bound.lo+bound.hi)/2)


def test_literal_vector_driving_and_unscaled_isotropic_drag():
    s=source();r=record();rates=(F(1,11),F(-1,13))
    response=api._trace_algebra(s,(r,),rates,size=2)
    traction=(F(-4704,625),F(3528,625))
    for i in range(2):
        assert encloses(response['prescribed_generalized_force_bounds'][i],3*traction[i])
        assert encloses(response['residual_bounds'][i],3*traction[i]-6*rates[i])
        for j in range(2):
            assert response['drag_bounds'][i][j]==I(6*int(i==j))
            assert response['velocity_tangent_bounds'][i][j]==I(-6*int(i==j))
    doubled=dict(s,amplitude_constant=F(2),amplitude_gradient=tuple(2*x for x in s['amplitude_gradient']))
    other=api._trace_algebra(doubled,(r,),rates,size=2)
    assert response['drag_bounds']==other['drag_bounds']
    for i in range(2):
        assert encloses(other['prescribed_generalized_force_bounds'][i],6*traction[i])


def test_actual_velocity_power_matches_independent_exact_contractions():
    rates=(F(1,11),F(-1,13));out=api._trace_algebra(source(),(record(),),rates,size=2)
    ps=3*(F(-4704,625)*rates[0]+F(3528,625)*rates[1])
    pd=6*(rates[0]**2+rates[1]**2)
    assert encloses(out['source_power_bounds_w'],ps)
    assert encloses(out['drag_dissipation_bounds_w'],pd)
    assert encloses(out['net_power_bounds_w'],ps-pd)
    assert out['drag_dissipation_bounds_w'].lo>0


def test_angular_extension_uses_fixed_source_radius_and_current_global_direction():
    s=source();x=(I(3),I(4),I(0));scaled=tuple(2*v for v in x)
    first,G,_=api._spatial_source_twojet(s,x)
    other,Gother,_=api._spatial_source_twojet(s,scaled)
    for i in range(3):
        assert api._intersection(first[i],other[i])
        for j in range(3):
            assert api._intersection(G[i][j],2*Gother[i][j])
    changed=dict(s,radius_m=F(14));doubled,_,_=api._spatial_source_twojet(changed,x)
    for i in range(3):
        assert api._intersection(doubled[i],2*first[i])


def test_full_vector_radial_velocity_dissipates_without_tangent_projection():
    r=record();r['global_coordinate_indices']=(0,)
    r['position_first_bounds']=((I(F(3,5)),),(I(F(4,5)),),(I(0),))
    r['position_second_bounds']=(((I(0),),),)*3
    out=api._trace_algebra(source(),(r,),(F(2),),size=1)
    assert out['drag_bounds']==((I(6),),)
    assert encloses(out['source_power_bounds_w'],F(0))
    assert out['drag_dissipation_bounds_w']==I(24)
    assert encloses(out['net_power_bounds_w'],F(-24))


def nonlinear_record(s,t):
    return dict(position_bounds_m=(I(3+s+s*t),I(4+t+s*s/2),I(2+s*t*t)),
        global_coordinate_indices=(0,1),
        position_first_bounds=((I(1+t),I(s)),(I(s),I(1)),(I(t*t),I(2*s*t))),
        position_second_bounds=(((I(0),I(1)),(I(1),I(0))),
            ((I(1),I(0)),(I(0),I(0))),((I(0),I(2*t)),(I(2*t),I(2*s)))),
        source_weight_bounds_m2=I(3))


def independent_nonlinear_residual(s,t,rates):
    # Independently evaluate the physical point, velocity, traction and local
    # virtual forces. This diagnostic uses no derivative formula from the core.
    y=(3+s+s*t,4+t+s*s/2,2+s*t*t)
    J=((1+t,s),(s,1),(t*t,2*s*t))
    radius=math.sqrt(sum(x*x for x in y));q=tuple(x/radius for x in y)
    traction=tuple(14*(1+q[0]/5)*x for x in (-3*q[1]/5,3*q[0]/5,0))
    velocity=tuple(sum(J[a][i]*rates[i] for i in range(2)) for a in range(3))
    return tuple(3*sum(J[a][i]*(traction[a]-2*velocity[a]) for a in range(3)) for i in range(2))


def test_state_tangent_contains_position_curvature_and_drag_state_terms():
    s,t=F(1,5),F(-1,7);rates=(F(3,2),F(-7,4))
    out=api._trace_algebra(source(),(nonlinear_record(s,t),),rates,size=2)
    h=1e-6;numericrates=tuple(float(v) for v in rates)
    for k in range(2):
        plus=[float(s),float(t)];minus=list(plus);plus[k]+=h;minus[k]-=h
        qp=independent_nonlinear_residual(*plus,numericrates)
        qm=independent_nonlinear_residual(*minus,numericrates)
        for i in range(2):
            assert middle(out['state_tangent_bounds'][i][k])==pytest.approx((qp[i]-qm[i])/(2*h),rel=2e-8,abs=2e-7)


def test_spatial_force_tangent_retains_nonconservative_antisymmetric_part():
    out=api._trace_algebra(source(),(record(),),(F(0),F(0)),size=2)
    H=out['state_tangent_bounds']
    assert H[0][1].hi < H[1][0].lo
    eps=1e-6;s=source();r=record()
    for k in range(2):
        x=list(map(float,(3,4,0)));xp=list(x);xm=list(x);xp[k]+=eps;xm[k]-=eps
        def physical(point):
            norm=math.sqrt(sum(v*v for v in point));q=[v/norm for v in point]
            return [14*(1+q[0]/5)*(-3*q[1]/5),14*(1+q[0]/5)*(3*q[0]/5)]
        fp,fm=physical(xp),physical(xm)
        for i in range(2):
            assert middle(H[i][k])==pytest.approx(3*(fp[i]-fm[i])/(2*eps),rel=1e-8,abs=2e-8)


def test_damping_jacobian_is_exact_rate_derivative_at_fixed_state():
    rates=(F(3,2),F(-7,4));r=nonlinear_record(F(1,5),F(-1,7))
    out=api._trace_algebra(source(),(r,),rates,size=2)
    for j in range(2):
        delta=F(2,3);shift=list(rates);shift[j]+=delta
        changed=api._trace_algebra(source(),(r,),tuple(shift),size=2)
        for i in range(2):
            difference=(changed['residual_bounds'][i]-out['residual_bounds'][i])/delta
            assert api._intersection(difference,out['velocity_tangent_bounds'][i][j])


def test_positive_interval_measure_preserves_same_field_gram_dissipation():
    r=nonlinear_record(F(1,5),F(-1,7));r['source_weight_bounds_m2']=I(3,4)
    out=api._trace_algebra(source(),(r,),(F(3,2),F(-7,4)),size=2)
    assert out['drag_bounds'][0][1]==out['drag_bounds'][1][0]
    assert out['drag_dissipation_bounds_w'].lo>0
    assert len(out['per_trace_receipts_json'])==1


def test_sparse_global_catalogue_injection_keeps_crossblocks_under_permutation():
    r=record();r['global_coordinate_indices']=(4,1);rates=(F(0),F(2),F(0),F(0),F(3))
    out=api._trace_algebra(source(),(r,),rates,size=5)
    perm=copy.deepcopy(r);perm['global_coordinate_indices']=(1,4)
    perm['position_first_bounds']=tuple(tuple(reversed(row)) for row in r['position_first_bounds'])
    again=api._trace_algebra(source(),(perm,),rates,size=5)
    for key in ('drag_bounds','residual_bounds','state_tangent_bounds','velocity_tangent_bounds'):
        assert out[key]==again[key]
    for index in (0,2,3):
        assert out['residual_bounds'][index]==I(0)
        assert out['state_tangent_bounds'][index]==(I(0),)*5


@pytest.mark.parametrize('bad',[None,(F(0),),(0,0),(F(0),0),[F(0),F(0)]])
def test_missing_implicit_or_nonliteral_rates_are_rejected(bad):
    with pytest.raises(ValueError,match='physical coordinate rate'):
        api._trace_algebra(source(),(record(),),bad,size=2)


@pytest.mark.parametrize('field,value',[('source_weight_bounds_m2',I(0,1)),
    ('global_coordinate_indices',(0,0)),('position_second_bounds',None),
    ('position_bounds_m',(I(0),I(0),I(0)))])
def test_invalid_trace_measure_catalogue_curvature_or_origin_is_rejected(field,value):
    r=record();r[field]=value
    with pytest.raises(ValueError):
        api._trace_algebra(source(),(r,),(F(0),F(0)),size=2)


@pytest.fixture(scope='module')
def context():
    root=Path(__file__).resolve().parents[1]
    return runpy.run_path(str(root/'analysis/geometric_native_common_basal_loading_validation/native_control.txt'))['context']()


@pytest.fixture(scope='module')
def trace_response(context):
    from tectonics import geometric_native_common_basal_trace as trace
    return trace.evaluate_native_common_basal_trace(context['trace_system'],(F(0),)*101,precision=96)


@pytest.fixture(scope='module')
def physical_rates():
    rates=[F(0)]*101
    rates[2]=F(1,10**12);rates[5]=F(-1,2*10**12)
    rates[6]=F(1,10**9);rates[36]=F(-1,10**9);rates[100]=F(1,10**6)
    return tuple(rates)


@pytest.fixture(scope='module')
def response(context,trace_response,physical_rates):
    return api.evaluate_native_common_basal_loading(context['loading_system'],trace_response,
        coordinate_rates=physical_rates,precision=96)


def test_authenticated_complete_catalogue_source_and_scope(context,response):
    system=context['loading_system']
    assert len(system.coordinate_order)==101
    assert system.coordinate_units==('rad',)*6+('m',)*94+('1',)
    assert system.basal_drag_pa_s_m==F(120000000000000)
    assert system.mantle_angular_velocity_rad_s==(F(0),F(0),F.from_float(1.1e-12))
    assert system.prescribed_amplitude_gradient==(F.from_float(0.17),F(0),F(0))
    assert response.complete_declared_discrete_basal_operator_computed is True
    assert response.continuum_whole_base_quadrature_certified is False
    assert response.general_state_full_physics_operator_certified is False
    assert response.new_mixed_force_equilibrium_certified is False
    assert response.full_system_temporal_path_work_certified is False
    assert response.clocks_advanced is False
    assert response.owner_query_zero_current_derivatives_only is True


def test_authenticated_response_retains_full_matrices_and_independent_power(response):
    assert len(response.prescribed_generalized_force_bounds)==101
    for matrix in (response.drag_bounds,response.state_tangent_bounds,response.velocity_tangent_bounds):
        assert len(matrix)==101 and all(len(row)==101 for row in matrix)
    assert response.drag_dissipation_bounds_w.lo>=0
    assert api._intersection(response.source_power_bounds_w,response.generalized_source_power_bounds_w)
    assert api._intersection(response.drag_dissipation_bounds_w,response.independently_projected_drag_dissipation_bounds_w)
    assert api._intersection(response.net_power_bounds_w,response.source_power_bounds_w-response.drag_dissipation_bounds_w)


def test_authenticated_internal_arm_only_zero_columns_have_explicit_source_law(response):
    for i in range(92,100):
        assert response.prescribed_generalized_force_bounds[i]==I(0)
        assert response.residual_bounds[i]==I(0)
        assert response.drag_bounds[i]==(I(0),)*101
        assert response.state_tangent_bounds[i]==(I(0),)*101
        assert all(row[i]==I(0) for row in response.drag_bounds)
    assert response.internal_contact_arm_only_basal_columns_exact_zero is True


def test_authenticated_complete_system_and_response_restart(context,trace_response,physical_rates,response):
    system=context['loading_system'];raw=api.native_common_basal_loading_to_dict(system)
    restarted=api.native_common_basal_loading_from_dict(copy.deepcopy(raw),context['trace_system'])
    assert api.native_common_basal_loading_to_dict(restarted)==raw
    dto=api.native_common_basal_loading_response_to_dict(response)
    restored=api.native_common_basal_loading_response_from_dict(copy.deepcopy(dto),system=restarted,
        trace_evaluation=trace_response,expected_coordinate_rates=physical_rates,precision=96)
    assert api.native_common_basal_loading_response_to_dict(restored)==dto


def test_authenticated_restart_rejects_claim_changes_and_rate_substitution(context,trace_response,physical_rates,response):
    raw=api.native_common_basal_loading_to_dict(context['loading_system'])
    raw['continuum_whole_base_quadrature_certified']=True
    with pytest.raises(ValueError,match='restart changed'):
        api.native_common_basal_loading_from_dict(raw,context['trace_system'])
    dto=api.native_common_basal_loading_response_to_dict(response)
    badrates=list(physical_rates);badrates[100]=F(0)
    with pytest.raises(ValueError,match='exact expected query and rates'):
        api.native_common_basal_loading_response_from_dict(dto,system=context['loading_system'],
            trace_evaluation=trace_response,expected_coordinate_rates=tuple(badrates),precision=96)


def test_unregistered_or_resealed_source_and_trace_are_rejected(context,trace_response,physical_rates):
    unregistered=replace(context['loading_system'])
    with pytest.raises(ValueError,match='unregistered'):
        api.evaluate_native_common_basal_loading(unregistered,trace_response,coordinate_rates=physical_rates)
    badtrace=replace(trace_response,binding_sha256='0'*64)
    with pytest.raises(ValueError):
        api.evaluate_native_common_basal_loading(context['loading_system'],badtrace,coordinate_rates=physical_rates)
    with pytest.raises(ValueError,match='physical coordinate rate'):
        api.evaluate_native_common_basal_loading(context['loading_system'],trace_response,coordinate_rates=(F(0),)*100)
