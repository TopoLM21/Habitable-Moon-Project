"""Whole cold T(lambda) work, prestress, signed power and source authority."""
from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path
import runpy

import pytest

from tectonics.geometric_native_pose import FractionInterval as I
from tectonics import geometric_native_common_cold_preparation_energy as api
from tectonics.geometric_native_common_cold_variational_work import _saved


def overlap(a,b):
    a,b=api._bounds(a),api._bounds(b)
    assert max(a.lo,b.lo)<=min(a.hi,b.hi)


def synthetic_coefficients(forces=(-3,4)):
    z=F(0)
    A=tuple(tuple(F(2 if i==j else 1) if i<2 and j<2 else F(int(i==j))
        for j in range(8)) for i in range(8))
    f=tuple(F(x) for x in forces)+(z,)*6
    K=(F(5),F(7))+(F(1),)*6
    r1=(F(1),F(2))+(z,)*6; r2=(F(3),F(-1))+(z,)*6
    result=api._coefficients(A,F(1000000),f,K,r1,r2,F(20),z,F(30),96)
    return result,A,f,K,r1,r2


def test_exact_nonlocal_quartic_retains_prestress_and_cross_bank_coupling():
    (coeff,p1,p2,residual),A,f,K,r1,r2=synthetic_coefficients()
    assert p1[:2]==(F(4),F(5)) and p2[:2]==(F(5),F(1))
    assert coeff==(F(1000000),F(8),F(259,2),F(128),F(89))
    assert residual==(F(13),F(-7),F(23))
    for lam in (F(0),F(1,7),F(1,2),F(1)):
        r=tuple(x*lam+y*lam**2 for x,y in zip(r1,r2))
        p=tuple(sum((a*x for a,x in zip(row,r)),F(0)) for row in A)
        direct=coeff[0]+sum((fi*x+Ki*x*x/2 for fi,Ki,x in zip(f,K,p)),F(0))
        direct+=20*lam**2+30*lam**4-sum((x*y/2 for x,y in zip(r,p)),F(0))
        assert api._polynomial(coeff,lam,96)==direct


def test_signed_recurrence_keeps_negative_work_and_reverses_reaction():
    coeff=synthetic_coefficients((-100,4))[0][0]
    a,b=F(1,9),F(4,5)
    work=api._difference(coeff,a,b,96)
    assert work==api._polynomial(coeff,b,96)-api._polynomial(coeff,a,96)
    assert work<0 and api._sign(work)=='NEGATIVE'
    assert api._difference(coeff,b,a,96)==-work
    assert api._difference(coeff,a,a,96)==0
    interval=api._difference(coeff,I(a,b),I(F(5,6),F(1)),96)
    for first in (a,F(1,2),b):
        for second in (F(5,6),F(1)):
            exact=api._difference(coeff,first,second,96)
            assert interval.lo<=exact<=interval.hi


def test_proof_threshold_brackets_lower_bound_zero_without_claiming_actual_work_zero():
    bracket=api._positive_work_bracket(F(10),F(20),F(3),96)
    phi=lambda x:10*x*x+20*x**4-3
    assert phi(bracket.lo)<=0<=phi(bracket.hi)
    assert bracket.width<F(1,10**96)
    assert api._positive_work_bracket(F(0),F(0),F(3),96) is None
    assert api._positive_work_bracket(F(10),F(20),F(0),96)==I(0)


@pytest.fixture(scope='module')
def context():
    root=Path(__file__).resolve().parents[1]
    new=runpy.run_path(str(root/'analysis/geometric_native_common_cold_preparation_energy_validation/native_control.txt'))
    ctx=new['context']()
    source,integral=ctx['source'],ctx['integral']
    energy=api.bind_native_common_cold_preparation_energy(source,integral)
    return source,integral,energy


def test_genuine_whole_source_binding_and_exact_unmodified_zero_energy(context):
    source,integral,energy=context
    assert energy.original_parent_ids==tuple('octant:'+str(i) for i in range(8))
    assert len(energy.nonempty_archive_domain_ids)==3
    assert energy.integrals_binding_sha256==integral.binding_sha256
    zero=api.evaluate_native_common_cold_preparation_work(energy,F(0))
    assert type(zero.energy_j) is F
    assert zero.energy_j==sum(source.literal_reference_energies_j,F(0))
    assert zero.potential_path_work_j==zero.physical_reaction_path_work_j==0
    assert zero.potential_path_power_j==energy.energy_polynomial_coefficients_j[1]
    assert zero.cold_constitutive_preparation_work_computed
    assert energy.whole_original_integral_certified and energy.source_beams_included_once
    assert not zero.reference_preparation_work_paid
    assert not zero.complete_physical_system_preparation_work_paid
    assert not zero.historical_three_dimensional_prestress_recovered
    assert not zero.whole_clock_all_pair_admission_authorized
    assert not zero.physical_temporal_path_admission_certified
    assert not zero.clocks_advanced and zero.accepted_new_native_intervals==0


@pytest.mark.parametrize('lam',[F(1,10),F(1,2),F(1)])
def test_whole_path_power_work_and_independent_complement_bound(context,lam):
    source,integral,energy=context
    result=api.evaluate_native_common_cold_preparation_work(energy,lam)
    coeff=energy.energy_polynomial_coefficients_j
    overlap(result.energy_j-coeff[0],result.potential_path_work_j)
    assert result.physical_reaction_path_work_j==-result.potential_path_work_j
    assert result.physical_reaction_path_power_j==-result.potential_path_power_j
    overlap(result.potential_path_power_j,
        sum((k*c*lam**(k-1) for k,c in enumerate(coeff) if k),F(0)))
    overlap(result.potential_path_second_derivative_j,
        sum((k*(k-1)*c*lam**(k-2) for k,c in enumerate(coeff) if k>1),F(0)))
    lower=lam**2*integral.inactive_complement_S2_bounds_j.lo+lam**4*integral.inactive_complement_S4_bounds_j.lo
    lower-=sum((f*f/(2*K) for f,K in zip(source.literal_reference_forces_n,source.tip_stiffnesses_n_m)),F(0))
    assert result.independent_potential_path_work_lower_bound_j==lower
    assert result.independent_physical_reaction_work_upper_bound_j==-lower
    assert result.orthogonal_residual_energy_j.lo>=0
    assert result.unintersected_orthogonal_residual_energy_j.hi>=0
    if lower>0:
        assert result.certified_potential_work_sign=='POSITIVE'
    raw=api.native_common_cold_preparation_work_to_dict(result)
    assert api.native_common_cold_preparation_work_from_dict(raw,energy=energy,
        expected_lambda_fraction=lam)==result


def test_interval_lambda_retains_whole_parameter_range_and_signed_work(context):
    _,_,energy=context
    bounds=I(F(1,4),F(3,4))
    current=api.evaluate_native_common_cold_preparation_work(energy,bounds)
    for lam in (F(1,4),F(1,2),F(3,4)):
        sample=api.evaluate_native_common_cold_preparation_work(energy,lam)
        for field in ('energy_j','potential_path_work_j','potential_path_power_j',
                'potential_path_second_derivative_j'):
            outer,inner=api._bounds(getattr(current,field)),api._bounds(getattr(sample,field))
            assert outer.lo<=inner.lo<=inner.hi<=outer.hi
        assert current.independent_potential_path_work_lower_bound_j<=sample.independent_potential_path_work_lower_bound_j
    assert current.physical_reaction_path_work_j==-current.potential_path_work_j


def test_forward_reverse_and_joint_path_restart_preserve_declared_endpoints(context):
    _,_,energy=context
    a,b=F(1,3),F(1)
    forward=api.compare_native_common_cold_preparation_path_work(energy,a,b)
    reverse=api.compare_native_common_cold_preparation_path_work(energy,b,a)
    assert _saved(reverse['potential_path_work_j'])==-_saved(forward['potential_path_work_j'])
    assert _saved(forward['physical_reaction_path_work_j'])==-_saved(forward['potential_path_work_j'])
    assert api.native_common_cold_preparation_path_work_from_dict(forward,energy=energy,
        expected_before_fraction=a,expected_after_fraction=b)==forward
    same=api.compare_native_common_cold_preparation_path_work(energy,I(a),I(a))
    assert _saved(same['potential_path_work_j'])==0
    changed=dict(forward); changed['complete_physical_system_preparation_work_paid']=True
    with pytest.raises(ValueError):
        api.native_common_cold_preparation_path_work_from_dict(changed,energy=energy,
            expected_before_fraction=a,expected_after_fraction=b)


def test_unregistered_arrays_and_resealed_moments_cannot_gain_material_authority(context):
    source,integral,energy=context
    with pytest.raises(ValueError):
        api.bind_native_common_cold_preparation_energy(source,{'r1':integral.r1_bounds_n})
    forged=replace(integral,r1_bounds_n=(I(0),)*8,binding_sha256='')
    with pytest.raises(ValueError):
        api.bind_native_common_cold_preparation_energy(source,forged)
    forged=replace(energy,complete_physical_system_preparation_work_paid=True)
    with pytest.raises(ValueError):
        api.evaluate_native_common_cold_preparation_work(forged,F(1))


def test_energy_restart_and_frozen_bypass_mutation_guard(context):
    source,integral,energy=context
    raw=api.native_common_cold_preparation_energy_to_dict(energy)
    assert api.native_common_cold_preparation_energy_from_dict(raw,source=source,integral=integral) is energy
    changed=dict(raw); changed['cold_constitutive_preparation_work_computed']=1
    with pytest.raises(ValueError):
        api.native_common_cold_preparation_energy_from_dict(changed,source=source,integral=integral)
    old=energy.complete_physical_system_preparation_work_paid
    try:
        object.__setattr__(energy,'complete_physical_system_preparation_work_paid',True)
        with pytest.raises(ValueError):
            api.evaluate_native_common_cold_preparation_work(energy,F(1))
    finally:
        object.__setattr__(energy,'complete_physical_system_preparation_work_paid',old)
    assert api.native_common_cold_preparation_energy_to_dict(energy)==raw


@pytest.mark.parametrize('parameter',[True,0,0.5,F(-1),F(2),I(F(-1,10),F(1,2))])
def test_literal_parameter_rejection(context,parameter):
    _,_,energy=context
    with pytest.raises(ValueError):
        api.evaluate_native_common_cold_preparation_work(energy,parameter)
