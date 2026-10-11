"""Actual prepared source assembly and exact independent beam/load arithmetic."""
import copy
from dataclasses import replace
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

from tectonics.geometric_native_pose import FractionInterval as I
from tectonics import geometric_native_common_preparation_balance as api


def exact(value):
    return (value.lo,value.hi) if type(value) is I else (value,value)


@pytest.fixture(scope='module')
def context():
    root = Path(__file__).resolve().parents[1]
    return runpy.run_path(str(root/'analysis/geometric_native_common_preparation_balance_validation/native_control.txt'))['context']()


@pytest.fixture(scope='module')
def system(context):
    return context['balance_system']


def pure_cold():
    # The bank order deliberately differs from original internal-coordinate
    # order, making a position-only beam injection detectably wrong.
    return SimpleNamespace(cold_energy_polynomial_coefficients_j=(F(101),F(3),F(5),F(7),F(11)),
        source_beam_forces_n=tuple(F(i+2) for i in range(8)),
        source_beam_stiffnesses_n_m=tuple(F(2*i+3) for i in range(8)),
        cold_tip_projection_linear_m=tuple(F(i+1,9) for i in range(8)),
        cold_tip_projection_quadratic_m=tuple(F(i+1,13) for i in range(8)),
        original_beam_internal_indices=(13,7,9,11,12,6,8,10),
        active_original100_indices=tuple(range(6))+tuple(range(92,100)))


def explicit_pure_energy(s,lam,internal):
    base = sum(c*lam**i for i,c in enumerate(s.cold_energy_polynomial_coefficients_j))
    correction = sum((force+K*(p1*lam+p2*lam*lam))*internal[index-6]
        +K*internal[index-6]**2/2 for force,K,p1,p2,index in zip(s.source_beam_forces_n,
        s.source_beam_stiffnesses_n_m,s.cold_tip_projection_linear_m,
        s.cold_tip_projection_quadratic_m,s.original_beam_internal_indices))
    return base+correction


def test_exact_same_beam_driver_shift_keeps_literal_prestress_and_bank_identity():
    s,lam = pure_cold(),F(2,5)
    internal = tuple(F(i-3,100) for i in range(8))
    base,correction,g,H,receipts = api._cold_twojet(s,lam,internal,96)
    assert exact(base+correction) == (explicit_pure_energy(s,lam,internal),)*2
    for k,index in enumerate(s.original_beam_internal_indices):
        expected = s.source_beam_forces_n[k]+s.source_beam_stiffnesses_n_m[k]*(
            lam*s.cold_tip_projection_linear_m[k]+lam*lam*s.cold_tip_projection_quadratic_m[k]+internal[index-6])
        assert exact(g[index]) == (expected,)*2
        assert exact(H[index][index]) == (s.source_beam_stiffnesses_n_m[k],)*2
        receipt = json.loads(receipts[k])
        assert receipt['original_internal_coordinate_index'] == s.active_original100_indices[index]
        assert receipt['old_beam_reference_energy_added_again'] is False


def test_exact_internal_preparation_cross_derivatives_match_independent_polynomial():
    s,lam = pure_cold(),F(2,5)
    internal = tuple(F(i-3,100) for i in range(8))
    _,_,g,H,_ = api._cold_twojet(s,lam,internal,96)
    coefficients = s.cold_energy_polynomial_coefficients_j
    first = sum(i*c*lam**(i-1) for i,c in enumerate(coefficients) if i)
    second = sum(i*(i-1)*c*lam**(i-2) for i,c in enumerate(coefficients) if i>=2)
    for K,p1,p2,index in zip(s.source_beam_stiffnesses_n_m,s.cold_tip_projection_linear_m,
            s.cold_tip_projection_quadratic_m,s.original_beam_internal_indices):
        first += K*internal[index-6]*(p1+2*lam*p2)
        second += 2*K*internal[index-6]*p2
        assert exact(H[index][14]) == (K*(p1+2*lam*p2),)*2
        assert H[index][14] == H[14][index]
    assert exact(g[14]) == (first,)*2
    assert exact(H[14][14]) == (second,)*2


def test_exact_gershgorin_owner_cap_includes_every_off_diagonal_and_torque():
    torque = tuple(I(F(i+1),F(i+2)) for i in range(6))
    drag = tuple(tuple(I(F(10+i)) if i==j else I(F(-1,10),F(1,5))
                      for j in range(6)) for i in range(6))
    symmetric,rows,alpha,cap = api._owner_relaxation_certificate(torque,drag,F(7))
    assert rows == tuple(F(9+i) for i in range(6))
    assert alpha == 9
    assert cap == F(7)*sum(F(i+2)**2 for i in range(6))/(2*alpha)
    assert symmetric[0][1] == I(F(-1,10),F(1,5))


def test_missing_or_non_positive_whole_owner_drag_is_rejected():
    with pytest.raises(ValueError,match='all36'):
        api._owner_relaxation_certificate((F(1),)*6,((F(1),)*5,)*6,F(1))
    with pytest.raises(ValueError,match='strictly positive'):
        api._owner_relaxation_certificate((F(1),)*6,((F(0),)*6,)*6,F(1))


def test_actual_catalogue_retains_frozen86_and_every_original_beam_once(system):
    assert len(system.original_coordinate_order) == 100
    assert len(system.coordinate_order) == 15
    assert system.active_original100_indices == tuple(range(6))+tuple(range(92,100))
    assert system.frozen_original100_indices == tuple(range(6,92))
    assert system.frozen_body_coordinate_values == (F(0),)*86
    assert set(system.original_beam_internal_indices) == set(range(6,14))
    assert system.old891_complete_beam_bulk_operator_added_again is False
    assert system.complete_modal_basal_loading_defined is False
    assert system.full_physical_preparation_balance_certified is False


def test_actual_prepared_zero_keeps_cold_and_source16_literal_energy(system):
    response = api.evaluate_native_common_preparation_balance(system)
    expected = sum(system.source_beam_energies_j)+system.source_contact_literal_energy_j
    assert exact(response.energy_bounds_j) == (expected,)*2
    assert exact(response.potential_path_work_from_prepared_zero_bounds_j) == (F(0),)*2
    assert response.coordinate_values == (F(0),)*15
    parts = json.loads(response.energy_partitions_json)
    assert parts['frozen_hot30_energy_bounds_j'] == [0,1]
    assert parts['frozen_owner6_load_drag_potential_bounds_j'] == [0,1]


def test_actual_internal_lambda_twojet_has_complete_mixed_curvature(system,context):
    lam,internal = F(1,2),tuple(F(i-3,10000000) for i in range(8))
    response = api.evaluate_native_common_preparation_balance(system,preparation_fraction=lam,
        internal_coordinates=internal)
    from tectonics.geometric_native_common_preparation_contacts import evaluate_native_common_preparation_contacts
    contact = evaluate_native_common_preparation_contacts(context['contacts_system'],
        preparation_fraction=lam,internal_coordinates=internal,precision=96)
    _,_,coldg,coldH,_ = api._cold_twojet(system,lam,internal,96)
    for index in range(6,15):
        expected = api._bounds(api._round(coldg[index]+contact.gradient_bounds[index],96))
        assert response.gradient_bounds[index] == expected
        for j in range(6,15):
            expectedH = api._bounds(api._round(coldH[index][j]+contact.hessian_bounds[index][j],96))
            assert response.hessian_bounds[index][j] == expectedH
    assert response.hessian_bounds[14][6] == response.hessian_bounds[6][14]
    assert response.complete_declared_active_fifteen_twojet_certified is True
    assert response.general_state_mixed_operator_certified is False


def test_actual_global_owner_spin_proves_constraint_reaction_not_free_equilibrium(system):
    for lam in (F(0),F(1,2),F(1)):
        response = api.evaluate_native_common_preparation_balance(system,preparation_fraction=lam)
        spin = response.common_owner_rotation_gradient_bounds_nm
        assert spin == tuple(api._round(-system.owner_torque_bounds_nm[i]
            -system.owner_torque_bounds_nm[i+3],96) for i in range(3))
        assert spin[2].hi < 0
        assert response.common_owner_z_rotation_strict_descent_certified is True
        assert response.clamped_owner_slice_is_free_mixed_equilibrium is False
        assert response.owner_motion_equilibrium_excluded is False


def test_actual_work_floor_counts_one_beam_cap_even_with_nonzero_internal(system):
    internal = tuple(F(i-3,10000000) for i in range(8))
    response = api.evaluate_native_common_preparation_balance(system,preparation_fraction=F(1),
        internal_coordinates=internal)
    c2,c4 = (api._bounds(v).lo for v in system.inactive_complement_half_norm_coefficients_j)
    expected = c2+c4-system.eight_literal_beam_relaxation_cap_j+system.source_contact_relaxation_lower_bound_j
    assert response.independent_fixed_owner_potential_work_lower_bound_j == expected
    assert response.independent_owner_relaxed_potential_work_lower_bound_j == expected-system.owner_potential_relaxation_cap_j
    assert response.independent_owner_relaxed_potential_work_lower_bound_j > 0
    assert system.owner_potential_relaxation_cap_j > -system.source_contact_relaxation_lower_bound_j


def test_actual_registered_source_comparison_uses_whole_saved_state(system,context):
    raw = context['current907']; mixed = raw['mixed_force_system']
    matches = [q['response'] for q in raw['evaluation_queries']
               if q['response']['coordinate_increments'] == mixed['bound_coordinate_increments']]
    assert len(matches) == 1
    assert system.registered_source_response_binding_sha256 == matches[0]['binding_sha256']
    assert system.registered_source_energy_bounds_j == api._saved(matches[0]['energy_bounds_j'])
    response = api.evaluate_native_common_preparation_balance(system,preparation_fraction=F(1))
    assert response.registered_source_potential_energy_difference_bounds_j == api._bounds(api._round(
        response.energy_bounds_j-system.registered_source_energy_bounds_j,96))
    assert response.independent_registered_source_potential_energy_difference_lower_bound_j > 0
    assert response.full_system_temporal_path_work_certified is False


def test_actual_complete_system_and_response_restart(context,system):
    dto = api.native_common_preparation_balance_to_dict(system)
    restored = api.native_common_preparation_balance_from_dict(dto,context['cold923'],context['cold899'],
        context['hot903'],context['current907'],contacts_system=context['contacts_system'])
    assert api.native_common_preparation_balance_to_dict(restored) == dto
    internal = tuple(F(i-3,10000000) for i in range(8))
    response = api.evaluate_native_common_preparation_balance(system,preparation_fraction=F(1,2),
        internal_coordinates=internal)
    dto = api.native_common_preparation_balance_response_to_dict(response)
    restored = api.native_common_preparation_balance_response_from_dict(dto,system=system,
        expected_preparation_fraction=F(1,2),expected_internal_coordinates=internal)
    assert api.native_common_preparation_balance_response_to_dict(restored) == dto


def test_actual_forged_or_mutated_typed_binding_is_rejected(system):
    with pytest.raises(ValueError,match='altered'):
        api.evaluate_native_common_preparation_balance(replace(system,owner_potential_relaxation_cap_j=F(0)))
    response = api.evaluate_native_common_preparation_balance(system,preparation_fraction=F(1))
    with pytest.raises(ValueError,match='altered'):
        api.native_common_preparation_balance_response_to_dict(replace(response,reference_preparation_work_paid=True))


def test_actual_full_authority_resealing_cannot_replace_whole_source(context):
    changed = copy.deepcopy(context['cold923'])
    changed['energy']['energy_polynomial_coefficients_j'][0][0] += 1
    changed['sha256'] = hashlib.sha256(api._json({k:v for k,v in changed.items()
        if k!='sha256'}).encode()).hexdigest()
    with pytest.raises(ValueError,match='entire frozen923'):
        api.bind_native_common_preparation_balance(changed,context['cold899'],context['hot903'],
            context['current907'],contacts_system=context['contacts_system'],provenance='tampered source')


def test_actual_scope_promotion_in_response_restart_is_rejected(system):
    response = api.evaluate_native_common_preparation_balance(system,preparation_fraction=F(1,2))
    dto = api.native_common_preparation_balance_response_to_dict(response)
    dto['full_physical_preparation_balance_certified'] = True
    with pytest.raises(ValueError,match='exact expected query'):
        api.native_common_preparation_balance_response_from_dict(dto,system=system,
            expected_preparation_fraction=F(1,2))


def test_actual_signed_state_differences_and_exact_equal_endpoint(system):
    forward = api.compare_native_common_preparation_balance_states(system,F(0),F(1))
    backward = api.compare_native_common_preparation_balance_states(system,F(1),F(0))
    f,b = api._saved(forward['potential_energy_change_bounds_j']),api._saved(backward['potential_energy_change_bounds_j'])
    assert f == -b
    same = api.compare_native_common_preparation_balance_states(system,F(1,2),F(1,2))
    assert same['potential_energy_change_bounds_j'] == {'lo':[0,1],'hi':[0,1]}
    assert forward['physical_reverse_unloading_authorized'] is False
    restored = api.native_common_preparation_balance_state_difference_from_dict(forward,
        system=system,before_fraction=F(0),after_fraction=F(1))
    assert restored == forward


def test_actual_full_physical_solve_rejects_missing86_basal_loads(system):
    with pytest.raises(ValueError,match='missing86 loads cannot be filled with zero'):
        api.require_native_common_preparation_full_physical_solve(system)


def test_actual_invalid_coordinate_kinds_do_not_alias_saved_queries(system):
    with pytest.raises(ValueError,match='literal Fraction lambda'):
        api.evaluate_native_common_preparation_balance(system,preparation_fraction=0.5)
    with pytest.raises(ValueError,match='literal immutable Fractions'):
        api.evaluate_native_common_preparation_balance(system,internal_coordinates=(0,)*8)
    with pytest.raises(ValueError,match='lambda'):
        api.evaluate_native_common_preparation_balance(system,preparation_fraction=F(2))
