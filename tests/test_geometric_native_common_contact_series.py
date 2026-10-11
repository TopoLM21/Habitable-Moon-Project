"""Exact three-spring condensation and genuine objective anchor trace calculus."""
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
from functools import lru_cache

import pytest

from tectonics.geometric_native_pose import FractionInterval as I
from tectonics import geometric_native_common_contact_series as api
from tests.test_geometric_native_common_contact_constitutive import (
    _actual, _mid, _d, _close, _numeric_actual, _energy, _fd_first, _fd_second,
)


@pytest.mark.parametrize("component", ("normal", "shear"))
def test_pure_exact_series_matches_independent_two_node_schur_complement(component):
    columns = (F(1000), F(10**9), F(4*10**8), F(1000), F(10**9), F(4*10**8))
    normal, shear, Kn, Kt = api._exact_coefficients(F(10**6), F(10**6), columns)
    ks = columns[1 if component == "normal" else 2]/columns[0]
    ko = columns[4 if component == "normal" else 5]/columns[3]
    kc, K = F(10**6), Kn if component == "normal" else Kt
    C = normal if component == "normal" else shear
    # Solve the physical two internal displacement nodes independently from
    # [[ks+kc,-kc],[-kc,kc+ko]] x = [0,ko*a].  Do not use a compliance shortcut
    # in this oracle.  Anchor a is the imposed relative displacement.
    determinant = (ks+kc)*(kc+ko)-kc*kc
    def nodes(a):
        return kc*ko*a/determinant, (ks+kc)*ko*a/determinant
    def U(a):
        xd, xr = nodes(a)
        return ks*xd*xd/2+kc*(xr-xd)**2/2+ko*(xr-a)**2/2
    anchor, h = F(-7, 1000), F(1, 100000)
    xd, xr = nodes(anchor)
    traction = kc*(xr-xd)
    assert ks*xd-traction == 0
    assert ko*(xr-anchor)+traction == 0
    assert traction == K*anchor
    assert U(anchor) == K*anchor*anchor/2
    assert (U(anchor+h)-U(anchor-h))/(2*h) == traction
    assert (U(anchor+h)-2*U(anchor)+U(anchor-h))/(h*h) == K
    assert K != kc+ks+ko
    assert tuple(C) == (1/ks, 1/ko)
    assert (xr-xd)-anchor == -traction*sum(C)


@lru_cache(maxsize=1)
def _declared():
    reference, port, state = _actual(0)
    declaration = api.declare_native_common_contact_series_carrier(reference, port, state,
        provenance="Independent test of the explicitly declared reduced anchor carrier",
        anchor_interpretation=api.ANCHOR_INTERPRETATION)
    return reference, port, state, declaration


@pytest.mark.parametrize("branch", ("relaxed_source", "frozen_memory_predictor"))
def test_actual_serial_energy_and_full_force_hessian_match_independent_finite_rotations(branch):
    reference, port, state, declaration = _declared()
    value = api.evaluate_native_common_contact_series_carrier(reference, port, state, declaration,
        branch=branch, precision=96)
    with localcontext() as context:
        context.prec = 110
        numeric = _numeric_actual(port)
        count = len(value.kinematics.coordinate_order)
        zero = (D(0),)*count
        function = lambda offsets: _energy(numeric, _mid(state.normal_pressure_bounds_pa),
            _d(declaration.exact_normal_series_stiffness_pa_m),
            _d(declaration.exact_shear_series_stiffness_pa_m), branch, offsets)
        _close(value.total_energy_bounds_j_m2, function(zero))
        body_count = value.kinematics.body_coordinate_count
        for i in range(count):
            hi = D("1e-7") if i < body_count else D("1e-12")
            _close(value.residual_bounds_per_reference_m2[i], _fd_first(function, zero, i, hi))
            for j in range(i, count):
                hj = D("1e-7") if j < body_count else D("1e-12")
                _close(value.hessian_bounds_per_reference_m2[i][j], _fd_second(function, zero, i, j, hi, hj))
                assert value.hessian_bounds_per_reference_m2[i][j] == value.hessian_bounds_per_reference_m2[j][i]
        first = value.kinematics.scalar_first_derivative_bounds
        differences = []
        for i in range(body_count, count):
            for j in range(body_count, count):
                gram = _d(declaration.exact_normal_series_stiffness_pa_m)*_mid(first[0][i])*_mid(first[0][j])
                if branch == "frozen_memory_predictor":
                    gram += _d(declaration.exact_shear_series_stiffness_pa_m)*_mid(first[1][i])*_mid(first[1][j])
                differences.append(abs(_mid(value.hessian_bounds_per_reference_m2[i][j])-gram))
        assert max(differences) > D(1)
    assert value.total_energy_bounds_j_m2 == sum(value.energy_component_bounds_j_m2, I(0))


def test_actual_zero_anchor_preserves_source_pressure_and_individual_rod_energy_once():
    from tectonics.geometric_native_common_contact_ports import evaluate_native_common_contact_port
    from tectonics.geometric_native_common_contact_state import observe_native_common_contact_state
    reference, original, _, _ = _declared()
    state = observe_native_common_contact_state(reference, original, precision=160)
    declaration = api.declare_native_common_contact_series_carrier(reference, original, state,
        provenance="Literal source pressure with exact individual serial compliances",
        anchor_interpretation=api.ANCHOR_INTERPRETATION)
    zero = evaluate_native_common_contact_port(reference, original.port,
        parameter_bounds=original.geometry.parameter_bounds,
        physical_radius_cubed_bounds_m3=original.geometry.common_physical_radius_cubed_bounds_m3,
        precision=96)
    value = api.evaluate_native_common_contact_series_carrier(reference, zero, state, declaration, precision=96)
    assert value.traction_bounds_pa == (-state.normal_pressure_bounds_pa, I(0))
    assert value.total_energy_bounds_j_m2 == value.reference_total_energy_bounds_j_m2
    assert value.reference_total_energy_bounds_j_m2.lo > 0
    assert value.energy_component_order == ("contact", "sub_column", "over_column")
    assert value.energy_component_bounds_j_m2[0] == value.energy_component_bounds_j_m2[1] == value.energy_component_bounds_j_m2[2]
    assert value.total_energy_bounds_j_m2 == 3*value.energy_component_bounds_j_m2[0]
    for compliance, shortening in zip(declaration.individual_normal_compliances_m_pa,
                                     value.individual_rod_normal_shortening_bounds_m):
        assert shortening == state.normal_pressure_bounds_pa*compliance
    assert declaration.exact_normal_series_stiffness_pa_m == F(1000000, 3)
    assert declaration.exact_shear_series_stiffness_pa_m == F(500000, 3)
    assert declaration.exact_minus_source_normal_series_stiffness_pa_m != 0
    assert declaration.exact_minus_source_shear_series_stiffness_pa_m != 0
    assert declaration.normal_anchor_minus_source_free_gap_bounds_m == declaration.normal_anchor_reference_bounds_m-state.free_gap_bounds_m
    assert not value.old_rod_three_dimensional_material_map_certified
    assert not value.old_world_or_generalized_force_handover_certified
    assert not value.additive_to_853_contact_only_energy_authorized
    assert not value.physical_interface_area_or_memory_allocation_certified
    assert not value.all_sixteen_source_carriers_physically_transferred


def test_actual_relaxed_serial_shear_updates_only_a_proposal_and_has_zero_rod_shear_energy():
    reference, port, state, declaration = _declared()
    relaxed = api.evaluate_native_common_contact_series_carrier(reference, port, state, declaration, precision=96)
    frozen = api.evaluate_native_common_contact_series_carrier(reference, port, state, declaration,
        branch="frozen_memory_predictor", precision=96)
    xi_t = relaxed.kinematics.scalar_increment_bounds_m[1]
    assert relaxed.traction_bounds_pa[1] == I(0)
    assert relaxed.elastic_contact_shear_strain_bounds_m == I(0)
    assert relaxed.individual_rod_shear_extension_bounds_m == (I(0),)*2
    assert relaxed.local_tangent_bounds_pa_m[1][1] == I(0)
    assert relaxed.proposed_plastic_increment_bounds_m == xi_t
    assert frozen.proposed_plastic_increment_bounds_m == I(0)
    assert frozen.local_tangent_bounds_pa_m[1][1] == I(declaration.exact_shear_series_stiffness_pa_m)
    _close(relaxed.proposed_plastic_slip_bounds_m, _mid(state.plastic_slip_bounds_m)+_mid(xi_t))
    _close(relaxed.proposed_cumulative_slip_bounds_m, _mid(state.cumulative_slip_bounds_m)+abs(_mid(xi_t)))
    assert relaxed.contact_jump_bounds_m == relaxed.proposed_plastic_slip_bounds_m
    for value in (relaxed, frozen):
        assert not value.source_memory_proposal_applied
        assert not value.source_history_or_thermal_state_advanced
        assert not value.old_rod_energy_added_to_existing_common_bulk_prestrain
        assert not value.force_equilibrium_certified
        assert not value.physical_material_admission_authorized


def test_actual_serial_declaration_and_response_replay_reject_resealed_forgery():
    reference, port, state, declaration = _declared()
    raw = api.native_common_contact_series_carrier_declaration_to_dict(declaration)
    replay = api.native_common_contact_series_carrier_declaration_from_dict(raw,
        reference=reference, port_response=port, source_observation=state)
    assert replay == declaration
    response = api.evaluate_native_common_contact_series_carrier(reference, port, state, declaration, precision=96)
    exported = api.native_common_contact_series_response_to_dict(response)
    assert api.native_common_contact_series_response_from_dict(exported, reference=reference,
        port_response=port, source_observation=state, declaration=declaration) == response
    forged = replace(declaration, exact_normal_series_stiffness_pa_m=declaration.exact_normal_series_stiffness_pa_m+1)
    forged = replace(forged, binding_sha256=api._seal(forged))
    with pytest.raises(ValueError, match="lacks authentic"):
        api.evaluate_native_common_contact_series_carrier(reference, port, state, forged, precision=96)
    changed = deepcopy(exported)
    changed["hessian_bounds_per_reference_m2"][0][0]["lo"][0] += 1
    with pytest.raises(ValueError, match="complete"):
        api.native_common_contact_series_response_from_dict(changed, reference=reference,
            port_response=port, source_observation=state, declaration=declaration)


def test_actual_serial_carrier_requires_explicit_anchor_provenance_and_no_hot_inheritance():
    from tests.test_geometric_native_common_contact_ports import _case
    reference, port, state, declaration = _declared()
    with pytest.raises(ValueError, match="nonempty provenance"):
        api.declare_native_common_contact_series_carrier(reference, port, state,
            provenance=" ", anchor_interpretation=api.ANCHOR_INTERPRETATION)
    with pytest.raises(ValueError, match="interpretation"):
        api.declare_native_common_contact_series_carrier(reference, port, state,
            provenance="A source observation is insufficient", anchor_interpretation="physical_3D_old_rods")
    with pytest.raises(ValueError, match="HOT"):
        api.declare_native_common_contact_series_carrier(reference, _case(5), state,
            provenance="Rejected missing other-side law", anchor_interpretation=api.ANCHOR_INTERPRETATION)
    assert declaration.anchor_coordinate_origin == "ZERO_MODAL_INCREMENT_IN_NEW_COMMON_REFERENCE"
    assert declaration.assembly_role == "REPLACEMENT_LOCAL_SERIES_COMPONENT_NOT_PARALLEL_CONTACT853"
