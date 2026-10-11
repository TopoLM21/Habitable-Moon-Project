"""New reference-depth profile mass and authentic physical coarea are distinct."""
from copy import deepcopy
from dataclasses import fields,replace
from decimal import Decimal as D,localcontext
from fractions import Fraction as F
import hashlib
import json

import pytest

from tectonics.geometric_native_pose import FractionInterval as I
from tectonics import geometric_native_common_contact_measure as api
from tectonics.geometric_native_common_contact_state import observe_native_common_contact_state
from tests.test_geometric_native_common_contact_ports import _reference,_case


def test_uniform_and_linear_profiles_have_exact_mass_and_independent_cdf():
    assert api._profile_integral((F(1),),I(0,1)) == 1
    assert api._profile_power_coefficients((F(0),F(2))) == (F(0),F(2))
    assert api._profile_integral((F(0),F(2)),I(F(1,3),F(2,3))) == F(1,3)
    assert api._profile_integral((F(0),F(2)),I(0,F(1,3))) == F(1,9)


def test_exact_bernstein_quadratic_mass_partitions_without_midpoints():
    profile = F(3,2),F(0),F(3,2)
    assert api._profile_power_coefficients(profile) == (F(3,2),F(-3),F(3))
    pieces = I(0,F(1,3)),I(F(1,3),F(2,3)),I(F(2,3),1)
    assert sum((api._profile_integral(profile,p) for p in pieces),F(0)) == 1
    assert api._profile_bounds(profile,I(0,1)).lo >= 0


def test_nonnegative_normalization_is_exact_and_not_a_floor():
    for profile in ((F(0),),(F(1),F(0)),(F(-1),F(3)),(1,),[F(1)]):
        with pytest.raises(ValueError,match='nonnegative exact'):
            api._profile(profile)


def test_depth_profile_rejects_queries_outside_the_declared_reference_interval():
    for z in (I(-1,0),I(0,2),F(1,2)):
        with pytest.raises(ValueError,match='Normalized depth'):
            api._profile_integral((F(1),),z)


def test_zero_measure_closed_seam_and_nonuniform_positive_interval():
    assert api._profile_integral((F(0),F(2)),I(F(1,3))) == 0
    assert api._profile_bounds((F(0),F(2)),I(F(1,3),F(2,3))) == I(F(2,3),F(4,3))


def _decl(response,profile=(F(1),)):
    state = observe_native_common_contact_state(_reference(),response,precision=96)
    declaration = api.declare_native_common_contact_reference_measure(_reference(),response,state,
        provenance='NEW reference-volume-depth profile; no inherited depth distribution',
        depth_profile_coefficients=profile)
    result = api.evaluate_native_common_contact_reference_measure(_reference(),response,state,declaration,precision=96)
    return state,declaration,result


@pytest.mark.parametrize('index',range(5))
def test_five_genuine_cut_densities_use_original_receiver_measure_and_true_coarea(index):
    response = _case(index)
    state,declaration,result = _decl(response)
    assert result.receiving_angle_orientation == -1
    assert result.receiving_native_angle_parameter_derivative_bounds.lo <= -1 <= result.receiving_native_angle_parameter_derivative_bounds.hi
    assert declaration.source_interface_area_density_m2_per_computational_radian == (
        declaration.source_radius_m*F(1000)/declaration.source_literal_original_frame_sine)
    cube = response.port.cut.common_physical_radius_cubed_bounds_m3
    assert declaration.physical_volume_depth_span_m == (cube.hi-cube.lo)/(3*declaration.source_radius_m**2)
    assert result.physical_area_density_bounds_m2_per_parameter_per_physical_volume_depth_m == response.geometry.area_density_bounds_m2_per_parameter_per_material_m
    assert result.source_current_memory_sha256 == state.source_current_memory_sha256
    assert not result.full_face_memory_measure_transfer_certified
    assert not result.global_receiver_source_allocation_certified
    assert not result.old_mechanical_atom_area_assigned
    assert not result.old_rod_energy_or_prestress_transferred
    # Independent analytic receiving-plane law is theta_native'=-1. The
    # physical coarea is |U_s|R^2/r sqrt(1+(r^3 A dot T/R^2)^2).
    from tests.test_geometric_native_common_contact_ports import _mid,_dot,_cross,_d
    with localcontext() as ctx:
        ctx.prec = 130
        g=response.geometry;R=_d(declaration.source_radius_m);r=_mid(g.physical_radius_bounds_m)
        U=tuple(_mid(x) for x in g.source_unit_world_direction_bounds)
        Up=tuple(_mid(x) for x in g.source_unit_world_direction_parameter_derivative_bounds)
        A=tuple(_d(x) for x in _reference().angular_rate_per_physical_volume_depth_m)
        c=r**3/(R*R)*_dot(A,Up)
        physical=R*R/r*(1+c*c).sqrt()
        bounds=result.physical_area_density_bounds_m2_per_parameter_per_physical_volume_depth_m
        assert _d(bounds.lo)-D('1e-85') <= physical <= _d(bounds.hi)+D('1e-85')
    full=I(0,declaration.source_computational_extent_radians)
    assert api.source_reference_rectangle_area(declaration,computational_angle_bounds=full,normalized_depth_bounds=I(0,1)) == declaration.source_reference_interface_area_m2
    assert sum((api.source_reference_rectangle_area(declaration,computational_angle_bounds=full,
        normalized_depth_bounds=z) for z in (I(0,F(1,3)),I(F(1,3),1))),F(0)) == declaration.source_reference_interface_area_m2


def test_profile_changes_depth_density_while_modal_increment_leaves_reference_unchanged():
    from tectonics.geometric_native_common_contact_ports import evaluate_native_common_contact_port
    response=_case(2);state,declaration,result=_decl(response,(F(0),F(2)))
    zero=evaluate_native_common_contact_port(_reference(),response.port,parameter_bounds=response.geometry.parameter_bounds,
        physical_radius_cubed_bounds_m3=response.geometry.common_physical_radius_cubed_bounds_m3,precision=96)
    state0=observe_native_common_contact_state(_reference(),zero,precision=96)
    assert state0 == state
    result0=api.evaluate_native_common_contact_reference_measure(_reference(),zero,state0,declaration,precision=96)
    assert result0 == result
    exact_profile = 2*result.normalized_depth_bounds
    assert result.depth_profile_value_bounds.lo <= exact_profile.lo <= exact_profile.hi <= result.depth_profile_value_bounds.hi
    # A second true depth has a new physical coarea, not an old 1000m height.
    cube=response.port.cut.common_physical_radius_cubed_bounds_m3
    other=evaluate_native_common_contact_port(_reference(),response.port,parameter_bounds=response.geometry.parameter_bounds,
        physical_radius_cubed_bounds_m3=I(cube.lo+(cube.hi-cube.lo)*F(2,3)),precision=96)
    otherstate=observe_native_common_contact_state(_reference(),other,precision=96)
    otherresult=api.evaluate_native_common_contact_reference_measure(_reference(),other,otherstate,declaration,precision=96)
    assert otherresult.normalized_depth_bounds != result.normalized_depth_bounds
    assert otherresult.physical_area_density_bounds_m2_per_parameter_per_physical_volume_depth_m != result.physical_area_density_bounds_m2_per_parameter_per_physical_volume_depth_m
    full=I(0,declaration.source_computational_extent_radians)
    assert api.source_reference_rectangle_area(declaration,computational_angle_bounds=full,normalized_depth_bounds=I(0,F(1,3))) == declaration.source_reference_interface_area_m2/9


def test_depth_measure_serde_replays_source_and_rejects_resealed_forgery():
    response=_case(0);state,declaration,result=_decl(response,(F(0),F(2)))
    raw=api.native_common_contact_reference_measure_declaration_to_dict(declaration)
    assert api.native_common_contact_reference_measure_declaration_from_dict(raw,reference=_reference(),port_response=response,state=state) == declaration
    for key in ('source_interface_area_density_m2_per_computational_radian','physical_volume_depth_span_m','source_contact_elapsed_time_s'):
        changed=deepcopy(raw);changed[key]=[1,1]
        body={f.name:changed[f.name] for f in fields(declaration)};body.pop('binding_sha256')
        changed['binding_sha256']=hashlib.sha256(api._encoded(body)).hexdigest()
        with pytest.raises(ValueError,match='differs from source'):
            api.native_common_contact_reference_measure_declaration_from_dict(changed,reference=_reference(),port_response=response,state=state)
    rawresponse=api.native_common_contact_reference_measure_response_to_dict(result)
    assert api.native_common_contact_reference_measure_response_from_dict(rawresponse,reference=_reference(),port_response=response,state=state,declaration=declaration) == result
    forged=replace(result,normalized_depth_bounds=I(0));forged=replace(forged,binding_sha256=api._seal(forged))
    with pytest.raises(ValueError,match='lacks authentic replay'):
        api.validate_native_common_contact_reference_measure_response(forged)
    changed=deepcopy(rawresponse);changed['full_face_memory_measure_transfer_certified']=True
    with pytest.raises(ValueError,match='differs from authentic'):
        api.native_common_contact_reference_measure_response_from_dict(changed,reference=_reference(),port_response=response,state=state,declaration=declaration)


def test_hot_curve_bad_provenance_and_foreign_cut_cannot_borrow_measure():
    response=_case(0);state,declaration,_=_decl(response)
    with pytest.raises(ValueError,match='no source receiver'):
        api.declare_native_common_contact_reference_measure(_reference(),_case(5),state,provenance='NEW')
    with pytest.raises(ValueError,match='nonempty provenance'):
        api.declare_native_common_contact_reference_measure(_reference(),response,state,provenance=' ')
    with pytest.raises(ValueError,match='another material cut'):
        api.validate_native_common_contact_reference_measure_declaration(declaration,reference=_reference(),port_response=_case(1))
    with pytest.raises(ValueError,match='original computational extent'):
        api.source_reference_rectangle_area(declaration,computational_angle_bounds=I(-1,0),normalized_depth_bounds=I(0,1))
    # The two C8 cuts share one source memory family, but declaration lineage
    # must still reject another genuine cut even without optional port input.
    state8,decl8,_=_decl(_case(2))
    otherstate8=observe_native_common_contact_state(_reference(),_case(3),precision=96)
    assert state8.receiving_trace_id == otherstate8.receiving_trace_id
    with pytest.raises(ValueError,match='cannot substitute'):
        api.validate_native_common_contact_reference_measure_declaration(decl8,state=otherstate8)
