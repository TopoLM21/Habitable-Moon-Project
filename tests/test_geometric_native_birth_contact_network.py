"""Joint equilibrium checks independent of the network's matrix assembly."""
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
import json
from types import SimpleNamespace

import pytest

from tectonics.genesis_contact_law import ContactLawParameters
from tectonics.geometric_contact_mechanics import ElasticColumnParameters, solve_compressed_contact, state_from_dict
from tectonics.geometric_native_birth_contact import (
    BornContactParameters, BornContactReference, BornFoundationParameters,
    BornFoundationReference, activate_born_contact, activate_born_foundation,
)
from tectonics.geometric_native_birth_contact_network import (
    BornNetworkAttachment, BornNetworkFoundation, UnresolvedBornContactNetwork,
    _advance_original_closed_inviscid, _upper_float,
    birth_contact_network_to_dict, solve_birth_contact_network,
)
from tectonics.geometric_native_polygons import NativePlane


def _physics():
    columns = ElasticColumnParameters(1., 2., 2., 1., 2., 2.)
    law = ContactLawParameters(normal_stiffness_pa_m=1., tangential_stiffness_pa_m=1.,
        tensile_strength_pa=1., fracture_energy_j_m2=1., cohesion_pa=0.,
        friction_dry=0., friction_wet=0., wet_cohesion_fraction=1., viscosity_pa_s_m=0.)
    return columns, law


def _case(*, gap=-1., interface_area=1., basal_area=1., shared=False, old_contact=True):
    columns, law = _physics()
    parameters = BornContactParameters(columns, law, "Explicit independent normal and column closure")
    reference = BornContactReference("new-normal", "a"*64, "b"*64,
        "donor", "receiver", "donor-material", "hot-material", "cold-parent", "hot-parent",
        0, 1, NativePlane((1, 0, 0), label="Independent weak plane"), Fraction(1),
        interface_area, gap, 0., "Explicit test interface")
    foundation_parameters = BornFoundationParameters(1., 1., "Declared hydrostatic support")
    foundation_reference = BornFoundationReference("hot-patch", "a"*64, "b"*64,
        "receiver", "hot-material", "hot-parent", 1, Fraction(1), basal_area, 0.,
        "Independent basal support measure")
    contact = BornNetworkAttachment(parameters, reference, activate_born_contact(parameters, reference),
        (1., 0., 0., -1., 0., 0., -1.), (0.,)*7)
    foundation = BornNetworkFoundation(foundation_parameters, foundation_reference,
        activate_born_foundation(foundation_parameters, foundation_reference), 6)
    original = solve_compressed_contact(columns=columns, law=law, area_m2=1.,
        gap_free_m=-1., jump_free_m=0., dt_s=1., water=0.).state
    materials = (SimpleNamespace(columns=columns, law=law, water=0.),) if old_contact else ()
    contacts = (contact,)
    if shared:
        second_reference = replace(reference, family_id="second-normal", interface_area_m2=2.*interface_area)
        contacts += (BornNetworkAttachment(parameters, second_reference,
            activate_born_contact(parameters, second_reference), contact.normal_operator, contact.shear_operator),)
    kwargs = dict(radius_m=1., rotation_dof_count=6,
        old_point_ids=("kept-original",) if old_contact else (), old_materials=materials,
        old_areas_m2=(1.,) if old_contact else (),
        old_anchor_operators=(((1., 0., 0., -1., 0., 0.), (0.,)*6),) if old_contact else (),
        old_stiffness=tuple((0.,)*6 for _ in range(6)), old_states=(original,) if old_contact else (),
        born_attachments=contacts, born_foundations=(foundation,), coordinates_before=(0.,)*7,
        drag_nm_s=tuple(tuple(float(i == j and i < 6) for j in range(7)) for i in range(7)),
        source_torque_nm=(0.,)*7, dt_s=1., decimal_precision=120)
    return kwargs


def test_joint_closed_solution_matches_independent_three_spring_equations():
    inputs = _case()
    result = solve_birth_contact_network(**inputs)
    # Symmetry gives q0=x, q3=-x and every other angular coordinate zero.
    # k_original=k_born=1/2; foundation=1.  The equations are
    # x + 1/2(-1+2x) + 1/2(-1+2x-w)=0,
    # w - 1/2(-1+2x-w)=0, hence x=5/16, w=-1/8.
    assert result.coordinates[0] == pytest.approx(5./16., rel=2e-14)
    assert result.coordinates[3] == pytest.approx(-5./16., rel=2e-14)
    assert result.coordinates[6] == pytest.approx(-1./8., rel=2e-14)
    old = inputs["old_states"][0]
    kept = result.original_point_states[0][1]
    assert kept.step_count == old.step_count+1
    assert kept.columns_signature == old.columns_signature
    assert kept.area_m2 == old.area_m2
    assert kept.elapsed_time_s == old.elapsed_time_s+1.
    assert result.born_contact_steps[0].state.step_count == 1
    assert max(result.per_coordinate_relative_residual) < 1e-90
    assert dict(result.diagnostics)["force_geometry_admission_authorized"] is False


def test_positive_born_gap_has_zero_traction_and_actual_equilibrium_changes_on_reclosure():
    open_inputs = _case(gap=1.)
    opened = solve_birth_contact_network(**open_inputs)
    assert opened.active_born_contacts == ()
    assert opened.born_contact_steps[0].state.normal_traction_pa == 0.
    assert opened.coordinates[0] == pytest.approx(.25)
    closed = solve_birth_contact_network(**_case(gap=-1.))
    assert closed.coordinates != opened.coordinates
    assert closed.born_contact_steps[0].state.normal_traction_pa < 0.


def test_shared_hot_patch_has_one_foundation_and_two_contact_histories():
    inputs = _case(shared=True, old_contact=False)
    result = solve_birth_contact_network(**inputs)
    # Two contacts share the same patch, total normal k=1/2+1=3/2.
    # x=-Fcontact, w=Fcontact gives x=3/11, w=-3/11.
    assert result.coordinates[0] == pytest.approx(3./11.)
    assert result.coordinates[6] == pytest.approx(-3./11.)
    assert len(result.born_foundation_steps) == 1
    assert len(result.born_contact_steps) == 2
    assert dict(result.diagnostics)["foundation_patches_counted_once"] is True
    duplicate = inputs["born_foundations"]*2
    with pytest.raises(ValueError, match="exactly once"):
        solve_birth_contact_network(**{**inputs, "born_foundations": duplicate,
            "coordinates_before": (0.,)*8,
            "drag_nm_s": tuple(tuple(float(i == j and i < 6) for j in range(8)) for i in range(8)),
            "source_torque_nm": (0.,)*8,
            "born_attachments": tuple(replace(item, normal_operator=item.normal_operator+(0.,),
                shear_operator=item.shear_operator+(0.,)) for item in inputs["born_attachments"])})


def test_tiny_foundation_is_not_removed_by_global_stiffness_or_residual_tolerance():
    inputs = _case(interface_area=1e-12, basal_area=1e-36, old_contact=False)
    result = solve_birth_contact_network(**inputs)
    with localcontext() as context:
        context.prec = 120
        # Symmetry independently condenses two angular dampers (relative
        # compliance 2) and the one foundation (compliance 1/k_f).
        a = Decimal.from_float(1e-12)/2
        b = Decimal.from_float(1e-36)
        expected_force = -a/(1+2*a+a/b)
        expected_x = -expected_force
        expected_w = expected_force/b
        q = tuple(Decimal(value) for value in result.coordinates_decimal)
        assert abs(q[0]-expected_x) < abs(expected_x)*Decimal("1e-85")
        assert abs(q[6]-expected_w) < abs(expected_w)*Decimal("1e-85")
        actual_gap = Decimal(-1)+q[0]-q[3]-q[6]
        assert actual_gap < 0
        assert actual_gap != 0
        # Rounded representatives lose the compression between unit-scale
        # values; this is recorded instead of silently becoming zero force.
        rounded_gap = -1.+result.coordinates[0]-result.coordinates[3]-result.coordinates[6]
        assert rounded_gap == 0.
        assert result.born_contact_steps[0].state.normal_traction_pa < 0.
        assert max(result.per_coordinate_relative_residual) < 1e-75


def test_decimal_coordinate_history_survives_next_actual_force_increment():
    inputs = _case(interface_area=1e-12, basal_area=1e-36, old_contact=False)
    first = solve_birth_contact_network(**inputs)
    next_inputs = {**inputs, "coordinates_before": first.coordinates_decimal,
        "born_attachments": tuple(replace(item, previous_state=step.state)
            for item, step in zip(inputs["born_attachments"], first.born_contact_steps)),
        "born_foundations": tuple(replace(item, previous_state=step.state)
            for item, step in zip(inputs["born_foundations"], first.born_foundation_steps)),
        "source_torque_nm": (1e-36, 0., 0., -1e-36, 0., 0., 0.)}
    second = solve_birth_contact_network(**next_inputs)
    assert second.coordinates_decimal != first.coordinates_decimal
    assert second.born_contact_steps[0].state.step_count == 2
    assert second.born_foundation_steps[0].state.step_count == 2
    assert max(second.per_coordinate_relative_residual) < 1e-70


def test_full_input_and_result_export_replays_and_keeps_independent_physical_measures():
    inputs = _case(interface_area=1e-12, basal_area=1e-36, old_contact=False)
    result = solve_birth_contact_network(**inputs)
    packet = birth_contact_network_to_dict(result, **inputs)
    raw = json.dumps(packet, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert json.loads(raw) == packet
    assert packet["born_contacts"][0]["reference"]["interface_area_m2"] == 1e-12
    assert packet["born_foundations"][0]["reference"]["basal_area_m2"] == 1e-36
    assert packet["result"]["coordinates_decimal"] == list(result.coordinates_decimal)
    with pytest.raises(ValueError, match="independent current input replay"):
        birth_contact_network_to_dict(replace(result, coordinates=(0.,)*7), **inputs)
    # Python equality aliases False and a floating zero; force evidence must
    # retain the exact independently reconstructed public types.
    aliased = replace(result, coordinates=result.coordinates[:1]+(False,)+result.coordinates[2:])
    assert aliased == result
    with pytest.raises(ValueError, match="independent current input replay"):
        birth_contact_network_to_dict(aliased, **inputs)


def test_old_friction_or_common_rotation_leak_is_rejected_without_mutation():
    inputs = _case()
    saved = inputs["old_states"]
    material = inputs["old_materials"][0]
    wrong = SimpleNamespace(columns=material.columns, law=replace(material.law, friction_dry=.1), water=0.)
    with pytest.raises(ValueError, match="frictionless"):
        solve_birth_contact_network(**{**inputs, "old_materials": (wrong,)})
    contact = inputs["born_attachments"][0]
    with pytest.raises(ValueError, match="common angular rotation"):
        solve_birth_contact_network(**{**inputs, "born_attachments":
            (replace(contact, normal_operator=(1., 0., 0., -.99, 0., 0., -1.)),)})
    assert inputs["old_states"] == saved


def test_atomic_domain_refusal_precedes_all_constitutive_adoption():
    inputs = _case()
    saved = tuple(item.previous_state for item in inputs["born_attachments"])
    def refuse(coordinates):
        raise UnresolvedBornContactNetwork("Explicit physical support refused")
    with pytest.raises(UnresolvedBornContactNetwork, match="physical support"):
        solve_birth_contact_network(**inputs, domain_check=refuse)
    assert tuple(item.previous_state for item in inputs["born_attachments"]) == saved


def test_nonbinary_common_rotation_operator_is_checked_without_decimal_rounding():
    inputs = _case()
    contact = inputs["born_attachments"][0]
    lever = 12345.6789012345
    result = solve_birth_contact_network(**{**inputs, "born_attachments":
        (replace(contact, normal_operator=(lever, 0., 0., -lever, 0., 0., -1.)),)})
    assert max(result.actual_per_coordinate_relative_residual) < 1e-12


def test_original_opening_is_not_converted_to_a_new_unilateral_original_law():
    inputs = _case(gap=1.)
    with pytest.raises(UnresolvedBornContactNetwork, match="original compressed contact"):
        solve_birth_contact_network(**{**inputs, "source_torque_nm": (10., 0., 0., -10., 0., 0., 0.)})


def test_existing_signed_slip_memory_and_paid_preload_survive_the_composed_solve():
    inputs = _case()
    material = inputs["old_materials"][0]
    first = solve_compressed_contact(columns=material.columns, law=material.law,
        area_m2=1., gap_free_m=-1., jump_free_m=.5, dt_s=1., water=0.).state
    history = solve_compressed_contact(columns=material.columns, law=material.law,
        area_m2=1., gap_free_m=-1., jump_free_m=-.25, dt_s=1., water=0., previous=first).state
    inputs = {**inputs, "old_states": (history,),
        "old_anchor_operators": (((1., 0., 0., -1., 0., 0.),
            (.25, 0., 0., -.25, 0., 0.)),)}
    result = solve_birth_contact_network(**inputs)
    final = result.original_point_states[0][1]
    expected_jump = history.jump_free_m+.25*(result.coordinates[0]-result.coordinates[3])
    assert final.plastic_slip_m == pytest.approx(expected_jump)
    assert final.cumulative_slip_m == pytest.approx(history.cumulative_slip_m+abs(expected_jump-history.plastic_slip_m))
    assert final.cumulative_slip_m > history.cumulative_slip_m
    assert final.step_count == history.step_count+1
    assert final.cumulative_external_work_j == pytest.approx(
        history.cumulative_external_work_j+result.original_point_steps[0].external_work_j)
    # The separately paid HOT activation is not reissued as source work.
    assert dict(result.diagnostics)["born_activation_work_is_interval_source_work"] is False


def test_exported_coordinate_and_local_work_error_bounds_are_rounded_outward():
    result = solve_birth_contact_network(**_case(interface_area=.3, basal_area=.3, old_contact=False))
    with localcontext() as context:
        context.prec = 200
        for representative, precise, bound in zip(result.coordinates,
                result.coordinates_decimal, result.coordinate_representative_error):
            actual_error = abs(Decimal.from_float(representative)-Decimal(precise))
            assert Decimal.from_float(bound) >= actual_error
    assert _upper_float(Decimal("1e-1000")) > 0.


def test_actual_native_original_brent_failure_has_an_exact_same_anchor_inviscid_limit():
    # Captured from the real 16-point original state during native preflight;
    # this is a complete accepted history, not a zero-time replacement.
    previous = state_from_dict(dict(area_m2=54640911.29997197,
        bulk_energy_j=516494771.4245693,
        columns_signature=[1000., 1e9, 4e8, 1000., 1e9, 4e8],
        contact_energy_j=258247385.71228465,
        cumulative_external_work_j=166886526460.82748,
        cumulative_friction_work_j=0., cumulative_numerical_dissipation_j=166111784303.69064,
        cumulative_slip_m=.0283135964900136, cumulative_viscous_work_j=0.,
        elapsed_time_s=2e6, gap_free_m=-.009223489389366069,
        gap_m=-.003074496463122023, jump_free_m=-.0283135964900136,
        jump_m=-.0283135964900136,
        law_signature=[1e6, 1e6, 2e6, 2e7, 0., 0., 0., .5, 0.],
        plastic_slip_m=-.0283135964900136, step_count=2,
        traction_normal_pa=-3074.4964631220228, traction_shear_pa=0., water=.7))
    columns = ElasticColumnParameters(1000., 1e9, 4e8, 1000., 1e9, 4e8)
    law = ContactLawParameters(normal_stiffness_pa_m=1e6, tangential_stiffness_pa_m=1e6,
        tensile_strength_pa=2e6, fracture_energy_j_m2=2e7, cohesion_pa=0.,
        friction_dry=0., friction_wet=0., wet_cohesion_fraction=.5, viscosity_pa_s_m=0.)
    material = SimpleNamespace(columns=columns, law=law, water=.7)
    gap, jump, duration = -.004846855612926414, -.00831059089542696, 1e6
    with pytest.raises(ValueError, match=r"f\(a\) and f\(b\)"):
        solve_compressed_contact(columns=columns, law=law, area_m2=previous.area_m2,
            gap_free_m=gap, jump_free_m=jump, dt_s=duration, water=.7, previous=previous)
    final, used_fallback = _advance_original_closed_inviscid(material,
        previous.area_m2, gap, jump, previous, duration)
    assert used_fallback is True
    assert final.state.gap_free_m == gap
    assert final.state.jump_free_m == jump
    assert final.state.plastic_slip_m == jump
    assert final.state.traction_shear_pa == 0.
    assert final.state.columns_signature == previous.columns_signature
    assert final.state.law_signature == previous.law_signature
    assert final.state.area_m2 == previous.area_m2
    assert final.state.elapsed_time_s == 3e6
    assert final.state.step_count == 3
    assert final.state.cumulative_slip_m == pytest.approx(previous.cumulative_slip_m+abs(jump-previous.plastic_slip_m))
    assert final.energy_balance_relative_error < 2e-10
    assert final.force_relative_residual < 2e-10
    assert final.state.cumulative_external_work_j == pytest.approx(previous.cumulative_external_work_j+final.external_work_j)
