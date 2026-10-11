"""Explicit HOT support retains tiny measures and unresolved birth fronts."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tectonics.geometric_contact_mechanics import ElasticColumnParameters
from tectonics.genesis_contact_law import ContactLawParameters
from tectonics.geometric_native_birth_contact import BornContactParameters, BornFoundationParameters
from tectonics.geometric_native_birth_contact_support import (
    NativeHotFoundationDeclaration, NativeBirthInterfaceDeclaration,
    declare_native_birth_contact_support, native_birth_contact_support_to_dict,
    native_birth_contact_support_from_dict, native_birth_point_port,
    native_birth_point_reactions, native_born_contact_references,
    native_born_foundation_references,
)
from tectonics.geometric_native_polygons import native_plane_to_dict
from tests.test_geometric_native_implicit_epoch_admission import get_actual_newborn_geometry


@lru_cache(maxsize=1)
def get_birth_support_case():
    geometry = get_actual_newborn_geometry()
    epoch = geometry.source_epoch
    foundations = tuple(NativeHotFoundationDeclaration(row.component_id,
        BornFoundationParameters(2800., 1.5,
            "Explicit hydrostatic Winkler foundation; no thermal inference"),
        "lower", 0., "Entire actual HOT birth mask; fixed lower constitutive reference area")
        for row in sorted(epoch.components, key=lambda row: row.component_id)
        if row.material_kind == "hot_birth")
    columns = ElasticColumnParameters(1000., 1e9, 4e8, 1000., 1e9, 4e8)
    parameters = BornContactParameters(columns,
        ContactLawParameters(friction_dry=0., friction_wet=0., cohesion_pa=0., viscosity_pa_s_m=0.),
        "Declared unbonded frictionless HOT interface, independent from cold history")
    interfaces = tuple(NativeBirthInterfaceDeclaration("born-shared-"+str(index),
        row.donor_component_id, row.receiver_component_id,
        row.donor_piece_sequence, row.receiver_piece_sequence, 0, parameters,
        math.pi/6, 1000., "lower", -0.01, 0.,
        "Explicit paid compressed birth reference on the actual native shared arc")
        for index, row in enumerate(geometry.positive_contacts) if row.immediate_incoming_edges)
    support = declare_native_birth_contact_support(epoch, foundations, interfaces,
        incoming_geometry=geometry)
    return dict(epoch=epoch, geometry=geometry, foundations=foundations,
        interfaces=interfaces, support=support)


@pytest.fixture(scope="module")
def case():
    return get_birth_support_case()


def test_all_five_entire_hot_masks_have_unique_independent_foundations(case):
    support = case["support"]
    hot = {row.component_id: row for row in case["epoch"].components if row.material_kind == "hot_birth"}
    assert len(support.foundations) == 5
    assert {row.component.component_id for row in support.foundations} == set(hot)
    radius = F.from_float(case["epoch"].radius_km)*1000
    for row in support.foundations:
        assert row.component == hot[row.component.component_id]
        assert row.area_bounds_m2 == row.component.area_bounds_sr*radius**2
        assert row.fixed_reference_area_m2 == row.area_bounds_m2.lo > 0
        assert row.stiffness_n_m == 2800*F(3, 2)*row.fixed_reference_area_m2
        assert row.material_birth_time_myr == case["epoch"].time_myr
        assert row.float_reference_area_error_m2 == abs(F.from_float(row.float_reference_area_m2)-row.fixed_reference_area_m2)
    assert any(row.fixed_reference_area_m2 < F(1, 10**35) for row in support.foundations)


def test_native_shared_arcs_retain_r1x_and_unrotated_x_planes(case):
    carriers = case["support"].interface_carriers
    assert len(carriers) == 2
    cold = next(row for row in carriers if row.donor_component.material_kind == "original_live")
    hot = next(row for row in carriers if row.donor_component.material_kind == "hot_birth")
    assert cold.edge.length_radians == pytest.approx(5.4126014385191013e-20, rel=1e-14, abs=0.)
    assert hot.edge.length_radians == pytest.approx(6.086459240782105e-21, rel=1e-14, abs=0.)
    assert native_plane_to_dict(cold.receiver_inward_plane)["rotations"]
    assert not native_plane_to_dict(hot.receiver_inward_plane)["rotations"]
    assert len(cold.foundation_component_ids) == 1
    assert len(hot.foundation_component_ids) == 2
    assert cold.receiver_component.component_id == hot.receiver_component.component_id


def test_generic_references_keep_patch_ids_and_distinct_material_measures(case):
    foundations = native_born_foundation_references(case["support"])
    contacts = native_born_contact_references(case["support"])
    assert len(foundations) == 5 and len(contacts) == 2
    assert len({row.patch_id for row in foundations}) == 5
    assert contacts[0].receiver_cohort_id == contacts[1].receiver_cohort_id
    for generic, native in zip(foundations, case["support"].foundations):
        assert generic.patch_id == native.component.component_id
        assert generic.basal_area_m2 == native.float_reference_area_m2
        assert generic.material_id == native.component.material_id
        assert generic.birth_time_myr == native.material_birth_time_myr
    for generic, native in zip(contacts, case["support"].interface_carriers):
        assert generic.plane == native.receiver_inward_plane
        assert generic.interface_area_m2 == native.float_reference_interface_area_m2
        assert generic.source_epoch_sha256 == case["epoch"].epoch_sha256


def test_inclined_interface_measure_has_sin_factor_and_is_not_overlap_area(case):
    radius = F.from_float(case["epoch"].radius_km)*1000
    for carrier in case["support"].interface_carriers:
        expected = radius*1000/F.from_float(math.sin(math.pi/6))
        assert carrier.inclined_measure_density_m2_per_radian == expected
        assert carrier.interface_area_bounds_m2 == expected*carrier.native_angle_bounds_radians
        assert carrier.fixed_reference_interface_area_m2 == expected*carrier.native_angle_bounds_radians.lo > 0
        assert carrier.float_reference_interface_area_error_m2 == abs(
            F.from_float(carrier.float_reference_interface_area_m2)-carrier.fixed_reference_interface_area_m2)
        assert carrier.native_angle_bounds_radians.width < carrier.native_angle_bounds_radians.lo/F(10**40)
    raw = native_birth_contact_support_to_dict(case["support"])
    assert all(not row["overlap_area_used_as_interface_measure"] for row in raw["interface_carriers"])


def test_two_vertex_fronts_remain_positive_unresolved_without_initial_contact_area(case):
    support = case["support"]
    assert len(support.unresolved_fronts) == 2
    assert all(row.endpoint_overlap_area_bounds_sr.lo > 0 and row.initial_shared_arc_count == 0
        for row in support.unresolved_fronts)
    raw = native_birth_contact_support_to_dict(support)
    assert raw["known_positive_witness_count"] == 4
    assert not raw["complete_all_pairs_temporal_coverage"]
    assert all(row["front_potential_retained"] and not row["endpoint_overlap_converted_to_contact_area"]
        for row in raw["unresolved_fronts"])
    assert "force_admission_authorized" not in raw


def test_hot_foundation_does_not_invent_km_cantilever_clamp(case):
    raw = native_birth_contact_support_to_dict(case["support"])
    assert all(row["whole_implicit_material_mask_retained"]
        and not row["cold_cantilever_clamp_required"] and not row["reservoir_area_debit"]
        for row in raw["foundations"])
    assert all("width_m" not in row["declaration"] for row in raw["foundations"])
    assert case["epoch"].paired_corrections[0].area_bounds_sr.lo > 0


def test_missing_hot_patch_and_duplicate_patch_are_refused(case):
    for declarations in (case["foundations"][:-1], case["foundations"]+(case["foundations"][-1],)):
        with pytest.raises(ValueError, match="exactly once"):
            declare_native_birth_contact_support(case["epoch"], declarations,
                case["interfaces"], incoming_geometry=case["geometry"])


def test_roundtrip_uses_independent_whole_epoch_geometry_and_declarations(case):
    raw = native_birth_contact_support_to_dict(case["support"])
    replay = native_birth_contact_support_from_dict(deepcopy(raw), expected_epoch=case["epoch"],
        expected_foundation_declarations=case["foundations"],
        expected_interface_declarations=case["interfaces"], expected_incoming_geometry=case["geometry"])
    assert native_birth_contact_support_to_dict(replay) == raw
    changed = deepcopy(raw)
    changed["foundations"][0]["fixed_constitutive_reference_area_m2"] = [1, 1]
    changed["support_sha256"] = hashlib.sha256(json.dumps({key: value for key, value in changed.items()
        if key != "support_sha256"}, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    with pytest.raises(ValueError, match="independent complete epoch"):
        native_birth_contact_support_from_dict(changed, expected_epoch=case["epoch"],
            expected_foundation_declarations=case["foundations"],
            expected_interface_declarations=case["interfaces"], expected_incoming_geometry=case["geometry"])


def test_typed_source_mutation_and_changed_foundation_parameters_refuse_saved_support(case):
    changed = replace(case["epoch"], time_myr=case["epoch"].time_myr+F(1, 10**50))
    with pytest.raises(ValueError):
        declare_native_birth_contact_support(changed, case["foundations"],
            case["interfaces"], incoming_geometry=case["geometry"])
    declarations = (replace(case["foundations"][0], reference_area_endpoint="upper"),)+case["foundations"][1:]
    with pytest.raises(ValueError, match="independent complete epoch"):
        native_birth_contact_support_from_dict(native_birth_contact_support_to_dict(case["support"]),
            expected_epoch=case["epoch"], expected_foundation_declarations=declarations,
            expected_interface_declarations=case["interfaces"], expected_incoming_geometry=case["geometry"])


@pytest.mark.parametrize("mutation", ("tuple", "bool", "nonfinite"))
def test_public_support_dto_rejects_nonliteral_and_changed_numeric_types(case, mutation):
    raw = native_birth_contact_support_to_dict(case["support"])
    assert json.loads(json.dumps(raw, allow_nan=False)) == raw
    if mutation == "tuple":
        row = raw["foundations"][0]["native_basal_area_bounds_m2"]
        row["lo"] = tuple(row["lo"])
    elif mutation == "bool":
        raw["foundations"][0]["declaration"]["radial_birth_m"] = False
    else:
        raw["foundations"][0]["float_reference_area_m2"] = float("nan")
    with pytest.raises(ValueError):
        native_birth_contact_support_from_dict(raw, expected_epoch=case["epoch"],
            expected_foundation_declarations=case["foundations"],
            expected_interface_declarations=case["interfaces"], expected_incoming_geometry=case["geometry"])


def _port(rotation=np.eye(3)):
    sub = np.array((0., 0., -1.))
    over = np.array((0., 0., -1.))
    horizontal = np.array((-1., 0., 0.))
    return native_birth_point_port(donor_owner=1, receiver_owner=0,
        donor_radial=tuple(map(float, rotation@sub)), receiver_radial=tuple(map(float, rotation@over)),
        horizontal_sub_to_over=tuple(map(float, rotation@horizontal)), dip_radians=math.pi/6,
        radius_m=100000., donor_displacement_m=0.013, receiver_displacement_m=-0.02,
        dimension=19, angular_owner_dofs=((0, (0, 1, 2)), (1, (3, 4, 5))),
        donor_radial_coupling=((16, 1.),), receiver_radial_coupling=((14, 1.),))


def test_objective_ports_annihilate_common_rotation_with_shared_hot_coordinates():
    port = _port()
    for row in (port.normal_operator, port.shear_operator):
        assert row[:3] == tuple(-value for value in row[3:6])
        assert row[6:14] == (0.,)*8
        assert row[14] == -row[16]
    # Both interfaces may use the same HOT receiver14 and donorHOT16 without
    # allocating a new foundation coordinate or adding another foundation.
    assert port.receiver_radial_coupling == ((14, 1.),)


def test_physical_pair_couple_reproduces_operator_virtual_work():
    port = _port()
    area, traction = 3e-13, (-5000., 0.)
    reactions = native_birth_point_reactions(port, interface_area_m2=area, traction_pa=traction)
    expected = -area*(traction[0]*np.asarray(port.normal_operator)+traction[1]*np.asarray(port.shear_operator))
    np.testing.assert_allclose(reactions["generalized_force"], expected, rtol=5e-15, atol=0.)
    np.testing.assert_allclose(np.asarray(reactions["force_torque_on_over_nm"])
        +reactions["carrier_couple_on_over_nm"], reactions["torque_on_over_nm"], rtol=5e-15, atol=0.)
    increments = np.linspace(-0.03, 0.05, 19)
    assert float(np.asarray(reactions["generalized_force"])@increments) == pytest.approx(
        float(expected@increments), rel=5e-15, abs=0.)


def test_world_rotation_rotates_angular_operator_and_reactions_only():
    rotation = Rotation.from_rotvec((0.3, -0.2, 0.5)).as_matrix()
    original, rotated = _port(), _port(rotation)
    for before, after in zip((original.normal_operator, original.shear_operator),
            (rotated.normal_operator, rotated.shear_operator)):
        np.testing.assert_allclose(after[:3], rotation@before[:3], rtol=1e-14, atol=1e-11)
        np.testing.assert_allclose(after[3:6], rotation@before[3:6], rtol=1e-14, atol=1e-11)
        np.testing.assert_allclose(after[6:], before[6:], rtol=1e-14, atol=1e-11)
    first = native_birth_point_reactions(original, interface_area_m2=1e-12, traction_pa=(-30., 0.))
    second = native_birth_point_reactions(rotated, interface_area_m2=1e-12, traction_pa=(-30., 0.))
    np.testing.assert_allclose(second["force_on_sub_n"], rotation@first["force_on_sub_n"], rtol=1e-14, atol=1e-25)


def test_typed_port_operator_mutation_cannot_reuse_physical_reaction_binding():
    port = _port()
    changed = replace(port, normal_operator=(0.,)*len(port.normal_operator))
    with pytest.raises(ValueError, match="complete objective world-frame port replay"):
        native_birth_point_reactions(changed, interface_area_m2=1e-12, traction_pa=(-30., 0.))


@pytest.mark.parametrize("change", ({"donor_radial_coupling": ((0, 1.),)},
    {"receiver_radial_coupling": ((14, 1.), (14, 1.))}, {"dip_radians": 0.}))
def test_invalid_port_coordinate_bindings_refuse(change):
    arguments = dict(donor_owner=1, receiver_owner=0, donor_radial=(0., 0., -1.),
        receiver_radial=(0., 0., -1.), horizontal_sub_to_over=(-1., 0., 0.), dip_radians=math.pi/6,
        radius_m=100000., donor_displacement_m=0., receiver_displacement_m=0., dimension=19,
        angular_owner_dofs=((0, (0, 1, 2)), (1, (3, 4, 5))),
        donor_radial_coupling=((16, 1.),), receiver_radial_coupling=((14, 1.),))
    arguments.update(change)
    with pytest.raises(ValueError):
        native_birth_point_port(**arguments)
