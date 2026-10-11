"""Actual finite volume, whole-depth collision and authentic source coverage."""
from dataclasses import replace
from fractions import Fraction as F
import math

import pytest

from tectonics.geometric_native_pose import FractionInterval as I
from tectonics.geometric_native_solid_layer_placement import (
    LocalDippingLayerChart, bound_local_dipping_layer_separation,
    build_native_contact_solid_layer_chart, local_layer_point,
    local_layer_world_point, native_component_solid_thickness_m,
    require_complete_physical_contact_face,
)


ZERO = (F(0),)*3
ZERO_TRANSLATION = (I(0),)*3


def chart(**changes):
    base = dict(strike_length_m=F(10), donor_width_m=F(3),
        receiver_width_m=F(4), donor_thickness_m=F(2000),
        receiver_thickness_m=F(2000), cot_dip=F(2),
        force_contact_height_m=F(2000), provenance="Explicit finite material prism test")
    return LocalDippingLayerChart(**dict(base, **changes))


def separation(value, **changes):
    arguments = dict(donor_axis=ZERO, receiver_axis=ZERO,
        relative_translation_receiver_bounds_m=ZERO_TRANSLATION,
        time_fraction_bounds=I(0, 1))
    return bound_local_dipping_layer_separation(value, **dict(arguments, **changes))


def test_two_full_layers_share_exact_dipping_face_at_every_depth():
    value = chart()
    for s in (F(0), F(7), F(10)):
        for depth in (F(0), F(1, 10**80), F(1000), F(2000)):
            points = tuple(local_layer_point(value, bank=bank, strike_m=s,
                bank_coordinate_m=F(0), depth_m=depth) for bank in ("donor", "receiver"))
            assert points[0] == points[1] == (s, depth*2, -depth)
            assert sum(a*b for a, b in zip(value.normal_covector, points[0])) == 0
    donor = local_layer_point(value, bank="donor", strike_m=F(0), bank_coordinate_m=F(-3), depth_m=F(2000))
    receiver = local_layer_point(value, bank="receiver", strike_m=F(0), bank_coordinate_m=F(4), depth_m=F(2000))
    assert sum(a*b for a, b in zip(value.normal_covector, donor)) == -3
    assert sum(a*b for a, b in zip(value.normal_covector, receiver)) == 4


def test_exact_unit_shear_determinant_and_complete_volume_no_removed_wedge():
    value = chart()
    m = value.depth_shear_matrix
    determinant = (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
        -m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
        +m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))
    assert determinant == value.volume_jacobian == 1
    assert value.donor_volume_m3 == 60_000
    assert value.receiver_volume_m3 == 80_000
    assert value.physical_horizontal_extent_m == I(-3, 4004)


def test_narrow_bank_is_valid_and_does_not_need_depth_times_cot_width():
    width = F(1, 10**70)
    value = chart(donor_width_m=width, receiver_width_m=width)
    assert value.donor_volume_m3 == 20_000*width > 0
    assert separation(value).local_prism_intersection_excluded
    assert value.physical_horizontal_extent_m.hi == 4000+width


def test_incomplete_one_kilometre_force_height_is_not_full_hot_face():
    value = chart(force_contact_height_m=F(1000))
    assert value.common_physical_face_height_m == 2000
    assert not value.whole_physical_face_force_coverage
    assert separation(value).local_prism_intersection_excluded
    with pytest.raises(ValueError, match="complete finite-thickness physical face"):
        require_complete_physical_contact_face(value)
    assert require_complete_physical_contact_face(chart()) == 2000


def test_common_face_uses_actual_minimum_layer_thickness():
    value = chart(donor_thickness_m=F(7000), receiver_thickness_m=F(2000))
    assert value.common_physical_face_height_m == 2000
    assert value.donor_volume_m3 == 210_000
    assert require_complete_physical_contact_face(value) == 2000


def test_millimetre_downward_separation_preserves_full_two_kilometre_layers():
    value = chart()
    result = separation(value, relative_translation_receiver_bounds_m=(I(0), I(0), I(-F(1, 1000))))
    assert result.local_prism_intersection_excluded
    assert result.common_face_normal_gap_bounds_m.lo > F(8, 10_000)
    assert result.common_face_normal_gap_bounds_m.hi < F(1, 1000)
    # A radial rectangular column at the same top displacement would overlap
    # the receiver by almost its full thickness.  The actual shared dipping
    # face, rather than an artificial two-kilometre offset, makes this valid.
    assert value.receiver_thickness_m-F(1, 1000) > 1999
    assert value.donor_volume_m3 == 60_000
    assert not result.physical_material_admission_authorized


def test_positive_lateral_crossing_is_not_discarded_for_tiny_width():
    width = F(1, 10**100)
    value = chart(donor_width_m=width, receiver_width_m=width)
    result = separation(value, relative_translation_receiver_bounds_m=(I(0), I(width/2), I(0)))
    assert not result.local_prism_intersection_excluded
    assert result.common_face_normal_gap_bounds_m.hi < 0
    assert result.donor_whole_solid_signed_plane_bounds_m.hi > 0


def test_common_rigid_rotation_is_objective_without_unit_vector_cancellation():
    axis = (F(1, 100), F(-2, 100), F(3, 100))
    result = separation(chart(), donor_axis=axis, receiver_axis=axis)
    assert result.common_face_normal_gap_bounds_m == I(0)
    assert result.local_prism_intersection_excluded


def test_whole_time_translation_must_pass_not_just_safe_endpoint():
    value = chart()
    endpoint = separation(value, relative_translation_receiver_bounds_m=(I(0), I(-1), I(0)), time_fraction_bounds=I(1))
    whole = separation(value, relative_translation_receiver_bounds_m=(I(0), I(-1, 1), I(0)))
    assert endpoint.local_prism_intersection_excluded
    assert not whole.local_prism_intersection_excluded
    assert whole.time_fraction_bounds == I(0, 1)


def test_finite_rotation_changes_face_gap_through_the_entire_depth():
    value = chart()
    axis = (F(1, 1000), F(0), F(0))
    result = separation(value, donor_axis=axis, time_fraction_bounds=I(1), precision=96)
    assert not result.local_prism_intersection_excluded
    assert result.common_face_normal_gap_bounds_m.lo < -4
    # A top-edge-only test sees exact touching.  At depth the same finite
    # rotation penetrates the receiver halfspace; no radius-sized tolerance.
    assert result.common_face_normal_gap_bounds_m.hi == 0
    theta, depth = 0.001, 2000.
    y, z = depth*2, -depth
    actual_signed = ((math.cos(theta)*y-math.sin(theta)*z)
        +2*(math.sin(theta)*y+math.cos(theta)*z))/math.sqrt(5)
    assert float(result.donor_whole_solid_signed_plane_bounds_m.lo) <= actual_signed
    assert actual_signed <= float(result.donor_whole_solid_signed_plane_bounds_m.hi)


def test_face_safe_alone_cannot_hide_rotated_donor_interior():
    # A half turn reverses the inward material direction.  With a sufficiently
    # wide body, every source face point can be left of the plane while its
    # interior crosses into the receiving halfspace.  The complete body bound
    # retains the independently declared width and catches that crossing.
    value = chart(donor_width_m=F(100_000), donor_thickness_m=F(1), receiver_thickness_m=F(1))
    result = separation(value, donor_axis=(F(0), F(0), F(3)),
        relative_translation_receiver_bounds_m=(I(0), I(-10), I(0)),
        time_fraction_bounds=I(1), precision=96, order=48)
    assert result.common_face_normal_gap_bounds_m.lo > 0
    assert result.donor_inward_coordinate_plane_coefficient.hi < 0
    assert not result.local_prism_intersection_excluded
    assert result.donor_whole_solid_signed_plane_bounds_m.hi > 0


def test_exact_world_frame_places_actual_point_and_rejects_nonorthonormal_frame():
    value = chart()
    args = dict(bank="donor", strike_m=F(7), bank_coordinate_m=F(-2), depth_m=F(3),
        origin_m=(F(10), F(20), F(30)), strike_direction=(F(0), F(1), F(0)),
        horizontal_direction=(F(-1), F(0), F(0)), outward_direction=(F(0), F(0), F(-1)))
    assert local_layer_world_point(value, **args) == (F(6), F(27), F(33))
    with pytest.raises(ValueError, match="exactly orthonormal"):
        local_layer_world_point(value, **dict(args, horizontal_direction=(F(-2), F(0), F(0))))


@pytest.mark.parametrize("field,bad", [("donor_width_m", None), ("receiver_width_m", F(0)),
    ("donor_thickness_m", 2000.), ("cot_dip", True), ("force_contact_height_m", F(-1)),
    ("provenance", "")])
def test_missing_or_aliased_physical_declaration_refuses(field, bad):
    with pytest.raises(ValueError):
        chart(**{field: bad})


def test_unknown_material_point_bank_depth_and_wrong_clock_refuse():
    value = chart()
    args = dict(bank="donor", strike_m=F(0), bank_coordinate_m=F(0), depth_m=F(2001))
    with pytest.raises(ValueError, match="outside"):
        local_layer_point(value, **args)
    with pytest.raises(ValueError, match="bank"):
        local_layer_point(value, **dict(args, bank="hot"))
    with pytest.raises(ValueError, match="clock"):
        separation(value, time_fraction_bounds=I(-1, 1))
    with pytest.raises(ValueError, match="exact three-component"):
        separation(value, donor_axis=(0., 0., 0.))


@pytest.fixture(scope="module")
def native_case():
    from tests.test_geometric_native_birth_contact_support import get_birth_support_case
    return get_birth_support_case()


def test_genuine_source_hot_thickness_and_declared_contact_height(native_case):
    support = native_case["support"]
    for carrier in support.interface_carriers:
        value = build_native_contact_solid_layer_chart(support, carrier,
            donor_width_m=F(1, 100), receiver_width_m=F(1, 100),
            provenance="New local prism declaration; global embedding unresolved")
        assert value.receiver_thickness_m == 2000
        if carrier.donor_component.material_kind == "hot_birth":
            assert value.donor_thickness_m == 2000
        else:
            parent = next(row for row in support.source_epoch.source_material_state.journal.source_surface.fragments
                          if row.fragment_id == carrier.donor_component.original_parent_id)
            assert value.donor_thickness_m == sum(map(F.from_float, parent.parcel.specific_properties[:2]))*1000
        assert value.force_contact_height_m == 1000
        assert not value.whole_physical_face_force_coverage
        with pytest.raises(ValueError, match="complete finite-thickness"):
            require_complete_physical_contact_face(value)
        assert not value.whole_spherical_solid_embedding_certified


def test_whole_typed_native_source_and_carrier_tamper_refuse(native_case):
    support, epoch = native_case["support"], native_case["epoch"]
    carrier = support.interface_carriers[0]
    with pytest.raises(ValueError):
        native_component_solid_thickness_m(replace(epoch, epoch_sha256="0"*64), carrier.receiver_component.component_id)
    with pytest.raises(ValueError, match="unique physical"):
        native_component_solid_thickness_m(epoch, "missing-material")
    with pytest.raises(ValueError, match="complete physical carrier"):
        build_native_contact_solid_layer_chart(support, replace(carrier, float_reference_interface_area_m2=1.),
            donor_width_m=F(1), receiver_width_m=F(1), provenance="Source replacement must fail")

