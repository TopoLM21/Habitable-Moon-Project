"""Independent arithmetic-range and schema checks for native rigid rays."""
from copy import deepcopy
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_EVEN, localcontext

import numpy as np
import pytest

from tectonics.geometric_intrinsic_geometry import (
    IntrinsicGeometry,
    RotationIncrement,
    intrinsic_geometry_from_dict,
    intrinsic_geometry_to_dict,
)
from tectonics.geometric_surface import GeometricFragment
from test_geometric_intrinsic_surface import ACTUAL, _area, _atan, _sine_cosine
from test_geometric_precision import NEXT_DT_MYR, thin_partition


OCTANT = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))


def _thin():
    return next(fragment for fragment in thin_partition().fragments
                if fragment.fragment_id == "thin")


def test_arbitrary_pose_preserves_thin_shape_when_float_projection_collapses():
    fragment = _thin()
    geometry = IntrinsicGeometry(fragment.polygon).rotated((.023, -.037, .049), .71)
    projected = geometry.world_polygon()
    assert len(set(projected)) < len(geometry.body_polygon)
    assert geometry.area_sr > 0.
    with localcontext() as context:
        context.prec = 175
        measured = _area(geometry.decimal_world_rays())
        assert measured > 0
        expected = Decimal.from_float(geometry.area_sr)
        assert abs(measured/expected-1) < Decimal("2e-13")
    restored = intrinsic_geometry_from_dict(intrinsic_geometry_to_dict(geometry))
    assert restored == geometry
    assert restored.area_sr == geometry.area_sr
    assert restored.decimal_world_rays() == geometry.decimal_world_rays()
    with pytest.raises(ValueError):
        GeometricFragment(fragment.fragment_id, projected, fragment.parcel)


def test_smallest_nonzero_binary64_angular_velocity_is_not_replaced_by_zero():
    smallest = np.nextafter(0., 1.)
    geometry = IntrinsicGeometry(OCTANT).rotated((float(smallest), 0., 0.), 1.)
    assert geometry.rotations[0].omega_rad_per_myr[0] == smallest
    assert geometry.rotations[0].omega_rad_per_myr[0] != 0.
    native = geometry.decimal_world_rays()
    # +y rotates toward +z, even though squaring omega in binary64 gives zero.
    with localcontext() as context:
        context.prec = 400
        expected = Decimal.from_float(float(smallest))
        assert native[1][2] > 0
        assert abs(native[1][2]/expected-1) < Decimal("1e-65")
    projected = geometry.world_polygon()
    assert projected[1][2] == smallest
    assert geometry.area_sr == IntrinsicGeometry(OCTANT).area_sr


def test_overflowing_binary64_angle_matches_independent_machin_reduction():
    huge = 1e308
    with np.errstate(over="ignore"):
        assert not np.isfinite(np.float64(huge)*np.float64(huge))
    geometry = IntrinsicGeometry(OCTANT).rotated((0., 0., huge), huge)
    # Independent Machin pi, direct argument reduction and Taylor trig differ
    # from the production Gauss-Legendre/half-angle projection implementation.
    with localcontext() as context:
        context.prec = 850
        pi = Decimal(16)*_atan(Decimal(1)/5)-Decimal(4)*_atan(Decimal(1)/239)
        angle = Decimal.from_float(huge)*Decimal.from_float(huge)
        period = Decimal(2)*pi
        turns = (angle/period).to_integral_value(rounding=ROUND_HALF_EVEN)
        reduced = angle-turns*period
        sine, cosine = _sine_cosine(reduced)
        expected = ((float(cosine), float(sine), 0.),
                    (-float(sine), float(cosine), 0.), (0., 0., 1.))
    actual = np.asarray(geometry.world_polygon())
    for row, reference in zip(actual, expected):
        for value, target in zip(row, reference):
            assert abs(value-target) <= 4.*abs(np.spacing(target))
    assert geometry.area_sr == IntrinsicGeometry(OCTANT).area_sr
    assert geometry.rotations[0].omega_rad_per_myr == (0., 0., huge)
    assert geometry.rotations[0].dt_myr == huge


def test_native_projection_does_not_depend_on_callers_decimal_context():
    geometry = IntrinsicGeometry(_thin().polygon).rotated(ACTUAL, NEXT_DT_MYR)
    geometry = geometry.rotated((.023, -.037, .049), .71)
    expected_rays = geometry.decimal_world_rays()
    expected_projection = geometry.world_polygon()
    expected_precision = geometry.projection_precision
    with localcontext() as context:
        context.prec = 7
        context.rounding = ROUND_DOWN
        context.Emax = 20
        context.Emin = -20
        context.clamp = 1
        restored = intrinsic_geometry_from_dict(intrinsic_geometry_to_dict(geometry))
        assert restored.decimal_world_rays() == expected_rays
        assert restored.world_polygon() == expected_projection
        assert restored.projection_precision == expected_precision


@pytest.mark.parametrize("corruption", ["body_bool", "body_string", "body_nonunit",
                                        "body_duplicate", "omega_bool", "omega_nan",
                                        "dt_string", "dt_zero", "extra_geometry_field",
                                        "extra_pose_field", "wrong_version"])
def test_standalone_geometry_reader_validates_complete_native_schema(corruption):
    geometry = IntrinsicGeometry(OCTANT).rotated(ACTUAL, NEXT_DT_MYR)
    payload = deepcopy(intrinsic_geometry_to_dict(geometry))
    if corruption == "body_bool":
        payload["body_rays"][0][0] = True
    elif corruption == "body_string":
        payload["body_rays"][0][0] = "1"
    elif corruption == "body_nonunit":
        payload["body_rays"][0][0] = 2.
    elif corruption == "body_duplicate":
        payload["body_rays"][1] = payload["body_rays"][0].copy()
    elif corruption == "omega_bool":
        payload["rotations"][0]["omega_rad_per_myr"][0] = True
    elif corruption == "omega_nan":
        payload["rotations"][0]["omega_rad_per_myr"][0] = float("nan")
    elif corruption == "dt_string":
        payload["rotations"][0]["dt_myr"] = "1"
    elif corruption == "dt_zero":
        payload["rotations"][0]["dt_myr"] = 0.
    elif corruption == "extra_geometry_field":
        payload["projected_area_is_authoritative"] = True
    elif corruption == "extra_pose_field":
        payload["rotations"][0]["project_omega_to_axis"] = True
    else:
        payload["version"] = "unsupported-native-geometry"
    with pytest.raises(ValueError):
        intrinsic_geometry_from_dict(payload)


def test_standalone_writer_revalidates_forged_body_and_increment():
    geometry = IntrinsicGeometry(OCTANT).rotated(ACTUAL, NEXT_DT_MYR)
    object.__setattr__(geometry, "body_rays", ((2., 0., 0.),)+OCTANT[1:])
    with pytest.raises(ValueError):
        intrinsic_geometry_to_dict(geometry)
    geometry = IntrinsicGeometry(OCTANT).rotated(ACTUAL, NEXT_DT_MYR)
    object.__setattr__(geometry.rotations[0], "omega_rad_per_myr", (True, 0., 0.))
    with pytest.raises(ValueError):
        intrinsic_geometry_to_dict(geometry)


def test_zero_angular_velocity_is_an_explicit_finite_identity_increment():
    geometry = IntrinsicGeometry(OCTANT).rotated((0., 0., 0.), 1.)
    assert geometry.rotations == (RotationIncrement((0., 0., 0.), 1.),)
    assert geometry.world_polygon() == OCTANT
    assert geometry.area_sr == IntrinsicGeometry(OCTANT).area_sr
