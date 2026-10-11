"""Coordinate rounding stays local and retains thin-lambda sensitivities."""
from copy import deepcopy

import numpy as np
import pytest

from analysis.genesis_volume_clipped_validation.independent_point_archive_bounds import (
    PointArchiveBoundRefusal, _Radius, point_archive_conversion_bounds,
)
from analysis.genesis_volume_clipped_validation.independent_bulk_audit import independent_point_fields


def _case():
    source = dict(triangle_m=((0., 0., 3e6), (1e6, 0., 3e6), (0., 1e6, 3e6)),
        displacement_A_m=np.zeros((3, 3)).tolist(), slope_P_m=np.zeros((3, 3)).tolist(),
        lambda_bounds=(.99999, 1.), normal_xyz=(0., 1., 0.), offset_m=0.)
    domain = dict(geometry_source=source, seed_center_world_m=(0., 0., 3e6), seed_radius_m=100.,
        barycentric=((1/3, 1/3, 1/3), (.3, .35, .35)), radial_scale=(.999995, .999996),
        centered_birth_offsets_m=((10., 20., -3.), (-8., 17., 11.)), side=(.5, -.5))
    memory = (np.arange(18).reshape(3, 6)-7)*1e-4
    return domain, np.eye(6), np.eye(3), memory, memory*.3


def test_thin_lambda_archive_preimage_errors_are_enclosed_componentwise():
    domain, C, Q, e0, e1 = _case()
    bound = point_archive_conversion_bounds(domain, C, Q, e0, e1)
    center = independent_point_fields(domain, C, Q, e0, e1)
    eps = np.longdouble(np.finfo(float).eps)
    for sign in (-1, 1):
        shifted = deepcopy(domain)
        for name in ("barycentric", "radial_scale", "centered_birth_offsets_m"):
            value = np.asarray(domain[name], np.longdouble)
            shifted[name] = value+sign*eps*np.abs(value)*np.longdouble(.4)
        # Barycentric preimages are constrained to sum one.  Antisymmetric
        # perturbations keep that identity while testing individual rounding.
        value = np.asarray(domain["barycentric"], np.longdouble)
        delta = sign*eps*np.minimum(value[:, 0], value[:, 1])*np.longdouble(.4)
        shifted["barycentric"] = value+np.column_stack((delta, -delta, np.zeros(2)))
        fields = independent_point_fields(shifted, C, Q, e0, e1)
        for name in bound:
            error = np.abs(fields[name]-center[name])
            # LD operation differences belong to the separate construction
            # budget, kept here on each compared component's own operands.
            arithmetic = 128*np.finfo(np.longdouble).eps*np.maximum(np.abs(fields[name]), np.abs(center[name]))
            assert np.all(error <= bound[name]+arithmetic), name
    assert np.max(bound["bubble_gradient_reference_m_inverse"]) > 1e-13


def test_radial_derivative_conversion_survives_zero_radial_derivative_center():
    domain, C, Q, e0, e1 = _case()
    domain["radial_scale"] = (.5*(.99999+1),)*2
    bound = point_archive_conversion_bounds(domain, C, Q, e0, e1)
    assert np.max(bound["enrichment_strain_operator"]) > 1e-13
    assert np.max(bound["bubble_gradient_reference_m_inverse"]) > 1e-13


def test_zero_coordinate_radius_has_no_positive_absolute_floor():
    zero = _Radius.coordinate(np.zeros(3))
    assert np.all(zero.radius == 0)
    assert np.all((zero*1e20).radius == 0)
    assert np.all((zero/2).radius == 0)


def test_divisor_interval_containing_zero_is_refused():
    with pytest.raises(PointArchiveBoundRefusal, match="contains zero"):
        _Radius.fixed(1)/_Radius(np.array(1.), np.array(2.))
