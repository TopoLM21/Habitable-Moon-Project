"""Regressions from actual 5120-cell geometry and saved coordinate audits."""
from dataclasses import asdict
import json

import numpy as np
import pytest

from tectonics.fractional_surface import SurfaceParcel
from tectonics.geometric_surface import GeometricFragment
from tectonics.mesh import build_icosphere
from tectonics.spherical_polygons import (GeometryDiagnostics, intersect_convex,
    normalize_polygon, polygon_area, rotate_polygon)


@pytest.mark.parametrize('cells,omega,dt', [
    ((288, 704), [.0007, -.0004, .0002], .25),
    ((528, 608), [.003, -.007, .002], 1.),
])
def test_common_rotation_preserves_exact_point_incidence(cells, omega, dt):
    # Direct dot products with rounded edge normals previously fabricated
    # intersections of order 1e-35 sr at these common mesh vertices.
    mesh = build_icosphere(4)
    a, b = [rotate_polygon(mesh.vertices[mesh.faces[cell]], omega, dt) for cell in cells]
    assert not len(intersect_convex(a, b, diagnostics=GeometryDiagnostics()))
    assert not len(intersect_convex(b, a, diagnostics=GeometryDiagnostics()))


@pytest.mark.parametrize('clockwise', [False, True])
@pytest.mark.parametrize('width', [1e-6, 1e-10, 1e-14])
def test_fragment_orientation_preserves_raw_coordinate_bits(clockwise, width):
    narrow = normalize_polygon([(0., 1.-width, 1.), (1., 1.-width, 1.),
                                (1., 1., 1.), (0., 1., 1.)])
    points = rotate_polygon(narrow, [.31, .17, -.24], 1.)
    # Put the short edge first: raw cross(v0,v1) loses relative precision.
    points = np.roll(points, -1, axis=0)
    raw = points[::-1] if clockwise else points
    area = polygon_area(raw)
    parcel = SurfaceParcel(0, 0, 'roundoff-audit', area, area, 0., 0., 0.)
    fragment = GeometricFragment('audit', tuple(map(tuple, raw)), parcel)
    np.testing.assert_array_equal(fragment.polygon, points)
    saved = json.loads(json.dumps(asdict(fragment)))
    restored = GeometricFragment(saved['fragment_id'], saved['polygon'],
                                 SurfaceParcel(**saved['parcel']), saved['parent_fragment_id'])
    assert asdict(restored) == asdict(fragment)
