"""Independent exact halfspace controls for persistent native cut geometry.

The oracle enumerates intersections of all support-plane pairs. Production
clips one edge at a time; the oracle never reads production scratch rays,
area receipts or rounded display geometry to decide a reference intersection.
"""
from copy import deepcopy
from decimal import Context, Decimal, localcontext
from fractions import Fraction
from functools import cmp_to_key
import json

import pytest

from tectonics.geometric_intrinsic_geometry import IntrinsicGeometry, RotationIncrement
from tectonics.geometric_native_polygons import (
    NativePlane, NativePolygon, native_clip, native_intersection,
    native_shared_boundary_arcs, native_subtract,
    native_plane_to_dict,
    native_polygon_from_dict as poly_from_dict,
    native_polygon_to_dict as poly_to_dict,
)
from test_geometric_intrinsic_surface import _area, _atan
from test_geometric_native_pose import _analytic_matrix
from test_geometric_precision import NEXT_DT_MYR, NEXT_OMEGA, thin_partition


COMMON = RotationIncrement((.023, -.037, .049), .71)
FOLLOWING = RotationIncrement((-.16, .09, .13), .29)


def _dot(first, second):
    return sum((a*b for a, b in zip(first, second)), 0)


def _cross(first, second):
    return (first[1]*second[2]-first[2]*second[1],
            first[2]*second[0]-first[0]*second[2],
            first[0]*second[1]-first[1]*second[0])


def _exact_ray(values):
    return tuple(Fraction(value) for value in values)


def _canonical_ray(values):
    scale = max(abs(value) for value in values)
    return tuple(value/scale for value in values) if scale else None


def _reference_planes(vertices):
    vertices = tuple(_exact_ray(vertex) for vertex in vertices)
    center = tuple(sum(vertex[i] for vertex in vertices) for i in range(3))
    planes = tuple(_cross(a,b) for a,b in zip(vertices, vertices[1:]+vertices[:1]))
    return tuple(tuple(-value for value in plane) if _dot(plane, center)<0 else plane
                 for plane in planes)


def _reference_intersection(vertices, extra_planes=()):
    """Enumerate the exact feasible vertices; no sequential clipping."""
    vertices = tuple(_exact_ray(vertex) for vertex in vertices)
    center = tuple(sum(vertex[i] for vertex in vertices) for i in range(3))
    planes = _reference_planes(vertices)+tuple(_exact_ray(plane) for plane in extra_planes)
    candidates = set()
    for i, first in enumerate(planes):
        for second in planes[i+1:]:
            crossing = _cross(first, second)
            for candidate in (crossing, tuple(-value for value in crossing)):
                saved = _canonical_ray(candidate)
                if (saved is not None and _dot(saved, center)>0
                        and all(_dot(plane, saved)>=0 for plane in planes)):
                    candidates.add(saved)
    if len(candidates)<3:
        return ()
    candidates = sorted(candidates)
    center = tuple(sum(vertex[i] for vertex in candidates) for i in range(3))
    reference = candidates[0]
    norm2 = _dot(center, center)
    def coordinates(vertex):
        return (norm2*_dot(reference, vertex)-_dot(reference, center)*_dot(vertex, center),
                _dot(center, _cross(reference, vertex)))
    def compare(first, second):
        ax, ay = coordinates(first)
        bx, by = coordinates(second)
        ah = 0 if ay>0 or ay==0 and ax>=0 else 1
        bh = 0 if by>0 or by==0 and bx>=0 else 1
        if ah != bh:
            return -1 if ah<bh else 1
        side = _dot(center, _cross(first, second))
        return -1 if side>0 else 1 if side<0 else 0
    ordered = tuple(sorted(candidates, key=cmp_to_key(compare)))
    if not any(_dot(ordered[0], _cross(first,second))>0
               for first,second in zip(ordered[1:-1],ordered[2:])):
        return ()
    return ordered


def _decimal_fraction(value):
    value = Fraction(value)
    return Decimal(value.numerator)/Decimal(value.denominator)


def _reference_area(vertices, precision=400):
    if len(vertices)<3:
        return Decimal(0)
    with localcontext(Context(prec=precision)):
        rays = []
        for vertex in vertices:
            ray = tuple(_decimal_fraction(value) for value in vertex)
            norm = _dot(ray,ray).sqrt()
            rays.append(tuple(value/norm for value in ray))
        return _area(tuple(rays))


def _reference_rotation(values, history, precision=400):
    matrix = _analytic_matrix(history, precision)
    with localcontext(Context(prec=precision)):
        return tuple(tuple(_dot(row, tuple(_decimal_fraction(value) for value in ray))
                           for row in matrix) for ray in values)


def _reference_arc_length(vertices, precision=400):
    assert len(vertices)==2
    with localcontext(Context(prec=precision)):
        rays=[]
        for vertex in vertices:
            ray=tuple(_decimal_fraction(value) for value in vertex)
            norm=_dot(ray,ray).sqrt()
            rays.append(tuple(value/norm for value in ray))
        a,b=rays
        crossing=_cross(a,b)
        sine=_dot(crossing,crossing).sqrt()
        # The half-angle formula avoids acos(1)'s cancellation on positive
        # arcs much shorter than a world-coordinate binary64 ULP.
        return 2*_atan(sine/(1+_dot(a,b)))


def _thin_body(width=1e-33):
    original = next(fragment.polygon for fragment in thin_partition().fragments
                    if fragment.fragment_id == 'thin')
    return tuple((x,y,width if z else z) for x,y,z in original)


def _snapshot(polygon):
    return json.dumps(poly_to_dict(polygon), sort_keys=True, separators=(',', ':'),
                      allow_nan=False)


def _assert_positive_geometry(polygon, expected):
    assert polygon is not None and expected>0
    enclosure = polygon.area_bounds()
    assert enclosure.lo > 0
    assert enclosure.width/enclosure.lo < Fraction(1,10**10)
    oracle_value = Fraction(expected)
    oracle_error = Fraction(1,10**350)
    assert enclosure.lo-oracle_error <= oracle_value <= enclosure.hi+oracle_error
    assert polygon.area_sr == pytest.approx(float(expected), rel=2e-12, abs=0.)


@pytest.mark.parametrize('width', [1e-33, 1e-80, 1e-150])
def test_positive_thin_cut_children_and_exact_sibling_zero_survive_collapsed_display(width):
    body = _thin_body(width)
    geometry = IntrinsicGeometry(body).rotated(COMMON.omega_rad_per_myr, COMMON.dt_myr)
    parent = NativePolygon.from_intrinsic(geometry)
    # At this pose, a final binary64 endpoint view cannot express the shape.
    assert len(set(parent.world_polygon())) < len(body)
    # Exact half-height is z >= (width/2)*y. Clear the Fraction denominator
    # before constructing the integer primitive support plane.
    half_height = Fraction.from_float(width)/2
    normal = (0,-half_height.numerator,half_height.denominator)
    plane = NativePlane(normal, rotations=(COMMON,), label='physical half-height')
    opposite = NativePlane(tuple(-v for v in normal), rotations=(COMMON,),
                           label='same support, opposite side')
    top, bottom = native_clip(parent, plane), native_clip(parent, opposite)
    expected_top = _reference_intersection(body, (normal,))
    expected_bottom = _reference_intersection(body, (tuple(-v for v in normal),))
    _assert_positive_geometry(top, _reference_area(expected_top))
    _assert_positive_geometry(bottom, _reference_area(expected_bottom))
    assert native_intersection(top,bottom) is None
    assert native_intersection(bottom,top) is None
    parts = native_subtract(parent,top)
    assert len(parts)==1
    _assert_positive_geometry(parts[0], _reference_area(expected_bottom))
    with localcontext(Context(prec=400)):
        expected_parent = _reference_area(body)
        assert abs((_reference_area(expected_top)+_reference_area(expected_bottom)) /
                   expected_parent-1) < Decimal('1e-220')
    assert top.area_sr+bottom.area_sr == pytest.approx(parent.area_sr, rel=2e-12, abs=0.)
    arcs = native_shared_boundary_arcs(top,bottom)
    assert len(arcs)==1 and arcs[0].length_radians>0
    shared=tuple(vertex for vertex in expected_top if _dot(normal,vertex)==0)
    assert arcs[0].length_radians == pytest.approx(float(_reference_arc_length(shared)),
                                                 rel=2e-12,abs=0.)
    assert _snapshot(poly_from_dict(poly_to_dict(top))) == _snapshot(top)


def test_native_cut_predicates_never_consult_rounded_world_projection(monkeypatch):
    parent = NativePolygon.from_intrinsic(IntrinsicGeometry(_thin_body()).rotated(
        COMMON.omega_rad_per_myr, COMMON.dt_myr))
    plane = NativePlane((0,-1,2*10**33), rotations=(COMMON,))
    opposite = NativePlane((0,1,-2*10**33), rotations=(COMMON,))
    def forbidden_projection(*args, **kwargs):
        raise AssertionError('A display projection cannot determine native material topology')
    monkeypatch.setattr(NativePolygon, 'world_polygon', forbidden_projection)
    top, bottom = native_clip(parent,plane), native_clip(parent,opposite)
    assert top is not None and bottom is not None
    assert native_intersection(top,bottom) is None
    assert native_subtract(parent,top)
    assert native_shared_boundary_arcs(top,bottom)


@pytest.mark.parametrize('normal', [(1,0,0),(0,0,-1)])
def test_exact_tangent_arc_touch_does_not_create_positive_material(normal):
    body = _thin_body()
    parent = NativePolygon.from_intrinsic(IntrinsicGeometry(body).rotated(
        COMMON.omega_rad_per_myr, COMMON.dt_myr))
    assert _reference_intersection(body,(normal,)) == ()
    assert native_clip(parent, NativePlane(normal,rotations=(COMMON,))) is None


def test_different_complete_physical_poses_cut_then_resume_native_children_exactly():
    body = _thin_body()
    material_pose = RotationIncrement(NEXT_OMEGA[0][1],NEXT_DT_MYR)
    receiver_pose = RotationIncrement(NEXT_OMEGA[0][0],NEXT_DT_MYR)
    parent = NativePolygon.from_intrinsic(IntrinsicGeometry(body).rotated(
        material_pose.omega_rad_per_myr,material_pose.dt_myr))
    cut_normal = (2*10**6,1,0)
    receiver_plane = NativePlane(cut_normal,rotations=(receiver_pose,),label='actual receiver')
    source_before = _snapshot(parent)
    first = native_clip(parent,receiver_plane)
    assert _snapshot(parent)==source_before
    expected_world = _reference_rotation(body,(material_pose,))
    expected_plane = _reference_rotation((cut_normal,),(receiver_pose,))[0]
    expected_first = _reference_intersection(expected_world,(expected_plane,))
    _assert_positive_geometry(first,_reference_area(expected_first))
    assert 0<first.area_sr<parent.area_sr
    histories={tuple((step.omega_rad_per_myr,step.dt_myr) for step in plane.rotations)
               for plane in first.support_planes}
    assert ((material_pose.omega_rad_per_myr,material_pose.dt_myr),) in histories
    assert ((receiver_pose.omega_rad_per_myr,receiver_pose.dt_myr),) in histories
    opposite=NativePlane(tuple(-v for v in cut_normal),rotations=(receiver_pose,))
    other_child=native_clip(parent,opposite)
    arcs=native_shared_boundary_arcs(first,other_child)
    assert len(arcs)==1
    exact_plane=_exact_ray(expected_plane)
    shared=tuple(vertex for vertex in expected_first if _dot(exact_plane,vertex)==0)
    expected_length=_reference_arc_length(shared)
    assert 0<expected_length<Decimal('1e-30')
    assert arcs[0].length_radians == pytest.approx(float(expected_length),rel=2e-12,abs=0.)
    first_saved = _snapshot(first)
    restored = poly_from_dict(json.loads(first_saved))
    # The next cut deliberately acts on saved cut children, not the seed body.
    direct = first.rotated(FOLLOWING.omega_rad_per_myr,FOLLOWING.dt_myr)
    resumed = restored.rotated(FOLLOWING.omega_rad_per_myr,FOLLOWING.dt_myr)
    history = (material_pose,FOLLOWING)
    second_normal = (0,-1,2*10**33)
    second_plane = NativePlane(second_normal,rotations=history,label='later material cut')
    direct = native_clip(direct,second_plane)
    resumed = native_clip(resumed,second_plane)
    assert direct is not None and resumed is not None
    assert _snapshot(direct)==_snapshot(resumed)
    assert _snapshot(first)==first_saved
    transformed_first = _reference_rotation(expected_first,(FOLLOWING,))
    transformed_plane = _reference_rotation((second_normal,),history)[0]
    expected_second = _reference_intersection(transformed_first,(transformed_plane,))
    _assert_positive_geometry(direct,_reference_area(expected_second))
    assert 0<direct.area_sr<first.area_sr
    assert _snapshot(poly_from_dict(poly_to_dict(direct))) == _snapshot(direct)


@pytest.mark.parametrize('corruption', ['untrusted_area','untrusted_projection','extra_schema_field'])
def test_native_polygon_reader_rejects_unvalidated_receipts(corruption):
    polygon = NativePolygon.from_intrinsic(IntrinsicGeometry(_thin_body()))
    raw = deepcopy(poly_to_dict(polygon))
    if corruption=='untrusted_area':
        raw['area_sr'] = 1.
    elif corruption=='untrusted_projection':
        raw['world_polygon'] = [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
    else:
        raw['accept_without_lineage'] = True
    with pytest.raises(ValueError):
        poly_from_dict(raw)


def test_exact_empty_body_edge_support_is_independent_of_optional_labels():
    parent = NativePolygon.from_intrinsic(IntrinsicGeometry(_thin_body()))
    first = native_clip(parent,NativePlane((0,-1,2*10**33),label='receiver A'))
    second = native_clip(parent,NativePlane((0,1,-2*10**33),label='receiver B'))
    assert first is not None and second is not None
    assert native_intersection(first,second) is None
    assert native_shared_boundary_arcs(first,second)


def test_tiny_nonparallel_omega_cannot_authorize_an_axis_fixed_plane():
    body=((1.,0.,0.),(0.,1.,0.),(0.,0.,1.))
    parent=NativePolygon.from_intrinsic(IntrinsicGeometry(body))
    # The off-axis squared term underflows binary64, while its physical
    # displacement and the removed spherical area are still representable.
    step=RotationIncrement((1e-300,0.,.01),1.)
    tilted=NativePlane((0,0,1),rotations=(step,))
    kept=native_clip(parent,tilted)
    assert kept is not None
    removed=native_subtract(parent,kept)
    assert len(removed)==1 and removed[0].area_sr>0
    expected_plane=_reference_rotation(((0,0,1),),(step,),precision=760)[0]
    expected_removed=_reference_intersection(body,(tuple(-Fraction(v) for v in expected_plane),))
    _assert_positive_geometry(removed[0],_reference_area(expected_removed,precision=760))
    assert native_plane_to_dict(tilted)['rotations'][0]['omega_rad_per_myr']==[1e-300,0.,.01]


def test_exact_axis_fixed_support_across_different_histories_retains_full_motion():
    upper=((1.,0.,0.),(0.,1.,0.),(0.,0.,1.))
    lower=((1.,0.,0.),(0.,0.,-1.),(0.,1.,0.))
    first=RotationIncrement((0.,0.,.03),1.)
    second=RotationIncrement((0.,0.,.01),1.)
    a=NativePolygon.from_intrinsic(IntrinsicGeometry(upper).rotated(
        first.omega_rad_per_myr,first.dt_myr))
    b=NativePolygon.from_intrinsic(IntrinsicGeometry(lower).rotated(
        second.omega_rad_per_myr,second.dt_myr))
    arcs=native_shared_boundary_arcs(a,b)
    assert len(arcs)==1
    with localcontext(Context(prec=400)):
        pi=16*_atan(Decimal(1)/5)-4*_atan(Decimal(1)/239)
        expected=pi/2-abs(Decimal.from_float(.03)-Decimal.from_float(.01))
    assert arcs[0].length_radians == pytest.approx(float(expected),rel=2e-12,abs=0.)
    assert arcs[0].plane.rotations==(first,)
    saved_plane=native_plane_to_dict(arcs[0].plane)
    assert saved_plane['rotations'][0]['omega_rad_per_myr']==[0.,0.,.03]
    assert saved_plane['rotations'][0]['dt_myr']==1.


@pytest.mark.parametrize('lower',[
    ((-1.,0.,0.),(0.,-1.,0.),(0.,0.,-1.)),
    ((-1.,0.,0.),(0.,1.,0.),(0.,0.,-1.)),
])
def test_disjoint_or_single_endpoint_minor_arcs_never_include_a_remote_antipode(lower):
    upper=((1.,0.,0.),(0.,1.,0.),(0.,0.,1.))
    a=NativePolygon.from_intrinsic(IntrinsicGeometry(upper).rotated(
        COMMON.omega_rad_per_myr,COMMON.dt_myr))
    b=NativePolygon.from_intrinsic(IntrinsicGeometry(lower).rotated(
        COMMON.omega_rad_per_myr,COMMON.dt_myr))
    assert native_intersection(a,b) is None
    assert native_shared_boundary_arcs(a,b)==()
