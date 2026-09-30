"""Contact geometry and exact arc kinematics, independent of mesh ownership."""
from dataclasses import dataclass, replace
import math

import numpy as np
import pytest

from tectonics.fractional_surface import SurfaceParcel
from tectonics.geometric_contacts import contacts_between, extract_contacts
from tectonics.mesh import build_icosphere
from tectonics.spherical_polygons import polygon_area, rotate_polygon


@dataclass(frozen=True)
class Fragment:
    fragment_id: str
    polygon: object
    parcel: SurfaceParcel


def polygon(points):
    values = np.asarray([(x, y, 1.) for x, y in points], dtype=float)
    return values/np.linalg.norm(values, axis=1)[:, None]


def fragment(name, plate, points, cell=0):
    vertices = polygon(points)
    area = polygon_area(vertices)
    parcel = SurfaceParcel(cell, plate, name, area, area, 0., 0., 0.)
    return Fragment(name, vertices, parcel)


@pytest.fixture
def pair():
    return (fragment("a", 0, [(-1., -1.), (0., -1.), (0., 1.), (-1., 1.)]),
            fragment("b", 1, [(0., -1.), (1., -1.), (1., 1.), (0., 1.)]))


def test_actual_shared_arc_has_oriented_normal_exact_length_and_kinematics(pair):
    contact, = contacts_between(*pair, 1000., [[0., 0., 0.], [-.1, .2, 0.]])
    assert contact.length_km == pytest.approx(math.pi*500., rel=2e-14)
    np.testing.assert_allclose(contact.midpoint, (0., 0., 1.), atol=2e-15)
    np.testing.assert_allclose(contact.normal_a_to_b, (1., 0., 0.), atol=2e-15)
    np.testing.assert_allclose(contact.tangent, (0., 1., 0.), atol=2e-15)
    assert contact.normal_velocity_km_per_myr == pytest.approx(200.)
    assert contact.tangential_velocity_km_per_myr == pytest.approx(100.)
    assert contact.normal_area_rate_km2_per_myr == pytest.approx(.2e6*math.sqrt(2.))
    # A long arc's midpoint rule is measurably different from its exact flux.
    assert abs(contact.normal_area_rate_km2_per_myr
               - contact.length_km*contact.normal_velocity_km_per_myr) > 1e4
    assert contact.material_a == "a" and contact.material_b == "b"
    assert contact.edge_a == 1 and contact.edge_b == 3


def test_same_cell_does_not_connect_disjoint_polygons_or_vertex_touching():
    a = fragment("a", 0, [(-2., -1.), (-1., -1.), (-1., 1.), (-2., 1.)])
    b = fragment("b", 1, [(1., -1.), (2., -1.), (2., 1.), (1., 1.)])
    assert not contacts_between(a, b, 1.)
    c = fragment("c", 1, [(-1., 1.), (0., 1.), (0., 2.), (-1., 2.)])
    assert not contacts_between(a, c, 1.)


def test_adjacent_fragments_in_different_cells_still_have_contact(pair):
    a, b = pair
    b = replace(b, parcel=replace(b.parcel, cell=782))
    assert len(contacts_between(a, b, 1.)) == 1


def test_same_owner_material_boundary_is_not_plate_contact(pair):
    a, b = pair
    b = replace(b, parcel=replace(b.parcel, plate=0))
    assert not contacts_between(a, b, 1.)


def test_coincident_external_edges_of_overlapping_interiors_are_not_contacts(pair):
    a, _ = pair
    b = replace(a, fragment_id="b", parcel=replace(a.parcel, plate=1, material_id="b"))
    assert not contacts_between(a, b, 1.)


def test_contact_is_input_order_invariant_and_common_rotation_covariant(pair):
    omega = np.asarray([[.11, -.15, .2], [-.1, .2, .04]])
    original, = contacts_between(*pair, 1000., omega)
    assert contacts_between(*reversed(pair), 1000., omega) == (original,)
    rotation = np.asarray([.31, .17, -.24])
    # rotate_polygon validates convex polygons; Rodrigues on vectors below
    # independently gives the corresponding frame rotation.
    magnitude = np.linalg.norm(rotation)
    axis = rotation/magnitude
    skew = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]],
                     [-axis[1], axis[0], 0.]])
    matrix = np.eye(3)+math.sin(magnitude)*skew+(1.-math.cos(magnitude))*(skew@skew)
    rotated = tuple(replace(item, polygon=rotate_polygon(item.polygon, rotation, 1.))
                    for item in pair)
    contact, = contacts_between(*rotated, 1000., omega@matrix.T)
    assert contact.contact_id == original.contact_id
    assert contact.length_km == pytest.approx(original.length_km, rel=5e-14)
    assert contact.normal_velocity_km_per_myr == pytest.approx(original.normal_velocity_km_per_myr)
    assert contact.normal_area_rate_km2_per_myr == pytest.approx(original.normal_area_rate_km2_per_myr)
    for name in ("start", "end", "midpoint", "tangent", "normal_a_to_b",
                 "moment_arm_cross_normal_km2", "integrated_position_unit_km"):
        np.testing.assert_allclose(getattr(contact, name), matrix@np.asarray(getattr(original, name)),
                                   rtol=2e-13, atol=1e-9 if "km" in name else 2e-14)


def test_common_plate_velocity_has_exactly_zero_relative_kinematics(pair):
    contact, = contacts_between(*pair, 1., [[.1, -.2, .3], [.1, -.2, .3]])
    assert contact.normal_velocity_km_per_myr == 0.
    assert contact.tangential_velocity_km_per_myr == 0.
    assert contact.normal_area_rate_km2_per_myr == 0.


@pytest.mark.parametrize("width", [1e-3, 1e-6, 1e-9, 1e-12])
def test_common_rotation_keeps_short_contact_and_fragment_identity(pair, width):
    a, _ = pair
    b = fragment("b", 1, [(0., 1.-width), (1., 1.-width), (1., 1.), (0., 1.)])
    original, = contacts_between(a, b, 1.)
    rotation = np.asarray([.31, .17, -.24])
    rotated = tuple(replace(item, polygon=rotate_polygon(item.polygon, rotation, 1.))
                    for item in (a, b))
    contact, = contacts_between(*rotated, 1.)
    assert contact.length_km > 0.
    assert contact.contact_id == original.contact_id
    # Coordinates have fixed absolute rounding uncertainty; demanding a fixed
    # relative error for an arbitrarily small arc would be an invalid oracle.
    assert contact.length_km == pytest.approx(original.length_km, rel=0., abs=64.*np.finfo(float).eps)


def test_split_contact_preserves_total_length_flux_and_moment(pair):
    a, b = pair
    omega = np.asarray([[.11, -.15, .2], [-.1, .2, .04]])
    whole, = contacts_between(a, b, 5000., omega)
    # Split only B's arc in two; its two material fragments have one owner.
    bottom = fragment("b0", 1, [(0., -1.), (1., -1.), (1., 0.), (0., 0.)])
    top = fragment("b1", 1, [(0., 0.), (1., 0.), (1., 1.), (0., 1.)])
    pieces = extract_contacts((a, bottom, top), 5000., omega)
    assert len(pieces) == 2
    assert sum(item.length_km for item in pieces) == pytest.approx(whole.length_km, rel=2e-14)
    assert sum(item.normal_area_rate_km2_per_myr for item in pieces) == pytest.approx(
        whole.normal_area_rate_km2_per_myr, rel=2e-14)
    for name in ("moment_arm_cross_normal_km2", "integrated_position_unit_km"):
        np.testing.assert_allclose(np.sum([getattr(item, name) for item in pieces], axis=0),
                                   getattr(whole, name), rtol=2e-14, atol=1e-8)


def test_multiple_disconnected_arcs_of_same_owner_pair_stay_separate():
    items = (fragment("a0", 0, [(-1., -3.), (0., -3.), (0., -2.), (-1., -2.)]),
             fragment("b0", 1, [(0., -3.), (1., -3.), (1., -2.), (0., -2.)]),
             fragment("a1", 0, [(-1., 2.), (0., 2.), (0., 3.), (-1., 3.)]),
             fragment("b1", 1, [(0., 2.), (1., 2.), (1., 3.), (0., 3.)]))
    contacts = extract_contacts(items, 1.)
    assert len(contacts) == 2
    assert {frozenset((item.fragment_a, item.fragment_b)) for item in contacts} == {
        frozenset(("a0", "b0")), frozenset(("a1", "b1"))}
    assert contacts[0].contact_id != contacts[1].contact_id


def test_spherical_cap_broad_phase_matches_mesh_edge_oracle():
    mesh = build_icosphere(1)
    owners = np.arange(mesh.cell_count)%4
    items = []
    for i, face in enumerate(mesh.faces):
        area = float(mesh.areas_unit_sphere[i])
        parcel = SurfaceParcel(i, int(owners[i]), str(i), area, area, 0., 0., 0.)
        items.append(Fragment(str(i), mesh.vertices[face], parcel))
    contacts = extract_contacts(items, 1.)
    expected = {frozenset((str(a), str(b))) for a, b, _, _ in mesh.shared_edges
                if owners[a] != owners[b]}
    assert len(contacts) == len(expected)
    assert {frozenset((item.fragment_a, item.fragment_b)) for item in contacts} == expected


@pytest.mark.parametrize("radius", [0., -1., math.nan, math.inf, True])
def test_bad_radius_rejected(pair, radius):
    with pytest.raises(ValueError, match="radius"):
        contacts_between(*pair, radius)


@pytest.mark.parametrize("omega", [np.zeros((1, 3)), np.zeros((2, 2)), [[0., 0., 0.], [0., math.nan, 0.]]])
def test_bad_angular_velocities_rejected(pair, omega):
    with pytest.raises(ValueError, match="angular velocities"):
        extract_contacts(pair, 1., omega)


def test_duplicate_fragment_identity_rejected(pair):
    with pytest.raises(ValueError, match="identities"):
        extract_contacts((pair[0], pair[0]), 1.)


def test_empty_and_single_fragment_collections_have_no_contacts(pair):
    assert extract_contacts((), 1.) == ()
    assert extract_contacts(pair[:1], 1.) == ()
