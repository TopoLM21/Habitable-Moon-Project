"""Saved-edge incidence at tilted poles, actual contacts and archive supports."""
from dataclasses import replace
from functools import lru_cache
import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tests.underthrust_case import advance_case
from tectonics.geometric_boundary import (
    _arc_interval, boundary_from_dict, boundary_to_dict,
)
from tectonics.geometric_contacts import extract_contacts
from tectonics.geometric_polarity import intersect_arcs
from tectonics.geometric_transport import _arc_meets_contact
from tectonics.spherical_polygons import (
    _edge_distance, arc_length, normalize_polygon, polygon_area, shared_boundary_arcs,
)


RECEIVER = np.asarray([
    [-8.950365874871867e-7, .9999999999995994, 1.7584044141472642e-30],
    [-3.366688077358961e-27, 1.7553911051400742e-30, -1.],
])
ACTUAL = np.asarray([
    [-8.950365874871867e-7, .9999999999995994, 1.7584044141472646e-30],
    [-3.366688077358961e-27, 1.7553911051400746e-30, -1.],
])


@lru_cache(maxsize=None)
def actual_step():
    return advance_case(viscosity_pa_s_m=1e9)


def tilted_neighbors(matrix=None):
    first = normalize_polygon((*RECEIVER, (1., 0., 0.)))
    second = normalize_polygon((*RECEIVER, (-1., 0., 0.)))
    if matrix is not None:
        first, second = first@matrix.T, second@matrix.T
    return first, second


def test_long_tilted_pole_edge_cannot_reject_its_own_defining_endpoint():
    normal = np.cross(RECEIVER[0], RECEIVER[1]-RECEIVER[0])
    normal /= np.linalg.norm(normal)
    # The observed defect: forming this rounded normal loses its tiny z
    # component and its own saved pole endpoint has a nonzero dot product.
    assert normal[2] == 0. and float(RECEIVER[1]@normal) != 0.
    shared = intersect_arcs(*RECEIVER, *RECEIVER)
    assert shared == tuple(map(tuple, RECEIVER))
    assert intersect_arcs(*RECEIVER, *ACTUAL) is not None


def test_adjacent_polygons_share_a_positive_arc_despite_the_rounded_normal_defect():
    first, second = tilted_neighbors()
    shared = shared_boundary_arcs(first, second)
    assert len(shared) == 1
    assert arc_length(*shared[0]) == pytest.approx(math.pi/2., rel=4e-15)
    assert shared_boundary_arcs(first, first) == ()


@pytest.mark.parametrize("angle", [-1e-8, 1e-8, 1e-12])
def test_resolved_foreign_plane_offset_is_not_replaced_by_a_distance_floor(angle):
    matrix = Rotation.from_rotvec((0., 0., angle)).as_matrix()
    foreign = ACTUAL@matrix.T
    assert intersect_arcs(*RECEIVER, *foreign) is None
    first, second = tilted_neighbors()
    assert shared_boundary_arcs(first, second@matrix.T) == ()


@pytest.mark.parametrize("vector", [(0., 0., 0.), (.3, -.7, .2), (-.21, .42, .34)])
def test_plane_incidence_and_positive_arc_length_are_coordinate_covariant(vector):
    matrix = Rotation.from_rotvec(vector).as_matrix()
    arc = intersect_arcs(*(RECEIVER@matrix.T), *(ACTUAL@matrix.T))
    assert arc is not None
    assert arc_length(*arc) == pytest.approx(math.pi/2., rel=4e-15)
    first, second = tilted_neighbors(matrix)
    assert len(shared_boundary_arcs(first, second)) == 1


def test_a_single_shared_endpoint_remains_zero_support():
    assert intersect_arcs(*RECEIVER, RECEIVER[1], -RECEIVER[0]) is None
    first = normalize_polygon(((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)))
    second = normalize_polygon(((1., 0., 0.), (0., -1., 0.), (0., 0., -1.)))
    assert shared_boundary_arcs(first, second) == ()


@pytest.mark.parametrize("width", [1e-10, 1e-12, 1e-14])
def test_positive_short_shared_arcs_have_no_minimum_length(width):
    def chart(points):
        return normalize_polygon(tuple((x, y, 1.) for x, y in points))
    first = chart(((-.2, -.2), (0., -.2), (0., .2), (-.2, .2)))
    second = chart(((0., .1), (.2, .1), (.2, .1+width), (0., .1+width)))
    shared = shared_boundary_arcs(first, second)
    assert len(shared) == 1 and arc_length(*shared[0]) > 0.


def test_actual_differential_force_step_retains_the_actual_current_material_contacts():
    _, origin, _, step = actual_step()
    current = step.state.current_surface
    contacts = extract_contacts(current.fragments, current.radius_km)
    assert len(extract_contacts(origin.fragments, origin.radius_km)) == 4
    assert len(contacts) >= 4
    assert len(step.geometric_transaction.losses) >= 2
    assert len(step.geometric_transaction.births) >= 2
    assert len(step.state.current_inventory.contacts) == len(contacts)
    assert {contact.contact_id for contact in contacts} == {
        contact.contact_id for contact in step.state.current_inventory.contacts}
    # Preserve the actual off-axis force solution, including representable
    # tiny entries.  The geometry correction does not impose axial motion.
    increments = np.asarray(step.rotation_increment_radians)
    assert np.count_nonzero(increments[:, :2]) == 4
    assert np.max(np.abs(increments[:, :2])) < 1e-20
    assert np.all(increments[:, 2] > 0.) and increments[0, 2] != increments[1, 2]


def test_actual_contact_defining_endpoints_support_the_transport_and_archive_predicates():
    _, _, _, step = actual_step()
    for contact in step.state.current_inventory.contacts:
        assert _arc_meets_contact(np.asarray(contact.start), np.asarray(contact.end), contact)
        interval = _arc_interval(contact.start, contact.end, contact)
        assert interval is not None and interval[0] == 0.
        assert interval[1] == pytest.approx(arc_length(contact.start, contact.end), rel=4e-15)


def test_actual_archive_saved_supports_bind_live_contacts_and_preserve_payloads_on_roundtrip():
    _, _, _, step = actual_step()
    inventory = step.state.current_inventory
    assert inventory.cohorts
    for cohort in inventory.cohorts:
        assert cohort.supports and cohort.attachment_status == "geometric_contact"
        assert not cohort.unresolved_support
        for support in cohort.supports:
            contact = next(contact for contact in inventory.contacts
                           if contact.contact_id == support.contact_id)
            assert _arc_interval(support.start, support.end, contact) is not None
    assert boundary_from_dict(boundary_to_dict(inventory)) == inventory


def test_transport_contact_incidence_still_rejects_a_resolved_foreign_arc():
    _, _, _, step = actual_step()
    contact = step.state.current_inventory.contacts[0]
    matrix = Rotation.from_rotvec((0., 0., 1e-8)).as_matrix()
    foreign = np.asarray((contact.start, contact.end))@matrix.T
    assert not _arc_meets_contact(*foreign, contact)
    assert _arc_interval(*foreign, contact) is None


@pytest.mark.parametrize("width", [1e-20, 1e-30, 1e-100])
def test_positive_arc_at_a_long_arc_endpoint_survives_global_angle_rounding(width):
    start, end = np.asarray((0., 1., 0.)), np.asarray((0., 0., 1.))
    near = np.asarray((0., width, 1.))
    assert math.atan2(near[2], near[1]) == math.pi/2.
    shared = intersect_arcs(start, end, near, end)
    assert shared == (tuple(near), tuple(end))
    assert arc_length(*shared) == pytest.approx(width, rel=4e-15)
    assert intersect_arcs(start, end, end, (0., -width, 1.)) is None


@pytest.mark.parametrize("shift", range(3))
def test_positive_pole_triangle_area_does_not_depend_on_the_fan_anchor(shift):
    polygon = np.asarray(((math.sqrt(.5), math.sqrt(.5), 0.),
                          (-1e-26, 0., 1.), (1e-26, 0., 1.)))
    polygon = np.roll(polygon, shift, axis=0)
    assert polygon_area(polygon) == pytest.approx(math.sqrt(2.)*1e-26, rel=4e-15)
    assert len(normalize_polygon(polygon)) == 3


def test_long_edge_incidence_distinguishes_two_resolved_tiny_poles_in_both_directions():
    start = np.asarray((0., 1., 0.))
    end = np.asarray((-7.124709773194533e-27, 1.7529614878320066e-30, 1.))
    foreign = np.asarray((3.366688077358961e-27, -1.7553911051400746e-30, 1.))
    assert _edge_distance(foreign, start, end, None) > 0.
    assert _edge_distance(foreign, end, start, None) < 0.
    assert intersect_arcs(start, end, start, foreign) is None


def test_archive_endpoint_incidence_resolves_alternate_constructions_of_one_endpoint():
    start = (.5278560940159082, -.8493338236584416, 0.)
    end = (.5270425374377207, -.8498338229070751, .00293890700003773)
    constructed_end = (.5270425374377207, -.8498338229070751, .0029389070000377165)
    contact = SimpleNamespace(start=start, end=end)
    assert constructed_end != end
    assert _arc_interval(start, constructed_end, contact) == (0., arc_length(start, end))


@pytest.mark.parametrize("width", [1e-20, 1e-27, 1e-30])
def test_archive_endpoint_incidence_retains_resolved_tiny_partial_support(width):
    start, end = (0., width, 1.), (0., 0., 1.)
    midpoint = (0., width/2., 1.)
    contact = SimpleNamespace(start=start, end=end)
    assert _arc_interval(start, midpoint, contact) == pytest.approx((0., width/2.), rel=4e-15)
    assert _arc_interval(start, (width, width/2., 1.), contact) is None


def test_archive_endpoint_incidence_rejects_antipodal_support():
    start = np.asarray((1., 0., 0.))
    end = np.asarray((math.cos(.2), math.sin(.2), 0.))
    contact = SimpleNamespace(start=start, end=end)
    assert _arc_interval(-start, -end, contact) is None
    assert _arc_interval(start, -start, contact) is None


@pytest.mark.parametrize("reverse", [False, True])
def test_exact_one_ulp_contact_end_is_not_rewritten_as_its_start(reverse):
    # Captured from the second acceptance of the passive material-history
    # integration: approximate incidence overlaps at both saved endpoints.
    # Literal endpoint identity must retain their strictly positive interval.
    start = (.5240287601747566, .8517005685742539, 0.)
    end = (.5240287601747565, .8517005685742539, 0.)
    contact = SimpleNamespace(start=start, end=end)
    extent = arc_length(start, end)
    assert extent > 0.
    pair = (end, start) if reverse else (start, end)
    assert _arc_interval(*pair, contact) == (0., extent)
    # This transported historical arc only touches the exact current end.
    # Relabelling it as start previously invented a full positive support.
    farther = (-.0007890583148992472, .5260384820702815, -.8504604004703392)
    assert _arc_interval(end, farther, contact) is None
