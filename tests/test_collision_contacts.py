from dataclasses import replace

import numpy as np
import pytest

from tectonics.collision_contacts import strongest_connected_continental_contacts
from tectonics.kinematics import BoundaryRecord, BoundaryType
from tectonics.mesh import build_icosphere


RADIUS_KM = 5287.0


def _continental_cycle(subdivision=2):
    """An actual closed two-plate boundary, ordered along its mesh edges."""
    mesh = build_icosphere(subdivision)
    owner = (mesh.centroids[:, 0] > 0.0).astype(np.int32)
    records = []
    at_vertex = {}
    for fa, fb, u, v in mesh.shared_edges:
        if owner[fa] == owner[fb]:
            continue
        midpoint = mesh.vertices[u] + mesh.vertices[v]
        midpoint /= np.linalg.norm(midpoint)
        boundary = BoundaryRecord(
            fa, fb, u, v, int(owner[fa]), int(owner[fb]), midpoint,
            -20.0, 0.0, 20.0, BoundaryType.CONVERGENT,
        )
        i = len(records)
        records.append(boundary)
        at_vertex.setdefault(u, []).append(i)
        at_vertex.setdefault(v, []).append(i)
    assert all(len(indices) == 2 for indices in at_vertex.values())
    ordered = [records[0]]
    previous = 0
    vertex = records[0].vertex_v
    while len(ordered) < len(records):
        i = next(index for index in at_vertex[vertex] if index != previous)
        boundary = records[i]
        ordered.append(boundary)
        vertex = boundary.vertex_v if boundary.vertex_u == vertex else boundary.vertex_u
        previous = i
    return mesh, ordered


def _contacts(mesh, records, fraction=None):
    if fraction is None:
        fraction = np.ones(mesh.cell_count)
    return strongest_connected_continental_contacts(mesh, records, fraction, RADIUS_KM)


def test_disconnected_short_contacts_cannot_pool_to_initiate_collision():
    mesh, records = _continental_cycle()
    first, second = records[:2], records[len(records) // 2:len(records) // 2 + 2]
    lengths = [_contacts(mesh, segment)[(0, 1)].length_km for segment in (first, second)]
    threshold = max(lengths) + 0.25 * min(lengths)
    contact = _contacts(mesh, first + second)[(0, 1)]
    assert sum(lengths) > threshold
    assert contact.length_km == max(lengths)
    assert not contact.can_initiate(threshold, 65.0)
    assert not contact.can_maintain(threshold, 8.0)


@pytest.mark.parametrize("subdivision", [1, 2, 3])
def test_coherent_collision_and_stationary_weld_use_same_physical_length(subdivision):
    mesh, records = _continental_cycle(subdivision)
    contact = _contacts(mesh, records)[(0, 1)]
    assert contact.can_initiate(900.0, 65.0)
    assert contact.convergent_length_fraction == 1.0
    assert contact.mean_normal_rate_km_per_myr == pytest.approx(-20.0)
    assert not contact.is_quiet(8.0, 1.5)
    quiet_records = [replace(b, normal_rate_km_per_myr=0.0,
                             relative_speed_km_per_myr=2.0,
                             boundary_type=BoundaryType.INACTIVE) for b in records]
    quiet = _contacts(mesh, quiet_records)[(0, 1)]
    assert quiet.length_km == contact.length_km
    assert quiet.can_maintain(900.0, 8.0)
    assert quiet.is_quiet(8.0, 1.5)
    assert not quiet.can_initiate(900.0, 65.0)


def test_mostly_divergent_seam_cannot_initiate_despite_negative_signed_mean():
    mesh, records = _continental_cycle()
    mixed = [replace(b, normal_rate_km_per_myr=2.0,
                     relative_speed_km_per_myr=2.0,
                     boundary_type=BoundaryType.DIVERGENT) for b in records[:10]]
    mixed[0] = replace(mixed[0], normal_rate_km_per_myr=-60.0,
                       relative_speed_km_per_myr=60.0,
                       boundary_type=BoundaryType.CONVERGENT)
    contact = _contacts(mesh, mixed)[(0, 1)]
    assert contact.mean_normal_rate_km_per_myr < 0.0
    assert contact.mean_relative_speed_km_per_myr < 65.0
    assert contact.divergent_length_fraction > 0.8
    assert not contact.can_initiate(900.0, 65.0)


def test_compression_cannot_cancel_opening_in_maintenance_or_weld_criterion():
    mesh, records = _continental_cycle()
    mixed = [replace(b, normal_rate_km_per_myr=(-1.0 if i % 2 else 1.0) * 6.0,
                     relative_speed_km_per_myr=6.0,
                     boundary_type=(BoundaryType.CONVERGENT if i % 2 else BoundaryType.DIVERGENT))
             for i, b in enumerate(records)]
    contact = _contacts(mesh, mixed)[(0, 1)]
    assert abs(contact.mean_normal_rate_km_per_myr) < 1.5
    assert contact.mean_positive_divergence_km_per_myr > 1.5
    assert not contact.can_maintain(900.0, 1.5)
    assert not contact.is_quiet(8.0, 1.5)


def test_remote_quiet_fragment_cannot_dilute_active_collision_for_early_welding():
    mesh, records = _continental_cycle()
    moving = records[:5]
    quiet = [replace(b, normal_rate_km_per_myr=0.0,
                     relative_speed_km_per_myr=0.0,
                     boundary_type=BoundaryType.INACTIVE)
             for b in records[len(records) // 2:len(records) // 2 + 4]]
    moving = [replace(b, normal_rate_km_per_myr=-10.0,
                      relative_speed_km_per_myr=10.0) for b in moving]
    moving_length = _contacts(mesh, moving)[(0, 1)].length_km
    quiet_length = _contacts(mesh, quiet)[(0, 1)].length_km
    assert moving_length > quiet_length
    assert 10.0 * moving_length / (moving_length + quiet_length) < 8.0
    contact = _contacts(mesh, moving + quiet)[(0, 1)]
    assert contact.mean_relative_speed_km_per_myr == pytest.approx(10.0)
    assert not contact.is_quiet(8.0, 1.5)


def test_continental_fraction_weights_lengths_and_breaks_oceanic_gaps():
    mesh, records = _continental_cycle()
    b = records[0]
    fraction = np.ones(mesh.cell_count)
    fraction[b.face_a], fraction[b.face_b] = 0.5, 0.8
    full = _contacts(mesh, [b])[(0, 1)]
    mixed = _contacts(mesh, [b], fraction)[(0, 1)]
    assert mixed.length_km == pytest.approx(0.5 * full.length_km)
    fraction[b.face_a] = 0.249
    assert _contacts(mesh, [b], fraction) == {}
    fraction[b.face_a] = 0.25
    assert _contacts(mesh, [b], fraction)[(0, 1)].length_km == pytest.approx(0.25 * full.length_km)

    # Removing the middle section must disconnect the otherwise continuous seam.
    fraction = np.ones(mesh.cell_count)
    for gap in records[4:7]:
        fraction[gap.face_a] = 0.0
        fraction[gap.face_b] = 0.0
    filtered = [edge for edge in records[:11]
                if min(fraction[edge.face_a], fraction[edge.face_b]) >= 0.25]
    total = sum(_contacts(mesh, [edge])[(0, 1)].length_km for edge in filtered)
    contact = _contacts(mesh, records[:11], fraction)[(0, 1)]
    assert contact.length_km < total


def test_another_plate_pair_cannot_bridge_disconnected_contact_segments():
    mesh, records = _continental_cycle()
    records = records[:3]
    records[1] = replace(records[1], plate_a=2, plate_b=0)
    contacts = _contacts(mesh, records)
    assert list(contacts) == [(0, 1), (0, 2)]
    assert len(contacts[(0, 1)].edge_keys) == 1
    assert len(contacts[(0, 2)].edge_keys) == 1


def test_contact_selection_is_deterministic_under_input_order_and_orientation():
    mesh, records = _continental_cycle()
    records = records[:3] + records[len(records) // 2:len(records) // 2 + 3]
    expected = _contacts(mesh, records)
    reversed_records = [replace(b, vertex_u=b.vertex_v, vertex_v=b.vertex_u,
                                plate_a=b.plate_b, plate_b=b.plate_a,
                                face_a=b.face_b, face_b=b.face_a) for b in reversed(records)]
    assert _contacts(mesh, reversed_records) == expected
    generator = np.random.default_rng(8)
    for _ in range(5):
        assert _contacts(mesh, [records[i] for i in generator.permutation(len(records))]) == expected


def test_empty_contacts_return_empty_mapping():
    mesh, _ = _continental_cycle()
    assert _contacts(mesh, []) == {}
