"""Exact-locality checks for optional legacy direction translation."""
from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pytest

from tectonics.fractional_surface import FractionalSurfaceState, SurfaceParcel
from tectonics.geometric_contacts import extract_contacts
from tectonics.geometric_polarity_source import PROVENANCE, legacy_polarity_evidence
from tectonics.geometric_surface import from_fractional_surface, rotate_surface
from tectonics.mesh import build_icosphere
from tectonics.young_boundary import (
    SlabThermalCohort, YoungBoundaryState, YoungContact, YoungSlabSegment,
)


def fixture():
    mesh = build_icosphere(0)
    areas = mesh.physical_cell_areas_km2(100.)
    owners = (mesh.centroids[:, 0] > 0.).astype(int)
    parcels = tuple(SurfaceParcel(i, int(owner), f"source:{i}", float(area),
        float(2.*area), float(10.*area), float(1e12*area), 5.)
        for i, (area, owner) in enumerate(zip(areas, owners)))
    surface = from_fractional_surface(mesh, FractionalSurfaceState(0., tuple(areas), parcels), 100.)
    geometric = extract_contacts(surface.fragments, 100.)[0]
    by_id = {fragment.fragment_id: fragment for fragment in surface.fragments}
    a, b = by_id[geometric.fragment_a].parcel.cell, by_id[geometric.fragment_b].parcel.cell
    shared = sorted(set(mesh.faces[a]) & set(mesh.faces[b]))
    key = f"{shared[0]}:{shared[1]}"
    contact = YoungContact(key, a, b, geometric.plate_a, geometric.plate_b,
        geometric.length_km, list(geometric.midpoint), [0., 0., 1.])
    segment = YoungSlabSegment("old-segment", key, geometric.plate_a, geometric.plate_b,
        mesh.centroids[a].tolist(), mesh.centroids[b].tolist(), list(geometric.midpoint),
        geometric.length_km, thermal_cohorts=[
            SlabThermalCohort(0., 10., 20., 100., 1e12, [0., 0., 1e12], 10., .25)])
    inventory = YoungBoundaryState(contacts={key: contact}, segments={segment.key: segment})
    return mesh, surface, inventory, geometric


def test_exact_local_direction_translates_without_importing_mass_or_mutating_inputs():
    mesh, surface, inventory, geometric = fixture()
    old_surface, old_inventory = asdict(surface), deepcopy(asdict(inventory))
    old_vertices, old_faces = mesh.vertices.copy(), mesh.faces.copy()
    evidence, audit = legacy_polarity_evidence(mesh, surface, inventory)
    assert len(evidence) == 1
    assert evidence[0].evidence_id == "legacy:old-segment"
    assert evidence[0].contact_id == geometric.contact_id
    assert evidence[0].provenance == PROVENANCE
    assert audit["counts"]["oriented_contacts"] == 1
    assert audit["mapping"][0]["legacy_retained_quantities_not_imported"]["retained_area_km2"] == 7.5
    assert audit["imported_material"] is audit["imported_force"] is False
    assert audit["source_inventory_sha256_before"] == audit["source_inventory_sha256_after"]
    assert asdict(surface) == old_surface and asdict(inventory) == old_inventory
    np.testing.assert_array_equal(mesh.vertices, old_vertices)
    np.testing.assert_array_equal(mesh.faces, old_faces)


def test_opposing_retained_directions_remain_two_conflicting_evidence_records():
    mesh, surface, inventory, _ = fixture()
    first = inventory.segments["old-segment"]
    other = deepcopy(first)
    other.key = "opposite"
    other.subducting_plate, other.overriding_plate = first.overriding_plate, first.subducting_plate
    inventory.segments[other.key] = other
    evidence, audit = legacy_polarity_evidence(mesh, surface, inventory)
    assert len(evidence) == 2
    assert len({e.subducting_plate for e in evidence}) == 2
    assert audit["counts"]["conflicting_contacts"] == 1
    assert audit["counts"]["oriented_contacts"] == 0


@pytest.mark.parametrize("case,reason", [
    ("absent", "absent_legacy_contact"),
    ("detached", "detached_segment"),
    ("deep", "no_retained_material"),
    ("empty", "no_retained_material"),
    ("bad_key", "not_exact_source_mesh_edge"),
    ("wrong_owner", "current_owner_mismatch"),
    ("wrong_pair", "legacy_owner_pair_mismatch"),
    ("invalid_cohort", "invalid_retained_cohort"),
])
def test_stale_detached_empty_or_invalid_history_is_explicitly_excluded(case, reason):
    mesh, surface, inventory, _ = fixture()
    segment = inventory.segments["old-segment"]
    contact = inventory.contacts[segment.contact_key]
    if case == "absent":
        contact.present = False
    elif case == "detached":
        segment.attached = False
    elif case == "deep":
        segment.thermal_cohorts[0].deep_transfer_fraction = 1.
    elif case == "empty":
        segment.thermal_cohorts = []
    elif case == "bad_key":
        inventory.contacts = {"wrong:edge": contact}
        contact.key = segment.contact_key = "wrong:edge"
    elif case == "wrong_owner":
        fragment = surface.fragments[contact.face_a]
        updated = replace(fragment, parcel=replace(fragment.parcel, plate=contact.plate_b))
        surface = replace(surface, fragments=surface.fragments[:contact.face_a]+(updated,)+surface.fragments[contact.face_a+1:])
    elif case == "wrong_pair":
        segment.overriding_plate = 9
    else:
        segment.thermal_cohorts[0].deep_transfer_fraction = 2.
    evidence, audit = legacy_polarity_evidence(mesh, surface, inventory)
    assert not evidence
    assert reason in audit["excluded_segments"][0]["reasons"]


def test_rotated_source_is_rejected_even_if_fragment_and_cell_ids_are_unchanged():
    mesh, surface, inventory, _ = fixture()
    rotated = rotate_surface(surface, np.array([[.01, .02, .03]]*2), .25)
    with pytest.raises(ValueError, match="noninitial footprint"):
        legacy_polarity_evidence(mesh, rotated, inventory)


def test_multiple_fragments_occupying_one_provenance_cell_are_rejected():
    mesh, surface, inventory, _ = fixture()
    duplicate = replace(surface.fragments[1], parcel=replace(surface.fragments[1].parcel, cell=0))
    mixed = replace(surface, fragments=(surface.fragments[0], duplicate)+surface.fragments[2:])
    with pytest.raises(ValueError, match="mixed"):
        legacy_polarity_evidence(mesh, mixed, inventory)


@pytest.mark.parametrize("none", [False, True])
def test_empty_starter_inventory_supplies_no_orientation(none):
    mesh, surface, _, _ = fixture()
    evidence, audit = legacy_polarity_evidence(mesh, surface, None if none else YoungBoundaryState())
    assert evidence == ()
    assert audit["counts"]["geometry_matched_segments"] == 0
    assert audit["counts"]["oriented_contacts"] == 0
