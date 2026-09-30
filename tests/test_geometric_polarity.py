"""Independent local-polarity tests using actual moved spherical intersections."""
from dataclasses import asdict, replace
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tectonics.fractional_surface import FractionalSurfaceState, SurfaceParcel
from tectonics.geometric_contacts import extract_contacts
from tectonics.geometric_polarity import (
    PolarityEvidence, PolarityResolution, contact_polarity_report, make_polarity_resolver,
    transport_polarity_evidence,
)
from tectonics.geometric_surface import candidate_pairs, from_fractional_surface, rotate_surface, split_geometry
from tectonics.geometric_transport import UnresolvedPolarityError, advance_geometric_surface
from tectonics.mesh import build_icosphere
from tectonics.spherical_polygons import clip_hemisphere, intersect_convex


OMEGA = np.array([[0., 0., 0.], [0., 0., .02]])
DT = .1


def source(*, tied=False):
    mesh = build_icosphere(0)
    areas = mesh.physical_cell_areas_km2(100.)
    owners = (mesh.centroids[:, 0] > 0.).astype(int)
    parcels = tuple(SurfaceParcel(i, int(owner), f"origin:{i}", float(area), 2.*area,
        10.*area, (1e12 if tied or owner else 2e12)*area,
        5. if tied or owner else 10., (("damage", i/20.), ("water", (20-i)/20.)))
        for i, (area, owner) in enumerate(zip(areas, owners)))
    return mesh, from_fractional_surface(mesh, FractionalSurfaceState(0., tuple(areas), parcels), 100.)


def birth(cell, plate, area, time, serial):
    return SurfaceParcel(cell, plate, f"birth:{time.hex()}:{serial}", area, 2.*area, 0., 0., 0., ())


def contacts_and_evidence(state, omega=OMEGA, *, subducting=1):
    contacts = extract_contacts(state.fragments, state.radius_km, omega)
    evidence = tuple(PolarityEvidence(f"declared:{c.contact_id}:{subducting}", c.contact_id,
        subducting, 1-subducting, "explicit oriented synthetic fault") for c in contacts)
    return contacts, evidence


def overlaps(state, omega=OMEGA):
    moved = rotate_surface(state, omega, DT)
    found = []
    for i, j in candidate_pairs(moved.fragments):
        a, b = moved.fragments[i], moved.fragments[j]
        if a.parcel.plate == b.parcel.plate:
            continue
        polygon = intersect_convex(a.polygon, b.polygon)
        if len(polygon):
            found.append((a, b, polygon))
    assert found
    return found


def matched_overlap(state, *, omega=OMEGA, subducting=1):
    contacts, evidence = contacts_and_evidence(state, omega, subducting=subducting)
    resolver = make_polarity_resolver(state, omega, DT, contacts=contacts, evidence=evidence)
    for a, b, polygon in overlaps(state, omega):
        resolution = resolver(a, b, polygon)
        if resolution is not None and resolution.comparison:
            return contacts, evidence, a, b, polygon, resolution
    raise AssertionError("No real convergent overlap acquired its declared local contact polarity")


def test_explicit_local_memory_overrides_reversed_buoyancy_and_age_order():
    _, state = source()
    _, _, a, b, _, resolution = matched_overlap(state, subducting=1)
    # Plate 1 is younger and lighter, deliberately opposite the fresh fallback.
    assert resolution.comparison == (1 if a.parcel.plate == 1 else -1)
    assert resolution.evidence_ids and resolution.contact_ids


def test_reversing_overlap_argument_order_reverses_comparison_only():
    _, state = source()
    contacts, evidence, a, b, polygon, direct = matched_overlap(state)
    resolver = make_polarity_resolver(state, OMEGA, DT, contacts=contacts, evidence=evidence)
    reverse = resolver(b, a, polygon)
    assert reverse is not None
    assert reverse.comparison == -direct.comparison
    assert set(reverse.evidence_ids) == set(direct.evidence_ids)
    assert set(reverse.contact_ids) == set(direct.contact_ids)


def test_same_owner_pair_evidence_on_remote_arc_cannot_choose_local_overlap():
    _, state = source()
    contacts, evidence, a, b, polygon, direct = matched_overlap(state)
    by_id = {c.contact_id: c for c in contacts}
    used = [by_id[identity] for identity in direct.contact_ids]
    midpoint = np.mean([c.midpoint for c in used], axis=0)
    distant = max((e for e in evidence if e.contact_id not in direct.contact_ids),
                  key=lambda e: np.linalg.norm(np.asarray(by_id[e.contact_id].midpoint)-midpoint))
    resolver = make_polarity_resolver(state, OMEGA, DT, contacts=contacts, evidence=(distant,))
    assert resolver(a, b, polygon) is None


def test_changed_receiver_material_cannot_reuse_an_old_local_contact():
    _, state = source()
    contacts, evidence, a, b, polygon, _ = matched_overlap(state)
    resolver = make_polarity_resolver(state, OMEGA, DT, contacts=contacts, evidence=evidence)
    # The declared receiver is plate 0. Its footprint/ID alone cannot authorize
    # silently replacing the supporting material history.
    changed_a = replace(a, parcel=replace(a.parcel, material_id="other:history")) if a.parcel.plate == 0 else a
    changed_b = replace(b, parcel=replace(b.parcel, material_id="other:history")) if b.parcel.plate == 0 else b
    assert resolver(changed_a, changed_b, polygon) is None


def test_common_coordinate_rotation_preserves_local_polarity():
    _, state = source()
    contacts, evidence, a, b, polygon, direct = matched_overlap(state)
    matrix = Rotation.from_rotvec([.431, -.267, .182]).as_matrix()
    rotated = replace(state, fragments=tuple(replace(f,
        polygon=tuple(map(tuple, np.asarray(f.polygon)@matrix.T))) for f in state.fragments))
    omega = OMEGA@matrix.T
    rotated_contacts = extract_contacts(rotated.fragments, rotated.radius_km, omega)
    assert {c.contact_id for c in rotated_contacts} == {c.contact_id for c in contacts}
    resolver = make_polarity_resolver(rotated, omega, DT, contacts=rotated_contacts, evidence=evidence)
    moved = {f.fragment_id: f for f in rotate_surface(rotated, omega, DT).fragments}
    actual = resolver(moved[a.fragment_id], moved[b.fragment_id], polygon@matrix.T)
    assert actual is not None
    assert actual.comparison == direct.comparison
    assert set(actual.evidence_ids) == set(direct.evidence_ids)


def test_owner_relabeling_and_fragment_reordering_preserve_the_physical_choice():
    _, state = source()
    _, evidence, a, b, polygon, direct = matched_overlap(state)
    relabeled = replace(state, fragments=tuple(replace(f,
        parcel=replace(f.parcel, plate=1-f.parcel.plate)) for f in reversed(state.fragments)))
    omega = OMEGA[::-1]
    contacts = extract_contacts(relabeled.fragments, relabeled.radius_km, omega)
    relabeled_evidence = tuple(replace(e, subducting_plate=1-e.subducting_plate,
        overriding_plate=1-e.overriding_plate) for e in evidence)
    resolver = make_polarity_resolver(relabeled, omega, DT, contacts=contacts, evidence=relabeled_evidence)
    moved = {f.fragment_id: f for f in rotate_surface(relabeled, omega, DT).fragments}
    actual = resolver(moved[a.fragment_id], moved[b.fragment_id], polygon)
    assert actual is not None and actual.comparison == direct.comparison
    assert set(actual.evidence_ids) == set(direct.evidence_ids)


def test_opposite_local_histories_block_fallback_even_with_unequal_buoyancy():
    mesh, state = source()
    contacts, evidence = contacts_and_evidence(state, subducting=1)
    opposite = tuple(replace(e, evidence_id="opposite:"+e.evidence_id,
        subducting_plate=0, overriding_plate=1) for e in evidence)
    resolver = make_polarity_resolver(state, OMEGA, DT, contacts=contacts, evidence=evidence+opposite)
    resolutions = [resolver(a, b, polygon) for a, b, polygon in overlaps(state)]
    assert any(r is not None and r.comparison == 0 for r in resolutions)
    snapshot = json.dumps(asdict(state), sort_keys=True)
    calls = []
    def forbidden_birth(*args):
        calls.append(args)
        raise AssertionError("Conflicting physical polarity must reject before birth")
    with pytest.raises(UnresolvedPolarityError):
        advance_geometric_surface(mesh, state, OMEGA, DT, birth_factory=forbidden_birth,
                                  polarity_resolver=resolver)
    assert not calls
    assert json.dumps(asdict(state), sort_keys=True) == snapshot


def test_scalar_damage_and_water_do_not_supply_missing_fault_orientation():
    mesh, state = source(tied=True)
    contacts = extract_contacts(state.fragments, state.radius_km, OMEGA)
    resolver = make_polarity_resolver(state, OMEGA, DT, contacts=contacts, evidence=())
    assert all(resolver(a, b, polygon) is None for a, b, polygon in overlaps(state))
    with pytest.raises(UnresolvedPolarityError):
        advance_geometric_surface(mesh, state, OMEGA, DT, birth_factory=birth,
                                  polarity_resolver=resolver)


@pytest.mark.parametrize("change", [
    {"evidence_id": ""}, {"contact_id": ""}, {"provenance": ""},
    {"evidence_id": 4}, {"provenance": []}, {"subducting_plate": True},
    {"subducting_plate": .5}, {"subducting_plate": -1},
    {"overriding_plate": 0}, {"overriding_plate": False},
])
def test_evidence_requires_valid_oriented_identity_and_provenance(change):
    values = dict(evidence_id="history", contact_id="contact", subducting_plate=0,
                  overriding_plate=1, provenance="oriented seed")
    with pytest.raises(ValueError):
        PolarityEvidence(**(values | change))


@pytest.mark.parametrize("comparison,basis", [(True, "seed"), (1., "seed"),
    (2, "seed"), (-2, "seed"), (0, ""), (1, 5), (1, []), (1, None)])
def test_callback_resolution_requires_strict_direction_and_provenance(comparison, basis):
    with pytest.raises(ValueError):
        PolarityResolution(comparison, basis)


def test_contact_report_retains_contradictory_histories_and_rejects_duplicate_evidence():
    _, state = source()
    contacts, evidence = contacts_and_evidence(state)
    selected = evidence[0]
    opposed = replace(selected, evidence_id="opposed", subducting_plate=0, overriding_plate=1)
    rows = contact_polarity_report(contacts, (selected, opposed))
    assert len(rows) == 1 and rows[0]["status"] == "conflicting"
    assert set(map(tuple, rows[0]["directions"])) == {(0, 1), (1, 0)}
    assert set(rows[0]["evidence_ids"]) == {selected.evidence_id, opposed.evidence_id}
    with pytest.raises(ValueError, match="unique"):
        contact_polarity_report(contacts, (selected, selected))


@pytest.mark.parametrize("kind", ["absent_contact", "wrong_pair", "untyped_record", "stale_geometry"])
def test_resolver_rejects_unmatched_or_forged_local_evidence(kind):
    _, state = source()
    contacts, evidence = contacts_and_evidence(state)
    one = evidence[0]
    if kind == "absent_contact":
        evidence = (replace(one, contact_id="absent:contact"),)
    elif kind == "wrong_pair":
        evidence = (replace(one, overriding_plate=2),)
    elif kind == "untyped_record":
        evidence = (asdict(one),)
    else:
        contacts = (replace(contacts[0], length_km=contacts[0].length_km*1.01),)+contacts[1:]
    with pytest.raises(ValueError):
        make_polarity_resolver(state, OMEGA, DT, contacts=contacts, evidence=evidence)


@pytest.mark.parametrize("dt", [0., -1., float("nan"), float("inf")])
def test_resolver_rejects_invalid_advancement_time(dt):
    _, state = source()
    contacts, evidence = contacts_and_evidence(state)
    with pytest.raises(ValueError):
        make_polarity_resolver(state, OMEGA, dt, contacts=contacts, evidence=evidence)


def test_external_polarity_conditions_follow_common_rotation_without_creating_new_origins():
    _, before = source()
    omega = np.array([[.02, -.03, .01]]*2)
    contacts_before, evidence = contacts_and_evidence(before, omega)
    after = rotate_surface(before, omega, DT)
    contacts_after = extract_contacts(after.fragments, after.radius_km, omega)
    propagated, unresolved = transport_polarity_evidence(before, after, omega, DT,
        contacts_before=contacts_before, contacts_after=contacts_after, evidence=evidence)
    assert unresolved == ()
    assert set(propagated) == set(evidence)


def test_external_condition_splits_and_merges_only_on_its_local_receiver_material():
    _, before = source()
    omega = np.zeros((2, 3))
    contacts_before, evidence = contacts_and_evidence(before, omega)
    original = contacts_before[0]
    condition = next(e for e in evidence if e.contact_id == original.contact_id)
    receiver_id = original.fragment_a if original.plate_a == 0 else original.fragment_b
    advanced = rotate_surface(before, omega, DT)
    receiver = next(f for f in advanced.fragments if f.fragment_id == receiver_id)
    midpoint = np.asarray(original.midpoint)
    interior = np.mean(receiver.polygon, axis=0)
    normal = np.cross(midpoint, interior)
    normal /= np.linalg.norm(normal)
    parts = split_geometry(receiver, (clip_hemisphere(receiver.polygon, normal),
        clip_hemisphere(receiver.polygon, -normal)), label="receiver-refinement")
    after = replace(advanced, fragments=tuple(f for f in advanced.fragments if f.fragment_id != receiver_id)+parts)
    contacts_after = extract_contacts(after.fragments, after.radius_km, omega)
    propagated, unresolved = transport_polarity_evidence(before, after, omega, DT,
        contacts_before=contacts_before, contacts_after=contacts_after, evidence=(condition,))
    assert unresolved == () and len(propagated) == 2
    assert {e.origin_evidence_id for e in propagated} == {condition.origin_evidence_id}
    next_surface = rotate_surface(advanced, omega, DT)
    next_contacts = extract_contacts(next_surface.fragments, next_surface.radius_km, omega)
    merged, unresolved = transport_polarity_evidence(after, next_surface, omega, DT,
        contacts_before=contacts_after, contacts_after=next_contacts, evidence=propagated)
    assert unresolved == () and len(merged) == 1
    assert merged[0].contact_id == condition.contact_id
    assert merged[0].origin_evidence_id == condition.origin_evidence_id
    # A condition known on only one child cannot be expanded over the whole
    # original arc just because geometric fragments were merged again.
    partial, unresolved = transport_polarity_evidence(after, next_surface, omega, DT,
        contacts_before=contacts_after, contacts_after=next_contacts, evidence=propagated[:1])
    assert partial == () and len(unresolved) == 1
    assert unresolved[0]["reason"] == "partial_polarity_contact_coverage"
    assert unresolved[0]["coverage_fraction"] == pytest.approx(.5, rel=1e-12)


def test_external_condition_does_not_teleport_to_remote_same_pair_contact():
    _, before = source()
    omega = np.zeros((2, 3))
    contacts_before, evidence = contacts_and_evidence(before, omega)
    condition = evidence[0]
    original = next(c for c in contacts_before if c.contact_id == condition.contact_id)
    after = rotate_surface(before, omega, DT)
    contacts_after = extract_contacts(after.fragments, after.radius_km, omega)
    distant = max(contacts_after, key=lambda c: np.linalg.norm(
        np.asarray(c.midpoint)-np.asarray(original.midpoint)))
    propagated, unresolved = transport_polarity_evidence(before, after, omega, DT,
        contacts_before=contacts_before, contacts_after=(distant,), evidence=(condition,))
    assert propagated == () and len(unresolved) == 1
    assert unresolved[0]["origin_evidence_id"] == condition.origin_evidence_id
    assert unresolved[0]["reason"] == "unresolved_polarity_lineage"


def test_external_condition_disappears_when_receiver_material_is_replaced():
    _, before = source()
    omega = np.zeros((2, 3))
    contacts_before, evidence = contacts_and_evidence(before, omega)
    condition = evidence[0]
    original = next(c for c in contacts_before if c.contact_id == condition.contact_id)
    receiver_id = original.fragment_a if original.plate_a == 0 else original.fragment_b
    advanced = rotate_surface(before, omega, DT)
    fragments = tuple(replace(f, parcel=replace(f.parcel, material_id="replacement:receiver"))
        if f.fragment_id == receiver_id else f for f in advanced.fragments)
    after = replace(advanced, fragments=fragments,
        known_material_ids=advanced.known_material_ids+("replacement:receiver",))
    contacts_after = extract_contacts(after.fragments, after.radius_km, omega)
    propagated, unresolved = transport_polarity_evidence(before, after, omega, DT,
        contacts_before=contacts_before, contacts_after=contacts_after, evidence=(condition,))
    assert propagated == () and len(unresolved) == 1
