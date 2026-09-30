"""Independent geometry/lineage tests: no reconstruction from fractional areas."""
from dataclasses import asdict, replace
import json
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tectonics.fractional_surface import FractionalSurfaceState, SurfaceParcel, split_parcel
from tectonics.geometric_contacts import contacts_between, extract_contacts
from tectonics.geometric_surface import (
    GeometricFragment, GeometricSurfaceState, audit_partition, from_fractional_surface,
    load_geometric_checkpoint, project_to_mesh, rotate_surface, save_geometric_checkpoint,
)
from tectonics.mesh import build_icosphere


EXTENSIVE = ("area_km2", "oceanic_volume_km3", "cold_mantle_volume_km3", "density_excess_mass_kg")


def fractional_fixture(mesh, radius=100., owners=None):
    areas = mesh.physical_cell_areas_km2(radius)
    if owners is None:
        owners = (mesh.centroids[:, 0] > 0.).astype(int)
    parcels = tuple(SurfaceParcel(i, int(owners[i]), f"source:{i}", float(a),
        float(a)*2., float(a)*(10.+i), float(a)*(1e12+i*1e10), 5.+i/10.,
        (("damage", i/len(areas)), ("water", .2))) for i, a in enumerate(areas))
    return FractionalSurfaceState(0., tuple(areas), parcels)


def grouped(parcels, key):
    accum = {}
    for parcel in parcels:
        accum.setdefault(parcel.material_id, []).append(getattr(parcel, key))
    return {identity: math.fsum(parts) for identity, parts in accum.items()}


def assert_same_material(first, second):
    first, second = tuple(first), tuple(second)
    for key in EXTENSIVE:
        expected, actual = grouped(first, key), grouped(second, key)
        assert actual.keys() == expected.keys()
        for identity, total in expected.items():
            assert actual[identity] == pytest.approx(total, rel=5e-12, abs=0.)


def frozen_snapshot(state):
    return json.dumps(asdict(state), sort_keys=True, allow_nan=False)


@pytest.mark.parametrize("subdivisions", [0, 1])
def test_pure_cell_import_uses_exact_saved_triangle_and_material(subdivisions):
    mesh = build_icosphere(subdivisions)
    source = fractional_fixture(mesh)
    state = from_fractional_surface(mesh, source, 100.)
    assert len(state.fragments) == mesh.cell_count
    assert len({f.fragment_id for f in state.fragments}) == mesh.cell_count
    assert state.known_material_ids == source.known_material_ids
    assert state.time_myr == source.time_myr
    assert state.radius_km == 100.
    for fragment in state.fragments:
        assert fragment.parcel == source.parcels[fragment.parcel.cell]
        expected = mesh.vertices[mesh.faces[fragment.parcel.cell]]
        # Polygon orientation/cyclic starting vertex can legitimately change.
        distances = np.linalg.norm(np.asarray(fragment.polygon)[:, None, :]-expected[None, :, :], axis=2)
        np.testing.assert_allclose(distances.min(axis=1), 0., atol=3e-15)
    assert_same_material(source.parcels, (f.parcel for f in state.fragments))


def test_mixed_fractional_checkpoint_cannot_silently_invent_a_boundary():
    mesh = build_icosphere(0)
    source = fractional_fixture(mesh)
    first = source.parcels[0]
    half = split_parcel(first, .5)
    second = replace(half, material_id="other-half", plate=1-first.plate)
    mixed = replace(source, parcels=(half, second)+source.parcels[1:],
                    known_material_ids=source.known_material_ids+("other-half",))
    with pytest.raises(ValueError):
        from_fractional_surface(mesh, mixed, 100.)


def test_even_two_same_owner_histories_need_explicit_geometry():
    mesh = build_icosphere(0)
    source = fractional_fixture(mesh)
    half = split_parcel(source.parcels[0], .5)
    mixed = replace(source, parcels=(half, half)+source.parcels[1:])
    with pytest.raises(ValueError):
        from_fractional_surface(mesh, mixed, 100.)


def test_common_rotation_preserves_exact_lineage_and_all_material_extents():
    mesh = build_icosphere(0)
    state = from_fractional_surface(mesh, fractional_fixture(mesh), 100.)
    before = frozen_snapshot(state)
    omega = np.array([[.021, -.033, .046]]*2)
    moved = rotate_surface(state, omega, .7)
    expected = Rotation.from_rotvec(omega[0]*.7)
    assert frozen_snapshot(state) == before
    assert moved.time_myr == pytest.approx(.7)
    assert len(moved.fragments) == len(state.fragments)
    by_id = {f.fragment_id: f for f in moved.fragments}
    for old in state.fragments:
        new = by_id[old.fragment_id]
        np.testing.assert_allclose(new.polygon, expected.apply(old.polygon), atol=3e-15, rtol=0.)
        assert new.parcel.age_myr == pytest.approx(old.parcel.age_myr+.7)
        assert new.parcel.material_fields == old.parcel.material_fields
        assert new.parcel.specific_properties == old.parcel.specific_properties
        for field in EXTENSIVE:
            assert getattr(new.parcel, field) == getattr(old.parcel, field)


def test_full_turn_returns_same_geometry_and_contact_material():
    mesh = build_icosphere(0)
    state = from_fractional_surface(mesh, fractional_fixture(mesh), 100.)
    direction = np.array([1., 2., -3.])/math.sqrt(14.)
    moved = rotate_surface(state, np.tile(direction*2.*math.pi, (2, 1)), 1.)
    for first, final in zip(state.fragments, moved.fragments):
        assert first.fragment_id == final.fragment_id
        np.testing.assert_allclose(first.polygon, final.polygon, atol=2e-14, rtol=0.)
    assert_same_material((f.parcel for f in state.fragments), (f.parcel for f in moved.fragments))


def test_projection_creates_actual_mixed_cells_and_preserves_each_origin_budget():
    mesh = build_icosphere(0)
    state = from_fractional_surface(mesh, fractional_fixture(mesh), 100.)
    state = rotate_surface(state, np.array([[.021, -.033, .046]]*2), .7)
    before = frozen_snapshot(state)
    projected = project_to_mesh(mesh, state)
    assert frozen_snapshot(state) == before
    assert_same_material((f.parcel for f in state.fragments), (f.parcel for f in projected))
    source_ids = {f.fragment_id for f in state.fragments}
    assert all(f.parent_fragment_id in source_ids or f.fragment_id in source_ids for f in projected)
    area = np.bincount([f.parcel.cell for f in projected], weights=[f.parcel.area_km2 for f in projected],
                       minlength=mesh.cell_count)
    np.testing.assert_allclose(area, mesh.physical_cell_areas_km2(100.), rtol=5e-12, atol=0.)
    owners = {}
    for fragment in projected:
        owners.setdefault(fragment.parcel.cell, set()).add(fragment.parcel.plate)
    assert sum(len(ids)>1 for ids in owners.values()) > 0


def test_rebinning_refines_integration_only_without_changing_physical_state():
    coarse, fine = build_icosphere(0), build_icosphere(1)
    state = from_fractional_surface(coarse, fractional_fixture(coarse), 100.)
    state = rotate_surface(state, np.array([[.021, -.033, .046]]*2), .7)
    before = frozen_snapshot(state)
    projected = project_to_mesh(fine, state)
    assert frozen_snapshot(state) == before
    assert len(projected) > len(state.fragments)
    assert_same_material((f.parcel for f in state.fragments), (f.parcel for f in projected))
    area = np.bincount([f.parcel.cell for f in projected], weights=[f.parcel.area_km2 for f in projected],
                       minlength=fine.cell_count)
    np.testing.assert_allclose(area, fine.physical_cell_areas_km2(100.), rtol=5e-12, atol=0.)


def test_geometry_and_rotation_are_covariant_under_independent_frame_rotation():
    mesh = build_icosphere(0)
    state = from_fractional_surface(mesh, fractional_fixture(mesh), 100.)
    frame = Rotation.from_rotvec([.7, -.2, .5])
    framed = replace(state, fragments=tuple(replace(f,
        polygon=tuple(map(tuple, frame.apply(f.polygon)))) for f in state.fragments))
    omega = np.array([[.021, -.033, .046], [-.011, .023, -.025]])
    normal = rotate_surface(state, omega, .7)
    transformed = rotate_surface(framed, frame.apply(omega), .7)
    for first, second in zip(normal.fragments, transformed.fragments):
        np.testing.assert_allclose(frame.apply(first.polygon), second.polygon, atol=3e-14, rtol=0.)
        assert first.parcel == second.parcel


def test_identical_owner_fractions_can_have_different_contact_directions():
    mesh = build_icosphere(0)
    source = fractional_fixture(mesh).parcels[0]
    half = split_parcel(source, .5)
    material_a = replace(half, plate=0, material_id="half:a")
    material_b = replace(half, plate=1, material_id="half:b")
    vertices = mesh.vertices[mesh.faces[0]]
    layouts = []
    for shift in (0, 1):
        a, b, c = np.roll(vertices, shift, axis=0)
        midpoint = (b+c)/np.linalg.norm(b+c)
        layouts.append((
            GeometricFragment("a", tuple(map(tuple, (a, b, midpoint))), material_a),
            GeometricFragment("b", tuple(map(tuple, (a, midpoint, c))), material_b),
        ))
    first = contacts_between(*layouts[0], 100.)
    second = contacts_between(*layouts[1], 100.)
    assert len(first) == len(second) == 1
    # Cell labels, owner areas and histories are identical in the two cases.
    assert tuple(f.parcel for f in layouts[0]) == tuple(f.parcel for f in layouts[1])
    assert first[0].length_km == pytest.approx(second[0].length_km, rel=3e-14)
    assert abs(np.dot(first[0].normal_a_to_b, second[0].normal_a_to_b)) < .9
    assert np.linalg.norm(np.array(first[0].midpoint)-second[0].midpoint) > .1


def test_contact_length_and_identity_survive_common_rotation_and_rebinning():
    mesh, fine = build_icosphere(0), build_icosphere(1)
    state = from_fractional_surface(mesh, fractional_fixture(mesh), 100.)
    initial = extract_contacts(state.fragments, 100.)
    moved = rotate_surface(state, np.array([[.021, -.033, .046]]*2), .7)
    actual = extract_contacts(moved.fragments, 100.)
    assert len(actual) == len(initial) > 0
    first_by_id = {c.contact_id: c for c in initial}
    for contact in actual:
        assert contact.contact_id in first_by_id
        assert contact.length_km == pytest.approx(first_by_id[contact.contact_id].length_km, rel=5e-13)
    before = frozen_snapshot(moved)
    project_to_mesh(fine, moved)
    assert frozen_snapshot(moved) == before
    assert extract_contacts(moved.fragments, 100.) == actual


@pytest.mark.parametrize("dt", [0., -1., float("nan"), float("inf")])
def test_invalid_rotation_timestep_is_atomic(dt):
    mesh = build_icosphere(0)
    state = from_fractional_surface(mesh, fractional_fixture(mesh), 100.)
    before = frozen_snapshot(state)
    with pytest.raises(ValueError):
        rotate_surface(state, np.array([[.01, .02, .03]]*2), dt)
    assert frozen_snapshot(state) == before


@pytest.mark.parametrize("omega", [np.zeros((1, 3)), np.zeros((2, 2)), np.full((2, 3), np.nan)])
def test_invalid_rotation_arrays_are_rejected(omega):
    mesh = build_icosphere(0)
    state = from_fractional_surface(mesh, fractional_fixture(mesh), 100.)
    with pytest.raises(ValueError):
        rotate_surface(state, omega, .5)


def test_independent_rotations_are_explicit_unresolved_candidates():
    mesh = build_icosphere(0)
    state = from_fractional_surface(mesh, fractional_fixture(mesh), 100.)
    moved = rotate_surface(state, np.array([[.021, -.033, .046], [-.011, .023, -.025]]), .7)
    assert moved.phase == "advected_candidates"
    with pytest.raises(ValueError, match="not resolved"):
        audit_partition(moved)
    # A later common rotation cannot silently bless the unresolved overlaps.
    still_unresolved = rotate_surface(moved, np.array([[.021, -.033, .046]]*2), .7)
    assert still_unresolved.phase == "advected_candidates"


def test_checkpoint_resume_reproduces_polygon_bits_lineage_contact_ids_and_clock(tmp_path):
    mesh = build_icosphere(0)
    initial = from_fractional_surface(mesh, fractional_fixture(mesh), 100.)
    omega = np.array([[.021, -.033, .046]]*2)
    first = rotate_surface(initial, omega, .7)
    direct = rotate_surface(first, omega, .7)
    path = tmp_path/"geometry.json"
    metadata = {"source_sha256": "test-provenance", "step": 1}
    save_geometric_checkpoint(path, first, provenance=metadata)
    loaded, provenance = load_geometric_checkpoint(path)
    assert frozen_snapshot(loaded) == frozen_snapshot(first)
    assert provenance == metadata
    continued = rotate_surface(loaded, omega, .7)
    assert frozen_snapshot(continued) == frozen_snapshot(direct)
    assert extract_contacts(continued.fragments, 100.) == extract_contacts(direct.fragments, 100.)


def test_checkpoint_preserves_unresolved_candidate_status(tmp_path):
    mesh = build_icosphere(0)
    state = from_fractional_surface(mesh, fractional_fixture(mesh), 100.)
    candidate = rotate_surface(state, np.array([[0., 0., .01], [0., 0., -.01]]), 1.)
    path = tmp_path/"candidate.json"
    save_geometric_checkpoint(path, candidate)
    loaded, _ = load_geometric_checkpoint(path)
    assert frozen_snapshot(loaded) == frozen_snapshot(candidate)
    assert loaded.phase == "advected_candidates"


def test_geometric_checkpoint_refuses_tampering_and_existing_output(tmp_path):
    mesh = build_icosphere(0)
    state = from_fractional_surface(mesh, fractional_fixture(mesh), 100.)
    path = tmp_path/"geometry.json"
    save_geometric_checkpoint(path, state)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_geometric_checkpoint(path, state)
    assert path.read_bytes() == original
    data = json.loads(original)
    data["state"]["fragments"][0]["polygon"][0][0] += .01
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        load_geometric_checkpoint(path)


def test_fractional_only_checkpoint_is_explicitly_not_a_geometry_checkpoint(tmp_path):
    path = tmp_path/"fractional.json"
    path.write_text(json.dumps({"format": "fractional-surface-checkpoint-1", "provenance": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="persisted spherical geometry"):
        load_geometric_checkpoint(path)
