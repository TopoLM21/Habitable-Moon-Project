"""Independent joint geometric-surface/passive-boundary checkpoint contract."""
from dataclasses import asdict, replace
import hashlib
import json

import numpy as np
import pytest

from tectonics.fractional_surface import FractionalSurfaceState, SurfaceParcel
from tectonics.geometric_boundary import initialize_boundary, consume_geometric_transaction
from tectonics.geometric_boundary_io import load_boundary_checkpoint, save_boundary_checkpoint
from tectonics.geometric_surface import from_fractional_surface, save_geometric_checkpoint
from tectonics.geometric_transport import advance_geometric_surface
from tectonics.mesh import build_icosphere


@pytest.fixture
def accepted():
    mesh = build_icosphere(0)
    areas = mesh.physical_cell_areas_km2(100.)
    owners = (mesh.centroids[:, 0] > 0.).astype(int)
    parcels = tuple(SurfaceParcel(i, int(owner), f"history:{i}", float(area), 2.*area, 8.*area,
        (1e12 if owner else 2e12)*area, 5. if owner else 10., ())
        for i, (area, owner) in enumerate(zip(areas, owners)))
    initial = from_fractional_surface(mesh, FractionalSurfaceState(0., tuple(areas), parcels), 100.)
    def birth(cell, plate, area, time, serial):
        return SurfaceParcel(cell, plate, f"birth:{time.hex()}:{serial}", area, 2.*area, 0., 0., 0., ())
    motion = np.array([[0., 0., 0.], [0., 0., .01]])
    result = advance_geometric_surface(mesh, initial, motion, .1, birth_factory=birth)
    boundary = consume_geometric_transaction(initial, result, initialize_boundary(initial), motion, .1)
    return initial, result.state, boundary


def test_joint_checkpoint_exactly_roundtrips_geometry_material_and_passive_history(accepted, tmp_path):
    _, surface, boundary = accepted
    path = tmp_path/"joint.json"
    provenance = {"experiment": "passive", "evidence": ["declared local history"], "dt_myr": .1}
    save_boundary_checkpoint(path, surface, boundary, provenance=provenance)
    loaded, inventory, metadata = load_boundary_checkpoint(path)
    assert asdict(loaded) == asdict(surface)
    assert asdict(inventory) == asdict(boundary)
    assert metadata == provenance
    assert inventory.cohorts


def test_mismatched_surface_and_inventory_rejected_before_file_creation(accepted, tmp_path):
    initial, _, boundary = accepted
    path = tmp_path/"mismatch.json"
    with pytest.raises(ValueError):
        save_boundary_checkpoint(path, initial, boundary)
    assert not path.exists()


def test_equal_time_but_different_material_history_cannot_bind_inventory(accepted, tmp_path):
    _, surface, boundary = accepted
    fragment = surface.fragments[0]
    different = replace(surface, fragments=(replace(fragment, parcel=replace(fragment.parcel,
        material_fields=(("damage", .123),))),)+surface.fragments[1:])
    with pytest.raises(ValueError):
        save_boundary_checkpoint(tmp_path/"wrong-history.json", different, boundary)


def test_geometry_only_checkpoint_cannot_silently_initialize_missing_slab_state(accepted, tmp_path):
    _, surface, _ = accepted
    path = tmp_path/"old-format.json"
    save_geometric_checkpoint(path, surface, provenance={"no_mechanical_state": True})
    with pytest.raises(ValueError):
        load_boundary_checkpoint(path)


@pytest.mark.parametrize("target", ["provenance", "surface", "boundary"])
def test_joint_checkpoint_tampering_is_rejected(accepted, tmp_path, target):
    _, surface, boundary = accepted
    path = tmp_path/"tampered.json"
    save_boundary_checkpoint(path, surface, boundary, provenance={"seed": "original"})
    text = path.read_text(encoding="utf-8")
    if target == "provenance":
        assert "original" in text
        text = text.replace("original", "changed", 1)
    elif target == "surface":
        assert "history:0" in text
        text = text.replace("history:0", "foreign:0", 1)
    else:
        assert boundary.cohorts[0].event_id in text
        text = text.replace(boundary.cohorts[0].event_id, "forged-event", 1)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_boundary_checkpoint(path)


def test_checkpoint_save_never_overwrites_existing_file(accepted, tmp_path):
    _, surface, boundary = accepted
    path = tmp_path/"already-exists.json"
    original = b"preserve unrelated result"
    path.write_bytes(original)
    with pytest.raises((ValueError, FileExistsError)):
        save_boundary_checkpoint(path, surface, boundary)
    assert path.read_bytes() == original


def test_nonfinite_provenance_cannot_publish_a_checkpoint(accepted, tmp_path):
    _, surface, boundary = accepted
    path = tmp_path/"nan.json"
    with pytest.raises(ValueError):
        save_boundary_checkpoint(path, surface, boundary,
                                 provenance={"bad": float("nan")})
    assert not path.exists()


@pytest.mark.parametrize("target", ["radius", "contact_geometry"])
def test_joint_writer_requires_inventory_geometry_to_match_bound_surface(accepted, tmp_path, target):
    _, surface, boundary = accepted
    if target == "radius":
        altered = replace(boundary, radius_km=boundary.radius_km*2.)
    else:
        contact = boundary.contacts[0]
        altered = replace(boundary, contacts=(replace(contact, length_km=contact.length_km*1.01),)
            +boundary.contacts[1:])
    path = tmp_path/"semantically-wrong.json"
    with pytest.raises(ValueError):
        save_boundary_checkpoint(path, surface, altered)
    assert not path.exists()


@pytest.mark.parametrize("target", ["radius", "contact_geometry"])
def test_joint_reader_checks_semantics_even_when_outer_checksum_was_recomputed(accepted, tmp_path, target):
    _, surface, boundary = accepted
    path = tmp_path/"rehashed.json"
    save_boundary_checkpoint(path, surface, boundary)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("payload_sha256")
    if target == "radius":
        payload["boundary"]["radius_km"] *= 2.
    else:
        payload["boundary"]["contacts"][0]["length_km"] *= 1.01
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                         allow_nan=False).encode("utf-8")
    payload["payload_sha256"] = hashlib.sha256(encoded).hexdigest()
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError):
        load_boundary_checkpoint(path)
