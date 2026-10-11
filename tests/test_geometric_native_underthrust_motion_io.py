"""A native current pair must remain bound to the replayed physical seed."""
from dataclasses import asdict
from functools import lru_cache
import copy
import hashlib
import json

import pytest

from tests.underthrust_case import physical_inputs
from tectonics.geometric_native_buried_support import native_buried_registry_to_dict
from tectonics.geometric_native_surface import native_surface_to_dict
from tectonics.geometric_native_underthrust_motion import initialize_native_underthrust_motion
from tectonics.geometric_native_underthrust_motion_io import (
    FORMAT, _encoded, load_native_underthrust_motion_checkpoint,
    save_native_underthrust_motion_checkpoint,
)
from tectonics.geometric_surface import save_geometric_checkpoint
from tectonics.geometric_underthrust_motion import HotBirthParameters


@lru_cache(maxsize=None)
def initial_case(*, owners=(0, 1)):
    mesh, origin, inventory, seed, load = physical_inputs(owners=owners)
    state = initialize_native_underthrust_motion(mesh, origin, inventory, seed, load,
        birth_parameters=HotBirthParameters(2., 0., 0.,
            "Explicit hot local-vacancy birth closure for native checkpoint tests"),
        max_anchor_transport_error_m=1e-4)
    return mesh, origin, state


@pytest.fixture(scope="module")
def initial_checkpoint_bytes(tmp_path_factory):
    mesh, origin, state = initial_case()
    path = tmp_path_factory.mktemp("native-physical-joint-checkpoint") / "initial.json"
    save_native_underthrust_motion_checkpoint(path, mesh, origin, state)
    return path.read_bytes()


def write_rehashed(path, body):
    body = copy.deepcopy(body)
    body.pop("payload_sha256", None)
    body["payload_sha256"] = hashlib.sha256(_encoded(body)).hexdigest()
    path.write_bytes(_encoded(body) + b"\n")


def test_initial_native_joint_restart_preserves_authoritative_geometry_and_exact_bytes(tmp_path):
    mesh, origin, state = initial_case()
    path, restored_path = tmp_path / "initial.json", tmp_path / "restored.json"
    provenance = {"purpose": "Native pair bound to original physical seed", "stages": []}
    save_native_underthrust_motion_checkpoint(path, mesh, origin, state, provenance=provenance)
    restored_origin, restored, user = load_native_underthrust_motion_checkpoint(path, mesh=mesh)
    assert (restored_origin, restored, user) == (origin, state, provenance)
    save_native_underthrust_motion_checkpoint(restored_path, mesh, restored_origin,
                                             restored, provenance=user)
    assert path.read_bytes() == restored_path.read_bytes()
    body = json.loads(path.read_bytes())
    assert body["format"] == FORMAT
    assert body["origin_surface"] == json.loads(_encoded(asdict(origin)))
    assert body["current_surface"] == json.loads(_encoded(native_surface_to_dict(state.current_surface)))
    assert body["current_inventory"] == json.loads(_encoded(
        native_buried_registry_to_dict(state.current_inventory)))
    assert all("geometry" in fragment and "polygon" not in fragment
               for fragment in body["current_surface"]["fragments"])
    assert not state.current_inventory.records
    assert not state.receipts


def test_exclusive_writer_preserves_existing_bytes_and_cleans_temporary_file(tmp_path,
                                                                           initial_checkpoint_bytes):
    mesh, origin, state = initial_case()
    path = tmp_path / "existing.json"
    sentinel = b"This existing checkpoint belongs to the user.\n"
    path.write_bytes(sentinel)
    with pytest.raises(FileExistsError):
        save_native_underthrust_motion_checkpoint(path, mesh, origin, state)
    assert path.read_bytes() == sentinel
    assert tuple(tmp_path.iterdir()) == (path,)


@pytest.mark.parametrize("provenance", [False, 0., "text", []])
def test_writer_rejects_nonobject_provenance_before_creating_parent(tmp_path, provenance):
    mesh, origin, state = initial_case()
    parent = tmp_path / "absent-parent"
    with pytest.raises(ValueError, match="JSON object"):
        save_native_underthrust_motion_checkpoint(parent / "invalid.json", mesh, origin,
                                                 state, provenance=provenance)
    assert not parent.exists()


def test_writer_rejects_nonfinite_provenance_before_creating_parent(tmp_path):
    mesh, origin, state = initial_case()
    parent = tmp_path / "nonfinite-parent"
    with pytest.raises(ValueError):
        save_native_underthrust_motion_checkpoint(parent / "invalid.json", mesh, origin,
                                                 state, provenance={"bad": float("nan")})
    assert not parent.exists()


def test_writer_rejects_wrong_state_before_creating_parent(tmp_path):
    mesh, origin, _ = initial_case()
    parent = tmp_path / "wrong-state-parent"
    with pytest.raises((ValueError, TypeError)):
        save_native_underthrust_motion_checkpoint(parent / "invalid.json", mesh, origin, object())
    assert not parent.exists()


def test_old_surface_writer_cannot_silently_label_native_current_geometry_as_legacy(tmp_path):
    _, _, state = initial_case()
    parent = tmp_path / "legacy-parent"
    with pytest.raises(TypeError):
        save_geometric_checkpoint(parent / "legacy.json", state.current_surface)
    assert not parent.exists()


@pytest.mark.parametrize("mutate", [
    lambda body: body.update(format="geometric-force-generated-underthrust-checkpoint-1"),
    lambda body: body.update(extra="unsupported"),
    lambda body: body.pop("origin_surface"),
    lambda body: body.update(provenance=[]),
    lambda body: body.update(provenance=False),
])
def test_rehashed_root_requires_explicit_native_joint_schema(tmp_path, initial_checkpoint_bytes, mutate):
    mesh, _, _ = initial_case()
    body = json.loads(initial_checkpoint_bytes)
    mutate(body)
    path = tmp_path / "mutant.json"
    write_rehashed(path, body)
    with pytest.raises(ValueError, match="native underthrust checkpoint schema"):
        load_native_underthrust_motion_checkpoint(path, mesh=mesh)


@pytest.mark.parametrize("field,value", [("time_myr", False), ("time_myr", "0.0"),
                                         ("radius_km", "100.0")])
def test_rehashed_original_seed_refuses_numeric_aliases(tmp_path, initial_checkpoint_bytes, field, value):
    mesh, _, _ = initial_case()
    body = json.loads(initial_checkpoint_bytes)
    body["origin_surface"][field] = value
    path = tmp_path / "bad-origin.json"
    write_rehashed(path, body)
    with pytest.raises(ValueError, match="original clock/radius"):
        load_native_underthrust_motion_checkpoint(path, mesh=mesh)


@pytest.mark.parametrize("field,value", [("age_myr", "5.0"), ("age_myr", False),
                                         ("plate", False)])
def test_rehashed_original_material_refuses_numeric_aliases(tmp_path, initial_checkpoint_bytes,
                                                           field, value):
    mesh, _, _ = initial_case()
    body = json.loads(initial_checkpoint_bytes)
    body["origin_surface"]["fragments"][0]["parcel"][field] = value
    path = tmp_path / "bad-origin-material.json"
    write_rehashed(path, body)
    with pytest.raises(ValueError):
        load_native_underthrust_motion_checkpoint(path, mesh=mesh)


@pytest.mark.parametrize("value", [False, "0.0", 1e-8])
def test_rehashed_current_native_clock_cannot_replace_replayed_epoch(tmp_path,
        initial_checkpoint_bytes, value):
    mesh, _, _ = initial_case()
    body = json.loads(initial_checkpoint_bytes)
    body["current_surface"]["time_myr"] = value
    path = tmp_path / "wrong-current-clock.json"
    write_rehashed(path, body)
    with pytest.raises(ValueError):
        load_native_underthrust_motion_checkpoint(path, mesh=mesh)


def test_rehashed_current_registry_must_bind_the_native_surface(tmp_path, initial_checkpoint_bytes):
    mesh, _, _ = initial_case()
    body = json.loads(initial_checkpoint_bytes)
    registry = body["current_inventory"]["registry"]
    registry["current_surface_sha256"] = "0" * 64
    registry_body = dict(registry)
    registry_body.pop("registry_sha256")
    registry["registry_sha256"] = hashlib.sha256(_encoded(registry_body)).hexdigest()
    path = tmp_path / "wrong-current-registry.json"
    write_rehashed(path, body)
    with pytest.raises(ValueError):
        load_native_underthrust_motion_checkpoint(path, mesh=mesh)


def test_rehashed_registry_owner_cannot_use_boolean_integer_alias(tmp_path, initial_checkpoint_bytes):
    mesh, _, _ = initial_case()
    body = json.loads(initial_checkpoint_bytes)
    registry = body["current_inventory"]["registry"]
    assert registry["owner_histories"][0][0] == 0
    registry["owner_histories"][0][0] = False
    registry_body = dict(registry)
    registry_body.pop("registry_sha256")
    registry["registry_sha256"] = hashlib.sha256(_encoded(registry_body)).hexdigest()
    path = tmp_path / "boolean-owner.json"
    write_rehashed(path, body)
    with pytest.raises(ValueError):
        load_native_underthrust_motion_checkpoint(path, mesh=mesh)


def test_rehashed_native_ray_cannot_use_boolean_integer_alias(tmp_path, initial_checkpoint_bytes):
    mesh, _, _ = initial_case()
    body = json.loads(initial_checkpoint_bytes)
    ray = body["current_surface"]["fragments"][0]["geometry"]["vertices"][0]
    assert ray["kind"] == "seed"
    ray["reference"][0] = False
    path = tmp_path / "boolean-native-ray.json"
    write_rehashed(path, body)
    with pytest.raises(ValueError):
        load_native_underthrust_motion_checkpoint(path, mesh=mesh)


def test_rehashed_physical_snapshot_cannot_change_retained_work(tmp_path, initial_checkpoint_bytes):
    mesh, _, _ = initial_case()
    body = json.loads(initial_checkpoint_bytes)
    snapshot = body["native_underthrust_motion"]["snapshot"]
    snapshot["cumulative_source_work_j"] += 1.
    path = tmp_path / "wrong-work.json"
    write_rehashed(path, body)
    with pytest.raises(ValueError, match="snapshot|replay"):
        load_native_underthrust_motion_checkpoint(path, mesh=mesh)


def test_rehashed_physical_snapshot_cannot_use_boolean_zero_work_alias(tmp_path,
                                                                     initial_checkpoint_bytes):
    mesh, _, _ = initial_case()
    body = json.loads(initial_checkpoint_bytes)
    snapshot = body["native_underthrust_motion"]["snapshot"]
    assert snapshot["cumulative_source_work_j"] == 0.
    snapshot["cumulative_source_work_j"] = False
    path = tmp_path / "boolean-work.json"
    write_rehashed(path, body)
    with pytest.raises(ValueError, match="snapshot|replay"):
        load_native_underthrust_motion_checkpoint(path, mesh=mesh)


@pytest.mark.parametrize("target", ["clock", "contact_owner", "contact_vertex"])
def test_rehashed_legacy_preparation_cannot_preserve_boolean_numeric_aliases(tmp_path,
        initial_checkpoint_bytes, target):
    mesh, _, _ = initial_case()
    body = json.loads(initial_checkpoint_bytes)
    motion = body["native_underthrust_motion"]
    for inventory in (motion["initial_inventory"], motion["snapshot"]["initial_inventory"]):
        if target == "clock":
            assert inventory["time_myr"] == 0.
            inventory["time_myr"] = False
        elif target == "contact_owner":
            contact = next(contact for contact in inventory["contacts"] if contact["plate_a"] == 0)
            contact["plate_a"] = False
        else:
            vertex = next(contact["start"] for contact in inventory["contacts"]
                          if any(value == 0. for value in contact["start"]))
            vertex[next(index for index, value in enumerate(vertex) if value == 0.)] = False
    path = tmp_path / "boolean-preparation.json"
    write_rehashed(path, body)
    with pytest.raises(ValueError):
        load_native_underthrust_motion_checkpoint(path, mesh=mesh)


def test_checksum_mismatch_is_rejected_before_replay(tmp_path, initial_checkpoint_bytes):
    mesh, _, _ = initial_case()
    body = json.loads(initial_checkpoint_bytes)
    body["provenance"]["unexpected"] = True
    path = tmp_path / "integrity.json"
    path.write_bytes(_encoded(body) + b"\n")
    with pytest.raises(ValueError, match="integrity mismatch"):
        load_native_underthrust_motion_checkpoint(path, mesh=mesh)


def test_duplicate_json_keys_are_refused_before_checksum_or_replay(tmp_path):
    mesh, _, _ = initial_case()
    path = tmp_path / "duplicates.json"
    path.write_text('{"format":"first","format":"second"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON keys"):
        load_native_underthrust_motion_checkpoint(path, mesh=mesh)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_json_constants_are_refused_before_checksum_or_replay(tmp_path, constant):
    mesh, _, _ = initial_case()
    path = tmp_path / "nonfinite.json"
    path.write_text('{"value":' + constant + '}', encoding="utf-8")
    with pytest.raises(ValueError, match="nonfinite JSON number"):
        load_native_underthrust_motion_checkpoint(path, mesh=mesh)


def test_restart_requires_original_source_integration_mesh(tmp_path, initial_checkpoint_bytes):
    from tectonics.mesh import build_icosphere

    path = tmp_path / "other-mesh.json"
    path.write_bytes(initial_checkpoint_bytes)
    with pytest.raises(ValueError):
        load_native_underthrust_motion_checkpoint(path, mesh=build_icosphere(1))


def test_native_joint_checkpoint_preserves_sparse_physical_owner_identity(tmp_path):
    mesh, origin, state = initial_case(owners=(7, 10**12))
    path = tmp_path / "sparse.json"
    save_native_underthrust_motion_checkpoint(path, mesh, origin, state)
    restored_origin, restored, _ = load_native_underthrust_motion_checkpoint(path, mesh=mesh)
    assert (restored_origin, restored) == (origin, state)
    assert restored.initial_mechanics.plate_ids == (7, 10**12)
    assert tuple(owner for owner, _ in restored.owner_histories) == (7, 10**12)
    assert tuple(owner for owner, _ in restored.current_inventory.owner_histories) == (7, 10**12)
