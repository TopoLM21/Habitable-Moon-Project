"""Joint current-state checkpoints bind actual underthrust to force replay."""
from dataclasses import asdict, replace
from functools import lru_cache
import copy
import hashlib
import json

import pytest

from tests.underthrust_case import DT_S, PROVENANCE, SECOND_SOURCE, SOURCE, advance_case, initialize_case
from tectonics.geometric_boundary import boundary_to_dict, surface_digest
from tectonics.geometric_boundary_io import (
    _encoded, _validate_pair, load_boundary_checkpoint, save_boundary_checkpoint,
)
from tectonics.geometric_spatial_mantle_loading import assemble_physical_spatial_mantle_load
from tectonics.geometric_transport import UnresolvedPolarityError
from tectonics.geometric_underthrust_motion import advance_underthrust_motion
from tectonics.geometric_underthrust_motion_io import (
    load_underthrust_motion_checkpoint, save_underthrust_motion_checkpoint,
)


@lru_cache(maxsize=None)
def case(*, owners=(0, 1)):
    mesh, origin, initial, first = advance_case(owners=owners)
    return mesh, origin, initial, first, None


@pytest.fixture(scope="module")
def first_checkpoint_bytes(tmp_path_factory):
    mesh, origin, _, first, _ = case()
    path = tmp_path_factory.mktemp("actual-underthrust-checkpoint")/"first.json"
    save_underthrust_motion_checkpoint(path, mesh, origin, first.state)
    return path.read_bytes()


def _write_rehashed(path, body):
    body = copy.deepcopy(body)
    body.pop("payload_sha256", None)
    body["payload_sha256"] = hashlib.sha256(_encoded(body)).hexdigest()
    path.write_bytes(_encoded(body)+b"\n")


def test_initial_restart_accepts_true_underthrust_and_identical_final_checkpoint_bytes(tmp_path):
    mesh, origin, initial, first, _ = case()
    path = tmp_path/"initial.json"
    provenance = {"purpose": "Actual differential underthrust restart", "intervals": [1]}
    save_underthrust_motion_checkpoint(path, mesh, origin, initial, provenance=provenance)
    restored_origin, restored, user = load_underthrust_motion_checkpoint(path, mesh=mesh)
    assert (restored_origin, restored, user) == (origin, initial, provenance)
    load = assemble_physical_spatial_mantle_load(mesh, restored.current_surface, SOURCE)
    assert load == initial.initial_load
    resumed = advance_underthrust_motion(mesh, restored_origin, restored, load, DT_S,
                                        provenance=PROVENANCE)
    assert resumed == first
    assert len(resumed.geometric_transaction.losses) >= 2
    assert resumed.geometric_transaction.births
    assert resumed.rotation_increment_radians[0] != resumed.rotation_increment_radians[1]
    direct_path, resumed_path = tmp_path/"direct.json", tmp_path/"resumed.json"
    save_underthrust_motion_checkpoint(direct_path, mesh, origin, first.state, provenance=provenance)
    save_underthrust_motion_checkpoint(resumed_path, mesh, restored_origin, resumed.state, provenance=user)
    assert direct_path.read_bytes() == resumed_path.read_bytes()
    final_origin, final_state, final_user = load_underthrust_motion_checkpoint(resumed_path, mesh=mesh)
    assert (final_origin, final_state, final_user) == (origin, first.state, provenance)
    current, inventory, _ = load_boundary_checkpoint(resumed_path)
    assert (current, inventory) == (first.state.current_surface, first.state.current_inventory)
    assert inventory.time_myr == current.time_myr > origin.time_myr
    _validate_pair(current, inventory)


@pytest.mark.parametrize("source", [SOURCE, SECOND_SOURCE])
def test_actual_underthrust_restart_reproduces_unsupported_new_pair_rejection_atomically(
        tmp_path, first_checkpoint_bytes, source):
    mesh, origin, _, first, _ = case()
    path = tmp_path/"accepted-first.json"
    path.write_bytes(first_checkpoint_bytes)
    restored_origin, restored, _ = load_underthrust_motion_checkpoint(path, mesh=mesh)
    assert (restored_origin, restored) == (origin, first.state)
    assert restored.current_inventory.cohorts == first.state.current_inventory.cohorts
    assert restored.buried_registry == first.state.buried_registry
    direct_load = assemble_physical_spatial_mantle_load(mesh, first.state.current_surface, source)
    replay_load = assemble_physical_spatial_mantle_load(mesh, restored.current_surface, source)
    assert replay_load == direct_load
    before_direct, before_restored = (_encoded(asdict(state)) for state in (first.state, restored))
    with pytest.raises(UnresolvedPolarityError) as direct_error:
        advance_underthrust_motion(mesh, origin, first.state, direct_load, DT_S, provenance=PROVENANCE)
    with pytest.raises(UnresolvedPolarityError) as restored_error:
        advance_underthrust_motion(mesh, restored_origin, restored, replay_load, DT_S, provenance=PROVENANCE)
    assert type(restored_error.value) is type(direct_error.value)
    assert restored_error.value.args == direct_error.value.args
    assert restored_error.value.overlaps == direct_error.value.overlaps
    assert any(overlap.reason == "new_material_requires_new_physical_pair"
               for overlap in direct_error.value.overlaps)
    assert all(overlap.area_km2 > 0. for overlap in direct_error.value.overlaps)
    assert _encoded(asdict(first.state)) == before_direct
    assert _encoded(asdict(restored)) == before_restored
    assert path.read_bytes() == first_checkpoint_bytes


def test_joint_checkpoint_keeps_new_buried_cohorts_births_and_material_support(tmp_path):
    mesh, origin, initial, first, _ = case()
    assert not initial.current_inventory.cohorts
    assert first.geometric_transaction.losses and first.geometric_transaction.births
    path = tmp_path/"new-burial.json"
    save_underthrust_motion_checkpoint(path, mesh, origin, first.state)
    restored_origin, restored, user = load_underthrust_motion_checkpoint(path, mesh=mesh)
    assert (restored_origin, restored, user) == (origin, first.state, {})
    assert len(restored.current_inventory.cohorts) > 0
    assert restored.current_inventory.cohorts == first.state.current_inventory.cohorts
    assert restored.buried_registry == first.state.buried_registry
    joint_surface, joint_archive, _ = load_boundary_checkpoint(path)
    assert (joint_surface, joint_archive) == (restored.current_surface, restored.current_inventory)
    assert joint_archive.time_myr == joint_surface.time_myr > origin.time_myr


def test_sparse_actual_owner_checkpoint_preserves_compact_solver_and_source_ids(tmp_path):
    mesh, origin, _, first, _ = case(owners=(7, 10**12))
    path = tmp_path/"sparse.json"
    save_underthrust_motion_checkpoint(path, mesh, origin, first.state)
    restored_origin, restored, _ = load_underthrust_motion_checkpoint(path, mesh=mesh)
    assert (restored_origin, restored) == (origin, first.state)
    assert restored.initial_mechanics.plate_ids == (7, 10**12)
    assert restored.initial_load.plate_ids == (7, 10**12)


def test_exclusive_underthrust_checkpoint_does_not_replace_existing_bytes(tmp_path):
    mesh, origin, _, first, _ = case()
    path = tmp_path/"exclusive.json"
    save_underthrust_motion_checkpoint(path, mesh, origin, first.state)
    original_bytes = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_underthrust_motion_checkpoint(path, mesh, origin, first.state,
                                          provenance={"replace": True})
    assert path.read_bytes() == original_bytes


def test_rehashed_old_valid_joint_pair_cannot_replace_current_underthrust_epoch(tmp_path, first_checkpoint_bytes):
    mesh, origin, initial, first, _ = case()
    path = tmp_path/"old-epoch.json"
    body = json.loads(first_checkpoint_bytes)
    body["surface"] = asdict(origin)
    body["boundary"] = boundary_to_dict(initial.current_inventory)
    _write_rehashed(path, body)
    # The old pair is valid independently, but the force trajectory says it moved.
    old_surface, old_inventory, _ = load_boundary_checkpoint(path)
    assert (old_surface, old_inventory) == (origin, initial.current_inventory)
    with pytest.raises(ValueError, match="current surface/archive"):
        load_underthrust_motion_checkpoint(path, mesh=mesh)


@pytest.mark.parametrize("mutate", [
    lambda metadata: metadata.update(format="geometric-force-generated-bending-checkpoint-1"),
    lambda metadata: metadata.update(extra="unsupported"),
    lambda metadata: metadata.pop("origin_surface"),
    lambda metadata: metadata.update(user_provenance=[]),
    lambda metadata: metadata.update(user_provenance=False),
])
def test_rehashed_metadata_requires_exact_underthrust_schema(tmp_path, mutate, first_checkpoint_bytes):
    mesh, origin, _, first, _ = case()
    path = tmp_path/"bad-metadata.json"
    body = json.loads(first_checkpoint_bytes)
    mutate(body["provenance"])
    _write_rehashed(path, body)
    with pytest.raises(ValueError, match="underthrust checkpoint metadata"):
        load_underthrust_motion_checkpoint(path, mesh=mesh)


def test_rehashed_origin_cannot_rebind_source_and_bank_history_to_a_new_epoch(tmp_path, first_checkpoint_bytes):
    mesh, origin, _, first, _ = case()
    path = tmp_path/"wrong-origin.json"
    body = json.loads(first_checkpoint_bytes)
    body["provenance"]["origin_surface"]["time_myr"] += 1e-6
    _write_rehashed(path, body)
    with pytest.raises(ValueError):
        load_underthrust_motion_checkpoint(path, mesh=mesh)


def test_rehashed_snapshot_cannot_replace_numeric_zero_with_boolean_alias(tmp_path):
    mesh, origin, initial, _ = initialize_case()
    path = tmp_path/"boolean-snapshot.json"
    save_underthrust_motion_checkpoint(path, mesh, origin, initial)
    body = json.loads(path.read_text())
    snapshot = body["provenance"]["underthrust_motion"]["snapshot"]
    assert snapshot["elapsed_time_s"] == 0.
    snapshot["elapsed_time_s"] = False
    _write_rehashed(path, body)
    with pytest.raises(ValueError, match="snapshot"):
        load_underthrust_motion_checkpoint(path, mesh=mesh)


@pytest.mark.parametrize("changed_state", ["boundary", "surface"])
def test_rehashed_initial_current_pair_cannot_replace_numeric_clock_with_boolean(tmp_path, changed_state):
    mesh, origin, initial, _ = initialize_case()
    path = tmp_path/"boolean-current-pair.json"
    save_underthrust_motion_checkpoint(path, mesh, origin, initial)
    body = json.loads(path.read_text())
    assert body[changed_state]["time_myr"] == 0.
    body[changed_state]["time_myr"] = False
    if changed_state == "surface":
        body["boundary"]["time_myr"] = False
        body["boundary"]["surface_digest"] = surface_digest(replace(origin, time_myr=False))
    _write_rehashed(path, body)
    # The plain joint reader accepts this otherwise consistent pair.  Its
    # redundant current state must still match the strict replay types.
    current_surface, current_inventory, _ = load_boundary_checkpoint(path)
    changed = current_inventory if changed_state == "boundary" else current_surface
    assert changed.time_myr is False
    _validate_pair(current_surface, current_inventory)
    if changed_state == "boundary":
        assert current_inventory == initial.current_inventory
        assert _encoded(asdict(current_inventory)) != _encoded(asdict(initial.current_inventory))
    with pytest.raises(ValueError, match="current surface/archive"):
        load_underthrust_motion_checkpoint(path, mesh=mesh)


def test_rehashed_snapshot_cannot_relax_retained_elastic_free_anchors(tmp_path, first_checkpoint_bytes):
    mesh, origin, _, first, _ = case()
    path = tmp_path/"relaxed-anchor.json"
    body = json.loads(first_checkpoint_bytes)
    snapshot = body["provenance"]["underthrust_motion"]["snapshot"]
    snapshot["free_anchor_fields"][0][1][0] += .1
    _write_rehashed(path, body)
    with pytest.raises(ValueError, match="snapshot"):
        load_underthrust_motion_checkpoint(path, mesh=mesh)


def test_plain_joint_archive_cannot_be_silently_migrated_to_underthrust_history(tmp_path):
    mesh, origin, initial, _, _ = case()
    path = tmp_path/"plain-joint.json"
    save_boundary_checkpoint(path, origin, initial.current_inventory,
                             provenance={"purpose": "Independent old joint state"})
    with pytest.raises(ValueError, match="underthrust checkpoint metadata"):
        load_underthrust_motion_checkpoint(path, mesh=mesh)


@pytest.mark.parametrize("provenance", [False, 0., "text", []])
def test_writer_rejects_nonobject_provenance_before_creating_file(tmp_path, provenance):
    mesh, origin, _, first, _ = case()
    path = tmp_path/"invalid-provenance.json"
    with pytest.raises(ValueError, match="JSON object"):
        save_underthrust_motion_checkpoint(path, mesh, origin, first.state, provenance=provenance)
    assert not path.exists()


def test_writer_rejects_nonfinite_json_before_creating_file(tmp_path):
    mesh, origin, _, first, _ = case()
    path = tmp_path/"nonfinite-provenance.json"
    with pytest.raises(ValueError):
        save_underthrust_motion_checkpoint(path, mesh, origin, first.state,
                                          provenance={"unserializable": float("nan")})
    assert not path.exists()


def test_restart_requires_the_trusted_source_integration_mesh(tmp_path):
    from tectonics.mesh import build_icosphere
    mesh, origin, _, first, _ = case()
    path = tmp_path/"other-mesh.json"
    save_underthrust_motion_checkpoint(path, mesh, origin, first.state)
    with pytest.raises(ValueError):
        load_underthrust_motion_checkpoint(path, mesh=build_icosphere(1))
