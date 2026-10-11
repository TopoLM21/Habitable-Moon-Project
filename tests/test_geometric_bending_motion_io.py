"""Physical checkpoints bind the actual moved geometry/archive to force replay."""
import copy
import hashlib
import json

import pytest

from tests.test_geometric_bending_contact_dynamics import DT
from tests.test_geometric_bending_motion import (
    PROVENANCE, SECOND_SOURCE, archive_motion_fixture, motion_fixture,
)
from tectonics.geometric_bending_motion import advance_bending_motion
from tectonics.geometric_bending_motion_io import load_bending_motion_checkpoint, save_bending_motion_checkpoint
from tectonics.geometric_boundary_io import _encoded, _validate_pair, load_boundary_checkpoint
from tectonics.geometric_rigid_mantle_loading import assemble_physical_rigid_mantle_load


def _write_rehashed(path, body):
    body = copy.deepcopy(body)
    body.pop("payload_sha256", None)
    body["payload_sha256"] = hashlib.sha256(_encoded(body)).hexdigest()
    path.write_bytes(_encoded(body)+b"\n")


def test_actual_current_joint_pair_restart_accepts_identical_next_motion_and_bytes(tmp_path):
    mesh, origin, _, first, second = motion_fixture()
    path = tmp_path/"middle.json"
    metadata = {"purpose": "Accepted physical material restart", "intervals": [1, 2]}
    save_bending_motion_checkpoint(path, mesh, origin, first.state, provenance=metadata)
    recovered_origin, restored, user = load_bending_motion_checkpoint(path, mesh=mesh)
    assert (recovered_origin, restored, user) == (origin, first.state, metadata)
    current, inventory, _ = load_boundary_checkpoint(path)
    assert current == first.state.current_surface and inventory == first.state.current_inventory
    assert current.time_myr == inventory.time_myr > origin.time_myr
    _validate_pair(current, inventory)
    load = assemble_physical_rigid_mantle_load(mesh, restored.current_surface, SECOND_SOURCE)
    resumed = advance_bending_motion(mesh, recovered_origin, restored, load, DT, provenance=PROVENANCE)
    assert resumed == second
    direct_path, resumed_path = tmp_path/"direct.json", tmp_path/"resumed.json"
    save_bending_motion_checkpoint(direct_path, mesh, origin, second.state, provenance=metadata)
    save_bending_motion_checkpoint(resumed_path, mesh, recovered_origin, resumed.state, provenance=user)
    assert direct_path.read_bytes() == resumed_path.read_bytes()


def test_nonempty_archive_current_joint_checkpoint_restores_payloads_and_real_time(tmp_path):
    mesh, origin, _, first = archive_motion_fixture()
    path = tmp_path/"nonempty.json"
    save_bending_motion_checkpoint(path, mesh, origin, first.state)
    recovered_origin, restored, metadata = load_bending_motion_checkpoint(path, mesh=mesh)
    assert (recovered_origin, restored, metadata) == (origin, first.state, {})
    assert len(restored.current_inventory.cohorts) == 2
    joint_surface, joint_archive, _ = load_boundary_checkpoint(path)
    assert joint_archive == restored.current_inventory
    assert joint_archive.time_myr == joint_surface.time_myr > origin.time_myr
    assert joint_archive.cohorts == first.state.current_inventory.cohorts


def test_exclusive_physical_checkpoint_does_not_replace_existing_bytes(tmp_path):
    mesh, origin, _, first, _ = motion_fixture()
    path = tmp_path/"exclusive.json"
    save_bending_motion_checkpoint(path, mesh, origin, first.state)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_bending_motion_checkpoint(path, mesh, origin, first.state, provenance={"replacement": True})
    assert path.read_bytes() == before


def test_rehashed_metadata_cannot_substitute_original_archive_for_current_trajectory(tmp_path):
    mesh, origin, initial, first, _ = motion_fixture()
    path = tmp_path/"wrong-epoch.json"
    save_bending_motion_checkpoint(path, mesh, origin, first.state)
    body = json.loads(path.read_text())
    # A genuine old joint pair remains valid as a standalone boundary checkpoint,
    # but it does not describe the physical trajectory's current epoch.
    from tectonics.geometric_boundary import boundary_to_dict
    from dataclasses import asdict
    body["surface"] = asdict(origin)
    body["boundary"] = boundary_to_dict(initial.current_inventory)
    _write_rehashed(path, body)
    with pytest.raises(ValueError, match="current surface/archive"):
        load_bending_motion_checkpoint(path, mesh=mesh)
