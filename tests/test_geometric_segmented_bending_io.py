"""Actual changing-source work survives joint original material checkpoints."""
from dataclasses import replace
from functools import lru_cache
import copy
import hashlib
import json

import numpy as np
import pytest

from tests.test_geometric_bending_contact_dynamics import BOUNDS, DT, prepared
from tests.test_geometric_bending_fault_transport_io import nonempty_fixture
from tests.test_geometric_distributed_contact_invariants import world
from tests.test_geometric_segmented_bending_dynamics import fixture, _load
from tectonics.basal_coupling import PrescribedBasalParameters
from tectonics.geometric_basal_loading import _body as default_load_body, _hash as default_load_hash
from tectonics.geometric_bending_contact_dynamics import initialize_bending_contact_dynamics
from tectonics.geometric_boundary import initialize_boundary
from tectonics.geometric_boundary_io import _encoded, _validate_pair, load_boundary_checkpoint
from tectonics.geometric_rigid_mantle_loading import (
    PrescribedRigidMantleSource, assemble_physical_rigid_mantle_load,
)
from tectonics.geometric_segmented_bending_dynamics import (
    advance_segmented_bending_dynamics, initialize_segmented_bending_dynamics,
)
from tectonics.geometric_segmented_bending_io import (
    FORMAT, load_segmented_bending_checkpoint, save_segmented_bending_checkpoint,
)


PROVENANCE = "Source-segment restart preserving evolved plastic memory and past work"


@lru_cache(maxsize=1)
def changing_fixture():
    surface, _, _, initial = fixture(evolved=True)
    first_source = _load(surface, amplitude=.6, seed=31, drag=1.2)
    state = advance_segmented_bending_dynamics(surface, initial, first_source, DT,
                                               provenance=PROVENANCE).state
    next_source = _load(surface, amplitude=.8, seed=29, drag=.9)
    return surface, initialize_boundary(surface), state, next_source


def _source_inputs(state, *, mesh=None):
    base_mesh, _, _ = world()
    mesh = base_mesh if mesh is None else mesh
    loads = [state.initial_load, *(receipt.current_load for receipt in state.segments)]
    result = {}
    for load in loads:
        if hasattr(load, "source"):
            result[load.load_sha256] = dict(mesh=mesh, source=load.source, provenance=load.provenance)
        else:
            result[load.load_sha256] = dict(mesh=mesh,
                parameters=PrescribedBasalParameters(**dict(load.source_parameters)),
                thermal_lid_by_fragment=dict(load.thermal_lid_by_fragment), provenance=load.provenance)
    return result


def _write_rehashed(path, body):
    body = copy.deepcopy(body)
    body.pop("payload_sha256", None)
    body["payload_sha256"] = hashlib.sha256(_encoded(body)).hexdigest()
    path.write_bytes(_encoded(body)+b"\n")


def test_evolved_plastic_work_restart_accepts_identical_next_source_and_checkpoint_bytes(tmp_path):
    surface, inventory, state, next_source = changing_fixture()
    assert any(local.plastic_slip_m != 0. for _, local in state.states)
    assert state.cumulative_source_work_j != 0.
    assert state.initial_mechanics.step_count > 0 and len(state.segments) == 1
    metadata = {"purpose": "Physical incremental-source restart", "nested": {"segments": [1, 2]}}
    path = tmp_path/"middle.json"
    save_segmented_bending_checkpoint(path, surface, inventory, state, provenance=metadata)
    loaded_surface, loaded_inventory, restored, user = load_segmented_bending_checkpoint(
        path, source_inputs=_source_inputs(state))
    assert (loaded_surface, loaded_inventory, restored, user) == (surface, inventory, state, metadata)
    direct = advance_segmented_bending_dynamics(surface, state, next_source, DT, provenance=PROVENANCE)
    resumed = advance_segmented_bending_dynamics(loaded_surface, restored, next_source, DT, provenance=PROVENANCE)
    assert direct == resumed
    left, right = tmp_path/"direct.json", tmp_path/"resumed.json"
    save_segmented_bending_checkpoint(left, surface, inventory, direct.state, provenance=metadata)
    save_segmented_bending_checkpoint(right, loaded_surface, loaded_inventory, resumed.state, provenance=user)
    assert left.read_bytes() == right.read_bytes()
    _validate_pair(loaded_surface, loaded_inventory)
    assert restored.preparation_work_j == state.initial_mechanics.preparation_work_j


def test_nonempty_passive_archive_keeps_original_joint_clock_and_cohort_payloads(tmp_path):
    mesh, surface, load, _, transport = nonempty_fixture()
    state = initialize_segmented_bending_dynamics(surface, transport.current_mechanics, load)
    state = advance_segmented_bending_dynamics(surface, state, load, 1e8, provenance=PROVENANCE).state
    path = tmp_path/"nonempty.json"
    save_segmented_bending_checkpoint(path, surface, transport.reference_inventory, state)
    loaded, inventory, restored, user = load_segmented_bending_checkpoint(path,
        source_inputs=_source_inputs(state, mesh=mesh))
    assert (loaded, restored, user) == (surface, state, {})
    assert inventory == transport.reference_inventory
    assert len(inventory.cohorts) == 2
    assert inventory.transactions == transport.reference_inventory.transactions
    assert inventory.time_myr == surface.time_myr
    assert restored.elapsed_time_s > restored.initial_mechanics.elapsed_time_s
    _validate_pair(loaded, inventory)


def test_physical_rigid_source_kind_survives_joint_restart_without_quadratic_metadata(tmp_path):
    mesh, _, _ = world()
    surface, traces, _, template = prepared(gap=0.)
    source = PrescribedRigidMantleSource((2e-12, -1e-12, 3e-12), 1e14,
        "Independent physical rigid mantle for source-kind restart")
    load = assemble_physical_rigid_mantle_load(mesh, surface, source)
    seed = initialize_bending_contact_dynamics(surface, traces, load, template.bending_parameters, **BOUNDS)
    state = initialize_segmented_bending_dynamics(surface, seed, load)
    state = advance_segmented_bending_dynamics(surface, state, load, DT, provenance=PROVENANCE).state
    path = tmp_path/"rigid.json"
    save_segmented_bending_checkpoint(path, surface, initialize_boundary(surface), state)
    loaded, inventory, restored, _ = load_segmented_bending_checkpoint(path,
        source_inputs={load.load_sha256: dict(mesh=mesh, source=source)})
    assert restored == state and restored.initial_load.source == source
    assert restored.segments[0].current_load.source == source
    _validate_pair(loaded, inventory)


def test_trusted_reassembly_rejects_self_consistent_fake_default_operator_receipts(tmp_path):
    surface, _, _, initial = fixture(evolved=True)
    true_source = _load(surface, amplitude=.7)
    fake = replace(true_source, torque_nm=tuple(1.01*value for value in true_source.torque_nm))
    fake = replace(fake, load_sha256=default_load_hash(default_load_body(fake)))
    # The historical loader's content checks alone do not authenticate source
    # operators. A replay can be physically consistent with a fabricated RHS.
    state = advance_segmented_bending_dynamics(surface, initial, fake, DT, provenance=PROVENANCE).state
    path = tmp_path/"fake-source.json"
    save_segmented_bending_checkpoint(path, surface, initialize_boundary(surface), state)
    assert load_segmented_bending_checkpoint(path)[2] == state
    with pytest.raises(ValueError, match="trusted source reassembly"):
        load_segmented_bending_checkpoint(path, source_inputs=_source_inputs(state))


def test_trusted_source_schedule_rejects_changed_prescribed_parameters(tmp_path):
    surface, inventory, state, _ = changing_fixture()
    path = tmp_path/"source-identity.json"
    save_segmented_bending_checkpoint(path, surface, inventory, state)
    inputs = _source_inputs(state)
    original = inputs[state.segments[0].current_load.load_sha256]["parameters"]
    inputs[state.segments[0].current_load.load_sha256]["parameters"] = replace(original, seed=original.seed+1)
    with pytest.raises(ValueError, match="prescribed source changed"):
        load_segmented_bending_checkpoint(path, source_inputs=inputs)


@pytest.mark.parametrize("kind", ("missing", "extra", "unknown_option", "not_mapping"))
def test_explicit_trusted_inputs_cover_every_distinct_source_exactly(tmp_path, kind):
    surface, inventory, state, _ = changing_fixture()
    path = tmp_path/"source-coverage.json"
    save_segmented_bending_checkpoint(path, surface, inventory, state)
    inputs = _source_inputs(state)
    if kind == "missing":
        inputs.pop(state.initial_load.load_sha256)
    elif kind == "extra":
        inputs["0"*64] = {}
    elif kind == "unknown_option":
        inputs[state.initial_load.load_sha256]["implicit_load_fitting"] = True
    else:
        inputs = []
    with pytest.raises(ValueError):
        load_segmented_bending_checkpoint(path, source_inputs=inputs)


def test_exclusive_destination_preserves_existing_checkpoint_bytes(tmp_path):
    surface, inventory, state, _ = changing_fixture()
    path = tmp_path/"exclusive.json"
    save_segmented_bending_checkpoint(path, surface, inventory, state)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_segmented_bending_checkpoint(path, surface, inventory, state, provenance={"replacement": True})
    assert path.read_bytes() == before


@pytest.mark.parametrize("kind", ("archive_hash", "wrong_format", "extra_metadata", "user_provenance"))
def test_rehashed_malformed_metadata_and_archive_bindings_are_rejected(tmp_path, kind):
    surface, inventory, state, _ = changing_fixture()
    path = tmp_path/"malformed.json"
    save_segmented_bending_checkpoint(path, surface, inventory, state)
    body = json.loads(path.read_text())
    if kind == "archive_hash":
        body["provenance"]["reference_inventory_sha256"] = "0"*64
    elif kind == "wrong_format":
        body["provenance"]["format"] = "unverified_endpoint_mechanics"
    elif kind == "extra_metadata":
        body["provenance"]["accepted_new_material"] = True
    else:
        body["provenance"]["user_provenance"] = []
    _write_rehashed(path, body)
    with pytest.raises(ValueError):
        load_segmented_bending_checkpoint(path)


@pytest.mark.parametrize("provenance", (True, [], "text", {"nonfinite": float("nan")}))
def test_invalid_user_provenance_creates_no_checkpoint(tmp_path, provenance):
    surface, inventory, state, _ = changing_fixture()
    path = tmp_path/"bad-provenance.json"
    with pytest.raises(ValueError):
        save_segmented_bending_checkpoint(path, surface, inventory, state, provenance=provenance)
    assert not path.exists()


def test_wrong_joint_archive_clock_is_rejected_before_creating_file(tmp_path):
    surface, inventory, state, _ = changing_fixture()
    path = tmp_path/"wrong-archive.json"
    with pytest.raises(ValueError):
        save_segmented_bending_checkpoint(path, surface, replace(inventory, time_myr=inventory.time_myr+1.), state)
    assert not path.exists()
