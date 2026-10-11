"""Joint finite-transport checkpoints preserve the genuine reference pair."""
from dataclasses import replace
from functools import lru_cache
import copy
import hashlib
import json

import numpy as np
import pytest

from tests.test_geometric_bending_contact_dynamics import prepared, DT, BOUNDS
from tests.test_geometric_bending_fault_transport import fixture, PROVENANCE
from tests.test_geometric_distributed_contact_invariants import PARAMETERS, world
from analysis.geometric_distributed_contact_validation.trace_case import _archive_fixture
from tectonics.geometric_basal_loading import assemble_geometric_basal_load
from tectonics.geometric_bending_contact_dynamics import initialize_bending_contact_dynamics
from tectonics.geometric_bending_fault_transport import (
    advance_bending_fault_transport, initialize_bending_fault_transport,
)
from tectonics.geometric_bending_fault_transport_io import (
    save_bending_fault_transport_checkpoint, load_bending_fault_transport_checkpoint,
)
from tectonics.geometric_boundary import (
    initialize_boundary, polarity_evidence, boundary_to_dict, surface_digest,
)
from tectonics.geometric_boundary_io import _validate_pair, load_boundary_checkpoint, _encoded
from tectonics.geometric_contacts import extract_contacts
from tectonics.geometric_fault_traces import declare_fault_trace_set, TraceQuadrature
from tectonics.geometric_fault_transport import material_atlas_surface
from tectonics.geometric_polarity import PolarityEvidence
from tectonics.mesh import build_icosphere


@lru_cache(maxsize=1)
def nonempty_fixture():
    mesh, surface, archive = _archive_fixture()
    assert archive.cohorts and archive.transactions
    _, _, _, template = prepared(mixed=True)
    lid = {fragment.fragment_id: .2+1.3*index for index, fragment in enumerate(surface.fragments)}
    load = assemble_geometric_basal_load(mesh, surface, PARAMETERS, lid,
        "Explicit nonempty-archive IO reference source")
    traces = []
    for index, contact in enumerate(extract_contacts(surface.fragments, surface.radius_km)):
        horizontal = np.asarray(contact.normal_a_to_b)*(1. if contact.plate_b == 1 else -1.)
        start, end = contact.start, contact.end
        if np.cross(start, end)@horizontal < 0.:
            start, end = end, start
        traces.append(replace(template.fault_traces.traces[0], trace_id=f"io-archive:{index}",
            start=start, end=end, subducting_plate=0, overriding_plate=1,
            law=replace(template.fault_traces.traces[0].law, cohesion_pa=1e9,
                        friction_dry=0., friction_wet=0.),
            provenance="Explicit physical plane in existing nonempty IO archive"))
    trace_set = declare_fault_trace_set(surface, traces, TraceQuadrature(1, 2))
    banks = tuple(replace(template.bending_parameters[0], trace_id=trace.trace_id)
                  for trace in trace_set.traces)
    history = tuple(PolarityEvidence(**row) for row in polarity_evidence(archive))
    mechanics = initialize_bending_contact_dynamics(surface, trace_set, load, banks,
                                                   history=history, **BOUNDS)
    state = initialize_bending_fault_transport(surface, archive, mechanics, load)
    first = advance_bending_fault_transport(surface, state, load, 1e8, provenance=PROVENANCE)
    return mesh, surface, load, lid, first.state


def _write_rehashed(path, payload):
    body = copy.deepcopy(payload)
    body.pop("payload_sha256", None)
    body["payload_sha256"] = hashlib.sha256(_encoded(body)).hexdigest()
    path.write_bytes(_encoded(body)+b"\n")


def test_full_checkpoint_restart_restores_source_origin_archive_and_identical_next_step(tmp_path):
    surface, load, _, first = fixture()
    path = tmp_path/"middle.json"
    metadata = {"purpose": "Finite candidate exact restart", "nested": {"runs": [1, 2]}}
    save_bending_fault_transport_checkpoint(path, surface, first.state, load, provenance=metadata)
    loaded_surface, restored, restored_load, provenance = load_bending_fault_transport_checkpoint(path)
    assert loaded_surface == surface and restored == first.state and restored_load == load
    assert provenance == metadata
    direct = advance_bending_fault_transport(surface, first.state, load, DT, provenance=PROVENANCE)
    resumed = advance_bending_fault_transport(loaded_surface, restored, restored_load, DT,
                                             provenance=PROVENANCE)
    assert direct == resumed
    direct_path, resumed_path = tmp_path/"direct.json", tmp_path/"resumed.json"
    save_bending_fault_transport_checkpoint(direct_path, surface, direct.state, load, provenance=metadata)
    save_bending_fault_transport_checkpoint(resumed_path, loaded_surface, resumed.state, restored_load,
                                           provenance=provenance)
    assert direct_path.read_bytes() == resumed_path.read_bytes()
    joint_surface, joint_archive, _ = load_boundary_checkpoint(path)
    _validate_pair(joint_surface, joint_archive)
    assert joint_archive == first.state.reference_inventory
    assert joint_surface.time_myr == surface.time_myr < restored.material_atlas.time_myr
    candidate = material_atlas_surface(joint_surface, restored.material_atlas)
    with pytest.raises(ValueError):
        _validate_pair(candidate, joint_archive)


def test_nonempty_archive_checkpoint_keeps_real_cohorts_transactions_and_original_pair(tmp_path):
    mesh, surface, load, lid, state = nonempty_fixture()
    path = tmp_path/"nonempty.json"
    save_bending_fault_transport_checkpoint(path, surface, state, load)
    loaded, restored, rebuilt, metadata = load_bending_fault_transport_checkpoint(path, mesh=mesh,
        parameters=PARAMETERS, thermal_lid_by_fragment=lid, source_provenance=load.provenance)
    assert (loaded, restored, rebuilt, metadata) == (surface, state, load, {})
    assert restored.reference_inventory.cohorts == state.reference_inventory.cohorts
    assert restored.reference_inventory.transactions == state.reference_inventory.transactions
    assert restored.current_mechanics.polarity_history
    _validate_pair(loaded, restored.reference_inventory)
    assert restored.reference_inventory.surface_digest == surface_digest(surface)
    assert restored.reference_inventory.surface_digest != restored.material_atlas.current_surface_sha256
    assert restored.reference_inventory.time_myr == surface.time_myr
    assert restored.material_atlas.time_myr > surface.time_myr


def test_checkpoint_destination_is_exclusive_and_existing_bytes_are_preserved(tmp_path):
    surface, load, _, first = fixture()
    path = tmp_path/"exclusive.json"
    save_bending_fault_transport_checkpoint(path, surface, first.state, load)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_bending_fault_transport_checkpoint(path, surface, first.state, load,
                                               provenance={"replacement": True})
    assert path.read_bytes() == before


def test_candidate_cannot_replace_origin_surface_when_saving_original_archive(tmp_path):
    _, load, _, first = fixture()
    path = tmp_path/"wrong-reference.json"
    with pytest.raises(ValueError):
        save_bending_fault_transport_checkpoint(path, first.candidate_surface, first.state, load)
    assert not path.exists()


@pytest.mark.parametrize("provenance", [True, [], "text", {"nonfinite": float("nan")}])
def test_invalid_provenance_rejected_before_creating_checkpoint(tmp_path, provenance):
    surface, load, _, first = fixture()
    path = tmp_path/"invalid-provenance.json"
    with pytest.raises(ValueError):
        save_bending_fault_transport_checkpoint(path, surface, first.state, load, provenance=provenance)
    assert not path.exists()


def test_complete_trusted_source_inputs_rebuild_the_original_operators(tmp_path):
    surface, load, _, first = fixture()
    mesh, _, lid = world()
    path = tmp_path/"trusted.json"
    save_bending_fault_transport_checkpoint(path, surface, first.state, load)
    loaded, restored, checked, _ = load_bending_fault_transport_checkpoint(path, mesh=mesh,
        parameters=PARAMETERS, thermal_lid_by_fragment=lid, source_provenance=load.provenance)
    assert loaded == surface and restored == first.state and checked == load


@pytest.mark.parametrize("wrong", ["mesh", "parameters", "lid", "provenance"])
def test_mismatching_trusted_source_inputs_are_rejected(tmp_path, wrong):
    surface, load, _, first = fixture()
    mesh, _, lid = world()
    path = tmp_path/"untrusted.json"
    save_bending_fault_transport_checkpoint(path, surface, first.state, load)
    trusted = dict(mesh=mesh, parameters=PARAMETERS,
                   thermal_lid_by_fragment=lid, source_provenance=load.provenance)
    if wrong == "mesh":
        trusted["mesh"] = build_icosphere(1)
    elif wrong == "parameters":
        trusted["parameters"] = replace(PARAMETERS, convective_traction_pa=2.*PARAMETERS.convective_traction_pa)
    elif wrong == "lid":
        trusted["thermal_lid_by_fragment"] = {identity: 2.*value for identity, value in lid.items()}
    else:
        trusted["source_provenance"] = "Different trusted source provenance"
    with pytest.raises(ValueError, match="changed"):
        load_bending_fault_transport_checkpoint(path, **trusted)


@pytest.mark.parametrize("mutate", [
    lambda payload: payload["provenance"].update(format="wrong-format"),
    lambda payload: payload["provenance"].update(unknown="unrecognized"),
    lambda payload: payload["provenance"].update(user_provenance=[]),
    lambda payload: payload["provenance"]["fault_transport"].update(origin_surface_sha256="0"*64),
])
def test_rehashed_metadata_schema_and_reference_binding_edits_rejected(tmp_path, mutate):
    surface, load, _, first = fixture()
    path = tmp_path/"rehashed-metadata.json"
    save_bending_fault_transport_checkpoint(path, surface, first.state, load)
    payload = json.loads(path.read_bytes())
    mutate(payload)
    _write_rehashed(path, payload)
    with pytest.raises(ValueError):
        load_bending_fault_transport_checkpoint(path)


@pytest.mark.parametrize("outer", [False, True])
def test_rehashed_valid_but_different_archives_rejected_by_joint_reference_binding(tmp_path, outer):
    _, surface, load, _, state = nonempty_fixture()
    path = tmp_path/"archive-inconsistency.json"
    save_bending_fault_transport_checkpoint(path, surface, state, load)
    payload = json.loads(path.read_bytes())
    alternate = boundary_to_dict(initialize_boundary(surface))
    if outer:
        payload["boundary"] = alternate
    else:
        payload["provenance"]["fault_transport"]["reference_inventory"] = alternate
    _write_rehashed(path, payload)
    with pytest.raises(ValueError, match="reference archive differs"):
        load_bending_fault_transport_checkpoint(path)


def test_edited_joint_origin_cannot_preserve_old_reference_archive_by_rehashing(tmp_path):
    surface, load, _, first = fixture()
    path = tmp_path/"wrong-origin.json"
    save_bending_fault_transport_checkpoint(path, surface, first.state, load)
    payload = json.loads(path.read_bytes())
    payload["surface"]["fragments"][0]["parcel"]["age_myr"] += 1.
    _write_rehashed(path, payload)
    with pytest.raises(ValueError, match="do not describe the same state"):
        load_bending_fault_transport_checkpoint(path)


def test_outer_checkpoint_checksum_detects_unrehashed_receipt_edit(tmp_path):
    surface, load, _, first = fixture()
    path = tmp_path/"bad-checksum.json"
    save_bending_fault_transport_checkpoint(path, surface, first.state, load)
    payload = json.loads(path.read_bytes())
    payload["provenance"]["fault_transport"]["receipts"][0]["source_work_j"] += 1.
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="integrity mismatch"):
        load_bending_fault_transport_checkpoint(path)


def test_complete_trusted_source_reassembly_rejects_self_consistent_fabricated_operators(tmp_path):
    """A content hash cannot establish that a torque was integrated from the source."""
    from tectonics.geometric_basal_loading import _body as load_body, _hash as load_hash
    surface, load, original, _ = fixture()
    forged = replace(load, torque_nm=tuple(2.*value for value in load.torque_nm), load_sha256="")
    forged = replace(forged, load_sha256=load_hash(load_body(forged)))
    reference = original.initial_mechanics
    mechanics = initialize_bending_contact_dynamics(surface, reference.fault_traces, forged,
        reference.bending_parameters, history=reference.polarity_history, **BOUNDS)
    state = initialize_bending_fault_transport(surface, original.reference_inventory, mechanics, forged)
    path = tmp_path/"self-consistent-fabricated-source.json"
    save_bending_fault_transport_checkpoint(path, surface, state, forged)
    assert load_bending_fault_transport_checkpoint(path)[2] == forged
    mesh, _, lid = world()
    with pytest.raises(ValueError, match="do not match trusted source reassembly"):
        load_bending_fault_transport_checkpoint(path, mesh=mesh, parameters=PARAMETERS,
            thermal_lid_by_fragment=lid, source_provenance=load.provenance)
