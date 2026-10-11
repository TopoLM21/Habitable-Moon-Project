"""Material cuts cannot change evolved cantilever or contact histories."""
from dataclasses import asdict, replace

import numpy as np
import pytest

from tests.test_geometric_contact_subdivision import cut_load, cut_surface
from tests.test_geometric_distributed_contact_dynamics import setup
from tests.test_geometric_distributed_contact_invariants import (
    PARAMETERS, PROVENANCE, archive_world,
)
from tectonics.geometric_basal_loading import assemble_geometric_basal_load
from tectonics.geometric_bending import ElasticBendingBankParameters, FaultBendingParameters
from tectonics.geometric_bending_contact_dynamics import (
    advance_bending_contact_dynamics, bending_contact_dynamics_from_dict,
    bending_contact_dynamics_to_dict, initialize_bending_contact_dynamics,
)
from tectonics.geometric_bending_contact_subdivision import (
    rebind_bending_boundary_subdivision, rebind_bending_contact_subdivision,
)
from tectonics.geometric_boundary_io import load_boundary_checkpoint, save_boundary_checkpoint
from tectonics.geometric_fault_traces import TraceQuadrature, declare_fault_trace_set
from tectonics.mesh import build_icosphere


def bending_parameters(traces):
    return tuple(FaultBendingParameters(trace.trace_id,
        ElasticBendingBankParameters(30000., 2000., 70e9, .25,
            "Declared sub-bank cantilever; physical clamp distance independent of material cuts"),
        ElasticBendingBankParameters(25000., 2500., 65e9, .23,
            "Declared over-bank cantilever; physical clamp distance independent of material cuts"))
        for trace in traces)


def prepare_bending(surface, traces, load, *, history=()):
    return initialize_bending_contact_dynamics(surface, traces, load,
        bending_parameters(traces.traces), initial_loading_dt_s=1e6,
        max_rotation_radians=.01, max_tip_slope=.05, max_bending_strain=.005,
        max_deflection_radius_ratio=.005, history=history)


def test_evolved_sliding_and_bending_histories_survive_a_material_contact_cut():
    surface, contacts, traces, load, _ = setup(panel_count=1, gauss_order=4,
                                             friction=0., cohesion=0.)
    initial = prepare_bending(surface, traces, load)
    first = advance_bending_contact_dynamics(surface, initial, load, 1e10).state
    assert any(local.cumulative_slip_m > 0. for _, local in first.states)
    assert any(abs(w) > 0. for _, sub, over in first.transverse_displacement_m for w in (sub, over))
    after = cut_surface(surface, contacts[0])
    new_load = cut_load(build_icosphere(0), surface, after, load)
    rebound = rebind_bending_contact_subdivision(surface, after, first, load, new_load)
    permitted = {"surface_sha256", "basal_source_sha256", "fault_traces"}
    for name, value in asdict(first).items():
        if name not in permitted:
            assert asdict(rebound)[name] == value
    direct = advance_bending_contact_dynamics(surface, first, load, 1e10)
    cut = advance_bending_contact_dynamics(after, rebound, new_load, 1e10)
    np.testing.assert_allclose(cut.state.rotation_radians, direct.state.rotation_radians,
                               rtol=3e-11, atol=1e-21)
    for new, old in zip(cut.state.transverse_displacement_m, direct.state.transverse_displacement_m):
        assert new[0] == old[0]
        assert new[1:] == pytest.approx(old[1:], rel=3e-11, abs=1e-14)
    with pytest.raises(ValueError):
        bending_contact_dynamics_from_dict(
            bending_contact_dynamics_to_dict(surface, first, load), after, new_load)


def test_nonempty_archive_cut_preserves_bending_and_exact_joint_restart(tmp_path):
    mesh, before, lid, declared, archive, history = archive_world()
    load = assemble_geometric_basal_load(mesh, before, PARAMETERS, lid, PROVENANCE)
    traces = declare_fault_trace_set(before, declared, TraceQuadrature(1, 4))
    initial = prepare_bending(before, traces, load, history=history)
    first = advance_bending_contact_dynamics(before, initial, load, 1e8).state
    support = next(row for row in archive.cohorts if row.supports).supports[0]
    contact = next(row for row in archive.contacts if row.contact_id == support.contact_id)
    after = cut_surface(before, contact, receiver=support.receiver_fragment_id)
    new_load = cut_load(mesh, before, after, load)
    old_objects = asdict(first), asdict(archive), asdict(before)
    rebound, inventory = rebind_bending_boundary_subdivision(
        before, after, first, load, new_load, archive)
    assert (asdict(first), asdict(archive), asdict(before)) == old_objects
    assert rebound.states == first.states
    assert rebound.transverse_displacement_m == first.transverse_displacement_m
    assert rebound.bending_parameters == first.bending_parameters
    assert rebound.bending_source_sha256 == first.bending_source_sha256
    assert rebound.polarity_history == first.polarity_history
    assert inventory.transactions == archive.transactions and len(inventory.representations) == 1
    payloads = lambda inv: tuple((c.event_id, c.payload_sha256, c.loss) for c in inv.cohorts)
    assert payloads(inventory) == payloads(archive)
    save_boundary_checkpoint(tmp_path / "cut.json", after, inventory,
        provenance={"bending_contact": bending_contact_dynamics_to_dict(after, rebound, new_load)})
    loaded, loaded_inventory, metadata = load_boundary_checkpoint(tmp_path / "cut.json")
    restored = bending_contact_dynamics_from_dict(metadata["bending_contact"], loaded, new_load)
    direct = advance_bending_contact_dynamics(after, rebound, new_load, 1e8).state
    resumed = advance_bending_contact_dynamics(loaded, restored, new_load, 1e8).state
    uncut = advance_bending_contact_dynamics(before, first, load, 1e8).state
    assert direct == resumed and loaded_inventory == inventory
    np.testing.assert_allclose(direct.rotation_radians, uncut.rotation_radians, rtol=3e-11, atol=1e-21)
    for a, b in zip(direct.transverse_displacement_m, uncut.transverse_displacement_m):
        assert a[1:] == pytest.approx(b[1:], rel=3e-11, abs=1e-14)
    save_boundary_checkpoint(tmp_path / "direct.json", after, inventory,
        provenance={"bending_contact": bending_contact_dynamics_to_dict(after, direct, new_load)})
    save_boundary_checkpoint(tmp_path / "resumed.json", loaded, loaded_inventory,
        provenance={"bending_contact": bending_contact_dynamics_to_dict(loaded, resumed, new_load)})
    assert (tmp_path / "direct.json").read_bytes() == (tmp_path / "resumed.json").read_bytes()


def test_cut_rejects_changed_material_or_basal_source_without_modifying_bending():
    surface, contacts, traces, load, _ = setup(panel_count=1, gauss_order=4)
    initial = prepare_bending(surface, traces, load)
    after = cut_surface(surface, contacts[0])
    good_load = cut_load(build_icosphere(0), surface, after, load)
    child = after.fragments[-1]
    changed = replace(after, fragments=after.fragments[:-1] + (replace(child,
        parcel=replace(child.parcel, age_myr=child.parcel.age_myr + 1.)),))
    original = asdict(initial)
    with pytest.raises(ValueError, match="material history"):
        rebind_bending_contact_subdivision(surface, changed, initial, load, good_load)
    with pytest.raises(ValueError):
        rebind_bending_contact_subdivision(surface, after, initial, load, load)
    assert asdict(initial) == original
