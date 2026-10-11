"""Evolved physical history survives explicitly verified material cuts."""
from dataclasses import asdict, replace

import numpy as np
import pytest

from tests.test_geometric_distributed_contact_dynamics import setup
from tests.test_geometric_distributed_contact_invariants import archive_world, prepare
from tectonics.basal_coupling import PrescribedBasalParameters
from tectonics.geometric_basal_loading import assemble_geometric_basal_load
from tectonics.geometric_boundary_io import load_boundary_checkpoint, save_boundary_checkpoint
from tectonics.geometric_contact_subdivision import (
    rebind_distributed_boundary_subdivision, rebind_distributed_contact_subdivision,
)
from tectonics.geometric_distributed_contact_dynamics import (
    advance_distributed_contact_dynamics, distributed_contact_dynamics_from_dict,
    distributed_contact_dynamics_to_dict, initialize_distributed_contact_dynamics,
)
from tectonics.geometric_material_subdivision import validate_material_subdivision
from tectonics.geometric_polarity import PolarityEvidence
from tectonics.geometric_surface import split_geometry
from tectonics.mesh import build_icosphere
from tectonics.spherical_polygons import clip_hemisphere


def cut_surface(surface, contact, *, receiver=None):
    fragment = next(row for row in surface.fragments
                    if row.fragment_id == (contact.fragment_a if receiver is None else receiver))
    midpoint = np.asarray(contact.start)+np.asarray(contact.end)
    midpoint /= np.linalg.norm(midpoint)
    normal = np.cross(midpoint, np.asarray(fragment.polygon).sum(axis=0))
    children = split_geometry(fragment,
        (clip_hemisphere(fragment.polygon, normal), clip_hemisphere(fragment.polygon, -normal)),
        label="representation-cut")
    return replace(surface, fragments=tuple(row for row in surface.fragments
        if row.fragment_id != fragment.fragment_id)+children)


def cut_load(mesh, before, after, old):
    mapping = validate_material_subdivision(before, after)
    inherited = {child: dict(old.thermal_lid_by_fragment)[parent] for child, parent in mapping}
    return assemble_geometric_basal_load(mesh, after,
        PrescribedBasalParameters(**dict(old.source_parameters)), inherited, old.provenance)


def test_evolved_sliding_history_is_not_reprepared_when_material_contact_is_split():
    surface, contacts, _, load, initial = setup(panel_count=1, gauss_order=4, friction=0., cohesion=0.)
    first = advance_distributed_contact_dynamics(surface, initial, load, 1e10).state
    assert any(local.cumulative_slip_m > 0. for _, local in first.states)
    after = cut_surface(surface, contacts[0])
    new_load = cut_load(build_icosphere(0), surface, after, load)
    rebound = rebind_distributed_contact_subdivision(surface, after, first, load, new_load)
    for name in ("states", "rotation_radians", "polarity_history", "initial_loading_dt_s",
                 "preparation_work_j", "elapsed_time_s", "step_count",
                 "cumulative_source_work_j", "cumulative_basal_dissipation_j"):
        assert getattr(rebound, name) == getattr(first, name)
    direct = advance_distributed_contact_dynamics(surface, first, load, 1e10)
    cut = advance_distributed_contact_dynamics(after, rebound, new_load, 1e10)
    np.testing.assert_allclose(cut.angular_velocity_rad_s, direct.angular_velocity_rad_s,
                               rtol=3e-11, atol=1e-28)
    for new, old in zip(cut.points, direct.points):
        assert new.point_id == old.point_id
        assert new.step.state.plastic_slip_m == pytest.approx(old.step.state.plastic_slip_m, rel=3e-11, abs=1e-13)
    # Loading another surface is still forbidden without the checked operation.
    with pytest.raises(ValueError):
        distributed_contact_dynamics_from_dict(
            distributed_contact_dynamics_to_dict(surface, first, load), after, new_load)


def test_cut_preserves_original_local_polarity_evidence_on_physical_support():
    surface, contacts, traces, load, _ = setup(panel_count=1, gauss_order=4)
    origin = PolarityEvidence("retained-event", contacts[0].contact_id, 1, 0,
                              "existing accepted material, retained physical interval")
    initial = initialize_distributed_contact_dynamics(surface, traces, load,
        initial_loading_dt_s=1e10, max_rotation_radians=.01, history=(origin,))
    first = advance_distributed_contact_dynamics(surface, initial, load, 1e10).state
    after = cut_surface(surface, contacts[0])
    new_load = cut_load(build_icosphere(0), surface, after, load)
    rebound = rebind_distributed_contact_subdivision(surface, after, first, load, new_load)
    assert rebound.polarity_history == first.polarity_history
    assert rebound.polarity_history[0].origin == origin
    raw = distributed_contact_dynamics_to_dict(after, rebound, new_load)
    assert distributed_contact_dynamics_from_dict(raw, after, new_load) == rebound


def test_joint_nonempty_archive_cut_keeps_payloads_and_exact_restart(tmp_path):
    mesh, surface, lid, declared, archive, history = archive_world()
    load, _, initial = prepare(mesh, surface, lid, declared=declared, history=history)
    first = advance_distributed_contact_dynamics(surface, initial, load, 1e8).state
    cohort = next(row for row in archive.cohorts if row.supports)
    support = cohort.supports[0]
    contact = next(row for row in archive.contacts if row.contact_id == support.contact_id)
    after = cut_surface(surface, contact, receiver=support.receiver_fragment_id)
    new_load = cut_load(mesh, surface, after, load)
    original = asdict(first), asdict(archive), asdict(surface)
    rebound, inventory = rebind_distributed_boundary_subdivision(
        surface, after, first, load, new_load, archive)
    assert (asdict(first), asdict(archive), asdict(surface)) == original
    assert rebound.states == first.states and rebound.polarity_history == first.polarity_history
    assert inventory.transactions == archive.transactions
    assert len(inventory.representations) == 1
    assert tuple((row.event_id, row.payload_sha256, row.loss) for row in inventory.cohorts) == tuple(
        (row.event_id, row.payload_sha256, row.loss) for row in archive.cohorts)
    save_boundary_checkpoint(tmp_path/"cut.json", after, inventory,
        provenance={"distributed_contact": distributed_contact_dynamics_to_dict(after, rebound, new_load)})
    loaded, loaded_inventory, metadata = load_boundary_checkpoint(tmp_path/"cut.json")
    restored = distributed_contact_dynamics_from_dict(metadata["distributed_contact"], loaded, new_load)
    direct = advance_distributed_contact_dynamics(after, rebound, new_load, 1e8).state
    resumed = advance_distributed_contact_dynamics(loaded, restored, new_load, 1e8).state
    assert direct == resumed and loaded_inventory == inventory
    save_boundary_checkpoint(tmp_path/"direct.json", after, inventory,
        provenance={"distributed_contact": distributed_contact_dynamics_to_dict(after, direct, new_load)})
    save_boundary_checkpoint(tmp_path/"resumed.json", loaded, loaded_inventory,
        provenance={"distributed_contact": distributed_contact_dynamics_to_dict(loaded, resumed, new_load)})
    assert (tmp_path/"direct.json").read_bytes() == (tmp_path/"resumed.json").read_bytes()


def test_joint_rebind_rejects_material_or_source_changes_atomically():
    surface, contacts, _, load, initial = setup(panel_count=1, gauss_order=4)
    after = cut_surface(surface, contacts[0])
    good = cut_load(build_icosphere(0), surface, after, load)
    original = asdict(initial)
    lid = dict(good.thermal_lid_by_fragment)
    lid[after.fragments[-1].fragment_id] += 1.
    changed_source = assemble_geometric_basal_load(build_icosphere(0), after,
        PrescribedBasalParameters(**dict(load.source_parameters)), lid, load.provenance)
    with pytest.raises(ValueError, match="thermal lid"):
        rebind_distributed_contact_subdivision(surface, after, initial, load, changed_source)
    child = after.fragments[-1]
    changed = replace(after, fragments=after.fragments[:-1]+(replace(child,
        parcel=replace(child.parcel, age_myr=child.parcel.age_myr+1.)),))
    with pytest.raises(ValueError, match="material history"):
        rebind_distributed_contact_subdivision(surface, changed, initial, load, good)
    assert asdict(initial) == original
