"""Independent end-to-end passive slab checks on actual spherical transactions.

These tests deliberately ignore fixed integration-cell attachment and inspect
material histories in the geometric loss ledger instead.
"""
from dataclasses import asdict, replace
import json
import math

import numpy as np
import pytest

from tectonics.fractional_surface import EXTENSIVE_FIELDS, FractionalSurfaceState, SurfaceParcel
from tectonics.geometric_boundary import (
    boundary_from_dict, boundary_to_dict, consume_geometric_transaction, initialize_boundary,
)
from tectonics.geometric_surface import from_fractional_surface, load_geometric_checkpoint, save_geometric_checkpoint
from tectonics.geometric_transport import advance_geometric_surface
from tectonics.mesh import build_icosphere


def source(*, one_integration_cell=False):
    mesh = build_icosphere(0)
    areas = mesh.physical_cell_areas_km2(100.)
    owners = (mesh.centroids[:, 0] > 0.).astype(int)
    parcels = tuple(SurfaceParcel(i, int(owner), f"history:{i}", float(area),
        float(area)*(2.+i/20.), float(area)*(8.+i/10.),
        float(area)*(1e12 if owner else 2e12), 5. if owner else 10.,
        (("damage", i/20.),)) for i, (area, owner) in enumerate(zip(areas, owners)))
    state = from_fractional_surface(mesh, FractionalSurfaceState(0., tuple(areas), parcels), 100.)
    if one_integration_cell:
        state = replace(state, fragments=tuple(replace(fragment,
            parcel=replace(fragment.parcel, cell=0)) for fragment in state.fragments))
    return mesh, state


def birth(cell, plate, area, time, serial):
    return SurfaceParcel(cell, plate, f"birth:{time.hex()}:{serial}", area, 2.*area, 0., 0., 0.,
                         (("damage", 0.),))


def snapshot(value):
    return json.dumps(asdict(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def group(parcels, field):
    grouped = {}
    for parcel in parcels:
        grouped.setdefault(parcel.material_id, []).append(getattr(parcel, field))
    return {identity: math.fsum(values) for identity, values in grouped.items()}


def assert_combined_budget(initial, state, inventory, births):
    for field in EXTENSIVE_FIELDS:
        before = group((f.parcel for f in initial.fragments), field)
        after = group((f.parcel for f in state.fragments), field)
        removed = group((c.loss.fragment.parcel for c in inventory.cohorts), field)
        created = group((f.parcel for f in births), field)
        for identity in before.keys() | after.keys() | removed.keys() | created.keys():
            assert after.get(identity, 0.)+removed.get(identity, 0.) == pytest.approx(
                before.get(identity, 0.)+created.get(identity, 0.), rel=5e-12, abs=0.)


MOTION = np.array([[0., 0., 0.], [0., 0., .01]])
COMMON = np.array([[.002, -.003, .001]]*2)


@pytest.mark.parametrize("one_cell", [False, True])
def test_geometric_losses_are_accepted_once_without_second_surface_removal(one_cell):
    mesh, initial = source(one_integration_cell=one_cell)
    inventory = initialize_boundary(initial)
    initial_snapshot, inventory_snapshot = snapshot(initial), snapshot(inventory)
    result = advance_geometric_surface(mesh, initial, MOTION, .1, birth_factory=birth)
    before_acceptance = snapshot(result.state)
    accepted = consume_geometric_transaction(initial, result, inventory, MOTION, .1)
    assert result.losses and result.births
    assert len(accepted.cohorts) == len(result.losses)
    assert len({cohort.event_id for cohort in accepted.cohorts}) == len(result.losses)
    assert sorted(snapshot(cohort.loss) for cohort in accepted.cohorts) == sorted(
        snapshot(loss) for loss in result.losses)
    assert snapshot(initial) == initial_snapshot
    assert snapshot(inventory) == inventory_snapshot
    assert snapshot(result.state) == before_acceptance
    assert accepted.time_myr == result.state.time_myr
    assert_combined_budget(initial, result.state, accepted, result.births)
    if one_cell:
        assert len(accepted.cohorts) > 1
        assert {cohort.loss.fragment.parcel.cell for cohort in accepted.cohorts} == {0}


def test_replaying_accepted_or_altered_transaction_is_atomic_rejection():
    mesh, initial = source()
    inventory = initialize_boundary(initial)
    result = advance_geometric_surface(mesh, initial, MOTION, .1, birth_factory=birth)
    accepted = consume_geometric_transaction(initial, result, inventory, MOTION, .1)
    immutable = snapshot(accepted)
    with pytest.raises(ValueError):
        consume_geometric_transaction(initial, result, accepted, MOTION, .1)
    changed_loss = replace(result.losses[0], time_myr=.09)
    altered = replace(result, losses=(changed_loss,)+result.losses[1:])
    with pytest.raises(ValueError):
        consume_geometric_transaction(initial, altered, accepted, MOTION, .1)
    assert snapshot(accepted) == immutable


def test_wrong_pre_step_material_state_cannot_consume_another_states_events():
    mesh, initial = source()
    result = advance_geometric_surface(mesh, initial, MOTION, .1, birth_factory=birth)
    inventory = initialize_boundary(initial)
    fragment = initial.fragments[0]
    altered = replace(initial, fragments=(replace(fragment, parcel=replace(fragment.parcel,
        material_fields=(("damage", .987),))),)+initial.fragments[1:])
    with pytest.raises(ValueError):
        consume_geometric_transaction(altered, result, inventory, MOTION, .1)


def test_multiple_acceptance_times_preserve_each_material_history_and_all_budgets():
    mesh, initial = source()
    state, inventory, births, records = initial, initialize_boundary(initial), [], {}
    for _ in range(4):
        result = advance_geometric_surface(mesh, state, MOTION, .1, birth_factory=birth)
        updated = consume_geometric_transaction(state, result, inventory, MOTION, .1)
        for old in inventory.cohorts:
            records[old.event_id] = snapshot(old.loss)
        assert all(snapshot(cohort.loss) == records[cohort.event_id]
            for cohort in updated.cohorts if cohort.event_id in records)
        births.extend(result.births)
        state, inventory = result.state, updated
        assert_combined_budget(initial, state, inventory, births)
    times = {cohort.loss.time_myr for cohort in inventory.cohorts}
    assert len(times) == 4
    originals = {f.parcel.material_id: f.parcel for f in initial.fragments}
    for cohort in inventory.cohorts:
        parcel = cohort.loss.fragment.parcel
        if parcel.material_id in originals:
            ancestor = originals[parcel.material_id]
            assert parcel.material_fields == ancestor.material_fields
            assert parcel.specific_properties == ancestor.specific_properties
            assert parcel.age_myr == pytest.approx(ancestor.age_myr+cohort.loss.time_myr)


def test_geometric_slab_checkpoint_resume_is_exact_after_differential_then_common_motion(tmp_path):
    mesh, initial = source()
    first = advance_geometric_surface(mesh, initial, MOTION, .1, birth_factory=birth)
    inventory = consume_geometric_transaction(initial, first, initialize_boundary(initial), MOTION, .1)
    path = tmp_path/"with-geometric-slabs.json"
    save_geometric_checkpoint(path, first.state, provenance={"boundary": boundary_to_dict(inventory)})
    resumed_surface, metadata = load_geometric_checkpoint(path)
    resumed_boundary = boundary_from_dict(metadata["boundary"])
    assert snapshot(resumed_boundary) == snapshot(inventory)
    direct_surface, direct_boundary = first.state, inventory
    for _ in range(2):
        direct = advance_geometric_surface(mesh, direct_surface, COMMON, .1, birth_factory=birth)
        direct_boundary = consume_geometric_transaction(direct_surface, direct, direct_boundary, COMMON, .1)
        direct_surface = direct.state
        resumed = advance_geometric_surface(mesh, resumed_surface, COMMON, .1, birth_factory=birth)
        resumed_boundary = consume_geometric_transaction(resumed_surface, resumed, resumed_boundary, COMMON, .1)
        resumed_surface = resumed.state
    assert snapshot(direct_surface) == snapshot(resumed_surface)
    assert boundary_to_dict(direct_boundary) == boundary_to_dict(resumed_boundary)
    assert len(direct_boundary.cohorts) == len(inventory.cohorts)
    assert sorted(snapshot(c.loss) for c in direct_boundary.cohorts) == sorted(
        snapshot(c.loss) for c in inventory.cohorts)
    prior = {c.event_id: c for c in inventory.cohorts}
    for cohort in direct_boundary.cohorts:
        previous = prior[cohort.event_id]
        assert cohort.attachment_status == previous.attachment_status
        assert cohort.unresolved_support == previous.unresolved_support
        assert {lineage for support in cohort.supports for lineage in support.lineage_ids} == {
            lineage for support in previous.supports for lineage in support.lineage_ids}
    assert_combined_budget(initial, direct_surface, direct_boundary, first.births)


def test_empty_common_rotation_does_not_invent_slab_material():
    mesh, initial = source()
    result = advance_geometric_surface(mesh, initial, COMMON, .1, birth_factory=birth)
    inventory = consume_geometric_transaction(initial, result, initialize_boundary(initial), COMMON, .1)
    assert not inventory.cohorts
    assert result.losses == result.births == ()
    assert_combined_budget(initial, result.state, inventory, ())
