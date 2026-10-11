"""A zero-time pure cut preserves actual archived slabs and replay history."""
from dataclasses import replace
import copy
import math

import numpy as np
import pytest

from tectonics.fractional_surface import EXTENSIVE_FIELDS
from tectonics.geometric_boundary import (
    _hash, _representation_payload, boundary_from_dict, boundary_to_dict,
    consume_geometric_transaction, initialize_boundary,
)
from tectonics.geometric_boundary_io import load_boundary_checkpoint, save_boundary_checkpoint
from tectonics.geometric_boundary_subdivision import rebind_boundary_for_material_subdivision
from tectonics.geometric_surface import split_geometry
from tectonics.geometric_transport import advance_geometric_surface
from tectonics.spherical_polygons import arc_length, clip_hemisphere
from test_geometric_transport import birth, fixture


@pytest.fixture(scope="module")
def archived():
    mesh, initial = fixture()
    omega = np.array([[0., 0., .01], [0., 0., -.01]])
    empty = initialize_boundary(initial)
    step = advance_geometric_surface(mesh, initial, omega, .25, birth_factory=birth)
    inventory = consume_geometric_transaction(initial, step, empty, omega, .25)
    cohort = max((row for row in inventory.cohorts if row.attachment_status == "geometric_contact"),
                 key=lambda row: row.parcel.area_km2)
    return mesh, step.state, inventory, cohort.event_id


def cut_receiver(surface, inventory, event_id, *, label="zero-time-support-cut"):
    cohort = next(row for row in inventory.cohorts if row.event_id == event_id)
    support = max(cohort.supports, key=lambda item: arc_length(item.start, item.end))
    receiver = next(row for row in surface.fragments if row.fragment_id == support.receiver_fragment_id)
    midpoint = np.asarray(support.start)+np.asarray(support.end)
    midpoint /= np.linalg.norm(midpoint)
    center = np.asarray(receiver.polygon).sum(axis=0)
    normal = np.cross(midpoint, center)
    polygons = (clip_hemisphere(receiver.polygon, normal), clip_hemisphere(receiver.polygon, -normal))
    children = split_geometry(receiver, polygons, label=label)
    fragments = tuple(row for row in surface.fragments if row.fragment_id != receiver.fragment_id)+children
    return replace(surface, fragments=fragments)


def test_nonempty_archive_keeps_one_payload_and_complete_support_after_cut(archived):
    _, surface, inventory, event_id = archived
    before = boundary_to_dict(inventory)
    cut = cut_receiver(surface, inventory, event_id)
    rebound = rebind_boundary_for_material_subdivision(surface, cut, inventory)
    assert boundary_to_dict(inventory) == before
    assert rebound.time_myr == inventory.time_myr == cut.time_myr
    assert rebound.transactions == inventory.transactions
    assert len(rebound.cohorts) == len(inventory.cohorts) > 0
    assert len(rebound.representations) == 1
    for old, new in zip(inventory.cohorts, rebound.cohorts):
        assert (old.event_id, old.payload_sha256, old.loss) == (new.event_id, new.payload_sha256, new.loss)
        assert new.unresolved_support == old.unresolved_support
        assert math.fsum(arc_length(item.start, item.end) for item in new.supports) == pytest.approx(
            math.fsum(arc_length(item.start, item.end) for item in old.supports), rel=512*np.finfo(float).eps)
        for field in EXTENSIVE_FIELDS:
            assert getattr(new.parcel, field) == getattr(old.parcel, field)
    old = next(row for row in inventory.cohorts if row.event_id == event_id)
    new = next(row for row in rebound.cohorts if row.event_id == event_id)
    assert len(old.supports) == 1 and len(new.supports) == 2
    assert new.attachment_status == "unresolved_lateral_allocation"
    assert not new.unresolved_support
    assert {key for support in new.supports for key in support.lineage_ids} == set(old.supports[0].lineage_ids)
    assert new.connectivity_history[-1].reason == "material_subdivision"
    assert new.connectivity_history[-1].uncovered_length_km == 0.
    assert boundary_from_dict(boundary_to_dict(rebound)) == rebound


def test_joint_checkpoint_after_cut_and_continued_physical_transaction(archived, tmp_path):
    mesh, surface, inventory, event_id = archived
    cut = cut_receiver(surface, inventory, event_id)
    rebound = rebind_boundary_for_material_subdivision(surface, cut, inventory)
    path = save_boundary_checkpoint(tmp_path/"cut.json", cut, rebound, provenance={"case": "nonempty-pure-cut"})
    loaded_surface, loaded_inventory, provenance = load_boundary_checkpoint(path)
    assert loaded_surface == cut and loaded_inventory == rebound
    assert provenance == {"case": "nonempty-pure-cut"}
    zero = np.zeros((2, 3))
    step = advance_geometric_surface(mesh, cut, zero, .25, birth_factory=birth)
    continued = consume_geometric_transaction(cut, step, rebound, zero, .25)
    resumed = consume_geometric_transaction(loaded_surface, step, loaded_inventory, zero, .25)
    assert continued == resumed
    assert continued.representations == rebound.representations
    assert continued.transactions[:-1] == inventory.transactions
    assert len(continued.cohorts) == len(inventory.cohorts)
    assert boundary_from_dict(boundary_to_dict(continued)) == continued


def test_two_consecutive_cuts_keep_zero_time_registry_order(archived):
    _, surface, inventory, event_id = archived
    first = cut_receiver(surface, inventory, event_id)
    rebound = rebind_boundary_for_material_subdivision(surface, first, inventory)
    second = cut_receiver(first, rebound, event_id, label="second-support-cut")
    twice = rebind_boundary_for_material_subdivision(first, second, rebound)
    assert len(twice.representations) == 2
    assert twice.transactions == inventory.transactions
    assert twice.time_myr == inventory.time_myr
    assert boundary_from_dict(boundary_to_dict(twice)) == twice
    corrupt = boundary_to_dict(twice)
    corrupt["representations"].reverse()
    with pytest.raises(ValueError, match="ordered surface history"):
        boundary_from_dict(corrupt)


def test_old_v1_without_optional_representation_registry_reads_exactly(archived):
    _, _, inventory, _ = archived
    data = boundary_to_dict(inventory)
    assert data.pop("representations") == []
    assert boundary_from_dict(data) == inventory


def test_identical_surface_representation_is_noop(archived):
    _, surface, inventory, _ = archived
    reordered = replace(surface, fragments=surface.fragments[::-1])
    assert rebind_boundary_for_material_subdivision(surface, reordered, inventory) is inventory


@pytest.mark.parametrize("corruption", [
    "remove", "duplicate", "parents", "before", "time", "rehashed_before", "rehashed_time", "transaction",
])
def test_representation_cannot_bypass_acceptance_or_surface_history(archived, corruption):
    _, surface, inventory, event_id = archived
    cut = cut_receiver(surface, inventory, event_id)
    rebound = rebind_boundary_for_material_subdivision(surface, cut, inventory)
    data = boundary_to_dict(rebound)
    record = data["representations"][0]
    if corruption == "remove":
        data["representations"] = []
    elif corruption == "duplicate":
        data["representations"].append(copy.deepcopy(record))
    elif corruption == "parents":
        record["fragment_parents"][0][1] = "unverified-parent"
    elif corruption.endswith("before"):
        record["before_surface_digest"] = "a"*64
    elif corruption.endswith("time"):
        record["time_myr"] = 0.
    elif corruption == "transaction":
        data["transactions"] = []
    if corruption.startswith("rehashed_"):
        digest = _hash(_representation_payload(record["before_surface_digest"],
            record["after_surface_digest"], record["time_myr"], record["fragment_parents"]))
        record["payload_sha256"] = digest
        record["record_id"] = "geometric-representation:"+digest
    with pytest.raises(ValueError):
        boundary_from_dict(data)


def test_rebind_rejects_history_change_and_old_binding_atomically(archived):
    _, surface, inventory, event_id = archived
    cut = cut_receiver(surface, inventory, event_id)
    snapshot = boundary_to_dict(inventory)
    changed_fragment = replace(cut.fragments[-1],
        parcel=replace(cut.fragments[-1].parcel, age_myr=cut.fragments[-1].parcel.age_myr+1.))
    changed = replace(cut, fragments=cut.fragments[:-1]+(changed_fragment,))
    with pytest.raises(ValueError, match="material history"):
        rebind_boundary_for_material_subdivision(surface, changed, inventory)
    with pytest.raises(ValueError, match="same state"):
        rebind_boundary_for_material_subdivision(cut, surface, inventory)
    assert boundary_to_dict(inventory) == snapshot


def test_empty_archive_can_subdivide_before_first_physical_transaction(archived):
    mesh, surface, _, _ = archived
    inventory = initialize_boundary(surface)
    contact = inventory.contacts[0]
    receiver = next(row for row in surface.fragments if row.fragment_id == contact.fragment_a)
    normal = np.cross(contact.midpoint, np.asarray(receiver.polygon).sum(axis=0))
    children = split_geometry(receiver,
        (clip_hemisphere(receiver.polygon, normal), clip_hemisphere(receiver.polygon, -normal)),
        label="empty-archive-cut")
    cut = replace(surface, fragments=tuple(row for row in surface.fragments
        if row.fragment_id != receiver.fragment_id)+children)
    rebound = rebind_boundary_for_material_subdivision(surface, cut, inventory)
    assert not rebound.cohorts and not rebound.transactions
    assert len(rebound.representations) == 1
    zero = np.zeros((2, 3))
    step = advance_geometric_surface(mesh, cut, zero, .25, birth_factory=birth)
    continued = consume_geometric_transaction(cut, step, rebound, zero, .25)
    assert boundary_from_dict(boundary_to_dict(continued)) == continued


@pytest.mark.parametrize("clock,boolean", [(0., False), (1., True)])
def test_rehashed_boolean_representation_clock_is_rejected(clock, boolean):
    # The numeric equality False == 0 and True == 1 would otherwise also pass
    # the complete contiguous-clock check. Use an empty archive so this tests
    # the representation's own clock semantics without an acceptance failure.
    _, initial = fixture()
    surface = replace(initial, time_myr=clock)
    inventory = initialize_boundary(surface)
    contact = inventory.contacts[0]
    receiver = next(row for row in surface.fragments if row.fragment_id == contact.fragment_a)
    normal = np.cross(contact.midpoint, np.asarray(receiver.polygon).sum(axis=0))
    children = split_geometry(receiver,
        (clip_hemisphere(receiver.polygon, normal), clip_hemisphere(receiver.polygon, -normal)),
        label="boolean-clock-cut")
    cut = replace(surface, fragments=tuple(row for row in surface.fragments
        if row.fragment_id != receiver.fragment_id)+children)
    rebound = rebind_boundary_for_material_subdivision(surface, cut, inventory)
    data = boundary_to_dict(rebound)
    record = data["representations"][0]
    record["time_myr"] = boolean
    digest = _hash(_representation_payload(record["before_surface_digest"],
        record["after_surface_digest"], boolean, record["fragment_parents"]))
    record["payload_sha256"] = digest
    record["record_id"] = "geometric-representation:"+digest
    with pytest.raises(ValueError, match="representation record"):
        boundary_from_dict(data)
