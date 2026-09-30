"""Local support lineage and passive inventory; no force or raster adapter."""
from dataclasses import asdict, replace
import copy
import math

import numpy as np
import pytest

from tectonics.fractional_surface import EXTENSIVE_FIELDS
from tectonics.geometric_boundary import (
    _arc_interval, _contact_frame, boundary_diagnostics, boundary_from_dict,
    boundary_to_dict, consume_geometric_transaction, initialize_boundary,
    polarity_evidence, surface_digest,
)
from tectonics.geometric_contacts import extract_contacts
from tectonics.geometric_surface import GeometricFragment, rotate_surface, split_geometry
from tectonics.geometric_transport import GeometricTransportResult, advance_geometric_surface
from tectonics.spherical_polygons import arc_length, clip_hemisphere
from test_geometric_transport import birth, fixture


@pytest.fixture(scope="module")
def accepted():
    mesh, initial = fixture()
    omega = np.array([[0., 0., .01], [0., 0., -.01]])
    before = initialize_boundary(initial)
    step = advance_geometric_surface(mesh, initial, omega, .25, birth_factory=birth)
    inventory = consume_geometric_transaction(initial, step, before, omega, .25)
    return mesh, initial, step, before, inventory


def _single_cohort(inventory):
    return max((cohort for cohort in inventory.cohorts if cohort.attachment_status == "geometric_contact"),
               key=lambda cohort: cohort.parcel.area_km2)


def _remesh_transaction(surface, fragments, dt):
    after = replace(surface, time_myr=surface.time_myr+dt, fragments=tuple(fragments))
    return GeometricTransportResult(after, (), (), extract_contacts(fragments, surface.radius_km), {})


def test_atomic_loss_acceptance_keeps_each_extensive_quantity_once(accepted):
    _, _, step, empty, inventory = accepted
    assert not empty.cohorts and not empty.transactions
    assert len(inventory.cohorts) == len(step.losses)
    diagnostics = boundary_diagnostics(inventory)
    for field in EXTENSIVE_FIELDS:
        actual = math.fsum(getattr(loss.fragment.parcel, field) for loss in step.losses)
        assert diagnostics["accepted"][field] == actual
        assert sum(group[field] for group in diagnostics["by_attachment"].values()) == pytest.approx(actual)
    assert diagnostics["forces_active"] is False
    assert any(cohort.attachment_status == "unresolved_lateral_allocation" for cohort in inventory.cohorts)


def test_json_roundtrip_preserves_all_bits_and_rejects_loss_payload_change(accepted):
    *_, inventory = accepted
    payload = boundary_to_dict(inventory)
    assert boundary_from_dict(payload) == inventory
    changed = copy.deepcopy(payload)
    changed["cohorts"][0]["loss"]["fragment"]["parcel"]["age_myr"] += 1.
    with pytest.raises(ValueError, match="payload digest"):
        boundary_from_dict(changed)


@pytest.mark.parametrize("corruption", ["repeat_event", "remove_registry", "change_surface", "bad_support"])
def test_internal_checkpoint_registry_is_validated(accepted, corruption):
    *_, inventory = accepted
    payload = boundary_to_dict(inventory)
    if corruption == "repeat_event":
        payload["cohorts"].append(payload["cohorts"][0])
    elif corruption == "remove_registry":
        payload["transactions"] = []
    elif corruption == "change_surface":
        payload["surface_digest"] = "0"*64
    else:
        cohort = next(row for row in payload["cohorts"] if row["supports"])
        cohort["supports"][0]["contact_id"] = "absent"
    with pytest.raises(ValueError):
        boundary_from_dict(payload)


def test_polarity_evidence_excludes_unresolved_multiarc_material(accepted):
    *_, inventory = accepted
    evidence = polarity_evidence(inventory)
    usable = [cohort for cohort in inventory.cohorts if cohort.attachment_status == "geometric_contact"]
    assert {row["evidence_id"] for row in evidence} == {cohort.event_id for cohort in usable}
    assert all(len(cohort.supports) == 1 for cohort in usable)


def test_common_rotation_keeps_support_identity_and_material(accepted):
    mesh, _, first, _, inventory = accepted
    common = np.array([[.002, -.003, .001]]*2)
    step = advance_geometric_surface(mesh, first.state, common, .25, birth_factory=birth)
    after = consume_geometric_transaction(first.state, step, inventory, common, .25)
    for old, new in zip(inventory.cohorts, after.cohorts):
        assert old.loss == new.loss
        assert old.attachment_status == new.attachment_status
        assert old.unresolved_support == new.unresolved_support
        assert {key for support in old.supports for key in support.lineage_ids} == {
            key for support in new.supports for key in support.lineage_ids}
        # The polygon kernel can represent incidental arcs shorter than one
        # binary64 coordinate ULP differently after rotation. Keep those
        # records/material, but do not confuse their IDs with resolved geometry.
        resolved = lambda supports: {support.contact_id for support in supports
            if arc_length(support.start, support.end) > 512.*np.finfo(float).eps}
        assert resolved(old.supports) == resolved(new.supports)
        if old.attachment_status == "geometric_contact":
            assert old.connectivity_history == new.connectivity_history
        assert all(change.reason != "support_disappeared" and change.uncovered_length_km == 0.
                   for change in new.connectivity_history)
    assert boundary_from_dict(boundary_to_dict(after)) == after


def test_contact_split_and_merge_preserve_one_cohort_and_local_lineage(accepted):
    _, _, first, _, inventory = accepted
    selected = _single_cohort(inventory)
    support = selected.supports[0]
    radius = first.state.radius_km
    dt = .25
    aged = rotate_surface(first.state, np.zeros((2, 3)), dt)
    receiver = next(fragment for fragment in aged.fragments if fragment.fragment_id == support.receiver_fragment_id)
    contact = next(contact for contact in inventory.contacts if contact.contact_id == support.contact_id)
    midpoint = np.asarray(contact.midpoint)
    center = np.asarray(receiver.polygon).sum(axis=0)
    normal = np.cross(midpoint, center)
    polygons = [clip_hemisphere(receiver.polygon, normal), clip_hemisphere(receiver.polygon, -normal)]
    children = split_geometry(receiver, polygons, label="integration-test-split")
    fragments = [fragment for fragment in aged.fragments if fragment.fragment_id != receiver.fragment_id]+list(children)
    step = _remesh_transaction(first.state, fragments, dt)
    split = consume_geometric_transaction(first.state, step, inventory, np.zeros((2, 3)), dt)
    cohort = next(cohort for cohort in split.cohorts if cohort.event_id == selected.event_id)
    assert cohort.loss == selected.loss
    assert len(cohort.supports) == 2
    assert cohort.attachment_status == "unresolved_lateral_allocation"
    assert not cohort.unresolved_support
    assert {lineage for item in cohort.supports for lineage in item.lineage_ids} == set(support.lineage_ids)
    assert sum(np.arccos(np.clip(np.dot(item.start, item.end), -1., 1.))*radius
               for item in cohort.supports) == pytest.approx(contact.length_km, rel=2e-10)

    # A material-preserving geometric merge has one continuous local receiver
    # again. It does not add the same historical slab twice.
    next_aged = rotate_surface(step.state, np.zeros((2, 3)), dt)
    merged = replace(receiver, fragment_id=receiver.fragment_id+"/merged",
                     parcel=replace(receiver.parcel, age_myr=receiver.parcel.age_myr+dt))
    child_ids = {child.fragment_id for child in children}
    merged_fragments = [fragment for fragment in next_aged.fragments if fragment.fragment_id not in child_ids]+[merged]
    merge_step = _remesh_transaction(step.state, merged_fragments, dt)
    joined = consume_geometric_transaction(step.state, merge_step, split, np.zeros((2, 3)), dt)
    joined_cohort = next(cohort for cohort in joined.cohorts if cohort.event_id == selected.event_id)
    assert len(joined_cohort.supports) == 1
    assert joined_cohort.attachment_status == "geometric_contact"
    assert joined_cohort.loss == selected.loss
    assert len(joined.cohorts) == len(inventory.cohorts)
    assert boundary_from_dict(boundary_to_dict(joined)) == joined


def test_missing_contact_metadata_is_rejected_atomically(accepted):
    _, _, first, _, inventory = accepted
    selected = _single_cohort(inventory)
    aged = rotate_surface(first.state, np.zeros((2, 3)), .25)
    selected_id = selected.supports[0].contact_id
    # Missing metadata is not evidence of a disappearing physical contact.
    remaining = tuple(contact for contact in extract_contacts(aged.fragments, aged.radius_km)
                      if contact.contact_id != selected_id)
    result = GeometricTransportResult(aged, (), (), remaining, {})
    snapshot = boundary_to_dict(inventory)
    with pytest.raises(ValueError, match="every actual current contact"):
        consume_geometric_transaction(first.state, result, inventory, np.zeros((2, 3)), .25)
    assert boundary_to_dict(inventory) == snapshot


def test_passive_warming_diagnostic_does_not_rewrite_archival_mass(accepted):
    mesh, _, first, _, inventory = accepted
    before = boundary_to_dict(inventory)
    zero = np.zeros((2, 3))
    step = advance_geometric_surface(mesh, first.state, zero, .25, birth_factory=birth)
    after = consume_geometric_transaction(first.state, step, inventory, zero, .25)
    diagnostics = boundary_diagnostics(after, thermal_diffusivity_m2_s=1e-6)
    assert 0. < diagnostics["passive_current_density_excess_mass_kg"] < diagnostics["accepted"]["density_excess_mass_kg"]
    assert boundary_to_dict(inventory) == before
    assert diagnostics["accepted"] == boundary_diagnostics(inventory)["accepted"]


def test_surface_hash_does_not_depend_on_fragment_iteration_order(accepted):
    _, initial, *_ = accepted
    assert surface_digest(initial) == surface_digest(replace(initial, fragments=initial.fragments[::-1]))
