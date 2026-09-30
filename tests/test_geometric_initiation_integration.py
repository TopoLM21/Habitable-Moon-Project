"""Independent full-sphere material checks of explicit fault admission.

These controls test physical locality, refusal before material changes, and
conservative positive transactions. Their stresses are prescribed synthetic
inputs, not inferred from existing scalar fracture history.
"""
from dataclasses import asdict, replace
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from analysis.geometric_initiation_validation.admission_case import (
    COMMON, DT, MOTION, birth, blocked_case, declared_fault, faults_for, fixture, ledger, snapshot)
from tectonics.geometric_boundary import consume_geometric_transaction, initialize_boundary
from tectonics.geometric_boundary_io import load_boundary_checkpoint, save_boundary_checkpoint
from tectonics.geometric_contacts import extract_contacts
from tectonics.geometric_initiation_adapter import make_forced_underthrust_resolver
from tectonics.geometric_polarity import PolarityEvidence
from tectonics.geometric_surface import candidate_pairs, rotate_surface
from tectonics.geometric_transport import advance_geometric_surface
from tectonics.spherical_polygons import intersect_convex


def overlaps(state, omega=MOTION):
    moved = rotate_surface(state, omega, DT)
    for i, j in candidate_pairs(moved.fragments):
        a, b = moved.fragments[i], moved.fragments[j]
        if a.parcel.plate != b.parcel.plate:
            overlap = intersect_convex(a.polygon, b.polygon)
            if len(overlap):
                yield a, b, overlap


def accepted(mesh, state, omega=MOTION, faults=None):
    if faults is None:
        _, faults = faults_for(state, omega=omega)
    resolver, audit = make_forced_underthrust_resolver(state, omega, DT, faults=faults)
    result = advance_geometric_surface(mesh, state, omega, DT, birth_factory=birth,
                                      polarity_resolver=resolver)
    boundary = consume_geometric_transaction(state, result, initialize_boundary(state), omega, DT)
    return result, boundary, audit


@pytest.mark.parametrize("tied", [False, True])
def test_absent_oriented_fault_never_falls_back_to_material_age_or_buoyancy(tied):
    mesh, state = fixture(tied=tied)
    report = blocked_case(mesh, state, ())
    assert report["overlap_reasons"] == {"no_oriented_fault": 2}
    assert report["birth_calls"] == 0 and report["material_unchanged"]


@pytest.mark.parametrize("tied", [False, True])
def test_explicit_mechanical_admission_accepts_positive_local_material_once(tied):
    mesh, state = fixture(tied=tied)
    original = snapshot(state)
    result, inventory, audit = accepted(mesh, state)
    assert len(result.losses) == len(inventory.cohorts) == 2
    assert audit["contact_status_counts"] == {
        "forced_underthrust_admissible": 2, "nonconvergent": 2}
    assert all(loss.fragment.parcel.plate == 1 and loss.receiver_plate == 0 for loss in result.losses)
    assert all(loss.polarity_basis == "forced_underthrust_admission" for loss in result.losses)
    assert all(loss.polarity_evidence_ids for loss in result.losses)
    assert result.births and all(c.parcel.area_km2 > 0. for c in inventory.cohorts)
    assert max(ledger(state, result.state, inventory, result.births).values()) < 5e-12
    assert snapshot(state) == original
    with pytest.raises(ValueError):
        consume_geometric_transaction(state, result, inventory, MOTION, DT)


def test_locked_fault_rejects_before_any_material_factory_or_archive_change():
    mesh, state = fixture(tied=False)
    _, faults = faults_for(state, locked=True)
    report = blocked_case(mesh, state, faults)
    assert report["overlap_reasons"] == {"locked": 2}
    assert report["contact_status_counts"] == {"locked": 2, "nonconvergent": 2}


def test_reversing_velocity_does_not_keep_old_convergent_contact_permission():
    mesh, state = fixture()
    _, faults = faults_for(state)
    _, audit = make_forced_underthrust_resolver(state, MOTION, DT, faults=faults)
    admitted = {row["contact_id"] for row in audit["contacts"]
                if row["status"] == "forced_underthrust_admissible"}
    selected = tuple(fault for fault in faults if fault.contact_id in admitted)
    report = blocked_case(mesh, state, selected, omega=-MOTION)
    assert report["overlap_reasons"] == {"no_oriented_fault": 2}
    assert report["contact_status_counts"]["nonconvergent"] == 2


def test_mechanically_admissible_opposite_directions_are_not_ranked_by_margin():
    mesh, state = fixture(tied=False)
    _, faults = faults_for(state, both=True)
    # Different valid excess shear does not define an unprovided selection law.
    faults = tuple(replace(f, cohesion_pa=1e6) if f.subducting_plate == 0 else f for f in faults)
    report = blocked_case(mesh, state, faults)
    assert report["overlap_reasons"] == {"multiple_admissible_directions": 2}


def test_conflicting_inherited_history_is_not_deleted_by_one_admissible_candidate():
    mesh, state = fixture()
    contacts, faults = faults_for(state)
    history = tuple(PolarityEvidence(f"accepted:{c.contact_id}:{sub}", c.contact_id, sub, 1-sub,
        "synthetic accepted inherited material") for c in contacts for sub in (0, 1))
    report = blocked_case(mesh, state, faults, history=history)
    assert report["overlap_reasons"] == {"conflicting_local_history": 2}


def test_inherited_direction_does_not_bypass_current_mechanical_lock():
    mesh, state = fixture()
    contacts, faults = faults_for(state, locked=True)
    history = tuple(PolarityEvidence(f"accepted:{c.contact_id}", c.contact_id, 1, 0,
        "synthetic accepted inherited material") for c in contacts)
    report = blocked_case(mesh, state, faults, history=history)
    assert report["overlap_reasons"] == {"history_mechanics_mismatch": 2}


def test_remote_same_pair_fault_cannot_authorize_a_local_candidate_overlap():
    _, state = fixture()
    contacts, faults = faults_for(state)
    full, _ = make_forced_underthrust_resolver(state, MOTION, DT, faults=faults)
    a, b, polygon = next(overlaps(state))
    allowed = full(a, b, polygon)
    assert allowed.comparison and allowed.contact_ids
    remote = next(f for f in faults if f.contact_id not in allowed.contact_ids
                  and np.dot(next(c.midpoint for c in contacts if c.contact_id == f.contact_id),
                             np.mean(polygon, axis=0)) < 0.)
    local, _ = make_forced_underthrust_resolver(state, MOTION, DT, faults=(remote,))
    rejected = local(a, b, polygon)
    assert rejected.comparison == 0 and rejected.basis == "no_oriented_fault"


def test_global_coordinate_rotation_preserves_admission_and_material_choice():
    mesh, state = fixture(tied=False)
    _, faults = faults_for(state)
    first, boundary, _ = accepted(mesh, state, faults=faults)
    matrix = Rotation.from_rotvec([.31, -.17, .24]).as_matrix()
    rotated = replace(state, fragments=tuple(replace(f,
        polygon=tuple(map(tuple, np.asarray(f.polygon)@matrix.T))) for f in state.fragments))
    rotated_faults = tuple(replace(f, effective_stress_pa=matrix@np.asarray(f.effective_stress_pa)@matrix.T)
                           for f in faults)
    moved, inventory, audit = accepted(mesh, rotated, MOTION@matrix.T, rotated_faults)
    assert audit["contact_status_counts"] == {"forced_underthrust_admissible": 2, "nonconvergent": 2}
    original_loss = {loss.fragment.parcel.material_id: loss.fragment.parcel.area_km2 for loss in first.losses}
    rotated_loss = {loss.fragment.parcel.material_id: loss.fragment.parcel.area_km2 for loss in moved.losses}
    assert original_loss.keys() == rotated_loss.keys()
    for identity in original_loss:
        assert rotated_loss[identity] == pytest.approx(original_loss[identity], rel=2e-11)
    assert max(ledger(rotated, moved.state, inventory, moved.births).values()) < 5e-12


def test_owner_relabeling_preserves_the_physical_donor_material():
    mesh, state = fixture(tied=False)
    _, faults = faults_for(state)
    first, _, _ = accepted(mesh, state, faults=faults)
    relabeled = replace(state, fragments=tuple(replace(f, parcel=replace(f.parcel, plate=1-f.parcel.plate))
                                              for f in reversed(state.fragments)))
    relabeled_faults = tuple(replace(f, subducting_plate=1-f.subducting_plate,
                                   overriding_plate=1-f.overriding_plate) for f in faults)
    second, _, _ = accepted(mesh, relabeled, MOTION[::-1], relabeled_faults)
    assert sorted(loss.fragment.parcel.material_id for loss in second.losses) == sorted(
        loss.fragment.parcel.material_id for loss in first.losses)
    assert all(loss.fragment.parcel.plate == 0 for loss in second.losses)


def test_joint_checkpoint_preserves_admitted_archive_and_exact_common_motion_resume(tmp_path):
    mesh, state = fixture()
    result, inventory, audit = accepted(mesh, state)
    metadata = json.loads(json.dumps({"mechanical_admission": audit, "force_evolution": False}))
    saved = tmp_path/"accepted.json"
    save_boundary_checkpoint(saved, result.state, inventory, provenance=metadata)
    loaded, loaded_inventory, loaded_metadata = load_boundary_checkpoint(saved)
    assert loaded_metadata == metadata
    assert snapshot(loaded) == snapshot(result.state)
    assert snapshot(loaded_inventory) == snapshot(inventory)
    finals = []
    for before, previous in ((result.state, inventory), (loaded, loaded_inventory)):
        resolver, _ = make_forced_underthrust_resolver(before, COMMON, DT)
        advance = advance_geometric_surface(mesh, before, COMMON, DT, birth_factory=birth,
                                            polarity_resolver=resolver)
        current = consume_geometric_transaction(before, advance, previous, COMMON, DT)
        assert advance.losses == advance.births == ()
        finals.append((advance.state, current))
    assert [snapshot(value) for value in finals[0]] == [snapshot(value) for value in finals[1]]
    for index, (surface, boundary) in enumerate(finals):
        save_boundary_checkpoint(tmp_path/f"final-{index}.json", surface, boundary, provenance=metadata)
    assert (tmp_path/"final-0.json").read_bytes() == (tmp_path/"final-1.json").read_bytes()
