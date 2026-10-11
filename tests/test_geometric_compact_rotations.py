"""Actual sparse-owner material transactions and passive archive acceptance."""
from dataclasses import asdict, replace
from functools import lru_cache
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tests.test_geometric_distributed_contact_invariants import world
from tectonics.fractional_surface import EXTENSIVE_FIELDS, SurfaceParcel
from tectonics.geometric_boundary import (
    _hash, boundary_from_dict, boundary_to_dict, consume_geometric_transaction,
    initialize_boundary, surface_digest,
)
from tectonics.geometric_boundary_io import load_boundary_checkpoint, save_boundary_checkpoint
from tectonics.geometric_contacts import extract_contacts
from tectonics.geometric_rotation_vectors import compact_rotation_vectors
from tectonics.geometric_surface import audit_partition, rotate_surface, rotations
from tectonics.geometric_transport import advance_geometric_surface


OWNERS = (7, 10**12)
COMMON = (.002, -.003, .001)


def birth(cell, owner, area, time, serial):
    return SurfaceParcel(cell, owner, f"compact-birth:{time.hex()}:{serial}",
        area, 2.*area, 0., 0., 0., (("damage", 0.),))


@lru_cache(maxsize=2)
def fixture(*, sparse=True):
    mesh, surface, _ = world()
    if sparse:
        surface = replace(surface, fragments=tuple(replace(item,
            parcel=replace(item.parcel, plate=OWNERS[item.parcel.plate]))
            for item in surface.fragments))
    return mesh, surface


def _totals_by_history(fragments, field):
    groups = {}
    for item in fragments:
        groups.setdefault(item.parcel.material_id, []).append(getattr(item.parcel, field))
    return {identity: math.fsum(values) for identity, values in groups.items()}


def _check_four_history_balances(before, result):
    for field in EXTENSIVE_FIELDS:
        initial = _totals_by_history(before.fragments, field)
        retained = _totals_by_history(result.state.fragments, field)
        born = _totals_by_history(result.births, field)
        lost = _totals_by_history((loss.fragment for loss in result.losses), field)
        for identity in initial.keys() | retained.keys() | born.keys() | lost.keys():
            assert retained.get(identity, 0.)+lost.get(identity, 0.) == pytest.approx(
                initial.get(identity, 0.)+born.get(identity, 0.), rel=5e-12, abs=0.)


def test_real_common_sparse_rotation_keeps_every_material_history_without_source_or_sink():
    mesh, surface = fixture()
    initial = asdict(surface)
    velocities = {OWNERS[1]: np.asarray(COMMON), OWNERS[0]: list(COMMON)}
    result = advance_geometric_surface(mesh, surface, velocities, .25, birth_factory=birth)
    inventory = consume_geometric_transaction(surface, result, initialize_boundary(surface), velocities, .25)
    assert result.losses == result.births == ()
    assert result.state.phase == "partition"
    audit_partition(result.state)
    assert asdict(surface) == initial
    assert inventory.cohorts == () and len(inventory.transactions) == 1
    assert {item.parcel.plate for item in result.state.fragments} == set(OWNERS)
    _check_four_history_balances(surface, result)
    matrix = Rotation.from_rotvec(.25*np.asarray(COMMON)).as_matrix()
    for old, new in zip(surface.fragments, result.state.fragments):
        np.testing.assert_allclose(new.polygon, np.asarray(old.polygon)@matrix.T,
                                   rtol=0., atol=5*np.finfo(float).eps)
        assert new.parcel.age_myr == old.parcel.age_myr+.25
        assert replace(new.parcel, age_myr=old.parcel.age_myr) == old.parcel
    assert boundary_from_dict(boundary_to_dict(inventory)) == inventory


def test_compact_contact_velocities_use_actual_sparse_owner_pair():
    _, surface = fixture()
    values = {OWNERS[0]: (.021, -.033, .046), OWNERS[1]: (.01, .023, -.05)}
    contacts = extract_contacts(surface.fragments, surface.radius_km, values)
    assert contacts
    for contact in contacts:
        relative = np.asarray(values[contact.plate_b])-values[contact.plate_a]
        physical_velocity = surface.radius_km*np.cross(relative, contact.midpoint)
        assert contact.normal_velocity_km_per_myr == pytest.approx(
            physical_velocity@contact.normal_a_to_b, rel=2e-15, abs=1e-15)
        assert contact.tangential_velocity_km_per_myr == pytest.approx(
            physical_velocity@contact.tangent, rel=2e-15, abs=1e-15)
        assert contact.normal_area_rate_km2_per_myr == pytest.approx(
            relative@contact.moment_arm_cross_normal_km2, rel=2e-15, abs=1e-15)


@lru_cache(maxsize=1)
def archived():
    mesh, before = fixture()
    motion = {OWNERS[0]: (0., 0., 0.), OWNERS[1]: (0., .01, 0.)}
    result = advance_geometric_surface(mesh, before, motion, .001, birth_factory=birth)
    inventory = consume_geometric_transaction(before, result, initialize_boundary(before), motion, .001)
    assert len(inventory.cohorts) == len(result.losses) == 2
    _check_four_history_balances(before, result)
    return mesh, result.state, inventory


def test_real_compact_acceptance_transfers_two_existing_cohorts_and_restarts_exactly(tmp_path):
    mesh, before, inventory = archived()
    original = asdict(before), boundary_to_dict(inventory)
    velocities = {owner: COMMON for owner in OWNERS}
    result = advance_geometric_surface(mesh, before, velocities, .1, birth_factory=birth)
    after = consume_geometric_transaction(before, result, inventory, velocities, .1)
    assert result.losses == result.births == ()
    assert len(after.cohorts) == 2 and after.transactions[:-1] == inventory.transactions
    _check_four_history_balances(before, result)
    rotation = Rotation.from_rotvec(.1*np.asarray(COMMON)).as_matrix()
    for old, new in zip(inventory.cohorts, after.cohorts):
        assert (old.event_id, old.payload_sha256, old.loss) == (new.event_id, new.payload_sha256, new.loss)
        assert old.attachment_status == new.attachment_status
        assert old.unresolved_support == new.unresolved_support
        assert len(old.supports) == len(new.supports) > 0
        for support, moved in zip(old.supports, new.supports):
            assert support.contact_id == moved.contact_id and support.lineage_ids == moved.lineage_ids
            np.testing.assert_allclose((moved.start, moved.end),
                np.asarray((support.start, support.end))@rotation.T, rtol=0., atol=5*np.finfo(float).eps)
        assert all(change.uncovered_length_km == 0. for change in new.connectivity_history)
        for field in EXTENSIVE_FIELDS:
            assert getattr(old.parcel, field) == getattr(new.parcel, field)
    assert (asdict(before), boundary_to_dict(inventory)) == original
    save_boundary_checkpoint(tmp_path/"compact.json", result.state, after, provenance={"case": "compact"})
    restored_surface, restored_inventory, metadata = load_boundary_checkpoint(tmp_path/"compact.json")
    assert restored_surface == result.state and restored_inventory == after and metadata == {"case": "compact"}
    direct_step = advance_geometric_surface(mesh, result.state, velocities, .1, birth_factory=birth)
    resumed_step = advance_geometric_surface(mesh, restored_surface, velocities, .1, birth_factory=birth)
    direct = consume_geometric_transaction(result.state, direct_step, after, velocities, .1)
    resumed = consume_geometric_transaction(restored_surface, resumed_step, restored_inventory, velocities, .1)
    assert direct_step == resumed_step and direct == resumed
    save_boundary_checkpoint(tmp_path/"direct.json", direct_step.state, direct)
    save_boundary_checkpoint(tmp_path/"resumed.json", resumed_step.state, resumed)
    assert (tmp_path/"direct.json").read_bytes() == (tmp_path/"resumed.json").read_bytes()


def test_existing_dense_transaction_hash_encoding_stays_unchanged():
    mesh, surface = fixture(sparse=False)
    dense = np.asarray((COMMON, COMMON))
    result = advance_geometric_surface(mesh, surface, dense, .25, birth_factory=birth)
    empty = initialize_boundary(surface)
    accepted = consume_geometric_transaction(surface, result, empty, dense, .25)
    # This is the established dense transaction body, before compact maps exist.
    original_payload = dict(before_surface_digest=surface_digest(surface),
        after_surface_digest=surface_digest(result.state), omega=dense.tolist(), dt=.25,
        losses=[], births=[], contacts=[asdict(contact) for contact in result.contacts])
    assert accepted.transactions[0].payload_sha256 == _hash(original_payload)
    compact = {1: list(COMMON), 0: np.asarray(COMMON)}
    compact_result = advance_geometric_surface(mesh, surface, compact, .25, birth_factory=birth)
    assert compact_result == result
    mapped = consume_geometric_transaction(surface, compact_result, empty, compact, .25)
    compact_payload = dict(original_payload, omega={
        "format": "compact-material-owner-angular-velocities-1",
        "owner_vectors": [[0, list(COMMON)], [1, list(COMMON)]]})
    assert mapped.transactions[0].payload_sha256 == _hash(compact_payload)
    assert mapped.transactions[0].payload_sha256 != accepted.transactions[0].payload_sha256
    assert mapped.cohorts == accepted.cohorts


def test_prepared_compact_vectors_copy_inputs_and_cannot_be_made_mutable():
    mesh, surface = fixture()
    source = np.asarray(COMMON)
    mapping = {owner: source for owner in OWNERS}
    validated = rotations(surface, mapping, .1)
    source[:] = 0.
    mapping[OWNERS[0]] = np.zeros(3)
    for owner in OWNERS:
        np.testing.assert_array_equal(validated[owner], COMMON)
        with pytest.raises(ValueError):
            validated[owner][0] = 0.
        with pytest.raises(ValueError):
            validated[owner].flags.writeable = True
    with pytest.raises(TypeError):
        validated[OWNERS[0]] = np.zeros(3)
    result = advance_geometric_surface(mesh, surface, validated, .1, birth_factory=birth)
    expected = advance_geometric_surface(mesh, surface, {owner: COMMON for owner in OWNERS}, .1, birth_factory=birth)
    assert result == expected


@pytest.mark.parametrize("bad_owner", [True, np.bool_(True), -1, "7", 7., None])
def test_actual_rotation_and_contacts_reject_untyped_or_negative_mapping_owner(bad_owner):
    mesh, surface = fixture()
    mapping = {bad_owner: COMMON, OWNERS[1]: COMMON}
    before = asdict(surface)
    for action in (
            lambda: rotate_surface(surface, mapping, .1),
            lambda: advance_geometric_surface(mesh, surface, mapping, .1, birth_factory=birth),
            lambda: extract_contacts(surface.fragments, surface.radius_km, mapping)):
        with pytest.raises(ValueError):
            action()
    assert asdict(surface) == before


@pytest.mark.parametrize("keys", [(OWNERS[0],), (OWNERS[1],), (*OWNERS, 42), (7, 42)])
def test_compact_maps_cover_exactly_existing_material_owners(keys):
    mesh, surface = fixture()
    mapping = {owner: COMMON for owner in keys}
    for action in (
            lambda: rotate_surface(surface, mapping, .1),
            lambda: advance_geometric_surface(mesh, surface, mapping, .1, birth_factory=birth),
            lambda: extract_contacts(surface.fragments, surface.radius_km, mapping)):
        with pytest.raises(ValueError, match="actual owner"):
            action()


@pytest.mark.parametrize("bad_vector", [np.asarray(1.), np.zeros((3, 1)), np.zeros((1, 3)),
    (1., 2.), (1., 2., 3., 4.), (True, 0., 0.), (np.bool_(False), 0., 0.),
    (math.nan, 0., 0.), (math.inf, 0., 0.), ("0", 0., 0.), (1j, 0., 0.), None])
def test_actual_rotation_and_contacts_reject_invalid_mapping_vector(bad_vector):
    mesh, surface = fixture()
    mapping = {OWNERS[0]: bad_vector, OWNERS[1]: COMMON}
    for action in (
            lambda: rotate_surface(surface, mapping, .1),
            lambda: advance_geometric_surface(mesh, surface, mapping, .1, birth_factory=birth),
            lambda: extract_contacts(surface.fragments, surface.radius_km, mapping)):
        with pytest.raises(ValueError):
            action()


@pytest.mark.parametrize("dt", [True, np.bool_(False), 0., -1., math.nan, math.inf, ".1", None, np.asarray(.1)])
def test_compact_public_transport_requires_a_typed_positive_advancing_clock(dt):
    mesh, surface = fixture()
    mapping = {owner: COMMON for owner in OWNERS}
    for action in (
            lambda: rotations(surface, mapping, dt),
            lambda: rotate_surface(surface, mapping, dt),
            lambda: advance_geometric_surface(mesh, surface, mapping, dt, birth_factory=birth)):
        with pytest.raises(ValueError):
            action()


@pytest.mark.parametrize("dt", [True, "1"])
def test_compact_archive_acceptance_rejects_coercible_untyped_clocks(dt):
    mesh, surface = fixture()
    mapping = {owner: COMMON for owner in OWNERS}
    result = advance_geometric_surface(mesh, surface, mapping, 1., birth_factory=birth)
    empty = initialize_boundary(surface)
    with pytest.raises(ValueError):
        consume_geometric_transaction(surface, result, empty, mapping, dt)
    assert empty.cohorts == empty.transactions == ()


def test_compact_clock_must_advance_saved_finite_surface_time():
    mesh, surface = fixture()
    later = replace(surface, time_myr=1.)
    mapping = {owner: COMMON for owner in OWNERS}
    for action in (
            lambda: rotate_surface(later, mapping, np.nextafter(0., 1.)),
            lambda: advance_geometric_surface(mesh, later, mapping, np.nextafter(0., 1.), birth_factory=birth)):
        with pytest.raises(ValueError, match="advancing"):
            action()


def test_compact_helper_requires_mapping_and_accepts_typed_numpy_owner_keys():
    with pytest.raises(ValueError, match="mapping"):
        compact_rotation_vectors(np.zeros((2, 3)), OWNERS)
    _, surface = fixture()
    mapping = {np.int64(OWNERS[0]): COMMON, np.int64(OWNERS[1]): COMMON}
    assert extract_contacts(surface.fragments, surface.radius_km, mapping) == extract_contacts(
        surface.fragments, surface.radius_km, {owner: COMMON for owner in OWNERS})
