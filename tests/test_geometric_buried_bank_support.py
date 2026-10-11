"""Original two-dimensional beam mass, acceptance poses and live clamps."""
from dataclasses import asdict, replace
from functools import lru_cache
import json
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tests.test_geometric_bending_contact_dynamics import prepared
from tests.test_geometric_distributed_contact_invariants import world
from tectonics.fractional_surface import EXTENSIVE_FIELDS, SurfaceParcel
from tectonics.geometric_boundary import (
    consume_geometric_transaction, initialize_boundary, surface_digest,
)
from tectonics.geometric_buried_bank_support import (
    SUPPORT_MODEL, _constant_width_quad, _hash, _inventory_hash,
    _pullback, _sealed, advance_buried_bank_registry, buried_bank_registry_from_dict,
    buried_bank_registry_to_dict, initial_buried_bank_registry,
    validate_buried_bank_support,
)
from tectonics.geometric_fault_motion_support import UnresolvedFaultMotionSupport
from tectonics.geometric_fault_traces import _arc_frame
from tectonics.geometric_surface import split_geometry
from tectonics.spherical_polygons import (
    _edge_distance, clip_hemisphere, normalize_polygon, polygon_area,
)


def birth(cell, owner, area, moment, serial):
    return SurfaceParcel(cell, owner, f"bank-birth:{moment.hex()}:{serial}",
                         area, 2.*area, 0., 0., 0., (("damage", 0.),))


@lru_cache(maxsize=None)
def initial():
    origin, _, _, seed = prepared()
    inventory = initialize_boundary(origin)
    registry = initial_buried_bank_registry(origin, inventory)
    poses = tuple((owner, tuple(map(tuple, np.eye(3)))) for owner in seed.plate_ids)
    return origin, seed, inventory, registry, poses


@lru_cache(maxsize=None)
def accepted(*, count=2, rotation=.00001, noncommuting=False):
    mesh, _, _ = world()
    origin, seed, archive, registry, pose_rows = initial()
    current = origin
    poses = dict(pose_rows)
    transactions = []
    for index in range(count):
        vector = (rotation, 0., 0.) if noncommuting and index % 2 else (0., rotation, 0.)
        velocities = {0: np.asarray(vector)/.01, 1: np.zeros(3)}
        # These explicitly prescribed transactions test material support only;
        # no claim that this fixture derives its motion from a force solution.
        from tectonics.geometric_transport import advance_geometric_surface
        transaction = advance_geometric_surface(mesh, current, velocities, .01, birth_factory=birth)
        after = consume_geometric_transaction(current, transaction, archive, velocities, .01)
        poses = {owner: Rotation.from_rotvec(velocities[owner]*.01).as_matrix()@np.asarray(matrix)
                 for owner, matrix in poses.items()}
        registry = advance_buried_bank_registry(origin, registry, archive, after, poses,
                                               roundoff_steps=index+1)
        current, archive = transaction.state, after
        transactions.append(transaction)
    return origin, seed, current, archive, registry, poses, tuple(transactions)


def certify(case, steps):
    origin, seed, current, archive, registry, poses = case[:6]
    return validate_buried_bank_support(origin, seed, current, archive, poses, registry,
                                        roundoff_steps=steps)


def rebound_inventory(surface, inventory, registry):
    # Only used for adversarial partitions without cohorts.  Their archive is
    # reconstructed honestly rather than leaving a trivially broken digest.
    assert not inventory.cohorts
    archive = initialize_boundary(surface)
    registry = _sealed(replace(registry, current_inventory_sha256=_inventory_hash(archive)),
                       "registry_sha256")
    return archive, registry


def test_initial_receipt_certifies_whole_original_2d_domains_and_continuous_live_clamps():
    origin, seed, archive, registry, poses = initial()
    checked = validate_buried_bank_support(origin, seed, origin, archive, poses, registry)
    assert checked.model == SUPPORT_MODEL
    assert len(checked.original_material_domains) == len(origin.fragments)
    assert len(checked.live_clamps) == 2*len(seed.fault_traces.traces)
    assert not registry.records and not registry.initial_event_payloads
    for domain in checked.original_material_domains:
        assert len(domain.live_fragment_ids) == 1 and not domain.buried_event_ids
        assert {field for field, _ in domain.conserved_reservoirs} == set(EXTENSIVE_FIELDS)
    for clamp in checked.live_clamps:
        assert clamp.live_intervals
        assert min(low for _, low, _ in clamp.live_intervals) == 0.
        assert max(high for _, _, high in clamp.live_intervals) == clamp.extent_radians


def test_actual_burial_retains_full_original_beam_domain_and_each_material_reservoir():
    case = accepted()
    origin, _, current, archive, registry, _, transactions = case
    checked = certify(case, 2)
    assert len(registry.records) == len(archive.cohorts) == 4
    assert all(transaction.losses and transaction.births for transaction in transactions)
    assert any(domain.buried_event_ids for domain in checked.original_material_domains)
    originals = {fragment.parcel.material_id for fragment in origin.fragments}
    assert {fragment.parcel.material_id for fragment in current.fragments}-originals
    assert all(domain.material_id in originals for domain in checked.original_material_domains)
    for domain in checked.original_material_domains:
        wanted = next(fragment.parcel for fragment in origin.fragments
                      if fragment.parcel.material_id == domain.material_id)
        material = [fragment.parcel for fragment in current.fragments
                    if fragment.parcel.material_id == domain.material_id]
        material += [cohort.parcel for cohort in archive.cohorts
                     if cohort.event_id in domain.buried_event_ids]
        for field, expected in domain.conserved_reservoirs:
            assert expected == getattr(wanted, field)
            assert math.fsum(getattr(parcel, field) for parcel in material) == pytest.approx(
                expected, rel=16*np.finfo(float).eps)


def test_noncommuting_owner_maps_preserve_retired_acceptance_coordinates():
    case = accepted(noncommuting=True)
    _, _, _, archive, registry, poses, _ = case
    checked = certify(case, 2)
    assert checked.registry_sha256 == registry.registry_sha256
    cohort = {item.event_id: item for item in archive.cohorts}
    first = registry.records[0]
    assert first.acceptance_roundoff_steps == 1
    assert first.original_polygon == _pullback(cohort[first.event_id].loss.fragment.polygon,
                                               first.acceptance_rotation)
    assert first.original_polygon != _pullback(cohort[first.event_id].loss.fragment.polygon,
                                               poses[first.donor_owner])
    assert registry.records[:2] == accepted(count=1)[4].records


def test_later_map_cannot_be_rehashed_as_the_acceptance_map_of_an_old_loss():
    origin, seed, current, archive, registry, poses, _ = accepted(noncommuting=True)
    records = []
    cohorts = {cohort.event_id: cohort for cohort in archive.cohorts}
    for record in registry.records:
        matrix = tuple(map(tuple, poses[record.donor_owner]))
        records.append(_sealed(replace(record, acceptance_rotation=matrix,
            acceptance_roundoff_steps=2,
            original_polygon=_pullback(cohorts[record.event_id].loss.fragment.polygon, matrix)),
            "record_sha256"))
    changed = _sealed(replace(registry, records=tuple(records)), "registry_sha256")
    with pytest.raises(UnresolvedFaultMotionSupport, match="2D footprint|overlap|missing 2D"):
        validate_buried_bank_support(origin, seed, current, archive, poses, changed, roundoff_steps=2)


def test_archived_material_does_not_replace_a_buried_clamp_even_with_complete_domain_mass():
    case = accepted(count=1, rotation=.15)
    assert case[-1][0].losses
    with pytest.raises(UnresolvedFaultMotionSupport, match="clamp"):
        certify(case, 1)


def test_same_owner_newborn_surrogate_cannot_replace_an_original_beam_footprint():
    origin, seed, archive, registry, poses = initial()
    target = origin.fragments[0]
    replacement = replace(target, parcel=replace(target.parcel, material_id="newborn-surrogate"))
    changed = replace(origin, fragments=(replacement, *origin.fragments[1:]),
                      known_material_ids=origin.known_material_ids+("newborn-surrogate",))
    archive, registry = rebound_inventory(changed, archive, registry)
    with pytest.raises(UnresolvedFaultMotionSupport, match="missing 2D"):
        validate_buried_bank_support(origin, seed, changed, archive, poses, registry)


@pytest.mark.parametrize("width", [1e-4, 1e-10, 1e-12])
def test_positive_2d_material_sliver_cannot_be_replaced_between_saved_gauss_nodes(width):
    origin, seed, archive, registry, poses = initial()
    target = next(fragment for fragment in origin.fragments if fragment.parcel.plate == 0
                  and np.any(np.asarray(fragment.polygon)[:, 0] == 1.))
    # A meridional strip inside the original octant, independent of saved
    # contact quadrature.  Replacing its material keeps total area and mass.
    first = np.asarray((1., 0., -math.tan(.03)))
    last = np.asarray((1., 0., -math.tan(.03+width)))
    strip = clip_hemisphere(clip_hemisphere(target.polygon, first), -last)
    pieces = (clip_hemisphere(target.polygon, -first), strip,
              clip_hemisphere(target.polygon, last))
    assert polygon_area(strip) > 0.
    children = list(split_geometry(target, pieces, label="between-nodes-bank-hole"))
    children[1] = replace(children[1], parcel=replace(children[1].parcel,
                                                   material_id="fresh-strip"))
    changed = replace(origin, fragments=tuple(fragment for fragment in origin.fragments
        if fragment.fragment_id != target.fragment_id)+tuple(children),
        known_material_ids=origin.known_material_ids+("fresh-strip",))
    archive, registry = rebound_inventory(changed, archive, registry)
    with pytest.raises(UnresolvedFaultMotionSupport, match="missing 2D"):
        validate_buried_bank_support(origin, seed, changed, archive, poses, registry)


def test_balanced_same_owner_material_exchange_fails_the_per_material_2d_gate():
    origin, seed, archive, registry, poses = initial()
    first, second = tuple(fragment for fragment in origin.fragments if fragment.parcel.plate == 0)[:2]
    # Equal area, same owner and all four equal reservoirs do not authenticate
    # the material position.  Swap only the complete geometric footprints.
    changed = replace(origin, fragments=tuple(replace(fragment, polygon=second.polygon)
        if fragment.fragment_id == first.fragment_id else replace(fragment, polygon=first.polygon)
        if fragment.fragment_id == second.fragment_id else fragment for fragment in origin.fragments))
    archive, registry = rebound_inventory(changed, archive, registry)
    with pytest.raises(UnresolvedFaultMotionSupport, match="outside its 2D footprint"):
        validate_buried_bank_support(origin, seed, changed, archive, poses, registry)


@pytest.mark.parametrize("field,value", [("age_myr", 99.), ("material_fields", (("damage", .9),))])
def test_original_material_history_cannot_change_while_footprints_and_budgets_agree(field, value):
    origin, seed, archive, registry, poses = initial()
    target = origin.fragments[0]
    changed = replace(origin, fragments=(replace(target, parcel=replace(target.parcel,
        **{field: value})), *origin.fragments[1:]))
    archive, registry = rebound_inventory(changed, archive, registry)
    with pytest.raises(ValueError, match="history"):
        validate_buried_bank_support(origin, seed, changed, archive, poses, registry)


def test_acceptance_payloads_remain_immutable_when_buried_support_is_registered():
    one = accepted(count=1)
    two = accepted(count=2)
    old = {cohort.event_id: cohort for cohort in one[3].cohorts}
    new = {cohort.event_id: cohort for cohort in two[3].cohorts}
    assert one[4].records == two[4].records[:len(one[4].records)]
    for identity, cohort in old.items():
        assert new[identity].loss == cohort.loss
        assert new[identity].payload_sha256 == cohort.payload_sha256
        assert new[identity].parcel == cohort.parcel


def test_preexisting_cohorts_are_retained_but_never_invented_as_original_bank_mass():
    origin, _, current, archive, _, _, _ = accepted(count=1)
    registry = initial_buried_bank_registry(current, archive)
    assert not registry.records
    assert registry.initial_event_payloads == tuple(sorted(
        (cohort.event_id, cohort.payload_sha256) for cohort in archive.cohorts))
    assert registry.origin_surface_sha256 == surface_digest(current)
    assert archive.cohorts and current != origin


def test_exact_registry_roundtrip_and_future_registration_match_after_restart():
    origin, _, _, archive, registry, poses, _ = accepted(count=1)
    raw = buried_bank_registry_to_dict(registry)
    restored = buried_bank_registry_from_dict(json.loads(json.dumps(raw)), origin, archive)
    assert restored == registry
    assert json.dumps(buried_bank_registry_to_dict(restored), sort_keys=True) == json.dumps(raw, sort_keys=True)
    second = accepted(count=2)
    next_registry = advance_buried_bank_registry(origin, restored, archive, second[3], second[5],
                                                roundoff_steps=2)
    assert next_registry == second[4]
    assert poses


@pytest.mark.parametrize("mutation", ["missing_record", "duplicate_record", "payload", "polygon", "initial_id"])
def test_rehashed_registry_edits_do_not_supply_missing_or_unaccepted_material(mutation):
    origin, seed, current, archive, registry, poses, _ = accepted(count=1)
    if mutation == "missing_record":
        changed = replace(registry, records=registry.records[1:])
    elif mutation == "duplicate_record":
        changed = replace(registry, records=registry.records+(registry.records[0],))
    elif mutation == "initial_id":
        changed = replace(registry, initial_event_payloads=((registry.records[0].event_id,
            registry.records[0].loss_payload_sha256),), records=registry.records[1:])
    else:
        record = registry.records[0]
        if mutation == "payload":
            record = replace(record, loss_payload_sha256="0"*64)
        else:
            record = replace(record, original_polygon=tuple(reversed(record.original_polygon)))
        record = _sealed(record, "record_sha256")
        changed = replace(registry, records=(record, *registry.records[1:]))
    changed = _sealed(changed, "registry_sha256")
    with pytest.raises(ValueError):
        validate_buried_bank_support(origin, seed, current, archive, poses, changed, roundoff_steps=1)


@pytest.mark.parametrize("mutation", ["extra", "bool_matrix", "bool_polygon", "bad_matrix", "record_extra"])
def test_strict_registry_schema_and_physical_numeric_types(mutation):
    origin, _, _, archive, registry, _, _ = accepted(count=1)
    raw = buried_bank_registry_to_dict(registry)
    if mutation == "extra":
        raw["extra"] = 1
    elif mutation == "record_extra":
        raw["registry"]["records"][0]["extra"] = 1
    else:
        row = raw["registry"]["records"][0]
        if mutation == "bool_matrix":
            row["acceptance_rotation"][0][0] = True
        elif mutation == "bool_polygon":
            row["original_polygon"][0][0] = False
        else:
            row["acceptance_rotation"][0][0] = 2.
        row["record_sha256"] = _hash({key: value for key, value in row.items() if key != "record_sha256"})
        body = raw["registry"]
        body["registry_sha256"] = _hash({key: value for key, value in body.items() if key != "registry_sha256"})
    with pytest.raises(ValueError):
        buried_bank_registry_from_dict(raw, origin, archive)


@pytest.mark.parametrize("steps", [-1, True, 1.5])
def test_roundoff_count_is_explicit_chronological_integer(steps):
    origin, seed, archive, registry, poses = initial()
    with pytest.raises(ValueError, match="roundoff steps"):
        validate_buried_bank_support(origin, seed, origin, archive, poses, registry, roundoff_steps=steps)


@pytest.mark.parametrize("poses", [{0: np.eye(3)}, {0: np.eye(3), 1: np.eye(3), 2: np.eye(3)},
                                    ((1, tuple(map(tuple, np.eye(3)))), (0, tuple(map(tuple, np.eye(3)))) )])
def test_owner_pose_manifest_cannot_omit_extend_or_reorder_actual_owners(poses):
    origin, seed, archive, registry, _ = initial()
    with pytest.raises(ValueError, match="canonical original owners"):
        validate_buried_bank_support(origin, seed, origin, archive, poses, registry)


@pytest.mark.parametrize("poses", [{0: np.eye(3), "1": np.eye(3)},
                                   {0: np.eye(3), True: np.eye(3)}])
def test_owner_pose_manifest_rejects_numeric_aliases_and_mixed_key_types(poses):
    origin, seed, archive, registry, _ = initial()
    with pytest.raises(ValueError, match="explicit nonnegative integer owners"):
        validate_buried_bank_support(origin, seed, origin, archive, poses, registry)


def test_rehashed_malformed_initial_manifest_is_rejected_cleanly():
    origin, seed, archive, registry, poses = initial()
    changed = _sealed(replace(registry, initial_event_payloads=({"event": "fake"},)),
                      "registry_sha256")
    with pytest.raises(ValueError, match="manifest"):
        validate_buried_bank_support(origin, seed, origin, archive, poses, changed)


@pytest.mark.parametrize("poses", [{0: None, 1: np.eye(3)},
    ((0, None), (1, tuple(map(tuple, np.eye(3)))) )])
def test_explicit_missing_map_cannot_infer_identity(poses):
    origin, seed, archive, registry, _ = initial()
    with pytest.raises(ValueError, match="explicit SO3 matrices"):
        validate_buried_bank_support(origin, seed, origin, archive, poses, registry)


@pytest.mark.parametrize("bank,sign", [("sub", -1.), ("over", 1.)])
def test_enclosing_quad_contains_the_entire_analytic_constant_width_strip(bank, sign):
    origin, seed, _, _, _ = initial()
    trace = seed.fault_traces.traces[0]
    material = getattr(seed.bending_parameters[0], bank)
    quad, extent = _constant_width_quad(trace, material.width_m, origin.radius_km*1000., sign)
    start, horizontal, tangent, _ = _arc_frame(trace.start, trace.end)
    width = material.width_m/(origin.radius_km*1000.)
    for theta in np.linspace(0., extent, 23):
        for inward in np.linspace(0., width, 17):
            # Independent analytic swept surface; no production point helper.
            point = (start*np.cos(theta)+tangent*np.sin(theta))*np.cos(inward)
            point += sign*horizontal*np.sin(inward)
            assert all(_edge_distance(point, a, b, None) >= 0.
                       for a, b in zip(quad, np.roll(quad, -1, axis=0)))
