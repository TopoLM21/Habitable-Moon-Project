"""Correlated full-time support signs and genuine current pair exclusions."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

import pytest

from tectonics.geometric_native_birth_pair_exclusion import (
    UnresolvedNativePairExclusion, certify_native_support_ray_nonpositive,
    certify_native_birth_pair_exclusion, native_birth_pair_exclusion_to_dict,
    native_birth_pair_exclusion_from_dict, validate_native_birth_pair_exclusion,
)
from tectonics.geometric_native_polygons import NativePlane, NativeRay
from tests.test_geometric_native_implicit_epoch_admission import get_actual_newborn_geometry


ROOT = Path(__file__).resolve().parents[1]
FIXED_INVENTORY = ROOT / "analysis/geometric_native_birth_contact_validation/control_release_01/native/mixed_pair_inventory.json"
EXPECTED_INVENTORY_RAW_SHA = "1eacbabd919632c727fb470ab0dcb10fc55a65fc74609c8b23c6e51dad735f1c"
PROVENANCE = "Independent complete support exclusion of the immutable actual mixed source"
ZERO = (0., 0., 0.)


def test_birth_incidence_divides_the_actual_clock_before_sign_enclosure():
    ray, plane = NativeRay("seed", (0, 0, -1)), NativePlane((-1, 0, 0))
    proof = certify_native_support_ray_nonpositive(ray, plane,
        (0., -1e-100, 6.), ZERO, 1e-8)
    assert proof.exact_initial_incidence and proof.divided_by_birth_clock
    assert proof.scalar_coefficients[0].lo == proof.scalar_coefficients[0].hi == 0
    assert proof.scalar_remainder_bound > 0
    assert all(bounds.hi < 0 for _, bounds in proof.cell_sign_bounds)


def test_arbitrarily_tiny_positive_entry_is_not_excluded():
    with pytest.raises(UnresolvedNativePairExclusion):
        certify_native_support_ray_nonpositive(NativeRay("seed", (0, 0, -1)),
            NativePlane((-1, 0, 0)), (0., 1e-100, 6.), ZERO, 1e-8)


def test_common_full_rotation_preserves_exact_zero_incidence():
    proof = certify_native_support_ray_nonpositive(NativeRay("seed", (0, 0, -1)),
        NativePlane((-1, 0, 0)), (1., 2., 3.), (1., 2., 3.), 1e-8,
        time_cells=4)
    assert proof.scalar_remainder_bound == 0
    assert all(bounds.lo == bounds.hi == 0 for _, bounds in proof.cell_sign_bounds)


@pytest.mark.parametrize("changes", [dict(time_cells=True), dict(precision=True),
    dict(order=True), dict(dt_myr=1), dict(point_owner_omega=(0, 0., 0.)),
    dict(plane_owner_omega=(False, 0., 0.))])
def test_literal_types_are_checked_before_any_cached_numeric_alias(changes):
    arguments = dict(point_owner_omega=ZERO, plane_owner_omega=ZERO,
        dt_myr=1e-8, precision=256, order=28, time_cells=1)
    certify_native_support_ray_nonpositive(NativeRay("seed", (0, 0, -1)),
        NativePlane((-1, 0, 0)), **arguments)
    arguments.update(changes)
    with pytest.raises(ValueError):
        certify_native_support_ray_nonpositive(NativeRay("seed", (0, 0, -1)),
            NativePlane((-1, 0, 0)), **arguments)


@pytest.fixture(scope="module")
def actual_source():
    data = FIXED_INVENTORY.read_bytes()
    assert hashlib.sha256(data).hexdigest() == EXPECTED_INVENTORY_RAW_SHA
    inventory = json.loads(data)["inventory"]
    epoch = get_actual_newborn_geometry().source_epoch
    assert epoch.epoch_sha256 == inventory["source_epoch_sha256"]
    motions = tuple((row[0], tuple(row[1])) for row in inventory["omega_rad_per_myr"])
    identities = {(row.material_kind, row.original_parent_id): row.component_id
        for row in epoch.components}
    return dict(epoch=epoch, motions=motions, dt=inventory["dt_myr"], identities=identities)


def _certify(case, donor_kind, donor_parent, receiver_kind, receiver_parent):
    return certify_native_birth_pair_exclusion(case["epoch"],
        case["identities"][(donor_kind, "octant:"+str(donor_parent))],
        case["identities"][(receiver_kind, "octant:"+str(receiver_parent))],
        case["motions"], case["dt"], precision=256, order=28, time_cells=1,
        provenance=PROVENANCE)


@pytest.mark.parametrize("pair", [
    ("hot_birth", 6, "hot_birth", 0), ("hot_birth", 6, "original_live", 3),
    ("hot_birth", 5, "original_live", 0),
    ("hot_birth", 4, "hot_birth", 2), ("hot_birth", 4, "hot_birth", 0),
    ("hot_birth", 4, "original_live", 0), ("hot_birth", 4, "original_live", 1),
    ("hot_birth", 4, "original_live", 3),
    ("original_live", 5, "original_live", 0),
    ("original_live", 6, "original_live", 0),
    ("original_live", 6, "original_live", 1),
    ("original_live", 4, "original_live", 0),
    ("original_live", 4, "original_live", 3),
])
def test_thirteen_actual_upper_only_pairs_have_complete_zero_area_proofs(actual_source, pair):
    proof = _certify(actual_source, *pair)
    assert proof.piece_proofs
    assert all(bound.hi <= 0 for piece in proof.piece_proofs for vertex in piece.vertex_proofs
        for _, bound in vertex.cell_sign_bounds)
    raw = native_birth_pair_exclusion_to_dict(proof)
    assert raw["all_outer_union_piece_pairs_excluded"]
    assert not raw["positive_area_tolerance_used"]
    assert not raw["force_admission_authorized"]
    assert raw["temporal_material_area_bounds_sr"] == dict(lo=[0, 1], hi=[0, 1])


def test_joint_receiver_support_closes_the_actual_diagonal_cold_pair(actual_source):
    proof = _certify(actual_source, "original_live", 6, "original_live", 1)
    piece = proof.piece_proofs[0]
    assert piece.separating_bank == "receiver"
    assert len(piece.separating_support_weights) == 2
    assert all(weight > 0 for _, weight in piece.separating_support_weights)
    assert all(bound.hi < 0 for vertex in piece.vertex_proofs
        for _, bound in vertex.cell_sign_bounds)


@pytest.mark.parametrize("pair", [("original_live", 6, "hot_birth", 2),
    ("original_live", 7, "original_live", 2)])
def test_actual_positive_born_contact_cannot_be_excluded(actual_source, pair):
    with pytest.raises(UnresolvedNativePairExclusion):
        _certify(actual_source, *pair)


def _replay(case, proof, raw):
    return native_birth_pair_exclusion_from_dict(raw, expected_epoch=case["epoch"],
        expected_donor_component_id=proof.donor_component_id,
        expected_receiver_component_id=proof.receiver_component_id,
        expected_omega_rad_per_myr=case["motions"], expected_dt_myr=case["dt"],
        expected_precision=256, expected_order=28, expected_time_cells=1,
        expected_provenance=PROVENANCE)


def _reseal(raw):
    raw.pop("proof_sha256")
    raw["proof_sha256"] = hashlib.sha256(json.dumps(raw, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def test_restart_binds_full_external_source_and_replays_signs(actual_source):
    proof = _certify(actual_source, "hot_birth", 6, "hot_birth", 0)
    raw = native_birth_pair_exclusion_to_dict(proof)
    assert _replay(actual_source, proof, raw) == proof
    forged = deepcopy(raw)
    forged["pieces"][0]["vertex_proofs"][0]["cell_sign_bounds"][0]["sign_bounds"]["hi"] = [1, 1]
    _reseal(forged)
    with pytest.raises(ValueError, match="independently supplied"):
        _replay(actual_source, proof, forged)


def test_restart_rejects_resealed_boolean_integer_alias(actual_source):
    proof = _certify(actual_source, "hot_birth", 6, "hot_birth", 0)
    forged = native_birth_pair_exclusion_to_dict(proof)
    forged["pieces"][0]["vertex_proofs"][0]["exact_initial_incidence"] = int(
        forged["pieces"][0]["vertex_proofs"][0]["exact_initial_incidence"])
    _reseal(forged)
    with pytest.raises(ValueError, match="independently supplied"):
        _replay(actual_source, proof, forged)


def test_typed_certificate_cannot_change_its_euler_source_after_sealing(actual_source):
    proof = _certify(actual_source, "hot_birth", 6, "hot_birth", 0)
    changed = replace(proof, omega_rad_per_myr=((0, ZERO), (1, ZERO)))
    with pytest.raises((ValueError, UnresolvedNativePairExclusion)):
        validate_native_birth_pair_exclusion(changed)
