"""The complete actual donor sweep must retain every thin transverse motion."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import pytest

from tectonics.geometric_intrinsic_geometry import RotationIncrement
from tectonics.geometric_native_buried_support import native_pullback
from tectonics.geometric_native_donor_sweep import certify_native_donor_patch_sweep
from tectonics.geometric_native_polygons import (
    NativeRay, _ray_same, _same_plane, _side, native_intersection,
    native_polygon_from_dict, native_subtract,
)
from tectonics.geometric_native_pose import FractionInterval as I
from tectonics.geometric_native_swept_domain import (
    certify_native_swept_domain, validate_native_swept_domain,
    native_swept_domain_from_dict, native_swept_domain_to_dict,
)


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT/"analysis/geometric_native_underthrust_validation/force_control_release_02"


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@pytest.fixture(scope="module")
def actual():
    initial_bytes = (CONTROL/"initial.json").read_bytes()
    probe_bytes = (CONTROL/"force_candidate_refusal_probe.json").read_bytes()
    initial, probe = json.loads(initial_bytes), json.loads(probe_bytes)
    fragments = {row["fragment_id"]: row for row in initial["current_surface"]["fragments"]}
    assessment, = [row for row in probe["admission"]["actual_overlap_assessments"]
                    if (row["fragment_a"], row["fragment_b"]) == ("octant:2", "octant:6")]
    return dict(donor=native_polygon_from_dict(fragments["octant:6"]["geometry"]),
        receiver=native_polygon_from_dict(fragments["octant:2"]["geometry"]),
        overlap=native_polygon_from_dict(assessment["native_overlap"]),
        donor_omega=tuple(probe["omega"]["1"]), receiver_omega=tuple(probe["omega"]["0"]),
        duration=probe["admission"]["dt_myr"], initial_bytes=initial_bytes, probe_bytes=probe_bytes)


@pytest.fixture(scope="module")
def certificate(actual):
    return certify_native_swept_domain(actual["donor"], actual["receiver"],
        actual["donor_omega"], actual["receiver_omega"], actual["duration"])


def test_complete_swept_source_preserves_all_actual_owner_components(actual, certificate):
    assert certificate.donor_polygon == actual["donor"]
    assert certificate.receiver_polygon == actual["receiver"]
    assert certificate.donor_omega == actual["donor_omega"]
    assert certificate.receiver_omega == actual["receiver_omega"]
    assert all(value != 0 for omega in (certificate.donor_omega, certificate.receiver_omega)
               for value in omega)
    assert certificate.dt_myr == actual["duration"]
    assert (CONTROL/"initial.json").read_bytes() == actual["initial_bytes"]
    assert (CONTROL/"force_candidate_refusal_probe.json").read_bytes() == actual["probe_bytes"]


def test_whole_footprint_equals_actual_endpoint_loss_pullback(actual, certificate):
    assert certificate.endpoint_world_overlap == actual["overlap"]
    step = RotationIncrement(actual["donor_omega"], actual["duration"])
    expected = native_pullback(actual["overlap"], (step,))
    assert certificate.endpoint_donor_domain == expected
    assert certificate.endpoint_area_bounds_sr.lo > Fraction(2, 10**7)
    assert certificate.endpoint_area_bounds_sr == actual["overlap"].area_bounds()
    assert not native_subtract(certificate.terminal_native_cone, expected)
    assert not native_subtract(expected, certificate.terminal_native_cone)


def test_full_area_contains_old_positive_recrossing_witness_without_discarding_it(actual, certificate):
    witness = certify_native_donor_patch_sweep(actual["donor"], actual["receiver"],
        actual["donor_omega"], actual["receiver_omega"], actual["duration"])
    assert not native_subtract(witness.patch_geometry, certificate.endpoint_donor_domain)
    assert certificate.endpoint_area_bounds_sr.lo > witness.patch_area_bounds.hi*10**100


def test_complete_time_thresholds_prove_containment_without_monotone_material_occupancy(certificate):
    assert certificate.c_bounds.lo > 0
    assert certificate.c_second_derivative_bounds.hi < 0
    assert certificate.main_y_first_derivative_bounds.lo > 0
    assert certificate.a_over_u_bounds.lo > 0 > certificate.b_over_u_bounds.hi
    assert certificate.turning_derivative_bounds.lo > 0
    assert certificate.initial_turning_bounds.hi < 0 < certificate.terminal_turning_bounds.lo
    assert 0 < certificate.ratio_turning_fraction.lo < certificate.ratio_turning_fraction.hi < 1
    assert certificate.initial_minus_terminal_q_bounds.lo > 0
    assert certificate.initial_q_bounds.lo > certificate.terminal_q_bounds.hi > 0
    assert certificate.terminal_p_bounds.lo > 0
    assert all(value.lo > 0 for _, values in certificate.terminal_other_halfspace_vertex_bounds
               for value in values)


def test_terminal_native_cone_has_complete_exact_source_supports(certificate):
    cone = certificate.terminal_native_cone
    assert len(cone.vertices) == len(cone.supports) == 3
    assert {support.normal for support in cone.supports} == {
        (1, 0, 0), (0, 0, -1), (-1, 0, 0)}
    final_main = next(p for p in cone.supports if p.normal == (-1, 0, 0))
    assert len(final_main.rotations) == 2
    assert final_main.rotations[0].omega_rad_per_myr == certificate.receiver_omega
    assert final_main.rotations[1].omega_rad_per_myr == tuple(-v for v in certificate.donor_omega)
    assert all(step.dt_myr == certificate.dt_myr for step in final_main.rotations)


def test_initial_interface_is_preserved_separately_and_not_called_a_2d_loss(actual, certificate):
    arc = certificate.initial_boundary_arc
    assert _same_plane(arc.plane, next(p for p in actual["donor"].supports
                                     if p.normal == (1, 0, 0)))
    south = NativeRay("seed", (0, 0, -1))
    assert _ray_same(arc.start, south) or _ray_same(arc.end, south)
    # This exact time-zero endpoint lies outside the positive-time union.  It
    # must remain explicit rather than be snapped into the endpoint footprint.
    final_main = next(p for p in certificate.terminal_native_cone.supports
                      if p.normal == (-1, 0, 0))
    assert _side(south, final_main) == -1
    raw = native_swept_domain_to_dict(certificate)
    assert raw["initial_boundary"]["discarded"] is False
    assert raw["initial_boundary"]["separate_from_positive_time_swept_domain"] is True
    assert "zero_2D_measure" in raw["initial_boundary"]["measure_class"]


def test_actual_sweep_is_not_nested_but_early_departures_remain_in_the_full_endpoint(actual, certificate):
    def original_domain(fraction):
        dt = float(actual["duration"]*fraction)
        step = RotationIncrement(actual["donor_omega"], dt)
        world = native_intersection(actual["receiver"].rotated(actual["receiver_omega"], dt),
                                    actual["donor"].rotated(actual["donor_omega"], dt))
        return native_pullback(world, (step,))
    early, middle = original_domain(1e-13), original_domain(1e-12)
    departed = native_subtract(early, middle)
    assert len(departed) == 2 and all(g.area_bounds().lo > 0 for g in departed)
    assert not native_subtract(early, certificate.endpoint_donor_domain)
    assert all(not native_subtract(g, certificate.endpoint_donor_domain) for g in departed)


def test_swept_dto_replays_and_never_claims_uniform_three_events_or_force_permission(certificate):
    raw = native_swept_domain_to_dict(certificate)
    assert native_swept_domain_from_dict(deepcopy(raw)) == certificate
    assert encoded(native_swept_domain_to_dict(native_swept_domain_from_dict(raw))) == encoded(raw)
    assert raw["admission_authorized"] is False
    assert raw["global_inventory_updated"] is False
    assert raw["physical_archive_allocation_adopted"] is False
    assert raw["uniform_same_event_count_for_all_material"] is False
    assert "0<u<=1" in raw["domain_identity"]


@pytest.mark.parametrize("field", ["c_bounds", "c_second_derivative_bounds",
    "main_y_first_derivative_bounds", "a_over_u_bounds", "b_over_u_bounds",
    "turning_derivative_bounds", "initial_turning_bounds", "terminal_turning_bounds",
    "ratio_turning_fraction", "initial_minus_terminal_q_bounds", "terminal_p_bounds"])
def test_resealed_claimed_interval_cannot_replace_whole_time_containment(certificate, field):
    raw = deepcopy(native_swept_domain_to_dict(certificate))
    raw["certificate"][field] = {"lo": [0, 1], "hi": [0, 1]}
    body = {k: v for k, v in raw.items() if k != "payload_sha256"}
    raw["payload_sha256"] = hashlib.sha256(encoded(body)).hexdigest()
    with pytest.raises(ValueError):
        native_swept_domain_from_dict(raw)


@pytest.mark.parametrize("damage", ["projected_omega", "drop_terminal_plane", "lost_initial_arc",
    "area_zero", "adopt_force", "float_precision", "extra_whole_world_claim"])
def test_sealed_geometry_and_scope_cannot_be_overridden(certificate, damage):
    raw = deepcopy(native_swept_domain_to_dict(certificate))
    if damage == "projected_omega":
        raw["trajectory"]["donor_omega"][0] = 0.
    elif damage == "drop_terminal_plane":
        raw["terminal_native_cone"]["supports"].pop()
    elif damage == "lost_initial_arc":
        raw["initial_boundary"]["discarded"] = True
    elif damage == "area_zero":
        raw["certificate"]["endpoint_area_bounds_sr"] = {"lo": [0, 1], "hi": [0, 1]}
    elif damage == "adopt_force":
        raw["admission_authorized"] = True
    elif damage == "float_precision":
        raw["precision"] = float(raw["precision"])
    else:
        raw["whole_world_space_time_partition_proved"] = True
    body = {k: v for k, v in raw.items() if k != "payload_sha256"}
    raw["payload_sha256"] = hashlib.sha256(encoded(body)).hexdigest()
    with pytest.raises(ValueError):
        native_swept_domain_from_dict(raw)


@pytest.mark.parametrize("precision,order", [(True, 20), (256., 20), (63, 20), (1025, 20),
    (256, True), (256, 20.), (256, 7), (256, 65)])
def test_numeric_equality_cache_cannot_bypass_declared_budgets(actual, certificate, precision, order):
    assert validate_native_swept_domain(certificate) == certificate
    with pytest.raises(ValueError):
        certify_native_swept_domain(actual["donor"], actual["receiver"],
            actual["donor_omega"], actual["receiver_omega"], actual["duration"],
            precision=precision, order=order)


@pytest.mark.parametrize("change", ["zero", "projected_z", "reverse", "moved_source", "swap_sources"])
def test_outside_bounded_full_vector_cone_proof_refuses(actual, change):
    donor, receiver = actual["donor"], actual["receiver"]
    wd, wr = actual["donor_omega"], actual["receiver_omega"]
    if change == "zero":
        wd = wr = (0., 0., 0.)
    elif change == "projected_z":
        wd, wr = (0., 0., wd[2]), (0., 0., wr[2])
    elif change == "reverse":
        wd, wr = tuple(-v for v in wd), tuple(-v for v in wr)
    elif change == "moved_source":
        donor = donor.rotated(wd, actual["duration"])
    else:
        donor, receiver = receiver, donor
    with pytest.raises(ValueError):
        certify_native_swept_domain(donor, receiver, wd, wr, actual["duration"])


def test_forged_frozen_swept_object_must_replay(certificate):
    with pytest.raises(ValueError):
        validate_native_swept_domain(replace(certificate, c_second_derivative_bounds=I(0)))
    with pytest.raises(AttributeError):
        certificate.dt_myr = 1.
