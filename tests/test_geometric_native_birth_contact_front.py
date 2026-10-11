"""Moving native HOT entry has positive measure and a causal full-time mask."""
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import pytest

from tectonics.geometric_native_birth_contact_front import (
    native_birth_arc_frame, native_birth_arc_frame_to_dict, native_birth_arc_frame_from_dict,
    certify_native_hot_shared_arc_front, native_hot_shared_arc_front_to_dict,
    native_hot_shared_arc_front_from_dict, native_hot_shared_arc_front_time_bounds,
    certify_native_hot_incoming_rectangle, native_hot_incoming_rectangle_to_dict,
    certify_native_birth_radial_ordering,
)
from tectonics.geometric_native_pose import FractionInterval as I
from tectonics.thermal import SECONDS_PER_MYR
from tests.test_geometric_native_birth_contact_support import get_birth_support_case

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT/"analysis/geometric_native_birth_contact_validation/control_release_01/native"


@lru_cache(maxsize=1)
def get_native_birth_front_case():
    source = get_birth_support_case()
    raw_force = (CONTROL/"actual_mixed_network.json").read_bytes()
    report = json.loads((CONTROL/"report.json").read_bytes())
    assert hashlib.sha256(raw_force).hexdigest() == report["force_packet_sha256"]
    records = tuple((owner, tuple(vector)) for owner, vector in report["force_candidate_omega_rad_per_myr"])
    carrier = next(row for row in source["support"].interface_carriers
        if row.donor_component.material_kind == "hot_birth")
    duration = report["force_candidate_dt_s"]/SECONDS_PER_MYR
    front = certify_native_hot_shared_arc_front(source["support"], carrier.declaration.family_id,
        records, duration, force_source_sha256=report["force_packet_sha256"])
    return dict(source=source, report=report, records=records, carrier=carrier,
        duration=duration, front=front)


@pytest.fixture(scope="module")
def case():
    return get_native_birth_front_case()


def _replay(raw, case, **changes):
    arguments = dict(expected_support=case["source"]["support"],
        expected_family_id=case["carrier"].declaration.family_id,
        expected_omega_rad_per_myr=case["records"], expected_dt_myr=case["duration"],
        expected_force_source_sha256=case["report"]["force_packet_sha256"])
    arguments.update(changes)
    return native_hot_shared_arc_front_from_dict(raw, **arguments)


def _d(value):
    if type(value) is F:
        return Decimal(value.numerator)/Decimal(value.denominator)
    return Decimal.from_float(value)


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _rotate(vector, axis):
    result, term = list(vector), tuple(vector)
    for index in range(1, 80):
        term = tuple(value/Decimal(index) for value in _cross(axis, term))
        result = [a+b for a, b in zip(result, term)]
        if max(map(abs, term)) < Decimal("1e-330"):
            return tuple(result)
    raise AssertionError("Independent Decimal series did not converge")


def _atan(value):
    result, term = value, value
    for index in range(1, 20):
        term *= -value*value
        following = term/Decimal(2*index+1)
        result += following
        if abs(following) < Decimal("1e-330"):
            return result
    raise AssertionError("Independent tiny-angle atan did not converge")


def _inside(value, bounds):
    assert _d(bounds.lo) <= value <= _d(bounds.hi)


def test_exact_hot_arc_frame_and_positive_tiny_measure_are_retained(case):
    frame = native_birth_arc_frame(case["carrier"])
    assert frame.nominal_start == (F(0), F(0), F(-1))
    assert frame.nominal_tangent == (F(0), F(1), F(0))
    assert frame.nominal_horizontal == (F(-1), F(0), F(0))
    assert frame.native_angle_bounds_radians.lo > 0
    raw = native_birth_arc_frame_to_dict(frame)
    assert raw["vectors"]["start"]["nominal_decimal"] == ["0", "0", "-1"]
    assert not raw["native_material_area_assigned_to_point"]
    assert raw["measure_density_m2_per_radian"] == [
        frame.measure_density_m2_per_radian.numerator, frame.measure_density_m2_per_radian.denominator]


def test_advected_cold_arc_has_outward_native_frame_uncertainties(case):
    carrier = next(row for row in case["source"]["support"].interface_carriers
        if row.donor_component.material_kind == "original_live")
    frame = native_birth_arc_frame(carrier)
    assert any(value.width > 0 for value in frame.horizontal_unit_bounds)
    assert frame.nominal_horizontal != (F(-1), F(0), F(0))
    raw = native_birth_arc_frame_to_dict(frame)
    assert raw["receiver_inward_plane"]["rotations"]
    assert native_birth_arc_frame_to_dict(native_birth_arc_frame_from_dict(raw,
        expected_carrier=carrier)) == raw


def test_actual_mixed_flux_has_unique_moving_root_in_the_whole_arc(case):
    front = case["front"]
    assert F(54, 10**23) < front.initial_flux_root_angle.lo < F(55, 10**23)
    assert front.full_time_flux_root_angle.lo > 0
    assert front.full_time_flux_root_angle.hi < front.frame.native_angle_bounds_radians.lo
    assert front.flux_angle_derivative_bounds_per_myr.lo > 6
    assert front.flux_root_time_derivative_bounds.lo > 0
    assert front.positive_incoming_angle_span_bounds.lo > F(4, 10**21)
    assert front.full_time_incoming_carrier_measure_m2.lo > 0


def test_complete_fixed_material_clock_is_concave_and_all_other_receivers_excluded(case):
    front = case["front"]
    assert front.first_entry_second_derivative_bounds.hi < 0
    assert front.donor_y_support_bounds.lo > 0
    assert len(front.other_receiver_exclusions) == 5
    assert all(bounds.hi < 0 for _, _, _, bounds in front.other_receiver_exclusions)
    expected = {row.component_id for row in front.support.source_epoch.components
        if row.owner == 0 and row.component_id != front.frame.carrier.receiver_component.component_id}
    assert {identity for identity, _, _, _ in front.other_receiver_exclusions} == expected


def test_independent_decimal_motion_stays_inside_time_root_enclosures(case):
    with localcontext() as context:
        context.prec = 360
        vectors = {owner: tuple(_d(value) for value in vector) for owner, vector in case["records"]}
        relative = tuple(a-b for a, b in zip(vectors[1], vectors[0]))
        duration = _d(F.from_float(case["duration"]))
        for fraction in (F(0), F(1, 3), F(1)):
            axis = tuple(-value*duration*_d(fraction) for value in vectors[0])
            B = _rotate(relative, axis)
            actual = _atan(-B[1]/B[2])
            bounds = native_hot_shared_arc_front_time_bounds(case["front"], I(fraction))
            _inside(actual, bounds["flux_root_angle"])
            _inside(actual, case["front"].full_time_flux_root_angle)
        final = native_hot_shared_arc_front_time_bounds(case["front"], I(1))["flux_root_angle"]
        assert final.lo > case["front"].initial_flux_root_angle.hi


def test_independent_material_ray_is_outside_at_every_checked_pre_entry_time(case):
    with localcontext() as context:
        context.prec = 360
        omega = {owner: tuple(_d(value) for value in vector) for owner, vector in case["records"]}
        duration = _d(F.from_float(case["duration"]))
        theta, entry = Decimal("2e-21"), Decimal("0.75")
        sine = theta-theta**3/6+theta**5/120-theta**7/5040
        cosine = 1-theta**2/2+theta**4/24-theta**6/720
        carrier = (Decimal(0), sine, -cosine)
        q = _rotate(_rotate(carrier, tuple(value*duration*entry for value in omega[0])),
            tuple(-value*duration*entry for value in omega[1]))
        assert q[0] > 0 and q[1] > 0
        for previous in (Decimal(0), Decimal("0.125"), Decimal("0.5")):
            local = _rotate(_rotate(q, tuple(value*duration*previous for value in omega[1])),
                tuple(-value*duration*previous for value in omega[0]))
            assert -local[0] < 0
        local_entry = _rotate(_rotate(q, tuple(value*duration*entry for value in omega[1])),
            tuple(-value*duration*entry for value in omega[0]))
        assert abs(local_entry[0]) < Decimal("1e-170")


def test_positive_rectangle_has_whole_time_first_entry_and_distinct_C_measure(case):
    rectangle = certify_native_hot_incoming_rectangle(case["front"],
        I(F(1, 10**21), F(3, 10**21)), I(0, 1))
    assert rectangle.flux_bounds_per_myr.lo > 0
    assert rectangle.positive_interface_carrier_measure_m2 == case["front"].frame.measure_density_m2_per_radian*F(2, 10**21)
    raw = native_hot_incoming_rectangle_to_dict(rectangle)
    assert raw["earliest_owner_entry_proved"] and raw["whole_source_donor_and_receiver_masks_bound"]
    assert not raw["interface_measure_used_as_archive_area"]
    assert not raw["normal_activation_or_down_dip_permission_inferred"]


@pytest.mark.parametrize("angle", (I(0, F(1, 10**22)), I(F(5, 10**21), F(6, 10**21))))
def test_outgoing_or_support_crossing_rectangles_refuse_whole_time_admission(case, angle):
    with pytest.raises(ValueError, match="causal moving incoming mask"):
        certify_native_hot_incoming_rectangle(case["front"], angle, I(0, 1))


def test_front_roundtrip_retains_actual_force_source_and_no_mechanical_adoption(case):
    raw = native_hot_shared_arc_front_to_dict(case["front"])
    assert native_hot_shared_arc_front_to_dict(_replay(deepcopy(raw), case)) == raw
    assert not raw["root_box_midpoint_used_as_event"]
    assert not raw["normal_constitutive_branch_selected_by_geometry"]
    assert not raw["radial_ordering_and_three_dimensional_collision_certified"]
    assert not raw["complete_all_pairs_temporal_coverage"]
    assert "force_admission_authorized" not in raw


@pytest.mark.parametrize("mutation", ("root", "tuple", "bool", "nonfinite"))
def test_resealed_or_nonliteral_front_dto_cannot_change_source_proof(case, mutation):
    raw = native_hot_shared_arc_front_to_dict(case["front"])
    if mutation == "root":
        raw["initial_flux_root_angle"]["lo"] = [0, 1]
        raw["front_sha256"] = hashlib.sha256(json.dumps({key: value for key, value in raw.items()
            if key != "front_sha256"}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    elif mutation == "tuple":
        raw["exact_dt_myr"] = tuple(raw["exact_dt_myr"])
    elif mutation == "bool":
        raw["omega_rad_per_myr"][0][0] = False
    else:
        raw["omega_rad_per_myr"][0][1][0] = float("nan")
    with pytest.raises(ValueError):
        _replay(raw, case)


@pytest.mark.parametrize("change", ("force_source", "clock", "omega"))
def test_saved_front_is_bound_to_independent_actual_force_and_clock(case, change):
    arguments = {}
    if change == "force_source":
        arguments["expected_force_source_sha256"] = "0"*64
    elif change == "clock":
        arguments["expected_dt_myr"] = F.from_float(case["duration"])+F(1, 10**40)
    else:
        vectors = list(case["records"])
        owner, vector = vectors[0]
        vectors[0] = (owner, (vector[0], vector[1]*1.001, vector[2]))
        arguments["expected_omega_rad_per_myr"] = tuple(vectors)
    with pytest.raises(ValueError):
        _replay(native_hot_shared_arc_front_to_dict(case["front"]), case, **arguments)


def test_typed_root_certificate_cannot_reuse_integrity_after_mutation(case):
    changed = replace(case["front"], full_time_flux_root_angle=I(0))
    with pytest.raises(ValueError, match="full material-source and clock replay"):
        native_hot_shared_arc_front_to_dict(changed)


def test_absolute_layer_ordering_hook_requires_actual_whole_footprints(case):
    arguments = dict(donor_component_id=case["carrier"].donor_component.component_id,
        receiver_component_id=case["carrier"].receiver_component.component_id,
        donor_surface_displacement_bounds_m=I(-3000, -2900),
        receiver_surface_displacement_bounds_m=I(0, 1), receiver_solid_layer_thickness_bounds_m=I(2000),
        source_force_sha256=case["report"]["force_packet_sha256"],
        field_scope="entire_material_footprints_and_physical_clock")
    proof = certify_native_birth_radial_ordering(**arguments)
    assert proof["strict_vertical_clearance_m"]["lo"] == [900, 1]
    assert not proof["full_three_dimensional_trajectory_collision_certified"]
    assert not proof["free_contact_gap_substituted_for_layer_thickness"]
    arguments["donor_surface_displacement_bounds_m"] = I(F(-1, 100), F(0))
    with pytest.raises(ValueError, match="receiver layer bottom"):
        certify_native_birth_radial_ordering(**arguments)
    arguments["field_scope"] = "single_point_gap_free"
    with pytest.raises(ValueError, match="whole footprints"):
        certify_native_birth_radial_ordering(**arguments)
