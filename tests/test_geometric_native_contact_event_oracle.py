"""Independent closed-form checks for the standalone contact-event oracle."""
from copy import deepcopy
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
import json

import pytest

from analysis.geometric_native_contact_event_validation.independent_event_oracle import (
    IndependentTrajectory, bisect_numeric_root, carrier_recrossing_evidence, compare_root_enclosure, decimal,
    sin_cos, trajectory_from_actual_probe,
)


ROOT = Path(__file__).resolve().parents[1]


def _increment(omega, duration):
    return dict(omega_rad_per_myr=list(omega), dt_myr=duration)


def _seed(reference, rotations=()):
    return dict(kind="seed", reference=list(reference), rotations=list(rotations))


def _trajectory(normal, point, donor=(0., 0., 0.), receiver=(0., 0., 1.),
                duration=.125, donor_history=(), point_history=()):
    return dict(donor_plane_before=dict(normal=list(normal), rotations=list(donor_history)),
        receiver_ray=_seed(point, point_history), receiver_pullback_history=[],
        donor_omega_rad_per_myr=list(donor), receiver_omega_rad_per_myr=list(receiver),
        dt_myr=duration)


@pytest.mark.parametrize("entry", [False, True])
def test_closed_form_midpoint_crossing_and_independent_bracket(entry):
    with localcontext() as context:
        context.prec = 160
        direction = -1. if entry else 1.
        raw = _trajectory((1, 0, 0), (0, 1, 0), receiver=(0., 0., direction),
            point_history=[_increment((0., 0., -direction), .0625)])
        trajectory = IndependentTrajectory(raw)
        root = bisect_numeric_root(trajectory, Decimal(".25"), Decimal(".75"), digits=120)
        assert abs(root-Decimal(".5")) < Decimal("1e-120")
        kind = "entry" if entry else "exit"
        receipt = compare_root_enclosure(trajectory, Decimal(".499"), Decimal(".501"),
            root, kind=kind, zero_bound="1e-115")
        assert receipt["crossing_direction_numerical"] == kind
        assert receipt["admission_authorized"] is False
        # The independently differentiated trajectory agrees with an analytic sine.
        for fraction in (Decimal(".1"), Decimal(".8")):
            gap, derivative, second = trajectory.derivatives(fraction)
            angle = Decimal(".0625")-Decimal(".125")*fraction
            sine, cosine = sin_cos(angle)
            assert abs(gap-Decimal(direction)*sine) < Decimal("1e-150")
            assert abs(derivative+Decimal(direction)*Decimal(".125")*cosine) < Decimal("1e-150")
            assert abs(second+Decimal(direction)*Decimal(".015625")*sine) < Decimal("1e-150")


def test_double_root_has_no_crossing_and_requires_more_than_sign_bisection():
    with localcontext() as context:
        context.prec = 180
        raw = _trajectory((1, 0, 0), (0, 0, 1), receiver=(1., 0., 1.),
            point_history=[_increment((-1., 0., -1.), .0625)])
        trajectory = IndependentTrajectory(raw)
        gap, derivative, second = trajectory.derivatives(Decimal(".5"))
        assert abs(gap) < Decimal("1e-170")
        assert abs(derivative) < Decimal("1e-170")
        assert abs(second-Decimal(".015625")) < Decimal("1e-170")
        assert trajectory.gap(Decimal(".25")) > 0
        assert trajectory.gap(Decimal(".75")) > 0
        with pytest.raises(ValueError, match="opposite numerical endpoint signs"):
            bisect_numeric_root(trajectory, Decimal(".25"), Decimal(".75"), digits=100)
        turning_point = bisect_numeric_root(trajectory, Decimal(".25"), Decimal(".75"),
            derivative_order=1, digits=140)
        assert abs(turning_point-Decimal(".5")) < Decimal("1e-140")


def test_two_crossings_in_one_declared_small_angle_interval():
    with localcontext() as context:
        context.prec = 180
        raw = _trajectory((1000, 0, 999), (1000, 0, -999),
            donor=(0., 0., 1.), receiver=(0., 0., -1.),
            donor_history=[_increment((0., 0., -1.), .0625)],
            point_history=[_increment((0., 0., 1.), .0625)])
        trajectory = IndependentTrajectory(raw)
        first = bisect_numeric_root(trajectory, Decimal(".1"), Decimal(".4"), digits=140)
        second = bisect_numeric_root(trajectory, Decimal(".6"), Decimal(".9"), digits=140)
        assert first < Decimal(".5") < second
        assert abs(first+second-1) < Decimal("1e-135")
        assert trajectory.derivatives(first)[1] > 0
        assert trajectory.derivatives(second)[1] < 0
        assert trajectory.gap(0) < 0 < trajectory.gap(Decimal(".5"))
        # n dot p = [10^6 cos(.125-.25u)-999^2] / sqrt(10^6+999^2).
        for fraction in (first, Decimal(".5"), second):
            _, cosine = sin_cos(Decimal(".125")-Decimal(".25")*fraction)
            expected = (Decimal(1_000_000)*cosine-Decimal(998_001))/Decimal(1_998_001).sqrt()
            assert abs(trajectory.gap(fraction)-expected) < Decimal("1e-165")


def test_near_endpoint_event_is_preserved_without_a_minimum_time():
    with localcontext() as context:
        context.prec = 220
        tiny_duration = 2.**-80
        raw = _trajectory((1, 0, 0), (0, 1, 0),
            point_history=[_increment((0., 0., -1.), tiny_duration)])
        trajectory = IndependentTrajectory(raw)
        root = bisect_numeric_root(trajectory, 0, Decimal("1e-20"), digits=170)
        assert abs(root-decimal(Fraction(1, 2**77))) < Decimal("1e-170")
        assert root > 0
        compare_root_enclosure(trajectory, 0, Decimal("1e-20"), root,
            kind="exit", zero_bound="1e-165")


def test_actual_probe_full_tiny_components_create_exit_and_return():
    probe = json.loads((ROOT/"analysis/geometric_native_underthrust_validation/force_control_release_02/force_candidate_refusal_probe.json").read_text())
    with localcontext() as context:
        context.prec = 500
        raw = trajectory_from_actual_probe(probe)
        actual = IndependentTrajectory(raw)
        assert max(abs(actual.gap(0)), abs(actual.gap(1))) < Decimal("1e-430")
        assert actual.derivatives(0)[1] < 0 < actual.derivatives(1)[1]
        assert actual.gap(Decimal(".5")) < 0
        assert Decimal("1e-40") < actual.derivatives(1)[1] < Decimal("1e-37")
        projected = deepcopy(raw)
        def project(value):
            if type(value) is dict:
                for key, item in value.items():
                    if key.endswith("omega_rad_per_myr"):
                        item[0] = item[1] = 0.
                    else:
                        project(item)
            elif type(value) is list:
                for item in value:
                    project(item)
        project(projected)
        flattened = IndependentTrajectory(projected)
        assert all(value == 0 for value in flattened.derivatives(Decimal(".5")))


def test_actual_positive_original_arc_band_exits_and_reenters_at_full_velocity():
    probe = json.loads((ROOT/"analysis/geometric_native_underthrust_validation/force_control_release_02/force_candidate_refusal_probe.json").read_text())
    with localcontext() as context:
        context.prec = 220
        evidence = carrier_recrossing_evidence(probe, digits=150)
        turning = Decimal(evidence["turning_point_fraction"])
        width = Decimal(evidence["positive_ratio_band_width"])
        assert Decimal("9.85e-12") < turning < Decimal("9.87e-12")
        assert Decimal("1.27e-54") < width < Decimal("1.28e-54")
        first, last = evidence["independently_bracketed_events"]
        assert first["kind"] == "exit" and last["kind"] == "entry"
        assert Decimal(first["fraction"]) < turning < Decimal(last["fraction"])
        assert Decimal(first["derivative_per_fraction"]) < 0 < Decimal(last["derivative_per_fraction"])
        assert max(abs(Decimal(event["numerical_gap"])) for event in (first, last)) < Decimal("1e-195")
        assert evidence["admission_authorized"] is False


def test_oracle_rejects_incomplete_vectors_and_wrong_certified_root_claims():
    with localcontext() as context:
        context.prec = 150
        raw = _trajectory((1, 0, 0), (0, 1, 0),
            point_history=[_increment((0., 0., -1.), .0625)])
        trajectory = IndependentTrajectory(raw)
        with pytest.raises(ValueError, match="omits"):
            compare_root_enclosure(trajectory, 0, Decimal(".4"), Decimal(".5"))
        with pytest.raises(ValueError, match="direction"):
            compare_root_enclosure(trajectory, Decimal(".49"), Decimal(".51"), Decimal(".5"),
                kind="entry", zero_bound="1e-140")
        raw["receiver_omega_rad_per_myr"] = [0., 1.]
        with pytest.raises(ValueError, match="all three"):
            IndependentTrajectory(raw)
