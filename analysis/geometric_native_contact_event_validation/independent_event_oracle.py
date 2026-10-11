"""Independent numerical evidence for native moving-plane contact events.

This file deliberately imports no tectonics module.  Saved primitive planes,
rays, and complete rotation journals are evaluated with Decimal Taylor series
and direct Rodrigues rotations.  Numerical roots can audit a certified event
enclosure; they cannot certify root completeness or authorize material burial.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, getcontext, localcontext
from fractions import Fraction
import hashlib
import json
from math import comb
from pathlib import Path


PRECISION = 500
ZERO = Decimal(0)
ONE = Decimal(1)


def decimal(value):
    """Preserve the exact value of an original binary64 or rational scalar."""
    if type(value) is float:
        result = Decimal.from_float(value)
    elif type(value) is int:
        result = Decimal(value)
    elif type(value) is Decimal:
        result = value
    elif type(value) is Fraction:
        result = Decimal(value.numerator) / Decimal(value.denominator)
    elif type(value) is str:
        result = Decimal(value)
    elif (type(value) is dict and set(value) == {"numerator", "denominator"}
          and type(value["numerator"]) is int and type(value["denominator"]) is int
          and value["denominator"] > 0):
        result = Decimal(value["numerator"]) / Decimal(value["denominator"])
    else:
        raise ValueError("Oracle scalar must preserve a finite explicit number")
    if not result.is_finite():
        raise ValueError("Oracle scalar must be finite")
    return result


def vector(values):
    if type(values) not in (list, tuple) or len(values) != 3:
        raise ValueError("Oracle requires all three Cartesian components")
    return tuple(decimal(value) for value in values)


def dot(a, b):
    return sum((x*y for x, y in zip(a, b)), ZERO)


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0])


def normalize(value):
    norm = dot(value, value).sqrt()
    if not norm:
        raise ValueError("Oracle cannot normalize a zero primitive")
    return tuple(component/norm for component in value)


def sin_cos(angle):
    """Taylor sin/cos; range reduction is unnecessary for declared small steps."""
    angle = decimal(angle)
    if abs(angle) > 1:
        raise ValueError("Oracle is scoped to angles no greater than one radian")
    threshold = Decimal(10) ** -(getcontext().prec+8)
    square = angle*angle
    sine = sine_term = angle
    cosine = cosine_term = ONE
    index = 0
    while True:
        sine_term *= -square / Decimal((2*index+2)*(2*index+3))
        cosine_term *= -square / Decimal((2*index+1)*(2*index+2))
        sine += sine_term
        cosine += cosine_term
        index += 1
        if max(abs(sine_term), abs(cosine_term)) < threshold:
            return sine, cosine


def rotate(value, omega, duration):
    speed = dot(omega, omega).sqrt()
    duration = decimal(duration)
    if not speed or not duration:
        return value
    axis = tuple(component/speed for component in omega)
    sine, cosine = sin_cos(speed*duration)
    crossed, projection = cross(axis, value), dot(axis, value)
    return tuple(component*cosine + turn*sine + direction*projection*(ONE-cosine)
                 for component, turn, direction in zip(value, crossed, axis))


def history(value, journal, *, inverse=False):
    if type(journal) not in (list, tuple):
        raise ValueError("Oracle needs an explicit complete rotation journal")
    for increment in reversed(journal) if inverse else journal:
        if (type(increment) is not dict
                or set(increment) != {"omega_rad_per_myr", "dt_myr"}):
            raise ValueError("Oracle rotation increment must preserve its complete schema")
        omega, duration = vector(increment["omega_rad_per_myr"]), decimal(increment["dt_myr"])
        if duration <= 0:
            raise ValueError("Oracle history requires positive original durations")
        value = rotate(value, omega, -duration if inverse else duration)
    return value


def plane(raw):
    return history(vector(raw["normal"]), raw["rotations"])


def ray(raw):
    if raw["kind"] == "seed":
        return normalize(history(vector(raw["reference"]), raw["rotations"]))
    if raw["kind"] != "cross" or len(raw["planes"]) != 2:
        raise ValueError("Oracle requires a primitive seed or two-plane cross")
    return normalize(tuple(decimal(raw["sign"])*value
                           for value in cross(*(plane(item) for item in raw["planes"]))))


class IndependentTrajectory:
    """g(u)=R_d(u DT)n_before dot R_r(u DT)p_before, 0 <= u <= 1."""

    def __init__(self, raw):
        self.raw = raw
        self.duration = decimal(raw["dt_myr"])
        if self.duration <= 0:
            raise ValueError("Oracle trajectory duration must be positive")
        self.donor_omega = vector(raw["donor_omega_rad_per_myr"])
        self.receiver_omega = vector(raw["receiver_omega_rad_per_myr"])
        self.donor_rate = tuple(value*self.duration for value in self.donor_omega)
        self.receiver_rate = tuple(value*self.duration for value in self.receiver_omega)
        if max(dot(value, value).sqrt() for value in (self.donor_rate, self.receiver_rate)) > 1:
            raise ValueError("Oracle trajectory exceeds its declared small-angle domain")
        self.normal_before = plane(raw["donor_plane_before"])
        self.point_before = history(ray(raw["receiver_ray"]),
            raw.get("receiver_pullback_history", []), inverse=True)

    def derivatives(self, fraction, order=2):
        if type(order) is not int or not 0 <= order <= 8:
            raise ValueError("Oracle derivative order must be an integer from zero to eight")
        instant = decimal(fraction)
        if not ZERO <= instant <= ONE:
            raise ValueError("Oracle event fraction must lie in [0,1]")
        normal = rotate(self.normal_before, self.donor_rate, instant)
        point = rotate(self.point_before, self.receiver_rate, instant)
        normal_terms, point_terms = [normal], [point]
        for _ in range(order):
            normal_terms.append(cross(self.donor_rate, normal_terms[-1]))
            point_terms.append(cross(self.receiver_rate, point_terms[-1]))
        return tuple(sum((Decimal(comb(level, index))*dot(normal_terms[index],
                     point_terms[level-index]) for index in range(level+1)), ZERO)
                     for level in range(order+1))

    def gap(self, fraction):
        return self.derivatives(fraction, 0)[0]


def bisect_scalar_root(evaluate, low, high, *, digits=430):
    """An independent sign-bracketed numerical root, never a completeness proof."""
    low, high = decimal(low), decimal(high)
    if not ZERO <= low < high <= ONE:
        raise ValueError("Oracle bisection requires an ordered fraction bracket")
    first, last = evaluate(low), evaluate(high)
    if not first:
        return low
    if not last:
        return high
    if (first > 0) == (last > 0):
        raise ValueError("Oracle bisection requires opposite numerical endpoint signs")
    threshold = Decimal(10) ** -digits
    while high-low > threshold:
        middle = (low+high)/2
        value = evaluate(middle)
        if not value:
            return middle
        if (value > 0) == (first > 0):
            low, first = middle, value
        else:
            high = middle
    return (low+high)/2


def bisect_numeric_root(trajectory, low, high, *, derivative_order=0, digits=430):
    if derivative_order not in (0, 1):
        raise ValueError("Oracle bisection requires gap or first derivative")
    evaluate = lambda fraction: trajectory.derivatives(fraction, derivative_order)[derivative_order]
    return bisect_scalar_root(evaluate, low, high, digits=digits)


def compare_root_enclosure(trajectory, low, high, root, *, kind=None, zero_bound="1e-420"):
    """Report whether a production enclosure contains independently found time."""
    low, high, root = decimal(low), decimal(high), decimal(root)
    if not ZERO <= low <= high <= ONE or not low <= root <= high:
        raise ValueError("Certified time enclosure omits the independent numerical root")
    gap, first, second = trajectory.derivatives(root)
    if abs(gap) >= decimal(zero_bound):
        raise ValueError("Independent event root has an excessive numerical gap residual")
    direction = "entry" if first > 0 else "exit" if first < 0 else "stationary"
    if kind in ("entry", "exit") and direction != kind:
        raise ValueError("Certified crossing direction differs from independent full-vector derivative")
    return dict(low_fraction=str(low), high_fraction=str(high),
        independently_computed_fraction=str(root), numerical_gap=str(gap),
        derivative_per_fraction=str(first), second_derivative_per_fraction=str(second),
        crossing_direction_numerical=direction, interval_contains_independent_root=True,
        admission_authorized=False)


def trajectory_from_actual_probe(probe, endpoint="end"):
    candidates = [row for row in probe["admission"]["actual_overlap_assessments"]
                  if (row["fragment_a"], row["fragment_b"]) == ("octant:2", "octant:6")]
    if len(candidates) != 1 or len(candidates[0]["receiver_arcs"]) != 1:
        raise ValueError("Oracle needs the actual octant:2/octant:6 carrier")
    receiver = candidates[0]["receiver_arcs"][0]
    if len(receiver["supports"]) != 1 or endpoint not in ("start", "end"):
        raise ValueError("Oracle needs one actual original support and a named endpoint")
    arc = receiver["supports"][0]["native_arc"]
    over, sub = str(receiver["overriding_plate"]), str(receiver["subducting_plate"])
    journal = arc["plane"]["rotations"]
    duration = probe["admission"]["dt_myr"]
    if (len(journal) != 1 or journal[0]["omega_rad_per_myr"] != probe["omega"][over]
            or journal[0]["dt_myr"] != duration):
        raise ValueError("Oracle cannot replace any actual owner component or duration")
    return dict(donor_plane_before=dict(normal=[1, 0, 0], rotations=[]),
        receiver_ray=arc[endpoint], receiver_pullback_history=journal,
        donor_omega_rad_per_myr=probe["omega"][sub],
        receiver_omega_rad_per_myr=probe["omega"][over], dt_myr=duration)


def _encoded(raw):
    return json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def carrier_recrossing_evidence(probe, *, digits=430):
    """Independently find an exit/re-entry point in the actual original arc.

    For the minor arc's positive homogeneous mixture r*p_start+p_end,
    g(u,r)=r*a(u)+b(u), with a and b computed from complete owner trajectories.
    The contact curve q=-b/a has derivative -W/a², W=a*b'-a'*b.
    This numerical research fixture finds roots in separately supplied sign
    brackets; no sampled sign is used as a material admission decision.
    """
    first = IndependentTrajectory(trajectory_from_actual_probe(probe, "start"))
    last = IndependentTrajectory(trajectory_from_actual_probe(probe, "end"))

    def values(fraction):
        a, a_prime = first.derivatives(fraction, 1)
        b, b_prime = last.derivatives(fraction, 1)
        return a, b, a_prime, b_prime

    def turning_function(fraction):
        a, b, a_prime, b_prime = values(fraction)
        return a*b_prime-a_prime*b

    turning = bisect_scalar_root(turning_function, Decimal("1e-13"), Decimal("1e-10"), digits=digits)
    _, _, a_prime, b_prime = values(ZERO)
    initial_ratio = -b_prime/a_prime
    a, b, _, _ = values(turning)
    maximum_ratio = -b/a
    positive_band_width = maximum_ratio-initial_ratio
    if positive_band_width <= 0:
        raise ValueError("Independent full-vector carrier has no positive recrossing band")
    mixture_ratio = (maximum_ratio+initial_ratio)/2

    def mixed_gap(fraction):
        a, b, _, _ = values(fraction)
        return mixture_ratio*a+b

    exit_time = bisect_scalar_root(mixed_gap, turning/16, turning, digits=digits)
    entry_time = bisect_scalar_root(mixed_gap, turning, 4*turning, digits=digits)
    events = []
    for kind, time in (("exit", exit_time), ("entry", entry_time)):
        a, b, a_prime, b_prime = values(time)
        derivative = mixture_ratio*a_prime+b_prime
        if (kind == "exit" and derivative >= 0) or (kind == "entry" and derivative <= 0):
            raise ValueError("Independent carrier gap root has an incorrect crossing direction")
        events.append(dict(kind=kind, fraction=str(time), time_myr=str(time*first.duration),
            numerical_gap=str(mixture_ratio*a+b), derivative_per_fraction=str(derivative)))
    return dict(parameterization="normalized positive mixture r*p_start+p_end of actual original carrier endpoints",
        exact_original_start_trajectory=first.raw, exact_original_end_trajectory=last.raw,
        contact_ratio_definition="q=-b/a; a=g(start), b=g(end), W=a*b'-a'*b",
        turning_point_fraction=str(turning), turning_point_time_myr=str(turning*first.duration),
        turning_function_numerical_residual=str(turning_function(turning)),
        initial_ratio_limit=str(initial_ratio), ratio_at_numerical_turning_point=str(maximum_ratio),
        positive_ratio_band_width=str(positive_band_width), selected_positive_mixture_ratio=str(mixture_ratio),
        independently_bracketed_events=events,
        signs_at_independent_bracket_points=[dict(fraction=str(time), gap=str(mixed_gap(time)))
            for time in (turning/16, turning, 4*turning)],
        conclusion="This actual original carrier point numerically exits and re-enters within the interval.",
        admission_authorized=False,
        limitation="Independently bracketed numerical roots are evidence, not a root-completeness or physical-admission proof.")


def audit_actual_probe(path):
    """Numerical turning point and exact-journal boundary-event evidence."""
    data = Path(path).read_bytes()
    probe = json.loads(data)
    with localcontext() as context:
        context.prec = PRECISION+30
        raw = trajectory_from_actual_probe(probe)
        trajectory = IndependentTrajectory(raw)
        stationary = bisect_numeric_root(trajectory, ZERO, ONE, derivative_order=1)
        samples = []
        for value in (ZERO, Decimal(".25"), stationary, Decimal(".75"), ONE):
            derivatives = trajectory.derivatives(value)
            samples.append(dict(fraction=str(value), gap=str(derivatives[0]),
                gap_derivative_per_fraction=str(derivatives[1]),
                gap_second_derivative_per_fraction=str(derivatives[2])))
        if (trajectory.derivatives(ZERO)[1] >= 0 or trajectory.derivatives(ONE)[1] <= 0
                or any(Decimal(item["gap"]) >= 0 for item in samples[1:-1])
                or max(abs(trajectory.gap(ZERO)), abs(trajectory.gap(ONE))) >= Decimal("1e-430")):
            raise ValueError("Actual full-journal independent sweep no longer matches boundary-exit/entry evidence")
        receipt = dict(format="independent-native-contact-event-decimal-oracle-1", status="PASS",
            precision_decimal_digits=PRECISION, probe_sha256=hashlib.sha256(data).hexdigest(),
            oracle_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            computation="Standalone Decimal Taylor sin/cos and full-vector Rodrigues; no production imports",
            trajectory=raw, fraction_derivative_turning_point=str(stationary), samples=samples,
            carrier_recrossing=carrier_recrossing_evidence(probe),
            numerical_boundary_events=[dict(fraction="0", kind="exit"), dict(fraction="1", kind="entry")],
            endpoint_gap_residual_bound="1e-430", admission_authorized=False,
            limitation="Numerical independent evidence; it does not prove completeness or authorize physical underthrust.")
    receipt["receipt_sha256"] = hashlib.sha256(_encoded(receipt)).hexdigest()
    return receipt


def _compare_scalar_interval(raw, value, name):
    if type(raw) is not dict or set(raw) != {"lower", "upper"}:
        raise ValueError("Independent control audit requires exact rational interval endpoints")
    low, high, value = decimal(raw["lower"]), decimal(raw["upper"]), decimal(value)
    if not low <= value <= high:
        raise ValueError(f"Production {name} enclosure omits independently computed value")
    return dict(quantity=name, certified_lower=str(low), certified_upper=str(high),
        independently_computed_value=str(value), interval_contains_independent_value=True)


def audit_event_control(directory):
    """Compare a frozen exact event control with unrelated high-precision math."""
    directory = Path(directory)
    report_data = (directory/"report.json").read_bytes()
    report = json.loads(report_data)
    if report.get("status") != "PASS" or report.get("admission_authorized") is not False:
        raise ValueError("Independent event audit requires a successful geometry-only event control")
    artifacts = report["artifacts_sha256"]
    for name, expected in artifacts.items():
        if Path(name).name != name:
            raise ValueError("Independent event artifact name must be local and explicit")
        if hashlib.sha256((directory/name).read_bytes()).hexdigest() != expected:
            raise ValueError("Independent event artifact digest differs from frozen control")
    probe_path = directory/"source_force_candidate_refusal_probe.json"
    if hashlib.sha256(probe_path.read_bytes()).hexdigest() != report["source_probe_sha256"]:
        raise ValueError("Independent oracle requires the original immutable candidate bytes")
    independent = audit_actual_probe(probe_path)
    carrier = independent["carrier_recrossing"]
    with localcontext() as context:
        context.prec = PRECISION+30
        comparisons = [
            _compare_scalar_interval(report["turning_fraction"], carrier["turning_point_fraction"], "turning_fraction"),
            _compare_scalar_interval(report["turning_time_myr"], carrier["turning_point_time_myr"], "turning_time_myr"),
            _compare_scalar_interval(report["initial_ratio"], carrier["initial_ratio_limit"], "initial_ratio"),
            _compare_scalar_interval(report["maximum_ratio"], carrier["ratio_at_numerical_turning_point"], "maximum_ratio"),
            _compare_scalar_interval(report["recrossing_ratio_width"], carrier["positive_ratio_band_width"], "recrossing_ratio_width"),
        ]
    if report["topology"] != "initial_exit_later_entry_positive_material_band":
        raise ValueError("Production carrier topology differs from independent exit/re-entry evidence")
    receipt = dict(format="independent-native-certified-contact-event-control-audit-1", status="PASS",
        computation=independent["computation"], precision_decimal_digits=PRECISION,
        oracle_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_probe_sha256=independent["probe_sha256"], report_sha256=hashlib.sha256(report_data).hexdigest(),
        event_certificate_sha256=artifacts["event_certificate.json"],
        checked_artifact_count=len(artifacts), certified_interval_comparisons=comparisons,
        independently_computed_evidence=independent,
        accepted_loaded_native_intervals=0, admission_authorized=False,
        limitation="Independent numerical corroboration of certified event intervals; no force state or physical admission is advanced.")
    receipt["receipt_sha256"] = hashlib.sha256(_encoded(receipt)).hexdigest()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--probe", type=Path)
    source.add_argument("--control", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    receipt = audit_event_control(args.control) if args.control else audit_actual_probe(args.probe)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(_encoded(receipt)+b"\n")
    print(json.dumps(dict(status=receipt["status"], output=str(args.out),
        receipt_sha256=receipt["receipt_sha256"]), sort_keys=True))


if __name__ == "__main__":
    main()
