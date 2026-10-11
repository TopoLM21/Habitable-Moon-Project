"""Independent Decimal/Rodrigues evidence for the first native force refusal.

This oracle imports no production geometry or admission implementation.  It
reads saved primitive rays, support planes and the complete solved owner
velocities, then evaluates analytic rotations with Decimal Taylor series.  A
positive opening signal is evidence for retaining the refusal.  The swept gap
samples explain why endpoint occupancy alone cannot authorize underthrust;
they neither prove a continuous event schedule nor grant physical admission.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path


PRECISION = 500


def _decimal(value):
    if type(value) is float:
        return Decimal.from_float(value)
    if type(value) is int:
        return Decimal(value)
    raise ValueError("Independent oracle requires explicit finite binary64 or integer scalars")


def _vector(values):
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError("Independent oracle requires complete Cartesian triples")
    result = tuple(_decimal(value) for value in values)
    if any(not value.is_finite() for value in result):
        raise ValueError("Independent oracle requires finite complete Cartesian triples")
    return result


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), Decimal(0))


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0])


def _normalize(vector):
    norm = _dot(vector, vector).sqrt()
    if not norm:
        raise ValueError("Independent oracle cannot normalize a zero ray")
    return tuple(value/norm for value in vector)


def _sin_cos(angle):
    threshold = Decimal(10) ** -(PRECISION+20)
    square = angle*angle
    sin_sum = sin_term = angle
    cos_sum = cos_term = Decimal(1)
    index = 0
    while True:
        sin_term *= -square / Decimal((2*index+2)*(2*index+3))
        cos_term *= -square / Decimal((2*index+1)*(2*index+2))
        sin_sum += sin_term
        cos_sum += cos_term
        index += 1
        if max(abs(sin_term), abs(cos_term)) < threshold:
            return sin_sum, cos_sum


def _rotate(vector, omega, duration):
    speed = _dot(omega, omega).sqrt()
    if not speed or not duration:
        return vector
    axis = tuple(value/speed for value in omega)
    sine, cosine = _sin_cos(speed*duration)
    crossed, projection = _cross(axis, vector), _dot(axis, vector)
    return tuple(value*cosine + turned*sine + component*projection*(1-cosine)
                 for value, turned, component in zip(vector, crossed, axis))


def _history(vector, history, *, inverse=False):
    journal = reversed(history) if inverse else history
    for increment in journal:
        omega = _vector(increment["omega_rad_per_myr"])
        duration = _decimal(increment["dt_myr"])
        if duration <= 0:
            raise ValueError("Independent oracle requires positive recorded durations")
        vector = _rotate(vector, omega, -duration if inverse else duration)
    return vector


def _plane(raw):
    return _history(_vector(raw["normal"]), raw["rotations"])


def _ray(raw):
    if raw["kind"] == "seed":
        return _normalize(_history(_vector(raw["reference"]), raw["rotations"]))
    if raw["kind"] != "cross" or len(raw["planes"]) != 2:
        raise ValueError("Independent oracle needs an explicit saved seed or two-plane cross ray")
    value = _cross(*(_plane(plane) for plane in raw["planes"]))
    return _normalize(tuple(_decimal(raw["sign"])*component for component in value))


def _sign(value):
    return "positive" if value > 0 else "negative" if value < 0 else "zero"


def _encoded(raw):
    return json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def audit(probe_path, *, radius_km=100.):
    probe_path = Path(probe_path)
    probe_bytes = probe_path.read_bytes()
    probe = json.loads(probe_bytes)
    candidates = [row for row in probe["admission"]["actual_overlap_assessments"]
                  if (row["fragment_a"], row["fragment_b"]) == ("octant:2", "octant:6")]
    if len(candidates) != 1 or len(candidates[0]["receiver_arcs"]) != 1:
        raise ValueError("Independent first-force oracle requires the actual octant:2/octant:6 arc")
    receiver = candidates[0]["receiver_arcs"][0]
    if len(receiver["supports"]) != 1:
        raise ValueError("Independent first-force oracle requires one actual original physical support")
    support = receiver["supports"][0]
    arc = support["native_arc"]
    if arc["plane"]["normal"] != [-1, 0, 0] or len(arc["plane"]["rotations"]) != 1:
        raise ValueError("Independent first-force oracle needs the full receiver -X rotation journal")
    endpoint = arc["end"]
    if (endpoint["kind"] != "cross" or len(endpoint["planes"]) != 2
            or [plane["normal"] for plane in endpoint["planes"]] != [[-1, 0, 0], [1, 0, 0]]):
        raise ValueError("Independent first-force oracle requires the saved moved-bank cross endpoint")
    over, sub = str(receiver["overriding_plate"]), str(receiver["subducting_plate"])
    over_history, sub_history = (plane["rotations"] for plane in endpoint["planes"])
    if (len(over_history) != 1 or len(sub_history) != 1
            or over_history[0]["omega_rad_per_myr"] != probe["omega"][over]
            or sub_history[0]["omega_rad_per_myr"] != probe["omega"][sub]
            or over_history != arc["plane"]["rotations"]
            or over_history[0]["dt_myr"] != sub_history[0]["dt_myr"]
            or over_history[0]["dt_myr"] != probe["admission"]["dt_myr"]):
        raise ValueError("Independent oracle cannot alter an actual owner component or interval")
    with localcontext() as context:
        context.prec = PRECISION+30
        omega_over, omega_sub = _vector(probe["omega"][over]), _vector(probe["omega"][sub])
        difference = tuple(a-b for a, b in zip(omega_over, omega_sub))
        duration, radius = _decimal(probe["admission"]["dt_myr"]), _decimal(radius_km)
        x = (Decimal(1), Decimal(0), Decimal(0))
        inward = tuple(-value for value in x)
        endpoint_values = []
        for name in ("start", "end"):
            point = _history(_ray(arc[name]), over_history, inverse=True)
            velocity = radius*_dot(inward, _cross(difference, point))
            endpoint_values.append(dict(endpoint=name, normal_velocity_km_per_myr=str(velocity),
                                        independently_computed_sign=_sign(velocity)))
        point_before = _history(_ray(endpoint), over_history, inverse=True)
        swept = []
        for fraction in (Decimal(0), Decimal(".25"), Decimal(".5"), Decimal(".75"), Decimal(1)):
            instant = duration*fraction
            donor_normal = _rotate(x, omega_sub, instant)
            receiver_point = _rotate(point_before, omega_over, instant)
            gap = _dot(donor_normal, receiver_point)
            derivative = _dot(donor_normal, _cross(difference, receiver_point))
            swept.append(dict(interval_fraction=str(fraction), time_myr=str(instant),
                signed_gap=str(gap), signed_gap_derivative_per_myr=str(derivative),
                signed_gap_sign=_sign(gap), derivative_sign=_sign(derivative)))
        zero_residual = max(abs(Decimal(swept[index]["signed_gap"])) for index in (0, 4))
        if (Decimal(endpoint_values[0]["normal_velocity_km_per_myr"]) >= 0
                or Decimal(endpoint_values[1]["normal_velocity_km_per_myr"]) <= 0
                or zero_residual >= Decimal("1e-430")
                or any(Decimal(row["signed_gap"]) >= 0 for row in swept[1:4])
                or Decimal(swept[0]["signed_gap_derivative_per_myr"]) >= 0
                or Decimal(swept[-1]["signed_gap_derivative_per_myr"]) <= 0):
            raise ValueError("Independent actual-source endpoint/sweep signs do not support this refusal evidence")
        signal = Decimal(endpoint_values[1]["normal_velocity_km_per_myr"])
        receipt = dict(format="independent-native-first-force-refusal-decimal-rodrigues-1",
            status="PASS", precision_decimal_digits=PRECISION,
            computation="standalone Decimal Taylor sin/cos and direct full-vector Rodrigues; no production imports",
            probe_sha256=hashlib.sha256(probe_bytes).hexdigest(),
            oracle_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            actual_original_pair=["octant:2", "octant:6"],
            actual_owner_omega_rad_per_myr=probe["omega"], dt_myr=probe["admission"]["dt_myr"],
            explicitly_declared_radius_km=radius_km,
            reported_production_reason=receiver["status"], endpoint_velocity_evidence=endpoint_values,
            positive_opening_endpoint_signal_km_per_myr=str(signal),
            swept_gap_definition="g(t)=R_sub(t)(+X) dot R_over(t)p_before; p_before=R_over(DT)^-1 saved endpoint",
            swept_gap_samples=swept, endpoint_zero_numerical_residual=str(zero_residual),
            endpoint_zero_check_bound="1e-430", zero_residual_over_positive_opening_signal=str(zero_residual/signal),
            conclusion="Actual full source produces mixed initial normal velocity. The selected endpoint initially leaves the donor halfspace and returns to its boundary by DT; endpoint occupancy does not authorize the whole continuous interval.",
            admission_authorized=False,
            limitation="High-precision numerical independent evidence for this saved first interval; sampled sweep does not certify a complete event schedule or a new physical entry/recrossing law.")
    receipt["receipt_sha256"] = hashlib.sha256(_encoded(receipt)).hexdigest()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--radius-km", type=float, default=100.)
    args = parser.parse_args()
    receipt = audit(args.probe, radius_km=args.radius_km)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(_encoded(receipt)+b"\n")
    print(json.dumps(dict(status=receipt["status"], receipt_sha256=receipt["receipt_sha256"],
        positive_opening_endpoint_signal_km_per_myr=receipt["positive_opening_endpoint_signal_km_per_myr"],
        endpoint_zero_numerical_residual=receipt["endpoint_zero_numerical_residual"],
        output=str(args.out)), sort_keys=True))


if __name__ == "__main__":
    main()
