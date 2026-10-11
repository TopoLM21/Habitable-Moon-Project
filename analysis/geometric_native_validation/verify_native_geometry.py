#!/usr/bin/env python3
"""Independent Decimal/Rodrigues audit of literal native checkpoint recipes.

The oracle reads primitive seed rays, primitive support planes and their full
ordered histories directly from JSON. It does not call the native pose, clip,
incidence, contact or area implementations to calculate an expected value.
Production certificates are read only after the independent oracle calculation
and compared with it. Decimal precision and the explicit oracle error budget
are recorded: this is an independent high-precision numerical check, rather
than a replacement for production's exact-rational interval proofs.
"""
from __future__ import annotations

import argparse
from decimal import Context, Decimal, localcontext
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _dot(first, second):
    return sum((a * b for a, b in zip(first, second)), Decimal(0))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _matmul(left, right):
    return tuple(tuple(sum((left[i][k] * right[k][j] for k in range(3)), Decimal(0))
                       for j in range(3)) for i in range(3))


def _map(matrix, ray):
    return tuple(_dot(row, ray) for row in matrix)


def _identity():
    return tuple(tuple(Decimal(int(i == j)) for j in range(3)) for i in range(3))


def _normalise(ray):
    scale = max(map(abs, ray))
    if not scale:
        raise AssertionError("The independent oracle found a zero ray")
    reduced = tuple(value / scale for value in ray)
    norm = _dot(reduced, reduced).sqrt()
    return tuple(value / norm for value in reduced)


def _sincos(theta, error):
    square = theta * theta
    sine = sine_term = theta
    cosine = cosine_term = Decimal(1)
    index = 1
    while True:
        sine_term *= -square / ((2 * index) * (2 * index + 1))
        cosine_term *= -square / ((2 * index - 1) * (2 * index))
        sine += sine_term
        cosine += cosine_term
        if max(abs(sine_term), abs(cosine_term)) < error:
            return sine, cosine
        index += 1


def _atan(value, error):
    if not value:
        return Decimal(0)
    direction = -1 if value < 0 else 1
    value = abs(value)
    multiplier = 1
    while value > Decimal("0.125"):
        value /= 1 + (1 + value * value).sqrt()
        multiplier *= 2
    term = total = value
    square = value * value
    index = 1
    while True:
        term *= -square
        addition = term / (2 * index + 1)
        total += addition
        if abs(addition) < error:
            return direction * multiplier * total
        index += 1


def _pi(error):
    return 16 * _atan(Decimal(1) / 5, error) - 4 * _atan(Decimal(1) / 239, error)


def _atan2_positive(y, x, pi, error):
    assert y > 0, "The oracle requires a positive oriented determinant"
    if x > 0:
        return _atan(y / x, error)
    if x < 0:
        return pi - _atan(y / -x, error)
    return pi / 2


def _history(raw):
    assert isinstance(raw, list)
    result = []
    for step in raw:
        assert isinstance(step, dict) and set(step) == {"omega_rad_per_myr", "dt_myr"}
        omega = step["omega_rad_per_myr"]
        dt = step["dt_myr"]
        assert isinstance(omega, list) and len(omega) == 3
        assert all(type(value) in (float, int) and math.isfinite(value) for value in omega)
        assert type(dt) in (float, int) and math.isfinite(dt) and dt > 0
        # The declared input contract is binary64, including JSON integers.
        result.append((tuple(float(value) for value in omega), float(dt)))
    return tuple(result)


class Oracle:
    def __init__(self, precision):
        self.precision = precision
        self.series_error = Decimal(10) ** (-(precision - 20))
        self.error = Decimal(10) ** (-(precision - 70))
        self.pi = _pi(self.series_error)
        self.histories = set()
        self.max_orthogonality_error = Decimal(0)
        self.max_determinant_error = Decimal(0)

    @lru_cache(maxsize=None)
    def matrix(self, history):
        self.histories.add(history)
        matrix = _identity()
        for omega, dt in history:
            vector = tuple(Decimal.from_float(value) * Decimal.from_float(dt) for value in omega)
            theta = _dot(vector, vector).sqrt()
            if not theta:
                continue
            assert theta <= Decimal("0.25"), "The saved increment exceeds the native angle domain"
            axis = tuple(value / theta for value in vector)
            sine, cosine = _sincos(theta, self.series_error)
            skew = ((Decimal(0), -axis[2], axis[1]),
                    (axis[2], Decimal(0), -axis[0]),
                    (-axis[1], axis[0], Decimal(0)))
            increment = tuple(tuple(cosine * int(i == j)
                + (1 - cosine) * axis[i] * axis[j] + sine * skew[i][j]
                for j in range(3)) for i in range(3))
            matrix = _matmul(increment, matrix)
        transpose = tuple(zip(*matrix))
        product = _matmul(transpose, matrix)
        orthogonality = max(abs(product[i][j] - int(i == j)) for i in range(3) for j in range(3))
        determinant = _dot(matrix[0], _cross(matrix[1], matrix[2]))
        self.max_orthogonality_error = max(self.max_orthogonality_error, orthogonality)
        self.max_determinant_error = max(self.max_determinant_error, abs(determinant - 1))
        assert orthogonality < self.error and abs(determinant - 1) < self.error
        return matrix

    def plane(self, raw):
        assert set(raw) == {"normal", "rotations", "label"}
        assert len(raw["normal"]) == 3 and all(type(value) is int for value in raw["normal"])
        ray = _normalise(tuple(Decimal(value) for value in raw["normal"]))
        return _normalise(_map(self.matrix(_history(raw["rotations"])), ray))

    def ray(self, raw):
        assert set(raw) == {"kind", "reference", "rotations", "planes", "sign"}
        if raw["kind"] == "seed":
            assert not raw["planes"] and raw["sign"] == 1
            assert len(raw["reference"]) == 3 and all(type(value) is int for value in raw["reference"])
            ray = _normalise(tuple(Decimal(value) for value in raw["reference"]))
            return _normalise(_map(self.matrix(_history(raw["rotations"])), ray))
        assert raw["kind"] == "cross" and not raw["reference"] and not raw["rotations"]
        assert type(raw["sign"]) is int and raw["sign"] in (-1, 1) and len(raw["planes"]) == 2
        first, second = (self.plane(plane) for plane in raw["planes"])
        return _normalise(tuple(raw["sign"] * value for value in _cross(first, second)))

    def area(self, rays):
        total = Decimal(0)
        for index in range(1, len(rays) - 1):
            a, b, c = rays[0], rays[index], rays[index + 1]
            numerator = _dot(a, _cross(b, c))
            if abs(numerator) <= self.error:
                continue
            assert numerator > 0, "The independent fan determinant is negative"
            denominator = 1 + _dot(a, b) + _dot(b, c) + _dot(c, a)
            total += 2 * _atan2_positive(numerator, denominator, self.pi, self.series_error)
        assert total > self.error, "Positive footprint is smaller than the oracle budget"
        return total

    def polygon(self, raw):
        assert set(raw) == {"version", "vertices", "supports", "pose"}
        assert raw["version"] == "geometric-native-polygons-1"
        matrix = self.matrix(_history(raw["pose"]))
        rays = tuple(_normalise(_map(matrix, self.ray(ray))) for ray in raw["vertices"])
        supports = tuple(_normalise(_map(matrix, self.plane(plane))) for plane in raw["supports"])
        assert len(rays) >= 3 and len(rays) == len(supports)
        smallest_side = Decimal(1)
        smallest_orientation = Decimal(1)
        for index, plane in enumerate(supports):
            a, b = rays[index], rays[(index + 1) % len(rays)]
            assert abs(_dot(a, plane)) < self.error and abs(_dot(b, plane)) < self.error
            orientation = _dot(_cross(a, b), plane)
            assert orientation > self.error, "The oracle cannot prove a positive minor edge"
            smallest_orientation = min(smallest_orientation, orientation)
            for ray in rays:
                side = _dot(ray, plane)
                assert side >= -self.error, "The independent footprint is not convex"
                smallest_side = min(smallest_side, side)
        return self.area(rays), smallest_orientation, smallest_side

    def arc(self, raw):
        assert set(raw) == {"start", "end", "plane", "length_radians"}
        a, b, plane = self.ray(raw["start"]), self.ray(raw["end"]), self.plane(raw["plane"])
        assert abs(_dot(a, plane)) < self.error and abs(_dot(b, plane)) < self.error
        crossing = _cross(a, b)
        assert _dot(crossing, plane) > self.error, "The independent contact has no positive oriented minor arc"
        sine = _dot(crossing, crossing).sqrt()
        angle = _atan2_positive(sine, _dot(a, b), self.pi, self.series_error)
        assert self.error < angle < self.pi
        received = Decimal.from_float(float(raw["length_radians"]))
        relative_error = abs(received / angle - 1)
        assert relative_error < Decimal("2e-12")
        return angle, relative_error


def _canonical(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _fraction(value):
    return Decimal(value.numerator) / Decimal(value.denominator)


def _short(value):
    return format(value, ".24E")


def audit(paths, precision, require_off_axis, step_paths=()):
    # Imports below provide the production receipt being checked. No oracle
    # matrix, ray, support, topology sign or area is calculated with them.
    from tectonics.geometric_intrinsic_geometry import RotationIncrement
    from tectonics.geometric_native_polygons import native_polygon_from_dict
    from tectonics.geometric_native_pose import certify_rotation_history

    receipts = []
    with localcontext(Context(prec=precision)):
        oracle = Oracle(precision)
        total_fragments = 0
        max_material_area_error = Decimal(0)
        radii = {}
        for path in paths:
            data = path.read_bytes()
            payload = json.loads(data)
            assert set(payload) == {"format", "state", "provenance", "payload_sha256"}
            digest = payload.pop("payload_sha256")
            assert digest == hashlib.sha256(_canonical(payload)).hexdigest()
            assert payload["format"] == "geometric-native-surface-checkpoint-1"
            state = payload["state"]
            radius = Decimal.from_float(float(state["radius_km"]))
            radii[path.parent.resolve()] = radius
            areas, orientations, minima = [], [], []
            native_cut_vertices = 0
            for fragment in state["fragments"]:
                raw = fragment["geometry"]
                # Independent calculation comes first.
                area, orientation, side = oracle.polygon(raw)
                geometry = native_polygon_from_dict(raw)
                certificate = geometry.area_bounds()
                expected = Fraction(area)
                allowance = Fraction(oracle.error)
                assert certificate.lo - allowance <= expected <= certificate.hi + allowance
                assert certificate.lo > 0 and certificate.width <= certificate.lo / 10**13
                material = Decimal.from_float(float(fragment["parcel"]["area_km2"]))
                area_error = abs(material / (area * radius * radius) - 1)
                assert area_error < Decimal("2e-10")
                max_material_area_error = max(max_material_area_error, area_error)
                native_cut_vertices += sum(ray["kind"] == "cross" for ray in raw["vertices"])
                areas.append(area)
                orientations.append(orientation)
                minima.append(side)
            sphere_error = abs(sum(areas, Decimal(0)) - 4 * oracle.pi)
            assert sphere_error < oracle.error * max(1, len(areas)), "The independent sphere area does not close"
            total_fragments += len(areas)
            receipts.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(data).hexdigest(),
                fragments=len(areas), native_cut_vertices=native_cut_vertices,
                min_area_sr=_short(min(areas)), sphere_area_absolute_error_sr=_short(sphere_error),
                min_positive_minor_edge_orientation=_short(min(orientations)),
                most_negative_halfspace_residual=_short(min(minima))))
        step_receipts = []
        for path in step_paths:
            data = path.read_bytes()
            record = json.loads(data)
            radius = radii[path.parent.resolve()]
            loss_areas, contact_lengths = [], []
            largest_contact_error = Decimal(0)
            for loss in record["losses"]:
                assert loss["attachment_status"] == "unresolved_native_support"
                area, orientation, side = oracle.polygon(loss["geometry"])
                certificate = native_polygon_from_dict(loss["geometry"]).area_bounds()
                assert certificate.lo - Fraction(oracle.error) <= Fraction(area) <= certificate.hi + Fraction(oracle.error)
                received = Decimal.from_float(float(loss["parcel"]["area_km2"]))
                area_error = abs(received / (area * radius * radius) - 1)
                assert area_error < Decimal("2e-10")
                max_material_area_error = max(max_material_area_error, area_error)
                loss_areas.append(area)
            for contact in record["contacts"]:
                angle, angle_error = oracle.arc(contact["arc"])
                received = Decimal.from_float(float(contact["length_km"]))
                assert abs(received / (angle * radius) - 1) < Decimal("2e-12")
                assert contact["plate_a"] != contact["plate_b"]
                contact_lengths.append(angle)
                largest_contact_error = max(largest_contact_error, angle_error)
            assert len(loss_areas) == record["loss_count"] and len(contact_lengths) == record["contact_count"]
            for row in record["independent_material_ledger"]["reservoirs"]:
                left = row["retained"] + row["lost"]
                right = row["initial"] + row["born"]
                assert math.isclose(left, right, rel_tol=5e-12, abs_tol=0.)
            step_receipts.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(data).hexdigest(),
                losses=len(loss_areas), contacts=len(contact_lengths),
                min_loss_area_sr=_short(min(loss_areas)) if loss_areas else None,
                min_contact_length_radians=_short(min(contact_lengths)) if contact_lengths else None,
                max_contact_relative_error=_short(largest_contact_error)))
        checked_entries = 0
        for history in oracle.histories:
            steps = tuple(RotationIncrement(omega, dt) for omega, dt in history)
            certificate = certify_rotation_history(steps, 128)
            expected = oracle.matrix(history)
            for i in range(3):
                for j in range(3):
                    interval = certificate.intervals[i][j]
                    assert _fraction(interval.lo) - oracle.error <= expected[i][j] <= _fraction(interval.hi) + oracle.error
                    checked_entries += 1
            # Independently check the rational scratch receipt is exactly SO(3).
            matrix = certificate.matrix
            product = tuple(tuple(sum((matrix[k][i] * matrix[k][j] for k in range(3)), Fraction(0))
                                  for j in range(3)) for i in range(3))
            assert product == tuple(tuple(Fraction(int(i == j)) for j in range(3)) for i in range(3))
            determinant = sum((matrix[0][i] * (matrix[1][(i+1)%3] * matrix[2][(i+2)%3]
                - matrix[1][(i+2)%3] * matrix[2][(i+1)%3]) for i in range(3)), Fraction(0))
            assert determinant == 1
        unique_steps = {step for history in oracle.histories for step in history}
        off_axis = [step for step in unique_steps if len([x for x in step[0] if x]) == 3
            and min(map(abs, step[0])) / max(map(abs, step[0])) < 1e-10]
        if require_off_axis:
            assert off_axis, "The supplied checkpoints contain no full tiny off-axis source motion"
        source_files = [ROOT / "tectonics" / name for name in (
            "geometric_native_pose.py", "geometric_native_polygons.py", "geometric_native_surface.py")]
        source_files.append(Path(__file__).resolve())
        return dict(status="passed", scope="native geometry numerical independent verification; no force law",
            oracle="full ordered Decimal Rodrigues; independent primitive cross recipes and spherical fan angles",
            decimal_precision=precision, absolute_oracle_error_budget=str(oracle.error),
            checkpoints=receipts, step_records=step_receipts, fragment_geometries=total_fragments,
            distinct_full_histories=len(oracle.histories), distinct_full_increments=len(unique_steps),
            rotation_interval_entries_checked=checked_entries,
            tiny_off_axis_full_increments=[dict(omega_rad_per_myr=list(omega),dt_myr=dt) for omega,dt in sorted(off_axis)],
            zero_angular_velocity_increments=sum(not any(omega) for omega, dt in unique_steps),
            max_oracle_SO3_orthogonality_error=_short(oracle.max_orthogonality_error),
            max_oracle_SO3_determinant_error=_short(oracle.max_determinant_error),
            max_material_area_relative_error=_short(max_material_area_error),
            source_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--precision", type=int, default=500)
    parser.add_argument("--require-off-axis", action="store_true")
    parser.add_argument("--step-records", nargs="*", type=Path, default=[],
                        help="Audit loss/contact records beside one of the supplied checkpoints")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 250 <= args.precision <= 2000:
        parser.error("precision must be between 250 and 2000 Decimal digits")
    report = audit(args.checkpoints, args.precision, args.require_off_axis, args.step_records)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
