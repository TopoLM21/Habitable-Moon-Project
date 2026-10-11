"""Independent numerical and artifact audit of portable common-motion epochs.

The area/pose oracle uses separate Decimal vector algebra, a direct Taylor
series and a Chudnovsky pi calculation. It imports no production numerical
helpers. Public objects are used only as the measured implementation. This is
not a loaded underthrust, native cutting, contact rematching or cooling test.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Context, Decimal, ROUND_CEILING, ROUND_DOWN, ROUND_HALF_EVEN, localcontext
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tectonics.geometric_intrinsic_geometry import IntrinsicGeometry, RotationIncrement
from tectonics.geometric_intrinsic_surface import load_intrinsic_checkpoint

PRECISION = 220
DT = 3.168808781402895e-8
EXPECTED_STEPS = (
    ((4.171092555597449e-18, 2.4581194098898567e-19, 34.86996946470772), DT),
    ((.023, -.037, .049), .71),
    ((3.5029521325109036e-18, -7.03581045461479e-20, 31.616288107981852), DT),
    ((-.16, .09, .13), .29),
) * 2
RESERVOIRS = ("area_km2", "oceanic_volume_km3", "cold_mantle_volume_km3",
              "density_excess_mass_kg")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def envelope(path, expected_format=None):
    raw = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    signature = raw.pop("payload_sha256")
    assert hashlib.sha256(encoded(raw)).hexdigest() == signature
    if expected_format is not None:
        assert set(raw) == {"format", "state", "provenance"}
        assert raw["format"] == expected_format
        assert set(raw["state"]) == {"reference_surface", "rotations", "version"}
        assert raw["state"]["version"] == "geometric-intrinsic-surface-1"
        for step in raw["state"]["rotations"]:
            assert set(step) == {"omega_rad_per_myr", "dt_myr"}
    return raw


def dec(value):
    return Decimal.from_float(float(value))


def dot(first, second):
    return sum((left*right for left, right in zip(first, second)), Decimal(0))


def cross(first, second):
    return tuple(first[j]*second[k]-first[k]*second[j]
                 for j, k in ((1, 2), (2, 0), (0, 1)))


def unit(ray):
    norm = dot(ray, ray).sqrt()
    assert norm > 0
    return tuple(value/norm for value in ray)


def pi_chudnovsky(precision):
    # An independent series, unlike production's Gauss-Legendre iteration.
    with localcontext(Context(prec=precision+30)):
        numerator = Decimal(426880)*Decimal(10005).sqrt()
        total = Decimal(13591409)
        multiplier, linear, power, index_term = Decimal(1), total, Decimal(1), Decimal(6)
        for index in range(1, precision//14+5):
            multiplier *= (index_term**3-16*index_term)/Decimal(index**3)
            linear += 545140134
            power *= -262537412640768000
            total += multiplier*linear/power
            index_term += 12
        return numerator/total


def sine_cosine(angle, precision, *, reduce_angle=False):
    if reduce_angle:
        period = 2*pi_chudnovsky(precision)
        turns = (angle/period).to_integral_value(rounding=ROUND_HALF_EVEN)
        angle -= turns*period
    square = angle*angle
    sine_term, cosine_term = angle, Decimal(1)
    sine, cosine = sine_term, cosine_term
    for index in range(1, 20000):
        sine_term *= -square/Decimal((2*index)*(2*index+1))
        cosine_term *= -square/Decimal((2*index-1)*(2*index))
        following_sine, following_cosine = sine+sine_term, cosine+cosine_term
        if following_sine == sine and following_cosine == cosine:
            return sine, cosine
        sine, cosine = following_sine, following_cosine
    raise AssertionError("Independent Taylor oracle did not converge")


def atan(value, precision):
    if value < 0:
        return -atan(-value, precision)
    multiplier = Decimal(1)
    while value > Decimal(".125"):
        value /= Decimal(1)+(Decimal(1)+value*value).sqrt()
        multiplier *= 2
    square = value*value
    term = total = value
    threshold = Decimal(10)**(-(precision-10))
    for index in range(1, 20000):
        term *= -square
        addition = term/Decimal(2*index+1)
        total += addition
        if abs(addition) < threshold:
            return multiplier*total
    raise AssertionError("Independent area oracle did not converge")


def area(rays, precision):
    result = Decimal(0)
    anchor = rays[0]
    for first, second in zip(rays[1:-1], rays[2:]):
        numerator = dot(anchor, cross(first, second))
        denominator = Decimal(1)+dot(anchor, first)+dot(first, second)+dot(second, anchor)
        assert denominator > 0
        result += 2*atan(numerator/denominator, precision)
    return abs(result)


def replay(body, steps, precision=PRECISION, *, reduce_angle=False):
    with localcontext(Context(prec=precision)):
        rays = tuple(unit(tuple(dec(value) for value in row)) for row in body)
        body_area = area(rays, precision)
        for vector, duration in steps:
            omega = tuple(dec(value) for value in vector)
            magnitude = dot(omega, omega).sqrt()
            if magnitude == 0:
                continue
            axis = tuple(value/magnitude for value in omega)
            sine, cosine = sine_cosine(magnitude*dec(duration), precision,
                                      reduce_angle=reduce_angle)
            next_rays = []
            for ray in rays:
                crossing = cross(axis, ray)
                projection = dot(axis, ray)
                next_rays.append(unit(tuple(ray[index]*cosine+crossing[index]*sine
                    +projection*axis[index]*(1-cosine) for index in range(3))))
            rays = tuple(next_rays)
        return rays, body_area, area(rays, precision)


def code_hashes():
    paths = [ROOT/"tectonics/geometric_intrinsic_geometry.py",
             ROOT/"tectonics/geometric_intrinsic_surface.py",
             ROOT/"analysis/geometric_intrinsic_validation/run_intrinsic_control.py",
             Path(__file__).resolve()]
    paths.extend(sorted((ROOT/"tests").glob("test_geometric_intrinsic*.py")))
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in paths}


def audit_control(control):
    report = json.loads((control/"report.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed_common_motion_intrinsic_geometry_control"
    assert report["accepted_common_geometry_steps"] == 8
    assert report["new_loaded_underthrust_steps"] == 0
    native = {name: envelope(control/name, "geometric-intrinsic-surface-checkpoint-1")
              for name in report["checkpoint_sha256"]}
    for name, digest in report["checkpoint_sha256"].items():
        assert sha(control/name) == digest
    for pair in report["byte_pairs"]:
        first, second = control/pair["first"], control/pair["second"]
        assert first.read_bytes() == second.read_bytes()
        assert sha(first) == pair["sha256"]
        assert first.stat().st_size == pair["bytes"]
    assert len(report["byte_pairs"]) == 3
    initial = native["initial.json"]
    source_path = ROOT/initial["provenance"]["source_checkpoint"]
    source_hash = sha(source_path)
    assert source_hash == report["source_checkpoint_sha256"]
    assert source_hash == initial["provenance"]["source_checkpoint_sha256"]
    original = envelope(source_path)
    reference = initial["state"]["reference_surface"]
    assert reference == original["surface"]
    assert initial["state"]["rotations"] == []
    fragment_records = []
    ledger = {}
    for fragment in reference["fragments"]:
        parcel = fragment["parcel"]
        for reservoir in RESERVOIRS:
            ledger.setdefault((parcel["material_id"], reservoir), []).append(parcel[reservoir])
    ledger = {key: math.fsum(values) for key, values in ledger.items()}
    max_area_drift = Decimal(0)
    max_body_area_error = 0.
    min_decimal_area = None
    projection_checks = 0
    for name in ("initial.json", "partial.json", "direct_final.json", "resumed_final.json"):
        saved = native[name]
        assert saved["state"]["reference_surface"] == reference
        steps = tuple((tuple(step["omega_rad_per_myr"]), step["dt_myr"])
                      for step in saved["state"]["rotations"])
        expected_count = {"initial.json": 0, "partial.json": 4,
                          "direct_final.json": 8, "resumed_final.json": 8}[name]
        assert steps == EXPECTED_STEPS[:expected_count]
        actual, provenance = load_intrinsic_checkpoint(control/name)
        assert provenance == saved["provenance"]
        expected_time = reference["time_myr"]
        for _, duration in steps:
            expected_time += duration
        assert actual.time_myr == expected_time
        current_ledger = {}
        for old, fragment in zip(reference["fragments"], actual.fragments):
            assert fragment.fragment_id == old["fragment_id"]
            assert fragment.parent_fragment_id == old["parent_fragment_id"]
            assert fragment.geometry.body_rays == tuple(map(tuple, old["polygon"]))
            expected_age = old["parcel"]["age_myr"]
            for _, duration in steps:
                expected_age += duration
            assert fragment.parcel.age_myr == expected_age
            parcel = asdict(fragment.parcel)
            parcel["age_myr"] = old["parcel"]["age_myr"]
            assert encoded(parcel) == encoded(old["parcel"])
            for reservoir in RESERVOIRS:
                current_ledger.setdefault((fragment.parcel.material_id, reservoir), []).append(
                    getattr(fragment.parcel, reservoir))
            oracle_rays, body_area, world_area = replay(old["polygon"], steps)
            assert body_area > 0 and world_area > 0
            with localcontext(Context(prec=PRECISION)):
                drift = abs(world_area/body_area-1)
                assert drift < Decimal("1e-130")
            max_area_drift = max(max_area_drift, drift)
            body_error = abs(fragment.geometry.area_sr/float(body_area)-1.)
            assert body_error <= 2e-10
            max_body_area_error = max(max_body_area_error, body_error)
            min_decimal_area = world_area if min_decimal_area is None else min(min_decimal_area, world_area)
            display = fragment.geometry.world_polygon()
            assert display == tuple(tuple(float(value) for value in ray) for ray in oracle_rays)
            decimal_display = fragment.geometry.decimal_world_rays()
            with localcontext(Context(prec=PRECISION)):
                assert all(abs(measured-expected) < Decimal("1e-65")
                           for measured_ray, expected_ray in zip(decimal_display, oracle_rays)
                           for measured, expected in zip(measured_ray, expected_ray))
            projection_checks += 1
        assert {key: math.fsum(values) for key, values in current_ledger.items()} == ledger
        assert len(actual.fragments) == len(reference["fragments"]) == 16
        fragment_records.append(dict(checkpoint=name, steps=len(steps), fragments=16,
                                     full_pose_and_all_reservoirs_exact=True))
    assert sha(source_path) == source_hash
    return dict(control_report_sha256=sha(control/"report.json"),
        source_checkpoint_sha256=source_hash, source_reference_exact=True,
        source_fragment_count=16, source_material_count=len({key[0] for key in ledger}),
        projection_checks=projection_checks, independent_precision=PRECISION,
        max_relative_oracle_rotation_area_drift=str(max_area_drift),
        max_relative_saved_body_area_error=max_body_area_error,
        min_positive_oracle_world_area_sr=str(min_decimal_area),
        three_native_byte_pairs_exact=True, checkpoints=fragment_records,
        checkpoint_sha256=report["checkpoint_sha256"])


def audit_extreme_angles():
    body = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
    cases = ((1., 1.), (1e308, 1e308), (1e-308, 1e-308), (5e-324, 1.),
             (1., 5e-324), (1e300, 1e-300), (1e-300, 1e300), (-1e308, 1e308))
    results = []
    for velocity, duration in cases:
        increment = RotationIncrement((0., 0., velocity), duration)
        measured = IntrinsicGeometry(body, (increment,))
        oracle, _, _ = replay(body, ((increment.omega_rad_per_myr, duration),),
                               precision=1400, reduce_angle=True)
        expected = tuple(tuple(float(value) for value in row) for row in oracle)
        assert measured.world_polygon() == expected
        results.append(dict(omega_z=velocity, dt_myr=duration,
                            selected_precision=measured.projection_precision,
                            binary64_projection_exact=True))
    return results


def audit_ambient_context():
    body = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
    measured = IntrinsicGeometry(body, tuple(
        RotationIncrement((1e-25, -1e-27, .89), 1e-6) for _ in range(9)))
    reference = (measured.projection_precision, measured.decimal_world_rays(),
                 measured.world_polygon())
    results = []
    for precision, rounding in ((1, ROUND_DOWN), (2, ROUND_CEILING), (7, ROUND_DOWN)):
        with localcontext(Context(prec=precision, rounding=rounding)):
            actual = (measured.projection_precision, measured.decimal_world_rays(),
                      measured.world_polygon())
        assert actual == reference
        results.append(dict(host_precision=precision, host_rounding=rounding, exact=True))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, default=ROOT/
                        "analysis/geometric_intrinsic_validation/final_common_motion_control")
    parser.add_argument("--out", type=Path, default=ROOT/
                        "analysis/geometric_intrinsic_validation/independent_audit_receipt.json")
    args = parser.parse_args()
    args.control = args.control.resolve()
    started = datetime.now(timezone.utc).isoformat()
    clock = time.perf_counter()
    before = code_hashes()
    control_receipt = audit_control(args.control)
    angle_receipt = audit_extreme_angles()
    context_receipt = audit_ambient_context()
    after = code_hashes()
    assert before == after
    receipt = dict(status="passed_independent_common_motion_intrinsic_geometry_audit",
        started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
        seconds=time.perf_counter()-clock, control_directory=args.control.relative_to(ROOT).as_posix(),
        code_sha256_before=before, code_sha256_after=after, code_unchanged=True,
        artifact_and_pose_audit=control_receipt, extreme_angles=angle_receipt,
        ambient_contexts=context_receipt,
        scope=dict(new_loaded_underthrust_steps=0, native_differential_cuts_tested=False,
                   new_physical_material_pairs_tested=False, certified_projection_bounds=False,
                   reused_1284_baseline_is_not_a_new_full_profile=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=receipt["status"], seconds=receipt["seconds"],
                         projection_checks=control_receipt["projection_checks"],
                         extreme_angle_cases=len(angle_receipt)), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
