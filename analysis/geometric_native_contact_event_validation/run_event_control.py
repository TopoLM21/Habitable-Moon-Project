"""Reproducible continuous native-contact diagnostic, with no force acceptance.

The input is the immutable saved first physical force-candidate refusal.  Its
complete receiver arc, owner angular velocities, and native endpoint expressions
are kept as evidence.  Repeating and reloading this diagnostic cannot advance
the original physical state or authorize burial.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_PROBE = (ROOT / "analysis/geometric_native_underthrust_validation/"
                 "force_control_release_02/force_candidate_refusal_probe.json")


def encoded(raw):
    return json.dumps(raw, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, raw):
    with Path(path).open("xb") as stream:
        stream.write(encoded(raw) + b"\n")


def code_snapshot():
    paths = set()
    for directory in ("tectonics", "tests", "scripts"):
        paths.update((ROOT / directory).rglob("*.py"))
    paths.update(Path(__file__).resolve().parent.glob("*.py"))
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in sorted(paths)}


def actual_arc_case(probe):
    """Recover the real mixed-convergence arc without a display projection."""
    from tectonics.geometric_intrinsic_geometry import RotationIncrement
    from tectonics.geometric_native_polygons import (
        NativeArc, NativePlane, native_plane_from_dict, native_ray_from_dict,
    )
    assessments = probe["admission"]["actual_overlap_assessments"]
    candidates = [item for item in assessments
                  if item["fragment_a"] == "octant:2" and item["fragment_b"] == "octant:6"]
    if len(candidates) != 1 or len(candidates[0]["receiver_arcs"]) != 1:
        raise ValueError("Control requires the actual octant:2/octant:6 overlap")
    receiver = candidates[0]["receiver_arcs"][0]
    if receiver["status"] != "mixed_convergence" or len(receiver["supports"]) != 1:
        raise ValueError("Control requires the saved complete mixed-convergence support")
    raw = receiver["supports"][0]["native_arc"]
    plane = native_plane_from_dict(raw["plane"])
    over, sub = receiver["overriding_plate"], receiver["subducting_plate"]
    duration = probe["admission"]["dt_myr"]
    omega = {int(owner): tuple(value) for owner, value in probe["omega"].items()}
    if (len(plane.rotations) != 1
            or plane.rotations[0].omega_rad_per_myr != omega[over]
            or plane.rotations[0].dt_myr != duration):
        raise ValueError("A control must preserve every original owner component and duration")
    arc = NativeArc(native_ray_from_dict(raw["start"]), native_ray_from_dict(raw["end"]),
                    plane, raw["length_radians"])
    inverse = tuple(RotationIncrement(tuple(-value for value in item.omega_rad_per_myr),
                                     item.dt_myr) for item in reversed(plane.rotations))
    before_arc = NativeArc(arc.start.rotated(inverse), arc.end.rotated(inverse),
                          arc.plane.rotated(inverse), arc.length_radians)
    evidence = dict(actual_overlap=dict(fragment_a="octant:2", fragment_b="octant:6"),
        source_contact_id=receiver["contact_id"],
        original_physical_trace_id=receiver["supports"][0]["trace_id"],
        receiver_owner=over, donor_owner=sub, receiver_material_id=receiver["receiver_material_id"],
        receiver_arc=deepcopy(raw), receiver_pullback_history=deepcopy(raw["plane"]["rotations"]),
        donor_plane_before=dict(normal=[1, 0, 0], rotations=[], label=""),
        receiver_omega_rad_per_myr=list(omega[over]),
        donor_omega_rad_per_myr=list(omega[sub]), dt_myr=duration,
        accepted_loaded_native_intervals=probe["accepted_loaded_native_intervals"],
        input_unchanged=probe["input_unchanged"])
    if evidence["accepted_loaded_native_intervals"] != 0 or not evidence["input_unchanged"]:
        raise ValueError("This event control must start from an unaccepted atomic force candidate")
    return before_arc, NativePlane((1, 0, 0)), omega[sub], omega[over], duration, evidence


def _interval(raw):
    def rational(value):
        return dict(numerator=value.numerator, denominator=value.denominator)
    return dict(lower=rational(raw.lo), upper=rational(raw.hi))


def material_ratio_queries(certificate):
    """Query real material inside the tiny cap as well as all other classes."""
    from tectonics.geometric_native_contact_events import isolate_native_contact_ratio_events
    choices = (
        ("boundary_endpoint", Fraction(0)),
        ("later_entry", certificate.initial_ratio.lo/2),
        ("exit_and_reentry", (certificate.initial_ratio.hi+certificate.maximum_ratio.lo)/2),
        ("inside_from_start", 2*certificate.maximum_ratio.hi),
    )
    result = []
    for label, ratio in choices:
        events = isolate_native_contact_ratio_events(certificate, ratio)
        result.append(dict(material_class=label,
            homogeneous_arc_ratio=dict(numerator=ratio.numerator, denominator=ratio.denominator),
            events=[dict(kind=event.kind, time_fraction=_interval(event.time_fraction),
                         time_myr=_interval(event.time_myr)) for event in events]))
    if [event["kind"] for event in result[2]["events"]] != ["entry", "exit", "entry"]:
        raise AssertionError("Positive recrossing-band material must retain both interior event times")
    return result


def run_control(probe_path, output):
    """Save exact proof DTO, repeated proof, and strict physical-input replay."""
    from tectonics.geometric_native_contact_events import (
        certify_native_contact_arc_events,
        native_contact_arc_events_from_dict, native_contact_arc_events_to_dict,
    )
    start = time.monotonic()
    probe_path, output = Path(probe_path).resolve(), Path(output)
    if output.exists():
        raise FileExistsError("Event control requires a fresh output directory")
    probe_bytes = probe_path.read_bytes()
    probe = json.loads(probe_bytes)
    arc, plane, donor, receiver, duration, evidence = actual_arc_case(probe)
    before = code_snapshot()
    certificate = certify_native_contact_arc_events(plane, arc, donor, receiver, duration)
    if (certificate.topology != "initial_exit_later_entry_positive_material_band"
            or not 0 < certificate.turning_fraction.lo < certificate.turning_fraction.hi < 1
            or certificate.recrossing_ratio_width.lo <= 0):
        raise AssertionError("Actual full-vector control lost its certified internal recrossing band")
    first = native_contact_arc_events_to_dict(certificate)
    loaded = native_contact_arc_events_from_dict(deepcopy(first))
    replayed = native_contact_arc_events_to_dict(loaded)
    repeated = native_contact_arc_events_to_dict(
        certify_native_contact_arc_events(plane, arc, donor, receiver, duration))
    if not encoded(first) == encoded(replayed) == encoded(repeated):
        raise AssertionError("Native contact diagnostic differs after full proof replay or repetition")
    queries = material_ratio_queries(certificate)
    replayed_queries = material_ratio_queries(loaded)
    if encoded(queries) != encoded(replayed_queries):
        raise AssertionError("Actual material entry/exit times changed after diagnostic restart")
    after = code_snapshot()
    if before != after or probe_path.read_bytes() != probe_bytes:
        raise AssertionError("Event control changed a source file or its physical candidate input")
    output.mkdir(parents=True)
    with (output / "source_force_candidate_refusal_probe.json").open("xb") as stream:
        stream.write(probe_bytes)
    for name, raw in (("code_before.json", before), ("code_after.json", after),
                      ("actual_native_arc_inputs.json", evidence),
                      ("event_certificate.json", first),
                      ("event_certificate_replayed.json", replayed),
                      ("event_certificate_repeated.json", repeated),
                      ("material_ratio_events.json", queries),
                      ("material_ratio_events_replayed.json", replayed_queries)):
        write_json(output / name, raw)
    byte_pairs = []
    first_path = output / "event_certificate.json"
    for other in ("event_certificate_replayed.json", "event_certificate_repeated.json"):
        path = output / other
        byte_pairs.append(dict(first=first_path.name, second=path.name,
            sha256=file_sha256(first_path), byte_exact=first_path.read_bytes() == path.read_bytes()))
    first_path, second_path = output / "material_ratio_events.json", output / "material_ratio_events_replayed.json"
    byte_pairs.append(dict(first=first_path.name, second=second_path.name,
        sha256=file_sha256(first_path), byte_exact=first_path.read_bytes() == second_path.read_bytes()))
    artifact_hashes = {path.name: file_sha256(path) for path in sorted(output.iterdir())}
    try:
        source_probe = str(probe_path.relative_to(ROOT))
    except ValueError:
        source_probe = str(probe_path)
    report = dict(format="native-contact-arc-event-control-1", status="PASS",
        duration_s=time.monotonic()-start, source_probe=source_probe,
        source_probe_sha256=hashlib.sha256(probe_bytes).hexdigest(),
        accepted_loaded_native_intervals=0, admission_authorized=False,
        topology=certificate.topology,
        turning_fraction=_interval(certificate.turning_fraction),
        turning_time_myr=_interval(certificate.turning_time_myr),
        initial_ratio=_interval(certificate.initial_ratio),
        maximum_ratio=_interval(certificate.maximum_ratio),
        recrossing_ratio_width=_interval(certificate.recrossing_ratio_width),
        queried_material_classes=[item["material_class"] for item in queries],
        sources_unchanged=True, source_count=len(before), source_input_unchanged=True,
        byte_pairs=byte_pairs, artifacts_sha256=artifact_hashes,
        scope="Complete saved receiver material arc and full owner rotation journals; bounded native sweep proof.",
        limitation="Event geometry does not supply transient material-history adoption, burial, or an accepted loaded checkpoint.")
    report["report_sha256"] = hashlib.sha256(encoded(report)).hexdigest()
    write_json(output / "report.json", report)
    print(json.dumps(dict(status=report["status"], output=str(output),
        accepted_loaded_native_intervals=0, report_sha256=report["report_sha256"]), sort_keys=True))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run_control(args.probe, args.out)


if __name__ == "__main__":
    main()
