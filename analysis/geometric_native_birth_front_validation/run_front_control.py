"""Exclusive full-arc frames and causal HOT entry from a genuinely solved source."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path
import time

from tectonics.geometric_native_birth_contact_front import (
    native_birth_arc_frame, native_birth_arc_frame_to_dict,
    native_birth_arc_frame_from_dict, certify_native_hot_shared_arc_front,
    native_hot_shared_arc_front_to_dict, native_hot_shared_arc_front_from_dict,
    native_hot_shared_arc_front_time_bounds, certify_native_hot_incoming_rectangle,
    native_hot_incoming_rectangle_to_dict,
)
from tectonics.geometric_native_birth_continuum_motion import (
    native_birth_continuum_candidate_to_dict,
)
from tectonics.geometric_native_birth_contact_support import native_birth_contact_support_to_dict
from tectonics.geometric_native_physical_history import _literal_json
from tectonics.geometric_native_pose import FractionInterval as I
from tectonics.thermal import SECONDS_PER_MYR

from .validation_support import digest, encoded, guard_sources, read, value_sha, write


def _interval(value):
    return dict(lo=[value.lo.numerator, value.lo.denominator],
        hi=[value.hi.numerator, value.hi.denominator])


def _refusal(name, function):
    try:
        function()
    except ValueError as error:
        return dict(control=name, status="REFUSED", message=str(error))
    raise AssertionError("The invalid control was accepted: " + name)


def run_front_control(out, *, mesh, previous, load, support, candidate,
        require_frozen=True, expected_candidate_packet=None):
    """Bind complete native geometry to fresh full-arc force replay.

    The complete typed force source is required.  A saved candidate hash or
    saved Euler vector alone cannot authenticate this control.  The optional
    saved packet must agree with the complete freshly replayed source.
    """
    before, inherited = guard_sources(require_frozen=require_frozen)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write(out / "sources_before.json", before)
    started = time.perf_counter()
    packet = native_birth_continuum_candidate_to_dict(candidate, mesh=mesh,
        previous=previous, load=load, support=support)
    _literal_json(packet)
    unsigned = {key: value for key, value in packet.items() if key != "candidate_sha256"}
    if (packet.get("candidate_sha256") != value_sha(unsigned)
            or packet.get("format") != "native-source-bound-birth-continuum-force-candidate-1"
            or packet.get("birth_support") != native_birth_contact_support_to_dict(support)
            or packet.get("material_admission_authorized") is not False
            or packet.get("born_finite_pose_branch_admission_proved") is not False
            or type(packet.get("accepted_new_native_intervals")) is not int
            or packet["accepted_new_native_intervals"] != 0):
        raise ValueError("The genuine full-arc candidate or its explicit physical scope differs")
    if expected_candidate_packet is not None:
        _literal_json(expected_candidate_packet)
        if encoded(packet) != encoded(expected_candidate_packet):
            raise ValueError("The independent saved candidate differs from complete genuine replay")
    write(out / "candidate_packet.json", packet)
    duration = packet["dt_s"]/SECONDS_PER_MYR
    omega = tuple((owner, tuple(vector)) for owner, vector in packet["omega_rad_per_myr"])
    force_source = packet["candidate_sha256"]
    packet_frames = {row["family_id"]: row for row in packet["native_frames"]}
    if (len(support.interface_carriers) != 2 or len(packet_frames) != 2
            or len(support.foundations) != 5):
        raise ValueError("This control requires the two declared arcs and five whole HOT foundations")
    frames, byte_pairs = [], []
    for index, carrier in enumerate(support.interface_carriers, 1):
        frame = native_birth_arc_frame(carrier,
            precision=max(256, candidate.network_step.decimal_precision+32))
        raw = native_birth_arc_frame_to_dict(frame)
        force_frame = packet_frames[carrier.declaration.family_id]
        expected_bounds = [raw["vectors"][name]["native_unit_bounds"]
            for name in ("start", "tangent", "horizontal")]
        if (force_frame["native_frame_bounds"] != expected_bounds
                or force_frame["native_angle_bounds"] != raw["native_angle_bounds_radians"]
                or force_frame["fixed_angle_rational"] != raw["fixed_constitutive_angle_radians"]
                or force_frame["measure_density_rational"] != raw["measure_density_m2_per_radian"]
                or force_frame["material_area_assigned_to_point"] is not False):
            raise ValueError("The force operators used a different complete native carrier frame")
        path = out / f"native_frame_{index:02d}.json"
        write(path, raw)
        replay = native_birth_arc_frame_from_dict(read(path), expected_carrier=carrier)
        replayed = out / f"native_frame_{index:02d}_replayed.json"
        write(replayed, native_birth_arc_frame_to_dict(replay))
        if path.read_bytes() != replayed.read_bytes():
            raise AssertionError("Independent full native frame replay changed bytes")
        byte_pairs.append([path.name, replayed.name])
        frames.append(dict(family_id=carrier.declaration.family_id,
            donor_component_id=carrier.donor_component.component_id,
            receiver_component_id=carrier.receiver_component.component_id,
            frame_sha256=raw["frame_sha256"], path=path.name))
    carrier = next(row for row in support.interface_carriers
        if row.donor_component.material_kind == "hot_birth")
    front = certify_native_hot_shared_arc_front(support, carrier.declaration.family_id,
        omega, duration, force_source_sha256=force_source)
    raw_front = native_hot_shared_arc_front_to_dict(front)
    write(out / "causal_front.json", raw_front)
    replay_arguments = dict(expected_support=support,
        expected_family_id=carrier.declaration.family_id, expected_omega_rad_per_myr=omega,
        expected_dt_myr=duration, expected_force_source_sha256=force_source)
    replayed_front = native_hot_shared_arc_front_from_dict(read(out / "causal_front.json"),
        **replay_arguments)
    write(out / "causal_front_replayed.json", native_hot_shared_arc_front_to_dict(replayed_front))
    if (out / "causal_front.json").read_bytes() != (out / "causal_front_replayed.json").read_bytes():
        raise AssertionError("Complete actual force, native source and clock replay changed bytes")
    byte_pairs.append(["causal_front.json", "causal_front_replayed.json"])
    windows = []
    for index in range(4):
        bounds = I(F(index, 4), F(index+1, 4))
        roots = native_hot_shared_arc_front_time_bounds(front, bounds)
        windows.append(dict(time_fraction_bounds=_interval(bounds),
            receiver_relative_omega_bounds=[_interval(value)
                for value in roots["receiver_relative_omega_bounds"]],
            flux_root_angle=_interval(roots["flux_root_angle"]),
            donor_upper_root_angle=_interval(roots["donor_upper_root_angle"]),
            incoming_angle_span=_interval(roots["incoming_angle_span"])))
    write(out / "moving_root_time_windows.json", dict(
        front_sha256=front.front_sha256, windows=windows,
        exact_moving_mask="0 < u <= 1; theta_F(u) < theta < min(L, theta_D(u))",
        root_box_midpoint_assigned_as_event=False))
    rectangle = certify_native_hot_incoming_rectangle(front,
        I(F(1, 10**21), F(3, 10**21)), I(0, 1))
    write(out / "positive_incoming_rectangle.json", native_hot_incoming_rectangle_to_dict(rectangle))
    resealed = deepcopy(raw_front)
    resealed["initial_flux_root_angle"]["lo"] = [0, 1]
    resealed["front_sha256"] = value_sha({key: value for key, value in resealed.items()
        if key != "front_sha256"})
    write(out / "deliberately_resealed_root.json", resealed)
    changed_time = dict(replay_arguments, expected_dt_myr=F.from_float(duration)+F(1, 10**40))
    changed_source = dict(replay_arguments, expected_force_source_sha256="0"*64)
    typed_change = replace(front, full_time_flux_root_angle=I(0))
    refusals = [
        _refusal("resealed_root", lambda: native_hot_shared_arc_front_from_dict(resealed, **replay_arguments)),
        _refusal("changed_exact_clock", lambda: native_hot_shared_arc_front_from_dict(raw_front, **changed_time)),
        _refusal("changed_actual_force_source", lambda: native_hot_shared_arc_front_from_dict(raw_front, **changed_source)),
        _refusal("mutated_typed_front_time_query", lambda: native_hot_shared_arc_front_time_bounds(typed_change, I(0, 1))),
        _refusal("outgoing_whole_clock_rectangle", lambda: certify_native_hot_incoming_rectangle(front,
            I(0, F(1, 10**22)), I(0, 1))),
        _refusal("donor_support_crossing_rectangle", lambda: certify_native_hot_incoming_rectangle(front,
            I(F(5, 10**21), F(6, 10**21)), I(0, 1))),
    ]
    write(out / "negative_controls.json", dict(status="PASS", controls=refusals))
    after, inherited_after = guard_sources(require_frozen=require_frozen)
    write(out / "sources_after.json", after)
    if after != before or inherited_after != inherited:
        raise ValueError("A source changed while the complete front control was running")
    artifacts = {path.name: dict(bytes=path.stat().st_size, sha256=digest(path))
        for path in sorted(out.iterdir()) if path.is_file()}
    report = dict(format="native-genuine-full-arc-causal-front-control-1", status="PASS",
        seconds=time.perf_counter()-started, source_count=len(before), inherited_source_count=len(inherited),
        source_guard_frozen=require_frozen, source_before_after_unchanged=True,
        complete_actual_force_replay_performed=True, candidate_sha256=force_source,
        candidate_packet_raw_sha256=digest(out / "candidate_packet.json"),
        source_epoch_sha256=support.source_epoch.epoch_sha256, source_support_sha256=support.support_sha256,
        actual_omega_rad_per_myr=packet["omega_rad_per_myr"], actual_dt_s=packet["dt_s"], dt_myr=duration,
        complete_native_arc_frame_count=len(frames), unique_whole_hot_foundation_count=len(support.foundations),
        frames=frames, certified_causal_pair_count=1, front_sha256=front.front_sha256,
        whole_clock_incoming_angle_span=_interval(front.positive_incoming_angle_span_bounds),
        strict_fixed_material_first_entry_concavity=_interval(front.first_entry_second_derivative_bounds),
        other_receiver_exclusion_count=len(front.other_receiver_exclusions),
        positive_incoming_rectangle_path="positive_incoming_rectangle.json",
        moving_flux_and_donor_support_roots_proved=True, byte_pairs=byte_pairs,
        negative_control_count=len(refusals), artifacts=artifacts,
        incoming_interface_measure_debited_as_material_area=False,
        point_contact_area_or_root_midpoint_used=False,
        mechanical_normal_branch_selected_by_geometry=False,
        physical_down_slip_and_3d_placement_certified=False,
        complete_all_pairs_temporal_coverage=False, force_geometry_admission_authorized=False,
        new_whole_native_intervals_accepted=0, whole_point_1_completed=False,
        scope="Both complete native arc frames; complete causal HOT6-to-HOT2 incoming mask in the actual force clock")
    return write(out / "report.json", report, "report_sha256")
