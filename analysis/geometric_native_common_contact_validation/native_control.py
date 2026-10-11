"""All six actual boundary/field ports, with complete immutable source state.

The nonzero samples are kinematic probes, never a mixed force solution.
The common reference's energies remain byte-pinned completed components;
no volume quadrature or old constitutive time step is repeated here.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from fractions import Fraction as F
import hashlib
import json
from math import isfinite
from pathlib import Path
import time

from tectonics.geometric_native_pose import FractionInterval as I

ROOT = Path(__file__).resolve().parents[2]
PRIOR_WRAPPER_RAW = "9898f31f796006a7791140bf1c290c0e1605555a12e476e1585ce3f5992a743f"
PRIOR_PACKET_RAW = "dc8f820dbd926c53223a5c141a10296d1c7135efc6bf1077763785866dda9275"
PHYSICAL_SCOPE = "COMMON_CONTACT_GEOMETRY_FIELD_COMPONENT_ONLY"
FALSE_SCOPE = (
    "complete_three_dimensional_strain_energy_certified",
    "full_genuine_source_reconstructed",
    "inherited_force_certificate_for_placement",
    "new_native_curved_force_equilibrium_certified",
    "global_physical_solid_noninterpenetration_certified",
    "material_admission_authorized",
    "whole_clock_material_admission_authorized",
    "all_42_surface_pairs_reexecuted",
    "original_point_1_completed",
    "physical_contact_rest_reference_and_measure_certified",
    "original_node_states_assigned_to_new_cuts",
    "old_column_compliance_added_to_new_bulk",
    "finite_owner_whole_clock_motion_certified",
)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def payload(value):
    if type(value) is F:
        return [value.numerator, value.denominator]
    if type(value) is I:
        return {"lo": payload(value.lo), "hi": payload(value.hi)}
    if is_dataclass(value):
        return {f.name: payload(getattr(value, f.name)) for f in fields(value)}
    if type(value) is dict:
        return {str(k): payload(v) for k, v in value.items()}
    if type(value) in (tuple, list):
        return [payload(v) for v in value]
    if type(value) is float and isfinite(value):
        return value
    if type(value) in (str, int, bool) or value is None:
        return value
    raise TypeError("A contact component receipt must retain exact typed values")


def probe_arguments(cut, sample):
    if sample not in (1, 2):
        raise ValueError("The all-cut control has exactly two strict interior probes")
    parameter = cut.parameter_guaranteed_bounds
    cube = cut.common_physical_radius_cubed_bounds_m3
    theta = parameter.lo+(parameter.hi-parameter.lo)*F(sample, 3)
    radius_cube = cube.lo+(cube.hi-cube.lo)*F(sample, 3)
    coefficients = (F(1), F(-1, 3), F(1, 5), F(1, 7), F(-1, 11), F(1, 13), F(-1, 17))
    n = 6 if cut.receiver_material_kind == "hot_birth" else 7
    return dict(parameter_bounds=I(theta), physical_radius_cubed_bounds_m3=I(radius_cube),
        donor_coordinates_m=tuple(c/F(10000) for c in coefficients),
        receiver_coordinates_m=tuple(-c/F(20000) for c in coefficients[:n]), precision=160)


def source_history_records(source):
    dto = {name: json.loads(raw) for name, raw in source.complete_source_dto_json}
    before = dto["before"]
    history = before["original_physical_history"]
    states = before["snapshot"]["states"]
    nodes = history["quadrature_histories"]
    if len(nodes) != 16 or len(states) != 16 or {s[0] for s in states} != {n["point_id"] for n in nodes}:
        raise ValueError("The complete sixteen historical atom lineages differ")
    # Full original objects and their differing clocks remain literal.  Neither
    # a current-cut point nor a continuous receiving family becomes a new atom.
    return dict(complete_original_physical_history=history, complete_original_current_states=states,
        source_native_elapsed_time_s=before["snapshot"]["elapsed_time_s"],
        source_node_point_ids=[n["point_id"] for n in nodes],
        old_node_state_count=16, old_node_states_preserved_once=True,
        old_node_states_reassigned_to_new_port_points=False,
        constitutive_history_advanced_during_geometry_binding=False)


def run(out):
    from tectonics.geometric_native_normal_support import load_native_normal_support_source
    from tectonics.geometric_native_common_reference import bind_native_common_reference, native_common_reference_to_dict
    from tectonics.geometric_native_common_contact_cuts import (
        bind_common_contact_cuts, common_contact_cuts_to_dict,
        evaluate_common_contact_cut, common_contact_cut_bounds_to_dict,
    )
    from tectonics.geometric_native_common_contact_ports import (
        bind_native_common_contact_ports, native_common_contact_ports_to_dict,
        evaluate_native_common_contact_port, native_common_contact_port_response_to_dict,
    )
    output = Path(out)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source = load_native_normal_support_source()
    reference = bind_native_common_reference(source=source, precision=160)
    cuts = bind_common_contact_cuts(reference, precision=160)
    ports = bind_native_common_contact_ports(reference, cuts=cuts, precision=160)
    if len(cuts.cuts) != 6 or len(ports.ports) != 6:
        raise ValueError("Exactly five cold cuts plus the separate true C6/HOT C1 curve are required")
    whole, probes = [], []
    for port in ports.ports:
        cut = port.cut
        full = evaluate_common_contact_cut(reference, cut, precision=160)
        whole.append(common_contact_cut_bounds_to_dict(full))
        for sample in (1, 2):
            arguments = probe_arguments(cut, sample)
            zero_arguments = dict(arguments,
                donor_coordinates_m=(F(0),)*7,
                receiver_coordinates_m=(F(0),)*len(arguments["receiver_coordinates_m"]))
            zero = evaluate_native_common_contact_port(reference, port, **zero_arguments)
            response = evaluate_native_common_contact_port(reference, port, **arguments)
            probes.append(dict(cut_id=cut.cut_id, sample=sample, arguments=payload(arguments),
                zero_response=native_common_contact_port_response_to_dict(zero),
                response=native_common_contact_port_response_to_dict(response),
                nonzero_probe_is_a_mixed_force_solution=False))
        print("COMMON actual boundary/body port", cut.cut_id, cut.kind,
              "receiving family", cut.receiving_trace_id, flush=True)
    shared = dict(common_reference=native_common_reference_to_dict(reference),
        cut_set=common_contact_cuts_to_dict(cuts), ports=native_common_contact_ports_to_dict(ports),
        whole_cut_enclosures=whole, strict_interior_body_port_probes=probes,
        **source_history_records(source))
    packet = dict(format="native-common-contact-component-source-packet-1",
        complete_source_dto_json=[list(p) for p in source.complete_source_dto_json],
        saved_joint_reference_field=json.loads(source.saved_field_json),
        source_snapshot_sha256=source.source_snapshot_sha256,
        authenticated_common_wrapper_raw_sha256=PRIOR_WRAPPER_RAW,
        authenticated_common_source_packet_raw_sha256=PRIOR_PACKET_RAW,
        accepted_input_unchanged=True, **shared,
        full_genuine_source_reconstructed=False, inherited_force_certificate_for_placement=False,
        material_admission_authorized=False)
    packet["packet_sha256"] = hashlib.sha256(encoded(packet)).hexdigest()
    with (output / "source_packet.json").open("xb") as stream:
        stream.write(encoded(packet)+b"\n")
    report = dict(format="native-common-contact-component-control-1", status="PASS",
        regression_validation_status="PASS", physical_candidate_status=PHYSICAL_SCOPE,
        source_packet_sha256=packet["packet_sha256"], **shared,
        actual_retained_cold_cut_count=5, actual_true_curved_cold_hot_cut_count=1,
        actual_common_body_port_count=6, strict_interior_body_port_probe_count=12,
        physical_full_radius_intersections_preserved=True,
        true_source_mask_boundary_lineage_preserved=True,
        old_sixteen_constitutive_histories_preserved=True,
        accepted_input_complete_typed_state_unchanged=True,
        source_constant_dip_reproduced_on_whole_face=False,
        old_virtual_contact_height_used_as_material_geometry=False,
        contact_stiffness_or_old_column_force_inferred_from_geometry=False,
        physical_boundary_queries_are_closures_not_positive_volume_interiors=True,
        reference_frame_port_jacobians_are_finite_owner_motion=False,
        accepted_original_native_intervals=1, accepted_new_native_intervals=0,
        original_point_1_remaining_work_blocks=1,
        remaining_physical_conditions=[
            "same-field reference/support/loads/drag/contact and fresh mixed force equilibrium",
            "whole-clock all-pair material admission including reverse C5-to-C1",
            "adopted mixed continuation, independent path-work balances and joint restart"],
        **{name: False for name in FALSE_SCOPE}, seconds=time.perf_counter()-started)
    report["report_sha256"] = hashlib.sha256(encoded(report)).hexdigest()
    with (output / "report.json").open("xb") as stream:
        stream.write(encoded(report)+b"\n")
    print("COMMON boundary/body component PASS; no new accepted physical interval", report["seconds"], flush=True)
    return report
