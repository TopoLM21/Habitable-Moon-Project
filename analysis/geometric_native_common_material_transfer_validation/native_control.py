"""Authentic original-material extension and one reduced source potential.

The source assembly is an explicit extension of COMMON's coordinate catalogue.
It supplies no reduction into the existing body fields or physical admission.
"""
from __future__ import annotations

from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

from tectonics.geometric_native_common_contact_ports import _encoded as encoded, _payload as payload

ROOT = Path(__file__).resolve().parents[2]
PRIOR_PACKET = ROOT / "analysis/geometric_native_common_source_carrier_validation/release_validation_02/native_control/source_packet.json"
PRIOR_PACKET_RAW = "265206430b4c864db32bce1be0a9a75b71fa180158f3d81e4f52131bebdf320c"
PRIOR_WRAPPER_RAW = "47b5c882c041f5a2c6d460c71eae11091a6217ff78a0df14af9b57f214ba633e"
PHYSICAL_SCOPE = "COMMON_ORIGINAL_MATERIAL_EXTENDED_REDUCED_SOURCE_COMPONENT_ONLY"
PROVENANCE = "Explicit same-original-material live/archive charts and independent internal source links; one archival-force potential with separate global beams"
FORCE_BASIS = "archival_old_reference"
REMAINING_CONDITIONS = (
    "Whole-source physical material and old-force/prestress/work handover, HOT law/mean chart, loads/drag and NEW mixed equilibrium",
    "Whole-clock all-pair admission including reverse C5 to C1",
    "Accepted continuation with independent path work, balances and joint restart",
)
FALSE_SCOPE = (
    "physical_source_material_handover_certified", "source_internal_links_reduced_into_common_body_modes",
    "full_three_dimensional_rod_endpoints_recovered", "continuous_measure_and_literal_atoms_energized_together",
    "hot_receiving_contact_law_inferred", "new_mixed_force_equilibrium_certified",
    "whole_clock_all_pair_admission_authorized", "physical_material_admission_authorized",
    "source_history_advanced", "original_point_1_completed",
)


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


@lru_cache(maxsize=1)
def source_context():
    """Replay the prior source DTOs, without old tests/solvers/quadratures."""
    from tectonics.geometric_native_normal_support import load_native_normal_support_source
    from tectonics.geometric_native_common_reference import native_common_reference_from_dict
    from tectonics.geometric_native_common_contact_cuts import common_contact_cuts_from_dict
    from tectonics.geometric_native_common_source_allocation import native_common_source_allocation_from_dict
    from tectonics.geometric_native_common_source_carrier import native_common_source_carrier_system_from_dict
    raw = PRIOR_PACKET.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PRIOR_PACKET_RAW:
        raise ValueError("The immutable867 source packet changed")
    packet = json.loads(raw)
    if packet["packet_sha256"] != digest({k: v for k, v in packet.items() if k != "packet_sha256"}):
        raise ValueError("The immutable867 source packet seal changed")
    source = load_native_normal_support_source()
    reference = native_common_reference_from_dict(packet["common_reference"], source=source)
    cuts = common_contact_cuts_from_dict(packet["cut_set"], reference=reference)
    allocation = native_common_source_allocation_from_dict(packet["allocation"], reference=reference, cuts=cuts)
    carrier = native_common_source_carrier_system_from_dict(packet["carrier_system"], allocation=allocation,
        reference=reference, cuts=cuts)
    return source, reference, cuts, allocation, carrier


@lru_cache(maxsize=1)
def material_system():
    from tectonics import geometric_native_common_material_transfer as api
    _, reference, _, _, carrier = source_context()
    return api.bind_native_common_material_transfer(carrier, reference=reference,
        force_basis=FORCE_BASIS, provenance=PROVENANCE, precision=96)


def query_cases(system):
    zero = (F(0),) * len(system.coordinate_order)
    increments = []
    for scale in (F(1, 1000000), F(-1, 2000000)):
        values = list(zero)
        for index, coordinate in enumerate(system.coordinate_order):
            if coordinate[0] == "source_internal":
                values[index] = scale if coordinate[-1] == "sub" else -scale/2
        if not any(values):
            raise ValueError("The eight internal source channels must be explicit")
        increments.append(tuple(values))
    return tuple(dict(coordinate_increments=values, branch=branch, precision=96)
        for values in (zero, *increments)
        for branch in ("relaxed_zero_strength_shear", "frozen_elastic_predictor"))


def material_query_cases(system):
    from tectonics.geometric_native_pose import FractionInterval as I
    return tuple(dict(point_id=a.point_id, side=a.side,
        material_depth_bounds_m=I(depth), precision=96)
        for a in system.attachments for depth in (F(0), a.areal_material_thickness_m/2))


def summary(count):
    return dict(physical_candidate_status=PHYSICAL_SCOPE,
        original_material_attachment_count=32, original_source_mechanism_count=16,
        archived_original_donor_attachment_count=8, live_original_donor_attachment_count=8,
        live_original_receiver_attachment_count=16, extended_coordinate_count=100,
        declared_common_coordinate_count=92, independent_internal_source_coordinate_count=8,
        source_coordinate_count=14, common_body_coordinates_without_source_reduction=86,
        independent_global_beam_count=8, complete_global_query_count=count,
        complete_material_point_query_count=64,
        chosen_force_basis=FORCE_BASIS, source_material_identity_preserved=True,
        one_global_source_embedding_defined=True, separate_global_beam_energized_once=True,
        original_source_work_lineages_retained=True, literal_source_interface_measures_retained=True,
        accepted_original_native_intervals=1, accepted_new_native_intervals=0,
        original_point_1_remaining_work_blocks=1, remaining_physical_conditions=list(REMAINING_CONDITIONS),
        **{name: False for name in FALSE_SCOPE})


def replay_packet(packet):
    from tectonics import geometric_native_common_material_transfer as api
    expected_keys = set(summary(6)) | {"format", "prior_867_packet_raw_sha256",
        "prior_867_wrapper_raw_sha256", "material_transfer", "global_queries",
        "material_point_queries", "packet_sha256"}
    if (type(packet) is not dict or set(packet) != expected_keys
            or packet.get("packet_sha256") != digest({k: v for k, v in packet.items() if k != "packet_sha256"})):
        raise ValueError("The complete reduced-source packet schema or selfseal changed")
    if (packet.get("format") != "native-common-original-material-reduced-source-packet-1"
            or packet.get("prior_867_packet_raw_sha256") != PRIOR_PACKET_RAW
            or packet.get("prior_867_wrapper_raw_sha256") != PRIOR_WRAPPER_RAW):
        raise ValueError("The original material transfer packet identity changed")
    _, reference, _, _, carrier = source_context()
    system = api.native_common_material_transfer_from_dict(packet["material_transfer"],
        carrier_system=carrier, reference=reference)
    if system.provenance != PROVENANCE or system.force_basis != FORCE_BASIS or system.precision != 96:
        raise ValueError("The declared reduced source model changed")
    expected = query_cases(system)
    rows = packet["global_queries"]
    if len(rows) != len(expected):
        raise ValueError("The complete six global reduced-source controls are required")
    for row, arguments in zip(rows, expected):
        if type(row) is not dict or set(row) != {"arguments", "response"} or row["arguments"] != payload(arguments):
            raise ValueError("The literal global query changed")
        result = api.native_common_material_transfer_response_from_dict(row["response"], system=system)
        if (result.coordinate_increments != arguments["coordinate_increments"]
                or result.branch != arguments["branch"] or result.precision != arguments["precision"]):
            raise ValueError("A genuine reduced response was transplanted onto another query")
        if api.native_common_material_transfer_response_to_dict(result) != row["response"]:
            raise ValueError("The complete reduced mechanical response changed")
    points = packet["material_point_queries"]
    expected_points = material_query_cases(system)
    if len(points) != len(expected_points):
        raise ValueError("All sixty-four original-material depth queries are required")
    for row, arguments in zip(points, expected_points):
        if type(row) is not dict or set(row) != {"arguments", "response"} or row["arguments"] != payload(arguments):
            raise ValueError("An original-material label or query depth changed")
        point = api.native_common_material_point_from_dict(row["response"], system=system)
        if (point.point_id != arguments["point_id"] or point.side != arguments["side"]
                or point.material_depth_bounds_m != arguments["material_depth_bounds_m"]
                or point.precision != arguments["precision"]):
            raise ValueError("A genuine material point was transplanted onto another query")
        if api.native_common_material_point_to_dict(point) != row["response"]:
            raise ValueError("The complete original-material depth response changed")
    result = summary(len(rows))
    if any(packet.get(k) != v for k, v in result.items()):
        raise ValueError("Material identities, work ownership or physical scope changed")
    return result


def run(out):
    from tectonics import geometric_native_common_material_transfer as api
    started = time.perf_counter()
    target = Path(out); target.mkdir(parents=True, exist_ok=False)
    system = material_system()
    rows = []
    for index, arguments in enumerate(query_cases(system), 1):
        response = api.evaluate_native_common_material_transfer(system, **arguments)
        rows.append(dict(arguments=payload(arguments),
            response=api.native_common_material_transfer_response_to_dict(response)))
        print("GLOBAL original-material source query", index, "PASS", flush=True)
    points = []
    for index, arguments in enumerate(material_query_cases(system), 1):
        point = api.evaluate_native_common_material_point(system, **arguments)
        points.append(dict(arguments=payload(arguments), response=api.native_common_material_point_to_dict(point)))
        if index % 8 == 0:
            print("ORIGINAL material depth query", index, "PASS", flush=True)
    result = summary(len(rows))
    packet = dict(format="native-common-original-material-reduced-source-packet-1",
        prior_867_packet_raw_sha256=PRIOR_PACKET_RAW, prior_867_wrapper_raw_sha256=PRIOR_WRAPPER_RAW,
        material_transfer=api.native_common_material_transfer_to_dict(system), global_queries=rows,
        material_point_queries=points, **result)
    packet["packet_sha256"] = digest(packet)
    (target/"source_packet.json").write_bytes(encoded(packet)+b"\n")
    report = dict(format="native-common-original-material-reduced-source-control-1", status="PASS",
        source_packet_sha256=packet["packet_sha256"], global_queries_sha256=digest(rows),
        material_point_queries_sha256=digest(points),
        material_transfer_sha256=system.binding_sha256, seconds=time.perf_counter()-started, **result)
    report["report_sha256"] = digest(report)
    (target/"report.json").write_bytes(encoded(report)+b"\n")
    print("ORIGINAL material reduced-source assembly PASS; physical body reduction remains open", flush=True)
    return report
