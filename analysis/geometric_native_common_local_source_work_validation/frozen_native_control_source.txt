"""Local force/work controls using immutable registered COMMON geometry.

The genuine NONZERO body probe stays fixed. Only authentic source-internal
increments vary; the separate body-zero certificate concerns the old source
subspace, and supplies no bulk integral at a new configuration.
"""
from __future__ import annotations

from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

from tectonics.geometric_native_common_contact_ports import _encoded as encoded, _payload as payload
from analysis.geometric_native_common_material_transfer_validation.native_control import (
    material_system, query_cases as old_material_query_cases,
)
from analysis.geometric_native_common_strain_projection_validation.native_control import strain_system
from analysis.geometric_native_common_material_surface_transport_validation import native_control as transport

ROOT = Path(__file__).resolve().parents[2]
PACKET_FORMAT = "native-common-local-source-work-packet-1"
CONTROL_FORMAT = "native-common-local-source-work-control-1"
PHYSICAL_SCOPE = "LOCAL_SOURCE_FORCE_WORK_WITH_DECLARED_INTERNAL_RESIDUAL_COMPONENT_ONLY"
PRIOR_875_PACKET = transport.PRIOR_875_PACKET
PRIOR_875_PACKET_RAW = transport.PRIOR_875_PACKET_RAW
PRIOR_875_PACKET_SEAL = transport.PRIOR_875_PACKET_SEAL
PRIOR_879_PACKET = ROOT / "analysis/geometric_native_common_material_surface_transport_validation/release_validation_02/native_control/source_packet.json"
PRIOR_879_PACKET_RAW = "f2884284f663564d304537f91614efbb01e5b7df5ebdab86ac7eb6e77287ee64"
PRIOR_879_PACKET_SEAL = "368a524a954ac5f2ba3a6b92c182d34d1c3567ba7df1314c0afd02b2cff2c23e"
REMAINING_CONDITIONS = transport.REMAINING_CONDITIONS
PROVENANCE = "One local convected archival source potential with authentic internal residuals, literal beams and registered orthogonal bulk remainder; geometry-only surface receipts"


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def summary():
    return dict(physical_candidate_status=PHYSICAL_SCOPE,
        global_coordinate_count=100, owner_coordinate_count=6,
        cold_coordinate_count=56, hot_coordinate_count=30,
        explicit_authentic_source_internal_coordinate_count=8,
        local_source_atom_count=16, original_material_attachment_count=32,
        archived_original_donor_attachment_count=8,
        genuine_geometry_only_surface_binding_count=5,
        complete_local_source_query_count=4, complete_source_reference_query_count=4,
        complete_prestress_second_variation_defined=True,
        body_internal_mixed_second_variations_retained=True,
        force_hyperstress_and_internal_virtual_work_defined=True,
        joint_body_internal_atom_path_work_defined=True,
        literal_global_beams_and_registered_bulk_remainders_energized_once=True,
        old_source14_structural_body_zero_first_second_variations_certified=True,
        full_nonzero_common_body_probe_is_registered_and_fixed=True,
        local_force_work_component_changes_old875_potential=True,
        geometry_only_surface_measure_and_memory_bindings_retained=True,
        source_internal_links_reduced_into_common_body_modes=False,
        unconditional92_coordinate_reduction_impossibility_proved=False,
        old875_nonzero_source_force_identity_claimed=False,
        unknown_body_zero_bulk_hessian_supplied=False,
        physical_source_material_handover_certified=False,
        full_three_dimensional_rod_endpoints_recovered=False,
        whole_archived_three_dimensional_history_recovered=False,
        whole_face_measure_or_memory_integral_certified=False,
        current_deformed_follower_normal_contact_law_defined=False,
        old_mechanical_atom_areas_smeared_into_continuous_density=False,
        continuous_face_energy_added_to_old_source_potential=False,
        hot_receiving_contact_law_inferred=False,
        new_mixed_force_equilibrium_certified=False,
        whole_clock_all_pair_admission_authorized=False,
        physical_material_admission_authorized=False,
        source_history_advanced=False, fresh_tensor_quadrature_executed=False,
        historical_force_solve_reexecuted=False,
        accepted_original_native_intervals=1, accepted_new_native_intervals=0,
        original_point_1_remaining_work_blocks=1, original_point_1_completed=False,
        remaining_physical_conditions=list(REMAINING_CONDITIONS))


def _authenticate_prior(path, raw_pin, seal):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != raw_pin:
        raise ValueError("An immutable predecessor packet changed: " + str(path))
    packet = json.loads(raw)
    if (packet.get("packet_sha256") != seal
            or digest({k: v for k, v in packet.items() if k != "packet_sha256"}) != seal):
        raise ValueError("An immutable predecessor packet seal changed")


@lru_cache(maxsize=1)
def previous_projection():
    return strain_system()


@lru_cache(maxsize=1)
def original_material_responses():
    return tuple(transport.original_case(index, F(0), True) for index in range(32))


@lru_cache(maxsize=1)
def surface_responses():
    return tuple(transport.surface_case(index, True)[1] for index in range(5))


@lru_cache(maxsize=1)
def system():
    from tectonics import geometric_native_common_local_source_work as api
    return api.bind_native_common_local_source_work(previous_projection(),
        original_material_responses=original_material_responses(),
        surface_responses=surface_responses(), provenance=PROVENANCE, precision=96)


def evaluation_arguments():
    material = material_system()
    projection = previous_projection()
    old_queries = old_material_query_cases(material)
    # Zero and the already authenticated positive source query, both branches.
    selected = old_queries[:4]
    if (len(material.coordinate_order) != 100
            or tuple(material.coordinate_order[:92]) != tuple(projection.coordinate_order)):
        raise ValueError("The old100 catalogue must extend the exact registered92 catalogue")
    return tuple(dict(coordinate_increments=tuple(projection.bound_coordinate_increments)
            + tuple(row["coordinate_increments"][92:]), branch=row["branch"], precision=96)
        for row in selected)


@lru_cache(maxsize=1)
def evaluation_cases():
    from tectonics import geometric_native_common_local_source_work as api
    return tuple((arguments, api.evaluate_native_common_local_source_work(system(), **arguments))
        for arguments in evaluation_arguments())


def source_reference_arguments():
    material = material_system()
    return tuple(dict(source_coordinate_increments=tuple(row["coordinate_increments"][index]
                for index in material.source_support_indices),
            branch=row["branch"], precision=96)
        for row in old_material_query_cases(material)[:4])


@lru_cache(maxsize=1)
def source_reference_cases():
    from tectonics import geometric_native_common_local_source_work as api
    return tuple((arguments, api.evaluate_native_common_local_source_reference_certificate(system(), **arguments))
        for arguments in source_reference_arguments())


def replay_packet(packet):
    from tectonics import geometric_native_common_local_source_work as api
    expected = set(summary()) | {"format", "prior_875_packet_raw_sha256", "prior_875_packet_selfseal",
        "prior_879_packet_raw_sha256", "prior_879_packet_selfseal", "local_source_work_system",
        "evaluation_queries", "source_reference_queries", "packet_sha256"}
    if (type(packet) is not dict or set(packet) != expected
            or packet.get("format") != PACKET_FORMAT
            or packet.get("packet_sha256") != digest({k: v for k, v in packet.items() if k != "packet_sha256"})):
        raise ValueError("The complete local force/work packet identity changed")
    for key, value in (("prior_875_packet_raw_sha256", PRIOR_875_PACKET_RAW),
            ("prior_875_packet_selfseal", PRIOR_875_PACKET_SEAL),
            ("prior_879_packet_raw_sha256", PRIOR_879_PACKET_RAW),
            ("prior_879_packet_selfseal", PRIOR_879_PACKET_SEAL)):
        if packet[key] != value:
            raise ValueError("A predecessor packet binding changed")
    _authenticate_prior(PRIOR_875_PACKET, PRIOR_875_PACKET_RAW, PRIOR_875_PACKET_SEAL)
    _authenticate_prior(PRIOR_879_PACKET, PRIOR_879_PACKET_RAW, PRIOR_879_PACKET_SEAL)
    if encoded({key: packet[key] for key in summary()}) != encoded(summary()):
        raise ValueError("The fixed component scope changed")
    dto = api.native_common_local_source_work_system_to_dict(system())
    if encoded(packet["local_source_work_system"]) != encoded(dto):
        raise ValueError("A complete registered local work system changed")
    restored = api.native_common_local_source_work_system_from_dict(packet["local_source_work_system"],
        projection=previous_projection(), original_material_responses=original_material_responses(),
        surface_responses=surface_responses())
    if encoded(api.native_common_local_source_work_system_to_dict(restored)) != encoded(dto):
        raise ValueError("The complete system replay changed")
    for name, cases, decoder, encoder in (
            ("evaluation_queries", evaluation_cases(),
                api.native_common_local_source_work_response_from_dict,
                api.native_common_local_source_work_response_to_dict),
            ("source_reference_queries", source_reference_cases(),
                api.native_common_local_source_reference_certificate_from_dict,
                api.native_common_local_source_reference_certificate_to_dict)):
        rows = packet[name]
        if type(rows) is not list or len(rows) != 4:
            raise ValueError("Exactly four complete physical queries are required")
        for row, (arguments, response) in zip(rows, cases):
            if (type(row) is not dict or set(row) != {"arguments", "response"}
                    or encoded(row["arguments"]) != encoded(arguments)):
                raise ValueError("A force/work receipt was transplanted to another query")
            query_name = ("expected_coordinate_increments" if name == "evaluation_queries"
                else "expected_source_coordinate_increments")
            query_value = arguments[("coordinate_increments" if name == "evaluation_queries"
                else "source_coordinate_increments")]
            rebuilt = decoder(row["response"], system=restored,
                **{query_name: query_value}, expected_branch=arguments["branch"],
                expected_precision=arguments["precision"])
            if encoded(encoder(rebuilt)) != encoded(encoder(response)):
                raise ValueError("A full force/work response changed")
    return summary()


def run(out):
    from tectonics import geometric_native_common_local_source_work as api
    started = time.perf_counter()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    evaluations = [dict(arguments=payload(arguments),
        response=api.native_common_local_source_work_response_to_dict(response))
        for arguments, response in evaluation_cases()]
    references = [dict(arguments=payload(arguments),
        response=api.native_common_local_source_reference_certificate_to_dict(response))
        for arguments, response in source_reference_cases()]
    packet = dict(format=PACKET_FORMAT, **summary(),
        prior_875_packet_raw_sha256=PRIOR_875_PACKET_RAW, prior_875_packet_selfseal=PRIOR_875_PACKET_SEAL,
        prior_879_packet_raw_sha256=PRIOR_879_PACKET_RAW, prior_879_packet_selfseal=PRIOR_879_PACKET_SEAL,
        local_source_work_system=api.native_common_local_source_work_system_to_dict(system()),
        evaluation_queries=evaluations, source_reference_queries=references)
    packet["packet_sha256"] = digest(packet)
    replay_packet(packet)
    (out / "source_packet.json").write_bytes(encoded(packet)+b"\n")
    report = dict(format=CONTROL_FORMAT, status="PASS", **summary(),
        source_packet_sha256=packet["packet_sha256"],
        local_source_work_system_sha256=digest(packet["local_source_work_system"]),
        evaluation_queries_sha256=digest(evaluations), source_reference_queries_sha256=digest(references),
        seconds=time.perf_counter()-started)
    report["report_sha256"] = digest(report)
    (out / "report.json").write_bytes(encoded(report)+b"\n")
    print(encoded(report).decode(), flush=True)
    return report
