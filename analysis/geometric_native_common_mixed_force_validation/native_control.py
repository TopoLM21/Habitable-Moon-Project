"""Declared NEW100 endpoint force assembly from full frozen authorities.

Only fresh material-label geometry and the new assembly run. The cold/HOT
integrals, previous controls, historical force solves and clocks are not
reexecuted. Original source atoms do not supply born COLD--HOT contact laws.
"""
from __future__ import annotations

from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
PHYSICAL_SCOPE = "DECLARED_ORIGINAL_SOURCE_ATOM_HUNDRED_COORDINATE_ENDPOINT_ASSEMBLY_ONLY"
PROVENANCE = ("Full frozen cold899/HOT903 operators once, NEW shared eight-tip internal "
              "coupling, original sixteen material-attached source atoms on full895 "
              "cold fields with unit Nanson normals and explicit zero-reference "
              "traction calibration, NEW frozen owner6 load/drag potential")
OPTIONS = dict(precision=96)
AUTHORITIES = (
    ("analysis/geometric_native_common_hot_mean_validation/release_validation_02/native_control/source_packet.json",
     "0f8a0ee8ae082e3434b136bab7515944354eaf059db95965d0e87259f6791035",
     "c53e4cd75219328eeac974f3b9982aa33ad78eaf2e07d823feafb9f07b289e6c"),
    ("analysis/geometric_native_common_cold_continuum_energy_validation/release_validation_01/native_control/source_packet.json",
     "4a5c059176cef063bf12ff7bd98df830395504d6c12051f269a177fcbc375fec",
     "c863be05c9d641025666a1a4c3929701c7d472f9d574085926d64a2d08b4c439"),
    ("analysis/geometric_native_common_carrier_source_work_validation/release_validation_02/native_control/source_packet.json",
     "b45d5d4d80c74c509bfac033c4a60c5b96c8327704e0c727fcf680406d82e57e",
     "1f1b07b977bbb092e08281d8d200a26f8bf9ab85e2a18f001b2061275f742c40"),
    ("analysis/geometric_native_common_material_surface_transport_validation/release_validation_02/native_control/source_packet.json",
     "f2884284f663564d304537f91614efbb01e5b7df5ebdab86ac7eb6e77287ee64",
     "368a524a954ac5f2ba3a6b92c182d34d1c3567ba7df1314c0afd02b2cff2c23e"),
)
GEOMETRY_AUTHORITIES = (
    ("analysis/geometric_native_common_cold_material_carrier_validation/release_validation_02/native_control/source_packet.json",
     "fbafd092a4eb98472e60f79665e1e0fe8e9e1c2c2b9d3679344a32e59593cc89",
     "27cbc05b687509fa22c00aa4339dd66d461c4def73ed37d4eae3acd62847da70"),
    ("analysis/geometric_native_archive_material_carrier_validation/release_validation_02/native_control/source_packet.json",
     "6d34f8acd0a5ba66221117277e7acd496689f5858e26320d06fd705918e0c8a7",
     "6833539d33c2b29a7ab1d884d4536727849bbdd58d8c819f46fdf90ecf6c7063"),
    ("analysis/geometric_native_common_reference_validation/release_validation_01/native_control/source_packet.json",
     "dc8f820dbd926c53223a5c141a10296d1c7135efc6bf1077763785866dda9275",
     "60259e6e56bb7335fa994988adfbbac5ac933b524ca164b0298c2e0a16a40446"),
)
MATERIAL_DIRECTORY = "analysis/geometric_native_first_entry_validation/control_release_02"
MATERIAL_PINS = {
    "material_state.json": "3d24eee293b4517c35359ae3c31feaac5b43a5ee066209f8faaf65457b35dfc3",
    "full_domain_journal.json": "8da581b103945cc85889d3fe8ac8a70361994d33c3f2dd6b31e917cc3273fb82",
}
REMAINING_CONDITIONS = (
    "Whole-source physical material and old-force/prestress/work handover, HOT law/mean chart, loads/drag and NEW mixed equilibrium",
    "Whole-clock all-pair admission including reverse C5 to C1",
    "Accepted continuation with independent path work, balances and joint restart",
)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def _read_authority(authority):
    path, raw_pin, seal = authority
    raw = (ROOT / path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != raw_pin:
        raise ValueError("The complete immutable force authority changed: " + path)
    packet = json.loads(raw)
    if (packet["packet_sha256"] != seal or
            digest({k: v for k, v in packet.items() if k != "packet_sha256"}) != seal):
        raise ValueError("The full force authority selfseal changed: " + path)
    return packet


@lru_cache(maxsize=1)
def source_packets():
    return tuple(_read_authority(authority) for authority in AUTHORITIES)


@lru_cache(maxsize=1)
def geometry_packets():
    return tuple(_read_authority(authority) for authority in GEOMETRY_AUTHORITIES)


@lru_cache(maxsize=1)
def source_state():
    """Public geometry/material restoration, with no old control imports."""
    from tectonics.geometric_native_domain_journal import native_domain_journal_from_dict
    from tectonics.geometric_native_first_entry_material import native_first_entry_material_from_dict
    documents = {}
    for name, pin in MATERIAL_PINS.items():
        raw = (ROOT / MATERIAL_DIRECTORY / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != pin:
            raise ValueError("The original material/journal authority changed")
        documents[name] = json.loads(raw)
    material = documents["material_state.json"]
    if (encoded(material) != encoded(geometry_packets()[1]["source_material_state"])
            or encoded(material) != encoded(json.loads(source_packets()[3][
                "material_transfer"]["complete_original_material_state_json"]))):
        raise ValueError("The complete original material authority changed")
    journal = native_domain_journal_from_dict(documents["full_domain_journal.json"])
    return native_first_entry_material_from_dict(material, expected_journal=journal,
        expected_original_mechanics_payload=material["original_mechanics_payload"])


@lru_cache(maxsize=1)
def archive_carrier():
    from tectonics.geometric_native_archive_material_carrier import native_archive_material_carrier_from_dict
    return native_archive_material_carrier_from_dict(geometry_packets()[1]["carrier_system"],
        expected_source_material_state=source_state())


@lru_cache(maxsize=1)
def cold_carrier():
    from tectonics.geometric_native_common_cold_material_carrier import native_common_cold_material_carrier_from_dict
    return native_common_cold_material_carrier_from_dict(geometry_packets()[0][
        "cold_material_carrier_system"], material_transport_packet=source_packets()[3],
        source_work_packet=source_packets()[2], archive_carrier=archive_carrier())


@lru_cache(maxsize=1)
def owner_drag_time_scale():
    reference = geometry_packets()[2]
    literal = reference["before"]["original_physical_history"]["literal_dt_s"]
    if type(literal) is not float or literal != 1000000.0:
        raise ValueError("The literal source loading time scale changed")
    # A scale in a NEW frozen increment potential, not a new elapsed interval.
    return F.from_float(literal)


@lru_cache(maxsize=1)
def system():
    from tectonics.geometric_native_common_mixed_force import bind_native_common_mixed_force
    return bind_native_common_mixed_force(*source_packets(), cold_carrier=cold_carrier(),
        provenance=PROVENANCE, owner_drag_time_scale_s=owner_drag_time_scale(), **OPTIONS)


@lru_cache(maxsize=16)
def response(coordinates=None):
    from tectonics.geometric_native_common_mixed_force import evaluate_native_common_mixed_force
    return evaluate_native_common_mixed_force(system(), coordinates, **OPTIONS)


def query_coordinates():
    hot, cold, work, transport = source_packets()
    result = []
    for cq in cold["evaluation_queries"]:
        c = tuple(F(*x) for x in cq["response"]["coordinate_increments"])
        for hq in hot["evaluation_queries"]:
            h = tuple(F(*x) for x in hq["response"]["coordinate_increments"])
            result.append((cq["label"] + "__" + hq["label"] + "__zero_internal",
                           (F(0),)*6 + h + c + (F(0),)*8))
    positive = tuple(F((-1 if i % 2 else 1)*(i+1), 1000000) for i in range(8))
    for cold_index, hot_index, sign in ((0, 1, 1), (0, 2, -1), (1, 3, 1), (1, 0, -1)):
        cq = cold["evaluation_queries"][cold_index]
        hq = hot["evaluation_queries"][hot_index]
        c = tuple(F(*x) for x in cq["response"]["coordinate_increments"])
        h = tuple(F(*x) for x in hq["response"]["coordinate_increments"])
        result.append((cq["label"] + "__" + hq["label"] + "__new_internal_" + str(sign),
                       (F(0),)*6 + h + c + tuple(sign*x for x in positive)))
    return tuple(result)


def summary():
    return dict(physical_candidate_status=PHYSICAL_SCOPE,
        complete_declared_endpoint_force_assembly_certified=True,
        complete_full_physical_mixed_force_operator_certified=False,
        complete_cold_continuum_operator_certified=True,
        complete_hot_material_constitutive_operator_certified=True,
        finite_hot_base_mean_chart_certified=True,
        whole_cold_material_support_geometry_certified=True,
        source_atom_current_deformed_follower_normal_law_defined=True,
        born_hot_current_cut_receiving_law_defined=False,
        whole_modal_basal_loading_defined=False,
        physical_source_force_prestress_work_handover_certified=False,
        historical_archived_three_dimensional_prestress_recovered=False,
        prepared_reference_full_system_work_certified=False,
        reference_change_full_system_work_certified=False,
        hot_constitutive_law_defined=True, finite_mean_chart_defined=True,
        finite_mean_equilibrium_certified=False,
        new_mixed_force_equilibrium_certified=False,
        physical_material_admission_authorized=False,
        whole_clock_all_pair_admission_authorized=False,
        full_system_temporal_path_work_certified=False,
        clocks_advanced=False, proposed_source_damage_adopted=False,
        historical_force_solve_reexecuted=False, previous_tensor_quadrature_reexecuted=False,
        source_geometry_elapsed_time_s=0, contact_geometry_elapsed_time_s=2000000,
        memory_elapsed_time_s=1000000, material_geometry_elapsed_time_s=1000000,
        accepted_original_native_intervals=1, accepted_new_native_intervals=0,
        original_point_1_remaining_work_blocks=1, original_point_1_completed=False,
        remaining_physical_conditions=list(REMAINING_CONDITIONS))


def _verify_authority(authority, coordinates, bound):
    """Authenticate the complete shared geometry before counting its labels."""
    if (authority["system_binding_sha256"] != bound.binding_sha256
            or authority["cold_coordinate_increments"] != [[x.numerator, x.denominator] for x in coordinates]
            or digest({k: v for k, v in authority.items() if k != "binding_sha256"}) != authority["binding_sha256"]):
        raise ValueError("The complete contact field authority changed its source/query binding")
    expected = dict(original_material_point_count=32, original_live_attachment_count=24,
        original_archived_attachment_count=8, original_atom_count=16,
        source_original_atom_area_measure_used_once=True, current_director_is_unit_nanson_normal=True,
        old_work_or_source_memory_advanced=False, current_continuous_face_contact_law_defined=False,
        physical_material_admission_authorized=False)
    if any(type(authority[k]) is not type(v) or authority[k] != v for k, v in expected.items()):
        raise ValueError("The contact geometry authority changed its declared scope")
    points = authority["point_receipts"]
    atoms = authority["original_atom_authorities"]
    pointmap = {(p["point_id"], p["side"]): p for p in points}
    if len(points) != 32 or len(pointmap) != 32 or len(atoms) != 16 or len({a["point_id"] for a in atoms}) != 16:
        raise ValueError("Every original sixteen atom and thirty-two material sides must occur once")
    allocations = {"live": 0, "archived": 0}
    for point in points:
        current, zero = point["current"], point["prepared_zero"]
        field = point["original_saved_attachment"]["field"]
        allocation = "archived" if field["original_label_membership"] == "ARCHIVED_ORIGINAL_MATERIAL" else "live"
        allocations[allocation] += 1
        for actual in (current, zero):
            if (actual["allocation"] != allocation or actual["archive_debits"] != int(allocation == "archived")
                    or actual["original_parent_id"] != field["original_parent_id"]
                    or actual["original_material_id"] != field["original_material_id"]
                    or actual["owner"] != field["owner"] or actual["material_membership_authenticated"] is not True
                    or F(*actual["volume_jacobian_bounds"]["lo"]) <= 0):
                raise ValueError("A complete attachment lost its original material/allocation or positive Jacobian")
    if allocations != {"live": 24, "archived": 8}:
        raise ValueError("Actual complete source allocations differ from twenty-four live/eight archived labels")
    return pointmap, {a["point_id"]: a for a in atoms}


def _verify_response(evaluated, coordinates, bound, authority, api):
    """Check actual dimensions, units, receipts and scope before declaring PASS."""
    if (evaluated.coordinate_increments != coordinates or evaluated.coordinate_order != bound.coordinate_order
            or len(coordinates) != 100 or len(evaluated.gradient_bounds) != 100
            or len(evaluated.hessian_bounds) != 100 or any(len(row) != 100 for row in evaluated.hessian_bounds)
            or evaluated.coordinate_units != bound.coordinate_units
            or evaluated.gradient_units != bound.gradient_units or evaluated.hessian_units != bound.hessian_units
            or evaluated.system_binding_sha256 != bound.binding_sha256
            or evaluated.contact_authority_binding_sha256 != authority["binding_sha256"]):
        raise ValueError("A complete100 mixed force response changed its query, dimensions, units or authority")
    if any(evaluated.hessian_bounds[i][j] != evaluated.hessian_bounds[j][i] for i in range(100) for j in range(100)):
        raise ValueError("The full100 declared potential tangent is not symmetric")
    expected = dict(whole_cold_parent_count=8, whole_hot_body_count=5, source_atom_count=16,
        source_material_point_count=32, source_internal_beam_count=8,
        complete_declared_endpoint_force_assembly_certified=True, cold_beams_included_once=True,
        internal_residuals_shift_existing_beams_once=True, old891_complete_beam_bulk_operator_added_again=False,
        born_hot_receiving_contact_law_certified=False, whole_physical_mixed_operator_certified=False,
        new_mixed_force_equilibrium_certified=False, finite_mean_equilibrium_certified=False,
        physical_source_force_prestress_work_handover_certified=False,
        historical_archived_three_dimensional_prestress_recovered=False,
        prepared_reference_full_system_work_certified=False, physical_material_admission_authorized=False,
        whole_clock_all_pair_admission_authorized=False, full_system_temporal_path_work_certified=False,
        accepted_original_native_intervals=1, accepted_new_native_intervals=0,
        source_geometry_elapsed_time_s=F(0), contact_geometry_elapsed_time_s=F(2000000),
        memory_elapsed_time_s=F(1000000), material_geometry_elapsed_time_s=F(1000000),
        clocks_advanced=False, original_point_1_completed=False, original_point_1_remaining_work_blocks=1,
        remaining_physical_conditions=REMAINING_CONDITIONS, endpoint_owner_chart_is_exact_zero=True,
        source_body_operator_queries_reexecuted=False, original_material_atom_measure_used_once=True,
        current_unit_nanson_normal_follower_component_defined=True, scope=PHYSICAL_SCOPE)
    if any(type(getattr(evaluated, k)) is not type(v) or getattr(evaluated, k) != v for k, v in expected.items()):
        raise ValueError("The complete response changed its declared physics/clock boundary")
    pointmap, atommap = _verify_authority(authority, coordinates[36:92], bound)
    references = [json.loads(raw) for raw in evaluated.source_point_receipts_json]
    if len(references) != 32 or {(r["point_id"], r["side"]) for r in references} != set(pointmap):
        raise ValueError("The response must reference every full authentic source material side")
    for receipt in references:
        point = pointmap[receipt["point_id"], receipt["side"]]
        if (receipt["complete_point_receipt_sha256"] != digest(point)
                or receipt["current_point_binding_sha256"] != point["current"]["binding_sha256"]
                or receipt["prepared_zero_point_binding_sha256"] != point["prepared_zero"]["binding_sha256"]
                or receipt["contact_authority_binding_sha256"] != authority["binding_sha256"]):
            raise ValueError("A source point reference changed its complete current/zero material authority")
    atoms = [json.loads(raw) for raw in evaluated.contact_atoms_json]
    if len(atoms) != 16 or {a["point_id"] for a in atoms} != set(atommap) or len(evaluated.internal_beam_corrections_json) != 8:
        raise ValueError("Every original source atom and each of the eight shared beam corrections is required")
    for atom in atoms:
        if encoded(json.loads(atom["original_atom_source_binding_json"])) != encoded(atommap[atom["point_id"]]["original_atom"]):
            raise ValueError("A declared contact changed its complete original source atom")
        for key, value in dict(original_material_nanson_normal_law_defined=True,
                normalized_cofactor_direction_and_full_second_derivatives_included=True,
                orientation_hyperstress_virtual_work_included=True, prepared_zero_source_traction_calibration=True,
                original_source_area_included_once=True, current_area_energy_multiplier_added=False,
                current_continuous_face_contact_law_defined=False, born_hot_receiving_law_inferred=False,
                proposed_normal_damage_or_fracture_work_applied=False, source_state_or_work_advanced=False).items():
            if atom[key] is not value:
                raise ValueError("An original contact changed its declared law/history boundary: " + key)
        residuals = atom["action_reaction_residual_bounds_n"] + atom["global_moment_residual_bounds_nm"]
        residuals += [v for i, v in atom["virtual_work_residual_bounds"]]
        if any(not api._interval(v).lo <= 0 <= api._interval(v).hi for v in residuals):
            raise ValueError("A complete material contact lost force, moment or orientation virtual-work closure")


def run(output_dir):
    from tectonics import geometric_native_common_mixed_force as api
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    bound = system()
    dto = api.native_common_mixed_force_to_dict(bound)
    packet = dict(format="native-common-mixed-force-assembly-packet-1", **summary(),
        source_authority_raw_sha256={p: raw for p, raw, seal in AUTHORITIES+GEOMETRY_AUTHORITIES},
        source_authority_packet_sha256={p: seal for p, raw, seal in AUTHORITIES+GEOMETRY_AUTHORITIES},
        material_source_document_raw_sha256={MATERIAL_DIRECTORY + "/" + p: raw
            for p, raw in MATERIAL_PINS.items()}, mixed_force_system=dto,
        source_literal_dt_s=1000000.0,
        owner_drag_time_scale_interpretation="NEW_FROZEN_INCREMENT_POTENTIAL_SCALE_NOT_ELAPSED_TIME",
        original_source_contact_graph_has_no_hot_attachments=True,
        cold_hot_force_integrals_reexecuted=False, evaluation_queries=[],
        contact_field_authorities=[])
    authorities = {}
    for cq in source_packets()[1]["evaluation_queries"]:
        coordinates = tuple(F(*x) for x in cq["response"]["coordinate_increments"])
        authority = api.native_common_mixed_force_contact_authority_to_dict(bound, coordinates)
        _verify_authority(authority, coordinates, bound)
        authorities[coordinates] = authority
        packet["contact_field_authorities"].append(dict(label=cq["label"],
            cold_coordinate_increments=api._payload(coordinates), authority=authority))
    queries = query_coordinates()
    if len(queries) != 12 or len({coordinates for label, coordinates in queries}) != 12 or len(authorities) != 2:
        raise ValueError("All eight frozen body pairings and four nonzero internal probes are required")
    for label, coordinates in queries:
        evaluated = response(coordinates)
        _verify_response(evaluated, coordinates, bound, authorities[coordinates[36:92]], api)
        exported = api.native_common_mixed_force_response_to_dict(evaluated)
        packet["evaluation_queries"].append(dict(label=label,
            arguments=dict(coordinate_increments=api._payload(coordinates), **OPTIONS),
            response=exported))
    restored = api.native_common_mixed_force_from_dict(dto, *source_packets(),
        cold_carrier=cold_carrier())
    if encoded(api.native_common_mixed_force_to_dict(restored)) != encoded(dto):
        raise ValueError("The complete declared assembly restart changed")
    # Rebuild only the new assembly from authentic saved constitutive operators.
    row = packet["evaluation_queries"][-1]
    coordinates = query_coordinates()[-1][1]
    replay = api.native_common_mixed_force_response_from_dict(row["response"], system=restored,
        expected_coordinate_increments=coordinates, **OPTIONS)
    if encoded(api.native_common_mixed_force_response_to_dict(replay)) != encoded(row["response"]):
        raise ValueError("The full100 assembly response restart changed")
    packet["complete_declared_mixed_assembly_restart_verified"] = True
    packet["complete_actual_cold_source_attachment_count"] = 32
    packet["complete_original_source_atom_count"] = 16
    packet["packet_sha256"] = digest(packet)
    (output / "source_packet.json").write_bytes(encoded(packet)+b"\n")
    report = dict(format="native-common-mixed-force-assembly-control-1", status="PASS", **summary(),
        complete_declared_mixed_assembly_restart_verified=True,
        complete_assembled_coordinate_count=100, complete_cold_coordinate_count=56,
        complete_hot_coordinate_count=30, owner_coordinate_count=6,
        source_internal_coordinate_count=8, source_atom_count=16,
        actual_material_attachment_count=32, live_attachment_count=24,
        archived_attachment_count=8, source_beam_count=8,
        full_cold_hot_saved_pairing_count=8, full_assembly_query_count=len(packet["evaluation_queries"]),
        complete_source_field_state_count=len(packet["contact_field_authorities"]),
        cold_hot_force_integrals_reexecuted=False,
        source_packet_sha256=packet["packet_sha256"],
        source_packet_raw_sha256=hashlib.sha256((output / "source_packet.json").read_bytes()).hexdigest(),
        elapsed_seconds=time.monotonic()-started)
    report["report_sha256"] = digest(report)
    (output / "report.json").write_bytes(encoded(report)+b"\n")
    print(json.dumps(report, sort_keys=True))
    return report
