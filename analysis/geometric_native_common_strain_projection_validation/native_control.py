"""One whole-coordinate reduction at eight authenticated NONZERO cold probes.

Only the sealed completed COMMON841 tensor integrals are replayed.  Neither
volume quadrature, an old force solve, nor a historical constitutive step is
executed here.  The bankwise strain projection is a component model; it does
not certify the original material faces or adopt a new mixed interval.
"""
from __future__ import annotations

from dataclasses import fields
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

from tectonics.geometric_native_common_contact_ports import _encoded as encoded, _payload as payload
from tectonics.geometric_native_pose import FractionInterval as I
from analysis.geometric_native_common_material_transfer_validation.native_control import (
    material_system as prior_material_system, source_context,
)

ROOT = Path(__file__).resolve().parents[2]
PRIOR_871_PACKET = ROOT / "analysis/geometric_native_common_material_transfer_validation/release_validation_01/native_control/source_packet.json"
PRIOR_871_PACKET_RAW = "a9fcc7874c0e68940e20ded658748300f55d9f1d296db5c9d67b59269ac8b3ed"
PRIOR_871_PACKET_SEAL = "cd1e2457a857b47891cd40a89219abfac3aa2228028aa9636f0e77f7a10082dd"
PRIOR_871_WRAPPER_RAW = "44d12b3fb10edfef90a83c50385278b45135d1b051a5733d2b2f7d98f2755b56"
PRIOR_841_PACKET = ROOT / "analysis/geometric_native_common_reference_validation/release_validation_01/native_control/source_packet.json"
PRIOR_841_PACKET_RAW = "dc8f820dbd926c53223a5c141a10296d1c7135efc6bf1077763785866dda9275"
PRIOR_841_PACKET_SEAL = "60259e6e56bb7335fa994988adfbbac5ac933b524ca164b0298c2e0a16a40446"
PHYSICAL_SCOPE = "COMMON_BANKWISE_COLD_STRAIN_PROJECTION_REDUCTION_COMPONENT_ONLY"
PROVENANCE = "One 92-coordinate bankwise cold strain projection of the authentic archival source potential; eight sealed COMMON841 NONZERO tensor probes and orthogonal remainders counted once"
FORCE_BASIS = "archival_old_reference"
REMAINING_CONDITIONS = (
    "Whole-source physical material and old-force/prestress/work handover, HOT law/mean chart, loads/drag and NEW mixed equilibrium",
    "Whole-clock all-pair admission including reverse C5 to C1",
    "Accepted continuation with independent path work, balances and joint restart",
)
FALSE_SCOPE = (
    "physical_source_material_handover_certified", "local_common_face_source_coupling_certified",
    "hot_source_contact_coupling_defined", "full_three_dimensional_archival_strain_field_recovered",
    "hot_receiving_contact_law_inferred", "new_mixed_force_equilibrium_certified",
    "whole_clock_all_pair_admission_authorized", "physical_material_admission_authorized",
    "source_history_advanced", "original_point_1_completed", "fresh_tensor_quadrature_executed",
    "historical_force_solve_reexecuted", "calibrated_common_beam_energy_added_again",
    "independent_source_internal_coordinates_retained",
)


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def _sealed_source(path, raw_sha256, selfseal):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != raw_sha256:
        raise ValueError("An immutable completed source packet changed")
    packet = json.loads(raw)
    if (packet.get("packet_sha256") != selfseal
            or digest({k: v for k, v in packet.items() if k != "packet_sha256"}) != selfseal):
        raise ValueError("An immutable completed source packet selfseal changed")
    return packet


@lru_cache(maxsize=1)
def material_system():
    from tectonics import geometric_native_common_material_transfer as material
    packet = _sealed_source(PRIOR_871_PACKET, PRIOR_871_PACKET_RAW, PRIOR_871_PACKET_SEAL)
    system = prior_material_system()
    if material.native_common_material_transfer_to_dict(system) != packet["material_transfer"]:
        raise ValueError("The authentic completed871 reduced source model was changed")
    return system


def _physical_reference_payload(value):
    """Remove only declared precision and precision-derived binding IDs."""
    result = dict(value)
    result.pop("geometry_enclosure_precision")
    result.pop("reference_snapshot_sha256")
    result["banks"] = [{k: v for k, v in bank.items()
        if k not in ("reference_snapshot_sha256", "bank_snapshot_sha256")}
        for bank in result["banks"]]
    return result


@lru_cache(maxsize=1)
def closed_common_context():
    from tectonics.geometric_native_common_reference import (
        native_common_reference_to_dict, native_common_reference_from_dict,
    )
    packet = _sealed_source(PRIOR_841_PACKET, PRIOR_841_PACKET_RAW, PRIOR_841_PACKET_SEAL)
    source, current_reference, *_ = source_context()
    current_dto = native_common_reference_to_dict(current_reference)
    closed_dto = packet["common_reference"]
    current_payload, closed_payload = current_dto["reference"], closed_dto["reference"]
    if (packet.get("format") != "native-common-reference-component-source-packet-1"
            or packet["source_snapshot_sha256"] != source.source_snapshot_sha256
            or {k: v for k, v in closed_dto.items() if k != "reference"}
                != {k: v for k, v in current_dto.items() if k != "reference"}
            or _physical_reference_payload(closed_payload) != _physical_reference_payload(current_payload)
            or closed_payload["geometry_enclosure_precision"] != 160
            or current_payload["geometry_enclosure_precision"] != 96):
        raise ValueError("Closed841 and current871 differ in full physical reference inputs")
    closed_reference = native_common_reference_from_dict(closed_dto, source=source)
    rows = packet["cold_components"]
    actual = tuple(b.component_id for b in source.banks if b.material_kind == "original_live")
    if (type(rows) is not list or len(rows) != 8
            or tuple(row.get("component_id") for row in rows) != actual):
        raise ValueError("Exactly the eight genuine original live cold probes are required")
    equivalence = dict(format="common-complete-physical-input-precision-equivalence-1",
        closed_reference_snapshot_sha256=closed_reference.reference_snapshot_sha256,
        current_reference_snapshot_sha256=current_reference.reference_snapshot_sha256,
        closed_geometry_enclosure_precision=160, current_geometry_enclosure_precision=96,
        full_closed_reference_dto_sha256=digest(closed_dto),
        full_current_reference_dto_sha256=digest(current_dto),
        complete_shared_physical_reference_sha256=digest(_physical_reference_payload(closed_payload)),
        physical_definition_bank_count=13,
        compared_all_reference_and_bank_fields_except_declared_precision_and_binding_ids=True,
        excluded_reference_fields=["geometry_enclosure_precision", "reference_snapshot_sha256"],
        excluded_bank_fields=["bank_snapshot_sha256", "reference_snapshot_sha256"],
        original_closed_context_rewritten=False, fresh_tensor_quadrature_executed=False)
    equivalence["equivalence_sha256"] = digest(equivalence)
    return closed_reference, tuple(rows), equivalence


def closed_cold_records():
    """Full closed rows remain attached to the original160 reference."""
    return closed_common_context()[1]


def reference_precision_equivalence():
    return closed_common_context()[2]


def _scalar(value):
    if type(value) is dict and set(value) == {"lo", "hi"}:
        return I(_fraction(value["lo"]), _fraction(value["hi"]))
    if type(value) is list:
        if len(value) == 2 and all(type(x) is int for x in value):
            return _fraction(value)
        return tuple(_scalar(x) for x in value)
    if type(value) in (str, int, bool) or value is None:
        return value
    raise ValueError("A closed tensor receipt contains a nonliteral scalar")


def _fraction(value):
    if (type(value) is not list or len(value) != 2
            or any(type(x) is not int for x in value) or value[1] <= 0):
        raise ValueError("A closed tensor coordinate must retain its exact Fraction")
    result = F(*value)
    if [result.numerator, result.denominator] != value:
        raise ValueError("A closed tensor Fraction must retain its canonical literal pair")
    return result


def _cell_receipt(value):
    from tectonics.geometric_native_cold_tensor_integration import (
        ColdTensorCellReceipt, ColdTensorCorrectionReceipt, ColdTensorEndpointStripReceipt,
    )
    if type(value) is dict:
        for kind in (ColdTensorCellReceipt, ColdTensorCorrectionReceipt, ColdTensorEndpointStripReceipt):
            if set(value) == {item.name for item in fields(kind)}:
                return kind(**{name: _scalar(scalar) for name, scalar in value.items()})
    raise ValueError("A closed tensor integration cell has an undeclared typed schema")


def _factory_integral(cold, body, material, raw):
    coordinates = tuple(_fraction(x) for x in raw["coordinates_m"])
    if len(coordinates) != 7 or not all(coordinates):
        raise ValueError("The authentic closed tensor probes must retain seven NONZERO coordinates")
    value = cold._common_raw_receipt(body, material, coordinates,
        tuple(_scalar(x) for x in raw["quarter_integral_bounds"]),
        tuple(_scalar(x) for x in raw["removed_material_correction_bounds"]),
        raw["integration_cell_count"],
        tuple(_scalar(x) for x in raw["maximum_taylor_remainder_absolute_upper_bounds"]),
        raw["taylor_order"], raw["precision"],
        tuple(_cell_receipt(x) for x in raw["cell_receipts"]),
        tuple(_cell_receipt(x) for x in raw["correction_receipts"]))
    return cold.validate_native_common_cold_tensor_integral(value, body=body, material=material)


def _binding_identity_paths():
    body_paths = (("body_snapshot_sha256",),
        ("embedding_bank", "bank_snapshot_sha256"),
        ("embedding_bank", "reference_snapshot_sha256"),
        *(("mathematical_body_view", key) for key in ("geometry_source_sha256",
            "body_snapshot_sha256", "reference_snapshot_sha256", "embedding_bank_snapshot_sha256")))
    material_paths = tuple((key,) for key in ("body_snapshot_sha256",
        "reference_snapshot_sha256", "material_snapshot_sha256"))
    raw_paths = tuple((key,) for key in ("body_snapshot_sha256", "material_snapshot_sha256",
        "reference_snapshot_sha256", "integral_snapshot_sha256"))
    return tuple(("body", *path) for path in body_paths) + tuple(
        ("material", *path) for path in material_paths) + tuple(
        ("raw_integrals", *path) for path in raw_paths) + tuple(
        ("diagnostic_response", "common_body_binding", *path) for path in body_paths) + tuple(
        ("diagnostic_response", "body", key) for key in ("geometry_source_sha256",
            "body_snapshot_sha256", "reference_snapshot_sha256", "embedding_bank_snapshot_sha256")) + tuple(
        ("diagnostic_response", "material", *path) for path in material_paths) + tuple(
        ("diagnostic_response", "raw_integrals", *path) for path in raw_paths) + (
        ("diagnostic_response", "support", "body_snapshot_sha256"),
        ("diagnostic_response", "support", "reference_snapshot_sha256"),
        ("diagnostic_response", "reference_snapshot_sha256"))


def _substitute_binding_ids(value, identities):
    """Copy only the explicitly enumerated binding fields, never text/data."""
    result = dict(value)
    for path in _binding_identity_paths():
        original, replacement = value, result
        for key in path[:-1]:
            if type(original) is not dict or key not in original or type(original[key]) is not dict:
                raise ValueError("A precision-equivalent DTO changed a declared binding-field schema")
            original = original[key]
            replacement[key] = dict(replacement[key])
            replacement = replacement[key]
        key = path[-1]
        if type(original) is not dict or key not in original or original[key] not in identities:
            raise ValueError("A precision-equivalent DTO changed an explicit binding identity")
        replacement[key] = identities[original[key]]
    return result


@lru_cache(maxsize=1)
def _cold_context_pairs():
    """Exact closed160 restore, then explicit equivalent-input871 rebuild.

    The two native contexts retain different genuine identities.  Each fresh
    factory input is read from the raw-byte pinned closed row.  Every physical
    field must match before and after the known binding-ID substitution; the
    full physical reference has already passed a separate exact comparison.
    No arbitrary DTO is registered and no volume quadrature is executed.
    """
    from tectonics import geometric_native_common_cold as cold
    source, current_reference, *_ = source_context()
    closed_reference, rows, _ = closed_common_context()
    result = []
    for row in rows:
        contexts = []
        dtos = []
        for reference in (closed_reference, current_reference):
            body = cold.bind_native_common_cold_body(row["component_id"], source=source,
                reference=reference, precision=160)
            material = cold.build_native_common_cold_strain_material(body)
            integral = _factory_integral(cold, body, material, row["diagnostic_response"]["raw_integrals"])
            response = cold.evaluate_native_common_cold_strain_energy(body, integral.coordinates_m,
                material=material, integrals=integral, precision=64)
            contexts.append((body, material, integral))
            dtos.append(dict(body=cold.native_common_cold_body_to_dict(body),
                material=cold.native_common_cold_strain_material_to_dict(material),
                raw_integrals=payload(integral),
                diagnostic_response=cold.native_common_cold_strain_energy_to_dict(response)))
        expected = dict(body=row["body"], material=row["material"],
            raw_integrals=row["diagnostic_response"]["raw_integrals"],
            diagnostic_response=row["diagnostic_response"])
        if dtos[0] != expected:
            raise ValueError("The full original160 body/material/integral/diagnostic DTO differs on native replay")
        old_body, old_material, old_integral = contexts[0]
        body, material, integral = contexts[1]
        identities = {closed_reference.reference_snapshot_sha256: current_reference.reference_snapshot_sha256,
            old_body.embedding_bank.bank_snapshot_sha256: body.embedding_bank.bank_snapshot_sha256,
            old_body.body_snapshot_sha256: body.body_snapshot_sha256,
            old_material.material_snapshot_sha256: material.material_snapshot_sha256,
            old_integral.integral_snapshot_sha256: integral.integral_snapshot_sha256}
        if (len(identities) != 5 or _substitute_binding_ids(expected, identities) != dtos[1]):
            raise ValueError("Equivalent reference precision changed a complete physical cold DTO field")
        metadata = dict(component_id=body.component_id, binding_id_substitutions=identities,
            allowed_binding_identity_paths=[list(path) for path in _binding_identity_paths()],
            full_closed_body_dto_sha256=digest(dtos[0]["body"]),
            full_rebuilt_body_dto_sha256=digest(dtos[1]["body"]),
            full_closed_material_dto_sha256=digest(dtos[0]["material"]),
            full_rebuilt_material_dto_sha256=digest(dtos[1]["material"]),
            full_closed_diagnostic_response_sha256=digest(dtos[0]["diagnostic_response"]),
            full_rebuilt_diagnostic_response_sha256=digest(dtos[1]["diagnostic_response"]),
            original_closed_full_dto_equality_checked=True,
            rebuilt_full_dto_equality_under_only_declared_binding_id_substitution_checked=True)
        metadata["replay_sha256"] = digest(metadata)
        result.append((contexts[0], contexts[1], metadata))
    return tuple(result)


def authentic_closed_cold_contexts():
    return tuple(pair[0] for pair in _cold_context_pairs())


@lru_cache(maxsize=1)
def cold_contexts():
    """Fresh genuine871-context receipts at the unchanged NONZERO probes."""
    return tuple(pair[1] for pair in _cold_context_pairs())


def cold_source_records():
    records = []
    for row, (old_context, context, metadata) in zip(closed_cold_records(), _cold_context_pairs()):
        body, material, raw = context
        records.append(dict(component_id=body.component_id,
            body_snapshot_sha256=body.body_snapshot_sha256,
            material_snapshot_sha256=material.material_snapshot_sha256,
            integral_snapshot_sha256=raw.integral_snapshot_sha256,
            coordinates_m=payload(raw.coordinates_m), raw_integrals=payload(raw),
            original_closed_raw_integrals=payload(old_context[2]),
            full_closed_row_sha256=digest(row),
            full_closed_raw_integral_sha256=digest(row["diagnostic_response"]["raw_integrals"]),
            full_rebuilt_raw_integral_sha256=digest(payload(raw)),
            complete_precision_equivalence_replay=metadata,
            reference_precision_equivalence_sha256=reference_precision_equivalence()["equivalence_sha256"],
            restored_by_original_native_raw_factory=True,
            rebuilt_by_original_native_raw_factory=True,
            complete_old_diagnostic_algebra_replayed=True, fresh_tensor_quadrature_executed=False))
    return records


@lru_cache(maxsize=1)
def strain_system():
    from tectonics import geometric_native_common_strain_projection as api
    return api.bind_native_common_strain_projection(material_system(),
        cold_contexts=cold_contexts(), provenance=PROVENANCE, precision=96)


def query_cases(system):
    values = system.bound_coordinate_increments
    if type(values) is not tuple or len(values) != 92 or any(type(x) is not F for x in values):
        raise ValueError("The source reduction requires its full authenticated92 coordinate vector")
    return tuple(dict(coordinate_increments=values, branch=branch, precision=96)
        for branch in ("relaxed_zero_strength_shear", "frozen_elastic_predictor"))


def summary(count):
    return dict(physical_candidate_status=PHYSICAL_SCOPE,
        declared_common_coordinate_count=92, owner_coordinate_count=6,
        cold_body_coordinate_count=56, hot_body_coordinate_count=30,
        reduced_internal_source_channel_count=8, original_source_mechanism_count=16,
        original_material_attachment_count=32, authentic_nonzero_common_cold_probe_count=8,
        independent_global_beam_count=8, orthogonal_cold_remainder_count=8,
        complete_global_query_count=count, source_coordinate_count=14,
        chosen_force_basis=FORCE_BASIS, source_material_identity_preserved=True,
        one_global_source_reduction_defined=True,
        source_internal_channels_reduced_into_cold_tensor_modes=True,
        full_potential_gradient_and_hessian_chain_rule_retained=True,
        nonlinear_projection_hessian_term_retained=True,
        orthogonal_cold_remainder_energy_energized_once=True,
        original_global_beam_energy_energized_once=True,
        original_source_work_lineages_retained=True,
        literal_source_interface_measures_retained=True,
        accepted_original_native_intervals=1, accepted_new_native_intervals=0,
        original_point_1_remaining_work_blocks=1, remaining_physical_conditions=list(REMAINING_CONDITIONS),
        **{name: False for name in FALSE_SCOPE})


def replay_packet(packet):
    from tectonics import geometric_native_common_strain_projection as api
    from tectonics import geometric_native_common_material_transfer as material
    expected_keys = set(summary(2)) | {"format", "prior_871_packet_raw_sha256",
        "prior_871_packet_sha256", "prior_871_wrapper_raw_sha256", "prior_841_packet_raw_sha256",
        "prior_841_packet_sha256", "material_transfer", "authentic_cold_contexts",
        "strain_projection", "global_queries", "reference_precision_equivalence", "packet_sha256"}
    if (type(packet) is not dict or set(packet) != expected_keys
            or packet.get("packet_sha256") != digest({k: v for k, v in packet.items() if k != "packet_sha256"})):
        raise ValueError("The complete strain-projection packet schema or selfseal changed")
    if (packet.get("format") != "native-common-bankwise-cold-strain-projection-packet-1"
            or packet.get("prior_871_packet_raw_sha256") != PRIOR_871_PACKET_RAW
            or packet.get("prior_871_packet_sha256") != PRIOR_871_PACKET_SEAL
            or packet.get("prior_871_wrapper_raw_sha256") != PRIOR_871_WRAPPER_RAW
            or packet.get("prior_841_packet_raw_sha256") != PRIOR_841_PACKET_RAW
            or packet.get("prior_841_packet_sha256") != PRIOR_841_PACKET_SEAL
            or encoded(packet.get("authentic_cold_contexts")) != encoded(cold_source_records())
            or encoded(packet.get("reference_precision_equivalence")) != encoded(reference_precision_equivalence())):
        raise ValueError("The complete original source and eight NONZERO typed context lineages changed")
    source_transfer = material_system()
    if encoded(packet["material_transfer"]) != encoded(material.native_common_material_transfer_to_dict(source_transfer)):
        raise ValueError("The original material and source-potential model changed")
    system = api.native_common_strain_projection_system_from_dict(packet["strain_projection"],
        transfer=source_transfer, cold_contexts=cold_contexts())
    if system.provenance != PROVENANCE or system.precision != 96:
        raise ValueError("The declared strain-projection reduction changed")
    expected = query_cases(system)
    rows = packet["global_queries"]
    if type(rows) is not list or len(rows) != len(expected):
        raise ValueError("Both complete92 coordinate controls and all eight tensor probes are required")
    for row, arguments in zip(rows, expected):
        if type(row) is not dict or set(row) != {"arguments", "response"} or encoded(row["arguments"]) != encoded(payload(arguments)):
            raise ValueError("The literal whole-coordinate query changed")
        result = api.native_common_strain_projection_response_from_dict(row["response"], system=system)
        if (result.coordinate_increments != arguments["coordinate_increments"]
                or result.branch != arguments["branch"] or result.precision != arguments["precision"]):
            raise ValueError("A genuine full response was transplanted onto another query")
        if encoded(api.native_common_strain_projection_response_to_dict(result)) != encoded(row["response"]):
            raise ValueError("The complete strain-projection mechanical response changed")
    result = summary(len(rows))
    if encoded({key: packet.get(key) for key in result}) != encoded(result):
        raise ValueError("The physical scope, once-only energy ownership or point1 status changed")
    return result


def run(out):
    from tectonics import geometric_native_common_strain_projection as api
    from tectonics import geometric_native_common_material_transfer as material
    started = time.perf_counter()
    target = Path(out); target.mkdir(parents=True, exist_ok=False)
    system = strain_system()
    print("AUTHENTIC COMMON841 NONZERO tensor probes:8 complete typed factory and diagnostic replays PASS", flush=True)
    rows = []
    for index, arguments in enumerate(query_cases(system), 1):
        response = api.evaluate_native_common_strain_projection(system, **arguments)
        rows.append(dict(arguments=payload(arguments),
            response=api.native_common_strain_projection_response_to_dict(response)))
        print("GLOBAL bankwise cold strain-projection query", index, "PASS", flush=True)
    result = summary(len(rows))
    source_records = cold_source_records()
    packet = dict(format="native-common-bankwise-cold-strain-projection-packet-1",
        prior_871_packet_raw_sha256=PRIOR_871_PACKET_RAW, prior_871_packet_sha256=PRIOR_871_PACKET_SEAL,
        prior_871_wrapper_raw_sha256=PRIOR_871_WRAPPER_RAW,
        prior_841_packet_raw_sha256=PRIOR_841_PACKET_RAW, prior_841_packet_sha256=PRIOR_841_PACKET_SEAL,
        material_transfer=material.native_common_material_transfer_to_dict(material_system()),
        authentic_cold_contexts=source_records,
        reference_precision_equivalence=reference_precision_equivalence(),
        strain_projection=api.native_common_strain_projection_system_to_dict(system),
        global_queries=rows, **result)
    packet["packet_sha256"] = digest(packet)
    (target/"source_packet.json").write_bytes(encoded(packet)+b"\n")
    report = dict(format="native-common-bankwise-cold-strain-projection-control-1", status="PASS",
        source_packet_sha256=packet["packet_sha256"], global_queries_sha256=digest(rows),
        authentic_cold_contexts_sha256=digest(source_records),
        reference_precision_equivalence_sha256=reference_precision_equivalence()["equivalence_sha256"],
        strain_projection_sha256=system.binding_sha256,
        material_transfer_sha256=material_system().binding_sha256,
        seconds=time.perf_counter()-started, **result)
    report["report_sha256"] = digest(report)
    (target/"report.json").write_bytes(encoded(report)+b"\n")
    print("COMMON bankwise cold strain-projection component PASS; local faces/HOT/equilibrium/admission remain open", flush=True)
    return report
