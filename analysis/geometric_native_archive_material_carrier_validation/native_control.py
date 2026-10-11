"""Fresh carrier controls from genuine sealed material, without a force replay.

The upper-layer reference is newly declared geometry. This control does not
transfer old force/prestress/work to it or accept a physical time interval.
"""
from __future__ import annotations

from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRECTORY = "analysis/geometric_native_first_entry_validation/control_release_02"
AUTHORITY_PATH = "analysis/geometric_native_common_material_transfer_validation/release_validation_01/native_control/source_packet.json"
AUTHORITY_RAW = "a9fcc7874c0e68940e20ded658748300f55d9f1d296db5c9d67b59269ac8b3ed"
AUTHORITY_SEAL = "cd1e2457a857b47891cd40a89219abfac3aa2228028aa9636f0e77f7a10082dd"
SOURCE_STATE_SEAL = "dbe625f618115e466214c2f46d98f64f974335dccb4a2e1da51c1b2eb6a39f0c"
SOURCE_PINS = {
    "material_state.json": "3d24eee293b4517c35359ae3c31feaac5b43a5ee066209f8faaf65457b35dfc3",
    "full_domain_journal.json": "8da581b103945cc85889d3fe8ac8a70361994d33c3f2dd6b31e917cc3273fb82",
    "first_entry_sweep.json": "fb80e06cebe93ec8c3db0dcc6667990bca3e56c672e21f3602ddbd8852f327c7",
    "point_certificates.json": "c57801a6e80051dffec2250321f6e812bfe36d10f19993d2a9c4172bb6e459b0",
    "material_point_queries.json": "70a962517eeb5dfdcf283f1b299ce23fac74ac24fd69f10bd549f4c648b56b98",
    "endpoint_material_queries.json": "4180cd74c81e52c11cb9baecc5c7fb3a584d83d67a3557c405da85e6489736d0",
}
PHYSICAL_SCOPE = "DECLARED_GLOBAL_PAIRED_THREE_DIMENSIONAL_ARCHIVE_CARRIER_ONLY"
PACKET_FORMAT = "native-archive-material-carrier-packet-1"
PROVENANCE = "Declared paired areal-volume upper reference and smooth irreversible down-dip carrier; genuine first-entry material authority"
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


def _read(path, raw_pin):
    raw = (ROOT / path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != raw_pin:
        raise ValueError("An immutable material authority changed: " + path)
    return json.loads(raw)


@lru_cache(maxsize=1)
def source_documents():
    authority = _read(AUTHORITY_PATH, AUTHORITY_RAW)
    if (authority["packet_sha256"] != AUTHORITY_SEAL or
            digest({k: v for k, v in authority.items() if k != "packet_sha256"}) != AUTHORITY_SEAL):
        raise ValueError("The retained original material packet seal changed")
    documents = {name: _read(SOURCE_DIRECTORY + "/" + name, pin)
                 for name, pin in SOURCE_PINS.items()}
    embedded = json.loads(authority["material_transfer"]["complete_original_material_state_json"])
    if (embedded["state_sha256"] != SOURCE_STATE_SEAL or
            encoded(embedded) != encoded(documents["material_state.json"])):
        raise ValueError("The readable material is not the complete retained source")
    return documents


@lru_cache(maxsize=1)
def source_state():
    from tectonics.geometric_native_domain_journal import native_domain_journal_from_dict
    from tectonics.geometric_native_first_entry_material import native_first_entry_material_from_dict
    documents = source_documents()
    raw = documents["material_state.json"]
    journal = native_domain_journal_from_dict(documents["full_domain_journal.json"])
    return native_first_entry_material_from_dict(raw, expected_journal=journal,
        expected_original_mechanics_payload=raw["original_mechanics_payload"])


@lru_cache(maxsize=1)
def source_point_rows():
    """Complete saved chronology, preserving rays and unresolved root windows."""
    from tectonics.geometric_native_first_entry_material import native_first_entry_material_point_chronology_from_dict
    result = []
    for row in source_documents()["point_certificates.json"]:
        raw = row["certificate"]
        result.append(dict(label=row["label"], original_parent_id=raw["original_parent_id"],
            original_ray=tuple(F(*value) for value in raw["original_point"]),
            chronology=native_first_entry_material_point_chronology_from_dict(raw, state=source_state())))
    return tuple(result)


@lru_cache(maxsize=1)
def system():
    from tectonics import geometric_native_archive_material_carrier as api
    return api.bind_native_archive_material_carrier(source_material_state=source_state(),
        provenance=PROVENANCE, precision=96, history_order=24, history_cells=64)


def summary():
    return dict(physical_candidate_status=PHYSICAL_SCOPE,
        original_material_parent_count=8, nonempty_original_archive_cohort_count=3,
        archive_depth_and_motion_newly_declared=True,
        receiver_reference_is_new_upper_layer=True,
        source_areal_volume_coordinate_preserved=True,
        archive_entry_time_midpoint_used_in_carrier_law=False,
        previous_physical_roof_or_prestress_recovered=False,
        physical_source_force_prestress_work_handover_certified=False,
        nonzero_common_cold_modes_attached_to_new_reference=False,
        hot_constitutive_law_defined=False,
        new_mixed_force_equilibrium_certified=False,
        whole_clock_all_pair_admission_authorized=False,
        physical_material_admission_authorized=False,
        full_system_temporal_path_work_certified=False,
        historical_force_solve_reexecuted=False,
        previous_tensor_quadrature_reexecuted=False,
        source_geometry_elapsed_time_s=0,
        contact_geometry_elapsed_time_s=2000000,
        memory_elapsed_time_s=1000000,
        material_geometry_elapsed_time_s=1000000,
        clocks_advanced=False, accepted_original_native_intervals=1,
        accepted_new_native_intervals=0,
        original_point_1_remaining_work_blocks=1,
        original_point_1_completed=False,
        remaining_physical_conditions=list(REMAINING_CONDITIONS))


def run(output_dir):
    """Fresh full-depth controls with explicit whole-domain and restart scope."""
    from tectonics import geometric_native_archive_material_carrier as api
    from tectonics.geometric_intrinsic_geometry import RotationIncrement
    from tectonics.geometric_native_pose import FractionInterval as I
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    carrier = system()
    dto = api.native_archive_material_carrier_to_dict(carrier)
    restored = api.native_archive_material_carrier_from_dict(dto, expected_source_material_state=source_state())
    if encoded(api.native_archive_material_carrier_to_dict(restored)) != encoded(dto):
        raise ValueError("The complete new carrier restart changed")
    packet = dict(format=PACKET_FORMAT, **summary(),
        source_authority_path=AUTHORITY_PATH, source_authority_raw_sha256=AUTHORITY_RAW,
        source_authority_packet_sha256=AUTHORITY_SEAL,
        source_document_raw_sha256={SOURCE_DIRECTORY + "/" + k: v for k, v in SOURCE_PINS.items()},
        source_material_state=source_documents()["material_state.json"],
        saved_source_point_chronologies=source_documents()["point_certificates.json"],
        carrier_system=dto, original_point_queries=[])
    cases = list(source_point_rows())
    cases.append(dict(label="whole_archive_seven", original_parent_id="octant:7",
        original_ray=(F(1, 10**9), F(1), F(1)), chronology=None))
    # Eight true parent interiors supplement the archived near-interface rays;
    # they are controls, not a replacement for the whole-domain proof.
    for index in range(8):
        cases.append(dict(label="parent_interior_" + str(index),
            original_parent_id="octant:" + str(index),
            original_ray=(F(1 if index & 4 else -1), F(1 if index & 2 else -1),
                          F(1 if index & 1 else -1)), chronology=None))
    archived_parents = set()
    for case in cases:
        parent = case["original_parent_id"]
        height = dict(carrier.whole_domain_certificate.original_parent_thicknesses_m)[parent]
        depths = (I(0), I(height/3), I(2*height/3), I(height), I(0, height))
        # The fine precision resolves genuine tiny swept/reentry history lobes.
        precision = 256 if case["label"] in ("swept_only_owner_entry_exit", "main_plane_reentry") else 96
        for depth in depths:
            arguments = dict(material_depth_bounds_m=depth, time_fraction=F(1), precision=precision)
            response = api.evaluate_native_archive_material_point(carrier, parent, case["original_ray"],
                chronology=case["chronology"], **arguments)
            if response.volume_jacobian_bounds.lo <= 0:
                raise ValueError("The actual full-depth query lacks a positive material Jacobian")
            if response.owner == carrier.donor_owner and response.radial_support_bounds_m.hi > carrier.radius_m:
                raise ValueError("A donor carrier exceeds its whole-domain separated radial support")
            if response.owner == carrier.receiver_owner and response.radial_support_bounds_m.lo < carrier.radius_m:
                raise ValueError("The declared upper receiving reference crossed its radial base")
            if response.allocation == "archived":
                archived_parents.add(parent)
                if response.archive_debits != 1:
                    raise ValueError("The genuine archive was debited more than once")
            row = dict(label=case["label"], arguments=dict(original_parent_id=parent,
                original_ray=api._payload(case["original_ray"]),
                material_depth_bounds_m=api._payload(depth), time_fraction=[1, 1], precision=precision),
                response=api.native_archive_material_point_to_dict(response, system=carrier))
            packet["original_point_queries"].append(row)
    if archived_parents != {"octant:5", "octant:6", "octant:7"}:
        raise ValueError("All three genuine nonempty archive carriers must be exercised")
    frame_history = (RotationIncrement((.01, .02, -.01), 1.),
                     RotationIncrement((-.015, .005, .02), 1.))
    selected = next(row for row in source_point_rows() if row["label"] == "ordinary_entry")
    frame = api.evaluate_native_archive_material_point(carrier, selected["original_parent_id"],
        selected["original_ray"], I(F(10000, 3)), F(1), precision=96,
        chronology=selected["chronology"], world_history=frame_history,
        world_translation_m=(F(7, 3), F(-9, 7), F(11, 5)))
    packet["ordered_observer_query"] = api.native_archive_material_point_to_dict(frame, system=carrier)
    probe = packet["original_point_queries"][0]
    expected = probe["arguments"]
    reloaded = api.native_archive_material_point_from_dict(probe["response"], system=restored,
        expected_original_parent_id=expected["original_parent_id"],
        expected_original_ray=tuple(F(*x) for x in expected["original_ray"]),
        expected_depth=I(F(*expected["material_depth_bounds_m"]["lo"]), F(*expected["material_depth_bounds_m"]["hi"])),
        expected_time_fraction=F(*expected["time_fraction"]), precision=expected["precision"])
    if encoded(api.native_archive_material_point_to_dict(reloaded, system=restored)) != encoded(probe["response"]):
        raise ValueError("Carrier point restart changed its full expected query")
    packet["complete_model_and_point_restart_verified"] = True
    packet["packet_sha256"] = digest(packet)
    (output / "source_packet.json").write_bytes(encoded(packet) + b"\n")
    report = dict(format="native-archive-material-carrier-control-1", status="PASS", **summary(),
        elapsed_seconds=time.monotonic() - started, packet_sha256=packet["packet_sha256"],
        packet_raw_sha256=hashlib.sha256((output / "source_packet.json").read_bytes()).hexdigest(),
        original_point_query_count=len(packet["original_point_queries"]),
        archived_original_parents_exercised=sorted(archived_parents),
        complete_model_and_point_restart_verified=True,
        whole_domain_minimum_volume_jacobian_bounds=api._payload(carrier.whole_domain_certificate.minimum_volume_jacobian_bounds))
    report["report_sha256"] = digest(report)
    (output / "report.json").write_bytes(encoded(report) + b"\n")
    print(json.dumps(report, sort_keys=True))
    return report
