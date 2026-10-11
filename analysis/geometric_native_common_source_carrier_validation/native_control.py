"""Whole literal source force/work carriers; no target material adoption."""
from __future__ import annotations

from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time

from analysis.geometric_native_common_contact_transfer_validation.native_control import encoded, payload

PHYSICAL_SCOPE = "COMMON_WHOLE_SOURCE_MECHANICAL_CARRIER_COMPONENT_ONLY"
PROVENANCE = "once-only original literal mechanical carriers with explicit archival and current objective bases"
ALLOCATION_PROVENANCE = "explicit whole-source receiver partition with one normalized depth profile per original family"
PRIOR_WRAPPER_RAW = "7df81606a168656cd51208a71d7ad425714751f736d9ed4ca50218263e106f95"
BASES = ("archival_old_reference", "current_objective")
BRANCHES = ("relaxed_zero_strength_shear", "frozen_elastic_predictor")
FALSE_SCOPE = (
    "physical_source_material_handover_certified", "old_world_force_pullback_preserved",
    "full_three_dimensional_rod_endpoints_recovered", "continuous_measure_and_literal_atoms_energized_together",
    "hot_receiving_contact_law_inferred", "new_mixed_force_equilibrium_certified",
    "whole_clock_all_pair_admission_authorized", "physical_material_admission_authorized",
    "source_history_advanced", "original_point_1_completed",
)
REMAINING_CONDITIONS = (
    "Whole-source physical material and old-force/prestress/work handover, HOT law/mean chart, loads/drag and NEW mixed equilibrium",
    "Whole-clock all-pair admission including reverse C5 to C1",
    "Accepted continuation with independent path work, balances and joint restart",
)


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def query_cases(allocation, system):
    zero = (F(0),)*14
    entries = allocation.original_atom_ledger
    cases = [dict(point_id=entry.point_id, basis=basis, branch=branch,
                  coordinate_increments=zero, precision=96)
             for entry in entries for basis in BASES for branch in BRANCHES]
    # The two angular classes select original nodes, not new material points.
    choices = []
    for prefix in ("retained-contact:", "inactive:"):
        entry = next(entry for entry in entries if entry.source_accounting_bucket.startswith(prefix))
        increment = list(zero)
        increment[system.coordinate_order.index(("radial", entry.trace_id, "sub"))] = F(1, 1000000)
        choices.append((entry, tuple(increment)))
    cases += [dict(point_id=entry.point_id, basis="current_objective", branch=branch,
                   coordinate_increments=increment, precision=96)
              for entry, increment in choices for branch in BRANCHES]
    return tuple(cases)


def summary(allocation, system, count):
    return dict(physical_candidate_status=PHYSICAL_SCOPE,
        original_source_family_count=len(allocation.families), source_carrier_count=16,
        source_coordinate_count=14, preserved_original_atom_count=len(allocation.original_atom_ledger),
        active_source_atom_accounting_count=allocation.active_current_receiver_source_atom_count,
        inactive_source_atom_accounting_count=allocation.inactive_current_receiver_source_atom_count,
        complete_carrier_query_count=count, baseline_carrier_query_count=64,
        finite_radial_carrier_query_count=4, source_mechanical_basis_count=2,
        declared_shear_branch_count=2, original_source_work_lineages_retained=True,
        literal_source_interface_measures_retained=True,
        receiver_frame_couple_included=True,
        angular_coordinate_units="radian", radial_coordinate_units="metre",
        accepted_original_native_intervals=1, accepted_new_native_intervals=0,
        original_point_1_remaining_work_blocks=1, remaining_physical_conditions=list(REMAINING_CONDITIONS),
        **{name: False for name in FALSE_SCOPE})


def run(out):
    from tectonics.geometric_native_normal_support import load_native_normal_support_source
    from tectonics.geometric_native_common_reference import bind_native_common_reference, native_common_reference_to_dict
    from tectonics.geometric_native_common_contact_cuts import bind_common_contact_cuts, common_contact_cuts_to_dict
    from analysis.geometric_native_common_source_allocation_validation.native_control import declared_profiles
    from tectonics import geometric_native_common_source_allocation as allocation_api
    from tectonics import geometric_native_common_source_carrier as api
    started = time.perf_counter()
    target = Path(out); target.mkdir(parents=True, exist_ok=False)
    source = load_native_normal_support_source()
    reference = bind_native_common_reference(source=source, precision=96)
    cuts = bind_common_contact_cuts(reference, precision=96)
    allocation = allocation_api.bind_native_common_source_allocation(reference, cuts,
        depth_profiles=declared_profiles(source), provenance=ALLOCATION_PROVENANCE,
        precision=96)
    system = api.bind_native_common_source_carrier_system(allocation, reference=reference,
        cuts=cuts, provenance=PROVENANCE, precision=96)
    rows = []
    for index, arguments in enumerate(query_cases(allocation, system), 1):
        response = api.evaluate_native_common_source_carrier(system, **arguments)
        rows.append(dict(arguments=payload(arguments),
                         response=api.native_common_source_carrier_response_to_dict(response)))
        if index % 4 == 0:
            print("SOURCE mechanical carrier queries", index, "PASS", flush=True)
    result = summary(allocation, system, len(rows))
    packet = dict(format="native-common-whole-source-carrier-packet-1",
        prior_863_wrapper_raw_sha256=PRIOR_WRAPPER_RAW,
        source_snapshot_sha256=source.source_snapshot_sha256,
        common_reference=native_common_reference_to_dict(reference),
        cut_set=common_contact_cuts_to_dict(cuts),
        allocation=allocation_api.native_common_source_allocation_to_dict(allocation),
        carrier_system=api.native_common_source_carrier_system_to_dict(system),
        carrier_queries=rows, **result)
    packet["packet_sha256"] = digest(packet)
    (target/"source_packet.json").write_bytes(encoded(packet)+b"\n")
    report = dict(format="native-common-whole-source-carrier-control-1", status="PASS",
        source_packet_sha256=packet["packet_sha256"], carrier_queries_sha256=digest(rows),
        carrier_system_sha256=system.binding_sha256, seconds=time.perf_counter()-started, **result)
    report["report_sha256"] = digest(report)
    (target/"report.json").write_bytes(encoded(report)+b"\n")
    print("WHOLE source mechanical carrier PASS; target material handover remains unadopted", flush=True)
    return report
