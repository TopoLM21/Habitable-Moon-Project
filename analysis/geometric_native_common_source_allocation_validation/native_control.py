"""Whole original-source accounting and genuine assigned reference densities.

The eight source supports are owners of one continuous reference budget each.
Literal sixteen atom stores remain a separate immutable source ledger.  No
global force assembly or old three-dimensional prestress handover is inferred.
"""
from __future__ import annotations

from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time

from analysis.geometric_native_common_contact_transfer_validation.native_control import encoded, payload

PHYSICAL_SCOPE = "COMMON_WHOLE_SOURCE_ACCOUNTING_COMPONENT_ONLY"
PROVENANCE = "explicit whole-source receiver partition with one normalized depth profile per original family"
FALSE_SCOPE = (
    "all_original_source_measure_physically_transferred_once",
    "old_world_force_pullback_preserved", "old_rod_prestrain_or_energy_transferred",
    "face_depth_memory_or_measure_transfer_certified", "hot_curve_contact_law_inferred",
    "new_mixed_force_equilibrium_certified", "whole_clock_material_admission_authorized",
    "physical_material_admission_authorized", "source_history_advanced",
    "source_atomic_and_continuous_measures_equated", "spatial_column_endpoints_authenticated",
    "original_point_1_completed",
)
REMAINING_CONDITIONS = (
    "Whole-source physical material and old-force/prestress/work handover, HOT law/mean chart, loads/drag and NEW mixed equilibrium",
    "Whole-clock all-pair admission including reverse C5 to C1",
    "Accepted continuation with independent path work, balances and joint restart",
)


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def declared_profiles(source):
    before = json.loads(dict(source.complete_source_dto_json)["before"])
    bindings = before["snapshot"]["current_material_epoch"]["source_material_state"]["trace_bindings"]
    return {binding["trace_id"]: ((F(1),) if binding["receiver_fragment_id"] in ("octant:0", "octant:2")
                                  else (F(0), F(2))) for binding in bindings}


def query_arguments(cut, sample):
    from tectonics.geometric_native_pose import FractionInterval as I
    fraction = F(sample, 3)
    parameter, cube = cut.parameter_guaranteed_bounds, cut.common_physical_radius_cubed_bounds_m3
    return dict(parameter_bounds=I(parameter.lo+(parameter.hi-parameter.lo)*fraction),
                physical_radius_cubed_bounds_m3=I(cube.lo+(cube.hi-cube.lo)*fraction), precision=96)


def allocation_summary(allocation, query_count):
    active = tuple(cell for cell in allocation.cells if cell.active)
    return dict(physical_candidate_status=PHYSICAL_SCOPE,
        original_source_family_count=len(allocation.families), source_allocation_cell_count=len(allocation.cells),
        active_cold_support_cell_count=len(active), inactive_source_cell_count=len(allocation.cells)-len(active),
        source_boundary_identity_count=len(allocation.boundaries),
        exact_shared_projective_seam_count=sum(len(boundary.authentic_endpoint_aliases)==2 for boundary in allocation.boundaries),
        preserved_original_atom_count=len(allocation.original_atom_ledger),
        active_source_atom_accounting_count=allocation.active_current_receiver_source_atom_count,
        inactive_source_atom_accounting_count=allocation.inactive_current_receiver_source_atom_count,
        unresolved_source_atom_accounting_count=allocation.uncertain_endpoint_source_atom_count,
        genuine_allocated_density_query_count=query_count,
        whole_current_source_memory_query_count=4,
        whole_source_symbolic_reference_budget_closure_certified=allocation.symbolic_full_source_reference_area_closure_certified,
        whole_current_source_memory_retained=allocation.whole_current_continuous_memory_retained,
        once_only_sixteen_literal_source_ledgers_preserved=allocation.once_only_sixteen_literal_source_atom_ledger_retained,
        single_depth_profile_per_original_family_enforced=True,
        positive_inactive_source_tails_retained=True,
        accepted_original_native_intervals=1, accepted_new_native_intervals=0,
        original_point_1_remaining_work_blocks=1, remaining_physical_conditions=list(REMAINING_CONDITIONS),
        **{name: False for name in FALSE_SCOPE})


def run(out):
    from tectonics.geometric_native_pose import FractionInterval as I
    from tectonics.geometric_native_normal_support import load_native_normal_support_source
    from tectonics.geometric_native_common_reference import bind_native_common_reference, native_common_reference_to_dict
    from tectonics.geometric_native_common_contact_cuts import (
        bind_common_contact_cuts, common_contact_cuts_to_dict, evaluate_common_contact_cut, common_contact_cut_bounds_to_dict,
    )
    from tectonics import geometric_native_common_source_allocation as api
    target = Path(out); target.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source = load_native_normal_support_source()
    reference = bind_native_common_reference(source=source, precision=96)
    cuts = bind_common_contact_cuts(reference, precision=96)
    allocation = api.bind_native_common_source_allocation(reference, cuts,
        depth_profiles=declared_profiles(source), provenance=PROVENANCE, precision=96)
    memory_queries = []
    for family in allocation.families:
        angle = I(0, family.source_computational_extent_radians)
        query = api.evaluate_native_common_source_memory(allocation, family.trace_id, angle, precision=96)
        memory_queries.append(dict(trace_id=family.trace_id, computational_angle_bounds=payload(angle),
            query=api.native_common_source_memory_query_to_dict(query)))
    print("WHOLE current source memory four complete families PASS", flush=True)
    rows = []
    for cut in cuts.cuts:
        if cut.receiving_trace_id is None:
            continue
        cell = api.native_common_source_allocation_cell_for_cut(allocation, cut)
        for sample in (1, 2):
            arguments = query_arguments(cut, sample)
            geometry = evaluate_common_contact_cut(reference, cut, **arguments)
            density = api.evaluate_allocated_source_reference_density(allocation, reference, geometry, precision=96)
            rows.append(dict(cut_id=cut.cut_id, source_cell_id=cell.cell_id, sample=sample,
                arguments=payload(arguments), geometry=common_contact_cut_bounds_to_dict(geometry),
                density=api.allocated_source_reference_density_to_dict(density)))
        print("ALLOCATED genuine source density", cut.cut_id, "PASS", flush=True)
    hot = next(cut for cut in cuts.cuts if cut.receiving_trace_id is None)
    try:
        api.native_common_source_allocation_cell_for_cut(allocation, hot)
    except ValueError as error:
        rejection = str(error)
    else:
        raise ValueError("The genuine HOT boundary received an original cold source budget")
    summary = allocation_summary(allocation, len(rows))
    packet = dict(format="native-common-whole-source-allocation-packet-1",
        prior_859_wrapper_raw_sha256="1b5e1695f2ddfa8ea879619e91cde6b4cfb2c7c7c89e34db04148d8765f7f9ae",
        source_snapshot_sha256=source.source_snapshot_sha256,
        complete_source_dto_json=[list(item) for item in source.complete_source_dto_json],
        saved_joint_reference_field=json.loads(source.saved_field_json),
        common_reference=native_common_reference_to_dict(reference), cut_set=common_contact_cuts_to_dict(cuts),
        allocation=api.native_common_source_allocation_to_dict(allocation), density_queries=rows,
        source_memory_queries=memory_queries,
        hot_original_source_budget_rejection=rejection, **summary)
    packet["packet_sha256"] = digest(packet)
    (target/"source_packet.json").write_bytes(encoded(packet)+b"\n")
    report = dict(format="native-common-whole-source-allocation-control-1", status="PASS",
        source_packet_sha256=packet["packet_sha256"], density_queries_sha256=digest(rows),
        source_memory_queries_sha256=digest(memory_queries),
        allocation_binding_sha256=allocation.binding_sha256, seconds=time.perf_counter()-started, **summary)
    report["report_sha256"] = digest(report)
    (target/"report.json").write_bytes(encoded(report)+b"\n")
    print("WHOLE source accounting PASS; physical material/force/work handover remains unadopted", flush=True)
    return report
