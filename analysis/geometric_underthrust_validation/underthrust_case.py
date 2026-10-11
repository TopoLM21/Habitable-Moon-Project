"""Force-generated differential underthrust on persistent material-bank pairs.

All physical inputs are declared before either material velocity is solved.
The ordinary 20-cell integration mesh samples a globally positive, affine
mantle amplitude. Four original weak-plane arcs retain their complete material
domains and memories while their owners move differently. This bounded control
does not supply cooling, stress-created faults, or unrestricted slab dynamics.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
from itertools import product
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.cloud_handoff import verify as verify_cloud_sources
from tectonics.fractional_surface import EXTENSIVE_FIELDS, SurfaceParcel
from tectonics.genesis_contact_law import ContactLawParameters
from tectonics.geometric_bending import (
    ElasticBendingBankParameters, FaultBendingParameters,
)
from tectonics.geometric_bending_contact_dynamics import initialize_bending_contact_dynamics
from tectonics.geometric_boundary import initialize_boundary
from tectonics.geometric_contacts import extract_contacts
from tectonics.geometric_contact_mechanics import ElasticColumnParameters
from tectonics.geometric_fault_traces import PhysicalFaultTrace, TraceQuadrature, declare_fault_trace_set
from tectonics.geometric_spatial_mantle_loading import (
    PrescribedSpatialMantleSource, _body as spatial_source_body,
    _hash as spatial_source_hash, assemble_physical_spatial_mantle_load,
)
from tectonics.geometric_surface import GeometricFragment, GeometricSurfaceState, totals
from tectonics.geometric_underthrust_motion import (
    HotBirthParameters, advance_underthrust_motion, initialize_underthrust_motion,
    underthrust_motion_from_dict, underthrust_motion_to_dict,
    validate_underthrust_motion_state,
)
from tectonics.geometric_underthrust_motion_io import (
    load_underthrust_motion_checkpoint, save_underthrust_motion_checkpoint,
)
from tectonics.mesh import build_icosphere
from tectonics.spherical_polygons import normalize_polygon, polygon_area
from tectonics.thermal import SECONDS_PER_MYR


PROVENANCE = "Independent affine x mantle traction on declared x-hemisphere weak-plane ports"
DT_S = 1e6
ANCHOR_ERROR_BOUND_M = 1e-4
BOUNDS = dict(initial_loading_dt_s=1e6, max_rotation_radians=.01,
    max_tip_slope=.01, max_bending_strain=.001, max_deflection_radius_ratio=.001)
SOURCE_INTERVALS = (
    PrescribedSpatialMantleSource((0., 0., 1e-12), 1e14, 1., (.2, 0., 0.), PROVENANCE),
    PrescribedSpatialMantleSource((0., 0., 1.1e-12), 1.2e14, 1., (.17, 0., 0.),
        "Explicit second spatial source selected before either force result"),
)


def _snapshot(value):
    return json.dumps(asdict(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hashes():
    paths = sorted((ROOT/"tectonics").glob("*.py"))
    paths += sorted((ROOT/"tests").glob("*underthrust*.py"))
    paths += [ROOT/"tests/test_geometric_spatial_mantle_loading.py",
        ROOT/"tests/test_geometric_material_pair_reference.py",
        ROOT/"tests/test_geometric_buried_bank_support.py",
        ROOT/"scripts/cloud_handoff.py", Path(__file__).resolve()]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(paths)) if path.is_file()}


def physical_inputs(*, initial_gap_m=-.1):
    """Standalone explicit inputs equivalent to the focused physical fixture."""
    mesh, radius, fragments = build_icosphere(0), 100., []
    for index, signs in enumerate(product((-1., 1.), repeat=3)):
        polygon = normalize_polygon(np.diag(signs))
        area = polygon_area(polygon)*radius**2
        owner = int(signs[0] > 0.)
        parcel = SurfaceParcel(index, owner, f"octant-material:{index}", area,
            2.*area, 8.*area, area*1e12, 5., (("damage", .8), ("water", .5)))
        fragments.append(GeometricFragment(f"octant:{index}", tuple(map(tuple, polygon)), parcel))
    origin = GeometricSurfaceState(0., radius, tuple(fragments))
    law = ContactLawParameters(normal_stiffness_pa_m=1e6, tangential_stiffness_pa_m=1e6,
        tensile_strength_pa=2e6, fracture_energy_j_m2=2e7, cohesion_pa=0.,
        friction_dry=0., friction_wet=0., wet_cohesion_fraction=.5, viscosity_pa_s_m=0.)
    columns = ElasticColumnParameters(1000., 1e9, 4e8, 1000., 1e9, 4e8)
    traces = []
    for contact in extract_contacts(origin.fragments, radius):
        start, end = contact.start, contact.end
        wanted = np.asarray(contact.normal_a_to_b)*(1. if contact.plate_b == 0 else -1.)
        if float(np.cross(start, end)@wanted) < 0.:
            start, end = end, start
        traces.append(PhysicalFaultTrace("x-port:"+contact.contact_id, start, end,
            1, 0, math.pi/6., columns, law, 1000., .7, 0., initial_gap_m, 0.,
            "Explicit inclined physical port; pressure from declared preload"))
    trace_set = declare_fault_trace_set(origin, traces, TraceQuadrature(1, 4))
    sub = ElasticBendingBankParameters(10000., 2000., 7e10, .25,
        "Declared original donor cantilever bank, including archived measure")
    over = ElasticBendingBankParameters(15000., 3000., 8e10, .3,
        "Declared original receiver cantilever bank")
    banks = tuple(FaultBendingParameters(trace.trace_id, sub, over) for trace in trace_set.traces)
    load = assemble_physical_spatial_mantle_load(mesh, origin, SOURCE_INTERVALS[0])
    seed = initialize_bending_contact_dynamics(origin, trace_set, load, banks, **BOUNDS)
    assert seed.rotation_radians == ((0., 0., 0.),)*2
    assert all(a == b == 0. for _, a, b in seed.transverse_displacement_m)
    assert all(local.plastic_slip_m == local.cumulative_slip_m == local.jump_free_m == 0.
               for _, local in seed.states)
    return mesh, origin, initialize_boundary(origin), seed, load


def _initialize(mesh, origin, inventory, seed, load, *, error_bound=ANCHOR_ERROR_BOUND_M):
    return initialize_underthrust_motion(mesh, origin, inventory, seed, load,
        birth_parameters=HotBirthParameters(2., 0., 0.,
            "Explicit hot departure-footprint closure for actual material vacancies"),
        max_anchor_transport_error_m=error_bound)


def _grouped(parcels, field):
    groups = {}
    for parcel in parcels:
        groups.setdefault(parcel.material_id, []).append(getattr(parcel, field))
    return {identity: math.fsum(values) for identity, values in groups.items()}


def _ledger(origin, state, births):
    maxima = {}
    for field in EXTENSIVE_FIELDS:
        groups = [_grouped(parcels, field) for parcels in (
            (fragment.parcel for fragment in origin.fragments),
            (fragment.parcel for fragment in state.current_surface.fragments),
            (cohort.parcel for cohort in state.current_inventory.cohorts),
            (fragment.parcel for fragment in births))]
        maxima[field] = 0.
        for identity in set().union(*(group.keys() for group in groups)):
            old, remaining, buried, born = [group.get(identity, 0.) for group in groups]
            error = abs(math.fsum((remaining, buried, -old, -born)))/max(abs(old)+abs(born), 1.)
            assert error < 5e-12, (field, identity, error)
            maxima[field] = max(maxima[field], error)
    return maxima


def _source_sample_work(before, load, increments):
    """Independent physical traction work; no assembled torque is used here."""
    radius = np.longdouble(before.radius_km*1000.)
    motion = {owner: np.asarray(row, dtype=np.longdouble)
        for owner, row in zip(load.plate_ids, increments)}
    work = np.longdouble(0.)
    for sample in load.samples:
        displacement = radius*np.cross(motion[sample.plate_id],
            np.asarray(sample.position, dtype=np.longdouble))
        work += np.longdouble(sample.area_m2)*np.asarray(sample.source_traction_pa,
            dtype=np.longdouble)@displacement
    return float(work)


def _step_checks(origin, before, load, step, births):
    state, metrics = step.state, dict(step.diagnostics)
    increments = np.asarray(step.rotation_increment_radians)
    assert np.linalg.norm(increments[0]-increments[1]) > 0.
    assert state.initial_mechanics == before.initial_mechanics
    assert state.initial_load == before.initial_load
    assert state.preparation_work_j == before.preparation_work_j
    assert state.initial_inventory == before.initial_inventory
    assert state.current_surface.time_myr == state.current_inventory.time_myr
    assert state.current_surface.time_myr == before.current_surface.time_myr+DT_S/SECONDS_PER_MYR
    assert state.elapsed_time_s == math.fsum((before.elapsed_time_s, DT_S))
    old_nodes = dict(before.states)
    assert all(local.step_count == old_nodes[identity].step_count+1
        and local.elapsed_time_s == math.fsum((old_nodes[identity].elapsed_time_s, DT_S))
        for identity, local in state.states)
    assert state.spatial_memory.stage_count == before.spatial_memory.stage_count+1
    assert state.spatial_memory.elapsed_time_s == math.fsum((before.spatial_memory.elapsed_time_s, DT_S))
    assert any(a != 0. or b != 0. for _, a, b in state.transverse_displacement_m)
    assert any(local.plastic_slip_m != old_nodes[identity].plastic_slip_m
        for identity, local in state.states)
    assert metrics["anchor_transport_error_m"] <= ANCHOR_ERROR_BOUND_M
    assert metrics["continuous_gap_max_m"] <= 0.
    for name in ("torque_signal_relative_residual", "incremental_energy_balance_relative_error",
                 "cumulative_energy_balance_relative_error", "material_reaction_relative_error",
                 "angular_signal_relative_residual", "transverse_force_relative_residual"):
        assert metrics[name] < 5e-10, (name, metrics[name])
    physical_work = _source_sample_work(before.current_surface, load, increments)
    relative_work_error = abs(physical_work-metrics["source_work_j"])/max(
        abs(physical_work), abs(metrics["source_work_j"]), np.finfo(float).tiny)
    assert relative_work_error < 3e-14
    transaction = step.geometric_transaction
    assert transaction.losses and transaction.births
    assert all(loss.polarity_basis == "solved_spatial_underthrust_admission" for loss in transaction.losses)
    assert all(loss.contact_ids and loss.attachment_status == "geometric_contact"
        for loss in transaction.losses)
    assert len(state.current_inventory.cohorts) == len(before.current_inventory.cohorts)+len(transaction.losses)
    assert len(state.buried_registry.records) == len(before.buried_registry.records)+len(transaction.losses)
    assert len(step.support_receipt.live_clamps) == 2*len(state.initial_mechanics.fault_traces.traces)
    assert len(step.support_receipt.original_material_domains) == len(origin.fragments)
    old_cohorts = {cohort.event_id: cohort for cohort in before.current_inventory.cohorts}
    for cohort in state.current_inventory.cohorts:
        assert cohort.supports and not cohort.unresolved_support
        assert cohort.attachment_status == "geometric_contact"
        if cohort.event_id in old_cohorts:
            old = old_cohorts[cohort.event_id]
            assert cohort.loss == old.loss and cohort.parcel == old.parcel
            assert cohort.payload_sha256 == old.payload_sha256
            assert cohort.acceptance_time_myr == old.acceptance_time_myr
    budgets = _ledger(origin, state, births)
    certificate_branches = Counter(piece.branch for certificate in step.spatial_step.certificates
        for piece in certificate.pieces)
    return dict(sequence=len(state.receipts), source=asdict(load.source), load_sha256=load.load_sha256,
        before_surface_sha256=load.surface_sha256, end_time_myr=state.current_surface.time_myr,
        rotation_increment_radians=step.rotation_increment_radians,
        angular_velocity_rad_s=step.angular_velocity_rad_s,
        diagnostics=metrics, independent_physical_source_work_relative_error=relative_work_error,
        birth_count=len(transaction.births), loss_count=len(transaction.losses),
        minimum_admitted_area_km2=min(loss.fragment.parcel.area_km2 for loss in transaction.losses),
        minimum_hot_birth_area_km2=min(fragment.parcel.area_km2 for fragment in transaction.births),
        born=totals(transaction.births), buried=totals(loss.fragment for loss in transaction.losses),
        cohort_count=len(state.current_inventory.cohorts),
        attached_cohort_count=sum(bool(cohort.supports) for cohort in state.current_inventory.cohorts),
        archive_support_arc_count=sum(len(cohort.supports) for cohort in state.current_inventory.cohorts),
        unresolved_archive_support_count=sum(cohort.unresolved_support for cohort in state.current_inventory.cohorts),
        registered_buried_original_domains=len(state.buried_registry.records),
        continuous_constitutive_branch_counts=dict(sorted(certificate_branches.items())),
        admission=step.admission_audit, material_budget_max_relative_residual=budgets)


def _expect_atomic_rejection(name, origin, state, operation):
    before = _snapshot(state)
    try:
        operation()
    except (ValueError, RuntimeError, OverflowError) as error:
        assert _snapshot(state) == before
        details = dict(case=name, rejected=True, input_unchanged=True,
            exception_type=type(error).__name__, reason=str(error))
        if hasattr(error, "overlaps"):
            details["overlaps"] = [asdict(overlap) for overlap in error.overlaps]
            details["overlap_reason_counts"] = dict(sorted(Counter(
                overlap.reason for overlap in error.overlaps).items()))
        return details
    raise AssertionError(f"Invalid {name} was accepted")


def execute(output, *, allow_code_changes=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Control output must be an empty directory; receipts are never overwritten")
    started, hashes = time.perf_counter(), _hashes()
    source_verification = verify_cloud_sources()
    mesh, origin, inventory, seed, load = physical_inputs()
    initial = state = _initialize(mesh, origin, inventory, seed, load)
    assert initial.states == seed.states
    assert initial.transverse_displacement_m == seed.transverse_displacement_m
    assert not state.receipts and not state.buried_registry.records
    user = dict(control="physical differential source-generated underthrust", original_point_memory=True)
    save_underthrust_motion_checkpoint(output/"initial.json", mesh, origin, initial, provenance=user)
    restored_origin, restored_initial, restored_user = load_underthrust_motion_checkpoint(output/"initial.json", mesh=mesh)
    assert restored_origin == origin and _snapshot(restored_initial) == _snapshot(initial) and restored_user == user
    save_underthrust_motion_checkpoint(output/"initial_resaved.json", mesh, restored_origin,
                                      restored_initial, provenance=restored_user)
    first_source = SOURCE_INTERVALS[0]
    load = assemble_physical_spatial_mantle_load(mesh, state.current_surface, first_source)
    step = advance_underthrust_motion(mesh, origin, state, load, DT_S, provenance=first_source.provenance)
    births = list(step.geometric_transaction.births)
    rows = [_step_checks(origin, state, load, step, births)]
    state = step.state
    save_underthrust_motion_checkpoint(output/"direct_final.json", mesh, origin, state, provenance=user)
    restarted_load = assemble_physical_spatial_mantle_load(mesh, restored_initial.current_surface, first_source)
    restarted_first = advance_underthrust_motion(mesh, restored_origin, restored_initial,
        restarted_load, DT_S, provenance=first_source.provenance)
    resumed = restarted_first.state
    assert _snapshot(resumed) == _snapshot(state)
    assert _snapshot(restarted_first) == _snapshot(step)
    save_underthrust_motion_checkpoint(output/"resumed_final.json", mesh, origin, resumed, provenance=restored_user)
    final_origin, final_restored, final_user = load_underthrust_motion_checkpoint(output/"direct_final.json", mesh=mesh)
    assert final_origin == origin and _snapshot(final_restored) == _snapshot(state) and final_user == user
    save_underthrust_motion_checkpoint(output/"final_resaved.json", mesh, final_origin,
                                      final_restored, provenance=final_user)
    pairs = (("initial.json", "initial_resaved.json"), ("direct_final.json", "resumed_final.json"),
             ("direct_final.json", "final_resaved.json"))
    byte_pairs = {f"{a}|{b}": (output/a).read_bytes() == (output/b).read_bytes() for a,b in pairs}
    assert all(byte_pairs.values())
    validate_underthrust_motion_state(mesh, origin, state)
    accepted_checkpoint_bytes = (output/"direct_final.json").read_bytes()
    next_rejections = []
    for source in SOURCE_INTERVALS:
        current = assemble_physical_spatial_mantle_load(mesh, state.current_surface, source)
        direct_rejected = _expect_atomic_rejection("next interval: "+source.provenance, origin, state,
            lambda: advance_underthrust_motion(mesh, origin, state, current, DT_S, provenance=source.provenance))
        resumed_current = assemble_physical_spatial_mantle_load(mesh, final_restored.current_surface, source)
        resumed_rejected = _expect_atomic_rejection("next interval: "+source.provenance, origin, final_restored,
            lambda: advance_underthrust_motion(mesh, origin, final_restored, resumed_current,
                DT_S, provenance=source.provenance))
        assert direct_rejected == resumed_rejected
        assert "new_material_requires_new_physical_pair" in direct_rejected.get("overlap_reason_counts", {})
        direct_rejected["source"] = asdict(source)
        direct_rejected["direct_and_resumed_rejection_outputs_exact"] = True
        next_rejections.append(direct_rejected)
    assert (output/"direct_final.json").read_bytes() == accepted_checkpoint_bytes
    bad_load = replace(load, torque_nm=tuple(2.*value for value in load.torque_nm))
    bad_load = replace(bad_load, load_sha256=spatial_source_hash(spatial_source_body(bad_load)))
    rejected = [_expect_atomic_rejection("rehashed source torque", origin, initial,
        lambda: advance_underthrust_motion(mesh, origin, initial, bad_load, DT_S, provenance=PROVENANCE))]
    bad_state = replace(initial, elapsed_time_s=False)
    rejected.append(_expect_atomic_rejection("boolean clock alias", origin, bad_state,
        lambda: validate_underthrust_motion_state(mesh, origin, bad_state)))
    tight = _initialize(mesh, origin, inventory, seed, load, error_bound=1e-20)
    rejected.append(_expect_atomic_rejection("finite anchor transport error bound", origin, tight,
        lambda: advance_underthrust_motion(mesh, origin, tight, load, DT_S, provenance=PROVENANCE)))
    raw = underthrust_motion_to_dict(mesh, origin, state)
    raw["snapshot"]["free_anchor_fields"][0][1][0] += 1e-5
    rejected.append(_expect_atomic_rejection("changed accepted anchor snapshot", origin, state,
        lambda: underthrust_motion_from_dict(raw, mesh, origin)))
    direct_work = math.fsum(receipt.source_work_j for receipt in state.receipts)
    source_ledger_error = abs(state.cumulative_source_work_j-direct_work)/max(
        abs(state.cumulative_source_work_j), abs(direct_work), np.finfo(float).tiny)
    assert source_ledger_error < 4*np.finfo(float).eps
    assert source_verification == verify_cloud_sources()
    unchanged = hashes == _hashes()
    if not allow_code_changes:
        assert unchanged, "Implementation changed during control; rerun after code freeze"
    report = dict(status="passed", final_eligible=unchanged and not allow_code_changes,
        model="objective_incremental_original_material_pairs_spatial_underthrust",
        elapsed_seconds=time.perf_counter()-started,
        runtime=dict(python=platform.python_version(), platform=platform.platform(),
            packages={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "PyYAML", "pytest")}),
        repository=dict(branch=subprocess.check_output(("git","rev-parse","--abbrev-ref","HEAD"),
            cwd=ROOT,text=True).strip(),
            head=subprocess.check_output(("git","rev-parse","HEAD"),cwd=ROOT,text=True).strip()),
        physical_inputs=dict(accepted_first_source=asdict(SOURCE_INTERVALS[0]),
            independently_declared_next_source_candidates=[asdict(source) for source in SOURCE_INTERVALS],
            dt_s=DT_S, original_surface=asdict(origin), original_traces=asdict(seed.fault_traces),
            original_bending_banks=[asdict(bank) for bank in seed.bending_parameters],
            deformation_bounds=BOUNDS, max_anchor_transport_error_m=ANCHOR_ERROR_BOUND_M,
            birth_parameters=asdict(initial.birth_parameters)),
        retained_seed=dict(original_prepared_pose_retained=True, no_anchor_reset=True,
            original_point_and_radial_history_exact_at_initialization=True,
            preparation_work_j=seed.preparation_work_j, seed_source_work_j=seed.cumulative_source_work_j,
            initial_rotation_radians=seed.rotation_radians,
            initial_transverse_displacement_m=seed.transverse_displacement_m),
        accepted_physical_steps=rows, accepted_step_count=len(rows),
        original_point_count=len(state.states), original_trace_count=len(seed.fault_traces.traces),
        total_admitted_loss_count=sum(row["loss_count"] for row in rows),
        total_admitted_area_km2=math.fsum(row["buried"]["area_km2"] for row in rows),
        minimum_admitted_area_km2=min(row["minimum_admitted_area_km2"] for row in rows),
        minimum_hot_birth_area_km2=min(row["minimum_hot_birth_area_km2"] for row in rows),
        positive_material_events_are_never_filtered=True,
        total_hot_birth_count=sum(row["birth_count"] for row in rows),
        final_archive_cohort_count=len(state.current_inventory.cohorts),
        material_budget_max_relative_residual=_ledger(origin,state,births),
        source_work=dict(sum_of_accepted_segment_work_j=direct_work,
            saved_cumulative_work_j=state.cumulative_source_work_j,
            cumulative_ledger_relative_error=source_ledger_error,
            history_work_not_reassigned=True, rejected_next_sources_add_no_work=True,
            changing_source_continuation_not_demonstrated=True),
        maximum_anchor_transport_error_m=max(row["diagnostics"]["anchor_transport_error_m"] for row in rows),
        maximum_force_relative_residual=max(row["diagnostics"]["torque_signal_relative_residual"] for row in rows),
        maximum_transverse_force_relative_residual=max(row["diagnostics"]["transverse_force_relative_residual"] for row in rows),
        maximum_incremental_work_relative_error=max(row["diagnostics"]["incremental_energy_balance_relative_error"] for row in rows),
        maximum_cumulative_work_relative_error=max(row["diagnostics"]["cumulative_energy_balance_relative_error"] for row in rows),
        atomic_rejections=rejected, unsupported_next_intervals=next_rejections,
        restart_exact=True, accepted_first_outputs_exact=True,
        rejected_next_step_outputs_exact=True, accepted_checkpoint_bytes_unchanged_after_rejections=True,
        checkpoint_bytes_identical=byte_pairs,
        checkpoint_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output.glob("*.json"))},
        cloud_sources=source_verification,
        cloud_source_bundle_sha256={name:hashlib.sha256((ROOT/"handoff"/name).read_bytes()).hexdigest()
            for name in ("cloud_sources.zip","cloud_sources_manifest.json")},
        code_sha256=hashes,code_unchanged=unchanged,
        scope="One loaded, small-step first-order motion of original material-bank pairs: genuine differential force results drive every positive overlap burial and hot vacancy closure, including microscopic topology, under continuous spatial constitutive permission. Original complete 2D live/buried material domains and live clamps remain supported and every new cohort has resolved support. The next repeated or explicitly changed source interval is rejected because actual newborn material needs new physical pairs; unloading and unsupported partial arcs may also block. Initial restart reproduces the accepted first result and bytes exactly; current restart reproduces both next-step rejections exactly without changing accepted state or work. Continuous multi-source underthrust is not demonstrated. The spatial source is an explicitly selected manufactured control; cooling, general physical rematching, stress-generated faults and full slab dynamics remain outside this model.")
    filename = "smoke_report.json" if allow_code_changes else "report.json"
    with (output/filename).open("x",encoding="utf-8") as handle:
        json.dump(report,handle,indent=2,allow_nan=False)
        handle.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--allow-code-changes",action="store_true",
        help="Smoke output only; never certifies a final frozen implementation")
    args = parser.parse_args()
    result = execute(args.out,allow_code_changes=args.allow_code_changes)
    print(json.dumps({name:result[name] for name in ("status","final_eligible","elapsed_seconds",
        "accepted_step_count","total_admitted_loss_count","total_admitted_area_km2",
        "final_archive_cohort_count","material_budget_max_relative_residual",
        "maximum_anchor_transport_error_m","maximum_force_relative_residual",
        "maximum_incremental_work_relative_error","restart_exact",
        "accepted_first_outputs_exact","rejected_next_step_outputs_exact","checkpoint_bytes_identical")},indent=2))
