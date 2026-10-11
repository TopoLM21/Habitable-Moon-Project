"""Resolved contact, continuum convergence and nonempty-archive subdivision.

The physical arcs, full thermal lid, weak planes and preload are declared
synthetic inputs. Source and drag use existing SI basal quadrature. Surface
motion used to create the archival fixture is prescribed separately, before
the frozen mechanical experiment; solved slip never creates slab material.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from analysis.geometric_initiation_validation.admission_case import fixture, snapshot
from analysis.geometric_contact_dynamics_validation.dynamics_case import _inputs, SOURCE_PROVENANCE
from tectonics.basal_coupling import PrescribedBasalParameters
from tectonics.fractional_surface import EXTENSIVE_FIELDS, SurfaceParcel
from tectonics.geometric_basal_loading import (
    assemble_geometric_basal_load, basal_load_from_dict, basal_load_to_dict,
)
from tectonics.geometric_boundary import (
    consume_geometric_transaction, initialize_boundary, polarity_evidence,
)
from tectonics.geometric_boundary_io import load_boundary_checkpoint, save_boundary_checkpoint
from tectonics.geometric_contact_subdivision import rebind_distributed_boundary_subdivision
from tectonics.geometric_contacts import extract_contacts
from tectonics.geometric_distributed_contact_dynamics import (
    advance_distributed_contact_dynamics, continuous_contact_gap_bounds,
    distributed_contact_dynamics_from_dict, distributed_contact_dynamics_to_dict,
    initialize_distributed_contact_dynamics,
)
from tectonics.geometric_fault_traces import PhysicalFaultTrace, TraceQuadrature, declare_fault_trace_set
from tectonics.geometric_material_subdivision import validate_material_subdivision
from tectonics.geometric_polarity import PolarityEvidence
from tectonics.geometric_surface import split_geometry, totals
from tectonics.geometric_transport import advance_geometric_surface
from tectonics.spherical_polygons import clip_hemisphere

DT_S = 1e11
STEP_COUNT = 8


def _hashes():
    files = sorted((ROOT/"tectonics").glob("*.py"))
    files += [ROOT/"analysis/geometric_initiation_validation/admission_case.py",
              ROOT/"analysis/geometric_contact_dynamics_validation/dynamics_case.py", Path(__file__).resolve()]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def _physical_traces(surface, experiments):
    contacts = {c.contact_id: c for c in extract_contacts(surface.fragments, surface.radius_km)}
    result = []
    for experiment in experiments:
        fault = experiment.fault
        contact = contacts[fault.contact_id]
        h = np.asarray(contact.normal_a_to_b)*(1. if fault.overriding_plate == contact.plate_b else -1.)
        start, end = contact.start, contact.end
        if np.dot(np.cross(start, end), h) < 0.:
            start, end = end, start
        result.append(PhysicalFaultTrace(fault.fault_id, start, end, fault.subducting_plate,
            fault.overriding_plate, fault.dip_radians, experiment.columns, experiment.law,
            experiment.contact_height_m, experiment.water, 0., -20., 0.,
            "Declared persistent synthetic physical arc; endpoints orient sub-to-over; "
            "fixed columns, weak law and preload, independent of later material cuts"))
    return tuple(sorted(result, key=lambda t: t.trace_id))


def _prepare(surface, traces, load, *, panels=1, order=4, initial_dt=DT_S):
    trace_set = declare_fault_trace_set(surface, traces, TraceQuadrature(panels, order))
    return initialize_distributed_contact_dynamics(surface, trace_set, load,
        initial_loading_dt_s=initial_dt, max_rotation_radians=.005)


def _metadata(surface, state, load):
    return dict(basal_source=basal_load_to_dict(surface, load),
                distributed_contact=distributed_contact_dynamics_to_dict(surface, state, load))


def _row(step, surface):
    branches = {trace.trace_id: dict(locked=0, sliding=0) for trace in step.state.fault_traces.traces}
    points = []
    for point in step.points:
        branch = "sliding" if point.step.slip_increment_m != 0. else "locked"
        branches[point.trace_id][branch] += 1
        points.append(dict(point_id=point.point_id, trace_id=point.trace_id, position=point.position,
            area_m2=point.area_m2, branch=branch, traction_pa=point.step.traction_pa,
            slip_increment_m=point.step.slip_increment_m,
            plastic_slip_m=point.step.state.plastic_slip_m,
            gap_free_m=point.step.state.gap_free_m, jump_free_m=point.step.state.jump_free_m,
            force_on_sub_n=point.force_on_sub_n, torque_on_sub_nm=point.torque_on_sub_nm))
    metrics = {name: getattr(step, name) for name in (
        "angular_velocity_rad_s", "source_work_j", "basal_dissipation_j", "contact_work_j",
        "stored_energy_increment_j", "friction_work_j", "viscous_work_j", "numerical_dissipation_j",
        "energy_balance_residual_j", "torque_signal_relative_residual",
        "incremental_energy_balance_relative_error", "minimum_certified_eigenvalue", "iterations")}
    assert metrics["torque_signal_relative_residual"] < 2e-10
    assert metrics["incremental_energy_balance_relative_error"] < 2e-10
    return dict(step=step.state.step_count, branches=branches, points=points, **metrics,
        rotation_radians=step.state.rotation_radians,
        continuous_gap_bounds_m=continuous_contact_gap_bounds(surface,
            step.state.fault_traces, step.state.plate_ids, step.state.rotation_radians))


def _arc_moments(trace, radius):
    start, end = np.asarray(trace.start), np.asarray(trace.end)
    angle = math.atan2(np.linalg.norm(np.cross(start, end)), float(start@end))
    h = np.cross(start, end)/np.linalg.norm(np.cross(start, end))
    middle = (start+end)/np.linalg.norm(start+end)
    tangent = np.cross(h, middle)
    beta, gamma = np.cross(middle, h), np.cross(tangent, h)
    first = 2.*radius**2*math.sin(angle/2.)*beta
    second = radius**3*(.5*(angle+math.sin(angle))*np.outer(beta, beta)
                       +.5*(angle-math.sin(angle))*np.outer(gamma, gamma))
    return first, second


def _continuum_control(surface, traces, load):
    # Unequal declared heights avoid symmetry cancellation of quadrature error.
    # Inverse preload scaling preserves a uniform initial line resultant.
    declared = tuple(replace(trace, contact_height_m=trace.contact_height_m*(1.+.2*i),
        initial_gap_m=trace.initial_gap_m/(1.+.2*i),
        law=replace(trace.law, cohesion_pa=1e7),
        provenance=trace.provenance+"; independent elastic oracle with declared unequal heights")
        for i, trace in enumerate(traces))
    ids = load.plate_ids
    positions = {pid: i for i, pid in enumerate(ids)}
    stiffness, preload = np.zeros_like(load.drag_nm_s, dtype=float), np.zeros(len(load.torque_nm))
    for trace in declared:
        first, second = _arc_moments(trace, surface.radius_km*1000.)
        sine, cosine = math.sin(trace.dip_radians), math.cos(trace.dip_radians)
        alpha = 1./(1./trace.columns.normal_stiffness_pa_m+1./trace.law.normal_stiffness_pa_m)
        shear = 1./(1./trace.columns.shear_stiffness_pa_m+1./trace.law.tangential_stiffness_pa_m)
        factor = trace.contact_height_m/sine
        block = factor*(alpha*sine*sine+shear*cosine*cosine)*second
        sub, over = (slice(3*positions[p], 3*positions[p]+3)
                     for p in (trace.subducting_plate, trace.overriding_plate))
        stiffness[sub, sub] += block; stiffness[over, over] += block
        stiffness[sub, over] -= block; stiffness[over, sub] -= block
        free = factor*(alpha*sine*trace.initial_gap_m+shear*cosine*trace.initial_jump_m)*first
        preload[sub] -= free; preload[over] += free
    dt = 1e10
    expected = np.linalg.solve(np.asarray(load.drag_nm_s)+dt*stiffness,
                              np.asarray(load.torque_nm)-preload)
    rows = []
    for panels, order in ((1, 2), (1, 4), (2, 4), (1, 8)):
        initial = _prepare(surface, declared, load, panels=panels, order=order)
        step = advance_distributed_contact_dynamics(surface, initial, load, dt)
        assert all(point.step.slip_increment_m == 0. for point in step.points)
        relative = np.linalg.norm(np.asarray(step.angular_velocity_rad_s).ravel()-expected)/np.linalg.norm(expected)
        rows.append(dict(panel_count=panels, gauss_order=order, points=len(step.points),
                         angular_velocity_relative_error=float(relative)))
    assert rows[-1]["angular_velocity_relative_error"] < 2e-11
    assert rows[-1]["angular_velocity_relative_error"] < rows[0]["angular_velocity_relative_error"]
    return dict(declared_elastic_traces=[asdict(t) for t in declared], step_duration_s=dt,
                exact_angular_velocity_rad_s=expected.reshape((-1, 3)).tolist(), quadrature_cases=rows)


def _archive_fixture():
    mesh, surface = fixture()
    # Explicitly unequal histories give the old prescribed transport a unique
    # donor; this only builds pre-existing passive test data.
    surface = replace(surface, fragments=tuple(replace(f, parcel=replace(f.parcel,
        age_myr=10. if f.parcel.plate == 0 else 5.,
        density_excess_mass_kg=f.parcel.area_km2*(2e12 if f.parcel.plate == 0 else 1e12),
        specific_properties=()))
        for f in surface.fragments))

    def birth(cell, plate, area, moment, serial):
        return SurfaceParcel(cell, plate, f"birth:{moment.hex()}:{serial}", area, 2.*area,
                             0., 0., 0., (("damage", 0.),))

    motion = np.array([[0., 0., 0.], [0., 0., .01]])
    result = advance_geometric_surface(mesh, surface, motion, .001, birth_factory=birth)
    archive = consume_geometric_transaction(surface, result, initialize_boundary(surface), motion, .001)
    return mesh, result.state, archive


def _archive_cut_control(output, template):
    mesh, before, archive = _archive_fixture()
    assert archive.cohorts
    parameters = PrescribedBasalParameters(17, 8e4, 1e14, 2.)
    lid = {f.fragment_id: .2+1.3*i for i, f in enumerate(before.fragments)}
    provenance = "Explicit synthetic nonempty-archive frozen source and full thermal lid"
    load = assemble_geometric_basal_load(mesh, before, parameters, lid, provenance)
    traces = []
    for i, contact in enumerate(extract_contacts(before.fragments, before.radius_km)):
        h = np.asarray(contact.normal_a_to_b)*(1. if contact.plate_b == 1 else -1.)
        a, b = contact.start, contact.end
        if np.cross(a, b)@h < 0.: a, b = b, a
        traces.append(replace(template, trace_id=f"archive-physical:{i}", start=a, end=b,
            subducting_plate=0, overriding_plate=1, initial_gap_m=-1.,
            law=replace(template.law, cohesion_pa=1e7),
            provenance="Explicit frozen physical arc in prescribed archival fixture"))
    trace_set = declare_fault_trace_set(before, traces, TraceQuadrature(1, 4))
    origin = tuple(PolarityEvidence(**row) for row in polarity_evidence(archive))
    state = initialize_distributed_contact_dynamics(before, trace_set, load,
        initial_loading_dt_s=1e6, max_rotation_radians=.005, history=origin)
    first = advance_distributed_contact_dynamics(before, state, load, 1e8).state
    cohort = next(row for row in archive.cohorts if row.supports)
    support = cohort.supports[0]
    receiver = next(f for f in before.fragments if f.fragment_id == support.receiver_fragment_id)
    midpoint = np.asarray(support.start)+np.asarray(support.end)
    midpoint /= np.linalg.norm(midpoint)
    normal = np.cross(midpoint, np.asarray(receiver.polygon).sum(axis=0))
    children = split_geometry(receiver,
        (clip_hemisphere(receiver.polygon, normal), clip_hemisphere(receiver.polygon, -normal)),
        label="physical-history-control-cut")
    after = replace(before, fragments=tuple(f for f in before.fragments
        if f.fragment_id != receiver.fragment_id)+children)
    mapping = validate_material_subdivision(before, after)
    new_lid = {child: lid[parent] for child, parent in mapping}
    new_load = assemble_geometric_basal_load(mesh, after, parameters, new_lid, provenance)
    rebound, new_archive = rebind_distributed_boundary_subdivision(before, after, first, load, new_load, archive)
    assert rebound.states == first.states and rebound.polarity_history == first.polarity_history
    assert rebound.preparation_work_j == first.preparation_work_j
    assert new_archive.transactions == archive.transactions
    payloads = lambda inv: tuple((c.event_id, c.payload_sha256, c.loss) for c in inv.cohorts)
    assert payloads(new_archive) == payloads(archive)
    save_boundary_checkpoint(output/"archive_after_cut.json", after, new_archive,
                             provenance=_metadata(after, rebound, new_load))
    loaded, inventory, metadata = load_boundary_checkpoint(output/"archive_after_cut.json")
    restored_load = basal_load_from_dict(metadata["basal_source"], loaded, mesh=mesh,
        parameters=parameters, thermal_lid_by_fragment=new_lid, provenance=provenance)
    restored = distributed_contact_dynamics_from_dict(metadata["distributed_contact"], loaded, restored_load)
    direct = advance_distributed_contact_dynamics(after, rebound, new_load, 1e8).state
    resumed = advance_distributed_contact_dynamics(loaded, restored, restored_load, 1e8).state
    uncut = advance_distributed_contact_dynamics(before, first, load, 1e8).state
    assert direct == resumed
    difference = np.linalg.norm(np.asarray(direct.rotation_radians)-np.asarray(uncut.rotation_radians))
    relative = difference/np.linalg.norm(uncut.rotation_radians)
    assert relative < 3e-11
    save_boundary_checkpoint(output/"archive_direct_final.json", after, new_archive,
                             provenance=_metadata(after, direct, new_load))
    save_boundary_checkpoint(output/"archive_resumed_final.json", loaded, inventory,
                             provenance=_metadata(loaded, resumed, restored_load))
    assert (output/"archive_direct_final.json").read_bytes() == (output/"archive_resumed_final.json").read_bytes()
    for field in EXTENSIVE_FIELDS:
        old, new = totals(before.fragments)[field], totals(after.fragments)[field]
        assert math.isclose(old, new, rel_tol=64*np.finfo(float).eps, abs_tol=0.)
    return dict(cohort_count=len(archive.cohorts), surface_fragments_before=len(before.fragments),
        surface_fragments_after=len(after.fragments), original_contact_count=len(archive.contacts),
        new_contact_count=len(new_archive.contacts), physical_point_count=len(rebound.states),
        physical_histories_unchanged_at_rebind=True, retained_polarity_support_count=len(rebound.polarity_history),
        original_evidence=[asdict(item) for item in origin],
        representation_records=[asdict(item) for item in new_archive.representations],
        original_acceptance_transactions_preserved=True, cohort_payloads_preserved=True,
        surface_extensive_residual={field: totals(after.fragments)[field]-totals(before.fragments)[field]
                                    for field in EXTENSIVE_FIELDS},
        cut_vs_uncut_rotation_relative_difference=float(relative),
        restart_exact=True, checkpoint_bytes_identical=True,
        note="Material subdivision changes support bookkeeping; it neither accepts new slab mass nor advances a clock")


def execute(output):
    started, hashes = time.perf_counter(), _hashes()
    output = Path(output)
    if output.exists(): raise ValueError("Validation output must be new")
    output.mkdir(parents=True)
    mesh, surface = fixture()
    archive = initialize_boundary(surface)
    original = snapshot(surface), snapshot(archive)
    parameters, lid, experiments, _ = _inputs(surface)
    traces = _physical_traces(surface, experiments)
    load = assemble_geometric_basal_load(mesh, surface, parameters, lid, SOURCE_PROVENANCE)
    current = _prepare(surface, traces, load)
    preparation = distributed_contact_dynamics_to_dict(surface, current, load)
    rows = []
    for index in range(STEP_COUNT):
        step = advance_distributed_contact_dynamics(surface, current, load, DT_S)
        current = step.state
        rows.append(_row(step, surface))
        if index == 2:
            save_boundary_checkpoint(output/"middle.json", surface, archive,
                                     provenance=_metadata(surface, current, load))
    save_boundary_checkpoint(output/"direct_final.json", surface, archive,
                             provenance=_metadata(surface, current, load))
    loaded, inventory, metadata = load_boundary_checkpoint(output/"middle.json")
    restored_load = basal_load_from_dict(metadata["basal_source"], loaded, mesh=mesh,
        parameters=parameters, thermal_lid_by_fragment=lid, provenance=SOURCE_PROVENANCE)
    resumed = distributed_contact_dynamics_from_dict(metadata["distributed_contact"], loaded, restored_load)
    for _ in range(3, STEP_COUNT):
        resumed = advance_distributed_contact_dynamics(loaded, resumed, restored_load, DT_S).state
    assert resumed == current
    save_boundary_checkpoint(output/"resumed_final.json", loaded, inventory,
                             provenance=_metadata(loaded, resumed, restored_load))
    assert (output/"direct_final.json").read_bytes() == (output/"resumed_final.json").read_bytes()
    assert (snapshot(surface), snapshot(archive)) == original
    spatial = []
    previous = current
    for panels in (2, 4):
        fine = _prepare(surface, traces, load, panels=panels, order=4)
        for _ in range(STEP_COUNT): fine = advance_distributed_contact_dynamics(surface, fine, load, DT_S).state
        delta = np.linalg.norm(np.asarray(previous.rotation_radians)-np.asarray(fine.rotation_radians))
        spatial.append(dict(panel_count=panels, gauss_order=4, points=len(fine.states),
            difference_from_previous_layout_rad=float(delta),
            relative_difference_from_previous_layout=float(delta/np.linalg.norm(fine.rotation_radians))))
        previous = fine
    continuum = _continuum_control(surface, traces, load)
    archived = _archive_cut_control(output, traces[0])
    mixed = sum(branch["locked"] > 0 and branch["sliding"] > 0
                for row in rows for branch in row["branches"].values())
    assert mixed > 0
    report = dict(format="geometric-distributed-contact-validation-1", status="complete",
        elapsed_seconds=time.perf_counter()-started,
        runtime=dict(python=platform.python_version(), platform=platform.platform(),
            packages={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "PyYAML", "pytest")}),
        source_model="frozen eight-octant/two-hemisphere synthetic source; no speed target",
        inputs=dict(basal_source=basal_load_to_dict(surface, load), step_duration_s=DT_S,
            step_count=STEP_COUNT, initial_preparation=preparation),
        load_steps=rows, final_state=distributed_contact_dynamics_to_dict(surface, current, load),
        point_count=len(current.states), mixed_branch_trace_steps=mixed,
        sliding_point_steps=sum(branch["sliding"] for row in rows for branch in row["branches"].values()),
        maximum_resultant_torque_relative_residual=max(row["torque_signal_relative_residual"] for row in rows),
        maximum_incremental_work_relative_residual=max(row["incremental_energy_balance_relative_error"] for row in rows),
        spatial_sensitivity=dict(layouts=spatial,
            interpretation="Separate fresh histories on finer fixed grids; nonlinear spatial sensitivity, not history regridding or a convergence claim"),
        independent_continuum_elastic_control=continuum, nonempty_archive_subdivision=archived,
        source_surface_preserved=True, main_passive_archive_preserved=True,
        transferred_material_during_mechanical_steps_km2=0., new_slab_cohorts_during_mechanical_steps=0,
        restart_exact=True, checkpoint_bytes_identical=True,
        checkpoint_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.glob("*.json"))},
        cloud_source_bundle_sha256={name: hashlib.sha256((ROOT/"handoff"/name).read_bytes()).hexdigest()
            for name in ("cloud_sources.zip", "cloud_sources_manifest.json")},
        code_sha256=hashes, code_unchanged=hashes == _hashes(),
        scope="Fixed physical dipping arcs and quadrature; continuous closure envelope; shared Euler/contact balance; "
              "verified material subdivision preserves mechanical history and archival payloads; no solved material motion, "
              "fault advection, bending, vector strike friction, thermal feedback or slab pull")
    assert report["code_unchanged"]
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    result = execute(parser.parse_args().out)
    print(json.dumps({name: result[name] for name in (
        "status", "elapsed_seconds", "point_count", "mixed_branch_trace_steps", "sliding_point_steps",
        "maximum_resultant_torque_relative_residual", "maximum_incremental_work_relative_residual",
        "restart_exact", "checkpoint_bytes_identical", "spatial_sensitivity",
        "nonempty_archive_subdivision")}, indent=2))
