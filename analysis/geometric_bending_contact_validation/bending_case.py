"""Frozen physical contact with independent, declared two-bank bending.

The strip widths and elastic materials below are explicit synthetic inputs,
not inferred from fragment geometry or a target speed. The linear continuum
oracle uses exact great-circle first and second moments and a directly
integrated cantilever stiffness; it never reads solver assembly rows.
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
from analysis.geometric_distributed_contact_validation.trace_case import _archive_fixture, _physical_traces
from tectonics.basal_coupling import PrescribedBasalParameters
from tectonics.fractional_surface import EXTENSIVE_FIELDS
from tectonics.geometric_basal_loading import (
    assemble_geometric_basal_load, basal_load_from_dict, basal_load_to_dict,
)
from tectonics.geometric_bending import ElasticBendingBankParameters, FaultBendingParameters
from tectonics.geometric_bending_contact_dynamics import (
    advance_bending_contact_dynamics, bending_contact_dynamics_from_dict,
    bending_contact_dynamics_to_dict, initialize_bending_contact_dynamics,
)
from tectonics.geometric_bending_contact_subdivision import rebind_bending_boundary_subdivision
from tectonics.geometric_boundary import initialize_boundary, polarity_evidence
from tectonics.geometric_boundary_io import load_boundary_checkpoint, save_boundary_checkpoint
from tectonics.geometric_contacts import extract_contacts
from tectonics.geometric_fault_traces import TraceQuadrature, declare_fault_trace_set
from tectonics.geometric_material_subdivision import validate_material_subdivision
from tectonics.geometric_polarity import PolarityEvidence
from tectonics.geometric_surface import split_geometry, totals
from tectonics.spherical_polygons import clip_hemisphere

DT_S = 1e10
STEP_COUNT = 8
LIMITS = dict(max_rotation_radians=.005, max_tip_slope=.05,
              max_bending_strain=.005, max_deflection_radius_ratio=.005)


def _hashes():
    files = sorted((ROOT/"tectonics").glob("*.py"))
    files += [ROOT/"analysis/geometric_initiation_validation/admission_case.py",
              ROOT/"analysis/geometric_contact_dynamics_validation/dynamics_case.py",
              ROOT/"analysis/geometric_distributed_contact_validation/trace_case.py",
              Path(__file__).resolve()]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def _banks(traces):
    return tuple(FaultBendingParameters(trace.trace_id,
        ElasticBendingBankParameters(20000., 4000., 70e9, .25,
            "Declared synthetic sub-bank: independent 4 km elastic thickness and 20 km inward clamp distance"),
        ElasticBendingBankParameters(25000., 4000., 65e9, .23,
            "Declared synthetic over-bank: independent 4 km elastic thickness and 25 km inward clamp distance"))
        for trace in traces)


def _prepare(surface, traces, load, *, panels=1, order=4, initial_dt=DT_S,
             banks=None, history=()):
    trace_set = declare_fault_trace_set(surface, traces, TraceQuadrature(panels, order))
    return initialize_bending_contact_dynamics(surface, trace_set, load,
        _banks(traces) if banks is None else banks,
        initial_loading_dt_s=initial_dt, history=history, **LIMITS)


def _metadata(surface, state, load):
    return dict(basal_source=basal_load_to_dict(surface, load),
                bending_contact=bending_contact_dynamics_to_dict(surface, state, load))


def _coordinates(state, radius):
    return np.r_[np.asarray(state.rotation_radians).ravel(),
        np.asarray([item[1:] for item in state.transverse_displacement_m]).ravel()/radius]


def _continuous_gaps(surface, state):
    # Derive the exact affine sinusoidal envelope independently. Constant
    # two-bank trench deflection changes its offset, not the stationary angles.
    q = np.asarray(state.rotation_radians)
    owners = {owner: index for index, owner in enumerate(state.plate_ids)}
    tips = {identity: (sub, over) for identity, sub, over in state.transverse_displacement_m}
    rows = []
    for trace in state.fault_traces.traces:
        start, end = np.asarray(trace.start), np.asarray(trace.end)
        horizontal = np.cross(start, end)/np.linalg.norm(np.cross(start, end))
        tangent = np.cross(horizontal, start)
        extent = math.atan2(np.linalg.norm(np.cross(start, end)), float(start@end))
        relative = q[owners[trace.overriding_plate]]-q[owners[trace.subducting_plate]]
        vector = surface.radius_km*1000.*math.sin(trace.dip_radians)*np.cross(horizontal, relative)
        a, b = float(start@vector), float(tangent@vector)
        values = [a, a*math.cos(extent)+b*math.sin(extent)]
        stationary = math.atan2(b, a)
        for multiple in (-2, -1, 0, 1, 2):
            angle = stationary+multiple*math.pi
            if 0. < angle < extent:
                values.append(a*math.cos(angle)+b*math.sin(angle))
        sub, over = tips[trace.trace_id]
        offset = trace.initial_gap_m+math.cos(trace.dip_radians)*(over-sub)
        low, high = offset+min(values), offset+max(values)
        assert high <= 0.
        rows.append((trace.trace_id, low, high))
    return rows


def _row(step, surface, old):
    branches = {trace.trace_id: dict(locked=0, sliding=0) for trace in step.state.fault_traces.traces}
    points = []
    q = np.asarray(step.state.rotation_radians)-np.asarray(old.rotation_radians)
    old_tips = {identity: (sub, over) for identity, sub, over in old.transverse_displacement_m}
    tip_increments = {identity: np.asarray((sub, over))-old_tips[identity]
        for identity, sub, over in step.state.transverse_displacement_m}
    trace_by_id = {trace.trace_id: trace for trace in step.state.fault_traces.traces}
    owners = {owner: index for index, owner in enumerate(step.state.plate_ids)}
    reaction_work = []
    for point in step.points:
        branch = "sliding" if point.step.slip_increment_m != 0. else "locked"
        branches[point.trace_id][branch] += 1
        trace, radial = trace_by_id[point.trace_id], np.asarray(point.position)
        sub, over = owners[trace.subducting_plate], owners[trace.overriding_plate]
        sub_w, over_w = tip_increments[point.trace_id]
        u_sub = surface.radius_km*1000.*np.cross(q[sub], radial)+sub_w*radial
        u_over = surface.radius_km*1000.*np.cross(q[over], radial)+over_w*radial
        reaction_work.append(float(np.dot(point.force_on_sub_n, u_sub)+np.dot(point.force_on_over_n, u_over)))
        points.append(dict(point_id=point.point_id, trace_id=point.trace_id, position=point.position,
            area_m2=point.area_m2, branch=branch, traction_pa=point.step.traction_pa,
            slip_increment_m=point.step.slip_increment_m,
            plastic_slip_m=point.step.state.plastic_slip_m,
            gap_free_m=point.step.state.gap_free_m, jump_free_m=point.step.state.jump_free_m,
            force_on_sub_n=point.force_on_sub_n, force_on_over_n=point.force_on_over_n,
            torque_on_sub_nm=point.torque_on_sub_nm, torque_on_over_nm=point.torque_on_over_nm))
    metrics = {name: getattr(step, name) for name in (
        "angular_velocity_rad_s", "source_work_j", "basal_dissipation_j", "contact_work_j",
        "stored_energy_increment_j", "friction_work_j", "viscous_work_j", "numerical_dissipation_j",
        "energy_balance_residual_j", "torque_signal_relative_residual",
        "incremental_energy_balance_relative_error", "minimum_certified_eigenvalue", "iterations",
        "bending_energy_j", "bending_stored_energy_increment_j", "bending_numerical_dissipation_j",
        "bending_work_j", "transverse_velocity_m_s", "bending_force_n", "radial_contact_force_n",
        "transverse_residual_n", "transverse_force_relative_residual",
        "generalized_force_relative_residual", "transverse_force_rounding_tolerance_n")}
    assert metrics["torque_signal_relative_residual"] < 2e-9
    assert metrics["incremental_energy_balance_relative_error"] < 2e-9
    assert metrics["generalized_force_relative_residual"] < 2e-9
    work = math.fsum(reaction_work)
    scale = max(abs(work), abs(step.contact_work_j), np.finfo(float).tiny)
    relative = abs(work+step.contact_work_j)/scale
    assert relative < 2e-9
    banks = {pair.trace_id: pair for pair in step.state.bending_parameters}
    forces = {identity: (sub, over) for identity, sub, over in step.bending_force_n}
    tips = {identity: (sub, over) for identity, sub, over in step.state.transverse_displacement_m}
    beam_rows, independent_work, independent_energy, independent_remainder = [], [], [], []
    for trace in step.state.fault_traces.traces:
        extent = math.atan2(np.linalg.norm(np.cross(trace.start, trace.end)), float(np.asarray(trace.start)@np.asarray(trace.end)))
        length = surface.radius_km*1000.*extent
        pair = banks[trace.trace_id]
        for side, bank in enumerate((pair.sub, pair.over)):
            rigidity = bank.young_modulus_pa*bank.elastic_thickness_m**3/(12.*(1.-bank.poisson_ratio**2))
            spring = 3.*rigidity*length/bank.width_m**3
            tip, increment = tips[trace.trace_id][side], tip_increments[trace.trace_id][side]
            expected_force = -spring*tip
            assert math.isclose(forces[trace.trace_id][side], expected_force, rel_tol=2e-12, abs_tol=0.)
            independent_work.append(spring*tip*increment)
            independent_energy.append(.5*spring*tip*tip)
            independent_remainder.append(.5*spring*increment*increment)
            beam_rows.append(dict(trace_id=trace.trace_id, side="sub" if side == 0 else "over",
                stiffness_n_m=spring, tip_displacement_m=tip,
                restoring_force_n=expected_force,
                maximum_slope=1.5*abs(tip)/bank.width_m,
                maximum_fibre_strain=1.5*bank.elastic_thickness_m*abs(tip)/bank.width_m**2,
                deflection_radius_ratio=abs(tip)/(surface.radius_km*1000.)))
    for actual, expected in ((step.bending_work_j, math.fsum(independent_work)),
                             (step.bending_energy_j, math.fsum(independent_energy)),
                             (step.bending_numerical_dissipation_j, math.fsum(independent_remainder))):
        assert math.isclose(actual, expected, rel_tol=2e-11, abs_tol=0.)
    return dict(step=step.state.step_count, branches=branches, points=points, **metrics,
        rotation_radians=step.state.rotation_radians,
        transverse_displacement_m=step.state.transverse_displacement_m,
        declared_bank_diagnostics=beam_rows,
        physical_contact_reaction_work_j=work,
        physical_contact_virtual_work_relative_error=relative,
        continuous_gap_bounds_m=_continuous_gaps(surface, step.state))


def _arc_moments(trace, radius):
    start, end = np.asarray(trace.start), np.asarray(trace.end)
    extent = math.atan2(np.linalg.norm(np.cross(start, end)), float(start@end))
    horizontal = np.cross(start, end)/np.linalg.norm(np.cross(start, end))
    middle = (start+end)/np.linalg.norm(start+end)
    tangent = np.cross(horizontal, middle)
    beta, gamma = np.cross(middle, horizontal), np.cross(tangent, horizontal)
    first = 2.*radius**2*math.sin(extent/2.)*beta
    second = radius**3*(.5*(extent+math.sin(extent))*np.outer(beta, beta)
                       +.5*(extent-math.sin(extent))*np.outer(gamma, gamma))
    return radius*extent, first, second


def _exact_continuum_solution(surface, traces, load, banks, dt):
    radius, owners = surface.radius_km*1000., load.plate_ids
    q_size, dimension = 3*len(owners), 3*len(owners)+2*len(traces)
    stiffness, preload = np.zeros((dimension, dimension)), np.zeros(dimension)
    stiffness[:q_size, :q_size] = np.asarray(load.drag_nm_s)/dt
    source = np.r_[np.asarray(load.torque_nm), np.zeros(dimension-q_size)]
    bank_by_id = {item.trace_id: item for item in banks}
    spring_values = []
    for index, trace in enumerate(traces):
        length, first, second = _arc_moments(trace, radius)
        incidence = np.zeros((dimension, 3))
        sub, over = owners.index(trace.subducting_plate), owners.index(trace.overriding_plate)
        incidence[3*sub:3*sub+3] = -np.eye(3)
        incidence[3*over:3*over+3] = np.eye(3)
        vertical = np.zeros(dimension)
        vertical[q_size+2*index:q_size+2*index+2] = (-radius, radius)
        kn = 1./(trace.columns.sub_length_m/trace.columns.sub_normal_modulus_pa
            +trace.columns.over_length_m/trace.columns.over_normal_modulus_pa
            +1./trace.law.normal_stiffness_pa_m)
        kt = 1./(trace.columns.sub_length_m/trace.columns.sub_shear_modulus_pa
            +trace.columns.over_length_m/trace.columns.over_shear_modulus_pa
            +1./trace.law.tangential_stiffness_pa_m)
        sine, cosine = math.sin(trace.dip_radians), math.cos(trace.dip_radians)
        width, horizontal_first = trace.contact_height_m/sine, incidence@first
        stiffness += width*(kn*sine*sine+kt*cosine*cosine)*(incidence@second@incidence.T)
        mixed = width*(kn-kt)*sine*cosine*np.outer(horizontal_first, vertical)
        stiffness += mixed+mixed.T
        stiffness += width*length*(kn*cosine*cosine+kt*sine*sine)*np.outer(vertical, vertical)
        preload += width*(kn*trace.initial_gap_m*sine+kt*trace.initial_jump_m*cosine)*horizontal_first
        preload += width*length*(kn*trace.initial_gap_m*cosine-kt*trace.initial_jump_m*sine)*vertical
        pair = bank_by_id[trace.trace_id]
        for side, bank in enumerate((pair.sub, pair.over)):
            rigidity = bank.young_modulus_pa*bank.elastic_thickness_m**3/(12.*(1.-bank.poisson_ratio**2))
            spring = 3.*rigidity*length/bank.width_m**3
            position = q_size+2*index+side
            stiffness[position, position] += spring*radius**2
            spring_values.append(spring)
    coordinates = np.linalg.solve(stiffness, source-preload)
    expected_energy = .5*float(np.dot(spring_values, (coordinates[q_size:]*radius)**2))
    return coordinates, expected_energy


def _continuum_control(surface, traces, load):
    declared = tuple(replace(trace,
        contact_height_m=trace.contact_height_m*(1.+.2*i),
        initial_gap_m=trace.initial_gap_m/(1.+.2*i),
        law=replace(trace.law, cohesion_pa=1e9, friction_dry=0., friction_wet=0.),
        provenance=trace.provenance+"; explicit purely locked continuum oracle with unequal interface heights")
        for i, trace in enumerate(traces))
    banks, dt = _banks(declared), 1e10
    expected, energy = _exact_continuum_solution(surface, declared, load, banks, dt)
    q_size, radius = 3*len(load.plate_ids), surface.radius_km*1000.
    rows = []
    for panels, order in ((1, 2), (1, 4), (2, 4), (1, 8)):
        initial = _prepare(surface, declared, load, panels=panels, order=order, banks=banks)
        step = advance_bending_contact_dynamics(surface, initial, load, dt)
        assert all(point.step.slip_increment_m == 0. for point in step.points)
        actual = _coordinates(step.state, radius)
        relative = lambda a, b: float(np.linalg.norm(a-b)/np.linalg.norm(b))
        rows.append(dict(panel_count=panels, gauss_order=order, points=len(step.points),
            all_coordinates_relative_error=relative(actual, expected),
            rotation_relative_error=relative(actual[:q_size], expected[:q_size]),
            transverse_displacement_relative_error=relative(actual[q_size:], expected[q_size:]),
            bending_energy_relative_error=abs(step.bending_energy_j-energy)/energy))
    assert rows[-1]["all_coordinates_relative_error"] < 2e-9
    assert rows[-1]["all_coordinates_relative_error"] < rows[0]["all_coordinates_relative_error"]
    return dict(declared_elastic_traces=[asdict(t) for t in declared],
        bending_parameters=[asdict(item) for item in banks], step_duration_s=dt,
        exact_rotation_radians=expected[:q_size].reshape((-1, 3)).tolist(),
        exact_transverse_displacement_m=(expected[q_size:]*radius).reshape((-1, 2)).tolist(),
        exact_bending_energy_j=energy, quadrature_cases=rows,
        derivation="Exact great-circle integrals of b and b tensor b, constant two-bank w; normal strain sin(delta)*bq+cos(delta)*(w_over-w_sub), tangential strain cos(delta)*bq-sin(delta)*(w_over-w_sub); independent isotropic strip k=3*E*Te^3*arc_length/[12*(1-nu^2)*clamp_width^3]")


def _archive_cut_control(output, template):
    mesh, before, archive = _archive_fixture()
    assert archive.cohorts
    parameters = PrescribedBasalParameters(17, 8e4, 1e14, 2.)
    lid = {f.fragment_id: .2+1.3*i for i, f in enumerate(before.fragments)}
    provenance = "Explicit synthetic nonempty-archive frozen source and full thermal lid; bending stage"
    load = assemble_geometric_basal_load(mesh, before, parameters, lid, provenance)
    traces = []
    for index, contact in enumerate(extract_contacts(before.fragments, before.radius_km)):
        horizontal = np.asarray(contact.normal_a_to_b)*(1. if contact.plate_b == 1 else -1.)
        start, end = contact.start, contact.end
        if np.cross(start, end)@horizontal < 0.: start, end = end, start
        traces.append(replace(template, trace_id=f"archive-bending-physical:{index}",
            start=start, end=end, subducting_plate=0, overriding_plate=1, initial_gap_m=-1.,
            law=replace(template.law, cohesion_pa=1e9, friction_dry=0., friction_wet=0.),
            provenance="Explicit frozen physical arc in prescribed archival fixture with two declared bending banks"))
    traces = tuple(sorted(traces, key=lambda trace: trace.trace_id))
    origin = tuple(PolarityEvidence(**row) for row in polarity_evidence(archive))
    initial = _prepare(before, traces, load, initial_dt=1e6, history=origin)
    first = advance_bending_contact_dynamics(before, initial, load, 1e8).state
    cohort = next(row for row in archive.cohorts if row.supports)
    support = cohort.supports[0]
    receiver = next(f for f in before.fragments if f.fragment_id == support.receiver_fragment_id)
    midpoint = np.asarray(support.start)+np.asarray(support.end)
    midpoint /= np.linalg.norm(midpoint)
    normal = np.cross(midpoint, np.asarray(receiver.polygon).sum(axis=0))
    children = split_geometry(receiver,
        (clip_hemisphere(receiver.polygon, normal), clip_hemisphere(receiver.polygon, -normal)),
        label="physical-bending-history-control-cut")
    after = replace(before, fragments=tuple(f for f in before.fragments
        if f.fragment_id != receiver.fragment_id)+children)
    parents = validate_material_subdivision(before, after)
    new_lid = {child: lid[parent] for child, parent in parents}
    new_load = assemble_geometric_basal_load(mesh, after, parameters, new_lid, provenance)
    rebound, new_archive = rebind_bending_boundary_subdivision(before, after, first, load, new_load, archive)
    excluded = {"surface_sha256", "basal_source_sha256", "fault_traces"}
    assert {key: value for key, value in asdict(rebound).items() if key not in excluded} == {
        key: value for key, value in asdict(first).items() if key not in excluded}
    assert new_archive.transactions == archive.transactions
    payloads = lambda inv: tuple((c.event_id, c.payload_sha256, c.loss) for c in inv.cohorts)
    assert payloads(new_archive) == payloads(archive)
    save_boundary_checkpoint(output/"archive_after_cut.json", after, new_archive,
                             provenance=_metadata(after, rebound, new_load))
    loaded, inventory, metadata = load_boundary_checkpoint(output/"archive_after_cut.json")
    restored_load = basal_load_from_dict(metadata["basal_source"], loaded, mesh=mesh,
        parameters=parameters, thermal_lid_by_fragment=new_lid, provenance=provenance)
    restored = bending_contact_dynamics_from_dict(metadata["bending_contact"], loaded, restored_load)
    direct = advance_bending_contact_dynamics(after, rebound, new_load, 1e8).state
    resumed = advance_bending_contact_dynamics(loaded, restored, restored_load, 1e8).state
    uncut = advance_bending_contact_dynamics(before, first, load, 1e8).state
    assert direct == resumed
    radius = before.radius_km*1000.
    relative = float(np.linalg.norm(_coordinates(direct, radius)-_coordinates(uncut, radius))
                     /np.linalg.norm(_coordinates(uncut, radius)))
    assert relative < 3e-10
    save_boundary_checkpoint(output/"archive_direct_final.json", after, new_archive,
                             provenance=_metadata(after, direct, new_load))
    save_boundary_checkpoint(output/"archive_resumed_final.json", loaded, inventory,
                             provenance=_metadata(loaded, resumed, restored_load))
    assert (output/"archive_direct_final.json").read_bytes() == (output/"archive_resumed_final.json").read_bytes()
    old_totals, new_totals = totals(before.fragments), totals(after.fragments)
    for field in EXTENSIVE_FIELDS:
        assert math.isclose(old_totals[field], new_totals[field], rel_tol=64*np.finfo(float).eps, abs_tol=0.)
    return dict(cohort_count=len(archive.cohorts), surface_fragments_before=len(before.fragments),
        surface_fragments_after=len(after.fragments), original_contact_count=len(archive.contacts),
        new_contact_count=len(new_archive.contacts), physical_point_count=len(rebound.states),
        physical_histories_unchanged_at_rebind=True, bending_parameters_unchanged_at_rebind=True,
        transverse_displacements_and_bending_ledgers_unchanged_at_rebind=True,
        retained_polarity_support_count=len(rebound.polarity_history),
        original_evidence=[asdict(item) for item in origin],
        representation_records=[asdict(item) for item in new_archive.representations],
        original_acceptance_transactions_preserved=True, cohort_payloads_preserved=True,
        surface_extensive_residual={field: new_totals[field]-old_totals[field] for field in EXTENSIVE_FIELDS},
        cut_vs_uncut_full_coordinates_relative_difference=relative,
        restart_exact=True, checkpoint_bytes_identical=True,
        note="Pure material subdivision preserves physical strips and their history, remaps passive support without accepting mass, and advances neither mechanical nor surface time")


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
    preparation = bending_contact_dynamics_to_dict(surface, current, load)
    rows = []
    for index in range(STEP_COUNT):
        step = advance_bending_contact_dynamics(surface, current, load, DT_S)
        rows.append(_row(step, surface, current))
        current = step.state
        if index == 2:
            save_boundary_checkpoint(output/"middle.json", surface, archive,
                                     provenance=_metadata(surface, current, load))
    save_boundary_checkpoint(output/"direct_final.json", surface, archive,
                             provenance=_metadata(surface, current, load))
    loaded, inventory, metadata = load_boundary_checkpoint(output/"middle.json")
    restored_load = basal_load_from_dict(metadata["basal_source"], loaded, mesh=mesh,
        parameters=parameters, thermal_lid_by_fragment=lid, provenance=SOURCE_PROVENANCE)
    resumed = bending_contact_dynamics_from_dict(metadata["bending_contact"], loaded, restored_load)
    for _ in range(3, STEP_COUNT):
        resumed = advance_bending_contact_dynamics(loaded, resumed, restored_load, DT_S).state
    assert resumed == current and restored_load == load
    save_boundary_checkpoint(output/"resumed_final.json", loaded, inventory,
                             provenance=_metadata(loaded, resumed, restored_load))
    assert (output/"direct_final.json").read_bytes() == (output/"resumed_final.json").read_bytes()
    assert (snapshot(surface), snapshot(archive)) == original
    assert snapshot(loaded) == original[0] and snapshot(inventory) == original[1]
    radius, previous = surface.radius_km*1000., current
    spatial = []
    for panels in (2, 4):
        fine = _prepare(surface, traces, load, panels=panels, order=4)
        for _ in range(STEP_COUNT):
            fine = advance_bending_contact_dynamics(surface, fine, load, DT_S).state
        old_vector, new_vector = _coordinates(previous, radius), _coordinates(fine, radius)
        q_size = 3*len(load.plate_ids)
        spatial.append(dict(panel_count=panels, gauss_order=4, points=len(fine.states),
            rotation_difference_from_previous_layout_rad=float(np.linalg.norm(old_vector[:q_size]-new_vector[:q_size])),
            rotation_relative_difference_from_previous_layout=float(np.linalg.norm(old_vector[:q_size]-new_vector[:q_size])/np.linalg.norm(new_vector[:q_size])),
            transverse_displacement_difference_from_previous_layout_m=float(radius*np.linalg.norm(old_vector[q_size:]-new_vector[q_size:])),
            transverse_displacement_relative_difference_from_previous_layout=float(np.linalg.norm(old_vector[q_size:]-new_vector[q_size:])/np.linalg.norm(new_vector[q_size:])),
            full_coordinates_relative_difference_from_previous_layout=float(np.linalg.norm(old_vector-new_vector)/np.linalg.norm(new_vector))))
        previous = fine
    half = _prepare(surface, traces, load)
    for _ in range(2*STEP_COUNT):
        half = advance_bending_contact_dynamics(surface, half, load, DT_S/2.).state
    sensitivity = float(np.linalg.norm(_coordinates(current, radius)-_coordinates(half, radius))
                        /np.linalg.norm(_coordinates(current, radius)))
    continuum = _continuum_control(surface, traces, load)
    archived = _archive_cut_control(output, traces[0])
    mixed = sum(branch["locked"] > 0 and branch["sliding"] > 0
                for row in rows for branch in row["branches"].values())
    sliding = sum(branch["sliding"] for row in rows for branch in row["branches"].values())
    assert sliding > 0
    report = dict(format="geometric-bending-contact-validation-1", status="complete",
        elapsed_seconds=time.perf_counter()-started,
        runtime=dict(python=platform.python_version(), platform=platform.platform(),
            packages={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "PyYAML", "pytest")}),
        source_model="frozen eight-octant/two-hemisphere synthetic source and explicit two-bank cantilevers; no speed target",
        inputs=dict(basal_source=basal_load_to_dict(surface, load), step_duration_s=DT_S,
            step_count=STEP_COUNT, deformation_limits=LIMITS, initial_preparation=preparation),
        load_steps=rows, final_state=bending_contact_dynamics_to_dict(surface, current, load),
        point_count=len(current.states), mixed_branch_trace_steps=mixed, sliding_point_steps=sliding,
        maximum_resultant_torque_relative_residual=max(row["torque_signal_relative_residual"] for row in rows),
        maximum_incremental_work_relative_residual=max(row["incremental_energy_balance_relative_error"] for row in rows),
        maximum_generalized_force_relative_residual=max(row["generalized_force_relative_residual"] for row in rows),
        maximum_transverse_force_relative_residual=max(row["transverse_force_relative_residual"] for row in rows),
        maximum_physical_contact_virtual_work_relative_error=max(row["physical_contact_virtual_work_relative_error"] for row in rows),
        maximum_tip_displacement_m=max(abs(bank["tip_displacement_m"]) for row in rows for bank in row["declared_bank_diagnostics"]),
        maximum_bank_slope=max(bank["maximum_slope"] for row in rows for bank in row["declared_bank_diagnostics"]),
        maximum_bank_fibre_strain=max(bank["maximum_fibre_strain"] for row in rows for bank in row["declared_bank_diagnostics"]),
        final_bending_energy_j=rows[-1]["bending_energy_j"],
        spatial_sensitivity=dict(layouts=spatial,
            interpretation="Separate fresh histories on finer fixed physical grids; nonlinear spatial sensitivity, not interpolation of evolved histories or a convergence claim"),
        time_step_sensitivity=dict(comparison_step_duration_s=DT_S/2.,
            final_full_coordinates_relative_difference=sensitivity,
            interpretation="Backward-Euler timestep sensitivity, not an assertion of convergence"),
        independent_continuum_elastic_control=continuum, nonempty_archive_subdivision=archived,
        source_surface_preserved=True, main_passive_archive_preserved=True,
        transferred_material_during_mechanical_steps_km2=0., new_slab_cohorts_during_mechanical_steps=0,
        restart_exact=True, checkpoint_bytes_identical=True,
        checkpoint_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.glob("*.json"))},
        cloud_source_bundle_sha256={name: hashlib.sha256((ROOT/"handoff"/name).read_bytes()).hexdigest()
            for name in ("cloud_sources.zip", "cloud_sources_manifest.json")},
        code_sha256=hashes, code_unchanged=hashes == _hashes(),
        scope="Declared radial uniform-trace tip DOFs and locally planar elastic cantilevers; fixed physical dipping arcs and quadrature, exact continuous normal-gap envelope, Euler and radial virtual-work balance; material subdivision preserves mechanics and passive archive. No finite fault advection, along-strike bending, thermal feedback, material acceptance, slab pull or automatically selected elastic thickness/clamp distance")
    assert report["code_unchanged"]
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    result = execute(parser.parse_args().out)
    print(json.dumps({name: result[name] for name in (
        "status", "elapsed_seconds", "point_count", "mixed_branch_trace_steps", "sliding_point_steps",
        "maximum_resultant_torque_relative_residual", "maximum_incremental_work_relative_residual",
        "maximum_physical_contact_virtual_work_relative_error", "final_bending_energy_j",
        "restart_exact", "checkpoint_bytes_identical", "spatial_sensitivity", "time_step_sensitivity")}, indent=2))
