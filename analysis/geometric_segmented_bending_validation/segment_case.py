"""Changing-source bending histories and independent physical frame controls.

The prescribed load and full thermal lid vary explicitly between source
intervals. This is an externally prescribed source history, not a cooling
model. Original preparation and constitutive history are retained; work is
integrated over source intervals rather than reassigned to the newest load.
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

from analysis.geometric_bending_contact_validation.bending_case import (
    DT_S, LIMITS, _banks, _coordinates, _prepare,
)
from analysis.geometric_contact_dynamics_validation.dynamics_case import _inputs, SOURCE_PROVENANCE
from analysis.geometric_distributed_contact_validation.trace_case import _archive_fixture, _physical_traces
from analysis.geometric_initiation_validation.admission_case import fixture, snapshot
from tectonics.basal_coupling import PrescribedBasalParameters
from tectonics.geometric_basal_loading import assemble_geometric_basal_load, basal_load_to_dict
from tectonics.geometric_bending_contact_dynamics import advance_bending_contact_dynamics
from tectonics.geometric_bending_reference import current_bending_reference
from tectonics.geometric_boundary import initialize_boundary, polarity_evidence
from tectonics.geometric_contacts import extract_contacts
from tectonics.geometric_polarity import PolarityEvidence
from tectonics.geometric_surface import project_to_mesh, totals
from tectonics.mesh import SphereMesh
from tectonics.geometric_segmented_bending_dynamics import (
    advance_segmented_bending_dynamics, continuous_segmented_gap_bounds,
    initialize_segmented_bending_dynamics, segmented_bending_dynamics_from_dict,
    segmented_bending_dynamics_to_dict,
)

STEP_COUNT = 4


def _hashes():
    files = sorted((ROOT/"tectonics").glob("*.py"))
    files += [ROOT/"analysis/geometric_initiation_validation/admission_case.py",
              ROOT/"analysis/geometric_contact_dynamics_validation/dynamics_case.py",
              ROOT/"analysis/geometric_distributed_contact_validation/trace_case.py",
              ROOT/"analysis/geometric_bending_contact_validation/bending_case.py",
              Path(__file__).resolve()]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def _source_intervals(mesh, surface, initial_parameters, initial_lid, *, count=STEP_COUNT):
    declarations = ((initial_parameters.seed, 1., 1., 1.), (47, .8, 1.6, .85),
                    (59, 1.2, .7, 1.25), (73, .65, 1.25, .7))[:count]
    result = []
    for index, (seed, amplitude, drag, lid_factor) in enumerate(declarations):
        parameters = replace(initial_parameters, seed=seed,
            convective_traction_pa=initial_parameters.convective_traction_pa*amplitude,
            basal_drag_pa_s_m=initial_parameters.basal_drag_pa_s_m*drag)
        lid = {identity: value*lid_factor*(1. if index == 0 else 1.+.07*math.sin(.6*ordinal))
               for ordinal, (identity, value) in enumerate(sorted(initial_lid.items()))}
        provenance = f"Declared synthetic changing mantle source interval {index+1}; independent full lid values, no cooling claim"
        load = assemble_geometric_basal_load(mesh, surface, parameters, lid, provenance)
        result.append((parameters, lid, provenance, load))
    return tuple(result)


def _arc_moments(trace, radius):
    start, end = np.asarray(trace.start), np.asarray(trace.end)
    angle = math.atan2(np.linalg.norm(np.cross(start, end)), float(start@end))
    horizontal = np.cross(start, end)/np.linalg.norm(np.cross(start, end))
    middle = (start+end)/np.linalg.norm(start+end)
    tangent = np.cross(horizontal, middle)
    beta, gamma = np.cross(middle, horizontal), np.cross(tangent, horizontal)
    first = 2.*radius**2*math.sin(angle/2.)*beta
    second = radius**3*(.5*(angle+math.sin(angle))*np.outer(beta, beta)
                       +.5*(angle-math.sin(angle))*np.outer(gamma, gamma))
    return radius*angle, first, second


def _elastic_matrix(surface, traces, banks, owners):
    """Exact continuous-arc contact and beam matrix; no solver rows are read."""
    radius = surface.radius_km*1000.
    rotational = 3*len(owners)
    dimension = rotational+2*len(traces)
    matrix, preload = np.zeros((dimension, dimension)), np.zeros(dimension)
    bank_by_id = {pair.trace_id: pair for pair in banks}
    for index, trace in enumerate(traces):
        length, first, second = _arc_moments(trace, radius)
        incidence = np.zeros((dimension, 3))
        for sign, owner in ((-1., trace.subducting_plate), (1., trace.overriding_plate)):
            offset = 3*owners.index(owner)
            incidence[offset:offset+3] = sign*np.eye(3)
        vertical = np.zeros(dimension)
        vertical[rotational+2*index:rotational+2*index+2] = (-radius, radius)
        normal = 1./(trace.columns.sub_length_m/trace.columns.sub_normal_modulus_pa
            +trace.columns.over_length_m/trace.columns.over_normal_modulus_pa
            +1./trace.law.normal_stiffness_pa_m)
        shear = 1./(trace.columns.sub_length_m/trace.columns.sub_shear_modulus_pa
            +trace.columns.over_length_m/trace.columns.over_shear_modulus_pa
            +1./trace.law.tangential_stiffness_pa_m)
        sine, cosine = math.sin(trace.dip_radians), math.cos(trace.dip_radians)
        width, first_row = trace.contact_height_m/sine, incidence@first
        matrix += width*(normal*sine*sine+shear*cosine*cosine)*(incidence@second@incidence.T)
        mixed = width*(normal-shear)*sine*cosine*np.outer(first_row, vertical)
        matrix += mixed+mixed.T
        matrix += width*length*(normal*cosine*cosine+shear*sine*sine)*np.outer(vertical, vertical)
        preload += width*(normal*trace.initial_gap_m*sine+shear*trace.initial_jump_m*cosine)*first_row
        preload += width*length*(normal*trace.initial_gap_m*cosine-shear*trace.initial_jump_m*sine)*vertical
        pair = bank_by_id[trace.trace_id]
        for side, bank in enumerate((pair.sub, pair.over)):
            stiffness = bank.young_modulus_pa*bank.elastic_thickness_m**3*length/(4.*(1.-bank.poisson_ratio**2)*bank.width_m**3)
            position = rotational+2*index+side
            matrix[position, position] += stiffness*radius**2
    return matrix, preload


def _rodrigues(vector):
    vector = np.asarray(vector, dtype=float)
    angle = np.linalg.norm(vector)
    if angle == 0.: return np.eye(3)
    x, y, z = vector/angle
    cross = np.array(((0., -z, y), (z, 0., -x), (-y, x, 0.)))
    return np.eye(3)+math.sin(angle)*cross+(1.-math.cos(angle))*(cross@cross)


def _physical_quadrature(mesh, surface, parameters, lid, rotation):
    """Reintegrate rotated footprints/mesh and an explicitly rotated potential.

    The potential tensor is a physical source and changes Cartesian components
    as R M R.T. Reusing the old seed in rotated world axes would change it.
    Conditional vertex transmission follows the declared basal quadrature.
    """
    matrix = np.asarray(rotation)
    moved_mesh = SphereMesh(mesh.vertices@matrix.T, mesh.faces.copy(), mesh.centroids@matrix.T,
        mesh.areas_unit_sphere.copy(), mesh.neighbors, mesh.shared_edges)
    view = tuple(sorted(project_to_mesh(moved_mesh, surface),
        key=lambda fragment: (fragment.parcel.cell, fragment.parent_fragment_id, fragment.fragment_id)))
    rng = np.random.default_rng(parameters.seed)
    potential = rng.normal(size=(3, 3))
    potential = .5*(potential+potential.T)
    potential -= np.eye(3)*np.trace(potential)/3.
    normalization = 2.*np.linalg.norm(potential, 2)
    potential = matrix@potential@matrix.T
    vertex_source = 2.*(moved_mesh.vertices@potential.T)
    vertex_source -= moved_mesh.vertices*np.sum(vertex_source*moved_mesh.vertices, axis=1)[:, None]
    vertex_source *= parameters.convective_traction_pa/normalization
    capacities = moved_mesh.physical_cell_areas_km2(surface.radius_km)
    coupling = -np.expm1(-np.asarray([lid[fragment.parent_fragment_id] for fragment in view])/parameters.traction_coupling_depth_km)
    means = np.zeros(moved_mesh.cell_count)
    for fragment, fraction in zip(view, coupling):
        means[fragment.parcel.cell] += fragment.parcel.area_km2*fraction/capacities[fragment.parcel.cell]
    vertex_area, vertex_fraction = np.zeros(moved_mesh.vertex_count), np.zeros(moved_mesh.vertex_count)
    for cell, vertices in enumerate(moved_mesh.faces):
        for vertex in vertices:
            vertex_area[vertex] += moved_mesh.areas_unit_sphere[cell]
            vertex_fraction[vertex] += moved_mesh.areas_unit_sphere[cell]*means[cell]
    vertex_fraction /= vertex_area
    owners = tuple(sorted({fragment.parcel.plate for fragment in surface.fragments}))
    drag, torque = np.zeros((3*len(owners), 3*len(owners))), np.zeros(3*len(owners))
    radius = surface.radius_km*1000.
    for fragment, fraction in zip(view, coupling):
        cell = fragment.parcel.cell
        vertices = moved_mesh.faces[cell]
        weights = vertex_fraction[vertices]+moved_mesh.areas_unit_sphere[cell]/vertex_area[vertices]*(fraction-means[cell])
        traction = np.mean(weights[:, None]*vertex_source[vertices], axis=0)
        radial = moved_mesh.centroids[cell]
        traction -= radial*(radial@traction)
        area = fragment.parcel.area_km2*1e6
        offset = 3*owners.index(fragment.parcel.plate)
        torque[offset:offset+3] += radius*area*np.cross(radial, traction)
        drag[offset:offset+3, offset:offset+3] += parameters.basal_drag_pa_s_m*radius**2*area*(np.eye(3)-np.outer(radial, radial))
    return drag, torque, len(view)


def _segmented_coordinates(state, radius):
    return np.r_[np.asarray(state.rotation_path_material_radians).ravel(),
        np.asarray([row[1:] for row in state.transverse_displacement_m]).ravel()/radius]


def _row(surface, previous, step):
    branches = {trace.trace_id: dict(locked=0, sliding=0)
                for trace in step.state.initial_mechanics.fault_traces.traces}
    owners = {owner: index for index, owner in enumerate(step.state.plate_ids)}
    traces = {trace.trace_id: trace for trace in step.state.initial_mechanics.fault_traces.traces}
    old_tips = {identity: (sub, over) for identity, sub, over in previous.transverse_displacement_m}
    increments = {identity: np.asarray((sub, over))-old_tips[identity]
                  for identity, sub, over in step.state.transverse_displacement_m}
    rotation = np.asarray(step.rotation_increment_radians)
    radius = surface.radius_km*1000.
    physical_work, points = [], []
    for point in step.points:
        trace, radial = traces[point.trace_id], np.asarray(point.position)
        branch = "sliding" if point.step.slip_increment_m != 0. else "locked"
        branches[point.trace_id][branch] += 1
        for side, owner, force in ((0, trace.subducting_plate, point.force_on_sub_n),
                                   (1, trace.overriding_plate, point.force_on_over_n)):
            displacement = radius*np.cross(rotation[owners[owner]], radial)+increments[point.trace_id][side]*radial
            physical_work.append(float(np.asarray(force)@displacement))
        points.append(dict(point_id=point.point_id, trace_id=point.trace_id,
            branch=branch, slip_increment_m=point.step.slip_increment_m,
            plastic_slip_m=point.step.state.plastic_slip_m,
            cumulative_slip_m=point.step.state.cumulative_slip_m,
            gap_free_m=point.step.state.gap_free_m, jump_free_m=point.step.state.jump_free_m))
    work = math.fsum(physical_work)
    relative = abs(work+step.contact_work_j)/max(abs(work), abs(step.contact_work_j), np.finfo(float).tiny)
    assert relative < 2e-9
    metrics = {name: getattr(step, name) for name in (
        "source_work_j", "basal_dissipation_j", "contact_work_j", "stored_energy_increment_j",
        "friction_work_j", "viscous_work_j", "numerical_dissipation_j", "bending_energy_j",
        "bending_stored_energy_increment_j", "bending_numerical_dissipation_j", "bending_work_j",
        "generalized_force_relative_residual", "torque_signal_relative_residual",
        "transverse_force_relative_residual", "incremental_energy_balance_relative_error", "iterations")}
    assert metrics["torque_signal_relative_residual"] < 2e-9
    assert metrics["generalized_force_relative_residual"] < 2e-9
    assert metrics["incremental_energy_balance_relative_error"] < 2e-9
    return dict(sequence=step.state.segment_count, step_count=step.state.step_count,
        points=points, branches=branches, **metrics,
        current_frame_rotation_increment_rad=step.rotation_increment_radians,
        cumulative_material_chart_rotation_path_rad=step.state.rotation_path_material_radians,
        transverse_displacement_m=step.state.transverse_displacement_m,
        physical_virtual_work_relative_error=relative,
        continuous_gap_bounds_m=continuous_segmented_gap_bounds(step.state))


def _elastic_control(surface, traces, intervals):
    declared = tuple(replace(trace,
        contact_height_m=trace.contact_height_m*(1.+.2*index),
        initial_gap_m=trace.initial_gap_m/(1.+.2*index),
        law=replace(trace.law, cohesion_pa=1e9, friction_dry=0., friction_wet=0.),
        provenance=trace.provenance+"; declared locked elastic source-segment oracle")
        for index, trace in enumerate(traces))
    banks = _banks(declared)
    seed = _prepare(surface, declared, intervals[0][3], order=8, banks=banks)
    state = initialize_segmented_bending_dynamics(surface, seed, intervals[0][3])
    matrix, preload = _elastic_matrix(surface, declared, banks, seed.plate_ids)
    radius, rotational = surface.radius_km*1000., 3*len(seed.plate_ids)
    expected = _coordinates(seed, radius)
    rows = []
    for _, _, _, load in intervals:
        drag = np.zeros_like(matrix)
        drag[:rotational, :rotational] = np.asarray(load.drag_nm_s)/DT_S
        torque = np.pad(np.asarray(load.torque_nm), (0, len(expected)-rotational))
        expected = np.linalg.solve(matrix+drag, torque-preload+drag@expected)
        step = advance_segmented_bending_dynamics(surface, state, load, DT_S,
            provenance="Independent locked continuum control with a freshly declared source interval")
        assert all(point.step.slip_increment_m == 0. for point in step.points)
        actual = _segmented_coordinates(step.state, radius)
        error = float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
        assert error < 2e-9
        rows.append(dict(sequence=step.state.segment_count, all_coordinates_relative_error=error,
            exact_rotation_path_rad=expected[:rotational].reshape((-1, 3)).tolist(),
            exact_transverse_displacement_m=(expected[rotational:]*radius).reshape((-1, 2)).tolist(),
            source_work_j=step.source_work_j))
        state = step.state
    return dict(point_count=len(state.states), source_intervals=len(intervals), cases=rows,
        maximum_all_coordinates_relative_error=max(row["all_coordinates_relative_error"] for row in rows),
        derivation="Independent exact great-circle first and second moments and k=3EI*length/L^3; (D_k/dt+K_elastic)x_k=T_k-preload+(D_k/dt)x_(k-1). No solver rows or tangent are read")


def _passive_control(output, mesh, surface, inventory, seed, initial_load, parameters, lid):
    from tectonics.geometric_rigid_mantle_loading import (
        PrescribedRigidMantleSource, assemble_physical_rigid_mantle_load,
    )
    baseline = initialize_segmented_bending_dynamics(surface, seed, initial_load)
    represented = initialize_segmented_bending_dynamics(surface, seed, initial_load)
    rows = []
    source_inputs = {initial_load.load_sha256: dict(mesh=mesh, parameters=parameters,
        thermal_lid_by_fragment=lid, provenance=SOURCE_PROVENANCE)}
    chart_vectors = ((.23, -.31, .17), (-.16, .28, .32))
    angular_velocities = ((1.1e-14, -1.7e-14, .8e-14), (-.9e-14, 1.3e-14, 1.8e-14))
    for index, (chart_vector, omega) in enumerate(zip(chart_vectors, angular_velocities)):
        rotation = _rodrigues(chart_vector)
        reference = current_bending_reference(surface, seed, rotation)
        # Independent anisotropic quadratic potential audit on co-transformed
        # geometry and mesh. This is not stored as an ordinary seed-only load.
        drag, torque, count = _physical_quadrature(mesh, reference.surface, parameters, lid, rotation)
        original = assemble_geometric_basal_load(mesh, surface, parameters, lid, SOURCE_PROVENANCE)
        transform = np.kron(np.eye(len(seed.plate_ids)), rotation)
        drag_error = float(np.linalg.norm(drag-transform@np.asarray(original.drag_nm_s)@transform.T)/np.linalg.norm(drag))
        torque_error = float(np.linalg.norm(torque-transform@np.asarray(original.torque_nm))/np.linalg.norm(torque))
        assert max(drag_error, torque_error) < 2e-11
        # The rigid-mantle source has explicit physical Cartesian components;
        # a transformed source is independently rebuilt in the current chart.
        beta = 1e15*(1.+.3*index)
        source = PrescribedRigidMantleSource(omega, beta, "Declared rigid-mantle source for a zero-duration coordinate covariance control")
        native_load = assemble_physical_rigid_mantle_load(mesh, surface, source)
        moved_mesh = SphereMesh(mesh.vertices@rotation.T, mesh.faces.copy(), mesh.centroids@rotation.T,
            mesh.areas_unit_sphere.copy(), mesh.neighbors, mesh.shared_edges)
        moved_source = replace(source, angular_velocity_rad_s=tuple(map(float, rotation@omega)))
        current_load = assemble_physical_rigid_mantle_load(moved_mesh, reference.surface, moved_source)
        source_inputs[current_load.load_sha256] = dict(mesh=moved_mesh, source=moved_source)
        native_step = advance_segmented_bending_dynamics(surface, baseline, native_load, DT_S,
            provenance="Source interval in original Cartesian representation")
        moved_step = advance_segmented_bending_dynamics(surface, represented, current_load, DT_S,
            reference_rotation=rotation, provenance="Same physical source interval in a co-transformed Cartesian representation; no material transport")
        baseline, represented = native_step.state, moved_step.state
        native, moved = _segmented_coordinates(baseline, surface.radius_km*1000.), _segmented_coordinates(represented, surface.radius_km*1000.)
        coordinate_error = float(np.linalg.norm(native-moved)/np.linalg.norm(native))
        work_error = abs(native_step.source_work_j-moved_step.source_work_j)/max(abs(native_step.source_work_j), abs(moved_step.source_work_j), np.finfo(float).tiny)
        assert max(coordinate_error, work_error) < 2e-10
        history_errors = []
        for (_, local), (_, moved_local) in zip(baseline.states, represented.states):
            a = np.asarray((local.traction_normal_pa, local.traction_shear_pa))
            b = np.asarray((moved_local.traction_normal_pa, moved_local.traction_shear_pa))
            history_errors.append(float(np.linalg.norm(a-b)/max(np.linalg.norm(a), np.finfo(float).tiny)))
        assert max(history_errors) < 2e-10
        rows.append(dict(sequence=index+1, rotation=rotation.tolist(), projected_component_count=count,
            independently_rotated_quadratic_source_torque_relative_error=torque_error,
            independently_rotated_quadratic_source_drag_relative_error=drag_error,
            segmented_material_coordinates_relative_error=coordinate_error,
            segmented_source_work_relative_error=work_error,
            point_traction_history_maximum_relative_error=max(history_errors),
            radial_tip_memory_preserved=True))
    _save(output/"passive_final.json", surface, inventory, represented, "Co-transformed Cartesian physical source and mesh")
    origin, restored_inventory, restored, _ = _load(output/"passive_final.json", source_inputs=source_inputs)
    assert restored == represented and restored_inventory == inventory and origin == surface
    return dict(cases=rows, trusted_full_source_restart_exact=True,
        interpretation="Zero-duration changes of Cartesian representation. Footprints, mesh and physical source components transform together; original point measure and bending stiffness are carried exactly. These controls do not claim physical advection")


def _save(path, origin, inventory, state, provenance):
    from tectonics.geometric_segmented_bending_io import save_segmented_bending_checkpoint
    save_segmented_bending_checkpoint(path, origin, inventory, state, provenance={"experiment": provenance})


def _load(path, *, source_inputs=None):
    from tectonics.geometric_segmented_bending_io import load_segmented_bending_checkpoint
    origin, inventory, state, provenance = load_segmented_bending_checkpoint(path, source_inputs=source_inputs)
    return origin, inventory, state, provenance["experiment"]


def _archive_control(output, template):
    mesh, surface, inventory = _archive_fixture()
    original = snapshot(surface), snapshot(inventory)
    assert len(inventory.cohorts) == 2
    parameters = PrescribedBasalParameters(17, 8e4, 1e14, 2.)
    lid = {fragment.fragment_id: .2+1.3*index for index, fragment in enumerate(surface.fragments)}
    intervals = _source_intervals(mesh, surface, parameters, lid, count=3)
    traces = []
    for index, contact in enumerate(extract_contacts(surface.fragments, surface.radius_km)):
        horizontal = np.asarray(contact.normal_a_to_b)*(1. if contact.plate_b == 1 else -1.)
        start, end = contact.start, contact.end
        if np.cross(start, end)@horizontal < 0.: start, end = end, start
        traces.append(replace(template, trace_id=f"archived-source-trace:{index}", start=start, end=end,
            subducting_plate=0, overriding_plate=1, initial_gap_m=-1.,
            law=replace(template.law, cohesion_pa=1e9, friction_dry=0., friction_wet=0.),
            provenance="Declared physical archival reference for independently changing source intervals"))
    history = tuple(PolarityEvidence(**row) for row in polarity_evidence(inventory))
    seed = _prepare(surface, traces, intervals[0][3], initial_dt=1e6, history=history)
    state = initialize_segmented_bending_dynamics(surface, seed, intervals[0][3])
    for index, (_, _, _, load) in enumerate(intervals):
        state = advance_segmented_bending_dynamics(surface, state, load, 1e8,
            provenance="Accepted changing-source interval on unchanged original archival material support").state
        if index == 0: _save(output/"archive_middle.json", surface, inventory, state, "Two-cohort archive and full segmented source history")
    first_parameters, first_lid, first_provenance, first_load = intervals[0]
    source_inputs = {first_load.load_sha256: dict(mesh=mesh, parameters=first_parameters,
        thermal_lid_by_fragment=first_lid, provenance=first_provenance)}
    origin, restored_inventory, resumed, user = _load(output/"archive_middle.json", source_inputs=source_inputs)
    for _, _, _, load in intervals[1:]:
        resumed = advance_segmented_bending_dynamics(origin, resumed, load, 1e8,
            provenance="Accepted changing-source interval on unchanged original archival material support").state
    assert resumed == state and snapshot(restored_inventory) == original[1]
    _save(output/"archive_direct_final.json", surface, inventory, state, user)
    _save(output/"archive_resumed_final.json", origin, restored_inventory, resumed, user)
    identical = (output/"archive_direct_final.json").read_bytes() == (output/"archive_resumed_final.json").read_bytes()
    assert identical and (snapshot(surface), snapshot(inventory)) == original
    assert state.initial_mechanics == seed and state.initial_mechanics.polarity_history == seed.polarity_history
    return dict(cohort_count=len(inventory.cohorts), retained_physical_polarity_interval_count=len(seed.polarity_history),
        source_segments=state.segment_count, original_acceptance_events_and_cohort_payloads_unchanged=True,
        original_surface_and_archive_preserved=True, restart_exact=True, checkpoint_bytes_identical=identical,
        material_total_residual={field: totals(surface.fragments)[field]-totals(origin.fragments)[field]
                                 for field in totals(surface.fragments)})


def _physical_common_control(output, template):
    from tectonics.geometric_bending_motion import (
        advance_bending_motion, initialize_bending_motion,
    )
    from tectonics.geometric_rigid_mantle_loading import (
        PrescribedRigidMantleSource, assemble_physical_rigid_mantle_load,
    )
    from tectonics.thermal import SECONDS_PER_MYR
    mesh, origin, inventory = _archive_fixture()
    assert len(inventory.cohorts) == 2
    historical_payloads = tuple((cohort.event_id, cohort.payload_sha256, cohort.loss)
                                for cohort in inventory.cohorts)
    traces = []
    for index, contact in enumerate(extract_contacts(origin.fragments, origin.radius_km)):
        horizontal = np.asarray(contact.normal_a_to_b)*(1. if contact.plate_b == 1 else -1.)
        start, end = contact.start, contact.end
        if np.cross(start, end)@horizontal < 0.: start, end = end, start
        traces.append(replace(template, trace_id=f"physical-common-arc:{index}", start=start, end=end,
            subducting_plate=0, overriding_plate=1, initial_gap_m=0., initial_jump_m=0.,
            provenance="Declared zero-strain physical paired interface in a pre-existing two-cohort archive; manufactured common-flow validation"))
    banks = tuple(replace(pair, sub=replace(pair.sub, width_m=10000.),
                           over=replace(pair.over, width_m=15000.)) for pair in _banks(traces))
    history = tuple(PolarityEvidence(**row) for row in polarity_evidence(inventory))
    # Every source is declared before computing motion. These are independent
    # manufactured physical inputs, not velocities fitted to a solved result.
    omegas = ((1.1e-12, -.7e-12, 1.5e-12), (-.9e-12, .8e-12, 1.2e-12),
              (1.4e-12, .6e-12, -.8e-12))
    betas = (1e14, 1.4e14, .8e14)
    sources = tuple(PrescribedRigidMantleSource(omega, beta,
        f"Explicit manufactured physical rigid-mantle source {index+1}; fully transmitted, no plate-speed fit")
        for index, (omega, beta) in enumerate(zip(omegas, betas)))
    initial_load = assemble_physical_rigid_mantle_load(mesh, origin, sources[0])
    seed = _prepare(origin, traces, initial_load, banks=banks, initial_dt=1e6, history=history)
    assert all(local.traction_normal_pa == 0. and local.traction_shear_pa == 0.
               for _, local in seed.states)
    state = initialize_bending_motion(origin, inventory, seed, initial_load)
    dt, expected = 1e9, np.eye(3)
    original = snapshot(origin), snapshot(inventory), snapshot(seed)
    rows, loads = [], []
    source_inputs = {initial_load.load_sha256: dict(mesh=mesh, source=sources[0])}
    for index, source in enumerate(sources):
        before = state.current_surface
        load = assemble_physical_rigid_mantle_load(mesh, before, source)
        loads.append(load)
        source_inputs[load.load_sha256] = dict(mesh=mesh, source=source)
        result = advance_bending_motion(mesh, origin, state, load, dt,
            provenance=f"Accepted force-generated common material step {index+1}; source and geometry held at beginning of interval")
        expected_increment = np.asarray(source.angular_velocity_rad_s)*dt
        increments = np.asarray(result.mechanical_step.rotation_increment_radians)
        assert np.array_equal(increments[0], increments[1])
        assert np.array_equal(increments[0], expected_increment)
        expected = _rodrigues(expected_increment)@expected
        row = _row(origin, state.mechanics, result.mechanical_step)
        assert not result.geometric_transaction.births and not result.geometric_transaction.losses
        assert totals(before.fragments) == totals(result.state.current_surface.fragments)
        assert tuple((cohort.event_id, cohort.payload_sha256, cohort.loss)
                     for cohort in result.state.current_inventory.cohorts) == historical_payloads
        assert all(local.traction_normal_pa == 0. and local.traction_shear_pa == 0.
                   and local.plastic_slip_m == 0. and local.cumulative_slip_m == 0.
                   for _, local in result.state.mechanics.states)
        row.update(actual_surface_time_myr=result.state.current_surface.time_myr,
            declared_mantle_angular_velocity_rad_s=source.angular_velocity_rad_s,
            exact_manufactured_rotation_increment_rad=tuple(expected_increment),
            actual_geometric_transaction_births=0, actual_geometric_transaction_losses=0,
            extensive_residual={field: totals(result.state.current_surface.fragments)[field]-totals(origin.fragments)[field]
                               for field in totals(origin.fragments)})
        rows.append(row)
        state = result.state
        if index == 0:
            _save_motion(output/"physical_middle.json", mesh, origin, state,
                "Accepted physical common motion with source, mechanics, current material and two-cohort archive")
            middle_source_inputs = dict(source_inputs)
    _save_motion(output/"physical_direct_final.json", mesh, origin, state,
        "Accepted physical common motion with source, mechanics, current material and two-cohort archive")
    loaded_origin, resumed, user = _load_motion(output/"physical_middle.json", mesh,
                                               source_inputs=middle_source_inputs)
    for index, source in enumerate(sources[1:], start=1):
        rebuilt = assemble_physical_rigid_mantle_load(mesh, resumed.current_surface, source)
        assert rebuilt == loads[index]
        resumed = advance_bending_motion(mesh, loaded_origin, resumed, rebuilt, dt,
            provenance=f"Accepted force-generated common material step {index+1}; source and geometry held at beginning of interval").state
    assert resumed == state and loaded_origin == origin
    _save_motion(output/"physical_resumed_final.json", mesh, loaded_origin, resumed, user)
    identical = (output/"physical_direct_final.json").read_bytes() == (output/"physical_resumed_final.json").read_bytes()
    assert identical
    matrix_error = float(np.linalg.norm(np.asarray(state.common_rotation)-expected))
    assert matrix_error < 2e-14
    footprint_error = max(float(np.max(np.linalg.norm(np.asarray(new.polygon)-np.asarray(old.polygon)@expected.T, axis=1)))
                          for old, new in zip(origin.fragments, state.current_surface.fragments))
    assert footprint_error < 2e-14
    actual_elapsed = state.current_surface.time_myr-origin.time_myr
    assert math.isclose(actual_elapsed, len(sources)*dt/SECONDS_PER_MYR, rel_tol=2e-12)
    for old, new in zip(origin.fragments, state.current_surface.fragments):
        assert replace(new.parcel, age_myr=old.parcel.age_myr) == old.parcel
        assert math.isclose(new.parcel.age_myr-old.parcel.age_myr, actual_elapsed, rel_tol=2e-9)
    assert (snapshot(origin), snapshot(inventory), snapshot(seed)) == original
    # A separate precompressed fixture allows the real differential solve to
    # succeed before the physical material-pair gate refuses its commit.
    parameters = PrescribedBasalParameters(31, 1e6, 1e15, 50.)
    lid = {fragment.fragment_id: 20.+5.*index for index, fragment in enumerate(origin.fragments)}
    differential_load = assemble_geometric_basal_load(mesh, origin, parameters, lid,
        "Independent prescribed quadratic source for atomic differential material-motion rejection")
    closed = tuple(replace(trace, initial_gap_m=-20.) for trace in traces)
    closed_seed = _prepare(origin, closed, differential_load, banks=banks)
    rejected = initialize_bending_motion(origin, inventory, closed_seed, differential_load)
    snapshot_before = snapshot(rejected)
    try:
        advance_bending_motion(mesh, origin, rejected, differential_load, DT_S,
            provenance="Differential source must await spatial material admission and cannot partially commit geometry or source work")
    except ValueError as error:
        rejection = str(error)
    else:
        raise AssertionError("Differential force response must not be accepted as paired common physical motion")
    assert snapshot(rejected) == snapshot_before
    return dict(accepted_physical_steps=len(rows), cohort_count=len(state.current_inventory.cohorts),
        retained_polarity_interval_count=len(seed.polarity_history), source_model="Explicit fully transmitted rigid mantle traction with independent positive drag",
        steps=rows, chronological_rotation_matrix=expected.tolist(),
        independent_SO3_composition_matrix_error=matrix_error,
        maximum_independent_footprint_vertex_error=footprint_error,
        actual_material_elapsed_time_myr=actual_elapsed,
        actual_archive_transactions_added=len(state.current_inventory.transactions)-len(inventory.transactions),
        actual_surface_and_archive_are_matching_current_endpoint=True,
        original_loss_payloads_and_material_properties_preserved=True,
        material_total_residual={field: totals(state.current_surface.fragments)[field]-totals(origin.fragments)[field]
                                 for field in totals(origin.fragments)},
        source_work_sum_j=math.fsum(row["source_work_j"] for row in rows),
        restart_exact=True, checkpoint_bytes_identical=identical,
        differential_motion_rejected_atomically=rejection,
        differential_rejection_preserves_geometry_archive_mechanics_and_work=True,
        interpretation="Genuine beginning-stage force/source solve followed by complete actual material transport over the same dt; common paired material maps only. Quadrature and geometry are frozen within each interval, making this a first-order coupled step, not exact time integration or general differential underthrust")


def _save_motion(path, mesh, origin, state, provenance):
    from tectonics.geometric_bending_motion_io import save_bending_motion_checkpoint
    save_bending_motion_checkpoint(path, mesh, origin, state, provenance={"experiment": provenance})


def _load_motion(path, mesh, *, source_inputs=None):
    from tectonics.geometric_bending_motion_io import load_bending_motion_checkpoint
    origin, state, provenance = load_bending_motion_checkpoint(path, mesh=mesh)
    if source_inputs is not None:
        from tectonics.geometric_basal_sources import basal_source_from_dict, basal_source_to_dict
        loads = [(origin, state.mechanics.initial_load)]
        loads += [(receipt.reference_configuration.surface, receipt.current_load)
                  for receipt in state.mechanics.segments]
        assert set(source_inputs) == {load.load_sha256 for _, load in loads}
        for surface, load in loads:
            restored = basal_source_from_dict(json.loads(json.dumps(basal_source_to_dict(surface, load))), surface,
                **source_inputs[load.load_sha256])
            assert restored == load
    return origin, state, provenance["experiment"]


def execute(output):
    started, hashes = time.perf_counter(), _hashes()
    output = Path(output)
    if output.exists(): raise ValueError("Validation output must be new")
    output.mkdir(parents=True)
    mesh, surface = fixture()
    inventory = initialize_boundary(surface)
    original = snapshot(surface), snapshot(inventory)
    parameters, lid, experiments, _ = _inputs(surface)
    traces = _physical_traces(surface, experiments)
    initial_load = assemble_geometric_basal_load(mesh, surface, parameters, lid, SOURCE_PROVENANCE)
    initial = _prepare(surface, traces, initial_load)
    seed_step = advance_bending_contact_dynamics(surface, initial, initial_load, DT_S)
    seed = seed_step.state
    assert any(local.plastic_slip_m != 0. for _, local in seed.states)
    original_seed = snapshot(seed)
    intervals = _source_intervals(mesh, surface, parameters, lid)
    state = initialize_segmented_bending_dynamics(surface, seed, initial_load)
    assert state.states == seed.states and state.initial_mechanics == seed
    assert state.transverse_displacement_m == seed.transverse_displacement_m
    assert state.cumulative_source_work_j == seed.cumulative_source_work_j
    rows = []
    for index, (_, _, _, load) in enumerate(intervals):
        step = advance_segmented_bending_dynamics(surface, state, load, DT_S,
            provenance=f"Accepted independently declared changing-source interval {index+1}; original material support unchanged")
        rows.append(_row(surface, state, step))
        state = step.state
        if index == 1: _save(output/"middle.json", surface, inventory, state, "Accepted changing-source bending history")
    _save(output/"direct_final.json", surface, inventory, state, "Accepted changing-source bending history")
    source_inputs = {initial_load.load_sha256: dict(mesh=mesh, parameters=parameters,
        thermal_lid_by_fragment=lid, provenance=SOURCE_PROVENANCE)}
    for p, h, provenance, load in intervals[:2]:
        source_inputs[load.load_sha256] = dict(mesh=mesh, parameters=p, thermal_lid_by_fragment=h, provenance=provenance)
    origin, restored_inventory, resumed, user = _load(output/"middle.json", source_inputs=source_inputs)
    assert resumed.initial_mechanics == seed and resumed.initial_load == initial_load
    for _, _, _, load in intervals[2:]:
        resumed = advance_segmented_bending_dynamics(origin, resumed, load, DT_S,
            provenance=f"Accepted independently declared changing-source interval {resumed.segment_count+1}; original material support unchanged").state
    assert resumed == state and snapshot(restored_inventory) == original[1]
    assert tuple(receipt.current_load for receipt in resumed.segments) == tuple(row[3] for row in intervals)
    _save(output/"resumed_final.json", origin, restored_inventory, resumed, user)
    identical = (output/"direct_final.json").read_bytes() == (output/"resumed_final.json").read_bytes()
    assert identical and (snapshot(surface), snapshot(inventory)) == original and snapshot(seed) == original_seed
    exact_work = math.fsum((seed.cumulative_source_work_j, *(row["source_work_j"] for row in rows)))
    assert math.isclose(exact_work, state.cumulative_source_work_j, rel_tol=2e-14, abs_tol=0.)
    wrong_work = float(np.asarray(intervals[-1][3].torque_nm)@np.asarray(state.rotation_path_material_radians).ravel())
    work_difference = abs(wrong_work-exact_work)/max(abs(exact_work), abs(wrong_work), np.finfo(float).tiny)
    assert work_difference > 1e-3
    directions = [np.asarray(row[3].torque_nm)/np.linalg.norm(row[3].torque_nm) for row in intervals]
    assert any(abs(float(directions[0]@direction)) < .99 for direction in directions[1:])
    elastic = _elastic_control(surface, traces, intervals)
    passive = _passive_control(output, mesh, surface, inventory, seed, initial_load, parameters, lid)
    archive = _archive_control(output, traces[0])
    physical = _physical_common_control(output, traces[0])
    mixed = sum(branch["locked"] > 0 and branch["sliding"] > 0 for row in rows for branch in row["branches"].values())
    sliding = sum(branch["sliding"] for row in rows for branch in row["branches"].values())
    assert sliding > 0
    report = dict(format="geometric-segmented-bending-validation-1", status="complete",
        elapsed_seconds=time.perf_counter()-started,
        runtime=dict(python=platform.python_version(), platform=platform.platform(),
            packages={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "PyYAML", "pytest")}),
        inputs=dict(source_intervals=[dict(parameters=asdict(p), thermal_lid_by_fragment=h,
            source_provenance=provenance, basal_source=basal_load_to_dict(surface, load))
            for p, h, provenance, load in intervals], step_duration_s=DT_S, deformation_limits=LIMITS),
        retained_seed=dict(previously_accepted_steps=seed.step_count,
            previously_accumulated_source_work_j=seed.cumulative_source_work_j,
            preparation_work_j=seed.preparation_work_j,
            nonzero_plastic_history_point_count=sum(local.plastic_slip_m != 0. for _, local in seed.states),
            full_original_seed_unchanged=True, local_and_radial_histories_exact_at_initialization=True),
        accepted_source_steps=rows, point_count=len(state.states), source_segment_count=state.segment_count,
        mixed_branch_trace_steps=mixed, sliding_point_steps=sliding,
        maximum_resultant_torque_relative_residual=max(row["torque_signal_relative_residual"] for row in rows),
        maximum_generalized_force_relative_residual=max(row["generalized_force_relative_residual"] for row in rows),
        maximum_incremental_work_relative_residual=max(row["incremental_energy_balance_relative_error"] for row in rows),
        maximum_physical_virtual_work_relative_error=max(row["physical_virtual_work_relative_error"] for row in rows),
        source_work=dict(exact_seed_plus_segment_work_j=exact_work,
            incorrect_newest_source_times_total_rotation_j=wrong_work,
            relative_mismatch=work_difference, current_load_changes_are_nonproportional=True,
            history_work_is_never_reassigned=True),
        independent_locked_continuum_control=elastic, passive_cartesian_covariance_control=passive,
        nonempty_passive_archive=archive,
        accepted_physical_common_motion=physical,
        final_state=segmented_bending_dynamics_to_dict(surface, state),
        frozen_core_origin_surface_and_archive_preserved=True,
        underthrust_admitted_material_km2=0., new_slab_cohorts=0,
        accepted_physical_geometry_steps=physical["accepted_physical_steps"],
        restart_exact=True, checkpoint_bytes_identical=identical,
        checkpoint_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.glob("*.json"))},
        cloud_source_bundle_sha256={name: hashlib.sha256((ROOT/"handoff"/name).read_bytes()).hexdigest()
            for name in ("cloud_sources.zip", "cloud_sources_manifest.json")},
        code_sha256=hashes, code_unchanged=hashes == _hashes(),
        scope="Accepted increments with independently bound changing sources, preserved preparation/plastic/radial histories and exact source work segments; continuous evolving gap harmonics and zero-duration Cartesian covariance. A separate declared manufactured rigid-mantle experiment accepts actual complete common material transport and current two-cohort archive transactions; differential motion is atomically rejected. No cooling, general spatial rematching or differential material admission is inferred")
    assert report["code_unchanged"]
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    result = execute(parser.parse_args().out)
    print(json.dumps({name: result[name] for name in (
        "status", "elapsed_seconds", "point_count", "source_segment_count", "mixed_branch_trace_steps",
        "sliding_point_steps", "maximum_resultant_torque_relative_residual", "maximum_generalized_force_relative_residual",
        "maximum_incremental_work_relative_residual", "source_work", "accepted_physical_geometry_steps",
        "restart_exact", "checkpoint_bytes_identical")}, indent=2))
