"""Finite material transport proposals, with a separately frozen force predictor.

Only the mechanical solve on the original geometry is accepted. Its exact work
receipt is retained when SO(3) motion constructs an unaccepted endpoint. Large
prescribed rotations below test kinematics, not a finite mechanical solution.
The fixed mantle source is independently reassembled as an input audit; that
new load never replaces the source in an old mechanical history.
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
    DT_S, LIMITS, _banks, _prepare, _row,
)
from analysis.geometric_contact_dynamics_validation.dynamics_case import _inputs, SOURCE_PROVENANCE
from analysis.geometric_distributed_contact_validation.trace_case import _archive_fixture, _physical_traces
from analysis.geometric_initiation_validation.admission_case import fixture, snapshot
from tectonics.basal_coupling import PrescribedBasalParameters
from tectonics.fractional_surface import EXTENSIVE_FIELDS
from tectonics.geometric_basal_loading import assemble_geometric_basal_load, basal_load_to_dict
from tectonics.geometric_bending_contact_dynamics import validate_bending_contact_dynamics_state
from tectonics.geometric_bending_fault_transport import (
    advance_bending_fault_transport, bending_fault_transport_to_dict,
    initialize_bending_fault_transport,
)
from tectonics.geometric_bending_fault_transport_io import (
    load_bending_fault_transport_checkpoint, save_bending_fault_transport_checkpoint,
)
from tectonics.geometric_boundary import initialize_boundary, polarity_evidence
from tectonics.geometric_contacts import extract_contacts
from tectonics.geometric_fault_motion_support import (
    continuous_bank_pair_separation_m, validate_reference_bending_bank_support,
)
from tectonics.geometric_fault_traces import (
    TraceQuadrature, declare_fault_trace_set, validate_fault_trace_set,
)
from tectonics.geometric_fault_transport import (
    advance_fault_material_atlas, fault_material_atlas_from_dict,
    fault_material_atlas_to_dict, initialize_fault_material_atlas,
    material_atlas_surface, transported_fault_geometry,
)
from tectonics.geometric_polarity import PolarityEvidence
from tectonics.geometric_surface import audit_partition, totals

STEP_COUNT = 4
PROPOSAL_PROVENANCE = "Declared synthetic frozen-reference predictor followed by an unaccepted exact material transport over the same physical duration"


def _hashes():
    files = sorted((ROOT/"tectonics").glob("*.py"))
    files += [ROOT/"analysis/geometric_initiation_validation/admission_case.py",
              ROOT/"analysis/geometric_contact_dynamics_validation/dynamics_case.py",
              ROOT/"analysis/geometric_distributed_contact_validation/trace_case.py",
              ROOT/"analysis/geometric_bending_contact_validation/bending_case.py",
              Path(__file__).resolve()]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def _declared_banks(traces):
    return tuple(replace(pair,
        sub=replace(pair.sub, width_m=10000., provenance=pair.sub.provenance+"; explicitly declared 10 km material clamp"),
        over=replace(pair.over, width_m=15000., provenance=pair.over.provenance+"; explicitly declared 15 km material clamp"))
        for pair in _banks(traces))


def _rodrigues(vector):
    """Independent elementary finite-rotation oracle, without scipy Rotation."""
    vector = np.asarray(vector, dtype=float)
    angle = np.linalg.norm(vector)
    if angle == 0.: return np.eye(3)
    x, y, z = vector/angle
    cross = np.array(((0., -z, y), (z, 0., -x), (-y, x, 0.)))
    return np.eye(3)+math.sin(angle)*cross+(1.-math.cos(angle))*(cross@cross)


def _candidate_material_audit(origin, atlas):
    candidate = material_atlas_surface(origin, atlas)
    old = {fragment.fragment_id: fragment for fragment in origin.fragments}
    residual = {field: totals(candidate.fragments)[field]-totals(origin.fragments)[field]
                for field in EXTENSIVE_FIELDS}
    assert all(value == 0. for value in residual.values())
    moved = []
    for fragment in candidate.fragments:
        reference = old[fragment.fragment_id]
        assert fragment.parent_fragment_id == reference.parent_fragment_id
        assert replace(fragment.parcel, age_myr=reference.parcel.age_myr) == reference.parcel
        assert fragment.parcel.age_myr == reference.parcel.age_myr+atlas.elapsed_time_myr
        moved.append(float(np.max(np.linalg.norm(np.asarray(fragment.polygon)-reference.polygon, axis=1))))
    assert candidate.time_myr == atlas.time_myr
    return dict(surface_clock_myr=candidate.time_myr, elapsed_time_myr=atlas.elapsed_time_myr,
        material_age_advance_myr=atlas.elapsed_time_myr,
        extensive_residual=residual, all_other_parcel_properties_preserved=True,
        maximum_vertex_chord_displacement=max(moved), phase=candidate.phase)


def _covariant_reaction_work(surface, previous, step, matrix):
    """Dot-product work in a common rotated representation of one old solve."""
    radius = surface.radius_km*1000.
    owners = {owner: index for index, owner in enumerate(step.state.plate_ids)}
    delta = np.asarray(step.state.rotation_radians)-previous.rotation_radians
    before = {identity: (sub, over) for identity, sub, over in previous.transverse_displacement_m}
    tips = {identity: np.asarray((sub, over))-before[identity]
            for identity, sub, over in step.state.transverse_displacement_m}
    traces = {trace.trace_id: trace for trace in step.state.fault_traces.traces}
    original, transformed = [], []
    for point in step.points:
        trace, radial = traces[point.trace_id], np.asarray(point.position)
        for side, owner, force in ((0, trace.subducting_plate, point.force_on_sub_n),
                                   (1, trace.overriding_plate, point.force_on_over_n)):
            velocity = radius*np.cross(delta[owners[owner]], radial)+tips[point.trace_id][side]*radial
            moved = radius*np.cross(matrix@delta[owners[owner]], matrix@radial)+tips[point.trace_id][side]*(matrix@radial)
            original.append(float(np.asarray(force)@velocity))
            transformed.append(float((matrix@force)@moved))
    old, new = math.fsum(original), math.fsum(transformed)
    error = abs(old-new)/max(abs(old), abs(new), np.finfo(float).tiny)
    assert error < 2e-12
    return dict(reference_contact_reaction_work_j=old,
        rotated_representation_contact_reaction_work_j=new, relative_error=error,
        interpretation="Common rotation of an already solved reference step; invariant virtual work, not continuation with a fixed external mantle source")


def _kinematic_controls(surface, mechanics, load, mesh, parameters, lid, final_step, previous):
    atlas = initialize_fault_material_atlas(surface, mechanics.fault_traces,
        mechanics.bending_parameters, mechanics.polarity_history)
    reference = atlas
    vectors = ((.42, 0., 0.), (0., -.37, 0.), (0., 0., .31), (.18, .14, -.09))
    expected = np.eye(3)
    for index, vector in enumerate(vectors):
        expected = _rodrigues(vector)@expected
        atlas = advance_fault_material_atlas(surface, atlas,
            tuple((owner, vector) for owner in atlas.plate_ids), .025,
            provenance=f"Prescribed noncommuting common rigid kinematic control {index+1}; no force solve")
        if index == 1:
            restored = fault_material_atlas_from_dict(fault_material_atlas_to_dict(surface, atlas), surface)
            assert restored == atlas
    assert np.max(np.abs(np.asarray(atlas.owner_rotations[0][1])-expected)) < 2e-14
    wrong = _rodrigues(np.sum(vectors, axis=0))
    wrong_error = float(np.linalg.norm(expected-wrong))
    assert wrong_error > .03
    moved = transported_fault_geometry(surface, atlas)
    assert tuple(point.point_id for point in moved.points) == tuple(point.point_id for point in reference.reference_points)
    point_errors, clamp_errors, frame_errors = [], [], []
    for old, new in zip(reference.reference_points, moved.points):
        assert old.point_id == new.point_id and old.area_m2 == new.area_m2
        for name in ("position", "horizontal", "normal", "down_dip"):
            frame_errors.append(float(np.linalg.norm(np.asarray(getattr(new, name))-expected@getattr(old, name))))
        point_errors.append(frame_errors[-4])
    for old, new in zip(reference.reference_bank_clamps, moved.bank_points):
        point = next(point for point in reference.reference_points if point.point_id == old.point_id)
        radius = surface.radius_km*1000.
        radial, horizontal = np.asarray(point.position), np.asarray(point.horizontal)
        exact_sub = math.cos(old.sub_width_m/radius)*radial-math.sin(old.sub_width_m/radius)*horizontal
        exact_over = math.cos(old.over_width_m/radius)*radial+math.sin(old.over_width_m/radius)*horizontal
        assert np.linalg.norm(np.asarray(old.sub_clamp)-exact_sub) < 2e-14
        assert np.linalg.norm(np.asarray(old.over_clamp)-exact_over) < 2e-14
        for name in ("sub_clamp", "over_clamp"):
            clamp_errors.append(float(np.linalg.norm(np.asarray(getattr(new, name))-expected@getattr(old, name))))
        assert new.bank_separation_m == 0.
    assert max(frame_errors+clamp_errors) < 2e-14
    for old, new in zip(mechanics.fault_traces.traces, moved.traces):
        assert replace(new, start=old.start, end=old.end) == old
    for old, new in zip(mechanics.polarity_history, moved.history):
        assert replace(new, start=old.start, end=old.end) == old
    candidate = material_atlas_surface(surface, atlas)
    footprint_error = max(float(np.max(np.linalg.norm(np.asarray(new.polygon)-np.asarray(old.polygon)@expected.T, axis=1)))
                          for old, new in zip(surface.fragments, candidate.fragments))
    assert footprint_error < 2e-14
    audit_partition(candidate)
    new_set = declare_fault_trace_set(candidate, moved.traces, mechanics.fault_traces.quadrature)
    validate_fault_trace_set(candidate, new_set, moved.history)
    new_load = assemble_geometric_basal_load(mesh, candidate, parameters, lid, SOURCE_PROVENANCE)
    reference_torque = np.asarray(load.torque_nm).reshape((-1, 3))
    current_torque = np.asarray(new_load.torque_nm).reshape((-1, 3))
    co_rotated = reference_torque@expected.T
    source_difference = float(np.linalg.norm(current_torque-co_rotated)/np.linalg.norm(current_torque))
    assert source_difference > 1e-4
    source_audit = dict(reference_load_sha256=load.load_sha256, candidate_load_sha256=new_load.load_sha256,
        candidate_basal_source=basal_load_to_dict(candidate, new_load),
        reassembled_torque_difference_from_co_rotated_old_source_relative=source_difference,
        reference_source_work_j=mechanics.cumulative_source_work_j,
        archived_source_work_unchanged=True,
        interpretation="Fresh projection on the same fixed mesh and prescribed mantle source. This source differs from a co-moving load and is not attached to the archived reference mechanics")
    differential = advance_fault_material_atlas(surface, reference,
        tuple((owner, ((.2 if index == 0 else -.2), 0., 0.)) for index, owner in enumerate(reference.plate_ids)),
        .1, provenance="Prescribed opposite rotations about the shared hemisphere normal; boundary-set coincidence does not preserve material point pairs")
    different_geometry = transported_fault_geometry(surface, differential)
    different_surface = material_atlas_surface(surface, differential)
    # Owner hemispheres remain the same partition as a set, despite different
    # material labels along the equator and a conservatively unaccepted phase.
    audit_partition(replace(different_surface, phase="partition"))
    continuous = []
    maps = dict(differential.owner_rotations)
    for trace in mechanics.fault_traces.traces:
        low, high = continuous_bank_pair_separation_m(trace,
            maps[trace.subducting_plate], maps[trace.overriding_plate], surface.radius_km*1000.)
        assert math.isclose(low, 2.*surface.radius_km*1000.*math.sin(.2), rel_tol=2e-13)
        assert math.isclose(high, low, rel_tol=2e-13)
        continuous.append(dict(trace_id=trace.trace_id, minimum_separation_m=low, maximum_separation_m=high))
    assert all(not item.bank_maps_identical and item.maximum_bank_separation_m > 1000.
               for item in different_geometry.trace_mismatches)
    return dict(common_noncommuting=dict(rotation_vectors=vectors,
        total_prescribed_duration_myr=atlas.elapsed_time_myr,
        chronological_composition_matrix=expected.tolist(),
        incorrect_rotation_vector_sum_matrix_difference=wrong_error,
        maximum_point_frame_error=max(frame_errors), maximum_bank_clamp_error=max(clamp_errors),
        maximum_footprint_vertex_error=footprint_error,
        analytic_geodesic_reference_bank_clamps_verified=True,
        material_audit=_candidate_material_audit(surface, atlas), point_ids_preserved=True,
        physical_plane_parameters_and_pressure_preserved=True,
        polarity_interval_provenance_preserved=True, full_continuous_arc_support_preserved=True,
        atlas_restart_exact=True, status=atlas.status,
        virtual_work=_covariant_reaction_work(surface, previous, final_step, expected)),
        fixed_mantle_source_input_audit=source_audit,
        differential_boundary_set=dict(continuous_material_pair_separation=continuous,
            traced_point_pairs_remain_unresolved=True,
            actual_owner_partition_as_set_preserved=True,
            physical_history_rematching_performed=False,
            status=differential.status, material_audit=_candidate_material_audit(surface, differential)))


def _archive_control(output, template):
    mesh, surface, inventory = _archive_fixture()
    assert len(inventory.cohorts) == 2
    frozen = snapshot(inventory), snapshot(surface)
    parameters = PrescribedBasalParameters(17, 8e4, 1e14, 2.)
    lid = {fragment.fragment_id: .2+1.3*index for index, fragment in enumerate(surface.fragments)}
    provenance = "Explicit nonempty archive source for a frozen-reference transport proposal"
    load = assemble_geometric_basal_load(mesh, surface, parameters, lid, provenance)
    traces = []
    for index, contact in enumerate(extract_contacts(surface.fragments, surface.radius_km)):
        horizontal = np.asarray(contact.normal_a_to_b)*(1. if contact.plate_b == 1 else -1.)
        start, end = contact.start, contact.end
        if np.cross(start, end)@horizontal < 0.: start, end = end, start
        traces.append(replace(template, trace_id=f"archive-physical:{index}", start=start, end=end,
            subducting_plate=0, overriding_plate=1, initial_gap_m=-1.,
            law=replace(template.law, cohesion_pa=1e7),
            provenance="Declared supported physical arc in a pre-existing two-cohort archival fixture"))
    history = tuple(PolarityEvidence(**row) for row in polarity_evidence(inventory))
    mechanics = _prepare(surface, traces, load, banks=_declared_banks(traces), initial_dt=1e6, history=history)
    state = initialize_bending_fault_transport(surface, inventory, mechanics, load)
    first = advance_bending_fault_transport(surface, state, load, 1e8, provenance=PROPOSAL_PROVENANCE)
    save_bending_fault_transport_checkpoint(output/"archive_middle.json", surface, first.state, load,
        provenance={"experiment": "nonempty passive archive with unaccepted finite transport"})
    origin, restored, restored_load, user = load_bending_fault_transport_checkpoint(output/"archive_middle.json",
        mesh=mesh, parameters=parameters, thermal_lid_by_fragment=lid, source_provenance=provenance)
    direct = advance_bending_fault_transport(surface, first.state, load, 1e8, provenance=PROPOSAL_PROVENANCE)
    resumed = advance_bending_fault_transport(origin, restored, restored_load, 1e8, provenance=PROPOSAL_PROVENANCE)
    assert direct.state == resumed.state and restored_load == load
    assert snapshot(direct.state.reference_inventory) == frozen[0]
    assert snapshot(surface) == frozen[1]
    assert direct.state.current_mechanics.polarity_history == mechanics.polarity_history
    moved = transported_fault_geometry(surface, direct.state.material_atlas)
    assert len(moved.history) == len(mechanics.polarity_history) > 0
    for old, new in zip(mechanics.polarity_history, moved.history):
        assert replace(new, start=old.start, end=old.end) == old
    save_bending_fault_transport_checkpoint(output/"archive_direct_final.json", surface, direct.state, load, provenance=user)
    save_bending_fault_transport_checkpoint(output/"archive_resumed_final.json", origin, resumed.state, restored_load, provenance=user)
    identical = (output/"archive_direct_final.json").read_bytes() == (output/"archive_resumed_final.json").read_bytes()
    assert identical
    return dict(cohort_count=len(inventory.cohorts), historical_transaction_count=len(inventory.transactions),
        retained_physical_polarity_interval_count=len(moved.history),
        original_archive_unchanged=True, original_surface_unchanged=True,
        original_acceptance_events_and_cohort_payloads_unchanged=True,
        no_candidate_archive_created=True, restart_exact=True, checkpoint_bytes_identical=True,
        candidate_material_audit=_candidate_material_audit(surface, direct.state.material_atlas),
        unresolved_material_pairing=direct.unresolved_material_pairing,
        trace_material_pair_separation_m=direct.trace_material_pair_separation_m)


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
    load = assemble_geometric_basal_load(mesh, surface, parameters, lid, SOURCE_PROVENANCE)
    mechanics = _prepare(surface, traces, load, banks=_declared_banks(traces))
    supports = validate_reference_bending_bank_support(surface, mechanics.fault_traces, mechanics.bending_parameters)
    state = initialize_bending_fault_transport(surface, inventory, mechanics, load)
    rows, final_step, previous = [], None, None
    for index in range(STEP_COUNT):
        previous = state.current_mechanics
        proposal = advance_bending_fault_transport(surface, state, load, DT_S, provenance=PROPOSAL_PROVENANCE)
        final_step = proposal.predictor
        row = _row(final_step, surface, previous)
        row.update(candidate_material_audit=_candidate_material_audit(surface, proposal.state.material_atlas),
            model=proposal.model, unresolved_material_pairing=proposal.unresolved_material_pairing,
            trace_material_pair_separation_m=proposal.trace_material_pair_separation_m,
            point_geometry_defects=[asdict(item) for item in proposal.point_geometry_defects],
            reference_source_work_receipt_preserved=True,
            candidate_mechanical_history_accepted=False)
        assert row["unresolved_material_pairing"]
        rows.append(row)
        state = proposal.state
        if index == 1:
            save_bending_fault_transport_checkpoint(output/"middle.json", surface, state, load,
                provenance={"experiment": "frozen mechanical predictor and unaccepted finite geometry"})
    save_bending_fault_transport_checkpoint(output/"direct_final.json", surface, state, load,
        provenance={"experiment": "frozen mechanical predictor and unaccepted finite geometry"})
    origin, resumed, restored_load, user = load_bending_fault_transport_checkpoint(output/"middle.json",
        mesh=mesh, parameters=parameters, thermal_lid_by_fragment=lid, source_provenance=SOURCE_PROVENANCE)
    for _ in range(2, STEP_COUNT):
        resumed = advance_bending_fault_transport(origin, resumed, restored_load, DT_S,
            provenance=PROPOSAL_PROVENANCE).state
    assert resumed == state and restored_load == load
    save_bending_fault_transport_checkpoint(output/"resumed_final.json", origin, resumed, restored_load, provenance=user)
    identical = (output/"direct_final.json").read_bytes() == (output/"resumed_final.json").read_bytes()
    assert identical and (snapshot(surface), snapshot(inventory)) == original
    try:
        validate_bending_contact_dynamics_state(material_atlas_surface(surface, state.material_atlas),
                                               state.current_mechanics, load)
    except ValueError as error:
        candidate_binding_refusal = str(error)
    else:
        raise AssertionError("An unaccepted endpoint must not inherit the original mechanical source binding")
    source_work = math.fsum(row["source_work_j"] for row in rows)
    assert math.isclose(source_work, state.current_mechanics.cumulative_source_work_j, rel_tol=2e-14, abs_tol=0.)
    kinematics = _kinematic_controls(surface, state.current_mechanics, load, mesh, parameters, lid, final_step, previous)
    archive = _archive_control(output, traces[0])
    mixed = sum(branch["locked"] > 0 and branch["sliding"] > 0 for row in rows for branch in row["branches"].values())
    sliding = sum(branch["sliding"] for row in rows for branch in row["branches"].values())
    assert sliding > 0
    report = dict(format="geometric-fault-transport-validation-1", status="complete",
        elapsed_seconds=time.perf_counter()-started,
        runtime=dict(python=platform.python_version(), platform=platform.platform(),
            packages={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "PyYAML", "pytest")}),
        inputs=dict(reference_basal_source=basal_load_to_dict(surface, load), step_duration_s=DT_S,
            step_count=STEP_COUNT, deformation_limits=LIMITS,
            declared_bending_banks=[asdict(bank) for bank in mechanics.bending_parameters]),
        reference_bank_support=asdict(supports), reference_force_steps=rows,
        point_count=len(mechanics.states), mixed_branch_trace_steps=mixed, sliding_point_steps=sliding,
        maximum_resultant_torque_relative_residual=max(row["torque_signal_relative_residual"] for row in rows),
        maximum_generalized_force_relative_residual=max(row["generalized_force_relative_residual"] for row in rows),
        maximum_incremental_work_relative_residual=max(row["incremental_energy_balance_relative_error"] for row in rows),
        reference_source_work_sum_j=source_work,
        candidate_mechanical_binding_refused=candidate_binding_refusal,
        maximum_candidate_gap_anchor_defect_m=max(abs(point["gap_anchor_defect_m"])
            for row in rows for point in row["point_geometry_defects"]),
        maximum_candidate_jump_anchor_defect_m=max(abs(point["jump_anchor_defect_m"])
            for row in rows for point in row["point_geometry_defects"]),
        final_transport_state=bending_fault_transport_to_dict(surface, state, load),
        finite_kinematic_controls=kinematics, nonempty_passive_archive=archive,
        source_surface_preserved=True, source_archive_preserved=True,
        transferred_material_km2=0., new_slab_cohorts=0, accepted_finite_geometry_steps=0,
        restart_exact=True, checkpoint_bytes_identical=identical,
        checkpoint_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.glob("*.json"))},
        cloud_source_bundle_sha256={name: hashlib.sha256((ROOT/"handoff"/name).read_bytes()).hexdigest()
            for name in ("cloud_sources.zip", "cloud_sources_manifest.json")},
        code_sha256=hashes, code_unchanged=hashes == _hashes(),
        scope="Exact chronological SO(3) material maps, two bank labels, overriding material-frame physical planes and clamp references, strict origin-linked checkpoints, continuous material-pair defect evidence. Force and work are accepted only on the frozen reference geometry. Endpoint proposals remain unaccepted; no spatial contact-history rematching, coupled changing-source work continuation, material admission, topology resolution or cooling feedback")
    assert report["code_unchanged"]
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    result = execute(parser.parse_args().out)
    print(json.dumps({name: result[name] for name in (
        "status", "elapsed_seconds", "point_count", "mixed_branch_trace_steps", "sliding_point_steps",
        "maximum_resultant_torque_relative_residual", "maximum_generalized_force_relative_residual",
        "maximum_incremental_work_relative_residual", "reference_source_work_sum_j",
        "restart_exact", "checkpoint_bytes_identical", "accepted_finite_geometry_steps")}, indent=2))
