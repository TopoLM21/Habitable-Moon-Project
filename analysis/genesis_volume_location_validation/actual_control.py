"""Accepted Genesis material stress and explicitly declared plane sections.

The actual 0.8684 Myr parent is replayed once through its public loader.  The
observer exposes the existing three-dimensional cohort memory at material
coordinates, and the section probes retain the complete current A/P chart.
Probe normals and offsets are computational inputs: they do not select a
physical nucleation point, dip, creation law, interface, or new time step.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import fields, is_dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.genesis_volume_slip_validation.actual_control import audit_state
from tectonics.genesis_volume_fracture_onset import (
    bind_calculated_volume_fracture_onset, calculated_volume_fracture_onset_to_dict,
)
from tectonics.genesis_volume_material_location import (
    bind_volume_material_location, volume_material_location_to_dict,
    volume_material_location_from_dict,
)
from tectonics.genesis_volume_material_sections import (
    prepare_volume_material_sections, evaluate_cohort_material_section,
    volume_material_sections_to_dict, volume_material_sections_from_dict,
)
from tectonics.genesis_volume_material_stress import (
    bind_volume_material_stress, evaluate_volume_material_stress,
    volume_material_stress_observer_to_dict,
    validate_volume_material_stress_observer_payload,
    volume_material_stress_to_dict, volume_material_stress_from_dict,
)
from tectonics.genesis_volume_slip_checkpoint import (
    load_volume_slip_checkpoint, save_volume_slip_checkpoint,
    volume_slip_checkpoint_to_dict,
)


PARENT_RAW_SHA256 = "62579a0bad128830a66e2f44579cf2b1f5d3835b691225b8d431102f3e4486d7"
PARENT_CHECKPOINT_SHA256 = "5d282ceeccc8d76acc38427d08a7e65575fddb509fc4b93115cbe6d318e91349"
FORMATION_SHA256 = "cd79ee2a5139c4b91abdb1b4714a761c9153cc9f4490f5aa63fa2facba458c89"
FORMAT = "genesis-volume-material-location-control-0.1"
SCOPE = "accepted_full_material_stress_and_declared_current_sections_not_physical_localization"
EXPLICIT_NORMALS = ((1., 2., 3.), (-2., 1., .5), (.7, -1.3, 1.1))
ANGULAR_PROBES = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
DEPTH_FRACTIONS = (0., .50, 1.)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value):
        return {field.name: plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def require(condition, message):
    if not condition:
        raise ValueError(message)


def seal(body, key):
    raw = {name: value for name, value in body.items() if name != key}
    return {**raw, key: digest(canonical(raw))}


def refused(name, raw, validator, *, seal_key, mutation_path, old_value, new_value):
    """Record a real source-context refusal after a correct DTO reseal."""
    resealed = seal(raw, seal_key)
    try:
        validator(resealed)
    except ValueError as error:
        return dict(name=name, status="REJECTED", resealed_descriptor_sha256=resealed[seal_key],
            descriptor_seal_recomputed=True, source_context_unchanged=True,
            mutation_path=mutation_path, old_value=old_value, new_value=new_value,
            exception_type=type(error).__name__, reason=str(error))
    raise ValueError(f"Resealed material-location mutation was accepted: {name}")


def accepted_point(state, cohort, barycentric, q):
    """Explicit full current kinematics used only to declare probe support."""
    from tectonics.genesis_volume_geometry import _chart, _scale
    face = np.asarray(state.geometry.faces[cohort.face_index], int)
    X = np.asarray(state.geometry.vertices_m, float)[face]
    U = np.asarray(state.displacement_m, float).reshape(-1, 6)[face]
    _, _, coefficient = _chart(X,
        state.geometry.column_mass_kg[cohort.face_index], state.geometry.density_kg_m3)
    lam = _scale(coefficient, float(q))
    point = np.asarray(barycentric, float)@(lam*(X+U[:, 3:])+(U[:, :3]-U[:, 3:]))
    return lam, point


def selected_material(state):
    nface = len(state.geometry.faces)
    selected_faces = (0, nface//2, nface-1)
    cohorts_by_face = {face: [c for c in state.geometry.cohorts if c.face_index == face]
                       for face in selected_faces}
    require(all(len(rows) >= 3 for rows in cohorts_by_face.values()),
            "Actual selected faces lack distinct inherited material cohorts")
    ids, barycentric, q, normals, index_records = [], [], [], [], []
    for face, declared in zip(selected_faces, EXPLICIT_NORMALS):
        rows = cohorts_by_face[face]
        chosen = (rows[0], rows[len(rows)//2], rows[-1])
        normal = np.asarray(declared, float)
        normal /= np.linalg.norm(normal)
        for cohort in chosen:
            for N in ANGULAR_PROBES:
                for fraction in DEPTH_FRACTIONS:
                    material_q = (cohort.q_lo if fraction == 0 else cohort.live_q_hi
                        if fraction == 1 else cohort.q_lo+fraction*(cohort.live_q_hi-cohort.q_lo))
                    ids.append(cohort.id); barycentric.append(N); q.append(material_q)
                    normals.append(normal.tolist())
                    index_records.append(dict(cohort_id=int(cohort.id), face_index=face,
                        owner_id=cohort.owner_id, material_id=cohort.material_id,
                        angular_barycentric=list(N), mass_coordinate_q=material_q,
                        depth_fraction_of_actual_live_q_interval=fraction,
                        normal_world=normal.tolist(), normal_provenance="explicit_computational_input"))
    return selected_faces, cohorts_by_face, (ids, barycentric, q, normals), index_records


def reconstruct_sections(raw, *, state, normal, offset, face):
    return volume_material_sections_from_dict(raw, geometry=state.geometry,
        displacement_m=state.displacement_m, normal_xyz=normal, offset_m=offset,
        face_indices=[face])


def run(parent_path, output):
    output.mkdir(parents=True, exist_ok=False)
    parent_bytes = parent_path.read_bytes()
    require(digest(parent_bytes) == PARENT_RAW_SHA256,
            "Material location control requires the sealed authentic accepted parent")
    parent_packet = json.loads(parent_bytes)
    require(parent_packet["checkpoint_sha256"] == PARENT_CHECKPOINT_SHA256,
            "Material location parent semantic source pin differs")
    # This is the single public planetary restore.  Descriptor validators below
    # recompute against this trusted runtime rather than loading it again.
    model, state = load_volume_slip_checkpoint(parent_path,
        expected_source_checkpoint_sha256=FORMATION_SHA256)
    require(len(state.geometry.cohorts) == 800 and abs(state.time_myr-.8684) < 1e-14
            and len(state.accepted_step_sizes_myr) == 4,
            "Material location authentic time, ancestry, or accepted interval count differs")
    require(canonical(volume_slip_checkpoint_to_dict(model, state)) == canonical(parent_packet),
            "Full accepted parent replay changed the archive")
    save_volume_slip_checkpoint(output/"parent_before.json", model, state)
    require((output/"parent_before.json").read_bytes() == parent_bytes,
            "Restored location parent is not byte-identical")
    actual_audit = audit_state(model, state)
    onset = bind_calculated_volume_fracture_onset(model, state)
    onset_payload = calculated_volume_fracture_onset_to_dict(onset)
    require(onset_payload["new_activation_event"] is None
            and onset_payload["newly_activating_faces"] == []
            and onset_payload["existing_active_faces"] == [],
            "Actual parent unexpectedly acquired a calculated activation event")
    observer = bind_volume_material_stress(model, state)
    observer_payload = volume_material_stress_observer_to_dict(observer)
    validate_volume_material_stress_observer_payload(observer_payload,
        expected_observer=observer)
    location = bind_volume_material_location(model, state, observer=observer, onset=onset)
    location_payload = volume_material_location_to_dict(location)
    volume_material_location_from_dict(location_payload, model=model, state=state)
    require(location_payload["orientation_offset_families"] == []
            and location_payload["status"] == "no_calculated_activation_event"
            and location_payload["selected_sharp_material_surface"] is None,
            "An explicit plane probe became an actual calculated material surface")
    for name, packet in (("actual_onset_prerequisite", onset_payload),
                         ("material_stress_observer", observer_payload),
                         ("bound_location", location_payload)):
        (output/f"{name}.json").write_bytes(canonical(packet)+b"\n")
    faces, by_face, coordinates, sample_index = selected_material(state)
    ids, N, q, normals = coordinates
    samples = evaluate_volume_material_stress(observer, ids, N, q,
                                             normal_world=normals)
    samples_payload = volume_material_stress_to_dict(samples)
    volume_material_stress_from_dict(samples_payload, observer=observer)
    (output/"material_stress_samples.json").write_bytes(canonical(samples_payload)+b"\n")
    # These probes expose source-context mutation barriers.  They do not
    # re-evolve the planet or allocate a fracture event from the observed field.
    barriers = []
    forged_observer = copy.deepcopy(observer_payload)
    original = forged_observer["elastic_constant"][0][0][0]
    forged_observer["elastic_constant"][0][0][0] = original+1e-4
    barriers.append(refused("accepted_cohort_memory_replaced", forged_observer,
        lambda raw: validate_volume_material_stress_observer_payload(raw, expected_observer=observer),
        seal_key="binding_sha256", mutation_path=["elastic_constant", 0, 0, 0],
        old_value=original, new_value=original+1e-4))
    forged_sample = copy.deepcopy(samples_payload)
    original = forged_sample["stress_world_tensor"][0][0][0]
    forged_sample["stress_world_tensor"][0][0][0] = original+1e6
    barriers.append(refused("pointwise_stress_fabricated", forged_sample,
        lambda raw: volume_material_stress_from_dict(raw, observer=observer),
        seal_key="sample_sha256", mutation_path=["stress_world_tensor", 0, 0, 0],
        old_value=original, new_value=original+1e6))
    forged_location = copy.deepcopy(location_payload)
    forged_surface = {"computational_plane_claimed_as_physical_surface": True}
    forged_location["selected_sharp_material_surface"] = forged_surface
    barriers.append(refused("explicit_probe_promoted_to_actual_surface", forged_location,
        lambda raw: volume_material_location_from_dict(raw, model=model, state=state),
        seal_key="location_sha256", mutation_path=["selected_sharp_material_surface"],
        old_value=None, new_value=forged_surface))
    section_cases, summaries = [], []
    for face, declared in zip(faces, EXPLICIT_NORMALS):
        selected_cohort = by_face[face][len(by_face[face])//2]
        anchor_N = np.full(3, 1/3)
        anchor_q = .5*(selected_cohort.q_lo+selected_cohort.live_q_hi)
        anchor_lambda, origin = accepted_point(state, selected_cohort, anchor_N, anchor_q)
        normal = np.asarray(declared, float); normal /= np.linalg.norm(normal)
        offset = float(normal@origin)
        sections = prepare_volume_material_sections(state.geometry, state.displacement_m,
            normal, offset, face_indices=[face])
        section_payload = volume_material_sections_to_dict(sections)
        reconstruct_sections(section_payload, state=state, normal=normal, offset=offset, face=face)
        if not section_cases:
            forged_section = copy.deepcopy(section_payload)
            original = forged_section["cohorts"][0]["slope_P_m"][0][0]
            forged_section["cohorts"][0]["slope_P_m"][0][0] = original+1.
            barriers.append(refused("through_depth_P_warp_replaced", forged_section,
                lambda raw: reconstruct_sections(raw, state=state, normal=normal, offset=offset, face=face),
                seal_key="sections_sha256", mutation_path=["cohorts", 0, "slope_P_m", 0, 0],
                old_value=original, new_value=original+1.))
        probes, probe_ids, probe_N, probe_q = [], [], [], []
        for record in sections.cohorts:
            for interval_index, interval in enumerate(record.intervals):
                lam = .5*sum(interval.lambda_bounds)
                for fraction in (.23, .77):
                    point = evaluate_cohort_material_section(record, lam, fraction=fraction)
                    probes.append(dict(cohort_id=int(record.cohort_id),
                        face_index=int(record.face_index), interval_index=interval_index,
                        radial_scale=lam, fraction=fraction, point=plain(point)))
                    probe_ids.append(record.cohort_id)
                    probe_N.append(point.barycentric.tolist()); probe_q.append(point.q)
        require(bool(probes), "Declared current material plane lacks computable interior support")
        probe_samples = evaluate_volume_material_stress(observer, probe_ids, probe_N, probe_q,
                                                       normal_world=normal)
        probe_samples_payload = volume_material_stress_to_dict(probe_samples)
        volume_material_stress_from_dict(probe_samples_payload, observer=observer)
        case = dict(name=f"supplied_current_plane_face_{face}", computational_fixture=True,
            plane_provenance="explicit_normal_and_centroid_anchor_not_source_nucleation_or_dip_law",
            normal_xyz=normal.tolist(), explicitly_supplied_unnormalized_normal=list(declared),
            offset_m=offset, face_indices=[face], origin_xyz_m=origin.tolist(),
            anchor=dict(cohort_id=int(selected_cohort.id), angular_barycentric=anchor_N.tolist(),
                mass_coordinate_q=anchor_q, radial_scale=anchor_lambda),
            sections=section_payload, probes=probes, probe_material_stress_samples=probe_samples_payload,
            physical_surface_selected=False, source_creation_energy_law=None,
            creation_work_debited_from_Genesis_j=0., physical_creation_admitted=False,
            physical_release_admitted=False, physical_contact_admitted=False,
            dofs_created_or_released=False, material_memory_changed=False,
            integration_scope="analytic_slice_geometry_no_area_or_volume_quadrature_claim")
        section_cases.append(case)
        (output/f"{case['name']}.json").write_bytes(canonical(case)+b"\n")
        summaries.append(dict(name=case["name"], original_face_index=face,
            retained_actual_cohorts=len(sections.cohorts),
            interior_cut_cohorts=sum(record.interior_cut for record in sections.cohorts),
            constant_lambda_sections=sum(len(record.constant_lambda_sections) for record in sections.cohorts),
            nondegenerate_lambda_pieces=sum(len(record.intervals) for record in sections.cohorts),
            probed_full_material_points=len(probes),
            maximum_plane_residual_m=max(abs(item["point"]["plane_residual_m"]) for item in probes),
            physical_surface_selected=False, source_creation_energy_law=None))
    fixture = dict(format=FORMAT, scope=SCOPE, parent_raw_sha256=PARENT_RAW_SHA256,
        parent_path=str(parent_path.resolve()), parent_checkpoint=parent_packet,
        actual_onset_prerequisite=onset_payload, bound_location=location_payload,
        material_stress_observer=observer_payload, material_stress_samples=samples_payload,
        material_stress_sample_index=sample_index, section_cases=section_cases,
        observed_resealed_source_context_mutation_barriers=barriers,
        stress_scope="current_accepted_endpoint_full_cohort_elastic_memory_not_last_interval_average",
        stress_representation="inherited_fixed_orthonormal_cohort_basis_small_strain_world_tensor_no_finite_deformation_pushforward",
        material_chart="x(N,lambda)=N@(lambda*(X+P)+(A-P))",
        source_nucleation_location_law=None, source_physical_dip_law=None,
        source_surface_creation_energy_law=None, selected_sharp_material_surface=None,
        new_Genesis_time_steps=0, actual_activation_accepted=False,
        creation_work_debited_from_Genesis_j=0., physical_creation_admitted=False,
        physical_release_admitted=False, physical_contact_admitted=False,
        material_memory_changed=False, dofs_created_or_released=False,
        history_reinitialized=False, point_one_complete=False)
    fixture = seal(fixture, "fixture_sha256")
    (output/"control_fixture.json").write_bytes(canonical(fixture)+b"\n")
    save_volume_slip_checkpoint(output/"parent_after.json", model, state)
    require((output/"parent_after.json").read_bytes() == parent_bytes
            and parent_path.read_bytes() == parent_bytes,
            "Current material observations changed the accepted source or immutable parent")
    stress_values = np.asarray(samples.stress_world_tensor).reshape(-1, 3, 3)
    per_cohort_variation = [float(np.max(np.ptp(stress_values[np.asarray(ids) == cohort_id], axis=0)))
                            for cohort_id in sorted(set(ids))]
    pose_rows = np.asarray(state.displacement_m, float).reshape(-1, 6)
    sample_faces = np.asarray(state.geometry.faces, int)[np.asarray(samples.face_indices, int)]
    top_positions = np.einsum("ni,nij->nj", np.asarray(N, float),
        np.asarray(state.geometry.vertices_m, float)[sample_faces]+pose_rows[sample_faces, :3])
    radial_copy_positions = np.asarray(samples.radial_scale)[:, None]*top_positions
    full_minus_radial_copy = np.asarray(samples.current_position_m)-radial_copy_positions
    report = dict(format="actual-genesis-volume-material-location-control-0.1",
        status="PASS", scope=SCOPE, parent_raw_sha256=PARENT_RAW_SHA256,
        parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
        formation_checkpoint_sha256=FORMATION_SHA256, time_myr=state.time_myr,
        cohort_count=len(state.geometry.cohorts), accepted_Genesis_intervals_replayed=4,
        parent_replayed_once_from_formation_checkpoint=True,
        parent_before_after_entire_packets_byte_equal=True,
        actual_state_audit=actual_audit, fixture_sha256=fixture["fixture_sha256"],
        source_calculated_onset_sha256=onset_payload["onset_sha256"],
        bound_location_sha256=location_payload["location_sha256"],
        accepted_material_stress_sample_count=len(ids), selected_actual_faces=list(faces),
        selected_actual_cohort_count=len(set(ids)), section_cases=summaries,
        angular_probe_scope="three_original_material_triangle_corners",
        depth_probe_scope="explicit_one_sided_live_q_endpoints_and_midpoint",
        selected_cohort_stress_variation_max_pa=max(per_cohort_variation),
        selected_cohort_stress_variation_min_pa=min(per_cohort_variation),
        maximum_current_pose_difference_from_radial_top_copy_m=float(np.max(np.linalg.norm(full_minus_radial_copy, axis=1))),
        source_context_resealed_mutation_barriers=barriers,
        actual_new_activation_event=None, actual_active_weak_plane_count=0,
        actual_orientation_offset_families=0, source_nucleation_location_law_unknown=True,
        source_physical_dip_law_unknown=True, actual_surface_creation_law_unknown=True,
        new_Genesis_time_steps=0, no_replacement_initialization=True,
        sharp_physical_banks_created=False, ties_released=False,
        actual_creation_work_assigned=False, actual_creation_work_debited_j=0.,
        physical_banks_contact_underthrust_admitted=False, point_one_complete=False)
    report = seal(report, "report_sha256")
    (output/"control_report.json").write_bytes(canonical(report)+b"\n")
    print(json.dumps(report, sort_keys=True))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", type=Path, default=ROOT/
        "analysis/genesis_volume_creation_validation/evidence/actual_continuation/direct_after_second.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.parent_checkpoint, args.output)


if __name__ == "__main__":
    main()
