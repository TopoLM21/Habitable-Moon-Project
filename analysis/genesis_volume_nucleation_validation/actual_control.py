"""Authentic Genesis nucleation gates and computational embedded-surface proof.

The accepted planet is restored once through the public slip loader. Prospective
constitutive inputs and explicit embedded-plane probes do not fabricate a
physical activation, funded fracture, new material history or clock interval.
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
from tectonics.genesis_volume_creation_law import ProspectiveCreationLaw
from tectonics.genesis_volume_embedded_surface import (
    prepare_volume_embedded_surface, evaluate_embedded_surface_positions,
    evaluate_embedded_surface_jump, integrate_embedded_surface_traction,
    integrate_embedded_surface_virtual_work, volume_embedded_surface_to_dict,
    volume_embedded_surface_from_dict,
)
from tectonics.genesis_volume_embedded_prestress import (
    prepare_volume_embedded_prestress, volume_embedded_prestress_to_dict,
    volume_embedded_prestress_from_dict,
)
from tectonics.genesis_volume_fracture_onset import (
    bind_calculated_volume_fracture_onset, calculated_volume_fracture_onset_to_dict,
)
from tectonics.genesis_volume_material_stress import (
    bind_volume_material_stress, volume_material_stress_observer_to_dict,
)
from tectonics.genesis_volume_nucleation import (
    ProspectiveVolumeNucleationLaw, bind_volume_nucleation_proposal,
    evaluate_volume_nucleation_criterion, volume_nucleation_proposal_to_dict,
    volume_nucleation_proposal_from_dict,
)
from tectonics.genesis_volume_slip_checkpoint import (
    load_volume_slip_checkpoint, save_volume_slip_checkpoint,
    volume_slip_checkpoint_to_dict,
)

PARENT_RAW_SHA256 = "62579a0bad128830a66e2f44579cf2b1f5d3835b691225b8d431102f3e4486d7"
PARENT_CHECKPOINT_SHA256 = "5d282ceeccc8d76acc38427d08a7e65575fddb509fc4b93115cbe6d318e91349"
FORMATION_SHA256 = "cd79ee2a5139c4b91abdb1b4714a761c9153cc9f4490f5aa63fa2facba458c89"
FORMAT = "genesis-volume-nucleation-embedded-control-0.1"
SCOPE = "prospective_source_bound_birth_gate_and_computational_embedded_trace_not_funded_physical_release"
PREVIOUS_CASE_DIRECTORY = ROOT/"analysis/genesis_volume_location_validation/evidence/actual_control"
QUADRATURE_ORDERS = (4, 8, 12)


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
    """Run the real producer validator after correctly recomputing its seal."""
    resealed = seal(raw, seal_key)
    try:
        validator(resealed)
    except ValueError as error:
        return dict(name=name, status="REJECTED", resealed_descriptor_sha256=resealed[seal_key],
            descriptor_seal_recomputed=True, source_context_unchanged=True,
            mutation_path=mutation_path, old_value=old_value, new_value=new_value,
            exception_type=type(error).__name__, reason=str(error))
    raise ValueError(f"Resealed nucleation/embedded mutation was accepted: {name}")


def computational_laws():
    # New explicit examples. Neither Gc nor stiffness is read from historical
    # damage/slip dissipation or the old contact penalty parameters.
    creation = ProspectiveCreationLaw(fracture_energy_j_m2=1e6,
        peak_traction_pa=1e8, cohesive_stiffness_pa_m=1e10,
        compression_stiffness_pa_m=1e10)
    law = ProspectiveVolumeNucleationLaw(tensile_strength_pa=12e6,
        shear_cohesion_pa=4e6, friction_coefficient=.6, seed_radius_m=100.,
        creation_law=creation)
    return law, creation


def manufactured_criterion(law):
    # This tests the constitutive algebra; it never overwrites the accepted
    # planetary stress, activation age, damage or face eligibility.
    stresses = np.asarray((((20e6, 0., 0.), (0., -2e6, 0.), (0., 0., -8e6)),
        ((-10e6, 20e6, 0.), (20e6, -10e6, 0.), (0., 0., -10e6))), float)
    criterion = evaluate_volume_nucleation_criterion(stresses, law)
    require(np.all(np.asarray(criterion.yield_score) > 0.),
            "Manufactured criterion lacks a positive constitutive overstress")
    require(np.isclose(criterion.tensile_score[0], 20e6/law.tensile_strength_pa-1.,
        rtol=2e-14, atol=0.), "Manufactured principal-tension criterion differs")
    return dict(name="explicit_manufactured_overstress_not_actual_eligibility",
        computational_fixture=True, stress_tensor_pa=stresses.tolist(),
        criterion=plain(criterion), inherited_planetary_fields_replaced=False,
        actual_activation_accepted=False, physical_creation_admitted=False,
        creation_work_debited_from_Genesis_j=0.)


def virtual_increments(ndof):
    index = np.arange(ndof, dtype=float)
    # Two independent banks using the complete existing A/P parent DOFs.
    return (1e-5*np.sin(.173*(index+1.)), 1e-5*np.cos(.119*(index+2.)))


def reconstruct_surface(raw, *, state, normal, offset, face, order):
    return volume_embedded_surface_from_dict(raw, geometry=state.geometry,
        displacement_m=state.displacement_m, normal_xyz=normal, offset_m=offset,
        face_indices=[face], quadrature_order=order)


def source_plane_case(face, parent_sha):
    path = PREVIOUS_CASE_DIRECTORY/f"supplied_current_plane_face_{face}.json"
    data = path.read_bytes()
    prior = json.loads(data)
    require(prior["computational_fixture"] is True
        and prior["physical_surface_selected"] is False
        and prior["physical_creation_admitted"] is False
        and prior["physical_release_admitted"] is False
        and prior["face_indices"] == [face]
        and prior["probe_material_stress_samples"]["parent_checkpoint_sha256"] == parent_sha,
        "Previous computational plane differs from the authentic parent")
    return prior, dict(path=str(path.relative_to(ROOT)), raw_sha256=digest(data),
        plane_provenance=prior["plane_provenance"],
        inherited_anchor=prior["anchor"], inherited_origin_xyz_m=prior["origin_xyz_m"])


def force_family_errors(actual, expected):
    actual, expected = np.asarray(actual).reshape(2, -1, 6), np.asarray(expected).reshape(2, -1, 6)
    return [float(np.linalg.norm(actual[:, :, start:start+3]-expected[:, :, start:start+3])/
        max(np.linalg.norm(expected[:, :, start:start+3]), 1.)) for start in (0, 3)]


def embedded_work_fixture(surface, prestress_payload, paired, absolute_displacement):
    pullback = prestress_payload["pullback"]
    reference_traction = np.asarray(pullback["reference_traction_world_pa"], float)
    birth_traction = np.asarray(pullback["birth_traction_world_pa"], float)
    reference_force = integrate_embedded_surface_traction(surface, reference_traction,
        area_convention="reference")
    birth_force = integrate_embedded_surface_traction(surface, birth_traction,
        area_convention="current")
    saved_force = np.asarray(pullback["paired_parent_force_n"], float)
    reference_saved_errors = force_family_errors(reference_force, saved_force)
    current_reference_errors = force_family_errors(birth_force, reference_force)
    require(max(reference_saved_errors+current_reference_errors) < 3e-12,
            "Reference/birth metric conversion changed the complete paired A/P force")
    jump = evaluate_embedded_surface_jump(surface, paired)
    zero_jump = evaluate_embedded_surface_jump(surface, np.stack((paired[0], paired[0])))
    require(not np.any(zero_jump), "Paired interpolation failed exact zero-jump tying")
    positions = evaluate_embedded_surface_positions(surface, absolute_displacement)
    position_error = float(np.max(np.linalg.norm(positions-surface.current_position_m, axis=1), initial=0.))
    position_scale = max(float(np.max(np.linalg.norm(surface.current_position_m, axis=1), initial=0.)), 1.)
    require(position_error <= 64*np.finfo(float).eps*position_scale,
            "Frozen material trace failed to reproduce the complete accepted absolute A/P pose")
    reference_work = integrate_embedded_surface_virtual_work(surface, reference_traction,
        paired, area_convention="reference")
    birth_work = integrate_embedded_surface_virtual_work(surface, birth_traction,
        paired, area_convention="current")
    nodal_work = float(np.sum(reference_force*paired))
    scale = max(abs(reference_work), abs(birth_work), abs(nodal_work),
        float(np.sum(np.abs(reference_force*paired))), 1.)
    errors = [abs(reference_work-birth_work)/scale, abs(reference_work-nodal_work)/scale]
    require(max(errors) < 3e-12,
            "Embedded stress trace lost its discrete force/virtual-work duality")
    return dict(parent_ndof=surface.parent_ndof,
        virtual_increment_provenance="two_explicit_deterministic_parent_A_P_fields_not_released_DOFs",
        paired_parent_virtual_increments_m=paired.tolist(),
        virtual_jump_m=jump.tolist(), exact_zero_jump_for_equal_fields=True,
        frozen_material_absolute_pose_reproduction_max_error_m=position_error,
        reference_paired_parent_force_n=reference_force.tolist(),
        equivalent_birth_paired_parent_force_n=birth_force.tolist(),
        reference_saved_force_relative_errors_A_P=reference_saved_errors,
        birth_reference_force_relative_errors_A_P=current_reference_errors,
        reference_virtual_work_j=reference_work, birth_virtual_work_j=birth_work,
        paired_nodal_virtual_work_j=nodal_work, virtual_work_relative_errors=errors,
        virtual_work_claim="discrete_trace_duality_not_parent_constraint_replacement_or_release_balance")


def run(parent_path, output):
    output.mkdir(parents=True, exist_ok=False)
    parent_bytes = parent_path.read_bytes()
    require(digest(parent_bytes) == PARENT_RAW_SHA256,
            "Nucleation control requires the sealed authentic accepted parent")
    parent_packet = json.loads(parent_bytes)
    require(parent_packet["checkpoint_sha256"] == PARENT_CHECKPOINT_SHA256,
            "Nucleation parent semantic source pin differs")
    # This is the single public planetary restore. All later descriptor
    # validators recompute against this trusted runtime without replaying it.
    model, state = load_volume_slip_checkpoint(parent_path,
        expected_source_checkpoint_sha256=FORMATION_SHA256)
    require(len(state.geometry.cohorts) == 800 and abs(state.time_myr-.8684) < 1e-14
        and len(state.accepted_step_sizes_myr) == 4,
        "Nucleation authentic time, ancestry, or accepted interval count differs")
    require(canonical(volume_slip_checkpoint_to_dict(model, state)) == canonical(parent_packet),
            "Full accepted nucleation parent replay changed the archive")
    save_volume_slip_checkpoint(output/"parent_before.json", model, state)
    require((output/"parent_before.json").read_bytes() == parent_bytes,
            "Restored nucleation parent is not byte-identical")
    actual_audit = audit_state(model, state)
    observer = bind_volume_material_stress(model, state)
    observer_payload = volume_material_stress_observer_to_dict(observer)
    onset = bind_calculated_volume_fracture_onset(model, state)
    onset_payload = calculated_volume_fracture_onset_to_dict(onset)
    require(onset_payload["new_activation_event"] is None
        and onset_payload["newly_activating_faces"] == []
        and onset_payload["existing_active_faces"] == [],
        "Actual parent unexpectedly acquired a calculated activation event")
    law, creation_law = computational_laws()
    proposals = {}
    for name, declared in (("unconfigured", None), ("explicit_computational_law", law)):
        proposal = bind_volume_nucleation_proposal(model, state, declared,
            observer=observer, onset=onset)
        raw = volume_nucleation_proposal_to_dict(proposal)
        volume_nucleation_proposal_from_dict(raw, model=model, state=state, law=declared)
        require(raw["status"] == "no_calculated_activation_event"
            and raw["newly_activating_faces"] == [] and raw["existing_active_faces"] == []
            and raw["candidate"] is None and raw["corner_catalogue"] == []
            and raw["computational_candidate_proposals"] == []
            and raw["selected_sharp_material_surface"] is None
            and raw["funding_admitted"] is False
            and raw["creation_work_debited_from_Genesis_j"] == 0.,
            "Prospective parameters bypassed the inherited activation gate")
        proposals[name] = raw
        (output/f"nucleation_{name}.json").write_bytes(canonical(raw)+b"\n")
    (output/"actual_onset_prerequisite.json").write_bytes(canonical(onset_payload)+b"\n")
    (output/"material_stress_observer.json").write_bytes(canonical(observer_payload)+b"\n")
    manufactured = manufactured_criterion(law)
    (output/"manufactured_criterion.json").write_bytes(canonical(manufactured)+b"\n")
    barriers = []
    forged_law = copy.deepcopy(proposals["explicit_computational_law"])
    original = forged_law["declared_prospective_law"]["seed_radius_m"]
    forged_law["declared_prospective_law"]["seed_radius_m"] = original+1.
    barriers.append(refused("declared_seed_extent_replaced", forged_law,
        lambda raw: volume_nucleation_proposal_from_dict(raw, model=model, state=state, law=law),
        seal_key="nucleation_sha256", mutation_path=["declared_prospective_law", "seed_radius_m"],
        old_value=original, new_value=original+1.))
    forged_candidate = copy.deepcopy(proposals["explicit_computational_law"])
    fake_candidate = {"forged_eligible_Gamma_on_inactive_actual_source": True}
    forged_candidate["candidate"] = fake_candidate
    barriers.append(refused("physical_seed_invented_on_actual_noevent_source", forged_candidate,
        lambda raw: volume_nucleation_proposal_from_dict(raw, model=model, state=state, law=law),
        seal_key="nucleation_sha256", mutation_path=["candidate"],
        old_value=None, new_value=fake_candidate))
    paired = np.stack(virtual_increments(len(state.displacement_m)))
    cases, summaries = [], []
    for face in (0, len(state.geometry.faces)//2, len(state.geometry.faces)-1):
        prior, prior_pin = source_plane_case(face, PARENT_CHECKPOINT_SHA256)
        normal, offset = np.asarray(prior["normal_xyz"], float), float(prior["offset_m"])
        records, archived_surface, archived_prestress, archived_work = [], None, None, None
        for order in QUADRATURE_ORDERS:
            surface = prepare_volume_embedded_surface(state.geometry, state.displacement_m,
                normal, offset, face_indices=[face], quadrature_order=order)
            require(surface.parent_ndof == 252 and surface.point_count > 0
                and len(surface.selected_cohort_ids) == 10
                and len(np.unique(surface.cohort_ids)) == 10,
                "Embedded computational plane lost the complete original face material support")
            prestress = prepare_volume_embedded_prestress(model, state, surface,
                observer=observer, creation_law=creation_law)
            prestress_payload = volume_embedded_prestress_to_dict(prestress)
            work = embedded_work_fixture(surface, prestress_payload, paired, state.displacement_m)
            records.append(dict(quadrature_order=order, point_count=surface.point_count,
                patch_count=surface.patch_count,
                total_reference_area_m2=surface.total_reference_area_m2,
                total_current_area_m2=surface.total_current_area_m2,
                reference_paired_parent_force_n=work["reference_paired_parent_force_n"],
                reference_virtual_work_j=work["reference_virtual_work_j"],
                equivalent_birth_virtual_work_j=work["birth_virtual_work_j"],
                full_separation_cost_on_explicit_surface_j=prestress_payload["pullback"][
                    "full_separation_cost_on_explicit_surface_j"],
                physical_creation_admitted=False, funding_admitted=False))
            if order == 8:
                archived_surface = volume_embedded_surface_to_dict(surface)
                archived_prestress, archived_work = prestress_payload, work
                reconstruct_surface(archived_surface, state=state, normal=normal,
                    offset=offset, face=face, order=order)
                volume_embedded_prestress_from_dict(archived_prestress, model=model, state=state,
                    surface=surface, observer=observer, creation_law=creation_law)
                if not cases:
                    forged_surface = copy.deepcopy(archived_surface)
                    original = forged_surface["mass_coordinate_q"][0]
                    forged_surface["mass_coordinate_q"][0] = original+1e-5
                    barriers.append(refused("frozen_original_material_q_replaced", forged_surface,
                        lambda raw: reconstruct_surface(raw, state=state, normal=normal,
                            offset=offset, face=face, order=order),
                        seal_key="embedded_surface_sha256", mutation_path=["mass_coordinate_q", 0],
                        old_value=original, new_value=original+1e-5))
                    forged_stress = copy.deepcopy(archived_prestress)
                    original = forged_stress["pullback"]["reference_traction_world_pa"][0][0]
                    forged_stress["pullback"]["reference_traction_world_pa"][0][0] = original+1e6
                    forged_stress["pullback"] = seal(forged_stress["pullback"], "pullback_sha256")
                    item = refused("saved_pointwise_reference_traction_replaced", forged_stress,
                        lambda raw: volume_embedded_prestress_from_dict(raw, model=model, state=state,
                            surface=surface, observer=observer, creation_law=creation_law),
                        seal_key="embedded_prestress_sha256",
                        mutation_path=["pullback", "reference_traction_world_pa", 0, 0],
                        old_value=original, new_value=original+1e6)
                    item["nested_descriptor_seals_recomputed"] = ["pullback_sha256"]
                    barriers.append(item)
        final = records[-1]
        convergence = []
        for row in records[:-1]:
            area_errors = [abs(row[k]-final[k])/max(abs(final[k]), 1.) for k in
                ("total_reference_area_m2", "total_current_area_m2")]
            force_errors = force_family_errors(row["reference_paired_parent_force_n"],
                final["reference_paired_parent_force_n"])
            require(max(area_errors+force_errors) < 2e-7,
                    "Embedded quadrature lacks reference/current area or full A/P force convergence")
            convergence.append(dict(quadrature_order=row["quadrature_order"], reference_order=12,
                area_relative_errors_reference_current=area_errors,
                reference_force_relative_errors_A_P=force_errors))
        require(archived_surface is not None and archived_prestress is not None,
                "Embedded order-eight archive was not produced")
        case = dict(name=f"explicit_embedded_plane_face_{face}", computational_fixture=True,
            plane_provenance="same_prior_explicit_normal_and_centroid_anchor_not_physical_nucleation",
            previous_plane_source_pin=prior_pin, normal_xyz=normal.tolist(), offset_m=offset,
            face_indices=[face], embedded_surface=archived_surface,
            embedded_prestress=archived_prestress, virtual_work=archived_work,
            quadrature_convergence_records=records,
            quadrature_convergence_comparisons=convergence,
            creation_law_provenance="explicit_new_computational_constitutive_example_not_historical_Gc",
            physical_seed_sphere_clipping_performed=False, actual_nucleation_candidate=None,
            physical_creation_admitted=False, physical_release_admitted=False,
            physical_contact_admitted=False, funding_admitted=False,
            dofs_created_or_released=False, material_memory_changed=False,
            creation_work_debited_from_Genesis_j=0.,
            paired_force_scope="reference_material_stress_trace_duality_not_parent_constraint_reaction")
        cases.append(case)
        (output/f"{case['name']}.json").write_bytes(canonical(case)+b"\n")
        summaries.append(dict(name=case["name"], original_face_index=face,
            retained_original_cohorts=10, quadrature_order=8,
            embedded_points=records[1]["point_count"],
            reference_area_m2=records[1]["total_reference_area_m2"],
            current_area_m2=records[1]["total_current_area_m2"],
            quadrature_convergence=convergence,
            maximum_force_virtual_work_relative_error=max(archived_work["virtual_work_relative_errors"]),
            full_separation_cost_computational_j=records[1]["full_separation_cost_on_explicit_surface_j"],
            actual_creation_work_debited_from_Genesis_j=0., physical_creation_admitted=False))
    require(len(barriers) == 4, "Nucleation control lost a real resealed source-context refusal")
    fixture = dict(format=FORMAT, scope=SCOPE, parent_raw_sha256=PARENT_RAW_SHA256,
        parent_path=str(parent_path.resolve()), parent_checkpoint=parent_packet,
        actual_onset_prerequisite=onset_payload, material_stress_observer=observer_payload,
        nucleation_unconfigured=proposals["unconfigured"],
        nucleation_explicit_computational_law=proposals["explicit_computational_law"],
        manufactured_criterion=manufactured, embedded_cases=cases,
        observed_resealed_source_context_mutation_barriers=barriers,
        original_parent_ndof=252, prospective_nucleation_law_provenance=
            "explicit_new_convex_constitutive_choice_with_inherited_damage_persistence_gate",
        stress_scope="current_accepted_endpoint_full_cohort_elastic_memory_not_last_interval_average",
        stress_representation="inherited_fixed_frame_small_strain_reference_energy_stress_not_finite_Cauchy_pushforward",
        material_chart="x(N,lambda)=N@(lambda*(X+P)+(A-P))",
        nucleation_seed_clipping_implemented=False,
        physical_Gamma_selected_on_actual_source=False, actual_activation_accepted=False,
        new_Genesis_time_steps=0, creation_work_debited_from_Genesis_j=0.,
        physical_creation_admitted=False, physical_release_admitted=False,
        physical_contact_admitted=False, funding_admitted=False,
        parent_constraint_reaction_replaced=False, actual_birth_energy_balance_accepted=False,
        material_memory_changed=False, dofs_created_or_released=False,
        history_reinitialized=False, point_one_complete=False)
    fixture = seal(fixture, "fixture_sha256")
    (output/"control_fixture.json").write_bytes(canonical(fixture)+b"\n")
    save_volume_slip_checkpoint(output/"parent_after.json", model, state)
    require((output/"parent_after.json").read_bytes() == parent_bytes
        and parent_path.read_bytes() == parent_bytes,
        "Prospective nucleation/embedded observations changed the accepted parent")
    report = dict(format="actual-genesis-volume-nucleation-embedded-control-0.1",
        status="PASS", scope=SCOPE, parent_raw_sha256=PARENT_RAW_SHA256,
        parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
        formation_checkpoint_sha256=FORMATION_SHA256, time_myr=state.time_myr,
        cohort_count=len(state.geometry.cohorts), accepted_Genesis_intervals_replayed=4,
        parent_replayed_once_from_formation_checkpoint=True,
        parent_before_after_entire_packets_byte_equal=True, actual_state_audit=actual_audit,
        fixture_sha256=fixture["fixture_sha256"],
        actual_active_weak_plane_count=0, actual_new_activation_event=None,
        actual_unconfigured_nucleation_status=proposals["unconfigured"]["status"],
        actual_declared_nucleation_status=proposals["explicit_computational_law"]["status"],
        actual_inherited_activation_gate_retained=True,
        actual_physical_nucleation_candidate=None,
        source_creation_toughness_recovered=False,
        explicit_new_constitutive_examples_calibrated=False,
        manufactured_overstress_criterion_cases=2,
        embedded_cases=summaries, original_parent_ndof=252,
        source_context_resealed_mutation_barriers=barriers,
        new_Genesis_time_steps=0, no_replacement_initialization=True,
        actual_seed_sphere_clipping_performed=False, actual_creation_work_debited_j=0.,
        sharp_physical_banks_created=False, ties_released=False,
        parent_constraint_reaction_replaced=False, actual_birth_energy_balance_accepted=False,
        funding_admitted=False, physical_banks_contact_underthrust_admitted=False,
        point_one_complete=False)
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
