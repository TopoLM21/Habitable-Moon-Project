"""Zero-time prestress insertion probes on the genuine accepted Genesis state.

The 0.8684 Myr archive is restored through its full four-interval replay. The
actual source remains uncut and has no new activation event or creation law.
Explicit edge supports, reaction representatives and constitutive paths are
computational fixtures. They replace a tied reaction once without updating
the planet's accepted memory, clock, forcing or historical energy ledgers.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.genesis_volume_creation_validation.actual_control import plain, surface_packet
from analysis.genesis_volume_slip_validation.actual_control import audit_state
from tectonics.genesis_volume_bank_reaction import (
    prepare_volume_bank_reaction, prepare_volume_reaction_traction,
    volume_bank_reaction_to_dict, volume_reaction_traction_to_dict,
    validate_volume_bank_reaction_payload, validate_volume_reaction_traction_payload,
)
from tectonics.genesis_volume_banks import prepare_volume_banks
from tectonics.genesis_volume_creation import bind_volume_creation_context, creation_context_to_dict
from tectonics.genesis_volume_creation_law import ProspectiveCreationLaw
from tectonics.genesis_volume_fracture_onset import (
    bind_calculated_volume_fracture_onset, calculated_volume_fracture_onset_to_dict,
    calculated_volume_fracture_onset_from_dict,
)
from tectonics.genesis_volume_fracture_surface import (
    prepare_volume_fracture_surface, evaluate_surface_jump, integrate_surface_traction,
)
from tectonics.genesis_volume_prestress_creation import (
    prepare_prestress_creation, evaluate_prestress_creation, PrestressCreationRefusal,
)
from tectonics.genesis_volume_slip_checkpoint import (
    load_volume_slip_checkpoint, save_volume_slip_checkpoint, volume_slip_checkpoint_to_dict,
)


PARENT_RAW_SHA256 = "62579a0bad128830a66e2f44579cf2b1f5d3835b691225b8d431102f3e4486d7"
PARENT_CHECKPOINT_SHA256 = "5d282ceeccc8d76acc38427d08a7e65575fddb509fc4b93115cbe6d318e91349"
FORMATION_SHA256 = "cd79ee2a5139c4b91abdb1b4714a761c9153cc9f4490f5aa63fa2facba458c89"
FORMAT = "genesis-volume-insertion-control-0.1"
HISTORY_FORMAT = "genesis-computational-prestress-insertion-history-0.1"
SCOPE = "accepted_interval_reaction_replacement_and_seeded_creation_probes_not_physical_fracture"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, name, *, rtol=8e-11, atol=1e-8):
    a, b = np.asarray(actual, float), np.asarray(expected, float)
    require(a.shape == b.shape, f"{name}: dimensions differ")
    scale = max(float(np.max(np.abs(b), initial=0.)), 1e-300)
    error = float(np.max(np.abs(a-b), initial=0.))
    require(error <= atol+rtol*scale, f"{name}: force or work identity failed")
    return error/max(scale, atol, 1e-300)


def force_identity(actual, expected, reference_force, name):
    """Check A and thin-depth P separately against the replaced force.

    The accepted remaining residual can be much smaller than individual
    forces. Its tolerance uses the same force balance's A/P scales, preserving
    that residual instead of treating its rounded near-zero value as a scale.
    """
    a, b, reference = (np.asarray(value, float).reshape(-1, 6)
                       for value in (actual, expected, reference_force))
    require(a.shape == b.shape == reference.shape, f"{name}: dimensions differ")
    errors = []
    rounding = 1024*np.finfo(float).eps*float(np.linalg.norm(reference))
    for components in (slice(0, 3), slice(3, 6)):
        error = float(np.linalg.norm((a-b)[:, components]))
        scale = float(np.linalg.norm(reference[:, components]))
        bound = 3e-9*scale+rounding
        require(error <= bound, f"{name}: complete A/P reaction replacement differs")
        errors.append(error/max(scale, rounding, np.finfo(float).tiny))
    return errors


def frame(surface):
    edges = surface.patch_edge_indices
    return np.stack((surface.edge_normals_xyz[edges],
        surface.edge_tangent1_xyz[edges], surface.edge_tangent2_xyz[edges]), axis=-1)


def seed_packet(seed):
    body = {item.name: (plain({f.name: getattr(seed.law, f.name) for f in fields(seed.law)})
                       if item.name == "law" else plain(getattr(seed, item.name)))
            for item in fields(seed)}
    return {**body, "seed_sha256": digest(canonical(body))}


def make_history(surface, seed, parent_checkpoint_sha256):
    return dict(parent_checkpoint_sha256=parent_checkpoint_sha256,
        surface_binding_sha256=surface.binding_sha256,
        seed_sha256=seed_packet(seed)["seed_sha256"],
        bank_displacement_from_fixture_origin_m=np.zeros(surface.banks.ndof),
        local_gap_m=np.zeros(surface.area_weights_m2.shape+(3,)),
        maximum_separation_m=seed.initial_maximum_separation_m.copy(),
        stored_interface_energy_j=0., cumulative_fracture_work_j=0.,
        cumulative_cohesive_work_j=0., accepted_path_segments=[])


def step_history(surface, seed, previous, bank_u):
    raw = np.asarray(bank_u)
    require(raw.dtype.kind in "ifu" and raw.shape == (surface.banks.ndof,)
            and np.isfinite(raw).all(), "Fixture history requires a complete finite metre vector")
    u = np.asarray(raw, float)
    world_gap = evaluate_surface_jump(surface, u)
    axes = frame(surface)
    local_gap = np.einsum("pji,pqj->pqi", axes, world_gap)
    local = evaluate_prestress_creation(seed, previous["local_gap_m"], local_gap,
                                      previous["maximum_separation_m"])
    world_traction = np.einsum("pij,pqj->pqi", axes, local.traction_pa)
    bank_force = integrate_surface_traction(surface, world_traction)
    weights = surface.area_weights_m2
    energy = float(np.sum(weights*local.stored_energy_j_m2))
    dissipation = float(np.sum(weights*local.fracture_dissipation_j_m2))
    increment = float(np.sum(weights*local.fracture_dissipation_increment_j_m2))
    work = float(np.sum(weights*local.mechanical_work_j_m2))
    total_work = previous["cumulative_cohesive_work_j"]+work
    close(work, energy-previous["stored_interface_energy_j"]+increment,
          "Seeded interface incremental work", atol=4e3)
    close(total_work, energy+dissipation, "Seeded interface cumulative work", atol=4e3)
    close(previous["cumulative_fracture_work_j"]+increment, dissipation,
          "Seeded interface single irreversible debit", atol=4e3)
    require(np.all(local.maximum_separation_m >= previous["maximum_separation_m"]),
            "Seeded maximum history healed")
    require(increment >= -1e-8, "Seeded irreversible work decreased")
    current = {**previous,
        "bank_displacement_from_fixture_origin_m": u.copy(), "local_gap_m": local_gap,
        "maximum_separation_m": local.maximum_separation_m,
        "stored_interface_energy_j": energy, "cumulative_fracture_work_j": dissipation,
        "cumulative_cohesive_work_j": total_work,
        "accepted_path_segments": previous["accepted_path_segments"]+[u.tolist()]}
    response = dict(local=local, world_gap=world_gap, world_traction=world_traction,
        bank_force=bank_force, energy=energy, dissipation=dissipation,
        increment=increment, work=work)
    return current, response


def history_to_dict(history):
    body = dict(format=HISTORY_FORMAT, scope=SCOPE, **plain(history),
        stored_energy_semantics="signed_change_from_prestressed_reference_not_new_historical_energy",
        physical_creation_admitted=False, physical_release_admitted=False,
        normal_contact_admitted=False)
    return {**body, "state_sha256": digest(canonical(body))}


def history_from_dict(raw, *, surface, seed, parent_checkpoint_sha256):
    require(type(raw) is dict and type(raw.get("accepted_path_segments")) is list,
            "Computational insertion history schema differs")
    require(raw.get("state_sha256") == digest(canonical({k: v for k, v in raw.items()
                                                        if k != "state_sha256"})),
            "Computational insertion history seal differs")
    current = make_history(surface, seed, parent_checkpoint_sha256)
    for vector in raw["accepted_path_segments"]:
        require(type(vector) is list and all(type(x) in (int, float) for x in vector),
                "Computational insertion path needs literal real values")
        current, _ = step_history(surface, seed, current, vector)
    require(canonical(history_to_dict(current)) == canonical(raw),
            "Computational insertion history differs from full path replay")
    return current


def prescribed_paths(surface):
    if not surface.patch_count:
        return [("empty_surface_zero_probe", np.zeros(surface.banks.ndof))]
    n, t1, t2 = (surface.edge_normals_xyz[0], surface.edge_tangent1_xyz[0],
                 surface.edge_tangent2_xyz[0])
    specs = (
        ("zero_gap_seeded_birth", 0., 0., 0.),
        ("elastic_opening_with_both_shears", .00002, .000006, .000004),
        ("mixed_mode_softening", .007, .005, .003),
        ("partial_unloading", .0025, .0009, .0004),
        ("rotation_during_unloading", .0010, -.0035, .0020),
        ("compression_before_complete_separation", -.0050, .0010, -.0012),
        ("reload_through_complete_separation", .025, .009, .004),
        ("large_rotation_after_separation", .010, -.035, .020),
        ("large_compression_with_both_shears", -.050, .010, -.012),
        ("complete_mixed_mode_separation", 6.0, 4.0, 3.0),
        ("compression_after_separation", -.070, -.020, .010),
    )
    original_vertices = surface.banks.parent_assembly.ndof//6
    depth = 1.-float(np.min(surface.lambda_bounds[:, 0]))
    require(np.isfinite(depth) and depth > 0,
            "A nonempty prospective surface must have positive reference depth")
    paths = []
    for name, gn, gt1, gt2 in specs:
        u = np.zeros((surface.banks.ndof//6, 6))
        u[original_vertices:, :3] = gn*n+gt1*t1+gt2*t2
        u[original_vertices:, 3:] = .18*(gt2*n+gn*t1+gt1*t2)/depth
        paths.append((name, u.ravel()))
    return paths


def path_packet(name, previous, current, response):
    local = response["local"]
    return dict(name=name,
        old_bank_displacement_from_fixture_origin_m=previous["bank_displacement_from_fixture_origin_m"].tolist(),
        bank_displacement_from_fixture_origin_m=current["bank_displacement_from_fixture_origin_m"].tolist(),
        old_kappa_m=previous["maximum_separation_m"].tolist(),
        world_jump_m=response["world_gap"].tolist(), local_jump_m=current["local_gap_m"].tolist(),
        kappa_m=local.maximum_separation_m.tolist(), damage=local.damage.tolist(),
        traction_local_pa=local.traction_pa.tolist(), tangent_local_pa_m=local.tangent_pa_m.tolist(),
        traction_world_pa=response["world_traction"].tolist(), bank_force_n=response["bank_force"].tolist(),
        stored_energy_density_j_m2=local.stored_energy_j_m2.tolist(),
        cumulative_dissipation_density_j_m2=local.fracture_dissipation_j_m2.tolist(),
        dissipation_increment_density_j_m2=local.fracture_dissipation_increment_j_m2.tolist(),
        mechanical_work_increment_density_j_m2=local.mechanical_work_j_m2.tolist(),
        stored_energy_j=response["energy"], cumulative_dissipation_j=response["dissipation"],
        dissipation_increment_j=response["increment"], mechanical_work_increment_j=response["work"])


def run(parent_path, output):
    output.mkdir(parents=True, exist_ok=False)
    parent_bytes = parent_path.read_bytes()
    require(digest(parent_bytes) == PARENT_RAW_SHA256,
            "Insertion control requires the sealed accepted 0.8684 Myr parent")
    parent_packet = json.loads(parent_bytes)
    require(parent_packet["checkpoint_sha256"] == PARENT_CHECKPOINT_SHA256,
            "Insertion accepted parent semantic pin differs")
    model, state = load_volume_slip_checkpoint(parent_path,
        expected_source_checkpoint_sha256=FORMATION_SHA256)
    require(len(state.geometry.cohorts) == 800 and abs(state.time_myr-.8684) < 1e-14
            and len(state.accepted_step_sizes_myr) == 4,
            "Insertion accepted clock or material ancestry differs")
    require(canonical(volume_slip_checkpoint_to_dict(model, state)) == canonical(parent_packet),
            "Full accepted parent replay changed its archive")
    actual_audit = audit_state(model, state)
    save_volume_slip_checkpoint(output/"parent_before.json", model, state)
    require((output/"parent_before.json").read_bytes() == parent_bytes,
            "Restored insertion parent is not byte-identical")
    onset = bind_calculated_volume_fracture_onset(model, state)
    onset_payload = calculated_volume_fracture_onset_to_dict(onset)
    calculated_volume_fracture_onset_from_dict(onset_payload, model=model, state=state)
    require(onset_payload["new_activation_event"] is None
            and onset_payload["newly_activating_faces"] == []
            and onset_payload["existing_active_faces"] == [],
            "Actual accepted source unexpectedly calculated an activation event")
    require(onset_payload["time_myr"] == state.time_myr
            and onset_payload["event_evaluation_scope"] == "next_interval_start_at_current_accepted_state",
            "Onset witness confused current-start evaluation with a previous event")
    (output/"actual_onset_prerequisite.json").write_bytes(canonical(onset_payload)+b"\n")
    actual_context = creation_context_to_dict(bind_volume_creation_context(model, state))
    require(actual_context["physical_cut_status"] == "no_eligible_cut"
            and actual_context["source_calculated_creation_law"] is None
            and actual_context["declared_prospective_creation_law"] is None,
            "Actual uncut source acquired an invented creation law or eligibility")

    # Preserve the original explicit law and its typed overstrength refusal.
    # The separate stronger numerical law tests force/work assembly only; it
    # is an explicitly declared mechanical example, not source calibration.
    original_law = ProspectiveCreationLaw(fracture_energy_j_m2=1e8, peak_traction_pa=1e8,
        cohesive_stiffness_pa_m=1e10, compression_stiffness_pa_m=2e10)
    law = ProspectiveCreationLaw(fracture_energy_j_m2=1e8, peak_traction_pa=1e10,
        cohesive_stiffness_pa_m=1e14, compression_stiffness_pa_m=2e14)
    original_law_dto = {item.name: getattr(original_law, item.name) for item in fields(original_law)}
    law_dto = {item.name: getattr(law, item.name) for item in fields(law)}
    law_provenance = "computational_only_after_declared_overstrength_refusal"
    parent, basal = model._assembly(state.geometry, state.fault_history.damage)
    faces = np.asarray(state.geometry.faces, int)
    tri = faces[0]
    cycle = [(int(tri[a]), int(tri[b])) for a, b in ((0, 1), (1, 2), (2, 0))]
    cases, summaries = [], []
    for name, edges in (("empty_reference_support", []),
                        ("supplied_closed_first_face_reference_cycle", cycle)):
        banks = prepare_volume_banks(parent, edges, basal=basal)
        surface = prepare_volume_fracture_surface(banks)
        reaction = prepare_volume_bank_reaction(model, state, banks)
        traction = prepare_volume_reaction_traction(reaction, surface)
        reaction_dto = volume_bank_reaction_to_dict(reaction)
        traction_dto = volume_reaction_traction_to_dict(traction)
        validate_volume_bank_reaction_payload(reaction_dto, expected_reaction=reaction)
        validate_volume_reaction_traction_payload(traction_dto, expected_traction=traction)
        initial_local = np.einsum("pji,pqj->pqi", frame(surface), traction.world_traction_pa)
        positive_traction = initial_local.copy()
        positive_traction[..., 0] = np.maximum(positive_traction[..., 0], 0.)
        maximum_traction = float(np.max(np.linalg.norm(positive_traction, axis=-1), initial=0.))
        try:
            prepare_prestress_creation(original_law, initial_local)
        except PrestressCreationRefusal as error:
            require(surface.patch_count > 0 and maximum_traction > original_law.peak_traction_pa,
                    "Original prospective seed failed for a reason other than declared overstrength")
            original_refusal_status = "typed_virgin_overstrength_refusal"
            original_exception_type, original_reason = type(error).__name__, str(error)
        else:
            require(surface.patch_count == 0,
                    "Actual nonempty reference unexpectedly admitted the original low-strength virgin law")
            original_refusal_status = "empty_surface_no_traction_to_refuse"
            original_exception_type = original_reason = None
        original_refusal = dict(status=original_refusal_status,
            declared_law=original_law_dto, largest_initial_cohesive_traction_pa=maximum_traction,
            declared_peak_traction_pa=original_law.peak_traction_pa,
            exception_type=original_exception_type, reason=original_reason,
            original_parameters_retained=True, actual_initial_traction_unchanged=True,
            law_calibrated_from_Genesis=False,
            physical_creation_admitted=False, physical_release_admitted=False)
        seed = prepare_prestress_creation(law, initial_local)
        seed_dto = seed_packet(seed)
        history = make_history(surface, seed, parent_packet["checkpoint_sha256"])
        initial_history = history_to_dict(history)
        (output/f"{name}_initial.json").write_bytes(canonical(initial_history)+b"\n")
        zero_state, zero = step_history(surface, seed, history, np.zeros(banks.ndof))
        close(zero["local"].traction_pa, initial_local, "Zero-gap seeded traction", atol=1e-6)
        close(zero["world_traction"], traction.world_traction_pa,
              "Zero-gap seeded world traction", atol=1e-6)
        birth_force_errors = force_identity(zero["bank_force"], reaction.constraint_force_n,
              reaction.constraint_force_n, "Zero-gap seeded replacement force")
        replacement_residual = reaction.bank_residual_n+zero["bank_force"]
        birth_residual_errors = force_identity(replacement_residual, reaction.remaining_residual_n,
              reaction.constraint_force_n, "Single reaction replacement preserves accepted residual")
        require(zero_state["stored_interface_energy_j"] == 0.
                and zero_state["cumulative_fracture_work_j"] == 0.
                and zero_state["cumulative_cohesive_work_j"] == 0.,
                "Zero-gap insertion invented reference energy or creation work")
        insertion = dict(bank_force_n=zero["bank_force"].tolist(),
            old_tied_residual_n=reaction.remaining_residual_n.tolist(),
            new_seeded_residual_n=replacement_residual.tolist(),
            bulk_plus_seeded_internal_force_n=(reaction.bank_average_internal_force_n+zero["bank_force"]).tolist(),
            replaces_constraint_force_once=True, retains_parent_residual=True,
            force_replacement_relative_errors_A_P=birth_force_errors,
            retained_residual_relative_errors_A_P=birth_residual_errors,
            relative_stored_interface_energy_j=0., fracture_work_j=0., mechanical_work_j=0.,
            existing_Genesis_memory_or_energy_reinitialized=False,
            physical_creation_admitted=False, physical_release_admitted=False)
        prescribed = prescribed_paths(surface)
        paths, midpoint = [], None
        mid_index = len(prescribed)//2
        for index, (path_name, u) in enumerate(prescribed):
            previous = history
            history, response = step_history(surface, seed, previous, u)
            paths.append(path_packet(path_name, previous, history, response))
            virtual = np.sin(.19*np.arange(banks.ndof)+.3)
            close(response["bank_force"]@virtual,
                np.einsum("pqa,pqa,pq->", response["world_traction"],
                    evaluate_surface_jump(surface, virtual), surface.area_weights_m2),
                "Seeded surface force-trace duality", atol=1e-3)
            if index == mid_index:
                midpoint = history_to_dict(history)
                (output/f"{name}_restart.json").write_bytes(canonical(midpoint)+b"\n")
        terminal = history_to_dict(history)
        (output/f"{name}_direct_terminal.json").write_bytes(canonical(terminal)+b"\n")
        require(midpoint is not None, "Insertion fixture omitted its restart point")
        resumed = history_from_dict(midpoint, surface=surface, seed=seed,
                                    parent_checkpoint_sha256=parent_packet["checkpoint_sha256"])
        for _, u in prescribed[mid_index+1:]:
            resumed, _ = step_history(surface, seed, resumed, u)
        resumed_terminal = history_to_dict(resumed)
        (output/f"{name}_resumed_terminal.json").write_bytes(canonical(resumed_terminal)+b"\n")
        require(canonical(resumed_terminal) == canonical(terminal),
                "Seeded direct and resumed complete fixture histories differ")
        replayed = history_from_dict(terminal, surface=surface, seed=seed,
                                     parent_checkpoint_sha256=parent_packet["checkpoint_sha256"])
        require(canonical(history_to_dict(replayed)) == canonical(terminal),
                "Seeded terminal history failed its full path replay")
        cases.append(dict(name=name, computational_fixture=True,
            cut_edges=banks.topology.cut_edges.tolist(), bank_ndof=banks.ndof,
            topology=dict(parent_vertex=banks.topology.parent_vertex.tolist(),
                          faces=banks.topology.mesh.faces.tolist()),
            geometry=surface_packet(surface), reaction=reaction_dto,
            reaction_traction=traction_dto,
            original_prospective_virgin_refusal=original_refusal,
            law=law_dto, law_provenance=law_provenance, law_calibrated_from_Genesis=False,
            seed=seed_dto, interface_replacement=insertion, paths=paths,
            terminal_state=terminal, direct_resumed_computational_state_byte_equal=True))
        summaries.append(dict(name=name, paired_patch_count=surface.patch_count,
            paired_interface_area_m2=surface.total_area_m2,
            canonical_jump_rows=banks.J.shape[0], reaction_traction_rank=traction.rank,
            reconstructed_force_relative_errors_A_P=list(traction.force_relative_error_A_P),
            reaction_trace_range_relative_errors_A_P=list(traction.range_relative_error_A_P),
            seeded_birth_force_replacement_relative_errors_A_P=birth_force_errors,
            seeded_birth_retained_residual_relative_errors_A_P=birth_residual_errors,
            largest_initial_cohesive_traction_pa=maximum_traction,
            original_prospective_virgin_refusal=original_refusal,
            original_declared_peak_traction_pa=original_law.peak_traction_pa,
            computational_work_fixture_peak_traction_pa=law.peak_traction_pa,
            law_provenance=law_provenance, law_calibrated_from_Genesis=False,
            seeded_zero_gap_work_j=0., reference_energy_debited_from_Genesis_j=0.,
            accepted_computational_path_segments=len(history["accepted_path_segments"]),
            terminal_signed_stored_interface_energy_j=history["stored_interface_energy_j"],
            terminal_cumulative_fracture_work_j=history["cumulative_fracture_work_j"],
            terminal_cumulative_cohesive_work_j=history["cumulative_cohesive_work_j"],
            direct_resumed_computational_state_byte_equal=True,
            physical_creation_admitted=False, physical_release_admitted=False,
            normal_contact_admitted=False))
    fixture = dict(format=FORMAT, scope=SCOPE, parent_raw_sha256=PARENT_RAW_SHA256,
        parent_path=str(parent_path.resolve()), parent_checkpoint=parent_packet,
        actual_onset_prerequisite=onset_payload, actual_creation_context=actual_context,
        original_declared_prospective_law=original_law_dto,
        computational_work_fixture_law=law_dto,
        computational_work_fixture_law_provenance=law_provenance,
        fixture_cases=cases,
        traction_scope="last_accepted_interval_average_not_instantaneous_birth_stress",
        bank_displacement_semantics="from_zero_computational_fixture_origin_not_absolute_accepted_Genesis_pose",
        no_new_Genesis_time_steps=True, no_physical_fracture_or_contact_admission=True,
        no_source_creation_law_calibration_claim=True,
        parent_residual_preserved_without_forcing_it_to_zero=True)
    fixture["fixture_sha256"] = digest(canonical(fixture))
    (output/"control_fixture.json").write_bytes(canonical(fixture)+b"\n")
    save_volume_slip_checkpoint(output/"parent_after.json", model, state)
    require((output/"parent_after.json").read_bytes() == parent_bytes
            and parent_path.read_bytes() == parent_bytes,
            "Insertion probes changed the accepted parent or its immutable archive")
    report = dict(format="actual-genesis-volume-prestress-insertion-control-0.1",
        status="PASS", scope=SCOPE, parent_raw_sha256=PARENT_RAW_SHA256,
        parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
        formation_checkpoint_sha256=FORMATION_SHA256, time_myr=state.time_myr,
        cohort_count=len(state.geometry.cohorts), accepted_Genesis_intervals_replayed=4,
        parent_replayed_once_from_formation_checkpoint=True,
        parent_before_after_entire_packets_byte_equal=True,
        no_replacement_initialization=True, new_Genesis_time_steps=0,
        actual_active_weak_plane_count=0, actual_new_activation_event=None,
        actual_surface_creation_law_unknown=True,
        source_calculated_onset_sha256=onset_payload["onset_sha256"],
        actual_state_audit=actual_audit, fixture_sha256=fixture["fixture_sha256"],
        fixture_cases=summaries,
        original_declared_prospective_law=original_law_dto,
        computational_work_fixture_law=law_dto,
        computational_work_fixture_law_provenance=law_provenance,
        sharp_physical_banks_created=False, ties_released=False,
        actual_creation_work_assigned=False,
        physical_banks_contact_underthrust_admitted=False, point_one_complete=False,
        prestress_seed_is_computational_not_an_actual_inserted_interface=True,
        computational_restart_is_not_an_actual_active_fault_restart=True)
    report["report_sha256"] = digest(canonical(report))
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
