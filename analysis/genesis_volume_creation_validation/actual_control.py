"""Preserve the genuine Genesis parent and exercise prospective creation only.

The accepted 0.8682 Myr archive is replayed once from its literal formation
source. No Genesis time step, fault activation, cut selection or initialization
is performed. Supplied mesh cuts, bank motions and cohesive parameters are
explicit computational fixtures; they are not calibrated Genesis events.
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

from tectonics.genesis_volume_banks import prepare_volume_banks
from tectonics.genesis_volume_creation import (
    bind_volume_creation_context, creation_context_to_dict,
    make_computational_creation_state, step_computational_creation_state,
    creation_fixture_state_to_dict, creation_fixture_state_from_dict,
)
from tectonics.genesis_volume_creation_law import ProspectiveCreationLaw
from tectonics.genesis_volume_fracture_surface import (
    prepare_volume_fracture_surface, evaluate_surface_jump,
)
from tectonics.genesis_volume_slip_checkpoint import (
    load_volume_slip_checkpoint, save_volume_slip_checkpoint,
    volume_slip_checkpoint_to_dict,
)


PARENT_RAW_SHA256 = "34cc53c215b2e9b8300a8f47b33801d49182b611a3ad985217a9c4cef25f2aed"
FORMATION_SHA256 = "cd79ee2a5139c4b91abdb1b4714a761c9153cc9f4490f5aa63fa2facba458c89"
FORMAT = "genesis-volume-creation-control-0.1"
SCOPE = "source_bound_prospective_surface_and_creation_kernel_not_physical_genesis_fracture"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, name, *, rtol=5e-11, atol=1e-8):
    a, b = np.asarray(actual, float), np.asarray(expected, float)
    require(a.shape == b.shape, f"{name}: dimensions differ")
    scale = max(float(np.max(np.abs(b), initial=0.)), 1e-300)
    error = float(np.max(np.abs(a-b), initial=0.))
    require(error <= atol+rtol*scale, f"{name}: physical or work identity failed")
    return error/max(scale, atol, 1e-300)


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


def surface_packet(surface):
    return {item.name: plain(getattr(surface, item.name))
            for item in fields(surface) if item.name != "banks"}


def prescribed_paths(surface):
    """Full A/P probes from the zero fixture origin, with three local modes.

    Only appended duplicate vertices move. A and P are distinct, and the P
    scale makes its finite-depth contribution observable without changing any
    parent cohort, chart, normal or actual accepted Genesis displacement.
    """
    ndof = surface.banks.ndof
    if not surface.patch_count:
        return [("empty_surface_zero_probe", np.zeros(ndof))]
    n = surface.edge_normals_xyz[0]
    t1, t2 = surface.edge_tangent1_xyz[0], surface.edge_tangent2_xyz[0]
    specs = (
        ("zero_new_fixture_origin", 0., 0., 0.),
        ("elastic_opening_with_both_shears", .0010, .0003, .0002),
        ("mixed_mode_softening", .040, .025, .017),
        ("partial_unloading", .011, .006, .004),
        ("rotation_during_unloading", .006, -.014, .008),
        ("compression_with_both_shears", -.025, .009, -.005),
        ("complete_mixed_mode_separation", 4.0, 2.8, 1.9),
        ("compression_after_separation", -.030, -.010, .007),
    )
    parent_vertices = surface.banks.parent_assembly.ndof//6
    depth_scale = max(1.-float(np.min(surface.lambda_bounds[:, 0])), 1e-12)
    paths = []
    for name, normal, shear1, shear2 in specs:
        u = np.zeros((ndof//6, 6))
        u[parent_vertices:, :3] = normal*n+shear1*t1+shear2*t2
        u[parent_vertices:, 3:] = .18*(shear2*n+normal*t1+shear1*t2)/depth_scale
        paths.append((name, u.ravel()))
    return paths


def path_packet(name, surface, previous, current, response):
    local = response.local_response
    return dict(name=name,
        old_bank_displacement_from_fixture_origin_m=previous.bank_displacement_from_fixture_origin_m.tolist(),
        bank_displacement_from_fixture_origin_m=current.bank_displacement_from_fixture_origin_m.tolist(),
        old_kappa_m=previous.maximum_separation_m.tolist(),
        world_jump_m=evaluate_surface_jump(surface,
            current.bank_displacement_from_fixture_origin_m).tolist(),
        local_jump_m=current.local_gap_m.tolist(),
        kappa_m=local.maximum_separation_m.tolist(), damage=local.damage.tolist(),
        traction_local_pa=local.traction_pa.tolist(), tangent_local_pa_m=local.tangent_pa_m.tolist(),
        traction_world_pa=response.world_traction_pa.tolist(), bank_force_n=response.bank_force_n.tolist(),
        stored_energy_density_j_m2=local.stored_energy_j_m2.tolist(),
        cumulative_dissipation_density_j_m2=local.fracture_dissipation_j_m2.tolist(),
        dissipation_increment_density_j_m2=local.fracture_dissipation_increment_j_m2.tolist(),
        mechanical_work_increment_density_j_m2=local.mechanical_work_j_m2.tolist(),
        stored_energy_j=response.integrated_stored_energy_j,
        cumulative_dissipation_j=response.integrated_total_fracture_dissipation_j,
        dissipation_increment_j=response.integrated_fracture_increment_j,
        mechanical_work_increment_j=response.integrated_mechanical_work_j)


def run(parent_path, output):
    output.mkdir(parents=True, exist_ok=False)
    parent_bytes = parent_path.read_bytes()
    parent_raw_sha = hashlib.sha256(parent_bytes).hexdigest()
    require(parent_raw_sha == PARENT_RAW_SHA256,
            "Control requires the sealed accepted 0.8682 Myr parent")
    parent_packet = json.loads(parent_bytes)
    # Full semantic checkpoint replay is performed once, without a replacement
    # initial state or a new model.step(). Computational paths below never
    # mutate this model/state or its material history.
    model, state = load_volume_slip_checkpoint(parent_path,
        expected_source_checkpoint_sha256=FORMATION_SHA256)
    require(len(state.geometry.cohorts) == 640 and abs(state.time_myr-.8682) < 1e-14,
            "Accepted cohort catalogue or material clock differs")
    require(not state.fault_history.fault_active.any(),
            "The actual parent unexpectedly has an active fault")
    require(canonical(volume_slip_checkpoint_to_dict(model, state)) == canonical(parent_packet),
            "Semantic parent replay changed the accepted checkpoint")
    save_volume_slip_checkpoint(output/"parent_before.json", model, state)
    require((output/"parent_before.json").read_bytes() == parent_bytes,
            "Restored accepted checkpoint is not byte-identical")
    actual = bind_volume_creation_context(model, state)
    actual_payload = creation_context_to_dict(actual)
    require(actual_payload["physical_cut_status"] == "no_eligible_cut",
            "Actual zero-active-fault source acquired a physical cut")
    require(not actual_payload["physical_creation_admitted"]
            and not actual_payload["physical_release_admitted"],
            "Actual source admitted uncalculated fracture creation or release")
    require(actual_payload["source_calculated_creation_law"] is None
            and actual_payload["declared_prospective_creation_law"] is None,
            "Actual unknown creation law was replaced by fixture parameters")
    require(actual_payload["creation_work_debited_from_Genesis_j"] == 0.
            and actual_payload["historical_dissipation_available_for_creation_j"] == 0.
            and not actual_payload["existing_damage_or_slip_work_reclassified"],
            "Historical Genesis loss was reused as new surface-creation credit")
    (output/"actual_creation_context.json").write_bytes(canonical(actual_payload)+b"\n")

    # These explicit numbers are research probes. Neither Gc nor the surface
    # location is obtained from Genesis, finite-width slip or legacy defaults.
    law = ProspectiveCreationLaw(fracture_energy_j_m2=1e6,
        peak_traction_pa=4e6, cohesive_stiffness_pa_m=1e9,
        compression_stiffness_pa_m=2e9)
    context = bind_volume_creation_context(model, state, law=law)
    computational_context = creation_context_to_dict(context)
    parent, basal = model._assembly(state.geometry, state.fault_history.damage)
    faces = np.asarray(state.geometry.faces, int)
    first = faces[0]
    cycle = [(int(first[a]), int(first[b])) for a, b in ((0, 1), (1, 2), (2, 0))]
    cases, reports = [], []
    for name, edges in (("empty_reference_support", []),
                        ("supplied_closed_first_face_reference_cycle", cycle)):
        banks = prepare_volume_banks(parent, edges, basal=basal)
        surface = prepare_volume_fracture_surface(banks)
        current = make_computational_creation_state(context, surface)
        initial = creation_fixture_state_to_dict(current)
        (output/f"{name}_initial.json").write_bytes(canonical(initial)+b"\n")
        prescribed = prescribed_paths(surface)
        mid_index = len(prescribed)//2
        paths, errors, restart_packet = [], [], None
        for index, (path_name, u) in enumerate(prescribed):
            previous = current
            current, response = step_computational_creation_state(context, surface, previous, u)
            record = path_packet(path_name, surface, previous, current, response)
            error = close(response.integrated_mechanical_work_j,
                current.stored_interface_energy_j-previous.stored_interface_energy_j
                    +response.integrated_fracture_increment_j,
                "Integrated creation work", atol=2e3)
            cumulative_error = close(current.cumulative_cohesive_work_j,
                current.stored_interface_energy_j+current.cumulative_fracture_work_j,
                "Cumulative fixture creation work", atol=2e3)
            virtual = np.sin(.13*np.arange(banks.ndof)+.2)
            virtual_jump = evaluate_surface_jump(surface, virtual)
            close(response.bank_force_n@virtual,
                np.einsum("pqa,pqa,pq->", response.world_traction_pa,
                    virtual_jump, surface.area_weights_m2),
                "Surface force and trace duality", atol=1e-3)
            require(np.all(current.maximum_separation_m >= previous.maximum_separation_m),
                    "Computational maximum separation memory decreased")
            require(response.integrated_fracture_increment_j >= -1e-8,
                    "Computational irreversible fracture work decreased")
            paths.append(record)
            errors.append(dict(name=path_name, work_relative_error=error,
                               cumulative_work_relative_error=cumulative_error))
            if index == mid_index:
                restart_packet = creation_fixture_state_to_dict(current)
                (output/f"{name}_restart.json").write_bytes(canonical(restart_packet)+b"\n")
        terminal = creation_fixture_state_to_dict(current)
        (output/f"{name}_direct_terminal.json").write_bytes(canonical(terminal)+b"\n")
        require(restart_packet is not None, "Computational path did not retain a restart point")
        resumed = creation_fixture_state_from_dict(restart_packet, context=context, surface=surface)
        require(canonical(creation_fixture_state_to_dict(resumed)) == canonical(restart_packet),
                "Computational semantic restart altered its accepted state")
        for _, u in prescribed[mid_index+1:]:
            resumed, _ = step_computational_creation_state(context, surface, resumed, u)
        resumed_terminal = creation_fixture_state_to_dict(resumed)
        (output/f"{name}_resumed_terminal.json").write_bytes(canonical(resumed_terminal)+b"\n")
        require(canonical(resumed_terminal) == canonical(terminal),
                "Direct and resumed computational fixture terminal states differ")
        # Also replay the entire terminal path, rather than trusting a digest.
        replayed = creation_fixture_state_from_dict(terminal, context=context, surface=surface)
        require(canonical(creation_fixture_state_to_dict(replayed)) == canonical(terminal),
                "Terminal computational state failed full semantic path replay")
        cases.append(dict(name=name, computational_fixture=True,
            cut_edges=banks.topology.cut_edges.tolist(), bank_ndof=banks.ndof,
            topology=dict(parent_vertex=banks.topology.parent_vertex.tolist(),
                          faces=banks.topology.mesh.faces.tolist()),
            geometry=surface_packet(surface),
            law={item.name: getattr(law, item.name) for item in fields(law)},
            paths=paths, terminal_state=terminal,
            direct_resumed_computational_state_byte_equal=True))
        reports.append(dict(name=name, paired_patch_count=surface.patch_count,
            unmatched_solid_side_patch_count=len(surface.unmatched_edge_indices),
            paired_interface_area_m2=surface.total_area_m2,
            surface_binding_sha256=surface.binding_sha256,
            accepted_computational_path_segments=len(current.accepted_path_segments),
            terminal_stored_interface_energy_j=current.stored_interface_energy_j,
            terminal_cumulative_fracture_work_j=current.cumulative_fracture_work_j,
            terminal_cumulative_cohesive_work_j=current.cumulative_cohesive_work_j,
            work_checks=errors, direct_resumed_computational_state_byte_equal=True,
            physical_creation_admitted=False, physical_release_admitted=False))
    fixture = dict(format=FORMAT, scope=SCOPE, parent_raw_sha256=parent_raw_sha,
        parent_path=str(parent_path.resolve()), parent_checkpoint=parent_packet,
        actual_creation_context=actual_payload,
        computational_creation_context=computational_context,
        fixture_cases=cases,
        bank_displacement_semantics="from_zero_computational_fixture_origin_not_absolute_accepted_Genesis_pose",
        cohesive_energy_measure="one_original_paired_interface_area_counts_both_created_banks",
        no_prestress_replacement_or_release_equilibrium_claim=True,
        no_physical_fracture_or_contact_admission=True)
    fixture["fixture_sha256"] = hashlib.sha256(canonical(fixture)).hexdigest()
    (output/"control_fixture.json").write_bytes(canonical(fixture)+b"\n")
    save_volume_slip_checkpoint(output/"parent_after.json", model, state)
    require((output/"parent_after.json").read_bytes() == parent_bytes,
            "Prospective computation changed the entire accepted parent archive")
    require(parent_path.read_bytes() == parent_bytes, "Immutable accepted parent was modified")
    report = dict(format="actual-genesis-volume-prospective-creation-control-0.1",
        status="PASS", scope=SCOPE, parent_raw_sha256=parent_raw_sha,
        parent_checkpoint_sha256=parent_packet["checkpoint_sha256"],
        formation_checkpoint_sha256=FORMATION_SHA256, time_myr=state.time_myr,
        cohort_count=len(state.geometry.cohorts), actual_active_weak_plane_count=0,
        source_replayed_once_from_formation_checkpoint=True,
        no_replacement_initialization=True, parent_before_after_entire_packets_byte_equal=True,
        new_Genesis_time_steps=0, transition_time_increment_myr=0.,
        actual_cut_status=actual_payload["physical_cut_status"],
        actual_source_creation_law_unknown=True,
        sharp_physical_banks_created=False, ties_released=False,
        actual_fracture_work_assigned=False,
        physical_banks_contact_underthrust_admitted=False, point_one_complete=False,
        fixture_sha256=fixture["fixture_sha256"], fixture_cases=reports,
        explicit_computational_law_is_not_Genesis_calibration=True,
        computational_restart_is_not_an_actual_active_fault_restart=True)
    report["report_sha256"] = hashlib.sha256(canonical(report)).hexdigest()
    (output/"control_report.json").write_bytes(canonical(report)+b"\n")
    print(json.dumps(report, sort_keys=True))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", type=Path, default=ROOT/
        "analysis/genesis_volume_slip_validation/evidence/actual_control/direct_after_second.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.parent_checkpoint, args.output)


if __name__ == "__main__":
    main()
