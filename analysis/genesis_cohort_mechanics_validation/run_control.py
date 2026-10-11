"""Trace a small Genesis control from its molten birth.

This records newly calculated depth material mechanics and validates the
Fault-to-Coupled path, not unrestricted volumetric mechanics or differentiated
continental material. No mature initial state is
constructed, and no thermal, phase, or mechanical guard is relaxed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields, is_dataclass, replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tectonics.genesis import diagnose, parameters_from_config
from tectonics.genesis_coupled import (CoupledModel, load_coupled_checkpoint,
                                      save_coupled_checkpoint)
from tectonics.genesis_fault_law import weak_plane_parameters_from_config
from tectonics.genesis_faults import FaultModel, save_fault_checkpoint, load_fault_checkpoint
from tectonics.genesis_bulk_birth import bulk_birth_to_dict, validate_bulk_birth_continuation
from tectonics.genesis_bulk_export import export_genesis_bulk_material, validate_genesis_bulk_material
from tectonics.genesis_cohort_mechanics import cohort_mechanics_to_dict
from tectonics.genesis_mobile import mobile_parameters_from_config
from tectonics.genesis_onset import onset_parameters_from_config
from tectonics.genesis_shell import (maximum_total_strain, rock_temperature,
                                    shell_parameters_from_config)
from tectonics.genesis_tides import tidal_parameters_from_config
from tectonics.simulation import load_config


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2,
                                   allow_nan=False)+"\n", encoding="utf-8")


def same(first, second, name="state"):
    """Compare complete state, including all dataclass histories and arrays."""
    if isinstance(first, np.ndarray):
        np.testing.assert_array_equal(first, second, err_msg=name)
    elif is_dataclass(first):
        assert type(first) is type(second), name
        for item in fields(first):
            same(getattr(first, item.name), getattr(second, item.name), name+"."+item.name)
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second), name
        for index, (left, right) in enumerate(zip(first, second)):
            same(left, right, f"{name}[{index}]")
    else:
        assert first == second, name


def cohort_summary(state):
    """State-derived mechanics facts, including an independently evaluated ledger."""
    memory = state.cohort_mechanics
    assert memory is not None, "The control starts from genuine molten formation"
    if state.bulk_birth_history.remelted_mass_kg == 0.:
        assert memory.remelt_energy_removed_j == 0., "No retired material can remove remelt energy"
    live = [c for c in memory.cohorts if c.elastic_strain is not None]
    by_face = {}
    for cohort in live:
        by_face.setdefault(cohort.face_index, []).append(cohort)
    temperature_contrast = 0.
    strain_contrast = 0.
    for cohorts in by_face.values():
        temperatures = [c.temperature_k for c in cohorts]
        strain = np.asarray([c.elastic_strain for c in cohorts], dtype=float)
        temperature_contrast = max(temperature_contrast, max(temperatures)-min(temperatures))
        strain_contrast = max(strain_contrast, float(np.max(np.ptp(strain, axis=0))))
    expected = (memory.mechanical_work_j-memory.plastic_work_j+memory.thermal_work_j
                -memory.viscous_dissipation_j+memory.damage_energy_change_j
                -memory.remelt_energy_removed_j)
    independent_residual = memory.elastic_energy_j-expected
    scale = max(1., abs(memory.elastic_energy_j), abs(memory.mechanical_work_j),
                abs(memory.thermal_work_j), memory.viscous_dissipation_j)
    assert np.isfinite(independent_residual)
    return {"established": memory.established,
            "live_mechanical_cohorts": len(memory.cohorts),
            "active_elastic_cohorts": len(live),
            "elastic_energy_j": memory.elastic_energy_j,
            "mechanical_work_j": memory.mechanical_work_j,
            "thermal_work_j": memory.thermal_work_j,
            "viscous_dissipation_j": memory.viscous_dissipation_j,
            "plastic_work_j": memory.plastic_work_j,
            "damage_energy_change_j": memory.damage_energy_change_j,
            "remelt_energy_removed_j": memory.remelt_energy_removed_j,
            "stored_balance_residual_j": memory.balance_residual_j,
            "independent_balance_residual_j": independent_residual,
            "relative_independent_balance_residual": abs(independent_residual)/scale,
            "max_same_face_temperature_contrast_k": temperature_contrast,
            "max_same_face_elastic_component_contrast": strain_contrast}


def sample(model, state, thermal, orbit):
    thermal_row, material, motion, _ = model.diagnostics(state, thermal, orbit)
    data = model.fields(state, thermal)
    return {"time_myr": state.time_myr,
            "surface_temperature_k": thermal_row["surface_temperature_k"],
            "mantle_temperature_k": thermal_row["mantle_temperature_k"],
            "steam_pressure_bar": thermal_row["steam_pressure_bar"],
            "ocean_fraction": thermal_row["ocean_fraction"],
            "ocean_mass_kg": thermal_row["ocean_mass_kg"],
            "surface_melt_fraction": thermal_row["surface_melt_fraction"],
            "min_solid_depth_km": float(np.min(data["lid_thickness_km"])),
            "max_solid_depth_km": float(np.max(data["lid_thickness_km"])),
            "membrane_established": state.membrane_established,
            "max_elastic_strain": maximum_total_strain(state.elastic_strain),
            "max_damage": float(np.max(state.damage)),
            "fault_count": int(np.count_nonzero(state.fault_active)),
            "max_fault_candidate_age_myr": float(np.max(state.fault_candidate_age_myr)),
            "max_velocity_km_myr": float(np.max(np.linalg.norm(state.velocity_km_myr, axis=1))),
            "mass_kg": float(np.sum(state.layer_mass_kg)),
            "relative_column_energy_residual": material["relative_column_energy_residual"],
            "relative_global_energy_residual": thermal_row["relative_energy_residual"],
            "orbit_heat_transfer_relative_residual": motion["orbit_heat_transfer_relative_residual"],
            "friction_work_j": state.friction_work_j,
            "viscous_fault_work_j": state.viscous_fault_work_j,
            "mechanical_equilibrium_residual": state.equilibrium_residual,
            "accepted_steps": state.accepted_steps,
            "rejected_steps": state.rejected_steps,
            "stopped_reason": state.stopped_reason,
            "thermal_stopped_reason": thermal.stopped_reason,
            "cohort_mechanics": cohort_summary(state)}


def run(output, *, traction_pa=20000., duration_myr=1.2, deadline_seconds=120.):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    config_path = ROOT/"configs/genesis_moon.yaml"
    config_bytes = config_path.read_bytes()
    (output/"source_config.yaml").write_bytes(config_bytes)
    config = load_config(config_path)
    source_hashes = {name: sha256(module.__file__) for name, module in sys.modules.copy().items()
                     if (name.startswith("tectonics.genesis") or name in {
                         "tectonics.mesh", "tectonics.basal_coupling", "tectonics.mantle_convection"})
                     and getattr(module, "__file__", None)}
    write_json(output/"physics_source_hashes.json", source_hashes)
    thermal_p = parameters_from_config(config)
    shell_p = replace(shell_parameters_from_config(config), subdivisions=1,
                      convective_traction_pa=traction_pa)
    model = FaultModel(shell_p, thermal_p, onset_parameters_from_config(config),
                       tidal_parameters_from_config(config, thermal_p),
                       mobile_parameters_from_config(config),
                       weak_plane_parameters_from_config(config))
    state, thermal, orbit = model.initial()
    initial_mass = state.layer_mass_kg.copy()
    initial = sample(model, state, thermal, orbit)
    assert state.time_myr == thermal.time_myr == orbit.time_myr == 0.
    assert initial["surface_melt_fraction"] == 1.
    assert initial["ocean_mass_kg"] == 0.
    assert initial["steam_pressure_bar"] > 100.
    assert np.min(rock_temperature(state.column_enthalpy, thermal_p)) > thermal_p.liquidus_k
    assert initial["max_solid_depth_km"] == 0.
    assert not state.membrane_established
    assert np.all(state.elastic_strain == 0.)
    assert np.all(state.damage == 0.) and not np.any(state.fault_active)
    parameters = {"thermal": asdict(thermal_p), "shell": asdict(shell_p),
                  "onset": asdict(model.onset_p), "tides": asdict(model.tides_p),
                  "mobile": asdict(model.mobile_p), "faults": asdict(model.fault_p)}
    controls = {"outer_step_myr": .002, "thermal_max_step_myr": .01,
                "max_duration_myr": duration_myr, "deadline_seconds": deadline_seconds,
                "fault_step_years": 1.,
                "coupled_step_years": 1.}
    provenance = {"source_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                  "source_config_path": str(config_path),
                  "explicit_overrides": {"genesis_shell.subdivisions": 1,
                                         "genesis_shell.convective_traction_pa": traction_pa},
                  "canonical_traction_pa": config["genesis_shell"]["convective_traction_pa"],
                  "initialization": "FaultModel.initial(), physically molten, no state replacement"}
    write_json(output/"parameters.json", {"parameters": parameters, "controls": controls,
                                          "provenance": provenance})
    assert state.bulk_birth_history.live_mass_kg == 0. and not state.bulk_birth_history.cohorts
    assert not state.cohort_mechanics.cohorts and not state.cohort_mechanics.established
    trace = [initial]
    partial = None
    while not state.membrane_established and state.time_myr < duration_myr:
        if time.monotonic()-started > deadline_seconds:
            break
        target = min(duration_myr, state.time_myr+controls["outer_step_myr"])
        state, thermal, orbit, _ = model.step(state, thermal, orbit, target,
                                             controls["thermal_max_step_myr"])
        np.testing.assert_array_equal(state.layer_mass_kg, initial_mass)
        assert state.time_myr == thermal.time_myr == orbit.time_myr
        row = sample(model, state, thermal, orbit)
        trace.append(row)
        if state.bulk_birth_history.cohorts and not state.membrane_established and partial is None:
            partial_path = output/"precursor_fault_checkpoint.npz"
            save_fault_checkpoint(partial_path, model, state, thermal, orbit, controls, provenance)
            partial = (partial_path, state.time_myr)
        if len(trace) % 25 == 0 or state.membrane_established or state.stopped_reason:
            print(json.dumps({"progress": row, "wall_seconds": time.monotonic()-started}), flush=True)
        if state.stopped_reason or thermal.stopped_reason:
            break
    write_json(output/"formation_trace.json", trace)
    fault_path = output/"born_fault_checkpoint.npz"
    save_fault_checkpoint(fault_path, model, state, thermal, orbit, controls, provenance)
    canonical_forcing = traction_pa == config["genesis_shell"]["convective_traction_pa"]
    control_scope = ("80-cell mesh control with canonical configured physics" if canonical_forcing
                     else "80-cell low-load control with explicit forcing override")
    report = {"format": "genesis-cohort-mechanics-continuation-control-0.1", "initial": initial,
              "fault_source": trace[-1], "source_fault_sha256": sha256(fault_path),
              "source_config_sha256": sha256(output/"source_config.yaml"),
              "provenance": provenance, "controls": controls,
              "control_scope": control_scope,
              "birth_to_coupled_verified": False,
              "limits": [control_scope+"; not a production-resolution world calculation",
                         "No continental chemical differentiation or volumetric/native-943 handover",
                         "Calculated cohort memory uses in-plane plane-stress Maxwell closure, not unrestricted historical 3D kinematics",
                         "Pre-membrane isolated solid rafts have no mechanical motion in this model",
                         "Inherited velocities are finite-difference diagnostics, not a new trajectory solver",
                         "Previous fault work is retained in embedded source, not copied into contact-work ledgers",
                         "Separate approximate mechanical, column and global thermal ledgers; no closed shared work/heat claim"]}
    if not state.membrane_established or state.stopped_reason or thermal.stopped_reason:
        report["result"] = "formation_not_admissible_for_coupled"
        report["wall_seconds"] = time.monotonic()-started
        write_json(output/"report.json", report)
        print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
        return 2
    assert partial is not None, "No actual solid precursor checkpoint was collected"
    restored_model, restored_state, restored_thermal, restored_orbit, _ = load_fault_checkpoint(partial[0])
    assert restored_state.bulk_birth_history.cohorts and not restored_state.membrane_established
    while restored_state.time_myr < state.time_myr:
        target = min(state.time_myr, restored_state.time_myr+controls["outer_step_myr"])
        restored_state, restored_thermal, restored_orbit, _ = restored_model.step(
            restored_state, restored_thermal, restored_orbit, target, controls["thermal_max_step_myr"])
    same((state, thermal, orbit), (restored_state, restored_thermal, restored_orbit))
    save_fault_checkpoint(output/"precursor_resumed_born_fault.npz", restored_model,
                          restored_state, restored_thermal, restored_orbit, controls, provenance)
    fault_target1 = state.time_myr+controls["fault_step_years"]/1e6
    fault_after1 = model.step(state, thermal, orbit, fault_target1, controls["thermal_max_step_myr"])
    assert fault_after1[0].stopped_reason is None and fault_after1[1].stopped_reason is None
    assert fault_after1[0].time_myr == fault_after1[1].time_myr == fault_after1[2].time_myr == fault_target1
    fault_restart_path = output/"fault_after.npz"
    save_fault_checkpoint(fault_restart_path, model, *fault_after1[:3], controls, provenance)
    fault_loaded, fault_restored, fault_heat, fault_tide, _ = load_fault_checkpoint(fault_restart_path)
    same(fault_after1[:3], (fault_restored, fault_heat, fault_tide))
    fault_target2 = fault_target1+controls["fault_step_years"]/1e6
    fault_direct = model.step(*fault_after1[:3], fault_target2, controls["thermal_max_step_myr"])
    fault_resumed = fault_loaded.step(fault_restored, fault_heat, fault_tide, fault_target2,
                                     controls["thermal_max_step_myr"])
    assert fault_direct[0].stopped_reason is None and fault_direct[1].stopped_reason is None
    assert fault_direct[0].time_myr == fault_direct[1].time_myr == fault_direct[2].time_myr == fault_target2
    same(fault_direct[:3], fault_resumed[:3])
    assert fault_direct[3] == fault_resumed[3]
    save_fault_checkpoint(output/"fault_direct_continued.npz", model,
                          *fault_direct[:3], controls, provenance)
    save_fault_checkpoint(output/"fault_restart_continued.npz", fault_loaded,
                          *fault_resumed[:3], controls, provenance)
    write_json(output/"cohort_mechanics_fault.json", cohort_mechanics_to_dict(state.cohort_mechanics))
    write_json(output/"cohort_mechanics_fault_continued.json",
               cohort_mechanics_to_dict(fault_resumed[0].cohort_mechanics))
    packet = export_genesis_bulk_material(fault_path.read_bytes(),
        {"purpose": "actual molten formation to calculated cohort mechanics export", "source_config_sha256": provenance["source_config_sha256"]})
    validated_packet = validate_genesis_bulk_material(packet,
        expected_source_checkpoint_sha256=sha256(fault_path))
    assert validated_packet == packet
    exported_mean_energy = packet["instantaneous_plane_stress_3d"]["elastic_energy_of_face_mean_j"]
    exported_cohort_energy = packet["instantaneous_cohort_plane_stress_3d"]["elastic_energy_j"]
    np.testing.assert_allclose(exported_cohort_energy, state.cohort_mechanics.elastic_energy_j,
                               rtol=2e-12, atol=1e-6)
    write_json(output/"material_export.json", packet)
    write_json(output/"bulk_history_fault.json", bulk_birth_to_dict(state.bulk_birth_history))
    assert not packet["capabilities"]["physical_force_work_admission"]
    assert all(value is None for value in packet["unknown_fields"].values())
    coupled, before, heat, tide = CoupledModel.from_fault_checkpoint(fault_path)
    same(before.bulk_birth_history, state.bulk_birth_history)
    same(before.cohort_mechanics, state.cohort_mechanics)
    assert heat.time_myr == tide.time_myr == before.time_myr == state.time_myr
    assert coupled.source_bytes == fault_path.read_bytes()
    for key in ("column_enthalpy", "elastic_strain", "damage", "water_access",
                "fault_active", "plane_normal", "fault_candidate_age_myr", "velocity_km_myr"):
        np.testing.assert_array_equal(getattr(before, key), getattr(state, key))
    assert not np.shares_memory(before.velocity_km_myr, coupled.source.source_state.velocity_km_myr)
    np.testing.assert_array_equal(coupled.layer_mass_kg, initial_mass)
    save_coupled_checkpoint(output/"coupled_before.npz", coupled, before, heat, tide)
    first_target = before.time_myr+1./1e6
    after, heat1, tide1, rows1 = coupled.step(before, heat, tide, first_target)
    assert after.stopped_reason is None and heat1.stopped_reason is None
    assert after.time_myr == heat1.time_myr == tide1.time_myr == first_target
    assert after.accepted_steps > before.accepted_steps
    np.testing.assert_array_equal(coupled.layer_mass_kg, initial_mass)
    after_path = output/"coupled_after.npz"
    save_coupled_checkpoint(after_path, coupled, after, heat1, tide1)
    loaded, restored, restored_heat, restored_tide = load_coupled_checkpoint(after_path)
    same((after, heat1, tide1), (restored, restored_heat, restored_tide))
    second_target = after.time_myr+1./1e6
    direct = coupled.step(after, heat1, tide1, second_target)
    resumed = loaded.step(restored, restored_heat, restored_tide, second_target)
    assert direct[0].stopped_reason is None and direct[1].stopped_reason is None
    assert direct[0].time_myr == direct[1].time_myr == direct[2].time_myr == second_target
    same(direct[:3], resumed[:3])
    assert direct[3] == resumed[3]
    save_coupled_checkpoint(output/"coupled_direct_continued.npz", coupled, *direct[:3])
    save_coupled_checkpoint(output/"coupled_restart_continued.npz", loaded, *resumed[:3])
    validate_bulk_birth_continuation(resumed[0].bulk_birth_history, state.bulk_birth_history)
    write_json(output/"bulk_history_coupled.json", bulk_birth_to_dict(resumed[0].bulk_birth_history))
    write_json(output/"cohort_mechanics_coupled.json",
               cohort_mechanics_to_dict(resumed[0].cohort_mechanics))
    current_hashes = {name: sha256(sys.modules[name].__file__) for name in source_hashes}
    assert current_hashes == source_hashes, "Physics source bytes changed during the control"
    report.update(result="verified_real_genesis_cohort_mechanics_continuation",
                  birth_precursor_restart_exact=True, birth_precursor_time_myr=partial[1],
                  bulk_source_identity=state.bulk_birth_history.source_identity,
                  bulk_cohort_count=len(state.bulk_birth_history.cohorts),
                  bulk_live_mass_kg=state.bulk_birth_history.live_mass_kg,
                  bulk_remelted_mass_kg=state.bulk_birth_history.remelted_mass_kg,
                  source_accepted_birth_intervals=state.bulk_birth_history.accepted_intervals,
                  final_accepted_birth_intervals=resumed[0].bulk_birth_history.accepted_intervals,
                  bulk_depth_birth_history_transferred_exact=True,
                  strict_material_export_readback_exact=True,
                  physical_force_work_admission=False, unrestricted_depth_3d_history_calculated=False,
                  prospective_cohort_maxwell_memory_calculated=True,
                  fault_one_plus_one_year_restart_exact=True,
                  fault_continuation=sample(fault_loaded, *fault_resumed[:3]),
                  source_cohort_mechanics=cohort_summary(state),
                  source_membrane_formation_elastic_energy_j=state.cohort_mechanics.elastic_energy_j,
                  exported_source_face_mean_elastic_energy_j=exported_mean_energy,
                  exported_source_cohort_elastic_energy_j=exported_cohort_energy,
                  exported_source_within_face_strain_variance_energy_j=(
                      exported_cohort_energy-exported_mean_energy),
                  final_coupled_cohort_mechanics=cohort_summary(resumed[0]),
                  inherited_cohort_mechanics_exact=True,
                  material_export_sha256=sha256(output/"material_export.json"),
                  birth_to_coupled_verified=True, exact_restart_verified=True,
                  inherited_material_fields_exact=True, inherited_source_bytes_exact=True,
                  inherited_velocity_exact=True, inherited_velocity_independent_copy=True,
                  inherited_fault_velocity_km_myr_max=float(np.max(np.linalg.norm(state.velocity_km_myr, axis=1))),
                  initial_coupled_velocity_km_myr_max=float(np.max(np.linalg.norm(before.velocity_km_myr, axis=1))),
                  first_coupled_step=rows1[-1], restart_continuation_step=resumed[3][-1],
                  coupled_after_sha256=sha256(after_path),
                  physics_sources_unchanged=True,
                  inherited_fault_work_j={"friction": state.friction_work_j,
                                          "viscous": state.viscous_fault_work_j},
                  wall_seconds=time.monotonic()-started)
    write_json(output/"report.json", report)
    print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--traction-pa", type=float, default=20000.)
    parser.add_argument("--duration-myr", type=float, default=1.2)
    parser.add_argument("--deadline-seconds", type=float, default=120.)
    args = parser.parse_args()
    return run(args.output, traction_pa=args.traction_pa,
               duration_myr=args.duration_myr, deadline_seconds=args.deadline_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
