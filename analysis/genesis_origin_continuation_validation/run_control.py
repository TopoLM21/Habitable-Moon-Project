"""Trace a small Genesis control from its molten birth.

This validates the existing Fault-to-Coupled path, not the new volumetric
mechanics or differentiated continental material. No mature initial state is
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
from tectonics.genesis_faults import FaultModel, save_fault_checkpoint
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
            "thermal_stopped_reason": thermal.stopped_reason}


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
                "coupled_step_years": 1.}
    provenance = {"source_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                  "source_config_path": str(config_path),
                  "explicit_overrides": {"genesis_shell.subdivisions": 1,
                                         "genesis_shell.convective_traction_pa": traction_pa},
                  "canonical_traction_pa": config["genesis_shell"]["convective_traction_pa"],
                  "initialization": "FaultModel.initial(), physically molten, no state replacement"}
    write_json(output/"parameters.json", {"parameters": parameters, "controls": controls,
                                          "provenance": provenance})
    trace = [initial]
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
    report = {"format": "genesis-origin-continuation-control-0.1", "initial": initial,
              "fault_source": trace[-1], "source_fault_sha256": sha256(fault_path),
              "source_config_sha256": sha256(output/"source_config.yaml"),
              "provenance": provenance, "controls": controls,
              "control_scope": control_scope,
              "birth_to_coupled_verified": False,
              "limits": [control_scope+"; not a production-resolution world calculation",
                         "No continental chemical differentiation or volumetric/native-943 handover",
                         "Bulk memory is the existing depth-averaged in-plane Maxwell model",
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
    coupled, before, heat, tide = CoupledModel.from_fault_checkpoint(fault_path)
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
    save_coupled_checkpoint(output/"coupled_restart_continued.npz", loaded, *resumed[:3])
    current_hashes = {name: sha256(sys.modules[name].__file__) for name in source_hashes}
    assert current_hashes == source_hashes, "Physics source bytes changed during the control"
    report.update(result="verified_existing_genesis_continuation",
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
