"""Moving, continuously tied shell from the physical thermal gate to onset.

The material support is prescribed from the earlier diagnostic; its nodes,
connectivity and child mass fractions are transported rather than reinserted.
Mechanics uses finite corotation and current-geometry force balance on every
physical step. No activated contact is moved by this owner. A detected local
birth is saved as a separate current-reference snapshot, then this run stops.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.genesis_path_dynamics_validation import SOURCE, TRACE, _hash, _exact
from analysis.genesis_path_precursor_validation import build_case
from tectonics.genesis import GenesisState
from tectonics.genesis_coupled_thermal import advance_thermal_loading, evolve_coupled_damage
from tectonics.genesis_material import face_deformation, polar_increment
from tectonics.genesis_mobile import _external_force
from tectonics.genesis_moving_support import MaterialPathSupport
from tectonics.genesis_moving_tied import MovingTiedLoading, MovingTiedMechanics, MovingTiedRetry
from tectonics.genesis_moving_path_diagnostics import observe_material_path
from tectonics.genesis_path_activation import activate_tied_onset, load_born_path
from tectonics.genesis_shell import mantle_traction, maximum_total_strain
from tectonics.genesis_tides import TidalOrbitState

PRE_GATE = ROOT/"results/genesis_runs/path_precursor_current250_20260926/pre_gate_fault_checkpoint.npz"


@dataclass
class MovingThermalContext:
    thermal: GenesisState
    orbit: TidalOrbitState
    column_enthalpy: np.ndarray
    damage: np.ndarray
    water_access: np.ndarray
    boundary_energy_j: float
    tidal_heat_received_j: float


class MovingGenesisCase:
    """Bounded analysis owner; no changed bulk damage or active-interface growth."""

    def __init__(self, precursor):
        self.source_model = precursor.source_model
        self.mass = precursor.before.layer_mass_kg.copy()
        self.mass.setflags(write=False)
        self.initial_column_energy_j = precursor.before.initial_column_energy_j
        self.model = MovingTiedMechanics(precursor.mesh, precursor.radius_m,
            parameters=precursor.model.parameters, poisson_ratio=self.source_model.p.poisson_ratio)
        self.support = MaterialPathSupport(precursor.mesh, precursor.basis.insertion,
            precursor.radius_m, self.source_model.p.poisson_ratio)
        self.initial = self.model.initial(precursor.before.elastic_strain)
        c = precursor.context
        self.context = MovingThermalContext(deepcopy(c.thermal), deepcopy(c.orbit),
            c.column_enthalpy.copy(), c.damage.copy(), c.water_access.copy(),
            c.boundary_energy_j, 0.)
        self.gate_age_myr = c.thermal.time_myr

    def force(self, mesh, radius_m, depth_m, membrane):
        p = self.source_model.p
        traction = mantle_traction(mesh, p, depth_m/1000.)
        return _external_force(membrane, traction, radius_m/1000., p.young_modulus_pa)*(p.young_modulus_pa*radius_m*1000.)

    def trial(self, state, context, years):
        """Every thermal and mechanical probe starts from one accepted pair."""
        expected_age = self.gate_age_myr+state.elapsed_years/1e6
        clock_tolerance = 128*np.finfo(float).eps*max(abs(expected_age), 1.)
        if (context.thermal.time_myr != context.orbit.time_myr
                or abs(context.thermal.time_myr-expected_age) > clock_tolerance):
            raise ValueError("Moving mechanics and thermal/orbit clocks must form one accepted pair")
        mesh = self.model.mesh_for(state)
        p = self.source_model.p
        thermal = advance_thermal_loading(self.source_model, mesh=mesh,
            radius_km=state.radius_m/1000., layer_mass_kg=self.mass,
            column_enthalpy=context.column_enthalpy, elastic_strain=state.elastic_strain,
            damage=context.damage, water_access=context.water_access,
            boundary_energy_j=context.boundary_energy_j,
            thermal_state=context.thermal, orbit=context.orbit,
            target_myr=context.thermal.time_myr+years/1e6)
        if abs(thermal.dt_myr*1e6-years) > max(2e-9, years*1e-9):
            raise ValueError("Thermal and requested mechanics step disagree")
        volume = self.mass.sum(axis=1)*thermal.fraction/p.density_kg_m3
        young = p.young_modulus_pa*(p.residual_stiffness
            +(1-p.residual_stiffness)*(1-thermal.damage0)**2)
        loading = MovingTiedLoading(years, volume, young, thermal.memory,
            thermal.effective_b, self.force)
        candidate = self.model.trial(state, loading)
        moved = self.model.mesh_for(candidate)
        stress = self.model.stress(candidate)
        proposed_damage, _ = evolve_coupled_damage(self.source_model, thermal,
            moved, candidate.radius_m/1000., stress)
        damage_increment = float(np.max(np.abs(proposed_damage-thermal.damage0)))
        if damage_increment != 0.:
            raise MovingTiedRetry("moving_genesis_bulk_damage_activation")
        following = MovingThermalContext(thermal.thermal_state, thermal.orbit,
            thermal.column_enthalpy, thermal.damage0, thermal.water_access,
            thermal.boundary_energy_j, context.tidal_heat_received_j+thermal.tidal_heat_received_j)
        return candidate, following

    def observe(self, state, context):
        return observe_material_path(self.model, state, self.support,
            context.water_access, self.force)

    def row(self, state, context, observation):
        mesh = self.model.mesh_for(state)
        _, accumulated = polar_increment(face_deformation(self.model.template_mesh,
            mesh, self.model.initial_radius_m/1000., state.radius_m/1000.))
        depth = state.last_volume_m3/(mesh.areas_unit_sphere*state.radius_m**2)
        heat = float(np.sum(self.mass*context.column_enthalpy))
        forces = self.model.force_diagnostics(state)
        raw_residual = float(np.linalg.norm(forces["residual_force_n"]))
        raw_scale = max(float(np.linalg.norm(forces["internal_force_n"])),
            float(np.linalg.norm(state.last_external_force_n)), 1.)
        return {"elapsed_years": state.elapsed_years,
            "age_myr": context.thermal.time_myr,
            "strength_ratio": observation.onset.maximum_observed_ratio,
            "max_normal_ratio": float(np.max(observation.onset.normal_ratio[observation.recovery.trace_observed])),
            "max_shear_ratio": float(np.max(observation.onset.shear_ratio[observation.recovery.trace_observed])),
            "accumulated_hencky_strain": maximum_total_strain(accumulated),
            "radius_m": state.radius_m, "radius_fraction": state.radius_m/self.model.initial_radius_m,
            "max_material_travel_m": float(state.cumulative_motion_m.max()),
            "elastic_strain": maximum_total_strain(state.elastic_strain),
            "depth_m_quantiles": np.quantile(depth, [0., .5, 1.]).tolist(),
            "force_residual": state.equilibrium_residual,
            "force_residual_norm_n": raw_residual,
            "force_raw_relative_residual": raw_residual/raw_scale,
            "force_roundoff_tolerance_n": forces["absolute_roundoff_tolerance_n"],
            "parent_force_projection_error": observation.parent_force_relative_error,
            "parent_elastic_energy_projection_error": observation.parent_energy_relative_error,
            "column_heat_relative_residual": (heat-self.initial_column_energy_j-context.boundary_energy_j)
                /max(abs(self.initial_column_energy_j), 1.),
            "external_work_j": state.external_work_j, "drag_work_j": state.drag_work_j,
            "bulk_work_j": state.bulk_work_j, "bulk_loading_correction_j": state.bulk_loading_correction_j,
            "mechanical_remainder_j": state.mechanical_remainder_j,
            "accepted_steps": state.accepted_steps, "rejected_steps": state.rejected_steps}

    def save_context(self, path, context):
        metadata = {"version": "moving-genesis-diagnostic-context-0.1",
            "thermal": asdict(context.thermal), "orbit": asdict(context.orbit),
            "boundary_energy_j": context.boundary_energy_j,
            "tidal_heat_received_j": context.tidal_heat_received_j,
            "gate_age_myr": self.gate_age_myr, "production_restart": False}
        np.savez_compressed(path, layer_mass_kg=self.mass,
            column_enthalpy=context.column_enthalpy, damage=context.damage, water_access=context.water_access,
            metadata=np.array(json.dumps(metadata, allow_nan=False, sort_keys=True)))

    def load_context(self, path):
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {"layer_mass_kg", "column_enthalpy", "damage", "water_access", "metadata"}:
                raise ValueError("Incomplete moving thermal context")
            meta = json.loads(str(data["metadata"]))
            if (meta.get("version") != "moving-genesis-diagnostic-context-0.1"
                    or meta.get("gate_age_myr") != self.gate_age_myr
                    or not np.array_equal(data["layer_mass_kg"], self.mass)):
                raise ValueError("Context belongs to different moving columns")
            return MovingThermalContext(GenesisState(**meta["thermal"]), TidalOrbitState(**meta["orbit"]),
                data["column_enthalpy"].copy(), data["damage"].copy(), data["water_access"].copy(),
                meta["boundary_energy_j"], meta["tidal_heat_received_j"])


def run(case, output, max_step, target):
    state, context = deepcopy(case.initial), deepcopy(case.context)
    original_mass = case.mass.copy()
    dt, minimum = max_step, .01
    rows, last_observation, event = [], None, None
    restart_checked = False
    while state.elapsed_years < target:
        attempted = min(dt, target-state.elapsed_years)
        try:
            candidate, following = case.trial(state, context, attempted)
        except MovingTiedRetry as exc:
            state = replace(state, rejected_steps=state.rejected_steps+1)
            if attempted <= minimum*(1+1e-12):
                state = replace(state, stopped_reason=str(exc))
                break
            dt = max(minimum, attempted/2)
            continue
        observation = case.observe(candidate, following)
        score = observation.onset.maximum_observed_ratio
        if score >= 1.:
            if last_observation is None:
                # Recompute every short probe from the unstressed thermal gate.
                low_duration, low, low_context, low_observation = 0., state, context, None
            else:
                low_duration, low, low_context, low_observation = 0., state, context, last_observation
            high_duration, high_observation = attempted, observation
            calls = 0
            while high_duration-low_duration > 1e-5:
                middle = .5*(low_duration+high_duration)
                probe, probe_context = case.trial(state, context, middle)
                measured = case.observe(probe, probe_context)
                calls += 1
                if measured.onset.maximum_observed_ratio < 1.:
                    low_duration, low, low_context, low_observation = middle, probe, probe_context, measured
                else:
                    high_duration, high_observation = middle, measured
                if calls >= 64:
                    raise RuntimeError("Moving strength bracket did not resolve")
            if low_observation is None:
                raise RuntimeError("No positive admissible lower event sample")
            state, context, last_observation = low, low_context, low_observation
            event = {"lower_elapsed_years": state.elapsed_years,
                "upper_elapsed_years": high_observation.state.elapsed_years,
                "width_years": high_duration-low_duration,
                "lower_strength_ratio": low_observation.onset.maximum_observed_ratio,
                "upper_strength_ratio": high_observation.onset.maximum_observed_ratio,
                "callback_count": calls,
                "scope": "sampled crossing on the prescribed transported support; not globally earliest fracture"}
            try:
                born = activate_tied_onset(low_observation.model, low_observation.state,
                    low_observation.stress_pa, low_observation.water_per_trace)
            except ValueError as exc:
                event.update(activation_performed=False, activation_limitation=str(exc))
            else:
                born.model.save_state(output/"local_birth_checkpoint.npz", born.state)
                restored_model, restored = load_born_path(low_observation.model, output/"local_birth_checkpoint.npz")
                event.update(activation_performed=True, governing_mode=born.governing_mode,
                    governing_trace=born.governing_trace, interval_m=asdict(born.state.active_interval),
                    force_replacement_relative_error=born.replacement_force_relative_error,
                    interface_energy_change_j=born.interface_energy_change_j,
                    birth_restart_exact=_exact(restored, born.state))
            rows.append(case.row(state, context, last_observation))
            break
        state, context, last_observation = candidate, following, observation
        rows.append(case.row(state, context, observation))
        dt = min(max_step, attempted*1.5)
        if not restart_checked and state.elapsed_years >= min(10000., target/2):
            case.model.save_state(output/"restart_mechanics.npz", state)
            case.save_context(output/"restart_thermal.npz", context)
            restored_state = case.model.load_state(output/"restart_mechanics.npz")
            restored_context = case.load_context(output/"restart_thermal.npz")
            a, ac = case.trial(state, context, max_step)
            b, bc = case.trial(restored_state, restored_context, max_step)
            if not _exact(a, b) or not _exact(ac, bc):
                raise RuntimeError("Moving thermo-mechanical restart is not exact")
            restart_checked = True
        if len(rows) % 40 == 0:
            print(json.dumps({k: rows[-1][k] for k in ("elapsed_years", "strength_ratio",
                "accumulated_hencky_strain", "force_residual")}), flush=True)
    if state.accepted_steps:
        case.model.save_state(output/"final_mechanics.npz", state)
        case.save_context(output/"final_thermal.npz", context)
        if last_observation is not None:
            last_observation.model.save_state(output/"tied_path_snapshot.npz", last_observation.state)
            np.savez_compressed(output/"path_fields.npz",
                vertices=last_observation.model.basis.subdivision.mesh.vertices,
                faces=last_observation.model.basis.subdivision.mesh.faces,
                parent_face=last_observation.model.basis.subdivision.parent_face,
                traction_pa=last_observation.recovery.traction_pa,
                path_vertices=last_observation.model.basis.insertion.path_vertex_ids,
                path_arclength_m=last_observation.model.basis.insertion.path_arclength_m,
                child_material_fraction=case.support.area_fraction)
    return {"rows": rows, "event": event, "stopped_reason": state.stopped_reason,
        "restart_exact": restart_checked, "layer_mass_unchanged": np.array_equal(original_mass, case.mass),
        "last_accepted_years": state.elapsed_years}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pre-gate-checkpoint", type=Path, default=PRE_GATE)
    parser.add_argument("--max-step-years", type=float, default=500.)
    parser.add_argument("--target-years", type=float, default=200000.)
    args = parser.parse_args()
    if not (np.isfinite(args.max_step_years) and 0 < args.max_step_years <= 10000
            and np.isfinite(args.target_years) and args.target_years > 0):
        raise ValueError("Invalid validation step or target")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    files = [Path(__file__), ROOT/"analysis/genesis_path_precursor_validation.py"]
    files += [ROOT/"tectonics"/name for name in ("genesis_moving_tied.py", "genesis_moving_support.py",
        "genesis_moving_path_diagnostics.py", "genesis_material.py", "genesis_shell.py",
        "genesis_coupled_thermal.py", "genesis_path_basis.py", "genesis_path_dynamics.py",
        "genesis_path_activation.py", "genesis_extrinsic_contact_law.py", "genesis_path_birth.py")]
    code = {str(path.relative_to(ROOT)): _hash(path) for path in files}
    sources = {str(path): _hash(path) for path in (SOURCE, TRACE, args.pre_gate_checkpoint.resolve())}
    case = MovingGenesisCase(build_case(args.pre_gate_checkpoint))
    report = run(case, output, args.max_step_years, args.target_years)
    report.update(scope=__doc__, gate_age_myr=case.gate_age_myr,
        source_cells=case.model.template_mesh.cell_count, max_step_years=args.max_step_years,
        source_sha256=sources, code_sha256=code, wall_seconds=perf_counter()-started)
    rows = report["rows"]
    report["checks"] = {"layer_mass_unchanged": report["layer_mass_unchanged"],
        "restart_exact": report["restart_exact"],
        "accepted_moved_equilibrium": bool(rows) and max(r["force_residual"] for r in rows) <= case.model.parameters.equilibrium_tolerance,
        "parent_force_and_energy_preserved_in_tied_support": bool(rows) and max(max(
            r["parent_force_projection_error"], r["parent_elastic_energy_projection_error"]) for r in rows) < 1e-10,
        "column_energy_accounted": bool(rows) and max(abs(r["column_heat_relative_residual"]) for r in rows) < 1e-10,
        "old_total_strain_limit_crossed": bool(rows) and rows[-1]["accumulated_hencky_strain"] > .005,
        "sources_unchanged": all(_hash(path)==digest for path,digest in sources.items()),
        "code_unchanged_during_run": all(_hash(ROOT/path)==digest for path,digest in code.items())}
    report["artifact_sha256"] = {path.name: _hash(path) for path in output.glob("*.npz")}
    (output/"validation.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in {"rows", "source_sha256", "code_sha256", "artifact_sha256"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
