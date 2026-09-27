"""Continue one thermally born contact with moving background and solidification.

The source is the saved 5120-cell physical thermal onset, not a manufactured
stress ramp. The material path and one active vertex remain prescribed. This
bounded local continuation does not select a new crack or release its fronts.
Thermal columns are conservatively subdivided once; all retries start from one
accepted mechanics/thermal pair. Global mantle and column heat remain separate.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.genesis_moving_genesis_validation import (
    MovingGenesisCase, MovingThermalContext, PRE_GATE)
from analysis.genesis_path_dynamics_validation import SOURCE, TRACE, _exact, _hash
from analysis.genesis_path_precursor_validation import build_case
from tectonics.genesis import GenesisState
from tectonics.genesis_coupled_thermal import advance_thermal_loading, evolve_coupled_damage
from tectonics.genesis_moving_contact import (
    MovingContactMechanics, MovingContactLoading, MovingContactRetry)
from tectonics.genesis_path_activation import load_born_path
from tectonics.genesis_path_birth import recover_tied_tractions, classify_tied_onset
from tectonics.genesis_path_dynamics import PathMechanics
from tectonics.genesis_tides import TidalOrbitState

BIRTH = ROOT / "results/genesis_runs/moving_genesis250_20260927"


class MovingContactCase:
    """Fine material heat history coupled to one active moving contact."""

    def __init__(self, birth_directory=BIRTH):
        self.birth_directory = Path(birth_directory).resolve()
        previous = MovingGenesisCase(build_case(PRE_GATE))
        moving = previous.model.load_state(self.birth_directory / "final_mechanics.npz")
        context = previous.load_context(self.birth_directory / "final_thermal.npz")
        observed = previous.observe(moving, context)
        born, state = load_born_path(observed.model,
            self.birth_directory / "local_birth_checkpoint.npz")
        self.model = MovingContactMechanics(previous.model, previous.support,
            born, state, moving)
        self.initial = deepcopy(self.model.initial_state)
        self.source_model = previous.source_model
        self.force = previous.force
        self.gate_age_myr = previous.gate_age_myr
        self.birth_elapsed_years = moving.elapsed_years
        self.birth_age_myr = context.thermal.time_myr
        self.mass = previous.support.extensive(previous.mass)
        self.mass.setflags(write=False)
        projection = observed.model.basis.subdivision
        self.context = MovingThermalContext(deepcopy(context.thermal), deepcopy(context.orbit),
            projection.intensive(context.column_enthalpy), projection.intensive(context.damage),
            projection.intensive(context.water_access), context.boundary_energy_j,
            context.tidal_heat_received_j)
        self.initial_column_energy_j = previous.initial_column_energy_j
        self.parent_column_energy_j = float(np.sum(previous.mass * context.column_enthalpy))
        self.birth_column_energy_j = float(np.sum(self.mass * self.context.column_enthalpy))
        self.initial_mass_sum = float(previous.mass.sum())
        self.initial_volume = self.initial.last_volume_m3.copy()
        parent_mesh = previous.model.mesh_for(moving)
        parent_depth = previous.mass.sum(axis=1)/(self.source_model.p.density_kg_m3
            *parent_mesh.areas_unit_sphere*moving.radius_m**2)
        child_depth = self.mass.sum(axis=1)/(self.source_model.p.density_kg_m3
            *projection.mesh.areas_unit_sphere*moving.radius_m**2)
        self.child_column_depth_change_fraction = float(np.max(np.abs(
            child_depth/parent_depth[projection.parent_face]-1)))

    def trial(self, state, context, years):
        expected_age = self.gate_age_myr + state.elapsed_years / 1e6
        tolerance = 256*np.finfo(float).eps*max(abs(expected_age), 1.)
        if (context.thermal.time_myr != context.orbit.time_myr
                or abs(context.thermal.time_myr-expected_age) > tolerance):
            raise ValueError("Moving contact and thermal/orbit clocks must form one accepted pair")
        basis = self.model.basis_for(state)
        mesh = basis.subdivision.mesh
        p = self.source_model.p
        thermal = advance_thermal_loading(self.source_model, mesh=mesh,
            radius_km=state.radius_m/1000., layer_mass_kg=self.mass,
            column_enthalpy=context.column_enthalpy, elastic_strain=state.elastic_strain,
            damage=context.damage, water_access=context.water_access,
            boundary_energy_j=context.boundary_energy_j,
            thermal_state=context.thermal, orbit=context.orbit,
            target_myr=context.thermal.time_myr+years/1e6)
        if abs(thermal.dt_myr*1e6-years) > max(2e-9, years*1e-9):
            raise ValueError("Thermal and requested contact step disagree")
        volume = self.mass.sum(axis=1)*thermal.fraction/p.density_kg_m3
        young = p.young_modulus_pa*(p.residual_stiffness
            +(1-p.residual_stiffness)*(1-thermal.damage0)**2)
        water = np.repeat(thermal.water_access[basis.topology.seam_faces].mean(axis=1), 2)
        loading = MovingContactLoading(years, volume, young, thermal.memory,
            thermal.effective_b, water, self.force)
        candidate = self.model.trial(state, loading)
        current = self.model.basis_for(candidate)
        proposed_damage, _ = evolve_coupled_damage(self.source_model, thermal,
            current.subdivision.mesh, candidate.radius_m/1000., self.model.stress(candidate))
        if np.max(np.abs(proposed_damage-thermal.damage0)) != 0.:
            raise MovingContactRetry("moving_contact_bulk_damage_activation")
        following = MovingThermalContext(thermal.thermal_state, thermal.orbit,
            thermal.column_enthalpy, thermal.damage0, thermal.water_access,
            thermal.boundary_energy_j, context.tidal_heat_received_j+thermal.tidal_heat_received_j)
        return candidate, following

    def save_context(self, path, context):
        meta = {"version": "moving-contact-thermal-context-0.1",
            "gate_age_myr": self.gate_age_myr, "birth_elapsed_years": self.birth_elapsed_years,
            "thermal": asdict(context.thermal), "orbit": asdict(context.orbit),
            "boundary_energy_j": context.boundary_energy_j,
            "tidal_heat_received_j": context.tidal_heat_received_j,
            "production_restart": False}
        np.savez_compressed(path, layer_mass_kg=self.mass,
            column_enthalpy=context.column_enthalpy, damage=context.damage,
            water_access=context.water_access, metadata=np.array(json.dumps(meta, allow_nan=False)))

    def load_context(self, path):
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {"layer_mass_kg", "column_enthalpy", "damage", "water_access", "metadata"}:
                raise ValueError("Incomplete contact thermal context")
            meta = json.loads(str(data["metadata"]))
            if (meta.get("version") != "moving-contact-thermal-context-0.1"
                    or meta.get("gate_age_myr") != self.gate_age_myr
                    or meta.get("birth_elapsed_years") != self.birth_elapsed_years
                    or not np.array_equal(data["layer_mass_kg"], self.mass)):
                raise ValueError("Context belongs to different contact material columns")
            return MovingThermalContext(GenesisState(**meta["thermal"]), TidalOrbitState(**meta["orbit"]),
                data["column_enthalpy"].copy(), data["damage"].copy(), data["water_access"].copy(),
                meta["boundary_energy_j"], meta["tidal_heat_received_j"])

    def row(self, state, context):
        jumps = self.model.jumps(state)
        forces = self.model.force_diagnostics(state)
        free = forces["free_dofs"]
        residual = float(np.linalg.norm(forces["residual_force_n"][free]))
        scale = max(*(float(np.linalg.norm(forces[name][free])) for name in
            ("internal_force_n", "contact_force_n", "external_force_n")), 1.)
        initial, added = state.history.initial, state.history.added
        heat = float(np.sum(self.mass*context.column_enthalpy))
        row = {"elapsed_years": state.elapsed_years,
            "years_after_birth": state.elapsed_years-self.birth_elapsed_years,
            "age_myr": context.thermal.time_myr,
            "radius_m": state.radius_m,
            "max_opening_m": float(np.max(jumps[:, 0], initial=0)),
            "max_slip_m": float(np.max(np.abs(jumps[:, 1]), initial=0)),
            "penetration_m": float(np.max(-jumps[:, 0], initial=0)),
            "initial_cohort_damage_max": float(np.max(initial.damage, initial=0)),
            "initial_max_opening_m": float(np.max(initial.max_opening_m, initial=0)),
            "new_cohort_count": len(added.trace_index),
            "new_unbonded_cohort_count": int(np.count_nonzero(~added.bonded)),
            "new_cohort_area_m2": float(added.area_ref_m2.sum()),
            "initial_cohort_area_m2": float(initial.area_ref_m2.sum()),
            "solid_volume_m3": float(state.last_volume_m3.sum()),
            "max_solid_volume_growth_fraction": float(np.max(state.last_volume_m3/self.initial_volume-1)),
            "force_relative_residual": state.equilibrium_residual,
            "force_raw_relative_residual": residual/scale,
            "force_residual_norm_n": residual,
            "reference_to_current_area_ratio_range": [
                float(np.min(forces["reference_to_current_area_ratio"])),
                float(np.max(forces["reference_to_current_area_ratio"]))],
            "contact_potential_j": self.model.history_model.energy(state.history, jumps[:, 0], jumps[:, 1]),
            "column_heat_relative_residual": (heat-self.initial_column_energy_j-context.boundary_energy_j)
                /max(abs(self.initial_column_energy_j), 1.),
            "accepted_steps": state.accepted_steps, "rejected_steps": state.rejected_steps}
        for name in ("external_work_j", "drag_work_j", "bulk_work_j",
                     "bulk_loading_correction_j", "mechanical_remainder_j",
                     "contact_geometric_work_j", "cohort_parameter_energy_j"):
            row[name] = float(getattr(state, name))
            row["post_birth_"+name] = float(getattr(state, name)-getattr(self.initial, name))
        for name in ("fracture_work_j", "friction_work_j", "viscous_work_j", "shear_remainder_j"):
            row[name] = float(getattr(initial, name).sum()+getattr(added, name).sum())
        row.update(self.held_strength(state, context))
        return row

    def held_strength(self, state, context):
        """Observe only constrained vertices; never reinterpret an open contact.

        A disposable all-tied diagnostic carries the current bulk stress and
        measured held reactions. Open-vertex traces are explicitly excluded
        from its reconstruction. This does not alter active mechanics/history.
        """
        basis = self.model.basis_for(state)
        depth = state.last_volume_m3/(basis.subdivision.mesh.areas_unit_sphere*state.radius_m**2)
        tied = PathMechanics(basis, depth, self.model.parameters, self.model.law_parameters)
        snapshot = tied.initial(state.elastic_strain)
        snapshot.elapsed_years = state.elapsed_years
        snapshot.last_step_years = state.last_step_years
        snapshot.accepted_steps = state.accepted_steps
        snapshot.equilibrium_residual = state.equilibrium_residual
        snapshot.constraint_reaction_n = state.constraint_reaction_n.copy()
        stress = state.last_young_modulus_pa[:, None]*(state.elastic_strain@basis.membrane.d.T)
        recovered = recover_tied_tractions(tied, snapshot, stress)
        water = np.repeat(context.water_access[basis.topology.seam_faces].mean(axis=1), 2)
        onset = classify_tied_onset(recovered, water, tied.law_parameters)
        relative_free = self.model.free[self.model.free >= basis.nparent]
        touched = np.asarray(tied.jump_operator[:, relative_free].power(2).sum(axis=1)).ravel()
        eligible = recovered.trace_observed & (touched.reshape(-1, 2).sum(axis=1) == 0)
        ratio = np.maximum(onset.normal_ratio, onset.shear_ratio)
        selected = np.flatnonzero(eligible)
        trace = int(selected[np.argmax(ratio[selected])]) if len(selected) else None
        return {"max_held_strength_ratio": float(ratio[trace]) if trace is not None else 0.,
            "governing_held_trace": trace,
            "held_traction_recovery_residual": recovered.relative_residual}


def run(case, output, duration, maximum, minimum):
    state, context = deepcopy(case.initial), deepcopy(case.context)
    rows = [case.row(state, context)]
    target, dt = state.elapsed_years+duration, maximum
    restart_checked = False
    next_event = None
    started = perf_counter()
    while state.elapsed_years < target:
        attempted = min(dt, target-state.elapsed_years)
        try:
            candidate, following = case.trial(state, context, attempted)
        except MovingContactRetry as exc:
            state = replace(state, rejected_steps=state.rejected_steps+1)
            if attempted <= minimum*(1+1e-12):
                state = replace(state, stopped_reason=str(exc))
                break
            dt = max(minimum, attempted/2)
            continue
        observed = case.row(candidate, following)
        if observed["max_held_strength_ratio"] >= 1.:
            # A single-contact experiment must stop before another held vertex
            # is driven beyond its strength. Locate from the same accepted pair.
            lo, hi = 0., attempted
            low, low_context, low_row = state, context, rows[-1]
            calls = 0
            while hi-lo > 1e-3:
                middle = (lo+hi)/2
                probe, probe_context = case.trial(state, context, middle)
                measured = case.row(probe, probe_context)
                calls += 1
                if measured["max_held_strength_ratio"] < 1.:
                    lo, low, low_context, low_row = middle, probe, probe_context, measured
                else:
                    hi, observed = middle, measured
                if calls > 60:
                    raise RuntimeError("Next held strength event did not localize")
            next_event = {"lower_years_after_birth": low.elapsed_years-case.birth_elapsed_years,
                "upper_years_after_birth": state.elapsed_years+hi-case.birth_elapsed_years,
                "lower_ratio": low_row["max_held_strength_ratio"],
                "upper_ratio": observed["max_held_strength_ratio"],
                "held_trace": observed["governing_held_trace"], "trial_count": calls,
                "action": "Stop at the admissible lower sample; no second birth or front advance."}
            state, context = low, low_context
            if lo > 0:
                rows.append(low_row)
            break
        state, context = candidate, following
        rows.append(observed)
        print(json.dumps({key: rows[-1][key] for key in ("years_after_birth",
            "max_opening_m", "initial_cohort_damage_max", "new_unbonded_cohort_count",
            "force_raw_relative_residual", "max_held_strength_ratio")}), flush=True)
        if not restart_checked:
            case.model.save_state(output/"restart_mechanics.npz", state)
            case.save_context(output/"restart_thermal.npz", context)
            restored = case.model.load_state(output/"restart_mechanics.npz")
            restored_context = case.load_context(output/"restart_thermal.npz")
            if not _exact(state, restored) or not _exact(context, restored_context):
                raise RuntimeError("Contact mechanics/thermal checkpoint is not exact")
            # Compare an actual following step from the same accepted history.
            a, ac = case.trial(state, context, min(maximum, duration))
            b, bc = case.trial(restored, restored_context, min(maximum, duration))
            if not _exact(a, b) or not _exact(ac, bc):
                raise RuntimeError("Contact restart does not reproduce the next physical step")
            restart_checked = True
        dt = min(maximum, attempted*1.5)
    case.model.save_state(output/"final_mechanics.npz", state)
    case.save_context(output/"final_thermal.npz", context)
    np.savez_compressed(output/"final_fields.npz", vertices=state.vertices,
        elastic_strain=state.elastic_strain, jumps_m=case.model.jumps(state),
        initial_damage=state.history.initial.damage,
        added_bonded=state.history.added.bonded,
        added_trace_index=state.history.added.trace_index)
    seed = case.initial.history.initial
    unchanged_material = all(np.array_equal(getattr(seed, name), getattr(state.history.initial, name))
        for name in ("trace_index", "z_lo_ref_m", "z_hi_ref_m", "area_ref_m2",
                     "birth_time_myr", "birth_gap_m", "birth_jump_m", "bonded"))
    return {"scope": __doc__, "birth_directory": str(case.birth_directory),
        "birth_age_myr": case.birth_age_myr, "birth_elapsed_years": case.birth_elapsed_years,
        "parent_cells": case.model.support.parent_mesh.cell_count,
        "material_cells": len(case.mass), "maximum_step_years": maximum,
        "requested_duration_years": duration, "wall_seconds": perf_counter()-started,
        "stopped_reason": state.stopped_reason, "next_held_event": next_event, "rows": rows,
        "child_column_depth_change_fraction_at_transfer": case.child_column_depth_change_fraction,
        "checks": {"duration_reached_or_next_strength_event_located": state.elapsed_years >= target or next_event is not None,
            "restart_exact": restart_checked,
            "mass_subdivision_conservative": abs(case.mass.sum()-case.initial_mass_sum)
                <= 2e-15*case.initial_mass_sum,
            "birth_column_energy_preserved": abs(case.birth_column_energy_j-case.parent_column_energy_j)
                <= 2e-15*case.parent_column_energy_j,
            "initial_contact_material_measures_unchanged": unchanged_material,
            "column_heat_accounted": max(abs(row["column_heat_relative_residual"]) for row in rows) < 1e-12,
            "new_solid_material_tracked": len(state.history.added.trace_index) > 0,
            "no_held_strength_overshoot_committed": max(row["max_held_strength_ratio"] for row in rows) < 1.,
            "moving_equilibrium_accepted": max(row["force_relative_residual"] for row in rows)
                <= case.model.parameters.equilibrium_tolerance}}


def plot(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = report["rows"]
    t = [r["years_after_birth"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    entries = (("max_opening_m", "Раскрытие родившегося контакта", "метры"),
        ("initial_cohort_damage_max", "Повреждение исходного контакта", "доля"),
        ("new_cohort_area_m2", "Учтённое новое твёрдое сечение", "м²"),
        ("force_raw_relative_residual", "Прямая относительная невязка сил", "невязка"))
    for ax, (key, title, ylabel) in zip(axes.flat, entries):
        ax.plot(t, [r[key] for r in rows], "o-", markersize=3)
        ax.set(title=title, xlabel="Годы после рождения контакта", ylabel=ylabel)
        ax.grid(alpha=.25)
    axes[1, 1].set_yscale("log")
    fig.suptitle("Продолжение теплового генезиса: один контакт, меняющаяся геометрия\n"
        "Заданная линия и фиксированные концы активной области; сеть плит ещё не формируется")
    fig.savefig(output/"validation.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--birth-directory", type=Path, default=BIRTH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-years", type=float, default=100.)
    parser.add_argument("--max-step-years", type=float, default=20.)
    parser.add_argument("--min-step-years", type=float, default=.01)
    args = parser.parse_args()
    if (not np.isfinite([args.min_step_years, args.max_step_years, args.duration_years]).all()
            or not 0 < args.min_step_years <= args.max_step_years or args.duration_years <= 0):
        raise ValueError("Positive duration and ordered step bounds are required")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Validation output must be new or empty")
    sources = [SOURCE, TRACE, PRE_GATE]+[args.birth_directory/name for name in
        ("final_mechanics.npz", "final_thermal.npz", "local_birth_checkpoint.npz", "validation.json")]
    code = sorted((ROOT/"tectonics").glob("genesis*.py"))+[Path(__file__).resolve(),
        ROOT/"analysis/genesis_moving_genesis_validation.py",
        ROOT/"analysis/genesis_path_precursor_validation.py",
        ROOT/"analysis/genesis_path_dynamics_validation.py"]
    source_hashes = {str(path.resolve()): _hash(path) for path in sources}
    code_hashes = {str(path.relative_to(ROOT)): _hash(path) for path in code}
    case = MovingContactCase(args.birth_directory)
    output.mkdir(parents=True, exist_ok=True)
    report = run(case, output, args.duration_years, args.max_step_years, args.min_step_years)
    plot(report, output)
    report["source_sha256"], report["code_sha256"] = source_hashes, code_hashes
    report["artifact_sha256"] = {path.name: _hash(path) for path in output.iterdir() if path.is_file()}
    report["checks"].update(sources_unchanged=all(_hash(Path(p)) == h for p,h in source_hashes.items()),
        code_unchanged_during_run=all(_hash(ROOT/p) == h for p,h in code_hashes.items()))
    report["checks"] = {key: bool(value) for key, value in report["checks"].items()}
    (output/"validation.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"last": report["rows"][-1], "checks": report["checks"],
        "stopped_reason": report["stopped_reason"]}, indent=2), flush=True)
    if not all(report["checks"].values()):
        raise SystemExit("Moving contact continuation did not pass every requested check; see report")


if __name__ == "__main__":
    main()
