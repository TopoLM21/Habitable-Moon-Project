"""Manufactured stress onset and force-preserving birth on actual 5120 geometry.

This is a constitutive/discretization control. The source supplies only mesh,
radius, path, and layer depth. Its prestress, thermal history and diffuse damage
are not evolved, rescaled, or used to manufacture an earlier planetary event.
The prescribed path is not an automatically discovered nucleation location.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, fields
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.genesis_path_dynamics_validation import SOURCE, TRACE, _hash, _array_hash, _exact, _relative, _row
from tectonics.genesis_coupled import CoupledModel
from tectonics.genesis_crack_path import ReferenceCrackPath
from tectonics.genesis_path_activation import activate_tied_onset, load_born_path
from tectonics.genesis_path_basis import EmbeddedPathBasis
from tectonics.genesis_path_birth import recover_tied_tractions, classify_tied_onset
from tectonics.genesis_path_dynamics import PathMechanics, PathLoading
from tectonics.genesis_path_geometry import PathGeometryParameters
from tectonics.genesis_path_mesh import insert_crack_path
from tectonics.genesis_path_onset_event import locate_tied_onset


def build_fixture():
    coupled = CoupledModel(SOURCE.read_bytes(), source_path=str(SOURCE))
    source = coupled.source.source_state
    with np.load(TRACE, allow_pickle=False) as data:
        path = ReferenceCrackPath(data["points_xyz"], source.radius_km)
    insertion = insert_crack_path(coupled.original_mesh, path,
        front_coordinates_m=path.length_m*np.array([.25, .5, .75]))
    basis = EmbeddedPathBasis(coupled.original_mesh, insertion, coupled.radius_m,
                              coupled.source_model.p.poisson_ratio)
    model = PathMechanics(basis, basis.subdivision.intensive(coupled.source.depth_m),
        coupled.contact_parameters, coupled.law_parameters,
        geometry_parameters=PathGeometryParameters())
    n = basis.subdivision.mesh.cell_count
    elasticity = np.broadcast_to(60e9*basis.membrane.d, (n, 3, 3)).copy()
    volume = model.reference_volume_m3.copy()
    memory = np.tile([1e-5, 1e-5, 0.], (n, 1))
    _, parent_force = basis.bulk(volume, elasticity, memory)
    parent_force[basis.nparent:] = 0.
    return coupled, model, volume, elasticity, memory, parent_force


def loading(model, volume, elasticity, memory, force, dt):
    n = len(volume)
    return PathLoading(dt, volume, elasticity, memory, np.ones(n), force, np.zeros(n))


def plot_result(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].plot([0, 8, 20], [.2, 1., 2.2], label="Заданная нагрузка")
    axes[0, 0].axhline(1., color="black", linestyle=":", label="Порог прочности")
    bracket = report["onset_bracket"]
    axes[0, 0].plot([bracket["lower_years"]], [bracket["lower_score"]], "o", color="tab:red", label="Активация")
    axes[0, 0].set(xlabel="Годы контрольного нагружения", ylabel="Отношение к прочности",
                   title="Найденный момент заданного события")
    axes[0, 0].legend(fontsize=8)
    for label, result in report["evolution"].items():
        rows = result["rows"]
        time = [row["years_since_prescribed_release"] for row in rows]
        axes[0, 1].plot(time, [row["max_opening_m"] for row in rows], "o-", markersize=3,
                         label=f"Шаг до {label} года")
        axes[1, 0].plot(time, [row["max_cohesive_damage"] for row in rows], "o-", markersize=3,
                         label=f"Шаг до {label} года")
        axes[1, 1].plot(time, [row["fracture_work_j"]/1e12 for row in rows], "o-", markersize=3,
                         label=f"Шаг до {label} года")
    axes[0, 1].set(xlabel="Годы после локальной активации", ylabel="Раскрытие, м",
                   title="Один освобождённый узел разлома")
    axes[1, 0].set(xlabel="Годы после локальной активации", ylabel="Повреждение контакта",
                   title="Разрушение при дальнейшем нагружении")
    axes[1, 1].set(xlabel="Годы после локальной активации", ylabel="Работа разрушения, ТДж",
                   title="Положительная работа Mode I")
    for axis in axes.ravel():
        axis.grid(alpha=.25)
    for axis in (axes[0, 1], axes[1, 0], axes[1, 1]):
        axis.legend(fontsize=8)
    figure.suptitle("Сетка 5120 ячеек: контроль рождения одного контакта\n"
                   "Искусственно заданные напряжения; время не является возрастом планеты", fontsize=12)
    figure.savefig(output/"validation.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Validation output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    code_paths = [Path(__file__), ROOT/"analysis/genesis_path_dynamics_validation.py"]
    code_paths += [ROOT/"tectonics"/name for name in (
        "genesis_path_activation.py", "genesis_extrinsic_contact_law.py",
        "genesis_path_onset_event.py", "genesis_path_birth.py", "genesis_path_dynamics.py",
        "genesis_path_geometry.py", "genesis_path_basis.py", "genesis_path_mesh.py",
        "genesis_path_material.py", "genesis_contact_geometry.py", "genesis_contact_growth.py",
        "genesis_contact_law.py", "genesis_coupled.py")]
    code_hashes = {str(path.relative_to(ROOT)): _hash(path) for path in code_paths}
    source_hashes = {str(path.relative_to(ROOT)): _hash(path) for path in (SOURCE, TRACE)}
    coupled, tied, volume, elasticity, unit_memory, unit_force = build_fixture()
    source = coupled.source.source_state
    source_array_hashes = {field.name: _array_hash(getattr(source, field.name))
        for field in fields(source) if isinstance(getattr(source, field.name), np.ndarray)}
    n = len(volume)
    empty = tied.initial(np.zeros((n, 3)))
    water = np.zeros(len(tied.trace_depth_m))

    def prescribed_trial(old, dt, factor):
        state = tied.trial(old, loading(tied, volume, elasticity,
            unit_memory*factor, unit_force*factor, dt))
        stress = np.einsum("fij,fj->fi", elasticity, state.elastic_strain)
        return state, stress, water.copy()

    calibration, stress, _ = prescribed_trial(empty, 1., 1.)
    calibration_ratio = classify_tied_onset(recover_tied_tractions(tied, calibration, stress),
        water, tied.law_parameters).maximum_observed_ratio
    start, start_stress, _ = prescribed_trial(empty, 1., .2/calibration_ratio)
    original_start = deepcopy(start)
    call_durations = []
    def callback(dt):
        call_durations.append(dt)
        return prescribed_trial(start, dt, (.2+.1*dt)/calibration_ratio)
    bracket = locate_tied_onset(tied, start, start_stress, water, callback, 20.,
        time_tolerance_years=1e-6)
    event = bracket.lower
    event_stress = np.einsum("fij,fj->fi", elasticity, event.state.elastic_strain)
    before = deepcopy(event.state)
    transition = activate_tied_onset(tied, event.state, event_stress, water)
    born, model = transition.state, transition.model
    born_snapshot = deepcopy(born)
    base_factor = (.2+.1*event.duration_years)/calibration_ratio
    baseline_force = unit_force*base_factor
    replacement = np.asarray(model.jump_operator.T@(
        model.geometry.interface_area_m2[:, None]*born.cohorts.traction_pa).ravel()).ravel()
    recovered = born.constraint_reaction_n+replacement
    force_error = _relative(recovered, before.constraint_reaction_n)
    virtual_direction = np.random.default_rng(1701).normal(size=tied.basis.ndof)
    virtual_error = abs(float(virtual_direction@(recovered-before.constraint_reaction_n)))/max(
        abs(float(virtual_direction@before.constraint_reaction_n)), 1.)
    model.save_state(output/"birth_checkpoint.npz", born)
    tied.save_state(output/"tied_event_checkpoint.npz", event.state)

    hold_rows = []
    for duration in (.01, 1., 100.):
        after = model.trial(born, loading(model, volume, elasticity, born.elastic_strain,
            baseline_force, duration))
        hold_rows.append({"duration_years": duration,
            "displacement_change_norm_m": float(np.linalg.norm(after.displacement_m-born.displacement_m)),
            "elastic_strain_unchanged": np.array_equal(after.elastic_strain, born.elastic_strain),
            "traction_unchanged": np.array_equal(after.cohorts.traction_pa, born.cohorts.traction_pa),
            "added_fracture_work_j": float((after.cohorts.fracture_work_j-born.cohorts.fracture_work_j).sum()),
            "added_drag_work_j": after.drag_work_j-born.drag_work_j})
    old_release = tied.release(before, born.active_interval)
    old_advanced = tied.trial(old_release, loading(tied, volume, elasticity,
        old_release.elastic_strain, baseline_force, .01))
    old_motion = float(np.linalg.norm(old_advanced.displacement_m-old_release.displacement_m))
    print(json.dumps({"stage": "birth", "force_error": force_error,
        "bracket_years": [bracket.lower.duration_years, bracket.upper.duration_years],
        "old_release_motion_m": old_motion}), flush=True)

    # A continuing prescribed background ramp. The evolved local elastic
    # strain is retained; only a fresh imposed isotropic increment is added.
    # External force depends solely on the prescribed background clock, never
    # on deformed-state stresses or on the contact response.
    after_birth_slope = .01
    def factory(state, dt):
        factor_increment = after_birth_slope*dt/calibration_ratio
        factor = base_factor+after_birth_slope*(state.elapsed_years+dt-born.elapsed_years)/calibration_ratio
        return loading(model, volume, elasticity, state.elastic_strain+unit_memory*factor_increment,
                       unit_force*factor, dt)

    evolution, final_states, midway = {}, {}, deepcopy(born)
    model.save_state(output/"midway_checkpoint.npz", midway)
    for maximum in (1., .5):
        state = deepcopy(born)
        rows = [_row(model, state, born.elapsed_years)]
        for year in range(1, 11):
            state = model.advance(state, born.elapsed_years+year, factory,
                max_step_years=maximum, min_step_years=.001)
            rows.append(_row(model, state, born.elapsed_years))
            if maximum == .5 and year == 5 and not state.stopped_reason:
                midway = deepcopy(state)
                model.save_state(output/"midway_checkpoint.npz", state)
            if state.stopped_reason:
                break
        model.save_state(output/f"final_step_{maximum:g}_checkpoint.npz", state)
        final_states[str(maximum)] = state
        evolution[str(maximum)] = {"max_step_years": maximum, "rows": rows,
            "reached_requested_10_years": state.elapsed_years == born.elapsed_years+10.,
            "stopped_reason": state.stopped_reason}
    restored_model, restored = load_born_path(tied, output/"midway_checkpoint.npz")
    restarted = restored
    restart_year = int(round(midway.elapsed_years-born.elapsed_years))
    for year in range(restart_year+1, 11):
        restarted = restored_model.advance(restarted, born.elapsed_years+year, factory,
            max_step_years=.5, min_step_years=.001)
        if restarted.stopped_reason:
            break
    fine, coarse = final_states["0.5"], final_states["1.0"]
    model.save_state(output/"restart_final_checkpoint.npz", restarted)
    jump = (model.jump_operator@fine.displacement_m).reshape(-1, 2)
    np.savez_compressed(output/"controlled_material_and_fields.npz",
        volume_m3=volume, elasticity_pa=elasticity, unit_memory=unit_memory,
        prescribed_unit_parent_force_n=unit_force, physical_jump_m=jump,
        birth_traction_pa=model.birth_traction_pa, final_traction_pa=fine.cohorts.traction_pa,
        final_damage=fine.cohorts.damage, source_depth_m=tied.depth_m,
        metadata=np.array(json.dumps({"scope": "manufactured_stress_real_geometry",
            "calibration_ratio": calibration_ratio, "base_factor": base_factor,
            "post_birth_score_slope_per_year": after_birth_slope,
            "birth_elapsed_years": born.elapsed_years}, allow_nan=False)))

    unchanged_cohorts = all(np.array_equal(getattr(before.cohorts, f.name), getattr(born.cohorts, f.name))
        for f in fields(before.cohorts) if f.name != "traction_pa")
    unchanged_bulk = all(np.array_equal(getattr(before, name), getattr(born, name)) for name in (
        "elapsed_years", "displacement_m", "elastic_strain", "drag_work_j", "external_work_j",
        "bulk_work_j", "bulk_loading_correction_j", "mechanical_remainder_j", "accepted_steps",
        "rejected_steps", "last_step_years"))
    sources_unchanged = all(_hash(ROOT/name) == digest for name, digest in source_hashes.items()) and all(
        _array_hash(getattr(source, name)) == digest for name, digest in source_array_hashes.items())
    code_unchanged = all(_hash(ROOT/name) == digest for name, digest in code_hashes.items())
    checks = {
        "known_eight_year_event_bracketed": bracket.lower.duration_years <= 8. <= bracket.upper.duration_years,
        "admissible_event_near_strength": 1-1e-6 <= transition.strength_ratio <= 1.,
        "accepted_start_and_event_unchanged": _exact(start, original_start) and _exact(event.state, before),
        "birth_preserves_bulk_and_material_history": unchanged_bulk and unchanged_cohorts,
        "birth_adds_no_interface_energy": transition.interface_energy_change_j == 0.,
        "birth_replaces_reaction_and_virtual_work": force_error < 1e-12 and virtual_error < 1e-12,
        "unchanged_load_has_no_release_motion": all(row["displacement_change_norm_m"] == 0 and
            row["elastic_strain_unchanged"] and row["traction_unchanged"] and
            row["added_fracture_work_j"] == 0 and row["added_drag_work_j"] == 0 for row in hold_rows),
        "old_zero_traction_release_moves": old_motion > 1e-5,
        "both_steps_reach_ten_years": all(value["reached_requested_10_years"] for value in evolution.values()),
        "further_loading_opens_and_damages": jump[:, 0].max() > 0 and fine.cohorts.damage.max() > 0
            and fine.cohorts.fracture_work_j.sum() > 0,
        "checkpoint_and_same_sequence_future_exact": _exact(midway, restored) and _exact(fine, restarted),
        "born_input_unchanged": _exact(born, born_snapshot),
        "sources_unchanged": sources_unchanged,
        "code_unchanged_during_run": code_unchanged,
    }
    report = {
        "scope": "manufactured isotropic stress ramp on actual 5120 geometry and depth; one local contact birth",
        "limitations": ["manufactured elapsed years are not planetary age or a predicted genesis event time",
            "no inherited source prestress or diffuse damage; source used only for geometry, radius, path and depth",
            "homogeneous 60 GPa elasticity, fixed thickness and zero water; no thermal, orbit or Maxwell evolution",
            "prescribed candidate path; one local vertex activation, no propagation or birth of plates",
            "further competing strength events remain tied; subsequent local evolution is a controlled experiment",
            "step comparison is a sensitivity control, not an error-controlled convergence proof"],
        "source_sha256": source_hashes, "source_array_sha256": source_array_hashes,
        "code_sha256": code_hashes, "source_cells": coupled.original_mesh.cell_count,
        "child_cells": tied.basis.subdivision.mesh.cell_count,
        "generalized_dofs": tied.basis.ndof, "path_length_km": tied.basis.insertion.path.length_m/1000.,
        "calibration_ratio": calibration_ratio, "tied_fingerprint": tied.fingerprint,
        "born_fingerprint": model.fingerprint,
        "onset_bracket": {"lower_years": bracket.lower.duration_years, "upper_years": bracket.upper.duration_years,
            "width_years": bracket.time_width_years, "lower_score": bracket.lower.onset.maximum_observed_ratio,
            "upper_score": bracket.upper.onset.maximum_observed_ratio, "callback_count": bracket.callback_count,
            "durations_years": call_durations},
        "birth": {"elapsed_years": born.elapsed_years, "governing_trace": transition.governing_trace,
            "governing_mode": transition.governing_mode, "interval_m": asdict(born.active_interval),
            "released_dof_count": transition.released_dof_count,
            "strength_ratio": transition.strength_ratio,
            "replacement_force_relative_error": transition.replacement_force_relative_error,
            "global_reaction_identity_relative_error": force_error, "virtual_work_relative_error": virtual_error,
            "interface_energy_change_j": transition.interface_energy_change_j,
            "selected_traction_pa": model.birth_traction_pa[np.any(model.birth_traction_pa != 0, axis=1)].tolist()},
        "hold_controls": hold_rows, "old_zero_traction_release_motion_norm_m_at_0p01_years": old_motion,
        "post_birth_score_slope_per_year": after_birth_slope, "evolution": evolution,
        "timestep_comparison_at_ten_years": {
            "displacement_relative_difference": _relative(coarse.displacement_m, fine.displacement_m),
            "fracture_work_relative_difference": _relative(coarse.cohorts.fracture_work_j, fine.cohorts.fracture_work_j),
            "friction_work_relative_difference": _relative(coarse.cohorts.friction_work_j, fine.cohorts.friction_work_j),
            "max_damage_absolute_difference": float(np.max(np.abs(coarse.cohorts.damage-fine.cohorts.damage)))},
        "restart": {"midway_state_exact": _exact(midway, restored), "future_state_exact": _exact(fine, restarted)},
        "checks": checks, "wall_seconds": perf_counter()-started,
    }
    plot_result(report, output)
    report["artifact_sha256"] = {path.name: _hash(path) for path in output.iterdir() if path.suffix in (".npz", ".png")}
    def serial(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    (output/"validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2,
        allow_nan=False, default=serial)+"\n", encoding="utf-8")
    print(json.dumps({"report": str(output/"validation.json"), "checks": checks,
        "final": evolution["0.5"]["rows"][-1], "wall_seconds": report["wall_seconds"]}, default=serial), flush=True)
    if not all(checks.values()):
        raise RuntimeError("Activation validation failed; inspect the report")


if __name__ == "__main__":
    main()
