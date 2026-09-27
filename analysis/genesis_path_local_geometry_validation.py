"""Controlled source continuation with explicit local fixed-geometry limits.

The support and release interval remain prescribed. Temperature, water, bulk
damage, phase and source thermal/orbit clocks remain frozen. No reference reset
or automatic fracture nucleation occurs in these mechanical-time experiments.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.genesis_path_dynamics_validation import (
    SOURCE, TRACE, CoupledModel, ReferenceCrackPath, insert_crack_path,
    EmbeddedPathBasis, PathMechanics, CrackInterval, _external, _row,
    _exact, _relative, _array_hash)
from analysis.genesis_path_geometry_audit import audit
from tectonics.genesis_contact import SECONDS_PER_YEAR
from tectonics.genesis_contact_growth import evaluate_cohorts
from tectonics.genesis_path_dynamics import _PathRetry
from tectonics.genesis_path_geometry import PathGeometryParameters

OLD_OUTPUT = ROOT/"results/genesis_runs/path_dynamics_verified_20260926"


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RecordedPathMechanics(PathMechanics):
    """Observer records only successful trials; no solver behavior changes."""
    def trial(self, state, loading):
        result = super().trial(state, loading)
        self.accepted.append((loading.dt_years, result))
        self.last_before, self.last_after = state, result
        return result


def _make_model(basis, coupled, *, local, record=False):
    cls = RecordedPathMechanics if record else PathMechanics
    result = cls(basis, basis.subdivision.intensive(coupled.source.depth_m),
        coupled.contact_parameters, coupled.law_parameters,
        geometry_parameters=PathGeometryParameters() if local else None)
    if record:
        result.accepted = []
    return result


def _loading(coupled, model):
    source, p = coupled.source.source_state, coupled.source_model.p
    projection = model.basis.subdivision
    degradation = p.residual_stiffness+(1-p.residual_stiffness)*(1-source.damage)**2
    elasticity = projection.intensive((p.young_modulus_pa*degradation)[:, None, None]*coupled.original_membrane.d)
    temp = coupled.source.source_fields["temperature_k"]
    viscosity = projection.intensive(np.clip(p.viscosity_reference_pa_s*np.exp(np.clip(
        p.activation_energy_j_mol/8.314462618*(1/np.maximum(temp, 1)-1/p.viscosity_reference_temperature_k), -60, 60)),
        p.viscosity_min_pa_s, p.viscosity_max_pa_s))
    _, external = _external(coupled, model.basis, coupled.source.depth_m/1000.)
    arrays = {"volume_m3": model.reference_volume_m3, "elasticity": elasticity,
        "viscosity_pa_s": viscosity, "external_force_n": external,
        "water_access": projection.intensive(source.water_access)}
    def factory(state, dt):
        return model.isothermal_loading(state, dt, arrays["volume_m3"], elasticity,
            viscosity, p.young_modulus_pa, external, arrays["water_access"])
    return factory, arrays


def _fresh_release(model, coupled, factory):
    state = model.initial(model.basis.subdivision.tensor(coupled.source.source_state.elastic_strain,
                                                       engineering=True))
    state = model.advance(state, 1., factory, max_step_years=1.)
    if state.stopped_reason:
        raise RuntimeError("Tied initialization stopped: "+state.stopped_reason)
    path = model.basis.insertion.path
    return model.release(state, CrackInterval(path.length_m*.25, path.length_m*.75))


def _plot(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    rows = report["accepted_history"]
    time = np.array([r["years_since_prescribed_release"] for r in rows])
    axes[0, 0].plot(time, [r["max_opening_m"] for r in rows], label="Раскрытие")
    axes[0, 0].plot(time, [r["max_slip_m"] for r in rows], label="Сдвиг")
    axes[0, 0].axvline(report["old_saved_state"]["years_since_prescribed_release"], ls=":", c="gray", label="Прежняя остановка")
    axes[0, 0].set(xlabel="Лет после заданного освобождения", ylabel="Метры", title="Продолжение движения берегов")
    for key, cap, label in (("motion_edge_fraction", .02, "Старый глобальный предел"),
        ("constitutive_added_strain", .005, "Конститутивная деформация"),
        ("linear_strain_error", 1e-4, "Ошибка линейной геометрии"),
        ("relative_contact_jump_error", .02, "Ошибка контактной системы координат")):
        axes[0, 1].semilogy(time, [max(r[key]/cap, 1e-10) for r in rows], label=label)
    axes[0, 1].axhline(1., c="black", ls=":")
    axes[0, 1].set(xlabel="Лет после заданного освобождения", ylabel="Доля предела", title="Локальные ограничения остаются включены")
    axes[1, 0].plot(time, [r["fully_broken_cohort_count"] for r in rows], label="Полностью потеряли сцепление")
    axes[1, 0].plot(time, [r["damaged_cohort_count"] for r in rows], label="Есть контактное повреждение")
    axes[1, 0].set(xlabel="Лет после заданного освобождения", ylabel="Контактные точки", title="История контакта сохраняется")
    for key, label in (("displacement_relative_to_finest", "Перемещения"),
                       ("friction_work_relative_to_finest", "Работа трения"),
                       ("fracture_work_relative_to_finest", "Работа разрушения")):
        coarse = report["timestep_comparison"][:-1]
        axes[1, 1].loglog([r["max_step_years"] for r in coarse],
            [max(r[key], 1e-12) for r in coarse], "o-", label=label)
    axes[1, 1].set(xlabel="Максимальный шаг, лет", ylabel="Относительное отличие от шага 1,25 года",
                    title=f"Сравнение на {report['convergence_duration_years']:g}-м году")
    for axis in axes.ravel():
        axis.legend(fontsize=8)
        axis.grid(alpha=.25)
    final = report["milestones"][-1]
    stop = final["stopped_reason"] or "запрошенное время достигнуто"
    fig.suptitle("5120 исходных ячеек: проверка локальных ограничений геометрии\n"
        f"Заданное освобождение; температура и орбита фиксированы. Итог: {final['years_since_prescribed_release']:.3f} года, {stop}", fontsize=11)
    fig.savefig(output/"validation.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    code_files = [Path(__file__), ROOT/"analysis/genesis_path_dynamics_validation.py",
        ROOT/"analysis/genesis_path_geometry_audit.py"]+[ROOT/"tectonics"/name for name in (
        "genesis_path_geometry.py", "genesis_path_dynamics.py", "genesis_path_basis.py",
        "genesis_path_mesh.py", "genesis_path_material.py", "genesis_contact_geometry.py",
        "genesis_contact_growth.py", "genesis_contact_law.py", "genesis_coupled.py")]
    code_hashes = {str(path.relative_to(ROOT)): _hash(path) for path in code_files}
    source_files = [SOURCE, TRACE, OLD_OUTPUT/"released_mechanics_checkpoint.npz",
                    OLD_OUTPUT/"released_loading_context.npz"]
    source_hashes = {str(path.relative_to(ROOT)): _hash(path) for path in source_files}
    coupled = CoupledModel(SOURCE.read_bytes(), source_path=str(SOURCE))
    source = coupled.source.source_state
    source_arrays = {item.name: _array_hash(getattr(source, item.name)) for item in fields(source)
                     if isinstance(getattr(source, item.name), np.ndarray)}
    with np.load(TRACE, allow_pickle=False) as data:
        path = ReferenceCrackPath(data["points_xyz"], source.radius_km)
    inserted = insert_crack_path(coupled.original_mesh, path,
        front_coordinates_m=path.length_m*np.array([.25, .5, .75]))
    basis = EmbeddedPathBasis(coupled.original_mesh, inserted, coupled.radius_m,
                              coupled.source_model.p.poisson_ratio)
    model = _make_model(basis, coupled, local=True, record=True)
    plain = _make_model(basis, coupled, local=True)
    legacy = _make_model(basis, coupled, local=False)
    factory, loading_arrays = _loading(coupled, model)
    with np.load(OLD_OUTPUT/"released_loading_context.npz", allow_pickle=False) as data:
        loading_identical = all(np.array_equal(value, data[name]) for name, value in loading_arrays.items())
    np.savez_compressed(output/"loading_context.npz", **loading_arrays,
        metadata=np.array(json.dumps({"format": "isothermal_fixed_phase_mechanics_loading",
            "young_modulus_pa": coupled.source_model.p.young_modulus_pa,
            "global_thermal_orbit_time_myr": source.time_myr, "global_clocks_frozen": True})))
    old_saved = legacy.load_state(OLD_OUTPUT/"released_mechanics_checkpoint.npz")
    started = perf_counter()
    release = _fresh_release(model, coupled, factory)
    model.accepted.clear()
    state, milestones = release, []
    for years in (1., 10., 100., 200., 500.):
        state = model.advance(state, release.elapsed_years+years, factory, max_step_years=10.)
        row = _row(model, state, release.elapsed_years)
        row["requested_years_since_release"] = years
        milestones.append(row)
        print(json.dumps({"stage": "milestone", **row}, allow_nan=False), flush=True)
        if state.stopped_reason:
            break
    main_seconds = perf_counter()-started
    model.save_state(output/"local_final_checkpoint.npz", state)
    final_audit, final_fields = audit(model, state)
    np.savez_compressed(output/"local_final_geometry.npz", **final_fields)
    # Independent traction evaluation at the final accepted interval compares
    # two input jump frames with the same old history; no correction feeds back.
    before, after = model.last_before, model.last_after
    dt = after.elapsed_years-before.elapsed_years
    water_trace = np.repeat(np.mean(loading_arrays["water_access"][basis.topology.seam_faces], axis=1), 2)
    _, after_fields = audit(model, after)
    linear, actual = after_fields["linear_jump_m"], after_fields["mean_corotated_actual_jump_m"]
    c0, _, _ = evaluate_cohorts(before.cohorts, linear[:, 0], linear[:, 1], dt*SECONDS_PER_YEAR, water_trace, model.law_parameters)
    c1, _, _ = evaluate_cohorts(before.cohorts, actual[:, 0], actual[:, 1], dt*SECONDS_PER_YEAR, water_trace, model.law_parameters)
    traction = {"diagnostic_only_no_history_or_force_feedback": True,
        "same_history_linear_response_matches_committed": bool(np.allclose(c0.traction_pa, after.cohorts.traction_pa, rtol=0, atol=1e-6)),
        "traction_relative_difference": _relative(c0.traction_pa, c1.traction_pa),
        "max_normal_traction_difference_pa": float(np.max(np.abs(c1.traction_pa[:, 0]-c0.traction_pa[:, 0]))),
        "max_shear_traction_difference_pa": float(np.max(np.abs(c1.traction_pa[:, 1]-c0.traction_pa[:, 1]))),
        "interval_years": dt}
    # Replaying accepted intervals isolates the changed admissibility rule from
    # output cadence or adaptation. Mechanical state/ledgers must match exactly.
    legacy_factory, _ = _loading(coupled, legacy)
    legacy_state = _fresh_release(legacy, coupled, legacy_factory)
    replay_exact, replayed, first_rejection = True, 0, None
    for interval, expected in model.accepted:
        try:
            candidate = legacy.trial(legacy_state, legacy_factory(legacy_state, interval))
        except _PathRetry as exc:
            first_rejection = {"reason": str(exc), "attempted_dt_years": interval,
                "from_years_since_release": legacy_state.elapsed_years-release.elapsed_years,
                "candidate_years_since_release": expected.elapsed_years-release.elapsed_years}
            break
        # Failed attempts do not change mechanical arrays, but do change counts.
        from dataclasses import replace
        replay_exact &= _exact(replace(candidate, rejected_steps=expected.rejected_steps), expected)
        legacy_state, replayed = candidate, replayed+1
    history = [_row(model, release, release.elapsed_years)]+[
        _row(model, item, release.elapsed_years) for _, item in model.accepted]
    # Main controller still records its final rejected attempts in the milestone.
    history[-1] = _row(model, state, release.elapsed_years)
    print(json.dumps({"stage": "legacy_replay", "accepted_intervals": replayed,
        "exact_except_rejected_count": bool(replay_exact), "first_rejection": first_rejection}), flush=True)
    maximum_common = min(100., np.floor((state.elapsed_years-release.elapsed_years)/10)*10)
    if maximum_common < 20:
        raise RuntimeError("Not enough admissible common time for planned comparison")
    convergence, convergence_states = [], []
    convergence_started = perf_counter()
    for maximum in (5., 2.5, 1.25):
        current = plain.advance(release, release.elapsed_years+maximum_common, factory, max_step_years=maximum)
        if current.stopped_reason:
            raise RuntimeError("Convergence common target not reached: "+current.stopped_reason)
        plain.save_state(output/f"convergence_{str(maximum).replace('.', 'p')}.npz", current)
        convergence_states.append(current)
        convergence.append({"max_step_years": maximum, **_row(plain, current, release.elapsed_years)})
        print(json.dumps({"stage": "timestep", "max_step": maximum, "reached": maximum_common,
            "opening": convergence[-1]["max_opening_m"]}), flush=True)
    finest = convergence_states[-1]
    for row, current in zip(convergence, convergence_states):
        row.update(same_final_time_as_finest=current.elapsed_years == finest.elapsed_years,
            displacement_relative_to_finest=_relative(current.displacement_m, finest.displacement_m),
            enrichment_relative_to_finest=_relative(current.displacement_m[basis.nparent:], finest.displacement_m[basis.nparent:]),
            elastic_relative_to_finest=_relative(current.elastic_strain, finest.elastic_strain),
            friction_work_relative_to_finest=_relative(current.cohorts.friction_work_j, finest.cohorts.friction_work_j),
            fracture_work_relative_to_finest=_relative(current.cohorts.fracture_work_j, finest.cohorts.fracture_work_j))
    halfway = plain.advance(release, release.elapsed_years+10., factory, max_step_years=5.)
    plain.save_state(output/"restart_10_years.npz", halfway)
    restored = plain.load_state(output/"restart_10_years.npz")
    direct = plain.advance(halfway, release.elapsed_years+20., factory, max_step_years=5.)
    resumed = plain.advance(restored, release.elapsed_years+20., factory, max_step_years=5.)
    plain.save_state(output/"restart_20_years.npz", resumed)
    plain.save_state(output/"restart_near_limit.npz", before)
    restored_near = plain.load_state(output/"restart_near_limit.npz")
    direct_near = plain.advance(before, release.elapsed_years+500., factory, max_step_years=10.)
    resumed_near = plain.advance(restored_near, release.elapsed_years+500., factory, max_step_years=10.)
    plain.save_state(output/"restart_limit.npz", resumed_near)
    rejected_policies = {}
    for name, owner, file in (("legacy_rejects_local", legacy, output/"restart_10_years.npz"),
        ("local_rejects_legacy", plain, OLD_OUTPUT/"released_mechanics_checkpoint.npz")):
        try:
            owner.load_state(file)
        except ValueError as exc:
            rejected_policies[name] = "different geometry or parameters" in str(exc)
        else:
            rejected_policies[name] = False
    final_roundtrip = _exact(state, plain.load_state(output/"local_final_checkpoint.npz"))
    sources_unchanged = all(_hash(ROOT/name) == digest for name, digest in source_hashes.items())
    source_arrays_unchanged = all(_array_hash(getattr(source, name)) == digest for name, digest in source_arrays.items())
    code_unchanged = all(_hash(ROOT/name) == digest for name, digest in code_hashes.items())
    checks = {"original_loading_identical": loading_identical, "sources_unchanged": sources_unchanged,
        "source_arrays_unchanged": source_arrays_unchanged, "code_unchanged": code_unchanged,
        "mechanical_trajectory_unchanged_before_old_guard": bool(replay_exact and replayed > 0),
        "passed_old_stop_without_reference_reset": state.elapsed_years > old_saved.elapsed_years,
        "early_restart_exact": _exact(halfway, restored) and _exact(direct, resumed),
        "near_limit_restart_exact": _exact(before, restored_near) and _exact(direct_near, resumed_near),
        "final_checkpoint_exact": final_roundtrip, "policy_fingerprint_isolation": all(rejected_policies.values()),
        "common_convergence_time_reached": all(r["same_final_time_as_finest"] for r in convergence),
        "finite_result": all(np.isfinite(a).all() for a in final_fields.values()),
        "traction_replay_matches": traction["same_history_linear_response_matches_committed"]}
    report = {"scope": "explicit_cohesive_release_fixed_temperature_phase_bulk_damage_water_and_orbit",
        "source_global_time_myr": source.time_myr, "mechanical_clock_is_elapsed_years_only": True,
        "geometry_parameters": asdict(model.geometry_parameters), "geometry_policy_opt_in": True,
        "path_length_km": path.length_m/1000, "source_cells": coupled.original_mesh.cell_count,
        "child_cells": basis.topology.mesh.cell_count, "source_sha256": source_hashes,
        "source_array_sha256": source_arrays, "code_sha256": code_hashes,
        "mechanics_fingerprint": model.fingerprint, "legacy_fingerprint": legacy.fingerprint,
        "main_run_wall_seconds": main_seconds, "comparison_wall_seconds": perf_counter()-convergence_started,
        "old_saved_state": _row(legacy, old_saved, release.elapsed_years),
        "milestones": milestones, "accepted_history": history,
        "legacy_replay": {"accepted_interval_count": replayed, "exact_except_rejected_count": bool(replay_exact),
                           "first_rejected_interval": first_rejection},
        "final_geometry_audit": final_audit, "contact_frame_traction_diagnostic": traction,
        "convergence_duration_years": maximum_common, "timestep_comparison": convergence,
        "restart": {"early_exact": checks["early_restart_exact"], "near_limit_exact": checks["near_limit_restart_exact"],
            "near_limit_start_years_after_release": before.elapsed_years-release.elapsed_years,
            "near_limit_result": _row(plain, resumed_near, release.elapsed_years),
            "fingerprint_policy_rejection": rejected_policies}, "checks": checks,
        "artifact_sha256": {p.name: _hash(p) for p in output.glob("*.npz")}}
    (output/"validation.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    _plot(report, output)
    print(json.dumps({"stage": "final", "checks": checks, "final": milestones[-1],
        "convergence": [{k:r[k] for k in ("max_step_years", "displacement_relative_to_finest", "friction_work_relative_to_finest", "fracture_work_relative_to_finest")} for r in convergence],
        "wall_seconds": perf_counter()-started}, allow_nan=False), flush=True)
    if not all(checks.values()):
        raise SystemExit("Validation checks failed")


if __name__ == "__main__":
    main()
