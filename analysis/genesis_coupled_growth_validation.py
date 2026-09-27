"""Independent timestep/layer refinement and exact-restart audit for growing contacts."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.genesis_coupled_validation import _fingerprint, sample
from tectonics.genesis_coupled import (
    COUPLED_VERSION, CoupledModel, CoupledParameters, load_coupled_checkpoint,
    save_coupled_checkpoint,
)


CASES = ((10., 1.), (5., 1.), (10., .5), (10., .25))
METRICS = (
    "max_opening_m", "max_abs_jump_m", "max_penetration_m",
    "mean_lid_thickness_km", "mean_damage", "mean_water_access",
    "surface_temperature_k", "mantle_temperature_k", "eccentricity",
    "max_added_strain", "max_elastic_strain", "seam_count", "new_seam_count",
    "component_count", "interface_cohort_count", "interface_pending_depth_m",
    "interface_represented_area_m2", "interface_born_unbonded_area_m2",
    "interface_birth_energy_j", "friction_work_j", "viscous_work_j",
    "fracture_work_j", "drag_work_j", "mechanical_remainder_relative_work",
)
RESIDUALS = (
    "relative_mass_residual", "relative_global_energy_residual",
    "relative_column_energy_residual", "orbit_heat_transfer_relative_residual",
    "equilibrium_residual",
)
IMPLEMENTATION_PATHS = (
    "tectonics/genesis_coupled.py", "tectonics/genesis_coupled_thermal.py",
    "tectonics/genesis_coupled_topology.py", "tectonics/genesis_coupled_checkpoint.py",
    "tectonics/genesis_contact_growth.py", "tectonics/genesis_contact_law.py",
    "analysis/genesis_coupled_validation.py", "analysis/genesis_coupled_growth_validation.py",
)


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _hashes():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in IMPLEMENTATION_PATHS}


def _digest(state, thermal, orbit):
    return hashlib.sha256(_fingerprint((state, thermal, orbit))).hexdigest()


def _sample(model, state, thermal, orbit):
    row = sample(model, state, thermal, orbit)
    cohorts = state.cohorts
    born_later = cohorts.birth_time_myr > model.source_time_myr
    row.update(
        post_initial_cohort_count=int(np.count_nonzero(born_later)),
        unbonded_cohort_count=int(np.count_nonzero(~cohorts.bonded)),
        post_initial_cohort_area_m2=float(cohorts.area_ref_m2[born_later].sum()),
        max_cohort_depth_m=float(np.max(cohorts.z_hi_ref_m-cohorts.z_lo_ref_m))
            if len(cohorts.trace_index) else 0.,
    )
    return row


def simulate(source, source_path, cap, layer, duration, interval, output, *, restart=False):
    started = perf_counter()
    name = f"step_{cap:g}yr_layer_{layer:g}m"
    case_dir = output/name
    case_dir.mkdir()
    parameters = CoupledParameters(max_step_years=cap, growth_layer_m=layer)
    model = CoupledModel(source, parameters, source_path=str(source_path.resolve()))
    state = model.initial()
    thermal, orbit = model.source.thermal_state, model.source.orbit
    history = [_sample(model, state, thermal, orbit)]
    count = round(duration/interval)
    split_index = count//2
    split_path = None
    end = model.source_time_myr+duration/1e6
    _json(case_dir/"parameters.json", {
        "coupled": asdict(parameters), "contact": asdict(model.contact_parameters),
        "law": asdict(model.law_parameters), "duration_years": duration,
        "output_interval_years": interval,
    })
    with (case_dir/"history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerow(history[0])
        for index in range(1, count+1):
            target = min(model.source_time_myr+index*interval/1e6, end)
            state, thermal, orbit, _ = model.step(
                state, thermal, orbit, target, max_step_myr=interval/1e6)
            row = _sample(model, state, thermal, orbit)
            history.append(row)
            writer.writerow(row)
            handle.flush()
            print(f"{name}: {row['elapsed_years']:.6f} yr; accepted={state.accepted_steps}; "
                  f"cohorts={row['interface_cohort_count']}; opening={row['max_opening_m']:.6g} m; "
                  f"status={state.stopped_reason or 'running'}", flush=True)
            if restart and index == split_index and not state.stopped_reason:
                split_path = case_dir/"split_checkpoint.npz"
                save_coupled_checkpoint(split_path, model, state, thermal, orbit)
            if state.stopped_reason:
                break
    save_coupled_checkpoint(case_dir/"coupled_checkpoint.npz", model, state, thermal, orbit)
    run_seconds = perf_counter()-started
    state_hash = _digest(state, thermal, orbit)
    resumed_audit = None
    if split_path is not None:
        restart_started = perf_counter()
        fresh, resumed, fresh_thermal, fresh_orbit = load_coupled_checkpoint(split_path)
        # A restart uses the saved model without replacing any numerical parameters.
        assert fresh.parameters == model.parameters
        assert fresh.contact_parameters == model.contact_parameters
        assert fresh.law_parameters == model.law_parameters
        assert fresh.source_hash == model.source_hash
        split_time = resumed.time_myr
        for index in range(split_index+1, len(history)):
            target = min(fresh.source_time_myr+index*interval/1e6, end)
            resumed, fresh_thermal, fresh_orbit, _ = fresh.step(
                resumed, fresh_thermal, fresh_orbit, target, max_step_myr=interval/1e6)
            if resumed.stopped_reason:
                break
        resumed_hash = _digest(resumed, fresh_thermal, fresh_orbit)
        resumed_audit = {
            "split_elapsed_years": (split_time-model.source_time_myr)*1e6,
            "bitwise_equal_all_nested_state": _fingerprint((state, thermal, orbit)) ==
                _fingerprint((resumed, fresh_thermal, fresh_orbit)),
            "parameters_preserved": True,
            "continuous_state_sha256": state_hash,
            "resumed_state_sha256": resumed_hash,
            "wall_seconds": perf_counter()-restart_started,
        }
        save_coupled_checkpoint(case_dir/"resumed_checkpoint.npz", fresh, resumed, fresh_thermal, fresh_orbit)
        print(f"{name}: restart exact={resumed_audit['bitwise_equal_all_nested_state']}", flush=True)
    result = {
        "parameters": asdict(parameters), "requested_duration_years": duration,
        "output_interval_years": interval, "status": state.stopped_reason or "completed",
        "wall_seconds": run_seconds, "final_state_sha256": state_hash,
        "history": history, "final": history[-1], "restart": resumed_audit,
        "cut_edges": state.cut_edges.tolist(),
        "maximum_sampled_absolute_residuals": {
            key: max(abs(row[key]) for row in history) for key in RESIDUALS},
    }
    _json(case_dir/"result.json", result)
    return name, result, state


def compare(first_name, first, first_state, second_name, second, second_state):
    result = {
        "reference": first_name, "refined": second_name,
        "same_final_time": first["final"]["time_myr"] == second["final"]["time_myr"],
        "identical_cut_edges": first["cut_edges"] == second["cut_edges"],
        "identical_accepted_step_count": first_state.accepted_steps == second_state.accepted_steps,
        "cut_edge_symmetric_difference_count": len(set(map(tuple, first["cut_edges"])) ^
            set(map(tuple, second["cut_edges"]))),
    }
    if result["same_final_time"]:
        a, b = first["final"], second["final"]
        result["final_differences"] = {key: {
            "signed_difference": b[key]-a[key],
            "relative_difference": (b[key]-a[key])/abs(a[key]) if a[key] else None,
        } for key in METRICS}
        result["material_field_max_absolute_difference"] = {
            key: float(np.max(np.abs(getattr(first_state, key)-getattr(second_state, key))))
            for key in ("column_enthalpy", "elastic_strain", "damage", "water_access", "velocity_km_myr")}
        if result["identical_cut_edges"]:
            result["max_nodal_displacement_difference_m"] = float(np.max(np.abs(
                first_state.contact.displacement_m-second_state.contact.displacement_m)))
    return result


def plot(cases, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = (("max_opening_m", "Раскрытие, м"),
              ("interface_cohort_count", "Число участков контакта"),
              ("interface_pending_depth_m", "Ожидающая контактная глубина, м"),
              ("friction_work_j", "Накопленная работа трения, Дж"))
    with plt.rc_context({"font.size": 10, "figure.facecolor": "#f8fafc"}):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        try:
            for ax, (key, title) in zip(axes.flat, panels):
                for case in cases.values():
                    p = case["parameters"]
                    label = f"Шаг ≤ {p['max_step_years']:g} лет; слой {p['growth_layer_m']:g} м"
                    ax.plot([r["elapsed_years"] for r in case["history"]],
                            [r[key] for r in case["history"]], marker=".", label=label)
                ax.set(title=title, xlabel="Физическое продолжение, лет")
                ax.grid(alpha=.2)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .93), ncol=2, frameon=False)
            fig.suptitle("Растущий контакт: чувствительность к шагу и толщине слоя", y=.98)
            fig.text(.5, .015, "Одна сетка и один сценарий. Механическая работа пока не возвращается в тепловые резервуары.",
                     ha="center", fontsize=9)
            fig.subplots_adjust(left=.085, right=.97, top=.79, bottom=.10, hspace=.4, wspace=.29)
            fig.savefig(output, dpi=150)
        finally:
            plt.close(fig)


def write_report(report, path):
    cases = report["cases"]
    labels = [f"{c['parameters']['max_step_years']:g} лет / {c['parameters']['growth_layer_m']:g} м"
              for c in cases.values()]
    first = next(iter(cases.values()))
    rows = ["# Проверка роста контактов версии 0.2", "",
        f"Исходный возраст: {first['history'][0]['time_myr']:g} млн лет. "
        f"Продолжение: {first['requested_duration_years']:g} лет; общие границы вывода через "
        f"{first['output_interval_years']:g} лет. Сетка и исходное состояние одинаковы.", "",
        "| Показатель / шаг и контактный слой | " + " | ".join(labels) + " |",
        "|---|" + "---:|"*len(cases)]
    for key, title in (
        ("accepted_steps", "Принятые шаги"), ("rejected_steps", "Отклонённые попытки"),
        ("max_opening_m", "Раскрытие, м"), ("max_abs_jump_m", "Сдвиг, м"),
        ("max_penetration_m", "Штрафное проникновение, м"), ("seam_count", "Разрезанные рёбра"),
        ("component_count", "Связные области"), ("interface_cohort_count", "Участки контакта"),
        ("post_initial_cohort_count", "Участки, рождённые после начала"),
        ("unbonded_cohort_count", "Участки без сцепления при рождении"),
        ("interface_pending_depth_m", "Максимальная ожидающая глубина, м"),
        ("interface_area_change_fraction", "Геометрическое изменение опорной площади"),
        ("interface_birth_energy_j", "Энергия сжатия новых участков, Дж"),
        ("friction_work_j", "Работа трения, Дж"), ("viscous_work_j", "Вязкие потери, Дж"),
        ("fracture_work_j", "Работа разрушения, Дж"),
        ("mechanical_remainder_relative_work", "Механическая невязка / масштаб работы"),
    ):
        rows.append("| " + title + " | " + " | ".join(f"{c['final'][key]:.9g}" for c in cases.values()) + " |")
    rows.append("| Время основного прогона, с | " + " | ".join(f"{c['wall_seconds']:.3f}" for c in cases.values()) + " |")
    rows += ["", "Состояния завершения: " + ", ".join(
        f"`{name}` — `{case['status']}`" for name, case in cases.items()) + ".", "",
        "Изменения при независимом уточнении времени и разбиения контакта:", ""]
    for name, comparison in report["comparisons"].items():
        if not comparison["same_final_time"]:
            rows.append(f"- `{name}`: конечные моменты различаются; прямое сравнение неприменимо.")
            continue
        d = comparison["final_differences"]
        pct = lambda key: f"{100*d[key]['relative_difference']:+.6g}%" if d[key]["relative_difference"] is not None else "н/д"
        rows.append(f"- `{name}`: раскрытие {pct('max_opening_m')}; сдвиг {pct('max_abs_jump_m')}; "
            f"трение {pct('friction_work_j')}; вязкие потери {pct('viscous_work_j')}; "
            f"разрушение {pct('fracture_work_j')}. Разрезы совпадают: {comparison['identical_cut_edges']}.")
    rows += ["", "Максимальные абсолютные относительные невязки на сохранённых отсчётах:", ""]
    for key in RESIDUALS:
        rows.append(f"- `{key}`: " + ", ".join(f"{label}: {case['maximum_sampled_absolute_residuals'][key]:.4g}"
            for label, case in zip(labels, cases.values())) + ".")
    restart = first["restart"]
    if restart:
        rows += ["", f"Сохранение на {restart['split_elapsed_years']:.6g} годах и продолжение с неизменными "
            f"параметрами: побитовое совпадение всех вложенных материальных, контактных, тепловых и орбитальных "
            f"состояний — **{restart['bitwise_equal_all_nested_state']}**. "
            "SHA-256 непрерывного и возобновлённого состояния сохранены в JSON."]
    rows += ["", "Это измерение чувствительности одного сценария на одной сетке. Тепловой шаг предшествует "
        "механическому; рождение участка использует геометрию начала пробного шага, поэтому расщепление "
        "имеет первый порядок. Заметное изменение работы трения при уточнении не следует считать "
        "сошедшейся диссипацией. Малые изменения кинематики сами по себе не доказывают сходимость всей модели.", "",
        "Массовый, глобальный тепловой, столбцовый тепловой и механический учёт остаются различными "
        "проверками. Механическая невязка нормирована на |работа приращения оболочки| + |внешняя работа|; "
        "механические потери не возвращаются в тепло. Связные области ещё не объявляются зрелыми плитами.", "",
        "![Сравнение вариантов](comparison.png)", ""]
    path.write_text("\n".join(rows), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--duration-years", type=float, default=2000.)
    parser.add_argument("--output-step-years", type=float, default=500.)
    args = parser.parse_args(argv)
    if not all(math.isfinite(v) and v > 0 for v in (args.duration_years, args.output_step_years)):
        parser.error("Durations must be finite and positive")
    count = args.duration_years/args.output_step_years
    if not math.isclose(count, round(count), rel_tol=0., abs_tol=1e-10) or round(count) < 2 or round(count) % 2:
        parser.error("Duration must contain an even positive number of output intervals for the halfway restart")
    if args.output.exists():
        parser.error("Output must be a new directory")
    source = args.checkpoint.read_bytes()
    implementation = _hashes()
    args.output.mkdir(parents=True)
    started_at = datetime.now(timezone.utc).isoformat()
    started = perf_counter()
    cases, states = {}, {}
    for index, (cap, layer) in enumerate(CASES):
        name, result, state = simulate(source, args.checkpoint, cap, layer, args.duration_years,
            args.output_step_years, args.output, restart=index == 0)
        cases[name], states[name] = result, state
    names = list(cases)
    pairings = {"time_10_to_5_years_at_1m": (0, 1),
                "layer_1_to_0.5m_at_10years": (0, 2),
                "layer_0.5_to_0.25m_at_10years": (2, 3)}
    after_hashes = _hashes()
    report = {
        "model": COUPLED_VERSION, "started_at_utc": started_at,
        "wall_seconds": perf_counter()-started,
        "source": {"path": str(args.checkpoint.resolve()), "sha256": hashlib.sha256(source).hexdigest()},
        "implementation_sha256_at_start": implementation,
        "implementation_sha256_at_end": after_hashes,
        "implementation_files_unchanged_during_audit": implementation == after_hashes,
        "cases": cases,
        "comparisons": {name: compare(names[a], cases[names[a]], states[names[a]],
            names[b], cases[names[b]], states[names[b]]) for name, (a, b) in pairings.items()},
    }
    _json(args.output/"validation.json", report)
    plot(cases, args.output/"comparison.png")
    write_report(report, args.output/"validation_report.md")
    if not implementation == after_hashes:
        print("WARNING: implementation files changed during audit; inspect recorded hashes", flush=True)
    print("COUPLED_GROWTH_VALIDATION_COMPLETE", args.output.resolve(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
