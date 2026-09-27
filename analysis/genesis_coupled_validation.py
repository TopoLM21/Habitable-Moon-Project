"""Physical coupled-clock step sensitivity, conservation and exact restart audit."""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields, is_dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tectonics.genesis_coupled import (
    COUPLED_VERSION, CoupledModel, CoupledParameters, load_coupled_checkpoint,
    save_coupled_checkpoint,
)
from tectonics.genesis_shell import maximum_total_strain


def sample(model, state, thermal, orbit):
    row = model.diagnostics(state, thermal, orbit)
    geometry = model._geometry(state.cut_edges)
    row["max_added_strain"] = maximum_total_strain(geometry._strain(state.contact.displacement_m))
    row["max_elastic_strain"] = maximum_total_strain(state.elastic_strain)
    row["total_column_energy_j"] = float(np.sum(model.layer_mass_kg*state.column_enthalpy))
    row["tidal_heat_received_j"] = state.tidal_heat_received_j
    row["orbital_dissipated_energy_j"] = orbit.dissipated_energy_j
    row["external_work_j"] = state.contact.external_work_j
    # Scale with actual mechanical throughput; large stored inherited energy
    # would hide discretization losses of the short continuation itself.
    scale = abs(row["bulk_increment_work_j"])+abs(row["external_work_j"])
    row["mechanical_work_scale_j"] = scale
    row["mechanical_remainder_relative_work"] = row["mechanical_energy_remainder_j"]/scale if scale else 0.
    return row


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def _fingerprint(value):
    """Canonical nested state bytes, preserving float signs and array dtype."""
    if is_dataclass(value):
        return b"dataclass:"+b"".join(f.name.encode()+b":"+_fingerprint(getattr(value, f.name)) for f in fields(value))
    if isinstance(value, np.ndarray):
        return b"array:"+str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()
    if isinstance(value, (tuple, list)):
        return b"list:"+b"".join(_fingerprint(item) for item in value)
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def simulate(source, source_path, cap, duration, interval, output, *, resume_audit=False):
    parameters = CoupledParameters(max_step_years=cap)
    model = CoupledModel(source, parameters, source_path=str(source_path.resolve()))
    state = model.initial()
    thermal, orbit = model.source.thermal_state, model.source.orbit
    history = [sample(model, state, thermal, orbit)]
    name = f"max_step_{cap:g}yr"
    end = model.source_time_myr+duration/1e6
    count = math.ceil(duration/interval)
    split = max(1, count//2)
    resume_path, resume_index = None, None
    for index in range(1, count+1):
        target = min(model.source_time_myr+index*interval/1e6, end)
        state, thermal, orbit, _ = model.step(state, thermal, orbit, target, max_step_myr=interval/1e6)
        row = sample(model, state, thermal, orbit)
        history.append(row)
        print(f"{name}: {row['elapsed_years']:.6f} yr; accepted={state.accepted_steps}; "
              f"rejected={state.rejected_steps}; opening={row['max_opening_m']:.6g} m; "
              f"new cuts={state.new_seam_count}; status={state.stopped_reason or 'running'}", flush=True)
        if resume_audit and index == split and not state.stopped_reason:
            resume_path = output/f"{name}_split_checkpoint.npz"
            save_coupled_checkpoint(resume_path, model, state, thermal, orbit)
            resume_index = index
        if state.stopped_reason:
            break
    save_coupled_checkpoint(output/f"{name}_checkpoint.npz", model, state, thermal, orbit)
    restart = None
    if resume_path is not None:
        fresh, resumed, fresh_thermal, fresh_orbit = load_coupled_checkpoint(resume_path)
        restart_start = resumed.time_myr
        for index in range(resume_index+1, len(history)):
            target = min(fresh.source_time_myr+index*interval/1e6, end)
            resumed, fresh_thermal, fresh_orbit, _ = fresh.step(
                resumed, fresh_thermal, fresh_orbit, target, max_step_myr=interval/1e6)
            if resumed.stopped_reason:
                break
        continuous = _fingerprint((state, thermal, orbit))
        reloaded = _fingerprint((resumed, fresh_thermal, fresh_orbit))
        restart = {"split_elapsed_years": (restart_start-model.source_time_myr)*1e6,
            "bitwise_equal_material_contact_thermal_orbit_state": continuous == reloaded,
            "continuous_state_sha256": hashlib.sha256(continuous).hexdigest(),
            "resumed_state_sha256": hashlib.sha256(reloaded).hexdigest()}
        save_coupled_checkpoint(output/f"{name}_resumed_checkpoint.npz", fresh, resumed, fresh_thermal, fresh_orbit)
    result = {"parameters": asdict(parameters), "requested_output_interval_years": interval,
        "requested_duration_years": duration, "status": state.stopped_reason or "completed",
        "history": history, "final": history[-1], "restart": restart,
        "cut_edges": state.cut_edges.tolist(),
        "maximum_sampled_residuals": {key: max(abs(row[key]) for row in history) for key in (
            "relative_mass_residual", "relative_global_energy_residual", "relative_column_energy_residual",
            "orbit_heat_transfer_relative_residual", "equilibrium_residual")}}
    _json(output/f"{name}.json", result)
    return name, result, (model, state, thermal, orbit)


def compare(cases, states):
    name, reference = next(iter(cases.items()))
    metrics = ("surface_temperature_k", "mantle_temperature_k", "mean_lid_thickness_km",
        "mean_damage", "mean_water_access", "eccentricity", "max_opening_m", "max_abs_jump_m",
        "max_penetration_m", "max_added_strain", "max_elastic_strain", "seam_count", "new_seam_count",
        "component_count", "friction_work_j", "viscous_work_j", "fracture_work_j", "drag_work_j",
        "maxwell_relaxation_release_j", "mechanical_remainder_relative_work")
    results = {}
    for other, case in cases.items():
        if other == name:
            continue
        match = abs(case["final"]["time_myr"]-reference["final"]["time_myr"]) < 1e-13
        result = {"same_final_time": match,
            "identical_cut_edges": case["cut_edges"] == reference["cut_edges"],
            "cut_edge_symmetric_difference_count": len(set(map(tuple, case["cut_edges"]))^
                                                          set(map(tuple, reference["cut_edges"]))) }
        if match:
            result["final_differences"] = {key: {
                "signed_difference": case["final"][key]-reference["final"][key],
                "relative_difference": ((case["final"][key]-reference["final"][key])/abs(reference["final"][key])
                                        if reference["final"][key] else None)} for key in metrics}
            a, b = states[name][1], states[other][1]
            result["material_field_max_absolute_difference"] = {field: float(np.max(np.abs(
                getattr(a, field)-getattr(b, field)))) for field in (
                "column_enthalpy", "elastic_strain", "damage", "water_access", "velocity_km_myr")}
            if result["identical_cut_edges"]:
                result["maximum_nodal_displacement_difference_m"] = float(np.max(np.abs(
                    a.contact.displacement_m-b.contact.displacement_m)))
            strip_counts = lambda s: replace(s, rejected_steps=0,
                contact=replace(s.contact, rejected_steps=0))
            result["bitwise_equal_state_except_rejected_step_counters"] = (
                _fingerprint((strip_counts(a), *states[name][2:])) ==
                _fingerprint((strip_counts(b), *states[other][2:])))
            result["identical_accepted_step_count"] = a.accepted_steps == b.accepted_steps
        results[other] = result
    successive = {}
    case_items = list(cases.items())
    for (first_name, first), (second_name, second) in zip(case_items, case_items[1:]):
        if abs(first["final"]["time_myr"]-second["final"]["time_myr"]) < 1e-13:
            a, b = first["final"], second["final"]
            successive[second_name+"_versus_"+first_name] = {key: {
                "signed_difference": b[key]-a[key],
                "relative_difference": (b[key]-a[key])/abs(a[key]) if a[key] else None}
                for key in metrics}
    return {"reference": name, "differences": results, "successive_refinements": successive,
        "interpretation": "Identical output boundaries and thermal step cap. Adaptive displacement/damage limits may reduce both requested caps to the same accepted subdivision; inspect accepted counts and last steps. This is temporal sensitivity, not mesh convergence or physical calibration."}


def plot(cases, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = (("max_opening_m", "Раскрытие берегов, м", 1),
              ("max_abs_jump_m", "Относительный сдвиг, м", 1),
              ("mean_lid_thickness_km", "Средняя толщина оболочки, км", 1),
              ("mechanical_remainder_relative_work", "Механическая невязка, % работы", 100))
    with plt.rc_context({"font.size": 10, "figure.facecolor": "#f8fafc"}):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
        try:
            for ax, (key, title, multiplier) in zip(axes.flat, panels):
                for name, case in cases.items():
                    ax.plot([r["elapsed_years"] for r in case["history"]],
                            [r[key]*multiplier for r in case["history"]],
                            label=f"Максимальный шаг {case['parameters']['max_step_years']:g} лет")
                ax.set(xlabel="Физическое время после исходного состояния, лет", title=title)
                ax.grid(alpha=.2)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .925), frameon=False, ncol=2)
            fig.suptitle("Совместное охлаждение и контакт: чувствительность к шагу", y=.98)
            fig.text(.5, .015, "Тепловой, орбитальный и механический возраст совпадают. Тепловые и механические бюджеты остаются раздельными.",
                     ha="center", fontsize=9)
            fig.subplots_adjust(left=.085, right=.975, top=.78, bottom=.105, hspace=.42, wspace=.25)
            fig.savefig(output, dpi=150)
        finally:
            plt.close(fig)


def write_report(report, path):
    cases = report["cases"]
    baseline = next(iter(cases.values()))
    headers = [f"{case['parameters']['max_step_years']:g} лет" for case in cases.values()]
    rows = ["# Проверка совместного охлаждения и контакта", "",
        f"Исходный возраст: {baseline['history'][0]['time_myr']:g} млн лет. "
        f"Длительность: {baseline['requested_duration_years']:g} физических лет. "
        f"Общие границы вывода: каждые {baseline['requested_output_interval_years']:g} лет.", "",
        "| Показатель / максимальный шаг | "+" | ".join(headers)+" |",
        "|---|"+"---:|"*len(cases)]
    labels = (("accepted_steps", "Принятых шагов"), ("rejected_steps", "Отклонённых попыток"),
        ("max_opening_m", "Максимальное раскрытие, м"), ("max_abs_jump_m", "Максимальный сдвиг, м"),
        ("max_penetration_m", "Проникновение штрафного контакта, м"),
        ("seam_count", "Разрезанных рёбер"), ("component_count", "Связных областей оболочки"),
        ("mean_lid_thickness_km", "Средняя толщина оболочки, км"),
        ("surface_temperature_k", "Температура поверхности, К"),
        ("mean_damage", "Среднее повреждение"), ("mean_water_access", "Средняя доступность жидкой воды"),
        ("friction_work_j", "Работа трения, Дж"),
        ("mechanical_remainder_relative_work", "Механическая невязка / масштаб работы"))
    for key, label in labels:
        rows.append("| "+label+" | "+" | ".join(f"{case['final'][key]:.9g}" for case in cases.values())+" |")
    rows += ["", "Тепловой, орбитальный и механический возраст совпадают. "
        "Масштаб механической невязки: |работа приращения оболочки| + |внешняя работа|. "
        "Механические потери не возвращаются в тепловые резервуары; это три отдельных баланса.", "",
        "Максимальные абсолютные относительные невязки по сохранённым временным отсчётам:", ""]
    for key in baseline["maximum_sampled_residuals"]:
        rows.append(f"- `{key}`: "+", ".join(f"{header}: {case['maximum_sampled_residuals'][key]:.4g}"
            for header, case in zip(headers, cases.values()))+".")
    restart = baseline["restart"]
    if restart:
        rows += ["", f"Продолжение после сохранения на {restart['split_elapsed_years']:.6g} годах: "
            f"побитовое совпадение всех состояний = **{restart['bitwise_equal_material_contact_thermal_orbit_state']}**."]
    rows += ["", "Сравнение с первым расчётом:", ""]
    for name, comparison in report["comparison"]["differences"].items():
        if not comparison["same_final_time"]:
            rows.append(f"- `{name}` остановился в другой момент; конечные поля напрямую не сравниваются.")
            continue
        d = comparison["final_differences"]
        percent = lambda key: f"{100*d[key]['relative_difference']:+.6g}%" if d[key]["relative_difference"] is not None else "н/д"
        rows.append(f"- `{name}`: раскрытие {percent('max_opening_m')}; сдвиг {percent('max_abs_jump_m')}; "
            f"работа трения {percent('friction_work_j')}. Совпадение разрезов: {comparison['identical_cut_edges']}. "
            f"Побитовое совпадение состояния без счётчиков отклонённых шагов: "
            f"{comparison['bitwise_equal_state_except_rejected_step_counters']}.")
    rows += ["", "Изменения между последовательными уточнениями шага:", ""]
    for name, d in report["comparison"]["successive_refinements"].items():
        percent = lambda key: f"{100*d[key]['relative_difference']:+.6g}%" if d[key]["relative_difference"] is not None else "н/д"
        rows.append(f"- `{name}`: раскрытие {percent('max_opening_m')}; сдвиг {percent('max_abs_jump_m')}; "
            f"работа трения {percent('friction_work_j')}; вязкие потери {percent('viscous_work_j')}; "
            f"работа раскрытия трещин {percent('fracture_work_j')}.")
    rows += ["", "Равенство двух максимальных шагов не является доказательством сходимости: "
        "ограничитель скачка берегов может привести к одинаковым принятым подшагам. "
        "Различие работы трения при меньшем шаге сохраняется. Проверка относится к временному шагу "
        "на одной сетке; пространственная сходимость и физическая калибровка здесь не установлены. "
        "Связные области оболочки ещё не объявляются зрелыми жёсткими плитами.", "",
        "![Сравнение расчётов](comparison.png)", ""]
    path.write_text("\n".join(rows), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--duration-years", type=float, default=1000.)
    parser.add_argument("--output-step-years", type=float, default=100.)
    parser.add_argument("--steps", nargs="+", type=float, default=[100., 50., 10., 5.])
    args = parser.parse_args(argv)
    if not all(math.isfinite(v) and v > 0 for v in [args.duration_years, args.output_step_years, *args.steps]):
        parser.error("Durations and steps must be finite and positive")
    if len(set(args.steps)) != len(args.steps):
        parser.error("Step caps must be unique")
    if args.output.exists():
        parser.error("Output must be a new directory")
    source = args.checkpoint.read_bytes()
    args.output.mkdir(parents=True)
    cases, states = {}, {}
    for index, cap in enumerate(args.steps):
        name, result, state = simulate(source, args.checkpoint, cap, args.duration_years,
            args.output_step_years, args.output, resume_audit=index == 0)
        cases[name], states[name] = result, state
    report = {"model": COUPLED_VERSION,
        "source": {"path": str(args.checkpoint.resolve()), "sha256": hashlib.sha256(source).hexdigest()},
        "implementation_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [ROOT/"tectonics/genesis_coupled.py", ROOT/"tectonics/genesis_coupled_thermal.py",
                         ROOT/"tectonics/genesis_coupled_topology.py"]},
        "cases": cases, "comparison": compare(cases, states),
        "energy_accounting": "Global enthalpy, material-column enthalpy and mechanical work are distinct ledgers. Mechanical dissipation is not returned as heat. Mechanical remainder is normalized by |bulk increment work| + |external work|."}
    _json(args.output/"validation.json", report)
    plot(cases, args.output/"comparison.png")
    write_report(report, args.output/"validation_report.md")
    print("COUPLED_VALIDATION_COMPLETE", args.output.resolve(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
