"""Audit actual fault-run checkpoints and compare saved physical parameters.

This reads existing experiments; it neither changes loads nor reconstructs
unobserved events from sparse animation frames.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tectonics.genesis_faults import load_fault_checkpoint
from tectonics.genesis_shell import Membrane, principal_tensile


def inspect_run(path):
    path = Path(path).resolve()
    parameters = json.loads((path / "parameters.json").read_text(encoding="utf-8"))
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    checkpoint = path / "fault_checkpoint.npz"
    model, state, thermal, orbit, metadata = load_fault_checkpoint(checkpoint)
    with (path / "shell_history.csv").open(encoding="utf-8", newline="") as handle:
        history = list(csv.DictReader(handle))
    mesh = model.mesh_for(state)
    membrane = Membrane(mesh, model.p.poisson_ratio)
    degradation = model.p.residual_stiffness + (1-model.p.residual_stiffness)*(1-state.damage)**2
    stress = (state.elastic_strain @ membrane.d.T)*(model.p.young_modulus_pa*degradation[:, None])
    tensile = principal_tensile(stress)
    strength = model.p.tensile_strength_pa*(1-(1-model.onset_p.wet_strength_fraction)*state.water_access)
    actual = model.diagnostics(state, thermal, orbit)
    if actual[1]["time_myr"] != summary["shell"]["time_myr"]:
        raise ValueError("Summary and checkpoint refer to different times")
    sections = {"thermal": model.thermal, "shell": model.p, "onset": model.onset_p,
                "tides": model.tides_p, "mobile": model.mobile_p, "faults": model.fault_p}
    for name, value in sections.items():
        if parameters[name] != asdict(value):
            raise ValueError(f"Saved parameter file disagrees with checkpoint: {name}")
    peak_sample = max(history, key=lambda row: float(row["mean_damage"]))
    return {
        "path": str(path), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "parameters": parameters, "status": summary["status"],
        "cell_count": mesh.cell_count, "physical_time_myr": state.time_myr,
        "synchronized_clocks": state.time_myr == thermal.time_myr == orbit.time_myr,
        "first_fracture_time_myr": state.first_fracture_time_myr,
        "active_cells": int(state.fault_active.sum()),
        "ever_activated_cells": int(np.count_nonzero(state.activation_time_myr >= 0)),
        "max_damage": float(state.damage.max()),
        "activation_damage": model.fault_p.activation_damage,
        "activation_persistence_years": model.fault_p.activation_persistence_myr*1e6,
        "max_candidate_age_years": float(state.fault_candidate_age_myr.max())*1e6,
        "max_equivalent_slip_km": float(state.cumulative_shear.max())*model.fault_p.band_width_km,
        "friction_work_j": state.friction_work_j, "viscous_fault_work_j": state.viscous_fault_work_j,
        "final_tensile_strength_mpa_range": [float(strength.min())/1e6, float(strength.max())/1e6],
        "final_max_tensile_to_strength_ratio": float(np.max(tensile/strength)),
        "peak_saved_mean_damage": float(peak_sample["mean_damage"]),
        "peak_saved_mean_damage_time_myr": float(peak_sample["time_myr"]),
        "saved_samples": len(history),
        "all_saved_slip_values_zero": all(float(row["max_equivalent_slip_km"]) == 0 for row in history),
        "final_diagnostics": dict(global_state=actual[0], shell=actual[1], onset=actual[2], orbit=actual[3]),
    }


def differences(a, b):
    result = {}
    for section in ("thermal", "shell", "onset", "tides", "mobile", "faults", "controls"):
        for key in sorted(set(a[section]) | set(b[section])):
            if a[section].get(key) != b[section].get(key):
                result[f"{section}.{key}"] = [a[section].get(key), b[section].get(key)]
    return result


def plot_matched_groups(inspected, output):
    """Compare spatial resolutions only where all other saved settings match."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = {}
    for item in inspected:
        groups.setdefault(item["parameters"]["shell"]["convective_traction_pa"], []).append(item)
    figures = []
    for traction, group in groups.items():
        if len(group) < 2:
            continue
        group.sort(key=lambda item: item["cell_count"])
        if any(set(differences(group[0]["parameters"], item["parameters"])) - {"shell.subdivisions"}
               for item in group[1:]):
            continue
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
        try:
            for item in group:
                with (Path(item["path"])/"shell_history.csv").open(encoding="utf-8", newline="") as handle:
                    history = list(csv.DictReader(handle))
                times = [float(row["time_myr"]) for row in history]
                for ax, key, factor in zip(axes,
                        ("mean_damage", "fault_active_area_fraction", "max_equivalent_slip_km"), (100, 100, 1)):
                    ax.plot(times, [factor*float(row[key]) for row in history],
                            label=f"{item['cell_count']} ячеек", linewidth=2)
            for ax, title, unit in zip(axes,
                    ("Среднее повреждение оболочки", "Площадь активированных зон", "Максимальный эквивалентный сдвиг"),
                    ("%", "% поверхности", "км")):
                ax.set(title=title, xlabel="Возраст · млн лет", ylabel=unit)
                ax.grid(alpha=.2)
            axes[0].legend()
            fig.suptitle(f"Нагрузка {traction/1000:g} кПа · одинаковые параметры, отличается только сетка", fontsize=14)
            fig.text(.5, .012, "Эквивалентный сдвиг — деформация зоны конечной ширины, а не раскрытие отдельной границы.",
                     ha="center", fontsize=9)
            fig.tight_layout(rect=(0, .04, 1, .92))
            filename = f"matched_{traction/1000:g}kpa.png"
            fig.savefig(output/filename, dpi=140)
            figures.append(filename)
        finally:
            plt.close(fig)
    return figures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--compare", action="append", type=Path, default=[])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("run_audit.json", "run_audit.md"):
        if (args.output/name).exists():
            parser.error(f"Refusing to replace an existing {name}")
    inspected = [inspect_run(path) for path in [args.run, *args.compare]]
    figures = plot_matched_groups(inspected, args.output)
    user = inspected[0]
    report = {"user_run": user, "comparisons": inspected[1:],
        "parameter_differences_from_user": {
            item["path"]: differences(user["parameters"], item["parameters"]) for item in inspected[1:]},
        "interpretation": {
            "mode": "Continuous material shell with finite-width weak planes; no split-bank contact yet.",
            "rate_sampling": "Animation samples the last accepted mechanics substep; cumulative slip retains all accepted increments.",
            "inactive_planes": "Inactive fault-plane stress/strength arrays are undefined physically, although stored as zeros.",
            "positive_control": "A larger imposed traction is a separate sensitivity experiment, not calibration or proof of real planetary fracture.",
        }}
    (args.output/"run_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    rows = [f"# Проверка пользовательского прогона на {user['cell_count']} ячейках", "",
        f"Источник: `{user['path']}`. Архив успешно проверен штатным загрузчиком; общие часы: "
        f"{user['synchronized_clocks']}. Статус: `{user['status']}`.", "",
        "| Сценарий | Ячеек | Нагрузка, кПа | Возраст, млн лет | Активных ячеек зон | Максимальный эквивалентный сдвиг, км | Статус |",
        "|---|---:|---:|---:|---:|---:|---|"]
    for item in inspected:
        rows.append(f"| {Path(item['path']).name} | {item['cell_count']} | "
            f"{item['parameters']['shell']['convective_traction_pa']/1000:g} | {item['physical_time_myr']:.6g} | "
            f"{item['active_cells']} | {item['max_equivalent_slip_km']:.6g} | {item['status']} |")
    rows += ["", "## Что произошло в пользовательском расчёте", "",
        f"Активировавшихся когда-либо ячеек: {user['ever_activated_cells']}; накопленный сдвиг — "
        f"{user['max_equivalent_slip_km']:g} км; работа трения — {user['friction_work_j']:g} Дж. "
        "Активация необратима, накопленный сдвиг не уменьшается: при нулевых финальных значениях "
        "пропущенного между кадрами эпизода сдвига нет.", "",
        f"Максимальное повреждение в финале {user['max_damage']:.8g}; порог активации "
        f"{user['activation_damage']:g} должен сохраняться {user['activation_persistence_years']:g} лет. "
        f"Пик среднего повреждения среди сохранённых кадров — {100*user['peak_saved_mean_damage']:.4g}% "
        f"при {user['peak_saved_mean_damage_time_myr']:g} млн лет. Повреждение залечивается и разбавляется "
        "новым затвердевшим материалом. Это объясняет слабые всплески на карте повреждения.", "",
        "## Сопоставимость экспериментов", ""]
    for path, delta in report["parameter_differences_from_user"].items():
        rows += [f"- `{Path(path).name}`: " + "; ".join(
            f"`{key}`: {values[0]} → {values[1]}" for key, values in delta.items()) + "."]
    rows += ["", "Случаи с одинаковой нагрузкой позволяют отделить влияние сетки от изменения условий. "
        "Сравнение при 50 кПа служит проверкой работы механизма, но не обосновывает выбор нагрузки "
        "для реального спутника. Конечные времена разных сценариев нельзя подменять друг другом.", "",
        "## Отображение и следующий этап", "",
        "Карты напряжения и сопротивления сдвигу относятся только к активированным разломным плоскостям. "
        "Их исходные нули не означают отсутствия напряжений в сплошной оболочке. Новая визуализация "
        "показывает отсутствие активных зон и маскирует эти несуществующие плоскости.", "",
        f"Кадры пользовательского опыта разделены {user['parameters']['controls']['sample_interval_myr']*1e6:g} годами, "
        f"запрошенный механический шаг — {user['parameters']['controls']['shell_step_myr']*1e6:g} лет. "
        "Карта скорости относится только к последнему принятому шагу; накопленный сдвиг учитывает весь расчёт.", "",
        "В пользовательском опыте отдельных плит ещё нет. Проверять устойчивое движение фрагментов "
        "нужно после возникновения устойчивых слабых зон и их развития в совместном тепловом и "
        "контактном расчёте. Проверка числа связных областей сама по себе этого не заменяет.", ""]
    for filename in figures:
        rows += [f"![Сопоставление сеток]({filename})", ""]
    (args.output/"run_audit.md").write_text("\n".join(rows), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "inspected_runs": len(inspected)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
