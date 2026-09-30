"""Publication-style comparison of speed and the actual SI dissipation budget."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=HERE / "runs")
    parser.add_argument("--output", type=Path, default=HERE / "speed_and_power.png")
    args = parser.parse_args()
    rows = []
    for age in (50, 100, 200, 400):
        # Failed retry folders have no continuation.json and are excluded.
        candidates = list(args.runs.glob(f"validated_sinking*sub4_dt1/elapsed_{age:04d}/continuation.json"))
        if not candidates:
            raise ValueError(f"No completed primary-mesh sinking result at {age} Myr")
        folder = max(candidates, key=lambda path:path.stat().st_mtime).parent
        report = read(folder / "continuation.json")
        control = args.runs / "validated_disabled_sub4_dt1" / f"elapsed_{age:04d}"
        baseline = read(control / "continuation.json")
        trace = json.loads(folder.with_suffix(".dynamics.jsonl").read_text(encoding="utf-8").splitlines()[-1])["trace"]
        if not all(report["checks"].values()) or not all(baseline["checks"].values()):
            raise ValueError("Cannot draw an accepted result from failed continuation checks")
        rows.append(dict(age_myr=age, main_path=str(folder), disabled_path=str(control),
            mean_speed_mm_yr=report["final_mean_surface_speed_km_myr"],
            max_speed_mm_yr=report["final_max_surface_speed_km_myr"],
            disabled_mean_speed_mm_yr=baseline["final_mean_surface_speed_km_myr"],
            disabled_max_speed_mm_yr=baseline["final_max_surface_speed_km_myr"],
            powers_w={key:trace[key] for key in ("basal_drag_dissipation_w", "slab_bending_dissipation_w",
                "slab_mantle_dissipation_w", "total_source_power_w", "slab_power_w", "slab_constraint_power_w")}))
    plt.rcParams.update({"font.size":10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios":[1.12, 1.]})
    for path, label, color, style in ((Path(rows[-1]["main_path"]), "Погружение, изгиб и конечная прочность", "#006b83", "-"),
            (Path(rows[-1]["disabled_path"]), "Контроль: погружение выключено", "#858585", "--")):
        report = read(path / "continuation.json")
        origin, history = report["import"]["origin_time_myr"], report["history"]
        axes[0].plot([item["time_myr"]-origin for item in history],
            [item["mean_surface_speed_km_myr"] for item in history], label=label, color=color, linestyle=style, lw=1.9)
    axes[0].scatter([row["age_myr"] for row in rows], [row["mean_speed_mm_yr"] for row in rows], s=22, color="#006b83", zorder=3)
    axes[0].set_xlabel("Время после первого разделения, млн лет")
    axes[0].set_ylabel("Средняя скорость плит, мм/год")
    axes[0].set_title("Скорость следует из баланса сил", fontsize=11)
    axes[0].legend(fontsize=8, loc="best", frameon=False)
    positions, bottom = np.arange(len(rows)), np.zeros(len(rows))
    for key, label, color in (("basal_drag_dissipation_w", "Базальное трение", "#788b9b"),
            ("slab_bending_dissipation_w", "Изгиб плиты", "#d69a58"),
            ("slab_mantle_dissipation_w", "Мантия вокруг слэба", "#4a9c9b")):
        values = np.array([row["powers_w"][key] for row in rows])/1e9
        axes[1].bar(positions, values, bottom=bottom, label=label, color=color, width=.56)
        bottom += values
    axes[1].scatter(positions, [row["powers_w"]["total_source_power_w"]/1e9 for row in rows],
        color="#202020", marker="D", s=24, label="Работа всех источников", zorder=4)
    axes[1].set_xticks(positions, [str(row["age_myr"]) for row in rows])
    axes[1].set_xlabel("Время после первого разделения, млн лет")
    axes[1].set_ylabel("Мощность, ГВт")
    axes[1].set_title("Источники покрывают вязкую диссипацию", fontsize=11)
    axes[1].legend(fontsize=8, frameon=False, loc="best")
    axes[1].set_ylim(top=max(float(bottom.max()), 1e-12)*1.38)
    for ax in axes:
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(bottom=0)
    fig.suptitle("Проверка модели 0.4 при неизменной исходной тяге мантии", fontsize=13)
    fig.text(.5, .025, "Обе серии квазистатические, из одного Starter. Столбцы: последний механический шаг каждого участка.\n"
        "Перенос остаётся растровым; сходимость по сетке и шагу времени пока не установлена.", ha="center", va="bottom", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, .13, 1, .94))
    fig.savefig(args.output, dpi=180)
    args.output.with_suffix(".json").write_text(json.dumps(dict(
        physical_comparison="Same source, same quasistatic mechanics, only slab closure differs.",
        numerical_provenance="Main 0–100 Myr used original successful solver; 100–400 resumed after dual null-circulation repair. Independent fresh-50 replay validates the change.",
        cases=rows), ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(str(args.output))


if __name__ == "__main__":
    main()
