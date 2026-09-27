"""Plot common-time comparisons from genesis_onset_validation.py output.

Usage: python analysis/plot_genesis_onset_validation.py path/to/validation.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


BLUE = "#235ca3"
TEAL = "#078679"
ORANGE = "#bd6b1c"
PURPLE = "#865295"
INK = "#203044"


def _number(value, precision=3):
    return f"{value:.{precision}f}".replace(".", ",")


def _series(case, name):
    history = case["history"]
    times = np.asarray([row["time_myr"] for row in history], dtype=float)
    values = np.asarray([row[name] for row in history], dtype=float)
    if (len(times) == 0 or not np.isfinite(times).all()
            or not np.isfinite(values).all() or np.any(np.diff(times) <= 0)):
        raise ValueError(f"Invalid finite, ordered history for {name}")
    return times, values


def _at_time(case, time_myr):
    matches = [row for row in case["history"]
               if math.isclose(row["time_myr"], time_myr, rel_tol=0, abs_tol=1e-10)]
    if len(matches) != 1:
        raise ValueError(f"Comparison needs exactly one saved row at {time_myr} Myr")
    return matches[0]


def save_comparison(results, output: Path, comparison_time_myr=0.96):
    """Render stored samples without interpolating or comparing unequal ages."""
    reference = results["reference"]
    variants = (
        ("reference", "Базовый опыт", BLUE, "-"),
        ("without_tides", "Без приливов", ORANGE, (0, (5, 3))),
        ("without_water_weakening", "Без ослабления водой", PURPLE, (0, (1, 2))),
    )
    strong_names = ("strong_traction", "strong_half_step", "strong_fine_mesh")
    strong = [results[name] for name in strong_names]
    common = [_at_time(case, comparison_time_myr) for case in strong]
    areas = np.asarray([row["damaged_area_fraction"]*100 for row in common])
    if not np.isfinite(areas).all() or np.any((areas < 0) | (areas > 100)):
        raise ValueError("Damaged areas must be finite percentages in [0, 100]")
    thresholds = [case["shell_parameters"]["damage_threshold"] for case in strong]
    tractions = [case["shell_parameters"]["convective_traction_pa"]/1000 for case in strong]
    if len(set(thresholds)) != 1 or len(set(tractions)) != 1:
        raise ValueError("Common-time bars require equal damage thresholds and tractions")
    labels = [f"{20*4**case['shell_parameters']['subdivisions']} ячеек\n"
              f"Δt = {_number(case['dt_myr'])} млн лет" for case in strong]

    time, eccentricity = _series(reference, "eccentricity")
    _, water = _series(reference, "mean_water_access")
    curves = [(label, color, style, *_series(results[name], "mean_damage"))
              for name, label, color, style in variants]
    end = min(values[-1] for _, _, _, values, _ in curves)
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "axes.edgecolor": "#abb6c4",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
    }):
        fig = plt.figure(figsize=(13.4, 9.6))
        grid = fig.add_gridspec(2, 2, left=.095, right=.94, top=.85,
                               bottom=.17, wspace=.39, hspace=.69,
                               height_ratios=(1.08, 1.))
        orbit_ax = fig.add_subplot(grid[0, 0])
        damage_ax = fig.add_subplot(grid[0, 1])
        area_ax = fig.add_subplot(grid[1, :])
        # The two-line bar labels need more horizontal space than the upper
        # numerical y axes; reserve it explicitly instead of clipping at save.
        area_box = area_ax.get_position()
        area_ax.set_position([.16, area_box.y0, .78, area_box.height])
        try:
            fig.suptitle("Генезис: приливы, вода и чувствительность расчёта",
                         x=.095, ha="left", y=.97, fontsize=17, fontweight="bold")
            ref_cells = 20*4**reference["shell_parameters"]["subdivisions"]
            fig.text(.095, .925,
                f"Базовый опыт: {ref_cells} ячеек · шаг {_number(reference['dt_myr'])} млн лет · "
                f"нагрузка {_number(reference['shell_parameters']['convective_traction_pa']/1000, 0)} кПа",
                fontsize=10, color="#5b687a")

            orbit_ax.set_title("A  Орбита и доступ воды в породы", loc="left", pad=14)
            orbit_line, = orbit_ax.plot(time, eccentricity*1e4, color=BLUE, lw=2.3,
                                       label="Эксцентриситет e")
            orbit_ax.set_ylabel("Эксцентриситет e × 10⁴", color=BLUE)
            orbit_ax.tick_params(axis="y", colors=BLUE)
            orbit_ax.set_ylim(0, max(reference["tides_parameters"]["eccentricity"]*1e4*1.07, .1))
            orbit_ax.set_xlim(0, time[-1])
            orbit_ax.set_xlabel("Возраст, млн лет")
            orbit_ax.grid(axis="both", alpha=.18)
            water_ax = orbit_ax.twinx()
            water_line, = water_ax.plot(time, water, color=TEAL, lw=2.3,
                                       linestyle=(0, (5, 2)), label="Средний доступ воды")
            water_ax.set_ylabel("Средний доступ воды, доля", color=TEAL)
            water_ax.tick_params(axis="y", colors=TEAL)
            water_ax.spines["right"].set_visible(True)
            water_ax.set_ylim(0, 1)
            orbit_ax.legend(handles=[orbit_line, water_line], loc="upper right",
                            frameon=False, fontsize=9)
            orbit_ax.text(.28, .56, "Изолированное затухание орбиты;\nподдержка резонансами не задана",
                          transform=orbit_ax.transAxes, fontsize=8.5, color="#5b687a")

            damage_ax.set_title("B  Среднее повреждение оболочки", loc="left", pad=14)
            positive = []
            for label, color, style, times, damage in curves:
                mask = (times <= end+1e-10) & (damage > 0)
                positive.extend(damage[mask])
                damage_ax.plot(times, np.where(mask, damage, np.nan),
                               color=color, lw=2., linestyle=style, label=label)
            if not positive:
                raise ValueError("Mean damage is zero in all comparison samples")
            damage_ax.set_yscale("log")
            damage_ax.set_ylim(min(positive)*.7, max(positive)*1.8)
            damage_ax.set_xlim(min(times[np.flatnonzero(damage > 0)[0]]
                                   for _, _, _, times, damage in curves if np.any(damage > 0)), end)
            damage_ax.set_xlabel("Возраст, млн лет")
            damage_ax.set_ylabel("Среднее D · логарифмическая шкала")
            damage_ax.legend(loc="upper right", frameon=False, fontsize=9)
            damage_ax.grid(which="major", alpha=.18)
            damage_ax.text(.04, .05, "При выбранной орбите кривые с приливами\nи без них почти совпадают",
                           transform=damage_ax.transAxes, fontsize=8.5, color="#5b687a")

            area_ax.set_title(
                f"C  Усиленная нагрузка {_number(tractions[0], 0)} кПа · "
                f"общий возраст {_number(comparison_time_myr)} млн лет",
                loc="left", pad=15)
            bars = area_ax.barh(np.arange(3), areas, height=.55,
                                color=[BLUE, ORANGE, TEAL])
            area_ax.set_yticks(np.arange(3), labels=labels)
            area_ax.invert_yaxis()
            area_ax.set_xlim(0, max(float(areas.max())*1.2, 1.))
            area_ax.set_xlabel(f"Площадь с D ≥ {_number(thresholds[0], 2)}, % поверхности")
            area_ax.grid(axis="x", alpha=.18)
            area_ax.set_axisbelow(True)
            for bar, value in zip(bars, areas):
                area_ax.text(value+area_ax.get_xlim()[1]*.012,
                             bar.get_y()+bar.get_height()/2,
                             f"{_number(value, 2)} %", va="center", fontweight="bold")

            decimal = FuncFormatter(lambda x, _: f"{x:g}".replace(".", ","))
            for ax in (orbit_ax, damage_ax, area_ax):
                ax.xaxis.set_major_formatter(decimal)
            orbit_ax.yaxis.set_major_formatter(decimal)
            water_ax.yaxis.set_major_formatter(decimal)
            fig.text(.095, .09,
                     "Меньший шаг заметно меняет площадь повреждения. Это проверка чувствительности, сходимость ещё не установлена.",
                     fontsize=9, color=INK)
            fig.text(.095, .058,
                     "Модель ранних малых деформаций. Повреждённые области ещё не являются самостоятельными тектоническими плитами.",
                     fontsize=9, color="#5b687a")
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=170)
        finally:
            plt.close(fig)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--comparison-time-myr", type=float, default=.96)
    args = parser.parse_args(argv)
    with args.validation.open(encoding="utf-8") as handle:
        results = json.load(handle)
    output = args.output or args.validation.parent/"onset_validation.png"
    print(save_comparison(results, output, args.comparison_time_myr).resolve())


if __name__ == "__main__":
    main()
