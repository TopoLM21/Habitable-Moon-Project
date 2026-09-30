"""Static scientific comparison of the independent0.4 and0.5 trajectories."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from validation_results import HERE, ROOT, ordered_paths, segment


def main():
    oldroot = ROOT / "analysis/slab_sinking_validation/runs"
    old = [segment(oldroot / ("validated_sinking_sub4_dt1" if age <= 100 else "validated_sinking_fixed2_sub4_dt1")
        / f"elapsed_{age:04d}") for age in (50,100,200,400)]
    new = [segment(path) for path in ordered_paths()]
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":11,
        "axes.spines.top":False,"axes.spines.right":False,"axes.titleweight":"bold"})
    fig, axes = plt.subplots(2,1,figsize=(11,7.5),sharex=True,layout="constrained")
    for label,color,items in (("0.4 · усреднённая холодная масса","#777f89",old),
            ("0.5 · упорядоченные тепловые порции","#006f9f",new)):
        report = items[-1]["report"]
        origin = report["import"]["origin_time_myr"]
        times = np.array([r["time_myr"]-origin for r in report["history"]])
        speeds = np.array([r["mean_surface_speed_km_myr"] for r in report["history"]])
        axes[0].plot(times,speeds,color=color,lw=1.4,label=label)
        selected = (times>300)&(times<=400)
        if selected.any():
            mean = speeds[selected].mean()
            axes[0].plot([300,400],[mean,mean],color=color,lw=2,ls="--")
        rows = [row for item in items for row in item["trace"]]
        force_times = [row["state_time_myr"]-origin for row in rows]
        fraction = [100*row["accepted_slab_inventory"]["unresolved_or_detached_fraction_of_accepted_volume"] for row in rows]
        axes[1].plot(force_times,fraction,color=color,lw=1.6)
        axes[1].scatter([report["duration_myr"]],
            [100*report["accepted_slab_inventory"]["unresolved_or_detached_fraction_of_accepted_volume"]],
            color=color,s=28,zorder=5)
    axes[0].set(title="Движение плит из одного состояния Starter",ylabel="Средняя скорость (мм/год)")
    axes[0].legend(loc="upper right",frameon=False)
    axes[0].text(.025,.91,"Пунктир: среднее за 300–400 млн лет",transform=axes[0].transAxes,
        fontsize=9,color="#555555")
    axes[1].set(title="Принятый материал, утративший связь с границей или оторвавшийся",
        ylabel="Доля от принятого объёма (%)",
        xlabel="Время после Starter (млн лет)",ylim=(-2,102),xlim=(0,400))
    for ax in axes:
        ax.grid(alpha=.18)
    fig.suptitle("Учёт положения холодных порций меняет движение; сходимость по сетке не доказана",fontsize=13,fontweight="bold")
    fig.supxlabel("Скорость — средняя по площади после переноса. Доля — при расчёте сил; точки — сохранённые состояния.\nУтрата связи с границей и механический отрыв учтены вместе; это не объём переноса в глубокую мантию.",
        fontsize=9,color="#555555")
    output = HERE/"comparison.png"
    fig.savefig(output,dpi=180)
    print(output)


if __name__=="__main__":
    main()
