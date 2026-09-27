"""Physical cooling/contact snapshots on the actual split material triangles."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from .genesis_contact import material_polygons
from .genesis_connectivity import connectivity_text


def save_coupled_snapshot(mesh, fields, path, history, summary):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    polygons, owners = material_polygons(mesh)
    damage = np.asarray(fields["damage"], dtype=float)
    centers = np.asarray(fields["seam_centers_xyz"], dtype=float)
    gaps = np.asarray(fields["seam_gap_m"], dtype=float)
    slips = np.asarray(fields["seam_slip_m"], dtype=float)
    if damage.shape != (len(mesh.faces),) or not np.isfinite(damage).all():
        raise ValueError("damage must have one finite value per face")
    if centers.ndim != 2 or centers.shape[1] != 3 or not np.isfinite(centers).all():
        raise ValueError("seam centers must be finite 3-D coordinates")
    if any(value.shape != (len(centers), 2) or not np.isfinite(value).all() for value in (gaps, slips)):
        raise ValueError("contact fields must have two finite values per seam")
    lengths = np.linalg.norm(centers, axis=1)
    if np.any(lengths <= 0):
        raise ValueError("contact centers must be nonzero")
    centers = centers / lengths[:, None]
    lon, lat = np.arctan2(centers[:, 1], centers[:, 0]), np.arcsin(np.clip(centers[:, 2], -1, 1))
    gap = np.where(gaps.min(axis=1) < 0, gaps.min(axis=1), gaps.max(axis=1))
    slip = np.abs(slips).max(axis=1)
    final = history[-1] if history else summary.get("final", {})
    times = np.array([row["elapsed_years"] for row in history], dtype=float)
    with plt.rc_context({"font.size": 10, "figure.facecolor": "#f7f9fc", "text.color": "#172638"}):
        fig = plt.figure(figsize=(13, 9.5))
        try:
            grid = fig.add_gridspec(2, 2, left=.075, right=.90, top=.76, bottom=.19,
                                   height_ratios=(1.2, .8), hspace=.58, wspace=.33)
            for index, (data, title, norm, cmap) in enumerate((
                (gap, "Раскрытие и сжатие берегов · м", TwoSlopeNorm(vmin=-5., vcenter=0., vmax=100.), "coolwarm"),
                (slip, "Относительный сдвиг берегов · м", matplotlib.colors.Normalize(0., 100.), "viridis"),
            )):
                ax = fig.add_subplot(grid[0, index], projection="mollweide")
                artist = PolyCollection(polygons, array=damage[owners], cmap="Greys", edgecolors="none", alpha=.35)
                artist.set_clim(0, 1)
                ax.add_collection(artist, autolim=False)
                dots = ax.scatter(lon, lat, c=data, cmap=cmap, norm=norm, s=14, linewidths=.2, edgecolors="#263446")
                extend = "both" if len(data) and data.min() < norm.vmin and data.max() > norm.vmax else "min" if len(data) and data.min() < norm.vmin else "max" if len(data) and data.max() > norm.vmax else "neither"
                fig.colorbar(dots, ax=ax, orientation="horizontal", pad=.13, fraction=.05, extend=extend,
                             ticks=[-5., 0., 50., 100.] if index == 0 else None)
                ax.set_title(title, pad=14)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.grid(alpha=.2)
                if not len(data):
                    ax.text(.5, .5, "Разделённых рёбер пока нет", transform=ax.transAxes, ha="center")
                else:
                    ax.text(.5, -.035, f"Диапазон: {data.min():.4g} … {data.max():.4g} м", transform=ax.transAxes,
                            ha="center", fontsize=9)
            temp = fig.add_subplot(grid[1, 0])
            lid = fig.add_subplot(grid[1, 1])
            ocean = lid.twinx()
            for key, label, color in (("surface_temperature_k", "Поверхность", "#d07a3c"),
                                      ("mantle_temperature_k", "Мантия", "#9b4268")):
                if history and all(key in row for row in history):
                    values = np.array([row[key] for row in history], dtype=float)
                    temp.plot(times, values - values[0], label=f"{label} · {values[0]:.2f} K в начале", color=color)
            if history and all("mean_lid_thickness_km" in row for row in history):
                values = np.array([row["mean_lid_thickness_km"] for row in history], dtype=float)
                lid.plot(times, (values - values[0]) * 1000., color="#58647d", label=f"В начале: {values[0]:.5f} км")
                lid.legend(fontsize=8, loc="upper left")
            if history and all("ocean_fraction" in row for row in history):
                ocean.plot(times, [row["ocean_fraction"] for row in history], color="#257d9b", ls="--", label="Жидкая вода")
            temp.set_title("Изменение температуры за расчёт")
            temp.set_ylabel("Изменение относительно начала · K")
            if temp.lines:
                temp.legend(fontsize=9)
            lid.set_title("Рост оболочки и конденсация воды")
            lid.set_ylabel("Прирост средней толщины · м")
            ocean.set_ylabel("Доля жидкой воды", color="#257d9b")
            ocean.set_ylim(-.02, 1.02)
            for ax in (temp, lid):
                ax.set_xlabel("Физических лет после исходного состояния")
                ax.grid(alpha=.2)
                ax.ticklabel_format(axis="x", style="plain", useOffset=False)
            fig.suptitle("Генезис · остывание и развитие разломов", fontsize=16, fontweight="bold", y=.96)
            fig.text(.5, .91, f"Возраст: {final.get('time_myr', 0.):.6f} млн лет · продолжение: {final.get('elapsed_years', 0.):.4g} лет", ha="center", fontsize=11)
            fig.text(.5, .865, "Тепло, орбита, релаксация напряжений и контакт берегов развиваются совместно", ha="center", fontsize=9)
            fig.text(.5, .815, connectivity_text(final), ha="center", fontsize=9)
            note = ("Эксперимент с малыми деформациями и скольжениями; только исходно соседние контактные рёбра.\n"
                    "Тепловой и механический балансы учитываются отдельно. Без субдукции и заполнения разрывов новой корой.")
            if summary.get("status", "completed") != "completed":
                note += "\nОстановка: " + str(summary["status"])
            fig.text(.5, .065, note, ha="center", va="center", fontsize=9)
            fig.savefig(path, dpi=130)
        finally:
            plt.close(fig)
    return path
