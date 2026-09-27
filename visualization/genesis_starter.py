"""Maps and physical-time history for the coarse genesis starter."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np

from .raster import rasterize_cells


def save_starter_snapshot(mesh, state, path, history, summary):
    """Render a diagnostic partition; map labels do not assert mobile plates."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    damage = np.asarray(state.damage, dtype=float)
    domains = np.asarray(state.system.cell_plate, dtype=int)
    if damage.shape != (mesh.cell_count,) or domains.shape != damage.shape or not np.isfinite(damage).all():
        raise ValueError("Starter maps require one finite damage value and domain label per cell")
    t = np.asarray([r["time_myr"] for r in history], dtype=float)
    if not len(t):
        raise ValueError("Starter history cannot be empty")
    with plt.rc_context({"font.size": 10, "axes.titlesize": 12, "figure.facecolor": "#f7f9fc",
                         "axes.facecolor": "white", "text.color": "#172638", "axes.labelcolor": "#29374d"}):
        fig = plt.figure(figsize=(14, 8.5), layout="constrained")
        try:
            grid = fig.add_gridspec(2, 2)
            damage_ax = fig.add_subplot(grid[0, 0], projection="mollweide")
            domain_ax = fig.add_subplot(grid[0, 1], projection="mollweide")
            heat_ax = fig.add_subplot(grid[1, 0])
            failure_ax = fig.add_subplot(grid[1, 1])
            for ax in (damage_ax, domain_ax):
                ax.grid(alpha=.2)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            lon, lat, raster = rasterize_cells(mesh, damage)
            artist = damage_ax.pcolormesh(lon, lat, raster, shading="flat", cmap="magma", vmin=0, vmax=1, rasterized=True)
            damage_ax.set_title("Повреждение покрышки")
            fig.colorbar(artist, ax=damage_ax, orientation="horizontal", fraction=.055, pad=.05, label="Параметр повреждения")
            count = int(np.max(domains)) + 1
            lon, lat, raster = rasterize_cells(mesh, domains.astype(float))
            cmap = matplotlib.colormaps["tab20"].resampled(max(2, count))
            artist = domain_ax.pcolormesh(lon, lat, raster, shading="flat", cmap=cmap,
                                        norm=BoundaryNorm(np.arange(count + 1) - .5, cmap.N), rasterized=True)
            domain_ax.set_title(f"Области оболочки: {count}" if count > 1 else "Единая оболочка — разделения ещё нет")
            bar = fig.colorbar(artist, ax=domain_ax, orientation="horizontal", fraction=.055, pad=.05,
                               ticks=np.arange(count), label="Номер области; подвижность ещё не проверена")
            bar.ax.set_xticklabels([str(i + 1) for i in range(count)])
            heat_ax.plot(t, [r["surface_temperature_k"] for r in history], label="Поверхность", color="#e59624")
            heat_ax.plot(t, [r["mantle_temperature_k"] for r in history], label="Недра", color="#c94c36")
            heat_ax.set(title="Остывание после расплавленного старта", ylabel="Температура, K")
            heat_ax.legend(loc="best")
            failure_ax.plot(t, [r["max_yield_ratio"] for r in history], label="Макс. нагрузка / прочность", color="#8c53a5")
            failure_ax.axhline(1, color="#8a94a3", linestyle=":", linewidth=1)
            failure_ax.set(title="Нагрузка и накопление повреждения", ylabel="Нагрузка / прочность")
            damage_history_ax = failure_ax.twinx()
            damage_history_ax.plot(t, [r["damaged_area_fraction"] for r in history], label="Повреждённая площадь", color="#198a93")
            damage_history_ax.set(ylabel="Доля повреждённой площади", ylim=(-.02, 1.02))
            lines = failure_ax.lines[:1] + damage_history_ax.lines
            failure_ax.legend(lines, [line.get_label() for line in lines], loc="best", fontsize=9)
            for ax in (heat_ax, failure_ax):
                ax.set_xlabel("Возраст после расплавленного старта, млн лет")
                ax.grid(alpha=.16)
                ax.set_xlim(float(t[0]), float(t[-1]) if t[-1] > t[0] else float(t[0]) + .1)
            candidate = summary.get("candidate_partition", False)
            subtitle = "Найден кандидат начального разделения" if candidate else "Устойчивые подвижные плиты ещё не получены"
            fig.suptitle(f"Стартер генезиса · {state.time_myr:.4f} млн лет\n{subtitle}", fontsize=16, fontweight="bold")
            fig.supxlabel("Параметризованное разрушение на гладкой сфере без исходного рельефа.\nКонтиненты, рельеф и перенос в зрелую тектонику здесь не рассчитываются.", fontsize=10)
            fig.savefig(path, dpi=135)
        finally:
            plt.close(fig)
    return path
