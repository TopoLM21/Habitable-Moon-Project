"""Maps of tidal loading, water access, and early small-strain shell motion."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from .genesis_shell import crack_segments
from .raster import rasterize_cells


def save_onset_snapshot(mesh, fields: dict, path: str | Path, time_myr: float,
                        summary: dict | None = None) -> Path:
    """Six maps on the supplied sphere, with fixed scales across saved frames.

    Values above a scale are shown by a colorbar extension rather than silently
    rescaling the maps. ``summary['plot_limits']`` may override individual scales.
    With ``summary['geometry'] == 'material'`` the caller must provide the
    current material mesh. Otherwise displacement remains a diagnostic on the
    reference sphere. The renderer never applies displacement a second time.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = summary or {}
    material = summary.get("geometry") == "material"
    limits = summary.get("plot_limits", {})
    panels = (
        ("lid_thickness_km", "Толщина твёрдой покрышки", "км", "cividis", (0, 20)),
        ("damage", "Повреждение оболочки", "Доля повреждения", "magma", (0, 1)),
        ("water_access", "Доступ жидкой воды в породы", "Доля доступа", "Blues", (0, 1)),
        ("speed_cm_yr", "Скорость боковых смещений", "см/год · без приливных колебаний", "viridis", (0, 5)),
        ("displacement_km", "Пройденный боковой путь", "км", "cividis", (0, 300)),
        ("tidal_stress_mpa", "Циклическая нагрузка последнего шага", "МПа · максимум за орбитальный цикл", "magma", (0, 0.05)),
    )
    # Validate before creating a figure so invalid input never leaves a partial plot.
    validated = []
    for name, title, unit, cmap, defaults in panels:
        values = np.asarray(fields[name], dtype=float)
        if values.shape != (mesh.cell_count,) or not np.isfinite(values).all():
            raise ValueError(f"{name} must contain one finite value per cell")
        scale = np.asarray(limits.get(name, defaults), dtype=float)
        if scale.shape != (2,) or not np.isfinite(scale).all() or scale[0] >= scale[1]:
            raise ValueError(f"plot_limits[{name!r}] must contain increasing finite limits")
        validated.append((values, title, unit, cmap, scale))
    segments = crack_segments(mesh, fields["failed_edges"])
    with plt.rc_context({"font.size": 10, "axes.titlesize": 12,
                         "figure.facecolor": "#f7f9fc", "text.color": "#172638",
                         "axes.labelcolor": "#29374d"}):
        fig, axes = plt.subplots(3, 2, figsize=(13, 11.5),
                                 subplot_kw={"projection": "mollweide"})
        try:
            fig.subplots_adjust(left=0.04, right=0.96, top=0.86, bottom=0.12,
                                hspace=0.44, wspace=0.15)
            for ax, (values, title, unit, cmap, (vmin, vmax)) in zip(axes.flat, validated):
                lon, lat, raster = rasterize_cells(mesh, values)
                artist = ax.pcolormesh(lon, lat, raster, shading="flat", cmap=cmap,
                                       vmin=vmin, vmax=vmax, rasterized=True)
                ax.set_title(title, pad=12)
                ax.grid(alpha=0.18, color="#697585", linewidth=0.5)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                extension = ("both" if values.min() < vmin and values.max() > vmax
                             else "min" if values.min() < vmin
                             else "max" if values.max() > vmax else "neither")
                fig.colorbar(artist, ax=ax, orientation="horizontal", pad=0.07,
                             fraction=0.055, aspect=32, label=unit, extend=extension)
            if segments:
                axes[0, 1].add_collection(LineCollection(
                    segments, colors="#65ecf2", linewidths=0.7, alpha=0.95), autolim=False)
            orbit = summary.get("orbit", {})
            orbit_note = "Приливные циклы усреднены; медленные смещения показаны отдельно"
            if "eccentricity" in orbit:
                orbit_note = f"Эксцентриситет: {orbit['eccentricity']:.4g}  ·  " + orbit_note
            title = "движущаяся оболочка" if material else "приливы, вода и первые смещения"
            fig.suptitle(f"Генезис · {title}\n{time_myr:.3f} млн лет",
                         fontsize=15, fontweight="bold", y=0.97)
            fig.text(0.5, 0.908, orbit_note, ha="center", va="center", fontsize=9)
            note = (("Карты на текущей геометрии материала. " if material else "Карты на исходной сфере. ")
                    + "Голубые линии — контуры повреждённых областей.\n"
                    + "Движение оболочки ещё не означает появления самостоятельных тектонических плит.")
            shell = summary.get("shell", {})
            status = summary.get("status", shell.get("stopped_reason"))
            if material and status and status != "completed":
                note += f"\nРасчёт остановлен: {status}."
            elif status == "shell_small_strain_limit":
                note += "\nДостигнут предел малых деформаций; для продолжения требуется обновление геометрии."
            fig.text(0.5, 0.042, note, ha="center", va="center", fontsize=9)
            fig.savefig(path, dpi=125)
        finally:
            plt.close(fig)
    return path
