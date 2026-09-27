"""Fault-band maps on the current, continuous material shell."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from .genesis_shell import crack_segments
from .raster import rasterize_cells


def save_fault_snapshot(mesh, fields: dict, path: str | Path, time_myr: float,
                        summary: dict | None = None) -> Path:
    """Render six fixed-scale maps using the supplied current material mesh.

    Equivalent slip is a constitutive measure: fault-band width multiplied by
    accumulated shear. It is not a displacement jump between detached plates.
    Optional ``summary['plot_limits']`` overrides remain fixed for each caller's
    frame sequence. Out-of-range data use explicit colorbar extensions.
    Stress and strength describe activated fault planes, not the bulk shell;
    inactive cells are masked when ``fault_active`` is supplied.
    """
    summary = summary or {}
    limits = summary.get("plot_limits", {})
    panels = (
        ("damage", "Повреждение оболочки", "Доля повреждения", "magma", (0, 1)),
        ("water_access", "Доступ жидкой воды в породы", "Доля доступа", "Blues", (0, 1)),
        ("equivalent_slip_km", "Накопленный эквивалентный сдвиг", "км · ширина зоны × накопленный сдвиг", "cividis", (0, 20)),
        ("slip_rate_cm_yr", "Скорость эквивалентного сдвига", "см/год · последний расчётный шаг", "viridis", (0, 5)),
        ("shear_stress_mpa", "Сдвиговое напряжение\nна активированных плоскостях", "МПа", "magma", (0, 10)),
        ("shear_strength_mpa", "Сопротивление сдвигу\nна активированных плоскостях", "МПа · сцепление и трение", "cividis", (0, 10)),
    )
    if not np.isfinite(time_myr) or time_myr < 0:
        raise ValueError("time_myr must be finite and nonnegative")
    validated = []
    for name, title, unit, cmap, defaults in panels:
        values = np.asarray(fields[name], dtype=float)
        if values.shape != (mesh.cell_count,) or not np.isfinite(values).all():
            raise ValueError(f"{name} must contain one finite value per cell")
        scale = np.asarray(limits.get(name, defaults), dtype=float)
        if scale.shape != (2,) or not np.isfinite(scale).all() or scale[0] >= scale[1]:
            raise ValueError(f"plot_limits[{name!r}] must contain increasing finite limits")
        validated.append((name, values, title, unit, cmap, scale))
    active = fields.get("fault_active")
    if active is not None:
        active = np.asarray(active)
        if active.dtype != bool or active.shape != (mesh.cell_count,):
            raise ValueError("fault_active must be a boolean mask over cells")
    failed = np.asarray(fields.get("failed_edges", np.zeros(len(mesh.shared_edges), dtype=bool)))
    if failed.dtype != bool or failed.shape != (len(mesh.shared_edges),):
        raise ValueError("failed_edges must be a boolean mask over shared edges")
    segments = crack_segments(mesh, failed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 10, "axes.titlesize": 12,
                         "figure.facecolor": "#f7f9fc", "text.color": "#172638",
                         "axes.labelcolor": "#29374d"}):
        fig, axes = plt.subplots(3, 2, figsize=(13, 11.5),
                                 subplot_kw={"projection": "mollweide"})
        try:
            fig.subplots_adjust(left=0.04, right=0.96, top=0.84, bottom=0.13,
                                hspace=0.57, wspace=0.15)
            for ax, (name, values, title, unit, cmap, (vmin, vmax)) in zip(axes.flat, validated):
                plane_field = name in {"shear_stress_mpa", "shear_strength_mpa"}
                masked = plane_field and active is not None
                displayed = np.where(active, values, np.nan) if masked else values
                visible = values[active] if masked else values
                lon, lat, raster = rasterize_cells(mesh, displayed)
                palette = plt.get_cmap(cmap).with_extremes(bad="#e4e8ef")
                artist = ax.pcolormesh(lon, lat, np.ma.masked_invalid(raster), shading="flat", cmap=palette,
                                      vmin=vmin, vmax=vmax, rasterized=True)
                ax.set_title(title, pad=12)
                ax.grid(alpha=0.18, color="#697585", linewidth=0.5)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                below = visible.size > 0 and visible.min() < vmin
                above = visible.size > 0 and visible.max() > vmax
                extension = "both" if below and above else "min" if below else "max" if above else "neither"
                fig.colorbar(artist, ax=ax, orientation="horizontal", pad=0.07,
                             fraction=0.055, aspect=32, label=unit, extend=extension)
                if masked and not np.any(active):
                    ax.text(0.5, 0.5, "Активных разломных зон нет", transform=ax.transAxes,
                            ha="center", va="center", fontsize=10, color="#455268")
            if segments:
                axes[0, 0].add_collection(LineCollection(
                    segments, colors="#65ecf2", linewidths=0.7, alpha=0.95), autolim=False)
            fig.suptitle(f"Генезис · трение и сдвиг разломных зон\n{time_myr:.3f} млн лет",
                         fontsize=15, fontweight="bold", y=0.97)
            active_text = (f"Активные разломные зоны: {np.count_nonzero(active)} из {mesh.cell_count} ячеек"
                           if active is not None else "Активация разломных зон: данные не переданы")
            fig.text(0.5, 0.903,
                     f"{active_text}\n"
                     f"Максимум повреждения: {np.max(fields['damage']):.4g}; "
                     f"накопленного эквивалентного сдвига: {np.max(fields['equivalent_slip_km']):.4g} км",
                     ha="center", va="center", fontsize=9)
            note = ("Карты на текущей геометрии материала. Голубые линии — контуры повреждённых областей.\n"
                    "Эквивалентный сдвиг = ширина зоны × накопленная сдвиговая деформация.\n"
                    "Оболочка остаётся связной: разрыв перемещений и контакт самостоятельных плит не рассчитываются.")
            if active is not None:
                note += "\nСерые области: разломные плоскости не активированы; напряжения оболочки здесь не показаны."
            status = summary.get("status", summary.get("shell", {}).get("stopped_reason"))
            if status and status != "completed":
                note += f"\nРасчёт остановлен: {status}."
            fig.text(0.5, 0.043, note, ha="center", va="center", fontsize=9)
            fig.savefig(path, dpi=125)
        finally:
            plt.close(fig)
    return path
