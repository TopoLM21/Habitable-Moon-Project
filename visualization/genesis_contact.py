"""Draw split material triangles and small-sliding contact diagnostics.

Unlike nearest-cell maps, these polygons do not colour over geometric gaps.
Map projection is for inspection only, not an estimate of global coverage.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from .genesis_connectivity import connectivity_text


def _clip_longitude(polygon, boundary, keep_greater):
    """Clip a longitude/latitude polygon to one longitude half plane."""
    if not len(polygon):
        return polygon
    result = []
    previous = polygon[-1]
    previous_inside = previous[0] >= boundary if keep_greater else previous[0] <= boundary
    for current in polygon:
        inside = current[0] >= boundary if keep_greater else current[0] <= boundary
        if inside != previous_inside:
            fraction = (boundary-previous[0])/(current[0]-previous[0])
            result.append(previous+fraction*(current-previous))
        if inside:
            result.append(current)
        previous, previous_inside = current, inside
    return np.asarray(result, dtype=float).reshape(-1, 2)


def material_polygons(mesh):
    """Return projected triangle pieces and their material-face indices."""
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or faces.ndim != 2 or faces.shape[1] != 3
            or not np.issubdtype(faces.dtype, np.integer)
            or np.any(faces < 0) or np.any(faces >= len(vertices))):
        raise ValueError("Expected finite vertices and triangular material connectivity")
    polygons, owners = [], []
    for face_index, triangle in enumerate(vertices[faces]):
        points = []
        for i in range(3):
            a, b = triangle[i], triangle[(i+1) % 3]
            # Normalized interpolation follows the same minor great-circle arc.
            arc = (1-np.arange(8)[:, None]/8)*a+(np.arange(8)[:, None]/8)*b
            arc /= np.linalg.norm(arc, axis=1)[:, None]
            points.extend(arc)
        points = np.asarray(points)
        longitude = np.unwrap(np.arctan2(points[:, 1], points[:, 0]))
        longitude -= 2*np.pi*np.floor((longitude.mean()+np.pi)/(2*np.pi))
        polygon = np.column_stack((longitude, np.arcsin(np.clip(points[:, 2], -1, 1))))
        for shift in (-2*np.pi, 0., 2*np.pi):
            shifted = polygon.copy()
            shifted[:, 0] += shift
            clipped = _clip_longitude(_clip_longitude(shifted, -np.pi, True), np.pi, False)
            if len(clipped) >= 3:
                polygons.append(clipped)
                owners.append(face_index)
    return polygons, np.asarray(owners, dtype=int)


def _history_column(history, names):
    for name in names:
        if history and all(name in row for row in history):
            values = np.asarray([row[name] for row in history], dtype=float)
            if np.isfinite(values).all():
                return values
    return None


def save_contact_snapshot(mesh, fields: dict, path: str | Path, source_time_myr: float,
                          elapsed_years: float, history=None, summary=None) -> Path:
    """Render real bank gaps/slip plus force and dissipation histories."""
    if not all(np.isfinite(x) and x >= 0 for x in (source_time_myr, elapsed_years)):
        raise ValueError("Source time and contact elapsed time must be finite and nonnegative")
    history, summary = history or [], summary or {}
    damage = np.asarray(fields["damage"], dtype=float)
    if damage.shape != (len(mesh.faces),) or not np.isfinite(damage).all():
        raise ValueError("damage must contain one finite value per material face")
    centers = np.asarray(fields["seam_centers_xyz"], dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3 or not np.isfinite(centers).all():
        raise ValueError("seam_centers_xyz must contain finite seam locations")
    seam_count = len(centers)
    values = {}
    for key in ("seam_gap_m", "seam_slip_m"):
        value = np.asarray(fields[key], dtype=float)
        if value.shape != (seam_count, 2) or not np.isfinite(value).all():
            raise ValueError(f"{key} must contain two finite endpoint values per seam")
        values[key] = value
    lengths = np.linalg.norm(centers, axis=1)
    if np.any(lengths <= 0):
        raise ValueError("Seam locations must be nonzero")
    centers = centers/lengths[:, None]
    lon = np.arctan2(centers[:, 1], centers[:, 0])
    lat = np.arcsin(np.clip(centers[:, 2], -1, 1))
    polygons, owners = material_polygons(mesh)
    # A compressive endpoint takes precedence so penetration is not hidden by
    # a positive gap at the other end of a rotating interface.
    gap = np.where(values["seam_gap_m"].min(axis=1) < 0,
                   values["seam_gap_m"].min(axis=1), values["seam_gap_m"].max(axis=1))
    slip = np.abs(values["seam_slip_m"]).max(axis=1)
    limits = summary.get("plot_limits", {})
    panels = ((gap, "Зазор между берегами", "м · отрицательный зазор — проникновение",
               "coolwarm", limits.get("gap_m", (-5., 100.))),
              (slip, "Относительный сдвиг берегов", "м · действительный скачок перемещения",
               "viridis", limits.get("slip_m", (0., 100.))))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 10, "figure.facecolor": "#f7f9fc",
                         "text.color": "#172638", "axes.labelcolor": "#29374d"}):
        fig = plt.figure(figsize=(13, 9.5))
        try:
            grid = fig.add_gridspec(2, 2, height_ratios=(1.2, .75), left=.075, right=.94,
                                   top=.76, bottom=.18, wspace=.25, hspace=.52)
            for column, (data, title, label, cmap, scale) in enumerate(panels):
                scale = np.asarray(scale, dtype=float)
                if scale.shape != (2,) or not np.isfinite(scale).all() or scale[0] >= scale[1]:
                    raise ValueError("Plot limits must be two increasing finite values")
                ax = fig.add_subplot(grid[0, column], projection="mollweide")
                background = PolyCollection(polygons, array=damage[owners], cmap="Greys",
                                            edgecolors="none", linewidths=0., alpha=.32)
                background.set_clim(0, 1)
                # Mollweide axes already own fixed longitude/latitude bounds.
                # Autoscaling a polygon collection asks the inverse projection
                # about points outside its ellipse and produces spurious NaNs.
                ax.add_collection(background, autolim=False)
                if column == 0 and not scale[0] < 0 < scale[1]:
                    raise ValueError("Gap plot limits must include compression and opening around zero")
                color_scale = ({"norm": TwoSlopeNorm(vmin=scale[0], vcenter=0., vmax=scale[1])}
                               if column == 0 else {"vmin": scale[0], "vmax": scale[1]})
                artist = ax.scatter(lon, lat, c=data, cmap=cmap, **color_scale,
                                    s=14, linewidths=.2, edgecolors="#29374d", zorder=3)
                extension = ("both" if len(data) and data.min() < scale[0] and data.max() > scale[1]
                             else "min" if len(data) and data.min() < scale[0]
                             else "max" if len(data) and data.max() > scale[1] else "neither")
                ticks = ([scale[0], 0., scale[1]/2, scale[1]] if column == 0 else None)
                fig.colorbar(artist, ax=ax, orientation="horizontal", pad=.11, fraction=.055,
                             aspect=26, label=label, extend=extension, ticks=ticks)
                ax.set_title(title, pad=13)
                ax.grid(alpha=.18)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                if len(data):
                    ax.text(.5, -.035, f"Диапазон: {data.min():.4g} … {data.max():.4g} м",
                            transform=ax.transAxes, ha="center", fontsize=9)
                else:
                    ax.text(.5, .5, "Разделённых рёбер пока нет", transform=ax.transAxes,
                            ha="center", va="center", fontsize=11)
            energy_ax = fig.add_subplot(grid[1, 0])
            residual_ax = fig.add_subplot(grid[1, 1])
            times = _history_column(history, ("elapsed_years", "time_years"))
            energy = _history_column(history, ("contact_dissipation_j", "interface_dissipation_j",
                                                "friction_work_j"))
            drag = _history_column(history, ("basal_drag_work_j", "drag_work_j"))
            residual = _history_column(history, ("equilibrium_residual", "force_residual"))
            if times is not None and energy is not None:
                energy_ax.plot(times, energy, color="#9852ad", label="Контакт")
            if times is not None and drag is not None:
                energy_ax.plot(times, drag, color="#267c9b", label="Сопротивление мантии")
            if energy_ax.lines:
                energy_ax.legend(fontsize=8)
                energy_ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
            else:
                energy_ax.text(.5, .5, "История работы отсутствует", transform=energy_ax.transAxes,
                               ha="center", va="center", fontsize=9)
            if times is not None and residual is not None:
                residual_ax.semilogy(times, np.maximum(residual, 1e-16), color="#b66036")
                tolerance = float(summary.get("equilibrium_tolerance", 1e-8))
                if np.isfinite(tolerance) and tolerance > 0:
                    residual_ax.axhline(tolerance, color="#7c8995", ls="--", lw=1,
                                        label=f"Допуск: {tolerance:g}")
                    residual_ax.legend(fontsize=8)
            else:
                residual_ax.text(.5, .5, "История невязки отсутствует", transform=residual_ax.transAxes,
                                 ha="center", va="center", fontsize=9)
            energy_ax.set_title("Накопленная диссипация")
            energy_ax.set_ylabel("Дж")
            residual_ax.set_title("Невязка равновесия")
            residual_ax.set_ylabel("Относительная невязка")
            for ax in (energy_ax, residual_ax):
                ax.set_xlabel("Лет после исходного состояния")
                ax.grid(alpha=.2)
            fig.suptitle("Генезис · разделённые берега и контакт", fontsize=16, fontweight="bold", y=.96)
            fig.text(.5, .91, f"Исходное состояние: {source_time_myr:.4f} млн лет · механическое продолжение: {elapsed_years:.4g} лет",
                     ha="center", fontsize=11)
            fig.text(.5, .865, "Точки показывают исходно соседние рёбра; фон — текущие материальные треугольники",
                     ha="center", fontsize=9)
            final = history[-1] if history else summary.get("final", {})
            fig.text(.5, .815, connectivity_text(final), ha="center", fontsize=9)
            note = ("Малые скольжения: независимые перемещения берегов, раскрытие, сжатие и трение.\n"
                    "Температура, орбита, повреждение и доступ воды зафиксированы в исходном состоянии.\n"
                    "Без общего поиска столкновений, субдукции и заполнения разрывов новой корой.")
            status = summary.get("status")
            if status and status != "completed":
                note += f"\nРасчёт остановлен: {status}."
            fig.text(.5, .065, note, ha="center", va="center", fontsize=9)
            fig.savefig(path, dpi=130)
        finally:
            plt.close(fig)
    return path
