"""Spatial genesis diagnostics: a cooling lid and the contours of damaged patches."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from PIL import Image

from .raster import rasterize_cells


def crack_segments(mesh, failed_edges) -> list[np.ndarray]:
    """Great-circle mesh edges split at the map seam, never centroid links.

    Interpolation is on the unit sphere. Small segments at the antimeridian
    end at opposite map borders rather than drawing a chord across the map.
    """
    failed = np.asarray(failed_edges, dtype=bool)
    if failed.shape != (len(mesh.shared_edges),):
        raise ValueError("failed_edges must contain one flag per shared mesh edge")
    segments = []
    for index in np.flatnonzero(failed):
        _, _, u, v = mesh.shared_edges[index]
        a, b = mesh.vertices[[u, v]]
        angle = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
        fraction = np.linspace(0, 1, max(2, int(np.ceil(angle / np.deg2rad(2))) + 1))
        points = (1 - fraction[:, None]) * a + fraction[:, None] * b
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        lon = np.arctan2(points[:, 1], points[:, 0])
        lat = np.arcsin(np.clip(points[:, 2], -1, 1))
        current = [[lon[0], lat[0]]]
        for i in range(1, len(lon)):
            delta = lon[i] - lon[i - 1]
            if abs(delta) > np.pi:
                unwrapped = lon[i] - np.copysign(2 * np.pi, delta)
                if abs(unwrapped - lon[i - 1]) < 1e-12:
                    lon[i] = lon[i - 1]
                    current.append([lon[i], lat[i]])
                    continue
                border = np.copysign(np.pi, lon[i - 1])
                weight = (border - lon[i - 1]) / (unwrapped - lon[i - 1])
                crossing_lat = lat[i - 1] + weight * (lat[i] - lat[i - 1])
                current.append([border, crossing_lat])
                segments.append(np.asarray(current))
                current = [[-border, crossing_lat]]
            current.append([lon[i], lat[i]])
        if len(current) > 1:
            segments.append(np.asarray(current))
    return segments


def save_shell_snapshot(mesh, fields: dict, path: str | Path, time_myr: float,
                        summary: dict | None = None) -> Path:
    """Render four filled global maps; optional plot_limits keep custom scales stable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = summary or {}
    limits = summary.get("plot_limits", {})
    panels = (
        ("temperature_k", "Средняя температура твёрдого слоя", "K", "inferno", (250, 2500)),
        ("lid_thickness_km", "Толщина твёрдой покрышки", "км", "cividis", (0, 60)),
        ("tensile_stress_mpa", "Растягивающее напряжение", "МПа", "magma", (0, 100)),
        ("damage", "Повреждение и контуры слабых областей", "Доля повреждения", "magma", (0, 1)),
    )
    with plt.rc_context({"font.size": 10, "axes.titlesize": 12,
                         "figure.facecolor": "#f7f9fc", "text.color": "#172638",
                         "axes.labelcolor": "#29374d"}):
        fig, axes = plt.subplots(2, 2, figsize=(13, 8.5),
                                 subplot_kw={"projection": "mollweide"})
        try:
            fig.subplots_adjust(left=0.04, right=0.96, top=0.86, bottom=0.13,
                                hspace=0.42, wspace=0.15)
            for ax, (name, title, unit, cmap, defaults) in zip(axes.flat, panels):
                values = np.asarray(fields[name], dtype=float)
                if values.shape != (mesh.cell_count,) or not np.isfinite(values).all():
                    raise ValueError(f"{name} must contain one finite value per cell")
                lon, lat, raster = rasterize_cells(mesh, values)
                vmin, vmax = limits.get(name, defaults)
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
            segments = crack_segments(mesh, fields["failed_edges"])
            if segments:
                axes[1, 1].add_collection(LineCollection(
                    segments, colors="#65ecf2", linewidths=0.7, alpha=0.95), autolim=False)
            shell = summary.get("shell", summary)
            first = shell.get("first_fracture_time_myr")
            onset = (f"Порог повреждения: {first:.3f} млн лет" if first is not None else
                     "Слабые области отмечены голубыми контурами" if segments else "Порог повреждения ещё не достигнут")
            fig.suptitle(f"Генезис · формирование твёрдой покрышки\n{time_myr:.3f} млн лет  ·  {onset}",
                         fontsize=15, fontweight="bold", y=0.97)
            note = "Голубые линии — контуры повреждённых областей по рёбрам сетки. Подвижные плиты ещё не рассчитываются."
            if "intact_region_count" in shell:
                note += f"\nСвязных областей: {int(shell['intact_region_count'])}"
                if "cracked_length_km" in shell:
                    note += f"  ·  Длина контуров: {shell['cracked_length_km']:,.0f} км".replace(",", " ")
            if summary.get("status") == "shell_small_strain_limit" or shell.get("stopped_reason") == "shell_small_strain_limit":
                note += "\nДостигнут предел малых деформаций; дальнейшая эволюция требует изменения геометрии."
            fig.text(0.5, 0.035, note, ha="center", va="center", fontsize=9)
            fig.savefig(path, dpi=125)
        finally:
            plt.close(fig)
    return path


def save_shell_animation(frame_paths, path: str | Path, *, max_frames: int = 40,
                         duration_ms: int = 180) -> Path:
    """Save a bounded-memory GIF, including the first and final snapshots."""
    paths = list(frame_paths)
    if not paths:
        raise ValueError("At least one frame is required")
    if max_frames < 2 or duration_ms <= 0:
        raise ValueError("max_frames must be at least 2 and duration_ms must be positive")
    indices = np.unique(np.linspace(0, len(paths) - 1, min(len(paths), max_frames), dtype=int))
    frames = []
    try:
        for index in indices:
            with Image.open(paths[index]) as image:
                frame = image.convert("RGB")
                frame.thumbnail((1170, 765), Image.Resampling.LANCZOS)
                frames.append(frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
                frame.close()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms,
                       loop=0, optimize=False, disposal=2)
    finally:
        for frame in frames:
            frame.close()
    return path
