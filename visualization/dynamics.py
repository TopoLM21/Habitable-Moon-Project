"""Visual diagnostics for v0.5 dynamic plate motion."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from tectonics.dynamics import DynamicsDiagnostics, angular_velocity_vectors
from tectonics.kinematics import BoundaryType
from tectonics.lithosphere import CrustType, LithosphereState
from tectonics.mesh import SphereMesh
from tectonics.plates import PlateSystem


def _lon_lat(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon = np.arctan2(points[:, 1], points[:, 0])
    lat = np.arcsin(np.clip(points[:, 2], -1.0, 1.0))
    return lon, lat


def save_dynamics_frame(mesh: SphereMesh, state: LithosphereState, system: PlateSystem, boundaries, dyn: DynamicsDiagnostics | None, path: str | Path, dpi: int = 120) -> None:
    lon, lat = _lon_lat(mesh.centroids)
    fig = plt.figure(figsize=(12, 6.8))
    ax = fig.add_subplot(111, projection='mollweide')
    # Ocean vs continent as 0/1, with plate ownership lightly overplotted.
    ax.scatter(lon, lat, c=state.crust_type, cmap='terrain', vmin=0, vmax=1, s=2.5, linewidths=0, rasterized=True)
    for kind in BoundaryType:
        pts = np.asarray([b.midpoint for b in boundaries if b.boundary_type == kind])
        if len(pts):
            blon, blat = _lon_lat(pts)
            ax.scatter(blon, blat, s=3.5, linewidths=0, alpha=0.7, label=kind.name.lower())

    # Euler poles: marker size scales with speed.  Axis contains the sign, speed is nonnegative in v0.5.
    axes = np.asarray([p.euler_axis for p in system.plates], dtype=float)
    plon, plat = _lon_lat(axes)
    speeds = np.rad2deg(np.asarray([abs(p.angular_speed_rad_per_myr) for p in system.plates]))
    ax.scatter(plon, plat, s=30.0 + 85.0 * speeds, marker='*', linewidths=0.6, label='Euler poles')

    cont_frac = float(np.mean(state.crust_type == int(CrustType.CONTINENTAL)))
    title = f"v0.5 force-driven plates — t={state.time_myr:g} Myr | continent cells={100*cont_frac:.1f}%"
    if dyn is not None:
        title += f"\nmean speed={dyn.mean_speed_deg_per_myr:.3f}°/Myr max={dyn.max_speed_deg_per_myr:.3f}°/Myr | mean pole turn={dyn.mean_axis_turn_deg:.2f}°/step"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def save_speed_history(rows: list[dict], path: str | Path, dpi: int = 160) -> None:
    t=np.asarray([r['time_myr'] for r in rows]); mean=np.asarray([r['mean_speed_deg_per_myr'] for r in rows]); mx=np.asarray([r['max_speed_deg_per_myr'] for r in rows]); turn=np.asarray([r['mean_axis_turn_deg'] for r in rows])
    fig,ax=plt.subplots(figsize=(10,6)); ax.plot(t,mean,label='mean plate speed'); ax.plot(t,mx,label='maximum plate speed'); ax.set_xlabel('Time, Myr'); ax.set_ylabel('Angular speed, deg/Myr'); ax.grid(True,alpha=0.3); ax.legend(loc='upper left')
    ax2=ax.twinx(); ax2.plot(t,turn,linestyle='--',label='mean Euler-pole turn'); ax2.set_ylabel('Mean pole turn per step, deg'); ax.set_title('v0.5 dynamic Euler motion'); fig.tight_layout(); fig.savefig(path,dpi=dpi); plt.close(fig)


def save_force_history(rows: list[dict], path: str | Path, dpi: int = 160) -> None:
    t=np.asarray([r['time_myr'] for r in rows]);
    fig,ax=plt.subplots(figsize=(10,6))
    ax.plot(t,[r['ridge_boundary_length_km'] for r in rows],label='divergent / ridge')
    ax.plot(t,[r['slab_boundary_length_km'] for r in rows],label='subduction candidates')
    ax.plot(t,[r['continental_collision_length_km'] for r in rows],label='continent collision')
    ax.plot(t,[r['transform_boundary_length_km'] for r in rows],label='transform')
    ax.set_xlabel('Time, Myr'); ax.set_ylabel('Boundary length, km'); ax.set_title('Boundary processes driving/resisting plates'); ax.grid(True,alpha=0.3); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=dpi); plt.close(fig)


def save_euler_poles(system: PlateSystem, path: str | Path, title: str='Final Euler poles', dpi: int=180) -> None:
    axes=np.asarray([p.euler_axis for p in system.plates],dtype=float); lon,lat=_lon_lat(axes); speeds=np.rad2deg(np.asarray([abs(p.angular_speed_rad_per_myr) for p in system.plates]))
    fig=plt.figure(figsize=(10,5.8)); ax=fig.add_subplot(111,projection='mollweide'); sc=ax.scatter(lon,lat,c=speeds,s=90,linewidths=0); fig.colorbar(sc,ax=ax,orientation='horizontal',pad=0.08,fraction=0.05,label='Angular speed, deg/Myr')
    for i,(x,y) in enumerate(zip(lon,lat)): ax.text(x,y,str(i),fontsize=8)
    ax.grid(True,alpha=0.3); ax.set_title(title); fig.tight_layout(); fig.savefig(path,dpi=dpi); plt.close(fig)


def build_gif(frame_paths: list[Path], output_path: str | Path, frame_duration_ms: int=350) -> None:
    images=[Image.open(p).convert('P',palette=Image.Palette.ADAPTIVE) for p in frame_paths]
    try:
        images[0].save(output_path,save_all=True,append_images=images[1:],duration=int(frame_duration_ms),loop=0,optimize=False)
    finally:
        for im in images: im.close()
