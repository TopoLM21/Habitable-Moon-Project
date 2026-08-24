"""Plots for v0.9 global thermal evolution."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def save_thermal_history(rows, path: str | Path, dpi: int = 160) -> None:
    if not rows: return
    t=np.asarray([r['time_myr'] for r in rows],float)
    temp=np.asarray([r['mantle_temperature_k'] for r in rows],float)
    lith=np.asarray([r['thermal_lithosphere_thickness_km'] for r in rows],float)
    fig,ax=plt.subplots(figsize=(10,6));ax.plot(t,temp,label='mantle temperature')
    ax.set_xlabel('Simulation time, Myr');ax.set_ylabel('Mantle temperature, K');ax.grid(True,alpha=.3)
    ax2=ax.twinx();ax2.plot(t,lith,linestyle='--',label='thermal lithosphere thickness');ax2.set_ylabel('Thermal lithosphere thickness, km')
    ax.set_title('v0.9 thermal state')
    lines=ax.get_lines()+ax2.get_lines();ax.legend(lines,[x.get_label() for x in lines],loc='best')
    fig.tight_layout();fig.savefig(path,dpi=dpi);plt.close(fig)


def save_heat_budget(rows, path: str | Path, dpi: int = 160) -> None:
    if not rows:return
    t=np.asarray([r['time_myr'] for r in rows],float)
    fig,ax=plt.subplots(figsize=(10,6))
    ax.plot(t,[r['convective_heat_flux_w_m2'] for r in rows],label='convective loss')
    ax.plot(t,[r['radiogenic_heat_flux_w_m2'] for r in rows],label='radiogenic')
    ax.plot(t,[r['tidal_heat_flux_w_m2'] for r in rows],label='tidal')
    ax.plot(t,[r['radiogenic_heat_flux_w_m2']+r['tidal_heat_flux_w_m2'] for r in rows],linestyle='--',label='total internal input')
    ax.set_xlabel('Simulation time, Myr');ax.set_ylabel('Global mean heat flux, W/m²');ax.grid(True,alpha=.3);ax.legend()
    ax.set_title('v0.9 mantle heat budget');fig.tight_layout();fig.savefig(path,dpi=dpi);plt.close(fig)


def save_activity_history(rows, path: str | Path, dpi: int = 160) -> None:
    if not rows:return
    t=np.asarray([r['time_myr'] for r in rows],float)
    fig,ax=plt.subplots(figsize=(10,6))
    ax.plot(t,[r['tectonic_activity_factor'] for r in rows],label='tectonic activity factor')
    ax.set_xlabel('Simulation time, Myr');ax.set_ylabel('Relative activity (initial = 1)');ax.grid(True,alpha=.3);ax.legend()
    ax2=ax.twinx();ax2.plot(t,np.asarray([r['viscosity_pa_s'] for r in rows])/1e20,linestyle='--',label='mantle viscosity')
    ax2.set_ylabel('Effective mantle viscosity, ×10²⁰ Pa s')
    ax.set_title('v0.9 thermal control of tectonic vigor')
    lines=ax.get_lines()+ax2.get_lines();ax.legend(lines,[x.get_label() for x in lines],loc='best')
    fig.tight_layout();fig.savefig(path,dpi=dpi);plt.close(fig)
