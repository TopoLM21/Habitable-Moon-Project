"""Read-only source/flow/stress provenance probe on saved corrected checkpoints.

Re-evaluated fields are counterfactual frozen probes, not simulated histories.
Production physics and checkpoints are not modified.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tectonics.genesis import SECONDS_PER_MYR
from tectonics.genesis_shell import mantle_traction
from tectonics.genesis_starter_continuation import load_starter_source, _load_cp
from tectonics.mantle import MantleFlowState, plate_rigid_mantle_fit
from tectonics.plate_velocity_diagnostics import weighted_stats
from tectonics.simulation import load_config


def main():
    source = ROOT / "results/genesis_runs/starter_20260928_192306_547551/starter_checkpoint.npz"
    model, state, _ = load_starter_source(source)
    mesh, radius = model.mesh, model.thermal.radius_km
    x, area = mesh.centroids, model.areas
    shell = replace(model.shell, seed=model.parameters.seed)
    drag = model.parameters.basal_drag_pa_s_m
    experiments = ROOT / "analysis/plate_velocity_validation/experiments/velocity_least_squares"
    saved = json.loads((experiments / "elapsed_0400/continuation.json").read_text(encoding="utf-8"))

    def field(thickness):
        traction = mantle_traction(mesh, shell, np.full(mesh.cell_count, thickness))
        v = traction[mesh.faces].mean(axis=1) / drag * SECONDS_PER_MYR / 1000.
        v -= x * np.sum(v*x, axis=1)[:, None]
        return np.cross(x, v) / radius

    def speeds(w):
        return weighted_stats(np.linalg.norm(np.cross(w, x) * radius, axis=1), area)

    def coupling(h):
        return float(-np.expm1(-h / shell.traction_coupling_depth_km))

    first = saved["history"][0]
    qref, c0 = first["mantle_to_surface_flux_w_m2"], coupling(first["mechanical_lid_thickness_km"])
    origin_field = field(first["mechanical_lid_thickness_km"])
    rows = []
    for i in (0, 1, 5, 9, 10, 50, 100, 200, 400):
        row = saved["history"][i]
        h, q = row["mechanical_lid_thickness_km"], row["mantle_to_surface_flux_w_m2"]
        c = coupling(h)
        live = field(h)
        rows.append(dict(elapsed_myr=i, time_myr=row["time_myr"], lid_km=h,
            coupling=c, coupling_relative_to_partition=c/c0,
            counterfactual_current_loading_velocity=speeds(live),
            actual_surface_mean=row["mean_surface_speed_km_myr"],
            reference_total_flux=qref, current_total_flux=q,
            total_flux_ratio=q/qref, activity=float(np.clip(q/qref,.35,1.8)),
            source_solid_flux=first["solid_convective_heat_flux_w_m2"],
            current_solid_flux=row["solid_convective_heat_flux_w_m2"],
            source_viscosity=first["viscosity_pa_s"], current_viscosity=row["viscosity_pa_s"],
            source_rayleigh=first["rayleigh_number"], current_rayleigh=row["rayleigh_number"],
            current_yield_max=row["young_fracture"]["max_yield_ratio"]))

    checkpoints = []
    for elapsed in (5, 50, 100, 200, 400):
        folder = experiments / f"elapsed_{elapsed:04d}"
        cfg = load_config(folder / "mature_config.yaml")
        cp = _load_cp(folder / "mature_checkpoint", cfg)
        current = cp.mantle_flow.cell_omega_rad_per_myr
        h = saved["history"][elapsed]["mechanical_lid_thickness_km"]
        live = field(h)
        def rigid(w):
            mantle = MantleFlowState(cp.state.time_myr, w, cp.mantle_flow.formation_rms_rad_per_myr)
            fit = plate_rigid_mantle_fit(mesh, cp.state.cell_plate, len(cp.system.plates), radius, mantle)
            v = np.cross(fit.omega_rad_per_myr[cp.state.cell_plate], x)*radius
            return weighted_stats(np.linalg.norm(v, axis=1), area)
        v0, v1 = np.cross(origin_field, x), np.cross(current, x)
        numerator = np.sum(area[:,None]*v0*v1)
        cosine = float(numerator/np.sqrt(np.sum(area[:,None]*v0*v0)*np.sum(area[:,None]*v1*v1)))
        projection = float(numerator/np.sum(area[:,None]*v0*v0))
        radial = np.sum(current*x, axis=1)
        radial_fraction = float(np.sum(area*radial**2)/np.sum(area*np.sum(current*current,axis=1)))
        checkpoints.append(dict(elapsed_myr=elapsed, time_myr=cp.state.time_myr,
            actual_mantle_local=speeds(current), counterfactual_current_loading_local=speeds(live),
            actual_rigid_fit=rigid(current), counterfactual_current_loading_rigid_fit=rigid(live),
            source_pattern_velocity_cosine=cosine, source_pattern_velocity_amplitude=projection,
            nonphysical_radial_omega_squared_fraction=radial_fraction,
            activity=cp.thermal.tectonic_activity_factor))

    result = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        history_source=str(experiments / "elapsed_0400/continuation.json"),
        note="All live-loading values are frozen diagnostic re-evaluations with unchanged nominal traction/drag, not predictions or physical repairs.",
        coupling_at_partition=c0, source_field=speeds(origin_field),
        beta_pa_s_m=drag, nominal_traction_pa=shell.convective_traction_pa,
        history=rows, checkpoints=checkpoints)
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
