"""Diagnostic-only SI basal torque audit and frozen kinematic counterfactuals.

No model coefficient or checkpoint is modified. The SI interpretation below
is a hypothesis to test consistency, not a replacement production force law.
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

from tectonics.dynamics import angular_velocity_vectors, system_from_omega
from tectonics.genesis import SECONDS_PER_MYR
from tectonics.genesis_starter_continuation import _load_cp, load_starter_source
from tectonics.kinematics import classify_boundaries
from tectonics.mantle import plate_rigid_mantle_fit
from tectonics.plate_velocity_diagnostics import weighted_stats, boundary_budget
from tectonics.simulation import load_config


def audit(path, source_coupling):
    cfg = load_config(path/"mature_config.yaml")
    model, young, _ = load_starter_source(path/"young_context/starter_checkpoint.npz")
    cp = _load_cp(path/"mature_checkpoint", cfg)
    x, owner = model.mesh.centroids, cp.state.cell_plate
    radius_m = model.thermal.radius_km*1000.
    areas = model.areas*1e6
    n = len(cp.system.plates)
    beta = model.parameters.basal_drag_pa_s_m
    fit = plate_rigid_mantle_fit(model.mesh, owner, n, model.thermal.radius_km, cp.mantle_flow)
    omega = angular_velocity_vectors(cp.system)
    f = cfg["plate_dynamics"]["mantle_memory_fraction"]
    flow_ms = np.cross(cp.mantle_flow.cell_omega_rad_per_myr, x)*radius_m/SECONDS_PER_MYR
    def torque(w):
        velocity = np.cross(w[owner], x)*radius_m/SECONDS_PER_MYR
        shear = beta*(flow_ms-velocity)
        local = radius_m*np.cross(x, shear)*areas[:, None]
        tor = np.array([local[owner == p].sum(axis=0) for p in range(n)])
        return tor, weighted_stats(np.linalg.norm(shear, axis=1), areas)
    tload, _ = torque(np.zeros_like(omega))
    tactual, shear = torque(omega)
    tfit, fit_shear = torque(fit.omega_rad_per_myr)
    ttarget, _ = torque(f*fit.omega_rad_per_myr)
    np.testing.assert_allclose(ttarget, (1-f)*tload, rtol=2e-12, atol=1e12)
    assert np.max(np.linalg.norm(tfit, axis=1)/np.linalg.norm(tload, axis=1)) < 1e-10
    rows = []
    for p in range(n):
        rows.append(dict(plate=p, loading_torque_Nm=tload[p].tolist(),
            actual_unbalanced_basal_torque_Nm=tactual[p].tolist(),
            actual_residual_to_loading_ratio=float(np.linalg.norm(tactual[p])/np.linalg.norm(tload[p])),
            target_before_gauge_residual_to_loading_ratio=float(np.linalg.norm(ttarget[p])/np.linalg.norm(tload[p])),
            best_fit_residual_to_loading_ratio=float(np.linalg.norm(tfit[p])/np.linalg.norm(tload[p]))))
    masks = np.bincount(owner, weights=areas, minlength=n)
    scenarios = []
    candidates = [
        ("actual_checkpoint", omega, False),
        ("existing_mantle_only_target", f*fit.omega_rad_per_myr, True),
        ("same_field_zero_external_torque_LS", fit.omega_rad_per_myr, True),
        ("hypothetical_saturated_source_existing_memory", f*fit.omega_rad_per_myr/source_coupling, True),
        ("hypothetical_saturated_source_zero_external_torque", fit.omega_rad_per_myr/source_coupling, True)]
    for name, value, center in candidates:
        value = value.copy()
        if center:
            value -= np.sum(masks[:, None]*value, axis=0)/masks.sum()
        speed = np.linalg.norm(np.cross(value[owner], x), axis=1)*model.thermal.radius_km
        system = system_from_omega(owner, cp.system, value)
        boundaries = classify_boundaries(model.mesh, system, model.thermal.radius_km, **cfg["classification"])
        scenarios.append(dict(name=name, speed_km_myr=weighted_stats(speed, areas),
            boundaries=boundary_budget(model.mesh, boundaries, model.thermal.radius_km)))
    snapshot = model.loading.sample(young.thermal_context)
    return dict(source=str(path), source_sha256=hashlib.sha256((path/"mature_checkpoint/state.npz").read_bytes()).hexdigest(),
        age_myr=cp.state.time_myr, beta_pa_s_m=beta, memory_fraction=f,
        equivalent_unmodelled_anchoring_drag_ratio=1./f-1.,
        torque_note="Conditional SI test: tau=beta*(u_m-u_plate), using the saved already-coupled mantle velocity. This is not the mature solver's force law. Nonzero residual needs an explicit countertorque, or the velocity map is empirical.",
        gauge_note="Counterfactual speeds are centered for comparison. Torque ratios use the saved mantle frame; target before centering isolates .22 exactly. A future SI solver must transform mantle and plates together when changing frame.",
        actual_basal_shear_pa=shear, best_fit_basal_shear_pa=fit_shear, plates=rows, scenarios=scenarios,
        current_lid_coupling=float(-np.expm1(-snapshot.lid_thickness_km/model.shell.traction_coupling_depth_km)),
        thermal_activity=cp.thermal.tectonic_activity_factor,
        raw_heat_flux_ratio=snapshot.thermal["mantle_to_surface_flux_w_m2"]/cp.thermal.reference_convective_flux_w_m2)


def main():
    source = ROOT/"results/genesis_runs/starter_20260928_192306_547551/starter_checkpoint.npz"
    model, state, _ = load_starter_source(source)
    h = model.loading.sample(state.thermal_context).lid_thickness_km
    coupling = float(-np.expm1(-h/model.shell.traction_coupling_depth_km))
    roots = [ROOT/"results/gui_runs/genesis_20260928_192334_470716/gui_checkpoint_0000110p8781_Myr",
             ROOT/"analysis/plate_velocity_validation/experiments/velocity_least_squares/elapsed_0400"]
    rows = [audit(path, coupling) for path in roots]
    out = Path(__file__).with_name("torque_probe.json")
    out.write_text(json.dumps(dict(source_coupling=coupling,
        scope="frozen geometry diagnostic counterfactuals, not trajectories or proposed speed multipliers",
        cases=rows), indent=2)+"\n", encoding="utf-8")
    for row in rows:
        print(row["age_myr"],[(r["name"],r["speed_km_myr"]["mean"],r["boundaries"]["active_length_fraction"]) for r in row["scenarios"]])
        print("actual torque/loading", [p["actual_residual_to_loading_ratio"] for p in row["plates"]])


if __name__ == "__main__":
    main()
