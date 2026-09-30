"""Independent read-only source, heat-owner and initial basal-force baseline.

No fractional coupling code is imported. The material reference mass and the
passive conductive column never become additional Genesis thermal reservoirs.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from tectonics.basal_coupling import prescribed_cell_basal_state, velocity_to_local_omega
from tectonics.genesis import SECONDS_PER_MYR
from tectonics.genesis_starter_continuation import load_starter_source, build_starter_continuation
from tectonics.genesis_starter_material import primary_material_inventory, mantle_source_parameters
from tectonics.mantle import MantleFlowState
from tectonics.simulation import load_config
from tectonics.young_plate_dynamics import basal_torque_system

SOURCE = ROOT / "results/genesis_runs/starter_20260928_192306_547551/starter_checkpoint.npz"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def context_payload(context):
    value = asdict(context)
    value["column_enthalpy"] = context.column_enthalpy.tolist()
    return value


def independent_basal_oracle(mesh, owner, areas_km2, radius_km, velocity_m_s, plate_count):
    """Fit dimensional rigid-body velocity with rectangular least squares."""
    omega = []
    radius_m = radius_km * 1000.
    for plate in range(plate_count):
        chosen = owner == plate
        positions = np.asarray(mesh.centroids)[chosen]
        design = np.stack([radius_m*np.cross(axis, positions) for axis in np.eye(3)], axis=2)
        weight = np.sqrt(areas_km2[chosen])
        matrix = (design*weight[:, None, None]).reshape(-1, 3)
        rhs = (velocity_m_s[chosen]*weight[:, None]).ravel()
        omega.append(np.linalg.lstsq(matrix, rhs, rcond=None)[0] * SECONDS_PER_MYR)
    return np.asarray(omega)


def run_baseline(output=HERE / "baseline.json", dt_myr=.25):
    inputs = {str(p): sha(p) for p in (SOURCE, SOURCE.parent / "parameters.json")}
    model, archived, _ = load_starter_source(SOURCE)
    cfg = load_config(ROOT / "configs/canonical_moon.yaml")
    cfg["young_shell"] = {"mechanics_model_version": "young-mechanics-0.5"}
    bundle, cfg, imported = build_starter_continuation(model, archived, cfg)
    cp = bundle.checkpoint
    sample = model.loading.sample(archived.thermal_context)
    radius = model.thermal.radius_km
    areas = model.mesh.physical_cell_areas_km2(radius)
    owner = np.asarray(cp.state.cell_plate)
    count = len(cp.system.plates)
    basal = prescribed_cell_basal_state(model.mesh, mantle_source_parameters(model), sample.lid_thickness_km)
    field = velocity_to_local_omega(model.mesh.centroids, basal.equilibrium_velocity_m_s, radius)
    drag, torque, velocity = basal_torque_system(model.mesh, owner, count, radius,
        MantleFlowState(cp.state.time_myr, field, 1.), model.parameters.basal_drag_pa_s_m)
    omega = np.linalg.solve(drag, torque[..., None])[..., 0] * SECONDS_PER_MYR
    oracle = independent_basal_oracle(model.mesh, owner, areas, radius, velocity, count)
    np.testing.assert_allclose(omega, oracle, rtol=5e-13, atol=1e-18)
    plate_u = np.cross(omega[owner], model.mesh.centroids)*radius*1000./SECONDS_PER_MYR
    power = float(np.sum(areas*1e6*np.sum(basal.transmitted_traction_pa*plate_u, axis=1)))
    dissipation = float(model.parameters.basal_drag_pa_s_m*np.sum(areas*1e6*np.sum(plate_u**2, axis=1)))
    speed = np.linalg.norm(plate_u, axis=1)*SECONDS_PER_MYR/1000.
    before = context_payload(archived.thermal_context)
    following, samples = model.loading.advance(archived.thermal_context, archived.time_myr+dt_myr,
        max_sample_myr=model.parameters.max_loading_interval_myr)
    assert context_payload(archived.thermal_context) == before
    material_state = deepcopy(archived)
    material_state.thermal_context = following
    inventory = primary_material_inventory(model, material_state)
    end = model.loading.sample(following)
    result = dict(scope="Read-only source and heat-owner baseline; basal-only initial equilibrium, not a plate-speed prediction.",
        source_sha256=inputs, source_time_myr=archived.time_myr, plate_count=count,
        cell_count=model.mesh.cell_count, radius_km=radius,
        source_import=imported,
        basal=dict(total_lid_thickness_km=sample.lid_thickness_km, omega_rad_myr=omega.tolist(),
            least_squares_omega_rad_myr=oracle.tolist(),
            oracle_max_absolute_error_rad_myr=float(np.max(np.abs(omega-oracle))),
            mean_speed_mm_yr=float(areas@speed/areas.sum()), max_speed_mm_yr=float(speed.max()),
            source_power_w=power, drag_dissipation_w=dissipation,
            relative_power_residual=abs(power-dissipation)/max(abs(power), 1.),
            convention="Pure uniform primordial H; old cell-to-vertex coupling is exact here. No slab/ridge force."),
        heat=dict(duration_myr=dt_myr, context=context_payload(following),
            sample_count=len(samples), energy_relative_residual=end.thermal["relative_energy_residual"],
            partition_energy_relative_residual=inventory.thermal_energy_relative_residual,
            silicate_mass_relative_residual=inventory.silicate_mass_relative_residual,
            water_mass_relative_residual=inventory.water_mass_relative_residual,
            input_context_unchanged=True, heat_owner="Genesis only; parcel deficits and reference column remain passive."),
        source_unchanged=all(sha(p)==value for p, value in inputs.items()))
    assert result["source_unchanged"]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run_baseline()
    print(json.dumps(dict(basal=result["basal"], source_unchanged=result["source_unchanged"],
        heat_residual=result["heat"]["energy_relative_residual"]), indent=2))
