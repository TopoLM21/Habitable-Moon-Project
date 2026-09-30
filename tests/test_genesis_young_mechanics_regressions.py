"""Saved force probes and moving-ownership metadata use current geometry."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json

import numpy as np
import yaml

from tectonics.checkpoint import save_checkpoint
from tectonics.dynamics import DynamicsParameters, angular_velocity_vectors
from tectonics.plate_velocity_diagnostics import diagnose_checkpoint, weighted_stats
from tectonics.genesis_starter_material import independent_mantle_omega
from tectonics.topology import PlateTopologyParameters
from test_genesis_starter_fracture_coupling import installed, source


def test_saved_initial_thin_lid_diagnostic_matches_installed_frozen_force_probe(source, tmp_path):
    model, cp, cfg, coupling, runner = installed(source)
    young = tmp_path / "young_context"
    young.mkdir()
    model.save_state(young / "starter_checkpoint.npz", coupling.source_state)
    coupling.fracture.save(young / "fracture_memory.npz")
    (young / "parameters.json").write_text(json.dumps({
        "format": "genesis-starter-run-0.1", **model.configuration}), encoding="utf-8")
    (tmp_path / "mature_config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    save_checkpoint(tmp_path / "mature_checkpoint", cp)
    digest = {str(path.relative_to(tmp_path)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in tmp_path.rglob("*") if path.is_file()}
    (tmp_path / "continuation.json").write_text(json.dumps({
        "checkpoint_sha256": digest, "checks": {}}), encoding="utf-8")
    before = deepcopy(cp.state)
    trace = {}
    runner.base.update_plate_dynamics(model.mesh, cp.state, cp.system, cp.baseline,
        model.thermal.radius_km, .5, **cfg["classification"],
        params=DynamicsParameters(**cfg["plate_dynamics"]), mantle_flow=cp.mantle_flow,
        subduction_memory=cp.subduction_memory, trace=trace)
    report = diagnose_checkpoint(tmp_path, step_myr=.5)
    for key in ("target_omega", "final_omega", "basal_driving_torque_nm",
                "ridge_torque_nm", "slab_torque_nm", "basal_drag_dissipation_w"):
        np.testing.assert_array_equal(report["trace"][key], trace[key])
    # The source is stored uncoupled, but the force probe must report current
    # transmitted velocity. Chemical crust is not a substitute for cold lid.
    lid = model.loading.sample(coupling.source_state.thermal_context).lid_thickness_km
    assert lid < cp.state.crust_thickness_km.min()
    expected_flow = independent_mantle_omega(model, coupling.source_state)
    expected_speed = np.linalg.norm(np.cross(expected_flow, model.mesh.centroids)
                                    * model.thermal.radius_km, axis=1)
    expected = weighted_stats(expected_speed, model.areas)
    np.testing.assert_allclose(report["mantle"]["local_velocity_km_per_myr"]["rms"],
                               expected["rms"], rtol=3e-15)
    assert report["uncoupled_prescribed_source_speed_km_per_myr"]["rms"] > expected["rms"]
    np.testing.assert_array_equal(cp.state.crust_age_myr, before.crust_age_myr)
    np.testing.assert_array_equal(cp.state.mantle_lithosphere_thickness_km,
                                  before.mantle_lithosphere_thickness_km)
    for path, expected_hash in digest.items():
        assert hashlib.sha256((tmp_path / path).read_bytes()).hexdigest() == expected_hash


def test_tracking_manager_retains_current_seeds_without_a_topology_event(source):
    model, cp, cfg, _, runner = installed(source)
    original = deepcopy(cp.system)
    for pid, plate in enumerate(cp.system.plates):
        plate.seed_cell = int(np.flatnonzero(cp.system.cell_plate != pid)[0])
    stale_seeds = [p.seed_cell for p in cp.system.plates]
    params = replace(PlateTopologyParameters(**cfg["plate_topology"]),
                     split_enabled=False, merge_enabled=False)
    manager = runner.base.PlateTopologyManager(params)
    updated, diag, events = manager.update(model.mesh, cp.state, cp.system, [],
                                           model.thermal.radius_km, 1.)
    assert not events and not diag.topology_changed
    np.testing.assert_array_equal(updated.cell_plate, original.cell_plate)
    np.testing.assert_array_equal(angular_velocity_vectors(updated), angular_velocity_vectors(original))
    for pid, plate in enumerate(updated.plates):
        assert updated.cell_plate[plate.seed_cell] == pid
    assert [p.seed_cell for p in cp.system.plates] == stale_seeds
