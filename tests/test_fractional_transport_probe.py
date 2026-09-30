"""CLI experiment keeps its source intact and restarts the same material flow."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

import run_fractional_transport_probe as probe
from tectonics.lithosphere import initialize_lithosphere
from tectonics.mesh import build_icosphere
from tectonics.plates import random_plate_system
from tectonics.fractional_surface import state_to_json
from tectonics.fractional_surface_io import load_fractional_checkpoint


@pytest.fixture
def source(monkeypatch, tmp_path):
    mesh = build_icosphere(0)
    system = random_plate_system(mesh, 2, 68, 0., .1, .1)
    state = initialize_lithosphere(mesh, system, continental_fraction=0., continental_nuclei=0)
    areas = mesh.physical_cell_areas_km2(100.)
    state.oceanic_volume_km3 = areas*2.
    state.mantle_lithosphere_thickness_km = np.full(mesh.cell_count, 10.)
    state.mantle_lithosphere_density_anomaly_kg_m3 = np.full(mesh.cell_count, 100.)
    n = mesh.cell_count
    fracture = SimpleNamespace(memory=SimpleNamespace(damage=state.tidal_damage.copy(),
        cooling_stress_pa=np.zeros(n), water_access=np.zeros(n), yield_ratio=np.zeros(n),
        strength_pa=np.full(n, 5e6), eligible=np.zeros(n, dtype=bool), consumed_band=np.zeros(n, dtype=bool)))
    model = SimpleNamespace(shell=SimpleNamespace(tensile_strength_pa=5e6), strength_factor=np.ones(n))
    path = tmp_path/"source.json"
    path.write_text("immutable source", encoding="utf-8")
    provenance = dict(source=str(path), source_sha256={str(path): probe.digest(path)},
        source_mechanics_version="young-mechanics-0.5", source_time_myr=0.,
        origin_time_myr=0., subdivisions=0, radius_km=100., newborn_crust_thickness_km=2.)
    checkpoint = SimpleNamespace(state=state, system=system)
    def load(_):
        return mesh, deepcopy(checkpoint), deepcopy(fracture), model, deepcopy(provenance)
    monkeypatch.setattr(probe, "load_probe_source", load)
    return path, mesh


def test_short_experiment_restart_is_identical_and_source_unchanged(source, tmp_path):
    path, mesh = source
    original = path.read_bytes()
    full = probe.execute_probe(path, tmp_path/"full", 1., .5)
    first = probe.execute_probe(path, tmp_path/"first", .5, .5)
    second = probe.execute_probe(path, tmp_path/"second", .5, .5,
        resume=tmp_path/"first/fractional_checkpoint.json")
    final_a, _ = load_fractional_checkpoint(tmp_path/"full/fractional_checkpoint.json", mesh, 100.)
    final_b, _ = load_fractional_checkpoint(tmp_path/"second/fractional_checkpoint.json", mesh, 100.)
    assert state_to_json(final_a) == state_to_json(final_b)
    assert full["history"] == second["history"]
    assert full["cumulative_losses"] == second["cumulative_losses"]
    assert first["source_unchanged"]
    assert path.read_bytes() == original
    assert max(abs(x) for x in full["history"][-1]["relative_residuals"].values()) < 1e-13


def test_probe_rejects_resume_with_different_motion(source, tmp_path):
    path, _ = source
    probe.execute_probe(path, tmp_path/"first", .5, .5)
    with pytest.raises(ValueError, match="identical source"):
        probe.execute_probe(path, tmp_path/"invalid", .5, .5,
            resume=tmp_path/"first/fractional_checkpoint.json", common_omega=[0., 0., .02])
    assert not (tmp_path/"invalid").exists()


def test_probe_output_is_never_overwritten(source, tmp_path):
    path, _ = source
    output = tmp_path/"existing"
    output.mkdir()
    marker = output/"report.json"
    marker.write_text("existing", encoding="utf-8")
    with pytest.raises(ValueError, match="never overwritten"):
        probe.execute_probe(path, output, .5, .5)
    assert marker.read_text(encoding="utf-8") == "existing"


def test_totals_consumes_a_loss_generator_once_for_all_four_quantities():
    values = (SimpleNamespace(**{key: amount*(index+1) for index, key in enumerate(probe.EXTENSIVE)})
              for amount in (3., 4.))
    assert probe.totals(values) == dict(zip(probe.EXTENSIVE, (7., 14., 21., 28.)))
