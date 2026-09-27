"""Whole starter invariants and a real molten-to-partition experiment."""
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest

from tectonics.genesis import GenesisParameters
from tectonics.genesis_shell import ShellParameters
from tectonics.genesis_starter import StarterModel, StarterParameters
from tectonics.genesis_tides import TidalParameters, SYNCHRONOUS_SPIN
from tectonics.mesh import build_icosphere, connected_components


def model(traction=20000., **parameters):
    thermal = GenesisParameters()
    tides = TidalParameters(enabled=True, spin_state=SYNCHRONOUS_SPIN)
    shell = ShellParameters(subdivisions=2, convective_traction_pa=traction)
    return StarterModel(build_icosphere(2), thermal, tides, shell, StarterParameters(**parameters))


def equal_states(a, b):
    for name in ("damage", "cooling_stress_pa", "water_access", "yield_ratio", "strength_pa",
                 "eligible", "split_band"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert np.array_equal(a.system.cell_plate, b.system.cell_plate)
    assert a.thermal_context.thermal == b.thermal_context.thermal
    assert a.thermal_context.orbit == b.thermal_context.orbit
    assert np.array_equal(a.thermal_context.column_enthalpy, b.thermal_context.column_enthalpy)
    assert a.time_myr == b.time_myr
    assert a.stopped_reason == b.stopped_reason
    assert json.dumps(a.events, sort_keys=True) == json.dumps(b.events, sort_keys=True)


def test_molten_start_has_no_preassigned_plates_crust_relief_or_damage():
    m = model()
    a = m.initial_state()
    d = m.diagnose(a)
    assert d["plate_count"] == 0 and d["domain_count"] == 1
    assert d["lid_thickness_km"] == 0 and d["mantle_melt_fraction"] == 1
    assert d["ocean_fraction"] == 0 and d["continental_volume_km3"] == 0
    assert d["initial_relief_m"] == 0 and d["max_damage"] == 0
    assert not np.any(a.damage) and not a.events
    assert not d["mature_handoff_ready"]


def test_real_stronger_load_partitions_only_previously_weakened_cells():
    m = model(50000.)
    initial = m.initial_state()
    saved = deepcopy(initial)
    a = m.advance(initial, 1.)
    equal_states(initial, saved)
    assert a.stopped_reason == "first_partition"
    assert len(a.system.plates) == 2
    assert 0 < a.time_myr <= 1
    assert np.all(a.damage[a.split_band] >= m.parameters.rupture_damage)
    for pid in range(2):
        cells = np.flatnonzero(a.system.cell_plate == pid)
        assert len(connected_components(cells, m.mesh.neighbors)) == 1
        assert m.areas[cells].sum() >= m.parameters.min_child_area_km2
    assert len(a.events) == 1
    assert a.time_myr == a.thermal_context.orbit.time_myr
    assert a.events[0]["time_myr"] == a.time_myr
    assert 0 < a.event_time_uncertainty_myr <= m.parameters.max_loading_interval_myr
    assert m.diagnose(a)["mature_handoff_ready"] is False
    with pytest.raises(ValueError, match="non-stopped"):
        m.advance(a, 2.)


def test_resume_reproduces_actual_cooling_damage_and_topology(tmp_path):
    m = model(50000.)
    a = m.advance(m.initial_state(), .7)
    path = tmp_path/"starter.npz"
    m.save_state(path, a)
    restored = m.load_state(path)
    equal_states(a, restored)
    whole = m.advance(a, 1.5)
    resumed = model(50000.).advance(restored, 1.5)
    equal_states(whole, resumed)
    assert whole.stopped_reason == "first_partition"
    m.save_state(path, whole)
    equal_states(whole, m.load_state(path))
    assert len(list(tmp_path.iterdir())) == 1


def test_configuration_or_seed_mismatch_cannot_silently_resume(tmp_path):
    m = model()
    path = tmp_path/"starter.npz"
    m.save_state(path, m.initial_state())
    with pytest.raises(ValueError, match="another model"):
        model(seed=12).load_state(path)


@pytest.mark.parametrize("name,value", [
    ("damage", float("nan")), ("damage", 1.2), ("water_access", -.1),
    ("strength_pa", 0.), ("yield_ratio", -.1),
])
def test_corrupt_material_checkpoint_rejected(tmp_path, name, value):
    m = model()
    path = tmp_path/"starter.npz"
    m.save_state(path, m.initial_state())
    with np.load(path, allow_pickle=False) as saved:
        content = {key: saved[key].copy() for key in saved.files}
    content[name][0] = value
    np.savez_compressed(path, **content)
    with pytest.raises(ValueError):
        m.load_state(path)


def test_eligibility_cannot_be_injected_into_intact_material(tmp_path):
    m = model()
    a = m.initial_state()
    a.eligible[:] = True
    with pytest.raises(ValueError, match="derived"):
        m.save_state(tmp_path/"bad.npz", a)


@pytest.mark.parametrize("changes", [
    {"seed": True}, {"seed": 1.5}, {"rupture_damage": 1},
    {"cooling_contrast_fraction": -1}, {"strength_variation_fraction": 1},
    {"tidal_mechanics": 1}, {"max_loading_interval_myr": 0},
    {"basal_drag_pa_s_m": float("inf")}, {"shear_cohesion_pa": 0},
])
def test_invalid_effective_parameters_rejected(changes):
    with pytest.raises(ValueError):
        StarterParameters(**changes).validate()


def test_fractional_end_is_respected_and_thermal_budgets_remain_separate():
    m = model()
    a = m.advance(m.initial_state(), 1.1)
    d = m.diagnose(a)
    assert a.time_myr == 1.1
    assert abs(d["thermal_energy_relative_residual"]) < 1e-10
    assert abs(d["column_energy_relative_residual"]) < 1e-10
    assert d["ocean_volume_km3"] <= m.thermal.water_volume_km3
    assert a.thermal_samples > 1  # Cheap forcing quadrature, not orbital steps.


def test_seed_changes_only_smooth_loading_not_initial_geometry_or_partition():
    a, b = model(seed=10), model(seed=11)
    assert np.array_equal(a.mesh.vertices, b.mesh.vertices)
    assert np.array_equal(a.initial_state().system.cell_plate, b.initial_state().system.cell_plate)
    assert not np.array_equal(a.mantle_tensor, b.mantle_tensor)
    assert abs(a.areas@a.anomaly/a.areas.sum()) < 1e-15


@pytest.mark.parametrize("target", [0., -1., float("nan"), float("inf"), True])
def test_bad_target_does_not_mutate_state(target):
    m = model()
    initial = m.initial_state()
    with pytest.raises(ValueError):
        m.advance(initial, target)
    assert initial.time_myr == 0 and not np.any(initial.damage)
