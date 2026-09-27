"""Material heat, clock and rollback contracts of the research thermal owner."""
from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

import analysis.genesis_moving_genesis_validation as owner
from analysis.genesis_path_dynamics_validation import _exact
from tectonics.genesis import diagnose
from tectonics.genesis_contact import ContactParameters
from tectonics.genesis_crack_path import ReferenceCrackPath
from tectonics.genesis_moving_tied import MovingTiedRetry
from tectonics.genesis_path_mesh import insert_crack_path
from tests.test_genesis_contact import _source_bytes


@pytest.fixture
def case(tmp_path):
    # A small cold material fixture exercises ownership, not the historical
    # thermal-gate localization or geological time of a planetary crack.
    _, source, (state, thermal, orbit), _ = _source_bytes(tmp_path,
        traction_pa=20000., elastic=(0., 0., 0.), active=False)
    state = replace(state, damage=np.zeros_like(state.damage))
    mesh = source.mesh_for(state)
    points = np.array([[1., .12, .23], [1., .42, .49], [1., .72, .53]])
    points /= np.linalg.norm(points, axis=1)[:, None]
    path = ReferenceCrackPath(points, state.radius_km)
    insertion = insert_crack_path(mesh, path)
    context = SimpleNamespace(thermal=thermal, orbit=orbit,
        column_enthalpy=state.column_enthalpy, damage=state.damage,
        water_access=state.water_access, boundary_energy_j=state.boundary_energy_j)
    precursor = SimpleNamespace(source_model=source, before=state,
        mesh=mesh, radius_m=state.radius_km*1000.,
        model=SimpleNamespace(parameters=ContactParameters()),
        basis=SimpleNamespace(insertion=insertion), context=context)
    return owner.MovingGenesisCase(precursor)


def test_pure_thermal_mechanical_trial_preserves_columns_and_energy(case):
    state, context = case.initial, case.context
    old_state, old_context, old_mass = deepcopy(state), deepcopy(context), case.mass.copy()
    old_child_mass = case.support.extensive(case.mass)
    after, following = case.trial(state, context, 40.)
    assert _exact(state, old_state) and _exact(context, old_context)
    np.testing.assert_array_equal(case.mass, old_mass)
    np.testing.assert_array_equal(case.support.extensive(case.mass), old_child_mass)
    assert following.thermal.time_myr == following.orbit.time_myr
    assert following.thermal.time_myr == pytest.approx(
        case.gate_age_myr+after.elapsed_years/1e6, rel=0, abs=2e-15)
    energy = float(np.sum(case.mass*following.column_enthalpy))
    residual = energy-case.initial_column_energy_j-following.boundary_energy_j
    assert abs(residual)/case.initial_column_energy_j < 2e-14
    assert abs(diagnose(following.thermal, case.source_model.thermal)[
        "relative_energy_residual"]) < 2e-14
    observation = case.observe(after, following)
    assert observation.state.active_interval is None
    assert observation.parent_energy_relative_error < 2e-14
    assert observation.parent_force_relative_error < 2e-14


def test_rejected_mechanics_does_not_commit_successful_thermal_predictor(case, monkeypatch):
    state, context = deepcopy(case.initial), deepcopy(case.context)
    old_state, old_context = deepcopy(state), deepcopy(context)
    seen = []
    def reject(accepted, loading):
        seen.append(loading.dt_years)
        raise MovingTiedRetry("independent_rejected_mechanics")
    monkeypatch.setattr(case.model, "trial", reject)
    with pytest.raises(MovingTiedRetry, match="independent_rejected_mechanics"):
        case.trial(state, context, 40.)
    assert seen == [40.]
    assert _exact(state, old_state) and _exact(context, old_context)


def test_following_heat_step_uses_current_material_geometry_and_elastic_frame(case, monkeypatch):
    state, context = case.trial(case.initial, case.context, 40.)
    original = owner.advance_thermal_loading
    seen = []
    def checked(source, **kwargs):
        np.testing.assert_array_equal(kwargs["mesh"].vertices, state.vertices)
        assert kwargs["radius_km"] == state.radius_m/1000.
        np.testing.assert_array_equal(kwargs["elastic_strain"], state.elastic_strain)
        np.testing.assert_array_equal(kwargs["layer_mass_kg"], case.mass)
        np.testing.assert_array_equal(kwargs["column_enthalpy"], context.column_enthalpy)
        seen.append(kwargs["target_myr"])
        return original(source, **kwargs)
    monkeypatch.setattr(owner, "advance_thermal_loading", checked)
    after, following = case.trial(state, context, 17.)
    assert len(seen) == 1
    assert after.elapsed_years == 57.
    assert following.thermal.time_myr == pytest.approx(57e-6, rel=0, abs=2e-15)


def test_separate_mechanics_and_thermal_files_resume_identically(case, tmp_path):
    state, context = case.trial(case.initial, case.context, 40.)
    case.model.save_state(tmp_path/"mechanics.npz", state)
    case.save_context(tmp_path/"thermal.npz", context)
    resumed_state = case.model.load_state(tmp_path/"mechanics.npz")
    resumed_context = case.load_context(tmp_path/"thermal.npz")
    assert _exact(state, resumed_state)
    assert _exact(context, resumed_context)
    direct, direct_context = case.trial(state, context, 17.)
    resumed, following = case.trial(resumed_state, resumed_context, 17.)
    assert _exact(direct, resumed)
    assert _exact(direct_context, following)
    assert asdict(direct_context.thermal) == asdict(following.thermal)


@pytest.mark.parametrize("clock_mismatch", ["earlier_context", "earlier_mechanics"])
def test_unpaired_mechanics_and_thermal_clocks_are_rejected(case, clock_mismatch):
    state, context = case.trial(case.initial, case.context, 40.)
    if clock_mismatch == "earlier_context":
        context = case.context
    else:
        state = case.initial
    with pytest.raises(ValueError, match="clock|pair|time"):
        case.trial(state, context, 17.)
