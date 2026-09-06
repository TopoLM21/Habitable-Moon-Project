"""Diagnostic instrumentation must leave small-model physics byte-identical."""
from copy import deepcopy
from dataclasses import fields, is_dataclass
import inspect
import struct
import sys

import numpy as np
import pytest

from analysis.profile_cpu_stages import Budget, install_detailed_physics_timers, install_timers


def assert_exact(left, right):
    assert type(left) is type(right)
    if isinstance(left, np.ndarray):
        assert left.shape == right.shape and left.dtype == right.dtype
        assert left.tobytes() == right.tobytes()
    elif is_dataclass(left):
        for field in fields(left):
            assert_exact(getattr(left, field.name), getattr(right, field.name))
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_exact(a, b)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_exact(left[key], right[key])
    elif isinstance(left, float):
        assert struct.pack("d", left) == struct.pack("d", right)
    else:
        assert left == right


@pytest.mark.parametrize("outer_timers", [False, True])
def test_detailed_timers_preserve_dynamics_and_relief(monkeypatch, outer_timers):
    # Load runner aliases before installation, exactly as the diagnostic child.
    import run_long_evolution_v131 as runner
    import tectonics.dynamics as dynamics
    import tectonics.lithosphere as lithosphere
    import tectonics.topography as topography
    from tectonics.flexure import FlexureParameters
    from tectonics.mesh import build_icosphere
    from tectonics.plates import random_plate_system
    from tectonics.topology import PlateTopologyManager

    mesh = build_icosphere(1)
    system = random_plate_system(mesh, 4, 82342, 0.2, 0.1, 0.3)
    state = lithosphere.initialize_lithosphere(mesh, system, 0.55, 4, 7.0, 35.0, 500.0,
                                              radius_km=5287.0)
    continental = state.crust_type == int(lithosphere.CrustType.CONTINENTAL)
    state.crust_thickness_km[continental] = np.linspace(40.0, 80.0, np.count_nonzero(continental))
    params = dynamics.DynamicsParameters()
    dynamics_args = (mesh, state, system, system, 5287.0, 4.0, 4.0, 1.0, params)
    expected_dynamics = dynamics.update_plate_dynamics(*dynamics_args)
    assert expected_dynamics[1].mean_gpe_drive > 0.0
    bounds = expected_dynamics[2]
    topo_params = topography.TopographyParameters()
    previous = topography.initialize_topography(mesh, state, bounds, topo_params)
    topo_args = (mesh, state, bounds, previous, 4.0, 5287.0, topo_params)
    topo_keywords = {"flexure_params": FlexureParameters(), "gravity_m_s2": 1.62}
    expected_topo = topography.advance_topography(*topo_args, **topo_keywords)
    saved_state = deepcopy(state)

    # Register no-op patches before instrumentation so teardown restores every
    # alias and class method modified by either installer, including runner.main.
    for name, module in list(sys.modules.items()):
        if module and name.startswith(("tectonics.", "visualization.", "run_long_evolution_v")):
            for alias, value in list(vars(module).items()):
                if inspect.isfunction(value):
                    monkeypatch.setattr(module, alias, value)
    monkeypatch.setattr(PlateTopologyManager, "update", PlateTopologyManager.update)
    monkeypatch.setattr(lithosphere, "_cpu_stage_scope", None, raising=False)
    monkeypatch.setattr(dynamics, "_gpu_profile_scope", None, raising=False)
    monkeypatch.setattr(runner.base, "_cpu_stage_scope", None, raising=False)

    budget = Budget()
    if outer_timers:
        install_timers(runner, budget)
    install_detailed_physics_timers(budget)
    with budget.scope("physics/other"):
        actual_dynamics = dynamics.update_plate_dynamics(*dynamics_args)
        actual_topo = topography.advance_topography(*topo_args, **topo_keywords)
    assert_exact(expected_dynamics, actual_dynamics)
    assert_exact(expected_topo, actual_topo)
    assert_exact(saved_state, state)
    assert budget.current == "imports_and_instrumentation"
    expected = {
        "physics/plate_dynamics", "physics/dynamics_boundary_loop", "physics/dynamics_gpe_loop",
        "physics/dynamics_ridge_factors", "physics/topography_forcing",
        "physics/topography_equilibrium_other", "physics/topography_erosion", "physics/flexure",
        "physics/material_endmembers",
    }
    for category in expected:
        assert budget.calls[category] >= 1, category
        assert budget.seconds[category] >= 0.0, category
    assert budget.calls["physics/dynamics_boundary_loop"] == 1
    assert budget.calls["physics/dynamics_gpe_loop"] == 1
