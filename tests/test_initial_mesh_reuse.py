from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tectonics.cpu_runtime import CpuExecution, current_execution
from tectonics.simulation import build_initial_mesh, build_prototype, load_config


@pytest.fixture
def config():
    root = Path(__file__).resolve().parents[1]
    value = load_config(root / "configs" / "canonical_moon.yaml")
    value["mesh"]["subdivisions"] = 1
    value["plates"]["count"] = 4
    return value


def test_reference_and_opt_out_do_not_reuse_geometry(config):
    assert build_initial_mesh(config) is not build_initial_mesh(config)
    with CpuExecution(reuse_initial_mesh=False):
        assert build_initial_mesh(config) is not build_initial_mesh(config)


def test_only_geometry_is_shared_and_prototypes_stay_exact(config):
    reference = build_prototype(config)
    with CpuExecution() as execution:
        mesh = build_initial_mesh(config)
        actual = build_prototype(config)
        other = build_prototype(config)
        assert actual.mesh is mesh is other.mesh
        assert actual.plates is not other.plates
        assert not np.shares_memory(actual.plates.cell_plate, other.plates.cell_plate)
        for name in ("vertices", "faces", "centroids", "areas_unit_sphere"):
            want, got = getattr(reference.mesh, name), getattr(mesh, name)
            assert want.dtype == got.dtype and want.tobytes() == got.tobytes()
        assert mesh.neighbors == reference.mesh.neighbors
        assert mesh.shared_edges == reference.mesh.shared_edges
        assert reference.rigid_residual == actual.rigid_residual
        assert reference.plates.cell_plate.tobytes() == actual.plates.cell_plate.tobytes()
        assert repr(reference.boundaries) == repr(actual.boundaries)
        changed = deepcopy(config)
        changed["plates"]["seed"] += 1
        changed_actual = build_prototype(changed)
        assert changed_actual.mesh is mesh
    assert execution._initial_mesh is None
    changed_reference = build_prototype(changed)
    assert changed_reference.plates.cell_plate.tobytes() == changed_actual.plates.cell_plate.tobytes()
    with CpuExecution():
        assert build_initial_mesh(config) is not mesh


def test_resolution_changes_and_failed_build_do_not_cache_wrong_mesh(config):
    with CpuExecution() as execution:
        first = build_initial_mesh(config)
        changed = deepcopy(config)
        changed["mesh"]["subdivisions"] = 2
        second = build_initial_mesh(changed)
        assert second is not first and second.cell_count == 4 * first.cell_count
        def fail(_):
            raise ValueError("failed")
        with pytest.raises(ValueError):
            execution.initial_mesh(-1, fail)
        assert build_initial_mesh(changed) is second
    assert current_execution() is None


@pytest.mark.parametrize("reuse, expected_prototypes, expected_meshes", [(False, 2, 2), (True, 1, 1)])
def test_craton_setup_and_main_build_only_required_objects(
        config, monkeypatch, reuse, expected_prototypes, expected_meshes):
    import run_long_evolution_v124 as wrapper
    import tectonics.simulation as simulation

    calls = {"prototype": 0, "mesh": 0}
    original_mesh = simulation.build_icosphere
    original_prototype = wrapper.base.build_prototype

    def mesh_builder(subdivisions):
        calls["mesh"] += 1
        return original_mesh(subdivisions)

    def prototype_builder(config):
        calls["prototype"] += 1
        return original_prototype(config)

    monkeypatch.setattr(simulation, "build_icosphere", mesh_builder)
    monkeypatch.setattr(wrapper.base, "build_prototype", prototype_builder)
    monkeypatch.setattr(wrapper.base, "load_config", lambda _: config)
    monkeypatch.setattr(wrapper, "_original_parse_args",
                        lambda: SimpleNamespace(config="unused", output="unused"))
    with CpuExecution(reuse_initial_mesh=reuse):
        wrapper._parse_args_v124()
        prototype = wrapper.base.build_prototype(config)
        assert (wrapper._mesh is prototype.mesh) == reuse
        assert calls == {"prototype": expected_prototypes, "mesh": expected_meshes}
