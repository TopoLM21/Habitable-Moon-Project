"""CLI time routing, protected outputs and per-step coupled checkpoint writes."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from tectonics.mesh import build_icosphere


def _source(path, kind="genesis-faults-0.1", time=1.4):
    np.savez(path, metadata=np.array(json.dumps({"format": kind, "source_time_myr": 1.4,
                                               "state": {"time_myr": time}})))
    return path


@pytest.fixture
def cli(monkeypatch):
    core = ModuleType("tectonics.genesis_coupled")
    calls, saves = [], []
    mesh = build_icosphere(0)

    class Model:
        source_time_myr = 1.4
        stopped = False
        stuck = False

        @classmethod
        def from_fault_checkpoint(cls, path):
            return cls(), SimpleNamespace(time_myr=1.4, stopped_reason=None), SimpleNamespace(stopped_reason=None), None

        def step(self, state, thermal, orbit, target_myr, max_step_myr):
            calls.append((target_myr, max_step_myr))
            reason = "coupled_geometry_limit" if self.stopped else None
            return SimpleNamespace(time_myr=state.time_myr if self.stuck else target_myr, stopped_reason=reason), thermal, orbit, []

        def diagnostics(self, state, thermal, orbit):
            elapsed = (state.time_myr - self.source_time_myr) * 1e6
            return {"time_myr": state.time_myr, "elapsed_years": elapsed, "surface_temperature_k": 1200. - elapsed / 100,
                    "mantle_temperature_k": 1600., "ocean_fraction": .5, "mean_lid_thickness_km": 8.,
                    "max_opening_m": elapsed / 10, "max_abs_jump_m": elapsed / 100, "stopped_reason": state.stopped_reason}

        def mesh_for(self, state):
            return mesh

        def fields(self, state, thermal):
            return {"damage": np.zeros(mesh.cell_count), "seam_centers_xyz": np.empty((0, 3)),
                    "seam_gap_m": np.empty((0, 2)), "seam_slip_m": np.empty((0, 2))}

    def load(path):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
        return Model(), SimpleNamespace(time_myr=metadata["state"]["time_myr"], stopped_reason=None), SimpleNamespace(stopped_reason=None), None

    def save(path, model, state, thermal, orbit):
        saves.append(state.time_myr)
        _source(path, "genesis-coupled-0.2", state.time_myr)

    core.CoupledModel, core.load_coupled_checkpoint, core.save_coupled_checkpoint = Model, load, save
    monkeypatch.setitem(sys.modules, "tectonics.genesis_coupled", core)
    spec = importlib.util.spec_from_file_location("_coupled_cli_contract", Path(__file__).parents[1] / "run_genesis_coupled.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import visualization.genesis_coupled as plots
    import visualization.genesis_shell as shell_plots

    def picture(mesh, fields, path, history, summary):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test image")

    monkeypatch.setattr(plots, "save_coupled_snapshot", picture)
    monkeypatch.setattr(shell_plots, "save_shell_animation", lambda frames, path: path.write_bytes(b"test animation"))
    return module, Model, calls, saves


def test_cli_advances_one_physical_clock_and_saves_every_step(cli, tmp_path):
    module, _, calls, saves = cli
    path = _source(tmp_path / "fault.npz")
    output = tmp_path / "out"
    assert module.main(["--checkpoint", str(path), "--output", str(output), "--duration-years", "250", "--step-years", "100"]) == 0
    np.testing.assert_allclose(np.array(calls)[:, 0], [1.4001, 1.4002, 1.40025], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(np.array(calls)[:, 1], [.0001] * 3)
    np.testing.assert_allclose(saves, [1.4, 1.4001, 1.4002, 1.40025], rtol=0, atol=1e-15)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["model"] == "genesis-coupled" and not summary["mature_handoff"]
    assert summary["final"]["elapsed_years"] == pytest.approx(250.)
    with (output / "coupled_history.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert float(rows[-1]["surface_temperature_k"]) < float(rows[0]["surface_temperature_k"])
    assert (output / "genesis_coupled.png").is_file() and (output / "genesis_coupled.gif").is_file()


def test_resume_duration_remains_relative_to_original_fault_time(cli, tmp_path):
    module, _, calls, saves = cli
    path = _source(tmp_path / "coupled.npz", "genesis-coupled-0.2", 1.4002)
    output = tmp_path / "out"
    assert module.main(["--resume", str(path), "--output", str(output), "--duration-years", "300", "--no-frames"]) == 0
    assert len(calls) == 1 and calls[0][0] == pytest.approx(1.4003)
    assert saves == [1.4002, pytest.approx(1.4003)]
    assert (output / "genesis_coupled.png").exists()
    assert not (output / "coupled_frames").exists()


@pytest.mark.parametrize("flag", ["--checkpoint", "--resume"])
def test_frozen_contact_rejected_before_output_creation(cli, tmp_path, flag):
    module, _, calls, _ = cli
    path = _source(tmp_path / "contact.npz", "genesis-contact-0.1")
    output = tmp_path / "out"
    assert module.main([flag, str(path), "--output", str(output)]) == 1
    assert not calls and not output.exists()


@pytest.mark.parametrize("option,value", [("--duration-years", "nan"), ("--duration-years", "0"), ("--step-years", "-1"), ("--step-years", "inf")])
def test_invalid_time_controls_do_not_start(cli, tmp_path, option, value):
    module, _, calls, _ = cli
    path = _source(tmp_path / "fault.npz")
    assert module.main(["--checkpoint", str(path), "--output", str(tmp_path / "out"), option, value]) == 1
    assert not calls


def test_existing_results_are_preserved(cli, tmp_path):
    module, _, calls, _ = cli
    path = _source(tmp_path / "fault.npz")
    output = tmp_path / "out"
    output.mkdir()
    (output / "keep.txt").write_text("original")
    assert module.main(["--checkpoint", str(path), "--output", str(output)]) == 1
    assert not calls and (output / "keep.txt").read_text() == "original"


def test_physical_limit_saves_partial_result(cli, tmp_path):
    module, Model, calls, saves = cli
    Model.stopped = True
    path = _source(tmp_path / "fault.npz")
    output = tmp_path / "out"
    assert module.main(["--checkpoint", str(path), "--output", str(output), "--no-frames"]) == 0
    assert len(calls) == 1 and len(saves) == 2
    assert json.loads((output / "summary.json").read_text())["status"] == "coupled_geometry_limit"


def test_unexplained_failure_to_advance_is_error(cli, tmp_path):
    module, Model, _, _ = cli
    Model.stuck = True
    path = _source(tmp_path / "fault.npz")
    output = tmp_path / "out"
    assert module.main(["--checkpoint", str(path), "--output", str(output), "--no-frames"]) == 1
    assert (output / "coupled_checkpoint.npz").is_file()
