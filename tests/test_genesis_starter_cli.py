"""Runner time routing, no forced partition, restart, and protected outputs."""
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cli(monkeypatch):
    fake = ModuleType("tectonics.genesis_starter")
    calls, saves, instances = [], [], []

    @dataclass(frozen=True)
    class Parameters:
        seed: int = 20260927
        cooling_contrast_fraction: float = .1
        tidal_mechanics: bool = True
        water_weakening: bool = True

    class Model:
        stop_at = None
        stuck = False

        def __init__(self, mesh, thermal, tides, shell, parameters):
            self.mesh, self.thermal, self.tides, self.shell, self.parameters = mesh, thermal, tides, shell, parameters
            instances.append(self)

        def _state(self, time, reason=None):
            return SimpleNamespace(time_myr=time, stopped_reason=reason, events=[],
                                   damage=np.zeros(self.mesh.cell_count), yield_ratio=np.zeros(self.mesh.cell_count),
                                   system=SimpleNamespace(cell_plate=np.zeros(self.mesh.cell_count, dtype=int)))

        def initial_state(self):
            return self._state(0.)

        def advance(self, old, target):
            calls.append(target)
            if self.stuck:
                return old
            if self.stop_at is not None and target >= self.stop_at:
                return self._state(self.stop_at, "first_partition")
            return self._state(target)

        def diagnose(self, state):
            return {"time_myr": state.time_myr, "domain_count": 2 if state.stopped_reason else 1,
                    "damaged_area_fraction": 0., "max_yield_ratio": 0., "surface_temperature_k": 2200.,
                    "mantle_temperature_k": 2300.}

        def save_state(self, path, state):
            saves.append(state.time_myr)
            path.write_text(json.dumps({"time": state.time_myr, "stopped": state.stopped_reason}))

        def load_state(self, path):
            raw = json.loads(path.read_text())
            return self._state(raw["time"], raw["stopped"])

    fake.StarterParameters, fake.StarterModel = Parameters, Model
    monkeypatch.setitem(sys.modules, "tectonics.genesis_starter", fake)
    spec = importlib.util.spec_from_file_location("_starter_cli_contract", ROOT / "run_genesis_starter.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import visualization.genesis_starter as plots
    monkeypatch.setattr(plots, "save_starter_snapshot", lambda mesh, state, path, history, summary: path.write_bytes(b"picture"))
    return module, Model, calls, saves, instances


def test_terminal_age_and_fractional_step_are_saved_without_inventing_partition(cli, tmp_path, capsys):
    module, _, calls, saves, _ = cli
    out = tmp_path / "run"
    assert module.main(["--output", str(out), "--duration-myr", "2.3", "--step-myr", "1", "--subdivisions", "1"]) == 0
    assert calls == [1., 2., 2.3]
    assert saves == [0., 1., 2., 2.3]
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "completed"
    assert not summary["candidate_partition"] and not summary["mature_handoff"]
    assert summary["final"]["domain_count"] == 1
    assert "GENESIS_STARTER_COMPLETE" in capsys.readouterr().out


def test_first_partition_stops_at_actual_event_age(cli, tmp_path):
    module, model, calls, saves, _ = cli
    model.stop_at = 1.23
    out = tmp_path / "run"
    assert module.main(["--output", str(out), "--duration-myr", "20", "--subdivisions", "1"]) == 0
    assert calls == [1., 2.]
    assert saves[-1] == 1.23
    summary = json.loads((out / "summary.json").read_text())
    assert summary["candidate_partition"] and not summary["mature_handoff"]


def test_intact_control_retains_orbit_heat_but_disables_mechanical_drivers(cli, tmp_path):
    module, _, _, _, instances = cli
    assert module.main(["--output", str(tmp_path / "run"), "--duration-myr", ".01", "--subdivisions", "1", "--control", "intact"]) == 0
    model = instances[-1]
    assert model.tides.enabled
    assert model.shell.convective_traction_pa == 0
    assert model.parameters.cooling_contrast_fraction == 0
    assert not model.parameters.tidal_mechanics


def test_resume_retains_physics_and_uses_absolute_age(cli, tmp_path):
    module, _, calls, saves, _ = cli
    first, second = tmp_path / "first", tmp_path / "second"
    assert module.main(["--output", str(first), "--duration-myr", "1.5", "--subdivisions", "1", "--seed", "42"]) == 0
    calls.clear()
    saves.clear()
    assert module.main(["--resume", str(first / "starter_checkpoint.npz"), "--output", str(second), "--duration-myr", "2.75", "--step-myr", ".5"]) == 0
    assert calls == [2., 2.5, 2.75]
    assert saves == [1.5, 2., 2.5, 2.75]
    metadata = json.loads((second / "parameters.json").read_text())
    assert metadata["starter"]["seed"] == 42 and metadata["shell"]["subdivisions"] == 1


def test_resume_rejects_physics_overrides_and_stopped_state(cli, tmp_path):
    module, model, _, _, _ = cli
    source = tmp_path / "first"
    model.stop_at = .2
    assert module.main(["--output", str(source), "--duration-myr", "1", "--subdivisions", "1"]) == 0
    checkpoint = source / "starter_checkpoint.npz"
    assert module.main(["--resume", str(checkpoint), "--output", str(tmp_path / "override"), "--seed", "4"]) == 1
    assert module.main(["--resume", str(checkpoint), "--output", str(tmp_path / "stopped")]) == 1
    assert not (tmp_path / "override").exists() and not (tmp_path / "stopped").exists()


def test_existing_output_and_invalid_time_controls_are_rejected(cli, tmp_path):
    module, _, calls, _, _ = cli
    out = tmp_path / "run"
    out.mkdir()
    marker = out / "keep.txt"
    marker.write_text("keep")
    assert module.main(["--output", str(out)]) == 1
    assert marker.read_text() == "keep" and not calls
    for invalid in ("0", "-1", "nan", "inf"):
        assert module.main(["--output", str(tmp_path / "new"), "--duration-myr", invalid]) == 1
    assert not (tmp_path / "new").exists()


def test_nonadvancing_backend_is_reported_instead_of_looping(cli, tmp_path):
    module, model, calls, _, _ = cli
    model.stuck = True
    assert module.main(["--output", str(tmp_path / "run"), "--duration-myr", "1", "--subdivisions", "1"]) == 1
    assert calls == [1.]


def test_opt_in_continuation_skips_missing_partition(cli, tmp_path, capsys):
    module, _, _, _, _ = cli
    out = tmp_path / "run"
    assert module.main(["--output", str(out), "--duration-myr", "1", "--subdivisions", "1", "--continue-myr", "10"]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["continuation"]["status"] == "skipped_no_partition"
    assert not (out / "continuation").exists()
    assert "GENESIS_STARTER_CONTINUATION_SKIPPED" in capsys.readouterr().out


def test_opt_in_continuation_reuses_same_worker_entry_and_preserves_source(cli, tmp_path, monkeypatch):
    module, model, _, _, _ = cli
    model.stop_at = .2
    import run_genesis_starter_continuation as continuation
    calls = []

    def run(source, output, duration, step):
        calls.append((source, output, duration, step))
        assert source.is_file()
        return {"final_time_myr": 10.2, "physical_handoff_certified": False}

    monkeypatch.setattr(continuation, "run_continuation", run)
    out = tmp_path / "run"
    assert module.main(["--output", str(out), "--duration-myr", "1", "--subdivisions", "1", "--continue-myr", "10", "--continuation-step-myr", ".5"]) == 0
    assert calls == [(out / "starter_checkpoint.npz", out / "continuation", 10., .5)]
    summary = json.loads((out / "summary.json").read_text())
    assert summary["continuation"]["status"] == "completed"
    assert not summary["mature_handoff"]
    assert json.loads((out / "starter_checkpoint.npz").read_text())["time"] == .2


def test_failed_continuation_retains_successful_starter_checkpoint(cli, tmp_path, monkeypatch):
    module, model, _, _, _ = cli
    model.stop_at = .2
    import run_genesis_starter_continuation as continuation

    def fail(*args):
        raise RuntimeError("mature runner failed")

    monkeypatch.setattr(continuation, "run_continuation", fail)
    out = tmp_path / "run"
    assert module.main(["--output", str(out), "--duration-myr", "1", "--subdivisions", "1", "--continue-myr", "10"]) == 1
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "first_partition"
    assert summary["continuation"]["status"] == "failed"
    assert summary["continuation"]["error"] == "mature runner failed"
    assert (out / "starter_checkpoint.npz").is_file()


def test_plot_distinguishes_single_shell_from_verified_plates(tmp_path):
    from tectonics.mesh import build_icosphere
    from visualization.genesis_starter import save_starter_snapshot
    mesh = build_icosphere(1)
    state = SimpleNamespace(time_myr=1., damage=np.zeros(mesh.cell_count),
                            system=SimpleNamespace(cell_plate=np.zeros(mesh.cell_count, dtype=int)))
    rows = [{"time_myr": t, "surface_temperature_k": 2300. - 200 * t,
             "mantle_temperature_k": 2300., "max_yield_ratio": 0., "damaged_area_fraction": 0.} for t in (0., 1.)]
    path = save_starter_snapshot(mesh, state, tmp_path / "starter.png", rows, {"candidate_partition": False})
    assert path.stat().st_size > 10_000
