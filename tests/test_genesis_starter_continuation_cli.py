"""The continuation entry preserves source, age convention and isolated API use."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cli(monkeypatch):
    fake = ModuleType("tectonics.genesis_starter_continuation")
    calls = []

    def run(source, output, **kwargs):
        calls.append((source, output, kwargs))
        return {"status": "completed"}

    fake.run_starter_continuation = run
    monkeypatch.setitem(sys.modules, "tectonics.genesis_starter_continuation", fake)
    spec = importlib.util.spec_from_file_location("_starter_continuation_cli", ROOT / "run_genesis_starter_continuation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, calls, fake


def test_fresh_routes_exact_starter_and_elapsed_duration(cli, tmp_path, capsys):
    module, calls, _ = cli
    source, output = tmp_path / "starter.npz", tmp_path / "continuation"
    assert module.main(["--starter-checkpoint", str(source), "--output", str(output), "--duration-myr", "10", "--step-myr", ".5"]) == 0
    assert calls == [(source, output, {"duration_myr": 10., "step_myr": .5, "resume": False, "mature_config": None})]
    assert "GENESIS_STARTER_CONTINUATION_COMPLETE" in capsys.readouterr().out


def test_resume_keeps_total_elapsed_duration_for_core_to_resolve(cli, tmp_path):
    module, calls, _ = cli
    source, output = tmp_path / "earlier_continuation", tmp_path / "later"
    config = tmp_path / "mature.yaml"
    assert module.main(["--resume", str(source), "--output", str(output), "--duration-myr", "20", "--mature-config", str(config)]) == 0
    assert calls == [(source, output, {"duration_myr": 20., "step_myr": 1., "resume": True, "mature_config": config})]


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_invalid_duration_rejected_without_calling_core(cli, tmp_path, value):
    module, calls, _ = cli
    assert module.main(["--starter-checkpoint", "source.npz", "--output", str(tmp_path / "out"), "--duration-myr", value]) == 1
    assert not calls


def test_core_failure_produces_error_without_success_marker(cli, tmp_path, capsys):
    module, calls, fake = cli

    def fail(*args, **kwargs):
        raise ValueError("source is not a partition")

    fake.run_starter_continuation = fail
    assert module.main(["--starter-checkpoint", "source.npz", "--output", str(tmp_path / "out")]) == 1
    output = capsys.readouterr()
    assert "source is not a partition" in output.err
    assert "GENESIS_STARTER_CONTINUATION_COMPLETE" not in output.out


def test_source_modes_are_exclusive(cli, tmp_path):
    module, calls, _ = cli
    with pytest.raises(SystemExit):
        module.main(["--starter-checkpoint", "source.npz", "--resume", "previous", "--output", str(tmp_path / "out")])
    assert not calls
