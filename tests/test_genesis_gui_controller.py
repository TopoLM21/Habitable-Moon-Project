"""Paired saves remain authoritative through the GUI worker lifecycle."""
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QProcess
import pytest

from moon_gui.app import SimulationController
from moon_gui.backend import RunSpec, read_genesis_continuation
from test_genesis_continuation_backend import write_pair, update_report


@pytest.fixture
def controller_spec(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    source = write_pair(tmp_path / "results" / "source")
    (tmp_path / "run_genesis_starter_continuation.py").write_text("pass\n")
    info = read_genesis_continuation(source)
    spec = RunSpec(tmp_path, info.config, tmp_path / "results" / "next",
        subdivisions=info.subdivisions, end_time_myr=info.time_myr+2., dt_myr=1.,
        checkpoint_interval_myr=1., genesis_continuation=source)
    controller = SimulationController()
    monkeypatch.setattr(controller.process, "start", lambda: None)
    monkeypatch.setattr(controller, "_read_output", lambda: None)
    yield controller, spec
    controller.diagnostics_timer.stop()
    controller.deleteLater()
    app.processEvents()


def test_initial_and_later_segments_resume_complete_pair_and_pause_safely(controller_spec):
    controller, spec = controller_spec
    completed = []
    controller.segment_completed.connect(lambda time, path: completed.append((time, path)))
    controller.start(spec)
    assert controller.resume_checkpoint == spec.genesis_continuation
    argv = controller.process.arguments()
    assert Path(argv[argv.index("--resume")+1]) == spec.genesis_continuation
    assert "run_genesis_starter_continuation.py" in " ".join(argv)
    controller.request_pause()
    first = controller.active_checkpoint
    write_pair(first, duration=11.)
    controller._process_finished(0, QProcess.ExitStatus.NormalExit)
    assert controller.state == "Paused"
    assert controller.resume_checkpoint == first
    assert completed == [(spec.start_time_myr()+1., str(first))]
    controller.resume()
    argv = controller.process.arguments()
    assert Path(argv[argv.index("--resume")+1]) == first
    assert controller.state == "Running"
    assert float(argv[argv.index("--duration-myr")+1]) == 12.
    second = controller.active_checkpoint
    write_pair(second, duration=12.)
    controller._process_finished(0, QProcess.ExitStatus.NormalExit)
    assert controller.state == "Completed"
    assert controller.resume_checkpoint == second
    assert controller.current_time == spec.end_time_myr


@pytest.mark.parametrize("failure", ["missing", "failed_checks", "wrong_age", "process_error", "cancel"])
def test_incomplete_or_failed_segment_never_replaces_last_complete_pair(controller_spec, failure):
    controller, spec = controller_spec
    notices = []
    completions = []
    controller.run_failed.connect(notices.append)
    controller.segment_completed.connect(lambda *args: completions.append(args))
    controller.start(spec)
    if failure != "missing":
        write_pair(controller.active_checkpoint, duration=12. if failure == "wrong_age" else 11.)
    if failure == "failed_checks":
        update_report(controller.active_checkpoint, checks={"clocks_agree": False})
    if failure == "cancel":
        controller.cancel_requested = True
    controller._process_finished(7 if failure == "process_error" else 0, QProcess.ExitStatus.NormalExit)
    assert controller.resume_checkpoint == spec.genesis_continuation
    assert controller.current_time == spec.start_time_myr()
    assert not completions
    assert controller.state == ("Stopped" if failure == "cancel" else "Error")
    assert bool(notices) == (failure != "cancel")


def test_diagnostic_wrapper_preserves_a_cli_returned_failure_code(tmp_path):
    root = Path(__file__).resolve().parents[1]
    runner = tmp_path / "return_failure.py"
    runner.write_text("def main():\n    return 7\n", encoding="utf-8")
    result = subprocess.run([sys.executable, str(root / "run_with_diagnostics.py"),
        "--diagnostics-dir", str(tmp_path / "diagnostics"), str(runner)],
        cwd=root, capture_output=True, text=True, timeout=30)
    assert result.returncode == 7, result.stdout + result.stderr
