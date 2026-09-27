"""Nonblocking Qt lifecycle and explicit starter status, without GUI interaction."""
import json
import os
from pathlib import Path
from time import monotonic
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QByteArray, QProcess, QTimer
from PySide6.QtWidgets import QApplication
import pytest

from moon_gui.genesis_starter_dialog import GenesisStarterDialog, ROOT


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def pump_until(app, predicate, timeout=15):
    deadline = monotonic() + timeout
    while not predicate() and monotonic() < deadline:
        app.processEvents()
    assert predicate()


def test_arguments_expose_coarse_controls_without_claiming_handoff(app, tmp_path):
    dialog = GenesisStarterDialog()
    args = dialog.arguments(tmp_path)
    assert args[args.index("--duration-myr") + 1] == "20.0"
    assert args[args.index("--step-myr") + 1] == "1.0"
    assert args[args.index("--subdivisions") + 1] == "4"
    assert "--continue-myr" not in args
    assert not dialog.continuation_duration.isEnabled()
    dialog.continue_mature.setChecked(True)
    assert dialog.continuation_duration.isEnabled() and dialog.continuation_step.isEnabled()
    args = dialog.arguments(tmp_path)
    assert args[args.index("--continue-myr") + 1] == "10.0"
    assert args[args.index("--continuation-step-myr") + 1] == "1.0"
    dialog.intact_control.setChecked(True)
    args = dialog.arguments(tmp_path)
    assert "--control" in args and "--no-tides" not in args
    assert not dialog.convective_traction.isEnabled()
    dialog.tides.setChecked(False)
    dialog.water_weakening.setChecked(False)
    args = dialog.arguments(tmp_path)
    assert "--no-tides" in args and "--no-water-weakening" in args
    dialog.close()


@pytest.mark.parametrize("candidate", [False, True])
def test_result_reports_candidate_or_no_partition_without_mature_claim(app, tmp_path, candidate):
    dialog = GenesisStarterDialog()
    dialog.output_path = tmp_path
    summary = {"candidate_partition": candidate, "status": "first_partition" if candidate else "completed",
               "final": {"time_myr": 3., "domain_count": 2 if candidate else 1, "damaged_area_fraction": .12}}
    adopted = []
    dialog.continuation_ready.connect(adopted.append)
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    dialog._finished(0, QProcess.ExitStatus.NormalExit)
    text = dialog.status.text()
    assert ("кандидат" in text) if candidate else ("без разделения" in text)
    assert "успешно сформированы" not in text
    assert not adopted and not dialog.continue_button.isEnabled()
    dialog.close()


def test_close_stops_worker_and_stale_kill_callback_is_harmless(app, tmp_path):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "genesis_moon.yaml").write_bytes((ROOT / "configs" / "genesis_moon.yaml").read_bytes())
    (tmp_path / "run_genesis_starter.py").write_text("import time\nprint('ready', flush=True)\ntime.sleep(30)\n")
    dialog = GenesisStarterDialog(project_root=tmp_path)
    dialog.show()
    dialog.start()
    pump_until(app, lambda: "ready" in dialog.log.toPlainText())
    assert not dialog.start_button.isEnabled() and not dialog.duration.isEnabled()
    dialog._kill_if_current(dialog._generation - 1)
    assert dialog.is_running()
    dialog.close()
    pump_until(app, lambda: not dialog.is_running() and not dialog.isVisible())
    assert dialog._stopping


def test_completed_continuation_status_and_preferred_preview(app, tmp_path, monkeypatch):
    from PySide6.QtGui import QImage
    from moon_gui import backend
    dialog = GenesisStarterDialog()
    dialog.output_path = tmp_path
    continuation = tmp_path / "continuation"
    continuation.mkdir()
    verified = []
    def read_continuation(path, *, verify_integrity):
        verified.append((path, verify_integrity))
        return SimpleNamespace(root=path, time_myr=11., origin_time_myr=1., plate_count=4)
    monkeypatch.setattr(backend, "read_genesis_continuation", read_continuation)
    (continuation / "continuation.json").write_text(json.dumps({"final_plate_count": 4}))
    adopted = []
    dialog.continuation_ready.connect(adopted.append)
    picture = QImage(16, 16, QImage.Format.Format_RGB32)
    picture.fill(0x00AA88)
    picture.save(str(continuation / "continuation.png"))
    summary = {"candidate_partition": True, "status": "first_partition",
               "final": {"time_myr": 1., "domain_count": 2, "damaged_area_fraction": .1},
               "continuation": {"status": "completed", "requested_duration_myr": 10.}}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    dialog._finished(0, QProcess.ExitStatus.NormalExit)
    assert "Продолжение" in dialog.status.text() or "продолжение" in dialog.status.text()
    assert "10 млн лет после разделения" in dialog.status.text()
    assert "11.0000 млн лет; областей: 4" in dialog.status.text()
    assert "1.0000 млн лет; областей: 2" not in dialog.status.text()
    assert verified == [(continuation, True)]
    assert adopted == [str(continuation)]
    assert dialog.continue_button.isEnabled()
    assert dialog._result_image() == continuation / "continuation.png"
    assert not dialog._pixmap.isNull()
    # A duplicate finish notification or leaving the dialog never restarts or
    # adopts the same completed result for a second time.
    dialog._finished(0, QProcess.ExitStatus.NormalExit)
    dialog.continue_button.click()
    assert adopted == [str(continuation)]
    assert dialog.result() == dialog.DialogCode.Accepted
    dialog.close()


@pytest.mark.parametrize("exit_code,exit_status,stopping", [
    (1, QProcess.ExitStatus.NormalExit, False),
    (0, QProcess.ExitStatus.CrashExit, False),
    (0, QProcess.ExitStatus.NormalExit, True),
])
def test_failed_or_cancelled_continuation_is_never_adopted(app, tmp_path, exit_code, exit_status, stopping):
    dialog = GenesisStarterDialog()
    dialog.output_path = tmp_path
    dialog._stopping = stopping
    (tmp_path / "summary.json").write_text(json.dumps({
        "candidate_partition": True, "status": "first_partition",
        "final": {"time_myr": 1., "domain_count": 2, "damaged_area_fraction": .1},
        "continuation": {"status": "completed", "requested_duration_myr": 10.},
    }))
    adopted = []
    dialog.continuation_ready.connect(adopted.append)
    dialog._finished(exit_code, exit_status)
    assert adopted == []
    assert dialog.continuation_path is None
    assert not dialog.continue_button.isEnabled()
    dialog.close()


def test_incomplete_continuation_is_not_adopted_even_with_completed_summary(app, tmp_path):
    dialog = GenesisStarterDialog()
    dialog.output_path = tmp_path
    (tmp_path / "summary.json").write_text(json.dumps({
        "candidate_partition": True, "status": "first_partition",
        "final": {"time_myr": 1., "domain_count": 2, "damaged_area_fraction": .1},
        "continuation": {"status": "completed", "requested_duration_myr": 10.},
    }))
    adopted = []
    dialog.continuation_ready.connect(adopted.append)
    dialog._finished(0, QProcess.ExitStatus.NormalExit)
    assert adopted == []
    assert dialog.continuation_path is None
    assert not dialog.continue_button.isEnabled()
    assert "Не удалось прочитать результат" in dialog.status.text()
    dialog.close()


def test_continuation_start_status_handles_split_process_output(app, monkeypatch):
    dialog = GenesisStarterDialog()
    chunks = iter([b"GENESIS_STARTER_CONTINU", b"ATION_START\n"])
    monkeypatch.setattr(dialog.process, "readAllStandardOutput", lambda: QByteArray(next(chunks)))
    dialog._read_output()
    assert not dialog._continuation_started
    dialog._read_output()
    assert dialog._continuation_started
    assert "заданное время продолжения" in dialog.status.text()
    dialog.close()


def test_new_starter_run_clears_previous_continuation_navigation(app, tmp_path, monkeypatch):
    dialog = GenesisStarterDialog()
    dialog.continuation_path = tmp_path
    dialog._published_continuation = tmp_path
    dialog.continue_button.setEnabled(True)
    monkeypatch.setattr(dialog.process, "start", lambda *_: None)
    dialog.start()
    assert dialog.continuation_path is None
    assert dialog._published_continuation is None
    assert not dialog.continue_button.isEnabled()
    assert not dialog.subdivisions.isEnabled()
    assert not dialog.duration.isEnabled()
    dialog._set_busy(False)
    dialog.close()


def test_escape_exits_modal_dialog(app):
    dialog = GenesisStarterDialog()
    QTimer.singleShot(20, dialog.reject)
    dialog.exec()
    assert not dialog.isVisible()


def test_real_starter_child_writes_preview_while_qt_remains_responsive(app, tmp_path, monkeypatch):
    dialog = GenesisStarterDialog()
    dialog.duration.setValue(2.)
    dialog.subdivisions.setCurrentIndex(dialog.subdivisions.findData(2))
    original_arguments = dialog.arguments
    output = tmp_path / "real_run"
    monkeypatch.setattr(dialog, "arguments", lambda _: original_arguments(output))
    ticks = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(10)
    try:
        dialog.start()
        dialog.output_path = output
        pump_until(app, lambda: not dialog.is_running() and dialog.start_button.isEnabled(), timeout=30)
        assert ticks
        assert not dialog._pixmap.isNull()
        assert dialog.open_result_button.isEnabled()
        assert "без разделения" in dialog.status.text()
        summary = json.loads((output / "summary.json").read_text())
        assert summary["status"] == "completed" and not summary["mature_handoff"]
    finally:
        timer.stop()
        dialog.close()


def test_real_starter_and_mature_continuation_share_one_responsive_gui_worker(app, tmp_path, monkeypatch):
    dialog = GenesisStarterDialog()
    dialog.duration.setValue(2.)
    dialog.subdivisions.setCurrentIndex(dialog.subdivisions.findData(2))
    dialog.convective_traction.setValue(.05)
    dialog.continue_mature.setChecked(True)
    dialog.continuation_duration.setValue(1.)
    dialog.continuation_step.setValue(1.)
    output = Path(os.environ.get("GENESIS_STARTER_E2E_OUTPUT", str(tmp_path / "full_entry")))
    original_arguments = dialog.arguments
    monkeypatch.setattr(dialog, "arguments", lambda _: original_arguments(output))
    ticks = []
    adopted = []
    dialog.continuation_ready.connect(adopted.append)
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(10)
    try:
        dialog.start()
        dialog.output_path = output
        assert not dialog.continue_mature.isEnabled()
        pump_until(app, lambda: not dialog.is_running() and dialog.start_button.isEnabled(), timeout=60)
        assert ticks
        starter = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        continuation = json.loads((output / "continuation" / "continuation.json").read_text(encoding="utf-8"))
        assert starter["status"] == "first_partition"
        assert starter["continuation"]["status"] == "completed"
        assert continuation["mature_engine_executed"]
        assert continuation["duration_myr"] == 1.
        assert continuation["final_time_myr"] == pytest.approx(starter["final"]["time_myr"] + 1., abs=1e-12)
        assert all(continuation["checks"].values())
        assert not continuation["physical_handoff_certified"]
        assert dialog._result_image() == output / "continuation" / "continuation.png"
        assert not dialog._pixmap.isNull()
        assert "1 млн лет после разделения" in dialog.status.text()
        assert f"Текущий возраст: {continuation['final_time_myr']:.4f}" in dialog.status.text()
        assert adopted == [str((output / "continuation").resolve())]
        assert dialog.continue_button.isEnabled()
        assert "GENESIS_STARTER_CONTINUATION_COMPLETE" in dialog.log.toPlainText()
        if "GENESIS_STARTER_E2E_OUTPUT" in os.environ:
            (output / "gui_validation.json").write_text(json.dumps(
                {"qt_timer_ticks_during_run": len(ticks), "preview_loaded": True,
                 "starter_and_mature_completed": True, "status_text": dialog.status.text(),
                 "preview": str(dialog._result_image())}, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        timer.stop()
        dialog.close()
        if dialog.is_running():
            pump_until(app, lambda: not dialog.is_running())


def test_starter_button_tracks_mature_run_and_pause(app):
    from moon_gui.app import MoonWindow
    window = MoonWindow()
    assert window.genesis_starter_button.isEnabled()
    for state in ("Preparing", "Running", "Pausing", "Paused", "Stopping"):
        window.controller._set_state(state)
        assert not window.genesis_starter_button.isEnabled()
    window.controller._set_state("Stopped")
    assert window.genesis_starter_button.isEnabled()
    window.close()
