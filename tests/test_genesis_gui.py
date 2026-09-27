"""Exercise the independent Qt process without opening a desktop window."""
import json
import os
from pathlib import Path
from time import monotonic

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QProcess, QTimer
from PySide6.QtWidgets import QApplication
import pytest

from moon_gui.genesis_dialog import GenesisDialog, ROOT


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def pump_until(app, predicate, timeout=20):
    deadline = monotonic() + timeout
    while not predicate() and monotonic() < deadline:
        loop = QEventLoop()
        QTimer.singleShot(10, loop.quit)
        loop.exec()
    assert predicate(), "Qt process did not settle"


def test_dialog_keeps_full_precision_config_until_parameters_are_changed(app, tmp_path):
    dialog = GenesisDialog()
    args = dialog.arguments(tmp_path / "run")
    assert "--stellar-flux-w-m2" not in args
    assert "--water-volume-km3" not in args
    assert "--initial-temperature-k" not in args
    dialog.stellar_flux.setValue(2000)
    dialog.water_volume.setValue(0.5)
    dialog.temperature.setValue(2400)
    args = dialog.arguments(tmp_path / "run")
    assert float(args[args.index("--stellar-flux-w-m2") + 1]) == 2000
    assert float(args[args.index("--water-volume-km3") + 1]) == 5e8
    assert float(args[args.index("--initial-temperature-k") + 1]) == 2400
    dialog.close()


def test_real_process_generates_preview_without_blocking_qt(app, tmp_path, monkeypatch):
    dialog = GenesisDialog()
    dialog.duration.setValue(0.1)
    original_arguments = dialog.arguments
    monkeypatch.setattr(dialog, "arguments", lambda _: original_arguments(tmp_path / "result"))
    dialog.start()
    dialog.output_path = tmp_path / "result"
    assert not dialog.start_button.isEnabled()
    assert not dialog.stellar_flux.isEnabled()
    ticks = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(10)
    pump_until(app, lambda: not dialog.is_running() and dialog.start_button.isEnabled())
    timer.stop()
    assert ticks
    assert not dialog._pixmap.isNull()
    assert dialog.folder_button.isEnabled()
    assert "не началась" in dialog.status.text()
    assert json.loads((dialog.output_path / "summary.json").read_text())["status"] == "completed"
    dialog.close()


def test_closing_dialog_stops_child_and_old_stop_timer_does_not_kill_new_generation(app, tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "genesis_moon.yaml").write_bytes((ROOT / "configs" / "genesis_moon.yaml").read_bytes())
    # An inert long-lived worker tests lifecycle without an expensive simulation.
    (tmp_path / "run_genesis.py").write_text("import time\nprint('ready', flush=True)\ntime.sleep(20)\n")
    dialog = GenesisDialog(project_root=tmp_path)
    dialog.show()
    dialog.start()
    pump_until(app, lambda: "ready" in dialog.log.toPlainText())
    old_generation = dialog._generation - 1
    dialog._kill_if_current(old_generation)
    assert dialog.is_running()
    dialog.close()
    pump_until(app, lambda: not dialog.is_running() and not dialog.isVisible())
    assert dialog._stopping


def test_modal_dialog_close_exits_its_event_loop(app):
    dialog = GenesisDialog()
    QTimer.singleShot(30, dialog.reject)
    dialog.exec()
    assert not dialog.isVisible()


def test_genesis_button_tracks_mature_process_state(app):
    from moon_gui.app import MoonWindow
    window = MoonWindow()
    assert window.genesis_button.isEnabled()
    for state in ("Preparing", "Running", "Pausing", "Paused", "Stopping"):
        window.controller._set_state(state)
        assert not window.genesis_button.isEnabled()
    window.controller._set_state("Stopped")
    assert window.genesis_button.isEnabled()
    window.close()
