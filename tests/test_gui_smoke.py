from __future__ import annotations

import os


def test_gui_constructs_offscreen() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from moon_gui.app import MoonWindow

    app = QApplication.instance() or QApplication([])
    window = MoonWindow()
    assert window.windowTitle().startswith("Moon Tectonics")
    assert window.subdivisions.currentText() == "5"
    assert window.controller.state == "Idle"
    assert "ETA" in window.eta_label.text()
    assert window.cpu_mode.currentData() is True
    assert window._make_spec().cpu_workers == 1
    assert window._make_spec().render_workers == 4
    assert window._make_spec().cell_kernels is True
    window.render_workers.setCurrentText("4")
    assert window._make_spec().render_workers == 4
    window.cpu_mode.setCurrentIndex(0)
    assert not window.cpu_workers.isEnabled()
    assert not window.render_workers.isEnabled()
    assert window._make_spec().render_workers == 1
    assert window._make_spec().cell_kernels is False
    assert window._make_spec().cpu_optimized is False
    window.close()
    app.processEvents()


def test_eta_label_updates_without_starting_a_model(monkeypatch) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    import moon_gui.app as gui
    from moon_gui.timing import RunTiming

    app = QApplication.instance() or QApplication([])
    window = gui.MoonWindow()
    timing = RunTiming(400.0, 500.0)
    timing.start_segment(420.0, 0.0)
    timing.finish_segment(30.0)
    timing.start_segment(440.0, 30.0)
    timing.finish_segment(60.0)
    timing.start_segment(460.0, 60.0)
    window.controller.timing = timing
    monkeypatch.setattr(gui, "monotonic", lambda: 75.0)
    window.controller._set_state("Running")
    assert "ETA расчёта: ≈ 1 мин 15 с" in window.eta_label.text()
    timing.finish_segment(90.0)
    monkeypatch.setattr(gui, "monotonic", lambda: 9000.0)
    window.controller._set_state("Paused")
    assert "Прошло без пауз: 1 мин 30 с" in window.eta_label.text()
    assert "После возобновления: ≈ 1 мин 0 с" in window.eta_label.text()
    window.controller._set_state("Stopped")
    assert "ETA недоступно" in window.eta_label.text()
    window.close()
    app.processEvents()


def test_external_checkpoint_gets_isolated_output_and_saved_config(tmp_path, monkeypatch):
    import json
    import numpy as np
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    import moon_gui.app as gui

    project = tmp_path / "experiment"
    project.mkdir()
    original = tmp_path / "stable_run"
    checkpoint = original / "gui_checkpoint_000700_Myr"
    checkpoint.mkdir(parents=True)
    (checkpoint / "meta.json").write_text(json.dumps({"format": "moon_tectonics_checkpoint", "time_myr": 700.0}))
    np.savez(checkpoint / "state.npz", state_cell_plate=np.zeros(1280, dtype=np.int32))
    saved_config = original / "gui_runtime_config.yaml"
    saved_config.write_text("mesh: {subdivisions: 3}\n")
    monkeypatch.setattr(gui, "PROJECT_ROOT", project)
    monkeypatch.setattr(gui.QFileDialog, "getExistingDirectory", lambda *args: str(checkpoint))
    app = QApplication.instance() or QApplication([])
    window = gui.MoonWindow()
    window._browse_checkpoint()
    assert window.resume_field.path() == checkpoint
    assert window.output_field.path().is_relative_to(project / "results")
    assert window.config_field.path() == saved_config
    assert not window.output_field.path().exists()
    assert saved_config.read_text() == "mesh: {subdivisions: 3}\n"
    window.close()
    app.processEvents()


def test_old_stop_timer_cannot_kill_a_new_segment():
    from moon_gui.app import SimulationController, QProcess
    class Process:
        killed = False
        def state(self):
            return QProcess.ProcessState.Running
        def processId(self):
            return 202
        def kill(self):
            self.killed = True
    class Controller:
        process = Process()
    controller = Controller()
    SimulationController._kill_if_running(controller, 101)
    assert not controller.process.killed
    SimulationController._kill_if_running(controller, 202)
    assert controller.process.killed
