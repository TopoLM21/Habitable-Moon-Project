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
