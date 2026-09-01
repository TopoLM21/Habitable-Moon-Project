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
    window.close()
    app.processEvents()
