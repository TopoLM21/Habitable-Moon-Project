#!/usr/bin/env python3
"""Start the Moon Tectonics desktop application."""

import os
import sys


def main() -> int:
    # Exercise the same BAT/interpreter/window startup without leaving a window
    # or starting a numerical run; intended for setup verification and CI.
    if "--smoke-test" in sys.argv:
        sys.argv.remove("--smoke-test")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv)
        QTimer.singleShot(500, app.quit)
        print(f"GUI startup smoke test: {sys.executable}", flush=True)
    from moon_gui.app import main as gui_main
    return gui_main()


if __name__ == "__main__":
    raise SystemExit(main())
