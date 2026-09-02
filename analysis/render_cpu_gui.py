"""Offscreen layout check, without launching the numerical model."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication
from moon_gui.app import MoonWindow, install_application_font


def main() -> None:
    app = QApplication([])
    install_application_font(app)
    window = MoonWindow()
    window.resize(1540, 1120)
    window.show()
    app.processEvents()
    output = ROOT / "results" / "cpu_performance" / "gui_preview.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not window.grab().save(str(output)):
        raise RuntimeError("Could not save GUI layout preview")
    window.close()
    app.processEvents()
    print(output)


if __name__ == "__main__":
    main()
