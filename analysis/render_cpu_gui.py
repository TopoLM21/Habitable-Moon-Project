"""Offscreen layout check, without launching the numerical model."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication
from moon_gui.app import MoonWindow, install_application_font


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', action='store_true', help='Preview the GPU selector and native CPU options')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = (args.output or ROOT / 'results' / 'cpu_performance' / 'gui_preview.png').resolve()
    if not output.is_relative_to((ROOT / 'results').resolve()):
        parser.error('Preview output must stay under results')
    if args.output and output.exists():
        parser.error('Explicit preview output must be new')
    app = QApplication([])
    install_application_font(app)
    window = MoonWindow()
    if args.gpu:
        index = window.cpu_mode.findData('gpu_surface')
        if index < 0:
            raise RuntimeError('GPU mode is missing from the GUI')
        window.cpu_mode.setCurrentIndex(index)
        window.assignment_columns.setChecked(True)
        window.boundary_forces.setChecked(True)
    window.resize(1540, 1120)
    window.show()
    app.processEvents()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not window.grab().save(str(output)):
        raise RuntimeError("Could not save GUI layout preview")
    window.close()
    app.processEvents()
    print(output)


if __name__ == "__main__":
    main()
