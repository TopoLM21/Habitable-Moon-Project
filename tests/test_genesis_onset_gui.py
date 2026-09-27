"""Onset UI routing and observable diagnostics, preserving earlier experiments."""
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image
from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QApplication
import pytest

from moon_gui.genesis_dialog import GenesisDialog, ROOT
from tectonics.mesh import build_icosphere
from visualization.genesis_onset import save_onset_snapshot


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def dialog(app, tmp_path):
    config = tmp_path / "configs"
    config.mkdir()
    (config / "genesis_moon.yaml").write_bytes((ROOT / "configs/genesis_moon.yaml").read_bytes())
    (tmp_path / "run_genesis_onset.py").touch()
    (tmp_path / "run_genesis_shell.py").touch()
    widget = GenesisDialog(project_root=tmp_path)
    yield widget
    widget.close()


def test_onset_defaults_and_independent_effect_switches(dialog, tmp_path):
    assert dialog.mode.currentData() == "onset"
    args = dialog.arguments(tmp_path / "run")
    assert Path(args[1]).name == "run_genesis_onset.py"
    assert "--no-tides" not in args and "--no-water-weakening" not in args
    assert float(args[args.index("--regularization-km") + 1]) == 800
    assert float(args[args.index("--shell-step-myr") + 1]) == 0.002
    dialog.tides.setChecked(False)
    dialog.regularization.setValue(400)
    args = dialog.arguments(tmp_path / "run")
    assert "--no-tides" in args and "--no-water-weakening" not in args
    assert float(args[args.index("--regularization-km") + 1]) == 400
    dialog.water_weakening.setChecked(False)
    assert "--no-water-weakening" in dialog.arguments(tmp_path / "run")


def test_control_disables_tidal_load_and_restores_preference(dialog, tmp_path):
    dialog.intact_control.setChecked(True)
    args = dialog.arguments(tmp_path / "control")
    assert args[args.index("--control") + 1] == "intact"
    assert "--no-tides" in args
    assert not dialog.tides.isEnabled()
    assert not dialog.convective_traction.isEnabled()
    dialog.intact_control.setChecked(False)
    assert dialog.tides.isEnabled() and dialog.tides.isChecked()
    assert "--no-tides" not in dialog.arguments(tmp_path / "run")


@pytest.mark.parametrize("mode,runner", [("shell", "run_genesis_shell.py"), ("thermal", "run_genesis.py")])
def test_older_modes_do_not_receive_onset_options(dialog, tmp_path, mode, runner):
    dialog.tides.setChecked(False)
    dialog.water_weakening.setChecked(False)
    dialog.mode.setCurrentIndex(dialog.mode.findData(mode))
    args = dialog.arguments(tmp_path / "run")
    assert Path(args[1]).name == runner
    assert not {"--no-tides", "--no-water-weakening", "--regularization-km"}.intersection(args)
    assert all(not widget.isEnabled() for widget in dialog._onset_controls)
    dialog.mode.setCurrentIndex(dialog.mode.findData("onset"))
    assert all(widget.isEnabled() for widget in dialog._onset_controls)
    dialog._set_busy(True)
    assert all(not widget.isEnabled() for widget in dialog._onset_controls)


def test_onset_results_are_selected_first_and_motion_is_reported(dialog, tmp_path):
    summary = {"status": "completed", "events_myr": {"ocean_start": 1.045},
               "shell": {"damaged_area_fraction": 0.25, "intact_region_count": 4},
               "onset": {"mean_speed_cm_yr": 0.123}}
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    for name in ("genesis_onset.png", "genesis_onset.gif", "genesis_shell.png", "genesis_history.png"):
        Image.new("RGB", (80, 40), "navy").save(tmp_path / name)
    dialog.output_path = tmp_path
    dialog._finished(0, QProcess.ExitStatus.NormalExit)
    assert "0.123 см/год" in dialog.status.text()
    assert "Связных областей: 4" in dialog.status.text()
    assert "плит: 4" not in dialog.status.text()
    assert dialog.result_view.count() == 4
    assert dialog.result_view.currentData().endswith("genesis_onset.png")
    assert not dialog._pixmap.isNull()
    dialog.result_view.setCurrentIndex(1)
    assert dialog._movie is not None and dialog._movie.isValid()


def test_cancelled_onset_names_joint_checkpoint(dialog, tmp_path):
    dialog._stopping = True
    dialog._run_mode = "onset"
    dialog.output_path = tmp_path
    dialog._finished(0, QProcess.ExitStatus.NormalExit)
    assert "run_genesis_onset.py --resume onset_checkpoint.npz" in dialog.status.text()


def _fields(mesh):
    fraction = np.linspace(0, 1, mesh.cell_count)
    return {"lid_thickness_km": fraction * 15, "damage": fraction,
            "water_access": fraction, "speed_cm_yr": fraction * 4,
            "displacement_km": fraction * 250, "tidal_stress_mpa": fraction * 0.04,
            "failed_edges": np.arange(len(mesh.shared_edges)) % 7 == 0}


def test_onset_maps_render_with_limits_and_stop_context(tmp_path):
    mesh = build_icosphere(1)
    snapshot = save_onset_snapshot(mesh, _fields(mesh), tmp_path / "onset.png", 1.2,
                                   {"status": "shell_small_strain_limit", "orbit": {"eccentricity": 0.05},
                                    "plot_limits": {"speed_cm_yr": [0, 2]}})
    with Image.open(snapshot) as image:
        assert image.width >= 1300 and image.height >= 1100
        assert image.convert("RGB").getextrema()[0][0] < 100


@pytest.mark.parametrize("fault", ["nonfinite", "wrong_shape", "reversed_scale"])
def test_onset_map_rejects_misleading_values_or_scale(tmp_path, fault):
    mesh = build_icosphere(1)
    fields, summary = _fields(mesh), {}
    if fault == "nonfinite":
        fields["tidal_stress_mpa"][0] = np.nan
    elif fault == "wrong_shape":
        fields["displacement_km"] = np.zeros(mesh.vertex_count)
    else:
        summary["plot_limits"] = {"speed_cm_yr": [5, 0]}
    path = tmp_path / "invalid.png"
    with pytest.raises(ValueError):
        save_onset_snapshot(mesh, fields, path, 1, summary)
    assert not path.exists()
