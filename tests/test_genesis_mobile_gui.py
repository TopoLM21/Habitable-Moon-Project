"""Moving-shell UI routing, independent checkpoints, and current-geometry maps."""
from dataclasses import replace
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
    for runner in ("run_genesis_mobile.py", "run_genesis_onset.py", "run_genesis_shell.py"):
        (tmp_path / runner).touch()
    widget = GenesisDialog(project_root=tmp_path)
    yield widget
    widget.close()


def test_mobile_mode_defaults_and_retains_effect_controls(dialog, tmp_path):
    assert dialog.mode.currentData() == "mobile"
    assert dialog.mode.count() == 4
    args = dialog.arguments(tmp_path / "run")
    assert Path(args[1]).name == "run_genesis_mobile.py"
    assert float(args[args.index("--shell-step-myr") + 1]) == .002
    assert float(args[args.index("--regularization-km") + 1]) == 800
    assert not {"--no-tides", "--no-water-weakening"}.intersection(args)
    dialog.tides.setChecked(False)
    dialog.water_weakening.setChecked(False)
    dialog.convective_traction.setValue(.05)
    args = dialog.arguments(tmp_path / "changed")
    assert "--no-tides" in args and "--no-water-weakening" in args
    assert float(args[args.index("--convective-traction-mpa") + 1]) == .05
    dialog.intact_control.setChecked(True)
    args = dialog.arguments(tmp_path / "control")
    assert args[args.index("--control") + 1] == "intact"
    assert "--no-tides" in args and "--convective-traction-mpa" not in args
    assert not dialog.tides.isEnabled()
    assert not dialog.convective_traction.isEnabled()
    dialog._set_busy(True)
    assert all(not control.isEnabled() for control in (*dialog._shell_controls, *dialog._onset_controls))


@pytest.mark.parametrize("mode,runner,effects", [
    ("onset", "run_genesis_onset.py", True),
    ("shell", "run_genesis_shell.py", False),
    ("thermal", "run_genesis.py", False),
])
def test_prior_modes_still_route_to_their_runners(dialog, tmp_path, mode, runner, effects):
    dialog.mode.setCurrentIndex(dialog.mode.findData(mode))
    dialog.tides.setChecked(False)
    dialog.water_weakening.setChecked(False)
    args = dialog.arguments(tmp_path / "legacy")
    assert Path(args[1]).name == runner
    assert ("--no-tides" in args) is effects
    assert ("--no-water-weakening" in args) is effects
    assert ("--regularization-km" in args) is effects
    assert ("--shell-step-myr" in args) is (mode != "thermal")


def test_mobile_results_selected_first_with_stop_reason(dialog, tmp_path):
    summary = {"status": "mobile_mesh_quality_limit", "geometry": "material",
               "events_myr": {"ocean_start": 1.04},
               "shell": {"damaged_area_fraction": .3, "intact_region_count": 2},
               "onset": {"mean_speed_cm_yr": .234}}
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    for name in ("genesis_mobile.png", "genesis_mobile.gif", "genesis_onset.png",
                 "genesis_shell.png", "genesis_history.png"):
        Image.new("RGB", (80, 40), "navy").save(tmp_path / name)
    dialog.output_path = tmp_path
    dialog._finished(0, QProcess.ExitStatus.NormalExit)
    assert "Расчёт остановлен: mobile_mesh_quality_limit" in dialog.status.text()
    assert "0.234 см/год" in dialog.status.text()
    assert "для продолжения требуется" not in dialog.status.text()
    assert dialog.result_view.currentData().endswith("genesis_mobile.png")
    assert not dialog._pixmap.isNull()
    dialog.result_view.setCurrentIndex(1)
    assert dialog.result_view.currentData().endswith("genesis_mobile.gif")
    assert dialog._movie is not None and dialog._movie.isValid()


def test_mobile_cancel_names_its_own_checkpoint(dialog, tmp_path):
    dialog._stopping = True
    dialog._run_mode = "mobile"
    dialog.output_path = tmp_path
    dialog._finished(0, QProcess.ExitStatus.NormalExit)
    assert "run_genesis_mobile.py --resume mobile_checkpoint.npz" in dialog.status.text()
    assert "onset_checkpoint" not in dialog.status.text()


def test_mobile_renderer_uses_current_geometry_and_names_its_limit(tmp_path, monkeypatch):
    import matplotlib.figure
    import visualization.genesis_onset as plots

    mesh = build_icosphere(1)
    angle = .2
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    current = replace(mesh, vertices=mesh.vertices @ rotation.T, centroids=mesh.centroids @ rotation.T)
    fraction = np.linspace(0, 1, mesh.cell_count)
    fields = {"lid_thickness_km": fraction*15, "damage": fraction, "water_access": fraction,
              "speed_cm_yr": fraction*4, "displacement_km": fraction*250,
              "tidal_stress_mpa": fraction*.04,
              "failed_edges": np.arange(len(mesh.shared_edges)) % 7 == 0}
    sampled_meshes, labels = [], []
    rasterize = plots.rasterize_cells
    savefig = matplotlib.figure.Figure.savefig

    def observe_geometry(actual_mesh, values):
        sampled_meshes.append(actual_mesh)
        return rasterize(actual_mesh, values)

    def observe_labels(figure, *args, **kwargs):
        labels.extend(text.get_text() for text in figure.texts)
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(plots, "rasterize_cells", observe_geometry)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", observe_labels)
    snapshot = save_onset_snapshot(current, fields, tmp_path / "mobile.png", 1.2,
                                  {"geometry": "material", "status": "mobile_mesh_quality_limit"})
    assert len(sampled_meshes) == 6 and all(actual is current for actual in sampled_meshes)
    assert "текущей геометрии материала" in "\n".join(labels)
    assert "исходной сфере" not in "\n".join(labels)
    assert "mobile_mesh_quality_limit" in "\n".join(labels)
    with Image.open(snapshot) as image:
        assert image.width >= 1300 and image.height >= 1100
        assert image.convert("RGB").getextrema()[0][0] < 100


def test_mobile_unavailable_falls_back_to_onset(app, tmp_path):
    config = tmp_path / "configs"
    config.mkdir()
    (config / "genesis_moon.yaml").write_bytes((ROOT / "configs/genesis_moon.yaml").read_bytes())
    (tmp_path / "run_genesis_onset.py").touch()
    widget = GenesisDialog(project_root=tmp_path)
    try:
        assert widget.mode.currentData() == "onset"
    finally:
        widget.close()
