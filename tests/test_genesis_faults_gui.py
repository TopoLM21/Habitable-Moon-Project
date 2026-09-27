"""Fault-mode routing, independent result files, and honest material maps."""
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
from visualization.genesis_faults import save_fault_snapshot


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _project(path, *, faults=True):
    config = path / "configs"
    config.mkdir()
    (config / "genesis_moon.yaml").write_bytes((ROOT / "configs/genesis_moon.yaml").read_bytes())
    for name in ("mobile", "onset", "shell"):
        (path / f"run_genesis_{name}.py").touch()
    if faults:
        (path / "run_genesis_faults.py").touch()


@pytest.fixture
def dialog(app, tmp_path):
    _project(tmp_path)
    widget = GenesisDialog(project_root=tmp_path)
    yield widget
    widget.close()


def test_faults_default_and_retain_orbit_water_time_controls(dialog, tmp_path):
    assert dialog.mode.currentData() == "faults"
    assert dialog.mode.count() == 5
    assert dialog.mode.currentText() == "Трение и сдвиг разломных зон"
    args = dialog.arguments(tmp_path / "run")
    assert Path(args[1]).name == "run_genesis_faults.py"
    assert float(args[args.index("--shell-step-myr") + 1]) == .002
    assert float(args[args.index("--regularization-km") + 1]) == 800
    assert not {"--no-tides", "--no-water-weakening"}.intersection(args)
    dialog.tides.setChecked(False)
    dialog.water_weakening.setChecked(False)
    args = dialog.arguments(tmp_path / "changed")
    assert "--no-tides" in args and "--no-water-weakening" in args
    dialog.intact_control.setChecked(True)
    args = dialog.arguments(tmp_path / "control")
    assert args[args.index("--control") + 1] == "intact"
    assert "--no-tides" in args
    assert not dialog.tides.isEnabled()
    assert not dialog.convective_traction.isEnabled()
    dialog._set_busy(True)
    assert all(not control.isEnabled() for control in (*dialog._shell_controls, *dialog._onset_controls))


@pytest.mark.parametrize("mode,runner,effects", [
    ("mobile", "run_genesis_mobile.py", True),
    ("onset", "run_genesis_onset.py", True),
    ("shell", "run_genesis_shell.py", False),
    ("thermal", "run_genesis.py", False),
])
def test_fault_mode_preserves_all_prior_routes(dialog, tmp_path, mode, runner, effects):
    dialog.mode.setCurrentIndex(dialog.mode.findData(mode))
    dialog.tides.setChecked(False)
    args = dialog.arguments(tmp_path / "legacy")
    assert Path(args[1]).name == runner
    assert ("--no-tides" in args) is effects
    assert ("--regularization-km" in args) is effects
    assert ("--shell-step-myr" in args) is (mode != "thermal")


def test_fault_mode_unavailable_preserves_four_mode_mobile_fallback(app, tmp_path):
    _project(tmp_path, faults=False)
    widget = GenesisDialog(project_root=tmp_path)
    try:
        assert widget.mode.currentData() == "mobile"
        assert widget.mode.count() == 4
        assert widget.mode.findData("faults") == -1
    finally:
        widget.close()


def test_fault_preview_precedes_mobile_maps_and_preserves_limit(dialog, tmp_path):
    summary = {"geometry": "material", "status": "fault_equilibrium_limit",
               "events_myr": {"ocean_start": 1.04},
               "shell": {"damaged_area_fraction": .3, "intact_region_count": 2},
               "onset": {"mean_speed_cm_yr": .234}}
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    for name in ("genesis_faults.png", "genesis_faults.gif", "genesis_mobile.png",
                 "genesis_shell.png", "genesis_history.png"):
        Image.new("RGB", (80, 40), "navy").save(tmp_path / name)
    dialog.output_path = tmp_path
    dialog._finished(0, QProcess.ExitStatus.NormalExit)
    assert "Расчёт остановлен: fault_equilibrium_limit" in dialog.status.text()
    assert "0.234 см/год" in dialog.status.text()
    assert dialog.result_view.currentData().endswith("genesis_faults.png")
    assert not dialog._pixmap.isNull()
    dialog.result_view.setCurrentIndex(1)
    assert dialog.result_view.currentData().endswith("genesis_faults.gif")
    assert dialog._movie is not None and dialog._movie.isValid()


def test_fault_cancel_names_independent_checkpoint(dialog, tmp_path):
    dialog._stopping = True
    dialog._run_mode = "faults"
    dialog.output_path = tmp_path
    dialog._finished(0, QProcess.ExitStatus.NormalExit)
    assert "run_genesis_faults.py --resume fault_checkpoint.npz" in dialog.status.text()
    assert "mobile_checkpoint" not in dialog.status.text()


def test_completed_fault_run_reports_no_formed_zones(dialog, tmp_path):
    summary = {"status": "completed", "shell": {
        "first_fracture_time_myr": None, "fault_active_area_fraction": 0.,
        "max_equivalent_slip_km": 0., "intact_region_count": 1}}
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    dialog.output_path = tmp_path
    dialog._finished(0, QProcess.ExitStatus.NormalExit)
    text = dialog.status.text()
    assert "Расчёт завершён до заданного времени" in text
    assert "Активные разломные зоны не образовались (0% площади)" in text
    assert "Максимальный эквивалентный сдвиг: 0 км" in text


def test_fault_result_keeps_small_nonzero_activity_visible(dialog, tmp_path):
    summary = {"status": "completed", "shell": {
        "fault_active_area_fraction": 1e-6, "max_equivalent_slip_km": 1e-5}}
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    dialog.output_path = tmp_path
    dialog._finished(0, QProcess.ExitStatus.NormalExit)
    text = dialog.status.text()
    assert "Активные разломные зоны: 0.0001% площади" in text
    assert "Максимальный эквивалентный сдвиг: 1e-05 км" in text
    assert "не образовались" not in text


def _fields(mesh):
    fraction = np.linspace(0, 1, mesh.cell_count)
    return {"damage": fraction.copy(), "water_access": fraction.copy(),
            "equivalent_slip_km": fraction*30, "slip_rate_cm_yr": fraction*4,
            "shear_stress_mpa": fraction*8, "shear_strength_mpa": fraction*12,
            "failed_edges": np.arange(len(mesh.shared_edges)) % 7 == 0}


def test_fault_maps_use_current_geometry_fixed_limits_and_equivalent_slip_label(tmp_path, monkeypatch):
    import matplotlib.figure
    import visualization.genesis_faults as plots

    mesh = build_icosphere(1)
    angle = .2
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    current = replace(mesh, vertices=mesh.vertices @ rotation.T, centroids=mesh.centroids @ rotation.T)
    sampled_meshes, labels, scales, extensions = [], [], [], []
    rasterize, savefig = plots.rasterize_cells, matplotlib.figure.Figure.savefig
    colorbar = matplotlib.figure.Figure.colorbar

    def observe_geometry(actual_mesh, values):
        sampled_meshes.append(actual_mesh)
        return rasterize(actual_mesh, values)

    def observe_labels(figure, *args, **kwargs):
        labels.extend(text.get_text() for text in figure.texts)
        return savefig(figure, *args, **kwargs)

    def observe_scale(figure, mappable, *args, **kwargs):
        scales.append(mappable.get_clim())
        extensions.append(kwargs.get("extend"))
        return colorbar(figure, mappable, *args, **kwargs)

    monkeypatch.setattr(plots, "rasterize_cells", observe_geometry)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", observe_labels)
    monkeypatch.setattr(matplotlib.figure.Figure, "colorbar", observe_scale)
    snapshot = save_fault_snapshot(current, _fields(current), tmp_path / "faults.png", 1.2,
                                   {"status": "fault_equilibrium_limit"})
    assert len(sampled_meshes) == 6 and all(actual is current for actual in sampled_meshes)
    assert scales == [(0, 1), (0, 1), (0, 20), (0, 5), (0, 10), (0, 10)]
    assert extensions == ["neither", "neither", "max", "neither", "neither", "max"]
    text = "\n".join(labels)
    assert "текущей геометрии материала" in text and "исходной сфере" not in text
    assert "Эквивалентный сдвиг = ширина зоны" in text
    assert "Оболочка остаётся связной" in text and "fault_equilibrium_limit" in text
    with Image.open(snapshot) as image:
        assert image.width >= 1300 and image.height >= 1100
        assert image.convert("RGB").getextrema()[0][0] < 100


@pytest.mark.parametrize("active_count", [0, 40])
def test_fault_maps_mask_only_inactive_plane_fields(tmp_path, monkeypatch, active_count):
    import matplotlib.figure
    import visualization.genesis_faults as plots

    mesh = build_icosphere(1)
    data = _fields(mesh)
    active = np.arange(mesh.cell_count) < active_count
    data["fault_active"] = active
    sampled_values, texts = [], []
    rasterize = plots.rasterize_cells

    def observe_values(actual_mesh, values):
        sampled_values.append(values.copy())
        return rasterize(actual_mesh, values)

    def observe_figure(figure, *args, **kwargs):
        texts.extend(text.get_text() for text in figure.texts)
        for axis in figure.axes:
            texts.append(axis.get_title())
            texts.append(axis.get_xlabel())
            texts.extend(text.get_text() for text in axis.texts)

    monkeypatch.setattr(plots, "rasterize_cells", observe_values)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", observe_figure)
    save_fault_snapshot(mesh, data, tmp_path / "masked.png", 1.)
    assert all(np.isfinite(values).all() for values in sampled_values[:4])
    for values, name in zip(sampled_values[4:], ("shear_stress_mpa", "shear_strength_mpa")):
        np.testing.assert_array_equal(np.isnan(values), ~active)
        np.testing.assert_array_equal(values[active], data[name][active])
    text = "\n".join(texts)
    assert f"Активные разломные зоны: {active_count} из {mesh.cell_count} ячеек" in text
    assert "Максимум повреждения: 1" in text
    assert "накопленного эквивалентного сдвига: 30 км" in text
    assert "последний расчётный шаг" in text
    assert text.count("на активированных плоскостях") == 2
    assert ("Активных разломных зон нет" in text) == (active_count == 0)


@pytest.mark.parametrize("name", ["damage", "water_access", "equivalent_slip_km",
                                   "slip_rate_cm_yr", "shear_stress_mpa", "shear_strength_mpa"])
@pytest.mark.parametrize("bad", ["nonfinite", "shape"])
def test_fault_maps_reject_nonfinite_or_wrong_cell_fields(tmp_path, name, bad):
    mesh = build_icosphere(1)
    fields = _fields(mesh)
    if bad == "nonfinite":
        fields[name][0] = np.nan
    else:
        fields[name] = np.zeros(mesh.vertex_count)
    path = tmp_path / "invalid.png"
    with pytest.raises(ValueError, match=name):
        save_fault_snapshot(mesh, fields, path, 1)
    assert not path.exists()


@pytest.mark.parametrize("scale", [[10, 0], [0, np.inf], [0, 1, 2]])
def test_fault_maps_reject_invalid_color_scale(tmp_path, scale):
    mesh = build_icosphere(1)
    path = tmp_path / "invalid.png"
    with pytest.raises(ValueError, match="plot_limits"):
        save_fault_snapshot(mesh, _fields(mesh), path, 1,
                            {"plot_limits": {"equivalent_slip_km": scale}})
    assert not path.exists()
