"""High-resolution GUI contracts without constructing or evolving a large mesh."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from moon_gui.backend import (
    SUBDIVISION_CHOICES,
    RunSpec,
    cell_count,
    characteristic_cell_size_km,
    resolution_note,
    subdivision_for_cell_count,
    write_runtime_config,
)


@pytest.fixture
def resolution_project(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "project"
    configs = root / "configs"
    configs.mkdir(parents=True)
    for runner in ("run_long_evolution_v131.py", "run_long_evolution_v131_cpu.py"):
        (root / runner).write_text("pass\n", encoding="utf-8")
    config = configs / "canonical_moon.yaml"
    config.write_text(
        "mesh: {subdivisions: 5}\nmoon: {radius_km: 5287.0}\n", encoding="utf-8"
    )
    return root, config


def _checkpoint(path: Path, subdivisions: int = 5) -> Path:
    path.mkdir(parents=True)
    (path / "meta.json").write_text(
        json.dumps({"format": "moon_tectonics_checkpoint", "time_myr": 40.0}),
        encoding="utf-8",
    )
    np.savez_compressed(
        path / "state.npz",
        state_cell_plate=np.zeros(cell_count(subdivisions), dtype=np.int32),
    )
    return path


def test_resolution_choices_and_default_remain_explicit(resolution_project):
    root, config = resolution_project
    assert SUBDIVISION_CHOICES == (3, 4, 5, 6, 7, 8)
    assert RunSpec(root, config, root / "out").subdivisions == 5
    assert cell_count(7) == 327_680
    assert cell_count(8) == 1_310_720
    assert subdivision_for_cell_count(327_680) == 7
    assert subdivision_for_cell_count(1_310_720) == 8


@pytest.mark.parametrize("subdivisions, expected_km", [(7, 32.7), (8, 16.4)])
def test_resolution_note_uses_physical_scale_without_a_timing_promise(
    subdivisions, expected_km
):
    size = characteristic_cell_size_km(subdivisions, 5287.0)
    assert size == pytest.approx(expected_km, abs=0.05)
    assert characteristic_cell_size_km(subdivisions, 10574.0) == pytest.approx(2 * size)
    note = resolution_note(subdivisions, 5287.0)
    assert f"{expected_km:.1f}" in note
    assert "не измерены" in note
    assert isinstance(resolution_note(subdivisions), str)


@pytest.mark.parametrize("subdivisions", [7, 8])
def test_high_resolution_spec_and_config_need_no_mesh(resolution_project, subdivisions):
    root, config = resolution_project
    original = config.read_bytes()
    spec = RunSpec(root, config, root / "out", subdivisions=subdivisions)
    spec.validate()
    generated = yaml.safe_load(write_runtime_config(spec).read_text(encoding="utf-8"))
    assert generated["mesh"]["subdivisions"] == subdivisions
    assert generated["moon"]["radius_km"] == 5287.0
    assert config.read_bytes() == original


def test_subdivision_nine_is_outside_the_gui_contract(resolution_project):
    root, config = resolution_project
    with pytest.raises(ValueError):
        RunSpec(root, config, root / "out", subdivisions=9).validate()


def test_resume_cannot_change_checkpoint_resolution(resolution_project):
    root, config = resolution_project
    checkpoint = _checkpoint(root / "prior" / "checkpoint")
    with pytest.raises(ValueError):
        RunSpec(
            root, config, root / "out", subdivisions=7, resume_checkpoint=checkpoint
        ).validate()
    RunSpec(root, config, root / "out", resume_checkpoint=checkpoint).validate()


def test_resume_rejects_a_conflicting_reused_runtime_config(resolution_project):
    root, config = resolution_project
    output = root / "out"
    checkpoint = _checkpoint(output / "checkpoint")
    spec = RunSpec(root, config, output, resume_checkpoint=checkpoint)
    spec.runtime_config.write_text("mesh: {subdivisions: 7}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        spec.validate()
    spec.runtime_config.write_text("mesh: {subdivisions: 5}\n", encoding="utf-8")
    spec.validate()


@pytest.fixture
def resolution_window(resolution_project, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    import moon_gui.app as gui

    root, _ = resolution_project
    monkeypatch.setattr(gui, "PROJECT_ROOT", root)
    application = QApplication.instance() or QApplication([])
    window = gui.MoonWindow()
    yield gui, window
    window.close()
    application.processEvents()


def test_gui_offers_high_resolution_without_changing_default(resolution_window):
    _, window = resolution_window
    assert window.subdivisions.currentText() == "5"
    assert tuple(
        int(window.subdivisions.itemText(index))
        for index in range(window.subdivisions.count())
    ) == SUBDIVISION_CHOICES
    for subdivisions in (7, 8):
        window.subdivisions.setCurrentText(str(subdivisions))
        assert window._make_spec().subdivisions == subdivisions


def test_gui_resolution_label_follows_selected_radius_and_handles_invalid_config(
    resolution_project, resolution_window
):
    root, _ = resolution_project
    _, window = resolution_window
    config = root / "double_radius.yaml"
    config.write_text("moon: {radius_km: 10574.0}\n", encoding="utf-8")
    window.subdivisions.setCurrentText("7")
    window.config_field.set_path(config)
    assert "65.5" in window.resolution_label.text()
    assert "10574" in window.resolution_label.text()
    invalid = root / "invalid.yaml"
    invalid.write_text("moon: [\n", encoding="utf-8")
    window.config_field.set_path(invalid)
    assert "недоступен" in window.resolution_label.text()
    assert "327,680" in window.resolution_label.text()


@pytest.mark.parametrize("radius", [0, -1, float("inf"), float("nan")])
def test_cell_size_rejects_invalid_radius(radius):
    with pytest.raises(ValueError, match="radius"):
        characteristic_cell_size_km(7, radius)


@pytest.mark.parametrize("subdivisions", [7, 8])
def test_checkpoint_selection_recognizes_high_resolution_without_loading_mesh(
    resolution_project, resolution_window, monkeypatch, subdivisions
):
    root, _ = resolution_project
    gui, window = resolution_window
    # A small checkpoint is sufficient: only size introspection is replaced.
    checkpoint = _checkpoint(root / "results" / "prior" / "checkpoint", 3)
    monkeypatch.setattr(gui, "checkpoint_cell_count", lambda _: cell_count(subdivisions))
    monkeypatch.setattr(gui.QFileDialog, "getExistingDirectory", lambda *args: str(checkpoint))
    window._browse_checkpoint()
    assert window.resume_field.path() == checkpoint
    assert window.subdivisions.currentText() == str(subdivisions)


def test_unsupported_checkpoint_does_not_change_any_selected_fields(
    resolution_project, resolution_window, monkeypatch
):
    root, _ = resolution_project
    gui, window = resolution_window
    checkpoint = _checkpoint(root / "results" / "prior" / "checkpoint", 3)
    monkeypatch.setattr(gui, "checkpoint_cell_count", lambda _: cell_count(9))
    monkeypatch.setattr(gui.QFileDialog, "getExistingDirectory", lambda *args: str(checkpoint))
    warnings = []
    monkeypatch.setattr(gui.QMessageBox, "warning", lambda *args: warnings.append(args))
    before = (
        window.resume_field.edit.text(),
        window.config_field.edit.text(),
        window.output_field.edit.text(),
        window.subdivisions.currentText(),
    )
    window._browse_checkpoint()
    assert warnings
    assert before == (
        window.resume_field.edit.text(),
        window.config_field.edit.text(),
        window.output_field.edit.text(),
        window.subdivisions.currentText(),
    )


@pytest.mark.parametrize("subdivisions", [7, 8])
def test_high_resolution_confirmation_defaults_to_no_and_cancel_does_not_start(
    resolution_project, resolution_window, monkeypatch, subdivisions
):
    root, config = resolution_project
    gui, window = resolution_window
    spec = RunSpec(root, config, root / "out", subdivisions=subdivisions)
    monkeypatch.setattr(window, "_make_spec", lambda: spec)
    started = []
    questions = []
    monkeypatch.setattr(window.controller, "start", lambda value: started.append(value))

    def decline(*args, **kwargs):
        questions.append((args, kwargs))
        return gui.QMessageBox.StandardButton.No

    monkeypatch.setattr(gui.QMessageBox, "question", decline)
    window._start_run()
    assert len(questions) == 1
    args, kwargs = questions[0]
    default_button = kwargs.get("defaultButton", args[4] if len(args) > 4 else None)
    assert default_button == gui.QMessageBox.StandardButton.No
    assert not started


def test_high_resolution_confirmation_can_start_the_selected_spec(
    resolution_project, resolution_window, monkeypatch
):
    root, config = resolution_project
    gui, window = resolution_window
    spec = RunSpec(root, config, root / "out", subdivisions=7)
    monkeypatch.setattr(window, "_make_spec", lambda: spec)
    monkeypatch.setattr(
        gui.QMessageBox, "question", lambda *args, **kwargs: gui.QMessageBox.StandardButton.Yes
    )
    started = []
    monkeypatch.setattr(window.controller, "start", lambda value: started.append(value))
    window._start_run()
    assert len(started) == 1
    assert started[0].subdivisions == 7
