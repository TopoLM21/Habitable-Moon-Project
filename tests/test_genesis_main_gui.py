"""The completed Genesis world stays usable in the main Qt workflow."""
from dataclasses import replace
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QProcess, QTimer, Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QDialog, QListWidgetItem
import pytest

from moon_gui import app as gui
from moon_gui.app import MATURE_SCENARIO_ID, MoonWindow
from moon_gui.backend import read_genesis_continuation
from moon_gui.genesis_schema import SatelliteOrigin
from moon_gui.genesis_starter_dialog import GenesisStarterDialog
from test_genesis_continuation_backend import update_report, write_pair as write_checkpoint_pair


def surface_path(root):
    age = json.loads((root / "continuation.json").read_text(encoding="utf-8"))["final_time_myr"]
    return root / "mature_run/hydrosphere_frames" / f"surface_{age:013.4f}_Myr.png"


def plate_path(root):
    age = json.loads((root / "continuation.json").read_text(encoding="utf-8"))["final_time_myr"]
    return root / "mature_run/plate_frames" / f"plate_{age:013.4f}_Myr.png"


def write_pair(root, **kwargs):
    root = write_checkpoint_pair(root, **kwargs)
    for path, color in ((surface_path(root), 0x227799), (plate_path(root), 0xAABB33),
                        (root / "continuation.png", 0x779922)):
        path.parent.mkdir(parents=True, exist_ok=True)
        preview = QImage(24, 24, QImage.Format.Format_RGB32)
        preview.fill(color)
        assert preview.save(str(path))
    return root


@pytest.fixture
def window():
    app = QApplication.instance() or QApplication([])
    result = MoonWindow()
    result.refresh_timer.stop()
    yield result
    result.controller._set_state("Stopped")
    result.close()
    app.processEvents()


@pytest.fixture
def pair(tmp_path):
    root = write_pair(tmp_path / "starter" / "continuation", subdivisions=2)
    preview = QImage(24, 24, QImage.Format.Format_RGB32)
    preview.fill(0x227799)
    assert preview.save(str(root / "continuation.png"))
    (root.parent / "summary.json").write_text(json.dumps({
        "candidate_partition": True, "status": "first_partition",
        "final": {"time_myr": .9906219482421875, "domain_count": 2, "damaged_area_fraction": .2},
        "continuation": {"status": "completed", "requested_duration_myr": 10.},
    }), encoding="utf-8")
    return root


def select(window, scenario):
    window.scenario.setCurrentIndex(window.scenario.findData(scenario))


def metric_values(window):
    return {window.metrics.item(row, 0).text(): window.metrics.item(row, 1).text()
            for row in range(window.metrics.rowCount())}


def form_values(window):
    return (window.config_field.edit.text(), window.output_field.edit.text(),
            window.resume_field.edit.text(), window.subdivisions.currentText(),
            window.end_time.value(), window.dt.value(), window.checkpoint_interval.value(),
            window.frame_interval.value(), window._execution_form_values())


def test_completed_starter_returns_to_main_with_same_world(window, pair, monkeypatch):
    from moon_gui import genesis_starter_dialog

    outcomes = []
    published = []

    class SavedResultDialog(GenesisStarterDialog):
        def __init__(self, parent):
            super().__init__(parent)
            self.output_path = pair.parent
            self.continuation_ready.connect(published.append)

        def exec(self):
            # A failed automatic return is bounded and produces Rejected, rather
            # than hanging the test inside the modal event loop.
            timeout = QTimer(self)
            timeout.setSingleShot(True)
            timeout.timeout.connect(self.reject)
            timeout.start(1000)
            QTimer.singleShot(0, lambda: self._finished(0, QProcess.ExitStatus.NormalExit))
            outcome = super().exec()
            timeout.stop()
            outcomes.append(outcome)
            return outcome

    monkeypatch.setattr(genesis_starter_dialog, "GenesisStarterDialog", SavedResultDialog)
    monkeypatch.setattr(window.controller, "start", lambda _: pytest.fail("Adoption cannot start a fresh run"))
    window._open_genesis_starter()
    assert outcomes == [QDialog.DialogCode.Accepted]
    assert published == [str(pair)]
    assert window.genesis_continuation.root == pair
    assert window.scenario.currentData() == SatelliteOrigin.DISK_QUIET.value
    assert window.controller.current_time == read_genesis_continuation(pair).time_myr
    assert window.preview_view.currentData() == "surface"
    assert window.current_artifact == surface_path(pair)
    assert not window.preview.pixmap().isNull()
    assert metric_values(window)["plate_count"] == "2"
    assert metric_values(window)["mantle_temperature_k"] == "1580"
    assert window.start_button.text() == "Продолжить расчёт"
    assert window.start_button.isEnabled()


def test_loaded_pair_exposes_normal_cpu_output_and_grid_controls(window, pair):
    window._adopt_genesis_continuation(pair)
    assert window.numerical_group.isEnabled()
    assert window.end_time_label.text() == "Продлить на"
    assert window.end_time.isEnabled()
    assert window.dt.isEnabled()
    assert window.checkpoint_interval.isEnabled()
    assert window.subdivisions.isEnabled()
    assert window.subdivisions.currentText() == "2"
    assert not window.config_field.isEnabled()
    assert not window.resume_field.isEnabled()
    assert window.cpu_mode.isEnabled()
    assert window.frame_interval.isEnabled()
    assert window.cpu_workers.isEnabled() and not window.cpu_workers.isHidden()
    assert window.render_workers.isEnabled() and not window.render_workers.isHidden()
    assert window.surface_only.isEnabled() and window.finalize.isEnabled()
    assert not window.output_group.isHidden()
    assert "генезиса A" in window.scenario.currentText()
    assert window.config_field.path() == pair / "mature_config.yaml"
    assert window.resume_field.path() == pair


def test_continuing_uses_exact_saved_age_plus_additional_duration(window, pair, monkeypatch):
    window._adopt_genesis_continuation(pair)
    info = window.genesis_continuation
    window.end_time.setValue(.1234)
    window.dt.setValue(.0001)
    window.checkpoint_interval.setValue(.1234)
    spec = window._make_spec().normalized()
    spec.validate()
    assert spec.genesis_continuation == pair
    assert spec.resume_checkpoint is None
    assert spec.source_config == info.config
    assert spec.subdivisions == info.subdivisions
    assert spec.start_time_myr() == info.time_myr
    assert spec.end_time_myr == info.time_myr + .1234
    assert spec.end_time_myr != round(info.time_myr, 4) + .1234
    assert spec.dt_myr == .0001
    assert spec.runner.name == "run_genesis_starter_continuation.py"
    assert spec.cpu_optimized and not spec.gpu_surface
    assert spec.output_dir != pair
    started = []
    monkeypatch.setattr(window.controller, "start", started.append)
    monkeypatch.setattr(window, "_open_genesis_starter", lambda: pytest.fail("Continue must not reopen the starter"))
    window.start_button.click()
    assert len(started) == 1
    assert started[0].genesis_continuation == pair
    assert started[0].end_time_myr == info.time_myr + .1234


def test_normal_execution_and_output_choices_reach_paired_runner(window, pair):
    window._adopt_genesis_continuation(pair)
    window.cpu_workers.setCurrentText("4")
    window.render_workers.setCurrentText("8")
    window.low_priority.setChecked(True)
    window.cell_kernels.setChecked(True)
    window.assignment_optimized.setChecked(False)
    window.assignment_columns.setChecked(True)
    window.boundary_forces.setChecked(True)
    window.dt.setValue(.5)
    window.frame_interval.setValue(2.)
    window.surface_only.setChecked(True)
    window.finalize.setChecked(False)
    spec = window._make_spec().normalized()
    spec.validate()
    assert spec.genesis_continuation == pair
    assert spec.cpu_optimized and spec.cpu_workers == 4 and spec.render_workers == 8
    assert spec.process_priority == "below_normal"
    assert spec.cell_kernels and spec.assignment_columns and spec.boundary_forces
    assert not spec.assignment_optimized
    assert spec.frame_interval_myr == 2.
    assert spec.surface_only_frames and not spec.finalize
    window.cpu_mode.setCurrentIndex(window.cpu_mode.findData(False))
    plain = window._make_spec().normalized()
    plain.validate()
    assert not plain.cpu_optimized
    assert plain.cpu_workers == plain.render_workers == 1
    assert plain.process_priority == "normal"
    assert not plain.cell_kernels and not plain.assignment_columns and not plain.boundary_forces
    assert not window.cpu_workers.isEnabled()


def test_gpu_is_visible_but_unavailable_for_paired_world_and_restored_for_mature(window, pair):
    gpu_index = window.cpu_mode.findData("gpu_surface")
    window.cpu_mode.setCurrentIndex(gpu_index)
    window.gpu_device.setValue(2)
    window._adopt_genesis_continuation(pair)
    assert window.cpu_mode.findData("gpu_surface") == gpu_index
    assert not window.cpu_mode.model().item(gpu_index).isEnabled()
    assert window.cpu_mode.currentData() is True
    assert not window.execution_hint.isHidden()
    assert "GPU" in window.execution_hint.text() and "не поддерживается" in window.execution_hint.text()
    assert not window._make_spec().gpu_surface
    select(window, MATURE_SCENARIO_ID)
    assert window.cpu_mode.model().item(gpu_index).isEnabled()
    assert window.cpu_mode.currentData() == "gpu_surface"
    assert window.gpu_device.value() == 2
    assert window.execution_hint.isHidden()


def test_target_grid_requests_refinement_without_mutating_saved_world(window, pair):
    from moon_gui.backend import build_segment_command

    window._adopt_genesis_continuation(pair)
    before = (pair / "mature_config.yaml").read_bytes()
    window.subdivisions.setCurrentText("4")
    spec = window._make_spec().normalized()
    spec.validate()
    assert spec.subdivisions == 4
    assert spec.genesis_continuation == pair
    assert read_genesis_continuation(pair).subdivisions == 2
    assert (pair / "mature_config.yaml").read_bytes() == before
    command = build_segment_command(spec, target_time_myr=spec.end_time_myr,
        checkpoint_dir=spec.output_dir / "gui_checkpoint_refined_Myr",
        resume_checkpoint=pair, final_segment=True)
    assert command[command.index("--subdivisions") + 1] == "4"
    assert command[command.index("--resume") + 1] == str(pair)
    assert "новые детали не восстанавливаются" in window.subdivisions.toolTip()
    select(window, MATURE_SCENARIO_ID)
    select(window, SatelliteOrigin.DISK_QUIET.value)
    assert window.subdivisions.currentText() == "4"
    assert window.subdivisions.findText("2") >= 0


def test_coarsening_is_unavailable_and_cannot_be_forced_programmatically(window, tmp_path):
    source = write_pair(tmp_path / "refined", subdivisions=4)
    window._adopt_genesis_continuation(source)
    coarse = window.subdivisions.findText("3")
    assert not window.subdivisions.model().item(coarse).isEnabled()
    window.subdivisions.setCurrentIndex(coarse)
    with pytest.raises(ValueError, match="[Cc]oarsen|[Rr]efin|smaller"):
        window._make_spec().normalized().validate()


def test_switching_scenarios_restores_each_form_without_leaking_young_config(window, pair, tmp_path):
    window.output_field.set_path(tmp_path / "ordinary_output")
    window.resume_field.set_path(tmp_path / "ordinary_checkpoint")
    window.end_time.setValue(640.)
    window.dt.setValue(4.)
    window.checkpoint_interval.setValue(20.)
    window.frame_interval.setValue(40.)
    window.cpu_workers.setCurrentText("2")
    ordinary = form_values(window)
    window._adopt_genesis_continuation(pair)
    window.end_time.setValue(12.5)
    window.dt.setValue(.5)
    window.checkpoint_interval.setValue(2.5)
    window.frame_interval.setValue(2.5)
    window.cpu_workers.setCurrentText("4")
    window.render_workers.setCurrentText("8")
    window.low_priority.setChecked(False)
    young = form_values(window)
    select(window, MATURE_SCENARIO_ID)
    assert form_values(window) == ordinary
    assert window.subdivisions.findText("2") == -1
    assert window.config_field.isEnabled() and window.subdivisions.isEnabled()
    ordinary_spec = window._make_spec()
    assert ordinary_spec.genesis_continuation is None
    assert ordinary_spec.source_config != pair / "mature_config.yaml"
    assert ordinary_spec.resume_checkpoint == tmp_path / "ordinary_checkpoint"
    window.end_time.setValue(800.)
    window.dt.setValue(8.)
    window.checkpoint_interval.setValue(40.)
    window.cpu_workers.setCurrentText("8")
    revised_ordinary = form_values(window)
    select(window, SatelliteOrigin.DISK_QUIET.value)
    assert form_values(window) == young
    assert window._make_spec().genesis_continuation == pair
    select(window, MATURE_SCENARIO_ID)
    assert form_values(window) == revised_ordinary


@pytest.mark.parametrize("state", ["Preparing", "Running", "Pausing", "Paused", "Stopping"])
def test_active_or_paused_pair_locks_context_and_time(window, pair, monkeypatch, state):
    window._adopt_genesis_continuation(pair)
    monkeypatch.setattr(window.controller, "start", lambda _: pytest.fail("Busy world cannot restart"))
    window.controller._set_state(state)
    for control in (window.scenario, window.start_button, window.end_time, window.dt,
                    window.checkpoint_interval, window.subdivisions, window.load_genesis_button,
                    window.genesis_starter_button, window.cpu_mode, window.cpu_workers,
                    window.render_workers, window.frame_interval, window.finalize):
        assert not control.isEnabled()
    select(window, MATURE_SCENARIO_ID)
    assert window.scenario.currentData() == SatelliteOrigin.DISK_QUIET.value
    with pytest.raises(ValueError, match="текущий прогон"):
        window._adopt_genesis_continuation(pair)
    window._start_run()
    assert window.genesis_continuation.root == pair
    window.controller._set_state("Stopped")
    assert window.end_time.isEnabled() and window.dt.isEnabled()
    assert window.subdivisions.isEnabled()


def test_failed_starter_does_not_replace_previously_loaded_world(window, pair, monkeypatch, tmp_path):
    from moon_gui import genesis_starter_dialog

    window._adopt_genesis_continuation(pair)
    before = window.genesis_continuation
    outcomes = []

    class FailedDialog(GenesisStarterDialog):
        def __init__(self, parent):
            super().__init__(parent)
            self.output_path = tmp_path / "failed_starter"

        def exec(self):
            QTimer.singleShot(0, lambda: self._finished(1, QProcess.ExitStatus.NormalExit))
            QTimer.singleShot(10, self.reject)
            result = super().exec()
            outcomes.append(result)
            return result

    monkeypatch.setattr(genesis_starter_dialog, "GenesisStarterDialog", FailedDialog)
    window._open_genesis_starter()
    assert outcomes == [QDialog.DialogCode.Rejected]
    assert window.genesis_continuation is before
    assert window.resume_field.path() == pair
    assert window.current_artifact == surface_path(pair)


def test_completed_segment_becomes_source_for_next_main_run(window, pair, tmp_path):
    window._adopt_genesis_continuation(pair)
    output = tmp_path / "next_run"
    completed = write_pair(output / "gui_checkpoint_20_Myr", duration=20., subdivisions=2)
    info = read_genesis_continuation(completed)
    window.end_time.setValue(10.)
    window.controller.spec = replace(window._make_spec(), output_dir=output)
    window.output_field.set_path(output)
    window.controller.current_time = info.time_myr
    window.controller.resume_checkpoint = completed
    window._segment_completed(info.time_myr, str(completed))
    window._run_completed(str(output))
    assert window.genesis_continuation.root == completed
    assert window.resume_field.path() == completed
    assert window.config_field.path() == info.config
    assert window.current_artifact == surface_path(completed)
    next_spec = window._make_spec()
    assert next_spec.genesis_continuation == completed
    assert next_spec.source_config == info.config
    assert next_spec.end_time_myr == info.time_myr + 10.


@pytest.mark.parametrize("view", ["surface", "plates", "genesis"])
def test_preview_choice_survives_result_refresh_and_new_segment(window, pair, tmp_path, view):
    window._adopt_genesis_continuation(pair)
    window.preview_view.setCurrentIndex(window.preview_view.findData(view))
    expected = {"surface": surface_path(pair), "plates": plate_path(pair), "genesis": pair / "continuation.png"}
    window._refresh_results()
    window._refresh_results()
    assert window.preview_view.currentData() == view
    assert window.current_artifact == expected[view]
    output = tmp_path / "next_preview_run"
    completed = write_pair(output / "gui_checkpoint_20_Myr", duration=20., subdivisions=2)
    info = read_genesis_continuation(completed)
    window.controller.spec = replace(window._make_spec(), output_dir=output)
    window.output_field.set_path(output)
    window._segment_completed(info.time_myr, str(completed))
    window._run_completed(str(output))
    expected = {"surface": surface_path(completed), "plates": plate_path(completed),
                "genesis": completed / "continuation.png"}
    assert window.preview_view.currentData() == view
    assert window.current_artifact == expected[view]
    assert not window.preview.pixmap().isNull()


def test_manual_artifact_stays_selected_until_a_preview_action(window, pair, tmp_path):
    window._adopt_genesis_continuation(pair)
    diagnostic = pair / "continuation.png"
    item = QListWidgetItem("Chosen saved graph")
    item.setData(Qt.ItemDataRole.UserRole, str(diagnostic))
    window._artifact_activated(item)
    window._refresh_results()
    assert window.current_artifact == diagnostic
    output = tmp_path / "manual_preview_run"
    completed = write_pair(output / "gui_checkpoint_20_Myr", duration=20., subdivisions=2)
    window.controller.spec = replace(window._make_spec(), output_dir=output)
    window.output_field.set_path(output)
    window._segment_completed(read_genesis_continuation(completed).time_myr, str(completed))
    assert window.current_artifact == diagnostic
    window._show_latest()
    assert window.preview_view.currentData() == "surface"
    assert window.current_artifact == surface_path(completed)


def test_missing_selected_map_does_not_show_a_different_image(window, pair):
    plate_path(pair).unlink()
    window._adopt_genesis_continuation(pair)
    assert window.current_artifact == surface_path(pair)
    window.preview_view.setCurrentIndex(window.preview_view.findData("plates"))
    window._refresh_results()
    assert window.current_artifact is None
    assert window.preview.pixmap().isNull()
    assert "Карта плит" in window.preview.text()
    assert window.preview_view.currentData() == "plates"
    image = QImage(24, 24, QImage.Format.Format_RGB32)
    image.fill(0x55AA22)
    assert image.save(str(plate_path(pair)))
    window._refresh_results()
    assert window.current_artifact == plate_path(pair)
    assert window.preview_view.currentData() == "plates"


def test_incomplete_new_image_is_retried_on_next_refresh(window, pair):
    plate_path(pair).unlink()
    window._adopt_genesis_continuation(pair)
    window.preview_view.setCurrentIndex(window.preview_view.findData("plates"))
    plate_path(pair).write_bytes(b"incomplete PNG from an active renderer")
    window._refresh_results()
    assert window.current_artifact is None
    image = QImage(24, 24, QImage.Format.Format_RGB32)
    image.fill(0x11CC55)
    assert image.save(str(plate_path(pair)))
    window._refresh_results()
    assert window.current_artifact == plate_path(pair)
    assert not window.preview.pixmap().isNull()


def test_new_handoff_returns_to_surface_after_prior_diagnostic_choice(window, pair):
    window.preview_view.setCurrentIndex(window.preview_view.findData("genesis"))
    window._adopt_genesis_continuation(pair)
    assert window.preview_view.currentData() == "surface"
    assert window.current_artifact == surface_path(pair)


def test_new_completed_world_missing_a_view_never_falls_back_to_older_map(window, pair, tmp_path):
    window._adopt_genesis_continuation(pair)
    window.preview_view.setCurrentIndex(window.preview_view.findData("plates"))
    output = tmp_path / "new_completed_run"
    completed = write_pair(output / "gui_checkpoint_20_Myr", duration=20., subdivisions=2)
    plate_path(completed).unlink()
    # The result refresh may occur before the process-finished signal updates
    # the window's stored source; the newer completed pair is already visible.
    window.output_field.set_path(output)
    window._refresh_results()
    assert window.genesis_continuation.root == pair
    assert window.current_artifact is None
    assert "Карта плит" in window.preview.text()
    assert window.preview_view.currentData() == "plates"


def test_pending_run_can_still_display_the_last_completed_source(window, pair, tmp_path):
    window._adopt_genesis_continuation(pair)
    window.preview_view.setCurrentIndex(window.preview_view.findData("plates"))
    window.output_field.set_path(tmp_path / "no_completed_segment_yet")
    window._refresh_results()
    assert window.current_artifact == plate_path(pair)


def test_invalid_pair_cannot_replace_current_context(window, pair, tmp_path):
    window._adopt_genesis_continuation(pair)
    before = window.genesis_continuation
    incomplete = write_pair(tmp_path / "incomplete", duration=20., subdivisions=2)
    update_report(incomplete, status="validation_failed")
    with pytest.raises(ValueError, match="validation"):
        window._adopt_genesis_continuation(incomplete)
    assert window.genesis_continuation is before
    assert window.resume_field.path() == pair
    assert window.current_artifact == surface_path(pair)


@pytest.mark.parametrize("relative", ["", "mature_checkpoint"])
def test_ordinary_checkpoint_picker_redirects_paired_world_to_genesis(window, pair, monkeypatch, relative):
    monkeypatch.setattr(gui.QFileDialog, "getExistingDirectory", lambda *_: str(pair / relative))
    window._browse_checkpoint()
    assert window.scenario.currentData() == SatelliteOrigin.DISK_QUIET.value
    assert window.genesis_continuation.root == pair
    spec = window._make_spec()
    assert spec.resume_checkpoint is None
    assert spec.genesis_continuation == pair


def test_corrupt_paired_selection_warns_and_keeps_current_world(window, pair, monkeypatch, tmp_path):
    window._adopt_genesis_continuation(pair)
    previous = window.genesis_continuation
    corrupt = write_pair(tmp_path / "corrupt_pair", subdivisions=2)
    archive = corrupt / "young_context" / "fracture_memory.npz"
    archive.write_bytes(archive.read_bytes() + b"altered after checkpoint")
    warnings = []
    monkeypatch.setattr(gui.QFileDialog, "getExistingDirectory", lambda *_: str(corrupt / "mature_checkpoint"))
    monkeypatch.setattr(gui.QMessageBox, "warning", lambda *args: warnings.append(args[1:]))
    window._browse_checkpoint()
    assert len(warnings) == 1
    assert "генезис" in warnings[0][0]
    assert "integrity" in warnings[0][1]
    assert window.genesis_continuation is previous
    assert window.resume_field.path() == pair
    assert window.current_artifact == surface_path(pair)


def test_dedicated_picker_accepts_whole_starter_output(window, pair, monkeypatch):
    monkeypatch.setattr(gui.QFileDialog, "getExistingDirectory", lambda *_: str(pair.parent))
    monkeypatch.setattr(gui.QMessageBox, "warning", lambda *_: pytest.fail("A completed starter root must open"))
    window._browse_genesis_continuation()
    assert window.genesis_continuation.root == pair
    assert window.controller.current_time == read_genesis_continuation(pair).time_myr
    assert window.current_artifact == surface_path(pair)
    assert window.start_button.text() == "Продолжить расчёт"


def test_dedicated_picker_selects_latest_completed_segment_from_gui_run(window, monkeypatch, tmp_path):
    output = tmp_path / "gui_run"
    write_pair(output / "gui_checkpoint_20_Myr", duration=20., subdivisions=2)
    completed = write_pair(output / "gui_checkpoint_30_Myr", duration=30., subdivisions=2)
    incomplete = write_pair(output / "gui_checkpoint_40_Myr", duration=40., subdivisions=2)
    update_report(incomplete, status="validation_failed")
    monkeypatch.setattr(gui.QFileDialog, "getExistingDirectory", lambda *_: str(output))
    monkeypatch.setattr(gui.QMessageBox, "warning", lambda *_: pytest.fail("A completed GUI segment must open"))
    window._browse_genesis_continuation()
    info = read_genesis_continuation(completed)
    assert window.genesis_continuation.root == completed
    assert window.controller.current_time == info.time_myr
    assert window.resume_field.path() == completed
    assert window.current_artifact == surface_path(completed)
    spec = window._make_spec()
    assert spec.genesis_continuation == completed
    assert spec.start_time_myr() == info.time_myr
    assert spec.end_time_myr == info.time_myr + window.end_time.value()
