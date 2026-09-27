"""Exercise real starter -> CPU/render controls -> refinement -> pause/resume."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import yaml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox
from moon_gui.app import MoonWindow
from moon_gui.backend import preferred_preview, read_genesis_continuation
import moon_gui.genesis_starter_dialog as starter_ui


def verify_pair(info, subdivisions):
    """Check actual archive dimensions and clocks, not only the display label."""
    cells = 20 * 4**subdivisions
    root = info.root
    with np.load(root / "young_context/starter_checkpoint.npz", allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
        assert saved["damage"].shape == (cells,)
        assert saved["cell_plate"].shape == (cells,)
    with np.load(root / "young_context/fracture_memory.npz", allow_pickle=False) as saved:
        fracture_metadata = json.loads(str(saved["metadata"]))
        damage = saved["damage"]
        assert damage.shape == (cells,)
    with np.load(root / "mature_checkpoint/state.npz", allow_pickle=False) as saved:
        assert saved["state_cell_plate"].shape == (cells,)
        assert np.array_equal(damage, saved["tidal_damage"])
    config = yaml.safe_load((root / "mature_config.yaml").read_text(encoding="utf-8"))
    parameters = json.loads((root / "young_context/parameters.json").read_text(encoding="utf-8"))
    assert config["mesh"]["subdivisions"] == subdivisions
    assert parameters["shell"]["subdivisions"] == subdivisions
    assert metadata["configuration"]["shell"]["subdivisions"] == subdivisions
    assert metadata["fingerprint"] == fracture_metadata["fingerprint"]
    assert all(math.isclose(value, info.time_myr, rel_tol=0., abs_tol=1e-9) for value in
               [metadata["thermal"]["time_myr"], fracture_metadata["time_myr"], *info.report["clocks"].values()])
    assert all(info.report["checks"].values())
    return {"cells": cells, "time_myr": info.time_myr, "fingerprint": metadata["fingerprint"]}


def verify_execution_and_frames(info):
    timing = json.loads((info.root / "render_timings.json").read_text(encoding="utf-8"))
    execution = info.report["execution"]
    assert execution["cpu_workers"] == timing["cpu_workers"] == 4
    assert execution["render_workers"] == timing["render_workers"] == 2
    assert execution["frame_interval_myr"] == 1.
    assert execution["surface_only_frames"] and not execution["finalize"]
    assert timing["jobs_completed"] > 0, "Frame interval produced no actual render jobs"
    frames = list((info.root / "mature_run/hydrosphere_frames").glob("*.png"))
    assert frames, "Surface frames are absent despite the chosen frame interval"
    return {"cpu_workers": 4, "render_workers": 2, "render_jobs": timing["jobs_completed"],
            "render_pids": sorted({job["pid"] for job in timing["jobs"]}),
            "frames": {str(path.relative_to(info.root)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in frames}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/genesis_runs" /
                        f"gui_unified_continuation_validation_{datetime.now():%Y%m%d_%H%M%S_%f}")
    output = parser.parse_args().output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    app = QApplication.instance() or QApplication([])
    window = MoonWindow()
    window.resize(1560, 1000)
    window.show()
    errors = []
    logs = []
    commands = []
    window.controller.log_line.connect(logs.append)
    window.controller.process.started.connect(lambda: commands.append(window.controller.process.arguments()))
    original_warning, original_critical = QMessageBox.warning, QMessageBox.critical
    QMessageBox.warning = lambda *args: errors.append(str(args[2]))
    QMessageBox.critical = lambda *args: errors.append(str(args[2]))
    heartbeat = [0]
    timer = QTimer()
    timer.timeout.connect(lambda: heartbeat.__setitem__(0, heartbeat[0]+1))
    timer.start(10)
    began = perf_counter()
    original = starter_ui.GenesisStarterDialog

    class AutomaticStarter(original):
        def __init__(self, parent):
            super().__init__(parent)
            self.subdivisions.setCurrentIndex(self.subdivisions.findData(2))
            self.duration.setValue(1.)
            self.convective_traction.setValue(.05)
            self.continue_mature.setChecked(True)
            self.continuation_duration.setValue(1.)
            self.continuation_step.setValue(1.)
            self.watchdog = QTimer(self)
            self.watchdog.setSingleShot(True)
            self.watchdog.timeout.connect(self.close)
            self.watchdog.start(180000)
            QTimer.singleShot(0, self.start)

        def arguments(self, _requested_output):
            self.output_path = output / "starter"
            return super().arguments(self.output_path)

    def wait_for(states):
        loop = QEventLoop()
        watchdog = QTimer()
        watchdog.setSingleShot(True)
        watchdog.timeout.connect(loop.quit)
        def observe(state):
            if state in states or state == "Error":
                loop.quit()
        window.controller.state_changed.connect(observe)
        watchdog.start(180000)
        if window.controller.state not in states:
            loop.exec()
        window.controller.state_changed.disconnect(observe)
        watchdog.stop()
        assert window.controller.state in states, (window.controller.state, errors)

    try:
        print("Running a real 320-cell starter and automatic main-window adoption", flush=True)
        starter_ui.GenesisStarterDialog = AutomaticStarter
        window._open_genesis_starter()
        starter_ui.GenesisStarterDialog = original
        assert not errors, errors
        source = window.genesis_continuation
        assert source is not None, "Successful starter did not return to the main window"
        assert window.controller.state == "Completed"
        assert window.preview_view.currentData() == "surface"
        assert window.current_artifact == preferred_preview(source.root, view="surface")
        assert window.current_artifact is not None
        assert window.start_button.text() == "Продолжить расчёт"
        assert window.end_time.isEnabled() and window.dt.isEnabled()
        assert window.subdivisions.isEnabled()
        assert window.cpu_workers.isEnabled() and window.render_workers.isEnabled()
        source_details = verify_pair(source, 2)
        before = {name: hashlib.sha256((source.root/name).read_bytes()).hexdigest()
                  for name in source.report["checkpoint_sha256"]}
        window.end_time.setValue(2.)
        window.dt.setValue(1.)
        window.checkpoint_interval.setValue(1.)
        window.frame_interval.setValue(1.)
        window.subdivisions.setCurrentText("3")
        window.cpu_workers.setCurrentText("4")
        window.render_workers.setCurrentText("2")
        window.surface_only.setChecked(True)
        window.finalize.setChecked(False)
        make_spec = window._make_spec
        window._make_spec = lambda: replace(make_spec(), output_dir=output / "main_run")
        print("Refining to 1280 cells in the first main-window segment, CPU=4/render=2", flush=True)
        window._start_run()
        assert window.controller.state == "Running", errors
        window.controller.request_pause()
        wait_for({"Paused"})
        paused = window.genesis_continuation
        assert paused.time_myr == source.time_myr+1.
        assert not window.end_time.isEnabled()
        assert not window.subdivisions.isEnabled() and not window.cpu_workers.isEnabled()
        paused_details = verify_pair(paused, 3)
        paused_execution = verify_execution_and_frames(paused)
        assert len(paused.report["mesh_history"]) == 1
        mesh_event = paused.report["mesh_history"][0]
        assert mesh_event["source_cell_count"] == 320 and mesh_event["target_cell_count"] == 1280
        assert mesh_event["source_subdivisions"] == 2 and mesh_event["target_subdivisions"] == 3
        assert len(commands) == 1 and "--subdivisions" in commands[0]
        assert commands[0][commands[0].index("--subdivisions")+1] == "3"
        print("Safe pause reached; resuming the existing refined pair without another remesh", flush=True)
        window.controller.resume()
        wait_for({"Completed"})
        final = read_genesis_continuation(window.genesis_continuation.root)
        assert final.time_myr == source.time_myr+2.
        assert final.origin_time_myr == source.origin_time_myr
        final_details = verify_pair(final, 3)
        final_execution = verify_execution_and_frames(final)
        assert final.report["mesh_history"] == paused.report["mesh_history"]
        assert len(commands) == 2 and "--subdivisions" not in commands[1]
        assert commands[1][commands[1].index("--resume")+1] == str(paused.root)
        assert final.report["history"][:len(source.report["history"])] == source.report["history"]
        assert final.report["history"][:len(paused.report["history"])] == paused.report["history"]
        assert all(hashlib.sha256((paused.root/name).read_bytes()).hexdigest() == digest
                   for name, digest in paused_execution["frames"].items())
        assert final.plate_count == final.report["history"][-1]["plate_count"]
        assert window.preview_view.currentData() == "surface"
        assert window.current_artifact == preferred_preview(final.root, view="surface")
        assert window.current_artifact is not None
        assert window.end_time.isEnabled() and window.dt.isEnabled()
        assert window.subdivisions.isEnabled() and window.subdivisions.currentText() == "3"
        assert window.cpu_workers.isEnabled() and window.render_workers.isEnabled()
        assert not errors, errors
        unchanged = all(hashlib.sha256((source.root/name).read_bytes()).hexdigest() == digest
                        for name, digest in before.items())
        assert unchanged
        app.processEvents()
        assert window.grab().save(str(output / "main_after_continuation.png"))
        result = {"checks": {"automatic_main_handoff": True, "paired_main_resume": True,
            "safe_pause_and_resume": True, "same_origin": True, "source_unchanged": unchanged,
            "editable_duration_step_and_mesh": True, "normal_cpu_and_render_controls": True,
            "actual_cell_counts_match_young_and_mature_state": True, "all_clocks_and_configurations_agree": True,
            "refinement_only_in_first_segment": True, "history_preserved": True,
            "real_rendered_frames": True, "previous_frames_unchanged": True,
            "gui_heartbeat": heartbeat[0] > 100},
            "heartbeat_ticks": heartbeat[0], "wall_seconds": perf_counter()-began,
            "source": str(source.root), "source_time_myr": source.time_myr,
            "paused": str(paused.root), "paused_time_myr": paused.time_myr,
            "final": str(final.root), "final_time_myr": final.time_myr,
            "source_details": source_details, "paused_details": paused_details, "final_details": final_details,
            "paused_execution": paused_execution, "final_execution": final_execution,
            "mesh_history": final.report["mesh_history"], "commands": commands,
            "continuation_checks": final.report["checks"], "errors": errors}
        assert all(result["checks"].values())
        (output / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        (output / "validation_failed.json").write_text(json.dumps({"error": repr(exc), "errors": errors,
            "wall_seconds": perf_counter()-began, "state": window.controller.state}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
    finally:
        (output / "gui.log").write_text("\n".join(logs), encoding="utf-8")
        starter_ui.GenesisStarterDialog = original
        QMessageBox.warning, QMessageBox.critical = original_warning, original_critical
        timer.stop()
        if window.controller.is_active():
            window.controller.cancel_requested = True
            window.controller.process.kill()
            window.controller.process.waitForFinished(3000)
        window.close()
        app.processEvents()


if __name__ == "__main__":
    main()
