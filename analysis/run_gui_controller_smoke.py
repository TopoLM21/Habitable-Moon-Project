#!/usr/bin/env python3
"""Exercise the real Qt QProcess controller without opening a window."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QCoreApplication, QTimer

from moon_gui.app import SimulationController
from moon_gui.backend import RunSpec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--end-time", type=float, default=4.0)
    parser.add_argument("--subdivisions", type=int, default=3)
    parser.add_argument("--dt", type=float, default=4.0)
    parser.add_argument("--checkpoint-interval", type=float, default=4.0)
    parser.add_argument("--frame-interval", type=float, default=4.0)
    parser.add_argument("--full-frames", action="store_true")
    parser.add_argument("--no-finalize", action="store_true")
    parser.add_argument("--cpu-optimized", action="store_true")
    parser.add_argument("--cpu-workers", type=int, default=1)
    parser.add_argument("--render-workers", type=int, default=1)
    parser.add_argument("--cell-kernels", action="store_true")
    parser.add_argument("--pause-once", action="store_true", help="Pause after the first checkpoint, then resume automatically")
    parser.add_argument("--stop-on-second-segment-frame", action="store_true")
    parser.add_argument(
        "--resume",
        help="Resume from a completed checkpoint inside the selected output folder.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=300.0,
        help="Controller watchdog; use 0 to disable it for long production runs.",
    )
    args = parser.parse_args()

    root = ROOT
    output = Path(args.output).resolve()
    resume = Path(args.resume).resolve() if args.resume else None
    if resume is None and output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty smoke output: {output}")

    app = QCoreApplication([])
    controller = SimulationController()
    result = {"code": 2}
    controller.log_line.connect(print)
    saved_checkpoint = {}
    if args.stop_on_second_segment_frame:
        def record_checkpoint(time_myr, path):
            folder = Path(path)
            saved_checkpoint.update(time=time_myr, path=path, hashes={
                name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                for name in ("meta.json", "state.npz")})
        def stop_on_frame(line):
            if controller.target_index == 1 and controller.state == "Running" and "Surface frame:" in line:
                controller.stop_now()
        def stopped(state):
            if state != "Stopped":
                return
            folder = Path(saved_checkpoint["path"])
            hashes = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                      for name in ("meta.json", "state.npz")}
            valid = hashes == saved_checkpoint["hashes"] and controller.current_time == saved_checkpoint["time"]
            (output / "stop_validation.json").write_text(json.dumps({
                "last_completed_checkpoint_unchanged": valid, "time_myr": controller.current_time}, indent=2))
            print("GUI STOP CHECK:", valid)
            result["code"] = 0 if valid else 1
            app.quit()
        controller.segment_completed.connect(record_checkpoint)
        controller.log_line.connect(stop_on_frame)
        controller.state_changed.connect(stopped)

    def complete(path: str) -> None:
        print("GUI CONTROLLER COMPLETE:", path)
        result["code"] = 0
        app.quit()

    def failed(message: str) -> None:
        print("GUI CONTROLLER FAILED:", message)
        result["code"] = 1
        app.quit()

    controller.run_completed.connect(complete)
    controller.run_failed.connect(failed)
    if args.pause_once:
        pause_done = {"value": False}
        def checkpoint_done(_time: float, _path: str) -> None:
            if not pause_done["value"]:
                pause_done["value"] = True
                controller.request_pause()
        def state_changed(state: str) -> None:
            if state == "Paused":
                print("GUI CONTROLLER SAFE PAUSE VERIFIED")
                QTimer.singleShot(100, controller.resume)
        controller.segment_completed.connect(checkpoint_done)
        controller.state_changed.connect(state_changed)
    if args.timeout_seconds > 0:
        def timeout() -> None:
            print("GUI CONTROLLER TIMEOUT; stopping the active segment safely.")
            controller.stop_now()

        QTimer.singleShot(round(args.timeout_seconds * 1000), timeout)
    controller.start(
        RunSpec(
            project_root=root,
            source_config=root / "configs" / "canonical_moon.yaml",
            output_dir=output,
            subdivisions=int(args.subdivisions),
            end_time_myr=float(args.end_time),
            dt_myr=float(args.dt),
            checkpoint_interval_myr=float(args.checkpoint_interval),
            frame_interval_myr=float(args.frame_interval),
            surface_only_frames=not bool(args.full_frames),
            finalize=not bool(args.no_finalize),
            resume_checkpoint=resume,
            cpu_optimized=bool(args.cpu_optimized),
            cpu_workers=int(args.cpu_workers),
            render_workers=int(args.render_workers),
            cell_kernels=bool(args.cell_kernels),
        )
    )
    app.exec()
    return int(result["code"])


if __name__ == "__main__":
    raise SystemExit(main())
