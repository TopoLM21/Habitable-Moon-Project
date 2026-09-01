#!/usr/bin/env python3
"""Exercise the real Qt QProcess controller without opening a window."""

from __future__ import annotations

import argparse
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
        )
    )
    app.exec()
    return int(result["code"])


if __name__ == "__main__":
    raise SystemExit(main())
