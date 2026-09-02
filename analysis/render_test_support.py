"""Small spawn-importable jobs for render lifetime and snapshot tests."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from time import monotonic, sleep


def save_marker(values, path, delay=0.0, barrier=None):
    if barrier is not None:
        barrier = Path(barrier)
        (barrier / f"worker_{os.getpid()}").touch()
        deadline = monotonic() + 10
        while len(list(barrier.glob("worker_*"))) < 2:
            if monotonic() > deadline:
                raise TimeoutError("Second render process did not start")
            sleep(0.02)
    sleep(delay)
    Path(path).write_text(json.dumps({"values": values, "pid": os.getpid()}))


def save_invalid_image(path):
    from matplotlib.figure import Figure
    figure = Figure()
    figure.savefig(path, format="not-a-supported-format")


def save_long_job(path):
    Path(path).write_text(str(os.getpid()))
    sleep(30)
    Path(path).with_suffix(".unexpected").touch()


def orphan_owner(directory):
    from visualization.render_runtime import RenderExecution
    path = Path(directory) / "worker.pid"
    with RenderExecution(2) as rendering:
        rendering.submit(save_long_job, path)
        deadline = monotonic() + 15
        while not path.exists():
            if monotonic() > deadline:
                raise TimeoutError("Worker did not initialize")
            sleep(0.02)
        # Deliberately bypass context cleanup, like forced GUI termination.
        os._exit(0)


if __name__ == "__main__":
    # Import the named module so the worker target remains spawn-importable.
    from analysis.render_test_support import orphan_owner
    orphan_owner(sys.argv[1])
