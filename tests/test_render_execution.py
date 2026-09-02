import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic, sleep

import pytest

from analysis.render_test_support import save_invalid_image, save_marker
from visualization.render_runtime import RenderExecution, flush_rendering


def test_render_context_is_opt_in_and_validates_limits(tmp_path):
    flush_rendering()
    with RenderExecution(1) as rendering:
        rendering.submit(save_marker, [3], tmp_path / "serial.json")
        assert (tmp_path / "serial.json").is_file()
        with pytest.raises(RuntimeError):
            with RenderExecution(2):
                pass
    for workers in (0, 3, 8, True, 1.5):
        with pytest.raises(ValueError):
            RenderExecution(workers)


def test_snapshot_is_captured_before_mutation_and_two_processes_work(tmp_path):
    values = [1, 2]
    with RenderExecution(2, max_pending=2) as rendering:
        rendering.submit(save_marker, values, tmp_path / "first.json", barrier=tmp_path)
        values[0] = 99
        rendering.submit(save_marker, values, tmp_path / "second.json", barrier=tmp_path)
        values.clear()
        flush_rendering()
        assert not rendering.pending
    first = json.loads((tmp_path / "first.json").read_text())
    second = json.loads((tmp_path / "second.json").read_text())
    assert first["values"] == [1, 2]
    assert second["values"] == [99, 2]
    assert first["pid"] != second["pid"]
    assert os.getpid() not in (first["pid"], second["pid"])
    assert rendering.peak_pending == 2


def test_duplicate_destination_retains_order_and_queue_is_bounded(tmp_path):
    target = tmp_path / "same.json"
    with RenderExecution(2, max_pending=1) as rendering:
        rendering.submit(save_marker, [1], target, delay=0.1)
        rendering.submit(save_marker, [2], target)
        rendering.submit(save_marker, [3], tmp_path / "other.json")
    assert json.loads(target.read_text())["values"] == [2]
    assert rendering.peak_pending == 1
    assert rendering.report()["jobs_completed"] == 3


def test_oversized_snapshot_rejected_before_submission(tmp_path):
    with RenderExecution(2, max_snapshot_bytes=16) as rendering:
        with pytest.raises(ValueError, match="memory budget"):
            rendering.submit(save_marker, list(range(100)), tmp_path / "large.json")
        assert not rendering.pending
    assert not (tmp_path / "large.json").exists()


def test_failed_render_preserves_existing_image_and_reports_error(tmp_path):
    target = tmp_path / "existing.png"
    target.write_bytes(b"previous complete image")
    with pytest.raises(RuntimeError, match="Render job failed"):
        with RenderExecution(2) as rendering:
            rendering.submit(save_invalid_image, target)
    assert target.read_bytes() == b"previous complete image"
    assert not list(tmp_path.glob(".moon-render-*"))
    with RenderExecution(1):
        pass


def test_parallel_image_matches_serial_bytes(tmp_path):
    from visualization.topology import save_plate_count_history
    rows = [{"time_myr": t, "plate_count_after": n, "split_events": 0,
             "merge_events": 0, "absorbed_small_plates": 0} for t, n in [(0, 4), (4, 5)]]
    reference = tmp_path / "serial.png"
    actual = tmp_path / "parallel.png"
    save_plate_count_history(rows, reference, dpi=50)
    with RenderExecution(2) as rendering:
        rendering.submit(save_plate_count_history, rows, actual, dpi=50)
    assert actual.read_bytes() == reference.read_bytes()


def test_hooks_preserve_value_returning_renderers_and_restore_aliases():
    import run_long_evolution_v131 as runner
    original = runner.base.save_plate_count_history
    value_returning = runner.base.save_volcanic_arc_maps
    with RenderExecution(2) as rendering:
        rendering.install_runner_hooks()
        assert runner.base.save_plate_count_history is not original
        assert runner.base.save_volcanic_arc_maps is value_returning
    assert runner.base.save_plate_count_history is original
    with pytest.raises(RuntimeError, match="single-use"):
        with rendering:
            pass


def _alive(pid):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        api = ctypes.WinDLL("kernel32", use_last_error=True)
        api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        api.OpenProcess.restype = wintypes.HANDLE
        api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        api.WaitForSingleObject.restype = wintypes.DWORD
        api.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = api.OpenProcess(0x00100000, False, pid)
        if not handle:
            return False
        try:
            return api.WaitForSingleObject(handle, 0) == 258
        finally:
            api.CloseHandle(handle)
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists() and stat.read_text().split(")", 1)[1].split()[0] == "Z":
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def test_forced_owner_exit_terminates_workers(tmp_path):
    root = Path(__file__).resolve().parent.parent
    env = dict(os.environ, PYTHONPATH=str(root), MPLBACKEND="Agg")
    with (tmp_path / "owner.log").open("w") as log:
        owner = subprocess.Popen([sys.executable, "-m", "analysis.render_test_support", str(tmp_path)],
                                 cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            assert owner.wait(timeout=25) == 0
        finally:
            if owner.poll() is None:
                owner.kill()
                owner.wait()
    pid = int((tmp_path / "worker.pid").read_text())
    deadline = monotonic() + 5
    while _alive(pid) and monotonic() < deadline:
        sleep(0.05)
    assert not _alive(pid)
    assert not (tmp_path / "worker.unexpected").exists()
