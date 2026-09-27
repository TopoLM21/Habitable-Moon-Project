"""CPU/render policies do not replace the coupled young-world state."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from test_genesis_starter_continuation_cli import cli

ROOT = Path(__file__).resolve().parents[1]


def test_cli_forwards_execution_and_output_controls(cli, tmp_path):
    module, calls, _ = cli
    assert module.main(["--resume", "saved", "--output", str(tmp_path / "next"),
        "--cpu-workers", "4", "--render-workers", "2", "--process-priority", "below_normal",
        "--cell-kernels", "--cell-workers", "2", "--no-assignment-optimized", "--assignment-columns",
        "--boundary-forces", "--frame-interval", "2", "--surface-only-frames", "--finalize",
        "--subdivisions", "4"]) == 0
    options = calls[0][2]
    assert options["cpu_workers"] == 4 and options["render_workers"] == 2
    assert options["process_priority"] == "below_normal"
    assert options["cell_kernels"] and options["cell_workers"] == 2
    assert not options["assignment_optimized"] and options["assignment_columns"]
    assert options["boundary_forces"] and options["surface_only_frames"] and options["finalize"]
    assert options["frame_interval_myr"] == 2. and options["subdivisions"] == 4


@pytest.mark.parametrize("flag, value", [("--cpu-workers", "0"), ("--render-workers", "3"),
                                         ("--process-priority", "high"), ("--cell-workers", "3")])
def test_cli_rejects_invalid_execution_without_starting_core(cli, tmp_path, flag, value):
    module, calls, _ = cli
    with pytest.raises(SystemExit):
        module.main(["--resume", "saved", "--output", str(tmp_path / "next"), flag, value])
    assert not calls


def _run(arguments, output):
    environment = dict(os.environ, PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    completed = subprocess.run([sys.executable, *map(str, arguments)], cwd=ROOT, env=environment,
                               capture_output=True, text=True, encoding="utf-8", timeout=180)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    assert completed.returncode == 0, completed.stdout[-2500:] + completed.stderr[-2500:]


def _assert_same_saved_state(left, right):
    for relative in ("mature_checkpoint/state.npz", "young_context/starter_checkpoint.npz",
                     "young_context/fracture_memory.npz"):
        with np.load(left / relative, allow_pickle=False) as a, np.load(right / relative, allow_pickle=False) as b:
            assert a.files == b.files
            for name in a.files:
                np.testing.assert_array_equal(a[name], b[name], err_msg=f"{relative}:{name}")
    a = json.loads((left / "mature_checkpoint/meta.json").read_text(encoding="utf-8"))
    b = json.loads((right / "mature_checkpoint/meta.json").read_text(encoding="utf-8"))
    assert a == b


def test_real_endpoint_maps_without_interval_and_surface_only_resume(tmp_path):
    """A paired continuation must publish exact-age maps even without a movie cadence."""
    from PIL import Image

    root = Path(os.environ.get("GENESIS_MAP_E2E_OUTPUT", str(tmp_path / "endpoint_maps")))
    starter = root / "starter"
    _run([ROOT / "run_genesis_starter.py", "--output", starter, "--duration-myr", "2",
          "--subdivisions", "2", "--convective-traction-mpa", ".05"], starter)
    script = ROOT / "run_genesis_starter_continuation.py"
    source = starter / "starter_checkpoint.npz"
    plain, periodic, resumed = (root / name for name in ("no_interval", "periodic", "surface_only_resume"))
    common = ["--step-myr", "1", "--cpu-workers", "1"]
    _run([script, "--starter-checkpoint", source, "--output", plain, "--duration-myr", "1", *common], plain)
    _run([script, "--starter-checkpoint", source, "--output", periodic, "--duration-myr", "1",
          "--frame-interval", "1", *common], periodic)
    _run([script, "--resume", plain, "--output", resumed, "--duration-myr", "2",
          "--surface-only-frames", *common], resumed)
    _assert_same_saved_state(plain, periodic)

    cases = {}
    for path, surface_only in ((plain, False), (periodic, False), (resumed, True)):
        report = json.loads((path / "continuation.json").read_text(encoding="utf-8"))
        assert report["status"] == "completed" and all(report["checks"].values())
        assert report["material_ledger"]["balanced"]
        token = f"{report['final_time_myr']:013.4f}"
        surface = path / "mature_run" / "hydrosphere_frames" / f"surface_{token}_Myr.png"
        plate = path / "mature_run" / "plate_frames" / f"plate_{token}_Myr.png"
        assert list(surface.parent.glob("surface_*_Myr.png")) == [surface]
        expected_plates = [] if surface_only else [plate]
        assert list(plate.parent.glob("plate_*_Myr.png")) == expected_plates
        for image in [surface, *expected_plates]:
            with Image.open(image) as rendered:
                assert rendered.format == "PNG" and min(rendered.size) > 100
                rendered.verify()
        assert float(token) == pytest.approx(report["final_time_myr"], abs=5.1e-5)
        # The fractional Genesis age must not be rounded down to the ordinary
        # mature runner's historical 0.1 Myr filename precision.
        assert abs(float(token) - round(report["final_time_myr"], 1)) > 1e-3
        cases[path.name] = {"final_time_myr": report["final_time_myr"],
            "surface": str(surface), "plate": None if surface_only else str(plate),
            "checks": report["checks"], "material_balanced": True}
    assert cases["surface_only_resume"]["final_time_myr"] > cases["no_interval"]["final_time_myr"]
    (root / "validation.json").write_text(json.dumps({
        "status": "passed", "cells": 320, "cases": cases,
        "endpoint_without_interval": True, "four_decimal_age_filenames": True,
        "one_endpoint_when_also_periodic": True, "surface_only_resume_respected": True,
        "no_interval_vs_periodic_bitwise_equal": True,
    }, indent=2), encoding="utf-8")


def test_real_parallel_workers_and_rendering_preserve_state_and_resume(tmp_path):
    """Exercise real pools in separate CLI processes, including a changed resume policy."""
    root = Path(os.environ.get("GENESIS_EXECUTION_E2E_OUTPUT", str(tmp_path / "execution")))
    starter = root / "starter"
    _run([ROOT / "run_genesis_starter.py", "--output", starter, "--duration-myr", "2",
          "--subdivisions", "3", "--convective-traction-mpa", ".05"], starter)
    script = ROOT / "run_genesis_starter_continuation.py"
    source = starter / "starter_checkpoint.npz"
    common = ["--step-myr", "1", "--frame-interval", "1", "--surface-only-frames"]
    one, many, first, resumed = (root / name for name in ("one", "many", "first", "resumed"))
    _run([script, "--starter-checkpoint", source, "--output", one, "--duration-myr", "3",
          "--cpu-workers", "1", *common], one)
    parallel = ["--cpu-workers", "4", "--render-workers", "2", "--process-priority", "below_normal"]
    _run([script, "--starter-checkpoint", source, "--output", many, "--duration-myr", "3",
          *parallel, *common], many)
    _run([script, "--starter-checkpoint", source, "--output", first, "--duration-myr", "1",
          "--cpu-workers", "1", *common], first)
    _run([script, "--resume", first, "--output", resumed, "--duration-myr", "3",
          *parallel, *common], resumed)
    kernels = root / "kernels"
    _run([script, "--starter-checkpoint", source, "--output", kernels, "--duration-myr", "3",
          *parallel, *common, "--cell-kernels", "--cell-workers", "2", "--boundary-forces",
          "--assignment-columns", "--no-assignment-optimized"], kernels)
    _assert_same_saved_state(one, many)
    _assert_same_saved_state(one, resumed)
    _assert_same_saved_state(one, kernels)
    reports = [json.loads((path / "continuation.json").read_text(encoding="utf-8"))
               for path in (one, many, first, resumed, kernels)]
    for report in reports:
        assert report["status"] == "completed" and all(report["checks"].values())
        assert report["material_ledger"]["balanced"]
    for path, count in ((one,3), (many,3), (first,1), (resumed,2), (kernels,3)):
        assert len(list((path / "mature_run" / "hydrosphere_frames").glob("surface_*_Myr.png"))) == count
    assert reports[0]["history"] == reports[1]["history"] == reports[3]["history"]
    timing = json.loads((many / "render_timings.json").read_text(encoding="utf-8"))
    assert timing["cpu_workers"] == 4 and timing["render_workers"] == 2
    assert timing["jobs_completed"] > 0
    render_processes = len({job["pid"] for job in timing["jobs"]})
    assert 1 <= render_processes <= 2
    assert timing["numerical_execution"]["arc_query_workers"] == 4
    assert reports[1]["execution"]["process_priority"] == "below_normal"
    accelerated = json.loads((kernels / "render_timings.json").read_text(encoding="utf-8"))["numerical_execution"]
    assert accelerated["boundary_forces"]["calls"] > 0
    assert accelerated["cell_workers"] == 2
    assert accelerated["assignment_columns"]["enabled"]
    assert 1 <= len(accelerated["cell_thread_ids"]) <= 2
    from execution_policy import is_lower_priority
    assert is_lower_priority(timing["coordinator_priority"])
    assert all(is_lower_priority(job["priority"]) for job in timing["jobs"])
    (root / "validation.json").write_text(json.dumps({
        "status": "passed", "cells": 1280, "duration_myr": 3.,
        "one_vs_four_cpu_bitwise_equal": True, "changed_workers_resume_bitwise_equal": True,
        "optional_cpu_kernels_bitwise_equal": True, "all_balances": True,
        "render_jobs_completed": timing["jobs_completed"], "render_processes": render_processes,
        "boundary_force_calls": accelerated["boundary_forces"]["calls"],
        "cell_threads": len(accelerated["cell_thread_ids"]),
        "lower_priority_verified": True,
    }, indent=2), encoding="utf-8")
