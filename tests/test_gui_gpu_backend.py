"""GUI command/provenance contracts without importing Qt, CuPy or CUDA."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from moon_gui.backend import (
    CPU_RUNNER_NAME,
    GPU_RUNNER_NAME,
    RUNNER_NAME,
    RunSpec,
    build_segment_command,
    write_run_record,
    write_runtime_config,
)


@pytest.fixture
def spec(tmp_path: Path) -> RunSpec:
    root = tmp_path / "project"
    root.mkdir()
    for name in (RUNNER_NAME, CPU_RUNNER_NAME, GPU_RUNNER_NAME):
        (root / name).write_text("pass\n", encoding="utf-8")
    config = root / "canonical.yaml"
    config.write_text("mesh: {subdivisions: 5}\n", encoding="utf-8")
    return RunSpec(root, config, root / "results" / "experiment")


def command(spec: RunSpec, *, resume: Path | None = None) -> list[str]:
    return build_segment_command(
        spec,
        target_time_myr=20,
        checkpoint_dir=spec.output_dir / "checkpoint",
        resume_checkpoint=resume,
        final_segment=False,
    )


def test_default_mode_and_command_remain_non_gpu(spec: RunSpec) -> None:
    spec = spec.normalized()
    spec.validate()
    assert spec.runner.name == RUNNER_NAME
    assert (spec.gpu_surface, spec.gpu_device, spec.assignment_columns, spec.boundary_forces) == (
        False, 0, False, False,
    )
    args = command(spec)
    assert not any("gpu-" in arg or "assignment-columns" in arg or "boundary-forces" in arg for arg in args)


@pytest.mark.parametrize("gpu_surface,runner", [(False, CPU_RUNNER_NAME), (True, GPU_RUNNER_NAME)])
@pytest.mark.parametrize("assignment_columns,boundary_forces", [(False, False), (True, False), (False, True), (True, True)])
def test_optimized_flags_are_independent_and_explicit(
    spec: RunSpec, gpu_surface: bool, runner: str, assignment_columns: bool, boundary_forces: bool,
) -> None:
    spec = replace(
        spec, cpu_optimized=True, gpu_surface=gpu_surface, gpu_device=2,
        assignment_columns=assignment_columns, boundary_forces=boundary_forces,
    ).normalized()
    spec.validate()
    args = command(spec)
    assert spec.runner.name == runner
    assert args[0] == str(spec.project_root / runner)
    assert ("--assignment-columns" in args) is assignment_columns
    assert ("--no-assignment-columns" in args) is not assignment_columns
    assert ("--boundary-forces" in args) is boundary_forces
    assert ("--no-boundary-forces" in args) is not boundary_forces
    assert ("--gpu-surface" in args) is gpu_surface
    assert ("--gpu-device" in args) is gpu_surface
    assert "--gpu-arcs" not in args
    if gpu_surface:
        assert args[args.index("--gpu-device") + 1] == "2"


@pytest.mark.parametrize("gpu_device", [-1, 0.5, 2.0, "0", True, None])
def test_gpu_device_must_be_non_negative_integer(spec: RunSpec, gpu_device: object) -> None:
    invalid = replace(spec, cpu_optimized=True, gpu_surface=True, gpu_device=gpu_device)
    with pytest.raises(ValueError, match="non-negative integer"):
        invalid.validate()
    with pytest.raises(ValueError, match="non-negative integer"):
        invalid.normalized()


def test_gpu_requires_optimized_execution(spec: RunSpec) -> None:
    with pytest.raises(ValueError, match="GPU surface requires"):
        replace(spec, gpu_surface=True).validate()


@pytest.mark.parametrize("option", ["assignment_columns", "boundary_forces"])
def test_new_cpu_options_require_optimized_execution(spec: RunSpec, option: str) -> None:
    with pytest.raises(ValueError, match="optimized CPU or GPU"):
        replace(spec, **{option: True}).validate()


def test_gpu_preserves_shared_command_and_worker_settings(spec: RunSpec) -> None:
    optimized = replace(
        spec, cpu_optimized=True, cpu_workers=4, render_workers=8,
        cell_kernels=True, process_priority="below_normal", assignment_columns=True,
    )
    gpu = replace(optimized, gpu_surface=True, gpu_device=1)
    resume = spec.project_root / "prior"
    original_args = command(optimized, resume=resume)
    gpu_args = command(gpu, resume=resume)
    assert gpu_args[1:-3] == original_args[1:]
    assert gpu_args[-3:] == ["--gpu-surface", "--gpu-device", "1"]
    for option, value in (
        ("--config", str(spec.runtime_config)), ("--output", str(spec.output_dir)),
        ("--cpu-workers", "4"), ("--render-workers", "8"),
        ("--process-priority", "below_normal"), ("--resume", str(resume)),
    ):
        assert gpu_args[gpu_args.index(option) + 1] == value
    assert "--cell-kernels" in gpu_args


def test_gpu_reuses_experimental_output_isolation(spec: RunSpec, tmp_path: Path) -> None:
    gpu = replace(spec, cpu_optimized=True, gpu_surface=True)
    with pytest.raises(ValueError, match="workspace's results"):
        replace(gpu, output_dir=tmp_path / "outside").validate()
    gpu.output_dir.mkdir(parents=True)
    (gpu.output_dir / "existing.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="empty experimental"):
        replace(gpu, resume_checkpoint=tmp_path / "external").validate()
    assert (gpu.output_dir / "existing.txt").read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("assignment_columns,boundary_forces", [(True, False), (False, True)])
def test_run_record_stores_gpu_and_optimization_choices(
    spec: RunSpec, assignment_columns: bool, boundary_forces: bool,
) -> None:
    spec = replace(
        spec, cpu_optimized=True, gpu_surface=True, gpu_device=3,
        assignment_columns=assignment_columns, boundary_forces=boundary_forces,
    )
    record = json.loads(write_run_record(spec, write_runtime_config(spec)).read_text(encoding="utf-8"))
    assert record["runner"] == GPU_RUNNER_NAME
    assert record["cpu_optimized"] is True
    assert record["gpu_surface"] is True
    assert record["gpu_device"] == 3
    assert record["assignment_columns"] is assignment_columns
    assert record["boundary_forces"] is boundary_forces
