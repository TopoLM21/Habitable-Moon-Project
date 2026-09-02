from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from moon_gui.backend import (
    RunSpec,
    build_segment_command,
    cell_count,
    checkpoint_cell_count,
    checkpoint_name,
    discover_artifacts,
    load_run_metrics,
    read_checkpoint_time,
    segment_targets,
    subdivision_for_cell_count,
    write_runtime_config,
    write_run_record,
)


def _project(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "project"
    root.mkdir()
    (root / "run_long_evolution_v131.py").write_text("pass\n", encoding="utf-8")
    (root / "run_long_evolution_v131_cpu.py").write_text("pass\n", encoding="utf-8")
    config = root / "canonical.yaml"
    config.write_text("mesh:\n  subdivisions: 5\nmoon:\n  name: Test\n", encoding="utf-8")
    return root, config


def test_icosphere_resolution_contract() -> None:
    assert cell_count(3) == 1_280
    assert cell_count(4) == 5_120
    assert cell_count(5) == 20_480
    assert cell_count(6) == 81_920
    assert subdivision_for_cell_count(20_480) == 5
    with pytest.raises(ValueError):
        subdivision_for_cell_count(123)


def test_segment_targets_include_short_aligned_final_segment() -> None:
    assert segment_targets(0.0, 52.0, 20.0, 4.0) == [20.0, 40.0, 52.0]
    assert segment_targets(40.0, 80.0, 20.0, 4.0) == [60.0, 80.0]
    with pytest.raises(ValueError):
        segment_targets(0.0, 51.0, 20.0, 4.0)


def test_runtime_config_overrides_only_mesh_and_adds_gui_metadata(tmp_path: Path) -> None:
    root, config = _project(tmp_path)
    spec = RunSpec(root, config, root / "out", subdivisions=4)
    runtime = write_runtime_config(spec)
    data = yaml.safe_load(runtime.read_text(encoding="utf-8"))
    assert data["mesh"]["subdivisions"] == 4
    assert data["moon"]["name"] == "Test"
    assert data["gui"]["runner"] == "v0.31-flow-coupled-plumes"


def test_runtime_config_absolutizes_external_eccentricity_history(tmp_path: Path) -> None:
    root, config = _project(tmp_path)
    config.write_text(
        "mesh: {subdivisions: 5}\ntides:\n  eccentricity_history_csv: data/e.csv\n",
        encoding="utf-8",
    )
    spec = RunSpec(root, config, root / "out")
    runtime = write_runtime_config(spec)
    data = yaml.safe_load(runtime.read_text(encoding="utf-8"))
    assert Path(data["tides"]["eccentricity_history_csv"]).is_absolute()


def test_command_uses_resume_and_finalization_only_when_requested(tmp_path: Path) -> None:
    root, config = _project(tmp_path)
    spec = RunSpec(
        root,
        config,
        root / "out",
        subdivisions=3,
        surface_only_frames=True,
        finalize=True,
    ).normalized()
    spec.output_dir.mkdir()
    spec.runtime_config.write_text("mesh: {subdivisions: 3}\n", encoding="utf-8")
    checkpoint = root / "prior"
    command = build_segment_command(
        spec,
        target_time_myr=40.0,
        checkpoint_dir=spec.output_dir / checkpoint_name(40.0),
        resume_checkpoint=checkpoint,
        final_segment=True,
    )
    assert command[0].endswith("run_long_evolution_v131.py")
    assert "--resume" in command
    assert "--surface-only-frames" in command
    assert "--finalize" in command


def test_checkpoint_introspection_and_spec_validation(tmp_path: Path) -> None:
    root, config = _project(tmp_path)
    checkpoint = root / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "meta.json").write_text(
        json.dumps({"format": "moon_tectonics_checkpoint", "time_myr": 40.0}),
        encoding="utf-8",
    )
    np.savez_compressed(checkpoint / "state.npz", state_cell_plate=np.zeros(1_280, dtype=np.int32))
    assert read_checkpoint_time(checkpoint) == 40.0
    assert checkpoint_cell_count(checkpoint) == 1_280
    spec = RunSpec(
        root,
        config,
        root / "out",
        subdivisions=3,
        end_time_myr=80.0,
        resume_checkpoint=checkpoint,
    ).normalized()
    spec.validate()


def test_artifact_discovery_and_checkpoint_metrics(tmp_path: Path) -> None:
    output = tmp_path / "out"
    frame = output / "hydrosphere_frames" / "surface_0000.png"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"png")
    gif = output / "history.gif"
    gif.write_bytes(b"gif")
    assert set(discover_artifacts(output)) == {frame, gif}

    checkpoint = output / checkpoint_name(20.0)
    checkpoint.mkdir()
    (checkpoint / "meta.json").write_text(
        json.dumps(
            {
                "format": "moon_tectonics_checkpoint",
                "version": "0.31-flow-coupled-plumes",
                "time_myr": 20.0,
                "system_plates": [{}, {}, {}],
                "thermal": {"mantle_temperature_k": 1800.0, "tectonic_activity_factor": 0.8},
                "hydrosphere_rows": [{"sea_level_m": 12.0, "land_area_fraction": 0.3}],
                "events": [{"kind": "split"}],
            }
        ),
        encoding="utf-8",
    )
    metrics = load_run_metrics(output)
    assert metrics["time_myr"] == 20.0
    assert metrics["plate_count"] == 3
    assert metrics["topology_events"] == 1


def test_cpu_mode_is_explicit_and_cannot_write_to_stable_outputs(tmp_path):
    root, config = _project(tmp_path)
    spec = RunSpec(root, config, root / "results" / "experiment", cpu_optimized=True, cpu_workers=4).normalized()
    spec.validate()
    command = build_segment_command(spec, target_time_myr=20.0, checkpoint_dir=spec.output_dir / checkpoint_name(20.0),
                                    resume_checkpoint=None, final_segment=False)
    assert command[0].endswith("run_long_evolution_v131_cpu.py")
    assert command[command.index("--cpu-workers") + 1] == "4"
    with pytest.raises(ValueError, match="workspace"):
        RunSpec(root, config, tmp_path / "stable", cpu_optimized=True).validate()
    with pytest.raises(ValueError, match="workers"):
        RunSpec(root, config, root / "results" / "experiment", cpu_workers=0).validate()
    runtime = write_runtime_config(spec)
    record = write_run_record(spec, runtime)
    assert json.loads(record.read_text())["runner"] == "run_long_evolution_v131_cpu.py"
    with pytest.raises(ValueError, match="empty"):
        RunSpec(root, config, spec.output_dir, cpu_optimized=True, resume_checkpoint=tmp_path / "external_checkpoint").validate()


def test_render_worker_command_validation_and_provenance(tmp_path):
    root, config = _project(tmp_path)
    spec = RunSpec(root, config, root / "results" / "render", cpu_optimized=True, render_workers=4, cell_kernels=True).normalized()
    spec.validate()
    command = build_segment_command(spec, target_time_myr=20, checkpoint_dir=spec.output_dir / "checkpoint",
                                    resume_checkpoint=None, final_segment=False)
    assert command[command.index("--render-workers") + 1] == "4"
    assert "--cell-kernels" in command
    record = write_run_record(spec, write_runtime_config(spec))
    assert json.loads(record.read_text())["render_workers"] == 4
    assert json.loads(record.read_text())["cell_kernels"] is True
    with pytest.raises(ValueError, match="Render workers"):
        RunSpec(root, config, root / "results" / "bad", cpu_optimized=True, render_workers=3).validate()
    with pytest.raises(ValueError, match="experimental"):
        RunSpec(root, config, root / "out", render_workers=2).validate()
