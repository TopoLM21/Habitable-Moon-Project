from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest
import yaml

from moon_gui.backend import (
    GENESIS_CHECKPOINT_FILES,
    GENESIS_CONTINUATION_FORMAT,
    GENESIS_CONTINUATION_RUNNER_NAME,
    RunSpec,
    build_segment_command,
    cell_count,
    checkpoint_name,
    latest_genesis_continuation,
    load_run_metrics,
    preferred_preview,
    read_genesis_continuation,
    write_run_record,
    write_runtime_config,
)


def write_pair(root: Path, *, duration: float = 10., subdivisions: int = 3,
               origin: float = .9906219482421875) -> Path:
    """Small structurally complete GUI fixture, not a numerical model fixture."""
    mature = root / "mature_checkpoint"
    young = root / "young_context"
    mature.mkdir(parents=True)
    young.mkdir()
    time = origin + duration
    (mature / "meta.json").write_text(json.dumps({
        "format": "moon_tectonics_checkpoint", "time_myr": time,
        "system_plates": [{}, {}], "events": [{"kind": "split"}],
        "hydrosphere_rows": [{"sea_level_m": 20., "land_area_fraction": 0.}],
    }), encoding="utf-8")
    np.savez_compressed(mature / "state.npz", state_cell_plate=np.zeros(cell_count(subdivisions), dtype=int))
    np.savez_compressed(young / "starter_checkpoint.npz", metadata=np.array(json.dumps({
        "fingerprint": "test-model", "thermal": {"time_myr": time},
    })))
    np.savez_compressed(young / "fracture_memory.npz", metadata=np.array(json.dumps({
        "fingerprint": "test-model", "time_myr": time,
    })))
    (young / "parameters.json").write_text("{}", encoding="utf-8")
    (root / "mature_config.yaml").write_text(yaml.safe_dump({"mesh": {"subdivisions": subdivisions}}), encoding="utf-8")
    report = {
        "format": GENESIS_CONTINUATION_FORMAT, "status": "completed", "mature_engine_executed": True,
        "checks": {"clocks_agree": True, "material_ledger": True},
        "import": {"origin_time_myr": origin}, "duration_myr": duration,
        "final_time_myr": time, "step_myr": 1., "final_plate_count": 2,
        "history": [{"mantle_temperature_k": 1580.}], "ocean_fraction": .99,
        "final_mean_surface_speed_km_myr": .004,
        "checkpoint_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                              for name in GENESIS_CHECKPOINT_FILES},
    }
    (root / "continuation.json").write_text(json.dumps(report), encoding="utf-8")
    (root / "continuation.png").write_bytes(b"preview")
    return root


def update_report(root: Path, **updates) -> None:
    path = root / "continuation.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report.update(updates)
    path.write_text(json.dumps(report), encoding="utf-8")


@pytest.fixture
def paired_spec(tmp_path):
    project = tmp_path / "project"
    source = write_pair(project / "results" / "starter" / "continuation")
    (project / GENESIS_CONTINUATION_RUNNER_NAME).write_text("pass\n", encoding="utf-8")
    info = read_genesis_continuation(source)
    return RunSpec(project, info.config, project / "results" / "continued",
                   subdivisions=3, end_time_myr=info.time_myr + 20., dt_myr=1.,
                   genesis_continuation=source).normalized()


@pytest.mark.parametrize("relative", ["", "continuation.json", "mature_checkpoint"])
def test_paired_checkpoint_paths_resolve_without_dropping_young_state(paired_spec, relative):
    info = read_genesis_continuation(paired_spec.genesis_continuation / relative)
    assert info.root == paired_spec.genesis_continuation
    assert info.plate_count == 2
    assert info.subdivisions == 3
    assert info.time_myr == info.origin_time_myr + 10.


def test_paired_run_uses_saved_configuration_and_records_provenance(paired_spec):
    spec = paired_spec
    spec.validate()
    original = spec.source_config.read_bytes()
    assert spec.runner.name == GENESIS_CONTINUATION_RUNNER_NAME
    assert write_runtime_config(spec) == spec.source_config
    record = json.loads(write_run_record(spec, spec.runtime_config).read_text(encoding="utf-8"))
    assert record["genesis_continuation"] == str(spec.genesis_continuation)
    assert record["runner"] == GENESIS_CONTINUATION_RUNNER_NAME
    assert spec.source_config.read_bytes() == original
    assert not (spec.output_dir / "gui_runtime_config.yaml").exists()


def test_segment_cli_uses_elapsed_origin_time_and_previous_paired_segment(paired_spec):
    spec = paired_spec
    origin = read_genesis_continuation(spec.genesis_continuation).origin_time_myr
    prior = write_pair(spec.output_dir / checkpoint_name(origin + 20.), duration=20.)
    target = origin + 30.
    command = build_segment_command(spec, target_time_myr=target,
        checkpoint_dir=spec.output_dir / checkpoint_name(target), resume_checkpoint=prior, final_segment=True)
    assert command[command.index("--resume") + 1] == str(prior.resolve())
    assert float(command[command.index("--duration-myr") + 1]) == 30.
    assert float(command[command.index("--step-myr") + 1]) == 1.
    assert "--config" not in command
    assert "--finalize" in command
    assert "--checkpoint" not in command


@pytest.mark.parametrize("relative", GENESIS_CHECKPOINT_FILES)
def test_every_linked_file_is_integrity_checked(paired_spec, relative):
    path = paired_spec.genesis_continuation / relative
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="integrity"):
        paired_spec.validate()


@pytest.mark.parametrize("update, match", [
    ({"format": "genesis-starter-continuation-0.1"}, "0.2"),
    ({"status": "validation_failed"}, "validation"),
    ({"checks": {"clocks_agree": False}}, "validation"),
    ({"checkpoint_sha256": {}}, "digest"),
    ({"final_time_myr": 99.}, "time"),
    ({"mature_engine_executed": False}, "execution"),
    ({"final_plate_count": float("nan")}, "plate"),
])
def test_incomplete_or_inconsistent_pairs_are_rejected(paired_spec, update, match):
    update_report(paired_spec.genesis_continuation, **update)
    with pytest.raises(ValueError, match=match):
        read_genesis_continuation(paired_spec.genesis_continuation)


@pytest.mark.parametrize("change, match", [
    ({"resume_checkpoint": Path("separate")}, "paired"),
    ({"subdivisions": 2}, "Coarsening"),
    ({"source_config": Path("different.yaml")}, "configuration"),
    ({"gpu_surface": True}, "GPU"),
    ({"cpu_workers": 0}, "workers"),
    ({"render_workers": 3}, "workers"),
    ({"process_priority": "high"}, "priority"),
    ({"dt_myr": float("nan")}, "finite"),
    ({"dt_myr": 3.}, "multiple"),
    ({"end_time_myr": 1.}, "after"),
])
def test_paired_spec_rejects_unsupported_or_inconsistent_changes(paired_spec, change, match):
    with pytest.raises(ValueError, match=match):
        replace(paired_spec, **change).validate()


def test_paired_grid_can_be_small_and_can_refine(paired_spec):
    source = write_pair(paired_spec.project_root / "results" / "small", subdivisions=2)
    info = read_genesis_continuation(source)
    spec = replace(paired_spec, genesis_continuation=source, source_config=info.config, subdivisions=2)
    spec.validate()
    assert cell_count(info.subdivisions) == 320
    refined = replace(spec, subdivisions=3)
    refined.validate()
    command = build_segment_command(refined, target_time_myr=refined.end_time_myr,
        checkpoint_dir=refined.output_dir / "next", resume_checkpoint=None, final_segment=False)
    assert command[command.index("--subdivisions") + 1] == "3"


def test_paired_execution_uses_selected_cpu_and_render_controls(paired_spec):
    spec = replace(paired_spec, cpu_optimized=True, cpu_workers=4, render_workers=2,
        process_priority="below_normal", cell_kernels=True, assignment_optimized=False,
        assignment_columns=True, boundary_forces=True, surface_only_frames=True, frame_interval_myr=2.)
    spec.validate()
    command = build_segment_command(spec, target_time_myr=spec.end_time_myr,
        checkpoint_dir=spec.output_dir / "next", resume_checkpoint=None, final_segment=False)
    for flag, value in (("--cpu-workers", "4"), ("--render-workers", "2"),
                        ("--process-priority", "below_normal"), ("--frame-interval", "2.0")):
        assert command[command.index(flag) + 1] == value
    for flag in ("--cell-kernels", "--no-assignment-optimized", "--assignment-columns",
                 "--boundary-forces", "--surface-only-frames"):
        assert flag in command
    assert "--finalize" not in command


def test_ordinary_resume_cannot_drop_young_context(paired_spec):
    spec = replace(paired_spec, genesis_continuation=None,
                   resume_checkpoint=paired_spec.genesis_continuation / "mature_checkpoint")
    with pytest.raises(ValueError, match="whole Genesis"):
        spec.validate()


@pytest.mark.parametrize("relative", [
    "initial_mature_checkpoint",
    "initial_import/mature_checkpoint",
    "mature_run/checkpoints/intermediate",
])
def test_ordinary_resume_cannot_bypass_pair_through_intermediate_snapshot(paired_spec, relative):
    source = paired_spec.genesis_continuation
    checkpoint = source / relative
    shutil.copytree(source / "mature_checkpoint", checkpoint)
    spec = replace(paired_spec, genesis_continuation=None, resume_checkpoint=checkpoint)
    with pytest.raises(ValueError, match="whole Genesis"):
        spec.validate()


def test_standalone_import_without_paired_marker_keeps_ordinary_resume(paired_spec):
    checkpoint = paired_spec.project_root / "results" / "standalone_import" / "mature_checkpoint"
    shutil.copytree(paired_spec.genesis_continuation / "mature_checkpoint", checkpoint)
    (paired_spec.project_root / "run_long_evolution_v131.py").write_text("pass\n", encoding="utf-8")
    spec = replace(paired_spec, genesis_continuation=None, resume_checkpoint=checkpoint)
    spec.validate()


def test_valid_continuation_can_have_one_remaining_plate(paired_spec):
    update_report(paired_spec.genesis_continuation, final_plate_count=1)
    assert read_genesis_continuation(paired_spec.genesis_continuation).plate_count == 1


def test_output_never_reuses_a_saved_or_nonempty_folder(paired_spec):
    with pytest.raises(ValueError, match="output folder"):
        replace(paired_spec, output_dir=paired_spec.genesis_continuation).validate()
    paired_spec.output_dir.mkdir()
    (paired_spec.output_dir / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        paired_spec.validate()


def test_discovery_uses_latest_complete_pair_and_ignores_incomplete_segment(paired_spec):
    output = paired_spec.output_dir
    older = write_pair(output / checkpoint_name(20.), duration=20.)
    newer = write_pair(output / checkpoint_name(30.), duration=30.)
    incomplete = output / checkpoint_name(40.)
    incomplete.mkdir()
    (incomplete / "continuation.png").write_bytes(b"partial")
    assert latest_genesis_continuation(output) == newer
    assert preferred_preview(output) == newer / "continuation.png"
    metrics = load_run_metrics(output)
    assert metrics["time_myr"] == read_genesis_continuation(newer).time_myr
    assert metrics["plate_count"] == 2
    assert metrics["mantle_temperature_k"] == 1580.
    assert metrics["sea_level_m"] == 20.
    assert metrics["topology_events"] == 1
    update_report(newer, status="validation_failed")
    assert latest_genesis_continuation(output) == older


def test_clock_agreement_is_checked_even_with_valid_digests(paired_spec):
    root = paired_spec.genesis_continuation
    archive = root / "young_context" / "starter_checkpoint.npz"
    np.savez_compressed(archive, metadata=np.array(json.dumps({
        "fingerprint": "test-model", "thermal": {"time_myr": 999.},
    })))
    report = json.loads((root / "continuation.json").read_text())
    report["checkpoint_sha256"]["young_context/starter_checkpoint.npz"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    update_report(root, checkpoint_sha256=report["checkpoint_sha256"])
    with pytest.raises(ValueError, match="disagree in time"):
        read_genesis_continuation(root)


@pytest.mark.parametrize("corruption", ["yaml", "truncated_zip", "empty_archive"])
def test_lightweight_discovery_ignores_corrupted_paired_data(paired_spec, corruption):
    output = paired_spec.output_dir
    older = write_pair(output / checkpoint_name(20.), duration=20.)
    broken = write_pair(output / checkpoint_name(30.), duration=30.)
    if corruption == "yaml":
        (broken / "mature_config.yaml").write_text("mesh: [unfinished", encoding="utf-8")
    else:
        archive = broken / "young_context" / "starter_checkpoint.npz"
        archive.write_bytes(archive.read_bytes()[:30] if corruption == "truncated_zip" else b"")
    with pytest.raises(ValueError, match="Invalid Genesis continuation metadata"):
        read_genesis_continuation(broken, verify_integrity=False)
    assert latest_genesis_continuation(output) == older
    assert preferred_preview(output) == older / "continuation.png"


def _preview_file(root, relative):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"preview")
    return path


def test_current_surface_and_plate_maps_are_distinct_from_genesis_diagnostics(paired_spec):
    root = paired_spec.genesis_continuation
    age = read_genesis_continuation(root).time_myr
    surface = _preview_file(root, f"mature_run/hydrosphere_frames/surface_{age:013.4f}_Myr.png")
    plates = _preview_file(root, f"mature_run/plate_frames/plate_{age:013.4f}_Myr.png")
    assert preferred_preview(root) == surface
    assert preferred_preview(root, view="surface") == surface
    assert preferred_preview(root, view="plates") == plates
    assert preferred_preview(root, view="genesis") == root / "continuation.png"


def test_map_age_wins_over_recently_modified_historical_frame(tmp_path):
    old = _preview_file(tmp_path, "hydrosphere_frames/surface_00000010.0000_Myr.png")
    new = _preview_file(tmp_path, "hydrosphere_frames/surface_00000020.0000_Myr.png")
    os.utime(old, (new.stat().st_mtime + 1000, new.stat().st_mtime + 1000))
    assert preferred_preview(tmp_path, view="surface") == new


def test_standalone_surface_prefers_elevation_relative_to_sea_level(tmp_path):
    sea_relative = _preview_file(tmp_path, "surface_relative_sea_level.png")
    datum = _preview_file(tmp_path, "elevation_final.png")
    os.utime(datum, (sea_relative.stat().st_mtime+1000, sea_relative.stat().st_mtime+1000))
    assert preferred_preview(tmp_path, view="surface") == sea_relative


def test_standalone_surface_retains_legacy_datum_fallback(tmp_path):
    datum = _preview_file(tmp_path, "elevation_final.png")
    assert preferred_preview(tmp_path, view="surface") == datum


def test_explicit_map_does_not_show_older_segment_or_current_diagnostics(paired_spec):
    output = paired_spec.output_dir
    older = write_pair(output / checkpoint_name(20.), duration=20.)
    newer = write_pair(output / checkpoint_name(30.), duration=30.)
    old_age = read_genesis_continuation(older).time_myr
    _preview_file(older, f"mature_run/hydrosphere_frames/surface_{old_age:013.4f}_Myr.png")
    _preview_file(newer, f"mature_run/hydrosphere_frames/surface_{old_age:013.4f}_Myr.png")
    assert preferred_preview(output, view="surface") is None
    assert preferred_preview(output, view="plates") is None
    assert preferred_preview(output) == newer / "continuation.png"
    assert preferred_preview(output, view="genesis") == newer / "continuation.png"


def test_endpoint_snapshot_beats_old_frame_despite_mtime(paired_spec):
    root = paired_spec.genesis_continuation
    age = read_genesis_continuation(root).time_myr
    mature = root / "mature_run"
    surface = _preview_file(mature, f"hydrosphere_frames/surface_{age:013.4f}_Myr.png")
    plates = _preview_file(mature, f"plate_frames/plate_{age:013.4f}_Myr.png")
    old = _preview_file(mature, f"hydrosphere_frames/surface_{age-1:013.4f}_Myr.png")
    os.utime(old, (surface.stat().st_mtime+1000, surface.stat().st_mtime+1000))
    assert preferred_preview(root) == surface
    assert preferred_preview(root, view="plates") == plates
