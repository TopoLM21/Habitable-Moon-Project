"""Pure-Python planning and result discovery used by the Qt application.

Keeping this module free of Qt makes the run contract easy to test on both
Windows and Linux.  The GUI never imports or mutates the numerical model in
its own process; it launches the production v0.31 runner in checkpoint-sized
subprocesses instead.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

import yaml


RUNNER_NAME = "run_long_evolution_v131.py"
RUNTIME_CONFIG_NAME = "gui_runtime_config.yaml"
RUN_RECORD_NAME = "gui_run.json"


def cell_count(subdivisions: int) -> int:
    """Return the number of triangular cells in the subdivided icosphere."""

    if subdivisions < 0:
        raise ValueError("subdivisions must be non-negative")
    return 20 * 4**int(subdivisions)


def resolution_note(subdivisions: int) -> str:
    cells = cell_count(subdivisions)
    relative = (cells / 20_480) ** 1.35
    return f"{cells:,} cells | estimated compute load {relative:.2f}x canonical sub-5"


def _is_multiple(value: float, step: float, tolerance: float = 1.0e-9) -> bool:
    if step <= 0:
        return False
    return abs(round(value / step) * step - value) <= tolerance


@dataclass(frozen=True, slots=True)
class RunSpec:
    project_root: Path
    source_config: Path
    output_dir: Path
    subdivisions: int = 5
    end_time_myr: float = 500.0
    dt_myr: float = 4.0
    checkpoint_interval_myr: float = 20.0
    frame_interval_myr: float = 20.0
    surface_only_frames: bool = False
    finalize: bool = True
    resume_checkpoint: Path | None = None

    def normalized(self) -> "RunSpec":
        return RunSpec(
            project_root=self.project_root.resolve(),
            source_config=self.source_config.resolve(),
            output_dir=self.output_dir.resolve(),
            subdivisions=int(self.subdivisions),
            end_time_myr=float(self.end_time_myr),
            dt_myr=float(self.dt_myr),
            checkpoint_interval_myr=float(self.checkpoint_interval_myr),
            frame_interval_myr=float(self.frame_interval_myr),
            surface_only_frames=bool(self.surface_only_frames),
            finalize=bool(self.finalize),
            resume_checkpoint=(
                None if self.resume_checkpoint is None else self.resume_checkpoint.resolve()
            ),
        )

    @property
    def runner(self) -> Path:
        return self.project_root / RUNNER_NAME

    @property
    def runtime_config(self) -> Path:
        return self.output_dir / RUNTIME_CONFIG_NAME

    def start_time_myr(self) -> float:
        if self.resume_checkpoint is None:
            return 0.0
        return read_checkpoint_time(self.resume_checkpoint)

    def validate(self) -> None:
        if not self.project_root.is_dir():
            raise ValueError(f"Project root does not exist: {self.project_root}")
        if not self.runner.is_file():
            raise ValueError(f"v0.31 runner does not exist: {self.runner}")
        if not self.source_config.is_file():
            raise ValueError(f"Configuration does not exist: {self.source_config}")
        if self.subdivisions not in {3, 4, 5, 6}:
            raise ValueError("GUI supports subdivisions 3, 4, 5, or 6")
        if self.dt_myr <= 0:
            raise ValueError("Time step must be positive")
        if self.checkpoint_interval_myr <= 0:
            raise ValueError("Checkpoint interval must be positive")
        if self.frame_interval_myr <= 0:
            raise ValueError("Frame interval must be positive")
        if not _is_multiple(self.checkpoint_interval_myr, self.dt_myr):
            raise ValueError("Checkpoint interval must be an integer multiple of dt")
        if not _is_multiple(self.frame_interval_myr, self.dt_myr):
            raise ValueError("Frame interval must be an integer multiple of dt")
        start = self.start_time_myr()
        if self.end_time_myr <= start:
            raise ValueError(
                f"End time ({self.end_time_myr:g}) must be after start time ({start:g})"
            )
        if not _is_multiple(self.end_time_myr - start, self.dt_myr):
            raise ValueError("Run duration must be an integer multiple of dt")
        if self.resume_checkpoint is not None:
            if not (self.resume_checkpoint / "meta.json").is_file():
                raise ValueError("Resume checkpoint has no meta.json")
            if not (self.resume_checkpoint / "state.npz").is_file():
                raise ValueError("Resume checkpoint has no state.npz")


def read_checkpoint_time(checkpoint: Path) -> float:
    with (Path(checkpoint) / "meta.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("format") != "moon_tectonics_checkpoint":
        raise ValueError(f"Not a Moon Tectonics checkpoint: {checkpoint}")
    return float(metadata["time_myr"])


def checkpoint_cell_count(checkpoint: Path) -> int:
    """Read the mesh size without loading every checkpoint array."""

    import numpy as np

    with np.load(Path(checkpoint) / "state.npz", allow_pickle=False) as archive:
        return int(archive["state_cell_plate"].shape[0])


def subdivision_for_cell_count(cells: int) -> int:
    for subdivisions in range(0, 10):
        if cell_count(subdivisions) == int(cells):
            return subdivisions
    raise ValueError(f"Unsupported icosphere cell count: {cells}")


def segment_targets(start: float, end: float, interval: float, dt: float) -> list[float]:
    if end <= start:
        return []
    if interval <= 0 or dt <= 0:
        raise ValueError("interval and dt must be positive")
    if not _is_multiple(end - start, dt):
        raise ValueError("end-start must align to dt")
    targets: list[float] = []
    current = float(start)
    while current < end - 1.0e-9:
        candidate = min(current + interval, end)
        # The last partial segment is valid only when it still aligns to dt.
        if not _is_multiple(candidate - current, dt):
            candidate = current + max(dt, int((candidate - current) / dt) * dt)
            candidate = min(candidate, end)
        if candidate <= current + 1.0e-9:
            raise ValueError("Unable to build a positive checkpoint segment")
        targets.append(float(candidate))
        current = candidate
    return targets


def checkpoint_name(time_myr: float) -> str:
    if abs(time_myr - round(time_myr)) < 1.0e-9:
        token = f"{int(round(time_myr)):06d}"
    else:
        token = f"{time_myr:012.4f}".replace(".", "p")
    return f"gui_checkpoint_{token}_Myr"


def write_runtime_config(spec: RunSpec) -> Path:
    """Copy the selected YAML and override only GUI-owned mesh settings."""

    spec = spec.normalized()
    with spec.source_config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    eccentricity_csv = config.get("tides", {}).get("eccentricity_history_csv")
    if eccentricity_csv:
        history_path = Path(str(eccentricity_csv))
        if not history_path.is_absolute():
            # Production configs live in <project>/configs and the numerical
            # runner historically resolves their data paths from <project>.
            history_path = (spec.source_config.parent.parent / history_path).resolve()
        config["tides"]["eccentricity_history_csv"] = str(history_path)
    config.setdefault("mesh", {})["subdivisions"] = int(spec.subdivisions)
    config.setdefault("gui", {}).update(
        {
            "runner": "v0.31-flow-coupled-plumes",
            "checkpoint_interval_myr": float(spec.checkpoint_interval_myr),
            "frame_interval_myr": float(spec.frame_interval_myr),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    with spec.runtime_config.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    return spec.runtime_config


def write_run_record(spec: RunSpec, runtime_config: Path) -> Path:
    payload = asdict(spec.normalized())
    payload = {key: (None if value is None else str(value)) if isinstance(value, Path) or value is None else value for key, value in payload.items()}
    payload.update(
        {
            "format": "moon_tectonics_gui_run",
            "runner": RUNNER_NAME,
            "runtime_config": str(runtime_config),
            "cell_count": cell_count(spec.subdivisions),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    path = spec.output_dir / RUN_RECORD_NAME
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def build_segment_command(
    spec: RunSpec,
    *,
    target_time_myr: float,
    checkpoint_dir: Path,
    resume_checkpoint: Path | None,
    final_segment: bool,
) -> list[str]:
    command = [
        str(spec.runner),
        "--config",
        str(spec.runtime_config),
        "--output",
        str(spec.output_dir),
        "--end-time",
        f"{target_time_myr:g}",
        "--dt",
        f"{spec.dt_myr:g}",
        "--checkpoint",
        str(checkpoint_dir),
        "--frame-interval",
        f"{spec.frame_interval_myr:g}",
    ]
    if resume_checkpoint is not None:
        command.extend(["--resume", str(resume_checkpoint)])
    if spec.surface_only_frames:
        command.append("--surface-only-frames")
    if final_segment and spec.finalize:
        command.append("--finalize")
    return command


def discover_artifacts(output_dir: Path, suffixes: Iterable[str] = (".png", ".gif")) -> list[Path]:
    root = Path(output_dir)
    if not root.exists():
        return []
    wanted = {suffix.lower() for suffix in suffixes}
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in wanted),
        key=lambda path: (path.stat().st_mtime, str(path)),
        reverse=True,
    )


def preferred_preview(output_dir: Path) -> Path | None:
    root = Path(output_dir)
    patterns = (
        "hotspot_track_frames/hotspot_tracks_*_Myr.png",
        "hydrosphere_frames/surface_*_Myr.png",
        "frames/frame_*_Myr.png",
        "*evolution.gif",
        "history.gif",
        "plate_map_final.png",
        "elevation_final.png",
    )
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(root.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_run_metrics(output_dir: Path) -> dict[str, Any]:
    """Return a compact, user-facing snapshot from summary or checkpoint data."""

    root = Path(output_dir)
    summary = root / "summary_v131.json"
    if summary.is_file():
        with summary.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        keys = (
            "duration_myr",
            "mesh_cells",
            "final_plate_count",
            "final_continental_area_fraction",
            "final_mantle_temperature_k",
            "final_tectonic_activity_factor",
            "final_sea_level_m",
            "topology_event_count",
            "final_mean_resolved_plume_flow_speed_km_per_myr",
            "final_mean_effective_plume_speed_km_per_myr",
        )
        return {key: data[key] for key in keys if key in data}

    checkpoints = list(
        (path for path in (root / "checkpoints").glob("*") if (path / "meta.json").is_file()),
    )
    # GUI checkpoints live beside the traditional checkpoints directory.
    checkpoints.extend(
        path for path in root.glob("gui_checkpoint_*_Myr") if (path / "meta.json").is_file()
    )
    if not checkpoints:
        return {}
    checkpoints.sort(key=lambda path: (path / "meta.json").stat().st_mtime, reverse=True)
    with (checkpoints[0] / "meta.json").open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    thermal = meta.get("thermal") or {}
    hydro_rows = meta.get("hydrosphere_rows") or []
    latest_hydro = hydro_rows[-1] if hydro_rows else {}
    return {
        "time_myr": meta.get("time_myr"),
        "checkpoint_version": meta.get("version"),
        "plate_count": len(meta.get("system_plates") or []),
        "mantle_temperature_k": thermal.get("mantle_temperature_k"),
        "tectonic_activity_factor": thermal.get("tectonic_activity_factor"),
        "sea_level_m": latest_hydro.get("sea_level_m"),
        "land_area_fraction": latest_hydro.get("land_area_fraction"),
        "topology_events": len(meta.get("events") or []),
    }
