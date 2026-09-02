#!/usr/bin/env python3
"""Moon Tectonics v0.30 mobile plume sources and bent hotspot tracks."""

from __future__ import annotations
from visualization.render_runtime import flush_rendering

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import run_long_evolution_v129 as v129
from tectonics.plume_drift import (
    PlumeDriftParameters,
    diagnose_plume_drift,
    plume_parameters_with_drift,
    source_path_rows,
)
from visualization.hotspot_tracks import (
    build_hotspot_track_gif,
    save_hotspot_track_frame,
)
from visualization.plume_drift import (
    save_plume_drift_history,
    save_plume_source_paths,
)

v128 = v129.v128
v127 = v129.v127
v126 = v129.v126
v125 = v129.v125
v124 = v129.v124
base = v129.base

_params = PlumeDriftParameters()
_rows: list[dict] = []
_source_rows: list[dict] = []
_output = Path("outputs_v130_mobile_plumes")
_dpi = 180
_frame_version_label = "v0.30"


def _row(diagnostics) -> dict:
    return {
        name: getattr(diagnostics, name)
        for name in diagnostics.__dataclass_fields__
    }


def _parse_args_v130():
    global _params, _rows, _source_rows, _output, _dpi
    args = v129._parse_args_v129()
    config = base.load_config(args.config)
    _params = base.dc(PlumeDriftParameters, config.get("plume_drift", {}))
    v125._params = plume_parameters_with_drift(v125._params, _params)
    _rows = []
    _source_rows = []
    _output = Path(args.output)
    _dpi = int(config.get("output", {}).get("dpi", 180))
    return args


def _record_source_state(plate_system, *, force_path: bool = False) -> None:
    if v124._mesh is None or v125._plume_state is None:
        return
    diagnostics = diagnose_plume_drift(
        v124._mesh,
        v125._plume_state,
        plate_system,
        v124._radius_km,
        _params,
    )
    _rows.append(_row(diagnostics))
    interval = float(_params.path_sample_interval_myr)
    time = float(v125._plume_state.time_myr)
    on_interval = abs(time / interval - round(time / interval)) < 1.0e-9
    if force_path or on_interval:
        _source_rows.extend(
            source_path_rows(
                v124._mesh,
                v125._plume_state,
                plate_system,
                v124._radius_km,
            )
        )


def _initialize_topography_v130(*args, **kwargs):
    topography = v129._initialize_topography_v129(*args, **kwargs)
    _record_source_state(None, force_path=True)
    return topography


def _load_checkpoint_v130(path, manager):
    global _rows, _source_rows
    checkpoint = v129._load_checkpoint_v129(path, manager)
    _rows = list(checkpoint.plume_drift_rows)
    _source_rows = list(checkpoint.plume_source_path_rows)
    if not _rows:
        _record_source_state(checkpoint.system, force_path=not _source_rows)
    return checkpoint


def _advance_cycle_v130(*args, **kwargs):
    result = v129._advance_cycle_v129(*args, **kwargs)
    plate_system = kwargs.get("plate_system")
    _record_source_state(plate_system)
    return result


def _build_checkpoint_v130(*args, **kwargs):
    checkpoint = v129._build_checkpoint_v129(*args, **kwargs)
    checkpoint.plume_drift_rows = list(_rows)
    checkpoint.plume_source_path_rows = list(_source_rows)
    return checkpoint


def _save_frame_v130(lithosphere, topography) -> None:
    if (
        v129._state is None
        or v128._state is None
        or v124._mesh is None
        or v125._plume_state is None
    ):
        return
    save_hotspot_track_frame(
        v124._mesh,
        lithosphere,
        topography,
        v125._plume_state,
        v128._state,
        v129._state,
        v124._radius_km,
        int(np.unique(lithosphere.cell_plate).size),
        _output
        / "hotspot_track_frames"
        / f"hotspot_tracks_{lithosphere.time_myr:08.1f}_Myr.png",
        v129._frame_dpi,
        source_path_rows=_source_rows,
        version_label=_frame_version_label,
    )


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_v130_outputs() -> None:
    _output.mkdir(parents=True, exist_ok=True)
    _write_csv(_rows, _output / "plume_drift_history.csv")
    _write_csv(_source_rows, _output / "plume_source_paths.csv")
    save_plume_drift_history(_rows, _output / "plume_drift_history.png", _dpi)
    save_plume_source_paths(_source_rows, _output / "plume_source_paths.png", _dpi)
    if v129._finalize:
        flush_rendering()
    frames = sorted(
        (_output / "hotspot_track_frames").glob("hotspot_tracks_*_Myr.png")
    )
    if v129._finalize:
        build_hotspot_track_gif(
            frames,
            _output / "mobile_hotspot_evolution.gif",
            v129._gif_frame_duration_ms,
        )

    summary_path = _output / base.SUMMARY_FILENAME
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        if _rows:
            final = _rows[-1]
            summary.update(
                {
                    "plume_source_drift_enabled": bool(final["enabled"]),
                    "final_active_mobile_plume_sources": int(
                        final["active_source_count"]
                    ),
                    "final_mean_source_speed_km_per_myr": float(
                        final["mean_source_speed_km_per_myr"]
                    ),
                    "final_mean_overlying_plate_speed_km_per_myr": float(
                        final["mean_overlying_plate_speed_km_per_myr"]
                    ),
                    "final_mean_relative_track_speed_km_per_myr": float(
                        final["mean_relative_track_speed_km_per_myr"]
                    ),
                    "final_mean_source_motion_deflection_deg": float(
                        final["mean_source_motion_deflection_deg"]
                    ),
                    "cumulative_plume_source_path_length_km": float(
                        final["population_source_path_length_km"]
                    ),
                    "cumulative_plume_source_bend_angle_deg": float(
                        final["population_source_bend_angle_deg"]
                    ),
                }
            )
        summary.setdefault("notes", []).append(
            "v0.30 gives each deep plume source a deterministic checkpointed piecewise-great-circle drift. Diagnostics explicitly separate overlying-plate velocity, source velocity and their relative hotspot-track velocity."
        )
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)


def main() -> None:
    base.RUN_VERSION = "0.30-mobile-plumes"
    base.RUN_MODEL = (
        "v0.29 plume-head/tail hotspot tracks plus deterministic mobile deep "
        "sources and explicit source/plate relative kinematics"
    )
    base.SUMMARY_FILENAME = "summary_v130.json"
    base.RUN_DESCRIPTION = "v0.30 mobile plume sources and bent hotspot tracks"
    base.DEFAULT_OUTPUT = "outputs_v130_mobile_plumes"
    base.parse_args = _parse_args_v130
    base.initialize_lithosphere = v127._original_initialize_lithosphere_v126
    base.load_checkpoint = _load_checkpoint_v130
    base.advance_lithosphere = v129._advance_lithosphere_v129
    base.advance_continental_cycle = _advance_cycle_v130
    base.refresh_mechanical_lithosphere = v124._refresh_mechanical_lithosphere_v124
    v128._topography_fields = v129._topography_fields_v129
    base.initialize_topography = _initialize_topography_v130
    base.advance_topography = v129._advance_topography_v129
    base.equilibrium_elevation = v128._equilibrium_v128
    base.topography_components = v128._components_v128
    base.build_checkpoint = _build_checkpoint_v130
    v129._save_frame = _save_frame_v130
    base.main()
    v124._write_v124_outputs()
    v125._write_v125_outputs()
    v126._write_v126_outputs()
    v127._write_v127_outputs()
    v128._write_v128_outputs()
    v129._write_v129_outputs()
    _write_v130_outputs()
    if _rows:
        final = _rows[-1]
        print(
            "v0.30 mobile plume sources: "
            f"source={final['mean_source_speed_km_per_myr']:.2f} km/Myr | "
            f"plate={final['mean_overlying_plate_speed_km_per_myr']:.2f} km/Myr | "
            f"relative={final['mean_relative_track_speed_km_per_myr']:.2f} km/Myr | "
            f"deflection={final['mean_source_motion_deflection_deg']:.1f} deg"
        )


if __name__ == "__main__":
    main()
