#!/usr/bin/env python3
"""Moon Tectonics v0.27 transient plume dynamic-topography runner."""

from __future__ import annotations
from visualization.render_runtime import flush_rendering

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import run_long_evolution_v126 as v126
from tectonics.plume_dynamic_topography import (
    PlumeDynamicTopographyParameters,
    advance_plume_dynamic_topography,
    diagnose_plume_dynamic_topography,
    initialize_plume_dynamic_topography,
)
from visualization.plume_dynamic_topography import (
    build_plume_dynamic_topography_gif,
    save_plume_dynamic_topography_frame,
    save_plume_dynamic_topography_history,
    save_plume_dynamic_topography_maps,
)

v125 = v126.v125
v124 = v126.v124
base = v126.base

_original_parse_v126 = v126._parse_args_v126
_original_initialize_lithosphere_v126 = v126._initialize_lithosphere_v126
_original_load_v126 = v126._load_checkpoint_v126
_original_advance_lithosphere_v126 = v126._advance_lithosphere_v126
_original_advance_cycle_v125 = v126._original_advance_cycle_v125
_original_build_checkpoint_v126 = v126._build_checkpoint_v126
_original_initialize_topography = base.initialize_topography
_original_advance_topography = base.advance_topography
_original_equilibrium_elevation = base.equilibrium_elevation
_original_topography_components = base.topography_components

_params = PlumeDynamicTopographyParameters()
_state = None
_rows: list[dict] = []
_output = Path("outputs_v127_plume_dynamic_topography")
_dpi = 180
_frame_dpi = 105
_frame_interval = None
_finalize = False
_gif_frame_duration_ms = 350
_last_lithosphere = None
_last_topography = None


def _row(diagnostics) -> dict:
    return {
        name: getattr(diagnostics, name)
        for name in diagnostics.__dataclass_fields__
    }


def _parse_args_v127():
    global _params, _state, _rows, _output, _dpi, _frame_dpi
    global _frame_interval, _finalize, _gif_frame_duration_ms
    global _last_lithosphere, _last_topography
    args = _original_parse_v126()
    config = base.load_config(args.config)
    _params = base.dc(
        PlumeDynamicTopographyParameters,
        config.get("plume_dynamic_topography", {}),
    )
    _state = None
    _rows = []
    _output = Path(args.output)
    output_config = dict(config.get("output", {}))
    evolution_config = dict(config.get("evolution", {}))
    _dpi = int(output_config.get("dpi", 180))
    _frame_dpi = int(output_config.get("thermal_dpi", 105))
    _frame_interval = (
        None if args.frame_interval is None else float(args.frame_interval)
    )
    _finalize = bool(args.finalize)
    _gif_frame_duration_ms = int(
        evolution_config.get("gif_frame_duration_ms", 350)
    )
    _last_lithosphere = None
    _last_topography = None
    return args


def _save_frame(lithosphere, topography) -> None:
    if _state is None or v124._mesh is None or v125._plume_state is None:
        return
    save_plume_dynamic_topography_frame(
        v124._mesh,
        lithosphere,
        topography,
        v125._plume_state,
        _state,
        int(np.unique(lithosphere.cell_plate).size),
        _output
        / "plume_dynamic_topography_frames"
        / f"plume_dynamic_topography_{lithosphere.time_myr:08.1f}_Myr.png",
        _frame_dpi,
    )


def _initialize_topography_v127(*args, **kwargs):
    global _state, _rows, _last_lithosphere, _last_topography
    topography = _original_initialize_topography(*args, **kwargs)
    mesh = args[0] if args else kwargs["mesh"]
    lithosphere = args[1] if len(args) > 1 else kwargs["state"]
    radius_km = float(args[4] if len(args) > 4 else kwargs["radius_km"])
    if v125._plume_state is None:
        raise RuntimeError("v0.27 requires an initialized v0.25 plume population")
    _state = initialize_plume_dynamic_topography(mesh, lithosphere.time_myr)
    _rows = [
        _row(
            diagnose_plume_dynamic_topography(
                mesh,
                v125._plume_state,
                _state,
                radius_km,
                _params,
            )
        )
    ]
    _last_lithosphere = lithosphere
    _last_topography = topography
    if _frame_interval is not None:
        _save_frame(lithosphere, topography)
    return topography


def _load_checkpoint_v127(path, manager):
    global _state, _rows, _last_lithosphere, _last_topography
    checkpoint = _original_load_v126(path, manager)
    _state = checkpoint.plume_dynamic_topography_state
    if _state is None:
        _state = initialize_plume_dynamic_topography(
            v124._mesh, checkpoint.state.time_myr
        )
    _rows = list(checkpoint.plume_dynamic_topography_rows)
    if not _rows:
        _rows.append(
            _row(
                diagnose_plume_dynamic_topography(
                    v124._mesh,
                    v125._plume_state,
                    _state,
                    v124._radius_km,
                    _params,
                )
            )
        )
    _last_lithosphere = checkpoint.state
    _last_topography = checkpoint.topo
    return checkpoint


def _advance_topography_v127(*args, **kwargs):
    global _state, _last_lithosphere, _last_topography
    mesh = args[0] if args else kwargs["mesh"]
    lithosphere = args[1] if len(args) > 1 else kwargs["lithosphere"]
    dt_myr = float(args[4] if len(args) > 4 else kwargs["dt_myr"])
    radius_km = float(args[5] if len(args) > 5 else kwargs["radius_km"])
    if v125._plume_state is None:
        raise RuntimeError("v0.27 requires an initialized v0.25 plume population")
    if _state is None:
        _state = initialize_plume_dynamic_topography(
            mesh, lithosphere.time_myr - dt_myr
        )
    _state, diagnostics = advance_plume_dynamic_topography(
        mesh,
        v125._plume_state,
        _state,
        dt_myr,
        radius_km,
        _params,
    )
    _rows.append(_row(diagnostics))
    existing = kwargs.get("dynamic_topography_m")
    if existing is None:
        combined = _state.realized_dynamic_topography_m
    else:
        existing_array = np.asarray(existing, dtype=np.float64)
        if existing_array.shape != (mesh.cell_count,):
            raise ValueError("existing dynamic topography must have shape (cell_count,)")
        combined = existing_array + _state.realized_dynamic_topography_m
    kwargs["dynamic_topography_m"] = combined
    result = _original_advance_topography(*args, **kwargs)
    topography = result[0]
    _last_lithosphere = lithosphere
    _last_topography = topography
    if (
        _frame_interval is not None
        and _frame_interval > 0.0
        and abs(
            lithosphere.time_myr / _frame_interval
            - round(lithosphere.time_myr / _frame_interval)
        )
        < 1.0e-9
    ):
        _save_frame(lithosphere, topography)
    return result


def _equilibrium_elevation_v127(*args, **kwargs):
    if _state is not None:
        kwargs.setdefault(
            "dynamic_topography_m", _state.realized_dynamic_topography_m
        )
    return _original_equilibrium_elevation(*args, **kwargs)


def _topography_components_v127(*args, **kwargs):
    if _state is not None:
        kwargs.setdefault(
            "dynamic_topography_m", _state.realized_dynamic_topography_m
        )
    return _original_topography_components(*args, **kwargs)


def _build_checkpoint_v127(*args, **kwargs):
    checkpoint = _original_build_checkpoint_v126(*args, **kwargs)
    checkpoint.plume_dynamic_topography_state = _state
    checkpoint.plume_dynamic_topography_rows = list(_rows)
    return checkpoint


def _write_v127_outputs() -> None:
    if _state is None or v124._mesh is None:
        return
    _output.mkdir(parents=True, exist_ok=True)
    if _rows:
        with (_output / "plume_dynamic_topography_history.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(_rows[0].keys()))
            writer.writeheader()
            writer.writerows(_rows)
        save_plume_dynamic_topography_history(
            _rows,
            _output / "plume_dynamic_topography_history.png",
        )
    save_plume_dynamic_topography_maps(v124._mesh, _state, _output, _dpi)

    if _finalize and _last_lithosphere is not None and _last_topography is not None:
        _save_frame(_last_lithosphere, _last_topography)
    if _finalize:
        flush_rendering()
    frames = sorted(
        (_output / "plume_dynamic_topography_frames").glob(
            "plume_dynamic_topography_*_Myr.png"
        )
    )
    if _finalize:
        build_plume_dynamic_topography_gif(
            frames,
            _output / "plume_dynamic_topography_history.gif",
            _gif_frame_duration_ms,
        )

    summary_path = _output / base.SUMMARY_FILENAME
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        if _rows:
            final = _rows[-1]
            summary.update(
                {
                    "plume_dynamic_topography_enabled": bool(final["enabled"]),
                    "final_max_plume_dynamic_uplift_m": float(
                        final["maximum_realized_uplift_m"]
                    ),
                    "final_min_plume_dynamic_subsidence_m": float(
                        final["minimum_realized_subsidence_m"]
                    ),
                    "final_rms_plume_dynamic_topography_m": float(
                        final["rms_realized_anomaly_m"]
                    ),
                    "final_plume_dynamic_uplift_area_fraction": float(
                        final["affected_surface_area_fraction"]
                    ),
                    "final_plume_weighted_dynamic_uplift_m": float(
                        final["plume_weighted_mean_uplift_m"]
                    ),
                    "maximum_plume_dynamic_uplift_over_run_m": float(
                        max(row["maximum_realized_uplift_m"] for row in _rows)
                    ),
                    "maximum_abs_dynamic_displacement_volume_km3": float(
                        max(abs(row["displacement_volume_km3"]) for row in _rows)
                    ),
                    "cumulative_mean_positive_dynamic_support_m_myr": float(
                        final["cumulative_mean_positive_support_m_myr"]
                    ),
                }
            )
        summary.setdefault("notes", []).append(
            "v0.27 adds delayed, reversible plume dynamic topography as a non-flexed mantle-support anomaly. Its area-weighted degree-zero component is removed every step, and it creates no crustal material."
        )
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)


def main() -> None:
    base.RUN_VERSION = "0.27-plume-dynamic-topography"
    base.RUN_MODEL = (
        "v0.26 plume-driven rifting plus delayed, reversible, zero-mean "
        "plume dynamic topography"
    )
    base.SUMMARY_FILENAME = "summary_v127.json"
    base.RUN_DESCRIPTION = "v0.27 transient plume dynamic topography"
    base.DEFAULT_OUTPUT = "outputs_v127_plume_dynamic_topography"
    base.parse_args = _parse_args_v127
    base.initialize_lithosphere = _original_initialize_lithosphere_v126
    base.load_checkpoint = _load_checkpoint_v127
    base.advance_lithosphere = _original_advance_lithosphere_v126
    base.advance_continental_cycle = _original_advance_cycle_v125
    base.refresh_mechanical_lithosphere = v124._refresh_mechanical_lithosphere_v124
    base.initialize_topography = _initialize_topography_v127
    base.advance_topography = _advance_topography_v127
    base.equilibrium_elevation = _equilibrium_elevation_v127
    base.topography_components = _topography_components_v127
    base.build_checkpoint = _build_checkpoint_v127
    base.main()
    v124._write_v124_outputs()
    v125._write_v125_outputs()
    v126._write_v126_outputs()
    _write_v127_outputs()
    if _rows:
        final = _rows[-1]
        print(
            "v0.27 plume dynamic topography: "
            f"enabled={final['enabled']} | "
            f"max_uplift={final['maximum_realized_uplift_m']:.1f} m | "
            f"min_subsidence={final['minimum_realized_subsidence_m']:.1f} m | "
            f"RMS={final['rms_realized_anomaly_m']:.1f} m"
        )


if __name__ == "__main__":
    main()
