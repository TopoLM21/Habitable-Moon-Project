#!/usr/bin/env python3
"""Moon Tectonics v0.31 mantle-flow-coupled mobile plume sources."""

from __future__ import annotations
from visualization.render_runtime import flush_rendering

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import run_long_evolution_v130 as v130
from tectonics.plume_flow_coupling import (
    PlumeFlowCouplingParameters,
    diagnose_plume_flow_coupling,
    plume_parameters_with_flow_coupling,
)
from tectonics.plumes import update_plume_source_flow
from visualization.hotspot_tracks import build_hotspot_track_gif
from visualization.plume_flow_coupling import save_plume_flow_coupling_history

v129 = v130.v129
v128 = v130.v128
v127 = v130.v127
v126 = v130.v126
v125 = v130.v125
v124 = v130.v124
base = v130.base

_original_initialize_mantle_flow = base.initialize_mantle_flow
_original_advance_mantle_flow = base.advance_mantle_flow
_original_advance_mantle_plumes = v125.advance_mantle_plumes

_params = PlumeFlowCouplingParameters()
_mantle_flow = None
_rows: list[dict] = []
_output = Path("outputs_v131_flow_coupled_plumes")
_dpi = 180


def _row(diagnostics) -> dict:
    return {
        name: getattr(diagnostics, name)
        for name in diagnostics.__dataclass_fields__
    }


def _parse_args_v131():
    global _params, _mantle_flow, _rows, _output, _dpi
    args = v130._parse_args_v130()
    config = base.load_config(args.config)
    _params = base.dc(
        PlumeFlowCouplingParameters, config.get("plume_flow_coupling", {})
    )
    v125._params = plume_parameters_with_flow_coupling(v125._params, _params)
    _mantle_flow = None
    _rows = []
    _output = Path(args.output)
    _dpi = int(config.get("output", {}).get("dpi", 180))
    return args


def _initialize_mantle_flow_v131(*args, **kwargs):
    global _mantle_flow
    _mantle_flow = _original_initialize_mantle_flow(*args, **kwargs)
    return _mantle_flow


def _advance_mantle_flow_v131(*args, **kwargs):
    global _mantle_flow
    result = _original_advance_mantle_flow(*args, **kwargs)
    _mantle_flow = result[0]
    return result


def _advance_mantle_plumes_v131(*args, **kwargs):
    if _params.enabled and _mantle_flow is not None:
        kwargs["source_flow_omega_field_rad_per_myr"] = (
            _mantle_flow.cell_omega_rad_per_myr
        )
    return _original_advance_mantle_plumes(*args, **kwargs)


def _record_flow_coupling() -> None:
    if v125._plume_state is None:
        return
    _rows.append(
        _row(
            diagnose_plume_flow_coupling(
                v125._plume_state, v124._radius_km, _params
            )
        )
    )


def _initialize_topography_v131(*args, **kwargs):
    if (
        _params.enabled
        and _mantle_flow is not None
        and v124._mesh is not None
        and v125._plume_state is not None
    ):
        update_plume_source_flow(
            v124._mesh,
            v125._plume_state,
            _mantle_flow.cell_omega_rad_per_myr,
            v124._radius_km,
            v125._params,
            initialize_effective_velocity=True,
        )
    topography = v130._initialize_topography_v130(*args, **kwargs)
    _record_flow_coupling()
    return topography


def _load_checkpoint_v131(path, manager):
    global _mantle_flow, _rows
    checkpoint = v130._load_checkpoint_v130(path, manager)
    _mantle_flow = checkpoint.mantle_flow
    _rows = list(checkpoint.plume_flow_coupling_rows)
    if not _rows:
        if (
            _params.enabled
            and _mantle_flow is not None
            and v124._mesh is not None
            and v125._plume_state is not None
        ):
            update_plume_source_flow(
                v124._mesh,
                v125._plume_state,
                _mantle_flow.cell_omega_rad_per_myr,
                v124._radius_km,
                v125._params,
                initialize_effective_velocity=True,
            )
        _record_flow_coupling()
    return checkpoint


def _advance_cycle_v131(*args, **kwargs):
    result = v130._advance_cycle_v130(*args, **kwargs)
    _record_flow_coupling()
    return result


def _build_checkpoint_v131(*args, **kwargs):
    checkpoint = v130._build_checkpoint_v130(*args, **kwargs)
    checkpoint.plume_flow_coupling_rows = list(_rows)
    return checkpoint


def _write_v131_outputs() -> None:
    _output.mkdir(parents=True, exist_ok=True)
    if _rows:
        with (_output / "plume_flow_coupling_history.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(_rows[0].keys()))
            writer.writeheader()
            writer.writerows(_rows)
        save_plume_flow_coupling_history(
            _rows, _output / "plume_flow_coupling_history.png", _dpi
        )
    if v129._finalize:
        flush_rendering()
    frames = sorted(
        (_output / "hotspot_track_frames").glob("hotspot_tracks_*_Myr.png")
    )
    if v129._finalize:
        build_hotspot_track_gif(
            frames,
            _output / "flow_coupled_hotspot_evolution.gif",
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
                    "plume_source_flow_coupling_enabled": bool(final["enabled"]),
                    "final_mean_resolved_plume_flow_speed_km_per_myr": float(
                        final["mean_resolved_flow_speed_km_per_myr"]
                    ),
                    "final_mean_residual_plume_speed_km_per_myr": float(
                        final["mean_residual_speed_km_per_myr"]
                    ),
                    "final_mean_effective_plume_speed_km_per_myr": float(
                        final["mean_effective_source_speed_km_per_myr"]
                    ),
                    "final_mean_plume_flow_alignment": float(
                        final["mean_effective_flow_alignment"]
                    ),
                }
            )
        summary.setdefault("notes", []).append(
            "v0.31 couples deep plume-source motion to a Gaussian sample of the checkpointed Eulerian mantle-flow field and retains a smaller deterministic residual drift."
        )
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)


def main() -> None:
    base.RUN_VERSION = "0.31-flow-coupled-plumes"
    base.RUN_MODEL = (
        "v0.30 mobile plume sources coupled to the independent fixed-grid "
        "mantle-flow memory with a smaller unresolved residual"
    )
    base.SUMMARY_FILENAME = "summary_v131.json"
    base.RUN_DESCRIPTION = "v0.31 mantle-flow-coupled mobile plume sources"
    base.DEFAULT_OUTPUT = "outputs_v131_flow_coupled_plumes"
    base.parse_args = _parse_args_v131
    base.initialize_mantle_flow = _initialize_mantle_flow_v131
    base.advance_mantle_flow = _advance_mantle_flow_v131
    v125.advance_mantle_plumes = _advance_mantle_plumes_v131
    base.initialize_lithosphere = v127._original_initialize_lithosphere_v126
    base.load_checkpoint = _load_checkpoint_v131
    base.advance_lithosphere = v129._advance_lithosphere_v129
    base.advance_continental_cycle = _advance_cycle_v131
    base.refresh_mechanical_lithosphere = v124._refresh_mechanical_lithosphere_v124
    v128._topography_fields = v129._topography_fields_v129
    base.initialize_topography = _initialize_topography_v131
    base.advance_topography = v129._advance_topography_v129
    base.equilibrium_elevation = v128._equilibrium_v128
    base.topography_components = v128._components_v128
    base.build_checkpoint = _build_checkpoint_v131
    v130._frame_version_label = "v0.31"
    v129._save_frame = v130._save_frame_v130
    base.main()
    v124._write_v124_outputs()
    v125._write_v125_outputs()
    v126._write_v126_outputs()
    v127._write_v127_outputs()
    v128._write_v128_outputs()
    v129._write_v129_outputs()
    v130._write_v130_outputs()
    _write_v131_outputs()
    if _rows:
        final = _rows[-1]
        print(
            "v0.31 flow-coupled plumes: "
            f"flow={final['mean_resolved_flow_speed_km_per_myr']:.2f} km/Myr | "
            f"residual={final['mean_residual_speed_km_per_myr']:.2f} km/Myr | "
            f"effective={final['mean_effective_source_speed_km_per_myr']:.2f} km/Myr | "
            f"alignment={final['mean_effective_flow_alignment']:+.3f}"
        )


if __name__ == "__main__":
    main()
