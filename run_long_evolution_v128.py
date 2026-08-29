#!/usr/bin/env python3
"""Moon Tectonics v0.28 permanent plume-magmatism runner."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import run_long_evolution_v127 as v127
from tectonics.plume_magmatism import (
    PlumeMagmatismParameters,
    advect_plume_magmatism,
    advance_plume_magmatism,
    diagnose_plume_magmatism,
    igneous_ledger_error_km3,
    initialize_plume_magmatism,
    magmatic_topography_fields,
)
from visualization.plume_magmatism import (
    build_plume_magmatism_gif,
    save_plume_magmatism_frame,
    save_plume_magmatism_history,
    save_plume_magmatism_maps,
)

v126 = v127.v126
v125 = v127.v125
v124 = v127.v124
base = v127.base

_original_parse_v127 = v127._parse_args_v127
_original_initialize_topography_v127 = v127._initialize_topography_v127
_original_load_v127 = v127._load_checkpoint_v127
_original_advance_lithosphere_v126 = v127._original_advance_lithosphere_v126
_original_advance_cycle_v125 = v127._original_advance_cycle_v125
_original_advance_topography_v127 = v127._advance_topography_v127
_original_equilibrium_v127 = v127._equilibrium_elevation_v127
_original_components_v127 = v127._topography_components_v127
_original_build_checkpoint_v127 = v127._build_checkpoint_v127

_params = PlumeMagmatismParameters()
_state = None
_rows: list[dict] = []
_output = Path("outputs_v128_plume_magmatism")
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


def _parse_args_v128():
    global _params, _state, _rows, _output, _dpi, _frame_dpi
    global _frame_interval, _finalize, _gif_frame_duration_ms
    global _last_lithosphere, _last_topography
    args = _original_parse_v127()
    config = base.load_config(args.config)
    _params = base.dc(
        PlumeMagmatismParameters, config.get("plume_magmatism", {})
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


def _topography_fields(mesh, radius_km: float) -> dict:
    if _state is None:
        zero = np.zeros(mesh.cell_count, dtype=np.float64)
        return {
            "magmatic_extrusive_thickness_m": zero.copy(),
            "magmatic_extrusive_load_m": zero.copy(),
            "magmatic_intrusive_support_m": zero.copy(),
        }
    extrusive, load, intrusive, _ = magmatic_topography_fields(
        mesh, _state, radius_km, _params
    )
    return {
        "magmatic_extrusive_thickness_m": extrusive,
        "magmatic_extrusive_load_m": load,
        "magmatic_intrusive_support_m": intrusive,
    }


def _save_frame(lithosphere, topography) -> None:
    if (
        _state is None
        or v124._mesh is None
        or v125._plume_state is None
    ):
        return
    save_plume_magmatism_frame(
        v124._mesh,
        lithosphere,
        topography,
        v125._plume_state,
        _state,
        v124._radius_km,
        int(np.unique(lithosphere.cell_plate).size),
        _output
        / "plume_magmatism_frames"
        / f"plume_magmatism_{lithosphere.time_myr:08.1f}_Myr.png",
        _frame_dpi,
    )


def _initialize_topography_v128(*args, **kwargs):
    global _state, _rows, _last_lithosphere, _last_topography
    mesh = args[0] if args else kwargs["mesh"]
    lithosphere = args[1] if len(args) > 1 else kwargs["state"]
    radius_km = float(args[4] if len(args) > 4 else kwargs["radius_km"])
    _state = initialize_plume_magmatism(mesh, lithosphere.time_myr)
    _rows = [
        _row(
            diagnose_plume_magmatism(
                mesh, _state, radius_km, _params
            )
        )
    ]
    kwargs.update(_topography_fields(mesh, radius_km))
    topography = _original_initialize_topography_v127(*args, **kwargs)
    _last_lithosphere = lithosphere
    _last_topography = topography
    if _frame_interval is not None:
        _save_frame(lithosphere, topography)
    return topography


def _load_checkpoint_v128(path, manager):
    global _state, _rows, _last_lithosphere, _last_topography
    checkpoint = _original_load_v127(path, manager)
    _state = checkpoint.plume_magmatism_state
    if _state is None:
        _state = initialize_plume_magmatism(
            v124._mesh, checkpoint.state.time_myr
        )
    _rows = list(checkpoint.plume_magmatism_rows)
    if not _rows:
        _rows.append(
            _row(
                diagnose_plume_magmatism(
                    v124._mesh, _state, v124._radius_km, _params
                )
            )
        )
    _last_lithosphere = checkpoint.state
    _last_topography = checkpoint.topo
    return checkpoint


def _advance_lithosphere_v128(*args, **kwargs):
    global _state
    mesh = args[0] if args else kwargs["mesh"]
    lithosphere = args[2] if len(args) > 2 else kwargs["state"]
    dt_myr = float(args[3] if len(args) > 3 else kwargs["dt_myr"])
    if _state is None:
        _state = initialize_plume_magmatism(mesh, lithosphere.time_myr)
    result = _original_advance_lithosphere_v126(*args, **kwargs)
    diagnostics = result[3]
    if diagnostics.material_source_index is None:
        raise RuntimeError("v0.28 requires the material source-index diagnostic")
    _state = advect_plume_magmatism(
        _state, diagnostics.material_source_index, dt_myr
    )
    return result


def _advance_cycle_v128(*args, **kwargs):
    global _state
    result = _original_advance_cycle_v125(*args, **kwargs)
    mesh = args[0] if args else kwargs["mesh"]
    dt_myr = float(args[4] if len(args) > 4 else kwargs["dt_myr"])
    radius_km = float(args[5] if len(args) > 5 else kwargs["radius_km"])
    if _state is None:
        _state = initialize_plume_magmatism(mesh, result[0].time_myr - dt_myr)
    if v126._state is None:
        raise RuntimeError("v0.28 requires the v0.26 plume-productivity state")
    _state, diagnostics = advance_plume_magmatism(
        mesh,
        _state,
        v126._state.last_magmatic_productivity,
        v126._state.last_extension_forcing,
        dt_myr,
        radius_km,
        _params,
    )
    _rows.append(_row(diagnostics))
    ledger = igneous_ledger_error_km3(_state)
    if abs(ledger) > 1.0e-4:
        raise RuntimeError(
            f"igneous material ledger drift at t={_state.time_myr:.1f} Myr: "
            f"{ledger:+.6g} km3"
        )
    return result


def _advance_topography_v128(*args, **kwargs):
    global _last_lithosphere, _last_topography
    mesh = args[0] if args else kwargs["mesh"]
    lithosphere = args[1] if len(args) > 1 else kwargs["lithosphere"]
    radius_km = float(args[5] if len(args) > 5 else kwargs["radius_km"])
    kwargs.update(_topography_fields(mesh, radius_km))
    result = _original_advance_topography_v127(*args, **kwargs)
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


def _equilibrium_v128(*args, **kwargs):
    mesh = args[0] if args else kwargs["mesh"]
    radius_km = float(args[4] if len(args) > 4 else kwargs["radius_km"])
    kwargs.update(_topography_fields(mesh, radius_km))
    return _original_equilibrium_v127(*args, **kwargs)


def _components_v128(*args, **kwargs):
    mesh = args[0] if args else kwargs["mesh"]
    radius_km = float(args[4] if len(args) > 4 else kwargs["radius_km"])
    kwargs.update(_topography_fields(mesh, radius_km))
    return _original_components_v127(*args, **kwargs)


def _build_checkpoint_v128(*args, **kwargs):
    checkpoint = _original_build_checkpoint_v127(*args, **kwargs)
    checkpoint.plume_magmatism_state = _state
    checkpoint.plume_magmatism_rows = list(_rows)
    return checkpoint


def _write_v128_outputs() -> None:
    if _state is None or v124._mesh is None:
        return
    _output.mkdir(parents=True, exist_ok=True)
    if _rows:
        with (_output / "plume_magmatism_history.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(_rows[0].keys()))
            writer.writeheader()
            writer.writerows(_rows)
        save_plume_magmatism_history(
            _rows, _output / "plume_magmatism_history.png"
        )
    save_plume_magmatism_maps(
        v124._mesh, _state, v124._radius_km, _params, _output, _dpi
    )
    if _finalize and _last_lithosphere is not None and _last_topography is not None:
        _save_frame(_last_lithosphere, _last_topography)
    frames = sorted(
        (_output / "plume_magmatism_frames").glob(
            "plume_magmatism_*_Myr.png"
        )
    )
    if _finalize:
        build_plume_magmatism_gif(
            frames,
            _output / "plume_magmatism_history.gif",
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
                    "plume_magmatism_enabled": bool(final["enabled"]),
                    "final_surface_plume_igneous_volume_km3": float(
                        final["surface_extrusive_volume_km3"]
                        + final["surface_dyke_volume_km3"]
                        + final["surface_underplate_volume_km3"]
                    ),
                    "cumulative_generated_plume_igneous_volume_km3": float(
                        final["cumulative_generated_total_volume_km3"]
                    ),
                    "deep_recycled_plume_igneous_volume_km3": float(
                        final["deep_recycled_total_volume_km3"]
                    ),
                    "final_igneous_ledger_error_km3": float(
                        final["global_igneous_ledger_error_km3"]
                    ),
                    "final_maximum_plume_igneous_thickness_km": float(
                        final["maximum_igneous_thickness_km"]
                    ),
                    "final_plume_track_area_fraction": float(
                        final["mapped_track_area_fraction"]
                    ),
                    "final_maximum_plume_track_age_myr": float(
                        final["maximum_track_age_myr"]
                    ),
                    "final_maximum_magmatic_isostatic_support_m": float(
                        final["maximum_density_aware_support_m"]
                    ),
                }
            )
        summary.setdefault("notes", []).append(
            "v0.28 converts plume decompression-melt productivity into permanent, transported extrusive-basalt, dyke/sill and underplate reservoirs. Lost surface parcels close to an explicit deep-recycling ledger; reservoir densities control flexed crustal support."
        )
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)


def main() -> None:
    base.RUN_VERSION = "0.28-plume-magmatism"
    base.RUN_MODEL = (
        "v0.27 transient plume support plus permanent transported extrusive, "
        "dyke/sill and mafic-underplate reservoirs"
    )
    base.SUMMARY_FILENAME = "summary_v128.json"
    base.RUN_DESCRIPTION = "v0.28 permanent plume magmatism and volcanic tracks"
    base.DEFAULT_OUTPUT = "outputs_v128_plume_magmatism"
    base.parse_args = _parse_args_v128
    base.initialize_lithosphere = v127._original_initialize_lithosphere_v126
    base.load_checkpoint = _load_checkpoint_v128
    base.advance_lithosphere = _advance_lithosphere_v128
    base.advance_continental_cycle = _advance_cycle_v128
    base.refresh_mechanical_lithosphere = v124._refresh_mechanical_lithosphere_v124
    base.initialize_topography = _initialize_topography_v128
    base.advance_topography = _advance_topography_v128
    base.equilibrium_elevation = _equilibrium_v128
    base.topography_components = _components_v128
    base.build_checkpoint = _build_checkpoint_v128
    base.main()
    v124._write_v124_outputs()
    v125._write_v125_outputs()
    v126._write_v126_outputs()
    v127._write_v127_outputs()
    _write_v128_outputs()
    if _rows:
        final = _rows[-1]
        surface = (
            final["surface_extrusive_volume_km3"]
            + final["surface_dyke_volume_km3"]
            + final["surface_underplate_volume_km3"]
        )
        print(
            "v0.28 plume magmatism: "
            f"enabled={final['enabled']} | "
            f"surface={surface/1.0e6:.3f} million km3 | "
            f"max_thickness={final['maximum_igneous_thickness_km']:.3f} km | "
            f"ledger={final['global_igneous_ledger_error_km3']:+.3e} km3"
        )


if __name__ == "__main__":
    main()
