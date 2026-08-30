#!/usr/bin/env python3
"""Moon Tectonics v0.29 plume heads, tails and age-progressive tracks."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import run_long_evolution_v128 as v128
from tectonics.hotspot_tracks import (
    HotspotTrackParameters,
    advect_hotspot_tracks,
    advance_hotspot_tracks,
    diagnose_hotspot_tracks,
    initialize_hotspot_tracks,
    magmatic_extension_forcing,
    plume_parameters_with_head_tail,
    underplate_density_field,
)
from tectonics.plume_magmatism import (
    igneous_ledger_error_km3,
    magmatic_topography_fields,
)
from visualization.hotspot_tracks import (
    build_hotspot_track_gif,
    save_hotspot_track_frame,
    save_hotspot_track_history,
    save_hotspot_track_maps,
)

v127 = v128.v127
v126 = v128.v126
v125 = v128.v125
v124 = v128.v124
base = v128.base

_params = HotspotTrackParameters()
_state = None
_rows: list[dict] = []
_output = Path("outputs_v129_hotspot_tracks")
_dpi = 180
_frame_dpi = 105
_frame_interval = None
_finalize = False
_gif_frame_duration_ms = 350


def _row(diagnostics) -> dict:
    return {
        name: getattr(diagnostics, name)
        for name in diagnostics.__dataclass_fields__
    }


def _parse_args_v129():
    global _params, _state, _rows, _output, _dpi, _frame_dpi
    global _frame_interval, _finalize, _gif_frame_duration_ms
    args = v128._parse_args_v128()
    config = base.load_config(args.config)
    _params = base.dc(HotspotTrackParameters, config.get("hotspot_tracks", {}))
    # Older runners default to their original single broad plume.  Only this
    # runner opts the shared plume population into the v0.29 component fields.
    v125._params = plume_parameters_with_head_tail(v125._params, _params)
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
    return args


def _topography_fields_v129(mesh, radius_km: float) -> dict:
    if v128._state is None:
        zero = np.zeros(mesh.cell_count, dtype=np.float64)
        return {
            "magmatic_extrusive_thickness_m": zero.copy(),
            "magmatic_extrusive_load_m": zero.copy(),
            "magmatic_intrusive_support_m": zero.copy(),
        }
    density = None
    if _state is not None:
        density = underplate_density_field(_state, v128._params, _params)
    extrusive, load, intrusive, _ = magmatic_topography_fields(
        mesh,
        v128._state,
        radius_km,
        v128._params,
        underplate_density_kg_m3=density,
    )
    return {
        "magmatic_extrusive_thickness_m": extrusive,
        "magmatic_extrusive_load_m": load,
        "magmatic_intrusive_support_m": intrusive,
    }


def _save_frame(lithosphere, topography) -> None:
    if (
        _state is None
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
        _state,
        v124._radius_km,
        int(np.unique(lithosphere.cell_plate).size),
        _output
        / "hotspot_track_frames"
        / f"hotspot_tracks_{lithosphere.time_myr:08.1f}_Myr.png",
        _frame_dpi,
    )


def _initialize_topography_v129(*args, **kwargs):
    global _state, _rows
    mesh = args[0] if args else kwargs["mesh"]
    lithosphere = args[1] if len(args) > 1 else kwargs["state"]
    radius_km = float(args[4] if len(args) > 4 else kwargs["radius_km"])
    _state = initialize_hotspot_tracks(mesh, lithosphere.time_myr)
    topography = v128._initialize_topography_v128(*args, **kwargs)
    if v128._state is None or v125._plume_state is None:
        raise RuntimeError("v0.29 requires initialized plume and magmatic states")
    _rows = [
        _row(
            diagnose_hotspot_tracks(
                mesh,
                _state,
                v128._state,
                v125._plume_state,
                radius_km,
                _params,
            )
        )
    ]
    if _frame_interval is not None:
        _save_frame(lithosphere, topography)
    return topography


def _load_checkpoint_v129(path, manager):
    global _state, _rows
    checkpoint = v128._load_checkpoint_v128(path, manager)
    _state = checkpoint.hotspot_track_state
    if _state is None:
        _state = initialize_hotspot_tracks(v124._mesh, checkpoint.state.time_myr)
    _rows = list(checkpoint.hotspot_track_rows)
    if not _rows:
        _rows.append(
            _row(
                diagnose_hotspot_tracks(
                    v124._mesh,
                    _state,
                    v128._state,
                    v125._plume_state,
                    v124._radius_km,
                    _params,
                )
            )
        )
    return checkpoint


def _advance_lithosphere_v129(*args, **kwargs):
    global _state
    mesh = args[0] if args else kwargs["mesh"]
    lithosphere = args[2] if len(args) > 2 else kwargs["state"]
    if _state is None:
        _state = initialize_hotspot_tracks(mesh, lithosphere.time_myr)
    thermal_forcing = magmatic_extension_forcing(_state, _params)
    existing = kwargs.get("continental_extension_external_forcing")
    if existing is None:
        combined = thermal_forcing
    else:
        existing_array = np.asarray(existing, dtype=np.float64)
        if existing_array.shape != (mesh.cell_count,):
            raise ValueError("existing external forcing must match cell count")
        combined = np.maximum(existing_array, thermal_forcing)
    kwargs["continental_extension_external_forcing"] = combined
    result = v128._advance_lithosphere_v128(*args, **kwargs)
    diagnostics = result[3]
    if diagnostics.material_source_index is None:
        raise RuntimeError("v0.29 requires the material source-index diagnostic")
    _state = advect_hotspot_tracks(_state, diagnostics.material_source_index)
    return result


def _advance_cycle_v129(*args, **kwargs):
    global _state
    result = v128._original_advance_cycle_v125(*args, **kwargs)
    mesh = args[0] if args else kwargs["mesh"]
    dt_myr = float(args[4] if len(args) > 4 else kwargs["dt_myr"])
    radius_km = float(args[5] if len(args) > 5 else kwargs["radius_km"])
    lithosphere = result[0]
    if _state is None:
        _state = initialize_hotspot_tracks(mesh, lithosphere.time_myr - dt_myr)
    if v128._state is None or v126._state is None or v125._plume_state is None:
        raise RuntimeError("v0.29 requires plume, rifting and magmatic states")
    rift_extension = (
        np.zeros(mesh.cell_count, dtype=np.float64)
        if lithosphere.rift_extension is None
        else np.asarray(lithosphere.rift_extension, dtype=np.float64)
    )
    _state, v128._state, track_diag, magmatic_diag = advance_hotspot_tracks(
        mesh,
        _state,
        v128._state,
        v125._plume_state,
        rift_extension,
        v126._state.last_extension_forcing,
        v126._state.last_magmatic_productivity,
        dt_myr,
        radius_km,
        v128._params,
        _params,
    )
    v128._rows.append(v128._row(magmatic_diag))
    _rows.append(_row(track_diag))
    ledger = igneous_ledger_error_km3(v128._state)
    if abs(ledger) > 1.0e-4:
        raise RuntimeError(
            f"igneous material ledger drift at t={v128._state.time_myr:.1f} Myr: "
            f"{ledger:+.6g} km3"
        )
    return result


def _advance_topography_v129(*args, **kwargs):
    result = v128._advance_topography_v128(*args, **kwargs)
    lithosphere = args[1] if len(args) > 1 else kwargs["lithosphere"]
    if (
        _frame_interval is not None
        and _frame_interval > 0.0
        and abs(
            lithosphere.time_myr / _frame_interval
            - round(lithosphere.time_myr / _frame_interval)
        )
        < 1.0e-9
    ):
        _save_frame(lithosphere, result[0])
    return result


def _build_checkpoint_v129(*args, **kwargs):
    checkpoint = v128._build_checkpoint_v128(*args, **kwargs)
    checkpoint.hotspot_track_state = _state
    checkpoint.hotspot_track_rows = list(_rows)
    return checkpoint


def _write_v129_outputs() -> None:
    if _state is None or v128._state is None or v124._mesh is None:
        return
    _output.mkdir(parents=True, exist_ok=True)
    if _rows:
        with (_output / "hotspot_track_history.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(_rows[0].keys()))
            writer.writeheader()
            writer.writerows(_rows)
        save_hotspot_track_history(_rows, _output / "hotspot_track_history.png")
    save_hotspot_track_maps(
        v124._mesh, v128._state, _state, v124._radius_km, _output, _dpi
    )
    if _finalize and v128._last_lithosphere is not None and v128._last_topography is not None:
        _save_frame(v128._last_lithosphere, v128._last_topography)
    frames = sorted(
        (_output / "hotspot_track_frames").glob("hotspot_tracks_*_Myr.png")
    )
    if _finalize:
        build_hotspot_track_gif(
            frames,
            _output / "hotspot_track_evolution.gif",
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
                    "head_tail_separation_enabled": bool(
                        final["head_tail_separation_enabled"]
                    ),
                    "magmatic_thermal_weakening_enabled": bool(
                        final["magmatic_thermal_weakening_enabled"]
                    ),
                    "rift_dike_localization_enabled": bool(
                        final["dike_localization_enabled"]
                    ),
                    "underplate_evolution_enabled": bool(
                        final["underplate_evolution_enabled"]
                    ),
                    "final_maximum_tail_productivity": float(
                        final["maximum_tail_productivity"]
                    ),
                    "final_maximum_magmatic_thermal_anomaly": float(
                        final["maximum_thermal_anomaly"]
                    ),
                    "final_eclogitized_underplate_volume_km3": float(
                        final["eclogitized_underplate_volume_km3"]
                    ),
                    "cumulative_delaminated_underplate_volume_km3": float(
                        final["cumulative_delaminated_underplate_volume_km3"]
                    ),
                    "final_hotspot_track_age_distance_correlation": float(
                        final["hotspot_track_age_distance_correlation"]
                    ),
                    "cumulative_head_generated_igneous_volume_km3": float(
                        final["cumulative_head_generated_volume_km3"]
                    ),
                    "cumulative_tail_generated_igneous_volume_km3": float(
                        final["cumulative_tail_generated_volume_km3"]
                    ),
                    "final_maximum_rift_dike_localization": float(
                        final["maximum_dike_localization"]
                    ),
                }
            )
        summary.setdefault("notes", []).append(
            "v0.29 separates the short broad LIP-forming plume head from a persistent narrow tail, adds transported magmatic heat and syn-rift dyke localization, and conservatively transfers delaminated eclogitized underplate to deep recycling."
        )
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)


def main() -> None:
    base.RUN_VERSION = "0.29-hotspot-tracks"
    base.RUN_MODEL = (
        "v0.28 permanent plume magmatism plus separated LIP heads and hotspot "
        "tails, transported heat, rift-localized dykes and underplate evolution"
    )
    base.SUMMARY_FILENAME = "summary_v129.json"
    base.RUN_DESCRIPTION = "v0.29 plume heads, tails and age-progressive hotspot tracks"
    base.DEFAULT_OUTPUT = "outputs_v129_hotspot_tracks"
    base.parse_args = _parse_args_v129
    base.initialize_lithosphere = v127._original_initialize_lithosphere_v126
    base.load_checkpoint = _load_checkpoint_v129
    base.advance_lithosphere = _advance_lithosphere_v129
    base.advance_continental_cycle = _advance_cycle_v129
    base.refresh_mechanical_lithosphere = v124._refresh_mechanical_lithosphere_v124
    v128._topography_fields = _topography_fields_v129
    base.initialize_topography = _initialize_topography_v129
    base.advance_topography = _advance_topography_v129
    base.equilibrium_elevation = v128._equilibrium_v128
    base.topography_components = v128._components_v128
    base.build_checkpoint = _build_checkpoint_v129
    base.main()
    v124._write_v124_outputs()
    v125._write_v125_outputs()
    v126._write_v126_outputs()
    v127._write_v127_outputs()
    v128._write_v128_outputs()
    _write_v129_outputs()
    if _rows:
        final = _rows[-1]
        print(
            "v0.29 hotspot tracks: "
            f"tail={final['maximum_tail_productivity']:.3f} | "
            f"thermal={final['maximum_thermal_anomaly']:.3f} | "
            f"eclogite={final['eclogitized_underplate_volume_km3']/1.0e6:.3f} million km3 | "
            f"age-distance r={final['hotspot_track_age_distance_correlation']:+.3f}"
        )


if __name__ == "__main__":
    main()
