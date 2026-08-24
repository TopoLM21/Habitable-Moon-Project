#!/usr/bin/env python3
"""Moon Tectonics v0.25 mantle-plume and metasomatic-root runner.

v0.25 wraps the stable v0.24 cratonic-memory runner.  Deep plume forcing is
mantle-fixed while lithospheric material moves across it; the response is
stored in transported age, depletion and strength memory, plus explicit root
erosion.  The long v0.23 integration loop remains unchanged.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

# The bundled Windows Python used by Codex has no Tcl/Tk runtime.  All project
# visualizations are file outputs, so force Matplotlib's non-interactive backend
# before any v0.24 visualization module imports pyplot.
matplotlib.use("Agg")

import run_long_evolution_v124 as v124
from tectonics.cratons import diagnose_craton_memory
from tectonics.plumes import (
    MantlePlumeParameters,
    advance_mantle_plumes,
    diagnose_mantle_plumes,
    initialize_mantle_plumes,
)
from visualization.plumes import save_plume_history, save_plume_maps

base = v124.base

_original_parse_v124 = v124._parse_args_v124
_original_initialize_v124 = v124._initialize_lithosphere_v124
_original_load_v124 = v124._load_checkpoint_v124
_original_advance_cycle_v124 = v124._advance_continental_cycle_v124
_original_build_checkpoint_v124 = v124._build_checkpoint_v124

_params = MantlePlumeParameters()
_plume_state = None
_plume_rows: list[dict] = []
_output = Path("outputs_v125_plumes")


def _row(diag) -> dict:
    return {name: getattr(diag, name) for name in diag.__dataclass_fields__}


def _parse_args_v125():
    global _params, _plume_state, _plume_rows, _output
    args = _original_parse_v124()
    config = base.load_config(args.config)
    plume_config = dict(config.get("mantle_plumes", {}))
    if "seed" not in plume_config:
        plume_config["seed"] = int(config.get("plates", {}).get("seed", 0)) + 247
    _params = base.dc(MantlePlumeParameters, plume_config)
    _plume_state = None
    _plume_rows = []
    _output = Path(args.output)
    return args


def _initialize_lithosphere_v125(*args, **kwargs):
    global _plume_state, _plume_rows
    state = _original_initialize_v124(*args, **kwargs)
    mesh = args[0] if args else kwargs["mesh"]
    _plume_state = initialize_mantle_plumes(mesh, state.time_myr, _params)
    _plume_rows = [
        _row(
            diagnose_mantle_plumes(
                mesh, state, _plume_state, v124._radius_km, _params
            )
        )
    ]
    return state


def _load_checkpoint_v125(path, manager):
    global _plume_state, _plume_rows
    checkpoint = _original_load_v124(path, manager)
    _plume_state = checkpoint.plume_state
    if _plume_state is None:
        # Explicit upgrade path: a v0.24 checkpoint begins a new deterministic
        # plume chronology at its resume time rather than inventing past forcing.
        _plume_state = initialize_mantle_plumes(
            v124._mesh, checkpoint.state.time_myr, _params
        )
    _plume_rows = list(checkpoint.plume_rows)
    if not _plume_rows:
        _plume_rows.append(
            _row(
                diagnose_mantle_plumes(
                    v124._mesh,
                    checkpoint.state,
                    _plume_state,
                    v124._radius_km,
                    _params,
                )
            )
        )
    return checkpoint


def _advance_continental_cycle_v125(*args, **kwargs):
    global _plume_state
    state, cycle, diagnostics = _original_advance_cycle_v124(*args, **kwargs)
    mesh = args[0] if args else kwargs["mesh"]
    dt_myr = float(args[4] if len(args) > 4 else kwargs["dt_myr"])
    radius_km = float(args[5] if len(args) > 5 else kwargs["radius_km"])
    if _plume_state is None:
        _plume_state = initialize_mantle_plumes(
            mesh, state.time_myr - dt_myr, _params
        )
    state, _plume_state, plume_diagnostics = advance_mantle_plumes(
        mesh,
        state,
        _plume_state,
        dt_myr,
        radius_km,
        _params,
        v124._params,
    )
    _plume_rows.append(_row(plume_diagnostics))
    # v0.24 recorded diagnostics just before plume modification.  Replace the
    # last row with the post-plume state while preserving its juvenile-volume
    # bookkeeping for this step.
    new_volume = (
        float(v124._craton_rows[-1].get("new_continental_material_volume_km3", 0.0))
        if v124._craton_rows
        else 0.0
    )
    post = diagnose_craton_memory(
        mesh,
        state,
        radius_km,
        v124._params,
        dt_myr=dt_myr,
        new_continental_material_volume_km3=new_volume,
    )
    if v124._craton_rows:
        v124._craton_rows[-1] = v124._row(post)
    else:
        v124._craton_rows.append(v124._row(post))
    v124._last_state = state
    return state, cycle, diagnostics


def _build_checkpoint_v125(*args, **kwargs):
    checkpoint = _original_build_checkpoint_v124(*args, **kwargs)
    checkpoint.plume_state = _plume_state
    checkpoint.plume_rows = list(_plume_rows)
    return checkpoint


def _write_v125_outputs() -> None:
    if _plume_state is None or v124._mesh is None:
        return
    _output.mkdir(parents=True, exist_ok=True)
    if _plume_rows:
        with (_output / "mantle_plume_history.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(_plume_rows[0].keys()))
            writer.writeheader()
            writer.writerows(_plume_rows)
        save_plume_history(_plume_rows, _output / "mantle_plume_history.png")
    save_plume_maps(v124._mesh, _plume_state, _output, v124._dpi)

    summary_path = _output / base.SUMMARY_FILENAME
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        if _plume_rows:
            final = _plume_rows[-1]
            summary.update(
                {
                    "final_active_mantle_plumes": int(final["active_plume_count"]),
                    "final_max_mantle_plume_flux": float(final["max_surface_flux"]),
                    "final_plume_affected_surface_fraction": float(
                        final["affected_surface_area_fraction"]
                    ),
                    "final_plume_exposed_continental_fraction": float(
                        final["exposed_continental_material_fraction"]
                    ),
                    "cumulative_mean_plume_exposure_myr": float(
                        final["cumulative_mean_surface_exposure_myr"]
                    ),
                    "cumulative_max_plume_exposure_myr": float(
                        final["cumulative_max_surface_exposure_myr"]
                    ),
                }
            )
        summary.setdefault("notes", []).append(
            "v0.25 adds deterministic mantle-fixed plume heads. Overlying continental material is thermally rejuvenated, metasomatically refertilized and basally eroded; all forcing and response histories are checkpointed."
        )
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)


def main() -> None:
    base.RUN_VERSION = "0.25-mantle-plumes"
    base.RUN_MODEL = (
        "v0.24 cratonic memory plus mantle-fixed plume forcing, metasomatic "
        "refertilization, thermal rejuvenation and basal root erosion"
    )
    base.SUMMARY_FILENAME = "summary_v125.json"
    base.RUN_DESCRIPTION = "v0.25 mantle plumes and craton-root modification"
    base.DEFAULT_OUTPUT = "outputs_v125_plumes"
    base.parse_args = _parse_args_v125
    base.initialize_lithosphere = _initialize_lithosphere_v125
    base.load_checkpoint = _load_checkpoint_v125
    base.advance_lithosphere = v124._advance_lithosphere_v124
    base.advance_continental_cycle = _advance_continental_cycle_v125
    base.refresh_mechanical_lithosphere = v124._refresh_mechanical_lithosphere_v124
    base.build_checkpoint = _build_checkpoint_v125
    base.main()
    v124._write_v124_outputs()
    _write_v125_outputs()
    if _plume_rows:
        final = _plume_rows[-1]
        print(
            "v0.25 plumes: "
            f"active={final['active_plume_count']} | "
            f"max_flux={final['max_surface_flux']:.3f} | "
            f"exposed_continent={100.0*final['exposed_continental_material_fraction']:.2f}% | "
            f"cumulative_mean_exposure={final['cumulative_mean_surface_exposure_myr']:.2f} Myr"
        )


if __name__ == "__main__":
    main()
