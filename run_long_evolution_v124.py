#!/usr/bin/env python3
"""Moon Tectonics v0.24 cratonic-memory long runner.

v0.24 deliberately layers its new state transition around the stable v0.23
integration loop instead of duplicating that long runner.  The wrappers below
activate continuous continental-lithosphere age, mantle depletion and craton
strength, preserve the fields in checkpoints, and add v0.24 diagnostics.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import run_long_evolution_v123 as base
from tectonics.cratons import (
    CratonParameters,
    advance_craton_memory,
    diagnose_craton_memory,
    initialize_craton_memory,
)
from visualization.cratons import save_craton_history, save_craton_maps


_original_parse_args = base.parse_args
_original_initialize_lithosphere = base.initialize_lithosphere
_original_load_checkpoint = base.load_checkpoint
_original_advance_lithosphere = base.advance_lithosphere
_original_advance_continental_cycle = base.advance_continental_cycle
_original_refresh_mechanical_lithosphere = base.refresh_mechanical_lithosphere
_original_build_checkpoint = base.build_checkpoint

_params = CratonParameters()
_mechanical_config: dict = {}
_craton_rows: list[dict] = []
_mesh = None
_last_state = None
_radius_km = 5287.0
_output = Path("outputs_v124_cratons")
_dpi = 180


def _row(diag) -> dict:
    return {name: getattr(diag, name) for name in diag.__dataclass_fields__}


def _parse_args_v124():
    global _params, _mechanical_config, _craton_rows, _mesh
    global _radius_km, _output, _dpi, _last_state
    args = _original_parse_args()
    config = base.load_config(args.config)
    _params = base.dc(CratonParameters, config.get("cratons", {}))
    _mechanical_config = dict(config.get("mechanical_lithosphere", {}))
    _radius_km = float(config["moon"]["radius_km"])
    _output = Path(args.output)
    _dpi = int(config.get("output", {}).get("dpi", 180))
    _mesh = base.build_prototype(config).mesh
    _craton_rows = []
    _last_state = None
    return args


def _initialize_lithosphere_v124(*args, **kwargs):
    global _last_state, _craton_rows
    state = _original_initialize_lithosphere(*args, **kwargs)
    mesh = args[0] if args else kwargs["mesh"]
    initialize_craton_memory(mesh, state, _radius_km, _params)
    _last_state = state
    _craton_rows = [_row(diagnose_craton_memory(mesh, state, _radius_km, _params))]
    return state


def _load_checkpoint_v124(path, manager):
    global _last_state, _craton_rows
    checkpoint = _original_load_checkpoint(path, manager)
    initialize_craton_memory(_mesh, checkpoint.state, _radius_km, _params)
    _last_state = checkpoint.state
    _craton_rows = list(checkpoint.craton_rows)
    if not _craton_rows:
        _craton_rows.append(
            _row(diagnose_craton_memory(_mesh, checkpoint.state, _radius_km, _params))
        )
    return checkpoint


def _advance_lithosphere_v124(*args, **kwargs):
    kwargs.setdefault("craton_extension_resistance_gain", float(_params.extension_resistance_gain))
    kwargs.setdefault("craton_min_extension_factor", float(_params.minimum_extension_factor))
    return _original_advance_lithosphere(*args, **kwargs)


def _advance_continental_cycle_v124(*args, **kwargs):
    global _last_state
    lithosphere = args[1] if len(args) > 1 else kwargs["lithosphere"]
    before_volume = (
        np.zeros(len(lithosphere.crust_age_myr), dtype=np.float64)
        if lithosphere.continental_volume_km3 is None
        else np.asarray(lithosphere.continental_volume_km3, dtype=np.float64).copy()
    )
    state, cycle, diagnostics = _original_advance_continental_cycle(*args, **kwargs)
    mesh = args[0] if args else kwargs["mesh"]
    dt_myr = args[4] if len(args) > 4 else kwargs["dt_myr"]
    radius_km = args[5] if len(args) > 5 else kwargs["radius_km"]
    state, craton_diagnostics = advance_craton_memory(
        mesh,
        state,
        float(dt_myr),
        float(radius_km),
        _params,
        pre_cycle_continental_volume_km3=before_volume,
    )
    _craton_rows.append(_row(craton_diagnostics))
    _last_state = state
    return state, cycle, diagnostics


def _refresh_mechanical_lithosphere_v124(*args, **kwargs):
    kwargs.setdefault(
        "craton_root_thickening_km",
        float(_mechanical_config.get("craton_root_thickening_km", 75.0)),
    )
    kwargs.setdefault(
        "craton_depletion_density_reduction_kg_m3",
        float(_mechanical_config.get("craton_depletion_density_reduction_kg_m3", 28.0)),
    )
    return _original_refresh_mechanical_lithosphere(*args, **kwargs)


def _build_checkpoint_v124(*args, **kwargs):
    checkpoint = _original_build_checkpoint(*args, **kwargs)
    checkpoint.craton_rows = list(_craton_rows)
    return checkpoint


def _write_v124_outputs() -> None:
    if _last_state is None or _mesh is None:
        return
    _output.mkdir(parents=True, exist_ok=True)
    history_path = _output / "craton_history.csv"
    if _craton_rows:
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(_craton_rows[0].keys()))
            writer.writeheader()
            writer.writerows(_craton_rows)
        save_craton_history(_craton_rows, _output / "craton_history.png")
    save_craton_maps(_mesh, _last_state, _radius_km, _output, _dpi)

    summary_path = _output / base.SUMMARY_FILENAME
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        if _craton_rows:
            final = _craton_rows[-1]
            summary.update(
                {
                    "final_mean_continental_lithosphere_age_myr": float(
                        final["mean_continental_lithosphere_age_myr"]
                    ),
                    "final_max_continental_lithosphere_age_myr": float(
                        final["max_continental_lithosphere_age_myr"]
                    ),
                    "final_mean_mantle_depletion_fraction": float(
                        final["mean_mantle_depletion_fraction"]
                    ),
                    "final_mean_craton_strength": float(final["mean_craton_strength"]),
                    "final_max_craton_strength": float(final["max_craton_strength"]),
                    "final_cratonic_area_fraction_of_surface": float(
                        final["cratonic_area_fraction_of_surface"]
                    ),
                    "final_cratonic_fraction_of_continental_material": float(
                        final["cratonic_fraction_of_continental_material"]
                    ),
                    "final_mean_craton_extension_factor": float(
                        final["mean_extension_factor_continental"]
                    ),
                }
            )
        summary.setdefault("notes", []).append(
            "v0.24 transports continuous continental-lithosphere age and mantle-depletion memory, derives cratonic strength, grows thicker buoyant roots, redirects rifting toward weak belts, and rejuvenates roots under sustained extension or thermal weakening."
        )
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)


def main() -> None:
    base.RUN_VERSION = "0.24-cratonic-memory"
    base.RUN_MODEL = (
        "v0.23 conservative sediments plus transported continental-lithosphere "
        "age, mantle depletion, cratonic strength, root buoyancy and rift resistance"
    )
    base.SUMMARY_FILENAME = "summary_v124.json"
    base.RUN_DESCRIPTION = "v0.24 continental-lithosphere maturation and cratonic memory"
    base.DEFAULT_OUTPUT = "outputs_v124_cratons"
    base.parse_args = _parse_args_v124
    base.initialize_lithosphere = _initialize_lithosphere_v124
    base.load_checkpoint = _load_checkpoint_v124
    base.advance_lithosphere = _advance_lithosphere_v124
    base.advance_continental_cycle = _advance_continental_cycle_v124
    base.refresh_mechanical_lithosphere = _refresh_mechanical_lithosphere_v124
    base.build_checkpoint = _build_checkpoint_v124
    base.main()
    _write_v124_outputs()
    if _craton_rows:
        final = _craton_rows[-1]
        print(
            "v0.24 cratons: "
            f"mean_strength={final['mean_craton_strength']:.3f} | "
            f"cratonic_continent={100.0*final['cratonic_fraction_of_continental_material']:.2f}% | "
            f"mean_root_age={final['mean_continental_lithosphere_age_myr']:.1f} Myr"
        )


if __name__ == "__main__":
    main()
