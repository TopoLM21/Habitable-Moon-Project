#!/usr/bin/env python3
"""Moon Tectonics v0.26 plume-driven mechanical rifting runner.

v0.26 retains the complete v0.25 plume weakening model and adds a separate,
independently switchable mechanical forcing.  The forcing is injected through
the existing external continental-extension input; breakup and topology remain
owned by the stable lithosphere/topology implementation.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import run_long_evolution_v125 as v125
from tectonics.plume_rifting import (
    PlumeRiftingParameters,
    advance_plume_rifting,
    diagnose_plume_rifting,
    initialize_plume_rifting,
)
from visualization.plume_rifting import (
    build_plume_rifting_gif,
    save_plume_rifting_frame,
    save_plume_rifting_history,
    save_plume_rifting_maps,
)

v124 = v125.v124
base = v125.base

_original_parse_v125 = v125._parse_args_v125
_original_initialize_v125 = v125._initialize_lithosphere_v125
_original_load_v125 = v125._load_checkpoint_v125
_original_advance_cycle_v125 = v125._advance_continental_cycle_v125
_original_build_checkpoint_v125 = v125._build_checkpoint_v125

_params = PlumeRiftingParameters()
_state = None
_rows: list[dict] = []
_output = Path("outputs_v126_plume_rifting")
_dpi = 180
_frame_dpi = 105
_frame_interval = None
_finalize = False
_gif_frame_duration_ms = 350


def _row(diag) -> dict:
    return {name: getattr(diag, name) for name in diag.__dataclass_fields__}


def _parse_args_v126():
    global _params, _state, _rows, _output, _dpi, _frame_dpi
    global _frame_interval, _finalize, _gif_frame_duration_ms
    args = _original_parse_v125()
    config = base.load_config(args.config)
    _params = base.dc(PlumeRiftingParameters, config.get("plume_rifting", {}))
    _state = None
    _rows = []
    _output = Path(args.output)
    output_config = dict(config.get("output", {}))
    evolution_config = dict(config.get("evolution", {}))
    _dpi = int(output_config.get("dpi", 180))
    _frame_dpi = int(output_config.get("thermal_dpi", 105))
    _frame_interval = None if args.frame_interval is None else float(args.frame_interval)
    _finalize = bool(args.finalize)
    _gif_frame_duration_ms = int(evolution_config.get("gif_frame_duration_ms", 350))
    return args


def _initialize_lithosphere_v126(*args, **kwargs):
    global _state, _rows
    lithosphere = _original_initialize_v125(*args, **kwargs)
    mesh = args[0] if args else kwargs["mesh"]
    _state = initialize_plume_rifting(mesh, lithosphere.time_myr)
    _rows = [
        _row(
            diagnose_plume_rifting(
                mesh,
                lithosphere,
                _state,
                v124._radius_km,
                _params,
            )
        )
    ]
    return lithosphere


def _load_checkpoint_v126(path, manager):
    global _state, _rows
    checkpoint = _original_load_v125(path, manager)
    _state = checkpoint.plume_rifting_state
    if _state is None:
        _state = initialize_plume_rifting(v124._mesh, checkpoint.state.time_myr)
    _rows = list(checkpoint.plume_rifting_rows)
    if not _rows:
        _rows.append(
            _row(
                diagnose_plume_rifting(
                    v124._mesh,
                    checkpoint.state,
                    _state,
                    v124._radius_km,
                    _params,
                )
            )
        )
    return checkpoint


def _advance_lithosphere_v126(*args, **kwargs):
    global _state
    mesh = args[0] if args else kwargs["mesh"]
    system = args[1] if len(args) > 1 else kwargs["initial_system"]
    lithosphere = args[2] if len(args) > 2 else kwargs["state"]
    dt_myr = float(args[3] if len(args) > 3 else kwargs["dt_myr"])
    radius_km = float(args[4] if len(args) > 4 else kwargs["radius_km"])
    if _state is None:
        _state = initialize_plume_rifting(mesh, lithosphere.time_myr)
    if v125._plume_state is None:
        raise RuntimeError("v0.26 requires an initialized v0.25 plume population")

    _state, plume_forcing, diagnostics = advance_plume_rifting(
        mesh,
        lithosphere,
        v125._plume_state,
        _state,
        dt_myr,
        radius_km,
        _params,
    )
    _rows.append(_row(diagnostics))

    existing = kwargs.get("continental_extension_external_forcing")
    if existing is None:
        combined = plume_forcing
    else:
        existing_array = np.asarray(existing, dtype=np.float64)
        if existing_array.shape != (mesh.cell_count,):
            raise ValueError("existing external forcing must have shape (cell_count,)")
        combined = np.maximum(existing_array, plume_forcing)
    kwargs["continental_extension_external_forcing"] = combined
    result = v124._advance_lithosphere_v124(*args, **kwargs)

    new_lithosphere = result[0]
    if (
        _frame_interval is not None
        and _frame_interval > 0.0
        and abs(
            new_lithosphere.time_myr / _frame_interval
            - round(new_lithosphere.time_myr / _frame_interval)
        )
        < 1.0e-9
    ):
        save_plume_rifting_frame(
            mesh,
            new_lithosphere,
            v125._plume_state,
            _state,
            len(system.plates),
            _output
            / "plume_rift_frames"
            / f"plume_rift_{new_lithosphere.time_myr:08.1f}_Myr.png",
            _frame_dpi,
        )
    return result


def _build_checkpoint_v126(*args, **kwargs):
    checkpoint = _original_build_checkpoint_v125(*args, **kwargs)
    checkpoint.plume_rifting_state = _state
    checkpoint.plume_rifting_rows = list(_rows)
    return checkpoint


def _write_v126_outputs() -> None:
    if _state is None or v124._mesh is None:
        return
    _output.mkdir(parents=True, exist_ok=True)
    if _rows:
        with (_output / "plume_rifting_history.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(_rows[0].keys()))
            writer.writeheader()
            writer.writerows(_rows)
        save_plume_rifting_history(
            _rows, _output / "plume_rifting_history.png"
        )
    save_plume_rifting_maps(v124._mesh, _state, _output, _dpi)

    frames = sorted((_output / "plume_rift_frames").glob("plume_rift_*_Myr.png"))
    if _finalize:
        build_plume_rifting_gif(
            frames,
            _output / "plume_rift_history.gif",
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
                    "plume_rifting_enabled": bool(final["enabled"]),
                    "final_max_plume_extension_forcing": float(
                        final["max_surface_extension_forcing"]
                    ),
                    "final_mean_continental_plume_extension_forcing": float(
                        final["mean_continental_extension_forcing"]
                    ),
                    "final_plume_forced_continental_fraction": float(
                        final["forced_continental_material_fraction"]
                    ),
                    "cumulative_mean_plume_extension_impulse_myr": float(
                        final["cumulative_mean_extension_impulse_myr"]
                    ),
                    "cumulative_max_plume_extension_impulse_myr": float(
                        final["cumulative_max_extension_impulse_myr"]
                    ),
                    "final_max_diagnostic_plume_uplift_m": float(
                        final["max_dynamic_uplift_m"]
                    ),
                    "final_max_diagnostic_magmatic_productivity": float(
                        final["max_magmatic_productivity"]
                    ),
                }
            )
        summary.setdefault("notes", []).append(
            "v0.26 adds an independently switchable plume-head mechanical extension field. It combines broad radial-flow/dynamic-uplift forcing with flank localization and passes the result through the existing progressive continental-rift and breakup solver. Dynamic uplift and magmatic productivity remain diagnostic only."
        )
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)


def main() -> None:
    base.RUN_VERSION = "0.26-plume-rifting"
    base.RUN_MODEL = (
        "v0.25 mantle-plume weakening plus independently switchable radial "
        "plume-head extension and flank strain localization"
    )
    base.SUMMARY_FILENAME = "summary_v126.json"
    base.RUN_DESCRIPTION = "v0.26 plume-driven mechanical continental rifting"
    base.DEFAULT_OUTPUT = "outputs_v126_plume_rifting"
    base.parse_args = _parse_args_v126
    base.initialize_lithosphere = _initialize_lithosphere_v126
    base.load_checkpoint = _load_checkpoint_v126
    base.advance_lithosphere = _advance_lithosphere_v126
    base.advance_continental_cycle = _original_advance_cycle_v125
    base.refresh_mechanical_lithosphere = v124._refresh_mechanical_lithosphere_v124
    base.build_checkpoint = _build_checkpoint_v126
    base.main()
    v124._write_v124_outputs()
    v125._write_v125_outputs()
    _write_v126_outputs()
    if _rows:
        final = _rows[-1]
        print(
            "v0.26 plume rifting: "
            f"enabled={final['enabled']} | "
            f"max_forcing={final['max_surface_extension_forcing']:.3f} | "
            f"forced_continent={100.0*final['forced_continental_material_fraction']:.2f}% | "
            f"max_impulse={final['cumulative_max_extension_impulse_myr']:.2f} Myr"
        )


if __name__ == "__main__":
    main()
