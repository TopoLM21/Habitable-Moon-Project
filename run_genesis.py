"""Run the independent magma/steam/ocean experiment (times are in Myr)."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

from tectonics.genesis import (MODEL_VERSION, diagnose, initial_state, load_checkpoint,
                              parameters_from_config, run_genesis, save_checkpoint)
from tectonics.simulation import load_config

ROOT = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "genesis_moon.yaml")
    parser.add_argument("--output", type=Path, required=True, help="New or empty result directory")
    parser.add_argument("--duration-myr", type=float, default=10.0, help="Absolute end time since molten start")
    parser.add_argument("--sample-interval-myr", type=float, default=0.1)
    parser.add_argument("--max-step-myr", type=float, default=0.01)
    parser.add_argument("--stellar-flux-w-m2", type=float)
    parser.add_argument("--water-volume-km3", type=float)
    parser.add_argument("--initial-temperature-k", type=float)
    parser.add_argument("--resume", type=Path, help="Thermal genesis checkpoint.json; output must be a new directory")
    args = parser.parse_args(argv)
    controls = {"sample_interval_myr": args.sample_interval_myr, "max_step_myr": args.max_step_myr}
    try:
        if args.resume:
            if any(v is not None for v in (args.stellar_flux_w_m2, args.water_volume_km3, args.initial_temperature_k)):
                raise ValueError("Resume uses saved physics; overrides require a new experiment")
            state, p, saved = load_checkpoint(args.resume)
            if saved.get("controls") != controls:
                raise ValueError("Resume must keep saved sampling interval and max step")
            provenance = dict(saved.get("provenance", {}))
            provenance["resume_source"] = str(args.resume.resolve())
        else:
            config = load_config(args.config)
            p = parameters_from_config(config)
            overrides = {}
            for key in ("stellar_flux_w_m2", "water_volume_km3", "initial_temperature_k"):
                if getattr(args, key) is not None:
                    overrides[key] = getattr(args, key)
            if "initial_temperature_k" in overrides:
                overrides["initial_surface_temperature_k"] = overrides["initial_temperature_k"]
            p = replace(p, **overrides)
            p.validate()
            state = initial_state(p)
            provenance = {"config_path": str(args.config.resolve()),
                          "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                          "water_inventory_source": config["genesis"].get("water_inventory_source", "explicit parameter"),
                          "stellar_flux_source": config["genesis"].get("stellar_flux_source", "explicit parameter"),
                          "cli_overrides": overrides}
            if "water_volume_km3" in overrides:
                provenance["water_inventory_source"] = "explicit CLI water-volume override"
            if "stellar_flux_w_m2" in overrides:
                provenance["stellar_flux_source"] = "explicit CLI stellar-flux override"
        import math
        if not all(math.isfinite(v) and v > 0 for v in (args.duration_myr, *controls.values())):
            raise ValueError("Time controls must be finite and positive")
        if args.duration_myr <= state.time_myr or state.stopped_reason:
            raise ValueError("End time must follow a checkpoint that has not reached a model limit")
        if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
            raise ValueError("Output must be new or empty; existing runs will not be overwritten")
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "parameters.json").write_text(json.dumps({"model_version": MODEL_VERSION,
             "parameters": asdict(p), "controls": controls, "provenance": provenance}, indent=2) + "\n", encoding="utf-8")
        history = [diagnose(state, p)]
        save_checkpoint(args.output / "checkpoint.json", state, p, controls, provenance)
        with (args.output / "history.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerow(history[0])
            handle.flush()
            def sample(current, new_rows):
                writer.writerows(new_rows)
                handle.flush()
                save_checkpoint(args.output / "checkpoint.json", current, p, controls, provenance)
                row = new_rows[-1]
                print(f"t={current.time_myr:.6f} Myr | mantle={row['mantle_temperature_k']:.1f} K | "
                      f"surface={row['surface_temperature_k']:.1f} K | water={100*row['ocean_fraction']:.1f}% ocean", flush=True)
            state, history = run_genesis(p, args.duration_myr, state=state, on_sample=sample, **controls)
        summary = {"model_version": MODEL_VERSION, "status": state.stopped_reason or "completed",
                   "requested_end_time_myr": args.duration_myr, "events_myr": state.events,
                   "final": history[-1], "max_relative_energy_residual": max(abs(row["relative_energy_residual"]) for row in history),
                   "limitations": ["Parameterised atmosphere and interior heat transfer; not a calibrated climate prediction.",
                                   "No orbital/spin evolution, atmospheric escape, dissolved mantle water or ice.",
                                   "Near-critical water partition is regularised; not a supercritical equation of state.",
                                   "No mobile plates, continental crust, spatial fractures or mature-tectonics handoff."],
                   "provenance": provenance}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        from visualization.genesis import save_genesis_history
        save_genesis_history(history, args.output / "genesis_history.png", state.events)
        print(f"GENESIS_COMPLETE {args.output.resolve()}", flush=True)
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Genesis error: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
