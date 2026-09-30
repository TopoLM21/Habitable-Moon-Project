"""Run the coarse molten-start experiment up to a first candidate partition."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

from tectonics.genesis import GenesisParameters, parameters_from_config
from tectonics.genesis_checkpoint_compat import MODEL_VERSION, require_thermal_model_version
from tectonics.genesis_shell import ShellParameters, shell_parameters_from_config
from tectonics.genesis_starter import StarterModel, StarterParameters
from tectonics.genesis_tides import TidalParameters, tidal_parameters_from_config
from tectonics.mesh import build_icosphere
from tectonics.simulation import load_config

ROOT = Path(__file__).resolve().parent
FORMAT = "genesis-starter-run-0.1"
LIMITATIONS = [
    "Coarse parameterized loading and damage; not a resolved fracture-mechanics solution.",
    "A partition is a candidate initial state, not evidence of persistent mobile plates.",
    "Starter stops at the first partition; an explicitly requested mature continuation is a separate approximate-model stage.",
    "Flat reference sphere with no prescribed continents, relief, or chemical-crust differentiation.",
    "No mature-model reservoir transfer, volcanic history, or petrological reconstruction is certified.",
]


def _write_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _resume_model(path):
    saved = json.loads((path.parent / "parameters.json").read_text(encoding="utf-8"))
    if saved.get("format") != FORMAT:
        raise ValueError("Resume requires the starter parameters.json alongside its checkpoint")
    require_thermal_model_version(saved)
    thermal = GenesisParameters(**saved["thermal"])
    tides = TidalParameters(**saved["tides"])
    shell = ShellParameters(**saved["shell"])
    parameters = StarterParameters(**saved["starter"])
    model = StarterModel(build_icosphere(shell.subdivisions), thermal, tides, shell, parameters)
    return model, model.load_state(path), saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "genesis_moon.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-myr", type=float, default=20., help="Absolute end age after molten start, also on resume")
    parser.add_argument("--step-myr", type=float, default=1., help="Output/checkpoint interval; internal loading steps may be shorter")
    parser.add_argument("--subdivisions", type=int, choices=[1, 2, 3, 4], default=None, help="Default: 4 (5120 cells)")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--convective-traction-mpa", type=float)
    parser.add_argument("--no-tides", action="store_true")
    parser.add_argument("--no-water-weakening", action="store_true")
    parser.add_argument("--control", choices=["intact"])
    parser.add_argument("--resume", type=Path, help="starter_checkpoint.npz with its sibling parameters.json")
    parser.add_argument("--continue-myr", type=float, default=0., help="Opt-in mature continuation after a first partition; elapsed Myr, default disabled")
    parser.add_argument("--continuation-step-myr", type=float, default=1.)
    args = parser.parse_args(argv)
    try:
        if not all(math.isfinite(x) and x > 0 for x in (args.duration_myr, args.step_myr)):
            raise ValueError("Duration and output step must be finite and positive")
        if (not math.isfinite(args.continue_myr) or args.continue_myr < 0
                or not math.isfinite(args.continuation_step_myr) or args.continuation_step_myr <= 0):
            raise ValueError("Continuation duration must be nonnegative and its step finite and positive")
        if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
            raise ValueError("Output must be new or empty; existing results are protected")
        if args.resume:
            if (args.subdivisions is not None or args.seed is not None or args.convective_traction_mpa is not None
                    or args.no_tides or args.no_water_weakening or args.control):
                raise ValueError("Resume must retain the saved physics and mesh")
            model, state, saved = _resume_model(args.resume)
            provenance = dict(saved["provenance"], resume_source=str(args.resume.resolve()))
        else:
            config = load_config(args.config)
            thermal = parameters_from_config(config)
            tides = tidal_parameters_from_config(config, thermal)
            shell = replace(shell_parameters_from_config(config), subdivisions=args.subdivisions or 4)
            parameters = StarterParameters()
            if args.seed is not None:
                parameters = replace(parameters, seed=args.seed)
            if args.convective_traction_mpa is not None:
                shell = replace(shell, convective_traction_pa=args.convective_traction_mpa * 1e6)
            if args.no_tides:
                tides = replace(tides, enabled=False)
            if args.no_water_weakening:
                parameters = replace(parameters, water_weakening=False)
            if args.control == "intact":
                shell = replace(shell, convective_traction_pa=0.)
                parameters = replace(parameters, cooling_contrast_fraction=0., tidal_mechanics=False)
            model = StarterModel(build_icosphere(shell.subdivisions), thermal, tides, shell, parameters)
            state = model.initial_state()
            provenance = {
                "config_path": str(args.config.resolve()),
                "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "control": args.control,
                "control_description": ("No mantle, differential-cooling, or tidal mechanical loading; orbital heating remains enabled unless --no-tides is supplied."
                                        if args.control == "intact" else None),
            }
        if state.stopped_reason or args.duration_myr <= state.time_myr:
            raise ValueError("End age must follow a checkpoint that has not already stopped")
        args.output.mkdir(parents=True, exist_ok=True)
        metadata = {"format": FORMAT, "thermal_model_version": MODEL_VERSION,
                    "thermal": asdict(model.thermal), "tides": asdict(model.tides),
                    "shell": asdict(model.shell), "starter": asdict(model.parameters),
                    "step_myr": args.step_myr, "provenance": provenance}
        _write_json(args.output / "parameters.json", metadata)
        history = []

        def save_progress():
            row = model.diagnose(state)
            history.append(row)
            model.save_state(args.output / "starter_checkpoint.npz", state)
            result = {"format": FORMAT, "status": state.stopped_reason or ("completed" if state.time_myr >= args.duration_myr else "running"),
                      "requested_end_time_myr": args.duration_myr, "final": row,
                      "events": state.events, "mature_handoff": False,
                      "candidate_partition": state.stopped_reason == "first_partition",
                      "limitations": LIMITATIONS, "provenance": provenance}
            _write_json(args.output / "summary.json", result)
            return row, result

        row, summary = save_progress()
        with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
            handle.flush()
            while state.time_myr < args.duration_myr and not state.stopped_reason:
                previous = state.time_myr
                target = min(args.duration_myr, previous + args.step_myr)
                state = model.advance(state, target)
                if not math.isfinite(state.time_myr) or state.time_myr <= previous or state.time_myr > target + 1e-12:
                    raise RuntimeError("Starter failed to advance its physical clock within the requested interval")
                row, summary = save_progress()
                writer.writerow(row)
                handle.flush()
                print(f"t={state.time_myr:.6f} Myr | domains={row['domain_count']} | damaged area={row['damaged_area_fraction']:.4%} | max yield ratio={row['max_yield_ratio']:.5g}", flush=True)
        np.savez_compressed(args.output / "starter_fields.npz", vertices=model.mesh.vertices,
                            faces=model.mesh.faces, centroids=model.mesh.centroids, damage=state.damage,
                            yield_ratio=state.yield_ratio, domain_id=state.system.cell_plate)
        from visualization.genesis_starter import save_starter_snapshot
        save_starter_snapshot(model.mesh, state, args.output / "genesis_starter.png", history, summary)
        print(f"GENESIS_STARTER_COMPLETE {args.output.resolve()}", flush=True)
        if args.continue_myr > 0:
            continuation_output = args.output / "continuation"
            continuation = {"requested_duration_myr": args.continue_myr,
                            "step_myr": args.continuation_step_myr,
                            "output": str(continuation_output.resolve()),
                            "status": "running" if state.stopped_reason == "first_partition" else "skipped_no_partition"}
            summary["continuation"] = continuation
            _write_json(args.output / "summary.json", summary)
            if state.stopped_reason == "first_partition":
                print("GENESIS_STARTER_CONTINUATION_START", flush=True)
                from run_genesis_starter_continuation import run_continuation
                try:
                    report = run_continuation(args.output / "starter_checkpoint.npz", continuation_output,
                                              args.continue_myr, args.continuation_step_myr)
                except Exception as exc:
                    continuation.update(status="failed", error=str(exc))
                    _write_json(args.output / "summary.json", summary)
                    raise
                continuation.update(status="completed", report=report)
                _write_json(args.output / "summary.json", summary)
                print(f"GENESIS_STARTER_CONTINUATION_COMPLETE {continuation_output.resolve()}", flush=True)
            else:
                print("GENESIS_STARTER_CONTINUATION_SKIPPED no_partition", flush=True)
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        print(f"Genesis starter error: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
