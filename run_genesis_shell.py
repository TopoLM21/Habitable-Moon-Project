"""Run passive local solidification and membrane damage driven by thermal genesis."""
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

from tectonics.genesis import (advance, diagnose, initial_state, parameters_from_config,
                              temperatures)
from tectonics.genesis_shell import (SHELL_VERSION, Membrane, advance_shell, build_icosphere,
                                     diagnose_shell, initialize_shell, load_shell_checkpoint,
                                     save_shell_checkpoint, shell_fields, shell_parameters_from_config)
from tectonics.simulation import load_config

ROOT = Path(__file__).resolve().parent


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs"/"genesis_moon.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-myr", type=float, default=3.)
    parser.add_argument("--sample-interval-myr", type=float, default=0.1)
    parser.add_argument("--max-step-myr", type=float, default=0.01)
    parser.add_argument("--shell-step-myr", type=float, default=0.002)
    parser.add_argument("--subdivisions", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--control", choices=["intact"])
    parser.add_argument("--stellar-flux-w-m2", type=float)
    parser.add_argument("--water-volume-km3", type=float)
    parser.add_argument("--initial-temperature-k", type=float)
    parser.add_argument("--convective-traction-mpa", type=float)
    parser.add_argument("--resume", type=Path, help="Joint shell_checkpoint.npz from an earlier shell run")
    parser.add_argument("--no-frames", action="store_true", help="Save final map only, skip animation")
    args = parser.parse_args(argv)
    controls = {"sample_interval_myr": args.sample_interval_myr, "max_step_myr": args.max_step_myr,
                "shell_step_myr": args.shell_step_myr}
    try:
        if not all(math.isfinite(v) and v > 0 for v in (args.duration_myr, *controls.values())):
            raise ValueError("Finite positive time controls required")
        # A fixed mechanics grid gives unambiguous continuation, independent of output frequency.
        if any(round(v/args.shell_step_myr) < 1 or abs(v/args.shell_step_myr-round(v/args.shell_step_myr)) > 1e-7
               for v in (args.duration_myr, args.sample_interval_myr)):
            raise ValueError("End time and sample interval must be multiples of shell-step-myr")
        if args.resume:
            if any(v is not None for v in (args.control, args.subdivisions, args.stellar_flux_w_m2,
                                           args.water_volume_km3, args.initial_temperature_k, args.convective_traction_mpa)):
                raise ValueError("Resume must retain saved physics and mesh")
            shell, global_state, p, thermal, saved = load_shell_checkpoint(args.resume)
            if controls != saved["controls"]:
                raise ValueError("Resume must retain saved time controls")
            provenance = dict(saved["provenance"], resume_source=str(args.resume.resolve()))
        else:
            config = load_config(args.config)
            thermal = parameters_from_config(config)
            p = shell_parameters_from_config(config)
            overrides = {k: getattr(args, k) for k in ("stellar_flux_w_m2", "water_volume_km3", "initial_temperature_k") if getattr(args, k) is not None}
            if "initial_temperature_k" in overrides:
                overrides["initial_surface_temperature_k"] = overrides["initial_temperature_k"]
            thermal = replace(thermal, **overrides)
            if args.subdivisions is not None:
                p = replace(p, subdivisions=args.subdivisions)
            if args.convective_traction_mpa is not None:
                p = replace(p, convective_traction_pa=args.convective_traction_mpa*1e6)
            if args.control == "intact":
                p = replace(p, initial_temperature_anomaly_k=0., convective_traction_pa=0.)
            thermal.validate()
            p.validate(thermal)
            global_state = initial_state(thermal)
            shell = None
            provenance = {"config_path": str(args.config.resolve()),
                          "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                          "water_inventory_source": config["genesis"].get("water_inventory_source", "explicit inventory"),
                          "stellar_flux_source": config["genesis"].get("stellar_flux_source", "explicit flux"),
                          "thermal_overrides": overrides, "control": args.control,
                          "coupling": "one_way_thermal_boundary_forcing; no feedback to global thermal inventory"}
            if "water_volume_km3" in overrides:
                provenance["water_inventory_source"] = "explicit CLI override"
            if "stellar_flux_w_m2" in overrides:
                provenance["stellar_flux_source"] = "explicit CLI override"
        if args.duration_myr <= global_state.time_myr or global_state.stopped_reason or (shell is not None and shell.stopped_reason):
            raise ValueError("End time must follow a checkpoint that has not reached a model limit")
        if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
            raise ValueError("Output must be new or empty; existing results are protected")
        mesh = build_icosphere(p.subdivisions)
        membrane = Membrane(mesh, p.poisson_ratio)
        shell = shell or initialize_shell(mesh, p, thermal)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/"parameters.json").write_text(json.dumps({"model_version": SHELL_VERSION,
            "thermal": asdict(thermal), "shell": asdict(p), "controls": controls, "provenance": provenance}, indent=2)+"\n", encoding="utf-8")
        from visualization.genesis_shell import save_shell_snapshot, save_shell_animation
        from visualization.genesis import save_genesis_history
        global_rows = [diagnose(global_state, thermal)]
        tm, ts = temperatures(np.asarray(global_state.energy), thermal)
        shell_rows = [diagnose_shell(shell, mesh, p, thermal, ts, tm)]
        frames = []
        sample_every = round(args.sample_interval_myr/args.shell_step_myr)
        frame_every = max(1, math.ceil((args.duration_myr-global_state.time_myr)/args.sample_interval_myr/30))
        last_frame_sample = 0

        def save_state():
            save_shell_checkpoint(args.output/"shell_checkpoint.npz", shell, global_state, p, thermal, controls, provenance)

        def frame(sample_index):
            nonlocal last_frame_sample
            if args.no_frames:
                return
            last_frame_sample = sample_index
            directory = args.output/"shell_frames"
            directory.mkdir(exist_ok=True)
            path = directory/f"shell_{sample_index:06d}.png"
            save_shell_snapshot(mesh, shell_fields(shell, mesh, p, thermal, ts, tm), path, shell.time_myr, shell_rows[-1])
            frames.append(path)

        save_state()
        frame(0)
        with (args.output/"history.csv").open("w", newline="", encoding="utf-8") as global_file, (args.output/"shell_history.csv").open("w", newline="", encoding="utf-8") as shell_file:
            gw = csv.DictWriter(global_file, fieldnames=list(global_rows[0]))
            sw = csv.DictWriter(shell_file, fieldnames=list(shell_rows[0]))
            gw.writeheader(); sw.writeheader()
            gw.writerow(global_rows[0]); sw.writerow(shell_rows[0])
            start_index = round(global_state.time_myr/args.shell_step_myr)
            end_index = round(args.duration_myr/args.shell_step_myr)
            for index in range(start_index+1, end_index+1):
                old_tm, old_ts = tm, ts
                global_state, event_rows = advance(global_state, thermal, index*args.shell_step_myr, args.max_step_myr)
                tm, ts = temperatures(np.asarray(global_state.energy), thermal)
                shell = advance_shell(shell, mesh, membrane, p, thermal, global_state.time_myr, old_ts, ts, old_tm, tm)
                # Capture thermal events, but don't force output times into the mechanical grid.
                for row in event_rows[:-1]:
                    global_rows.append(row); gw.writerow(row)
                if index % sample_every == 0 or index == end_index or global_state.stopped_reason or shell.stopped_reason:
                    global_rows.append(diagnose(global_state, thermal))
                    shell_rows.append(diagnose_shell(shell, mesh, p, thermal, ts, tm))
                    gw.writerow(global_rows[-1]); sw.writerow(shell_rows[-1])
                    global_file.flush(); shell_file.flush()
                    save_state()
                    number = len(shell_rows)-1
                    if number % frame_every == 0:
                        frame(number)
                    row = shell_rows[-1]
                    print(f"t={shell.time_myr:.6f} Myr | shell={100*row['solid_surface_fraction']:.1f}% | damage={100*row['damaged_area_fraction']:.2f}% | peak stress={row['peak_tensile_stress_mpa']:.2f} MPa", flush=True)
                if global_state.stopped_reason or shell.stopped_reason:
                    break
        summary = {"model_version": SHELL_VERSION, "status": global_state.stopped_reason or shell.stopped_reason or "completed",
                   "requested_end_time_myr": args.duration_myr, "events_myr": global_state.events,
                   "final": global_rows[-1], "shell": shell_rows[-1],
                   "first_fracture_time_resolution_myr": args.shell_step_myr,
                   "max_relative_column_energy_residual": max(abs(r["relative_column_energy_residual"]) for r in shell_rows),
                   "max_mechanical_equilibrium_residual": max(r["mechanical_equilibrium_residual"] for r in shell_rows),
                   "limitations": ["One-way thermal boundary forcing; column energy is a separate response ledger.",
                                   "Small-strain membrane with diffuse isotropic damage, no discrete crack opening/contact.",
                                   "Outlined damage patches are not mobile plate boundaries or measured fault lengths.",
                                   "No continental differentiation, orbital evolution or v0.31 handoff."], "provenance": provenance}
        (args.output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        fields = shell_fields(shell, mesh, p, thermal, ts, tm)
        np.savez_compressed(args.output/"shell_fields.npz", vertices=mesh.vertices, faces=mesh.faces,
                            centroids=mesh.centroids, **fields)
        save_shell_snapshot(mesh, fields, args.output/"genesis_shell.png", shell.time_myr, shell_rows[-1])
        if not args.no_frames:
            if last_frame_sample != len(shell_rows)-1:
                frame(len(shell_rows)-1)
            save_shell_animation(frames, args.output/"genesis_shell.gif")
        save_genesis_history(global_rows, args.output/"genesis_history.png", global_state.events)
        print(f"GENESIS_COMPLETE {args.output.resolve()}", flush=True)
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Genesis shell error: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
