"""Run finite-width fault-zone friction and irreversible slip on a material shell."""
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

from tectonics.genesis import parameters_from_config
from tectonics.genesis_shell import shell_parameters_from_config
from tectonics.genesis_onset import onset_parameters_from_config
from tectonics.genesis_mobile import mobile_parameters_from_config
from tectonics.genesis_faults import (FAULT_VERSION, FaultModel,
                                    load_fault_checkpoint, save_fault_checkpoint)
from tectonics.genesis_fault_law import weak_plane_parameters_from_config
from tectonics.genesis_tides import tidal_parameters_from_config
from tectonics.simulation import load_config

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs"/"genesis_moon.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-myr", type=float, default=3.)
    parser.add_argument("--sample-interval-myr", type=float, default=.1)
    parser.add_argument("--max-step-myr", type=float, default=.01)
    parser.add_argument("--shell-step-myr", type=float, default=.002)
    parser.add_argument("--subdivisions", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--regularization-km", type=float)
    parser.add_argument("--control", choices=["intact"])
    parser.add_argument("--no-tides", action="store_true")
    parser.add_argument("--no-water-weakening", action="store_true")
    parser.add_argument("--no-fault-slip", action="store_true")
    parser.add_argument("--stellar-flux-w-m2", type=float)
    parser.add_argument("--water-volume-km3", type=float)
    parser.add_argument("--initial-temperature-k", type=float)
    parser.add_argument("--convective-traction-mpa", type=float)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-frames", action="store_true")
    args = parser.parse_args(argv)
    controls = {"sample_interval_myr": args.sample_interval_myr,
                "shell_step_myr": args.shell_step_myr, "max_step_myr": args.max_step_myr}
    try:
        if not all(math.isfinite(x) and x > 0 for x in (args.duration_myr, *controls.values())):
            raise ValueError("Finite positive time controls required")
        if any(round(x/args.shell_step_myr) < 1 or abs(x/args.shell_step_myr-round(x/args.shell_step_myr)) > 1e-7
               for x in (args.duration_myr, args.sample_interval_myr)):
            raise ValueError("End time and sample interval must be multiples of shell-step-myr")
        if args.resume:
            if args.no_tides or args.no_water_weakening or args.no_fault_slip or any(x is not None for x in
                (args.subdivisions, args.regularization_km, args.control, args.stellar_flux_w_m2,
                 args.water_volume_km3, args.initial_temperature_k, args.convective_traction_mpa)):
                raise ValueError("Resume must retain saved physics")
            model, state, global_state, orbit, saved = load_fault_checkpoint(args.resume)
            if controls != saved["controls"]:
                raise ValueError("Resume must retain saved time controls")
            provenance = dict(saved["provenance"], resume_source=str(args.resume.resolve()))
        else:
            config = load_config(args.config)
            thermal = parameters_from_config(config)
            shell_p = shell_parameters_from_config(config)
            onset_p = onset_parameters_from_config(config)
            mobile_p = mobile_parameters_from_config(config)
            fault_p = weak_plane_parameters_from_config(config)
            if args.no_fault_slip:
                fault_p = replace(fault_p, enabled=False)
            thermal_overrides = {k: getattr(args, k) for k in
                ("stellar_flux_w_m2", "water_volume_km3", "initial_temperature_k") if getattr(args, k) is not None}
            if "initial_temperature_k" in thermal_overrides:
                thermal_overrides["initial_surface_temperature_k"] = thermal_overrides["initial_temperature_k"]
            thermal = replace(thermal, **thermal_overrides)
            tides_p = tidal_parameters_from_config(config, thermal)
            if args.subdivisions is not None:
                shell_p = replace(shell_p, subdivisions=args.subdivisions)
            if args.convective_traction_mpa is not None:
                shell_p = replace(shell_p, convective_traction_pa=args.convective_traction_mpa*1e6)
            if args.regularization_km is not None:
                onset_p = replace(onset_p, regularization_km=args.regularization_km)
            if args.no_water_weakening:
                onset_p = replace(onset_p, water_weakening=False)
            if args.no_tides or args.control == "intact":
                tides_p = replace(tides_p, enabled=False)
            if args.control == "intact":
                shell_p = replace(shell_p, initial_temperature_anomaly_k=0., convective_traction_pa=0.)
            model = FaultModel(shell_p, thermal, onset_p, tides_p, mobile_p, fault_p)
            state, global_state, orbit = model.initial()
            provenance = {"config_path": str(args.config.resolve()),
                "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "orbit_source": config.get("genesis_tides", {}).get("orbit_source", "explicit tidal parameters"),
                "water_inventory_source": ("explicit CLI override" if args.water_volume_km3 is not None
                    else config["genesis"].get("water_inventory_source", "explicit inventory")),
                "thermal_overrides": thermal_overrides, "control": args.control,
                "coupling": "moving material shell with finite-width frictional slip; isolated eccentricity orbit energy to global heat"}
        if args.duration_myr <= global_state.time_myr or global_state.stopped_reason or state.stopped_reason:
            raise ValueError("End time must follow a checkpoint that has not reached a model limit")
        if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
            raise ValueError("Output must be new or empty; existing results are protected")
        args.output.mkdir(parents=True, exist_ok=True)
        parameters = {"model_version": FAULT_VERSION, "geometry": "material",
            "thermal": asdict(model.thermal), "shell": asdict(model.p),
            "onset": asdict(model.onset_p), "tides": asdict(model.tides_p), "mobile": asdict(model.mobile_p),
            "faults": asdict(model.fault_p),
            "controls": controls, "provenance": provenance}
        (args.output/"parameters.json").write_text(json.dumps(parameters, indent=2)+"\n", encoding="utf-8")
        from visualization.genesis_faults import save_fault_snapshot
        from visualization.genesis_onset import save_onset_snapshot
        from visualization.genesis_shell import save_shell_snapshot, save_shell_animation
        from visualization.genesis import save_genesis_history
        g, s, o, orbit_row = model.diagnostics(state, global_state, orbit)
        histories = [[g], [s], [o]]
        frames = []
        last_frame_index = -1
        sample_every = round(args.sample_interval_myr/args.shell_step_myr)
        frame_every = max(1, math.ceil((args.duration_myr-global_state.time_myr)/args.sample_interval_myr/30))

        def summary():
            return {"model_version": FAULT_VERSION, "geometry": "material",
                "status": global_state.stopped_reason or state.stopped_reason or "completed",
                "requested_end_time_myr": args.duration_myr, "events_myr": global_state.events,
                "final": histories[0][-1], "shell": histories[1][-1], "onset": histories[2][-1],
                "orbit": orbit_row, "radius_km": state.radius_km,
                "first_fracture_time_resolution_myr": args.shell_step_myr,
                "limitations": [
                    "Finite-width pressure-dependent irreversible slip; equivalent slip is width times accumulated shear.",
                    "Shared material mesh remains continuous: no displacement jump, detached plates or free-surface contact.",
                    "Moving material mesh with fixed topology; geometric quality limits still stop the calculation.",
                    "Closed-membrane motion begins after a globally load-bearing lid; separate floating rafts are not solved.",
                    "Small-elastic-strain corotational Maxwell approximation; mechanical work is not thermalized.",
                    "Mechanical radius evolves; canonical thermal/orbital radius remains one-way thin-shell forcing.",
                    "Synchronous zero-obliquity small-e satellite tides; prescribed illustrative Love response.",
                    "Isolated eccentricity damping; giant spin tides, resonances and nonsynchronous history absent.",
                    "Water-access weakening is a constitutive proxy, not bound-water mass or mineral chemistry.",
                    "Daily tide motion does not accumulate as plate drift.",
                    "Regularization length is an explicit coarse-scale parameter, not measured fault width.",
                    "No subduction, continental differentiation or mature-model handoff."],
                "provenance": provenance}

        def save_state():
            save_fault_checkpoint(args.output/"fault_checkpoint.npz", model, state, global_state,
                                   orbit, controls, provenance)

        def frame(sample):
            nonlocal last_frame_index
            if args.no_frames:
                return
            path = args.output/"fault_frames"/f"fault_{sample:06d}.png"
            save_fault_snapshot(model.mesh_for(state), model.fields(state, global_state), path,
                                state.time_myr, summary())
            frames.append(path)
            last_frame_index = sample

        save_state()
        frame(0)
        from contextlib import ExitStack
        with ExitStack() as stack:
            files = [stack.enter_context((args.output/name).open("w", newline="", encoding="utf-8"))
                     for name in ("history.csv", "shell_history.csv", "onset_history.csv")]
            writers = [csv.DictWriter(handle, fieldnames=list(rows[0])) for handle, rows in zip(files, histories)]
            for writer, rows in zip(writers, histories):
                writer.writeheader()
                writer.writerow(rows[0])
            for index in range(round(state.time_myr/args.shell_step_myr)+1, round(args.duration_myr/args.shell_step_myr)+1):
                state, global_state, orbit, event_rows = model.step(state, global_state, orbit,
                    index*args.shell_step_myr, max_step_myr=args.max_step_myr)
                for event in event_rows[:-1]:
                    histories[0].append(event)
                    writers[0].writerow(event)
                if index % sample_every == 0 or index == round(args.duration_myr/args.shell_step_myr) or state.stopped_reason or global_state.stopped_reason:
                    g, s, o, orbit_row = model.diagnostics(state, global_state, orbit)
                    for writer, history, row, handle in zip(writers, histories, (g, s, o), files):
                        history.append(row)
                        writer.writerow(row)
                        handle.flush()
                    save_state()
                    number = len(histories[1])-1
                    if number % frame_every == 0:
                        frame(number)
                    print(f"t={state.time_myr:.6f} Myr | e={orbit.eccentricity:.7g} | water access={o['mean_water_access']:.3f} | damage={s['damaged_area_fraction']:.3f} | mean motion={o['mean_speed_cm_yr']:.4g} cm/yr", flush=True)
                if state.stopped_reason or global_state.stopped_reason:
                    break
        final = summary()
        (args.output/"summary.json").write_text(json.dumps(final, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        data = model.fields(state, global_state)
        mesh = model.mesh_for(state)
        np.savez_compressed(args.output/"fault_fields.npz", vertices=mesh.vertices,
                            faces=mesh.faces, centroids=mesh.centroids,
                            radius_km=np.asarray(state.radius_km), **data)
        save_fault_snapshot(mesh, data, args.output/"genesis_faults.png", state.time_myr, final)
        save_onset_snapshot(mesh, data, args.output/"genesis_mobile.png", state.time_myr, final)
        save_shell_snapshot(mesh, data, args.output/"genesis_shell.png", state.time_myr, final)
        save_genesis_history(histories[0], args.output/"genesis_history.png", global_state.events)
        if not args.no_frames:
            if last_frame_index != len(histories[1])-1:
                frame(len(histories[1])-1)
            save_shell_animation(frames, args.output/"genesis_faults.gif")
        print(f"GENESIS_COMPLETE {args.output.resolve()}", flush=True)
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Genesis faults error: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
