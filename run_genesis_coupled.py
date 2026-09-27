"""Continue fault formation with coevolving cooling, orbit, Maxwell stress and bank contact."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

from tectonics.genesis_coupled import CoupledModel, load_coupled_checkpoint, save_coupled_checkpoint


LIMITATIONS = [
    "Edge-alignment extraction can produce two-cell rhombi in a smooth weak-plane field; cut components are not validated crack paths or plates.",
    "Cohesive connectivity includes any remaining bonded contact layer and shared crack-tip vertices; disconnected regions can still interact through compression and friction.",
    "Cooling, orbit, water access, shell strength and Maxwell stress evolve on the same physical clock as bank contact.",
    "This is an experimental small-strain, small-sliding shell calculation, not a full mature plate handoff.",
    "Initially adjacent material edges remain the only possible contact partners; no general collision search or remeshing.",
    "Newly solidified contact depth is added in independent material layers (default growth quantum 1 m); existing layer histories and accumulated work are preserved.",
    "A layer born between open banks starts unbonded; solidification does not fill the gap or create a cohesive bridge.",
    "The default 2% interface-area guard bounds geometric reference distortion, not growth by solidification (saved parameters define the actual threshold).",
    "Thermal advance precedes mechanics; birth geometry is taken at the start of each trial. Timestep and growth-layer refinement are required.",
    "Basal mantle traction is prescribed by source shell parameters, not an evolving mantle-convection force solution.",
    "Bulk stress follows Maxwell relaxation; the original finite-width weak-plane plastic shear return map is replaced by split-edge contact slip.",
    "The mechanical and thermal energy ledgers remain separate; mechanical dissipation is not returned as heat.",
    "No continental differentiation, subduction or generation of new crust inside open gaps.",
    "Stop conditions include geometry, solver, remelting of represented interface material and cohort-count limits; a stopped checkpoint must not be forced onward.",
]


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=_json_default) + "\n", encoding="utf-8")


def _row(model, state, thermal, orbit):
    row = dict(model.diagnostics(state, thermal, orbit))
    row["time_myr"] = float(state.time_myr)
    row["elapsed_years"] = (float(state.time_myr) - float(model.source_time_myr)) * 1e6
    return row


def _validate_source_format(path, resume):
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
        version = str(metadata["format"])
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Source does not contain valid genesis checkpoint metadata") from exc
    expected = "genesis-coupled-" if resume else "genesis-faults-"
    if not version.startswith(expected):
        raise ValueError("Physical continuation requires an original fault checkpoint or --resume coupled checkpoint; frozen contact snapshots are incompatible")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="Original fault_checkpoint.npz")
    source.add_argument("--resume", type=Path, help="Saved coupled_checkpoint.npz; frozen contact checkpoints are incompatible")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-years", type=float, default=1000., help="Target elapsed physical years since the ORIGINAL fault checkpoint")
    parser.add_argument("--step-years", type=float, default=100., help="Output and maximum requested physical step; solver may subdivide")
    parser.add_argument("--no-frames", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not all(math.isfinite(x) and x > 0 for x in (args.duration_years, args.step_years)):
            raise ValueError("Finite positive duration and step are required")
        if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
            raise ValueError("Output must be new or empty; existing results are protected")
        source_path = (args.resume or args.checkpoint).resolve()
        _validate_source_format(source_path, bool(args.resume))
        model, state, thermal, orbit = (load_coupled_checkpoint(source_path) if args.resume
                                        else CoupledModel.from_fault_checkpoint(source_path))
        if getattr(state, "stopped_reason", None) or getattr(thermal, "stopped_reason", None):
            raise ValueError("The saved coupled state has already reached a model limit")
        end_time = float(model.source_time_myr) + args.duration_years / 1e6
        if end_time <= float(state.time_myr):
            raise ValueError("Target elapsed time must follow the saved physical state")
        controls = {"duration_years": args.duration_years, "step_years": args.step_years}
        provenance = {"source_path": str(source_path),
                      "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                      "source_kind": "coupled_checkpoint" if args.resume else "fault_checkpoint",
                      "source_time_myr": float(model.source_time_myr),
                      "coupling": "coevolving thermal-orbital-Maxwell-contact physical time"}
        args.output.mkdir(parents=True, exist_ok=True)
        _write_json(args.output / "parameters.json", {"model": "genesis-coupled", "controls": controls,
                    "provenance": provenance, "physics_parameters": "stored in coupled_checkpoint.npz"})
        from visualization.genesis_coupled import save_coupled_snapshot
        from visualization.genesis_shell import save_shell_animation
        history = [_row(model, state, thermal, orbit)]
        paths = []
        count = max(1, math.ceil((end_time - state.time_myr) * 1e6 / args.step_years - 1e-9))
        frame_every = max(1, math.ceil(count / 28))
        last_frame = -1

        def summary():
            reason = getattr(state, "stopped_reason", None) or getattr(thermal, "stopped_reason", None)
            return {"model": "genesis-coupled", "status": reason or "completed",
                    "requested_elapsed_years": args.duration_years, "final": history[-1],
                    "source_time_myr": float(model.source_time_myr), "controls": controls,
                    "provenance": provenance, "limitations": LIMITATIONS, "mature_handoff": False}

        def save():
            save_coupled_checkpoint(args.output / "coupled_checkpoint.npz", model, state, thermal, orbit)

        def frame(index):
            nonlocal last_frame
            if args.no_frames:
                return
            path = args.output / "coupled_frames" / f"coupled_{index:06d}.png"
            save_coupled_snapshot(model.mesh_for(state), model.fields(state, thermal), path, history, summary())
            paths.append(path)
            last_frame = index

        save()
        frame(0)
        start = float(state.time_myr)
        completed = 0
        with (args.output / "coupled_history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerow(history[0])
            handle.flush()
            for index in range(1, count + 1):
                target = min(start + index * args.step_years / 1e6, end_time)
                before = float(state.time_myr)
                state, thermal, orbit, _ = model.step(state, thermal, orbit, target,
                                                      max_step_myr=args.step_years / 1e6)
                row = _row(model, state, thermal, orbit)
                history.append(row)
                writer.writerow(row)
                handle.flush()
                save()
                completed = index
                if index % frame_every == 0:
                    frame(index)
                print(f"coupled t={state.time_myr:.9g} Myr | elapsed={row['elapsed_years']:.6g} yr | "
                      f"surface={row.get('surface_temperature_k', 0.):.5g} K | "
                      f"lid={row.get('mean_lid_thickness_km', 0.):.5g} km | "
                      f"opening={row.get('max_opening_m', 0.):.5g} m", flush=True)
                if getattr(state, "stopped_reason", None) or getattr(thermal, "stopped_reason", None):
                    break
                if state.time_myr <= before:
                    raise RuntimeError("Coupled solver did not advance physical time or report a limit")
        final = summary()
        _write_json(args.output / "summary.json", final)
        mesh = model.mesh_for(state)
        fields = dict(model.fields(state, thermal))
        fields.update(vertices=mesh.vertices, faces=mesh.faces, centroids=mesh.centroids,
                      time_myr=np.asarray(state.time_myr), source_time_myr=np.asarray(model.source_time_myr))
        np.savez_compressed(args.output / "coupled_fields.npz", **fields)
        save_coupled_snapshot(mesh, fields, args.output / "genesis_coupled.png", history, final)
        if not args.no_frames:
            if last_frame != completed:
                frame(completed)
            save_shell_animation(paths, args.output / "genesis_coupled.gif")
        print(f"GENESIS_COUPLED_COMPLETE {args.output.resolve()}", flush=True)
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Genesis coupled error: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
