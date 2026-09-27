"""Continue a saved genesis shell with split banks and small-sliding contact.

This is a bounded mechanical experiment. Thermal, orbital and material
weakening fields remain fixed at the source fault checkpoint.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

from tectonics.genesis_contact import (ContactModel, load_contact_checkpoint,
                                      save_contact_checkpoint)


LIMITATIONS = [
    "Edge-alignment extraction can produce two-cell rhombi in a smooth weak-plane field; cut components are not validated crack paths or plates.",
    "Cohesive connectivity includes remaining bonded endpoints and shared crack-tip vertices; disconnected regions can still interact through compression and friction.",
    "Mechanical continuation from one fault checkpoint; thermal state, orbit, face damage and water access are frozen.",
    "Material vertices are split across selected persistent faults; gaps and jumps are actual bank displacements.",
    "Contact is limited to initially paired edges and small sliding; no general collision search or mature plate handoff.",
    "Bulk increments are elastic about the inherited state; this is not a continuing Maxwell cooling calculation.",
    "Basal drag supplies a mantle-relative mechanical reference; inertial rupture propagation is not solved.",
    "Material face and layer masses are preserved; openings are not filled with newly generated crust.",
    "Contact and drag work are recorded separately, without thermal feedback.",
    "No subduction, remeshing, continental differentiation or long-term orbital evolution.",
]


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=_json_default)+"\n",
                    encoding="utf-8")


def _diagnostic_row(model, state):
    row = dict(model.diagnostics(state))
    row.setdefault("elapsed_years", float(state.elapsed_years))
    row["contact_dissipation_j"] = sum(float(row.get(key, 0.)) for key in
                                     ("friction_work_j", "viscous_work_j", "fracture_work_j"))
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="Source fault_checkpoint.npz")
    source.add_argument("--resume", type=Path, help="Saved contact_checkpoint.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-years", type=float, default=1000.,
                        help="Target elapsed mechanical time, measured from the fault checkpoint")
    parser.add_argument("--step-years", type=float, default=10.)
    parser.add_argument("--no-frames", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not all(math.isfinite(value) and value > 0 for value in
                   (args.duration_years, args.step_years)):
            raise ValueError("Finite positive duration and step are required")
        if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
            raise ValueError("Output must be new or empty; existing results are protected")
        source_path = (args.resume or args.checkpoint).resolve()
        if args.resume:
            model, state = load_contact_checkpoint(source_path)
        else:
            model, state = ContactModel.from_fault_checkpoint(source_path)
        if getattr(state, "stopped_reason", None):
            raise ValueError("The saved contact state has already reached a model limit")
        if args.duration_years <= state.elapsed_years:
            raise ValueError("Target elapsed time must follow the saved state")
        controls = {"duration_years": args.duration_years, "step_years": args.step_years}
        provenance = {"source_path": str(source_path),
                      "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                      "source_kind": "contact_checkpoint" if args.resume else "fault_checkpoint",
                      "source_time_myr": float(model.source_time_myr),
                      "coupling": "frozen-source small-sliding mechanical continuation"}
        args.output.mkdir(parents=True, exist_ok=True)
        _write_json(args.output/"parameters.json", {"model": "genesis-contact",
                    "contact": asdict(model.parameters), "law": asdict(model.law_parameters),
                    "controls": controls, "provenance": provenance})
        from visualization.genesis_contact import save_contact_snapshot
        from visualization.genesis_shell import save_shell_animation
        history = [_diagnostic_row(model, state)]
        frame_paths = []
        last_frame_step = -1
        count = math.ceil((args.duration_years-state.elapsed_years)/args.step_years)
        frame_every = max(1, math.ceil(count/28))

        def summary():
            return {"model": "genesis-contact", "status": getattr(state, "stopped_reason", None) or "completed",
                    "requested_elapsed_years": args.duration_years,
                    "source_time_myr": float(model.source_time_myr),
                    "equilibrium_tolerance": model.parameters.equilibrium_tolerance,
                    "final": history[-1], "controls": controls,
                    "limitations": LIMITATIONS, "provenance": provenance}

        def save():
            save_contact_checkpoint(args.output/"contact_checkpoint.npz", model, state)

        def frame(index):
            nonlocal last_frame_step
            if args.no_frames:
                return
            path = args.output/"contact_frames"/f"contact_{index:06d}.png"
            save_contact_snapshot(model.mesh_for(state), model.fields(state), path,
                                  model.source_time_myr, state.elapsed_years, history, summary())
            frame_paths.append(path)
            last_frame_step = index

        save()
        frame(0)
        start = state.elapsed_years
        completed_steps = 0
        with (args.output/"contact_history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerow(history[0])
            handle.flush()
            for index in range(1, count+1):
                target = min(start+index*args.step_years, args.duration_years)
                previous_time = state.elapsed_years
                state = model.step(state, target)
                row = _diagnostic_row(model, state)
                history.append(row)
                writer.writerow(row)
                handle.flush()
                save()
                completed_steps = index
                if index % frame_every == 0:
                    frame(index)
                print(f"contact t={state.elapsed_years:.6g} yr | "
                      f"opening={row.get('max_opening_m', 0.):.5g} m | "
                      f"jump={row.get('max_abs_jump_m', 0.):.5g} m | "
                      f"residual={row.get('equilibrium_residual', 0.):.3g}", flush=True)
                if getattr(state, "stopped_reason", None):
                    break
                if state.elapsed_years <= previous_time:
                    raise RuntimeError("Contact solver did not advance time or return a stop reason")
        final = summary()
        _write_json(args.output/"summary.json", final)
        fields = dict(model.fields(state))
        mesh = model.mesh_for(state)
        fields.update(vertices=mesh.vertices, faces=mesh.faces, centroids=mesh.centroids,
                      source_time_myr=np.asarray(model.source_time_myr),
                      elapsed_years=np.asarray(state.elapsed_years))
        np.savez_compressed(args.output/"contact_fields.npz", **fields)
        save_contact_snapshot(mesh, fields, args.output/"genesis_contact.png",
                              model.source_time_myr, state.elapsed_years, history, final)
        if not args.no_frames:
            if last_frame_step != completed_steps:
                frame(completed_steps)
            save_shell_animation(frame_paths, args.output/"genesis_contact.gif")
        print(f"GENESIS_COMPLETE {args.output.resolve()}", flush=True)
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Genesis contact error: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
