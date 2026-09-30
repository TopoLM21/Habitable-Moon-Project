"""Independent fractional surface experiment with frozen prescribed velocities.

This advances material transport only, not the coupled plate/heat/fracture
model. It never resumes or rewrites a mature checkpoint. Run --help for usage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "results/genesis_runs/starter_20260928_192306_547551/starter_checkpoint.npz"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes(path):
    path = Path(path).resolve()
    if path.is_file():
        return {str(p): digest(p) for p in (path, path.parent/"parameters.json")}
    report = json.loads((path/"continuation.json").read_text(encoding="utf-8"))
    files = {str((path/name).resolve()): sha for name, sha in report["checkpoint_sha256"].items()}
    if any(digest(name) != sha for name, sha in files.items()):
        raise ValueError("Saved source integrity mismatch")
    files[str(path/"continuation.json")] = digest(path/"continuation.json")
    return files


def load_probe_source(source):
    """Return mesh, checkpoint, fracture, model, provenance without installing hooks."""
    from tectonics.genesis_starter_continuation import (
        load_starter_source, build_starter_continuation, _load_cp,
    )
    from tectonics.genesis_starter_fracture import YoungShellFracture
    from tectonics.simulation import load_config
    source = Path(source).resolve()
    hashes = source_hashes(source)
    if source.is_file():
        model, starter, _ = load_starter_source(source)
        config = load_config(ROOT/"configs/canonical_moon.yaml")
        # This importer remains pinned even if the coupled model later changes.
        config.setdefault("young_shell", {})["mechanics_model_version"] = "young-mechanics-0.5"
        bundle, config, imported = build_starter_continuation(model, starter, config)
        checkpoint = bundle.checkpoint
        fracture = YoungShellFracture(model, starter)
        origin = imported["origin_time_myr"]
    else:
        report = json.loads((source/"continuation.json").read_text(encoding="utf-8"))
        model, starter, _ = load_starter_source(source/"young_context/starter_checkpoint.npz")
        fracture = YoungShellFracture.load(model, source/"young_context/fracture_memory.npz")
        config = load_config(source/"mature_config.yaml")
        checkpoint = _load_cp(source/"mature_checkpoint", config)
        origin = report["import"]["origin_time_myr"]
    if not math.isclose(checkpoint.state.time_myr, fracture.time_myr, rel_tol=0, abs_tol=1e-10):
        raise ValueError("Source material/fracture clocks disagree")
    provenance = dict(source=str(source), source_sha256=hashes,
                      source_mechanics_version=config["young_shell"]["mechanics_model_version"],
                      source_time_myr=float(checkpoint.state.time_myr), origin_time_myr=float(origin),
                      subdivisions=int(model.shell.subdivisions), radius_km=float(model.thermal.radius_km),
                      newborn_crust_thickness_km=float(config["lithosphere"]["oceanic_thickness_km"]))
    return model.mesh, checkpoint, fracture, model, provenance


def make_birth_factory(surface, model, provenance):
    """Explicit zero-age, hot oceanic birth, using the saved source parameters."""
    from tectonics.fractional_surface import SurfaceParcel
    names = sorted({name for parcel in surface.parcels for name, _ in parcel.material_fields})
    thickness = float(provenance["newborn_crust_thickness_km"])
    if not math.isfinite(thickness) or thickness <= 0:
        raise ValueError("Newborn ocean requires explicit positive source crust thickness")
    strength = model.shell.tensile_strength_pa * np.asarray(model.strength_factor)
    def birth(cell, plate, area_km2, time_myr, serial_id):
        fields = {name: 0. for name in names}
        if "fracture_strength_pa" in fields:
            fields["fracture_strength_pa"] = float(strength[cell])
        return SurfaceParcel(cell=int(cell), plate=int(plate),
            material_id=f"birth:{float(time_myr).hex()}:{cell}:{plate}:{serial_id}",
            area_km2=float(area_km2), oceanic_volume_km3=float(area_km2)*thickness,
            cold_mantle_volume_km3=0., density_excess_mass_kg=0., age_myr=0.,
            material_fields=tuple(sorted(fields.items())))
    return birth


EXTENSIVE = ("area_km2", "oceanic_volume_km3", "cold_mantle_volume_km3", "density_excess_mass_kg")


def totals(parcels):
    parcels = tuple(parcels)
    return {key: math.fsum(getattr(p, key) for p in parcels) for key in EXTENSIVE}


def execute_probe(source, output, duration_myr, step_myr, *, resume=None, common_omega=None):
    from tectonics.fractional_surface_io import (
        surface_from_lithosphere, save_fractional_checkpoint, load_fractional_checkpoint,
    )
    from tectonics.fractional_transport import advance_fractional_transport
    if not all(math.isfinite(x) and x > 0 for x in (duration_myr, step_myr)):
        raise ValueError("Probe duration and step must be finite and positive")
    ratio = duration_myr/step_myr
    if abs(ratio-round(ratio)) > 1e-9:
        raise ValueError("Probe duration must be an integer number of steps")
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Probe output must be new; existing results are never overwritten")
    mesh, checkpoint, fracture, model, provenance = load_probe_source(source)
    radius = provenance["radius_km"]
    omega = np.array([p.euler_axis*p.angular_speed_rad_per_myr for p in checkpoint.system.plates])
    if common_omega is not None:
        vector = np.asarray(common_omega, dtype=float)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError("Common rotation must have three finite components in rad/Myr")
        omega[:] = vector
    provenance["fixed_omega_rad_per_myr"] = omega.tolist()
    provenance["scope"] = "Transport only: fixed rotations, frozen mechanical fields and damage; no heat/force/fracture evolution"
    if resume is None:
        state = surface_from_lithosphere(mesh, checkpoint.state, radius, fracture_memory=fracture.memory)
        origin_totals = totals(state.parcels)
        lost = {key: 0. for key in EXTENSIVE}
        born = dict(lost)
        history = []
    else:
        state, saved = load_fractional_checkpoint(resume, mesh, radius)
        if saved.get("experiment") != provenance:
            raise ValueError("Fractional resume requires the identical source, rotations and birth parameters")
        origin_totals, lost, born, history = (saved[key] for key in
            ("initial_totals", "cumulative_losses", "cumulative_births", "history"))
        if state.time_myr < provenance["source_time_myr"]:
            raise ValueError("Fractional checkpoint precedes the source")
    birth_factory = make_birth_factory(state, model, provenance)
    initial_time = state.time_myr
    start = time.perf_counter()
    output.mkdir(parents=True)
    for index in range(int(round(ratio))):
        result = advance_fractional_transport(mesh, state, omega, radius, step_myr, birth_factory=birth_factory)
        state = result.state
        for key, value in totals(loss.parcel for loss in result.losses).items():
            lost[key] += value
        for key, value in totals(result.births).items():
            born[key] += value
        remaining = totals(state.parcels)
        residuals = {key: (remaining[key]+lost[key]-born[key]-origin_totals[key]) /
                      max(abs(origin_totals[key]), abs(born[key]), 1.) for key in EXTENSIVE}
        if any(abs(value) > 5e-12 for value in residuals.values()):
            raise RuntimeError("Fractional experiment cumulative material ledger does not close")
        history.append(dict(time_myr=state.time_myr, elapsed_since_source_myr=state.time_myr-provenance["source_time_myr"],
                            parcel_count=len(state.parcels), retained=remaining, cumulative_losses=dict(lost),
                            cumulative_births=dict(born), relative_residuals=residuals, diagnostics=result.diagnostics))
    if any(digest(path) != sha for path, sha in provenance["source_sha256"].items()):
        raise RuntimeError("Probe source changed during the experiment")
    saved = dict(experiment=provenance, initial_totals=origin_totals,
                 cumulative_losses=lost, cumulative_births=born, history=history)
    save_fractional_checkpoint(output/"fractional_checkpoint.json", mesh, state, radius, provenance=saved)
    production = {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p)
                  for p in sorted((ROOT/"tectonics").glob("fractional_*.py"))}
    report = dict(format="fractional-transport-probe-1", **saved, wall_seconds=time.perf_counter()-start,
                  resumed_from=None if resume is None else str(Path(resume).resolve()),
                  segment_start_time_myr=initial_time, segment_duration_myr=duration_myr, step_myr=step_myr,
                  source_unchanged=True, production_sha256=production,
                  checkpoint_sha256=digest(output/"fractional_checkpoint.json"),
                  limitations=["Independent oceanic transport experiment; not a coupled plate-speed prediction",
                      "First-order spatial upwinding diffuses interfaces; no geometric interface reconstruction",
                      "Nonzero continental or sediment reservoirs are explicitly unsupported",
                      "Per-parcel age advances but thermal/damage evolution is frozen",
                      "Local overlap removal is an explicit material policy, not a resolved subduction initiation law",
                      "Sparse material count grows; no unaccounted truncation or age averaging is used"])
    (output/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-myr", type=float, default=1.)
    parser.add_argument("--step-myr", type=float, default=.25)
    parser.add_argument("--resume", type=Path, help="Independent fractional checkpoint; original source must also match")
    parser.add_argument("--common-omega", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        help="Give every plate this same rigid rotation in rad/Myr, a conservation control")
    args = parser.parse_args()
    report = execute_probe(args.source, args.output, args.duration_myr, args.step_myr,
                           resume=args.resume, common_omega=args.common_omega)
    last = report["history"][-1]
    print(json.dumps(dict(output=str(args.output.resolve()), parcel_count=last["parcel_count"],
                         relative_residuals=last["relative_residuals"],
                         cumulative_losses=last["cumulative_losses"],
                         wall_seconds=report["wall_seconds"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
