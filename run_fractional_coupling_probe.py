"""Run the experimental fractional heat/basal/material loop without GUI hooks.

This explicitly excludes ridge/slab forces and fracture/topology evolution.
Output and resumable checkpoint paths must be new; source saves are read-only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from run_fractional_transport_probe import (
    ROOT, DEFAULT_SOURCE, digest, load_probe_source, make_birth_factory,
)
from tectonics.fractional_coupling import (
    VERSION, LEGACY_VERSION, DEFAULT_TIME_SCHEME, TIME_SCHEMES, LIMITATIONS, initialize_coupling, advance_coupling,
    load_coupled_checkpoint, save_coupled_checkpoint, ledger_diagnostics,
)
from tectonics.fractional_surface_io import surface_from_lithosphere
from tectonics.simulation import load_config


def execute_probe(source, output, duration_myr, step_myr, *, resume=None, time_scheme=DEFAULT_TIME_SCHEME):
    if time_scheme not in TIME_SCHEMES:
        raise ValueError("Unsupported fractional event time scheme")
    if not all(math.isfinite(x) and x > 0 for x in (duration_myr, step_myr)):
        raise ValueError("Coupled duration and step must be finite and positive")
    ratio = duration_myr/step_myr
    if abs(ratio-round(ratio)) > 1e-9:
        raise ValueError("Coupled duration must be an integer number of steps")
    output, source = Path(output).resolve(), Path(source).resolve()
    if output.exists():
        raise ValueError("Coupled output must be new; existing results are never overwritten")
    mesh, checkpoint, fracture, model, provenance = load_probe_source(source)
    config = load_config(ROOT/"configs/canonical_moon.yaml" if source.is_file() else source/"mature_config.yaml")
    local_kappa = float(config.get("mechanical_lithosphere", {}).get("thermal_diffusivity_m2_s", 1e-6))
    thermal_path = source if source.is_file() else source/"young_context/starter_checkpoint.npz"
    thermal_source = model.load_state(thermal_path)
    version = LEGACY_VERSION if time_scheme == "endpoint_v1" else VERSION
    provenance.update(coupling_version=version, reference_density_kg_m3=model.thermal.surface_layer_density_kg_m3,
        local_cooling_diffusivity_m2_s=local_kappa,
        scope="Dynamic fractional surface with passive local cooling, Genesis heat and basal-only forces",
        source_frame="fixed_mesh", force_components=["prescribed_basal_traction", "linear_basal_drag"],
        fracture_evolution=False, slab_evolution=False, topology_evolution=False)
    if time_scheme != "endpoint_v1":
        provenance["time_scheme"] = time_scheme
    if source.is_dir():
        source_report = json.loads((source/"continuation.json").read_text(encoding="utf-8"))
        provenance["inherited_slab_inventory_scope"] = "Preserved in original source; excluded from experimental dynamics and new removal archive"
        provenance["initial_reference_mantle_mass_kg"] = float(source_report["remaining_mantle_material_mass_kg"])
    else:
        provenance["inherited_slab_inventory_scope"] = "No inherited attached slabs in the Starter source"
        provenance["initial_reference_mantle_mass_kg"] = model.thermal.area_m2*model.thermal.mantle_column_kg_m2
    plate_count = len(checkpoint.system.plates)
    if resume is None:
        surface = surface_from_lithosphere(mesh, checkpoint.state, provenance["radius_km"],
            fracture_memory=fracture.memory)
        state = initialize_coupling(mesh, surface, model, thermal_source.thermal_context, provenance, plate_count)
    else:
        state = load_coupled_checkpoint(resume, mesh, model, provenance)
        if state.plate_count != plate_count:
            raise ValueError("Coupled checkpoint plate count differs from source")
    birth_factory = make_birth_factory(state.surface, model, provenance)
    initial_time = state.surface.time_myr
    start = time.perf_counter()
    # No output files are published until the entire interval is valid.
    for _ in range(int(round(ratio))):
        state = advance_coupling(mesh, state, model, step_myr, birth_factory=birth_factory)
    if any(digest(path) != sha for path, sha in provenance["source_sha256"].items()):
        raise RuntimeError("Coupled source changed during the experiment")
    output.mkdir(parents=True)
    save_coupled_checkpoint(output/"fractional_checkpoint.json", mesh, state)
    production = {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p)
                  for p in sorted((ROOT/"tectonics").glob("fractional_*.py"))}
    production["run_fractional_coupling_probe.py"] = digest(Path(__file__))
    report = dict(format=version, experiment=provenance, source_unchanged=True,
        initial_totals=state.initial_totals, initial_diagnostics=state.initial_diagnostics,
        history=list(state.history), **ledger_diagnostics(state),
        segment_start_time_myr=initial_time, segment_duration_myr=duration_myr, step_myr=step_myr,
        thermal_sample_count=state.thermal_sample_count,
        resumed_from=None if resume is None else str(Path(resume).resolve()),
        wall_seconds=time.perf_counter()-start, production_sha256=production,
        checkpoint_sha256=digest(output/"fractional_checkpoint.json"), limitations=list(LIMITATIONS))
    if time_scheme == "endpoint_v1":
        report["limitations"] = [item for item in report["limitations"]
                                 if not item.startswith("First-order force/advection update;")]
        report["limitations"].append("Legacy endpoint births and frozen removal cold properties; timing error retained for reproduction")
    (output/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-myr", type=float, default=1.)
    parser.add_argument("--step-myr", type=float, default=.25)
    parser.add_argument("--resume", type=Path, help="Fractional coupling checkpoint from this experiment and source")
    parser.add_argument("--time-scheme", choices=TIME_SCHEMES, default=DEFAULT_TIME_SCHEME,
                        help="Within-step time cohorts (default); endpoint_v1 reproduces old coupling saves")
    args = parser.parse_args()
    report = execute_probe(args.source, args.output, args.duration_myr, args.step_myr,
                           resume=args.resume, time_scheme=args.time_scheme)
    last = report["history"][-1]
    print(json.dumps(dict(output=str(args.output.resolve()), time_myr=last["time_myr"],
        mean_speed_mm_per_year=last["endpoint_dynamics"]["mean_speed_mm_per_year"],
        max_speed_mm_per_year=last["endpoint_dynamics"]["max_speed_mm_per_year"],
        parcel_count=last["parcel_count"], relative_residuals=last["relative_residuals"],
        energy_relative_residual=last["heat"]["relative_energy_residual"],
        wall_seconds=report["wall_seconds"], scope=report["experiment"]["scope"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
