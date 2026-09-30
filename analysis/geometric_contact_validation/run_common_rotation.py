"""Real-checkpoint geometric common-rotation and integration-mesh validation."""
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from run_fractional_transport_probe import load_probe_source, digest
from tectonics.fractional_surface_io import surface_from_lithosphere
from tectonics.geometric_contacts import extract_contacts
from tectonics.geometric_surface import (
    from_fractional_surface, load_geometric_checkpoint, project_to_mesh,
    rotate_surface, save_geometric_checkpoint, totals,
)
from tectonics.mesh import build_icosphere


def signature(state):
    return hashlib.sha256(json.dumps(asdict(state), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def by_material(fragments, field):
    groups = {}
    for fragment in fragments:
        groups.setdefault(fragment.parcel.material_id, []).append(getattr(fragment.parcel, field))
    return {key: math.fsum(values) for key, values in groups.items()}


def execute(source, output, subdivisions):
    if output.exists():
        raise ValueError("Validation output must be new")
    output.mkdir(parents=True)
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    code_paths = [ROOT/"tectonics"/name for name in (
        "geometric_surface.py", "geometric_contacts.py", "spherical_polygons.py")]
    code_hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in code_paths}
    mesh, cp, fracture, model, provenance = load_probe_source(source)
    radius = model.thermal.radius_km
    source_surface = surface_from_lithosphere(mesh, cp.state, radius, fracture_memory=fracture.memory)
    state = from_fractional_surface(mesh, source_surface, radius)
    before = signature(state)
    omega = np.tile([.0007, -.0004, .0002], (len(cp.system.plates), 1))
    first = rotate_surface(state, omega, .25)
    direct = rotate_surface(first, omega, .25)
    initial_contacts = extract_contacts(state.fragments, radius)
    contacts = extract_contacts(first.fragments, radius)
    assert {c.contact_id for c in contacts} == {c.contact_id for c in initial_contacts}
    contact_length = math.fsum(c.length_km for c in contacts)
    initial_length = math.fsum(c.length_km for c in initial_contacts)
    assert math.isclose(contact_length, initial_length, rel_tol=2e-12)
    print(json.dumps({"stage": "rotated", "contacts": len(contacts), "elapsed_seconds": time.perf_counter()-started}), flush=True)
    save_geometric_checkpoint(output/"geometric_checkpoint.json", first, provenance=provenance)
    loaded, loaded_provenance = load_geometric_checkpoint(output/"geometric_checkpoint.json")
    resumed = rotate_surface(loaded, omega, .25)
    assert signature(loaded) == signature(first)
    assert signature(resumed) == signature(direct)
    assert provenance == loaded_provenance
    print(json.dumps({"stage": "restart", "bitwise_equal": True, "elapsed_seconds": time.perf_counter()-started}), flush=True)
    target = build_icosphere(subdivisions)
    projected = project_to_mesh(target, first)
    origins_error = 0.
    for field in totals(state.fragments):
        expected, actual = by_material(first.fragments, field), by_material(projected, field)
        assert actual.keys() == expected.keys()
        for key in expected:
            error = abs(actual[key]-expected[key])/max(abs(expected[key]), 1.)
            origins_error = max(origins_error, error)
    assert origins_error < 5e-12
    area = np.bincount([f.parcel.cell for f in projected], weights=[f.parcel.area_km2 for f in projected],
                       minlength=target.cell_count)
    coverage_error = float(np.max(np.abs(area/target.physical_cell_areas_km2(radius)-1.)))
    assert coverage_error < 2e-10
    owners = {}
    for fragment in projected:
        owners.setdefault(fragment.parcel.cell, set()).add(fragment.parcel.plate)
    mixed_cells = sum(len(value)>1 for value in owners.values())
    assert mixed_cells > 0
    assert signature(state) == before
    assert signature(loaded) == signature(first)
    assert all(digest(path) == expected for path, expected in provenance["source_sha256"].items())
    assert all(digest(ROOT/path) == expected for path, expected in code_hashes.items())
    report = dict(provenance=provenance, scope="Fixed common rotation, no thermal or plate-force integration",
                  start_time_myr=state.time_myr, end_time_myr=first.time_myr,
                  source_fragment_count=len(state.fragments), target_cell_count=target.cell_count,
                  projected_fragment_count=len(projected), mixed_cell_count=mixed_cells,
                  contact_count=len(contacts), contact_length_km=contact_length,
                  contact_length_relative_error=abs(contact_length-initial_length)/initial_length,
                  maximum_per_material_relative_error=origins_error,
                  maximum_cell_coverage_relative_error=coverage_error,
                  initial_totals=totals(state.fragments), retained_totals=totals(first.fragments),
                  geometric_birth_count=0, geometric_loss_count=0,
                  common_rotation_checkpoint_restart_bitwise_equal=True,
                  checkpoint_sha256=digest(output/"geometric_checkpoint.json"),
                  source_unchanged=True, production_unchanged_during_run=True,
                  production_sha256=code_hashes, started_utc=started_utc,
                  wall_seconds=time.perf_counter()-started)
    with (output/"report.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("starter", "saved50"), default="starter")
    parser.add_argument("--target-subdivisions", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = ROOT/("results/genesis_runs/starter_20260928_192306_547551/starter_checkpoint.npz"
                   if args.source == "starter" else "analysis/slab_sinking_followup/runs/ordered_sub4_dt1/elapsed_0050")
    execute(source, args.output.resolve(), args.target_subdivisions)
