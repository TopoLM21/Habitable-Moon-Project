"""Inventory actual inputs without writing or reconstructing their geometry."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from run_fractional_transport_probe import load_probe_source, totals
from tectonics.fractional_surface_io import surface_from_lithosphere


def summarize(source):
    mesh, checkpoint, fracture, model, provenance = load_probe_source(source)
    surface = surface_from_lithosphere(mesh, checkpoint.state, model.thermal.radius_km,
                                      fracture_memory=fracture.memory)
    owners = np.array([p.plate for p in surface.parcels])
    active = [(a, b, u, v) for a, b, u, v in mesh.shared_edges if owners[a] != owners[b]]
    omega = np.array([p.euler_axis*p.angular_speed_rad_per_myr for p in checkpoint.system.plates])
    a = np.array([p.specific_properties[2] for p in surface.parcels])
    ages = np.array([p.age_myr for p in surface.parcels])
    tied = sum(a[i] == a[j] and ages[i] == ages[j] for i, j, _, _ in active)
    lengths = [model.thermal.radius_km*np.arctan2(np.linalg.norm(np.cross(mesh.vertices[u], mesh.vertices[v])),
                                               np.dot(mesh.vertices[u], mesh.vertices[v])) for _, _, u, v in active]
    return {
        "provenance": provenance,
        "cell_count": mesh.cell_count,
        "plate_count": len(checkpoint.system.plates),
        "parcel_count": len(surface.parcels),
        "cross_owner_edge_count": len(active),
        "cross_owner_edge_length_km": float(sum(lengths)),
        "edges_with_identical_density_excess_and_age": int(tied),
        "totals": totals(surface.parcels),
        "fixed_omega_rad_per_myr": omega.tolist(),
    }


def main():
    sources = {
        "starter": ROOT / "results/genesis_runs/starter_20260928_192306_547551/starter_checkpoint.npz",
        "saved50": ROOT / "analysis/slab_sinking_followup/runs/ordered_sub4_dt1/elapsed_0050",
    }
    report = {name: summarize(source) for name, source in sources.items()}
    with Path(__file__).with_name("source_summary.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({name: {k: r[k] for k in ("cell_count", "plate_count", "cross_owner_edge_count", "edges_with_identical_density_excess_and_age")}
                      for name, r in report.items()}))


if __name__ == "__main__":
    main()
