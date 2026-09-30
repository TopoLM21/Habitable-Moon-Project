"""Actual-source geometric contact oracle independent of the new state adapter."""
from dataclasses import dataclass
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from run_fractional_transport_probe import load_probe_source
from tectonics.fractional_surface_io import surface_from_lithosphere
from tectonics.geometric_contacts import extract_contacts


@dataclass(frozen=True)
class Fragment:
    fragment_id: str
    polygon: tuple
    parcel: object


def execute(source):
    mesh, cp, fracture, model, provenance = load_probe_source(source)
    radius = model.thermal.radius_km
    surface = surface_from_lithosphere(mesh, cp.state, radius, fracture_memory=fracture.memory)
    fragments = tuple(Fragment(f"source:{p.cell}", tuple(map(tuple, mesh.vertices[mesh.faces[p.cell]])), p)
                      for p in surface.parcels)
    owners = np.array([p.plate for p in surface.parcels])
    expected_edges = [(a, b, u, v) for a, b, u, v in mesh.shared_edges if owners[a] != owners[b]]
    expected_length = math.fsum(radius*math.atan2(np.linalg.norm(np.cross(mesh.vertices[u], mesh.vertices[v])),
                                                np.dot(mesh.vertices[u], mesh.vertices[v]))
                               for _, _, u, v in expected_edges)
    started = time.perf_counter()
    contacts = extract_contacts(fragments, radius)
    wall = time.perf_counter()-started
    assert len(contacts) == len(expected_edges)
    actual_length = math.fsum(contact.length_km for contact in contacts)
    assert math.isclose(actual_length, expected_length, rel_tol=2e-12)
    frame = Rotation.from_rotvec([.73, -.22, .57])
    rotated = tuple(Fragment(f.fragment_id, tuple(map(tuple, frame.apply(f.polygon))), f.parcel)
                    for f in fragments)
    started = time.perf_counter()
    changed = extract_contacts(rotated, radius)
    rotated_wall = time.perf_counter()-started
    before = {c.contact_id: c for c in contacts}
    assert len(changed) == len(before)
    maximum_normal_error = 0.
    maximum_length_error = 0.
    for c in changed:
        prior = before[c.contact_id]
        maximum_normal_error = max(maximum_normal_error, float(np.linalg.norm(frame.apply(prior.normal_a_to_b)-c.normal_a_to_b)))
        maximum_length_error = max(maximum_length_error, abs(c.length_km-prior.length_km)/prior.length_km)
    assert maximum_normal_error < 1e-11
    assert maximum_length_error < 1e-11
    return dict(source=provenance, contacts=len(contacts), expected_contacts=len(expected_edges),
                total_length_km=actual_length, expected_length_km=expected_length,
                extraction_seconds=wall, rotated_extraction_seconds=rotated_wall,
                rotation_maximum_normal_error=maximum_normal_error,
                rotation_maximum_relative_length_error=maximum_length_error,
                contact_ids_preserved_after_rotation=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("static_contact_probe.json"))
    args = parser.parse_args()
    sources = {
        "starter": ROOT/"results/genesis_runs/starter_20260928_192306_547551/starter_checkpoint.npz",
        "saved50": ROOT/"analysis/slab_sinking_followup/runs/ordered_sub4_dt1/elapsed_0050",
    }
    paths = [ROOT/"tectonics"/name for name in ("geometric_contacts.py", "spherical_polygons.py")]
    code = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    started = datetime.now(timezone.utc).isoformat()
    result = {name: execute(source) for name, source in sources.items()}
    assert code == {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result["_production"] = {"started_utc": started, "sha256": code, "changed_during_run": False}
    target = args.output
    with target.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({name: {k: v for k, v in result[name].items() if k != "source"} for name in sources}))
