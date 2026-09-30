"""Read-only audit of saved raster slab directions and their exact locality.

Run with the repository Python. Only the requested audit JSON is written;
loading a source installs no simulation hooks and advances no state.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from run_fractional_transport_probe import DEFAULT_SOURCE, digest, load_probe_source
from tectonics.young_boundary import slab_thermal_deficit_fraction

SAVED50 = ROOT / "analysis/slab_sinking_followup/runs/ordered_sub4_dt1/elapsed_0050"
QUANTITIES = (
    "retained_area_km2", "retained_oceanic_volume_km3",
    "retained_initial_cold_volume_km3", "retained_initial_excess_mass_kg",
    "current_thermal_excess_mass_kg",
)


def retained_quantities(segment, inventory, time_myr):
    values = {name: [] for name in QUANTITIES}
    for cohort in segment.thermal_cohorts:
        retained = 1. - cohort.deep_transfer_fraction
        thermal = slab_thermal_deficit_fraction(
            time_myr - cohort.acceptance_time_myr, cohort.initial_thickness_km,
            inventory.thermal_diffusivity_m2_s)
        contributions = (
            cohort.accepted_area_km2 * retained,
            cohort.oceanic_volume_km3 * retained,
            cohort.cold_mantle_volume_km3 * retained,
            cohort.initial_density_excess_mass_kg * retained,
            cohort.initial_density_excess_mass_kg * retained * thermal,
        )
        for name, value in zip(QUANTITIES, contributions):
            values[name].append(value)
    return {name: math.fsum(items) for name, items in values.items()}


def summarize_source(source):
    mesh, checkpoint, fracture, model, provenance = load_probe_source(source)
    memory = checkpoint.subduction_memory
    inventory = None if memory is None else memory.young_boundary_state
    output = {
        "provenance": provenance,
        "cell_count": mesh.cell_count,
        "plate_count": len(checkpoint.system.plates),
        "checkpoint_inventory_access": "checkpoint.subduction_memory.young_boundary_state",
        "source_inventory_present": inventory is not None,
        "fracture_scalar_ranges": {
            name: [float(np.min(getattr(fracture.memory, name))),
                   float(np.max(getattr(fracture.memory, name)))]
            for name in ("damage", "cooling_stress_pa", "water_access", "yield_ratio", "strength_pa")
        },
        "interpretation": "Saved directions are inherited raster conditions, not resolved physical initiation.",
    }
    if inventory is None:
        output["counts"] = {"contacts": 0, "segments": 0, "attached": 0}
        return output

    rows = []
    status = Counter(attached=0, valid_attached_history=0, valid_exact_current_locality=0)
    by_contact = defaultdict(list)
    for segment in sorted(inventory.segments.values(), key=lambda item: item.key):
        contact = inventory.contacts.get(segment.contact_key)
        quantities = retained_quantities(segment, inventory, memory.time_myr)
        checks = {
            "attached": bool(segment.attached),
            "contact_present": contact is not None and bool(contact.present),
            "contact_pair_matches_segment": contact is not None and
                {contact.plate_a, contact.plate_b} ==
                {segment.subducting_plate, segment.overriding_plate},
            "retained_material_positive": quantities["retained_area_km2"] > 0. and
                quantities["retained_oceanic_volume_km3"] > 0.,
            "contact_faces_are_exact_mesh_edge": False,
            "current_surface_owners_match_contact": False,
        }
        if contact is not None and all(0 <= face < mesh.cell_count for face in
                                       (contact.face_a, contact.face_b)):
            shared = sorted(set(mesh.faces[contact.face_a]) & set(mesh.faces[contact.face_b]))
            checks["contact_faces_are_exact_mesh_edge"] = (
                len(shared) == 2 and segment.contact_key == f"{shared[0]}:{shared[1]}")
            checks["current_surface_owners_match_contact"] = (
                int(checkpoint.state.cell_plate[contact.face_a]) == contact.plate_a and
                int(checkpoint.state.cell_plate[contact.face_b]) == contact.plate_b)
        valid_history = all(checks[name] for name in (
            "attached", "contact_present", "contact_pair_matches_segment", "retained_material_positive"))
        valid_local = all(checks.values())
        row = {
            "segment_key": segment.key, "contact_key": segment.contact_key,
            "subducting_plate": segment.subducting_plate,
            "overriding_plate": segment.overriding_plate,
            "checks": checks,
            "valid_attached_history": valid_history,
            "valid_exact_current_locality": valid_local,
            **quantities,
        }
        if contact is not None:
            row.update(face_a=contact.face_a, face_b=contact.face_b,
                       midpoint=contact.midpoint, trench_length_km=contact.trench_length_km)
        rows.append(row)
        if segment.attached:
            status["attached"] += 1
            if valid_history:
                status["valid_attached_history"] += 1
                by_contact[segment.contact_key].append(row)
            if valid_local:
                status["valid_exact_current_locality"] += 1

    contacts = []
    for key, group in sorted(by_contact.items()):
        directions = defaultdict(list)
        for row in group:
            directions[(row["subducting_plate"], row["overriding_plate"])].append(row)
        direction_rows = []
        for (sub, over), items in sorted(directions.items()):
            direction_rows.append({
                "subducting_plate": sub, "overriding_plate": over,
                "segment_keys": [row["segment_key"] for row in items],
                "segment_count": len(items),
                "all_exact_current_locality": all(row["valid_exact_current_locality"] for row in items),
                **{name: math.fsum(row[name] for row in items) for name in QUANTITIES},
            })
        contact = inventory.contacts[key]
        contacts.append({
            "contact_key": key, "conflicting_directions": len(directions) > 1,
            "face_a": contact.face_a, "face_b": contact.face_b,
            "plate_a": contact.plate_a, "plate_b": contact.plate_b,
            "midpoint": contact.midpoint, "trench_length_km": contact.trench_length_km,
            "directions": direction_rows,
        })
    output.update(
        connectivity_model=inventory.connectivity_model,
        buoyancy_geometry_model=inventory.buoyancy_geometry_model,
        counts=dict(contacts=len(inventory.contacts),
                    present_contacts=sum(c.present for c in inventory.contacts.values()),
                    segments=len(inventory.segments), **status,
                    attached_history_contact_keys=len(contacts),
                    unambiguous_contact_keys=sum(not c["conflicting_directions"] for c in contacts),
                    conflicting_contact_keys=sum(c["conflicting_directions"] for c in contacts)),
        attached_contact_directions=contacts,
        segments=rows,
        conflicting_contacts_totals={
            name: math.fsum(d[name] for c in contacts if c["conflicting_directions"]
                            for d in c["directions"]) for name in QUANTITIES},
    )
    if any(digest(path) != expected for path, expected in provenance["source_sha256"].items()):
        raise RuntimeError("An audited source changed while being read")
    output["source_hashes_unchanged"] = True
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("source_polarity_audit.json"))
    args = parser.parse_args()
    report = {
        "format": "legacy-source-polarity-audit-1",
        "method": "Read existing cohorts; exclude deep transfer; check exact source mesh edge and owners.",
        "mass_meaning": "Initial excess mass and thermally surviving excess mass are reported separately.",
        "legacy_direction_origin": "Buoyancy, age, prior owner, nearest source, then lexicographic position.",
        "sources": {"starter": summarize_source(DEFAULT_SOURCE), "saved50": summarize_source(SAVED50)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({name: source["counts"] for name, source in report["sources"].items()}))


if __name__ == "__main__":
    main()
