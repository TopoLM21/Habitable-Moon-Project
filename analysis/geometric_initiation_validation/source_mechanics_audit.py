"""Read-only audit of initiation-relevant fields in the existing source saves.

The only write is the sibling JSON report. This does not advance a simulation
or infer a down-going plate from a tangent stress/plane or scalar damage.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import fields
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from run_fractional_transport_probe import DEFAULT_SOURCE, digest, load_probe_source
from tectonics.genesis_fault_law import WeakPlaneParameters, return_map
from tectonics.genesis_faults import FaultState
from tectonics.genesis_starter import StarterState
from tectonics.genesis_starter_fracture import YoungFractureMemory

SAVED50 = ROOT / "analysis/slab_sinking_followup/runs/ordered_sub4_dt1/elapsed_0050"
MODULES = (
    "tectonics/genesis_fault_law.py", "tectonics/genesis_faults.py",
    "tectonics/genesis_starter.py", "tectonics/genesis_starter_fracture.py",
    "tectonics/genesis_plate_membrane_diagnostic.py", "tectonics/mantle.py",
    "tectonics/young_boundary.py",
)


def summarize_array(value):
    value = np.asarray(value)
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "minimum": float(np.min(value)), "maximum": float(np.max(value)),
            "nonzero_count": int(np.count_nonzero(value))}


def archive_keys(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: {"shape": list(saved[key].shape), "dtype": str(saved[key].dtype)}
                for key in saved.files if key != "metadata"}


def source_audit(path):
    mesh, checkpoint, fracture, model, provenance = load_probe_source(path)
    scalar_names = ("damage", "cooling_stress_pa", "water_access", "yield_ratio", "strength_pa")
    inventory = (checkpoint.subduction_memory.young_boundary_state
                 if checkpoint.subduction_memory is not None else None)
    report = {
        "provenance": provenance,
        "checkpoint_state_fields": [f.name for f in fields(checkpoint.state)],
        "fracture_memory_fields": [f.name for f in fields(fracture.memory)],
        "fracture_fields": {name: summarize_array(getattr(fracture.memory, name))
                            for name in scalar_names},
        "starter_effective_mantle_tensor": summarize_array(model.mantle_tensor),
        "mature_intraplate_stress": summarize_array(checkpoint.state.intraplate_stress),
        "initiation_input_availability": {
            "damage_strength_water": True,
            "plate_tangent_velocity": True,
            "local_dipping_fault_geometry": False,
            "transported_contact_stress_tensor": False,
            "normal_radial_mantle_velocity": False,
            "resolved_vertical_overburden_profile": False,
            "fault_pore_pressure_profile": False,
        },
    }
    archives = [path] if path.is_file() else [
        path / "young_context/starter_checkpoint.npz",
        path / "young_context/fracture_memory.npz", path / "mature_checkpoint/state.npz"]
    report["raw_archive_keys"] = {str(p.relative_to(ROOT)): archive_keys(p) for p in archives}
    groups = defaultdict(list)
    if inventory is not None:
        for segment in inventory.segments.values():
            contact = inventory.contacts.get(segment.contact_key)
            retained = [c for c in segment.thermal_cohorts
                        if c.accepted_area_km2 * (1. - c.deep_transfer_fraction) > 0.]
            if (segment.attached and contact is not None and contact.present and retained
                    and {segment.subducting_plate, segment.overriding_plate}
                    == {contact.plate_a, contact.plate_b}):
                groups[segment.contact_key].append({
                    "segment_key": segment.key,
                    "subducting_plate": segment.subducting_plate,
                    "overriding_plate": segment.overriding_plate,
                    "first_acceptance_time_myr": segment.first_acceptance_time_myr,
                    "last_acceptance_time_myr": segment.last_acceptance_time_myr,
                    "retained_cohort_time_range_myr": [
                        min(c.acceptance_time_myr for c in retained),
                        max(c.acceptance_time_myr for c in retained)],
                    "retained_area_km2": sum(c.accepted_area_km2 * (1. - c.deep_transfer_fraction)
                                             for c in retained),
                })
    conflicts = {key: rows for key, rows in sorted(groups.items())
                 if len({(row["subducting_plate"], row["overriding_plate"])
                         for row in rows}) > 1}
    report["inherited_contact_count"] = len(groups)
    report["conflicting_inherited_contacts"] = conflicts
    report["conflict_interpretation"] = (
        "Both directions retain attached accepted material. Different timestamps, mass, or "
        "current effective scalar stress do not establish which history was physical. "
        "Do not silently pick newest/largest or discard one reservoir. Replay from an "
        "unambiguous earlier state under a specified initiation/contact law, or retain "
        "both unresolved until a mechanical transition explicitly accounts for both.")
    report["source_hashes_unchanged"] = all(digest(name) == expected
                                            for name, expected in provenance["source_sha256"].items())
    assert report["source_hashes_unchanged"]
    return report


def tangent_plane_sign_audit():
    # A tangent weak-plane normal is a line, not a directed downward dip.
    normals = np.asarray([[.6, .8], [1., 0.], [0., 1.]])
    strains = np.asarray([[-.002, .001, .006], [-.003, .001, .008], [.001, -.002, .006]])
    kwargs = dict(elastic_trial=strains, active=np.ones(3, dtype=bool),
                  damage=np.full(3, .8), water=np.full(3, .5), effective_b=np.full(3, .7),
                  dt_myr=.001, young_pa=6e10, poisson_ratio=.25, params=WeakPlaneParameters())
    first = return_map(plane_normal=normals, **kwargs)
    opposite = return_map(plane_normal=-normals, **kwargs)
    difference = {key: float(np.max(np.abs(np.asarray(first[key]) - np.asarray(opposite[key]))))
                  for key in first}
    assert all(value == 0. for value in difference.values())
    assert np.any(first["shear_increment"] != 0.)
    return {
        "input_planes": normals.tolist(),
        "shear_increment": first["shear_increment"].tolist(),
        "maximum_difference_after_normal_sign_flip_by_output": difference,
        "interpretation": "n and -n produce identical in-plane return maps, including signed shear. "
                          "Neither normal sign nor signed shear supplies a down-dip direction.",
    }


def main():
    hashes = {name: digest(ROOT / name) for name in MODULES}
    report = {
        "format": "geometric-initiation-source-mechanics-audit-1",
        "method": "Read real Starter and saved50 checkpoints; inspect exact persisted fields; "
                  "evaluate existing constitutive map for tangent-plane sign invariance.",
        "production_sha256": hashes,
        "source_type_fields": {cls.__name__: [f.name for f in fields(cls)]
                               for cls in (StarterState, YoungFractureMemory, FaultState)},
        "tangent_plane_sign_control": tangent_plane_sign_audit(),
        "sources": {"starter": source_audit(DEFAULT_SOURCE), "saved50": source_audit(SAVED50)},
        "integration_path": [
            "Require explicit local 3D dipping-plane geometry and physically sourced stress/traction.",
            "Compute closed-contact effective compression and directed slip admissibility before material removal.",
            "Treat no admissible mode as locked/compressive and tied/opposed modes as unresolved; "
            "do not choose by plate number, coordinate order, plane-normal sign or floating-point noise.",
            "Couple locked/compressive response to the motion/contact solve; an endpoint transport probe "
            "cannot itself store unresolved horizontal overlap as valid single-covered surface material.",
            "Preserve all inherited slab material; replay contradictory legacy acceptance if seeking "
            "a dynamically consistent newly initiated history.",
        ],
    }
    assert hashes == {name: digest(ROOT / name) for name in MODULES}
    destination = Path(__file__).with_suffix(".json")
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination),
                      "source_hashes_unchanged": {name: row["source_hashes_unchanged"]
                                                  for name, row in report["sources"].items()},
                      "conflicting_contacts": len(report["sources"]["saved50"]["conflicting_inherited_contacts"]),
                      "tangent_normal_sign_invariant": True}, indent=2))


if __name__ == "__main__":
    main()
