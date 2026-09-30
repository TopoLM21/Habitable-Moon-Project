"""Explicit-load spherical fault admission controls; no coupled speed prediction.

Eight spherical octants form two hemispheric plates. The ordinary 20-cell
icosphere is only an integration mesh. A differential rotation gives two
convergent and two opening arcs, with zero velocity at isolated pole endpoints.
The stress and fault planes below are declared synthetic boundary conditions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from tectonics.fractional_surface import EXTENSIVE_FIELDS, SurfaceParcel
from tectonics.geometric_boundary import (boundary_diagnostics, boundary_to_dict,
    consume_geometric_transaction, initialize_boundary)
from tectonics.geometric_boundary_io import load_boundary_checkpoint, save_boundary_checkpoint
from tectonics.geometric_contacts import extract_contacts
from tectonics.geometric_initiation import OrientedFault
from tectonics.geometric_initiation_adapter import (
    fault_snapshot_to_dict, make_forced_underthrust_resolver)
from tectonics.geometric_polarity import PolarityEvidence
from tectonics.geometric_surface import GeometricFragment, GeometricSurfaceState, totals
from tectonics.geometric_transport import UnresolvedPolarityError, advance_geometric_surface
from tectonics.mesh import build_icosphere
from tectonics.spherical_polygons import normalize_polygon, polygon_area

MOTION = np.array([[0., 0., 0.], [0., 0., .02]])
COMMON = np.array([[.002, -.003, .001]]*2)
DT = .1


def fixture(*, tied=True):
    mesh = build_icosphere(0)
    radius = 100.
    fragments = []
    for index, signs in enumerate(product((-1., 1.), repeat=3)):
        polygon = normalize_polygon(np.diag(signs))
        area = polygon_area(polygon)*radius**2
        owner = int(signs[0] > 0.)
        parcel = SurfaceParcel(index, owner, f"octant-material:{index}", area, 2.*area, 8.*area,
            area*(1e12 if tied or owner else 2e12), 5. if tied or owner else 10.,
            (("damage", .8), ("water", .5)))
        fragments.append(GeometricFragment(f"octant:{index}", tuple(map(tuple, polygon)), parcel))
    return mesh, GeometricSurfaceState(0., radius, tuple(fragments))


def birth(cell, plate, area, event_time, serial):
    return SurfaceParcel(cell, plate, f"birth:{event_time.hex()}:{serial}", area, 2.*area,
                         0., 0., 0., (("damage", 0.), ("water", 0.)))


def declared_fault(contact, *, subducting=1, locked=False, suffix="", shear_pa=1e7):
    radial = np.asarray(contact.midpoint)
    over = 1-subducting
    horizontal = np.asarray(contact.normal_a_to_b)*(1 if contact.plate_b == over else -1)
    # Compression-positive Pa. The resolved normal stress is 123.660254 MPa
    # and down-dip shear 30.980762 MPa at dip30deg for the admitted candidate.
    tensor = 1e8*np.eye(3)
    if not locked:
        tensor += 6e7*np.outer(horizontal, horizontal)
        tensor += shear_pa*(np.outer(horizontal, radial)+np.outer(radial, horizontal))
    return OrientedFault(f"synthetic:{contact.contact_id}:{subducting}{suffix}", contact.contact_id,
        subducting, over, math.pi/6., tensor, 2e6, .2,
        "explicit synthetic oriented dipping fault and prescribed effective stress")


def faults_for(state, *, omega=MOTION, locked=False, both=False):
    contacts = extract_contacts(state.fragments, state.radius_km, omega)
    # The symmetric control gives both conjugate planes the same global tensor;
    # opposite horizontal directions then have identical normal/shear loading.
    faults = tuple(declared_fault(c, subducting=sub, locked=locked, shear_pa=0. if both else 1e7)
                   for c in contacts for sub in ((1, 0) if both else (1,)))
    return contacts, faults


def snapshot(value):
    return json.dumps(asdict(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def grouped(parcels, field):
    groups = {}
    for parcel in parcels:
        groups.setdefault(parcel.material_id, []).append(getattr(parcel, field))
    return {key: math.fsum(values) for key, values in groups.items()}


def ledger(initial, final, inventory, births):
    maximum = {field: 0. for field in EXTENSIVE_FIELDS}
    for field in EXTENSIVE_FIELDS:
        groups = [grouped(parcels, field) for parcels in (
            (f.parcel for f in initial.fragments), (f.parcel for f in final.fragments),
            (c.parcel for c in inventory.cohorts), (f.parcel for f in births))]
        for identity in set().union(*(group.keys() for group in groups)):
            old, remaining, removed, born = [group.get(identity, 0.) for group in groups]
            residual = abs(remaining+removed-old-born)/max(old+born, 1.)
            assert residual < 5e-12
            maximum[field] = max(maximum[field], residual)
    return maximum


def blocked_case(mesh, state, faults, *, omega=MOTION, history=()):
    resolver, audit = make_forced_underthrust_resolver(state, omega, DT,
                                                      faults=faults, history=history)
    original = snapshot(state)
    calls = []
    def forbidden_birth(*args):
        calls.append(args)
        raise AssertionError("Rejected admission called the material birth factory")
    try:
        advance_geometric_surface(mesh, state, omega, DT, birth_factory=forbidden_birth,
                                  polarity_resolver=resolver)
    except UnresolvedPolarityError as error:
        assert not calls and snapshot(state) == original
        return dict(status="blocked", material_unchanged=True, birth_calls=0,
                    contact_status_counts=audit["contact_status_counts"],
                    overlap_reasons=dict(Counter(item.reason for item in error.overlaps)),
                    unresolved_overlap_area_km2=math.fsum(item.area_km2 for item in error.overlaps))
    raise AssertionError("Expected the synthetic control to reject admission")


def code_hashes():
    names = ["tectonics/"+name+".py" for name in (
        "fractional_surface", "geometric_surface", "geometric_contacts", "geometric_transport",
        "geometric_polarity", "geometric_boundary", "geometric_boundary_io", "spherical_polygons",
        "geometric_initiation", "geometric_initiation_adapter", "mesh")]
    names.append(str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"))
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


def execute(output):
    started, hashes = time.perf_counter(), code_hashes()
    output = Path(output)
    if output.exists():
        raise ValueError("Validation output must be new; existing results are never overwritten")
    mesh, initial = fixture()
    original = snapshot(initial)
    contacts, faults = faults_for(initial)
    resolver, audit = make_forced_underthrust_resolver(initial, MOTION, DT, faults=faults)
    first = advance_geometric_surface(mesh, initial, MOTION, DT, birth_factory=birth,
                                      polarity_resolver=resolver)
    inventory = consume_geometric_transaction(initial, first, initialize_boundary(initial), MOTION, DT)
    assert first.losses and first.births and inventory.cohorts
    assert all(loss.polarity_basis == "forced_underthrust_admission" for loss in first.losses)
    assert all(loss.fragment.parcel.plate == 1 and loss.receiver_plate == 0 for loss in first.losses)
    _, locked = faults_for(initial, locked=True)
    _, opposite = faults_for(initial, both=True)
    active_ids = {row["contact_id"] for row in audit["contacts"]
                  if row["status"] == "forced_underthrust_admissible"}
    active_faults = tuple(f for f in faults if f.contact_id in active_ids)
    history = tuple(PolarityEvidence(f"old:{c.contact_id}:{sub}", c.contact_id, sub, 1-sub,
                    "synthetic opposed inherited histories") for c in contacts for sub in (0, 1))
    controls = {
        "no_declared_fault": blocked_case(mesh, initial, ()),
        "hydrostatic_locked_fault": blocked_case(mesh, initial, locked),
        "two_admissible_opposed_faults": blocked_case(mesh, initial, opposite),
        "conflicting_inherited_history": blocked_case(mesh, initial, faults, history=history),
        "reverse_velocity_on_initially_admitted_faults": blocked_case(mesh, initial, active_faults, omega=-MOTION),
    }
    output.mkdir(parents=True)
    provenance = json.loads(json.dumps(dict(
        experiment="synthetic prescribed-load forced-underthrust admission", dt_myr=DT,
        force_evolution=False, thermal_evolution=False, faults=fault_snapshot_to_dict(initial, faults),
        motions=dict(differential=MOTION.tolist(), common=COMMON.tolist()))))
    save_boundary_checkpoint(output/"accepted.json", first.state, inventory, provenance=provenance)
    loaded, loaded_inventory, restored_provenance = load_boundary_checkpoint(output/"accepted.json")
    assert restored_provenance == provenance
    direct_resolver, _ = make_forced_underthrust_resolver(first.state, COMMON, DT)
    resumed_resolver, _ = make_forced_underthrust_resolver(loaded, COMMON, DT)
    direct = advance_geometric_surface(mesh, first.state, COMMON, DT, birth_factory=birth,
                                      polarity_resolver=direct_resolver)
    resumed = advance_geometric_surface(mesh, loaded, COMMON, DT, birth_factory=birth,
                                       polarity_resolver=resumed_resolver)
    direct_inventory = consume_geometric_transaction(first.state, direct, inventory, COMMON, DT)
    resumed_inventory = consume_geometric_transaction(loaded, resumed, loaded_inventory, COMMON, DT)
    assert direct.losses == direct.births == resumed.losses == resumed.births == ()
    assert snapshot(direct.state) == snapshot(resumed.state)
    assert boundary_to_dict(direct_inventory) == boundary_to_dict(resumed_inventory)
    save_boundary_checkpoint(output/"direct_final.json", direct.state, direct_inventory, provenance=provenance)
    save_boundary_checkpoint(output/"resumed_final.json", resumed.state, resumed_inventory, provenance=provenance)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    checkpoint_hashes = {name: digest(output/name) for name in (
        "accepted.json", "direct_final.json", "resumed_final.json")}
    assert checkpoint_hashes["direct_final.json"] == checkpoint_hashes["resumed_final.json"]
    assert snapshot(initial) == original
    report = dict(format="geometric-forced-underthrust-synthetic-validation-1", status="complete",
        elapsed_seconds=time.perf_counter()-started, source_integration_cells=mesh.cell_count,
        source_octant_fragments=len(initial.fragments), admitted=audit, controls=controls,
        accepted_boundary=boundary_diagnostics(inventory), final_boundary=boundary_diagnostics(direct_inventory),
        removed=totals(loss.fragment for loss in first.losses), born=totals(first.births),
        maximum_relative_material_ledger_residuals=ledger(initial, direct.state, direct_inventory, first.births),
        source_preserved=True, exact_restart=True, checkpoint_bytes_identical=True,
        checkpoint_sha256=checkpoint_hashes, code_sha256=hashes, code_unchanged=hashes == code_hashes(),
        scope="Synthetic prescribed loading on declared faults, not spontaneous or self-sustaining "
              "subduction; no feedback to plate forces or speed, no old source import.")
    assert report["code_unchanged"]
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_suffix(""))
    args = parser.parse_args()
    result = execute(args.out)
    print(json.dumps({key: result[key] for key in (
        "status", "elapsed_seconds", "removed", "maximum_relative_material_ledger_residuals",
        "exact_restart", "checkpoint_bytes_identical", "code_unchanged")}, indent=2))
