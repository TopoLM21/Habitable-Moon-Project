"""Reproducible positive geometry-to-passive-slab experiment, without forces."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from tectonics.fractional_surface import EXTENSIVE_FIELDS, FractionalSurfaceState, SurfaceParcel
from tectonics.geometric_boundary import (boundary_diagnostics, boundary_to_dict,
    consume_geometric_transaction, initialize_boundary, polarity_evidence)
from tectonics.geometric_boundary_io import load_boundary_checkpoint, save_boundary_checkpoint
from tectonics.geometric_surface import from_fractional_surface, totals
from tectonics.geometric_transport import advance_geometric_surface
from tectonics.mesh import build_icosphere


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_hashes():
    names = ["tectonics/"+name+".py" for name in (
        "fractional_surface", "geometric_surface", "geometric_contacts", "geometric_transport",
        "geometric_polarity", "geometric_boundary", "geometric_boundary_io", "spherical_polygons", "mesh")]
    names.append(str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"))
    return {name: digest(ROOT/name) for name in names}


def fixture():
    mesh = build_icosphere(0)
    areas = mesh.physical_cell_areas_km2(100.)
    owners = (mesh.centroids[:, 0] > 0.).astype(int)
    parcels = tuple(SurfaceParcel(i, int(owner), f"history:{i}", float(area),
        float(area)*(2.+i/20.), float(area)*(8.+i/10.),
        float(area)*(1e12 if owner else 2e12), 5. if owner else 10.,
        (("damage", i/20.),)) for i, (area, owner) in enumerate(zip(areas, owners)))
    return mesh, from_fractional_surface(mesh,
        FractionalSurfaceState(0., tuple(areas), parcels), 100.)


def birth(cell, plate, area, event_time, serial):
    return SurfaceParcel(cell, plate, f"birth:{event_time.hex()}:{serial}", area, 2.*area,
                         0., 0., 0., (("damage", 0.),))


def grouped(parcels, field):
    groups = {}
    for parcel in parcels:
        groups.setdefault(parcel.material_id, []).append(getattr(parcel, field))
    return {key: math.fsum(values) for key, values in groups.items()}


def check_ledger(initial, final, boundary, births):
    maximum = {field: 0. for field in EXTENSIVE_FIELDS}
    for field in EXTENSIVE_FIELDS:
        ledgers = [grouped(values, field) for values in (
            (f.parcel for f in initial.fragments), (f.parcel for f in final.fragments),
            (c.parcel for c in boundary.cohorts), (f.parcel for f in births))]
        for identity in set().union(*(ledger.keys() for ledger in ledgers)):
            old, remaining, removed, created = [ledger.get(identity, 0.) for ledger in ledgers]
            assert math.isclose(old+created, remaining+removed, rel_tol=5e-12, abs_tol=0.)
            maximum[field] = max(maximum[field], abs(remaining+removed-old-created)/max(old+created, 1.))
    return maximum


def execute(output):
    started = time.perf_counter()
    hashes = code_hashes()
    output = Path(output)
    if output.exists():
        raise ValueError("Positive-case output must be new; existing results are never overwritten")
    mesh, initial = fixture()
    initial_snapshot = asdict(initial)
    differential = np.array([[0., 0., 0.], [0., 0., .01]])
    common = np.array([[.002, -.003, .001]]*2)
    dt = .1
    first = advance_geometric_surface(mesh, initial, differential, dt, birth_factory=birth)
    accepted = consume_geometric_transaction(initial, first, initialize_boundary(initial), differential, dt)
    assert first.losses and first.births and accepted.cohorts
    output.mkdir(parents=True)
    provenance = dict(experiment="synthetic geometric passive slab connection", dt_myr=dt,
        thermal_evolution="endpoint archival properties; no surface warming", forces_active=False,
        motions=dict(differential=differential.tolist(), common=common.tolist()))
    save_boundary_checkpoint(output/"accepted.json", first.state, accepted, provenance=provenance)
    loaded_surface, loaded_boundary, loaded_provenance = load_boundary_checkpoint(output/"accepted.json")
    assert loaded_provenance == provenance
    direct = advance_geometric_surface(mesh, first.state, common, dt, birth_factory=birth)
    direct_boundary = consume_geometric_transaction(first.state, direct, accepted, common, dt)
    resumed = advance_geometric_surface(mesh, loaded_surface, common, dt, birth_factory=birth)
    resumed_boundary = consume_geometric_transaction(loaded_surface, resumed, loaded_boundary, common, dt)
    assert asdict(direct.state) == asdict(resumed.state)
    assert boundary_to_dict(direct_boundary) == boundary_to_dict(resumed_boundary)
    assert polarity_evidence(direct_boundary) == polarity_evidence(resumed_boundary)
    assert direct.losses == direct.births == resumed.losses == resumed.births == ()
    assert asdict(initial) == initial_snapshot
    before_cohorts = {c.event_id: c for c in accepted.cohorts}
    after_cohorts = {c.event_id: c for c in direct_boundary.cohorts}
    assert before_cohorts.keys() == after_cohorts.keys()
    for identity, previous in before_cohorts.items():
        current = after_cohorts[identity]
        assert asdict(previous.loss) == asdict(current.loss)
        assert previous.attachment_status == current.attachment_status
        assert previous.unresolved_support == current.unresolved_support
        assert {lineage for support in previous.supports for lineage in support.lineage_ids} == {
            lineage for support in current.supports for lineage in support.lineage_ids}
    try:
        consume_geometric_transaction(initial, first, accepted, differential, dt)
    except ValueError as error:
        replay = dict(rejected=True, reason=str(error))
    else:
        raise AssertionError("Accepted transaction replay was not rejected")
    save_boundary_checkpoint(output/"direct_final.json", direct.state, direct_boundary, provenance=provenance)
    save_boundary_checkpoint(output/"resumed_final.json", resumed.state, resumed_boundary, provenance=provenance)
    checkpoint_hashes = {name: digest(output/name) for name in (
        "accepted.json", "direct_final.json", "resumed_final.json")}
    assert checkpoint_hashes["direct_final.json"] == checkpoint_hashes["resumed_final.json"]
    residual = check_ledger(initial, direct.state, direct_boundary, first.births)
    assert code_hashes() == hashes
    report = dict(status="complete", elapsed_seconds=time.perf_counter()-started,
        scenario=provenance, source_mesh_cells=mesh.cell_count,
        initial_time_myr=initial.time_myr, accepted_time_myr=first.state.time_myr,
        final_time_myr=direct.state.time_myr,
        initial=totals(initial.fragments), remaining=totals(direct.state.fragments),
        removed=totals(loss.fragment for loss in first.losses), born=totals(first.births),
        maximum_relative_material_ledger_residuals=residual,
        accepted_boundary=boundary_diagnostics(accepted), final_boundary=boundary_diagnostics(direct_boundary),
        accepted_event_ids=sorted(before_cohorts), final_event_ids=sorted(after_cohorts),
        source_preserved=True, exact_restart=True, checkpoint_bytes_identical=True,
        cohort_payloads_preserved=True, attachment_statuses_preserved=True,
        support_lineages_preserved=True, replay=replay,
        final_polarity_evidence=polarity_evidence(direct_boundary),
        checkpoint_sha256=checkpoint_hashes, code_unchanged=True, code_sha256=hashes)
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    location = ROOT/"analysis/geometric_slab_validation/passive_connection_case"
    report = execute(location)
    print(json.dumps({key: report[key] for key in (
        "status", "elapsed_seconds", "initial", "remaining", "removed", "born",
        "maximum_relative_material_ledger_residuals", "exact_restart", "checkpoint_bytes_identical",
        "attachment_statuses_preserved", "support_lineages_preserved", "code_unchanged")}, indent=2))
