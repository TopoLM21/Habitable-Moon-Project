"""State binding and strict mechanical policy, independent of legacy age order."""
from dataclasses import asdict, replace
import json
import math

import numpy as np
import pytest

from tectonics.fractional_surface import FractionalSurfaceState, SurfaceParcel
from tectonics.geometric_contacts import extract_contacts
from tectonics.geometric_initiation import OrientedFault, evaluate_fault
from tectonics.geometric_initiation_adapter import (fault_snapshot_from_dict,
    fault_snapshot_to_dict, make_forced_underthrust_resolver)
from tectonics.geometric_polarity import PolarityEvidence
from tectonics.geometric_surface import from_fractional_surface, rotate_surface
from tectonics.mesh import build_icosphere


OMEGA = np.array([[0., 0., 0.], [.010134052286718277, -.005125323519132389, .0014618146771253283]])


def source():
    mesh = build_icosphere(0)
    areas = mesh.physical_cell_areas_km2(100.)
    owners = (mesh.centroids[:, 0] > 0.).astype(int)
    parcels = tuple(SurfaceParcel(i, int(owner), f"origin:{i}", float(area),
        2.*area, 10.*area, 1e12*area, 5.) for i, (area, owner) in enumerate(zip(areas, owners)))
    return from_fractional_surface(mesh, FractionalSurfaceState(0., tuple(areas), parcels), 100.)


def inputs():
    state = source()
    contacts = extract_contacts(state.fragments, state.radius_km, OMEGA)
    for contact in contacts:
        n = np.asarray(contact.normal_a_to_b)
        f = OrientedFault(f"weak:{contact.contact_id}", contact.contact_id,
            contact.plate_a, contact.plate_b, math.pi/6,
            np.eye(3)*10e6+100e6*np.outer(n, n), 1e5, .1, "declared analytic loading")
        if evaluate_fault(f, contact, OMEGA, state.radius_km).status == "forced_underthrust_admissible":
            return state, contacts, f
    raise AssertionError("No convergent analytical fixture")


def row(audit, identity):
    return next(r for r in audit["contacts"] if r["contact_id"] == identity)


def test_fault_snapshot_json_roundtrip_is_exact_and_state_bound():
    state, _, f = inputs()
    raw = json.loads(json.dumps(fault_snapshot_to_dict(state, (f,)), allow_nan=False))
    assert fault_snapshot_from_dict(raw, state) == (f,)
    with pytest.raises(ValueError, match="stale"):
        fault_snapshot_from_dict(raw, replace(state, time_myr=.1))
    moved = rotate_surface(state, np.array([[.01, .02, .03]]*2), .1)
    moved = replace(moved, time_myr=state.time_myr)
    with pytest.raises(ValueError, match="surface_sha256"):
        fault_snapshot_from_dict(raw, moved)


@pytest.mark.parametrize("name,value", [("time_myr", False), ("radius_km", 1.),
    ("stress_frame", "unlabeled"), ("stress_spatial_model", "globally_uniform_tensor"),
    ("load_evolution", "advect_only_geometry"),
    ("candidate_set_assumption", "infer_other_directions"), ("format", "legacy"),
    ("surface_sha256", "not-current"), ("faults", {})])
def test_snapshot_rejects_wrong_clock_geometry_and_loading_semantics(name, value):
    state, _, f = inputs()
    raw = fault_snapshot_to_dict(state, (f,))
    raw[name] = value
    with pytest.raises(ValueError):
        fault_snapshot_from_dict(raw, state)


def test_snapshot_rejects_unknown_properties_instead_of_ignoring_physics():
    state, _, f = inputs()
    raw = fault_snapshot_to_dict(state, (f,))
    raw["faults"][0]["automatic_pressure_from_water"] = True
    with pytest.raises(ValueError, match="fault record"):
        fault_snapshot_from_dict(raw, state)
    raw = fault_snapshot_to_dict(state, ())
    raw["old_slab_mass"] = 1e10
    with pytest.raises(ValueError, match="unknown"):
        fault_snapshot_from_dict(raw, state)


def test_duplicate_or_nonlocal_fault_identity_is_rejected():
    state, _, f = inputs()
    with pytest.raises(ValueError, match="unique"):
        fault_snapshot_to_dict(state, (f, f))
    with pytest.raises(ValueError, match="current spatial contact"):
        fault_snapshot_to_dict(state, (replace(f, contact_id="remote-contact"),))
    with pytest.raises(ValueError, match="owners"):
        fault_snapshot_to_dict(state, (replace(f, overriding_plate=7),))


def test_contact_manifest_cannot_hide_missing_arcs_or_stale_geometry():
    state, contacts, f = inputs()
    for incorrect in (contacts[:-1], (*contacts, contacts[0])):
        with pytest.raises(ValueError, match="complete and unique"):
            make_forced_underthrust_resolver(state, OMEGA, .1, faults=(f,), contacts=incorrect)
    stale = (replace(contacts[0], length_km=contacts[0].length_km*2.), *contacts[1:])
    with pytest.raises(ValueError, match="stale"):
        make_forced_underthrust_resolver(state, OMEGA, .1, faults=(f,), contacts=stale)


def test_missing_stress_cannot_be_treated_as_a_locked_excluded_conjugate():
    state, contacts, f = inputs()
    unknown = replace(f, fault_id="unknown-opposite", subducting_plate=f.overriding_plate,
        overriding_plate=f.subducting_plate, effective_stress_pa=None)
    _, audit = make_forced_underthrust_resolver(state, OMEGA, .1, faults=(f, unknown), contacts=contacts)
    assert row(audit, f.contact_id)["status"] == "missing_fault_stress"
    assert not audit["accepted_evidence"]


def test_history_may_select_an_admissible_conjugate_but_cannot_unlock_a_fault():
    state, contacts, f = inputs()
    opposite = replace(f, fault_id="other-conjugate", subducting_plate=f.overriding_plate,
                       overriding_plate=f.subducting_plate)
    history = (PolarityEvidence("previous-direction", f.contact_id, f.subducting_plate,
                               f.overriding_plate, "accepted local history"),)
    _, audit = make_forced_underthrust_resolver(state, OMEGA, .1, faults=(f, opposite), history=history)
    assert row(audit, f.contact_id)["status"] == "forced_underthrust_admissible"
    assert {(r["subducting_plate"], r["overriding_plate"]) for r in audit["accepted_evidence"]} == {
        (f.subducting_plate, f.overriding_plate)}
    locked = replace(f, cohesion_pa=1e12)
    _, denied = make_forced_underthrust_resolver(state, OMEGA, .1, faults=(locked,), history=history)
    assert row(denied, f.contact_id)["status"] == "history_mechanics_mismatch"
    assert not denied["accepted_evidence"]


def test_conflicting_history_cannot_be_overwritten_by_new_favorable_loading():
    state, _, f = inputs()
    history = tuple(PolarityEvidence(f"history:{sub}", f.contact_id, sub, over, "previous condition")
        for sub, over in ((f.subducting_plate, f.overriding_plate), (f.overriding_plate, f.subducting_plate)))
    before = asdict(state)
    _, audit = make_forced_underthrust_resolver(state, OMEGA, .1, faults=(f,), history=history)
    assert row(audit, f.contact_id)["status"] == "conflicting_local_history"
    assert not audit["accepted_evidence"]
    assert asdict(state) == before


def test_candidate_order_does_not_change_audit_or_direction():
    state, _, f = inputs()
    opposite = replace(f, fault_id="other-conjugate", subducting_plate=f.overriding_plate,
                       overriding_plate=f.subducting_plate, cohesion_pa=1e12)
    _, first = make_forced_underthrust_resolver(state, OMEGA, .1, faults=(f, opposite))
    _, second = make_forced_underthrust_resolver(state, OMEGA, .1, faults=(opposite, f))
    assert first == second
    assert row(first, f.contact_id)["status"] == "forced_underthrust_admissible"
