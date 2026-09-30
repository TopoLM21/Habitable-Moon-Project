"""Strict local forced-underthrust admission for a prescribed-motion probe.

The imposed fault set and effective stress are explicit physical boundary
conditions. Coulomb reactivation does not establish a self-sustaining slab.
An inadmissible overlap is rejected atomically: no shortening solver is hidden
in this adapter and no age/buoyancy fallback grants permission to remove mass.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math

from .geometric_boundary import surface_digest
from .geometric_contacts import extract_contacts
from .geometric_initiation import OrientedFault, evaluate_fault
from .geometric_polarity import (PolarityEvidence, PolarityResolution,
    contact_polarity_report, make_polarity_resolver)
from .geometric_surface import rotations


FORMAT = "geometric-forced-underthrust-input-1"
STRESS_FRAME = "global_cartesian_effective_compression_positive_pa"
STRESS_SPATIAL_MODEL = "contactwise_parallel_transport_from_midpoint"
LOAD_EVOLUTION = "single_step_frozen_snapshot"
CANDIDATE_SET = "only_declared_oriented_weak_faults"
_SPATIAL = ("fragment_a", "fragment_b", "material_a", "material_b", "plate_a", "plate_b",
            "start", "end", "normal_a_to_b", "length_km", "midpoint", "tangent")


def _validated_faults(state, faults, contacts):
    faults = tuple(faults)
    by_id = {c.contact_id: c for c in contacts}
    seen = set()
    for fault in faults:
        if not isinstance(fault, OrientedFault):
            raise ValueError("Mechanical admission requires explicit OrientedFault records")
        if fault.fault_id in seen:
            raise ValueError("Oriented fault IDs must be unique")
        seen.add(fault.fault_id)
        contact = by_id.get(fault.contact_id)
        if contact is None or {contact.plate_a, contact.plate_b} != {
                fault.subducting_plate, fault.overriding_plate}:
            raise ValueError("Oriented fault must name a current spatial contact and its owners")
    return tuple(sorted(faults, key=lambda f: f.fault_id))


def fault_snapshot_to_dict(state, faults):
    """Bind a declared load/fault set to exactly one initial surface state.

    The same data may be used in a step-size study from this state. It cannot
    be reused after motion or warming changes that state. The input is a
    frozen-load experiment, not a transported or dynamically evolved stress.
    """
    if state.phase != "partition":
        raise ValueError("Fault snapshot requires a resolved surface partition")
    faults = _validated_faults(state, faults, extract_contacts(state.fragments, state.radius_km))
    return dict(format=FORMAT, time_myr=state.time_myr, radius_km=state.radius_km,
        surface_sha256=surface_digest(state), stress_frame=STRESS_FRAME,
        stress_spatial_model=STRESS_SPATIAL_MODEL,
        load_evolution=LOAD_EVOLUTION, candidate_set_assumption=CANDIDATE_SET,
        faults=[asdict(fault) for fault in faults])


def fault_snapshot_from_dict(raw, state):
    if not isinstance(raw, dict):
        raise ValueError("Fault snapshot must be an object")
    expected = fault_snapshot_to_dict(state, ())
    if set(raw) != set(expected):
        raise ValueError("Fault snapshot has missing or unknown fields")
    for key, value in expected.items():
        if key == "faults":
            continue
        supplied = raw[key]
        if isinstance(supplied, bool) or supplied != value:
            raise ValueError(f"Fault snapshot is stale or has incompatible {key}")
    if not isinstance(raw["faults"], list):
        raise ValueError("Fault snapshot faults must be a list")
    try:
        faults = tuple(OrientedFault(**record) for record in raw["faults"])
    except (TypeError, KeyError) as error:
        raise ValueError("Invalid oriented fault record") from error
    return _validated_faults(state, faults, extract_contacts(state.fragments, state.radius_km))


def _admission_evidence(fault):
    payload = json.dumps(asdict(fault), sort_keys=True, separators=(",", ":"), allow_nan=False)
    suffix = hashlib.sha256(payload.encode()).hexdigest()[:24]
    return PolarityEvidence(f"fault-admission:{fault.fault_id}:{suffix}", fault.contact_id,
        fault.subducting_plate, fault.overriding_plate,
        f"prescribed_effective_stress_forced_underthrust:{fault.provenance}")


def make_forced_underthrust_resolver(state, omega_rad_per_myr, dt_myr, *,
                                    faults=(), contacts=None, history=()):
    """Return a strict resolver and a JSON-native audit of all real contacts.

    A known orientation may select among mechanically admissible conjugates;
    it never bypasses a locked fault. Conflicting history is retained. With
    no history, two admissible directions stay ambiguous even when their
    margins differ: no unproved largest-margin selection law is introduced.

    Every contact touching a candidate overlap must grant consistent local
    permission. When a portion requires separate treatment the whole proposed
    transaction is blocked rather than assigning its mass arbitrarily.
    """
    if state.phase != "partition":
        raise ValueError("Mechanical admission requires a resolved partition")
    if isinstance(dt_myr, bool) or not math.isfinite(float(dt_myr)) or dt_myr <= 0.:
        raise ValueError("Mechanical admission requires a positive finite time step")
    omega = rotations(state, omega_rad_per_myr, float(dt_myr))
    canonical = extract_contacts(state.fragments, state.radius_km, omega)
    by_id = {c.contact_id: c for c in canonical}
    if contacts is not None:
        contacts = tuple(contacts)
        supplied = {c.contact_id: c for c in contacts}
        if len(supplied) != len(contacts) or set(supplied) != set(by_id):
            raise ValueError("Mechanical contact manifest must be complete and unique")
        for identity, c in supplied.items():
            if any(getattr(c, key) != getattr(by_id[identity], key) for key in _SPATIAL):
                raise ValueError("Mechanical contact geometry is stale")
    faults = _validated_faults(state, faults, canonical)
    history = tuple(history)
    historical = {r["contact_id"]: r for r in contact_polarity_report(canonical, history)}
    grouped = {identity: [] for identity in by_id}
    for fault in faults:
        grouped[fault.contact_id].append(fault)
    rows, evidence, status = [], [], {}
    for contact in canonical:
        candidates = grouped[contact.contact_id]
        assessments = [evaluate_fault(f, contact, omega, state.radius_km) for f in candidates]
        record = historical.get(contact.contact_id)
        old_directions = set() if record is None else {tuple(pair) for pair in record["directions"]}
        admitted = [fault for fault, assessment in zip(candidates, assessments)
                    if assessment.status == "forced_underthrust_admissible"]
        directions = {(f.subducting_plate, f.overriding_plate) for f in admitted}
        chosen = []
        if len(old_directions) > 1:
            reason = "conflicting_local_history"
        elif not candidates:
            reason = "no_oriented_fault"
        elif any(a.status == "missing_stress" for a in assessments):
            # An unassessed opposite candidate cannot be silently excluded.
            reason = "missing_fault_stress"
        elif old_directions and not old_directions.issubset(directions):
            reason = "history_mechanics_mismatch"
        elif len(directions) > 1 and not old_directions:
            reason = "multiple_admissible_directions"
        elif directions:
            selected = old_directions or directions
            chosen = [f for f in admitted if (f.subducting_plate, f.overriding_plate) in selected]
            reason = "forced_underthrust_admissible"
        else:
            reasons = {a.status for a in assessments}
            reason = next(iter(reasons)) if len(reasons) == 1 else "no_admissible_fault"
        accepted = [_admission_evidence(f) for f in chosen]
        evidence.extend(accepted)
        status[contact.contact_id] = reason
        rows.append(dict(contact_id=contact.contact_id, status=reason,
            historical_directions=[list(pair) for pair in sorted(old_directions)],
            assessments=[asdict(a) for a in assessments],
            accepted_evidence_ids=[e.evidence_id for e in accepted]))

    # Two virtual orientations are used only to locate the support of each
    # actual arc in either receiver frame. They can never grant admission.
    scope = tuple(PolarityEvidence(f"scope:{c.contact_id}:{sub}", c.contact_id, sub, over,
                                  "geometric_scope_only_not_physical_evidence")
                  for c in canonical for sub, over in ((c.plate_a, c.plate_b), (c.plate_b, c.plate_a)))
    locate = make_polarity_resolver(state, omega, dt_myr, contacts=canonical, evidence=scope)
    permitted = make_polarity_resolver(state, omega, dt_myr, contacts=canonical, evidence=evidence)

    def resolve(a, b, overlap):
        local = locate(a, b, overlap)
        if local is None:
            return PolarityResolution(0, "no_resolved_initial_contact")
        reasons = {status[identity] for identity in local.contact_ids}
        blocked = reasons - {"forced_underthrust_admissible"}
        if blocked:
            reason = ("conflicting_local_history" if "conflicting_local_history" in blocked else
                      next(iter(blocked)) if len(blocked) == 1 else "unresolved_local_admission")
            return PolarityResolution(0, reason, (), local.contact_ids)
        decision = permitted(a, b, overlap)
        if decision is None:
            return PolarityResolution(0, "no_local_downdip_support", (), local.contact_ids)
        if not decision.comparison:
            return PolarityResolution(0, "multiple_local_underthrust_directions",
                                      decision.evidence_ids, decision.contact_ids)
        return PolarityResolution(decision.comparison, "forced_underthrust_admission",
                                  decision.evidence_ids, decision.contact_ids)

    audit = dict(format="geometric-forced-underthrust-audit-1", time_myr=state.time_myr,
        dt_myr=float(dt_myr), surface_sha256=surface_digest(state),
        stress_frame=STRESS_FRAME, stress_spatial_model=STRESS_SPATIAL_MODEL,
        load_evolution=LOAD_EVOLUTION,
        candidate_set_assumption=CANDIDATE_SET,
        scope="Local forced reactivation of declared dipping faults; no bending or self-sustaining initiation solve",
        overlap_policy="Every local support must mechanically admit the same direction; no age/buoyancy fallback",
        blocked_motion="Atomic rejection; compression, shortening and contact reactions are not evolved",
        faults=[asdict(f) for f in faults], contacts=rows,
        contact_status_counts={reason: sum(r["status"] == reason for r in rows)
                               for reason in sorted(set(status.values()))},
        accepted_evidence=[asdict(e) for e in evidence])
    return resolve, audit
