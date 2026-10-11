"""Calculated activation prerequisite bound to an accepted Genesis state.

This uses the same accepted-start stress and activation rule as the slip
solver. An orientation is not a located sharp material surface. No accepted
history, clock, constraint or energy ledger is changed by this witness.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from .genesis_volume_checkpoint import _canonical
from .genesis_volume_fault_evolution import (
    prepare_volume_fault_activation, volume_fault_history_to_dict)
from .genesis_volume_slip_checkpoint import volume_slip_checkpoint_to_dict
from .genesis_volume_slip_dynamics import VolumeSlipModel, VolumeSlipState, _lift_plane, _vertices

FORMAT = "genesis-volume-calculated-fracture-onset-prerequisite-0.1"
SCOPE = "accepted_start_weak_plane_event_not_located_sharp_surface_or_release"


def _seal(body):
    return hashlib.sha256(_canonical(body)).hexdigest()


@dataclass(frozen=True, slots=True)
class CalculatedVolumeFractureOnset:
    payload_json: str
    binding_sha256: str


def bind_calculated_volume_fracture_onset(model, state):
    """Recompute the next accepted-start event without accepting it.

    This is a trusted runtime entry point. An untrusted planetary checkpoint
    must first undergo the public slip checkpoint's full deterministic replay.
    Caller-supplied stress, ready masks or edge cuts are never inputs here.
    """
    if type(model) is not VolumeSlipModel or type(state) is not VolumeSlipState:
        raise ValueError("Calculated onset requires accepted Genesis slip model/state")
    checkpoint = volume_slip_checkpoint_to_dict(model, state)
    stress = model.face_response(state)["stress_pa"]
    candidate, event = prepare_volume_fault_activation(
        state.fault_history, model.source_packet, stress)
    new_faces = np.flatnonzero(candidate.fault_active & ~state.fault_history.fault_active)
    normal, tangent = _lift_plane(candidate.plane_normal, candidate.fault_active,
                                 _vertices(state), state.geometry.faces)
    cohorts = [dict(cohort_id=int(c.id), face_index=int(c.face_index),
                    owner_id=c.owner_id, material_id=c.material_id)
               for c in state.geometry.cohorts]
    runtime = checkpoint["runtime_state"]
    body = dict(format=FORMAT, scope=SCOPE,
        parent_checkpoint_sha256=checkpoint["checkpoint_sha256"],
        formation_checkpoint_sha256=checkpoint["source_checkpoint_sha256"],
        source_identity=state.geometry.source_identity, time_myr=state.time_myr,
        accepted_step_sizes_myr=list(state.accepted_step_sizes_myr),
        accepted_steps=state.mechanics.accepted_steps,
        event_evaluation_scope="next_interval_start_at_current_accepted_state",
        computed_face_stress_pa=np.asarray(stress).tolist(),
        source_fault_parameters=asdict(model.fault_parameters),
        accepted_fault_history=volume_fault_history_to_dict(state.fault_history),
        prepared_fault_history=volume_fault_history_to_dict(candidate),
        new_activation_event=event, newly_activating_faces=new_faces.tolist(),
        existing_active_faces=np.flatnonzero(state.fault_history.fault_active).tolist(),
        prepared_plane_normal_world=normal.tolist(),
        prepared_slip_direction_world=tangent.tolist(), live_cohorts=cohorts,
        inherited_runtime_ledgers={k: v for k, v in runtime.items() if k.endswith("_j")},
        material_surface_status=("no_calculated_activation_event" if not len(new_faces)
            else "missing_in_face_location_and_through_depth_surface"),
        selected_sharp_material_surface=None, source_creation_energy_law=None,
        physical_creation_admitted=False, physical_release_admitted=False,
        physical_fault_contact_admission_claimed=False,
        creation_work_debited_from_Genesis_j=0., history_reinitialized=False,
        activation_accepted=False, clock_advanced=False)
    payload = {**body, "onset_sha256": _seal(body)}
    text = _canonical(payload).decode()
    return CalculatedVolumeFractureOnset(text, hashlib.sha256(text.encode()).hexdigest())


def calculated_volume_fracture_onset_to_dict(onset):
    if (type(onset) is not CalculatedVolumeFractureOnset
            or onset.binding_sha256 != hashlib.sha256(onset.payload_json.encode()).hexdigest()):
        raise ValueError("Calculated onset lost its immutable accepted-state binding")
    raw = json.loads(onset.payload_json)
    body = {k: v for k, v in raw.items() if k != "onset_sha256"}
    if raw.get("onset_sha256") != _seal(body):
        raise ValueError("Calculated onset seal differs")
    for name in ("physical_creation_admitted", "physical_release_admitted",
                 "physical_fault_contact_admission_claimed", "history_reinitialized",
                 "activation_accepted", "clock_advanced"):
        if raw.get(name) is not False:
            raise ValueError("Calculated onset cannot claim accepted physical creation/release")
    if (raw.get("format") != FORMAT or raw.get("scope") != SCOPE
            or raw.get("selected_sharp_material_surface") is not None
            or raw.get("source_creation_energy_law") is not None
            or type(raw.get("creation_work_debited_from_Genesis_j")) is not float
            or raw["creation_work_debited_from_Genesis_j"] != 0.):
        raise ValueError("Calculated onset changed prerequisite semantics")
    return raw


def calculated_volume_fracture_onset_from_dict(raw, *, model, state):
    """A resealed DTO is accepted only if the entire source recomputation agrees."""
    expected = bind_calculated_volume_fracture_onset(model, state)
    if type(raw) is not dict or _canonical(raw) != _canonical(
            calculated_volume_fracture_onset_to_dict(expected)):
        raise ValueError("Calculated onset differs from full accepted-source recomputation")
    return expected
