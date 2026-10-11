"""Accepted-source orientation families in the full current material chart.

The inherited initiation rule calculates a face orientation, not an offset
or a physical dip.  This module exposes all admissible world-plane offsets
through the live material.  It never selects a centroid, a mesh edge, or a
surface from that family.  Explicit sections remain computational probes.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from .genesis_volume_checkpoint import _canonical
from .genesis_volume_fracture_onset import (
    bind_calculated_volume_fracture_onset,
    calculated_volume_fracture_onset_to_dict,
)
from .genesis_volume_slip_dynamics import VolumeSlipModel, VolumeSlipState


FORMAT = "genesis-source-bound-material-plane-location-family-0.1"
SCOPE = "calculated_orientation_full_current_offset_family_without_surface_selection"
CHART = "x(N,lambda)=N@(lambda*(X+P)+(A-P))"


def _digest(body):
    return hashlib.sha256(_canonical(body)).hexdigest()


@dataclass(frozen=True, slots=True)
class VolumeMaterialLocation:
    payload_json: str
    binding_sha256: str


def _cohort_offset_family(cohort, triangle, amplitudes, normal):
    """Exact range of a bilinear function on triangle times radial interval."""
    X, U, n = (np.asarray(value, float) for value in (triangle, amplitudes, normal))
    if (X.shape != (3, 3) or U.shape != (3, 6) or n.shape != (3,)
            or not all(np.isfinite(value).all() for value in (X, U, n))
            or not np.isclose(np.linalg.norm(n), 1., rtol=0, atol=3e-12)):
        raise ValueError("Material location requires finite six-DOF geometry and unit orientation")
    inner, outer = float(cohort.inner_scale), float(cohort.outer_scale)
    if not 0 < inner < outer <= 1:
        raise ValueError("Material location requires a positive live cohort interval")
    Y, Z = X+U[:, 3:], U[:, :3]-U[:, 3:]
    y, z = Y@n, Z@n
    corner_offsets = np.stack((inner*y+z, outer*y+z))
    return dict(cohort_id=int(cohort.id), face_index=int(cohort.face_index),
        layer_index=int(cohort.layer_index), material_id=cohort.material_id,
        owner_id=cohort.owner_id, mass_coordinate_q_bounds=[cohort.q_lo, cohort.live_q_hi],
        lambda_bounds=[inner, outer], projected_Y_m=y.tolist(), projected_Z_m=z.tolist(),
        corner_offsets_m=corner_offsets.tolist(),
        offset_bounds_m=[float(corner_offsets.min()), float(corner_offsets.max())],
        range_scope="closed_geometric_range_including_degenerate_touches_not_release_admission")


def bind_volume_material_location(model, state, *, observer=None, onset=None):
    """Bind the orientation's complete offset family to an accepted runtime.

    An untrusted planetary archive first passes the public full checkpoint
    replay.  Cached onset/observer objects are checked against this runtime;
    they cannot introduce an event, pose, stress, or localization rule.
    """
    if type(model) is not VolumeSlipModel or type(state) is not VolumeSlipState:
        raise ValueError("Material location requires accepted Genesis slip model/state")
    expected = bind_calculated_volume_fracture_onset(model, state)
    raw = calculated_volume_fracture_onset_to_dict(expected)
    if onset is not None and _canonical(calculated_volume_fracture_onset_to_dict(onset)) != _canonical(raw):
        raise ValueError("Material location onset differs from accepted source recomputation")
    if observer is not None:
        from .genesis_volume_material_stress import (
            bind_volume_material_stress, volume_material_stress_observer_to_dict)
        actual_observer = volume_material_stress_observer_to_dict(observer)
        expected_observer = volume_material_stress_observer_to_dict(
            bind_volume_material_stress(model, state))
        if _canonical(actual_observer) != _canonical(expected_observer):
            raise ValueError("Material location observer differs from accepted source recomputation")
    vertices = np.asarray(state.geometry.vertices_m, float)
    faces = np.asarray(state.geometry.faces, int)
    amplitudes = np.asarray(state.displacement_m, float).reshape(-1, 6)
    prepared = raw["prepared_fault_history"]
    active = np.asarray(prepared["fault_active"], bool)
    normals = np.asarray(raw["prepared_plane_normal_world"], float)
    new = set(raw["newly_activating_faces"])
    families = []
    for face in np.flatnonzero(active):
        cohorts = [_cohort_offset_family(c, vertices[faces[face]], amplitudes[faces[face]], normals[face])
            for c in state.geometry.cohorts if c.face_index == face]
        if not cohorts:
            raise ValueError("Calculated orientation has no live material support")
        families.append(dict(face_index=int(face), normal_world=normals[face].tolist(),
            orientation_status="prospective_new_activation" if face in new else "inherited_active_weak_plane",
            orientation_scope="top_face_2d_covector_not_calculated_physical_dip",
            offset_bounds_m=[min(c["offset_bounds_m"][0] for c in cohorts),
                             max(c["offset_bounds_m"][1] for c in cohorts)],
            cohort_offset_families=cohorts,
            selected_offset_m=None, selected_nucleation_point=None, physical_surface_selected=False))
    body = dict(format=FORMAT, scope=SCOPE, current_material_chart=CHART,
        parent_checkpoint_sha256=raw["parent_checkpoint_sha256"],
        formation_checkpoint_sha256=raw["formation_checkpoint_sha256"],
        source_identity=raw["source_identity"], time_myr=raw["time_myr"],
        accepted_steps=raw["accepted_steps"], accepted_step_sizes_myr=raw["accepted_step_sizes_myr"],
        onset_sha256=raw["onset_sha256"], source_new_activation_event=raw["new_activation_event"],
        newly_activating_faces=raw["newly_activating_faces"],
        existing_active_faces=raw["existing_active_faces"], orientation_offset_families=families,
        status=("no_calculated_activation_event" if not families
                else "orientation_offset_families_require_explicit_nucleation_and_dip_law"),
        source_nucleation_location_law=None, source_physical_dip_law=None,
        source_surface_creation_energy_law=None, selected_sharp_material_surface=None,
        unresolved_physical_inputs=["source_nucleation_location_and_extent",
            "source_through_depth_dip_and_propagation", "source_surface_creation_energy_and_funding",
            "physical_insertion_release_work_and_contact_admission"],
        inherited_runtime_ledgers=raw["inherited_runtime_ledgers"],
        physical_creation_admitted=False, physical_release_admitted=False,
        physical_fault_contact_admission_claimed=False, history_reinitialized=False,
        activation_accepted=False, clock_advanced=False, creation_work_debited_from_Genesis_j=0.)
    payload = {**body, "location_sha256": _digest(body)}
    text = _canonical(payload).decode()
    return VolumeMaterialLocation(text, hashlib.sha256(text.encode()).hexdigest())


def volume_material_location_to_dict(location):
    if (type(location) is not VolumeMaterialLocation or location.binding_sha256 !=
            hashlib.sha256(location.payload_json.encode()).hexdigest()):
        raise ValueError("Material location lost its immutable source binding")
    raw = json.loads(location.payload_json)
    if (raw.get("format") != FORMAT or raw.get("scope") != SCOPE
            or raw.get("location_sha256") != _digest({k: v for k, v in raw.items() if k != "location_sha256"})):
        raise ValueError("Material location descriptor seal or scope differs")
    for name in ("physical_creation_admitted", "physical_release_admitted",
                 "physical_fault_contact_admission_claimed", "history_reinitialized",
                 "activation_accepted", "clock_advanced"):
        if raw.get(name) is not False:
            raise ValueError("An orientation family cannot claim physical surface admission")
    if (any(raw.get(name) is not None for name in ("source_nucleation_location_law",
             "source_physical_dip_law", "source_surface_creation_energy_law", "selected_sharp_material_surface"))
            or type(raw.get("creation_work_debited_from_Genesis_j")) is not float
            or raw["creation_work_debited_from_Genesis_j"] != 0.):
        raise ValueError("An orientation family cannot invent a physical localization/energy law")
    return raw


def volume_material_location_from_dict(raw, *, model, state):
    expected = bind_volume_material_location(model, state)
    if type(raw) is not dict or _canonical(raw) != _canonical(volume_material_location_to_dict(expected)):
        raise ValueError("Material location differs from complete accepted-source recomputation")
    return expected
