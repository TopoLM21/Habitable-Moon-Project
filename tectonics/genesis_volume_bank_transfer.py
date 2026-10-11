"""Zero-time transfer from accepted Genesis memory to prospective tied banks.

The context must be a trusted runtime (an archive caller first uses the full
source-to-current slip checkpoint replay). This descriptor is a contextual
certificate, not a replacement checkpoint or a fracture-creation law.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from .genesis_volume_banks import (
    GenesisVolumeBanks, _hash_value, validate_volume_banks)
from .genesis_volume_checkpoint import _canonical
from .genesis_volume_dynamics import coefficient_arrays
from .genesis_volume_operator import _immutable
from .genesis_volume_slip_checkpoint import volume_slip_checkpoint_to_dict
from .genesis_volume_slip_dynamics import VolumeSlipModel, VolumeSlipState

FORMAT = "genesis-volume-bank-transfer-0.1"
SCOPE = "zero_time_accepted_genesis_memory_on_tied_six_dof_support"
UNRESOLVED_PHYSICAL_INPUTS = (
    "calculated_material_fracture_surface_and_through_depth_geometry",
    "paired_physical_depth_intervals_with_cohort_side_ownership",
    "source_calculated_surface_creation_law_and_creation_work",
    "accepted_interval_traction_and_release_balance",
    "both_tangential_contact_modes_and_spatial_underthrust_admission",
)


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _array_digest(value):
    digest = hashlib.sha256()
    _hash_value(digest, np.asarray(value))
    return digest.hexdigest()


def _object_digest(value):
    digest = hashlib.sha256()
    _hash_value(digest, value)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class GenesisVolumeBankTransfer:
    payload_json: str
    displacement_m: np.ndarray
    velocity_m_s: np.ndarray
    elastic_constant: np.ndarray
    elastic_inverse_scale: np.ndarray

    def __post_init__(self):
        for name in ("displacement_m", "velocity_m_s", "elastic_constant",
                     "elastic_inverse_scale"):
            object.__setattr__(self, name, _immutable(getattr(self, name)))


def bind_volume_bank_transfer(model, state, banks=None):
    """Copy current kinematics and memory, without time, birth or cut creation.

    A supplied edge split is an explicit computational support. It cannot
    override physical eligibility or make a sharp surface out of a diffuse
    plane. Absolute displacement is copied for pose; it is NEVER reapplied
    as a strain increment to the already accepted cohort memory.
    """
    if type(model) is not VolumeSlipModel or type(state) is not VolumeSlipState:
        raise ValueError("Bank transfer requires accepted source-bound volume slip context")
    checkpoint = volume_slip_checkpoint_to_dict(model, state)
    parent, basal = model._assembly(state.geometry, state.fault_history.damage)
    e0, e1 = coefficient_arrays(state.mechanics)
    if banks is None:
        q, v = state.displacement_m, state.velocity_m_s
        kind, cut_edges, bank_pin = "original_connected_domain", [], None
        ndof, vertex_parent = parent.ndof, list(range(parent.ndof//6))
    else:
        if type(banks) is not GenesisVolumeBanks:
            raise ValueError("Bank transfer requires a prepared prospective volume support")
        validate_volume_banks(banks, parent_assembly=parent)
        if (banks.basal is None
                or _object_digest(banks.basal.parent_basal) != _object_digest(basal)):
            raise ValueError("Bank transfer basal mass, drag and force differ from accepted context")
        q, v = banks.R@state.displacement_m, banks.R@state.velocity_m_s
        if np.any(banks.J@q) or np.any(banks.J@v):
            raise ValueError("Zero-time transferred banks must remain exactly tied")
        kind = "explicit_computational_edge_split"
        cut_edges = banks.topology.cut_edges.tolist()
        bank_pin, ndof = banks.binding_sha256, banks.ndof
        vertex_parent = banks.topology.parent_vertex.tolist()
    active = np.flatnonzero(state.fault_history.fault_active).tolist()
    runtime = checkpoint["runtime_state"]
    # Preserve all literal ledgers and closures, rather than inventing a new
    # energy offset that conceals creation or relaxation work.
    excluded = {"geometry", "thermal", "history", "mechanics", "fault_history",
                "displacement_m", "velocity_m_s", "last_report", "last_certificate",
                "plane_normal_world", "slip_direction_world"}
    body = dict(
        format=FORMAT, scope=SCOPE,
        parent_checkpoint_sha256=checkpoint["checkpoint_sha256"],
        formation_checkpoint_sha256=checkpoint["source_checkpoint_sha256"],
        source_identity=parent.source_identity,
        time_myr=state.time_myr,
        accepted_step_sizes_myr=list(state.accepted_step_sizes_myr),
        accepted_steps=state.mechanics.accepted_steps,
        cohort_ids=[int(x) for x in parent.cohort_ids],
        material_geometry_sha256=_digest(runtime["geometry"]),
        birth_history_sha256=_digest(runtime["history"]),
        full_cohort_memory_sha256=_digest(runtime["mechanics"]),
        fault_history_sha256=_digest(runtime["fault_history"]),
        thermal_history_sha256=_digest(runtime["thermal"]),
        inherited_runtime_ledgers_and_closures={k: runtime[k] for k in runtime if k not in excluded},
        support_kind=kind, supplied_computational_cut_edges=cut_edges,
        support_binding_sha256=bank_pin, ndof=ndof, parent_vertex=vertex_parent,
        field_sha256={"displacement_m": _array_digest(q), "velocity_m_s": _array_digest(v),
                      "elastic_constant": _array_digest(e0), "elastic_inverse_scale": _array_digest(e1)},
        active_weak_plane_faces=active,
        physical_cut_status="no_eligible_cut" if not active else "unresolved_physical_cut",
        unresolved_physical_inputs=list(UNRESOLVED_PHYSICAL_INPUTS),
        sharp_physical_banks_created=False, physical_release_admitted=False,
        spatial_underthrust_admitted=False, physical_creation_work_assigned=False,
        endpoint_force_is_release_traction=False,
        transition_time_increment_myr=0., transition_work_j=0., transition_impulse_kg_m_s=0.,
        memory_update_increment_m=0.,
        history_reinitialized=False, connected_material_owner_ids_changed=False,
    )
    payload = {**body, "transfer_sha256": _digest(body)}
    return GenesisVolumeBankTransfer(_canonical(payload).decode(), q, v, e0, e1)


def volume_bank_transfer_to_dict(transfer):
    """Return the literal descriptor after checking its immutable field binding."""
    if type(transfer) is not GenesisVolumeBankTransfer:
        raise ValueError("Expected a context-bound volume bank transfer")
    raw = json.loads(transfer.payload_json)
    body = {k: v for k, v in raw.items() if k != "transfer_sha256"}
    if (raw.get("format") != FORMAT or raw.get("scope") != SCOPE
            or raw.get("transfer_sha256") != _digest(body)):
        raise ValueError("Bank transfer descriptor seal or semantics mismatch")
    actual = {name: _array_digest(getattr(transfer, name)) for name in
              ("displacement_m", "velocity_m_s", "elastic_constant", "elastic_inverse_scale")}
    if raw["field_sha256"] != actual:
        raise ValueError("Bank transfer fields differ from their accepted memory binding")
    return raw


def validate_volume_bank_transfer_payload(raw, *, expected_transfer):
    """Validate an untrusted descriptor against a freshly bound trusted context.

    Re-sealing edited claims cannot authorize a cut or replace current memory.
    The expected transfer comes from ``bind_volume_bank_transfer`` after the
    archive loader has replayed the full accepted parent, not from this DTO.
    """
    expected = volume_bank_transfer_to_dict(expected_transfer)
    if type(raw) is not dict or _canonical(raw) != _canonical(expected):
        raise ValueError("Bank transfer differs from the accepted parent/support context")
    return expected
