"""Prospective creation-work kernels bound to an accepted Genesis context.

An explicit new cohesive law is not recovered historical toughness. The
computational histories below test material surfaces and work accounting;
they do not insert a zero-traction interface into the prestressed planet.
Actual surface selection, prestress replacement and dynamic release require
their own physical admission. The original Genesis state is never updated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from .genesis_volume_banks import _hash_value, validate_volume_banks
from .genesis_volume_checkpoint import _canonical
from .genesis_volume_creation_law import ProspectiveCreationLaw, evaluate_creation_law
from .genesis_volume_fracture_surface import (
    evaluate_surface_jump, integrate_surface_traction, validate_volume_fracture_surface)
from .genesis_volume_operator import _immutable
from .genesis_volume_slip_checkpoint import volume_slip_checkpoint_to_dict
from .genesis_volume_slip_dynamics import VolumeSlipModel, VolumeSlipState

CONTEXT_FORMAT = "genesis-volume-creation-context-0.1"
STATE_FORMAT = "genesis-computational-cohesive-surface-history-0.1"
SCOPE = "prospective_material_surface_work_kernel_not_physical_Genesis_release"


def _seal(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _binding(value):
    digest = hashlib.sha256()
    _hash_value(digest, value)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class GenesisVolumeCreationContext:
    parent_assembly: object
    parent_basal: object
    law: ProspectiveCreationLaw | None
    payload_json: str
    binding_sha256: str


def bind_volume_creation_context(model, state, law=None):
    """Bind a new-law declaration to the complete accepted runtime.

    This is a trusted-context entry point. Untrusted source archives first
    undergo full deterministic checkpoint replay. No material is reborn here.
    """
    if type(model) is not VolumeSlipModel or type(state) is not VolumeSlipState:
        raise ValueError("Creation context requires genuine accepted Genesis slip state")
    if law is not None:
        if type(law) is not ProspectiveCreationLaw:
            raise ValueError("Creation requires an explicitly declared prospective law")
        # Recheck explicit parameters even if a caller bypassed frozen fields.
        ProspectiveCreationLaw(**asdict(law))
    packet = volume_slip_checkpoint_to_dict(model, state)
    parent, basal = model._assembly(state.geometry, state.fault_history.damage)
    rt = packet["runtime_state"]
    top_ledgers = {k: v for k, v in rt.items() if k.endswith("_j")}
    mechanics_ledgers = {k: v for k, v in rt["mechanics"]["state"].items() if k.endswith("_j")}
    fault_ledgers = {k: v for k, v in rt["fault_history"].items() if k.endswith("_j")}
    active = np.flatnonzero(state.fault_history.fault_active).tolist()
    body = dict(format=CONTEXT_FORMAT, scope=SCOPE,
        formation_checkpoint_sha256=packet["source_checkpoint_sha256"],
        parent_checkpoint_sha256=packet["checkpoint_sha256"],
        source_identity=parent.source_identity, time_myr=state.time_myr,
        accepted_step_sizes_myr=list(state.accepted_step_sizes_myr),
        cohort_ids=[int(x) for x in parent.cohort_ids],
        geometry_sha256=_seal(rt["geometry"]), history_sha256=_seal(rt["history"]),
        mechanics_sha256=_seal(rt["mechanics"]), fault_sha256=_seal(rt["fault_history"]),
        thermal_sha256=_seal(rt["thermal"]),
        inherited_genesis_ledgers=dict(runtime=top_ledgers, mechanics=mechanics_ledgers, fault=fault_ledgers),
        declared_prospective_creation_law=None if law is None else asdict(law),
        parameter_provenance=None if law is None else "explicit_new_constitutive_input_not_historical_toughness",
        source_calculated_creation_law=None,
        active_weak_plane_faces=active,
        physical_cut_status=("no_eligible_cut" if not active else
            "missing_prospective_creation_law" if law is None else
            "unresolved_material_surface_selection_and_prestress_release"),
        physical_creation_admitted=False, physical_release_admitted=False,
        creation_work_debited_from_Genesis_j=0.,
        historical_dissipation_available_for_creation_j=0.,
        existing_damage_or_slip_work_reclassified=False,
        material_history_reinitialized=False,
        interface_area_convention="paired_Gamma_once_Gc_includes_both_banks",
    )
    payload = {**body, "context_sha256": _seal(body)}
    value = GenesisVolumeCreationContext(parent, basal, law, _canonical(payload).decode(), "")
    from dataclasses import replace
    return replace(value, binding_sha256=_binding(value))


def creation_context_to_dict(context):
    if (type(context) is not GenesisVolumeCreationContext
            or context.binding_sha256 != _binding(context)):
        raise ValueError("Creation context differs from its accepted source binding")
    raw = json.loads(context.payload_json)
    if raw.get("context_sha256") != _seal({k: v for k, v in raw.items() if k != "context_sha256"}):
        raise ValueError("Creation context seal differs")
    return raw


def _surface_context(context, surface):
    creation_context_to_dict(context)
    validate_volume_fracture_surface(surface)
    validate_volume_banks(surface.banks, parent_assembly=context.parent_assembly)
    if (surface.banks.basal is None
            or _binding(surface.banks.basal.parent_basal) != _binding(context.parent_basal)):
        raise ValueError("Creation surface uses a different accepted basal assembly")
    if context.law is None:
        raise ValueError("No explicit prospective creation-energy law has been declared")


@dataclass(frozen=True, slots=True)
class ComputationalCreationState:
    context_binding_sha256: str
    surface_binding_sha256: str
    bank_displacement_from_fixture_origin_m: np.ndarray
    local_gap_m: np.ndarray
    maximum_separation_m: np.ndarray
    stored_interface_energy_j: float
    cumulative_fracture_work_j: float
    cumulative_cohesive_work_j: float
    accepted_path_segments: tuple
    binding_sha256: str

    def __post_init__(self):
        for name in ("bank_displacement_from_fixture_origin_m", "local_gap_m", "maximum_separation_m"):
            object.__setattr__(self, name, _immutable(getattr(self, name)))
        object.__setattr__(self, "accepted_path_segments",
                           tuple(tuple(float(x) for x in row) for row in self.accepted_path_segments))


def _bound_state(**kwargs):
    from dataclasses import replace
    state = ComputationalCreationState(**kwargs, binding_sha256="")
    return replace(state, binding_sha256=_binding(state))


def make_computational_creation_state(context, surface):
    """Create a zero-gap *fixture* reference, never an actual prestress insertion."""
    _surface_context(context, surface)
    shape = surface.area_weights_m2.shape
    return _bound_state(context_binding_sha256=context.binding_sha256,
        surface_binding_sha256=surface.binding_sha256,
        bank_displacement_from_fixture_origin_m=np.zeros(surface.banks.ndof),
        local_gap_m=np.zeros(shape+(3,)), maximum_separation_m=np.zeros(shape),
        stored_interface_energy_j=0., cumulative_fracture_work_j=0.,
        cumulative_cohesive_work_j=0., accepted_path_segments=())


@dataclass(frozen=True, slots=True)
class ComputationalCreationResponse:
    local_response: object
    world_traction_pa: np.ndarray
    bank_force_n: np.ndarray
    integrated_stored_energy_j: float
    integrated_total_fracture_dissipation_j: float
    integrated_fracture_increment_j: float
    integrated_mechanical_work_j: float

    def __post_init__(self):
        for name in ("world_traction_pa", "bank_force_n"):
            object.__setattr__(self, name, _immutable(getattr(self, name)))


def step_computational_creation_state(context, surface, previous,
                                      bank_displacement_from_fixture_origin_m):
    """Evaluate one straight gap segment and preserve separate cohesive ledgers.

    The displacement is relative to the computational fixture origin. It is
    not the inherited absolute planetary q and never updates bulk strain.
    Mechanical work supplies stored interface energy plus irreversible
    creation dissipation; historical bulk losses contribute no credit.
    """
    _surface_context(context, surface)
    if (type(previous) is not ComputationalCreationState
            or previous.binding_sha256 != _binding(previous)
            or previous.context_binding_sha256 != context.binding_sha256
            or previous.surface_binding_sha256 != surface.binding_sha256):
        raise ValueError("Creation history is not bound to this source/surface")
    u = np.asarray(bank_displacement_from_fixture_origin_m)
    mixed_bool = (isinstance(bank_displacement_from_fixture_origin_m, (list, tuple))
                  and any(isinstance(x, (bool, np.bool_)) for x in bank_displacement_from_fixture_origin_m))
    if mixed_bool or u.dtype.kind not in "fiu" or u.shape != (surface.banks.ndof,) or not np.isfinite(u).all():
        raise ValueError("Creation path requires a complete finite real metre vector")
    u = np.asarray(u, float)
    world_gap = evaluate_surface_jump(surface, u)
    edges = surface.patch_edge_indices
    frame = np.stack((surface.edge_normals_xyz[edges], surface.edge_tangent1_xyz[edges],
                      surface.edge_tangent2_xyz[edges]), axis=-1)
    local_gap = np.einsum("pji,pqj->pqi", frame, world_gap)
    law = evaluate_creation_law(context.law, previous.local_gap_m, local_gap,
                               previous.maximum_separation_m)
    traction = np.einsum("pij,pqj->pqi", frame, law.traction_pa)
    force = integrate_surface_traction(surface, traction)
    weights = surface.area_weights_m2
    energy = float(np.sum(weights*law.stored_energy_j_m2))
    dissipation = float(np.sum(weights*law.fracture_dissipation_j_m2))
    increment = float(np.sum(weights*law.fracture_dissipation_increment_j_m2))
    work = float(np.sum(weights*law.mechanical_work_j_m2))
    if not np.isfinite([energy, dissipation, increment, work]).all():
        raise ValueError("Integrated computational cohesive work must remain finite")
    total_work = previous.cumulative_cohesive_work_j+work
    scale = max(1., abs(energy), abs(dissipation), abs(total_work))
    if abs(total_work-energy-dissipation) > 4e-12*scale:
        raise ValueError("Computational cohesive work/creation balance changed")
    if abs(previous.cumulative_fracture_work_j+increment-dissipation) > 4e-12*scale:
        raise ValueError("Computational creation ledger was debited twice")
    state = _bound_state(context_binding_sha256=context.binding_sha256,
        surface_binding_sha256=surface.binding_sha256,
        bank_displacement_from_fixture_origin_m=u, local_gap_m=local_gap,
        maximum_separation_m=law.maximum_separation_m,
        stored_interface_energy_j=energy, cumulative_fracture_work_j=dissipation,
        cumulative_cohesive_work_j=total_work,
        accepted_path_segments=previous.accepted_path_segments+(tuple(u),))
    response = ComputationalCreationResponse(law, traction, force, energy, dissipation, increment, work)
    return state, response


def creation_fixture_state_to_dict(state):
    if type(state) is not ComputationalCreationState or state.binding_sha256 != _binding(state):
        raise ValueError("Computational creation state differs from its prepared binding")
    body = dict(format=STATE_FORMAT, scope=SCOPE,
        context_binding_sha256=state.context_binding_sha256,
        surface_binding_sha256=state.surface_binding_sha256,
        bank_displacement_from_fixture_origin_m=state.bank_displacement_from_fixture_origin_m.tolist(),
        local_gap_m=state.local_gap_m.tolist(), maximum_separation_m=state.maximum_separation_m.tolist(),
        stored_interface_energy_j=state.stored_interface_energy_j,
        cumulative_fracture_work_j=state.cumulative_fracture_work_j,
        cumulative_cohesive_work_j=state.cumulative_cohesive_work_j,
        accepted_path_segments=[list(row) for row in state.accepted_path_segments],
        physical_creation_admitted=False, physical_release_admitted=False)
    return {**body, "state_sha256": _seal(body)}


def creation_fixture_state_from_dict(raw, *, context, surface):
    """Replay all computational path segments against a trusted source context.

    This is a computational cohesive history, not an actual Genesis bank save.
    A new seal on edited fields cannot supply the missing fracture history.
    """
    if type(raw) is not dict or type(raw.get("accepted_path_segments")) is not list:
        raise ValueError("Computational creation save schema mismatch")
    if raw.get("state_sha256") != _seal({k: v for k, v in raw.items() if k != "state_sha256"}):
        raise ValueError("Computational creation save seal mismatch")
    state = make_computational_creation_state(context, surface)
    for segment in raw["accepted_path_segments"]:
        if (type(segment) is not list
                or any(type(x) not in (int, float) for x in segment)):
            raise ValueError("Computational creation path schema mismatch")
        state, _ = step_computational_creation_state(context, surface, state, segment)
    if _canonical(creation_fixture_state_to_dict(state)) != _canonical(raw):
        raise ValueError("Computational creation save differs from full path replay")
    return state
