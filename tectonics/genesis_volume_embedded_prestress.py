"""Reference-work prestress on an explicit embedded material surface.

Saved Genesis small-strain stress is conjugate to reference volume.  Its
material boundary traction is sigma*n_reference per reference area.  On the
fixed current area at birth the equivalent density includes the reference /
birth area ratio.  This pullback preserves the discrete virtual work; it is
not a finite-stress pushforward or replacement of a partitioned bulk reaction.
No bulk domain, DOF, Genesis history, energy ledger or physical tie is changed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json

import numpy as np

from .genesis_volume_checkpoint import _canonical
from .genesis_volume_creation_law import ProspectiveCreationLaw
from .genesis_volume_embedded_surface import (
    _numeric, integrate_embedded_surface_traction, prepare_volume_embedded_surface,
    validate_volume_embedded_surface, volume_embedded_surface_to_dict,
)
from .genesis_volume_material_stress import (
    bind_volume_material_stress, evaluate_volume_material_stress,
    volume_material_stress_observer_to_dict, volume_material_stress_to_dict,
)
from .genesis_volume_prestress_creation import prepare_prestress_creation
from .genesis_volume_slip_checkpoint import volume_slip_checkpoint_to_dict
from .genesis_volume_slip_dynamics import VolumeSlipModel, VolumeSlipState


FORMAT = "genesis-source-bound-embedded-reference-work-prestress-0.1"
SCOPE = "explicit_material_boundary_reference_work_not_partitioned_bulk_reaction_or_physical_release"


class EmbeddedPrestressRefusal(ValueError):
    """A source, material support or work-conjugate pullback differs."""


def _owned(value):
    a = np.asarray(value, float)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _birth_basis(normal):
    # An isotropic tangent law needs no physical preferred tangent.  This
    # coordinate convention chooses the least parallel Cartesian axis.
    axis = np.eye(3)[np.argmin(np.abs(normal), axis=1)]
    tangent = axis-normal*np.sum(axis*normal, axis=1)[:, None]
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    second = np.cross(normal, tangent)
    return np.stack((normal, tangent, second), axis=2)


@dataclass(frozen=True, slots=True)
class EmbeddedPrestressPullback:
    surface_sha256: str
    stress_world_pa: np.ndarray
    reference_traction_world_pa: np.ndarray
    birth_area_ratio: np.ndarray
    birth_traction_world_pa: np.ndarray
    birth_basis_xyz: np.ndarray
    birth_traction_local_pa: np.ndarray
    reference_traction_local_pa: np.ndarray
    paired_parent_force_n: np.ndarray
    seed: object | None
    full_separation_cost_on_explicit_surface_j: float | None
    binding_sha256: str

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, np.ndarray):
                object.__setattr__(self, f.name, _owned(v))


def _pullback_body(value):
    return dict(
        surface_sha256=value.surface_sha256,
        **{f.name: _plain(getattr(value, f.name)) for f in fields(value)
           if f.name not in ("surface_sha256", "seed", "binding_sha256")},
        seed=None if value.seed is None else _plain(asdict(value.seed)),
        traction_measure="fixed_inherited_small_strain_sigma_times_reference_normal_per_reference_area",
        equivalent_birth_measure="reference_traction_times_reference_area_over_fixed_birth_area",
        creation_area_measure="original_reference_material_area_Gc_counts_both_banks_once",
        seed_traction_measure="reference_area_traction_in_fixed_birth_normal_tangent_basis",
        work_scope="algebraic_duality_under_the_same_declared_quadrature",
        bulk_reaction_replaced=False, finite_stress_pushforward_claimed=False,
        physical_surface_selected=False, physical_creation_admitted=False,
        physical_release_admitted=False, funding_admitted=False,
        creation_work_debited_from_Genesis_j=0.)


def pull_back_embedded_prestress(surface, stress_world_pa, *, creation_law=None):
    """Pure computational traction pullback, with no source admission.

The returned force is a paired virtual boundary force in original modes.
It is not the constraint reaction of a cut/subdivided bulk domain.  Supplying
Gc only prices full separation of this explicitly supplied surface; that cost
is neither available work nor a debit from the planet's stored energy.
"""
    validate_volume_embedded_surface(surface)
    try:
        stress = _numeric(stress_world_pa, "Embedded material stress", (surface.point_count, 3, 3))
    except ValueError as error:
        raise EmbeddedPrestressRefusal(str(error)) from error
    scale = np.maximum(np.max(np.abs(stress), axis=(1, 2)), 1.)
    if np.any(np.max(np.abs(stress-stress.swapaxes(-1, -2)), axis=(1, 2))
              > 64*np.finfo(float).eps*scale):
        raise EmbeddedPrestressRefusal("Embedded material stress must be symmetric")
    if creation_law is not None and type(creation_law) is not ProspectiveCreationLaw:
        raise EmbeddedPrestressRefusal("An explicit prospective creation law is required")
    if np.any(surface.current_area_weights_m2 <= 0) or np.any(surface.reference_area_weights_m2 <= 0):
        raise EmbeddedPrestressRefusal("Embedded area measures must be positive")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        reference = np.einsum("pij,pj->pi", stress, surface.reference_normal_xyz)
        ratio = surface.reference_area_weights_m2/surface.current_area_weights_m2
        birth = reference*ratio[:, None]
        basis = _birth_basis(surface.current_normal_xyz)
        local = np.einsum("pji,pj->pi", basis, birth)
        reference_local = np.einsum("pji,pj->pi", basis, reference)
    if (np.any(ratio <= 0) or
            not all(np.isfinite(v).all() for v in (reference, ratio, birth, basis, local, reference_local))):
        raise EmbeddedPrestressRefusal("Embedded prestress pullback is not numerically finite")
    force = integrate_embedded_surface_traction(surface, reference, area_convention="reference")
    # The inherited cohesive kernel's Gc is per ORIGINAL material area.  Its
    # strength, stiffness, traction and energy all use that same reference
    # measure; changing only Gc to a birth-area density would change the law.
    seed = None if creation_law is None else prepare_prestress_creation(creation_law, reference_local)
    cost = None if seed is None else seed.law.fracture_energy_j_m2*surface.total_reference_area_m2
    if cost is not None and not np.isfinite(cost):
        raise EmbeddedPrestressRefusal("Explicit surface separation cost overflowed")
    value = EmbeddedPrestressPullback(volume_embedded_surface_to_dict(surface)["embedded_surface_sha256"],
        stress, reference, ratio, birth, basis, local, reference_local, force, seed, cost, "")
    return EmbeddedPrestressPullback(**{f.name: getattr(value, f.name) for f in fields(value)
        if f.name != "binding_sha256"}, binding_sha256=_digest(_pullback_body(value)))


def embedded_prestress_pullback_to_dict(value):
    if type(value) is not EmbeddedPrestressPullback:
        raise EmbeddedPrestressRefusal("An immutable embedded prestress pullback is required")
    body = _pullback_body(value)
    if value.binding_sha256 != _digest(body):
        raise EmbeddedPrestressRefusal("Embedded prestress pullback binding differs")
    return {**body, "pullback_sha256": value.binding_sha256}


@dataclass(frozen=True, slots=True)
class BoundEmbeddedPrestress:
    payload_json: str
    binding_sha256: str


def prepare_volume_embedded_prestress(model, state, surface, *, observer=None, creation_law=None):
    """Recompute stress and geometry against a complete accepted runtime.

An untrusted planet must first pass the public checkpoint replay.  Explicit
world-plane inputs are computational probes.  The source's diffuse activation
or a caller's law cannot turn them into a funded sharp material fracture.
"""
    if type(model) is not VolumeSlipModel or type(state) is not VolumeSlipState:
        raise EmbeddedPrestressRefusal("Embedded prestress requires accepted Genesis model/state")
    validate_volume_embedded_surface(surface)
    sections = surface.material_sections
    if not surface.source_geometry_bound or sections is None:
        raise EmbeddedPrestressRefusal("Embedded prestress requires original source material geometry")
    # Preserve the literal declared coefficients. Re-normalizing the already
    # normalized section plane can move its offset by another ULP and breaks
    # exact source reconstruction despite geometrically equivalent inputs.
    rebuilt = prepare_volume_embedded_surface(state.geometry, state.displacement_m,
        surface.declared_normal_xyz, surface.declared_offset_m, face_indices=sections.face_indices,
        cohort_ids=surface.selected_cohort_ids, quadrature_order=surface.quadrature_order)
    if _canonical(volume_embedded_surface_to_dict(surface)) != _canonical(volume_embedded_surface_to_dict(rebuilt)):
        raise EmbeddedPrestressRefusal("Embedded surface differs from accepted current material support")
    expected_observer = bind_volume_material_stress(model, state)
    if observer is not None and _canonical(volume_material_stress_observer_to_dict(observer)) != _canonical(
            volume_material_stress_observer_to_dict(expected_observer)):
        raise EmbeddedPrestressRefusal("Embedded prestress observer differs from accepted source recomputation")
    observer = expected_observer
    samples = evaluate_volume_material_stress(observer, surface.cohort_ids, surface.barycentric,
        surface.mass_coordinate_q)
    for actual, expected, name in ((samples.current_position_m, surface.current_position_m, "current"),
                                  (samples.reference_position_m, surface.reference_position_m, "reference")):
        tolerance = 128*np.finfo(float).eps*np.maximum(np.abs(expected), 1.)
        if np.any(np.abs(actual-expected) > tolerance):
            raise EmbeddedPrestressRefusal(f"Embedded prestress {name} material position differs")
    pullback = pull_back_embedded_prestress(surface, samples.stress_world_tensor, creation_law=creation_law)
    checkpoint = volume_slip_checkpoint_to_dict(model, state)
    body = dict(format=FORMAT, scope=SCOPE,
        parent_checkpoint_sha256=checkpoint["checkpoint_sha256"],
        formation_checkpoint_sha256=checkpoint["source_checkpoint_sha256"],
        source_identity=state.geometry.source_identity, time_myr=state.time_myr,
        accepted_steps=state.mechanics.accepted_steps,
        accepted_step_sizes_myr=list(state.accepted_step_sizes_myr),
        observer_binding_sha256=observer.binding_sha256,
        surface_sha256=volume_embedded_surface_to_dict(surface)["embedded_surface_sha256"],
        source_stress_samples=volume_material_stress_to_dict(samples),
        pullback=embedded_prestress_pullback_to_dict(pullback),
        parameter_provenance=None if creation_law is None else "explicit_new_constitutive_input_not_historical_toughness",
        inherited_runtime_ledgers={k: v for k, v in checkpoint["runtime_state"].items() if k.endswith("_j")},
        source_historical_surface_creation_energy_law=None, selected_sharp_material_surface=None,
        partitioned_bulk_domains_prepared=False, dynamic_bank_reaction_replaced=False,
        physical_creation_admitted=False, physical_release_admitted=False,
        physical_fault_contact_admission_claimed=False, funding_admitted=False,
        history_reinitialized=False, material_memory_changed=False,
        activation_accepted=False, clock_advanced=False, creation_work_debited_from_Genesis_j=0.)
    raw = {**body, "embedded_prestress_sha256": _digest(body)}
    payload = _canonical(raw).decode()
    return BoundEmbeddedPrestress(payload, hashlib.sha256(payload.encode()).hexdigest())


def volume_embedded_prestress_to_dict(value):
    if type(value) is not BoundEmbeddedPrestress or value.binding_sha256 != hashlib.sha256(
            value.payload_json.encode()).hexdigest():
        raise EmbeddedPrestressRefusal("Embedded prestress lost its immutable source binding")
    raw = json.loads(value.payload_json)
    if (raw.get("format") != FORMAT or raw.get("scope") != SCOPE or
            raw.get("embedded_prestress_sha256") != _digest({k: v for k, v in raw.items()
                if k != "embedded_prestress_sha256"})):
        raise EmbeddedPrestressRefusal("Embedded prestress scope or seal differs")
    for key in ("partitioned_bulk_domains_prepared", "dynamic_bank_reaction_replaced",
                "physical_creation_admitted", "physical_release_admitted",
                "physical_fault_contact_admission_claimed", "funding_admitted",
                "history_reinitialized", "material_memory_changed", "activation_accepted", "clock_advanced"):
        if raw.get(key) is not False:
            raise EmbeddedPrestressRefusal("A reference-work boundary probe cannot admit physical fracture or funding")
    if (raw.get("source_historical_surface_creation_energy_law") is not None or
            raw.get("selected_sharp_material_surface") is not None or
            type(raw.get("creation_work_debited_from_Genesis_j")) is not float or
            raw["creation_work_debited_from_Genesis_j"] != 0.):
        raise EmbeddedPrestressRefusal("Embedded prestress cannot invent history, a surface or funding")
    return raw


def volume_embedded_prestress_from_dict(raw, *, model, state, surface, observer=None, creation_law=None):
    expected = prepare_volume_embedded_prestress(model, state, surface,
        observer=observer, creation_law=creation_law)
    try:
        equal = type(raw) is dict and _canonical(raw) == _canonical(volume_embedded_prestress_to_dict(expected))
    except (TypeError, ValueError, OverflowError):
        equal = False
    if not equal:
        raise EmbeddedPrestressRefusal("Embedded prestress differs from complete source/metric/stress recomputation")
    return expected
