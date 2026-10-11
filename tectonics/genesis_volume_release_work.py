"""Source-bound work and momentum checks on a private, fixed-material path.

The accepted Genesis checkpoint supplies every old displacement, velocity,
cohort and elastic memory.  An explicit computational seed and prospective
cohesive law add a straight-path diagnostic only.  The complete compact-jump
kernel stays tied; no result of this module creates or funds a physical fault.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
import hashlib
import json
import math

import numpy as np
from scipy import sparse

from .genesis_volume_checkpoint import _canonical
from .genesis_volume_creation_law import ProspectiveCreationLaw, _effective_gap, _immutable, _numeric
from .genesis_volume_embedded_prestress import pull_back_embedded_prestress
from .genesis_volume_enriched_memory import (
    EnrichedMemoryState, _inner_parts, _cross_moment, accept_enriched_loading,
    enriched_memory_to_dict, evaluate_enriched_loading, prepare_enriched_loading,
    validate_enriched_memory,
)
from .genesis_volume_finite_seed import validate_volume_finite_seed, volume_finite_seed_to_dict
from .genesis_volume_fracture_onset import (
    CalculatedVolumeFractureOnset, calculated_volume_fracture_onset_to_dict,
)
from .genesis_volume_material_stress import _pointwise_fields
from .genesis_volume_nucleation import volume_nucleation_proposal_to_dict
from .genesis_volume_operator import evaluate_genesis_volume_operator
from .genesis_volume_operator import ANGULAR_BARYCENTRIC
from .genesis_volume_loading import average_volume_loading_stiffness_factors
from .genesis_volume_release_history import (
    _rebuild_native_release_state, _advance_native_release_state,
    expand_native_release_increment, native_release_state_to_dict,
)
from .genesis_volume_release_quadrature import (
    SCOPE as QUADRATURE_SCOPE, prepare_restricted_source_quadrature,
    restricted_source_quadrature_to_dict,
)
from .genesis_volume_release_cohesive import (
    prepare_volume_release_cohesive, evaluate_volume_release_cohesive,
    volume_release_cohesive_to_dict, volume_release_cohesive_response_to_dict,
    VolumeReleaseCohesiveRefusal,
)
from .genesis_volume_release_modes import (
    resolve_compact_release_modes, validate_release_increment,
    compact_release_modes_to_dict, _stable_norm as _release_norm,
)
from .genesis_volume_seed_insertion import _surface_enrichment
from .genesis_volume_seed_memory import (
    SourceSeedMemoryBirth, validate_source_seed_memory, source_seed_memory_loading,
    restart_source_seed_memory,
)


FORMAT = "genesis-volume-computational-release-work-0.1"
SCOPE = "source_bound_private_fixed_material_straight_trajectory_no_physical_release"
SECONDS_PER_MYR = 1e6*365.25*24*3600
_EPS = np.finfo(float).eps
_FALSE = ("physical_creation_admitted", "physical_release_admitted", "funding_admitted",
          "accepted_planet_memory_changed", "dynamic_bank_reaction_replaced",
          "source_selected_clipped_bulk_accuracy_certified", "normal_contact_admitted")


class VolumeReleaseWorkRefusal(ValueError):
    """A computational path differs from its authentic, fixed material birth."""


def _plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, (tuple, list)): return [_plain(v) for v in value]
    if isinstance(value, dict): return {k: _plain(v) for k, v in value.items()}
    return value


def _digest(value):
    return hashlib.sha256(_canonical(_plain(value))).hexdigest()


def _typed_pin(value):
    """Guard immutable live inputs without serializing the giant partition DTO."""
    h = hashlib.sha256()
    def visit(v):
        h.update(type(v).__qualname__.encode()+b"\0")
        if isinstance(v, np.ndarray):
            h.update(v.dtype.str.encode()+repr(v.shape).encode()+v.tobytes())
        elif sparse.issparse(v):
            h.update(repr(v.shape).encode())
            for a in (v.data, v.indices, v.indptr): visit(a)
        elif is_dataclass(v):
            for f in fields(v):
                h.update(f.name.encode()+b"\0"); visit(getattr(v, f.name))
        elif isinstance(v, (list, tuple)):
            h.update(str(len(v)).encode())
            for item in v: visit(item)
        elif isinstance(v, dict):
            for key in sorted(v): visit(key); visit(v[key])
        elif isinstance(v, (str, bytes)):
            b = v.encode() if isinstance(v, str) else v
            h.update(str(len(b)).encode()+b"\0"+b)
        elif v is None or isinstance(v, (bool, int, float, np.generic)):
            h.update(repr(v).encode())
        else:
            raise VolumeReleaseWorkRefusal("Unsupported mutable value in source work context")
    visit(value)
    return h.hexdigest()


def _array(value, shape, name):
    a = _numeric(value, name)
    if a.shape != shape:
        raise VolumeReleaseWorkRefusal(f"{name} shape differs from the source")
    return a


def _exact(a, b, name):
    if not np.array_equal(a, b):
        raise VolumeReleaseWorkRefusal(f"{name} differs from source reconstruction")


def _analytic_null(seed):
    section = seed.surface.section_records[0]
    scalar_a = (section.triangle_m+section.displacement_A_m)@section.normal_xyz-section.offset_m
    scalar_p = (section.triangle_m+section.slope_P_m)@section.normal_xyz
    z = np.zeros((18, 3))
    for vertex in range(3):
        for axis in range(3):
            z[6*vertex+axis, axis] = scalar_a[vertex]
            z[6*vertex+3+axis, axis] = scalar_p[vertex]
    lengths = np.linalg.norm(z, axis=0)
    if np.any(lengths == 0.):
        raise VolumeReleaseWorkRefusal("Source analytic compact-jump null witness is unresolved")
    return z/lengths


def _source_surface(birth, seed):
    p, s = birth.partition, seed.surface
    rc = json.loads(birth.reaction.payload_json)
    if (rc["finite_seed_binding_sha256"] != seed.binding_sha256
            or p.selected_cohort_id != seed.selected_cohort_id
            or s.source_checkpoint_sha256 != p.source_checkpoint_sha256
            or s.source_identity != p.source_identity or s.time_myr != p.time_myr
            or not s.source_geometry_bound):
        raise VolumeReleaseWorkRefusal("Finite seed source, material or memory birth differs")
    ci = p.selected_cohort_index
    if np.any(s.cohort_indices != ci):
        raise VolumeReleaseWorkRefusal("Compact surface includes material outside the selected cohort")
    n = s.point_count
    sigma = _pointwise_fields(
        np.broadcast_to(p.memory0[ci], (n, 3, 6)),
        np.broadcast_to(p.memory1[ci], (n, 3, 6)),
        np.broadcast_to(p.parent_assembly.C[ci], (n, 6, 6)),
        np.broadcast_to(p.parent_assembly.basis_xyz[ci], (n, 3, 3)),
        s.barycentric, s.radial_scale)[-1]
    pullback = pull_back_embedded_prestress(s, sigma)
    _, trace = _surface_enrichment(seed)
    return pullback, trace


@dataclass(frozen=True, slots=True)
class VolumeReleaseWorkContext:
    birth: SourceSeedMemoryBirth
    finite_seed: object
    modes: object
    cohesive: object
    operative_cohesive: object
    observable_force_basis: np.ndarray
    observable_force_singular_values_m: np.ndarray
    initial_velocity_m_s: np.ndarray
    archived_constraint_multipliers_n: np.ndarray
    initial_surface_force_n: np.ndarray
    archived_reaction_minus_initial_surface_force_n: np.ndarray
    physical_refusals: tuple[str, ...]
    payload_json: str
    trusted_objects_sha256: str
    binding_sha256: str

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, np.ndarray): object.__setattr__(self, f.name, _immutable(v))

    @property
    def partition(self): return self.birth.partition

    @property
    def physical_release_admitted(self): return False

    @property
    def funding_admitted(self): return False


def _live_objects(c):
    return (c.birth, c.finite_seed, c.modes, c.cohesive, c.operative_cohesive,
        c.observable_force_basis, c.observable_force_singular_values_m, c.initial_velocity_m_s,
        c.archived_constraint_multipliers_n, c.initial_surface_force_n,
        c.archived_reaction_minus_initial_surface_force_n, c.physical_refusals)


def prepare_volume_release_work(birth, finite_seed, law, *, source_nucleation_proposal,
                                source_insertion=None, parent_velocity_m_s=None):
    """Reconstruct reference traction and forcing from the accepted source only.

    The accepted-interval tie reaction and the initial surface force remain
    separate diagnostics with explicitly different scopes.  Their mismatch is
    retained, never used to manufacture an external load or available work.
    """
    validate_source_seed_memory(birth)
    validate_volume_finite_seed(finite_seed)
    if type(law) is not ProspectiveCreationLaw:
        raise VolumeReleaseWorkRefusal("An explicit prospective creation law is required")
    law = ProspectiveCreationLaw(**asdict(law))
    p, reaction = birth.partition, birth.reaction
    if p.M is None or p.D is None or p.external_force_n is None:
        raise VolumeReleaseWorkRefusal("Source mass, drag and external forcing are required")
    rc, bc = json.loads(reaction.payload_json), json.loads(birth.payload_json)
    proposal = volume_nucleation_proposal_to_dict(source_nucleation_proposal)
    for key, expected in (("parent_checkpoint_sha256", bc["accepted_parent_checkpoint_sha256"]),
            ("formation_checkpoint_sha256", bc["formation_checkpoint_sha256"]),
            ("source_identity", p.source_identity), ("time_myr", p.time_myr),
            ("accepted_steps", birth.initial_memory.parent_state.accepted_steps),
            ("accepted_step_sizes_myr", bc["accepted_step_sizes_myr"])):
        if proposal.get(key) != expected:
            raise VolumeReleaseWorkRefusal(f"Nucleation proposal {key} differs from source birth")
    onset_raw = proposal.get("source_onset")
    if type(onset_raw) is not dict:
        raise VolumeReleaseWorkRefusal("Nucleation proposal requires its calculated source onset")
    onset_payload = _canonical(onset_raw).decode()
    onset = calculated_volume_fracture_onset_to_dict(CalculatedVolumeFractureOnset(
        onset_payload, hashlib.sha256(onset_payload.encode()).hexdigest()))
    for key in ("parent_checkpoint_sha256", "formation_checkpoint_sha256", "source_identity",
                "time_myr", "accepted_steps", "accepted_step_sizes_myr"):
        if onset.get(key) != proposal.get(key):
            raise VolumeReleaseWorkRefusal("Nested calculated onset source context differs")
    history = onset.get("accepted_fault_history")
    if type(history) is not dict or _digest(history) != rc["fault_history_sha256"]:
        raise VolumeReleaseWorkRefusal("Nucleation source fault history differs from accepted birth")
    source_packet = json.loads(p.parent_assembly.geometry.source_packet_json)
    parameters = source_packet["source"]["parameters"]["faults"]
    if onset.get("source_fault_parameters") != parameters:
        raise VolumeReleaseWorkRefusal("Calculated onset fault parameters differ from source")
    active = np.asarray(history.get("fault_active"), dtype=bool)
    ready = (~active & (np.asarray(history.get("damage")) >= parameters["activation_damage"])
        & (np.asarray(history.get("fault_candidate_age_myr")) >= parameters["activation_persistence_myr"]))
    if not parameters["enabled"] or not history.get("membrane_established"):
        ready[:] = False
    for key, expected in (("existing_active_faces", np.flatnonzero(active).tolist()),
                          ("newly_activating_faces", np.flatnonzero(ready).tolist())):
        if onset.get(key) != expected or proposal.get(key) != expected:
            raise VolumeReleaseWorkRefusal("Nucleation source face activation flags differ from accepted history")
    if not np.any(ready) and onset.get("prepared_fault_history") != history:
        raise VolumeReleaseWorkRefusal("Inactive calculated onset cannot replace prepared material history")
    pullback, trace = _source_surface(birth, finite_seed)
    insertion_pin = None
    if source_insertion is not None:
        if type(source_insertion) is not dict:
            raise VolumeReleaseWorkRefusal("Source insertion must be an archived, sealed descriptor")
        insertion = source_insertion
        seal_key = "seed_insertion_sha256"
        if insertion.get(seal_key) != _digest({k: v for k, v in insertion.items() if k != seal_key}):
            raise VolumeReleaseWorkRefusal("Source insertion canonical seal differs")
        for key, expected in (("parent_checkpoint_sha256", bc["accepted_parent_checkpoint_sha256"]),
                ("formation_checkpoint_sha256", bc["formation_checkpoint_sha256"]),
                ("source_identity", p.source_identity), ("time_myr", p.time_myr)):
            if insertion.get(key) != expected:
                raise VolumeReleaseWorkRefusal("Source insertion accepted material context differs")
        if insertion.get("finite_seed") != volume_finite_seed_to_dict(finite_seed):
            raise VolumeReleaseWorkRefusal("Source insertion finite support differs")
        _exact(insertion.get("surface_enrichment_trace"), trace, "Source insertion compact trace")
        _exact(insertion.get("prestress_pullback", {}).get("reference_traction_world_pa"),
               pullback.reference_traction_world_pa, "Source insertion prestress")
        insertion_pin = insertion[seal_key]
    modes = resolve_compact_release_modes(trace, finite_seed.surface.reference_area_weights_m2,
        p.binding_sha256, analytic_null_modes=_analytic_null(finite_seed))
    cohesive = prepare_volume_release_cohesive(law, pullback.reference_traction_world_pa,
        finite_seed.surface.reference_area_weights_m2, trace, pullback.birth_basis_xyz)
    weighted = (np.sqrt(finite_seed.surface.reference_area_weights_m2)[:, None, None]*trace).reshape(-1, 18)
    _, singular_values, right = np.linalg.svd(weighted@modes.release_basis, full_matrices=False)
    if (len(singular_values) != modes.rank or not np.isfinite(singular_values).all()
            or np.any(singular_values <= 0.)):
        raise VolumeReleaseWorkRefusal("Unscaled observable force directions are unresolved")
    observable = modes.release_basis@right.T
    operative_trace = np.asarray(np.einsum("pai,ij->paj", np.asarray(trace, np.longdouble),
        np.asarray(observable, np.longdouble)), float)
    operative_cohesive = prepare_volume_release_cohesive(law, pullback.reference_traction_world_pa,
        finite_seed.surface.reference_area_weights_m2, operative_trace, pullback.birth_basis_xyz)
    parent_v = 2*reaction.parent_increment_m/reaction.dt_s-reaction.parent_before_velocity_m_s
    if parent_velocity_m_s is not None:
        _exact(_array(parent_velocity_m_s, (p.parent_ndof,), "Accepted parent velocity"),
               parent_v, "Accepted parent velocity")
    initial_v = np.r_[parent_v, np.zeros(18)]
    surface_force = np.einsum("pai,pa,p->i", trace, pullback.reference_traction_world_pa,
                             finite_seed.surface.reference_area_weights_m2)
    refusals = []
    if not proposal.get("newly_activating_faces"):
        refusals.append("accepted_Genesis_has_no_new_source_activation")
    if proposal.get("declared_prospective_law") is None:
        refusals.append("no_source_selected_prospective_nucleation_law")
    elif proposal["declared_prospective_law"].get("creation_law") != asdict(law):
        refusals.append("computational_creation_law_differs_from_declared_source_proposal_law")
    candidate = proposal.get("candidate")
    if (candidate is None or candidate.get("cohort_id") != finite_seed.selected_cohort_id
            or candidate.get("seed_radius_m") != finite_seed.seed_radius_m
            or not np.array_equal(candidate.get("seed_center_world_m"), finite_seed.seed_center_world_m)):
        refusals.append("computational_seed_not_selected_by_Genesis_source_proposal")
    refusals.extend(("prospective_creation_parameters_are_explicit_uncalibrated_computational_input",
        "source_selected_clipped_bulk_accuracy_not_certified",
        "moving_thermal_material_boundaries_require_mass_heat_and_work_transfer",
        "full_time_normal_contact_and_all_pair_underthrust_not_admitted",
        "archived_interval_reaction_not_replaced_by_initial_surface_traction"))
    body = dict(format=FORMAT, scope=SCOPE,
        source_memory_birth_binding_sha256=birth.binding_sha256,
        partition_binding_sha256=p.binding_sha256, finite_seed_binding_sha256=finite_seed.binding_sha256,
        accepted_parent_checkpoint_sha256=bc["accepted_parent_checkpoint_sha256"],
        formation_checkpoint_sha256=bc["formation_checkpoint_sha256"],
        source_identity=p.source_identity, accepted_time_myr=p.time_myr,
        fault_history_sha256=rc["fault_history_sha256"],
        prospective_creation_law=asdict(law),
        parameter_provenance="explicit_new_computational_input_not_historical_Genesis_toughness",
        source_nucleation_proposal=proposal, source_insertion_sha256=insertion_pin,
        surface_trace_xyz=trace, reference_area_weights_m2=finite_seed.surface.reference_area_weights_m2,
        initial_reference_traction_world_pa=pullback.reference_traction_world_pa,
        birth_basis_xyz=pullback.birth_basis_xyz,
        release_modes=compact_release_modes_to_dict(modes),
        observable_force_basis=observable, observable_force_singular_values_m=singular_values,
        momentum_force_basis_scope="right_singular_directions_of_actual_unscaled_weighted_jump_trace",
        cohesive_chart=volume_release_cohesive_to_dict(cohesive), initial_velocity_m_s=initial_v,
        operative_cohesive_chart=volume_release_cohesive_to_dict(operative_cohesive),
        operative_bulk_field_scope=QUADRATURE_SCOPE, weak_geometry_accuracy_certified=False,
        legacy_witness_differences_available_for_funding=False,
        archived_constraint_multipliers_n=reaction.constraint_multipliers_n,
        initial_surface_force_n=surface_force,
        archived_reaction_minus_initial_surface_force_n=reaction.constraint_multipliers_n-surface_force,
        reaction_difference_scope="saved_accepted_interval_full_tie_reaction_minus_endpoint_surface_reference_force",
        inherited_operator_energy_roundoff_j=birth.initial_memory.birth_operator_energy_roundoff_j,
        arithmetic_difference_is_available_work=False, complete_jump_kernel_remains_tied=True,
        physical_refusals=refusals, historical_dissipation_available_for_creation_j=0.,
        creation_work_debited_from_Genesis_j=0., new_accepted_Genesis_time_myr=0.,
        new_material_birth_count=0, **{k: False for k in _FALSE})
    body = _plain(body)
    raw = {**body, "release_work_context_sha256": _digest(body)}
    payload = _canonical(raw).decode()
    c = VolumeReleaseWorkContext(birth, finite_seed, modes, cohesive, operative_cohesive,
        observable, singular_values, initial_v,
        reaction.constraint_multipliers_n, surface_force, reaction.constraint_multipliers_n-surface_force,
        tuple(refusals), payload, "", hashlib.sha256(payload.encode()).hexdigest())
    return VolumeReleaseWorkContext(**{f.name: getattr(c, f.name) for f in fields(c)
        if f.name != "trusted_objects_sha256"}, trusted_objects_sha256=_typed_pin(_live_objects(c)))


def validate_volume_release_work(context):
    if (type(context) is not VolumeReleaseWorkContext
            or context.binding_sha256 != hashlib.sha256(context.payload_json.encode()).hexdigest()
            or context.trusted_objects_sha256 != _typed_pin(_live_objects(context))):
        raise VolumeReleaseWorkRefusal("Immutable source release-work context changed")
    return context


def volume_release_work_to_dict(context):
    validate_volume_release_work(context)
    return json.loads(context.payload_json)


def volume_release_work_from_dict(raw, birth, finite_seed, law, *, source_nucleation_proposal,
                                 source_insertion=None, parent_velocity_m_s=None):
    expected = prepare_volume_release_work(birth, finite_seed, law,
        source_nucleation_proposal=source_nucleation_proposal, source_insertion=source_insertion,
        parent_velocity_m_s=parent_velocity_m_s)
    if type(raw) is not dict or _canonical(raw) != _canonical(volume_release_work_to_dict(expected)):
        raise VolumeReleaseWorkRefusal("Release-work packet differs from trusted source reconstruction")
    return expected


def _journal_velocity(context, state):
    velocity = np.array(context.initial_velocity_m_s)
    time = context.partition.time_myr
    for step in state.journal:
        dt = (step.after_time_myr-time)*SECONDS_PER_MYR
        if not math.isfinite(dt) or dt <= 0.:
            raise VolumeReleaseWorkRefusal("Private trajectory journal time is invalid")
        d = np.r_[step.parent_increment_m, step.enrichment_increment_m]
        validate_release_increment(context.modes, step.enrichment_increment_m)
        velocity = 2*d/dt-velocity
        time = step.after_time_myr
    if time != state.time_myr:
        raise VolumeReleaseWorkRefusal("Private velocity journal and material clock differ")
    return velocity


def _full18_kernel_witnesses(context, vectors):
    """Measure literal archive leakage without changing a source coefficient.

    Native coordinates have only the resolved W axes.  Their accumulated
    float64 full18 archive can retain addition/Maxwell rounding after large
    cancelling steps; this measurement is never a native admission test.
    """
    result, passed = {}, True
    z = context.modes.tied_null_modes
    for name, raw in vectors.items():
        a = np.asarray(raw, float)
        scale = float(np.max(np.abs(a), initial=0.))
        scaled = a if scale == 0. else a/scale
        norm = _release_norm(scaled)
        leakage = _release_norm(z.T@scaled)
        ok = bool(leakage <= 256*_EPS*norm)
        result[name] = dict(norm_magnitude=scale*norm,
            null_component_norm=scale*leakage,
            relative_null_component=0. if norm == 0. else leakage/norm,
            literal18_kernel_tolerance_passed=ok)
        passed = passed and ok
    return result, passed


def _dot_bound(a, b):
    return 128*_EPS*float(np.sum(np.abs(a*b)))


def _summation_roundoff(magnitude, operation_count, *, epsilon=None):
    """Own absolute operands bound a reduction, including a cancelled result."""
    eps = np.longdouble(np.finfo(np.longdouble).eps if epsilon is None else epsilon)
    factor = np.longdouble(operation_count)*eps
    if factor >= 1. or operation_count < 1:
        raise VolumeReleaseWorkRefusal("Arithmetic reduction count is not resolvable")
    return (factor/(1-factor))*np.asarray(magnitude, np.longdouble)


def _parent_force_operand_magnitude(p, mean0, mean1):
    """Literal original analytic operator operands, before cohort cancellation."""
    a, ld = p.parent_assembly, np.longdouble
    e0, e1 = np.abs(np.asarray(mean0, ld)), np.abs(np.asarray(mean1, ld))
    c = np.abs(np.asarray(a.elasticity_pa, ld))
    im = np.abs(np.asarray(a.inverse_scale_mean, ld))
    variance = np.abs(np.asarray(a.inverse_scale_variance, ld))
    b0, b1 = np.abs(np.asarray(a.B0, ld)), np.abs(np.asarray(a.B1, ld))
    stress = np.einsum("cab,cqb->cqa", c, e0+im[:, None, None]*e1)
    inverse = np.einsum("cab,cqb->cqa", c, e1)
    local = (np.einsum("cqai,cqa->ci", b0+im[:, None, None, None]*b1, stress)
        +variance[:, None]*np.einsum("cqai,cqa->ci", b1, inverse))
    local *= np.asarray(a.volume_m3, ld)[:, None]/3
    magnitude = np.zeros(p.parent_ndof, dtype=ld)
    np.add.at(magnitude, a.dofs.ravel(), local.ravel())
    return magnitude


def _source_operand_magnitudes(s):
    """Direct restricted point operands; no expanded eighteen-mode forces."""
    g, e, b, t, c, weights = (np.abs(np.asarray(v, np.longdouble)) for v in (
        s.G_w, s.E_w, s.parent_strain_operator, s.parent_trace,
        s.elasticity_pa, s.reference_volume_weights_m3))
    return dict(
        stiffness_cross=np.einsum("pai,ab,pbj,p->ij", b, c, g, weights),
        stiffness_enriched=np.einsum("pai,ab,pbj,p->ij", g, c, g, weights),
        mass_cross=np.longdouble(s.density_kg_m3)*np.einsum("pai,paj,p->ij", t, e, weights),
        mass_enriched=np.longdouble(s.density_kg_m3)*np.einsum("pai,paj,p->ij", e, e, weights),
        point_reduction_operation_count=32+36*len(weights))


def _journal_cohesive_maximum(context, state, *, operative=True, native_state=None):
    """The positive-part norm is convex, so every segment maximum is at an end."""
    chart = context.operative_cohesive if operative else context.cohesive
    history = np.array(chart.seed.initial_maximum_separation_m)
    if native_state is not None:
        a = np.zeros(context.modes.rank)
        for step in native_state.journal:
            a += np.asarray(step["increment_w_m"])
            gap = np.einsum("pij,j->pi", chart.compact_jump_trace_local, a)
            _, radius = _effective_gap(chart.seed.reference_offset_m+gap)
            history = np.maximum(history, radius)
        return history
    a = np.zeros(18)
    for step in state.journal:
        a += np.asarray(step.enrichment_increment_m)
        coordinates = np.asarray(np.asarray(context.observable_force_basis, np.longdouble).T
            @np.asarray(a, np.longdouble), float) if operative else a
        gap = np.einsum("pij,j->pi", chart.compact_jump_trace_local, coordinates)
        _, radius = _effective_gap(chart.seed.reference_offset_m+gap)
        history = np.maximum(history, radius)
    return history


def _point_memory(p, e0, e1):
    l, ci = p.local, p.selected_cohort_index
    angular = np.asarray(l.barycentric, np.longdouble)@np.asarray(np.linalg.inv(ANGULAR_BARYCENTRIC), np.longdouble)
    return angular@np.asarray(e0[ci], np.longdouble)+(angular@np.asarray(e1[ci], np.longdouble))/np.asarray(l.radial_scale, np.longdouble)[:, None]


def _stable_bulk_parts(context, quadrature, state, loading, bulk, parent_force, d,
        *, native_coordinates=None):
    """Reduce weak field energies/actions before any eighteen-column Gram."""
    p, s, ld = context.partition, quadrature, np.longdouble
    w = np.asarray(s.basis_w, ld)
    old0 = np.asarray([c.strain0 for c in state.parent_state.cohorts])
    old1 = np.asarray([c.strain1 for c in state.parent_state.cohorts])
    eold, enew, emean = (_point_memory(p, a, b) for a, b in (
        (old0, old1), (bulk.strain0, bulk.strain1), (bulk.mean0, bulk.mean1)))
    if native_coordinates is None:
        zold = w.T@np.asarray(state.coefficients_m, ld)
        zend = w.T@np.asarray(bulk.coefficients_m, ld)
        zmean = w.T@np.asarray(bulk.mean_coefficients_m, ld)
        dw = w.T@np.asarray(d[p.parent_ndof:], ld)
    else:
        zold, zend, zmean, dw = (np.asarray(native_coordinates[k], ld)
            for k in ("zold", "zend", "zmean", "dw"))
    ezo, ezn, ezm, edw = (np.einsum("pai,i->pa", s.G_w, value)
        for value in (zold, zend, zmean, dw))
    dq = np.asarray(d[:p.parent_ndof], ld)
    eq = np.einsum("pai,i->pa", s.parent_strain_operator, dq[p.selected_parent_dofs])
    def inner(a, b):
        return np.einsum("pa,ab,pb,p->", a, s.elasticity_pa, b, s.reference_volume_weights_m3)
    def absolute_inner(a, b):
        return np.einsum("pa,ab,pb,p->", np.abs(a), np.abs(s.elasticity_pa),
            np.abs(b), s.reference_volume_weights_m3)
    before = np.array((.5*_inner_parts(p, old0, old1, state.coefficients_m,
        old0, old1, state.coefficients_m)[0], inner(eold, ezo), .5*inner(ezo, ezo)), ld)
    after = np.array((bulk.energy_decomposition_j[0], inner(enew, ezn), .5*inner(ezn, ezn)), ld)
    work = np.array((bulk.work_decomposition_j[0], inner(eq, ezm)+inner(edw, emean), inner(edw, ezm)), ld)
    x = ld(loading.ratio[p.selected_cohort_index])
    beta = ld(loading.beta[p.selected_cohort_index])
    rr = -np.expm1(-2*x)/(2*x) if x != 0 else ld(1.)
    aa, ab = x*rr, 2*(beta-rr)
    bb = (1-2*beta+rr)/x if x != 0 else ld(0.)
    if x < ld(1e-3):
        ab = x-x*x+7*x**3/12-x**4/4
        bb = x/3-x*x/4+7*x**3/60-x**4/24
    shift = ab/(2*aa) if aa != 0 else ld(0.)
    remainder = bb-aa*shift*shift
    if remainder < 0.:
        raise VolumeReleaseWorkRefusal("Stable Maxwell temporal Gram is not positive")
    e_shift, z_shift = eold+shift*eq, ezo+shift*edw
    loss = np.array((bulk.dissipation_decomposition_j[0],
        2*aa*inner(e_shift, z_shift)+2*remainder*inner(eq, edw),
        aa*inner(z_shift, z_shift)+remainder*inner(edw, edw)), ld)
    delta = np.array((work[0]-loss[0],
        inner(enew-eold, ezn)+inner(eold, ezn-ezo),
        .5*inner(ezn-ezo, ezn+ezo)), ld)
    cross_q = s.stiffness_cross@zmean
    cross_w = np.einsum("pai,ab,pb,p->i", s.G_w, s.elasticity_pa, emean, s.reference_volume_weights_m3)
    self_w = s.stiffness_enriched@zmean
    parent = np.asarray(parent_force, ld).copy()
    np.add.at(parent, p.selected_parent_dofs, cross_q)
    active = np.r_[parent, cross_w+self_w]
    force_work = np.array((np.asarray(parent_force, ld)@dq,
        cross_q@dq[p.selected_parent_dofs]+cross_w@dw, self_w@dw), ld)
    operands = _source_operand_magnitudes(s)
    cross_q_magnitude = operands["stiffness_cross"]@np.abs(zmean)
    cross_w_magnitude = np.einsum("pai,ab,pb,p->i", np.abs(s.G_w),
        np.abs(s.elasticity_pa), np.abs(emean), s.reference_volume_weights_m3)
    self_w_magnitude = operands["stiffness_enriched"]@np.abs(zmean)
    parent_magnitude = _parent_force_operand_magnitude(p, bulk.mean0, bulk.mean1)
    parent_bound = _summation_roundoff(parent_magnitude,
        128+3*len(p.parent_assembly.cohort_ids), epsilon=_EPS)
    point_count = operands["point_reduction_operation_count"]
    cross_q_bound = _summation_roundoff(cross_q_magnitude, point_count+len(zmean))
    cross_w_bound = _summation_roundoff(cross_w_magnitude, point_count)
    self_w_bound = _summation_roundoff(self_w_magnitude, point_count+len(zmean))
    active_bound = np.r_[parent_bound.copy(), cross_w_bound+self_w_bound]
    np.add.at(active_bound, p.selected_parent_dofs, cross_q_bound)
    work_magnitudes = np.array((np.abs(np.asarray(parent_force, ld))@np.abs(dq),
        absolute_inner(eq, ezm)+absolute_inner(edw, emean), absolute_inner(edw, ezm)), ld)
    force_dot_magnitudes = np.array((work_magnitudes[0],
        np.abs(cross_q)@np.abs(dq[p.selected_parent_dofs])+np.abs(cross_w)@np.abs(dw),
        np.abs(self_w)@np.abs(dw)), ld)
    work_bounds = np.r_[_dot_bound(np.asarray(parent_force, ld), dq),
        _summation_roundoff(work_magnitudes[1:], point_count)]
    force_dot_bounds = np.r_[_dot_bound(np.asarray(parent_force, ld), dq),
        _summation_roundoff(force_dot_magnitudes[1:], p.ndof+32)]
    # Coupling errors use these same weak source point operands.  The
    # inherited parent energy never supplies a tolerance for a tiny term.
    force_dot_bounds[0] += parent_bound@np.abs(dq)
    force_dot_bounds[1] += (cross_q_bound@np.abs(dq[p.selected_parent_dofs])
                           +cross_w_bound@np.abs(dw))
    force_dot_bounds[2] += self_w_bound@np.abs(dw)
    full_energy = before[0]+before[1]+before[2]
    if (before[2] < 0. or after[2] < 0. or loss[2] < 0.
            or full_energy < 0. or after.sum() < 0. or loss.sum() < 0.):
        raise VolumeReleaseWorkRefusal("Stable complete field energy or Maxwell dissipation is negative")
    return dict(before=before, after=after, work=work, loss=loss, delta=delta,
        force_work=force_work, active_force=active, cross_q=cross_q, cross_w=cross_w,
        self_w=self_w, zold=zold, zend=zend, zmean=zmean, dw=dw,
        operand_magnitudes=operands, active_force_roundoff_bound=active_bound,
        work_operand_magnitudes=work_magnitudes, force_dot_operand_magnitudes=force_dot_magnitudes,
        work_roundoff_bound=work_bounds, force_dot_roundoff_bound=force_dot_bounds)


def _source_expansion_witnesses(context, quadrature, state, bulk, d, v0, v1,
        *, native_coordinates=None):
    """Expose source field lift/projection errors at their own operand scales."""
    ld, eps = np.longdouble, np.longdouble(_EPS)
    w = np.asarray(context.observable_force_basis, ld)
    result, passed = {}, True
    vectors = dict(old_displacement=state.displacement_m, new_displacement=bulk.displacement_m,
        displacement_increment=d[context.partition.parent_ndof:],
        before_velocity=v0[context.partition.parent_ndof:], after_velocity=v1[context.partition.parent_ndof:],
        old_elastic_memory=state.coefficients_m, new_elastic_memory=bulk.coefficients_m,
        mean_elastic_memory=bulk.mean_coefficients_m)
    field_pairs = dict(displacement=(context.partition.local.enrichment_trace, quadrature.E_w),
        strain=(context.partition.local.enrichment_strain_operator, quadrature.G_w),
        interface_gap=(context.cohesive.compact_jump_trace_world,
                       context.operative_cohesive.compact_jump_trace_world))
    for name, raw in vectors.items():
        a = np.asarray(raw, ld)
        coordinates = (w.T@a if native_coordinates is None
            else np.asarray(native_coordinates[name], ld))
        for field_name, (full, restricted) in field_pairs.items():
            full, restricted = np.asarray(full, ld), np.asarray(restricted, ld)
            literal = np.einsum("pai,i->pa", full, a)
            operative = np.einsum("pai,i->pa", restricted, coordinates)
            # The source coefficients, W and public chart all have float64
            # precision.  Their actual multiplication/lift operands bound the
            # allowed arithmetic discrepancy; no unrelated energy norm enters.
            lift_magnitude = np.abs(w)@(np.abs(w).T@np.abs(a))
            magnitude = (np.einsum("pai,i->pa", np.abs(full), np.abs(a)+lift_magnitude)
                         +np.einsum("pai,i->pa", np.abs(restricted), np.abs(coordinates)))
            bound = 256*eps*magnitude
            error = np.abs(literal-operative)
            ok = bool(np.all(error <= bound))
            passed = passed and ok
            ratios = np.divide(error, bound, out=np.zeros_like(error), where=bound != 0.)
            result[name+"_"+field_name] = dict(
                maximum_absolute_difference=float(np.max(error, initial=0.)),
                maximum_operand_arithmetic_bound=float(np.max(bound, initial=0.)),
                maximum_difference_to_arithmetic_bound_ratio=float(np.max(ratios, initial=0.)),
                passed=ok)
    return result, passed


def _endpoint_addition_witness(context, before_memory, bulk, d, old_a, new_a, dw,
        *, native_state=None):
    """Separate endpoint addition arithmetic from the supplied path increment."""
    ld = np.longdouble
    old_full, new_full = (np.asarray(v, ld) for v in
        (before_memory.displacement_m, bulk.displacement_m))
    full_d = np.asarray(d[context.partition.parent_ndof:], ld)
    full_difference = new_full-old_full-full_d
    full_bound = _summation_roundoff(np.abs(old_full)+np.abs(full_d), 1, epsilon=_EPS)
    if native_state is None:
        w = np.asarray(context.observable_force_basis, ld)
        bound = (np.abs(w).T@full_bound
            +_summation_roundoff(np.abs(w).T@(np.abs(old_full)+np.abs(new_full)+np.abs(full_d)), 18)
            +_summation_roundoff(np.abs(w.T@old_full)+np.abs(w.T@new_full), 1, epsilon=_EPS))
    else:
        bound = _summation_roundoff(np.abs(np.asarray(old_a, ld))+np.abs(np.asarray(dw, ld)), 1, epsilon=_EPS)
    endpoint_increment = np.asarray(new_a, ld)-np.asarray(old_a, ld)
    difference = endpoint_increment-np.asarray(dw, ld)
    bound += _summation_roundoff(np.abs(np.asarray(old_a, ld))+np.abs(np.asarray(new_a, ld))
        +np.abs(np.asarray(dw, ld)), 3)
    passed = bool(np.all(np.abs(difference) <= bound) and np.all(np.abs(full_difference) <= full_bound))
    return dict(endpoint_increment=endpoint_increment, difference=difference, bound=bound,
        full_difference=full_difference, full_bound=full_bound, passed=passed)


@dataclass(frozen=True, slots=True)
class VolumeReleaseWorkResponse:
    bulk_response: object
    cohesive_response: object
    after_memory: EnrichedMemoryState
    payload_json: str
    binding_sha256: str
    trusted_objects_sha256: str
    after_operative_state: object = None

    @property
    def after_native(self): return self.after_operative_state

    @property
    def balanced_trajectory(self): return json.loads(self.payload_json)["balanced_trajectory"]

    @property
    def physical_release_admitted(self): return False

    @property
    def funding_admitted(self): return False


def evaluate_release_work(context, before_memory, old_maximum_separation_m,
        increment_full_m, after_time_myr, *, before_velocity_m_s=None,
        momentum_relative_tolerance=3e-8, work_relative_tolerance=3e-9,
        operative_state=None, operative_increment_m=None):
    """Check the supplied actual path; no work credit or external load is accepted.

    Endpoint velocities satisfy midpoint kinematics.  Source forcing supplies
    only Fext dot increment.  Momentum is checked on every old parent direction
    AND on the complete jump-observable subspace, while kernel forces remain
    tie reactions.  Scalar work alone never certifies a balanced trajectory.
    """
    validate_volume_release_work(context)
    p = context.partition
    validate_enriched_memory(p, before_memory)
    replayed = restart_source_seed_memory(context.birth, enriched_memory_to_dict(before_memory))
    if enriched_memory_to_dict(replayed) != enriched_memory_to_dict(before_memory):
        raise VolumeReleaseWorkRefusal("Private before-memory differs from its genuine source journal")
    if (operative_state is None) != (operative_increment_m is None):
        raise VolumeReleaseWorkRefusal("Native state and its authoritative increment are required together")
    if operative_state is not None:
        operative_state = _rebuild_native_release_state(context,
            native_release_state_to_dict(operative_state), before_memory)
    if operative_state is None:
        for a in (before_memory.displacement_m, before_memory.coefficients_m):
            validate_release_increment(context.modes, a)
    _exact(_array(old_maximum_separation_m, context.cohesive.seed.initial_maximum_separation_m.shape,
                 "Old cohesive history"), _journal_cohesive_maximum(context, before_memory,
                     native_state=operative_state),
           "Old cohesive history")
    d = _array(increment_full_m, (p.ndof,), "Full displacement increment")
    if operative_state is not None:
        native_d = _array(operative_increment_m, (p.parent_ndof+context.modes.rank,), "Native increment")
        _exact(d, expand_native_release_increment(context, native_d), "Source full18 expanded witness increment")
    after = _array(after_time_myr, (), "Private after time")
    dt = (float(after)-before_memory.time_myr)*SECONDS_PER_MYR
    if dt <= 0. or not math.isfinite(dt):
        raise VolumeReleaseWorkRefusal("Private after time must advance the fixed-material branch")
    for tol, name in ((momentum_relative_tolerance, "momentum tolerance"),
                      (work_relative_tolerance, "work tolerance")):
        t = _array(tol, (), name)
        if not 0. < t < 1.:
            raise VolumeReleaseWorkRefusal(f"{name} must lie in (0,1)")
    expected_v = _journal_velocity(context, before_memory)
    v0 = expected_v if before_velocity_m_s is None else _array(before_velocity_m_s, (p.ndof,), "Before velocity")
    _exact(v0, expected_v, "Private before velocity")
    # Each source-expanded step remains checked.  A native cumulative archive
    # is a rounded witness, while its source-born W journal is authoritative.
    validate_release_increment(context.modes, d[p.parent_ndof:],
        v0[p.parent_ndof:] if operative_state is None else None)
    v1 = 2*d/dt-v0
    if operative_state is None:
        validate_release_increment(context.modes, before_memory.displacement_m+d[p.parent_ndof:], v1[p.parent_ndof:])
    loading = source_seed_memory_loading(context.birth, before_memory, float(after))
    trial = prepare_enriched_loading(p, before_memory, loading)
    bulk = evaluate_enriched_loading(p, trial, d[:p.parent_ndof], d[p.parent_ndof:])
    if operative_state is None:
        validate_release_increment(context.modes, bulk.coefficients_m)
    kernel_witnesses, kernel_witness_passed = _full18_kernel_witnesses(context, dict(
        old_displacement=before_memory.displacement_m, new_displacement=bulk.displacement_m,
        displacement_increment=d[p.parent_ndof:], before_velocity=v0[p.parent_ndof:],
        after_velocity=v1[p.parent_ndof:], old_elastic_memory=before_memory.coefficients_m,
        new_elastic_memory=bulk.coefficients_m, mean_elastic_memory=bulk.mean_coefficients_m))
    legacy_cohesive, legacy_cohesive_refusal = None, None
    try:
        legacy_cohesive = evaluate_volume_release_cohesive(context.cohesive,
            before_memory.displacement_m, bulk.displacement_m,
            _journal_cohesive_maximum(context, before_memory, operative=False),
            **({"max_subdivisions": 0} if operative_state is not None else {}))
    except VolumeReleaseCohesiveRefusal as error:
        numerical_refusals = (
            "Cohesive mean traction/work quadrature did not converge component by component",
            "Cohesive assembled mean force/work failed component convergence or exact path duality",
        )
        if operative_state is None or str(error) not in numerical_refusals:
            raise
        legacy_cohesive_refusal = dict(exception_class=type(error).__name__, message=str(error))
    wld = np.asarray(context.observable_force_basis, np.longdouble)
    native_coordinates = None
    if operative_state is None:
        operative_old_a = np.asarray(wld.T@np.asarray(before_memory.displacement_m, np.longdouble), float)
        operative_new_a = np.asarray(wld.T@np.asarray(bulk.displacement_m, np.longdouble), float)
    else:
        dw = native_d[p.parent_ndof:]
        alpha, beta = float(np.exp(-loading.ratio[p.selected_cohort_index])), float(loading.beta[p.selected_cohort_index])
        theta = float(average_volume_loading_stiffness_factors(loading)[p.selected_cohort_index])
        operative_old_a = operative_state.displacement_w_m
        operative_new_a = np.asarray(operative_old_a+dw, float)
        native_coordinates = dict(zold=operative_state.coefficients_w_m,
            zend=alpha*operative_state.coefficients_w_m+beta*dw,
            zmean=beta*operative_state.coefficients_w_m+theta*dw, dw=dw)
    cohesive = evaluate_volume_release_cohesive(context.operative_cohesive,
        operative_old_a, operative_new_a, old_maximum_separation_m)
    inertia = 2*(p.M@(d-dt*v0))/(dt*dt)
    drag = p.D@d/dt
    if legacy_cohesive is not None:
        legacy_gamma_force = legacy_cohesive.mean_generalized_force_n
    else:
        legacy_gamma_force = np.einsum("pai,pa,p->i", context.cohesive.compact_jump_trace_world,
            cohesive.mean_traction_world_pa, context.cohesive.reference_area_weights_m2)
    legacy_endpoint_gamma_force = np.einsum("pai,pa,p->i", context.cohesive.compact_jump_trace_world,
        cohesive.endpoint_traction_world_pa, context.cohesive.reference_area_weights_m2)
    gamma = np.r_[np.zeros(p.parent_ndof), legacy_gamma_force]
    external = p.external_force_n
    forces = (inertia, drag, bulk.mean_generalized_force_n, gamma, -external)
    residual = sum(forces)
    b, z, w = context.modes.release_basis, context.modes.tied_null_modes, context.observable_force_basis
    def project(a): return np.r_[a[:p.parent_ndof], w.T@a[p.parent_ndof:]]
    projected = project(residual)
    projected_scale = sum(np.abs(project(a)) for a in forces)
    projection_roundoff = 256*_EPS*sum(np.r_[np.abs(a[:p.parent_ndof]),
        np.abs(w).T@np.abs(a[p.parent_ndof:])] for a in forces)
    component_bound = float(momentum_relative_tolerance)*projected_scale+projection_roundoff
    momentum_passed = bool(np.all(np.abs(projected) <= component_bound))
    nonzero = projected_scale > 0.
    relative_error = float(np.max(np.divide(np.abs(projected), projected_scale,
        out=np.zeros_like(projected), where=nonzero), initial=0.))
    parent_old_force = evaluate_genesis_volume_operator(p.parent_assembly,
        np.zeros(p.parent_ndof), elastic_constant=bulk.mean0, elastic_inverse_scale=bulk.mean1).force
    q, da = d[:p.parent_ndof], d[p.parent_ndof:]
    cross_parent_force = p.local.stiffness_cross@bulk.mean_coefficients_m
    cross_extra_force = _cross_moment(p, bulk.mean0, bulk.mean1)
    self_extra_force = p.local.stiffness_enriched@bulk.mean_coefficients_m
    force_work_parts = np.array((parent_old_force@q,
        cross_parent_force@q[p.selected_parent_dofs]+cross_extra_force@da, self_extra_force@da))
    work_parts = np.asarray(bulk.work_decomposition_j)
    loss_parts = np.asarray(bulk.dissipation_decomposition_j)
    old0 = np.asarray([c.strain0 for c in before_memory.parent_state.cohorts])
    old1 = np.asarray([c.strain1 for c in before_memory.parent_state.cohorts])
    old_energy_parts = .5*_inner_parts(p, old0, old1, before_memory.coefficients_m,
        old0, old1, before_memory.coefficients_m)
    new_energy_parts = np.asarray(bulk.energy_decomposition_j)
    # The enormous inherited old energy must not erase small signed enriched
    # terms.  Its increment uses the original incremental constitutive ledger;
    # cross/self differences are reduced separately from their own magnitudes.
    energy_increments = np.r_[work_parts[0]-loss_parts[0], new_energy_parts[1:]-old_energy_parts[1:]]
    small_residual = energy_increments[1:]-(work_parts[1:]-loss_parts[1:])
    small_scale = np.abs(new_energy_parts[1:])+np.abs(old_energy_parts[1:])+np.abs(work_parts[1:])+np.abs(loss_parts[1:])
    small_passed = bool(np.all(np.abs(small_residual) <= (float(work_relative_tolerance)+256*_EPS)*small_scale))
    force_bounds = np.array((_dot_bound(parent_old_force, q),
        _dot_bound(cross_parent_force, q[p.selected_parent_dofs])+_dot_bound(cross_extra_force, da),
        _dot_bound(self_extra_force, da)))
    duality_scale = np.abs(force_work_parts)+np.abs(work_parts)
    bulk_duality_passed = bool(np.all(np.abs(force_work_parts-work_parts)
        <= float(work_relative_tolerance)*duality_scale+force_bounds))
    # Frozen expanded matrices and the previous enriched response are retained
    # literally as witnesses.  Operative weak directions use source point
    # fields transformed BEFORE their Gram or force contractions.
    legacy_force_work_parts = force_work_parts.copy()
    legacy_work_parts, legacy_loss_parts = work_parts.copy(), loss_parts.copy()
    legacy_before_energy_parts, legacy_after_energy_parts = old_energy_parts.copy(), new_energy_parts.copy()
    quadrature = prepare_restricted_source_quadrature(context)
    expansion_coordinates = None if native_coordinates is None else dict(
        old_displacement=operative_old_a, new_displacement=operative_new_a,
        displacement_increment=native_coordinates["dw"], before_velocity=operative_state.velocity_w_m_s,
        after_velocity=2*native_coordinates["dw"]/dt-operative_state.velocity_w_m_s,
        old_elastic_memory=native_coordinates["zold"], new_elastic_memory=native_coordinates["zend"],
        mean_elastic_memory=native_coordinates["zmean"])
    expansion_witnesses, expansion_passed = _source_expansion_witnesses(context, quadrature,
        before_memory, bulk, d, v0, v1, native_coordinates=expansion_coordinates)
    if not expansion_passed and operative_state is None:
        raise VolumeReleaseWorkRefusal("Operative point field differs from literal source expansion beyond arithmetic precision")
    stable = _stable_bulk_parts(context, quadrature, before_memory, loading, bulk, parent_old_force, d,
        native_coordinates=native_coordinates)
    ld = np.longdouble
    work_parts, loss_parts = stable["work"], stable["loss"]
    old_energy_parts, new_energy_parts = stable["before"], stable["after"]
    energy_increments, force_work_parts = stable["delta"], stable["force_work"]
    small_residual = energy_increments[1:]-(work_parts[1:]-loss_parts[1:])
    small_scale = np.abs(new_energy_parts[1:])+np.abs(old_energy_parts[1:])+np.abs(work_parts[1:])+np.abs(loss_parts[1:])
    small_passed = bool(np.all(np.abs(small_residual) <= (ld(work_relative_tolerance)+256*_EPS)*small_scale))
    duality_scale = np.abs(force_work_parts)+np.abs(work_parts)
    bulk_duality_bound = (ld(work_relative_tolerance)*duality_scale
        +stable["work_roundoff_bound"]+stable["force_dot_roundoff_bound"])
    bulk_duality_passed = bool(np.all(np.abs(force_work_parts-work_parts)
        <= bulk_duality_bound))
    operative_d = np.r_[np.asarray(q, ld), stable["dw"]]
    operative_v0 = np.r_[np.asarray(v0[:p.parent_ndof], ld),
        wld.T@np.asarray(v0[p.parent_ndof:], ld) if operative_state is None
        else np.asarray(operative_state.velocity_w_m_s, ld)]
    operative_v1 = 2*operative_d/ld(dt)-operative_v0
    if operative_state is not None:
        # Wire velocities use the same float64 recurrence as both the
        # immutable original parent journal and native coordinate journal.
        operative_v1 = np.r_[np.asarray(v1[:p.parent_ndof], ld),
            np.asarray(2*native_d[p.parent_ndof:]/dt-operative_state.velocity_w_m_s, ld)]
    drive = operative_d-ld(dt)*operative_v0
    mq = p.M[:p.parent_ndof, :p.parent_ndof]
    old_inertia = np.asarray(2*(mq@np.asarray(drive[:p.parent_ndof], float))/(dt*dt), ld)
    cross_old_inertia = 2*(quadrature.mass_cross@drive[p.parent_ndof:])/(ld(dt)*ld(dt))
    cross_added_inertia = 2*(quadrature.mass_cross.T@drive[p.selected_parent_dofs])/(ld(dt)*ld(dt))
    self_added_inertia = 2*(quadrature.mass_enriched@drive[p.parent_ndof:])/(ld(dt)*ld(dt))
    operative_inertia = np.r_[old_inertia.copy(), cross_added_inertia+self_added_inertia]
    np.add.at(operative_inertia, p.selected_parent_dofs, cross_old_inertia)
    operative_drag = np.r_[np.asarray(drag[:p.parent_ndof], ld), np.zeros(context.modes.rank, dtype=ld)]
    operative_gamma = np.r_[np.zeros(p.parent_ndof, dtype=ld),
        np.einsum("pai,pa,p->i", quadrature.surface_trace_local_w,
            np.asarray(cohesive.mean_traction_local_pa, ld), quadrature.surface_area_weights_m2)]
    operative_external = np.r_[np.asarray(external[:p.parent_ndof], ld),
        wld.T@np.asarray(external[p.parent_ndof:], ld)]
    operative_forces = (operative_inertia, operative_drag, stable["active_force"], operative_gamma, -operative_external)
    projected = sum(operative_forces)
    projected_scale = sum(np.abs(a) for a in operative_forces)
    operands = stable["operand_magnitudes"]
    point_count = operands["point_reduction_operation_count"]
    factor = 2/(ld(dt)*ld(dt))
    old_mass_magnitude = np.asarray(abs(mq)@np.asarray(np.abs(drive[:p.parent_ndof]), float), ld)*factor
    old_mass_bound = _summation_roundoff(old_mass_magnitude, p.parent_ndof+32, epsilon=_EPS)
    cross_mass_q_bound = _summation_roundoff(factor*(operands["mass_cross"]
        @np.abs(drive[p.parent_ndof:])), point_count+context.modes.rank+32)
    cross_mass_w_bound = _summation_roundoff(factor*(operands["mass_cross"].T
        @np.abs(drive[p.selected_parent_dofs])), point_count+len(p.selected_parent_dofs)+32)
    self_mass_w_bound = _summation_roundoff(factor*(operands["mass_enriched"]
        @np.abs(drive[p.parent_ndof:])), point_count+context.modes.rank+32)
    inertia_roundoff_bound = np.r_[old_mass_bound.copy(), cross_mass_w_bound+self_mass_w_bound]
    np.add.at(inertia_roundoff_bound, p.selected_parent_dofs, cross_mass_q_bound)
    drag_operand_magnitude = np.asarray(abs(p.D[:p.parent_ndof, :p.parent_ndof])
        @np.abs(q), ld)/ld(dt)
    drag_roundoff_bound = np.r_[_summation_roundoff(drag_operand_magnitude,
        p.parent_ndof+32, epsilon=_EPS), np.zeros(context.modes.rank, dtype=ld)]
    gamma_operand_magnitude = np.einsum("pai,pa,p->i", np.abs(quadrature.surface_trace_local_w),
        np.asarray(cohesive.mean_traction_integrated_absolute_pa, ld), quadrature.surface_area_weights_m2)
    gamma_point_roundoff = np.einsum("pai,pa,p->i", np.abs(quadrature.surface_trace_local_w),
        np.asarray(cohesive.mean_traction_component_roundoff_bound_pa, ld), quadrature.surface_area_weights_m2)
    gamma_roundoff_bound = np.r_[np.zeros(p.parent_ndof, dtype=ld),
        _summation_roundoff(gamma_operand_magnitude, 32+3*len(quadrature.surface_area_weights_m2))
        +gamma_point_roundoff+np.asarray(cohesive.mean_force_component_order_error_n, ld)]
    external_roundoff_bound = np.r_[np.zeros(p.parent_ndof, dtype=ld),
        _summation_roundoff(np.abs(wld).T@np.abs(np.asarray(external[p.parent_ndof:], ld)), 32)]
    operative_roundoff_bounds = (inertia_roundoff_bound, drag_roundoff_bound,
        stable["active_force_roundoff_bound"], gamma_roundoff_bound, external_roundoff_bound)
    projection_roundoff = sum(operative_roundoff_bounds)+256*_EPS*projected_scale
    component_bound = ld(momentum_relative_tolerance)*projected_scale+projection_roundoff
    momentum_passed = bool(np.all(np.abs(projected) <= component_bound))
    relative_error = float(np.max(np.divide(np.abs(projected), projected_scale,
        out=np.zeros_like(projected), where=projected_scale != 0.), initial=0.))
    external_work = float(operative_external@operative_d)
    kinetic_change = float(operative_d@operative_inertia)
    drag_loss = float(operative_d@operative_drag)
    if drag_loss < -_dot_bound(d, drag):
        raise VolumeReleaseWorkRefusal("Source basal dissipation became negative")
    kinetic_before = .5*float(v0@(p.M@v0))
    kinetic_after = .5*float(v1@(p.M@v1))
    kinetic_parts = np.array((operative_d[:p.parent_ndof]@old_inertia,
        operative_d[p.selected_parent_dofs]@cross_old_inertia+stable["dw"]@cross_added_inertia,
        stable["dw"]@self_added_inertia), ld)
    # The integrated vector, rather than subtraction of endpoint reference
    # energies, supplies interface work.  Preserve the endpoint kernel ledger
    # separately; an unresolved tiny change cannot satisfy the funding check.
    gamma_work = float(operative_gamma@operative_d)
    endpoint_addition = _endpoint_addition_witness(context, before_memory, bulk, d,
        operative_old_a, operative_new_a, stable["dw"], native_state=operative_state)
    gamma_duality_residual = gamma_work-cohesive.numerically_integrated_work_j
    gamma_duality_bound = (float(work_relative_tolerance)*(abs(gamma_work)
        +abs(cohesive.numerically_integrated_work_j))
        +_dot_bound(operative_gamma, operative_d)+cohesive.work_integral_roundoff_bound_j
        +float(gamma_roundoff_bound@np.abs(operative_d)))
    gamma_duality_passed = bool(abs(gamma_duality_residual) <= gamma_duality_bound
        and endpoint_addition["passed"])
    required_work = math.fsum((kinetic_change, drag_loss, *work_parts, gamma_work))
    balance = external_work-required_work
    residual_work = float(projected@operative_d)
    identity_scale = math.fsum((abs(external_work), abs(kinetic_change), abs(drag_loss),
        *(abs(float(v)) for v in work_parts), abs(gamma_work)))
    identity_bound = (float(work_relative_tolerance)*identity_scale
        +sum(_dot_bound(a, operative_d) for a in operative_forces)
        +float(sum(operative_roundoff_bounds)@np.abs(operative_d))
        +float(np.sum(stable["work_roundoff_bound"]+stable["force_dot_roundoff_bound"]))
        +cohesive.work_integral_roundoff_bound_j)
    identity_passed = bool(abs(balance+residual_work) <= identity_bound and small_passed
        and bulk_duality_passed and gamma_duality_passed)
    work_balance_passed = bool(abs(balance) <= identity_bound)
    state = accept_enriched_loading(p, trial, bulk)
    after_native = None if operative_state is None else _advance_native_release_state(
        context, operative_state, state, native_d, loading)
    body = dict(format=FORMAT+"-response", scope=SCOPE,
        context_binding_sha256=context.binding_sha256,
        before_memory_sha256=before_memory.binding_sha256,
        after_memory_sha256=state.binding_sha256,
        native_coordinate_history_used=operative_state is not None,
        native_coordinate_scope=None if operative_state is None else "source_born_authoritative_W_coordinates_fixed_material_private_journal",
        before_native_history_binding_sha256=None if operative_state is None else operative_state.binding_sha256,
        after_native_history_binding_sha256=None if after_native is None else after_native.binding_sha256,
        after_native_history=None if after_native is None else native_release_state_to_dict(after_native),
        full18_roundtrip_used_for_operative_coordinates=operative_state is None,
        full18_archive_fields_scope="expanded_float64_arithmetic_witness_only_not_authoritative_coordinates" if operative_state is not None else "strict_literal_full18_private_source_journal",
        legacy_full18_kernel_witnesses=kernel_witnesses,
        legacy_full18_kernel_witness_all_passed=kernel_witness_passed,
        legacy_full18_kernel_witness_scope="rounded_full18_cumulative_source_journal_arithmetic_witness_only_not_native_admission"
            if operative_state is not None else "strict_literal_full18_kernel_checks_required_for_admission",
        native_active_coordinate_count=None if operative_state is None else context.modes.rank,
        native_tied_kernel_coordinate_count=None if operative_state is None else 0,
        legacy_full18_cohesive_path_ledger_available=legacy_cohesive is not None,
        legacy_full18_cohesive_witness_evaluation_budget=dict(max_subdivisions=0,
            scope="bounded_optional_rounded_full18_witness_not_intrinsic_failure_certification")
            if operative_state is not None else dict(max_subdivisions=16, scope="strict_literal18_default_evaluation"),
        legacy_full18_cohesive_path_refusal=legacy_cohesive_refusal,
        legacy_full18_cohesive_force_scope="literal_full18_generic_path_arithmetic_witness" if legacy_cohesive is not None
            else "literal_full18_trace_contraction_of_authoritative_native_point_tractions_arithmetic_witness_only",
        legacy_full18_cohesive_witness_force_scope=None if legacy_cohesive is not None else
            "expanded_full_trace_action_of_primary_native_point_mean_traction_without_legacy_path_ledger",
        legacy_expanded_cohesive_endpoint_force_n=np.r_[np.zeros(p.parent_ndof), legacy_endpoint_gamma_force],
        before_time_myr=before_memory.time_myr, after_time_myr=float(after), dt_s=dt,
        increment_full_m=d, before_velocity_m_s=v0, after_velocity_m_s=v1,
        inertia_force_n=inertia, basal_drag_force_n=drag, bulk_mean_force_n=bulk.mean_generalized_force_n,
        cohesive_mean_force_n=gamma, external_force_n=external, residual_n=residual,
        parent_residual_n=projected[:p.parent_ndof],
        released_residual_n=np.asarray(b, ld).T@np.asarray(w, ld)@projected[p.parent_ndof:],
        observable_force_residual_n=projected[p.parent_ndof:],
        tied_kernel_residual_n=z.T@residual[p.parent_ndof:],
        projected_residual_scale_n=projected_scale, projected_momentum_component_bound_n=component_bound,
        operative_force_component_roundoff_bound_n=projection_roundoff,
        operative_inertia_component_roundoff_bound_n=inertia_roundoff_bound,
        operative_basal_drag_component_roundoff_bound_n=drag_roundoff_bound,
        operative_bulk_component_roundoff_bound_n=stable["active_force_roundoff_bound"],
        operative_cohesive_component_roundoff_bound_n=gamma_roundoff_bound,
        operative_external_component_roundoff_bound_n=external_roundoff_bound,
        operative_force_roundoff_scope="actual_source_point_and_original_parent_operands_gamma_n_reduction_bounds_no_floor",
        projected_momentum_relative_error=relative_error,
        full_force_fields_scope="legacy_float64_expanded_actions_arithmetic_witness_only",
        operative_bulk_field_scope=QUADRATURE_SCOPE,
        bulk_work_decomposition_scope="original_parent_analytic_plus_direct_restricted_source_point_cross_and_self_terms",
        weak_geometry_accuracy_certified=False, legacy_witness_differences_available_for_funding=False,
        source_point_expansion_witnesses=expansion_witnesses,
        source_point_expansion_passed=expansion_passed,
        source_point_expansion_is_native_admission_requirement=operative_state is None,
        source_point_expansion_bound_scope="256_float64_epsilon_times_actual_source_lift_projection_operands_no_floor",
        source_quadrature_binding_sha256=quadrature.binding_sha256,
        restricted_source_quadrature=restricted_source_quadrature_to_dict(quadrature),
        operative_increment_m=operative_d, operative_before_velocity_m_s=operative_v0,
        operative_after_velocity_m_s=operative_v1,
        operative_endpoint_enrichment_increment_m=endpoint_addition["endpoint_increment"],
        operative_endpoint_addition_difference_m=endpoint_addition["difference"],
        operative_endpoint_addition_roundoff_bound_m=endpoint_addition["bound"],
        full_enrichment_endpoint_addition_difference_m=endpoint_addition["full_difference"],
        full_enrichment_endpoint_addition_roundoff_bound_m=endpoint_addition["full_bound"],
        operative_endpoint_addition_passed=endpoint_addition["passed"],
        operative_endpoint_addition_scope="actual_source_coefficient_addition_and_coordinate_operands_gamma_n_no_floor_no_credit",
        operative_inertia_force_n=operative_inertia, operative_basal_drag_force_n=operative_drag,
        operative_bulk_mean_force_n=stable["active_force"], operative_cohesive_mean_force_n=operative_gamma,
        operative_external_force_n=operative_external, operative_projected_residual_n=projected,
        operative_bulk_cross_parent_force_n=stable["cross_q"],
        operative_bulk_cross_enrichment_force_n=stable["cross_w"],
        operative_bulk_self_enrichment_force_n=stable["self_w"],
        operative_enrichment_old_displacement_m=operative_old_a,
        operative_enrichment_new_displacement_m=operative_new_a,
        operative_enrichment_old_memory_m=stable["zold"], operative_enrichment_new_memory_m=stable["zend"],
        operative_enrichment_mean_memory_m=stable["zmean"],
        momentum_relative_tolerance=float(momentum_relative_tolerance),
        work_relative_tolerance=float(work_relative_tolerance), kinematic_identity_passed=True,
        cohesive_history_replayed_from_source_birth=True,
        bulk_force_work_decomposition_j=force_work_parts, bulk_work_decomposition_j=work_parts,
        bulk_energy_before_decomposition_j=old_energy_parts, bulk_energy_after_decomposition_j=new_energy_parts,
        bulk_energy_increment_decomposition_j=energy_increments,
        bulk_dissipation_decomposition_j=loss_parts,
        bulk_small_component_balance_residual_j=small_residual,
        bulk_force_work_duality_bound_decomposition_j=bulk_duality_bound,
        bulk_work_operand_magnitude_decomposition_j=stable["work_operand_magnitudes"],
        bulk_force_dot_operand_magnitude_decomposition_j=stable["force_dot_operand_magnitudes"],
        bulk_work_roundoff_bound_decomposition_j=stable["work_roundoff_bound"],
        bulk_force_dot_roundoff_bound_decomposition_j=stable["force_dot_roundoff_bound"],
        legacy_rounded_bulk_force_work_decomposition_j=legacy_force_work_parts,
        legacy_rounded_bulk_work_decomposition_j=legacy_work_parts,
        legacy_rounded_bulk_dissipation_decomposition_j=legacy_loss_parts,
        legacy_rounded_bulk_energy_before_decomposition_j=legacy_before_energy_parts,
        legacy_rounded_bulk_energy_after_decomposition_j=legacy_after_energy_parts,
        bulk_small_component_balance_passed=small_passed, bulk_force_work_duality_passed=bulk_duality_passed,
        bulk_parent_mean_force_n=parent_old_force, bulk_cross_parent_force_n=cross_parent_force,
        bulk_cross_enrichment_force_n=cross_extra_force, bulk_self_enrichment_force_n=self_extra_force,
        external_work_j=external_work, kinetic_change_stable_j=kinetic_change,
        kinetic_change_decomposition_j=kinetic_parts,
        kinetic_energy_before_j=kinetic_before, kinetic_energy_after_j=kinetic_after,
        kinetic_energy_before_after_scope="literal_full18_rounded_kinetic_subtraction_arithmetic_witness_only",
        kinetic_subtraction_roundoff_witness_j=kinetic_after-kinetic_before-kinetic_change,
        basal_dissipation_j=drag_loss, bulk_work_j=float(work_parts.sum()),
        cohesive_work_j=gamma_work, cohesive_endpoint_kernel_work_j=cohesive.mechanical_work_j,
        cohesive_force_work_duality_residual_j=gamma_duality_residual,
        cohesive_force_work_duality_bound_j=gamma_duality_bound,
        cohesive_force_work_duality_passed=gamma_duality_passed,
        cohesive_endpoint_work_agrees_at_integral_precision=cohesive.endpoint_work_agrees_at_integral_precision,
        cohesive_stored_energy_increment_j=cohesive.stored_energy_j-cohesive.old_stored_energy_j,
        fracture_dissipation_increment_j=cohesive.fracture_dissipation_increment_j,
        budget_required_work_j=required_work, work_balance_residual_j=balance,
        force_residual_work_j=residual_work, work_identity_bound_j=identity_bound,
        work_identity_passed=identity_passed, discrete_work_balance_passed=work_balance_passed,
        projected_momentum_passed=momentum_passed,
        balanced_trajectory=bool(momentum_passed and work_balance_passed and identity_passed),
        endpoint_ledger_resolved=cohesive.endpoint_work_agrees_at_integral_precision,
        funding_numerically_resolved=bool(momentum_passed and work_balance_passed and identity_passed
            and cohesive.endpoint_work_agrees_at_integral_precision),
        inherited_operator_energy_roundoff_j=before_memory.birth_operator_energy_roundoff_j,
        arithmetic_difference_is_available_work=False, complete_jump_kernel_remains_tied=True,
        historical_dissipation_available_for_creation_j=0., creation_work_debited_from_Genesis_j=0.,
        new_accepted_Genesis_time_myr=0., new_material_birth_count=0,
        physical_refusals=context.physical_refusals+(() if cohesive.endpoint_work_agrees_at_integral_precision
            else ("cohesive_endpoint_cumulative_energy_dissipation_increment_is_arithmetic_unresolved",))
            +(() if gamma_duality_passed else ("cohesive_mean_force_work_disagrees_with_actual_source_path_increment",)),
        new_maximum_separation_m=cohesive.new_maximum_separation_m,
        bulk_response_minimal=dict(mean0=bulk.mean0, mean1=bulk.mean1,
            mean_coefficients_m=bulk.mean_coefficients_m, coefficients_m=bulk.coefficients_m,
            displacement_m=bulk.displacement_m),
        cohesive_response=volume_release_cohesive_response_to_dict(cohesive),
        legacy_rounded_cohesive_response=None if legacy_cohesive is None else volume_release_cohesive_response_to_dict(legacy_cohesive),
        **{k: False for k in _FALSE})
    # Public packets keep ordinary JSON float64 numbers.  Extended precision
    # is private reduction arithmetic; padding bytes never enter a seal.
    body = {k: np.asarray(v, float) if isinstance(v, np.ndarray) else v for k, v in body.items()}
    body = _plain(body)
    payload = _canonical({**body, "release_work_response_sha256": _digest(body)}).decode()
    return VolumeReleaseWorkResponse(bulk, cohesive, state, payload, hashlib.sha256(payload.encode()).hexdigest(),
                                     _typed_pin((bulk, cohesive, state, after_native)), after_native)


def volume_release_work_response_to_dict(response):
    if (type(response) is not VolumeReleaseWorkResponse
            or response.binding_sha256 != hashlib.sha256(response.payload_json.encode()).hexdigest()
            or response.trusted_objects_sha256 != _typed_pin((response.bulk_response,
                 response.cohesive_response, response.after_memory, response.after_operative_state))):
        raise VolumeReleaseWorkRefusal("Release work response canonical binding differs")
    return json.loads(response.payload_json)


def replay_volume_release_work_response(context, raw, before_memory, old_maximum_separation_m):
    """Replay a saved computational step using trusted birth and prior history."""
    if type(raw) is not dict:
        raise VolumeReleaseWorkRefusal("Release work response packet is required")
    expected = evaluate_release_work(context, before_memory, old_maximum_separation_m,
        raw.get("increment_full_m"), raw.get("after_time_myr"),
        before_velocity_m_s=raw.get("before_velocity_m_s"),
        momentum_relative_tolerance=raw.get("momentum_relative_tolerance"),
        work_relative_tolerance=raw.get("work_relative_tolerance"))
    if _canonical(raw) != _canonical(volume_release_work_response_to_dict(expected)):
        raise VolumeReleaseWorkRefusal("Saved release work differs from source trajectory replay")
    return expected


def evaluate_native_release_work(context, before_memory, before_native,
        old_maximum_separation_m, increment_native_m, after_time_myr, *,
        before_velocity_m_s=None, momentum_relative_tolerance=3e-8, work_relative_tolerance=3e-9):
    """Continue the explicit source-born native chain; full18 remains a witness."""
    d = _array(increment_native_m, (context.partition.parent_ndof+context.modes.rank,), "Native increment")
    expected_v = np.r_[_journal_velocity(context, before_memory)[:context.partition.parent_ndof],
        before_native.velocity_w_m_s]
    if before_velocity_m_s is not None:
        _exact(_array(before_velocity_m_s, expected_v.shape, "Native before velocity"), expected_v,
            "Native before velocity")
    return evaluate_release_work(context, before_memory, old_maximum_separation_m,
        expand_native_release_increment(context, d), after_time_myr,
        momentum_relative_tolerance=momentum_relative_tolerance,
        work_relative_tolerance=work_relative_tolerance,
        operative_state=before_native, operative_increment_m=d)


def replay_native_release_work_response(context, raw, before_memory, before_native,
        old_maximum_separation_m):
    if type(raw) is not dict or not raw.get("native_coordinate_history_used"):
        raise VolumeReleaseWorkRefusal("Native release response packet is required")
    expected = evaluate_native_release_work(context, before_memory, before_native,
        old_maximum_separation_m, raw.get("operative_increment_m"), raw.get("after_time_myr"),
        before_velocity_m_s=raw.get("operative_before_velocity_m_s"),
        momentum_relative_tolerance=raw.get("momentum_relative_tolerance"),
        work_relative_tolerance=raw.get("work_relative_tolerance"))
    if _canonical(raw) != _canonical(volume_release_work_response_to_dict(expected)):
        raise VolumeReleaseWorkRefusal("Saved native work differs from source-born journal replay")
    return expected
