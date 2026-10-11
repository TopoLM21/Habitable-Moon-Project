"""Explicit prospective Rankine/Coulomb seeds on the inherited Genesis field.

The convex three-dimensional envelope is a NEW constitutive choice.  It is
not the compression-clamped finite-width slip rule, nor a toughness recovered
from formation.  The old diffuse damage/persistence gate remains compulsory.
Six material corners certify the maximum of the affine-angular/inverse-radius
stress field.  A unique qualifying maximum may define a prospective plane
and a sphere-clipped, single-original-cohort seed; its positive area, insertion
reaction, available work, and physical release are deliberately not certified.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real

import numpy as np

from .genesis_volume_checkpoint import _canonical
from .genesis_volume_creation_law import ProspectiveCreationLaw
from .genesis_volume_fracture_onset import (
    bind_calculated_volume_fracture_onset, calculated_volume_fracture_onset_to_dict)
from .genesis_volume_material_stress import (
    bind_volume_material_stress, evaluate_volume_material_stress,
    validate_volume_material_stress_observer_payload,
    volume_material_stress_observer_to_dict)
from .genesis_volume_slip_dynamics import VolumeSlipModel, VolumeSlipState


FORMAT = "genesis-source-bound-prospective-volume-nucleation-0.1"
SCOPE = "explicit_new_convex_source_gated_seed_proposal_not_funded_physical_surface"
CRITERION = "Rankine_and_classical_unclamped_Mohr_Coulomb_convex_envelope"
PROVENANCE = "explicit_prospective_constitutive_inputs_not_recovered_history"
CORNER_ORDER = "outer_then_inner_radial_endpoint_each_original_angular_vertex"
_FALSE_CLAIMS = ("physical_creation_admitted", "physical_release_admitted",
    "physical_fault_contact_admission_claimed", "funding_admitted",
    "history_reinitialized", "activation_accepted", "clock_advanced",
    "positive_finite_seed_area_certified", "insertion_reaction_replaced",
    "historical_dissipation_reclassified_as_funding")
_EPS = np.finfo(float).eps


class VolumeNucleationRefusal(ValueError):
    """The inputs cannot define an admissible prospective seed proposal."""


class VolumeNucleationTieRefusal(VolumeNucleationRefusal):
    """A numerical or physical degeneracy prevents a unique source choice."""


class VolumeNucleationEigenRefusal(VolumeNucleationRefusal):
    """The chosen eigenmode does not have a unique three-dimensional normal."""


class VolumeNucleationSupportRefusal(VolumeNucleationRefusal):
    """The plane only supports/touches a boundary, rather than cutting material."""


def _digest(body):
    return hashlib.sha256(_canonical(body)).hexdigest()


def _number(value, name, *, positive=True):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise VolumeNucleationRefusal(f"Nucleation {name} requires a finite real")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise VolumeNucleationRefusal(f"Nucleation {name} is outside its positive finite range")
    return value


def _array(value, name, shape=None):
    def boolean(v):
        return isinstance(v, (bool, np.bool_)) or (
            isinstance(v, (list, tuple)) and any(boolean(x) for x in v))
    if boolean(value):
        raise VolumeNucleationRefusal(f"Nucleation {name} requires real numeric data")
    try:
        a = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise VolumeNucleationRefusal(f"Nucleation {name} requires real numeric data") from error
    if a.dtype.kind not in "iuf":
        raise VolumeNucleationRefusal(f"Nucleation {name} requires real numeric data")
    with np.errstate(over="ignore", invalid="ignore"):
        a = np.asarray(a, float)
    if (shape is not None and a.shape != shape) or not np.isfinite(a).all():
        raise VolumeNucleationRefusal(f"Nucleation {name} has invalid shape or nonfinite values")
    return a


def _owned(a):
    a = np.asarray(a, float)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True, slots=True)
class ProspectiveVolumeNucleationLaw:
    """All required inputs are declared prospectively; none has a default.

    ``seed_radius_m`` restricts a proposed embryo to a world sphere intersected
    with the selected original cohort and plane.  It is not the inherited
    diffuse regularization length, a calculated propagation extent, or a new
    mesh scale.  ``creation_law`` supplies Gc/stiffness/strength only as an
    explicit new kernel.  Gc counts both banks once per REFERENCE material area.
    """
    tensile_strength_pa: float
    shear_cohesion_pa: float
    friction_coefficient: float
    seed_radius_m: float
    creation_law: ProspectiveCreationLaw

    def __post_init__(self):
        for name in ("tensile_strength_pa", "shear_cohesion_pa", "seed_radius_m"):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        mu = _number(self.friction_coefficient, "friction coefficient", positive=False)
        if not math.isfinite(math.hypot(1., mu)+mu):
            raise VolumeNucleationRefusal("Nucleation friction coefficient cannot resolve the convex envelope")
        object.__setattr__(self, "friction_coefficient", mu)
        if type(self.creation_law) is not ProspectiveCreationLaw:
            raise VolumeNucleationRefusal("Nucleation requires an explicit prospective creation law")
        object.__setattr__(self, "creation_law", ProspectiveCreationLaw(**asdict(self.creation_law)))


def _law(law):
    if type(law) is not ProspectiveVolumeNucleationLaw:
        raise VolumeNucleationRefusal("Nucleation requires explicit prospective constitutive inputs")
    # A frozen type alone does not protect against object.__setattr__ bypass.
    return ProspectiveVolumeNucleationLaw(**{
        k: getattr(law, k) for k in ("tensile_strength_pa", "shear_cohesion_pa",
            "friction_coefficient", "seed_radius_m", "creation_law")})


@dataclass(frozen=True, slots=True)
class VolumeNucleationCriterion:
    principal_stress_pa: np.ndarray
    tensile_score: np.ndarray
    shear_score: np.ndarray
    yield_score: np.ndarray

    def __post_init__(self):
        for name in ("principal_stress_pa", "tensile_score", "shear_score", "yield_score"):
            object.__setattr__(self, name, _owned(getattr(self, name)))


def evaluate_volume_nucleation_criterion(stress_tensor_pa, law):
    """Evaluate a new convex Rankine/classical Mohr-Coulomb envelope.

    For ordered s1>=s2>=s3 and h=sqrt(1+mu^2), f_t=s1/T-1 and
    f_s=((h+mu)s1-(h-mu)s3)/(2c)-1.  Both coefficients of s1 and -s3
    are positive, hence max(f_t,f_s) is convex in the complete stress tensor.
    Unlike the old slip rule this classical envelope does not clamp friction
    at tensile normal stress; its extrapolation is an explicit NEW closure.
    No eigenvectors, onset history or physical admission are inferred here.
    """
    law = _law(law)
    stress = _array(stress_tensor_pa, "stress tensor")
    if stress.ndim < 2 or stress.shape[-2:] != (3, 3) or not stress.size:
        raise VolumeNucleationRefusal("Nucleation stress requires nonempty (...,3,3) tensors")
    scale = np.maximum(np.max(np.abs(stress), axis=(-2, -1)), np.finfo(float).tiny)
    if np.any(np.max(np.abs(stress-stress.swapaxes(-1, -2)), axis=(-2, -1)) > 64*_EPS*scale):
        raise VolumeNucleationRefusal("Nucleation stress tensor must be symmetric")
    try:
        principal = np.linalg.eigvalsh(stress)
    except np.linalg.LinAlgError as error:
        raise VolumeNucleationRefusal("Nucleation stress eigenvalues are unresolved") from error
    high, low = principal[..., 2], principal[..., 0]
    coefficient = math.hypot(1., law.friction_coefficient)+law.friction_coefficient
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        tensile = high/law.tensile_strength_pa-1.
        # h-mu=1/(h+mu), avoiding cancellation for large finite mu.
        shear = .5*(coefficient*(high/law.shear_cohesion_pa)
                    -(low/law.shear_cohesion_pa)/coefficient)-1.
    if not np.isfinite(principal).all() or not np.isfinite(tensile).all() or not np.isfinite(shear).all():
        raise VolumeNucleationRefusal("Nucleation criterion overflowed its finite representation")
    return VolumeNucleationCriterion(principal, tensile, shear, np.maximum(tensile, shear))


def _unique_index(scores):
    scores = np.asarray(scores, float).reshape(-1)
    maximum = float(scores.max())
    tolerance = 128*_EPS*max(1., abs(maximum))
    matches = np.flatnonzero(maximum-scores <= tolerance)
    if len(matches) != 1:
        raise VolumeNucleationTieRefusal("Nucleation maximum is not unique at distinct material corners")
    return int(matches[0])


def _unit(value, name):
    a = _array(value, name, (3,))
    if not np.isclose(np.linalg.norm(a), 1., rtol=0., atol=64*_EPS):
        raise VolumeNucleationRefusal(f"Nucleation {name} must be a unit world vector")
    return a


def _unoriented(normal):
    normal = np.asarray(normal, float)
    return normal if normal[np.argmax(np.abs(normal))] >= 0 else -normal


def _orientation(stress, criterion, law, source_orientation):
    tensile, shear = float(criterion.tensile_score), float(criterion.shear_score)
    tolerance = 128*_EPS*max(1., abs(tensile), abs(shear))
    if abs(tensile-shear) <= tolerance:
        raise VolumeNucleationTieRefusal("Nucleation tensile/shear mode maximum is not unique")
    values, vectors = np.linalg.eigh(stress)
    eig_tol = 128*_EPS*max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    if values[2]-values[1] <= eig_tol:
        raise VolumeNucleationEigenRefusal("Nucleation dominant principal eigenvector is degenerate")
    if tensile > shear:
        return "tensile", _unoriented(vectors[:, 2]), None
    if values[1]-values[0] <= eig_tol:
        raise VolumeNucleationEigenRefusal("Nucleation shear minimum principal eigenvector is degenerate")
    theta = math.pi/4-.5*math.atan(law.friction_coefficient)
    plus = math.cos(theta)*vectors[:, 2]+math.sin(theta)*vectors[:, 0]
    minus = math.cos(theta)*vectors[:, 2]-math.sin(theta)*vectors[:, 0]
    preference = np.abs(np.array((plus@source_orientation, minus@source_orientation)))
    if abs(preference[0]-preference[1]) <= 128*_EPS:
        raise VolumeNucleationTieRefusal("Nucleation shear conjugates have equal inherited orientation preference")
    chosen = plus if preference[0] > preference[1] else minus
    return "shear", _unoriented(chosen), preference.tolist()


def propose_from_cohort_corner_fields(cohort, corner_stress_world_pa,
        corner_position_m, current_top_normal_world, source_orientation_world, law):
    """Pure computational six-corner fixture, never a trusted planet entry point.

    Radial order is outer, inner; each row contains the original three angular
    vertices.  Between corners the declared stress interpolation is bilinear
    in barycentric N and inverse reference radial scale.  A sphere/plane/cohort
    definition is returned; no clipping, area integration or funding is done.
    """
    law = _law(law)
    stress = _array(corner_stress_world_pa, "corner stress", (2, 3, 3, 3))
    position = _array(corner_position_m, "corner position", (2, 3, 3))
    top_normal = _unit(current_top_normal_world, "current top normal")
    source_normal = _unit(source_orientation_world, "inherited source orientation")
    for name in ("id", "face_index", "layer_index"):
        value = getattr(cohort, name, None)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
            raise VolumeNucleationRefusal("Nucleation fixture requires literal nonnegative cohort identity")
    qlo, qhi = _number(cohort.q_lo, "q low", positive=False), _number(cohort.live_q_hi, "q high")
    outer, inner = _number(cohort.outer_scale, "outer scale"), _number(cohort.inner_scale, "inner scale")
    if not (0 <= qlo < qhi <= 1 and 0 < inner < outer <= 1):
        raise VolumeNucleationRefusal("Nucleation fixture leaves its original live material interval")
    if not all(isinstance(getattr(cohort, name, None), str) and getattr(cohort, name)
               for name in ("owner_id", "material_id")):
        raise VolumeNucleationRefusal("Nucleation fixture requires inherited owner/material identity")
    criterion = evaluate_volume_nucleation_criterion(stress, law)
    maximum = float(criterion.yield_score.max())
    if maximum < 0:
        return dict(status="below_prospective_convex_threshold", maximum_yield_score=maximum,
            corner_order=CORNER_ORDER, candidate=None, scope="computational_corner_fields_not_planetary_activation")
    index = _unique_index(criterion.yield_score)
    radial, angular = divmod(index, 3)
    selected = evaluate_volume_nucleation_criterion(stress[radial, angular], law)
    mode, normal, preference = _orientation(stress[radial, angular], selected, law, source_normal)
    anchor = position[radial, angular]
    projection = (position.reshape(6, 3)-anchor)@normal
    resolution = 64*_EPS*max(1., float(np.max(np.linalg.norm(position.reshape(6, 3), axis=1))))
    if float(projection.min()) >= -resolution or float(projection.max()) <= resolution:
        raise VolumeNucleationSupportRefusal("Nucleation plane is boundary-only or touch-only on the selected cohort")
    offset = float(normal@anchor)
    dip = math.acos(float(np.clip(abs(normal@top_normal), 0., 1.)))
    body = dict(status="unique_prospective_seed_definition_requires_finite_clipping_and_insertion",
        scope="computational_corner_fields_not_planetary_activation", criterion=CRITERION,
        parameter_provenance=PROVENANCE, declared_prospective_law=asdict(law),
        maximum_yield_score=maximum, corner_order=CORNER_ORDER,
        corner_yield_scores=criterion.yield_score.tolist(),
        candidate=dict(cohort_id=int(cohort.id), face_index=int(cohort.face_index),
            layer_index=int(cohort.layer_index), owner_id=cohort.owner_id, material_id=cohort.material_id,
            angular_barycentric=np.eye(3)[angular].tolist(),
            mass_coordinate_q=qlo if radial == 0 else qhi,
            radial_scale=outer if radial == 0 else inner,
            mass_coordinate_q_bounds=[qlo, qhi], lambda_bounds=[inner, outer],
            current_nucleation_point_m=anchor.tolist(), selected_corner_index=index,
            dominant_mode=mode, corner_stress_world_pa=stress[radial, angular].tolist(),
            principal_stress_pa=selected.principal_stress_pa.tolist(),
            tensile_score=float(selected.tensile_score), shear_score=float(selected.shear_score),
            normal_world=normal.tolist(), plane_offset_m=offset, dip_from_current_top_plane_rad=dip,
            dip_reference="current_lambda_one_column_top_triangle_X_plus_A",
            shear_conjugate_preference=preference,
            shear_preference_rule="strict_absolute_alignment_with_inherited_source_orientation",
            finite_seed_definition="current_world_plane_intersection_sphere_intersection_selected_original_cohort",
            seed_radius_m=law.seed_radius_m, seed_center_world_m=anchor.tolist(),
            finite_seed_domain="selected_single_original_live_cohort_only",
            finite_seed_support_status="declared_not_clipped_or_positive_area_certified",
            finite_seed_reference_area_m2=None, full_creation_cost_j=None,
            prospective_gc_j_m2=law.creation_law.fracture_energy_j_m2,
            gc_area_convention="both_banks_once_per_reference_material_area",
            funding_source=None, physical_surface_selected=False),
        positive_finite_seed_area_certified=False, physical_creation_admitted=False,
        physical_release_admitted=False, funding_admitted=False,
        creation_work_debited_from_Genesis_j=0.)
    return {**body, "proposal_sha256": _digest(body)}


@dataclass(frozen=True, slots=True)
class VolumeNucleationProposal:
    payload_json: str
    binding_sha256: str


def bind_volume_nucleation_proposal(model, state, law=None, *, observer=None, onset=None):
    """Prepare prospective source proposals while preserving the inherited gate.

    An untrusted checkpoint first undergoes the public complete slip replay.
    No constitutive parameters are inferred if ``law`` is absent.  Previously
    active finite-width planes are not retroactively assigned sharp surfaces.
    """
    if type(model) is not VolumeSlipModel or type(state) is not VolumeSlipState:
        raise VolumeNucleationRefusal("Nucleation requires accepted Genesis slip model/state")
    declared = None if law is None else _law(law)
    expected_onset = bind_calculated_volume_fracture_onset(model, state)
    raw_onset = calculated_volume_fracture_onset_to_dict(expected_onset)
    if onset is not None and _canonical(calculated_volume_fracture_onset_to_dict(onset)) != _canonical(raw_onset):
        raise VolumeNucleationRefusal("Nucleation onset differs from complete accepted-source recomputation")
    expected_observer = bind_volume_material_stress(model, state)
    if observer is not None:
        validate_volume_material_stress_observer_payload(
            volume_material_stress_observer_to_dict(observer), expected_observer=expected_observer)
    observer = expected_observer
    new_faces = raw_onset["newly_activating_faces"]
    candidates, corner_catalogue, refusal = [], [], None
    status = "no_calculated_activation_event"
    if new_faces and declared is None:
        status = "missing_explicit_prospective_constitutive_inputs"
    elif new_faces:
        selected_cohorts = [c for c in state.geometry.cohorts if c.face_index in new_faces]
        if not selected_cohorts:
            raise VolumeNucleationRefusal("Nucleation event has no original live material support")
        ids = np.repeat([c.id for c in selected_cohorts], 6)
        N = np.tile(np.tile(np.eye(3), (2, 1)), (len(selected_cohorts), 1))
        q = np.concatenate([np.repeat((c.q_lo, c.live_q_hi), 3) for c in selected_cohorts])
        samples = evaluate_volume_material_stress(observer, ids, N, q)
        stresses = samples.stress_world_tensor.reshape(-1, 2, 3, 3, 3)
        positions = samples.current_position_m.reshape(-1, 2, 3, 3)
        criteria = evaluate_volume_nucleation_criterion(stresses, declared)
        for i, c in enumerate(selected_cohorts):
            corner_catalogue.append(dict(cohort_id=int(c.id), face_index=int(c.face_index),
                corner_yield_scores=criteria.yield_score[i].tolist(),
                corner_tensile_scores=criteria.tensile_score[i].tolist(),
                corner_shear_scores=criteria.shear_score[i].tolist()))
        maximum = float(criteria.yield_score.max())
        if maximum < 0:
            status = "source_gate_ready_but_below_prospective_convex_threshold"
        else:
            try:
                flat_index = _unique_index(criteria.yield_score)
                ci = flat_index//6
                c = selected_cohorts[ci]
                vertices = np.asarray(state.geometry.vertices_m)+state.displacement_m.reshape(-1, 6)[:, :3]
                triangle = vertices[np.asarray(state.geometry.faces)[c.face_index]]
                normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
                magnitude = float(np.linalg.norm(normal))
                if not math.isfinite(magnitude) or magnitude <= 0:
                    raise VolumeNucleationSupportRefusal("Nucleation current top face is degenerate")
                source_n = raw_onset["prepared_plane_normal_world"][c.face_index]
                proposal = propose_from_cohort_corner_fields(c, stresses[ci], positions[ci],
                    normal/magnitude, source_n, declared)
                candidates.append(proposal)
                status = "source_gated_unique_prospective_seed_not_physical_admission"
            except (VolumeNucleationTieRefusal, VolumeNucleationEigenRefusal,
                    VolumeNucleationSupportRefusal) as error:
                refusal = dict(type=type(error).__name__, reason=str(error))
                status = "source_gate_ready_but_unique_supported_seed_refused"
    body = dict(format=FORMAT, scope=SCOPE, criterion=CRITERION,
        parent_checkpoint_sha256=observer.parent_checkpoint_sha256,
        formation_checkpoint_sha256=observer.formation_checkpoint_sha256,
        source_identity=observer.source_identity, time_myr=state.time_myr,
        accepted_steps=observer.accepted_steps, accepted_step_sizes_myr=list(observer.accepted_step_sizes_myr),
        observer_binding_sha256=observer.binding_sha256, source_onset=raw_onset,
        status=status, newly_activating_faces=list(new_faces),
        existing_active_faces=list(raw_onset["existing_active_faces"]),
        declared_prospective_law=None if declared is None else asdict(declared),
        parameter_provenance=None if declared is None else PROVENANCE,
        source_recovered_creation_energy_law=None,
        corner_order=CORNER_ORDER, corner_catalogue=corner_catalogue,
        maximum_certificate="convex_tensor_envelope_on_affine_angular_inverse_lambda_corner_hull",
        numerical_uniqueness_tolerance="128_machine_epsilon_scaled_no_physical_threshold_shift",
        candidate=None if not candidates else candidates[0]["candidate"],
        computational_candidate_proposals=candidates, refusal=refusal,
        selected_sharp_material_surface=None,
        missing_physical_admission=["finite_seed_clipping_and_positive_reference_area",
            "embedded_or_constrained_subdomain_force_reaction_replacement",
            "calculated_incremental_work_and_creation_funding",
            "full_time_contact_and_release_trajectory"],
        inherited_runtime_ledgers=raw_onset["inherited_runtime_ledgers"],
        historical_dissipation_available_for_creation_j=0.,
        creation_work_debited_from_Genesis_j=0., **{k: False for k in _FALSE_CLAIMS})
    payload = {**body, "nucleation_sha256": _digest(body)}
    encoded = _canonical(payload).decode()
    return VolumeNucleationProposal(encoded, hashlib.sha256(encoded.encode()).hexdigest())


def volume_nucleation_proposal_to_dict(proposal):
    if type(proposal) is not VolumeNucleationProposal or proposal.binding_sha256 != hashlib.sha256(
            proposal.payload_json.encode()).hexdigest():
        raise VolumeNucleationRefusal("Nucleation proposal lost its immutable source binding")
    raw = json.loads(proposal.payload_json)
    if (raw.get("format") != FORMAT or raw.get("scope") != SCOPE
            or raw.get("criterion") != CRITERION or raw.get("nucleation_sha256") != _digest({
                k: v for k, v in raw.items() if k != "nucleation_sha256"})):
        raise VolumeNucleationRefusal("Nucleation proposal seal or scope differs")
    if any(raw.get(k) is not False for k in _FALSE_CLAIMS):
        raise VolumeNucleationRefusal("A prospective seed cannot claim physical creation/release/funding")
    if (raw.get("selected_sharp_material_surface") is not None
            or raw.get("source_recovered_creation_energy_law") is not None
            or type(raw.get("creation_work_debited_from_Genesis_j")) is not float
            or raw["creation_work_debited_from_Genesis_j"] != 0.
            or raw.get("historical_dissipation_available_for_creation_j") != 0.):
        raise VolumeNucleationRefusal("A prospective seed cannot invent history or creation funding")
    return raw


def volume_nucleation_proposal_from_dict(raw, *, model, state, law=None):
    expected = bind_volume_nucleation_proposal(model, state, law)
    try:
        equal = type(raw) is dict and _canonical(raw) == _canonical(volume_nucleation_proposal_to_dict(expected))
    except (TypeError, ValueError):
        equal = False
    if not equal:
        raise VolumeNucleationRefusal("Nucleation proposal differs from complete accepted-source recomputation")
    return expected
