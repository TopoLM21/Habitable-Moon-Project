"""Bounded source-forced computational trajectory on a virgin cohesive branch.

The old parent directions are free and the complete compact-jump kernel is
tied.  The sole supported cohesive branch has equal normal-compression and
cohesive stiffness and no history beyond the peak.  A direct, scaled solve
uses source mass, drag, forcing and the full Maxwell mean-force tangent; its
result must also pass the independent path momentum/work gate.  No result is
accepted into the planet or represented as physical fracture admission.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np
from scipy import sparse

from .genesis_volume_creation_law import _numeric
from .genesis_volume_enriched_memory import (
    prepare_enriched_loading, evaluate_enriched_loading, validate_enriched_memory,
)
from .genesis_volume_loading import average_volume_loading_stiffness_factors
from .genesis_volume_release_modes import validate_release_increment
from .genesis_volume_release_quadrature import prepare_restricted_source_quadrature
from .genesis_volume_operator import ANGULAR_BARYCENTRIC, evaluate_genesis_volume_operator
from .genesis_volume_release_work import (
    SECONDS_PER_MYR, _journal_velocity, evaluate_release_work,
    validate_volume_release_work, volume_release_work_response_to_dict,
)
from .genesis_volume_seed_memory import source_seed_memory_loading


FORMAT = "genesis-source-forced-virgin-release-solve-0.1"
SCOPE = "computational_fixed_material_virgin_equal_stiffness_no_physical_release"
_EPS = np.finfo(float).eps


class VolumeReleaseSolveRefusal(ValueError):
    """The bounded branch, source trajectory or linear solve is unresolved."""


def _array(value, shape, name):
    try:
        array = _numeric(value, name)
    except (ValueError, TypeError, OverflowError) as error:
        raise VolumeReleaseSolveRefusal(str(error)) from error
    if array.shape != shape:
        raise VolumeReleaseSolveRefusal(f"{name} shape differs from the source")
    return array


def _plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [_plain(item) for item in value]
    return value


def _canonical(value):
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _matrix_pin(value):
    array = np.ascontiguousarray(value, dtype=float)
    return hashlib.sha256(repr(array.shape).encode()+array.dtype.str.encode()+array.tobytes()).hexdigest()


def _effective_radius(internal):
    effective = internal.copy()
    effective[:, 0] = np.maximum(effective[:, 0], 0.)
    return np.hypot.reduce(effective, axis=1)


def _virgin_branch(context, memory, old_maximum):
    chart, law = context.cohesive, context.cohesive.seed.law
    if law.compression_stiffness_pa_m != law.cohesive_stiffness_pa_m:
        raise VolumeReleaseSolveRefusal("Bounded solver requires equal compression/cohesive stiffness")
    count = len(chart.reference_area_weights_m2)
    old = _array(old_maximum, (count,), "Old maximum separation")
    if np.any(old < 0.) or np.any(old > law.peak_separation_m):
        raise VolumeReleaseSolveRefusal("Bounded solver does not support damaged history or softening")
    fields = prepare_restricted_source_quadrature(context)
    coordinates = np.asarray(fields.basis_w, np.longdouble).T@np.asarray(memory.displacement_m, np.longdouble)
    local_gap = np.asarray(np.einsum("pai,i->pa", fields.surface_trace_local_w, coordinates), float)
    internal = chart.seed.reference_offset_m+local_gap
    required = np.maximum(chart.seed.initial_maximum_separation_m, _effective_radius(internal))
    roundoff = 32*_EPS*np.maximum(np.maximum(old, required), law.peak_separation_m)
    if np.any(old < required-roundoff):
        raise VolumeReleaseSolveRefusal("Old cohesive history omits the source reference or current separation")
    if np.any(required > law.peak_separation_m):
        raise VolumeReleaseSolveRefusal("Bounded solver old path is outside the virgin branch")
    return old, internal, fields


def _mean_tangent(partition, loading):
    """Assemble the original cohort-specific theta factors and all cross blocks."""
    parent = partition.parent_assembly
    theta = average_volume_loading_stiffness_factors(loading)
    dofs = parent.dofs
    local = theta[:, None, None]*parent.cohort_stiffness
    rows = np.repeat(dofs, dofs.shape[1], axis=1).ravel()
    columns = np.tile(dofs, (1, dofs.shape[1])).ravel()
    qq = sparse.coo_matrix((local.ravel(), (rows, columns)),
        shape=(partition.parent_ndof, partition.parent_ndof)).tocsr()
    qa = np.zeros((partition.parent_ndof, 18))
    selected_theta = float(theta[partition.selected_cohort_index])
    np.add.at(qa, partition.selected_parent_dofs,
        selected_theta*partition.local.stiffness_cross)
    aa = selected_theta*partition.local.stiffness_enriched
    return sparse.bmat(((qq, sparse.csr_matrix(qa)),
        (sparse.csr_matrix(qa.T), sparse.csr_matrix(aa))), format="csr"), theta


def _linear_system(context, memory, after_time_myr, before_velocity, fields):
    p, chart = context.partition, context.cohesive
    dt = (after_time_myr-memory.time_myr)*SECONDS_PER_MYR
    loading = source_seed_memory_loading(context.birth, memory, after_time_myr)
    trial = prepare_enriched_loading(p, memory, loading)
    zero = evaluate_enriched_loading(p, trial, np.zeros(p.parent_ndof), np.zeros(18))
    bulk_tangent, theta = _mean_tangent(p, loading)
    n, r = p.parent_ndof, context.modes.rank
    dtype = np.longdouble
    stiffness_matrix = np.zeros((n+r, n+r), dtype=dtype)
    stiffness_matrix[:n, :n] = bulk_tangent[:n, :n].toarray()
    ci_theta = theta[p.selected_cohort_index]
    qw = ci_theta*fields.stiffness_cross
    stiffness_matrix[p.selected_parent_dofs, n:] = qw
    stiffness_matrix[n:, p.selected_parent_dofs] = qw.T
    stiffness_matrix[n:, n:] = ci_theta*fields.stiffness_enriched
    mass = np.zeros_like(stiffness_matrix)
    mass[:n, :n] = p.M[:n, :n].toarray()
    mass[p.selected_parent_dofs, n:] = fields.mass_cross
    mass[n:, p.selected_parent_dofs] = fields.mass_cross.T
    mass[n:, n:] = fields.mass_enriched
    # Compact support vanishes on the original basal material boundaries;
    # the unchanged source drag operator therefore has no added-mode blocks.
    if (np.any(p.D[n:, :].data != 0.) or np.any(p.D[:, n:].data != 0.)):
        raise VolumeReleaseSolveRefusal("Stable bounded solver requires the source's zero compact basal-drag blocks")
    drag = np.zeros_like(stiffness_matrix)
    drag[:n, :n] = p.D[:n, :n].toarray()
    gram = fields.surface_gram
    stiffness = chart.seed.law.cohesive_stiffness_pa_m
    matrix = 2*mass/(dt*dt)+drag/dt+stiffness_matrix
    matrix[n:, n:] += .5*stiffness*gram
    w = np.asarray(fields.basis_w, dtype)
    z_mean = w.T@np.asarray(zero.mean_coefficients_m, dtype)
    old_parent = evaluate_genesis_volume_operator(p.parent_assembly, np.zeros(n),
        elastic_constant=zero.mean0, elastic_inverse_scale=zero.mean1).force
    mean_force = np.r_[np.asarray(old_parent, dtype), np.zeros(r, dtype=dtype)]
    np.add.at(mean_force, p.selected_parent_dofs, fields.stiffness_cross@z_mean)
    angular = np.asarray(p.local.barycentric, dtype)@np.asarray(np.linalg.inv(ANGULAR_BARYCENTRIC), dtype)
    ci = p.selected_cohort_index
    point = angular@np.asarray(zero.mean0[ci], dtype)
    point += (angular@np.asarray(zero.mean1[ci], dtype))/np.asarray(p.local.radial_scale, dtype)[:, None]
    stress = point@np.asarray(fields.elasticity_pa, dtype).T
    mean_force[n:] = np.einsum("pai,pa,p->i", fields.G_w, stress,
        fields.reference_volume_weights_m3)+fields.stiffness_enriched@z_mean
    coordinates = w.T@np.asarray(memory.displacement_m, dtype)
    old_internal = np.asarray(chart.seed.reference_offset_m, dtype)+np.einsum(
        "pai,i->pa", fields.surface_trace_local_w, coordinates)
    old_gamma = stiffness*np.einsum("pai,pa,p->i", fields.surface_trace_local_w,
        old_internal, fields.surface_area_weights_m2)
    restricted_velocity = np.r_[np.asarray(before_velocity[:n], dtype), w.T@np.asarray(before_velocity[n:], dtype)]
    external = np.r_[np.asarray(p.external_force_n[:n], dtype), w.T@np.asarray(p.external_force_n[n:], dtype)]
    rhs = external+2*(mass@restricted_velocity)/dt-mean_force-np.r_[np.zeros(n), old_gamma]
    restriction = sparse.block_diag((sparse.eye(p.parent_ndof, format="csr"),
        sparse.csr_matrix(np.asarray(fields.basis_w, float))), format="csr")
    reduced = np.asarray(matrix, float)
    reduced_rhs = np.asarray(rhs, float)
    if not np.isfinite(reduced).all() or not np.isfinite(reduced_rhs).all():
        raise VolumeReleaseSolveRefusal("Source linear system overflowed")
    return reduced, reduced_rhs, restriction, theta


def _solve_scaled_system(matrix, rhs):
    """Positive-diagonal scaling, direct solve and original-system refinement.

    No diagonal is replaced by a floor.  Conditioning is refused at the
    floating arithmetic resolution, and residuals use each component's own
    absolute matrix products rather than an unrelated force norm.
    """
    matrix = _array(matrix, np.asarray(matrix).shape, "Reduced matrix")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not len(matrix):
        raise VolumeReleaseSolveRefusal("Reduced matrix must be nonempty square")
    rhs = _array(rhs, (len(matrix),), "Reduced source force")
    diagonal = np.diag(matrix)
    if np.any(diagonal <= 0.):
        raise VolumeReleaseSolveRefusal("Reduced source operator has nonpositive diagonal")
    roots = np.sqrt(diagonal)
    scaled = (matrix/roots[:, None])/roots[None, :]
    scaled_rhs = rhs/roots
    if not np.isfinite(scaled).all() or not np.isfinite(scaled_rhs).all():
        raise VolumeReleaseSolveRefusal("Reduced diagonal scaling overflowed")
    symmetry_error = float(np.max(np.abs(scaled-scaled.T), initial=0.))
    symmetry_scale = float(np.max(np.abs(scaled), initial=0.))
    if symmetry_error > 256*_EPS*symmetry_scale:
        raise VolumeReleaseSolveRefusal("Reduced source operator is not symmetric at arithmetic precision")
    try:
        singular = np.linalg.svd(scaled, compute_uv=False)
        if singular[-1] <= 64*_EPS*singular[0]:
            raise VolumeReleaseSolveRefusal("Reduced source operator conditioning is unresolved")
        eigenvalues = np.linalg.eigvalsh(.5*(scaled+scaled.T))
        if eigenvalues[0] <= 0.:
            raise VolumeReleaseSolveRefusal("Reduced source operator is not positive definite")
        solution = np.linalg.solve(scaled, scaled_rhs)/roots
    except np.linalg.LinAlgError as error:
        raise VolumeReleaseSolveRefusal("Reduced source direct solve failed") from error
    matrix_ld, rhs_ld = np.asarray(matrix, np.longdouble), np.asarray(rhs, np.longdouble)
    refinements = 0
    for attempt in range(9):
        product = matrix_ld@np.asarray(solution, np.longdouble)
        residual = rhs_ld-product
        component_scale = np.abs(rhs_ld)+np.abs(matrix_ld)@np.abs(np.asarray(solution, np.longdouble))
        bound = 256*_EPS*component_scale
        if np.all(np.abs(residual) <= bound):
            break
        if attempt == 8:
            raise VolumeReleaseSolveRefusal("Reduced source direct solve failed component residual admission")
        try:
            correction = np.linalg.solve(scaled, np.asarray(residual, float)/roots)/roots
        except np.linalg.LinAlgError as error:
            raise VolumeReleaseSolveRefusal("Reduced source refinement failed") from error
        solution = solution+correction
        refinements += 1
        if not np.isfinite(solution).all():
            raise VolumeReleaseSolveRefusal("Reduced source solution overflowed")
    relative = np.divide(np.abs(residual), component_scale,
        out=np.zeros_like(residual), where=component_scale != 0.)
    diagnostics = dict(scaled_condition_number=float(singular[0]/singular[-1]),
        scaled_minimum_eigenvalue=float(eigenvalues[0]),
        scaled_symmetry_error=symmetry_error, refinement_iterations=refinements,
        maximum_component_backward_error=float(np.max(relative, initial=0.)),
        component_backward_error_bound=256*_EPS,
        diagonal_scaling=roots, reduced_matrix_sha256=_matrix_pin(matrix),
        reduced_source_force_sha256=_matrix_pin(rhs),
        positive_diagonal_scaled_without_floor=True,
        external_force_invented=False, historical_tie_reaction_used_as_force=False)
    return solution, diagnostics


@dataclass(frozen=True, slots=True)
class VolumeReleaseSolveResult:
    response: object
    payload_json: str
    binding_sha256: str

    @property
    def physical_release_admitted(self): return False

    @property
    def funding_admitted(self): return False


def solve_release_work(context, before_memory, old_maximum_separation_m,
        after_time_myr, *, before_velocity_m_s=None):
    """Solve one private, source-forced straight trajectory and require its gate.

    Effective cohesive radius is convex on the straight path; checking BOTH
    endpoint radii and old history certifies that the entire supported path
    stays virgin, including a normal sign change.  Softening is refused.
    """
    validate_volume_release_work(context)
    p = context.partition
    validate_enriched_memory(p, before_memory)
    after = float(_array(after_time_myr, (), "After time"))
    dt = (after-before_memory.time_myr)*SECONDS_PER_MYR
    if not math.isfinite(dt) or dt <= 0.:
        raise VolumeReleaseSolveRefusal("After time must advance the private material clock")
    old, old_internal, fields = _virgin_branch(context, before_memory, old_maximum_separation_m)
    expected_velocity = _journal_velocity(context, before_memory)
    velocity = expected_velocity if before_velocity_m_s is None else _array(
        before_velocity_m_s, (p.ndof,), "Before velocity")
    if not np.array_equal(velocity, expected_velocity):
        raise VolumeReleaseSolveRefusal("Before velocity differs from the trusted source trajectory")
    validate_release_increment(context.modes, before_memory.displacement_m, velocity[p.parent_ndof:])
    matrix, rhs, restriction, theta = _linear_system(context, before_memory, after, velocity, fields)
    reduced_increment, diagnostics = _solve_scaled_system(matrix, rhs)
    increment = np.asarray(restriction@reduced_increment).ravel()
    validate_release_increment(context.modes, increment[p.parent_ndof:])
    chart, law = context.cohesive, context.cohesive.seed.law
    recovered_coordinates = np.asarray(fields.basis_w, np.longdouble).T@np.asarray(
        increment[p.parent_ndof:], np.longdouble)
    intended_coordinates = np.asarray(reduced_increment[p.parent_ndof:], np.longdouble)
    expansion_error = recovered_coordinates-intended_coordinates
    expansion_scale = np.abs(recovered_coordinates)+np.abs(intended_coordinates)
    expansion_relative = np.divide(np.abs(expansion_error), expansion_scale,
        out=np.zeros_like(expansion_error), where=expansion_scale != 0.)
    new_internal = old_internal+np.asarray(np.einsum("pai,i->pa", fields.surface_trace_local_w,
        recovered_coordinates), float)
    new_radius = _effective_radius(new_internal)
    if not np.isfinite(new_radius).all() or np.any(new_radius > law.peak_separation_m):
        raise VolumeReleaseSolveRefusal("Bounded source-forced path crosses into unsupported softening")
    response = evaluate_release_work(context, before_memory, old, increment, after,
        before_velocity_m_s=velocity)
    raw_response = volume_release_work_response_to_dict(response)
    if (not raw_response.get("balanced_trajectory")
            or not raw_response.get("projected_momentum_passed")
            or not raw_response.get("work_identity_passed")
            or not raw_response.get("discrete_work_balance_passed")):
        raise VolumeReleaseSolveRefusal("Solved trajectory failed the independent full momentum/work gate")
    if response.cohesive_response.fracture_dissipation_increment_j != 0.:
        raise VolumeReleaseSolveRefusal("Bounded virgin trajectory acquired fracture dissipation")
    body = _plain(dict(format=FORMAT, scope=SCOPE,
        context_binding_sha256=context.binding_sha256,
        before_memory_sha256=before_memory.binding_sha256,
        response_binding_sha256=response.binding_sha256,
        after_time_myr=after, dt_s=dt, parent_ndof=p.parent_ndof,
        compact_jump_rank=context.modes.rank, tied_kernel_dimension=context.modes.nullity,
        source_quadrature_binding_sha256=fields.binding_sha256,
        operative_basis="observable_orthogonal_rotation_within_identical_complete_B_range",
        weak_source_fields_contracted_before_Gram_assembly=True,
        intended_release_coordinates_m=np.asarray(intended_coordinates, float),
        recovered_actual_release_coordinates_m=np.asarray(recovered_coordinates, float),
        release_coordinate_expansion_error_m=np.asarray(expansion_error, float),
        release_coordinate_expansion_component_scale_m=np.asarray(expansion_scale, float),
        maximum_release_coordinate_expansion_relative_error=float(np.max(expansion_relative, initial=0.)),
        virgin_path_checked_on_expanded_actual_float64_increment=True,
        increment_full_m=increment, solver=diagnostics,
        Maxwell_mean_stiffness_factors=theta,
        maximum_old_history_m=float(np.max(old)),
        maximum_old_effective_separation_m=float(np.max(_effective_radius(old_internal))),
        maximum_new_effective_separation_m=float(np.max(new_radius)),
        peak_separation_m=law.peak_separation_m,
        whole_straight_path_virgin_by_convex_effective_radius=True,
        balanced_trajectory=True, complete_jump_kernel_remains_tied=True,
        new_accepted_Genesis_time_myr=0., new_material_birth_count=0,
        creation_work_debited_from_Genesis_j=0.,
        physical_release_admitted=False, funding_admitted=False,
        accepted_planet_memory_changed=False))
    payload = _canonical({**body, "release_solve_sha256": _digest(body)}).decode()
    return VolumeReleaseSolveResult(response, payload, hashlib.sha256(payload.encode()).hexdigest())


def volume_release_solve_to_dict(result):
    if (type(result) is not VolumeReleaseSolveResult
            or result.binding_sha256 != hashlib.sha256(result.payload_json.encode()).hexdigest()):
        raise VolumeReleaseSolveRefusal("Source-forced solve immutable binding differs")
    raw = json.loads(result.payload_json)
    response = volume_release_work_response_to_dict(result.response)
    if (raw.get("format") != FORMAT or raw.get("scope") != SCOPE
            or raw.get("release_solve_sha256") != _digest({k: v for k, v in raw.items()
                if k != "release_solve_sha256"})
            or raw.get("response_binding_sha256") != result.response.binding_sha256
            or raw.get("balanced_trajectory") is not True
            or not response.get("balanced_trajectory")
            or any(raw.get(key) is not False for key in
                ("physical_release_admitted", "funding_admitted", "accepted_planet_memory_changed"))):
        raise VolumeReleaseSolveRefusal("Source-forced solve response, scope or physical claim differs")
    return raw
