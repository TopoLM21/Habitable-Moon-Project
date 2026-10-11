"""Source-forced virgin trajectories with authoritative complete-range coordinates.

The fifteen (or source-resolved rank) observable coordinates are saved at
source birth and advanced by their own journal.  Expanding them to eighteen
original coefficients is a disclosed arithmetic witness, never a new
initial state or an input to the operative momentum/work calculation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from .genesis_volume_enriched_memory import (
    validate_enriched_memory, prepare_enriched_loading, evaluate_enriched_loading,
)
from .genesis_volume_operator import ANGULAR_BARYCENTRIC, evaluate_genesis_volume_operator
from .genesis_volume_release_history import validate_native_release_state
from .genesis_volume_release_quadrature import prepare_restricted_source_quadrature
from .genesis_volume_release_solve import (
    VolumeReleaseSolveRefusal, _array, _plain, _canonical, _digest,
    _effective_radius, _mean_tangent, _solve_scaled_system,
)
from .genesis_volume_release_work import (
    SECONDS_PER_MYR, _journal_velocity, validate_volume_release_work,
    evaluate_native_release_work, volume_release_work_response_to_dict,
)
from .genesis_volume_seed_memory import source_seed_memory_loading


FORMAT = "genesis-source-forced-native-virgin-release-solve-0.1"
SCOPE = "source_birth_native_range_journal_private_virgin_no_physical_release"
_EPS = np.finfo(float).eps


def _native_virgin_branch(context, native, old_maximum, fields):
    law, chart = context.cohesive.seed.law, context.operative_cohesive
    if law.compression_stiffness_pa_m != law.cohesive_stiffness_pa_m:
        raise VolumeReleaseSolveRefusal("Bounded native solver requires equal compression/cohesive stiffness")
    old = _array(old_maximum, (len(chart.reference_area_weights_m2),), "Old maximum separation")
    if np.any(old < 0.) or np.any(old > law.peak_separation_m):
        raise VolumeReleaseSolveRefusal("Bounded native solver does not support damaged history or softening")
    gap = np.einsum("pai,i->pa", fields.surface_trace_local_w,
        np.asarray(native.displacement_w_m, np.longdouble))
    internal = chart.seed.reference_offset_m+np.asarray(gap, float)
    required = np.maximum(chart.seed.initial_maximum_separation_m, _effective_radius(internal))
    roundoff = 32*_EPS*np.maximum(np.maximum(old, required), law.peak_separation_m)
    if np.any(old < required-roundoff):
        raise VolumeReleaseSolveRefusal("Old native history omits the source reference or current separation")
    if np.any(required > law.peak_separation_m):
        raise VolumeReleaseSolveRefusal("Bounded native solver old path is outside the virgin branch")
    return old, internal


def _native_linear_system(context, memory, native, after, velocity, fields):
    """Use original QQ source blocks and transformed point cross/self blocks."""
    p, ld = context.partition, np.longdouble
    n, r = p.parent_ndof, context.modes.rank
    dt = (after-memory.time_myr)*SECONDS_PER_MYR
    loading = source_seed_memory_loading(context.birth, memory, after)
    trial = prepare_enriched_loading(p, memory, loading)
    zero = evaluate_enriched_loading(p, trial, np.zeros(n), np.zeros(18))
    tangent, theta = _mean_tangent(p, loading)
    ci = p.selected_cohort_index
    K = np.zeros((n+r, n+r), dtype=ld)
    K[:n, :n] = tangent[:n, :n].toarray()
    K[p.selected_parent_dofs, n:] = theta[ci]*fields.stiffness_cross
    K[n:, p.selected_parent_dofs] = K[p.selected_parent_dofs, n:].T
    K[n:, n:] = theta[ci]*fields.stiffness_enriched
    M = np.zeros_like(K)
    M[:n, :n] = p.M[:n, :n].toarray()
    M[p.selected_parent_dofs, n:] = fields.mass_cross
    M[n:, p.selected_parent_dofs] = fields.mass_cross.T
    M[n:, n:] = fields.mass_enriched
    if np.any(p.D[n:, :].data != 0.) or np.any(p.D[:, n:].data != 0.):
        raise VolumeReleaseSolveRefusal("Native bounded solver requires zero source compact basal-drag blocks")
    D = np.zeros_like(K)
    D[:n, :n] = p.D[:n, :n].toarray()
    stiffness = context.cohesive.seed.law.cohesive_stiffness_pa_m
    A = 2*M/(dt*dt)+D/dt+K
    A[n:, n:] += .5*stiffness*fields.surface_gram
    # Old q/e0/e1 are unchanged original material history.  Added memory is
    # beta*z from the source-born journal; its rounded full18 view is ignored.
    zmean = ld(loading.beta[ci])*np.asarray(native.coefficients_w_m, ld)
    old_parent = evaluate_genesis_volume_operator(p.parent_assembly, np.zeros(n),
        elastic_constant=zero.mean0, elastic_inverse_scale=zero.mean1).force
    f0 = np.r_[np.asarray(old_parent, ld), np.zeros(r, dtype=ld)]
    np.add.at(f0, p.selected_parent_dofs, fields.stiffness_cross@zmean)
    angular = np.asarray(p.local.barycentric, ld)@np.asarray(np.linalg.inv(ANGULAR_BARYCENTRIC), ld)
    point = angular@np.asarray(zero.mean0[ci], ld)
    point += (angular@np.asarray(zero.mean1[ci], ld))/np.asarray(p.local.radial_scale, ld)[:, None]
    stress = point@np.asarray(fields.elasticity_pa, ld).T
    f0[n:] = np.einsum("pai,pa,p->i", fields.G_w, stress, fields.reference_volume_weights_m3)
    f0[n:] += fields.stiffness_enriched@zmean
    internal = np.asarray(context.operative_cohesive.seed.reference_offset_m, ld)+np.einsum(
        "pai,i->pa", fields.surface_trace_local_w, np.asarray(native.displacement_w_m, ld))
    gamma0 = stiffness*np.einsum("pai,pa,p->i", fields.surface_trace_local_w,
        internal, fields.surface_area_weights_m2)
    w = np.asarray(fields.basis_w, ld)
    F = np.r_[np.asarray(p.external_force_n[:n], ld), w.T@np.asarray(p.external_force_n[n:], ld)]
    rhs = F+2*(M@np.asarray(velocity, ld))/dt-f0-np.r_[np.zeros(n), gamma0]
    matrix, force = np.asarray(A, float), np.asarray(rhs, float)
    if not np.isfinite(matrix).all() or not np.isfinite(force).all():
        raise VolumeReleaseSolveRefusal("Native source system overflowed")
    return matrix, force, theta


@dataclass(frozen=True, slots=True)
class NativeVolumeReleaseSolveResult:
    response: object
    payload_json: str
    binding_sha256: str

    @property
    def physical_release_admitted(self): return False

    @property
    def funding_admitted(self): return False


def solve_native_release_work(context, before_memory, before_native,
        old_maximum_separation_m, after_time_myr, *, before_velocity_m_s=None):
    """Solve one private straight path, preserving every source jump-range mode."""
    validate_volume_release_work(context)
    p = context.partition
    validate_enriched_memory(p, before_memory)
    validate_native_release_state(context, before_native, before_memory)
    after = float(_array(after_time_myr, (), "After time"))
    dt = (after-before_memory.time_myr)*SECONDS_PER_MYR
    if not math.isfinite(dt) or dt <= 0.:
        raise VolumeReleaseSolveRefusal("After time must advance the private material clock")
    fields = prepare_restricted_source_quadrature(context)
    old, old_internal = _native_virgin_branch(context, before_native, old_maximum_separation_m, fields)
    expected = np.r_[_journal_velocity(context, before_memory)[:p.parent_ndof],
        before_native.velocity_w_m_s]
    velocity = expected if before_velocity_m_s is None else _array(
        before_velocity_m_s, (p.parent_ndof+context.modes.rank,), "Native before velocity")
    if not np.array_equal(velocity, expected):
        raise VolumeReleaseSolveRefusal("Native before velocity differs from the trusted source journal")
    matrix, rhs, theta = _native_linear_system(context, before_memory, before_native, after, velocity, fields)
    increment, diagnostics = _solve_scaled_system(matrix, rhs)
    # The actual float64 reduced increment is primary.  No full18 lift or
    # projection can alter either the endpoint checked here or the gate.
    new_internal = old_internal+np.asarray(np.einsum("pai,i->pa", fields.surface_trace_local_w,
        np.asarray(increment[p.parent_ndof:], np.longdouble)), float)
    new_radius = _effective_radius(new_internal)
    if not np.isfinite(new_radius).all() or np.any(new_radius > context.cohesive.seed.peak_separation_m):
        raise VolumeReleaseSolveRefusal("Bounded native path crosses into unsupported softening")
    response = evaluate_native_release_work(context, before_memory, before_native,
        old, increment, after, before_velocity_m_s=velocity)
    raw = volume_release_work_response_to_dict(response)
    if not all(raw.get(key) for key in ("balanced_trajectory", "projected_momentum_passed",
            "work_identity_passed", "discrete_work_balance_passed")):
        raise VolumeReleaseSolveRefusal("Native trajectory failed the independent full momentum/work gate")
    if response.cohesive_response.fracture_dissipation_increment_j != 0.:
        raise VolumeReleaseSolveRefusal("Bounded native virgin path acquired fracture dissipation")
    body = _plain(dict(format=FORMAT, scope=SCOPE,
        context_binding_sha256=context.binding_sha256,
        before_memory_sha256=before_memory.binding_sha256,
        before_native_history_binding_sha256=before_native.binding_sha256,
        response_binding_sha256=response.binding_sha256,
        source_quadrature_binding_sha256=fields.binding_sha256,
        after_native_history_binding_sha256=response.after_native.binding_sha256,
        after_time_myr=after, dt_s=dt, parent_ndof=p.parent_ndof,
        compact_jump_rank=context.modes.rank, tied_kernel_dimension=context.modes.nullity,
        increment_native_m=increment, solver=diagnostics, Maxwell_mean_stiffness_factors=theta,
        operative_coordinates="source_birth_complete_B_range_orthogonal_W_rotation",
        native_displacement_memory_velocity_authoritative=True,
        full18_expansion_scope="rounded_arithmetic_witness_not_operative_history",
        source_birth_native_displacement_memory_velocity_exact_zero=True,
        weak_source_fields_contracted_before_Gram_assembly=True,
        maximum_old_history_m=float(np.max(old)),
        maximum_old_effective_separation_m=float(np.max(_effective_radius(old_internal))),
        maximum_new_effective_separation_m=float(np.max(new_radius)),
        peak_separation_m=context.cohesive.seed.peak_separation_m,
        whole_straight_path_virgin_by_convex_effective_radius=True,
        complete_jump_kernel_remains_tied=True, balanced_trajectory=True,
        no_full18_roundtrip_used_for_operative_path=True,
        no_ULP_jitter_or_tied_kernel_freedom_used=True,
        new_accepted_Genesis_time_myr=0., new_material_birth_count=0,
        creation_work_debited_from_Genesis_j=0., physical_release_admitted=False,
        funding_admitted=False, accepted_planet_memory_changed=False))
    payload = _canonical({**body, "native_release_solve_sha256": _digest(body)}).decode()
    return NativeVolumeReleaseSolveResult(response, payload, hashlib.sha256(payload.encode()).hexdigest())


def volume_native_release_solve_to_dict(result):
    if (type(result) is not NativeVolumeReleaseSolveResult
            or result.binding_sha256 != hashlib.sha256(result.payload_json.encode()).hexdigest()):
        raise VolumeReleaseSolveRefusal("Native solve immutable binding differs")
    raw = json.loads(result.payload_json)
    response = volume_release_work_response_to_dict(result.response)
    if (raw.get("format") != FORMAT or raw.get("scope") != SCOPE
            or raw.get("native_release_solve_sha256") != _digest({k:v for k,v in raw.items()
                if k != "native_release_solve_sha256"})
            or raw.get("response_binding_sha256") != result.response.binding_sha256
            or raw.get("context_binding_sha256") != response.get("context_binding_sha256")
            or raw.get("before_memory_sha256") != response.get("before_memory_sha256")
            or raw.get("before_native_history_binding_sha256") != response.get("before_native_history_binding_sha256")
            or raw.get("source_quadrature_binding_sha256") != response.get("source_quadrature_binding_sha256")
            or raw.get("after_native_history_binding_sha256") != result.response.after_native.binding_sha256
            or not np.array_equal(np.asarray(raw.get("increment_native_m")),
                np.asarray(response.get("operative_increment_m")))
            or raw.get("balanced_trajectory") is not True or not response.get("balanced_trajectory")
            or any(raw.get(key) is not False for key in (
                "physical_release_admitted", "funding_admitted", "accepted_planet_memory_changed"))):
        raise VolumeReleaseSolveRefusal("Native solve response, source history or physical claim differs")
    return raw
