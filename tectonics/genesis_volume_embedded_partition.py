"""Tied compact embedded enrichment of the original Genesis bulk operator.

The prospective field is T q + H(phi_birth) w_birth T a, H=+/-1/2.
The sphere, angular and radial bubbles make w zero on the seed front and
all original cohort boundaries.  The surrounding ligament remains intact.
Only regular one-sided gradients enter bulk energy: the distributional
interface gradient is deliberately excluded.  All added amplitudes remain
constrained to zero.  Nonzero a is an explicitly computational virtual probe,
not a released physical state or representable old affine strain memory.

Quadrature integrates the small CURRENT seed ball itself, in two hemispheres,
and uniquely inverts the frozen current chart at each point.  Its reference
volume weight is the current weight times det(J_reference)/det(J_birth).
Thus a 100 m seed is not a mask on a million-metre parent quadrature.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib
import math

import numpy as np
from scipy import sparse

from .genesis_volume_checkpoint import _canonical
from .genesis_volume_operator import (
    ANGULAR_BARYCENTRIC, GenesisVolumeOperator, _immutable, conical_radial_moments,
    conical_strain_operators, evaluate_genesis_volume_operator, _sym_gradient_column,
)
from .genesis_volume_basal import local_consistent_mass_matrix
from .genesis_volume_banks import _frozen_csr, _hash_value
from .genesis_volume_material_sections import CohortMaterialSection, _validate_cohort_section
from .genesis_volume_embedded_surface import _numeric, _order, _current_chart_orientation
from .genesis_volume_finite_seed import (
    GenesisVolumeFiniteSeed, prepare_volume_finite_seed, validate_volume_finite_seed,
)

FORMAT = "genesis-tied-finite-seed-compact-bulk-enrichment-0.1"
SCOPE = "prospective_compact_virtual_enrichment_original_memory_no_release_or_funding"
BUBBLE = "H_plus_minus_half_times_sphere_squared_times_27_N0_N1_N2_times_4_t_1_minus_t"


class EmbeddedPartitionRefusal(ValueError):
    """A compact bulk support, work identity or declared input is unresolved."""


def _plain(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if sparse.issparse(v):
        return dict(shape=list(v.shape), indptr=v.indptr.tolist(),
                    indices=v.indices.tolist(), data=v.data.tolist())
    if isinstance(v, (tuple, list)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if hasattr(v, "__dataclass_fields__"):
        return {f.name: _plain(getattr(v, f.name)) for f in fields(v)}
    return v


def _digest(body):
    return hashlib.sha256(_canonical(body)).hexdigest()


def _matrix(v, shape, name):
    try:
        return _numeric(v, name, shape)
    except (TypeError, ValueError, OverflowError) as error:
        raise EmbeddedPartitionRefusal(str(error)) from error


def _freeze_record(value, *, float_arrays=False):
    for f in fields(value):
        v = getattr(value, f.name)
        if isinstance(v, np.ndarray):
            object.__setattr__(value, f.name, _immutable(v, float if float_arrays else v.dtype))


def _chart(section, N, lam):
    X, A, P = section.triangle_m, section.displacement_A_m, section.slope_P_m
    C, D = X+P, A-P
    Y = lam*C+D
    reference = lam*(N@X)
    current = N@Y
    Jr = np.column_stack((lam*(X[1]-X[0]), lam*(X[2]-X[0]), N@X))
    Jb = np.column_stack((Y[1]-Y[0], Y[2]-Y[0], N@C))
    return reference, current, Jr, Jb


def _inverse_coefficients(section):
    C = section.triangle_m+section.slope_P_m
    D = section.displacement_A_m-section.slope_P_m
    c1, c2, d1, d2 = C[1]-C[0], C[2]-C[0], D[1]-D[0], D[2]-D[0]
    v0 = np.cross(d1, d2)
    v1 = np.cross(d1, c2)+np.cross(c1, d2)
    v2 = np.cross(c1, c2)
    return C, D, v0, v1, v2


def _inverse_birth_point(section, point, coefficients=None):
    """All cubic radial roots, refusing multiple in-cohort inverse images."""
    C, D, v0, v1, v2 = _inverse_coefficients(section) if coefficients is None else coefficients
    delta = point-D[0]
    polynomial = np.array((-C[0]@v2, delta@v2-C[0]@v1,
                           delta@v1-C[0]@v0, delta@v0))
    nonzero = np.flatnonzero(polynomial != 0)
    if not len(nonzero):
        raise EmbeddedPartitionRefusal("Frozen birth chart inverse polynomial is unresolved")
    roots = np.roots(polynomial[nonzero[0]:]/np.max(np.abs(polynomial)))
    b, a = section.lambda_bounds
    candidates = []
    for root in roots:
        if abs(root.imag) > 128*np.finfo(float).eps*max(abs(root.real), 1.):
            continue
        lam = float(root.real)
        tolerance = 256*np.finfo(float).eps*max(a, 1.)
        if not b-tolerance <= lam <= a+tolerance:
            continue
        lam = min(max(lam, b), a)
        Y = lam*C+D
        edges = (Y[1:]-Y[0]).T
        uv = np.linalg.lstsq(edges, point-Y[0], rcond=None)[0]
        N = np.r_[1-uv.sum(), uv]
        n_tolerance = 256*np.finfo(float).eps*max(float(np.linalg.cond(edges)), 1.)
        residual = np.linalg.norm(N@Y-point)
        x_tolerance = 512*np.finfo(float).eps*max(np.linalg.norm(point), np.linalg.norm(Y), 1.)
        if np.min(N) < -n_tolerance or np.max(N) > 1+n_tolerance or residual > x_tolerance:
            continue
        # Only floating endpoint equivalence is canonicalized here.
        N = np.clip(N, 0., 1.)
        N /= N.sum()
        if not any(abs(lam-old[1]) <= tolerance for old in candidates):
            candidates.append((N, lam))
    if len(candidates) > 1:
        raise EmbeddedPartitionRefusal("Seed ball has multiple original-cohort birth inverse images")
    return candidates[0] if candidates else None


def _det3(matrix):
    return np.dot(matrix[:, 0], np.cross(matrix[:, 1], matrix[:, 2]))


def _solve3_precise(matrix, vector):
    """Three-vector adjugate solve, retaining longdouble arithmetic."""
    determinant = _det3(matrix)
    if not np.isfinite(determinant) or determinant == 0:
        raise EmbeddedPartitionRefusal("Precise frozen birth chart Jacobian is unresolved")
    return np.array((np.dot(vector, np.cross(matrix[:, 1], matrix[:, 2])),
                     np.dot(matrix[:, 0], np.cross(vector, matrix[:, 2])),
                     np.dot(matrix[:, 0], np.cross(matrix[:, 1], vector))))/determinant


def _precise_birth_inverse(section, center, ball_offset, coefficients):
    """Keep the cubic uniqueness guard, then refine in a centered chart.

    The coefficients remain the original float64 X+P and A-P sums. The
    small ball offset is retained before adding the large world center;
    it is never recovered by subtracting two rounded world positions.
    """
    center = np.asarray(center, np.longdouble)
    ball_offset = np.asarray(ball_offset, np.longdouble)
    chosen = _inverse_birth_point(section, np.asarray(center+ball_offset, float), coefficients)
    if chosen is None:
        return None
    C, D = (np.asarray(v, np.longdouble) for v in coefficients[:2])
    original_N, original_lam = chosen
    anchor_uv = np.asarray(original_N[1:], np.longdouble)
    anchor_lam = np.longdouble(original_lam)
    anchor_N = np.r_[1-anchor_uv.sum(), anchor_uv]
    vertices = anchor_lam*C+D
    relative_anchor = anchor_N@vertices-center
    dc1, dc2 = C[1]-C[0], C[2]-C[0]
    anchor_J = np.column_stack((vertices[1]-vertices[0], vertices[2]-vertices[0], anchor_N@C))
    increment = np.zeros(3, np.longdouble)
    for _ in range(8):
        du, dv, dl = increment
        relative = relative_anchor+anchor_J@increment+dl*(du*dc1+dv*dc2)
        jacobian = anchor_J+np.column_stack((dl*dc1, dl*dc2, du*dc1+dv*dc2))
        correction = _solve3_precise(jacobian, relative-ball_offset)
        increment -= correction
        if np.max(np.abs(correction)) <= 16*np.finfo(np.longdouble).eps:
            break
    du, dv, dl = increment
    relative = relative_anchor+anchor_J@increment+dl*(du*dc1+dv*dc2)
    resolution = 128*np.finfo(np.longdouble).eps*max(float(np.linalg.norm(ball_offset)), 1.)
    if np.max(np.abs(relative-ball_offset)) > resolution:
        raise EmbeddedPartitionRefusal("Precise centered birth inverse residual is unresolved")
    uv, lam = anchor_uv+increment[:2], anchor_lam+dl
    N = np.r_[1-uv.sum(), uv]
    b, a = section.lambda_bounds
    if np.min(N) < 0 or not b <= lam <= a:
        return None
    return N, lam, relative


def _bubble(section, center, radius, N, lam, *, birth_relative_m=None):
    """Value and reference-coordinate gradient of the frozen compact bubble."""
    reference, current, Jr, Jb = _chart(section, N, lam)
    b, a = section.lambda_bounds
    t = (lam-b)/(a-b)
    relative = current-center if birth_relative_m is None else birth_relative_m
    if birth_relative_m is not None:
        current = np.asarray(center, np.longdouble)+relative
    r2 = np.dot(relative, relative)/radius**2
    angular, radial = 27*np.prod(N), 4*t*(1-t)
    if r2 >= 1 or angular <= 0 or radial <= 0:
        return 0., np.zeros(3), reference, current, Jr, Jb
    sphere = (1-r2)**2
    angular_gradient = np.array((27*N[2]*(N[0]-N[1]),
                                 27*N[1]*(N[0]-N[2]), 0.))
    radial_gradient = np.array((0., 0., 4*(1-2*t)/(a-b)))
    sphere_gradient = -4*(1-r2)*(Jb.T@relative)/radius**2
    parameter_gradient = (sphere*radial*angular_gradient+
                          angular*radial*sphere_gradient+angular*sphere*radial_gradient)
    gradient = (_solve3_precise(Jr.T, parameter_gradient) if birth_relative_m is not None
                else np.linalg.solve(Jr.T, parameter_gradient))
    return sphere*angular*radial, gradient, reference, current, Jr, Jb


def _trace(N, lam):
    return np.concatenate([N[i]*np.concatenate((np.eye(3), (lam-1)*np.eye(3)), axis=1)
                           for i in range(3)], axis=1)


def evaluate_cohort_enrichment(section, center, radius, basis_xyz, barycentric, radial_scale):
    """Pointwise frozen bubble/trace/regular strain; pure computational helper."""
    _validate_cohort_section(section)
    N = _matrix(barycentric, (3,), "Enrichment barycentric point")
    if np.any(N < 0) or not np.isclose(N.sum(), 1., rtol=0, atol=2e-14):
        raise EmbeddedPartitionRefusal("Enrichment point lies outside the original angular cohort")
    if isinstance(radial_scale, (bool, np.bool_)) or not np.isfinite(radial_scale):
        raise EmbeddedPartitionRefusal("Enrichment lambda must be finite")
    lam = float(radial_scale)
    if not section.lambda_bounds[0] <= lam <= section.lambda_bounds[1]:
        raise EmbeddedPartitionRefusal("Enrichment point lies outside the original radial cohort")
    Q = _matrix(basis_xyz, (3, 3), "Enrichment proper basis")
    center = _matrix(center, (3,), "Enrichment seed center")
    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0:
        raise EmbeddedPartitionRefusal("Enrichment seed radius must be positive")
    return _cohort_enrichment_fields(section, center, radius, Q, N, lam)


def _cohort_enrichment_fields(section, center, radius, Q, N, lam, *, birth_relative_m=None):
    w, grad, xr, xb, Jr, Jb = _bubble(section, center, radius, N, lam,
                                    birth_relative_m=birth_relative_m)
    B0, B1 = conical_strain_operators(section.triangle_m, Q, N[None])
    B, T = B0[0]+B1[0]/lam, _trace(N, lam)
    # Retain the literal section plane rather than silently translating it
    # to the rounded sphere center. Extended precision protects subtraction.
    phi = float(np.dot(np.asarray(section.normal_xyz, np.longdouble),
                       np.asarray(xb, np.longdouble))-np.longdouble(section.offset_m))
    H = .5 if phi >= 0 else -.5
    extra = np.stack([_sym_gradient_column(Q.T@T[:, i], Q.T@grad) for i in range(18)], axis=1)
    return dict(bubble=w, gradient_reference_m_inverse=grad, reference_position_m=xr,
                birth_position_m=xb, reference_jacobian=Jr, birth_jacobian=Jb,
                side=H, parent_trace=T, parent_strain_operator=B,
                enrichment_trace=H*w*T, enrichment_strain_operator=H*(w*B+extra),
                jump_trace=w*T)


@dataclass(frozen=True, slots=True)
class CohortEmbeddedPartition:
    section: CohortMaterialSection
    seed_center_world_m: np.ndarray
    seed_radius_m: float
    elasticity_pa: np.ndarray
    basis_xyz: np.ndarray
    memory0: np.ndarray
    memory1: np.ndarray
    quadrature_order: int
    barycentric: np.ndarray
    radial_scale: np.ndarray
    reference_volume_weights_m3: np.ndarray
    birth_volume_weights_m3: np.ndarray
    reference_position_m: np.ndarray
    birth_position_m: np.ndarray
    side: np.ndarray
    bubble: np.ndarray
    bubble_gradient_reference_m_inverse: np.ndarray
    parent_trace: np.ndarray
    enrichment_trace: np.ndarray
    parent_strain_operator: np.ndarray
    enrichment_strain_operator: np.ndarray
    point_elastic_strain: np.ndarray
    point_stress_pa: np.ndarray
    parent_stiffness: np.ndarray
    parent_mass: np.ndarray
    parent_force_n: np.ndarray
    parent_energy_j: float
    stiffness_cross: np.ndarray
    stiffness_enriched: np.ndarray
    mass_cross: np.ndarray
    mass_enriched: np.ndarray
    enrichment_force_n: np.ndarray
    support_parent_stiffness_witness: np.ndarray
    numerical_enrichment_trace_rank: int
    numerical_enrichment_A_trace_rank: int
    numerical_enrichment_P_trace_rank: int
    normalized_enrichment_trace_condition_number: float
    center_plane_residual_m: float
    scaled_augmented_stiffness_minimum_eigenvalue: float

    def __post_init__(self):
        # Extended precision is private quadrature arithmetic. Public DTO
        # arrays retain the original float64 schema and canonical encoding.
        _freeze_record(self, float_arrays=True)


def prepare_cohort_embedded_partition(section, elasticity_pa, basis_xyz, strain0, strain1,
                                     *, seed_center_m, seed_radius_m, quadrature_order=8):
    """Compact local bulk insertion with all 18 new amplitudes still tied."""
    _validate_cohort_section(section)
    _current_chart_orientation(section)
    if not section.interior_cut:
        raise EmbeddedPartitionRefusal("A positive relative-interior finite sheet is required")
    order = _order(quadrature_order)
    C = _matrix(elasticity_pa, (6, 6), "Embedded elasticity")
    if not np.allclose(C, C.T, rtol=0, atol=64*np.finfo(float).eps*np.max(np.abs(C))) or np.linalg.eigvalsh(C).min() <= 0:
        raise EmbeddedPartitionRefusal("Embedded elasticity must be symmetric positive definite")
    Q = _matrix(basis_xyz, (3, 3), "Embedded basis")
    e0, e1 = (_matrix(v, (3, 6), "Complete saved angular memory") for v in (strain0, strain1))
    center = _matrix(seed_center_m, (3,), "Embedded seed center")
    if isinstance(seed_radius_m, (bool, np.bool_)):
        raise EmbeddedPartitionRefusal("Embedded radius must be a positive real")
    radius = float(seed_radius_m)
    if not math.isfinite(radius) or radius <= 64*np.finfo(float).eps*max(np.linalg.norm(center), 1.):
        raise EmbeddedPartitionRefusal("Embedded seed radius is nonpositive or unresolved")
    plane_error = abs(float(section.normal_xyz@center-section.offset_m))
    if plane_error > 64*np.finfo(float).eps*max(np.sum(np.abs(section.normal_xyz*center))+abs(section.offset_m), 1.):
        raise EmbeddedPartitionRefusal("Embedded sphere center differs from its declared plane")
    # This is an isotropic quadrature-coordinate convention, not a physical tangent.
    normal_precise = np.asarray(section.normal_xyz, np.longdouble)
    center_precise, radius_precise = np.asarray(center, np.longdouble), np.longdouble(radius)
    axis = np.eye(3, dtype=np.longdouble)[np.argmin(np.abs(normal_precise))]
    tangent = axis-normal_precise*(axis@normal_precise)
    tangent /= np.linalg.norm(tangent)
    tangent2 = np.cross(normal_precise, tangent)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    g, gw = (nodes+1)/2, weights/2
    anodes, aweights = np.polynomial.legendre.leggauss(2*order)
    angles, angleweights = math.pi*(anodes+1), math.pi*aweights
    coefficients = _inverse_coefficients(section)
    records = []
    for sign in (-1., 1.):
        for zfraction, zw in zip(g, gw):
            z = np.longdouble(sign)*radius_precise*np.longdouble(zfraction)
            rho2 = radius_precise*radius_precise-z*z
            for s, sw in zip(g, gw):
                rho = np.sqrt(rho2)*np.longdouble(s)
                for theta, tw in zip(angles, angleweights):
                    offset = z*normal_precise+rho*(math.cos(theta)*tangent+math.sin(theta)*tangent2)
                    inverse = _precise_birth_inverse(section, center_precise, offset, coefficients)
                    if inverse is None:
                        continue
                    N, lam, relative = inverse
                    data = _cohort_enrichment_fields(section, center_precise, radius_precise, Q, N, lam,
                                                     birth_relative_m=relative)
                    if data["bubble"] <= 0:
                        continue
                    current_weight = radius_precise*np.longdouble(zw)*rho2*np.longdouble(s)*np.longdouble(sw)*np.longdouble(tw)
                    dr, db = _det3(data["reference_jacobian"]), _det3(data["birth_jacobian"])
                    if not dr > 0 or not db > 0:
                        raise EmbeddedPartitionRefusal("Embedded seed volume chart has nonpositive Jacobian")
                    # Ball hemispheres are quadrature subdivisions only; the
                    # regular-field side is always the literal section plane.
                    records.append((N, lam, current_weight*dr/db, current_weight, data))
    if not records or set(r[4]["side"] for r in records) != {-.5, .5}:
        raise EmbeddedPartitionRefusal("Seed volume quadrature does not resolve both positive-volume sides")
    Ns, lams, refw, birthw = (np.asarray([r[i] for r in records]) for i in range(4))
    def collected(key):
        return np.asarray([r[4][key] for r in records])
    T, E, B, G = (collected(k) for k in ("parent_trace", "enrichment_trace", "parent_strain_operator", "enrichment_strain_operator"))
    angular_coeff = Ns@np.linalg.inv(ANGULAR_BARYCENTRIC)
    elastic = angular_coeff@e0+(angular_coeff@e1)/lams[:, None]
    stress = elastic@C.T
    Kqa = np.asarray(np.einsum("pai,ab,pbj,p->ij", B, C, G, refw), float)
    Kaa = np.asarray(np.einsum("pai,ab,pbj,p->ij", G, C, G, refw), float)
    density = section.mass_kg/section.volume_m3
    # Keep signed opposite-side mass contributions in extended precision.
    # Mqa AA can be many orders smaller than its positive Gram scale.
    Mqa = np.asarray(density*np.einsum("pai,paj,p->ij", T, E, refw), float)
    Maa = np.asarray(density*np.einsum("pai,paj,p->ij", E, E, refw), float)
    Fa = np.asarray(np.einsum("pai,pa,p->i", G, stress, refw), float)
    Ksupport = np.asarray(np.einsum("pai,ab,pbj,p->ij", B, C, B, refw), float)
    b, a = section.lambda_bounds
    smean, svar = conical_radial_moments(a, b)
    B0, B1 = conical_strain_operators(section.triangle_m, Q, ANGULAR_BARYCENTRIC)
    Bmean = B0+smean*B1
    Kparent = section.volume_m3/3*(np.einsum("qai,ab,qbj->ij", Bmean, C, Bmean)+
                                 svar*np.einsum("qai,ab,qbj->ij", B1, C, B1))
    emean = e0+smean*e1
    Fparent = section.volume_m3/3*(np.einsum("qai,qa->i", Bmean, emean@C.T)+
                                 svar*np.einsum("qai,qa->i", B1, e1@C.T))
    energy = section.volume_m3/6*(np.einsum("qa,ab,qb->", emean, C, emean)+
                                svar*np.einsum("qa,ab,qb->", e1, C, e1))
    Mparent = local_consistent_mass_matrix(section.mass_kg, a, b)
    augmented = np.block([[Kparent, Kqa], [Kqa.T, Kaa]])
    diagonal = np.sqrt(np.maximum(np.diag(augmented), np.finfo(float).tiny))
    scaled = augmented/diagonal[:, None]/diagonal[None, :]
    minimum = float(np.linalg.eigvalsh((scaled+scaled.T)/2).min())
    if not np.isfinite(augmented).all() or not np.trace(Kaa) > 0 or not np.trace(Maa) > 0 or minimum < -1e-9:
        raise EmbeddedPartitionRefusal("Compact augmented bulk energy is nonpositive or numerically unresolved")
    massaug = np.block([[Mparent, Mqa], [Mqa.T, Maa]])
    massdiag = np.sqrt(np.diag(massaug))
    if np.linalg.eigvalsh(massaug/massdiag[:, None]/massdiag[None, :]).min() < -1e-9:
        raise EmbeddedPartitionRefusal("Compact augmented inertia is nonpositive or numerically unresolved")
    # Report the rank of the actual archived float64 traces, not a private
    # precision estimate that the public DTO cannot reproduce.
    stacked = (np.sqrt(np.asarray(refw, float))[:, None, None]*np.asarray(E, float)).reshape(-1, 18)
    norms = np.linalg.norm(stacked, axis=0)
    normalized = stacked/np.maximum(norms, np.finfo(float).tiny)
    singular = np.linalg.svd(normalized, compute_uv=False)
    rank = int(np.linalg.matrix_rank(normalized))
    Acolumns = np.array((0, 1, 2, 6, 7, 8, 12, 13, 14))
    Pcolumns = Acolumns+3
    Arank, Prank = (int(np.linalg.matrix_rank(normalized[:, cols])) for cols in (Acolumns, Pcolumns))
    condition = float(singular[0]/max(singular[-1], np.finfo(float).tiny))
    center_residual = float(np.dot(np.asarray(section.normal_xyz, np.longdouble),
                                  np.asarray(center, np.longdouble))-np.longdouble(section.offset_m))
    if rank < 3:
        raise EmbeddedPartitionRefusal("Compact enriched displacement basis is numerically unresolved")
    return CohortEmbeddedPartition(section, center, radius, C, Q, e0, e1, order,
        Ns, lams, refw, birthw, collected("reference_position_m"), collected("birth_position_m"),
        collected("side"), collected("bubble"), collected("gradient_reference_m_inverse"),
        T, E, B, G, elastic, stress, Kparent, Mparent, Fparent, float(energy),
        Kqa, Kaa, Mqa, Maa, Fa, Ksupport, rank, Arank, Prank, condition, center_residual, minimum)


@dataclass(frozen=True, slots=True)
class GenesisVolumeEmbeddedPartition:
    parent_assembly: GenesisVolumeOperator
    parent_response_binding_sha256: str
    local: CohortEmbeddedPartition
    source_checkpoint_sha256: str
    source_identity: str
    time_myr: float
    parent_ndof: int
    ndof: int
    selected_cohort_id: int
    selected_cohort_index: int
    selected_parent_dofs: np.ndarray
    memory0: np.ndarray
    memory1: np.ndarray
    stiffness: object
    mass_matrix: object | None
    basal_drag_matrix: object | None
    external_force_n: np.ndarray | None
    initial_force_n: np.ndarray
    initial_elastic_energy_j: float
    prolongation: object
    tie_operator: object
    binding_sha256: str
    format: str = FORMAT
    scope: str = SCOPE

    def __post_init__(self):
        _freeze_record(self)

    @property
    def K(self): return self.stiffness
    @property
    def M(self): return self.mass_matrix
    @property
    def D(self): return self.basal_drag_matrix
    @property
    def R(self): return self.prolongation
    @property
    def J(self): return self.tie_operator
    @property
    def physical_creation_admitted(self): return False
    @property
    def physical_release_admitted(self): return False
    @property
    def released_dof_count(self): return 0


def _global_cross(local, dofs, n):
    rows = np.repeat(dofs, 18)
    columns = np.tile(np.arange(18), len(dofs))
    return sparse.coo_matrix((local.ravel(), (rows, columns)), shape=(n, 18)).tocsr()


def _parent_response_binding(a):
    h = hashlib.sha256()
    for name in ("ndof", "source_checkpoint_sha256", "source_identity", "time_myr",
                 "dofs", "cohort_ids", "face_indices", "volume_m3", "basis_xyz",
                 "elasticity_pa", "source_elastic_strain", "B0", "B1",
                 "vertex_B0", "vertex_B1", "inverse_scale_mean",
                 "inverse_scale_variance", "outer_scale", "inner_scale",
                 "cohort_stiffness", "stiffness"):
        _hash_value(h, getattr(a, name))
    return h.hexdigest()


def _partition_body(p):
    return dict(format=p.format, scope=p.scope, source_checkpoint_sha256=p.source_checkpoint_sha256,
        source_identity=p.source_identity, parent_response_binding_sha256=p.parent_response_binding_sha256,
        time_myr=p.time_myr, parent_ndof=p.parent_ndof,
        ndof=p.ndof, selected_cohort_id=p.selected_cohort_id,
        selected_cohort_index=p.selected_cohort_index,
        selected_parent_dofs=_plain(p.selected_parent_dofs), memory0=_plain(p.memory0),
        memory1=_plain(p.memory1), local=_plain(p.local), stiffness=_plain(p.K),
        mass_matrix=_plain(p.M), basal_drag_matrix=_plain(p.D),
        external_force_n=_plain(p.external_force_n), initial_force_n=_plain(p.initial_force_n),
        initial_elastic_energy_j=p.initial_elastic_energy_j,
        prolongation=_plain(p.R), tie_operator=_plain(p.J), bubble_definition=BUBBLE,
        bulk_gradient="regular_one_sided_reference_gradient_distributional_interface_delta_excluded",
        quadrature_domain="current_seed_ball_two_grid_hemispheres_unique_frozen_birth_chart_inverse",
        material_side_convention="sign_of_literal_section_normal_dot_birth_position_minus_section_offset",
        sphere_center_plane_residual_semantics="represented_plane_residual_no_physical_threshold_or_plane_translation",
        normalized_rank_semantics="formal_18_columns_retained_resolved_rank_reported_no_inversion_or_release_solver",
        quadrature_accuracy="approximate_cohort_clipping_requires_order_convergence_no_exact_integral_claim",
        reaction_semantics="negative_static_endpoint_bulk_elastic_enrichment_gradient_not_accepted_dynamic_birth_reaction",
        added_amplitudes="18_original_A_xyz_P_xyz_modes_per_selected_cohort_all_constrained_zero",
        enriched_memory_representable_by_old_affine_coefficients=False,
        physical_creation_admitted=False, physical_release_admitted=False, funding_admitted=False,
        released_dof_count=0, new_material_birth_count=0, creation_work_debited_from_Genesis_j=0.)


def prepare_embedded_partition(assembly, seed, elastic_constant, elastic_inverse_scale, *,
                               basal=None, quadrature_order=8):
    """Prepare an immutable tied prospective basis; accepted-source binding is external.

    This does not call a public checkpoint replay. The accepted model/state
    adapter must bind assembly, saved memory and seed to the true runtime.
    Absolute birth A/P affects geometry only, never gets added to strain twice.
    """
    if type(assembly) is not GenesisVolumeOperator or type(seed) is not GenesisVolumeFiniteSeed:
        raise EmbeddedPartitionRefusal("Original volume operator and finite seed are required")
    validate_volume_finite_seed(seed)
    if len(seed.surface.section_records) != 1:
        raise EmbeddedPartitionRefusal("One original selected-cohort seed is required")
    if not seed.surface.source_geometry_bound or seed.surface.parent_ndof != assembly.ndof:
        raise EmbeddedPartitionRefusal("Global compact insertion requires the original source-bound finite seed")
    section = seed.surface.section_records[0]
    matches = np.flatnonzero(assembly.cohort_ids == section.cohort_id)
    if len(matches) != 1:
        raise EmbeddedPartitionRefusal("Finite seed cohort differs from original volume material")
    ci = int(matches[0])
    cohort = assembly.geometry.cohorts[ci]
    if (section.cohort_index != ci or section.face_index != int(assembly.face_indices[ci]) or
        section.owner_id != cohort.owner_id or section.material_id != cohort.material_id or
        not np.array_equal(section.triangle_m, cohort.top_triangle_m) or
        section.lambda_bounds != (float(assembly.inner_scale[ci]), float(assembly.outer_scale[ci])) or
        section.volume_m3 != float(assembly.volume_m3[ci]) or
        section.mass_kg != float(cohort.mass_kg) or
        section.q_bounds != (float(cohort.q_lo), float(cohort.live_q_hi))):
        raise EmbeddedPartitionRefusal("Finite seed changed original cohort chart or identity")
    if seed.surface.source_geometry_bound and (
            seed.surface.source_checkpoint_sha256 != assembly.source_checkpoint_sha256 or
            seed.surface.source_identity != assembly.source_identity or seed.surface.time_myr != assembly.time_myr):
        raise EmbeddedPartitionRefusal("Finite seed source or clock differs from original operator")
    shape = (len(assembly.cohort_ids), 3, 6)
    e0, e1 = (_matrix(v, shape, "Full retained original cohort memory")
              for v in (elastic_constant, elastic_inverse_scale))
    local = prepare_cohort_embedded_partition(section, assembly.C[ci], assembly.basis_xyz[ci], e0[ci], e1[ci],
        seed_center_m=seed.seed_center_world_m, seed_radius_m=seed.seed_radius_m, quadrature_order=quadrature_order)
    n, dofs = assembly.ndof, assembly.dofs[ci]
    cross = _global_cross(local.stiffness_cross, dofs, n)
    K = _frozen_csr(sparse.bmat([[assembly.K, cross], [cross.T, sparse.csr_matrix(local.stiffness_enriched)]], format="csr"))
    M, D, Fext = None, None, None
    if basal is not None:
        if (basal.ndof != n or basal.source_checkpoint_sha256 != assembly.source_checkpoint_sha256 or
            basal.source_identity != assembly.source_identity or basal.time_myr != assembly.time_myr or
            basal.mass_matrix.shape != (n, n) or basal.basal_drag_matrix.shape != (n, n)):
            raise EmbeddedPartitionRefusal("Parent basal/inertial source, clock or support differs")
        crossm = _global_cross(local.mass_cross, dofs, n)
        M = _frozen_csr(sparse.bmat([[basal.mass_matrix, crossm], [crossm.T, sparse.csr_matrix(local.mass_enriched)]], format="csr"))
        # Bubble zero on every cohort boundary => no enriched basal traction/drag.
        D = _frozen_csr(sparse.block_diag((basal.basal_drag_matrix, sparse.csr_matrix((18, 18))), format="csr"))
        Fext = _immutable(np.r_[basal.external_force_n, np.zeros(18)])
    parent = evaluate_genesis_volume_operator(assembly, np.zeros(n), elastic_constant=e0, elastic_inverse_scale=e1)
    force = np.r_[parent.force, local.enrichment_force_n]
    R = _frozen_csr(sparse.vstack((sparse.eye(n), sparse.csr_matrix((18, n))), format="csr"))
    J = _frozen_csr(sparse.hstack((sparse.csr_matrix((18, n)), sparse.eye(18)), format="csr"))
    p = GenesisVolumeEmbeddedPartition(assembly, _parent_response_binding(assembly), local, assembly.source_checkpoint_sha256,
        assembly.source_identity, assembly.time_myr, n, n+18, section.cohort_id, ci, dofs, e0, e1,
        K, M, D, Fext, force, parent.energy_j, R, J, "")
    return replace(p, binding_sha256=_digest(_partition_body(p)))


def prepare_volume_embedded_partition(assembly, basal, displacement_m, normal_xyz, offset_m, *,
                                     cohort_id, seed_center_m, seed_radius_m, elastic_constant,
                                     elastic_inverse_scale, quadrature_order=8):
    seed = prepare_volume_finite_seed(assembly.geometry, displacement_m, normal_xyz, offset_m,
        cohort_id=cohort_id, seed_center_world_m=seed_center_m, seed_radius_m=seed_radius_m,
        quadrature_order=quadrature_order)
    return prepare_embedded_partition(assembly, seed, elastic_constant, elastic_inverse_scale,
                                     basal=basal, quadrature_order=quadrature_order)


def validate_volume_embedded_partition(p):
    if (type(p) is not GenesisVolumeEmbeddedPartition or p.format != FORMAT or p.scope != SCOPE or
        p.parent_response_binding_sha256 != _parent_response_binding(p.parent_assembly) or
        p.binding_sha256 != _digest(_partition_body(p))):
        raise EmbeddedPartitionRefusal("Embedded partition differs from immutable declared work/support binding")
    return p


def volume_embedded_partition_to_dict(p):
    validate_volume_embedded_partition(p)
    return {**_partition_body(p), "embedded_partition_sha256": p.binding_sha256}


@dataclass(frozen=True, slots=True)
class EmbeddedPartitionResponse:
    elastic_energy_j: float
    generalized_force_n: np.ndarray
    static_tie_reaction_n: np.ndarray
    finite_increment_bulk_work_j: float
    parent_memory0: np.ndarray
    parent_memory1: np.ndarray
    point_elastic_strain: np.ndarray
    computational_enrichment_probe: bool

    def __post_init__(self):
        _freeze_record(self)


def evaluate_embedded_partition(p, parent_increment_m, enrichment_virtual_m=None):
    """Evaluate quadratic virtual bulk energy, never accept/release a state."""
    validate_volume_embedded_partition(p)
    q = _matrix(parent_increment_m, (p.parent_ndof,), "New parent increment")
    a = np.zeros(18) if enrichment_virtual_m is None else _matrix(enrichment_virtual_m, (18,), "Computational enrichment probe")
    parent = evaluate_genesis_volume_operator(p.parent_assembly, q,
        elastic_constant=p.memory0, elastic_inverse_scale=p.memory1)
    qlocal, l = q[p.selected_parent_dofs], p.local
    added_force = l.enrichment_force_n+l.stiffness_cross.T@qlocal+l.stiffness_enriched@a
    parent_force = np.array(parent.force, copy=True)
    np.add.at(parent_force, p.selected_parent_dofs, l.stiffness_cross@a)
    addition = float(a@l.enrichment_force_n+a@l.stiffness_cross.T@qlocal+.5*a@l.stiffness_enriched@a)
    energy = parent.energy_j+addition
    delta = np.r_[q, a]
    work = float(p.initial_force_n@delta+.5*delta@(p.K@delta))
    point_strain = l.point_elastic_strain+np.einsum("pai,i->pa", l.parent_strain_operator, qlocal)+np.einsum("pai,i->pa", l.enrichment_strain_operator, a)
    if not math.isfinite(energy) or not np.isfinite(point_strain).all():
        raise EmbeddedPartitionRefusal("Embedded virtual bulk response overflowed")
    return EmbeddedPartitionResponse(float(energy), np.r_[parent_force, added_force],
        -added_force, work, parent.memory0, parent.memory1, point_strain, bool(np.any(a)))


def evaluate_embedded_enrichment_trace(p, barycentric, radial_scale):
    """Jump trace w*T at declared material points; no birth-plane reintersection."""
    validate_volume_embedded_partition(p)
    N = np.asarray(barycentric, float)
    lam = np.asarray(radial_scale, float)
    if N.ndim != 2 or N.shape[1] != 3 or lam.shape != (len(N),):
        raise EmbeddedPartitionRefusal("Embedded enrichment trace point shape mismatch")
    return _immutable(np.asarray([evaluate_cohort_enrichment(p.local.section,
        p.local.seed_center_world_m, p.local.seed_radius_m, p.local.basis_xyz, n, float(l))["jump_trace"]
        for n, l in zip(N, lam)]).reshape(-1, 3, 18))
