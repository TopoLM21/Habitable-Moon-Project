"""Regular bulk operators on the resolved original-cohort/ball intersection.

The original full-cohort QQ blocks and complete saved e0/e1 field are retained.
Only QA/AA fields are integrated on the clipped support.  All eighteen added
coordinates remain tied: neither a quadrature nor agreement between finite
orders supplies a birth law, work source, release, or material/time advance.
Live arithmetic uses longdouble; a float64 archive is a separate logical DTO
and must be rebuilt from its authenticated source inputs for live precision.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib
import math

import numpy as np

from .genesis_volume_basal import local_consistent_mass_matrix
from .genesis_volume_checkpoint import _canonical
from .genesis_volume_clipped_domain import (
    CohortClippedDomain, clipped_domain_to_dict, validate_cohort_clipped_domain,
)
from .genesis_volume_embedded_partition import _solve3_precise
from .genesis_volume_operator import (
    ANGULAR_BARYCENTRIC, _immutable, conical_radial_moments,
    conical_strain_operators,
)

FORMAT = "genesis-original-cohort-clipped-regular-bulk-0.1"
SCOPE = "resolved_geometry_finite_order_computational_tied_no_physical_admission"
LD = np.longdouble
_EPS = np.finfo(LD).eps
_BLOCK_NAMES = ("stiffness_cross", "stiffness_enriched", "mass_cross",
                "mass_enriched", "enrichment_force_n")
_A = np.array((0, 1, 2, 6, 7, 8, 12, 13, 14))
_P = _A+3


class ClippedBulkRefusal(ValueError):
    """A declared field, contraction or finite-order comparison is unresolved."""


def _array(value, shape, name, *, dtype=LD):
    raw = np.asarray(value)
    if raw.dtype.kind not in "ifu" or raw.shape != shape:
        raise ClippedBulkRefusal(f"{name} requires finite real shape {shape}")
    result = np.asarray(raw, dtype=dtype)
    if not np.isfinite(result).all():
        raise ClippedBulkRefusal(f"{name} requires finite real shape {shape}")
    return result


def _plain(value):
    if isinstance(value, np.ndarray):
        return np.asarray(value, float).tolist()
    if isinstance(value, np.generic):
        return float(value) if value.dtype.kind == "f" else value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def _seal(body):
    return hashlib.sha256(_canonical(_plain(body))).hexdigest()


def _gamma(count):
    error = LD(count)*_EPS
    if error >= LD(.01):
        raise ClippedBulkRefusal("Reduction operation count is unresolved")
    return error/(1-error)


def _precision_digest(arrays):
    """Hash exact logical values as binary64 high/residual, never LD padding.

    On this arithmetic each LD value has at most 64 significant bits.  Two
    binary64 numbers therefore represent it exactly; the reconstruction guard
    refuses any exponent outside the portable pair representation.
    """
    digest = hashlib.sha256(b"exact-longdouble-binary64-high-residual-v1\0")
    for name, value in arrays:
        a = np.asarray(value, LD)
        digest.update(_canonical(dict(name=name, shape=list(a.shape))))
        flat = a.reshape(-1)
        for start in range(0, len(flat), 65536):
            part = flat[start:start+65536]
            high = np.asarray(part, dtype="<f8")
            low = np.asarray(part-high.astype(LD), dtype="<f8")
            if not np.isfinite(high).all() or not np.array_equal(high.astype(LD)+low.astype(LD), part):
                raise ClippedBulkRefusal("Live value has no exact portable high/residual representation")
            pair = np.column_stack((high, low)).astype("<f8", copy=False)
            digest.update(pair.tobytes(order="C"))
    return digest.hexdigest()


def _integrate(left, middle, right, weights, left_budget=None, right_budget=None):
    """Componentwise signed integral and its own absolute-operand budget."""
    n, _, lcols = left.shape
    rcols = right.shape[-1]
    result, absolute = np.zeros((lcols, rcols), LD), np.zeros((lcols, rcols), LD)
    propagated = np.zeros_like(result)
    for start in range(0, n, 512):
        stop = start+512
        L, R, w = left[start:stop], right[start:stop], weights[start:stop]
        result += np.einsum("pai,ab,pbj,p->ij", L, middle, R, w, optimize=True)
        absolute += np.einsum("pai,ab,pbj,p->ij", np.abs(L), np.abs(middle),
                              np.abs(R), w, optimize=True)
        if left_budget is not None:
            propagated += np.einsum("pai,ab,pbj,p->ij", left_budget[start:stop], np.abs(middle), np.abs(R), w, optimize=True)
        if right_budget is not None:
            propagated += np.einsum("pai,ab,pbj,p->ij", np.abs(L), np.abs(middle), right_budget[start:stop], w, optimize=True)
        if left_budget is not None and right_budget is not None:
            propagated += np.einsum("pai,ab,pbj,p->ij", left_budget[start:stop], np.abs(middle), right_budget[start:stop], w, optimize=True)
    return result, _gamma(n+4*middle.shape[0]+math.ceil(n/512)+8)*absolute+propagated


def _force(G, C, elastic, weights):
    n = len(weights)
    value = np.zeros(18, LD)
    absolute = np.zeros(18, LD)
    for start in range(0, n, 512):
        stop = start+512
        g, e, w = G[start:stop], elastic[start:stop], weights[start:stop]
        value += np.einsum("pai,ab,pb,p->i", g, C, e, w, optimize=True)
        absolute += np.einsum("pai,ab,pb,p->i", np.abs(g), np.abs(C), np.abs(e), w, optimize=True)
    return value, _gamma(n+32+math.ceil(n/512))*absolute


def _elastic(domain, e0, e1):
    angular = domain.barycentric @ np.asarray(np.linalg.inv(ANGULAR_BARYCENTRIC), LD)
    return angular@e0+(angular@e1)/domain.radial_scale[:, None]


def _parent_blocks(section, C, Q, e0, e1):
    """The established float64 full-cohort analytic QQ closure, unchanged."""
    b, a = section.lambda_bounds
    mean, variance = conical_radial_moments(a, b)
    B0, B1 = conical_strain_operators(section.triangle_m, Q, ANGULAR_BARYCENTRIC)
    Bmean = B0+mean*B1
    K = section.volume_m3/3*(np.einsum("qai,ab,qbj->ij", Bmean, C, Bmean)+
                            variance*np.einsum("qai,ab,qbj->ij", B1, C, B1))
    elastic = e0+mean*e1
    F = section.volume_m3/3*(np.einsum("qai,qa->i", Bmean, elastic@C.T)+
                            variance*np.einsum("qai,qa->i", B1, e1@C.T))
    return K, local_consistent_mass_matrix(section.mass_kg, a, b), F


def _parent_force_operand_budget(section, C, Q, e0, e1):
    """Own operands of the inherited float64 analytic parent force."""
    b, a = section.lambda_bounds
    mean, variance = conical_radial_moments(a, b)
    B0, B1 = conical_strain_operators(section.triangle_m, Q, ANGULAR_BARYCENTRIC)
    Bmeanabs = np.abs(np.asarray(B0, LD))+LD(abs(mean))*np.abs(np.asarray(B1, LD))
    emeanabs = np.abs(e0)+LD(abs(mean))*np.abs(e1)
    absolute = LD(section.volume_m3)/3*(np.einsum("qai,ab,qb->i", Bmeanabs, np.abs(np.asarray(C, LD)), emeanabs)+
        LD(abs(variance))*np.einsum("qai,ab,qb->i", np.abs(np.asarray(B1, LD)), np.abs(np.asarray(C, LD)), np.abs(e1)))
    return LD(256*np.finfo(float).eps)*absolute


@dataclass(frozen=True, slots=True)
class CohortClippedBulkOperator:
    domain: CohortClippedDomain
    elasticity_pa: np.ndarray
    basis_xyz: np.ndarray
    memory0: np.ndarray
    memory1: np.ndarray
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
    stiffness_cross: np.ndarray
    stiffness_enriched: np.ndarray
    mass_cross: np.ndarray
    mass_enriched: np.ndarray
    enrichment_force_n: np.ndarray
    stiffness_cross_budget: np.ndarray
    stiffness_enriched_budget: np.ndarray
    mass_cross_budget: np.ndarray
    mass_enriched_budget: np.ndarray
    enrichment_force_n_budget: np.ndarray
    live_precision_binding_sha256: str
    binding_sha256: str
    format: str = FORMAT
    scope: str = SCOPE

    @property
    def physical_creation_admitted(self):
        return False

    @property
    def physical_release_admitted(self):
        return False

    @property
    def point_count(self):
        return self.domain.point_count


def _live_arrays(op):
    return tuple((f.name, getattr(op, f.name)) for f in fields(op)
                 if isinstance(getattr(op, f.name), np.ndarray))


def _identity(op):
    return dict(format=op.format, scope=op.scope,
        domain_binding_sha256=op.domain.binding_sha256,
        source_section_binding_sha256=op.domain.source_section_binding_sha256,
        live_precision_binding_sha256=op.live_precision_binding_sha256,
        source_inputs=dict(elasticity_pa=op.elasticity_pa, basis_xyz=op.basis_xyz,
                           memory0=op.memory0, memory1=op.memory1),
        domain_geometry_resolved=True, finite_order_agreement_certified=False,
        physical_contraction_certified=False, physical_creation_admitted=False,
        physical_release_admitted=False, available_for_creation_funding=False,
        accepted_time_advance_myr=0., created_material_kg=0., work_debit_j=0.,
        added_coordinate_count=18, added_coordinates_tied=True,
        parent_QQ_semantics="unchanged_original_full_cohort_analytic_float64_blocks",
        live_arithmetic="longdouble_original_float64_source_vertex_fields_interpolated_before_Gram",
        archive_arithmetic="literal_float64_logical_tables_reconstruct_live_from_source_inputs",
        roundoff_available_for_funding=False)


def validate_clipped_bulk_operator(op):
    if type(op) is not CohortClippedBulkOperator or op.format != FORMAT or op.scope != SCOPE:
        raise ClippedBulkRefusal("Invalid clipped bulk operator format or scope")
    validate_cohort_clipped_domain(op.domain)
    if _precision_digest(_live_arrays(op)) != op.live_precision_binding_sha256 or _seal(_identity(op)) != op.binding_sha256:
        raise ClippedBulkRefusal("Clipped bulk differs from its immutable live/source binding")
    return True


def prepare_clipped_bulk_operator(domain, C, Q, e0, e1):
    """Integrate regular full-18 fields on the geometrically clipped support."""
    validate_cohort_clipped_domain(domain)
    if domain.point_count == 0:
        raise ClippedBulkRefusal("Clipped bulk requires nonempty positive-volume support")
    if domain.point_count > 250000:
        raise ClippedBulkRefusal("Full-18 bulk point table exceeds its bounded 250000-point limit")
    C, Q = (_array(v, shape, name, dtype=float) for v, shape, name in
            ((C, (6, 6), "Source elasticity"), (Q, (3, 3), "Source proper basis")))
    e0, e1 = (_array(v, (3, 6), "Complete source angular memory", dtype=float) for v in (e0, e1))
    if not np.allclose(C, C.T, rtol=0, atol=64*np.finfo(float).eps*np.max(np.abs(C))) or np.linalg.eigvalsh(C).min() <= 0:
        raise ClippedBulkRefusal("Source elasticity must be symmetric positive definite")
    section = domain.section
    try:
        vb0, vb1 = conical_strain_operators(section.triangle_m, Q, np.eye(3))
    except ValueError as error:
        raise ClippedBulkRefusal(str(error)) from error
    N, lam, offset = domain.barycentric, domain.radial_scale, domain.centered_birth_offsets_m
    n = len(lam)
    B = np.einsum("pi,iaj->paj", N, np.asarray(vb0, LD))
    B += np.einsum("pi,iaj->paj", N, np.asarray(vb1, LD))/lam[:, None, None]
    T = np.zeros((n, 3, 18), LD)
    for i in range(3):
        T[:, :, 6*i:6*i+3] = N[:, i, None, None]*np.eye(3, dtype=LD)
        T[:, :, 6*i+3:6*i+6] = N[:, i, None, None]*(lam-1)[:, None, None]*np.eye(3, dtype=LD)
    b, a = map(LD, section.lambda_bounds)
    t = (lam-b)/(a-b)
    r2 = np.einsum("pi,pi->p", offset, offset)/LD(domain.seed_radius_m)**2
    sphere, angular, radial = (1-r2)**2, 27*np.prod(N, axis=1), 4*t*(1-t)
    w = sphere*angular*radial
    if np.any(w <= 0) or np.any(r2 >= 1):
        raise ClippedBulkRefusal("Resolved domain point has no positive compact bubble")
    birthC = np.asarray(section.triangle_m+section.slope_P_m, LD)
    birthD = np.asarray(section.displacement_A_m-section.slope_P_m, LD)
    dc1, dc2 = birthC[1]-birthC[0], birthC[2]-birthC[0]
    dd1, dd2 = birthD[1]-birthD[0], birthD[2]-birthD[0]
    spheregrad = -4*(1-r2)[:, None]/LD(domain.seed_radius_m)**2*np.column_stack((
        np.einsum("pi,pi->p", lam[:, None]*dc1+dd1, offset),
        np.einsum("pi,pi->p", lam[:, None]*dc2+dd2, offset),
        np.einsum("pi,pi->p", N@birthC, offset)))
    angulargrad = np.column_stack((27*N[:, 2]*(N[:, 0]-N[:, 1]),
                                  27*N[:, 1]*(N[:, 0]-N[:, 2]), np.zeros(n, LD)))
    radialgrad = np.zeros((n, 3), LD)
    radialgrad[:, 2] = 4*(1-2*t)/(a-b)
    param = sphere[:, None]*radial[:, None]*angulargrad
    param += angular[:, None]*radial[:, None]*spheregrad
    param += angular[:, None]*sphere[:, None]*radialgrad
    X = np.asarray(section.triangle_m, LD)
    chart = np.column_stack((X[1]-X[0], X[2]-X[0], X[0]))
    inverse = np.column_stack([_solve3_precise(chart, v) for v in np.eye(3, dtype=LD)])
    du, dv, dl = param.T
    gradient = (du/lam)[:, None]*inverse[0]+(dv/lam)[:, None]*inverse[1]
    gradient += (dl-(du*N[:, 1]+dv*N[:, 2])/lam)[:, None]*inverse[2]
    localT = np.einsum("ai,pij->paj", np.asarray(Q.T, LD), T)
    localgrad = gradient@np.asarray(Q, LD)
    extra = np.stack((localT[:, 0]*localgrad[:, 0, None],
        localT[:, 1]*localgrad[:, 1, None], localT[:, 2]*localgrad[:, 2, None],
        localT[:, 0]*localgrad[:, 1, None]+localT[:, 1]*localgrad[:, 0, None],
        localT[:, 0]*localgrad[:, 2, None]+localT[:, 2]*localgrad[:, 0, None],
        localT[:, 1]*localgrad[:, 2, None]+localT[:, 2]*localgrad[:, 1, None]), axis=1)
    H = domain.side
    E, G = H[:, None, None]*w[:, None, None]*T, H[:, None, None]*(w[:, None, None]*B+extra)
    elastic = _elastic(domain, np.asarray(e0, LD), np.asarray(e1, LD))
    cld, weights = np.asarray(C, LD), domain.reference_volume_weights_m3
    stress = elastic@cld.T
    Kqa, Kqb = _integrate(B, cld, G, weights)
    Kaa, Kab = _integrate(G, cld, G, weights)
    density = LD(section.mass_kg)/LD(section.volume_m3)
    Mqa, Mqb = _integrate(T, np.eye(3, dtype=LD), E, weights)
    Maa, Mab = _integrate(E, np.eye(3, dtype=LD), E, weights)
    Mqa, Mqb, Maa, Mab = (density*v for v in (Mqa, Mqb, Maa, Mab))
    Fa, Fab = _force(G, cld, elastic, weights)
    Kparent, Mparent, Fparent = _parent_blocks(section, C, Q, e0, e1)
    values = (C, Q, e0, e1, w, gradient, T, E, B, G, elastic, stress,
              Kparent, Mparent, Fparent, Kqa, Kaa, Mqa, Maa, Fa, Kqb, Kab, Mqb, Mab, Fab)
    if any(not np.isfinite(v).all() for v in values):
        raise ClippedBulkRefusal("Clipped bulk field or integral overflowed")
    values = tuple(_immutable(v, v.dtype) for v in values)
    op = CohortClippedBulkOperator(domain, *values, "", "")
    op = replace(op, live_precision_binding_sha256=_precision_digest(_live_arrays(op)))
    return replace(op, binding_sha256=_seal(_identity(op)))


def memory_cross_force(op, e0, e1):
    """Full inherited angular/normal/shear memory crossed with added fields."""
    validate_clipped_bulk_operator(op)
    e0, e1 = (_array(v, (3, 6), "Declared complete angular memory") for v in (e0, e1))
    force, budget = _force(op.enrichment_strain_operator, np.asarray(op.elasticity_pa, LD),
                           _elastic(op.domain, e0, e1), op.domain.reference_volume_weights_m3)
    return dict(force_n=_immutable(force, LD), absolute_roundoff_budget_n=_immutable(budget, LD))


def _dot(vector, force, budget):
    value = vector@force
    bound = np.abs(vector)@budget+_gamma(20)*(np.abs(vector)@np.abs(force))
    return value, bound


def _project(field, coefficients):
    value = np.einsum("pai,ij->paj", field, coefficients, optimize=True)
    budget = _gamma(20)*np.einsum("pai,ij->paj", np.abs(field), np.abs(coefficients), optimize=True)
    return value, budget


def _point_inner(left, left_budget, C, right, right_budget, weights):
    value, budget = _integrate(left[:, :, None], C, right[:, :, None], weights,
        left_budget[:, :, None], right_budget[:, :, None])
    return value[0, 0], budget[0, 0]


def _elastic_budget(domain, e0, e1):
    inverse = np.asarray(np.linalg.inv(ANGULAR_BARYCENTRIC), LD)
    angular = domain.barycentric@inverse
    angular_budget = _gamma(4)*(np.abs(domain.barycentric)@np.abs(inverse))
    absolute = np.abs(angular)@np.abs(e0)+(np.abs(angular)@np.abs(e1))/domain.radial_scale[:, None]
    return _gamma(16)*absolute+angular_budget@np.abs(e0)+(angular_budget@np.abs(e1))/domain.radial_scale[:, None]


def clipped_bulk_projected_blocks(op, added_basis, parent_basis=None):
    """Contract declared operative coordinates at points before any Gram.

    Weak directions retain their own projection and reduction budgets.  A
    source surface W is a declared arithmetic basis, never a physical release
    of its null complement or a proof of exact bulk integration accuracy.
    """
    validate_clipped_bulk_operator(op)
    added = np.asarray(added_basis)
    parent = np.eye(18) if parent_basis is None else np.asarray(parent_basis)
    for name, basis in (("Added", added), ("Parent", parent)):
        if basis.ndim != 2 or basis.shape[0] != 18 or not 1 <= basis.shape[1] <= 18:
            raise ClippedBulkRefusal(f"{name} declared basis requires 18 by 1..18 coordinates")
    added = _array(added, added.shape, "Added declared basis")
    parent = _array(parent, parent.shape, "Parent declared basis")
    B, Bb = _project(op.parent_strain_operator, parent)
    G, Gb = _project(op.enrichment_strain_operator, added)
    T, Tb = _project(op.parent_trace, parent)
    E, Eb = _project(op.enrichment_trace, added)
    C, weights = np.asarray(op.elasticity_pa, LD), op.domain.reference_volume_weights_m3
    density = LD(op.domain.section.mass_kg)/LD(op.domain.section.volume_m3)
    values = {}
    for name, left, lb, middle, right, rb, factor in (
        ("stiffness_cross", B, Bb, C, G, Gb, LD(1)),
        ("stiffness_enriched", G, Gb, C, G, Gb, LD(1)),
        ("mass_cross", T, Tb, np.eye(3, dtype=LD), E, Eb, density),
        ("mass_enriched", E, Eb, np.eye(3, dtype=LD), E, Eb, density)):
        value, budget = _integrate(left, middle, right, weights, lb, rb)
        values[name] = _immutable(factor*value, LD)
        values[name+"_budget"] = _immutable(factor*budget, LD)
    elastic_budget = _elastic_budget(op.domain, np.asarray(op.memory0, LD), np.asarray(op.memory1, LD))
    value, budget = _integrate(G, C, op.point_elastic_strain[:, :, None], weights,
                               Gb, elastic_budget[:, :, None])
    values["enrichment_force_n"] = _immutable(value[:, 0], LD)
    values["enrichment_force_n_budget"] = _immutable(budget[:, 0], LD)
    return dict(values, projection_before_Gram=True, physical_contraction_certified=False)


def clipped_bulk_contractions(op, parent_increment, added_increment, *,
        mean_e0=None, mean_e1=None, mean_z_m=None):
    """Declared virtual work, signed cross/self parts kept on their own scale.

    No whole-operator energy subtraction, funding, motion solve, Maxwell
    update or accepted history is performed.  Parent work is the unchanged
    selected full cohort's analytic old field, not the whole planet ledger.
    """
    validate_clipped_bulk_operator(op)
    q, a = (_array(v, (18,), "Declared Cartesian metre increment") for v in (parent_increment, added_increment))
    e0 = _array(op.memory0 if mean_e0 is None else mean_e0, (3, 6), "Declared complete mean memory0")
    e1 = _array(op.memory1 if mean_e1 is None else mean_e1, (3, 6), "Declared complete mean memory1")
    z = _array(np.zeros(18) if mean_z_m is None else mean_z_m, (18,), "Declared added elastic memory")
    _, _, old = _parent_blocks(op.domain.section, op.elasticity_pa, op.basis_xyz,
                                np.asarray(e0, float), np.asarray(e1, float))
    # The inherited parent closure remains float64.  Its reduction bound is
    # attached only to parent work and cannot cover any small added term.
    oldbudget = _parent_force_operand_budget(op.domain.section, op.elasticity_pa, op.basis_xyz, e0, e1)
    Bq, Bqb = _project(op.parent_strain_operator, q[:, None])
    Ga, Gab = _project(op.enrichment_strain_operator, a[:, None])
    Gz, Gzb = _project(op.enrichment_strain_operator, z[:, None])
    Tq, Tqb = _project(op.parent_trace, q[:, None])
    Ea, Eab = _project(op.enrichment_trace, a[:, None])
    Bq, Bqb, Ga, Gab, Gz, Gzb, Tq, Tqb, Ea, Eab = (
        v[:, :, 0] for v in (Bq, Bqb, Ga, Gab, Gz, Gzb, Tq, Tqb, Ea, Eab))
    elastic, eb = _elastic(op.domain, e0, e1), _elastic_budget(op.domain, e0, e1)
    C, weights = np.asarray(op.elasticity_pa, LD), op.domain.reference_volume_weights_m3
    density = LD(op.domain.section.mass_kg)/LD(op.domain.section.volume_m3)
    terms = dict(parent_work_j=_dot(q, np.asarray(old, LD), oldbudget),
        cross_parent_work_j=_point_inner(Bq, Bqb, C, Gz, Gzb, weights),
        cross_added_work_j=_point_inner(Ga, Gab, C, elastic, eb, weights),
        self_work_j=_point_inner(Ga, Gab, C, Gz, Gzb, weights),
        stiffness_cross_bilinear_j=_point_inner(Bq, Bqb, C, Ga, Gab, weights),
        stiffness_self_bilinear_j=_point_inner(Ga, Gab, C, Ga, Gab, weights),
        mass_cross_bilinear_kg_m2=tuple(density*v for v in _point_inner(Tq, Tqb, np.eye(3, dtype=LD), Ea, Eab, weights)),
        mass_self_bilinear_kg_m2=tuple(density*v for v in _point_inner(Ea, Eab, np.eye(3, dtype=LD), Ea, Eab, weights)))
    added_parent_force, added_parent_force_budget = _integrate(op.parent_strain_operator, C,
        Gz[:, :, None], weights, right_budget=Gzb[:, :, None])
    total_added_force, total_added_force_budget = _integrate(op.enrichment_strain_operator, C,
        (elastic+Gz)[:, :, None], weights, right_budget=(eb+Gzb)[:, :, None])
    return dict(scope="declared_virtual_probes_no_source_motion_or_funding",
        terms={k:dict(value=float(v), absolute_roundoff_budget=float(b+LD(np.finfo(float).eps)*abs(v))) for k, (v, b) in terms.items()},
        enrichment_force_n=_plain(total_added_force[:, 0]),
        enrichment_force_budget_n=_plain(total_added_force_budget[:, 0]+LD(np.finfo(float).eps)*np.abs(total_added_force[:, 0])),
        added_parent_force_n=_plain(added_parent_force[:, 0]),
        added_parent_force_budget_n=_plain(added_parent_force_budget[:, 0]+LD(np.finfo(float).eps)*np.abs(added_parent_force[:, 0])),
        projection_before_Gram=True,
        physical_contraction_certified=False, available_for_creation_funding=False)


def clipped_bulk_operator_to_dict(op, include_points=True):
    validate_clipped_bulk_operator(op)
    if type(include_points) is not bool:
        raise ClippedBulkRefusal("Point archive option must be boolean")
    body = _plain(dict(_identity(op), domain_geometry_source=op.domain.geometry_source,
        domain=clipped_domain_to_dict(op.domain),
        elasticity_pa=op.elasticity_pa, basis_xyz=op.basis_xyz,
        memory0=op.memory0, memory1=op.memory1,
        density_kg_m3=LD(op.domain.section.mass_kg)/LD(op.domain.section.volume_m3),
        seed_center_world_m=op.domain.seed_center_world_m, seed_radius_m=op.domain.seed_radius_m,
        radial_order=op.domain.radial_order, angular_order=op.domain.angular_order,
        ray_order=op.domain.ray_order, point_count=op.point_count,
        blocks={name:getattr(op, name) for name in _BLOCK_NAMES},
        **{name:getattr(op, name) for name in _BLOCK_NAMES},
        roundoff_component_bounds={name:getattr(op, name+"_budget") for name in _BLOCK_NAMES},
        reduction_budgets={name:getattr(op, name+"_budget") for name in _BLOCK_NAMES},
        parent_blocks=dict(stiffness=op.parent_stiffness, mass=op.parent_mass, force_n=op.parent_force_n),
        operator_binding_sha256=op.binding_sha256, points_included=include_points))
    if include_points:
        points = _plain(dict(barycentric=op.domain.barycentric,
            radial_scale=op.domain.radial_scale, side=op.domain.side,
            centered_birth_offsets_m=op.domain.centered_birth_offsets_m,
            reference_volume_weights_m3=op.domain.reference_volume_weights_m3,
            birth_volume_weights_m3=op.domain.birth_volume_weights_m3,
            **{name:getattr(op, name) for name in ("bubble", "bubble_gradient_reference_m_inverse",
                "parent_trace", "enrichment_trace", "parent_strain_operator", "enrichment_strain_operator",
                "point_elastic_strain", "point_stress_pa")}))
        body.update(points)
        body["point_archive_arithmetic"] = "float64_rounding_of_bound_live_longdouble_fields"
    return dict(body, binding_sha256=_seal(body))


def _agreement(lo, hi, lobudget, hibudget, tolerance):
    delta = np.abs(hi-lo)
    allowance = tolerance*np.maximum(np.abs(lo), np.abs(hi))+lobudget+hibudget
    good = delta <= allowance
    report = dict(agreed=bool(np.all(good)), component_count=int(good.size),
        failed_component_count=int(np.count_nonzero(~good)),
        maximum_absolute_difference=float(np.max(delta, initial=0)),
        maximum_allowance=float(np.max(allowance, initial=0)),
        own_component_relative_scale=True, positive_scale_floor_used=False)
    return good, report


def compare_clipped_bulk_orders(sequence, relative_tolerance=2e-6, probes=(), *,
        added_basis=None, parent_basis=None):
    """Compare each successive declared order, component and explicit probe.

    Agreement is evidence at these finite orders, not a rigorous bound on
    integration error or arbitrary weak-mode contractions.  All four A/P
    blocks and every component retain their own scale and signed values.
    """
    sequence, probes = tuple(sequence), tuple(probes)
    if len(sequence) < 2:
        raise ClippedBulkRefusal("At least two declared finite orders are required")
    if isinstance(relative_tolerance, (bool, np.bool_)):
        raise ClippedBulkRefusal("Finite-order relative tolerance must be positive and finite")
    tolerance = LD(relative_tolerance)
    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ClippedBulkRefusal("Finite-order relative tolerance must be positive and finite")
    for op in sequence:
        validate_clipped_bulk_operator(op)
    first = sequence[0]
    for op in sequence[1:]:
        if (op.domain.source_section_binding_sha256 != first.domain.source_section_binding_sha256 or
                not np.array_equal(op.domain.seed_center_world_m, first.domain.seed_center_world_m) or
                op.domain.seed_radius_m != first.domain.seed_radius_m or
                any(not np.array_equal(getattr(op, name), getattr(first, name))
                    for name in ("elasticity_pa", "basis_xyz", "memory0", "memory1"))):
            raise ClippedBulkRefusal("Finite orders must share identical original geometry and source fields")
    for lo, hi in zip(sequence, sequence[1:]):
        low_orders = (lo.domain.radial_order, lo.domain.angular_order, lo.domain.ray_order)
        high_orders = (hi.domain.radial_order, hi.domain.angular_order, hi.domain.ray_order)
        if not all(h >= l for l, h in zip(low_orders, high_orders)) or high_orders == low_orders:
            raise ClippedBulkRefusal("Successive finite orders must increase at least one declared order")
    prepared_probes = []
    for i, probe in enumerate(probes):
        if not isinstance(probe, dict) or set(probe)-{"name", "parent_increment_m", "enrichment_increment_m", "mean_e0", "mean_e1", "mean_z_m"}:
            raise ClippedBulkRefusal("Invalid declared contraction probe schema")
        if not isinstance(probe.get("name"), str) or not probe["name"]:
            raise ClippedBulkRefusal("Declared contraction probe requires a name")
        args = (probe.get("parent_increment_m"), probe.get("enrichment_increment_m"))
        options = {name:probe[name] for name in ("mean_e0", "mean_e1", "mean_z_m") if name in probe}
        prepared_probes.append((probe["name"], args, options))
    projected = None if added_basis is None else [clipped_bulk_projected_blocks(op, added_basis, parent_basis) for op in sequence]
    pairs = []
    for pair_index, (lo, hi) in enumerate(zip(sequence, sequence[1:])):
        blocks, agreed = {}, True
        for name in _BLOCK_NAMES:
            mask, report = _agreement(getattr(lo, name), getattr(hi, name),
                getattr(lo, name+"_budget"), getattr(hi, name+"_budget"), tolerance)
            if mask.ndim == 2:
                report["A_P_blocks"] = {label:dict(agreed=bool(np.all(mask[np.ix_(rows, cols)])),
                    failed_component_count=int(np.count_nonzero(~mask[np.ix_(rows, cols)])))
                    for label, rows, cols in (("AA", _A, _A), ("AP", _A, _P), ("PA", _P, _A), ("PP", _P, _P))}
            blocks[name] = report
            agreed = agreed and report["agreed"]
        projected_reports = {}
        if projected is not None:
            low, high = projected[pair_index:pair_index+2]
            for name in _BLOCK_NAMES:
                _, report = _agreement(low[name], high[name], low[name+"_budget"], high[name+"_budget"], tolerance)
                projected_reports[name] = report
                agreed = agreed and report["agreed"]
        probe_results = []
        for name, args, options in prepared_probes:
            low, high = (clipped_bulk_contractions(op, *args, **options) for op in (lo, hi))
            term_reports = {}
            for term in low["terms"]:
                l, h = low["terms"][term], high["terms"][term]
                _, report = _agreement(LD(l["value"]), LD(h["value"]), LD(l["absolute_roundoff_budget"]),
                    LD(h["absolute_roundoff_budget"]), tolerance)
                term_reports[term] = report
                agreed = agreed and report["agreed"]
            for force, budget in (("enrichment_force_n", "enrichment_force_budget_n"),
                                  ("added_parent_force_n", "added_parent_force_budget_n")):
                _, report = _agreement(np.asarray(low[force], LD), np.asarray(high[force], LD),
                    np.asarray(low[budget], LD), np.asarray(high[budget], LD), tolerance)
                term_reports[force] = report
                agreed = agreed and report["agreed"]
            probe_results.append(dict(name=name, contractions=term_reports))
        pairs.append(dict(low_orders=(lo.domain.radial_order, lo.domain.angular_order, lo.domain.ray_order),
            high_orders=(hi.domain.radial_order, hi.domain.angular_order, hi.domain.ray_order),
            finite_order_agreement=agreed, components=blocks,
            declared_projected_components=projected_reports, declared_probe_results=probe_results))
    body = _plain(dict(format="genesis-clipped-bulk-finite-order-comparison-0.1",
        scope="declared_finite_orders_own_component_and_probe_scales_no_rigorous_error_certificate",
        operator_bindings=[op.binding_sha256 for op in sequence], relative_tolerance=tolerance,
        probes=probes, declared_added_basis=added_basis, declared_parent_basis=parent_basis,
        projection_before_Gram=projected is not None,
        pair_results=pairs, finite_order_agreement=all(p["finite_order_agreement"] for p in pairs),
        geometry_resolved=True, arbitrary_contraction_certified=False,
        physical_contraction_certified=False, physical_creation_admitted=False,
        physical_release_admitted=False, available_for_creation_funding=False,
        roundoff_available_for_funding=False))
    return dict(body, binding_sha256=_seal(body))
