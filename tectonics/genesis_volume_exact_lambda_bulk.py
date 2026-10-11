"""Centered exact-radial contractions of the existing tied Genesis fields.

Only the lambda integral is algebraic here.  Angular coverage/convergence and
physical admission remain separate questions.  The operative parent strain
coefficients are the literal float64 vertex coefficients of the frozen source
operator, not a newly differentiated parent field.  All eighteen A/P columns,
the full inherited memory, and each declared mean memory are retained.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from types import MappingProxyType
from collections.abc import Mapping

import numpy as np

from .genesis_volume_clipped_bulk import (
    _parent_blocks, _parent_force_operand_budget, _precision_digest, _seal,
)
from .genesis_volume_embedded_partition import _solve3_precise
from .genesis_volume_exact_lambda import (
    prepare_exact_lambda_source, prepare_fixed_N_interval,
    integrate_polynomial, exact_polynomial_moments,
)
from .genesis_volume_material_sections import (
    CohortMaterialSection, MaterialSectionInterval,
    ConstantLambdaMaterialSection, _validate_cohort_section,
)
from .genesis_volume_operator import (
    ANGULAR_BARYCENTRIC, _immutable, conical_strain_operators,
)

LD = np.longdouble
EPS = np.finfo(LD).eps
FORMAT = "genesis-centered-exact-lambda-regular-bulk-0.1"
BLOCK_NAMES = ("stiffness_cross", "stiffness_enriched", "mass_cross",
               "mass_enriched", "enrichment_force_n")
SCOPE = "exact_radial_polynomial_computational_tied_no_physical_admission"


class ExactLambdaBulkRefusal(ValueError):
    """A source operand, declared contraction or finite arithmetic is unresolved."""


def _gamma(n):
    x = LD(n)*EPS
    if x >= LD(.01):
        raise ExactLambdaBulkRefusal("Polynomial arithmetic operation bound exhausted")
    return x/(1-x)


def _array(value, shape, name, dtype=LD):
    raw = np.asarray(value)
    if raw.dtype.kind not in "ifu" or raw.shape != shape:
        raise ExactLambdaBulkRefusal(f"{name} requires finite real shape {shape}")
    result = np.asarray(raw, dtype=dtype)
    if not np.isfinite(result).all():
        raise ExactLambdaBulkRefusal(f"{name} requires finite real shape {shape}")
    return result


def _get(raw, primary, alias=None, default=None):
    if primary in raw:
        if alias in raw and not np.array_equal(np.asarray(raw[primary]), np.asarray(raw[alias])):
            raise ExactLambdaBulkRefusal(f"Conflicting source aliases {primary}/{alias}")
        return raw[primary]
    if alias is not None and alias in raw:
        return raw[alias]
    if default is not None:
        return default
    raise ExactLambdaBulkRefusal("Missing source operand: "+primary)


def _section(raw):
    if isinstance(raw, CohortMaterialSection):
        _validate_cohort_section(raw)
        return raw
    if not isinstance(raw, dict) or set(raw) != {f.name for f in fields(CohortMaterialSection)}:
        raise ExactLambdaBulkRefusal("Original complete section descriptor is required")
    values = dict(raw)
    for name in ("triangle_m", "displacement_A_m", "slope_P_m", "normal_xyz",
                 "projection_slopes", "projection_offsets"):
        values[name] = _immutable(values[name], float)
    for name in ("q_bounds", "lambda_bounds", "lambda_breakpoints"):
        values[name] = tuple(values[name])
    values["intervals"] = tuple(MaterialSectionInterval(
        tuple(v["lambda_bounds"]), tuple(tuple(p) for p in v["edge_pairs"]), v["slice_kind"])
        for v in values["intervals"])
    values["constant_lambda_sections"] = tuple(ConstantLambdaMaterialSection(**v)
        for v in values["constant_lambda_sections"])
    result = CohortMaterialSection(**values)
    _validate_cohort_section(result)
    return result


@dataclass(frozen=True, slots=True)
class ExactLambdaBulkSource:
    geometry: object
    elasticity_pa: np.ndarray
    basis_xyz: np.ndarray
    memory0: np.ndarray
    memory1: np.ndarray
    accepted_interval_mean_e0: np.ndarray
    accepted_interval_mean_e1: np.ndarray
    literal_vertex_B0: np.ndarray
    literal_vertex_B1: np.ndarray
    angular_inverse: np.ndarray
    reference_chart_inverse: np.ndarray
    reference_chart_inverse_roundoff: np.ndarray
    parent_stiffness: np.ndarray
    parent_mass: np.ndarray
    parent_force_n: np.ndarray
    source_identity: str
    binding_sha256: str
    live_precision_binding_sha256: str

    @property
    def section(self):
        return self.geometry.section

    @property
    def physical_contraction_certified(self):
        return False

    @property
    def available_for_creation_funding(self):
        return False


def _source_arrays(source):
    return [(f.name, getattr(source, f.name)) for f in fields(source)
            if isinstance(getattr(source, f.name), np.ndarray)]+[
            ("geometry."+name, np.asarray(getattr(source.geometry, name), LD))
            for name in ("seed_center_world_m", "seed_radius_m", "birth_C_m", "birth_D_m",
                         "lambda_bounds", "reference_determinant_m3")]+[
            ("section."+name, np.asarray(getattr(source.section, name), LD))
            for name in ("triangle_m", "displacement_A_m", "slope_P_m", "normal_xyz", "offset_m")]


def _source_body(source):
    return dict(format=FORMAT, scope=SCOPE, section_binding=source.section.binding_sha256,
        source_identity=source.source_identity,
        live_precision_binding_sha256=source.live_precision_binding_sha256,
        arrays={k: np.asarray(v, float).tolist() for k, v in _source_arrays(source)},
        parent_convention="literal_frozen_float64_conical_vertex_B0_B1",
        physical_contraction_certified=False, available_for_creation_funding=False)


def validate_source(source):
    if not isinstance(source, ExactLambdaBulkSource):
        raise ExactLambdaBulkRefusal("Typed exact-lambda source is required")
    _validate_cohort_section(source.section)
    if _precision_digest(_source_arrays(source)) != source.live_precision_binding_sha256:
        raise ExactLambdaBulkRefusal("Exact native source operands differ")
    if _seal(_source_body(source)) != source.binding_sha256:
        raise ExactLambdaBulkRefusal("Source binding differs")


def prepare_source(raw):
    """Bind original source fields without restoring or advancing a planet.

    ``section`` may be the original typed section or its complete literal DTO.
    The fixed support requires ``seed_center_world_m`` and ``seed_radius_m``;
    the shorter legacy aliases are accepted only when unambiguous.
    """
    if not isinstance(raw, dict):
        raise ExactLambdaBulkRefusal("Source inputs must be a dictionary")
    section = _section(_get(raw, "section"))
    C = _array(_get(raw, "elasticity_pa", "C"), (6, 6), "Elasticity", float)
    Q = _array(_get(raw, "basis_xyz", "Q"), (3, 3), "Source proper frame", float)
    e0 = _array(_get(raw, "memory0", "e0"), (3, 6), "Complete memory0", float)
    e1 = _array(_get(raw, "memory1", "e1"), (3, 6), "Complete memory1", float)
    m0 = _array(raw.get("accepted_interval_mean_e0", e0), (3, 6), "Saved mean0", float)
    m1 = _array(raw.get("accepted_interval_mean_e1", e1), (3, 6), "Saved mean1", float)
    if not np.allclose(C, C.T, rtol=0, atol=64*np.finfo(float).eps*np.max(np.abs(C))) or np.linalg.eigvalsh(C).min() <= 0:
        raise ExactLambdaBulkRefusal("Source elasticity must be symmetric positive definite")
    center = _array(_get(raw, "seed_center_world_m", "seed_center_m"), (3,), "Seed center", float)
    radius = _get(raw, "seed_radius_m", "radius_m")
    geometry = prepare_exact_lambda_source(section, center, radius)
    # This call, including its float64 source arithmetic, is the declared
    # operative convention.  These represented vertex values are exact inputs
    # to the following private extended-precision polynomial construction.
    vb0, vb1 = conical_strain_operators(section.triangle_m, Q, np.eye(3))
    X = np.asarray(section.triangle_m, LD)
    chart = np.column_stack((X[1]-X[0], X[2]-X[0], X[0]))
    inverse = np.column_stack([_solve3_precise(chart, v) for v in np.eye(3, dtype=LD)])
    # Local inverse residual plus its own arithmetic operands.  No other field
    # or global condition/energy scale contributes to this bound.
    residual = np.abs(np.eye(3, dtype=LD)-chart@inverse)
    residual += _gamma(12)*(np.abs(chart)@np.abs(inverse))
    invbound = np.abs(inverse)@residual
    if np.linalg.norm(np.asarray(residual, float), ord=np.inf) >= .01:
        raise ExactLambdaBulkRefusal("Reference chart inverse is arithmetically unresolved")
    parent = _parent_blocks(section, C, Q, e0, e1)
    values = [geometry, C, Q, e0, e1, m0, m1, vb0, vb1,
              np.linalg.inv(ANGULAR_BARYCENTRIC), inverse, invbound, *parent,
              str(raw.get("source_identity", "")), "", ""]
    for i, v in enumerate(values):
        if isinstance(v, np.ndarray):
            values[i] = _immutable(v, v.dtype)
    source = ExactLambdaBulkSource(*values)
    object.__setattr__(source, "live_precision_binding_sha256", _precision_digest(_source_arrays(source)))
    object.__setattr__(source, "binding_sha256", _seal(_source_body(source)))
    return source


@dataclass(slots=True)
class _Poly:
    """Degree-first value and propagated absolute arithmetic radius."""
    value: np.ndarray
    error: np.ndarray


def _p(value, error=None):
    v = np.asarray(value, LD)
    return _Poly(v, np.zeros_like(v) if error is None else np.asarray(error, LD))


def _constant(value, error=None):
    return _p(np.asarray(value, LD)[None], None if error is None else np.asarray(error, LD)[None])


def _broadcast(a, trailing):
    extra = len(trailing)-a.value.ndim+1
    shape = (len(a.value),)+(1,)*extra+a.value.shape[1:]
    return _Poly(np.broadcast_to(a.value.reshape(shape), (len(a.value),)+trailing),
                 np.broadcast_to(a.error.reshape(shape), (len(a.value),)+trailing))


def _add(a, b):
    shape = np.broadcast_shapes(a.value.shape[1:], b.value.shape[1:])
    a, b = _broadcast(a, shape), _broadcast(b, shape)
    n = max(len(a.value), len(b.value))
    av, bv, ae, be = [np.zeros((n,)+shape, LD) for _ in range(4)]
    av[:len(a.value)], ae[:len(a.value)] = a.value, a.error
    bv[:len(b.value)], be[:len(b.value)] = b.value, b.error
    return _p(av+bv, ae+be+_gamma(2)*(np.abs(av)+np.abs(bv)))


def _scale(a, scalar):
    s = LD(scalar)
    return _p(a.value*s, a.error*abs(s)+_gamma(2)*np.abs(a.value*s))


def _mul(a, b):
    shape = np.broadcast_shapes(a.value.shape[1:], b.value.shape[1:])
    a, b = _broadcast(a, shape), _broadcast(b, shape)
    v = np.zeros((len(a.value)+len(b.value)-1,)+shape, LD)
    e = np.zeros_like(v)
    products = a.value[:, None]*b.value[None, :]
    errors = (a.error[:, None]*np.abs(b.value[None, :])+
              np.abs(a.value[:, None])*b.error[None, :]+
              a.error[:, None]*b.error[None, :]+_gamma(3)*np.abs(products))
    for k in range(len(v)):
        indices = np.arange(max(0, k-len(b.value)+1), min(k+1, len(a.value)))
        terms = products[indices, k-indices]
        v[k] = terms.sum(axis=0)
        e[k] = errors[indices, k-indices].sum(axis=0)+_gamma(len(indices))*np.abs(terms).sum(axis=0)
    return _p(v, e)


def _component(a, index):
    return _p(a.value[(slice(None),)+index], a.error[(slice(None),)+index])


def _sum_components(a, axis):
    return _p(a.value.sum(axis=axis), a.error.sum(axis=axis)+
              _gamma(a.value.shape[axis]+1)*np.abs(a.value).sum(axis=axis))


def _dot_polys(a, b):
    return _sum_components(_mul(a, b), -1)


def _project(poly, basis):
    basis = np.asarray(basis, LD)
    v = np.einsum("kfi,ij->kfj", poly.value, basis, optimize=False)
    operands = np.einsum("kfi,ij->kfj", np.abs(poly.value), np.abs(basis), optimize=False)
    e = np.einsum("kfi,ij->kfj", poly.error, np.abs(basis), optimize=False)
    return _p(v, e+_gamma(2*basis.shape[0]+1)*operands)


def _bilinear(left, middle, right):
    """Coefficient products BEFORE their exact paired radial moments."""
    C = np.asarray(middle, LD)
    v = np.zeros((len(left.value)+len(right.value)-1,
                  left.value.shape[-1], right.value.shape[-1]), LD)
    e = np.zeros_like(v)
    LC = np.einsum("kai,ab->kbi", left.value, C, optimize=False)
    LCown = np.einsum("kai,ab->kbi", np.abs(left.value), np.abs(C), optimize=False)
    LCe = np.einsum("kai,ab->kbi", left.error, np.abs(C), optimize=False)+_gamma(2*C.shape[0]+1)*LCown
    products = np.einsum("kai,laj->klij", LC, right.value, optimize=False)
    operands = np.einsum("kai,laj->klij", LCown, np.abs(right.value), optimize=False)
    errors = (np.einsum("kai,laj->klij", LCe, np.abs(right.value), optimize=False)+
              np.einsum("kai,laj->klij", np.abs(LC), right.error, optimize=False)+
              np.einsum("kai,laj->klij", LCe, right.error, optimize=False)+
              _gamma(2*C.shape[1]+2)*operands)
    for k in range(len(v)):
        indices = np.arange(max(0, k-len(right.value)+1), min(k+1, len(left.value)))
        terms = products[indices, k-indices]
        v[k] = terms.sum(axis=0)
        e[k] = errors[indices, k-indices].sum(axis=0)+_gamma(len(indices))*np.abs(terms).sum(axis=0)
    return _p(v, e)


def _memory(source, N, lam, e0, e1):
    inv = np.asarray(source.angular_inverse, LD)
    angular = N@inv
    ab = _gamma(8)*(np.abs(N)@np.abs(inv))
    c0, c1 = angular@e0, angular@e1
    b0 = ab@np.abs(e0)+_gamma(8)*(np.abs(angular)@np.abs(e0))
    b1 = ab@np.abs(e1)+_gamma(8)*(np.abs(angular)@np.abs(e1))
    return _add(_mul(lam, _constant(c0, b0)), _constant(c1, b1))


def _fields(source, interval):
    N = np.asarray(interval.barycentric, LD)
    lm, lh = LD(interval.lambda_mid), LD(interval.lambda_halfwidth)
    lam = _p(np.array((lm, lh), LD))
    offset = _p(interval.birth_relative_coefficients)
    # Geometric source arithmetic is separately witnessed by the interval;
    # this radius covers only subsequent coefficient arithmetic.
    radius2 = LD(source.geometry.seed_radius_m)**2
    S = _add(_constant(1), _scale(_dot_polys(offset, offset), -1/radius2))
    sphere = _mul(S, S)
    b, a = map(LD, source.section.lambda_bounds)
    t = _scale(_p(np.array((interval.relative_lambda_mid_to_original_lower, lh))), 1/(a-b))
    tbar = _scale(_p(np.array((interval.relative_lambda_mid_to_original_upper, -lh))), 1/(a-b))
    radial = _scale(_mul(t, tbar), 4)
    # The centered relative distances avoid subtracting world lambda values.
    # Geometry-operand accuracy is explicitly a separate unresolved scope.
    angular = _p(np.array((27*np.prod(N),), LD), np.array((0,), LD))
    angular.error[0] = _gamma(5)*(27*np.prod(np.abs(N)))
    w = _mul(_mul(angular, radial), sphere)
    Cb = np.asarray(source.geometry.birth_C_m, LD)
    Db = np.asarray(source.geometry.birth_D_m, LD)
    edge1 = _add(_mul(lam, _constant(Cb[1]-Cb[0])), _constant(Db[1]-Db[0]))
    edge2 = _add(_mul(lam, _constant(Cb[2]-Cb[0])), _constant(Db[2]-Db[0]))
    u = N@Cb
    ub = _gamma(8)*(np.abs(N)@np.abs(Cb))
    duS2 = _scale(_mul(S, _dot_polys(edge1, offset)), -4/radius2)
    dvS2 = _scale(_mul(S, _dot_polys(edge2, offset)), -4/radius2)
    dlS2 = _scale(_mul(S, _dot_polys(_constant(u, ub), offset)), -4/radius2)
    ag1 = 27*N[2]*(N[0]-N[1])
    ag2 = 27*N[1]*(N[0]-N[2])
    ag1b = _gamma(5)*27*abs(N[2])*(abs(N[0])+abs(N[1]))
    ag2b = _gamma(5)*27*abs(N[1])*(abs(N[0])+abs(N[2]))
    du = _add(_mul(_mul(_constant(ag1, ag1b), radial), sphere),
              _mul(_mul(angular, radial), duS2))
    dv = _add(_mul(_mul(_constant(ag2, ag2b), radial), sphere),
              _mul(_mul(angular, radial), dvS2))
    rd = _scale(_add(tbar, _scale(t, -1)), 4/(a-b))
    dl = _add(_mul(_mul(angular, radial), dlS2), _mul(_mul(angular, sphere), rd))
    inv, ib = source.reference_chart_inverse, source.reference_chart_inverse_roundoff
    radialpart = _add(_mul(lam, dl), _scale(_add(_scale(du, N[1]), _scale(dv, N[2])), -1))
    L = _add(_add(_mul(du, _constant(inv[0], ib[0])),
                  _mul(dv, _constant(inv[1], ib[1]))),
             _mul(radialpart, _constant(inv[2], ib[2])))
    B0 = np.einsum("i,iaj->aj", N, np.asarray(source.literal_vertex_B0, LD))
    B1 = np.einsum("i,iaj->aj", N, np.asarray(source.literal_vertex_B1, LD))
    b0 = _gamma(8)*np.einsum("i,iaj->aj", np.abs(N), np.abs(source.literal_vertex_B0))
    b1 = _gamma(8)*np.einsum("i,iaj->aj", np.abs(N), np.abs(source.literal_vertex_B1))
    Bhat = _add(_mul(lam, _constant(B0, b0)), _constant(B1, b1))
    tv = np.zeros((2, 3, 18), LD)
    te = np.zeros_like(tv)
    for i in range(3):
        tv[0, :, 6*i:6*i+3] = N[i]*np.eye(3, dtype=LD)
        tv[:, :, 6*i+3:6*i+6] = np.array((lm-1, lh))[:, None, None]*N[i]*np.eye(3, dtype=LD)
        te[0, :, 6*i+3:6*i+6] = _gamma(4)*(abs(lm)+1)*abs(N[i])*np.eye(3, dtype=LD)
        te[1, :, 6*i+3:6*i+6] = _gamma(2)*abs(lh*N[i])*np.eye(3, dtype=LD)
    T = _p(tv, te)
    Q = np.asarray(source.basis_xyz, LD)
    lt = _p(np.einsum("af,kfj->kaj", Q.T, T.value),
            np.einsum("af,kfj->kaj", np.abs(Q.T), T.error)+
            _gamma(8)*np.einsum("af,kfj->kaj", np.abs(Q.T), np.abs(T.value)))
    lg = _p(L.value@Q, L.error@np.abs(Q)+_gamma(8)*(np.abs(L.value)@np.abs(Q)))
    components = []
    for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)):
        term = _mul(_component(lt, (i, slice(None))), _component(lg, (j,)))
        if i != j:
            term = _add(term, _mul(_component(lt, (j, slice(None))), _component(lg, (i,))))
        components.append(term)
    extra = _p(np.stack([p.value for p in components], axis=1),
               np.stack([p.error for p in components], axis=1))
    Ghat = _add(_mul(w, Bhat), extra)
    E = _mul(w, T)
    memory = _memory(source, N, lam, np.asarray(source.memory0, LD), np.asarray(source.memory1, LD))
    return dict(lam=lam, w=w, L=L, Bhat=Bhat, Ghat_without_H=Ghat,
                T=T, E_without_H=E, memoryhat=memory)


def _integral(interval, polynomial, side_power, factor=1):
    if len(polynomial.value) > 17:
        raise ExactLambdaBulkRefusal("Declared radial polynomial exceeds degree16")
    primitive = integrate_polynomial(interval, polynomial.value, side_power=side_power)
    moments = exact_polynomial_moments(interval, len(polynomial.value)-1, side_power=side_power)
    propagated = np.tensordot(np.abs(moments), polynomial.error, axes=(0, 0))
    f = LD(factor)
    value = f*np.asarray(primitive.value, LD)
    own = abs(f)*(np.asarray(primitive.roundoff_bound, LD)+propagated)
    own += _gamma(2)*np.abs(value)
    operands = abs(f)*np.asarray(primitive.absolute_operand_scale, LD)
    return value, own, operands


def _integrate_blocks(interval, source, f, V, W):
    B, G = _project(f["Bhat"], V), _project(f["Ghat_without_H"], W)
    T, E = _project(f["T"], V), _project(f["E_without_H"], W)
    C = source.elasticity_pa
    lam2 = _mul(f["lam"], f["lam"])
    det = LD(interval.reference_determinant_m3)
    density = LD(source.section.mass_kg)/LD(source.section.volume_m3)
    data, budgets, operands = {}, {}, {}
    for name, p, side, factor in (
        ("stiffness_cross", _bilinear(B, C, G), 1, det),
        ("stiffness_enriched", _bilinear(G, C, G), 2, det),
        ("mass_cross", _mul(lam2, _bilinear(T, np.eye(3, dtype=LD), E)), 1, det*density),
        ("mass_enriched", _mul(lam2, _bilinear(E, np.eye(3, dtype=LD), E)), 2, det*density),
        ("enrichment_force_n", _bilinear(G, C, _p(f["memoryhat"].value[:, :, None], f["memoryhat"].error[:, :, None])), 1, det)):
        v, b, a = _integral(interval, p, side, factor)
        if name == "enrichment_force_n":
            v, b, a = v[:, 0], b[:, 0], a[:, 0]
        data[name], budgets[name], operands[name] = v, b, a
    return data, budgets, operands


def _probe_vectors(probe):
    if not isinstance(probe, Mapping) or not isinstance(probe.get("name"), str) or not probe["name"]:
        raise ExactLambdaBulkRefusal("Every declared probe requires a stable name")
    q = _array(_get(probe, "parent_increment_m", "q",
                   probe.get("parent_increment")), (18,), "Parent probe")
    a = _array(_get(probe, "enrichment_increment_m", "a",
                   probe.get("added_increment")), (18,), "Added probe")
    z = _array(probe.get("mean_z_m", np.zeros(18)), (18,), "Mean added memory")
    return q, a, z


def parent_probe_work(source, probe):
    """The unchanged whole-cohort parent term, recorded once outside line sums."""
    validate_source(source)
    q, _, _ = _probe_vectors(probe)
    e0 = _array(probe.get("mean_e0", source.memory0), (3, 6), "Declared mean memory0")
    e1 = _array(probe.get("mean_e1", source.memory1), (3, 6), "Declared mean memory1")
    _, _, force = _parent_blocks(source.section, source.elasticity_pa, source.basis_xyz,
                                np.asarray(e0, float), np.asarray(e1, float))
    forcebudget = _parent_force_operand_budget(source.section, source.elasticity_pa,
                                              source.basis_xyz, e0, e1)
    value = q@np.asarray(force, LD)
    budget = np.abs(q)@forcebudget+_gamma(40)*(np.abs(q)@np.abs(force))
    return dict(value=value, absolute_roundoff_budget=budget,
                scope="unchanged_whole_original_cohort_analytic_parent_work",
                physical_contraction_certified=False, available_for_creation_funding=False)


def _integrate_probe(interval, source, f, probe):
    q, a, z = _probe_vectors(probe)
    e0 = _array(probe.get("mean_e0", source.memory0), (3, 6), "Declared mean memory0")
    e1 = _array(probe.get("mean_e1", source.memory1), (3, 6), "Declared mean memory1")
    mean = _memory(source, interval.barycentric, f["lam"], e0, e1)
    mean = _p(mean.value[:, :, None], mean.error[:, :, None])
    Bq = _project(f["Bhat"], q[:, None])
    Ga, Gz = _project(f["Ghat_without_H"], a[:, None]), _project(f["Ghat_without_H"], z[:, None])
    Tq, Ea = _project(f["T"], q[:, None]), _project(f["E_without_H"], a[:, None])
    lam2 = _mul(f["lam"], f["lam"])
    det = LD(interval.reference_determinant_m3)
    density = LD(source.section.mass_kg)/LD(source.section.volume_m3)
    C = source.elasticity_pa
    definitions = (
        ("cross_parent_work_j", _bilinear(Bq, C, Gz), 1, det),
        ("cross_added_work_j", _bilinear(Ga, C, mean), 1, det),
        ("self_work_j", _bilinear(Ga, C, Gz), 2, det),
        ("stiffness_cross_bilinear_j", _bilinear(Bq, C, Ga), 1, det),
        ("stiffness_self_bilinear_j", _bilinear(Ga, C, Ga), 2, det),
        ("mass_cross_bilinear_kg_m2", _mul(lam2, _bilinear(Tq, np.eye(3, dtype=LD), Ea)), 1, det*density),
        ("mass_self_bilinear_kg_m2", _mul(lam2, _bilinear(Ea, np.eye(3, dtype=LD), Ea)), 2, det*density),
    )
    values, budgets, operands = {}, {}, {}
    for name, p, side, factor in definitions:
        v, b, op = _integral(interval, p, side, factor)
        values[name], budgets[name], operands[name] = v[0, 0], b[0, 0], op[0, 0]
    B, G = f["Bhat"], f["Ghat_without_H"]
    v, b, op = _integral(interval, _bilinear(B, C, Gz), 1, det)
    values["added_parent_force_n"], budgets["added_parent_force_n"], operands["added_parent_force_n"] = v[:, 0], b[:, 0], op[:, 0]
    v1, b1, o1 = _integral(interval, _bilinear(G, C, mean), 1, det)
    v2, b2, o2 = _integral(interval, _bilinear(G, C, Gz), 2, det)
    values["enrichment_force_n"] = v1[:, 0]+v2[:, 0]
    budgets["enrichment_force_n"] = b1[:, 0]+b2[:, 0]+_gamma(2)*(np.abs(v1[:, 0])+np.abs(v2[:, 0]))
    operands["enrichment_force_n"] = o1[:, 0]+o2[:, 0]
    return values, budgets, operands


def _basis(raw, name):
    v = np.asarray(raw)
    if v.ndim != 2 or v.shape[0] != 18 or not 1 <= v.shape[1] <= 18:
        raise ExactLambdaBulkRefusal(name+" requires 18 by 1..18 real coordinates")
    return _array(v, v.shape, name)


def _frozen_map(value):
    return MappingProxyType({k: _immutable(v, LD) if isinstance(v, np.ndarray)
                             else np.longdouble(v) for k, v in value.items()})


def _interval_body(interval):
    if interval is None:
        return None
    return dict(barycentric=np.asarray(interval.barycentric, float).tolist(),
        lambda_mid=float(interval.lambda_mid), lambda_halfwidth=float(interval.lambda_halfwidth),
        lambda_bounds=list(map(float, interval.lambda_bounds)),
        birth_relative_coefficients=np.asarray(interval.birth_relative_coefficients, float).tolist(),
        literal_phi_coefficients=np.asarray(interval.literal_phi_coefficients, float).tolist(),
        gamma_root_xi=None if interval.gamma_root_xi is None else float(interval.gamma_root_xi),
        gamma_endpoint_distance_xi=None if interval.gamma_endpoint_distance_xi is None else float(interval.gamma_endpoint_distance_xi),
        gamma_left_width_xi=None if interval.gamma_left_width_xi is None else float(interval.gamma_left_width_xi),
        gamma_right_width_xi=None if interval.gamma_right_width_xi is None else float(interval.gamma_right_width_xi),
        relative_lambda_mid_to_original_lower=float(interval.relative_lambda_mid_to_original_lower),
        relative_lambda_mid_to_original_upper=float(interval.relative_lambda_mid_to_original_upper),
        gamma_root_arithmetic_estimate=float(interval.gamma_root_arithmetic_estimate),
        constant_side=None if interval.constant_side is None else float(interval.constant_side),
        reference_determinant_m3=float(interval.reference_determinant_m3),
        geometry_operand_accuracy_certified=False)


def _line_arrays(line):
    arrays = [("values", line.values), ("budgets", line.budgets), ("own", line.absolute_operand_scales)]
    for label, mapping in (("full", line.blocks), ("fullbudget", line.roundoff_component_bounds),
            ("projected", line.projected_blocks), ("projectedbudget", line.projected_roundoff_component_bounds),
            ("coefficient", line.coefficients), ("coefficient_bound", line.coefficient_roundoff_bounds)):
        arrays.extend((label+"."+k, np.asarray(v, LD)) for k, v in mapping.items())
    for name, probe in line.probe_contractions.items():
        for label, mapping in probe.items():
            arrays.extend(("probe."+name+"."+label+"."+k, np.asarray(v, LD)) for k, v in mapping.items())
    if line.interval is not None:
        arrays.extend(("interval."+k, np.asarray(getattr(line.interval, k), LD))
            for k in ("barycentric", "lambda_mid", "lambda_halfwidth", "lambda_bounds",
                      "birth_relative_coefficients", "literal_phi_coefficients", "reference_determinant_m3",
                      "relative_lambda_mid_to_original_lower", "relative_lambda_mid_to_original_upper",
                      "gamma_root_arithmetic_estimate"))
        arrays.extend(("interval."+k, np.asarray(getattr(line.interval, k), LD))
            for k in ("gamma_endpoint_distance_xi", "gamma_left_width_xi", "gamma_right_width_xi")
            if getattr(line.interval, k) is not None)
        arrays.append(("interval.side", np.array((line.interval.gamma_root_xi
            if line.interval.gamma_root_xi is not None else line.interval.constant_side,), LD)))
    return arrays


def _line_body(line):
    return dict(source=line.source_binding_sha256, live=line.live_precision_binding_sha256,
                names=list(line.names), interval=_interval_body(line.interval))


@dataclass(frozen=True, slots=True)
class ExactLambdaLineResult:
    interval: object
    values: np.ndarray
    budgets: np.ndarray
    names: tuple
    absolute_operand_scales: np.ndarray
    blocks: object
    roundoff_component_bounds: object
    projected_blocks: object
    projected_roundoff_component_bounds: object
    probe_contractions: object
    coefficients: object
    coefficient_roundoff_bounds: object
    source_binding_sha256: str
    binding_sha256: str
    live_precision_binding_sha256: str

    @property
    def flattened_values(self):
        return self.values

    @property
    def flattened_roundoff_bounds(self):
        return self.budgets

    @property
    def component_names(self):
        return self.names

    @property
    def physical_contraction_certified(self):
        return False

    @property
    def available_for_creation_funding(self):
        return False


def _evaluate_line(source, N, W, V, probes, *, witness, prepared_names=None):
    """Integrate the complete clipped line, with H cross and H² self paired.

    Values are densities per original ``dN1 dN2``.  ``values/budgets/names``
    are parallel immutable flattened arrays for a bounded angular harness.
    Each budget is finite arithmetic from that component's actual operands;
    it is not an angular error estimate or a physical contraction certificate.
    Empty lines have the same component schema and exact zero contributions.
    """
    interval = prepare_fixed_N_interval(source.geometry, N)
    f = None if interval is None else _fields(source, interval)
    shapes = {k: ((18,) if k == "enrichment_force_n" else (18, 18)) for k in BLOCK_NAMES}
    if f is None:
        blocks = {k: np.zeros(s, LD) for k, s in shapes.items()}
        budgets, operands = ({k: np.zeros_like(v) for k, v in blocks.items()} for _ in range(2))
    else:
        blocks, budgets, operands = _integrate_blocks(interval, source, f, np.eye(18, dtype=LD), np.eye(18, dtype=LD))
    projected, projected_budgets, projected_operands = {}, {}, {}
    if W is not None:
        if f is None:
            projected = {k: np.zeros((W.shape[1],) if k == "enrichment_force_n" else
                         ((W.shape[1], W.shape[1]) if k.endswith("enriched") else
                          (V.shape[1], W.shape[1])), LD) for k in BLOCK_NAMES}
            projected_budgets, projected_operands = ({k: np.zeros_like(v) for k, v in projected.items()} for _ in range(2))
        else:
            projected, projected_budgets, projected_operands = _integrate_blocks(interval, source, f, V, W)
    pv, pb, po = {}, {}, {}
    for probe in probes:
        name = probe["name"]
        if f is None:
            names = ("cross_parent_work_j", "cross_added_work_j", "self_work_j", "stiffness_cross_bilinear_j",
                     "stiffness_self_bilinear_j", "mass_cross_bilinear_kg_m2", "mass_self_bilinear_kg_m2")
            pv[name] = {k: LD(0) for k in names}
            pv[name].update(added_parent_force_n=np.zeros(18, LD), enrichment_force_n=np.zeros(18, LD))
            pb[name], po[name] = ({k: np.zeros_like(v) for k, v in pv[name].items()} for _ in range(2))
        else:
            pv[name], pb[name], po[name] = _integrate_probe(interval, source, f, probe)
    flatv, flatb, flato, names = [], [], [], []
    for prefix, values, bound, own in [("full", blocks, budgets, operands),
            ("projected", projected, projected_budgets, projected_operands)]+[
            ("probe."+p["name"], pv[p["name"]], pb[p["name"]], po[p["name"]]) for p in probes]:
        for key, value in values.items():
            a = np.asarray(value, LD)
            if prepared_names is None:
                for index in np.ndindex(a.shape):
                    suffix = "["+",".join(map(str, index))+"]" if index else ""
                    names.append(prefix+"."+key+suffix)
            flatv.append(a.reshape(-1)); flatb.append(np.asarray(bound[key], LD).reshape(-1))
            flato.append(np.asarray(own[key], LD).reshape(-1))
    values, bounds, own = (np.concatenate(v) for v in (flatv, flatb, flato))
    if not all(np.isfinite(v).all() for v in (values, bounds, own)) or np.any(bounds < 0):
        raise ExactLambdaBulkRefusal("Exact radial contraction or bound overflowed")
    if not witness:
        return NativeLineValues(_immutable(values, LD), _immutable(bounds, LD), prepared_names, interval)
    names = tuple(names) if prepared_names is None else prepared_names
    coeffs = {} if f is None else {k: p.value for k, p in f.items()}
    cbounds = {} if f is None else {k: p.error for k, p in f.items()}
    line = ExactLambdaLineResult(interval, _immutable(values, LD), _immutable(bounds, LD), tuple(names),
        _immutable(own, LD), _frozen_map(blocks), _frozen_map(budgets),
        _frozen_map(projected), _frozen_map(projected_budgets),
        MappingProxyType({name: MappingProxyType(dict(values=_frozen_map(pv[name]),
            roundoff_component_bounds=_frozen_map(pb[name]), absolute_operand_scales=_frozen_map(po[name]))) for name in pv}),
        _frozen_map(coeffs), _frozen_map(cbounds), source.binding_sha256, "", "")
    object.__setattr__(line, "live_precision_binding_sha256", _precision_digest(_line_arrays(line)))
    object.__setattr__(line, "binding_sha256", _seal(_line_body(line)))
    return line


def _prepare_declared(source, added_basis, parent_basis, probes):
    validate_source(source)
    W = None if added_basis is None else _immutable(_basis(added_basis, "Added basis"), LD)
    V = _immutable(np.eye(18, dtype=LD) if parent_basis is None else _basis(parent_basis, "Parent basis"), LD)
    if len(probes) > 16 or len({p.get("name") for p in probes}) != len(probes):
        raise ExactLambdaBulkRefusal("At most16 uniquely named declared probes are allowed")
    prepared = []
    for probe in probes:
        q, a, z = _probe_vectors(probe)
        e0 = _array(probe.get("mean_e0", source.memory0), (3, 6), "Declared mean memory0")
        e1 = _array(probe.get("mean_e1", source.memory1), (3, 6), "Declared mean memory1")
        prepared.append(MappingProxyType(dict(name=probe["name"], parent_increment_m=_immutable(q, LD),
            enrichment_increment_m=_immutable(a, LD), mean_z_m=_immutable(z, LD),
            mean_e0=_immutable(e0, LD), mean_e1=_immutable(e1, LD))))
    return W, V, tuple(prepared)


def integrate_line(source, N, *, added_basis=None, parent_basis=None, probes=()):
    """Return a sealed compact witness through the prepared arithmetic path."""
    W, V, prepared = _prepare_declared(source, added_basis, parent_basis, probes)
    return _evaluate_line(source, N, W, V, prepared, witness=True)


@dataclass(frozen=True, slots=True)
class NativeLineValues:
    """Unsealed transient native contribution; no point table or archive."""
    values: np.ndarray
    budgets: np.ndarray
    names: tuple
    interval: object


@dataclass(frozen=True, slots=True)
class PreparedLineEvaluator:
    source: ExactLambdaBulkSource
    added_basis: object
    parent_basis: np.ndarray
    probes: tuple
    names: tuple

    def evaluate(self, N):
        return _evaluate_line(self.source, N, self.added_basis, self.parent_basis,
                              self.probes, witness=False, prepared_names=self.names)

    def __call__(self, N):
        return self.evaluate(N)

    def witness(self, N):
        return _evaluate_line(self.source, N, self.added_basis, self.parent_basis,
                              self.probes, witness=True, prepared_names=self.names)


def prepare_line_evaluator(source, *, added_basis=None, parent_basis=None, probes=()):
    """Validate/freeze once; stream native line integrals with identical math.

    Per-line hashes, public DTO conversion and component-name construction are
    omitted on the transient route.  Few selected lines can still be sealed
    with ``evaluator.witness(N)``.  No line/point cache is retained.
    """
    W, V, prepared = _prepare_declared(source, added_basis, parent_basis, probes)
    # An exterior N is not guaranteed to miss every support.  Build the schema
    # from shapes instead of sampling or evaluating source fields here.
    names = []
    shapes = [("full", k, (18,) if k == "enrichment_force_n" else (18, 18)) for k in BLOCK_NAMES]
    if W is not None:
        shapes += [("projected", k, (W.shape[1],) if k == "enrichment_force_n" else
            ((W.shape[1], W.shape[1]) if k.endswith("enriched") else (V.shape[1], W.shape[1]))) for k in BLOCK_NAMES]
    terms = ("cross_parent_work_j", "cross_added_work_j", "self_work_j", "stiffness_cross_bilinear_j",
             "stiffness_self_bilinear_j", "mass_cross_bilinear_kg_m2", "mass_self_bilinear_kg_m2")
    for probe in prepared:
        prefix = "probe."+probe["name"]
        shapes += [(prefix, name, ()) for name in terms]
        shapes += [(prefix, name, (18,)) for name in ("added_parent_force_n", "enrichment_force_n")]
    for prefix, name, shape in shapes:
        for index in np.ndindex(shape):
            suffix = "["+",".join(map(str, index))+"]" if index else ""
            names.append(prefix+"."+name+suffix)
    return PreparedLineEvaluator(source, W, V, prepared, tuple(names))


def validate_line(line):
    if not isinstance(line, ExactLambdaLineResult):
        raise ExactLambdaBulkRefusal("Typed exact radial line result is required")
    if _precision_digest(_line_arrays(line)) != line.live_precision_binding_sha256:
        raise ExactLambdaBulkRefusal("Exact native line operands differ")
    if _seal(_line_body(line)) != line.binding_sha256:
        raise ExactLambdaBulkRefusal("Line binding differs")


def source_to_dict(source):
    validate_source(source)
    body = _source_body(source)
    body["source_binding_sha256"] = source.binding_sha256
    body["archive_sha256"] = _seal(body)
    return body


def line_to_dict(line, *, include_coefficients=True, include_native_precision=True):
    """Literal public float64 witness; private precision retains its own seal."""
    validate_line(line)
    def plain_map(raw):
        return {k: np.asarray(v, float).tolist() for k, v in raw.items()}
    body = dict(format=FORMAT, scope=SCOPE, source_binding_sha256=line.source_binding_sha256,
        live_precision_binding_sha256=line.live_precision_binding_sha256,
        line_binding_sha256=line.binding_sha256, component_names=list(line.names),
        values=np.asarray(line.values, float).tolist(),
        roundoff_component_bounds=np.asarray(line.budgets, float).tolist(),
        absolute_operand_scales=np.asarray(line.absolute_operand_scales, float).tolist(),
        blocks=plain_map(line.blocks), projected_blocks=plain_map(line.projected_blocks),
        block_roundoff_component_bounds=plain_map(line.roundoff_component_bounds),
        projected_roundoff_component_bounds=plain_map(line.projected_roundoff_component_bounds),
        probe_contractions={k: {a: plain_map(b) for a, b in v.items()} for k, v in line.probe_contractions.items()},
        physical_contraction_certified=False, available_for_creation_funding=False,
        physical_creation_admitted=False, physical_release_admitted=False,
        geometry_operand_accuracy_certified=False, angular_accuracy_certified=False,
        new_material_count=0, accepted_time_advance_myr=0., funding_debit_j=0.)
    body["interval"] = _interval_body(line.interval)
    if include_coefficients:
        body["coefficients"] = plain_map(line.coefficients)
        body["coefficient_roundoff_bounds"] = plain_map(line.coefficient_roundoff_bounds)
    if include_native_precision:
        # Compact selected-line witnesses can be independently checked without
        # silently round-tripping centered lambda or coefficients via float64.
        # These are exact logical ratios, never opaque LD storage/padding.
        body["native_exact_binary_ratio"] = {
            name: dict(shape=list(np.asarray(value).shape), numerator_denominator=[
                [str(n), str(d)] for n, d in (LD(x).as_integer_ratio() for x in np.asarray(value, LD).reshape(-1))])
            for name, value in _line_arrays(line)
            if include_coefficients or not name.startswith(("coefficient.", "coefficient_bound."))}
    body["archive_sha256"] = _seal(body)
    return body
