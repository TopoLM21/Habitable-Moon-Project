"""Exact rational, producer-free coefficient proofs for represented material lines.

The source's declared binary64 vertex convention is retained.  All subsequent
coefficient algebra is exact.  A canonical error witness is the exact distance
between that algebra and the archived native binary coefficient, not a multiple
of the final field's magnitude.  This proves only the supplied source/N and
represented midpoint/halfwidth; it does not certify the sphere or Gamma roots,
angular quadrature, physical activation, or an error bound for future lines.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math

import numpy as np


FIELDS = ("lam", "w", "L", "Bhat", "Ghat_without_H", "T", "E_without_H", "memoryhat")
SCOPE = "exact_source_coefficient_residual_at_literal_represented_line"


class CoefficientProofRefusal(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise CoefficientProofRefusal(message)


def fraction(value):
    _require(not isinstance(value, (bool, np.bool_)), "Boolean rational operand")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value))
    if isinstance(value, (float, np.floating)):
        _require(np.isfinite(value), "Nonfinite rational operand")
        return Fraction(*value.as_integer_ratio())
    _require(isinstance(value, str) and len(value) <= 4096, "Unsupported rational operand")
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise CoefficientProofRefusal("Invalid rational operand") from error


def rational_array(value, shape=None):
    raw = np.asarray(value, dtype=object)
    _require(shape is None or raw.shape == shape, "Rational operand shape differs")
    return np.asarray([fraction(x) for x in raw.flat], dtype=object).reshape(raw.shape)


def decode_native_array(witness, label):
    records = witness.get("native_exact_binary_ratio")
    _require(type(records) is dict and label in records, "Missing native ratio: " + label)
    record = records[label]
    _require(type(record) is dict and type(record.get("shape")) is list, "Native ratio shape absent")
    shape = record["shape"]
    _require(len(shape) <= 4 and all(type(n) is int and 0 <= n <= 10000 for n in shape),
             "Native ratio shape is unbounded")
    pairs = record.get("numerator_denominator")
    _require(type(pairs) is list and len(pairs) == math.prod(shape), "Native ratio count differs")
    values = []
    for pair in pairs:
        _require(type(pair) is list and len(pair) == 2 and
                 all(type(x) is str and len(x) <= 2048 for x in pair), "Native ratio pair differs")
        try:
            n, d = map(int, pair)
        except ValueError as error:
            raise CoefficientProofRefusal("Native ratio is not integer") from error
        _require(d > 0, "Native ratio denominator is nonpositive")
        values.append(Fraction(n, d))
    return np.asarray(values, dtype=object).reshape(shape)


def encode_rational_array(value):
    value = rational_array(value)
    return dict(shape=list(value.shape), numerator_denominator=[
        [str(x.numerator), str(x.denominator)] for x in value.flat])


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _zeros(shape):
    result = np.empty(shape, dtype=object)
    result.fill(Fraction())
    return result


def _inverse3(a):
    a = rational_array(a, (3, 3))
    determinant = (a[0, 0]*(a[1, 1]*a[2, 2]-a[1, 2]*a[2, 1])
                   -a[0, 1]*(a[1, 0]*a[2, 2]-a[1, 2]*a[2, 0])
                   +a[0, 2]*(a[1, 0]*a[2, 1]-a[1, 1]*a[2, 0]))
    _require(determinant != 0, "Singular source chart")
    cofactors = _zeros((3, 3))
    for i in range(3):
        for j in range(3):
            rows = [k for k in range(3) if k != i]
            cols = [k for k in range(3) if k != j]
            cofactors[i, j] = ((-1)**(i+j)*(a[rows[0], cols[0]]*a[rows[1], cols[1]]
                              -a[rows[0], cols[1]]*a[rows[1], cols[0]]))
    return cofactors.T/determinant


def _literal_vertices(X, Q):
    """Independently execute the original binary64 plane B0/B1 convention."""
    X, Q = np.asarray(X, float), np.asarray(Q, float)
    local = X@Q
    h = float(np.mean(local[:, 2]))
    _require(h > 0 and np.allclose(local[:, 2], h, rtol=2e-12, atol=2e-9),
             "Source plane differs")
    design = np.column_stack((np.ones(3), local[:, :2]))
    _require(np.linalg.det(design) > 0, "Source triangle orientation differs")
    gradients = np.linalg.inv(design)[1:]
    B0, B1 = np.zeros((3, 6, 18)), np.zeros((3, 6, 18))
    for point in range(3):
        for vertex in range(3):
            gradient = np.r_[gradients[:, vertex], -local[point, :2]@gradients[:, vertex]/h]
            normal = np.array((0., 0., float(point == vertex)/h))
            for axis in range(3):
                vector = Q[axis]
                def sym(g):
                    return np.array((vector[0]*g[0], vector[1]*g[1], vector[2]*g[2],
                        vector[0]*g[1]+vector[1]*g[0], vector[0]*g[2]+vector[2]*g[0],
                        vector[1]*g[2]+vector[2]*g[1]))
                column = sym(gradient)
                B1[point, :, 6*vertex+axis] = column
                B0[point, :, 6*vertex+3+axis] = column+sym(normal)
                B1[point, :, 6*vertex+3+axis] = -column
    return rational_array(B0), rational_array(B1)


@dataclass(frozen=True)
class _Poly:
    value: np.ndarray
    operands: np.ndarray


def _p(value, operands=None):
    v = rational_array(value)
    return _Poly(v, abs(v) if operands is None else rational_array(operands, v.shape))


def _constant(value, operands=None):
    return _p(np.asarray(value, object)[None], None if operands is None else np.asarray(operands, object)[None])


def _broadcast(p, trailing):
    extra = len(trailing)-p.value.ndim+1
    shape = (len(p.value),)+(1,)*extra+p.value.shape[1:]
    return _Poly(np.broadcast_to(p.value.reshape(shape), (len(p.value),)+trailing),
                 np.broadcast_to(p.operands.reshape(shape), (len(p.value),)+trailing))


def _add(a, b):
    shape = np.broadcast_shapes(a.value.shape[1:], b.value.shape[1:])
    a, b = _broadcast(a, shape), _broadcast(b, shape)
    n = max(len(a.value), len(b.value))
    av, bv, ao, bo = [_zeros((n,)+shape) for _ in range(4)]
    av[:len(a.value)], ao[:len(a.value)] = a.value, a.operands
    bv[:len(b.value)], bo[:len(b.value)] = b.value, b.operands
    return _Poly(av+bv, ao+bo)


def _scale(a, scalar):
    scalar = fraction(scalar)
    return _Poly(a.value*scalar, a.operands*abs(scalar))


def _mul(a, b):
    shape = np.broadcast_shapes(a.value.shape[1:], b.value.shape[1:])
    a, b = _broadcast(a, shape), _broadcast(b, shape)
    value, operands = [_zeros((len(a.value)+len(b.value)-1,)+shape) for _ in range(2)]
    for i in range(len(a.value)):
        for j in range(len(b.value)):
            value[i+j] += a.value[i]*b.value[j]
            operands[i+j] += a.operands[i]*b.operands[j]
    return _Poly(value, operands)


def _component(a, index):
    return _Poly(a.value[(slice(None),)+index], a.operands[(slice(None),)+index])


def _dot(a, b):
    product = _mul(a, b)
    return _Poly(product.value.sum(axis=-1), product.operands.sum(axis=-1))


def source_operand_families(raw, N, lambda_mid, lambda_halfwidth):
    """Exact coefficient arrays and unsigned source-construction majorants.

    Majorants describe the operands of the independent algebra.  They are not
    themselves rounding-error bounds, and no multiplier or positive floor is
    applied to them.  Exact residuals supply the selected-line proof instead.
    """
    _require(type(raw) is dict and type(raw.get("section")) is dict, "Missing source section")
    section = raw["section"]
    N = rational_array(N, (3,))
    # The original API can retain either binary64 or extended-precision N.
    # Use its binary64 admissibility envelope without modifying any coordinate;
    # authentication of the particular source classification belongs to the
    # geometry lane.  Native N equality is separately required by the audit.
    _require(all(n >= 0 for n in N) and abs(sum(N)-1) <= Fraction(1, 2**46)*sum(abs(n) for n in N),
             "Source N differs")
    lm, lh = fraction(lambda_mid), fraction(lambda_halfwidth)
    _require(lm > 0 and lh > 0, "Represented interval differs")
    X, A, P = [np.asarray(section[k], float) for k in
               ("triangle_m", "displacement_A_m", "slope_P_m")]
    _require(all(v.shape == (3, 3) and np.isfinite(v).all() for v in (X, A, P)), "Source geometry shape differs")
    Cb, Db = rational_array(X+P), rational_array(A-P)
    Xr = rational_array(X)
    Q = rational_array(raw.get("basis_xyz", raw.get("Q")), (3, 3))
    center = rational_array(raw.get("seed_center_world_m", raw.get("seed_center_m")), (3,))
    radius = fraction(raw.get("seed_radius_m", raw.get("radius_m")))
    _require(radius > 0, "Source radius is nonpositive")
    lower, upper = map(fraction, section["lambda_bounds"])
    _require(0 < lower < upper and lower <= lm <= upper, "Source radial bounds differ")
    lam = _p((lm, lh))
    u, d = N@Cb, N@Db
    offset = _p(np.stack((lm*u+d-center, lh*u)),
                np.stack((abs(lm)*(abs(N)@abs(Cb))+abs(N)@abs(Db)+abs(center),
                          abs(lh)*(abs(N)@abs(Cb)))))
    radius2 = radius*radius
    S = _add(_constant(1), _scale(_dot(offset, offset), -1/radius2))
    sphere = _mul(S, S)
    t, tbar = _scale(_p((lm-lower, lh)), 1/(upper-lower)), _scale(_p((upper-lm, -lh)), 1/(upper-lower))
    radial = _scale(_mul(t, tbar), 4)
    angular = _constant(27*N[0]*N[1]*N[2])
    w = _mul(_mul(angular, radial), sphere)
    def edge(i):
        return _add(_mul(lam, _constant(Cb[i]-Cb[0], abs(Cb[i])+abs(Cb[0]))),
                    _constant(Db[i]-Db[0], abs(Db[i])+abs(Db[0])))
    duS2 = _scale(_mul(S, _dot(edge(1), offset)), -4/radius2)
    dvS2 = _scale(_mul(S, _dot(edge(2), offset)), -4/radius2)
    dlS2 = _scale(_mul(S, _dot(_constant(u, abs(N)@abs(Cb)), offset)), -4/radius2)
    ag1, ag2 = 27*N[2]*(N[0]-N[1]), 27*N[1]*(N[0]-N[2])
    ag1o, ag2o = 27*abs(N[2])*(abs(N[0])+abs(N[1])), 27*abs(N[1])*(abs(N[0])+abs(N[2]))
    du = _add(_mul(_mul(_constant(ag1, ag1o), radial), sphere), _mul(_mul(angular, radial), duS2))
    dv = _add(_mul(_mul(_constant(ag2, ag2o), radial), sphere), _mul(_mul(angular, radial), dvS2))
    rd = _scale(_add(tbar, _scale(t, -1)), 4/(upper-lower))
    dl = _add(_mul(_mul(angular, radial), dlS2), _mul(_mul(angular, sphere), rd))
    chart = np.column_stack((Xr[1]-Xr[0], Xr[2]-Xr[0], Xr[0]))
    inv = _inverse3(chart)
    radialpart = _add(_mul(lam, dl), _scale(_add(_scale(du, N[1]), _scale(dv, N[2])), -1))
    L = _add(_add(_mul(du, _constant(inv[0])), _mul(dv, _constant(inv[1]))),
             _mul(radialpart, _constant(inv[2])))
    vertices0, vertices1 = _literal_vertices(X, np.asarray(raw.get("basis_xyz", raw.get("Q")), float))
    B0 = sum((N[i]*vertices0[i] for i in range(3)), _zeros((6, 18)))
    B1 = sum((N[i]*vertices1[i] for i in range(3)), _zeros((6, 18)))
    B0o = sum((abs(N[i])*abs(vertices0[i]) for i in range(3)), _zeros((6, 18)))
    B1o = sum((abs(N[i])*abs(vertices1[i]) for i in range(3)), _zeros((6, 18)))
    Bhat = _add(_mul(lam, _constant(B0, B0o)), _constant(B1, B1o))
    tv = _zeros((2, 3, 18))
    for i in range(3):
        for j in range(3):
            tv[0, j, 6*i+j] = N[i]
            tv[0, j, 6*i+3+j] = (lm-1)*N[i]
            tv[1, j, 6*i+3+j] = lh*N[i]
    T = _p(tv)
    lt = _Poly(np.stack([Q.T@v for v in T.value]), np.stack([abs(Q.T)@v for v in T.operands]))
    lg = _Poly(L.value@Q, L.operands@abs(Q))
    components = []
    for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)):
        term = _mul(_component(lt, (i, slice(None))), _component(lg, (j,)))
        if i != j:
            term = _add(term, _mul(_component(lt, (j, slice(None))), _component(lg, (i,))))
        components.append(term)
    extra = _Poly(np.stack([p.value for p in components], axis=1),
                  np.stack([p.operands for p in components], axis=1))
    Ghat, E = _add(_mul(w, Bhat), extra), _mul(w, T)
    sample = np.array(((2/3, 1/6, 1/6), (1/6, 2/3, 1/6), (1/6, 1/6, 2/3)))
    angular_inverse = rational_array(np.linalg.inv(sample))
    interpolation = N@angular_inverse
    interpolation_o = abs(N)@abs(angular_inverse)
    e0 = rational_array(raw.get("memory0", raw.get("e0")), (3, 6))
    e1 = rational_array(raw.get("memory1", raw.get("e1")), (3, 6))
    memory = _add(_mul(lam, _constant(interpolation@e0, interpolation_o@abs(e0))),
                  _constant(interpolation@e1, interpolation_o@abs(e1)))
    polynomials = dict(lam=lam, w=w, L=L, Bhat=Bhat, Ghat_without_H=Ghat,
                       T=T, E_without_H=E, memoryhat=memory)
    return {k: p.value for k, p in polynomials.items()}, {k: p.operands for k, p in polynomials.items()}


def exact_source_coefficients(raw, N, lambda_mid, lambda_halfwidth):
    return source_operand_families(raw, N, lambda_mid, lambda_halfwidth)[0]


def proof_input_binding(raw, N, lambda_mid, lambda_halfwidth):
    return digest(dict(source=raw, N=encode_rational_array(N),
                       lambda_mid=encode_rational_array(lambda_mid),
                       lambda_halfwidth=encode_rational_array(lambda_halfwidth)))


def audit_coefficients(raw, N, witness):
    """Prove exact canonical coefficient residuals; assess saved bound sufficiency."""
    lm = decode_native_array(witness, "interval.lambda_mid").item()
    lh = decode_native_array(witness, "interval.lambda_halfwidth").item()
    nativeN = decode_native_array(witness, "interval.barycentric")
    _require(np.array_equal(rational_array(N, (3,)), nativeN), "Witness N is not the source N")
    expected, operands = source_operand_families(raw, nativeN, lm, lh)
    families, failures, count, exact_count = {}, [], 0, 0
    for key in FIELDS:
        actual = decode_native_array(witness, "coefficient."+key)
        saved = decode_native_array(witness, "coefficient_bound."+key)
        _require(actual.shape == expected[key].shape == saved.shape, "Coefficient shape differs: "+key)
        _require(all(x >= 0 for x in saved.flat), "Negative coefficient budget: "+key)
        residual = abs(actual-expected[key])
        passed = residual <= saved
        indices = [list(index) for index in np.ndindex(actual.shape) if not passed[index]]
        failures += [{"family": key, "index": index} for index in indices]
        count += actual.size
        exact_count += sum(int(x == 0) for x in residual.flat)
        families[key] = dict(shape=list(actual.shape), coefficient_count=int(actual.size),
            exact_residual=encode_rational_array(residual),
            exact_source_coefficients=encode_rational_array(expected[key]),
            source_construction_operand_majorants=encode_rational_array(operands[key]),
            saved_budget_sufficient=bool(passed.all()), insufficient_indices=indices)
    body = dict(format="genesis-exact-source-coefficient-proof-0.1", scope=SCOPE,
        proof_input_sha256=proof_input_binding(raw, nativeN, lm, lh),
        native_coefficient_witness_sha256=digest({key: value for key, value in
            witness["native_exact_binary_ratio"].items()
            if key.startswith(("coefficient.", "coefficient_bound."))}),
        coefficient_count=int(count), exact_native_coefficient_count=int(exact_count),
        saved_coefficient_budget_status="PASS" if not failures else "OPEN",
        saved_coefficient_budget_failure_count=len(failures), saved_coefficient_budget_failures=failures,
        exact_canonical_coefficient_residual_certified=True,
        source_coefficient_accuracy_against_saved_bounds_certified=not failures,
        advertised_forward_construction_budget_authenticated=False,
        canonical_residual_scope="unique_exact_distance_for_supplied_native_coefficient_and_source_inputs",
        source_operand_majorant_scope="unsigned_independent_construction_algebra_not_a_roundoff_budget",
        geometry_endpoint_accuracy_certified=False, angular_integral_accuracy_certified=False,
        physical_creation_admitted=False, physical_release_admitted=False,
        available_for_creation_funding=False, families=families)
    body["coefficient_proof_sha256"] = digest(body)
    return body


def verify_coefficient_proof(raw, N, witness, proof):
    """Recompute canonical content; consistently resealed inflation still fails."""
    _require(type(proof) is dict and audit_coefficients(raw, N, witness) == proof,
             "Canonical coefficient proof differs")
    return True
