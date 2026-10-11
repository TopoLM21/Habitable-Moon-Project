"""Directed coefficient enclosures for arbitrary represented Genesis lines.

This module bounds the exact algebra of the original binary source operands,
conditioned on the supplied represented midpoint and halfwidth.  It does not
bound the sphere endpoints, Gamma root, radial primitive or angular quadrature.
The original binary64 parent vertices and angular inverse remain literal
inputs.  No history, operator or admission state is changed.

Each elementary scalar operation is followed by ``nextafter`` in both
directions.  Products and reductions are explicit; no BLAS error estimate or
empirical multiplier enters the enclosure.  Only six small birth-offset leaves
use rational arithmetic to preserve their world-coordinate cancellation.  The
coefficient tensors themselves use directed native arithmetic throughout.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
import hashlib
import json

import numpy as np

LD = np.longdouble
SCOPE = "directed_coefficients_conditioned_on_literal_represented_line"
FIELDS = ("lam", "w", "L", "Bhat", "Ghat_without_H", "T", "E_without_H", "memoryhat")


class OperandBoundRefusal(ValueError):
    """The requested exact coefficient enclosure is unresolved."""


def _readonly(value):
    array = np.array(value, dtype=LD, copy=True)
    return np.frombuffer(array.tobytes(), dtype=LD).reshape(array.shape)


def _finite(value):
    value = np.asarray(value, dtype=LD)
    if not np.isfinite(value).all():
        raise OperandBoundRefusal("Directed operation is outside finite native range")
    return value


def _down(value):
    return _finite(np.nextafter(_finite(value), LD(-np.inf)))


def _up(value):
    return _finite(np.nextafter(_finite(value), LD(np.inf)))


@dataclass(frozen=True)
class DirectedInterval:
    """Inclusive lower/upper bounds on a real scalar or array."""
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self):
        lo, hi = _finite(self.lower), _finite(self.upper)
        if lo.shape != hi.shape or np.any(lo > hi):
            raise OperandBoundRefusal("Directed interval shape or ordering differs")
        object.__setattr__(self, "lower", _readonly(lo))
        object.__setattr__(self, "upper", _readonly(hi))

    @property
    def shape(self):
        return self.lower.shape

    def __getitem__(self, index):
        return DirectedInterval(self.lower[index], self.upper[index])

    def reshape(self, shape):
        return DirectedInterval(self.lower.reshape(shape), self.upper.reshape(shape))

    def contains(self, value):
        value = np.asarray(value, dtype=LD)
        return (value >= self.lower) & (value <= self.upper)

    def absolute_error_for(self, value):
        """Conservative distance from a represented value to either endpoint."""
        value = _finite(value)
        if value.shape != self.shape:
            raise OperandBoundRefusal("Compared coefficient shape differs")
        result = np.maximum(_up(np.abs(value-self.lower)),
                            _up(np.abs(self.upper-value)))
        exact = (self.lower == self.upper) & (value == self.lower)
        return _readonly(np.where(exact, LD(0), result))


def point(value):
    """Treat an already represented finite binary floating value as exact."""
    return DirectedInterval(value, value)


def add(a, b):
    a, b = _interval(a), _interval(b)
    lo = _down(a.lower+b.lower)
    hi = _up(a.upper+b.upper)
    az = (a.lower == 0) & (a.upper == 0)
    bz = (b.lower == 0) & (b.upper == 0)
    lo = np.where(az, b.lower, np.where(bz, a.lower, lo))
    hi = np.where(az, b.upper, np.where(bz, a.upper, hi))
    return DirectedInterval(lo, hi)


def subtract(a, b):
    b = _interval(b)
    return add(a, DirectedInterval(-b.upper, -b.lower))


def multiply(a, b):
    a, b = _interval(a), _interval(b)
    products = np.stack(np.broadcast_arrays(a.lower*b.lower, a.lower*b.upper,
                                            a.upper*b.lower, a.upper*b.upper))
    lo, hi = _down(products.min(axis=0)), _up(products.max(axis=0))
    zero = (((a.lower == 0) & (a.upper == 0)) |
            ((b.lower == 0) & (b.upper == 0)))
    # These exact binary identities require no rounding step.
    ap = (a.lower == 1) & (a.upper == 1)
    bp = (b.lower == 1) & (b.upper == 1)
    an = (a.lower == -1) & (a.upper == -1)
    bn = (b.lower == -1) & (b.upper == -1)
    lo = np.where(ap, b.lower, np.where(bp, a.lower, np.where(an, -b.upper,
                  np.where(bn, -a.upper, lo))))
    hi = np.where(ap, b.upper, np.where(bp, a.upper, np.where(an, -b.lower,
                  np.where(bn, -a.lower, hi))))
    return DirectedInterval(np.where(zero, LD(0), lo), np.where(zero, LD(0), hi))


def divide(a, b):
    a, b = _interval(a), _interval(b)
    if np.any((b.lower <= 0) & (b.upper >= 0)):
        raise OperandBoundRefusal("Directed denominator contains zero")
    quotients = np.stack(np.broadcast_arrays(a.lower/b.lower, a.lower/b.upper,
                                             a.upper/b.lower, a.upper/b.upper))
    lo, hi = _down(quotients.min(axis=0)), _up(quotients.max(axis=0))
    zero = (a.lower == 0) & (a.upper == 0)
    return DirectedInterval(np.where(zero, LD(0), lo), np.where(zero, LD(0), hi))


def _interval(value):
    return value if isinstance(value, DirectedInterval) else point(value)


def sum_interval(a, axis):
    a = _interval(a)
    axis %= len(a.shape)
    lo, hi = np.moveaxis(a.lower, axis, 0), np.moveaxis(a.upper, axis, 0)
    total = point(np.zeros(lo.shape[1:], dtype=LD))
    for i in range(len(lo)):
        total = add(total, DirectedInterval(lo[i], hi[i]))
    return total


def _dot(a, b):
    return sum_interval(multiply(a, b), -1)


def _matmul(a, b):
    a, b = _interval(a), _interval(b)
    if len(a.shape) < 1 or len(b.shape) != 2 or a.shape[-1] != b.shape[0]:
        raise OperandBoundRefusal("Directed matrix-product shape differs")
    av = a.reshape(a.shape+(1,))
    bv = b.reshape((1,)*(len(a.shape)-1)+b.shape)
    return sum_interval(multiply(av, bv), -2)


def inverse3(a):
    """Enclose the exact inverse by directed determinant and adjugate algebra."""
    a = _interval(a)
    if a.shape != (3, 3):
        raise OperandBoundRefusal("Three-dimensional source chart is required")
    cof = [[None]*3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            rows = [k for k in range(3) if k != i]
            cols = [k for k in range(3) if k != j]
            minor = subtract(multiply(a[rows[0], cols[0]], a[rows[1], cols[1]]),
                             multiply(a[rows[0], cols[1]], a[rows[1], cols[0]]))
            cof[i][j] = multiply(minor, (-1)**(i+j))
    determinant = point(0)
    for j in range(3):
        determinant = add(determinant, multiply(a[0, j], cof[0][j]))
    inv = [[divide(cof[j][i], determinant) for j in range(3)] for i in range(3)]
    return DirectedInterval([[v.lower for v in row] for row in inv],
                            [[v.upper for v in row] for row in inv])


def _fraction(value):
    return Fraction(*LD(value).as_integer_ratio())


def rational_enclosure(value):
    """Bracket one exact rational without assuming its conversion rounds once."""
    value = Fraction(value)
    if not value:
        return point(0)
    numerator, denominator = abs(value.numerator), value.denominator
    keep = np.finfo(LD).nmant+8
    ns, ds = max(numerator.bit_length()-keep, 0), max(denominator.bit_length()-keep, 0)
    guess = np.ldexp(LD(numerator >> ns)/LD(denominator >> ds), ns-ds)
    if value < 0:
        guess = -guess
    if not np.isfinite(guess):
        raise OperandBoundRefusal("Exact source leaf is outside native range")
    lower = upper = guess
    # The comparisons are exact rational comparisons, so this termination
    # guard refuses instead of accepting a guessed conversion-error bound.
    for _ in range(32):
        if _fraction(lower) <= value <= _fraction(upper):
            return DirectedInterval(lower, upper)
        if _fraction(lower) > value:
            lower = LD(_down(lower))
        if _fraction(upper) < value:
            upper = LD(_up(upper))
    raise OperandBoundRefusal("Exact source leaf could not be bracketed")


def _stack(items, axis=0):
    return DirectedInterval(np.stack([p.lower for p in items], axis=axis),
                            np.stack([p.upper for p in items], axis=axis))


def _constant(value):
    value = _interval(value)
    return value.reshape((1,)+value.shape)


def _broadcast_poly(poly, trailing):
    shape = (poly.shape[0],)+(1,)*(len(trailing)-len(poly.shape)+1)+poly.shape[1:]
    lower = np.broadcast_to(poly.lower.reshape(shape), (poly.shape[0],)+trailing)
    upper = np.broadcast_to(poly.upper.reshape(shape), (poly.shape[0],)+trailing)
    return DirectedInterval(lower, upper)


def _poly_add(a, b):
    trailing = np.broadcast_shapes(a.shape[1:], b.shape[1:])
    a, b = _broadcast_poly(a, trailing), _broadcast_poly(b, trailing)
    n = max(a.shape[0], b.shape[0])
    av, au, bv, bu = [np.zeros((n,)+trailing, dtype=LD) for _ in range(4)]
    av[:a.shape[0]], au[:a.shape[0]] = a.lower, a.upper
    bv[:b.shape[0]], bu[:b.shape[0]] = b.lower, b.upper
    return add(DirectedInterval(av, au), DirectedInterval(bv, bu))


def _poly_mul(a, b):
    trailing = np.broadcast_shapes(a.shape[1:], b.shape[1:])
    a, b = _broadcast_poly(a, trailing), _broadcast_poly(b, trailing)
    out = [point(np.zeros(trailing, dtype=LD)) for _ in range(a.shape[0]+b.shape[0]-1)]
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            out[i+j] = add(out[i+j], multiply(a[i], b[j]))
    return _stack(out)


def _poly_dot(a, b):
    return sum_interval(_poly_mul(a, b), -1)


def _component(a, index):
    return a[(slice(None),)+index]


@dataclass(frozen=True)
class CoefficientBoundCertificate:
    fields: object
    source_binding_sha256: str
    line_operand_sha256: str
    scope: str = SCOPE
    coefficient_construction_accuracy_certified: bool = True
    geometry_operand_accuracy_certified: bool = False
    radial_integral_accuracy_certified: bool = False
    angular_integral_convergence_certified: bool = False
    physical_contraction_certified: bool = False
    available_for_creation_funding: bool = False

    def error_bounds_for(self, coefficients):
        if set(coefficients) != set(FIELDS):
            raise OperandBoundRefusal("Complete original coefficient family is required")
        return MappingProxyType({name: self.fields[name].absolute_error_for(coefficients[name])
                                 for name in FIELDS})


class PreparedOperandBounds:
    """Reusable source-bound coefficient enclosure at arbitrary original N."""
    __slots__ = ("source", "C", "D", "X", "Q", "radius2", "radial_width",
                 "chart_inverse", "angular_inverse", "B0", "B1", "_C", "_D", "_center", "_sealed")

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("Prepared original source bounds are immutable")
        object.__setattr__(self, name, value)

    def __init__(self, source):
        from .genesis_volume_exact_lambda_bulk import validate_source
        validate_source(source)
        self.source = source
        self.C = point(source.geometry.birth_C_m)
        self.D = point(source.geometry.birth_D_m)
        self.X = point(source.section.triangle_m)
        self.Q = point(source.basis_xyz)
        self.radius2 = multiply(source.geometry.seed_radius_m, source.geometry.seed_radius_m)
        self.radial_width = subtract(source.section.lambda_bounds[1], source.section.lambda_bounds[0])
        chart_columns = (subtract(self.X[1], self.X[0]),
                         subtract(self.X[2], self.X[0]), self.X[0])
        self.chart_inverse = inverse3(_stack(chart_columns, axis=1))
        self.angular_inverse = point(source.angular_inverse)
        self.B0 = point(source.literal_vertex_B0)
        self.B1 = point(source.literal_vertex_B1)
        self._C = tuple(tuple(_fraction(v) for v in row) for row in source.geometry.birth_C_m)
        self._D = tuple(tuple(_fraction(v) for v in row) for row in source.geometry.birth_D_m)
        self._center = tuple(_fraction(v) for v in source.geometry.seed_center_world_m)
        self._sealed = True

    def _offset(self, N, lm, lh):
        n, mid, half = list(map(_fraction, N)), _fraction(lm), _fraction(lh)
        u = [sum((n[i]*self._C[i][j] for i in range(3)), Fraction()) for j in range(3)]
        d = [sum((n[i]*self._D[i][j] for i in range(3)), Fraction()) for j in range(3)]
        return _stack([_stack([rational_enclosure(d[j]-self._center[j]+mid*u[j])
                               for j in range(3)]),
                       _stack([rational_enclosure(half*u[j]) for j in range(3)])])

    def evaluate(self, interval):
        if interval.source is not self.source.geometry:
            raise OperandBoundRefusal("Interval is not bound to the prepared original source")
        N = point(interval.barycentric)
        lm, lh = interval.lambda_mid, interval.lambda_halfwidth
        lam = point([lm, lh])
        offset = self._offset(interval.barycentric, lm, lh)
        one = _constant(1)
        S = _poly_add(one, multiply(_poly_dot(offset, offset), divide(-1, self.radius2)))
        sphere = _poly_mul(S, S)
        lower, upper = self.source.section.lambda_bounds
        t = divide(_stack([subtract(lm, lower), point(lh)]), self.radial_width)
        tbar = divide(_stack([subtract(upper, lm), point(-lh)]), self.radial_width)
        radial = multiply(_poly_mul(t, tbar), 4)
        angular = _constant(multiply(multiply(multiply(27, N[0]), N[1]), N[2]))
        w = _poly_mul(_poly_mul(angular, radial), sphere)
        edge1 = _poly_add(_poly_mul(lam, _constant(subtract(self.C[1], self.C[0]))),
                          _constant(subtract(self.D[1], self.D[0])))
        edge2 = _poly_add(_poly_mul(lam, _constant(subtract(self.C[2], self.C[0]))),
                          _constant(subtract(self.D[2], self.D[0])))
        u = _matmul(N, self.C)
        duS2 = multiply(_poly_mul(S, _poly_dot(edge1, offset)), divide(-4, self.radius2))
        dvS2 = multiply(_poly_mul(S, _poly_dot(edge2, offset)), divide(-4, self.radius2))
        dlS2 = multiply(_poly_mul(S, _poly_dot(_constant(u), offset)), divide(-4, self.radius2))
        ag1 = multiply(multiply(27, N[2]), subtract(N[0], N[1]))
        ag2 = multiply(multiply(27, N[1]), subtract(N[0], N[2]))
        du = _poly_add(_poly_mul(_poly_mul(_constant(ag1), radial), sphere),
                       _poly_mul(_poly_mul(angular, radial), duS2))
        dv = _poly_add(_poly_mul(_poly_mul(_constant(ag2), radial), sphere),
                       _poly_mul(_poly_mul(angular, radial), dvS2))
        rd = multiply(_poly_add(tbar, multiply(t, -1)), divide(4, self.radial_width))
        dl = _poly_add(_poly_mul(_poly_mul(angular, radial), dlS2),
                       _poly_mul(_poly_mul(angular, sphere), rd))
        radialpart = _poly_add(_poly_mul(lam, dl),
                               multiply(_poly_add(multiply(du, N[1]), multiply(dv, N[2])), -1))
        inv = self.chart_inverse
        L = _poly_add(_poly_add(_poly_mul(du, _constant(inv[0])),
                                _poly_mul(dv, _constant(inv[1]))),
                       _poly_mul(radialpart, _constant(inv[2])))
        n_vertices = N.reshape((3, 1, 1))
        B0 = sum_interval(multiply(n_vertices, self.B0), 0)
        B1 = sum_interval(multiply(n_vertices, self.B1), 0)
        Bhat = _poly_add(_poly_mul(lam, _constant(B0)), _constant(B1))
        tl, th = np.zeros((2, 3, 18), dtype=LD), np.zeros((2, 3, 18), dtype=LD)
        shift = _stack([subtract(lm, 1), point(lh)])
        eye = point(np.eye(3, dtype=LD))
        for i in range(3):
            a = multiply(N[i], eye)
            p = multiply(multiply(shift.reshape((2, 1, 1)), N[i]), eye)
            tl[0, :, 6*i:6*i+3], th[0, :, 6*i:6*i+3] = a.lower, a.upper
            tl[:, :, 6*i+3:6*i+6], th[:, :, 6*i+3:6*i+6] = p.lower, p.upper
        T = DirectedInterval(tl, th)
        # Q.T @ T for every coefficient, by an explicit three-term reduction.
        lt = sum_interval(multiply(self.Q.reshape((1, 3, 3, 1)),
                                   T.reshape((2, 3, 1, 18))), 1)
        lg = _matmul(L, self.Q)
        components = []
        for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)):
            term = _poly_mul(_component(lt, (i, slice(None))), _component(lg, (j,)))
            if i != j:
                term = _poly_add(term, _poly_mul(_component(lt, (j, slice(None))),
                                                _component(lg, (i,))))
            components.append(term)
        extra = _stack(components, axis=1)
        G = _poly_add(_poly_mul(w, Bhat), extra)
        E = _poly_mul(w, T)
        angular_memory = _matmul(N, self.angular_inverse)
        m0 = _matmul(angular_memory, point(self.source.memory0))
        m1 = _matmul(angular_memory, point(self.source.memory1))
        memory = _poly_add(_poly_mul(lam, _constant(m0)), _constant(m1))
        fields = dict(lam=lam, w=w, L=L, Bhat=Bhat, Ghat_without_H=G,
                      T=T, E_without_H=E, memoryhat=memory)
        operands = dict(source=self.source.binding_sha256,
            N=[list(map(str, LD(v).as_integer_ratio())) for v in interval.barycentric],
            midpoint=list(map(str, LD(lm).as_integer_ratio())),
            halfwidth=list(map(str, LD(lh).as_integer_ratio())))
        binding = hashlib.sha256(json.dumps(operands, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return CoefficientBoundCertificate(MappingProxyType(fields), self.source.binding_sha256, binding)


def prepare_operand_bounds(source):
    return PreparedOperandBounds(source)
