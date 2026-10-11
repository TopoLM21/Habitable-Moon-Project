"""Directed source-integral enclosures on original Genesis material lines.

The original binary source operands define every coefficient, determinant,
density, memory field and Gamma bank.  The native radial chart is only a
coordinate system: the moments integrate over the independently enclosed
source sphere/cohort endpoints, including endpoints outside that chart's
[-1, 1].  These bounds certify radial numerical arithmetic and source endpoint
transport; they do not certify angular convergence or physical admission.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
import hashlib
import json

import numpy as np

from .genesis_volume_operand_bounds import (
    DirectedInterval, OperandBoundRefusal, point, add, subtract, multiply,
    divide, sum_interval, prepare_operand_bounds,
    _matmul, _poly_add, _poly_mul, _constant, _stack, _readonly,
)

LD = np.longdouble
_UNSET = object()
SCOPE = "directed_literal_source_radial_integral_including_endpoint_transport"
BLOCKS = ("stiffness_cross", "stiffness_enriched", "mass_cross",
          "mass_enriched", "enrichment_force_n")
VOLUMES = ("reference_volume_negative_m3", "reference_volume_positive_m3")
PROBE_SCALARS = ("cross_parent_work_j", "cross_added_work_j", "self_work_j",
    "stiffness_cross_bilinear_j", "stiffness_self_bilinear_j",
    "mass_cross_bilinear_kg_m2", "mass_self_bilinear_kg_m2")


class IntegralBoundRefusal(OperandBoundRefusal):
    """The requested original source integral cannot be enclosed."""


def multiply(a, b):
    """Directed endpoint products, freezing only the completed tensor.

    The four exact real endpoint products attain the interval extrema.  One
    native predecessor/successor encloses each finite rounded product.  This
    is the same elementary theorem as the frozen coefficient implementation;
    tensor temporaries need no immutable DTO or repeated finiteness scan.
    """
    a = a if isinstance(a, DirectedInterval) else point(a)
    b = b if isinstance(b, DirectedInterval) else point(b)
    products = np.broadcast_arrays(a.lower*b.lower, a.lower*b.upper,
                                   a.upper*b.lower, a.upper*b.upper)
    lo = np.nextafter(np.minimum.reduce(products), LD(-np.inf))
    hi = np.nextafter(np.maximum.reduce(products), LD(np.inf))
    zero = (((a.lower == 0) & (a.upper == 0)) |
            ((b.lower == 0) & (b.upper == 0)))
    ap, bp = (a.lower == 1) & (a.upper == 1), (b.lower == 1) & (b.upper == 1)
    an, bn = (a.lower == -1) & (a.upper == -1), (b.lower == -1) & (b.upper == -1)
    lo = np.where(ap, b.lower, np.where(bp, a.lower, np.where(an, -b.upper, np.where(bn, -a.upper, lo))))
    hi = np.where(ap, b.upper, np.where(bp, a.upper, np.where(an, -b.lower, np.where(bn, -a.lower, hi))))
    return DirectedInterval(np.where(zero, LD(0), lo), np.where(zero, LD(0), hi))


def sum_interval(a, axis):
    """Sequential outward reduction without freezing every partial sum."""
    a = a if isinstance(a, DirectedInterval) else point(a)
    axis %= len(a.shape)
    lower, upper = np.moveaxis(a.lower, axis, 0), np.moveaxis(a.upper, axis, 0)
    lo = np.zeros(lower.shape[1:], dtype=LD)
    hi = np.zeros(upper.shape[1:], dtype=LD)
    for l, h in zip(lower, upper):
        total_zero, term_zero = (lo == 0) & (hi == 0), (l == 0) & (h == 0)
        rounded_lo = np.nextafter(lo+l, LD(-np.inf))
        rounded_hi = np.nextafter(hi+h, LD(np.inf))
        lo = np.where(total_zero, l, np.where(term_zero, lo, rounded_lo))
        hi = np.where(total_zero, h, np.where(term_zero, hi, rounded_hi))
    return DirectedInterval(lo, hi)


def _matmul(a, b):
    a = a if isinstance(a, DirectedInterval) else point(a)
    b = b if isinstance(b, DirectedInterval) else point(b)
    if len(a.shape) < 1 or len(b.shape) != 2 or a.shape[-1] != b.shape[0]:
        raise IntegralBoundRefusal("Directed matrix-product shape differs")
    av = a.reshape(a.shape+(1,))
    bv = b.reshape((1,)*(len(a.shape)-1)+b.shape)
    return sum_interval(multiply(av, bv), -2)


def _transpose(value, axes):
    return DirectedInterval(value.lower.transpose(axes), value.upper.transpose(axes))


def polynomial_project(polynomial, basis):
    """Project columns before products, with an explicit directed reduction."""
    if not isinstance(polynomial, DirectedInterval):
        polynomial = point(polynomial)
    if not isinstance(basis, DirectedInterval):
        basis = point(basis)
    if len(polynomial.shape) != 3 or len(basis.shape) != 2 or polynomial.shape[-1] != basis.shape[0]:
        raise IntegralBoundRefusal("Polynomial projection shape differs")
    if basis.shape[0] == basis.shape[1] and np.array_equal(basis.lower, basis.upper) and np.array_equal(basis.lower, np.eye(basis.shape[0])):
        return polynomial
    return _matmul(polynomial, basis)


def _bilinear_pairs(left, middle, right):
    """Exact algebra enclosed as degree-pair tensors, before integration."""
    left, middle, right = (v if isinstance(v, DirectedInterval) else point(v)
                           for v in (left, middle, right))
    if len(left.shape) != 3 or len(right.shape) != 3 or middle.shape != (left.shape[1], right.shape[1]):
        raise IntegralBoundRefusal("Polynomial bilinear shape differs")
    if (middle.shape[0] == middle.shape[1] and
            np.array_equal(middle.lower, middle.upper) and
            np.array_equal(middle.lower, np.eye(middle.shape[0]))):
        transformed = left
    else:
        transformed = _transpose(_matmul(_transpose(left, (0, 2, 1)), middle), (0, 2, 1))
    pl, channels, columns = transformed.shape
    pr, _, outputs = right.shape
    products = multiply(transformed.reshape((pl, 1, channels, columns, 1)),
                        right.reshape((1, pr, channels, 1, outputs)))
    return sum_interval(products, 2)


def polynomial_bilinear(left, middle, right):
    """Enclose every coefficient of left.T @ middle @ right."""
    pairs = _bilinear_pairs(left, middle, right)
    coefficients = []
    for degree in range(pairs.shape[0]+pairs.shape[1]-1):
        terms = [pairs[i, degree-i] for i in range(pairs.shape[0])
                 if 0 <= degree-i < pairs.shape[1]]
        coefficients.append(sum_interval(_stack(terms), 0))
    return _stack(coefficients)


def polynomial_integral(polynomial, moments):
    """Enclose a polynomial integral using directed, source-bound moments."""
    polynomial = polynomial if isinstance(polynomial, DirectedInterval) else point(polynomial)
    moments = moments if isinstance(moments, DirectedInterval) else point(moments)
    if len(moments.shape) != 1 or not 1 <= polynomial.shape[0] <= len(moments.lower):
        raise IntegralBoundRefusal("Polynomial moment shape differs")
    selected = moments[:polynomial.shape[0]].reshape((polynomial.shape[0],)+(1,)*(len(polynomial.shape)-1))
    return sum_interval(multiply(polynomial, selected), 0)


def _weighted_moments(moments, polynomial):
    """Moments of xi**k times a scalar polynomial (for the lambda² measure)."""
    if len(polynomial.shape) != 1 or len(moments.shape) != 1:
        raise IntegralBoundRefusal("Scalar measure polynomial is required")
    count = moments.shape[0]-polynomial.shape[0]+1
    if count < 1:
        raise IntegralBoundRefusal("Measure polynomial exceeds the moment degree")
    result = []
    for degree in range(count):
        result.append(sum_interval(multiply(polynomial, moments[degree:degree+polynomial.shape[0]]), 0))
    return _stack(result)


def _bilinear_integral(left, middle, right, moments, factor):
    # Integrate the right polynomial separately for each left degree.  This
    # exact distributive identity avoids constructing the degree-pair tensor
    # of every output matrix component (and its much larger temporary arrays).
    # Each reduction remains directed; no BLAS rounding premise is needed.
    if left.shape[1] != right.shape[1] or middle.shape != (left.shape[1], right.shape[1]):
        raise IntegralBoundRefusal("Polynomial bilinear shape differs")
    index = np.arange(left.shape[0])[:, None]+np.arange(right.shape[0])[None, :]
    if int(index.max()) >= moments.shape[0]:
        raise IntegralBoundRefusal("Bilinear degree exceeds the source moments")
    weights = DirectedInterval(moments.lower[index], moments.upper[index]).reshape(index.shape+(1, 1))
    integrated_right = sum_interval(multiply(weights, right.reshape((1,)+right.shape)), 1)
    if (middle.shape[0] == middle.shape[1] and
            np.array_equal(middle.lower, middle.upper) and
            np.array_equal(middle.lower, np.eye(middle.shape[0]))):
        transformed = left
    else:
        transformed = _transpose(_matmul(_transpose(left, (0, 2, 1)), middle), (0, 2, 1))
    degree, channels, columns = transformed.shape
    outputs = integrated_right.shape[-1]
    products = multiply(transformed.reshape((degree, channels, columns, 1)),
                        integrated_right.reshape((degree, channels, 1, outputs)))
    value = sum_interval(sum_interval(products, 1), 0)
    return multiply(value, factor)


def _determinant(triangle):
    x = point(triangle)
    determinant = point(0)
    for j in range(3):
        minor = subtract(multiply(x[1, (j+1)%3], x[2, (j+2)%3]),
                         multiply(x[1, (j+2)%3], x[2, (j+1)%3]))
        determinant = add(determinant, multiply(x[0, j], minor))
    if determinant.lower <= 0:
        raise IntegralBoundRefusal("Positive source reference determinant is unresolved")
    return determinant


def _basis(value, name):
    value = np.asarray(value, dtype=LD)
    if value.ndim != 2 or value.shape[0] != 18 or not 1 <= value.shape[1] <= 18 or not np.isfinite(value).all():
        raise IntegralBoundRefusal(name+" basis shape differs")
    return point(value)


def _probe_vector(probe, name, alias=None, default=None):
    value = probe.get(name, probe.get(alias, default) if alias else default)
    value = np.asarray(value, dtype=LD)
    if value.shape != (18,) or not np.isfinite(value).all():
        raise IntegralBoundRefusal("Declared probe vector shape differs: "+name)
    return point(value[:, None])


def _memory(source, N, lam, e0, e1):
    angular = _matmul(point(N), point(source.angular_inverse))
    e0, e1 = np.asarray(e0, dtype=LD), np.asarray(e1, dtype=LD)
    if e0.shape != (3, 6) or e1.shape != (3, 6) or not np.isfinite(e0).all() or not np.isfinite(e1).all():
        raise IntegralBoundRefusal("Declared probe memory shape differs")
    first, second = _matmul(angular, point(e0)), _matmul(angular, point(e1))
    return _poly_add(_poly_mul(lam, _constant(first)), _constant(second)).reshape((2, 6, 1))


def _frozen_map(mapping):
    return MappingProxyType(dict(mapping))


@dataclass(frozen=True)
class IntegralBoundCertificate:
    """A true-source radial enclosure, distinct from historical error budgets."""
    enclosure: DirectedInterval
    names: tuple
    components: object
    source_binding_sha256: str
    line_operand_sha256: str
    geometry_operand_sha256: str
    component_witness_included: bool = True
    scope: str = SCOPE
    radial_integral_accuracy_certified: bool = True
    source_endpoint_transport_certified: bool = True
    coefficient_construction_accuracy_certified: bool = True
    geometry_operand_accuracy_certified: bool = True
    angular_integral_convergence_certified: bool = False
    historical_saved_budget_authentication_certified: bool = False
    physical_contraction_certified: bool = False
    available_for_creation_funding: bool = False

    @property
    def lower(self):
        return self.enclosure.lower

    @property
    def upper(self):
        return self.enclosure.upper

    def error_bounds_for(self, represented_values, names=None):
        if names is not None and tuple(names) != self.names:
            raise IntegralBoundRefusal("Compared component schema differs")
        if isinstance(represented_values, dict):
            if set(represented_values) != set(self.names):
                raise IntegralBoundRefusal("Compared component schema differs")
            represented_values = [represented_values[name] for name in self.names]
        return self.enclosure.absolute_error_for(represented_values)

    def absolute_error_for(self, represented_values):
        return self.error_bounds_for(represented_values)


class PreparedIntegralBounds:
    """Reusable directed source bounds for the complete original line schema."""
    __slots__ = ("source", "coefficients", "geometry", "C", "determinant", "mass_factor",
                 "parent_basis", "added_basis", "probes", "probe_q", "probe_a", "probe_z",
                 "probe_e0", "probe_e1", "include_volumes", "names", "options_binding", "_sealed")

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("Prepared original integral bounds are immutable")
        object.__setattr__(self, name, value)

    def __init__(self, source, raw, *, added_basis=None, parent_basis=None, probes=(), include_volumes=True):
        from .genesis_volume_exact_lambda_bulk import prepare_source, validate_source
        from .genesis_volume_integral_geometry_bounds import prepare_integral_geometry
        validate_source(source)
        if prepare_source(raw).binding_sha256 != source.binding_sha256:
            raise IntegralBoundRefusal("Literal source operands differ from prepared source")
        self.source = source
        self.coefficients = prepare_operand_bounds(source)
        self.geometry = prepare_integral_geometry(raw)
        self.C = point(source.elasticity_pa)
        self.determinant = _determinant(source.section.triangle_m)
        density = divide(source.section.mass_kg, source.section.volume_m3)
        if density.lower <= 0:
            raise IntegralBoundRefusal("Original density is not positive")
        self.mass_factor = multiply(self.determinant, density)
        self.parent_basis = _basis(np.eye(18) if parent_basis is None else parent_basis, "Parent")
        self.added_basis = None if added_basis is None else _basis(added_basis, "Added")
        if len(probes) > 16 or any(not isinstance(p, dict) for p in probes) or len({p.get("name") for p in probes}) != len(probes):
            raise IntegralBoundRefusal("Declared probe names/count differ")
        prepared = []
        for probe in probes:
            if not isinstance(probe.get("name"), str) or not probe["name"]:
                raise IntegralBoundRefusal("Declared probe needs a stable name")
            q = _probe_vector(probe, "parent_increment_m", "q")
            a = _probe_vector(probe, "enrichment_increment_m", "a")
            z = _probe_vector(probe, "mean_z_m", "mean_z", np.zeros(18))
            e0 = _readonly(probe.get("mean_e0", source.memory0))
            e1 = _readonly(probe.get("mean_e1", source.memory1))
            if e0.shape != (3, 6) or e1.shape != (3, 6):
                raise IntegralBoundRefusal("Declared probe memory shape differs")
            prepared.append(MappingProxyType(dict(name=probe["name"], q=q, a=a, z=z, e0=e0, e1=e1)))
        self.probes = tuple(prepared)
        self.probe_q = point(np.column_stack([p["q"].lower[:, 0] for p in self.probes])) if self.probes else None
        self.probe_a = point(np.column_stack([p["a"].lower[:, 0] for p in self.probes])) if self.probes else None
        self.probe_z = point(np.column_stack([p["z"].lower[:, 0] for p in self.probes])) if self.probes else None
        self.probe_e0 = point(np.stack([p["e0"] for p in self.probes], axis=-1)) if self.probes else None
        self.probe_e1 = point(np.stack([p["e1"] for p in self.probes], axis=-1)) if self.probes else None
        self.include_volumes = bool(include_volumes)
        shapes = [("full", name, (18,) if name == "enrichment_force_n" else (18, 18)) for name in BLOCKS]
        if self.added_basis is not None:
            nv, nw = self.parent_basis.shape[1], self.added_basis.shape[1]
            shapes += [("projected", name, (nw,) if name == "enrichment_force_n" else
                        ((nw, nw) if name.endswith("enriched") else (nv, nw))) for name in BLOCKS]
        for probe in self.probes:
            prefix = "probe."+probe["name"]
            shapes += [(prefix, name, ()) for name in PROBE_SCALARS]
            shapes += [(prefix, name, (18,)) for name in ("added_parent_force_n", "enrichment_force_n")]
        names = list(VOLUMES) if self.include_volumes else []
        for prefix, name, shape in shapes:
            for index in np.ndindex(shape):
                suffix = "["+",".join(map(str, index))+"]" if index else ""
                names.append(prefix+"."+name+suffix)
        self.names = tuple(names)
        def ratios(values):
            return [list(map(str, LD(v).as_integer_ratio())) for v in np.asarray(values).flat]
        options = dict(names=self.names, V=ratios(self.parent_basis.lower),
            W=None if self.added_basis is None else ratios(self.added_basis.lower),
            probes=[dict(name=p["name"], **{k:ratios(p[k].lower if isinstance(p[k], DirectedInterval) else p[k]) for k in ("q","a","z","e0","e1")}) for p in self.probes])
        self.options_binding = hashlib.sha256(json.dumps(options, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self._sealed = True

    def evaluate(self, material_line, native_interval=_UNSET, *, geometry=None, witness=True):
        from .genesis_volume_exact_lambda import prepare_fixed_N_interval
        if hasattr(material_line, "lambda_mid"):
            if native_interval is not _UNSET:
                raise IntegralBoundRefusal("Native interval was supplied twice")
            interval = material_line
            if interval.source is not self.source.geometry:
                raise IntegralBoundRefusal("Interval is not bound to the prepared original source")
            N = interval.barycentric
        else:
            N = np.asarray(material_line, dtype=LD)
            interval = (prepare_fixed_N_interval(self.source.geometry, N)
                        if native_interval is _UNSET else native_interval)
            if interval is not None and (interval.source is not self.source.geometry or
                    not np.array_equal(interval.barycentric, N)):
                raise IntegralBoundRefusal("Interval is not bound to the prepared original source line")
        if geometry is None:
            geometry = self.geometry.evaluate(N, interval)
        elif not self.geometry.matches(geometry, N, interval):
            raise IntegralBoundRefusal("Geometry moments are not bound to this source line")
        if interval is None:
            # The geometry evaluator refuses positive source support with an
            # empty native chart.  Thus exact zero needs no approximate chart.
            zeros = np.zeros(len(self.names), dtype=LD)
            components = MappingProxyType({name: point(0) for name in self.names} if witness else {})
            return self._certificate(point(zeros), components, geometry, witness)
        f = self.coefficients.evaluate(interval).fields
        moments = {side: geometry.moments(16, side) for side in (0, 1, 2)}
        lam2 = _poly_mul(f["lam"], f["lam"])
        mass_moments = {side: _weighted_moments(moments[side], lam2) for side in (1, 2)}
        eye3 = point(np.eye(3))
        def integrate(left, middle, right, side, mass=False):
            return _bilinear_integral(left, middle, right,
                mass_moments[side] if mass else moments[side], self.mass_factor if mass else self.determinant)
        def blocks(V, W):
            B, G = polynomial_project(f["Bhat"], V), polynomial_project(f["Ghat_without_H"], W)
            T, E = polynomial_project(f["T"], V), polynomial_project(f["E_without_H"], W)
            memory = f["memoryhat"].reshape((f["memoryhat"].shape[0], 6, 1))
            return dict(stiffness_cross=integrate(B, self.C, G, 1),
                stiffness_enriched=integrate(G, self.C, G, 2),
                mass_cross=integrate(T, eye3, E, 1, True),
                mass_enriched=integrate(E, eye3, E, 2, True),
                enrichment_force_n=integrate(G, self.C, memory, 1)[:, 0])
        components, flat_lower, flat_upper = {}, [], []
        def append(prefix, values):
            for name, bound in values.items():
                flat_lower.append(bound.lower.reshape(-1))
                flat_upper.append(bound.upper.reshape(-1))
                if witness:
                    for index in np.ndindex(bound.shape):
                        suffix = "["+",".join(map(str,index))+"]" if index else ""
                        components[prefix+"."+name+suffix] = bound[index] if index else bound
        if self.include_volumes:
            for name, sign in zip(VOLUMES, (-1, 1)):
                bank = geometry.bank_moments(2, sign)
                value = multiply(polynomial_integral(lam2, bank), self.determinant)
                flat_lower.append(value.lower.reshape(-1))
                flat_upper.append(value.upper.reshape(-1))
                if witness:
                    components[name] = value
        append("full", blocks(point(np.eye(18)), point(np.eye(18))))
        if self.added_basis is not None:
            append("projected", blocks(self.parent_basis, self.added_basis))
        if self.probes:
            # Columns retain each declared probe separately.  The diagonal of
            # these small probe matrices equals the individual contractions;
            # batching shares directed reductions without mixing their memory.
            Bq = polynomial_project(f["Bhat"], self.probe_q)
            Ga = polynomial_project(f["Ghat_without_H"], self.probe_a)
            Gz = polynomial_project(f["Ghat_without_H"], self.probe_z)
            Tq = polynomial_project(f["T"], self.probe_q)
            Ea = polynomial_project(f["E_without_H"], self.probe_a)
            angular = _matmul(point(N), point(self.source.angular_inverse)).reshape((3, 1, 1))
            m0 = sum_interval(multiply(angular, self.probe_e0), 0)
            m1 = sum_interval(multiply(angular, self.probe_e1), 0)
            mean = _poly_add(_poly_mul(f["lam"], _constant(m0)), _constant(m1))
            matrices = dict(
                cross_parent_work_j=integrate(Bq, self.C, Gz, 1),
                cross_added_work_j=integrate(Ga, self.C, mean, 1),
                self_work_j=integrate(Ga, self.C, Gz, 2),
                stiffness_cross_bilinear_j=integrate(Bq, self.C, Ga, 1),
                stiffness_self_bilinear_j=integrate(Ga, self.C, Ga, 2),
                mass_cross_bilinear_kg_m2=integrate(Tq, eye3, Ea, 1, True),
                mass_self_bilinear_kg_m2=integrate(Ea, eye3, Ea, 2, True))
            parent_force = integrate(f["Bhat"], self.C, Gz, 1)
            added_force = add(integrate(f["Ghat_without_H"], self.C, mean, 1),
                              integrate(f["Ghat_without_H"], self.C, Gz, 2))
            for i, probe in enumerate(self.probes):
                values = {name: matrices[name][i, i] for name in PROBE_SCALARS}
                values.update(added_parent_force_n=parent_force[:, i], enrichment_force_n=added_force[:, i])
                append("probe."+probe["name"], values)
        if witness and tuple(components) != self.names:
            raise IntegralBoundRefusal("Directed integral component schema differs")
        enclosure = DirectedInterval(np.concatenate(flat_lower), np.concatenate(flat_upper))
        if enclosure.shape != (len(self.names),):
            raise IntegralBoundRefusal("Directed integral component count differs")
        return self._certificate(enclosure, _frozen_map(components), geometry, witness)

    def _certificate(self, enclosure, components, geometry, witness):
        operands = dict(source=self.source.binding_sha256, geometry=geometry.line_operand_sha256,
                        options=self.options_binding)
        binding = hashlib.sha256(json.dumps(operands, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return IntegralBoundCertificate(enclosure, self.names, components,
            self.source.binding_sha256, binding, geometry.line_operand_sha256,
            component_witness_included=bool(witness))

    def verify(self, certificate, material_line, native_interval=_UNSET):
        """Authenticate canonical bounds, including rejection of wider bounds.

        Containment alone accepts arbitrary widening.  This verifier instead
        recomputes the declared construction and requires literal equality of
        every bound, provenance operand and scope flag.
        """
        if type(certificate) is not IntegralBoundCertificate or type(certificate.component_witness_included) is not bool:
            raise IntegralBoundRefusal("Canonical integral certificate type differs")
        canonical = self.evaluate(material_line, native_interval,
                                  witness=certificate.component_witness_included)
        for field in certificate.__dataclass_fields__:
            if field not in ("enclosure", "components"):
                given, expected = getattr(certificate, field), getattr(canonical, field)
                if type(given) is not type(expected) or given != expected:
                    raise IntegralBoundRefusal("Canonical integral certificate metadata differs: "+field)
        if (not np.array_equal(certificate.lower, canonical.lower) or
                not np.array_equal(certificate.upper, canonical.upper)):
            raise IntegralBoundRefusal("Canonical integral enclosure differs")
        if tuple(certificate.components) != tuple(canonical.components):
            raise IntegralBoundRefusal("Canonical component witnesses differ")
        for name in canonical.components:
            given, expected = certificate.components[name], canonical.components[name]
            if (type(given) is not DirectedInterval or
                    not np.array_equal(given.lower, expected.lower) or
                    not np.array_equal(given.upper, expected.upper)):
                raise IntegralBoundRefusal("Canonical component witness differs: "+name)
        return True


def prepare_integral_bounds(source, raw, *, added_basis=None, parent_basis=None, probes=(), include_volumes=True):
    return PreparedIntegralBounds(source, raw, added_basis=added_basis,
        parent_basis=parent_basis, probes=probes, include_volumes=include_volumes)


def verify_integral_bound_certificate(prepared, material_line, native_interval, certificate):
    if type(prepared) is not PreparedIntegralBounds:
        raise IntegralBoundRefusal("Prepared original integral bounds are required")
    return prepared.verify(certificate, material_line, native_interval)
