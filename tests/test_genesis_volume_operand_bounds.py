"""Exact-rational checks of the directed arbitrary-line coefficient enclosure."""
from dataclasses import asdict
from fractions import Fraction

import numpy as np
import pytest

from tectonics.genesis_volume_geometry import ConicalCohortGeometry
from tectonics.genesis_volume_material_sections import prepare_cohort_material_section
from tectonics.genesis_volume_operator import isotropic_elasticity
from tectonics.genesis_volume_exact_lambda import prepare_fixed_N_interval
from tectonics.genesis_volume_exact_lambda_bulk import prepare_source, integrate_line
from tectonics.genesis_volume_operand_bounds import (
    DirectedInterval, OperandBoundRefusal, point, add, subtract, multiply, divide,
    inverse3, rational_enclosure, prepare_operand_bounds,
)
from analysis.genesis_volume_operand_validation.independent_coefficients import (
    exact_source_coefficients,
)

LD = np.longdouble


def _f(value):
    return Fraction(*LD(value).as_integer_ratio())


def _contains_fraction(bound, value):
    assert bound.shape == np.shape(value)
    for lo, hi, exact in zip(bound.lower.flat, bound.upper.flat, np.asarray(value, object).flat):
        assert _f(lo) <= exact <= _f(hi)


def _source(*, lower=.8, upper=.95, radius=.25, displacement=None, rotation=False):
    X = np.array(((0., -5., 10.), (5., -5., 10.), (0., 5., 10.)))
    volume = np.linalg.det(X)*(upper**3-lower**3)/6
    cohort = ConicalCohortGeometry(17, 0, 2, (1-upper**3)/.5,
        (1-lower**3)/.5, 2*volume, volume, tuple(map(tuple, X)),
        upper, lower, (), "unchanged-original-owner", "unchanged-original-material")
    section = prepare_cohort_material_section(cohort,
        np.zeros((3, 6)) if displacement is None else displacement,
        (0., 0., 1.), 9.03, conical_mass_coefficient=.5)
    e0 = (np.arange(18).reshape(3, 6)-8)*1e-4
    e1 = (np.arange(18).reshape(3, 6)[::-1]-7)*2e-4
    Q = np.array(((0., -1., 0.), (1., 0., 0.), (0., 0., 1.))) if rotation else np.eye(3)
    raw = dict(section=asdict(section), elasticity_pa=isotropic_elasticity(3e8, .27),
        basis_xyz=Q, memory0=e0, memory1=e1, seed_center_world_m=(1., 0., 9.),
        seed_radius_m=radius, source_identity="original-source")
    return raw, prepare_source(raw)


@pytest.mark.parametrize("operation, exact", [
    (add, lambda a, b: a+b), (subtract, lambda a, b: a-b),
    (multiply, lambda a, b: a*b), (divide, lambda a, b: a/b),
])
@pytest.mark.parametrize("a,b", [
    (LD("1.0000000000000000001"), LD("0.9999999999999999999")),
    (LD("1e-2000"), LD("3e-1700")),
    (LD("-2.731000000000000001"), LD("0.100000000000000001")),
])
def test_directed_elementary_operation_encloses_exact_binary_operands(operation, exact, a, b):
    _contains_fraction(operation(a, b), exact(_f(a), _f(b)))


def test_interval_arithmetic_encloses_all_endpoint_choices_and_preserves_exact_zero():
    a, b = DirectedInterval(-2, 3), DirectedInterval(4, 7)
    for operation, exact in ((add, lambda x, y:x+y), (subtract, lambda x, y:x-y),
                             (multiply, lambda x, y:x*y), (divide, lambda x, y:x/y)):
        bound = operation(a, b)
        for x in (Fraction(-2), Fraction(3)):
            for y in (Fraction(4), Fraction(7)):
                _contains_fraction(bound, exact(x, y))
    zero = multiply(a, 0)
    assert zero.lower == zero.upper == 0
    assert zero.absolute_error_for(np.asarray(0, dtype=LD)) == 0
    same = multiply(a, 1)
    assert same.lower == a.lower and same.upper == a.upper
    negative = multiply(-1, a)
    assert negative.lower == -a.upper and negative.upper == -a.lower
    with pytest.raises(OperandBoundRefusal, match="contains zero"):
        divide(1, DirectedInterval(-1, 1))


@pytest.mark.parametrize("value", [Fraction(1, 3), Fraction(-7, 19),
    Fraction(12345678901234567890123456789, 2**97), Fraction(1, 2**16440), Fraction()])
def test_rational_leaf_bracket_uses_exact_comparison(value):
    _contains_fraction(rational_enclosure(value), value)


def test_interval_inverse_encloses_exact_inverse_with_cancellation():
    from analysis.genesis_volume_operand_validation.independent_coefficients import _inverse3
    X = np.array(((LD("2000000.125"), LD("2000000.126"), 1),
                  (LD("-3000000.125"), LD("-3000000.124"), 2),
                  (LD("1000000.0625"), LD("1000000.0665"), 3)), dtype=LD)
    exact = _inverse3(np.asarray([_f(v) for v in X.flat], dtype=object).reshape(3, 3))
    _contains_fraction(inverse3(X), exact)
    with pytest.raises(OperandBoundRefusal, match="contains zero"):
        inverse3(np.ones((3, 3), dtype=LD))


@pytest.mark.parametrize("N", [(.31, .20, .49), (.3, .21, .49), (.32, .2, .48)])
@pytest.mark.parametrize("rotation,displaced", [(False, False), (True, False), (False, True)])
def test_full_original_fields_enclose_independent_exact_algebra_at_arbitrary_N(N, rotation, displaced):
    displacement = np.zeros((3, 6))
    if displaced:
        displacement[:, :3] = np.array(((.0003, -.0004, .0001),
            (-.0001, .0002, -.0003), (.0004, .0003, -.0002)))
        displacement[:, 3:] = np.array(((.0001, -.0003, .0002),
            (.0002, .0004, -.0001), (-.0003, .0001, .0004)))
    raw, source = _source(rotation=rotation, displacement=displacement)
    N = np.asarray(N, dtype=LD)
    N /= N.sum()
    interval = prepare_fixed_N_interval(source.geometry, N)
    assert interval is not None
    certificate = prepare_operand_bounds(source).evaluate(interval)
    exact = exact_source_coefficients(raw, N, interval.lambda_mid, interval.lambda_halfwidth)
    for name, coefficients in exact.items():
        _contains_fraction(certificate.fields[name], coefficients)
    line = integrate_line(source, N)
    required = certificate.error_bounds_for(line.coefficients)
    for name, coefficients in exact.items():
        for value, want, bound in zip(line.coefficients[name].flat, coefficients.flat, required[name].flat):
            assert abs(_f(value)-want) <= _f(bound)
    assert certificate.coefficient_construction_accuracy_certified
    assert not certificate.geometry_operand_accuracy_certified
    assert not certificate.radial_integral_accuracy_certified
    assert not certificate.angular_integral_convergence_certified
    assert not certificate.physical_contraction_certified
    assert not certificate.available_for_creation_funding


def test_source_binding_and_immutable_intervals_are_enforced():
    _, source = _source()
    N = np.asarray((.31, .2, .49), dtype=LD)
    N /= N.sum()
    interval = prepare_fixed_N_interval(source.geometry, N)
    prepared = prepare_operand_bounds(source)
    bounds = prepared.evaluate(interval)
    assert bounds.source_binding_sha256 == source.binding_sha256
    assert len(bounds.line_operand_sha256) == 64
    assert prepared.evaluate(interval).line_operand_sha256 == bounds.line_operand_sha256
    with pytest.raises(AttributeError, match="immutable"):
        prepared.C = point(np.ones((3, 3)))
    with pytest.raises(ValueError):
        bounds.fields["L"].lower.setflags(write=True)
    with pytest.raises(TypeError):
        bounds.fields["L"] = point(0)
    _, other = _source(radius=.3)
    with pytest.raises(OperandBoundRefusal, match="not bound"):
        prepared.evaluate(prepare_fixed_N_interval(other.geometry, N))
    with pytest.raises(OperandBoundRefusal, match="Complete original"):
        bounds.error_bounds_for({"w": np.asarray([0])})


def test_directed_runtime_does_not_certify_old_producer_bound_by_default():
    _, source = _source()
    N = np.asarray((.31, .2, .49), dtype=LD)
    N /= N.sum()
    line = integrate_line(source, N)
    certificate = prepare_operand_bounds(source).evaluate(line.interval)
    errors = certificate.error_bounds_for(line.coefficients)
    # This is an independent enclosure width, not a manufactured assertion
    # that every historical producer budget contains that width.
    assert all(np.isfinite(v).all() and np.all(v >= 0) for v in errors.values())
    assert sum(v.size for v in errors.values()) == 1662
