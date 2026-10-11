"""Independent exact-algebra checks of full directed radial contractions."""
from dataclasses import asdict, replace
from fractions import Fraction

import numpy as np
import pytest

from tectonics.genesis_volume_geometry import ConicalCohortGeometry
from tectonics.genesis_volume_material_sections import prepare_cohort_material_section
from tectonics.genesis_volume_operator import isotropic_elasticity
from tectonics.genesis_volume_exact_lambda import prepare_fixed_N_interval
from tectonics.genesis_volume_exact_lambda_bulk import prepare_source, integrate_line, line_to_dict
from tectonics.genesis_volume_operand_bounds import DirectedInterval, point
from tectonics.genesis_volume_integral_bounds import (
    IntegralBoundRefusal, polynomial_project, polynomial_bilinear,
    polynomial_integral, prepare_integral_bounds, verify_integral_bound_certificate,
    multiply, sum_interval,
)
from analysis.genesis_volume_operand_validation.independent_coefficients import exact_source_coefficients
from analysis.genesis_volume_operand_validation.independent_contractions import (
    polynomial_project as exact_project,
    polynomial_bilinear as exact_bilinear,
    exact_polynomial_integral,
    exact_line_contractions,
)

LD = np.longdouble


def _f(value):
    return Fraction(*LD(value).as_integer_ratio())


def _rational_array(value):
    value = np.asarray(value, dtype=LD)
    return np.asarray([_f(v) for v in value.flat], object).reshape(value.shape)


def _contains_exact(bound, values):
    assert bound.shape == np.shape(values)
    for lo, hi, value in zip(bound.lower.flat, bound.upper.flat, np.asarray(values, object).flat):
        assert _f(lo) <= value <= _f(hi)


def _source(radius=100., lower=.8, upper=.95, displacement=False):
    X = np.array(((0., -5., 10.), (5., -5., 10.), (0., 5., 10.)))
    volume = np.linalg.det(X)*(upper**3-lower**3)/6
    cohort = ConicalCohortGeometry(17, 0, 2, (1-upper**3)/.5,
        (1-lower**3)/.5, 2*volume, volume, tuple(map(tuple, X)),
        upper, lower, (), "unchanged-original-owner", "unchanged-original-material")
    dofs = np.zeros((3, 6))
    if displacement:
        dofs[:, :3] = np.array(((.0003, -.0004, .0001),
            (-.0001, .0002, -.0003), (.0004, .0003, -.0002)))
        dofs[:, 3:] = np.array(((.0001, -.0003, .0002),
            (.0002, .0004, -.0001), (-.0003, .0001, .0004)))
    section = prepare_cohort_material_section(cohort, dofs,
        (0., 0., 1.), 9.03, conical_mass_coefficient=.5)
    raw = dict(section=asdict(section), elasticity_pa=isotropic_elasticity(3e8, .27),
        basis_xyz=np.array(((0., -1., 0.), (1., 0., 0.), (0., 0., 1.))),
        memory0=(np.arange(18).reshape(3, 6)-8)*1e-4,
        memory1=(np.arange(18).reshape(3, 6)[::-1]-7)*2e-4,
        seed_center_world_m=(1., 0., 9.), seed_radius_m=radius,
        source_identity="original-source")
    return raw, prepare_source(raw)


def _options(raw):
    W = np.zeros((18, 3))
    W[[0, 7, 14], [0, 1, 2]] = 1
    W[3, 0] = -.125
    V = np.zeros((18, 2))
    V[[2, 8], [0, 1]] = 1
    V[6, 1] = .25
    q = (np.arange(18)-8)*1e-6
    a = (np.arange(18)[::-1]-9)*2e-7
    probes = [dict(name="original-memory", q=q, a=a, mean_z=q*.2),
        dict(name="accepted-mean-memory", q=q*.3, a=a*.4, mean_z=a*.7,
             mean_e0=np.asarray(raw["memory0"])*.37,
             mean_e1=np.asarray(raw["memory1"])*-.43)]
    return dict(added_basis=W, parent_basis=V, probes=probes)


@pytest.mark.parametrize("seed", [0, 1, 4])
def test_directed_projection_bilinear_and_primitive_enclose_independent_exact_algebra(seed):
    rng = np.random.default_rng(seed)
    left = rng.normal(size=(3, 3, 4)).astype(LD)*LD("0.03")
    right = rng.normal(size=(4, 3, 2)).astype(LD)*LD("0.07")
    middle = rng.normal(size=(3, 3)).astype(LD)*LD("1000000.13")
    basis = rng.normal(size=(4, 3)).astype(LD)
    projection = polynomial_project(point(left), basis)
    _contains_exact(projection, exact_project(_rational_array(left), _rational_array(basis)))
    polynomial = polynomial_bilinear(left, middle, right)
    exact = exact_bilinear(_rational_array(left), _rational_array(middle), _rational_array(right))
    _contains_exact(polynomial, exact)
    moments = np.array([LD("0.13"), LD("-0.17"), LD("0.021"), LD("0.018"), LD("-0.009"), LD("0.003")])
    _contains_exact(polynomial_integral(polynomial, moments),
                    exact_polynomial_integral(exact, _rational_array(moments)))


def test_bilinear_shape_and_primitive_degree_refusals():
    with pytest.raises(IntegralBoundRefusal, match="projection shape"):
        polynomial_project(np.ones((2, 3, 4)), np.ones((5, 2)))
    with pytest.raises(IntegralBoundRefusal, match="bilinear shape"):
        polynomial_bilinear(np.ones((2, 3, 4)), np.eye(2), np.ones((2, 3, 4)))
    with pytest.raises(IntegralBoundRefusal, match="moment shape"):
        polynomial_integral(np.ones((5, 3)), np.ones(4))


@pytest.mark.parametrize("values", [
    [LD("1e-2000"), LD("-1e-2000"), LD("3e-2040")],
    [LD("1.0000000000000000001"), LD("-1"), LD("0.0000000000000000001")],
    [LD(0), LD(0), LD(0)],
])
def test_fast_tensor_reduction_preserves_outward_elementary_theorem(values):
    _contains_exact(sum_interval(point(values), 0), sum(map(_f, values), Fraction()))
    for value in values:
        for multiplier in (LD("1.23"), LD("-0.91"), LD(0), LD(1), LD(-1)):
            _contains_exact(multiply(value, multiplier), _f(value)*_f(multiplier))


@pytest.mark.parametrize("N", [(.31, .2, .49), (.27, .28, .45), (.37, .17, .46)])
@pytest.mark.parametrize("displacement", [False, True])
def test_full_projected_and_mean_memory_probe_enclosures_match_independent_source_contractions(N, displacement):
    raw, source = _source(displacement=displacement)
    options = _options(raw)
    N = np.asarray(N, dtype=LD)
    N /= N.sum()
    native = integrate_line(source, N, **options)
    assert native.interval is not None
    prepared = prepare_integral_bounds(source, raw, include_volumes=False, **options)
    bound = prepared.evaluate(native.interval)
    assert bound.names == native.names
    witness = line_to_dict(native)
    coefficients = exact_source_coefficients(raw, N, native.interval.lambda_mid, native.interval.lambda_halfwidth)
    expected = exact_line_contractions(raw, N, witness, coefficients=coefficients,
        gamma_mode="literal_source", weight_mode="literal_source", **options)
    # The large sphere contains the whole original cohort, so the independently
    # certified source endpoints equal the exact original cohort endpoints.
    _contains_exact(bound.enclosure, np.array([expected[name] for name in bound.names], object))
    errors = bound.error_bounds_for(native.values, native.names)
    for value, exact, error in zip(native.values, (expected[name] for name in bound.names), errors):
        assert abs(_f(value)-exact) <= _f(error)
    assert bound.radial_integral_accuracy_certified
    assert bound.source_endpoint_transport_certified
    assert not bound.historical_saved_budget_authentication_certified
    assert not bound.angular_integral_convergence_certified
    assert not bound.physical_contraction_certified
    assert not bound.available_for_creation_funding


def test_reference_volume_banks_enclose_exact_cohort_measure_and_preserve_zero_bank():
    raw, source = _source()
    N = np.asarray((.31, .2, .49), dtype=LD)
    N /= N.sum()
    bound = prepare_integral_bounds(source, raw).evaluate(N)
    assert bound.names[:2] == ("reference_volume_negative_m3", "reference_volume_positive_m3")
    triangle = _rational_array(source.section.triangle_m)
    determinant = sum((triangle[0,j]*(triangle[1,(j+1)%3]*triangle[2,(j+2)%3]-
                                    triangle[1,(j+2)%3]*triangle[2,(j+1)%3]) for j in range(3)), Fraction())
    lower, upper = map(_f, source.section.lambda_bounds)
    n = list(map(_f, N))
    C = _rational_array(source.geometry.birth_C_m)
    D = _rational_array(source.geometry.birth_D_m)
    normal = _rational_array(source.section.normal_xyz)
    u = np.asarray(n, object) @ C
    d = np.asarray(n, object) @ D
    root = (_f(source.section.offset_m)-sum(normal*d, Fraction()))/sum(normal*u, Fraction())
    assert lower < root < upper
    _contains_exact(bound.components[bound.names[0]], determinant*(root**3-lower**3)/3)
    _contains_exact(bound.components[bound.names[1]], determinant*(upper**3-root**3)/3)


def test_source_binding_declared_operand_freezing_and_component_error_schema():
    raw, source = _source()
    options = _options(raw)
    prepared = prepare_integral_bounds(source, raw, **options)
    N = np.asarray((.31, .2, .49), dtype=LD)
    N /= N.sum()
    before = prepared.evaluate(N)
    options["added_basis"][:] = 999
    options["probes"][1]["mean_e0"][:] = 999
    after = prepared.evaluate(N)
    assert before.line_operand_sha256 == after.line_operand_sha256
    assert np.array_equal(before.lower, after.lower)
    assert np.array_equal(before.upper, after.upper)
    with pytest.raises(AttributeError, match="immutable"):
        prepared.include_volumes = False
    with pytest.raises(ValueError):
        before.lower.setflags(write=True)
    with pytest.raises(TypeError):
        before.components["new"] = point(0)
    with pytest.raises(IntegralBoundRefusal, match="schema"):
        before.error_bounds_for(np.zeros(len(before.names)), ("wrong",))
    changed = dict(raw, seed_radius_m=99.)
    with pytest.raises(IntegralBoundRefusal, match="Literal source"):
        prepare_integral_bounds(source, changed)
    _, other = _source(radius=99.)
    interval = prepare_fixed_N_interval(other.geometry, N)
    with pytest.raises(IntegralBoundRefusal, match="not bound"):
        prepared.evaluate(interval)


def test_complete_original_schema_and_empty_material_line_are_enclosed_without_history_effects():
    raw, source = _source(radius=.25)
    options = _options(raw)
    prepared = prepare_integral_bounds(source, raw, **options)
    empty = prepared.evaluate((1., 0., 0.))
    assert len(empty.names) == 2+1314+33+2*43
    assert np.array_equal(empty.lower, np.zeros(len(empty.names)))
    assert np.array_equal(empty.upper, np.zeros(len(empty.names)))
    assert not empty.available_for_creation_funding
    assert not empty.physical_contraction_certified


def test_canonical_certificate_recomputation_rejects_inflation_and_scope_promotion():
    raw, source = _source()
    prepared = prepare_integral_bounds(source, raw, **_options(raw))
    N = np.asarray((.31, .2, .49), dtype=LD)
    N /= N.sum()
    interval = prepare_fixed_N_interval(source.geometry, N)
    certificate = prepared.evaluate(N, interval, witness=False)
    detailed = prepared.evaluate(N, interval)
    assert np.array_equal(certificate.lower, detailed.lower)
    assert np.array_equal(certificate.upper, detailed.upper)
    assert certificate.names == detailed.names
    assert certificate.line_operand_sha256 == detailed.line_operand_sha256
    assert not certificate.component_witness_included
    assert not certificate.components
    assert verify_integral_bound_certificate(prepared, N, interval, certificate)
    inflated = replace(certificate, enclosure=DirectedInterval(
        np.nextafter(certificate.lower, LD(-np.inf)),
        np.nextafter(certificate.upper, LD(np.inf))))
    with pytest.raises(IntegralBoundRefusal, match="enclosure differs"):
        prepared.verify(inflated, N, interval)
    for field in ("physical_contraction_certified", "available_for_creation_funding",
                  "angular_integral_convergence_certified", "historical_saved_budget_authentication_certified"):
        with pytest.raises(IntegralBoundRefusal, match="metadata differs"):
            prepared.verify(replace(certificate, **{field: True}), N, interval)
    with pytest.raises(IntegralBoundRefusal, match="metadata differs"):
        prepared.verify(replace(certificate, scope="planetary_accuracy"), N, interval)
    with pytest.raises(IntegralBoundRefusal, match="metadata differs"):
        prepared.verify(replace(certificate, radial_integral_accuracy_certified=1), N, interval)
