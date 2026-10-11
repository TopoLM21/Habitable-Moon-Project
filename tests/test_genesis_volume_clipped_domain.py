"""Geometry and integral checks for the original-cohort/ball intersection.

These are declared, metre-scale computational cells.  The analytic volumes
test geometric clipping before integration; they do not select a Genesis
fault or establish a physical bulk-operator accuracy certificate.
"""
from dataclasses import replace
import math

import numpy as np
import pytest

from tectonics.genesis_volume_geometry import ConicalCohortGeometry
from tectonics.genesis_volume_material_sections import prepare_cohort_material_section
from tectonics.genesis_volume_clipped_domain import (
    ClippedDomainRefusal,
    clipped_domain_to_dict,
    prepare_cohort_clipped_domain,
    validate_cohort_clipped_domain,
)


X = np.array(((0., -5., 10.), (5., -5., 10.), (0., 5., 10.)))
LAMBDA_LO, LAMBDA_HI, COEFFICIENT = .8, .95, .5
RADIUS = .25


def _section(displacement=None, normal=(0., 1., 0.), offset=0.):
    """A radial slab whose angular boundary x=0 cuts a ball analytically."""
    qlo = (1-LAMBDA_HI**3)/COEFFICIENT
    qhi = (1-LAMBDA_LO**3)/COEFFICIENT
    volume = np.linalg.det(X)*(LAMBDA_HI**3-LAMBDA_LO**3)/6
    cohort = ConicalCohortGeometry(17, 0, 2, qlo, qhi, 2*volume, volume,
        tuple(map(tuple, X)), LAMBDA_HI, LAMBDA_LO, (),
        "original-computational-owner", "original-computational-material")
    if displacement is None:
        displacement = np.zeros((3, 6))
    return prepare_cohort_material_section(cohort, displacement, normal, offset,
        conical_mass_coefficient=COEFFICIENT)


def _domain(section=None, center=(1., 0., 9.), radius=RADIUS, order=10, **kwargs):
    if section is None:
        section = _section()
    return prepare_cohort_clipped_domain(section, center, radius,
        radial_order=order, angular_order=order, ray_order=order, **kwargs)


def _assert_inside(domain, section, center, radius):
    N, lam = domain.barycentric, domain.radial_scale
    count = len(lam)
    assert N.shape == (count, 3)
    assert domain.centered_birth_offsets_m.shape == (count, 3)
    assert domain.reference_volume_weights_m3.shape == (count,)
    assert domain.birth_volume_weights_m3.shape == (count,)
    assert domain.side.shape == (count,)
    assert np.all(N >= 0.)
    np.testing.assert_allclose(N.sum(axis=1), 1., rtol=0, atol=8e-15)
    assert np.all((lam >= section.lambda_bounds[0]) & (lam <= section.lambda_bounds[1]))
    assert np.all(domain.reference_volume_weights_m3 > 0.)
    assert np.all(domain.birth_volume_weights_m3 > 0.)
    assert set(domain.side) <= {-.5, .5}
    h = np.einsum("pi,pij->pj", N,
        lam[:, None, None]*(section.triangle_m+section.slope_P_m)[None, :, :]
        +(section.displacement_A_m-section.slope_P_m)[None, :, :])
    np.testing.assert_allclose(domain.centered_birth_offsets_m, h-np.asarray(center),
                               rtol=0, atol=3e-14)
    assert np.all(np.linalg.norm(domain.centered_birth_offsets_m, axis=1) <= radius*(1+2e-13))
    signed = h@section.normal_xyz-section.offset_m
    assert np.all(signed[domain.side == .5] >= -2e-13)
    assert np.all(signed[domain.side == -.5] <= 2e-13)


def _moments(domain):
    weights = domain.birth_volume_weights_m3
    offsets = domain.centered_birth_offsets_m
    volume = float(weights.sum())
    first = np.einsum("p,pi->i", weights, offsets)/volume
    second = np.einsum("p,pi,pj->ij", weights, offsets, offsets)/volume
    return volume, first, second


@pytest.mark.parametrize("center,fraction,mean", (
    ((1., 0., 9.), 1., (0., 0., 0.)),
    ((1., 0., 8.), .5, (0., 0., 3*RADIUS/8)),
    ((0., 0., 9.), .5, (3*RADIUS/8, 0., 0.)),
    ((0., 0., 8.), .25, (3*RADIUS/8, 0., 3*RADIUS/8)),
))
def test_analytic_ball_radial_cap_angular_cap_and_corner_intersections(center, fraction, mean):
    section = _section()
    domain = _domain(section, center, order=12, require_both_sides=True)
    _assert_inside(domain, section, center, RADIUS)
    volume, first, second = _moments(domain)
    expected = fraction*4*math.pi*RADIUS**3/3
    assert volume == pytest.approx(expected, rel=2e-11, abs=2e-14)
    # A hemisphere's x moment has a (R²-z²)^(3/2) radial integrand.  Finite
    # Gaussian order is approximate even when its volume is polynomial.
    np.testing.assert_allclose(first, mean, rtol=2e-5, atol=3e-11)
    np.testing.assert_allclose(np.diag(second), RADIUS**2/5, rtol=0, atol=3e-11)
    assert set(domain.side) == {-.5, .5}
    for side in (-.5, .5):
        select = domain.side == side
        assert domain.birth_volume_weights_m3[select].sum() == pytest.approx(expected/2, rel=2e-11)
    assert domain.lambda_event_records
    assert domain.topology_records


def test_offset_radial_slab_has_analytic_volume_and_first_moment():
    # The ball is clipped at centered z=h, retaining z >= h.  Its moving
    # circle/triangle intersection is resolved before quadrature nodes exist.
    h = RADIUS/3
    center = (1., 0., 8.-h)
    domain = _domain(center=center)
    volume, first, _ = _moments(domain)
    expected = math.pi*(2*RADIUS**3/3-RADIUS**2*h+h**3/3)
    first_z_integral = math.pi*(RADIUS**2-h*h)**2/4
    assert volume == pytest.approx(expected, rel=2e-11)
    np.testing.assert_allclose(first, (0., 0., first_z_integral/expected), rtol=0, atol=3e-11)


def test_nonpolynomial_hemisphere_first_moment_converges_to_analytic_value():
    values = [_moments(_domain(center=(0., 0., 9.), order=order))[1][0]
              for order in (4, 8, 12)]
    errors = np.abs(np.asarray(values)-3*RADIUS/8)
    assert errors[0] > 1e-6
    assert errors[1] < .4*errors[0]
    assert errors[2] < .4*errors[1]
    assert errors[2] < 1e-5*(3*RADIUS/8)


def test_offset_angular_cap_volume_converges_to_analytic_ball_cap():
    # The cohort x >= 0 boundary is off the ball center.  Circle/chord
    # topology changes with lambda, so neither angular nor radial node
    # masking or an assumed half-ball volume can pass this test.
    center = (.08, 0., 9.)
    h = -center[0]
    expected = math.pi*(2*RADIUS**3/3-RADIUS**2*h+h**3/3)
    domains = [_domain(center=center, order=order) for order in (4, 8, 12)]
    errors = np.abs(np.asarray([_moments(domain)[0] for domain in domains])-expected)
    for domain in domains:
        _assert_inside(domain, _section(), center, RADIUS)
    assert errors[0] > 1e-8
    assert errors[1] < .5*errors[0]
    assert errors[2] < .5*errors[1]
    assert errors[2] < 5e-5*expected


def test_affine_full_A_P_chart_preserves_exact_birth_ball_and_reference_volume():
    transform = np.array(((1.1, .07, 0.), (.02, .9, 0.), (0., 0., 1.06)))
    translation = np.array((.12, -.07, .04))
    P = X@(transform-np.eye(3)).T
    A = P+translation
    displacement = np.column_stack((A, P))
    center = transform@np.array((1., 0., 9.))+translation
    # The transformed lambda slices retain exact z alignment.  Gamma y=cy
    # bisects each disk; the source still carries all six A/P components.
    normal = np.array((0., 1., 0.))
    section = _section(displacement, normal, float(normal@center))
    source_binding = section.binding_sha256
    source_arrays = [array.tobytes() for array in
        (section.triangle_m, section.displacement_A_m, section.slope_P_m)]
    domain = _domain(section, center, order=12)
    _assert_inside(domain, section, center, RADIUS)
    volume, first, second = _moments(domain)
    expected = 4*math.pi*RADIUS**3/3
    assert volume == pytest.approx(expected, rel=5e-11)
    assert domain.reference_volume_weights_m3.sum() == pytest.approx(expected/np.linalg.det(transform), rel=5e-11)
    np.testing.assert_allclose(domain.birth_volume_weights_m3,
        np.linalg.det(transform)*domain.reference_volume_weights_m3, rtol=2e-13, atol=0)
    np.testing.assert_allclose(first, 0., rtol=0, atol=2e-10)
    np.testing.assert_allclose(second, np.eye(3)*RADIUS**2/5, rtol=0, atol=2e-10)
    assert section.binding_sha256 == source_binding
    assert source_arrays == [array.tobytes() for array in
        (section.triangle_m, section.displacement_A_m, section.slope_P_m)]
    assert np.linalg.norm(section.displacement_A_m) > 0
    assert np.linalg.norm(section.slope_P_m) > 0


def test_heterogeneous_A_P_weights_use_same_material_point_jacobians():
    displacement = np.array(((.03, -.02, .01, -.015, .01, .003),
        (-.01, .02, -.02, -.02, -.01, .007),
        (.02, .01, .03, .01, -.02, -.005)))
    N0, lam0 = np.array((.3, .3, .4)), .9
    center = N0@(lam0*(X+displacement[:, 3:])+displacement[:, :3]-displacement[:, 3:])
    section = _section(displacement, offset=float(center[1]))
    domain = _domain(section, center, order=12)
    _assert_inside(domain, section, center, RADIUS)
    N, lam = domain.barycentric, domain.radial_scale
    vertices = lam[:, None, None]*(X+section.slope_P_m)[None, :, :]+(
        section.displacement_A_m-section.slope_P_m)[None, :, :]
    d1, d2 = vertices[:, 1]-vertices[:, 0], vertices[:, 2]-vertices[:, 0]
    dl = N@(X+section.slope_P_m)
    birth_det = np.einsum("pi,pi->p", np.cross(d1, d2), dl)
    reference_det = lam**2*np.linalg.det(X)
    assert np.all(birth_det > 0.)
    np.testing.assert_allclose(domain.birth_volume_weights_m3,
        domain.reference_volume_weights_m3*birth_det/reference_det, rtol=3e-13, atol=0)
    expected = 4*math.pi*RADIUS**3/3
    # Generic A/P and Gamma orientation create nonpolynomial slice/chord
    # integrals.  Pointwise Jacobians above remain checked to roundoff;
    # their integrals require an explicit order sequence below.
    lower = [_domain(section, center, order=order) for order in (4, 8)]
    errors = np.abs(np.asarray([_moments(item)[0] for item in (*lower, domain)])-expected)
    assert errors[0] > 1e-8
    assert errors[1] < .5*errors[0]
    assert errors[2] < .5*errors[1]
    assert errors[2] < 3e-5*expected


def test_raw_empty_and_single_side_domains_do_not_manufacture_a_second_side():
    empty = _domain(center=(100., 100., 100.))
    assert len(empty.radial_scale) == 0
    assert empty.barycentric.shape == (0, 3)
    assert empty.centered_birth_offsets_m.shape == (0, 3)
    single = _domain(_section(offset=-2.))
    assert set(single.side) == {.5}
    assert single.birth_volume_weights_m3.sum() == pytest.approx(4*math.pi*RADIUS**3/3, rel=2e-11)
    with pytest.raises(ClippedDomainRefusal):
        _domain(_section(offset=-2.), require_both_sides=True)
    with pytest.raises(ClippedDomainRefusal):
        _domain(center=(100., 100., 100.), require_both_sides=True)


def test_quadrature_is_immutable_and_replay_binding_detects_changed_weights():
    domain = _domain()
    validate_cohort_clipped_domain(domain)
    raw = clipped_domain_to_dict(domain)
    assert isinstance(raw, dict)
    for name in ("physical_creation_admitted", "physical_release_admitted",
                 "physical_force_work_admission", "available_for_creation_funding",
                 "numerical_integral_convergence_certified"):
        assert raw[name] is False
    assert domain.section is not None
    assert len(domain.binding_sha256) == 64
    for array in (domain.barycentric, domain.radial_scale, domain.reference_volume_weights_m3,
                  domain.birth_volume_weights_m3, domain.side, domain.centered_birth_offsets_m):
        assert not array.flags.writeable
    changed = np.array(domain.reference_volume_weights_m3)
    changed[0] *= 1.01
    forged = replace(domain, reference_volume_weights_m3=changed)
    with pytest.raises((ClippedDomainRefusal, ValueError)):
        validate_cohort_clipped_domain(forged)
    with pytest.raises((ClippedDomainRefusal, ValueError)):
        clipped_domain_to_dict(forged)


def test_live_longdouble_operand_binding_detects_change_hidden_by_float64_archive():
    if np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        pytest.skip("This platform has no finer longdouble operand precision")
    domain = _domain(order=4)
    original = domain.reference_volume_weights_m3
    changed = np.array(original, dtype=np.longdouble)
    for index, value in enumerate(original):
        candidate = np.nextafter(np.longdouble(value), np.longdouble(np.inf))
        if float(candidate) == float(value) and candidate != value:
            changed[index] = candidate
            break
    else:
        raise AssertionError("No sub-float64 live operand perturbation was represented")
    # The portable literal float64 table remains byte-exact: only the live
    # arithmetic operand changed, so an ordinary DTO hash cannot detect it.
    np.testing.assert_array_equal(np.asarray(changed, float), np.asarray(original, float))
    forged = replace(domain, reference_volume_weights_m3=changed)
    with pytest.raises(ClippedDomainRefusal, match="live numeric operands"):
        validate_cohort_clipped_domain(forged)


def test_birth_chart_with_reversed_material_orientation_is_refused():
    displacement = np.column_stack((-2*X, -2*X))
    section = _section(displacement)
    with pytest.raises(ClippedDomainRefusal):
        _domain(section)


def test_ball_smaller_than_represented_world_coordinate_resolution_is_refused():
    with pytest.raises(ClippedDomainRefusal, match="position resolution"):
        _domain(center=(1e9, 0., 9.), radius=1e-8)


@pytest.mark.parametrize("kwargs", (
    {"radial_order": 0}, {"angular_order": True}, {"ray_order": 2.5},
    {"max_points": 1}, {"max_lambda_events": 1},
    {"max_points": True}, {"max_lambda_events": 1.5},
))
def test_invalid_orders_and_finite_resource_guards_refuse(kwargs):
    with pytest.raises((ClippedDomainRefusal, ValueError)):
        prepare_cohort_clipped_domain(_section(), (1., 0., 9.), RADIUS, **kwargs)


@pytest.mark.parametrize("center,radius", (
    ((1., 0.), RADIUS), ((1., np.nan, 9.), RADIUS),
    ((1.+0j, 0., 9.), RADIUS), ((1., 0., 9.), 0.),
    ((1., 0., 9.), -1.), ((1., 0., 9.), float("inf")),
    ((1., 0., 9.), True),
))
def test_invalid_ball_inputs_refuse(center, radius):
    with pytest.raises((ClippedDomainRefusal, ValueError)):
        prepare_cohort_clipped_domain(_section(), center, radius)


def test_modified_original_section_binding_is_refused():
    section = _section()
    changed = replace(section, offset_m=section.offset_m+.01)
    with pytest.raises((ClippedDomainRefusal, ValueError)):
        _domain(changed)
