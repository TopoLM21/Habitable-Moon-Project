"""Small field arithmetic regressions; these fixtures are not planet states."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tectonics.genesis_volume_release_quadrature import (
    RestrictedSourceQuadratureRefusal, prepare_restricted_source_quadrature,
    restricted_point_gram, restricted_source_quadrature_to_dict,
)


def weak_fields():
    # Two independent displacements have a difference far below the precision
    # of their rounded Gram, although the original point field resolves it.
    epsilon = 1e-12
    operator = np.zeros((1, 6, 18))
    operator[0, 0, :2] = 1.
    operator[0, 1, 1] = epsilon
    trace = operator[:, :3].copy()
    basis = np.zeros((18, 1))
    basis[:2, 0] = (-1., 1.)/np.sqrt(2.)
    local = SimpleNamespace(enrichment_strain_operator=operator,
        enrichment_trace=trace, parent_strain_operator=np.zeros_like(operator),
        parent_trace=np.zeros_like(trace), reference_volume_weights_m3=np.ones(1),
        elasticity_pa=np.eye(6), section=SimpleNamespace(mass_kg=2., volume_m3=1.))
    context = SimpleNamespace(partition=SimpleNamespace(local=local, binding_sha256='a'*64),
        observable_force_basis=basis, cohesive=SimpleNamespace(
            compact_jump_trace_local=trace, reference_area_weights_m2=np.ones(1)))
    return context, epsilon


def test_before_Gram_contraction_preserves_energy_that_rounded_Gram_erases():
    context, epsilon = weak_fields()
    field = context.partition.local.enrichment_strain_operator[0]
    basis = context.observable_force_basis
    rounded = field.T@field
    assert float((basis.T@rounded@basis)[0, 0]) == 0.
    quadrature = prepare_restricted_source_quadrature(context)
    expected = np.longdouble(epsilon)**2*np.longdouble(basis[1, 0])**2
    assert quadrature.stiffness_enriched[0, 0] > 0.
    assert quadrature.stiffness_enriched[0, 0] == pytest.approx(expected, rel=2e-18)
    assert quadrature.mass_enriched[0, 0] == pytest.approx(2*expected, rel=2e-18)
    assert quadrature.surface_gram[0, 0] == pytest.approx(expected, rel=2e-18)


def test_signed_cross_term_keeps_its_sign_and_is_not_an_energy_credit():
    left = np.array([[[1.], [2.]], [[-3.], [4.]]])
    right = np.array([[[-2.], [-1.]], [[5.], [-6.]]])
    weights = np.array((.5, 2.))
    expected = .5*(-2.-2.)+2*(-15.-24.)
    assert restricted_point_gram(left, right, weights)[0, 0] == expected < 0.
    context, _ = weak_fields()
    descriptor = restricted_source_quadrature_to_dict(prepare_restricted_source_quadrature(context))
    assert descriptor['funding_admitted'] is False
    assert descriptor['weak_geometry_accuracy_certified'] is False


def test_zero_fields_have_exact_zero_Gram_without_regularization():
    field = np.zeros((2, 3, 4))
    assert np.array_equal(restricted_point_gram(field, field, [1., 2.]), np.zeros((4, 4)))


@pytest.mark.parametrize('weight_scale', [1e-150, 1e150])
def test_change_of_weight_units_preserves_small_positive_field_work(weight_scale):
    field = np.array([[[1e-100]], [[2e-100]]])
    weights = np.array((weight_scale, 3*weight_scale))
    actual = restricted_point_gram(field, field, weights)[0, 0]
    expected = (np.longdouble(weights[0])+4*np.longdouble(weights[1]))*np.longdouble(1e-100)**2
    assert actual > 0.
    assert abs(actual-expected) <= 3e-18*expected


@pytest.mark.parametrize('left,right,weights', [
    (np.zeros((1, 2)), np.zeros((1, 2)), [1.]),
    (np.zeros((1, 3, 1)), np.zeros((2, 3, 1)), [1.]),
    (np.zeros((1, 3, 1)), np.zeros((1, 3, 1)), [0.]),
    (np.zeros((1, 3, 1)), np.zeros((1, 3, 1)), [-1.]),
    (np.full((1, 3, 1), np.nan), np.zeros((1, 3, 1)), [1.]),
])
def test_unresolved_point_integrals_are_refused(left, right, weights):
    with pytest.raises(RestrictedSourceQuadratureRefusal):
        restricted_point_gram(left, right, weights)


def test_descriptor_binding_is_repeatable_and_owns_readonly_arrays():
    context, _ = weak_fields()
    first = prepare_restricted_source_quadrature(context)
    second = prepare_restricted_source_quadrature(context)
    assert restricted_source_quadrature_to_dict(first) == restricted_source_quadrature_to_dict(second)
    with pytest.raises(ValueError):
        first.G_w[0, 0, 0] = 1.
    changed = replace(first, density_kg_m3=first.density_kg_m3+1.)
    with pytest.raises(RestrictedSourceQuadratureRefusal, match='binding'):
        restricted_source_quadrature_to_dict(changed)
