"""Independent elastic balance and all-branch monotonicity certificates."""
from types import SimpleNamespace

import numpy as np
import pytest

from tectonics.genesis_contact_law import ContactLawParameters, contact_return_map
from tectonics.geometric_contact_dynamics import (
    UnresolvedContactDynamics, _equilibrate_contact_network, _generalized_tangent_lower,
)
from tectonics.geometric_contact_mechanics import ElasticColumnParameters, solve_compressed_contact


def test_massless_transverse_release_matches_two_spring_analytic_solution_without_drag():
    columns = ElasticColumnParameters(1., 1e6, 1e6, 1., 1e6, 1e6)
    law = ContactLawParameters(normal_stiffness_pa_m=1e6, tangential_stiffness_pa_m=2e6,
        tensile_strength_pa=2e6, fracture_energy_j_m2=2e7, cohesion_pa=1e12,
        friction_dry=0., friction_wet=0., wet_cohesion_fraction=1., viscosity_pa_s_m=0.)
    material = SimpleNamespace(columns=columns, law=law, water=0.)
    anchors = np.array([[-1., .5]])
    area = 10.
    stiffness = np.diag([2e7, 3e7])
    initial = solve_compressed_contact(columns=columns, law=law, area_m2=area,
        gap_free_m=anchors[0, 0], jump_free_m=anchors[0, 1], dt_s=1., water=0.).state
    # Independent series-spring compliance, not a solver tangent or helper.
    local_stiffness = 1./np.array([1./columns.normal_stiffness_pa_m+1e-6,
                                 1./columns.shear_stiffness_pa_m+1./2e6])
    expected = -area*local_stiffness*anchors[0]/(np.diag(stiffness)+area*local_stiffness)
    solved = _equilibrate_contact_network(np.zeros(2), np.zeros((2, 2)), np.zeros(2),
        np.zeros((1, 2)), np.array([area]), np.zeros((1, 2)), (material,), anchors,
        (initial,), 1., .01, anchor_operators=np.eye(2)[None, :, :],
        elastic_stiffness=stiffness, rotation_dof_count=0)
    np.testing.assert_allclose(solved.q, expected, rtol=2e-14, atol=1e-16)
    bending_energy = .5*expected@stiffness@expected
    assert solved.diagnostics['bending_energy_increment_j'] == pytest.approx(bending_energy, rel=2e-14)
    assert solved.diagnostics['bending_numerical_dissipation_j'] == pytest.approx(bending_energy, rel=2e-14)
    assert solved.diagnostics['bending_work_j'] == pytest.approx(2.*bending_energy, rel=2e-14)
    assert solved.diagnostics['source_work_j'] == 0.
    assert solved.diagnostics['basal_dissipation_j'] == 0.
    assert solved.diagnostics['incremental_energy_balance_relative_error'] < 2e-14


@pytest.mark.parametrize('viscosity', [0., 1e5, 1e12])
@pytest.mark.parametrize('water', [0., 1.])
def test_common_lower_bounds_actual_return_map_tangents_for_locked_and_both_sliding_signs(viscosity, water):
    columns = ElasticColumnParameters(3., 2e6, 8e5, 5., 4e6, 7e5)
    law = ContactLawParameters(normal_stiffness_pa_m=1e6, tangential_stiffness_pa_m=2e6,
        tensile_strength_pa=2e6, fracture_energy_j_m2=2e7, cohesion_pa=10.,
        friction_dry=.6, friction_wet=.2, wet_cohesion_fraction=.5,
        viscosity_pa_s_m=viscosity)
    material = SimpleNamespace(columns=columns, law=law, water=water)
    rng = np.random.default_rng(712)
    inverse_bulk = np.diag([1./columns.normal_stiffness_pa_m, 1./columns.shear_stiffness_pa_m])
    for _ in range(12):
        # Non-orthogonal arbitrary generalized operators exercise mixed modes.
        operator = rng.normal(size=(2, 5))
        lower = _generalized_tangent_lower(material, 1e4, *operator)
        for jump in (0., -1000., 1000.):
            response = contact_return_map(np.array([-1.]), np.array([jump]), np.zeros(1),
                np.zeros(1), np.zeros(1), 1e4, np.array([water]), law, old_damage=np.zeros(1))
            tangent = response['tangent_pa_m'][0]
            condensed = np.linalg.solve(np.eye(2)+tangent@inverse_bulk, tangent)
            actual = operator.T@(.5*(condensed+condensed.T))@operator
            difference = .5*((actual-lower)+(actual-lower).T)
            scale = max(np.linalg.norm(actual), np.linalg.norm(lower), np.finfo(float).tiny)
            assert np.linalg.eigvalsh(difference)[0] >= -2e-13*scale


def test_unsupported_null_support_cannot_be_used_as_artificial_transverse_damping():
    with pytest.raises(UnresolvedContactDynamics, match='positive definite'):
        _equilibrate_contact_network(np.zeros(2), np.zeros((2, 2)), np.zeros(2),
            np.zeros((0, 2)), np.zeros(0), np.zeros((0, 2)), (), np.zeros((0, 2)),
            (), 1., .01, anchor_operators=np.zeros((0, 2, 2)),
            elastic_stiffness=np.zeros((2, 2)), rotation_dof_count=0)
