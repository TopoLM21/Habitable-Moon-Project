from dataclasses import FrozenInstanceError, asdict, fields, replace
import json

import numpy as np
import pytest

from tectonics.genesis_crack_energy import DCBOracle


def test_compliance_from_two_cantilever_deflections():
    model = DCBOracle(young_pa=64e9, arm_height_m=.003, width_m=.021)
    a, force = .087, 12.5
    second_moment = model.width_m*model.arm_height_m**3/12.
    arm_deflection = force*a**3/(3.*model.young_pa*second_moment)
    total_opening = 2.*arm_deflection
    assert model.compliance_m_n(a) == pytest.approx(total_opening/force)
    assert model.reaction_n(a, total_opening) == pytest.approx(force)
    # Each arm stores one half of force times its own deflection.
    assert model.stored_energy_j(a, total_opening) == pytest.approx(force*arm_deflection)


def test_energy_release_is_independent_fixed_displacement_derivative():
    model = DCBOracle()
    a, opening, epsilon = .072, .0009, 1e-7
    derivative = (model.stored_energy_j(a+epsilon, opening)
                  -model.stored_energy_j(a-epsilon, opening))/(2.*epsilon)
    assert model.release_rate_j_m2(a, opening) == pytest.approx(
        -derivative/model.width_m, rel=2e-10)
    force = model.reaction_n(a, opening)
    assert model.release_rate_j_m2(a, opening) == pytest.approx(
        3.*force*opening/(2.*model.width_m*a))


def test_equilibrium_satisfies_griffith_and_stable_growth_energy_balance():
    model = DCBOracle()
    opening_start, opening_end = .0007, .0013
    start = model.equilibrium_length_m(opening_start)
    end = model.equilibrium_length_m(opening_end)
    assert end > start > 0.
    for opening, length in ((opening_start, start), (opening_end, end)):
        assert model.release_rate_j_m2(length, opening) == pytest.approx(
            model.fracture_energy_j_m2)
        assert model.stored_energy_j(length, opening) == pytest.approx(
            model.fracture_energy_j_m2*model.width_m*length/3.)
    fracture = model.fracture_cost_j(start, end)
    elastic_change = (model.stored_energy_j(end, opening_end)
                      -model.stored_energy_j(start, opening_start))
    # Exact integral of P(delta) along the continuous equilibrium branch.
    external_work = 4.*model.fracture_energy_j_m2*model.width_m*(end-start)/3.
    assert external_work == pytest.approx(elastic_change+fracture)


def test_width_scaling_and_projected_area_counted_once():
    model = DCBOracle(fracture_energy_j_m2=321.)
    wider = replace(model, width_m=3.*model.width_m)
    a, next_a, opening = .07, .085, .0008
    assert wider.compliance_m_n(a) == pytest.approx(model.compliance_m_n(a)/3.)
    assert wider.reaction_n(a, opening) == pytest.approx(3.*model.reaction_n(a, opening))
    assert wider.stored_energy_j(a, opening) == pytest.approx(3.*model.stored_energy_j(a, opening))
    assert wider.release_rate_j_m2(a, opening) == pytest.approx(model.release_rate_j_m2(a, opening))
    assert wider.equilibrium_length_m(opening) == pytest.approx(model.equilibrium_length_m(opening))
    cost = model.fracture_energy_j_m2*model.width_m*(next_a-a)
    assert model.fracture_cost_j(a, next_a) == pytest.approx(cost)
    assert wider.fracture_cost_j(a, next_a) == pytest.approx(3.*cost)
    assert model.fracture_cost_j(a, a) == 0.


def test_zero_opening_has_zero_force_energy_release_and_unclipped_equilibrium():
    model = DCBOracle()
    assert model.reaction_n(.1, 0.) == 0.
    assert model.stored_energy_j(.1, 0.) == 0.
    assert model.release_rate_j_m2(.1, 0.) == 0.
    assert model.equilibrium_length_m(0.) == 0.
    with pytest.raises(FrozenInstanceError):
        model.width_m = .2


@pytest.mark.parametrize("invalid", [0., -1., float("nan"), float("inf"), True, "1", None])
def test_invalid_material_and_geometry_parameters(invalid):
    for field in fields(DCBOracle):
        with pytest.raises(ValueError):
            DCBOracle(**{field.name: invalid})


@pytest.mark.parametrize("invalid", [0., -1., float("nan"), float("inf"), True, "1", None])
def test_invalid_crack_lengths(invalid):
    model = DCBOracle()
    for call in (
        lambda: model.compliance_m_n(invalid),
        lambda: model.reaction_n(invalid, .001),
        lambda: model.stored_energy_j(invalid, .001),
        lambda: model.release_rate_j_m2(invalid, .001),
        lambda: model.fracture_cost_j(invalid, .1),
        lambda: model.fracture_cost_j(.1, invalid),
    ):
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize("invalid", [-1., float("nan"), float("inf"), True, "1", None])
def test_invalid_openings(invalid):
    model = DCBOracle()
    for call in (
        lambda: model.reaction_n(.1, invalid),
        lambda: model.stored_energy_j(.1, invalid),
        lambda: model.release_rate_j_m2(.1, invalid),
        lambda: model.equilibrium_length_m(invalid),
    ):
        with pytest.raises(ValueError):
            call()


def test_negative_growth_is_rejected():
    with pytest.raises(ValueError, match="nonnegative growth"):
        DCBOracle().fracture_cost_j(.1, .09)


def test_scalar_numpy_reals_are_accepted_but_vectors_are_rejected():
    model = DCBOracle(young_pa=np.float64(70e9))
    assert model.reaction_n(np.float64(.1), np.float32(.001)) > 0.
    with pytest.raises(ValueError):
        model.compliance_m_n(np.array([.1]))


def test_parameters_are_normalized_to_json_serializable_builtin_floats():
    model = DCBOracle(young_pa=np.int64(70_000_000_000),
                      arm_height_m=np.float32(.002),
                      width_m=np.float64(.025),
                      fracture_energy_j_m2=500)
    payload = asdict(model)
    assert all(type(value) is float for value in payload.values())
    restored = DCBOracle(**json.loads(json.dumps(payload, allow_nan=False)))
    assert restored == model
    assert restored.stored_energy_j(.07, .0008) == model.stored_energy_j(.07, .0008)


def test_unrepresentable_computed_results_are_rejected():
    model = DCBOracle()
    for call in (
        lambda: model.compliance_m_n(1e200),
        lambda: model.compliance_m_n(1e-200),
        lambda: model.reaction_n(1e-90, 1e100),
        lambda: model.stored_energy_j(.1, 1e200),
        lambda: model.release_rate_j_m2(.1, 1e200),
        lambda: model.equilibrium_length_m(1e200),
        lambda: replace(model, fracture_energy_j_m2=1e300).fracture_cost_j(.1, 1e100),
    ):
        with pytest.raises(ValueError):
            call()
