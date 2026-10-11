import json
from dataclasses import asdict, fields, replace

import numpy as np
import pytest

from tectonics.geometric_contact_mechanics import (
    ElasticColumnParameters, LocalContactState, solve_compressed_contact,
    state_from_dict, state_to_dict, validate_contact_state,
)
from tectonics.genesis_contact_law import ContactLawParameters


def inputs():
    return dict(
        columns=ElasticColumnParameters(1000., 1e9, 1e8, 1000., 1e9, 1e8),
        law=ContactLawParameters(
            normal_stiffness_pa_m=1e6, tangential_stiffness_pa_m=1e5,
            tensile_strength_pa=2e6, fracture_energy_j_m2=2e7,
            cohesion_pa=1e4, friction_dry=.4, friction_wet=.2,
            wet_cohesion_fraction=.5, viscosity_pa_s_m=1e8),
        area_m2=1e4, gap_free_m=-1., jump_free_m=-1., dt_s=1e6, water=0.)


def test_elastic_series_solution_and_locked_shortening():
    p = inputs()
    answer = solve_compressed_contact(**p)
    s, law, col = answer.state, p["law"], p["columns"]
    expected_g = p["gap_free_m"]*col.normal_stiffness_pa_m/(col.normal_stiffness_pa_m+law.normal_stiffness_pa_m)
    expected_j = p["jump_free_m"]*col.shear_stiffness_pa_m/(col.shear_stiffness_pa_m+law.tangential_stiffness_pa_m)
    assert s.gap_m == pytest.approx(expected_g)
    assert s.jump_m == pytest.approx(expected_j)
    assert s.plastic_slip_m == 0.
    assert s.gap_m < 0 and s.bulk_energy_j > 0 and s.contact_energy_j > 0
    assert answer.friction_work_j == answer.viscous_work_j == 0.
    assert answer.force_relative_residual < 1e-12
    assert answer.energy_balance_relative_error < 1e-12
    stored = s.bulk_energy_j+s.contact_energy_j
    assert answer.external_work_j == pytest.approx(2*stored)
    assert answer.backward_euler_remainder_j == pytest.approx(stored)


@pytest.mark.parametrize("sign", [-1., 1.])
def test_viscous_slip_matches_independent_series_spring_formula(sign):
    p = inputs()
    p["jump_free_m"] = sign*10.
    answer = solve_compressed_contact(**p)
    col, law = p["columns"], p["law"]
    equivalent_k = 1./(1./col.shear_stiffness_pa_m+1./law.tangential_stiffness_pa_m)
    pressure = -law.normal_stiffness_pa_m*p["gap_free_m"]*col.normal_stiffness_pa_m/(col.normal_stiffness_pa_m+law.normal_stiffness_pa_m)
    strength = law.cohesion_pa+law.friction_dry*pressure
    eta_dt = law.viscosity_pa_s_m/p["dt_s"]
    expected_slip = sign*(equivalent_k*10.-strength)/(equivalent_k+eta_dt)
    expected_traction = sign*(strength+eta_dt*abs(expected_slip))
    assert answer.slip_increment_m == pytest.approx(expected_slip)
    assert answer.traction_pa[1] == pytest.approx(expected_traction)
    assert answer.friction_work_j == pytest.approx(p["area_m2"]*strength*abs(expected_slip))
    assert answer.viscous_work_j == pytest.approx(p["area_m2"]*eta_dt*expected_slip**2)
    assert answer.energy_balance_relative_error < 1e-12
    assert answer.force_relative_residual < 1e-12


def test_loading_unloading_reverse_and_restart_preserve_full_history():
    p = inputs()
    last = None
    for normal, shear, dt in [(-1., -1., 1e6), (-1., -10., 1e6),
                              (-.5, -9., 2e6), (-2., 15., 1e6),
                              (-.25, 10., 5e5), (0., 10., 1e6)]:
        p.update(gap_free_m=normal, jump_free_m=shear, dt_s=dt)
        direct = solve_compressed_contact(**p, previous=last)
        if last is not None:
            before = asdict(last)
            serialized = json.dumps(state_to_dict(last), sort_keys=True, allow_nan=False)
            restored = state_from_dict(json.loads(serialized))
            resumed = solve_compressed_contact(**p, previous=restored)
            assert resumed == direct
            assert asdict(last) == before
        assert direct.energy_balance_relative_error < 1e-12
        assert direct.friction_work_j >= 0 and direct.viscous_work_j >= 0
        assert direct.backward_euler_remainder_j >= 0
        last = direct.state
    assert last.step_count == 6
    assert last.elapsed_time_s == 6.5e6
    assert last.cumulative_slip_m >= abs(last.plastic_slip_m)
    accounted = (last.bulk_energy_j+last.contact_energy_j+last.cumulative_friction_work_j
                 +last.cumulative_viscous_work_j+last.cumulative_numerical_dissipation_j)
    assert last.cumulative_external_work_j == pytest.approx(accounted, rel=1e-12)


def test_incremental_global_remainder_uses_elastic_slip_not_fixed_jump_relaxation():
    p = inputs()
    p["jump_free_m"] = -10.
    first = solve_compressed_contact(**p)
    p["jump_free_m"] = -20.
    second = solve_compressed_contact(**p, previous=first.state)
    old, new = first.state, second.state
    col, law, area = p["columns"], p["law"], p["area_m2"]
    expected = .5*area*(
        col.normal_stiffness_pa_m*((new.gap_m-new.gap_free_m)-(old.gap_m-old.gap_free_m))**2
        +col.shear_stiffness_pa_m*((new.jump_m-new.jump_free_m)-(old.jump_m-old.jump_free_m))**2
        +law.normal_stiffness_pa_m*(new.gap_m-old.gap_m)**2
        +law.tangential_stiffness_pa_m*((new.jump_m-new.plastic_slip_m)-(old.jump_m-old.plastic_slip_m))**2)
    fixed_jump_remainder = .5*area*law.tangential_stiffness_pa_m*second.slip_increment_m**2
    assert second.backward_euler_remainder_j == pytest.approx(expected)
    assert second.backward_euler_remainder_j != pytest.approx(fixed_jump_remainder)
    assert second.energy_balance_relative_error < 1e-12


def test_unchanged_anchors_relax_viscously_without_rebasing():
    p = inputs()
    p["jump_free_m"] = -10.
    first = solve_compressed_contact(**p)
    second = solve_compressed_contact(**p, previous=first.state)
    assert second.external_work_j == 0.
    assert second.slip_increment_m < 0.
    assert second.state.contact_energy_j < first.state.contact_energy_j
    assert second.state.cumulative_slip_m > first.state.cumulative_slip_m
    assert second.energy_balance_relative_error < 1e-12


def test_rate_independent_solution_exists_without_claiming_strict_admission():
    p = inputs()
    p.update(jump_free_m=-10., law=replace(p["law"], viscosity_pa_s_m=0.))
    answer = solve_compressed_contact(**p)
    assert answer.slip_increment_m < 0
    assert -answer.traction_pa[1] == pytest.approx(answer.shear_strength_pa)
    assert answer.viscous_work_j == 0.


@pytest.mark.parametrize("name,value", [("area_m2", 0.), ("area_m2", True),
    ("gap_free_m", .01), ("jump_free_m", float("inf")), ("dt_s", 0.),
    ("dt_s", True), ("water", -.1), ("water", 1.1), ("water", True)])
def test_invalid_or_opening_inputs_rejected(name, value):
    p = inputs()
    p[name] = value
    with pytest.raises(ValueError):
        solve_compressed_contact(**p)


@pytest.mark.parametrize("value", [0., -1., True, float("nan"), float("inf")])
def test_singular_column_inputs_rejected(value):
    with pytest.raises(ValueError):
        ElasticColumnParameters(value, 1e9, 1e8, 1000., 1e9, 1e8)


def test_all_column_physical_parameters_required():
    with pytest.raises(TypeError):
        ElasticColumnParameters()
    assert all(f.default is __import__("dataclasses").MISSING for f in fields(ElasticColumnParameters))
    p = inputs()
    p.pop("water")
    with pytest.raises(TypeError):
        solve_compressed_contact(**p)


@pytest.mark.parametrize("what", ["columns", "law", "area_m2", "water"])
def test_restart_rejects_changed_physical_model(what):
    p = inputs()
    first = solve_compressed_contact(**p)
    changes = dict(columns=replace(p["columns"], sub_length_m=2000.),
                   law=replace(p["law"], friction_dry=.5), area_m2=20000., water=.5)
    p[what] = changes[what]
    with pytest.raises(ValueError, match="frozen"):
        solve_compressed_contact(**p, previous=first.state)


@pytest.mark.parametrize("name,value", [("traction_shear_pa", 123.),
    ("bulk_energy_j", 0.), ("cumulative_external_work_j", 0.),
    ("step_count", True), ("cumulative_slip_m", -1.), ("gap_m", .1)])
def test_corrupt_restart_history_is_rejected(name, value):
    original = state_to_dict(solve_compressed_contact(**inputs()).state)
    original[name] = value
    with pytest.raises(ValueError):
        state_from_dict(original)


def test_restart_schema_rejects_unknown_missing_fields_and_bool_signatures():
    original = state_to_dict(solve_compressed_contact(**inputs()).state)
    with pytest.raises(ValueError):
        state_from_dict({**original, "extra": 1})
    missing = original.copy()
    del missing["gap_m"]
    with pytest.raises(ValueError):
        state_from_dict(missing)
    original["columns_signature"][0] = True
    with pytest.raises(ValueError):
        state_from_dict(original)


def test_area_subdivision_preserves_uniform_pressure_and_additive_work():
    p = inputs()
    p["jump_free_m"] = -10.
    whole = solve_compressed_contact(**p)
    p["area_m2"] /= 7
    piece = solve_compressed_contact(**p)
    assert piece.traction_pa == whole.traction_pa
    assert piece.slip_increment_m == whole.slip_increment_m
    for name in ("external_work_j", "friction_work_j", "viscous_work_j",
                 "backward_euler_remainder_j", "bulk_energy_increment_j", "contact_energy_increment_j"):
        assert 7*getattr(piece, name) == pytest.approx(getattr(whole, name), rel=1e-12)


def test_kilometre_accumulated_slip_survives_restart_and_cancellation():
    p = inputs()
    p.update(columns=ElasticColumnParameters(1000., 1e10, 1e10, 1000., 1e10, 1e10),
        law=ContactLawParameters(normal_stiffness_pa_m=1e7,tangential_stiffness_pa_m=1e7,
            tensile_strength_pa=2e6,fracture_energy_j_m2=2e7,cohesion_pa=2e6,
            friction_dry=.6,friction_wet=.2,wet_cohesion_fraction=.5,viscosity_pa_s_m=1e14),
        gap_free_m=-.1,jump_free_m=-1000.,dt_s=1e13)
    first = solve_compressed_contact(**p)
    restored = state_from_dict(json.loads(json.dumps(state_to_dict(first.state))))
    assert first.state == restored
    assert first.force_relative_residual < 1e-10
    assert first.energy_balance_relative_error < 1e-10
    p["jump_free_m"] = -1001.
    second = solve_compressed_contact(**p,previous=restored)
    assert second.force_relative_residual < 1e-10
    assert second.energy_balance_relative_error < 1e-10


def test_binding_validator_does_not_advance_mechanical_time():
    p = inputs()
    original = solve_compressed_contact(**p).state
    result = validate_contact_state(original,columns=p["columns"],law=p["law"],
                                   area_m2=p["area_m2"],water=p["water"])
    assert result is original
    assert result.elapsed_time_s == p["dt_s"]
