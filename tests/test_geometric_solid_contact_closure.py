"""Physical elastic compression is not missing preload or negative void."""
from dataclasses import replace
from fractions import Fraction as F
import pytest

from tectonics.geometric_contact_mechanics import ElasticColumnParameters
from tectonics.genesis_contact_law import ContactLawParameters
from tectonics.geometric_solid_contact_closure import (
    compute_series_normal_closure, geometry_to_anchor,
    require_contact_zone_compaction, solve_solid_normal_closure,
    solve_solid_tangential_closure,
)


def _physics():
    return (ElasticColumnParameters(1000., 1e9, 4e8, 2000., 2e9, 8e8),
        ContactLawParameters(normal_stiffness_pa_m=1e6,
            tangential_stiffness_pa_m=1e6, cohesion_pa=0.,
            friction_dry=0., friction_wet=0., viscosity_pa_s_m=0.))


def _normal(gap=F(-1, 10), rest=F(1)):
    columns, law = _physics()
    return solve_solid_normal_closure(columns=columns, law=law,
        free_gap_m=gap, contact_zone_rest_normal_thickness_m=rest)


def _geometry(normal, **changes):
    supplied = dict(normal=normal, physical_face_void_m=normal.physical_face_void_m,
        sub_column_shortening_m=normal.sub_column_shortening_m,
        over_column_shortening_m=normal.over_column_shortening_m,
        fault_indentation_m=normal.fault_indentation_m)
    supplied.update(changes)
    return geometry_to_anchor(**supplied)


def test_exact_series_compression_has_zero_void_and_separate_material_shortening():
    closure = _normal()
    assert closure.normal_pressure_pa == F(100000, 3)
    assert closure.physical_face_void_m == 0
    assert closure.sub_column_shortening_m == closure.over_column_shortening_m == F(1, 30)
    assert closure.fault_indentation_m == F(1, 30)
    assert closure.constitutive_contact_gap_m == -F(1, 30)
    assert closure.deformed_contact_zone_thickness_m == F(29, 30)
    assert closure.anchor_balance_residual_m == closure.energy_balance_residual_j_m2 == 0
    assert _geometry(closure).anchor_residual_m == 0
    assert closure.material_admission_authorized is False
    assert closure.finite_solid_placement_authorized is False


def test_independent_force_energy_and_virtual_work_equalities():
    closure = _normal(F(-3, 7), F(2))
    pressure = closure.normal_pressure_pa
    assert pressure == F(1000000, 7)
    assert pressure*closure.sub_normal_compliance_m_pa == closure.sub_column_shortening_m
    assert pressure*closure.over_normal_compliance_m_pa == closure.over_column_shortening_m
    assert pressure*closure.fault_normal_compliance_m_pa == closure.fault_indentation_m
    independent_energy = sum((displacement*displacement/(2*compliance)
        for displacement, compliance in zip((closure.sub_column_shortening_m,
            closure.over_column_shortening_m, closure.fault_indentation_m),
            (closure.sub_normal_compliance_m_pa, closure.over_normal_compliance_m_pa,
                closure.fault_normal_compliance_m_pa))), F(0))
    assert closure.total_energy_j_m2 == independent_energy == pressure*F(3, 7)/2
    epsilon = F(1, 100000)
    before, after = _normal(F(-3, 7)-epsilon, F(2)), _normal(F(-3, 7)+epsilon, F(2))
    assert (after.total_energy_j_m2-before.total_energy_j_m2)/(2*epsilon) == -pressure


@pytest.mark.parametrize("gap,branch", [(F(3, 11), "open_unbonded"), (F(0), "touching_unloaded")])
def test_unbonded_opening_is_positive_void_and_has_no_elastic_compression(gap, branch):
    closure = _normal(gap)
    assert closure.branch == branch
    assert closure.normal_pressure_pa == closure.total_energy_j_m2 == 0
    assert closure.physical_face_void_m == closure.constitutive_contact_gap_m == gap
    assert closure.sub_column_shortening_m == closure.over_column_shortening_m == closure.fault_indentation_m == 0
    assert _geometry(closure).anchor_residual_m == 0


def test_contact_zone_length_cannot_be_inferred_from_preload_or_stiffness():
    columns, law = _physics()
    with pytest.raises(TypeError, match="contact_zone_rest_normal_thickness_m"):
        solve_solid_normal_closure(columns=columns, law=law, free_gap_m=F(-1, 10))
    with pytest.raises(ValueError, match="rest thickness"):
        _normal(rest=F(1, 30))
    small, large = _normal(rest=F(1)), _normal(rest=F(5))
    assert small.normal_pressure_pa == large.normal_pressure_pa
    assert small.total_energy_j_m2 == large.total_energy_j_m2
    assert large.deformed_contact_zone_thickness_m-small.deformed_contact_zone_thickness_m == 4


def test_true_source_decomposition_does_not_invent_missing_contact_zone_geometry():
    columns, law = _physics()
    normal = compute_series_normal_closure(columns=columns, law=law, free_gap_m=F(-1, 10))
    assert normal.normal_pressure_pa == F(100000, 3)
    assert normal.fault_indentation_m == F(1, 30)
    assert normal.contact_zone_rest_normal_thickness_m is None
    assert normal.deformed_contact_zone_thickness_m is None
    assert normal.contact_zone_geometry_declared is False
    assert normal.physical_geometry_authorized is False
    assert _geometry(normal).anchor_residual_m == 0
    with pytest.raises(TypeError, match="contact_zone_rest_normal_thickness_m"):
        require_contact_zone_compaction(normal)
    with pytest.raises(ValueError, match="finite literal"):
        require_contact_zone_compaction(normal, contact_zone_rest_normal_thickness_m=None)
    declared = require_contact_zone_compaction(normal, contact_zone_rest_normal_thickness_m=F(1))
    assert declared.contact_zone_geometry_declared is True
    assert declared.physical_geometry_authorized is False
    assert declared.normal_pressure_pa == normal.normal_pressure_pa
    with pytest.raises(ValueError, match="cannot be rebased"):
        require_contact_zone_compaction(declared, contact_zone_rest_normal_thickness_m=F(2))


def test_full_column_collapse_is_refused_independently_of_contact_zone_length():
    with pytest.raises(ValueError, match="rest thickness"):
        _normal(gap=F(-3000), rest=F(100000))


def test_positive_bevel_void_cannot_rebase_or_discard_paid_compression():
    closure = _normal()
    with pytest.raises(ValueError, match="free anchor"):
        _geometry(closure, physical_face_void_m=F(3, 40),
            sub_column_shortening_m=F(0), over_column_shortening_m=F(0),
            fault_indentation_m=F(0))


def test_total_anchor_match_cannot_hide_wrong_pressure_partition():
    closure = _normal()
    with pytest.raises(ValueError, match="pressure partition"):
        _geometry(closure, sub_column_shortening_m=F(1, 15),
            over_column_shortening_m=F(0))


def test_physical_void_cannot_be_literal_negative_constitutive_gap():
    closure = _normal()
    with pytest.raises(ValueError, match="nonnegative"):
        _geometry(closure, physical_face_void_m=closure.constitutive_contact_gap_m)


def test_geometry_check_preserves_explicit_representation_error_without_forging_closure():
    closure = _normal()
    epsilon = F(1, 10**10)
    check = _geometry(closure, fault_indentation_m=closure.fault_indentation_m+epsilon,
        tolerance_m=epsilon)
    assert check.anchor_residual_m == -epsilon
    assert check.fault_indentation_residual_m == epsilon
    assert check.material_admission_authorized is False
    assert check.finite_solid_placement_authorized is False
    with pytest.raises(ValueError, match="free anchor"):
        _geometry(closure, fault_indentation_m=closure.fault_indentation_m+2*epsilon,
            tolerance_m=epsilon)


def test_forged_frozen_result_is_not_authority_for_geometry():
    closure = _normal()
    with pytest.raises(ValueError, match="declared elastic physics"):
        _geometry(replace(closure, normal_pressure_pa=closure.normal_pressure_pa+1))


def test_equal_boolean_or_float_cannot_replace_exact_reconstructed_fields():
    closure = _normal()
    for changed in (replace(closure, physical_face_void_m=False),
            replace(closure, anchor_balance_residual_m=0.)):
        with pytest.raises(ValueError, match="exact Fraction fields"):
            _geometry(changed)


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), "0.1", None])
def test_nonliteral_or_nonfinite_gap_is_refused(bad):
    with pytest.raises(ValueError, match="finite literal"):
        _normal(gap=bad)


@pytest.mark.parametrize("bad", [0, -1, True, float("inf")])
def test_independent_rest_thickness_requires_finite_positive_physics(bad):
    with pytest.raises(ValueError):
        _normal(rest=bad)


def test_parameter_dict_and_boolean_contact_parameter_cannot_supply_physics():
    columns, law = _physics()
    with pytest.raises(ValueError, match="parameter types"):
        solve_solid_normal_closure(columns={"sub_length_m": 1000.}, law=law,
            free_gap_m=0, contact_zone_rest_normal_thickness_m=1)
    with pytest.raises(ValueError):
        solve_solid_normal_closure(columns=columns,
            law=replace(law, normal_stiffness_pa_m=True), free_gap_m=0,
            contact_zone_rest_normal_thickness_m=1)


def test_tangential_series_compatibility_matches_independent_spring_forces():
    columns, law = _physics()
    closure = solve_solid_tangential_closure(columns=columns, law=law,
        free_jump_m=F(3, 20), plastic_slip_m=F(1, 20))
    assert closure.shear_traction_pa == F(50000, 3)
    assert closure.sub_column_shear_m == closure.over_column_shear_m == F(1, 24)
    assert closure.fault_elastic_shear_m == F(1, 60)
    assert closure.constitutive_contact_jump_m == F(1, 15)
    assert closure.anchor_balance_residual_m == 0
    assert closure.total_energy_j_m2 == closure.shear_traction_pa*F(1, 10)/2


def test_reverse_frictionless_sliding_can_be_elastically_valid_without_underpass_permission():
    columns, law = _physics()
    previous = solve_solid_tangential_closure(columns=columns, law=law,
        free_jump_m=F(-1, 20), plastic_slip_m=F(-1, 20))
    current = solve_solid_tangential_closure(columns=columns, law=law,
        free_jump_m=F(-47, 1000), plastic_slip_m=F(-47, 1000))
    assert current.plastic_slip_m-previous.plastic_slip_m == F(3, 1000)
    assert previous.shear_traction_pa == current.shear_traction_pa == 0
    assert current.total_energy_j_m2 == current.anchor_balance_residual_m == 0
    assert current.material_admission_authorized is False


@pytest.mark.parametrize("sign", [-1, 1])
def test_elastic_shear_supports_signed_loading_and_nonnegative_energy(sign):
    columns, law = _physics()
    closure = solve_solid_tangential_closure(columns=columns, law=law,
        free_jump_m=sign*F(1, 10), plastic_slip_m=0)
    assert sign*closure.shear_traction_pa > 0
    assert sign*closure.fault_elastic_shear_m > 0
    assert closure.total_energy_j_m2 > 0
    assert closure.anchor_balance_residual_m == 0


def test_literal_float_is_retained_exactly_without_decimal_reinterpretation():
    closure = _normal(-0.1)
    assert closure.free_gap_m == F.from_float(-0.1)
    assert closure.free_gap_m != F(-1, 10)
