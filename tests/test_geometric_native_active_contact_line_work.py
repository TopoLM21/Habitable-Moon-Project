"""Original-measure weighted pair budgets and independent harmonic work."""
from copy import deepcopy
from decimal import Decimal, localcontext
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import itertools
import json

import pytest

from analysis.geometric_native_active_line_validation.independent_line_oracle import (
    decimal, independent_frozen_potential_integral, original_measure_and_coefficients,
)
from tectonics.geometric_native_pose import FractionInterval as I
from tectonics.geometric_native_physical_history import native_physical_history_binding_to_dict
from tests.test_geometric_native_active_contact_line import get_active_line_case


BUDGETS = ("source_work_bounds_j", "friction_dissipation_bounds_j",
    "bulk_energy_increment_bounds_j", "contact_energy_increment_bounds_j",
    "total_energy_increment_bounds_j", "compliance_balance_residual_bounds_j")


@lru_cache(maxsize=1)
def get_active_line_work_case():
    from tectonics.geometric_native_active_contact_line_work import integrate_native_active_contact_line_work
    case = get_active_line_case()
    value = integrate_native_active_contact_line_work(case["history"],
        expected_source_history_dto=case["source"], line_certificate=case["certificate"])
    return dict(case=case, value=value)


@pytest.fixture(scope="module")
def line_work_case():
    return get_active_line_work_case()


def contains_interval(outer, inner):
    assert outer.lo <= inner.lo <= inner.hi <= outer.hi


def overlaps(first, second):
    assert max(first.lo, second.lo) <= min(first.hi, second.hi)


def reseal(raw):
    raw["integral_sha256"] = hashlib.sha256(json.dumps(
        {name: value for name, value in raw.items() if name != "integral_sha256"},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def restore_arguments(case):
    return dict(history=case["history"], expected_source_history_dto=case["source"],
        expected_line_certificate=case["certificate"])


def test_all_sixteen_pairs_and_every_original_interface_measure_are_retained(line_work_case):
    case, value = line_work_case["case"], line_work_case["value"]
    banks = case["history"].material_state.trace_bindings
    expected_pairs = set(itertools.product((row.donor_fragment_id for row in banks),
        (row.receiver_fragment_id for row in banks)))
    actual_pairs = [(row.donor_fragment_id, row.receiver_fragment_id) for row in value.receiving_pair_budgets]
    assert len(actual_pairs) == len(set(actual_pairs)) == 16
    assert set(actual_pairs) == expected_pairs
    total_area = F(0)
    for quarter in value.quarters:
        geometric = next(row for row in case["certificate"].quarters
            if row.receiver_trace_id == quarter.receiver_trace_id)
        coefficient, _ = original_measure_and_coefficients(case["source"], quarter.receiver_trace_id)
        assert quarter.original_interface_area_density_m2_per_computational_radian == coefficient
        assert quarter.whole_original_interface_area_m2 == coefficient*geometric.computational_extent_radians
        assert quarter.core_original_interface_area_m2+quarter.strip_original_interface_area_m2 \
            == quarter.whole_original_interface_area_m2
        assert quarter.strip_original_interface_area_m2 > 0
        overlaps(quarter.native_angle_per_computational_radian_bounds*geometric.computational_extent_radians,
            geometric.native_extent_radians)
        total_area += quarter.whole_original_interface_area_m2
    assert value.original_interface_area_m2 == total_area
    assert value.explicitly_enclosed_strip_interface_area_m2 == sum(
        row.strip_original_interface_area_m2 for row in value.quarters)
    assert 0 < value.explicitly_enclosed_strip_interface_area_m2 < value.original_interface_area_m2*F(1, 10**30)


@pytest.mark.parametrize("quarter_index", [0, 1, 2, 3])
def test_independent_real_harmonic_full_rotation_core_integral_is_enclosed(line_work_case, quarter_index):
    case, value = line_work_case["case"], line_work_case["value"]
    geometric = case["certificate"].quarters[quarter_index]
    pair = next(row for row in value.receiving_pair_budgets
        if row.receiver_trace_id == geometric.receiver_trace_id and row.donor_trace_id == geometric.original_donor_trace_id)
    core = geometric.core_computational_angle_bounds
    independent = independent_frozen_potential_integral(case["source"], geometric.receiver_trace_id,
        core.lo, core.hi, geometric.computational_extent_radians)
    assert pair.core_potential_increment_integral_bounds_m4.lo <= F(independent) \
        <= pair.core_potential_increment_integral_bounds_m4.hi
    assert pair.core_potential_increment_integral_bounds_m4.hi < 0
    assert pair.core_potential_increment_integral_bounds_m4.width < abs(F(independent))*F(1, 10**100)


def test_proved_negative_seams_keep_original_full_frozen_work(line_work_case):
    case, value = line_work_case["case"], line_work_case["value"]
    geometric = next(row for row in case["certificate"].quarters if row.signs_y_z == (-1, -1))
    pair = next(row for row in value.receiving_pair_budgets if row.receiver_trace_id == geometric.receiver_trace_id
        and row.donor_trace_id == geometric.original_donor_trace_id)
    independent = independent_frozen_potential_integral(case["source"], geometric.receiver_trace_id,
        F(0), geometric.computational_extent_radians, geometric.computational_extent_radians)
    assert pair.potential_increment_integral_bounds_m4.lo <= F(independent) <= pair.potential_increment_integral_bounds_m4.hi
    for row in value.receiving_pair_budgets:
        if row.receiver_trace_id == geometric.receiver_trace_id and row.donor_trace_id != geometric.original_donor_trace_id:
            assert row.enclosed_strip_original_interface_area_m2 == 0
            assert row.potential_increment_integral_bounds_m4 == I(0)


def test_every_pair_has_one_scalar_potential_and_the_exact_signed_literal_first_law(line_work_case):
    source, value = line_work_case["case"]["source"], line_work_case["value"]
    for pair in value.receiving_pair_budgets:
        _, coefficients = original_measure_and_coefficients(source, pair.receiver_trace_id)
        assert pair.literal_normal_coefficients_pa_m == coefficients
        kn, bulk, contact = coefficients
        assert kn-bulk-contact != 0
        amounts = (kn, F(0), bulk, contact, bulk+contact, kn-bulk-contact)
        for field, coefficient in zip(BUDGETS, amounts):
            contains_interval(getattr(pair, field), pair.potential_increment_integral_bounds_m4*coefficient)
        assert pair.friction_dissipation_bounds_j == I(0)
        if pair.enclosed_strip_original_interface_area_m2:
            assert pair.strip_potential_increment_integral_bounds_m4.hi <= 0
            assert pair.strip_potential_increment_integral_bounds_m4.lo < 0
    for field in BUDGETS:
        contains_interval(getattr(value, field), sum((getattr(row, field)
            for row in value.receiving_pair_budgets), I(0)))
    assert value.source_work_bounds_j.hi < 0
    assert value.friction_dissipation_bounds_j == I(0)
    assert value.first_law_residual_bounds_j.lo <= 0 <= value.first_law_residual_bounds_j.hi
    assert value.first_law_residual_bounds_j.width < abs(value.source_work_bounds_j.lo)*F(1, 10**30)
    overlaps(value.source_work_bounds_j, value.independently_enclosed_source_work_bounds_j)


def test_full_angle_carried_branch_is_compressive_and_monotone_from_preparation(line_work_case):
    value = line_work_case["value"]
    for quarter in value.quarters:
        branch = quarter.branch_proof
        assert branch.initial_gap_bounds_m == I(F.from_float(-.1))
        assert branch.whole_line_gap_bounds_m.hi < 0
        assert branch.whole_line_gap_rate_bounds_m_per_fraction.lo > 0
        assert branch.whole_line_jump_rate_bounds_m_per_fraction.hi < 0
        assert 0 < branch.original_shape_bounds.lo <= branch.original_shape_bounds.hi == 1
        assert branch.carried_state_at_switch and branch.compression_proved and branch.monotone_down_dip_proved
        overlaps(quarter.final_normal_potential_integral_bounds_m4-quarter.initial_normal_potential_integral_bounds_m4,
            quarter.independent_potential_increment_integral_bounds_m4)


def test_work_keeps_old_atoms_affine_archive_and_force_candidate_separate(line_work_case):
    from tectonics.geometric_native_active_contact_line_work import native_active_contact_line_work_to_dict
    case, value = line_work_case["case"], line_work_case["value"]
    raw = native_active_contact_line_work_to_dict(value, expected_source_history_dto=case["source"])
    for key in ("temporal_quadrature_used", "root_midpoint_selected", "strip_centroid_or_representative_selected",
        "tiny_strip_work_zeroed", "original_nodal_ledgers_summed_with_continuous_work",
        "original_affine_source_work_included", "swept_material_area_used_as_contact_measure",
        "candidate_mechanics_adopted", "physical_pair_admission_authorized"):
        assert raw[key] is False
    for key in ("transfer_work_j", "transfer_energy_increment_j", "transfer_dissipation_j"):
        assert raw[key] == [0, 1]
    assert raw["accepted_native_force_intervals"] == 0
    assert len(case["history"].quadrature_histories) == 16
    assert case["history"].accepted_before.cumulative_source_work_j == 0
    assert case["history"].accepted_before.receipts == ()
    assert native_physical_history_binding_to_dict(case["history"]) == case["source"]


def test_weighted_line_work_restart_replays_the_external_source_and_selector(line_work_case):
    from tectonics.geometric_native_active_contact_line_work import (
        native_active_contact_line_work_from_dict, native_active_contact_line_work_to_dict,
    )
    case, value = line_work_case["case"], line_work_case["value"]
    raw = native_active_contact_line_work_to_dict(value, expected_source_history_dto=case["source"])
    replay = native_active_contact_line_work_from_dict(deepcopy(raw), **restore_arguments(case))
    assert native_active_contact_line_work_to_dict(replay, expected_source_history_dto=case["source"]) == raw


@pytest.mark.parametrize("mutation", ["measure", "pair_bank", "pair_potential", "work", "residual",
    "shape", "zero_strip", "adoption", "nodal_work", "selector", "source", "unknown"])
def test_resealed_weighted_restart_rejects_semantic_substitution(line_work_case, mutation):
    from tectonics.geometric_native_active_contact_line_work import (
        native_active_contact_line_work_from_dict, native_active_contact_line_work_to_dict,
    )
    case, value = line_work_case["case"], line_work_case["value"]
    raw = deepcopy(native_active_contact_line_work_to_dict(value, expected_source_history_dto=case["source"]))
    if mutation == "measure":
        raw["measure"] = "R_squared_coarea_du_dtheta"
    elif mutation == "pair_bank":
        raw["receiving_pair_budgets"][0]["donor_fragment_id"] = "octant:fake"
    elif mutation == "pair_potential":
        raw["receiving_pair_budgets"][0]["potential_increment_integral_bounds_m4"] = {"lo": [0, 1], "hi": [0, 1]}
    elif mutation == "work":
        raw["source_work_bounds_j"] = {"lo": [0, 1], "hi": [0, 1]}
    elif mutation == "residual":
        raw["compliance_balance_residual_bounds_j"] = {"lo": [0, 1], "hi": [0, 1]}
    elif mutation == "shape":
        raw["quarters"][0]["branch_proof"]["original_shape_bounds"] = {"lo": [-1, 1], "hi": [1, 1]}
    elif mutation == "zero_strip":
        raw["explicitly_enclosed_strip_interface_area_m2"] = [0, 1]
    elif mutation == "adoption":
        raw["accepted_native_force_intervals"] = 1
        raw["physical_pair_admission_authorized"] = True
    elif mutation == "nodal_work":
        raw["original_nodal_ledgers_summed_with_continuous_work"] = True
    elif mutation == "selector":
        raw["line_selector"]["selection_law"] = "reproject_every_clock"
    elif mutation == "source":
        raw["complete_original_source_history_sha256"] = "0"*64
    else:
        raw["unrequested_budget"] = True
    reseal(raw)
    with pytest.raises(ValueError):
        native_active_contact_line_work_from_dict(raw, **restore_arguments(case))
