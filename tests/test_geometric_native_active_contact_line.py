"""Whole-angle geometry, positive-measure tiny sectors and strict restart."""
from copy import deepcopy
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import pytest

from analysis.geometric_native_active_line_validation.independent_line_oracle import (
    derivative_tail, fresh_depth_whole_two_clock_bounds, initial_depth_divided_clock_bounds,
    joint_pullback_coefficients, polynomial_at, polynomial_bounds, turning_numerator_coefficients,
)
from tectonics.geometric_native_contact_reference_transport import (
    ContactHistoryTransportLaw, TRANSPORT_LAW_MODEL,
)
from tectonics.geometric_native_physical_history import native_physical_history_binding_to_dict
from tectonics.geometric_native_pose import FractionInterval as I
from tests.test_geometric_native_physical_history import get_physical_case


ROOT = Path(__file__).resolve().parents[1]
SAVED_SOURCE = ROOT / "analysis/geometric_native_active_binding_validation/control_release_02/source_history_before.json"


def saved_source():
    return json.loads(SAVED_SOURCE.read_bytes())


@lru_cache(maxsize=1)
def get_active_line_case():
    from tectonics.geometric_native_active_contact_line import certify_native_active_contact_line
    original = get_physical_case()
    history = original["binding"]
    source = native_physical_history_binding_to_dict(history)
    law = ContactHistoryTransportLaw(TRANSPORT_LAW_MODEL,
        "Whole receiving line: switch only at a unique proved positive fresh seam; retain at a proved negative seam")
    certificate = certify_native_active_contact_line(history, law=law,
        expected_source_history_dto=source)
    return dict(original=original, history=history, source=source, law=law, certificate=certificate)


@pytest.fixture(scope="module")
def line_case():
    return get_active_line_case()


def reseal(raw):
    raw["certificate_sha256"] = hashlib.sha256(json.dumps(
        {name: value for name, value in raw.items() if name != "certificate_sha256"},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def restore_arguments(case):
    return dict(expected_history=case["history"], expected_law=case["law"],
        expected_source_history_dto=case["source"])


def test_independent_saved_source_preserves_strict_full_angle_owner_signs():
    source = saved_source()
    depth = initial_depth_divided_clock_bounds(source)
    fresh = fresh_depth_whole_two_clock_bounds(source)
    for y, z in (depth, fresh):
        assert 0 < y[0] <= y[1]
        assert 0 < z[0] <= z[1]
        assert F(602124736259, 10**31) < z[0]/y[1]
        assert z[1]/y[0] < F(602124736260, 10**31)
        # These are uniform homogeneous angular sectors, rather than a
        # representative angle at which a fresh gate happens to pass.
        assert y[0]*F(1, 10**19)-z[1] > 0
        assert y[1]*F(1, 10**21)-z[0] < 0


def test_independent_positive_width_angle_sector_has_exactly_two_quarter_changes():
    source = saved_source()
    rows, norm = joint_pullback_coefficients(source)
    degree = len(rows)-1
    low, high = F(1, 10**56), F(2, 10**55)
    # Every carrier (0,1,lambda), lambda in this complete finite strip,
    # has a strictly convex z pullback and two sign changes.
    z_y = tuple(row[2][1] for row in rows)
    z_z = tuple(row[2][2] for row in rows)
    second_y, second_z = polynomial_bounds(z_y, 2), polynomial_bounds(z_z, 2)
    error = derivative_tail(norm, degree, 2)*(1+high)
    assert second_y[0]+min(low*second_z[0], high*second_z[0])-error > 0
    error = derivative_tail(norm, degree)*(1+high)
    for clock, sign in ((F(1, 10**11), -1), (F(2, 10**11), 1), (F(1), 1)):
        first, second = polynomial_at(z_y, clock), polynomial_at(z_z, clock)
        bounds = (first+min(low*second, high*second)-error,
            first+max(low*second, high*second)+error)
        assert bounds[1] < 0 if sign < 0 else bounds[0] > 0
    native_width_low = (high-low)/(1+high*high)
    assert native_width_low > 0
    assert native_width_low <= high-low


def test_full_joint_coefficients_equal_independent_exact_matrix_convolution(line_case):
    value = line_case["certificate"]
    independent, norm = joint_pullback_coefficients(line_case["source"], value.order)
    for column in value.columns:
        assert column.rate_bound == norm
        assert column.coefficients == tuple(tuple(row[i][column.initial_coordinate_index]
            for i in range(3)) for row in independent)
    assert all(item != 0 for axis in (value.donor_axis, value.receiver_axis) for item in axis)


def test_complete_quarters_partition_the_original_exact_computational_line(line_case):
    value = line_case["certificate"]
    assert {row.receiver_fragment_id for row in value.quarters} == {
        "octant:0", "octant:1", "octant:2", "octant:3"}
    for row in value.quarters:
        start, core, end = (row.start_strip_computational_angle_bounds,
            row.core_computational_angle_bounds, row.end_strip_computational_angle_bounds)
        assert start.lo == 0 < start.hi == core.lo < core.hi == end.lo < end.hi
        assert end.hi == row.computational_extent_radians
        assert start.width+core.width+end.width == row.computational_extent_radians
        assert not row.native_extent_radians.lo <= row.computational_extent_radians <= row.native_extent_radians.hi
        assert len(row.possible_donor_trace_ids) == len(set(row.possible_donor_trace_ids)) == 3
        assert row.start_axis_coordinate_index != row.end_axis_coordinate_index
    assert value.native_strip_width_upper_bound_radians < F(1, 10**30)


@pytest.mark.parametrize("axis_index", [0, 1])
def test_independent_turning_numerator_brackets_unique_ratio_turn_on_entire_clock(line_case, axis_index):
    value = line_case["certificate"]
    axis = value.axes[axis_index]
    polynomial, error = turning_numerator_coefficients(line_case["source"], axis.axis_coordinate_index)
    derivative = polynomial_bounds(polynomial, 1)
    increasing = axis.turning_numerator_derivative_bounds.lo > 0
    assert derivative[0]-40*error > 0 if increasing else derivative[1]+40*error < 0
    first = polynomial_at(polynomial, axis.unique_turning_clock_bounds.lo)
    last = polynomial_at(polynomial, axis.unique_turning_clock_bounds.hi)
    assert first+error < 0 < last-error if increasing else last+error < 0 < first-error
    assert axis.diagonal_bounds.lo > 0
    assert 0 < axis.offdiagonal_ratio_absolute_bound < F(1, 10**30)
    assert axis.positive_axis_owner_depth_over_clock_bounds.lo > 0
    assert axis.positive_axis_other_support_bounds.lo > 0
    assert axis.positive_axis_fresh_owner_derivative_bounds.lo > 0
    assert axis.negative_axis_fresh_owner_derivative_bounds.hi < 0


def test_whole_line_selection_retains_proved_negative_seams_without_new_admission(line_case):
    from tectonics.geometric_native_active_contact_line import native_active_contact_line_to_dict
    value = line_case["certificate"]
    raw = native_active_contact_line_to_dict(value)
    assert value.whole_line_selection_proved and not value.unresolved_positive_measure
    assert raw["negative_axis_seams"] == "proved_negative_owner_entry_retain_current_frozen_binding"
    assert raw["inconclusive_owner_entry_proof_retained_as_if_negative"] is False
    assert raw["closed_axis_endpoint_policy"] == "retain_both_original_closed_ancestries_null_interface_measure"
    for key in ("geometric_proof_from_point_samples", "tiny_cross_axis_strips_discarded",
            "geometric_crossings_identified_with_material_handovers", "material_label_reprojected_during_segment",
            "root_midpoint_selected", "identifier_tie_break_used", "physical_pair_admission_authorized",
            "candidate_mechanics_adopted"):
        assert raw[key] is False
    assert raw["accepted_native_force_intervals"] == 0
    assert native_physical_history_binding_to_dict(line_case["history"]) == line_case["source"]


def test_whole_line_restart_is_byte_identical_from_external_source_and_law(line_case):
    from tectonics.geometric_native_active_contact_line import (
        native_active_contact_line_from_dict, native_active_contact_line_to_dict,
    )
    raw = native_active_contact_line_to_dict(line_case["certificate"])
    replay = native_active_contact_line_from_dict(deepcopy(raw), **restore_arguments(line_case))
    assert native_active_contact_line_to_dict(replay) == raw
    changed = restore_arguments(line_case)
    changed["expected_law"] = ContactHistoryTransportLaw(TRANSPORT_LAW_MODEL, "Different explicit lineage")
    with pytest.raises(ValueError):
        native_active_contact_line_from_dict(deepcopy(raw), **changed)


@pytest.mark.parametrize("ratio", [F(1, 10**56), F(1, 10**55), F(2, 10**55)])
def test_positive_width_two_switch_sector_selects_ordered_fresh_implicit_branches(line_case, ratio):
    from tectonics.geometric_native_active_contact_line import select_native_active_contact_line_carrier
    selected = select_native_active_contact_line_carrier(line_case["certificate"], "octant:3",
        (F(0), F(1), ratio), expected_source_history_dto=line_case["source"])
    assert selected.classification == "positive_fresh_owner_unique_monotone_implicit_branches"
    assert [branch.known_existence for branch in selected.implicit_branches] == ["proved_present"]*2
    assert [branch.incoming_donor_fragment_id for branch in selected.implicit_branches] == ["octant:6", "octant:7"]
    assert [branch.incoming_seam_sign for branch in selected.implicit_branches] == [-1, 1]
    assert all(branch.owner_depth_derivative_bounds.lo > 0 for branch in selected.implicit_branches)


def test_negative_antipodal_sector_retains_frozen_original_binding(line_case):
    from tectonics.geometric_native_active_contact_line import select_native_active_contact_line_carrier
    selected = select_native_active_contact_line_carrier(line_case["certificate"], "octant:0",
        (F(0), F(-1), F(-1, 10**55)), expected_source_history_dto=line_case["source"])
    assert selected.classification == "proved_negative_owner_entry_retain_original_frozen_binding"
    assert selected.original_donor_fragment_id == "octant:4"
    assert selected.implicit_branches == ()


def test_mixed_quarter_no_seam_core_does_not_require_a_fresh_entry(line_case):
    from tectonics.geometric_native_active_contact_line import select_native_active_contact_line_carrier
    # Near the independent mixed sign threshold, both Y/Z supports remain
    # away from their ~1e-33 seam strips.  It keeps prepared frozen material.
    for ray in ((F(0), F(1), F(-1)), (F(0), F(1, 10**20), F(-1))):
        selected = select_native_active_contact_line_carrier(line_case["certificate"], "octant:2", ray,
            expected_source_history_dto=line_case["source"])
        assert selected.classification == "original_frozen_binding_no_quarter_change"
        assert selected.original_donor_fragment_id == "octant:6"
        assert selected.implicit_branches == ()


def test_exact_turning_range_condition_is_kept_without_promoting_interval_midpoint(line_case):
    from tectonics.geometric_native_active_contact_line import (
        native_active_contact_line_carrier_to_dict, select_native_active_contact_line_carrier,
    )
    axis = next(row for row in line_case["certificate"].axes if row.axis_coordinate_index == 1)
    ratio = axis.seam_ratio_at_turning_bounds.midpoint
    selected = select_native_active_contact_line_carrier(line_case["certificate"], "octant:3",
        (F(0), F(1), ratio), expected_source_history_dto=line_case["source"])
    assert any(branch.known_existence == "exact_analytic_condition" for branch in selected.implicit_branches)
    raw = native_active_contact_line_carrier_to_dict(selected, expected_source_history_dto=line_case["source"])
    assert raw["implicit_branch_condition_refers_to_exact_expressions"] is True
    assert raw["interval_endpoint_or_midpoint_promoted_to_exact_root"] is False
    assert raw["numerical_tolerance_used_for_branch_acceptance"] is False


def test_closed_original_endpoint_keeps_both_ancestries_and_no_finite_point_area(line_case):
    from tectonics.geometric_native_active_contact_line import (
        native_active_contact_line_carrier_to_dict, select_native_active_contact_line_carrier,
    )
    selected = select_native_active_contact_line_carrier(line_case["certificate"], "octant:3",
        (F(0), F(1), F(0)), expected_source_history_dto=line_case["source"])
    assert selected.classification == "closed_original_endpoint_both_ancestries_zero_interface_measure"
    assert set(selected.closed_endpoint_donor_fragment_ids) == {"octant:6", "octant:7"}
    assert selected.implicit_branches == ()
    raw = native_active_contact_line_carrier_to_dict(selected, expected_source_history_dto=line_case["source"])
    assert raw["original_closed_endpoint_has_finite_interface_area"] is False
    assert raw["identifier_tie_break_used"] is False


@pytest.mark.parametrize("mutation", ["branch", "fresh", "clock", "condition", "ray", "adoption"])
def test_carrier_restart_rejects_resealed_implicit_branch_substitution(line_case, mutation):
    from tectonics.geometric_native_active_contact_line import (
        native_active_contact_line_carrier_from_dict, native_active_contact_line_carrier_to_dict,
        select_native_active_contact_line_carrier,
    )
    ray = (F(0), F(1), F(1, 10**55))
    selected = select_native_active_contact_line_carrier(line_case["certificate"], "octant:3", ray,
        expected_source_history_dto=line_case["source"])
    raw = deepcopy(native_active_contact_line_carrier_to_dict(selected,
        expected_source_history_dto=line_case["source"]))
    if mutation == "branch":
        raw["implicit_branches"][0]["incoming_donor_fragment_id"] = "octant:7"
    elif mutation == "fresh":
        raw["implicit_branches"][0]["owner_depth_derivative_bounds"] = {"lo": [-1, 1], "hi": [1, 1]}
    elif mutation == "clock":
        raw["implicit_branches"][0]["time_branch"] = "midpoint_clock"
    elif mutation == "condition":
        raw["implicit_branches"][0]["exact_existence_condition"] = "always_true"
    elif mutation == "ray":
        raw["receiver_reference_coordinates"][2] = [0, 1]
    else:
        raw["accepted_native_force_intervals"] = 1
    raw["selection_sha256"] = hashlib.sha256(json.dumps(
        {name: value for name, value in raw.items() if name != "selection_sha256"},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    args = dict(expected_certificate=line_case["certificate"], expected_receiver_fragment_id="octant:3",
        expected_receiver_reference_coordinates=ray, expected_source_history_dto=line_case["source"])
    with pytest.raises(ValueError):
        native_active_contact_line_carrier_from_dict(raw, **args)


@pytest.mark.parametrize("mutation", ["coefficient", "strip_measure", "turn_clock", "fresh_gate",
    "bank", "source", "retention", "unknown", "force_adoption"])
def test_resealed_whole_line_restart_rejects_semantic_substitution(line_case, mutation):
    from tectonics.geometric_native_active_contact_line import (
        native_active_contact_line_from_dict, native_active_contact_line_to_dict,
    )
    raw = deepcopy(native_active_contact_line_to_dict(line_case["certificate"]))
    if mutation == "coefficient":
        raw["columns"][0]["coefficients"][1][0] = [0, 1]
    elif mutation == "strip_measure":
        raw["common_computational_strip_width_radians"] = [0, 1]
    elif mutation == "turn_clock":
        raw["axes"][0]["unique_turning_clock_bounds"] = {"lo": [0, 1], "hi": [1, 1]}
    elif mutation == "fresh_gate":
        raw["axes"][0]["positive_axis_fresh_owner_derivative_bounds"] = {"lo": [-1, 1], "hi": [1, 1]}
    elif mutation == "bank":
        raw["quarters"][0]["original_donor_fragment_id"] = "octant:7"
    elif mutation == "source":
        raw["source_history"]["actual_candidate"]["omega_rad_per_myr"][0][1][0] = 0.0
    elif mutation == "retention":
        raw["negative_axis_seams"] = "switch_any_geometric_seam"
    elif mutation == "unknown":
        raw["unexpected"] = True
    else:
        raw["physical_pair_admission_authorized"] = True
        raw["accepted_native_force_intervals"] = 1
    reseal(raw)
    with pytest.raises(ValueError):
        native_active_contact_line_from_dict(raw, **restore_arguments(line_case))


@pytest.mark.parametrize("precision,order", [(True, 16), (95, 16), (128, True), (128, 15)])
def test_invalid_whole_line_proof_budget_is_refused(line_case, precision, order):
    from tectonics.geometric_native_active_contact_line import certify_native_active_contact_line
    with pytest.raises(ValueError):
        certify_native_active_contact_line(line_case["history"], law=line_case["law"],
            expected_source_history_dto=line_case["source"], precision=precision, order=order)
