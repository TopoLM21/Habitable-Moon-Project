"""Whole-source partition, literal ledgers and family depth-profile ownership.

The independent expected atom classification is not a new quadrature rule:
the three microscopic active cuts carry positive continuous source measure
while containing none of the original sixteen mechanical atoms.
"""
from copy import deepcopy
from dataclasses import fields, replace
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json

import pytest

from tectonics.geometric_native_pose import FractionInterval as I
from tectonics import geometric_native_common_source_allocation as api
from tectonics.geometric_native_common_contact_cuts import bind_common_contact_cuts
from tests.test_geometric_native_common_contact_ports import _reference, _case


PROVENANCE = "NEW family-normalized depth lift over the whole literal source partition"


@lru_cache(maxsize=1)
def _cuts():
    return bind_common_contact_cuts(_reference(), precision=96)


@lru_cache(maxsize=1)
def _before():
    return json.loads(dict(_reference().complete_source_dto_json)["before"])


def _current_trace(trace_id):
    return next(row for row in _before()["snapshot"]["spatial_memory"]["traces"]
                if row["trace_id"] == trace_id)


def _source_atoms(trace_id=None):
    rows = _before()["original_physical_history"]["quadrature_histories"]
    return tuple(row for row in rows if trace_id is None or row["trace_id"] == trace_id)


def _fraction(raw):
    return F(*raw)


def _literal(raw):
    return F.from_float(raw)


def _reseal(raw, value):
    """Make a syntactically valid forgery; source replay must still reject it."""
    body = {field.name: raw[field.name] for field in fields(value)
            if field.name != "binding_sha256"}
    raw["binding_sha256"] = hashlib.sha256(api._encoded(body)).hexdigest()
    return raw


def _profiles(linear=False):
    return {row["trace_id"]: (F(0), F(2)) if linear else (F(1),)
            for row in _before()["snapshot"]["spatial_memory"]["traces"]}


@lru_cache(maxsize=2)
def _allocation(linear=False):
    return api.bind_native_common_source_allocation(_reference(), _cuts(),
        depth_profiles=_profiles(linear), provenance=PROVENANCE, precision=96)


def _family(parent, linear=False):
    return next(family for family in _allocation(linear).families
                if family.receiver_original_parent_id == parent)


def _cell(parent, donor=None, linear=False):
    if donor is None:
        return next(cell for cell in _allocation(linear).cells
                    if not cell.active and cell.receiver_original_parent_id == parent)
    cut_id = f"retained-contact:{donor}:{parent}"
    return api.native_common_source_allocation_cell_for_cut(_allocation(linear), cut_id)


def _source_budget(atoms):
    keys = ("contact_energy_j", "bulk_energy_j", "cumulative_external_work_j",
            "cumulative_friction_work_j", "cumulative_viscous_work_j",
            "cumulative_numerical_dissipation_j")
    epochs = ("prepared_state", "previous_state", "unaccepted_candidate_state")
    return {
        epoch: {key: sum((_literal(atom[epoch][key]) for atom in atoms), F(0))
                for key in keys}
        for epoch in epochs
    }


def test_four_original_families_and_eight_cells_account_for_the_complete_source():
    allocation = _allocation()
    assert api.validate_native_common_source_allocation(allocation,
        reference=_reference(), cuts=_cuts()) is allocation
    assert allocation.source_snapshot_sha256 == _reference().source_snapshot_sha256
    assert allocation.source_epoch_sha256 == _reference().source_epoch_sha256
    assert allocation.cut_set_snapshot_sha256 == _cuts().set_snapshot_sha256
    assert len(allocation.families) == 4 and len(allocation.cells) == 8
    assert {family.receiver_original_parent_id for family in allocation.families} == {
        "octant:0", "octant:1", "octant:2", "octant:3"}
    active = {cell.cut_id for cell in allocation.cells if cell.active}
    assert active == {cut.cut_id for cut in _cuts().cuts if cut.receiving_trace_id is not None}
    assert sum(not cell.active for cell in allocation.cells) == 3
    assert allocation.symbolic_full_source_reference_area_closure_certified
    assert allocation.whole_current_continuous_memory_retained
    assert not allocation.old_world_force_equivalence_certified
    assert not allocation.full_face_physical_memory_measure_transfer_certified
    assert not allocation.old_rod_energy_or_prestress_transferred
    assert not allocation.force_equilibrium_certified
    assert not allocation.new_native_interval_accepted


def test_shared_projective_boundaries_close_each_family_by_exact_cancellation():
    allocation = _allocation()
    boundaries = {row.endpoint_id: row for row in allocation.boundaries}
    assert len(boundaries) == len(allocation.boundaries) == 12
    for family in allocation.families:
        cells = [next(row for row in allocation.cells if row.cell_id == cell_id)
                 for cell_id in family.cell_ids]
        assert cells[0].lower_endpoint_id == family.trace_id + ":zero"
        assert cells[-1].upper_endpoint_id == family.trace_id + ":extent"
        for left, right in zip(cells, cells[1:]):
            assert left.upper_endpoint_id == right.lower_endpoint_id
            assert left.computational_angle_endpoint_bounds_radians[1] == right.computational_angle_endpoint_bounds_radians[0]
        # Independent algebra of named endpoints, without adding uncertain
        # interval widths and treating that enclosure as an exact total.
        coefficients = {}
        for cell in cells:
            for endpoint_id, coefficient in cell.source_area_endpoint_linear_expression:
                coefficients[endpoint_id] = coefficients.get(endpoint_id, F(0)) + coefficient
        coefficients = {key: value for key, value in coefficients.items() if value}
        C = family.source_interface_area_density_m2_per_computational_radian
        assert coefficients == {family.trace_id + ":zero": -C, family.trace_id + ":extent": C}
        total = sum((coefficient * boundaries[key].exact_rational_value_radians
                     for key, coefficient in coefficients.items()), F(0))
        assert total == family.source_reference_interface_area_m2
        assert family.summed_area_endpoint_linear_expression == family.full_area_endpoint_linear_expression
    for parent, suffix in (("octant:2", "sigma8"), ("octant:3", "sigma11")):
        boundary = boundaries[_family(parent).trace_id + ":" + suffix]
        assert boundary.endpoint_kind == "exact_projective_ray"
        assert len(boundary.authentic_endpoint_aliases) == 2
        assert json.loads(boundary.canonical_projective_expression_json)[0] == "cross"


def test_positive_microscopic_and_inactive_supports_do_not_depend_on_gauss_nodes():
    allocation = _allocation()
    assert all(cell.source_reference_area_bounds_m2.lo > 0 for cell in allocation.cells)
    atom_buckets = {entry.source_accounting_bucket for entry in allocation.original_atom_ledger}
    microscopic = (
        _cell("octant:1", "octant:5"),
        _cell("octant:2", "octant:7"),
        _cell("octant:3", "octant:5"),
    )
    for cell in microscopic:
        assert cell.cell_id not in atom_buckets
        assert cell.source_reference_area_bounds_m2.hi < 1
        assert not cell.finite_original_atom_areas_assigned_to_cell
    tail = _cell("octant:2")
    assert tail.cell_id not in atom_buckets
    assert 0 < tail.source_reference_area_bounds_m2.lo <= tail.source_reference_area_bounds_m2.hi < F(1, 10**8)
    assert allocation.active_current_receiver_source_atom_count == 8
    assert allocation.inactive_current_receiver_source_atom_count == 8
    assert allocation.uncertain_endpoint_source_atom_count == 0


def test_whole_current_memory_anchors_and_distinct_source_clocks_remain_literal():
    allocation = _allocation()
    before = _before()
    history = before["original_physical_history"]
    path = history["contact_time_path"]
    assert json.loads(allocation.complete_current_memory_json) == path["step"]["memory"] == before["snapshot"]["spatial_memory"]
    assert json.loads(allocation.complete_original_physical_history_json) == history
    assert json.loads(allocation.complete_current_atom_states_json) == before["snapshot"]["states"]
    assert allocation.complete_source_dto_json == _reference().complete_source_dto_json
    assert allocation.complete_historical_saved_field_json == _reference().saved_field_json
    assert allocation.source_native_elapsed_time_s == allocation.source_memory_elapsed_time_s == F(1000000)
    assert allocation.source_memory_stage_count == 1
    assert allocation.source_contact_elapsed_time_s == F(2000000)
    assert allocation.source_contact_step_count == 2
    assert allocation.source_current_node_epoch == "unaccepted_candidate"
    literal_anchor_differences = []
    for family in allocation.families:
        current = _current_trace(family.trace_id)
        assert json.loads(family.complete_current_memory_trace_json) == current
        anchors = next(row for row in path["step"]["anchor_fields"] if row[0] == family.trace_id)
        assert json.loads(family.complete_endpoint_anchor_fields_json) == anchors
        literal_anchor_differences.append(current["pieces"][0]["plastic_slip"]["coefficients"][0] != anchors[2]["coefficients"][0])
        reached = F(0)
        for piece in current["pieces"]:
            assert _literal(piece["low_radians"]) <= reached
            reached = max(reached, _literal(piece["high_radians"]))
            assert len(piece["plastic_slip"]["coefficients"]) == len(piece["cumulative_slip"]["coefficients"]) == 5
        assert reached == family.source_computational_extent_radians
    assert any(literal_anchor_differences)
    assert not allocation.source_state_or_thermal_state_advanced


def test_all_sixteen_atom_histories_have_one_literal_unaccepted_state_and_bucket():
    allocation = _allocation()
    atoms = {atom["point_id"]: atom for atom in _source_atoms()}
    current_states = dict(_before()["snapshot"]["states"])
    entries = {entry.point_id: entry for entry in allocation.original_atom_ledger}
    assert len(entries) == len(allocation.original_atom_ledger) == 16
    assert set(entries) == set(atoms) == set(current_states)
    expected_buckets = {
        "octant:0": "inactive:octant:0",
        "octant:1": "inactive:octant:1",
        "octant:2": "retained-contact:octant:6:octant:2",
        "octant:3": "retained-contact:octant:7:octant:3",
    }
    for point_id, entry in entries.items():
        atom = atoms[point_id]
        assert json.loads(entry.complete_original_atom_history_json) == atom
        assert json.loads(entry.complete_current_source_state_json) == current_states[point_id] == atom["unaccepted_candidate_state"]
        assert entry.computational_angle_exact_radians == _fraction(atom["computational_angle"])
        assert entry.original_area_m2 == _literal(atom["original_area_m2"])
        assert entry.source_accounting_bucket == expected_buckets[entry.receiver_original_parent_id]
        assert entry.source_current_node_epoch == "unaccepted_candidate"
        assert entry.source_contact_elapsed_time_s == F(2000000) and entry.source_contact_step_count == 2
        assert not entry.assigned_to_new_physical_material_point_or_area


def test_all_source_energy_and_work_budgets_close_as_exact_literal_atom_sums():
    allocation = _allocation()
    all_atoms = _source_atoms()
    all_budget = allocation.all16_source_atom_budget
    expected = _source_budget(all_atoms)
    assert all_budget.count == 16
    assert set(all_budget.point_ids) == {atom["point_id"] for atom in all_atoms}
    assert all_budget.original_area_m2 == sum((_literal(atom["original_area_m2"]) for atom in all_atoms), F(0))
    assert {epoch: dict(stores) for epoch, stores in all_budget.source_epoch_work_and_energy_j} == expected
    increments = {key: expected["unaccepted_candidate_state"][key] - expected["previous_state"][key]
                  for key in expected["previous_state"]}
    assert dict(all_budget.current_minus_previous_work_and_energy_j) == increments
    entries = {entry.point_id: entry for entry in allocation.original_atom_ledger}
    budgets = dict(allocation.source_bucket_atom_budgets)
    assert sum(budget.count for budget in budgets.values()) == 16
    for bucket, budget in budgets.items():
        subset = tuple(atom for atom in all_atoms if entries[atom["point_id"]].source_accounting_bucket == bucket)
        assert set(budget.point_ids) == {atom["point_id"] for atom in subset}
        assert {epoch: dict(stores) for epoch, stores in budget.source_epoch_work_and_energy_j} == _source_budget(subset)
    for family in allocation.families:
        atoms = _source_atoms(family.trace_id)
        C = family.source_radius_m * _literal(atoms[0]["original_material"]["contact_height_m"]) / _literal(atoms[0]["original_frame"][0])
        assert family.source_interface_area_density_m2_per_computational_radian == C
        assert family.source_reference_interface_area_m2 == C * _literal(_current_trace(family.trace_id)["extent_radians"])
        # This source actually differs by about3.19e-8 m². The two measures
        # must remain distinct even when their decimal displays look equal.
        assert sum((_literal(atom["original_area_m2"]) for atom in atoms), F(0)) != family.source_reference_interface_area_m2
    assert not allocation.literal_atom_area_equated_to_continuous_source_measure
    assert not allocation.global_work_handover_certified


def test_uniform_and_linear_family_profiles_preserve_the_same_angular_marginal():
    uniform, linear = _allocation(), _allocation(True)
    for first, second in zip(uniform.families, linear.families):
        assert first.trace_id == second.trace_id
        assert first.depth_profile_coefficients == (F(1),)
        assert second.depth_profile_coefficients == (F(0), F(2))
        assert first.depth_profile_integral_exact == second.depth_profile_integral_exact == 1
        assert first.source_reference_interface_area_m2 == second.source_reference_interface_area_m2
    for first, second in zip(uniform.cells, linear.cells):
        assert first.source_reference_area_bounds_m2 == second.source_reference_area_bounds_m2
        assert second.depth_profile_coefficients == (F(0), F(2))
    # Distinct donor/receiver roof offsets give different genuine depth
    # spans for the two C8 cuts. A normalized conditional lift preserves
    # each disjoint angular marginal; it is not an inherited common depth.
    long, tiny = _cell("octant:2", "octant:6"), _cell("octant:2", "octant:7")
    assert long.common_physical_radius_cubed_bounds_m3 != tiny.common_physical_radius_cubed_bounds_m3
    assert not linear.inherited_three_dimensional_source_depth_profile


def _measure(index, profile=(F(1),)):
    from tectonics.geometric_native_common_contact_measure import declare_native_common_contact_reference_measure
    from tectonics.geometric_native_common_contact_state import observe_native_common_contact_state
    response = _case(index)
    state = observe_native_common_contact_state(_reference(), response, precision=96)
    declaration = declare_native_common_contact_reference_measure(_reference(), response, state,
        provenance=PROVENANCE, depth_profile_coefficients=profile)
    return response, state, declaration


def test_matching_family_profile_links_both_genuine_cuts_without_a_second_area_weight():
    allocation = _allocation(True)
    for index in (2, 3):
        response, state, declaration = _measure(index, (F(0), F(2)))
        cell = api.validate_native_common_source_allocation_measure(allocation, declaration,
            reference=_reference(), port_response=response, state=state)
        assert cell.cut_snapshot_sha256 == response.port.cut.cut_snapshot_sha256
        assert cell.trace_id == declaration.receiving_trace_id
        assert cell.depth_profile_coefficients == (F(0), F(2))
        assert declaration.source_reference_interface_area_m2 == _family("octant:2", True).source_reference_interface_area_m2
        assert cell.source_reference_area_bounds_m2.hi < declaration.source_reference_interface_area_m2
        assert not cell.finite_original_atom_areas_assigned_to_cell


def test_measure_link_rejects_changed_family_profile_and_same_family_foreign_cut():
    response, state, declaration = _measure(2)
    with pytest.raises(ValueError, match="family profile"):
        api.validate_native_common_source_allocation_measure(_allocation(True), declaration,
            reference=_reference(), port_response=response, state=state)
    other_response, other_state, _ = _measure(3)
    assert other_state.receiving_trace_id == state.receiving_trace_id
    with pytest.raises(ValueError):
        api.validate_native_common_source_allocation_measure(_allocation(), declaration,
            reference=_reference(), port_response=other_response, state=other_state)
    with pytest.raises(ValueError):
        api.validate_native_common_source_allocation_measure(_allocation(), declaration, state=other_state)


def test_missing_extra_and_invalid_profiles_cannot_change_the_four_original_families():
    valid = _profiles()
    missing = dict(valid)
    missing.pop(next(iter(missing)))
    extra = dict(valid, fabricated_hot_receiver=(F(1),))
    nonnormal = dict(valid)
    nonnormal[next(iter(nonnormal))] = (F(1), F(0))
    negative = dict(valid)
    negative[next(iter(negative))] = (F(-1), F(3))
    for profiles in (missing, extra, nonnormal, negative):
        with pytest.raises(ValueError):
            api.bind_native_common_source_allocation(_reference(), _cuts(),
                depth_profiles=profiles, provenance=PROVENANCE, precision=96)
    with pytest.raises(ValueError, match="provenance"):
        api.bind_native_common_source_allocation(_reference(), _cuts(),
            depth_profiles=valid, provenance=" ", precision=96)
    incomplete = replace(_cuts(), cuts=_cuts().cuts[:-1])
    with pytest.raises(ValueError):
        api.bind_native_common_source_allocation(_reference(), incomplete,
            depth_profiles=valid, provenance=PROVENANCE, precision=96)
    counterfeit = replace(_cuts().cuts[2], source_parameter_expression="borrowed-cut-with-the-same-id")
    with pytest.raises(ValueError):
        api.native_common_source_allocation_cell_for_cut(_allocation(), counterfeit)


def test_true_hot_boundary_stays_separate_from_original_source_law_and_budget():
    allocation = _allocation()
    hot = next(cut for cut in _cuts().cuts if cut.receiving_trace_id is None)
    assert allocation.excluded_hot_cut_snapshot_sha256 == hot.cut_snapshot_sha256
    assert hot.receiving_family_id not in {family.trace_id for family in allocation.families}
    assert not allocation.hot_receiving_contact_law_recovered
    for cut_or_id in (hot, hot.cut_id, "retained-contact:invented"):
        with pytest.raises(ValueError, match="unallocated or HOT"):
            api.native_common_source_allocation_cell_for_cut(allocation, cut_or_id)


def test_source_allocation_serde_rejects_resealed_duplicate_budgets_memory_and_clocks():
    allocation = _allocation()
    raw = api.native_common_source_allocation_to_dict(allocation)
    assert api.native_common_source_allocation_from_dict(raw,
        reference=_reference(), cuts=_cuts()) == allocation
    for target in ("duplicate_cell", "atom_area", "memory", "source_clock", "work"):
        changed = deepcopy(raw)
        if target == "duplicate_cell":
            changed["cells"].append(deepcopy(changed["cells"][0]))
        elif target == "atom_area":
            changed["original_atom_ledger"][0]["original_area_m2"] = [1, 1]
        elif target == "memory":
            memory = json.loads(changed["families"][0]["complete_current_memory_trace_json"])
            memory["pieces"][0]["plastic_slip"]["coefficients"][0] = 0.0
            changed["families"][0]["complete_current_memory_trace_json"] = api._encoded(memory).decode()
        elif target == "source_clock":
            changed["source_contact_elapsed_time_s"] = [1000000, 1]
        else:
            history = json.loads(changed["complete_original_physical_history_json"])
            history["unaccepted_candidate_work_ledgers"][0][1] = 0.0
            changed["complete_original_physical_history_json"] = api._encoded(history).decode()
        with pytest.raises(ValueError, match="differs from complete source"):
            api.native_common_source_allocation_from_dict(_reseal(changed, allocation),
                reference=_reference(), cuts=_cuts())
    forged = replace(allocation, cells=allocation.cells + (allocation.cells[0],))
    forged = replace(forged, binding_sha256=api._seal(forged))
    with pytest.raises(ValueError, match="lacks complete authentic replay"):
        api.validate_native_common_source_allocation(forged)


@lru_cache(maxsize=3)
def _geometry(index):
    from tectonics.geometric_native_common_contact_cuts import evaluate_common_contact_cut
    cut = _cuts().cuts[index]
    parameter = cut.parameter_guaranteed_bounds
    cube = cut.common_physical_radius_cubed_bounds_m3
    return evaluate_common_contact_cut(_reference(), cut,
        I(parameter.lo + (parameter.hi - parameter.lo) / 3),
        I(cube.hi - (cube.hi - cube.lo) / 3), precision=96)


def test_direct_allocated_density_uses_the_family_profile_and_one_reference_measure():
    # Use the two genuine C8 cuts. No old constitutive/Hessian query is
    # repeated to evaluate this source measure on the actual cut geometry.
    for index in (2, 3):
        geometry = _geometry(index)
        uniform = api.evaluate_allocated_source_reference_density(_allocation(), _reference(), geometry, precision=96)
        linear = api.evaluate_allocated_source_reference_density(_allocation(True), _reference(), geometry, precision=96)
        assert uniform.receiving_trace_id == linear.receiving_trace_id == _family("octant:2").trace_id
        assert uniform.receiving_native_angle_parameter_derivative_bounds == I(-1)
        general = uniform.general_receiving_native_angle_parameter_derivative_bounds
        assert general.lo <= -1 <= general.hi
        assert uniform.normalized_depth_bounds == linear.normalized_depth_bounds == I(F(1, 3))
        assert uniform.depth_profile_value_bounds == I(1)
        assert linear.depth_profile_value_bounds == I(F(2, 3))
        name = "reference_area_density_bounds_m2_per_parameter_per_physical_volume_depth_m"
        base_density = getattr(uniform, name)
        assert getattr(linear, name) == base_density * F(2, 3)
        physical = geometry.area_density_bounds_m2_per_parameter_per_material_m
        assert uniform.physical_area_density_bounds_m2_per_parameter_per_physical_volume_depth_m == physical
        assert uniform.reference_to_physical_area_density_ratio_bounds == base_density / physical
        cube = geometry.cut.common_physical_radius_cubed_bounds_m3
        R = _family("octant:2").source_radius_m
        assert uniform.full_physical_volume_depth_span_m == (cube.hi - cube.lo) / (3 * R**2)
        assert not uniform.extra_support_fraction_multiplier_applied
        assert not uniform.finite_point_area_assigned
        assert not uniform.globally_integrated_force_solution


def test_direct_density_serde_rejects_foreign_cut_profile_hot_and_resealed_area_flags():
    allocation, geometry = _allocation(), _geometry(2)
    density = api.evaluate_allocated_source_reference_density(allocation, _reference(), geometry, precision=96)
    raw = api.allocated_source_reference_density_to_dict(density)
    assert api.allocated_source_reference_density_from_dict(raw,
        allocation=allocation, reference=_reference(), cut_bounds=geometry) == density
    with pytest.raises(ValueError, match="another genuine cut query"):
        api.validate_allocated_source_reference_density(density,
            allocation=allocation, reference=_reference(), cut_bounds=_geometry(3))
    with pytest.raises(ValueError, match="another family model"):
        api.validate_allocated_source_reference_density(density,
            allocation=_allocation(True), reference=_reference(), cut_bounds=geometry)
    changed = deepcopy(raw)
    changed["extra_support_fraction_multiplier_applied"] = True
    with pytest.raises(ValueError, match="differs from source"):
        api.allocated_source_reference_density_from_dict(_reseal(changed, density),
            allocation=allocation, reference=_reference(), cut_bounds=geometry)
    forged = replace(density, finite_point_area_assigned=True)
    forged = replace(forged, binding_sha256=api._seal(forged))
    with pytest.raises(ValueError, match="lacks authentic replay"):
        api.validate_allocated_source_reference_density(forged)
    with pytest.raises(ValueError, match="unallocated or HOT"):
        api.evaluate_allocated_source_reference_density(allocation, _reference(), _geometry(5), precision=96)


def _source_field_decimal(field, theta):
    """Independent harmonic oracle from literal coefficients and Taylor sums."""
    x = D(theta.numerator) / D(theta.denominator)
    sine, cosine, sine_term, cosine_term = x, D(1), x, D(1)
    for degree in range(1, 200):
        sine_term *= -x*x / D(2*degree*(2*degree+1))
        cosine_term *= -x*x / D((2*degree-1)*2*degree)
        sine += sine_term
        cosine += cosine_term
    basis = D(1), cosine, sine, 2*cosine*cosine - 1, 2*sine*cosine
    return sum((D.from_float(coefficient) * component
                for coefficient, component in zip(field["coefficients"], basis)), D(0))


@lru_cache(maxsize=4)
def _whole_memory(parent):
    family = _family(parent)
    return api.evaluate_native_common_source_memory(_allocation(), family.trace_id,
        I(0, family.source_computational_extent_radians), precision=96)


def test_all_four_whole_current_memory_families_enclose_independent_literal_fields():
    for family in _allocation().families:
        query = _whole_memory(family.receiver_original_parent_id)
        current = _current_trace(family.trace_id)
        assert query.trace_id == family.trace_id
        assert query.computational_angle_bounds == I(0, family.source_computational_extent_radians)
        assert len(query.memory_piece_intersections) == len(current["pieces"])
        assert query.source_current_memory_sha256 == _allocation().source_current_memory_sha256
        assert query.source_native_elapsed_time_s == query.source_memory_elapsed_time_s == F(1000000)
        assert query.source_contact_elapsed_time_s == F(2000000)
        assert query.source_current_node_epoch == "unaccepted_candidate"
        assert query.literal_current_memory_used
        assert not query.current_memory_replaced_with_previous_or_dense_query
        assert not query.three_dimensional_material_memory_advection_certified
        with localcontext() as context:
            context.prec = 160
            for numerator in (1, 2, 3):
                theta = family.source_computational_extent_radians * F(numerator, 4)
                piece = next(piece for piece in current["pieces"]
                    if _literal(piece["low_radians"]) <= theta <= _literal(piece["high_radians"]))
                for field_name, bounds in (("plastic_slip", query.plastic_slip_bounds_m),
                                           ("cumulative_slip", query.cumulative_slip_bounds_m)):
                    expected = F(_source_field_decimal(piece[field_name], theta))
                    assert bounds.lo <= expected <= bounds.hi
        assert query.plastic_slip_bounds_m.hi < 0 < query.cumulative_slip_bounds_m.lo
    # The fourth source carrier has no active current receiver cut. Its
    # complete nonzero memory is nevertheless available and remains owned.
    inactive = _whole_memory("octant:0")
    assert inactive.plastic_slip_bounds_m != inactive.cumulative_slip_bounds_m != I(0)


def test_authentic_current_memory_queries_keep_both_closed_family_endpoints():
    # The authentic CURRENT source has one piece in each family. Check its
    # real closed endpoints rather than fabricate a multi-piece source seam.
    for family in _allocation().families:
        current = _current_trace(family.trace_id)
        for theta in (F(0), family.source_computational_extent_radians):
            query = api.evaluate_native_common_source_memory(_allocation(), family.trace_id,
                I(theta), precision=96)
            pieces = [piece for piece in current["pieces"]
                      if _literal(piece["low_radians"]) <= theta <= _literal(piece["high_radians"])]
            assert len(query.memory_piece_intersections) == len(pieces) == 1
            assert query.memory_piece_intersections[0].computational_angle_bounds == I(theta)
            assert json.loads(query.memory_piece_intersections[0].original_piece_json) == pieces[0]
            with localcontext() as context:
                context.prec = 160
                for field_name, bounds in (("plastic_slip", query.plastic_slip_bounds_m),
                                           ("cumulative_slip", query.cumulative_slip_bounds_m)):
                    expected = F(_source_field_decimal(pieces[0][field_name], theta))
                    assert bounds.lo <= expected <= bounds.hi
            assert query.all_closed_current_piece_intersections_retained


def test_source_memory_serde_rejects_changed_coefficients_clock_family_and_extent():
    allocation = _allocation()
    query = _whole_memory("octant:0")
    raw = api.native_common_source_memory_query_to_dict(query)
    assert api.native_common_source_memory_query_from_dict(raw, allocation=allocation) == query
    for target in ("coefficient", "clock", "family"):
        changed = deepcopy(raw)
        if target == "coefficient":
            piece = json.loads(changed["memory_piece_intersections"][0]["original_piece_json"])
            piece["plastic_slip"]["coefficients"][0] = 0.0
            changed["memory_piece_intersections"][0]["original_piece_json"] = api._encoded(piece).decode()
        elif target == "clock":
            changed["source_contact_elapsed_time_s"] = [1000000, 1]
        else:
            changed["trace_id"] = _family("octant:1").trace_id
        with pytest.raises(ValueError, match="differs from actual current"):
            api.native_common_source_memory_query_from_dict(_reseal(changed, query), allocation=allocation)
    for angle in (I(-1, 0), I(0, query.source_computational_extent_radians + 1)):
        with pytest.raises(ValueError, match="original computational extent"):
            api.evaluate_native_common_source_memory(allocation, query.trace_id, angle, precision=96)
    with pytest.raises(ValueError, match="authentic original receiving family"):
        api.evaluate_native_common_source_memory(allocation, "fabricated-hot-family", I(0), precision=96)
    forged = replace(query, source_memory_stage_count=2)
    forged = replace(forged, binding_sha256=api._seal(forged))
    with pytest.raises(ValueError, match="lacks actual current-family replay"):
        api.validate_native_common_source_memory_query(forged, allocation=allocation)
