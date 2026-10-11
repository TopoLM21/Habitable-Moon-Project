"""A whole-domain registry owns each receiving history and work lineage once."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from functools import lru_cache
import hashlib
import json

import pytest

from tectonics.geometric_native_physical_history import native_physical_history_binding_to_dict
from tectonics.geometric_native_pose import FractionInterval as I
from tests.test_geometric_native_contact_field_work import independent_interface_integral
from tests.test_geometric_native_contact_field_history import enclosed
from tests.test_geometric_native_first_entry_contact_handover import (
    ENTRY_LABELS, LABELS, get_first_entry_handover_case,
)


F = Fraction


@lru_cache(maxsize=1)
def get_domain_handover_case():
    from tectonics.geometric_native_first_entry_contact_domain_handover import (
        certify_native_first_entry_contact_domain_handover,
        evaluate_native_first_entry_contact_domain_point,
    )
    source = get_first_entry_handover_case()
    registry = certify_native_first_entry_contact_domain_handover(source["history"],
        law=source["law"], expected_source_history_dto=source["source"])
    points = {label: evaluate_native_first_entry_contact_domain_point(registry, binding,
        expected_source_history_dto=source["source"])
        for label, binding in source["bindings"].items()}
    return dict(source=source, registry=registry, points=points)


@pytest.fixture(scope="module")
def domain_handover_case():
    return get_domain_handover_case()


def test_all_receiving_families_are_committed_once_for_complete_causal_branch_registry(
        domain_handover_case):
    case, value = domain_handover_case["source"], domain_handover_case["registry"]
    assert len(value.families) == 4
    assert len(value.branches) == 16
    assert len({family.family_id for family in value.families}) == 4
    families = {family.family_id: family for family in value.families}
    traces = {row.trace_id for row in case["history"].material_state.trace_bindings}
    assert {family.trace_id for family in value.families} == traces
    assert len({family.active_binding_id for family in value.families}) == 4
    for family in value.families:
        assert family.revision == 1
        assert family.consumed_binding_ids == (family.original_binding_id,)
        assert family.active_binding_id != family.original_binding_id
        assert family.memory_reference
        assert family.reference_clock_expression
    observed = set()
    for branch in value.branches:
        mask = branch.mask
        pair = mask.original_donor_fragment_id, mask.original_receiver_fragment_id
        assert pair not in observed
        observed.add(pair)
        assert branch.source_family_id in families
        assert families[branch.source_family_id].trace_id == mask.original_receiver_trace_id
        assert branch.receiver_trace_id == mask.original_receiver_trace_id
        assert branch.donor_trace_id == mask.original_donor_trace_id
        assert branch.target_banks.sub_fragment_id == mask.original_donor_fragment_id
        assert branch.target_banks.over_fragment_id == mask.original_receiver_fragment_id
        assert branch.anchor_expression
        assert branch.frozen_first_event_expression
        assert mask.first_entry_only
        assert not mask.reentry_allocates_again
    assert len(observed) == 16


@pytest.mark.parametrize("label", LABELS)
def test_registered_points_reuse_whole_source_family_without_creating_point_area_or_work(
        domain_handover_case, label):
    case, value = domain_handover_case["source"], domain_handover_case["points"][label]
    registry = domain_handover_case["registry"]
    assert value.registry is registry
    assert value.binding is case["bindings"][label]
    assert value.point_handover == case["handovers"][label]
    assert not value.registered_history_created
    assert value.finite_point_area_m2 is value.finite_point_work_j is None
    if label not in ENTRY_LABELS:
        assert value.status == "no_first_entry"
        assert value.branches == value.source_family_ids == value.source_work_primitives == ()
        return
    assert value.status == "first_entry_field_attachment_committed"
    branch, = value.branches
    attachment, = value.point_handover.attachments
    family_id, = value.source_family_ids
    assert branch.source_family_id == family_id
    assert branch.mask.original_donor_fragment_id == attachment.desired_fragment_pair[0]
    assert branch.mask.original_receiver_fragment_id == attachment.desired_fragment_pair[1]
    primitive, = value.source_work_primitives
    assert primitive.final_time_fraction_bounds == value.binding.time_fraction_bounds
    assert primitive.source_field_query.trace_id == attachment.original_receiver_trace_id
    assert primitive.source_field_query.cumulative_work_density_j_m2 is None
    assert value.point_handover.additional_handover_work_j == F(0)
    assert value.point_handover.additional_handover_dissipation_j == F(0)


def test_actual_cross_pair_six_to_three_consumes_seven_receiving_family_only_once(
        domain_handover_case):
    case = domain_handover_case
    value = case["points"]["transverse_masked_main_entry"]
    branch, = value.branches
    family = next(row for row in case["registry"].families if row.family_id == branch.source_family_id)
    assert family.original_banks.sub_fragment_id == "octant:7"
    assert family.original_banks.over_fragment_id == "octant:3"
    assert branch.target_banks.sub_fragment_id == "octant:6"
    assert branch.target_banks.over_fragment_id == "octant:3"
    assert branch.donor_trace_id != family.trace_id
    assert branch.receiver_trace_id == family.trace_id
    assert family.consumed_binding_ids == (family.original_binding_id,)
    assert len(case["registry"].families) == 4
    assert "octant:2" not in (branch.target_banks.sub_fragment_id, branch.target_banks.over_fragment_id)


def test_all_incoming_bank_strips_have_a_whole_motion_bound_and_own_ritz_shape(
        domain_handover_case):
    case, registry = domain_handover_case["source"], domain_handover_case["registry"]
    history = case["history"]
    supports = {row.donor_trace_id: row for row in registry.source_beam_supports}
    assert len(supports) == 4
    total = history.dt_myr*sum((abs(component) for _, omega in history.omega_records
        for component in omega), F(0))
    assert 0 < total < 1
    maximum = total/(1-total*total)
    radius = F.from_float(history.origin_surface.radius_km)*1000
    for binding in history.material_state.trace_bindings:
        row = supports[binding.trace_id]
        parameter = next(row for row in history.initial_mechanics.bending_parameters
            if row.trace_id == binding.trace_id)
        width = F.from_float(parameter.sub.width_m)
        assert row.donor_fragment_id == binding.donor_fragment_id
        assert row.radius_m == radius
        assert row.width_m == width
        assert row.rotation_path_bound_rad == total
        assert row.max_inward_angle_rad == maximum
        assert row.max_inward_distance_m == radius*maximum < width
        assert row.strip_membership_proved
        ratio = radius*maximum/width
        minimum_shape = (1-ratio)**2*(ratio+2)/2
        assert row.shape_bounds == I(minimum_shape, F(1))
    for label in ENTRY_LABELS:
        point = domain_handover_case["points"][label]
        attachment, = point.point_handover.attachments
        row = supports[attachment.original_donor_trace_id]
        assert F(0) <= attachment.sub_material_chart.inward_distance_m.lo
        assert attachment.sub_material_chart.inward_distance_m.hi <= row.max_inward_distance_m
        assert row.shape_bounds.lo <= attachment.sub_material_chart.shape.lo <= attachment.sub_material_chart.shape.hi <= 1


def test_four_original_work_budgets_use_once_owned_interfaces_and_independent_angular_integrals(
        domain_handover_case):
    case, registry = domain_handover_case["source"], domain_handover_case["registry"]
    history = case["history"]
    assert len(registry.source_work_budgets) == 4
    assert len({row.source_time_integral.source_field_query.trace_id
        for row in registry.source_work_budgets}) == 4
    for budget in registry.source_work_budgets:
        field = budget.source_time_integral.source_field_query
        assert budget.source_time_integral.initial_time_fraction == 0
        assert budget.source_time_integral.final_time_fraction == 1
        assert field.computational_angle_bounds.lo == 0
        extent = F.from_float(next(row for row in history.previous_memory.traces
            if row.trace_id == field.trace_id).extent_radians)
        assert field.computational_angle_bounds.hi == extent
        for name, amount in independent_interface_integral(history, field.trace_id,
                F(0), extent, F(0), F(1)).items():
            enclosed(getattr(budget, name), amount)
        assert budget.friction_dissipation_bounds_j == I(F(0))
        assert not budget.original_nodal_ledgers_replaced
    assert native_physical_history_binding_to_dict(history) == case["source"]
    assert history.accepted_before.elapsed_time_s == 0
    assert history.accepted_before.cumulative_source_work_j == 0
    assert history.accepted_before.receipts == ()


def test_original_parent_balances_and_all_sixteen_full_loaded_atoms_remain_owned_once(
        domain_handover_case):
    case, registry = domain_handover_case["source"], domain_handover_case["registry"]
    history = case["history"]
    assert len(registry.parents) == 8
    assert registry.parents is registry.interface_integral.parent_measures
    assert {row.original_parent_id for row in registry.parents} == {
        row.fragment_id for row in history.material_state.parent_allocations}
    assert registry.original_atoms is history.quadrature_histories
    assert len(registry.original_atoms) == 16
    lineages = [(history.original_checkpoint_sha256, row.point_id) for row in registry.original_atoms]
    assert len(set(lineages)) == len(lineages)
    assert all(row.previous_state.contact_energy_j > 0 for row in registry.original_atoms)
    for row in registry.original_atoms:
        assert row.original_area_m2 == row.previous_state.area_m2 == row.unaccepted_candidate_state.area_m2
    assert registry.whole_domain_field_transfer_committed
    assert registry.whole_domain_handover_work_j == F(0)
    assert registry.whole_domain_handover_dissipation_j == F(0)
    assert registry.whole_domain_handover_energy_change_j == F(0)
    assert not registry.physical_pair_admission_authorized
    assert not registry.candidate_mechanics_adopted
    assert not registry.interface_integral.weighted_contact_energy_integrals_evaluated


def registry_arguments(case):
    return dict(expected_history=case["history"], expected_source_history_dto=case["source"],
        expected_law=case["law"])


def test_global_registry_restart_replays_full_source_once_and_keeps_committed_family_ids(
        domain_handover_case):
    from tectonics.geometric_native_first_entry_contact_domain_handover import (
        native_first_entry_contact_domain_handover_from_dict,
        native_first_entry_contact_domain_handover_to_dict,
    )
    case, value = domain_handover_case["source"], domain_handover_case["registry"]
    raw = native_first_entry_contact_domain_handover_to_dict(value)
    rebuilt = native_first_entry_contact_domain_handover_from_dict(deepcopy(raw), **registry_arguments(case))
    assert native_first_entry_contact_domain_handover_to_dict(rebuilt) == raw
    assert tuple(row.family_id for row in rebuilt.families) == tuple(row.family_id for row in value.families)
    assert tuple(row.consumed_binding_ids for row in rebuilt.families) == tuple(
        row.consumed_binding_ids for row in value.families)


@pytest.mark.parametrize("mutation", ["family_owner", "double_family", "branch_pair", "work_budget",
    "beam_width", "source_work", "accepted_force"])
def test_rehashed_registry_cannot_create_history_change_measure_or_adopt_force(
        domain_handover_case, mutation):
    from tectonics.geometric_native_first_entry_contact_domain_handover import (
        native_first_entry_contact_domain_handover_from_dict,
        native_first_entry_contact_domain_handover_to_dict,
    )
    case, value = domain_handover_case["source"], domain_handover_case["registry"]
    raw = deepcopy(native_first_entry_contact_domain_handover_to_dict(value))
    if mutation == "family_owner":
        raw["families"][0]["trace_id"] = raw["families"][1]["trace_id"]
    elif mutation == "double_family":
        raw["families"].append(deepcopy(raw["families"][0]))
    elif mutation == "branch_pair":
        raw["branches"][0]["source_family_id"] = raw["families"][1]["family_id"]
    elif mutation == "work_budget":
        raw["source_work_budgets"][0]["source_work_bounds_j"] = {"lo": [1, 1], "hi": [1, 1]}
    elif mutation == "beam_width":
        raw["source_beam_supports"][0]["width_m"] = [1, 1]
    elif mutation == "source_work":
        key = "source_history" if "source_history" in raw else "complete_original_source_history"
        raw[key]["actual_candidate"]["states"][0][1]["cumulative_external_work_j"] += 1
    else:
        raw["candidate_mechanics_adopted"] = True
    seal = next(key for key in raw if key.endswith("sha256"))
    raw[seal] = hashlib.sha256(json.dumps({key: item for key, item in raw.items() if key != seal},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    with pytest.raises(ValueError):
        native_first_entry_contact_domain_handover_from_dict(raw, **registry_arguments(case))


def test_typed_registry_cannot_expand_its_spring_history_by_pointwise_duplication(
        domain_handover_case):
    from tectonics.geometric_native_first_entry_contact_domain_handover import (
        validate_native_first_entry_contact_domain_handover,
    )
    case, registry = domain_handover_case["source"], domain_handover_case["registry"]
    forged = replace(registry, families=(*registry.families, registry.families[0]))
    with pytest.raises(ValueError):
        validate_native_first_entry_contact_domain_handover(forged)


@pytest.mark.parametrize("parent,point,receiver", [
    ("octant:6", (F(1, 10**7), F(1), F(-1)), "octant:2"),
    ("octant:7", (F(1, 10**7), F(1), F(1)), "octant:3"),
])
def test_whole_registry_evaluates_new_material_labels_without_registering_new_loaded_history(
        domain_handover_case, parent, point, receiver):
    from tectonics.geometric_native_first_entry_material import certify_native_first_entry_material_point
    from tectonics.geometric_native_first_entry_physical_binding import certify_native_first_entry_physical_binding
    from tectonics.geometric_native_first_entry_contact_domain_handover import (
        evaluate_native_first_entry_contact_domain_point,
        native_first_entry_contact_domain_handover_to_dict,
    )
    case, registry = domain_handover_case["source"], domain_handover_case["registry"]
    assert point not in tuple(row.original_point for row in case["contexts"].values())
    before = native_first_entry_contact_domain_handover_to_dict(registry)
    chronology = certify_native_first_entry_material_point(case["history"].material_state, parent, point)
    binding = certify_native_first_entry_physical_binding(chronology, case["history"])
    value = evaluate_native_first_entry_contact_domain_point(registry, binding,
        expected_source_history_dto=case["source"])
    assert value.status == "first_entry_field_attachment_committed"
    branch, = value.branches
    assert (branch.mask.original_donor_fragment_id, branch.mask.original_receiver_fragment_id) == (parent, receiver)
    attachment, = value.point_handover.attachments
    assert attachment.sub_material_chart.reference_coordinates == point
    assert attachment.source_field_query.time_fraction_bounds == binding.time_fraction_bounds
    assert not value.registered_history_created
    assert len(registry.families) == 4
    assert native_first_entry_contact_domain_handover_to_dict(registry) == before


@pytest.mark.parametrize("label", ["transverse_masked_main_entry", "no_entry"])
def test_registered_point_restart_requires_the_same_whole_registry_and_true_first_binding(
        domain_handover_case, label):
    from tectonics.geometric_native_first_entry_contact_domain_handover import (
        native_first_entry_contact_domain_point_from_dict,
        native_first_entry_contact_domain_point_to_dict,
    )
    case, registry = domain_handover_case["source"], domain_handover_case["registry"]
    value = domain_handover_case["points"][label]
    raw = native_first_entry_contact_domain_point_to_dict(value)
    arguments = dict(expected_registry=registry, expected_binding=case["bindings"][label],
        expected_source_history_dto=case["source"])
    rebuilt = native_first_entry_contact_domain_point_from_dict(deepcopy(raw), **arguments)
    assert native_first_entry_contact_domain_point_to_dict(rebuilt) == raw
    assert rebuilt.source_family_ids == value.source_family_ids
    with pytest.raises(ValueError):
        native_first_entry_contact_domain_point_from_dict(deepcopy(raw),
            **dict(arguments, expected_binding=case["bindings"]["ordinary_entry"]))
