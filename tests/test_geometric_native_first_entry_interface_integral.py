"""Whole first-entry masks keep causal material and contact measures distinct."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import hashlib
import itertools
import json

import pytest

from tectonics.geometric_native_pose import FractionInterval as I
from tests.test_geometric_native_first_entry_contact_handover import (
    ENTRY_LABELS, LABELS, get_first_entry_handover_case,
)
from tests.test_geometric_native_first_entry_contact_measure import oracle, enclosed


F = Fraction


@pytest.fixture(scope="module")
def interface_integral_case():
    from tectonics.geometric_native_first_entry_interface_integral import (
        certify_native_first_entry_interface_integral,
    )
    case = get_first_entry_handover_case()
    value = certify_native_first_entry_interface_integral(case["history"],
        expected_source_history_dto=case["source"])
    return dict(case=case, value=value)


def test_complete_original_parent_and_quarter_masks_are_retained_once(interface_integral_case):
    case, value = interface_integral_case["case"], interface_integral_case["value"]
    history, state = case["history"], case["history"].material_state
    allocations = {row.fragment_id: row for row in state.parent_allocations}
    assert {row.original_parent_id for row in value.parent_measures} == set(allocations)
    assert len(value.parent_measures) == 8
    donors = {row.fragment_id for row in state.parent_allocations if row.owner == 1}
    receivers = {row.fragment_id for row in state.parent_allocations if row.owner == 0}
    assert len(donors) == len(receivers) == 4
    pairs = tuple((row.original_donor_fragment_id, row.original_receiver_fragment_id)
        for row in value.carrier_masks)
    assert len(pairs) == 16
    assert set(pairs) == set(itertools.product(donors, receivers))
    assert len(set(pairs)) == len(pairs)
    bindings = history.material_state.trace_bindings
    donors_by_id = {row.donor_fragment_id: row.trace_id for row in bindings}
    receivers_by_id = {row.receiver_fragment_id: row.trace_id for row in bindings}
    for mask in value.carrier_masks:
        assert mask.original_donor_trace_id == donors_by_id[mask.original_donor_fragment_id]
        assert mask.original_receiver_trace_id == receivers_by_id[mask.original_receiver_fragment_id]
        assert mask.archive_domain_id == allocations[mask.original_donor_fragment_id].archive_domain_id
    assert value.once_only_material_measure_certified
    assert not value.receiver_pair_integrals_numerically_enclosed
    assert not value.weighted_contact_energy_integrals_evaluated


@pytest.mark.parametrize("label", LABELS)
def test_exact_earliest_event_selects_only_original_first_receiver(interface_integral_case, label):
    from tectonics.geometric_native_first_entry_interface_integral import (
        evaluate_native_first_entry_interface_carrier_mask,
    )
    case, value = interface_integral_case["case"], interface_integral_case["value"]
    chronology = case["contexts"][label]
    results = [(mask, evaluate_native_first_entry_interface_carrier_mask(mask, chronology))
        for mask in value.carrier_masks]
    assert all(result in (False, True, None) for _, result in results)
    selected = [mask for mask, result in results if result is True]
    if label not in ENTRY_LABELS:
        assert selected == []
        return
    attachment, = case["handovers"][label].attachments
    mask, = selected
    assert (mask.original_donor_fragment_id, mask.original_receiver_fragment_id) == attachment.desired_fragment_pair
    assert mask.original_receiver_trace_id == attachment.source_field_query.trace_id
    assert mask.original_donor_trace_id == attachment.sub_material_chart.trace_id
    if label == "transverse_masked_main_entry":
        assert mask.original_donor_fragment_id == "octant:6"
        assert mask.original_receiver_fragment_id == "octant:3"
        assert mask.original_receiver_trace_id != mask.original_donor_trace_id


def test_reentry_and_receiver_switch_do_not_erase_or_repeat_first_measure(interface_integral_case):
    from tectonics.geometric_native_first_entry_interface_integral import (
        evaluate_native_first_entry_interface_carrier_mask,
    )
    case, value = interface_integral_case["case"], interface_integral_case["value"]
    chronology = case["contexts"]["transverse_masked_main_entry"]
    by_pair = {(row.original_donor_fragment_id, row.original_receiver_fragment_id): row
        for row in value.carrier_masks}
    assert evaluate_native_first_entry_interface_carrier_mask(by_pair["octant:6", "octant:3"], chronology)
    assert evaluate_native_first_entry_interface_carrier_mask(by_pair["octant:6", "octant:2"], chronology) is False
    assert len([row for row in value.parent_measures if row.original_parent_id == "octant:6"]) == 1
    assert len([row for row in case["history"].material_state.archive_cohorts
        if row.original_parent_id == "octant:6"]) == 1


def test_parent_measure_closure_includes_empty_and_tiny_curved_swept_domains(interface_integral_case):
    case, value = interface_integral_case["case"], interface_integral_case["value"]
    history = case["history"]
    allocations = {row.fragment_id: row for row in history.material_state.parent_allocations}
    radius = F.from_float(history.origin_surface.radius_km)*1000
    for row in value.parent_measures:
        original = allocations[row.original_parent_id]
        assert row.archive_domain_id == original.archive_domain_id
        assert row.measure_reference == original.measure_reference
        assert row.archive_solid_angle_bounds_sr == original.archived_area_bounds_sr
        assert row.parent_solid_angle_bounds_sr == original.parent_area_bounds_sr
        expected = original.archived_area_bounds_sr*radius*radius
        assert row.archive_surface_area_bounds_m2.lo <= expected.lo <= expected.hi <= row.archive_surface_area_bounds_m2.hi
        together = row.live_solid_angle_bounds_sr+row.archive_solid_angle_bounds_sr
        assert together.lo <= row.parent_solid_angle_bounds_sr.lo <= row.parent_solid_angle_bounds_sr.hi <= together.hi
        if row.owner == 0 or row.original_parent_id == "octant:4":
            assert row.archive_solid_angle_bounds_sr == I(F(0))
    donor_five, = [row for row in history.material_state.sweep_certificate.donor_domains
        if row.fragment_id == "octant:5"]
    assert donor_five.domain_case == "endpoint_plus_curved_swept_extra"
    assert donor_five.extra_swept_area_bounds_sr.lo > 0
    assert donor_five.extra_swept_area_bounds_sr.hi < F(1, 10**60)
    five_masks = [row for row in value.carrier_masks if row.original_donor_fragment_id == "octant:5"]
    assert len(five_masks) == 4
    assert all(row.domain_case == donor_five.domain_case for row in five_masks)


def test_contact_interface_measure_and_all_saved_atoms_survive_once_without_bulk_density_invention(
        interface_integral_case):
    case, value = interface_integral_case["case"], interface_integral_case["value"]
    history = case["history"]
    old_atoms = {node.point_id: F.from_float(node.original_area_m2)
        for node in history.quadrature_histories}
    assert len(value.original_interfaces) == 4
    atoms = [atom for row in value.original_interfaces for atom in row.quadrature_atoms]
    assert len(atoms) == len(old_atoms) == 16
    assert dict(atoms) == old_atoms
    assert len({point_id for point_id, _ in atoms}) == len(atoms)
    radius = F.from_float(history.origin_surface.radius_km)*1000
    for interface in value.original_interfaces:
        trace = next(row for row in history.initial_mechanics.fault_traces.traces
            if row.trace_id == interface.trace_id)
        node = next(row for row in history.quadrature_histories if row.trace_id == trace.trace_id)
        extent = next(row for row in history.previous_memory.traces if row.trace_id == trace.trace_id).extent_radians
        density = radius*F.from_float(trace.contact_height_m)/F.from_float(node.original_frame[0])
        assert interface.area_density_m2_per_radian == density
        assert interface.computational_extent_radians == F.from_float(extent)
        assert interface.whole_interface_area_m2 == density*F.from_float(extent)
        assert interface.whole_interface_area_m2 != sum((area for _, area in interface.quadrature_atoms), F(0))


@pytest.mark.parametrize("theta,time", [(F(0), F(0)), (F(1, 10**400), F(1, 10**400)),
    (F(5, 9), F(2, 7)), (F(7, 5), F(1))])
def test_every_receiver_full_rotation_kernel_contains_independent_so3_derivative(
        interface_integral_case, theta, time):
    value = interface_integral_case["value"]
    # Repeated donor masks may share one source quarter kernel; inspect all
    # distinct receiving carriers, including the tiny off-axis angular terms.
    kernels = {mask.original_receiver_trace_id: mask.measure_kernel for mask in value.carrier_masks}
    assert len(kernels) == 4
    for kernel in kernels.values():
        parameters = dict(start=kernel.start, end=kernel.end,
            donor_owner=kernel.donor_owner, receiver_owner=kernel.receiver_owner,
            omega_records=kernel.omega_records, dt_myr=kernel.dt_myr)
        expected = oracle(parameters, theta, time)
        for name in ("source_material_direction", "source_time_derivative", "signed_solid_angle_jacobian"):
            actual, expected_value = getattr(kernel, name), expected[name]
            if isinstance(expected_value, tuple):
                for bound, coordinate in zip(actual, expected_value):
                    enclosed(bound, coordinate)
            else:
                enclosed(actual, expected_value)
        assert not kernel.rectangle_is_once_only_archive_area


def integral_arguments(case):
    return dict(expected_history=case["history"], expected_source_history_dto=case["source"])


def test_integral_restart_replays_complete_source_and_implicit_measure_identity(interface_integral_case):
    from tectonics.geometric_native_first_entry_interface_integral import (
        native_first_entry_interface_integral_from_dict,
        native_first_entry_interface_integral_to_dict,
    )
    case, value = interface_integral_case["case"], interface_integral_case["value"]
    raw = native_first_entry_interface_integral_to_dict(value)
    rebuilt = native_first_entry_interface_integral_from_dict(deepcopy(raw), **integral_arguments(case))
    assert native_first_entry_interface_integral_to_dict(rebuilt) == raw


@pytest.mark.parametrize("mutation", ["mask", "parent_measure", "interface_measure", "atom_area",
    "source_work", "first_event_identity"])
def test_rehashed_integral_cannot_replace_source_or_first_entry_measure(interface_integral_case, mutation):
    from tectonics.geometric_native_first_entry_interface_integral import (
        native_first_entry_interface_integral_from_dict,
        native_first_entry_interface_integral_to_dict,
    )
    case, value = interface_integral_case["case"], interface_integral_case["value"]
    raw = deepcopy(native_first_entry_interface_integral_to_dict(value))
    if mutation == "mask":
        raw["carrier_masks"][0]["original_receiver_fragment_id"] = "octant:3"
    elif mutation == "parent_measure":
        raw["parent_measures"][0]["archive_solid_angle_bounds_sr"] = {"lo": [1, 1], "hi": [1, 1]}
    elif mutation == "interface_measure":
        raw["original_interfaces"][0]["whole_interface_area_m2"] = [1, 1]
    elif mutation == "atom_area":
        raw["original_interfaces"][0]["quadrature_atoms"][0]["literal_area_m2"] = [1, 1]
    elif mutation == "source_work":
        source_key = "source_history" if "source_history" in raw else "complete_original_source_history"
        raw[source_key]["actual_candidate"]["states"][0][1]["cumulative_external_work_j"] += 1
    else:
        raw["receiver_pair_integrals_numerically_enclosed"] = True
    seal = next(key for key in raw if key.endswith("sha256"))
    raw[seal] = hashlib.sha256(json.dumps({key: item for key, item in raw.items() if key != seal},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    with pytest.raises(ValueError):
        native_first_entry_interface_integral_from_dict(raw, **integral_arguments(case))


def test_mask_refuses_forged_material_parent_chronology(interface_integral_case):
    from tectonics.geometric_native_first_entry_interface_integral import (
        evaluate_native_first_entry_interface_carrier_mask,
    )
    case, value = interface_integral_case["case"], interface_integral_case["value"]
    chronology = case["contexts"]["transverse_masked_main_entry"]
    mask = next(row for row in value.carrier_masks if
        (row.original_donor_fragment_id, row.original_receiver_fragment_id) == ("octant:6", "octant:3"))
    with pytest.raises(ValueError):
        evaluate_native_first_entry_interface_carrier_mask(mask,
            replace(chronology, original_parent_id="octant:5"))
