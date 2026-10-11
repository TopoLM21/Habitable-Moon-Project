"""Independent full-rotation work and memory along the actual active path."""
from copy import deepcopy
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache
import json

import pytest

from tectonics.geometric_native_pose import FractionInterval as I
from tests.test_geometric_native_active_contact_binding import (
    decimal, dot, get_active_binding_case, independent_event_root, path_axes,
    reseal, rotate, unit_direction,
)
from tests.test_geometric_native_contact_field_history import independent_field, enclosed


F = Fraction
BUDGET_FIELDS = (
    "source_work_density_bounds_j_m2", "friction_dissipation_density_bounds_j_m2",
    "bulk_energy_increment_density_bounds_j_m2", "contact_energy_increment_density_bounds_j_m2",
    "total_energy_increment_density_bounds_j_m2", "compliance_balance_residual_density_bounds_j_m2",
)
STATE_FIELDS = (
    "free_gap_bounds_m", "free_jump_bounds_m", "plastic_slip_bounds_m", "cumulative_slip_bounds_m",
    "normal_pressure_bounds_pa", "shear_traction_bounds_pa", "bulk_energy_density_bounds_j_m2",
    "contact_energy_density_bounds_j_m2", "total_energy_density_bounds_j_m2",
)


@lru_cache(maxsize=1)
def get_active_work_case():
    from tectonics.geometric_native_active_contact_path_work import query_native_active_contact_path_work
    case = get_active_binding_case()
    path = case["paths"]["two_changes"]
    first, second = path.events
    clocks = dict(initial=I(F(0)), first=first.time_fraction_bounds,
        between=I((first.time_fraction_bounds.hi+second.time_fraction_bounds.lo)/2),
        second=second.time_fraction_bounds, half=I(F(1, 2)), final=I(F(1)))
    queries = {label: query_native_active_contact_path_work(path, clock,
        expected_source_history_dto=case["source"]) for label, clock in clocks.items()}
    unchanged = query_native_active_contact_path_work(case["paths"]["unchanged"], I(F(1)),
        expected_source_history_dto=case["source"])
    return dict(case=case, path=path, clocks=clocks, queries=queries, unchanged=unchanged)


@pytest.fixture(scope="module")
def active_work_case():
    return get_active_work_case()


def atan_small(value):
    total = term = value
    for index in range(1, 4000):
        term *= -value*value
        addition = term/Decimal(2*index+1)
        total += addition
        if abs(addition) < Decimal("1e-590"):
            return total
    raise AssertionError("Independent material angle did not converge")


def independent_pi():
    return 16*atan_small(Decimal(1)/5)-4*atan_small(Decimal(1)/239)


def atan_positive(value, pi):
    if value > 1:
        return pi/2-atan_positive(1/value, pi)
    if value > Decimal(".5"):
        return pi/4+atan_small((value-1)/(value+1))
    return atan_small(value)


def independent_computational_angle(path):
    trace = next(row for row in path.history.initial_mechanics.fault_traces.traces
        if row.trace_id == path.receiver_trace_id)
    c = unit_direction(path.receiver_reference_coordinates)
    x, y = (dot(c, tuple(map(decimal, endpoint))) for endpoint in (trace.start, trace.end))
    pi = independent_pi()
    angle = pi/2 if x == 0 else atan_positive(y/x, pi)
    extent = next(row.extent_radians for row in path.history.previous_memory.traces
        if row.trace_id == path.receiver_trace_id)
    return angle/(pi/2)*decimal(extent)


def independent_geometry(path, donor_trace_id, frozen, time):
    """Direct coordinates at 620 digits, independent of stable production deltas."""
    history = path.history
    trace = next(row for row in history.initial_mechanics.fault_traces.traces if row.trace_id == donor_trace_id)
    parameters = next(row for row in history.initial_mechanics.bending_parameters if row.trace_id == donor_trace_id)
    start, end = (tuple(map(decimal, endpoint)) for endpoint in (trace.start, trace.end))
    # The actual original quarter has inward direction +X. Preserve its
    # original orientation rather than assuming the receiver's own beam.
    from tests.test_geometric_native_active_contact_binding import cross
    inward = tuple(-item for item in cross(start, end))
    x = dot(frozen, inward)
    plane_size = (dot(frozen, start)**2+dot(frozen, end)**2).sqrt()
    radius = decimal(history.origin_surface.radius_km)*1000
    distance = radius*atan_small(x/plane_size)
    coordinate = distance/decimal(parameters.sub.width_m)
    shape = (1-coordinate)**2*(coordinate+2)/2
    initial = {row[0]: row[1:] for row in history.accepted_before.transverse_displacement_m}
    candidate = {row[0]: row[1:] for row in history.actual_candidate.transverse_displacement_m}
    ws0, ws1 = decimal(initial[donor_trace_id][0]), decimal(candidate[donor_trace_id][0])
    wo0, wo1 = decimal(initial[path.receiver_trace_id][1]), decimal(candidate[path.receiver_trace_id][1])
    ws, wo = ws0+time*(ws1-ws0), wo0+time*(wo1-wo0)
    donor_axis, receiver_axis = path_axes(path)
    c = unit_direction(path.receiver_reference_coordinates)
    donor_now = rotate(rotate(frozen, donor_axis, time), receiver_axis, -time)
    relative = tuple((radius+wo)*a-(radius+shape*ws)*b for a, b in zip(c, donor_now))
    node = next(row for row in history.quadrature_histories if row.trace_id == path.receiver_trace_id)
    sine, cosine = map(decimal, node.original_frame)
    horizontal = tuple(map(decimal, node.original_point.horizontal))
    normal = tuple(sine*h+cosine*a for h, a in zip(horizontal, c))
    down = tuple(cosine*h-sine*a for h, a in zip(horizontal, c))
    return dot(normal, relative), dot(down, relative)


def independent_path_state(path, current):
    """Carry geometry, free anchors and irreversible memory through every seam."""
    with localcontext() as context:
        context.prec = 620
        theta = independent_computational_angle(path)
        initial = independent_field(path.history, path.receiver_trace_id, F(theta), F(0))
        gap, jump = initial["free_gap_bounds_m"], initial["free_jump_bounds_m"]
        cumulative = initial["cumulative_slip_bounds_m"]
        initial_gap = gap
        clock = Decimal(0)
        frozen = unit_direction(path.receiver_reference_coordinates)
        donor_trace_id = path.segments[0].donor_trace_id
        stop = decimal(current)
        for event in path.events:
            event_time, new_frozen = independent_event_root(path, event)
            if event_time > stop:
                break
            old_geometry = independent_geometry(path, donor_trace_id, frozen, clock)
            current_geometry = independent_geometry(path, donor_trace_id, frozen, event_time)
            new_jump = jump+current_geometry[1]-old_geometry[1]
            cumulative += abs(new_jump-jump)
            gap += current_geometry[0]-old_geometry[0]
            jump = new_jump
            clock, frozen, donor_trace_id = event_time, new_frozen, event.new_donor_trace_id
        old_geometry = independent_geometry(path, donor_trace_id, frozen, clock)
        current_geometry = independent_geometry(path, donor_trace_id, frozen, stop)
        new_jump = jump+current_geometry[1]-old_geometry[1]
        cumulative += abs(new_jump-jump)
        gap += current_geometry[0]-old_geometry[0]
        jump = new_jump
        trace = next(row for row in path.history.initial_mechanics.fault_traces.traces
            if row.trace_id == path.receiver_trace_id)
        column, contact = decimal(trace.columns.normal_stiffness_pa_m), decimal(trace.law.normal_stiffness_pa_m)
        effective = decimal(1/(1/trace.columns.normal_stiffness_pa_m+1/trace.law.normal_stiffness_pa_m))
        bulk_coefficient = column*(effective/contact-1)**2
        contact_coefficient = effective**2/contact
        square_change = (gap*gap-initial_gap*initial_gap)/2
        work = effective*square_change
        bulk_change, contact_change = bulk_coefficient*square_change, contact_coefficient*square_change
        return dict(free_gap_bounds_m=gap, free_jump_bounds_m=jump, plastic_slip_bounds_m=jump,
            cumulative_slip_bounds_m=cumulative, normal_pressure_bounds_pa=-effective*gap,
            shear_traction_bounds_pa=Decimal(0), bulk_energy_density_bounds_j_m2=bulk_coefficient*gap*gap/2,
            contact_energy_density_bounds_j_m2=contact_coefficient*gap*gap/2,
            total_energy_density_bounds_j_m2=(bulk_coefficient+contact_coefficient)*gap*gap/2,
            source_work_density_bounds_j_m2=work, friction_dissipation_density_bounds_j_m2=Decimal(0),
            bulk_energy_increment_density_bounds_j_m2=bulk_change,
            contact_energy_increment_density_bounds_j_m2=contact_change,
            total_energy_increment_density_bounds_j_m2=bulk_change+contact_change,
            compliance_balance_residual_density_bounds_j_m2=work-bulk_change-contact_change)


@pytest.mark.parametrize("label", ["between", "half", "final"])
def test_full_rotation_state_and_work_match_independent_frozen_material_chain(active_work_case, label):
    query = active_work_case["queries"][label]
    clock = active_work_case["clocks"][label]
    expected = independent_path_state(active_work_case["path"], clock.lo)
    for field in STATE_FIELDS:
        enclosed(getattr(query.current_state, field), expected[field])
    for field in BUDGET_FIELDS:
        enclosed(getattr(query, field), expected[field])
    assert query.current_state.whole_segment_gap_bounds_m.hi <= 0
    assert query.current_state.whole_segment_jump_rate_bounds_m_per_fraction.hi <= 0
    assert query.current_state.plastic_slip_rate_bounds_m_per_fraction == query.current_state.free_jump_rate_bounds_m_per_fraction
    assert query.current_state.cumulative_slip_rate_bounds_m_per_fraction == -query.current_state.free_jump_rate_bounds_m_per_fraction


def test_unchanged_quarter_evolves_its_initial_material_with_full_geometry(active_work_case):
    query = active_work_case["unchanged"]
    path = active_work_case["case"]["paths"]["unchanged"]
    expected = independent_path_state(path, F(1))
    assert query.revision == 0
    assert len(query.segments) == 1
    for field in STATE_FIELDS:
        enclosed(getattr(query.current_state, field), expected[field])
    for field in BUDGET_FIELDS:
        enclosed(getattr(query, field), expected[field])


@pytest.mark.parametrize("label,revision", [("initial", 0), ("first", 1), ("second", 2)])
def test_at_each_handover_the_evolved_state_is_inherited_with_exact_zero_additional_work(active_work_case, label, revision):
    query = active_work_case["queries"][label]
    state = query.current_state
    seed = state.reference.initial_state
    assert query.revision == state.reference.revision == revision
    assert state.at_reference_event
    assert state.projected_gap_change_bounds_m == state.projected_jump_change_bounds_m == I(F(0))
    assert state.receiver_relative_direction_change_bounds == (I(F(0)),)*3
    for field in STATE_FIELDS:
        assert getattr(state, field) == getattr(seed, field)
    for field in BUDGET_FIELDS:
        assert getattr(state, field) == I(F(0))
    if revision:
        assert seed == query.segments[-2]
        assert seed.current_time_fraction_bounds == state.reference.switch_time_fraction_bounds
        assert state.reference.source_field_query.time_fraction_bounds == I(F(0))


def test_second_handover_consumes_the_first_binding_once_and_keeps_accumulated_memory(active_work_case):
    query = active_work_case["queries"]["final"]
    path = active_work_case["path"]
    assert query.revision == 2
    assert query.consumed_binding_ids == tuple(segment.active_binding_id for segment in path.segments[:-1])
    assert len(set(query.consumed_binding_ids)) == 2
    assert query.active_binding_id == path.segments[-1].active_binding_id
    for previous, current in zip(query.segments, query.segments[1:]):
        assert current.reference.initial_state == previous
        assert current.reference.source_gap_at_switch_bounds_m == previous.free_gap_bounds_m
        assert current.reference.source_jump_at_switch_bounds_m == previous.free_jump_bounds_m
        assert current.reference.source_cumulative_slip_at_switch_bounds_m == previous.cumulative_slip_bounds_m
    for field, accumulated in zip(BUDGET_FIELDS, (
            "accumulated_segment_work_density_bounds_j_m2",
            "accumulated_segment_dissipation_density_bounds_j_m2",
            "accumulated_segment_bulk_energy_increment_density_bounds_j_m2",
            "accumulated_segment_contact_energy_increment_density_bounds_j_m2",
            "accumulated_segment_total_energy_increment_density_bounds_j_m2",
            "accumulated_segment_compliance_balance_residual_density_bounds_j_m2")):
        total = getattr(query, field)
        last = getattr(query.current_state, accumulated)
        assert total.lo <= last.hi and last.lo <= total.hi


def test_segment_balance_retains_saved_stiffness_residual_and_zero_friction(active_work_case):
    query = active_work_case["queries"]["final"]
    for segment in query.segments:
        assert segment.friction_dissipation_density_bounds_j_m2 == I(F(0))
        work = segment.source_work_density_bounds_j_m2
        balance = (segment.total_energy_increment_density_bounds_j_m2
            + segment.compliance_balance_residual_density_bounds_j_m2)
        assert work.lo <= balance.hi and balance.lo <= work.hi
        if not segment.at_reference_event:
            residual = segment.compliance_balance_residual_density_bounds_j_m2
            assert residual.lo > 0 or residual.hi < 0
    assert query.friction_dissipation_density_bounds_j_m2 == I(F(0))


def test_positive_clock_reference_cannot_reset_to_original_diagnostic_source(active_work_case):
    from tectonics.geometric_native_active_contact_work import make_native_active_contact_segment_reference
    state = active_work_case["queries"]["second"].current_state
    ref = state.reference
    with pytest.raises(ValueError):
        make_native_active_contact_segment_reference(ref.history, ref.receiver_trace_id,
            ref.donor_trace_id, ref.receiver_original_direction, ref.switch_time_fraction_bounds,
            switch_chronology=ref.switch_chronology, switch_event_sequence=ref.switch_event_sequence,
            root_proof_context=json.loads(ref.root_proof_context_payload_json),
            initial_state=ref.source_field_query, expected_source_history_dto=active_work_case["case"]["source"])


def test_path_refuses_an_interval_that_crosses_two_different_event_expressions(active_work_case):
    from tectonics.geometric_native_active_contact_path_work import query_native_active_contact_path_work
    path = active_work_case["path"]
    first, second = path.events
    with pytest.raises(ValueError):
        query_native_active_contact_path_work(path, I(first.time_fraction_bounds.lo, second.time_fraction_bounds.hi),
            expected_source_history_dto=active_work_case["case"]["source"])


@pytest.mark.parametrize("label", ["first", "second", "final"])
def test_path_work_restart_requires_the_same_external_binding_and_clock(active_work_case, label):
    from tectonics.geometric_native_active_contact_path_work import (
        native_active_contact_path_work_from_dict, native_active_contact_path_work_to_dict,
    )
    query, source = active_work_case["queries"][label], active_work_case["case"]["source"]
    raw = native_active_contact_path_work_to_dict(query, expected_source_history_dto=source)
    kwargs = dict(expected_binding=active_work_case["path"],
        expected_time_fraction_bounds=query.time_fraction_bounds, expected_source_history_dto=source)
    restored = native_active_contact_path_work_from_dict(deepcopy(raw), **kwargs)
    assert native_active_contact_path_work_to_dict(restored, expected_source_history_dto=source) == raw
    kwargs["expected_time_fraction_bounds"] = I(F(1, 3))
    with pytest.raises(ValueError):
        native_active_contact_path_work_from_dict(deepcopy(raw), **kwargs)


@pytest.mark.parametrize("mutation", ["work", "dissipation", "state", "consumed_twice", "affine_double_count", "force_adoption"])
def test_resealed_work_cannot_forge_state_memory_balance_or_adoption(active_work_case, mutation):
    from tectonics.geometric_native_active_contact_path_work import (
        native_active_contact_path_work_from_dict, native_active_contact_path_work_to_dict,
    )
    query, source = active_work_case["queries"]["final"], active_work_case["case"]["source"]
    raw = deepcopy(native_active_contact_path_work_to_dict(query, expected_source_history_dto=source))
    if mutation in ("work", "dissipation"):
        name = "source_work_density_bounds_j_m2" if mutation == "work" else "friction_dissipation_density_bounds_j_m2"
        raw["budgets"][name] = {"lo": [1, 1], "hi": [1, 1]}
    elif mutation == "state":
        raw["segments"][-1]["cumulative_slip_bounds_m"] = {"lo": [0, 1], "hi": [0, 1]}
        reseal(raw["segments"][-1], "work_sha256")
    elif mutation == "consumed_twice":
        raw["consumed_binding_ids"][1] = raw["consumed_binding_ids"][0]
    elif mutation == "affine_double_count":
        raw["original_affine_source_work_included"] = True
    else:
        raw["candidate_mechanics_adopted"] = True
        raw["accepted_native_force_intervals"] = 1
    reseal(raw, "query_sha256")
    with pytest.raises(ValueError):
        native_active_contact_path_work_from_dict(raw, expected_binding=active_work_case["path"],
            expected_time_fraction_bounds=query.time_fraction_bounds, expected_source_history_dto=source)


def test_actual_work_has_no_finite_point_weight_or_global_force_adoption(active_work_case):
    from tectonics.geometric_native_active_contact_path_work import native_active_contact_path_work_to_dict
    query, source = active_work_case["queries"]["final"], active_work_case["case"]["source"]
    raw = native_active_contact_path_work_to_dict(query, expected_source_history_dto=source)
    assert query.finite_point_area_m2 is query.finite_point_work_j is None
    assert raw["finite_point_area_m2"] is raw["finite_point_work_j"] is None
    assert raw["transfer_work_density_j_m2"] == raw["transfer_energy_increment_density_j_m2"] == [0, 1]
    assert not raw["original_affine_source_work_included"]
    assert not raw["original_nodal_ledgers_summed_with_continuous_work"]
    assert not raw["receiving_pair_weighted_contact_integrals_numerically_enclosed"]
    assert not raw["candidate_mechanics_adopted"]
    assert raw["accepted_native_force_intervals"] == 0


def test_original_continuous_interface_measure_and_sixteen_literal_atoms_remain_distinct(active_work_case):
    history = active_work_case["case"]["history"]
    assert len(history.quadrature_histories) == 16
    radius = F.from_float(history.origin_surface.radius_km)*1000
    for trace in history.initial_mechanics.fault_traces.traces:
        nodes = tuple(node for node in history.quadrature_histories if node.trace_id == trace.trace_id)
        assert len(nodes) == 4
        sine = F.from_float(nodes[0].original_frame[0])
        extent = F.from_float(next(row.extent_radians for row in history.previous_memory.traces
            if row.trace_id == trace.trace_id))
        continuum_measure = radius*F.from_float(trace.contact_height_m)/sine*extent
        literal_atom_measure = sum((F.from_float(node.original_area_m2) for node in nodes), F(0))
        assert continuum_measure != literal_atom_measure
        for node in nodes:
            assert node.original_area_m2 == node.previous_state.area_m2 == node.unaccepted_candidate_state.area_m2
