"""Once-only local work along an authenticated frozen material-binding path.

Every segment, including the initial attachment at the exact original clock,
uses the full SO(3) geometry of its frozen material label.  A seam event takes
the evolved preceding segment state as its initial state.  The original
affine diagnostic contact-work primitive is deliberately a separate budget.
This is a local prescribed-motion calculation per original interface area;
neither a finite point area nor an accepted global force interval is assigned.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib

from .geometric_native_contact_field_history import _outward
from .geometric_native_physical_history import _encoded, _literal_json
from .geometric_native_pose import FractionInterval as I


FORMAT = "native-active-contact-path-work-query-1"
MODEL = "sequential_full_rotation_frozen_material_segments_with_once_only_state_transfer"
_BUDGET_FIELDS = (
    "source_work_density_bounds_j_m2",
    "friction_dissipation_density_bounds_j_m2",
    "bulk_energy_increment_density_bounds_j_m2",
    "contact_energy_increment_density_bounds_j_m2",
    "total_energy_increment_density_bounds_j_m2",
    "compliance_balance_residual_density_bounds_j_m2",
)


@dataclass(frozen=True, slots=True)
class NativeActiveContactPathWorkQuery:
    binding: object
    time_fraction_bounds: I
    segments: tuple
    consumed_binding_ids: tuple[str, ...]
    active_binding_id: str
    source_work_density_bounds_j_m2: I
    friction_dissipation_density_bounds_j_m2: I
    bulk_energy_increment_density_bounds_j_m2: I
    contact_energy_increment_density_bounds_j_m2: I
    total_energy_increment_density_bounds_j_m2: I
    compliance_balance_residual_density_bounds_j_m2: I

    @property
    def current_state(self):
        return self.segments[-1]

    @property
    def revision(self):
        return len(self.consumed_binding_ids)

    @property
    def candidate_mechanics_adopted(self):
        return False

    @property
    def accepted_native_force_intervals(self):
        return 0

    @property
    def finite_point_area_m2(self):
        return None

    @property
    def finite_point_work_j(self):
        return None


def _clock(value):
    if (type(value) is not I or type(value.lo) is not Fraction
            or type(value.hi) is not Fraction or not 0 <= value.lo <= value.hi <= 1):
        raise ValueError("The active path query retains an exact original Fraction clock interval")
    return value


def _expression(binding, sequence, event=None):
    return dict(format="native-active-contact-path-segment-expression-1",
        binding_path_sha256=binding.path_sha256, segment_sequence=sequence,
        event_sequence=None if event is None else event.sequence,
        clock_identity="original_exact_zero" if event is None
            else "certified_original_donor_quarter_seam_root",
        state_source="original_field_at_exact_zero" if event is None
            else "evolved_previous_frozen_material_segment",
        root_midpoint_selected=False)


def query_native_active_contact_path_work(binding, time_fraction_bounds, *,
        expected_source_history_dto):
    """Follow the unique certified path through all switches before this clock.

An interval crossing a switching root is ambiguous and is refused.  Equality
with the complete certified root window explicitly denotes that same event
expression and observes the new attachment without an energy or memory jump.
"""
    from .geometric_native_active_contact_binding import validate_native_active_contact_binding
    from .geometric_native_active_contact_work import (
        make_native_active_contact_segment_reference, query_native_active_contact_segment_work,
    )

    validate_native_active_contact_binding(binding)
    current = _clock(time_fraction_bounds)
    # Passing the external full source into each factory authenticates source
    # contents, rather than accepting the path's own declared hash as authority.
    reference = make_native_active_contact_segment_reference(binding.history,
        binding.receiver_trace_id, binding.segments[0].donor_trace_id,
        binding.receiver_reference_coordinates, I(Fraction(0)),
        switch_chronology=None, switch_event_sequence=None,
        root_proof_context=_expression(binding, 0), initial_state=None,
        expected_source_history_dto=expected_source_history_dto,
        precision=binding.precision, order=binding.order)
    completed, consumed = [], []
    active = binding.segments[0]
    at_current_event = current == I(Fraction(0))
    for index, event in enumerate(binding.events):
        clock = event.time_fraction_bounds
        same_event = current == clock
        if current.hi < clock.lo:
            break
        if not same_event and current.lo <= clock.hi:
            raise ValueError("Query clock intersects a switching root without its expression identity")
        evolved = query_native_active_contact_segment_work(reference, clock,
            expected_source_history_dto=expected_source_history_dto)
        completed.append(evolved)
        if active.active_binding_id in consumed:
            raise ValueError("An active binding was consumed more than once")
        consumed.append(active.active_binding_id)
        reference = make_native_active_contact_segment_reference(binding.history,
            binding.receiver_trace_id, event.new_donor_trace_id,
            binding.receiver_reference_coordinates, clock,
            switch_chronology=event.incoming_chronology,
            switch_event_sequence=event.incoming_occupancy_event_sequence,
            root_proof_context=_expression(binding, index + 1, event), initial_state=evolved,
            expected_source_history_dto=expected_source_history_dto,
            precision=binding.precision, order=binding.order)
        active = binding.segments[index + 1]
        at_current_event = same_event
        if same_event:
            break
    final = query_native_active_contact_segment_work(reference, current,
        expected_source_history_dto=expected_source_history_dto,
        at_reference_event=at_current_event)
    segments = tuple(completed) + (final,)
    values = {name: _outward(sum((getattr(segment, name) for segment in segments),
        I(Fraction(0))), binding.precision) for name in _BUDGET_FIELDS}
    return NativeActiveContactPathWorkQuery(binding, current, segments,
        tuple(consumed), active.active_binding_id, **values)


def validate_native_active_contact_path_work(query, *, expected_source_history_dto):
    if type(query) is not NativeActiveContactPathWorkQuery:
        raise ValueError("A local active-path query retains its typed segment chain")
    actual = query_native_active_contact_path_work(query.binding, query.time_fraction_bounds,
        expected_source_history_dto=expected_source_history_dto)
    if query != actual:
        raise ValueError("Active-path state or once-only work differs from the authenticated replay")


def _interval(value):
    return dict(lo=[value.lo.numerator, value.lo.denominator],
        hi=[value.hi.numerator, value.hi.denominator])


def native_active_contact_path_work_to_dict(query, *, expected_source_history_dto):
    from .geometric_native_active_contact_binding import native_active_contact_binding_to_dict
    from .geometric_native_active_contact_work import native_active_contact_segment_work_to_dict

    validate_native_active_contact_path_work(query,
        expected_source_history_dto=expected_source_history_dto)
    value = dict(format=FORMAT, model=MODEL,
        binding=native_active_contact_binding_to_dict(query.binding),
        time_fraction_bounds=_interval(query.time_fraction_bounds),
        segments=[native_active_contact_segment_work_to_dict(segment) for segment in query.segments],
        consumed_binding_ids=list(query.consumed_binding_ids),
        active_binding_id=query.active_binding_id, revision=query.revision,
        budgets={name: _interval(getattr(query, name)) for name in _BUDGET_FIELDS},
        transfer_work_density_j_m2=[0, 1], transfer_dissipation_density_j_m2=[0, 1],
        transfer_energy_increment_density_j_m2=[0, 1],
        original_affine_source_work_included=False,
        initial_attachment_evolved_from_original_exact_zero=True,
        measure="joules_per_square_metre_of_original_interface_area",
        original_nodal_ledgers_summed_with_continuous_work=False,
        receiving_pair_weighted_contact_integrals_numerically_enclosed=False,
        candidate_mechanics_adopted=False, accepted_native_force_intervals=0,
        finite_point_area_m2=None, finite_point_work_j=None)
    value["query_sha256"] = hashlib.sha256(_encoded(value)).hexdigest()
    return value


def native_active_contact_path_work_from_dict(raw, *, expected_binding,
        expected_time_fraction_bounds, expected_source_history_dto):
    if type(raw) is not dict:
        raise ValueError("An active-path restart requires its complete JSON object")
    _literal_json(raw)
    query = query_native_active_contact_path_work(expected_binding, expected_time_fraction_bounds,
        expected_source_history_dto=expected_source_history_dto)
    if _encoded(raw) != _encoded(native_active_contact_path_work_to_dict(query,
            expected_source_history_dto=expected_source_history_dto)):
        raise ValueError("Active-path restart differs from the external source, schedule or segment replay")
    return query
