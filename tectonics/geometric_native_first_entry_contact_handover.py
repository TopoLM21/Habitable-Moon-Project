"""Commit a continuous contact attachment at its proved implicit first entry.

The material label is an original bulk label, never an old Gauss spring tip.
Its own source beam supplies the incoming displacement; the first receiver
supplies the complete original contact field, frame and interface measure.
The transport freezes the original field expression at the implicit entry
clock and changes its geometric reference.  Consequently a later geometric
variation cannot accidentally replay the original source clock a second time.
This is a material correspondence, not a force acceptance or a whole-domain
integration rule.  Literal nodal work and material-volume budgets stay in the
complete source history and are never assigned to this zero-area observation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction as F
import hashlib
import json

from .geometric_native_pose import FractionInterval as I, _rounded_interval
from .geometric_native_first_entry_physical_binding import (
    NativeFirstEntryPhysicalBinding, validate_native_first_entry_physical_binding,
    native_first_entry_physical_binding_to_dict,
)
from .geometric_native_physical_history import (
    _dyadic, _literal_json, native_physical_history_binding_to_dict,
)
from .geometric_native_contact_time_support import (
    ADMISSIBLE, native_quarter_material_coordinate_bounds,
)
from .geometric_native_contact_field_history import (
    query_native_contact_field_history, native_contact_field_history_to_dict,
)
from .geometric_native_first_entry_contact_measure import (
    certify_native_first_entry_contact_measure, native_first_entry_contact_measure_to_dict,
)
from .geometric_native_beam_material_chart import (
    certify_native_beam_material_chart, native_beam_material_chart_to_dict,
)
from .geometric_native_contact_reference_transport import (
    ContactHistoryTransportLaw, make_contact_bank_attachment,
    initialize_contact_reference_transport_symbols, transport_contact_reference_symbols,
    contact_reference_transport_symbols_to_dict, _proof_key,
)
from .geometric_native_physical_pair_time import _rotated


FORMAT = "native-first-entry-continuous-contact-handover-1"
MODEL = "receiver_attached_original_field_frozen_at_implicit_first_entry"
COMMITTED = "first_entry_field_attachment_committed"
_AUTHENTICATED = {}
_DERIVATIONS = {}


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _fraction(value):
    return [value.numerator, value.denominator]


def _ib(value):
    return dict(lo=_fraction(value.lo), hi=_fraction(value.hi))


def _iv(values):
    return [_ib(value) for value in values]


def _vector(values):
    return tuple(_dyadic(value) for value in values)


def _compress(value, precision):
    return _rounded_interval(value, precision+16)


def _source(binding, law, expected):
    binding = validate_native_first_entry_physical_binding(binding)
    if type(law) is not ContactHistoryTransportLaw:
        raise ValueError("First-entry handover requires the explicit complete-configuration law")
    law.__post_init__()
    if type(expected) is not dict:
        raise ValueError("First-entry handover needs the entire independent physical source DTO")
    _literal_json(expected)
    payload = _encoded(expected)
    if payload != _encoded(native_physical_history_binding_to_dict(binding.history)):
        raise ValueError("First-entry handover source differs from the complete original replay")
    return payload


def _tip(history, trace_id, bank, time):
    index = 1 if bank == "sub" else 2
    old = next(row for row in history.accepted_before.transverse_displacement_m
               if row[0] == trace_id)
    new = next(row for row in history.actual_candidate.transverse_displacement_m
               if row[0] == trace_id)
    a, b = _dyadic(old[index]), _dyadic(new[index])
    return I(a)+(b-a)*time


@dataclass(frozen=True, slots=True)
class NativeFirstEntryContactAttachment:
    original_donor_trace_id: str
    original_receiver_trace_id: str
    desired_fragment_pair: tuple
    original_donor_trace_pair: tuple
    original_receiver_trace_pair: tuple
    source_field_query: object
    sub_material_chart: object
    interface_measure: object
    transport_family: object
    expression_context_payload_json: bytes
    sub_tip_displacement_bounds_m: I
    over_tip_displacement_bounds_m: I
    receiver_world_direction: tuple
    receiver_world_normal: tuple
    receiver_world_down_dip: tuple
    sub_position_bounds_m: tuple
    over_position_bounds_m: tuple
    reference_gap_bounds: I
    reference_jump_bounds: I

    @property
    def expression_context(self):
        return json.loads(self.expression_context_payload_json)

    @property
    def source_free_gap_bounds(self):
        return self.source_field_query.free_gap_bounds_m

    @property
    def source_free_jump_bounds(self):
        return self.source_field_query.free_jump_bounds_m

    @property
    def transported_free_gap_bounds(self):
        return self.source_free_gap_bounds

    @property
    def transported_free_jump_bounds(self):
        return self.source_free_jump_bounds

    @property
    def attachment_committed(self):
        return self.transport_family.revision == 1

    @property
    def additional_handover_work_j(self):
        return F(0)

    @property
    def finite_point_area_m2(self):
        return None

    @property
    def finite_point_work_j(self):
        return None


@dataclass(frozen=True, slots=True)
class NativeFirstEntryContactHandover:
    binding: NativeFirstEntryPhysicalBinding
    law: ContactHistoryTransportLaw
    attachments: tuple[NativeFirstEntryContactAttachment, ...]
    status: str
    precision: int
    order: int
    source_history_payload_json: bytes
    handover_sha256: str

    @property
    def additional_handover_work_j(self):
        return F(0)

    @property
    def additional_handover_dissipation_j(self):
        return F(0)

    @property
    def physical_pair_admission_authorized(self):
        return False

    @property
    def candidate_mechanics_adopted(self):
        return False


def _attachment(binding, row, law, expected, precision, order):
    history, time = binding.history, binding.time_fraction_bounds
    traces = {trace.trace_id: trace for trace in history.initial_mechanics.fault_traces.traces}
    banks = {item.trace_id: item for item in history.material_state.trace_bindings}
    donor, receiver = traces[row.donor_trace_id], traces[row.receiver_trace_id]
    donor_bank, receiver_bank = banks[row.donor_trace_id], banks[row.receiver_trace_id]
    if (donor_bank.donor_fragment_id != binding.chronology.original_parent_id
            or (donor_bank.donor_fragment_id, receiver_bank.receiver_fragment_id)
            != row.desired_fragment_pair):
        raise ValueError("First-entry target banks differ from the exact first receiving ancestry")
    source_nodes = tuple(node for node in history.quadrature_histories
                         if node.trace_id == receiver.trace_id)
    if not source_nodes or any(node.original_frame != source_nodes[0].original_frame
            or node.original_point.horizontal != source_nodes[0].original_point.horizontal
            for node in source_nodes):
        raise ValueError("First receiver has inconsistent saved original material frames")
    # The old literal frame coefficients are retained, including their binary64
    # norm defect.  Neither dip nor original Gauss labels are renormalized.
    sine, cosine = map(_dyadic, source_nodes[0].original_frame)
    horizontal = _vector(source_nodes[0].original_point.horizontal)
    radius = _dyadic(history.origin_surface.radius_km)*1000
    extent = next(memory.extent_radians for memory in history.previous_memory.traces
                  if memory.trace_id == receiver.trace_id)
    density = radius*_dyadic(receiver.contact_height_m)/sine
    native_angle = row.receiver_native_angle_bounds
    computational_angle = native_quarter_material_coordinate_bounds(
        history.contact_time_path, receiver.trace_id, native_angle, precision=precision)
    measure = certify_native_first_entry_contact_measure(receiver.trace_id,
        _vector(receiver.start), _vector(receiver.end), donor_bank.subducting_plate,
        receiver_bank.overriding_plate, history.omega_records, history.dt_myr,
        native_angle, time, _dyadic(extent), density, precision=precision, order=order)
    parameters = next(parameter for parameter in history.initial_mechanics.bending_parameters
                      if parameter.trace_id == donor.trace_id)
    sub_tip = _tip(history, donor.trace_id, "sub", time)
    over_tip = _tip(history, receiver.trace_id, "over", time)
    # This homogeneous chart uses the unchanged original bulk label, not a
    # newly selected vector from its first-root coordinate enclosure.
    chart = certify_native_beam_material_chart(donor.trace_id, "sub",
        _vector(donor.start), _vector(donor.end), binding.chronology.original_point,
        radius, _dyadic(parameters.sub.width_m), sub_tip, precision=precision)
    context = dict(model=MODEL,
        point_chronology_sha256=binding.chronology.point_chronology_sha256,
        first_physical_binding_sha256=binding.binding_sha256,
        original_parent_id=binding.chronology.original_parent_id,
        original_bulk_label=[_fraction(x) for x in binding.chronology.original_point],
        first_entry_time_fraction_bounds=_ib(time),
        receiver_native_angle_bounds=_ib(native_angle),
        original_computational_angle_bounds=_ib(computational_angle),
        source_trace_id=receiver.trace_id, incoming_own_beam_trace_id=donor.trace_id,
        target_fragment_pair=list(row.desired_fragment_pair),
        source_expression="original_receiver_field(theta(first_root), first_root)",
        reference_expression="incoming_own_beam_and_first_receiver_at_same_implicit_root",
        source_expression_frozen_for_later_geometry_variations=True,
        incoming_bulk_direction_normalization="exact_unit_carrier_identity_at_proved_first_root",
        original_gauss_labels_unchanged=True,
        same_point_and_time_on_both_sides=True,
        material_label_frozen_during_subsequent_force_variations=True,
        first_root_midpoint_used=False)
    field = query_native_contact_field_history(history, receiver.trace_id,
        computational_angle, time, expected_source_history_dto=expected,
        source_expression_context=context, precision=precision)
    if field.constitutive_status != ADMISSIBLE:
        raise ValueError("First-entry transport needs the proved continuous inviscid source branch")
    axis = next(tuple(item*history.dt_myr for item in omega)
                for owner, omega in history.omega_records
                if owner == receiver_bank.overriding_plate)
    world = measure.receiver_world_direction
    world_horizontal = _rotated(horizontal, axis, time, precision, order)
    # Exact first-carrier incidence implies c.h=0 and c.c=1.  The interval
    # box merely encloses that event; it is not used to re-prove incidence.
    normal = tuple(_compress(sine*h+cosine*c, precision)
                   for h, c in zip(world_horizontal, world))
    down = tuple(_compress(cosine*h-sine*c, precision)
                 for h, c in zip(world_horizontal, world))
    relative = over_tip-chart.displacement_m
    h_new = (_compress(cosine*relative, precision), _compress(-sine*relative, precision))
    positions = tuple(tuple(_compress((radius+w)*c, precision) for c in world)
                      for w in (chart.displacement_m, over_tip))
    prefix = "first-entry:"+binding.chronology.point_chronology_sha256+":"+receiver.trace_id
    family = initialize_contact_reference_transport_symbols(history, receiver.trace_id,
        law=law, source_binding_id=prefix+":original", geometry_context=dict(context,
            attachment="original_receiver_contact_field"), expected_source_history_dto=expected)
    target = make_contact_bank_attachment(donor_bank.donor_fragment_id,
        donor_bank.donor_material_id, donor_bank.subducting_plate,
        receiver_bank.receiver_fragment_id, receiver_bank.receiver_material_id,
        receiver_bank.overriding_plate, correspondence=context)
    family = transport_contact_reference_symbols(family, target_binding_id=prefix+":incoming:1",
        target_banks=target, geometry_context=context, target_columns=donor.columns,
        target_contact_law=donor.law, target_water=donor.water,
        expected_source_history_dto=expected)
    return NativeFirstEntryContactAttachment(donor.trace_id, receiver.trace_id,
        row.desired_fragment_pair, row.original_donor_trace_pair, row.original_receiver_trace_pair,
        field, chart, measure, family, _encoded(context), sub_tip, over_tip,
        world, normal, down, *positions, *h_new)


def certify_native_first_entry_contact_handover(binding, *, law,
        expected_source_history_dto, precision=256, order=32):
    """Change proved first-entry attachments without assigning a node reservoir.

    Negative contexts are explicit no-ops.  Multiple possible receivers or an
    unproved carrier remain unresolved; there is no sorted-ID tie breaker.
    """
    payload = _source(binding, law, expected_source_history_dto)
    if (type(precision) is not int or not 32 <= precision <= 1024
            or type(order) is not int or not 8 <= order <= 64):
        raise ValueError("First-entry transport requires bounded literal proof budgets")
    key = _proof_key((binding, law, precision, order))
    if key in _DERIVATIONS:
        return _DERIVATIONS[key]
    attachments = ()
    if binding.time_fraction_bounds is None:
        status = "no_first_entry"
    elif (binding.first_receiver_ancestry_status != "proved" or len(binding.rows) != 1
          or binding.rows[0].receiver_coordinate_status
          != "first_owner_root_receiver_carrier_coordinate_proved"):
        status = "unresolved_first_receiver_or_carrier"
    else:
        attachments = (_attachment(binding, binding.rows[0], law,
                                   expected_source_history_dto, precision, order),)
        status = COMMITTED
    value = NativeFirstEntryContactHandover(binding, law, attachments, status,
                                          precision, order, payload, "")
    body = _encoded(_body(value))
    value = replace(value, handover_sha256=hashlib.sha256(body).hexdigest())
    _AUTHENTICATED[_proof_key(value)] = body
    _DERIVATIONS[key] = value
    return value


def evaluate_native_first_entry_contact_free_anchors(value, attachment_index,
        h_current, *, geometry_context):
    """Evaluate a geometric variation of the frozen first-event reference.

    The return value is a kinematic anchor, not a constitutive time step.  A
    future force solver still has to evolve the preserved plastic state and
    assemble admissible spatial supports and work in one accepted interval.
    """
    validate_native_first_entry_contact_handover(value)
    if type(attachment_index) is not int or not 0 <= attachment_index < len(value.attachments):
        raise ValueError("A geometric variation requires an existing first-entry attachment")
    if (type(h_current) is not tuple or len(h_current) != 2
            or any(type(x) is not I or x.lo > x.hi for x in h_current)):
        raise ValueError("Current projected geometry needs two exact interval expressions")
    if type(geometry_context) is not dict or not geometry_context:
        raise ValueError("Current projected geometry requires its complete expression context")
    _literal_json(geometry_context)
    attachment = value.attachments[attachment_index]
    source = attachment.source_free_gap_bounds, attachment.source_free_jump_bounds
    reference = attachment.reference_gap_bounds, attachment.reference_jump_bounds
    if _encoded(geometry_context) == attachment.expression_context_payload_json:
        if h_current != reference:
            raise ValueError("The same first-event expression cannot have changed geometry bounds")
        return source
    return tuple(_compress(free+current-old, value.precision)
                 for free, current, old in zip(source, h_current, reference))


def _attachment_body(value, history, expected):
    return dict(original_donor_trace_id=value.original_donor_trace_id,
        original_receiver_trace_id=value.original_receiver_trace_id,
        desired_fragment_pair=list(value.desired_fragment_pair),
        original_donor_trace_pair=list(value.original_donor_trace_pair),
        original_receiver_trace_pair=list(value.original_receiver_trace_pair),
        source_field_query=native_contact_field_history_to_dict(value.source_field_query,
            history=history, expected_source_history_dto=expected),
        sub_material_chart=native_beam_material_chart_to_dict(value.sub_material_chart),
        interface_measure=native_first_entry_contact_measure_to_dict(value.interface_measure),
        transport_family=contact_reference_transport_symbols_to_dict(value.transport_family),
        expression_context=value.expression_context,
        sub_tip_displacement_bounds_m=_ib(value.sub_tip_displacement_bounds_m),
        over_tip_displacement_bounds_m=_ib(value.over_tip_displacement_bounds_m),
        receiver_world_direction=_iv(value.receiver_world_direction),
        receiver_world_normal=_iv(value.receiver_world_normal),
        receiver_world_down_dip=_iv(value.receiver_world_down_dip),
        sub_position_bounds_m=_iv(value.sub_position_bounds_m),
        over_position_bounds_m=_iv(value.over_position_bounds_m),
        reference_gap_bounds=_ib(value.reference_gap_bounds),
        reference_jump_bounds=_ib(value.reference_jump_bounds),
        transported_free_gap_bounds=_ib(value.transported_free_gap_bounds),
        transported_free_jump_bounds=_ib(value.transported_free_jump_bounds),
        attachment_committed=value.attachment_committed,
        additional_handover_work_j=[0, 1], additional_handover_dissipation_j=[0, 1],
        finite_point_area_m2=None, finite_point_work_j=None,
        plastic_and_energy_density_preserved_by_source_expression_identity=True)


def _body(value):
    expected = json.loads(value.source_history_payload_json)
    return dict(format=FORMAT, model=MODEL, status=value.status,
        precision=value.precision, order=value.order,
        explicit_law=dict(model=value.law.model, provenance=value.law.provenance),
        original_physical_binding=native_first_entry_physical_binding_to_dict(value.binding),
        complete_original_source_history=expected,
        attachments=[_attachment_body(item, value.binding.history, expected)
                     for item in value.attachments],
        first_root_midpoint_used=False, additional_handover_work_j=[0, 1],
        additional_handover_dissipation_j=[0, 1],
        original_nodal_areas_states_work_and_beam_dofs_unchanged=True,
        source_field_frozen_at_implicit_first_entry=True,
        whole_first_entry_domain_integration_adopted=False,
        whole_spatial_bank_and_clamp_support_certified=False,
        physical_pair_admission_authorized=False, candidate_mechanics_adopted=False,
        accepted_loaded_native_intervals=0, accepted_source_work_j=[0, 1])


def validate_native_first_entry_contact_handover(value):
    if type(value) is not NativeFirstEntryContactHandover:
        raise ValueError("Expected an immutable first-entry continuous contact handover")
    if _proof_key(value) not in _AUTHENTICATED:
        raise ValueError("First-entry attachment was not built from its full authenticated sources")
    return value


def native_first_entry_contact_handover_to_dict(value):
    validate_native_first_entry_contact_handover(value)
    body = json.loads(_AUTHENTICATED[_proof_key(value)])
    body["handover_sha256"] = value.handover_sha256
    return body


def native_first_entry_contact_handover_from_dict(raw, *, expected_binding,
        expected_source_history_dto, expected_law):
    if type(raw) is not dict:
        raise ValueError("First-entry handover requires an object")
    _literal_json(raw)
    rebuilt = certify_native_first_entry_contact_handover(expected_binding,
        law=expected_law, expected_source_history_dto=expected_source_history_dto,
        precision=raw.get("precision"), order=raw.get("order"))
    if _encoded(raw) != _encoded(native_first_entry_contact_handover_to_dict(rebuilt)):
        raise ValueError("Saved first-entry handover differs from complete independent source replay")
    return rebuilt
