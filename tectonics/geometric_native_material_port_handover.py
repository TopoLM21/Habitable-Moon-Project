"""Loaded candidate contacts attached to proved incoming source-beam labels.

This is an explicit receiver-attached configuration transport law, evaluated
at the refused candidate endpoint. It preserves the old interface quadrature
measure and all local constitutive state. It changes actual donor attachments
only where the pullback label lies inside its original material and beam.
Retreating ports retain their existing attachments; no new donor is invented.

The incoming radial displacement is the original donor beam's cubic value at
that material coordinate. Interior loading uses that cubic as a one-mode Ritz
ansatz. Neither the old force equilibrium nor native spatial admission follows
from this handover. Global material, clocks and accepted work remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction as F
import hashlib
import json
import math

from .geometric_native_pose import FractionInterval as I
from .geometric_native_physical_pair_time import (
    certify_native_relative_entry_coordinates, _rotated,
)
from .geometric_native_physical_history import (
    validate_native_physical_history_binding, native_physical_history_binding_to_dict,
    query_native_original_pair_history,
)
from .geometric_native_beam_material_chart import (
    certify_native_beam_material_chart, native_beam_material_chart_to_dict,
)
from .geometric_native_contact_reference_transport import (
    ContactHistoryTransportLaw,
    initialize_contact_reference_transport, transport_contact_reference,
    make_contact_bank_attachment, contact_reference_transport_to_dict,
    _proof_key as _typed_proof_key,
)


FORMAT = "native-candidate-material-port-handover-1"
MODEL = "receiver_attached_loaded_configuration_source_beam_Ritz_handover"
_CACHE = {}
_DERIVATIONS = {}


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _proof_key(value):
    # Keys are learned only from complete typed replay, never from a supplied
    # DTO seal. Compact keys keep live immutable results valid across builds.
    return _typed_proof_key(value)


def _literal_json(value):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _literal_json(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _literal_json(item)
        return
    raise ValueError("Material port DTO requires literal JSON containers and finite values")


def _f(value):
    return F.from_float(float(value))


def _vector(value):
    return tuple(_f(item) for item in value)


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), F(0))


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _fraction(value):
    return [value.numerator, value.denominator]


def _interval(value):
    return dict(lo=_fraction(value.lo), hi=_fraction(value.hi))


def _ivector(values):
    return [_interval(value) for value in values]


def _source_membership(fragment, coordinates):
    """Prove membership in every original unposed material halfspace."""
    if fragment.geometry.pose or any(p.rotations for p in fragment.geometry.supports):
        raise ValueError("Candidate handover requires the authenticated original unposed material chart")
    margins = tuple(_dot(plane.normal, coordinates) for plane in fragment.geometry.supports)
    if any(value.lo < 0 for value in margins):
        raise ValueError("Incoming material label lacks original fragment support")
    return margins


@dataclass(frozen=True, slots=True)
class NativeCandidateMaterialPort:
    point_id: str
    trace_id: str
    target_reference_coordinates: tuple
    target_material_margins: tuple
    target_chart: object
    source_geometry: object
    world_direction: tuple
    sub_position_m: tuple
    over_position_m: tuple
    normal: tuple
    down_direction: tuple
    normal_operator: tuple
    down_operator: tuple
    transport_state: object


@dataclass(frozen=True, slots=True)
class NativeCandidateMaterialPortHandover:
    history: object
    law: object
    ports: tuple[NativeCandidateMaterialPort, ...]
    retained_point_ids: tuple[str, ...]
    precision: int
    source_history_payload_json: bytes
    handover_sha256: str

    @property
    def physical_pair_admission_authorized(self):
        return False

    @property
    def candidate_mechanics_adopted(self):
        return False


def _operator(position, frame, reference_dot, shape):
    # The over angular row includes rotation of the contact frame. With this
    # carrier couple its lever is r_sub, not r_over. Radial DOFs are metres.
    lever = _cross(position, frame)
    return (tuple(-x for x in lever), lever, -shape*reference_dot, I(reference_dot))


def _operator_raw(row):
    return dict(sub_angular=_ivector(row[0]), over_angular=_ivector(row[1]),
                sub_radial=_interval(row[2]), over_radial=_interval(row[3]),
                angular_coordinate_units="radian", radial_coordinate_units="metre",
                receiver_frame_couple_included=True)


def _body(value):
    return dict(format=FORMAT, model=MODEL, precision=value.precision,
        source_history=json.loads(value.source_history_payload_json),
        transport_law=dict(model=value.law.model, provenance=value.law.provenance),
        time_fraction=_fraction(F(1)), source_epoch="unaccepted_candidate",
        ports=[dict(point_id=p.point_id, trace_id=p.trace_id,
            target_reference_coordinates=_ivector(p.target_reference_coordinates),
            target_material_margins=[_interval(x) for x in p.target_material_margins],
            target_chart=native_beam_material_chart_to_dict(p.target_chart),
            source_geometry=native_original_geometry_to_dict(p.source_geometry),
            world_direction=_ivector(p.world_direction), sub_position_m=_ivector(p.sub_position_m),
            over_position_m=_ivector(p.over_position_m), normal=_ivector(p.normal),
            down_direction=_ivector(p.down_direction),
            normal_operator=_operator_raw(p.normal_operator), down_operator=_operator_raw(p.down_operator),
            transport_state=contact_reference_transport_to_dict(p.transport_state)) for p in value.ports],
        retained_point_ids=list(value.retained_point_ids),
        unchanged_global_beam_state=True, original_interface_measure_preserved=True,
        outgoing_active_attachments_replaced_once=True,
        finite_measure_assigned_to_first_entry_point=False,
        transported_candidate_is_force_equilibrium=False,
        native_full_bank_and_clamp_support_certified=False,
        candidate_mechanics_adopted=False, physical_pair_admission_authorized=False,
        transfer_external_work_j=0, transfer_dissipation_j=0,
        accepted_loaded_native_intervals=0, accepted_source_work_j=0)


def native_original_geometry_to_dict(value):
    from .geometric_native_physical_pair_time import native_original_pair_tip_maps_to_dict
    return native_original_pair_tip_maps_to_dict(value)


def certify_native_candidate_material_port_handover(history, *, law,
        expected_source_history_dto, precision=128):
    """Replace supported incoming attachments and preserve the full source ledger.

    The new material label is the exact analytic expression Q_sub^-1 Q_over q
    at u=1. Its interval vector is an enclosure, never a selected replacement
    label. Full original fragment and source strip membership are proved.
    This source epoch has no cross-quarter handovers at its Gauss nodes;
    off-quadrature first-entry fields still require a spatial integration law.
    """
    history = validate_native_physical_history_binding(history)
    if type(law) is not ContactHistoryTransportLaw:
        raise ValueError("Material handover requires an explicit immutable transport law")
    law.__post_init__()
    if type(precision) is not int or not 32 <= precision <= 1024:
        raise ValueError("Material port handover requires a bounded literal precision")
    source_raw = native_physical_history_binding_to_dict(history)
    _literal_json(expected_source_history_dto)
    if type(expected_source_history_dto) is not dict or _encoded(source_raw) != _encoded(expected_source_history_dto):
        raise ValueError("Material port handover needs the complete independently supplied source history")
    derivation_key = _proof_key((history, law, precision))
    if derivation_key in _DERIVATIONS:
        return _DERIVATIONS[derivation_key]
    fragments = {f.fragment_id: f for f in history.material_state.journal.source_surface.fragments}
    bindings = {b.trace_id: b for b in history.material_state.trace_bindings}
    parameters = {p.trace_id: p for p in history.initial_mechanics.bending_parameters}
    radial = {row[0]: row[1:] for row in history.actual_candidate.transverse_displacement_m}
    axes = {owner: tuple(x*history.dt_myr for x in axis) for owner, axis in history.omega_records}
    radius = _f(history.origin_surface.radius_km)*1000
    ports, retained = [], []
    for node in history.quadrature_histories:
        binding, material, point = bindings[node.trace_id], node.original_material, node.original_point
        q, n, d = _vector(point.position), _vector(point.normal), _vector(point.down_dip)
        label = certify_native_relative_entry_coordinates(q, binding.overriding_plate,
            binding.subducting_plate, history.omega_records, history.dt_myr, I(F(1)),
            precision=precision, order=32)
        start, end = _vector(material.start), _vector(material.end)
        inward = tuple(-x for x in _cross(start, end))
        distance_sign = _dot(inward, label)
        if distance_sign.hi < 0:
            retained.append(node.point_id)
            continue
        if distance_sign.lo <= 0:
            raise ValueError("Candidate material port has unresolved incoming or retreating direction")
        margins = _source_membership(fragments[binding.donor_fragment_id], label)
        # Receiver remains its original literal material coordinate, measure
        # and frame. Its own original tip is not reassigned to the donor.
        _source_membership(fragments[binding.receiver_fragment_id], tuple(I(x) for x in q))
        w_sub, w_over = (_f(x) for x in radial[node.trace_id])
        chart = certify_native_beam_material_chart(node.trace_id, "sub", start, end,
            label, radius, _f(parameters[node.trace_id].sub.width_m), I(w_sub), precision=precision)
        world = _rotated(q, axes[binding.overriding_plate], I(F(1)), precision, 32)
        normal = _rotated(n, axes[binding.overriding_plate], I(F(1)), precision, 32)
        down = _rotated(d, axes[binding.overriding_plate], I(F(1)), precision, 32)
        sub_position = tuple((radius+chart.displacement_m)*x for x in world)
        over_position = tuple((radius+w_over)*x for x in world)
        relative_radial = I(w_over)-chart.displacement_m
        h_new = (relative_radial*_dot(q, n), relative_radial*_dot(q, d))
        source = query_native_original_pair_history(history, node.point_id, I(F(1)), precision=precision).tip_maps
        h_old = (source.gap_m, source.jump_m)
        context = dict(coordinate_model="exact_analytic_owner_pullback_expression",
            time_fraction=_fraction(F(1)), original_receiver_label=[_fraction(x) for x in q],
            donor_owner=binding.subducting_plate, receiver_owner=binding.overriding_plate,
            owner_motion=[[owner, [_fraction(x) for x in axis]] for owner, axis in history.omega_records],
            duration_myr=_fraction(history.dt_myr),
            target_label_expression="Q_sub(1)^-1 Q_over(1) original_receiver_label",
            target_label_evaluated_once_at_transfer=True,
            material_label_frozen_during_subsequent_force_variations=True,
            target_label_enclosure=_ivector(label), source_beam_chart=native_beam_material_chart_to_dict(chart),
            original_interface_point_id=node.point_id, original_interface_area_m2=node.original_area_m2,
            source_epoch="unaccepted_candidate", candidate_mechanics_adopted=False)
        state = initialize_contact_reference_transport(history, node.point_id, law=law,
            source_epoch="unaccepted_candidate", source_binding_id=node.point_id+":original",
            h_old=h_old, geometry_context=dict(context, attachment="original_pair"),
            expected_source_history_dto=expected_source_history_dto)
        target = make_contact_bank_attachment(
            sub_fragment_id=binding.donor_fragment_id, sub_material_id=binding.donor_material_id,
            sub_owner=binding.subducting_plate, over_fragment_id=binding.receiver_fragment_id,
            over_material_id=binding.receiver_material_id, over_owner=binding.overriding_plate,
            correspondence=context)
        state = transport_contact_reference(state, target_binding_id=node.point_id+":incoming:1",
            target_banks=target, h_new=h_new, geometry_context=context,
            target_columns=material.columns, target_contact_law=material.law,
            target_water=material.water, target_area_m2=node.original_area_m2,
            expected_source_history_dto=expected_source_history_dto)
        ports.append(NativeCandidateMaterialPort(node.point_id, node.trace_id, label, margins, chart,
            source, world, sub_position, over_position, normal, down,
            _operator(sub_position, normal, _dot(q, n), chart.shape),
            _operator(sub_position, down, _dot(q, d), chart.shape), state))
    identities = tuple(p.point_id for p in ports)+tuple(retained)
    if len(set(identities)) != len(identities) or set(identities) != {n.point_id for n in history.quadrature_histories}:
        raise ValueError("Handover must cover every original interface history exactly once")
    value = NativeCandidateMaterialPortHandover(history, law, tuple(ports), tuple(retained),
        precision, _encoded(source_raw), "")
    value = replace(value, handover_sha256=hashlib.sha256(_encoded(_body(value))).hexdigest())
    _CACHE[_proof_key(value)] = value.handover_sha256
    if len(_DERIVATIONS) >= 16:
        _DERIVATIONS.clear()
    _DERIVATIONS[derivation_key] = value
    return value


def native_candidate_material_port_handover_to_dict(value):
    if type(value) is not NativeCandidateMaterialPortHandover:
        raise ValueError("Expected an immutable candidate material port handover")
    if _CACHE.get(_proof_key(value)) != value.handover_sha256:
        raise ValueError("Candidate material port handover is not authenticated by full source replay")
    return json.loads(_encoded(dict(_body(value), handover_sha256=value.handover_sha256)))


def native_candidate_material_port_handover_from_dict(raw, *, expected_history,
        expected_source_history_dto, expected_law):
    if type(raw) is not dict:
        raise ValueError("Candidate handover requires a complete literal DTO")
    _literal_json(raw)
    repeated = certify_native_candidate_material_port_handover(expected_history, law=expected_law,
        expected_source_history_dto=expected_source_history_dto, precision=raw.get("precision"))
    if _encoded(raw) != _encoded(native_candidate_material_port_handover_to_dict(repeated)):
        raise ValueError("Candidate handover differs from full physical and material replay")
    return repeated
