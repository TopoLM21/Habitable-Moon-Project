"""Explicit receiver-attached transport of a complete contact configuration.

This is a new constitutive reference-transport law, not an inference from a
material owner or a geometric overlap.  A declared material correspondence
changes the active attachment while the complete source spring configuration
and its single work lineage survive.  The receiver axes, parameters and
original interface measure must remain unchanged.  New free-anchor references
are expressions ``a_new = old_free - h_new``; no interval midpoint is taken.
The geometric contexts are declared inputs, not geometry certificates.  An
independent material-port binding must prove their coordinates and receiver
axes before this component can participate in physical pair admission.

The nodal operator commits only this standalone component attachment.  Its
source candidate remains unaccepted by the native force solver.  The separate
symbol family describes the same strain identity on an implicit material
coordinate and clock without assigning a quadrature area or nodal work there.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from fractions import Fraction
import hashlib
import json

from .geometric_contact_mechanics import (
    LocalContactState, state_to_dict, validate_contact_state,
)
from .geometric_native_contact_time_support import (
    _add, _affine_bounds, _field, _scale,
)
from .geometric_native_physical_history import (
    NativePhysicalHistoryBinding, _encoded, _literal_json, _typed_fingerprint,
    native_physical_history_binding_to_dict, validate_native_physical_history_binding,
)
from .geometric_native_pose import FractionInterval as I


TRANSPORT_LAW_MODEL = "receiver_attached_complete_lumped_contact_configuration"
FORMAT = "native-contact-reference-transport-1"
FAMILY_FORMAT = "native-contact-reference-transport-symbol-family-1"
OFFSET_EXPRESSION = "a_new = original_free_anchor - h_new"
ANCHOR_EXPRESSION = "free_current = original_free_anchor + (h_current - h_new)"
_AUTHENTICATED = {}


def _identity(value, name):
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} requires an explicit nonempty identity")
    return value


def _context(value, name):
    if type(value) is not dict or not value:
        raise ValueError(f"{name} requires a complete declared JSON context")
    _literal_json(value)
    return _encoded(value)


def _geometry(value):
    if (type(value) is not tuple or len(value) != 2
            or any(type(item) not in (Fraction, I) for item in value)):
        raise ValueError("Geometry must be a literal exact Fraction/interval pair")
    if any(type(item) is I and (type(item.lo) is not Fraction
            or type(item.hi) is not Fraction or item.lo > item.hi) for item in value):
        raise ValueError("Geometry intervals must retain ordered literal Fraction endpoints")
    return value


def _interval(value):
    return value if type(value) is I else I(value)


def _fraction(value):
    # Actual source scalars are retained binary64 values, not rounded decimal
    # or re-evaluated continuous fields.
    if type(value) is not float:
        raise ValueError("Transport requires the original literal binary64 source scalar")
    return Fraction.from_float(value)


def _hash(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _portable(value):
    return json.loads(_encoded(value))


def _rational(value):
    if (type(value) is not list or len(value) != 2
            or any(type(item) is not int for item in value) or value[1] <= 0):
        raise ValueError("Transport rational scalars require literal reduced integer pairs")
    result = Fraction(*value)
    if [result.numerator, result.denominator] != value:
        raise ValueError("Transport rational scalars must be canonical")
    return result


def _geometry_to_dict(value):
    _geometry(value)
    return [dict(kind="fraction", value=list(item.as_integer_ratio()))
            if type(item) is Fraction else dict(kind="interval", lo=list(item.lo.as_integer_ratio()),
                                              hi=list(item.hi.as_integer_ratio()))
            for item in value]


def _geometry_from_dict(value):
    if type(value) is not list or len(value) != 2:
        raise ValueError("Transport geometry requires exactly two tagged scalar entries")
    result = []
    for item in value:
        if type(item) is not dict:
            raise ValueError("Transport geometry scalar must have its literal type tag")
        if item.get("kind") == "fraction" and set(item) == {"kind", "value"}:
            result.append(_rational(item["value"]))
        elif item.get("kind") == "interval" and set(item) == {"kind", "lo", "hi"}:
            result.append(I(_rational(item["lo"]), _rational(item["hi"])))
        else:
            raise ValueError("Transport geometry has unknown fields or a changed scalar type")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ContactHistoryTransportLaw:
    model: str
    provenance: str

    def __post_init__(self):
        if type(self.model) is not str or self.model != TRANSPORT_LAW_MODEL:
            raise ValueError("An explicit supported complete-configuration transport law is required")
        _identity(self.provenance, "Transport law provenance")


@dataclass(frozen=True, slots=True)
class ContactBankAttachment:
    sub_fragment_id: str
    sub_material_id: str
    sub_owner: int
    over_fragment_id: str
    over_material_id: str
    over_owner: int
    correspondence_payload_json: bytes

    def __post_init__(self):
        for name in ("sub_fragment_id", "sub_material_id", "over_fragment_id", "over_material_id"):
            _identity(getattr(self, name), name)
        for name in ("sub_owner", "over_owner"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError("Bank owners require literal nonnegative integer identities")
        if type(self.correspondence_payload_json) is not bytes:
            raise ValueError("Bank correspondence requires immutable canonical JSON bytes")
        try:
            raw = json.loads(self.correspondence_payload_json)
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError("Bank correspondence is not canonical JSON") from error
        if _context(raw, "Bank correspondence") != self.correspondence_payload_json:
            raise ValueError("Bank correspondence changed its canonical complete payload")


def make_contact_bank_attachment(sub_fragment_id, sub_material_id, sub_owner,
        over_fragment_id, over_material_id, over_owner, *, correspondence):
    return ContactBankAttachment(sub_fragment_id, sub_material_id, sub_owner,
        over_fragment_id, over_material_id, over_owner,
        _context(correspondence, "Declared material-bank correspondence"))


def _banks_to_dict(banks):
    return dict(sub_fragment_id=banks.sub_fragment_id, sub_material_id=banks.sub_material_id,
        sub_owner=banks.sub_owner, over_fragment_id=banks.over_fragment_id,
        over_material_id=banks.over_material_id, over_owner=banks.over_owner,
        correspondence=json.loads(banks.correspondence_payload_json))


def _banks_from_dict(raw):
    names = {"sub_fragment_id", "sub_material_id", "sub_owner", "over_fragment_id",
             "over_material_id", "over_owner", "correspondence"}
    if type(raw) is not dict or set(raw) != names:
        raise ValueError("Transport bank attachment changed its complete schema")
    return make_contact_bank_attachment(**raw)


def _source(history, expected_source_history_dto):
    validate_native_physical_history_binding(history)
    if type(expected_source_history_dto) is not dict:
        raise ValueError("The independent complete original physical source DTO is required")
    _literal_json(expected_source_history_dto)
    actual = native_physical_history_binding_to_dict(history)
    if _encoded(actual) != _encoded(expected_source_history_dto):
        raise ValueError("Transport source differs from the caller's complete authenticated history")
    return actual


def _trace_binding(history, trace_id):
    binding = next((item for item in history.material_state.trace_bindings
                    if item.trace_id == trace_id), None)
    if binding is None:
        raise ValueError("Transport requires an authenticated original physical trace")
    return binding


def _original_banks(history, trace_id, point_id):
    source = _trace_binding(history, trace_id)
    return make_contact_bank_attachment(source.donor_fragment_id, source.donor_material_id,
        source.subducting_plate, source.receiver_fragment_id, source.receiver_material_id,
        source.overriding_plate, correspondence=dict(model="original_authenticated_pair",
        trace_id=trace_id, source_point_id=point_id,
        receiver_axes="unchanged_original_receiver_material_axes"))


def _validate_banks(history, original, target):
    if type(target) is not ContactBankAttachment:
        raise ValueError("Target requires an explicit immutable material-bank attachment")
    # Touch every literal field even when a dataclass was forged via replace.
    target.__post_init__()
    if ((target.over_fragment_id, target.over_material_id, target.over_owner)
            != (original.over_fragment_id, original.over_material_id, original.over_owner)):
        raise ValueError("This law cannot move or replace the receiver material frame")
    fragments = {item.fragment_id: item for item in history.material_state.journal.source_surface.fragments}
    for fragment_id, material_id, owner in (
            (target.sub_fragment_id, target.sub_material_id, target.sub_owner),
            (target.over_fragment_id, target.over_material_id, target.over_owner)):
        fragment = fragments.get(fragment_id)
        if (fragment is None or fragment.parcel.material_id != material_id
                or fragment.parcel.plate != owner):
            raise ValueError("Target bank labels differ from the actual original material fragments")


def _law(law):
    if type(law) is not ContactHistoryTransportLaw:
        raise ValueError("Transport requires an explicit immutable law, without an implicit policy default")
    law.__post_init__()


def _parameters(original, target_columns, target_contact_law, target_water,
        target_area_m2=None, original_area_m2=None):
    if (_typed_fingerprint(target_columns) != _typed_fingerprint(original.columns)
            or _typed_fingerprint(target_contact_law) != _typed_fingerprint(original.law)
            or _typed_fingerprint(target_water) != _typed_fingerprint(original.water)):
        raise ValueError("Zero-work reference transport cannot change columns, contact law or water")
    if original_area_m2 is not None and (
            _typed_fingerprint(target_area_m2) != _typed_fingerprint(original_area_m2)):
        raise ValueError("Transport cannot change or invent the original quadrature measure")


@dataclass(frozen=True, slots=True)
class ContactReferenceTransportReceipt:
    revision: int
    source_binding_id: str
    target_binding_id: str
    source_banks: ContactBankAttachment
    target_banks: ContactBankAttachment
    source_geometry: tuple
    target_geometry: tuple
    source_geometry_context_json: bytes
    target_geometry_context_json: bytes
    original_state_sha256: str
    signed_external_work_j: Fraction = Fraction(0)
    dissipation_j: Fraction = Fraction(0)
    energy_change_j: Fraction = Fraction(0)


@dataclass(frozen=True, slots=True)
class ContactReferenceTransportState:
    history: NativePhysicalHistoryBinding
    point_id: str
    trace_id: str
    source_epoch: str
    law: ContactHistoryTransportLaw
    local_state: LocalContactState
    original_banks: ContactBankAttachment
    initial_binding_id: str
    initial_geometry: tuple
    initial_geometry_context_json: bytes
    active_binding_id: str
    active_banks: ContactBankAttachment
    reference_geometry: tuple
    reference_geometry_context_json: bytes
    receipts: tuple[ContactReferenceTransportReceipt, ...]
    state_sha256: str

    @property
    def revision(self):
        return len(self.receipts)

    @property
    def consumed_binding_ids(self):
        return tuple(receipt.source_binding_id for receipt in self.receipts)

    @property
    def original_area_m2(self):
        return self.local_state.area_m2

    @property
    def state(self):
        return self.local_state

    @property
    def active_history_count(self):
        return 1

    @property
    def work_lineage_id(self):
        # Before and candidate snapshots are successive records of ONE
        # physical quadrature history, never two independent energy stores.
        return (self.history.original_checkpoint_sha256, self.point_id)

    @property
    def attachment_committed(self):
        return bool(self.receipts)

    @property
    def global_force_candidate_adopted(self):
        return False

    @property
    def nodal_work_assigned_to_material_volume(self):
        return False


def _node(history, point_id):
    _identity(point_id, "Original quadrature point ID")
    node = next((item for item in history.quadrature_histories if item.point_id == point_id), None)
    if node is None:
        raise ValueError("Transport cannot allocate original quadrature history to a new finite point")
    return node


def _local_state(history, point_id, source_epoch):
    node = _node(history, point_id)
    if type(source_epoch) is not str or source_epoch not in ("before", "unaccepted_candidate"):
        raise ValueError("Nodal transport requires an explicit actual before or unaccepted-candidate epoch")
    local = node.previous_state if source_epoch == "before" else node.unaccepted_candidate_state
    if type(local) is not LocalContactState:
        raise ValueError("Transport requires the actual original local contact state")
    validate_contact_state(local, columns=node.original_material.columns,
        law=node.original_material.law, area_m2=node.original_area_m2,
        water=node.original_material.water)
    return node, local


def _receipt_to_dict(receipt):
    return dict(revision=receipt.revision, source_binding_id=receipt.source_binding_id,
        target_binding_id=receipt.target_binding_id,
        source_banks=_banks_to_dict(receipt.source_banks), target_banks=_banks_to_dict(receipt.target_banks),
        source_geometry=_geometry_to_dict(receipt.source_geometry),
        target_geometry=_geometry_to_dict(receipt.target_geometry),
        source_geometry_context=json.loads(receipt.source_geometry_context_json),
        target_geometry_context=json.loads(receipt.target_geometry_context_json),
        original_state_sha256=receipt.original_state_sha256,
        signed_external_work_j=[0, 1], dissipation_j=[0, 1], energy_change_j=[0, 1],
        strain_identity="contact(g,j-p), column(g-g_free,j-j_free), p,c unchanged",
        source_binding_consumed=True, target_binding_active=True,
        original_work_lineage_count=1)


def _state_body(state):
    node = _node(state.history, state.point_id)
    return dict(format=FORMAT, law=asdict(state.law),
        source_history=native_physical_history_binding_to_dict(state.history),
        point_id=state.point_id, trace_id=state.trace_id, source_epoch=state.source_epoch,
        original_local_state=state_to_dict(state.local_state),
        original_columns=asdict(node.original_material.columns),
        original_contact_law=asdict(node.original_material.law),
        original_water=node.original_material.water, original_area_m2=node.original_area_m2,
        original_banks=_banks_to_dict(state.original_banks),
        initial_binding_id=state.initial_binding_id,
        initial_geometry=_geometry_to_dict(state.initial_geometry),
        initial_geometry_context=json.loads(state.initial_geometry_context_json),
        active_binding_id=state.active_binding_id, active_banks=_banks_to_dict(state.active_banks),
        reference_geometry=_geometry_to_dict(state.reference_geometry),
        reference_geometry_context=json.loads(state.reference_geometry_context_json),
        receipts=[_receipt_to_dict(item) for item in state.receipts], revision=state.revision,
        consumed_binding_ids=list(state.consumed_binding_ids),
        offset_expression=OFFSET_EXPRESSION, anchor_expression=ANCHOR_EXPRESSION,
        receiver_frame_model="unchanged_original_receiver_material_axes",
        geometric_correspondence_authenticated=False,
        active_history_count=1, work_lineage_id=list(state.work_lineage_id),
        component_attachment_committed=state.attachment_committed,
        global_force_candidate_adopted=False, nodal_work_assigned_to_material_volume=False,
        consumption_scope="one immutable lineage; assembled ledgers must deduplicate source histories")


def _remember(state):
    # Authority is populated only by complete source/transition replay.  Keep
    # compact strict-type keys so many revisions neither duplicate the large
    # physical source fingerprint nor evict valid live immutable states.
    _AUTHENTICATED[_proof_key(state)] = state.state_sha256


def _proof_key(state):
    digest = hashlib.sha256()

    def visit(value):
        # Actual class identity matters: a dynamically created class with the
        # same module/name must not alias an authenticated nested dataclass.
        if isinstance(value, type):
            payload, tag = str(id(value)).encode("ascii"), b"class"
        elif type(value) is tuple:
            digest.update(b"tuple[")
            for item in value:
                visit(item)
            digest.update(b"]")
            return
        elif type(value) is bytes:
            payload, tag = value, b"bytes"
        elif type(value) is str:
            payload, tag = value.encode("utf-8"), b"str"
        elif type(value) is bool:
            payload, tag = str(value).encode("ascii"), b"bool"
        elif type(value) is int:
            payload, tag = str(value).encode("ascii"), b"int"
        elif value is None:
            payload, tag = b"", b"none"
        else:
            raise ValueError("Transport proof contains an unsupported literal fingerprint field")
        digest.update(tag+b":"+str(len(payload)).encode("ascii")+b":"+payload+b";")

    visit(_typed_fingerprint(state))
    return digest.hexdigest()


def _seal(state, body):
    result = replace(state, state_sha256=_hash(body(state)))
    _remember(result)
    return result


def initialize_contact_reference_transport(history, point_id, *, law, source_epoch,
        source_binding_id, h_old, geometry_context, expected_source_history_dto):
    """Attach an existing complete local state without evolving it or resetting it."""
    _source(history, expected_source_history_dto)
    _law(law)
    _identity(source_binding_id, "Source active binding ID")
    geometry = _geometry(h_old)
    context = _context(geometry_context, "Source projected-geometry context")
    node, local = _local_state(history, point_id, source_epoch)
    banks = _original_banks(history, node.trace_id, point_id)
    result = ContactReferenceTransportState(history, point_id, node.trace_id,
        source_epoch, law, local, banks, source_binding_id, geometry, context,
        source_binding_id, banks, geometry, context, (), "")
    return _seal(result, _state_body)


def _next_identity(state, target_binding_id):
    _identity(target_binding_id, "Target active binding ID")
    if target_binding_id == state.active_binding_id or target_binding_id in state.consumed_binding_ids:
        raise ValueError("Transport cannot reuse an active or consumed binding identity")


def transport_contact_reference(state, *, target_binding_id, target_banks, h_new,
        geometry_context, target_columns, target_contact_law, target_water,
        target_area_m2, expected_source_history_dto):
    """Commit a new attachment/reference with identical strains, preload and work.

    This pure immutable operator consumes the previous attachment in its own
    ledger.  A caller assembling multiple lineages must reject duplicate
    ``work_lineage_id`` values; immutability alone cannot prohibit forks.
    """
    validate_contact_reference_transport(state)
    _source(state.history, expected_source_history_dto)
    _next_identity(state, target_binding_id)
    _validate_banks(state.history, state.original_banks, target_banks)
    geometry = _geometry(h_new)
    context = _context(geometry_context, "Target projected-geometry context")
    node = _node(state.history, state.point_id)
    _parameters(node.original_material, target_columns, target_contact_law,
                target_water, target_area_m2, node.original_area_m2)
    validate_contact_state(state.local_state, columns=target_columns,
        law=target_contact_law, area_m2=target_area_m2, water=target_water)
    receipt = ContactReferenceTransportReceipt(state.revision+1,
        state.active_binding_id, target_binding_id, state.active_banks, target_banks,
        state.reference_geometry, geometry, state.reference_geometry_context_json,
        context, _hash(state_to_dict(state.local_state)))
    result = replace(state, active_binding_id=target_binding_id, active_banks=target_banks,
        reference_geometry=geometry, reference_geometry_context_json=context,
        receipts=(*state.receipts, receipt), state_sha256="")
    return _seal(result, _state_body)


def evaluate_contact_reference_free_anchors(state, h_current, *, geometry_context):
    """Evaluate the reference expression, retaining its event cancellation.

    Identical authenticated geometry expressions cancel symbolically, including
    interval-valued expressions.  Different contexts receive ordinary enclosing
    interval subtraction; the operator never pretends correlated uncertainty
    is a scalar known by selecting an interval midpoint.
    """
    validate_contact_reference_transport(state)
    geometry = _geometry(h_current)
    context = _context(geometry_context, "Current projected-geometry context")
    free = tuple(_fraction(value) for value in
                 (state.local_state.gap_free_m, state.local_state.jump_free_m))
    if context == state.reference_geometry_context_json:
        if _typed_fingerprint(geometry) != _typed_fingerprint(state.reference_geometry):
            raise ValueError("The identical geometry context cannot change its projected expression")
        return tuple(I(value) for value in free)
    return tuple(I(value)+(_interval(current)-_interval(reference))
                 for value, current, reference in zip(free, geometry, state.reference_geometry))


def validate_contact_reference_transport(state):
    if type(state) is not ContactReferenceTransportState:
        raise ValueError("Expected an authenticated immutable contact-reference transport state")
    if _AUTHENTICATED.get(_proof_key(state)) != state.state_sha256:
        raise ValueError("Contact-reference state was not built by a complete source/transition replay")
    return state


def contact_reference_transport_to_dict(state):
    validate_contact_reference_transport(state)
    return _portable(dict(**_state_body(state), state_sha256=state.state_sha256))


def contact_reference_transport_from_dict(raw, *, history,
        expected_source_history_dto, law):
    """Strictly replay the entire transition ledger against an external source."""
    if type(raw) is not dict:
        raise ValueError("Contact-reference restart requires the complete DTO")
    _literal_json(raw)
    _law(law)
    try:
        if _encoded(raw["law"]) != _encoded(asdict(law)):
            raise ValueError("Restart changed the caller's explicitly declared transport law")
        _source(history, expected_source_history_dto)
        if _encoded(raw["source_history"]) != _encoded(expected_source_history_dto):
            raise ValueError("Restart changed the independent complete physical source")
        state = initialize_contact_reference_transport(history, raw["point_id"],
            law=law, source_epoch=raw["source_epoch"], source_binding_id=raw["initial_binding_id"],
            h_old=_geometry_from_dict(raw["initial_geometry"]),
            geometry_context=raw["initial_geometry_context"],
            expected_source_history_dto=expected_source_history_dto)
        if type(raw["receipts"]) is not list:
            raise ValueError("Restart transition ledger requires literal JSON rows")
        node = _node(history, state.point_id)
        for receipt in raw["receipts"]:
            if type(receipt) is not dict:
                raise ValueError("Restart transition ledger contains a nonliteral row")
            state = transport_contact_reference(state,
                target_binding_id=receipt["target_binding_id"],
                target_banks=_banks_from_dict(receipt["target_banks"]),
                h_new=_geometry_from_dict(receipt["target_geometry"]),
                geometry_context=receipt["target_geometry_context"],
                target_columns=node.original_material.columns,
                target_contact_law=node.original_material.law,
                target_water=node.original_material.water, target_area_m2=node.original_area_m2,
                expected_source_history_dto=expected_source_history_dto)
    except (KeyError, TypeError) as error:
        raise ValueError("Contact-reference restart has missing or changed schema fields") from error
    if _encoded(contact_reference_transport_to_dict(state)) != _encoded(raw):
        raise ValueError("Contact-reference restart differs from full literal transition replay")
    return state


@dataclass(frozen=True, slots=True)
class ContactReferenceTransportSymbolReceipt:
    revision: int
    source_binding_id: str
    target_binding_id: str
    source_banks: ContactBankAttachment
    target_banks: ContactBankAttachment
    source_geometry_context_json: bytes
    target_geometry_context_json: bytes


@dataclass(frozen=True, slots=True)
class ContactReferenceTransportSymbolFamily:
    history: NativePhysicalHistoryBinding
    trace_id: str
    law: ContactHistoryTransportLaw
    original_banks: ContactBankAttachment
    initial_binding_id: str
    initial_geometry_context_json: bytes
    active_binding_id: str
    active_banks: ContactBankAttachment
    reference_geometry_context_json: bytes
    receipts: tuple[ContactReferenceTransportSymbolReceipt, ...]
    state_sha256: str

    @property
    def revision(self):
        return len(self.receipts)

    @property
    def consumed_binding_ids(self):
        return tuple(item.source_binding_id for item in self.receipts)

    @property
    def source_contact_time_path(self):
        return self.history.contact_time_path

    @property
    def source_continuous_memory(self):
        return self.history.contact_time_path.step.previous_memory

    @property
    def finite_point_area_m2(self):
        return None

    @property
    def finite_point_work_j(self):
        return None

    @property
    def global_force_candidate_adopted(self):
        return False


def _symbol_receipt_to_dict(receipt):
    return dict(revision=receipt.revision, source_binding_id=receipt.source_binding_id,
        target_binding_id=receipt.target_binding_id,
        source_banks=_banks_to_dict(receipt.source_banks), target_banks=_banks_to_dict(receipt.target_banks),
        source_geometry_context=json.loads(receipt.source_geometry_context_json),
        target_geometry_context=json.loads(receipt.target_geometry_context_json),
        signed_transfer_external_work_identity="0", transfer_dissipation_identity="0",
        strain_energy_change_identity="0 at every common material coordinate and exact clock",
        source_binding_consumed=True, finite_point_area_m2=None, finite_point_work_j=None)


def _family_body(family):
    trace = next(item for item in family.history.initial_mechanics.fault_traces.traces
                 if item.trace_id == family.trace_id)
    return dict(format=FAMILY_FORMAT, law=asdict(family.law),
        source_history=native_physical_history_binding_to_dict(family.history),
        trace_id=family.trace_id, original_banks=_banks_to_dict(family.original_banks),
        initial_binding_id=family.initial_binding_id,
        initial_geometry_context=json.loads(family.initial_geometry_context_json),
        active_binding_id=family.active_binding_id, active_banks=_banks_to_dict(family.active_banks),
        reference_geometry_context=json.loads(family.reference_geometry_context_json),
        receipts=[_symbol_receipt_to_dict(item) for item in family.receipts],
        revision=family.revision, consumed_binding_ids=list(family.consumed_binding_ids),
        original_columns=asdict(trace.columns), original_contact_law=asdict(trace.law),
        original_water=trace.water,
        continuous_interface_measure="original R * contact_height / sin(dip) * dtheta; not a finite point area",
        original_contact_time_path_sha256=family.source_contact_time_path.path_sha256,
        memory_identity="original continuous p(theta,u), c(theta,u) retained in the declared dense path",
        strain_identity="g,j-p,g-g_free,j-j_free retained at the same original theta,u",
        offset_expression=OFFSET_EXPRESSION+"(theta,u)",
        anchor_expression=ANCHOR_EXPRESSION+"(theta,u)",
        receiver_frame_model="unchanged_original_receiver_material_axes",
        geometric_correspondence_authenticated=False,
        finite_point_area_m2=None, finite_point_work_j=None,
        field_scope="implicit exact-clock reference identity; no midpoint or finite nodal state",
        global_force_candidate_adopted=False, nodal_work_assigned_to_material_volume=False)


def initialize_contact_reference_transport_symbols(history, trace_id, *, law,
        source_binding_id, geometry_context, expected_source_history_dto):
    """Retain the entire original theta/time history as a symbolic source.

    A family is not a LocalContactState and carries no finite quadrature area or
    work.  A caller can compose its new geometry expression with an implicit
    first-root clock without substituting a root-window midpoint.  Its identity
    compares the same theta/u on both sides; it is not a future time-stepping
    rule with a reference frozen at one chosen transfer clock.
    """
    _source(history, expected_source_history_dto)
    _law(law)
    _identity(source_binding_id, "Source symbolic binding ID")
    _trace_binding(history, trace_id)
    context = _context(geometry_context, "Original symbolic geometry expression")
    banks = _original_banks(history, trace_id, "continuous_original_material_coordinate")
    result = ContactReferenceTransportSymbolFamily(history, trace_id, law, banks,
        source_binding_id, context, source_binding_id, banks, context, (), "")
    return _seal(result, _family_body)


def validate_contact_reference_transport_symbols(family):
    if type(family) is not ContactReferenceTransportSymbolFamily:
        raise ValueError("Expected an authenticated immutable reference-transport symbol family")
    if _AUTHENTICATED.get(_proof_key(family)) != family.state_sha256:
        raise ValueError("Symbol family was not built by a complete physical source/transition replay")
    return family


def transport_contact_reference_symbols(family, *, target_binding_id, target_banks,
        geometry_context, target_columns, target_contact_law, target_water,
        expected_source_history_dto):
    validate_contact_reference_transport_symbols(family)
    _source(family.history, expected_source_history_dto)
    _next_identity(family, target_binding_id)
    _validate_banks(family.history, family.original_banks, target_banks)
    trace = next(item for item in family.history.initial_mechanics.fault_traces.traces
                 if item.trace_id == family.trace_id)
    _parameters(trace, target_columns, target_contact_law, target_water)
    context = _context(geometry_context, "Target implicit symbolic geometry expression")
    receipt = ContactReferenceTransportSymbolReceipt(family.revision+1,
        family.active_binding_id, target_binding_id, family.active_banks, target_banks,
        family.reference_geometry_context_json, context)
    return _seal(replace(family, active_binding_id=target_binding_id, active_banks=target_banks,
        reference_geometry_context_json=context, receipts=(*family.receipts, receipt),
        state_sha256=""), _family_body)


def evaluate_contact_reference_transport_symbols(family, material_angle_bounds,
        time_fraction_bounds, h_current, h_reference, *, geometry_context,
        reference_geometry_context, precision=128):
    """Evaluate original free fields plus a declared geometric difference.

    This is an enclosing observation of an implicit family.  It is never used
    to construct a finite state at a root midpoint or to infer plastic memory
    from nearby quadrature nodes.
    """
    validate_contact_reference_transport_symbols(family)
    if type(precision) is not int or not 32 <= precision <= 1024:
        raise ValueError("Symbol-family precision must be a literal integer from 32 to 1024")
    if (type(material_angle_bounds) is not I or type(time_fraction_bounds) is not I
            or any(type(value) is not Fraction for value in
                   (material_angle_bounds.lo, material_angle_bounds.hi,
                    time_fraction_bounds.lo, time_fraction_bounds.hi))
            or not 0 <= time_fraction_bounds.lo <= time_fraction_bounds.hi <= 1):
        raise ValueError("Symbol family requires exact angle and normalized clock intervals")
    index = next(i for i, row in enumerate(family.history.contact_time_path.previous_anchor_fields)
                 if row[0] == family.trace_id)
    path = family.history.contact_time_path
    extent = _fraction(path.step.previous_memory.traces[index].extent_radians)
    if not 0 <= material_angle_bounds.lo <= material_angle_bounds.hi <= extent:
        raise ValueError("Symbol query exceeds its original continuous material trace")
    current, reference = _geometry(h_current), _geometry(h_reference)
    context = _context(geometry_context, "Current symbolic projection context")
    reference_context = _context(reference_geometry_context, "Reference symbolic projection context")
    if reference_context != family.reference_geometry_context_json:
        raise ValueError("Symbol query changed the active reference expression")
    _, old_gap, old_jump = path.previous_anchor_fields[index]
    _, new_gap, new_jump = path.step.anchor_fields[index]
    values = []
    for old, new in ((old_gap, new_gap), (old_jump, new_jump)):
        old_coefficients, new_coefficients = _field(old), _field(new)
        values.append(_affine_bounds(old_coefficients,
            _add(new_coefficients, _scale(old_coefficients, -1)),
            material_angle_bounds.lo, material_angle_bounds.hi,
            time_fraction_bounds.lo, time_fraction_bounds.hi, precision))
    if context == reference_context:
        if _typed_fingerprint(current) != _typed_fingerprint(reference):
            raise ValueError("Identical symbolic expressions cannot have different projections")
        return tuple(values)
    return tuple(value+(_interval(now)-_interval(ref))
                 for value, now, ref in zip(values, current, reference))


def contact_reference_transport_symbols_to_dict(family):
    validate_contact_reference_transport_symbols(family)
    return _portable(dict(**_family_body(family), state_sha256=family.state_sha256))


def contact_reference_transport_symbols_from_dict(raw, *, history,
        expected_source_history_dto, law):
    if type(raw) is not dict:
        raise ValueError("Symbol-family restart requires its complete DTO")
    _literal_json(raw)
    _law(law)
    try:
        if _encoded(raw["law"]) != _encoded(asdict(law)):
            raise ValueError("Symbol-family restart changed the explicit transport law")
        _source(history, expected_source_history_dto)
        if _encoded(raw["source_history"]) != _encoded(expected_source_history_dto):
            raise ValueError("Symbol-family restart changed the complete independent source")
        family = initialize_contact_reference_transport_symbols(history, raw["trace_id"],
            law=law, source_binding_id=raw["initial_binding_id"],
            geometry_context=raw["initial_geometry_context"],
            expected_source_history_dto=expected_source_history_dto)
        if type(raw["receipts"]) is not list:
            raise ValueError("Symbol-family transition ledger requires literal JSON rows")
        trace = next(item for item in history.initial_mechanics.fault_traces.traces
                     if item.trace_id == family.trace_id)
        for receipt in raw["receipts"]:
            family = transport_contact_reference_symbols(family,
                target_binding_id=receipt["target_binding_id"],
                target_banks=_banks_from_dict(receipt["target_banks"]),
                geometry_context=receipt["target_geometry_context"],
                target_columns=trace.columns, target_contact_law=trace.law,
                target_water=trace.water, expected_source_history_dto=expected_source_history_dto)
    except (KeyError, TypeError) as error:
        raise ValueError("Symbol-family restart has missing or changed schema fields") from error
    if _encoded(contact_reference_transport_symbols_to_dict(family)) != _encoded(raw):
        raise ValueError("Symbol-family restart differs from the entire literal transition replay")
    return family


def validate_contact_reference_transport_ledger(states):
    """Enforce one active energy/work lineage in an assembled nodal ledger."""
    if type(states) is not tuple:
        raise ValueError("Transport ledger must be an immutable tuple of active states")
    histories, attachments = set(), set()
    for state in states:
        validate_contact_reference_transport(state)
        if state.work_lineage_id in histories or state.active_binding_id in attachments:
            raise ValueError("Assembled ledger duplicates an original energy/work history or active binding")
        histories.add(state.work_lineage_id)
        attachments.add(state.active_binding_id)
    return states
