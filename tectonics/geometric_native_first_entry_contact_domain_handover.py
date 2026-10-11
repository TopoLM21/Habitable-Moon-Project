"""One contact-field registry on the complete implicit first-entry domains.

Four receiving fields are owned once.  Sixteen causal branches refer to those
fields and to the original incoming material; they are not sixteen additional
springs, areas or work stores.  The first-event map freezes each material
reference on its own implicit event.  A conservative whole-strip bound licenses
the original donor's Ritz shape on every incoming label, including cross-pair
entry.  Source contact-line budgets and swept material coarea retain their
different measures.  Original literal quadrature work remains a separate,
unchanged source representation.  No energy density on archived bulk material
or accepted force interval is inferred from any of these correspondences.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction as F
import hashlib
import json

from .geometric_native_pose import FractionInterval as I
from .geometric_native_physical_history import (
    _dyadic, _literal_json, native_physical_history_binding_to_dict,
    validate_native_physical_history_binding,
)
from .geometric_native_contact_reference_transport import (
    ContactHistoryTransportLaw, make_contact_bank_attachment, _banks_to_dict,
    _parameters, _proof_key, _validate_banks,
)
from .geometric_native_first_entry_interface_integral import (
    certify_native_first_entry_interface_integral,
    evaluate_native_first_entry_interface_carrier_mask,
    native_first_entry_interface_integral_to_dict,
)
from .geometric_native_first_entry_physical_binding import (
    validate_native_first_entry_physical_binding,
)
from .geometric_native_first_entry_contact_handover import (
    certify_native_first_entry_contact_handover,
    native_first_entry_contact_handover_to_dict,
)
from .geometric_native_contact_field_work import (
    integrate_native_contact_field_work_over_original_interface,
    query_native_contact_field_work_primitive,
    native_contact_field_work_primitive_to_dict,
)
from .geometric_native_contact_field_history import _portable


FORMAT = "native-whole-first-entry-contact-domain-handover-1"
POINT_FORMAT = "native-whole-first-entry-contact-domain-point-query-1"
MODEL = "four_once_owned_receiver_fields_sixteen_causal_material_pullbacks"
_AUTHENTICATED = {}
_DERIVATIONS = {}


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _source(history, expected, law):
    validate_native_physical_history_binding(history)
    if type(law) is not ContactHistoryTransportLaw:
        raise ValueError("Whole field transport requires an explicit complete-configuration law")
    law.__post_init__()
    if type(expected) is not dict:
        raise ValueError("Whole field transport needs its entire independent physical source DTO")
    _literal_json(expected)
    payload = _encoded(expected)
    if payload != _encoded(native_physical_history_binding_to_dict(history)):
        raise ValueError("Whole field transport differs from the complete authenticated source")
    return payload


@dataclass(frozen=True, slots=True)
class NativeFirstEntryReceiverFieldFamily:
    family_id: str
    trace_id: str
    original_binding_id: str
    active_binding_id: str
    original_banks: object
    consumed_binding_ids: tuple
    memory_reference: str
    reference_clock_expression: str

    @property
    def revision(self):
        return 1


@dataclass(frozen=True, slots=True)
class NativeFirstEntryDomainContactBranch:
    branch_id: str
    mask: object
    source_family_id: str
    donor_trace_id: str
    receiver_trace_id: str
    target_banks: object
    anchor_expression: str
    frozen_first_event_expression: str


@dataclass(frozen=True, slots=True)
class NativeFirstEntryWholeDonorBeamSupport:
    donor_trace_id: str
    donor_fragment_id: str
    radius_m: F
    width_m: F
    rotation_path_bound_rad: F
    max_inward_angle_rad: F
    max_inward_distance_m: F
    shape_bounds: I

    @property
    def strip_membership_proved(self):
        return True

    @property
    def native_force_and_clamp_support_certified(self):
        return False


@dataclass(frozen=True, slots=True)
class NativeFirstEntryContactDomainHandover:
    history: object
    law: ContactHistoryTransportLaw
    interface_integral: object
    families: tuple
    branches: tuple
    source_beam_supports: tuple
    source_work_budgets: tuple
    precision: int
    order: int
    source_history_payload_json: bytes
    state_sha256: str

    @property
    def parents(self):
        return self.interface_integral.parent_measures

    @property
    def original_atoms(self):
        return self.history.quadrature_histories

    @property
    def whole_domain_field_transfer_committed(self):
        return True

    @property
    def whole_domain_handover_work_j(self):
        return F(0)

    @property
    def whole_domain_handover_dissipation_j(self):
        return F(0)

    @property
    def whole_domain_handover_energy_change_j(self):
        return F(0)

    @property
    def physical_pair_admission_authorized(self):
        return False

    @property
    def candidate_mechanics_adopted(self):
        return False


@dataclass(frozen=True, slots=True)
class NativeFirstEntryContactDomainPointQuery:
    registry: NativeFirstEntryContactDomainHandover
    binding: object
    point_handover: object
    branches: tuple
    source_family_ids: tuple
    source_work_primitives: tuple

    @property
    def status(self):
        return self.point_handover.status

    @property
    def registered_history_created(self):
        return False

    @property
    def finite_point_area_m2(self):
        return None

    @property
    def finite_point_work_j(self):
        return None


def _beam_supports(history, integral):
    # Rotation curve length bounds the geodesic distance from the original
    # X=0 carrier to its donor pullback. The L1 bound uses every saved axis
    # component and both full rotations; it is not a relative-axis shortcut.
    bound = history.dt_myr*sum((sum(map(abs, omega), F(0))
                               for _, omega in history.omega_records), F(0))
    if not 0 < bound < F(1, 4):
        raise ValueError("Whole donor-strip proof needs a positive bounded original rotation path")
    # For a unit q on the causal source mask, 0<=q_x<=B.  Asin(q_x)
    # <= q_x/(1-B²) follows by integrating 1/sqrt(1-t²). Raw original bulk
    # labels use the same homogeneous angle; saved Gauss labels are untouched.
    angle = bound/(1-bound*bound)
    parameters = {item.trace_id: item for item in history.initial_mechanics.bending_parameters}
    result = []
    for binding in history.material_state.trace_bindings:
        width = _dyadic(parameters[binding.trace_id].sub.width_m)
        distance = integral.radius_m*angle
        if width <= 0 or distance >= width:
            raise ValueError("An original incoming material domain exceeds its own donor beam strip")
        z = distance/width
        minimum = 1-F(3, 2)*z+F(1, 2)*z**3
        if not 0 < minimum < 1:
            raise ValueError("Whole incoming Ritz-shape bound lost the original positive support")
        result.append(NativeFirstEntryWholeDonorBeamSupport(binding.trace_id,
            binding.donor_fragment_id, integral.radius_m, width, bound, angle,
            distance, I(minimum, F(1))))
    return tuple(result)


def certify_native_first_entry_contact_domain_handover(history, *, law,
        expected_source_history_dto, precision=256, order=32):
    """Commit a complete functional field transfer, with one source per receiver.

    The global balances follow from unchanged field expressions at every
    common first-event coordinate and unchanged interface measure. Per-pair
    numerical weighted integrals are not substituted for those identities.
    """
    payload = _source(history, expected_source_history_dto, law)
    key = _proof_key((history, law, precision, order))
    if key in _DERIVATIONS:
        return _DERIVATIONS[key]
    integral = certify_native_first_entry_interface_integral(history,
        expected_source_history_dto=expected_source_history_dto,
        precision=precision, order=order)
    traces = {item.trace_id: item for item in history.initial_mechanics.fault_traces.traces}
    bindings = {item.trace_id: item for item in history.material_state.trace_bindings}
    families, budgets = [], []
    prefix = "domain-contact:"+history.original_checkpoint_sha256+":"
    for interface in integral.original_interfaces:
        bank = bindings[interface.trace_id]
        identity = prefix+interface.trace_id
        original_banks = make_contact_bank_attachment(bank.donor_fragment_id,
            bank.donor_material_id, bank.subducting_plate, bank.receiver_fragment_id,
            bank.receiver_material_id, bank.overriding_plate,
            correspondence=dict(model="original_receiver_attached_full_field",
                trace_id=interface.trace_id, source_path=history.contact_time_path.path_sha256))
        families.append(NativeFirstEntryReceiverFieldFamily(identity, interface.trace_id,
            identity+":original", identity+":first-entry-field:1", original_banks,
            (identity+":original",), history.contact_time_path.path_sha256,
            "each material reference is frozen at its unique implicit first_entry(q)"))
        budget = integrate_native_contact_field_work_over_original_interface(history,
            interface.trace_id, F(0), interface.computational_extent_radians, F(0), F(1),
            expected_source_history_dto=expected_source_history_dto,
            source_expression_context=dict(model=MODEL, family_id=identity,
                scope="original_receiver_line_once_unaccepted_source_path"), precision=precision)
        if (budget.continuum_interface_area_m2 != interface.whole_interface_area_m2
                or budget.original_nodal_interface_area_m2 != interface.quadrature_area_total_m2):
            raise ValueError("Whole source work changed the declared original interface measures")
        budgets.append(budget)
    family_map = {item.trace_id: item for item in families}
    branches = []
    for mask in integral.carrier_masks:
        family = family_map[mask.original_receiver_trace_id]
        donor, receiver = traces[mask.original_donor_trace_id], traces[mask.original_receiver_trace_id]
        source, target = bindings[donor.trace_id], bindings[receiver.trace_id]
        _parameters(receiver, donor.columns, donor.law, donor.water)
        context = dict(model=MODEL, family_id=family.family_id,
            donor_parent_id=mask.original_donor_fragment_id,
            first_receiver_id=mask.original_receiver_fragment_id,
            causal_predicate=mask.causal_predicate,
            label_expression="original_q=Q_d(first_entry)^T*Q_r(first_entry)*c(theta(first_entry))",
            label_frozen_during_later_force_variations=True,
            reference_clock_frozen=True, midpoint_used=False)
        banks = make_contact_bank_attachment(source.donor_fragment_id, source.donor_material_id,
            source.subducting_plate, target.receiver_fragment_id, target.receiver_material_id,
            target.overriding_plate, correspondence=context)
        _validate_banks(history, family.original_banks, banks)
        branch_id = family.active_binding_id+":"+source.donor_fragment_id
        branches.append(NativeFirstEntryDomainContactBranch(branch_id, mask, family.family_id,
            donor.trace_id, receiver.trace_id, banks,
            "source_free(theta*,u*) + h_current(original_q) - h_first_entry(original_q)",
            "theta*=receiver_coordinate(first_entry(original_q)); u*=first_entry(original_q)"))
    identities = tuple((item.mask.original_donor_fragment_id,
                        item.mask.original_receiver_fragment_id) for item in branches)
    if len(families) != 4 or len(branches) != 16 or len(set(identities)) != 16:
        raise ValueError("Whole contact transport must cover all original causal donor/receiver branches")
    atoms = history.quadrature_histories
    if len(atoms) != 16 or len({(history.original_checkpoint_sha256, x.point_id) for x in atoms}) != 16:
        raise ValueError("Whole contact transport lost its once-only literal source-work lineages")
    value = NativeFirstEntryContactDomainHandover(history, law, integral, tuple(families),
        tuple(branches), _beam_supports(history, integral), tuple(budgets),
        precision, order, payload, "")
    body = _encoded(_body(value))
    value = replace(value, state_sha256=hashlib.sha256(body).hexdigest())
    _AUTHENTICATED[_proof_key(value)] = body
    _DERIVATIONS[key] = value
    return value


def validate_native_first_entry_contact_domain_handover(value):
    if type(value) is not NativeFirstEntryContactDomainHandover or _proof_key(value) not in _AUTHENTICATED:
        raise ValueError("Whole contact field was not committed by complete typed source replay")
    return value


def evaluate_native_first_entry_contact_domain_point(registry, binding, *,
        expected_source_history_dto):
    """Observe an original material label without registering another history."""
    validate_native_first_entry_contact_domain_handover(registry)
    _source(registry.history, expected_source_history_dto, registry.law)
    validate_native_first_entry_physical_binding(binding)
    if _proof_key(binding.history) != _proof_key(registry.history):
        raise ValueError("A domain contact observation belongs to another physical source")
    handover = certify_native_first_entry_contact_handover(binding, law=registry.law,
        expected_source_history_dto=expected_source_history_dto,
        precision=registry.precision, order=registry.order)
    branches, families, primitives = [], [], []
    for attachment in handover.attachments:
        matches = tuple(branch for branch in registry.branches
            if (branch.mask.original_donor_fragment_id, branch.mask.original_receiver_fragment_id)
            == attachment.desired_fragment_pair)
        if len(matches) != 1:
            raise ValueError("A first-event observation lacks its whole causal material branch")
        branch, = matches
        if evaluate_native_first_entry_interface_carrier_mask(branch.mask, binding.chronology) is not True:
            raise ValueError("A first-event observation did not prove its complete causal mask")
        branches.append(branch)
        families.append(branch.source_family_id)
        field = attachment.source_field_query
        primitives.append(query_native_contact_field_work_primitive(registry.history,
            field.trace_id, field.computational_angle_bounds, F(0), field.time_fraction_bounds,
            expected_source_history_dto=expected_source_history_dto,
            source_expression_context=attachment.expression_context, precision=registry.precision))
    return NativeFirstEntryContactDomainPointQuery(registry, binding, handover,
                                                  tuple(branches), tuple(families), tuple(primitives))


def _budget_body(budget):
    source = budget.source_time_integral.source_field_query
    body = {name: _portable(getattr(budget, name)) for name in budget.__dataclass_fields__
            if name != "source_time_integral"}
    body.update(trace_id=source.trace_id, source_path=source.history.contact_time_path.path_sha256,
        source_expression_context=source.source_expression_context,
        computational_angle_bounds=_portable(source.computational_angle_bounds),
        time_fraction_bounds=_portable(source.time_fraction_bounds),
        balance_identity="source_work=total_energy_increment+friction_dissipation+signed_compliance_residual",
        literal_atom_ledgers_replaced=False, is_swept_material_energy_density=False)
    return body


def _body(value):
    coarea = native_first_entry_interface_integral_to_dict(value.interface_integral,
        expected_source_history_dto=json.loads(value.source_history_payload_json))
    return dict(format=FORMAT, model=MODEL,
        source_history=json.loads(value.source_history_payload_json),
        precision=value.precision, order=value.order,
        explicit_law=dict(model=value.law.model, provenance=value.law.provenance),
        interface_integral=coarea,
        families=[dict(family_id=item.family_id, trace_id=item.trace_id,
            original_binding_id=item.original_binding_id, active_binding_id=item.active_binding_id,
            original_banks=_banks_to_dict(item.original_banks),
            consumed_binding_ids=list(item.consumed_binding_ids), revision=1,
            memory_reference=item.memory_reference, reference_clock_expression=item.reference_clock_expression,
            original_history_count=1, active_history_count=1,
            inactive_or_reentry_branch="retains original attachment; no second family or work debit")
            for item in value.families],
        branches=[dict(branch_id=item.branch_id, source_family_id=item.source_family_id,
            donor_trace_id=item.donor_trace_id, receiver_trace_id=item.receiver_trace_id,
            original_donor_fragment_id=item.mask.original_donor_fragment_id,
            original_receiver_fragment_id=item.mask.original_receiver_fragment_id,
            archive_domain_id=item.mask.archive_domain_id,
            target_banks=_banks_to_dict(item.target_banks),
            anchor_expression=item.anchor_expression,
            frozen_first_event_expression=item.frozen_first_event_expression,
            new_finite_history_created=False, additional_area_or_work_created=False)
            for item in value.branches],
        source_beam_supports=[dict(_portable(item), strip_membership_proved=True,
            proof="unit q on original parent; |q_x|<=sum full rotation curve lengths=B; asin(q_x)<=B/(1-B^2); R*angle<L",
            original_label_normalization="homogeneous incoming bulk chart; original Gauss labels unchanged",
            native_force_and_clamp_support_certified=False) for item in value.source_beam_supports],
        source_work_budgets=[_budget_body(item) for item in value.source_work_budgets],
        original_atom_lineages=[dict(checkpoint_sha256=value.history.original_checkpoint_sha256,
            point_id=item.point_id, trace_id=item.trace_id, original_area_m2=_portable(_dyadic(item.original_area_m2)),
            state_preserved_once=True, full_state_reference="source_history.quadrature_histories")
            for item in value.original_atoms],
        whole_domain_field_transfer_committed=True, original_parent_count=8,
        receiver_field_family_count=4, causal_material_branch_count=16,
        exact_global_transfer_balances=dict(external_work_j=[0, 1], dissipation_j=[0, 1],
            strain_energy_change_j=[0, 1],
            proof="identical source p,c,strains,tractions,energy at every common implicit first event with identical receiver interface measure",
            source_history_owned_once=True, first_event_mask_excludes_reentry=True),
        source_line_work_integrals_evaluated=True,
        receiving_pair_weighted_contact_integrals_numerically_enclosed=False,
        interface_energy_or_work_density_on_archived_bulk_declared=False,
        contactline_work_and_literal_atom_work_are_summed=False,
        source_mechanical_candidate_work_replaced_by_contactline_budget=False,
        physical_pair_admission_authorized=False, candidate_mechanics_adopted=False,
        accepted_native_force_intervals=0, accepted_source_work_j=[0, 1])


def native_first_entry_contact_domain_handover_to_dict(value):
    validate_native_first_entry_contact_domain_handover(value)
    body = json.loads(_AUTHENTICATED[_proof_key(value)])
    body["state_sha256"] = value.state_sha256
    return body


def native_first_entry_contact_domain_handover_from_dict(raw, *, expected_history,
        expected_source_history_dto, expected_law):
    _literal_json(raw)
    if type(raw) is not dict:
        raise ValueError("Whole contact registry requires a complete object")
    repeated = certify_native_first_entry_contact_domain_handover(expected_history,
        law=expected_law, expected_source_history_dto=expected_source_history_dto,
        precision=raw.get("precision"), order=raw.get("order"))
    if _encoded(raw) != _encoded(native_first_entry_contact_domain_handover_to_dict(repeated)):
        raise ValueError("Whole contact registry differs from its independent complete source and law")
    return repeated


def native_first_entry_contact_domain_point_to_dict(query):
    repeated = evaluate_native_first_entry_contact_domain_point(query.registry, query.binding,
        expected_source_history_dto=json.loads(query.registry.source_history_payload_json))
    if _proof_key(query) != _proof_key(repeated):
        raise ValueError("Material contact query differs from its whole registered field")
    body = dict(format=POINT_FORMAT, registry_sha256=query.registry.state_sha256,
        point_handover=native_first_entry_contact_handover_to_dict(query.point_handover),
        branches=[branch.branch_id for branch in query.branches],
        source_family_ids=list(query.source_family_ids),
        source_work_primitives=[native_contact_field_work_primitive_to_dict(primitive,
            history=query.registry.history,
            expected_source_history_dto=json.loads(query.registry.source_history_payload_json))
            for primitive in query.source_work_primitives],
        status=query.status, registered_history_created=False,
        finite_point_area_m2=None, finite_point_work_j=None,
        additional_handover_work_j=[0, 1], physical_pair_admission_authorized=False)
    body["query_sha256"] = hashlib.sha256(_encoded(body)).hexdigest()
    return body


def native_first_entry_contact_domain_point_from_dict(raw, *, expected_registry,
        expected_binding, expected_source_history_dto):
    _literal_json(raw)
    repeated = evaluate_native_first_entry_contact_domain_point(expected_registry,
        expected_binding, expected_source_history_dto=expected_source_history_dto)
    if type(raw) is not dict or _encoded(raw) != _encoded(native_first_entry_contact_domain_point_to_dict(repeated)):
        raise ValueError("Material contact query differs from the independent registered source and first event")
    return repeated
