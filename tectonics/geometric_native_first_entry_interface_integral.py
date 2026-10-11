"""Whole first-entry carrier masks and their once-only coarea measure.

For a receiving quarter ``c(theta)`` the original donor label is
``q(theta,u)=Q_d(u)^T Q_r(u)c(theta)``.  This module gives the *causal* mask,
not just the rectangle's Jacobian: an original donor support, an original
receiving quarter, and the earliest positive owner entry are all required.
Thus an exit followed by reentry does not allocate a second material point.

The area formula for an injective first-event map identifies the sum of its
masked ``R^2 |J| dtheta du`` integrals with the authenticated whole swept
domain measure.  It does not numerically split that measure between receiving
quarters.  The original one-dimensional interface measure ``C dtheta_comp``
and the literal old quadrature atoms remain separate measures; neither is
silently turned into a swept material energy or work density.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction as F
import hashlib
import json

from .geometric_native_pose import FractionInterval as I, interval_pi, _rounded_interval
from .geometric_native_physical_history import (
    NativePhysicalHistoryBinding, validate_native_physical_history_binding,
    native_physical_history_binding_to_dict, _dyadic, _literal_json,
)
from .geometric_native_first_entry_material import (
    validate_native_first_entry_material, validate_native_first_entry_material_point,
    _first_receivers,
)
from .geometric_native_first_entry_contact_measure import (
    NativeFirstEntryContactMeasure,
    certify_native_first_entry_contact_measure,
    native_first_entry_contact_measure_to_dict,
)
from .geometric_native_contact_reference_transport import _proof_key
from .geometric_native_physical_pair_time import _axes, _rotated


FORMAT = "native-whole-first-entry-interface-coarea-1"
MODEL = "causal_first_event_carrier_masks_authentic_whole_domain_coarea"
SCOPE = "whole_material_coarea_identity_and_separate_original_interface_measures"
_DERIVATIONS = {}
_AUTHENTICATED_MASKS = set()


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _fraction(value):
    return [value.numerator, value.denominator]


def _ib(value):
    return dict(lo=_fraction(value.lo), hi=_fraction(value.hi))


def _vector(values):
    return tuple(_dyadic(value) for value in values)


def _dot(first, second):
    return sum((a*b for a, b in zip(first, second)), F(0))


def _cross(first, second):
    return (first[1]*second[2]-first[2]*second[1],
            first[2]*second[0]-first[0]*second[2],
            first[0]*second[1]-first[1]*second[0])


def _source(history, expected):
    if type(history) is not NativePhysicalHistoryBinding:
        raise ValueError("Whole carrier integration requires the complete typed physical history")
    history = validate_native_physical_history_binding(history)
    if type(expected) is not dict:
        raise ValueError("Whole carrier integration requires an independent full source DTO")
    _literal_json(expected)
    payload = _encoded(expected)
    if payload != _encoded(native_physical_history_binding_to_dict(history)):
        raise ValueError("Whole carrier integration source differs from the complete original replay")
    validate_native_first_entry_material(history.material_state)
    return payload


@dataclass(frozen=True, slots=True)
class NativeFirstEntryInterfaceCarrierMask:
    original_donor_fragment_id: str
    original_receiver_fragment_id: str
    original_donor_trace_id: str
    original_receiver_trace_id: str
    archive_domain_id: str
    domain_case: str
    source_supports: tuple
    receiver_supports: tuple
    measure_kernel: object
    history: NativePhysicalHistoryBinding

    @property
    def material_state(self):
        return self.history.material_state

    @property
    def causal_predicate(self):
        return (
            "q(theta,u) in original_donor; c(theta) in original_receiver_quarter; "
            "owner_depth(q,u)=0; owner_depth(q,v)<=0 for every 0<=v<u; "
            "owner_depth has a positive right germ at u"
        )

    @property
    def first_entry_only(self):
        return True

    @property
    def reentry_allocates_again(self):
        return False

    @property
    def receiving_pair_area_numerically_enclosed(self):
        return False


@dataclass(frozen=True, slots=True)
class NativeFirstEntryInterfaceParentMeasure:
    original_parent_id: str
    owner: int
    archive_domain_id: str
    measure_reference: str
    archive_solid_angle_bounds_sr: I
    archive_surface_area_bounds_m2: I
    live_solid_angle_bounds_sr: I
    parent_solid_angle_bounds_sr: I

    @property
    def archive_measure_expression(self):
        return "native_area("+self.archive_domain_id+")"

    @property
    def live_measure_expression(self):
        return "native_area("+self.original_parent_id+") - "+self.archive_measure_expression


@dataclass(frozen=True, slots=True)
class NativeFirstEntryOriginalInterfaceMeasure:
    trace_id: str
    computational_extent_radians: F
    area_density_m2_per_radian: F
    whole_interface_area_m2: F
    quadrature_atoms: tuple

    @property
    def quadrature_area_total_m2(self):
        return sum((area for _, area in self.quadrature_atoms), F(0))


@dataclass(frozen=True, slots=True)
class NativeFirstEntryInterfaceIntegral:
    history: NativePhysicalHistoryBinding
    precision: int
    order: int
    radius_m: F
    carrier_masks: tuple
    parent_measures: tuple
    original_interfaces: tuple
    owner_boundary_regular_axis_projection_bounds: I
    total_archive_solid_angle_bounds_sr: I
    total_archive_surface_area_bounds_m2: I
    source_history_payload_json: bytes
    integral_sha256: str

    @property
    def once_only_material_measure_certified(self):
        return True

    @property
    def jacobian_critical_set_zero_material_measure_proved(self):
        return True

    @property
    def receiver_pair_integrals_numerically_enclosed(self):
        return False

    @property
    def weighted_contact_energy_integrals_evaluated(self):
        return False

    @property
    def contact_interface_to_swept_material_density_declared(self):
        return False

    @property
    def source_nodal_work_reassigned(self):
        return False

    @property
    def physical_pair_admission_authorized(self):
        return False

    @property
    def candidate_mechanics_adopted(self):
        return False


def _quarters(history):
    """Authenticate a complete, nonduplicated axis-quarter owner boundary."""
    state = history.material_state
    fragments = {item.fragment_id: item for item in state.journal.source_surface.fragments}
    traces = {item.trace_id: item for item in history.initial_mechanics.fault_traces.traces}
    rows = state.trace_bindings
    if len(rows) != 4 or len(traces) != 4:
        raise ValueError("Whole carrier integration requires all four original physical quarters")
    donor_ids, receiver_ids, labels = set(), set(), set()
    authenticated = []
    for row in rows:
        trace = traces.get(row.trace_id)
        donor = fragments.get(row.donor_fragment_id)
        receiver = fragments.get(row.receiver_fragment_id)
        if (trace is None or donor is None or receiver is None
                or donor.parcel.plate != state.sweep_certificate.donor_owner
                or receiver.parcel.plate != state.sweep_certificate.receiver_owner
                or (trace.subducting_plate, trace.overriding_plate)
                != (row.subducting_plate, row.overriding_plate)):
            raise ValueError("Whole carrier banks differ from the actual source physical direction")
        start, end = _vector(trace.start), _vector(trace.end)
        if (any(sum(value != 0 for value in vector) != 1
                    or _dot(vector, vector) != 1 or vector[0] != 0
                    for vector in (start, end))
                or _dot(start, end) != 0 or _cross(start, end) != (F(-1), F(0), F(0))):
            raise ValueError("Original quarters must cover the oriented exact axis owner boundary")
        donor_supports = tuple(plane.normal for plane in donor.geometry.supports)
        receiver_supports = tuple(plane.normal for plane in receiver.geometry.supports)
        if (any(_dot(normal, vector) < 0 for normal in donor_supports
                for vector in (start, end))
                or any(_dot(normal, vector) < 0 for normal in receiver_supports
                       for vector in (start, end))):
            raise ValueError("Original contact quarter lies outside one of its actual source banks")
        y, z = start[1]+end[1], start[2]+end[2]
        label = (int(y), int(z))
        if (label in labels or row.donor_fragment_id in donor_ids
                or row.receiver_fragment_id in receiver_ids):
            raise ValueError("Whole carrier boundary duplicates an original bank or quarter")
        labels.add(label)
        donor_ids.add(row.donor_fragment_id)
        receiver_ids.add(row.receiver_fragment_id)
        authenticated.append((row, trace, donor_supports, receiver_supports))
    expected_donors = {item.fragment_id for item in state.parent_allocations
                       if item.owner == state.sweep_certificate.donor_owner}
    expected_receivers = {item.fragment_id for item in state.parent_allocations
                          if item.owner == state.sweep_certificate.receiver_owner}
    if (labels != {(-1, -1), (-1, 1), (1, -1), (1, 1)}
            or donor_ids != expected_donors or receiver_ids != expected_receivers):
        raise ValueError("Whole carrier map omits an original donor or receiving quarter")
    return tuple(authenticated)


def certify_native_first_entry_interface_integral(history, *,
        expected_source_history_dto, precision=256, order=32):
    """Identify whole first-entry material area by the causal coarea formula.

    This constructor checks the complete source, all original banks and the
    quarter partition.  The original domain proof supplies the exact archive
    measure symbols and their enclosures.  It does not evaluate individual
    receiving-pair or weighted-energy integrals by a numerical rectangle rule.
    """
    payload = _source(history, expected_source_history_dto)
    if type(precision) is not int or not 96 <= precision <= 1024:
        raise ValueError("Whole carrier integration requires an explicit bounded precision")
    if type(order) is not int or not 16 <= order <= 48:
        raise ValueError("Whole carrier integration requires an explicit bounded Taylor order")
    key = _proof_key((history, precision, order, payload))
    if key in _DERIVATIONS:
        return _DERIVATIONS[key]
    state = history.material_state
    quarters = _quarters(history)
    radius = _dyadic(history.origin_surface.radius_km)*1000
    donor_axis, receiver_axis = _axes(history.omega_records, history.dt_myr,
        state.sweep_certificate.donor_owner, state.sweep_certificate.receiver_owner)
    relative_axis = tuple(a-b for a, b in zip(donor_axis, receiver_axis))
    receiver_z = _rotated((F(0), F(0), F(1)), receiver_axis, I(0, 1), precision, order)
    projection = _rounded_interval(_dot(relative_axis, receiver_z), precision+16)
    if projection.lo <= 0 <= projection.hi:
        raise ValueError("Whole owner-boundary Jacobian requires a proved nonzero projected relative axis")
    native_quarter = I(0, (interval_pi(precision+24)/2).hi)
    domains = {item.fragment_id: item for item in state.sweep_certificate.donor_domains}
    allocations = {item.fragment_id: item for item in state.parent_allocations}
    memory = {item.trace_id: item for item in history.previous_memory.traces}
    interfaces, kernels = [], {}
    for row, trace, _, _ in quarters:
        nodes = tuple(item for item in history.quadrature_histories if item.trace_id == row.trace_id)
        if (not nodes or any(item.original_frame != nodes[0].original_frame for item in nodes)
                or memory.get(row.trace_id) is None):
            raise ValueError("Original interface has inconsistent saved frame or continuous extent")
        sine = _dyadic(nodes[0].original_frame[0])
        extent = _dyadic(memory[row.trace_id].extent_radians)
        if sine <= 0 or extent <= 0:
            raise ValueError("Original interface has nonpositive literal dimensional measure")
        density = radius*_dyadic(trace.contact_height_m)/sine
        interfaces.append(NativeFirstEntryOriginalInterfaceMeasure(row.trace_id,
            extent, density, density*extent,
            tuple((item.point_id, _dyadic(item.original_area_m2)) for item in nodes)))
        kernels[row.trace_id] = certify_native_first_entry_contact_measure(
            row.trace_id, _vector(trace.start), _vector(trace.end), row.subducting_plate,
            row.overriding_plate, history.omega_records, history.dt_myr,
            native_quarter, I(0, 1), extent, density, precision=precision, order=order)
    masks = []
    for donor_row, _, donor_supports, _ in quarters:
        allocation = allocations[donor_row.donor_fragment_id]
        domain = domains[donor_row.donor_fragment_id]
        if domain.swept_area_bounds_sr != allocation.archived_area_bounds_sr:
            raise ValueError("Carrier archive measure differs from the authenticated full donor domain")
        for receiver_row, _, _, receiver_supports in quarters:
            masks.append(NativeFirstEntryInterfaceCarrierMask(donor_row.donor_fragment_id,
                receiver_row.receiver_fragment_id, donor_row.trace_id, receiver_row.trace_id,
                allocation.archive_domain_id, domain.domain_case, donor_supports,
                receiver_supports, kernels[receiver_row.trace_id], history))
    parents = tuple(NativeFirstEntryInterfaceParentMeasure(item.fragment_id, item.owner,
        item.archive_domain_id, item.measure_reference, item.archived_area_bounds_sr,
        item.archived_area_bounds_sr*(radius*radius),
        I(max(F(0), item.parent_area_bounds_sr.lo-item.archived_area_bounds_sr.hi),
          item.parent_area_bounds_sr.hi-item.archived_area_bounds_sr.lo),
        item.parent_area_bounds_sr) for item in state.parent_allocations)
    # Keep the whole authenticated domain's enclosure rather than replacing
    # its exact symbol with independently rounded sums or endpoint overlaps.
    total = state.sweep_certificate.swept_area_bounds_sr
    result = NativeFirstEntryInterfaceIntegral(history, precision, order, radius,
        tuple(masks), parents, tuple(interfaces), projection,
        total, total*(radius*radius), payload, "")
    result = replace(result, integral_sha256=hashlib.sha256(_encoded(_body(result))).hexdigest())
    if len(_DERIVATIONS) >= 16:
        _DERIVATIONS.clear()
    _DERIVATIONS[key] = result
    if len(_AUTHENTICATED_MASKS) >= 1024:
        _AUTHENTICATED_MASKS.clear()
    _AUTHENTICATED_MASKS.update(_proof_key(mask) for mask in result.carrier_masks)
    return result


def evaluate_native_first_entry_interface_carrier_mask(mask, chronology):
    """Evaluate a whole mask on an already proved original point chronology.

    No event midpoint or new point certificate is constructed.  Ambiguous
    receiving ancestry remains ``None``; seams can belong to more than one
    closed quarter and carry zero two-dimensional material measure.
    """
    if (type(mask) is not NativeFirstEntryInterfaceCarrierMask
            or type(mask.measure_kernel) is not NativeFirstEntryContactMeasure):
        raise ValueError("Expected an immutable whole first-entry carrier mask")
    # A cache key fingerprints every field with its literal type, including
    # the whole physical source. No supplied checksum can grant admission.
    mask_key = _proof_key(mask)
    if mask_key not in _AUTHENTICATED_MASKS:
        replay = certify_native_first_entry_interface_integral(mask.history,
            expected_source_history_dto=native_physical_history_binding_to_dict(mask.history),
            precision=mask.measure_kernel.precision, order=mask.measure_kernel.order)
        expected = next((item for item in replay.carrier_masks
            if (item.original_donor_fragment_id, item.original_receiver_fragment_id)
            == (mask.original_donor_fragment_id, mask.original_receiver_fragment_id)), None)
        if expected is None or mask_key != _proof_key(expected):
            raise ValueError("Carrier mask differs from its complete physical source and whole-domain replay")
        _AUTHENTICATED_MASKS.add(mask_key)
    chronology = validate_native_first_entry_material_point(chronology)
    if (mask.material_state is not chronology.state
            and _proof_key(mask.material_state) != _proof_key(chronology.state)):
        raise ValueError("Carrier mask point belongs to a different complete material source")
    if chronology.original_parent_id != mask.original_donor_fragment_id:
        return False
    certificate = chronology.first_entry_certificate
    if certificate is None or certificate.first_entry_time_fraction is None:
        return False
    receivers, status = _first_receivers(chronology)
    if mask.original_receiver_fragment_id not in receivers:
        return False
    return True if status == "proved" else None


def _body(result):
    # Sixteen masks reference four kernels. Authenticate and serialize each
    # complete typed kernel once during this body construction.
    measure_bodies = {}

    def measure_body(kernel):
        key = _proof_key(kernel)
        if key not in measure_bodies:
            measure_bodies[key] = native_first_entry_contact_measure_to_dict(kernel)
        return measure_bodies[key]

    return dict(format=FORMAT, model=MODEL, scope=SCOPE,
        source_history=json.loads(result.source_history_payload_json),
        precision=result.precision, order=result.order, radius_m=_fraction(result.radius_m),
        carrier_map="q(theta,u)=Q_d(u)^T Q_r(u)*(S*cos(theta)+E*sin(theta))",
        material_coarea_integrand="R^2*abs((a_d-a_r).Q_r(u)*c_theta(theta))*causal_mask(theta,u)",
        owner_depth="-C(v)*q_x+Y(v)*q_y+Z(v)*q_z",
        causal_mask_contract="earliest_positive_owner_entry_with_complete_original_supports",
        coarea_proof=dict(first_event_map_injective=True,
            reason="a fixed original unit material direction has one earliest positive entry; each open receiver quarter has one carrier angle",
            homogeneous_labels="positive rescalings denote the same original spherical point; old literal Gauss labels are not normalized or changed",
            original_receiver_boundary_complete=True,
            carrier_map_regular="analytic full SO3 map is C1 on each complete closed parameter rectangle",
            owner_boundary_regular_axis_projection_bounds=_ib(result.owner_boundary_regular_axis_projection_bounds),
            jacobian_critical_set_zero_material_measure_proved=True,
            critical_set_reason="a nonzero projected relative axis makes J(theta,u) a nonzero sinusoid at every u; finitely many theta roots have zero two-dimensional parameter measure",
            quarter_seams="retained closed ancestry; zero two-dimensional material area",
            original_parent_internal_boundaries="all closed ancestry retained; zero original two-dimensional material area",
            tangencies="no positive right germ means no first entry; zero Jacobian contributes zero area",
            initial_contacts="original owner-boundary material has zero two-dimensional area",
            exact_identity="sum_receivers integral(causal_mask*R^2*abs(J), dtheta du) = original_archive_measure",
            measure_authority="authenticated complete first-entry domain certificate, including curved swept-only excess",
            endpoint_polygon_substitution=False, root_midpoint_used=False,
            rectangle_multiplicity_used_as_archive_area=False),
        carrier_masks=[dict(original_donor_fragment_id=item.original_donor_fragment_id,
            original_receiver_fragment_id=item.original_receiver_fragment_id,
            original_donor_trace_id=item.original_donor_trace_id,
            original_receiver_trace_id=item.original_receiver_trace_id,
            archive_domain_id=item.archive_domain_id, domain_case=item.domain_case,
            source_supports=[[_fraction(x) for x in vector] for vector in item.source_supports],
            receiver_supports=[[_fraction(x) for x in vector] for vector in item.receiver_supports],
            causal_predicate=item.causal_predicate, first_entry_only=True,
            reentry_allocates_again=False, receiving_pair_area_numerically_enclosed=False,
            measure_kernel=measure_body(item.measure_kernel))
            for item in result.carrier_masks],
        parent_measures=[dict(original_parent_id=item.original_parent_id, owner=item.owner,
            archive_domain_id=item.archive_domain_id, measure_reference=item.measure_reference,
            archive_measure_expression=item.archive_measure_expression,
            live_measure_expression=item.live_measure_expression,
            archive_solid_angle_bounds_sr=_ib(item.archive_solid_angle_bounds_sr),
            archive_surface_area_bounds_m2=_ib(item.archive_surface_area_bounds_m2),
            live_solid_angle_bounds_sr=_ib(item.live_solid_angle_bounds_sr),
            parent_solid_angle_bounds_sr=_ib(item.parent_solid_angle_bounds_sr),
            exact_identity="live_measure+archive_measure=original_parent_measure")
            for item in result.parent_measures],
        original_interfaces=[dict(trace_id=item.trace_id,
            computational_extent_radians=_fraction(item.computational_extent_radians),
            area_density_m2_per_radian=_fraction(item.area_density_m2_per_radian),
            whole_interface_area_m2=_fraction(item.whole_interface_area_m2),
            quadrature_atoms=[dict(point_id=point, literal_area_m2=_fraction(area))
                              for point, area in item.quadrature_atoms],
            continuum_measure_equals_literal_quadrature_total_asserted=False,
            source_family_owned_once=True)
            for item in result.original_interfaces],
        total_archive_solid_angle_bounds_sr=_ib(result.total_archive_solid_angle_bounds_sr),
        total_archive_surface_area_bounds_m2=_ib(result.total_archive_surface_area_bounds_m2),
        once_only_material_measure_certified=True,
        receiver_pair_integrals_numerically_enclosed=False,
        weighted_contact_energy_integrals_evaluated=False,
        contact_interface_to_swept_material_density_declared=False,
        source_nodal_work_reassigned=False, physical_pair_admission_authorized=False,
        candidate_mechanics_adopted=False)


def validate_native_first_entry_interface_integral(result, *, expected_source_history_dto):
    if type(result) is not NativeFirstEntryInterfaceIntegral:
        raise ValueError("Expected the immutable whole first-entry interface integral")
    replay = certify_native_first_entry_interface_integral(result.history,
        expected_source_history_dto=expected_source_history_dto,
        precision=result.precision, order=result.order)
    if (_proof_key(result) != _proof_key(replay)
            or _encoded(_body(result)) != _encoded(_body(replay))):
        raise ValueError("Whole first-entry interface integral differs from the full typed replay")
    return replay


def native_first_entry_interface_integral_to_dict(result, *, expected_source_history_dto=None):
    if type(result) is not NativeFirstEntryInterfaceIntegral:
        raise ValueError("Expected the immutable whole first-entry interface integral")
    expected = (json.loads(result.source_history_payload_json)
                if expected_source_history_dto is None else expected_source_history_dto)
    result = validate_native_first_entry_interface_integral(result,
        expected_source_history_dto=expected)
    return dict(_body(result), integral_sha256=result.integral_sha256)


def native_first_entry_interface_integral_from_dict(raw, *, expected_history,
        expected_source_history_dto):
    if type(raw) is not dict or raw.get("format") != FORMAT or raw.get("model") != MODEL:
        raise ValueError("Invalid whole first-entry interface integral schema")
    _literal_json(raw)
    try:
        replay = certify_native_first_entry_interface_integral(expected_history,
            expected_source_history_dto=expected_source_history_dto,
            precision=raw["precision"], order=raw["order"])
        expected = native_first_entry_interface_integral_to_dict(replay,
            expected_source_history_dto=expected_source_history_dto)
        if _encoded(raw) != _encoded(expected):
            raise ValueError("Saved whole first-entry interface integral differs from full source and analytic replay")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Invalid whole first-entry interface integral fields") from error
    return replay
