"""Complete original banks and live clamps for a temporal material closure.

The archive is the complete positive first-entry set, including its curved
swept-only part.  Its complement supplies live original material.  Neither set
is replaced by an endpoint polygon, and a shared measure preserves each of the
four original reservoirs.  An exact exponential majorant proves that every
point of each donor clamp enclosure stays outside the archive for all time.

This is a support certificate on replay-authenticated, UNACCEPTED physics.  It
does not establish contact reactions, spatial admission, or force equilibrium.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction as F
import hashlib
import json

from .fractional_surface import EXTENSIVE_FIELDS
from .geometric_native_buried_support import _quarter_clamp_envelope, _subtract_all
from .geometric_native_first_entry_material import validate_native_first_entry_material
from .geometric_native_material_history import _fragment_body
from .geometric_native_physical_history import (
    _literal_json, _typed_fingerprint, native_physical_history_binding_to_dict,
    validate_native_physical_history_binding,
)
from .geometric_native_polygons import native_polygon_to_dict
from .geometric_native_pose import FractionInterval as I


FORMAT = "native-implicit-original-bank-live-clamp-support-1"
MODEL = "complete_temporal_first_entry_partition_with_exact_live_original_clamps"
_AUTHENTICATED = {}


class UnresolvedNativeImplicitForceSupport(ValueError):
    """A complete original bank or an entire live clamp was not proved."""


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()


def _hash(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _r(value):
    if type(value) is not F:
        raise ValueError("Implicit support quantities must retain exact rational types")
    return [value.numerator, value.denominator]


def _i(value):
    if type(value) is not I:
        raise ValueError("Implicit support measures require exact native intervals")
    return dict(lo=_r(value.lo), hi=_r(value.hi))


def _normal_majorant(omega_records, duration):
    """Bound every component of Qd(u)^T Qr(u)n - n, for all 0<=u<=1.

For a skew generator A, ||A||_infinity <= ||axis||_1.  The product of
both exponential series is therefore bounded by exp(rho); after removing
the identity, exp(rho)-1 <= rho/(1-rho).  Every literal axis component is
included.  This is a sufficient enclosure, not a projected rotation axis.
"""
    if (type(duration) is not F or duration <= 0 or type(omega_records) is not tuple
            or len(omega_records) != 2
            or any(type(row) is not tuple or len(row) != 2
                or type(row[0]) is not int or type(row[1]) is not tuple
                or len(row[1]) != 3 or any(type(x) is not F for x in row[1])
                for row in omega_records)
            or len({owner for owner, _ in omega_records}) != 2):
        raise ValueError("Clamp proof requires both complete exact original axes and duration")
    rho = duration*sum((abs(x) for _, omega in omega_records for x in omega), F(0))
    if not 0 < rho < F(1, 4):
        raise UnresolvedNativeImplicitForceSupport("The full relative path exceeds the bounded clamp proof")
    return rho, rho/(1-rho)


def _entry_depth_upper(point, normal_component_error):
    """Homogeneous upper depth in the original overriding X<0 hemisphere."""
    if (type(point) is not tuple or len(point) != 3
            or any(type(x) not in (int, F) for x in point)
            or type(normal_component_error) is not F or normal_component_error < 0):
        raise ValueError("Clamp depth requires an exact homogeneous ray and nonnegative bound")
    return -F(point[0])+normal_component_error*sum((abs(F(x)) for x in point), F(0))


@dataclass(frozen=True, slots=True)
class NativeImplicitOriginalBankSupport:
    original_fragment: object
    parent_allocation: object
    archive_case: str
    source_payload_sha256: str


@dataclass(frozen=True, slots=True)
class NativeImplicitLiveClampSupport:
    trace_id: str
    bank: str
    owner: int
    width_m: F
    exact_width_radians: F
    enclosing_geometry: object
    original_live_parent_ids: tuple
    vertex_entry_depth_upper_bounds: tuple
    proof: str


@dataclass(frozen=True, slots=True)
class NativeImplicitForceSupport:
    history: object
    source_history_payload_json: bytes
    source_history_sha256: str
    material_state_sha256: str
    original_checkpoint_sha256: str
    source_surface_sha256: str
    trace_sha256: str
    bending_source_sha256: str
    radius_m: F
    time_myr: I
    omega_records: tuple
    rotation_path_l1_bound: F
    normal_component_error_bound: F
    parent_supports: tuple[NativeImplicitOriginalBankSupport, ...]
    live_clamps: tuple[NativeImplicitLiveClampSupport, ...]
    support_sha256: str

    @property
    def complete_original_bank_support_certified(self):
        return True

    @property
    def live_original_clamp_support_certified(self):
        return True

    @property
    def physical_pair_admission_authorized(self):
        return False

    @property
    def candidate_mechanics_adopted(self):
        return False

    @property
    def native_force_intervals_accepted(self):
        return 0


def _bank_body(bank):
    fragment, allocation = bank.original_fragment, bank.parent_allocation
    return dict(fragment_id=fragment.fragment_id, material_id=fragment.parcel.material_id,
        owner=fragment.parcel.plate, source_payload_sha256=bank.source_payload_sha256,
        original_geometry=native_polygon_to_dict(fragment.geometry),
        archive_domain_id=allocation.archive_domain_id, live_domain_id=allocation.live_domain_id,
        archive_case=bank.archive_case, measure_reference=allocation.measure_reference,
        original_area_bounds_sr=_i(allocation.parent_area_bounds_sr),
        archived_area_bounds_sr=_i(allocation.archived_area_bounds_sr),
        archive_ratio_bounds=_i(allocation.archive_ratio_bounds),
        original_age_myr=_r(allocation.original_age_myr),
        endpoint_age_myr=_r(allocation.material_age_at_endpoint_myr),
        cohort_id=allocation.cohort_id,
        reservoirs=[dict(reservoir=row.reservoir, parent_amount=_r(row.parent_amount),
            archived_amount_bounds=_i(row.archived_amount_bounds),
            live_amount_bounds=_i(row.live_amount_bounds),
            archived_expression=row.archived_expression, live_expression=row.live_expression,
            measure_reference=row.measure_reference, conservation_identity=row.conservation_identity)
            for row in allocation.reservoirs],
        coverage_identity="original_parent = live_original disjoint_union once_only_archive",
        archive_mask=("empty: original receiver owner excluded from selected donor mask"
            if bank.archive_case == "unarchived_receiver" else
            "original_parent and exists 0<u<=1: (-C(u)*x+Y(u)*y+Z(u)*z)>0"),
        live_mask="original_parent minus once_only_archive; boundary remains live",
        curved_archive_replaced_by_polygon=False, newborn_material_supplies_original_bank=False)


def _clamp_body(clamp):
    return dict(trace_id=clamp.trace_id, bank=clamp.bank, owner=clamp.owner,
        width_m=_r(clamp.width_m), exact_width_radians=_r(clamp.exact_width_radians),
        enclosing_geometry=native_polygon_to_dict(clamp.enclosing_geometry),
        original_live_parent_ids=list(clamp.original_live_parent_ids),
        vertex_entry_depth_upper_bounds=[_r(x) for x in clamp.vertex_entry_depth_upper_bounds],
        proof=clamp.proof, time_fraction=dict(lo=[0, 1], hi=[1, 1]),
        archived_material_supplies_clamp=False, newborn_material_supplies_clamp=False,
        continuum_coverage="positive conic combinations of every exact enclosure vertex")


def _body(value):
    return dict(format=FORMAT, model=MODEL,
        source_history=json.loads(value.source_history_payload_json),
        source_history_sha256=value.source_history_sha256,
        material_state_sha256=value.material_state_sha256,
        original_checkpoint_sha256=value.original_checkpoint_sha256,
        source_surface_sha256=value.source_surface_sha256,
        trace_sha256=value.trace_sha256, bending_source_sha256=value.bending_source_sha256,
        radius_m=_r(value.radius_m), time_myr=_i(value.time_myr),
        omega_records=[[owner, [_r(x) for x in axis]] for owner, axis in value.omega_records],
        rotation_path_l1_bound=_r(value.rotation_path_l1_bound),
        normal_component_error_bound=_r(value.normal_component_error_bound),
        normal_majorant="||Qd(u)^T Qr(u)n-n||_infinity <= exp(rho)-1 <= rho/(1-rho)",
        parent_supports=[_bank_body(bank) for bank in value.parent_supports],
        live_clamps=[_clamp_body(clamp) for clamp in value.live_clamps],
        complete_original_bank_support_certified=True, live_original_clamp_support_certified=True,
        physical_pair_admission_authorized=False, candidate_mechanics_adopted=False,
        accepted_loaded_native_intervals=0, accepted_source_work_j=0,
        original_work_ledger_preserved=True,
        scope="original eight-parent two-hemisphere epoch; support only; temporal admission and reactions required")


def certify_native_implicit_force_support(history, *, expected_source_history_dto):
    """Prove whole original bank partition and LIVE continuous original clamps."""
    history = validate_native_physical_history_binding(history)
    _literal_json(expected_source_history_dto)
    source = native_physical_history_binding_to_dict(history)
    if type(expected_source_history_dto) is not dict or _encoded(source) != _encoded(expected_source_history_dto):
        raise ValueError("Implicit support requires the independent complete actual source history")
    material = validate_native_first_entry_material(history.material_state)
    journal, domains, seed = material.journal, material.sweep_certificate, history.initial_mechanics
    parents = journal.source_surface.fragments
    allocations = {item.fragment_id: item for item in material.parent_allocations}
    donor_domains = {item.fragment_id: item for item in domains.donor_domains}
    traces, parameters = seed.fault_traces.traces, {item.trace_id: item for item in seed.bending_parameters}
    if (len(parents) != 8 or len(allocations) != 8 or len(traces) != 4
            or len(parameters) != 4 or len(material.trace_bindings) != 4
            or set(allocations) != {parent.fragment_id for parent in parents}
            or set(parameters) != {trace.trace_id for trace in traces}
            or domains.donor_owner == domains.receiver_owner
            or set(owner for owner, _ in history.omega_records)
                != {domains.donor_owner, domains.receiver_owner}):
        raise UnresolvedNativeImplicitForceSupport("Implicit support requires every original parent and physical quarter")
    bindings = {item.trace_id: item for item in material.trace_bindings}
    if any((trace.subducting_plate, trace.overriding_plate)
            != (domains.donor_owner, domains.receiver_owner) for trace in traces):
        raise UnresolvedNativeImplicitForceSupport("Archive direction and original physical bank owners disagree")
    rho, error = _normal_majorant(history.omega_records, history.dt_myr)
    banks = []
    for parent in parents:
        allocation = allocations[parent.fragment_id]
        if (allocation.parent_payload_sha256 != _hash(_fragment_body(parent))
                or allocation.material_id != parent.parcel.material_id
                or allocation.owner != parent.parcel.plate
                or tuple(row.reservoir for row in allocation.reservoirs) != EXTENSIVE_FIELDS
                or any(row.parent_amount != F.from_float(getattr(parent.parcel, row.reservoir))
                    for row in allocation.reservoirs)):
            raise ValueError("Implicit original bank changed its actual parent payload or reservoir")
        domain = donor_domains.get(parent.fragment_id)
        case = "unarchived_receiver" if domain is None else domain.domain_case
        if domain is None and allocation.archived_area_bounds_sr != I(0):
            raise ValueError("An unselected original owner was placed into the archive")
        banks.append(NativeImplicitOriginalBankSupport(parent, allocation, case, allocation.parent_payload_sha256))
    clamps = []
    radius = F.from_float(history.origin_surface.radius_km)*1000
    for trace in traces:
        binding = bindings[trace.trace_id]
        for bank, owner, properties, sign, parent_id in (
                ("sub", trace.subducting_plate, parameters[trace.trace_id].sub, -1, binding.donor_fragment_id),
                ("over", trace.overriding_plate, parameters[trace.trace_id].over, 1, binding.receiver_fragment_id)):
            envelope, width = _quarter_clamp_envelope(trace, properties.width_m,
                history.origin_surface.radius_km, sign, bank)
            original_cover = tuple(parent.geometry for parent in parents if parent.parcel.plate == owner)
            if _subtract_all(envelope, original_cover):
                raise UnresolvedNativeImplicitForceSupport("Original clamp enclosure crosses a foreign bank")
            bound = tuple(_entry_depth_upper(ray.reference, error) for ray in envelope.vertices) if bank == "sub" else ()
            if bank == "sub" and any(value >= 0 for value in bound):
                raise UnresolvedNativeImplicitForceSupport("The entire donor clamp was not proved live for all time")
            proof = ("every vertex depth upper is strictly negative; exponential majorant and convex cone cover all u"
                if bank == "sub" else "selected archive excludes receiver owner; original enclosure remains wholly live")
            # Whole envelopes can straddle a quarter seam; every covering
            # original parent is retained rather than resolving ancestry by ID.
            covers = tuple(parent.fragment_id for parent in parents if parent.parcel.plate == owner
                and parent.fragment_id in allocations)
            if parent_id not in covers:
                raise ValueError("A physical clamp lost its original material bank ancestry")
            clamps.append(NativeImplicitLiveClampSupport(trace.trace_id, bank, owner,
                F.from_float(properties.width_m), width, envelope, covers, bound, proof))
    start = F.from_float(journal.source_surface.time_myr)
    value = NativeImplicitForceSupport(history, _encoded(source), _hash(source), material.state_sha256,
        history.original_checkpoint_sha256, journal.source_surface_sha256, seed.fault_traces.trace_sha256,
        seed.bending_source_sha256, radius, I(start, start+history.dt_myr), history.omega_records,
        rho, error, tuple(banks), tuple(clamps), "")
    value = replace(value, support_sha256=_hash(_body(value)))
    if len(_AUTHENTICATED) >= 16:
        _AUTHENTICATED.clear()
    _AUTHENTICATED[_typed_fingerprint(value)] = value.support_sha256
    return value


def validate_native_implicit_force_support(value):
    if type(value) is not NativeImplicitForceSupport:
        raise ValueError("Expected an immutable complete implicit force-support certificate")
    fingerprint = _typed_fingerprint(value)
    if _AUTHENTICATED.get(fingerprint) != value.support_sha256:
        replay = certify_native_implicit_force_support(value.history,
            expected_source_history_dto=json.loads(value.source_history_payload_json))
        if _typed_fingerprint(replay) != fingerprint:
            raise ValueError("Implicit force support differs from full original material and clamp replay")
    return value


def native_implicit_force_support_to_dict(value):
    validate_native_implicit_force_support(value)
    return dict(_body(value), support_sha256=value.support_sha256)


def native_implicit_force_support_from_dict(raw, *, expected_history, expected_source_history_dto):
    _literal_json(raw)
    if type(raw) is not dict:
        raise ValueError("Implicit force support requires a complete literal object")
    replay = certify_native_implicit_force_support(expected_history,
        expected_source_history_dto=expected_source_history_dto)
    if _encoded(raw) != _encoded(native_implicit_force_support_to_dict(replay)):
        raise ValueError("Implicit force support differs from independent full source, partition and clamp replay")
    return replay
