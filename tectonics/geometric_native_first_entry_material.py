"""A complete standalone first-entry material closure on an original epoch.

The original physical weak planes declare the donor direction.  This module
explicitly adopts a MATERIAL law: each original point enters a passive archive
once, at its first entry into the overriding owner.  It does not inherit force
admission from a seed direction or an endpoint overlap.  The force candidate,
constitutive memory and work remain immutable external evidence.

All original parents are represented.  Archive and live reservoirs share one
exact geometric measure symbol; rational intervals only enclose that measure.
The curved swept-only domain cannot be replaced by a NativePolygon.  Actual
endpoint vacancies are complements of ALL retained material, and the declared
hot closure creates separate newborn histories on those complete complements.

This format is deliberately separate from NativeSurfaceState/NativeBuriedRegistry:
those require endpoint polygon losses, which cannot express this temporal law.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace, fields, is_dataclass
from fractions import Fraction
import hashlib
import json
import math

from .fractional_surface import EXTENSIVE_FIELDS
from .geometric_intrinsic_geometry import RotationIncrement
from .geometric_native_domain_journal import (
    NativeDomainJournal, native_domain_journal_to_dict,
    native_domain_journal_from_dict, validate_native_domain_journal,
    certify_native_domain_journal_owner_point,
    native_domain_owner_point_chronology_to_dict,
    native_domain_owner_point_chronology_from_dict,
)
from .geometric_native_surface import native_surface_to_dict
from .geometric_native_material_history import _fragment_body, _memory_body
from .geometric_native_polygons import (
    NativeRay, _cross, _side, native_ray_to_dict, native_shared_boundary_arcs,
)
from .geometric_native_pose import FractionInterval as I
from .geometric_underthrust_motion import HotBirthParameters, _birth_parameters
from .thermal import SECONDS_PER_MYR


FORMAT = "native-full-first-entry-material-state-1"
MODEL = "explicit_original_seed_direction_once_only_full_first_entry_hot_endpoint"
POINT_FORMAT = "native-full-first-entry-material-point-1"
QUERY_FORMAT = "native-full-first-entry-material-point-query-1"
ENDPOINT_QUERY_FORMAT = "native-full-first-entry-material-endpoint-query-1"


# Only fully reconstructed objects enter these bounded caches.  Every lookup
# fingerprints ALL fields with their literal types; a checksum or dataclass
# equality (which aliases bool/int and float/Fraction) cannot grant admission.
_STATE_CACHE = {}
_POINT_CONTEXT_CACHE = {}
_FIRST_RECEIVER_CACHE = {}


def _typed_fingerprint(value):
    kind = type(value)
    if kind in (str, bytes, int, bool, type(None)):
        return (kind, value)
    if kind is float:
        return (kind, value.hex())
    if kind is Fraction:
        return (kind, value.numerator, value.denominator)
    if kind is tuple:
        return (kind, tuple(_typed_fingerprint(item) for item in value))
    from .geometric_native_surface import NativeSurfaceState
    if kind is NativeSurfaceState:
        return (kind, tuple((name, _typed_fingerprint(getattr(value, name)))
                           for name in NativeSurfaceState.__slots__))
    if is_dataclass(value):
        return (kind, tuple((field.name, _typed_fingerprint(getattr(value, field.name)))
                           for field in fields(value)))
    raise ValueError("Material replay cache requires completely immutable typed source fields")


def _remember(cache, value, body_bytes, limit):
    if len(cache) >= limit:
        cache.clear()
    cache[_typed_fingerprint(value)] = body_bytes


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _rational(value):
    if type(value) is not Fraction:
        raise ValueError("Material closure requires exact rational clocks and amounts")
    return [value.numerator, value.denominator]


def _read_rational(value):
    if (type(value) is not list or len(value) != 2
            or any(type(v) is not int for v in value) or value[1] <= 0):
        raise ValueError("Material closure requires canonical rational pairs")
    result = Fraction(*value)
    if _rational(result) != value:
        raise ValueError("Material closure rational pair is not reduced")
    return result


def _interval(value):
    if type(value) is not I:
        raise ValueError("Material measure requires an exact native enclosure")
    return dict(lo=_rational(value.lo), hi=_rational(value.hi))


def _text(value):
    if type(value) is not str or not value.strip():
        raise ValueError("Material closure needs explicit nonempty provenance")
    return value


def _point(value):
    if (type(value) is not tuple or len(value) != 3
            or any(type(v) not in (int, Fraction) for v in value)):
        raise ValueError("Material point requires an exact immutable Cartesian triple")
    result = tuple(Fraction(v) for v in value)
    if not any(result):
        raise ValueError("Material point must be a nonzero direction")
    return result


def _fraction_time(value):
    if type(value) not in (int, Fraction):
        raise ValueError("Material query time must be an exact Fraction or integer")
    result = Fraction(value)
    if not 0 <= result <= 1:
        raise ValueError("Material query time must lie in the original interval")
    return result


@dataclass(frozen=True, slots=True)
class NativeFirstEntryTraceBinding:
    trace_id: str
    donor_fragment_id: str
    receiver_fragment_id: str
    donor_material_id: str
    receiver_material_id: str
    subducting_plate: int
    overriding_plate: int
    trace_payload_sha256: str


@dataclass(frozen=True, slots=True)
class NativeFirstEntryReservoir:
    reservoir: str
    parent_amount: Fraction
    archived_amount_bounds: I
    live_amount_bounds: I
    archived_expression: str
    live_expression: str
    measure_reference: str
    conservation_identity: str = "live_amount + archived_amount = original_parent_amount"


@dataclass(frozen=True, slots=True)
class NativeFirstEntryParentAllocation:
    fragment_id: str
    material_id: str
    owner: int
    parent_payload_sha256: str
    archive_domain_id: str
    live_domain_id: str
    measure_reference: str
    archived_area_bounds_sr: I
    parent_area_bounds_sr: I
    archive_ratio_bounds: I
    reservoirs: tuple[NativeFirstEntryReservoir, ...]
    cohort_id: str | None
    original_age_myr: Fraction
    material_age_at_endpoint_myr: Fraction
    material_fields: tuple
    specific_properties: tuple


@dataclass(frozen=True, slots=True)
class NativeFirstEntryArchiveCohort:
    cohort_id: str
    original_parent_id: str
    material_id: str
    owner: int
    receiver_owner: int
    archive_domain_id: str
    material_time_function: str = "first positive owner entry of each original point"
    receiver_ancestry_function: str = "all original receiving fragments incident at that first entry"
    archive_law: str = "once_only_passive_archive; exit and reentry never allocate twice"


@dataclass(frozen=True, slots=True)
class NativeFirstEntryHotMaterial:
    material_id: str
    fixed_original_parent_id: str
    owner: int
    cell: int
    vacancy_domain_id: str
    birth_time_myr: Fraction
    area_km2_bounds: I
    oceanic_volume_km3_bounds: I
    cold_mantle_volume_km3: Fraction
    density_excess_mass_kg: Fraction
    age_myr: Fraction
    material_fields: tuple
    specific_properties: tuple
    area_expression: str
    oceanic_volume_expression: str


@dataclass(frozen=True, slots=True)
class NativeFirstEntryEndpointVacancy:
    fixed_original_parent_id: str
    vacancy_domain_id: str
    measure_reference: str
    area_bounds_sr: I
    raw_vacancy_ids: tuple
    curved_extra_donor_ids: tuple
    newborn: NativeFirstEntryHotMaterial | None


@dataclass(frozen=True, slots=True)
class NativeFirstEntryMaterialState:
    journal: NativeDomainJournal
    sweep_certificate: object
    original_mechanics_payload_json: bytes
    original_mechanics_sha256: str
    trace_bindings: tuple[NativeFirstEntryTraceBinding, ...]
    birth_parameters: HotBirthParameters
    provenance: str
    parent_allocations: tuple[NativeFirstEntryParentAllocation, ...]
    archive_cohorts: tuple[NativeFirstEntryArchiveCohort, ...]
    endpoint_vacancies: tuple[NativeFirstEntryEndpointVacancy, ...]
    state_sha256: str

    @property
    def material_closure_applied(self):
        return True

    @property
    def candidate_mechanics_adopted(self):
        return False

    @property
    def physical_pair_admission_authorized(self):
        return False

    @property
    def native_force_intervals_accepted(self):
        return 0


@dataclass(frozen=True, slots=True)
class NativeFirstEntryMaterialPointChronology:
    state: NativeFirstEntryMaterialState
    original_parent_id: str
    original_point: tuple
    first_entry_certificate: object | None
    receiver_chronology: object | None
    point_chronology_sha256: str


@dataclass(frozen=True, slots=True)
class NativeFirstEntryMaterialPointQuery:
    chronology: NativeFirstEntryMaterialPointChronology
    time_fraction: Fraction
    material_allocation: str
    contact_occupancy: str
    first_entry_time_fraction: I | None
    receiver_original_fragment_ids: tuple
    original_trace_ids: tuple
    receiver_trace_ids: tuple
    first_entry_receiver_ancestry_status: str
    cohort_id: str | None
    material_age_myr: Fraction
    material_id: str
    material_owner: int
    archive_debits: int
    physical_pair_ancestry_authenticated: bool = False
    finite_material_reservoir_assigned_to_point: bool = False


@dataclass(frozen=True, slots=True)
class NativeFirstEntryMaterialEndpointQuery:
    chronology: NativeFirstEntryMaterialPointChronology
    endpoint_ray: NativeRay
    point_query: NativeFirstEntryMaterialPointQuery
    raw_coverage: bool | None
    retained_material: bool | None
    actual_material_vacancy: bool | None
    fixed_original_parent_ids: tuple
    birth_material_ids: tuple
    material_age_myr: Fraction
    physical_pair_admission_authorized: bool = False
    candidate_mechanics_adopted: bool = False


def _original_direction(journal, certificate, payload):
    """Bind declared original weak planes to genuine source material banks."""
    if type(payload) is not dict:
        raise ValueError("Material closure needs the complete original mechanical seed payload")
    frozen = _encoded(payload)
    original = json.loads(frozen)
    before = json.loads(journal.memory_reference.before_payload_json)
    candidate = json.loads(journal.memory_reference.candidate_after_payload_json)
    if (candidate.get("candidate_adopted") is not False
            or type(candidate.get("candidate_sha256")) is not str
            or len(candidate["candidate_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in candidate["candidate_sha256"])
            or type(candidate.get("dt_s")) is not float
            or not math.isfinite(candidate["dt_s"]) or candidate["dt_s"] <= 0
            or not math.isclose(candidate["dt_s"], journal.dt_myr * SECONDS_PER_MYR,
                rel_tol=8 * math.ulp(1.), abs_tol=0.)
            or type(candidate.get("whole_time_support")) is not dict
            or candidate["whole_time_support"].get("path_sha256")
                != journal.memory_reference.constitutive_path_sha256):
        raise ValueError("Material closure requires the same explicitly unadopted force candidate and physical clock")
    required_bindings = {"original_mechanics_payload", "original_surface_payload",
        "original_force_checkpoint_sha256"}
    if not required_bindings.issubset(before):
        raise ValueError("Material adoption requires full original source, mechanical seed and force-checkpoint bindings")
    if (_encoded(before["original_mechanics_payload"]) != frozen
            or _encoded(before["original_surface_payload"]) != _encoded(
                native_surface_to_dict(journal.source_surface))):
        raise ValueError("Material closure changed its immutable externally bound original source or mechanics")
    checkpoint_sha = before["original_force_checkpoint_sha256"]
    if (type(checkpoint_sha) is not str or len(checkpoint_sha) != 64
            or any(c not in "0123456789abcdef" for c in checkpoint_sha)):
        raise ValueError("Original force checkpoint requires a canonical immutable SHA-256 reference")
    for name in ("elapsed_time_s", "cumulative_source_work_j", "cumulative_basal_dissipation_j",
            "cumulative_bending_numerical_dissipation_j", "preparation_work_j"):
        if name in before and name in original and _encoded(before[name]) != _encoded(original[name]):
            raise ValueError("Original source mechanical clock or work changed before material closure")
    required = {"fault_traces", "surface_sha256", "plate_ids", "states", "polarity_history",
        "elapsed_time_s", "step_count", "transverse_displacement_m", "cumulative_source_work_j"}
    if not required.issubset(original):
        raise ValueError("Original mechanical seed payload is incomplete")
    faults = original["fault_traces"]
    if (type(faults) is not dict or type(faults.get("traces")) is not list
            or type(original["plate_ids"]) is not list
            or any(type(owner) is not int for owner in original["plate_ids"])
            or len(original["plate_ids"]) != 2
            or type(original["elapsed_time_s"]) is not float
            or type(original["step_count"]) is not int
            or type(faults.get("radius_km")) is not float
            or type(faults.get("time_myr")) is not float
            or faults.get("surface_sha256") != original["surface_sha256"]
            or faults.get("radius_km") != journal.source_surface.radius_km
            or faults.get("time_myr") != journal.source_surface.time_myr
            or original["elapsed_time_s"] != 0.0 or original["step_count"] != 0
            or set(original["plate_ids"]) != {certificate.donor_owner, certificate.receiver_owner}):
        raise ValueError("Material closure requires the original prepared mechanical epoch")
    fragments = journal.source_surface.fragments
    contacts = []
    for donor in fragments:
        if donor.parcel.plate != certificate.donor_owner:
            continue
        for receiver in fragments:
            if receiver.parcel.plate != certificate.receiver_owner:
                continue
            for arc in native_shared_boundary_arcs(donor.geometry, receiver.geometry):
                if arc.start.kind != "seed" or arc.end.kind != "seed":
                    raise ValueError("Original trace direction requires literal primitive bank endpoints")
                contacts.append((donor, receiver,
                    frozenset((arc.start.reference, arc.end.reference))))
    bindings, identities, covered = [], set(), set()
    for trace in faults["traces"]:
        if (type(trace) is not dict or type(trace.get("trace_id")) is not str
                or trace["trace_id"] in identities
                or type(trace.get("subducting_plate")) is not int
                or type(trace.get("overriding_plate")) is not int
                or trace["subducting_plate"] != certificate.donor_owner
                or trace["overriding_plate"] != certificate.receiver_owner
                or type(trace.get("provenance")) is not str or not trace["provenance"].strip()
                or type(trace.get("dip_radians")) is not float
                or not 0 < trace["dip_radians"] < math.pi / 2):
            raise ValueError("Original physical weak planes must explicitly agree on the donor direction")
        endpoints = []
        for name in ("start", "end"):
            value = trace.get(name)
            if (type(value) is not list or len(value) != 3
                    or any(type(v) is not float or not math.isfinite(v) for v in value)):
                raise ValueError("Original physical trace endpoints require finite canonical floats")
            endpoints.append(tuple(Fraction.from_float(v) for v in value))
        if _cross(*endpoints) != (Fraction(-1), Fraction(0), Fraction(0)):
            raise ValueError("Original physical weak plane orientation changed its material banks")
        matches = [(donor, receiver) for donor, receiver, ends in contacts
                   if ends == frozenset(endpoints)]
        if len(matches) != 1:
            raise ValueError("Physical trace does not have exactly one original material bank pair")
        donor, receiver = matches[0]
        pair = (donor.fragment_id, receiver.fragment_id)
        if pair in covered:
            raise ValueError("Original physical pair has repeated direction evidence")
        covered.add(pair)
        identities.add(trace["trace_id"])
        bindings.append(NativeFirstEntryTraceBinding(trace["trace_id"],
            donor.fragment_id, receiver.fragment_id, donor.parcel.material_id,
            receiver.parcel.material_id, certificate.donor_owner, certificate.receiver_owner,
            _hash(trace)))
    if len(bindings) != 4 or covered != {(d.fragment_id, r.fragment_id) for d, r, _ in contacts}:
        raise ValueError("Material direction must cover all four original hemisphere weak planes")
    return frozen, tuple(sorted(bindings, key=lambda item: item.trace_id))


def _reservoir_body(item):
    return dict(reservoir=item.reservoir, parent_amount=_rational(item.parent_amount),
        archived_amount_bounds=_interval(item.archived_amount_bounds),
        live_amount_bounds=_interval(item.live_amount_bounds),
        archived_expression=item.archived_expression, live_expression=item.live_expression,
        measure_reference=item.measure_reference, conservation_identity=item.conservation_identity,
        interval_midpoint_used_as_debit=False)


def _allocation_body(item):
    return dict(fragment_id=item.fragment_id, material_id=item.material_id, owner=item.owner,
        parent_payload_sha256=item.parent_payload_sha256,
        archive_domain_id=item.archive_domain_id, live_domain_id=item.live_domain_id,
        measure_reference=item.measure_reference, archived_area_bounds_sr=_interval(item.archived_area_bounds_sr),
        parent_area_bounds_sr=_interval(item.parent_area_bounds_sr),
        archive_ratio_bounds=_interval(item.archive_ratio_bounds),
        reservoirs=[_reservoir_body(row) for row in item.reservoirs], cohort_id=item.cohort_id,
        original_age_myr=_rational(item.original_age_myr),
        material_age_at_endpoint_myr=_rational(item.material_age_at_endpoint_myr),
        material_fields=[list(row) for row in item.material_fields],
        specific_properties=list(item.specific_properties),
        archive_domain="{q in original_parent: first directed owner entry(q) <= endpoint_time}",
        live_domain="original_parent minus archive_domain",
        reservoir_allocation_time="end of the declared original interval",
        exact_partition_identity="live_domain disjoint_union archive_domain = original_parent",
        receiver_internal_boundary_allocates_new_cohort=False)


def _hot_body(item):
    if item is None:
        return None
    return dict(material_id=item.material_id, fixed_original_parent_id=item.fixed_original_parent_id,
        owner=item.owner, cell=item.cell, vacancy_domain_id=item.vacancy_domain_id,
        birth_time_myr=_rational(item.birth_time_myr),
        area_km2_bounds=_interval(item.area_km2_bounds),
        oceanic_volume_km3_bounds=_interval(item.oceanic_volume_km3_bounds),
        cold_mantle_volume_km3=_rational(item.cold_mantle_volume_km3),
        density_excess_mass_kg=_rational(item.density_excess_mass_kg), age_myr=_rational(item.age_myr),
        material_fields=[list(row) for row in item.material_fields],
        specific_properties=[_rational(v) for v in item.specific_properties],
        area_expression=item.area_expression, oceanic_volume_expression=item.oceanic_volume_expression,
        birth_law="hot_endpoint_fill_of_actual_material_vacancy",
        continuous_birth_or_newborn_motion_defined=False,
        newborn_physical_pair_admission_authorized=False)


def _vacancy_body(item):
    return dict(fixed_original_parent_id=item.fixed_original_parent_id,
        vacancy_domain_id=item.vacancy_domain_id, measure_reference=item.measure_reference,
        area_bounds_sr=_interval(item.area_bounds_sr), raw_vacancy_ids=list(item.raw_vacancy_ids),
        curved_extra_donor_ids=list(item.curved_extra_donor_ids), newborn=_hot_body(item.newborn),
        domain="fixed_original_parent minus union_of_ALL_moved_live_original_domains",
        exact_disjoint_decomposition="raw_geometric_gap disjoint_union moved_archived_swept_only_extra",
        native_polygon_proxy_used=False)


def _body(state):
    from .geometric_native_first_entry_domains import native_first_entry_domains_to_dict
    return dict(format=FORMAT, model=MODEL,
        journal=native_domain_journal_to_dict(state.journal),
        sweep_certificate=native_first_entry_domains_to_dict(state.sweep_certificate),
        original_mechanics_payload=json.loads(state.original_mechanics_payload_json),
        original_mechanics_sha256=state.original_mechanics_sha256,
        trace_bindings=[asdict(item) for item in state.trace_bindings],
        birth_parameters=asdict(state.birth_parameters), provenance=state.provenance,
        parent_allocations=[_allocation_body(item) for item in state.parent_allocations],
        archive_cohorts=[asdict(item) for item in state.archive_cohorts],
        endpoint_vacancies=[_vacancy_body(item) for item in state.endpoint_vacancies],
        memory_reference=_memory_body(state.journal.memory_reference),
        endpoint_time_myr=_rational(Fraction.from_float(state.journal.source_surface.time_myr)
            + Fraction.from_float(state.journal.dt_myr)),
        known_material_ids=sorted(set(state.journal.source_surface.known_material_ids)
            | {item.newborn.material_id for item in state.endpoint_vacancies if item.newborn is not None}),
        material_closure_applied=True,
        material_law_declared=True, original_seed_direction_used=True,
        seed_direction_is_physical_force_admission=False,
        physical_pair_admission_authorized=False, candidate_mechanics_adopted=False,
        native_force_intervals_accepted=0,
        endpoint_polygon_registry_compatible=False,
        mechanical_work_assigned_to_material_volume=False,
        domain_measure_policy="exact geometric symbols with rational enclosures; no midpoint debits",
        original_reservoir_conservation="every original live plus archive reservoir equals its parent",
        endpoint_coverage="all retained original domains disjoint_union all endpoint newborn vacancies = sphere",
        actual_material_vacancy_measure_identity="native_area(all endpoint vacancies) = native_area(all once-only archived domains)",
        vacancy_measure_identity_basis="two originally disjoint hemispheres have equal conserved 2*pi measure; receiver kept whole; raw gaps equal endpoint owner overlap; swept-only extras are disjoint from both",
        source_intensives_and_memory_preserved=True)


def initialize_native_first_entry_material(journal, sweep_certificate, *,
        original_mechanics_payload, birth_parameters, provenance):
    """Apply the complete explicit material closure, preserving refused physics."""
    from .geometric_native_first_entry_domains import validate_native_first_entry_domains
    journal = validate_native_domain_journal(journal)
    certificate = validate_native_first_entry_domains(sweep_certificate)
    if certificate.journal != journal:
        raise ValueError("First-entry geometry belongs to another complete source/physical candidate")
    parameters = _birth_parameters(birth_parameters)
    if type(parameters) is not HotBirthParameters:
        raise ValueError("Material hot birth needs the explicit canonical parameter type")
    parameters = HotBirthParameters(float(parameters.crust_thickness_km),
        float(parameters.damage), float(parameters.water), parameters.provenance)
    provenance = _text(provenance)
    mechanics, bindings = _original_direction(journal, certificate, original_mechanics_payload)
    mechanics_sha = hashlib.sha256(mechanics).hexdigest()
    context = _hash(dict(journal_sha256=journal.journal_sha256,
        original_mechanics_sha256=mechanics_sha, birth_parameters=asdict(parameters),
        provenance=provenance, model=MODEL))
    domains = {item.fragment_id: item for item in certificate.donor_domains}
    allocations, cohorts = [], []
    for parent in journal.source_surface.fragments:
        source_area = parent.geometry.area_bounds()
        domain = domains.get(parent.fragment_id)
        archived = I(0) if domain is None else domain.swept_area_bounds_sr
        if not 0 <= archived.lo <= archived.hi < source_area.lo:
            raise ValueError("Whole first-entry domain must enclose a proper original parent subset")
        ratio = archived / source_area
        archive_id = "native-first-entry-archive-domain:" + _hash(dict(
            context=context, original_parent_id=parent.fragment_id))
        live_id = "native-first-entry-live-domain:" + _hash(dict(
            archive_domain_id=archive_id, original_parent_id=parent.fragment_id))
        measure = "native_area(" + archive_id + ") / native_area(" + parent.fragment_id + ")"
        reservoirs = tuple(NativeFirstEntryReservoir(field,
            Fraction.from_float(getattr(parent.parcel, field)),
            ratio * Fraction.from_float(getattr(parent.parcel, field)),
            (I(1) - ratio) * Fraction.from_float(getattr(parent.parcel, field)),
            "original_parent_amount * " + measure,
            "original_parent_amount - archived_amount", measure)
            for field in EXTENSIVE_FIELDS)
        cohort_id = None
        if archived.hi > 0:
            if archived.lo <= 0:
                raise ValueError("A material cohort requires positive certified complete swept measure")
            cohort_id = "native-first-entry-cohort:" + _hash(dict(
                context=context, original_parent_id=parent.fragment_id, archive_domain_id=archive_id))
            cohorts.append(NativeFirstEntryArchiveCohort(cohort_id, parent.fragment_id,
                parent.parcel.material_id, parent.parcel.plate, certificate.receiver_owner, archive_id))
        allocations.append(NativeFirstEntryParentAllocation(parent.fragment_id,
            parent.parcel.material_id, parent.parcel.plate, _hash(_fragment_body(parent)),
            archive_id, live_id, measure, archived, source_area, ratio, reservoirs, cohort_id,
            Fraction.from_float(parent.parcel.age_myr),
            Fraction.from_float(parent.parcel.age_myr) + Fraction.from_float(journal.dt_myr), parent.parcel.material_fields,
            parent.parcel.specific_properties))
    vacancies = []
    endpoint = Fraction.from_float(journal.source_surface.time_myr) + Fraction.from_float(journal.dt_myr)
    radius = Fraction.from_float(journal.source_surface.radius_km)
    for parent in journal.source_surface.fragments:
        raw = tuple(v for v in journal.raw_vacancies if v.fixed_original_parent_id == parent.fragment_id)
        extra = tuple(d for d in certificate.donor_domains
            if d.extra_swept_area_bounds_sr.hi > 0
            and certificate.extra_endpoint_fixed_parent_id == parent.fragment_id)
        area = sum((v.area_bounds_sr for v in raw), I(0)) + sum(
            (d.extra_swept_area_bounds_sr for d in extra), I(0))
        vacancy_id = "native-first-entry-actual-vacancy:" + _hash(dict(
            context=context, fixed_original_parent_id=parent.fragment_id))
        measure = "native_area(" + vacancy_id + ")"
        newborn = None
        if area.hi > 0:
            if area.lo <= 0:
                raise ValueError("Endpoint hot material requires a positive certified whole vacancy")
            identity = "native-first-entry-hot-birth:" + _hash(dict(
                context=context, vacancy_domain_id=vacancy_id))
            if identity in journal.source_surface.known_material_ids:
                raise ValueError("Endpoint newborn material identity was already present")
            actual_area = area * radius * radius
            thickness = Fraction.from_float(parameters.crust_thickness_km)
            newborn = NativeFirstEntryHotMaterial(identity, parent.fragment_id,
                parent.parcel.plate, parent.parcel.cell, vacancy_id, endpoint, actual_area,
                actual_area * thickness, Fraction(0), Fraction(0), Fraction(0),
                (("damage", parameters.damage), ("water", parameters.water)),
                (thickness, Fraction(0), Fraction(0)),
                measure + " * radius_km^2", "area_km2 * declared_hot_crust_thickness_km")
        vacancies.append(NativeFirstEntryEndpointVacancy(parent.fragment_id, vacancy_id,
            measure, area, tuple(v.vacancy_sha256 for v in raw),
            tuple(d.fragment_id for d in extra), newborn))
    unsealed = NativeFirstEntryMaterialState(journal, certificate, mechanics, mechanics_sha,
        bindings, parameters, provenance, tuple(allocations), tuple(cohorts), tuple(vacancies), "")
    body_bytes = _encoded(_body(unsealed))
    result = replace(unsealed, state_sha256=hashlib.sha256(body_bytes).hexdigest())
    _remember(_STATE_CACHE, result, body_bytes, 8)
    return result


def validate_native_first_entry_material(state):
    if (type(state) is not NativeFirstEntryMaterialState
            or type(state.original_mechanics_payload_json) is not bytes
            or type(state.birth_parameters) is not HotBirthParameters
            or any(type(items) is not tuple for items in (state.trace_bindings,
                state.parent_allocations, state.archive_cohorts, state.endpoint_vacancies))
            or any(type(item) is not NativeFirstEntryTraceBinding for item in state.trace_bindings)
            or any(type(item) is not NativeFirstEntryParentAllocation
                or type(item.original_age_myr) is not Fraction
                or type(item.reservoirs) is not tuple
                or any(type(row) is not NativeFirstEntryReservoir
                    or type(row.parent_amount) is not Fraction for row in item.reservoirs)
                for item in state.parent_allocations)
            or any(type(item) is not NativeFirstEntryArchiveCohort for item in state.archive_cohorts)
            or any(type(item) is not NativeFirstEntryEndpointVacancy
                or (item.newborn is not None and type(item.newborn) is not NativeFirstEntryHotMaterial)
                for item in state.endpoint_vacancies)):
        raise ValueError("Expected the immutable typed complete first-entry material transaction")
    if _typed_fingerprint(state) in _STATE_CACHE:
        return state
    try:
        mechanics = json.loads(state.original_mechanics_payload_json)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Original mechanical seed is not a canonical JSON object") from error
    result = initialize_native_first_entry_material(state.journal, state.sweep_certificate,
        original_mechanics_payload=mechanics, birth_parameters=state.birth_parameters,
        provenance=state.provenance)
    if (state.original_mechanics_payload_json != result.original_mechanics_payload_json
            or state.state_sha256 != result.state_sha256 or _encoded(_body(state)) != _encoded(_body(result))):
        raise ValueError("First-entry material state differs from complete source/geometry/closure replay")
    return state


def native_first_entry_material_to_dict(state):
    state = validate_native_first_entry_material(state)
    body = _STATE_CACHE.get(_typed_fingerprint(state))
    return dict(json.loads(body) if body is not None else _body(state), state_sha256=state.state_sha256)


def native_first_entry_material_from_dict(raw, *, expected_journal=None,
        expected_original_mechanics_payload=None):
    if type(raw) is not dict or raw.get("format") != FORMAT or raw.get("model") != MODEL:
        raise ValueError("Expected the explicit complete first-entry material format")
    from .geometric_native_first_entry_domains import native_first_entry_domains_from_dict
    try:
        journal = native_domain_journal_from_dict(raw["journal"], expected_memory_reference=
            None if expected_journal is None else expected_journal.memory_reference)
        if expected_journal is not None and journal != validate_native_domain_journal(expected_journal):
            raise ValueError("Material checkpoint belongs to another external source/force candidate")
        mechanics = raw["original_mechanics_payload"]
        if (expected_original_mechanics_payload is not None
                and _encoded(mechanics) != _encoded(expected_original_mechanics_payload)):
            raise ValueError("Material checkpoint changed the externally authenticated original mechanics")
        parameters = raw["birth_parameters"]
        if type(parameters) is not dict or set(parameters) != set(HotBirthParameters.__dataclass_fields__):
            raise ValueError("Endpoint hot material has an invalid explicit parameter schema")
        result = initialize_native_first_entry_material(journal,
            native_first_entry_domains_from_dict(raw["sweep_certificate"]),
            original_mechanics_payload=mechanics,
            birth_parameters=HotBirthParameters(**parameters), provenance=raw["provenance"])
    except KeyError as error:
        raise ValueError("Incomplete whole first-entry material checkpoint") from error
    if _encoded(raw) != _encoded(dict(_body(result), state_sha256=result.state_sha256)):
        raise ValueError("Saved whole first-entry ledger differs from complete immutable replay")
    return result


def _parent(state, identity):
    if type(identity) is not str:
        raise ValueError("Material query requires a literal original parent identity")
    matches = tuple(parent for parent in state.journal.source_surface.fragments
                    if parent.fragment_id == identity)
    if len(matches) != 1:
        raise ValueError("Material query has no actual original parent")
    return matches[0]


def _point_body(chronology):
    from .geometric_native_first_entry_domains import native_first_entry_point_to_dict
    return dict(format=POINT_FORMAT, model=MODEL,
        material_state_sha256=chronology.state.state_sha256,
        original_parent_id=chronology.original_parent_id,
        original_point=[_rational(v) for v in chronology.original_point],
        first_entry_certificate=None if chronology.first_entry_certificate is None else
            native_first_entry_point_to_dict(chronology.first_entry_certificate),
        receiver_chronology=None if chronology.receiver_chronology is None else
            native_domain_owner_point_chronology_to_dict(chronology.receiver_chronology),
        finite_material_reservoir_assigned_to_point=False,
        physical_pair_admission_authorized=False, candidate_mechanics_adopted=False)


def certify_native_first_entry_material_point(state, original_parent_id,
        original_point, *, precision=None, order=None, max_depth=192, max_cells=4096):
    """Certify one original trajectory, with first receiving material ancestry."""
    from .geometric_native_first_entry_domains import certify_native_first_entry_point
    state = validate_native_first_entry_material(state)
    parent, point = _parent(state, original_parent_id), _point(original_point)
    if any(_side(NativeRay("seed", point), support) < 0 for support in parent.geometry.supports):
        raise ValueError("Material query point is outside its actual original parent")
    certificate = state.sweep_certificate
    if precision is not None and (type(precision) is not int or precision != certificate.precision):
        raise ValueError("Material point cannot silently change its whole-domain proof precision")
    if order is not None and (type(order) is not int or order != certificate.order):
        raise ValueError("Material point cannot silently change its whole-domain series order")
    if (type(max_depth) is not int or not 1 <= max_depth <= 768
            or type(max_cells) is not int or not 1 <= max_cells <= 65536):
        raise ValueError("Material point resources require explicit bounded integer budgets")
    first, receiving = None, None
    if parent.parcel.plate == certificate.donor_owner:
        first = certify_native_first_entry_point(certificate, original_parent_id, point,
            max_depth=max_depth, max_cells=max_cells)
        receiving = certify_native_domain_journal_owner_point(state.journal,
            original_parent_id, certificate.receiver_owner, point,
            precision=certificate.precision, order=certificate.order,
            max_depth=max_depth, max_cells=max_cells)
    unsealed = NativeFirstEntryMaterialPointChronology(state, original_parent_id,
        point, first, receiving, "")
    body_bytes = _encoded(_point_body(unsealed))
    result = replace(unsealed, point_chronology_sha256=hashlib.sha256(body_bytes).hexdigest())
    _remember(_POINT_CONTEXT_CACHE, result, body_bytes, 64)
    return result


def validate_native_first_entry_material_point(chronology):
    from .geometric_native_first_entry_domains import NativeFirstEntryPointCertificate
    if (type(chronology) is not NativeFirstEntryMaterialPointChronology
            or type(chronology.original_point) is not tuple
            or any(type(v) is not Fraction for v in chronology.original_point)
            or (chronology.first_entry_certificate is not None
                and type(chronology.first_entry_certificate) is not NativeFirstEntryPointCertificate)):
        raise ValueError("Expected an immutable complete original material point context")
    if _typed_fingerprint(chronology) in _POINT_CONTEXT_CACHE:
        return chronology
    first = chronology.first_entry_certificate
    result = certify_native_first_entry_material_point(chronology.state,
        chronology.original_parent_id, chronology.original_point,
        max_depth=192 if first is None else first.max_depth,
        max_cells=4096 if first is None else first.max_cells)
    if (chronology.point_chronology_sha256 != result.point_chronology_sha256
            or _encoded(_point_body(chronology)) != _encoded(_point_body(result))):
        raise ValueError("Material point differs from complete first-entry and receiving ancestry replay")
    return chronology


def _first_receivers(chronology):
    """Keep every possible receiving material at the certified first root.

    The common owner-X support is exactly incident at the true event.  Every
    other original receiver side must retain its certified sign throughout
    the root enclosure.  Coincidence or another overlapping event preserves
    possible ancestry, rather than choosing a representative root timestamp.
    """
    from .geometric_native_point_chronology import evaluate_native_point_chronology
    first = chronology.first_entry_certificate
    if first is None or first.first_entry_time_fraction is None:
        return (), "no_first_entry"
    key = _typed_fingerprint(chronology)
    if key in _FIRST_RECEIVER_CACHE:
        return _FIRST_RECEIVER_CACHE[key]
    window = first.first_entry_time_fraction
    definite, possible = [], []
    for identity, certificate in chronology.receiver_chronology.receiver_certificates:
        low = evaluate_native_point_chronology(certificate, window.lo)
        high = evaluate_native_point_chronology(certificate, window.hi)
        known, unresolved = [], False
        for index, plane in enumerate(certificate.plane_chronologies):
            if plane.support.normal == (-1, 0, 0):
                # All source fragments on the overriding hemisphere share
                # this exact owner-plane root and complete analytic rotation.
                known.append(0)
                continue
            overlaps = any(event.time_fraction.lo <= window.hi
                and window.lo <= event.time_fraction.hi for event in plane.events)
            signs = (low.plane_signs[index], high.plane_signs[index])
            if not overlaps and signs[0] is not None and signs[0] == signs[1]:
                known.append(signs[0])
            elif plane.coincident:
                known.append(0)
            else:
                known.append(None)
                unresolved = True
        if any(sign is not None and sign < 0 for sign in known):
            continue
        possible.append(identity)
        if not unresolved:
            definite.append(identity)
    if not possible:
        raise ValueError("Proved owner first entry lost all original receiving material ancestry")
    status = "proved" if possible == definite else "event_window_ambiguous"
    result = tuple(possible), status
    if len(_FIRST_RECEIVER_CACHE) >= 64:
        _FIRST_RECEIVER_CACHE.clear()
    _FIRST_RECEIVER_CACHE[key] = result
    return result


def evaluate_native_first_entry_material_point(chronology, time_fraction):
    from .geometric_native_first_entry_domains import evaluate_native_first_entry_point
    chronology = validate_native_first_entry_material_point(chronology)
    time = _fraction_time(time_fraction)
    state = chronology.state
    parent = _parent(state, chronology.original_parent_id)
    allocation = next(item for item in state.parent_allocations
                      if item.fragment_id == parent.fragment_id)
    first = chronology.first_entry_certificate
    if first is None:
        status, occupancy, window, debits = "live", "not_a_selected_donor", None, 0
    else:
        query = evaluate_native_first_entry_point(first, time)
        status, occupancy, window, debits = (query.material_allocation,
            query.contact_occupancy, query.first_entry_time_fraction, query.archive_debits)
    receivers, ancestry = _first_receivers(chronology)
    donor_traces = tuple(binding.trace_id for binding in state.trace_bindings
                        if binding.donor_fragment_id == parent.fragment_id)
    receiver_traces = tuple(binding.trace_id for binding in state.trace_bindings
                           if binding.receiver_fragment_id in receivers)
    return NativeFirstEntryMaterialPointQuery(chronology, time, status, occupancy,
        window, receivers, donor_traces, receiver_traces, ancestry,
        allocation.cohort_id, Fraction.from_float(parent.parcel.age_myr)
        + time * Fraction.from_float(state.journal.dt_myr),
        parent.parcel.material_id, parent.parcel.plate, debits)


def _point_query_body(query):
    return dict(format=QUERY_FORMAT, model=MODEL,
        material_state_sha256=query.chronology.state.state_sha256,
        point_chronology_sha256=query.chronology.point_chronology_sha256,
        original_parent_id=query.chronology.original_parent_id,
        original_point=[_rational(v) for v in query.chronology.original_point],
        time_fraction=_rational(query.time_fraction),
        material_allocation=query.material_allocation, contact_occupancy=query.contact_occupancy,
        first_entry_time_fraction=None if query.first_entry_time_fraction is None else
            _interval(query.first_entry_time_fraction),
        receiver_original_fragment_ids=list(query.receiver_original_fragment_ids),
        original_trace_ids=list(query.original_trace_ids), receiver_trace_ids=list(query.receiver_trace_ids),
        first_entry_receiver_ancestry_status=query.first_entry_receiver_ancestry_status,
        cohort_id=query.cohort_id, material_age_myr=_rational(query.material_age_myr),
        material_id=query.material_id, material_owner=query.material_owner,
        archive_debits=query.archive_debits,
        finite_material_reservoir_assigned_to_point=query.finite_material_reservoir_assigned_to_point,
        physical_pair_ancestry_authenticated=query.physical_pair_ancestry_authenticated,
        original_trace_references_preserved=True,
        donor_receiver_trace_difference_creates_new_fault=False,
        candidate_mechanics_adopted=False, physical_pair_admission_authorized=False)


def native_first_entry_material_point_chronology_to_dict(chronology):
    chronology = validate_native_first_entry_material_point(chronology)
    body = _POINT_CONTEXT_CACHE.get(_typed_fingerprint(chronology))
    return dict(json.loads(body) if body is not None else _point_body(chronology),
        point_chronology_sha256=chronology.point_chronology_sha256)


def native_first_entry_material_point_chronology_from_dict(raw, *, state):
    from .geometric_native_first_entry_domains import native_first_entry_point_from_dict
    if type(raw) is not dict or raw.get("format") != POINT_FORMAT or raw.get("model") != MODEL:
        raise ValueError("Expected a complete original first-entry material point context")
    try:
        first_raw = raw["first_entry_certificate"]
        first = None if first_raw is None else native_first_entry_point_from_dict(first_raw)
        result = certify_native_first_entry_material_point(state, raw["original_parent_id"],
            tuple(_read_rational(v) for v in raw["original_point"]),
            max_depth=192 if first is None else first.max_depth,
            max_cells=4096 if first is None else first.max_cells)
    except KeyError as error:
        raise ValueError("Incomplete first-entry material point context") from error
    if _encoded(raw) != _encoded(native_first_entry_material_point_chronology_to_dict(result)):
        raise ValueError("Saved material point context differs from its complete authoritative ledger replay")
    return result


def native_first_entry_material_point_query_to_dict(query):
    if (type(query) is not NativeFirstEntryMaterialPointQuery
            or type(query.time_fraction) is not Fraction or type(query.archive_debits) is not int
            or type(query.material_owner) is not int or type(query.material_age_myr) is not Fraction
            or query.physical_pair_ancestry_authenticated is not False
            or query.finite_material_reservoir_assigned_to_point is not False):
        raise ValueError("Material point query cannot assign volume or authenticate a new physical pair")
    result = evaluate_native_first_entry_material_point(query.chronology, query.time_fraction)
    if (query != result or _encoded(_point_query_body(query)) != _encoded(_point_query_body(result))):
        raise ValueError("Material point query differs from its immutable authoritative context")
    return _point_query_body(result)


def native_first_entry_material_point_query_from_dict(raw, *, chronology):
    if type(raw) is not dict or raw.get("format") != QUERY_FORMAT or raw.get("model") != MODEL:
        raise ValueError("Expected the original material point query schema")
    try:
        result = evaluate_native_first_entry_material_point(chronology, _read_rational(raw["time_fraction"]))
    except KeyError as error:
        raise ValueError("Incomplete material point query") from error
    if _encoded(raw) != _encoded(native_first_entry_material_point_query_to_dict(result)):
        raise ValueError("Saved material point query differs from full authoritative point replay")
    return result


def evaluate_native_first_entry_material_endpoint_point(chronology):
    """Query the exact endpoint ray of an authenticated original trajectory.

    A source Fraction point and its complete owner pose are retained.  General
    arbitrary world rays have no automatic rational pullback; this API does
    not project such rays onto a fabricated source point.
    """
    chronology = validate_native_first_entry_material_point(chronology)
    state = chronology.state
    parent = _parent(state, chronology.original_parent_id)
    omega = dict(state.journal.omega_rad_per_myr)
    ray = NativeRay("seed", chronology.original_point).rotated((RotationIncrement(
        omega[parent.parcel.plate], state.journal.dt_myr),))
    query = evaluate_native_first_entry_material_point(chronology, Fraction(1))
    old_parents = tuple(original.fragment_id for original in state.journal.source_surface.fragments
        if all(_side(ray, support) >= 0 for support in original.geometry.supports))
    if not old_parents:
        raise ValueError("Endpoint ray lost the complete original fixed-world vacancy partition")
    raw = True  # Its original moved parent always covers this authenticated ray.
    if parent.parcel.plate == state.sweep_certificate.receiver_owner:
        retained, vacancy = True, False
    elif query.contact_occupancy in ("inside", "boundary"):
        retained, vacancy = True, False
    elif query.material_allocation in ("live", "boundary"):
        retained, vacancy = True, False
    elif query.material_allocation == "archived" and query.contact_occupancy == "outside":
        retained, vacancy = False, True
    else:
        retained, vacancy = None, None
    births = ()
    if vacancy is True:
        births = tuple(item.newborn.material_id for item in state.endpoint_vacancies
            if item.fixed_original_parent_id in old_parents and item.newborn is not None)
        if not births:
            raise ValueError("A proved endpoint material vacancy has no actual hot material domain")
    return NativeFirstEntryMaterialEndpointQuery(chronology, ray, query, raw,
        retained, vacancy, old_parents, births, query.material_age_myr)


def evaluate_native_first_entry_material_endpoint(chronology):
    return evaluate_native_first_entry_material_endpoint_point(chronology)


def _endpoint_query_body(query):
    return dict(format=ENDPOINT_QUERY_FORMAT, model=MODEL,
        material_state_sha256=query.chronology.state.state_sha256,
        point_chronology_sha256=query.chronology.point_chronology_sha256,
        original_parent_id=query.chronology.original_parent_id,
        original_point=[_rational(v) for v in query.chronology.original_point],
        endpoint_ray=native_ray_to_dict(query.endpoint_ray),
        point_query=_point_query_body(query.point_query),
        raw_coverage=query.raw_coverage, retained_material=query.retained_material,
        actual_material_vacancy=query.actual_material_vacancy,
        fixed_original_parent_ids=list(query.fixed_original_parent_ids),
        birth_material_ids=list(query.birth_material_ids), material_age_myr=_rational(query.material_age_myr),
        physical_pair_admission_authorized=query.physical_pair_admission_authorized,
        candidate_mechanics_adopted=query.candidate_mechanics_adopted,
        point_domain="exact endpoint ray of the original donor trajectory",
        raw_coverage_does_not_promote_archived_material_to_live=True)


def native_first_entry_material_endpoint_query_to_dict(query):
    if (type(query) is not NativeFirstEntryMaterialEndpointQuery
            or type(query.raw_coverage) not in (bool, type(None))
            or type(query.retained_material) not in (bool, type(None))
            or type(query.actual_material_vacancy) not in (bool, type(None))
            or type(query.material_age_myr) is not Fraction
            or query.physical_pair_admission_authorized is not False
            or query.candidate_mechanics_adopted is not False):
        raise ValueError("Endpoint material query cannot adopt an unaccepted physical force state")
    result = evaluate_native_first_entry_material_endpoint_point(query.chronology)
    # Validate the nested point query before equality can alias bool/int or
    # float/Fraction fields.  Canonical bytes retain every authoritative type.
    native_first_entry_material_point_query_to_dict(query.point_query)
    if (query != result or _encoded(_endpoint_query_body(query)) != _encoded(_endpoint_query_body(result))):
        raise ValueError("Endpoint material query differs from its exact source trajectory and complete ledger")
    return _endpoint_query_body(result)


def native_first_entry_material_endpoint_query_from_dict(raw, *, chronology):
    if type(raw) is not dict or raw.get("format") != ENDPOINT_QUERY_FORMAT or raw.get("model") != MODEL:
        raise ValueError("Expected the actual endpoint material vacancy query schema")
    result = evaluate_native_first_entry_material_endpoint_point(chronology)
    if _encoded(raw) != _encoded(native_first_entry_material_endpoint_query_to_dict(result)):
        raise ValueError("Saved endpoint material vacancy differs from the complete original trajectory replay")
    return result
