"""Whole original native domains and unresolved endpoint transport evidence.

Every original fragment and its complete parcel payload is retained.  Complete
owner rotations produce all endpoint overlaps and their two original-material
pullbacks.  Raw geometric gaps are computed against ALL moved original domains
and partitioned by the fixed original world footprints.

An overlap supplies two potential directions; it does not decide which material
is removed.  First-contact archive/live domain definitions remain conditional
on separately authenticated physical pairs and direction.  Their four reservoir
complements conserve the original symbolic amounts without adopting a loss.

The existing birth closure is endpoint-only: at the end of an accepted step,
hot material fills fixed old footprints minus the actually retained surface.
This journal reserves only RAW kinematic gaps, without creating a parcel.  A raw
gap is not a proof of the vacancy left by a once-only passive archive: material
which has already entered an archive can later leave geometric contact.  No
continuous hot-birth, newborn transport or physical pair is inferred here.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from collections.abc import Mapping
from itertools import product
from numbers import Integral, Real

from .fractional_surface import EXTENSIVE_FIELDS
from .geometric_intrinsic_geometry import RotationIncrement
from .geometric_native_buried_support import native_pullback
from .geometric_native_material_history import (
    NativeMaterialMemoryReference, _checked_memory, _fragment_body, _memory_body,
    make_native_material_memory_reference,
)
from .geometric_native_polygons import (
    NativePolygon, native_intersection, native_polygon_to_dict, native_subtract,
)
from .geometric_native_pose import FractionInterval as I
from .geometric_native_surface import (
    NativeSurfaceState, native_surface_from_dict, native_surface_to_dict,
)


FORMAT = "native-whole-original-domain-journal-1"
MODEL = "complete_original_domains_with_unresolved_pair_direction_and_raw_gaps"


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _rational(value):
    if type(value) is not Fraction:
        raise ValueError("Whole-domain journal requires exact Fraction amounts and clocks")
    return [value.numerator, value.denominator]


def _interval(value):
    if type(value) is not I:
        raise ValueError("Whole-domain geometry requires native exact measure enclosures")
    return dict(lo=_rational(value.lo), hi=_rational(value.hi))


def _checked_source(surface):
    if type(surface) is not NativeSurfaceState:
        raise ValueError("Whole-domain journal requires the complete original native partition")
    original = native_surface_to_dict(surface)
    checked = native_surface_from_dict(original)
    if _encoded(original) != _encoded(native_surface_to_dict(checked)):
        raise ValueError("Original native source changed during complete typed reconstruction")
    return checked


def _omega(surface, supplied):
    if not isinstance(supplied, Mapping):
        raise ValueError("Whole-domain journal requires a complete explicit owner mapping")
    checked = {}
    for owner, vector in supplied.items():
        if isinstance(owner, bool) or not isinstance(owner, Integral) or owner < 0:
            raise ValueError("Every original owner must be an explicit nonnegative integer")
        owner = int(owner)
        if owner in checked:
            raise ValueError("Whole-domain owner mapping contains repeated identities")
        if (not isinstance(vector, (tuple, list)) or len(vector) != 3
                or any(isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(float(value)) for value in vector)):
            raise ValueError("Each original owner needs its full three finite Euler components")
        checked[owner] = tuple(float(value) for value in vector)
    owners = {fragment.parcel.plate for fragment in surface.fragments}
    if set(checked) != owners:
        raise ValueError("Whole-domain motion must name every original owner exactly once")
    return tuple(sorted(checked.items()))


def _duration(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("Whole-domain interval requires an explicit positive physical number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("Whole-domain interval must be positive and finite")
    return result


@dataclass(frozen=True, slots=True)
class NativeDomainMovedFragment:
    fragment_id: str
    owner: int
    material_id: str
    geometry: NativePolygon
    material_age_at_endpoint_myr: Fraction
    parent_payload_sha256: str


@dataclass(frozen=True, slots=True)
class NativeDomainEndpointPair:
    """Both potential donor pullbacks; neither direction is selected."""
    fragment_a: str
    fragment_b: str
    owner_a: int
    owner_b: int
    overlap_geometry: NativePolygon
    donor_a_original_domain: NativePolygon
    donor_b_original_domain: NativePolygon
    area_bounds_sr: I
    pair_sha256: str


@dataclass(frozen=True, slots=True)
class NativeDomainVacancy:
    fixed_original_parent_id: str
    geometry: NativePolygon
    area_bounds_sr: I
    vacancy_sha256: str


@dataclass(frozen=True, slots=True)
class NativeDomainJournal:
    source_surface: NativeSurfaceState
    omega_rad_per_myr: tuple
    dt_myr: float
    memory_reference: NativeMaterialMemoryReference
    source_surface_sha256: str
    moved_fragments: tuple[NativeDomainMovedFragment, ...]
    endpoint_pairs: tuple[NativeDomainEndpointPair, ...]
    raw_vacancies: tuple[NativeDomainVacancy, ...]
    journal_sha256: str

    @property
    def overlaps(self):
        return self.endpoint_pairs


@dataclass(frozen=True, slots=True)
class NativeDomainPointChronology:
    journal: NativeDomainJournal
    donor_fragment_id: str
    receiver_fragment_id: str
    original_point: tuple[Fraction, Fraction, Fraction]
    chronology_certificate: object
    point_chronology_sha256: str


@dataclass(frozen=True, slots=True)
class NativeDomainPairPointQuery:
    chronology: NativeDomainPointChronology
    point_query: object
    conditional_material_allocation: str
    conditional_first_entry_time_fraction: I | None
    conditional_cohort_id: str
    material_age_myr: Fraction
    memory_reference: NativeMaterialMemoryReference
    original_parent_reservoir_references: tuple
    pair_direction_authenticated: bool = False
    finite_material_reservoir_assigned_to_point: bool = False
    admission_authorized: bool = False

    @property
    def contact_occupancy(self):
        return self.point_query.occupancy

    @property
    def events(self):
        return self.chronology.chronology_certificate.occupancy_events


@dataclass(frozen=True, slots=True)
class NativeDomainWorldPointChronology:
    journal: NativeDomainJournal
    fixed_world_point: tuple[Fraction, Fraction, Fraction]
    receiver_certificates: tuple
    point_chronology_sha256: str


@dataclass(frozen=True, slots=True)
class NativeDomainWorldPointQuery:
    chronology: NativeDomainWorldPointChronology
    time_fraction: Fraction
    receiver_queries: tuple
    raw_gap_status: str
    boolean_sign_assignments_checked: int
    memory_reference: NativeMaterialMemoryReference
    actual_passive_archive_vacancy_proved: bool = False
    newborn_material_created: bool = False
    admission_authorized: bool = False


@dataclass(frozen=True, slots=True)
class NativeDomainOwnerPointChronology:
    journal: NativeDomainJournal
    donor_fragment_id: str
    receiver_owner: int
    original_point: tuple[Fraction, Fraction, Fraction]
    receiver_certificates: tuple
    point_chronology_sha256: str


@dataclass(frozen=True, slots=True)
class NativeDomainOwnerPointQuery:
    chronology: NativeDomainOwnerPointChronology
    time_fraction: Fraction
    receiver_queries: tuple
    contact_occupancy: str
    boolean_sign_assignments_checked: int
    memory_reference: NativeMaterialMemoryReference
    material_loss_decided: bool = False
    new_fault_inferred_from_internal_boundary: bool = False
    admission_authorized: bool = False


def _moved_body(item):
    return dict(fragment_id=item.fragment_id, owner=item.owner, material_id=item.material_id,
        geometry=native_polygon_to_dict(item.geometry),
        material_age_at_endpoint_myr=_rational(item.material_age_at_endpoint_myr),
        parent_payload_sha256=item.parent_payload_sha256)


def _pair_body(item):
    return dict(fragment_a=item.fragment_a, fragment_b=item.fragment_b,
        owner_a=item.owner_a, owner_b=item.owner_b,
        overlap_geometry=native_polygon_to_dict(item.overlap_geometry),
        donor_a_original_domain=native_polygon_to_dict(item.donor_a_original_domain),
        donor_b_original_domain=native_polygon_to_dict(item.donor_b_original_domain),
        area_bounds_sr=_interval(item.area_bounds_sr),
        direction_status="unresolved_both_potential_original_material_directions")


def _vacancy_body(item):
    return dict(fixed_original_parent_id=item.fixed_original_parent_id,
        geometry=native_polygon_to_dict(item.geometry), area_bounds_sr=_interval(item.area_bounds_sr),
        domain="fixed_original_parent minus union_of_ALL_moved_original_domains",
        vacancy_kind="raw_kinematic_gap_not_passive_archive_material_vacancy")


def _subtract_all(geometry, covers):
    pieces = (geometry,)
    for cover in covers:
        pieces = tuple(part for piece in pieces for part in native_subtract(piece, cover))
        if not pieces:
            break
    return pieces


@lru_cache(maxsize=32)
def _endpoint_geometry(source_json, omega_rows, dt_myr):
    """Cache only canonical bytes and normalized, publicly checked primitives."""
    surface = native_surface_from_dict(json.loads(source_json))
    motion = dict(omega_rows)
    increments = {owner: RotationIncrement(vector, dt_myr) for owner, vector in omega_rows}
    duration = Fraction.from_float(dt_myr)
    moved = tuple(NativeDomainMovedFragment(fragment.fragment_id, fragment.parcel.plate,
        fragment.parcel.material_id, fragment.geometry.rotated(motion[fragment.parcel.plate], dt_myr),
        Fraction.from_float(fragment.parcel.age_myr) + duration, _hash(_fragment_body(fragment)))
        for fragment in surface.fragments)
    pairs = []
    for index, first in enumerate(moved):
        for second in moved[index + 1:]:
            if first.owner == second.owner:
                continue
            overlap = native_intersection(first.geometry, second.geometry)
            if overlap is None:
                continue
            unsealed = NativeDomainEndpointPair(first.fragment_id, second.fragment_id,
                first.owner, second.owner, overlap,
                native_pullback(overlap, (increments[first.owner],)),
                native_pullback(overlap, (increments[second.owner],)), overlap.area_bounds(), "")
            from dataclasses import replace
            pairs.append(replace(unsealed, pair_sha256=_hash(_pair_body(unsealed))))
    vacancies = []
    covers = tuple(fragment.geometry for fragment in moved)
    for fragment in surface.fragments:
        for geometry in _subtract_all(fragment.geometry, covers):
            unsealed = NativeDomainVacancy(fragment.fragment_id, geometry, geometry.area_bounds(), "")
            from dataclasses import replace
            vacancies.append(replace(unsealed, vacancy_sha256=_hash(_vacancy_body(unsealed))))
    return moved, tuple(pairs), tuple(vacancies)


def _parent_domains_body(journal):
    bodies = []
    for parent in journal.source_surface.fragments:
        payload_sha = _hash(_fragment_body(parent))
        contacts = [dict(receiver_fragment_id=receiver.fragment_id,
            receiver_owner=receiver.parcel.plate,
            receiver_payload_sha256=_hash(_fragment_body(receiver)),
            predicate="q in P_parent and all (exp(-Omega_parent*u*D) exp(Omega_receiver*u*D) receiver_normal) dot q >= 0",
            predicate_authority="literal_receiver_supports_and_full_owner_rotation_journals",
            physical_pair_and_direction_authenticated=False)
            for receiver in journal.source_surface.fragments
            if receiver.parcel.plate != parent.parcel.plate]
        reservoirs = []
        for field in EXTENSIVE_FIELDS:
            amount = Fraction.from_float(getattr(parent.parcel, field))
            reservoirs.append(dict(reservoir=field, original_amount=_rational(amount),
                conditional_archive_expression="original_amount * native_area({q in parent: exists s<=u with physically_authenticated_first_contact(q,s)}) / native_area(parent)",
                conditional_live_expression="original_amount - conditional_archive_amount",
                each_conditional_amount_bounds=_interval(I(0, amount)),
                conservation_identity="conditional_live_amount + conditional_archive_amount = original_amount",
                material_allocation_adopted=False))
        bodies.append(dict(fragment_id=parent.fragment_id, material_id=parent.parcel.material_id,
            owner=parent.parcel.plate, parent_payload_sha256=payload_sha,
            original_geometry_sha256=_hash(native_polygon_to_dict(parent.geometry)),
            potential_contact_domains=contacts, original_reservoirs=reservoirs,
            material_fields=list(parent.parcel.material_fields),
            specific_properties=list(parent.parcel.specific_properties),
            original_age_myr=_rational(Fraction.from_float(parent.parcel.age_myr)),
            conditional_archive_domain="first authenticated entry of each original point; once-only passive temporal extension",
            conditional_live_domain="original_parent minus conditional_archive_domain",
            contact_exit_does_not_unbury_under_conditional_law=True,
            scope="complete_original_parent_domain; conditional_material_adoption_unresolved"))
    return bodies


def _birth_reservations_body(journal):
    radius = Fraction.from_float(journal.source_surface.radius_km)
    endpoint = Fraction.from_float(journal.source_surface.time_myr) + Fraction.from_float(journal.dt_myr)
    return [dict(vacancy_sha256=item.vacancy_sha256,
        fixed_original_parent_id=item.fixed_original_parent_id,
        raw_area_km2_bounds=_interval(item.area_bounds_sr * radius * radius),
        possible_birth_time_myr=_rational(endpoint),
        birth_law="endpoint_hot_fill_of_actual_retained_surface_vacancy_only",
        original_kinematic_gap_is_only_a_reservation=True,
        actual_retained_surface_available=False,
        newborn_parcel_created=False,
        prospective_area_expression="native_area(actual_accepted_vacancy) * radius_km^2",
        prospective_oceanic_volume_expression="prospective_area_km2 * separately_declared_hot_crust_thickness_km",
        prospective_cold_mantle_volume=[0, 1], prospective_density_excess_mass=[0, 1],
        prospective_age_at_birth=[0, 1],
        damage_water_and_material_identity_require_explicit_birth_factory=True,
        continuous_birth_or_newborn_motion_defined=False)
        for item in journal.raw_vacancies]


def _body(journal):
    return dict(format=FORMAT, model=MODEL,
        source_surface=native_surface_to_dict(journal.source_surface),
        source_surface_sha256=journal.source_surface_sha256,
        omega_rad_per_myr=[[owner, list(vector)] for owner, vector in journal.omega_rad_per_myr],
        dt_myr=journal.dt_myr,
        exact_interval_myr=_rational(Fraction.from_float(journal.dt_myr)),
        memory_reference=_memory_body(journal.memory_reference),
        moved_fragments=[_moved_body(item) for item in journal.moved_fragments],
        endpoint_pairs=[dict(_pair_body(item), pair_sha256=item.pair_sha256)
                        for item in journal.endpoint_pairs],
        raw_vacancies=[dict(_vacancy_body(item), vacancy_sha256=item.vacancy_sha256)
                       for item in journal.raw_vacancies],
        whole_original_parent_domains=_parent_domains_body(journal),
        endpoint_hot_birth_reservations=_birth_reservations_body(journal),
        raw_gap_domain="sphere minus union_of_ALL_moved_original_domains",
        passive_archive_material_vacancy_domain="sphere minus union_of_moved_conditionally_live_original_domains",
        raw_gap_equals_passive_archive_material_vacancy_proved=False,
        endpoint_pair_direction_resolved=False, material_loss_decided=False,
        original_material_allocation_adopted=False, newborn_material_created=False,
        mechanical_memory_policy="preserve_external_before_and_unaccepted_candidate_references",
        candidate_mechanics_adopted=False, global_inventory_updated=False,
        admission_authorized=False)


def initialize_native_domain_journal(surface, omega_rad_per_myr, dt_myr, *, memory_reference):
    """Compute every original domain, endpoint potential overlap and raw gap."""
    source = _checked_source(surface)
    omega, dt = _omega(source, omega_rad_per_myr), _duration(dt_myr)
    memory = _checked_memory(memory_reference)
    source_body = native_surface_to_dict(source)
    moved, pairs, vacancies = _endpoint_geometry(_encoded(source_body), omega, dt)
    unsealed = NativeDomainJournal(source, omega, dt, memory, _hash(source_body),
        moved, pairs, vacancies, "")
    from dataclasses import replace
    return replace(unsealed, journal_sha256=_hash(_body(unsealed)))


def validate_native_domain_journal(journal):
    """Uncached typed entry point, then complete canonical geometry replay."""
    if type(journal) is not NativeDomainJournal:
        raise ValueError("Expected the complete immutable original-domain journal")
    if (type(journal.dt_myr) is not float or type(journal.omega_rad_per_myr) is not tuple
            or any(type(row) is not tuple or len(row) != 2 or type(row[0]) is not int
                or type(row[1]) is not tuple or len(row[1]) != 3
                or any(type(value) is not float for value in row[1])
                for row in journal.omega_rad_per_myr)):
        raise ValueError("Saved domain motion must retain its exact typed full owner mapping")
    if (any(type(items) is not tuple for items in (
            journal.moved_fragments, journal.endpoint_pairs, journal.raw_vacancies))
            or any(type(item) is not NativeDomainMovedFragment
                or type(item.material_age_at_endpoint_myr) is not Fraction
                or type(item.owner) is not int for item in journal.moved_fragments)
            or any(type(item) is not NativeDomainEndpointPair
                or type(item.owner_a) is not int or type(item.owner_b) is not int
                or type(item.area_bounds_sr) is not I for item in journal.endpoint_pairs)
            or any(type(item) is not NativeDomainVacancy or type(item.area_bounds_sr) is not I
                for item in journal.raw_vacancies)):
        raise ValueError("Domain geometry records must retain immutable exact typed provenance")
    reproduced = initialize_native_domain_journal(journal.source_surface,
        dict(journal.omega_rad_per_myr), journal.dt_myr, memory_reference=journal.memory_reference)
    if (journal.journal_sha256 != reproduced.journal_sha256
            or _encoded(_body(journal)) != _encoded(_body(reproduced))):
        raise ValueError("Domain journal differs from complete original source and geometry replay")
    return journal


def native_domain_journal_to_dict(journal):
    journal = validate_native_domain_journal(journal)
    return json.loads(_encoded(dict(_body(journal), journal_sha256=journal.journal_sha256)))


def native_domain_journal_from_dict(raw, *, expected_memory_reference=None):
    """Recompute all overlaps, pullbacks and vacancies rather than trust seals."""
    if type(raw) is not dict or raw.get("format") != FORMAT or raw.get("model") != MODEL:
        raise ValueError("Expected the explicit whole original native domain journal schema")
    try:
        rows = raw["omega_rad_per_myr"]
        if (type(rows) is not list or any(type(row) is not list or len(row) != 2
                or type(row[0]) is not int or type(row[1]) is not list or len(row[1]) != 3
                or any(type(value) is not float for value in row[1]) for row in rows)
                or len({row[0] for row in rows}) != len(rows) or type(raw["dt_myr"]) is not float):
            raise ValueError("Invalid exact typed whole-domain motion schema")
        reference = raw["memory_reference"]
        if (type(reference) is not dict or set(reference) != {"before_payload", "candidate_after_payload",
                "before_sha256", "candidate_after_sha256", "constitutive_path_sha256"}):
            raise ValueError("Invalid whole-domain mechanical reference schema")
        memory = make_native_material_memory_reference(reference["before_payload"],
            reference["candidate_after_payload"], constitutive_path_sha256=reference["constitutive_path_sha256"])
        if _encoded(reference) != _encoded(_memory_body(memory)):
            raise ValueError("Whole-domain mechanical memory references changed")
        if expected_memory_reference is not None and memory != _checked_memory(expected_memory_reference):
            raise ValueError("Domain journal differs from the external physical memory evidence")
        result = initialize_native_domain_journal(native_surface_from_dict(raw["source_surface"]),
            {owner: tuple(vector) for owner, vector in rows}, raw["dt_myr"], memory_reference=memory)
    except KeyError as error:
        raise ValueError("Incomplete whole original native domain journal") from error
    expected = dict(_body(result), journal_sha256=result.journal_sha256)
    if _encoded(raw) != _encoded(expected):
        raise ValueError("Saved whole-domain journal differs from full canonical geometry replay")
    return result


def _source_fragment(journal, identity):
    if type(identity) is not str:
        raise ValueError("Domain point query requires a literal original fragment identity")
    matches = tuple(fragment for fragment in journal.source_surface.fragments
                    if fragment.fragment_id == identity)
    if len(matches) != 1:
        raise ValueError("Domain point query does not name an actual original source fragment")
    return matches[0]


def _point(point):
    if (type(point) is not tuple or len(point) != 3
            or any(type(value) not in (int, Fraction) for value in point)):
        raise ValueError("Domain point query requires an immutable exact Cartesian triple")
    result = tuple(Fraction(value) for value in point)
    if not any(result):
        raise ValueError("Domain point direction must be nonzero")
    return result


def _point_context_body(chronology):
    from .geometric_native_point_chronology import native_point_chronology_to_dict
    parent = _source_fragment(chronology.journal, chronology.donor_fragment_id)
    receiver = _source_fragment(chronology.journal, chronology.receiver_fragment_id)
    return dict(format="native-whole-domain-pair-point-chronology-1", model=MODEL,
        journal_sha256=chronology.journal.journal_sha256,
        source_surface_sha256=chronology.journal.source_surface_sha256,
        donor_fragment_id=parent.fragment_id, receiver_fragment_id=receiver.fragment_id,
        donor_payload_sha256=_hash(_fragment_body(parent)),
        receiver_payload_sha256=_hash(_fragment_body(receiver)),
        original_point=[_rational(value) for value in chronology.original_point],
        chronology_certificate=native_point_chronology_to_dict(chronology.chronology_certificate),
        pair_direction_authenticated=False, material_loss_decided=False,
        admission_authorized=False, global_inventory_updated=False)


def certify_native_domain_journal_pair_point(journal, donor_fragment_id,
        receiver_fragment_id, point, *, precision=256, order=24,
        max_depth=192, max_cells=4096):
    """Bind complete all-side point events to two whole original domains.

    The point engine has its explicit unposed primitive source scope.  This
    wrapper does not project a journal domain to extend that scope and does not
    use endpoint overlap alone as permission for an earlier physical contact.
    """
    from .geometric_native_point_chronology import certify_native_point_chronology
    journal = validate_native_domain_journal(journal)
    donor, receiver = _source_fragment(journal, donor_fragment_id), _source_fragment(journal, receiver_fragment_id)
    if donor.parcel.plate == receiver.parcel.plate:
        raise ValueError("A potential underthrust pair requires different original material owners")
    point = _point(point)
    omega = dict(journal.omega_rad_per_myr)
    certificate = certify_native_point_chronology(donor.geometry, receiver.geometry,
        omega[donor.parcel.plate], omega[receiver.parcel.plate], journal.dt_myr, point,
        precision=precision, order=order, max_depth=max_depth, max_cells=max_cells)
    unsealed = NativeDomainPointChronology(journal, donor.fragment_id,
        receiver.fragment_id, point, certificate, "")
    from dataclasses import replace
    return replace(unsealed, point_chronology_sha256=_hash(_point_context_body(unsealed)))


def validate_native_domain_pair_point_chronology(chronology):
    from .geometric_native_point_chronology import NativePointChronologyCertificate
    if (type(chronology) is not NativeDomainPointChronology
            or type(chronology.original_point) is not tuple
            or any(type(value) is not Fraction for value in chronology.original_point)
            or type(chronology.chronology_certificate) is not NativePointChronologyCertificate):
        raise ValueError("Expected an exact whole-domain original material point chronology")
    certificate = chronology.chronology_certificate
    reproduced = certify_native_domain_journal_pair_point(chronology.journal,
        chronology.donor_fragment_id, chronology.receiver_fragment_id, chronology.original_point,
        precision=certificate.precision, order=certificate.order,
        max_depth=certificate.max_depth, max_cells=certificate.max_cells)
    if (chronology.point_chronology_sha256 != reproduced.point_chronology_sha256
            or _encoded(_point_context_body(chronology)) != _encoded(_point_context_body(reproduced))):
        raise ValueError("Domain point chronology differs from original domain and all-side event replay")
    return chronology


def evaluate_native_domain_journal_pair_point(chronology, time_fraction):
    """Query contact and a conditional once-only first-entry closure.

    The archive/live answer states what that explicit temporal law would do
    IF this pair and direction were subsequently authenticated.  The journal
    retains their unresolved status and assigns no finite volume to a point.
    """
    from .geometric_native_point_chronology import evaluate_native_point_chronology
    chronology = validate_native_domain_pair_point_chronology(chronology)
    certificate = chronology.chronology_certificate
    query = evaluate_native_point_chronology(certificate, time_fraction)
    first_entry = I(0) if certificate.initial_occupancy == "inside" else next(
        (event.time_fraction for event in certificate.occupancy_events if event.kind == "entry"), None)
    if first_entry is None or time_fraction < first_entry.lo:
        allocation = "live"
    elif first_entry.lo <= time_fraction <= first_entry.hi and first_entry.width:
        allocation = "first_entry_window"
    else:
        allocation = "archived"
    parent = _source_fragment(chronology.journal, chronology.donor_fragment_id)
    cohort = "conditional-native-domain-first-contact:" + _hash(dict(
        journal=chronology.journal.journal_sha256, donor=chronology.donor_fragment_id,
        receiver=chronology.receiver_fragment_id))
    return NativeDomainPairPointQuery(chronology, query, allocation, first_entry, cohort,
        Fraction.from_float(parent.parcel.age_myr) + time_fraction * Fraction.from_float(certificate.dt_myr),
        chronology.journal.memory_reference,
        tuple((field, Fraction.from_float(getattr(parent.parcel, field))) for field in EXTENSIVE_FIELDS))


def native_domain_pair_point_chronology_to_dict(chronology):
    chronology = validate_native_domain_pair_point_chronology(chronology)
    return dict(_point_context_body(chronology), point_chronology_sha256=chronology.point_chronology_sha256)


def native_domain_pair_point_chronology_from_dict(raw, *, journal):
    from .geometric_native_point_chronology import native_point_chronology_from_dict
    if type(raw) is not dict or raw.get("format") != "native-whole-domain-pair-point-chronology-1":
        raise ValueError("Expected the explicit whole-domain pair point chronology schema")
    try:
        certificate = native_point_chronology_from_dict(raw["chronology_certificate"])
        result = certify_native_domain_journal_pair_point(journal, raw["donor_fragment_id"],
            raw["receiver_fragment_id"], certificate.point, precision=certificate.precision,
            order=certificate.order, max_depth=certificate.max_depth, max_cells=certificate.max_cells)
    except KeyError as error:
        raise ValueError("Incomplete whole-domain pair point chronology") from error
    if _encoded(raw) != _encoded(native_domain_pair_point_chronology_to_dict(result)):
        raise ValueError("Saved pair point chronology differs from full contextual event replay")
    return result


def native_domain_pair_point_query_to_dict(query):
    from .geometric_native_point_chronology import native_point_chronology_query_to_dict
    if (type(query) is not NativeDomainPairPointQuery
            or type(query.material_age_myr) is not Fraction
            or query.pair_direction_authenticated is not False
            or query.finite_material_reservoir_assigned_to_point is not False
            or query.admission_authorized is not False):
        raise ValueError("Whole-domain pair query must retain exact age and its unresolved material status")
    reproduced = evaluate_native_domain_journal_pair_point(query.chronology, query.point_query.time_fraction)
    if query != reproduced:
        raise ValueError("Whole-domain pair point query differs from full event and material reference replay")
    parent = _source_fragment(query.chronology.journal, query.chronology.donor_fragment_id)
    return dict(format="native-whole-domain-pair-point-query-1", model=MODEL,
        journal_sha256=query.chronology.journal.journal_sha256,
        point_chronology_sha256=query.chronology.point_chronology_sha256,
        donor_fragment_id=query.chronology.donor_fragment_id,
        receiver_fragment_id=query.chronology.receiver_fragment_id,
        point_query=native_point_chronology_query_to_dict(query.point_query),
        conditional_material_allocation=query.conditional_material_allocation,
        conditional_first_entry_time_fraction=None if query.conditional_first_entry_time_fraction is None
            else _interval(query.conditional_first_entry_time_fraction),
        conditional_cohort_id=query.conditional_cohort_id,
        conditional_archive_law="once_only_passive_first_entry_if_pair_and_direction_are_authenticated",
        material_age_myr=_rational(query.material_age_myr), material_id=parent.parcel.material_id,
        material_owner=parent.parcel.plate,
        original_parent_reservoir_references=[dict(reservoir=field, original_amount=_rational(amount))
            for field, amount in query.original_parent_reservoir_references],
        memory_reference=_memory_body(query.memory_reference),
        pair_direction_authenticated=False, material_loss_decided=False,
        finite_material_reservoir_assigned_to_point=False, newborn_material_created=False,
        admission_authorized=False, global_inventory_updated=False)


def _world_context_body(chronology):
    from .geometric_native_point_chronology import native_point_chronology_to_dict
    return dict(format="native-whole-domain-fixed-world-point-chronology-1", model=MODEL,
        journal_sha256=chronology.journal.journal_sha256,
        source_surface_sha256=chronology.journal.source_surface_sha256,
        fixed_world_point=[_rational(value) for value in chronology.fixed_world_point],
        fixed_world_omega=[0., 0., 0.],
        receiver_certificates=[dict(receiver_fragment_id=identity,
            chronology_certificate=native_point_chronology_to_dict(certificate))
            for identity, certificate in chronology.receiver_certificates],
        scope="ALL_original_moving_fragments_raw_kinematic_union_only",
        actual_passive_archive_vacancy_proved=False, newborn_material_created=False,
        admission_authorized=False, global_inventory_updated=False)


def certify_native_domain_journal_world_point(journal, point, *, precision=256,
        order=24, max_depth=192, max_cells=4096):
    """Prove fixed-world occupancy against every moving original fragment."""
    from .geometric_native_point_chronology import certify_native_point_receiver_chronology
    journal = validate_native_domain_journal(journal)
    point, omega = _point(point), dict(journal.omega_rad_per_myr)
    certificates = tuple((fragment.fragment_id, certify_native_point_receiver_chronology(
        fragment.geometry, (0., 0., 0.), omega[fragment.parcel.plate], journal.dt_myr, point,
        precision=precision, order=order, max_depth=max_depth, max_cells=max_cells))
        for fragment in journal.source_surface.fragments)
    unsealed = NativeDomainWorldPointChronology(journal, point, certificates, "")
    from dataclasses import replace
    return replace(unsealed, point_chronology_sha256=_hash(_world_context_body(unsealed)))


def validate_native_domain_world_point_chronology(chronology):
    from .geometric_native_point_chronology import NativePointChronologyCertificate
    if (type(chronology) is not NativeDomainWorldPointChronology
            or type(chronology.fixed_world_point) is not tuple
            or any(type(value) is not Fraction for value in chronology.fixed_world_point)
            or type(chronology.receiver_certificates) is not tuple
            or len(chronology.receiver_certificates) != len(chronology.journal.source_surface.fragments)
            or any(type(row) is not tuple or len(row) != 2 or type(row[0]) is not str
                or type(row[1]) is not NativePointChronologyCertificate
                for row in chronology.receiver_certificates)):
        raise ValueError("Expected the exact ALL-domain fixed-world point chronology")
    certificate = chronology.receiver_certificates[0][1]
    reproduced = certify_native_domain_journal_world_point(chronology.journal,
        chronology.fixed_world_point, precision=certificate.precision, order=certificate.order,
        max_depth=certificate.max_depth, max_cells=certificate.max_cells)
    if (chronology.point_chronology_sha256 != reproduced.point_chronology_sha256
            or _encoded(_world_context_body(chronology)) != _encoded(_world_context_body(reproduced))):
        raise ValueError("Fixed-world domain chronology differs from complete all-fragment replay")
    return chronology


def evaluate_native_domain_journal_world_point(chronology, time_fraction):
    from .geometric_native_point_chronology import evaluate_native_point_chronology
    chronology = validate_native_domain_world_point_chronology(chronology)
    queries = tuple((identity, evaluate_native_point_chronology(certificate, time_fraction))
                    for identity, certificate in chronology.receiver_certificates)
    occupancy, count = _union_occupancy(chronology.journal, chronology.receiver_certificates, queries)
    status = "occupied" if occupancy == "inside" else "raw_vacant" if occupancy == "outside" else "event_window"
    return NativeDomainWorldPointQuery(chronology, time_fraction, queries, status, count,
        chronology.journal.memory_reference)


def native_domain_world_point_chronology_to_dict(chronology):
    chronology = validate_native_domain_world_point_chronology(chronology)
    return dict(_world_context_body(chronology), point_chronology_sha256=chronology.point_chronology_sha256)


def native_domain_world_point_chronology_from_dict(raw, *, journal):
    from .geometric_native_point_chronology import native_point_chronology_from_dict
    if (type(raw) is not dict
            or raw.get("format") != "native-whole-domain-fixed-world-point-chronology-1"):
        raise ValueError("Expected the explicit all-domain fixed-world point chronology")
    try:
        rows = raw["receiver_certificates"]
        if type(rows) is not list or not rows or type(rows[0]) is not dict:
            raise ValueError("Fixed-world chronology must retain every original receiver certificate")
        certificate = native_point_chronology_from_dict(rows[0]["chronology_certificate"])
        result = certify_native_domain_journal_world_point(journal, certificate.point,
            precision=certificate.precision, order=certificate.order,
            max_depth=certificate.max_depth, max_cells=certificate.max_cells)
    except KeyError as error:
        raise ValueError("Incomplete all-domain fixed-world chronology") from error
    if _encoded(raw) != _encoded(native_domain_world_point_chronology_to_dict(result)):
        raise ValueError("Saved fixed-world chronology differs from complete all-domain replay")
    return result


def native_domain_world_point_query_to_dict(query):
    from .geometric_native_point_chronology import native_point_chronology_query_to_dict
    if (type(query) is not NativeDomainWorldPointQuery or type(query.time_fraction) is not Fraction
            or type(query.boolean_sign_assignments_checked) is not int
            or query.actual_passive_archive_vacancy_proved is not False
            or query.newborn_material_created is not False or query.admission_authorized is not False):
        raise ValueError("Fixed-world raw-gap query cannot grant material vacancy or birth admission")
    reproduced = evaluate_native_domain_journal_world_point(query.chronology, query.time_fraction)
    if reproduced != query:
        raise ValueError("Fixed-world raw-gap query differs from complete all-fragment replay")
    return dict(format="native-whole-domain-fixed-world-point-query-1", model=MODEL,
        journal_sha256=query.chronology.journal.journal_sha256,
        point_chronology_sha256=query.chronology.point_chronology_sha256,
        time_fraction=_rational(query.time_fraction), raw_gap_status=query.raw_gap_status,
        boolean_sign_assignments_checked=query.boolean_sign_assignments_checked,
        logical_coverage_budget_exhausted=query.boolean_sign_assignments_checked == 0,
        maximum_boolean_sign_assignments=4096,
        union_proof="exhaustive_compatible_shared_plane_sign_assignments; opposite_owner_planes_exact_complements",
        receiver_queries=[dict(receiver_fragment_id=identity,
            point_query=native_point_chronology_query_to_dict(item)) for identity, item in query.receiver_queries],
        memory_reference=_memory_body(query.memory_reference),
        actual_passive_archive_vacancy_proved=False, newborn_material_created=False,
        admission_authorized=False, global_inventory_updated=False)


def _union_occupancy(journal, certificates, queries, *, max_assignments=4096):
    """Exact logical union of closed halfspaces with shared sign variables.

    All planes on one owner have the same complete relative rotation.  Equal
    primitive normals therefore share one scalar sign and opposite normals
    negate it exactly.  Exhaustive assignments to unresolved signs prove
    coverage without choosing a time or discarding an internal boundary.
    """
    allowed, clauses = {}, []
    query_by_id = dict(queries)
    for identity, certificate in certificates:
        owner = _source_fragment(journal, identity).parcel.plate
        query = query_by_id[identity]
        clause = []
        for support, sign in zip(certificate.receiver_polygon.supports, query.plane_signs):
            if query.time_fraction == 0:
                # The source is unposed and its saved point is exact.  The
                # closed initial boundary has a known zero side even when a
                # one-sided crossing query retains its event window at u=0.
                value = sum(normal * coordinate for normal, coordinate in
                            zip(support.normal, certificate.point))
                sign = 1 if value > 0 else -1 if value < 0 else 0
            normal = support.normal
            negative = tuple(-value for value in normal)
            canonical = min(normal, negative)
            orientation = 1 if normal == canonical else -1
            key = (owner, canonical)
            values = {-1, 0, 1} if sign is None else {orientation * sign}
            allowed[key] = allowed.get(key, {-1, 0, 1}).intersection(values)
            if not allowed[key]:
                raise ValueError("Shared original owner plane signs contradict their exact common rotation")
            clause.append((key, orientation))
        clauses.append(tuple(clause))
    keys = tuple(sorted(allowed))
    count = math.prod(len(allowed[key]) for key in keys)
    if count > max_assignments:
        # This is an explicit unresolved logical coverage budget, not a
        # geometric or material sign tolerance.
        return "event_window", 0
    covered = set()
    for signs in product(*(sorted(allowed[key]) for key in keys)):
        assignment = dict(zip(keys, signs))
        covered.add(any(all(orientation * assignment[key] >= 0 for key, orientation in clause)
                        for clause in clauses))
    return ("inside" if covered == {True} else "outside" if covered == {False} else "event_window"), count


def _owner_context_body(chronology):
    from .geometric_native_point_chronology import native_point_chronology_to_dict
    return dict(format="native-whole-domain-owner-point-chronology-1", model=MODEL,
        journal_sha256=chronology.journal.journal_sha256,
        source_surface_sha256=chronology.journal.source_surface_sha256,
        donor_fragment_id=chronology.donor_fragment_id, receiver_owner=chronology.receiver_owner,
        original_point=[_rational(value) for value in chronology.original_point],
        receiver_certificates=[dict(receiver_fragment_id=identity,
            chronology_certificate=native_point_chronology_to_dict(certificate))
            for identity, certificate in chronology.receiver_certificates],
        owner_domain="union_of_ALL_original_fragments_on_named_receiver_owner",
        internal_material_boundaries_are_new_faults=False,
        material_loss_decided=False, admission_authorized=False, global_inventory_updated=False)


def certify_native_domain_journal_owner_point(journal, donor_fragment_id,
        receiver_owner, point, *, precision=256, order=24,
        max_depth=192, max_cells=4096):
    """Bind one original point to the union of a receiver owner's material."""
    from .geometric_native_point_chronology import certify_native_point_chronology
    journal = validate_native_domain_journal(journal)
    donor, point = _source_fragment(journal, donor_fragment_id), _point(point)
    if type(receiver_owner) is not int or receiver_owner == donor.parcel.plate:
        raise ValueError("Receiver owner must name another actual original material owner")
    receivers = tuple(fragment for fragment in journal.source_surface.fragments
                      if fragment.parcel.plate == receiver_owner)
    if not receivers:
        raise ValueError("Named receiver owner has no original source domains")
    omega = dict(journal.omega_rad_per_myr)
    certificates = tuple((receiver.fragment_id, certify_native_point_chronology(
        donor.geometry, receiver.geometry, omega[donor.parcel.plate], omega[receiver_owner],
        journal.dt_myr, point, precision=precision, order=order,
        max_depth=max_depth, max_cells=max_cells)) for receiver in receivers)
    unsealed = NativeDomainOwnerPointChronology(journal, donor.fragment_id,
        receiver_owner, point, certificates, "")
    from dataclasses import replace
    return replace(unsealed, point_chronology_sha256=_hash(_owner_context_body(unsealed)))


def validate_native_domain_owner_point_chronology(chronology):
    from .geometric_native_point_chronology import NativePointChronologyCertificate
    if (type(chronology) is not NativeDomainOwnerPointChronology
            or type(chronology.receiver_owner) is not int
            or type(chronology.original_point) is not tuple
            or any(type(value) is not Fraction for value in chronology.original_point)
            or type(chronology.receiver_certificates) is not tuple
            or not chronology.receiver_certificates
            or any(type(row) is not tuple or len(row) != 2 or type(row[0]) is not str
                or type(row[1]) is not NativePointChronologyCertificate
                for row in chronology.receiver_certificates)):
        raise ValueError("Expected exact ALL-material receiver owner point chronology")
    certificate = chronology.receiver_certificates[0][1]
    reproduced = certify_native_domain_journal_owner_point(chronology.journal,
        chronology.donor_fragment_id, chronology.receiver_owner, chronology.original_point,
        precision=certificate.precision, order=certificate.order,
        max_depth=certificate.max_depth, max_cells=certificate.max_cells)
    if (chronology.point_chronology_sha256 != reproduced.point_chronology_sha256
            or _encoded(_owner_context_body(chronology)) != _encoded(_owner_context_body(reproduced))):
        raise ValueError("Owner point chronology differs from complete all-material domain replay")
    return chronology


def evaluate_native_domain_journal_owner_point(chronology, time_fraction):
    from .geometric_native_point_chronology import evaluate_native_point_chronology
    chronology = validate_native_domain_owner_point_chronology(chronology)
    queries = tuple((identity, evaluate_native_point_chronology(certificate, time_fraction))
                    for identity, certificate in chronology.receiver_certificates)
    occupancy, count = _union_occupancy(chronology.journal, chronology.receiver_certificates, queries)
    return NativeDomainOwnerPointQuery(chronology, time_fraction, queries,
        occupancy, count, chronology.journal.memory_reference)


def native_domain_owner_point_chronology_to_dict(chronology):
    chronology = validate_native_domain_owner_point_chronology(chronology)
    return dict(_owner_context_body(chronology), point_chronology_sha256=chronology.point_chronology_sha256)


def native_domain_owner_point_chronology_from_dict(raw, *, journal):
    from .geometric_native_point_chronology import native_point_chronology_from_dict
    if type(raw) is not dict or raw.get("format") != "native-whole-domain-owner-point-chronology-1":
        raise ValueError("Expected the explicit all-material owner point chronology")
    try:
        rows = raw["receiver_certificates"]
        if type(rows) is not list or not rows or type(rows[0]) is not dict:
            raise ValueError("Owner chronology must retain every original receiver certificate")
        certificate = native_point_chronology_from_dict(rows[0]["chronology_certificate"])
        result = certify_native_domain_journal_owner_point(journal, raw["donor_fragment_id"],
            raw["receiver_owner"], certificate.point, precision=certificate.precision,
            order=certificate.order, max_depth=certificate.max_depth, max_cells=certificate.max_cells)
    except KeyError as error:
        raise ValueError("Incomplete all-material owner point chronology") from error
    if _encoded(raw) != _encoded(native_domain_owner_point_chronology_to_dict(result)):
        raise ValueError("Saved owner point chronology differs from complete all-material replay")
    return result


def native_domain_owner_point_query_to_dict(query):
    from .geometric_native_point_chronology import native_point_chronology_query_to_dict
    if (type(query) is not NativeDomainOwnerPointQuery or type(query.time_fraction) is not Fraction
            or type(query.boolean_sign_assignments_checked) is not int
            or query.material_loss_decided is not False
            or query.new_fault_inferred_from_internal_boundary is not False
            or query.admission_authorized is not False):
        raise ValueError("Owner contact query cannot infer new faults or adopt original material")
    reproduced = evaluate_native_domain_journal_owner_point(query.chronology, query.time_fraction)
    if query != reproduced:
        raise ValueError("Owner contact query differs from exact all-material logical union replay")
    return dict(format="native-whole-domain-owner-point-query-1", model=MODEL,
        journal_sha256=query.chronology.journal.journal_sha256,
        point_chronology_sha256=query.chronology.point_chronology_sha256,
        donor_fragment_id=query.chronology.donor_fragment_id,
        receiver_owner=query.chronology.receiver_owner, time_fraction=_rational(query.time_fraction),
        contact_occupancy=query.contact_occupancy,
        receiver_queries=[dict(receiver_fragment_id=identity,
            point_query=native_point_chronology_query_to_dict(item)) for identity, item in query.receiver_queries],
        boolean_sign_assignments_checked=query.boolean_sign_assignments_checked,
        logical_coverage_budget_exhausted=query.boolean_sign_assignments_checked == 0,
        maximum_boolean_sign_assignments=4096,
        union_proof="exhaustive_compatible_shared_plane_sign_assignments; opposite_owner_planes_exact_complements",
        memory_reference=_memory_body(query.memory_reference),
        material_loss_decided=False, new_fault_inferred_from_internal_boundary=False,
        admission_authorized=False, global_inventory_updated=False)
