"""Passive slab cohorts accepted from actual geometric material transactions.

This inventory never removes material, selects a polarity, or supplies forces.
Its attachment is evidence of a local shared arc, not a mechanical breakoff or
subduction-initiation law. A cohort spanning several arcs retains its material
once; allocating it laterally to force sections is deliberately unresolved.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math

import numpy as np

from .fractional_surface import EXTENSIVE_FIELDS, SurfaceParcel
from .geometric_contacts import GeometricContact, extract_contacts
from .geometric_surface import GeometricFragment, GeometricSurfaceState, rotations
from .geometric_transport import GeometricLoss
from .spherical_polygons import arc_length, intersect_convex, polygon_area, shared_boundary_arcs

VERSION = "geometric-boundary-1"
_EPS = np.finfo(float).eps
_STATUSES = {"geometric_contact", "unresolved_attachment", "unresolved_connectivity",
             "unresolved_lateral_allocation"}


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def surface_digest(surface):
    """Bind the archive to an exact surface, independent of tuple iteration order."""
    data = asdict(surface)
    data["fragments"] = sorted(data["fragments"], key=lambda row: row["fragment_id"])
    return _hash(data)


@dataclass(frozen=True, slots=True)
class GeometricSupport:
    contact_id: str
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    receiver_fragment_id: str
    receiver_material_id: str
    subducting_plate: int
    overriding_plate: int
    lineage_ids: tuple[str, ...]
    parent_contact_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GeometricConnectivityChange:
    time_myr: float
    reason: str
    previous_supports: tuple[GeometricSupport, ...]
    current_supports: tuple[GeometricSupport, ...]
    uncovered_length_km: float = 0.


@dataclass(frozen=True, slots=True)
class GeometricSlabCohort:
    event_id: str
    payload_sha256: str
    loss: GeometricLoss
    supports: tuple[GeometricSupport, ...]
    attachment_status: str
    connectivity_reason: str
    unresolved_support: bool = False
    connectivity_history: tuple[GeometricConnectivityChange, ...] = ()

    @property
    def acceptance_time_myr(self):
        return self.loss.time_myr

    @property
    def parcel(self):
        return self.loss.fragment.parcel

    @property
    def initial_thickness_km(self):
        return self.parcel.cold_mantle_volume_km3/self.parcel.area_km2

    @property
    def subducting_plate(self):
        return self.parcel.plate

    @property
    def overriding_plate(self):
        return self.loss.receiver_plate


@dataclass(frozen=True, slots=True)
class GeometricTransactionRecord:
    transaction_id: str
    payload_sha256: str
    before_surface_digest: str
    after_surface_digest: str
    start_time_myr: float
    end_time_myr: float
    event_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GeometricBoundaryState:
    time_myr: float
    radius_km: float
    surface_digest: str
    cohorts: tuple[GeometricSlabCohort, ...]
    contacts: tuple[GeometricContact, ...]
    transactions: tuple[GeometricTransactionRecord, ...] = ()
    version: str = VERSION


def _event_identity(loss):
    return "geometric-loss:"+_hash((float(loss.time_myr).hex(), loss.source_fragment_id,
        loss.fragment.fragment_id, loss.receiver_fragment_id))


def _loss_payload(loss):
    data = asdict(loss)
    data["contact_ids"] = sorted(data["contact_ids"])
    return data


def _contact_geometry(contact):
    return {name: value for name, value in asdict(contact).items() if name not in {
        "normal_velocity_km_per_myr", "tangential_velocity_km_per_myr", "normal_area_rate_km2_per_myr"}}


def _validate_contacts(surface, contacts):
    contacts = tuple(contacts)
    if any(not isinstance(contact, GeometricContact) for contact in contacts):
        raise ValueError("Boundary contact records must be geometric contacts")
    contacts = tuple(sorted(contacts, key=lambda contact: contact.contact_id))
    if len({contact.contact_id for contact in contacts}) != len(contacts):
        raise ValueError("Boundary contact IDs must be unique")
    expected = {contact.contact_id: contact for contact in extract_contacts(surface.fragments, surface.radius_km)}
    if set(expected) != {contact.contact_id for contact in contacts}:
        raise ValueError("Boundary contact manifest must contain every actual current contact")
    for contact in contacts:
        actual = expected.get(contact.contact_id)
        if actual is None:
            raise ValueError("Boundary contact has no actual local shared arc")
        supplied, measured = _contact_geometry(contact), _contact_geometry(actual)
        for name, wanted in measured.items():
            value = supplied[name]
            if isinstance(wanted, (str, int)):
                valid = value == wanted
            else:
                valid = np.allclose(value, wanted, rtol=128.*_EPS, atol=16.*_EPS)
            if not valid:
                raise ValueError(f"Boundary contact {name} disagrees with its actual geometry")
        if not all(math.isfinite(getattr(contact, name)) for name in (
                "normal_velocity_km_per_myr", "tangential_velocity_km_per_myr", "normal_area_rate_km2_per_myr")):
            raise ValueError("Boundary contact velocities must be finite")
    return contacts


def initialize_boundary(surface, contacts=None):
    """Create an empty inventory at an explicitly supplied resolved surface.

    Historical removed material is not reconstructed here. A caller importing
    an older archive must migrate those losses explicitly instead of pretending
    this empty state contains them.
    """
    if not isinstance(surface, GeometricSurfaceState) or surface.phase != "partition":
        raise ValueError("Boundary inventory requires a resolved geometric partition")
    contacts = extract_contacts(surface.fragments, surface.radius_km) if contacts is None else contacts
    return GeometricBoundaryState(surface.time_myr, surface.radius_km, surface_digest(surface),
        (), _validate_contacts(surface, contacts))


def _rotate_vectors(values, omega, dt):
    values = np.asarray(values, dtype=np.longdouble)
    omega = np.asarray(omega, dtype=np.longdouble)
    magnitude = np.sqrt(np.sum(omega*omega))
    if magnitude == 0.:
        return np.asarray(values, dtype=float)
    axis, angle = omega/magnitude, magnitude*np.longdouble(dt)
    rotated = (values*np.cos(angle)+np.cross(axis, values)*np.sin(angle)
        +(values@axis)[..., None]*axis*(2.*np.sin(angle/2.)**2))
    rotated /= np.sqrt(np.sum(rotated*rotated, axis=-1))[..., None]
    return np.asarray(rotated, dtype=float)


def _contact_frame(contact):
    origin = np.asarray(contact.start)
    tangent = np.cross(contact.normal_a_to_b, origin)
    tangent /= np.linalg.norm(tangent)
    if float(tangent@(np.asarray(contact.end)-origin)) < 0.:
        tangent = -tangent
    return origin, tangent, arc_length(contact.start, contact.end)


def _arc_interval(start, end, contact):
    """A positive common interval on a known contact plane; no distance search."""
    origin, tangent, extent = _contact_frame(contact)
    normal = np.asarray(contact.normal_a_to_b)
    points = [np.asarray(start, dtype=float), np.asarray(end, dtype=float)]
    endpoints = (np.asarray(contact.start), np.asarray(contact.end))
    for index, point in enumerate(points):
        # Same binary64 construction as the polygon kernel's vertex filter.
        # This resolves shared endpoints, never a physical minimum arc length.
        for endpoint in endpoints:
            if np.array_equal(point, endpoint):
                points[index] = endpoint
                break
        else:
            # Never merge the two distinct endpoints of a representable short
            # contact merely because both lie in one binary64 error ball.
            if extent > 8.*_EPS:
                nearest = min(endpoints, key=lambda endpoint: np.linalg.norm(point-endpoint))
                if np.linalg.norm(point-nearest) <= 4.*_EPS:
                    points[index] = nearest
        point = points[index]
        error_bound = 64.*_EPS*float(np.sum(np.abs(point*normal)))
        if abs(float(point@normal)) > error_bound:
            return None
    angles = []
    for point in points:
        if np.array_equal(point, endpoints[0]):
            angle = 0.
        elif np.array_equal(point, endpoints[1]):
            angle = extent
        else:
            angle = math.atan2(float(tangent@(point-origin)), float(origin@point))
        angles.append(angle)
    a, b = angles
    if b-a > math.pi:
        b -= 2.*math.pi
    elif b-a < -math.pi:
        b += 2.*math.pi
    for shift in (-2.*math.pi, 0., 2.*math.pi):
        low, high = max(0., min(a, b)+shift), min(extent, max(a, b)+shift)
        if high > low:
            return low, high
    return None


def _interval_points(contact, low, high, *, source_points=()):
    origin, tangent, extent = _contact_frame(contact)
    candidates = [(0., tuple(contact.start)), (extent, tuple(contact.end))]
    for value in source_points:
        value = np.asarray(value)
        angle = math.atan2(float(tangent@(value-origin)), float(origin@value))
        candidates.extend((angle+shift, tuple(map(float, value)))
                          for shift in (-2.*math.pi, 0., 2.*math.pi))
    def point(angle):
        if angle == 0.:
            return tuple(contact.start)
        if angle == extent:
            return tuple(contact.end)
        if source_points:
            # An interval endpoint is an existing endpoint of one input arc.
            # Reconstructing a tiny interval with sine/cosine can round both
            # ends to one point or move a support off its known plane.
            return min(candidates, key=lambda item: abs(item[0]-angle))[1]
        value = math.cos(angle)*origin+math.sin(angle)*tangent
        value /= np.linalg.norm(value)
        return tuple(map(float, value))
    return point(low), point(high)


def _receiver(contact, overriding_plate):
    if contact.plate_a == overriding_plate:
        return contact.fragment_a, contact.material_a
    if contact.plate_b == overriding_plate:
        return contact.fragment_b, contact.material_b
    raise ValueError("Geometric support owner is absent from its contact")


def _merge_supports(candidates, contacts):
    """Union coincident support intervals before counting their length or mass."""
    grouped = {}
    for support in candidates:
        contact = contacts[support.contact_id]
        interval = _arc_interval(support.start, support.end, contact)
        if interval is None:
            raise ValueError("Saved support does not lie on its stated current contact")
        grouped.setdefault(support.contact_id, []).append((*interval, support))
    output = []
    for identity, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: (row[0], row[1], row[2].lineage_ids))
        batches = []
        for low, high, support in rows:
            if batches and low <= batches[-1][1]:
                batches[-1][1] = max(batches[-1][1], high)
                batches[-1][2].append(support)
            else:
                batches.append([low, high, [support]])
        contact = contacts[identity]
        for low, high, members in batches:
            base = members[0]
            start, end = _interval_points(contact, low, high,
                source_points=tuple(point for member in members for point in (member.start, member.end)))
            if start == end:
                continue
            output.append(replace(base, start=start, end=end,
                lineage_ids=tuple(sorted({key for member in members for key in member.lineage_ids})),
                parent_contact_ids=tuple(sorted({key for member in members for key in member.parent_contact_ids}))))
    return tuple(output)


def _initial_supports(loss, event_id, fragments, contacts):
    if loss.attachment_status not in ("geometric_contact", "unresolved_attachment"):
        raise ValueError("Unknown geometric loss attachment status")
    if bool(loss.contact_ids) != (loss.attachment_status == "geometric_contact"):
        raise ValueError("Geometric loss status disagrees with its support IDs")
    if len(set(loss.contact_ids)) != len(loss.contact_ids):
        raise ValueError("Geometric loss repeats the same support ID")
    output, unresolved_claim = [], False
    for identity in sorted(loss.contact_ids):
        contact = contacts.get(identity)
        if contact is None:
            raise ValueError("Accepted loss refers to an absent local contact")
        if {contact.plate_a, contact.plate_b} != {loss.fragment.parcel.plate, loss.receiver_plate}:
            raise ValueError("Accepted loss and contact have different plate owners")
        receiver, material = _receiver(contact, loss.receiver_plate)
        if receiver != loss.receiver_fragment_id:
            raise ValueError("Accepted loss contact belongs to a different receiver")
        donor_id = contact.fragment_b if receiver == contact.fragment_a else contact.fragment_a
        donor = fragments[donor_id]
        if (donor.parcel.material_id != loss.fragment.parcel.material_id or
                (donor.fragment_id != loss.source_fragment_id and donor.parent_fragment_id != loss.source_fragment_id)):
            raise ValueError("Accepted loss contact is not adjacent to its actual donor descendant")
        arcs = shared_boundary_arcs(loss.fragment.polygon, donor.polygon)
        supported = False
        for start, end in arcs:
            interval = _arc_interval(start, end, contact)
            if interval is None:
                continue
            supported = True
            points = _interval_points(contact, *interval, source_points=(start, end))
            if points[0] == points[1]:
                unresolved_claim = True
                continue
            lineage = "support:"+_hash((event_id, identity))
            output.append(GeometricSupport(identity, *points, receiver, material,
                loss.fragment.parcel.plate, loss.receiver_plate, (lineage,)))
        if not supported:
            # A producer's local candidate may reduce to a shared endpoint at
            # binary64 resolution. Its material transaction remains valid;
            # an unverified mechanical/polarity attachment does not follow.
            unresolved_claim = True
    supports = _merge_supports(output, contacts)
    return supports, unresolved_claim or bool(output and not supports)


def _group_totals(fragments):
    grouped = {}
    for fragment in fragments:
        grouped.setdefault(fragment.parcel.material_id, []).append(fragment.parcel)
    return {identity: {field: math.fsum(getattr(parcel, field) for parcel in parcels)
            for field in EXTENSIVE_FIELDS} for identity, parcels in grouped.items()}


def _verify_transaction(before, result, inventory, omega, dt):
    dt = float(dt)
    omega = rotations(before, omega, dt)
    after = result.state
    if (before.phase != "partition" or after.phase != "partition" or
            before.radius_km != after.radius_km or before.radius_km != inventory.radius_km):
        raise ValueError("Slab transaction requires matching resolved surface geometry")
    if before.time_myr != inventory.time_myr or surface_digest(before) != inventory.surface_digest:
        raise ValueError("Slab inventory does not belong to the supplied before surface")
    if after.time_myr <= inventory.time_myr or after.time_myr != before.time_myr+dt:
        raise ValueError("Slab transaction time must advance exactly once by the supplied step")
    old = {fragment.fragment_id: fragment for fragment in before.fragments}
    fragments = {fragment.fragment_id: fragment for fragment in after.fragments}
    birth_ids = [fragment.fragment_id for fragment in result.births]
    if len(birth_ids) != len(set(birth_ids)):
        raise ValueError("Slab transaction repeats a birth fragment")
    for birth in result.births:
        if (fragments.get(birth.fragment_id) != birth or birth.parcel.material_id in before.known_material_ids):
            raise ValueError("Slab transaction births must be fresh material in the after surface")
    losses = tuple(result.losses)
    event_ids = [_event_identity(loss) for loss in losses]
    known_events = {cohort.event_id: cohort.payload_sha256 for cohort in inventory.cohorts}
    if len(set(event_ids)) != len(event_ids) or any(identity in known_events for identity in event_ids):
        raise ValueError("Duplicate geometric slab acceptance event; replay rejected")
    old_material = {}
    for fragment in before.fragments:
        old_material.setdefault(fragment.parcel.material_id, []).append(fragment.parcel)
    for fragment in after.fragments:
        if fragment.fragment_id in birth_ids:
            continue
        candidates = old_material.get(fragment.parcel.material_id, ())
        parcel = fragment.parcel
        if not any((parcel.plate == source.plate and parcel.material_fields == source.material_fields
                    and parcel.specific_properties == source.specific_properties
                    and parcel.age_myr == source.age_myr+dt) for source in candidates):
            raise ValueError("Passive geometric transaction changed a retained material history")
    by_source = {}
    for loss in losses:
        if not isinstance(loss, GeometricLoss) or loss.source_fragment_id not in old:
            raise ValueError("Slab acceptance requires an actual source fragment")
        if loss.time_myr != after.time_myr:
            raise ValueError("Endpoint geometric loss has the wrong acceptance time")
        source = old[loss.source_fragment_id]
        parcel = loss.fragment.parcel
        if (loss.fragment.fragment_id in fragments or
                loss.fragment.parent_fragment_id != source.fragment_id or
                parcel.material_id != source.parcel.material_id or parcel.plate != source.parcel.plate or
                parcel.material_fields != source.parcel.material_fields or
                parcel.specific_properties != source.parcel.specific_properties or
                parcel.age_myr != source.parcel.age_myr+dt):
            raise ValueError("Slab acceptance does not preserve its actual donor history")
        receiver = fragments.get(loss.receiver_fragment_id)
        if receiver is None or receiver.parcel.plate != loss.receiver_plate or receiver.parcel.plate == parcel.plate:
            raise ValueError("Slab acceptance needs its actual surviving receiver")
        overlap = intersect_convex(loss.fragment.polygon, receiver.polygon)
        if not math.isclose(polygon_area(overlap), polygon_area(loss.fragment.polygon), rel_tol=2e-10, abs_tol=0.):
            raise ValueError("Slab loss is not contained in its actual receiver footprint")
        moved_source = _rotate_vectors(source.polygon, omega[source.parcel.plate], dt)
        overlap = intersect_convex(loss.fragment.polygon, moved_source)
        if not math.isclose(polygon_area(overlap), polygon_area(loss.fragment.polygon), rel_tol=2e-10, abs_tol=0.):
            raise ValueError("Slab loss is not contained in its actual moved source")
        by_source.setdefault(source.fragment_id, []).append(parcel)
    for identity, parcels in by_source.items():
        for field in EXTENSIVE_FIELDS:
            amount = math.fsum(getattr(parcel, field) for parcel in parcels)
            available = getattr(old[identity].parcel, field)
            if amount > available and not math.isclose(amount, available, rel_tol=5e-12, abs_tol=0.):
                raise ValueError("Slab losses exceed their donor's material budget")
    ledgers = [_group_totals(rows) for rows in (before.fragments, after.fragments,
        (loss.fragment for loss in losses), result.births)]
    for identity in set().union(*(ledger for ledger in ledgers)):
        for field in EXTENSIVE_FIELDS:
            initial, remaining, removed, born = [ledger.get(identity, {}).get(field, 0.) for ledger in ledgers]
            if not math.isclose(initial+born, remaining+removed, rel_tol=5e-12, abs_tol=0.):
                raise ValueError(f"Slab transaction violates material {identity} ledger for {field}")
    return omega, losses, fragments


def _transfer_cohort(cohort, before, after, contacts, omega, dt, old_contacts):
    if not cohort.supports:
        return cohort
    old_fragments = {fragment.fragment_id: fragment for fragment in before.fragments}
    new_fragments = {fragment.fragment_id: fragment for fragment in after.fragments}
    possible = [contact for contact in contacts.values()
                if {contact.plate_a, contact.plate_b} == {cohort.subducting_plate, cohort.overriding_plate}]
    candidates, rigidly_transferred = [], 0
    owners = {fragment.parcel.plate for fragment in before.fragments}
    common = all(np.array_equal(omega[owner], omega[cohort.overriding_plate]) for owner in owners)
    total_length = math.fsum(arc_length(support.start, support.end) for support in cohort.supports)
    for support in cohort.supports:
        receiver_before = old_fragments.get(support.receiver_fragment_id)
        if receiver_before is None or receiver_before.parcel.material_id != support.receiver_material_id:
            raise ValueError("Saved support is missing its current receiver material")
        old_contact, current_contact = old_contacts.get(support.contact_id), contacts.get(support.contact_id)
        if common and old_contact is not None and current_contact is not None:
            old_interval = _arc_interval(support.start, support.end, old_contact)
            if (old_interval == (0., arc_length(old_contact.start, old_contact.end)) and
                    _receiver(current_contact, cohort.overriding_plate) ==
                    (support.receiver_fragment_id, support.receiver_material_id) and
                    np.allclose(_rotate_vectors((old_contact.start, old_contact.end),
                        omega[cohort.overriding_plate], dt), (current_contact.start, current_contact.end),
                        rtol=0., atol=32.*_EPS)):
                # Both the material and this full persisted arc undergo the
                # same rigid map. Independent endpoint rounding cannot create
                # a missing strip of historical slab in a common rotation.
                candidates.append(replace(support, start=tuple(current_contact.start), end=tuple(current_contact.end),
                    parent_contact_ids=(support.contact_id,)))
                rigidly_transferred += 1
                continue
        moved_arc = _rotate_vectors((support.start, support.end), omega[cohort.overriding_plate], dt)
        moved_receiver = None
        for contact in possible:
            receiver_id, material_id = _receiver(contact, cohort.overriding_plate)
            if material_id != support.receiver_material_id:
                continue
            interval = _arc_interval(*moved_arc, contact)
            if interval is None:
                continue
            if moved_receiver is None:
                moved_receiver = _rotate_vectors(receiver_before.polygon, omega[cohort.overriding_plate], dt)
            # Equal material labels alone do not establish lineage. The actual
            # transported receiver region must survive at this contact.
            receiver_after = new_fragments[receiver_id]
            if not len(intersect_convex(moved_receiver, receiver_after.polygon)):
                continue
            points = _interval_points(contact, *interval, source_points=tuple(moved_arc))
            if points[0] == points[1]:
                continue
            candidates.append(replace(support, contact_id=contact.contact_id, start=points[0], end=points[1],
                receiver_fragment_id=receiver_id, parent_contact_ids=(support.contact_id,)))
    supports = _merge_supports(candidates, contacts)
    covered_length = math.fsum(arc_length(support.start, support.end) for support in supports)
    tolerance = 512.*_EPS*max(total_length, covered_length)
    missing = 0. if rigidly_transferred == len(cohort.supports) else max(0., total_length-covered_length)
    if missing <= tolerance:
        missing = 0.
    unresolved = cohort.unresolved_support or missing > 0.
    if not supports:
        status, reason = "unresolved_connectivity", "support_disappeared"
    elif unresolved:
        status, reason = "unresolved_lateral_allocation", "partial_support_coverage"
    elif len(supports) > 1:
        status, reason = "unresolved_lateral_allocation", "multiple_local_arcs"
    else:
        status, reason = "geometric_contact", "single_local_arc"
    changed = (status != cohort.attachment_status or
        tuple(s.contact_id for s in supports) != tuple(s.contact_id for s in cohort.supports) or missing > 0.)
    history = cohort.connectivity_history
    if changed:
        history += (GeometricConnectivityChange(after.time_myr, reason, cohort.supports, supports,
            missing*after.radius_km),)
    return replace(cohort, supports=supports, attachment_status=status, connectivity_reason=reason,
        unresolved_support=unresolved, connectivity_history=history)


def consume_geometric_transaction(before, result, inventory, omega, dt):
    """Atomically accept actual losses once and transport passive arc support.

    All four extensive budgets are checked per material history. No caller
    transaction name can bypass replay protection. Contact splitting changes
    the support set of one cohort, never the number of copies of its material.
    The prescribed trench frame is that of its overriding receiver plate.
    """
    if not isinstance(inventory, GeometricBoundaryState) or inventory.version != VERSION:
        raise ValueError("Unsupported geometric boundary inventory")
    omega, losses, fragments = _verify_transaction(before, result, inventory, omega, dt)
    contacts = _validate_contacts(result.state, result.contacts)
    by_contact = {contact.contact_id: contact for contact in contacts}
    old_contacts = {contact.contact_id: contact for contact in inventory.contacts}
    cohorts = [_transfer_cohort(cohort, before, result.state, by_contact, omega, float(dt), old_contacts)
               for cohort in inventory.cohorts]
    for loss in sorted(losses, key=_event_identity):
        event_id = _event_identity(loss)
        supports, unresolved_claim = _initial_supports(loss, event_id, fragments, by_contact)
        status = ("unresolved_attachment" if not supports else
                  "geometric_contact" if len(supports) == 1 and not unresolved_claim else "unresolved_lateral_allocation")
        reason = ("incomplete_initial_local_support" if unresolved_claim else
                  "no_initial_local_arc" if not supports else
                  "single_local_arc" if len(supports) == 1 else "multiple_local_arcs")
        cohorts.append(GeometricSlabCohort(event_id, _hash(_loss_payload(loss)), loss, supports, status, reason,
            unresolved_support=unresolved_claim))
    after_digest = surface_digest(result.state)
    payload = dict(before_surface_digest=inventory.surface_digest, after_surface_digest=after_digest,
        omega=np.asarray(omega).tolist(), dt=float(dt),
        losses=sorted((_loss_payload(loss) for loss in losses), key=lambda row: row["fragment"]["fragment_id"]),
        births=sorted((asdict(fragment) for fragment in result.births), key=lambda row: row["fragment_id"]),
        contacts=[asdict(contact) for contact in contacts])
    digest = _hash(payload)
    record = GeometricTransactionRecord("geometric-transaction:"+digest, digest,
        inventory.surface_digest, after_digest, before.time_myr, result.state.time_myr,
        tuple(sorted(_event_identity(loss) for loss in losses)))
    if any(item.transaction_id == record.transaction_id for item in inventory.transactions):
        raise ValueError("Duplicate geometric slab transaction; replay rejected")
    return GeometricBoundaryState(result.state.time_myr, result.state.radius_km, after_digest,
        tuple(cohorts), contacts, inventory.transactions+(record,))


def polarity_evidence(inventory):
    """Return all unambiguous local history evidence; preserve opposing claims.

    Multiple records on one arc are intentionally not reduced by sort order.
    A polarity resolver must diagnose conflicting opposite accepted histories.
    """
    contacts = {contact.contact_id: contact for contact in inventory.contacts}
    def covers_contact(cohort):
        support = cohort.supports[0]
        contact = contacts[support.contact_id]
        interval = _arc_interval(support.start, support.end, contact)
        return interval == (0., arc_length(contact.start, contact.end))
    return [dict(contact_id=cohort.supports[0].contact_id,
        subducting_plate=cohort.subducting_plate, overriding_plate=cohort.overriding_plate,
        evidence_id=cohort.event_id, provenance="accepted_geometric_slab") for cohort in inventory.cohorts
        if cohort.attachment_status == "geometric_contact" and len(cohort.supports) == 1
        and not cohort.unresolved_support and covers_contact(cohort)]


def boundary_diagnostics(inventory, *, thermal_diffusivity_m2_s=None):
    """Archival extensive values; optional passive warming never edits them."""
    groups = {status: [cohort for cohort in inventory.cohorts if cohort.attachment_status == status]
              for status in sorted(_STATUSES)}
    extensive = lambda rows: {field: math.fsum(getattr(cohort.parcel, field) for cohort in rows)
                             for field in EXTENSIVE_FIELDS}
    output = dict(version=inventory.version, time_myr=inventory.time_myr,
        cohort_count=len(inventory.cohorts), transaction_count=len(inventory.transactions),
        accepted=extensive(inventory.cohorts), by_attachment={key: extensive(rows) for key, rows in groups.items()},
        attachment_counts={key: len(rows) for key, rows in groups.items()}, forces_active=False,
        thermal_evolution="archival values frozen; optional passive warming diagnostic only")
    if thermal_diffusivity_m2_s is not None:
        from .young_boundary import slab_thermal_deficit_fraction
        output["passive_current_density_excess_mass_kg"] = math.fsum(
            cohort.parcel.density_excess_mass_kg*slab_thermal_deficit_fraction(
                inventory.time_myr-cohort.acceptance_time_myr, cohort.initial_thickness_km,
                thermal_diffusivity_m2_s) for cohort in inventory.cohorts)
    return output


def boundary_to_dict(inventory):
    """JSON-compatible state including loss payloads, lineage and replay registry."""
    return json.loads(json.dumps(asdict(inventory), allow_nan=False))


def _support_from_dict(row):
    return GeometricSupport(**{**row, "start": tuple(row["start"]), "end": tuple(row["end"]),
        "lineage_ids": tuple(row["lineage_ids"]), "parent_contact_ids": tuple(row["parent_contact_ids"])})


def _contact_from_dict(row):
    vectors = {"start", "end", "midpoint", "tangent", "normal_a_to_b",
               "integrated_position_unit_km", "moment_arm_cross_normal_km2"}
    return GeometricContact(**{key: tuple(value) if key in vectors else value for key, value in row.items()})


def _loss_from_dict(row):
    fragment = row["fragment"]
    material = SurfaceParcel(**fragment["parcel"])
    geometry = GeometricFragment(fragment["fragment_id"], tuple(map(tuple, fragment["polygon"])),
        material, fragment["parent_fragment_id"])
    values = {**row, "fragment": geometry, "contact_ids": tuple(row["contact_ids"])}
    if "polarity_evidence_ids" in values:
        values["polarity_evidence_ids"] = tuple(values["polarity_evidence_ids"])
    return GeometricLoss(**values)


def boundary_from_dict(data):
    """Read a checkpoint without renormalizing saved coordinates or arc endpoints.

    The joint checkpoint's outer digest authenticates the complete state.
    This reader also validates the internal acceptance and transaction registry.
    """
    if not isinstance(data, dict) or data.get("version") != VERSION:
        raise ValueError("Unsupported geometric boundary checkpoint")
    try:
        cohorts = []
        for row in data["cohorts"]:
            changes = tuple(GeometricConnectivityChange(**{**change,
                "previous_supports": tuple(_support_from_dict(item) for item in change["previous_supports"]),
                "current_supports": tuple(_support_from_dict(item) for item in change["current_supports"])})
                for change in row["connectivity_history"])
            cohorts.append(GeometricSlabCohort(**{**row, "loss": _loss_from_dict(row["loss"]),
                "supports": tuple(_support_from_dict(item) for item in row["supports"]),
                "connectivity_history": changes}))
        contacts = tuple(_contact_from_dict(row) for row in data["contacts"])
        transactions = tuple(GeometricTransactionRecord(**{**row, "event_ids": tuple(row["event_ids"])})
            for row in data["transactions"])
        inventory = GeometricBoundaryState(**{**data, "cohorts": tuple(cohorts),
            "contacts": contacts, "transactions": transactions})
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError("Malformed geometric boundary checkpoint") from error
    if (not math.isfinite(inventory.time_myr) or inventory.time_myr < 0. or
            not math.isfinite(inventory.radius_km) or inventory.radius_km <= 0. or
            not isinstance(inventory.surface_digest, str) or len(inventory.surface_digest) != 64):
        raise ValueError("Invalid geometric boundary checkpoint clock or surface digest")
    contact_map = {contact.contact_id: contact for contact in contacts}
    if len(contact_map) != len(contacts):
        raise ValueError("Duplicate saved boundary contact")
    event_map = {}
    for cohort in cohorts:
        if cohort.event_id in event_map:
            raise ValueError("Duplicate saved acceptance event")
        event_map[cohort.event_id] = cohort
        if (cohort.event_id != _event_identity(cohort.loss) or
                cohort.payload_sha256 != _hash(_loss_payload(cohort.loss))):
            raise ValueError("Saved acceptance event identity or payload digest mismatch")
        if (cohort.attachment_status not in _STATUSES or cohort.acceptance_time_myr > inventory.time_myr or
                not math.isfinite(cohort.acceptance_time_myr) or cohort.acceptance_time_myr < 0. or
                not isinstance(cohort.connectivity_reason, str) or not cohort.connectivity_reason or
                (bool(cohort.supports) != (cohort.attachment_status in
                    {"geometric_contact", "unresolved_lateral_allocation"})) or
                (cohort.attachment_status == "geometric_contact" and
                 (len(cohort.supports) != 1 or cohort.unresolved_support))):
            raise ValueError("Invalid saved cohort attachment state")
        for support in cohort.supports:
            contact = contact_map.get(support.contact_id)
            points = np.asarray((support.start, support.end))
            if (points.shape != (2, 3) or not np.isfinite(points).all() or
                    not np.allclose(np.linalg.norm(points, axis=1), 1., rtol=0., atol=16.*_EPS) or
                    not (0. < arc_length(*points) < math.pi) or
                    contact is None or _receiver(contact, cohort.overriding_plate) !=
                    (support.receiver_fragment_id, support.receiver_material_id) or
                    support.subducting_plate != cohort.subducting_plate or
                    support.overriding_plate != cohort.overriding_plate or not support.lineage_ids or
                    any(not isinstance(identity, str) or not identity for identity in
                        (*support.lineage_ids, *support.parent_contact_ids)) or
                    _arc_interval(support.start, support.end, contact) is None):
                raise ValueError("Saved cohort support does not match its current contact")
    seen, transaction_ids = set(), set()
    previous = None
    for record in transactions:
        if (record.transaction_id in transaction_ids or record.transaction_id != "geometric-transaction:"+record.payload_sha256
                or not math.isfinite(record.start_time_myr) or not math.isfinite(record.end_time_myr)
                or record.start_time_myr < 0. or record.end_time_myr <= record.start_time_myr
                or record.end_time_myr > inventory.time_myr):
            raise ValueError("Invalid saved transaction identity or time")
        transaction_ids.add(record.transaction_id)
        if previous is not None and (record.start_time_myr != previous.end_time_myr or
                record.before_surface_digest != previous.after_surface_digest):
            raise ValueError("Saved slab transactions do not form a contiguous surface history")
        for identity in record.event_ids:
            if identity in seen or identity not in event_map or event_map[identity].acceptance_time_myr != record.end_time_myr:
                raise ValueError("Saved transaction acceptance registry is inconsistent")
            seen.add(identity)
        previous = record
    if seen != set(event_map):
        raise ValueError("Saved cohorts are absent from the acceptance registry")
    if previous is not None and (previous.end_time_myr != inventory.time_myr or
            previous.after_surface_digest != inventory.surface_digest):
        raise ValueError("Saved boundary is not at its final transaction surface")
    return inventory
