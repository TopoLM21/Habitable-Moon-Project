"""Local once-only passive material history on a certified native donor patch.

The sweep proves the history of a positive TWO dimensional original donor
domain.  This module explicitly extends the endpoint passive archive closure
to each material point's first entry.  That temporal extension is a declared
local law, not an inherited continuous archive law.  Later exit and reentry change contact
occupancy only: they never resurrect or allocate that material a second time.

The first-entry time varies over the patch.  Transfers therefore use implicit
material subsets and certified time windows, not a fictitious common timestamp.
Reservoir amounts remain symbolic native area ratios with rational enclosures;
even a share below the smallest binary64 number remains positive.  Material
age is an exact rational query.  Mechanical payloads are immutable references,
not an assignment of quadrature work to a two dimensional volume.

This is a local component.  It neither updates NativeBuriedRegistry nor grants
global force admission: complete swept domains, newborn material and their
physical pairs still need their own laws and proofs.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from numbers import Integral, Real

from .fractional_surface import EXTENSIVE_FIELDS
from .geometric_native_pose import FractionInterval as I
from .geometric_native_polygons import (
    native_polygon_from_dict, native_polygon_to_dict, native_subtract,
    validate_native_polygon,
)
from .geometric_native_surface import NativeFragment, _parcel_from_dict


FORMAT = "native-original-donor-patch-material-history-1"
MODEL = "once_only_passive_archive_first_entry"


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _fraction(value, name):
    if isinstance(value, bool):
        raise ValueError(name + " must be an explicit finite number")
    if type(value) is Fraction:
        return value
    if isinstance(value, Integral):
        return Fraction(int(value))
    if isinstance(value, Real) and math.isfinite(float(value)):
        return Fraction.from_float(float(value))
    raise ValueError(name + " must be an explicit finite number")


def _rational(value):
    return [value.numerator, value.denominator]


def _read_rational(raw):
    if (not isinstance(raw, list) or len(raw) != 2
            or any(type(value) is not int for value in raw) or raw[1] <= 0):
        raise ValueError("Material history requires a canonical exact rational")
    value = Fraction(*raw)
    if _rational(value) != raw:
        raise ValueError("Material history rational is not canonical")
    return value


def _interval(value):
    return dict(lo=_rational(value.lo), hi=_rational(value.hi))


def _sha(value, name):
    if (type(value) is not str or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(name + " requires a lowercase SHA256 reference")
    return value


def _parcel_body(parcel):
    from dataclasses import asdict
    return asdict(parcel)


def _fragment_body(fragment):
    return dict(fragment_id=fragment.fragment_id,
        geometry=native_polygon_to_dict(fragment.geometry),
        parcel=_parcel_body(fragment.parcel),
        parent_fragment_id=fragment.parent_fragment_id)


def _checked_fragment(fragment):
    if type(fragment) is not NativeFragment:
        raise ValueError("Material history needs an authoritative native parent fragment")
    geometry = validate_native_polygon(fragment.geometry)
    parcel = _parcel_from_dict(json.loads(_encoded(_parcel_body(fragment.parcel))))
    reconstructed = NativeFragment(fragment.fragment_id, geometry, parcel,
                                   fragment.parent_fragment_id)
    if _encoded(_fragment_body(fragment)) != _encoded(_fragment_body(reconstructed)):
        raise ValueError("Material parent payload changed during typed reconstruction")
    return reconstructed


def _fragment_from_body(raw):
    if (type(raw) is not dict
            or set(raw) != {"fragment_id", "geometry", "parcel", "parent_fragment_id"}):
        raise ValueError("Invalid material-history native parent payload")
    return _checked_fragment(NativeFragment(raw["fragment_id"],
        native_polygon_from_dict(raw["geometry"]), _parcel_from_dict(raw["parcel"]),
        raw["parent_fragment_id"]))


@dataclass(frozen=True, slots=True)
class NativeMaterialMemoryReference:
    """Exact external mechanical payloads, without material work allocation.

    These references bind the component to the caller's before state and
    unaccepted candidate.  The caller must authenticate that physical solve;
    the material-history replay verifies preservation of these payloads only.
    """
    before_payload_json: bytes
    candidate_after_payload_json: bytes
    before_sha256: str
    candidate_after_sha256: str
    constitutive_path_sha256: str


def make_native_material_memory_reference(before_payload, candidate_after_payload,
                                          *, constitutive_path_sha256):
    """Freeze canonical mechanical evidence; never adopt its candidate state."""
    if type(before_payload) is not dict or type(candidate_after_payload) is not dict:
        raise ValueError("Mechanical memory references require explicit JSON objects")
    first, last = _encoded(before_payload), _encoded(candidate_after_payload)
    return NativeMaterialMemoryReference(first, last,
        hashlib.sha256(first).hexdigest(), hashlib.sha256(last).hexdigest(),
        _sha(constitutive_path_sha256, "Constitutive path"))


def _checked_memory(reference):
    if type(reference) is not NativeMaterialMemoryReference:
        raise ValueError("Material history requires immutable mechanical memory references")
    if (type(reference.before_payload_json) is not bytes
            or type(reference.candidate_after_payload_json) is not bytes):
        raise ValueError("Mechanical references require frozen canonical bytes")
    try:
        result = make_native_material_memory_reference(
            json.loads(reference.before_payload_json),
            json.loads(reference.candidate_after_payload_json),
            constitutive_path_sha256=reference.constitutive_path_sha256)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid mechanical memory payload") from error
    if result != reference:
        raise ValueError("Mechanical memory reference or exact payload changed")
    return reference


def _memory_body(reference):
    return dict(before_payload=json.loads(reference.before_payload_json),
        candidate_after_payload=json.loads(reference.candidate_after_payload_json),
        before_sha256=reference.before_sha256,
        candidate_after_sha256=reference.candidate_after_sha256,
        constitutive_path_sha256=reference.constitutive_path_sha256)


@dataclass(frozen=True, slots=True)
class NativeMaterialShare:
    reservoir: str
    parent_amount: Fraction
    native_area_ratio_bounds: I
    amount_bounds: I
    numerator_geometry_sha256: str
    denominator_geometry_sha256: str
    expression: str = "parent_amount * native_area(patch) / native_area(parent)"


@dataclass(frozen=True, slots=True)
class NativeMaterialHistoryEvent:
    sequence: int
    kind: str
    time_fraction: I
    relative_time_myr: I
    absolute_time_myr: I
    operation: str
    cohort_id: str
    material_time_function: str = "unique_sweep_root_for_each_original_donor_point"


@dataclass(frozen=True, slots=True)
class NativeMaterialHistory:
    parent: NativeFragment
    receiver_parent: NativeFragment
    sweep_certificate: object
    memory_reference: NativeMaterialMemoryReference
    material_epoch_myr: Fraction
    parent_payload_sha256: str
    receiver_payload_sha256: str
    patch_geometry_sha256: str
    sweep_sha256: str
    cohort_id: str
    shares: tuple[NativeMaterialShare, ...]
    events: tuple[NativeMaterialHistoryEvent, ...]
    history_sha256: str


@dataclass(frozen=True, slots=True)
class NativeMaterialAllocation:
    reservoir: str
    live_amount_bounds: I
    archived_amount_bounds: I
    live_expression: str
    archived_expression: str
    conserved_total: NativeMaterialShare
    partition_identity: str = "live_subset disjoint_union archived_subset = original_patch"


@dataclass(frozen=True, slots=True)
class NativeMaterialHistorySnapshot:
    history: NativeMaterialHistory
    history_sha256: str
    time_fraction: Fraction
    time_myr: Fraction
    material_age_myr: Fraction
    contact_occupancy: str
    material_allocation: str
    archive_cohort_id: str
    allocations: tuple[NativeMaterialAllocation, ...]
    memory_reference: NativeMaterialMemoryReference
    admission_authorized: bool = False


@dataclass(frozen=True, slots=True)
class NativeMaterialPointHistorySnapshot:
    """One material point's chronology; no finite mass is assigned to it."""
    history: NativeMaterialHistory
    original_x: Fraction
    original_y: Fraction
    time_fraction: Fraction
    time_myr: Fraction
    material_age_myr: Fraction
    contact_occupancy: str
    material_allocation: str
    archive_cohort_id: str
    events: tuple[NativeMaterialHistoryEvent, ...]
    unresolved_event_sequence: int | None
    memory_reference: NativeMaterialMemoryReference
    admission_authorized: bool = False

    @property
    def history_sha256(self):
        return self.history.history_sha256

    @property
    def material_id(self):
        return self.history.parent.parcel.material_id

    @property
    def material_owner(self):
        return self.history.parent.parcel.plate


def _share_body(share):
    return dict(reservoir=share.reservoir, parent_amount=_rational(share.parent_amount),
        native_area_ratio_bounds=_interval(share.native_area_ratio_bounds),
        amount_bounds=_interval(share.amount_bounds),
        numerator_geometry_sha256=share.numerator_geometry_sha256,
        denominator_geometry_sha256=share.denominator_geometry_sha256,
        expression=share.expression)


def _parent_partition_body(share):
    complement = (I(1) - share.native_area_ratio_bounds) * share.parent_amount
    return dict(reservoir=share.reservoir, parent_amount=_rational(share.parent_amount),
        patch_share=_share_body(share), unmodelled_parent_complement_bounds=_interval(complement),
        unmodelled_parent_complement_expression="parent_amount * (1 - native_area(patch) / native_area(parent))",
        conservation_identity="patch_share + unmodelled_parent_complement = parent_amount")


def _event_body(event):
    return dict(sequence=event.sequence, kind=event.kind,
        time_fraction=_interval(event.time_fraction),
        relative_time_myr=_interval(event.relative_time_myr),
        absolute_time_myr=_interval(event.absolute_time_myr),
        operation=event.operation, cohort_id=event.cohort_id,
        material_time_function=event.material_time_function)


def _body(history):
    from .geometric_native_donor_sweep import native_donor_patch_sweep_to_dict
    return dict(format=FORMAT, model=MODEL,
        parent=_fragment_body(history.parent), receiver_parent=_fragment_body(history.receiver_parent),
        sweep_certificate=native_donor_patch_sweep_to_dict(history.sweep_certificate),
        memory_reference=_memory_body(history.memory_reference),
        material_epoch_myr=_rational(history.material_epoch_myr),
        parent_payload_sha256=history.parent_payload_sha256,
        receiver_payload_sha256=history.receiver_payload_sha256,
        patch_geometry_sha256=history.patch_geometry_sha256,
        sweep_sha256=history.sweep_sha256, cohort_id=history.cohort_id,
        shares=[_share_body(share) for share in history.shares],
        parent_reservoir_partitions=[_parent_partition_body(share) for share in history.shares],
        events=[_event_body(event) for event in history.events],
        allocation_model="implicit_first_entry_material_subsets_with_exact_native_area_ratios",
        mechanical_memory_policy="preserve_external_before_and_unaccepted_candidate_references",
        archive_policy="one_original_donor_cohort; contact_exit_does_not_unbury",
        admission_authorized=False, global_inventory_updated=False)


def initialize_native_material_history(parent, receiver_parent, sweep_certificate, *,
        material_epoch_myr=0, memory_reference):
    """Replay geometry and declare local first-entry allocation chronology."""
    from .geometric_native_donor_sweep import (
        NativeDonorPatchSweepCertificate, validate_native_donor_patch_sweep,
        native_donor_patch_sweep_to_dict,
    )
    parent, receiver = _checked_fragment(parent), _checked_fragment(receiver_parent)
    memory = _checked_memory(memory_reference)
    epoch = _fraction(material_epoch_myr, "Material epoch")
    if epoch < 0:
        raise ValueError("Material epoch must be nonnegative")
    if type(sweep_certificate) is not NativeDonorPatchSweepCertificate:
        raise ValueError("Material history requires the complete original donor patch sweep proof")
    sweep = validate_native_donor_patch_sweep(sweep_certificate)
    if (sweep.donor_polygon != parent.geometry or sweep.receiver_polygon != receiver.geometry
            or parent.parcel.plate == receiver.parcel.plate):
        raise ValueError("Donor sweep does not belong to these exact original physical parents")
    patch = validate_native_polygon(sweep.patch_geometry)
    if native_subtract(patch, parent.geometry):
        raise ValueError("Positive swept patch extends outside its original material parent")
    area, parent_area = sweep.patch_area_bounds, parent.geometry.area_bounds()
    if (type(area) is not I or area.lo <= 0 or parent_area.lo <= 0
            or area.hi >= parent_area.lo):
        raise ValueError("Swept patch needs a positive certified proper parent-area share")
    # The area bounds are enclosures of native measures, never exact numerical
    # weights.  The shared symbolic numerator retains the exact conservation
    # identity between live and archived subsets.
    ratio = area / parent_area
    parent_sha, receiver_sha = _hash(_fragment_body(parent)), _hash(_fragment_body(receiver))
    patch_sha, geometry_sha = _hash(native_polygon_to_dict(patch)), _hash(native_polygon_to_dict(parent.geometry))
    sweep_sha = _hash(native_donor_patch_sweep_to_dict(sweep))
    cohort = "native-passive-donor-patch:" + _hash(dict(parent=parent_sha,
        receiver=receiver_sha, patch=patch_sha, sweep=sweep_sha))
    shares = tuple(NativeMaterialShare(field, _fraction(getattr(parent.parcel, field), field),
        ratio, ratio * _fraction(getattr(parent.parcel, field), field), patch_sha, geometry_sha)
        for field in EXTENSIVE_FIELDS)
    raw_events = sweep.event_windows
    if (not isinstance(raw_events, tuple) or len(raw_events) != 3
            or tuple(event.kind for event in raw_events) != ("entry", "exit", "entry")):
        raise ValueError("Material chronology requires certified entry, exit and reentry for every patch point")
    dt = Fraction.from_float(sweep.dt_myr)
    operations = ("live_to_passive_archive_once", "contact_exit_keep_archive",
                  "contact_reentry_reuse_archive")
    events = []
    previous_high = Fraction(0)
    for sequence, (event, operation) in enumerate(zip(raw_events, operations), 1):
        window = event.time_fraction
        if (type(window) is not I or not 0 <= window.lo < window.hi <= 1
                or window.lo < previous_high or event.time_myr != window * dt):
            raise ValueError("Swept material event windows changed their single ordered physical clock")
        events.append(NativeMaterialHistoryEvent(sequence, event.kind, window,
            window * dt, I(epoch) + window * dt, operation, cohort))
        previous_high = window.hi
    unsealed = NativeMaterialHistory(parent, receiver, sweep, memory, epoch,
        parent_sha, receiver_sha, patch_sha, sweep_sha, cohort, shares, tuple(events), "")
    from dataclasses import replace
    return replace(unsealed, history_sha256=_hash(_body(unsealed)))


def _typed_share(share):
    if type(share) is not NativeMaterialShare or type(share.parent_amount) is not Fraction:
        raise ValueError("Material shares require immutable exact rational parent amounts")
    if (type(share.native_area_ratio_bounds) is not I or type(share.amount_bounds) is not I):
        raise ValueError("Material shares require native exact rational enclosures")


def validate_native_material_history(history):
    """Reconstruct proofs, parent payloads, shares, chronology and references.

    The public validation deliberately has no object-equality cache: Python
    numeric equality can alias a forged float with an exact Fraction.  Types
    and canonical replay must be checked before any equality can grant trust.
    """
    if type(history) is not NativeMaterialHistory:
        raise ValueError("Expected an immutable original donor material history")
    if type(history.material_epoch_myr) is not Fraction:
        raise ValueError("Material history epoch must remain an exact Fraction")
    if type(history.shares) is not tuple or type(history.events) is not tuple:
        raise ValueError("Material shares and events must remain immutable tuples")
    for share in history.shares:
        _typed_share(share)
    for event in history.events:
        if (type(event) is not NativeMaterialHistoryEvent or type(event.sequence) is not int
                or any(type(value) is not I for value in (
                    event.time_fraction, event.relative_time_myr, event.absolute_time_myr))):
            raise ValueError("Material events require immutable exact rational time windows")
    reproduced = initialize_native_material_history(history.parent, history.receiver_parent,
        history.sweep_certificate, material_epoch_myr=history.material_epoch_myr,
        memory_reference=history.memory_reference)
    if (reproduced != history or reproduced.history_sha256 != history.history_sha256
            or _encoded(_body(reproduced)) != _encoded(_body(history))):
        raise ValueError("Material history differs from complete native proof and material replay")
    return history


def evaluate_native_material_history(history, time_fraction):
    """Query exact age and implicit material allocations without state mutation.

    Inside a first-entry window the live and archived reservoir intervals are
    deliberately conservative.  Their common symbolic subset partition is the
    exact conservation proof; adding independent interval upper bounds would
    discard that correlation.  Neither window endpoint is a common point time.
    """
    history = validate_native_material_history(history)
    time = _fraction(time_fraction, "Material history time fraction")
    if not 0 <= time <= 1:
        raise ValueError("Material history queries require a fraction in [0, 1]")
    entry, exit_event, reentry = (event.time_fraction for event in history.events)
    if time <= entry.lo:
        contact, allocation = "outside", "live"
    elif time < entry.hi:
        contact, allocation = "mixed_entry_window", "implicit_first_entry_partition"
    elif time <= exit_event.lo:
        contact, allocation = "inside", "archived"
    elif time < exit_event.hi:
        contact, allocation = "mixed_exit_window", "archived"
    elif time <= reentry.lo:
        contact, allocation = "outside", "archived"
    elif time < reentry.hi:
        contact, allocation = "mixed_reentry_window", "archived"
    else:
        contact, allocation = "inside", "archived"
    amounts = []
    for share in history.shares:
        if allocation == "live":
            live, archived = share.amount_bounds, I(0)
            live_expr, archive_expr = share.expression, "0"
        elif allocation == "archived":
            live, archived = I(0), share.amount_bounds
            live_expr, archive_expr = "0", share.expression
        else:
            live = archived = I(0, share.amount_bounds.hi)
            live_expr = "parent_amount * native_area({q in patch: first_entry(q) > query_time}) / native_area(parent)"
            archive_expr = "parent_amount * native_area({q in patch: first_entry(q) <= query_time}) / native_area(parent)"
        amounts.append(NativeMaterialAllocation(share.reservoir, live, archived,
            live_expr, archive_expr, share))
    dt = Fraction.from_float(history.sweep_certificate.dt_myr)
    return NativeMaterialHistorySnapshot(history, history.history_sha256, time,
        history.material_epoch_myr + time * dt,
        Fraction.from_float(history.parent.parcel.age_myr) + time * dt,
        contact, allocation, history.cohort_id, tuple(amounts), history.memory_reference)


@lru_cache(maxsize=512)
def _material_point_events(history, x, y):
    from .geometric_native_donor_sweep import isolate_native_donor_patch_point_events
    return isolate_native_donor_patch_point_events(history.sweep_certificate, x, y)


def evaluate_native_material_point_history(history, x, y, time_fraction):
    """Execute the per-material-point entry time function and passive closure.

    Each root is retained as a strict sign bracket.  A query inside its
    enclosure returns an explicit event window; its midpoint never becomes
    an event timestamp.  These point queries do not integrate a partial patch
    volume or allocate mechanical quadrature work to material.
    """
    history = validate_native_material_history(history)
    if type(x) is not Fraction or type(y) is not Fraction:
        raise ValueError("Original donor material point coordinates must be exact Fraction values")
    time = _fraction(time_fraction, "Material point time fraction")
    if not 0 <= time <= 1:
        raise ValueError("Material point queries require a fraction in [0, 1]")
    point_events = _material_point_events(history, x, y)
    entry, exit_event, reentry = (event.time_fraction for event in point_events)
    unresolved = None
    if time < entry.lo:
        contact, allocation = "outside", "live"
    elif time <= entry.hi:
        contact, allocation, unresolved = "event_window", "first_entry_window", 1
    elif time < exit_event.lo:
        contact, allocation = "inside", "archived"
    elif time <= exit_event.hi:
        contact, allocation, unresolved = "event_window", "archived", 2
    elif time < reentry.lo:
        contact, allocation = "outside", "archived"
    elif time <= reentry.hi:
        contact, allocation, unresolved = "event_window", "archived", 3
    else:
        contact, allocation = "inside", "archived"
    dt = Fraction.from_float(history.sweep_certificate.dt_myr)
    events = tuple(NativeMaterialHistoryEvent(event.sequence, point.kind,
        point.time_fraction, point.time_myr,
        I(history.material_epoch_myr) + point.time_myr, event.operation, history.cohort_id)
        for event, point in zip(history.events, point_events))
    return NativeMaterialPointHistorySnapshot(history, x, y, time,
        history.material_epoch_myr + time * dt,
        Fraction.from_float(history.parent.parcel.age_myr) + time * dt,
        contact, allocation, history.cohort_id, events, unresolved, history.memory_reference)


def native_material_history_to_dict(history):
    history = validate_native_material_history(history)
    body = _body(history)
    body["history_sha256"] = history.history_sha256
    # JSON normalization preserves the complete native and parcel schema.
    return json.loads(_encoded(body))


def _snapshot_body(snapshot):
    return dict(format="native-original-donor-patch-material-query-1", model=MODEL,
        history_sha256=snapshot.history_sha256,
        time_fraction=_rational(snapshot.time_fraction),
        time_myr=_rational(snapshot.time_myr),
        material_age_myr=_rational(snapshot.material_age_myr),
        contact_occupancy=snapshot.contact_occupancy,
        material_allocation=snapshot.material_allocation,
        archive_cohort_id=snapshot.archive_cohort_id,
        allocations=[dict(reservoir=allocation.reservoir,
            live_amount_bounds=_interval(allocation.live_amount_bounds),
            archived_amount_bounds=_interval(allocation.archived_amount_bounds),
            live_expression=allocation.live_expression,
            archived_expression=allocation.archived_expression,
            conserved_total=_share_body(allocation.conserved_total),
            partition_identity=allocation.partition_identity)
            for allocation in snapshot.allocations],
        parent_reservoir_partitions=[_parent_partition_body(allocation.conserved_total)
                                    for allocation in snapshot.allocations],
        memory_reference=_memory_body(snapshot.memory_reference),
        archive_cohort_count_bounds=[0, 0] if snapshot.material_allocation == "live"
            else ([0, 1] if snapshot.material_allocation == "implicit_first_entry_partition" else [1, 1]),
        contact_exit_or_reentry_creates_new_cohort=False,
        candidate_mechanics_adopted=False, admission_authorized=False,
        global_inventory_updated=False)


def native_material_history_snapshot_to_dict(snapshot):
    """Encode a local query only after replay against its authoritative history."""
    if type(snapshot) is not NativeMaterialHistorySnapshot:
        raise ValueError("Expected an exact native material history query")
    if any(type(value) is not Fraction for value in (
            snapshot.time_fraction, snapshot.time_myr, snapshot.material_age_myr)):
        raise ValueError("Material query clocks and age must remain exact Fraction values")
    if type(snapshot.allocations) is not tuple:
        raise ValueError("Material query allocations must remain immutable tuples")
    for allocation in snapshot.allocations:
        if (type(allocation) is not NativeMaterialAllocation
                or type(allocation.live_amount_bounds) is not I
                or type(allocation.archived_amount_bounds) is not I):
            raise ValueError("Material query allocations require native exact enclosures")
        _typed_share(allocation.conserved_total)
    _checked_memory(snapshot.memory_reference)
    if snapshot.admission_authorized is not False:
        raise ValueError("A local material query cannot grant global force admission")
    _sha(snapshot.history_sha256, "Material history")
    reproduced = evaluate_native_material_history(snapshot.history, snapshot.time_fraction)
    if (reproduced != snapshot
            or _encoded(_snapshot_body(reproduced)) != _encoded(_snapshot_body(snapshot))):
        raise ValueError("Material query differs from exact native history replay")
    return _snapshot_body(snapshot)


def native_material_point_history_to_dict(snapshot):
    """Serialize a point query only after executing its complete replay."""
    if type(snapshot) is not NativeMaterialPointHistorySnapshot:
        raise ValueError("Expected an immutable native material point history query")
    if (any(type(value) is not Fraction for value in (
            snapshot.original_x, snapshot.original_y, snapshot.time_fraction,
            snapshot.time_myr, snapshot.material_age_myr))
            or snapshot.admission_authorized is not False
            or (snapshot.unresolved_event_sequence is not None
                and type(snapshot.unresolved_event_sequence) is not int)):
        raise ValueError("Material point query must retain its exact typed clocks, coordinates and refusal status")
    if (type(snapshot.events) is not tuple or any(
            type(event) is not NativeMaterialHistoryEvent or type(event.sequence) is not int
            or any(type(value) is not I for value in (
                event.time_fraction, event.relative_time_myr, event.absolute_time_myr))
            for event in snapshot.events)):
        raise ValueError("Material point event windows must retain their exact typed provenance")
    reproduced = evaluate_native_material_point_history(snapshot.history,
        snapshot.original_x, snapshot.original_y, snapshot.time_fraction)
    if (reproduced != snapshot
            or _encoded([_event_body(event) for event in reproduced.events])
                != _encoded([_event_body(event) for event in snapshot.events])):
        raise ValueError("Material point history differs from exact per-point event replay")
    return dict(format="native-original-donor-material-point-query-1", model=MODEL,
        history_sha256=snapshot.history_sha256,
        parent_payload_sha256=snapshot.history.parent_payload_sha256,
        patch_geometry_sha256=snapshot.history.patch_geometry_sha256,
        sweep_sha256=snapshot.history.sweep_sha256,
        original_coordinates=dict(x=_rational(snapshot.original_x),
                                  y=_rational(snapshot.original_y), z=[-1, 1]),
        material_id=snapshot.material_id, material_owner=snapshot.material_owner,
        time_fraction=_rational(snapshot.time_fraction),
        time_myr=_rational(snapshot.time_myr),
        material_age_myr=_rational(snapshot.material_age_myr),
        contact_occupancy=snapshot.contact_occupancy,
        material_allocation=snapshot.material_allocation,
        archive_cohort_id=snapshot.archive_cohort_id,
        events=[_event_body(event) for event in snapshot.events],
        unresolved_event_sequence=snapshot.unresolved_event_sequence,
        memory_reference=_memory_body(snapshot.memory_reference),
        finite_material_reservoir_assigned_to_point=False,
        contact_exit_or_reentry_creates_new_cohort=False,
        candidate_mechanics_adopted=False, admission_authorized=False,
        global_inventory_updated=False)


def native_material_point_history_from_dict(raw, *, history):
    """Reconstruct a saved point query against its authoritative history."""
    if (type(raw) is not dict
            or raw.get("format") != "native-original-donor-material-point-query-1"
            or raw.get("model") != MODEL):
        raise ValueError("Expected the native original donor material point query schema")
    try:
        coordinates = raw["original_coordinates"]
        if (type(coordinates) is not dict or set(coordinates) != {"x", "y", "z"}
                or coordinates["z"] != [-1, 1]):
            raise ValueError("Invalid original donor material point coordinates")
        snapshot = evaluate_native_material_point_history(history,
            _read_rational(coordinates["x"]), _read_rational(coordinates["y"]),
            _read_rational(raw["time_fraction"]))
    except KeyError as error:
        raise ValueError("Incomplete native material point query schema") from error
    if _encoded(raw) != _encoded(native_material_point_history_to_dict(snapshot)):
        raise ValueError("Saved material point query differs from complete native point-event replay")
    return snapshot


def native_material_history_from_dict(raw, *, expected_memory_reference=None):
    """Load by full reconstruction; a resealed derived field grants no trust."""
    from .geometric_native_donor_sweep import native_donor_patch_sweep_from_dict
    if type(raw) is not dict or raw.get("format") != FORMAT or raw.get("model") != MODEL:
        raise ValueError("Expected the explicit native original donor material history schema")
    # Exact key checking is completed against the reproduced canonical body.
    try:
        reference = raw["memory_reference"]
        if (type(reference) is not dict or set(reference) != {"before_payload", "candidate_after_payload",
                "before_sha256", "candidate_after_sha256", "constitutive_path_sha256"}):
            raise ValueError("Invalid mechanical memory reference schema")
        memory = make_native_material_memory_reference(reference["before_payload"],
            reference["candidate_after_payload"],
            constitutive_path_sha256=reference["constitutive_path_sha256"])
        if _encoded(reference) != _encoded(_memory_body(memory)):
            raise ValueError("Mechanical memory payload SHA references changed")
        if expected_memory_reference is not None and memory != _checked_memory(expected_memory_reference):
            raise ValueError("Material history mechanical references differ from the external physical evidence")
        result = initialize_native_material_history(_fragment_from_body(raw["parent"]),
            _fragment_from_body(raw["receiver_parent"]),
            native_donor_patch_sweep_from_dict(raw["sweep_certificate"]),
            material_epoch_myr=_read_rational(raw["material_epoch_myr"]), memory_reference=memory)
    except KeyError as error:
        raise ValueError("Incomplete native material history schema") from error
    expected = _body(result)
    expected["history_sha256"] = result.history_sha256
    if _encoded(raw) != _encoded(expected):
        raise ValueError("Saved material history differs from complete native material replay")
    return result
