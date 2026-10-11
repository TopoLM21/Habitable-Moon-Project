"""Explicit conservative endpoint transport on portable native footprints.

Every rotation, overlap, retained cut and vacancy uses the native polygon's
saved support lineage.  A world-coordinate projection is never fed back into
this transaction.  This experimental surface supplies material bookkeeping
and geometric adjacency; it supplies no force, cooling or slab law.

Import is explicit and audits the literal old reference geometry.  In
particular, a legacy partition which merely passed its floating predicates
cannot acquire a native certificate if its saved ray domains really overlap.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import hashlib
import heapq
from itertools import product
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np

from .fractional_surface import EXTENSIVE_FIELDS, SurfaceParcel, split_parcel
from .geometric_intrinsic_geometry import IntrinsicGeometry
from .geometric_intrinsic_surface import IntrinsicSurfaceState
from .geometric_native_polygons import (
    NativePolygon, native_arc_to_dict, native_intersection, native_polygon_from_dict,
    native_polygon_to_dict, native_shared_boundary_arcs, native_subtract,
    validate_native_polygon,
)
from .geometric_polarity import PolarityResolution
from .geometric_rotation_vectors import compact_rotation_vectors
from .geometric_surface import AREA_REL_TOL, GeometricSurfaceState
from .geometric_transport import UnresolvedPolarityError


VERSION = "geometric-native-surface-1"
FORMAT = "geometric-native-surface-checkpoint-1"
_PARCEL_KEYS = set(SurfaceParcel.__dataclass_fields__)


@lru_cache(maxsize=1)
def _sphere_octants():
    """Eight exact axis domains whose union is the entire unit sphere."""
    return tuple(NativePolygon.from_intrinsic(IntrinsicGeometry(
        ((x, 0., 0.), (0., y, 0.), (0., 0., z))))
        for x, y, z in product((-1., 1.), repeat=3))


@lru_cache(maxsize=256)
def _audit_native_coverage(geometries):
    # An identical post-cut pose is an exact bijective SO(3) map of the whole
    # sphere.  Audit its source epoch rather than create avoidable intersections
    # between this common map and the arbitrary axis reference domains.
    common = geometries[0].pose
    if common and all(geometry.pose == common for geometry in geometries[1:]):
        return _audit_native_coverage(tuple(replace(geometry, pose=()) for geometry in geometries))
    # Any certified SO(3) pose also maps the eight exact axis octants onto the
    # whole sphere.  Align this disposable audit basis with an existing body
    # epoch, reducing redundant intersections through a uniformly moving bulk.
    histories = {}
    for geometry in geometries:
        if geometry.pose:
            key = tuple((tuple(value.hex() for value in step.omega_rad_per_myr), step.dt_myr.hex())
                        for step in geometry.pose)
            count, history = histories.get(key, (0, geometry.pose))
            histories[key] = (count + 1, history)
    selected = min(histories, key=lambda key: (-histories[key][0], key)) if histories else None
    basis = histories[selected][1] if selected is not None else ()
    for reference_octant in _sphere_octants():
        octant = replace(reference_octant, pose=basis) if basis else reference_octant
        gaps = (octant,)
        for covered in geometries:
            remaining = []
            for gap in gaps:
                if native_intersection(gap, covered) is None:
                    remaining.append(gap)
                else:
                    remaining.extend(native_subtract(gap, covered))
            gaps = tuple(remaining)
            if not gaps:
                break
        if gaps:
            # Every returned native polygon has certified positive area.  Even
            # a gap too tiny to change the summed binary64 sphere area refuses
            # the partition; unproved cuts already refuse in the native kernel.
            raise ValueError("Native partition contains uncovered positive footprints")
    return True


def _real(value, label, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{label} must be an explicit finite real number")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"{label} must be an explicit finite real number") from error
    if not math.isfinite(result) or (result <= 0. if positive else result < 0.):
        raise ValueError(f"{label} must be finite and {'positive' if positive else 'nonnegative'}")
    return result


def _identity(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a stable nonempty string")
    return value


def _parcel_from_dict(raw):
    if not isinstance(raw, dict) or set(raw) != _PARCEL_KEYS:
        raise ValueError("Native material requires the complete explicit parcel schema")
    for name in (*EXTENSIVE_FIELDS, "age_myr"):
        _real(raw[name], "Native material " + name,
              positive=name in ("area_km2", "oceanic_volume_km3"))
    properties = raw["specific_properties"]
    if not isinstance(properties, (tuple, list)) or len(properties) != 3:
        raise ValueError("Native material requires three inherited specific properties")
    for value in properties:
        _real(value, "Native specific property")
    fields = raw["material_fields"]
    if not isinstance(fields, (tuple, list)):
        raise ValueError("Native material requires explicit named fields")
    for pair in fields:
        if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                or not isinstance(pair[0], str) or not pair[0]
                or isinstance(pair[1], (bool, np.bool_)) or not isinstance(pair[1], Real)):
            raise ValueError("Native material fields require named finite real values")
        try:
            finite = math.isfinite(float(pair[1]))
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Native material fields require named finite real values") from error
        if not finite:
            raise ValueError("Native material fields require named finite real values")
    return SurfaceParcel(**raw)


@dataclass(frozen=True, slots=True)
class NativeFragment:
    fragment_id: str
    geometry: NativePolygon
    parcel: SurfaceParcel
    parent_fragment_id: str | None = None

    def __post_init__(self):
        _identity(self.fragment_id, "Native fragment identity")
        if type(self.geometry) is not NativePolygon:
            raise ValueError("Native fragment requires an explicit native polygon")
        if type(self.parcel) is not SurfaceParcel:
            raise ValueError("Native fragment requires an explicit material parcel")
        if self.parent_fragment_id is not None:
            _identity(self.parent_fragment_id, "Native fragment parent")


class NativeSurfaceState:
    """An immutable, completely audited material partition in the native format."""

    # Generic legacy dataclass writers reject this object before opening a
    # destination.  Its explicit writer alone labels native support geometry.
    __slots__ = ("time_myr", "radius_km", "fragments", "known_material_ids", "phase", "version")

    def __init__(self, time_myr, radius_km, fragments, known_material_ids=(),
                 phase="partition", version=VERSION):
        if version != VERSION or phase != "partition":
            raise ValueError("Native surface requires its explicit resolved partition format")
        time = _real(time_myr, "Native surface clock")
        radius = _real(radius_km, "Native surface radius", positive=True)
        try:
            radius_squared = radius**2
        except OverflowError as error:
            raise ValueError("Native sphere area must remain positive and representable") from error
        if not math.isfinite(radius_squared) or radius_squared <= 0.:
            raise ValueError("Native sphere area must remain positive and representable")
        sphere_area = 4. * math.pi * radius_squared
        if not math.isfinite(sphere_area) or sphere_area <= 0.:
            raise ValueError("Native sphere area must remain positive and representable")
        try:
            fragments = tuple(fragments)
            known = tuple(known_material_ids)
        except TypeError as error:
            raise ValueError("Native surface requires explicit fragments and material identities") from error
        if not fragments or any(type(fragment) is not NativeFragment for fragment in fragments):
            raise ValueError("Native surface requires explicit native material fragments")
        checked = []
        ids = set()
        for fragment in fragments:
            if fragment.fragment_id in ids:
                raise ValueError("Native fragment identities must be unique")
            ids.add(fragment.fragment_id)
            geometry = validate_native_polygon(fragment.geometry)
            parcel = _parcel_from_dict(asdict(fragment.parcel))
            current = NativeFragment(fragment.fragment_id, geometry, parcel,
                                     fragment.parent_fragment_id)
            measured = geometry.area_sr * radius_squared
            if not math.isfinite(measured) or measured <= 0.:
                raise ValueError("Native material area must remain positive and representable")
            if not math.isclose(measured, parcel.area_km2, rel_tol=AREA_REL_TOL, abs_tol=0.):
                raise ValueError("Native material area and its footprint disagree")
            checked.append(current)
        try:
            area = math.fsum(fragment.parcel.area_km2 for fragment in checked)
        except OverflowError as error:
            raise ValueError("Native total material area must remain finite") from error
        if not math.isclose(area, sphere_area, rel_tol=AREA_REL_TOL, abs_tol=0.):
            raise ValueError("Native surface material area must equal the sphere area")
        for identity in known:
            _identity(identity, "Historical native material identity")
        origins = {fragment.parcel.material_id for fragment in checked}
        if known and not origins.issubset(known):
            raise ValueError("Native surface is missing historical material identities")
        for index, left in enumerate(checked):
            for right in checked[index + 1:]:
                if native_intersection(left.geometry, right.geometry) is not None:
                    raise ValueError("Native partition contains overlapping footprints")
        _audit_native_coverage(tuple(fragment.geometry for fragment in checked))
        for name, value in (("time_myr", time), ("radius_km", radius),
                            ("fragments", tuple(checked)),
                            ("known_material_ids", tuple(sorted(set(known) | origins))),
                            ("phase", phase), ("version", version)):
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        raise AttributeError("Native surface and its material partition are immutable")

    def __delattr__(self, name):
        raise AttributeError("Native surface and its material partition are immutable")

    def __eq__(self, other):
        if type(other) is not NativeSurfaceState:
            return NotImplemented
        return all(getattr(self, name) == getattr(other, name) for name in self.__slots__)


@dataclass(frozen=True, slots=True)
class NativeOverlap:
    fragment_a: str
    fragment_b: str
    plate_a: int
    plate_b: int
    geometry: NativePolygon
    area_km2: float
    reason: str


@dataclass(frozen=True, slots=True)
class NativeLoss:
    fragment: NativeFragment
    source_fragment_id: str
    receiver_fragment_id: str
    receiver_plate: int
    time_myr: float
    contact_ids: tuple[str, ...] = ()
    attachment_status: str = "unresolved_native_support"
    polarity_basis: str = "material_contrast"
    polarity_evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class NativeContact:
    contact_id: str
    fragment_a: str
    fragment_b: str
    material_a: str
    material_b: str
    plate_a: int
    plate_b: int
    arc: object
    length_km: float


@dataclass(frozen=True, slots=True)
class NativeTransportResult:
    state: NativeSurfaceState
    losses: tuple[NativeLoss, ...]
    births: tuple[NativeFragment, ...]
    contacts: tuple[NativeContact, ...]
    diagnostics: dict


def enroll_native_surface(surface):
    """Explicitly import an old reference, then audit native saved-ray domains."""
    if type(surface) is GeometricSurfaceState:
        if surface.phase != "partition":
            raise ValueError("Native import requires a strict resolved legacy partition")
        # Validate the old contract as well as the stronger native domain
        # contract; a caller cannot use import to bypass old area checks.
        from .geometric_intrinsic_surface import _strict_reference
        checked = _strict_reference(surface)
        fragments = tuple(NativeFragment(fragment.fragment_id,
            NativePolygon.from_intrinsic(IntrinsicGeometry(fragment.polygon)),
            fragment.parcel, fragment.parent_fragment_id) for fragment in checked.fragments)
    elif type(surface) is IntrinsicSurfaceState:
        checked = IntrinsicSurfaceState(surface.reference_surface, surface.rotations, surface.version)
        if checked != surface:
            raise ValueError("Intrinsic derived material disagrees with its reference and pose")
        fragments = tuple(NativeFragment(fragment.fragment_id,
            NativePolygon.from_intrinsic(fragment.geometry), fragment.parcel,
            fragment.parent_fragment_id) for fragment in checked.fragments)
    else:
        raise ValueError("Native import requires an explicit strict or intrinsic surface")
    return NativeSurfaceState(checked.time_myr, checked.radius_km, fragments,
                              checked.known_material_ids)


def _validated_state(state):
    if type(state) is not NativeSurfaceState:
        raise ValueError("Expected an explicit native surface")
    checked = NativeSurfaceState(state.time_myr, state.radius_km, state.fragments,
                                state.known_material_ids, state.phase, state.version)
    if checked != state:
        raise ValueError("Native material state disagrees with its validated geometry")
    return checked


def _totals(fragments):
    return {name: math.fsum(getattr(fragment.parcel, name) for fragment in fragments)
            for name in EXTENSIVE_FIELDS}


def _compare_polarity(left, right):
    for a, b in ((left.parcel.specific_properties[2], right.parcel.specific_properties[2]),
                 (left.parcel.age_myr, right.parcel.age_myr)):
        if not math.isclose(a, b, rel_tol=64. * math.ulp(1.), abs_tol=0.):
            return -1 if a < b else 1
    return 0


def _split_native(fragment, geometries, *, label):
    geometries = tuple(geometries)
    measured = tuple(geometry.area_sr for geometry in geometries)
    total = math.fsum(measured)
    if (not measured or any(not math.isfinite(area) or area <= 0. for area in measured)
            or not math.isclose(total, fragment.geometry.area_sr,
                                rel_tol=AREA_REL_TOL, abs_tol=0.)):
        raise ValueError("Native geometric children do not partition their source footprint")
    weights = [area / total for area in measured]
    largest = max(range(len(weights)), key=weights.__getitem__)
    weights[largest] += 1. - math.fsum(weights)
    children = []
    for index, (geometry, weight) in enumerate(zip(geometries, weights)):
        if not math.isfinite(weight) or weight <= 0. or weight > 1.:
            raise ValueError("Native geometric fraction must remain positive and representable")
        parcel = split_parcel(fragment.parcel, weight)
        children.append(NativeFragment(f"{fragment.fragment_id}/{label}:{index}",
                                        geometry, parcel, fragment.fragment_id))
    return tuple(children)


def _encoded(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def extract_native_contacts(fragments, radius_km):
    """Positive native shared arcs; no world proximity or mechanical inference."""
    radius = _real(radius_km, "Native contact radius", positive=True)
    ordered = sorted(tuple(fragments), key=lambda fragment: fragment.fragment_id)
    contacts = []
    for index, left in enumerate(ordered):
        for right in ordered[index + 1:]:
            if left.parcel.plate == right.parcel.plate:
                continue
            for arc in native_shared_boundary_arcs(left.geometry, right.geometry):
                length = float(arc.length_radians) * radius
                if not math.isfinite(length) or length <= 0.:
                    raise ValueError("Native contact length must remain positive and representable")
                # Native arc data contains primitive reference rays/planes and
                # complete physical histories, so identity is geometry-bound.
                identity = hashlib.sha256(_encoded(dict(
                    fragments=[left.fragment_id, right.fragment_id],
                    arc=native_arc_to_dict(arc)))).hexdigest()
                contacts.append(NativeContact(f"native-contact:{identity}",
                    left.fragment_id, right.fragment_id, left.parcel.material_id,
                    right.parcel.material_id, left.parcel.plate, right.parcel.plate,
                    arc, length))
    return tuple(contacts)


def _diagnostics(before, state, losses, births, contacts, decisions):
    initial, retained = _totals(before.fragments), _totals(state.fragments)
    removed = _totals(tuple(loss.fragment for loss in losses))
    born = _totals(births)
    residual = {name: (retained[name] + removed[name] - born[name] - initial[name]) /
                max(abs(initial[name]), abs(born[name]), 1.) for name in EXTENSIVE_FIELDS}
    if any(abs(value) > 5e-12 for value in residual.values()):
        raise RuntimeError("Native material transaction failed its extensive ledger")
    # Global sums can round away an entire tiny material history.  Audit each
    # origin and reservoir without an absolute floor, including exact zeros.
    groups = []
    for fragments in (before.fragments, state.fragments,
                      tuple(loss.fragment for loss in losses), births):
        values = {}
        for fragment in fragments:
            for name in EXTENSIVE_FIELDS:
                values.setdefault((fragment.parcel.material_id, name), []).append(
                    getattr(fragment.parcel, name))
        groups.append({key: math.fsum(parts) for key, parts in values.items()})
    original_origins, retained_origins, removed_origins, born_origins = groups
    keys = set().union(*(group.keys() for group in groups))
    material_residuals = {}
    for identity, name in sorted(keys):
        key = (identity, name)
        actual = retained_origins.get(key, 0.) + removed_origins.get(key, 0.)
        expected = original_origins.get(key, 0.) + born_origins.get(key, 0.)
        if not math.isclose(actual, expected, rel_tol=5e-12, abs_tol=0.):
            raise RuntimeError(f"Native material transaction lost an origin reservoir: {identity}/{name}")
        scale = max(abs(actual), abs(expected))
        error = abs(actual - expected) / scale if scale else 0.
        material_residuals.setdefault(identity, {})[name] = error
    maximum = max((error for values in material_residuals.values()
                   for error in values.values()), default=0.)
    return dict(retained=retained, removed=removed, born=born,
        relative_residuals=residual, contacts=len(contacts),
        per_material_relative_residuals=material_residuals,
        max_per_material_relative_residual=maximum,
        audited_material_origins=len(material_residuals),
        attached_loss_count=0, unresolved_attachment_count=len(losses),
        polarity_policy="greater excess mass/area loses, then greater age; 64eps arithmetic ties unresolved",
        polarity_resolution_counts={basis: sum(decision.basis == basis for decision in decisions.values())
                                    for basis in sorted({decision.basis for decision in decisions.values()})},
        birth_policy="actual native vacancy intersected with pre-step footprints; endpoint hot birth",
        time_scheme="endpoint_native_geometry_v1", thermal_evolution="frozen",
        geometry_authority="portable reference support planes and complete physical rotations",
        mechanical_attachment="native geometric adjacency only; no forces or slab attachment state")


def advance_native_surface(state, omega_rad_per_myr, dt_myr, *, birth_factory,
                           polarity_resolver=None):
    """Return one atomic native endpoint transaction preserving all reservoirs.

    The caller supplies the same explicit local hot-vacancy closure as ordinary
    geometric transport.  It is a geometric control, not a loaded force step.
    """
    state = _validated_state(state)
    if not callable(birth_factory):
        raise ValueError("Native advance requires an explicit birth factory")
    if polarity_resolver is not None and not callable(polarity_resolver):
        raise ValueError("Native polarity resolver must be callable")
    dt = _real(dt_myr, "Native transport timestep", positive=True)
    following = state.time_myr + dt
    if not math.isfinite(following) or following <= state.time_myr:
        raise ValueError("Native transport must advance the physical clock")
    if not isinstance(omega_rad_per_myr, Mapping):
        raise ValueError("Native motion requires an explicit material-owner mapping")
    omega = compact_rotation_vectors(omega_rad_per_myr,
                                    (fragment.parcel.plate for fragment in state.fragments))
    moved = tuple(NativeFragment(fragment.fragment_id,
        fragment.geometry.rotated(tuple(map(float, omega[fragment.parcel.plate])), dt),
        replace(fragment.parcel, age_myr=fragment.parcel.age_myr + dt),
        fragment.parent_fragment_id) for fragment in state.fragments)
    owners = tuple(omega)
    if all(np.array_equal(omega[owners[0]], omega[owner]) for owner in owners[1:]):
        resolved = NativeSurfaceState(following, state.radius_km, moved, state.known_material_ids)
        contacts = extract_native_contacts(resolved.fragments, state.radius_km)
        return NativeTransportResult(resolved, (), (), contacts,
                                     _diagnostics(state, resolved, (), (), contacts, {}))

    by_id = {fragment.fragment_id: fragment for fragment in moved}
    neighbours = {identity: [] for identity in by_id}
    successors = {identity: [] for identity in by_id}
    indegree = dict.fromkeys(by_id, 0)
    decisions, overlap_records, unresolved = {}, [], []
    for index, left in enumerate(moved):
        for right in moved[index + 1:]:
            overlap = native_intersection(left.geometry, right.geometry)
            if overlap is None:
                continue
            if left.parcel.plate == right.parcel.plate:
                raise ValueError("Rigidly moved native fragments of the same owner overlap")
            decision = None if polarity_resolver is None else polarity_resolver(left, right, overlap)
            if decision is None:
                decision = PolarityResolution(_compare_polarity(left, right), "material_contrast")
            if type(decision) is not PolarityResolution:
                raise ValueError("Native polarity resolver must return a PolarityResolution or None")
            decision = PolarityResolution(decision.comparison, decision.basis,
                                         decision.evidence_ids, decision.contact_ids)
            decisions[frozenset((left.fragment_id, right.fragment_id))] = decision
            record = NativeOverlap(left.fragment_id, right.fragment_id,
                left.parcel.plate, right.parcel.plate, overlap, overlap.area_sr * state.radius_km**2,
                "equal_buoyancy_and_age" if decision.basis == "material_contrast" else decision.basis)
            overlap_records.append(record)
            if decision.comparison == 0:
                unresolved.append(record)
            else:
                winner, loser = (left, right) if decision.comparison < 0 else (right, left)
                successors[winner.fragment_id].append(loser.fragment_id)
                indegree[loser.fragment_id] += 1
            neighbours[left.fragment_id].append(right.fragment_id)
            neighbours[right.fragment_id].append(left.fragment_id)
    if unresolved:
        raise UnresolvedPolarityError(unresolved)
    ready = [identity for identity, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    ordered = []
    while ready:
        identity = heapq.heappop(ready)
        ordered.append(by_id[identity])
        for loser in successors[identity]:
            indegree[loser] -= 1
            if indegree[loser] == 0:
                heapq.heappush(ready, loser)
    if len(ordered) != len(moved):
        raise UnresolvedPolarityError(tuple(replace(record, reason="priority_cycle")
            for record in overlap_records if indegree[record.fragment_a] and indegree[record.fragment_b]))

    retained, losses, accepted = [], [], {}
    for source in ordered:
        pieces, removed = [source.geometry], []
        for neighbour in sorted(neighbours[source.fragment_id]):
            for receiver in accepted.get(neighbour, ()):
                remaining = []
                for geometry in pieces:
                    overlap = native_intersection(geometry, receiver.geometry)
                    if overlap is None:
                        remaining.append(geometry)
                    else:
                        removed.append((overlap, receiver,
                                        decisions[frozenset((source.fragment_id, neighbour))]))
                        remaining.extend(native_subtract(geometry, receiver.geometry))
                pieces = remaining
        if not removed:
            children = (source,)
        else:
            parts = _split_native(source, pieces + [geometry for geometry, _, _ in removed],
                                  label=f"native-step:{following.hex()}")
            children = parts[:len(pieces)]
            for fragment, (_, receiver, decision) in zip(parts[len(pieces):], removed):
                losses.append(NativeLoss(fragment, source.fragment_id, receiver.fragment_id,
                    receiver.parcel.plate, following, polarity_basis=decision.basis,
                    polarity_evidence_ids=decision.evidence_ids))
        retained.extend(children)
        accepted[source.fragment_id] = children

    # Old footprints, rather than integration cells, declare vacancy ownership.
    births, known = [], set(state.known_material_ids)
    for departed in state.fragments:
        gaps = [departed.geometry]
        for covered in retained:
            remaining = []
            for geometry in gaps:
                if native_intersection(geometry, covered.geometry) is None:
                    remaining.append(geometry)
                else:
                    remaining.extend(native_subtract(geometry, covered.geometry))
            gaps = remaining
            if not gaps:
                break
        for geometry in gaps:
            area = geometry.area_sr * state.radius_km**2
            if not math.isfinite(area) or area <= 0.:
                raise ValueError("Native vacancy area must remain positive and representable")
            parcel = birth_factory(departed.parcel.cell, departed.parcel.plate,
                                   area, following, len(births))
            if type(parcel) is not SurfaceParcel:
                raise ValueError("Birth factory must supply an explicit material parcel")
            parcel = _parcel_from_dict(asdict(parcel))
            if (parcel.plate != departed.parcel.plate or parcel.cell != departed.parcel.cell
                    or parcel.age_myr != 0. or parcel.cold_mantle_volume_km3 != 0.
                    or parcel.density_excess_mass_kg != 0.
                    or not math.isclose(parcel.area_km2, area, rel_tol=2e-13, abs_tol=0.)
                    or parcel.material_id in known):
                raise ValueError("Birth factory must supply fresh, hot material for the actual native local vacancy")
            known.add(parcel.material_id)
            births.append(NativeFragment(f"native-new-fragment:{following.hex()}:{len(births)}",
                                         geometry, parcel, departed.fragment_id))
    resolved = NativeSurfaceState(following, state.radius_km, tuple(retained + births), tuple(known))
    contacts = extract_native_contacts(resolved.fragments, state.radius_km)
    return NativeTransportResult(resolved, tuple(losses), tuple(births), contacts,
        _diagnostics(state, resolved, tuple(losses), tuple(births), contacts, decisions))


def native_surface_to_dict(state):
    checked = _validated_state(state)
    return dict(version=checked.version, phase=checked.phase, time_myr=checked.time_myr,
        radius_km=checked.radius_km, known_material_ids=list(checked.known_material_ids),
        fragments=[dict(fragment_id=fragment.fragment_id,
            geometry=native_polygon_to_dict(fragment.geometry), parcel=asdict(fragment.parcel),
            parent_fragment_id=fragment.parent_fragment_id) for fragment in checked.fragments])


def native_surface_from_dict(raw):
    if (not isinstance(raw, dict) or set(raw) != set(NativeSurfaceState.__slots__)
            or not isinstance(raw["fragments"], list)
            or not isinstance(raw["known_material_ids"], list)):
        raise ValueError("Invalid native surface schema")
    fragments = []
    for row in raw["fragments"]:
        if (not isinstance(row, dict)
                or set(row) != {"fragment_id", "geometry", "parcel", "parent_fragment_id"}):
            raise ValueError("Invalid native fragment schema")
        fragments.append(NativeFragment(row["fragment_id"],
            native_polygon_from_dict(row["geometry"]), _parcel_from_dict(row["parcel"]),
            row["parent_fragment_id"]))
    return NativeSurfaceState(raw["time_myr"], raw["radius_km"], fragments,
        raw["known_material_ids"], raw["phase"], raw["version"])


def save_native_checkpoint(path, state, *, provenance=None):
    payload = dict(format=FORMAT, state=native_surface_to_dict(state),
                   provenance={} if provenance is None else provenance)
    if not isinstance(payload["provenance"], dict):
        raise ValueError("Native checkpoint provenance must be an explicit object")
    payload["payload_sha256"] = hashlib.sha256(_encoded(payload)).hexdigest()
    encoded = _encoded(payload) + b"\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(encoded)
    return path


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Native checkpoint contains duplicate JSON keys")
        result[key] = value
    return result


def load_native_checkpoint(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if (not isinstance(payload, dict)
            or set(payload) != {"format", "state", "provenance", "payload_sha256"}
            or payload["format"] != FORMAT or not isinstance(payload["provenance"], dict)):
        raise ValueError("Expected an explicit native surface checkpoint schema")
    saved = payload.pop("payload_sha256")
    if saved != hashlib.sha256(_encoded(payload)).hexdigest():
        raise ValueError("Native checkpoint integrity mismatch")
    return native_surface_from_dict(payload["state"]), payload["provenance"]
