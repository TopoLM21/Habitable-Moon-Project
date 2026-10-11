"""Conservative support of an original bending domain after material burial.

Only original material supplies the retained beam.  Each new archive loss is
pulled back with its donor's actual map at acceptance, never with a later map.
The complete original two-dimensional material footprints must be covered once
by their live and buried descendants, and all four reservoirs are retained.
An enclosing spherical quadrilateral certifies the whole original strip; the
complete small-circle clamp curve must still belong to original LIVE material.
The quadrilateral is a conservative sufficient gate, not a new beam shape.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import math
from numbers import Integral, Real

import numpy as np

from .fractional_surface import EXTENSIVE_FIELDS
from .geometric_bending_contact_dynamics import BendingContactState, _bending_signature
from .geometric_bending_reference import validate_reference_rotation
from .geometric_boundary import boundary_from_dict, boundary_to_dict, surface_digest
from .geometric_boundary_io import _validate_pair
from .geometric_fault_motion_support import (
    UnresolvedFaultMotionSupport, validate_reference_bending_bank_support,
)
from .geometric_fault_traces import _arc_frame
from .geometric_surface import AREA_REL_TOL, audit_partition
from .spherical_polygons import (
    _edge_distance, intersect_convex, normalize_polygon, polygon_area, subtract_convex,
)

FORMAT = "geometric-original-material-buried-bank-registry-1"
SUPPORT_MODEL = "complete_original_2D_material_domain_and_live_continuous_clamp"
_EPS = np.finfo(float).eps
_LD = np.longdouble


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _inventory_hash(inventory):
    return _hash(boundary_to_dict(inventory))


def _steps(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError("Buried bank roundoff steps must be a nonnegative integer")
    if 256.*_EPS*(int(value)+1) >= 1.:
        raise ValueError("Buried bank roundoff budget cannot certify geometry")
    return int(value)


def _poses(origin, supplied, steps):
    if isinstance(supplied, Mapping):
        rows = tuple(supplied.items())
        if any(isinstance(owner, (bool, np.bool_)) or not isinstance(owner, Integral)
               or owner < 0 for owner, _ in rows):
            raise ValueError("Buried bank owner maps require explicit nonnegative integer owners")
        rows = tuple(sorted(rows))
    elif isinstance(supplied, tuple):
        rows = supplied
    else:
        raise ValueError("Buried bank owner maps require an explicit mapping or immutable rows")
    owners = tuple(sorted({fragment.parcel.plate for fragment in origin.fragments}))
    if (any(not isinstance(row, tuple) or len(row) != 2 for row in rows)
            or any(isinstance(row[0], (bool, np.bool_)) or not isinstance(row[0], Integral)
                   or row[0] < 0 for row in rows)
            or tuple(row[0] for row in rows) != owners):
        raise ValueError("Buried bank owner maps must name exactly the canonical original owners")
    if any(matrix is None for _, matrix in rows):
        raise ValueError("Buried bank owner maps require explicit SO3 matrices")
    return tuple((int(owner), validate_reference_rotation(matrix,
                  reference_roundoff_steps=steps)) for owner, matrix in rows)


def _pullback(polygon, matrix):
    # Rows multiply Q to apply Q.T to Cartesian column vectors.  Normalization
    # belongs only to these proof coordinates; archived physical bits are kept.
    values = np.asarray(polygon, dtype=_LD)@np.asarray(matrix, dtype=_LD)
    values /= np.sqrt(np.sum(values*values, axis=1))[:, None]
    return tuple(tuple(map(float, row)) for row in values)


@dataclass(frozen=True, slots=True)
class BuriedMaterialReference:
    event_id: str
    material_id: str
    donor_owner: int
    acceptance_rotation: tuple[tuple[float, float, float], ...]
    acceptance_roundoff_steps: int
    loss_payload_sha256: str
    original_polygon: tuple[tuple[float, float, float], ...]
    record_sha256: str


@dataclass(frozen=True, slots=True)
class BuriedBankRegistry:
    origin_surface_sha256: str
    initial_inventory_sha256: str
    current_inventory_sha256: str
    initial_event_payloads: tuple[tuple[str, str], ...]
    records: tuple[BuriedMaterialReference, ...]
    registry_sha256: str


@dataclass(frozen=True, slots=True)
class OriginalMaterialDomain:
    material_id: str
    owner: int
    original_fragment_ids: tuple[str, ...]
    live_fragment_ids: tuple[str, ...]
    buried_event_ids: tuple[str, ...]
    original_measure_sr: float
    conserved_reservoirs: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class LiveClampSupport:
    trace_id: str
    bank: str
    owner: int
    width_m: float
    extent_radians: float
    live_intervals: tuple[tuple[str, float, float], ...]


@dataclass(frozen=True, slots=True)
class BuriedBankSupportReceipt:
    model: str
    origin_surface_sha256: str
    current_surface_sha256: str
    current_inventory_sha256: str
    trace_sha256: str
    bending_source_sha256: str
    registry_sha256: str
    owner_rotations: tuple
    roundoff_steps: int
    original_material_domains: tuple[OriginalMaterialDomain, ...]
    live_clamps: tuple[LiveClampSupport, ...]
    receipt_sha256: str


def _sealed(value, field):
    body = asdict(value)
    body.pop(field)
    return replace(value, **{field: _hash(body)})


def _validate_archive(inventory):
    # This also validates loss/event digests and the complete chronological
    # acceptance registry; a coincident unregistered polygon cannot add mass.
    if boundary_from_dict(boundary_to_dict(inventory)) != inventory:
        raise ValueError("Buried bank archive changed under strict reconstruction")


def initial_buried_bank_registry(origin, inventory):
    """Bind an accepted origin and exclude every already archived cohort.

    Existing cohorts remain in the archive.  Their donor maps at their former
    acceptance times are unknown here, and they are never invented as beam mass.
    """
    _validate_pair(origin, inventory)
    _validate_archive(inventory)
    audit_partition(origin)
    digest = _inventory_hash(inventory)
    events = tuple(sorted((cohort.event_id, cohort.payload_sha256) for cohort in inventory.cohorts))
    return _sealed(BuriedBankRegistry(surface_digest(origin), digest, digest,
        events, (), ""), "registry_sha256")


def _validate_registry(origin, registry, inventory):
    if (not isinstance(registry, BuriedBankRegistry)
            or registry.origin_surface_sha256 != surface_digest(origin)
            or registry.current_inventory_sha256 != _inventory_hash(inventory)
            or _sealed(registry, "registry_sha256") != registry):
        raise ValueError("Buried bank registry binding or integrity changed")
    _validate_archive(inventory)
    if (not isinstance(registry.initial_event_payloads, tuple)
            or any(not isinstance(row, tuple) or len(row) != 2
                   or not isinstance(row[0], str) or not row[0]
                   or not isinstance(row[1], str) or len(row[1]) != 64
                   or any(character not in "0123456789abcdef" for character in row[1])
                   for row in registry.initial_event_payloads)
            or tuple(sorted(set(registry.initial_event_payloads))) != registry.initial_event_payloads):
        raise ValueError("Buried bank initial cohort manifest is not immutable and canonical")
    cohorts = {cohort.event_id: cohort for cohort in inventory.cohorts}
    initial = dict(registry.initial_event_payloads)
    if (len(initial) != len(registry.initial_event_payloads)
            or any(identity not in cohorts or cohorts[identity].payload_sha256 != digest
                   or cohorts[identity].acceptance_time_myr > origin.time_myr
                   for identity, digest in initial.items())):
        raise ValueError("Buried bank original archive manifest changed")
    if (not isinstance(registry.records, tuple)
            or any(not isinstance(record, BuriedMaterialReference) for record in registry.records)
            or len({record.event_id for record in registry.records}) != len(registry.records)
            or {record.event_id for record in registry.records} != set(cohorts)-set(initial)):
        raise ValueError("Buried bank registry must contain each new accepted cohort exactly once")
    order = []
    owners = {fragment.parcel.plate for fragment in origin.fragments}
    for record in registry.records:
        cohort = cohorts[record.event_id]
        steps = _steps(record.acceptance_roundoff_steps)
        if record.acceptance_rotation is None:
            raise ValueError("Buried bank acceptance map requires an explicit SO3 matrix")
        matrix = validate_reference_rotation(record.acceptance_rotation,
                                             reference_roundoff_steps=steps)
        polygon = record.original_polygon
        if (not isinstance(record.acceptance_rotation, tuple)
                or any(not isinstance(row, tuple) for row in record.acceptance_rotation)
                or not isinstance(polygon, tuple) or len(polygon) < 3
                or any(not isinstance(row, tuple) or len(row) != 3 for row in polygon)
                or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                       or not math.isfinite(value) for row in polygon for value in row)
                or isinstance(record.donor_owner, (bool, np.bool_))
                or not isinstance(record.donor_owner, Integral) or record.donor_owner not in owners
                or record.material_id != cohort.parcel.material_id
                or record.donor_owner != cohort.subducting_plate
                or record.loss_payload_sha256 != cohort.payload_sha256
                or record.original_polygon != _pullback(cohort.loss.fragment.polygon, matrix)
                or _sealed(record, "record_sha256") != record
                or cohort.acceptance_time_myr <= origin.time_myr):
            raise ValueError("Buried bank loss payload or acceptance donor map changed")
        order.append((cohort.acceptance_time_myr, record.event_id))
    if order != sorted(order):
        raise ValueError("Buried bank acceptance records must preserve chronological order")
    return cohorts


def advance_buried_bank_registry(origin, previous, before_inventory, after_inventory,
                                 owner_rotations_at_endpoint, *, roundoff_steps=0):
    """Register exactly the new losses of one actual chronological transaction."""
    steps = _steps(roundoff_steps)
    poses = dict(_poses(origin, owner_rotations_at_endpoint, steps))
    old = _validate_registry(origin, previous, before_inventory)
    _validate_archive(after_inventory)
    new = {cohort.event_id: cohort for cohort in after_inventory.cohorts}
    if (after_inventory.radius_km != origin.radius_km
            or after_inventory.time_myr <= before_inventory.time_myr
            or len(after_inventory.transactions) != len(before_inventory.transactions)+1
            or after_inventory.transactions[:-1] != before_inventory.transactions
            or after_inventory.representations != before_inventory.representations
            or any(identity not in new or new[identity].loss != cohort.loss
                   or new[identity].payload_sha256 != cohort.payload_sha256
                   for identity, cohort in old.items())):
        raise ValueError("Buried bank registration needs one immutable chronological acceptance transaction")
    event_ids = set(new)-set(old)
    transaction = after_inventory.transactions[-1]
    if (event_ids != set(transaction.event_ids)
            or transaction.start_time_myr != before_inventory.time_myr
            or transaction.end_time_myr != after_inventory.time_myr):
        raise ValueError("Buried bank new losses disagree with the actual transaction registry")
    records = list(previous.records)
    for identity in sorted(event_ids):
        cohort = new[identity]
        if cohort.subducting_plate not in poses:
            raise ValueError("Buried donor has no original owner map")
        matrix = poses[cohort.subducting_plate]
        records.append(_sealed(BuriedMaterialReference(identity, cohort.parcel.material_id,
            cohort.subducting_plate, matrix, steps, cohort.payload_sha256,
            _pullback(cohort.loss.fragment.polygon, matrix), ""), "record_sha256"))
    result = _sealed(replace(previous, current_inventory_sha256=_inventory_hash(after_inventory),
                            records=tuple(records)), "registry_sha256")
    _validate_registry(origin, result, after_inventory)
    return result


def _subtract_all(polygon, covers):
    remaining = (np.asarray(polygon),)
    for cover in covers:
        remaining = tuple(piece for item in remaining for piece in subtract_convex(item, cover))
        if not remaining:
            break
    return remaining


def _align_pullback(polygon, original_polygons, physical_polygon, matrix, steps):
    """Resolve only inverse-rotation coordinate uncertainty at original edges.

    Bounds follow the actual matrix products component by component.  There is
    no angular/area floor: a resolved positive sliver outside its donor rejects.
    This proof does not modify a saved physical polygon or any material value.
    """
    result = np.asarray(polygon, dtype=_LD).copy()
    matrix = np.asarray(matrix, dtype=_LD)
    errors = _LD(256.*_EPS*(steps+1))*(np.abs(np.asarray(physical_polygon, dtype=_LD))
                                    @np.abs(matrix)+np.abs(result))
    for index, vertex in enumerate(result):
        for original in original_polygons:
            vertices = np.asarray(original, dtype=_LD)
            for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
                normal = np.cross(a, b-a)
                normal /= np.sqrt(np.sum(normal*normal))
                distance = np.dot(normal, vertex)
                bound = np.dot(np.abs(normal), errors[index])
                if abs(distance) <= bound:
                    vertex = vertex-distance*normal
                    vertex /= np.sqrt(np.sum(vertex*vertex))
            for candidate in vertices:
                if np.all(np.abs(vertex-candidate) <= errors[index]):
                    vertex = candidate.copy()
                    break
        result[index] = vertex
    return np.asarray(result, dtype=float)


def _constant_width_quad(trace, width_m, radius_m, sign):
    start, horizontal, _, extent = _arc_frame(trace.start, trace.end)
    width = width_m/radius_m
    if not 0. < width < math.pi/2.:
        raise UnresolvedFaultMotionSupport("Buried original bank width/radius must be below pi/2")
    end = np.asarray(trace.end)
    clamps = tuple(point*math.cos(width)+sign*horizontal*math.sin(width)
                   for point in (start, end))
    return normalize_polygon((start, end, clamps[1], clamps[0])), extent


def _clamp_position(start, tangent, horizontal, width, sign, angle):
    vector = (np.asarray(start, dtype=_LD)*np.cos(_LD(angle))
              +np.asarray(tangent, dtype=_LD)*np.sin(_LD(angle)))*np.cos(_LD(width))
    vector += _LD(sign)*np.asarray(horizontal, dtype=_LD)*np.sin(_LD(width))
    vector /= np.sqrt(np.sum(vector*vector))
    return np.asarray(vector, dtype=float)


def _clamp_intervals(trace, width_m, radius_m, sign, polygon):
    """All positive intervals of a small-circle clamp inside one live polygon."""
    start, horizontal, tangent, extent = _arc_frame(trace.start, trace.end)
    width = width_m/radius_m
    vertices = np.asarray(polygon)
    cuts = [0., extent]
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        normal = np.cross(a.astype(_LD), (b-a).astype(_LD))
        normal /= np.sqrt(np.sum(normal*normal))
        aa = np.dot(normal, np.asarray(start, dtype=_LD))*np.cos(_LD(width))
        bb = np.dot(normal, np.asarray(tangent, dtype=_LD))*np.cos(_LD(width))
        cc = _LD(sign)*np.dot(normal, np.asarray(horizontal, dtype=_LD))*np.sin(_LD(width))
        amplitude = np.hypot(aa, bb)
        if amplitude == 0. or abs(cc) > amplitude:
            continue
        phase, alpha = np.arctan2(bb, aa), np.arccos(-cc/amplitude)
        for root in (phase-alpha, phase+alpha):
            for winding in range(-2, 3):
                angle = float(root+_LD(2.*math.pi*winding))
                if 0. < angle < extent:
                    cuts.append(angle)
    output = []
    cuts = sorted(set(cuts))
    for low, high in zip(cuts, cuts[1:]):
        angle = _LD(low)+(_LD(high)-_LD(low))/_LD(2.)
        position = _clamp_position(start, tangent, horizontal, width, sign, angle)
        if all(_edge_distance(position, a, b, None) >= 0.
               for a, b in zip(vertices, np.roll(vertices, -1, axis=0))):
            output.append((low, high))
    return tuple(output), extent


def _cover_clamp(intervals, extent, steps):
    cursor = 0.
    # This is an explicit accumulated arithmetic budget on analytic roots.
    tolerance = 128.*_EPS*(steps+1)*extent
    for _, low, high in sorted(intervals, key=lambda row: (row[1], row[2], row[0])):
        if low-cursor > tolerance:
            raise UnresolvedFaultMotionSupport("Original bending clamp contains buried or newborn material")
        cursor = max(cursor, high)
    if extent-cursor > tolerance:
        raise UnresolvedFaultMotionSupport("Original bending clamp is no longer supported by LIVE original material")


def validate_buried_bank_support(origin, seed, current_surface, current_inventory,
                                 owner_rotations, registry, *, roundoff_steps=0):
    """Certify retained original beam mass and the complete live clamp curve."""
    steps = _steps(roundoff_steps)
    poses = _poses(origin, owner_rotations, steps)
    matrices = dict(poses)
    if (not isinstance(seed, BendingContactState) or seed.surface_sha256 != surface_digest(origin)
            or current_surface.radius_km != origin.radius_km
            or current_surface.time_myr < origin.time_myr):
        raise ValueError("Buried bank support requires its actual original mechanical configuration")
    bounds = tuple(getattr(seed, name) for name in ("max_rotation_radians", "max_tip_slope",
                   "max_bending_strain", "max_deflection_radius_ratio"))
    if seed.bending_source_sha256 != _bending_signature(origin, seed.fault_traces,
                                                       seed.bending_parameters, bounds):
        raise ValueError("Buried bank original bending parameters or trace signature changed")
    _validate_pair(current_surface, current_inventory)
    audit_partition(origin)
    audit_partition(current_surface)
    cohorts = _validate_registry(origin, registry, current_inventory)
    validate_reference_bending_bank_support(origin, seed.fault_traces, seed.bending_parameters)
    originals = {}
    for fragment in origin.fragments:
        originals.setdefault(fragment.parcel.material_id, []).append(fragment)
    live, buried = {identity: [] for identity in originals}, {identity: [] for identity in originals}
    for fragment in current_surface.fragments:
        if fragment.parcel.material_id in originals:
            live[fragment.parcel.material_id].append(fragment)
    for record in registry.records:
        if record.material_id in originals:
            buried[record.material_id].append((record, cohorts[record.event_id]))
    domains, live_geometry = [], []
    for identity, parents in sorted(originals.items()):
        parcel = parents[0].parcel
        if any((parent.parcel.plate, parent.parcel.material_fields, parent.parcel.specific_properties,
                parent.parcel.age_myr)
               != (parcel.plate, parcel.material_fields, parcel.specific_properties, parcel.age_myr)
               for parent in parents):
            raise UnresolvedFaultMotionSupport("One original material ID has heterogeneous beam properties")
        original_polygons = tuple(parent.polygon for parent in parents)
        pieces, carried = [], []
        for fragment in sorted(live[identity], key=lambda row: row.fragment_id):
            if fragment.parcel.plate != parcel.plate:
                raise ValueError("Original live beam material changed its owner")
            matrix = matrices[parcel.plate]
            polygon = _align_pullback(_pullback(fragment.polygon, matrix), original_polygons,
                                      fragment.polygon, matrix, steps)
            pieces.append((fragment.fragment_id, polygon))
            carried.append((fragment.parcel, current_surface.time_myr-origin.time_myr))
            live_geometry.append((fragment.fragment_id, parcel.plate, polygon))
        for record, cohort in buried[identity]:
            if record.donor_owner != parcel.plate:
                raise ValueError("Original buried beam material changed its owner")
            polygon = _align_pullback(record.original_polygon, original_polygons,
                cohort.loss.fragment.polygon, record.acceptance_rotation, record.acceptance_roundoff_steps)
            pieces.append((record.event_id, polygon))
            carried.append((cohort.parcel, cohort.acceptance_time_myr-origin.time_myr))
        for material, elapsed in carried:
            if (material.material_fields != parcel.material_fields
                    or material.specific_properties != parcel.specific_properties):
                raise ValueError("Original beam material changed its intensive history")
            expected_age = parcel.age_myr+elapsed
            age_bound = 256.*_EPS*(steps+1)*max(expected_age, material.age_myr,
                                              elapsed, np.finfo(float).tiny)
            if abs(material.age_myr-expected_age) > age_bound:
                raise ValueError("Original beam material changed its accepted age history")
        for name, polygon in pieces:
            if _subtract_all(polygon, original_polygons):
                raise UnresolvedFaultMotionSupport(f"Original material {identity} extends outside its 2D footprint")
        for index, (_, polygon) in enumerate(pieces):
            if any(len(intersect_convex(polygon, other)) for _, other in pieces[index+1:]):
                raise UnresolvedFaultMotionSupport(f"Original material {identity} live/buried domains overlap")
        for parent in parents:
            if _subtract_all(parent.polygon, tuple(polygon for _, polygon in pieces)):
                raise UnresolvedFaultMotionSupport(f"Original material {identity} has missing 2D beam support")
        measured = math.fsum(polygon_area(polygon) for _, polygon in pieces)
        original_measure = math.fsum(polygon_area(parent.polygon) for parent in parents)
        if not math.isclose(measured, original_measure, rel_tol=AREA_REL_TOL, abs_tol=0.):
            raise UnresolvedFaultMotionSupport("Original material geometric measure is not conserved")
        reservoirs = []
        for field in EXTENSIVE_FIELDS:
            expected = math.fsum(getattr(parent.parcel, field) for parent in parents)
            actual = math.fsum(getattr(material, field) for material, _ in carried)
            if not math.isclose(actual, expected, rel_tol=64.*_EPS*(steps+1), abs_tol=0.):
                raise ValueError(f"Original material {identity} lost its {field} reservoir")
            reservoirs.append((field, expected))
        domains.append(OriginalMaterialDomain(identity, parcel.plate,
            tuple(sorted(parent.fragment_id for parent in parents)),
            tuple(sorted(fragment.fragment_id for fragment in live[identity])),
            tuple(sorted(record.event_id for record, _ in buried[identity])),
            original_measure, tuple(reservoirs)))
    parameters = {item.trace_id: item for item in seed.bending_parameters}
    clamps = []
    for trace in seed.fault_traces.traces:
        parameter = parameters[trace.trace_id]
        for bank, owner, material, sign in (("sub", trace.subducting_plate, parameter.sub, -1.),
                                           ("over", trace.overriding_plate, parameter.over, 1.)):
            quad, extent = _constant_width_quad(trace, material.width_m, origin.radius_km*1000., sign)
            if _subtract_all(quad, tuple(fragment.polygon for fragment in origin.fragments
                                        if fragment.parcel.plate == owner)):
                raise UnresolvedFaultMotionSupport("Cannot certify the complete 2D original bending strip")
            intervals = []
            for identity, live_owner, polygon in live_geometry:
                if live_owner == owner:
                    covered, _ = _clamp_intervals(trace, material.width_m,
                                                origin.radius_km*1000., sign, polygon)
                    intervals.extend((identity, low, high) for low, high in covered)
            _cover_clamp(intervals, extent, steps)
            clamps.append(LiveClampSupport(trace.trace_id, bank, owner, material.width_m,
                                           extent, tuple(sorted(intervals))))
    return _sealed(BuriedBankSupportReceipt(SUPPORT_MODEL, surface_digest(origin),
        surface_digest(current_surface), _inventory_hash(current_inventory),
        seed.fault_traces.trace_sha256, seed.bending_source_sha256, registry.registry_sha256,
        poses, steps, tuple(domains), tuple(clamps), ""), "receipt_sha256")


def buried_bank_registry_to_dict(registry):
    """Encode immutable proof records; callers validate against their archive."""
    if not isinstance(registry, BuriedBankRegistry) or _sealed(registry, "registry_sha256") != registry:
        raise ValueError("Invalid buried bank registry integrity")
    return dict(format=FORMAT, registry=json.loads(json.dumps(asdict(registry), allow_nan=False)))


def buried_bank_registry_from_dict(raw, origin, inventory):
    """Restore exact acceptance maps and prove they bind the actual loss payloads."""
    if not isinstance(raw, dict) or set(raw) != {"format", "registry"} or raw["format"] != FORMAT:
        raise ValueError("Invalid buried bank registry schema")
    body = raw["registry"]
    if not isinstance(body, dict) or set(body) != {field.name for field in fields(BuriedBankRegistry)}:
        raise ValueError("Invalid buried bank registry fields")
    if not isinstance(body["records"], list) or not isinstance(body["initial_event_payloads"], list):
        raise ValueError("Buried bank registry requires explicit lists")
    records = []
    for row in body["records"]:
        if not isinstance(row, dict) or set(row) != {field.name for field in fields(BuriedMaterialReference)}:
            raise ValueError("Invalid buried material reference fields")
        try:
            records.append(BuriedMaterialReference(**{**row,
                "acceptance_rotation": tuple(tuple(item) for item in row["acceptance_rotation"]),
                "original_polygon": tuple(tuple(item) for item in row["original_polygon"])}))
        except TypeError as error:
            raise ValueError("Invalid buried material coordinate lists") from error
    try:
        result = BuriedBankRegistry(**{**body, "records": tuple(records),
            "initial_event_payloads": tuple(tuple(item) for item in body["initial_event_payloads"])})
    except TypeError as error:
        raise ValueError("Invalid buried bank initial cohort list") from error
    _validate_registry(origin, result, inventory)
    return result
