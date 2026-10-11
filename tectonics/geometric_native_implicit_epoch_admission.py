"""Fresh native witnesses that refuse unsupported newborn force contacts.

Every geometric calculation uses the current birth-clock footprint and the
new, complete Euler vectors.  Positive endpoint intersections are sufficient
counterexamples to blanket exclusion, not substitutes for a temporal mask.
An exact shared edge with positive incoming flux supplies the stronger
statement: unsupported contact begins at arbitrarily small positive time.

HotBirthParameters do not supply contact preload, bending banks, inclination
or an evolved receiving field.  This module therefore produces a refusal,
never a force acceptance.  The immutable previous force/material state stays
external; a disposable material continuation is separately authenticated.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction as F
import hashlib
import json
import math

from .geometric_native_contact_reference_transport import _proof_key
from .geometric_native_implicit_material_epoch import (
    NativeImplicitMaterialEpoch, validate_native_implicit_material_epoch,
)
from .geometric_native_physical_pair_time import _cross
from .geometric_native_physical_history import _literal_json, _typed_fingerprint
from .geometric_native_polygons import (
    _plane_values, _ray_values, _same_plane, native_arc_to_dict,
    native_intersection, native_polygon_to_dict, native_shared_boundary_arcs,
)
from .geometric_native_pose import FractionInterval as I, interval_sqrt
from .thermal import SECONDS_PER_MYR


FORMAT = "native-fresh-implicit-epoch-newborn-admission-refusal-1"
GEOMETRY_FORMAT = "native-fresh-implicit-epoch-newborn-incoming-geometry-1"
_GEOMETRY_CACHE = {}
_REFUSAL_CACHE = {}


class UnresolvedNativeImplicitEpochAdmission(ValueError):
    """No declared physical family admits the witnessed fresh newborn pair."""


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()


def _hash(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _portable_fingerprint_sha256(value):
    """Stable complete immutable values; process class identity stays in caches."""
    digest = hashlib.sha256()

    def visit(item):
        kind = type(item)
        if isinstance(item, type):
            payload, tag = (item.__module__+":"+item.__qualname__).encode(), b"class"
        elif kind is tuple:
            digest.update(b"tuple[")
            for child in item:
                visit(child)
            digest.update(b"]")
            return
        elif kind is bytes:
            payload, tag = item, b"bytes"
        elif kind is str:
            payload, tag = item.encode(), b"str"
        elif kind is bool:
            payload, tag = str(item).encode("ascii"), b"bool"
        elif kind is int:
            payload, tag = str(item).encode("ascii"), b"int"
        elif item is None:
            payload, tag = b"", b"none"
        else:
            raise ValueError("Portable native fingerprint contains an unsupported field")
        digest.update(tag+b":"+str(len(payload)).encode("ascii")+b":"+payload+b";")

    visit(_typed_fingerprint(value))
    return digest.hexdigest()


def _r(value):
    return [value.numerator, value.denominator]


def _i(value):
    return dict(lo=_r(value.lo), hi=_r(value.hi))


def _dot(first, second):
    return sum((a*b for a, b in zip(first, second)), I(0))


def _unit_ray(ray, precision):
    values = _ray_values(ray, precision)[1]
    norm = interval_sqrt(_dot(values, values), precision+16)
    if norm.lo <= 0:
        raise UnresolvedNativeImplicitEpochAdmission("An inherited shared-edge ray is not proved nonzero")
    return tuple(value/norm for value in values)


def _omega(epoch, records, dt_myr, precision):
    owners = tuple(sorted({component.owner for component in epoch.components}))
    if (type(records) is not tuple
            or any(type(row) is not tuple or len(row) != 2
                or type(row[0]) is not int or type(row[1]) is not tuple
                or len(row[1]) != 3 or any(type(x) is not float or not math.isfinite(x)
                    for x in row[1]) for row in records)
            or tuple(row[0] for row in records) != owners):
        raise ValueError("Newborn admission needs every saved literal full Euler vector in canonical order")
    if type(dt_myr) is not float or not math.isfinite(dt_myr) or dt_myr <= 0:
        raise ValueError("Newborn admission needs its new literal positive physical duration")
    if type(precision) is not int or not 128 <= precision <= 1024:
        raise ValueError("Newborn admission precision must be a literal integer from 128 to 1024")
    return {owner: tuple(F.from_float(x) for x in vector) for owner, vector in records}


@dataclass(frozen=True, slots=True)
class NativeImplicitImmediateIncomingEdge:
    edge: object
    receiver_inward_plane: object
    endpoint_unit_direction_bounds: tuple
    endpoint_incoming_flux_bounds_per_myr: tuple
    interior_direction_bounds: tuple
    donor_other_support_bounds: tuple
    receiver_other_support_bounds: tuple
    interior_incoming_flux_bounds_per_myr: I


@dataclass(frozen=True, slots=True)
class NativeImplicitPositiveNewbornContact:
    donor_component_id: str
    donor_material_id: str
    donor_original_parent_id: str
    donor_owner: int
    donor_material_kind: str
    donor_piece_sequence: int
    receiver_component_id: str
    receiver_material_id: str
    receiver_original_parent_id: str
    receiver_owner: int
    receiver_material_kind: str
    receiver_piece_sequence: int
    initial_donor_geometry: object
    initial_receiver_geometry: object
    endpoint_intersection_geometry: object
    endpoint_intersection_area_bounds_sr: I
    immediate_incoming_edges: tuple


@dataclass(frozen=True, slots=True)
class NativeImplicitEpochIncomingGeometry:
    source_epoch: object
    omega_rad_per_myr: tuple
    dt_myr: float
    precision: int
    positive_contacts: tuple
    geometry_sha256: str

    @property
    def positive_newborn_contact_proved(self):
        return bool(self.positive_contacts)

    @property
    def immediate_unsupported_incoming_proved(self):
        return any(contact.immediate_incoming_edges for contact in self.positive_contacts)

    @property
    def force_admission_authorized(self):
        return False


def _immediate_edges(donor, receiver, donor_omega, receiver_omega, precision):
    result = []
    difference = tuple(a-b for a, b in zip(donor_omega, receiver_omega))
    for edge in native_shared_boundary_arcs(donor, receiver):
        planes = tuple(plane for plane in receiver.support_planes if _same_plane(plane, edge.plane))
        if len(planes) != 1:
            continue
        inward = planes[0]
        normal = _plane_values(inward, precision)[1]
        points = tuple(_unit_ray(ray, precision) for ray in (edge.start, edge.end))
        flux = tuple(_dot(normal, _cross(difference, point)) for point in points)
        if any(value.lo <= 0 for value in flux):
            continue
        # This is an analytic interior edge expression, not a root midpoint
        # and not a finite material point assigned an area or work amount.
        interior = tuple((a+b)/2 for a, b in zip(*points))
        donor_bounds = tuple((index, _dot(_plane_values(plane, precision)[1], interior))
            for index, plane in enumerate(donor.support_planes) if not _same_plane(plane, edge.plane))
        receiver_bounds = tuple((index, _dot(_plane_values(plane, precision)[1], interior))
            for index, plane in enumerate(receiver.support_planes) if not _same_plane(plane, edge.plane))
        if not donor_bounds or not receiver_bounds or any(value.lo <= 0
                for _, value in donor_bounds+receiver_bounds):
            continue
        incoming = _dot(normal, _cross(difference, interior))
        if incoming.lo <= 0:
            continue
        result.append(NativeImplicitImmediateIncomingEdge(edge, inward, points, flux,
            interior, donor_bounds, receiver_bounds, incoming))
    return tuple(result)


def certify_native_implicit_epoch_incoming_geometry(epoch, omega_rad_per_myr, dt_myr,
        *, precision=256):
    """Find exact positive-area counterexamples for the supplied fresh motion.

    A proxy with a negative curved correction is excluded from this witness
    search: its positive intersection could contain already archived material.
    A positive hot correction adds material, so its polygon base is an exact
    lower subset and remains valid.  Absence of a witness is inconclusive.
    """
    if type(epoch) is not NativeImplicitMaterialEpoch:
        raise ValueError("This first continuation witness requires the exact initial implicit partition")
    epoch = validate_native_implicit_material_epoch(epoch)
    omega = _omega(epoch, omega_rad_per_myr, dt_myr, precision)
    key = _proof_key((epoch, omega_rad_per_myr, dt_myr, precision))
    if key in _GEOMETRY_CACHE:
        cached, fingerprint, body, seal = _GEOMETRY_CACHE[key]
        if (_typed_fingerprint(cached) != fingerprint or cached.geometry_sha256 != seal
                or _encoded(_geometry_body(cached)) != body):
            raise ValueError("Cached newborn geometry differs from its independently saved complete proof")
        return cached
    direction = epoch.source_material_state.sweep_certificate
    donors = tuple(component for component in epoch.components if component.owner == direction.donor_owner)
    receivers = tuple(component for component in epoch.components if component.owner == direction.receiver_owner)
    negative = {correction.negative_component_id for correction in epoch.paired_corrections}
    contacts = []
    for donor in donors:
        if donor.component_id in negative:
            continue
        for receiver in receivers:
            if receiver.component_id in negative or (donor.material_kind != "hot_birth"
                    and receiver.material_kind != "hot_birth"):
                continue
            for di, dg in enumerate(donor.proxy_geometries):
                for ri, rg in enumerate(receiver.proxy_geometries):
                    initial = native_intersection(dg, rg)
                    if initial is not None and initial.area_bounds().hi > 0:
                        raise ValueError("The trusted initial newborn partition already has positive material overlap")
                    moved_donor = dg.rotated(dict(omega_rad_per_myr)[donor.owner], dt_myr)
                    moved_receiver = rg.rotated(dict(omega_rad_per_myr)[receiver.owner], dt_myr)
                    overlap = native_intersection(moved_donor, moved_receiver)
                    if overlap is None:
                        continue
                    area = overlap.area_bounds()
                    if area.lo <= 0:
                        continue
                    contacts.append(NativeImplicitPositiveNewbornContact(donor.component_id,
                        donor.material_id, donor.original_parent_id, donor.owner,
                        donor.material_kind, di, receiver.component_id, receiver.material_id,
                        receiver.original_parent_id, receiver.owner, receiver.material_kind,
                        ri, dg, rg, overlap, area, _immediate_edges(dg, rg,
                            omega[donor.owner], omega[receiver.owner], precision)))
    value = NativeImplicitEpochIncomingGeometry(epoch, omega_rad_per_myr, dt_myr,
        precision, tuple(contacts), "")
    value = replace(value, geometry_sha256=_hash(_geometry_body(value)))
    if len(_GEOMETRY_CACHE) >= 8:
        _GEOMETRY_CACHE.clear()
    _GEOMETRY_CACHE[key] = (value, _typed_fingerprint(value),
        _encoded(_geometry_body(value)), value.geometry_sha256)
    return value


def _edge_body(edge):
    from .geometric_native_polygons import native_plane_to_dict
    return dict(exact_initial_shared_edge=native_arc_to_dict(edge.edge),
        receiver_inward_plane=native_plane_to_dict(edge.receiver_inward_plane),
        endpoint_unit_direction_bounds=[[_i(x) for x in point] for point in edge.endpoint_unit_direction_bounds],
        endpoint_incoming_flux_bounds_per_myr=[_i(value) for value in edge.endpoint_incoming_flux_bounds_per_myr],
        interior_direction_bounds=[_i(value) for value in edge.interior_direction_bounds],
        donor_other_support_bounds=[dict(index=index, bounds=_i(value)) for index, value in edge.donor_other_support_bounds],
        receiver_other_support_bounds=[dict(index=index, bounds=_i(value)) for index, value in edge.receiver_other_support_bounds],
        interior_incoming_flux_bounds_per_myr=_i(edge.interior_incoming_flux_bounds_per_myr),
        initial_shared_support_zero_is_native_expression_identity=True,
        proof="strict other supports and positive edge flux persist by smoothness; donor-side interior perturbation gives an open positive-area overlap for arbitrarily small positive time",
        safe_positive_duration_with_same_euler_vectors_exists=False,
        root_midpoint_selected=False, isolated_point_area_or_work_assigned=False)


def _geometry_body(value):
    return dict(format=GEOMETRY_FORMAT, source_epoch_sha256=value.source_epoch.epoch_sha256,
        complete_source_epoch_fingerprint_sha256=_portable_fingerprint_sha256(value.source_epoch),
        source_material_state_sha256=value.source_epoch.source_material_state.state_sha256,
        omega_rad_per_myr=[[owner, list(vector)] for owner, vector in value.omega_rad_per_myr],
        dt_myr=value.dt_myr, exact_literal_dt_myr=_r(F.from_float(value.dt_myr)), precision=value.precision,
        positive_contacts=[dict(**{name: getattr(contact, name) for name in (
            "donor_component_id", "donor_material_id", "donor_original_parent_id", "donor_owner",
            "donor_material_kind", "donor_piece_sequence", "receiver_component_id", "receiver_material_id",
            "receiver_original_parent_id", "receiver_owner", "receiver_material_kind", "receiver_piece_sequence")},
            initial_donor_geometry=native_polygon_to_dict(contact.initial_donor_geometry),
            initial_receiver_geometry=native_polygon_to_dict(contact.initial_receiver_geometry),
            initial_positive_area_intersection=False,
            endpoint_intersection_geometry=native_polygon_to_dict(contact.endpoint_intersection_geometry),
            endpoint_intersection_area_bounds_sr=_i(contact.endpoint_intersection_area_bounds_sr),
            immediate_incoming_edges=[_edge_body(edge) for edge in contact.immediate_incoming_edges])
            for contact in value.positive_contacts],
        positive_newborn_contact_proved=value.positive_newborn_contact_proved,
        immediate_unsupported_incoming_proved=value.immediate_unsupported_incoming_proved,
        full_prior_ordered_SO3_and_actual_birth_clock_preserved=True,
        previous_interval_root_certificates_reused=False,
        positive_measure_discarded_by_tolerance=False,
        negative_curved_correction_proxy_used_as_positive_witness=False,
        complete_all_pairs_exclusion_proved=False, force_admission_authorized=False)


def native_implicit_epoch_incoming_geometry_to_dict(value):
    if type(value) is not NativeImplicitEpochIncomingGeometry:
        raise ValueError("Expected a complete typed fresh newborn incoming geometry witness")
    replay = certify_native_implicit_epoch_incoming_geometry(value.source_epoch,
        value.omega_rad_per_myr, value.dt_myr, precision=value.precision)
    if _typed_fingerprint(replay) != _typed_fingerprint(value):
        raise ValueError("Fresh newborn geometry differs from complete native source replay")
    return dict(_geometry_body(value), geometry_sha256=value.geometry_sha256)


def native_implicit_epoch_incoming_geometry_from_dict(raw, *, expected_epoch,
        expected_omega_rad_per_myr, expected_dt_myr):
    """Replay the refusal geometry against independently supplied birth history."""
    _literal_json(raw)
    if type(raw) is not dict or "precision" not in raw:
        raise ValueError("Newborn geometry restart requires its complete literal JSON receipt")
    replay = certify_native_implicit_epoch_incoming_geometry(expected_epoch,
        expected_omega_rad_per_myr, expected_dt_myr, precision=raw["precision"])
    if _encoded(raw) != _encoded(native_implicit_epoch_incoming_geometry_to_dict(replay)):
        raise ValueError("Newborn geometry differs from the independent complete source and fresh motion")
    return replay


@dataclass(frozen=True, slots=True)
class NativeImplicitEpochAdmissionRefusal:
    previous: object
    candidate: object
    load: object
    material_epoch: object
    geometry: NativeImplicitEpochIncomingGeometry
    candidate_constitutive_branches: tuple
    refusal_sha256: str

    @property
    def positive_newborn_contact_proved(self):
        return self.geometry.positive_newborn_contact_proved

    @property
    def immediate_unsupported_incoming_proved(self):
        return self.geometry.immediate_unsupported_incoming_proved

    @property
    def physical_pair_admission_authorized(self):
        return False


def certify_native_implicit_epoch_admission(previous, candidate, material_epoch, *, load,
        precision=256):
    """Authenticate the genuine fresh force and return its material refusal."""
    from .geometric_native_implicit_underthrust_motion import validate_native_implicit_force_candidate
    from .geometric_native_implicit_material_continuation import validate_native_implicit_material_continuation
    validate_native_implicit_force_candidate(candidate, previous=previous, load=load)
    material = validate_native_implicit_material_continuation(material_epoch)
    if (_typed_fingerprint(material.previous_epoch) != _typed_fingerprint(previous.current_surface)
            or _typed_fingerprint(material.omega_rad_per_myr) != _typed_fingerprint(candidate.omega_rad_per_myr)
            or type(material.dt_myr) is not float or material.dt_myr != candidate.dt_s/SECONDS_PER_MYR):
        raise ValueError("Fresh admission material trial differs from the complete genuine previous state and new force")
    geometry = certify_native_implicit_epoch_incoming_geometry(previous.current_surface,
        candidate.omega_rad_per_myr, candidate.dt_s/SECONDS_PER_MYR, precision=precision)
    if not geometry.positive_newborn_contact_proved:
        raise UnresolvedNativeImplicitEpochAdmission("No newborn counterexample was proved; complete fresh temporal physical admission is still required")
    branches = tuple((trace.trace_id, tuple((piece.low_radians, piece.high_radians, piece.branch)
        for piece in trace.pieces)) for trace in candidate.spatial_step.certificates)
    value = NativeImplicitEpochAdmissionRefusal(previous, candidate, load, material,
        geometry, branches, "")
    value = replace(value, refusal_sha256=_hash(_refusal_body(value)))
    if len(_REFUSAL_CACHE) >= 8:
        _REFUSAL_CACHE.clear()
    _REFUSAL_CACHE[_proof_key(value)] = True
    return value


def _refusal_body(value):
    return dict(format=FORMAT, status="UNRESOLVED_NEWBORN_PHYSICAL_CONTACT",
        previous_state_sha256=value.previous.state_sha256,
        complete_previous_state_fingerprint_sha256=_portable_fingerprint_sha256(value.previous),
        complete_fresh_candidate_fingerprint_sha256=_portable_fingerprint_sha256(value.candidate),
        complete_current_load_fingerprint_sha256=_portable_fingerprint_sha256(value.load),
        current_load_sha256=value.load.load_sha256,
        material_candidate_epoch_sha256=value.material_epoch.epoch_sha256,
        complete_material_candidate_fingerprint_sha256=_portable_fingerprint_sha256(value.material_epoch),
        dt_s=value.candidate.dt_s, candidate_provenance=value.candidate.provenance,
        newborn_geometry=native_implicit_epoch_incoming_geometry_to_dict(value.geometry),
        candidate_constitutive_branches=[dict(trace_id=identity, pieces=[dict(low_radians=low,
            high_radians=high, branch=branch) for low, high, branch in pieces])
            for identity, pieces in value.candidate_constitutive_branches],
        missing_physical_inputs=["newborn receiving field and constitutive memory",
            "newborn donor/receiver bending bank parameters", "physical fault-family attachment and inclination", "contact preload and reference state"],
        HotBirthParameters_supply_newborn_contact_mechanics=False,
        original_cold_receiving_field_assigned_to_newborn=False,
        previous_full_line_down_dip_proof_reused=False,
        whole_next_receiving_line_down_dip_proved=False,
        previous_accepted_interval_count=len(value.previous.receipts),
        candidate_mechanics_adopted=False, material_candidate_committed=False,
        physical_pair_admission_authorized=False,
        previous_force_memory_work_and_parent_reservoirs_changed=False)


def validate_native_implicit_epoch_admission_refusal(value, *, expected_previous,
        expected_candidate, expected_load, expected_material_epoch):
    if type(value) is not NativeImplicitEpochAdmissionRefusal:
        raise ValueError("Expected the typed complete fresh material-admission refusal")
    replay = certify_native_implicit_epoch_admission(expected_previous, expected_candidate,
        expected_material_epoch, load=expected_load, precision=value.geometry.precision)
    if _typed_fingerprint(value) != _typed_fingerprint(replay):
        raise ValueError("Fresh admission refusal differs from its independent complete sources")
    return value


def native_implicit_epoch_admission_refusal_to_dict(value):
    validate_native_implicit_epoch_admission_refusal(value, expected_previous=value.previous,
        expected_candidate=value.candidate, expected_load=value.load, expected_material_epoch=value.material_epoch)
    return dict(_refusal_body(value), refusal_sha256=value.refusal_sha256)


def native_implicit_epoch_admission_refusal_from_dict(raw, *, expected_previous,
        expected_candidate, expected_load, expected_material_epoch):
    _literal_json(raw)
    if type(raw) is not dict:
        raise ValueError("Fresh admission refusal restart requires a complete literal JSON object")
    try:
        precision = raw["newborn_geometry"]["precision"]
    except (KeyError, TypeError) as error:
        raise ValueError("Fresh admission refusal omits its independently replayed geometry budget") from error
    replay = certify_native_implicit_epoch_admission(expected_previous, expected_candidate,
        expected_material_epoch, load=expected_load, precision=precision)
    if _encoded(raw) != _encoded(native_implicit_epoch_admission_refusal_to_dict(replay)):
        raise ValueError("Fresh admission refusal differs from the full genuine force, loading and material replay")
    return replay
