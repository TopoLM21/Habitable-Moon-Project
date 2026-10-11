"""Replay-bound original physical histories beside an implicit material epoch.

The original force solver is replayed before a binding can be constructed.
Every original quadrature measure, spring state and continuous material field
is retained. Queries are observations of the declared affine free-anchor and
radial candidate paths; they never create a new local state or allocate nodal
work to a material point, swept domain or archived volume.

Exact finite SO(3) tip maps are kept separate from the integrated first-order
mechanical anchors. A binding authenticates ORIGINAL physical pair history,
not a correspondence between incoming material and a different pair of banks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from fractions import Fraction
import hashlib
import json
import math

import numpy as np

from .geometric_bending_reference import current_bending_reference
from .geometric_native_contact_time_support import (
    MODEL as CONTACT_TIME_MODEL, NativeContactTimePath, _add, _affine_bounds,
    _bounds, _field, _scale, assess_native_contact_time_support,
    build_native_contact_time_path, native_contact_time_support_to_dict,
)
from .geometric_native_first_entry_material import (
    NativeFirstEntryMaterialState,
    native_first_entry_material_to_dict, validate_native_first_entry_material,
)
from .geometric_native_mantle_loading import (
    NativePhysicalSpatialMantleLoad, native_spatial_mantle_load_to_dict,
    validate_native_physical_spatial_mantle_load,
)
from .geometric_native_polygons import native_polygon_to_dict
from .geometric_native_pose import FractionInterval as I
from .geometric_native_surface import native_surface_to_dict
from .geometric_native_underthrust_motion import (
    NativeUnderthrustCandidate, NativeUnderthrustMotionState,
    UnresolvedNativeUnderthrustMotion, _advance_native,
    native_underthrust_motion_to_dict,
)
from .geometric_spatial_underthrust import _angles, _piece_at
from .thermal import SECONDS_PER_MYR


FORMAT = "native-original-physical-history-binding-1"
MODEL = "replayed_original_pairs_affine_candidate_observation_no_handover"
QUERY_FORMAT = "native-original-quadrature-history-query-1"
RADIAL_MODEL = "affine_previous_to_unaccepted_candidate_radial_displacement"
_AUTHENTICATED = {}
_WORK_NAMES = (
    "preparation_work_j", "cumulative_source_work_j",
    "cumulative_basal_dissipation_j", "cumulative_bending_numerical_dissipation_j",
)


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _literal_json(value):
    """DTO arrays and keys retain their explicit JSON types before replay."""
    kind = type(value)
    if kind in (str, int, bool, type(None)):
        return
    if kind is float:
        if not math.isfinite(value):
            raise ValueError("Physical history DTO contains a nonfinite number")
        return
    if kind is list:
        for item in value:
            _literal_json(item)
        return
    if kind is dict and all(type(key) is str for key in value):
        for item in value.values():
            _literal_json(item)
        return
    raise ValueError("Physical history DTO requires literal JSON lists, keys and scalar types")


def _typed_fingerprint(value):
    """Literal types, including the force engine's immutable NumPy scalars."""
    kind = type(value)
    if kind in (str, bytes, int, bool, type(None)):
        return kind, value
    if kind is float:
        return kind, value.hex()
    if kind is Fraction:
        return kind, value.numerator, value.denominator
    if isinstance(value, np.generic):
        return kind, value.dtype.str, value.tobytes()
    if kind is tuple:
        return kind, tuple(_typed_fingerprint(item) for item in value)
    if hasattr(kind, "__dataclass_fields__"):
        return kind, tuple((name, _typed_fingerprint(getattr(value, name)))
                           for name in value.__dataclass_fields__)
    from .geometric_native_surface import NativeSurfaceState

    if kind is NativeSurfaceState:
        return kind, tuple((name, _typed_fingerprint(getattr(value, name)))
                           for name in NativeSurfaceState.__slots__)
    raise ValueError("Physical replay requires completely immutable typed source fields")


def _sha(value):
    if (type(value) is not str or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError("Physical history requires a canonical original checkpoint SHA-256")
    return value


def _dyadic(value):
    if type(value) is not float or not math.isfinite(value):
        raise ValueError("Physical history must retain the literal finite binary64 source")
    return Fraction.from_float(value)


def _rational(raw):
    if (type(raw) is not list or len(raw) != 2
            or any(type(item) is not int for item in raw) or raw[1] <= 0):
        raise ValueError("Physical history bounds require exact canonical rational pairs")
    value = Fraction(*raw)
    if [value.numerator, value.denominator] != raw:
        raise ValueError("Physical history rational pairs must be reduced")
    return value


def _time(value):
    if type(value) is not I or not 0 <= value.lo <= value.hi <= 1:
        raise ValueError("Physical history time must be an exact interval in [0,1]")
    return value


def _precision(value):
    if type(value) is not int or not 32 <= value <= 1024:
        raise ValueError("Physical history precision must be an integer from 32 to 1024")
    return value


def _candidate_payload(candidate, overlaps):
    """The complete historical force-evidence schema, reproduced from the solve."""
    return dict(before_surface_sha256=candidate.before_surface_sha256,
        current_load_sha256=candidate.current_load_sha256, dt_s=candidate.dt_s,
        provenance=candidate.provenance,
        rotation_increment_radians=candidate.rotation_increment_radians,
        angular_velocity_rad_s=candidate.angular_velocity_rad_s,
        omega_rad_per_myr=candidate.omega_rad_per_myr,
        states=[(identity, asdict(local)) for identity, local in candidate.states],
        transverse_displacement_m=candidate.transverse_displacement_m,
        spatial_step=asdict(candidate.spatial_step),
        diagnostics=dict(candidate.candidate_diagnostics),
        admission=candidate.admission_audit, physical_points=candidate.points,
        unresolved=[dict(fragment_a=overlap.fragment_a, fragment_b=overlap.fragment_b,
            plate_a=overlap.plate_a, plate_b=overlap.plate_b, area_km2=overlap.area_km2,
            reason=overlap.reason, geometry=native_polygon_to_dict(overlap.geometry))
            for overlap in overlaps])


def _candidate_typed_payload(candidate):
    # Literal JSON bytes are retained separately, so key order or numeric
    # aliases in a purported typed candidate cannot pass a semantic comparison.
    return dict(**_candidate_payload(candidate, ()) ,
        admission_audit_json_hex=candidate.admission_audit_json.hex(),
        points_json_hex=candidate.points_json.hex())


def _whole_support(path, supplied, candidate_sha256):
    """Replay every saved rectangle rather than trusting its path checksum."""
    if type(supplied) is not dict or type(supplied.get("certificates")) is not list:
        raise ValueError("Physical material reference lacks complete contact-time support")
    rows = supplied["certificates"]
    if len(rows) != len(path.step.certificates):
        raise ValueError("Physical time support must retain every original trace")
    repeated = []
    for certificate, row in zip(path.step.certificates, rows):
        if type(row) is not dict or row.get("trace_id") != certificate.trace_id:
            raise ValueError("Physical time support changed its canonical original traces")
        expected_angle = [[0, 1], list(_dyadic(certificate.extent_radians).as_integer_ratio())]
        if (row.get("material_angle_bounds") != expected_angle
                or row.get("time_fraction_bounds") != [[0, 1], [1, 1]]):
            raise ValueError("Physical time support must cover the whole original trace and clock")
        support = assess_native_contact_time_support(path, certificate.trace_id,
            Fraction(0), _dyadic(certificate.extent_radians), Fraction(0), Fraction(1),
            precision=_precision(row.get("precision")))
        repeated.append(native_contact_time_support_to_dict(support))
    result = dict(path_sha256=path.path_sha256, path_model=CONTACT_TIME_MODEL,
        coordinate_model="original_retained_angle_scaled_from_exact_native_axis_quarter",
        certificates=repeated, candidate_sha256=candidate_sha256,
        geometry_permission=False, candidate_memory_adopted=False,
        scope="four_original_inviscid_pairs_whole_material_traces_whole_time_interval")
    if _encoded(result) != _encoded(supplied):
        raise ValueError("Physical time support differs from full original constitutive replay")
    return result


@dataclass(frozen=True, slots=True)
class NativeOriginalQuadratureHistory:
    point_id: str
    trace_id: str
    computational_angle: Fraction
    original_point: object
    original_material: object
    original_frame: tuple
    original_area_m2: float
    prepared_state: object
    previous_state: object
    unaccepted_candidate_state: object


@dataclass(frozen=True, slots=True)
class NativePhysicalHistoryBinding:
    origin_surface: object
    accepted_before: NativeUnderthrustMotionState
    actual_candidate: NativeUnderthrustCandidate
    current_load: NativePhysicalSpatialMantleLoad
    material_state: NativeFirstEntryMaterialState
    contact_time_path: NativeContactTimePath
    original_checkpoint_sha256: str
    original_reference: object
    quadrature_histories: tuple[NativeOriginalQuadratureHistory, ...]
    source_before_payload_json: bytes
    source_candidate_payload_json: bytes
    source_current_load_payload_json: bytes
    whole_time_support_payload_json: bytes
    before_work_ledgers: tuple
    unaccepted_candidate_work_ledgers: tuple
    literal_dt_s: float
    literal_dt_myr: float
    binding_sha256: str

    @property
    def initial_mechanics(self):
        return self.accepted_before.initial_mechanics

    @property
    def previous_memory(self):
        return self.contact_time_path.step.previous_memory

    @property
    def omega_records(self):
        return tuple((owner, tuple(_dyadic(value) for value in vector))
                     for owner, vector in self.actual_candidate.omega_rad_per_myr)

    @property
    def dt_myr(self):
        return _dyadic(self.literal_dt_myr)

    @property
    def original_points(self):
        return tuple(node.original_point for node in self.quadrature_histories)

    @property
    def unaccepted_candidate_memory(self):
        return self.contact_time_path.step.memory

    @property
    def previous_free_anchor_fields(self):
        return self.contact_time_path.previous_anchor_fields

    @property
    def unaccepted_candidate_free_anchor_fields(self):
        return self.contact_time_path.step.anchor_fields

    @property
    def original_pair_history_authenticated(self):
        return True

    @property
    def incoming_pair_handover_authorized(self):
        return False

    @property
    def candidate_mechanics_adopted(self):
        return False

    @property
    def mechanical_work_assigned_to_material_volume(self):
        return False


def _binding_body(binding):
    return dict(format=FORMAT, model=MODEL,
        original_checkpoint_sha256=binding.original_checkpoint_sha256,
        origin_surface=asdict(binding.origin_surface),
        accepted_before=json.loads(binding.source_before_payload_json),
        actual_candidate=json.loads(binding.source_candidate_payload_json),
        actual_candidate_literal_fields=_candidate_typed_payload(binding.actual_candidate),
        current_load=json.loads(binding.source_current_load_payload_json),
        material_state=native_first_entry_material_to_dict(binding.material_state),
        contact_time_path_sha256=binding.contact_time_path.path_sha256,
        contact_time_path=asdict(binding.contact_time_path),
        whole_time_support=json.loads(binding.whole_time_support_payload_json),
        quadrature_histories=_portable(binding.quadrature_histories),
        original_reference=asdict(binding.original_reference),
        before_work_ledgers=binding.before_work_ledgers,
        unaccepted_candidate_work_ledgers=binding.unaccepted_candidate_work_ledgers,
        literal_dt_s=binding.literal_dt_s, literal_dt_myr=binding.literal_dt_myr,
        radial_path_model=RADIAL_MODEL,
        original_pair_history_authenticated=True,
        incoming_pair_handover_authorized=False, candidate_mechanics_adopted=False,
        mechanical_work_assigned_to_material_volume=False,
        source_work_scope="original force quadrature endpoint ledger only; never a volume density")


def _remember(binding):
    # Cache authority comes from a genuine replay and fingerprints every typed
    # immutable field. A caller-supplied SHA or rehashed DTO is never a key.
    if len(_AUTHENTICATED) >= 64:
        _AUTHENTICATED.clear()
    _AUTHENTICATED[_typed_fingerprint(binding)] = binding.binding_sha256


def build_native_physical_history_binding(mesh, origin, accepted_before,
        actual_candidate, material_state, *, current_load,
        original_checkpoint_sha256, contact_time_path=None):
    """Authenticate the real original force and retain its complete histories.

    ``current_load`` is required because a candidate only stores its checksum.
    The genuine force solve is repeated with that independent physical source.
    Its existing refusal remains in force; construction is not a force step.
    """
    if (type(accepted_before) is not NativeUnderthrustMotionState
            or type(actual_candidate) is not NativeUnderthrustCandidate
            or type(current_load) is not NativePhysicalSpatialMantleLoad
            or type(material_state) is not NativeFirstEntryMaterialState):
        raise ValueError("Physical history requires explicit typed source, candidate and material state")
    _sha(original_checkpoint_sha256)
    before_payload = native_underthrust_motion_to_dict(mesh, origin, accepted_before)
    validate_native_physical_spatial_mantle_load(accepted_before.current_surface,
                                                current_load, mesh=mesh)
    try:
        _advance_native(mesh, origin, accepted_before, current_load,
                        actual_candidate.dt_s, actual_candidate.provenance)
    except UnresolvedNativeUnderthrustMotion as refusal:
        replayed, overlaps = refusal.candidate, refusal.overlaps
    else:
        raise ValueError("Physical history requires the actual unaccepted native force candidate")
    if _typed_fingerprint(actual_candidate) != _typed_fingerprint(replayed):
        raise ValueError("Physical history candidate differs from the complete genuine force replay")
    # The current implicit material epoch is the original preparation. Future
    # accepted epochs require their complete prior spatial replay, not reset.
    if accepted_before.receipts:
        raise ValueError("This original implicit epoch cannot skip accepted physical history")
    path = build_native_contact_time_path(accepted_before.initial_mechanics,
        replayed.spatial_step, accepted_before.free_anchor_fields)
    if contact_time_path is not None:
        if (type(contact_time_path) is not NativeContactTimePath
                or _typed_fingerprint(contact_time_path) != _typed_fingerprint(path)):
            raise ValueError("Physical history contact-time path differs from complete constitutive replay")
        path = contact_time_path
    validate_native_first_entry_material(material_state)
    journal = material_state.journal
    if (_encoded(native_surface_to_dict(journal.source_surface))
            != _encoded(native_surface_to_dict(accepted_before.current_surface))
            or _typed_fingerprint(journal.omega_rad_per_myr)
                != _typed_fingerprint(replayed.omega_rad_per_myr)
            or _dyadic(journal.dt_myr) != _dyadic(replayed.dt_s / SECONDS_PER_MYR)):
        raise ValueError("Physical material journal changed the actual source surface, full rotations or clock")
    seed = accepted_before.initial_mechanics
    # The material component froze the existing mechanical DTO, including its
    # format and source signatures, rather than a bare dataclass projection.
    seed_payload = before_payload["initial_mechanics"]
    if material_state.original_mechanics_payload_json != _encoded(seed_payload):
        raise ValueError("Physical material state changed its complete prepared spring configuration")
    reference = journal.memory_reference
    before_memory = json.loads(reference.before_payload_json)
    after_memory = json.loads(reference.candidate_after_payload_json)
    snapshot = before_payload["snapshot"]
    expected_before = {key: snapshot[key] for key in (
        "states", "spatial_memory", "free_anchor_fields", "elapsed_time_s",
        "cumulative_source_work_j", "cumulative_basal_dissipation_j",
        "cumulative_bending_numerical_dissipation_j")}
    expected_before.update(original_mechanics_payload=seed_payload,
        original_surface_payload=native_surface_to_dict(accepted_before.current_surface),
        original_force_checkpoint_sha256=original_checkpoint_sha256)
    if _encoded(expected_before) != _encoded(before_memory):
        raise ValueError("Physical material before reference differs from actual source memory or work")
    candidate_payload = _candidate_payload(replayed, overlaps)
    candidate_sha = _digest(candidate_payload)
    support = _whole_support(path, after_memory.get("whole_time_support"), candidate_sha)
    expected_after = dict(states=candidate_payload["states"],
        spatial_memory=candidate_payload["spatial_step"]["memory"],
        free_anchor_fields=candidate_payload["spatial_step"]["anchor_fields"],
        dt_s=replayed.dt_s, candidate_diagnostics=candidate_payload["diagnostics"],
        candidate_sha256=candidate_sha, whole_time_support=support, candidate_adopted=False)
    if (_encoded(expected_after) != _encoded(after_memory)
            or reference.constitutive_path_sha256 != path.path_sha256):
        raise ValueError("Physical material candidate reference differs from genuine fields, work or time path")
    original = current_bending_reference(origin, seed)
    old_states, candidate_states, prepared_states = (
        dict(accepted_before.states), dict(replayed.states), dict(seed.states))
    angles = _angles(seed)
    nodes = tuple(NativeOriginalQuadratureHistory(point.point_id, point.trace_id,
        _dyadic(theta), point, material, tuple(frame), float(area),
        prepared_states[point.point_id], old_states[point.point_id],
        candidate_states[point.point_id])
        for point, material, frame, area, (identity, trace, theta) in zip(
            original.points, original.materials, original.frames, original.areas, angles))
    if (len(nodes) != len(accepted_before.states)
            or tuple(node.point_id for node in nodes) != tuple(identity for identity, _ in accepted_before.states)
            or any(node.original_area_m2 != node.previous_state.area_m2
                   or node.original_area_m2 != node.unaccepted_candidate_state.area_m2
                   for node in nodes)):
        raise ValueError("Physical history cannot change original quadrature identities or measures")
    work_before = tuple((name, getattr(accepted_before, name)) for name in _WORK_NAMES)
    work_candidate = tuple((name, value) for name, value in replayed.candidate_diagnostics
        if name.endswith("_j") or "energy_balance" in name)
    binding = NativePhysicalHistoryBinding(origin, accepted_before, replayed, current_load,
        material_state, path, original_checkpoint_sha256, original, nodes,
        _encoded(before_payload), _encoded(candidate_payload),
        _encoded(native_spatial_mantle_load_to_dict(accepted_before.current_surface, current_load)),
        _encoded(support), work_before, work_candidate, replayed.dt_s, journal.dt_myr, "")
    binding = replace(binding, binding_sha256=_digest(_binding_body(binding)))
    _remember(binding)
    return binding


def validate_native_physical_history_binding(binding, *, mesh=None):
    if type(binding) is not NativePhysicalHistoryBinding:
        raise ValueError("Expected an immutable replay-authenticated physical history binding")
    fingerprint = _typed_fingerprint(binding)
    if _AUTHENTICATED.get(fingerprint) == binding.binding_sha256:
        return binding
    if mesh is None:
        raise ValueError("Physical history was not authenticated by genuine force replay; supply the source mesh")
    repeated = build_native_physical_history_binding(mesh, binding.origin_surface,
        binding.accepted_before, binding.actual_candidate, binding.material_state,
        current_load=binding.current_load,
        original_checkpoint_sha256=binding.original_checkpoint_sha256,
        contact_time_path=binding.contact_time_path)
    if _typed_fingerprint(repeated) != fingerprint:
        raise ValueError("Physical history binding differs from complete source replay")
    return binding


@dataclass(frozen=True, slots=True)
class NativeOriginalPairHistoryQuery:
    binding_sha256: str
    point_id: str
    trace_id: str
    computational_angle: Fraction
    time_fraction_bounds: I
    elapsed_time_s_bounds: I
    native_elapsed_time_s_bounds: I
    physical_time_myr_bounds: I
    free_gap_bounds_m: I
    free_jump_bounds_m: I
    previous_plastic_slip_bounds_m: I
    plastic_increment_bounds_m: I
    plastic_slip_bounds_m: I
    cumulative_slip_bounds_m: I
    normal_pressure_bounds_pa: I
    shear_traction_bounds_pa: I
    contact_gap_bounds_m: I
    contact_jump_bounds_m: I
    constitutive_status: str
    tip_maps: object
    original_quadrature_history: NativeOriginalQuadratureHistory
    precision: int

    @property
    def original_pair_history_authenticated(self):
        return True

    @property
    def incoming_pair_handover_authorized(self):
        return False

    @property
    def candidate_mechanics_adopted(self):
        return False

    @property
    def nodal_work_allocated_to_material_volume(self):
        return False


def _positive(value):
    return I(max(Fraction(0), value.lo), max(Fraction(0), value.hi))


def query_native_original_pair_history(binding, point_id, time_fraction_bounds, *, precision=128):
    """Observe one original quadrature history at exact, possibly singleton time.

    The continuous plastic value comes from the original polynomial field and
    the declared return-map family, never interpolation of nodal values. Nodal
    work is returned only as the unchanged old/candidate endpoint records of
    this same physical quadrature measure. No intermediate work is fabricated.
    """
    validate_native_physical_history_binding(binding)
    time, precision = _time(time_fraction_bounds), _precision(precision)
    if type(point_id) is not str:
        raise ValueError("Physical history queries require an original quadrature point ID")
    node = next((item for item in binding.quadrature_histories if item.point_id == point_id), None)
    if node is None:
        raise ValueError("Physical history cannot assign an original nodal history to a new point")
    path, trace, theta = binding.contact_time_path, node.original_material, node.computational_angle
    index = next(i for i, row in enumerate(path.previous_anchor_fields) if row[0] == trace.trace_id)
    _, old_gap, old_jump = path.previous_anchor_fields[index]
    _, new_gap, new_jump = path.step.anchor_fields[index]
    g0, j0, g1, j1 = map(_field, (old_gap, old_jump, new_gap, new_jump))
    dg, dj = _add(g1, _scale(g0, -1)), _add(j1, _scale(j0, -1))
    gap = _affine_bounds(g0, dg, theta, theta, time.lo, time.hi, precision)
    jump = _affine_bounds(j0, dj, theta, theta, time.lo, time.hi, precision)
    stored = path.step.previous_memory.traces[index]
    piece = _piece_at(stored.pieces, float(theta))
    old_p = _bounds(_field(piece.plastic_slip), theta, theta, precision)
    old_c = _bounds(_field(piece.cumulative_slip), theta, theta, precision)
    kn = _dyadic(1. / (1. / trace.columns.normal_stiffness_pa_m + 1. / trace.law.normal_stiffness_pa_m))
    kt = _dyadic(1. / (1. / trace.columns.shear_stiffness_pa_m + 1. / trace.law.tangential_stiffness_pa_m))
    mu = _dyadic(trace.law.friction_dry + (trace.law.friction_wet - trace.law.friction_dry) * trace.water)
    cohesion = _dyadic(trace.law.cohesion_pa * (1. - (1. - trace.law.wet_cohesion_fraction) * trace.water))
    eta, duration = _dyadic(trace.law.viscosity_pa_s_m), _dyadic(binding.literal_dt_s)
    pressure = -kn * gap
    strength = cohesion + mu * pressure
    trial = kt * (jump - old_p)
    numerator = _positive(trial - strength) - _positive(-trial - strength)
    if eta:
        increment = numerator * (time * duration) / (kt * time * duration + eta)
        status = "viscous_fractional_backward_euler_diagnostic_only"
    else:
        increment = numerator / kt
        status = "inviscid_original_pair_dense_return_map_diagnostic"
    plastic = old_p + increment
    absolute = I(0, max(-increment.lo, increment.hi)) if increment.lo <= 0 <= increment.hi else (
        -increment if increment.hi < 0 else increment)
    cumulative = old_c + absolute
    shear = kt * (jump - plastic)
    contact_gap = -pressure / _dyadic(trace.law.normal_stiffness_pa_m)
    contact_jump = plastic + shear / _dyadic(trace.law.tangential_stiffness_pa_m)
    from .geometric_native_physical_pair_time import certify_native_original_pair_tip_maps

    radial_before = next(row[1:] for row in binding.accepted_before.transverse_displacement_m
                         if row[0] == trace.trace_id)
    radial_candidate = next(row[1:] for row in binding.actual_candidate.transverse_displacement_m
                            if row[0] == trace.trace_id)
    tips = certify_native_original_pair_tip_maps(trace.trace_id,
        tuple(_dyadic(v) for v in node.original_point.position),
        tuple(_dyadic(v) for v in node.original_point.normal),
        tuple(_dyadic(v) for v in node.original_point.down_dip),
        trace.subducting_plate, trace.overriding_plate,
        tuple((owner, tuple(_dyadic(v) for v in values))
              for owner, values in binding.actual_candidate.omega_rad_per_myr),
        _dyadic(binding.literal_dt_myr), tuple(_dyadic(v) for v in radial_before),
        tuple(_dyadic(v) for v in radial_candidate), time,
        radius_m=_dyadic(float(binding.origin_surface.radius_km * 1000.)), precision=precision)
    return NativeOriginalPairHistoryQuery(binding.binding_sha256, node.point_id,
        node.trace_id, theta, time,
        I(_dyadic(node.previous_state.elapsed_time_s)) + duration * time,
        I(_dyadic(binding.accepted_before.elapsed_time_s)) + duration * time,
        I(_dyadic(float(binding.origin_surface.time_myr))) + _dyadic(binding.literal_dt_myr) * time,
        gap, jump, old_p, increment, plastic, cumulative, pressure, shear,
        contact_gap, contact_jump, status, tips, node, precision)


def _portable(value):
    if type(value) is Fraction:
        return [value.numerator, value.denominator]
    if type(value) is I:
        return dict(lo=_portable(value.lo), hi=_portable(value.hi))
    if hasattr(type(value), "__dataclass_fields__"):
        return {name: _portable(getattr(value, name)) for name in value.__dataclass_fields__}
    if type(value) in (tuple, list):
        return [_portable(item) for item in value]
    if type(value) is dict:
        return {key: _portable(item) for key, item in value.items()}
    return value


def native_physical_history_binding_to_dict(binding, *, mesh=None):
    validate_native_physical_history_binding(binding, mesh=mesh)
    return dict(json.loads(_encoded(_binding_body(binding))), binding_sha256=binding.binding_sha256)


def native_physical_history_binding_from_dict(raw, mesh, origin, accepted_before,
        actual_candidate, material_state, *, current_load, original_checkpoint_sha256,
        contact_time_path=None):
    """Rebuild with actual external sources before comparing every DTO field."""
    _literal_json(raw)
    repeated = build_native_physical_history_binding(mesh, origin, accepted_before,
        actual_candidate, material_state, current_load=current_load,
        original_checkpoint_sha256=original_checkpoint_sha256, contact_time_path=contact_time_path)
    if type(raw) is not dict or _encoded(raw) != _encoded(native_physical_history_binding_to_dict(repeated)):
        raise ValueError("Physical history DTO differs from complete genuine source and candidate replay")
    return repeated


def native_original_pair_history_to_dict(query, *, binding):
    if type(query) is not NativeOriginalPairHistoryQuery:
        raise ValueError("Expected an immutable original quadrature history query")
    repeated = query_native_original_pair_history(binding, query.point_id,
        query.time_fraction_bounds, precision=query.precision)
    if _typed_fingerprint(query) != _typed_fingerprint(repeated):
        raise ValueError("Original quadrature query differs from exact-clock field and tip replay")
    result = dict(format=QUERY_FORMAT, **_portable(query),
        original_pair_history_authenticated=True, incoming_pair_handover_authorized=False,
        candidate_mechanics_adopted=False, nodal_work_allocated_to_material_volume=False,
        work_scope="unchanged original nodal before/candidate endpoint records only",
        intermediate_work_computed=False,
        clock_scope="elapsed_time_s includes original local preparation; native_elapsed_time_s is the global native clock; identical exact fraction with retained dt_s and dt_myr",
        field_scope="exact dyadic-coefficient return-map diagnostic; saved binary64 nodal endpoints retained separately under original roundoff-aware replay",
        constitutive_path_model=CONTACT_TIME_MODEL, radial_path_model=RADIAL_MODEL)
    from .geometric_native_physical_pair_time import native_original_pair_tip_maps_to_dict

    result["tip_maps"] = native_original_pair_tip_maps_to_dict(query.tip_maps)
    result["query_sha256"] = _digest(result)
    return json.loads(_encoded(result))


def native_original_pair_history_from_dict(raw, *, binding):
    _literal_json(raw)
    if type(raw) is not dict or type(raw.get("time_fraction_bounds")) is not dict:
        raise ValueError("Original physical query requires an explicit exact-clock DTO")
    window = raw["time_fraction_bounds"]
    if set(window) != {"lo", "hi"}:
        raise ValueError("Original physical query changed its exact time bounds")
    query = query_native_original_pair_history(binding, raw.get("point_id"),
        I(_rational(window["lo"]), _rational(window["hi"])), precision=raw.get("precision"))
    if _encoded(raw) != _encoded(native_original_pair_history_to_dict(query, binding=binding)):
        raise ValueError("Original physical query DTO differs from exact field, tips and nodal work replay")
    return query
