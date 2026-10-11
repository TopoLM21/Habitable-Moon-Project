"""Objective local contact response on authentic common-reference body ports.

The receiver carries the NEW common-cut material directors.  This preserves
the observed source scalar preload, not the old world traction direction.
Results are per reference interface area: no face measure, spatial memory
distribution, current-normal law, equilibrium or admission is assigned.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from fractions import Fraction as F
import hashlib
import json

from .geometric_native_pose import FractionInterval as I, _rounded_interval
from .geometric_native_common_reference import validate_native_common_reference
from .geometric_native_common_contact_ports import validate_native_common_contact_port_response


FORMAT = "native-common-contact-objective-local-constitutive-1"
SCOPE = "SOURCE_CALIBRATED_CONTACT_ONLY_PER_REFERENCE_AREA"
FRAME = "common_cut_material"
BRANCHES = ("relaxed_source", "frozen_memory_predictor")
_REGISTRY = {}
_DEPENDENCIES = {}


def _payload(value):
    if type(value) is F:
        return [value.numerator, value.denominator]
    if type(value) is I:
        return dict(lo=_payload(value.lo), hi=_payload(value.hi))
    if is_dataclass(value):
        return {item.name: _payload(getattr(value, item.name)) for item in fields(value)}
    if type(value) in (tuple, list):
        return [_payload(x) for x in value]
    if type(value) is dict:
        return {key: _payload(x) for key, x in value.items()}
    if type(value) in (str, int, bool) or value is None:
        return value
    raise TypeError("Local contact receipts require immutable exact source scalars")


def _encoded(value):
    return json.dumps(_payload(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _seal(value):
    return hashlib.sha256(_encoded({item.name: getattr(value, item.name)
        for item in fields(value) if item.name != "binding_sha256"})).hexdigest()


def _remember(value, dependencies):
    result = replace(value, binding_sha256=_seal(value))
    _REGISTRY[type(result), result.binding_sha256] = _encoded(result)
    _DEPENDENCIES[type(result), result.binding_sha256] = dependencies
    return result


def _registered(value, kind):
    if (type(value) is not kind or value.binding_sha256 != _seal(value)
            or _REGISTRY.get((kind, value.binding_sha256)) != _encoded(value)):
        raise ValueError("An altered or re-sealed contact response lacks authentic replay")
    return _DEPENDENCIES[kind, value.binding_sha256]


def _round(value, precision):
    return _rounded_interval(value, precision)


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), I(0))


def _cross(a, b):
    return a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]


def _square(x):
    return I(0 if x.lo <= 0 <= x.hi else min(x.lo*x.lo, x.hi*x.hi),
             max(x.lo*x.lo, x.hi*x.hi))


def _absolute(x):
    return I(0 if x.lo <= 0 <= x.hi else min(abs(x.lo), abs(x.hi)),
             max(abs(x.lo), abs(x.hi)))


def _axis(label):
    return tuple(I(int(i == "xyz".index(label))) for i in range(3))


def _symmetric_owner_second(a, b, vector):
    ab, ba = _cross(a, _cross(b, vector)), _cross(b, _cross(a, vector))
    return tuple((x+y)/2 for x, y in zip(ab, ba))


class _LocalScope:
    @property
    def scope(self):
        return SCOPE

    @property
    def material_memory_or_interface_measure_transferred(self):
        return False

    @property
    def old_sixteen_atoms_or_rod_energy_assigned(self):
        return False

    @property
    def current_deformed_follower_normal_law_certified(self):
        return False

    @property
    def force_equilibrium_certified(self):
        return False

    @property
    def physical_material_admission_authorized(self):
        return False

    @property
    def finite_owner_motion_or_whole_clock_certified(self):
        return False

    @property
    def source_state_or_thermal_state_advanced(self):
        return False


@dataclass(frozen=True, slots=True)
class NativeCommonContactScalarKinematics(_LocalScope):
    source_snapshot_sha256: str
    reference_snapshot_sha256: str
    cut_snapshot_sha256: str
    geometry_bounds_sha256: str
    port_response_binding_sha256: str
    receiver_frame: str
    scalar_component_order: tuple
    coordinate_order: tuple
    body_coordinate_count: int
    donor_owner: int
    receiver_owner: int
    reference_director_bounds: tuple
    scalar_increment_bounds_m: tuple
    jump_first_derivative_bounds: tuple
    jump_second_derivative_bounds: tuple
    frame_first_derivative_bounds: tuple
    frame_second_derivative_bounds: tuple
    scalar_first_derivative_bounds: tuple
    scalar_second_derivative_bounds: tuple
    precision: int
    binding_sha256: str
    owner_chart: str = "FRESH_LEFT_EXPONENTIAL_CHART_AT_ZERO"
    symmetric_coordinate_owner_second_derivatives: bool = True
    full_receiver_frame_derivatives_included: bool = True
    shared_owner_scalar_partials_cancel_symbolically: bool = True
    old_world_traction_direction_preserved: bool = False


def _scalar_derivatives(jump, columns, jump_first, jump_second,
                        directions, receiver_owner, owner_order, body_count,
                        shared_owner, precision):
    """Pure algebra: complete scalar chain rule at the fresh left chart.

    The shared-owner identity e=R N, w=R(u_r-u_d) is evaluated before
    interval arithmetic.  It cancels all scalar owner partials exactly.
    """
    count = len(jump_first)
    zero = (I(0),)*3
    frame_first, frame_second, first, second = [], [], [], []
    for direction in directions:
        ei = [zero]*body_count + [_cross(_axis(axis), direction)
            if owner == receiver_owner else zero for owner, axis in owner_order]
        eij = [[zero for _ in range(count)] for _ in range(count)]
        for a, (owner_a, axis_a) in enumerate(owner_order, body_count):
            for b, (owner_b, axis_b) in enumerate(owner_order, body_count):
                if owner_a == owner_b == receiver_owner:
                    eij[a][b] = _symmetric_owner_second(_axis(axis_a), _axis(axis_b), direction)
        fi = tuple(I(0) if shared_owner and i >= body_count else
            _round(_dot(direction, jump_first[i])+_dot(ei[i], jump), precision)
            for i in range(count))
        hij = [[I(0) for _ in range(count)] for _ in range(count)]
        for i in range(count):
            for j in range(i, count):
                value = I(0) if shared_owner and (i >= body_count or j >= body_count) else _round(
                    _dot(direction, jump_second[i][j])+_dot(ei[i], jump_first[j])
                    +_dot(ei[j], jump_first[i])+_dot(eij[i][j], jump), precision)
                hij[i][j] = hij[j][i] = value
        frame_first.append(tuple(tuple(_round(x, precision) for x in v) for v in ei))
        frame_second.append(tuple(tuple(tuple(_round(x, precision) for x in v) for v in row) for row in eij))
        first.append(fi)
        second.append(tuple(tuple(row) for row in hij))
    increments = tuple(_round(_dot(direction, jump), precision) for direction in directions)
    return increments, tuple(frame_first), tuple(frame_second), tuple(first), tuple(second)


def evaluate_native_common_contact_scalar_kinematics(reference, port_response, *,
        receiver_frame=FRAME, precision=160):
    reference = validate_native_common_reference(reference)
    response = validate_native_common_contact_port_response(port_response, reference=reference)
    if receiver_frame != FRAME:
        raise ValueError("Declare the NEW common-cut material frame; source/current-normal frames are unsupported")
    if type(precision) is not int or precision < 32:
        raise ValueError("Local contact derivatives require an explicit adequate enclosure precision")
    donor_owner = response.port.donor_body.original_body.owner
    receiver_owner = response.port.receiver_body.original_body.owner
    owner_order = response.port.owner_rotation_coordinate_order
    if len(owner_order) != 6:
        raise ValueError("The authenticated common source must supply six global owner coordinates")
    body_count = len(response.mode_jump_columns)
    count = body_count+len(owner_order)
    zero = (I(0),)*3
    # These existing columns already use ACTUAL displaced sides and exact
    # shared-X cancellation; do not replace them with reference X alone.
    first = response.mode_jump_columns+response.owner_rotation_jump_columns_m
    second = [[zero for _ in range(count)] for _ in range(count)]
    for a, row in enumerate(response.owner_body_mode_cross_derivative_columns, body_count):
        for b, vector in enumerate(row):
            second[a][b] = second[b][a] = vector
    X = response.geometry.world_position_bounds_m
    ud = response.donor_fields.displacement_increment_bounds_m
    ur = response.receiver_fields.displacement_increment_bounds_m
    for a, (owner_a, axis_a) in enumerate(owner_order, body_count):
        for b, (owner_b, axis_b) in enumerate(owner_order, body_count):
            if owner_a != owner_b:
                continue
            d, r = int(owner_a == donor_owner), int(owner_a == receiver_owner)
            vector = tuple((r-d)*x+r*y-d*z for x, y, z in zip(X, ur, ud))
            second[a][b] = tuple(_round(x, precision) for x in
                _symmetric_owner_second(_axis(axis_a), _axis(axis_b), vector))
    directions = (response.geometry.receiver_inward_unit_normal_bounds,
                  response.geometry.inward_unit_down_direction_bounds,
                  response.geometry.angular_unit_tangent_bounds)
    result = _scalar_derivatives(response.displacement_jump_bounds_m,
        response.mode_jump_columns, first, second, directions, receiver_owner,
        owner_order, body_count, donor_owner == receiver_owner, precision)
    return _remember(NativeCommonContactScalarKinematics(
        response.port.source_snapshot_sha256, reference.reference_snapshot_sha256,
        response.port.cut.cut_snapshot_sha256, response.geometry.bounds_snapshot_sha256,
        response.binding_sha256, receiver_frame, ("normal", "down", "angular"),
        tuple(("body", side, name) for side, name in response.mode_coordinate_order)
            +tuple(("owner", owner, axis) for owner, axis in owner_order),
        body_count, donor_owner, receiver_owner, directions, result[0],
        tuple(first), tuple(tuple(row) for row in second), *result[1:], precision, ""),
        (reference, response))


def validate_native_common_contact_scalar_kinematics(value, *, reference=None, port_response=None):
    original_reference, original_port = _registered(value, NativeCommonContactScalarKinematics)
    reference = original_reference if reference is None else validate_native_common_reference(reference)
    port = original_port if port_response is None else port_response
    port = validate_native_common_contact_port_response(port, reference=reference)
    if (value.reference_snapshot_sha256 != reference.reference_snapshot_sha256
            or value.port_response_binding_sha256 != port.binding_sha256):
        raise ValueError("Contact derivative receipt belongs to another reference or authentic port")
    return value


@dataclass(frozen=True, slots=True)
class NativeCommonContactConstitutiveResponse(_LocalScope):
    kinematics: NativeCommonContactScalarKinematics
    source_observation_binding_sha256: str
    receiving_trace_id: str
    branch: str
    normal_stiffness_pa_m: F
    tangential_stiffness_pa_m: F
    contact_gap_bounds_m: I
    contact_jump_bounds_m: I
    trial_elastic_shear_extension_bounds_m: I
    elastic_shear_extension_bounds_m: I
    traction_bounds_pa: tuple
    local_tangent_bounds_pa_m: tuple
    reference_recoverable_energy_bounds_j_m2: I
    recoverable_energy_bounds_j_m2: I
    residual_bounds_per_reference_m2: tuple
    hessian_bounds_per_reference_m2: tuple
    proposed_plastic_slip_bounds_m: I
    proposed_plastic_increment_bounds_m: I
    proposed_cumulative_slip_bounds_m: I
    proposed_plastic_first_derivative_bounds: tuple
    proposed_plastic_second_derivative_bounds: tuple
    precision: int
    binding_sha256: str
    source_memory_proposal_applied: bool = False
    source_preload_geometric_hessian_included: bool = True
    source_relaxed_friction_free_branch: bool = True
    frozen_predictor_is_physical_relaxed_shear_law: bool = False
    angular_contact_stiffness_assigned: bool = False
    hot_receiving_law_inferred: bool = False


def _state_api():
    # The source observer authenticates the frozen u=1 state independently.
    from . import geometric_native_common_contact_state as api
    return api


def evaluate_native_common_contact_constitutive(reference, port_response, source_observation, *,
        branch="relaxed_source", receiver_frame=FRAME, precision=160):
    reference = validate_native_common_reference(reference)
    port = validate_native_common_contact_port_response(port_response, reference=reference)
    if port.receiver_fields.material_kind != "original_live" or port.port.cut.receiving_trace_id is None:
        raise ValueError("The true HOT receiver has no inherited receiving contact law")
    state = _state_api().validate_native_common_contact_state_observation(source_observation,
        reference=reference, port_response=port)
    if (branch not in BRANCHES or not state.source_relaxed_friction_free_branch
            or not state.compressed_source_observation):
        raise ValueError("Only the authentic friction-free compressed source branch is supported")
    kn, kt = state.normal_stiffness_pa_m, state.tangential_stiffness_pa_m
    if type(kn) is not F or type(kt) is not F or min(kn, kt) <= 0:
        raise ValueError("Contact stiffness must be the positive literal source contact coefficients")
    if state.shear_traction_bounds_pa != I(0):
        raise ValueError("A nonzero shear preload is incompatible with this friction-free current source state")
    kinematics = evaluate_native_common_contact_scalar_kinematics(reference, port,
        receiver_frame=receiver_frame, precision=precision)
    xi_n, xi_t, _ = kinematics.scalar_increment_bounds_m
    # Preserve pressure and zero-shear correlation symbolically.  In
    # particular, never subtract independent interval copies of p*.
    # A zero modal increment is an exact source identity, even when the
    # observer used finer precision than this response.  Preserve that
    # original pressure enclosure instead of expanding it onto a new grid.
    normal = -state.normal_pressure_bounds_pa if xi_n == I(0) else _round(
        -state.normal_pressure_bounds_pa+kn*xi_n, precision)
    trial_shear = _round(state.shear_traction_bounds_pa+kt*xi_t, precision)
    gap = state.contact_reference_gap_bounds_m if xi_n == I(0) else _round(normal/kn, precision)
    if gap.hi >= 0:
        raise ValueError("Split or reject a contact query reaching the opening branch")
    trial_elastic = _round(trial_shear/kt, precision)
    jump = _round(state.contact_reference_jump_bounds_m+xi_t, precision)
    relaxed = branch == "relaxed_source"
    shear = I(0) if relaxed else trial_shear
    elastic = I(0) if relaxed else trial_elastic
    stiffness = (I(kn), I(0) if relaxed else I(kt))
    tractions = normal, shear
    source_energy = _round(_square(state.normal_pressure_bounds_pa)/(2*kn)
        +_square(state.shear_traction_bounds_pa)/(2*kt), precision)
    energy = _round(_square(normal)/(2*kn)+(I(0) if relaxed else _square(shear)/(2*kt)), precision)
    first, second = kinematics.scalar_first_derivative_bounds, kinematics.scalar_second_derivative_bounds
    count = len(kinematics.coordinate_order)
    residual = tuple(_round(sum((tractions[a]*first[a][i] for a in range(2)), I(0)), precision)
                     for i in range(count))
    hessian = [[I(0) for _ in range(count)] for _ in range(count)]
    for i in range(count):
        for j in range(i, count):
            value = _round(sum((stiffness[a]*first[a][i]*first[a][j]
                +tractions[a]*second[a][i][j] for a in range(2)), I(0)), precision)
            hessian[i][j] = hessian[j][i] = value
    increment = trial_elastic if relaxed else I(0)
    plastic = _round(state.plastic_slip_bounds_m+increment, precision)
    cumulative = _round(state.cumulative_slip_bounds_m+_absolute(increment), precision)
    plastic_first = first[1] if relaxed else (I(0),)*count
    plastic_second = second[1] if relaxed else ((I(0),)*count,)*count
    return _remember(NativeCommonContactConstitutiveResponse(kinematics,
        state.binding_sha256, state.receiving_trace_id, branch, kn, kt, gap, jump, trial_elastic, elastic,
        tractions, ((stiffness[0], I(0)), (I(0), stiffness[1])), source_energy, energy,
        residual, tuple(tuple(row) for row in hessian), plastic, increment, cumulative,
        plastic_first, plastic_second, precision, ""), (reference, port, state))


def validate_native_common_contact_constitutive_response(value, *, reference=None,
        port_response=None, source_observation=None):
    original_reference, original_port, original_state = _registered(value, NativeCommonContactConstitutiveResponse)
    reference = original_reference if reference is None else validate_native_common_reference(reference)
    port = original_port if port_response is None else port_response
    state = original_state if source_observation is None else source_observation
    validate_native_common_contact_scalar_kinematics(value.kinematics, reference=reference, port_response=port)
    state = _state_api().validate_native_common_contact_state_observation(state,
        reference=reference, port_response=port)
    if value.source_observation_binding_sha256 != state.binding_sha256:
        raise ValueError("The contact law receipt belongs to another authentic current source observation")
    return value


def _export(value):
    return dict(format=FORMAT, scope=SCOPE, **_payload(value),
        material_memory_or_interface_measure_transferred=False,
        old_sixteen_atoms_or_rod_energy_assigned=False,
        current_deformed_follower_normal_law_certified=False,
        force_equilibrium_certified=False, physical_material_admission_authorized=False,
        finite_owner_motion_or_whole_clock_certified=False,
        source_state_or_thermal_state_advanced=False)


def native_common_contact_scalar_kinematics_to_dict(value):
    return _export(validate_native_common_contact_scalar_kinematics(value))


def native_common_contact_scalar_kinematics_from_dict(raw, *, reference, port_response):
    if type(raw) is not dict:
        raise ValueError("A contact derivative export must be a mapping")
    value = evaluate_native_common_contact_scalar_kinematics(reference, port_response,
        receiver_frame=raw["receiver_frame"], precision=raw["precision"])
    if native_common_contact_scalar_kinematics_to_dict(value) != raw:
        raise ValueError("Contact derivative export differs from complete authentic replay")
    return value


def native_common_contact_constitutive_response_to_dict(value):
    return _export(validate_native_common_contact_constitutive_response(value))


def native_common_contact_constitutive_response_from_dict(raw, *, reference, port_response, source_observation):
    if type(raw) is not dict:
        raise ValueError("A local constitutive export must be a mapping")
    value = evaluate_native_common_contact_constitutive(reference, port_response, source_observation,
        branch=raw["branch"], receiver_frame=raw["kinematics"]["receiver_frame"], precision=raw["precision"])
    if native_common_contact_constitutive_response_to_dict(value) != raw:
        raise ValueError("Local constitutive export differs from complete authentic source/port replay")
    return value
