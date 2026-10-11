"""Actual body-field ports on authenticated NEW common-reference cuts.

Each port uses one shared physical-radius point and the two genuine material
labels supplied by the cut proof.  Cold seven-mode fields retain the material
clamp and fixed global enrichment tensors; HOT has its actual six affine
modes.  These are kinematic jumps and derivative columns, with no inferred
contact stiffness, rest law, owner clock, or material-admission decision.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from fractions import Fraction as F
import hashlib
import json
from math import isfinite

from .geometric_native_pose import FractionInterval as I, interval_atan, interval_pi, interval_sqrt, _rounded_interval
from .geometric_native_joint_spherical_field import rotate_joint_physical_depth
from .geometric_native_affine_body import AFFINE_SYMMETRIC_GENERATORS
from .geometric_native_common_reference import validate_native_common_reference
from .geometric_native_common_contact_cuts import (
    bind_common_contact_cuts, validate_common_contact_cuts, validate_common_contact_cut,
    evaluate_common_contact_cut, validate_common_contact_cut_bounds,
)
from .geometric_native_common_cold import bind_native_common_cold_body, validate_native_common_cold_body
from .geometric_native_common_hot import bind_native_common_hot_body, validate_native_common_hot_body


FORMAT = "native-common-contact-body-ports-1"
SCOPE = "AUTHENTIC_COMMON_CUT_BODY_KINEMATICS_ONLY"
_REGISTRY = {}


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
    if type(value) is float and isfinite(value):
        return value
    if type(value) in (str, int, bool) or value is None:
        return value
    raise TypeError("Common-contact ports require immutable literal source data")


def _encoded(value):
    return json.dumps(_payload(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _seal(value):
    return hashlib.sha256(_encoded({item.name: getattr(value, item.name)
        for item in fields(value) if item.name != "binding_sha256"})).hexdigest()


def _remember(value):
    result = replace(value, binding_sha256=_seal(value))
    _REGISTRY[type(result), result.binding_sha256] = _encoded(result)
    return result


def _registered(value, kind):
    if (type(value) is not kind or value.binding_sha256 != _seal(value)
            or _REGISTRY.get((kind, value.binding_sha256)) != _encoded(value)):
        raise ValueError("An altered or re-sealed common contact port lacks authentic replay")
    return value


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), I(0))


def _map(matrix, vector):
    return tuple(_dot(row, vector) for row in matrix)


def _cross(a, b):
    return a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]


def _square(x):
    return I(F(0) if x.lo <= 0 <= x.hi else min(x.lo*x.lo, x.hi*x.hi), max(x.lo*x.lo, x.hi*x.hi))


def _round(value, precision):
    return _rounded_interval(value, precision)


def _coordinates(value, count):
    if value is None:
        return (F(0),)*count
    if type(value) is not tuple or len(value) != count or any(type(x) is not F for x in value):
        raise ValueError("A contact body port needs its complete literal metre coordinates")
    return value


def _body_sha(body):
    return body.body_snapshot_sha256 if hasattr(body, "embedding_bank") else body.binding_sha256


class _KinematicScope:
    @property
    def scope(self):
        return SCOPE

    @property
    def contact_constitutive_law_inferred(self):
        return False

    @property
    def contact_stiffness_or_rest_metric_inferred(self):
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
    def original_sixteen_coordinate_history_modified(self):
        return False


@dataclass(frozen=True, slots=True)
class NativeCommonContactPort(_KinematicScope):
    cut: object
    donor_body: object
    receiver_body: object
    reference_snapshot_sha256: str
    source_snapshot_sha256: str
    donor_mode_order: tuple
    receiver_mode_order: tuple
    owner_rotation_coordinate_order: tuple
    geometry_enclosure_precision: int
    binding_sha256: str

    @property
    def port_id(self):
        return self.cut.cut_id


@dataclass(frozen=True, slots=True)
class NativeCommonContactPorts(_KinematicScope):
    reference_snapshot_sha256: str
    source_snapshot_sha256: str
    cuts: object
    ports: tuple
    geometry_enclosure_precision: int
    binding_sha256: str


def bind_native_common_contact_ports(reference, *, cuts=None, precision=160):
    reference = validate_native_common_reference(reference)
    cuts = bind_common_contact_cuts(reference, precision=precision) if cuts is None else validate_common_contact_cuts(cuts, reference=reference)
    if cuts.geometry_enclosure_precision != precision:
        raise ValueError("Port body/geometry binding needs one declared source enclosure precision")
    banks = {bank.component_id: bank for bank in reference.banks}
    owners = tuple(sorted({bank.owner for bank in reference.banks}))
    owner_order = tuple((owner, axis) for owner in owners for axis in ("x", "y", "z"))
    bodies = {}
    for cut in cuts.cuts:
        validate_common_contact_cut(cut, reference=reference)
        for component in (cut.donor_component_id, cut.receiver_component_id):
            if component in bodies:
                continue
            bank = banks[component]
            if bank.material_kind == "original_live":
                bodies[component] = bind_native_common_cold_body(component, reference=reference, precision=precision)
            elif bank.material_kind == "hot_birth":
                bodies[component] = bind_native_common_hot_body(component, reference=reference)
            else:
                raise ValueError("The actual cut has no supported authentic cold or HOT body")
    ports = []
    for cut in cuts.cuts:
        donor, receiver = bodies[cut.donor_component_id], bodies[cut.receiver_component_id]
        if not hasattr(donor, "embedding_bank"):
            raise ValueError("The retained actual cut must have its authentic cold donor")
        ports.append(_remember(NativeCommonContactPort(cut, donor, receiver,
            reference.reference_snapshot_sha256, reference.source_snapshot_sha256,
            donor.coordinate_order, receiver.coordinate_order, owner_order, precision, "")))
    return _remember(NativeCommonContactPorts(reference.reference_snapshot_sha256,
        reference.source_snapshot_sha256, cuts, tuple(ports), precision, ""))


def validate_native_common_contact_port(port, *, reference=None):
    port = _registered(port, NativeCommonContactPort)
    validate_common_contact_cut(port.cut, reference=reference)
    validate_native_common_cold_body(port.donor_body)
    if hasattr(port.receiver_body, "embedding_bank"):
        validate_native_common_cold_body(port.receiver_body)
    else:
        validate_native_common_hot_body(port.receiver_body)
    if (port.donor_body.component_id != port.cut.donor_component_id
            or port.receiver_body.component_id != port.cut.receiver_component_id
            or port.reference_snapshot_sha256 != port.cut.reference_snapshot_sha256
            or port.donor_body.reference_snapshot_sha256 != port.reference_snapshot_sha256
            or port.receiver_body.reference_snapshot_sha256 != port.reference_snapshot_sha256):
        raise ValueError("A common contact port substituted its actual body or reference")
    return port


def validate_native_common_contact_ports(value, *, reference=None):
    value = _registered(value, NativeCommonContactPorts)
    validate_common_contact_cuts(value.cuts, reference=reference)
    if tuple(port.cut for port in value.ports) != value.cuts.cuts:
        raise ValueError("The port collection substituted a genuine material cut")
    for port in value.ports:
        validate_native_common_contact_port(port, reference=reference)
    return value


@dataclass(frozen=True, slots=True)
class NativeCommonBodyPortFields:
    component_id: str
    material_kind: str
    body_snapshot_sha256: str
    reference_snapshot_sha256: str
    coordinates_m: tuple
    coordinate_order: tuple
    original_unit_direction_bounds: tuple | None
    original_quarter_labels_bounds: tuple | None
    material_beta_bounds_radians: I | None
    material_beta_gradient_bounds_per_m: tuple | None
    source_shape_bounds: I
    source_shape_beta_derivative_bounds: I
    field_active_inside_material_clamp: bool
    mode_displacement_columns: tuple
    mode_spatial_gradient_columns_per_m: tuple
    displacement_increment_bounds_m: tuple
    displaced_point_bounds_m: tuple
    incremental_gradient_bounds: tuple


def _cold_beta(body, q, precision):
    if type(q) is not tuple or len(q) != 3 or any(type(x) is not I for x in q):
        raise ValueError("The cold port requires its authenticated ORIGINAL Cartesian unit label")
    labels = tuple(_dot(axis, q) for axis in (body.original_quarter_start,
                     body.original_quarter_end, body.original_quarter_inward))
    if any(x.lo < 0 for x in labels):
        raise ValueError("The actual cold body label lacks a proved original material quarter")
    horizontal = interval_sqrt(_square(labels[0])+_square(labels[1]), precision)
    if horizontal.lo > 0:
        beta = interval_atan(labels[2]/horizontal, precision)
    elif labels[2].lo > 0:
        lower = interval_atan(I(labels[2].lo/horizontal.hi), precision).lo if horizontal.hi else interval_pi(precision).lo/2
        beta = I(lower, interval_pi(precision).hi/2)
    else:
        raise ValueError("A cold port label has an unresolved transverse angular chart")
    b = body.beta_cut_radians
    if beta.lo < b < beta.hi:
        raise ValueError("Split the actual port query at the cold material clamp; no midpoint decides support")
    return labels, beta, horizontal, beta.lo < b


def _body_fields(body, q, geometry, coordinates, *, side, precision, order):
    X, r = geometry.world_position_bounds_m, geometry.physical_radius_bounds_m
    zero_vector, zero_matrix = (I(0),)*3, ((I(0),)*3,)*3
    labels = beta = grad_beta = None
    chi, slope, active = I(1), I(0), True
    if hasattr(body, "embedding_bank"):
        labels, beta, horizontal, active = _cold_beta(body, q, precision)
        if active:
            e = tuple(_round(x/r, precision) for x in X)
            n = body.current_quarter_inward_bounds
            n_rotated = rotate_joint_physical_depth(n,
                body.angular_rate_per_physical_volume_depth_m,
                geometry.common_physical_volume_depth_bounds_m, precision=precision, order=order)
            # Q t_beta=(Q n-e*sin(beta))/cos(beta).  The source n includes
            # the original inward sign, including improper donor B frames.
            rate = _dot(n, _cross(tuple(I(x) for x in body.angular_rate_per_physical_volume_depth_m),
                                geometry.source_unit_world_direction_bounds))/horizontal
            grad_beta = tuple(_round((n_rotated[i]-e[i]*labels[2])/(r*horizontal)
                +(r*r/body.radius_m**2)*rate*e[i], precision) for i in range(3))
            t = beta/body.beta_cut_radians
            chi = _round((I(1)-t)*(I(1)-t)*(t+2)/2, precision)
            slope = _round(-F(3, 2)*(I(1)-t)*(I(1)+t)/body.beta_cut_radians, precision)
            modes = [tuple(_round(chi*x, precision) for x in e)]
            gradients = [tuple(tuple(_round(chi/r*(I(int(i == j))-e[i]*e[j])
                +slope*e[i]*grad_beta[j], precision) for j in range(3)) for i in range(3))]
            for generator in AFFINE_SYMMETRIC_GENERATORS:
                direction = tuple(_round(x/body.radius_m, precision) for x in _map(generator, X))
                modes.append(tuple(_round(chi*x, precision) for x in direction))
                gradients.append(tuple(tuple(_round(chi*generator[i][j]/body.radius_m
                    +slope*direction[i]*grad_beta[j], precision) for j in range(3)) for i in range(3)))
        else:
            chi, slope, grad_beta = I(0), I(0), zero_vector
            modes, gradients = [zero_vector]*7, [zero_matrix]*7
        kind = "original_live"
    else:
        if q is not None:
            raise ValueError("A HOT curved receiver cannot be assigned an original cold material label")
        modes = [tuple(_round(x/body.radius_m, precision) for x in _map(generator, X))
                 for generator in body.fixed_global_symmetric_generators]
        gradients = [tuple(tuple(I(x/body.radius_m) for x in row) for row in generator)
                     for generator in body.fixed_global_symmetric_generators]
        kind = "hot_birth"
    increment = tuple(_round(sum((coordinate*mode[i] for coordinate, mode in zip(coordinates, modes)), I(0)), precision)
                      for i in range(3))
    displaced = tuple(_round(x+u, precision) for x, u in zip(X, increment))
    gradient = tuple(tuple(_round(I(int(i == j))+sum((coordinate*matrix[i][j]
        for coordinate, matrix in zip(coordinates, gradients)), I(0)), precision) for j in range(3)) for i in range(3))
    return NativeCommonBodyPortFields(body.component_id, kind, _body_sha(body), body.reference_snapshot_sha256,
        coordinates, body.coordinate_order, q, labels, beta, grad_beta, chi, slope, active,
        tuple(modes), tuple(gradients), increment, displaced, gradient)


@dataclass(frozen=True, slots=True)
class NativeCommonContactPortResponse(_KinematicScope):
    port: NativeCommonContactPort
    geometry: object
    donor_fields: NativeCommonBodyPortFields
    receiver_fields: NativeCommonBodyPortFields
    mode_coordinate_order: tuple
    displacement_jump_bounds_m: tuple
    normal_jump_bounds_m: I
    down_jump_bounds_m: I
    angular_jump_bounds_m: I
    mode_jump_columns: tuple
    mode_jump_spatial_gradient_columns_per_m: tuple
    mode_jump_parameter_derivative_columns: tuple
    mode_jump_material_depth_derivative_columns: tuple
    projected_mode_jacobian_bounds: tuple
    owner_rotation_jump_columns_m: tuple
    projected_owner_rotation_jacobian_bounds_m: tuple
    owner_body_mode_cross_derivative_columns: tuple
    projected_owner_body_mode_cross_derivative_bounds: tuple
    precision: int
    rotation_order: int
    binding_sha256: str
    one_common_reference_point_used: bool = True
    owner_columns_are_infinitesimal_left_chart_at_displaced_sides: bool = True
    projection_frame_is_reference_geometry: bool = True
    unresolved_parameter_bands_promoted_to_material: bool = False


def evaluate_native_common_contact_port(reference, port, *, parameter_bounds=None,
        physical_radius_cubed_bounds_m3=None, donor_coordinates_m=None,
        receiver_coordinates_m=None, precision=160, order=28):
    reference = validate_native_common_reference(reference)
    port = validate_native_common_contact_port(port, reference=reference)
    geometry = validate_common_contact_cut_bounds(evaluate_common_contact_cut(reference, port.cut, parameter_bounds,
        physical_radius_cubed_bounds_m3, precision=precision, order=order), reference=reference)
    if (not geometry.genuine_query_support_certified or
            (geometry.parameter_expression == "interval" and not geometry.every_point_of_parameter_enclosure_is_material)):
        raise ValueError("An unresolved cut root/trim band is not an authenticated body-material port query")
    donor_q = _coordinates(donor_coordinates_m, len(port.donor_mode_order))
    receiver_q = _coordinates(receiver_coordinates_m, len(port.receiver_mode_order))
    donor = _body_fields(port.donor_body, geometry.donor_original_unit_direction_bounds,
                         geometry, donor_q, side="donor", precision=precision, order=order)
    receiver = _body_fields(port.receiver_body, geometry.receiver_original_unit_direction_bounds,
                            geometry, receiver_q, side="receiver", precision=precision, order=order)
    # The two reference X are one correlated source expression.  Subtract
    # only increments; independent interval X-X would invent a rest jump.
    jump = tuple(_round(r-d, precision) for r, d in zip(receiver.displacement_increment_bounds_m,
                                                     donor.displacement_increment_bounds_m))
    columns = tuple(tuple(-x for x in mode) for mode in donor.mode_displacement_columns)+receiver.mode_displacement_columns
    gradients = tuple(tuple(tuple(-x for x in row) for row in matrix)
        for matrix in donor.mode_spatial_gradient_columns_per_m)+receiver.mode_spatial_gradient_columns_per_m
    projections = (geometry.receiver_inward_unit_normal_bounds, geometry.inward_unit_down_direction_bounds,
                   geometry.angular_unit_tangent_bounds)
    projected = tuple(tuple(_round(_dot(direction, column), precision) for column in columns) for direction in projections)
    parameter_columns = tuple(tuple(_round(x, precision) for x in _map(matrix, geometry.parameter_tangent_bounds_m)) for matrix in gradients)
    depth_columns = tuple(tuple(_round(x, precision) for x in _map(matrix, geometry.material_depth_tangent_bounds)) for matrix in gradients)
    donor_owner, receiver_owner = port.donor_body.original_body.owner, port.receiver_body.original_body.owner
    mode_owners = (donor_owner,)*len(donor_q)+(receiver_owner,)*len(receiver_q)
    owner_columns, cross_columns = [], []
    for owner, label in port.owner_rotation_coordinate_order:
        axis = tuple(I(int(i == "xyz".index(label))) for i in range(3))
        d, r = int(owner == donor_owner), int(owner == receiver_owner)
        # At nonzero q use the actual displaced sides.  A shared-owner
        # reference X cancels algebraically before interval operations.
        vector = tuple((r-d)*x+r*ur-d*ud for x, ur, ud in zip(geometry.world_position_bounds_m,
                       receiver.displacement_increment_bounds_m, donor.displacement_increment_bounds_m))
        owner_columns.append(tuple(_round(x, precision) for x in _cross(axis, vector)))
        cross_columns.append(tuple(tuple(_round(x, precision) for x in _cross(axis, column))
                                   if owner == body_owner else (I(0),)*3
                                   for column, body_owner in zip(columns, mode_owners)))
    mode_order = tuple(("donor", name) for name in port.donor_mode_order)+tuple(("receiver", name) for name in port.receiver_mode_order)
    return _remember(NativeCommonContactPortResponse(port, geometry, donor, receiver, mode_order, jump,
        *tuple(_round(_dot(direction, jump), precision) for direction in projections),
        columns, gradients, parameter_columns, depth_columns, projected, tuple(owner_columns),
        tuple(tuple(_round(_dot(direction, column), precision) for column in owner_columns) for direction in projections),
        tuple(cross_columns), tuple(tuple(tuple(_round(_dot(direction, column), precision) for column in row)
            for row in cross_columns) for direction in projections), precision, order, ""))


def validate_native_common_contact_port_response(value, *, reference=None):
    value = _registered(value, NativeCommonContactPortResponse)
    validate_native_common_contact_port(value.port, reference=reference)
    validate_common_contact_cut_bounds(value.geometry, reference=reference)
    if not value.geometry.genuine_query_support_certified:
        raise ValueError("A body port cannot authenticate an unresolved geometry query")
    return value


def _export(value):
    result = dict(format=FORMAT, scope=SCOPE, **_payload(value))
    for name in ("contact_constitutive_law_inferred", "contact_stiffness_or_rest_metric_inferred",
            "force_equilibrium_certified", "physical_material_admission_authorized",
            "finite_owner_motion_or_whole_clock_certified", "original_sixteen_coordinate_history_modified"):
        result[name] = False
    return result


def native_common_contact_ports_to_dict(value):
    return _export(validate_native_common_contact_ports(value))


def native_common_contact_port_response_to_dict(value):
    return _export(validate_native_common_contact_port_response(value))


def native_common_contact_ports_from_dict(raw, reference):
    if type(raw) is not dict or raw.get("format") != FORMAT:
        raise ValueError("Expected a complete common-contact ports packet")
    value = bind_native_common_contact_ports(reference, precision=raw["geometry_enclosure_precision"])
    if _encoded(native_common_contact_ports_to_dict(value)) != _encoded(raw):
        raise ValueError("Serialized common ports differ from authenticated full-source replay")
    return value


def _read_fraction(value):
    if type(value) is not list or len(value) != 2 or any(type(x) is not int for x in value) or value[1] <= 0:
        raise ValueError("A port replay requires literal fraction pairs")
    return F(*value)


def _read_interval(value):
    if type(value) is not dict or set(value) != {"lo", "hi"}:
        raise ValueError("A port replay requires complete outward bounds")
    return I(_read_fraction(value["lo"]), _read_fraction(value["hi"]))


def native_common_contact_port_response_from_dict(raw, reference):
    """Re-evaluate actual geometry and fields; a copied seal grants no trust."""
    if type(raw) is not dict or raw.get("format") != FORMAT:
        raise ValueError("Expected a complete common-contact response packet")
    collection = bind_native_common_contact_ports(reference,
        precision=raw["port"]["geometry_enclosure_precision"])
    matches = tuple(port for port in collection.ports if port.cut.cut_id == raw["port"]["cut"]["cut_id"])
    if len(matches) != 1:
        raise ValueError("A response replay has no unique genuine material cut")
    geometry = raw["geometry"]
    expression = geometry["parameter_expression"]
    parameter = expression[7:] if expression in ("source_start", "source_end") else _read_interval(geometry["parameter_bounds"])
    value = evaluate_native_common_contact_port(reference, matches[0], parameter_bounds=parameter,
        physical_radius_cubed_bounds_m3=_read_interval(geometry["common_physical_radius_cubed_bounds_m3"]),
        donor_coordinates_m=tuple(_read_fraction(x) for x in raw["donor_fields"]["coordinates_m"]),
        receiver_coordinates_m=tuple(_read_fraction(x) for x in raw["receiver_fields"]["coordinates_m"]),
        precision=raw["precision"], order=raw["rotation_order"])
    if _encoded(native_common_contact_port_response_to_dict(value)) != _encoded(raw):
        raise ValueError("Serialized port fields differ from authenticated geometry/body replay")
    return value
