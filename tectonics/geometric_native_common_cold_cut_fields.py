"""Whole curved-cut cold fields on authentic interval material labels.

The source cut supplies a correlated unit donor label, including its named
uncertain endpoints.  Its interval rectangle is never substituted for a
rational ray or passed through a point chronology oracle.  On this genuine
live-envelope boundary the declared archive carrier's positive-gap history
and its first material derivative vanish.  The bank displacement nevertheless
retains its original roof, depth twist and quintic quarter-seam partition.
This is a field port, not a contact law or a material admission certificate.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from fractions import Fraction as F

from .geometric_native_pose import FractionInterval as I, interval_sqrt, interval_atan
from .geometric_native_spherical_solid import interval_cbrt
from .geometric_native_joint_spherical_field import rotate_joint_physical_depth
from .geometric_native_archive_material_carrier import _rotation, _sin_cos
from .geometric_native_affine_body import AFFINE_SYMMETRIC_GENERATORS
from .geometric_native_common_cold_material_carrier import (
    _registered, _precision, _square, _signed_atan, _roof_chart_correction,
    _saved_json_kinds,
)
from .geometric_native_common_contact_cuts import (
    validate_common_contact_cut_bounds, common_contact_cut_bounds_to_dict,
)
from .geometric_native_common_carrier_source_work import (
    _round, _map, _cross, _multiply, _transpose, _determinant,
    _payload, _encoded, _digest, _typed_guard,
)

FORMAT = "native-common-whole-cold-curved-cut-field-1"
SCOPE = "AUTHENTIC_CURVED_COLD_CUT_MATERIAL_FIELDS_AND_MODE_GRADIENTS_ONLY"
_ZERO = (I(0),)*3
_ZM = (_ZERO,)*3
_GUARDS = {}


def _body(value):
    return {f.name: getattr(value, f.name) for f in fields(value)
            if f.name != "binding_sha256"}


def _intersection(a, b, message):
    if a.hi < b.lo or b.hi < a.lo:
        raise ValueError(message)
    return I(max(a.lo, b.lo), min(a.hi, b.hi))


def _absolute(value):
    return I(0 if value.lo <= 0 <= value.hi else min(abs(value.lo), abs(value.hi)),
             max(abs(value.lo), abs(value.hi)))


def _source_label(bank, bounds):
    """Use the exact source-expression quarter and unit identities.

    These intersections concern the actual correlated face.  They do not
    claim that each point of an endpoint rectangle is a material point.
    """
    q = tuple(_intersection(x, I(-1, 1), "A source unit label escaped its unit bound")
              for x in bounds.donor_original_unit_direction_bounds)
    result = list(q)
    for plane in (bank.original_quarter_start, bank.original_quarter_end,
                  bank.original_quarter_inward):
        nonzero = tuple(i for i, x in enumerate(plane) if x)
        if len(nonzero) != 1 or abs(plane[nonzero[0]]) != 1:
            raise ValueError("This source cut requires its authentic original axis quarter")
        axis = nonzero[0]
        sign = plane[axis]
        margin = sign*result[axis]
        if margin.hi < 0:
            raise ValueError("The exact cut source expression contradicts its original donor quarter")
        result[axis] = sign*I(max(F(0), margin.lo), margin.hi)
    return tuple(result)


def _quintic_enclosure(angle, delta):
    """Saturated quintic value and derivative, including crossing intervals."""
    clipped = I(max(-delta, min(delta, angle.lo)),
                max(-delta, min(delta, angle.hi)))
    t = (clipped+delta)/(2*delta)
    value = lambda x: x**3*(10+x*(-15+6*x))
    weight = I(value(t.lo), value(t.hi))
    candidates = (t.lo, t.hi)+((F(1, 2),) if t.lo <= F(1, 2) <= t.hi else ())
    slopes = tuple(30*x*x*(1-x)*(1-x)/(2*delta) for x in candidates)
    low = min(slopes[:2])
    if angle.lo <= -delta or angle.hi >= delta:
        low = F(0)
    return weight, I(low, max(slopes))


def _seam_axis(q, numerator, denominator, radius, delta, precision):
    a, b = q[numerator], _absolute(q[denominator])
    sine, cosine = _sin_cos(I(delta), precision)
    tangent = sine/cosine
    if a.lo >= tangent.hi*b.hi:
        return I(1), _ZERO
    if a.hi <= -tangent.hi*b.hi:
        return I(0), _ZERO
    if b.lo <= 0:
        raise ValueError("Split the cut label enclosure at an unresolved seam-axis pole")
    angle = _signed_atan(a/b, precision)
    weight, slope = _quintic_enclosure(angle, delta)
    if slope.hi == 0:
        return weight, _ZERO
    sign = 1 if q[denominator].lo > 0 else -1
    # Unit-source correlation retains q_y^2+q_z^2=1-q_x^2.  Intersecting
    # two independently valid enclosures avoids a rectangle's zero corner.
    transverse = _intersection(_square(q[1])+_square(q[2]), I(1)-_square(q[0]),
                               "A seam label contradicts the source unit identity")
    if transverse.lo <= 0:
        raise ValueError("Split the cut label enclosure away from the owner pole")
    derivative = [I(0)]*3
    derivative[numerator] = b/(radius*transverse)
    derivative[denominator] = -sign*a/(radius*transverse)
    return _round(weight, precision), tuple(_round(slope*x, precision) for x in derivative)


def _partitions(system, owner, q, radius, precision):
    delta = system.whole_domain_certificate.seam_half_width_radians
    wy, gy = _seam_axis(q, 1, 2, radius, delta, precision)
    wz, gz = _seam_axis(q, 2, 1, radius, delta, precision)
    result = []
    for bank in system.banks:
        if bank.owner != owner:
            continue
        signs = tuple(sum((axis[i] for axis in (bank.original_quarter_start,
                                               bank.original_quarter_end)), F(0)) for i in (1, 2))
        fy, fz = (wy if signs[0] > 0 else I(1)-wy), (wz if signs[1] > 0 else I(1)-wz)
        dy = gy if signs[0] > 0 else tuple(-x for x in gy)
        dz = gz if signs[1] > 0 else tuple(-x for x in gz)
        result.append((bank, _round(fy*fz, precision),
                       tuple(_round(dy[i]*fz+fy*dz[i], precision) for i in range(3))))
    return tuple(result)


def _bank_modes(bank, q, depth, precision):
    """Pure interval version of the authentic895 original-roof bank formula."""
    R = bank.radius_m
    top = R+bank.roof_radial_displacement_m
    r0 = interval_cbrt(I(R**3)-3*R**2*depth, precision)
    rold = interval_cbrt(I(top**3)-3*R**2*depth, precision)
    tau = depth+(R**3-top**3)/(3*R**2)
    Q = bank.material_owner_pose_bounds
    v = _map(Q, q)
    e = rotate_joint_physical_depth(v, bank.angular_rate_per_physical_volume_depth_m,
                                    tau, precision=precision, order=28)
    cols = tuple(rotate_joint_physical_depth(tuple(I(int(i == j)) for i in range(3)),
        bank.angular_rate_per_physical_volume_depth_m, tau, precision=precision, order=28)
        for j in range(3))
    depthpose = _transpose(cols)
    label = sum((a*x for a, x in zip(bank.original_quarter_inward, q)), I(0))
    label = _intersection(label, I(0, 1), "A bank field crossed the source owner hemisphere")
    b = bank.beta_cut_radians
    sinb = _sin_cos(I(b), precision)[0]
    if label.lo >= sinb.hi:
        beta, chi, slope = I(b, F(2)), I(0), I(0)
    else:
        h = interval_sqrt(I(1)-_square(label), precision+16)
        if h.lo <= 0:
            raise ValueError("Split an active curved-cut bank label away from its original pole")
        beta = interval_atan(label/h, precision)
        t = I(min(F(1), beta.lo/b), min(F(1), beta.hi/b))
        shape = lambda x: (1-x)**2*(x+2)/2
        chi = _round(I(shape(t.hi), shape(t.lo)), precision)
        slope = _round(I(-F(3, 2)*(1-t.lo*t.lo)/b,
                          -F(3, 2)*(1-t.hi*t.hi)/b), precision)
    X = tuple(_round(rold*x, precision) for x in e)
    axis_e = _cross(tuple(I(x) for x in bank.angular_rate_per_physical_volume_depth_m), e)
    dq = tuple(tuple((I(int(i == j))-q[i]*q[j])/r0 for j in range(3)) for i in range(3))
    de0 = _multiply(_multiply(depthpose, Q), dq)
    dtau = tuple(-r0*r0/R**2*x for x in q)
    de = tuple(tuple(_round(de0[i][j]+axis_e[i]*dtau[j], precision)
                     for j in range(3)) for i in range(3))
    drold = tuple(r0*r0/(rold*rold)*x for x in q)
    Aold = tuple(tuple(_round(e[i]*drold[j]+rold*de[i][j], precision)
                       for j in range(3)) for i in range(3))
    correction = _roof_chart_correction(q, r0, rold, precision)
    if chi.hi == 0:
        modes, gradients = (_ZERO,)*7, (_ZM,)*7
    else:
        h = interval_sqrt(I(1)-_square(label), precision+16)
        if h.lo <= 0:
            raise ValueError("Split an active curved-cut bank gradient away from its original pole")
        gb = tuple(_round((I(bank.original_quarter_inward[i])-label*q[i])/(r0*h), precision)
                   for i in range(3))
        modes = [tuple(_round(chi*x, precision) for x in e)]
        gradients = [tuple(tuple(_round(chi*de[i][j]+slope*e[i]*gb[j], precision)
                                 for j in range(3)) for i in range(3))]
        for E in AFFINE_SYMMETRIC_GENERATORS:
            EX = _map(E, X)
            modes.append(tuple(_round(chi*x/R, precision) for x in EX))
            ED = _multiply(tuple(tuple(I(x) for x in row) for row in E), Aold)
            gradients.append(tuple(tuple(_round(chi*ED[i][j]/R+slope*EX[i]/R*gb[j], precision)
                                           for j in range(3)) for i in range(3)))
        modes, gradients = tuple(modes), tuple(gradients)
    return dict(original_parent_id=bank.original_parent_id, component_id=bank.component_id,
                original_radius_bounds_m=r0, old_radius_bounds_m=rold,
                physical_volume_depth_bounds_m=tau, material_beta_bounds_radians=beta,
                source_shape_bounds=chi, source_shape_beta_derivative_bounds=slope,
                reference_position_bounds_m=X, reference_original_gradient_bounds=Aold,
                roof_chart_correction_bounds=correction,
                mode_displacement_columns=modes, mode_original_gradient_columns_per_m=gradients)


@dataclass(frozen=True, slots=True)
class NativeCommonColdCutField:
    system_binding_sha256: str
    archive_carrier_model_sha256: str
    cut_bounds_snapshot_sha256: str
    cut_source_json: str
    source_bank_json: str
    source_owner_banks_json: str
    original_parent_id: str
    original_material_id: str
    component_id: str
    owner: int
    original_unit_direction_bounds: tuple
    material_depth_bounds_m: I
    original_reference_radius_bounds_m: I
    archive_carrier_position_bounds_m: tuple
    archive_carrier_original_gradient_bounds: tuple
    fixed_radial_scale: F
    displacement_bounds_m: tuple
    position_bounds_m: tuple
    original_cartesian_gradient_bounds: tuple
    volume_jacobian_bounds: I
    density_pullback_multiplier_bounds: I
    seam_weights: tuple
    seam_weight_original_gradients: tuple
    bank_field_enclosures: tuple
    coordinate_order: tuple
    bound_coordinate_increments: tuple
    mode_displacement_columns: tuple
    mode_original_gradient_columns_per_m: tuple
    parameter_expression: str
    genuine_query_support_certified: bool
    every_point_of_parameter_enclosure_is_material: bool
    precision: int
    binding_sha256: str
    scope: str = SCOPE
    original_input_frame: str = "ORIGINAL_PARENT_CARTESIAN"
    source_unit_carrier_correlation_retained: bool = True
    exact_live_envelope_activation_and_gradient_zero: bool = True
    whole_source_query_enclosed: bool = True
    interval_midpoint_or_rational_ray_proxy_used: bool = False
    physical_contact_traction_law_defined: bool = False
    physical_material_admission_authorized: bool = False
    source_history_advanced: bool = False
    accepted_new_native_intervals: int = 0


def evaluate_native_common_cold_cut_field(system, cut_bounds, *,
        coordinate_increments=None, precision=96):
    """Enclose the entire actual queried curved face, including named roots.

    ``cut_bounds`` must come from the old authenticated geometric evaluator.
    Whole and named endpoint expressions remain distinct in the full source
    DTO.  The accepted cold displacement remains linear in its56 declared
    exact amplitudes; no owner chart or additional coordinate is introduced.
    """
    carrier = _registered(system)
    _precision(precision)
    bounds = validate_common_contact_cut_bounds(cut_bounds)
    cut = bounds.cut
    if (cut.kind != "true_curved_envelope" or cut.donor_material_kind != "original_live"
            or cut.receiver_material_kind != "hot_birth"):
        raise ValueError("This field port requires the authentic live-donor/HOT curved envelope")
    bank = next((b for b in system.banks if b.original_parent_id == cut.donor_original_parent_id), None)
    if (bank is None or bank.component_id != cut.donor_component_id
            or bank.owner != carrier.donor_owner or bank.radius_m != carrier.radius_m):
        raise ValueError("The actual cut donor differs from its registered whole cold bank")
    depth = bounds.donor_material_depth_bounds_m
    if type(depth) is not I or not 0 <= depth.lo <= depth.hi <= bank.areal_material_thickness_m:
        raise ValueError("The complete cut depth lies outside its authentic full cold material shell")
    coordinates = system.bound_coordinate_increments if coordinate_increments is None else coordinate_increments
    if (type(coordinates) is not tuple or len(coordinates) != 56
            or any(type(x) is not F for x in coordinates)
            or any(abs(x) > envelope for x, envelope in zip(coordinates, system.coordinate_envelope_abs_m))):
        raise ValueError("The cut field retains the exact declared56-coordinate support envelope")
    q = _source_label(bank, bounds)
    r0 = interval_cbrt(I(bank.radius_m**3)-3*bank.radius_m**2*depth, precision)
    pose = _rotation(carrier.donor_axis, F(1), precision)
    archive_x = tuple(_round(r0*x, precision) for x in _map(pose, q))
    partitions = _partitions(system, bank.owner, q, r0, precision)
    modes, gradients = [_ZERO]*56, [_ZM]*56
    index = {b.component_id: tuple(i for i, row in enumerate(system.coordinate_order)
                                   if row[1] == b.component_id) for b in system.banks}
    bank_fields = []
    for sourcebank, weight, dweight in partitions:
        if weight.hi == 0 and all(x.lo == x.hi == 0 for x in dweight):
            continue
        field = _bank_modes(sourcebank, q, depth, precision)
        bank_fields.append(field)
        for k, globalindex in enumerate(index[sourcebank.component_id]):
            psi = field["mode_displacement_columns"][k]
            G = field["mode_original_gradient_columns_per_m"][k]
            modes[globalindex] = tuple(_round(weight*x, precision) for x in psi)
            gradients[globalindex] = tuple(tuple(_round(weight*G[i][j]+psi[i]*dweight[j], precision)
                                                 for j in range(3)) for i in range(3))
    u = tuple(_round(sum((a*v[i] for a, v in zip(coordinates, modes)), I(0)), precision) for i in range(3))
    du = tuple(tuple(_round(sum((a*g[i][j] for a, g in zip(coordinates, gradients)), I(0)), precision)
                     for j in range(3)) for i in range(3))
    certificate = system.whole_domain_certificate.donor
    s = certificate.fixed_radial_scale
    position = tuple(_round(s*x+y, precision) for x, y in zip(archive_x, u))
    A = tuple(tuple(_round(s*pose[i][j]+du[i][j], precision) for j in range(3)) for i in range(3))
    direct = _round(_determinant(A), precision)
    jac = _intersection(direct, certificate.volume_jacobian_bounds,
                        "The actual cut derivative contradicts the whole-owner Jacobian theorem")
    if jac.lo <= 0:
        raise ValueError("The complete cold cut has no positive original Cartesian Jacobian")
    canonical = lambda value: _encoded(_payload(value)).decode()
    result = NativeCommonColdCutField(system.binding_sha256, carrier.model_sha256,
        bounds.bounds_snapshot_sha256, canonical(common_contact_cut_bounds_to_dict(bounds)),
        canonical(bank), canonical(tuple(b for b in system.banks if b.owner == bank.owner)),
        bank.original_parent_id, bank.original_material_id, bank.component_id, bank.owner, q, depth,
        r0, archive_x, pose, s, u, position, A, jac, _round(1/jac, precision),
        tuple((b.original_parent_id, w) for b, w, _ in partitions),
        tuple((b.original_parent_id, g) for b, _, g in partitions), tuple(bank_fields),
        system.coordinate_order, coordinates, tuple(modes), tuple(gradients), bounds.parameter_expression,
        bounds.genuine_query_support_certified, bounds.every_point_of_parameter_enclosure_is_material,
        precision, "")
    result = replace(result, binding_sha256=_digest(_body(result)))
    _GUARDS[result.binding_sha256] = _typed_guard(result)
    return result


def native_common_cold_cut_field_to_dict(value):
    if (type(value) is not NativeCommonColdCutField
            or _digest(_body(value)) != value.binding_sha256
            or _GUARDS.get(value.binding_sha256) != _typed_guard(value)):
        raise ValueError("An altered, resealed or typed-aliased cold cut field lacks source authority")
    return dict(format=FORMAT, **_payload(value))


def native_common_cold_cut_field_from_dict(raw, *, system, cut_bounds,
        expected_coordinate_increments=None, precision=96):
    _saved_json_kinds(raw)
    if type(raw) is not dict or raw.get("format") != FORMAT:
        raise ValueError("A complete authentic cold cut field DTO is required")
    rebuilt = evaluate_native_common_cold_cut_field(system, cut_bounds,
        coordinate_increments=expected_coordinate_increments, precision=precision)
    if _encoded(raw) != _encoded(native_common_cold_cut_field_to_dict(rebuilt)):
        raise ValueError("The complete cold cut field restart changed its expected source or query")
    return rebuilt


evaluate_native_common_cold_cut_fields = evaluate_native_common_cold_cut_field
