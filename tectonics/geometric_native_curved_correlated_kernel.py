"""Correlated scalar mechanics on the unchanged shared curved reference face.

The receiving-frame normal projection is evaluated before any world-coordinate
interval subtraction.  Source unit-frame identities and a proper relative
rotation are retained algebraically.  In particular a common owner rotation
has a literal zero relative increment, and ``u.(Q-I)u`` is
``-|(Q-I)u|**2/2``.  Frame boxes remain outward aliases of the same authenticated
native orthonormal frame; their midpoints never replace that frame.

This is the same saved-reference shear followed by radial and owner motion,
with the same current-area series-column potential.  The reduced potential
does not supply three-dimensional strain energy or material admission.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json

from .geometric_native_pose import FractionInterval as I, _rounded_interval, interval_sqrt
from .geometric_native_spherical_solid import interval_cbrt
from .geometric_native_physical_pair_time import _relative_change, _rotated
from .geometric_native_curved_contact_energy import _left_jacobian_columns


# A self-consistently re-sealed DTO is not a native geometry authority.  A
# native receipt must be rebuilt from an actually replayed field in this
# process; only its immutable snapshot is cached for subsequent cell work.
_NATIVE_AUTHENTICATED_SNAPSHOTS = set()


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), 0)


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _exact(value, name):
    if type(value) is not F:
        raise ValueError(name+" requires a literal Fraction")
    return value


def _interval(value, name):
    if type(value) is not I:
        raise ValueError(name+" requires a literal FractionInterval")
    return value


def _triple(value, name, kind):
    if type(value) is not tuple or len(value) != 3 or any(type(x) is not kind for x in value):
        raise ValueError(name+" requires an immutable exact triple")
    return value


def _payload(value):
    if type(value) is F:
        return [value.numerator, value.denominator]
    if type(value) is I:
        return {"lo": _payload(value.lo), "hi": _payload(value.hi)}
    if type(value) is tuple:
        return [_payload(x) for x in value]
    if type(value) in (str, int, bool) or value is None:
        return value
    raise TypeError("A correlated source snapshot contains an unsupported mutable value")


@dataclass(frozen=True, slots=True)
class CurvedCorrelatedSource:
    radius_m: F
    donor_radial_before_m: F
    receiver_radial_before_m: F
    donor_depth_offset_m: F
    receiver_depth_bounds_m: I
    angle_bounds_radians: I
    native_angle_end_bounds_radians: I
    start_direction_bounds: tuple
    tangent_direction_bounds: tuple
    normal_direction_bounds: tuple
    shared_depth_axis_per_m: tuple
    normal_orientation: int
    source_preload_gap_m: F
    normal_compliances_m_pa: tuple
    effective_normal_stiffness_pa_m: F
    normal_column_lengths_m: tuple
    family_id: str
    source_support_sha256: str
    source_epoch_sha256: str
    native_source_authenticated: bool
    precision: int
    order: int
    source_snapshot_sha256: str

    @property
    def source_frame_aliases_retained(self):
        return True

    @property
    def receiver_material_depth_bounds_m(self):
        return self.receiver_depth_bounds_m

    @property
    def donor_material_depth_bounds_m(self):
        return self.receiver_depth_bounds_m+self.donor_depth_offset_m

    @property
    def effective_stiffness_pa_m(self):
        return self.effective_normal_stiffness_pa_m

    @property
    def source_frame_midpoints_used_as_geometry(self):
        return False

    @property
    def source_preload_retained_at_activation(self):
        return True

    @property
    def complete_three_dimensional_strain_energy_certified(self):
        return False

    @property
    def physical_material_admission_authorized(self):
        return False


def _source_snapshot(source):
    return tuple(getattr(source, name) for name in source.__dataclass_fields__
                 if name != "source_snapshot_sha256")


def _snapshot_hash(values):
    return hashlib.sha256(json.dumps(_payload(values), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_curved_correlated_source(source):
    if type(source) is not CurvedCorrelatedSource:
        raise ValueError("A complete immutable correlated source receipt is required")
    if source.source_snapshot_sha256 != _snapshot_hash(_source_snapshot(source)):
        raise ValueError("The correlated source differs from its complete immutable snapshot")
    if source.native_source_authenticated and source.source_snapshot_sha256 not in _NATIVE_AUTHENTICATED_SNAPSHOTS:
        raise ValueError("A re-sealed source receipt cannot replace actual native field authentication")
    return source


def prepare_curved_correlated_reference(*, radius_m, donor_radial_before_m,
        receiver_radial_before_m, receiver_depth_bounds_m, angle_bounds_radians,
        start_direction_bounds, tangent_direction_bounds, normal_direction_bounds,
        shared_depth_axis_per_m, normal_orientation, source_preload_gap_m,
        normal_compliances_m_pa, precision=128, order=28):
    """Declare a pure orthonormal-frame model; this supplies no native authority.

    Non-singleton frame boxes declare aliases of one common orthonormal frame,
    rather than three independently selectable vectors.  Native callers obtain
    these aliases from the authenticated great-circle frame constructor below.
    """
    R, db, rb, preload = (_exact(x, "Reference scalar") for x in
        (radius_m, donor_radial_before_m, receiver_radial_before_m, source_preload_gap_m))
    depth, angle = (_interval(x, "Complete reference domain") for x in
        (receiver_depth_bounds_m, angle_bounds_radians))
    a, b, n = (_triple(v, "Reference unit-frame aliases", I) for v in
        (start_direction_bounds, tangent_direction_bounds, normal_direction_bounds))
    A = _triple(shared_depth_axis_per_m, "Shared angular depth flow", F)
    C = _triple(normal_compliances_m_pa, "Retained series compliances", F)
    if R <= 0 or R+db <= 0 or R+rb <= 0 or depth.lo < 0 or angle.lo < 0 or any(c <= 0 for c in C):
        raise ValueError("Reference radius, complete depth and compliances must be physical")
    if type(normal_orientation) is not int or normal_orientation not in (-1, 1):
        raise ValueError("A literal physical orientation +1 or -1 is required")
    if type(precision) is not int or not 64 <= precision <= 512 or type(order) is not int or not 8 <= order <= 64:
        raise ValueError("Correlated reference precision/order exceeds the declared native domain")
    if any(not _dot(v, v).lo <= 1 <= _dot(v, v).hi for v in (a, b, n)):
        raise ValueError("Reference frame aliases must enclose unit vectors")
    if any(not _dot(x, y).lo <= 0 <= _dot(x, y).hi for x, y in ((a, b), (a, n), (b, n))):
        raise ValueError("Reference frame aliases must enclose a common orthogonal frame")
    if any(x.hi < y.lo or y.hi < x.lo for x, y in zip(_cross(a, b), n)):
        raise ValueError("The complete oriented source frame must satisfy a cross b = n")
    offset = ((R+db)**3-(R+rb)**3)/(3*R*R)
    if depth.lo+offset < 0:
        raise ValueError("Shared physical reference depth lies outside the donor material")
    if (R+rb)**3-3*R*R*depth.hi <= 0:
        raise ValueError("The physical reference face cannot reach the sphere centre")
    tau = depth+(R**3-(R+rb)**3)/(3*R*R)
    if sum(map(abs, A), F(0))*max(abs(tau.lo), abs(tau.hi)) > F(1, 2):
        raise ValueError("The saved common depth shear exceeds its native small-rotation domain")
    values = (R, db, rb, offset, depth, angle, I(angle.hi), a, b, n, A,
        normal_orientation, preload, C, 1/sum(C, F(0)), (), "", "", "", False, precision, order)
    return CurvedCorrelatedSource(*values, _snapshot_hash(values))


def prepare_native_curved_correlated_source(field, family_id, *,
        donor_radial_before_m, receiver_radial_before_m, precision=128, order=28):
    """Authenticate an actual complete native carrier and immutable bank source once."""
    from .geometric_native_joint_spherical_field import validate_native_joint_spherical_field, joint_spherical_component
    from .geometric_native_birth_continuum_motion import native_birth_continuum_frame
    from .geometric_solid_contact_closure import _parameters
    field = validate_native_joint_spherical_field(field)
    rows = tuple(row for row in field.support.interface_carriers
        if row.declaration.family_id == family_id)
    if type(family_id) is not str or len(rows) != 1:
        raise ValueError("A unique actual native carrier is required")
    row = rows[0]
    donor = joint_spherical_component(field, row.donor_component.component_id)
    receiver = joint_spherical_component(field, row.receiver_component.component_id)
    db, rb = (_exact(x, "Authoritative bank before displacement") for x in
        (donor_radial_before_m, receiver_radial_before_m))
    if donor.layer.top_radial_displacement_m != db or receiver.layer.top_radial_displacement_m != rb:
        raise ValueError("Before displacements must match the authenticated source banks")
    if donor.layer.radius_m != receiver.layer.radius_m or donor.angular_rate_per_physical_volume_depth_m != receiver.angular_rate_per_physical_volume_depth_m:
        raise ValueError("Actual banks require the same radius and saved common depth flow")
    R = donor.layer.radius_m
    offset = ((R+db)**3-(R+rb)**3)/(3*R*R)
    depth = I(max(F(0), -offset), min(receiver.layer.areal_thickness_m,
        donor.layer.areal_thickness_m-offset))
    if depth.width <= 0:
        raise ValueError("The actual banks have no positive complete common physical face")
    a, b, inward = native_birth_continuum_frame(row, precision=max(128, precision))
    n = tuple(_rounded_interval(x, precision+8) for x in _cross(a, b))
    orientation = _dot(n, inward)
    sign = 1 if orientation.lo > 0 else -1 if orientation.hi < 0 else 0
    if not sign:
        raise ValueError("The actual native receiving orientation is unresolved")
    columns, law = row.declaration.parameters.columns, row.declaration.parameters.law
    _parameters(columns, law)
    if law.normal_stiffness_pa_m <= 0:
        raise ValueError("Actual curved mechanics requires positive source normal stiffness")
    C = (F.from_float(columns.sub_length_m)/F.from_float(columns.sub_normal_modulus_pa),
         F.from_float(columns.over_length_m)/F.from_float(columns.over_normal_modulus_pa),
         1/F.from_float(law.normal_stiffness_pa_m))
    base = prepare_curved_correlated_reference(radius_m=R, donor_radial_before_m=db,
        receiver_radial_before_m=rb, receiver_depth_bounds_m=depth,
        angle_bounds_radians=I(0, row.native_angle_bounds_radians.hi),
        start_direction_bounds=a, tangent_direction_bounds=b, normal_direction_bounds=n,
        shared_depth_axis_per_m=donor.angular_rate_per_physical_volume_depth_m,
        normal_orientation=sign, source_preload_gap_m=F.from_float(row.declaration.birth_gap_free_m),
        normal_compliances_m_pa=C, precision=precision, order=order)
    replacements = dict(native_angle_end_bounds_radians=row.native_angle_bounds_radians,
        family_id=family_id, source_support_sha256=field.support.support_sha256,
        source_epoch_sha256=field.support.source_epoch.epoch_sha256,
        normal_column_lengths_m=(F.from_float(columns.sub_length_m), F.from_float(columns.over_length_m)),
        native_source_authenticated=True)
    values = tuple(replacements.get(name, getattr(base, name)) for name in
        base.__dataclass_fields__ if name != "source_snapshot_sha256")
    source = CurvedCorrelatedSource(*values, _snapshot_hash(values))
    _NATIVE_AUTHENTICATED_SNAPSHOTS.add(source.source_snapshot_sha256)
    return source


class IntervalScalarAlgebra:
    """Small outward scalar protocol also implemented by the integration jets."""
    def __init__(self, precision):
        self.precision = precision

    def constant(self, value):
        return value if type(value) is I else I(value)

    def round(self, value):
        return _rounded_interval(value, self.precision)

    def sqrt(self, value):
        return interval_sqrt(value, self.precision)

    def cbrt(self, value):
        return interval_cbrt(value, self.precision)

    def sin(self, value):
        from .geometric_native_active_contact_line_work import _trig_ranges_signed
        return _trig_ranges_signed(value.lo, value.hi, self.precision)[0]

    def cos(self, value):
        from .geometric_native_active_contact_line_work import _trig_ranges_signed
        return _trig_ranges_signed(value.lo, value.hi, self.precision)[1]


@dataclass(frozen=True, slots=True)
class CurvedCorrelatedScalars:
    gap: object
    area: object
    potential: object
    pressure: object
    gap_gradient: tuple
    area_gradient: tuple
    potential_gradient: tuple
    gauss_newton_hessian: tuple
    relative_owner_matrix_exact_zero: bool

    @property
    def free_gap_bounds_m(self):
        return self.gap

    @property
    def area_density_bounds_m2_per_radian_per_material_m(self):
        return self.area

    @property
    def potential_density_bounds_j_per_radian_per_material_m(self):
        return self.potential

    @property
    def normal_pressure_bounds_pa(self):
        return self.pressure

    @property
    def gap_gradient_bounds(self):
        return self.gap_gradient

    @property
    def area_gradient_bounds(self):
        return self.area_gradient

    @property
    def potential_gradient_bounds(self):
        return self.potential_gradient

    @property
    def current_curved_area_and_normal_derivatives_included(self):
        return True

    @property
    def complete_three_dimensional_strain_energy_certified(self):
        return False

    @property
    def physical_material_admission_authorized(self):
        return False


@lru_cache(maxsize=256)
def _owner_matrices(da, ra, precision, order):
    basis = tuple(tuple(F(i == j) for i in range(3)) for j in range(3))
    change_columns = tuple(_relative_change(column, da, ra, I(1), precision, order)
        for column in basis)
    receiver_columns = tuple(_rotated(column, ra, I(1), precision, order) for column in basis)
    change = tuple(tuple(change_columns[j][i] for j in range(3)) for i in range(3))
    receiver = tuple(tuple(receiver_columns[j][i] for j in range(3)) for i in range(3))
    return change, receiver, tuple(_left_jacobian_columns(axis, precision, order) for axis in (da, ra))


def evaluate_curved_correlated_scalars(source, theta, z, *, donor_radial_increment_m,
        receiver_radial_increment_m, donor_owner_axis, receiver_owner_axis,
        time_fraction=F(1), algebra=None, include_gauss_newton=False):
    """Evaluate the correlated closed-branch rows in a caller's scalar algebra.

    The integrator supplies bivariate derivative jets.  This function performs
    no midpoint substitution or clipping, so those jets differentiate exactly
    the same current-area potential.  ``bound_curved_correlated_cell`` encloses
    the unilateral branch when its gap straddles zero.
    """
    validate_curved_correlated_source(source)
    di, ri, clock = (_exact(x, "Actual radial increment/clock") for x in
        (donor_radial_increment_m, receiver_radial_increment_m, time_fraction))
    da, ra = (_triple(v, "Actual literal owner axis", F) for v in (donor_owner_axis, receiver_owner_axis))
    if not 0 <= clock <= 1:
        raise ValueError("The actual synchronized clock must lie in [0,1]")
    da, ra = tuple(clock*x for x in da), tuple(clock*x for x in ra)
    if max(sum(map(abs, axis), F(0)) for axis in (da, ra)) > F(1, 2):
        raise ValueError("Owner motion exceeds the actual small-rotation domain")
    alg = IntervalScalarAlgebra(source.precision) if algebra is None else algebra
    c = alg.constant
    rounding = getattr(alg, "round", lambda x: x)
    def mul(a, b):
        return rounding(a*b)
    def add(a, b):
        return rounding(a+b)
    def dot(a, b):
        return rounding(sum((mul(x, y) for x, y in zip(a, b)), c(F(0))))
    def cross(a, b):
        return tuple(rounding(x) for x in _cross(a, b))
    def matvec(matrix, vector):
        return tuple(dot(row, vector) for row in matrix)
    def square(x):
        if type(x) is I:
            return rounding(I(0 if x.lo <= 0 <= x.hi else min(x.lo*x.lo, x.hi*x.hi),
                max(x.lo*x.lo, x.hi*x.hi)))
        return mul(x, x)
    theta, z = c(theta) if type(theta) in (F, I) else theta, c(z) if type(z) in (F, I) else z
    R, db, rb = source.radius_m, source.donor_radial_before_m, source.receiver_radial_before_m
    td, tr = R+db+clock*di, R+rb+clock*ri
    if td <= 0 or tr <= 0:
        raise ValueError("Current bank roofs must remain at positive radii")
    # Fixed source labels: donor z_d=z_r+offset; reference tau never changes
    # when the current radial deformation changes either roof.
    rdcube = c(td**3)-c(3*R*R)*(z+c(source.donor_depth_offset_m))
    rrcube = c(tr**3)-c(3*R*R)*z
    rd, rr = alg.cbrt(rdcube), alg.cbrt(rrcube)
    def cube_increment(before, increment):
        return 3*(R+before)**2*increment+3*(R+before)*increment**2+increment**3
    delta_cube = cube_increment(rb, clock*ri)-cube_increment(db, clock*di)
    delta_r = rounding(c(delta_cube)/(square(rr)+mul(rr, rd)+square(rd)))
    tau = z+c((R**3-(R+rb)**3)/(3*R*R))
    a, b, n = (tuple(c(x) for x in row) for row in
        (source.start_direction_bounds, source.tangent_direction_bounds, source.normal_direction_bounds))
    sine, cosine = alg.sin(theta), alg.cos(theta)
    tangent = tuple(add(-mul(x, sine), mul(y, cosine)) for x, y in zip(a, b))
    A = source.shared_depth_axis_per_m
    identity = tuple(tuple(c(F(i == j)) for j in range(3)) for i in range(3))
    if any(A):
        rho = interval_sqrt(I(sum((x*x for x in A), F(0))), source.precision+8)
        angle = mul(c(rho), tau)
        ss, cc = alg.sin(angle), alg.cos(angle)
        unit = tuple(c(I(x)/rho) for x in A)
        K = ((c(F(0)), -unit[2], unit[1]), (unit[2], c(F(0)), -unit[0]),
            (-unit[1], unit[0], c(F(0))))
        # K is constant, so this product does not create angle/depth jet
        # convolutions.  The source unit-axis aliases remain outward.
        K2 = tuple(tuple(dot(K[i], tuple(K[k][j] for k in range(3)))
            for j in range(3)) for i in range(3))
        shear = tuple(tuple(add(add(identity[i][j], mul(ss, K[i][j])),
            mul(c(F(1))-cc, K2[i][j])) for j in range(3)) for i in range(3))
    else:
        shear = identity
    change, receiver, pullbacks = _owner_matrices(da, ra, source.precision, source.order)
    Qr = tuple(tuple(c(x) for x in row) for row in receiver)
    # Invariants of S^-1 C S can be evaluated after S without constructing
    # nine dense bivariate matrix jets.  S is a proper rotation:
    # n.(S^-1 C S)u=(S n).C(S u), |S^-1 C(S u)|=|C(S u)|.
    Sa, Sb, Sn = matvec(shear, a), matvec(shear, b), matvec(shear, n)
    Su = tuple(add(mul(x, cosine), mul(y, sine)) for x, y in zip(Sa, Sb))
    St = tuple(add(-mul(x, sine), mul(y, cosine)) for x, y in zip(Sa, Sb))
    if da == ra:
        V = (c(F(0)),)*3
    else:
        C = tuple(tuple(c(x) for x in row) for row in change)
        # The two theta harmonics remain correlated through the same small C.
        Va, Vb = matvec(C, Sa), matvec(C, Sb)
        V = tuple(add(mul(x, cosine), mul(y, sine)) for x, y in zip(Va, Vb))
    p = dot(Sn, V)
    q = -sum((square(x) for x in V), c(F(0)))/2
    at = dot(tuple(c(x) for x in A), tangent)
    P, B = rounding(c(R*R)/rr), mul(square(rr), at)
    J = alg.sqrt(square(P)+square(B))
    T = rounding(-mul(B, delta_r)-mul(rd, mul(P, p)-mul(B, q)))
    orientation = c(F(source.normal_orientation))
    gap = (c(source.source_preload_gap_m) if da == ra and not delta_cube else
        add(c(source.source_preload_gap_m), rounding(orientation*T/J)))
    normal_numerator = tuple(add(mul(P, ni), -mul(B, ui)) for ni, ui in zip(Sn, Su))
    # n cross u = tangent and u cross u = 0 are source unit-frame identities,
    # kept exact instead of subtracting independent aliased coordinate products.
    spin_receiver = tuple(add(mul(P, ti), ci) for ti, ci in
        zip(St, cross(normal_numerator, V)))
    spin_world = matvec(Qr, spin_receiver)
    donor_spatial = tuple(rounding(orientation*rd*x/J) for x in spin_world)
    spatial = donor_spatial+tuple(-x for x in donor_spatial)
    chart_spin = tuple(rounding(c(clock)*dot(spatial[3*bank:3*bank+3],
        tuple(c(x) for x in column))) for bank in range(2) for column in pullbacks[bank])
    drd = rounding(c(td*td)/square(rd))
    drr = rounding(c(tr*tr)/square(rr))
    P_r = rounding(-P*drr/rr)
    B_r = rounding(c(F(2))*B*drr/rr)
    J_r = rounding((mul(P, P_r)+mul(B, B_r))/J)
    T_d = rounding(drd*(B*(c(F(1))+q)-P*p))
    T_r = rounding(-B_r*delta_r-B*drr-rd*(P_r*p-B_r*q))
    gap_d = rounding(c(clock)*orientation*T_d/J)
    gap_r = rounding(c(clock)*orientation*(T_r/J-T*J_r/square(J)))
    dg = chart_spin+(gap_d, gap_r)
    dJ = (c(F(0)),)*7+(rounding(c(clock)*J_r),)
    k = c(source.effective_normal_stiffness_pa_m)
    potential = rounding(J*k*square(gap)/2)
    pressure = rounding(-k*gap)
    gradient = tuple(rounding(J*k*gap*d+k*square(gap)*a/2) for d, a in zip(dg, dJ))
    GN = tuple(tuple(rounding(J*k*x*y) for y in dg) for x in dg) if include_gauss_newton else ()
    return CurvedCorrelatedScalars(gap, J, potential, pressure, dg, dJ, gradient, GN, da == ra)


def bound_curved_correlated_cell(source, *, angle_bounds_radians,
        receiver_depth_bounds_m, donor_radial_increment_m, receiver_radial_increment_m,
        donor_owner_axis, receiver_owner_axis, time_fraction=F(1), include_gauss_newton=True):
    """Enclose every point of an actual complete angle x material-depth cell."""
    validate_curved_correlated_source(source)
    angle, depth = (_interval(x, "Curved complete cell") for x in
        (angle_bounds_radians, receiver_depth_bounds_m))
    if (angle.lo < source.angle_bounds_radians.lo or angle.hi > source.angle_bounds_radians.hi
            or depth.lo < source.receiver_depth_bounds_m.lo or depth.hi > source.receiver_depth_bounds_m.hi):
        raise ValueError("A force cell cannot leave its complete authenticated source domain")
    result = evaluate_curved_correlated_scalars(source, angle, depth,
        donor_radial_increment_m=donor_radial_increment_m,
        receiver_radial_increment_m=receiver_radial_increment_m,
        donor_owner_axis=donor_owner_axis, receiver_owner_axis=receiver_owner_axis,
        time_fraction=time_fraction, include_gauss_newton=include_gauss_newton)
    if source.normal_column_lengths_m:
        pressure_upper = -source.effective_normal_stiffness_pa_m*min(result.gap.lo, F(0))
        if any(pressure_upper*C >= L for C, L in
                zip(source.normal_compliances_m_pa[:2], source.normal_column_lengths_m)):
            raise ValueError("The complete curved-face compression can collapse a declared source column")
    if result.gap.hi < 0:
        return result
    clipped = I(min(result.gap.lo, F(0)), min(result.gap.hi, F(0)))
    k, J = source.effective_normal_stiffness_pa_m, result.area
    square = I(0, clipped.lo*clipped.lo) if clipped.hi == 0 else I(clipped.hi**2, clipped.lo**2)
    W = k*square/2
    gradient = tuple(_rounded_interval(J*k*clipped*d+W*a, source.precision)
        for d, a in zip(result.gap_gradient, result.area_gradient))
    active = I(0) if result.gap.lo >= 0 else I(0, 1)
    GN = tuple(tuple(_rounded_interval(active*J*k*x*y, source.precision)
        for y in result.gap_gradient) for x in result.gap_gradient) if include_gauss_newton else ()
    return CurvedCorrelatedScalars(result.gap, J, _rounded_interval(J*W, source.precision),
        -k*clipped, result.gap_gradient, result.area_gradient, gradient, GN,
        result.relative_owner_matrix_exact_zero)
