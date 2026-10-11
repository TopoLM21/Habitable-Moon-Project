"""A conservative reduced normal law on a real curved spherical contact face.

The saved reference lift is X_ref=r_ref exp(A tau_ref)u, with
tau_ref=(R^3-r_ref^3)/(3R^2). Motion composes independent radial deformation
after that fixed reference shear: X(q)=Q r_current exp(A tau_ref)u. Updating
the reference shear during motion would incorrectly move each radial port
along the inclined face and remove its normal stiffness.
The receiving face's *current* area measure multiplies its series-column
potential. Its derivatives therefore include both the changing normal and
the changing measure. This is a new, unaccepted first-birth constitutive
model, not a re-interpretation of the old constant-height force certificate.
The one-dimensional column energy does not certify a three-dimensional
strain energy, spatial gluing, frictional history or material admission.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
from math import factorial

from .geometric_native_pose import FractionInterval as I, interval_sqrt, _rounded_interval
from .geometric_native_spherical_solid import interval_cbrt
from .geometric_native_physical_pair_time import _rotated, _rotation_change, _relative_change


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), I(0))


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _square(x):
    return I(0 if x.lo <= 0 <= x.hi else min(x.lo*x.lo, x.hi*x.hi), max(x.lo*x.lo, x.hi*x.hi))


def _exact(value, name):
    if type(value) is not F:
        raise ValueError(name+" requires a literal Fraction")
    return value


def _bounds(value, name):
    if type(value) is F:
        return I(value)
    if type(value) is not I:
        raise ValueError(name+" requires a literal Fraction or FractionInterval")
    return value


def _triple(value, name, kind):
    if type(value) is not tuple or len(value) != 3 or any(type(x) is not kind for x in value):
        raise ValueError(name+" requires an immutable exact Cartesian triple")
    return value


def _signed_rotation(vector, axis, time, precision, order, *, change=False):
    """Use the existing outward Taylor evaluator on both signed halves."""
    fn = _rotation_change if change else _rotated
    pieces = []
    if time.hi >= 0:
        pieces.append(fn(vector, axis, I(max(F(0), time.lo), time.hi), precision, order))
    if time.lo < 0:
        pieces.append(fn(vector, tuple(-x for x in axis), I(max(F(0), -time.hi), -time.lo), precision, order))
    return tuple(I(min(row[k].lo for row in pieces), max(row[k].hi for row in pieces)) for k in range(3))


@dataclass(frozen=True, slots=True)
class CurvedFaceGeometry:
    radius_m: F
    top_displacement_m: I
    reference_top_displacement_m: F
    material_depth_bounds_m: I
    direction_bounds: tuple
    tangent_bounds: tuple
    normal_bounds: tuple
    shared_depth_axis_per_m: tuple
    owner_axis: tuple
    normal_orientation: int
    radial_bounds_m: I
    physical_volume_depth_bounds_m: I
    reference_physical_volume_depth_bounds_m: I
    world_position_bounds_m: tuple
    oriented_area_vector_bounds: tuple
    area_density_bounds_m2_per_radian_per_material_m: I
    unit_normal_bounds: tuple
    radial_position_derivative_bounds: tuple
    radial_normal_derivative_bounds: tuple
    radial_area_derivative_bounds: I
    precision: int
    order: int


def bound_curved_face_geometry(*, radius_m, top_displacement_m, reference_top_displacement_m,
        material_depth_bounds_m, direction_bounds, tangent_bounds, normal_bounds,
        shared_depth_axis_per_m, owner_axis=(F(0),)*3, normal_orientation=1,
        precision=128, order=28):
    """Enclose the true cross-product normal, area and top-w derivatives.

    The input oriented frame encloses u,t,n=u cross t; native wrappers
    authenticate it against a complete source carrier. Material depth is
    the bank's volume coordinate, not its Euclidean shell height. The
    saved angular shear uses tau_ref=z+(R^3-(R+w_before)^3)/(3R^2);
    current radial deformation follows this fixed material reference map.
    """
    R = _exact(radius_m, "Reference radius")
    wr = _exact(reference_top_displacement_m, "Fixed reference top displacement")
    w, z = _bounds(top_displacement_m, "Top displacement"), _bounds(material_depth_bounds_m, "Material depth")
    u, t, n = (_triple(v, "Oriented unit frame", I) for v in (direction_bounds, tangent_bounds, normal_bounds))
    A, owner = (_triple(v, "Literal rotation axis", F) for v in (shared_depth_axis_per_m, owner_axis))
    if R <= 0 or z.lo < 0 or R+w.lo <= 0 or R+wr <= 0:
        raise ValueError("Curved faces require positive radii and nonnegative material depth")
    if type(normal_orientation) is not int or normal_orientation not in (-1, 1):
        raise ValueError("The physical receiving orientation must be explicitly +1 or -1")
    if type(precision) is not int or not 64 <= precision <= 512 or type(order) is not int or not 8 <= order <= 64:
        raise ValueError("Curved face precision/order lies outside the declared Taylor domain")
    if any(not _dot(v, v).lo <= 1 <= _dot(v, v).hi for v in (u, t, n)):
        raise ValueError("Every oriented frame enclosure must contain a unit vector")
    if any(not _dot(a, b).lo <= 0 <= _dot(a, b).hi for a, b in ((u, t), (u, n), (t, n))):
        raise ValueError("The declared face frame must enclose orthogonality")
    if any(a.hi < b.lo or b.hi < a.lo for a, b in zip(_cross(u, t), n)):
        raise ValueError("The declared carrier normal must enclose u cross t")
    top = I(R)+w
    cubes = top*top*top-3*R*R*z
    if cubes.lo <= 0:
        raise ValueError("A curved face cannot reach the sphere centre")
    r = interval_cbrt(cubes, precision)
    tau = (I(R**3)-cubes)/(3*R*R)
    reference_tau = z+(R**3-(R+wr)**3)/(3*R*R)
    if (R+wr)**3-3*R*R*z.hi <= 0:
        raise ValueError("The fixed reference face cannot reach the sphere centre")
    if sum(map(abs, A), F(0))*max(abs(reference_tau.lo), abs(reference_tau.hi)) > F(1, 2) or sum(map(abs, owner), F(0)) > F(1, 2):
        raise ValueError("Curved face rotation exceeds the declared small-increment domain")
    at = _dot(A, t)
    J = interval_sqrt(_square(R*R/r)+_square(r*r*at), precision)
    if J.lo <= 0:
        raise ValueError("A curved face must have a positive-rank area measure")
    def transform(v):
        return _rotated(_signed_rotation(v, A, reference_tau, precision, order), owner, I(1), precision, order)
    U, plane_normal = transform(u), transform(n)
    P = tuple(r*x for x in U)
    N = tuple(normal_orientation*(R*R/r*ni-r*r*at*ui) for ni, ui in zip(plane_normal, U))
    unit = tuple(x/J for x in N)
    dr = top*top/(r*r)
    Pw = tuple(dr*x for x in U)
    Nw = tuple(normal_orientation*dr*(-R*R/(r*r)*ni-2*r*at*ui) for ni, ui in zip(plane_normal, U))
    Jw = (-R**4/(r*r*r)+2*r*r*r*_square(at))*dr/J
    nw = tuple((x-y*Jw)/J for x, y in zip(Nw, unit))
    return CurvedFaceGeometry(R, w, wr, z, u, t, n, A, owner, normal_orientation, r, tau, reference_tau,
        P, N, J, unit, Pw, nw, Jw, precision, order)


def validate_curved_face_geometry(geometry):
    if type(geometry) is not CurvedFaceGeometry:
        raise ValueError("A complete declared curved face geometry is required")
    replay = bound_curved_face_geometry(radius_m=geometry.radius_m,
        top_displacement_m=geometry.top_displacement_m,
        reference_top_displacement_m=geometry.reference_top_displacement_m,
        material_depth_bounds_m=geometry.material_depth_bounds_m,
        direction_bounds=geometry.direction_bounds, tangent_bounds=geometry.tangent_bounds,
        normal_bounds=geometry.normal_bounds, shared_depth_axis_per_m=geometry.shared_depth_axis_per_m,
        owner_axis=geometry.owner_axis, normal_orientation=geometry.normal_orientation,
        precision=geometry.precision, order=geometry.order)
    if replay != geometry:
        raise ValueError("Curved area/normal differs from the complete declared geometry replay")
    return replay


@dataclass(frozen=True, slots=True)
class CurvedContactPointResponse:
    free_gap_bounds_m: I
    geometric_relative_displacement_bounds_m: tuple
    source_preload_gap_m: F
    activation_reference_projection_bounds_m: I
    normal_pressure_bounds_pa: I
    effective_normal_stiffness_pa_m: F
    normal_compliances_m_pa: tuple
    column_shortening_bounds_m: tuple
    fault_indentation_bounds_m: I
    energy_partition_bounds_j_m2: tuple
    potential_density_bounds_j_per_radian_per_material_m: I
    gap_gradient_bounds: tuple
    area_gradient_bounds: tuple
    potential_gradient_bounds: tuple
    coordinate_names: tuple = ("donor_spatial_spin_x", "donor_spatial_spin_y", "donor_spatial_spin_z",
        "receiver_spatial_spin_x", "receiver_spatial_spin_y", "receiver_spatial_spin_z", "donor_top_w_m", "receiver_top_w_m")

    @property
    def current_curved_area_and_normal_derivatives_included(self):
        return True

    @property
    def reduced_series_energy_partition_certified(self):
        # The exact compliances sum to 1/k; interval copies need not have
        # zero width for this algebraic identity to hold pointwise.
        return True

    @property
    def complete_three_dimensional_strain_energy_certified(self):
        return False

    @property
    def native_force_equilibrium_certified(self):
        return False

    @property
    def physical_material_admission_authorized(self):
        return False


def bound_curved_contact_point(donor, receiver, donor_reference, receiver_reference, *,
        columns, law, source_preload_gap_m, precision=128):
    """Conservative current-area normal potential with a fixed activation rest.

    Source preload is retained at the activation configuration. The spatial
    rest offset is n_ref dot (X_ref_receiver-X_ref_donor), never a silently
    rebased scalar trial gap. Derivatives are current spatial spin covectors;
    exponential-coordinate derivatives require the left-Jacobian pullback.
    """
    for geometry in (donor, receiver, donor_reference, receiver_reference):
        validate_curved_face_geometry(geometry)
    labels = ("radius_m", "reference_top_displacement_m", "material_depth_bounds_m", "direction_bounds", "tangent_bounds",
        "normal_bounds", "shared_depth_axis_per_m", "normal_orientation")
    if any(any(getattr(current, name) != getattr(reference, name) for name in labels)
            for current, reference in ((donor, donor_reference), (receiver, receiver_reference))):
        raise ValueError("A curved activation reference cannot change its material labels or depth flow")
    if donor.radius_m != receiver.radius_m or donor.shared_depth_axis_per_m != receiver.shared_depth_axis_per_m:
        raise ValueError("Both banks require the same reference radius and common depth flow")
    relative = tuple(b-a for a, b in zip(donor.world_position_bounds_m, receiver.world_position_bounds_m))
    reference = tuple(b-a for a, b in zip(donor_reference.world_position_bounds_m, receiver_reference.world_position_bounds_m))
    projection = _dot(receiver_reference.unit_normal_bounds, reference)
    return _point_response(donor, receiver, relative, projection, columns, law,
        source_preload_gap_m, precision)


def _point_response(donor, receiver, relative, reference_projection, columns, law, preload, precision):
    from .geometric_solid_contact_closure import _parameters
    _parameters(columns, law)
    _exact(preload, "Retained source preload")
    if law.normal_stiffness_pa_m <= 0:
        raise ValueError("A curved contact requires a positive source normal stiffness")
    C = (F.from_float(columns.sub_length_m)/F.from_float(columns.sub_normal_modulus_pa),
         F.from_float(columns.over_length_m)/F.from_float(columns.over_normal_modulus_pa),
         1/F.from_float(law.normal_stiffness_pa_m))
    k = 1/sum(C, F(0))
    n, J = receiver.unit_normal_bounds, receiver.area_density_bounds_m2_per_radian_per_material_m
    gap = _rounded_interval(I(preload)+_dot(n, relative)-reference_projection, precision)
    clipped = I(min(gap.lo, F(0)), min(gap.hi, F(0)))
    pressure = -k*clipped
    shortening = tuple(pressure*x for x in C)
    if shortening[0].hi >= F.from_float(columns.sub_length_m) or shortening[1].hi >= F.from_float(columns.over_length_m):
        raise ValueError("The complete curved-face compression can collapse a declared column")
    partition = tuple(_square(pressure)*x/2 for x in C)
    W = k*_square(clipped)/2
    axes = tuple(tuple(F(i == j) for i in range(3)) for j in range(3))
    dg_d = tuple(-_dot(n, _cross(axis, donor.world_position_bounds_m)) for axis in axes)
    # The two receiver-spin terms combine algebraically to -dg_d. This
    # retains exact objectivity instead of subtracting two 100km intervals.
    dg_r = tuple(-x for x in dg_d)
    dg = dg_d+dg_r+(-_dot(n, donor.radial_position_derivative_bounds),
        _dot(receiver.radial_normal_derivative_bounds, relative)+_dot(n, receiver.radial_position_derivative_bounds))
    dJ = (I(0),)*7+(receiver.radial_area_derivative_bounds,)
    gradient = tuple(_rounded_interval(J*k*clipped*d+W*a, precision) for d, a in zip(dg, dJ))
    return CurvedContactPointResponse(gap, relative, preload, reference_projection, pressure, k, C,
        shortening[:2], shortening[2], partition, _rounded_interval(J*W, precision), dg, dJ, gradient)


def _left_jacobian_columns(axis, precision, order):
    rate = sum(map(abs, axis), F(0))
    tail = rate**(order+1)/factorial(order+2)/(1-rate/F(order+3))
    columns = []
    for j in range(3):
        term = tuple(I(F(i == j)) for i in range(3))
        total = term
        for degree in range(1, order+1):
            term = tuple(_rounded_interval(x/F(degree+1), precision) for x in _cross(axis, term))
            total = tuple(_rounded_interval(a+b, precision) for a, b in zip(total, term))
        columns.append(tuple(x+I(-tail, tail) for x in total))
    return tuple(columns)


@dataclass(frozen=True, slots=True)
class NativeCurvedContactResponse:
    family_id: str
    source_support_sha256: str
    source_epoch_sha256: str
    source_preload_gap_m: F
    reference_receiver_material_depth_bounds_m: I
    reference_donor_material_depth_bounds_m: I
    angle_bounds_radians: I
    time_fraction_bounds: I
    potential_bounds_j: I
    activation_potential_bounds_j: I
    conservative_endpoint_potential_change_bounds_j: I
    integrated_pressure_bounds_n: I
    energy_partition_bounds_j: tuple
    spatial_coordinate_gradient_bounds: tuple
    literal_axis_and_radial_increment_gradient_bounds: tuple
    free_gap_bounds_m: I
    integration_cell_count: int
    source_force_certificate_reused: bool = False
    activation_history_adopted: bool = False
    complete_three_dimensional_strain_energy_certified: bool = False
    physical_material_admission_authorized: bool = False

    @property
    def whole_time_interval_force_and_collision_certified(self):
        return False

    @property
    def native_force_equilibrium_certified(self):
        return False

    @property
    def reduced_series_energy_partition_certified(self):
        return True

    @property
    def shared_physical_reference_face_coincident(self):
        return True

    @property
    def source_preload_retained_at_activation(self):
        return True

    @property
    def conservative_endpoint_change_definition(self):
        return "difference of the same current-area potential at the two endpoints; no independently integrated path-work certificate"


def bound_native_joint_curved_contact_response(field, family_id, *, donor_radial_before_m,
        receiver_radial_before_m, donor_radial_increment_m, receiver_radial_increment_m,
        donor_owner_axis, receiver_owner_axis, time_fraction_bounds=I(1),
        angle_cells=4, depth_cells=4, precision=128, order=28):
    """Integrate complete native angle x shared *physical reference* depth.

    Cells are full interval rectangles, not quadrature samples. No pressure,
    depth or area cutoff is applied. The new activation energy is evaluated
    from the retained source preload on the same complete physical face.
    """
    from .geometric_native_joint_spherical_field import validate_native_joint_spherical_field, joint_spherical_component
    from .geometric_native_birth_continuum_motion import native_birth_continuum_frame
    from .geometric_native_active_contact_line_work import _trig_ranges_signed
    field = validate_native_joint_spherical_field(field)
    rows = tuple(c for c in field.support.interface_carriers if c.declaration.family_id == family_id)
    if type(family_id) is not str or len(rows) != 1:
        raise ValueError("Curved mechanics requires an actual unique complete native carrier")
    carrier = rows[0]
    db, rb, di, ri = (_exact(x, "Authoritative bank displacement") for x in
        (donor_radial_before_m, receiver_radial_before_m, donor_radial_increment_m, receiver_radial_increment_m))
    da, ra = (_triple(v, "Literal owner axis", F) for v in (donor_owner_axis, receiver_owner_axis))
    if type(time_fraction_bounds) is not I or not 0 <= time_fraction_bounds.lo <= time_fraction_bounds.hi <= 1:
        raise ValueError("Curved mechanics requires the actual synchronized [0,1] clock")
    if any(type(x) is not int or not 1 <= x <= 32 for x in (angle_cells, depth_cells)):
        raise ValueError("Complete rectangle counts must be literal integers in [1,32]")
    donor = joint_spherical_component(field, carrier.donor_component.component_id)
    receiver = joint_spherical_component(field, carrier.receiver_component.component_id)
    if donor.layer.top_radial_displacement_m != db or receiver.layer.top_radial_displacement_m != rb:
        raise ValueError("The curved activation geometry must match the field's authoritative before banks")
    A = donor.angular_rate_per_physical_volume_depth_m
    if A != receiver.angular_rate_per_physical_volume_depth_m:
        raise ValueError("The two physical contact banks require one common depth flow")
    R = donor.layer.radius_m
    td, tr = ((R**3-(R+w)**3)/(3*R*R) for w in (db, rb))
    offset = tr-td
    low, high = max(F(0), -offset), min(receiver.layer.areal_thickness_m, donor.layer.areal_thickness_m-offset)
    if high <= low:
        raise ValueError("The source banks have no positive common physical reference face")
    a, b, inward = native_birth_continuum_frame(carrier, precision=max(128, precision))
    normal = _cross(a, b)
    orientation_bounds = _dot(normal, inward)
    orientation = 1 if orientation_bounds.lo > 0 else -1 if orientation_bounds.hi < 0 else 0
    if not orientation:
        raise ValueError("The actual receiving normal orientation is unresolved")
    preload = F.from_float(carrier.declaration.birth_gap_free_m)
    clock = time_fraction_bounds
    current_axes = tuple(tuple(x*clock.midpoint for x in axis) for axis in (da, ra))
    if clock.width:
        # A complete time enclosure must use interval rotation time, not a
        # midpoint pose. Geometry accepts only literal owner-axis declarations.
        raise ValueError("Whole-clock force integration requires a separate interval owner-path pullback")
    wc_d, wc_r = I(db)+clock*di, I(rb)+clock*ri
    def cube_increment(w, increment):
        return 3*(R+w)**2*increment+3*(R+w)*increment*increment+increment*increment*increment
    delta_cubes = cube_increment(rb, clock*ri)-cube_increment(db, clock*di)
    total, activation, pressure = I(0), I(0), I(0)
    parts, gradients = [I(0)]*3, [I(0)]*8
    gaps = []
    end = carrier.native_angle_bounds_radians
    intervals = [(F(0), end.lo, False)]
    if end.hi > end.lo:
        intervals.append((end.lo, end.hi, True))
    count = 0
    for alo, ahi, uncertain_endpoint in intervals:
        divisions = 1 if uncertain_endpoint else angle_cells
        for ia in range(divisions):
            theta = I(alo+(ahi-alo)*ia/divisions, alo+(ahi-alo)*(ia+1)/divisions)
            sine, cosine = _trig_ranges_signed(theta.lo, theta.hi, precision)
            u = tuple(x*cosine+y*sine for x, y in zip(a, b))
            t = tuple(-x*sine+y*cosine for x, y in zip(a, b))
            for iz in range(depth_cells):
                z = I(low+(high-low)*iz/depth_cells, low+(high-low)*(iz+1)/depth_cells)
                common = dict(radius_m=R, direction_bounds=u, tangent_bounds=t, normal_bounds=normal,
                    shared_depth_axis_per_m=A, normal_orientation=orientation, precision=precision, order=order)
                gd = bound_curved_face_geometry(**common, top_displacement_m=wc_d,
                    reference_top_displacement_m=db, material_depth_bounds_m=z+offset, owner_axis=current_axes[0])
                gr = bound_curved_face_geometry(**common, top_displacement_m=wc_r,
                    reference_top_displacement_m=rb, material_depth_bounds_m=z, owner_axis=current_axes[1])
                reference_radius = interval_cbrt(I((R+rb)**3)-3*R*R*z, precision)
                reference_area = interval_sqrt(_square(R*R/reference_radius)+
                    _square(reference_radius*reference_radius*_dot(A, t)), precision)
                # Shared reference radius/flow cancels symbolically. Keep the
                # tiny current difference separate from the 100km position.
                radius_delta = delta_cubes/(gr.radial_bounds_m*gr.radial_bounds_m+
                    gr.radial_bounds_m*gd.radial_bounds_m+gd.radial_bounds_m*gd.radial_bounds_m)
                base = _signed_rotation(u, A, gr.reference_physical_volume_depth_bounds_m, precision, order)
                relative_owner = _relative_change(base, current_axes[0], current_axes[1], I(1), precision, order)
                local_delta = tuple(radius_delta*x-gd.radial_bounds_m*q for x, q in zip(base, relative_owner))
                relative = _rotated(local_delta, current_axes[1], I(1), precision, order)
                result = _point_response(gd, gr, relative, I(0), carrier.declaration.parameters.columns,
                    carrier.declaration.parameters.law, preload, precision)
                weight = theta.width*z.width
                if uncertain_endpoint:
                    weight = I(0, weight)
                J = gr.area_density_bounds_m2_per_radian_per_material_m
                total = _rounded_interval(total+result.potential_density_bounds_j_per_radian_per_material_m*weight, precision)
                activation = _rounded_interval(activation+reference_area*
                    result.effective_normal_stiffness_pa_m*min(preload, F(0))**2/2*weight, precision)
                pressure = _rounded_interval(pressure+result.normal_pressure_bounds_pa*J*weight, precision)
                parts = [_rounded_interval(old+new*J*weight, precision) for old, new in zip(parts, result.energy_partition_bounds_j_m2)]
                gradients = [_rounded_interval(old+new*weight, precision) for old, new in zip(gradients, result.potential_gradient_bounds)]
                gaps.append(result.free_gap_bounds_m)
                count += 1
    # Simultaneous owner spin is exactly a rigid objectivity direction.
    gradients[3:6] = [-x for x in gradients[:3]]
    pullbacks = tuple(_left_jacobian_columns(axis, precision, order) for axis in current_axes)
    chart = tuple(clock*_dot(tuple(gradients[3*bank+k] for k in range(3)), column)
        for bank in range(2) for column in pullbacks[bank])+tuple(clock*g for g in gradients[6:])
    return NativeCurvedContactResponse(family_id, field.support.support_sha256,
        field.support.source_epoch.epoch_sha256, preload, I(low, high), I(low+offset, high+offset),
        I(0, end.hi), clock, total, activation, total-activation, pressure, tuple(parts), tuple(gradients),
        chart, I(min(x.lo for x in gaps), max(x.hi for x in gaps)), count)
