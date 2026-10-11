"""One source-bound spherical depth flow for each contact-connected group.

The common coordinate is physical volume-radius depth,
``tau=(R**3-r**3)/(3*R**2)``.  Unlike a bank's material depth it is identical
on both sides of a shared physical-radius face.  A group's single rotation
``exp(A*tau)`` therefore glues the reference face even when bank roofs differ.
This is an explicit reference deformation, not a constitutive energy law or
an admission of independently rotated, compressed or colliding bodies.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
import math

from .geometric_native_pose import FractionInterval as I, _rounded_interval, interval_sqrt
from .geometric_native_physical_pair_time import _rotated
from .geometric_native_spherical_solid import (
    NativeSphericalComponentSolid, SphericalLayerMap, _arc_frame, _cross, _dot, _fraction,
    _precision, _vector, bind_native_spherical_component_solid, interval_cbrt,
)


@dataclass(frozen=True, slots=True)
class JointSphericalLayerMap:
    """Pure declared map for independent geometric checks; no mask authority."""
    radius_m: F
    top_radial_displacement_m: F
    areal_thickness_m: F
    angular_rate_per_physical_volume_depth_m: tuple
    provenance: str

    def __post_init__(self):
        self.layer
        _vector(self.angular_rate_per_physical_volume_depth_m, "Shared physical-depth angular rate")
        bounds = (self.top_physical_volume_depth_m,
                  self.top_physical_volume_depth_m+self.areal_thickness_m)
        if max(value*value for value in bounds)*sum((value*value for value in
                self.angular_rate_per_physical_volume_depth_m), F(0)) > F(1, 16):
            raise ValueError("The shared physical-depth flow exceeds the source small-rotation domain")

    @property
    def layer(self):
        return SphericalLayerMap(self.radius_m, self.top_radial_displacement_m,
                                 self.areal_thickness_m, provenance=self.provenance)

    @property
    def top_physical_volume_depth_m(self):
        return (self.radius_m**3-self.layer.top_radius_m**3)/(3*self.radius_m**2)

    @property
    def physical_volume_jacobian(self):
        return self.radius_m**2

    @property
    def signed_volume_jacobian(self):
        return -self.radius_m**2

    @property
    def physical_material_admission_authorized(self):
        return False


def solve_exact_minimum_norm_angular_rate(constraints):
    """Solve ``row dot A = rhs`` exactly, refusing inconsistent declarations.

    Independent rows are retained by exact elimination.  The unique minimum
    Euclidean-norm solution is ``M.T (M M.T)^-1 rhs``; there is no tolerance,
    least-squares averaging or silently discarded incompatible carrier.
    """
    if type(constraints) is not tuple:
        raise ValueError("Angular constraints require an immutable literal tuple")
    basis = []
    independent = []
    for constraint in constraints:
        if type(constraint) is not tuple or len(constraint) != 2:
            raise ValueError("An angular constraint requires its exact row and target")
        row, rhs = constraint
        _vector(row, "Angular constraint row")
        _fraction(rhs, "Angular constraint target")
        reduced = list(row)+[rhs]
        for pivot, previous in basis:
            coefficient = reduced[pivot]
            reduced = [a-coefficient*b for a, b in zip(reduced, previous)]
        nonzero = tuple(index for index, value in enumerate(reduced[:3]) if value)
        if not nonzero:
            if reduced[3]:
                raise ValueError("Connected native carrier angular constraints are inconsistent")
            continue
        pivot = nonzero[0]
        coefficient = reduced[pivot]
        basis.append((pivot, tuple(value/coefficient for value in reduced)))
        independent.append((row, rhs))
    if not independent:
        return (F(0), F(0), F(0))
    count = len(independent)
    gram = [[sum((a*b for a, b in zip(first[0], second[0])), F(0))
             for second in independent]+[first[1]] for first in independent]
    for pivot in range(count):
        source = next((index for index in range(pivot, count) if gram[index][pivot]), None)
        if source is None:
            raise ValueError("Exact independent angular rows unexpectedly lost rank")
        gram[pivot], gram[source] = gram[source], gram[pivot]
        scale = gram[pivot][pivot]
        gram[pivot] = [value/scale for value in gram[pivot]]
        for index in range(count):
            if index != pivot:
                coefficient = gram[index][pivot]
                gram[index] = [a-coefficient*b for a, b in zip(gram[index], gram[pivot])]
    answer = tuple(sum((independent[index][0][coordinate]*gram[index][-1]
                        for index in range(count)), F(0)) for coordinate in range(3))
    if any(sum((a*b for a, b in zip(row, answer)), F(0)) != rhs for row, rhs in constraints):
        raise ValueError("Exact shared angular solution does not satisfy every native carrier")
    return answer


@dataclass(frozen=True, slots=True)
class NativeJointAngularConstraint:
    family_id: str
    donor_component_id: str
    receiver_component_id: str
    receiving_tangent_bounds: tuple
    receiving_tangent_midpoint: tuple
    source_cot_dip: F
    target_projection_per_m: F
    literal_projection_per_m: F
    geometric_projection_error_bounds_per_m: I


@dataclass(frozen=True, slots=True)
class NativeJointSphericalGroup:
    component_ids: tuple
    family_ids: tuple
    angular_rate_per_physical_volume_depth_m: tuple
    exact_midpoint_constraints_satisfied: bool


@dataclass(frozen=True, slots=True)
class NativeJointSphericalComponent:
    source_solid: NativeSphericalComponentSolid
    angular_rate_per_physical_volume_depth_m: tuple

    @property
    def component_id(self):
        return self.source_solid.component.component_id

    @property
    def layer(self):
        return self.source_solid.layer

    @property
    def joint_layer(self):
        return JointSphericalLayerMap(self.layer.radius_m, self.layer.top_radial_displacement_m,
            self.layer.areal_thickness_m, self.angular_rate_per_physical_volume_depth_m,
            self.layer.provenance)

    @property
    def top_physical_volume_depth_m(self):
        return (self.layer.radius_m**3-self.layer.top_radius_m**3)/(3*self.layer.radius_m**2)

    @property
    def top_angular_shift(self):
        return tuple(value*self.top_physical_volume_depth_m
                     for value in self.angular_rate_per_physical_volume_depth_m)

    @property
    def physical_volume_jacobian(self):
        return self.layer.radius_m**2

    @property
    def signed_volume_jacobian(self):
        # Positive outward-oriented solid-angle coordinates, inward material z.
        return -self.layer.radius_m**2

    @property
    def individual_complete_mask_embedding_certified(self):
        return True


@dataclass(frozen=True, slots=True)
class NativeJointSphericalField:
    support: object
    components: tuple
    groups: tuple
    constraints: tuple
    continuity_graph_edges: tuple
    top_radial_displacements_m: tuple
    source_epoch_fingerprint_sha256: str
    source_support_sha256: str
    provenance: str
    declaration_precision: int

    @property
    def one_depth_flow_per_contact_connected_group(self):
        return True

    @property
    def all_actual_implicit_material_masks_retained(self):
        return True

    @property
    def old_surface_classification_inherited(self):
        return False

    @property
    def independently_rotated_bank_faces_glued(self):
        return False

    @property
    def normal_column_deformation_and_strain_energy_glued(self):
        return False

    @property
    def whole_partition_noninterpenetration_certified(self):
        return False

    @property
    def physical_material_admission_authorized(self):
        return False


def bind_native_joint_spherical_field(support, *, top_radial_displacements_m,
        provenance, precision=128):
    """Bind all authentic masks to one exact shared flow per contact graph.

    Existing material trace bindings and unresolved positive fronts connect
    the graph, while providing no invented normal/dip constraint.  Isolated masks receive an explicit zero
    depth rate.  Constant roof displacements are declarations supplied by the
    caller, not an inferred bending field.  Disposable proxy charts remain
    membership containers and are never assigned as material reservoirs.
    """
    from .geometric_native_birth_contact_support import native_birth_contact_support_to_dict
    from .geometric_native_implicit_epoch_admission import _portable_fingerprint_sha256
    from .geometric_native_polygons import _plane_values
    _precision(precision)
    native_birth_contact_support_to_dict(support)
    if type(provenance) is not str or not provenance.strip():
        raise ValueError("A joint spherical field requires explicit physical model provenance")
    ids = tuple(sorted(row.component_id for row in support.source_epoch.components))
    if type(top_radial_displacements_m) is not tuple or any(
            type(row) is not tuple or len(row) != 2 or type(row[0]) is not str
            or type(row[1]) is not F for row in top_radial_displacements_m):
        raise ValueError("Every source mask requires one immutable literal roof declaration")
    if tuple(row[0] for row in top_radial_displacements_m) != ids:
        raise ValueError("Joint spherical roof declarations must cover every source mask exactly once in canonical order")
    parents = {component_id: component_id for component_id in ids}

    def root(component_id):
        while parents[component_id] != component_id:
            component_id = parents[component_id]
        return component_id

    def connect(first, second):
        if first not in parents or second not in parents:
            raise ValueError("The actual contact graph references a missing source mask")
        a, b = sorted((root(first), root(second)))
        parents[b] = a

    edges = []
    live_by_parent = {row.original_parent_id: row.component_id for row in support.source_epoch.components
                      if row.material_kind == "original_live"}
    if len(live_by_parent) != sum(row.material_kind == "original_live" for row in support.source_epoch.components):
        raise ValueError("An accepted material trace has ambiguous split live masks in the contact graph")
    for binding in support.source_epoch.source_material_state.trace_bindings:
        if binding.donor_fragment_id not in live_by_parent or binding.receiver_fragment_id not in live_by_parent:
            raise ValueError("An accepted material trace has no complete live mask in the contact graph")
        first, second = live_by_parent[binding.donor_fragment_id], live_by_parent[binding.receiver_fragment_id]
        connect(first, second)
        edges.append(("accepted_material_trace", binding.trace_id, first, second))
    for carrier in support.interface_carriers:
        first, second = carrier.donor_component.component_id, carrier.receiver_component.component_id
        connect(first, second)
        edges.append(("declared_positive_carrier", carrier.declaration.family_id, first, second))
    for index, front in enumerate(support.unresolved_fronts):
        connect(front.donor_component_id, front.receiver_component_id)
        edges.append(("unresolved_front", str(index), front.donor_component_id, front.receiver_component_id))
    partition = {}
    for component_id in ids:
        partition.setdefault(root(component_id), []).append(component_id)
    radius = F.from_float(support.source_epoch.radius_km)*1000
    raw = []
    for carrier in support.interface_carriers:
        start, _, _, _ = _arc_frame(carrier.edge, precision)
        inward = _plane_values(carrier.receiver_inward_plane, precision)[1]
        length = interval_sqrt(_dot(inward, inward), precision+8)
        receiving_tangent = _cross(tuple(value/length for value in inward), start)
        midpoint = tuple(value.midpoint for value in receiving_tangent)
        cot = F.from_float(math.cos(carrier.declaration.dip_radians))/F.from_float(
            math.sin(carrier.declaration.dip_radians))
        raw.append((carrier, receiving_tangent, midpoint, cot, -cot/radius))
    groups, rates = [], {}
    for component_ids in sorted(tuple(value) for value in partition.values()):
        selected = tuple(row for row in raw if row[0].donor_component.component_id in component_ids)
        rate = solve_exact_minimum_norm_angular_rate(tuple((row[2], row[4]) for row in selected))
        groups.append(NativeJointSphericalGroup(component_ids,
            tuple(row[0].declaration.family_id for row in selected), rate, True))
        rates.update((component_id, rate) for component_id in component_ids)
    constraints = tuple(NativeJointAngularConstraint(carrier.declaration.family_id,
        carrier.donor_component.component_id, carrier.receiver_component.component_id,
        tangent, midpoint, cot, target,
        sum((a*b for a, b in zip(midpoint, rates[carrier.donor_component.component_id])), F(0)),
        _dot(tangent, rates[carrier.donor_component.component_id])-I(target))
        for carrier, tangent, midpoint, cot, target in raw)
    components = tuple(NativeJointSphericalComponent(bind_native_spherical_component_solid(
        support.source_epoch, component_id, top_radial_displacement_m=displacement,
        provenance=provenance), rates[component_id])
        for component_id, displacement in top_radial_displacements_m)
    # Keep the explicit small source domain on the whole physical-volume depth.
    for component in components:
        extremes = (component.top_physical_volume_depth_m,
                    component.top_physical_volume_depth_m+component.layer.areal_thickness_m)
        rate_norm_squared = sum((value*value for value in component.angular_rate_per_physical_volume_depth_m), F(0))
        if max(value*value for value in extremes)*rate_norm_squared > F(1, 16):
            raise ValueError("The shared physical-depth flow exceeds the source small-rotation domain")
    return NativeJointSphericalField(support, components, tuple(groups), constraints, tuple(edges),
        top_radial_displacements_m, _portable_fingerprint_sha256(support.source_epoch),
        support.support_sha256, provenance, precision)


def validate_native_joint_spherical_field(field):
    from .geometric_native_physical_history import _typed_fingerprint
    if type(field) is not NativeJointSphericalField:
        raise ValueError("A complete authenticated joint spherical field is required")
    replay = bind_native_joint_spherical_field(field.support,
        top_radial_displacements_m=field.top_radial_displacements_m,
        provenance=field.provenance, precision=field.declaration_precision)
    if _typed_fingerprint(field) != _typed_fingerprint(replay):
        raise ValueError("Joint spherical field differs from its complete source/declaration replay")
    return replay


def joint_spherical_component(field, component_id):
    if type(field) is not NativeJointSphericalField:
        raise ValueError("A complete native joint spherical field is required")
    matches = tuple(row for row in field.components if row.component_id == component_id)
    if type(component_id) is not str or len(matches) != 1:
        raise ValueError("The joint field has no unique actual source component")
    return matches[0]


def joint_component_radius_cubed_bounds(component, material_depth_bounds_m):
    if type(component) is not NativeJointSphericalComponent or type(material_depth_bounds_m) is not I:
        raise ValueError("Joint solid placement requires an actual component and depth interval")
    depth = material_depth_bounds_m
    if not 0 <= depth.lo <= depth.hi <= component.layer.areal_thickness_m:
        raise ValueError("Joint solid depth is outside the complete source layer")
    return I(component.layer.top_radius_m**3)-3*component.layer.radius_m**2*depth


def bound_joint_radius_point(component, direction_bounds, physical_radius_cubed_bounds_m3,
        *, precision=128, order=28):
    """Enclose a common-radius image; direction membership remains separate."""
    if type(component) is not NativeJointSphericalComponent:
        raise ValueError("A source-bound joint component is required")
    return bound_joint_layer_radius_point(component.joint_layer, direction_bounds,
        physical_radius_cubed_bounds_m3, precision=precision, order=order)


def bound_joint_layer_point(joint_layer, direction_bounds, material_depth_bounds_m,
        *, precision=128, order=28):
    """Pure complete-depth geometric enclosure, without material membership."""
    if type(joint_layer) is not JointSphericalLayerMap or type(material_depth_bounds_m) is not I:
        raise ValueError("A declared joint layer and literal material-depth interval are required")
    depth = material_depth_bounds_m
    if not 0 <= depth.lo <= depth.hi <= joint_layer.areal_thickness_m:
        raise ValueError("Joint solid depth is outside the complete declared layer")
    cubes = I(joint_layer.layer.top_radius_m**3)-3*joint_layer.radius_m**2*depth
    return bound_joint_layer_radius_point(joint_layer, direction_bounds, cubes,
        precision=precision, order=order)


def bound_joint_layer_radius_point(component, direction_bounds, physical_radius_cubed_bounds_m3,
        *, precision=128, order=28):
    """Pure common physical-radius image of a declared layer, without masks."""
    if type(component) is not JointSphericalLayerMap:
        raise ValueError("A complete declared joint spherical layer is required")
    direction = _vector(direction_bounds, "Complete reference unit ray", intervals=True)
    _precision(precision)
    if type(order) is not int or not 8 <= order <= 64:
        raise ValueError("A joint point requires an adequate explicit Taylor order")
    cubes = physical_radius_cubed_bounds_m3
    if type(cubes) is not I or not component.layer.bottom_radius_cubed_m3 <= cubes.lo <= cubes.hi <= component.layer.top_radius_m**3:
        raise ValueError("The physical-radius interval lies outside the complete source layer")
    norm = _dot(direction, direction)
    if not norm.lo <= 1 <= norm.hi:
        raise ValueError("The reference enclosure cannot contain a unit material ray")
    radius = interval_cbrt(cubes, precision)
    tau = (I(component.layer.radius_m**3)-cubes)/(3*component.layer.radius_m**2)
    rotated = rotate_joint_physical_depth(direction,
        component.angular_rate_per_physical_volume_depth_m, tau, precision=precision, order=order)
    return tuple(_rounded_interval(radius*value, precision) for value in rotated)


def rotate_joint_physical_depth(direction_bounds, angular_rate, physical_volume_depth_bounds_m,
        *, precision=128, order=28):
    """Rigorous signed-depth adapter to the preserved nonnegative-time rotor."""
    _vector(direction_bounds, "Joint rotation vector", intervals=True)
    _vector(angular_rate, "Shared physical-depth angular rate")
    _precision(precision)
    tau = physical_volume_depth_bounds_m
    if type(tau) is not I or type(order) is not int or not 8 <= order <= 64:
        raise ValueError("Joint rotation requires literal signed depth and adequate Taylor order")
    if tau.lo >= 0:
        return _rotated(direction_bounds, angular_rate, tau, precision, order)
    opposite = tuple(-value for value in angular_rate)
    if tau.hi <= 0:
        return _rotated(direction_bounds, opposite, -tau, precision, order)
    negative = _rotated(direction_bounds, opposite, I(0, -tau.lo), precision, order)
    positive = _rotated(direction_bounds, angular_rate, I(0, tau.hi), precision, order)
    return tuple(I(min(a.lo, b.lo), max(a.hi, b.hi)) for a, b in zip(negative, positive))


def bound_joint_component_point(field, component_id, direction_bounds,
        material_depth_bounds_m, *, precision=128, order=28):
    field = validate_native_joint_spherical_field(field)
    component = joint_spherical_component(field, component_id)
    cubes = joint_component_radius_cubed_bounds(component, material_depth_bounds_m)
    return bound_joint_radius_point(component, direction_bounds, cubes,
                                    precision=precision, order=order)


@dataclass(frozen=True, slots=True)
class NativeJointSphericalFaceBounds:
    field: NativeJointSphericalField
    carrier: object
    angle_bounds_radians: I
    common_physical_radius_cubed_bounds_m3: I
    physical_radius_bounds_m: I
    common_physical_volume_depth_bounds_m: I
    donor_material_depth_bounds_m: I
    receiver_material_depth_bounds_m: I
    world_position_bounds_m: tuple
    receiver_inward_unit_normal_bounds: tuple
    area_density_bounds_m2_per_radian_per_material_m: I
    local_cot_dip_bounds: I

    @property
    def reference_bank_faces_match_at_every_common_physical_radius(self):
        return True

    @property
    def whole_common_reference_face_enclosed(self):
        return True

    @property
    def independently_rotated_or_compressed_bank_faces_match(self):
        return False

    @property
    def distributed_curved_force_equilibrium_certified(self):
        return False

    @property
    def physical_material_admission_authorized(self):
        return False


def bound_joint_contact_face(field, family_id, *, physical_radius_cubed_bounds_m3=None,
        angle_bounds_radians=None, precision=128, order=28):
    """Bound the actual complete shared curved face, its normal and metric.

    The overlap is taken in physical radius, never ``min(Hdonor,Hreceiver)``.
    The common flow proves matching reference faces only.  Independent owner
    rotations or column maps require a further kinematic gluing proof.
    """
    from .geometric_native_active_contact_line_work import _trig_ranges_signed
    from .geometric_native_polygons import _plane_values
    field = validate_native_joint_spherical_field(field)
    matches = tuple(row for row in field.support.interface_carriers if row.declaration.family_id == family_id)
    if type(family_id) is not str or len(matches) != 1:
        raise ValueError("The joint field has no unique complete native carrier family")
    carrier = matches[0]
    donor = joint_spherical_component(field, carrier.donor_component.component_id)
    receiver = joint_spherical_component(field, carrier.receiver_component.component_id)
    if donor.angular_rate_per_physical_volume_depth_m != receiver.angular_rate_per_physical_volume_depth_m:
        raise ValueError("A reference face cannot glue conflicting component depth flows")
    low = max(donor.layer.bottom_radius_cubed_m3, receiver.layer.bottom_radius_cubed_m3)
    high = min(donor.layer.top_radius_m**3, receiver.layer.top_radius_m**3)
    if low >= high:
        raise ValueError("The actual bank layers have no positive common physical-radius face")
    cubes = I(low, high) if physical_radius_cubed_bounds_m3 is None else physical_radius_cubed_bounds_m3
    if type(cubes) is not I or not low <= cubes.lo <= cubes.hi <= high:
        raise ValueError("The requested common face radius lies outside one actual bank")
    start, tangent, _, actual_angle = _arc_frame(carrier.edge, precision)
    angle = I(0, carrier.native_angle_bounds_radians.hi) if angle_bounds_radians is None else angle_bounds_radians
    if type(angle) is not I or not 0 <= angle.lo <= angle.hi <= actual_angle.hi:
        raise ValueError("The complete shared face angle lies outside its actual source carrier")
    sine, cosine = _trig_ranges_signed(angle.lo, angle.hi, precision)
    direction = tuple(a*cosine+b*sine for a, b in zip(start, tangent))
    angular_tangent = tuple(-a*sine+b*cosine for a, b in zip(start, tangent))
    arc_normal = _cross(start, tangent)
    receiving_plane = _plane_values(carrier.receiver_inward_plane, precision)[1]
    orientation = _dot(arc_normal, receiving_plane)
    if orientation.lo > 0:
        sign = F(1)
    elif orientation.hi < 0:
        sign = F(-1)
    else:
        raise ValueError("The common face receiving-normal orientation is not certified")
    radius = interval_cbrt(cubes, precision)
    reference_radius = donor.layer.radius_m
    tau = (I(reference_radius**3)-cubes)/(3*reference_radius**2)
    rate = donor.angular_rate_per_physical_volume_depth_m
    projection = _dot(rate, angular_tangent)
    radial_term = reference_radius**2/radius
    shear_term = radius*radius*projection

    def square(bounds):
        return I(0 if bounds.lo <= 0 <= bounds.hi else min(bounds.lo**2, bounds.hi**2),
                 max(bounds.lo**2, bounds.hi**2))

    area = interval_sqrt(square(radial_term)+square(shear_term), precision)
    cross_product = tuple(sign*(radial_term*n-shear_term*u) for n, u in zip(arc_normal, direction))
    normal = rotate_joint_physical_depth(cross_product, rate, tau, precision=precision, order=order)
    normal = tuple(_rounded_interval(value/area, precision) for value in normal)
    position = bound_joint_radius_point(donor, direction, cubes, precision=precision, order=order)
    donor_depth = (I(donor.layer.top_radius_m**3)-cubes)/(3*reference_radius**2)
    receiver_depth = (I(receiver.layer.top_radius_m**3)-cubes)/(3*reference_radius**2)
    cot = -sign*projection*I(cubes.lo, cubes.hi)/reference_radius**2
    return NativeJointSphericalFaceBounds(field, carrier, angle, cubes, radius, tau,
        donor_depth, receiver_depth, position, normal, area, cot)
