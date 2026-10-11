"""A volume-preserving spherical lift of complete implicit material masks.

Material depth is an *areal volume* coordinate: the source assigns ``R² H
dOmega`` of material, not the Euclidean shell between ``R`` and ``R-H``.
The exact lift is ``r(z)^3=(R+w)^3-3 R² lambda z``.  Consequently its volume
Jacobian is ``R² lambda`` even on a curved mask.  The physical radial height
is reported separately, rather than silently identifying it with ``H``.

An optional explicitly declared depth-dependent SO(3) shear preserves this
Jacobian and individual-mask injectivity.  It is not inferred from a contact
dip, and does not certify gluing to independently placed neighbouring banks.
Native proxy polygons remain integration charts: actual point membership
requires the independently replayed first-entry material chronology.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F

from .geometric_native_pose import (
    FractionInterval as I, _floor_log_two, _power_two, _precision_bits,
    _rounded_interval, interval_sqrt,
)
from .geometric_native_physical_pair_time import _rotated


def _fraction(value, name, *, positive=False):
    if type(value) is not F or (positive and value <= 0):
        raise ValueError(name + " requires a literal " + ("positive " if positive else "") + "Fraction")
    return value


def _vector(value, name, *, intervals=False):
    expected = I if intervals else F
    if type(value) is not tuple or len(value) != 3 or any(type(item) is not expected for item in value):
        raise ValueError(name + " requires an immutable exact Cartesian triple")
    return value


def _precision(value):
    if type(value) is not int or not 32 <= value <= 1024:
        raise ValueError("Spherical placement precision must be a literal integer in [32,1024]")
    return value


def _dot(first, second):
    return sum((a*b for a, b in zip(first, second)), I(0))


def _cross(first, second):
    return (first[1]*second[2]-first[2]*second[1],
            first[2]*second[0]-first[0]*second[2],
            first[0]*second[1]-first[1]*second[0])


def _integer_cbrt(value):
    """Floor cube root using integers only, including enormous rationals."""
    if value < 0:
        raise ValueError("An integer cube root must be nonnegative")
    if value < 8:
        return int(value > 0)
    current = 1 << ((value.bit_length()+2)//3)
    while True:
        following = (2*current+value//(current*current))//3
        if following >= current:
            break
        current = following
    while (current+1)**3 <= value:
        current += 1
    while current**3 > value:
        current -= 1
    return current


def _cbrt_point(value, precision):
    if value < 0:
        raise ValueError("Solid radii require nonnegative cube-root arguments")
    if not value:
        return I(0)
    numerator, denominator = _integer_cbrt(value.numerator), _integer_cbrt(value.denominator)
    if numerator**3 == value.numerator and denominator**3 == value.denominator:
        return I(F(numerator, denominator))
    shift = _precision_bits(precision)-(_floor_log_two(value)//3)
    scale = _power_two(shift)
    scaled = value*scale**3
    lower = _integer_cbrt(scaled.numerator//scaled.denominator)
    return I(F(lower)/scale, F(lower+1)/scale)


def interval_cbrt(value, precision=128):
    """Enclose a positive radius monotonically, with relative precision."""
    _precision(precision)
    if type(value) is not I or value.lo < 0:
        raise ValueError("Spherical cube roots require a nonnegative FractionInterval")
    low, high = _cbrt_point(value.lo, precision), _cbrt_point(value.hi, precision)
    return I(low.lo, high.hi)


@dataclass(frozen=True, slots=True)
class SphericalLayerMap:
    """A declared injective lift of one complete spherical material mask.

    ``depth_rotation_axis`` is the full rotation at material depth ``H``;
    intermediate depth rotates by ``z/H``.  Its choice and any volume stretch
    are new explicit deformation declarations, not hidden source mechanics.
    """
    radius_m: F
    top_radial_displacement_m: F
    areal_thickness_m: F
    volume_stretch: F = F(1)
    depth_rotation_axis: tuple = (F(0), F(0), F(0))
    provenance: str = ""

    def __post_init__(self):
        _fraction(self.radius_m, "Reference radius", positive=True)
        _fraction(self.top_radial_displacement_m, "Top radial displacement")
        _fraction(self.areal_thickness_m, "Areal material thickness", positive=True)
        _fraction(self.volume_stretch, "Explicit volume stretch", positive=True)
        _vector(self.depth_rotation_axis, "Declared full-depth SO(3) axis")
        if type(self.provenance) is not str or not self.provenance.strip():
            raise ValueError("A spherical layer requires explicit deformation provenance")
        if self.top_radius_m <= 0 or self.bottom_radius_cubed_m3 <= 0:
            raise ValueError("A complete spherical layer must remain strictly outside the sphere centre")
        if sum((item*item for item in self.depth_rotation_axis), F(0)) > F(1, 16):
            raise ValueError("Declared full-depth rotation exceeds the small-increment source domain")

    @property
    def top_radius_m(self):
        return self.radius_m+self.top_radial_displacement_m

    @property
    def bottom_radius_cubed_m3(self):
        return self.top_radius_m**3-3*self.radius_m**2*self.volume_stretch*self.areal_thickness_m

    @property
    def volume_jacobian(self):
        """Physical volume per reference steradian and material-depth metre."""
        return self.radius_m**2*self.volume_stretch

    @property
    def reference_volume_jacobian(self):
        return self.radius_m**2

    @property
    def physical_radial_height_bounds_m(self):
        return I(self.top_radius_m)-interval_cbrt(I(self.bottom_radius_cubed_m3))

    @property
    def whole_component_spherical_embedding_certified(self):
        # Radius identifies z uniquely; each z slice is an invertible SO(3).
        return True

    @property
    def source_contact_dip_reconstructed(self):
        return False

    @property
    def whole_partition_noninterpenetration_certified(self):
        return False

    @property
    def global_spherical_partition_embedding_certified(self):
        return False

    @property
    def physical_material_admission_authorized(self):
        return False


def _depth(layer, depth):
    if type(layer) is not SphericalLayerMap or type(depth) is not I:
        raise ValueError("Spherical placement requires its complete declared map and depth interval")
    if not 0 <= depth.lo <= depth.hi <= layer.areal_thickness_m:
        raise ValueError("The complete material depth interval lies outside the actual layer")
    return depth


def spherical_layer_radius_bounds(layer, depth_bounds_m, *, precision=128):
    depth = _depth(layer, depth_bounds_m)
    return interval_cbrt(I(layer.top_radius_m**3)-3*layer.volume_jacobian*depth, precision)


def spherical_layer_world_point_bounds(layer, direction_bounds, depth_bounds_m, *, precision=128, order=28):
    """Bound the whole Cartesian image of an enclosed reference unit ray.

    Interval coordinates must contain the actual unit ray; an arbitrary box
    is not a material-membership certificate.  Native wrappers authenticate
    its ray and the material chronology separately.
    """
    depth = _depth(layer, depth_bounds_m)
    direction = _vector(direction_bounds, "Reference unit-ray enclosure", intervals=True)
    _precision(precision)
    if type(order) is not int or not 8 <= order <= 64:
        raise ValueError("Spherical placement requires an explicit adequate Taylor order")
    norm = _dot(direction, direction)
    if not norm.lo <= 1 <= norm.hi:
        raise ValueError("The reference direction enclosure cannot contain a unit ray")
    rotated = _rotated(direction, layer.depth_rotation_axis,
                       depth/layer.areal_thickness_m, precision, order)
    # A coordinate along the literal rotation axis is analytically invariant;
    # the shared Taylor tail must not invent motion in that coordinate.
    rotated = tuple(direction[index] if all(layer.depth_rotation_axis[j] == 0
                    for j in range(3) if j != index) else item
                    for index, item in enumerate(rotated))
    radius = spherical_layer_radius_bounds(layer, depth, precision=precision)
    return tuple(_rounded_interval(radius*item, precision) for item in rotated)


@dataclass(frozen=True, slots=True)
class SphericalArcFaceBounds:
    layer: SphericalLayerMap
    native_arc: object
    angle_bounds_radians: I
    material_depth_bounds_m: I
    radial_bounds_m: I
    world_position_bounds_m: tuple
    area_density_bounds_m2_per_radian_per_material_m: I

    @property
    def whole_declared_curved_face_enclosed(self):
        return True

    @property
    def full_spherical_footprint_and_clock_collision_excluded(self):
        return False

    @property
    def physical_material_admission_authorized(self):
        return False


def _arc_frame(arc, precision):
    from .geometric_native_polygons import (
        NativeArc, _ray_values, _plane_values, _atan2_positive, _incident,
    )
    if type(arc) is not NativeArc:
        raise ValueError("Curved physical faces require their complete native arc")
    if not _incident(arc.start, arc.plane) or not _incident(arc.end, arc.plane):
        raise ValueError("The native face endpoints are not on their complete inherited carrier plane")
    first, second = tuple(_ray_values(ray, precision)[1] for ray in (arc.start, arc.end))
    raw_plane = _plane_values(arc.plane, precision)[1]
    start = tuple(value/interval_sqrt(_dot(first, first), precision+8) for value in first)
    normal = tuple(value/interval_sqrt(_dot(raw_plane, raw_plane), precision+8) for value in raw_plane)
    tangent = _cross(normal, start)
    orientation = _dot(tangent, second)
    if orientation.hi < 0:
        tangent = tuple(-value for value in tangent)
    elif orientation.lo <= 0:
        raise ValueError("The complete native face orientation is not certified")
    angle = _atan2_positive(interval_sqrt(_dot(_cross(first, second), _cross(first, second)), precision+8),
                            _dot(first, second), precision+8)
    if angle.lo <= 0:
        raise ValueError("A physical face requires a proved positive complete native carrier")
    return start, tangent, normal, angle


def bound_spherical_arc_face(layer, native_arc, angle_bounds_radians, depth_bounds_m, *, precision=128, order=28):
    """Enclose every angle and depth of a native arc's curved complete face.

    The face is the boundary of this declared whole-mask embedding, not a
    flat prism.  Its true area density follows both spherical radius change
    and the explicitly declared depth shear; a old force-area declaration
    is never substituted for this geometric area.
    """
    from .geometric_native_active_contact_line_work import _trig_ranges_signed
    depth = _depth(layer, depth_bounds_m)
    _precision(precision)
    if type(angle_bounds_radians) is not I:
        raise ValueError("A complete native face requires an exact angle interval")
    start, tangent, normal, actual_angle = _arc_frame(native_arc, precision)
    if not 0 <= angle_bounds_radians.lo <= angle_bounds_radians.hi <= actual_angle.hi:
        raise ValueError("The face angle interval lies outside its enclosing native carrier")
    sine, cosine = _trig_ranges_signed(angle_bounds_radians.lo, angle_bounds_radians.hi, precision)
    direction = tuple(a*cosine+b*sine for a, b in zip(start, tangent))
    angular_tangent = tuple(-a*sine+b*cosine for a, b in zip(start, tangent))
    radius = spherical_layer_radius_bounds(layer, depth, precision=precision)
    # t × dX/dz is the sum of perpendicular horizontal/radial terms.
    radial_term = layer.volume_jacobian/radius
    depth_shear_term = radius*radius*_dot(layer.depth_rotation_axis, angular_tangent)/layer.areal_thickness_m
    # Interval squaring can include negative values when a signed term crosses zero.
    def squared(bounds):
        return I(0 if bounds.lo <= 0 <= bounds.hi else min(bounds.lo**2, bounds.hi**2),
                 max(bounds.lo**2, bounds.hi**2))
    area = interval_sqrt(squared(radial_term)+squared(depth_shear_term), precision)
    position = spherical_layer_world_point_bounds(layer, direction, depth, precision=precision, order=order)
    return SphericalArcFaceBounds(layer, native_arc, angle_bounds_radians, depth,
                                  radius, position, area)


@dataclass(frozen=True, slots=True)
class NativeSphericalComponentSolid:
    """The complete exact implicit mask, coupled to an explicit 3D lift."""
    source_epoch: object
    component: object
    layer: SphericalLayerMap
    paired_corrections: tuple
    source_epoch_fingerprint_sha256: str
    source_component_fingerprint_sha256: str

    @property
    def reference_volume_bounds_m3(self):
        return self.component.area_bounds_sr*self.layer.reference_volume_jacobian*self.layer.areal_thickness_m

    @property
    def physical_volume_bounds_m3(self):
        return self.reference_volume_bounds_m3*self.layer.volume_stretch

    @property
    def proxy_polygons_are_actual_material_masks(self):
        return False

    @property
    def complete_implicit_material_mask_retained(self):
        return True

    @property
    def whole_spherical_solid_embedding_certified(self):
        return True

    @property
    def physical_material_admission_authorized(self):
        return False


def bind_native_spherical_component_solid(source_epoch, component_id, *,
        top_radial_displacement_m, provenance, volume_stretch=F(1),
        depth_rotation_axis=(F(0), F(0), F(0))):
    """Authenticate source mass/mask while declaring its placement explicitly.

    ``top_radial_displacement_m`` is a constant full-mask model declaration.
    A boundary cantilever displacement is not silently promoted to a spatial
    deformation field.  The source mass and all curved corrections remain
    authoritative regardless of this declared placement.
    """
    from .geometric_native_implicit_material_epoch import NativeImplicitMaterialEpoch, validate_native_implicit_material_epoch
    from .geometric_native_solid_layer_placement import native_component_solid_thickness_m
    from .geometric_native_implicit_epoch_admission import _portable_fingerprint_sha256
    if type(source_epoch) is not NativeImplicitMaterialEpoch:
        raise ValueError("Spherical mask placement requires the complete first implicit material epoch")
    epoch = validate_native_implicit_material_epoch(source_epoch)
    if epoch.phase != "partition":
        raise ValueError("An unresolved free-transport epoch cannot supply an authenticated material partition")
    matches = tuple(row for row in epoch.components if row.component_id == component_id)
    if type(component_id) is not str or len(matches) != 1:
        raise ValueError("The complete source epoch has no unique declared physical component")
    component = matches[0]
    layer = SphericalLayerMap(F.from_float(epoch.radius_km)*1000,
        top_radial_displacement_m, native_component_solid_thickness_m(epoch, component_id),
        volume_stretch, depth_rotation_axis, provenance)
    corrections = tuple(row for row in epoch.paired_corrections
        if component_id in (row.negative_component_id, row.positive_component_id))
    return NativeSphericalComponentSolid(epoch, component, layer, corrections,
        _portable_fingerprint_sha256(epoch), _portable_fingerprint_sha256(component))


def validate_native_spherical_component_solid(solid):
    from .geometric_native_physical_history import _typed_fingerprint
    if type(solid) is not NativeSphericalComponentSolid:
        raise ValueError("An authenticated complete spherical component is required")
    replay = bind_native_spherical_component_solid(solid.source_epoch,
        solid.component.component_id, top_radial_displacement_m=solid.layer.top_radial_displacement_m,
        provenance=solid.layer.provenance, volume_stretch=solid.layer.volume_stretch,
        depth_rotation_axis=solid.layer.depth_rotation_axis)
    if _typed_fingerprint(solid) != _typed_fingerprint(replay):
        raise ValueError("Spherical solid differs from its complete native source/mask replay")
    return replay


@dataclass(frozen=True, slots=True)
class NativeSphericalMaterialPoint:
    solid: NativeSphericalComponentSolid
    endpoint_query: object
    material_depth_m: F
    world_position_bounds_m: tuple
    material_membership_authenticated: bool

    @property
    def physical_material_admission_authorized(self):
        return False


def certify_native_spherical_material_point(solid, chronology, material_depth_m, *, precision=128, order=28):
    """Place an actual live or born material point, never a proxy-chart point."""
    from .geometric_native_first_entry_material import (
        NativeFirstEntryMaterialPointChronology, evaluate_native_first_entry_material_endpoint_point,
    )
    from .geometric_native_physical_history import _typed_fingerprint
    from .geometric_native_polygons import _ray_values
    solid = validate_native_spherical_component_solid(solid)
    _fraction(material_depth_m, "Actual material depth")
    if type(chronology) is not NativeFirstEntryMaterialPointChronology or (
            _typed_fingerprint(chronology.state) != _typed_fingerprint(solid.source_epoch.source_material_state)):
        raise ValueError("Material membership requires the complete independent source point chronology")
    query = evaluate_native_first_entry_material_endpoint_point(chronology)
    component = solid.component
    if component.material_kind == "original_live":
        member = (chronology.original_parent_id == component.original_parent_id
            and query.point_query.material_id == component.material_id
            and query.point_query.material_owner == component.owner
            and query.point_query.material_allocation in ("live", "boundary")
            and query.retained_material is True)
    else:
        member = (query.actual_material_vacancy is True
            and component.material_id in query.birth_material_ids
            and component.original_parent_id in query.fixed_original_parent_ids)
    if not member:
        raise ValueError("The independent chronology does not prove membership in this actual implicit mask")
    histories = dict(solid.source_epoch.additional_owner_histories)
    ray = query.endpoint_ray.rotated(histories[component.owner])
    vector = _ray_values(ray, precision)[1]
    norm = interval_sqrt(_dot(vector, vector), precision+8)
    direction = tuple(value/norm for value in vector)
    position = spherical_layer_world_point_bounds(solid.layer, direction, I(material_depth_m),
                                                   precision=precision, order=order)
    return NativeSphericalMaterialPoint(solid, query, material_depth_m, position, True)


@dataclass(frozen=True, slots=True)
class NativeSphericalContactSolidModel:
    """One explicit common depth flow bound to an actual native carrier.

    A single SO(3) angular rate acts on both banks at the same material depth.
    Its sign follows the actual receiving inward plane.  The surface dip is
    matched at the initial arc start and reference radius, with the native
    frame approximation enclosed explicitly.  It varies with radius and
    angle; it is not a reconstruction of the missing whole-mask strain field.
    """
    support: object
    carrier: object
    donor: NativeSphericalComponentSolid
    receiver: NativeSphericalComponentSolid
    depth_rotation_axis_per_material_m: tuple
    source_cot_dip: F
    reference_surface_cot_dip_bounds: I
    donor_surface_cot_dip_bounds: I
    receiver_surface_cot_dip_bounds: I
    depth_axis_declaration_error_bounds_per_m: tuple
    provenance: str
    declaration_precision: int

    @property
    def common_material_face_depth_m(self):
        return min(self.donor.layer.areal_thickness_m, self.receiver.layer.areal_thickness_m)

    @property
    def declared_force_vertical_contact_height_m(self):
        return F.from_float(self.carrier.declaration.vertical_contact_height_m)

    @property
    def whole_spherical_partition_embedding_certified(self):
        return False

    @property
    def normal_column_and_contact_deformation_glued(self):
        return False

    @property
    def physical_material_admission_authorized(self):
        return False


def declare_native_spherical_contact_solid_model(support, carrier, *,
        donor_top_radial_displacement_m, receiver_top_radial_displacement_m,
        provenance, precision=128):
    """Declare a curved volume-preserving common dipping flow, retaining mass.

    This is an explicit physical embedding model on the actual masks.  The
    requested constant bank displacements are model declarations; arbitrary
    depth offsets, bevel removal, lost material and force-height extensions
    are absent.  Native frame midpoints define a literal rational angular
    rate, whose difference from the source geometric direction is retained.
    """
    import math
    from .geometric_native_birth_contact_support import (
        NativeBirthInterfaceCarrier, native_birth_contact_support_to_dict,
    )
    from .geometric_native_polygons import _plane_values, _incident
    from .geometric_native_physical_history import _typed_fingerprint
    from .geometric_native_solid_layer_placement import native_component_solid_thickness_m
    _precision(precision)
    native_birth_contact_support_to_dict(support)
    if type(carrier) is not NativeBirthInterfaceCarrier or not any(
            _typed_fingerprint(carrier) == _typed_fingerprint(row) for row in support.interface_carriers):
        raise ValueError("A curved contact model requires its actual complete native source carrier")
    if type(provenance) is not str or not provenance.strip():
        raise ValueError("A curved contact embedding requires explicit model provenance")
    _fraction(donor_top_radial_displacement_m, "Declared donor top displacement")
    _fraction(receiver_top_radial_displacement_m, "Declared receiver top displacement")
    if not _incident(carrier.edge.start, carrier.receiver_inward_plane) or not _incident(
            carrier.edge.end, carrier.receiver_inward_plane):
        raise ValueError("The actual receiving plane does not contain the complete native contact carrier")
    start, _, _, _ = _arc_frame(carrier.edge, precision)
    inward = _plane_values(carrier.receiver_inward_plane, precision)[1]
    norm = interval_sqrt(_dot(inward, inward), precision+8)
    inward = tuple(value/norm for value in inward)
    tangent = _cross(inward, start)
    cot = F.from_float(math.cos(carrier.declaration.dip_radians))/F.from_float(math.sin(carrier.declaration.dip_radians))
    radius = F.from_float(support.source_epoch.radius_km)*1000
    ideal_rate = tuple(-value*cot/radius for value in tangent)
    rate = tuple(value.midpoint for value in ideal_rate)
    declaration_error = tuple(I(value)-ideal for value, ideal in zip(rate, ideal_rate))
    donor_h = native_component_solid_thickness_m(support.source_epoch, carrier.donor_component.component_id)
    receiver_h = native_component_solid_thickness_m(support.source_epoch, carrier.receiver_component.component_id)
    donor = bind_native_spherical_component_solid(support.source_epoch,
        carrier.donor_component.component_id, top_radial_displacement_m=donor_top_radial_displacement_m,
        provenance=provenance, depth_rotation_axis=tuple(value*donor_h for value in rate))
    receiver = bind_native_spherical_component_solid(support.source_epoch,
        carrier.receiver_component.component_id, top_radial_displacement_m=receiver_top_radial_displacement_m,
        provenance=provenance, depth_rotation_axis=tuple(value*receiver_h for value in rate))
    projected_rate = -_dot(rate, tangent)
    reference_cot = projected_rate*radius
    donor_cot = projected_rate*donor.layer.top_radius_m**3/donor.layer.volume_jacobian
    receiver_cot = projected_rate*receiver.layer.top_radius_m**3/receiver.layer.volume_jacobian
    return NativeSphericalContactSolidModel(support, carrier, donor, receiver, rate,
        cot, reference_cot, donor_cot, receiver_cot, declaration_error, provenance, precision)


def validate_native_spherical_contact_solid_model(model):
    from .geometric_native_physical_history import _typed_fingerprint
    if type(model) is not NativeSphericalContactSolidModel:
        raise ValueError("An authenticated complete native curved contact model is required")
    replay = declare_native_spherical_contact_solid_model(model.support, model.carrier,
        donor_top_radial_displacement_m=model.donor.layer.top_radial_displacement_m,
        receiver_top_radial_displacement_m=model.receiver.layer.top_radial_displacement_m,
        provenance=model.provenance, precision=model.declaration_precision)
    if _typed_fingerprint(model) != _typed_fingerprint(replay):
        raise ValueError("Curved native contact model differs from its complete source/model replay")
    return replay


def bound_native_spherical_contact_solid_face(model, *, bank, precision=128, order=28):
    """Cover the actual curved carrier and the entire declared common depth."""
    if type(model) is not NativeSphericalContactSolidModel or bank not in ("donor", "receiver"):
        raise ValueError("A native curved face requires its exact declared model and physical bank")
    model = validate_native_spherical_contact_solid_model(model)
    solid = model.donor if bank == "donor" else model.receiver
    return bound_spherical_arc_face(solid.layer, model.carrier.edge,
        I(0, model.carrier.native_angle_bounds_radians.hi),
        I(0, model.common_material_face_depth_m), precision=precision, order=order)


@dataclass(frozen=True, slots=True)
class SphericalSolidMotionBounds:
    layer: SphericalLayerMap
    time_fraction_bounds: I
    material_depth_bounds_m: I
    radial_bounds_m: I
    world_position_bounds_m: tuple

    @property
    def whole_declared_depth_and_clock_enclosed(self):
        return True

    @property
    def full_spherical_footprint_and_clock_collision_excluded(self):
        return False

    @property
    def physical_material_admission_authorized(self):
        return False


def bound_spherical_solid_motion(layer, direction_bounds, depth_bounds_m, *,
        owner_rotation_axis, top_radial_increment_m, time_fraction_bounds,
        precision=128, order=28):
    """Enclose the complete depth/clock image under a literal owner path.

    The top radial displacement is affine in the same time fraction.  Each
    slice uses the declared whole-mask depth flow before the actual ordered
    owner increment.  Callers must bind the supplied literal axis and radial
    increment to their physical candidate; this helper cannot adopt it.
    """
    depth = _depth(layer, depth_bounds_m)
    direction = _vector(direction_bounds, "Reference unit-ray enclosure", intervals=True)
    axis = _vector(owner_rotation_axis, "Literal owner rotation increment")
    _fraction(top_radial_increment_m, "Literal whole-mask top radial increment")
    _precision(precision)
    if type(time_fraction_bounds) is not I or not 0 <= time_fraction_bounds.lo <= time_fraction_bounds.hi <= 1:
        raise ValueError("A physical solid motion requires the complete clock interval inside [0,1]")
    if type(order) is not int or not 8 <= order <= 64:
        raise ValueError("A physical solid motion requires an explicit adequate Taylor order")
    norm = _dot(direction, direction)
    if not norm.lo <= 1 <= norm.hi:
        raise ValueError("The reference direction enclosure cannot contain a unit ray")
    top = I(layer.top_radius_m)+time_fraction_bounds*top_radial_increment_m
    if top.lo <= 0:
        raise ValueError("The whole-clock top radius is not strictly positive")
    cubes = I(top.lo**3, top.hi**3)-3*layer.volume_jacobian*depth
    if cubes.lo <= 0:
        raise ValueError("The whole-clock complete solid reaches the sphere centre")
    radius = interval_cbrt(cubes, precision)
    depth_direction = _rotated(direction, layer.depth_rotation_axis,
                                depth/layer.areal_thickness_m, precision, order)
    world_direction = _rotated(depth_direction, axis, time_fraction_bounds, precision, order)
    position = tuple(_rounded_interval(radius*item, precision) for item in world_direction)
    return SphericalSolidMotionBounds(layer, time_fraction_bounds, depth, radius, position)


def bound_spherical_arc_face_motion(layer, native_arc, angle_bounds_radians, depth_bounds_m, *,
        owner_rotation_axis, top_radial_increment_m, time_fraction_bounds,
        precision=128, order=28):
    """Enclose all carrier angles, all material depths and the entire clock."""
    from .geometric_native_active_contact_line_work import _trig_ranges_signed
    start, tangent, _, actual_angle = _arc_frame(native_arc, precision)
    if type(angle_bounds_radians) is not I or not 0 <= angle_bounds_radians.lo <= angle_bounds_radians.hi <= actual_angle.hi:
        raise ValueError("The complete moving face angle interval exceeds its native carrier")
    sine, cosine = _trig_ranges_signed(angle_bounds_radians.lo, angle_bounds_radians.hi, precision)
    direction = tuple(a*cosine+b*sine for a, b in zip(start, tangent))
    return bound_spherical_solid_motion(layer, direction, depth_bounds_m,
        owner_rotation_axis=owner_rotation_axis, top_radial_increment_m=top_radial_increment_m,
        time_fraction_bounds=time_fraction_bounds, precision=precision, order=order)


def native_polygon_direction_bounds(polygon, *, precision=128):
    """Enclose a complete spherical polygon, including interior extrema.

    Coordinate extrema of a spherical polygon occur on its complete boundary
    arcs or at the two stationary coordinate-axis rays.  Vertices alone are
    insufficient.  Unresolved axis halfspace predicates are included, giving
    a safe containing enclosure without changing material topology.
    """
    from .geometric_native_polygons import (
        NativeArc, NativeRay, _world, _side, validate_native_polygon,
    )
    from .geometric_native_active_contact_line_work import _trig_ranges_signed
    _precision(precision)
    world = _world(validate_native_polygon(polygon))
    intervals = []
    for first, second, plane in zip(world.vertices,
            world.vertices[1:]+world.vertices[:1], world.supports):
        arc = NativeArc(first, second, plane, 1.)
        start, tangent, _, angle = _arc_frame(arc, precision)
        sine, cosine = _trig_ranges_signed(F(0), angle.hi, precision)
        intervals.append(tuple(a*cosine+b*sine for a, b in zip(start, tangent)))
    low = [min(row[index].lo for row in intervals) for index in range(3)]
    high = [max(row[index].hi for row in intervals) for index in range(3)]
    for coordinate in range(3):
        for sign in (-1, 1):
            point = NativeRay("seed", tuple(F(sign if index == coordinate else 0) for index in range(3)))
            excluded = False
            for plane in world.supports:
                try:
                    if _side(point, plane) < 0:
                        excluded = True
                        break
                except ValueError:
                    # The coordinate-axis point is a possible stationary
                    # extremum until an independently proved side excludes it.
                    pass
            if not excluded:
                if sign < 0:
                    low[coordinate] = F(-1)
                else:
                    high[coordinate] = F(1)
    return tuple(I(max(F(-1), a), min(F(1), b)) for a, b in zip(low, high))


@dataclass(frozen=True, slots=True)
class NativeSphericalWholeMaskMotionBounds:
    solid: NativeSphericalComponentSolid
    containing_direction_bounds: tuple
    containing_chart_count: int
    positive_curved_correction_chart_count: int
    motion: SphericalSolidMotionBounds

    @property
    def whole_actual_implicit_mask_depth_and_clock_enclosed(self):
        return True

    @property
    def containing_charts_assign_material(self):
        return False

    @property
    def physical_material_admission_authorized(self):
        return False


def bound_native_spherical_whole_mask_motion(solid, *, owner_rotation_axis,
        top_radial_increment_m, time_fraction_bounds, precision=128, order=28):
    """Enclose the complete source mask and all depths over the actual clock.

    Live proxies contain their negative curved correction.  HOT proxies are
    augmented by every source-bound positive curved correction's containing
    chart.  Thus no implicit material portion is omitted, while none of these
    disposable containing charts is assigned as a material reservoir.
    """
    solid = validate_native_spherical_component_solid(solid)
    positive = tuple(row.containing_chart for row in solid.paired_corrections
        if row.positive_component_id == solid.component.component_id)
    charts = solid.component.proxy_geometries+positive
    if not charts:
        raise ValueError("A positive complete implicit material mask has no authenticated containing chart")
    bounds = tuple(native_polygon_direction_bounds(chart, precision=precision) for chart in charts)
    direction = tuple(I(min(row[index].lo for row in bounds),
                        max(row[index].hi for row in bounds)) for index in range(3))
    motion = bound_spherical_solid_motion(solid.layer, direction, I(0, solid.layer.areal_thickness_m),
        owner_rotation_axis=owner_rotation_axis, top_radial_increment_m=top_radial_increment_m,
        time_fraction_bounds=time_fraction_bounds, precision=precision, order=order)
    return NativeSphericalWholeMaskMotionBounds(solid, direction, len(charts), len(positive), motion)
