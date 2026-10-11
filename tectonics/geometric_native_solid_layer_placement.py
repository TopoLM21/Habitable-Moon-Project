"""Finite-thickness solids in an explicitly declared local dipping chart.

The material coordinates are strike ``s``, horizontal bank coordinate ``x``
and positive depth ``z``.  Both banks use the SAME depth shear,
``X(s,x,z) = (s, x + z*cot(dip), -z)``.  Their common face is therefore
``horizontal = depth*cot(dip)``.  This construction preserves the complete
material volume; it does not remove a receiver bevel or move a donor down by
its neighbour's thickness.  The shear of (s,x,outward) has determinant one.

This is a local affine prism model, not a reconstruction of the spherical
solid embedding.  A native adapter authenticates thickness and the existing
declared dip/contact height; bank widths remain explicit new declarations.
Neither adapter nor a local no-intersection result authenticates global
gluing, a curved whole-footprint solid, spring preload placement, forces or
material admission.  In particular, a one-kilometre force carrier cannot be
called a complete face of two-kilometre HOT crust.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
import math

from .geometric_native_pose import FractionInterval as I, interval_sqrt
from .geometric_native_physical_pair_time import _relative_change


def _positive(value, name):
    if type(value) is not F or value <= 0:
        raise ValueError(name + " requires a positive literal Fraction")
    return value


def _scalar(value, name):
    if type(value) is not F:
        raise ValueError(name + " requires a literal Fraction")
    return value


def _interval(value, name):
    if type(value) is not I:
        raise ValueError(name + " requires an exact FractionInterval")
    return value


def _vector(value, name, *, intervals=False):
    expected = I if intervals else F
    if type(value) is not tuple or len(value) != 3 or any(type(item) is not expected for item in value):
        raise ValueError(name + " requires an immutable exact three-component tuple")
    return value


def _dot(first, second):
    return sum((a*b for a, b in zip(first, second)), F(0))


@dataclass(frozen=True, slots=True)
class LocalDippingLayerChart:
    """Independent finite bank widths and thicknesses in one common chart.

    Widths are material dimensions.  They need not exceed ``H*cot(dip)``:
    arbitrarily narrow neighbouring prisms can share a depth-sheared face.
    Their shifted physical extent must be covered when this local model is
    subsequently embedded and glued to other material.
    """
    strike_length_m: F
    donor_width_m: F
    receiver_width_m: F
    donor_thickness_m: F
    receiver_thickness_m: F
    cot_dip: F
    force_contact_height_m: F
    provenance: str

    def __post_init__(self):
        for name in ("strike_length_m", "donor_width_m", "receiver_width_m",
                     "donor_thickness_m", "receiver_thickness_m", "cot_dip",
                     "force_contact_height_m"):
            _positive(getattr(self, name), name)
        if type(self.provenance) is not str or not self.provenance.strip():
            raise ValueError("The local solid chart requires explicit model provenance")

    @property
    def common_physical_face_height_m(self):
        return min(self.donor_thickness_m, self.receiver_thickness_m)

    @property
    def whole_physical_face_force_coverage(self):
        return self.force_contact_height_m >= self.common_physical_face_height_m

    @property
    def depth_shear_matrix(self):
        # Acts on the unsheared physical coordinates (strike, horizontal,
        # outward); positive material depth is minus outward.
        return ((F(1), F(0), F(0)), (F(0), F(1), -self.cot_dip),
                (F(0), F(0), F(1)))

    @property
    def volume_jacobian(self):
        return F(1)

    @property
    def donor_volume_m3(self):
        return self.strike_length_m*self.donor_width_m*self.donor_thickness_m

    @property
    def receiver_volume_m3(self):
        return self.strike_length_m*self.receiver_width_m*self.receiver_thickness_m

    @property
    def physical_horizontal_extent_m(self):
        return I(-self.donor_width_m,
                 max(self.donor_thickness_m*self.cot_dip,
                     self.receiver_width_m+self.receiver_thickness_m*self.cot_dip))

    @property
    def normal_covector(self):
        # sin(dip)*(horizontal + cot(dip)*outward); its sign is exact.
        return (F(0), F(1), self.cot_dip)

    @property
    def whole_spherical_solid_embedding_certified(self):
        return False

    @property
    def physical_material_admission_authorized(self):
        return False


def require_complete_physical_contact_face(chart):
    if type(chart) is not LocalDippingLayerChart:
        raise ValueError("Complete face coverage requires the declared local solid chart")
    if not chart.whole_physical_face_force_coverage:
        raise ValueError("The force contact height does not cover the complete finite-thickness physical face")
    return chart.common_physical_face_height_m


def local_layer_point(chart, *, bank, strike_m, bank_coordinate_m, depth_m):
    """Place one actual material point, preserving its bank and depth label."""
    if type(chart) is not LocalDippingLayerChart or type(bank) is not str or bank not in ("donor", "receiver"):
        raise ValueError("A material point requires its exact declared chart and bank")
    s, x, z = (_scalar(value, name) for value, name in
               ((strike_m, "Strike"), (bank_coordinate_m, "Bank coordinate"), (depth_m, "Depth")))
    low, high, height = ((-chart.donor_width_m, F(0), chart.donor_thickness_m)
                        if bank == "donor" else (F(0), chart.receiver_width_m, chart.receiver_thickness_m))
    if not 0 <= s <= chart.strike_length_m or not low <= x <= high or not 0 <= z <= height:
        raise ValueError("The complete point lies outside its actual material prism")
    return s, x+z*chart.cot_dip, -z


def local_layer_world_point(chart, *, bank, strike_m, bank_coordinate_m, depth_m,
                            origin_m, strike_direction, horizontal_direction, outward_direction):
    """Place a point in an exact orthonormal Cartesian frame.

    This frame belongs to the declared local model.  Replacing a varying
    native great-circle frame by it is a separate embedding approximation.
    """
    origin = _vector(origin_m, "World origin")
    basis = tuple(_vector(row, "Local world frame") for row in
                  (strike_direction, horizontal_direction, outward_direction))
    if any(_dot(row, row) != 1 for row in basis) or any(
            _dot(basis[i], basis[j]) for i in range(3) for j in range(i)):
        raise ValueError("A local world frame must be exactly orthonormal")
    point = local_layer_point(chart, bank=bank, strike_m=strike_m,
                             bank_coordinate_m=bank_coordinate_m, depth_m=depth_m)
    return tuple(origin[i]+sum((value*row[i] for value, row in zip(point, basis)), F(0)) for i in range(3))


@dataclass(frozen=True, slots=True)
class LocalLayerSeparationBounds:
    chart: LocalDippingLayerChart
    time_fraction_bounds: I
    common_face_normal_gap_bounds_m: I
    donor_whole_solid_signed_plane_bounds_m: I
    donor_inward_coordinate_plane_coefficient: I
    local_prism_intersection_excluded: bool

    @property
    def full_spherical_footprint_and_clock_collision_excluded(self):
        return False

    @property
    def finite_pose_force_equilibrium_proved(self):
        return False

    @property
    def physical_material_admission_authorized(self):
        return False


def bound_local_dipping_layer_separation(chart, *, donor_axis, receiver_axis,
        relative_translation_receiver_bounds_m, time_fraction_bounds,
        precision=128, order=28):
    """Bound the ENTIRE finite donor prism on a whole physical time window.

    Axes are the actual dimensionless finite rotation increments in this
    common Cartesian chart.  The relative placement is
    ``Q_receiver(u)^T Q_donor(u) X + t_receiver(u)``.  The supplied translation
    enclosure MUST cover that same receiver-frame path over the full window;
    an endpoint translation is insufficient.  This mathematical helper does
    not authenticate its caller's source or translation trajectory.

    The receiver occupies ``horizontal+cot(dip)*outward >= 0`` exactly.  A
    nonpositive upper bound on the WHOLE donor proves no interior collision
    in this local model, including cases where the face alone is insufficient
    after rotation.  Zero permits touching; no area/length tolerance is used.
    """
    if type(chart) is not LocalDippingLayerChart:
        raise ValueError("Layer separation requires the explicit local material chart")
    donor, receiver = _vector(donor_axis, "Donor axis"), _vector(receiver_axis, "Receiver axis")
    translation = _vector(relative_translation_receiver_bounds_m, "Whole-clock relative translation", intervals=True)
    clock = _interval(time_fraction_bounds, "Physical clock")
    if not 0 <= clock.lo <= clock.hi <= 1:
        raise ValueError("The actual physical clock window must lie inside [0,1]")
    if type(precision) is not int or not 32 <= precision <= 1024 or type(order) is not int or not 8 <= order <= 64:
        raise ValueError("Layer placement requires adequate literal rotation budgets")
    basis = ((F(1), F(0), F(0)), (F(0), F(1), F(0)), (F(0), F(0), F(1)))
    changes = tuple(_relative_change(row, donor, receiver, clock, precision, order) for row in basis)
    normal, alpha = chart.normal_covector, chart.cot_dip
    strike_coefficient = _dot(normal, changes[0])
    depth_coefficient = _dot(normal, tuple(alpha*a-b for a, b in zip(changes[1], changes[2])))
    inward_coefficient = I(1)+_dot(normal, changes[1])
    offset = _dot(normal, translation)
    face = offset+strike_coefficient*I(0, chart.strike_length_m)+depth_coefficient*I(0, chart.common_physical_face_height_m)
    whole_donor = (offset+strike_coefficient*I(0, chart.strike_length_m)
                   +depth_coefficient*I(0, chart.donor_thickness_m)
                   +inward_coefficient*I(-chart.donor_width_m, 0))
    scale = interval_sqrt(I(1+alpha*alpha), precision+16)
    return LocalLayerSeparationBounds(chart, clock, -face/scale, whole_donor/scale,
                                     inward_coefficient, whole_donor.hi <= 0)


def native_component_solid_thickness_m(source_epoch, component_id):
    """Authenticate actual crust PLUS cold-mantle material thickness.

    Elastic beam thickness and normal-column length never enter this sum.
    HOT material has its explicitly declared crust and zero cold mantle.
    The result authenticates an areal material amount, not its 3D placement.
    """
    from .geometric_native_implicit_material_epoch import NativeImplicitMaterialEpoch, validate_native_implicit_material_epoch
    if type(source_epoch) is not NativeImplicitMaterialEpoch:
        raise ValueError("This placement adapter requires the complete first implicit material epoch")
    epoch = validate_native_implicit_material_epoch(source_epoch)
    if type(component_id) is not str or not component_id:
        raise ValueError("An actual component ID is required for its source thickness")
    matches = tuple(row for row in epoch.components if row.component_id == component_id)
    if len(matches) != 1:
        raise ValueError("The source epoch has no unique physical material component")
    component = matches[0]
    if component.material_kind == "hot_birth":
        births = tuple(row.newborn for row in epoch.source_material_state.endpoint_vacancies
                       if row.newborn is not None and row.newborn.material_id == component.material_id)
        if len(births) != 1 or births[0].owner != component.owner:
            raise ValueError("The HOT component has no unique original material birth")
        specific = births[0].specific_properties
        if type(specific) is not tuple or len(specific) != 3 or any(type(value) is not F for value in specific):
            raise ValueError("HOT areal material properties lost their exact source types")
        crust, cold = specific[:2]
    elif component.material_kind == "original_live":
        parents = tuple(row for row in epoch.source_material_state.journal.source_surface.fragments
                        if row.fragment_id == component.original_parent_id)
        if len(parents) != 1 or parents[0].parcel.material_id != component.material_id or parents[0].parcel.plate != component.owner:
            raise ValueError("The live component has no unique original physical parent")
        crust, cold = map(F.from_float, parents[0].parcel.specific_properties[:2])
    else:
        raise ValueError("Only actual original live or HOT material has this solid thickness model")
    if crust <= 0 or cold < 0:
        raise ValueError("Source solid material amounts require positive crust and nonnegative mantle")
    return (crust+cold)*1000


def build_native_contact_solid_layer_chart(support, carrier, *, donor_width_m,
                                          receiver_width_m, provenance):
    """Bind a NEW local model to actual source thickness, arc and dip.

    The bank widths and embedding provenance are explicit new declarations.
    An upper bound on the native strike extent covers the entire original
    carrier in this affine chart; it does not certify its curved embedding.
    Existing force-height coverage is exposed rather than silently extended.
    """
    from .geometric_native_birth_contact_support import native_birth_contact_support_to_dict, NativeBirthInterfaceCarrier
    from .geometric_native_physical_history import _typed_fingerprint
    native_birth_contact_support_to_dict(support)
    if type(carrier) is not NativeBirthInterfaceCarrier or not any(
            _typed_fingerprint(carrier) == _typed_fingerprint(row) for row in support.interface_carriers):
        raise ValueError("The complete physical carrier is not part of the authenticated source support")
    radius = F.from_float(support.source_epoch.radius_km)*1000
    dip = carrier.declaration.dip_radians
    cot = F.from_float(math.cos(dip))/F.from_float(math.sin(dip))
    return LocalDippingLayerChart(radius*carrier.native_angle_bounds_radians.hi,
        donor_width_m, receiver_width_m,
        native_component_solid_thickness_m(support.source_epoch, carrier.donor_component.component_id),
        native_component_solid_thickness_m(support.source_epoch, carrier.receiver_component.component_id),
        cot, F.from_float(carrier.declaration.vertical_contact_height_m), provenance)
