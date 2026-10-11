"""Source-bound native declarations for supported, unbonded HOT contacts.

Foundations cover each entire declared HOT material mask once.  Their fixed
constitutive reference areas are explicit endpoints of native area bounds;
they never debit a reservoir or replace an implicit mask by a polygon.

The first continuation has two positive shared arcs and two vertex-created
fronts.  This adapter declares the arcs, retains the unresolved fronts, and
does not supply complete temporal admission.  Inclined interface measure is
``C dtheta``, C = R * vertical_contact_height / sin(dip), rather than overlap
area.  HOT foundations have independent radial coordinates and require no
cold cantilever clamp.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction as F
import hashlib
import json
import math

import numpy as np

from .geometric_native_birth_contact import (
    BornContactParameters, BornContactReference, BornFoundationParameters,
    BornFoundationReference, born_contact_parameters_to_dict,
    born_foundation_parameters_to_dict,
)
from .geometric_native_implicit_epoch_admission import (
    _portable_fingerprint_sha256, native_implicit_epoch_incoming_geometry_to_dict,
)
from .geometric_native_implicit_material_epoch import (
    NativeImplicitMaterialEpoch, validate_native_implicit_material_epoch,
)
from .geometric_native_polygons import (
    _atan2_positive, _cross, _dot, _plane_values, _ray_values, _same_plane,
    native_arc_to_dict, native_plane_to_dict, native_shared_boundary_arcs,
)
from .geometric_native_pose import FractionInterval as I, interval_sqrt
from .geometric_native_physical_history import _typed_fingerprint, _literal_json

FORMAT = "native-born-contact-foundation-support-declaration-1"


def _positive(value, name):
    if type(value) is not float or not math.isfinite(value) or value <= 0:
        raise ValueError(name + " requires an explicit positive finite float")
    return value


def _finite(value, name):
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(name + " requires an explicit finite float")
    return value


def _text(value, name):
    if type(value) is not str or not value.strip():
        raise ValueError(name + " requires explicit nonempty text")
    return value


def _r(value):
    return [value.numerator, value.denominator]


def _i(value):
    return dict(lo=_r(value.lo), hi=_r(value.hi))


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()


def _fixed_endpoint(bounds, choice):
    if choice not in ("lower", "upper") or type(choice) is not str:
        raise ValueError("Reference area endpoint must be explicitly lower or upper")
    result = bounds.lo if choice == "lower" else bounds.hi
    if result <= 0:
        raise ValueError("A fixed constitutive reference area must be positive")
    return result


@dataclass(frozen=True, slots=True)
class NativeHotFoundationDeclaration:
    component_id: str
    parameters: BornFoundationParameters
    reference_area_endpoint: str
    radial_birth_m: float
    provenance: str


@dataclass(frozen=True, slots=True)
class NativeBirthInterfaceDeclaration:
    family_id: str
    donor_component_id: str
    receiver_component_id: str
    donor_piece_sequence: int
    receiver_piece_sequence: int
    shared_edge_sequence: int
    parameters: BornContactParameters
    dip_radians: float
    vertical_contact_height_m: float
    reference_angle_endpoint: str
    birth_gap_free_m: float
    birth_jump_free_m: float
    provenance: str


@dataclass(frozen=True, slots=True)
class NativeHotFoundationPatch:
    declaration: NativeHotFoundationDeclaration
    component: object
    material_birth_time_myr: F
    area_bounds_m2: I
    fixed_reference_area_m2: F
    float_reference_area_m2: float
    float_reference_area_error_m2: F
    stiffness_n_m: F


@dataclass(frozen=True, slots=True)
class NativeBirthInterfaceCarrier:
    declaration: NativeBirthInterfaceDeclaration
    donor_component: object
    receiver_component: object
    edge: object
    receiver_inward_plane: object
    native_angle_bounds_radians: I
    inclined_measure_density_m2_per_radian: F
    interface_area_bounds_m2: I
    fixed_reference_interface_area_m2: F
    float_reference_interface_area_m2: float
    float_reference_interface_area_error_m2: F
    foundation_component_ids: tuple


@dataclass(frozen=True, slots=True)
class NativeUnresolvedBirthFront:
    donor_component_id: str
    receiver_component_id: str
    donor_piece_sequence: int
    receiver_piece_sequence: int
    endpoint_overlap_area_bounds_sr: I
    initial_shared_arc_count: int
    reason: str


@dataclass(frozen=True, slots=True)
class NativeBirthContactSupport:
    source_epoch: NativeImplicitMaterialEpoch
    incoming_geometry: object
    foundation_declarations: tuple
    interface_declarations: tuple
    foundations: tuple
    interface_carriers: tuple
    unresolved_fronts: tuple
    support_sha256: str


def _angle_bounds(edge, precision=256):
    first, second = (_ray_values(ray, precision)[1] for ray in (edge.start, edge.end))
    product = _cross(first, second)
    norm = interval_sqrt(_dot(product, product), precision+8)
    angle = _atan2_positive(norm, _dot(first, second), precision+8)
    if angle.lo <= 0:
        raise ValueError("A declared birth carrier must have proved positive native length")
    return angle


def _component_body(component):
    return dict(component_id=component.component_id, material_id=component.material_id,
        parent_id=component.original_parent_id, owner=component.owner,
        material_kind=component.material_kind, domain_expression=component.domain_expression,
        complete_component_fingerprint_sha256=_portable_fingerprint_sha256(component))


def _body(value):
    return dict(format=FORMAT,
        source_epoch_sha256=value.source_epoch.epoch_sha256,
        complete_source_epoch_fingerprint_sha256=_portable_fingerprint_sha256(value.source_epoch),
        source_material_state_sha256=value.source_epoch.source_material_state.state_sha256,
        contact_reference_birth_time_myr=_r(value.source_epoch.time_myr),
        incoming_geometry_sha256=value.incoming_geometry.geometry_sha256,
        complete_incoming_geometry_fingerprint_sha256=_portable_fingerprint_sha256(value.incoming_geometry),
        foundations=[dict(declaration=asdict(row.declaration),
            component=_component_body(row.component), material_birth_time_myr=_r(row.material_birth_time_myr),
            native_basal_area_bounds_m2=_i(row.area_bounds_m2),
            fixed_constitutive_reference_area_m2=_r(row.fixed_reference_area_m2),
            float_reference_area_m2=row.float_reference_area_m2,
            float_reference_area_error_m2=_r(row.float_reference_area_error_m2),
            stiffness_n_m=_r(row.stiffness_n_m), whole_implicit_material_mask_retained=True,
            reservoir_area_debit=False, cold_cantilever_clamp_required=False)
            for row in value.foundations],
        interface_carriers=[dict(declaration=asdict(row.declaration),
            donor=_component_body(row.donor_component), receiver=_component_body(row.receiver_component),
            exact_initial_shared_arc=native_arc_to_dict(row.edge),
            receiver_inward_plane=native_plane_to_dict(row.receiver_inward_plane),
            native_angle_bounds_radians=_i(row.native_angle_bounds_radians),
            inclined_measure_density_m2_per_radian=_r(row.inclined_measure_density_m2_per_radian),
            interface_area_bounds_m2=_i(row.interface_area_bounds_m2),
            fixed_constitutive_reference_interface_area_m2=_r(row.fixed_reference_interface_area_m2),
            float_reference_interface_area_m2=row.float_reference_interface_area_m2,
            float_reference_interface_area_error_m2=_r(row.float_reference_interface_area_error_m2),
            foundation_component_ids=list(row.foundation_component_ids),
            measure="R*vertical_contact_height/literal_sin(dip) dtheta_native",
            overlap_area_used_as_interface_measure=False)
            for row in value.interface_carriers],
        unresolved_fronts=[dict(donor_component_id=row.donor_component_id,
            receiver_component_id=row.receiver_component_id, donor_piece_sequence=row.donor_piece_sequence,
            receiver_piece_sequence=row.receiver_piece_sequence,
            endpoint_overlap_area_bounds_sr=_i(row.endpoint_overlap_area_bounds_sr),
            initial_shared_arc_count=row.initial_shared_arc_count, reason=row.reason,
            endpoint_overlap_converted_to_contact_area=False,
            front_potential_retained=True) for row in value.unresolved_fronts],
        known_positive_witness_count=len(value.incoming_geometry.positive_contacts),
        complete_all_pairs_temporal_coverage=False,
        scope="all HOT foundation masks; declared initial shared arcs; unresolved vertex fronts; support only")


def declare_native_birth_contact_support(epoch, foundation_declarations,
        interface_declarations, *, incoming_geometry):
    """Bind explicit new physical declarations without accepting a motion."""
    if type(epoch) is not NativeImplicitMaterialEpoch:
        raise ValueError("This birth adapter requires the exact first implicit material epoch")
    epoch = validate_native_implicit_material_epoch(epoch)
    native_implicit_epoch_incoming_geometry_to_dict(incoming_geometry)
    if _typed_fingerprint(incoming_geometry.source_epoch) != _typed_fingerprint(epoch):
        raise ValueError("Birth geometry must bind the complete independent material epoch")
    if type(foundation_declarations) is not tuple or type(interface_declarations) is not tuple:
        raise ValueError("Birth declarations must be literal immutable tuples")
    if (any(type(row) is not NativeHotFoundationDeclaration for row in foundation_declarations)
            or any(type(row) is not NativeBirthInterfaceDeclaration for row in interface_declarations)):
        raise ValueError("Birth declaration classes must retain their explicit native types")
    components = {row.component_id: row for row in epoch.components}
    hot = {key: row for key, row in components.items() if row.material_kind == "hot_birth"}
    if tuple(row.component_id for row in foundation_declarations) != tuple(sorted(hot)):
        raise ValueError("Every whole HOT foundation component must be declared exactly once in canonical order")
    births = {row.newborn.material_id: row.newborn.birth_time_myr
        for row in epoch.source_material_state.endpoint_vacancies if row.newborn is not None}
    radius = F.from_float(float(epoch.radius_km))*1000
    foundations = []
    for declaration in foundation_declarations:
        if type(declaration) is not NativeHotFoundationDeclaration or type(declaration.parameters) is not BornFoundationParameters:
            raise ValueError("HOT foundation requires its explicit independent physical declaration")
        born_foundation_parameters_to_dict(declaration.parameters)
        _finite(declaration.radial_birth_m, "Foundation birth displacement")
        _text(declaration.provenance, "Foundation provenance")
        component = hot[declaration.component_id]
        bounds = component.area_bounds_sr*radius**2
        area = _fixed_endpoint(bounds, declaration.reference_area_endpoint)
        numeric = float(area)
        _positive(numeric, "Representable foundation reference area")
        rho = F.from_float(_positive(declaration.parameters.density_kg_m3, "Foundation density"))
        gravity = F.from_float(_positive(declaration.parameters.gravity_m_s2, "Foundation gravity"))
        foundations.append(NativeHotFoundationPatch(declaration, component, births[component.material_id],
            bounds, area, numeric, abs(F.from_float(numeric)-area), rho*gravity*area))
    if tuple(row.family_id for row in interface_declarations) != tuple(sorted(set(row.family_id for row in interface_declarations))):
        raise ValueError("Born family IDs must be unique and canonical")
    carriers, covered = [], set()
    negative = {row.negative_component_id for row in epoch.paired_corrections}
    for declaration in interface_declarations:
        if type(declaration) is not NativeBirthInterfaceDeclaration or type(declaration.parameters) is not BornContactParameters:
            raise ValueError("Born interface requires explicit contact parameters")
        born_contact_parameters_to_dict(declaration.parameters)
        _text(declaration.family_id, "Born family ID")
        _text(declaration.provenance, "Born family provenance")
        dip = _positive(declaration.dip_radians, "Born dip")
        if dip >= math.pi/2:
            raise ValueError("Born dip must be strictly below pi/2")
        height = _positive(declaration.vertical_contact_height_m, "Vertical contact height")
        if _finite(declaration.birth_gap_free_m, "Birth gap") > 0:
            raise ValueError("Declared paid birth preparation requires compression or zero gap")
        _finite(declaration.birth_jump_free_m, "Birth shear reference")
        donor, receiver = (components[key] for key in (declaration.donor_component_id, declaration.receiver_component_id))
        if donor.owner == receiver.owner or not ({donor.component_id, receiver.component_id} & set(hot)):
            raise ValueError("A born interface requires opposing owners and actual HOT material")
        if donor.component_id in negative or receiver.component_id in negative:
            raise ValueError("A negatively corrected proxy cannot prove full positive material support")
        indices = (declaration.donor_piece_sequence, declaration.receiver_piece_sequence, declaration.shared_edge_sequence)
        if any(type(index) is not int or index < 0 for index in indices):
            raise ValueError("Native piece and edge indices require nonnegative literal integers")
        if indices[0] >= len(donor.proxy_geometries) or indices[1] >= len(receiver.proxy_geometries):
            raise ValueError("Born native piece index is outside the complete actual component")
        dg, rg = donor.proxy_geometries[indices[0]], receiver.proxy_geometries[indices[1]]
        edges = native_shared_boundary_arcs(dg, rg)
        if indices[2] >= len(edges):
            raise ValueError("Born native shared-edge index is outside the actual carrier geometry")
        edge = edges[indices[2]]
        planes = tuple(plane for plane in rg.support_planes if _same_plane(plane, edge.plane))
        if len(planes) != 1:
            raise ValueError("Born receiver inward plane is not uniquely source-bound")
        angles = _angle_bounds(edge)
        density = radius*F.from_float(height)/F.from_float(math.sin(dip))
        area_bounds = density*angles
        area = density*_fixed_endpoint(angles, declaration.reference_angle_endpoint)
        numeric = float(area)
        _positive(numeric, "Representable inclined interface area")
        carriers.append(NativeBirthInterfaceCarrier(declaration, donor, receiver, edge, planes[0],
            angles, density, area_bounds, area, numeric, abs(F.from_float(numeric)-area),
            tuple(sorted({donor.component_id, receiver.component_id} & set(hot)))))
        key = (donor.component_id, receiver.component_id, indices[0], indices[1])
        if key in covered:
            raise ValueError("Two declarations cannot silently duplicate the same initial carrier pair")
        covered.add(key)
    fronts = []
    for contact in incoming_geometry.positive_contacts:
        key = (contact.donor_component_id, contact.receiver_component_id,
            contact.donor_piece_sequence, contact.receiver_piece_sequence)
        if key not in covered:
            fronts.append(NativeUnresolvedBirthFront(*key, contact.endpoint_intersection_area_bounds_sr,
                len(contact.immediate_incoming_edges), "positive future contact needs a complete front measure and new temporal mechanical admission"))
    value = NativeBirthContactSupport(epoch, incoming_geometry, foundation_declarations,
        interface_declarations, tuple(foundations), tuple(carriers), tuple(fronts), "")
    from dataclasses import replace
    return replace(value, support_sha256=hashlib.sha256(_encoded(_body(value))).hexdigest())


def native_birth_contact_support_to_dict(value):
    if type(value) is not NativeBirthContactSupport:
        raise ValueError("Expected an immutable native birth support declaration")
    replay = declare_native_birth_contact_support(value.source_epoch, value.foundation_declarations,
        value.interface_declarations, incoming_geometry=value.incoming_geometry)
    if _typed_fingerprint(replay) != _typed_fingerprint(value):
        raise ValueError("Birth support differs from complete source and explicit declaration replay")
    return json.loads(_encoded(dict(_body(value), support_sha256=value.support_sha256)))


def native_birth_contact_support_from_dict(raw, *, expected_epoch,
        expected_foundation_declarations, expected_interface_declarations, expected_incoming_geometry):
    _literal_json(raw)
    replay = declare_native_birth_contact_support(expected_epoch, expected_foundation_declarations,
        expected_interface_declarations, incoming_geometry=expected_incoming_geometry)
    if type(raw) is not dict or _encoded(raw) != _encoded(native_birth_contact_support_to_dict(replay)):
        raise ValueError("Birth support differs from independent complete epoch, geometry and declarations")
    return replay


def native_born_foundation_references(support):
    """Return one generic reference for each full source-bound HOT patch."""
    native_birth_contact_support_to_dict(support)
    source = _portable_fingerprint_sha256(support.source_epoch)
    return tuple(BornFoundationReference(row.component.component_id, source,
        support.source_epoch.epoch_sha256, row.component.component_id,
        row.component.material_id, row.component.original_parent_id,
        row.component.owner, row.material_birth_time_myr, row.float_reference_area_m2,
        row.declaration.radial_birth_m, row.declaration.provenance)
        for row in support.foundations)


def native_born_contact_references(support):
    """Return only declared positive shared-arc references, never front areas."""
    native_birth_contact_support_to_dict(support)
    source = _portable_fingerprint_sha256(support.source_epoch)
    return tuple(BornContactReference(row.declaration.family_id, source,
        support.source_epoch.epoch_sha256, row.donor_component.component_id,
        row.receiver_component.component_id, row.donor_component.material_id,
        row.receiver_component.material_id, row.donor_component.original_parent_id,
        row.receiver_component.original_parent_id, row.donor_component.owner,
        row.receiver_component.owner, row.receiver_inward_plane,
        support.source_epoch.time_myr, row.float_reference_interface_area_m2,
        row.declaration.birth_gap_free_m, row.declaration.birth_jump_free_m,
        row.declaration.provenance) for row in support.interface_carriers)


@dataclass(frozen=True, slots=True)
class NativeBirthPointPort:
    donor_owner: int
    receiver_owner: int
    donor_position_m: tuple
    receiver_position_m: tuple
    donor_radial: tuple
    receiver_radial: tuple
    normal: tuple
    down_dip: tuple
    normal_operator: tuple
    shear_operator: tuple
    donor_radial_coupling: tuple
    receiver_radial_coupling: tuple
    angular_owner_dofs: tuple
    radius_m: float
    horizontal_sub_to_over: tuple
    dip_radians: float
    donor_displacement_m: float
    receiver_displacement_m: float


def native_birth_point_port(*, donor_owner, receiver_owner, donor_radial,
        receiver_radial, horizontal_sub_to_over, dip_radians, radius_m,
        donor_displacement_m, receiver_displacement_m, dimension, angular_owner_dofs,
        donor_radial_coupling, receiver_radial_coupling):
    """Build objective virtual-work operators; this does not prove coverage.

    Radial couplings are explicit (index, weight) tuples.  A shared HOT patch
    uses one existing registry index in every interface.  Numerical point
    coordinates have no material area or interface weight assigned here.
    """
    if type(dimension) is not int or dimension <= 0 or type(angular_owner_dofs) is not tuple:
        raise ValueError("Port coordinates require explicit compact dimension and angular owner records")
    if (type(donor_owner) is not int or type(receiver_owner) is not int
            or donor_owner < 0 or receiver_owner < 0
            or any(type(row) is not tuple or len(row) != 2 or type(row[0]) is not int
                or row[0] < 0 for row in angular_owner_dofs)
            or tuple(owner for owner, _ in angular_owner_dofs) != tuple(sorted({owner for owner, _ in angular_owner_dofs}))):
        raise ValueError("Physical owner records must use canonical unique nonnegative literal integers")
    owners = dict(angular_owner_dofs)
    if donor_owner == receiver_owner or donor_owner not in owners or receiver_owner not in owners:
        raise ValueError("Port requires two distinct declared physical owners")
    angular_indices = tuple(index for _, indices in angular_owner_dofs for index in indices)
    if len(set(angular_indices)) != len(angular_indices) or any(type(index) is not int or not 0 <= index < dimension for index in angular_indices):
        raise ValueError("Angular coordinate indices must be distinct literal indices")
    if any(type(indices) is not tuple or len(indices) != 3 for _, indices in angular_owner_dofs):
        raise ValueError("Each angular owner requires three coordinates")
    def unit(vector):
        if type(vector) is not tuple or len(vector) != 3:
            raise ValueError("Port world directions require explicit unit-vector tuples")
        values = np.asarray([_finite(item, "World direction") for item in vector])
        if abs(float(values@values)-1) > 64*np.finfo(float).eps:
            raise ValueError("Port world direction is not unit length")
        return values
    sub, over, horizontal = map(unit, (donor_radial, receiver_radial, horizontal_sub_to_over))
    if abs(float(over@horizontal)) > 64*np.finfo(float).eps:
        raise ValueError("Port horizontal frame must be tangent to the receiving bank")
    radius = _positive(radius_m, "Port radius")
    dip = _positive(dip_radians, "Port dip")
    if dip >= math.pi/2:
        raise ValueError("Port dip must be strictly below pi/2")
    positions = ((radius+_finite(donor_displacement_m, "Donor displacement"))*sub,
        (radius+_finite(receiver_displacement_m, "Receiver displacement"))*over)
    axes = (math.sin(dip)*horizontal+math.cos(dip)*over,
        math.cos(dip)*horizontal-math.sin(dip)*over)
    rows = []
    for coupling in (donor_radial_coupling, receiver_radial_coupling):
        if type(coupling) is not tuple or len({index for index, _ in coupling}) != len(coupling):
            raise ValueError("Radial couplings require unique literal immutable coordinate records")
        if any(type(index) is not int or not 0 <= index < dimension or index in angular_indices
                or type(weight) is not float or not math.isfinite(weight) for index, weight in coupling):
            raise ValueError("Radial coupling overlaps angular coordinates or has invalid weights")
    for axis in axes:
        row = np.zeros(dimension)
        lever = np.cross(positions[0], axis)
        row[list(owners[donor_owner])], row[list(owners[receiver_owner])] = -lever, lever
        for sign, direction, coupling in ((-1, sub, donor_radial_coupling), (1, over, receiver_radial_coupling)):
            for index, weight in coupling:
                row[index] += sign*radius*float(direction@axis)*weight
        rows.append(tuple(map(float, row)))
    return NativeBirthPointPort(donor_owner, receiver_owner, tuple(map(float, positions[0])),
        tuple(map(float, positions[1])), tuple(donor_radial), tuple(receiver_radial),
        tuple(map(float, axes[0])), tuple(map(float, axes[1])), *rows,
        donor_radial_coupling, receiver_radial_coupling, angular_owner_dofs, radius,
        horizontal_sub_to_over, dip, donor_displacement_m, receiver_displacement_m)


def native_birth_point_reactions(port, *, interface_area_m2, traction_pa):
    """Return physical pair forces/couple and their generalized virtual work."""
    if type(port) is not NativeBirthPointPort or type(traction_pa) is not tuple or len(traction_pa) != 2:
        raise ValueError("Birth reactions require an explicit port and traction pair")
    replay = native_birth_point_port(donor_owner=port.donor_owner, receiver_owner=port.receiver_owner,
        donor_radial=port.donor_radial, receiver_radial=port.receiver_radial,
        horizontal_sub_to_over=port.horizontal_sub_to_over, dip_radians=port.dip_radians,
        radius_m=port.radius_m, donor_displacement_m=port.donor_displacement_m,
        receiver_displacement_m=port.receiver_displacement_m, dimension=len(port.normal_operator),
        angular_owner_dofs=port.angular_owner_dofs, donor_radial_coupling=port.donor_radial_coupling,
        receiver_radial_coupling=port.receiver_radial_coupling)
    if _typed_fingerprint(replay) != _typed_fingerprint(port):
        raise ValueError("Born point reactions differ from complete objective world-frame port replay")
    area = _positive(interface_area_m2, "Inclined interface area")
    traction = tuple(_finite(item, "Contact traction") for item in traction_pa)
    force = area*(traction[0]*np.asarray(port.normal)+traction[1]*np.asarray(port.down_dip))
    sub_torque = np.cross(port.donor_position_m, force)
    over_force_torque = np.cross(port.receiver_position_m, -force)
    couple = np.cross(np.asarray(port.receiver_position_m)-np.asarray(port.donor_position_m), force)
    generalized = np.zeros(len(port.normal_operator))
    owners = dict(port.angular_owner_dofs)
    generalized[list(owners[port.donor_owner])] += sub_torque
    generalized[list(owners[port.receiver_owner])] -= sub_torque
    for sign, direction, coupling in ((1, port.donor_radial, port.donor_radial_coupling),
            (-1, port.receiver_radial, port.receiver_radial_coupling)):
        for index, weight in coupling:
            generalized[index] += sign*port.radius_m*float(np.asarray(direction)@force)*weight
    return dict(force_on_sub_n=tuple(map(float, force)), force_on_over_n=tuple(map(float, -force)),
        torque_on_sub_nm=tuple(map(float, sub_torque)), torque_on_over_nm=tuple(map(float, -sub_torque)),
        force_torque_on_over_nm=tuple(map(float, over_force_torque)), carrier_couple_on_over_nm=tuple(map(float, couple)),
        generalized_force=tuple(map(float, generalized)))
