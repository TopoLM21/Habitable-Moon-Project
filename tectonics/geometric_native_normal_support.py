"""Authenticated geometric preflight for the diagnostic normal-column law.

A negative witness concerns the complete source activation face.  It does not
inherit an endpoint force certificate, shorten a column, or authorize material
motion.  Roof exclusion is sufficient even for the union of every bank of the
same owner; no proxy polygon is promoted to a material reservoir.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

from .geometric_native_pose import FractionInterval as I, interval_sqrt, _rounded_interval
from .geometric_native_spherical_solid import _arc_frame, _cross, _dot, interval_cbrt
from .geometric_native_joint_spherical_field import (
    solve_exact_minimum_norm_angular_rate, rotate_joint_physical_depth,
)

ROOT = Path(__file__).resolve().parents[1]
CURVED_PACKET = ROOT / "analysis/geometric_native_curved_equilibrium_validation/release_validation_01/native_control/candidate_packet.json"
JOINT_PACKET = ROOT / "analysis/geometric_native_joint_spherical_validation/release_validation_01/native_control/candidate_packet.json"
CURVED_PACKET_RAW_SHA256 = "0e3554a55a8c93eb8ecbdae8351f3280a2ae4bd8596254950afef7f77b98d2cb"
JOINT_PACKET_RAW_SHA256 = "8580d79ec47263c9f581a646595ab19d87ae927770d082b3016803d4c405ac45"
SOURCE_NAMES = ("before", "current_load", "original_unaccepted_trial_birth_support", "full_reference_face_birth_support")
_AUTHENTICATED_SOURCE_SNAPSHOTS = set()


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _payload(value):
    if type(value) is F:
        return [value.numerator, value.denominator]
    if type(value) is I:
        return {"lo": _payload(value.lo), "hi": _payload(value.hi)}
    if is_dataclass(value):
        return {f.name: _payload(getattr(value, f.name)) for f in fields(value)}
    if type(value) is tuple:
        return [_payload(item) for item in value]
    if type(value) in (str, int, bool) or value is None:
        return value
    raise TypeError("Normal support receipts require immutable exact data")


def _snapshot(value):
    return tuple(getattr(value, f.name) for f in fields(value) if f.name != "source_snapshot_sha256")


def _hash_snapshot(value):
    return hashlib.sha256(_encoded(_payload(_snapshot(value)))).hexdigest()


def _read_fraction(raw):
    if type(raw) is not list or len(raw) != 2 or any(type(x) is not int for x in raw) or raw[1] <= 0:
        raise ValueError("A saved exact scalar requires its numerator and positive denominator")
    return F(*raw)


def _read_interval(raw):
    if type(raw) is not dict or set(raw) != {"lo", "hi"}:
        raise ValueError("A saved outward scalar requires both endpoints")
    return I(_read_fraction(raw["lo"]), _read_fraction(raw["hi"]))


def _require_interval(value, name):
    if type(value) is not I:
        raise ValueError(name + " requires a literal FractionInterval")
    return value


def _precision(value):
    if type(value) is not int or not 64 <= value <= 512:
        raise ValueError("Normal support precision must be an integer from 64 to 512")
    return value


def _square(value):
    return I(0 if value.lo <= 0 <= value.hi else min(value.lo**2, value.hi**2),
             max(value.lo**2, value.hi**2))


@dataclass(frozen=True, slots=True)
class NativeNormalSupportBank:
    component_id: str
    material_id: str
    original_parent_id: str
    owner: int
    material_kind: str
    roof_radial_displacement_m: F
    areal_material_thickness_m: F
    complete_mask_source_json: str


@dataclass(frozen=True, slots=True)
class NativeNormalSupportFamily:
    family_id: str
    donor_component_id: str
    receiver_component_id: str
    native_angle_end_bounds_radians: I
    start_unit_bounds: tuple
    tangent_unit_bounds: tuple
    normal_unit_bounds: tuple
    receiving_normal_orientation: int
    shared_depth_axis_per_m: tuple
    column_lengths_m: tuple
    source_preload_gap_m: F
    complete_carrier_source_json: str


@dataclass(frozen=True, slots=True)
class NativeNormalSupportSource:
    curved_packet_raw_sha256: str
    joint_packet_raw_sha256: str
    curved_packet_seal_sha256: str
    source_support_sha256: str
    source_epoch_sha256: str
    source_epoch_fingerprint_sha256: str
    saved_field_declaration_precision: int
    saved_field_provenance: str
    radius_m: F
    complete_source_dto_json: tuple
    saved_field_json: str
    banks: tuple
    families: tuple
    source_snapshot_sha256: str

    @property
    def complete_three_dimensional_strain_energy_certified(self):
        return False

    @property
    def physical_material_admission_authorized(self):
        return False


@dataclass(frozen=True, slots=True)
class NativeNormalCollarRoofExclusion:
    family_id: str
    source_snapshot_sha256: str
    source_support_sha256: str
    source_epoch_sha256: str
    source_packet_raw_sha256: str
    saved_field_declaration_precision: int
    activation_time_fraction: F
    donor_component_id: str
    receiver_component_id: str
    donor_owner: int
    receiver_owner: int
    donor_material_kind: str
    receiver_material_kind: str
    column_lengths_m: tuple
    source_preload_gap_m: F
    angle_bounds_radians: I
    native_angle_end_bounds_radians: I
    interior_depth_m: I
    common_reference_top_radius_m: F
    donor_material_depth_bounds_m: I
    receiver_material_depth_bounds_m: I
    reference_radius_bounds_m: I
    receiving_radial_normal_bounds: I
    reference_start_direction_bounds: tuple
    reference_tangent_direction_bounds: tuple
    reference_arc_normal_bounds: tuple
    shared_depth_axis_per_m: tuple
    reference_frame_ray_end_bounds_m: tuple
    reference_frame_unit_normal_bounds: tuple
    activation_world_ray_endpoint_bounds_m: tuple
    ray_distance_bounds_m: I
    ray_endpoint_radius_bounds_m: I
    all_same_owner_roofs: tuple
    owner_max_roof_radius_bounds_m: I
    outside_roof_margin_bounds_m: I
    positive_measure_activation_strip_proved: bool
    full_source_column_support_rejected: bool
    precision: int

    @property
    def full_normal_collars_placeable(self):
        return False

    @property
    def normal_columns_clipped(self):
        return False

    @property
    def source_normal_stiffness_reused_as_spatial_energy(self):
        return False

    @property
    def complete_three_dimensional_strain_energy_certified(self):
        return False

    @property
    def physical_material_admission_authorized(self):
        return False

    @property
    def previous_endpoint_force_certificate_inherited(self):
        return False


def _packet(raw, expected_raw, expected_format):
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != expected_raw:
        raise ValueError("The independently authenticated source packet raw bytes changed")
    value = json.loads(raw)
    if value.get("format") != expected_format:
        raise ValueError("The saved source packet has the wrong format")
    body = dict(value)
    seal = body.pop("packet_sha256", None)
    if seal != hashlib.sha256(_encoded(body)).hexdigest():
        raise ValueError("The complete source packet selfseal changed")
    return value


def _material_thickness(component, material):
    if component["material_kind"] == "original_live":
        parents = [p for p in material["journal"]["source_surface"]["fragments"]
                   if p["fragment_id"] == component["original_parent_id"]]
        if len(parents) != 1:
            raise ValueError("The actual bank lost its original material parent")
        parcel = parents[0]["parcel"]
        if parcel["material_id"] != component["material_id"] or parcel["plate"] != component["owner"]:
            raise ValueError("The actual bank differs from its original material allocation")
        crust, cold = tuple(F.from_float(float(x)) for x in parcel["specific_properties"][:2])
    elif component["material_kind"] == "hot_birth":
        births = [p["newborn"] for p in material["endpoint_vacancies"]
                  if p["newborn"] is not None and p["newborn"]["vacancy_domain_id"] == component["component_id"]]
        if len(births) != 1 or births[0]["material_id"] != component["material_id"] or births[0]["owner"] != component["owner"]:
            raise ValueError("The HOT bank lost its actual original birth")
        crust, cold = tuple(_read_fraction(x) for x in births[0]["specific_properties"][:2])
    else:
        raise ValueError("Normal support requires actual live or HOT source material")
    if crust <= 0 or cold < 0:
        raise ValueError("The source bank has invalid areal material amounts")
    return (crust+cold)*1000


@lru_cache(maxsize=1)
def _bind_authenticated_packets(curved_raw, joint_raw):
    from .geometric_native_polygons import NativeArc, native_plane_from_dict, native_ray_from_dict, _plane_values
    curved = _packet(curved_raw, CURVED_PACKET_RAW_SHA256, "native-curved-force-equilibrium-source-packet-1")
    joint = _packet(joint_raw, JOINT_PACKET_RAW_SHA256, "native-joint-spherical-source-and-curved-energy-packet-1")
    reference = curved["saved_joint_reference_field"]
    old_reference = joint["joint_reference_geometry"]
    if any(_encoded(curved[name]) != _encoded(joint[name]) for name in SOURCE_NAMES):
        raise ValueError("The source activation changed a complete prior joint input DTO")
    if (reference["authenticated_previous_reference_packet_raw_sha256"] != JOINT_PACKET_RAW_SHA256
            or reference["declaration_precision"] != 128
            or reference["provenance"] != "One source-bound saved physical-radius reference flow per authentic contact-connected group; radial motion follows saved shear"
            or _encoded(reference["groups"]) != _encoded(old_reference["groups"])):
        raise ValueError("The saved source depth flow was redeclared or substituted")
    support = curved["full_reference_face_birth_support"]
    epoch = curved["before"]["snapshot"]["current_material_epoch"]
    material = epoch["source_material_state"]
    R = F.from_float(float(material["journal"]["source_surface"]["radius_km"]))*1000
    if (support["source_epoch_sha256"] != epoch["epoch_sha256"]
            or reference["source_support_sha256"] != support["support_sha256"]
            or reference["source_epoch_fingerprint_sha256"] != support["complete_source_epoch_fingerprint_sha256"]):
        raise ValueError("The source bank epoch/support binding is stale")
    tops = {}
    for identity, value in reference["top_radial_displacements_m"]:
        bounds = _read_interval(value)
        if bounds.width or identity in tops:
            raise ValueError("Each saved source roof must be one literal declaration")
        tops[identity] = bounds.lo
    if set(tops) != {c["component_id"] for c in epoch["components"]}:
        raise ValueError("The saved source roofs must cover all actual source banks")
    banks = tuple(NativeNormalSupportBank(c["component_id"], c["material_id"], c["original_parent_id"],
        c["owner"], c["material_kind"], tops[c["component_id"]], _material_thickness(c, material),
        _encoded(c).decode()) for c in sorted(epoch["components"], key=lambda c:c["component_id"]))
    frame_rows = []
    for carrier in support["interface_carriers"]:
        arc_raw = carrier["exact_initial_shared_arc"]
        arc = NativeArc(native_ray_from_dict(arc_raw["start"]), native_ray_from_dict(arc_raw["end"]),
                        native_plane_from_dict(arc_raw["plane"]), arc_raw["length_radians"])
        start, tangent, _, angle = _arc_frame(arc, 128)
        inward = _plane_values(native_plane_from_dict(carrier["receiver_inward_plane"]), 128)[1]
        inward = tuple(x/interval_sqrt(_dot(inward, inward), 136) for x in inward)
        constraint_tangent = _cross(inward, start)
        midpoint_constraint = tuple(x.midpoint for x in constraint_tangent)
        declaration = carrier["declaration"]
        cot = F.from_float(math.cos(declaration["dip_radians"]))/F.from_float(math.sin(declaration["dip_radians"]))
        frame_rows.append((carrier, start, tangent, angle, inward, (midpoint_constraint, -cot/R)))
    families = []
    for carrier, start, tangent, angle, inward, _ in frame_rows:
        declaration = carrier["declaration"]
        identity = declaration["family_id"]
        source_angle = _read_interval(carrier["native_angle_bounds_radians"])
        if not 0 < angle.lo <= source_angle.lo <= source_angle.hi <= angle.hi:
            raise ValueError("The full saved native endpoint is outside its exact inherited arc")
        groups = [g for g in reference["groups"] if identity in g["family_ids"]]
        if len(groups) != 1:
            raise ValueError("The source family lacks one actual saved continuity group")
        group = groups[0]
        constraints = tuple(row[-1] for row in frame_rows if row[0]["declaration"]["family_id"] in group["family_ids"])
        A = solve_exact_minimum_norm_angular_rate(constraints)
        if any(not alias.lo <= value <= alias.hi for value, alias in zip(A,
                tuple(_read_interval(x) for x in group["angular_rate_per_physical_volume_depth_m"]))):
            raise ValueError("The literal128 source depth flow differs from its saved aliases")
        normal = _cross(start, tangent)
        orientation = _dot(normal, inward)
        sign = 1 if orientation.lo > 0 else -1 if orientation.hi < 0 else 0
        if not sign:
            raise ValueError("The source receiving normal has unresolved orientation")
        columns = declaration["parameters"]["columns"]
        families.append(NativeNormalSupportFamily(identity, declaration["donor_component_id"],
            declaration["receiver_component_id"], source_angle, start, tangent, normal, sign, A,
            (F.from_float(columns["sub_length_m"]), F.from_float(columns["over_length_m"])),
            F.from_float(declaration["birth_gap_free_m"]), _encoded(carrier).decode()))
    values = (CURVED_PACKET_RAW_SHA256, JOINT_PACKET_RAW_SHA256, curved["packet_sha256"],
        support["support_sha256"], epoch["epoch_sha256"], reference["source_epoch_fingerprint_sha256"],
        128, reference["provenance"], R, tuple((name, _encoded(curved[name]).decode()) for name in SOURCE_NAMES),
        _encoded(reference).decode(), banks, tuple(families))
    result = NativeNormalSupportSource(*values, "")
    result = NativeNormalSupportSource(*values, _hash_snapshot(result))
    _AUTHENTICATED_SOURCE_SNAPSHOTS.add(result.source_snapshot_sha256)
    return result


def load_native_normal_support_source(*, curved_packet_bytes=None, joint_packet_bytes=None):
    """Read fixed authenticated artifacts, without rebuilding the genuine case."""
    curved = CURVED_PACKET.read_bytes() if curved_packet_bytes is None else curved_packet_bytes
    joint = JOINT_PACKET.read_bytes() if joint_packet_bytes is None else joint_packet_bytes
    return validate_native_normal_support_source(_bind_authenticated_packets(curved, joint))


def validate_native_normal_support_source(source):
    if type(source) is not NativeNormalSupportSource:
        raise ValueError("A complete immutable authenticated source receipt is required")
    if (source.source_snapshot_sha256 != _hash_snapshot(source)
            or source.source_snapshot_sha256 not in _AUTHENTICATED_SOURCE_SNAPSHOTS):
        raise ValueError("A re-sealed or altered source cannot replace actual packet authentication")
    return source


def normal_ray_radius_bounds(radius_bounds_m, radial_normal_bounds, distance_bounds_m, *, precision=160):
    """Enclose |r u+ξ n|, retaining the common exact unit-frame identities."""
    _precision(precision)
    r = _require_interval(radius_bounds_m, "Reference radius")
    n = _require_interval(radial_normal_bounds, "Radial unit-normal projection")
    xi = _require_interval(distance_bounds_m, "Normal-ray distance")
    if r.lo <= 0 or n.lo < -1 or n.hi > 1:
        raise ValueError("The normal ray requires positive radii and a physical unit-normal projection")
    squared = _square(r)+2*r*xi*n+_square(xi)
    if squared.lo < 0:
        raise ValueError("The normal-ray interval needs subdivision to prove a nonnegative squared radius")
    return interval_sqrt(squared, precision)


def normal_ray_outer_roof_exit_bounds(radius_bounds_m, radial_normal_bounds, outer_roof_radius_bounds_m, *, precision=160):
    """Bound the first outward ray intersection with an enclosing source roof."""
    _precision(precision)
    r = _require_interval(radius_bounds_m, "Reference radius")
    n = _require_interval(radial_normal_bounds, "Outward radial unit-normal projection")
    roof = _require_interval(outer_roof_radius_bounds_m, "Enclosing outer roof")
    if r.lo <= 0 or not 0 < n.lo <= n.hi <= 1 or roof.lo < r.hi:
        raise ValueError("The roof exit requires an inward source point and a strictly outward normal ray")
    radicand = _square(r*n)+_square(roof)-_square(r)
    if radicand.lo < 0:
        raise ValueError("The roof exit interval needs subdivision")
    result = -r*n+interval_sqrt(radicand, precision)
    return I(max(F(0), result.lo), result.hi)


def bound_native_normal_collar_roof_exclusion(source, family_id, *, depth_below_shared_top_bounds_m=I(F(100),F(500)), precision=160):
    """Certify a positive-area source strip whose literal receiver rods exit all owner banks."""
    from .geometric_native_active_contact_line_work import _trig_ranges_signed
    source = validate_native_normal_support_source(source)
    _precision(precision)
    depth = _require_interval(depth_below_shared_top_bounds_m, "Interior activation depth")
    matches = tuple(row for row in source.families if row.family_id == family_id)
    if type(family_id) is not str or len(matches) != 1:
        raise ValueError("The source needs one unique actual carrier family")
    family = matches[0]
    banks = {bank.component_id:bank for bank in source.banks}
    donor, receiver = banks[family.donor_component_id], banks[family.receiver_component_id]
    R = source.radius_m
    donor_top, receiver_top = R+donor.roof_radial_displacement_m, R+receiver.roof_radial_displacement_m
    top = min(donor_top, receiver_top)
    donor_depth = depth+(donor_top**3-top**3)/(3*R**2)
    receiver_depth = depth+(receiver_top**3-top**3)/(3*R**2)
    if (not 0 < depth.lo < depth.hi
            or donor_depth.hi >= donor.areal_material_thickness_m
            or receiver_depth.hi >= receiver.areal_material_thickness_m):
        raise ValueError("The negative witness requires a strictly interior positive-depth strip in both actual banks")
    angle = I(0, family.native_angle_end_bounds_radians.hi)
    sine, cosine = _trig_ranges_signed(angle.lo, angle.hi, precision)
    direction = tuple(a*cosine+b*sine for a,b in zip(family.start_unit_bounds, family.tangent_unit_bounds))
    tangent = tuple(-a*sine+b*cosine for a,b in zip(family.start_unit_bounds, family.tangent_unit_bounds))
    cubes = I(top**3)-3*R**2*depth
    r = interval_cbrt(cubes, precision)
    P, B = R**2/r, _square(r)*_dot(family.shared_depth_axis_per_m, tangent)
    J = interval_sqrt(_square(P)+_square(B), precision)
    sign = family.receiving_normal_orientation
    nr = -sign*B/J
    xi = I(family.column_lengths_m[1])
    endpoint = normal_ray_radius_bounds(r, nr, xi, precision=precision)
    roofs = tuple((bank.component_id, I(R+bank.roof_radial_displacement_m))
                  for bank in source.banks if bank.owner == receiver.owner)
    roof = I(max(bounds.lo for _,bounds in roofs), max(bounds.hi for _,bounds in roofs))
    margin = endpoint-roof
    if margin.lo <= 0:
        raise ValueError("This source strip does not strictly exclude the complete receiver normal collar")
    normal = tuple(sign*(P*n-B*u)/J for n,u in zip(family.normal_unit_bounds,direction))
    ray = tuple(_rounded_interval(r*u+xi*n, precision) for u,n in zip(direction,normal))
    tau = (I(R**3)-cubes)/(3*R**2)
    world_ray = rotate_joint_physical_depth(ray, family.shared_depth_axis_per_m, tau, precision=precision, order=8)
    return NativeNormalCollarRoofExclusion(family.family_id, source.source_snapshot_sha256,
        source.source_support_sha256, source.source_epoch_sha256, source.curved_packet_raw_sha256,
        source.saved_field_declaration_precision, F(0), donor.component_id, receiver.component_id,
        donor.owner, receiver.owner, donor.material_kind, receiver.material_kind,
        family.column_lengths_m, family.source_preload_gap_m, angle, family.native_angle_end_bounds_radians,
        depth, top, donor_depth, receiver_depth, r, nr, family.start_unit_bounds,
        family.tangent_unit_bounds, family.normal_unit_bounds, family.shared_depth_axis_per_m,
        ray, normal, world_ray, xi, endpoint, roofs, roof, margin, family.native_angle_end_bounds_radians.lo>0, True, precision)


def preflight_native_normal_support(source=None, *, depth_below_shared_top_bounds_m=I(F(100),F(500)), precision=160):
    source = load_native_normal_support_source() if source is None else validate_native_normal_support_source(source)
    return tuple(bound_native_normal_collar_roof_exclusion(source, row.family_id,
        depth_below_shared_top_bounds_m=depth_below_shared_top_bounds_m, precision=precision)
        for row in source.families)


def validate_native_normal_support_certificate(certificate, *, source):
    if type(certificate) is not NativeNormalCollarRoofExclusion:
        raise ValueError("The normal support certificate requires its immutable typed receipt")
    replay = bound_native_normal_collar_roof_exclusion(source, certificate.family_id,
        depth_below_shared_top_bounds_m=certificate.interior_depth_m, precision=certificate.precision)
    if certificate != replay:
        raise ValueError("The source normal support certificate differs from its full semantic replay")
    return replay


def native_normal_support_to_dict(source, certificates):
    source = validate_native_normal_support_source(source)
    if type(certificates) is not tuple or tuple(row.family_id for row in certificates) != tuple(row.family_id for row in source.families):
        raise ValueError("Normal support output must retain every actual carrier family once")
    rows = tuple(validate_native_normal_support_certificate(row, source=source) for row in certificates)
    return {"format":"native-normal-collar-source-exclusion-1", "source":_payload(source),
        "certificates":_payload(rows), "status":"SOURCE_COLUMN_SUPPORT_REJECTED",
        "normal_columns_clipped":False, "complete_three_dimensional_strain_energy_certified":False,
        "physical_material_admission_authorized":False, "previous_endpoint_force_certificate_inherited":False}


def inverse_joint_spherical_material_depth_bounds(*, radius_m, top_radial_displacement_m, point_radius_bounds_m):
    """Exact radial inverse of the saved joint embedding, without mask membership.

    A negative depth certifies exterior placement above this source roof.
    Positive depth alone never proves membership in the angular material mask.
    """
    if type(radius_m) is not F or type(top_radial_displacement_m) is not F or radius_m <= 0:
        raise ValueError("The radial inverse requires literal physical Fraction declarations")
    radius = _require_interval(point_radius_bounds_m, "Physical point radius")
    if radius.lo <= 0 or radius_m+top_radial_displacement_m <= 0:
        raise ValueError("The radial inverse requires positive physical radii")
    return (I((radius_m+top_radial_displacement_m)**3)-_square(radius)*radius)/(3*radius_m**2)
