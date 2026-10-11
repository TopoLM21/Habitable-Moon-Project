"""Source-bound material-depth coverage and independent normal geometry.

The force carrier is the existing declared depth strip.  The remaining
physical face is retained as geometry and needs a separate separation proof;
neither its area nor its spring energy is silently added to paid activation.
Source areal-volume thickness is a material depth coordinate: it is not the
radial depth or metric face area after a curved spherical embedding.

Normal columns can be placed only after actual anchor positions and a
material-owned contact-zone rest length have been supplied independently.
The negative free anchor gap is then distributed over the original three
compliances.  It is never used as a negative physical void or as a fault-zone
rest length.  This one-dimensional construction is not a deformation field
on the complete spherical material mask.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction as F
import hashlib
import json
import math
import re

from .geometric_native_pose import FractionInterval as I
from .geometric_solid_contact_closure import (
    SolidNormalClosure, compute_series_normal_closure, geometry_to_anchor,
    require_contact_zone_compaction,
)


def _fraction(value, name, *, positive=False):
    if type(value) is not F or (positive and value <= 0):
        raise ValueError(name + " requires an exact" + (" positive" if positive else "") + " Fraction")
    return value


def _text(value, name):
    if type(value) is not str or not value.strip():
        raise ValueError(name + " requires explicit nonempty source text")
    return value


def _sha(value, name):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(name + " requires a literal SHA-256 source binding")
    return value


@dataclass(frozen=True, slots=True)
class PhysicalFaceCoverage:
    """Coverage in the source reference-depth chart, not curved force metric.

    The legacy ``physical_face`` properties refer to that local chart's
    complete material depth.  Spherical radius/depth and inclined face area
    need the separately declared actual embedding and force integration.
    """
    donor_component_id: str
    receiver_component_id: str
    source_epoch_sha256: str
    birth_support_sha256: str
    donor_thickness_m: F
    receiver_thickness_m: F
    force_contact_height_m: F
    inclined_measure_density_m2_per_radian: F
    native_angle_bounds: I
    provenance: str

    def __post_init__(self):
        for name in ("donor_component_id", "receiver_component_id", "provenance"):
            _text(getattr(self, name), name)
        if self.donor_component_id == self.receiver_component_id:
            raise ValueError("A physical face requires distinct source material components")
        for name in ("source_epoch_sha256", "birth_support_sha256"):
            _sha(getattr(self, name), name)
        for name in ("donor_thickness_m", "receiver_thickness_m", "force_contact_height_m",
                     "inclined_measure_density_m2_per_radian"):
            _fraction(getattr(self, name), name, positive=True)
        if type(self.native_angle_bounds) is not I or self.native_angle_bounds.lo <= 0:
            raise ValueError("The actual carrier needs a proved positive native angle interval")
        if self.force_contact_height_m > self.common_physical_face_height_m:
            raise ValueError("The declared force strip exceeds its common physical material face")

    @property
    def common_physical_face_height_m(self):
        return min(self.donor_thickness_m, self.receiver_thickness_m)

    @property
    def common_material_depth_m(self):
        return min(self.donor_thickness_m, self.receiver_thickness_m)

    @property
    def active_depth_bounds_m(self):
        return I(0, self.force_contact_height_m)

    @property
    def uncovered_depth_parts_m(self):
        height = self.common_physical_face_height_m
        return () if height == self.force_contact_height_m else (I(self.force_contact_height_m, height),)

    @property
    def whole_physical_face_force_coverage(self):
        """Complete reference-prism depth; no curved-metric assertion."""
        return self.force_contact_height_m == self.common_physical_face_height_m

    @property
    def whole_material_depth_force_reference_coverage(self):
        return self.force_contact_height_m == self.common_material_depth_m

    @property
    def actual_curved_face_force_metric_certified(self):
        return False

    @property
    def active_interface_area_bounds_m2(self):
        return self.inclined_measure_density_m2_per_radian*self.native_angle_bounds

    @property
    def full_height_reference_measure_density_m2_per_radian(self):
        return (self.inclined_measure_density_m2_per_radian
                *self.common_physical_face_height_m/self.force_contact_height_m)

    @property
    def uncovered_reference_measure_density_m2_per_radian(self):
        return self.full_height_reference_measure_density_m2_per_radian-self.inclined_measure_density_m2_per_radian

    @property
    def activation_paid_again(self):
        return False

    @property
    def added_activation_energy_j(self):
        return F(0)

    @property
    def physical_material_admission_authorized(self):
        return False

    @property
    def material_admission_authorized(self):
        return False


def declare_physical_face_coverage(**declaration):
    """Partition the original loaded strip and retained remaining face.

    A declaration supplies a geometric model, not an authenticated native
    source.  Use ``native_contact_face_coverage_from_packet`` to recover these
    dimensions from an independently trusted immutable source packet.
    """
    return PhysicalFaceCoverage(**declaration)


def require_complete_face_force_coverage(coverage):
    """Require the complete material-reference depth, not curved face metric."""
    if type(coverage) is not PhysicalFaceCoverage:
        raise ValueError("A complete physical contact face requires its exact coverage declaration")
    # Replay construction also rejects forged dataclasses/type substitutions.
    PhysicalFaceCoverage(**{name: getattr(coverage, name) for name in coverage.__dataclass_fields__})
    if not coverage.whole_physical_face_force_coverage:
        raise ValueError("The existing force carrier does not cover the complete physical face")
    return coverage.common_physical_face_height_m


@dataclass(frozen=True, slots=True)
class UnloadedFaceSeparationWitness:
    coverage: PhysicalFaceCoverage
    angle_bounds_radians: I
    depth_bounds_m: I
    time_fraction_bounds: I
    physical_face_void_bounds_m: I
    geometry_source_id: str

    @property
    def additional_contact_force(self):
        return F(0)

    @property
    def additional_activation_energy_j(self):
        return F(0)

    @property
    def material_admission_authorized(self):
        return False


def require_unloaded_face_separation(coverage, *, angle_bounds_radians,
        depth_bounds_m, time_fraction_bounds, physical_face_void_bounds_m,
        geometry_source_id):
    """Require full-angle/depth/clock geometry before calling a tail unloaded.

    The supplied interval must come from the caller's independent physical
    geometry.  This helper checks exact coverage and no interpenetration; it
    does not authenticate a curved deformation field from scalar preload.
    Touching is allowed on the zero-traction unilateral branch.
    """
    if type(coverage) is not PhysicalFaceCoverage:
        raise ValueError("A retained unloaded face requires its exact source coverage")
    PhysicalFaceCoverage(**{name: getattr(coverage, name) for name in coverage.__dataclass_fields__})
    if len(coverage.uncovered_depth_parts_m) != 1:
        raise ValueError("This carrier has no retained unloaded depth strip")
    if any(type(value) is not I for value in (angle_bounds_radians, depth_bounds_m,
            time_fraction_bounds, physical_face_void_bounds_m)):
        raise ValueError("Unloaded face geometry must retain exact interval bounds")
    if (angle_bounds_radians.lo != 0 or angle_bounds_radians.hi < coverage.native_angle_bounds.hi
            or depth_bounds_m != coverage.uncovered_depth_parts_m[0]
            or time_fraction_bounds != I(0, 1)):
        raise ValueError("The retained physical face needs its whole native angle, depth and clock")
    if physical_face_void_bounds_m.lo < 0:
        raise ValueError("A retained unloaded physical face has unexcluded interpenetration")
    _text(geometry_source_id, "Independent whole-face geometry source")
    return UnloadedFaceSeparationWitness(coverage, angle_bounds_radians, depth_bounds_m,
        time_fraction_bounds, physical_face_void_bounds_m, geometry_source_id)


@dataclass(frozen=True, slots=True)
class SolidContactNormalPlacement:
    normal: SolidNormalClosure
    donor_anchor_m: F
    receiver_anchor_m: F
    donor_column_end_m: F
    receiver_column_end_m: F
    donor_physical_face_m: F
    receiver_physical_face_m: F
    physical_face_void_m: F
    contact_zone_start_m: F
    contact_zone_end_m: F
    contact_zone_owner: str
    geometry_source_id: str
    anchor_balance_residual_m: F
    donor_column_normal_jacobian: F
    receiver_column_normal_jacobian: F
    contact_zone_normal_jacobian: F
    total_energy_j_m2: F
    added_external_work_j_m2: F = F(0)
    activation_paid_again: bool = False

    @property
    def material_admission_authorized(self):
        return False

    @property
    def whole_spherical_solid_embedding_certified(self):
        return False

    @property
    def contact_zone_rest_normal_thickness_m(self):
        return self.normal.contact_zone_rest_normal_thickness_m

    @property
    def donor_column_density_ratio(self):
        return 1/self.donor_column_normal_jacobian

    @property
    def receiver_column_density_ratio(self):
        return 1/self.receiver_column_normal_jacobian

    @property
    def contact_zone_density_ratio(self):
        return 1/self.contact_zone_normal_jacobian


def place_solid_contact_normal_segments(*, normal, donor_anchor_m,
        receiver_anchor_m, contact_zone_owner, geometry_source_id):
    """Place all three independently declared compliant material segments.

    Anchor coordinates point from donor to receiver and are physical source
    geometry, not inferred outputs.  Their distance must be
    ``L_sub + L_over + L_zone + g_free``.  The fault-zone rest length must
    already be declared by ``require_contact_zone_compaction`` and assigned
    to one actual material bank.  Assigning it here never creates material.

    Endpoint energies are observed unchanged.  Per-segment volume Jacobians
    expose the density changes required by the declared normal compaction;
    this routine does not allocate a spherical volume or add external work.
    """
    if type(normal) is not SolidNormalClosure:
        raise ValueError("Physical normal placement requires its exact column/contact closure")
    if normal.contact_zone_rest_normal_thickness_m is None:
        raise ValueError("Physical normal placement requires independently declared contact-zone rest thickness")
    replay = compute_series_normal_closure(columns=normal.columns, law=normal.law,
        free_gap_m=normal.free_gap_m)
    replay = require_contact_zone_compaction(replay,
        contact_zone_rest_normal_thickness_m=normal.contact_zone_rest_normal_thickness_m)
    if replay != normal:
        raise ValueError("Physical normal placement differs from the declared source pressure partition")
    donor = _fraction(donor_anchor_m, "Independent donor normal anchor")
    receiver = _fraction(receiver_anchor_m, "Independent receiver normal anchor")
    if type(contact_zone_owner) is not str or contact_zone_owner not in ("donor", "receiver"):
        raise ValueError("The existing material-owned contact zone needs its explicit donor or receiver owner")
    _text(geometry_source_id, "Independent normal anchor geometry source")
    sub_length = F.from_float(normal.columns.sub_length_m)
    over_length = F.from_float(normal.columns.over_length_m)
    zone_length = normal.contact_zone_rest_normal_thickness_m
    residual = receiver-donor-sub_length-over_length-zone_length-normal.free_gap_m
    if residual != 0:
        raise ValueError("Independent physical anchor positions do not reproduce the paid free anchor gap")
    donor_end = donor+normal.deformed_sub_column_length_m
    receiver_end = receiver-normal.deformed_over_column_length_m
    if contact_zone_owner == "donor":
        zone_start = donor_end
        zone_end = zone_start+normal.deformed_contact_zone_thickness_m
        donor_face, receiver_face = zone_end, receiver_end
    else:
        zone_end = receiver_end
        zone_start = zone_end-normal.deformed_contact_zone_thickness_m
        donor_face, receiver_face = donor_end, zone_start
    void = receiver_face-donor_face
    geometry_to_anchor(normal=normal, physical_face_void_m=void,
        sub_column_shortening_m=sub_length-(donor_end-donor),
        over_column_shortening_m=over_length-(receiver-receiver_end),
        fault_indentation_m=zone_length-(zone_end-zone_start))
    return SolidContactNormalPlacement(normal, donor, receiver, donor_end, receiver_end,
        donor_face, receiver_face, void, zone_start, zone_end, contact_zone_owner,
        geometry_source_id, residual,
        normal.deformed_sub_column_length_m/sub_length,
        normal.deformed_over_column_length_m/over_length,
        normal.deformed_contact_zone_thickness_m/zone_length,
        normal.total_energy_j_m2)


@dataclass(frozen=True, slots=True)
class SolidContactRobinPlacement:
    """The old contact spring represented by existing-bank boundary modes.

    This alternative geometry uses no extra material layer.  Its two mode
    amplitudes are an explicit partition of the same fault indentation, not
    a change to the bulk elastic-column pressure partition.  Tangential
    compensation supplies an exact local volume-preserving placement; its
    gluing to a spherical mask still needs an actual spatial field.
    """
    normal: SolidNormalClosure
    donor_anchor_m: F
    receiver_anchor_m: F
    donor_bulk_column_end_m: F
    receiver_bulk_column_end_m: F
    donor_physical_face_m: F
    receiver_physical_face_m: F
    physical_face_void_m: F
    fault_indentation_donor_fraction: F
    donor_robin_boundary_shortening_m: F
    receiver_robin_boundary_shortening_m: F
    donor_normal_stretch: F
    receiver_normal_stretch: F
    geometry_source_id: str
    anchor_balance_residual_m: F
    total_energy_j_m2: F
    added_external_work_j_m2: F = F(0)
    activation_paid_again: bool = False

    @property
    def donor_local_deformation_gradient(self):
        alpha = self.donor_normal_stretch
        return ((alpha, F(0), F(0)), (F(0), 1/alpha, F(0)), (F(0), F(0), F(1)))

    @property
    def receiver_local_deformation_gradient(self):
        alpha = self.receiver_normal_stretch
        return ((alpha, F(0), F(0)), (F(0), 1/alpha, F(0)), (F(0), F(0), F(1)))

    @property
    def donor_local_volume_jacobian(self):
        return F(1)

    @property
    def receiver_local_volume_jacobian(self):
        return F(1)

    @property
    def additional_material_volume_m3(self):
        return F(0)

    @property
    def material_admission_authorized(self):
        return False

    @property
    def whole_spherical_solid_embedding_certified(self):
        return False

    @property
    def force_area_interpretation(self):
        return "unchanged constitutive reference area; compensated physical area is a separate spatial field"


def place_solid_contact_robin_segments(*, normal, donor_anchor_m,
        receiver_anchor_m, fault_indentation_donor_fraction, geometry_source_id):
    """Place a declared zero-extra-volume Robin boundary mode in old banks.

    This is an explicitly different geometry interpretation from a separate
    finite fault-zone material layer.  The supplied closure must therefore
    have no contact-zone rest thickness.  The existing fault spring's energy
    is retained as the boundary mode's energy, while each original bulk
    shortening still equals its own pressure times compliance.

    A fraction ``lambda`` of the contact indentation retreats the donor face
    into its existing column and ``1-lambda`` retreats the receiver face.
    Both complete physical column maps must remain one-to-one.  For a local
    normal/strike/depth material chart the declared stretch is
    ``diag(alpha,1/alpha,1)``; its determinant is exactly one.  No transverse
    mode is claimed to be independently present in the old source model.
    It is this new placement model's explicit volume compensation.
    """
    if type(normal) is not SolidNormalClosure:
        raise ValueError("Robin placement requires the exact original series closure")
    if normal.contact_zone_rest_normal_thickness_m is not None:
        raise ValueError("Robin boundary placement cannot also allocate a separate contact-zone material layer")
    replay = compute_series_normal_closure(columns=normal.columns, law=normal.law,
        free_gap_m=normal.free_gap_m)
    if replay != normal:
        raise ValueError("Robin placement differs from the original source pressure partition")
    donor = _fraction(donor_anchor_m, "Independent donor normal anchor")
    receiver = _fraction(receiver_anchor_m, "Independent receiver normal anchor")
    fraction = _fraction(fault_indentation_donor_fraction, "Declared donor Robin indentation fraction")
    if not 0 <= fraction <= 1:
        raise ValueError("A Robin indentation partition must lie inside [0,1]")
    _text(geometry_source_id, "Independent Robin anchor geometry source")
    sub_length = F.from_float(normal.columns.sub_length_m)
    over_length = F.from_float(normal.columns.over_length_m)
    residual = receiver-donor-sub_length-over_length-normal.free_gap_m
    if residual:
        raise ValueError("Independent Robin anchor geometry does not reproduce the paid free anchor gap")
    donor_mode = fraction*normal.fault_indentation_m
    receiver_mode = (1-fraction)*normal.fault_indentation_m
    donor_bulk_end = donor+normal.deformed_sub_column_length_m
    receiver_bulk_end = receiver-normal.deformed_over_column_length_m
    donor_face = donor_bulk_end-donor_mode
    receiver_face = receiver_bulk_end+receiver_mode
    donor_alpha = (donor_face-donor)/sub_length
    receiver_alpha = (receiver-receiver_face)/over_length
    if donor_alpha <= 0 or receiver_alpha <= 0:
        raise ValueError("The Robin boundary mode consumes an existing normal material column")
    void = receiver_face-donor_face
    geometry_to_anchor(normal=normal, physical_face_void_m=void,
        sub_column_shortening_m=sub_length-(donor_bulk_end-donor),
        over_column_shortening_m=over_length-(receiver-receiver_bulk_end),
        fault_indentation_m=donor_mode+receiver_mode)
    return SolidContactRobinPlacement(normal, donor, receiver, donor_bulk_end,
        receiver_bulk_end, donor_face, receiver_face, void, fraction,
        donor_mode, receiver_mode, donor_alpha, receiver_alpha,
        geometry_source_id, residual, normal.total_energy_j_m2)


def robin_column_material_point(placement, *, bank, normal_coordinate_m,
        strike_coordinate_m, depth_coordinate_m):
    """Map a complete old-column material point with positive exact rank.

    The normal coordinate runs from that bank's actual anchor to its face.
    Strike and depth are material coordinates in the explicit local chart;
    their admitted ranges and global curved gluing belong to the caller.
    """
    if type(placement) is not SolidContactRobinPlacement:
        raise ValueError("A Robin material point needs its exact declared placement")
    repeated = place_solid_contact_robin_segments(normal=placement.normal,
        donor_anchor_m=placement.donor_anchor_m, receiver_anchor_m=placement.receiver_anchor_m,
        fault_indentation_donor_fraction=placement.fault_indentation_donor_fraction,
        geometry_source_id=placement.geometry_source_id)
    if repeated != placement:
        raise ValueError("The Robin material point differs from its full geometry replay")
    if type(bank) is not str or bank not in ("donor", "receiver"):
        raise ValueError("A Robin material point retains its actual bank label")
    coordinate, strike, depth = (_fraction(value, name) for value, name in
        ((normal_coordinate_m, "Normal material coordinate"),
         (strike_coordinate_m, "Strike material coordinate"),
         (depth_coordinate_m, "Depth material coordinate")))
    length = F.from_float(getattr(placement.normal.columns,
        "sub_length_m" if bank == "donor" else "over_length_m"))
    if not 0 <= coordinate <= length:
        raise ValueError("The Robin material point lies outside the complete existing normal column")
    alpha = placement.donor_normal_stretch if bank == "donor" else placement.receiver_normal_stretch
    normal_point = (placement.donor_anchor_m+alpha*coordinate if bank == "donor"
        else placement.receiver_anchor_m-alpha*coordinate)
    return normal_point, strike/alpha, depth


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _ratio(raw):
    if type(raw) is not list or len(raw) != 2 or any(type(value) is not int for value in raw) or raw[1] <= 0:
        raise ValueError("The immutable packet lost a literal exact rational")
    result = F(*raw)
    if [result.numerator, result.denominator] != raw:
        raise ValueError("The immutable packet lost a canonical exact rational")
    return result


def native_contact_face_coverage_from_packet(packet, family_id, *, expected_packet_sha256):
    """Recover source thickness and original paid carrier without fresh solves.

    ``expected_packet_sha256`` is the independently trusted packet body seal.
    A self-consistent edited packet is insufficient.  The complete original
    source and first material epoch are already embedded in that sealed DTO.
    This read adapter adds no motion, force, state or acceptance authority.
    """
    _sha(expected_packet_sha256, "Independent genuine force packet seal")
    _text(family_id, "Actual born family ID")
    if type(packet) is not dict or packet.get("format") != "native-born-finite-equilibrium-source-packet-1":
        raise ValueError("Coverage requires the complete genuine finite-force source packet")
    body = {name: value for name, value in packet.items() if name != "packet_sha256"}
    if packet.get("packet_sha256") != expected_packet_sha256 or hashlib.sha256(_encoded(body)).hexdigest() != expected_packet_sha256:
        raise ValueError("The contact-face packet differs from its independently trusted complete source")
    support = packet["birth_support"]
    epoch = packet["before"]["snapshot"]["current_material_epoch"]
    if epoch["epoch_sha256"] != support["source_epoch_sha256"]:
        raise ValueError("The contact-face support differs from its actual before material epoch")
    matches = [row for row in support["interface_carriers"] if row["declaration"]["family_id"] == family_id]
    if len(matches) != 1:
        raise ValueError("The genuine source has no unique requested contact-face carrier")
    carrier = matches[0]
    components = {row["component_id"]: row for row in epoch["components"]}
    state = epoch["source_material_state"]
    parents = state["journal"]["source_surface"]["fragments"]
    births = state["endpoint_vacancies"]
    def thickness(side):
        component = components[side["component_id"]]
        if any(component[name] != side[name] for name in ("component_id", "material_id", "material_kind", "owner")):
            raise ValueError("A carrier bank differs from its actual full source material component")
        if component["original_parent_id"] != side["parent_id"]:
            raise ValueError("A carrier bank lost its original source parent")
        if component["material_kind"] == "hot_birth":
            found = [row["newborn"] for row in births if row.get("newborn") is not None
                and row["newborn"]["material_id"] == component["material_id"]]
            if len(found) != 1 or found[0]["owner"] != component["owner"]:
                raise ValueError("A HOT bank has no unique actual material birth")
            crust, cold = map(_ratio, found[0]["specific_properties"][:2])
        elif component["material_kind"] == "original_live":
            found = [row for row in parents if row["fragment_id"] == component["original_parent_id"]]
            if (len(found) != 1 or found[0]["parcel"]["material_id"] != component["material_id"]
                    or found[0]["parcel"]["plate"] != component["owner"]):
                raise ValueError("A live bank has no unique actual original material parent")
            values = found[0]["parcel"]["specific_properties"][:2]
            if any(type(value) is not float or not math.isfinite(value) for value in values):
                raise ValueError("Original material thickness lost its literal saved float source")
            crust, cold = map(F.from_float, values)
        else:
            raise ValueError("The native face needs actual live or HOT material")
        if crust <= 0 or cold < 0:
            raise ValueError("Actual source material thickness must be positive")
        return (crust+cold)*1000
    angles = carrier["native_angle_bounds_radians"]
    height = carrier["declaration"]["vertical_contact_height_m"]
    if type(height) is not float or not math.isfinite(height):
        raise ValueError("The existing force carrier height lost its literal declaration")
    return PhysicalFaceCoverage(carrier["donor"]["component_id"], carrier["receiver"]["component_id"],
        support["source_epoch_sha256"], support["support_sha256"], thickness(carrier["donor"]),
        thickness(carrier["receiver"]), F.from_float(height),
        _ratio(carrier["inclined_measure_density_m2_per_radian"]),
        I(_ratio(angles["lo"]), _ratio(angles["hi"])),
        "Exact complete trusted native force packet; source material amounts and unchanged paid force strip")


def native_contact_face_coverage_from_support(support, family_id):
    """Bind reference-depth coverage to complete genuine typed native support."""
    from .geometric_native_birth_contact_support import (
        NativeBirthContactSupport, native_birth_contact_support_to_dict,
    )
    from .geometric_native_solid_layer_placement import native_component_solid_thickness_m
    if type(support) is not NativeBirthContactSupport:
        raise ValueError("Native face coverage requires the complete typed source support")
    _text(family_id, "Actual born family ID")
    native_birth_contact_support_to_dict(support)
    matches = [row for row in support.interface_carriers if row.declaration.family_id == family_id]
    if len(matches) != 1:
        raise ValueError("The genuine native support has no unique requested face carrier")
    carrier = matches[0]
    return PhysicalFaceCoverage(carrier.donor_component.component_id,
        carrier.receiver_component.component_id, support.source_epoch.epoch_sha256,
        support.support_sha256,
        native_component_solid_thickness_m(support.source_epoch, carrier.donor_component.component_id),
        native_component_solid_thickness_m(support.source_epoch, carrier.receiver_component.component_id),
        F.from_float(carrier.declaration.vertical_contact_height_m),
        carrier.inclined_measure_density_m2_per_radian,
        carrier.native_angle_bounds_radians,
        "Complete typed source support replay; source areal-volume thickness and declared reference force carrier")


@dataclass(frozen=True, slots=True)
class NativeFirstBirthFullFaceDeclaration:
    before_state_sha256: str
    original_trial_support_sha256: str
    full_face_support_sha256: str
    source_epoch_sha256: str
    family_height_records: tuple
    original_accepted_contact_count: int
    provenance: str
    original_trial_activation_reused: bool = False
    fresh_first_birth_activation_required: bool = True
    adopted_born_history_extended: bool = False

    @property
    def material_admission_authorized(self):
        return False

    @property
    def whole_material_depth_force_reference_coverage(self):
        return True

    @property
    def actual_curved_face_force_metric_certified(self):
        return False


def declare_native_full_face_first_birth_support(accepted_before, support, *,
        provenance, adopted_born_states=()):
    """Declare a new full-face first-birth trial from the accepted old state.

    The original one-kilometre born histories are *unaccepted trial* results,
    absent from ``NativeImplicitUnderthrustMotionState``.  A new first-birth
    model may therefore declare its actual full material-reference depth and pay
    its own activation once when initialized.  It cannot reuse the old trial
    activation ledger or extend an adopted born state.  Such a continuation
    needs a different, history-preserving preparation model.

    Full reference depth does not certify the curved face's physical metric
    area or its distributed forces.  Those depend on the separately declared
    spherical embedding; the source constitutive strip uses reference area.

    The accepted sixteen contact histories, geometry, load source and every
    normal-column/law/preload declaration remain unchanged.  This constructor
    returns a new support and its explicit branch receipt; it advances no
    clock and performs no activation or material adoption itself.
    """
    from .geometric_contact_mechanics import LocalContactState
    from .geometric_native_implicit_underthrust_motion import (
        NativeImplicitUnderthrustMotionState,
        validate_native_implicit_underthrust_motion_state,
    )
    from .geometric_native_birth_contact_support import (
        NativeBirthContactSupport, declare_native_birth_contact_support,
        native_birth_contact_support_to_dict,
    )
    from .geometric_native_solid_layer_placement import native_component_solid_thickness_m
    from .geometric_native_physical_history import _typed_fingerprint
    _text(provenance, "Explicit full-face first-birth model provenance")
    if type(adopted_born_states) is not tuple or adopted_born_states:
        raise ValueError("A full-face FIRST birth cannot extend or reset an adopted born history")
    if type(accepted_before) is not NativeImplicitUnderthrustMotionState:
        raise ValueError("A full-face FIRST birth requires the unchanged accepted original-only state")
    if type(support) is not NativeBirthContactSupport:
        raise ValueError("A full-face FIRST birth requires the original typed source support")
    validate_native_implicit_underthrust_motion_state(accepted_before)
    native_birth_contact_support_to_dict(support)
    if _typed_fingerprint(accepted_before.current_surface) != _typed_fingerprint(support.source_epoch):
        raise ValueError("The full-face birth support differs from its complete actual accepted material epoch")
    if (type(accepted_before.states) is not tuple or len(accepted_before.states) != 16
            or any(type(row) is not tuple or len(row) != 2 or type(row[1]) is not LocalContactState
                for row in accepted_before.states)
            or len({row[0] for row in accepted_before.states}) != 16
            or len(accepted_before.receipts) != 1):
        raise ValueError("A full-face FIRST birth cannot reset an adopted mixed history or a later source epoch")
    before_fingerprint = _typed_fingerprint(accepted_before)
    support_fingerprint = _typed_fingerprint(support)
    declarations, records = [], []
    for carrier in support.interface_carriers:
        donor = native_component_solid_thickness_m(support.source_epoch,
            carrier.donor_component.component_id)
        receiver = native_component_solid_thickness_m(support.source_epoch,
            carrier.receiver_component.component_id)
        common = min(donor, receiver)
        height = float(common)
        if not math.isfinite(height) or F.from_float(height) != common:
            raise ValueError("The complete actual physical face height is not exactly representable by its force declaration")
        if F.from_float(carrier.declaration.vertical_contact_height_m) > common:
            raise ValueError("The original trial force carrier exceeds the actual physical common face")
        declarations.append(replace(carrier.declaration, vertical_contact_height_m=height,
            provenance=provenance+"; new unadopted full-face FIRST-birth branch from "+support.support_sha256))
        records.append((carrier.declaration.family_id, donor, receiver,
            F.from_float(carrier.declaration.vertical_contact_height_m), common))
    full = declare_native_birth_contact_support(support.source_epoch,
        support.foundation_declarations, tuple(declarations),
        incoming_geometry=support.incoming_geometry)
    if (_typed_fingerprint(accepted_before) != before_fingerprint
            or _typed_fingerprint(support) != support_fingerprint):
        raise ValueError("The full-face branch changed its accepted source or original unaccepted trial support")
    receipt = NativeFirstBirthFullFaceDeclaration(accepted_before.state_sha256,
        support.support_sha256, full.support_sha256, support.source_epoch.epoch_sha256,
        tuple(records), len(accepted_before.states), provenance)
    return full, receipt
