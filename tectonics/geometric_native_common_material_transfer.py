"""Original-material charts and one explicitly reduced source potential.

The eight archived donor labels retain their original material and cohort.
Their passive COMMON chart extension is a declared reduced chart, not a live
cold-mask point, a recovered three-dimensional rod, or physical admission.
The single source embedding uses shared owners and eight internal radial modes;
the 86 COMMON body modes have no source coupling until a physical reduction.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

from .geometric_native_pose import FractionInterval as I, interval_sqrt
from .geometric_native_affine_body import AFFINE_COORDINATE_ORDER
from .geometric_native_common_reference import (
    validate_native_common_reference, bound_native_common_reference_bank_point)
from .geometric_native_common_contact_ports import _encoded, _payload
from .geometric_native_common_source_carrier import (
    validate_native_common_source_carrier_system, _literal, _precision, _round,
    _dot, _cross, _map, _add, _sub, _neg, _source_jet, _world_reactions,
    _matrix_histories, _read_fraction, BRANCHES)

FORMAT = "native-common-original-material-reduced-transfer-system-1"
RESPONSE_FORMAT = "native-common-original-material-reduced-transfer-response-1"
SCOPE = "COMMON_ORIGINAL_MATERIAL_EXTENDED_REDUCED_SOURCE_COMPONENT_ONLY"
FORCE_BASIS = "archival_old_reference"
BEAM_OWNERSHIP = "INTERNAL_SOURCE_BEAMS_REPLACE_COMMON_CALIBRATED_BEAM_ENERGY"
CHART_LAW = "SAME_ORIGINAL_PARENT_PASSIVE_COMMON_PHI_ARCHIVE_EXTENSION"
INTERNAL_MOTION_LAW = "FRESH_OWNER_ROTATION_OF_FROZEN_PHI_POINT_PLUS_INTERNAL_LITERAL_SOURCE_DIRECTION_DISPLACEMENT_WITH_FIXED_MATERIAL_LEVER"
_ROOT = Path(__file__).resolve().parents[1]
_AUDIT_INPUTS = (
    ("analysis/geometric_native_common_reference_validation/contact_history_audit.json",
     "6da68ccda2089921fbe197293a05e4f84c3ea83fd2abb648b98d20f9b1813901"),
    ("analysis/geometric_native_common_contact_validation/contact_energy_decomposition_audit.json",
     "b164936ac0217e4c8c4feaebc8d7495731edd218e4da93db36e1d2660a3815b0"),
)
_FLAGS = dict(
    same_original_material_identity_authenticated=True,
    passive_archive_chart_extension_declared=True,
    one_global_source_embedding_defined=True,
    one_potential_derivatives_defined=True,
    separate_global_beam_energized_once=True,
    source_work_ledgers_retained_without_advance=True,
    original_literal_q_normalized_in_source_mechanics=False,
    source_internal_links_reduced_into_common_body_modes=False,
    physical_source_material_handover_certified=False,
    authentic_archived_three_dimensional_body_motion_certified=False,
    full_three_dimensional_rod_endpoints_recovered=False,
    common_body_modal_coupling_certified=False,
    new_mixed_force_equilibrium_certified=False,
    whole_clock_all_pair_admission_authorized=False,
    physical_material_admission_authorized=False,
    source_history_advanced=False,
    original_point_1_completed=False,
)
_REGISTRY = {}
_DEPENDENCIES = {}
_BIND_CACHE = {}
_QUERY_CACHE = {}


def _seal(value):
    return hashlib.sha256(_encoded({f.name: getattr(value, f.name) for f in fields(value)
        if f.name != "binding_sha256"})).hexdigest()


def _mechanical_context_sha(carrier_system):
    # All runtime mechanics, both force bases and complete atom/work records.
    # Excludes only the large upstream whole DTOs not used by this component.
    return hashlib.sha256(_encoded((carrier_system.source_radius_m, carrier_system.carriers))).hexdigest()


def _geometry_context_sha(reference):
    # Bank points also authenticate their own registered source bank on query.
    return hashlib.sha256(_encoded((reference.reference_snapshot_sha256,
        reference.radius_m, reference.angular_rate_per_physical_volume_depth_m, reference.banks))).hexdigest()


def _remember(value, dependencies):
    value = replace(value, binding_sha256=_seal(value))
    key = type(value), value.binding_sha256
    _REGISTRY[key] = _encoded(value)
    _DEPENDENCIES[key] = dependencies
    return value


def _registered(value, kind):
    if (type(value) is not kind or value.binding_sha256 != _seal(value)
            or _REGISTRY.get((kind, value.binding_sha256)) != _encoded(value)):
        raise ValueError("Altered or resealed material transfer lacks authentic source replay")
    return _DEPENDENCIES[kind, value.binding_sha256]


def _audit_inputs():
    result = []
    for path, expected in _AUDIT_INPUTS:
        raw = (_ROOT/path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("The immutable original-material or beam audit changed")
        result.append(json.loads(raw))
    return tuple(result)


def _read_interval(raw):
    return I(_read_fraction(raw["lo"]), _read_fraction(raw["hi"]))


def _polynomial_bounds(coefficients, tail, domain, precision):
    result = I(0)
    for coefficient in reversed(coefficients):
        result = _round(result*domain+coefficient, precision)
    return _round(result+I(-tail, tail), precision)


def _source_predicates(material, q, precision):
    """Re-evaluate g and g/u from the source's own whole-clock polynomials.

    For Y(0)=Z(0)=0, their remainder divided by u is bounded by the
    certified first derivative remainder, including at the right germ u=0.
    No copied live/archive status decides a material membership.
    """
    rows = {row["name"]: row for row in material["sweep_certificate"]["analytic_polynomials"]}
    polys = {}
    for name in ("C", "Y", "Z"):
        row = rows[name]
        polys[name] = (tuple(_read_fraction(v) for v in row["coefficients"]),
                       tuple(_read_fraction(v) for v in row["derivative_tails"]))
    if q[0] != 0 or polys["Y"][0][0] != 0 or polys["Z"][0][0] != 0:
        raise ValueError("The source contact germ requires its literal x=0 hemisphere labels")
    endpoint = -q[0]*_polynomial_bounds(polys["C"][0], polys["C"][1][0], I(1), precision)
    divided = I(0)
    for name, coefficient in (("Y", q[1]), ("Z", q[2])):
        values, tails = polys[name]
        endpoint += coefficient*_polynomial_bounds(values, tails[0], I(1), precision)
        divided += coefficient*_polynomial_bounds(values[1:], tails[1], I(0, 1), precision)
    return _round(endpoint, precision), _round(divided, precision)


def _transpose(matrix):
    return tuple(tuple(matrix[j][i] for j in range(3)) for i in range(3))


@dataclass(frozen=True, slots=True)
class NativeCommonOriginalMaterialAttachment:
    point_id: str
    trace_id: str
    side: str
    original_parent_id: str
    original_material_id: str
    owner: int
    original_direction_literal: tuple
    original_parent_support_predicate_bounds: tuple
    original_label_membership: str
    receiver_endpoint_predicate_bounds: I
    whole_clock_receiver_predicate_divided_by_u_bounds: I
    live_domain_id: str
    archive_domain_id: str
    archive_cohort_id: str | None
    complete_parent_allocation_json: str
    complete_archive_cohort_json: str | None
    complete_original_parent_json: str
    geometry_field_component_id: str
    geometry_field_material_kind: str
    normalized_material_ray_bounds: tuple
    current_world_material_ray_bounds: tuple
    material_depth_bounds_m: I
    areal_material_thickness_m: F
    roof_radial_displacement_m: F
    material_owner_pose_bounds: tuple
    genuine_current_attachment_world_m: tuple
    material_reference_position_bounds_m: tuple
    source_tip_reference_bounds_m: tuple
    source_direction_reference_bounds: tuple
    material_local_lever_bounds_m: tuple
    material_local_internal_direction_bounds: tuple
    inherited_receiver_directors_material_local_bounds: tuple
    chosen_force_basis: str = FORCE_BASIS
    passive_chart_law: str = CHART_LAW
    original_material_membership_authenticated: bool = True
    current_live_cold_mask_membership_claimed: bool = False
    archived_three_dimensional_body_motion_certified: bool = False
    internal_material_motion_law: str = INTERNAL_MOTION_LAW

    @property
    def material_id(self):
        return self.original_material_id

    @property
    def archived(self):
        return self.original_label_membership == "ARCHIVED_ORIGINAL_MATERIAL"


@dataclass(frozen=True, slots=True)
class NativeCommonOriginalBeam:
    trace_id: str
    side: str
    original_parent_id: str
    component_id: str
    original_global_stiffness_index: int
    source_coordinate_index: int
    global_coordinate_index: int
    stiffness_exact_n_m: F
    reference_displacement_m: F
    reference_force_exact_n: F
    reference_energy_exact_j: F
    stiffness_binary64_n_m: F
    literal_reduced_beam_binary_force_evaluation_n: F
    literal_reduced_beam_binary_energy_evaluation_j: F
    binary_force_evaluation_minus_exact_n: F
    binary_energy_evaluation_minus_exact_j: F
    complete_source_bending_parameters_json: str
    complete_immutable_beam_audit_row_json: str
    archived_old_binary_force_replayed: bool = False
    energetic_owner: str = BEAM_OWNERSHIP


@dataclass(frozen=True, slots=True)
class NativeCommonMaterialTransferSystem:
    source_snapshot_sha256: str
    reference_snapshot_sha256: str
    source_carrier_binding_sha256: str
    source_mechanical_context_sha256: str
    common_geometry_context_sha256: str
    source_history_binding_sha256: str
    source_epoch_sha256: str
    force_basis: str
    coordinate_order: tuple
    source_coordinate_order: tuple
    source_support_indices: tuple
    source_to_global_injection: tuple
    global_zero_coordinate_indices: tuple
    attachments: tuple
    beams: tuple
    complete_original_material_state_json: str
    complete_current_source_states_json: str
    complete_current_source_memory_json: str
    complete_before_work_ledgers_json: str
    complete_unaccepted_candidate_work_ledgers_json: str
    original_source_work_lineages: tuple
    source_contact_elapsed_time_s: F
    source_contact_step_count: int
    source_memory_elapsed_time_s: F
    source_geometry_elapsed_time_s: F
    material_geometry_elapsed_time_s: F
    source_serial_reference_energy_j: F
    beam_reference_energy_j: F
    beam_mathematical_reference_energy_j: F
    beam_energy_rounding_residual_j: F
    audit_input_raw_sha256: tuple
    archived_donor_attachment_count: int
    live_donor_attachment_count: int
    live_receiver_attachment_count: int
    provenance: str
    precision: int
    binding_sha256: str
    energetic_owner: str = BEAM_OWNERSHIP
    reduced_coordinate_map_second_derivative_exact_zero: bool = True
    common_body_source_coupling_exact_zero: bool = True
    internal_material_motion_law: str = INTERNAL_MOTION_LAW
    material_point_queries_are_frozen_reference_chart_queries: bool = True

    @property
    def source_carrier_system(self):
        validate_native_common_material_transfer_system(self)
        source, reference = _DEPENDENCIES[type(self), self.binding_sha256]
        # Explicit access to the complete upstream object authenticates it;
        # the reduced evaluator uses its guarded compact mechanics instead.
        return validate_native_common_source_carrier_system(source, reference=reference)

    @property
    def scope(self):
        return SCOPE

    @property
    def source_coordinate_count(self):
        return 14

    @property
    def global_coordinate_count(self):
        return 100

    @property
    def source_state_or_work_advanced(self):
        return False

    @property
    def physical_source_material_handover_certified(self):
        return False


def bind_native_common_material_transfer(carrier_system, *, reference,
        force_basis=FORCE_BASIS, provenance, precision=96):
    carrier_system = validate_native_common_source_carrier_system(carrier_system, reference=reference)
    reference = validate_native_common_reference(reference)
    precision = _precision(precision)
    if force_basis != FORCE_BASIS or type(provenance) is not str or not provenance.strip():
        raise ValueError("Declare an archival0 original-material reduced force chart and its provenance")
    cache_key = (carrier_system.binding_sha256, reference.reference_snapshot_sha256,
                 force_basis, provenance, precision)
    if cache_key in _BIND_CACHE:
        return validate_native_common_material_transfer_system(_BIND_CACHE[cache_key],
            carrier_system=carrier_system, reference=reference)
    before = json.loads(dict(carrier_system.complete_source_dto_json)["before"])
    snapshot = before["snapshot"]
    history = before["original_physical_history"]
    material = snapshot["current_material_epoch"]["source_material_state"]
    parents = {r["fragment_id"]: r for r in material["journal"]["source_surface"]["fragments"]}
    allocations = {r["fragment_id"]: r for r in material["parent_allocations"]}
    cohorts = {r["original_parent_id"]: r for r in material["archive_cohorts"]}
    old_order = carrier_system.coordinate_order
    body_order = tuple(("body", bank.component_id, name)
        for bank in sorted(reference.banks, key=lambda b: b.component_id)
        for name in (("old_tip_increment",)+AFFINE_COORDINATE_ORDER
                     if bank.material_kind == "original_live" else AFFINE_COORDINATE_ORDER))
    order = old_order[:6]+body_order+tuple(("source_internal", c[1], c[2]) for c in old_order[6:])
    support = tuple(range(6))+tuple(range(92, 100))
    if len(body_order) != 86 or len(order) != 100:
        raise ValueError("This reduced catalogue requires the authentic eight COLD and five HOT banks")
    injection = tuple(tuple(F(int(i == support[j])) for j in range(14)) for i in range(100))
    banks = {b.original_parent_id: b for b in reference.banks if b.material_kind == "original_live"}
    _, current_pose = _matrix_histories(snapshot["owner_histories"], paired=True, precision=precision)
    membership_audit, beam_audit = _audit_inputs()
    audit_members = {r["point_id"]: r for r in membership_audit["independent_all16_source_label_membership"]}
    attachments = []
    for carrier in carrier_system.carriers:
        atom = json.loads(carrier.complete_original_atom_history_json)
        q = tuple(_literal(v) for v in atom["original_point"]["position"])
        endpoint, divided = _source_predicates(material, q, precision)
        if divided.lo > 0 and endpoint.lo > 0:
            donor_archived = True
        elif divided.hi < 0 and endpoint.hi < 0:
            donor_archived = False
        else:
            raise ValueError("The complete first-entry clock leaves an original donor label unresolved")
        peer = audit_members[carrier.point_id]
        expected_status = ("ARCHIVED_ORIGINAL_DONOR_LABEL_ABSENT_FROM_RETAINED_COLD_MASK"
                           if donor_archived else "ORIGINAL_DONOR_LABEL_RETAINED")
        if (peer["membership_status"] != expected_status
                or peer["donor_original_parent_id"] != carrier.donor_original_parent_id
                or peer["receiver_original_parent_id"] != carrier.receiver_original_parent_id
                or min(divided.hi, _read_interval(peer["whole_original_clock_receiver_predicate_divided_by_u_bounds"]).hi)
                   < max(divided.lo, _read_interval(peer["whole_original_clock_receiver_predicate_divided_by_u_bounds"]).lo)):
            raise ValueError("Recomputed original membership differs from the pinned independent whole-clock audit")
        norm = interval_sqrt(I(sum((v*v for v in q), F(0))), precision+16)
        qhat = tuple(I(v)/norm for v in q)
        basis = carrier.archival_old_reference
        for side, pid, owner in (("sub", carrier.donor_original_parent_id, carrier.sub_owner),
                                 ("over", carrier.receiver_original_parent_id, carrier.over_owner)):
            parent, allocation, bank = parents[pid], allocations[pid], banks[pid]
            if (parent["parcel"]["material_id"] != allocation["material_id"]
                    or bank.material_id != allocation["material_id"] or allocation["owner"] != owner
                    or parent["parcel"]["plate"] != owner or bank.owner != owner):
                raise ValueError("A reduced chart must use the same original material, parent and owner")
            supports = []
            for plane in parent["geometry"]["supports"]:
                if parent["geometry"]["pose"] or plane["rotations"]:
                    raise ValueError("An original parent support needs its complete original pose replay")
                supports.append(I(sum((F(n)*v for n, v in zip(plane["normal"], q)), F(0))))
            if any(value.lo < 0 for value in supports) or not any(value.lo > 0 for value in supports):
                raise ValueError("The literal old label is outside its genuine original parent support")
            archived = side == "sub" and donor_archived
            cohort = cohorts.get(pid) if archived else None
            if archived and (cohort is None or cohort["cohort_id"] != allocation["cohort_id"]
                    or cohort["archive_domain_id"] != allocation["archive_domain_id"]
                    or cohort["material_id"] != allocation["material_id"] or cohort["owner"] != owner):
                raise ValueError("An archived source label needs its original live history/reservoir cohort")
            Q = current_pose[owner]
            ray = tuple(_round(v, precision) for v in _map(Q, qhat))
            mapped = bound_native_common_reference_bank_point(bank, ray, I(0), precision=max(96, precision))
            X = mapped.world_position_bounds_m
            p = basis.sub_position_bounds_m if side == "sub" else basis.over_position_bounds_m
            v = basis.sub_direction_bounds if side == "sub" else basis.over_direction_bounds
            Qt = _transpose(Q)
            lever = tuple(_round(c, precision) for c in _map(Qt, _sub(p, X)))
            local_directors = tuple(tuple(_round(c, precision) for c in _map(Qt, e))
                for e in basis.receiver_director_bounds) if side == "over" else ()
            attachments.append(NativeCommonOriginalMaterialAttachment(carrier.point_id, carrier.trace_id,
                side, pid, allocation["material_id"], owner, q, tuple(supports),
                "ARCHIVED_ORIGINAL_MATERIAL" if archived else "LIVE_ORIGINAL_MATERIAL",
                endpoint, divided, allocation["live_domain_id"], allocation["archive_domain_id"],
                None if cohort is None else cohort["cohort_id"], _encoded(allocation).decode(),
                None if cohort is None else _encoded(cohort).decode(), _encoded(parent).decode(),
                bank.component_id, bank.material_kind, qhat, ray, I(0), bank.areal_material_thickness_m,
                bank.roof_radial_displacement_m, Q, X,
                tuple(_round(c, precision) for c in _map(Qt, X)), p, v, lever,
                tuple(_round(c, precision) for c in _map(Qt, v)), local_directors))
    counts = (sum(a.archived for a in attachments),
              sum(a.side == "sub" and not a.archived for a in attachments),
              sum(a.side == "over" and not a.archived for a in attachments))
    if counts != (8, 8, 16) or len(attachments) != 32:
        raise ValueError("The full original source requires eight archived/eight live donors and sixteen live receivers")
    tips = {r[0]: r[1:] for r in snapshot["transverse_displacement_m"]}
    stiffness = history["original_reference"]["stiffness"]
    R = carrier_system.source_radius_m
    beams = []
    for row in beam_audit["eight_cold_beam_calibration_rows"]:
        trace, side = row["trace_id"], row["source_bank"]
        source_index = old_order.index(("radial", trace, side))
        K = stiffness[source_index][source_index]
        k = _literal(K)/(R*R)
        wfloat = tips[trace][0 if side == "sub" else 1]
        w = _literal(wfloat)
        if (source_index != row["original_global_stiffness_index"]
                or k != _read_fraction(row["source_old_tip_stiffness_n_m"]["exact_F"])
                or w != _read_fraction(row["accepted_tip_displacement_m"]["exact_F"])
                or k*w*w/2 != _read_fraction(row["exact_F_beam_reference_energy_j"]["exact_F"])):
            raise ValueError("An original global beam differs from its pinned independent source decomposition")
        kfloat = K/(float(R)*float(R))
        ffloat, efloat = kfloat*wfloat, kfloat*wfloat*wfloat/2
        beams.append(NativeCommonOriginalBeam(trace, side, row["original_parent_id"], row["component_id"],
            source_index, source_index, support[source_index], k, w, k*w, k*w*w/2,
            _literal(kfloat), _literal(ffloat), _literal(efloat), _literal(ffloat)-k*w,
            _literal(efloat)-k*w*w/2, row["complete_source_bending_parameters_json"], _encoded(row).decode()))
    if len(beams) != 8 or len({b.source_coordinate_index for b in beams}) != 8:
        raise ValueError("Every original global beam must appear exactly once")
    ledger = dict(history["unaccepted_candidate_work_ledgers"])
    beamE = _literal(ledger["bending_energy_increment_j"])
    if beamE != _literal(history["actual_candidate"]["diagnostics"]["bending_energy_increment_j"]):
        raise ValueError("The literal original beam energy ledger differs from its actual source diagnostic")
    mathE = sum((b.reference_energy_exact_j for b in beams), F(0))
    value = NativeCommonMaterialTransferSystem(carrier_system.source_snapshot_sha256,
        reference.reference_snapshot_sha256, carrier_system.binding_sha256,
        _mechanical_context_sha(carrier_system), _geometry_context_sha(reference),
        carrier_system.source_history_binding_sha256, carrier_system.source_epoch_sha256,
        force_basis, order, old_order, support, injection, tuple(range(6, 92)),
        tuple(attachments), tuple(beams), _encoded(material).decode(),
        carrier_system.complete_current_source_states_json, carrier_system.complete_current_source_memory_json,
        carrier_system.complete_before_work_ledgers_json,
        carrier_system.complete_unaccepted_candidate_work_ledgers_json,
        tuple(c.original_source_work_lineage_id for c in carrier_system.carriers),
        carrier_system.source_contact_elapsed_time_s, carrier_system.source_contact_step_count,
        carrier_system.source_memory_elapsed_time_s, F(0), carrier_system.current_geometry_elapsed_time_s,
        carrier_system.all16_source_energy_exact_j, beamE, mathE, beamE-mathE,
        _AUDIT_INPUTS, *counts, provenance, precision, "")
    result = _remember(value, (carrier_system, reference))
    _BIND_CACHE[cache_key] = result
    return result


def validate_native_common_material_transfer_system(value, *, carrier_system=None, reference=None):
    actual_carrier, actual_reference = _registered(value, NativeCommonMaterialTransferSystem)
    if (_mechanical_context_sha(actual_carrier) != value.source_mechanical_context_sha256
            or _geometry_context_sha(actual_reference) != value.common_geometry_context_sha256):
        raise ValueError("Retained mechanical or COMMON chart dependency changed after authentic compact binding")
    if (carrier_system is not None and (type(carrier_system) is not type(actual_carrier)
            or carrier_system is not actual_carrier and _encoded(carrier_system) != _encoded(actual_carrier))):
        raise ValueError("The material transfer belongs to another authentic complete source carrier")
    if (reference is not None and (type(reference) is not type(actual_reference)
            or reference is not actual_reference and _encoded(reference) != _encoded(actual_reference))):
        raise ValueError("The material transfer belongs to another COMMON material reference")
    return value


def validate_native_common_material_transfer_energy_ownership(system, *, common_calibrated_beam_energy_active=False):
    validate_native_common_material_transfer_system(system)
    if type(common_calibrated_beam_energy_active) is not bool or common_calibrated_beam_energy_active:
        raise ValueError("The same eight old beam stores cannot energize both the internal source component and COMMON calibrated cold energy")
    return BEAM_OWNERSHIP


def validate_native_common_material_transfer_energy_assembly(system, *,
        common_calibrated_beam_reference_energized=False, continuous_source_measure_energized=False):
    validate_native_common_material_transfer_system(system)
    if (type(common_calibrated_beam_reference_energized) is not bool
            or type(continuous_source_measure_energized) is not bool
            or common_calibrated_beam_reference_energized or continuous_source_measure_energized):
        raise ValueError("The reduced potential owns old beams and literal atoms once; duplicated COMMON beam or continuous source stores are forbidden")
    return BEAM_OWNERSHIP


@dataclass(frozen=True, slots=True)
class NativeCommonOriginalMaterialPoint:
    system_binding_sha256: str
    point_id: str
    side: str
    original_parent_id: str
    original_material_id: str
    original_label_membership: str
    geometry_field_component_id: str
    original_direction_literal: tuple
    normalized_material_ray_bounds: tuple
    current_world_material_ray_bounds: tuple
    material_depth_bounds_m: I
    world_position_bounds_m: tuple
    input_position_bounds_m: tuple
    reference_gradient_bounds: tuple
    rotation_matrix_bounds: tuple
    radius_bounds_m: I
    physical_volume_depth_bounds_m: I
    determinant_exact: F
    signed_material_chart_jacobian_exact: F
    precision: int
    binding_sha256: str
    original_material_membership_authenticated: bool = True
    passive_chart_law: str = CHART_LAW
    existing_common_live_mask_membership_authenticated: bool = False
    physical_material_admission_authorized: bool = False
    reference_chart_query_not_internal_deformed_position: bool = True


def evaluate_native_common_material_point(system, point_id, side, *, material_depth_bounds_m, precision=96):
    system = validate_native_common_material_transfer_system(system)
    precision = _precision(precision)
    match = tuple(a for a in system.attachments if a.point_id == point_id and a.side == side)
    if len(match) != 1:
        raise ValueError("A material query needs its authentic original atom and bank side")
    a = match[0]
    _, reference = _DEPENDENCIES[type(system), system.binding_sha256]
    bank = next(b for b in reference.banks if b.component_id == a.geometry_field_component_id)
    point = bound_native_common_reference_bank_point(bank, a.current_world_material_ray_bounds,
        material_depth_bounds_m, precision=max(96, precision))
    result = NativeCommonOriginalMaterialPoint(system.binding_sha256, point_id, side,
        a.original_parent_id, a.original_material_id, a.original_label_membership,
        a.geometry_field_component_id, a.original_direction_literal, a.normalized_material_ray_bounds,
        a.current_world_material_ray_bounds, material_depth_bounds_m, point.world_position_bounds_m,
        point.input_position_bounds_m, point.reference_gradient_bounds, point.rotation_matrix_bounds,
        point.radius_bounds_m, point.physical_volume_depth_bounds_m, point.determinant_exact,
        point.signed_material_chart_jacobian_exact, precision, "")
    return _remember(result, (system,))


def validate_native_common_material_point(value, *, system=None):
    actual, = _registered(value, NativeCommonOriginalMaterialPoint)
    validate_native_common_material_transfer_system(actual)
    if system is not None and (type(system) is not type(actual)
            or system is not actual and _encoded(system) != _encoded(actual)):
        raise ValueError("The passive original-material point belongs to another complete source chart")
    return value


def native_common_material_point_to_dict(value):
    value = validate_native_common_material_point(value)
    raw = _payload(value)
    raw.update(format="native-common-original-material-passive-point-1", scope=SCOPE)
    return raw


def native_common_material_point_from_dict(raw, *, system):
    if type(raw) is not dict:
        raise ValueError("A passive original-material point requires its exact query DTO")
    result = evaluate_native_common_material_point(system, raw.get("point_id"), raw.get("side"),
        material_depth_bounds_m=_read_interval(raw["material_depth_bounds_m"]), precision=raw.get("precision"))
    if _encoded(raw) != _encoded(native_common_material_point_to_dict(result)):
        raise ValueError("The material point differs from its genuine original identity and declared full-depth chart")
    return result


@dataclass(frozen=True, slots=True)
class NativeCommonMaterialAtomResponse:
    point_id: str
    trace_id: str
    source_accounting_bucket: str
    basis: str
    branch: str
    original_area_m2: F
    sub_position_bounds_m: tuple
    over_position_bounds_m: tuple
    sub_direction_bounds: tuple
    over_direction_bounds: tuple
    receiver_director_bounds: tuple
    projected_geometry_bounds_m: tuple
    scalar_increment_bounds_m: tuple
    scalar_free_anchor_bounds_m: tuple
    scalar_first_derivative_bounds: tuple
    scalar_second_derivative_bounds: tuple
    traction_bounds_pa: tuple
    effective_tangent_stiffness_pa_m: tuple
    potential_bounds_j: I
    potential_gradient_bounds: tuple
    potential_hessian_bounds: tuple
    stored_source_energy_components_j: tuple
    world_reactions: tuple
    source_reference_world_reactions: tuple
    sub_attachment_position_bounds_m: tuple
    over_attachment_position_bounds_m: tuple
    sub_virtual_lever_bounds_m: tuple
    over_virtual_lever_bounds_m: tuple
    force_attachment_torque_on_sub_nm: tuple
    force_attachment_torque_on_over_nm: tuple
    lever_couple_on_sub_nm: tuple
    lever_couple_on_over_nm: tuple
    receiver_carrier_couple_on_over_nm: tuple
    complete_attachment_torque_on_sub_nm: tuple
    complete_attachment_torque_on_over_nm: tuple
    attachment_torque_closure_residual_bounds_nm: tuple
    proposed_plastic_increment_bounds_m: I
    source_work_lineage_id: tuple
    archival_literal_world_reactions_json: str
    archival_mathematical_minus_literal_reaction_bounds: tuple
    archival_literal_reaction_replay_certified: bool
    original_source_state_json: str

    @property
    def source_response(self):
        return self


@dataclass(frozen=True, slots=True)
class NativeCommonMaterialBeamResponse:
    trace_id: str
    side: str
    source_coordinate_index: int
    global_coordinate_index: int
    coordinate_increment_m: F
    potential_increment_j: F
    gradient_n: F
    hessian_n_m: F
    reference_energy_exact_j: F
    reference_force_exact_n: F


@dataclass(frozen=True, slots=True)
class NativeCommonMaterialTransferResponse:
    system_binding_sha256: str
    source_snapshot_sha256: str
    force_basis: str
    branch: str
    coordinate_order: tuple
    coordinate_increments: tuple
    source_coordinate_order: tuple
    source_coordinate_increments: tuple
    source_support_indices: tuple
    global_zero_coordinate_indices: tuple
    atom_responses: tuple
    beam_responses: tuple
    potential_j: I
    potential_gradient_bounds: tuple
    potential_hessian_bounds: tuple
    source_pullback_gradient_bounds: tuple
    source_pullback_hessian_bounds: tuple
    source_serial_potential_j: I
    beam_potential_j: F
    beam_reference_energy_j: F
    beam_energy_rounding_residual_j: F
    complete_unaccepted_candidate_work_ledgers_json: str
    original_source_work_lineages: tuple
    source_contact_elapsed_time_s: F
    source_geometry_elapsed_time_s: F
    material_geometry_elapsed_time_s: F
    precision: int
    binding_sha256: str
    energetic_owner: str = BEAM_OWNERSHIP
    common_body_source_coupling_exact_zero: bool = True
    global_coordinate_map_second_derivative_exact_zero: bool = True
    source_state_or_work_advanced: bool = False
    physical_source_material_handover_certified: bool = False
    old_force_archive_reproduced_in_current_objective_pose: bool = False
    internal_material_motion_law: str = INTERNAL_MOTION_LAW

    @property
    def scope(self):
        return SCOPE

    @property
    def potential_bounds_j(self):
        return self.potential_j


def _atom_response(system, carrier, increments, branch, precision, attachment_lookup):
    basis = carrier.archival_old_reference
    order = system.source_coordinate_order
    ds = increments[order.index(("radial", carrier.trace_id, "sub"))]
    do = increments[order.index(("radial", carrier.trace_id, "over"))]
    R = _DEPENDENCIES[type(system), system.binding_sha256][0].source_radius_m
    if R+ds <= 0 or R+do <= 0:
        raise ValueError("The original virtual source tips must remain at positive radii")
    vs, vo = basis.sub_direction_bounds, basis.over_direction_bounds
    ps = tuple(x+ds*v for x, v in zip(basis.sub_position_bounds_m, vs))
    po = tuple(x+do*v for x, v in zip(basis.over_position_bounds_m, vo))
    frames = basis.receiver_director_bounds
    geometry, first, second = _source_jet(order, carrier.sub_owner, carrier.over_owner,
        carrier.trace_id, (ps, po), (vs, vo), frames, precision)
    # Q_current Q_current^T is a correlated identity: reduce X+a to p_archive
    # symbolically before enclosure. No numerical midpoint force closure.
    eta = tuple(_round(_dot(e, tuple(do*o-ds*s for s, o in zip(vs, vo))), precision) for e in frames)
    stiffness = (carrier.series_stiffness_exact_pa_m[0],
                 F(0) if branch == BRANCHES[0] else carrier.series_stiffness_exact_pa_m[1])
    traction = tuple(_round(I(t)+k*d, precision) for t, k, d in
                     zip(carrier.literal_source_tractions_pa, stiffness, eta))
    if traction[0].hi >= 0:
        raise ValueError("The reduced source potential must remain on its compressed normal branch")
    area = carrier.original_area_m2
    E0 = sum(carrier.stored_source_energy_components_j, F(0))
    energy = _round(I(E0)+area*sum((I(t)*d+k*d*d/2 for t, k, d in
        zip(carrier.literal_source_tractions_pa, stiffness, eta)), I(0)), precision)
    gradient = tuple(_round(area*sum((traction[a]*first[a][i] for a in range(2)), I(0)), precision)
                     for i in range(14))
    hessian = tuple(tuple(_round(area*sum((stiffness[a]*first[a][i]*first[a][j]
        +traction[a]*second[a][i][j] for a in range(2)), I(0)), precision)
        for j in range(14)) for i in range(14))
    world = _world_reactions(area, traction, (ps, po), (vs, vo), frames, precision)
    W = dict(world)
    suba, overa = (attachment_lookup[carrier.point_id, side] for side in ("sub", "over"))
    xs = tuple(x+ds*v for x, v in zip(suba.genuine_current_attachment_world_m, vs))
    xo = tuple(x+do*v for x, v in zip(overa.genuine_current_attachment_world_m, vo))
    ls = _sub(basis.sub_position_bounds_m, suba.genuine_current_attachment_world_m)
    lo = _sub(basis.over_position_bounds_m, overa.genuine_current_attachment_world_m)
    Fs, Fo = W["force_on_sub_n"], W["force_on_over_n"]
    force_ts, force_to = _cross(xs, Fs), _cross(xo, Fo)
    couple_s, couple_o = _cross(ls, Fs), _cross(lo, Fo)
    receiver_couple = W["carrier_couple_on_over_nm"]
    total_s = tuple(_round(v, precision) for v in _add(force_ts, couple_s))
    total_o = tuple(_round(v, precision) for v in _add(_add(force_to, couple_o), receiver_couple))
    closure = (tuple(_round(x-y, precision) for x, y in zip(total_s, W["torque_on_sub_nm"])),
               tuple(_round(x-y, precision) for x, y in zip(total_o, W["torque_on_over_nm"])))
    return NativeCommonMaterialAtomResponse(carrier.point_id, carrier.trace_id,
        carrier.source_accounting_bucket, FORCE_BASIS, branch, area, ps, po, vs, vo, frames,
        geometry, eta, tuple(_round(I(f)+d, precision) for f, d in zip(carrier.stored_free_anchors_m, eta)),
        first, second, traction, stiffness, energy, gradient, hessian,
        carrier.stored_source_energy_components_j, world, basis.source_traction_world_reactions,
        xs, xo, ls, lo, force_ts, force_to, couple_s, couple_o, receiver_couple,
        total_s, total_o, closure, eta[1] if branch == BRANCHES[0] else I(0),
        carrier.original_source_work_lineage_id, carrier.archival_literal_world_reactions_json,
        carrier.archival_mathematical_minus_literal_reaction_bounds,
        carrier.archival_literal_reaction_replay_certified, carrier.source_local_state_json)


def evaluate_native_common_material_transfer(system, *, coordinate_increments=None,
        branch=BRANCHES[0], precision=96, common_calibrated_beam_energy_active=False):
    system = validate_native_common_material_transfer_system(system)
    validate_native_common_material_transfer_energy_ownership(system,
        common_calibrated_beam_energy_active=common_calibrated_beam_energy_active)
    precision = _precision(precision)
    values = (F(0),)*100 if coordinate_increments is None else coordinate_increments
    if (type(values) is not tuple or len(values) != 100 or any(type(x) is not F for x in values)
            or any(values[i] for i in range(92)) or branch not in BRANCHES):
        raise ValueError("Reduced queries require100 exact coordinates, zero finite owners/body86, and an explicit shear branch; owner2jets remain full")
    increments = tuple(values[i] for i in system.source_support_indices)
    cache_key = (system.binding_sha256, values, branch, precision)
    if cache_key in _QUERY_CACHE:
        return validate_native_common_material_transfer_response(_QUERY_CACHE[cache_key], system=system)
    lookup = {(a.point_id, a.side): a for a in system.attachments}
    source_context = _DEPENDENCIES[type(system), system.binding_sha256][0]
    atoms = tuple(_atom_response(system, carrier, increments, branch, precision, lookup)
                  for carrier in source_context.carriers)
    beams = tuple(NativeCommonMaterialBeamResponse(b.trace_id, b.side, b.source_coordinate_index,
        b.global_coordinate_index, increments[b.source_coordinate_index],
        b.stiffness_exact_n_m*(b.reference_displacement_m*increments[b.source_coordinate_index]
                             +increments[b.source_coordinate_index]**2/2),
        b.stiffness_exact_n_m*(b.reference_displacement_m+increments[b.source_coordinate_index]),
        b.stiffness_exact_n_m, b.reference_energy_exact_j, b.reference_force_exact_n) for b in system.beams)
    beamE = system.beam_reference_energy_j+sum((b.potential_increment_j for b in beams), F(0))
    serialE = _round(sum((a.potential_bounds_j for a in atoms), I(0)), precision)
    gradient = tuple(_round(sum((a.potential_gradient_bounds[i] for a in atoms), I(0))
        +sum((b.gradient_n for b in beams if b.source_coordinate_index == i), F(0)), precision)
        for i in range(14))
    hessian = tuple(tuple(_round(sum((a.potential_hessian_bounds[i][j] for a in atoms), I(0))
        +sum((b.hessian_n_m for b in beams if b.source_coordinate_index == i == j), F(0)), precision)
        for j in range(14)) for i in range(14))
    response = NativeCommonMaterialTransferResponse(system.binding_sha256, system.source_snapshot_sha256,
        FORCE_BASIS, branch, system.coordinate_order, values, system.source_coordinate_order,
        increments, system.source_support_indices, system.global_zero_coordinate_indices,
        atoms, beams, _round(serialE+beamE, precision), gradient, hessian, gradient, hessian,
        serialE, beamE, system.beam_reference_energy_j, system.beam_energy_rounding_residual_j,
        system.complete_unaccepted_candidate_work_ledgers_json, system.original_source_work_lineages,
        system.source_contact_elapsed_time_s, system.source_geometry_elapsed_time_s,
        system.material_geometry_elapsed_time_s, precision, "")
    result = _remember(response, (system,))
    _QUERY_CACHE[cache_key] = result
    return result


def validate_native_common_material_transfer_response(value, *, system=None):
    actual, = _registered(value, NativeCommonMaterialTransferResponse)
    validate_native_common_material_transfer_system(actual)
    if system is not None and (type(system) is not type(actual)
            or system is not actual and _encoded(system) != _encoded(actual)):
        raise ValueError("The complete reduced potential response belongs to another material system")
    return value


def expand_native_common_material_transfer_gradient(response):
    response = validate_native_common_material_transfer_response(response)
    support = dict(zip(response.source_support_indices, response.potential_gradient_bounds))
    return tuple(support.get(i, I(0)) for i in range(100))


def expand_native_common_material_transfer_hessian(response):
    response = validate_native_common_material_transfer_response(response)
    support = {value: index for index, value in enumerate(response.source_support_indices)}
    return tuple(tuple(response.potential_hessian_bounds[support[i]][support[j]]
        if i in support and j in support else I(0) for j in range(100)) for i in range(100))


def native_common_material_transfer_to_dict(value):
    value = validate_native_common_material_transfer_system(value)
    raw = _payload(value)
    raw.update(format=FORMAT, scope=SCOPE, **_FLAGS)
    return raw


def native_common_material_transfer_from_dict(raw, *, carrier_system, reference):
    if type(raw) is not dict:
        raise ValueError("Original-material transfer requires its complete JSON object")
    result = bind_native_common_material_transfer(carrier_system, reference=reference,
        force_basis=raw.get("force_basis"), provenance=raw.get("provenance"), precision=raw.get("precision"))
    if _encoded(raw) != _encoded(native_common_material_transfer_to_dict(result)):
        raise ValueError("Material transfer differs from original identity/clock/chart/beam replay")
    return result


native_common_material_transfer_system_to_dict = native_common_material_transfer_to_dict
native_common_material_transfer_system_from_dict = native_common_material_transfer_from_dict


def native_common_material_transfer_response_to_dict(value):
    value = validate_native_common_material_transfer_response(value)
    raw = _payload(value)
    raw.update(format=RESPONSE_FORMAT, scope=SCOPE, sparse_global_derivatives=True, **_FLAGS)
    return raw


def native_common_material_transfer_response_from_dict(raw, *, system):
    if type(raw) is not dict or type(raw.get("coordinate_increments")) is not list:
        raise ValueError("Reduced response requires its entire authentic global query")
    result = evaluate_native_common_material_transfer(system,
        coordinate_increments=tuple(_read_fraction(v) for v in raw["coordinate_increments"]),
        branch=raw.get("branch"), precision=raw.get("precision"))
    if _encoded(raw) != _encoded(native_common_material_transfer_response_to_dict(result)):
        raise ValueError("Reduced response differs from complete one-potential/source-couple replay")
    return result
