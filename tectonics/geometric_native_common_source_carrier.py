"""All sixteen literal source mechanisms, with their two distinct force bases.

This supplies source mechanical operators and one inherited energy/work lineage.
It does not attach archived donor labels to COMMON bodies, manufacture column
far endpoints, energize the separate C dtheta measure, or adopt an interval.
Owner derivatives use a fresh left exponential chart; finite queries currently
vary only the original physical radial coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from fractions import Fraction as F
import hashlib
import json
import math

from .geometric_intrinsic_geometry import rotation_increment_from_dict
from .geometric_native_pose import FractionInterval as I, certify_rotation_history, _rounded_interval
from .geometric_native_common_reference import validate_native_common_reference
from .geometric_native_common_contact_cuts import validate_common_contact_cuts
from .geometric_native_common_source_allocation import validate_native_common_source_allocation
from .geometric_native_common_contact_ports import _encoded, _payload
from .geometric_material_pair_reference import MaterialPairPort, material_pair_point_reactions

FORMAT = "native-common-source-complete-force-work-carrier-system-1"
RESPONSE_FORMAT = "native-common-source-complete-force-work-carrier-response-1"
MODEL = "literal_atom_source_mechanisms_dual_force_basis_rebased_integrated_anchors"
SCOPE = "SOURCE_COMPLETE_MECHANICAL_OPERATORS_NO_COMMON_MATERIAL_HANDOVER"
BASES = ("archival_old_reference", "current_objective")
BRANCHES = ("relaxed_zero_strength_shear", "frozen_elastic_predictor")
STORE_FIELDS = ("contact_energy_j", "bulk_energy_j", "cumulative_external_work_j",
                "cumulative_numerical_dissipation_j", "cumulative_friction_work_j",
                "cumulative_viscous_work_j")
_REACTION_NAMES = ("force_on_sub_n", "force_on_over_n", "torque_on_sub_nm",
                   "torque_on_over_nm", "force_torque_on_over_nm",
                   "carrier_couple_on_over_nm", "radial_force_n")
_REGISTRY = {}
_DEPENDENCIES = {}
_ZERO = (I(0),) * 3


def _literal(x):
    if type(x) is not float or not math.isfinite(x):
        raise ValueError("The source carrier requires literal finite binary64 source values")
    return F.from_float(x)


def _precision(x):
    if type(x) is not int or not 64 <= x <= 512:
        raise ValueError("Source carrier precision must be an integer from64 to512")
    return x


def _round(x, precision):
    return _rounded_interval(x if type(x) is I else I(x), precision + 16)


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), I(0))


def _cross(a, b):
    return a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]


def _add(a, b):
    return tuple(x+y for x, y in zip(a, b))


def _sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def _neg(a):
    return tuple(-x for x in a)


def _map(matrix, vector):
    return tuple(_dot(row, vector) for row in matrix)


def _axis(name):
    return tuple(F(int(c == name)) for c in "xyz")


def _symmetric_second(a, b, vector):
    return tuple((x+y)/2 for x, y in zip(_cross(a, _cross(b, vector)),
                                        _cross(b, _cross(a, vector))))


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _seal(value):
    return _digest({f.name: getattr(value, f.name) for f in fields(value)
                    if f.name != "binding_sha256"})


def _remember(value, dependencies):
    value = replace(value, binding_sha256=_seal(value))
    key = type(value), value.binding_sha256
    _REGISTRY[key] = _encoded(value)
    _DEPENDENCIES[key] = dependencies
    return value


def _registered(value, kind):
    if (type(value) is not kind or value.binding_sha256 != _seal(value)
            or _REGISTRY.get((kind, value.binding_sha256)) != _encoded(value)):
        raise ValueError("An altered or resealed source carrier lacks complete authentic source replay")
    return _DEPENDENCIES[kind, value.binding_sha256]


def _world_reactions(area, traction, positions, directions, frames, precision):
    force = tuple(_round(area*sum((traction[a]*frames[a][j] for a in range(2)), I(0)), precision)
                  for j in range(3))
    sub, over = positions
    torque = tuple(_round(x, precision) for x in _cross(sub, force))
    force_torque = tuple(_round(x, precision) for x in _cross(over, _neg(force)))
    couple = tuple(_round(x, precision) for x in _cross(_sub(over, sub), force))
    radial = (_round(_dot(directions[0], force), precision),
              _round(-_dot(directions[1], force), precision))
    return tuple(zip(_REACTION_NAMES, (force, _neg(force), torque, _neg(torque),
                                       force_torque, couple, radial)))


def _source_jet(coordinate_order, sub_owner, over_owner, trace_id, positions,
                directions, frames, precision):
    """Complete first/second product rule, including receiver-frame spin."""
    count = len(coordinate_order)
    first_d = [_ZERO for _ in range(count)]
    second_d = [[_ZERO for _ in range(count)] for _ in range(count)]
    p_s, p_o = positions
    v_s, v_o = directions
    for i, coordinate in enumerate(coordinate_order):
        if coordinate[0] == "owner":
            _, owner, name = coordinate
            axis = _axis(name)
            first_d[i] = _add(_cross(axis, p_o) if owner == over_owner else _ZERO,
                              _neg(_cross(axis, p_s)) if owner == sub_owner else _ZERO)
        elif coordinate[1] == trace_id:
            first_d[i] = _neg(v_s) if coordinate[2] == "sub" else v_o
        for j in range(i, count):
            other = coordinate_order[j]
            value = _ZERO
            if coordinate[0] == other[0] == "owner" and coordinate[1] == other[1]:
                owner = coordinate[1]
                a, b = _axis(coordinate[2]), _axis(other[2])
                value = _add(_symmetric_second(a, b, p_o) if owner == over_owner else _ZERO,
                             _neg(_symmetric_second(a, b, p_s)) if owner == sub_owner else _ZERO)
            elif (coordinate[0] == "owner" and other[0] == "radial"
                  or coordinate[0] == "radial" and other[0] == "owner"):
                angular, radial = (coordinate, other) if coordinate[0] == "owner" else (other, coordinate)
                if radial[1] == trace_id:
                    side_owner = sub_owner if radial[2] == "sub" else over_owner
                    if angular[1] == side_owner:
                        vector = v_s if radial[2] == "sub" else v_o
                        value = _cross(_axis(angular[2]), vector)
                        if radial[2] == "sub":
                            value = _neg(value)
            second_d[i][j] = second_d[j][i] = value
    D = _sub(p_o, p_s)
    first, second = [], []
    for frame in frames:
        fi = [_cross(_axis(c[2]), frame) if c[0] == "owner" and c[1] == over_owner
              else _ZERO for c in coordinate_order]
        row = []
        for i, coordinate in enumerate(coordinate_order):
            # Exact common-owner cancellation is used before interval arithmetic.
            if coordinate[0] == "owner":
                lever = _cross(p_s, frame)
                value = -lever["xyz".index(coordinate[2])] if coordinate[1] == sub_owner else (
                    lever["xyz".index(coordinate[2])] if coordinate[1] == over_owner else I(0))
                if sub_owner == over_owner:
                    value = I(0)
            else:
                value = _dot(frame, first_d[i])
            row.append(_round(value, precision))
        matrix = [[I(0) for _ in range(count)] for _ in range(count)]
        for i in range(count):
            for j in range(i, count):
                ci, cj = coordinate_order[i], coordinate_order[j]
                fij = (_symmetric_second(_axis(ci[2]), _axis(cj[2]), frame)
                       if ci[0] == cj[0] == "owner" and ci[1] == cj[1] == over_owner else _ZERO)
                value = _dot(frame, second_d[i][j])+_dot(fi[i], first_d[j])+_dot(fi[j], first_d[i])+_dot(fij, D)
                if sub_owner == over_owner and (ci[0] == "owner" or cj[0] == "owner"):
                    value = I(0)
                matrix[i][j] = matrix[j][i] = _round(value, precision)
        first.append(tuple(row))
        second.append(tuple(tuple(r) for r in matrix))
    geometry = tuple(_round(_dot(e, D), precision) for e in frames)
    return geometry, tuple(first), tuple(second)


@dataclass(frozen=True, slots=True)
class NativeCommonSourceCarrierBasis:
    basis: str
    source_geometry_elapsed_time_s: F
    complete_owner_histories_json: str
    source_radial_displacement_m: tuple
    sub_direction_bounds: tuple
    over_direction_bounds: tuple
    sub_position_bounds_m: tuple
    over_position_bounds_m: tuple
    receiver_director_bounds: tuple
    projected_geometry_bounds_m: tuple
    integrated_free_anchor_minus_geometric_reference_bounds_m: tuple
    scalar_first_derivative_bounds: tuple
    scalar_second_derivative_bounds: tuple
    source_traction_world_reactions: tuple
    precision: int


@dataclass(frozen=True, slots=True)
class NativeCommonSourceCarrier:
    point_id: str
    trace_id: str
    source_accounting_bucket: str
    original_area_m2: F
    computational_angle_exact_radians: F
    sub_owner: int
    over_owner: int
    donor_original_parent_id: str
    receiver_original_parent_id: str
    original_source_work_lineage_id: tuple
    complete_original_atom_history_json: str
    source_local_state_json: str
    complete_original_material_binding_json: str
    stored_free_anchors_m: tuple
    literal_source_tractions_pa: tuple
    stored_source_energy_components_j: tuple
    current_energy_and_work_stores_j: tuple
    series_stiffness_exact_pa_m: tuple
    series_stiffness_binary64_pa_m: tuple
    traction_rounding_residual_pa: tuple
    energy_rounding_residual_j: F
    archival_literal_world_reactions_json: str
    archival_literal_reaction_replay_certified: bool
    archival_mathematical_minus_literal_reaction_bounds: tuple
    archival_old_reference: NativeCommonSourceCarrierBasis
    current_objective: NativeCommonSourceCarrierBasis


_SCOPE_FLAGS = dict(
    all_sixteen_source_mechanical_carriers_retained=True,
    all_sixteen_archival_literal_world_reactions_replayed=True,
    full_source_owner_radial_first_second_variations_defined=True,
    integrated_source_free_anchors_rebased_without_reset=True,
    rotating_receiver_frame_couples_included=True,
    once_only_literal_atom_mechanism=True,
    current_objective_basis_equals_archival_force_basis=False,
    source_carriers_attached_to_authentic_common_material=False,
    old_rod_three_dimensional_material_map_certified=False,
    continuous_C_dtheta_measure_energized=False,
    source_atomic_and_continuous_measures_equated=False,
    hot_receiving_law_inferred=False,
    new_mixed_force_equilibrium_certified=False,
    whole_clock_all_pair_admission_certified=False,
    physical_material_admission_authorized=False,
    source_state_or_thermal_state_advanced=False,
    new_native_interval_accepted=False,
    original_point_1_completed=False,
)


@dataclass(frozen=True, slots=True)
class NativeCommonSourceCarrierSystem:
    source_snapshot_sha256: str
    reference_snapshot_sha256: str
    allocation_binding_sha256: str
    cut_set_snapshot_sha256: str
    source_epoch_sha256: str
    source_history_binding_sha256: str
    source_current_atom_states_sha256: str
    source_current_memory_sha256: str
    complete_source_dto_json: tuple
    complete_historical_saved_field_json: str
    complete_original_physical_history_json: str
    complete_current_source_memory_json: str
    complete_current_source_states_json: str
    complete_before_work_ledgers_json: str
    complete_unaccepted_candidate_work_ledgers_json: str
    complete_owner_rate_duration_conversion_json: str
    coordinate_order: tuple
    source_radius_m: F
    archival_geometry_elapsed_time_s: F
    current_geometry_elapsed_time_s: F
    source_contact_elapsed_time_s: F
    source_contact_step_count: int
    source_memory_elapsed_time_s: F
    source_memory_stage_count: int
    source_current_node_epoch: str
    carriers: tuple
    all16_original_area_m2: F
    all16_source_energy_exact_j: F
    all16_energy_and_work_stores_j: tuple
    active_source_atom_count: int
    inactive_source_atom_count: int
    provenance: str
    precision: int
    binding_sha256: str

    @property
    def scope(self):
        return SCOPE


def _matrix_histories(rows, *, paired, precision):
    records = tuple((owner, history) for owner, history in rows) if paired else tuple(
        (r["owner"], r["history"]) for r in rows)
    if len(records) != len({owner for owner, _ in records}):
        raise ValueError("Complete original owner histories cannot duplicate an owner")
    matrices = {owner: certify_rotation_history(tuple(rotation_increment_from_dict(step)
                for step in history), precision).intervals for owner, history in records}
    return records, matrices


def _basis(name, epoch, histories, matrices, radial, atom, binding, radius, order,
           traction, free, precision):
    point = atom["original_point"]
    q = tuple(I(_literal(v)) for v in point["position"])
    frames0 = tuple(tuple(I(_literal(v)) for v in point[k]) for k in ("normal", "down_dip"))
    sub, over = binding["subducting_plate"], binding["overriding_plate"]
    vs, vo = _map(matrices[sub], q), _map(matrices[over], q)
    ws, wo = radial
    ps, po = tuple((radius+ws)*x for x in vs), tuple((radius+wo)*x for x in vo)
    frames = tuple(_map(matrices[over], e) for e in frames0)
    geometry, first, second = _source_jet(order, sub, over, atom["trace_id"],
                                        (ps, po), (vs, vo), frames, precision)
    reactions = _world_reactions(_literal(atom["original_area_m2"]), tuple(I(x) for x in traction),
                                (ps, po), (vs, vo), frames, precision)
    return NativeCommonSourceCarrierBasis(name, epoch, _encoded(histories).decode(), radial,
        vs, vo, ps, po, frames, geometry, tuple(I(f)-h for f, h in zip(free, geometry)),
        first, second, reactions, precision)


def _archival_replay(atom, radius, archived):
    """Replay the literal existing force operation on its ACTUAL frozen basis."""
    p = atom["original_point"]
    m = atom["original_material"]
    q = tuple(p["position"])
    position = tuple(float(radius)*v for v in q)
    port = MaterialPairPort(atom["point_id"], atom["trace_id"], m["subducting_plate"],
        m["overriding_plate"], q, q, position, position, tuple(p["normal"]), tuple(p["down_dip"]))
    s = atom["unaccepted_candidate_state"]
    replay = material_pair_point_reactions(port, atom["original_area_m2"],
                                          (s["traction_normal_pa"], s["traction_shear_pa"]))
    if _encoded(replay) != _encoded(archived):
        raise ValueError("The all-sixteen archival world forces differ from their literal frozen-reference replay")
    return replay


def bind_native_common_source_carrier_system(allocation, *, reference, cuts, provenance, precision=96):
    reference = validate_native_common_reference(reference)
    cuts = validate_common_contact_cuts(cuts, reference=reference)
    allocation = validate_native_common_source_allocation(allocation, reference=reference, cuts=cuts)
    precision = _precision(precision)
    if type(provenance) is not str or not provenance.strip():
        raise ValueError("The complete source carrier needs an explicit reduced source-mechanism provenance")
    before = json.loads(dict(allocation.complete_source_dto_json)["before"])
    snapshot = before["snapshot"]
    history = before["original_physical_history"]
    archival = history["accepted_before"]["snapshot"]
    material = snapshot["current_material_epoch"]["source_material_state"]
    atoms = history["quadrature_histories"]
    states = dict(snapshot["states"])
    bindings = {r["trace_id"]: r for r in material["trace_bindings"]}
    source_owners = tuple(history["original_reference"]["plate_ids"])
    trace_order = tuple(row[0] for row in snapshot["transverse_displacement_m"])
    order = tuple(("owner", owner, axis) for owner in source_owners for axis in "xyz") + tuple(
        ("radial", tid, side) for tid in trace_order for side in ("sub", "over"))
    if (len(atoms) != 16 or len(states) != 16 or len(bindings) != 4 or len(order) != 14
            or len({r["point_id"] for r in atoms}) != 16
            or len(allocation.original_atom_ledger) != 16):
        raise ValueError("This source carrier requires every original sixteen atom/four-family mechanism")
    radius = _literal(float(history["origin_surface"]["radius_km"]*1000.))
    if radius != allocation.families[0].source_radius_m:
        raise ValueError("The literal physical source radius differs from the full source family")
    old_records, old_Q = _matrix_histories(archival["owner_histories"], paired=False, precision=precision)
    current_records, current_Q = _matrix_histories(snapshot["owner_histories"], paired=True, precision=precision)
    old_radial = {r[0]: tuple(_literal(x) for x in r[1:]) for r in archival["transverse_displacement_m"]}
    current_radial = {r[0]: tuple(_literal(x) for x in r[1:]) for r in snapshot["transverse_displacement_m"]}
    # The actual immutable source archive used the original zero-pose basis.
    # Other epochs require their own actual archival force producer, never a guessed pose.
    identity = [[float(i == j) for j in range(3)] for i in range(3)]
    if (any(rows for _, rows in old_records) or any(any(values) for values in old_radial.values())
            or _literal(archival["elapsed_time_s"]) != 0
            or any(matrix != identity for _, matrix in archival["owner_rotations"])):
        raise ValueError("This archival force replay requires its authentic original zero-pose basis")
    if (tuple(old_Q) != source_owners or tuple(current_Q) != source_owners
            or set(old_radial) != set(bindings) or set(current_radial) != set(bindings)
            or snapshot["transverse_displacement_m"] != history["actual_candidate"]["transverse_displacement_m"]):
        raise ValueError("The complete source owner/radial histories differ between the genuine epochs")
    raw_forces = history["actual_candidate"]["physical_points"]
    if len(raw_forces) != 16:
        raise ValueError("Every original archived world-force record is required")
    ledger = {r.point_id: r for r in allocation.original_atom_ledger}
    carriers = []
    for atom, archived_force in zip(atoms, raw_forces):
        identity = atom["point_id"]
        entry = ledger[identity]
        state = states[identity]
        if (state != atom["unaccepted_candidate_state"] or _encoded(atom).decode() != entry.complete_original_atom_history_json
                or _encoded(state).decode() != entry.complete_current_source_state_json
                or atom["original_area_m2"] != state["area_m2"]
                or atom["original_area_m2"] != atom["original_point"]["area_m2"]):
            raise ValueError("The complete source carrier cannot alter any original atom, measure or current state")
        binding = bindings[atom["trace_id"]]
        law, columns = atom["original_material"]["law"], atom["original_material"]["columns"]
        if (any(law[k] != 0. for k in ("friction_dry", "friction_wet", "cohesion_pa", "viscosity_pa_s_m"))
                or state["traction_normal_pa"] >= 0 or state["traction_shear_pa"] != 0.):
            raise ValueError("The authenticated current source must retain its compressed relaxed zero-strength shear branch")
        exact, binary = [], []
        for component in ("normal", "shear"):
            contact_key = "normal_stiffness_pa_m" if component == "normal" else "tangential_stiffness_pa_m"
            kn = _literal(law[contact_key])
            compliance = F(1)/kn
            float_compliance = 1./law[contact_key]
            for side in ("sub", "over"):
                L, E = columns[side+"_length_m"], columns[side+"_"+component+"_modulus_pa"]
                if _literal(L) <= 0 or _literal(E) <= 0:
                    raise ValueError("All source serial-column parameters must stay positive")
                compliance += _literal(L)/_literal(E)
                float_compliance += L/E
            exact.append(1/compliance)
            # Retain the old source's TWO-stage harmonic floating operation.
            col_stiff = 1./sum(columns[side+"_length_m"]/columns[side+"_"+component+"_modulus_pa"]
                                for side in ("sub", "over"))
            binary.append(_literal(law[contact_key]/(1.+law[contact_key]/col_stiff)))
        area = _literal(atom["original_area_m2"])
        free = _literal(state["gap_free_m"]), _literal(state["jump_free_m"])
        traction = _literal(state["traction_normal_pa"]), _literal(state["traction_shear_pa"])
        energy = _literal(state["contact_energy_j"]), _literal(state["bulk_energy_j"])
        plastic = _literal(state["plastic_slip_m"])
        residual = traction[0]-exact[0]*free[0], traction[1]-exact[1]*(free[1]-plastic)
        energy_residual = sum(energy)-area*sum((t*t/(2*k) for t, k in zip(traction, exact)), F(0))
        replay = _archival_replay(atom, radius, archived_force)
        old_basis = _basis(BASES[0], _literal(archival["elapsed_time_s"]), old_records, old_Q,
            old_radial[atom["trace_id"]], atom, binding, radius, order, traction, free, precision)
        current_basis = _basis(BASES[1], _literal(snapshot["elapsed_time_s"]), current_records, current_Q,
            current_radial[atom["trace_id"]], atom, binding, radius, order, traction, free, precision)
        differences = tuple((name, tuple(value-I(_literal(raw)) for value, raw in zip(values, replay[name])))
            for name, values in old_basis.source_traction_world_reactions)
        carriers.append(NativeCommonSourceCarrier(identity, atom["trace_id"], entry.source_accounting_bucket,
            area, entry.computational_angle_exact_radians, binding["subducting_plate"], binding["overriding_plate"],
            binding["donor_fragment_id"], binding["receiver_fragment_id"],
            (history["original_checkpoint_sha256"], identity), _encoded(atom).decode(), _encoded(state).decode(),
            _encoded(binding).decode(), free, traction, energy,
            tuple((key, _literal(state[key])) for key in STORE_FIELDS), tuple(exact), tuple(binary),
            residual, energy_residual, _encoded(replay).decode(), True, differences, old_basis, current_basis))
    duration = _literal(history["literal_dt_s"])
    myr = _literal(history["literal_dt_myr"])
    candidate = history["actual_candidate"]
    rates = dict(candidate["omega_rad_per_myr"])
    velocities = dict(zip(source_owners, candidate["angular_velocity_rad_s"]))
    increments = dict(zip(source_owners, candidate["rotation_increment_radians"]))
    conversions = dict(literal_dt_s=duration, literal_dt_myr=myr,
        omega_rad_per_myr=candidate["omega_rad_per_myr"], angular_velocity_rad_s=candidate["angular_velocity_rad_s"],
        rotation_increment_radians=candidate["rotation_increment_radians"],
        per_owner=tuple((owner, tuple(_literal(x)*myr for x in rates[owner]),
            tuple(_literal(x)*duration for x in velocities[owner]), tuple(_literal(x) for x in increments[owner]),
            tuple(_literal(x)*myr-_literal(y)*duration for x, y in zip(rates[owner], velocities[owner])))
            for owner in source_owners), angular_variation_units="radians", radial_variation_units="metres")
    total_work = tuple((key, sum((dict(c.current_energy_and_work_stores_j)[key] for c in carriers), F(0)))
                       for key in STORE_FIELDS)
    system = NativeCommonSourceCarrierSystem(allocation.source_snapshot_sha256, reference.reference_snapshot_sha256,
        allocation.binding_sha256, cuts.set_snapshot_sha256, allocation.source_epoch_sha256,
        allocation.source_history_binding_sha256, allocation.source_current_atom_states_sha256,
        allocation.source_current_memory_sha256, allocation.complete_source_dto_json,
        allocation.complete_historical_saved_field_json, allocation.complete_original_physical_history_json,
        allocation.complete_current_memory_json, allocation.complete_current_atom_states_json,
        _encoded(history["before_work_ledgers"]).decode(), _encoded(history["unaccepted_candidate_work_ledgers"]).decode(),
        _encoded(conversions).decode(), order, radius, _literal(archival["elapsed_time_s"]),
        allocation.source_native_elapsed_time_s, allocation.source_contact_elapsed_time_s,
        allocation.source_contact_step_count, allocation.source_memory_elapsed_time_s,
        allocation.source_memory_stage_count, allocation.source_current_node_epoch, tuple(carriers),
        sum((c.original_area_m2 for c in carriers), F(0)), sum((sum(c.stored_source_energy_components_j) for c in carriers), F(0)),
        total_work, allocation.active_current_receiver_source_atom_count, allocation.inactive_current_receiver_source_atom_count,
        provenance, precision, "")
    if (len({c.original_source_work_lineage_id for c in carriers}) != 16
            or system.active_source_atom_count != 8 or system.inactive_source_atom_count != 8):
        raise ValueError("All active and inactive source mechanisms must have one unique retained lineage")
    return _remember(system, (allocation, reference, cuts))


def validate_native_common_source_carrier_system(value, *, allocation=None, reference=None, cuts=None):
    actual_allocation, actual_reference, actual_cuts = _registered(value, NativeCommonSourceCarrierSystem)
    validate_native_common_source_allocation(actual_allocation, reference=actual_reference, cuts=actual_cuts)
    for supplied, actual in ((allocation, actual_allocation), (reference, actual_reference), (cuts, actual_cuts)):
        if supplied is not None and _encoded(supplied) != _encoded(actual):
            raise ValueError("The source carrier system belongs to another complete source allocation/reference/cutset")
    return value


@dataclass(frozen=True, slots=True)
class NativeCommonSourceCarrierResponse:
    source_snapshot_sha256: str
    system_binding_sha256: str
    point_id: str
    trace_id: str
    basis: str
    branch: str
    source_accounting_bucket: str
    coordinate_order: tuple
    coordinate_increments: tuple
    original_area_m2: F
    source_geometry_elapsed_time_s: F
    source_contact_elapsed_time_s: F
    source_contact_step_count: int
    source_current_node_epoch: str
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
    source_energy_exact_j: F
    stored_source_energy_components_j: tuple
    potential_bounds_j: I
    potential_gradient_bounds: tuple
    potential_hessian_bounds: tuple
    world_reactions: tuple
    source_reference_world_reactions: tuple
    series_stiffness_exact_pa_m: tuple
    series_stiffness_binary64_pa_m: tuple
    traction_rounding_residual_pa: tuple
    energy_rounding_residual_j: F
    proposed_plastic_increment_bounds_m: I
    source_local_state_json: str
    archival_literal_world_reactions_json: str
    archival_literal_reaction_replay_certified: bool
    source_work_lineage_id: tuple
    precision: int
    binding_sha256: str
    derivative_owner_chart: str = "FRESH_LEFT_EXPONENTIAL_CHART_AT_QUERY_ZERO"
    finite_owner_increments_supported: bool = False
    source_state_or_work_advanced: bool = False
    physical_common_material_handover_certified: bool = False
    old_force_archive_reproduced_in_current_pose: bool = False
    global_equilibrium_or_admission_authorized: bool = False


def evaluate_native_common_source_carrier(system, point_id, *, basis="current_objective",
        branch="relaxed_zero_strength_shear", coordinate_increments=None, precision=96):
    system = validate_native_common_source_carrier_system(system)
    precision = _precision(precision)
    if basis not in BASES or branch not in BRANCHES:
        raise ValueError("Declare the genuine source force basis and the explicit source shear branch")
    matches = tuple(c for c in system.carriers if c.point_id == point_id)
    if type(point_id) is not str or len(matches) != 1:
        raise ValueError("A source mechanical query requires exactly one literal original atom ID")
    carrier = matches[0]
    reference_basis = getattr(carrier, basis)
    count = len(system.coordinate_order)
    coordinates = (F(0),)*count if coordinate_increments is None else coordinate_increments
    if (type(coordinates) is not tuple or len(coordinates) != count or any(type(x) is not F for x in coordinates)
            or any(x for x, c in zip(coordinates, system.coordinate_order) if c[0] == "owner")):
        raise ValueError("Finite source queries require14 exact physical coordinates and zero finite owner increments; full owner2jets remain defined")
    if any(x for x, c in zip(coordinates, system.coordinate_order)
           if c[0] == "radial" and c[1] != carrier.trace_id):
        raise ValueError("A per-atom finite query cannot pretend other source carriers' radial variations are present")
    delta_s = next(x for x, c in zip(coordinates, system.coordinate_order) if c == ("radial", carrier.trace_id, "sub"))
    delta_o = next(x for x, c in zip(coordinates, system.coordinate_order) if c == ("radial", carrier.trace_id, "over"))
    radial = reference_basis.source_radial_displacement_m
    if any(system.source_radius_m+w+x <= 0 for w, x in zip(radial, (delta_s, delta_o))):
        raise ValueError("Original physical radial tips must remain at positive radii")
    vs, vo = reference_basis.sub_direction_bounds, reference_basis.over_direction_bounds
    ps = tuple(x+delta_s*v for x, v in zip(reference_basis.sub_position_bounds_m, vs))
    po = tuple(x+delta_o*v for x, v in zip(reference_basis.over_position_bounds_m, vo))
    frames = reference_basis.receiver_director_bounds
    geometry, first, second = _source_jet(system.coordinate_order, carrier.sub_owner, carrier.over_owner,
        carrier.trace_id, (ps, po), (vs, vo), frames, precision)
    # This correlated difference is exact0 at transfer, not I(h)-I(h).
    incremental_vector = tuple(delta_o*o-delta_s*s for s, o in zip(vs, vo))
    eta = tuple(_round(_dot(e, incremental_vector), precision) for e in frames)
    free = tuple(_round(I(f)+d, precision) for f, d in zip(carrier.stored_free_anchors_m, eta))
    relaxed = branch == BRANCHES[0]
    stiffness = carrier.series_stiffness_exact_pa_m[0], F(0) if relaxed else carrier.series_stiffness_exact_pa_m[1]
    traction = tuple(_round(I(t)+k*d, precision) for t, k, d in zip(carrier.literal_source_tractions_pa, stiffness, eta))
    if traction[0].hi >= 0:
        raise ValueError("The declared local source potential must remain on its compressed normal branch")
    area = carrier.original_area_m2
    energy0 = sum(carrier.stored_source_energy_components_j)
    energy = _round(I(energy0)+area*sum((I(t)*d+k*d*d/2 for t, k, d in
        zip(carrier.literal_source_tractions_pa, stiffness, eta)), I(0)), precision)
    gradient = tuple(_round(area*sum((traction[a]*first[a][i] for a in range(2)), I(0)), precision)
                     for i in range(count))
    hessian = [[I(0) for _ in range(count)] for _ in range(count)]
    for i in range(count):
        for j in range(i, count):
            value = _round(area*sum((stiffness[a]*first[a][i]*first[a][j]+traction[a]*second[a][i][j]
                for a in range(2)), I(0)), precision)
            hessian[i][j] = hessian[j][i] = value
    reactions = _world_reactions(area, traction, (ps, po), (vs, vo), frames, precision)
    response = NativeCommonSourceCarrierResponse(system.source_snapshot_sha256, system.binding_sha256,
        carrier.point_id, carrier.trace_id, basis, branch, carrier.source_accounting_bucket,
        system.coordinate_order, coordinates, area, reference_basis.source_geometry_elapsed_time_s,
        system.source_contact_elapsed_time_s, system.source_contact_step_count, system.source_current_node_epoch,
        ps, po, vs, vo, frames, geometry, eta, free, first, second, traction, stiffness, energy0,
        carrier.stored_source_energy_components_j, energy, gradient, tuple(tuple(row) for row in hessian),
        reactions, reference_basis.source_traction_world_reactions, carrier.series_stiffness_exact_pa_m,
        carrier.series_stiffness_binary64_pa_m, carrier.traction_rounding_residual_pa,
        carrier.energy_rounding_residual_j, eta[1] if relaxed else I(0), carrier.source_local_state_json,
        carrier.archival_literal_world_reactions_json, carrier.archival_literal_reaction_replay_certified,
        carrier.original_source_work_lineage_id, precision, "")
    return _remember(response, (system,))


def validate_native_common_source_carrier_response(value, *, system=None):
    actual, = _registered(value, NativeCommonSourceCarrierResponse)
    validate_native_common_source_carrier_system(actual)
    if system is not None and _encoded(system) != _encoded(actual):
        raise ValueError("The source mechanical response belongs to another registered whole-source system")
    return value


def native_common_source_carrier_system_to_dict(value):
    value = validate_native_common_source_carrier_system(value)
    raw = _payload(value)
    raw.update(format=FORMAT, model=MODEL, scope=SCOPE, **_SCOPE_FLAGS)
    return raw


def native_common_source_carrier_system_from_dict(raw, *, allocation, reference, cuts):
    if type(raw) is not dict:
        raise ValueError("The complete source mechanical system requires its literal JSON object")
    result = bind_native_common_source_carrier_system(allocation, reference=reference, cuts=cuts,
        provenance=raw.get("provenance"), precision=raw.get("precision"))
    if _encoded(raw) != _encoded(native_common_source_carrier_system_to_dict(result)):
        raise ValueError("Serialized source mechanics differ from all-sixteen dual-epoch source replay")
    return result


def native_common_source_carrier_response_to_dict(value):
    value = validate_native_common_source_carrier_response(value)
    raw = _payload(value)
    raw.update(format=RESPONSE_FORMAT, model=MODEL, scope=SCOPE)
    return raw


def _read_fraction(raw):
    if (type(raw) is not list or len(raw) != 2 or any(type(x) is not int for x in raw)
            or raw[1] <= 0):
        raise ValueError("Source mechanical coordinates require canonical rational JSON pairs")
    value = F(*raw)
    if [value.numerator, value.denominator] != raw:
        raise ValueError("Source mechanical coordinate fractions must be reduced")
    return value


def native_common_source_carrier_response_from_dict(raw, *, system):
    if type(raw) is not dict or type(raw.get("coordinate_increments")) is not list:
        raise ValueError("The source mechanical response requires its complete literal query coordinates")
    result = evaluate_native_common_source_carrier(system, raw.get("point_id"), basis=raw.get("basis"),
        branch=raw.get("branch"), coordinate_increments=tuple(_read_fraction(v) for v in raw["coordinate_increments"]),
        precision=raw.get("precision"))
    if _encoded(raw) != _encoded(native_common_source_carrier_response_to_dict(result)):
        raise ValueError("Serialized source mechanics differ from complete driver, force, anchor, energy and2jet replay")
    return result
