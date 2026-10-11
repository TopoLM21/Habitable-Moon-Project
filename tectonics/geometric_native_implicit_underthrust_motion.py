"""Original Gauss mechanics admitted by complete temporal material support.

This path retains the original objective, first-order force engine and its
explicit finite-anchor error bound.  Every mechanical atom remains in the
proved no-switch carrier core.  Full SO(3) temporal masks, the complete active
receiving line, original material banks and live clamps grant material
admission; a fixed endpoint arc's old refusal is never changed.

The continuous contact integral is an admission/transport certificate.  It
is never added to the original sixteen-atom mechanical work ledger.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from fractions import Fraction as F
import hashlib
import json
import math

import numpy as np
from scipy.spatial.transform import Rotation

from .geometric_bending_contact_dynamics import _displacements, _transverse_coordinates
from .geometric_contact_dynamics import _equilibrate_contact_network, _number
from .geometric_intrinsic_geometry import RotationIncrement, rotation_increment_to_dict
from .geometric_material_pair_reference import (
    _material_pair_network, material_pair_point_reactions, material_pair_exact_anchor_fields,
)
from .geometric_fault_traces import _arc_frame
from .geometric_native_active_contact_line import (
    certify_native_active_contact_line, native_active_contact_line_to_dict,
    validate_native_active_contact_line,
)
from .geometric_native_active_contact_line_work import (
    integrate_native_active_contact_line_work, native_active_contact_line_work_to_dict,
    validate_native_active_contact_line_work,
)
from .geometric_native_first_entry_interface_integral import (
    certify_native_first_entry_interface_integral, native_first_entry_interface_integral_to_dict,
)
from .geometric_native_implicit_force_support import (
    certify_native_implicit_force_support, native_implicit_force_support_to_dict,
)
from .geometric_native_implicit_material_epoch import (
    initialize_native_implicit_material_epoch, native_implicit_material_epoch_to_dict,
)
from .geometric_native_physical_history import (
    _literal_json, _typed_fingerprint, native_physical_history_binding_to_dict,
    validate_native_physical_history_binding,
)
from .geometric_native_underthrust_motion import _poses
from .geometric_spatial_underthrust import (
    advance_spatial_underthrust_memory, TrigPolynomial, trig_polynomial_extrema,
)
from .geometric_underthrust_motion import _domain, _sum_fields, _text
from .thermal import SECONDS_PER_MYR


FORMAT = "native-implicit-temporal-original-Gauss-underthrust-motion-1"
MODEL = "objective_first_order_original_Gauss_pairs_with_complete_temporal_material_admission"
_AUTHENTICATED = {}
_AUTHENTICATED_CANDIDATES = {}


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()


def _hash(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _r(value):
    return [value.numerator, value.denominator]


def _plain_fields(fields):
    return tuple((row[0], *(field.coefficients if type(field) is TrigPolynomial else tuple(field)
        for field in row[1:])) for row in fields)


@dataclass(frozen=True, slots=True)
class NativeImplicitTemporalAdmission:
    history: object
    line_certificate: object
    line_work: object
    interface_integral: object
    original_bank_support: object
    mechanical_atom_core_margins: tuple
    admission_sha256: str


def _admission_body(value):
    source = native_physical_history_binding_to_dict(value.history)
    return dict(format="native-implicit-temporal-material-admission-1",
        source_history_sha256=_hash(source), material_state_sha256=value.history.material_state.state_sha256,
        line_selector=native_active_contact_line_to_dict(value.line_certificate),
        line_work=native_active_contact_line_work_to_dict(value.line_work,
            expected_source_history_dto=source),
        material_coarea=native_first_entry_interface_integral_to_dict(value.interface_integral,
            expected_source_history_dto=source),
        original_bank_support=native_implicit_force_support_to_dict(value.original_bank_support),
        mechanical_atom_core_margins=[dict(point_id=identity,
            start_margin=_r(start), end_margin=_r(end))
            for identity, start, end in value.mechanical_atom_core_margins],
        temporal_material_gate="earliest positive owner entry with complete donor and receiver support",
        receiving_field="evolved globally selected receiver field at the same exact first-entry (theta,u)",
        donor_label="frozen on each active segment; change only at a proved positive fresh quarter seam",
        negative_fresh_seam_policy="proved negative entry retains existing material binding",
        archive_rule="once only; exit and reentry never allocate again",
        boundary_policy="closed ancestry retained; initial boundary and coarea critical set have zero material area",
        all_material_masks_on_complete_receiving_boundary=True,
        whole_line_strict_compression_and_monotone_down_dip=True,
        all_original_mechanical_atoms_keep_frozen_material_labels=True,
        endpoint_arc_refusal_overridden=False, continuous_work_added_to_Gauss_work=False,
        finite_area_assigned_to_isolated_material_label=False)


def certify_native_implicit_temporal_admission(history, *, law,
        expected_source_history_dto, line_certificate=None, line_work=None):
    """Join exact material coarea, active-line constitutive signs and live banks."""
    validate_native_physical_history_binding(history)
    source = native_physical_history_binding_to_dict(history)
    _literal_json(expected_source_history_dto)
    if type(expected_source_history_dto) is not dict or _encoded(source) != _encoded(expected_source_history_dto):
        raise ValueError("Temporal admission requires the complete independent solved source")
    line = (certify_native_active_contact_line(history, law=law,
        expected_source_history_dto=source) if line_certificate is None else line_certificate)
    validate_native_active_contact_line(line)
    if (_typed_fingerprint(line.history) != _typed_fingerprint(history)
            or _typed_fingerprint(line.law) != _typed_fingerprint(law)):
        raise ValueError("Temporal material admission changed the complete source or declared correspondence law")
    work = (integrate_native_active_contact_line_work(history, line_certificate=line,
        expected_source_history_dto=source) if line_work is None else line_work)
    validate_native_active_contact_line_work(work, expected_source_history_dto=source)
    if _typed_fingerprint(work.line_certificate) != _typed_fingerprint(line):
        raise ValueError("Temporal constitutive proof belongs to a different active selector")
    quarters = {row.receiver_trace_id: row for row in line.quarters}
    if len(quarters) != 4 or len(work.quarters) != 4:
        raise ValueError("Temporal admission needs every complete original receiving line")
    for quarter in work.quarters:
        proof = quarter.branch_proof
        if (proof.whole_line_gap_bounds_m.hi >= 0
                or proof.whole_line_jump_rate_bounds_m_per_fraction.hi >= 0
                or not proof.carried_state_at_switch):
            raise ValueError("An entire active line lacks compressive downward constitutive support")
    margins = []
    for atom in history.quadrature_histories:
        core = quarters[atom.trace_id].core_computational_angle_bounds
        start, end = atom.computational_angle-core.lo, core.hi-atom.computational_angle
        if min(start, end) <= 0:
            raise ValueError("A mechanical Gauss atom requires a freshly transferred force operator")
        margins.append((atom.point_id, start, end))
    if len(margins) != 16 or len({identity for identity, _, _ in margins}) != 16:
        raise ValueError("Temporal force admission omitted an original material atom")
    support = certify_native_implicit_force_support(history, expected_source_history_dto=source)
    integral = certify_native_first_entry_interface_integral(history,
        expected_source_history_dto=source, precision=max(128, line.precision),
        order=max(24, line.order))
    # The complete original receiver boundary and all four original donor
    # quarters define the 16 causal masks.  A point has one earliest positive
    # entry; summing its incident receiver ancestry does not allocate it twice.
    banks = history.material_state.trace_bindings
    expected_pairs = {(donor.donor_fragment_id, receiver.receiver_fragment_id)
        for donor in banks for receiver in banks}
    actual_pairs = {(mask.original_donor_fragment_id, mask.original_receiver_fragment_id)
        for mask in integral.carrier_masks}
    if (len(integral.carrier_masks) != 16 or actual_pairs != expected_pairs
            or len(integral.parent_measures) != 8
            or any(not mask.first_entry_only or mask.reentry_allocates_again
                for mask in integral.carrier_masks)):
        raise ValueError("Temporal admission lacks a complete once-only material carrier partition")
    value = NativeImplicitTemporalAdmission(history, line, work, integral, support, tuple(margins), "")
    return replace(value, admission_sha256=_hash(_admission_body(value)))


@dataclass(frozen=True, slots=True)
class NativeImplicitUnderthrustReceipt:
    sequence: int
    source_history_sha256: str
    before_sha256: str
    after_material_epoch_sha256: str
    dt_s: float
    provenance: str
    rotation_increment_radians: tuple
    admission_sha256: str
    source_work_j: float
    basal_dissipation_j: float
    bending_numerical_dissipation_j: float
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class NativeImplicitUnderthrustMotionState:
    history: object
    admission: NativeImplicitTemporalAdmission
    current_surface: object
    owner_histories: tuple
    states: tuple
    transverse_displacement_m: tuple
    free_anchor_fields: tuple
    spatial_memory: object
    elapsed_time_s: float
    cumulative_source_work_j: float
    cumulative_basal_dissipation_j: float
    cumulative_bending_numerical_dissipation_j: float
    receipts: tuple
    state_sha256: str

    @property
    def initial_mechanics(self):
        return self.history.initial_mechanics

    @property
    def plate_ids(self):
        return self.initial_mechanics.plate_ids

    @property
    def owner_rotations(self):
        return _poses(self.initial_mechanics, self.owner_histories, len(self.receipts))

    @property
    def current_material_epoch(self):
        return self.current_surface

    @property
    def preparation_work_j(self):
        return self.initial_mechanics.preparation_work_j

    @property
    def max_anchor_transport_error_m(self):
        return self.history.accepted_before.max_anchor_transport_error_m


@dataclass(frozen=True, slots=True)
class NativeImplicitUnderthrustMotionStep:
    state: NativeImplicitUnderthrustMotionState
    admission: NativeImplicitTemporalAdmission
    diagnostics: tuple


def _snapshot(state):
    return dict(current_material_epoch=native_implicit_material_epoch_to_dict(state.current_surface),
        owner_histories=[[owner, [rotation_increment_to_dict(step) for step in history]]
            for owner, history in state.owner_histories],
        states=[(identity, asdict(local)) for identity, local in state.states],
        transverse_displacement_m=state.transverse_displacement_m,
        free_anchor_fields=state.free_anchor_fields, spatial_memory=asdict(state.spatial_memory),
        elapsed_time_s=state.elapsed_time_s, cumulative_source_work_j=state.cumulative_source_work_j,
        cumulative_basal_dissipation_j=state.cumulative_basal_dissipation_j,
        cumulative_bending_numerical_dissipation_j=state.cumulative_bending_numerical_dissipation_j,
        preparation_work_j=state.preparation_work_j,
        admission_sha256=state.admission.admission_sha256,
        receipts=[asdict(receipt) for receipt in state.receipts],
        accepted_loaded_native_intervals=len(state.receipts),
        mechanical_model="original objective first-order Gauss ports with explicit finite-anchor error bound",
        continuous_contact_work_added_to_mechanical_ledger=False)


def accept_native_implicit_underthrust_motion(history, *, law,
        expected_source_history_dto, line_certificate=None, line_work=None):
    """Admit a genuinely solved original interval after all temporal proofs.

    The supplied history is authenticated by the unchanged genuine force
    solve.  This constructor neither prescribes its Omega nor changes any of
    its atoms, work, preparation, physical widths, dip or force operators.
    """
    admission = certify_native_implicit_temporal_admission(history, law=law,
        expected_source_history_dto=expected_source_history_dto,
        line_certificate=line_certificate, line_work=line_work)
    before, candidate = history.accepted_before, history.actual_candidate
    if before.receipts or before.elapsed_time_s != 0:
        raise ValueError("Original temporal admission cannot reset an already advanced physical epoch")
    metrics = dict(candidate.candidate_diagnostics)
    for key in ("material_reaction_relative_error", "angular_signal_relative_residual",
            "transverse_force_relative_residual", "cumulative_energy_balance_relative_error"):
        if not math.isfinite(metrics[key]) or metrics[key] >= 5e-10:
            raise ValueError("The actual original force candidate does not balance: "+key)
    if metrics["anchor_transport_error_m"] > before.max_anchor_transport_error_m:
        raise ValueError("The solved interval exceeds its declared finite-anchor approximation bound")
    epoch = initialize_native_implicit_material_epoch(history.material_state)
    if epoch.phase != "partition":
        raise ValueError("Temporal material transaction has no complete actual endpoint partition")
    velocities = dict(candidate.omega_rad_per_myr)
    histories = tuple((owner, old+(RotationIncrement(velocities[owner], history.literal_dt_myr),))
        for owner, old in before.owner_histories)
    source_work = math.fsum((before.cumulative_source_work_j, metrics["source_work_j"]))
    basal = math.fsum((before.cumulative_basal_dissipation_j, metrics["basal_dissipation_j"]))
    bending_loss = math.fsum((before.cumulative_bending_numerical_dissipation_j,
        metrics["bending_numerical_dissipation_j"]))
    elapsed = math.fsum((before.elapsed_time_s, candidate.dt_s))
    if elapsed <= before.elapsed_time_s:
        raise ValueError("The accepted physical clock cannot resolve its actual interval")
    receipt = NativeImplicitUnderthrustReceipt(1, _hash(expected_source_history_dto),
        _hash(json.loads(history.source_before_payload_json)), epoch.epoch_sha256,
        candidate.dt_s, candidate.provenance, candidate.rotation_increment_radians,
        admission.admission_sha256, metrics["source_work_j"], metrics["basal_dissipation_j"],
        metrics["bending_numerical_dissipation_j"], "")
    receipt_body = asdict(receipt)
    receipt_body.pop("receipt_sha256")
    receipt = replace(receipt, receipt_sha256=_hash(receipt_body))
    state = NativeImplicitUnderthrustMotionState(history, admission, epoch, histories,
        candidate.states, candidate.transverse_displacement_m, _plain_fields(candidate.spatial_step.anchor_fields),
        candidate.spatial_step.memory, elapsed, source_work, basal, bending_loss, (receipt,), "")
    radius = epoch.radius_km*1000.
    radial = _transverse_coordinates(state.initial_mechanics.fault_traces, state.transverse_displacement_m)
    coordinates = np.r_[np.asarray(candidate.rotation_increment_radians).reshape(-1),
        radial.reshape(-1)/radius]
    bending_energy = .5*float(coordinates@np.asarray(history.original_reference.stiffness)@coordinates)
    accounted = math.fsum((*(local.cumulative_external_work_j for _, local in state.states),
        basal, bending_energy, bending_loss))
    supplied = math.fsum((state.preparation_work_j, source_work))
    balance = abs(accounted-supplied)/max(abs(accounted), abs(supplied), np.finfo(float).tiny)
    if balance >= 5e-10:
        raise ValueError("Accepted source, original Gauss contacts and beams do not balance")
    metrics["accepted_cumulative_energy_balance_relative_error"] = balance
    state = replace(state, state_sha256=_hash(_snapshot(state)))
    if len(_AUTHENTICATED) >= 8:
        _AUTHENTICATED.clear()
    _AUTHENTICATED[_typed_fingerprint(state)] = state.state_sha256
    return NativeImplicitUnderthrustMotionStep(state, admission, tuple(sorted(metrics.items())))


def validate_native_implicit_underthrust_motion_state(state):
    if type(state) is not NativeImplicitUnderthrustMotionState:
        raise ValueError("Expected a complete typed implicit loaded state")
    fingerprint = _typed_fingerprint(state)
    if _AUTHENTICATED.get(fingerprint) != state.state_sha256:
        replay = accept_native_implicit_underthrust_motion(state.history,
            law=state.admission.line_certificate.law,
            expected_source_history_dto=native_physical_history_binding_to_dict(state.history),
            line_certificate=state.admission.line_certificate, line_work=state.admission.line_work).state
        if _typed_fingerprint(replay) != fingerprint:
            raise ValueError("Implicit loaded state differs from complete force, temporal material and support replay")
    return state


def native_implicit_underthrust_motion_to_dict(state):
    validate_native_implicit_underthrust_motion_state(state)
    body = dict(format=FORMAT, model=MODEL,
        original_physical_history=native_physical_history_binding_to_dict(state.history),
        transport_law=asdict(state.admission.line_certificate.law),
        temporal_admission=_admission_body(state.admission),
        snapshot=_snapshot(state), state_sha256=state.state_sha256)
    return json.loads(_encoded(body))


def native_implicit_underthrust_motion_from_dict(raw, *, expected_history,
        expected_source_history_dto, expected_law):
    _literal_json(raw)
    if type(raw) is not dict:
        raise ValueError("Implicit loaded restart needs a complete explicit object")
    try:
        proof = raw["temporal_admission"]
        selector = proof["line_selector"]
        work = proof["line_work"]
        line = certify_native_active_contact_line(expected_history, law=expected_law,
            expected_source_history_dto=expected_source_history_dto,
            precision=selector["precision"], order=selector["order"])
        integral = integrate_native_active_contact_line_work(expected_history,
            expected_source_history_dto=expected_source_history_dto, line_certificate=line,
            precision=work["precision"], order=work["order"])
        replay = accept_native_implicit_underthrust_motion(expected_history, law=expected_law,
            expected_source_history_dto=expected_source_history_dto,
            line_certificate=line, line_work=integral).state
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Implicit loaded restart has an incomplete source or proof schema") from error
    if _encoded(raw) != _encoded(native_implicit_underthrust_motion_to_dict(replay)):
        raise ValueError("Implicit loaded checkpoint differs from independent complete force and material replay")
    return replay


@dataclass(frozen=True, slots=True)
class NativeImplicitForceCandidate:
    before_state_sha256: str
    current_load_sha256: str
    dt_s: float
    provenance: str
    rotation_increment_radians: tuple
    omega_rad_per_myr: tuple
    states: tuple
    transverse_displacement_m: tuple
    free_anchor_fields: tuple
    spatial_step: object
    reactions_payload_json: bytes
    diagnostics: tuple

    @property
    def reactions(self):
        return json.loads(self.reactions_payload_json)


def solve_native_implicit_underthrust_candidate(mesh, previous, load, dt_s, *, provenance):
    """Solve on current original atoms; do not accept an unproved next closure."""
    from .geometric_native_implicit_mantle_loading import validate_native_implicit_physical_spatial_mantle_load

    validate_native_implicit_underthrust_motion_state(previous)
    validate_native_implicit_physical_spatial_mantle_load(mesh, previous.current_surface, load)
    dt, provenance = _number(dt_s, "Implicit physical interval", positive=True), _text(provenance)
    seed = previous.initial_mechanics
    original, _, points, ports, rows, operators = _material_pair_network(previous.history.origin_surface,
        seed, previous.owner_rotations, previous.transverse_displacement_m,
        roundoff_steps=len(previous.receipts))
    radius, rc = previous.current_surface.radius_km*1000., 3*len(previous.plate_ids)
    stiffness, operators = np.asarray(original.stiffness), np.asarray(operators)
    radial_old = _transverse_coordinates(seed.fault_traces, previous.transverse_displacement_m)
    zold = np.r_[np.zeros(rc), radial_old.reshape(-1)/radius]
    anchors = np.asarray([(local.gap_free_m, local.jump_free_m) for _, local in previous.states])
    offsets = anchors-np.einsum("ijk,k->ij", operators, zold)
    drag = np.zeros(stiffness.shape)
    drag[:rc, :rc] = np.asarray(load.drag_nm_s)
    if load.plate_ids != previous.plate_ids:
        raise ValueError("Implicit force source cannot replace original physical owners")
    def domain(z):
        return _domain(previous, np.asarray(z[:rc]).reshape((-1, 3)),
            np.asarray(z[rc:]).reshape((-1, 2))*radius)
    solved = _equilibrate_contact_network(zold, drag,
        np.pad(np.asarray(load.torque_nm), (0, len(zold)-rc)), np.asarray(rows),
        np.asarray(original.areas), np.asarray(original.frames), original.materials,
        offsets, tuple(local for _, local in previous.states), dt, seed.max_rotation_radians,
        anchor_operators=operators, elastic_stiffness=stiffness,
        rotation_dof_count=rc, domain_check=domain, common_rotation_gauge=True)
    increments = solved.q[:rc].reshape((-1, 3))
    radial = solved.q[rc:].reshape((-1, 2))*radius
    fields = _domain(previous, increments, radial)
    states = tuple((point.point_id, step.state) for point, step in zip(points, solved.steps))
    spatial = advance_spatial_underthrust_memory(seed, previous.spatial_memory, fields, dt,
        previous_point_states=previous.states, current_point_states=states)
    velocities = tuple((owner, tuple(map(float, row/(dt/SECONDS_PER_MYR))))
        for owner, row in zip(previous.plate_ids, increments))
    reactions = tuple(material_pair_point_reactions(port, area, step.traction_pa)
        for port, area, step in zip(ports, original.areas, solved.steps))
    # Retain the same independent physical angular/radial virtual-work and
    # finite-anchor gates as the original engine.  An unaccepted trial's
    # changed rotations never acquire the previous interval's certificate.
    histories = tuple((owner, history+(RotationIncrement(dict(velocities)[owner], dt/SECONDS_PER_MYR),))
        for owner, history in previous.owner_histories)
    poses = _poses(seed, histories, len(previous.receipts)+1)
    matrices = [np.asarray(matrix) for _, matrix in poses]
    relative_angle = max((np.linalg.norm(Rotation.from_matrix(first@second.T).as_rotvec())
        for first in matrices for second in matrices), default=0.)
    if relative_angle > seed.max_rotation_radians:
        raise ValueError("Fresh implicit candidate exceeds the original finite relative-pose bound")
    exact = material_pair_exact_anchor_fields(seed, poses,
        _displacements(seed.fault_traces, radial), roundoff_steps=len(previous.receipts)+1)
    prepared = tuple((trace.trace_id, (trace.initial_gap_m, 0., 0., 0., 0.),
        (trace.initial_jump_m, 0., 0., 0., 0.)) for trace in seed.fault_traces.traces)
    exact = _sum_fields(prepared, exact)
    defect = 0.
    for trace, actual, integrated in zip(seed.fault_traces.traces, exact, fields):
        extent = _arc_frame(trace.start, trace.end)[3]
        for first, second in zip(actual[1:], integrated[1:]):
            extrema = trig_polynomial_extrema(TrigPolynomial(tuple(x-y for x, y in zip(first, second))), 0., extent)
            defect = max(defect, *(abs(value) for value in extrema))
    if defect > previous.max_anchor_transport_error_m:
        raise ValueError("Fresh implicit candidate exceeds its explicit finite-pose anchor bound")
    physical = np.zeros(len(zold))
    positions = {owner: index for index, owner in enumerate(previous.plate_ids)}
    trace_positions = {trace.trace_id: index for index, trace in enumerate(seed.fault_traces.traces)}
    for port, reaction in zip(ports, reactions):
        physical[3*positions[port.subducting_plate]:3*positions[port.subducting_plate]+3] += reaction["torque_on_sub_nm"]
        physical[3*positions[port.overriding_plate]:3*positions[port.overriding_plate]+3] += reaction["torque_on_over_nm"]
        start = rc+2*trace_positions[port.trace_id]
        physical[start:start+2] += radius*np.asarray(reaction["radial_force_n"])
    eps = np.finfo(float).eps
    scale = math.fsum(float(np.linalg.norm(row)) for row in (physical, solved.contact_force))
    if np.linalg.norm(physical+solved.contact_force) > 512*eps*max(scale, np.finfo(float).tiny):
        raise ValueError("Fresh implicit reactions disagree with original generalized virtual work")
    radial_contact, radial_bending = physical[rc:]/radius, -solved.elastic_force[rc:]/radius
    radial_residual = radial_contact+radial_bending
    radial_scale = max(np.linalg.norm(radial_contact), np.linalg.norm(radial_bending), np.finfo(float).tiny)
    radial_operands = np.linalg.norm(radial_bending)+math.fsum(np.linalg.norm(row["radial_force_n"]) for row in reactions)
    angular_scale = max(np.linalg.norm(load.torque_nm), np.linalg.norm(solved.drag_force[:rc]),
        np.linalg.norm(physical[:rc]), np.finfo(float).tiny)
    angular_operands = np.linalg.norm(load.torque_nm)+np.linalg.norm(solved.drag_force[:rc])+math.fsum(
        math.sqrt(2.)*np.linalg.norm(row["torque_on_sub_nm"]) for row in reactions)
    if (np.linalg.norm(radial_residual) > 2e-10*radial_scale+512*eps*radial_operands
            or np.linalg.norm(solved.residual[:rc]) > 2e-10*angular_scale+512*eps*angular_operands):
        raise ValueError("Fresh implicit angular or transverse physical balance failed")
    metrics = dict(solved.diagnostics)
    accounted = math.fsum((*(local.cumulative_external_work_j for _, local in states),
        math.fsum((previous.cumulative_basal_dissipation_j, metrics["basal_dissipation_j"])),
        .5*float(solved.q@stiffness@solved.q),
        math.fsum((previous.cumulative_bending_numerical_dissipation_j,
            metrics["bending_numerical_dissipation_j"]))))
    supplied = math.fsum((previous.preparation_work_j,
        math.fsum((previous.cumulative_source_work_j, metrics["source_work_j"]))))
    balance = abs(accounted-supplied)/max(abs(accounted), abs(supplied), np.finfo(float).tiny)
    if balance >= 5e-10:
        raise ValueError("Fresh implicit candidate source, contacts and beams do not balance")
    metrics.update(anchor_transport_error_m=defect, relative_owner_angle_radians=float(relative_angle),
        material_reaction_relative_error=float(np.linalg.norm(physical+solved.contact_force))/max(scale, np.finfo(float).tiny),
        transverse_force_relative_residual=float(np.linalg.norm(radial_residual)/radial_scale),
        angular_signal_relative_residual=float(np.linalg.norm(solved.residual[:rc])/angular_scale),
        cumulative_energy_balance_relative_error=balance)
    candidate = NativeImplicitForceCandidate(previous.state_sha256, load.load_sha256, dt, provenance,
        tuple(tuple(map(float, row)) for row in increments), velocities, states,
        _displacements(seed.fault_traces, radial), fields, spatial, _encoded(reactions), tuple(sorted(metrics.items())))
    if len(_AUTHENTICATED_CANDIDATES) >= 16:
        _AUTHENTICATED_CANDIDATES.clear()
    _AUTHENTICATED_CANDIDATES[_typed_fingerprint(candidate)] = (
        previous.state_sha256, _typed_fingerprint(load))
    return candidate


def validate_native_implicit_force_candidate(candidate, *, previous, load):
    if (type(candidate) is not NativeImplicitForceCandidate
            or _AUTHENTICATED_CANDIDATES.get(_typed_fingerprint(candidate))
                != (previous.state_sha256, _typed_fingerprint(load))
            or candidate.before_state_sha256 != previous.state_sha256
            or candidate.current_load_sha256 != load.load_sha256):
        raise ValueError("Implicit candidate was not produced by the complete genuine current force solve")
    validate_native_implicit_underthrust_motion_state(previous)
    return candidate


def native_implicit_force_candidate_to_dict(candidate, *, previous, load):
    validate_native_implicit_force_candidate(candidate, previous=previous, load=load)
    body = dict(format="native-implicit-unaccepted-force-candidate-1",
        before_state_sha256=candidate.before_state_sha256, current_load_sha256=candidate.current_load_sha256,
        dt_s=candidate.dt_s, provenance=candidate.provenance,
        rotation_increment_radians=candidate.rotation_increment_radians,
        omega_rad_per_myr=candidate.omega_rad_per_myr,
        states=[(identity, asdict(local)) for identity, local in candidate.states],
        transverse_displacement_m=candidate.transverse_displacement_m,
        free_anchor_fields=candidate.free_anchor_fields,
        spatial_step=asdict(candidate.spatial_step), reactions=candidate.reactions,
        diagnostics=dict(candidate.diagnostics), candidate_mechanics_adopted=False,
        new_temporal_material_admission_required=True,
        before_accepted_intervals=len(previous.receipts))
    return json.loads(_encoded(body))


class UnresolvedNativeImplicitUnderthrustMotion(ValueError):
    """A genuine subsequent solve whose new material has no physical pair.

    The candidate material partition is separate evidence.  Neither it nor
    the balanced force trial replaces the preceding accepted state.
    """

    def __init__(self, previous, load, candidate, material_candidate, temporal_refusal,
            preserved_before_bytes):
        self.before_state = previous
        self.current_load = load
        self.candidate = candidate
        self.material_candidate = material_candidate
        self.temporal_refusal = temporal_refusal
        self.preserved_before_bytes = preserved_before_bytes
        super().__init__("Positive new-material first entry requires its own declared physical contact pair, "
            "constitutive memory and finite material-bank support")


def assess_native_implicit_underthrust_candidate(mesh, previous, load, candidate, *,
        material_precision=128, material_order=20, material_time_cells=1):
    """Check fresh temporal material without adopting an unsupported force trial."""
    from .geometric_native_implicit_mantle_loading import (
        validate_native_implicit_physical_spatial_mantle_load,
    )
    from .geometric_native_implicit_material_continuation import (
        advance_native_implicit_material_epoch,
    )
    from .geometric_native_implicit_epoch_admission import (
        NativeImplicitEpochAdmissionRefusal,
        certify_native_implicit_epoch_admission,
        validate_native_implicit_epoch_admission_refusal,
    )

    validate_native_implicit_force_candidate(candidate, previous=previous, load=load)
    validate_native_implicit_physical_spatial_mantle_load(mesh, previous.current_surface, load)
    preserved = _encoded(native_implicit_underthrust_motion_to_dict(previous))
    material = advance_native_implicit_material_epoch(previous.current_surface,
        dict(candidate.omega_rad_per_myr), candidate.dt_s/SECONDS_PER_MYR,
        precision=material_precision, order=material_order, time_cells=material_time_cells,
        provenance="Unadopted material continuation of the genuinely solved subsequent force interval: "
            +candidate.provenance)
    refusal = certify_native_implicit_epoch_admission(previous, candidate, material, load=load)
    if type(refusal) is not NativeImplicitEpochAdmissionRefusal:
        raise ValueError("A subsequent trial needs independently proved temporal physical admission")
    validate_native_implicit_epoch_admission_refusal(refusal, expected_previous=previous,
        expected_candidate=candidate, expected_load=load, expected_material_epoch=material)
    if _encoded(native_implicit_underthrust_motion_to_dict(previous)) != preserved:
        raise ValueError("A refused temporal material trial changed the preceding accepted state")
    raise UnresolvedNativeImplicitUnderthrustMotion(previous, load, candidate, material,
        refusal, preserved)


def advance_native_implicit_underthrust_motion(mesh, previous, load, dt_s, *, provenance,
        material_precision=128, material_order=20, material_time_cells=1):
    """Solve a fresh differential interval and enforce its actual temporal gate."""
    candidate = solve_native_implicit_underthrust_candidate(mesh, previous, load, dt_s,
        provenance=provenance)
    return assess_native_implicit_underthrust_candidate(mesh, previous, load, candidate,
        material_precision=material_precision, material_order=material_order,
        material_time_cells=material_time_cells)
