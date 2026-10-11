"""Atomic differential motion of persistent original material-bank pairs.

This bounded model retains the original column ports and whole beam domains.
It does not replace their histories with the current trench's new neighbours.
Objective operators, source and geometry are frozen at each interval's start;
the resulting Euler increments then drive a genuine surface/archive transaction.
Finite-pose anchor error has an explicit caller-supplied physical bound.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math

import numpy as np
from scipy.spatial.transform import Rotation

from .geometric_basal_sources import basal_source_from_dict, basal_source_to_dict
from .geometric_bending import bank_max_surface_strain, bank_max_tip_slope
from .geometric_bending_contact_dynamics import (
    _displacements, _transverse_coordinates, bending_contact_dynamics_from_dict,
    bending_contact_dynamics_to_dict, validate_bending_contact_dynamics_state,
)
from .geometric_bending_motion import _trusted_source
from .geometric_boundary import (
    boundary_from_dict, boundary_to_dict, consume_geometric_transaction,
    polarity_evidence, surface_digest,
)
from .geometric_boundary_io import _validate_pair
from .geometric_buried_bank_support import (
    initial_buried_bank_registry, advance_buried_bank_registry,
    validate_buried_bank_support,
)
from .geometric_contact_dynamics import (
    _basal, _encode, _equilibrate_contact_network, _number, UnresolvedContactDynamics,
)
from .geometric_fault_traces import _arc_frame
from .geometric_polarity import PolarityResolution
from .geometric_material_pair_reference import (
    canonical_material_pair_poses, current_material_pair_reference,
    material_pair_exact_anchor_fields, material_pair_increment_fields,
    material_pair_point_reactions,
)
from .geometric_spatial_underthrust import (
    TrigPolynomial, advance_spatial_underthrust_memory,
    initialize_spatial_underthrust_memory, make_spatial_underthrust_resolver,
    trig_polynomial_extrema,
)
from .geometric_transport import advance_geometric_surface
from .fractional_surface import SurfaceParcel
from .thermal import SECONDS_PER_MYR

FORMAT = "geometric-persistent-pair-underthrust-motion-1"
MODEL = "objective_incremental_original_material_pairs_spatial_underthrust"
_EPS = np.finfo(float).eps


def _hash(value):
    return hashlib.sha256(_encode(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class HotBirthParameters:
    """Explicit hot material closure for geometrical gaps, without cooling."""
    crust_thickness_km: float
    damage: float
    water: float
    provenance: str


@dataclass(frozen=True, slots=True)
class UnderthrustReceipt:
    sequence: int
    current_load: object
    dt_s: float
    provenance: str
    rotation_increment_radians: tuple
    before_sha256: str
    after_sha256: str
    source_work_j: float
    basal_dissipation_j: float
    bending_numerical_dissipation_j: float
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class UnderthrustMotionState:
    origin_surface_sha256: str
    initial_inventory: object
    initial_mechanics: object
    initial_load: object
    birth_parameters: HotBirthParameters
    max_anchor_transport_error_m: float
    current_surface: object
    current_inventory: object
    owner_rotations: tuple
    states: tuple
    transverse_displacement_m: tuple
    free_anchor_fields: tuple
    spatial_memory: object
    buried_registry: object
    elapsed_time_s: float
    cumulative_source_work_j: float
    cumulative_basal_dissipation_j: float
    cumulative_bending_numerical_dissipation_j: float
    receipts: tuple

    @property
    def plate_ids(self):
        return self.initial_mechanics.plate_ids

    @property
    def preparation_work_j(self):
        return self.initial_mechanics.preparation_work_j


@dataclass(frozen=True, slots=True)
class UnderthrustMotionStep:
    state: UnderthrustMotionState
    rotation_increment_radians: tuple
    angular_velocity_rad_s: tuple
    points: tuple
    geometric_transaction: object
    spatial_step: object
    support_receipt: object
    admission_audit: object
    diagnostics: tuple
    model: str = MODEL


def _body(state):
    raw = asdict(state)
    for key in ("initial_inventory", "initial_mechanics", "initial_load", "receipts"):
        raw.pop(key)
    return raw


def _seal(receipt):
    raw = asdict(receipt)
    raw.pop("receipt_sha256")
    return replace(receipt, receipt_sha256=_hash(raw))


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Physical underthrust stage requires explicit provenance")
    return value


def _birth_parameters(value):
    if not isinstance(value, HotBirthParameters):
        raise ValueError("Underthrust requires explicit hot birth parameters")
    _number(value.crust_thickness_km, "Hot crust thickness", positive=True)
    for name in ("damage", "water"):
        if not 0. <= _number(getattr(value, name), name) <= 1.:
            raise ValueError("Hot birth damage and water must lie in [0,1]")
    _text(value.provenance)
    return value


def initialize_underthrust_motion(mesh, origin, inventory, initial_mechanics, initial_load, *,
                                 birth_parameters, max_anchor_transport_error_m):
    """Retain a homogeneous prepared material epoch without resetting memory."""
    _validate_pair(origin, inventory)
    _trusted_source(mesh, origin, initial_load)
    validate_bending_contact_dynamics_state(origin, initial_mechanics, initial_load)
    seed = initial_mechanics
    if any(v != 0. for row in seed.rotation_radians for v in row) or any(
            v != 0. for row in seed.transverse_displacement_m for v in row[1:]):
        raise ValueError("Persistent-pair initialization requires the original unadvanced pose")
    memory = initialize_spatial_underthrust_memory(seed)
    poses = canonical_material_pair_poses(seed, {owner: np.eye(3) for owner in seed.plate_ids})
    registry = initial_buried_bank_registry(origin, inventory)
    state = UnderthrustMotionState(surface_digest(origin), inventory, seed, initial_load,
        _birth_parameters(birth_parameters),
        _number(max_anchor_transport_error_m, "Finite pose anchor error bound", positive=True),
        origin, inventory, poses, seed.states, seed.transverse_displacement_m,
        tuple((trace.trace_id, (trace.initial_gap_m, 0., 0., 0., 0.),
               (trace.initial_jump_m, 0., 0., 0., 0.)) for trace in seed.fault_traces.traces),
        memory, registry, seed.elapsed_time_s, seed.cumulative_source_work_j,
        seed.cumulative_basal_dissipation_j, seed.cumulative_bending_numerical_dissipation_j, ())
    current_material_pair_reference(origin, seed, origin, poses, state.transverse_displacement_m,
        inventory=inventory, buried_registry=registry)
    return state


def _sum_fields(first, second):
    if tuple(row[0] for row in first) != tuple(row[0] for row in second):
        raise ValueError("Original material trace fields changed their identity")
    return tuple((a[0], *(tuple(math.fsum((x, y)) for x, y in zip(f, g))
        for f, g in zip(a[1:], b[1:]))) for a, b in zip(first, second))


def _bounds(seed, fields):
    return tuple((trace.trace_id, *trig_polynomial_extrema(TrigPolynomial(field[1]),
        0., _arc_frame(trace.start, trace.end)[3]))
        for trace, field in zip(seed.fault_traces.traces, fields))


def _domain(previous, increment, radial):
    seed, radius = previous.initial_mechanics, previous.current_surface.radius_km*1000.
    if (not np.isfinite(increment).all() or not np.isfinite(radial).all()
            or np.max(np.linalg.norm(increment, axis=1), initial=0.) > seed.max_rotation_radians):
        raise UnresolvedContactDynamics("Underthrust exceeds the explicit angular increment bound")
    for parameter, values in zip(seed.bending_parameters, radial):
        for bank, w in zip((parameter.sub, parameter.over), values):
            if (bank_max_tip_slope(bank, float(w)) > seed.max_tip_slope
                    or bank_max_surface_strain(bank, float(w)) > seed.max_bending_strain
                    or abs(float(w))/radius > seed.max_deflection_radius_ratio):
                raise UnresolvedContactDynamics("Underthrust exceeds the explicit beam deformation bounds")
    fields = _sum_fields(previous.free_anchor_fields, material_pair_increment_fields(seed,
        previous.owner_rotations, previous.transverse_displacement_m, increment,
        _displacements(seed.fault_traces, radial), roundoff_steps=len(previous.receipts)))
    if any(high > 0. for _, _, high in _bounds(seed, fields)):
        raise UnresolvedContactDynamics("Continuous material-pair field requires unsupported opening")
    return fields


def _advance(mesh, origin, previous, load, dt_s, provenance):
    dt, provenance = _number(dt_s, "Physical underthrust interval", positive=True), _text(provenance)
    before, seed = previous.current_surface, previous.initial_mechanics
    _trusted_source(mesh, before, load)
    reference = current_material_pair_reference(origin, seed, before, previous.owner_rotations,
        previous.transverse_displacement_m, inventory=previous.current_inventory,
        buried_registry=previous.buried_registry, roundoff_steps=len(previous.receipts))
    ids, drag, torque, _ = _basal(before, load)
    if ids != previous.plate_ids:
        raise ValueError("Underthrust cannot replace original material owners")
    rc, radius = 3*len(ids), before.radius_km*1000.
    operators, stiffness = np.asarray(reference.operators), np.asarray(reference.stiffness)
    radial_old = _transverse_coordinates(seed.fault_traces, previous.transverse_displacement_m)
    zold = np.r_[np.zeros(rc), radial_old.reshape(-1)/radius]
    anchors = np.asarray([(local.gap_free_m, local.jump_free_m) for _, local in previous.states])
    offsets = anchors-np.einsum("ijk,k->ij", operators, zold)
    augmented_drag = np.zeros(stiffness.shape)
    augmented_drag[:rc, :rc] = drag
    def domain(z):
        return _domain(previous, np.asarray(z[:rc]).reshape((-1, 3)),
                       np.asarray(z[rc:]).reshape((-1, 2))*radius)
    solved = _equilibrate_contact_network(zold, augmented_drag,
        np.pad(torque, (0, len(zold)-rc)), np.asarray(reference.rows),
        np.asarray(reference.areas), np.asarray(reference.frames), reference.materials,
        offsets, tuple(local for _, local in previous.states), dt, seed.max_rotation_radians,
        anchor_operators=operators, elastic_stiffness=stiffness,
        rotation_dof_count=rc, domain_check=domain, common_rotation_gauge=True)
    increments = solved.q[:rc].reshape((-1, 3))
    radial = solved.q[rc:].reshape((-1, 2))*radius
    fields = _domain(previous, increments, radial)
    states = tuple((point.point_id, step.state) for point, step in zip(reference.points, solved.steps))
    spatial = advance_spatial_underthrust_memory(seed, previous.spatial_memory, fields, dt,
        previous_point_states=previous.states, current_point_states=states)
    poses = canonical_material_pair_poses(seed, tuple((owner, tuple(map(tuple,
        Rotation.from_rotvec(delta).as_matrix()@np.asarray(matrix))))
        for (owner, matrix), delta in zip(previous.owner_rotations, increments)),
        roundoff_steps=len(previous.receipts)+1)
    matrices = [np.asarray(matrix) for _, matrix in poses]
    relative_angle = max((np.linalg.norm(Rotation.from_matrix(a@b.T).as_rotvec())
        for a in matrices for b in matrices), default=0.)
    if relative_angle > seed.max_rotation_radians:
        raise UnresolvedContactDynamics("Persistent material pairs exceeded their finite relative-pose bound")
    exact = material_pair_exact_anchor_fields(seed, poses, _displacements(seed.fault_traces, radial),
        roundoff_steps=len(previous.receipts)+1)
    seed_fields = tuple((t.trace_id, (t.initial_gap_m, 0., 0., 0., 0.),
        (t.initial_jump_m, 0., 0., 0., 0.)) for t in seed.fault_traces.traces)
    exact = _sum_fields(seed_fields, exact)
    defect = 0.
    for trace, actual, integrated in zip(seed.fault_traces.traces, exact, fields):
        extent = _arc_frame(trace.start, trace.end)[3]
        for a, b in zip(actual[1:], integrated[1:]):
            extrema = trig_polynomial_extrema(TrigPolynomial(tuple(x-y for x,y in zip(a,b))), 0., extent)
            defect = max(defect, *(abs(v) for v in extrema))
    if defect > previous.max_anchor_transport_error_m:
        raise UnresolvedContactDynamics("Finite-pose anchor error exceeds its explicit physical bound")
    dt_myr = dt/SECONDS_PER_MYR
    velocities = {owner: tuple(map(float, row/dt_myr)) for owner, row in zip(ids, increments)}
    prior_poses = {owner: np.asarray(matrix) for owner, matrix in previous.owner_rotations}
    carriers = tuple(replace(trace,
        start=tuple(map(float, prior_poses[trace.overriding_plate]@trace.start)),
        end=tuple(map(float, prior_poses[trace.overriding_plate]@trace.end)))
        for trace in seed.fault_traces.traces)
    resolver, admission_audit = make_spatial_underthrust_resolver(before, velocities, dt_myr,
        initial_mechanics=seed, carrier_traces=carriers, step=spatial,
        history=polarity_evidence(previous.current_inventory))
    original_material_owners = {(f.parcel.material_id, f.parcel.plate) for f in origin.fragments}
    def retained_pair_resolver(a, b, overlap):
        if any((f.parcel.material_id, f.parcel.plate) not in original_material_owners for f in (a, b)):
            return PolarityResolution(0, "new_material_requires_new_physical_pair")
        return resolver(a, b, overlap)
    birth_parameters = previous.birth_parameters
    def birth(cell, owner, area, time, serial):
        identity = f"hot-birth:{previous.origin_surface_sha256}:{time.hex()}:{serial}"
        return SurfaceParcel(cell, owner, identity, area, area*birth_parameters.crust_thickness_km,
            0., 0., 0., (("damage", birth_parameters.damage), ("water", birth_parameters.water)))
    transaction = advance_geometric_surface(mesh, before, velocities, dt_myr,
        birth_factory=birth, polarity_resolver=retained_pair_resolver)
    if any(not loss.contact_ids or loss.attachment_status != "geometric_contact"
           for loss in transaction.losses):
        raise UnresolvedContactDynamics("New underthrust loss lacks an actual endpoint receiver contact")
    inventory = consume_geometric_transaction(before, transaction, previous.current_inventory,
        velocities, dt_myr)
    old_cohorts = {cohort.event_id: cohort for cohort in previous.current_inventory.cohorts}
    for cohort in inventory.cohorts:
        old = old_cohorts.get(cohort.event_id)
        if old is None and (not cohort.supports or cohort.unresolved_support
                            or cohort.attachment_status != "geometric_contact"):
            raise UnresolvedContactDynamics("New underthrust cohort lacks resolved receiver support")
        if old is not None and (cohort.loss != old.loss or cohort.payload_sha256 != old.payload_sha256
                or (not old.unresolved_support and cohort.unresolved_support)
                or (old.supports and not cohort.supports)):
            raise UnresolvedContactDynamics("Underthrust lost an existing immutable cohort or its receiver support")
    registry = advance_buried_bank_registry(origin, previous.buried_registry,
        previous.current_inventory, inventory, poses, roundoff_steps=len(previous.receipts)+1)
    support = validate_buried_bank_support(origin, seed, transaction.state, inventory, poses,
        registry, roundoff_steps=len(previous.receipts)+1)
    reactions = tuple(material_pair_point_reactions(port, area, step.traction_pa)
        for port, area, step in zip(reference.pair_ports, reference.areas, solved.steps))
    physical = np.zeros(len(zold))
    positions = {owner: index for index, owner in enumerate(ids)}
    trace_positions = {t.trace_id: index for index,t in enumerate(seed.fault_traces.traces)}
    for port, reaction in zip(reference.pair_ports, reactions):
        physical[3*positions[port.subducting_plate]:3*positions[port.subducting_plate]+3] += reaction["torque_on_sub_nm"]
        physical[3*positions[port.overriding_plate]:3*positions[port.overriding_plate]+3] += reaction["torque_on_over_nm"]
        index = rc+2*trace_positions[port.trace_id]
        physical[index:index+2] += radius*np.asarray(reaction["radial_force_n"])
    scale = math.fsum(float(np.linalg.norm(row)) for row in (physical, solved.contact_force))
    if np.linalg.norm(physical+solved.contact_force) > 512*_EPS*max(scale, np.finfo(float).tiny):
        raise UnresolvedContactDynamics("Objective material reactions disagree with generalized virtual work")
    metrics = dict(solved.diagnostics)
    radial_contact = physical[rc:]/radius
    radial_bending = -solved.elastic_force[rc:]/radius
    radial_residual = radial_contact+radial_bending
    radial_scale = max(np.linalg.norm(radial_contact),np.linalg.norm(radial_bending),np.finfo(float).tiny)
    radial_operands = np.linalg.norm(radial_bending)+math.fsum(
        np.linalg.norm(reaction["radial_force_n"]) for reaction in reactions)
    radial_rounding = 512*_EPS*radial_operands
    angular_scale = max(np.linalg.norm(torque),np.linalg.norm(solved.drag_force[:rc]),
                        np.linalg.norm(physical[:rc]),np.finfo(float).tiny)
    angular_operands = np.linalg.norm(torque)+np.linalg.norm(solved.drag_force[:rc])+math.fsum(
        math.sqrt(2.)*np.linalg.norm(reaction["torque_on_sub_nm"]) for reaction in reactions)
    if (np.linalg.norm(radial_residual) > 2e-10*radial_scale+radial_rounding
            or np.linalg.norm(solved.residual[:rc]) > 2e-10*angular_scale+512*_EPS*angular_operands):
        raise UnresolvedContactDynamics("Separate angular and physical transverse balances failed")
    metrics.update(anchor_transport_error_m=defect, relative_owner_angle_radians=float(relative_angle),
        material_reaction_relative_error=float(np.linalg.norm(physical+solved.contact_force))/max(scale,np.finfo(float).tiny),
        continuous_gap_max_m=max(high for _,_,high in _bounds(seed, fields)),
        transverse_force_relative_residual=float(np.linalg.norm(radial_residual)/radial_scale),
        transverse_force_rounding_tolerance_n=float(radial_rounding),
        angular_signal_relative_residual=float(np.linalg.norm(solved.residual[:rc])/angular_scale))
    elapsed = math.fsum((previous.elapsed_time_s, dt))
    if elapsed <= previous.elapsed_time_s:
        raise ValueError("Mechanical clock cannot resolve the underthrust interval")
    state = replace(previous, current_surface=transaction.state, current_inventory=inventory,
        owner_rotations=poses, states=states, transverse_displacement_m=_displacements(seed.fault_traces, radial),
        free_anchor_fields=fields, spatial_memory=spatial.memory, buried_registry=registry,
        elapsed_time_s=elapsed,
        cumulative_source_work_j=math.fsum((previous.cumulative_source_work_j,metrics["source_work_j"])),
        cumulative_basal_dissipation_j=math.fsum((previous.cumulative_basal_dissipation_j,metrics["basal_dissipation_j"])),
        cumulative_bending_numerical_dissipation_j=math.fsum((previous.cumulative_bending_numerical_dissipation_j,
            metrics["bending_numerical_dissipation_j"])))
    accounted = math.fsum((*(local.cumulative_external_work_j for _,local in state.states),
        state.cumulative_basal_dissipation_j, .5*float(solved.q@stiffness@solved.q),
        state.cumulative_bending_numerical_dissipation_j))
    supplied = math.fsum((state.preparation_work_j,state.cumulative_source_work_j))
    metrics["cumulative_energy_balance_relative_error"] = abs(accounted-supplied)/max(abs(accounted),abs(supplied),np.finfo(float).tiny)
    if metrics["cumulative_energy_balance_relative_error"] > 5e-10:
        raise UnresolvedContactDynamics("Retained contact, source and bending work do not balance")
    increment = tuple(tuple(map(float,row)) for row in increments)
    receipt = _seal(UnderthrustReceipt(len(previous.receipts)+1,load,dt,provenance,increment,
        _hash(_body(previous)),_hash(_body(state)),metrics["source_work_j"],metrics["basal_dissipation_j"],
        metrics["bending_numerical_dissipation_j"],""))
    state = replace(state,receipts=(*previous.receipts,receipt))
    return UnderthrustMotionStep(state,increment,tuple(tuple(map(float,row/dt)) for row in increments),
        reactions,transaction,spatial,support,admission_audit,tuple(sorted(metrics.items())))


def validate_underthrust_motion_state(mesh, origin, state):
    """Replay mechanics, all actual losses/births, buried support and work."""
    if (not isinstance(state,UnderthrustMotionState) or not isinstance(state.receipts,tuple)
            or state.origin_surface_sha256 != surface_digest(origin)):
        raise ValueError("Expected an original-bound underthrust state")
    expected = initialize_underthrust_motion(mesh,origin,state.initial_inventory,
        state.initial_mechanics,state.initial_load,birth_parameters=state.birth_parameters,
        max_anchor_transport_error_m=state.max_anchor_transport_error_m)
    for index, receipt in enumerate(state.receipts,1):
        if (not isinstance(receipt,UnderthrustReceipt) or type(receipt.sequence) is not int
                or receipt.sequence != index or receipt != _seal(receipt)):
            raise ValueError("Underthrust receipt identity or integrity mismatch")
        expected = _advance(mesh,origin,expected,receipt.current_load,receipt.dt_s,receipt.provenance).state
        if receipt != expected.receipts[-1]:
            raise ValueError("Underthrust receipt differs from complete physical replay")
    if _encode(asdict(state)) != _encode(asdict(expected)):
        raise ValueError("Underthrust state differs from its complete physical trajectory")
    return state


def advance_underthrust_motion(mesh,origin,previous,current_load,dt_s,*,provenance):
    validate_underthrust_motion_state(mesh,origin,previous)
    return _advance(mesh,origin,previous,current_load,dt_s,provenance)


def underthrust_motion_to_dict(mesh,origin,state):
    validate_underthrust_motion_state(mesh,origin,state)
    replay = initialize_underthrust_motion(mesh,origin,state.initial_inventory,state.initial_mechanics,
        state.initial_load,birth_parameters=state.birth_parameters,
        max_anchor_transport_error_m=state.max_anchor_transport_error_m)
    trajectory = []
    for receipt in state.receipts:
        trajectory.append(dict(current_load=basal_source_to_dict(replay.current_surface,receipt.current_load),
            dt_s=receipt.dt_s,provenance=receipt.provenance))
        replay = _advance(mesh,origin,replay,receipt.current_load,receipt.dt_s,receipt.provenance).state
    return json.loads(_encode(dict(format=FORMAT,model=MODEL,
        initial_inventory=boundary_to_dict(state.initial_inventory),
        initial_load=basal_source_to_dict(origin,state.initial_load),
        initial_mechanics=bending_contact_dynamics_to_dict(origin,state.initial_mechanics,state.initial_load),
        birth_parameters=asdict(state.birth_parameters),max_anchor_transport_error_m=state.max_anchor_transport_error_m,
        trajectory=trajectory,
        snapshot=asdict(state))))


def underthrust_motion_from_dict(raw,mesh,origin):
    keys = {"format","model","initial_inventory","initial_load","initial_mechanics",
            "birth_parameters","max_anchor_transport_error_m","trajectory","snapshot"}
    if not isinstance(raw,dict) or set(raw)!=keys or raw["format"]!=FORMAT or raw["model"]!=MODEL:
        raise ValueError("Expected explicit underthrust trajectory schema")
    load = basal_source_from_dict(raw["initial_load"],origin,mesh=mesh)
    seed = bending_contact_dynamics_from_dict(raw["initial_mechanics"],origin,load)
    state = initialize_underthrust_motion(mesh,origin,boundary_from_dict(raw["initial_inventory"]),seed,load,
        birth_parameters=HotBirthParameters(**raw["birth_parameters"]),
        max_anchor_transport_error_m=raw["max_anchor_transport_error_m"])
    if not isinstance(raw["trajectory"],list):
        raise ValueError("Underthrust trajectory must be an ordered JSON array")
    for row in raw["trajectory"]:
        if not isinstance(row,dict) or set(row)!={"current_load","dt_s","provenance"}:
            raise ValueError("Underthrust source stage schema mismatch")
        load = basal_source_from_dict(row["current_load"],state.current_surface,mesh=mesh)
        state = _advance(mesh,origin,state,load,row["dt_s"],row["provenance"]).state
    if _encode(raw["snapshot"]) != _encode(asdict(state)):
        raise ValueError("Underthrust saved snapshot differs from complete physical replay")
    return state
