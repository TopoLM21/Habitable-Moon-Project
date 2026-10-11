"""Loaded original material pairs with authoritative native surface/archives.

The existing objective force engine, physical columns, beam energies and
continuous memory are retained. Native geometry alone decides material cuts,
live bank support and receiver attachment. A loaded checkpoint replays both.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math

import numpy as np
from scipy.spatial.transform import Rotation

from .fractional_surface import SurfaceParcel
from .geometric_basal_sources import basal_source_from_dict, basal_source_to_dict
from .geometric_bending_contact_dynamics import (
    _displacements, _transverse_coordinates, bending_contact_dynamics_from_dict,
    bending_contact_dynamics_to_dict, validate_bending_contact_dynamics_state,
)
from .geometric_bending_motion import _trusted_source
from .geometric_boundary import boundary_from_dict, boundary_to_dict, initialize_boundary, surface_digest
from .geometric_boundary_io import _validate_pair
from .geometric_contact_dynamics import _encode, _equilibrate_contact_network, _number, UnresolvedContactDynamics
from .geometric_fault_traces import _arc_frame
from .geometric_intrinsic_geometry import RotationIncrement, rotation_increment_to_dict
from .geometric_material_pair_reference import (
    MaterialPairReference, _material_pair_network, canonical_material_pair_poses,
    material_pair_exact_anchor_fields, material_pair_increment_fields,
    material_pair_point_reactions,
)
from .geometric_native_buried_support import (
    initial_native_buried_registry, advance_native_buried_registry,
    validate_native_bending_bank_support, native_buried_registry_to_dict,
)
from .geometric_native_mantle_loading import (
    NativePhysicalSpatialMantleLoad, native_surface_digest,
    validate_native_physical_spatial_mantle_load,
    native_spatial_mantle_load_to_dict, native_spatial_mantle_load_from_dict,
)
from .geometric_native_pose import certify_rotation_history
from .geometric_native_spatial_underthrust import make_native_spatial_underthrust_resolver
from .geometric_native_surface import advance_native_surface, enroll_native_surface, native_surface_to_dict
from .geometric_polarity import PolarityResolution
from .geometric_spatial_underthrust import (
    TrigPolynomial, advance_spatial_underthrust_memory,
    initialize_spatial_underthrust_memory, trig_polynomial_extrema,
)
from .geometric_underthrust_motion import (
    HotBirthParameters, _birth_parameters, _bounds, _domain, _sum_fields, _text,
)
from .geometric_transport import UnresolvedPolarityError
from .thermal import SECONDS_PER_MYR

FORMAT = "geometric-native-persistent-pair-underthrust-motion-1"
MODEL = "objective_original_material_pairs_with_certified_native_support"
_EPS = np.finfo(float).eps


def _hash(value):
    return hashlib.sha256(_encode(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class NativeUnderthrustReceipt:
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
class NativeUnderthrustCandidate:
    """Unaccepted force evidence; it never advances the persistent ledger."""
    before_surface_sha256: str
    current_load_sha256: str
    dt_s: float
    provenance: str
    rotation_increment_radians: tuple
    angular_velocity_rad_s: tuple
    omega_rad_per_myr: tuple
    states: tuple
    transverse_displacement_m: tuple
    spatial_step: object
    candidate_diagnostics: tuple
    admission_audit_json: bytes
    points_json: bytes

    @property
    def admission_audit(self):
        return json.loads(self.admission_audit_json)

    @property
    def points(self):
        return json.loads(self.points_json)


class UnresolvedNativeUnderthrustMotion(UnresolvedPolarityError):
    """A physically solved candidate failed exact spatial admission."""
    def __init__(self, overlaps, *, candidate):
        if type(candidate) is not NativeUnderthrustCandidate:
            raise TypeError("Native refusal requires explicit unaccepted force evidence")
        self.candidate = candidate
        super().__init__(overlaps)


@dataclass(frozen=True, slots=True)
class NativeUnderthrustMotionState:
    origin_surface_sha256: str
    origin_native_sha256: str
    initial_inventory: object
    initial_mechanics: object
    initial_load: object
    birth_parameters: HotBirthParameters
    max_anchor_transport_error_m: float
    current_surface: object
    current_inventory: object
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

    @property
    def plate_ids(self):
        return self.initial_mechanics.plate_ids

    @property
    def preparation_work_j(self):
        return self.initial_mechanics.preparation_work_j

    @property
    def owner_rotations(self):
        if not self.owner_histories:
            raise ValueError("Native motion requires complete owner histories")
        return _poses(self.initial_mechanics,self.owner_histories,len(self.owner_histories[0][1]))

    @property
    def buried_registry(self):
        return self.current_inventory


@dataclass(frozen=True, slots=True)
class NativeUnderthrustMotionStep:
    state: NativeUnderthrustMotionState
    rotation_increment_radians: tuple
    angular_velocity_rad_s: tuple
    points: tuple
    geometric_transaction: object
    spatial_step: object
    support_receipt: object
    admission_audit: object
    diagnostics: tuple
    model: str = MODEL


def _poses(seed,histories,steps):
    if (not isinstance(histories,tuple) or tuple(owner for owner,_ in histories)!=seed.plate_ids
            or any(not isinstance(history,tuple) or len(history)!=steps for _,history in histories)):
        raise ValueError("Native owner histories must retain every complete physical interval")
    matrices = tuple((owner,tuple(tuple(float(x) for x in row)
        for row in certify_rotation_history(history,128).matrix)) for owner,history in histories)
    return canonical_material_pair_poses(seed,matrices,roundoff_steps=steps)


def _body(state):
    return dict(origin_surface_sha256=state.origin_surface_sha256,
        origin_native_sha256=state.origin_native_sha256,
        birth_parameters=asdict(state.birth_parameters),
        max_anchor_transport_error_m=state.max_anchor_transport_error_m,
        current_surface=native_surface_to_dict(state.current_surface),
        current_inventory=native_buried_registry_to_dict(state.current_inventory),
        owner_histories=[dict(owner=owner,history=[rotation_increment_to_dict(s) for s in history])
                         for owner,history in state.owner_histories],
        owner_rotations=state.owner_rotations,
        states=[(identity,asdict(local)) for identity,local in state.states],
        transverse_displacement_m=state.transverse_displacement_m,
        free_anchor_fields=state.free_anchor_fields,spatial_memory=asdict(state.spatial_memory),
        elapsed_time_s=state.elapsed_time_s,cumulative_source_work_j=state.cumulative_source_work_j,
        cumulative_basal_dissipation_j=state.cumulative_basal_dissipation_j,
        cumulative_bending_numerical_dissipation_j=state.cumulative_bending_numerical_dissipation_j)


def _seal(receipt):
    raw = asdict(receipt)
    raw.pop("receipt_sha256")
    return replace(receipt,receipt_sha256=_hash(raw))


def _native_reference(origin,seed,surface,histories,radial,registry,steps):
    native_origin = enroll_native_surface(origin)
    proof = validate_native_bending_bank_support(native_origin,seed,surface,registry,
        histories,legacy_origin=origin)
    poses = _poses(seed,histories,steps)
    original,records,points,ports,rows,operators = _material_pair_network(origin,seed,poses,
        radial,roundoff_steps=steps)
    return MaterialPairReference(surface,points,original.materials,rows,original.areas,
        original.frames,operators,original.stiffness,original.reference_points,original.offsets,
        original.plate_ids,records,tuple(radial),ports,original.reference_surface_sha256,
        original.reference_trace_sha256,original.reference_bending_sha256,
        native_surface_digest(surface),proof,steps)


def initialize_native_underthrust_motion(mesh,origin,inventory,initial_mechanics,initial_load,*,
                                       birth_parameters,max_anchor_transport_error_m):
    """Retain the prepared original epoch and enroll its exact native domains."""
    if _encode(boundary_to_dict(inventory)) != _encode(boundary_to_dict(initialize_boundary(origin))):
        raise ValueError("Native preparation requires the complete typed original empty boundary inventory")
    _validate_pair(origin,inventory)
    if inventory.cohorts:
        raise ValueError("Existing buried cohorts require replay of their native support lineage")
    _trusted_source(mesh,origin,initial_load)
    validate_bending_contact_dynamics_state(origin,initial_mechanics,initial_load)
    seed = initial_mechanics
    if any(v != 0. for row in seed.rotation_radians for v in row) or any(
            v != 0. for row in seed.transverse_displacement_m for v in row[1:]):
        raise ValueError("Native persistent pairs require the original unadvanced preparation")
    native = enroll_native_surface(origin)
    registry = initial_native_buried_registry(native)
    histories = tuple((owner,()) for owner in seed.plate_ids)
    state = NativeUnderthrustMotionState(surface_digest(origin),native_surface_digest(native),
        inventory,seed,initial_load,_birth_parameters(birth_parameters),
        _number(max_anchor_transport_error_m,"Finite pose anchor error bound",positive=True),
        native,registry,histories,seed.states,seed.transverse_displacement_m,
        tuple((t.trace_id,(t.initial_gap_m,0.,0.,0.,0.),(t.initial_jump_m,0.,0.,0.,0.))
              for t in seed.fault_traces.traces),initialize_spatial_underthrust_memory(seed),
        seed.elapsed_time_s,seed.cumulative_source_work_j,seed.cumulative_basal_dissipation_j,
        seed.cumulative_bending_numerical_dissipation_j,())
    _native_reference(origin,seed,native,histories,state.transverse_displacement_m,registry,0)
    return state


def _advance_native(mesh, origin, previous, load, dt_s, provenance):
    dt, provenance = _number(dt_s, "Physical underthrust interval", positive=True), _text(provenance)
    before, seed = previous.current_surface, previous.initial_mechanics
    validate_native_physical_spatial_mantle_load(before,load,mesh=mesh)
    reference = _native_reference(origin,seed,before,previous.owner_histories,
        previous.transverse_displacement_m,previous.current_inventory,len(previous.receipts))
    ids, drag, torque = load.plate_ids,np.asarray(load.drag_nm_s),np.asarray(load.torque_nm)
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
    dt_myr = dt/SECONDS_PER_MYR
    velocities = {owner: tuple(map(float,row/dt_myr)) for owner,row in zip(ids,increments)}
    histories = tuple((owner,history+(RotationIncrement(velocities[owner],dt_myr),))
                      for owner,history in previous.owner_histories)
    poses = _poses(seed,histories,len(previous.receipts)+1)
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
    # Check the genuine force candidate before geometry can refuse it.  The
    # preview sums remain local and are never added to persistent work/memory.
    accounted_preview = math.fsum((*(local.cumulative_external_work_j for _,local in states),
        math.fsum((previous.cumulative_basal_dissipation_j,metrics["basal_dissipation_j"])),
        .5*float(solved.q@stiffness@solved.q),
        math.fsum((previous.cumulative_bending_numerical_dissipation_j,
                   metrics["bending_numerical_dissipation_j"]))))
    supplied_preview = math.fsum((previous.preparation_work_j,
        math.fsum((previous.cumulative_source_work_j,metrics["source_work_j"]))))
    metrics["cumulative_energy_balance_relative_error"] = abs(accounted_preview-supplied_preview)/max(
        abs(accounted_preview),abs(supplied_preview),np.finfo(float).tiny)
    if metrics["cumulative_energy_balance_relative_error"] > 5e-10:
        raise UnresolvedContactDynamics("Candidate contact, source and bending work do not balance")
    dt_myr = dt/SECONDS_PER_MYR
    velocities = {owner: tuple(map(float, row/dt_myr)) for owner, row in zip(ids, increments)}
    resolver, admission_audit = make_native_spatial_underthrust_resolver(before,velocities,dt_myr,
        origin_surface=origin,initial_mechanics=seed,owner_histories=previous.owner_histories,
        step=spatial)
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
    try:
        transaction = advance_native_surface(before,velocities,dt_myr,
            birth_factory=birth,polarity_resolver=retained_pair_resolver)
    except UnresolvedPolarityError as error:
        candidate = NativeUnderthrustCandidate(native_surface_digest(before),load.load_sha256,
            dt,provenance,tuple(tuple(map(float,row)) for row in increments),
            tuple(tuple(map(float,row/dt)) for row in increments),
            tuple((owner,velocities[owner]) for owner in ids),states,
            _displacements(seed.fault_traces,radial),spatial,tuple(sorted(metrics.items())),
            _encode(admission_audit),_encode(reactions))
        raise UnresolvedNativeUnderthrustMotion(error.overlaps,candidate=candidate) from error
    from .geometric_native_spatial_underthrust import attach_native_underthrust_losses
    transaction, attachment_audit = attach_native_underthrust_losses(before,transaction,
        origin_surface=origin,initial_mechanics=seed,owner_histories_after=histories,
        step=spatial,admission_audit=admission_audit)
    if any(not loss.contact_ids or loss.attachment_status != "geometric_contact"
           for loss in transaction.losses):
        raise UnresolvedContactDynamics("Native loss lacks a certified original receiver attachment")
    inventory = advance_native_buried_registry(enroll_native_surface(origin),previous.current_inventory,
        before,transaction,histories)
    support = validate_native_bending_bank_support(enroll_native_surface(origin),seed,
        transaction.state,inventory,histories,legacy_origin=origin)
    admission_audit = dict(admission_audit,endpoint_attachments=attachment_audit)
    elapsed = math.fsum((previous.elapsed_time_s, dt))
    if elapsed <= previous.elapsed_time_s:
        raise ValueError("Mechanical clock cannot resolve the underthrust interval")
    state = replace(previous, current_surface=transaction.state, current_inventory=inventory,
        owner_histories=histories, states=states, transverse_displacement_m=_displacements(seed.fault_traces, radial),
        free_anchor_fields=fields, spatial_memory=spatial.memory,
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
    receipt = _seal(NativeUnderthrustReceipt(len(previous.receipts)+1,load,dt,provenance,increment,
        _hash(_body(previous)),_hash(_body(state)),metrics["source_work_j"],metrics["basal_dissipation_j"],
        metrics["bending_numerical_dissipation_j"],""))
    state = replace(state,receipts=(*previous.receipts,receipt))
    return NativeUnderthrustMotionStep(state,increment,tuple(tuple(map(float,row/dt)) for row in increments),
        reactions,transaction,spatial,support,admission_audit,tuple(sorted(metrics.items())))


def _snapshot(state):
    return dict(**_body(state),initial_inventory=boundary_to_dict(state.initial_inventory),
        initial_mechanics=asdict(state.initial_mechanics),initial_load=asdict(state.initial_load),
        receipts=[asdict(receipt) for receipt in state.receipts])


def _replay(mesh,origin,state,*,trajectory=False):
    if (type(state) is not NativeUnderthrustMotionState or not isinstance(state.receipts,tuple)
            or state.origin_surface_sha256 != surface_digest(origin)):
        raise ValueError("Expected an original-bound native loaded state")
    _poses(state.initial_mechanics,state.owner_histories,len(state.receipts))
    expected = initialize_native_underthrust_motion(mesh,origin,state.initial_inventory,
        state.initial_mechanics,state.initial_load,birth_parameters=state.birth_parameters,
        max_anchor_transport_error_m=state.max_anchor_transport_error_m)
    rows = []
    for index,receipt in enumerate(state.receipts,1):
        if (type(receipt) is not NativeUnderthrustReceipt or type(receipt.sequence) is not int
                or receipt.sequence != index or receipt != _seal(receipt)):
            raise ValueError("Native loaded receipt identity or integrity mismatch")
        if trajectory:
            rows.append(dict(current_load=native_spatial_mantle_load_to_dict(
                expected.current_surface,receipt.current_load),dt_s=receipt.dt_s,
                provenance=receipt.provenance))
        expected = _advance_native(mesh,origin,expected,receipt.current_load,
            receipt.dt_s,receipt.provenance).state
        if _encode(asdict(receipt)) != _encode(asdict(expected.receipts[-1])):
            raise ValueError("Native loaded receipt differs from complete physical replay")
    if _encode(_snapshot(state)) != _encode(_snapshot(expected)):
        raise ValueError("Native loaded state differs from complete physical replay")
    return expected,rows


def validate_native_underthrust_motion_state(mesh,origin,state):
    _replay(mesh,origin,state)
    return state


def advance_native_underthrust_motion(mesh,origin,previous,current_load,dt_s,*,provenance):
    validate_native_underthrust_motion_state(mesh,origin,previous)
    return _advance_native(mesh,origin,previous,current_load,dt_s,provenance)


def native_underthrust_motion_to_dict(mesh,origin,state):
    expected,trajectory = _replay(mesh,origin,state,trajectory=True)
    return json.loads(_encode(dict(format=FORMAT,model=MODEL,
        initial_inventory=boundary_to_dict(expected.initial_inventory),
        initial_load=basal_source_to_dict(origin,expected.initial_load),
        initial_mechanics=bending_contact_dynamics_to_dict(origin,expected.initial_mechanics,
            expected.initial_load),birth_parameters=asdict(expected.birth_parameters),
        max_anchor_transport_error_m=expected.max_anchor_transport_error_m,
        trajectory=trajectory,snapshot=_snapshot(expected))))


def native_underthrust_motion_from_dict(raw,mesh,origin):
    keys = {"format","model","initial_inventory","initial_load","initial_mechanics",
            "birth_parameters","max_anchor_transport_error_m","trajectory","snapshot"}
    if (not isinstance(raw,dict) or set(raw)!=keys or raw["format"]!=FORMAT or raw["model"]!=MODEL
            or not isinstance(raw["trajectory"],list) or not isinstance(raw["snapshot"],dict)
            or not isinstance(raw["birth_parameters"],dict)
            or set(raw["birth_parameters"])!=set(HotBirthParameters.__dataclass_fields__)):
        raise ValueError("Expected explicit native loaded trajectory schema")
    load = basal_source_from_dict(raw["initial_load"],origin,mesh=mesh)
    seed = bending_contact_dynamics_from_dict(raw["initial_mechanics"],origin,load)
    inventory = boundary_from_dict(raw["initial_inventory"])
    # Existing parsers may canonicalize Python numeric aliases. Authenticate
    # their actual complete JSON types before retaining the preparation.
    if (_encode(raw["initial_inventory"]) != _encode(boundary_to_dict(inventory))
            or _encode(raw["initial_load"]) != _encode(basal_source_to_dict(origin,load))
            or _encode(raw["initial_mechanics"]) != _encode(
                bending_contact_dynamics_to_dict(origin,seed,load))):
        raise ValueError("Native loaded preparation changed explicit JSON types")
    state = initialize_native_underthrust_motion(mesh,origin,inventory,seed,load,
        birth_parameters=HotBirthParameters(**raw["birth_parameters"]),
        max_anchor_transport_error_m=raw["max_anchor_transport_error_m"])
    for row in raw["trajectory"]:
        if not isinstance(row,dict) or set(row)!={"current_load","dt_s","provenance"}:
            raise ValueError("Native physical source stage schema mismatch")
        load = native_spatial_mantle_load_from_dict(row["current_load"],state.current_surface,mesh=mesh)
        state = _advance_native(mesh,origin,state,load,row["dt_s"],row["provenance"]).state
    if _encode(raw["snapshot"]) != _encode(_snapshot(state)):
        raise ValueError("Native joint snapshot differs from force/material/support replay")
    return state
