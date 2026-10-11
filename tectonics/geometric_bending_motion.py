"""Atomic force-generated material motion on unchanged paired fault supports.

Sources and geometry are held at the beginning of each mechanical interval.
The complete computed Euler increment then transports the surface and passive
archive over that same interval. Only a common material rotation is accepted
here; differential motion needs the spatial admission/history law.
"""
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import math
from numbers import Integral

import numpy as np
from scipy.spatial.transform import Rotation

from .basal_coupling import PrescribedBasalParameters
from .geometric_basal_loading import GeometricBasalLoad
from .geometric_basal_sources import basal_source_from_dict, basal_source_to_dict, validate_basal_source
from .geometric_bending_reference import (
    current_bending_reference, validate_reference_rotation, _surface_from_dict,
)
from .geometric_boundary import (
    boundary_from_dict, boundary_to_dict, consume_geometric_transaction, surface_digest,
)
from .geometric_boundary_io import _validate_pair
from .geometric_contact_dynamics import _encode, _number, UnresolvedContactDynamics
from .geometric_segmented_bending_dynamics import (
    advance_segmented_bending_dynamics, initialize_segmented_bending_dynamics,
    segmented_bending_dynamics_from_dict, segmented_bending_dynamics_to_dict,
    validate_segmented_bending_dynamics_state,
)
from .geometric_surface import totals
from .geometric_transport import advance_geometric_surface
from .thermal import SECONDS_PER_MYR


FORMAT = "geometric-force-generated-common-bending-motion-1"
MODEL = "begin_stage_source_and_geometry_common_material_motion"


def _hash(value):
    return hashlib.sha256(_encode(value)).hexdigest()


def _inventory_hash(inventory):
    return _hash(boundary_to_dict(inventory))


def _rows(matrix):
    return tuple(tuple(map(float, row)) for row in matrix)


@dataclass(frozen=True, slots=True)
class BendingMotionReceipt:
    sequence: int
    source_segment_id: str
    dt_s: float
    rotation_increment_radians: tuple[float, float, float]
    before_surface_sha256: str
    after_surface_sha256: str
    before_inventory_sha256: str
    after_inventory_sha256: str
    common_rotation: tuple[tuple[float, float, float], ...]
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class BendingMotionState:
    origin_surface_sha256: str
    initial_inventory: object
    mechanics: object
    current_surface: object
    current_inventory: object
    common_rotation: tuple[tuple[float, float, float], ...]
    motion_receipts: tuple[BendingMotionReceipt, ...]


@dataclass(frozen=True, slots=True)
class BendingMotionStep:
    state: BendingMotionState
    mechanical_step: object
    geometric_transaction: object
    model: str = MODEL


def _receipt_body(receipt):
    raw = asdict(receipt)
    raw.pop("receipt_sha256")
    return dict(format=FORMAT, model=MODEL, **raw)


def _seal(receipt):
    return replace(receipt, receipt_sha256=_hash(_receipt_body(receipt)))


def _trusted_source(mesh, surface, load):
    if isinstance(load, GeometricBasalLoad):
        return basal_source_from_dict(basal_source_to_dict(surface, load), surface, mesh=mesh,
            parameters=PrescribedBasalParameters(**dict(load.source_parameters)),
            thermal_lid_by_fragment=dict(load.thermal_lid_by_fragment), provenance=load.provenance)
    return validate_basal_source(surface, load, mesh=mesh)


def initialize_bending_motion(origin_surface, inventory, initial_mechanics, initial_load):
    """Retain the actual starting surface/archive and original mechanical epoch."""
    _validate_pair(origin_surface, inventory)
    mechanics = initialize_segmented_bending_dynamics(origin_surface, initial_mechanics, initial_load)
    current_bending_reference(origin_surface, initial_mechanics)
    return BendingMotionState(surface_digest(origin_surface), inventory, mechanics,
        origin_surface, inventory, _rows(np.eye(3)), ())


def _forbidden_birth(*args, **kwargs):
    raise UnresolvedContactDynamics("Common paired motion cannot create new material")


def _advance(mesh, origin, previous, current_load, dt_s, provenance):
    dt = _number(dt_s, "Physical motion interval", positive=True)
    before = previous.current_surface
    _trusted_source(mesh, before, current_load)
    mechanical = advance_segmented_bending_dynamics(origin, previous.mechanics, current_load, dt,
        reference_rotation=previous.common_rotation, provenance=provenance,
        reference_elapsed_time_myr=before.time_myr-origin.time_myr,
        reference_surface=before, reference_roundoff_steps=len(previous.motion_receipts))
    increments = np.asarray(mechanical.rotation_increment_radians)
    common = increments[0]
    if any(not np.array_equal(common, vector) for vector in increments[1:]):
        raise UnresolvedContactDynamics(
            "Computed differential motion requires spatial material admission and contact-history matching")
    dt_myr = dt/SECONDS_PER_MYR
    velocities = {owner: tuple(map(float, common/dt_myr)) for owner in previous.mechanics.plate_ids}
    transaction = advance_geometric_surface(mesh, before, velocities, dt_myr,
                                           birth_factory=_forbidden_birth)
    if transaction.losses or transaction.births or transaction.state.phase != "partition":
        raise UnresolvedContactDynamics("Paired common motion changed material topology")
    if totals(before.fragments) != totals(transaction.state.fragments):
        raise UnresolvedContactDynamics("Common motion changed an extensive material reservoir")
    inventory = consume_geometric_transaction(before, transaction, previous.current_inventory,
                                              velocities, dt_myr)
    old_cohorts = {cohort.event_id: cohort for cohort in previous.current_inventory.cohorts}
    if set(old_cohorts) != {cohort.event_id for cohort in inventory.cohorts}:
        raise UnresolvedContactDynamics("Common motion changed passive cohort identities")
    for cohort in inventory.cohorts:
        old = old_cohorts[cohort.event_id]
        if (cohort.loss != old.loss or cohort.payload_sha256 != old.payload_sha256
                or (not old.unresolved_support and cohort.unresolved_support)
                or (old.supports and not cohort.supports)):
            raise UnresolvedContactDynamics("Common motion lost existing material or archive support")
    rotation = _rows(Rotation.from_rotvec(common).as_matrix()@np.asarray(previous.common_rotation))
    current_bending_reference(origin, mechanical.state.initial_mechanics, rotation,
        reference_elapsed_time_myr=transaction.state.time_myr-origin.time_myr,
        reference_surface=transaction.state, reference_roundoff_steps=len(previous.motion_receipts)+1)
    receipt = _seal(BendingMotionReceipt(len(previous.motion_receipts)+1,
        mechanical.state.segments[-1].receipt_id, dt, tuple(map(float, common)),
        surface_digest(before), surface_digest(transaction.state),
        _inventory_hash(previous.current_inventory), _inventory_hash(inventory), rotation, ""))
    state = replace(previous, mechanics=mechanical.state, current_surface=transaction.state,
        current_inventory=inventory, common_rotation=rotation,
        motion_receipts=(*previous.motion_receipts, receipt))
    return BendingMotionStep(state, mechanical, transaction)


def validate_bending_motion_state(mesh, origin_surface, state):
    """Replay forces, source work, genuine transport and archive transactions."""
    if (not isinstance(state, BendingMotionState)
            or state.origin_surface_sha256 != surface_digest(origin_surface)
            or not isinstance(state.motion_receipts, tuple)):
        raise ValueError("Expected an original-bound physical bending motion state")
    _validate_pair(origin_surface, state.initial_inventory)
    validate_reference_rotation(state.common_rotation,
                                reference_roundoff_steps=len(state.motion_receipts))
    _trusted_source(mesh, origin_surface, state.mechanics.initial_load)
    validate_segmented_bending_dynamics_state(origin_surface, state.mechanics)
    if len(state.motion_receipts) != len(state.mechanics.segments):
        raise ValueError("Physical motion and source-segment counts disagree")
    expected = initialize_bending_motion(origin_surface, state.initial_inventory,
        state.mechanics.initial_mechanics, state.mechanics.initial_load)
    for index, (receipt, segment) in enumerate(zip(state.motion_receipts, state.mechanics.segments), 1):
        if (not isinstance(receipt, BendingMotionReceipt)
                or isinstance(receipt.sequence, (bool, np.bool_))
                or not isinstance(receipt.sequence, Integral) or receipt.sequence != index
                or receipt != _seal(receipt)):
            raise ValueError("Physical motion receipt identity or integrity mismatch")
        _number(receipt.dt_s, "Physical motion receipt interval", positive=True)
        if (not isinstance(receipt.rotation_increment_radians, tuple)
                or len(receipt.rotation_increment_radians) != 3):
            raise ValueError("Physical motion receipt requires an immutable angular triple")
        for component in receipt.rotation_increment_radians:
            _number(component, "Physical motion receipt angular increment")
        validate_reference_rotation(receipt.common_rotation, reference_roundoff_steps=index)
        reproduced = _advance(mesh, origin_surface, expected, segment.current_load,
                              segment.dt_s, segment.provenance)
        if receipt != reproduced.state.motion_receipts[-1]:
            raise ValueError("Physical motion receipt differs from force/geometry/archive replay")
        expected = reproduced.state
    _validate_pair(state.current_surface, state.current_inventory)
    if state != expected:
        raise ValueError("Physical motion state differs from its complete accepted trajectory")
    return state


def advance_bending_motion(mesh, origin_surface, previous, current_load, dt_s, *, provenance):
    """Commit one complete force-generated common step, or preserve all inputs."""
    validate_bending_motion_state(mesh, origin_surface, previous)
    return _advance(mesh, origin_surface, previous, current_load, dt_s, provenance)


def bending_motion_to_dict(mesh, origin_surface, state):
    validate_bending_motion_state(mesh, origin_surface, state)
    return json.loads(_encode(dict(format=FORMAT, model=MODEL,
        origin_surface_sha256=state.origin_surface_sha256,
        initial_inventory=boundary_to_dict(state.initial_inventory),
        mechanics=segmented_bending_dynamics_to_dict(origin_surface, state.mechanics),
        current_surface=asdict(state.current_surface), current_inventory=boundary_to_dict(state.current_inventory),
        common_rotation=state.common_rotation, motion_receipts=[asdict(r) for r in state.motion_receipts])))


def bending_motion_from_dict(raw, mesh, origin_surface):
    expected = {field.name for field in fields(BendingMotionState)}|{"format", "model"}
    if (not isinstance(raw, dict) or set(raw) != expected
            or raw["format"] != FORMAT or raw["model"] != MODEL):
        raise ValueError("Invalid accepted common bending motion schema")
    try:
        if not isinstance(raw["motion_receipts"], list):
            raise ValueError("Physical motion receipts require an explicit JSON list")
        receipts = []
        names = {field.name for field in fields(BendingMotionReceipt)}
        for row in raw["motion_receipts"]:
            if not isinstance(row, dict) or set(row) != names:
                raise ValueError("Invalid physical motion receipt schema")
            row = dict(row)
            row["rotation_increment_radians"] = tuple(row["rotation_increment_radians"])
            row["common_rotation"] = tuple(tuple(v) for v in row["common_rotation"])
            receipts.append(BendingMotionReceipt(**row))
        state = BendingMotionState(raw["origin_surface_sha256"],
            boundary_from_dict(raw["initial_inventory"]),
            segmented_bending_dynamics_from_dict(raw["mechanics"], origin_surface),
            _surface_from_dict(raw["current_surface"]), boundary_from_dict(raw["current_inventory"]),
            tuple(tuple(v) for v in raw["common_rotation"]), tuple(receipts))
        return validate_bending_motion_state(mesh, origin_surface, state)
    except (TypeError, KeyError, IndexError, OverflowError) as error:
        raise ValueError("Malformed accepted physical motion checkpoint") from error
