"""Frozen-surface elastic contact experiments on declared dipping planes.

Boundary displacements are local midpoint samples, parallel transported with
the fault frame along each arc. They are prescribed clamps, not solved Euler
velocities. Contact shortening and plastic slip stay in mechanical history;
this module never removes surface material or creates a slab cohort.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import json
import math
from numbers import Real

import numpy as np

from .genesis_contact_law import ContactLawParameters
from .geometric_boundary import surface_digest
from .geometric_contacts import extract_contacts
from .geometric_contact_mechanics import (
    ElasticColumnParameters, LocalContactState, LocalContactStep,
    solve_compressed_contact, state_from_dict, validate_contact_state,
)
from .geometric_initiation import OrientedFault
from .geometric_initiation_adapter import fault_snapshot_to_dict


FORMAT = "geometric-local-compressed-contact-1"
FRAME = "over_minus_sub_gap_and_jump_at_midpoint_parallel_transport"
EXTENSION = "same_plane_traction_only_zero_down_dip_and_strike_normal_stress"
EVOLUTION = "fixed_surface_prescribed_elastic_column_anchors_no_material_transfer"


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class ContactExperiment:
    """Declared local strip; height is radial depth in a locally flat section.

    The current closed skeleton-contact model supports explicitly zero pore
    pressure. Water selects the existing material friction/cohesion law only.
    """
    fault: OrientedFault
    columns: ElasticColumnParameters
    law: ContactLawParameters
    contact_height_m: float
    water: float
    pore_pressure_pa: float
    provenance: str

    def __post_init__(self):
        if not isinstance(self.fault, OrientedFault) or self.fault.effective_stress_pa is not None:
            raise ValueError("Declare fault geometry without a prescribed stress; the contact solves it")
        if not isinstance(self.columns, ElasticColumnParameters):
            raise ValueError("Explicit paired elastic columns are required")
        if not isinstance(self.law, ContactLawParameters):
            raise ValueError("An explicit existing contact law is required")
        self.law.validate()
        for name in ("contact_height_m", "water", "pore_pressure_pa"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"Contact {name} must be finite and real")
            object.__setattr__(self, name, float(value))
        if self.contact_height_m <= 0. or not 0. <= self.water <= 1.:
            raise ValueError("Contact height must be positive and water in [0,1]")
        if self.pore_pressure_pa != 0.:
            raise ValueError("This paired-column experiment supports explicitly zero pore pressure only")
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("Contact boundary conditions require explicit provenance")
        cohesion = self.law.cohesion_pa*(1-(1-self.law.wet_cohesion_fraction)*self.water)
        friction = self.law.friction_dry+(self.law.friction_wet-self.law.friction_dry)*self.water
        if self.fault.cohesion_pa != cohesion or self.fault.friction != friction:
            raise ValueError("Fault admission strength must match the actual wet contact law")


@dataclass(frozen=True)
class ContactMechanicsSnapshot:
    surface_sha256: str
    experiments: tuple[ContactExperiment, ...]
    states: tuple[tuple[str, LocalContactState], ...]


@dataclass(frozen=True)
class ContactMechanicsResult:
    surface_sha256: str
    fault_id: str
    contact_id: str
    step: LocalContactStep
    solved_fault: OrientedFault
    force_on_sub_n: tuple[float, float, float]
    force_on_over_n: tuple[float, float, float]
    torque_on_sub_nm: tuple[float, float, float]
    torque_on_over_nm: tuple[float, float, float]
    down_dip_plastic_increment_m: float


def _experiments(surface, experiments):
    if surface.phase != "partition":
        raise ValueError("Local mechanics requires a resolved fixed surface")
    values = tuple(experiments)
    if any(not isinstance(item, ContactExperiment) for item in values):
        raise ValueError("Explicit ContactExperiment records are required")
    if len({item.fault.fault_id for item in values}) != len(values):
        raise ValueError("Contact experiment fault IDs must be unique")
    values = tuple(sorted(values, key=lambda item: item.fault.fault_id))
    # Reuse the existing geometry/snapshot validator; no plate-pair shortcut.
    fault_snapshot_to_dict(surface, tuple(item.fault for item in values))
    return values


def _frame(contact, fault):
    radial = np.asarray(contact.midpoint)
    horizontal = np.asarray(contact.normal_a_to_b)
    if fault.overriding_plate != contact.plate_b:
        horizontal = -horizontal
    sine, cosine = math.sin(fault.dip_radians), math.cos(fault.dip_radians)
    return sine*horizontal+cosine*radial, cosine*horizontal-sine*radial


def _clocks(states):
    if len({(state.step_count, state.elapsed_time_s) for _, state in states}) > 1:
        raise ValueError("Contact branches in one snapshot must share the same mechanical clock")


def _integrated_reaction(contact, fault, height_m, traction_pa):
    """Integrate parallel-transported traction using existing exact arc moments.

    The reduced strip is referenced to the surface radius. Variation of lever
    arm through its small radial depth is outside this thin-strip model.
    """
    sign = 1. if fault.overriding_plate == contact.plate_b else -1.
    horizontal = sign*np.asarray(contact.normal_a_to_b)
    sine, cosine = math.sin(fault.dip_radians), math.cos(fault.dip_radians)
    tn, tt = traction_pa
    width = height_m/sine
    along_h, along_r = tn*sine+tt*cosine, tn*cosine-tt*sine
    force = width*(along_h*contact.length_km*1000.*horizontal
                   +along_r*np.asarray(contact.integrated_position_unit_km)*1000.)
    torque = width*along_h*sign*np.asarray(contact.moment_arm_cross_normal_km2)*1e6
    if not np.isfinite(force).all() or not np.isfinite(torque).all():
        raise ValueError("Integrated contact reaction overflowed")
    return force, torque


def solve_contact_mechanics(surface, experiments, bank_anchors_m, dt_s, *, previous=None):
    """Solve independent branches without selecting among alternative faults.

    ``bank_anchors_m[plate,3]`` gives each bank's prescribed displacement at
    each tested midpoint. Strain history is referenced to the original
    unstressed anchors, not to the preceding solution. Opposite planes are
    separate alternative boundary-value problems; their reactions must not
    be added as if both branches were simultaneously realized.
    """
    if (isinstance(dt_s, (bool, np.bool_)) or not isinstance(dt_s, Real)
            or not math.isfinite(dt_s) or dt_s <= 0.):
        raise ValueError("Local contact dt_s must be finite and positive")
    experiments = _experiments(surface, experiments)
    digest = surface_digest(surface)
    contacts = {item.contact_id: item for item in extract_contacts(surface.fragments, surface.radius_km)}
    raw = np.asarray(bank_anchors_m)
    if (raw.ndim != 2 or raw.shape[1] != 3 or raw.dtype.kind not in "iuf"
            or not np.isfinite(raw).all()):
        raise ValueError("Bank anchors must be finite real displacements [plate,3] in metres")
    if any(max(item.fault.subducting_plate, item.fault.overriding_plate) >= len(raw)
           for item in experiments):
        raise ValueError("Bank anchors must include both declared fault owners")
    if any(isinstance(value, (bool, np.bool_)) for row in bank_anchors_m for value in row):
        raise ValueError("Bank displacement components cannot be bools")
    anchors = np.asarray(raw, dtype=float)
    old = {}
    if previous is not None:
        if not isinstance(previous, ContactMechanicsSnapshot):
            raise ValueError("Restart requires a bound ContactMechanicsSnapshot")
        if previous.surface_sha256 != digest or previous.experiments != experiments:
            raise ValueError("Mechanical restart surface or boundary model changed")
        old = dict(previous.states)
        if len(old) != len(previous.states) or set(old) != {item.fault.fault_id for item in experiments}:
            raise ValueError("Mechanical restart must contain every declared branch exactly once")
        _clocks(previous.states)
    states, results = [], []
    for experiment in experiments:
        fault = experiment.fault
        contact = contacts[fault.contact_id]
        normal, down = _frame(contact, fault)
        relative = anchors[fault.overriding_plate]-anchors[fault.subducting_plate]
        gap_free, jump_free = float(normal@relative), float(down@relative)
        area = contact.length_km*1000.*experiment.contact_height_m/math.sin(fault.dip_radians)
        step = solve_compressed_contact(columns=experiment.columns, law=experiment.law,
            area_m2=area, gap_free_m=gap_free, jump_free_m=jump_free,
            dt_s=dt_s, water=experiment.water, previous=old.get(fault.fault_id))
        normal_traction, shear = step.traction_pa
        # Traction fixes Sigma n only. The other normal stresses are explicitly
        # zero; never use this extension to assess an unrelated/conjugate plane.
        tensor = (-normal_traction*np.outer(normal, normal)
                  -shear*(np.outer(down, normal)+np.outer(normal, down)))
        solved = replace(fault, effective_stress_pa=tensor,
            provenance=f"solved_local_elastic_contact:{experiment.provenance}:{fault.provenance}")
        force, torque = _integrated_reaction(contact, fault, experiment.contact_height_m, step.traction_pa)
        states.append((fault.fault_id, step.state))
        results.append(ContactMechanicsResult(digest, fault.fault_id, fault.contact_id, step, solved,
            tuple(map(float, force)), tuple(map(float, -force)),
            tuple(map(float, torque)), tuple(map(float, -torque)), -step.slip_increment_m))
    return ContactMechanicsSnapshot(digest, experiments, tuple(states)), tuple(results)


def mechanics_to_dict(surface, snapshot):
    experiments = _experiments(surface, snapshot.experiments)
    if snapshot.surface_sha256 != surface_digest(surface) or experiments != snapshot.experiments:
        raise ValueError("Mechanical snapshot belongs to another surface or experiment ordering")
    states = dict(snapshot.states)
    if len(states) != len(snapshot.states) or set(states) != {item.fault.fault_id for item in experiments}:
        raise ValueError("Mechanical snapshot branch history is incomplete or duplicated")
    _clocks(snapshot.states)
    payload = dict(format=FORMAT, surface_sha256=snapshot.surface_sha256,
        time_myr=surface.time_myr, radius_km=surface.radius_km,
        coordinate_convention=FRAME, stress_extension=EXTENSION, mechanics_evolution=EVOLUTION,
        experiments=[asdict(item) for item in experiments],
        states={key: asdict(value) for key, value in sorted(states.items())})
    # Exercise strict restoration before saving metadata to an existing joint
    # geometry/slab checkpoint. We do not create another material IO format.
    mechanics_from_dict(json.loads(_encoded(payload)), surface)
    return payload


def mechanics_from_dict(raw, surface):
    expected = dict(format=FORMAT, surface_sha256=surface_digest(surface),
        time_myr=surface.time_myr, radius_km=surface.radius_km,
        coordinate_convention=FRAME, stress_extension=EXTENSION, mechanics_evolution=EVOLUTION)
    if not isinstance(raw, dict) or set(raw) != {*expected, "experiments", "states"}:
        raise ValueError("Invalid local contact checkpoint schema")
    if any(isinstance(raw[key], bool) or raw[key] != value for key, value in expected.items()):
        raise ValueError("Local contact checkpoint is stale or uses another mechanical convention")
    if not isinstance(raw["experiments"], list) or not isinstance(raw["states"], dict):
        raise ValueError("Invalid contact experiments or branch histories")
    experiments = []
    try:
        for item in raw["experiments"]:
            values = dict(item)
            if not isinstance(values["law"], dict) or set(values["law"]) != {field.name for field in fields(ContactLawParameters)}:
                raise ValueError("Stored contact law must explicitly name every physical parameter")
            values["fault"] = OrientedFault(**values["fault"])
            values["columns"] = ElasticColumnParameters(**values["columns"])
            values["law"] = ContactLawParameters(**values["law"])
            experiments.append(ContactExperiment(**values))
        experiments = _experiments(surface, experiments)
        if set(raw["states"]) != {item.fault.fault_id for item in experiments}:
            raise ValueError("Missing or unknown contact branch history")
        states = tuple((item.fault.fault_id, state_from_dict(raw["states"][item.fault.fault_id]))
                       for item in experiments)
        _clocks(states)
        snapshot = ContactMechanicsSnapshot(expected["surface_sha256"], experiments, states)
        # Bind actual area and parameters without evolving or relaxing history.
        contacts = {item.contact_id: item for item in extract_contacts(surface.fragments, surface.radius_km)}
        for experiment, (_, history) in zip(experiments, states):
            area = contacts[experiment.fault.contact_id].length_km*1000.*experiment.contact_height_m/math.sin(experiment.fault.dip_radians)
            validate_contact_state(history, columns=experiment.columns, law=experiment.law,
                area_m2=area, water=experiment.water)
        return snapshot
    except (TypeError, KeyError) as error:
        raise ValueError("Invalid local contact checkpoint record") from error


def solved_fault_snapshot(surface, results):
    """Return the existing single-state admission input, with solved tractions.

    Slip and stress are local mechanical outputs. Plate velocities still need
    an independent physical source, and whole-sphere material transfer remains
    disabled in this experiment.
    """
    results = tuple(results)
    digest = surface_digest(surface)
    if any(not isinstance(item, ContactMechanicsResult) or item.surface_sha256 != digest for item in results):
        raise ValueError("Solved contact tractions belong to another exact surface; no implicit stress rebinding")
    return fault_snapshot_to_dict(surface, tuple(item.solved_fault for item in results))
