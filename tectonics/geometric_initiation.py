"""Signed reactivation of an explicitly oriented, pre-existing weak fault.

This is a local forced-underthrust admissibility test under prescribed loading,
not a solver for bending, deformation, or self-sustained subduction. Neither
fault dip nor a missing stress tensor is inferred from age, damage or plate ID.
The caller owns the snapshot's time validity. The prescribed boundary model
is contactwise parallel transport from the midpoint: effective stress is given
in global Cartesian axes there, then rotated along the great circle about the
contact normal together with the local radial/fault basis. Dip, cohesion and
friction are constant on that contact. Resolved local tractions are therefore
constant along the arc. This explicit homogeneous loading assumption is not a
spatial stress solution or permission to extrapolate an arbitrary point sample.
Pore pressure has already been subtracted once from the supplied tensor.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np

from .geometric_contacts import GeometricContact


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Fault {name} must be a nonempty string")


def _number(value, name, *, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or (float(value) <= 0 if positive else float(value) < 0)):
        raise ValueError(f"Fault {name} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


def _plate(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"Fault {name} must be a nonnegative integer")


def _tensor(value):
    if value is None:
        return None
    raw = np.asarray(value)
    if raw.shape != (3, 3) or raw.dtype.kind not in 'iuf' or not np.isfinite(raw).all():
        raise ValueError("Fault effective stress must be a finite real 3 by 3 tensor")
    # Explicit scalar validation rejects bools mixed with numbers before NumPy
    # coerces them into an otherwise plausible real tensor.
    for row in value:
        for item in row:
            if isinstance(item, (bool, np.bool_)) or not isinstance(item, Real):
                raise ValueError("Fault effective stress entries must be finite real numbers")
    array = np.asarray(value, dtype=float)
    scale = float(np.max(np.abs(array)))
    tolerance = 64*np.finfo(float).eps*scale
    if float(np.max(np.abs(array-array.T))) > tolerance:
        raise ValueError("Fault effective stress must be symmetric")
    # Canonicalize roundoff-level asymmetry, never an actual antisymmetric load.
    array = .5*array+.5*array.T
    return tuple(tuple(float(item) for item in row) for row in array)


@dataclass(frozen=True)
class OrientedFault:
    fault_id: str
    contact_id: str
    subducting_plate: int
    overriding_plate: int
    dip_radians: float
    effective_stress_pa: tuple[tuple[float, float, float], ...] | None
    cohesion_pa: float
    friction: float
    provenance: str

    def __post_init__(self):
        for name in ('fault_id', 'contact_id', 'provenance'):
            _text(getattr(self, name), name)
        for name in ('subducting_plate', 'overriding_plate'):
            _plate(getattr(self, name), name)
        if self.subducting_plate == self.overriding_plate:
            raise ValueError("Fault owners must differ")
        dip = _number(self.dip_radians, 'dip_radians', positive=True)
        if dip >= math.pi/2:
            raise ValueError("Fault dip must lie strictly between zero and pi/2")
        _number(self.cohesion_pa, 'cohesion_pa')
        _number(self.friction, 'friction')
        for name in ('subducting_plate', 'overriding_plate'):
            object.__setattr__(self, name, int(getattr(self, name)))
        for name in ('dip_radians', 'cohesion_pa', 'friction'):
            object.__setattr__(self, name, float(getattr(self, name)))
        object.__setattr__(self, 'effective_stress_pa', _tensor(self.effective_stress_pa))


@dataclass(frozen=True)
class FaultAssessment:
    fault_id: str
    contact_id: str
    subducting_plate: int
    overriding_plate: int
    status: str
    normal_stress_pa: float | None
    shear_stress_pa: float | None
    strength_pa: float | None
    margin_pa: float | None
    normal_velocity_min_km_per_myr: float
    normal_velocity_max_km_per_myr: float
    stress_tolerance_pa: float
    velocity_tolerance_km_per_myr: float


def _unit(value, name):
    raw = np.asarray(value)
    if raw.shape != (3,) or raw.dtype.kind not in 'iuf' or not np.isfinite(raw).all():
        raise ValueError(f"Contact {name} must be a finite real unit vector")
    array = np.asarray(value, dtype=float)
    if abs(float(np.linalg.norm(array))-1) > 2e-10:
        raise ValueError(f"Contact {name} must be a unit vector")
    return array/np.linalg.norm(array)


def _contact_geometry(contact, fault, radius):
    if not isinstance(contact, GeometricContact):
        raise ValueError("Fault evaluation requires a geometric contact")
    if contact.contact_id != fault.contact_id:
        raise ValueError("Fault and contact identity disagree")
    _plate(contact.plate_a, 'contact.plate_a')
    _plate(contact.plate_b, 'contact.plate_b')
    if ({contact.plate_a, contact.plate_b}
            != {fault.subducting_plate, fault.overriding_plate}):
        raise ValueError("Fault owners disagree with the contact")
    start, end, midpoint, normal = (
        _unit(getattr(contact, name), name)
        for name in ('start', 'end', 'midpoint', 'normal_a_to_b'))
    angle = math.atan2(float(np.linalg.norm(np.cross(start, end))), float(np.dot(start, end)))
    if not 0 < angle < math.pi:
        raise ValueError("Contact must have a positive minor arc")
    expected_midpoint = start+end
    expected_midpoint /= np.linalg.norm(expected_midpoint)
    if (np.linalg.norm(midpoint-expected_midpoint) > 2e-10
            or max(abs(float(np.dot(normal, point))) for point in (start, end, midpoint)) > 2e-10):
        raise ValueError("Contact midpoint and normal must match the shared arc")
    length = _number(contact.length_km, 'contact.length_km', positive=True)
    if not math.isclose(length, radius*angle, rel_tol=2e-10, abs_tol=64*np.finfo(float).eps*radius):
        raise ValueError("Contact length disagrees with its radius and arc")
    return start, end, midpoint, normal, angle


def _normal_velocity_extrema(start, end, normal, angle, omega_delta, radius):
    # r(t)=a cos(t)+b sin(t); velocity is an exact sinusoid on a great circle.
    # Recover the tangent from the long supporting normal, not a subtraction of
    # almost coincident endpoints. This also handles very short contact arcs.
    tangent = np.cross(normal, start)
    tangent /= np.linalg.norm(tangent)
    if float(np.dot(tangent, end-start)) < 0:
        tangent = -tangent
    coefficient = radius*np.cross(normal, omega_delta)
    a = float(np.dot(coefficient, start))
    b = float(np.dot(coefficient, tangent))
    values = [a, a*math.cos(angle)+b*math.sin(angle)]
    stationary = math.atan2(b, a)
    for k in (-2, -1, 0, 1, 2):
        position = stationary+k*math.pi
        if 0 < position < angle:
            values.append(a*math.cos(position)+b*math.sin(position))
    tolerance = 128*np.finfo(float).eps*radius*float(np.linalg.norm(omega_delta))
    if not all(math.isfinite(value) for value in (*values, tolerance)):
        raise ValueError("Contact velocity calculation overflowed")
    return min(values), max(values), tolerance


def evaluate_fault(fault, contact, omega_rad_per_myr, radius_km):
    """Evaluate signed Coulomb reactivation and convergence over the whole arc.

    The normal points from the proposed sinking bank into the overriding bank.
    Under the declared contactwise parallel-transport loading, the midpoint
    traction equals the traction everywhere on the arc: stress and fault basis
    undergo the same rotation about the supporting great-circle normal.
    For radial ``r`` and that horizontal direction ``h``, down-dip ``d`` is
    ``cos(dip)*h-sin(dip)*r`` and the plane normal is
    ``sin(dip)*h+cos(dip)*r``. In our compression-positive stress convention,
    ``d @ stress @ normal`` is positive for down-dip reactivation under
    horizontal compression. Reverse shear cannot activate this candidate.

    A point at exact yield is locked; tolerances only bound arithmetic error.
    Convergence must be nonpositive everywhere and strictly negative somewhere;
    an isolated stagnant endpoint does not veto a finite convergent arc.
    Kinematics are recomputed from the supplied omega, never cached velocities.
    """
    if not isinstance(fault, OrientedFault):
        raise ValueError("Expected an OrientedFault")
    radius = _number(radius_km, 'radius_km', positive=True)
    start, end, radial, contact_normal, angle = _contact_geometry(contact, fault, radius)
    raw = np.asarray(omega_rad_per_myr)
    if (raw.ndim != 2 or raw.shape[1] != 3 or raw.dtype.kind not in 'iuf'
            or not np.isfinite(raw).all()
            or max(contact.plate_a, contact.plate_b) >= len(raw)):
        raise ValueError("Fault kinematics require finite real angular velocities [plate,3]")
    omega = np.asarray(omega_rad_per_myr, dtype=float)
    minimum, maximum, velocity_tolerance = _normal_velocity_extrema(
        start, end, contact_normal, angle, omega[contact.plate_b]-omega[contact.plate_a], radius)
    normal_stress = shear = strength = margin = None
    stress_tolerance = 0.
    if fault.effective_stress_pa is not None:
        tensor = np.asarray(fault.effective_stress_pa, dtype=float)
        horizontal = contact_normal if fault.overriding_plate == contact.plate_b else -contact_normal
        sine, cosine = math.sin(fault.dip_radians), math.cos(fault.dip_radians)
        down_dip = cosine*horizontal-sine*radial
        plane_normal = sine*horizontal+cosine*radial
        normal_stress = float(plane_normal@tensor@plane_normal)
        shear = float(down_dip@tensor@plane_normal)
        strength = float(fault.cohesion_pa+fault.friction*max(normal_stress, 0.))
        margin = shear-strength
        scale = max(float(np.max(np.abs(tensor))), abs(shear), abs(normal_stress), strength)
        stress_tolerance = 128*np.finfo(float).eps*scale
        if not all(math.isfinite(value) for value in (normal_stress, shear, strength, margin, stress_tolerance)):
            raise ValueError("Fault stress calculation overflowed")
    if minimum < -velocity_tolerance and maximum > velocity_tolerance:
        status = 'mixed_convergence'
    elif minimum >= -velocity_tolerance:
        status = 'nonconvergent'
    elif fault.effective_stress_pa is None:
        status = 'missing_stress'
    elif normal_stress < -stress_tolerance:
        status = 'effective_tension'
    elif margin <= stress_tolerance:
        status = 'locked'
    else:
        status = 'forced_underthrust_admissible'
    return FaultAssessment(
        fault.fault_id, fault.contact_id, fault.subducting_plate, fault.overriding_plate,
        status, normal_stress, shear, strength, margin, minimum, maximum,
        stress_tolerance, velocity_tolerance)


__all__ = ['OrientedFault', 'FaultAssessment', 'evaluate_fault']
