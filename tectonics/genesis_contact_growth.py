"""Independent contact histories for newly solidified interface intervals.

Areas and depth intervals are material/reference measures. This helper neither
creates bulk material nor infers solidification from a changing geometry. A
cohort born across an open crack is permanently unbonded: freezing separate
banks does not supply an unmodelled bridge across the gap. Normal compression
always sees the actual gap, independently of cohesive birth offsets.

The small-strain traction law follows ``genesis_contact_law``. Birth references
are analogous to stress-free material activation, while compression may store
penalty energy immediately if the supplied birth configuration penetrates.
The caller must account for that parameter-energy change and reject remelting
unless it explicitly retires material and its recoverable energy.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from numbers import Integral, Real

import numpy as np

from .genesis_contact_law import cohesive_damage, cohesive_dissipation


@dataclass
class CohortState:
    trace_index: np.ndarray
    z_lo_ref_m: np.ndarray
    z_hi_ref_m: np.ndarray
    area_ref_m2: np.ndarray
    birth_time_myr: np.ndarray
    birth_gap_m: np.ndarray
    birth_jump_m: np.ndarray
    bonded: np.ndarray
    plastic_slip_m: np.ndarray
    cumulative_slip_m: np.ndarray
    max_opening_m: np.ndarray
    damage: np.ndarray
    traction_pa: np.ndarray
    friction_work_j: np.ndarray
    viscous_work_j: np.ndarray
    fracture_work_j: np.ndarray
    shear_remainder_j: np.ndarray


def empty_cohorts():
    return CohortState(**{
        f.name: np.empty((0, 2) if f.name == "traction_pa" else (0,),
                         dtype=np.int64 if f.name == "trace_index" else
                         bool if f.name == "bonded" else float)
        for f in fields(CohortState)})


def validate_cohorts(cohorts, law=None, ntraces=None):
    """Check local shapes, admissibility, and irreversible constitutive state.

    The owner additionally checks geometry, material-front coverage, clocks,
    and correspondence between the stored traction and current displacement.
    """
    if not isinstance(cohorts, CohortState):
        raise ValueError("Contact cohorts must be a CohortState")
    index = cohorts.trace_index
    if not isinstance(index, np.ndarray) or index.ndim != 1 or index.dtype.kind not in "iu":
        raise ValueError("Cohort trace_index must be a one-dimensional integer array")
    count = len(index)
    if np.any(index < 0):
        raise ValueError("Cohort trace indices must be nonnegative")
    if ntraces is not None:
        if isinstance(ntraces, bool) or not isinstance(ntraces, Integral) or ntraces < 0:
            raise ValueError("Contact trace count must be a nonnegative integer")
        if np.any(index >= ntraces):
            raise ValueError("Cohort trace index is outside the contact geometry")
    for f in fields(cohorts):
        value = getattr(cohorts, f.name)
        shape = (count, 2) if f.name == "traction_pa" else (count,)
        if not isinstance(value, np.ndarray) or value.shape != shape:
            raise ValueError(f"Cohort {f.name} has an invalid shape")
        kind = value.dtype.kind
        if f.name == "bonded":
            if kind != "b":
                raise ValueError("Cohort bonded must have boolean dtype")
        elif kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError(f"Cohort {f.name} must be finite numeric values")
    for name in ("z_lo_ref_m", "birth_time_myr", "birth_gap_m", "cumulative_slip_m",
                 "max_opening_m", "friction_work_j", "viscous_work_j",
                 "fracture_work_j", "shear_remainder_j"):
        if np.any(getattr(cohorts, name) < 0):
            raise ValueError(f"Cohort {name} must be nonnegative")
    if (np.any(cohorts.z_hi_ref_m <= cohorts.z_lo_ref_m)
            or np.any(cohorts.area_ref_m2 <= 0)):
        raise ValueError("Cohort depth intervals and reference areas must be positive")
    if (np.any((cohorts.damage < 0) | (cohorts.damage > 1))
            or np.any(np.abs(cohorts.plastic_slip_m) > cohorts.cumulative_slip_m+1e-10)):
        raise ValueError("Cohort damage or cumulative slip is inconsistent")
    unbonded = ~cohorts.bonded
    if (np.any(cohorts.damage[unbonded] != 1)
            or np.any(cohorts.max_opening_m[unbonded] != 0)
            or np.any(cohorts.fracture_work_j[unbonded] != 0)):
        raise ValueError("Unbonded birth cannot acquire cohesive damage history or fracture work")
    if law is not None:
        law.validate()
        bonded = cohorts.bonded
        if not np.allclose(cohorts.damage[bonded], cohesive_damage(cohorts.max_opening_m[bonded], law),
                           rtol=1e-12, atol=1e-12):
            raise ValueError("Cohort damage disagrees with maximum relative opening")
        expected = cohorts.area_ref_m2[bonded]*cohesive_dissipation(cohorts.max_opening_m[bonded], law)
        if not np.allclose(cohorts.fracture_work_j[bonded], expected, rtol=2e-10, atol=1e-6):
            raise ValueError("Cohort fracture work disagrees with irreversible opening")


def _trace_array(value, name, ntraces=None):
    value = np.asarray(value, dtype=float)
    if value.ndim != 1 or not np.isfinite(value).all() or (ntraces is not None and len(value) != ntraces):
        raise ValueError(f"Contact {name} must be a finite array per trace")
    return value


def append_cohorts(cohorts, trace_index, z_lo_ref_m, z_hi_ref_m,
                   edge_length_m_per_trace, gap_m_per_trace, jump_m_per_trace,
                   birth_time_myr, law, bonding_gap_tolerance_m):
    """Append fresh reference intervals without changing existing histories.

    ``trace_index`` and depth bounds describe only the new records; edge
    lengths and displacement jumps cover the complete current trace geometry.
    Areas are edge length times depth interval divided by two (two endpoints).
    The caller prevents interval overlap and unsupported material creation.
    """
    gap = _trace_array(gap_m_per_trace, "gap")
    n = len(gap)
    jump = _trace_array(jump_m_per_trace, "jump", n)
    length = _trace_array(edge_length_m_per_trace, "edge length", n)
    if np.any(length <= 0):
        raise ValueError("Contact reference edge lengths must be positive")
    validate_cohorts(cohorts, law, n)
    index = np.asarray(trace_index)
    if index.ndim != 1 or index.dtype.kind not in "iu" or np.any((index < 0) | (index >= n)):
        raise ValueError("New cohort trace indices must be valid integers")
    lo = _trace_array(z_lo_ref_m, "new lower depth", len(index))
    hi = _trace_array(z_hi_ref_m, "new upper depth", len(index))
    for value, name in ((birth_time_myr, "birth time"), (bonding_gap_tolerance_m, "bonding tolerance")):
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
            raise ValueError(f"Cohort {name} must be finite and nonnegative")
    if bonding_gap_tolerance_m >= law.damage_onset_opening_m:
        raise ValueError("Cohort bonding tolerance must be below the cohesive damage onset")
    bonded = gap[index] <= bonding_gap_tolerance_m
    zero = np.zeros(len(index))
    fresh = CohortState(index.astype(np.int64, copy=True), lo.copy(), hi.copy(),
        length[index]*(hi-lo)/2, np.full(len(index), birth_time_myr, dtype=float),
        np.maximum(gap[index], 0), jump[index].copy(), bonded,
        zero.copy(), zero.copy(), zero.copy(), (~bonded).astype(float),
        np.column_stack((law.normal_stiffness_pa_m*np.minimum(gap[index], 0), zero)),
        zero.copy(), zero.copy(), zero.copy(), zero.copy())
    validate_cohorts(fresh, law, n)
    return CohortState(**{f.name: np.concatenate((getattr(cohorts, f.name), getattr(fresh, f.name)), axis=0)
                          for f in fields(cohorts)})


def evaluate_cohorts(cohorts, gap_m_per_trace, jump_m_per_trace, dt_s, water_per_trace, law):
    """Propose local histories and summed force/Jacobian on each trace.

    Inputs remain immutable, so every Newton evaluation uses the same old
    state. Returned forces are N and tangents N/m, already area integrated.
    Dissipative ledgers include irreversible increments; their finite-step
    sum is not asserted to equal the global mechanical work exactly.
    """
    gap_all = _trace_array(gap_m_per_trace, "gap")
    n = len(gap_all)
    jump_all = _trace_array(jump_m_per_trace, "jump", n)
    water = _trace_array(water_per_trace, "water", n)
    if np.any((water < 0) | (water > 1)):
        raise ValueError("Contact water must lie in [0, 1]")
    if isinstance(dt_s, bool) or not isinstance(dt_s, Real) or not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("Contact dt_s must be finite and positive")
    validate_cohorts(cohorts, law, n)
    c = cohorts
    i = c.trace_index
    gap, jump = gap_all[i], jump_all[i]-c.birth_jump_m
    opening_now = np.maximum(gap-c.birth_gap_m, 0)
    opening = np.where(c.bonded, np.maximum(c.max_opening_m, opening_now), 0)
    damage = np.where(c.bonded, cohesive_damage(opening, law), 1.)
    derivative = np.zeros(len(i))
    onset, failure = law.damage_onset_opening_m, law.failure_opening_m
    softening = c.bonded & (opening_now > c.max_opening_m) & (opening_now > onset) & (opening_now < failure)
    derivative[softening] = failure*onset/((failure-onset)*opening_now[softening]**2)
    kn, kt = law.normal_stiffness_pa_m, law.tangential_stiffness_pa_m
    normal = kn*(np.minimum(gap, 0)+(1-damage)*opening_now)
    normal_tangent = kn*np.where(gap < 0, 1.,
        np.where(gap >= c.birth_gap_m, 1-damage-opening_now*derivative, 0))
    pressure = kn*np.maximum(-gap, 0)
    friction = law.friction_dry+(law.friction_wet-law.friction_dry)*water[i]
    cohesion = law.cohesion_pa*(1-(1-law.wet_cohesion_fraction)*water[i])
    strength = (1-damage)*cohesion+friction*pressure
    strength_derivative = -cohesion*derivative-friction*kn*(gap < 0)
    trial = kt*(jump-c.plastic_slip_m)
    yielding = np.abs(trial) > strength
    free = (gap >= 0) & (damage >= 1)
    viscous = np.full(len(i), law.viscosity_pa_s_m/dt_s)
    viscous[free] = 0
    increment = np.sign(trial)*np.maximum(np.abs(trial)-strength, 0)/(kt+viscous)
    slip = c.plastic_slip_m+increment
    slip[free] = jump[free]
    increment[free] = jump[free]-c.plastic_slip_m[free]
    shear = kt*(jump-slip)
    shear[free] = 0
    tangent = np.zeros((len(i), 2, 2))
    tangent[:, 0, 0] = normal_tangent
    tangent[:, 1, 1] = np.where(yielding, kt*viscous/(kt+viscous), kt)
    tangent[:, 1, 0] = np.where(yielding, np.sign(trial)*kt/(kt+viscous)*strength_derivative, 0)
    tangent[free, 1, :] = 0
    area = c.area_ref_m2
    fracture = np.where(c.bonded, cohesive_dissipation(opening, law)-cohesive_dissipation(c.max_opening_m, law), 0)
    updated = replace(c, plastic_slip_m=slip, cumulative_slip_m=c.cumulative_slip_m+np.abs(increment),
        max_opening_m=opening, damage=damage, traction_pa=np.column_stack((normal, shear)),
        friction_work_j=c.friction_work_j+area*strength*np.abs(increment),
        viscous_work_j=c.viscous_work_j+area*viscous*increment**2,
        fracture_work_j=c.fracture_work_j+area*fracture,
        shear_remainder_j=c.shear_remainder_j+area*.5*kt*increment**2)
    force = np.zeros((n, 2))
    integrated_tangent = np.zeros((n, 2, 2))
    np.add.at(force, i, updated.traction_pa*area[:, None])
    np.add.at(integrated_tangent, i, tangent*area[:, None, None])
    if not (np.isfinite(force).all() and np.isfinite(integrated_tangent).all()
            and all(np.isfinite(getattr(updated, f.name)).all() for f in fields(updated))):
        raise ValueError("Cohort constitutive response overflowed")
    return updated, force, integrated_tangent


def cohort_energy(cohorts, gap_pertrace, jump_pertrace, law):
    """Recoverable energy of current histories in the supplied configuration."""
    gap_all = _trace_array(gap_pertrace, "gap")
    jump_all = _trace_array(jump_pertrace, "jump", len(gap_all))
    validate_cohorts(cohorts, law, len(gap_all))
    c = cohorts
    gap = gap_all[c.trace_index]
    opening = np.maximum(gap-c.birth_gap_m, 0)
    elastic_slip = jump_all[c.trace_index]-c.birth_jump_m-c.plastic_slip_m
    density = .5*law.normal_stiffness_pa_m*(np.minimum(gap, 0)**2+(1-c.damage)*opening**2)
    density += .5*law.tangential_stiffness_pa_m*elastic_slip**2
    result = float(np.dot(c.area_ref_m2, density))
    if not np.isfinite(result):
        raise ValueError("Cohort recoverable energy overflowed")
    return result


def aggregate_cohorts(cohorts, ntraces):
    """ContactState-shaped DISPLAY summaries; never constitutive history.

    Mechanical history and traction are area weighted. Work arrays are sums
    of joules, preserving all irreversible work without dilution by growth.
    Empty traces have zero values. The owning model supplies global fields.
    """
    validate_cohorts(cohorts, ntraces=ntraces)
    i = cohorts.trace_index
    area = np.bincount(i, weights=cohorts.area_ref_m2, minlength=ntraces)
    result = {}
    for name, source in (("plastic_slip_m", "plastic_slip_m"), ("cumulative_slip_m", "cumulative_slip_m"),
                         ("max_opening_m", "max_opening_m"), ("interface_damage", "damage")):
        weighted = np.bincount(i, weights=cohorts.area_ref_m2*getattr(cohorts, source), minlength=ntraces)
        result[name] = np.divide(weighted, area, out=np.zeros(ntraces), where=area > 0)
    traction = np.zeros((ntraces, 2))
    np.add.at(traction, i, cohorts.area_ref_m2[:, None]*cohorts.traction_pa)
    result["traction_pa"] = np.divide(traction, area[:, None], out=np.zeros_like(traction), where=area[:, None] > 0)
    for name, source in (("friction_work_cell_j", "friction_work_j"), ("viscous_work_cell_j", "viscous_work_j"),
                         ("fracture_work_cell_j", "fracture_work_j"), ("shear_remainder_cell_j", "shear_remainder_j")):
        result[name] = np.bincount(i, weights=getattr(cohorts, source), minlength=ntraces)
    return result


def remap_cohorts(cohorts, trace_mapping):
    """Map old trace indices into an expanded topology without changing records."""
    mapping = np.asarray(trace_mapping)
    if mapping.ndim != 1 or mapping.dtype.kind not in "iu" or np.any(mapping < 0):
        raise ValueError("Cohort trace mapping must contain nonnegative integers")
    if len(np.unique(mapping)) != len(mapping):
        raise ValueError("Cohort trace remapping must be injective")
    validate_cohorts(cohorts, ntraces=len(mapping))
    return replace(cohorts, trace_index=mapping[cohorts.trace_index].astype(np.int64, copy=True))
