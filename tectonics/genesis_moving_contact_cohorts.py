"""Material contact sections during solidification after an extrinsic birth.

The original section retains its initial-stress cohesive law and immutable
birth traction. Later solid inherits no such prestress: ordinary cohorts are
activated at the OLD accepted jump before a mechanics trial. Every Newton
probe receives that same prepared history. A section freezing across an open
gap stays permanently unbonded, as in ``genesis_contact_growth``.

Depth, edge length and area are all measured in the contact-birth reference.
For a changing mesh the owner supplies depth from material volume divided by
birth face area, taking the minimum of the two adjacent faces. Current edge
length or current face area must never rescale previous contact history.
This module neither evolves bulk solid nor implements material remelting.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from numbers import Real

import numpy as np

from .genesis_contact_growth import (
    CohortState, append_cohorts, cohort_energy, empty_cohorts,
    evaluate_cohorts, validate_cohorts,
)
from .genesis_contact_law import ContactLawParameters
from .genesis_extrinsic_contact_law import (
    extrinsic_damage, extrinsic_dissipation, extrinsic_energy,
    extrinsic_return_map,
)


def _owned(value):
    array = np.ascontiguousarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _frozen(cohorts):
    return CohortState(**{f.name: _owned(getattr(cohorts, f.name))
                          for f in fields(CohortState)})


def _real(value, shape, name):
    array = np.asarray(value)
    if (array.shape != shape or array.dtype.kind not in "fiu"
            or not np.isfinite(array).all()):
        raise ValueError(f"{name} must be finite real values with shape {shape}")
    return np.asarray(array, dtype=float)


@dataclass(frozen=True)
class MovingContactHistory:
    """Separate constitutive histories; neither field is an averaged display."""

    initial: CohortState
    added: CohortState


class MovingContactCohorts:
    """Pure area-integrated extrinsic plus ordinary contact response.

    ``initial_cohorts`` contains one bonded, zero-reference record per trace
    from the original contact activation. Its material metadata and every
    birth traction are owned immutable copies. Trial histories returned by
    this helper also own read-only arrays, suitable for rollback/checkpoints.
    """

    _MATERIAL_FIELDS = (
        "trace_index", "z_lo_ref_m", "z_hi_ref_m", "area_ref_m2",
        "birth_time_myr", "birth_gap_m", "birth_jump_m", "bonded",
    )

    def __init__(self, initial_cohorts, birth_traction_pa,
                 birth_edge_length_m_per_trace, law_parameters=None,
                 bonding_gap_tolerance_m=1e-9):
        law = law_parameters or ContactLawParameters()
        if not isinstance(law, ContactLawParameters):
            raise ValueError("Contact law must be ContactLawParameters")
        law.validate()
        lengths = np.asarray(birth_edge_length_m_per_trace)
        if lengths.ndim != 1:
            raise ValueError("Birth edge length must be one-dimensional")
        self.ntraces = len(lengths)
        lengths = _real(lengths, (self.ntraces,), "birth edge length")
        if np.any(lengths <= 0):
            raise ValueError("Birth edge lengths must be positive")
        if (isinstance(bonding_gap_tolerance_m, (bool, np.bool_))
                or not isinstance(bonding_gap_tolerance_m, Real)
                or not np.isfinite(bonding_gap_tolerance_m)
                or not 0 <= bonding_gap_tolerance_m < law.damage_onset_opening_m):
            raise ValueError("Bonding gap tolerance must lie below damage onset")
        validate_cohorts(initial_cohorts, None, self.ntraces)
        c = initial_cohorts
        if (not np.array_equal(c.trace_index, np.arange(self.ntraces))
                or np.any(c.z_lo_ref_m != 0) or not np.all(c.bonded)
                or np.any(c.birth_gap_m != 0) or np.any(c.birth_jump_m != 0)):
            raise ValueError("Initial extrinsic section needs one bonded zero-reference cohort per trace")
        traction = _real(birth_traction_pa, (self.ntraces, 2), "birth traction")
        # Validate the extrinsic envelope without ever passing its damage to
        # the ordinary cohesive validator.
        extrinsic_damage(c.max_opening_m, traction, law)
        self.law_parameters = law
        self.bonding_gap_tolerance_m = float(bonding_gap_tolerance_m)
        self.birth_edge_length_m_per_trace = _owned(lengths)
        self.birth_traction_pa = _owned(traction)
        self.initial_cohorts = _frozen(c)
        self.validate(self.initial())

    def initial(self):
        """Own copies, so mutating a caller's CohortState object cannot leak."""
        return MovingContactHistory(_frozen(self.initial_cohorts), _frozen(empty_cohorts()))

    def _validate_sections(self, history):
        if not isinstance(history, MovingContactHistory):
            raise ValueError("Moving contact history must be MovingContactHistory")
        c, a = history.initial, history.added
        validate_cohorts(c, None, self.ntraces)
        validate_cohorts(a, self.law_parameters, self.ntraces)
        if np.any(a.bonded != (a.birth_gap_m <= self.bonding_gap_tolerance_m)):
            raise ValueError("Added cohort bonding must retain its original birth-gap decision")
        for name in self._MATERIAL_FIELDS:
            if not np.array_equal(getattr(c, name), getattr(self.initial_cohorts, name)):
                raise ValueError(f"Initial extrinsic material field {name} changed")
        expected_damage = extrinsic_damage(c.max_opening_m, self.birth_traction_pa,
                                           self.law_parameters)
        expected_work = c.area_ref_m2*extrinsic_dissipation(
            c.max_opening_m, self.birth_traction_pa, self.law_parameters)
        if not np.allclose(c.damage, expected_damage, rtol=1e-12, atol=1e-12):
            raise ValueError("Initial extrinsic damage disagrees with opening")
        if not np.allclose(c.fracture_work_j, expected_work, rtol=2e-10, atol=1e-6):
            raise ValueError("Initial extrinsic fracture work disagrees with opening")
        for section in (c, a):
            expected_area = (self.birth_edge_length_m_per_trace[section.trace_index]
                             *(section.z_hi_ref_m-section.z_lo_ref_m)/2)
            if not np.allclose(section.area_ref_m2, expected_area, rtol=2e-14, atol=0):
                raise ValueError("Cohort area must retain its material birth-reference measure")
        for trace in range(self.ntraces):
            selected = np.flatnonzero(a.trace_index == trace)
            if not len(selected):
                continue
            lo, hi = a.z_lo_ref_m[selected], a.z_hi_ref_m[selected]
            if lo[0] != c.z_hi_ref_m[trace] or np.any(lo[1:] != hi[:-1]):
                raise ValueError("Added material intervals must cover the front without gaps or overlap")
            times = a.birth_time_myr[selected]
            if times[0] < c.birth_time_myr[trace] or np.any(np.diff(times) < 0):
                raise ValueError("Added material birth clocks must be ordered")

    def _configuration(self, gap, jump):
        return (_real(gap, (self.ntraces,), "gap"),
                _real(jump, (self.ntraces,), "jump"))

    def validate(self, history, gap=None, jump=None):
        """Check distinct damage laws, material coverage, and optional traction.

        Supplying displacement checks an accepted state, including stored
        traction and free-open shear references. Omitting it also permits a
        prepared old history to be used for an arbitrary Newton trial.
        """
        self._validate_sections(history)
        if gap is None and jump is None:
            return
        if gap is None or jump is None:
            raise ValueError("Both gap and jump are required for state validation")
        gap, jump = self._configuration(gap, jump)
        c, a, law = history.initial, history.added, self.law_parameters
        if np.any(c.max_opening_m < np.maximum(gap, 0)):
            raise ValueError("Extrinsic history does not include the current opening")
        normal = ((1-c.damage)*self.birth_traction_pa[:, 0]
                  +law.normal_stiffness_pa_m*np.where(gap < 0, gap, (1-c.damage)*gap))
        shear = (self.birth_traction_pa[:, 1]
                 +law.tangential_stiffness_pa_m*(jump-c.plastic_slip_m))
        free = (gap >= 0) & (c.damage >= 1)
        reference = jump+self.birth_traction_pa[:, 1]/law.tangential_stiffness_pa_m
        if not np.allclose(c.plastic_slip_m[free], reference[free], rtol=1e-10, atol=1e-10):
            raise ValueError("Free-open extrinsic plastic reference is inconsistent")
        shear[free] = 0.
        if not np.allclose(c.traction_pa, np.column_stack((normal, shear)), rtol=1e-10, atol=1e-6):
            raise ValueError("Stored extrinsic traction disagrees with the accepted jump")
        i = a.trace_index
        opening = np.maximum(gap[i]-a.birth_gap_m, 0)
        if np.any(a.max_opening_m[a.bonded] < opening[a.bonded]):
            raise ValueError("Ordinary history does not include the current opening")
        normal = law.normal_stiffness_pa_m*(np.minimum(gap[i], 0)+(1-a.damage)*opening)
        shear = law.tangential_stiffness_pa_m*(jump[i]-a.birth_jump_m-a.plastic_slip_m)
        free = (gap[i] >= 0) & (a.damage >= 1)
        reference = jump[i]-a.birth_jump_m
        if not np.allclose(a.plastic_slip_m[free], reference[free], rtol=1e-10, atol=1e-10):
            raise ValueError("Free-open ordinary plastic reference is inconsistent")
        shear[free] = 0.
        if not np.allclose(a.traction_pa, np.column_stack((normal, shear)), rtol=1e-10, atol=1e-6):
            raise ValueError("Stored ordinary traction disagrees with the accepted jump")

    def front_depth(self, history):
        """Material depth covered by original and subsequently frozen cohorts."""
        self.validate(history)
        front = history.initial.z_hi_ref_m.copy()
        np.maximum.at(front, history.added.trace_index, history.added.z_hi_ref_m)
        return front

    def prepare(self, history, new_front_depth_m, old_gap, old_jump, birth_time_myr):
        """Append stress-free ordinary material once, returning (history, dE).

        Birth time is the END of the candidate trial, in the same elapsed-Myr
        clock as existing cohort birth times; the stress-free reference is
        the OLD accepted jump. This is the operator-split material activation
        used by the bulk thermal predictor. ``dE`` contains only newly born
        compression-penalty energy; it is a parameter change, not fracture,
        mechanical work, or heat. No Newton probe is allowed to append again.

        Negative front motion is rejected except roundoff of at most 128 eps
        times the existing depth, which leaves the old interval unchanged.
        """
        self.validate(history, old_gap, old_jump)
        old_gap, old_jump = self._configuration(old_gap, old_jump)
        front = self.front_depth(history)
        new_front = _real(new_front_depth_m, (self.ntraces,), "new material front")
        if (isinstance(birth_time_myr, (bool, np.bool_))
                or not isinstance(birth_time_myr, Real)
                or not np.isfinite(birth_time_myr) or birth_time_myr < 0):
            raise ValueError("Material birth time must be finite and nonnegative")
        previous_time = max(float(np.max(history.initial.birth_time_myr, initial=0)),
                            float(np.max(history.added.birth_time_myr, initial=0)))
        if birth_time_myr < previous_time:
            raise ValueError("Material birth time precedes existing history")
        tolerance = 128*np.finfo(float).eps*np.maximum(front, 1.)
        if np.any(new_front < front-tolerance):
            raise ValueError("Contact remelting is unsupported; material history cannot be removed")
        selected = np.flatnonzero(new_front > front)
        if not len(selected):
            return history, 0.
        added = append_cohorts(history.added, selected, front[selected], new_front[selected],
            self.birth_edge_length_m_per_trace, old_gap, old_jump, float(birth_time_myr),
            self.law_parameters, self.bonding_gap_tolerance_m)
        count = len(history.added.trace_index)
        fresh = CohortState(**{f.name: getattr(added, f.name)[count:] for f in fields(CohortState)})
        energy_change = cohort_energy(fresh, old_gap, old_jump, self.law_parameters)
        prepared = MovingContactHistory(_frozen(history.initial), _frozen(added))
        self.validate(prepared, old_gap, old_jump)
        return prepared, energy_change

    def evaluate(self, history, gap, jump, dt_s, water):
        """Return (trial history, trace force N, trace tangent N/m), purely."""
        self.validate(history)
        gap, jump = self._configuration(gap, jump)
        water = _real(water, (self.ntraces,), "water")
        c = history.initial
        response = extrinsic_return_map(gap, jump, c.plastic_slip_m, c.cumulative_slip_m,
            c.max_opening_m, dt_s, water, self.birth_traction_pa, self.law_parameters,
            old_damage=c.damage)
        arrays = {name: response[name] for name in (
            "plastic_slip_m", "cumulative_slip_m", "max_opening_m", "damage", "traction_pa")}
        for stored, local in (("friction_work_j", "friction_work_j_m2"),
                ("viscous_work_j", "viscous_work_j_m2"),
                ("fracture_work_j", "fracture_work_j_m2"),
                ("shear_remainder_j", "shear_relaxation_remainder_j_m2")):
            arrays[stored] = getattr(c, stored)+c.area_ref_m2*response[local]
        initial = replace(c, **arrays)
        added, force, tangent = evaluate_cohorts(history.added, gap, jump, dt_s,
                                                water, self.law_parameters)
        force += c.area_ref_m2[:, None]*response["traction_pa"]
        tangent += c.area_ref_m2[:, None, None]*response["tangent_pa_m"]
        result = MovingContactHistory(_frozen(initial), _frozen(added))
        self.validate(result, gap, jump)
        return result, force, tangent

    def energy(self, history, gap, jump):
        """Signed insertion-relative potential, with ordinary cohort energy.

        The extrinsic part retains its original signed potential convention;
        an apparent negative value is neither deposited heat nor new fracture
        work. Accepted-history consistency is required in this configuration.
        """
        self.validate(history, gap, jump)
        gap, jump = self._configuration(gap, jump)
        c = history.initial
        original = float(c.area_ref_m2@extrinsic_energy(gap, jump, c.plastic_slip_m,
            c.max_opening_m, self.birth_traction_pa, self.law_parameters))
        result = original+cohort_energy(history.added, gap, jump, self.law_parameters)
        if not np.isfinite(result):
            raise ValueError("Moving contact energy overflowed")
        return result
