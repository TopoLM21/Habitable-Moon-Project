"""Initial-stress cohesive contact for a newly inserted, zero-gap interface.

This law preserves a supplied tensile/shear traction at zero physical jump.
The supplied traction is a frozen material reference, not a balancing external
load. The normal envelope has the existing tensile peak and Mode-I work Gc.
Shear has its own cohesive Coulomb return; this is not a mixed-mode Gc law.

The potential is measured relative to insertion and can be negative. In
particular, freely sliding broken contact has shear potential -tau0**2/(2Kt).
This is an initial-stress potential convention, not an independently stored
bulk-energy transfer, deposited heat, or an additional fracture budget.
"""
from __future__ import annotations

import math
from numbers import Real

import numpy as np

from tectonics.genesis_contact_law import ContactLawParameters


def _vector(value, name, shape=None, *, nonnegative=False):
    result = np.asarray(value, dtype=float)
    if (result.ndim != 1 or (shape is not None and result.shape != shape)
            or not np.isfinite(result).all()):
        raise ValueError(f"Extrinsic {name} must be finite [interface]")
    if nonnegative and np.any(result < 0):
        raise ValueError(f"Extrinsic {name} must be nonnegative")
    return result


def _birth(birth_traction_pa, shape, params):
    if not isinstance(params, ContactLawParameters):
        raise ValueError("Extrinsic parameters must be ContactLawParameters")
    params.validate()
    birth = np.asarray(birth_traction_pa, dtype=float)
    if birth.shape != shape+(2,) or not np.isfinite(birth).all():
        raise ValueError("Extrinsic birth traction must be finite [interface, 2]")
    normal = birth[:, 0]
    if np.any((normal < 0) | (normal > params.tensile_strength_pa)):
        raise ValueError("Extrinsic birth normal traction must lie in [0, tensile strength]")
    return birth


def _scales(birth, params):
    kn, peak = params.normal_stiffness_pa_m, params.tensile_strength_pa
    initial = birth[:, 0]
    onset = (peak-initial)/kn
    ascending_work = .5*(peak+initial)*onset
    failure = onset+2*(params.fracture_energy_j_m2-ascending_work)/peak
    if (not np.isfinite(failure).all() or np.any(failure <= onset)):
        raise ValueError("Extrinsic opening scales must have a finite softening interval")
    return onset, failure


def _damage(opening, birth, params, onset, failure):
    # Clipping before multiplication also permits very large finite historical
    # maxima without overflow after complete rupture.
    k = np.minimum(opening, failure)
    denominator = birth[:, 0]+params.normal_stiffness_pa_m*k
    envelope = params.tensile_strength_pa*(failure-k)/(failure-onset)
    ratio = np.ones_like(k)
    np.divide(envelope, denominator, out=ratio, where=denominator > 0)
    return np.where(k <= onset, 0., np.clip(1-ratio, 0., 1.))


def extrinsic_damage(max_opening_m, birth_traction_pa, params):
    """Irreversible damage for nonnegative maximum physical opening [N]."""
    opening = _vector(max_opening_m, "maximum opening", nonnegative=True)
    birth = _birth(birth_traction_pa, opening.shape, params)
    onset, failure = _scales(birth, params)
    return _damage(opening, birth, params, onset, failure)


def _dissipation(opening, birth, params, onset, failure):
    # Integral of Y dD on the loading envelope. This factored form avoids
    # subtracting two nearly equal elastic energies at damage initiation.
    k = np.clip(opening, onset, failure)
    a = birth[:, 0]/params.normal_stiffness_pa_m
    factor = k/(a+k)+(a/(a+k))*(onset/(a+onset))
    result = (.5*params.tensile_strength_pa*((a+failure)/(failure-onset))
              *(k-onset)*factor)
    return np.where(opening >= failure, params.fracture_energy_j_m2,
                    np.clip(result, 0., params.fracture_energy_j_m2))


def extrinsic_dissipation(max_opening_m, birth_traction_pa, params):
    """Mode-I dissipation from insertion, ranging from zero to Gc (J/m²)."""
    opening = _vector(max_opening_m, "maximum opening", nonnegative=True)
    birth = _birth(birth_traction_pa, opening.shape, params)
    onset, failure = _scales(birth, params)
    return _dissipation(opening, birth, params, onset, failure)


def _energy(gap, jump, slip, damage, birth, params):
    kn, kt = params.normal_stiffness_pa_m, params.tangential_stiffness_pa_m
    elastic_slip = jump-slip
    normal = ((1-damage)*birth[:, 0]*gap
              +.5*kn*np.where(gap < 0, gap**2, (1-damage)*gap**2))
    shear = birth[:, 1]*elastic_slip+.5*kt*elastic_slip**2
    return normal+shear


def extrinsic_energy(gap_m, jump_m, plastic_slip_m, max_opening_m,
                     birth_traction_pa, params):
    """Signed relative potential per interface area, holding damage fixed.

    The historical maximum must include the current positive gap. The shear
    energy includes the linear birth-traction term and is not nonnegative.
    """
    gap = _vector(gap_m, "gap")
    jump = _vector(jump_m, "jump", gap.shape)
    slip = _vector(plastic_slip_m, "plastic slip", gap.shape)
    opening = _vector(max_opening_m, "maximum opening", gap.shape, nonnegative=True)
    if np.any(opening < np.maximum(gap, 0.)):
        raise ValueError("Extrinsic maximum opening must include the current gap")
    birth = _birth(birth_traction_pa, gap.shape, params)
    onset, failure = _scales(birth, params)
    result = _energy(gap, jump, slip, _damage(opening, birth, params, onset, failure), birth, params)
    if not np.isfinite(result).all():
        raise ValueError("Extrinsic potential overflowed")
    return result


def extrinsic_return_map(gap_m, jump_m, old_plastic_slip_m, old_cumulative_slip_m,
                         old_max_opening_m, dt_s, water, birth_traction_pa, params,
                         *, old_damage=None):
    """Pure vectorized trial and branchwise consistent traction Jacobian.

    History and jump arrays have shape [N]; birth traction has shape [N,2].
    Birth normal traction must lie between zero and the unchanged tensile
    strength. The caller must establish shear yield admissibility at birth;
    an overstressed input is returned plastically, not silently accepted as a
    new strength. Water changes shear strength, not the normal fracture work.

    Normal traction unloads with the current damage from the initial-stress
    spring. Compression retains its undamaged penalty stiffness and uses
    Kn*max(-gap,0) as Coulomb pressure. Fully broken open traces relax shear
    freely even for nonzero viscosity; their plastic reference is jump+tau0/Kt.

    ``recoverable_energy_j_m2`` is the signed insertion-relative potential.
    The shear relaxation remainder is the backward-Euler spring remainder,
    including free release, and is not classified as physical fracture work.
    No input array or accepted constitutive history is modified.
    """
    if isinstance(dt_s, bool) or not isinstance(dt_s, Real) or not math.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("Extrinsic dt_s must be finite and positive")
    gap = _vector(gap_m, "gap")
    jump = _vector(jump_m, "jump", gap.shape)
    old_slip = _vector(old_plastic_slip_m, "old plastic slip", gap.shape)
    cumulative = _vector(old_cumulative_slip_m, "old cumulative slip", gap.shape, nonnegative=True)
    old_opening = _vector(old_max_opening_m, "old maximum opening", gap.shape, nonnegative=True)
    wet = _vector(water, "water", gap.shape)
    if np.any(np.abs(old_slip) > cumulative+1e-10):
        raise ValueError("Extrinsic cumulative slip must bound signed plastic slip")
    if np.any((wet < 0) | (wet > 1)):
        raise ValueError("Extrinsic water must lie in [0, 1]")
    birth = _birth(birth_traction_pa, gap.shape, params)
    onset, failure = _scales(birth, params)
    previous_damage = _damage(old_opening, birth, params, onset, failure)
    if old_damage is not None:
        stored = _vector(old_damage, "old damage", gap.shape)
        if (np.any((stored < 0) | (stored > 1))
                or not np.allclose(stored, previous_damage, rtol=1e-12, atol=1e-12)):
            raise ValueError("Extrinsic old damage is inconsistent with maximum opening")

    kn, kt = params.normal_stiffness_pa_m, params.tangential_stiffness_pa_m
    initial_normal, initial_shear = birth[:, 0], birth[:, 1]
    opening = np.maximum(old_opening, np.maximum(gap, 0.))
    damage = _damage(opening, birth, params, onset, failure)
    derivative = np.zeros_like(gap)
    softening = (gap > old_opening) & (gap > onset) & (gap < failure)
    # Algebraically equivalent to Sn*(t0+Kn*gf)/(L*(t0+Kn*g)^2),
    # with only displacement scales squared.
    a = initial_normal/kn
    derivative[softening] = (params.tensile_strength_pa/kn
                            *(a[softening]+failure[softening])
                            /((failure[softening]-onset[softening])
                              *(a[softening]+gap[softening])**2))
    normal = (1-damage)*initial_normal+kn*np.where(gap < 0, gap, (1-damage)*gap)
    normal_tangent = (kn*np.where(gap < 0, 1., 1-damage)
                      -derivative*(initial_normal+kn*np.maximum(gap, 0.)))
    pressure = kn*np.maximum(-gap, 0.)
    friction = params.friction_dry+(params.friction_wet-params.friction_dry)*wet
    cohesion = params.cohesion_pa*(1-(1-params.wet_cohesion_fraction)*wet)
    strength = (1-damage)*cohesion+friction*pressure
    strength_derivative = -cohesion*derivative-friction*kn*(gap < 0)

    trial = initial_shear+kt*(jump-old_slip)
    yielding = np.abs(trial) > strength
    free = (gap >= 0) & (damage >= 1.)
    viscous = np.full_like(gap, params.viscosity_pa_s_m/dt_s)
    viscous[free] = 0.
    increment = np.sign(trial)*np.maximum(np.abs(trial)-strength, 0.)/(kt+viscous)
    slip = old_slip+increment
    slip[free] = jump[free]+initial_shear[free]/kt
    increment[free] = slip[free]-old_slip[free]
    shear = initial_shear+kt*(jump-slip)
    shear[free] = 0.
    tangent = np.zeros((len(gap), 2, 2))
    tangent[:, 0, 0] = normal_tangent
    tangent[:, 1, 1] = np.where(yielding, kt*viscous/(kt+viscous), kt)
    tangent[:, 1, 0] = np.where(yielding, np.sign(trial)*kt/(kt+viscous)*strength_derivative, 0.)
    tangent[free, 1, :] = 0.
    result = {
        "traction_pa": np.column_stack((normal, shear)),
        "tangent_pa_m": tangent,
        "plastic_slip_m": slip,
        "cumulative_slip_m": cumulative+np.abs(increment),
        "max_opening_m": opening,
        "damage": damage,
        "slip_increment_m": increment,
        "shear_strength_pa": strength,
        "normal_pressure_pa": pressure,
        "friction_work_j_m2": strength*np.abs(increment),
        "viscous_work_j_m2": viscous*increment**2,
        "shear_relaxation_remainder_j_m2": .5*kt*increment**2,
        "fracture_work_j_m2": (_dissipation(opening, birth, params, onset, failure)
                               -_dissipation(old_opening, birth, params, onset, failure)),
        "recoverable_energy_j_m2": _energy(gap, jump, slip, damage, birth, params),
        "damage_onset_opening_m": onset,
        "failure_opening_m": failure,
    }
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("Extrinsic response overflowed; check separation and parameter scales")
    return result
