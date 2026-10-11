"""Closed local contact between two supported elastic columns.

This is a frozen-geometry mechanical substep. The local displacement is the
overriding bank minus the sinking bank, resolved along the fault normal and
the down-dip tangent. Negative plastic tangential slip is down-dip sinking.
Column anchors are physical imposed boundary displacements, not new relaxed
references. No material transfer, plate velocity solve, opening or heat update
is performed here. All constitutive response comes from contact_return_map.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from numbers import Integral, Real

import numpy as np
from scipy.optimize import brentq

from tectonics.genesis_contact_law import ContactLawParameters, contact_return_map


def _real(value, name, *, positive=False, nonnegative=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(float(value))):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if (positive and value <= 0.) or (nonnegative and value < 0.):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return value


@dataclass(frozen=True, slots=True)
class ElasticColumnParameters:
    """Two 1-D spring columns per unit inclined-interface area.

    Lengths are measured along the declared fault normal, rather than radial
    lithosphere thicknesses. The effective normal modulus must encode the
    declared lateral boundary closure (for example E for laterally free
    uniaxial loading). Tangential stiffness uses the supplied shear modulus.
    """
    sub_length_m: float
    sub_normal_modulus_pa: float
    sub_shear_modulus_pa: float
    over_length_m: float
    over_normal_modulus_pa: float
    over_shear_modulus_pa: float

    def __post_init__(self):
        for item in fields(self):
            object.__setattr__(self, item.name,
                               _real(getattr(self, item.name), item.name, positive=True))
        if not all(math.isfinite(k) and k > 0 for k in (
                self.normal_stiffness_pa_m, self.shear_stiffness_pa_m)):
            raise ValueError("Elastic column compliance must give finite positive stiffness")

    @property
    def normal_stiffness_pa_m(self):
        compliance = (self.sub_length_m/self.sub_normal_modulus_pa
                      + self.over_length_m/self.over_normal_modulus_pa)
        return 1./compliance if compliance > 0. else math.inf

    @property
    def shear_stiffness_pa_m(self):
        compliance = (self.sub_length_m/self.sub_shear_modulus_pa
                      + self.over_length_m/self.over_shear_modulus_pa)
        return 1./compliance if compliance > 0. else math.inf


@dataclass(frozen=True, slots=True)
class LocalContactState:
    columns_signature: tuple[float, ...]
    law_signature: tuple[float, ...]
    water: float
    area_m2: float
    gap_free_m: float
    jump_free_m: float
    gap_m: float
    jump_m: float
    plastic_slip_m: float
    cumulative_slip_m: float
    traction_normal_pa: float
    traction_shear_pa: float
    bulk_energy_j: float
    contact_energy_j: float
    cumulative_external_work_j: float
    cumulative_friction_work_j: float
    cumulative_viscous_work_j: float
    cumulative_numerical_dissipation_j: float
    elapsed_time_s: float
    step_count: int


@dataclass(frozen=True, slots=True)
class LocalContactStep:
    state: LocalContactState
    traction_pa: tuple[float, float]
    slip_increment_m: float
    shear_strength_pa: float
    normal_pressure_pa: float
    external_work_j: float
    friction_work_j: float
    viscous_work_j: float
    backward_euler_remainder_j: float
    bulk_energy_increment_j: float
    contact_energy_increment_j: float
    energy_balance_residual_j: float
    energy_balance_relative_error: float
    normal_force_residual_pa: float
    shear_force_residual_pa: float
    force_relative_residual: float


def _signature(parameters):
    return tuple(float(getattr(parameters, item.name)) for item in fields(parameters))


def _close(actual, expected, name):
    scale = max(abs(actual), abs(expected), np.finfo(float).tiny)
    if abs(actual-expected) > 512*np.finfo(float).eps*scale:
        raise ValueError(f"Stored {name} is inconsistent with mechanical history")


def _equilibrium_close(stiffness, displacement, anchor, traction, name):
    # A long accumulated slip can leave a small elastic shortening. Account
    # for rounding in the displacement subtraction, rather than treating the
    # resulting small reaction as the magnitude of every arithmetic operand.
    spring = stiffness*(displacement-anchor)
    scale = max(abs(traction), stiffness*(abs(displacement)+abs(anchor)),
                np.finfo(float).tiny)
    if abs(spring+traction) > 512*np.finfo(float).eps*scale:
        raise ValueError(f"Stored {name} is inconsistent with mechanical history")


def _validate_state(state):
    if not isinstance(state, LocalContactState):
        raise ValueError("Previous history must be LocalContactState")
    for name, count in (("columns_signature", len(fields(ElasticColumnParameters))),
                        ("law_signature", len(fields(ContactLawParameters)))):
        values = getattr(state, name)
        if not isinstance(values, tuple) or len(values) != count:
            raise ValueError(f"Stored {name} has invalid shape")
        for value in values:
            _real(value, name)
    columns = ElasticColumnParameters(*state.columns_signature)
    law = ContactLawParameters(**dict(zip((f.name for f in fields(ContactLawParameters)),
                                         state.law_signature)))
    law.validate()
    for item in fields(LocalContactState):
        if item.name not in ("columns_signature", "law_signature", "step_count"):
            _real(getattr(state, item.name), f"Stored {item.name}")
    if (isinstance(state.step_count, (bool, np.bool_))
            or not isinstance(state.step_count, Integral) or state.step_count < 0):
        raise ValueError("Stored step_count must be a nonnegative integer")
    if state.step_count > 0 and state.elapsed_time_s <= 0:
        raise ValueError("Evolved mechanical history must have positive elapsed time")
    if not 0 <= state.water <= 1 or state.area_m2 <= 0:
        raise ValueError("Stored water or contact area is invalid")
    if state.gap_free_m > 0 or state.gap_m > 0:
        raise ValueError("Opening histories are unsupported in the closed contact model")
    if (state.cumulative_slip_m < 0
            or abs(state.plastic_slip_m) > state.cumulative_slip_m+1e-12*max(state.cumulative_slip_m, 1e-300)):
        raise ValueError("Stored cumulative slip must bound signed plastic slip")
    for name in ("bulk_energy_j", "contact_energy_j", "cumulative_friction_work_j",
                 "cumulative_viscous_work_j", "cumulative_numerical_dissipation_j",
                 "elapsed_time_s"):
        if getattr(state, name) < 0:
            raise ValueError(f"Stored {name} must be nonnegative")
    kn, kt = columns.normal_stiffness_pa_m, columns.shear_stiffness_pa_m
    tn = law.normal_stiffness_pa_m*state.gap_m
    tt = law.tangential_stiffness_pa_m*(state.jump_m-state.plastic_slip_m)
    _close(state.traction_normal_pa, tn, "normal traction")
    _close(state.traction_shear_pa, tt, "shear traction")
    _equilibrium_close(kn, state.gap_m, state.gap_free_m, tn, "normal equilibrium")
    _equilibrium_close(kt, state.jump_m, state.jump_free_m, tt, "shear equilibrium")
    bulk = .5*state.area_m2*(kn*(state.gap_m-state.gap_free_m)**2
                            + kt*(state.jump_m-state.jump_free_m)**2)
    contact = .5*state.area_m2*(law.normal_stiffness_pa_m*state.gap_m**2
                              + law.tangential_stiffness_pa_m
                              *(state.jump_m-state.plastic_slip_m)**2)
    _close(state.bulk_energy_j, bulk, "bulk energy")
    _close(state.contact_energy_j, contact, "contact energy")
    accounted = math.fsum((bulk, contact, state.cumulative_friction_work_j,
                          state.cumulative_viscous_work_j,
                          state.cumulative_numerical_dissipation_j))
    # The signed external ledger can decrease during unloading. Its net value
    # still equals stored energy plus accumulated dissipation from zero birth.
    ledger_scale = max(abs(state.cumulative_external_work_j), accounted,
                       np.finfo(float).tiny)
    if abs(state.cumulative_external_work_j-accounted) > 2e-10*ledger_scale:
        raise ValueError("Stored cumulative mechanical work balance is inconsistent")
    if state.step_count == 0 and any(getattr(state, item.name) != 0
            for item in fields(LocalContactState)
            if item.name not in ("columns_signature", "law_signature", "water", "area_m2", "step_count")):
        raise ValueError("Initial mechanical history must be unstressed at zero anchors")


def validate_contact_state(state, *, columns, law, area_m2, water):
    """Check accepted physical history/binding without evolving fictitious time."""
    if not isinstance(columns, ElasticColumnParameters) or not isinstance(law, ContactLawParameters):
        raise ValueError("Explicit elastic columns and contact law are required")
    law.validate()
    area = _real(area_m2, "area_m2", positive=True)
    wet = _real(water, "water", nonnegative=True)
    if wet > 1:
        raise ValueError("water must lie in [0, 1]")
    _validate_state(state)
    if (state.columns_signature != _signature(columns) or state.law_signature != _signature(law)
            or state.area_m2 != area or state.water != wet):
        raise ValueError("Mechanical restart cannot change frozen compliance, law, area or water")
    return state


def state_to_dict(state):
    """Strict JSON-compatible mechanical history without rebasing references."""
    _validate_state(state)
    result = asdict(state)
    result["columns_signature"] = list(state.columns_signature)
    result["law_signature"] = list(state.law_signature)
    return result


def state_from_dict(raw):
    if not isinstance(raw, dict) or set(raw) != {f.name for f in fields(LocalContactState)}:
        raise ValueError("Local contact history has missing or unknown fields")
    copied = dict(raw)
    for name in ("columns_signature", "law_signature"):
        if not isinstance(copied[name], (list, tuple)):
            raise ValueError(f"Stored {name} must be a list of physical parameters")
        copied[name] = tuple(_real(value, name) for value in copied[name])
    state = LocalContactState(**copied)
    _validate_state(state)
    return state


def solve_compressed_contact(*, columns, law, area_m2, gap_free_m, jump_free_m,
                             dt_s, water, previous=None):
    """Solve column/contact equilibrium and propose an immutable next history.

    Anchors, area, water and every physical law parameter are explicit. The
    law, compliance, area and water remain fixed throughout this local history.
    First use starts from zero anchors and zero elastic/plastic displacement.
    Positive free opening is rejected; no Mode-I damage branch is hidden.

    External work uses the endpoint anchor reaction times the anchor increment.
    Its exact backward-Euler balance includes the increments of both spring
    energies, friction, viscosity and the global spring-increment remainder.
    The return-map's fixed-jump shear relaxation remainder is NOT added here.
    """
    if not isinstance(columns, ElasticColumnParameters):
        raise ValueError("Explicit ElasticColumnParameters are required")
    if not isinstance(law, ContactLawParameters):
        raise ValueError("Explicit ContactLawParameters are required")
    law.validate()
    area = _real(area_m2, "area_m2", positive=True)
    gfree = _real(gap_free_m, "gap_free_m")
    jfree = _real(jump_free_m, "jump_free_m")
    dt = _real(dt_s, "dt_s", positive=True)
    wet = _real(water, "water", nonnegative=True)
    if wet > 1:
        raise ValueError("water must lie in [0, 1]")
    if gfree > 0:
        raise ValueError("Free normal opening is unsupported; use the compressed branch")
    cs, ls = _signature(columns), _signature(law)
    if previous is None:
        previous = LocalContactState(cs, ls, wet, area, *([0.]*15), 0)
    validate_contact_state(previous, columns=columns, law=law, area_m2=area, water=wet)
    kn, kt = columns.normal_stiffness_pa_m, columns.shear_stiffness_pa_m
    # Stable series-spring expression; no pressure is inferred from age or ID.
    gap = gfree/(1.+law.normal_stiffness_pa_m/kn)
    if not math.isfinite(gap):
        raise ValueError("Normal contact response overflowed")

    def constitutive(jump):
        return contact_return_map(np.array([gap]), np.array([jump]),
            np.array([previous.plastic_slip_m]), np.array([previous.cumulative_slip_m]),
            np.zeros(1), dt, np.array([wet]), law, old_damage=np.zeros(1))

    def residual(jump):
        value = kt*(jump-jfree)+float(constitutive(jump)["traction_pa"][0, 1])
        if not math.isfinite(value):
            raise ValueError("Tangential force residual overflowed")
        return value

    low, high = sorted((jfree, previous.plastic_slip_m))
    if low == high:
        jump = low
    else:
        displacement_scale = max(abs(low), abs(high), np.finfo(float).tiny)
        jump = brentq(residual, low, high,
            xtol=max(np.nextafter(0., 1.), 8*np.finfo(float).eps*displacement_scale),
            rtol=8*np.finfo(float).eps)
    response = constitutive(jump)
    scalar = lambda name: float(response[name][0])
    tn, tt = (float(v) for v in response["traction_pa"][0])
    slip = scalar("plastic_slip_m")
    slip_increment = scalar("slip_increment_m")
    bulk = .5*area*(kn*(gap-gfree)**2+kt*(jump-jfree)**2)
    contact = area*scalar("recoverable_energy_j_m2")
    friction = area*scalar("friction_work_j_m2")
    viscous = area*scalar("viscous_work_j_m2")
    dgfree, djfree = gfree-previous.gap_free_m, jfree-previous.jump_free_m
    external = area*math.fsum((tn*dgfree, tt*djfree))
    dg = gap-previous.gap_m
    dgn = (gap-gfree)-(previous.gap_m-previous.gap_free_m)
    djt = (jump-jfree)-(previous.jump_m-previous.jump_free_m)
    delastic = (jump-slip)-(previous.jump_m-previous.plastic_slip_m)
    remainder = .5*area*(kn*dgn**2+kt*djt**2
        +law.normal_stiffness_pa_m*dg**2+law.tangential_stiffness_pa_m*delastic**2)
    dbulk, dcontact = bulk-previous.bulk_energy_j, contact-previous.contact_energy_j
    balance = math.fsum((external, -dbulk, -dcontact, -friction, -viscous, -remainder))
    energy_scale = max(abs(external), abs(dbulk)+abs(dcontact)+friction+viscous+remainder,
                       bulk+previous.bulk_energy_j+contact+previous.contact_energy_j,
                       np.finfo(float).tiny)
    energy_error = abs(balance)/energy_scale
    rn, rt = kn*(gap-gfree)+tn, kt*(jump-jfree)+tt
    force_scale = max(abs(kn*(gap-gfree))+abs(tn), abs(kt*(jump-jfree))+abs(tt),
                      np.finfo(float).tiny)
    force_error = max(abs(rn), abs(rt))/force_scale
    finite = (bulk, contact, friction, viscous, remainder, external, balance,
              rn, rt, energy_error, force_error)
    if not all(math.isfinite(v) for v in finite):
        raise ValueError("Mechanical work or force calculation overflowed")
    if force_error > 2e-10 or energy_error > 2e-10:
        raise RuntimeError("Closed contact solve failed independent force/work balance checks")
    state = LocalContactState(cs, ls, wet, area, gfree, jfree, gap, jump, slip,
        scalar("cumulative_slip_m"), tn, tt, bulk, contact,
        math.fsum((previous.cumulative_external_work_j, external)),
        math.fsum((previous.cumulative_friction_work_j, friction)),
        math.fsum((previous.cumulative_viscous_work_j, viscous)),
        math.fsum((previous.cumulative_numerical_dissipation_j, remainder)),
        math.fsum((previous.elapsed_time_s, dt)),
        previous.step_count+1)
    _validate_state(state)
    return LocalContactStep(state, (tn, tt), slip_increment,
        scalar("shear_strength_pa"), scalar("normal_pressure_pa"), external,
        friction, viscous, remainder, dbulk, dcontact, balance, energy_error,
        rn, rt, force_error)
