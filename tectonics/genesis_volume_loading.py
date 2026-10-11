"""Prospective full-isotropic Maxwell memory in the Genesis conical chart.

Engineering order is (xx, yy, zz, xy, xz, yz). A field is ``e0+e1/lambda``
at three equal-weight angular quadrature points. Radial variance is retained
explicitly, so thin bands and zero-mean bending keep their elastic energy.
The initial strain is inherited elastic memory, never reconstructed historical
total strain. New bands activate at solidus in the accepted thermal-first
chart and receive only the current mechanical increment.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math

import numpy as np

from .genesis import SECONDS_PER_MYR

FORMAT = "genesis-volume-mechanics-0.1"
VOIGT_ORDER = ("xx", "yy", "zz", "xy", "xz", "yz")
ENERGY_SCOPE = "prospective_full_isotropic_conical_elastic_memory"
_LEDGERS = ("initial_energy_j", "elastic_energy_j", "mechanical_work_j",
            "thermal_work_j", "viscous_dissipation_j", "damage_energy_change_j",
            "remelt_energy_removed_j", "balance_residual_j")


def _array(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"Volume mechanics {name} shape or finite values mismatch")
    return result


def _scalar(value, name, nonnegative=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"Volume mechanics {name} requires a finite real")
    value = float(value)
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f"Volume mechanics {name} outside finite range")
    return value


def _ids(value):
    ids = tuple(value)
    if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) or i <= 0 for i in ids):
        raise ValueError("Volume mechanics IDs must be positive integers")
    ids = tuple(int(i) for i in ids)
    if ids != tuple(sorted(set(ids))):
        raise ValueError("Volume mechanics IDs must be unique and ordered")
    return ids


@dataclass(frozen=True, slots=True)
class RadialMoments:
    """Physical volume and normalized mean/variance of inverse lambda."""
    volume_m3: tuple
    inverse_radius_mean: tuple
    inverse_radius_variance: tuple

    def __post_init__(self):
        n = len(self.volume_m3)
        for name in ("volume_m3", "inverse_radius_mean", "inverse_radius_variance"):
            values = _array(getattr(self, name), (n,), name)
            if np.any(values < 0):
                raise ValueError("Volume radial moments cannot be negative")
            object.__setattr__(self, name, tuple(values.tolist()))
        live = np.asarray(self.volume_m3) > 0
        if np.any(np.asarray(self.inverse_radius_mean)[live] <= 0):
            raise ValueError("Positive volume requires positive inverse radius mean")


def coefficient_inner_product(a0, a1, b0, b1, constitutive_matrix, radial_moments):
    """Per-band exact integral a:C:b; no averaging away angular variance."""
    if type(radial_moments) is not RadialMoments:
        raise ValueError("Volume coefficients require typed radial moments")
    n = len(radial_moments.volume_m3)
    a0, a1, b0, b1 = (_array(v, (n, 3, 6), "coefficient") for v in (a0, a1, b0, b1))
    c = np.asarray(constitutive_matrix, dtype=float)
    c = np.broadcast_to(c, (n, 6, 6))
    if not np.isfinite(c).all():
        raise ValueError("Volume elasticity matrix is not finite")
    mean = np.asarray(radial_moments.inverse_radius_mean)[:, None, None]
    variance = np.asarray(radial_moments.inverse_radius_variance)
    am, bm = a0+mean*a1, b0+mean*b1
    density = np.einsum("cqi,cij,cqj->cq", am, c, bm)
    density += variance[:, None]*np.einsum("cqi,cij,cqj->cq", a1, c, b1)
    result = np.asarray(radial_moments.volume_m3)*density.mean(axis=1)
    if not np.isfinite(result).all():
        raise ValueError("Volume coefficient integral overflowed")
    return result


def isotropic_elasticity(young_modulus_pa, poisson_ratio):
    e = _scalar(young_modulus_pa, "Young modulus")
    nu = _scalar(poisson_ratio, "Poisson ratio")
    if e <= 0 or not -1 < nu < .5:
        raise ValueError("Volume elastic constants require E>0 and -1<nu<0.5")
    mu, lam = e/(2*(1+nu)), e*nu/((1+nu)*(1-2*nu))
    c = np.eye(6)*mu
    c[:3, :3] = lam
    c[np.arange(3), np.arange(3)] += 2*mu
    return c


@dataclass(frozen=True, slots=True)
class VolumeCohortMemory:
    id: int
    temperature_k: float
    strain0: tuple
    strain1: tuple

    def __post_init__(self):
        object.__setattr__(self, "id", _ids((self.id,))[0])
        temperature = _scalar(self.temperature_k, "temperature")
        if temperature <= 0:
            raise ValueError("Volume cohort temperature must be positive")
        object.__setattr__(self, "temperature_k", temperature)
        for name in ("strain0", "strain1"):
            values = _array(getattr(self, name), (3, 6), name)
            object.__setattr__(self, name, tuple(tuple(row) for row in values.tolist()))


@dataclass(frozen=True, slots=True)
class VolumeMechanicsState:
    source_identity: str
    time_myr: float
    accepted_steps: int
    cohorts: tuple
    initial_energy_j: float
    elastic_energy_j: float
    mechanical_work_j: float = 0.
    thermal_work_j: float = 0.
    viscous_dissipation_j: float = 0.
    damage_energy_change_j: float = 0.
    remelt_energy_removed_j: float = 0.
    balance_residual_j: float = 0.

    def __post_init__(self):
        pin = self.source_identity
        if type(pin) is not str or len(pin) != 64 or any(c not in "0123456789abcdef" for c in pin):
            raise ValueError("Volume source identity must be SHA256")
        object.__setattr__(self, "time_myr", _scalar(self.time_myr, "time", True))
        if isinstance(self.accepted_steps, bool) or type(self.accepted_steps) is not int or self.accepted_steps < 0:
            raise ValueError("Volume accepted steps must be a nonnegative integer")
        object.__setattr__(self, "cohorts", tuple(self.cohorts))
        if any(type(c) is not VolumeCohortMemory for c in self.cohorts):
            raise ValueError("Volume memory record type mismatch")
        _ids(c.id for c in self.cohorts)
        for name in _LEDGERS:
            object.__setattr__(self, name, _scalar(getattr(self, name), name))
        if any(getattr(self, name) < 0 for name in ("initial_energy_j", "elastic_energy_j", "viscous_dissipation_j", "remelt_energy_removed_j")):
            raise ValueError("Volume energy and dissipation cannot be negative")
        expected = (self.initial_energy_j+self.mechanical_work_j+self.thermal_work_j
                    -self.viscous_dissipation_j+self.damage_energy_change_j-self.remelt_energy_removed_j)
        scale = max(abs(expected), *(abs(getattr(self, n)) for n in _LEDGERS), 1.)
        residual = self.elastic_energy_j-expected
        if abs(residual-self.balance_residual_j) > 5e-12*scale or abs(residual) > 5e-12*scale:
            raise ValueError("Volume cumulative constitutive work identity mismatch")

    def __deepcopy__(self, memo):
        return self


def _damage_matrix(parameters, damage, n):
    damage = _array(damage, (n,), "damage")
    residual = _scalar(parameters.residual_stiffness, "residual stiffness")
    if not 0 < residual <= 1 or np.any((damage < 0) | (damage > 1)):
        raise ValueError("Volume damage or residual stiffness outside range")
    factor = residual+(1-residual)*(1-damage)**2
    return factor[:, None, None]*isotropic_elasticity(parameters.young_modulus_pa, parameters.poisson_ratio)


def volume_mechanics_energy(state, radial_moments, damage, parameters):
    n = len(state.cohorts)
    e0 = np.array([c.strain0 for c in state.cohorts]).reshape(n, 3, 6)
    e1 = np.array([c.strain1 for c in state.cohorts]).reshape(n, 3, 6)
    c = _damage_matrix(parameters, damage, n)
    return float(.5*coefficient_inner_product(e0, e1, e0, e1, c, radial_moments).sum())


def initial_volume_mechanics(source_identity, time_myr, ids, temperatures_k,
                             source_eps_world6, radial_moments, damage, parameters):
    """Import computed, depth-constant producer elastic memory.

    ``source_eps_world6`` must already be expressed in the SAME fixed
    orthonormal cohort basis as the operator B coefficients. The caller owns
    any rotation from world coordinates; the argument name does not perform
    or imply that rotation.
    """
    ids = _ids(ids)
    n = len(ids)
    temp = _array(temperatures_k, (n,), "source temperatures")
    strain = _array(source_eps_world6, (n, 6), "source elastic strain")
    e0 = np.repeat(strain[:, None, :], 3, axis=1)
    e1 = np.zeros_like(e0)  # computed constant producer field, not missing history
    c = _damage_matrix(parameters, damage, n)
    energy = float(.5*coefficient_inner_product(e0, e1, e0, e1, c, radial_moments).sum())
    records = tuple(VolumeCohortMemory(i, temp[j], e0[j], e1[j]) for j, i in enumerate(ids))
    return VolumeMechanicsState(source_identity, time_myr, 0, records, energy, energy)


@dataclass(frozen=True, slots=True)
class VolumeLoading:
    before_state: VolumeMechanicsState
    after_ids: tuple
    after_time_myr: float
    temperatures_k: np.ndarray
    radial_moments: RadialMoments
    old0: np.ndarray
    old1: np.ndarray
    thermal0: np.ndarray
    memory0: np.ndarray
    memory1: np.ndarray
    ratio: np.ndarray
    beta: np.ndarray
    elasticity_pa: np.ndarray
    damage_energy_change_j: float
    remelt_energy_removed_j: float

    def __post_init__(self):
        # Candidate evaluation may be repeated by Newton or rejected. Each
        # preparation owns its arrays and cannot modify accepted memory.
        for name in ("temperatures_k", "old0", "old1", "thermal0", "memory0",
                     "memory1", "ratio", "beta", "elasticity_pa"):
            values = np.array(getattr(self, name), copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)


@dataclass(frozen=True, slots=True)
class VolumeResponse:
    strain0: np.ndarray
    strain1: np.ndarray
    elastic_energy_j: float
    mechanical_work_j: float
    thermal_work_j: float
    viscous_dissipation_j: float
    damage_energy_change_j: float
    remelt_energy_removed_j: float
    balance_residual_j: float


def prepare_volume_loading(state, *, after_ids, after_temperature_k,
                           birth_temperature_k, radial_moments,
                           before_radial_moments, removed_radial_moments,
                           before_damage, after_damage, parameters, after_time_myr,
                           before_temperature_k=None):
    """Prepare one shared-clock trial without changing accepted memory.

    ``before_temperature_k``, when supplied, is the old conductive profile's
    mean on each actually retained after-ID interval, rather than its former
    uncut interval. Without it the previous cohort's stored uniform
    temperature is the explicit thermal closure. Newborn entries always use
    their solidus activation temperature, regardless of this optional array.
    """
    if type(state) is not VolumeMechanicsState:
        raise ValueError("Missing prospective volume memory; source import required")
    ids = _ids(after_ids)
    n, old_n = len(ids), len(state.cohorts)
    time = _scalar(after_time_myr, "next time", True)
    if time <= state.time_myr:
        raise ValueError("Volume trial requires a later shared thermal clock")
    for moments, count in ((radial_moments, n), (before_radial_moments, old_n), (removed_radial_moments, old_n)):
        if type(moments) is not RadialMoments or len(moments.volume_m3) != count:
            raise ValueError("Volume radial moments coverage mismatch")
    if any(v <= 0 for v in radial_moments.volume_m3):
        raise ValueError("Live volume bands require positive volume")
    temp = _array(after_temperature_k, (n,), "endpoint temperatures")
    old_profile_temperature = (None if before_temperature_k is None else
        _array(before_temperature_k, (n,), "retained old profile temperatures"))
    if old_profile_temperature is not None and np.any(old_profile_temperature <= 0):
        raise ValueError("Volume retained old profile temperatures must be positive")
    birth_temp = np.broadcast_to(np.asarray(birth_temperature_k, float), (n,))
    if not np.isfinite(birth_temp).all() or np.any(temp <= 0) or np.any(birth_temp <= 0):
        raise ValueError("Volume thermal activation requires positive finite temperatures")
    c_before, c_after = _damage_matrix(parameters, before_damage, old_n), _damage_matrix(parameters, after_damage, n)
    old0, old1 = np.zeros((n, 3, 6)), np.zeros((n, 3, 6))
    thermal = np.zeros_like(old0)
    existing = {c.id: (i, c) for i, c in enumerate(state.cohorts)}
    alpha = _scalar(parameters.linear_expansion_per_k, "thermal expansion", True)
    born = np.ones(n, dtype=bool)
    inherited_c = np.array(c_after, copy=True)
    after_index = {cohort_id: i for i, cohort_id in enumerate(ids)}
    for j, cohort_id in enumerate(ids):
        entry = existing.get(cohort_id)
        if entry is None:
            if state.cohorts and cohort_id <= max(existing):
                raise ValueError("Volume birth cannot reuse a retired cohort ID")
            delta = temp[j]-birth_temp[j]
        else:
            i, old = entry
            old0[j], old1[j] = old.strain0, old.strain1
            inherited_c[j] = c_before[i]
            before_temperature = old.temperature_k if old_profile_temperature is None else old_profile_temperature[j]
            delta, born[j] = temp[j]-before_temperature, False
        thermal[j, :, :3] = alpha*delta
    for i, old in enumerate(state.cohorts):
        j = after_index.get(old.id)
        retained_volume = 0. if j is None else radial_moments.volume_m3[j]
        before_volume, removed_volume = before_radial_moments.volume_m3[i], removed_radial_moments.volume_m3[i]
        if not math.isclose(before_volume, retained_volume+removed_volume, rel_tol=3e-12, abs_tol=0):
            raise ValueError("Volume retained and removed material do not conserve source mass")
    old_e0 = np.array([r.strain0 for r in state.cohorts]).reshape(old_n, 3, 6)
    old_e1 = np.array([r.strain1 for r in state.cohorts]).reshape(old_n, 3, 6)
    before_energy = float(.5*coefficient_inner_product(old_e0, old_e1, old_e0, old_e1, c_before, before_radial_moments).sum())
    if not math.isclose(before_energy, state.elastic_energy_j, rel_tol=5e-12, abs_tol=1e-10):
        raise ValueError("Volume stored energy disagrees with inherited coefficient field")
    removed = float(.5*coefficient_inner_product(old_e0, old_e1, old_e0, old_e1, c_before, removed_radial_moments).sum())
    retained = float(.5*coefficient_inner_product(old0, old1, old0, old1, inherited_c, radial_moments).sum())
    if not math.isclose(before_energy, retained+removed, rel_tol=5e-12, abs_tol=1e-10):
        raise ValueError("Volume radial material moments do not preserve inherited energy")
    damage_change = float(.5*coefficient_inner_product(old0, old1, old0, old1, c_after-inherited_c, radial_moments).sum())
    eta = np.clip(parameters.viscosity_reference_pa_s*np.exp(np.clip(
        parameters.activation_energy_j_mol/8.314462618*(1/temp-1/parameters.viscosity_reference_temperature_k), -60, 60)),
        parameters.viscosity_min_pa_s, parameters.viscosity_max_pa_s)
    if not np.isfinite(eta).all() or np.any(eta <= 0):
        raise ValueError("Volume endpoint viscosity must be positive and finite")
    ratio = (time-state.time_myr)*SECONDS_PER_MYR*parameters.young_modulus_pa/eta
    ratio[born] = 0.
    if not np.isfinite(ratio).all() or np.any(ratio < 0):
        raise ValueError("Volume Maxwell ratio must be finite and nonnegative")
    beta = np.divide(-np.expm1(-ratio), ratio, out=np.ones_like(ratio), where=ratio != 0)
    r = np.exp(-ratio)[:, None, None]
    memory0, memory1 = r*old0-beta[:, None, None]*thermal, r*old1
    return VolumeLoading(state, ids, time, temp.copy(), radial_moments, old0, old1,
                         thermal, memory0, memory1, ratio, beta, c_after, damage_change, removed)


def average_volume_loading_stiffness_factors(loading):
    """Exact affine average-force factor theta=(1-beta)/ratio.

    Its zero-ratio limit is 1/2, including newborn bands. This is the
    tangent of time-AVERAGE stress, distinct from endpoint tangent beta.
    The series avoids subtracting almost equal numbers for small ratios.
    """
    x = loading.ratio
    theta = np.divide(1-loading.beta, x, out=np.full_like(x, .5), where=x != 0)
    small = x < .1
    theta[small] = np.polyval((1/3628800, -1/362880, 1/40320, -1/5040,
                               1/720, -1/120, 1/24, -1/6, .5), x[small])
    return theta


def average_volume_loading_coefficients(loading, increment0, increment1):
    """Time-average elastic coefficients on the constant-driving path.

    With free thermal expansion stored as ``thermal0=alpha*deltaT``, the
    elastic driver is ``increment0-thermal0``. Thus average memory0 is
    beta*old0-theta*thermal0, memory1 is beta*old1, and each increment has
    tangent theta. Coordinates use the same fixed cohort basis as B.
    """
    n = len(loading.after_ids)
    inc0 = _array(increment0, (n, 3, 6), "average mechanical increment0")
    inc1 = _array(increment1, (n, 3, 6), "average mechanical increment1")
    beta = loading.beta[:, None, None]
    theta = average_volume_loading_stiffness_factors(loading)[:, None, None]
    return (beta*loading.old0+theta*(inc0-loading.thermal0),
            beta*loading.old1+theta*inc1)


def evaluate_volume_loading(loading, increment0, increment1):
    n = len(loading.after_ids)
    inc0 = _array(increment0, (n, 3, 6), "mechanical increment0")
    inc1 = _array(increment1, (n, 3, 6), "mechanical increment1")
    driver0, driver1 = inc0-loading.thermal0, inc1
    beta = loading.beta[:, None, None]
    end0, end1 = loading.memory0+beta*inc0, loading.memory1+beta*inc1
    x = loading.ratio
    rr = np.divide(-np.expm1(-2*x), 2*x, out=np.ones_like(x), where=x != 0)
    small = x < 1e-3
    average0, average1 = average_volume_loading_coefficients(loading, inc0, inc1)
    inner = lambda a0, a1, b0, b1: coefficient_inner_product(a0, a1, b0, b1, loading.elasticity_pa, loading.radial_moments)
    energy = float(.5*inner(end0, end1, end0, end1).sum())
    work = float(inner(inc0, inc1, average0, average1).sum())
    thermal_work = float(-inner(loading.thermal0, np.zeros_like(inc1), average0, average1).sum())
    aa, ab = x*rr, 2*(loading.beta-rr)
    bb = np.divide(1-2*loading.beta+rr, x, out=np.zeros_like(x), where=x != 0)
    ab[small] = x[small]-x[small]**2+7*x[small]**3/12-x[small]**4/4
    bb[small] = x[small]/3-x[small]**2/4+7*x[small]**3/60-x[small]**4/24
    # Positive decomposition of the exact 2x2 temporal Gram matrix.
    cross = np.divide(ab, 2*aa, out=np.zeros_like(x), where=aa != 0)
    remainder = bb-aa*cross*cross
    if np.any(remainder < -1e-12*np.maximum(bb, 1e-300)):
        raise ValueError("Volume Maxwell temporal Gram matrix lost positivity")
    remainder = np.maximum(remainder, 0.)
    shifted0 = loading.old0+cross[:, None, None]*driver0
    shifted1 = loading.old1+cross[:, None, None]*driver1
    dissipation = float((aa*inner(shifted0, shifted1, shifted0, shifted1)
        +remainder*inner(driver0, driver1, driver0, driver1)).sum())
    residual = energy-loading.before_state.elastic_energy_j-(work+thermal_work-dissipation
        +loading.damage_energy_change_j-loading.remelt_energy_removed_j)
    scale = max(energy, loading.before_state.elastic_energy_j, abs(work), abs(thermal_work), dissipation, 1.)
    if abs(residual) > 5e-12*scale:
        raise ValueError("Volume local constitutive work identity mismatch")
    return VolumeResponse(end0, end1, energy, work, thermal_work, dissipation,
        loading.damage_energy_change_j, loading.remelt_energy_removed_j, residual)


def accept_volume_loading(loading, response):
    n = len(loading.after_ids)
    e0 = _array(response.strain0, (n, 3, 6), "accepted strain0")
    e1 = _array(response.strain1, (n, 3, 6), "accepted strain1")
    actual_energy = float(.5*coefficient_inner_product(e0, e1, e0, e1,
        loading.elasticity_pa, loading.radial_moments).sum())
    if not math.isclose(actual_energy, response.elastic_energy_j, rel_tol=5e-12, abs_tol=1e-10):
        raise ValueError("Volume accepted response energy does not match strain field")
    records = tuple(VolumeCohortMemory(i, loading.temperatures_k[j], e0[j], e1[j])
                    for j, i in enumerate(loading.after_ids))
    old = loading.before_state
    cumulative = {name: getattr(old, name)+getattr(response, name) for name in (
        "mechanical_work_j", "thermal_work_j", "viscous_dissipation_j",
        "damage_energy_change_j", "remelt_energy_removed_j")}
    expected = old.initial_energy_j+cumulative["mechanical_work_j"]+cumulative["thermal_work_j"]-cumulative["viscous_dissipation_j"]+cumulative["damage_energy_change_j"]-cumulative["remelt_energy_removed_j"]
    return VolumeMechanicsState(old.source_identity, loading.after_time_myr,
        old.accepted_steps+1, records, old.initial_energy_j, response.elastic_energy_j,
        **cumulative, balance_residual_j=response.elastic_energy_j-expected)


def volume_mechanics_to_dict(state):
    body = {"format": FORMAT, "energy_scope": ENERGY_SCOPE, "voigt_order": list(VOIGT_ORDER), "state": asdict(state)}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {**body, "packet_sha256": hashlib.sha256(encoded.encode()).hexdigest()}


def volume_mechanics_from_dict(raw):
    if type(raw) is not dict or set(raw) != {"format", "energy_scope", "voigt_order", "state", "packet_sha256"}:
        raise ValueError("Volume archive fields mismatch")
    body = {key: raw[key] for key in ("format", "energy_scope", "voigt_order", "state")}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if (raw["format"] != FORMAT or raw["energy_scope"] != ENERGY_SCOPE or raw["voigt_order"] != list(VOIGT_ORDER)
            or raw["packet_sha256"] != hashlib.sha256(encoded.encode()).hexdigest()):
        raise ValueError("Volume archive version or seal mismatch")
    values = raw["state"]
    if type(values) is not dict or set(values) != {f.name for f in fields(VolumeMechanicsState)}:
        raise ValueError("Volume state archive fields mismatch")
    records = values["cohorts"]
    names = {f.name for f in fields(VolumeCohortMemory)}
    if not isinstance(records, (tuple, list)) or any(type(r) is not dict or set(r) != names for r in records):
        raise ValueError("Volume cohort archive fields mismatch")
    return VolumeMechanicsState(**{**values, "cohorts": tuple(VolumeCohortMemory(**r) for r in records)})
