"""Private fixed-material Maxwell continuation of a frozen finite seed.

The inherited elastic field is e0+e1/lambda; the additional elastic field is
G_birth z.  The compact displacement amplitude a and elastic-memory amplitude
z are distinct.  Both start at zero, without adding absolute birth motion to
the inherited strain.  This module neither advances Genesis nor releases or
funds a physical crack.  It currently rejects every thermal, damage, material
or moving-boundary change.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math

import numpy as np

from .genesis_volume_embedded_partition import (
    GenesisVolumeEmbeddedPartition, validate_volume_embedded_partition,
)
from .genesis_volume_loading import (
    RadialMoments, VolumeLoading, VolumeMechanicsState, VolumeResponse,
    accept_volume_loading, average_volume_loading_coefficients,
    average_volume_loading_stiffness_factors, coefficient_inner_product,
    evaluate_volume_loading, volume_mechanics_to_dict,
)
from .genesis_volume_operator import (
    ANGULAR_BARYCENTRIC, _immutable, evaluate_genesis_volume_operator,
)

FORMAT = "genesis-volume-enriched-memory-0.1"
SCOPE = "private_fixed_material_frozen_birth_Maxwell_no_release_or_funding"
_LEDGERS = ("initial_energy_j", "elastic_energy_j", "mechanical_work_j",
            "viscous_dissipation_j", "thermal_work_j", "damage_energy_change_j",
            "remelt_energy_removed_j", "balance_residual_j")


class EnrichedMemoryRefusal(ValueError):
    pass


def _array(value, shape, name):
    a = np.asarray(value, float)
    if a.shape != shape or not np.isfinite(a).all():
        raise EnrichedMemoryRefusal(f"Enriched memory {name} shape or finite values mismatch")
    return a


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def _digest(body):
    return hashlib.sha256(json.dumps(_plain(body), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _moments(p):
    a = p.parent_assembly
    return RadialMoments(tuple(a.volume_m3), tuple(a.inverse_scale_mean),
                         tuple(a.inverse_scale_variance))


def _coefficients(mechanics):
    n = len(mechanics.cohorts)
    return (np.asarray([c.strain0 for c in mechanics.cohorts]).reshape(n, 3, 6),
            np.asarray([c.strain1 for c in mechanics.cohorts]).reshape(n, 3, 6))


def _cross_moment(p, e0, e1):
    """Integral G^T C e in the literal archived frozen quadrature."""
    l = p.local
    dtype = np.longdouble
    angular = np.asarray(l.barycentric, dtype) @ np.asarray(
        np.linalg.inv(ANGULAR_BARYCENTRIC), dtype)
    point = angular @ np.asarray(e0[p.selected_cohort_index], dtype)
    point += (angular @ np.asarray(e1[p.selected_cohort_index], dtype)) / np.asarray(
        l.radial_scale, dtype)[:, None]
    return np.asarray(np.einsum("pai,pa,p->i",
        np.asarray(l.enrichment_strain_operator, dtype),
        point @ np.asarray(l.elasticity_pa, dtype).T,
        np.asarray(l.reference_volume_weights_m3, dtype)), float)


def _inner_parts(p, a0, a1, za, b0, b1, zb):
    old = float(coefficient_inner_product(a0, a1, b0, b1,
        p.parent_assembly.C, _moments(p)).sum())
    cross = 0.
    if np.any(za):
        cross += float(za @ _cross_moment(p, b0, b1))
    if np.any(zb):
        cross += float(zb @ _cross_moment(p, a0, a1))
    own = float(za @ p.local.stiffness_enriched @ zb)
    return np.asarray((old, cross, own))


def enriched_field_inner_product(p, a0, a1, za, b0, b1, zb):
    """Full-field inner product, retaining signed cross terms without clipping."""
    validate_volume_embedded_partition(p)
    shape = p.memory0.shape
    a0, a1, b0, b1 = (_array(v, shape, "elastic coefficients")
                       for v in (a0, a1, b0, b1))
    za, zb = (_array(v, (18,), "elastic enrichment") for v in (za, zb))
    return float(_inner_parts(p, a0, a1, za, b0, b1, zb).sum())


@dataclass(frozen=True, slots=True)
class EnrichedMemoryStep:
    after_time_myr: float
    parent_increment_m: tuple
    enrichment_increment_m: tuple
    ratio: tuple

    def __post_init__(self):
        if not math.isfinite(self.after_time_myr) or self.after_time_myr < 0:
            raise EnrichedMemoryRefusal("Private journal time must be finite and nonnegative")
        for name in ("parent_increment_m", "enrichment_increment_m", "ratio"):
            a = np.asarray(getattr(self, name), float)
            if a.ndim != 1 or not np.isfinite(a).all() or (name == "ratio" and np.any(a < 0)):
                raise EnrichedMemoryRefusal("Private journal coefficients are invalid")
            object.__setattr__(self, name, tuple(a.tolist()))


@dataclass(frozen=True, slots=True)
class EnrichedMemoryState:
    parent_state: VolumeMechanicsState
    partition_binding_sha256: str
    birth_parent_memory_sha256: str
    birth_time_myr: float
    birth_operator_energy_roundoff_j: float
    displacement_m: np.ndarray
    coefficients_m: np.ndarray
    initial_energy_j: float
    elastic_energy_j: float
    mechanical_work_j: float
    viscous_dissipation_j: float
    thermal_work_j: float
    damage_energy_change_j: float
    remelt_energy_removed_j: float
    balance_residual_j: float
    journal: tuple
    binding_sha256: str

    def __post_init__(self):
        if type(self.parent_state) is not VolumeMechanicsState:
            raise EnrichedMemoryRefusal("Typed retained parent mechanics are required")
        for name in ("partition_binding_sha256", "birth_parent_memory_sha256"):
            pin = getattr(self, name)
            if type(pin) is not str or len(pin) != 64 or any(c not in "0123456789abcdef" for c in pin):
                raise EnrichedMemoryRefusal("Private memory source binding must be SHA256")
        if not math.isfinite(self.birth_time_myr) or self.birth_time_myr < 0:
            raise EnrichedMemoryRefusal("Private memory birth time is invalid")
        if not math.isfinite(self.birth_operator_energy_roundoff_j):
            raise EnrichedMemoryRefusal("Inherited operator energy roundoff witness is invalid")
        for name in ("displacement_m", "coefficients_m"):
            object.__setattr__(self, name, _immutable(_array(getattr(self, name), (18,), name)))
        object.__setattr__(self, "journal", tuple(self.journal))
        if any(type(v) is not EnrichedMemoryStep for v in self.journal):
            raise EnrichedMemoryRefusal("Private journal record type mismatch")
        for name in _LEDGERS:
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not math.isfinite(value):
                raise EnrichedMemoryRefusal("Private full-field ledger must be finite")
        if any(getattr(self, n) < 0 for n in ("initial_energy_j", "elastic_energy_j", "viscous_dissipation_j", "remelt_energy_removed_j")):
            raise EnrichedMemoryRefusal("Full-field energy and dissipation must be nonnegative")
        expected = (self.initial_energy_j+self.mechanical_work_j+self.thermal_work_j
                    -self.viscous_dissipation_j+self.damage_energy_change_j-self.remelt_energy_removed_j)
        residual = self.elastic_energy_j-expected
        scale = max(1., *(abs(getattr(self, n)) for n in _LEDGERS))
        if abs(residual) > 5e-12*scale or abs(residual-self.balance_residual_j) > 5e-12*scale:
            raise EnrichedMemoryRefusal("Private cumulative full-field work identity mismatch")
        if self.binding_sha256 != _digest(_state_body(self)):
            raise EnrichedMemoryRefusal("Private memory state binding mismatch")

    @property
    def mechanics(self): return self.parent_state
    @property
    def time_myr(self): return self.parent_state.time_myr
    @property
    def private_steps(self): return len(self.journal)
    @property
    def physical_release_admitted(self): return False
    @property
    def funding_admitted(self): return False


def _step_body(step):
    return {f.name: _plain(getattr(step, f.name)) for f in fields(EnrichedMemoryStep)}


def _state_body(s):
    return dict(parent_mechanics=_plain(volume_mechanics_to_dict(s.parent_state)),
        partition_binding_sha256=s.partition_binding_sha256,
        birth_parent_memory_sha256=s.birth_parent_memory_sha256,
        birth_time_myr=s.birth_time_myr,
        birth_operator_energy_roundoff_j=s.birth_operator_energy_roundoff_j,
        displacement_m=_plain(s.displacement_m),
        coefficients_m=_plain(s.coefficients_m),
        **{n: getattr(s, n) for n in _LEDGERS},
        journal=[_step_body(v) for v in s.journal])


def _new_state(**values):
    # Construct a body before the frozen record's self-seal validation.
    class Pending: pass
    pending = Pending()
    for name, value in values.items(): setattr(pending, name, value)
    return EnrichedMemoryState(**values, binding_sha256=_digest(_state_body(pending)))


def initial_enriched_memory(p, parent_state):
    validate_volume_embedded_partition(p)
    if type(parent_state) is not VolumeMechanicsState:
        raise EnrichedMemoryRefusal("Source-bound typed original mechanics are required")
    e0, e1 = _coefficients(parent_state)
    if (parent_state.source_identity != p.source_identity or parent_state.time_myr != p.time_myr
            or tuple(c.id for c in parent_state.cohorts) != tuple(p.parent_assembly.cohort_ids)
            or not np.array_equal(e0, p.memory0) or not np.array_equal(e1, p.memory1)
            or not math.isclose(parent_state.elastic_energy_j, p.initial_elastic_energy_j,
                                rel_tol=5e-12, abs_tol=1e-6)):
        raise EnrichedMemoryRefusal("Birth must preserve the exact source clock, material and inherited memory")
    return _new_state(parent_state=parent_state, partition_binding_sha256=p.binding_sha256,
        birth_parent_memory_sha256=volume_mechanics_to_dict(parent_state)["packet_sha256"],
        birth_time_myr=p.time_myr,
        birth_operator_energy_roundoff_j=p.initial_elastic_energy_j-parent_state.elastic_energy_j,
        displacement_m=np.zeros(18), coefficients_m=np.zeros(18),
        **{n: getattr(parent_state, n) for n in _LEDGERS}, journal=())


def enriched_memory_energy(p, state):
    e0, e1 = _coefficients(state.parent_state)
    return float(.5*_inner_parts(p, e0, e1, state.coefficients_m,
                                e0, e1, state.coefficients_m).sum())


def validate_enriched_memory(p, state):
    validate_volume_embedded_partition(p)
    if (type(state) is not EnrichedMemoryState or state.partition_binding_sha256 != p.binding_sha256
            or state.binding_sha256 != _digest(_state_body(state))
            or state.birth_time_myr != p.time_myr
            or state.parent_state.source_identity != p.source_identity
            or tuple(c.id for c in state.parent_state.cohorts) != tuple(p.parent_assembly.cohort_ids)):
        raise EnrichedMemoryRefusal("Private state differs from frozen source/material binding")
    energy = enriched_memory_energy(p, state)
    scale = max(energy, state.elastic_energy_j, 1.)
    if abs(energy-state.elastic_energy_j) > 5e-12*scale:
        raise EnrichedMemoryRefusal("Private stored energy differs from its complete elastic field")
    return state


@dataclass(frozen=True, slots=True)
class EnrichedLoading:
    before_state: EnrichedMemoryState
    parent_loading: VolumeLoading
    binding_sha256: str


def _loading_body(p, state, loading):
    return dict(partition_binding_sha256=p.binding_sha256,
        before_state_sha256=state.binding_sha256, after_time_myr=loading.after_time_myr,
        ratio=_plain(loading.ratio), beta=_plain(loading.beta))


def prepare_enriched_loading(p, state, loading):
    """Bind a sourced original Maxwell trial to fixed compact material memory."""
    validate_enriched_memory(p, state)
    if type(loading) is not VolumeLoading or volume_mechanics_to_dict(loading.before_state) != volume_mechanics_to_dict(state.parent_state):
        raise EnrichedMemoryRefusal("Private trial must use the retained original mechanics")
    n = len(p.memory0)
    e0, e1 = _coefficients(state.parent_state)
    thermal = _array(loading.thermal0, (n, 3, 6), "thermal strain")
    fixed_temperature = np.asarray([v.temperature_k for v in state.parent_state.cohorts])
    moments = _moments(p)
    if (loading.after_ids != tuple(p.parent_assembly.cohort_ids)
            or loading.radial_moments != moments
            or not np.array_equal(loading.temperatures_k, fixed_temperature)
            or not np.array_equal(loading.elasticity_pa, p.parent_assembly.C)
            or np.any(thermal) or loading.damage_energy_change_j != 0
            or loading.remelt_energy_removed_j != 0):
        raise EnrichedMemoryRefusal("Compact memory requires fixed live material, temperature, elasticity and boundaries")
    x = _array(loading.ratio, (n,), "Maxwell ratio")
    if np.any(x < 0) or not math.isfinite(loading.after_time_myr) or loading.after_time_myr <= state.time_myr:
        raise EnrichedMemoryRefusal("Private Maxwell trial requires finite nonnegative ratios and a later clock")
    beta = np.divide(-np.expm1(-x), x, out=np.ones_like(x), where=x != 0)
    r = np.exp(-x)[:, None, None]
    if (not np.array_equal(loading.old0, e0) or not np.array_equal(loading.old1, e1)
            or not np.array_equal(loading.beta, beta)
            or not np.array_equal(loading.memory0, r*e0)
            or not np.array_equal(loading.memory1, r*e1)):
        raise EnrichedMemoryRefusal("Original and compact Maxwell factors or inherited fields differ")
    return EnrichedLoading(state, loading, _digest(_loading_body(p, state, loading)))


@dataclass(frozen=True, slots=True)
class EnrichedResponse:
    strain0: np.ndarray
    strain1: np.ndarray
    coefficients_m: np.ndarray
    displacement_m: np.ndarray
    mean0: np.ndarray
    mean1: np.ndarray
    mean_coefficients_m: np.ndarray
    mean_generalized_force_n: np.ndarray
    elastic_energy_j: float
    mechanical_work_j: float
    viscous_dissipation_j: float
    balance_residual_j: float
    energy_decomposition_j: tuple
    work_decomposition_j: tuple
    dissipation_decomposition_j: tuple
    parent_response: VolumeResponse
    parent_increment_m: np.ndarray
    enrichment_increment_m: np.ndarray
    trial_binding_sha256: str

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, np.ndarray): object.__setattr__(self, f.name, _immutable(v))
        parent = self.parent_response
        object.__setattr__(self, "parent_response", VolumeResponse(**{
            f.name: _immutable(getattr(parent, f.name)) if isinstance(getattr(parent, f.name), np.ndarray)
            else getattr(parent, f.name) for f in fields(VolumeResponse)}))


def evaluate_enriched_loading(p, trial, parent_increment_m, enrichment_increment_m=None):
    """Evaluate the complete constitutive field on a private constant-drive path."""
    if type(trial) is not EnrichedLoading:
        raise EnrichedMemoryRefusal("Bound private Maxwell trial is required")
    checked = prepare_enriched_loading(p, trial.before_state, trial.parent_loading)
    if checked.binding_sha256 != trial.binding_sha256:
        raise EnrichedMemoryRefusal("Private Maxwell trial binding changed")
    state, loading = trial.before_state, trial.parent_loading
    q = _array(parent_increment_m, (p.parent_ndof,), "parent displacement increment")
    da = np.zeros(18) if enrichment_increment_m is None else _array(enrichment_increment_m, (18,), "compact displacement increment")
    local = q[p.parent_assembly.dofs]
    inc0 = np.einsum("cqai,ci->cqa", p.parent_assembly.B0, local)
    inc1 = np.einsum("cqai,ci->cqa", p.parent_assembly.B1, local)
    parent = evaluate_volume_loading(loading, inc0, inc1)
    mean0, mean1 = average_volume_loading_coefficients(loading, inc0, inc1)
    theta = average_volume_loading_stiffness_factors(loading)
    ci, z = p.selected_cohort_index, state.coefficients_m
    zend = np.exp(-loading.ratio[ci])*z+loading.beta[ci]*da
    zmean = loading.beta[ci]*z+theta[ci]*da
    energy_parts = .5*_inner_parts(p, parent.strain0, parent.strain1, zend,
                                     parent.strain0, parent.strain1, zend)
    work_parts = _inner_parts(p, inc0, inc1, da, mean0, mean1, zmean)
    x = loading.ratio
    rr = np.divide(-np.expm1(-2*x), 2*x, out=np.ones_like(x), where=x != 0)
    aa, ab = x*rr, 2*(loading.beta-rr)
    bb = np.divide(1-2*loading.beta+rr, x, out=np.zeros_like(x), where=x != 0)
    small = x < 1e-3
    ab[small] = x[small]-x[small]**2+7*x[small]**3/12-x[small]**4/4
    bb[small] = x[small]/3-x[small]**2/4+7*x[small]**3/60-x[small]**4/24
    cross = np.divide(ab, 2*aa, out=np.zeros_like(x), where=aa != 0)
    remainder = bb-aa*cross*cross
    if np.any(remainder < -1e-12*np.maximum(bb, 1e-300)):
        raise EnrichedMemoryRefusal("Complete Maxwell temporal Gram matrix lost positivity")
    remainder = np.maximum(remainder, 0.)
    # The old analytic term uses each cohort's own x.  Only the selected
    # support carries z.  Signed cross corrections are retained verbatim.
    shifted0 = loading.old0+cross[:, None, None]*inc0
    shifted1 = loading.old1+cross[:, None, None]*inc1
    zshift = z+cross[ci]*da
    old_loss = parent.viscous_dissipation_j
    shifted = _inner_parts(p, shifted0, shifted1, zshift, shifted0, shifted1, zshift)
    driven = _inner_parts(p, inc0, inc1, da, inc0, inc1, da)
    loss_parts = np.r_[old_loss, aa[ci]*shifted[1:]+remainder[ci]*driven[1:]]
    energy, work, loss = (float(v.sum()) for v in (energy_parts, work_parts, loss_parts))
    residual = energy-state.elastic_energy_j-(work-loss)
    scale = max(1., energy, state.elastic_energy_j, abs(work), abs(loss))
    if energy < 0 or loss < 0 or abs(residual) > 5e-12*scale:
        raise EnrichedMemoryRefusal("Complete fixed-material Maxwell energy/work/dissipation identity failed")
    meanforce = evaluate_genesis_volume_operator(p.parent_assembly, np.zeros(p.parent_ndof),
        elastic_constant=mean0, elastic_inverse_scale=mean1).force.copy()
    np.add.at(meanforce, p.selected_parent_dofs, p.local.stiffness_cross@zmean)
    meanextra = _cross_moment(p, mean0, mean1)+p.local.stiffness_enriched@zmean
    return EnrichedResponse(parent.strain0, parent.strain1, zend, state.displacement_m+da,
        mean0, mean1, zmean, np.r_[meanforce, meanextra], energy, work, loss, residual,
        tuple(energy_parts), tuple(work_parts), tuple(loss_parts), parent,
        q, da, trial.binding_sha256)


def accept_enriched_loading(p, trial, response):
    """Accept only into the private constitutive branch, never the planet."""
    if type(response) is not EnrichedResponse or response.trial_binding_sha256 != trial.binding_sha256:
        raise EnrichedMemoryRefusal("Private response differs from trial binding")
    actual = evaluate_enriched_loading(p, trial, response.parent_increment_m,
                                       response.enrichment_increment_m)
    for f in fields(EnrichedResponse):
        a, b = getattr(response, f.name), getattr(actual, f.name)
        if f.name == "parent_response":
            if any(not np.array_equal(getattr(a, g.name), getattr(b, g.name)) for g in fields(VolumeResponse)):
                raise EnrichedMemoryRefusal("Private parent response was modified")
        elif not np.array_equal(a, b):
            raise EnrichedMemoryRefusal("Private full-field response was modified")
    old, loading = trial.before_state, trial.parent_loading
    mechanics = accept_volume_loading(loading, actual.parent_response)
    work = old.mechanical_work_j+actual.mechanical_work_j
    loss = old.viscous_dissipation_j+actual.viscous_dissipation_j
    expected = old.initial_energy_j+work+old.thermal_work_j-loss+old.damage_energy_change_j-old.remelt_energy_removed_j
    step = EnrichedMemoryStep(loading.after_time_myr, tuple(actual.parent_increment_m),
        tuple(actual.enrichment_increment_m), tuple(loading.ratio))
    return _new_state(parent_state=mechanics,
        partition_binding_sha256=old.partition_binding_sha256,
        birth_parent_memory_sha256=old.birth_parent_memory_sha256,
        birth_time_myr=old.birth_time_myr,
        birth_operator_energy_roundoff_j=old.birth_operator_energy_roundoff_j,
        displacement_m=actual.displacement_m,
        coefficients_m=actual.coefficients_m, initial_energy_j=old.initial_energy_j,
        elastic_energy_j=actual.elastic_energy_j, mechanical_work_j=work,
        viscous_dissipation_j=loss, thermal_work_j=old.thermal_work_j,
        damage_energy_change_j=old.damage_energy_change_j,
        remelt_energy_removed_j=old.remelt_energy_removed_j,
        balance_residual_j=actual.elastic_energy_j-expected, journal=old.journal+(step,))


def enriched_memory_to_dict(state):
    if type(state) is not EnrichedMemoryState or state.binding_sha256 != _digest(_state_body(state)):
        raise EnrichedMemoryRefusal("Private archive state binding mismatch")
    body = dict(format=FORMAT, scope=SCOPE, state=_state_body(state),
        physical_release_admitted=False, funding_admitted=False,
        new_Genesis_time_myr=0., new_material_birth_count=0)
    return {**body, "packet_sha256": _digest(body)}


def enriched_memory_from_dict(raw, p, trusted_birth_parent_state, loading_factory=None):
    """Replay a private journal from its trusted unchanged Genesis birth.

    A callback ``loading_factory(state, after_time_myr)`` must obtain every
    replayed VolumeLoading from the original material/constitutive context.
    A newly resealed archive alone cannot authorize replacement memory.
    """
    names = {"format", "scope", "state", "physical_release_admitted", "funding_admitted",
             "new_Genesis_time_myr", "new_material_birth_count", "packet_sha256"}
    if type(raw) is not dict or set(raw) != names:
        raise EnrichedMemoryRefusal("Private archive fields mismatch")
    body = {k: v for k, v in raw.items() if k != "packet_sha256"}
    if (raw["format"] != FORMAT or raw["scope"] != SCOPE or raw["packet_sha256"] != _digest(body)
            or raw["physical_release_admitted"] is not False or raw["funding_admitted"] is not False
            or raw["new_Genesis_time_myr"] != 0 or raw["new_material_birth_count"] != 0):
        raise EnrichedMemoryRefusal("Private archive version, seal or admission scope mismatch")
    state = initial_enriched_memory(p, trusted_birth_parent_state)
    saved = raw["state"]
    if type(saved) is not dict or set(saved) != set(_state_body(state)) or type(saved["journal"]) is not list:
        raise EnrichedMemoryRefusal("Private archive state schema mismatch")
    for item in saved["journal"]:
        if type(item) is not dict or set(item) != {f.name for f in fields(EnrichedMemoryStep)}:
            raise EnrichedMemoryRefusal("Private archive journal fields mismatch")
        if loading_factory is None:
            raise EnrichedMemoryRefusal("Nonzero private restart requires source-bound loading replay")
        loading = loading_factory(state, item["after_time_myr"])
        if not np.array_equal(loading.ratio, item["ratio"]):
            raise EnrichedMemoryRefusal("Private restart Maxwell factors differ from source replay")
        trial = prepare_enriched_loading(p, state, loading)
        response = evaluate_enriched_loading(p, trial, item["parent_increment_m"], item["enrichment_increment_m"])
        state = accept_enriched_loading(p, trial, response)
    if enriched_memory_to_dict(state) != raw:
        raise EnrichedMemoryRefusal("Private restart state differs from complete birth-journal replay")
    return state
