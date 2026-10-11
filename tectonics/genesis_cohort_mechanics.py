"""Prospective, parallel-layer Maxwell memory of Genesis birth cohorts.

Only the activated load-bearing membrane is represented. Before that gate,
elastic strain and activation geometry are unavailable; this law never fills
the unmodelled history of isolated rafts with zero strain. At the first gate
the accepted endpoint temperature and actual old chart define stress-free
activation. Subsequent births activate at solidus in the thermal-first chart;
their endpoint cooling is an instantaneous strain increment. Existing cohorts
use an exact constant-driving Maxwell step with endpoint local viscosity.

The constitutive work identity covers this plane-stress membrane, including
its signed damage energy change and removed elastic energy on remelting. It
does not close thermal, mantle, contact, or friction reservoirs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math

import numpy as np

from .genesis_material import rotate_tensor
from .genesis import SECONDS_PER_MYR


FORMAT = "genesis-cohort-mechanics-0.1"
ENERGY_SCOPE = "activated_loadbearing_plane_stress_membrane"
GATE_ACTIVATION = "whole_membrane_gate_endpoint_temperature"
BIRTH_ACTIVATION = "solidus_thermal_first_endpoint_cooling"
_LEDGERS = ("elastic_energy_j", "mechanical_work_j", "thermal_work_j",
            "viscous_dissipation_j", "plastic_work_j", "damage_energy_change_j",
            "remelt_energy_removed_j", "balance_residual_j")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _finite(value, name, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"Cohort mechanics {name} must be finite real")
    value = float(value)
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f"Cohort mechanics {name} is outside its finite range")
    return value


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"Cohort mechanics {name} must be a nonnegative integer")
    return int(value)


def _tensor_tuple(value, shape, name):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"Cohort mechanics {name} shape/finite mismatch")
    return tuple(a.tolist()) if len(shape) == 1 else tuple(tuple(row) for row in a.tolist())


def _pin(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Cohort mechanics source identity must be SHA256")
    return value


@dataclass(frozen=True, slots=True)
class MechanicalCohort:
    id: int
    face_index: int
    layer_index: int
    q_lo: float
    live_q_hi: float
    temperature_k: float
    elastic_strain: tuple | None
    activation_time_myr: float | None
    activation_temperature_k: float | None
    activation_closure: str | None
    activation_metric_m2: tuple | None
    activation_frame_xyz: tuple | None

    def __post_init__(self):
        for name in ("id", "face_index", "layer_index"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if self.id == 0:
            raise ValueError("Cohort mechanics IDs start at one")
        for name in ("q_lo", "live_q_hi", "temperature_k"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if not 0 <= self.q_lo < self.live_q_hi <= 1 or self.temperature_k <= 0:
            raise ValueError("Cohort mechanics invalid live interval/temperature")
        activation = (self.activation_time_myr, self.activation_temperature_k,
                      self.activation_closure, self.activation_metric_m2, self.activation_frame_xyz)
        if self.elastic_strain is None:
            if any(v is not None for v in activation):
                raise ValueError("Unactivated cohort has no invented activation")
        else:
            object.__setattr__(self, "elastic_strain", _tensor_tuple(self.elastic_strain, (3,), "elastic_strain"))
            for name in ("activation_time_myr", "activation_temperature_k"):
                object.__setattr__(self, name, _finite(getattr(self, name), name, True))
            if self.activation_temperature_k <= 0 or self.activation_closure not in (GATE_ACTIVATION, BIRTH_ACTIVATION):
                raise ValueError("Cohort mechanics activation closure/temperature mismatch")
            metric = _tensor_tuple(self.activation_metric_m2, (2, 2), "activation_metric_m2")
            frame = _tensor_tuple(self.activation_frame_xyz, (3, 3), "activation_frame_xyz")
            g, r = np.asarray(metric), np.asarray(frame)
            if (not np.array_equal(g, g.T) or np.min(np.linalg.eigvalsh(g)) <= 0
                    or not np.allclose(r.T@r, np.eye(3), rtol=0, atol=2e-12)
                    or not math.isclose(float(np.linalg.det(r)), 1., abs_tol=2e-12)):
                raise ValueError("Cohort mechanics activation chart is invalid")
            object.__setattr__(self, "activation_metric_m2", metric)
            object.__setattr__(self, "activation_frame_xyz", frame)


@dataclass(frozen=True, slots=True)
class CohortMechanicsState:
    source_identity: str
    time_myr: float
    accepted_intervals: int
    established: bool
    cohorts: tuple = ()
    elastic_energy_j: float = 0.
    mechanical_work_j: float = 0.
    thermal_work_j: float = 0.
    viscous_dissipation_j: float = 0.
    plastic_work_j: float = 0.
    damage_energy_change_j: float = 0.
    remelt_energy_removed_j: float = 0.
    balance_residual_j: float = 0.

    def __post_init__(self):
        _pin(self.source_identity)
        object.__setattr__(self, "time_myr", _finite(self.time_myr, "time_myr", True))
        object.__setattr__(self, "accepted_intervals", _integer(self.accepted_intervals, "accepted_intervals"))
        if type(self.established) is not bool:
            raise ValueError("Cohort mechanics established must be boolean")
        object.__setattr__(self, "cohorts", tuple(self.cohorts))
        if any(type(c) is not MechanicalCohort for c in self.cohorts):
            raise ValueError("Cohort mechanics records have incompatible types")
        ids = tuple(c.id for c in self.cohorts)
        if tuple(sorted(set(ids))) != ids:
            raise ValueError("Cohort mechanics IDs must be ordered and unique")
        for c in self.cohorts:
            if (c.elastic_strain is not None) != self.established:
                raise ValueError("Cohort mechanics activation does not match membrane gate")
            if c.activation_time_myr is not None and c.activation_time_myr > self.time_myr:
                raise ValueError("Cohort mechanics activation lies after current time")
        for name in _LEDGERS:
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.elastic_energy_j < 0 or self.viscous_dissipation_j < 0 or self.remelt_energy_removed_j < 0:
            raise ValueError("Cohort mechanics energy/dissipation cannot be negative")
        if not self.established and any(getattr(self, name) != 0 for name in _LEDGERS):
            raise ValueError("Unactivated membrane cannot carry calculated mechanical energy")

    def __deepcopy__(self, memo):
        return self

    @property
    def records(self):
        return self.cohorts


def _live(history):
    return tuple(c for c in history.cohorts if c.live_q_hi > c.q_lo)


def initial_cohort_mechanics(history):
    if history.time_myr != 0 or history.accepted_intervals or history.cohorts or any(history.front_fraction):
        raise ValueError("Cohort mechanics must begin at molten birth; replay missing history")
    return CohortMechanicsState(history.source_identity, 0., 0, False)


def validate_cohort_mechanics(state, history, established=None, *, parameters=None, damage=None):
    if type(state) is not CohortMechanicsState:
        raise ValueError("Missing cohort mechanics; replay from molten formation")
    if (state.source_identity != history.source_identity or state.time_myr != history.time_myr
            or state.accepted_intervals != history.accepted_intervals
            or (established is not None and state.established != established)):
        raise ValueError("Cohort mechanics source/clock/gate mismatch")
    live = _live(history)
    if len(live) != len(state.cohorts):
        raise ValueError("Cohort mechanics live coverage mismatch")
    for record, birth in zip(state.cohorts, live):
        if (record.id, record.face_index, record.layer_index, record.q_lo, record.live_q_hi) != (
                birth.id, birth.face_index, birth.layer_index, birth.q_lo, birth.live_q_hi):
            raise ValueError("Cohort mechanics ID/material interval mismatch")
        if record.activation_time_myr is not None:
            if record.activation_time_myr < birth.birth_after_time_myr:
                raise ValueError("Cohort mechanical activation predates accepted birth")
            if record.activation_closure == BIRTH_ACTIVATION and (
                    record.activation_time_myr != birth.birth_after_time_myr
                    or record.activation_temperature_k != birth.birth_temperature_k
                    or record.activation_metric_m2 != birth.birth_metric_m2
                    or record.activation_frame_xyz != birth.birth_frame_xyz):
                raise ValueError("Cohort mechanics birth activation disagrees with source")
    if (parameters is None) != (damage is None):
        raise ValueError("Cohort mechanics energy validation needs parameters and damage")
    if parameters is not None:
        energy = cohort_mechanics_energy(state, history, damage, parameters)
        if not math.isclose(energy, state.elastic_energy_j, rel_tol=3e-12, abs_tol=1e-10):
            raise ValueError("Cohort mechanics stored elastic energy mismatch")
    expected = (state.mechanical_work_j-state.plastic_work_j+state.thermal_work_j
                -state.viscous_dissipation_j+state.damage_energy_change_j-state.remelt_energy_removed_j)
    residual = state.elastic_energy_j-expected
    scale = max(abs(state.elastic_energy_j), abs(expected),
                abs(state.mechanical_work_j), abs(state.thermal_work_j), 1.)
    if abs(residual-state.balance_residual_j) > 5e-12*scale or abs(residual) > 5e-12*scale:
        raise ValueError("Cohort mechanics cumulative work identity mismatch")
    return state


def cohort_mechanics_to_dict(state):
    body = {"format": FORMAT, "energy_scope": ENERGY_SCOPE, "state": asdict(state)}
    return {**body, "packet_sha256": _sha(body)}


def cohort_mechanics_from_dict(raw):
    if type(raw) is not dict or set(raw) != {"format", "energy_scope", "state", "packet_sha256"}:
        raise ValueError("Malformed cohort mechanics archive fields")
    body = {k: raw[k] for k in ("format", "energy_scope", "state")}
    if raw["format"] != FORMAT or raw["energy_scope"] != ENERGY_SCOPE or raw["packet_sha256"] != _sha(body):
        raise ValueError("Cohort mechanics version/seal mismatch")
    values = raw["state"]
    if type(values) is not dict or set(values) != {f.name for f in fields(CohortMechanicsState)}:
        raise ValueError("Cohort mechanics state field mismatch")
    records = values["cohorts"]
    if not isinstance(records, (tuple, list)):
        raise ValueError("Cohort mechanics records must be a sequence")
    names = {f.name for f in fields(MechanicalCohort)}
    if any(type(r) is not dict or set(r) != names for r in records):
        raise ValueError("Cohort mechanics record field mismatch")
    return CohortMechanicsState(**{**values, "cohorts": tuple(MechanicalCohort(**r) for r in records)})


def cohort_temperature(history, column_temperature, surface_temperature, mantle_temperature):
    """Exact mean of the existing piecewise-linear profile on live q intervals."""
    mass = np.asarray(history.layer_mass_kg, float)
    temperature = np.asarray(column_temperature, float)
    n, layers = mass.shape
    if temperature.shape != mass.shape or not np.isfinite(temperature).all() or np.any(temperature <= 0):
        raise ValueError("Cohort mechanics conductive temperature shape/range mismatch")
    surface = np.broadcast_to(np.asarray(surface_temperature, float), (n,))
    mantle = np.broadcast_to(np.asarray(mantle_temperature, float), (n,))
    profile = np.column_stack((surface, temperature, mantle))
    if not np.isfinite(profile).all() or np.any(profile <= 0):
        raise ValueError("Cohort mechanics boundary temperature range mismatch")
    bounds = np.column_stack((np.zeros(n), np.cumsum(mass, axis=1)/mass.sum(axis=1)[:, None]))
    q = np.column_stack((np.zeros(n), .5*(bounds[:, :-1]+bounds[:, 1:]), np.ones(n)))
    span = np.diff(q, axis=1)
    primitive = np.column_stack((np.zeros(n), np.cumsum(.5*(profile[:, :-1]+profile[:, 1:])*span, axis=1)))
    live = _live(history)
    if not live:
        return np.empty(0)
    face = np.fromiter((c.face_index for c in live), dtype=int)
    lo = np.fromiter((c.q_lo for c in live), dtype=float)
    hi = np.fromiter((c.live_q_hi for c in live), dtype=float)

    def at(x):
        segment = np.clip(np.sum(x[:, None] >= q[face], axis=1)-1, 0, layers)
        dx = x-q[face, segment]
        slope = (profile[face, segment+1]-profile[face, segment])/span[face, segment]
        return primitive[face, segment]+profile[face, segment]*dx+.5*slope*dx**2

    return (at(hi)-at(lo))/(hi-lo)


def _d(parameters):
    nu = parameters.poisson_ratio
    return np.array(((1., nu, 0.), (nu, 1., 0.), (0., 0., (1-nu)/2)))/(1-nu**2)


def _damage_factor(damage, parameters, n):
    damage = np.asarray(damage, float)
    if damage.shape != (n,) or not np.isfinite(damage).all() or np.any((damage < 0)|(damage > 1)):
        raise ValueError("Cohort mechanics damage shape/range mismatch")
    return parameters.residual_stiffness+(1-parameters.residual_stiffness)*(1-damage)**2


def _volumes(history, parameters):
    mass = np.asarray(history.layer_mass_kg).sum(axis=1)
    live = _live(history)
    return np.array([(c.live_q_hi-c.q_lo)*mass[c.face_index]/parameters.density_kg_m3 for c in live])


def _mean(values, face, volume, n):
    total = np.bincount(face, weights=volume, minlength=n)
    sums = np.zeros((n, values.shape[1]))
    np.add.at(sums, face, volume[:, None]*values)
    return np.divide(sums, total[:, None], out=np.zeros_like(sums), where=total[:, None] > 0)


def cohort_mechanics_mean(state, history):
    validate_cohort_mechanics(state, history)
    n = len(history.front_fraction)
    live = _live(history)
    face = np.array([c.face_index for c in live], int)
    mass = np.asarray(history.layer_mass_kg).sum(axis=1)
    weights = np.array([(c.live_q_hi-c.q_lo)*mass[c.face_index] for c in live])
    elastic = np.array([c.elastic_strain if c.elastic_strain is not None else (0., 0., 0.) for c in state.cohorts]).reshape((-1, 3))
    return _mean(elastic, face, weights, n)


def cohort_mechanics_energy(state, history, damage, parameters):
    n = len(history.front_fraction)
    factors = _damage_factor(damage, parameters, n)
    live = _live(history)
    face = np.array([c.face_index for c in live], int)
    elastic = np.array([c.elastic_strain if c.elastic_strain is not None else (0., 0., 0.) for c in state.cohorts]).reshape((-1, 3))
    return float(.5*parameters.young_modulus_pa*np.sum(_volumes(history, parameters)*factors[face]*np.einsum("ci,ij,cj->c", elastic, _d(parameters), elastic)))


def cohort_mechanics_max_strain(state):
    elastic = np.array([c.elastic_strain for c in state.cohorts if c.elastic_strain is not None]).reshape((-1, 3))
    xx, yy, gamma = elastic.T
    mean, radius = .5*(xx+yy), np.hypot(.5*(xx-yy), .5*gamma)
    return float(np.max(np.maximum(np.abs(mean+radius), np.abs(mean-radius)), initial=0.))


@dataclass(frozen=True, slots=True)
class CohortLoading:
    before_state: CohortMechanicsState
    history: object
    parameters: object
    face_index: np.ndarray
    volume_m3: np.ndarray
    old_elastic: np.ndarray
    thermal_increment: np.ndarray
    memory: np.ndarray
    beta: np.ndarray
    ratio: np.ndarray
    mean_memory: np.ndarray
    effective_b: np.ndarray
    temperature_k: np.ndarray
    records: tuple
    before_damage_factor: np.ndarray
    old_total_energy_j: float
    remelt_energy_removed_j: float
    established: bool

    @property
    def maxwell_r(self):
        return np.exp(-self.ratio)

    @property
    def relaxed_elastic(self):
        return self.maxwell_r[:, None]*self.old_elastic


@dataclass(frozen=True, slots=True)
class CohortResponse:
    elastic: np.ndarray
    mean_elastic: np.ndarray
    stress: np.ndarray
    effective_b: np.ndarray
    tangent: np.ndarray
    energy_j: float
    mechanical_work_j: float
    thermal_work_j: float
    viscous_dissipation_j: float
    plastic_work_j: float
    damage_energy_change_j: float
    remelt_energy_removed_j: float
    balance_residual_j: float
    max_elastic_strain: float


def _activation_chart(mesh, radius):
    from .genesis_bulk_birth import _geometry
    _, _, metric, frame = _geometry(mesh, radius)
    return metric, frame


def prepare_cohort_loading(state, before_history, after_history, *,
                          before_column_temperature, after_column_temperature,
                          before_surface_temperature_k, after_surface_temperature_k,
                          before_mantle_temperature_k, after_mantle_temperature_k,
                          before_established, after_established, parameters,
                          before_damage, activation_mesh=None, activation_radius_km=None):
    validate_cohort_mechanics(state, before_history, before_established,
                             parameters=parameters, damage=before_damage)
    source_temperature = cohort_temperature(before_history, before_column_temperature,
        before_surface_temperature_k, before_mantle_temperature_k)
    if not np.allclose([c.temperature_k for c in state.cohorts], source_temperature,
                       rtol=2e-12, atol=1e-8):
        raise ValueError("Cohort mechanics stored temperature disagrees with conductive source")
    if (before_history.source_identity != after_history.source_identity
            or after_history.accepted_intervals != before_history.accepted_intervals+1
            or after_history.time_myr <= before_history.time_myr):
        raise ValueError("Cohort mechanics trial must follow one accepted birth interval")
    if type(after_established) is not bool or (before_established and not after_established):
        raise ValueError("Cohort mechanics cannot erase established activation history")
    live = _live(after_history)
    n = len(after_history.front_fraction)
    face = np.array([c.face_index for c in live], dtype=int)
    volumes = _volumes(after_history, parameters)
    if not np.isfinite(volumes).all() or np.any(volumes <= 0):
        raise ValueError("Cohort mechanics material volumes invalid")
    old_temperature = cohort_temperature(after_history, before_column_temperature,
                                        before_surface_temperature_k, before_mantle_temperature_k)
    temperature = cohort_temperature(after_history, after_column_temperature,
                                    after_surface_temperature_k, after_mantle_temperature_k)
    count = len(live)
    old_elastic = np.zeros((count, 3))
    thermal = np.zeros((count, 3))
    beta = np.ones(count)
    ratio = np.zeros(count)
    records = []
    existing = {c.id: c for c in state.cohorts}
    gate = after_established and not before_established
    if gate:
        if activation_mesh is None or activation_radius_km is None:
            raise ValueError("Cohort gate needs actual old-geometry activation chart")
        gate_metric, gate_frame = _activation_chart(activation_mesh, activation_radius_km)
    dt_s = (after_history.time_myr-before_history.time_myr)*SECONDS_PER_MYR
    if after_established:
        eta = np.clip(parameters.viscosity_reference_pa_s*np.exp(np.clip(
            parameters.activation_energy_j_mol/8.314462618*(1/temperature-1/parameters.viscosity_reference_temperature_k), -60, 60)),
            parameters.viscosity_min_pa_s, parameters.viscosity_max_pa_s)
        if not np.isfinite(eta).all() or parameters.young_modulus_pa <= 0 or parameters.density_kg_m3 <= 0:
            raise ValueError("Cohort mechanics constitutive parameters invalid")
        ratio[:] = dt_s*parameters.young_modulus_pa/eta
    for i, birth in enumerate(live):
        old = existing.get(birth.id)
        if old is not None and birth.live_q_hi > old.live_q_hi:
            raise ValueError("Cohort mechanics cannot revive retired material")
        if not after_established:
            record = MechanicalCohort(birth.id, birth.face_index, birth.layer_index,
                birth.q_lo, birth.live_q_hi, temperature[i], None, None, None, None, None, None)
        elif gate:
            ratio[i] = 0.
            record = MechanicalCohort(birth.id, birth.face_index, birth.layer_index,
                birth.q_lo, birth.live_q_hi, temperature[i], (0., 0., 0.), after_history.time_myr,
                temperature[i], GATE_ACTIVATION, tuple(map(tuple, gate_metric[birth.face_index])),
                tuple(map(tuple, gate_frame[birth.face_index])))
        elif old is None:
            if birth.birth_after_time_myr != after_history.time_myr:
                raise ValueError("Missing historical cohort memory; formation replay required")
            ratio[i] = 0.
            thermal[i, :2] = parameters.linear_expansion_per_k*(temperature[i]-birth.birth_temperature_k)
            record = MechanicalCohort(birth.id, birth.face_index, birth.layer_index,
                birth.q_lo, birth.live_q_hi, temperature[i], (0., 0., 0.), after_history.time_myr,
                birth.birth_temperature_k, BIRTH_ACTIVATION, birth.birth_metric_m2, birth.birth_frame_xyz)
        else:
            old_elastic[i] = old.elastic_strain
            thermal[i, :2] = parameters.linear_expansion_per_k*(temperature[i]-old_temperature[i])
            record = MechanicalCohort(birth.id, birth.face_index, birth.layer_index,
                birth.q_lo, birth.live_q_hi, temperature[i], old.elastic_strain, old.activation_time_myr,
                old.activation_temperature_k, old.activation_closure, old.activation_metric_m2, old.activation_frame_xyz)
        records.append(record)
    r = np.exp(-ratio)
    beta = np.divide(-np.expm1(-ratio), ratio, out=np.ones_like(ratio), where=ratio != 0)
    memory = r[:, None]*old_elastic-beta[:, None]*thermal
    if not after_established:
        memory[:] = 0.
    factors = _damage_factor(before_damage, parameters, n)
    old_energy = cohort_mechanics_energy(state, before_history, before_damage, parameters)
    # Count only actually retired material. Subtracting two large elastic
    # totals would label reduction-order roundoff as physical remelt work.
    next_hi = {c.id: c.live_q_hi for c in live}
    column_mass = np.asarray(before_history.layer_mass_kg).sum(axis=1)
    retired = [c for c in state.cohorts if c.elastic_strain is not None
               and next_hi.get(c.id, c.q_lo) < c.live_q_hi]
    if retired:
        retired_face = np.array([c.face_index for c in retired], int)
        retired_volume = np.array([(c.live_q_hi-next_hi.get(c.id, c.q_lo))*column_mass[c.face_index]
            /parameters.density_kg_m3 for c in retired])
        retired_elastic = np.array([c.elastic_strain for c in retired])
        removed = float(.5*parameters.young_modulus_pa*np.sum(retired_volume*factors[retired_face]
            *np.einsum("ci,ij,cj->c", retired_elastic, _d(parameters), retired_elastic)))
    else:
        removed = 0.
    mean_memory = _mean(memory, face, volumes, n)
    effective = _mean(beta[:, None], face, volumes, n)[:, 0]
    return CohortLoading(state, after_history, parameters, face, volumes, old_elastic,
        thermal, memory, beta, ratio, mean_memory, effective, temperature, tuple(records),
        factors, old_energy, removed, after_established)


def evaluate_cohort_loading(loading, increment, rotation, damage, *, plastic_increment=None):
    p, history = loading.parameters, loading.history
    n, face = len(history.front_fraction), loading.face_index
    inc = np.asarray(increment, float)
    rotation = np.broadcast_to(np.eye(2), (n, 2, 2)).copy() if rotation is None else np.asarray(rotation, float)
    plastic = np.zeros((n, 3)) if plastic_increment is None else np.asarray(plastic_increment, float)
    if inc.shape != (n, 3) or plastic.shape != (n, 3) or not np.isfinite(inc).all() or not np.isfinite(plastic).all():
        raise ValueError("Cohort mechanics increment shape/finite mismatch")
    if rotation.shape != (n, 2, 2):
        raise ValueError("Cohort mechanics rotation shape mismatch")
    factors = _damage_factor(damage, p, n)
    driver = inc[face]-plastic[face]-loading.thermal_increment
    end_old_frame = loading.memory+loading.beta[:, None]*(inc[face]-plastic[face])
    if not loading.established:
        if np.any(inc) or np.any(plastic):
            raise ValueError("Unactivated rafts have no membrane increment")
        end_old_frame[:] = 0.
    elastic = rotate_tensor(end_old_frame, rotation[face], engineering=True)
    d = _d(p)
    modulus = p.young_modulus_pa*factors[face]
    stress_cohort = (elastic@d.T)*modulus[:, None]
    mean = _mean(elastic, face, loading.volume_m3, n)
    stress = _mean(stress_cohort, face, loading.volume_m3, n)
    tangent = (p.young_modulus_pa*factors*loading.effective_b)[:, None, None]*d
    energy = float(.5*np.sum(loading.volume_m3*modulus*np.einsum("ci,ij,cj->c", elastic, d, elastic)))
    # Stable exact integrals: e(z)=r(z)e0 + driver*(1-r(z))/x.
    x = loading.ratio
    beta = loading.beta
    average_r = beta
    average_r2 = np.divide(-np.expm1(-2*x), 2*x, out=np.ones_like(x), where=x != 0)
    c = np.divide(1-beta, x, out=np.full_like(x, .5), where=x != 0)
    small = x < 1e-3
    c[small] = .5-x[small]/6+x[small]**2/24-x[small]**3/120
    average_elastic = average_r[:, None]*loading.old_elastic+c[:, None]*driver
    average_stress = (average_elastic@d.T)*modulus[:, None]
    work = float(np.sum(loading.volume_m3*np.einsum("ci,ci->c", inc[face], average_stress)))
    plastic_work = float(np.sum(loading.volume_m3*np.einsum("ci,ci->c", plastic[face], average_stress)))
    thermal_work = float(-np.sum(loading.volume_m3*np.einsum("ci,ci->c", loading.thermal_increment, average_stress)))
    # Dvisc follows the exact positive integral, avoiding cancellation at x=0.
    aa = x*average_r2
    ab = 2*(average_r-average_r2)
    bb = np.divide(1-2*average_r+average_r2, x, out=np.zeros_like(x), where=x != 0)
    bb[small] = x[small]/3-x[small]**2/4+7*x[small]**3/60-x[small]**4/24
    ab[small] = x[small]-x[small]**2+7*x[small]**3/12-x[small]**4/4
    loss_density = (aa*np.einsum("ci,ij,cj->c", loading.old_elastic, d, loading.old_elastic)
        +ab*np.einsum("ci,ij,cj->c", loading.old_elastic, d, driver)
        +bb*np.einsum("ci,ij,cj->c", driver, d, driver))*modulus
    loss = float(np.sum(loading.volume_m3*loss_density))
    if loss < -max(energy, loading.old_total_energy_j, 1.)*5e-13:
        raise ValueError("Cohort Maxwell dissipation is negative")
    loss = max(loss, 0.)
    damage_change = float(.5*p.young_modulus_pa*np.sum(loading.volume_m3*(factors-loading.before_damage_factor)[face]*np.einsum("ci,ij,cj->c", loading.old_elastic, d, loading.old_elastic)))
    residual = energy-loading.old_total_energy_j-(work-plastic_work+thermal_work-loss+damage_change-loading.remelt_energy_removed_j)
    xx, yy, gamma = elastic.T
    principal = .5*(xx+yy)
    radius = np.hypot(.5*(xx-yy), .5*gamma)
    maximum = float(np.max(np.maximum(np.abs(principal+radius), np.abs(principal-radius)), initial=0.))
    return CohortResponse(elastic, mean, stress, loading.effective_b, tangent, energy,
        work, thermal_work, loss, plastic_work, damage_change, loading.remelt_energy_removed_j,
        residual, maximum)


def accept_cohort_loading(loading, response):
    if response.elastic.shape != (len(loading.records), 3):
        raise ValueError("Cohort accepted response coverage mismatch")
    from dataclasses import replace
    records = tuple(replace(record, elastic_strain=tuple(response.elastic[i])) if loading.established else record
                    for i, record in enumerate(loading.records))
    before = loading.before_state
    values = {name: getattr(before, name)+getattr(response, name) for name in _LEDGERS
              if name not in ("elastic_energy_j", "balance_residual_j")}
    values["elastic_energy_j"] = response.energy_j
    expected = (values["mechanical_work_j"]-values["plastic_work_j"]+values["thermal_work_j"]
        -values["viscous_dissipation_j"]+values["damage_energy_change_j"]-values["remelt_energy_removed_j"])
    values["balance_residual_j"] = response.energy_j-expected
    state = CohortMechanicsState(loading.history.source_identity, loading.history.time_myr,
        loading.history.accepted_intervals, loading.established, records, **values)
    return validate_cohort_mechanics(state, loading.history, loading.established)
