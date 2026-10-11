"""Prospective heat/orbit and conductive births of actual Genesis material.

The embedded Fault source is replayed, not replaced by a mature initial
condition.  Global heat/orbit remains the inherited one-way model; conductive
columns keep their own boundary ledger.  Their future flux uses actual old
deformed top chord areas with the inherited constant-density column closure.
This does not claim a combined 3-D mechanical/thermal energy balance.

Bulk birth charts stay in the authentic fixed spherical reference. Actual
prospective activation triangles are returned separately, so a non-spherical
deformed surface is never passed off as a SphereMesh or historical geometry.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
import base64
import hashlib
import io
import json
import math
from types import SimpleNamespace

import numpy as np

from .genesis import GenesisState, GenesisParameters, SECONDS_PER_MYR, temperatures
from .genesis_bulk_birth import (advance_bulk_birth, bulk_birth_from_dict,
                                 validate_bulk_birth, validate_bulk_birth_continuation)
from .genesis_bulk_export import validate_genesis_bulk_material
from .genesis_cohort_mechanics import cohort_temperature
from .genesis_faults import load_fault_checkpoint
from .genesis_mobile import MobileModel
from .genesis_onset import advance_orbit_thermal
from .genesis_shell import ShellParameters, rock_temperature
from .genesis_tides import TidalParameters, TidalOrbitState
from .genesis_volume_geometry import (build_updated_genesis_volume_geometry,
                                     validate_genesis_volume_geometry, _updated_geometry, _chart, _scale)
from .genesis_volume_loading import RadialMoments
from .genesis_volume_operator import conical_radial_moments


FORMAT = "genesis-volume-thermal-0.1"
THERMAL_CLOSURE = "inherited_global_heat_orbit_separate_columns_actual_old_top_chord_area_constant_density"
BIRTH_CHART_SCOPE = "authentic_fixed_spherical_reference_not_actual_deformed_activation"
_CONTEXTS = OrderedDict()
_TRUSTED = OrderedDict()
_GEOMETRIES = OrderedDict()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _immutable(value):
    a = np.asarray(value, float)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"Volume thermal {name} requires a finite positive real")
    v = float(value)
    if not math.isfinite(v) or v <= 0:
        raise ValueError(f"Volume thermal {name} must be finite and positive")
    return v


def _context(source_json):
    cached = _CONTEXTS.get(source_json)
    if cached is not None:
        _CONTEXTS.move_to_end(source_json)
        return cached
    packet = json.loads(source_json)
    model, state, global_state, orbit, metadata = load_fault_checkpoint(io.BytesIO(
        base64.b64decode(packet["source"]["checkpoint_base64"], validate=True)))
    history = bulk_birth_from_dict(packet["bulk_birth_history"])
    context = SimpleNamespace(packet=packet, model=model, state=state,
        global_state=global_state, orbit=orbit, metadata=metadata,
        history=history, mesh=model.mesh_for(state))
    _CONTEXTS[source_json] = context
    while len(_CONTEXTS) > 4:
        _CONTEXTS.popitem(last=False)
    return context


def _remember(state):
    _TRUSTED[id(state)] = state
    _TRUSTED.move_to_end(id(state))
    while len(_TRUSTED) > 4:
        _TRUSTED.popitem(last=False)
    return state


def _remember_geometry(geometry):
    _GEOMETRIES[id(geometry)] = geometry
    _GEOMETRIES.move_to_end(id(geometry))
    while len(_GEOMETRIES) > 4:
        _GEOMETRIES.popitem(last=False)
    return geometry


@dataclass(frozen=True, slots=True)
class VolumeThermalBoundaryParameters:
    thermal: GenesisParameters
    tides: TidalParameters


@dataclass(frozen=True, slots=True)
class VolumeThermalInterval:
    before_time_myr: float
    after_time_myr: float
    surface_vertices_m: tuple
    max_global_step_myr: float
    requested_after_time_myr: float

    @property
    def dt_myr(self):
        return self.after_time_myr-self.before_time_myr


@dataclass(frozen=True, slots=True)
class VolumeThermalState:
    source_packet_json: str
    source_checkpoint_sha256: str
    source_identity: str
    initial_time_myr: float
    time_myr: float
    source_accepted_intervals: int
    accepted_steps: int
    layer_mass_kg: np.ndarray
    column_enthalpy: np.ndarray
    damage: np.ndarray
    global_state_json: str
    orbit: TidalOrbitState
    initial_column_energy_j: float
    boundary_energy_j: float
    tidal_heat_received_j: float
    intervals: tuple = ()

    @property
    def global_state(self):
        # Fresh list/dict storage protects this immutable accepted snapshot.
        return GenesisState(**json.loads(self.global_state_json))

    @property
    def source_shell_parameters(self):
        return _context(self.source_packet_json).model.p

    @property
    def source_boundary_parameters(self):
        model = _context(self.source_packet_json).model
        return VolumeThermalBoundaryParameters(model.thermal, model.tides_p)

    @property
    def combined_heat_mechanical_energy_closed(self):
        return False

    @property
    def birth_chart_scope(self):
        return BIRTH_CHART_SCOPE

    def __deepcopy__(self, memo):
        return self


def initial_volume_thermal_state(packet):
    """Inherit every actual source thermal field and clock without resets."""
    source = validate_genesis_bulk_material(packet)
    source_json = _json(source)
    c = _context(source_json)
    state = VolumeThermalState(source_json, source["source"]["checkpoint_sha256"],
        c.history.source_identity, c.state.time_myr, c.state.time_myr,
        c.history.accepted_intervals, 0, _immutable(c.state.layer_mass_kg),
        _immutable(c.state.column_enthalpy), _immutable(c.state.damage),
        _json(asdict(c.global_state)), c.orbit, c.state.initial_column_energy_j,
        c.state.boundary_energy_j, c.state.tidal_heat_received_j)
    return _remember(state)


def _parameters(state, shell=None, boundary=None):
    c = _context(state.source_packet_json)
    if shell is not None and (type(shell) is not ShellParameters or asdict(shell) != asdict(c.model.p)):
        raise ValueError("Volume thermal shell parameters differ from the actual source")
    expected = VolumeThermalBoundaryParameters(c.model.thermal, c.model.tides_p)
    if boundary is not None:
        if type(boundary) is GenesisParameters:
            equal = asdict(boundary) == asdict(c.model.thermal)
        else:
            equal = type(boundary) is VolumeThermalBoundaryParameters and asdict(boundary) == asdict(expected)
        if not equal:
            raise ValueError("Volume thermal boundary parameters differ from the actual source")
    return c


class _PhysicalAreaMesh:
    """Only the finite-volume area interface; deliberately not a SphereMesh."""
    def __init__(self, vertices, faces):
        triangle = np.asarray(vertices, float)[np.asarray(faces, int)]
        cross = np.cross(triangle[:, 1]-triangle[:, 0], triangle[:, 2]-triangle[:, 0])
        self.area_m2 = .5*np.linalg.norm(cross, axis=1)
        if not np.isfinite(self.area_m2).all() or np.any(self.area_m2 <= 0):
            raise ValueError("Volume thermal top triangles have nonpositive physical area")
        # The caller owns finite-volume admission; inverted outward faces are
        # refused here as well instead of silently taking absolute area.
        if np.any(np.einsum("fi,fi->f", cross, triangle.mean(axis=1)) <= 0):
            raise ValueError("Volume thermal top triangles inverted their outward orientation")
        self.cell_count = len(triangle)

    def physical_cell_areas_km2(self, radius_km):
        return self.area_m2/1e6


def _check_state_history(state, history):
    if type(state) is not VolumeThermalState:
        raise ValueError("A genuine inherited VolumeThermalState is required")
    if _TRUSTED.get(id(state)) is not state:
        body = json.loads(_json(_body(state, True)))
        thermal_from_dict({**body, "sha256": hashlib.sha256(_json(body).encode()).hexdigest()})
        _remember(state)
    c = _parameters(state)
    if (state.source_identity != c.history.source_identity
            or state.source_checkpoint_sha256 != c.packet["source"]["checkpoint_sha256"]
            or state.initial_time_myr != c.state.time_myr
            or state.source_accepted_intervals != c.history.accepted_intervals
            or isinstance(state.accepted_steps, bool)
            or not isinstance(state.accepted_steps, int) or state.accepted_steps < 0
            or state.accepted_steps != len(state.intervals)):
        raise ValueError("Volume thermal source/counter binding mismatch")
    if state.time_myr != state.global_state.time_myr or state.time_myr != state.orbit.time_myr:
        raise ValueError("Volume material, global thermal and orbit clocks disagree")
    if not np.array_equal(state.layer_mass_kg, c.state.layer_mass_kg):
        raise ValueError("Volume thermal original material masses changed")
    if (state.column_enthalpy.shape != state.layer_mass_kg.shape
            or not np.isfinite(state.column_enthalpy).all() or np.any(state.column_enthalpy <= 0)):
        raise ValueError("Volume thermal conductive profile is invalid")
    if not np.array_equal(state.damage, c.state.damage):
        raise ValueError("The thermal adapter cannot replace source damage history")
    tm, ts = temperatures(np.asarray(state.global_state.energy), c.model.thermal)
    front, _, _ = c.model._phase(state.column_enthalpy, ts, tm)
    validate_bulk_birth(history, layer_mass_kg=state.layer_mass_kg,
        time_myr=state.time_myr, front_fraction=front, source_identity=state.source_identity)
    if history.accepted_intervals != state.source_accepted_intervals+state.accepted_steps:
        raise ValueError("Volume thermal accepted intervals disagree with material births")
    validate_bulk_birth_continuation(history, c.history)
    return c, front


def _advance_core(state, history, target, vertices, max_global_step):
    c, before_front = _check_state_history(state, history)
    surface = np.asarray(vertices, float)
    if surface.shape != (c.mesh.vertex_count, 3) or not np.isfinite(surface).all():
        raise ValueError("Volume thermal physical top coordinates lost original vertex coverage")
    max_step = _positive(max_global_step, "global time step")
    _positive(target-state.time_myr, "time increment")
    global_after, orbit_after, heating, rows = advance_orbit_thermal(state.global_state,
        state.orbit, c.model.thermal, c.model.tides_p, target, max_step)
    actual_dt = global_after.time_myr-state.time_myr
    _positive(actual_dt, "actual synchronized increment")
    tm0, ts0 = temperatures(np.asarray(state.global_state.energy), c.model.thermal)
    tm1, ts1 = temperatures(np.asarray(global_after.energy), c.model.thermal)
    area_mesh = _PhysicalAreaMesh(surface, c.mesh.faces)
    columns = SimpleNamespace(layer_mass_kg=state.layer_mass_kg,
        column_enthalpy=state.column_enthalpy, boundary_energy_j=state.boundary_energy_j,
        radius_km=c.state.radius_km)
    h, boundary = MobileModel._conduct(c.model, columns, area_mesh, actual_dt,
                                      ts0, ts1, tm0, tm1)
    after_front, _, after_profile = c.model._phase(h, ts1, tm1)
    after_history = advance_bulk_birth(history, before_front_fraction=before_front,
        after_front_fraction=after_front, before_time_myr=state.time_myr,
        after_time_myr=global_after.time_myr, mesh=c.mesh,
        radius_km=c.state.radius_km, solidus_k=c.model.thermal.solidus_k)
    interval = VolumeThermalInterval(state.time_myr, global_after.time_myr,
        tuple(tuple(row) for row in surface.tolist()), max_step, float(target))
    result = replace(state, time_myr=global_after.time_myr,
        accepted_steps=state.accepted_steps+1, column_enthalpy=_immutable(h),
        global_state_json=_json(asdict(global_after)), orbit=orbit_after,
        boundary_energy_j=boundary,
        tidal_heat_received_j=state.tidal_heat_received_j+heating*actual_dt*SECONDS_PER_MYR*c.model.thermal.area_m2,
        intervals=state.intervals+(interval,))
    _remember(result)
    before_profile = rock_temperature(state.column_enthalpy, c.model.thermal)
    before_temperature = cohort_temperature(after_history, before_profile, ts0, tm0)
    after_temperature = cohort_temperature(after_history, after_profile, ts1, tm1)
    return result, after_history, before_temperature, after_temperature, rows


def _moments(cohorts):
    volume, mean, variance = [], [], []
    for c in cohorts:
        m, v = conical_radial_moments(c.outer_scale, c.inner_scale)
        volume.append(c.volume_m3); mean.append(m); variance.append(v)
    return RadialMoments(tuple(volume), tuple(mean), tuple(variance))


def _removed_moments(before_geometry, after_geometry):
    after = {c.id: c for c in after_geometry.cohorts}
    volume, mean, variance = [], [], []
    for c in before_geometry.cohorts:
        following = after.get(c.id)
        qlo = c.q_lo if following is None else following.live_q_hi
        if qlo < c.q_lo or qlo > c.live_q_hi:
            raise ValueError("Volume thermal material retirement does not retain a prefix")
        if qlo == c.live_q_hi:
            volume.append(0.); mean.append(0.); variance.append(0.)
            continue
        _, column_volume, coefficient = _chart(c.top_triangle_m,
            before_geometry.column_mass_kg[c.face_index], before_geometry.density_kg_m3)
        m, v = conical_radial_moments(_scale(coefficient, qlo), _scale(coefficient, c.live_q_hi))
        volume.append(column_volume*(c.live_q_hi-qlo)); mean.append(m); variance.append(v)
    return RadialMoments(tuple(volume), tuple(mean), tuple(variance))


@dataclass(frozen=True, slots=True)
class VolumeThermalStep:
    state: VolumeThermalState
    history: object
    geometry: object
    before_temperature_k: np.ndarray
    after_temperature_k: np.ndarray
    birth_temperature_k: np.ndarray
    before_radial_moments: RadialMoments
    radial_moments: RadialMoments
    removed_radial_moments: RadialMoments
    dt_myr: float
    activation_top_triangles_m: np.ndarray
    global_rows: tuple
    newborn_cohort_ids: tuple


def prepare_volume_thermal_step(state, history, geometry, dt_myr,
                                source_shell_parameters=None,
                                source_boundary_parameters=None, *,
                                displacement_m=None, max_global_step_myr=.01):
    """Private thermal-first trial; accepted inputs are never mutated."""
    dt = _positive(dt_myr, "time increment")
    _parameters(state, source_shell_parameters, source_boundary_parameters)
    if _GEOMETRIES.get(id(geometry)) is not geometry:
        validate_genesis_volume_geometry(geometry)
        _remember_geometry(geometry)
    if (geometry.source_identity != state.source_identity
            or geometry.source_checkpoint_sha256 != state.source_checkpoint_sha256
            or geometry.time_myr != state.time_myr):
        raise ValueError("Volume thermal geometry/source clock binding mismatch")
    X = np.asarray(geometry.vertices_m, float)
    if displacement_m is None:
        surface = X
    else:
        q = np.asarray(displacement_m, float)
        if q.shape != (6*len(X),) or not np.isfinite(q).all():
            raise ValueError("Volume thermal motion needs every finite Cartesian u,p DOF")
        surface = X+q.reshape(-1, 6)[:, :3]
    result, after_history, before_t, after_t, rows = _advance_core(state, history,
        state.time_myr+dt, surface, max_global_step_myr)
    # The complete immutable parent was checked above; _advance_core checked
    # its history and constructed this exact local append/retirement. Avoid
    # replaying the growing entire formation again on ordinary endpoints.
    after_geometry = _remember_geometry(_updated_geometry(geometry, after_history))
    live = [c for c in after_history.cohorts if c.live_q_hi > c.q_lo]
    triangles = surface[np.asarray(geometry.faces, int)]
    activation = triangles[np.asarray([c.face_index for c in live], int)]
    return VolumeThermalStep(result, after_history, after_geometry,
        _immutable(before_t), _immutable(after_t),
        _immutable([c.birth_temperature_k for c in live]),
        _moments(geometry.cohorts), _moments(after_geometry.cohorts),
        _removed_moments(geometry, after_geometry), result.time_myr-state.time_myr,
        _immutable(activation), tuple(json.loads(_json(rows))),
        tuple(c.id for c in after_geometry.cohorts if c.id > max((old.id for old in geometry.cohorts), default=0)))


def _body(state, include_source):
    return {"format": FORMAT, "thermal_closure": THERMAL_CLOSURE,
        "birth_chart_scope": BIRTH_CHART_SCOPE,
        "source_packet": json.loads(state.source_packet_json) if include_source else None,
        "source_checkpoint_sha256": state.source_checkpoint_sha256,
        "source_identity": state.source_identity,
        "initial_time_myr": state.initial_time_myr, "time_myr": state.time_myr,
        "source_accepted_intervals": state.source_accepted_intervals,
        "accepted_steps": state.accepted_steps,
        "layer_mass_kg": state.layer_mass_kg.tolist(),
        "column_enthalpy": state.column_enthalpy.tolist(), "damage": state.damage.tolist(),
        "global_state": json.loads(state.global_state_json), "orbit": asdict(state.orbit),
        "initial_column_energy_j": state.initial_column_energy_j,
        "boundary_energy_j": state.boundary_energy_j,
        "tidal_heat_received_j": state.tidal_heat_received_j,
        "intervals": [asdict(interval) for interval in state.intervals]}


def thermal_to_dict(state, *, include_source=True):
    if type(state) is not VolumeThermalState:
        raise ValueError("Cannot archive missing volume thermal continuation")
    if _TRUSTED.get(id(state)) is not state:
        thermal_from_dict({**_body(state, True), "sha256": hashlib.sha256(_json(_body(state, True)).encode()).hexdigest()})
    body = json.loads(_json(_body(state, include_source)))
    return {**body, "sha256": hashlib.sha256(_json(body).encode()).hexdigest()}


def thermal_from_dict(raw, *, source_packet=None):
    """Replay every runtime thermal input from the original actual source."""
    if type(raw) is not dict or type(raw.get("sha256")) is not str:
        raise ValueError("Malformed volume thermal checkpoint")
    body = {key: value for key, value in raw.items() if key != "sha256"}
    if hashlib.sha256(_json(body).encode()).hexdigest() != raw["sha256"]:
        raise ValueError("Volume thermal checkpoint seal mismatch")
    packet = body.get("source_packet") if source_packet is None else source_packet
    if packet is None:
        raise ValueError("Volume thermal restart requires its actual embedded source")
    state = initial_volume_thermal_state(packet)
    if body.get("source_packet") is not None and _json(body["source_packet"]) != state.source_packet_json:
        raise ValueError("Volume thermal restart changed its independently supplied source")
    history = _context(state.source_packet_json).history
    records = body.get("intervals")
    if type(records) is not list:
        raise ValueError("Volume thermal restart lost its accepted interval inputs")
    try:
        for row in records:
            if type(row) is not dict or set(row) != {"before_time_myr", "after_time_myr", "surface_vertices_m", "max_global_step_myr", "requested_after_time_myr"}:
                raise ValueError("Volume thermal interval fields mismatch")
            if row["before_time_myr"] != state.time_myr:
                raise ValueError("Volume thermal restart lost its clock prefix")
            state, history, _, _, _ = _advance_core(state, history, row["requested_after_time_myr"],
                row["surface_vertices_m"], row["max_global_step_myr"])
        if _json(body) != _json(_body(state, body.get("source_packet") is not None)):
            raise ValueError("Volume thermal checkpoint differs from complete actual-source replay")
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError("Malformed volume thermal checkpoint") from exc
    return _remember(state)


volume_thermal_to_dict = thermal_to_dict
volume_thermal_from_dict = thermal_from_dict
