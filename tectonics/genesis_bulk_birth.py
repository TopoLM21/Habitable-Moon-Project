"""Birth catalogue of the existing Genesis material columns.

The conductive producer supplies its accepted, surface-connected solidus front.
The front is a column mass coordinate: q=0 at the surface and q=1 at the bottom.
Growth activates sub-layer intervals; retreat retires the deepest overlap. A
later re-freezing creates a new identity and never revives a retired record.

Birth is bracketed by accepted endpoints, without an invented interior date.
Geometry is the actual old mechanical geometry of the thermal-first split.
The solidus temperature is the boundary law, not a sampled parcel temperature.
No total strain, natural metric, prestress, or motion of isolated frozen rafts
is inferred here. Columns are the existing one-way conductive representation;
this catalogue creates neither material nor an additional thermal reservoir.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import math
from numbers import Integral, Real

import numpy as np


MODEL_VERSION = "genesis-bulk-birth-0.1"
FORMAT = MODEL_VERSION
FRONT_CLOSURE = "surface_connected_solidus_column_mass_coordinate"
TIME_CLOSURE = "accepted_endpoint_birth_bracket"
GEOMETRY_CLOSURE = "thermal_first_old_mechanical_geometry"
REMELT_CLOSURE = "retire_deepest_reference_mass_overlap"
_TRUSTED = []  # Bounded immutable-object identities; never serialized trust.


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _real(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"Bulk birth {name} must be finite real")
    try:
        value = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"Bulk birth {name} must be finite real") from exc
    if not math.isfinite(value) or (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(f"Bulk birth {name} is outside its finite range")
    return value


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"Bulk birth {name} must be an integer >= {minimum}")
    return int(value)


def _pin(value, name):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"Bulk birth {name} must be a SHA256")
    return value


def _vector(values, name, *, fractions=False):
    try:
        result = tuple(_real(v, name) for v in values)
    except TypeError as exc:
        raise ValueError(f"Bulk birth {name} must be a real vector") from exc
    if fractions and any(v < 0 or v > 1 for v in result):
        raise ValueError(f"Bulk birth {name} must lie in [0, 1]")
    return result


def _matrix(values, name, shape=None, *, positive=False):
    try:
        result = tuple(tuple(_real(v, name, positive=positive) for v in row) for row in values)
    except TypeError as exc:
        raise ValueError(f"Bulk birth {name} must be a real matrix") from exc
    if not result or not result[0] or any(len(row) != len(result[0]) for row in result):
        raise ValueError(f"Bulk birth {name} must be a nonempty rectangular matrix")
    if shape is not None and (len(result), len(result[0])) != shape:
        raise ValueError(f"Bulk birth {name} has an invalid shape")
    return result


def _geometry(mesh, radius_km):
    radius = _real(radius_km, "radius_km", positive=True)
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3
            or vertices.dtype.kind not in "fiu" or not np.isfinite(vertices).all()
            or not np.allclose(np.linalg.norm(vertices, axis=1), 1., rtol=0, atol=2e-12)
            or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or faces.dtype.kind not in "iu" or np.any(faces < 0) or np.any(faces >= len(vertices))):
        raise ValueError("Bulk birth requires finite unit vertices and valid ordered material faces")
    xyz = vertices[faces]
    first, other = xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0]
    normal = np.cross(first, other)
    lengths, normals = np.linalg.norm(first, axis=1), np.linalg.norm(normal, axis=1)
    if np.any(lengths <= 0) or np.any(normals <= 0):
        raise ValueError("Bulk birth geometry contains collapsed faces")
    if np.any(np.einsum("fi,fi->f", xyz[:, 0], normal) <= 0):
        raise ValueError("Bulk birth geometry must be outward oriented")
    e1, en = first/lengths[:, None], normal/normals[:, None]
    frame = np.stack((e1, np.cross(en, e1), en), axis=2)
    edges_m = np.stack((first, other), axis=2)*(1000*radius)
    metric = np.einsum("fki,fkj->fij", edges_m, edges_m)
    return vertices, faces, metric, frame


@dataclass(frozen=True, slots=True)
class BulkBirthCohort:
    id: int
    face_index: int
    layer_index: int
    q_lo: float
    q_hi: float
    live_q_hi: float
    birth_before_time_myr: float
    birth_after_time_myr: float
    birth_temperature_k: float
    birth_metric_m2: tuple
    birth_frame_xyz: tuple
    birth_radius_km: float

    def __post_init__(self):
        for name, minimum in (("id", 1), ("face_index", 0), ("layer_index", 0)):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum))
        for name in ("q_lo", "q_hi", "live_q_hi", "birth_before_time_myr", "birth_after_time_myr"):
            object.__setattr__(self, name, _real(getattr(self, name), name, nonnegative=True))
        for name in ("birth_temperature_k", "birth_radius_km"):
            object.__setattr__(self, name, _real(getattr(self, name), name, positive=True))
        if not 0 <= self.q_lo < self.q_hi <= 1 or not self.q_lo <= self.live_q_hi <= self.q_hi:
            raise ValueError("Bulk birth cohort has an invalid formed/live mass interval")
        if self.birth_after_time_myr <= self.birth_before_time_myr:
            raise ValueError("Bulk birth cohort needs an advancing accepted birth bracket")
        metric = _matrix(self.birth_metric_m2, "birth_metric_m2", (2, 2))
        frame = _matrix(self.birth_frame_xyz, "birth_frame_xyz", (3, 3))
        object.__setattr__(self, "birth_metric_m2", metric)
        object.__setattr__(self, "birth_frame_xyz", frame)
        g, basis = np.asarray(metric), np.asarray(frame)
        if (not np.array_equal(g, g.T) or g[0, 0] <= 0 or g[1, 1] <= 0
                or np.linalg.det(g) <= 0):
            raise ValueError("Bulk birth face metric must be symmetric positive definite")
        if (not np.allclose(basis.T@basis, np.eye(3), rtol=0, atol=2e-12)
                or not math.isclose(float(np.linalg.det(basis)), 1., rel_tol=0, abs_tol=2e-12)):
            raise ValueError("Bulk birth face frame must be proper orthonormal")


@dataclass(frozen=True, slots=True)
class BulkBirthInterval:
    before_time_myr: float
    after_time_myr: float
    before_front_fraction: tuple
    after_front_fraction: tuple

    def __post_init__(self):
        for name in ("before_time_myr", "after_time_myr"):
            object.__setattr__(self, name, _real(getattr(self, name), name, nonnegative=True))
        for name in ("before_front_fraction", "after_front_fraction"):
            object.__setattr__(self, name, _vector(getattr(self, name), name, fractions=True))
        if self.after_time_myr <= self.before_time_myr:
            raise ValueError("Bulk birth accepted interval must advance time")
        if not self.before_front_fraction or len(self.before_front_fraction) != len(self.after_front_fraction):
            raise ValueError("Bulk birth accepted front vectors must agree")


@dataclass(frozen=True, slots=True)
class BulkBirthHistory:
    time_myr: float
    front_fraction: tuple
    source_identity: str
    configuration_json: str
    initial_vertices: tuple
    initial_faces: tuple
    initial_radius_km: float
    solidus_k: float
    layer_mass_kg: tuple
    cohorts: tuple = ()
    intervals: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "time_myr", _real(self.time_myr, "time_myr", nonnegative=True))
        object.__setattr__(self, "front_fraction", _vector(self.front_fraction, "front_fraction", fractions=True))
        _pin(self.source_identity, "source_identity")
        for name in ("initial_radius_km", "solidus_k"):
            object.__setattr__(self, name, _real(getattr(self, name), name, positive=True))
        object.__setattr__(self, "initial_vertices", _matrix(self.initial_vertices, "initial_vertices"))
        try:
            faces = tuple(tuple(_integer(v, "initial face vertex") for v in row) for row in self.initial_faces)
        except TypeError as exc:
            raise ValueError("Bulk birth initial faces must be integer triples") from exc
        object.__setattr__(self, "initial_faces", faces)
        object.__setattr__(self, "layer_mass_kg", _matrix(self.layer_mass_kg, "layer_mass_kg", positive=True))
        for name, expected in (("cohorts", BulkBirthCohort), ("intervals", BulkBirthInterval)):
            object.__setattr__(self, name, tuple(getattr(self, name)))
            if any(type(record) is not expected for record in getattr(self, name)):
                raise ValueError(f"Bulk birth {name} contain incompatible records")
        if len(self.front_fraction) != len(self.layer_mass_kg) or len(faces) != len(self.layer_mass_kg):
            raise ValueError("Bulk birth material face counts disagree")

    def __deepcopy__(self, memo):
        return self  # All nested state is scalar, text, frozen records or tuples.

    @property
    def accepted_intervals(self):
        return len(self.intervals)

    @property
    def next_id(self):
        return len(self.cohorts)+1

    @property
    def column_mass_kg(self):
        return tuple(math.fsum(row) for row in self.layer_mass_kg)

    def _mass_per_face(self, live):
        rows = [[] for _ in self.layer_mass_kg]
        mass = self.column_mass_kg
        for cohort in self.cohorts:
            hi = cohort.live_q_hi if live else cohort.q_hi
            rows[cohort.face_index].append((hi-cohort.q_lo)*mass[cohort.face_index])
        return tuple(math.fsum(row) for row in rows)

    @property
    def formed_mass_per_face_kg(self):
        return self._mass_per_face(False)

    @property
    def live_mass_per_face_kg(self):
        return self._mass_per_face(True)

    @property
    def remelted_mass_per_face_kg(self):
        return tuple(a-b for a, b in zip(self.formed_mass_per_face_kg, self.live_mass_per_face_kg))

    @property
    def formed_mass_kg(self):
        return math.fsum(self.formed_mass_per_face_kg)

    @property
    def live_mass_kg(self):
        return math.fsum(self.live_mass_per_face_kg)

    @property
    def remelted_mass_kg(self):
        return math.fsum(self.remelted_mass_per_face_kg)


def _source_body(history):
    return {"format": FORMAT, "front_closure": FRONT_CLOSURE,
        "configuration_json": history.configuration_json,
        "initial_vertices": history.initial_vertices, "initial_faces": history.initial_faces,
        "initial_radius_km": history.initial_radius_km, "solidus_k": history.solidus_k,
        "layer_mass_kg": history.layer_mass_kg}


def _remember(history):
    _TRUSTED[:] = [saved for saved in _TRUSTED if saved is not history]+[history]
    del _TRUSTED[:-4]


def _boundaries(mass):
    total = math.fsum(mass)
    result = [0.]
    for layer in range(1, len(mass)):
        result.append(math.fsum(mass[:layer])/total)
    result.append(1.)
    if any(b <= a for a, b in zip(result, result[1:])):
        raise ValueError("Bulk birth layer mass coordinates are not numerically resolved")
    return result


def _growth(before, after, boundaries):
    if after <= before:
        return
    for layer, (lo, hi) in enumerate(zip(boundaries, boundaries[1:])):
        a, b = max(before, lo), min(after, hi)
        if b > a:
            yield layer, a, b


def _check_owner(history, layer_mass_kg, time_myr, front_fraction, source_identity):
    if type(history) is not BulkBirthHistory:
        raise ValueError("Missing bulk birth history; recalculate from the molten Genesis origin")
    if _real(time_myr, "owner time_myr", nonnegative=True) != history.time_myr:
        raise ValueError("Bulk birth and owner clocks disagree")
    if _vector(front_fraction, "owner front_fraction", fractions=True) != history.front_fraction:
        raise ValueError("Bulk birth and conductive solidus fronts disagree")
    try:
        supplied = np.asarray(layer_mass_kg)
    except (TypeError, ValueError) as exc:
        raise ValueError("Bulk birth owner material inventory is invalid") from exc
    if supplied.dtype.kind not in "fiu" or not np.array_equal(supplied, np.asarray(history.layer_mass_kg)):
        raise ValueError("Bulk birth fixed material inventory changed")
    if source_identity is not None and _pin(source_identity, "owner source_identity") != history.source_identity:
        raise ValueError("Bulk birth producer identity changed")


def validate_bulk_birth(history, *, layer_mass_kg, time_myr, front_fraction, source_identity=None):
    """Check owner endpoints; replay every untrusted birth and retirement.

    Identity caching applies only to this immutable in-memory object. Fresh
    constructed states and every archive load undergo complete source/replay
    validation, independent of any serialized checksum.
    """
    _check_owner(history, layer_mass_kg, time_myr, front_fraction, source_identity)
    if any(saved is history for saved in _TRUSTED):
        return history
    try:
        configuration = json.loads(history.configuration_json)
        if type(configuration) is not dict or _canonical(configuration) != history.configuration_json:
            raise ValueError("Bulk birth configuration must be its complete canonical object")
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Bulk birth configuration is invalid") from exc
    if _digest(_source_body(history)) != history.source_identity:
        raise ValueError("Bulk birth source identity or fixed origin changed")
    from types import SimpleNamespace
    _geometry(SimpleNamespace(vertices=np.asarray(history.initial_vertices), faces=np.asarray(history.initial_faces)),
              history.initial_radius_km)
    boundaries = [_boundaries(row) for row in history.layer_mass_kg]
    count = len(boundaries)
    time, front = 0., (0.,)*count
    expected, per_face = [], [[] for _ in range(count)]
    for interval in history.intervals:
        if interval.before_time_myr != time or interval.before_front_fraction != front:
            raise ValueError("Bulk birth accepted thermal/front history contains a gap")
        if len(interval.after_front_fraction) != count:
            raise ValueError("Bulk birth accepted interval changed material face count")
        for face, (old, new) in enumerate(zip(front, interval.after_front_fraction)):
            if new < old:
                for index in per_face[face]:
                    record = expected[index]
                    record[4] = max(record[2], min(record[4], new))
            for layer, lo, hi in _growth(old, new, boundaries[face]):
                per_face[face].append(len(expected))
                expected.append([face, layer, lo, hi, hi, interval.before_time_myr, interval.after_time_myr])
        time, front = interval.after_time_myr, interval.after_front_fraction
    if time != history.time_myr or front != history.front_fraction:
        raise ValueError("Bulk birth history does not reach its current endpoint")
    if len(expected) != len(history.cohorts):
        raise ValueError("Bulk birth cohort catalogue differs from accepted front history")
    live = [[] for _ in range(count)]
    for sequence, (cohort, row) in enumerate(zip(history.cohorts, expected), 1):
        actual = [cohort.face_index, cohort.layer_index, cohort.q_lo, cohort.q_hi, cohort.live_q_hi,
                  cohort.birth_before_time_myr, cohort.birth_after_time_myr]
        if cohort.id != sequence or actual != row or cohort.birth_temperature_k != history.solidus_k:
            raise ValueError("Bulk birth identity, material interval, birth bracket or retirement changed")
        if cohort.live_q_hi > cohort.q_lo:
            live[cohort.face_index].append((cohort.q_lo, cohort.live_q_hi))
    for face, spans in enumerate(live):
        end = 0.
        for lo, hi in sorted(spans):
            if lo != end:
                raise ValueError("Bulk birth live material has an overlap or uncovered interval")
            end = hi
        if end != history.front_fraction[face]:
            raise ValueError("Bulk birth live coverage disagrees with the conductive front")
    target = np.asarray(history.column_mass_kg)*np.asarray(history.front_fraction)
    if not np.allclose(history.live_mass_per_face_kg, target, rtol=2e-13, atol=0):
        raise ValueError("Bulk birth live mass does not conserve the existing column inventory")
    _remember(history)
    return history


def initial_bulk_birth(mesh, layer_mass_kg, radius_km, configuration, solidus_k):
    """Create empty records at the caller's validated fully molten t=0 origin."""
    if not isinstance(configuration, Mapping):
        raise ValueError("Bulk birth needs the complete producer configuration")
    vertices, faces, _, _ = _geometry(mesh, radius_km)
    mass = _matrix(layer_mass_kg, "layer_mass_kg", positive=True)
    if len(mass) != len(faces):
        raise ValueError("Bulk birth initial mass and material face counts disagree")
    context = dict(configuration_json=_canonical(dict(configuration)),
        initial_vertices=tuple(map(tuple, vertices.tolist())), initial_faces=tuple(map(tuple, faces.tolist())),
        initial_radius_km=_real(radius_km, "initial_radius_km", positive=True),
        solidus_k=_real(solidus_k, "solidus_k", positive=True), layer_mass_kg=mass)
    pin = _digest({"format": FORMAT, "front_closure": FRONT_CLOSURE, **context})
    history = BulkBirthHistory(0., (0.,)*len(mass), pin, **context)
    return validate_bulk_birth(history, layer_mass_kg=mass, time_myr=0., front_fraction=history.front_fraction)


def advance_bulk_birth(history, *, before_front_fraction, after_front_fraction,
                       before_time_myr, after_time_myr, mesh, radius_km, solidus_k):
    """Propose one accepted conductive endpoint without mutating the parent."""
    if type(history) is not BulkBirthHistory:
        raise ValueError("Missing bulk birth history; recalculate from the molten Genesis origin")
    validate_bulk_birth(history, layer_mass_kg=history.layer_mass_kg,
        time_myr=before_time_myr, front_fraction=before_front_fraction)
    if _real(solidus_k, "solidus_k", positive=True) != history.solidus_k:
        raise ValueError("Bulk birth solidus boundary law changed")
    interval = BulkBirthInterval(before_time_myr, after_time_myr, before_front_fraction, after_front_fraction)
    if len(interval.after_front_fraction) != len(history.layer_mass_kg):
        raise ValueError("Bulk birth cannot change material face count")
    _, faces, metrics, frames = _geometry(mesh, radius_km)
    if len(faces) != len(history.layer_mass_kg):
        raise ValueError("Bulk birth geometry lost material face ordering/count")
    old, new = interval.before_front_fraction, interval.after_front_fraction
    cohorts = []
    for cohort in history.cohorts:
        face = cohort.face_index
        live = max(cohort.q_lo, min(cohort.live_q_hi, new[face])) if new[face] < old[face] else cohort.live_q_hi
        cohorts.append(cohort if live == cohort.live_q_hi else replace(cohort, live_q_hi=live))
    for face, (a, b) in enumerate(zip(old, new)):
        for layer, lo, hi in _growth(a, b, _boundaries(history.layer_mass_kg[face])):
            cohorts.append(BulkBirthCohort(len(cohorts)+1, face, layer, lo, hi, hi,
                interval.before_time_myr, interval.after_time_myr, history.solidus_k,
                tuple(map(tuple, metrics[face].tolist())), tuple(map(tuple, frames[face].tolist())), radius_km))
    result = replace(history, time_myr=interval.after_time_myr, front_fraction=new,
                     cohorts=tuple(cohorts), intervals=history.intervals+(interval,))
    # This is the checked local transition from a fully replayed immutable
    # parent. Do not repeat the growing complete replay at ordinary endpoints.
    _remember(result)
    return result


def bulk_birth_to_dict(history):
    if type(history) is not BulkBirthHistory:
        raise ValueError("Cannot archive missing bulk birth history")
    validate_bulk_birth(history, layer_mass_kg=history.layer_mass_kg,
        time_myr=history.time_myr, front_fraction=history.front_fraction)
    body = {"format": FORMAT, "front_closure": FRONT_CLOSURE, "time_closure": TIME_CLOSURE,
        "geometry_closure": GEOMETRY_CLOSURE, "remelt_closure": REMELT_CLOSURE, **asdict(history)}
    body = json.loads(_canonical(body))
    return {**body, "sha256": _digest(body)}


def validate_bulk_birth_continuation(history, source_history):
    """Require a genuine append-only continuation of an inherited catalogue.

    Source cohorts may retire further; their identity, formed interval, birth
    bracket and geometry remain literal inherited data. A replacement source,
    changed past interval or revived source material is rejected.
    """
    for value in (history, source_history):
        if type(value) is not BulkBirthHistory:
            raise ValueError("Bulk birth continuation requires complete source history")
        validate_bulk_birth(value, layer_mass_kg=value.layer_mass_kg,
            time_myr=value.time_myr, front_fraction=value.front_fraction)
    if (history.source_identity != source_history.source_identity
            or _source_body(history) != _source_body(source_history)
            or history.time_myr < source_history.time_myr
            or len(history.intervals) < len(source_history.intervals)
            or history.intervals[:len(source_history.intervals)] != source_history.intervals
            or len(history.cohorts) < len(source_history.cohorts)):
        raise ValueError("Bulk birth continuation substituted its origin or accepted history")
    for current, source in zip(history.cohorts, source_history.cohorts):
        if (current.live_q_hi > source.live_q_hi
                or replace(current, live_q_hi=source.live_q_hi) != source):
            raise ValueError("Bulk birth continuation replaced a source cohort or revived retired material")
    return history


def bulk_birth_from_dict(raw):
    """Restore only a complete self-contained record; never fill old history."""
    required = {f.name for f in fields(BulkBirthHistory)} | {
        "format", "front_closure", "time_closure", "geometry_closure", "remelt_closure", "sha256"}
    if type(raw) is not dict or set(raw) != required:
        raise ValueError("Missing or malformed bulk birth archive; recalculate from molten Genesis origin")
    try:
        body = {key: value for key, value in raw.items() if key != "sha256"}
        if (raw["format"] != FORMAT or raw["front_closure"] != FRONT_CLOSURE
                or raw["time_closure"] != TIME_CLOSURE or raw["geometry_closure"] != GEOMETRY_CLOSURE
                or raw["remelt_closure"] != REMELT_CLOSURE or _pin(raw["sha256"], "archive sha256") != _digest(body)):
            raise ValueError("Bulk birth archive hash, model or declared closure changed")
        records = {}
        for key, cls in (("cohorts", BulkBirthCohort), ("intervals", BulkBirthInterval)):
            if type(raw[key]) is not list or any(type(row) is not dict or set(row) != {f.name for f in fields(cls)} for row in raw[key]):
                raise ValueError(f"Bulk birth archive {key} must contain complete records")
            records[key] = tuple(cls(**row) for row in raw[key])
        history = BulkBirthHistory(**{f.name: raw[f.name] for f in fields(BulkBirthHistory)
                                    if f.name not in records}, **records)
        return validate_bulk_birth(history, layer_mass_kg=history.layer_mass_kg,
            time_myr=history.time_myr, front_fraction=history.front_fraction)
    except (TypeError, OverflowError, IndexError) as exc:
        raise ValueError("Malformed bulk birth archive") from exc
