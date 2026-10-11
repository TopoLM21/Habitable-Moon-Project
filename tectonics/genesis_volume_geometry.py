"""Mass-exact current polyhedral geometry for actual Genesis material.

The outer vertices are the saved material surface vertices, in metres. Each
face owns its radial cone. A fixed column mass fraction q is placed by
``x = a(q) * (X0 + u*(X1-X0) + v*(X2-X0))``, with
``a(q)**3 = 1 - 6*M*q/(rho*det[X0,X1,X2])``. Its parameter Jacobian has the
constant determinant ``-2*M/rho``. Thus this *declared current closure* has
exact material volume even on the coarse spherical producer mesh.

This chooses a current reference for subsequent increments. It does not
recover unknown raft motion, birth placement, total strain or natural metric,
and it does not reset the producer's constitutive prestress. The companion
operator must inherit every cohort's elastic memory separately.

The three tetrahedra partition each physical frustum for geometry purposes.
They are NOT fixed incompressible material cells under affine node motion:
their individual volume fractions change with the inner/outer radius ratio.
Material kinematics instead use the analytic mass-q Jacobian above.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from numbers import Integral, Real

import numpy as np


GEOMETRY_CLOSURE = "current_mass_exact_homothetic_polyhedral_cones"
REFERENCE_SCOPE = "current_reference_with_inherited_cohort_prestress"
TETRAHEDRON_SCOPE = "geometric_partition_not_affine_material_parcels"


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite real")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite real")
    return result


def _q(value, name="q"):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite mass fraction")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _chart(triangle_m, column_mass_kg, density):
    triangle = np.asarray(triangle_m, dtype=float)
    if triangle.shape != (3, 3) or not np.isfinite(triangle).all():
        raise ValueError("Conical chart requires three finite Cartesian surface vertices")
    radius = np.linalg.norm(triangle, axis=1)
    if not np.isfinite(radius).all() or np.any(radius <= 0) or not np.allclose(
            radius, radius[0], rtol=2e-12, atol=0):
        raise ValueError("Conical surface vertices must share one positive source radius")
    r = float(radius[0])
    determinant_unit = float(np.linalg.det(triangle/r))
    if not math.isfinite(determinant_unit) or determinant_unit <= 1e-15:
        raise ValueError("Conical triangle must be outward oriented and noncollapsed")
    mass = _positive(column_mass_kg, "column_mass_kg")
    rho = _positive(density, "density")
    volume = mass/rho
    if not math.isfinite(volume) or volume <= 0:
        raise ValueError("Conical column volume is not finite and positive")
    coefficient = 6*(volume/r/r/r)/determinant_unit
    if not math.isfinite(coefficient) or coefficient <= 0:
        raise ValueError("Conical mass coefficient is not finite and positive")
    return triangle, volume, coefficient


def _scale(coefficient, q):
    argument = coefficient*q
    if not math.isfinite(argument) or argument >= 1:
        raise ValueError("Conical column reaches or crosses the planet centre")
    return math.exp(math.log1p(-argument)/3)


def _uv(value):
    uv = np.asarray(value, dtype=float)
    if uv.shape != (2,) or not np.isfinite(uv).all() or np.any(uv < 0) or uv.sum() > 1:
        raise ValueError("Conical coordinates require u,v >= 0 and u+v <= 1")
    return uv


def conical_point(triangle_m, column_mass_kg, density, uv, q):
    """Place one material point in the declared current mass-exact chart."""
    triangle, _, coefficient = _chart(triangle_m, column_mass_kg, density)
    u, v = _uv(uv)
    scale = _scale(coefficient, _q(q))
    point = scale*(triangle[0]+u*(triangle[1]-triangle[0])+v*(triangle[2]-triangle[0]))
    if not np.isfinite(point).all():
        raise ValueError("Conical point overflowed")
    return point


def conical_jacobian(triangle_m, column_mass_kg, density, uv, q):
    """Return dx/d(u,v,q), with negative determinant -2*column_mass/rho.

    Positive q points inward; this is the reason for the determinant sign.
    Every comparison between current and reference charts uses the SAME
    material coordinates and therefore has positive volume Jacobian.
    """
    triangle, volume, coefficient = _chart(triangle_m, column_mass_kg, density)
    u, v = _uv(uv)
    scale = _scale(coefficient, _q(q))
    direction = triangle[0]+u*(triangle[1]-triangle[0])+v*(triangle[2]-triangle[0])
    derivative = -coefficient/(3*scale*scale)
    result = np.column_stack((scale*(triangle[1]-triangle[0]),
                              scale*(triangle[2]-triangle[0]), derivative*direction))
    determinant = float(np.linalg.det(result))
    if not np.isfinite(result).all() or not math.isfinite(determinant) or not math.isclose(
            determinant, -2*volume, rel_tol=2e-12, abs_tol=0):
        raise ValueError("Conical material Jacobian failed its mass determinant identity")
    return result


def conical_cohort_tetrahedra(triangle_m, column_mass_kg, density, q_lo, q_hi,
                             global_vertex_ids=None):
    """Return three positively oriented geometry tetrahedra (3,4,3).

    Sorting the actual global vertex IDs selects the same diagonal on shared
    side quadrilaterals when their radial levels match. Unequal adjacent
    radial subdivisions still have nonconforming interfaces. These tetrahedra
    are geometric partitions, not immutable affine material parcels.
    """
    triangle, volume, coefficient = _chart(triangle_m, column_mass_kg, density)
    lo, hi = _q(q_lo, "q_lo"), _q(q_hi, "q_hi")
    if lo >= hi:
        raise ValueError("Conical cohort must have positive live mass interval")
    if global_vertex_ids is not None:
        ids = tuple(global_vertex_ids)
        if (len(ids) != 3 or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral)
                                or x < 0 for x in ids) or len(set(ids)) != 3):
            raise ValueError("Conical tetrahedra require three distinct global vertex IDs")
        triangle = triangle[np.argsort(ids)]
    outer, inner = _scale(coefficient, lo), _scale(coefficient, hi)
    if not outer > inner > 0:
        raise ValueError("Conical live interval is below resolvable positive geometric thickness")
    lower, upper = inner*triangle, outer*triangle
    points = np.concatenate((lower, upper))
    indices = ((0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5))
    tetrahedra = points[np.asarray(indices)].copy()
    signed = np.linalg.det((tetrahedra[:, 1:]-tetrahedra[:, :1]).transpose(0, 2, 1))/6
    if not np.isfinite(signed).all() or np.any(signed == 0):
        raise ValueError("Conical geometry tetrahedron is collapsed or nonfinite")
    for i in np.flatnonzero(signed < 0):
        tetrahedra[i, [0, 1]] = tetrahedra[i, [1, 0]]
    measured = float(np.abs(signed).sum())
    expected = volume*(hi-lo)
    # Subtracting nearly equal large vertex coordinates loses a few ulps in
    # very thin newborn cohorts. The bound scales with geometric resolution,
    # rather than disguising this as a relative physical mass tolerance.
    edge = max(np.linalg.norm(triangle[1]-triangle[0]), np.linalg.norm(triangle[2]-triangle[0]))
    resolution = 64*np.finfo(float).eps*np.linalg.norm(triangle[0])*edge*edge
    if not math.isclose(measured, expected, rel_tol=3e-12, abs_tol=float(resolution)):
        raise ValueError("Conical tetrahedra fail their exact cohort volume identity")
    return tetrahedra


@dataclass(frozen=True, slots=True)
class ConicalCohortGeometry:
    id: int
    face_index: int
    layer_index: int
    q_lo: float
    live_q_hi: float
    mass_kg: float
    volume_m3: float
    top_triangle_m: tuple
    outer_scale: float
    inner_scale: float
    tetrahedra_m: tuple
    owner_id: str
    material_id: str


@dataclass(frozen=True, slots=True)
class GenesisVolumeGeometry:
    source_checkpoint_sha256: str
    source_identity: str
    time_myr: float
    vertices_m: tuple
    faces: tuple
    column_mass_kg: tuple
    density_kg_m3: float
    cohorts: tuple
    owner_ids_per_face: tuple
    owner_components: tuple
    source_packet_json: str = field(repr=False)
    runtime_birth_history_json: str | None = field(default=None, repr=False)
    closure: str = GEOMETRY_CLOSURE
    reference_scope: str = REFERENCE_SCOPE
    tetrahedron_scope: str = TETRAHEDRON_SCOPE

    def __deepcopy__(self, memo):
        return self

    @property
    def full_historical_three_dimensional_memory_present(self):
        return False

    @property
    def stress_free_reinitialization(self):
        return False

    @property
    def continental_plate_ownership_claimed(self):
        return False

    @property
    def physical_force_work_admission(self):
        return False


def _owners(faces, active, source_identity):
    edges = {}
    adjacency = [set() for _ in faces]
    for index, face in enumerate(faces):
        for i in range(3):
            edge = tuple(sorted((face[i], face[(i+1) % 3])))
            edges.setdefault(edge, []).append(index)
    if any(len(neighbors) != 2 for neighbors in edges.values()):
        raise ValueError("Genesis volume ownership requires a closed two-face edge topology")
    for neighbors in edges.values():
        first, second = neighbors
        if active[first] and active[second]:
            adjacency[first].add(second)
            adjacency[second].add(first)
    remaining = {i for i, solid in enumerate(active) if solid}
    components = []
    owners = [None]*len(faces)
    while remaining:
        start = min(remaining)
        todo, component = [start], set()
        while todo:
            face = todo.pop()
            if face in component:
                continue
            component.add(face)
            todo.extend(adjacency[face]-component)
        remaining.difference_update(component)
        ordered = tuple(sorted(component))
        owner = f"genesis:{source_identity}:connected-body:{ordered[0]}"
        for face in ordered:
            owners[face] = owner
        components.append(ordered)
    return tuple(owners), tuple(components)


def _cohort_geometry(history, vertices, faces, mass, rho, source_identity):
    front = np.asarray(history["front_fraction"])
    owners, components = _owners(faces, front > 0, source_identity)
    cohorts = []
    for birth in history["cohorts"]:
        lo, hi = birth["q_lo"], birth["live_q_hi"]
        if hi <= lo:
            continue
        face = birth["face_index"]
        triangle = vertices[np.asarray(faces[face])]
        _, volume, coefficient = _chart(triangle, mass[face], rho)
        tets = conical_cohort_tetrahedra(triangle, mass[face], rho, lo, hi, faces[face])
        material_id = (f"genesis:{source_identity}:face:{face}:"
                       f"layer:{birth['layer_index']}:cohort:{birth['id']}")
        cohorts.append(ConicalCohortGeometry(
            birth["id"], face, birth["layer_index"], lo, hi,
            float(mass[face]*(hi-lo)), float(volume*(hi-lo)),
            tuple(tuple(row) for row in triangle.tolist()),
            _scale(coefficient, lo), _scale(coefficient, hi),
            tuple(tuple(tuple(row) for row in tet) for tet in tets.tolist()),
            owners[face], material_id))
    return tuple(cohorts), owners, components


def _from_validated_packet(packet):
    vertices = np.asarray(packet["current_geometry"]["vertices_unit_xyz"], float)
    vertices *= 1000*packet["current_geometry"]["radius_km"]
    faces = tuple(tuple(row) for row in packet["current_geometry"]["faces"])
    mass = np.asarray(packet["current_columns"]["layer_mass_kg"], float).sum(axis=1)
    rho = float(packet["source"]["parameters"]["shell"]["density_kg_m3"])
    cohorts, owners, components = _cohort_geometry(packet["bulk_birth_history"],
        vertices, faces, mass, rho, packet["birth_source_identity"])
    return GenesisVolumeGeometry(
        packet["source"]["checkpoint_sha256"], packet["birth_source_identity"], packet["time_myr"],
        tuple(tuple(row) for row in vertices.tolist()), faces, tuple(float(x) for x in mass), rho,
        cohorts, owners, components,
        json.dumps(packet, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))


def build_genesis_volume_geometry(packet, *, expected_source_checkpoint_sha256=None):
    """Reconstruct a CURRENT reference from the actual embedded Fault source.

    All input claims are first rebuilt by the strict producer exporter. The
    reference binding remains to those source bytes; no older mechanical
    fixture, mean-strain replacement or missing activation history is used.
    """
    from .genesis_bulk_export import validate_genesis_bulk_material
    validated = validate_genesis_bulk_material(packet,
        expected_source_checkpoint_sha256=expected_source_checkpoint_sha256)
    return _from_validated_packet(validated)


def validate_genesis_volume_geometry(geometry, packet=None, *, expected_source_checkpoint_sha256=None):
    """Replay source and geometry, including resealed source-backed claims."""
    if type(geometry) is not GenesisVolumeGeometry:
        raise ValueError("A genuine source-bound GenesisVolumeGeometry is required")
    original = json.loads(geometry.source_packet_json) if packet is None else packet
    replay = build_genesis_volume_geometry(original,
        expected_source_checkpoint_sha256=expected_source_checkpoint_sha256)
    if geometry.runtime_birth_history_json is not None:
        from .genesis_bulk_birth import bulk_birth_from_dict, validate_bulk_birth_continuation
        history = bulk_birth_from_dict(json.loads(geometry.runtime_birth_history_json))
        source_history = bulk_birth_from_dict(json.loads(replay.source_packet_json)["bulk_birth_history"])
        validate_bulk_birth_continuation(history, source_history)
        replay = _updated_geometry(replay, history)
    if geometry != replay:
        raise ValueError("Genesis volume geometry differs from full source geometry replay")
    return geometry


def _updated_geometry(source_geometry, history):
    from dataclasses import replace
    from .genesis_bulk_birth import bulk_birth_to_dict
    history_dict = bulk_birth_to_dict(history)
    cohorts, owners, components = _cohort_geometry(history_dict,
        np.asarray(source_geometry.vertices_m), source_geometry.faces,
        np.asarray(source_geometry.column_mass_kg), source_geometry.density_kg_m3,
        source_geometry.source_identity)
    return replace(source_geometry, time_myr=history.time_myr, cohorts=cohorts,
        owner_ids_per_face=owners, owner_components=components,
        runtime_birth_history_json=json.dumps(history_dict, sort_keys=True,
            separators=(",", ":"), ensure_ascii=False, allow_nan=False))


def build_updated_genesis_volume_geometry(source_geometry, bulk_birth_history):
    """Extend live material in the SAME source-bound current reference.

    The evolving birth record is separate from the embedded producer packet:
    this function does not pretend the old source checkpoint contains new
    time, newly formed material or its later constitutive state. Prospective
    mechanics must supply actual stress-free activation and memory for new
    IDs; the geometry function supplies only their mass-q reference support.
    """
    from .genesis_bulk_birth import (bulk_birth_from_dict, bulk_birth_to_dict,
                                    validate_bulk_birth_continuation)
    validate_genesis_volume_geometry(source_geometry)
    history = bulk_birth_from_dict(bulk_birth_to_dict(bulk_birth_history))
    original = json.loads(source_geometry.source_packet_json)["bulk_birth_history"]
    previous = (original if source_geometry.runtime_birth_history_json is None
                else json.loads(source_geometry.runtime_birth_history_json))
    validate_bulk_birth_continuation(history, bulk_birth_from_dict(previous))
    return _updated_geometry(source_geometry, history)
