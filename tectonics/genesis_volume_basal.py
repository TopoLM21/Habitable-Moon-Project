"""Consistent inertia and reference-basal work for Genesis volume fields.

The Cartesian nodal field is the volume operator's literal
``v = sum(N_i*(A_i + (lambda-1)*P_i))``. Inertia integrates its kinetic
energy against actual conserved cohort masses. Basal drag acts only in the
tangent plane of the actual reference-bottom triangle; no normal drag,
artificial slope drag, diagonal mass or elastic support is added.

The prescribed mantle pattern and material parameters are the saved Shell
parameters. Its transmission uses actual conical normal thickness. The
existing ContactParameters drag coefficient is an explicit continuation
parameter, not a falsely attributed field of the earlier Fault checkpoint.
Loads are dead/reference loads on the chosen current reference with its
current live mass-q front. They do not declare a finite follower-load law,
continuous velocity history, static equilibrium or physical admission.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math

import numpy as np

from .genesis_volume_operator import GenesisVolumeOperator, _assemble_matrix, _immutable


FORMAT = "genesis-conical-consistent-inertia-reference-basal-0.1"
DRAG_CLOSURE = "actual_reference_basal_plane_tangential_linear_drag"
LOAD_CLOSURE = "actual_conical_normal_thickness_reference_basal_dead_traction"
CONSISTENT_ANGULAR_MASS = np.full((3, 3), 1/12)
np.fill_diagonal(CONSISTENT_ANGULAR_MASS, 1/6)
CONSISTENT_ANGULAR_MASS = _immutable(CONSISTENT_ANGULAR_MASS)


def radial_displacement_moments(outer_scale, inner_scale):
    """Return mean(lambda-1), variance(lambda) in the fixed mass-q measure.

    The radial measure is proportional to lambda**2 d(lambda). The midpoint
    expression preserves the small positive variance in very thin cohorts.
    It does not subtract two almost equal raw moments near lambda=1.
    """
    a, b = float(outer_scale), float(inner_scale)
    if not np.isfinite((a, b)).all() or not 0 < b < a <= 1:
        raise ValueError("Consistent mass requires positive ordered conical radial scales")
    centre = .5*(a+b)
    half_width = .5*(a-b)
    ratio2 = (half_width/centre)**2
    mean = .5*((a-1)+(b-1))+2*centre*ratio2/(3+ratio2)
    variance = half_width**2*(4+(1-ratio2)**2)/(15*(1+ratio2/3)**2)
    if not math.isfinite(mean) or not math.isfinite(variance) or variance <= 0:
        raise ValueError("Consistent radial mass moments are not finite and positive")
    return mean, variance


def local_consistent_mass_matrix(mass_kg, outer_scale, inner_scale):
    """Return the true 18x18 mass matrix in per-vertex (A_xyz,P_xyz) order."""
    mass = float(mass_kg)
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError("Consistent inertial mass must be finite and positive")
    mean, variance = radial_displacement_moments(outer_scale, inner_scale)
    radial = np.array(((1., mean), (mean, mean*mean+variance)))
    return mass*np.kron(CONSISTENT_ANGULAR_MASS, np.kron(radial, np.eye(3)))


def conical_interval_mass_matrix(top_triangle_m, column_mass_kg, density, q_lo, q_hi):
    """Integrate an actual positive material-q band without energy subtraction.

    The same function applies to whole live cohorts, truly born mass and
    truly removed remelt bands. An unchanged band must be skipped by the
    caller; it is never replaced by a numerical positive thickness floor.
    """
    from .genesis_volume_geometry import _chart, _q, _scale
    _, _, coefficient = _chart(top_triangle_m, column_mass_kg, density)
    lo, hi = _q(q_lo, "q_lo"), _q(q_hi, "q_hi")
    if not lo < hi:
        raise ValueError("Consistent inertial interval requires actual positive live mass")
    return local_consistent_mass_matrix(float(column_mass_kg)*(hi-lo),
        _scale(coefficient, lo), _scale(coefficient, hi))


def local_basal_drag_and_force(top_triangle_m, bottom_scale, vertex_traction_pa,
                               basal_drag_pa_s_m):
    """Consistent tangent drag and load on one physical reference-bottom face.

    The supplied traction vectors are literal prescribed world vectors;
    their work is integrated without fitting them to prestress. Drag uses
    the physical basal tangent projector. Returned area is in square metres.
    """
    triangle = np.asarray(top_triangle_m, float)
    traction = np.asarray(vertex_traction_pa, float)
    scale, beta = float(bottom_scale), float(basal_drag_pa_s_m)
    if (triangle.shape != (3, 3) or traction.shape != (3, 3)
            or not np.isfinite(triangle).all() or not np.isfinite(traction).all()
            or not math.isfinite(scale) or not 0 < scale <= 1
            or not math.isfinite(beta) or beta <= 0):
        raise ValueError("Basal integration requires finite geometry, traction and positive drag")
    cross = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
    length = float(np.linalg.norm(cross))
    if not math.isfinite(length) or length <= 0 or np.dot(cross, triangle[0]) <= 0:
        raise ValueError("Basal triangle must be outward oriented and noncollapsed")
    normal = cross/length
    tangent = np.eye(3)-np.outer(normal, normal)
    area = .5*length*scale*scale
    if not math.isfinite(area) or area <= 0:
        raise ValueError("Physical basal area must be finite and positive")
    h = scale-1
    radial = np.array(((1., h), (h, h*h)))
    drag = beta*area*np.kron(CONSISTENT_ANGULAR_MASS, np.kron(radial, tangent))
    a_force = area*(CONSISTENT_ANGULAR_MASS@traction)
    force = np.stack((a_force, h*a_force), axis=1).reshape(18)
    return drag, force, area


@dataclass(frozen=True, slots=True)
class GenesisVolumeBasalAssembly:
    source_checkpoint_sha256: str
    source_identity: str
    time_myr: float
    ndof: int
    mass_matrix: object
    basal_drag_matrix: object
    external_force_n: np.ndarray
    cohort_mass_matrices: np.ndarray
    face_basal_drag_matrices: np.ndarray
    face_external_force_n: np.ndarray
    face_dofs: np.ndarray
    basal_face_area_m2: np.ndarray
    basal_front_scale: np.ndarray
    actual_conical_thickness_km: np.ndarray
    source_thincolumn_thickness_km: np.ndarray
    basal_traction_at_vertices_pa: np.ndarray
    original_thincolumn_traction_at_vertices_pa: np.ndarray
    original_thincolumn_lumped_force_n: np.ndarray
    source_spherical_face_area_m2: np.ndarray
    contact_parameters_json: str
    total_live_mass_kg: float
    drag_closure: str = DRAG_CLOSURE
    load_closure: str = LOAD_CLOSURE

    @property
    def basal_drag_pa_s(self):
        return self.basal_drag_matrix

    @property
    def external_force(self):
        return self.external_force_n

    @property
    def basal_area_m2(self):
        return float(np.sum(self.basal_face_area_m2))

    @property
    def normal_basal_drag_added(self):
        return False

    @property
    def arbitrary_diagonal_mass_or_slope_drag_added(self):
        return False

    @property
    def full_continuous_velocity_history_inherited(self):
        return False

    @property
    def physical_force_work_admission(self):
        return False


@dataclass(frozen=True, slots=True)
class GenesisAcceptedVertexRate:
    source_checkpoint_sha256: str
    source_identity: str
    before_time_myr: float
    after_time_myr: float
    velocity_m_s: np.ndarray
    before_vertices_m: np.ndarray
    after_vertices_m: np.ndarray
    max_shared_node_mismatch_m: float
    coordinate_roundoff_bound_m: float
    max_metric_reconstruction_error_m2: float
    max_radius_reconstruction_error_m: float
    semantics: str = "last_accepted_chord_finite_difference_surface_rate"

    @property
    def continuous_instantaneous_velocity_history_recovered(self):
        return False


def recover_source_last_accepted_vertex_rate(packet):
    """Recover the last known nodal chord increment from recorded birth charts.

    G, the proper world frame and R uniquely determine an outward chord
    triangle on the sphere. All faces must have at least one birth chart in
    the SAME last accepted interval; missing coverage is refused. Shared
    nodes are checked against floating chart resolution, then the first
    canonical record is retained. No zero filling, least-squares velocity
    projection or invented continuous/through-depth history is performed.
    """
    from .genesis import SECONDS_PER_MYR
    from .genesis_bulk_export import validate_genesis_bulk_material

    source = validate_genesis_bulk_material(packet)
    history = source["bulk_birth_history"]
    intervals = history["intervals"]
    if not intervals:
        raise ValueError("Source has no last accepted material interval to recover")
    interval = intervals[-1]
    before, after = interval["before_time_myr"], interval["after_time_myr"]
    if after != source["time_myr"] or not before < after:
        raise ValueError("Source last accepted rate requires its exact current accepted interval")
    records = [c for c in history["cohorts"]
               if c["birth_before_time_myr"] == before and c["birth_after_time_myr"] == after]
    faces = np.asarray(source["current_geometry"]["faces"], int)
    if {c["face_index"] for c in records} != set(range(len(faces))):
        raise ValueError("Last accepted birth charts do not cover every source material face")
    current = (np.asarray(source["current_geometry"]["vertices_unit_xyz"], float)
               *1000*source["current_geometry"]["radius_km"])
    previous = np.full_like(current, np.nan)
    reconstructed = []
    max_metric, max_radius, bound = 0., 0., 0.
    for record in records:
        metric = np.asarray(record["birth_metric_m2"], float)
        frame = np.asarray(record["birth_frame_xyz"], float)
        radius = 1000*record["birth_radius_km"]
        first = math.sqrt(metric[0, 0])
        other_x = metric[0, 1]/first
        other_y2 = metric[1, 1]-other_x*other_x
        if not math.isfinite(other_y2) or other_y2 <= 0:
            raise ValueError("Last accepted source chart is collapsed")
        other_y = math.sqrt(other_y2)
        x = -.5*first
        y = (-.5*metric[1, 1]-x*other_x)/other_y
        normal2 = radius*radius-x*x-y*y
        if not math.isfinite(normal2) or normal2 <= 0:
            raise ValueError("Last accepted source chart cannot be placed outward on its recorded sphere")
        local = np.array(((x, y, math.sqrt(normal2)),
                          (x+first, y, math.sqrt(normal2)),
                          (x+other_x, y+other_y, math.sqrt(normal2))))
        triangle = local@frame.T
        edge = np.column_stack((triangle[1]-triangle[0], triangle[2]-triangle[0]))
        metric_error = float(np.max(np.abs(edge.T@edge-metric)))
        radius_error = float(np.max(np.abs(np.linalg.norm(triangle, axis=1)-radius)))
        # The minimum planar angle controls chart reconstruction conditioning;
        # this is a numerical geometry bound, not a physical slip tolerance.
        resolution = 128*np.finfo(float).eps*radius*max(1., first/other_y)
        if radius_error > resolution or metric_error > resolution*max(first, math.sqrt(metric[1, 1]))*4:
            raise ValueError("Last accepted source chart reconstruction exceeds floating resolution")
        max_metric, max_radius = max(max_metric, metric_error), max(max_radius, radius_error)
        bound = max(bound, float(resolution))
        reconstructed.append((faces[record["face_index"]], triangle))
        for vertex, point in zip(faces[record["face_index"]], triangle):
            if np.isnan(previous[vertex, 0]):
                previous[vertex] = point
    if not np.isfinite(previous).all():
        raise ValueError("Last accepted source charts lack global vertex coverage")
    mismatch = max(float(np.max(np.linalg.norm(previous[ids]-triangle, axis=1)))
                   for ids, triangle in reconstructed)
    if mismatch > 2*bound:
        raise ValueError("Last accepted source birth charts disagree on shared material nodes")
    velocity = (current-previous)/((after-before)*SECONDS_PER_MYR)
    if not np.isfinite(velocity).all():
        raise ValueError("Recovered last accepted chord rate is not finite")
    return GenesisAcceptedVertexRate(source["source"]["checkpoint_sha256"],
        source["birth_source_identity"], before, after, _immutable(velocity),
        _immutable(previous), _immutable(current), mismatch, bound, max_metric, max_radius)


def assemble_genesis_volume_basal(operator, contact_parameters=None):
    """Integrate source-bound live mass, actual basal drag and mantle forcing.

    Geometry may contain a checked prospective birth continuation. Its
    reference surface and embedded original producer remain unchanged.
    This assembly is computed once per accepted/trial live material front;
    velocity and displacement variables retain their explicit SI units.
    """
    from .genesis_contact import ContactParameters
    from .genesis_material import rebuild_material_mesh
    from .genesis_shell import ShellParameters, mantle_traction
    from .genesis_volume_geometry import validate_genesis_volume_geometry
    from .mesh import build_icosphere

    if type(operator) is not GenesisVolumeOperator:
        raise ValueError("Consistent basal dynamics requires its genuine volume operator")
    geometry = validate_genesis_volume_geometry(operator.geometry)
    if (operator.source_checkpoint_sha256 != geometry.source_checkpoint_sha256
            or operator.source_identity != geometry.source_identity
            or operator.time_myr != geometry.time_myr):
        raise ValueError("Basal dynamics operator source/clock differs from its material geometry")
    parameters = ContactParameters() if contact_parameters is None else contact_parameters
    if type(parameters) is not ContactParameters:
        raise ValueError("Declare the actual ContactParameters for volume continuation drag")
    parameters.validate()
    packet = json.loads(geometry.source_packet_json)
    shell = ShellParameters(**packet["source"]["parameters"]["shell"])
    shell.validate()
    faces = np.asarray(geometry.faces, int)
    vertices = np.asarray(geometry.vertices_m, float)
    nface, ndof = len(faces), 6*len(vertices)
    if (operator.ndof != ndof or tuple(operator.cohort_ids) != tuple(c.id for c in geometry.cohorts)
            or tuple(operator.face_indices) != tuple(c.face_index for c in geometry.cohorts)):
        raise ValueError("Basal dynamics operator and live material IDs differ")
    cohort_local = np.asarray([conical_interval_mass_matrix(c.top_triangle_m,
        geometry.column_mass_kg[c.face_index], geometry.density_kg_m3, c.q_lo, c.live_q_hi)
        for c in geometry.cohorts])
    dofs = (6*faces[np.asarray(operator.face_indices), :, None]+np.arange(6)).reshape(-1, 18)
    if not np.array_equal(dofs, operator.dofs):
        raise ValueError("Basal dynamics requires the identical nodal volume field")
    mass_matrix = _assemble_matrix(dofs, cohort_local, ndof)
    face_dofs = (6*faces[:, :, None]+np.arange(6)).reshape(-1, 18)
    bottom = np.ones(nface)
    live = np.zeros(nface, bool)
    for cohort in geometry.cohorts:
        face = cohort.face_index
        bottom[face] = min(bottom[face], cohort.inner_scale)
        live[face] = True
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    normals = cross/np.linalg.norm(cross, axis=1)[:, None]
    height = np.einsum("fi,fi->f", triangles[:, 0], normals)
    actual_thickness = (1-bottom)*height/1000
    radius_m = 1000*packet["current_geometry"]["radius_km"]
    mesh = rebuild_material_mesh(build_icosphere(shell.subdivisions), vertices/radius_m)
    if not np.array_equal(mesh.faces, faces):
        raise ValueError("Basal dynamics changed the actual material connectivity")
    history = (packet["bulk_birth_history"] if geometry.runtime_birth_history_json is None
               else json.loads(geometry.runtime_birth_history_json))
    spherical_area = mesh.areas_unit_sphere*radius_m*radius_m
    old_thickness = (np.asarray(geometry.column_mass_kg)*np.asarray(history["front_fraction"])
                     /geometry.density_kg_m3/spherical_area/1000)
    traction = mantle_traction(mesh, shell, actual_thickness)
    old_traction = mantle_traction(mesh, shell, old_thickness)
    face_drag, face_force, area = np.zeros((nface, 18, 18)), np.zeros((nface, 18)), np.zeros(nface)
    for face in np.flatnonzero(live):
        face_drag[face], face_force[face], area[face] = local_basal_drag_and_force(
            triangles[face], bottom[face], traction[faces[face]], parameters.basal_drag_pa_s_m)
    drag_matrix = _assemble_matrix(face_dofs, face_drag, ndof)
    external = np.zeros(ndof)
    np.add.at(external, face_dofs.ravel(), face_force.ravel())
    # Raw literal source-thincolumn lumped world load, retained only to
    # measure the changed area/H/quadrature closure. It is never fitted to
    # the inherited prestress or used to calibrate the new physical force.
    old_lumped = np.zeros((len(vertices), 3))
    np.add.at(old_lumped, faces.ravel(),
              (spherical_area[:, None, None]*old_traction[faces]/3).reshape(-1, 3))
    return GenesisVolumeBasalAssembly(
        geometry.source_checkpoint_sha256, geometry.source_identity, geometry.time_myr, ndof,
        mass_matrix, drag_matrix, _immutable(external), _immutable(cohort_local),
        _immutable(face_drag), _immutable(face_force), _immutable(face_dofs, int),
        *map(_immutable, (area, bottom, actual_thickness, old_thickness,
                         traction, old_traction, old_lumped, spherical_area)),
        json.dumps(asdict(parameters), sort_keys=True, separators=(",", ":"), allow_nan=False),
        float(sum(c.mass_kg for c in geometry.cohorts)))
