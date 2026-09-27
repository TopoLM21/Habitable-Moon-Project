"""Physical-time, moving-reference mechanics of a continuous material shell.

Each trial solves forces on the moved geometry, using one accepted old mesh
and one supplied Maxwell/thermal memory throughout Newton iterations. Material
faces keep their identity and volume during a trial. This is an unbroken shell
kernel: no cohesive interface, crack insertion, or plate handoff is performed.
The caller owns thermal/orbital evolution and supplies a pure force callback.

The material tangent omits geometric and follower-load derivatives; line
search therefore checks the full moved-geometry residual. Work integrals are
explicit quadrature diagnostics, not a claim of an exactly closed energy law.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .genesis_contact import ContactParameters, SECONDS_PER_YEAR
from .genesis_material import (face_deformation, geometry_diagnostics, move_mesh,
    polar_increment, rebuild_material_mesh, rotate_tensor)
from .genesis_shell import Membrane, maximum_total_strain
from .genesis_seams import _closed_edges
from .mesh import SphereMesh, _build_topology


VERSION = "genesis-moving-tied-0.1"


class MovingTiedRetry(RuntimeError):
    """A pure trial was inadmissible or did not converge; retry a smaller dt."""


def _positive(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _array(value, shape, name, *, positive=False):
    value = np.asarray(value)
    if (value.shape != shape or value.dtype.kind not in "fiu"
            or not np.isfinite(value).all()):
        raise ValueError(f"{name} must be a finite real array with shape {shape}")
    if positive and np.any(value <= 0):
        raise ValueError(f"{name} must be positive")
    return np.array(value, dtype=float, copy=True)


def _nodal_area(mesh, radius_m):
    area = np.zeros(mesh.vertex_count)
    np.add.at(area, mesh.faces.ravel(), np.repeat(mesh.areas_unit_sphere/3, 3))
    return area*radius_m**2


def _current_increment(old_mesh, mesh, old_radius_m, radius_m, membrane):
    """Forward sphere-log motion expressed in the *arrival* tangent frame.

    This is minus log_current(old), with arc length measured at current radius.
    It is objective under a common spatial rotation and gives positive drag
    work. Motions approaching antipodal points are outside this kernel.
    """
    dot = np.clip(np.einsum("vi,vi->v", old_mesh.vertices, mesh.vertices), -1., 1.)
    axis = np.cross(old_mesh.vertices, mesh.vertices)
    sine = np.linalg.norm(axis, axis=1)
    angle = np.arctan2(sine, dot)
    if np.any(angle >= np.pi/2):
        raise ValueError("Tied shell increment reaches a hemisphere")
    factor = np.divide(angle, sine, out=np.ones_like(angle), where=sine > 0)
    # Cross products retain exact zero for equal positions and avoid the
    # subtraction of almost equal unit vectors for microscopic increments.
    tangent = np.cross(axis, mesh.vertices)*factor[:, None]
    result = np.r_[(radius_m*np.einsum("vij,vi->vj", membrane.vertex_basis, tangent)).ravel(),
                   radius_m-old_radius_m]
    return result, angle


def _kinematics(old_mesh, mesh, old_radius_m, radius_m):
    # An exact same-vertex radius change should not manufacture shear from
    # matrix inversion/SVD roundoff. It is also the common no-motion iterate.
    if np.array_equal(old_mesh.vertices, mesh.vertices):
        rotation = np.broadcast_to(np.eye(2), (mesh.cell_count, 2, 2)).copy()
        increment = np.zeros((mesh.cell_count, 3))
        increment[:, :2] = np.log(radius_m/old_radius_m)
        return rotation, increment
    deformation = face_deformation(old_mesh, mesh, old_radius_m/1000., radius_m/1000.)
    return polar_increment(deformation)


@dataclass(frozen=True)
class MovingTiedLoading:
    dt_years: float
    volume_m3: np.ndarray
    young_modulus_pa: np.ndarray
    memory: np.ndarray
    effective_b: np.ndarray
    external_force: object


@dataclass
class MovingTiedState:
    vertices: np.ndarray
    radius_m: float
    elastic_strain: np.ndarray
    elapsed_years: float
    last_step_years: float
    accepted_steps: int
    rejected_steps: int
    equilibrium_residual: float
    external_work_j: float
    drag_work_j: float
    bulk_work_j: float
    bulk_loading_correction_j: float
    mechanical_remainder_j: float
    last_increment_current_m: np.ndarray
    last_external_force_n: np.ndarray
    last_drag_force_n: np.ndarray
    last_volume_m3: np.ndarray
    last_young_modulus_pa: np.ndarray
    cumulative_motion_m: np.ndarray
    stopped_reason: str | None = None


class MovingTiedMechanics:
    """Updated material geometry with finite corotation and physical basal drag.

    Limits constrain individual increments, mesh quality and thin-shell radius;
    accumulated in-plane deformation is not reset or capped at a small strain.
    """

    max_vertex_motion_rad = .01
    max_elastic_strain = .03
    max_radius_change_fraction = .1
    min_face_quality = .15
    min_area_ratio = .15
    max_area_ratio = 6.

    def __init__(self, template_mesh, initial_radius_m, parameters=None,
                 poisson_ratio=.25, max_incremental_strain=.005):
        self.parameters = parameters or ContactParameters()
        if not isinstance(self.parameters, ContactParameters):
            raise ValueError("parameters must be ContactParameters")
        self.parameters.validate()
        self.initial_radius_m = _positive(initial_radius_m, "initial_radius_m")
        if (isinstance(poisson_ratio, (bool, np.bool_)) or not isinstance(poisson_ratio, Real)
                or not np.isfinite(poisson_ratio) or not -1 < poisson_ratio < .5):
            raise ValueError("poisson_ratio must lie in (-1, .5)")
        self.poisson_ratio = float(poisson_ratio)
        self.max_incremental_strain = _positive(max_incremental_strain, "max_incremental_strain")
        if self.max_incremental_strain > .02:
            raise ValueError("max_incremental_strain exceeds the supported numerical domain")
        _closed_edges(template_mesh)
        self.template_mesh = rebuild_material_mesh(template_mesh, template_mesh.vertices)
        self.template_mesh.neighbors, self.template_mesh.shared_edges = _build_topology(self.template_mesh.faces)
        self.ndof = 2*self.template_mesh.vertex_count+1
        self._array_shapes = {
            "vertices": (self.template_mesh.vertex_count, 3),
            "elastic_strain": (self.template_mesh.cell_count, 3),
            "last_increment_current_m": (self.ndof,),
            "last_external_force_n": (self.ndof,), "last_drag_force_n": (self.ndof,),
            "last_volume_m3": (self.template_mesh.cell_count,),
            "last_young_modulus_pa": (self.template_mesh.cell_count,),
            "cumulative_motion_m": (self.template_mesh.vertex_count,)}
        config = {"parameters": asdict(self.parameters), "initial_radius_m": self.initial_radius_m,
            "poisson_ratio": self.poisson_ratio, "max_incremental_strain": self.max_incremental_strain,
            "geometry_limits": {name: getattr(self, name) for name in
                ("max_vertex_motion_rad", "max_elastic_strain", "max_radius_change_fraction",
                 "min_face_quality", "min_area_ratio", "max_area_ratio")}}
        digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
        for value in (self.template_mesh.vertices, self.template_mesh.faces):
            digest.update(np.ascontiguousarray(value).tobytes())
        self.fingerprint = digest.hexdigest()
        self.configuration = config

    def initial(self, elastic_strain):
        elastic = _array(elastic_strain, (self.template_mesh.cell_count, 3), "elastic_strain")
        state = MovingTiedState(self.template_mesh.vertices.copy(), self.initial_radius_m,
            elastic, 0., 0., 0, 0, 0., 0., 0., 0., 0., 0.,
            np.zeros(self.ndof), np.zeros(self.ndof), np.zeros(self.ndof),
            np.zeros(self.template_mesh.cell_count), np.zeros(self.template_mesh.cell_count),
            np.zeros(self.template_mesh.vertex_count))
        self._validate_state(state)
        return state

    def mesh_for(self, state):
        return rebuild_material_mesh(self.template_mesh, state.vertices)

    @staticmethod
    def _internal(membrane, stress, volume, radius_m):
        local = np.einsum("fai,fa,f->fi", membrane.b, stress, volume/radius_m)
        result = np.zeros(membrane.ndof)
        np.add.at(result, membrane.dofs.ravel(), local.ravel())
        return result

    def _force_floor(self, young, volume, radius_m, drag_coefficient):
        material_scale = float(np.max(young))*float(volume.sum())/radius_m
        # Unit-vector storage also bounds the resolvable velocity: differentiating
        # sub-nanometre changes and multiplying by drag/dt cannot give arbitrarily
        # accurate forces. Include that bound, explicitly dependent on timestep.
        roundoff = np.finfo(float).eps*(32*material_scale
            +4*radius_m*float(np.linalg.norm(drag_coefficient)))
        return max(1e-10*material_scale, roundoff/self.parameters.equilibrium_tolerance)

    def _residual(self, internal, drag, external, young, volume, radius_m, drag_coefficient):
        residual = internal+drag-external
        # Finite geometry cannot resolve elastic strains below floating-point
        # roundoff. This absolute floor is explicit and affects unloaded
        # controls, not the MPa-scale trajectory. No stress is clamped.
        floor = self._force_floor(young, volume, radius_m, drag_coefficient)
        scale = max(float(np.linalg.norm(internal)), float(np.linalg.norm(external)), floor)
        return residual, float(np.linalg.norm(residual)/scale), scale

    def stress(self, state):
        """Last accepted stress in current face frames (zero before a solve)."""
        membrane = Membrane(self.mesh_for(state), self.poisson_ratio)
        return (state.elastic_strain@membrane.d.T)*state.last_young_modulus_pa[:, None]

    def force_diagnostics(self, state):
        """Independently reassemble the saved accepted force balance."""
        mesh = self.mesh_for(state)
        membrane = Membrane(mesh, self.poisson_ratio)
        stress = (state.elastic_strain@membrane.d.T)*state.last_young_modulus_pa[:, None]
        internal = self._internal(membrane, stress, state.last_volume_m3, state.radius_m)
        drag_coefficient = (np.r_[np.repeat(_nodal_area(mesh, state.radius_m), 2), 0.]
            *self.parameters.basal_drag_pa_s_m/(state.last_step_years*SECONDS_PER_YEAR)
            if state.accepted_steps else np.zeros(self.ndof))
        residual, ratio, scale = self._residual(internal, state.last_drag_force_n,
            state.last_external_force_n, state.last_young_modulus_pa,
            state.last_volume_m3, state.radius_m, drag_coefficient) if state.accepted_steps else (np.zeros(self.ndof), 0., 0.)
        return {"internal_force_n": internal, "residual_force_n": residual,
                "relative_residual": ratio, "force_scale_n": scale,
                "force_floor_n": self._force_floor(state.last_young_modulus_pa,
                    state.last_volume_m3, state.radius_m, drag_coefficient),
                "absolute_roundoff_tolerance_n": self.parameters.equilibrium_tolerance*self._force_floor(
                    state.last_young_modulus_pa, state.last_volume_m3, state.radius_m, drag_coefficient)}

    def _geometry_check(self, mesh, radius, elastic, *, old_mesh=None, old_radius=None):
        if abs(radius/self.initial_radius_m-1) > self.max_radius_change_fraction:
            raise MovingTiedRetry("moving_tied_thin_shell_limit")
        metrics = geometry_diagnostics(mesh, self.template_mesh, radius, self.initial_radius_m)
        if (metrics["min_face_quality"] < self.min_face_quality
                or metrics["min_area_ratio"] < self.min_area_ratio
                or metrics["max_area_ratio"] > self.max_area_ratio):
            raise MovingTiedRetry("moving_tied_mesh_quality_limit")
        if maximum_total_strain(elastic) > self.max_elastic_strain:
            raise MovingTiedRetry("moving_tied_elastic_limit")
        if old_mesh is not None:
            _, increment = _kinematics(old_mesh, mesh, old_radius, radius)
            dot = np.einsum("vi,vi->v", old_mesh.vertices, mesh.vertices)
            angle = np.arctan2(np.linalg.norm(np.cross(old_mesh.vertices, mesh.vertices), axis=1), dot)
            if (maximum_total_strain(increment) > self.max_incremental_strain
                    or float(angle.max()) > self.max_vertex_motion_rad):
                raise MovingTiedRetry("moving_tied_increment_limit")

    def _validate_state(self, state, *, check_force=True):
        if not isinstance(state, MovingTiedState):
            raise ValueError("State must be MovingTiedState")
        for name, shape in self._array_shapes.items():
            if not isinstance(getattr(state, name), np.ndarray):
                raise ValueError(f"State {name} must be an array")
            _array(getattr(state, name), shape, name)
        counts = {"accepted_steps", "rejected_steps"}
        nonnegative = {"elapsed_years", "last_step_years", "equilibrium_residual", "drag_work_j"}
        for item in fields(state):
            name, value = item.name, getattr(state, item.name)
            if name in self._array_shapes or name == "stopped_reason":
                continue
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not np.isfinite(value)):
                raise ValueError(f"Invalid state {name}")
            if name in counts and (not isinstance(value, Integral) or value < 0):
                raise ValueError(f"State {name} must be a nonnegative integer")
            if name in nonnegative and value < 0:
                raise ValueError(f"State {name} must be nonnegative")
        _positive(state.radius_m, "radius_m")
        if state.stopped_reason is not None and (not isinstance(state.stopped_reason, str) or not state.stopped_reason):
            raise ValueError("State stopped_reason must be nonempty or None")
        if np.any(state.cumulative_motion_m < 0):
            raise ValueError("Cumulative material motion must be nonnegative")
        if state.last_step_years > state.elapsed_years+128*np.finfo(float).eps*max(state.elapsed_years, 1.):
            raise ValueError("Last step exceeds elapsed mechanics history")
        if state.equilibrium_residual > self.parameters.equilibrium_tolerance:
            raise ValueError("Stored force residual exceeds tolerance")
        mesh = self.mesh_for(state)
        try:
            self._geometry_check(mesh, state.radius_m, state.elastic_strain)
        except MovingTiedRetry as exc:
            raise ValueError(str(exc)) from exc
        if not state.accepted_steps:
            scalar_zero = ("elapsed_years", "last_step_years", "equilibrium_residual", "external_work_j",
                "drag_work_j", "bulk_work_j", "bulk_loading_correction_j", "mechanical_remainder_j")
            array_zero = set(self._array_shapes)-{"vertices", "elastic_strain"}
            if (any(getattr(state, name) != 0 for name in scalar_zero)
                    or any(np.any(getattr(state, name)) for name in array_zero)
                    or state.radius_m != self.initial_radius_m
                    or not np.array_equal(state.vertices, self.template_mesh.vertices)):
                raise ValueError("Unadvanced state contains invented mechanics history")
            return
        if state.last_step_years <= 0 or state.elapsed_years <= 0:
            raise ValueError("Accepted mechanics requires a positive clock and last step")
        if np.any(state.last_volume_m3 <= 0) or np.any(state.last_young_modulus_pa <= 0):
            raise ValueError("Accepted state requires positive material volume and modulus")
        membrane = Membrane(mesh, self.poisson_ratio)
        area = np.r_[np.repeat(_nodal_area(mesh, state.radius_m), 2), 0.]
        expected_drag = area*self.parameters.basal_drag_pa_s_m/(state.last_step_years*SECONDS_PER_YEAR)*state.last_increment_current_m
        if not np.allclose(expected_drag, state.last_drag_force_n, rtol=2e-12, atol=1e-6):
            raise ValueError("Stored drag force disagrees with physical-time motion")
        tangent = state.last_increment_current_m[:-1].reshape(-1, 2)/state.radius_m
        angle = np.linalg.norm(tangent, axis=1)
        if float(angle.max()) > self.max_vertex_motion_rad*(1+1e-10):
            raise ValueError("Stored last motion exceeds incremental limit")
        previous_radius = state.radius_m-state.last_increment_current_m[-1]
        if previous_radius <= 0 or abs(previous_radius/self.initial_radius_m-1) > self.max_radius_change_fraction*(1+1e-10):
            raise ValueError("Stored last radial motion has invalid previous radius")
        step_distance = .5*(previous_radius+state.radius_m)*angle
        if np.any(state.cumulative_motion_m+1e-8 < step_distance):
            raise ValueError("Cumulative material motion is shorter than its last step")
        world = np.einsum("vij,vj->vi", membrane.vertex_basis, tangent)
        sinc = np.sinc(angle/np.pi)
        previous_vertices = np.cos(angle)[:, None]*mesh.vertices-sinc[:, None]*world
        previous_vertices /= np.linalg.norm(previous_vertices, axis=1)[:, None]
        previous_mesh = rebuild_material_mesh(self.template_mesh, previous_vertices)
        try:
            self._geometry_check(mesh, state.radius_m, state.elastic_strain,
                old_mesh=previous_mesh, old_radius=previous_radius)
        except MovingTiedRetry as exc:
            raise ValueError(str(exc)) from exc
        if check_force:
            diagnostics = self.force_diagnostics(state)
            measured = diagnostics["relative_residual"]
            if measured > self.parameters.equilibrium_tolerance*(1+1e-7):
                raise ValueError("Stored fields do not satisfy moved force equilibrium")
            if abs(measured-state.equilibrium_residual) > max(1e-14, 1e-5*self.parameters.equilibrium_tolerance):
                raise ValueError("Stored force residual disagrees with material fields")

    def trial(self, state, loading):
        """Return one accepted candidate without modifying state or loading.

        The callback must be pure: ``force(mesh, radius_m, depth_m, membrane)``
        returns nodal generalized force in N. Its radial component is permitted.
        Maxwell factors and thermal memory are supplied once for this interval.
        """
        self._validate_state(state)
        if state.stopped_reason:
            raise ValueError("Cannot advance stopped moving mechanics")
        if not isinstance(loading, MovingTiedLoading):
            raise ValueError("Loading must be MovingTiedLoading")
        dt_years = _positive(loading.dt_years, "dt_years")
        if state.elapsed_years+dt_years == state.elapsed_years:
            raise ValueError("Time increment cannot be represented")
        count = self.template_mesh.cell_count
        volume = _array(loading.volume_m3, (count,), "volume_m3", positive=True)
        young = _array(loading.young_modulus_pa, (count,), "young_modulus_pa", positive=True)
        memory = _array(loading.memory, (count, 3), "memory")
        beta = _array(loading.effective_b, (count,), "effective_b", positive=True)
        if np.any(beta > 1) or not callable(loading.external_force):
            raise ValueError("effective_b must lie in (0, 1] and external_force must be callable")
        dt_s = dt_years*SECONDS_PER_YEAR
        old_mesh = self.mesh_for(state)
        p = self.parameters

        def evaluate(mesh, radius):
            _positive(radius, "candidate radius")
            membrane = Membrane(mesh, self.poisson_ratio)
            rotation, increment = _kinematics(old_mesh, mesh, state.radius_m, radius)
            elastic = rotate_tensor(memory+beta[:, None]*increment, rotation, engineering=True)
            stress = (elastic@membrane.d.T)*young[:, None]
            internal = self._internal(membrane, stress, volume, radius)
            depth = volume/(mesh.areas_unit_sphere*radius**2)
            external = _array(loading.external_force(mesh, radius, depth.copy(), membrane),
                (self.ndof,), "external_force")
            motion, angle = _current_increment(old_mesh, mesh, state.radius_m, radius, membrane)
            drag_coefficient = np.r_[np.repeat(_nodal_area(mesh, radius), 2), 0.]*p.basal_drag_pa_s_m/dt_s
            drag = drag_coefficient*motion
            residual, relative, scale = self._residual(internal, drag, external, young, volume, radius, drag_coefficient)
            if not np.isfinite(relative):
                raise ValueError("Moved mechanics evaluation is nonfinite")
            return (membrane, rotation, increment, elastic, stress, external, motion, angle,
                    drag_coefficient, drag, residual, relative, scale)

        mesh, radius = old_mesh, state.radius_m
        response = evaluate(mesh, radius)
        for _ in range(p.max_newton_iterations):
            if response[-2] <= p.equilibrium_tolerance:
                break
            membrane = response[0]
            local = membrane.ki*(young*beta*volume/radius**2)[:, None, None]
            matrix = sparse.coo_matrix((local.ravel(), (membrane.rr, membrane.cc)),
                shape=(self.ndof, self.ndof)).tocsr()+sparse.diags(response[8])
            scaling = 1/np.sqrt(np.maximum(np.abs(matrix.diagonal()), 1.))
            diagonal = sparse.diags(scaling)
            correction = scaling*spsolve(diagonal@matrix@diagonal, -scaling*response[-3])
            if not np.isfinite(correction).all():
                raise MovingTiedRetry("moving_tied_linear_solve_limit")
            for power in range(18):
                factor = .5**power
                try:
                    candidate = move_mesh(mesh, membrane.vertex_basis,
                        correction[:-1].reshape(-1, 2)*factor/radius)
                    candidate_radius = radius+factor*correction[-1]
                    trial = evaluate(candidate, candidate_radius)
                except (ValueError, np.linalg.LinAlgError):
                    continue
                if trial[-2] < response[-2] or trial[-2] <= p.equilibrium_tolerance:
                    mesh, radius, response = candidate, candidate_radius, trial
                    break
            else:
                raise MovingTiedRetry("moving_tied_equilibrium_limit")
        if response[-2] > p.equilibrium_tolerance:
            raise MovingTiedRetry("moving_tied_equilibrium_limit")
        membrane, rotation, increment, elastic, stress, external, motion, angle, _, drag, _, residual, _ = response
        self._geometry_check(mesh, radius, elastic, old_mesh=old_mesh, old_radius=state.radius_m)
        initial_stress = (memory@membrane.d.T)*young[:, None]
        stress_old_frame = rotate_tensor(stress, rotation.transpose(0, 2, 1))
        work = float(np.sum(np.einsum("fi,fi->f", .5*(initial_stress+stress_old_frame), increment)*volume))
        energy_change = .5*float(np.sum((np.einsum("fi,fi->f", elastic, stress)
            -np.einsum("fi,fi->f", memory, initial_stress))*volume))
        external_work, drag_work = float(external@motion), float(drag@motion)
        result = replace(state, vertices=mesh.vertices.copy(), radius_m=float(radius),
            elastic_strain=elastic, elapsed_years=state.elapsed_years+dt_years,
            last_step_years=dt_years, accepted_steps=state.accepted_steps+1,
            equilibrium_residual=residual, external_work_j=state.external_work_j+external_work,
            drag_work_j=state.drag_work_j+drag_work, bulk_work_j=state.bulk_work_j+work,
            bulk_loading_correction_j=state.bulk_loading_correction_j+work-energy_change,
            mechanical_remainder_j=state.mechanical_remainder_j+external_work-work-drag_work,
            last_increment_current_m=motion.copy(), last_external_force_n=external.copy(),
            last_drag_force_n=drag.copy(), last_volume_m3=volume, last_young_modulus_pa=young,
            cumulative_motion_m=state.cumulative_motion_m+.5*(state.radius_m+radius)*angle)
        return result

    def save_state(self, path, state):
        """Save mechanics plus its own template; thermal/loading history is external."""
        self._validate_state(state)
        arrays = {name: getattr(state, name) for name in self._array_shapes}
        metadata = {item.name: getattr(state, item.name) for item in fields(state)
                    if item.name not in self._array_shapes}
        metadata.update(version=VERSION, fingerprint=self.fingerprint, configuration=self.configuration)
        arrays.update(template_vertices=self.template_mesh.vertices, template_faces=self.template_mesh.faces,
                      metadata=np.array(json.dumps(metadata, allow_nan=False, sort_keys=True)))
        path = Path(path)
        temporary = Path(str(path)+".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(path)

    @classmethod
    def from_checkpoint(cls, path):
        """Restore a self-contained mechanics model and state, without a live owner.

        The returned model cannot invent its thermal/Maxwell history or force
        callback; these are supplied by the caller for subsequent physical steps.
        """
        try:
            with np.load(path, allow_pickle=False) as data:
                encoded = data["metadata"]
                if encoded.shape != () or encoded.dtype.kind not in "US":
                    raise ValueError("Moving checkpoint metadata must be a scalar JSON string")
                metadata = json.loads(str(encoded))
                if not isinstance(metadata, dict) or metadata.get("version") != VERSION:
                    raise ValueError("Unsupported moving mechanics checkpoint")
                config = metadata["configuration"]
                if not isinstance(config, dict) or set(config) != {
                        "parameters", "initial_radius_m", "poisson_ratio", "max_incremental_strain", "geometry_limits"}:
                    raise ValueError("Malformed moving mechanics configuration")
                vertices, faces = data["template_vertices"].copy(), data["template_faces"].copy()
            if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4
                    or vertices.dtype.kind not in "fiu" or not np.isfinite(vertices).all()
                    or faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 4
                    or faces.dtype.kind not in "iu" or np.any(faces < 0) or np.any(faces >= len(vertices))):
                raise ValueError("Malformed moving mechanics template arrays")
            neighbors, shared_edges = _build_topology(faces)
            template = SphereMesh(vertices, faces, np.empty((len(faces), 3)),
                np.empty(len(faces)), neighbors, shared_edges)
            template = rebuild_material_mesh(template, vertices)
            model = cls(template, config["initial_radius_m"], ContactParameters(**config["parameters"]),
                config["poisson_ratio"], config["max_incremental_strain"])
            return model, model.load_state(path)
        except (TypeError, KeyError, AttributeError, IndexError, RuntimeError) as exc:
            raise ValueError("Malformed moving mechanics checkpoint") from exc

    def load_state(self, path):
        """Validate and restore a complete state for this exact template and policy."""
        scalars = {item.name for item in fields(MovingTiedState)}-set(self._array_shapes)
        try:
            with np.load(path, allow_pickle=False) as data:
                if set(data.files) != set(self._array_shapes)|{"metadata", "template_vertices", "template_faces"}:
                    raise ValueError("Moving checkpoint has missing or unexpected arrays")
                encoded = data["metadata"]
                if encoded.shape != () or encoded.dtype.kind not in "US":
                    raise ValueError("Moving checkpoint metadata must be a scalar JSON string")
                metadata = json.loads(str(encoded))
                if not isinstance(metadata, dict) or set(metadata) != scalars|{"version", "fingerprint", "configuration"}:
                    raise ValueError("Moving checkpoint has missing or unexpected metadata")
                if (metadata.pop("version") != VERSION or metadata.pop("fingerprint") != self.fingerprint
                        or metadata.pop("configuration") != self.configuration
                        or not np.array_equal(data["template_vertices"], self.template_mesh.vertices)
                        or not np.array_equal(data["template_faces"], self.template_mesh.faces)):
                    raise ValueError("Moving checkpoint belongs to different geometry or parameters")
                state = MovingTiedState(**metadata, **{name: data[name].copy() for name in self._array_shapes})
            self._validate_state(state)
            return state
        except (TypeError, KeyError, AttributeError) as exc:
            raise ValueError("Malformed moving mechanics checkpoint") from exc
