"""Small-sliding, split-bank mechanics continued from a genesis fault snapshot.

The inherited heat, orbit, damage and Maxwell memory are frozen. This short
mechanical experiment uses reference-frame elastic increments, physical mantle
drag and paired cohesive/contact traces. It is not the long-term thermal
integrator, general collision search, finite sliding, or a mature plate model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import io
import json
import math
from numbers import Integral, Real
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .genesis_faults import load_fault_checkpoint
from .genesis_material import face_frames, material_column_depth
from .genesis_shell import Membrane, mantle_traction, maximum_total_strain
from .genesis_mobile import _external_force
from .genesis_seams import split_mesh, rebuild_seam_mesh
from .genesis_seam_diagnostics import seam_connectivity
from .genesis_contact_law import ContactLawParameters, contact_return_map, cohesive_damage, cohesive_dissipation
from .genesis_contact_geometry import build_interface_geometry
from .mesh import connected_components

CONTACT_VERSION = "genesis-contact-0.1"
SECONDS_PER_YEAR = 365.25*86400.


@dataclass(frozen=True)
class ContactParameters:
    basal_drag_pa_s_m: float = 1e14
    activation_damage: float = .65
    alignment_degrees: float = 35.
    max_added_strain: float = .005
    max_motion_edge_fraction: float = .02
    max_jump_edge_fraction: float = .02
    max_penetration_m: float = 5.
    max_step_jump_m: float = 2.
    max_frozen_duration_years: float = 10000.
    max_omitted_relaxation_fraction: float = .01
    min_step_years: float = .01
    equilibrium_tolerance: float = 1e-8
    max_newton_iterations: int = 40

    def validate(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"contact.{f.name} must be a finite positive number")
        if (not 0 < self.activation_damage < 1 or not 0 < self.alignment_degrees <= 90
                or self.max_added_strain > .01 or self.max_motion_edge_fraction > .05
                or self.max_jump_edge_fraction > .05 or self.equilibrium_tolerance > 1e-4
                or self.max_omitted_relaxation_fraction > .05
                or not isinstance(self.max_newton_iterations, Integral)):
            raise ValueError("Contact parameters exceed the small-sliding experiment limits")


@dataclass
class ContactState:
    elapsed_years: float
    displacement_m: np.ndarray
    plastic_slip_m: np.ndarray
    cumulative_slip_m: np.ndarray
    max_opening_m: np.ndarray
    interface_damage: np.ndarray
    traction_pa: np.ndarray
    friction_work_cell_j: np.ndarray
    viscous_work_cell_j: np.ndarray
    fracture_work_cell_j: np.ndarray
    shear_remainder_cell_j: np.ndarray
    drag_work_j: float = 0.
    external_work_j: float = 0.
    equilibrium_residual: float = 0.
    last_step_years: float = 0.
    accepted_steps: int = 0
    rejected_steps: int = 0
    stopped_reason: str | None = None


class _ContactRetry(RuntimeError):
    pass


def select_seams(mesh, state, parameters):
    """Sample persistent weak-plane orientations on existing material edges.

    This mesh-dependent extraction is explicit; it does not claim that the
    earlier diffuse damage field uniquely specifies a sharp crack network.
    Both adjacent material cells must have an activated weak plane and exceed
    the damage threshold. Either local normal may be unoriented (+/-).

    In particular, overlapping angular acceptance cones can cut two edge
    families of a triangular lattice and isolate two-cell rhombi even in a
    smooth direction field. These are potential contact interfaces, not a
    validated reconstruction of physical crack paths or a count of plates.
    Keep this legacy rule unchanged for exact historical checkpoint restart.
    """
    edges = np.asarray(mesh.shared_edges, dtype=int)
    a, b, u, v = edges.T
    normal = np.cross(mesh.vertices[u], mesh.vertices[v])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    plane = np.einsum("fij,fj->fi", face_frames(mesh), state.plane_normal)
    alignment = np.minimum(np.abs(np.einsum("fi,fi->f", normal, plane[a])),
                           np.abs(np.einsum("fi,fi->f", normal, plane[b])))
    active = (state.fault_active[a] & state.fault_active[b]
              & (state.damage[a] >= parameters.activation_damage)
              & (state.damage[b] >= parameters.activation_damage)
              & (alignment >= math.cos(math.radians(parameters.alignment_degrees))))
    return edges[active, 2:]


class ContactModel:
    @classmethod
    def from_fault_checkpoint(cls, path, parameters=None, law_parameters=None):
        data = Path(path).read_bytes()
        model = cls(data, parameters, law_parameters, str(Path(path).resolve()))
        return model, model.initial()

    def __init__(self, source_bytes, parameters=None, law_parameters=None, source_path="embedded", *, reference_cuts=None):
        self.parameters = parameters or ContactParameters()
        self.law_parameters = law_parameters or ContactLawParameters()
        self.parameters.validate()
        self.law_parameters.validate()
        self.source_bytes = bytes(source_bytes)
        self.source_path = source_path
        self.source_hash = hashlib.sha256(self.source_bytes).hexdigest()
        self.source_model, self.source_state, self.thermal_state, self.orbit, _ = load_fault_checkpoint(io.BytesIO(self.source_bytes))
        source = self.source_state
        if not source.membrane_established:
            raise ValueError("Contact continuation requires an established solid membrane")
        self.source_time_myr = source.time_myr
        self.radius_m = source.radius_km*1000
        original = self.source_model.mesh_for(source)
        # Explicit reference cuts are used by the evolving integrator's geometry
        # assembly. The original frozen experiment retains its onset checks.
        cuts = (select_seams(original, source, self.parameters) if reference_cuts is None
                else np.asarray(reference_cuts, dtype=np.int64).reshape(-1, 2))
        if len(cuts) == 0 and reference_cuts is None:
            raise ValueError("No aligned persistent weak edges in this fault snapshot")
        self.topology = split_mesh(original, cuts)
        if self.topology.mesh.vertex_count == original.vertex_count and reference_cuts is None:
            raise ValueError("Selected weak edges have no independent banks at this mesh resolution")
        mesh = self.topology.mesh
        self.membrane = Membrane(mesh, self.source_model.p.poisson_ratio)
        self.source_fields = self.source_model.fields(source, self.thermal_state)
        shell_p = self.source_model.p
        eta = np.clip(shell_p.viscosity_reference_pa_s*np.exp(np.clip(shell_p.activation_energy_j_mol/8.314462618
            *(1/np.maximum(self.source_fields["temperature_k"], 1)-1/shell_p.viscosity_reference_temperature_k), -60, 60)),
            shell_p.viscosity_min_pa_s, shell_p.viscosity_max_pa_s)
        self.min_maxwell_time_years = float(eta.min()/shell_p.young_modulus_pa/SECONDS_PER_YEAR)
        self.frozen_window_years = min(self.parameters.max_frozen_duration_years,
            -math.log1p(-self.parameters.max_omitted_relaxation_fraction)*self.min_maxwell_time_years)
        self.layer_mass_kg = source.layer_mass_kg.copy()
        self.column_enthalpy = source.column_enthalpy.copy()
        self.initial_mass_kg = float(self.layer_mass_kg.sum())
        self.column_energy_j = float(np.sum(self.layer_mass_kg*self.column_enthalpy))
        self.depth_m = self.source_fields["lid_thickness_km"]*1000
        if np.any(self.depth_m <= 0):
            raise ValueError("Every material face must have positive solid thickness")
        p = self.source_model.p
        degradation = p.residual_stiffness+(1-p.residual_stiffness)*(1-source.damage)**2
        self.elasticity = p.young_modulus_pa*degradation[:, None, None]*self.membrane.d
        self.reference_volume_m3 = mesh.areas_unit_sphere*self.radius_m**2*self.depth_m
        weight = mesh.areas_unit_sphere*self.depth_m
        local_k = np.einsum("fai,fab,fbj,f->fij", self.membrane.b, self.elasticity, self.membrane.b, weight)
        self.bulk_matrix = sparse.coo_matrix((local_k.ravel(), (self.membrane.rr, self.membrane.cc)),
                                             shape=(self.membrane.ndof, self.membrane.ndof)).tocsr()
        stress = np.einsum("fij,fj->fi", self.elasticity, source.elastic_strain)
        force = np.einsum("fai,fa,f->fi", self.membrane.b, stress, weight*self.radius_m)
        self.initial_bulk_force = np.zeros(self.membrane.ndof)
        np.add.at(self.initial_bulk_force, self.membrane.dofs.ravel(), force.ravel())
        # Use the identical inherited forcing projected in the original closed
        # mesh. Distribute each original nodal force by its incident face area
        # after splitting; no artificial per-fragment rotational constraint.
        original_membrane = Membrane(original, p.poisson_ratio)
        traction = mantle_traction(original, p, self.depth_m/1000)
        original_force = _external_force(original_membrane, traction, source.radius_km, p.young_modulus_pa)
        original_force *= p.young_modulus_pa*self.radius_m*1000
        vertex_area = np.zeros(mesh.vertex_count)
        np.add.at(vertex_area, mesh.faces.ravel(), np.repeat(mesh.areas_unit_sphere/3, 3))
        parent_area = np.bincount(self.topology.parent_vertex, weights=vertex_area, minlength=original.vertex_count)
        share = vertex_area/parent_area[self.topology.parent_vertex]
        self.external_force = np.zeros(self.membrane.ndof)
        self.external_force[:-1] = (original_force[:-1].reshape(-1, 2)[self.topology.parent_vertex]*share[:, None]).ravel()
        self.external_force[-1] = original_force[-1]
        self.drag_area_m2 = np.r_[np.repeat(vertex_area*self.radius_m**2, 2), 0.]
        self._build_interfaces(original)
        self.initial_elastic_energy_j = self._bulk_energy(np.zeros(self.membrane.ndof))
        self.component_count = len(connected_components(range(mesh.cell_count), mesh.neighbors))

    def _build_interfaces(self, original):
        geometry = build_interface_geometry(original, self.topology, self.radius_m,
                                            self.depth_m, self.membrane)
        for item in fields(geometry):
            setattr(self, item.name, getattr(geometry, item.name))
        self.interface_water = np.repeat(np.mean(
            self.source_state.water_access[self.topology.seam_faces], axis=1), 2)

    def initial(self):
        size = len(self.interface_area_m2)
        zero = lambda: np.zeros(size)
        return ContactState(0., np.zeros(self.membrane.ndof), zero(), zero(), zero(), zero(),
                            np.zeros((size, 2)), zero(), zero(), zero(), zero())

    def mesh_for(self, state):
        displacement = state.displacement_m[:-1].reshape(-1, 2)
        vertices = self.topology.mesh.vertices+np.einsum("vij,vj->vi", self.membrane.vertex_basis, displacement)/self.radius_m
        vertices /= np.linalg.norm(vertices, axis=1)[:, None]
        return rebuild_seam_mesh(self.topology, vertices)

    def _strain(self, q):
        return np.einsum("fai,fi->fa", self.membrane.b, q[self.membrane.dofs])/self.radius_m

    def _bulk_energy(self, q):
        elastic = self.source_state.elastic_strain+self._strain(q)
        return .5*float(np.sum(np.einsum("fi,fij,fj->f", elastic, self.elasticity, elastic)*self.reference_volume_m3))

    def _interface_energy(self, state):
        # Observing a state must not perform another viscous return mapping.
        gap, jump = (self.jump_operator@state.displacement_m).reshape(-1, 2).T
        normal = .5*self.law_parameters.normal_stiffness_pa_m*np.where(
            gap < 0, gap**2, (1-state.interface_damage)*gap**2)
        shear = .5*self.law_parameters.tangential_stiffness_pa_m*(jump-state.plastic_slip_m)**2
        return float(np.dot(self.interface_area_m2, normal+shear))

    def _geometric_jump(self, state, mesh=None):
        mesh = mesh or self.mesh_for(state)
        banks = mesh.vertices[self.topology.bank_vertices]
        difference = (banks[:, 1]-banks[:, 0]).reshape(-1, 3)*(self.radius_m+state.displacement_m[-1])
        return np.column_stack((np.einsum("pi,pi->p", difference, self.interface_normal),
                                np.einsum("pi,pi->p", difference, self.interface_tangent)))

    def _response(self, state, q, dt_s):
        jump = (self.jump_operator@q).reshape(-1, 2)
        return contact_return_map(jump[:, 0], jump[:, 1], state.plastic_slip_m,
            state.cumulative_slip_m, state.max_opening_m, dt_s, self.interface_water,
            self.law_parameters, old_damage=state.interface_damage)

    def _trial(self, state, target):
        if target > self.frozen_window_years*(1+1e-12):
            raise _ContactRetry("contact_frozen_state_limit")
        dt = target-state.elapsed_years
        dt_s = dt*SECONDS_PER_YEAR
        drag = self.drag_area_m2*self.parameters.basal_drag_pa_s_m/dt_s
        def evaluate(q):
            result = self._response(state, q, dt_s)
            contact_force = self.jump_operator.T@(result["traction_pa"]*self.interface_area_m2[:, None]).ravel()
            bulk = self.bulk_matrix@q+self.initial_bulk_force
            residual = bulk+contact_force+drag*(q-state.displacement_m)-self.external_force
            scale = max(np.linalg.norm(bulk), np.linalg.norm(contact_force), np.linalg.norm(self.external_force),
                        self.source_model.p.young_modulus_pa*self.radius_m*np.sum(self.topology.mesh.areas_unit_sphere*self.depth_m)*1e-10)
            return result, residual, float(np.linalg.norm(residual)/scale)
        q = state.displacement_m.copy()
        result, residual, norm = evaluate(q)
        for _ in range(self.parameters.max_newton_iterations):
            if norm <= self.parameters.equilibrium_tolerance:
                break
            local = sparse.coo_matrix(((result["tangent_pa_m"]*self.interface_area_m2[:, None, None]).ravel(),
                (self.contact_rows, self.contact_columns)), shape=(self.jump_operator.shape[0], self.jump_operator.shape[0])).tocsr()
            matrix = self.bulk_matrix+self.jump_operator.T@local@self.jump_operator+sparse.diags(drag)
            scaling = 1/np.sqrt(np.maximum(np.abs(matrix.diagonal()), 1.))
            diagonal = sparse.diags(scaling)
            correction = scaling*spsolve(diagonal@matrix@diagonal, -scaling*residual)
            if not np.isfinite(correction).all():
                raise _ContactRetry("contact_linear_solve_limit")
            accepted = False
            for power in range(18):
                candidate = q+correction*.5**power
                trial = evaluate(candidate)
                if trial[-1] < norm or trial[-1] <= self.parameters.equilibrium_tolerance:
                    q, (result, residual, norm), accepted = candidate, trial, True
                    break
            if not accepted:
                raise _ContactRetry("contact_equilibrium_limit")
        if norm > self.parameters.equilibrium_tolerance:
            raise _ContactRetry("contact_equilibrium_limit")
        jump = (self.jump_operator@q).reshape(-1, 2)
        previous = (self.jump_operator@state.displacement_m).reshape(-1, 2)
        if np.max(np.abs(jump-previous)) > self.parameters.max_step_jump_m:
            raise _ContactRetry("contact_jump_step_limit")
        if np.max(np.abs(jump)/np.repeat(self.edge_length_m, 2)[:, None]) > self.parameters.max_jump_edge_fraction:
            raise _ContactRetry("contact_small_sliding_limit")
        if max(0., -float(jump[:, 0].min())) > self.parameters.max_penetration_m:
            raise _ContactRetry("contact_penetration_limit")
        if (maximum_total_strain(self._strain(q)) > self.parameters.max_added_strain
                or np.linalg.norm(q[:-1].reshape(-1, 2), axis=1).max() > self.parameters.max_motion_edge_fraction*self.edge_length_m.min()
                or abs(q[-1])/self.radius_m > self.parameters.max_added_strain):
            raise _ContactRetry("contact_reference_geometry_limit")
        delta = q-state.displacement_m
        next_state = replace(state, elapsed_years=target, displacement_m=q,
            plastic_slip_m=result["plastic_slip_m"], cumulative_slip_m=result["cumulative_slip_m"],
            max_opening_m=result["max_opening_m"], interface_damage=result["damage"], traction_pa=result["traction_pa"],
            friction_work_cell_j=state.friction_work_cell_j+self.interface_area_m2*result["friction_work_j_m2"],
            viscous_work_cell_j=state.viscous_work_cell_j+self.interface_area_m2*result["viscous_work_j_m2"],
            fracture_work_cell_j=state.fracture_work_cell_j+self.interface_area_m2*result["fracture_work_j_m2"],
            shear_remainder_cell_j=state.shear_remainder_cell_j+self.interface_area_m2*result["shear_relaxation_remainder_j_m2"],
            drag_work_j=state.drag_work_j+float(np.dot(drag, delta**2)),
            external_work_j=state.external_work_j+float(self.external_force@delta),
            equilibrium_residual=norm, last_step_years=dt, accepted_steps=state.accepted_steps+1)
        try:
            mesh = self.mesh_for(next_state)
        except ValueError as exc:
            raise _ContactRetry("contact_mesh_quality_limit") from exc
        actual_normal = self._geometric_jump(next_state, mesh)[:, 0]
        if max(0., -float(actual_normal.min())) > self.parameters.max_penetration_m:
            raise _ContactRetry("contact_geometric_penetration_limit")
        return next_state

    def step(self, state, target_elapsed_years):
        if state.stopped_reason or not math.isfinite(target_elapsed_years) or target_elapsed_years <= state.elapsed_years:
            raise ValueError("Contact mechanical time must advance from an unstopped state")
        trial_dt = target_elapsed_years-state.elapsed_years
        while state.elapsed_years < target_elapsed_years-1e-10:
            try:
                result = self._trial(state, min(target_elapsed_years, state.elapsed_years+trial_dt))
            except _ContactRetry as exc:
                state = replace(state, rejected_steps=state.rejected_steps+1)
                if trial_dt <= self.parameters.min_step_years*(1+1e-9):
                    return replace(state, stopped_reason=str(exc))
                trial_dt = max(trial_dt/2, self.parameters.min_step_years)
                continue
            state = result
        return state

    def fields(self, state):
        mesh = self.mesh_for(state)
        banks = mesh.vertices[self.topology.bank_vertices]
        centers = banks.mean(axis=(1, 2))
        centers /= np.linalg.norm(centers, axis=1)[:, None]
        jump = self._geometric_jump(state, mesh).reshape(-1, 2, 2)
        depth = material_column_depth(mesh, (self.radius_m+state.displacement_m[-1])/1000,
                                      self.layer_mass_kg, self.source_model.p.density_kg_m3)
        return {"damage": self.source_state.damage, "water_access": self.source_state.water_access,
            "temperature_k": self.source_fields["temperature_k"], "lid_thickness_km": depth*(self.depth_m/1000)/self.source_fields["column_depth_km"],
            "column_depth_km": depth, "column_mass_kg": self.layer_mass_kg.sum(axis=1),
            "seam_centers_xyz": centers, "seam_gap_m": jump[:, :, 0], "seam_slip_m": jump[:, :, 1],
            "seam_plastic_slip_m": state.plastic_slip_m.reshape(-1, 2),
            "seam_damage": state.interface_damage.reshape(-1, 2),
            "seam_pressure_pa": np.maximum(-state.traction_pa[:, 0], 0).reshape(-1, 2)}

    def diagnostics(self, state):
        jump = (self.jump_operator@state.displacement_m).reshape(-1, 2)
        interface_energy = self._interface_energy(state)
        bulk_energy = self._bulk_energy(state.displacement_m)
        friction = float(state.friction_work_cell_j.sum()); viscous = float(state.viscous_work_cell_j.sum())
        fracture = float(state.fracture_work_cell_j.sum())
        mesh = self.mesh_for(state)
        geometric = self._geometric_jump(state, mesh)
        connectivity = seam_connectivity(self.topology, state.interface_damage < 1.)
        return {"source_time_myr": self.source_time_myr, "elapsed_years": state.elapsed_years,
            "frozen_window_years": self.frozen_window_years, "min_maxwell_time_years": self.min_maxwell_time_years,
            "omitted_maxwell_relaxation_fraction": -math.expm1(-state.elapsed_years/self.min_maxwell_time_years),
            "seam_count": len(self.topology.cut_edges), "split_vertex_count": len(self.topology.parent_vertex)-len(self.source_state.vertices),
            # Historical alias: these are cut-mesh components, not plates.
            "component_count": self.component_count, **connectivity,
            "max_opening_m": max(0., float(geometric[:, 0].max())),
            "max_penetration_m": max(0., -float(geometric[:, 0].min())), "max_abs_jump_m": float(np.abs(geometric[:, 1]).max()),
            "max_linearization_error_m": float(np.abs(geometric-jump).max()),
            "max_plastic_slip_m": float(np.abs(state.plastic_slip_m).max()),
            "max_interface_damage": float(state.interface_damage.max()),
            "fractured_endpoint_fraction": float(np.mean(state.interface_damage >= 1-1e-12)),
            "equilibrium_residual": state.equilibrium_residual, "max_added_strain": maximum_total_strain(self._strain(state.displacement_m)),
            "relative_mass_residual": (float(self.layer_mass_kg.sum())-self.initial_mass_kg)/self.initial_mass_kg,
            "relative_column_energy_residual": (float(np.sum(self.layer_mass_kg*self.column_enthalpy))-self.column_energy_j)/self.column_energy_j,
            "signed_area_coverage_residual": float((mesh.areas_unit_sphere.sum()-4*np.pi)/(4*np.pi)),
            "friction_work_j": friction, "viscous_work_j": viscous, "fracture_work_j": fracture,
            "shear_numerical_remainder_j": float(state.shear_remainder_cell_j.sum()), "drag_work_j": state.drag_work_j,
            "external_work_j": state.external_work_j, "bulk_elastic_energy_j": bulk_energy,
            "interface_elastic_energy_j": interface_energy,
            "mechanical_energy_remainder_j": self.initial_elastic_energy_j+state.external_work_j-bulk_energy-interface_energy-friction-viscous-fracture-state.drag_work_j,
            "accepted_steps": state.accepted_steps, "rejected_steps": state.rejected_steps,
            "last_step_years": state.last_step_years, "stopped_reason": state.stopped_reason}


def save_contact_checkpoint(path, model, state):
    arrays = {f.name: getattr(state, f.name) for f in fields(state) if isinstance(getattr(state, f.name), np.ndarray)}
    scalars = {f.name: getattr(state, f.name) for f in fields(state) if f.name not in arrays}
    parameters = {"contact": asdict(model.parameters), "law": asdict(model.law_parameters)}
    meta = {"format": CONTACT_VERSION, "parameters": parameters, "source_path": model.source_path,
        "source_hash": model.source_hash, "state": scalars,
        "parameter_hash": hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()}
    target = Path(path); temporary = Path(str(target)+".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, metadata=np.array(json.dumps(meta, allow_nan=False)),
            source_checkpoint=np.frombuffer(model.source_bytes, dtype=np.uint8), **arrays)
    temporary.replace(target)


def load_contact_checkpoint(path):
    try:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            source = data["source_checkpoint"]
            p = meta["parameters"]
            if (meta["format"] != CONTACT_VERSION or source.dtype != np.uint8 or source.ndim != 1
                    or hashlib.sha256(source.tobytes()).hexdigest() != meta["source_hash"]
                    or hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest() != meta["parameter_hash"]):
                raise ValueError("Contact checkpoint format or hash mismatch")
            model = ContactModel(source.tobytes(), ContactParameters(**p["contact"]), ContactLawParameters(**p["law"]), meta["source_path"])
            state = ContactState(**meta["state"], **{key: data[key].copy() for key in data.files if key not in {"metadata", "source_checkpoint"}})
        _validate_contact_state(model, state)
        return model, state
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Malformed contact checkpoint") from exc


def _validate_contact_state(model, state):
    count = len(model.interface_area_m2)
    shapes = {"displacement_m": (model.membrane.ndof,), "traction_pa": (count, 2)}
    initial = model.initial()
    array_names = {f.name for f in fields(state) if isinstance(getattr(initial, f.name), np.ndarray)}
    for f in fields(state):
        value = getattr(state, f.name)
        if f.name in array_names:
            if (not isinstance(value, np.ndarray) or value.dtype.kind not in "fiu"
                    or value.shape != shapes.get(f.name, (count,)) or not np.isfinite(value).all()):
                raise ValueError(f"Invalid contact history array {f.name}")
        elif f.name == "stopped_reason":
            if value is not None and not isinstance(value, str):
                raise ValueError("Invalid contact stop reason")
        elif isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"Invalid contact scalar {f.name}")
    if (state.elapsed_years < 0 or state.elapsed_years > model.frozen_window_years*(1+1e-12)
            or state.last_step_years < 0 or state.last_step_years > state.elapsed_years+1e-9
            or state.drag_work_j < 0 or state.equilibrium_residual < 0
            or any(not isinstance(x, Integral) or x < 0 for x in (state.accepted_steps, state.rejected_steps))
            or any(np.any(x < 0) for x in (state.cumulative_slip_m, state.max_opening_m, state.friction_work_cell_j,
                    state.viscous_work_cell_j, state.fracture_work_cell_j, state.shear_remainder_cell_j))
            or np.any(np.abs(state.plastic_slip_m)>state.cumulative_slip_m+1e-9)
            or np.any((state.interface_damage<0)|(state.interface_damage>1))):
        raise ValueError("Invalid contact history or work")
    if not np.allclose(state.interface_damage, cohesive_damage(state.max_opening_m, model.law_parameters), rtol=1e-12, atol=1e-12):
        raise ValueError("Contact damage disagrees with opening history")
    gap, jump = (model.jump_operator@state.displacement_m).reshape(-1, 2).T
    expected_normal = model.law_parameters.normal_stiffness_pa_m*np.where(gap<0, gap, (1-state.interface_damage)*gap)
    expected_shear = model.law_parameters.tangential_stiffness_pa_m*(jump-state.plastic_slip_m)
    if (np.any(state.max_opening_m+1e-9 < np.maximum(gap, 0))
            or not np.allclose(state.traction_pa, np.column_stack((expected_normal, expected_shear)), rtol=1e-10, atol=1e-6)
            or not np.allclose(state.fracture_work_cell_j,
                model.interface_area_m2*cohesive_dissipation(state.max_opening_m, model.law_parameters), rtol=1e-10, atol=1e-4)):
        raise ValueError("Contact traction, opening or fracture work history is inconsistent")
    if (state.elapsed_years > 0 and (state.accepted_steps == 0 or state.last_step_years <= 0
            or state.equilibrium_residual > model.parameters.equilibrium_tolerance)
            or maximum_total_strain(model._strain(state.displacement_m)) > model.parameters.max_added_strain
            or np.linalg.norm(state.displacement_m[:-1].reshape(-1, 2), axis=1).max() > model.parameters.max_motion_edge_fraction*model.edge_length_m.min()
            or abs(state.displacement_m[-1])/model.radius_m > model.parameters.max_added_strain
            or np.max(np.abs(np.column_stack((gap, jump)))/np.repeat(model.edge_length_m, 2)[:, None]) > model.parameters.max_jump_edge_fraction
            or max(0., -float(gap.min())) > model.parameters.max_penetration_m):
        raise ValueError("Contact checkpoint exceeds acceptance limits")
    geometric = model._geometric_jump(state)
    if max(0., -float(geometric[:, 0].min())) > model.parameters.max_penetration_m:
        raise ValueError("Contact geometric penetration exceeds tolerance")
    if state.elapsed_years == 0 and (np.any(state.displacement_m != 0) or state.accepted_steps != 0
            or state.drag_work_j != 0 or state.external_work_j != 0
            or any(np.any(getattr(state, key) != 0) for key in ("plastic_slip_m", "cumulative_slip_m", "max_opening_m", "friction_work_cell_j", "viscous_work_cell_j", "fracture_work_cell_j", "shear_remainder_cell_j"))):
        raise ValueError("Unadvanced contact state cannot contain mechanical history")
