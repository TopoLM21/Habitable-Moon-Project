"""Passive conductive columns and a freely contracting spherical membrane.

The thermal genesis supplies boundary temperatures to this one-way experiment.
Column enthalpy has a separate boundary-flux ledger: it is not added to the
already-counted global mantle energy. Triangular membrane FEM resolves local
compatibility. Damage represents distributed tensile weakness, not mobile plates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .genesis import GenesisParameters, GenesisState, SECONDS_PER_MYR, parameter_hash, diagnose
from .genesis_checkpoint_compat import MODEL_VERSION, require_thermal_model_version
from .mesh import SphereMesh, build_icosphere
from .basal_coupling import prescribed_vertex_traction

SHELL_VERSION = "genesis-shell-0.1"


@dataclass(frozen=True)
class ShellParameters:
    subdivisions: int = 3
    seed: int = 20260920
    column_layers: int = 32
    column_depth_km: float = 40.0
    conductivity_w_m_k: float = 3.6
    density_kg_m3: float = 3000.0
    initial_temperature_anomaly_k: float = 120.0
    convective_traction_pa: float = 20000.0
    traction_coupling_depth_km: float = 2.0
    anomaly_depth_km: float = 8.0
    young_modulus_pa: float = 6e10
    poisson_ratio: float = 0.25
    linear_expansion_per_k: float = 1e-5
    viscosity_reference_pa_s: float = 1e24
    viscosity_reference_temperature_k: float = 1000.0
    activation_energy_j_mol: float = 150000.0
    viscosity_min_pa_s: float = 1e17
    viscosity_max_pa_s: float = 1e28
    tensile_strength_pa: float = 12e6
    damage_timescale_myr: float = 0.01
    cold_healing_timescale_myr: float = 100.0
    hot_healing_timescale_myr: float = 0.01
    residual_stiffness: float = 0.02
    damage_threshold: float = 0.65
    min_load_bearing_thickness_km: float = 0.05
    max_total_strain: float = 0.05

    def validate(self, thermal: GenesisParameters | None = None, *, max_subdivisions: int = 4) -> None:
        # The membrane FEM retains its bounded 1..4 mesh range. The cheap
        # starter/continuation loading law can explicitly opt into finer grids.
        if isinstance(max_subdivisions, bool) or not isinstance(max_subdivisions, Integral) or not 1 <= max_subdivisions <= 8:
            raise ValueError("Maximum supported subdivisions must be an integer in 1..8")
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"shell.{f.name} must be a finite number")
            if value < 0 or (value == 0 and f.name not in {"seed", "initial_temperature_anomaly_k", "linear_expansion_per_k", "convective_traction_pa"}):
                raise ValueError(f"shell.{f.name} is outside its positive range")
        if not all(isinstance(getattr(self, name), Integral) for name in ("subdivisions", "seed", "column_layers")):
            raise ValueError("Shell subdivisions, seed and column_layers must be integers")
        if not 1 <= self.subdivisions <= max_subdivisions:
            raise ValueError(f"Shell subdivisions must be in 1..{max_subdivisions}")
        if int(self.seed) != self.seed or int(self.column_layers) != self.column_layers or not 8 <= self.column_layers <= 128:
            raise ValueError("Shell seed/layers must be integers; layers must be 8..128")
        if not 0 < self.poisson_ratio < 0.49 or not 0 < self.residual_stiffness <= 1 or not 0 < self.damage_threshold < 1:
            raise ValueError("Invalid shell elastic or damage parameters")
        if self.viscosity_min_pa_s > self.viscosity_max_pa_s:
            raise ValueError("Invalid viscosity bounds")
        if self.min_load_bearing_thickness_km >= self.column_depth_km:
            raise ValueError("Load-bearing threshold must be shallower than the column")
        if self.max_total_strain > 0.1:
            raise ValueError("Fixed-geometry membrane cannot support a strain limit above 0.1")
        if thermal and thermal.initial_temperature_k - self.initial_temperature_anomaly_k < thermal.liquidus_k:
            raise ValueError("Initial temperature anomaly must leave every column fully molten")


def shell_parameters_from_config(config: dict) -> ShellParameters:
    section = dict(config.get("genesis_shell", {}))
    if section.pop("schema_version", 1) != 1:
        raise ValueError("Unsupported genesis_shell schema")
    unknown = set(section) - {f.name for f in fields(ShellParameters)}
    if unknown:
        raise ValueError(f"Unknown shell parameters: {sorted(unknown)}")
    p = ShellParameters(**section)
    p.validate()
    return p


def rock_enthalpy(t, p: GenesisParameters):
    phi = np.clip((np.asarray(t) - p.solidus_k) / (p.liquidus_k - p.solidus_k), 0, 1)
    return p.silicate_heat_capacity_j_kg_k * np.asarray(t) + p.silicate_latent_heat_j_kg * phi


def rock_temperature(h, p: GenesisParameters):
    cp, latent = p.silicate_heat_capacity_j_kg_k, p.silicate_latent_heat_j_kg
    width = p.liquidus_k - p.solidus_k
    h = np.asarray(h)
    return np.where(h <= cp * p.solidus_k, h / cp,
                    np.where(h >= cp * p.liquidus_k + latent, (h-latent)/cp,
                             (h + latent*p.solidus_k/width) / (cp + latent/width)))


def smooth_anomaly(mesh: SphereMesh, seed: int) -> np.ndarray:
    """A low-degree physical field, sampled identically on different meshes."""
    rng = np.random.default_rng(int(seed))
    matrix = rng.normal(size=(3, 3))
    matrix = (matrix + matrix.T) / 2
    matrix -= np.eye(3) * np.trace(matrix) / 3
    x = mesh.centroids
    value = np.einsum("ni,ij,nj->n", x, matrix, x)
    dipole = rng.normal(size=3)
    value += 0.25 * (x @ dipole)
    value -= np.average(value, weights=mesh.areas_unit_sphere)
    # Bound independent of mesh extrema so refinement does not rescale forcing.
    bound = np.linalg.norm(matrix, 2) + 0.25 * np.linalg.norm(dipole)
    # The exact sampled bound also guarantees a fully molten initial state.
    return value / max(bound, float(np.max(np.abs(value))), 1e-12)


def maxwell_factors(dt_s: float, tau_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = dt_s / np.maximum(tau_s, 1e-300)
    r = np.exp(-x)
    b = np.ones_like(x)
    np.divide(-np.expm1(-x), x, out=b, where=x > 1e-10)
    return r, b


def evolve_damage(old, tensile_pa, strength_pa, temperature_k, dt_myr: float, p: ShellParameters):
    loading = np.maximum(np.asarray(tensile_pa) / strength_pa - 1, 0)**2 / p.damage_timescale_myr
    hot = np.clip((np.asarray(temperature_k) - 700) / 650, 0, 1)
    healing = 1/p.cold_healing_timescale_myr + hot**4 / p.hot_healing_timescale_myr
    rate = loading + healing
    equilibrium = loading / rate
    return equilibrium + (np.asarray(old) - equilibrium) * np.exp(-rate * dt_myr)


class Membrane:
    """Constant-strain triangular FEM, tangent vertex DOFs + free radius DOF.

    Coordinates/displacements are scaled by radius; a uniform radial strain
    gives [e, e, 0] in every face. Three Lagrange constraints remove rotations.
    Small residual stiffness regularizes damaged material; cracks are diffuse.
    """
    def __init__(self, mesh: SphereMesh, poisson_ratio: float):
        self.mesh = mesh
        n = mesh.vertices
        ref = np.tile([0., 0., 1.], (len(n), 1))
        ref[np.abs(n[:, 2]) > 0.9] = [0., 1., 0.]
        t1 = np.cross(ref, n)
        t1 /= np.linalg.norm(t1, axis=1)[:, None]
        t2 = np.cross(n, t1)
        self.vertex_basis = np.stack([t1, t2], axis=2)
        xyz = mesh.vertices[mesh.faces]
        e1 = xyz[:, 1] - xyz[:, 0]
        e1 /= np.linalg.norm(e1, axis=1)[:, None]
        normal = np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0])
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        e2 = np.cross(normal, e1)
        face_basis = np.stack([e1, e2], axis=1)
        xy = np.einsum("fai,fbi->fab", xyz, face_basis)
        design = np.concatenate([np.ones((len(xy), 3, 1)), xy], axis=2)
        grad = np.linalg.inv(design)[:, 1:, :]
        projection = np.einsum("fai,fbij->fbaj", face_basis, self.vertex_basis[mesh.faces])
        self.b = np.zeros((mesh.cell_count, 3, 7))
        for vertex in range(3):
            dx, dy = grad[:, 0, vertex], grad[:, 1, vertex]
            self.b[:, 0, 2*vertex:2*vertex+2] = dx[:, None] * projection[:, vertex, 0]
            self.b[:, 1, 2*vertex:2*vertex+2] = dy[:, None] * projection[:, vertex, 1]
            self.b[:, 2, 2*vertex:2*vertex+2] = dy[:, None]*projection[:, vertex, 0] + dx[:, None]*projection[:, vertex, 1]
        self.b[:, :2, -1] = 1
        nu = poisson_ratio
        self.d = np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]]) / (1-nu**2)
        self.ndof = 2*mesh.vertex_count + 1
        self.dofs = np.column_stack([(2*mesh.faces[:, :, None] + np.arange(2)).reshape(-1, 6), np.full(mesh.cell_count, self.ndof-1)])
        self.ki = np.einsum("fai,ab,fbj->fij", self.b, self.d, self.b)
        self.rr = np.repeat(self.dofs, 7, axis=1).ravel()
        self.cc = np.tile(self.dofs, (1, 7)).ravel()
        rotations = []
        for axis in np.eye(3):
            rot = np.cross(np.broadcast_to(axis, n.shape), n)
            vector = np.zeros(self.ndof)
            vector[:-1] = np.einsum("nij,ni->nj", self.vertex_basis, rot).ravel()
            rotations.append(vector / np.linalg.norm(vector))
        self.constraints = sparse.csr_matrix(np.array(rotations))
        self.last_displacement_rad = np.zeros((mesh.vertex_count, 2))

    def solve(self, eigenstrain, thickness_km, stiffness, young_pa: float, traction_pa=None, radius_km=1.):
        weight = self.mesh.areas_unit_sphere * np.maximum(thickness_km, 1e-4) * np.maximum(stiffness, 1e-9)
        k = sparse.coo_matrix(( (self.ki*weight[:, None, None]).ravel(), (self.rr, self.cc)), shape=(self.ndof, self.ndof)).tocsr()
        local_force = np.einsum("fai,ab,fb,f->fi", self.b, self.d, eigenstrain, weight)
        rhs = np.zeros(self.ndof)
        np.add.at(rhs, self.dofs.ravel(), local_force.ravel())
        if traction_pa is not None:
            # External work is integral(traction dot displacement dA). With
            # u/R DOFs and K scaled by E*R^2, load scales as traction*R/E.
            vertex_area = np.zeros(self.mesh.vertex_count)
            np.add.at(vertex_area, self.mesh.faces.ravel(), np.repeat(self.mesh.areas_unit_sphere/3, 3))
            force = np.zeros(self.ndof)
            force[:-1] = (np.einsum("nij,ni->nj", self.vertex_basis, traction_pa)*vertex_area[:, None]*radius_km/young_pa).ravel()
            # Remove any discrete rigid-rotation torque from prescribed forcing.
            c = self.constraints.toarray()
            force -= c.T @ np.linalg.solve(c@c.T, c@force)
            rhs += force
        c = self.constraints
        system = sparse.bmat([[k, c.T], [c, None]], format="csc")
        solution = spsolve(system, np.r_[rhs, [0., 0., 0.]])
        if not np.isfinite(solution).all():
            raise RuntimeError("Shell mechanical equilibrium did not converge")
        displacement = solution[:self.ndof]
        self.last_displacement_rad = displacement[:-1].reshape(-1, 2).copy()
        strain = np.einsum("fai,fi->fa", self.b, displacement[self.dofs])
        stress = np.einsum("ab,fb->fa", self.d, strain-eigenstrain) * (young_pa*np.asarray(stiffness))[:, None]
        residual = np.linalg.norm(system @ solution-np.r_[rhs, [0., 0., 0.]]) / max(np.linalg.norm(rhs), 1e-15)
        return strain, stress, float(residual), float(displacement[-1])

    def solve_correction(self, stress_pa, thickness_km, tangent_pa,
                         young_pa: float, traction_pa=None, radius_km=1.):
        """Newton strain correction for a supplied constitutive stress/tangent.

        Stress uses [sigma_xx, sigma_yy, sigma_xy], and the tangent differentiates
        this vector with respect to [epsilon_xx, epsilon_yy, gamma_xy]. The
        current full traction is balanced against the current internal stress.
        The tangent need not be symmetric, as in nonassociated friction laws.
        """
        for name, value in (("young_pa", young_pa), ("radius_km", radius_km)):
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")

        def checked_array(value, shape, name):
            try:
                array = np.asarray(value, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a finite array of shape {shape}") from exc
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"{name} must be a finite array of shape {shape}")
            return array

        count = self.mesh.cell_count
        stress = checked_array(stress_pa, (count, 3), "stress_pa")
        depth = checked_array(thickness_km, (count,), "thickness_km")
        tangent = checked_array(tangent_pa, (count, 3, 3), "tangent_pa")
        if np.any(depth < 0):
            raise ValueError("thickness_km must be nonnegative")
        traction = (None if traction_pa is None else checked_array(
            traction_pa, (self.mesh.vertex_count, 3), "traction_pa"))

        area = self.mesh.areas_unit_sphere
        weight = area * np.maximum(depth, 1e-4)
        local_k = np.einsum("fai,fab,fbj,f->fij", self.b,
                            tangent / young_pa, self.b, weight)
        k = sparse.coo_matrix((local_k.ravel(), (self.rr, self.cc)),
                              shape=(self.ndof, self.ndof)).tocsr()
        local_force = -np.einsum("fai,fa,f->fi", self.b,
                                 stress / young_pa, area * depth)
        rhs = np.zeros(self.ndof)
        np.add.at(rhs, self.dofs.ravel(), local_force.ravel())
        if traction is not None:
            vertex_area = np.zeros(self.mesh.vertex_count)
            np.add.at(vertex_area, self.mesh.faces.ravel(), np.repeat(area / 3, 3))
            force = np.zeros(self.ndof)
            force[:-1] = (np.einsum("nij,ni->nj", self.vertex_basis, traction)
                          * vertex_area[:, None] * radius_km / young_pa).ravel()
            c = self.constraints.toarray()
            force -= c.T @ np.linalg.solve(c @ c.T, c @ force)
            rhs += force
        c = self.constraints
        system = sparse.bmat([[k, c.T], [c, None]], format="csc")
        constrained_rhs = np.r_[rhs, [0., 0., 0.]]
        solution = spsolve(system, constrained_rhs)
        if not np.isfinite(solution).all():
            raise RuntimeError("Shell mechanical correction did not converge")
        displacement = solution[:self.ndof]
        strain = np.einsum("fai,fi->fa", self.b, displacement[self.dofs])
        residual = np.linalg.norm(system @ solution - constrained_rhs) / max(
            np.linalg.norm(rhs), 1e-15)
        if not np.isfinite(strain).all() or not math.isfinite(residual):
            raise RuntimeError("Shell mechanical correction did not converge")
        self.last_displacement_rad = displacement[:-1].reshape(-1, 2).copy()
        return strain, float(residual), float(displacement[-1])


@dataclass
class ShellState:
    time_myr: float
    column_enthalpy: np.ndarray
    lid_thickness_km: np.ndarray
    eigenstrain: np.ndarray
    strain: np.ndarray
    stress_pa: np.ndarray
    damage: np.ndarray
    peak_tensile_pa: np.ndarray
    first_fracture_time_myr: float | None
    initial_column_energy_j: float
    boundary_energy_j: float = 0.0
    equilibrium_residual: float = 0.0
    radial_strain: float = 0.0
    stopped_reason: str | None = None


def initialize_shell(mesh: SphereMesh, p: ShellParameters, thermal: GenesisParameters) -> ShellState:
    p.validate(thermal)
    z = (np.arange(p.column_layers)+0.5) * p.column_depth_km/p.column_layers
    initial = thermal.initial_temperature_k + p.initial_temperature_anomaly_k*smooth_anomaly(mesh, p.seed)[:, None]*np.exp(-z[None, :]/p.anomaly_depth_km)
    enthalpy = rock_enthalpy(initial, thermal)
    dz = p.column_depth_km*1000/p.column_layers
    areas = mesh.physical_cell_areas_km2(thermal.radius_km)*1e6
    total = float(np.sum(enthalpy*areas[:, None])*p.density_kg_m3*dz)
    return ShellState(0., enthalpy, np.zeros(mesh.cell_count), np.zeros((mesh.cell_count, 3)),
                      np.zeros((mesh.cell_count, 3)), np.zeros((mesh.cell_count, 3)),
                      np.zeros(mesh.cell_count), np.zeros(mesh.cell_count), None, total)


def mantle_traction(mesh: SphereMesh, p: ShellParameters, thickness_km):
    """Prescribed, smooth potential-flow basal shear; not a solved mantle flow."""
    return prescribed_vertex_traction(mesh, seed=p.seed,
        convective_traction_pa=p.convective_traction_pa, thickness_km=thickness_km,
        traction_coupling_depth_km=p.traction_coupling_depth_km)


def lid_geometry(column_temperature, surface_temperature, mantle_temperature, p: ShellParameters, thermal: GenesisParameters):
    """Depth of the first solidus crossing connected to the surface, interpolated."""
    n = len(column_temperature)
    dz = p.column_depth_km/p.column_layers
    z = np.r_[0., (np.arange(p.column_layers)+0.5)*dz, p.column_depth_km]
    profile = np.column_stack([np.full(n, surface_temperature), column_temperature, np.full(n, mantle_temperature)])
    liquid = profile > thermal.solidus_k
    first = np.argmax(liquid, axis=1)
    first[~np.any(liquid, axis=1)] = len(z)-1
    previous = np.maximum(first-1, 0)
    idx = np.arange(n)
    low, high = profile[idx, previous], profile[idx, first]
    f = np.divide(thermal.solidus_k-low, high-low, out=np.zeros(n), where=high != low)
    depth = z[previous] + np.clip(f, 0, 1)*(z[first]-z[previous])
    depth[liquid[:, 0]] = 0
    depth[~np.any(liquid, axis=1)] = p.column_depth_km
    span = np.clip(depth[:, None]-z[:-1], 0, np.diff(z))
    end_temp = profile[:, :-1] + (profile[:, 1:]-profile[:, :-1])*span/np.diff(z)
    integral = np.sum(0.5*(profile[:, :-1]+end_temp)*span, axis=1)
    mean = np.divide(integral, depth, out=column_temperature[:, 0].copy(), where=depth>0)
    return depth, mean


def principal_tensile(stress):
    average = (stress[:, 0]+stress[:, 1])/2
    radius = np.hypot((stress[:, 0]-stress[:, 1])/2, stress[:, 2])
    return np.maximum(average+radius, 0.)


def maximum_total_strain(strain):
    """Principal tensor strains: the third FEM component is engineering shear."""
    average = (strain[:, 0]+strain[:, 1])/2
    radius = np.hypot((strain[:, 0]-strain[:, 1])/2, strain[:, 2]/2)
    return float(np.max(np.maximum(np.abs(average+radius), np.abs(average-radius))))


def advance_shell(state: ShellState, mesh: SphereMesh, membrane: Membrane,
                  p: ShellParameters, thermal: GenesisParameters, target_myr: float,
                  surface_before_k: float, surface_after_k: float,
                  mantle_before_k: float, mantle_after_k: float, *, damage_update=None) -> ShellState:
    dt_myr = target_myr-state.time_myr
    if not math.isfinite(dt_myr) or dt_myr <= 0 or state.stopped_reason:
        raise ValueError("Shell target must advance time")
    dt = dt_myr*SECONDS_PER_MYR
    dz = p.column_depth_km*1000/p.column_layers
    old_temperature = rock_temperature(state.column_enthalpy, thermal)
    h = state.column_enthalpy.copy()
    cp = thermal.silicate_heat_capacity_j_kg_k
    stable = 0.20*p.density_kg_m3*cp*dz**2/p.conductivity_w_m_k
    count = max(1, math.ceil(dt/stable))
    ds = dt/count
    areas = mesh.physical_cell_areas_km2(thermal.radius_km)*1e6
    boundary = state.boundary_energy_j
    for i in range(count):
        fraction = (i+0.5)/count
        ts = surface_before_k + fraction*(surface_after_k-surface_before_k)
        tm = mantle_before_k + fraction*(mantle_after_k-mantle_before_k)
        temp = rock_temperature(h, thermal)
        flux = np.empty((mesh.cell_count, p.column_layers+1))
        flux[:, 0] = 2*p.conductivity_w_m_k*(ts-temp[:, 0])/dz
        flux[:, -1] = 2*p.conductivity_w_m_k*(temp[:, -1]-tm)/dz
        flux[:, 1:-1] = p.conductivity_w_m_k*(temp[:, :-1]-temp[:, 1:])/dz
        h += (flux[:, :-1]-flux[:, 1:])*ds/(p.density_kg_m3*dz)
        boundary += float(np.dot(areas, flux[:, 0]-flux[:, -1])*ds)
    new_temperature = rock_temperature(h, thermal)
    depth, mean = lid_geometry(new_temperature, surface_after_k, mantle_after_k, p, thermal)
    active = depth >= p.min_load_bearing_thickness_km
    old_active = state.lid_thickness_km >= p.min_load_bearing_thickness_km
    shared_depth = np.minimum(depth, state.lid_thickness_km)
    weights = np.clip(shared_depth[:, None]*1000-np.arange(p.column_layers)*dz, 0, dz)
    delta_t = np.divide(np.sum((new_temperature-old_temperature)*weights, axis=1), weights.sum(axis=1), out=np.zeros(mesh.cell_count), where=weights.sum(axis=1)>0)
    retained = np.divide(shared_depth, depth, out=np.zeros_like(depth), where=active) * old_active
    eta = np.clip(p.viscosity_reference_pa_s*np.exp(np.clip(p.activation_energy_j_mol/8.314462618*(1/np.maximum(mean, 1)-1/p.viscosity_reference_temperature_k), -60, 60)), p.viscosity_min_pa_s, p.viscosity_max_pa_s)
    r, b = maxwell_factors(dt, eta/p.young_modulus_pa)
    effective_b = retained*b + (1-retained)
    thermal_strain = np.zeros_like(state.strain)
    thermal_strain[:, :2] = (p.linear_expansion_per_k*delta_t)[:, None]
    eigen = state.strain + (retained/effective_b)[:, None] * (b[:, None]*thermal_strain-r[:, None]*(state.strain-state.eigenstrain))
    damage = state.damage * retained
    stiffness = (p.residual_stiffness+(1-p.residual_stiffness)*(1-damage)**2)*effective_b
    stiffness[~active] = 1e-9
    traction = mantle_traction(mesh, p, np.where(active, depth, 0.))
    if np.any(active):
        strain, stress, residual, radial = membrane.solve(eigen, depth, stiffness, p.young_modulus_pa, traction, thermal.radius_km)
        stress[~active] = 0
    else:
        strain, stress, residual, radial = np.zeros_like(state.strain), np.zeros_like(state.stress_pa), 0., 0.
        membrane.last_displacement_rad.fill(0.)
    tensile = principal_tensile(stress)
    # Surface initiation criterion; does not imply a through-going fault.
    if damage_update is None:
        damage = evolve_damage(damage, tensile, p.tensile_strength_pa, mean, dt_myr, p)
    else:
        damage = damage_update(damage, stress, mean, depth, retained, dt_myr)
        if damage.shape != state.damage.shape or not np.isfinite(damage).all() or np.any((damage < 0) | (damage > 1)):
            raise ValueError("Coupled damage update must return finite fractions in [0, 1]")
    damage[~active] = 0
    new_stiffness = (p.residual_stiffness+(1-p.residual_stiffness)*(1-damage)**2)*effective_b
    new_stiffness[~active] = 1e-9
    if np.any(active) and np.max(np.abs(new_stiffness-stiffness)) > 1e-8:
        strain, stress, residual, radial = membrane.solve(eigen, depth, new_stiffness, p.young_modulus_pa, traction, thermal.radius_km)
        stress[~active] = 0
    tensile = np.maximum(tensile, principal_tensile(stress))
    # Store the relaxed physical strain, not the effective integration modulus.
    final_eigen = strain-effective_b[:, None]*(strain-eigen)
    final_eigen[~active] = strain[~active]
    first = state.first_fracture_time_myr
    if first is None and np.any(damage >= p.damage_threshold):
        first = target_myr  # explicitly bracketed by the shell integration step
    stopped = "shell_small_strain_limit" if np.any(active) and maximum_total_strain(strain[active]) > p.max_total_strain else None
    return ShellState(target_myr, h, depth, final_eigen, strain, stress, damage,
                      np.maximum(state.peak_tensile_pa, tensile), first, state.initial_column_energy_j,
                      boundary, residual, radial, stopped)


def shell_fields(state: ShellState, mesh: SphereMesh, p: ShellParameters, thermal: GenesisParameters, surface_k: float, mantle_k: float):
    temp = rock_temperature(state.column_enthalpy, thermal)
    depth, mean = lid_geometry(temp, surface_k, mantle_k, p, thermal)
    solid = 1-np.clip((temp[:, 0]-thermal.solidus_k)/(thermal.liquidus_k-thermal.solidus_k), 0, 1)
    damaged = state.damage >= p.damage_threshold
    edges = np.array(mesh.shared_edges, dtype=int)
    # Outline of damaged patches; these mesh edges are NOT displacement jumps.
    failed = damaged[edges[:, 0]] != damaged[edges[:, 1]]
    return {"temperature_k": mean, "solid_fraction": solid, "lid_thickness_km": depth,
            "tensile_stress_mpa": principal_tensile(state.stress_pa)/1e6, "damage": state.damage,
            "failed_edges": failed, "peak_tensile_stress_mpa": state.peak_tensile_pa/1e6}


def diagnose_shell(state: ShellState, mesh: SphereMesh, p: ShellParameters, thermal: GenesisParameters, surface_k: float, mantle_k: float) -> dict:
    data = shell_fields(state, mesh, p, thermal, surface_k, mantle_k)
    area = mesh.areas_unit_sphere
    edges = np.array(mesh.shared_edges, dtype=int)
    lengths = np.arccos(np.clip(np.sum(mesh.vertices[edges[:, 2]]*mesh.vertices[edges[:, 3]], axis=1), -1, 1))*thermal.radius_km
    active = state.lid_thickness_km >= p.min_load_bearing_thickness_km
    intact = active & (state.damage < p.damage_threshold)
    remaining = set(np.flatnonzero(intact).tolist())
    regions = 0
    while remaining:
        regions += 1
        todo = [remaining.pop()]
        while todo:
            for neighbor in mesh.neighbors[todo.pop()]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    todo.append(neighbor)
    total = float(np.sum(state.column_enthalpy*mesh.physical_cell_areas_km2(thermal.radius_km)[:, None]*1e6)*p.density_kg_m3*p.column_depth_km*1000/p.column_layers)
    residual = total-state.initial_column_energy_j-state.boundary_energy_j
    return {"time_myr": state.time_myr, "first_fracture_time_myr": state.first_fracture_time_myr,
            "solid_surface_fraction": float(np.average(active, weights=area)),
            "mean_lid_thickness_km": float(np.average(state.lid_thickness_km, weights=area)),
            "max_lid_thickness_km": float(np.max(state.lid_thickness_km)),
            "damaged_area_fraction": float(np.average(state.damage>=p.damage_threshold, weights=area)),
            "mean_damage": float(np.average(state.damage, weights=area)),
            "max_tensile_stress_mpa": float(np.max(data["tensile_stress_mpa"])),
            "peak_tensile_stress_mpa": float(np.max(state.peak_tensile_pa)/1e6),
            "failed_edge_fraction": float(np.mean(data["failed_edges"])),
            "cracked_length_km": float(np.sum(lengths[data["failed_edges"]])),
            "intact_region_count": regions, "relative_column_energy_residual": residual/state.initial_column_energy_j,
            "mechanical_equilibrium_residual": state.equilibrium_residual, "radial_strain": state.radial_strain,
            "max_total_membrane_strain": maximum_total_strain(state.strain[active]) if np.any(active) else 0.,
            "stopped_reason": state.stopped_reason}


def save_shell_checkpoint(path: Path, state: ShellState, thermal_state: GenesisState,
                          p: ShellParameters, thermal: GenesisParameters, controls: dict, provenance: dict):
    """One atomically replaced NPZ contains metadata and both complete states."""
    arrays = {f.name: getattr(state, f.name) for f in fields(state) if isinstance(getattr(state, f.name), np.ndarray)}
    scalars = {f.name: getattr(state, f.name) for f in fields(state) if f.name not in arrays}
    meta = {"format": SHELL_VERSION, "thermal_model_version": MODEL_VERSION,
            "shell_parameters": asdict(p), "thermal_parameters": asdict(thermal),
            "thermal_parameter_hash": parameter_hash(thermal), "shell_state": scalars,
            "thermal_state": asdict(thermal_state), "controls": controls, "provenance": provenance}
    meta["shell_parameter_hash"] = hashlib.sha256(json.dumps(asdict(p), sort_keys=True).encode()).hexdigest()
    temporary = Path(str(path)+".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, metadata=np.array(json.dumps(meta, allow_nan=False)), **arrays)
    temporary.replace(path)


def load_shell_checkpoint(path: Path):
    try:
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata"]))
            if meta["format"] != SHELL_VERSION:
                raise ValueError("Unsupported shell checkpoint")
            require_thermal_model_version(meta)
            p, thermal = ShellParameters(**meta["shell_parameters"]), GenesisParameters(**meta["thermal_parameters"])
            p.validate(thermal)
            thermal.validate()
            if parameter_hash(thermal) != meta["thermal_parameter_hash"] or hashlib.sha256(json.dumps(asdict(p), sort_keys=True).encode()).hexdigest() != meta["shell_parameter_hash"]:
                raise ValueError("Shell checkpoint parameter hash mismatch")
            state = ShellState(**meta["shell_state"], **{k: archive[k].copy() for k in archive.files if k != "metadata"})
        thermal_state = GenesisState(**meta["thermal_state"])
        n = 20*4**p.subdivisions
        if state.time_myr != thermal_state.time_myr or not math.isfinite(state.time_myr) or state.time_myr < 0:
            raise ValueError("Shell/thermal checkpoint clocks differ")
        for key, shape in {"column_enthalpy": (n, p.column_layers), "lid_thickness_km": (n,), "damage": (n,),
                           "peak_tensile_pa": (n,), "eigenstrain": (n, 3), "strain": (n, 3), "stress_pa": (n, 3)}.items():
            a = getattr(state, key)
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError(f"Invalid shell checkpoint array {key}")
        if np.any((state.damage<0)|(state.damage>1)) or np.any(state.column_enthalpy <= 0):
            raise ValueError("Nonphysical shell checkpoint")
        for name in ("initial_column_energy_j", "boundary_energy_j", "equilibrium_residual", "radial_strain"):
            if not math.isfinite(getattr(state, name)):
                raise ValueError(f"Nonfinite shell checkpoint scalar {name}")
        if state.initial_column_energy_j <= 0 or state.equilibrium_residual < 0:
            raise ValueError("Invalid shell checkpoint ledger")
        if state.first_fracture_time_myr is not None and (not math.isfinite(state.first_fracture_time_myr)
                or not 0 <= state.first_fracture_time_myr <= state.time_myr):
            raise ValueError("Invalid fracture event time")
        if np.any((state.lid_thickness_km<0)|(state.lid_thickness_km>p.column_depth_km)) or np.any(state.peak_tensile_pa<0):
            raise ValueError("Nonphysical shell thickness or stress")
        if (len(thermal_state.energy) != 4 or not all(math.isfinite(v) and v >= 0 for v in thermal_state.energy)
                or not math.isfinite(thermal_state.initial_total_energy) or thermal_state.initial_total_energy <= 0
                or any(not math.isfinite(t) or not 0 <= t <= thermal_state.time_myr for t in thermal_state.events.values())
                or abs(diagnose(thermal_state, thermal)["relative_energy_residual"]) > 1e-8):
            raise ValueError("Invalid thermal checkpoint energy ledger")
        mesh = build_icosphere(p.subdivisions)
        current_energy = float(np.sum(state.column_enthalpy*mesh.physical_cell_areas_km2(thermal.radius_km)[:, None]*1e6)*p.density_kg_m3*p.column_depth_km*1000/p.column_layers)
        if abs(current_energy-state.initial_column_energy_j-state.boundary_energy_j)/state.initial_column_energy_j > 1e-8:
            raise ValueError("Invalid column checkpoint energy ledger")
        return state, thermal_state, p, thermal, meta
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Malformed shell checkpoint") from exc
