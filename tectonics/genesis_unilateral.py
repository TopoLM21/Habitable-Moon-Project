"""Hard frictionless contact in the frozen-shell virtual-extension experiment.

Minimize the same elastic dead-load potential subject to Jq >= 0. Contact
reactions are nonnegative, do no work at equilibrium, and spend no fracture
energy. This is not the dissipative Contact/Coupled integrator. Load-factor
continuation never moves the reference geometry or advances physical time.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from numbers import Integral, Real

import numpy as np
from scipy import sparse
from scipy.optimize import nnls
from scipy.sparse.linalg import norm as sparse_norm, splu

from .genesis_shell import maximum_total_strain
from .genesis_shell_release import FrozenShell, ShellEquilibrium, ShellRelease, _relative


VERSION = "genesis-unilateral-0.1"


def _positive(value, name, *, zero=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value < 0 or (not zero and value == 0)):
        raise ValueError(f"{name} must be a finite {'nonnegative' if zero else 'positive'} number")
    return float(value)


@dataclass(frozen=True)
class ContactSolveParameters:
    max_contact_rows: int = 256
    dual_rank_relative_tolerance: float = 1e-14
    complementarity_tolerance: float = 1e-9

    def __post_init__(self):
        if (isinstance(self.max_contact_rows, bool) or not isinstance(self.max_contact_rows, Integral)
                or not 1 <= self.max_contact_rows <= 1024):
            raise ValueError("Contact row budget must be an integer from 1 to 1024")
        for name in ("dual_rank_relative_tolerance", "complementarity_tolerance"):
            value = _positive(getattr(self, name), name)
            if value > 1e-7:
                raise ValueError("Contact tolerance is too loose")


@dataclass(frozen=True)
class ContactEquilibrium(ShellEquilibrium):
    normal_operator: object
    normal_reaction_n: np.ndarray
    complementarity_relative_error: float
    contact_work_j: float
    active_contact_count: int
    dual_rank: int
    load_factor: float

    @property
    def admissible_contact(self):
        return self.admissible_open_crack


@dataclass(frozen=True)
class ContactRelease(ShellRelease):
    contact_release_term_j: float
    lifted_min_gap_m: float
    energy_roundoff_bound_j: float

    @property
    def released_energy_resolved(self):
        """Positive release distinguishable from conservative arithmetic noise.

        This is not a fracture criterion or a mesh convergence certificate.
        """
        return self.release_j > self.energy_roundoff_bound_j


@dataclass(frozen=True)
class LoadPathParameters:
    max_load_increment: float = .25
    min_load_increment: float = 1e-5
    max_incremental_strain: float = .001
    max_incremental_motion_edge_fraction: float = .005
    max_attempts: int = 10000

    def __post_init__(self):
        for name in ("max_load_increment", "min_load_increment", "max_incremental_strain",
                     "max_incremental_motion_edge_fraction"):
            _positive(getattr(self, name), name)
        if self.min_load_increment > self.max_load_increment:
            raise ValueError("Minimum load increment exceeds maximum")
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, Integral) or self.max_attempts < 1:
            raise ValueError("Load attempt budget must be a positive integer")


@dataclass(frozen=True)
class LoadAttempt:
    load_factor: float
    accepted: bool
    reason: str
    increment_strain: float
    increment_motion_edge_fraction: float
    total_strain: float
    total_motion_edge_fraction: float


@dataclass(frozen=True)
class LoadPathResult:
    initial: ContactEquilibrium
    last_accepted: ContactEquilibrium | None
    attempts: tuple[LoadAttempt, ...]
    reached_target: bool
    stop_reason: str


class UnilateralShell:
    def __init__(self, shell: FrozenShell, parameters=None):
        if not isinstance(shell, FrozenShell):
            raise ValueError("Unilateral solve requires a frozen shell snapshot")
        self.shell = shell
        self.parameters = parameters or ContactSolveParameters()
        if not isinstance(self.parameters, ContactSolveParameters):
            raise ValueError("Invalid contact solve parameters")
        self._cache = {}
        edge = shell.mesh.vertices[np.asarray(shell.mesh.shared_edges)[:, 2:]]
        lengths = np.arctan2(np.linalg.norm(np.cross(edge[:, 0], edge[:, 1]), axis=1),
                             np.sum(edge[:, 0]*edge[:, 1], axis=1))*shell.radius_m
        self.min_edge_m = float(lengths.min())

    def _normal_operator(self, equilibrium):
        topology, membrane = equilibrium.topology, equilibrium.membrane
        xyz = self.shell.mesh.vertices[topology.cut_edges]
        normal = np.cross(xyz[:, 0], xyz[:, 1])
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        across = (self.shell.mesh.centroids[topology.seam_faces[:, 1]]
                  -self.shell.mesh.centroids[topology.seam_faces[:, 0]])
        normal *= np.where(np.einsum("ij,ij->i", normal, across) >= 0, 1., -1.)[:, None]
        rows, cols, values = [], [], []
        for seam, banks in enumerate(topology.bank_vertices):
            for end in range(2):
                for bank, sign in ((0, -1.), (1, 1.)):
                    vertex = banks[bank, end]
                    projection = normal[seam]@membrane.vertex_basis[vertex]
                    for component in range(2):
                        rows.append(2*seam+end)
                        cols.append(2*vertex+component)
                        values.append(sign*projection[component])
        j = sparse.coo_matrix((values, (rows, cols)), shape=(2*len(xyz), membrane.ndof)).tocsr()
        j.eliminate_zeros()  # Shared crack tips have identically zero rows.
        return j

    def _assembly(self, cuts):
        raw = np.asarray(cuts)
        if raw.size == 0:
            if raw.shape not in ((0,), (0, 2)):
                raise ValueError("Cuts must have shape (n,2)")
            key = ()
        else:
            if raw.ndim != 2 or raw.shape[1] != 2 or raw.dtype.kind not in "iu":
                raise ValueError("Cuts must be integer edge pairs")
            key = tuple(sorted(tuple(sorted(map(int, edge))) for edge in raw))
            if len(set(key)) != len(key):
                raise ValueError("Cuts must be unique")
        if key in self._cache:
            return self._cache[key]
        free = self.shell.solve(cuts)
        k, c = free.bulk_matrix, free.constraints
        scale = 1/np.sqrt(k.diagonal())
        s = sparse.diags(scale)
        cs = c@s
        cs = sparse.diags(1/np.sqrt(np.asarray(cs.multiply(cs).sum(axis=1)).ravel()))@cs
        lu = splu(sparse.bmat([[s@k@s, cs.T], [cs, None]], format="csc"))

        def inverse(force):
            if force.ndim == 1:
                return scale*lu.solve(np.r_[scale*force, np.zeros(3)])[:free.membrane.ndof]
            rhs = np.vstack((scale[:, None]*force, np.zeros((3, force.shape[1]))))
            return scale[:, None]*lu.solve(rhs)[:free.membrane.ndof]

        j = self._normal_operator(free)
        nonzero = np.flatnonzero(np.diff(j.indptr))
        if len(nonzero) > self.parameters.max_contact_rows:
            raise ValueError("Explicit crack exceeds dense contact diagnostic row budget")
        jj = j[nonzero]
        response = inverse(jj.T.toarray()) if len(nonzero) else np.empty((free.membrane.ndof, 0))
        if len(nonzero):
            w = np.asarray(jj@response)
            w = .5*(w+w.T)
            if np.any(w.diagonal() <= 0):
                raise ValueError("Contact compliance must have positive diagonal")
            dual_scale = 1/np.sqrt(w.diagonal())
            normalized = dual_scale[:, None]*w*dual_scale[None, :]
            eig, vectors = np.linalg.eigh(normalized)
            cutoff = self.parameters.dual_rank_relative_tolerance*eig[-1]
            if eig[0] < -cutoff:
                raise ValueError("Contact compliance is not positive semidefinite")
            selected = eig > cutoff
            basis, roots = vectors[:, selected], np.sqrt(eig[selected])
        else:
            dual_scale, basis, roots = np.empty(0), np.empty((0, 0)), np.empty(0)
        entry = free, inverse, j, nonzero, response, dual_scale, basis, roots
        self._cache[key] = entry
        return entry

    def solve(self, cuts, load_factor=1.):
        factor = _positive(load_factor, "load_factor", zero=True)
        free, inverse, j, rows, response, dual_scale, basis, roots = self._assembly(cuts)
        k, g = free.bulk_matrix, free.initial_bulk_force
        try:
            with np.errstate(over="raise", invalid="raise"):
                f = free.external_force*factor  # Frozen prestress is NOT scaled.
        except FloatingPointError as exc:
            raise ValueError("Scaled load is not finite at this magnitude") from exc
        q_free = inverse(f-g)
        if not np.isfinite(q_free).all():
            raise ValueError("Free equilibrium is not finite at this load magnitude")
        reaction = np.zeros(j.shape[0])
        free_gap = np.asarray(j[rows]@q_free)
        if len(rows) and np.min(free_gap) < 0:
            linear = dual_scale*free_gap
            magnitude = float(np.max(np.abs(linear)))
            if magnitude:
                linear_scaled = linear/magnitude
                outside = linear_scaled-basis@(basis.T@linear_scaled)
                cancellation = dual_scale*(abs(j[rows])@np.abs(q_free))
                null_tolerance = (1e-8*max(np.linalg.norm(linear_scaled), 1.)
                                  +100*np.finfo(float).eps*np.linalg.norm(cancellation)/magnitude)
                if np.linalg.norm(outside) > null_tolerance:
                    raise ValueError("Contact load has an unresolved null-space component")
                # W=A^T A. Keeping all columns retains every inequality even
                # when endpoint normals repeat and the multiplier is nonunique.
                a = roots[:, None]*basis.T
                b = -(basis.T@linear_scaled)/roots
                multipliers, _ = nnls(a, b, maxiter=max(100, 100*len(rows)))
                reaction[rows] = dual_scale*(magnitude*multipliers)
        q = q_free+response@reaction[rows]
        if not np.isfinite(q).all() or not np.isfinite(reaction).all():
            raise ValueError("Contact equilibrium or reaction is nonfinite")
        gap = np.asarray(j@q)
        measured_gap, jump = self.shell._gaps(free.topology, free.membrane, q)
        if not np.allclose(gap, measured_gap, atol=self.shell.penetration_tolerance_m, rtol=1e-12):
            raise ValueError("Contact and geometric gap operators disagree")
        internal = k@q+g
        contact = j.T@reaction
        force_scale = max(np.linalg.norm(k@q), np.linalg.norm(g), np.linalg.norm(f),
                          np.linalg.norm(contact), np.finfo(float).tiny)
        if not np.isfinite(force_scale):
            raise ValueError("Force norm is not finite at this load magnitude")
        residual = float(np.linalg.norm(internal-f-contact)/force_scale)
        gauge = float(np.linalg.norm(free.constraints@q)/max(np.linalg.norm(q), np.finfo(float).tiny))
        work = float(reaction@gap)
        energy_scale = max(abs(float(q@(k@q))), abs(float(g@q)), abs(float(f@q)),
                           float(np.sum(np.abs(reaction*np.asarray(j@q_free)))), np.finfo(float).tiny)
        complementarity = float(np.max(np.abs(reaction*gap), initial=0.)/energy_scale)
        min_gap = float(np.min(gap, initial=0.))
        if (not np.isfinite([residual, gauge, complementarity, energy_scale, work, min_gap]).all()
                or residual > self.shell.equilibrium_tolerance or gauge > self.shell.equilibrium_tolerance
                or min_gap < -self.shell.penetration_tolerance_m
                or complementarity > self.parameters.complementarity_tolerance):
            raise ValueError("Unilateral equilibrium failed force, gap or complementarity checks")
        membrane = free.membrane
        inc = np.einsum("fai,fi->fa", membrane.b, q[membrane.dofs])/self.shell.radius_m
        strain = self.shell.elastic_strain+inc
        stored = .5*float(np.einsum("fi,fij,fj,f->", strain, self.shell.elasticity_pa,
                                     strain, self.shell.reference_volume_m3))
        reduced = .5*float(q@(k@q))+float((g-f)@q)
        strain_max, motion = self._increment_measures(free, q)
        if not np.isfinite([stored, reduced, strain_max, motion]).all():
            raise ValueError("Contact energy or geometry diagnostic is nonfinite")
        reasons = []
        if strain_max > self.shell.max_strain or abs(q[-1])/self.shell.radius_m > self.shell.max_strain:
            reasons.append("reference_strain_limit")
        if motion > self.shell.max_motion_edge_fraction:
            reasons.append("reference_motion_limit")
        values = {field.name: getattr(free, field.name) for field in fields(ShellEquilibrium)}
        values.update(displacement_m=q, stored_energy_j=stored, external_force=f,
                      external_potential_work_j=float(f@q), potential_energy_j=self.shell.initial_energy_j+reduced,
                      reduced_potential_j=reduced, normal_gap_m=gap, tangential_jump_m=jump,
                      equilibrium_residual=residual, constraint_residual=gauge,
                      max_added_strain=strain_max, max_motion_edge_fraction=motion, min_gap_m=min_gap,
                      admissible_open_crack=not reasons, rejection_reasons=tuple(reasons))
        # Diagnostic count of resolved multiplier rows, not physical contact
        # sites: duplicate rows can share a reaction nonuniquely. Compare with
        # the force balance scale so pure-shear roundoff is not counted as load.
        active = int(np.count_nonzero(reaction > force_scale*1e-10))
        return ContactEquilibrium(**values, normal_operator=j, normal_reaction_n=reaction,
                                  complementarity_relative_error=complementarity, contact_work_j=work,
                                  active_contact_count=active, dual_rank=len(roots), load_factor=factor)

    def _increment_measures(self, equilibrium, displacement):
        membrane = equilibrium.membrane
        strain = np.einsum("fai,fi->fa", membrane.b, displacement[membrane.dofs])/self.shell.radius_m
        return (float(maximum_total_strain(strain)),
                float(np.linalg.norm(displacement[:-1].reshape(-1, 2), axis=1).max()/self.min_edge_m))

    def compare_extension(self, seed_cuts, trial_cuts, load_factor=1.):
        old, new = self.solve(seed_cuts, load_factor), self.solve(trial_cuts, load_factor)
        old_set = {tuple(edge) for edge in old.topology.cut_edges}
        if not old_set <= {tuple(edge) for edge in new.topology.cut_edges}:
            raise ValueError("Trial cuts must contain every existing cut")
        p = self.shell._prolongation(old, new)
        lifted = p@old.displacement_m
        lifted_gap = new.normal_operator@lifted
        lifted_min = float(np.min(lifted_gap, initial=0.))
        if lifted_min < -self.shell.penetration_tolerance_m:
            raise ValueError("Old equilibrium is not feasible in the extended contact space")
        k_error = float(sparse_norm(p.T@new.bulk_matrix@p-old.bulk_matrix)/sparse_norm(old.bulk_matrix))
        f_error = _relative(p.T@new.external_force-old.external_force, old.external_force)
        g_error = _relative(p.T@new.initial_bulk_force-old.initial_bulk_force, old.initial_bulk_force)
        c_error = float(sparse_norm(new.constraints@p-old.constraints)/sparse_norm(old.constraints))
        if max(k_error, f_error, g_error, c_error) > 1e-11:
            raise ValueError("Contact extension changed inherited material, load or gauge")
        lifted_potential = .5*float(lifted@(new.bulk_matrix@lifted))+float((new.initial_bulk_force-new.external_force)@lifted)
        embedding_error = lifted_potential-old.reduced_potential_j
        difference = old.reduced_potential_j-new.reduced_potential_j
        delta = new.displacement_m-lifted
        relaxation = .5*float(delta@(new.bulk_matrix@delta))
        gradient = new.bulk_matrix@new.displacement_m+new.initial_bulk_force-new.external_force
        released = relaxation-float(delta@gradient)
        # Unlike free banks, reaction work against the lifted old gaps can be
        # nonzero. Omitting this term would corrupt the virtual-extension energy.
        contact_term = float(new.normal_reaction_n@lifted_gap)
        energy_scale = max(abs(old.reduced_potential_j), abs(new.reduced_potential_j),
                           self.shell.initial_energy_j, np.finfo(float).tiny)
        roundoff = 200*np.finfo(float).eps*energy_scale
        error = abs(difference-released)
        tolerance = max(roundoff, self.shell.equilibrium_tolerance*abs(released))
        if error > tolerance or abs(released-relaxation-contact_term) > tolerance or released < -roundoff:
            raise ValueError("Contact virtual-extension energy identity failed")
        area = 0.
        for edge, faces in zip(new.topology.cut_edges, new.topology.seam_faces):
            if tuple(edge) not in old_set:
                a, b = self.shell.mesh.vertices[edge]
                length = np.arctan2(np.linalg.norm(np.cross(a, b)), float(a@b))*self.shell.radius_m
                area += length*float(np.min(self.shell.depth_m[faces]))
        reasons = tuple(sorted(set(old.rejection_reasons+new.rejection_reasons)))
        area_error = np.max(np.abs(new.topology.mesh.areas_unit_sphere-self.shell.mesh.areas_unit_sphere))
        volume_error = np.max(np.abs(new.topology.mesh.areas_unit_sphere*self.shell.radius_m**2*self.shell.depth_m
                                    -self.shell.reference_volume_m3))
        return ContactRelease(old, new, released, float(area), float(released/area) if area else 0.,
                              difference, relaxation, embedding_error, f_error, g_error, k_error, c_error,
                              error/max(abs(released), roundoff, np.finfo(float).tiny),
                              float(area_error/self.shell.mesh.areas_unit_sphere.max()),
                              float(volume_error/self.shell.reference_volume_m3.max()),
                              not reasons, reasons, contact_term, lifted_min, roundoff)

    def continue_loading(self, cuts, target_load_factor=1., parameters=None):
        target = _positive(target_load_factor, "target_load_factor", zero=True)
        parameters = parameters or LoadPathParameters()
        if not isinstance(parameters, LoadPathParameters):
            raise ValueError("Invalid load path parameters")
        initial = self.solve(cuts, load_factor=0.)
        if not initial.admissible_contact:
            return LoadPathResult(initial, None, (), False, "initial_reference_limit")
        current, attempts = initial, []
        step = parameters.max_load_increment
        while current.load_factor < target:
            if len(attempts) >= parameters.max_attempts:
                return LoadPathResult(initial, current, tuple(attempts), False, "attempt_budget")
            next_factor = min(target, current.load_factor+step)
            if next_factor <= current.load_factor:
                return LoadPathResult(initial, current, tuple(attempts), False, "load_resolution_limit")
            trial = self.solve(cuts, load_factor=next_factor)
            inc_strain, inc_motion = self._increment_measures(trial, trial.displacement_m-current.displacement_m)
            if not trial.admissible_contact:
                reason = "total_reference_limit"
            elif (inc_strain > parameters.max_incremental_strain
                  or inc_motion > parameters.max_incremental_motion_edge_fraction):
                reason = "increment_limit"
            else:
                reason = "accepted"
            attempts.append(LoadAttempt(next_factor, reason == "accepted", reason, inc_strain, inc_motion,
                                        trial.max_added_strain, trial.max_motion_edge_fraction))
            if reason == "total_reference_limit":
                return LoadPathResult(initial, current, tuple(attempts), False, reason)
            if reason == "increment_limit":
                step *= .5
                if step < parameters.min_load_increment:
                    return LoadPathResult(initial, current, tuple(attempts), False, "minimum_load_increment")
                continue
            current = trial
            step = min(parameters.max_load_increment, step*2)
        return LoadPathResult(initial, current, tuple(attempts), True, "target_reached")
