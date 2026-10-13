"""Versioned opt-in angular Maxwell fixture. Never loads a planetary checkpoint.

Six strain samples per face (two radial x three angular), engineering strains.
The frozen reduced fixture/defaults remain untouched. Temperature is a prescribed
mechanical load here, not a thermal-transport or phase-change continuation.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from tectonics.experimental_edge_weakening import face_damage
from .integrated_vector_operator import blocks
from .driven_shell import CONFIG

SCHEMA = 'habitable-moon/angular-maxwell-fixture/v1'
YEAR = 31557600.


def maxwell_update(old, increment, theta, dt, rate):
    return (old+increment-theta)/(1+dt*rate)


class AngularMaxwell:
    def __init__(self, grid, *, damage_feedback, expansion_per_k):
        self.grid = grid
        a = grid.a
        self.mesh = grid.mesh
        self.nf, self.nd = a.nf, a.nd
        self.M, self.force, self.free = a.M.copy(), a.force.copy(), a.free.copy()
        self.fixed = np.setdiff1d(np.arange(self.nd), self.free)
        self.frames = grid.frames.copy()
        self.old, self.base_scale = grid.old.copy(), grid.base_scale.copy()
        self.Cbase = a.C.reshape(self.nf, 2, 6, 6)[:, 0].copy()
        self.ids = a.ids.reshape(self.nf, 2, 18)[:, 0].copy()
        self.row, self.col = a.row.copy(), a.col.copy()
        inner = 1-CONFIG['thickness_km']/CONFIG['radius_km']
        geometry = [blocks(tri, inner) for tri in self.mesh.vertices[self.mesh.faces]*CONFIG['radius_km']*1000]
        self.B = np.array([x[0].reshape(6, 6, 18) for x in geometry])
        self.w = np.array([x[1].ravel() for x in geometry])
        self.points = np.array([x[2] for x in geometry])
        self.Kbase = np.einsum('fqki,fkl,fqlj,fq->fij', self.B, self.Cbase, self.B, self.w)
        self.feedback = bool(damage_feedback)
        self.alpha = float(expansion_per_k)
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError('Finite nonnegative expansion coefficient required')
        self.rate = CONFIG['young_pa']/CONFIG['maxwell_viscosity_pa_s']
        self.spec = dict(schema=SCHEMA, damage_feedback=self.feedback, expansion_per_k=self.alpha,
            config=CONFIG, strength_pa=1e6, damage_time_years=100., residual_stiffness=.02)
        signature = hashlib.sha256(json.dumps(self.spec, sort_keys=True).encode())
        for value in (self.mesh.vertices, self.mesh.faces, self.old, self.Cbase, self.B, self.w,
                      self.M.data, self.M.indices, self.M.indptr, self.force, self.free):
            signature.update(np.ascontiguousarray(value).tobytes())
        self.fingerprint = signature.hexdigest()

    def ratio(self, damage):
        if not self.feedback:
            return np.ones(self.nf)
        total = np.maximum(self.old, face_damage(self.mesh, damage))
        return (.02+.98*(1-total)**2)/self.base_scale

    def energy(self, memory, damage):
        e = np.asarray(memory).reshape(self.nf, 6, 6)
        base = .5*np.einsum('fqi,fij,fqj,fq->f', e, self.Cbase, e, self.w)
        return float(base@self.ratio(damage))

    def initial(self, *, memory=None, velocity=None, description=None):
        if (memory is not None or velocity is not None) and not description:
            raise ValueError('Describe the known initial field explicitly')
        description = description or 'exact zero angular memory'
        e = np.zeros((self.nf, 2, 3, 6)) if memory is None else np.asarray(memory, dtype=float)
        v = np.zeros(self.nd) if velocity is None else np.asarray(velocity, dtype=float)
        state = dict(schema=SCHEMA, operator_sha256=self.fingerprint, time_years=0.,
            initialization=dict(kind='declared known field', description=description),
            memory=e.tolist(), velocity=v.tolist(), displacement=np.zeros(self.nd).tolist(),
            temperature_change_k=np.zeros((self.nf, 2, 3)).tolist(),
            edge_damage=np.zeros(len(self.mesh.shared_edges)).tolist(), fracture_reserve_j=0.,
            cumulative={}, last_ledger={})
        self.validate(state)
        state['initial_elastic_j'] = self.energy(e, state['edge_damage'])
        state['initial_kinetic_j'] = float(.5*v@(self.M@v))
        return state

    def validate(self, state):
        if state.get('schema') != SCHEMA:
            raise ValueError('ANGULAR_HISTORY_REQUIRED: old/planetary formats cannot be continued uniquely')
        if state.get('operator_sha256') != self.fingerprint:
            raise ValueError('ANGULAR_OPERATOR_MISMATCH')
        for name, shape in [('memory', (self.nf, 2, 3, 6)), ('velocity', (self.nd,)),
                            ('displacement', (self.nd,)), ('temperature_change_k', (self.nf, 2, 3))]:
            value = np.asarray(state[name])
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f'ANGULAR_INVALID_{name}')
        if np.any(np.asarray(state['velocity'])[self.fixed] != 0) or np.any(np.asarray(state['displacement'])[self.fixed] != 0):
            raise ValueError('ANGULAR_FIXED_DOF_STATE')
        face_damage(self.mesh, state['edge_damage'])
        for name in ('time_years', 'fracture_reserve_j'):
            if not np.isfinite(state[name]) or state[name] < 0:
                raise ValueError(f'ANGULAR_INVALID_{name}')
        for ledger in ('cumulative', 'last_ledger'):
            if any(not np.isfinite(value) for value in state[ledger].values()):
                raise ValueError('ANGULAR_NONFINITE_LEDGER')

    def lift_reduced_fixture(self, radial_memory, *, assumption):
        """Memory-only zero-variance extension; velocity/damage/time start anew.

        Not a full-state converter and not reconstruction of old history.
        """
        if assumption != 'angular_uniform_zero_variance':
            raise ValueError('ANGULAR_HISTORY_NOT_IDENTIFIABLE')
        reduced = np.asarray(radial_memory, dtype=float)
        if reduced.shape != (self.nf, 2, 6) or not np.isfinite(reduced).all():
            raise ValueError('Only matching fixture radial samples are accepted')
        state = self.initial(memory=np.repeat(reduced[:, :, None, :], 3, axis=2),
            description='Explicit approximate angular-uniform extension of fixture radial samples; not a real-source import')
        old_energy = float(.5*np.einsum('fri,fij,frj,fr->', reduced, self.Cbase, reduced, self.w.reshape(self.nf,2,3).sum(axis=2)))
        state['initialization'].update(kind='approximate zero-variance extension',
            retained_radial_mean_exact=True, modeled_energy_difference_j=state['initial_elastic_j']-old_energy,
            omitted_angular_energy='nonnegative and unknown; not recoverable from means')
        return state

    def world_stress(self, stress):
        mean = np.einsum('fq,fqi->fi', self.w, stress)/self.w.sum(axis=1)[:, None]
        tensor = np.zeros((self.nf, 3, 3))
        tensor[:, range(3), range(3)] = mean[:, :3]
        for k, (i, j) in enumerate(((0,1), (0,2), (1,2)), 3):
            tensor[:, i, j] = tensor[:, j, i] = mean[:, k]
        return np.einsum('fai,fij,fbj->fab', self.frames, tensor, self.frames)

    def edge_shear(self, world):
        out = []
        for f, g, i, j in self.mesh.shared_edges:
            normal = np.cross(self.mesh.vertices[i], self.mesh.vertices[j]); normal /= np.linalg.norm(normal)
            traction = .5*(world[f]+world[g])@normal
            out.append(np.linalg.norm(traction-normal*(normal@traction)))
        return np.array(out)

    def step(self, state, years, *, temperature_increment_k=0.):
        self.validate(state)
        if not np.isfinite(years) or years <= 0:
            raise ValueError('Positive finite years required')
        deltaT = np.broadcast_to(np.asarray(temperature_increment_k, dtype=float), (self.nf, 2, 3))
        if not np.isfinite(deltaT).all():
            raise ValueError('Finite thermal load required')
        dt = years*YEAR
        relax = 1+dt*self.rate
        e0 = np.asarray(state['memory']).reshape(self.nf, 6, 6)
        v0 = np.asarray(state['velocity'])
        damage = np.asarray(state['edge_damage'])
        ratio = self.ratio(damage)
        C = self.Cbase*ratio[:, None, None]
        theta = self.alpha*deltaT.reshape(self.nf, 6, 1)*np.array([1.,1.,1.,0.,0.,0.])
        local = self.Kbase*ratio[:, None, None]/relax
        K = sparse.coo_matrix((local.ravel(), (self.row, self.col)), shape=self.M.shape).tocsr()
        A = self.M/dt**2+K
        stress0 = np.einsum('fij,fqj->fqi', C, e0-theta)/relax
        internal = np.zeros(self.nd)
        np.add.at(internal, self.ids.ravel(), np.einsum('fqij,fqi,fq->fj', self.B, stress0, self.w).ravel())
        rhs = self.M@v0/dt+self.force-internal
        u = np.zeros(self.nd)
        if len(self.free):
            active = A[self.free][:, self.free]; scale = 1/np.sqrt(active.diagonal())
            u[self.free] = scale*spsolve(sparse.diags(scale)@active@sparse.diags(scale), scale*rhs[self.free])
        v = u/dt
        increment = np.einsum('fqij,fj->fqi', self.B, u[self.ids])
        e = maxwell_update(e0, increment, theta, dt, self.rate)
        stress = np.einsum('fij,fqj->fqi', C, e)
        shear = self.edge_shear(self.world_stress(stress))
        updated = np.minimum(1., damage+years/100*np.maximum(shear/1e6-1, 0)**2) if self.feedback else damage.copy()
        new_ratio = self.ratio(updated)
        base_energy = .5*np.einsum('fqi,fij,fqj,fq->f', e, self.Cbase, e, self.w)
        release = float(base_energy@(ratio-new_ratio))
        if release < 0:
            raise ValueError('Negative fracture release')
        before = self.energy(e0, damage)
        up = float(base_energy@ratio); after = float(base_energy@new_ratio)
        de = e-e0
        kinetic0 = float(.5*v0@(self.M@v0)); kinetic1 = float(.5*v@(self.M@v))
        ledger = dict(elastic_change_j=after-before, kinetic_change_j=kinetic1-kinetic0,
            fracture_release_j=release, maxwell_j=dt*self.rate*2*up,
            thermal_work_j=float(-np.einsum('fqi,fqi,fq->', stress, theta, self.w)),
            numerical_memory_j=float(.5*np.einsum('fqi,fij,fqj,fq->', de, C, de, self.w)),
            numerical_kinetic_j=float(.5*(v-v0)@(self.M@(v-v0))),
            external_work_j=float(self.force@u), fracture_heat_j=0.)
        balance = (ledger['elastic_change_j']+ledger['kinetic_change_j']+release+ledger['maxwell_j']
            +ledger['numerical_memory_j']+ledger['numerical_kinetic_j']-ledger['thermal_work_j']-ledger['external_work_j'])
        denominator = max(before, after, kinetic0, kinetic1, abs(ledger['external_work_j']), abs(ledger['thermal_work_j']), 1.)
        momentum = np.linalg.norm((A@u-rhs)[self.free])/max(np.linalg.norm(rhs[self.free]), 1.)
        if momentum > 1e-8 or abs(balance)/denominator > 5e-8:
            raise ValueError(f'ANGULAR_BALANCE {momentum} {balance/denominator}')
        new = deepcopy(state)
        new.update(memory=e.reshape(self.nf, 2, 3, 6).tolist(), velocity=v.tolist(),
            displacement=(np.asarray(state['displacement'])+u).tolist(),
            temperature_change_k=(np.asarray(state['temperature_change_k'])+deltaT).tolist(),
            edge_damage=updated.tolist(), fracture_reserve_j=state['fracture_reserve_j']+release,
            time_years=state['time_years']+years, last_ledger=ledger)
        for key, value in ledger.items():
            new['cumulative'][key] = new['cumulative'].get(key, 0.)+value
        self.validate(new)
        return new, dict(momentum=float(momentum), balance_relative=float(abs(balance)/denominator),
            elastic_j=after, kinetic_j=kinetic1, max_damage=float(updated.max()), max_shear_pa=float(shear.max()))

    def save(self, state, path):
        self.validate(state)
        with Path(path).open('x') as stream:
            json.dump(state, stream, separators=(',', ':'), allow_nan=False)
            stream.write('\n')

    def load(self, path):
        state = json.loads(Path(path).read_text())
        self.validate(state)
        return state
