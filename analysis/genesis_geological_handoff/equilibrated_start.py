"""Explicit, opt-in compatible elastic equilibration; no fictitious time or heat.

This changes initial data, not the subsequent Maxwell/thermal equations.
Velocity balances the prescribed loads through existing drag and elastic force.
A mass-gauged compatible displacement preserves the incompatible part of the
inherited memory. The default permits creep and sets the frozen constitutive
internal-force derivative to zero as an initial slow-motion approximation.
"""
from copy import deepcopy
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve, null_space

from tectonics.experimental_edge_weakening import scale
from tectonics.genesis_transfer_bridge import encoded, digest
from tectonics.simplified_genesis_nodal import rate
from .spherical_coupled import to_tensor

VERSION = 'spherical-compatible-equilibrated-start/v2'
FORCE_TOL = 1e-8


def rigid_modes(X, radius):
    J = np.zeros((6 * len(X), 6))
    for i, x in enumerate(X / radius):
        cross = np.array([[0., -x[2], x[1]], [x[2], 0., -x[0]], [-x[1], x[0], 0.]])
        J[6*i:6*i+3, :3] = np.eye(3)
        J[6*i:6*i+3, 3:] = J[6*i+3:6*i+6, 3:] = -cross
    return J


class CompatibleProjection:
    """One stiffness factorization, six mass-orthogonal rigid gauges."""
    def __init__(self, L, H, ids, M, J):
        self.L, self.H, self.ids = L, H, ids
        self.nd = len(M)
        constraint = J.T @ M
        constraint /= np.linalg.norm(constraint, axis=1)[:, None]
        self.Q = null_space(constraint)
        K = np.zeros_like(M)
        local = np.einsum('qki,qkl,qlj->qij', L, H, L, optimize=True)
        # Accumulate quadrature locally, without a dense global strain matrix.
        np.add.at(K, (ids[:, :, None], ids[:, None, :]), local)
        self.K = K
        reduced = self.Q.T @ K @ self.Q
        self.s = 1 / np.sqrt(np.diag(reduced))
        reduced *= self.s[:, None] * self.s[None, :]
        self.factor = cho_factor((reduced + reduced.T) / 2, lower=True)

    def strain(self, du):
        return np.einsum('qij,qj->qi', self.L, du[self.ids])

    def force(self, e):
        local = np.einsum('qki,qkl,ql->qi', self.L, self.H, e, optimize=True)
        out = np.zeros(self.nd)
        np.add.at(out, self.ids.ravel(), local.ravel())
        return out

    def solve(self, force):
        return self.Q @ (self.s * cho_solve(self.factor, self.s * (self.Q.T @ force)))


def protected_state(a):
    """Everything except explicitly adjusted mechanics/derived diagnostics."""
    s = deepcopy(a.model.state)
    for k in ('spherical_memory', 'spherical_born_memory', 'memory',
              'nodal_velocity', 'nodal_displacement', 'omega', 'stress_pa'):
        s.pop(k, None)
    for b in s.get('born_material', []):
        b.pop('memory', None)
    return digest(encoded(dict(config=a.model.config, state=s)))


def set_memory(a, e):
    m = a.model
    m.state['spherical_memory'] = e[:16000].tolist()
    for j, b in enumerate(m.state.get('born_material', [])):
        m.state['spherical_born_memory'][b['cell_id']] = e[16000+50*j:16000+50*(j+1)].tolist()
    a.collapse()


def point_stress(a):
    _, _, f, _, e = a.arrays()
    return np.einsum('qij,qj->qi', a.model.C[f] * scale(a.model, a.model.state['damage'])[f, None, None], e)


def equilibrium_balance(a):
    basal = a.basal()
    F = basal['F']
    v = np.array(a.model.state['nodal_velocity'])
    internal = a.internal_force()
    drag = a.drag_action(basal, v)
    residual = internal + drag - F
    J = rigid_modes(a.path.X, a.R)
    # Physical force/torque use the same spherical reference virtual rigid fields.
    def physical(x):
        z = J.T @ x
        return dict(force_n=z[:3].tolist(), torque_nm=(a.R*z[3:]).tolist())
    rigid_residual = J.T @ residual
    rigid_external = J.T @ F
    # Also contract with rotations about the actual displaced positions. The
    # present small-strain internal law is objective to reference rotations only;
    # report its deformed torque rather than claiming finite-strain objectivity.
    current = J.copy()
    u = np.array(a.model.state['nodal_displacement']).reshape(-1,6)
    for i in range(len(u)):
        for part in range(2):
            x = u[i,3*part:3*part+3]/a.R
            current[6*i+3*part:6*i+3*part+3,3:] -= np.array([[0.,-x[2],x[1]],[x[2],0.,-x[0]],[-x[1],x[0],0.]])
    return dict(relative=float(np.linalg.norm(residual) / max(np.linalg.norm(F), 1.)),
                residual_norm_n=float(np.linalg.norm(residual)),
                external=physical(F), internal=physical(internal), drag=physical(drag), residual=physical(residual),
                rigid_relative=float(np.linalg.norm(rigid_residual) / max(np.linalg.norm(rigid_external), 1.)),
                force_relative=float(np.linalg.norm(rigid_residual[:3])/max(np.linalg.norm(rigid_external[:3]),1.)),
                torque_relative=float(np.linalg.norm(rigid_residual[3:])/max(np.linalg.norm(rigid_external[3:]),1.)),
                deformed_torques_nm={name:(a.R*(current.T@value)[3:]).tolist() for name,value in
                                    [('external',F),('internal',internal),('drag',drag),('residual',residual)]})


def equilibrate(a, *, max_iterations=12, max_seconds=90, mode='slow_force'):
    """Transactional elastic projection at unchanged thermal time.

    e1=e0+L du and u1=u0+du. Recompute moving basal geometry each iteration;
    beta, traction, coupling law and every damage value are unchanged.
    The default additionally sets the frozen-material internal-force derivative
    to zero: K v = L.T H R (e0 + L du). This permits ongoing Maxwell creep,
    rather than imposing a static elastic body. No relaxation interval, heat,
    damage evolution, phase event or time step is executed.
    """
    if 'equilibrated_start' in a.origin:
        raise ValueError('ALREADY_EQUILIBRATED')
    if mode not in ('slow_force', 'static_rigid'):
        raise ValueError('EQUILIBRATION_MODE')
    tick = time.monotonic()
    m = a.model
    saved = deepcopy(m.state)
    origin = deepcopy(a.origin)
    protected = protected_state(a)
    try:
        L, w, f, nodes, e0 = a.arrays()
        H = w[:, None, None] * m.C[f] * scale(m, m.state['damage'])[f, None, None]
        M = a.material_mass()
        J = rigid_modes(a.path.X, a.R)
        op = CompatibleProjection(L, H, a.ids[f], M, J)
        rates = rate(m, m.temperature(m.state['energy'])[nodes])
        if np.any(rates <= 0):
            raise ValueError('EQUILIBRATION_REQUIRES_POSITIVE_MAXWELL_RATES')
        if mode == 'slow_force':
            rate_op = CompatibleProjection(L, H*rates[:,None,None], a.ids[f], M, J)
            # K K_R^{-1} K is positive on the nonrigid subspace; drag supplies
            # the six remaining rigid directions. The gauges constrain du only.
            KQ = op.K @ rate_op.Q
            KR_inverse_K = rate_op.Q @ (rate_op.s[:,None] * cho_solve(rate_op.factor,rate_op.s[:,None]*KQ.T))
            effective = op.K @ KR_inverse_K
        u0 = np.array(saved['nodal_displacement'])
        v0 = np.array(saved['nodal_velocity'])
        du = np.zeros_like(u0)
        v = np.zeros_like(v0)
        before = equilibrium_balance(a)
        trace = []
        for iteration in range(max_iterations):
            if time.monotonic() - tick > max_seconds:
                raise TimeoutError('EQUILIBRATION_BUDGET')
            e = e0 + op.strain(du)
            m.state['nodal_displacement'] = (u0 + du).tolist()
            basal = a.basal()
            if mode == 'static_rigid':
                DJ = np.column_stack([a.drag_action(basal, j) for j in J.T])
                rigid_velocity = np.linalg.solve(J.T @ DJ, J.T @ basal['F'])
                v = J @ rigid_velocity
            target = basal['F'] - a.drag_action(basal, v)
            residual = target - op.force(e)
            relative = float(np.linalg.norm(residual)/max(np.linalg.norm(basal['F']), 1.))
            rate_residual = op.force(rates[:,None]*e-op.strain(v))
            derivative_relative = float(np.linalg.norm(rate_residual)/max(np.linalg.norm(op.force(rates[:,None]*e)),1.))
            jr = J.T@residual;jf=J.T@basal['F']
            force_relative=float(np.linalg.norm(jr[:3])/max(np.linalg.norm(jf[:3]),1.))
            torque_relative=float(np.linalg.norm(jr[3:])/max(np.linalg.norm(jf[3:]),1.))
            trace.append(dict(iteration=iteration,relative=relative,force_relative=force_relative,torque_relative=torque_relative,
                              force_derivative_relative=derivative_relative,seconds=time.monotonic()-tick))
            if max(relative,force_relative,torque_relative) <= FORCE_TOL and (mode == 'static_rigid' or derivative_relative <= FORCE_TOL):
                break
            if mode == 'static_rigid':
                du += op.solve(residual)
            else:
                A = basal['D'] + effective
                diagonal = 1/np.sqrt(np.diag(A))
                factor = cho_factor((A+A.T)/2*diagonal[:,None]*diagonal[None,:],lower=True)
                rate_correction = rate_op.solve(rate_residual)
                rhs = residual + op.force(op.strain(rate_correction))
                dv = diagonal*cho_solve(factor,diagonal*rhs)
                du += rate_op.solve(op.force(op.strain(dv))-rate_residual)
                v += dv
        else:
            raise ValueError(f'EQUILIBRATION_NOT_CONVERGED {trace[-1]}')
        set_memory(a, e)
        m.state['nodal_velocity'] = v.tolist()
        m.state['omega'] = [np.linalg.solve(J[:, 3:].T@M@J[:, 3:], J[:, 3:].T@M@v).tolist()]
        m.state['omega'] = (np.array(m.state['omega'])/a.R).tolist()
        sigma = point_stress(a)
        totals = np.zeros((m.nf,6)); volumes = np.zeros(m.nf)
        np.add.at(totals,f,w[:,None]*sigma);np.add.at(volumes,f,w)
        world = to_tensor(totals/volumes[:,None])
        local = np.einsum('fai,fab,fbj->fij',a.frames,world,a.frames)
        m.state['stress_pa'] = np.stack((local[:,0,0],local[:,1,1],local[:,0,1]),axis=1).tolist()
        # Enforce the existing reference-chart and small-strain limits now.
        total_du = np.array(m.state['nodal_displacement']) - m.config['nodal']['initial_displacement']
        max_strain = float(abs(op.strain(total_du)).max())
        outer = total_du.reshape(-1,6)[:,:3];n=a.path.X/a.R
        max_rotation = float(np.linalg.norm(outer-n*np.sum(outer*n,axis=1)[:,None],axis=1).max()/a.R)
        if max_strain > m.shell['max_total_strain'] or max_rotation > m.config['max_rotation_rad']:
            raise ValueError('EQUILIBRATION_REFERENCE_LIMIT')
        if protected_state(a) != protected:
            raise ValueError('EQUILIBRATION_CHANGED_PROTECTED_STATE')
        final = equilibrium_balance(a)
        if max(final[k] for k in ('relative','force_relative','torque_relative')) > FORCE_TOL:
            raise ValueError('EQUILIBRATION_FORCE_TORQUE')
        U0 = float(.5*np.einsum('qi,qij,qj->',e0,H,e0));U1=float(.5*np.einsum('qi,qij,qj->',e,H,e))
        K0=float(.5*v0@M@v0);K1=float(.5*v@M@v)
        identity=float(.5*(op.force(e0)+op.force(e))@du)
        def momenta(v):
            p=J.T@M@v
            return dict(linear_kg_m_s=p[:3].tolist(),angular_kg_m2_s=(a.R*p[3:]).tolist())
        ledger=dict(version=VERSION,mode=mode,seconds=time.monotonic()-tick,iterations=trace,
            initial_balance=before,final_balance=final,elastic_before_j=U0,elastic_after_j=U1,
            kinetic_before_j=K0,kinetic_after_j=K1,signed_elastic_change_j=U1-U0,signed_kinetic_change_j=K1-K0,
            signed_mechanical_change_j=U1-U0+K1-K0,initialization_exchange_j=U1-U0+K1-K0,
            elastic_path_work_j=identity,elastic_identity_relative=abs(U1-U0-identity)/max(abs(U0),abs(U1),1.),
            momentum_before=momenta(v0),momentum_after=momenta(v),momentum_change=momenta(v-v0),
            generalized_momentum_change_norm=float(np.linalg.norm(M@(v-v0))),
            compatible_displacement_max_m=float(np.linalg.norm(du.reshape(-1,6)[:,:3],axis=1).max()),
            strain_change_max=float(abs(op.strain(du)).max()),reference_strain_max=max_strain,reference_rotation_max=max_rotation,
            mass_gauge_relative=float(np.linalg.norm(J.T@M@du)/max(np.linalg.norm(J)*np.linalg.norm(M@du),1.)),
            strain_rate_max=float(abs(op.strain(v)).max()),force_derivative_relative=derivative_relative,
            protected_state_sha256=protected,
            thermal_heat_j=0.,time_advanced_years=0.,damage_changed=False,
            interpretation='New initial-data approximation; discarded initial dynamic path, heating and possible early damage are not reconstructed. Maxwell evolution remains active afterwards.')
        if ledger['elastic_identity_relative'] > 1e-10:
            raise ValueError('EQUILIBRATION_ENERGY_IDENTITY')
        a.origin['equilibrated_start'] = ledger
        a.validate();m.validate()
        return ledger
    except Exception:
        m.state = saved
        a.origin = origin
        raise
