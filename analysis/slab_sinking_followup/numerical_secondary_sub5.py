"""Read-only independent secondary-QP audit of the ordered sub5 failure."""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import Bounds, LinearConstraint, lsq_linear, minimize, linprog


def feasible_active_set(equality, value, feasible):
    """Diagnostic oracle using feasible line searches, independent of SLSQP."""
    current = feasible.copy()
    free = np.ones(len(current), dtype=bool)
    for iteration in range(1000):
        trial = np.zeros_like(current)
        trial[free] = np.linalg.lstsq(equality[:, free], value, rcond=None)[0]
        negative = free & (trial < -2e-12*max(np.linalg.norm(trial), 1.))
        if negative.any():
            ids = np.flatnonzero(negative)
            fractions = current[ids]/(current[ids]-trial[ids])
            chosen = ids[np.argmin(fractions)]
            alpha = max(0., min(1., float(fractions.min())))
            current += alpha*(trial-current)
            current[chosen] = 0.
            free[chosen] = False
            continue
        current = trial
        multiplier = np.linalg.lstsq(equality[:, free].T, current[free], rcond=None)[0]
        gradient = current-equality.T@multiplier
        violating = np.flatnonzero(~free & (gradient < -2e-10))
        if not len(violating):
            return current, iteration+1, free
        free[violating[np.argmin(gradient[violating])]] = True
    raise RuntimeError('Diagnostic active-set oracle did not converge')


ROOT = Path(__file__).resolve().parents[2]
path = ROOT/'analysis/slab_sinking_followup/runs/ordered_sub5_dt1/elapsed_0050.solver_failure.npz'
saved = np.load(path)
D = saved['drag_nm_s']; b = saved['driving_torque_nm'].ravel()
A = saved['feed_matrix_m']; w = saved['reaction_weights']
d = np.sqrt(np.diag(D))
L, _ = cho_factor(D/d[:, None]/d[None, :], lower=True)
C = solve_triangular(L, (A/d).T, lower=True)
rhs = solve_triangular(L, b/d, lower=True)
scale = np.linalg.norm(rhs); r = rhs/scale
column_norm = np.linalg.norm(C, axis=0)
Cn = C/column_norm
dual = lsq_linear(Cn, -r, bounds=(0., np.inf), method='bvls', tol=1e-13, max_iter=2000)
forces = dual.x*scale/column_norm
omega = cho_solve((L, True), (b+A.T@forces)/d)/d
free = cho_solve((L, True), b/d)/d
feed = A@omega
feed_scale = np.linalg.norm(A, axis=1)*max(np.linalg.norm(free), np.linalg.norm(omega))
ids = np.flatnonzero((feed <= 2e-10*feed_scale) | (forces > 0.))
root_weight = np.sqrt(w[ids]/np.max(w[ids]))
force_scale = np.max(forces[ids]/root_weight)
M = C[:, ids]*(root_weight*force_scale/scale)
target = C@forces/scale
u, singular, vh = np.linalg.svd(M, full_matrices=False)
keep = singular > singular[0]*2e-12
E = vh[keep]; v = (u[:, keep].T@target)/singular[keep]
initial = forces[ids]/(root_weight*force_scale)
secondary = minimize(lambda q: .5*float(q@q), initial, jac=lambda q:q,
    method='SLSQP', bounds=Bounds(0.,np.inf), constraints=LinearConstraint(E,v,v),
    options=dict(ftol=1e-13,maxiter=1000))
support = np.flatnonzero(secondary.x > 1e-7)
active_matrix = E[:, support]
z = np.zeros(len(ids)); z[support] = np.linalg.lstsq(active_matrix, v, rcond=None)[0]
lam = np.linalg.lstsq(active_matrix.T, z[support], rcond=None)[0]
gradient = z-E.T@lam
inactive = np.ones(len(ids), dtype=bool); inactive[support] = False
primal = minimize(lambda y:.5*float((y-r)@(y-r)), np.zeros_like(r), jac=lambda y:y-r,
    method='SLSQP', constraints=LinearConstraint(Cn.T,0.,np.inf),
    options=dict(ftol=1e-14,maxiter=5000))
redistributed = np.zeros_like(forces); redistributed[ids] = z*root_weight*force_scale
oracle, iterations, oracle_free = feasible_active_set(E, v, initial)
split_rows = []
for index in range(len(ids)):
    for alpha in (1e-6, .13, .5, .87, 1.-1e-6):
        # Splitting physical hinge area w into alpha*w and (1-alpha)*w
        # multiplies its normalized columns by sqrt(alpha),sqrt(1-alpha).
        E_split = np.column_stack((E[:, :index], E[:, index]*np.sqrt(alpha),
            E[:, index]*np.sqrt(1.-alpha), E[:, index+1:]))
        expected = np.r_[z[:index], z[index]*np.sqrt(alpha),
            z[index]*np.sqrt(1.-alpha), z[index+1:]]
        # A separate feasible active-set solve must recover the same optimum.
        initial_split = np.r_[initial[:index], initial[index]*np.sqrt(alpha),
            initial[index]*np.sqrt(1.-alpha), initial[index+1:]]
        actual, _, _ = feasible_active_set(E_split, v, initial_split)
        merged = np.r_[actual[:index],
            actual[index]*np.sqrt(alpha)+actual[index+1]*np.sqrt(1.-alpha), actual[index+2:]]
        split_rows.append(dict(index=index, alpha=alpha,
            objective_error=float(abs(.5*actual@actual-.5*z@z)),
            solution_error=float(np.linalg.norm(actual-expected)),
            generalized_torque_error=float(np.linalg.norm(M@(merged-z))),
            equality_error=float(np.linalg.norm(E_split@actual-v))))
report = dict(
    secondary_shape=list(E.shape), target_norm=float(np.linalg.norm(v)),
    matrix_singular_values=singular.tolist(),
    secondary_success=bool(secondary.success), secondary_message=secondary.message,
    secondary_values=secondary.x.tolist(), ids=ids.tolist(),
    primary_primal_success=bool(primal.success),
    primary_primal_dual_difference=float(np.linalg.norm(primal.x-(r+Cn@dual.x))),
    support=support.tolist(), support_singular=np.linalg.svd(active_matrix,compute_uv=False).tolist(),
    support_values=z[support].tolist(), equality_error=float(np.linalg.norm(E@z-v)),
    generalized_torque_error=float(np.linalg.norm(C@(redistributed-forces))/scale),
    direct_torque_error=float(np.linalg.norm(A.T@(redistributed-forces))/np.linalg.norm(b)),
    active_gradient_max=float(np.abs(gradient[support]).max(initial=0.)),
    inactive_gradient_min=float(gradient[inactive].min()),
    multiplier=lam.tolist(), objective=float(.5*z@z),
    active_set_iterations=iterations,
    active_set_support=np.flatnonzero(oracle_free).tolist(),
    active_set_reference_difference=float(np.linalg.norm(oracle-z)),
    split_cases=len(split_rows),
    split_max_objective_error=max(row['objective_error'] for row in split_rows),
    split_max_solution_error=max(row['solution_error'] for row in split_rows),
    split_max_generalized_torque_error=max(row['generalized_torque_error'] for row in split_rows),
    split_max_equality_error=max(row['equality_error'] for row in split_rows),
)
print(json.dumps(report,indent=2))
(ROOT/'analysis/slab_sinking_followup/numerical_secondary_sub5.json').write_text(
    json.dumps(report,indent=2)+'\n',encoding='utf-8')
