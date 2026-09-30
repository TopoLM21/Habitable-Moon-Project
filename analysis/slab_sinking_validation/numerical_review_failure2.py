"""Independent dimensionless primal/dual audit of the archived second failure."""
from pathlib import Path
import json
import sys
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import Bounds, LinearConstraint, lsq_linear, minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
path = ROOT/'analysis/slab_sinking_validation/runs/validated_sinking_fixed_sub4_dt1/elapsed_0200.solver_failure.npz'
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
dual = lsq_linear(Cn, -r, bounds=(0., np.inf), method='bvls', tol=1e-13, max_iter=1000)
forces = dual.x*scale/column_norm
y_dual = r+Cn@dual.x
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
z = E.T@v
initial = forces[ids]/(root_weight*force_scale)
secondary = minimize(lambda q: .5*float(q@q), initial, jac=lambda q:q,
    method='SLSQP', bounds=Bounds(0.,np.inf), constraints=LinearConstraint(E,v,v),
    options=dict(ftol=1e-13,maxiter=1000))
clipped = np.maximum(secondary.x, 0.)
redistributed = np.zeros_like(forces); redistributed[ids] = clipped*root_weight*force_scale
primal = minimize(lambda y:.5*float((y-r)@(y-r)), np.zeros_like(r), jac=lambda y:y-r,
    method='SLSQP', constraints=LinearConstraint(Cn.T,0.,np.inf),
    options=dict(ftol=1e-14,maxiter=5000))
support = np.flatnonzero(secondary.x > 1e-12*max(np.linalg.norm(secondary.x), 1.))
polished = np.zeros_like(secondary.x)
polished[support] = np.linalg.lstsq(E[:,support], v, rcond=2e-12)[0]
lagrange = np.linalg.lstsq(E[:,support].T, -polished[support], rcond=2e-12)[0]
inactive = np.ones(len(polished), dtype=bool); inactive[support] = False
polish_kkt = E.T@lagrange+polished
# Independent re-scaling: use unit-norm equality target, removing dependence
# on the arbitrarily large cancelling forces of the sparse primary solution.
second_scale = np.linalg.norm(v)
scaled = minimize(lambda z:.5*float(z@z), initial/second_scale, jac=lambda z:z,
    method='SLSQP', bounds=Bounds(0.,np.inf),
    constraints=LinearConstraint(E, v/second_scale, v/second_scale),
    options=dict(ftol=1e-13,maxiter=5000))
scaled_z = scaled.x*second_scale
best_cost = np.inf
best_support = None
best_vector = None
feasible_supports = 0
for mask in range(1, 1 << len(ids)):
    support_ids = np.array([i for i in range(len(ids)) if mask & (1 << i)])
    if len(support_ids) < len(v):
        continue
    candidate = np.linalg.lstsq(E[:,support_ids], v, rcond=2e-12)[0]
    if candidate.min() < -2e-14:
        continue
    candidate = np.maximum(candidate, 0.)
    if np.linalg.norm(E[:,support_ids]@candidate-v) > 2e-14:
        continue
    feasible_supports += 1
    cost = .5*float(candidate@candidate)
    if cost < best_cost:
        best_cost, best_support = cost, support_ids.tolist()
        best_vector = np.zeros(len(ids)); best_vector[support_ids] = candidate
report = dict(
    dual_success=bool(dual.success), dual_optimality=float(dual.optimality),
    primary_nonzero_count=int(np.count_nonzero(forces)), candidate_count=len(ids),
    maximum_force_n=float(forces.max()), minimum_feed_scaled=float((feed/feed_scale).min()),
    drag_eigenvalues=np.linalg.eigvalsh(D/d[:,None]/d[None,:]).tolist(),
    singular_values=singular.tolist(), keep=keep.tolist(),
    target_norm=float(np.linalg.norm(target)), force_scale=float(force_scale),
    unconstrained_candidate_min=float(z.min()),
    initial_eq_error=float(np.linalg.norm(E@initial-v)),
    initial_full_error=float(np.linalg.norm(M@initial-target)),
    secondary_success=bool(secondary.success), secondary_message=secondary.message,
    secondary_eq_error=float(np.linalg.norm(E@secondary.x-v)),
    secondary_full_error=float(np.linalg.norm(M@secondary.x-target)),
    clipped_full_error=float(np.linalg.norm(M@clipped-target)),
    generalized_torque_change=float(np.linalg.norm(C@(redistributed-forces))/scale),
    primal_success=bool(primal.success), primal_message=primal.message,
    primal_min_feed=float(np.min(Cn.T@primal.x)),
    primal_dual_relative_difference=float(np.linalg.norm(primal.x-y_dual)/max(np.linalg.norm(primal.x),1e-300)),
    primal_cost=float(primal.fun), dual_cost=float(.5*((y_dual-r)@(y_dual-r))),
    secondary_norm=float(np.linalg.norm(secondary.x)), secondary_values=secondary.x.tolist(),
    polish_support=support.tolist(), polished_min=float(polished.min()),
    polished_eq_error=float(np.linalg.norm(E@polished-v)),
    polished_full_error=float(np.linalg.norm(M@polished-target)),
    polished_min_inactive_kkt=float(polish_kkt[inactive].min(initial=0.)),
    polished_active_kkt_max=float(np.abs(polish_kkt[support]).max(initial=0.)),
    scaled_success=bool(scaled.success), scaled_message=scaled.message,
    scaled_full_error=float(np.linalg.norm(M@scaled_z-target)),
    enumerated_feasible_supports=feasible_supports,
    enumerated_minimum_cost=best_cost, enumerated_optimal_support=best_support,
    enumerated_vs_polished_difference=float(np.linalg.norm(best_vector-polished)),
    polished_cost=float(.5*(polished@polished)),
)
print(json.dumps(report, indent=2))
(ROOT/'analysis/slab_sinking_validation/numerical_review_failure2.json').write_text(
    json.dumps(report,indent=2)+'\n',encoding='utf-8')
