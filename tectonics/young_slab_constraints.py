"""Convex no-eduction solve for an explicitly irreversible attached slab branch.

The constraint is a modeling boundary condition, not a prediction that a slab
can never withdraw. The caller must cap total transmitted tensile force using
its declared neck-strength law and detach failed material before re-solving.
This module changes no material or history.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import Bounds, LinearConstraint, lsq_linear, minimize


@dataclass(frozen=True, slots=True)
class SlabConstraintSolution:
    omega_rad_s: np.ndarray
    reaction_torque_nm: np.ndarray
    normal_forces_n: np.ndarray
    active: np.ndarray
    unconstrained_feed_m_s: np.ndarray
    feed_m_s: np.ndarray
    relative_residual: float
    complementarity_relative_error: float
    primal_relative_violation: float
    reaction_work_w: float


def _polish_reaction_support(equality, value, candidate):
    """Recover a bound-active solution from exact least-squares KKT equations.

    SLSQP supplies an approximate feasible point. Its absolute stopping error
    can leave inactive bounds slightly positive, so a single threshold cannot
    reliably identify support. Equality-constrained minimum-norm steps stop at
    the first blocking bound; negative bound multipliers release a variable.
    Independent feasibility and KKT checks certify the final active set.
    """
    support = candidate > 0.
    current = np.maximum(candidate, 0.)
    for _ in range(max(100, 10*len(candidate))):
        if not np.any(support):
            if np.linalg.norm(value) == 0.:
                return np.zeros_like(candidate)
            raise RuntimeError("Slab reaction optimizer found no feasible positive support")
        active_matrix = equality[:, support]
        result = np.zeros_like(candidate)
        result[support] = np.linalg.lstsq(active_matrix, value, rcond=2e-12)[0]
        bound_tolerance = 2e-12*max(np.linalg.norm(result), 1.)
        negative = support & (result < -bound_tolerance)
        if np.any(negative):
            # Follow the minimum-norm direction only as far as nonnegativity
            # permits. This chooses which bound is active without changing a
            # force threshold or discarding all negative trial coordinates.
            ratios = np.full_like(current, np.inf)
            ratios[negative] = current[negative]/(current[negative]-result[negative])
            blocking = int(np.argmin(ratios))
            current += ratios[blocking]*(result-current)
            current[blocking] = 0.
            support[blocking] = False
            continue
        result = np.maximum(result, 0.)
        multiplier = np.linalg.lstsq(active_matrix.T, result[support], rcond=2e-12)[0]
        gradient = result-equality.T@multiplier
        gradient_scale = max(np.linalg.norm(result), np.linalg.norm(equality.T@multiplier), 1.)
        if np.min(gradient[~support], initial=0.) < -2e-10*gradient_scale:
            released = int(np.argmin(np.where(support, np.inf, gradient)))
            current = result
            support[released] = True
            continue
        break
    else:
        raise RuntimeError("Slab reaction support active set did not converge")
    if np.linalg.norm(equality@result-value) > 2e-12*max(np.linalg.norm(value), 1.):
        raise RuntimeError("Slab reaction support polish is not feasible")
    multiplier = np.linalg.lstsq(active_matrix.T, result[support], rcond=2e-12)[0]
    gradient = result-equality.T@multiplier
    gradient_scale = max(np.linalg.norm(result), np.linalg.norm(equality.T@multiplier), 1.)
    if (np.linalg.norm(gradient[support]) > 2e-10*gradient_scale
            or np.min(gradient[~support], initial=0.) < -2e-10*gradient_scale):
        raise RuntimeError("Slab reaction support polish failed optimality checks")
    return result


def _distribute_reactions(c_matrix, rhs_scale, forces, feed, feed_scale, weights):
    """Resolve dual indeterminacy by minimum sum(force^2 / hinge area).

    A basic NNLS solution can assign all reaction to one of two identical
    sections. That would create mesh-dependent tensile failure. The secondary
    strictly convex problem keeps exactly the same generalized reaction and
    distributes it according to the supplied positive compliance weights.
    """
    if not np.any(forces > 0.):
        return forces
    candidates = (feed <= 2e-10*feed_scale) | (forces > 0.)
    ids = np.flatnonzero(candidates)
    if ids.size < 2:
        return forces
    root_weight = np.sqrt(weights[ids]/np.max(weights[ids]))
    force_scale = np.max(forces[ids]/root_weight)
    matrix = c_matrix[:, ids]*(root_weight*force_scale/rhs_scale)
    target = c_matrix @ forces/rhs_scale
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    keep = singular > singular[0]*2e-12
    equality = vh[keep]
    value = (u[:, keep].T @ target)/singular[keep]
    # Orthonormal equality rows make this the unconstrained minimum norm.
    candidate = equality.T @ value
    tolerance = 2e-12*max(np.linalg.norm(candidate), 1.)
    if np.min(candidate) < -tolerance:
        initial = forces[ids]/(root_weight*force_scale)
        result = minimize(lambda z: .5*float(z@z), initial,
            jac=lambda z: z, method="SLSQP", bounds=Bounds(0., np.inf),
            constraints=LinearConstraint(equality, value, value),
            options=dict(ftol=1e-13, maxiter=1000))
        # A small equality residual in these orthonormal coordinates can still
        # be a large physical torque error. Use SLSQP only to identify support;
        # independently solve and certify the equality/bound KKT equations.
        candidate = _polish_reaction_support(equality, value, result.x)
    candidate = np.maximum(candidate, 0.)
    if np.linalg.norm(equality@candidate-value) > 2e-10*max(np.linalg.norm(value), 1.):
        raise RuntimeError("Slab reaction distribution lost its equilibrium constraint")
    redistributed = np.zeros_like(forces)
    redistributed[ids] = candidate*root_weight*force_scale
    if np.linalg.norm(c_matrix@(redistributed-forces))/rhs_scale > 5e-10:
        raise RuntimeError("Slab reaction redistribution changed the generalized torque")
    return redistributed


def solve_no_eduction(drag_nm_s, driving_torque_nm, feed_matrix_m, *, reaction_weights=None):
    """Minimize 1/2 omega.T D omega - b.T omega subject to A omega >= 0.

    D must be positive definite. The nonnegative dual force is solved by
    normalized bounded least squares, using a Cholesky factor after diagonal
    equilibration. BVLS uses rank-revealing least-squares subproblems: the
    Lawson-Hanson NNLS normal-equation active set can generate enormous
    cancelling forces on nearly dependent, oppositely directed trench rows.
    ``reaction_weights`` should be hinge cross-sectional area W*H; where
    reactions are indeterminate their minimum complementary-energy solution
    shares force in proportion to this area for identical sections. Equal
    weights are the explicit default. The caller owns all failure thresholds.
    """
    drag = np.asarray(drag_nm_s, dtype=float)
    torque = np.asarray(driving_torque_nm, dtype=float).reshape(-1)
    feed_map = np.asarray(feed_matrix_m, dtype=float)
    size = torque.size
    if (not size or size % 3 or drag.shape != (size, size) or feed_map.ndim != 2
            or feed_map.shape[1] != size or not np.isfinite(drag).all()
            or not np.isfinite(torque).all() or not np.isfinite(feed_map).all()):
        raise ValueError("Slab constraints require finite, dimensionally compatible SI arrays")
    scale = np.linalg.norm(drag)
    if scale == 0. or np.linalg.norm(drag-drag.T) > 1e-12*scale or np.any(np.diag(drag) <= 0.):
        raise ValueError("Slab constraint drag must be symmetric positive definite")
    count = feed_map.shape[0]
    weights = np.ones(count) if reaction_weights is None else np.asarray(reaction_weights, dtype=float)
    if weights.shape != (count,) or not np.isfinite(weights).all() or np.any(weights <= 0.):
        raise ValueError("Slab reaction weights must be positive finite section areas")
    diagonal = np.sqrt(np.diag(drag))
    equilibrated = drag/diagonal[:, None]/diagonal[None, :]
    try:
        factor, lower = cho_factor(equilibrated, lower=True, check_finite=False)
    except np.linalg.LinAlgError as error:
        raise ValueError("Slab constraint drag must be positive definite") from error
    scaled_rhs = torque/diagonal
    free_omega = cho_solve((factor, lower), scaled_rhs, check_finite=False)/diagonal
    free_feed = feed_map @ free_omega
    forces = np.zeros(count)
    norm_free = np.linalg.norm(free_omega)
    feed_scale = np.linalg.norm(feed_map, axis=1)*norm_free
    valid = np.linalg.norm(feed_map, axis=1) > 0.
    if np.any(valid) and np.any(free_feed < -2e-13*feed_scale):
        c_matrix = solve_triangular(factor, (feed_map/diagonal).T, lower=True, check_finite=False)
        whitened_rhs = solve_triangular(factor, scaled_rhs, lower=True, check_finite=False)
        rhs_norm = np.linalg.norm(whitened_rhs)
        column_norm = np.linalg.norm(c_matrix[:, valid], axis=0)
        dual = lsq_linear(c_matrix[:, valid]/column_norm,
            -whitened_rhs/rhs_norm, bounds=(0., np.inf), method="bvls",
            tol=1e-13, max_iter=max(300, 20*int(np.sum(valid))))
        if not dual.success:
            raise RuntimeError(f"Slab nonnegative dual solve did not converge: {dual.message}")
        coefficients = dual.x
        forces[valid] = coefficients*rhs_norm/column_norm
        reaction = feed_map.T @ forces
        candidate_omega = cho_solve((factor, lower), (torque+reaction)/diagonal,
                                   check_finite=False)/diagonal
        candidate_feed = feed_map @ candidate_omega
        feed_scale = np.linalg.norm(feed_map, axis=1)*max(norm_free, np.linalg.norm(candidate_omega))
        forces = _distribute_reactions(c_matrix, rhs_norm, forces,
                                       candidate_feed, feed_scale, weights)
    reaction = feed_map.T @ forces
    omega = cho_solve((factor, lower), (torque+reaction)/diagonal, check_finite=False)/diagonal
    feed = feed_map @ omega
    residual = drag@omega-torque-reaction
    denominator = max(np.linalg.norm(torque)+np.linalg.norm(reaction), np.finfo(float).tiny)
    relative_residual = float(np.linalg.norm(residual)/denominator)
    velocity_scale = np.linalg.norm(feed_map, axis=1)*max(norm_free, np.linalg.norm(omega))
    primal = float(np.max(np.divide(np.maximum(-feed, 0.), velocity_scale,
        out=np.zeros_like(feed), where=velocity_scale > 0.), initial=0.))
    power_scale = max(float(np.sum(np.abs(forces)*velocity_scale)), np.finfo(float).tiny)
    complementary = float(np.sum(np.abs(forces*feed))/power_scale)
    if primal > 2e-8 or complementary > 2e-8 or relative_residual > 2e-9:
        raise RuntimeError("Slab no-eduction solve failed its KKT residual checks")
    active = forces > (np.max(forces, initial=0.)*1e-12)
    return SlabConstraintSolution(omega.reshape(-1, 3), reaction.reshape(-1, 3), forces,
        active, free_feed, feed, relative_residual, complementary, primal,
        float(forces@feed))
