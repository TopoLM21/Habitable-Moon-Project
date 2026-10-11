"""Exact interpolation diagnostic for supplied scalar grids; no source evaluator.

Grid axes are t,s. This is neither a remainder bound nor a refinement policy.
Inputs must be actual values of the same mapped density on both grids.
"""
from fractions import Fraction as F


def lagrange(nodes, x):
    result = []
    for i, xi in enumerate(nodes):
        value = F(1)
        for j, xj in enumerate(nodes):
            if j != i:
                value *= (x - xj) / (xi - xj)
        result.append(value)
    return result


def residual(fine, coarse, fine_nodes, coarse_nodes, coarse_weights):
    if len(fine) != 5 or any(len(row) != 5 for row in fine):
        raise ValueError('Actual complete 5x5 density values required')
    if len(coarse) != 3 or any(len(row) != 3 for row in coarse):
        raise ValueError('Actual complete 3x3 density values required')
    basis = [lagrange(fine_nodes, x) for x in coarse_nodes]
    r = [[coarse[t][s] - sum(basis[t][j] * basis[s][i] * fine[j][i]
                            for j in range(5) for i in range(5))
          for s in range(3)] for t in range(3)]
    # Normalize only the descriptive marginal decomposition, not quadrature.
    weights = [w / sum(coarse_weights) for w in coarse_weights]
    mean = sum(weights[t] * weights[s] * r[t][s] for t in range(3) for s in range(3))
    sm = [sum(weights[t] * r[t][s] for t in range(3)) - mean for s in range(3)]
    tm = [sum(weights[s] * r[t][s] for s in range(3)) - mean for t in range(3)]
    interaction = [[r[t][s] - mean - sm[s] - tm[t] for s in range(3)] for t in range(3)]
    return dict(grid=r, mean=mean,
                s=sum(w * abs(v) for w, v in zip(weights, sm)),
                t=sum(w * abs(v) for w, v in zip(weights, tm)),
                interaction=sum(weights[t] * weights[s] * abs(interaction[t][s])
                                for t in range(3) for s in range(3)))
