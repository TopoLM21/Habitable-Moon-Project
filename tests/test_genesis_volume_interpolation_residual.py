from fractions import Fraction as F
import pytest
from analysis.genesis_volume_residual_diagnosis.residual import residual
from analysis.genesis_volume_ray_validation.experiment import gauss


def evaluate(fn):
    x5, _ = zip(*gauss(5)); x3, w3 = zip(*gauss(3))
    fine = [[fn(s, t) for s in x5] for t in x5]
    coarse = [[fn(s, t) for s in x3] for t in x3]
    return residual(fine, coarse, x5, x3, w3)


def test_quartic_shape_does_not_mask_underresolved_t():
    result = evaluate(lambda s, t: s**4 + F(1, 10000)*t**8)
    assert result['s'] == result['interaction'] == 0
    assert result['t'] > 0


@pytest.mark.parametrize('axis', ['s', 't'])
def test_recovers_direction_hidden_from_fine_nodes(axis):
    nodes, _ = zip(*gauss(5))
    def hidden(x):
        out = F(1)
        for n in nodes: out *= (x-n)**2
        return out
    result = evaluate(lambda s, t: hidden(s if axis == 's' else t))
    assert result[axis] > 0
    assert result['t' if axis == 's' else 's'] == 0


def test_tensor_quartic_residual_is_exactly_zero():
    result = evaluate(lambda s, t: (1+s+s**4)*(2-t+t**4))
    assert all(v == 0 for row in result['grid'] for v in row)


def test_integrated_residual_is_existing_diagonal_not_new_error_evidence():
    x5, w5 = zip(*gauss(5)); x3, w3 = zip(*gauss(3))
    fn = lambda s, t: s**8 + t**6 + s**4*t**4
    result = evaluate(fn)
    diagonal = sum(wt*ws*fn(s,t) for t,wt in zip(x3,w3) for s,ws in zip(x3,w3)) - sum(wt*ws*fn(s,t) for t,wt in zip(x5,w5) for s,ws in zip(x5,w5))
    # Literal binary Gauss weights have a tiny moment defect; no ideal-weight assertion.
    integrated = result['mean']*sum(w3)**2
    assert abs(integrated-diagonal) < F(1,10**14)


def test_unseen_function_can_hide_from_both_grids():
    nodes = set(x for n in (3,5) for x,w in gauss(n))
    def hidden(x):
        out = F(1)
        for n in nodes: out *= (x-n)**2
        return out
    result = evaluate(lambda s,t: hidden(s))
    assert all(v == 0 for row in result['grid'] for v in row)
    assert hidden(F(1,7)) > 0  # Nonnegative continuous polynomial has positive integral.


def test_missing_values_are_not_reconstructed_from_rule_sums():
    x5, _ = zip(*gauss(5)); x3, w3 = zip(*gauss(3))
    with pytest.raises(ValueError, match='Actual complete'):
        residual([], [], x5, x3, w3)
