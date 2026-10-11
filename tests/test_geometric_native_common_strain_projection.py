"""Independent controls of the declared nonlinear COMMON strain projection.

The eight old source channels are functions of genuine COLD bank fields.
The archived source material remains distinct from live field membership;
HOT source coupling and pointwise physical face transfer remain outside scope.
"""
from copy import deepcopy
from dataclasses import fields, make_dataclass, replace
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
from functools import lru_cache
from types import SimpleNamespace
import hashlib
import json
from math import isqrt

import pytest

from tectonics.geometric_native_pose import FractionInterval as I


BRANCHES = ("relaxed_zero_strength_shear", "frozen_elastic_predictor")
STORES = ("contact_energy_j", "bulk_energy_j", "cumulative_external_work_j",
          "cumulative_friction_work_j", "cumulative_viscous_work_j",
          "cumulative_numerical_dissipation_j")
EPOCHS = ("prepared_state", "previous_state", "unaccepted_candidate_state")


@lru_cache(maxsize=1)
def _api():
    from tectonics import geometric_native_common_strain_projection
    return geometric_native_common_strain_projection


@lru_cache(maxsize=1)
def _control():
    from analysis.geometric_native_common_strain_projection_validation import native_control
    return native_control


@lru_cache(maxsize=1)
def _system():
    return _control().strain_system()


@lru_cache(maxsize=1)
def _contexts():
    return _control().cold_contexts()


@lru_cache(maxsize=1)
def _before():
    return json.loads(dict(_control().source_context()[3].complete_source_dto_json)["before"])


def _carrier_system():
    return _control().source_context()[4]


def _transfer():
    return _control().material_system()


@lru_cache(maxsize=2)
def _response(branch=BRANCHES[0]):
    return _api().evaluate_native_common_strain_projection(_system(), branch=branch, precision=96)


def _literal(value):
    assert type(value) is float
    return F.from_float(value)


def _vector(value):
    return tuple(_literal(x) for x in value)


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), a[0]*b[0]*0)


def _cross(a, b):
    return a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]


def _scale(a, v):
    return tuple(a*x for x in v)


def _axis(label):
    return tuple(F(int(label == x)) for x in "xyz")


def _sym(a, b, v):
    x, y = _cross(a, _cross(b, v)), _cross(b, _cross(a, v))
    return tuple((u+w)/2 for u, w in zip(x, y))


def _square(value):
    if type(value) is not I:
        return value*value
    return I(F(0) if value.lo <= 0 <= value.hi else min(value.lo**2, value.hi**2),
             max(value.lo**2, value.hi**2))


def _encloses(interval, exact):
    assert type(interval) is I
    if type(exact) is I:
        # Independent evaluation of the same declared interval inputs.
        assert interval.lo <= exact.lo <= exact.hi <= interval.hi
    else:
        assert interval.lo <= exact <= interval.hi


def _overlaps(a, b):
    b = b if type(b) is I else I(b)
    assert max(a.lo, b.lo) <= min(a.hi, b.hi)


def _independent_cauchy_quotient(k, p, norm):
    # Contract the quotient itself using the original full N, not N/2.
    # A finer independent dyadic bound avoids copying the production cap.
    norm_upper = norm.hi/k.lo
    assert norm_upper >= 0
    numerator = norm_upper.numerator << 4096
    root = isqrt(numerator//norm_upper.denominator)
    root += int(root*root*norm_upper.denominator < numerator)
    cap = F(root, 1 << 2048)
    quotient = p/k
    return I(max(quotient.lo, -cap), min(quotient.hi, cap))


def _projection(raw):
    """C-orthogonal decomposition, independently read from the raw receipt."""
    k, p = raw.reference_tip_strain_stiffness_n_m, raw.projection_remainder_n
    dp, hp = raw.projection_remainder_gradient_n_m, raw.projection_hessian_n_m2
    quotient = _independent_cauchy_quotient(k, p, raw.remainder_strain_inner_product_j)
    s = I(raw.coordinates_m[0])+quotient
    ds = tuple(I(int(i == 0))+dp[i]/k for i in range(7))
    hs = tuple(tuple(hp[i][j]/k for j in range(7)) for i in range(7))
    r = raw.remainder_strain_inner_product_j/2-_square(p)/(2*k)
    assert r.hi >= 0
    r = I(max(F(0), r.lo), r.hi)
    gr = tuple(raw.remainder_local_gradient_n[i]-quotient*dp[i] for i in range(7))
    hr = tuple(tuple(raw.remainder_local_hessian_n_m[i][j]
        -dp[i]*dp[j]/k-quotient*hp[i][j] for j in range(7)) for i in range(7))
    return s, ds, hs, r, gr, hr


def _source_scalar(atom, radius, order, increments):
    """Differentiate rho_o(e.q)-(R_o e).(R_s rho_s q) after cancellation."""
    point, material = atom["original_point"], atom["original_material"]
    q = _vector(point["position"])
    sub, over = material["subducting_plate"], material["overriding_plate"]
    si = order.index(("radial", atom["trace_id"], "sub"))
    oi = order.index(("radial", atom["trace_id"], "over"))
    ps = _scale(I(radius)+increments[si], q)
    eta, first, second = [], [], []
    for name in ("normal", "down_dip"):
        e = _vector(point[name])
        eta.append((increments[oi]-increments[si])*_dot(q, e))
        b, h = [I(0)]*14, [[I(0)]*14 for _ in range(14)]
        for i, c in enumerate(order):
            if c[0] == "owner":
                lever = _cross(ps, e)["xyz".index(c[2])]
                b[i] = -lever if c[1] == sub else lever
            elif c[1] == atom["trace_id"]:
                b[i] = I((-1 if c[2] == "sub" else 1)*_dot(q, e))
            for j in range(i, 14):
                d, value = order[j], I(0)
                if c[0] == d[0] == "owner":
                    a, z = _axis(c[2]), _axis(d[2])
                    if c[1] == d[1] == sub:
                        value = -_dot(e, _sym(a, z, ps))
                    elif c[1] == d[1] == over:
                        value = -_dot(_sym(a, z, e), ps)
                    else:
                        sa, oa = (a, z) if c[1] == sub else (z, a)
                        value = -_dot(_cross(oa, e), _cross(sa, ps))
                elif {c[0], d[0]} == {"owner", "radial"}:
                    angular, radial = (c, d) if c[0] == "owner" else (d, c)
                    if radial[1:] == (atom["trace_id"], "sub"):
                        a = _axis(angular[2])
                        value = I(-_dot(e, _cross(a, q)) if angular[1] == sub
                                  else -_dot(_cross(a, e), q))
                h[i][j] = h[j][i] = value
        first.append(tuple(b))
        second.append(tuple(tuple(row) for row in h))
    return tuple(eta), tuple(first), tuple(second)


def _mechanism(atom, branch):
    columns, law = atom["original_material"]["columns"], atom["original_material"]["law"]
    k = tuple(1/(1/_literal(law[key])+sum((_literal(columns[s+"_length_m"])
        /_literal(columns[s+"_"+modulus]) for s in ("sub", "over")), F(0)))
        for key, modulus in (("normal_stiffness_pa_m", "normal_modulus_pa"),
                             ("tangential_stiffness_pa_m", "shear_modulus_pa")))
    if branch == BRANCHES[0]:
        k = k[0], F(0)
    state = atom["unaccepted_candidate_state"]
    t = tuple(_literal(state[name]) for name in ("traction_normal_pa", "traction_shear_pa"))
    e0 = sum((_literal(state[name]) for name in ("contact_energy_j", "bulk_energy_j")), F(0))
    return _literal(atom["original_area_m2"]), t, k, e0


def _beam_rows():
    before = _before()
    history, radius = before["original_physical_history"], _carrier_system().source_radius_m
    mechanics = history["accepted_before"]["initial_mechanics"]
    tips = {r[0]: r[1:] for r in before["snapshot"]["transverse_displacement_m"]}
    rows = []
    for n, trace in enumerate(mechanics["fault_traces"]["traces"]):
        for k, side in enumerate(("sub", "over")):
            i = 3*len(mechanics["plate_ids"])+2*n+k
            stiffness = history["original_reference"]["stiffness"][i]
            assert all(not value for j, value in enumerate(stiffness) if j != i)
            rows.append((("radial", trace["trace_id"], side),
                         _literal(stiffness[i])/radius**2, _literal(tips[trace["trace_id"]][k])))
    return tuple(rows)


@lru_cache(maxsize=2)
def _source_expected(branch):
    system, carrier = _system(), _carrier_system()
    order = carrier.coordinate_order
    increments = [I(0)]*14
    for block in system.projection_blocks:
        raw = next(raw for body, material, raw in _contexts() if body.component_id == block.component_id)
        increments[block.source_coordinate_index] = _projection(raw)[0]
    energy, gradient = I(0), [I(0)]*14
    hessian = [[I(0)]*14 for _ in range(14)]
    scalars = {}
    for atom in _before()["original_physical_history"]["quadrature_histories"]:
        eta, b, h = _source_scalar(atom, carrier.source_radius_m, order, increments)
        scalars[atom["point_id"]] = eta, b, h
        area, t0, k, e0 = _mechanism(atom, branch)
        traction = tuple(t+c*x for t, c, x in zip(t0, k, eta))
        energy += e0+area*sum((t*x+c*_square(x)/2 for t, c, x in zip(t0, k, eta)), I(0))
        for i in range(14):
            gradient[i] += area*sum((t*row[i] for t, row in zip(traction, b)), I(0))
            for j in range(14):
                hessian[i][j] += area*sum((c*row[i]*row[j]+t*matrix[i][j]
                    for c, row, t, matrix in zip(k, b, traction, h)), I(0))
    ledger = dict(_before()["original_physical_history"]["unaccepted_candidate_work_ledgers"])
    energy += _literal(ledger["bending_energy_increment_j"])
    for coordinate, k, w in _beam_rows():
        i = order.index(coordinate)
        s = increments[i]
        energy += k*(w*s+_square(s)/2)
        gradient[i] += k*(w+s)
        hessian[i][i] += k
    return tuple(increments), energy, tuple(gradient), tuple(tuple(row) for row in hessian), scalars


@lru_cache(maxsize=1)
def _global_maps():
    system = _system()
    mapping = {i: (i, I(1)) for i in range(6)}
    local = {}
    for block in system.projection_blocks:
        raw = next(raw for body, material, raw in _contexts() if body.component_id == block.component_id)
        values = _projection(raw)
        for j, i in enumerate(block.global_coordinate_indices):
            mapping[i] = block.source_coordinate_index, values[1][j]
            local[i] = block.component_id, j, values
    return mapping, local


def _global_entry(i, j, source_gradient, source_hessian):
    mapping, local = _global_maps()
    if i not in mapping or j not in mapping:
        return I(0), I(0), I(0)
    si, di = mapping[i]
    sj, dj = mapping[j]
    base = di*source_hessian[si][sj]*dj
    curvature, orthogonal = I(0), I(0)
    if i in local and j in local and local[i][0] == local[j][0]:
        _, li, values = local[i]
        lj = local[j][1]
        curvature = source_gradient[si]*values[2][li][lj]
        orthogonal = values[5][li][lj]
    return base+curvature+orthogonal, curvature, orthogonal


def _json_seal(raw, key, kind=None):
    body = ({f.name: raw[f.name] for f in fields(kind) if f.name != key}
            if kind is not None else {name: value for name, value in raw.items() if name != key})
    raw[key] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                       ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    return raw


DIM = 7
ZERO = (0,) * DIM


def poly_constant(value):
    value = F(value)
    return {ZERO: value} if value else {}


def poly_variable(index):
    exponents = [0] * DIM
    exponents[index] = 1
    return {tuple(exponents): F(1)}


def poly_add(*polynomials):
    result = {}
    for polynomial in polynomials:
        for powers, coefficient in polynomial.items():
            result[powers] = result.get(powers, F(0)) + coefficient
    return {powers: coefficient for powers, coefficient in result.items()
            if coefficient}


def poly_scale(polynomial, coefficient):
    coefficient = F(coefficient)
    return {powers: value * coefficient for powers, value in polynomial.items()
            if value * coefficient}


def poly_multiply(first, second):
    result = {}
    for p, x in first.items():
        for q, y in second.items():
            powers = tuple(a + b for a, b in zip(p, q))
            result[powers] = result.get(powers, F(0)) + x * y
    return {powers: coefficient for powers, coefficient in result.items()
            if coefficient}


def poly_derivative(polynomial, index):
    result = {}
    for powers, coefficient in polynomial.items():
        if powers[index]:
            reduced = list(powers)
            reduced[index] -= 1
            result[tuple(reduced)] = coefficient * powers[index]
    return result


def poly_value(polynomial, coordinates):
    if (type(coordinates) is not tuple or len(coordinates) != DIM
            or any(type(value) is not F for value in coordinates)):
        raise ValueError("The oracle needs seven exact Fraction coordinates")
    result = F(0)
    for powers, coefficient in polynomial.items():
        term = coefficient
        for value, degree in zip(coordinates, powers):
            term *= value ** degree
        result += term
    return result


def poly_jet(polynomial, coordinates):
    first = tuple(poly_derivative(polynomial, i) for i in range(DIM))
    return (poly_value(polynomial, coordinates),
            tuple(poly_value(p, coordinates) for p in first),
            tuple(tuple(poly_value(poly_derivative(p, j), coordinates)
                        for j in range(DIM)) for p in first))


def strain_polynomial_model():
    """E(a)=sum a_i L_i + 1/2 sum a_i a_j Q_ij, with L_0=0.

    H0=(2,-1,1), Q00=(1,2,3). Hence kappa=6, gamma=1/2,
    and the pure-tip orthogonal remainder is 25*a0**4/16.
    All six enrichments and all 28 mixed quadratic coefficients are present.
    """
    h0 = (F(2), F(-1), F(1))
    linear = ((F(0),) * 3, (F(1), F(2), F(-1)),
              (F(-2), F(1), F(3)), (F(2), F(0), F(1)),
              (F(-1), F(-3), F(2)), (F(3), F(1), F(0)),
              (F(1), F(-2), F(-2)))
    variables = tuple(poly_variable(i) for i in range(DIM))
    strain = [poly_add(*(poly_scale(variables[i], linear[i][k])
                         for i in range(DIM))) for k in range(3)]
    quadratic = {}
    for i in range(DIM):
        for j in range(i, DIM):
            value = ((F(1), F(2), F(3)) if i == j == 0 else
                     (F((i + 1) * (j + 1), 11),
                      F((-1) ** (i + j) * (i + j + 2), 7),
                      F((i - j) ** 2 + 1, 13)))
            quadratic[i, j] = value
            monomial = poly_multiply(variables[i], variables[j])
            for k in range(3):
                strain[k] = poly_add(strain[k], poly_scale(
                    monomial, value[k] * (F(1, 2) if i == j else F(1))))
    return h0, tuple(strain), quadratic


def projection_polynomials(*, reference_energy=F(5), source_force=F(11),
                           source_stiffness=F(7)):
    h0, strain, quadratic = strain_polynomial_model()
    kappa = sum((value * value for value in h0), F(0))
    projection = poly_add(*(poly_scale(value, direction)
                            for value, direction in zip(strain, h0)))
    norm = poly_add(*(poly_multiply(value, value) for value in strain))
    scalar = poly_add(poly_variable(0), poly_scale(projection, 1 / kappa))
    residual = poly_add(poly_scale(norm, F(1, 2)), poly_scale(
        poly_multiply(projection, projection), -1 / (2 * kappa)))
    source = poly_add(poly_constant(reference_energy),
                      poly_scale(scalar, source_force),
                      poly_scale(poly_multiply(scalar, scalar),
                                 source_stiffness / 2))
    potential = poly_add(source, residual)
    return dict(h0=h0, strain=strain, quadratic=quadratic, kappa=kappa,
                projection=projection, norm=norm, scalar=scalar,
                residual=residual, source=source, potential=potential,
                source_force=F(source_force), source_stiffness=F(source_stiffness))


def exact_projection_receipt(coordinates, **source_parameters):
    """Direct coefficient-dictionary differentiation, including all raw73.

    `rem_gradient` and `rem_hessian` differentiate N/2, not N. Their half
    convention is obtained by differentiating that polynomial independently.
    """
    model = projection_polynomials(**source_parameters)
    p, dp, d2p = poly_jet(model['projection'], coordinates)
    norm = poly_value(model['norm'], coordinates)
    _, rem_gradient, rem_hessian = poly_jet(
        poly_scale(model['norm'], F(1, 2)), coordinates)
    scalar, ds, d2s = poly_jet(model['scalar'], coordinates)
    residual, gr, hr = poly_jet(model['residual'], coordinates)
    potential, gradient, hessian = poly_jet(model['potential'], coordinates)
    upper = tuple((i, j) for i in range(DIM) for j in range(i, DIM))
    raw73 = (model['kappa'], p, *dp,
             *(d2p[i][j] for i, j in upper), norm, *rem_gradient,
             *(rem_hessian[i][j] for i, j in upper))
    assert len(raw73) == 73
    return dict(kappa=model['kappa'], p=p, dp=dp, d2p=d2p, N=norm,
                rem_gradient=rem_gradient, rem_hessian=rem_hessian,
                s=scalar, ds=ds, d2s=d2s, Rperp=residual,
                Gperp=gr, Hperp=hr, potential=potential,
                gradient=gradient, hessian=hessian, raw73=raw73,
                source_force=model['source_force'],
                source_stiffness=model['source_stiffness'])


def inverse_tip_twojet(**source_parameters):
    model = projection_polynomials(**source_parameters)
    h0, q00 = model['h0'], model['quadratic'][0, 0]
    gamma = sum((x * y for x, y in zip(h0, q00)), F(0)) / model['kappa']
    direct = exact_projection_receipt((F(0),) * DIM, **source_parameters)
    # For s(t)=t+gamma*t*t/2, the exact inverse has J=1, K=-gamma at0.
    # These are independently obtained by equating coefficients of s(t(w)).
    inverse_first, inverse_second = F(1), -gamma
    linear_pullback = direct['hessian'][0][0] * inverse_first ** 2
    full_pullback = linear_pullback + direct['gradient'][0] * inverse_second
    return dict(gamma=gamma, inverse_first=inverse_first,
                inverse_second=inverse_second,
                linear_pullback_hessian=linear_pullback,
                full_pullback_hessian=full_pullback,
                omitted_prestress_curvature=linear_pullback - full_pullback,
                expected_source_hessian=model['source_stiffness'])




def _decimal_polynomial(polynomial, coordinates):
    total, absolute_sum, operations = D(0), D(0), 0
    for powers, coefficient in polynomial.items():
        value = D(coefficient.numerator)/D(coefficient.denominator)
        for coordinate, degree in zip(coordinates, powers):
            if degree:
                value *= coordinate**degree
            operations += degree+2
        total += value
        absolute_sum += abs(value)
        operations += 3
    epsilon = D(10)**-94
    gamma = D(operations)*epsilon/(1-D(operations)*epsilon)
    return total, gamma*(1+absolute_sum)


def _directional_polynomial(polynomial, direction, degree):
    result = polynomial
    for _ in range(degree):
        result = poly_add(*(poly_scale(poly_derivative(result, i), direction[i]) for i in range(7)))
    return result


def test_authentic_ninety_two_coordinate_reduction_uses_all_eight_nonzero_cold_fields():
    system = _system()
    assert len(system.coordinate_order) == len(system.bound_coordinate_increments) == 92
    assert system.coordinate_order == _transfer().coordinate_order[:92]
    assert len(system.projection_blocks) == len(_contexts()) == 8
    assert len(system.hot_zero_coordinate_indices) == 30
    old = _carrier_system().coordinate_order
    seen = set()
    for body, material, raw in _contexts():
        assert len(raw.coordinates_m) == 7 and all(raw.coordinates_m)
        block = next(b for b in system.projection_blocks if b.component_id == body.component_id)
        assert block.coordinates_m == raw.coordinates_m
        assert block.trace_id == body.trace_id and block.side == body.source_bank
        assert old[block.source_coordinate_index] == ("radial", body.trace_id, body.source_bank)
        assert tuple(system.bound_coordinate_increments[i] for i in block.global_coordinate_indices) == raw.coordinates_m
        assert block.source_coordinate_index not in seen
        seen.add(block.source_coordinate_index)
    assert seen == set(range(6, 14))


def test_all_eight_projection_blocks_follow_the_true_strain_orthogonal_decomposition():
    for body, material, raw in _contexts():
        block = next(b for b in _system().projection_blocks if b.component_id == body.component_id)
        s, ds, hs, energy, gradient, hessian = _projection(raw)
        cap = block.projection_quotient_cauchy_absolute_upper_bound_m
        assert type(cap) is F and cap >= 0
        norm_ratio = raw.remainder_strain_inner_product_j.hi/raw.reference_tip_strain_stiffness_n_m.lo
        assert cap*cap >= norm_ratio
        step = F(1, 10**(_system().precision+16))
        assert cap == 0 or max(F(0), cap-step)**2 < norm_ratio
        quotient = block.projection_quotient_bounds_m
        original = raw.projection_remainder_n/raw.reference_tip_strain_stiffness_n_m
        assert original.lo <= quotient.lo <= quotient.hi <= original.hi
        assert -cap <= quotient.lo <= quotient.hi <= cap
        _overlaps(block.projection_increment_bounds_m, I(raw.coordinates_m[0])+quotient)
        _overlaps(block.projection_increment_bounds_m, s)
        _overlaps(block.orthogonal_energy_bounds_j, energy)
        for i in range(7):
            _overlaps(block.projection_gradient_bounds[i], ds[i])
            _overlaps(block.orthogonal_gradient_bounds_n[i], gradient[i])
            for j in range(7):
                _overlaps(block.projection_hessian_bounds[i][j], hs[i][j])
                _overlaps(block.orthogonal_hessian_bounds_n_m[i][j], hessian[i][j])


def test_all_sixteen_source_scalar_jets_keep_literal_geometry_and_receiver_rotation():
    for branch in BRANCHES:
        response = _response(branch)
        increments, energy, gradient, hessian, scalars = _source_expected(branch)
        assert len(response.source_atom_responses) == 16
        assert {a.point_id for a in response.source_atom_responses} == set(scalars)
        for atom in response.source_atom_responses:
            eta, b, h = scalars[atom.point_id]
            for scalar in range(2):
                _overlaps(atom.scalar_increment_bounds_m[scalar], eta[scalar])
                for i in range(14):
                    _overlaps(atom.scalar_first_derivative_bounds[scalar][i], b[scalar][i])
                    for j in range(14):
                        _overlaps(atom.scalar_second_derivative_bounds[scalar][i][j], h[scalar][i][j])
        for actual, expected in zip(response.source_coordinate_increment_bounds, increments):
            _overlaps(actual, expected)


def test_single_potential_counts_sixteen_serial_stores_eight_beams_and_eight_remainders_once():
    transfer = _transfer()
    literal_beam = _literal(dict(_before()["original_physical_history"]
        ["unaccepted_candidate_work_ledgers"])["bending_energy_increment_j"])
    mathematical_beam = sum((k*w*w/2 for coordinate, k, w in _beam_rows()), F(0))
    assert transfer.beam_reference_energy_j == literal_beam
    assert transfer.beam_mathematical_reference_energy_j == mathematical_beam
    assert transfer.beam_energy_rounding_residual_j == literal_beam-mathematical_beam
    assert literal_beam != mathematical_beam
    for branch in BRANCHES:
        response = _response(branch)
        _, source_energy, _, _, _ = _source_expected(branch)
        orthogonal = sum((_projection(raw)[3] for body, material, raw in _contexts()), I(0))
        _overlaps(response.source_potential_bounds_j, source_energy)
        _overlaps(response.orthogonal_energy_bounds_j, orthogonal)
        _overlaps(response.potential_bounds_j, source_energy+orthogonal)
        assert len(response.source_beam_responses) == 8
        assert len({(b.trace_id, b.side) for b in response.source_beam_responses}) == 8


def test_full_ninety_two_gradient_and_hessian_follow_one_global_nonlinear_pullback():
    mapping, local = _global_maps()
    for branch in BRANCHES:
        response = _response(branch)
        _, _, g, h, _ = _source_expected(branch)
        assert len(response.potential_gradient_bounds) == len(response.potential_hessian_bounds) == 92
        for i in range(14):
            _overlaps(response.source_gradient_bounds[i], g[i])
            for j in range(14):
                _overlaps(response.source_hessian_bounds[i][j], h[i][j])
        for i in range(92):
            expected = I(0)
            orthogonal_gradient = I(0)
            if i in mapping:
                source, derivative = mapping[i]
                expected = derivative*g[source]
                if i in local:
                    orthogonal_gradient = local[i][2][4][local[i][1]]
                    expected += orthogonal_gradient
            _overlaps(response.potential_gradient_bounds[i], expected)
            _overlaps(response.orthogonal_gradient_bounds[i], orthogonal_gradient)
            for j in range(92):
                expected_h, curvature, orthogonal_h = _global_entry(i, j, g, h)
                _overlaps(response.potential_hessian_bounds[i][j], expected_h)
                _overlaps(response.chainrule_curvature_hessian_bounds[i][j], curvature)
                _overlaps(response.orthogonal_hessian_bounds[i][j], orthogonal_h)
                assert response.potential_hessian_bounds[i][j] == response.potential_hessian_bounds[j][i]


def test_source_prestress_multiplies_the_projection_second_derivative_in_every_bank():
    nondegenerate = 0
    for branch in BRANCHES:
        response = _response(branch)
        for block in _system().projection_blocks:
            raw = next(raw for body, material, raw in _contexts() if body.component_id == block.component_id)
            force = response.source_gradient_bounds[block.source_coordinate_index]
            k = raw.reference_tip_strain_stiffness_n_m
            for i, gi in enumerate(block.global_coordinate_indices):
                for j, gj in enumerate(block.global_coordinate_indices):
                    # The raw integral does not prove this term's strict sign.
                    # Preserve its ENTIRE uncertainty enclosure: an omitted
                    # I(0) cannot contain a nondegenerate interval even when
                    # zero lies inside it. Source forces are independently
                    # checked against the original potential above.
                    expected = force*(raw.projection_hessian_n_m2[i][j]/k)
                    _encloses(response.chainrule_curvature_hessian_bounds[gi][gj], expected)
                    nondegenerate += int(expected.width > 0)
    assert nondegenerate, "The full genuine preload-curvature enclosure must survive assembly"


def test_directional_first_second_variations_include_mixed_owner_and_body_work():
    mapping, local = _global_maps()
    direction = tuple(F((-1)**i, 3+i) if i in mapping else F(0) for i in range(92))
    for branch in BRANCHES:
        response = _response(branch)
        _, _, g, h, _ = _source_expected(branch)
        expected_first, expected_second = I(0), I(0)
        for i in mapping:
            source, derivative = mapping[i]
            gi = derivative*g[source]
            if i in local:
                gi += local[i][2][4][local[i][1]]
            expected_first += direction[i]*gi
            for j in mapping:
                expected_second += direction[i]*direction[j]*_global_entry(i, j, g, h)[0]
        actual_first = sum((d*x for d, x in zip(direction, response.potential_gradient_bounds)), I(0))
        actual_second = sum((direction[i]*direction[j]*response.potential_hessian_bounds[i][j]
            for i in mapping for j in mapping), I(0))
        _overlaps(actual_first, expected_first)
        _overlaps(actual_second, expected_second)
        # A shared fresh owner rotation rotates bodies and receiver frames together.
        shared = tuple(F(int(c[0] == "owner" and c[2] == "z")) for c in _system().coordinate_order)
        rigid_work = sum((d*x for d, x in zip(shared, response.potential_gradient_bounds)), I(0))
        _encloses(rigid_work, F(0))


def test_projected_source_path_work_is_separate_from_finite_orthogonal_bulk_work():
    atoms = _before()["original_physical_history"]["quadrature_histories"]
    ledger = dict(_before()["original_physical_history"]["unaccepted_candidate_work_ledgers"])
    reference = sum((_mechanism(atom, BRANCHES[0])[3] for atom in atoms), F(0))
    reference += _literal(ledger["bending_energy_increment_j"])
    order = _carrier_system().coordinate_order
    for branch in BRANCHES:
        response = _response(branch)
        increments, energy, _, _, scalar = _source_expected(branch)
        work = I(0)
        for atom in atoms:
            eta = scalar[atom["point_id"]][0]
            area, traction, stiffness, e0 = _mechanism(atom, branch)
            work += area*sum(((t+k*x/2)*x for t, k, x in zip(traction, stiffness, eta)), I(0))
        for coordinate, stiffness, wref in _beam_rows():
            x = increments[order.index(coordinate)]
            work += stiffness*(wref+x/2)*x
        _overlaps(work, energy-reference)
        _overlaps(response.source_potential_bounds_j-reference, work)
        _overlaps(response.potential_bounds_j-reference,
                  work+response.orthogonal_energy_bounds_j)
        assert response.orthogonal_energy_bounds_j.hi > 0


def test_cold_source_coupling_is_nonzero_and_hot_source_coordinates_are_exactly_absent():
    system, response = _system(), _response()
    for block in system.projection_blocks:
        i = block.global_coordinate_indices[0]
        assert block.projection_gradient_bounds[0].lo > 0
        assert response.potential_hessian_bounds[i][i].hi > 0
        assert any(response.potential_gradient_bounds[j] != I(0) for j in block.global_coordinate_indices)
    for i in system.hot_zero_coordinate_indices:
        assert system.coordinate_order[i][0] == "body"
        assert response.potential_gradient_bounds[i] == response.orthogonal_gradient_bounds[i] == I(0)
        assert all(x == I(0) for x in response.potential_hessian_bounds[i])
        assert all(row[i] == I(0) for row in response.potential_hessian_bounds)
    assert all(c[0] != "source_internal" for c in system.coordinate_order)


def test_both_shear_branches_keep_original_material_epochs_clocks_and_work_lineages():
    before = _before()
    carrier = _carrier_system()
    transfer = _transfer()
    assert sum(a.archived for a in transfer.attachments) == 8
    assert len(transfer.attachments) == 32
    assert (transfer.source_geometry_elapsed_time_s, transfer.source_contact_elapsed_time_s,
            transfer.material_geometry_elapsed_time_s) == (F(0), F(2000000), F(1000000))
    atoms = {a["point_id"]: a for a in before["original_physical_history"]["quadrature_histories"]}
    for c in carrier.carriers:
        original = json.loads(c.complete_original_atom_history_json)
        assert original == atoms[c.point_id]
        assert json.loads(c.source_local_state_json) == original["unaccepted_candidate_state"]
        for epoch in EPOCHS:
            assert all(type(original[epoch][name]) is float for name in STORES)
    relaxed, frozen = _response(BRANCHES[0]), _response(BRANCHES[1])
    assert relaxed.branch != frozen.branch
    assert relaxed.coordinate_increments == frozen.coordinate_increments == _system().bound_coordinate_increments
    for branch, response in zip(BRANCHES, (relaxed, frozen)):
        for atom in response.source_atom_responses:
            if branch == BRANCHES[0]:
                assert atom.traction_bounds_pa[1] == I(0)
                assert atom.effective_tangent_stiffness_pa_m[1] == 0
            assert json.loads(atom.original_source_state_json) == atoms[atom.point_id]["unaccepted_candidate_state"]


def test_independent_strain_polynomial_expansion_checks_projection_and_complete_chain():
    api = _api()
    probes = ((F(0),)*7, tuple(F((-1)**i, i+8) for i in range(7)),
              (F(1, 3),)+(F(0),)*6)
    for coordinates in probes:
        expected = exact_projection_receipt(coordinates)
        # Feed pure production algebra, never register fabricated source material.
        values = api._projection_algebra(I(expected["kappa"]), I(expected["p"]),
            tuple(I(x) for x in expected["dp"]),
            tuple(tuple(I(x) for x in row) for row in expected["d2p"]), I(expected["N"]),
            tuple(I(x) for x in expected["rem_gradient"]),
            tuple(tuple(I(x) for x in row) for row in expected["rem_hessian"]), coordinates, 128)
        # The helper contract is fixed with the production binder.
        s, ds, hs = (values[name] for name in ("projection_increment_bounds_m",
            "projection_gradient_bounds", "projection_hessian_bounds"))
        energy, gradient, hessian = (values[name] for name in ("orthogonal_energy_bounds_j",
            "orthogonal_gradient_bounds_n", "orthogonal_hessian_bounds_n_m"))
        _encloses(s, expected["s"])
        _encloses(energy, expected["Rperp"])
        for i in range(7):
            _encloses(ds[i], expected["ds"][i])
            _encloses(gradient[i], expected["Gperp"][i])
            for j in range(7):
                _encloses(hs[i][j], expected["d2s"][i][j])
                _encloses(hessian[i][j], expected["Hperp"][i][j])
        block = SimpleNamespace(source_coordinate_index=6, global_coordinate_indices=tuple(range(6, 13)),
            projection_hessian_bounds=hs, orthogonal_gradient_bounds_n=gradient,
            orthogonal_hessian_bounds_n_m=hessian)
        maps = tuple(range(6))+(6,)*7+(None,)*79
        derivatives = (I(1),)*6+ds+(I(0),)*79
        system = SimpleNamespace(coordinate_to_source_index=maps,
            coordinate_source_projection_derivative_bounds=derivatives, projection_blocks=(block,))
        g = (I(0),)*6+(I(expected["source_force"])+expected["source_stiffness"]*s,)+(I(0),)*7
        h = tuple(tuple(I(expected["source_stiffness"] if i == j == 6 else 0)
                        for j in range(14)) for i in range(14))
        actual_g, actual_h, curvature, gr, hr = api._global_chain(system, g, h, 128)
        for i in range(7):
            _encloses(actual_g[6+i], expected["gradient"][i])
            for j in range(7):
                _encloses(actual_h[6+i][6+j], expected["hessian"][i][j])
        assert all(x == I(0) for x in actual_g[13:])
        # Differentiate the expanded polynomial independently. Degree4 gives
        # exact central-difference bias terms, without arbitrary tolerance.
        potential = projection_polynomials()["potential"]
        direction = tuple(F((-1)**i, i+3) for i in range(7))
        first = sum((direction[i]*expected["gradient"][i] for i in range(7)), F(0))
        second = sum((direction[i]*direction[j]*expected["hessian"][i][j]
                      for i in range(7) for j in range(7)), F(0))
        third = poly_value(_directional_polynomial(potential, direction, 3), coordinates)
        fourth = poly_value(_directional_polynomial(potential, direction, 4), coordinates)
        with localcontext() as context:
            context.prec = 96
            dec = lambda value: D(value.numerator)/D(value.denominator)
            step = D("1e-8")
            centre = tuple(dec(x) for x in coordinates)
            delta = tuple(step*dec(x) for x in direction)
            plus, ep = _decimal_polynomial(potential, tuple(x+d for x, d in zip(centre, delta)))
            minus, em = _decimal_polynomial(potential, tuple(x-d for x, d in zip(centre, delta)))
            zero, ez = _decimal_polynomial(potential, centre)
            f1, f2 = (plus-minus)/(2*step), (plus+minus-2*zero)/(step*step)
            first_bias = step*step*abs(dec(third))/6
            second_bias = step*step*abs(dec(fourth))/12
            epsilon = D(10)**-94
            gamma = 12*epsilon/(1-12*epsilon)
            arithmetic = gamma*(1+abs(plus)+abs(minus)+2*abs(zero)
                                +abs(dec(first))*step+abs(dec(second))*step*step)
            assert abs(f1-dec(first)) <= first_bias+(ep+em+arithmetic)/(2*step)
            assert abs(f2-dec(second)) <= second_bias+(ep+em+2*ez+arithmetic)/(step*step)
    inverse = inverse_tip_twojet()
    assert inverse["linear_pullback_hessian"] == F(25, 2)
    assert inverse["full_pullback_hessian"] == F(7)
    assert inverse["omitted_prestress_curvature"] == F(11, 2)
    # An uncertain positive integral norm must stay uncertain. Choosing its
    # midpoint would conceal the source coupling's certified input bounds.
    expected = exact_projection_receipt(probes[2])
    k = I(expected["kappa"]-F(1, 10), expected["kappa"]+F(1, 10))
    uncertain = api._projection_algebra(k, I(expected["p"]),
        tuple(I(x) for x in expected["dp"]),
        tuple(tuple(I(x) for x in row) for row in expected["d2p"]), I(expected["N"]),
        tuple(I(x) for x in expected["rem_gradient"]),
        tuple(tuple(I(x) for x in row) for row in expected["rem_hessian"]), probes[2], 128)
    bounds = I(probes[2][0])+_independent_cauchy_quotient(k, I(expected["p"]), I(expected["N"]))
    _encloses(uncertain["projection_increment_bounds_m"], bounds)
    assert uncertain["projection_increment_bounds_m"].width > 0
    # Broad independent raw intervals can include impossible p/k pairs.
    # Cauchy removes only those pairs; it does not clip the physical map or
    # differentiate the cap as though it were the strain projection.
    zero = (F(0),)*7
    zi = (I(0),)*7
    zh = tuple(zi for _ in range(7))
    contracted = api._projection_algebra(I(1, 4), I(-3, 3), zi, zh,
        I(0, 1), zi, zh, zero, 128)
    assert contracted["projection_quotient_bounds_m"] == I(-1, 1)
    assert contracted["projection_quotient_cauchy_absolute_upper_bound_m"] == 1
    assert contracted["projection_gradient_bounds"][0] == I(1)
    assert all(x == I(0) for x in contracted["projection_gradient_bounds"][1:])
    assert all(x == I(0) for row in contracted["projection_hessian_bounds"] for x in row)
    with pytest.raises(ValueError, match="Cauchy|norm"):
        api._projection_algebra(I(1), I(2, 3), zi, zh, I(1), zi, zh, zero, 128)
    with pytest.raises(ValueError, match="Cauchy|norm"):
        api._projection_algebra(I(1), I(0), zi, zh, I(-2, -1), zi, zh, zero, 128)


def test_one_global_inverse_twojet_preserves_source_hessian_and_linear_injection_does_not():
    system, response = _system(), _response()
    origin, current = system.reference_origin_inverse_2jet, response.current_source_inverse_2jet
    assert origin.anchor == "STRUCTURAL_GREEN_ZERO_REFERENCE_NO_ZERO_QUADRATURE"
    assert current.anchor == "ACTUAL_NONZERO_REGISTERED_COLD_PROBE"
    assert origin.global_anchor_coordinate_increments == (F(0),)*92
    assert current.global_anchor_coordinate_increments == system.bound_coordinate_increments
    assert origin.source_anchor_coordinate_bounds == (I(0),)*14
    assert current.source_anchor_coordinate_bounds == response.source_coordinate_increment_bounds
    assert len(origin.source_to_global_first_derivative_bounds) == 92
    for inverse in (origin, current):
        assert inverse.positive_tip_derivative_certified
        assert not inverse.zero_bulk_quadrature_invented
        assert inverse.source_only_operator_certificate
        assert not inverse.total_bulk_potential_hessian_identified_with_old_source
        assert len(inverse.source_to_global_second_derivative_blocks) == 8
        for value in inverse.first_composition_identity_residual_bounds:
            _encloses(value, F(0))
        for value in inverse.second_composition_identity_residual_bounds:
            _encloses(value, F(0))
    for n, block in enumerate(system.projection_blocks):
        raw = next(raw for body, material, raw in _contexts() if body.component_id == block.component_id)
        curvature = raw.projection_hessian_n_m2[0][0]/raw.reference_tip_strain_stiffness_n_m
        _overlaps(origin.inverse_tip_second_derivative_bounds[n], -curvature)
        assert origin.inverse_tip_first_derivative_bounds[n] == I(1)
        ds = block.projection_gradient_bounds[0]
        _overlaps(current.inverse_tip_first_derivative_bounds[n], 1/ds)
        _overlaps(current.inverse_tip_second_derivative_bounds[n], -curvature/(ds*ds*ds))
        i, r = block.global_coordinate_indices[0], block.source_coordinate_index
        assert system.reference_source_injection[i][r] == F(1)
        alpha, gamma = current.inverse_tip_first_derivative_bounds[n], current.inverse_tip_second_derivative_bounds[n]
        source_g = response.potential_gradient_bounds[i]-response.orthogonal_gradient_bounds[i]
        source_h = response.potential_hessian_bounds[i][i]-response.orthogonal_hessian_bounds[i][i]
        full = alpha*source_h*alpha+source_g*gamma
        _overlaps(full, response.source_hessian_bounds[r][r])
        _overlaps(response.source_only_inverse_pullback_hessian_bounds[r][r], full)
    for i in range(14):
        _encloses(response.source_only_inverse_first_variation_residual_bounds[i], F(0))
        for j in range(14):
            _encloses(response.source_only_inverse_second_variation_residual_bounds[i][j], F(0))
    assert not response.total_finite_work_identified_with_old_source_work


def test_system_semantic_replay_rejects_context_projection_and_energy_ownership_tampering():
    api, system, contexts, transfer = _api(), _system(), _contexts(), _transfer()
    raw = api.native_common_strain_projection_system_to_dict(system)
    restored = api.native_common_strain_projection_system_from_dict(raw, transfer=transfer, cold_contexts=contexts)
    assert api.native_common_strain_projection_system_to_dict(restored) == raw
    changes = (
        lambda value: value["bound_coordinate_increments"].__setitem__(6, [1, 1]),
        lambda value: value["projection_blocks"][0].__setitem__("trace_id", "another-source"),
        lambda value: value["projection_blocks"][0].__setitem__("raw_integral_sha256", "0"*64),
        lambda value: value["projection_blocks"][0].__setitem__("projection_increment_bounds_m", {"lo": [1, 1], "hi": [1, 1]}),
        lambda value: value["projection_blocks"][0].__setitem__("projection_quotient_cauchy_absolute_upper_bound_m", [0, 1]),
        lambda value: value["projection_blocks"][0].__setitem__("projection_quotient_bounds_m", {"lo": [0, 1], "hi": [0, 1]}),
        lambda value: value.__setitem__("source_beam_energy_rounding_residual_j", [0, 1]),
        lambda value: value.__setitem__("physical_source_material_handover_certified", True),
    )
    for change in changes:
        forged = deepcopy(raw)
        change(forged)
        _json_seal(forged, "binding_sha256", api.NativeCommonStrainProjectionSystem)
        with pytest.raises(ValueError):
            api.native_common_strain_projection_system_from_dict(forged, transfer=transfer, cold_contexts=contexts)
    body, material, integral = contexts[0]
    # A byte-identical JSON payload does not authenticate Python context types.
    counterfeit_type = make_dataclass("CounterfeitCommonTensorIntegral",
        [(f.name, f.type) for f in fields(integral)], frozen=True, slots=True)
    counterfeit = counterfeit_type(**{f.name: getattr(integral, f.name) for f in fields(integral)})
    list_coordinates = replace(integral, coordinates_m=list(integral.coordinates_m))
    for fake in (counterfeit, list_coordinates):
        assert _control().payload(fake) == _control().payload(integral)
        fake_contexts = ((body, material, fake),)+contexts[1:]
        with pytest.raises(ValueError):
            api.bind_native_common_strain_projection(transfer, cold_contexts=fake_contexts,
                                                     provenance=system.provenance)
        with pytest.raises(ValueError):
            api.validate_native_common_strain_projection_system(system, cold_contexts=fake_contexts)
    closed = _control().authentic_closed_cold_contexts()
    assert len(closed) == 8
    old_by_id = {b.component_id: (b, m, r) for b, m, r in closed}
    for fresh_body, fresh_material, fresh_raw in contexts:
        old_body, old_material, old_raw = old_by_id[fresh_body.component_id]
        assert fresh_body.reference_snapshot_sha256 == transfer.reference_snapshot_sha256
        assert old_body.reference_snapshot_sha256 != fresh_body.reference_snapshot_sha256
        assert old_raw.coordinates_m == fresh_raw.coordinates_m
        for name in ("quarter_integral_bounds", "removed_material_correction_bounds",
                     "maximum_taylor_remainder_absolute_upper_bounds", "cell_receipts", "correction_receipts"):
            assert getattr(old_raw, name) == getattr(fresh_raw, name)
        assert old_raw.integral_snapshot_sha256 != fresh_raw.integral_snapshot_sha256
    equivalence = _control().reference_precision_equivalence()
    assert equivalence["equivalence_sha256"]
    assert equivalence["physical_definition_bank_count"] == 13
    assert equivalence["compared_all_reference_and_bank_fields_except_declared_precision_and_binding_ids"]
    assert equivalence["closed_geometry_enclosure_precision"] == 160
    assert equivalence["current_geometry_enclosure_precision"] == 96
    assert not equivalence["original_closed_context_rewritten"]
    assert not equivalence["fresh_tensor_quadrature_executed"]
    forged_raw = replace(integral, projection_remainder_n=integral.projection_remainder_n+1)
    with pytest.raises(ValueError):
        api.bind_native_common_strain_projection(transfer,
            cold_contexts=((body, material, forged_raw),)+contexts[1:], provenance=system.provenance)
    with pytest.raises(ValueError):
        api.bind_native_common_strain_projection(transfer, cold_contexts=contexts[:-1], provenance=system.provenance)
    for key in ("separate_source_beam_energized", "full_common_calibrated_cold_energy_energized",
                "continuous_source_measure_energized"):
        for value in (True, 1):
            with pytest.raises(ValueError):
                api.evaluate_native_common_strain_projection(system, **{key: value})
    for coordinates in ((F(0),)*92, list(system.bound_coordinate_increments), (F(0),)*100):
        with pytest.raises(ValueError):
            api.evaluate_native_common_strain_projection(system, coordinate_increments=coordinates)
    # Cached queries must still validate the complete frozen raw dependency.
    original = integral.projection_remainder_n
    try:
        object.__setattr__(integral, "projection_remainder_n", original+1)
        with pytest.raises(ValueError):
            api.evaluate_native_common_strain_projection(system)
    finally:
        object.__setattr__(integral, "projection_remainder_n", original)
    # Full bank context matters even when radius and COMMON A are unchanged.
    bank = body.embedding_bank
    roof = bank.roof_radial_displacement_m
    try:
        object.__setattr__(bank, "roof_radial_displacement_m", roof+1)
        with pytest.raises(ValueError):
            api.evaluate_native_common_strain_projection(system)
    finally:
        object.__setattr__(bank, "roof_radial_displacement_m", roof)
    other_bank = next(b for b in _control().source_context()[1].banks if b.material_kind == "hot_birth")
    other_roof = other_bank.roof_radial_displacement_m
    try:
        object.__setattr__(other_bank, "roof_radial_displacement_m", other_roof+1)
        with pytest.raises(ValueError):
            api.evaluate_native_common_strain_projection(system)
    finally:
        object.__setattr__(other_bank, "roof_radial_displacement_m", other_roof)
    api.validate_native_common_strain_projection_system(system)


def test_response_and_complete_packet_replay_reject_valid_query_transplants():
    api, control, system = _api(), _control(), _system()
    rows = []
    for arguments in control.query_cases(system):
        response = _response(arguments["branch"])
        raw = api.native_common_strain_projection_response_to_dict(response)
        assert api.native_common_strain_projection_response_to_dict(
            api.native_common_strain_projection_response_from_dict(raw, system=system)) == raw
        rows.append({"arguments": control.payload(arguments), "response": raw})
    for field in ("source_gradient_bounds", "potential_gradient_bounds", "chainrule_curvature_hessian_bounds",
                  "orthogonal_gradient_bounds", "source_only_inverse_second_variation_residual_bounds"):
        forged = deepcopy(rows[0]["response"])
        target = forged[field]
        while type(target[0]) is list:
            target = target[0]
        target[0] = {"lo": [1, 1], "hi": [1, 1]}
        _json_seal(forged, "binding_sha256", api.NativeCommonStrainProjectionResponse)
        with pytest.raises(ValueError):
            api.native_common_strain_projection_response_from_dict(forged, system=system)
    from tectonics import geometric_native_common_material_transfer as material
    packet = dict(format="native-common-bankwise-cold-strain-projection-packet-1",
        prior_871_packet_raw_sha256=control.PRIOR_871_PACKET_RAW,
        prior_871_packet_sha256=control.PRIOR_871_PACKET_SEAL,
        prior_871_wrapper_raw_sha256=control.PRIOR_871_WRAPPER_RAW,
        prior_841_packet_raw_sha256=control.PRIOR_841_PACKET_RAW,
        prior_841_packet_sha256=control.PRIOR_841_PACKET_SEAL,
        material_transfer=material.native_common_material_transfer_to_dict(_transfer()),
        authentic_cold_contexts=control.cold_source_records(),
        reference_precision_equivalence=control.reference_precision_equivalence(),
        strain_projection=api.native_common_strain_projection_system_to_dict(system),
        global_queries=rows, **control.summary(2))
    _json_seal(packet, "packet_sha256")
    assert control.replay_packet(packet) == control.summary(2)
    forged = deepcopy(packet)
    forged["global_queries"][0]["response"], forged["global_queries"][1]["response"] = (
        forged["global_queries"][1]["response"], forged["global_queries"][0]["response"])
    _json_seal(forged, "packet_sha256")
    with pytest.raises(ValueError):
        control.replay_packet(forged)
