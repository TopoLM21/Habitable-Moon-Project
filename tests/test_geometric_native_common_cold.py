"""Independent common-reference cold geometry and actual-volume invariants.

The Decimal oracle works in the physical world frame, replays the literal
owner rotation, and never uses production angular/Taylor contractions.  The
whole-quarter oracle independently integrates polynomials in beta, physical
radius and the true common depth-shear direction.
"""
from dataclasses import replace
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
from functools import lru_cache
from math import comb
import json

import pytest

from tectonics.geometric_native_pose import FractionInterval as I


def _api():
    from tectonics import geometric_native_common_cold as module
    return module


@lru_cache(maxsize=1)
def _reference():
    from tectonics.geometric_native_common_reference import bind_native_common_reference
    return bind_native_common_reference(precision=96)


@lru_cache(maxsize=8)
def _body(parent):
    bank = next(b for b in _reference().banks if b.material_kind == "original_live"
                and b.original_parent_id == parent)
    return _api().bind_native_common_cold_body(bank.component_id, reference=_reference(), precision=96)


@lru_cache(maxsize=8)
def _material(parent):
    return _api().build_native_common_cold_strain_material(_body(parent))


def _d(value):
    return D(value.numerator)/D(value.denominator) if type(value) is F else D(value)


def _sin_cos(value):
    sine, cosine, a, b = value, D(1), value, D(1)
    for n in range(1, 160):
        a *= -value*value/D(2*n*(2*n+1))
        b *= -value*value/D((2*n-1)*2*n)
        sine += a
        cosine += b
    return sine, cosine


def _atan(value):
    answer, term = value, value
    for n in range(1, 190):
        term *= -value*value
        answer += term/D(2*n+1)
    return answer


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), D(0))


def _cross(a, b):
    return a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]


def _map(matrix, vector):
    return tuple(_dot(row, vector) for row in matrix)


def _transpose(matrix):
    return tuple(zip(*matrix))


def _rotate(vector, angle_vector):
    norm = _dot(angle_vector, angle_vector).sqrt()
    if not norm:
        return vector
    sine, cosine = _sin_cos(norm)
    cross = _cross(angle_vector, vector)
    second = _cross(angle_vector, cross)
    return tuple(vector[i]+sine/norm*cross[i]+(1-cosine)/(norm*norm)*second[i] for i in range(3))


def _owner_frame(body):
    columns = tuple(tuple(_d(x) for x in axis) for axis in
                    (body.original_quarter_start, body.original_quarter_end, body.original_quarter_inward))
    for step in json.loads(body.source_owner_history_json):
        omega = tuple(_d(F.from_float(x))*_d(F.from_float(step["dt_myr"])) for x in step["omega_rad_per_myr"])
        columns = tuple(_rotate(axis, omega) for axis in columns)
    return _transpose(columns)


def _point_oracle(body, theta, beta, z):
    R, top = _d(body.radius_m), _d(body.radius_m+body.roof_radial_displacement_m)
    r = (top**3-3*R*R*z)**(D(1)/3)
    tau = (R**3-r**3)/(3*R*R)
    st, ct = _sin_cos(theta)
    sb, cb = _sin_cos(beta)
    P = _owner_frame(body)
    direction = _map(P, (ct*cb, st*cb, sb))
    angle = tuple(_d(x)*tau for x in body.angular_rate_per_physical_volume_depth_m)
    return tuple(r*x for x in _rotate(direction, angle)), P


def _world_modes(body, world, P):
    R = _d(body.radius_m)
    r = _dot(world, world).sqrt()
    tau = (R**3-r**3)/(3*R*R)
    original = _rotate(tuple(x/r for x in world), tuple(-_d(x)*tau for x in body.angular_rate_per_physical_volume_depth_m))
    v = _map(_transpose(P), original)
    beta = _atan(v[2]/(v[0]*v[0]+v[1]*v[1]).sqrt())
    t = beta/_d(body.beta_cut_radians)
    chi = (1-t)**2*(t+2)/2
    modes = [tuple(chi*x/r for x in world)]
    for a, b in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)):
        modes.append(tuple(chi/R*(world[a] if i == a and a == b else
            world[b] if i == a else world[a] if i == b else D(0)) for i in range(3)))
    return tuple(modes)


def _contains(bounds, expected):
    expected = F(expected) if type(expected) is D else expected
    assert bounds.lo <= expected <= bounds.hi


def test_all_eight_authentic_bindings_keep_source_and_active_reference_separate():
    from tectonics.geometric_native_cold_body import validate_native_cold_body
    changed = []
    for n in range(8):
        body = _body("octant:"+str(n))
        assert validate_native_cold_body(body.original_body) is body.original_body
        assert body.angular_rate_per_physical_volume_depth_m == _reference().angular_rate_per_physical_volume_depth_m
        assert body.historical_angular_rate_per_physical_volume_depth_m == body.original_body.angular_rate_per_physical_volume_depth_m
        assert body.body_snapshot_sha256 != body.original_body.body_snapshot_sha256
        material = _material("octant:"+str(n))
        assert material.source_material_snapshot_sha256 == material.original_material.material_snapshot_sha256
        assert material.body_snapshot_sha256 == body.body_snapshot_sha256
        support = _api().certify_native_common_cold_strain_support(body, (F(1, 100),)*7)
        assert support.active_angular_rate_per_physical_volume_depth_m == body.angular_rate_per_physical_volume_depth_m
        assert support.whole_mask_local_positive_jacobian_certified
        assert support.new_transported_three_dimensional_clamp_used
        if body.angular_rate_per_physical_volume_depth_m != body.historical_angular_rate_per_physical_volume_depth_m:
            changed.append(n)
    assert changed == [0, 1, 3, 4, 5, 7]


@pytest.mark.parametrize("parent", ["octant:0", "octant:4"])
def test_changed_zero_rate_receiver_and_reflected_donor_gradients_match_world_finite_differences(parent):
    from tectonics.geometric_native_curved_certified_integration import TaylorAlgebra
    body = _body(parent)
    assert not any(body.historical_angular_rate_per_physical_volume_depth_m)
    theta, beta, z = F(3, 5), body.beta_cut_radians/3, body.areal_material_thickness_m*F(3, 4)
    algebra = TaylorAlgebra(0, 96)
    c = algebra.constant
    geometry = _api().evaluate_native_common_cold_modal_geometry(body,
        algebra.sin(c(theta)), algebra.cos(c(theta)), c(beta), c(z), algebra)
    with localcontext() as context:
        context.prec = 155
        world, P = _point_oracle(body, _d(theta), _d(beta), _d(z))
        for bounds, expected in zip(geometry.world_point_m, world):
            _contains(bounds.constant_bounds, expected)
        expected_modes = _world_modes(body, world, P)
        for mode, expected in zip(geometry.mode_displacement_per_m, expected_modes):
            for value, oracle in zip(mode, expected):
                _contains(value.constant_bounds, oracle)
        h = D("0.001")
        for axis in range(3):
            samples = {}
            for offset in (-2, -1, 1, 2):
                point = tuple(x+(h*offset if i == axis else D(0)) for i, x in enumerate(world))
                samples[offset] = _world_modes(body, point, P)
            for mode in range(7):
                for row in range(3):
                    derivative = (-samples[2][mode][row]+8*samples[1][mode][row]
                        -8*samples[-1][mode][row]+samples[-2][mode][row])/(12*h)
                    bounds = geometry.mode_gradient_m_inv[mode][row][axis].constant_bounds
                    midpoint = (_d(bounds.lo)+_d(bounds.hi))/2
                    assert abs(midpoint-derivative) < D("1e-24")
        clamp = _api().bound_native_common_cold_point(body, I(theta), I(body.beta_cut_radians), I(z), precision=96)
        outside = _api().bound_native_common_cold_point(body, I(theta), I(body.beta_cut_radians*F(3, 2)), I(z), precision=96)
        for value in (clamp, outside):
            assert all(x == I(0) for mode in value.modal_displacement_bounds for x in mode)
            assert all(x == I(0) for matrix in value.modal_gradient_bounds_m_inv for row in matrix for x in row)
            assert value.incremental_determinant_bounds == I(1)
            assert not value.field_active_inside_clamp


def test_resealed_source_or_embedding_cannot_enter_the_new_common_binding():
    body = _body("octant:0")
    forged = replace(body, embedding_bank=replace(body.embedding_bank,
        angular_rate_per_physical_volume_depth_m=(F(0),)*3))
    with pytest.raises(ValueError, match="altered|common|reference|bank|embedding"):
        _api().validate_native_common_cold_body(forged)
    material = _material("octant:0")
    with pytest.raises(ValueError, match="altered|common|material"):
        _api().validate_native_common_cold_strain_material(replace(material,
            body_snapshot_sha256=body.original_body.body_snapshot_sha256), body=body)


def _world_gradient_oracle(body, theta, beta, z):
    world, P = _point_oracle(body, theta, beta, z)
    R = _d(body.radius_m)
    r = _dot(world, world).sqrt()
    e = tuple(x/r for x in world)
    tau = (R**3-r**3)/(3*R*R)
    A = tuple(_d(x) for x in body.angular_rate_per_physical_volume_depth_m)
    angle = tuple(x*tau for x in A)
    p = _rotate(e, tuple(-x for x in angle))
    n = tuple(row[2] for row in P)
    sb, cb = _sin_cos(beta)
    transverse = tuple((n[i]-sb*p[i])/cb for i in range(3))
    rotated = _rotate(transverse, angle)
    rate = _dot(n, _cross(A, p))/cb
    gbeta = tuple(rotated[i]/r+r*r/(R*R)*rate*e[i] for i in range(3))
    t = beta/_d(body.beta_cut_radians)
    chi, slope = (1-t)**2*(t+2)/2, -D("1.5")*(1-t*t)/_d(body.beta_cut_radians)
    matrices = [tuple(tuple(chi/r*(D(int(i == j))-e[i]*e[j])+slope*e[i]*gbeta[j]
                           for j in range(3)) for i in range(3))]
    for a, b in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)):
        E = tuple(tuple(D(int(i == j == a)) if a == b else D(int((i, j) in ((a, b), (b, a))))
                        for j in range(3)) for i in range(3))
        direction = _map(E, world)
        matrices.append(tuple(tuple(chi/R*E[i][j]+slope/R*direction[i]*gbeta[j]
                                    for j in range(3)) for i in range(3)))
    return tuple(matrices)


def test_all_seven_point_forces_and_full_hessian_match_independent_world_quartic_variations():
    body, material = _body("octant:4"), _material("octant:4")
    theta, beta, z = F(3, 5), body.beta_cut_radians/3, body.areal_material_thickness_m*F(3, 4)
    q = tuple(F(n+1, 100) for n in range(7))
    primary = _api().bound_native_common_cold_tensor_density(body, material, q,
        I(theta), I(beta), I(z), precision=96, layer="lower")
    k, prem, prem_i = primary[0], primary[1], primary[2:9]
    pairs = tuple((i, j) for i in range(7) for j in range(i, 7))
    p_ij = dict(zip(pairs, primary[9:37]))
    rem_h = dict(zip(pairs, primary[45:73]))
    full_gradient = tuple((q[0]*k+prem if i == 0 else I(0))+q[0]*prem_i[i]+primary[38+i] for i in range(7))
    full_hessian = {(i, j): (k if i == j == 0 else I(0))+(prem_i[j] if i == 0 else I(0))
        +(prem_i[i] if j == 0 else I(0))+rem_h[i, j]+q[0]*p_ij[i, j] for i, j in pairs}
    with localcontext() as context:
        context.prec = 155
        gradients = _world_gradient_oracle(body, _d(theta), _d(beta), _d(z))
        H0 = tuple(tuple((gradients[0][i][j]+gradients[0][j][i])/2 for j in range(3)) for i in range(3))
        lam, shear = map(_d, material.lame_coefficients("lower"))
        def inner(first, second):
            return lam*sum(first[i][i] for i in range(3))*sum(second[i][i] for i in range(3))\
                +2*shear*sum(first[i][j]*second[i][j] for i in range(3) for j in range(3))
        coordinates = tuple(_d(x) for x in q)
        def energy(values):
            L = tuple(tuple(sum(values[a]*gradients[a][i][j] for a in range(7)) for j in range(3)) for i in range(3))
            green = tuple(tuple((L[i][j]+L[j][i]+sum(L[a][i]*L[a][j] for a in range(3)))/2
                                for j in range(3)) for i in range(3))
            return inner(green, green)/2, inner(H0, green)
        def shifted(*increments):
            values = list(coordinates)
            for index, amount in increments:
                values[index] += amount
            return energy(values)[0]
        value, projection = energy(coordinates)
        _contains(k, inner(H0, H0))
        _contains(q[0]*k+prem, projection)
        _contains((q[0]*q[0]*k+2*q[0]*prem+primary[37])/2, value)
        h = _d(F(7, 11))
        for i in range(7):
            samples = {n: shifted((i, n*h)) for n in (-2, -1, 1, 2)}
            derivative = (-samples[2]+8*samples[1]-8*samples[-1]+samples[-2])/(12*h)
            curvature = (-samples[2]+16*samples[1]-30*value+16*samples[-1]-samples[-2])/(12*h*h)
            _contains(full_gradient[i], derivative)
            _contains(full_hessian[i, i], curvature)
            for j in range(i+1, 7):
                def mixed(step):
                    return (shifted((i, step), (j, step))-shifted((i, step), (j, -step))
                        -shifted((i, -step), (j, step))+shifted((i, -step), (j, -step)))/(4*step*step)
                _contains(full_hessian[i, j], (4*mixed(h/2)-mixed(h))/3)


# Independent Laurent polynomials: (power of beta/b, depth-rate direction,
# physical radius).  Their coefficient arithmetic is literal rational;
# negative radial exponents are needed only by the radial-tip strain.
def _add(*polynomials):
    result = {}
    for p in polynomials:
        for key, value in p.items():
            result[key] = result.get(key, F(0))+value
    return {key: value for key, value in result.items() if value}


def _scale(p, coefficient):
    return {key: value*coefficient for key, value in p.items() if value*coefficient}


def _multiply(a, b):
    result = {}
    for i, x in a.items():
        for j, y in b.items():
            key = tuple(k+l for k, l in zip(i, j))
            result[key] = result.get(key, F(0))+x*y
    return {key: value for key, value in result.items() if value}


def _power(p, n):
    answer = {(0, 0, 0): F(1)}
    for _ in range(n):
        answer = _multiply(answer, p)
    return answer


def _quarter_moment(i, j, pi):
    if j >= 2:
        return D(j-1)/D(i+j)*_quarter_moment(i, j-2, pi)
    if j == 1:
        return D(1)/D(i+1)
    if i >= 2:
        return D(i-1)/D(i)*_quarter_moment(i-2, 0, pi)
    return pi/2 if i == 0 else D(1)


def _integrate_polynomial(p, body, top, bottom, theta_rate_coefficients, pi):
    b = _d(body.beta_cut_radians)
    sb, cb = _sin_cos(b)
    degree = max(key[0] for key in p)
    moments = [sb, b*sb+cb-1]
    for n in range(2, degree+1):
        moments.append(b**n*sb+n*b**(n-1)*cb-n*(n-1)*moments[n-2])
    d, e = theta_rate_coefficients
    angular = {}
    for n in set(key[1] for key in p):
        angular[n] = sum((D(comb(n, k))*d**k*e**(n-k)*_quarter_moment(k, n-k, pi)
                          for k in range(n+1)), D(0))
    result = D(0)
    for (t_power, rate_power, radial_power), coefficient in p.items():
        assert radial_power != -3
        radial = (top**(radial_power+3)-bottom**(radial_power+3))/D(radial_power+3)
        result += _d(coefficient)*moments[t_power]/b**t_power*angular[rate_power]*radial
    return result


def _quarter_oracle(body, material, amplitude):
    """Exact beta/theta/radius moments for genuine nonzero common A*."""
    assert body.endpoint_archive_area_bounds_sr == I(0)
    assert not body.paired_curved_removed_area_bounds_sr
    with localcontext() as context:
        context.prec = 155
        shape = {(0, 0, 0): F(1), (1, 0, 0): F(-3, 2), (3, 0, 0): F(1, 2)}
        slope = {(0, 0, 0): F(-3, 2)/body.beta_cut_radians, (2, 0, 0): F(3, 2)/body.beta_cut_radians}
        one = {(0, 0, 0): F(1)}
        R = body.radius_m
        a = _scale(shape, amplitude/R)
        y = _scale(slope, amplitude/R)
        d = _multiply(_scale(slope, amplitude/R**3), {(0, 1, 3): F(1)})
        # F=[[1+a+d,y,0],[0,1+a,0],[0,0,1+a]] in (e,t_beta,t_theta).
        ee = _scale(_add(_power(_add(one, a, d), 2), _scale(one, -1)), F(1, 2))
        tt = _scale(_add(_power(_add(one, a), 2), _power(y, 2), _scale(one, -1)), F(1, 2))
        hh = _scale(_add(_power(_add(one, a), 2), _scale(one, -1)), F(1, 2))
        et = _scale(_multiply(_add(one, a, d), y), F(1, 2))
        trace = _add(ee, tt, hh)
        norm = _add(_power(ee, 2), _power(tt, 2), _power(hh, 2), _scale(_power(et, 2), 2))
        hrad = _multiply(_scale(slope, F(1)/R**2), {(0, 1, 2): F(1)})
        htan = _multiply(shape, {(0, 0, -1): F(1)})
        hshear = _multiply(_scale(slope, F(1, 2)), {(0, 0, -1): F(1)})
        htrace = _add(hrad, _scale(htan, 2))
        hnorm = _add(_power(hrad, 2), _scale(_power(htan, 2), 2), _scale(_power(hshear, 2), 2))
        hp = _add(_multiply(hrad, ee), _multiply(htan, _add(tt, hh)), _scale(_multiply(hshear, et), 2))
        P = _owner_frame(body)
        A = tuple(_d(x) for x in body.angular_rate_per_physical_volume_depth_m)
        columns = _transpose(P)
        rate = (_dot(columns[2], _cross(A, columns[0])), _dot(columns[2], _cross(A, columns[1])))
        pi = 16*_atan(D(1)/5)-4*_atan(D(1)/239)
        top = _d(body.radius_m+body.roof_radial_displacement_m)
        middle = top-_d(body.elastic_thickness_m)
        bottom = (top**3-3*_d(R)**2*_d(body.areal_material_thickness_m))**(D(1)/3)
        metric, projection, raw_green = D(0), D(0), D(0)
        for layer, upper, lower in (("upper", top, middle), ("lower", middle, bottom)):
            lam, shear = material.lame_coefficients(layer)
            ug = _add(_scale(_power(trace, 2), lam/2), _scale(norm, shear))
            p = _add(_scale(_multiply(htrace, trace), lam), _scale(hp, 2*shear))
            k = _add(_scale(_power(htrace, 2), lam), _scale(hnorm, 2*shear))
            raw_green += _integrate_polynomial(ug, body, upper, lower, rate, pi)
            projection += _integrate_polynomial(p, body, upper, lower, rate, pi)
            metric += _integrate_polynomial(k, body, upper, lower, rate, pi)
        wref, old = _d(body.accepted_tip_displacement_m), _d(body.source_old_tip_stiffness_n_m)
        calibrated = old*(wref+projection/metric)**2/2+raw_green-projection**2/(2*metric)
        return calibrated, raw_green, projection, metric


@lru_cache(maxsize=1)
def _whole_case():
    body, material, amplitude = _body("octant:0"), _material("octant:0"), F(17, 10)
    q = (F(0), amplitude, amplitude, amplitude, F(0), F(0), F(0))
    response = _api().evaluate_native_common_cold_strain_energy(body, q, material=material,
        precision=64, taylor_order=0, beta_cell_count=1, depth_cell_count=1)
    return body, material, amplitude, response


def test_actual_changed_zero_rate_whole_volume_matches_independent_common_shear_invariants():
    body, material, amplitude, response = _whole_case()
    assert not any(body.historical_angular_rate_per_physical_volume_depth_m)
    assert any(body.angular_rate_per_physical_volume_depth_m)
    energy, raw_green, projection, metric = _quarter_oracle(body, material, amplitude)
    _contains(response.energy_bounds_j, energy)
    _contains(response.raw_green_energy_bounds_j, raw_green)
    _contains(response.raw_integrals.projection_n, projection)
    _contains(response.raw_integrals.reference_tip_strain_stiffness_n_m, metric)
    assert response.raw_integrals.reference_tip_strain_stiffness_n_m.lo > 0
    assert response.raw_integrals.integration_cell_count == 2
    assert len(response.generalized_gradient_bounds_n) == 7
    assert all(len(row) == 7 for row in response.generalized_hessian_bounds_n_m)
    assert response.generalized_hessian_bounds_n_m == tuple(zip(*response.generalized_hessian_bounds_n_m))
    assert response.reference_energy_j == body.source_old_tip_stiffness_n_m*body.accepted_tip_displacement_m**2/2 > 0
    assert response.reference_tip_force_n == body.source_old_tip_stiffness_n_m*body.accepted_tip_displacement_m != 0
    assert not response.mixed_force_equilibrium_certified
    assert not response.material_admission_authorized


def test_actual_common_reference_whole_force_and_hessian_match_quartic_energy_variations():
    body, material, amplitude, response = _whole_case()
    with localcontext() as context:
        context.prec = 130
        h = F(7, 11)
        values = {n: _quarter_oracle(body, material, amplitude+n*h)[0] for n in (-2, -1, 0, 1, 2)}
        derivative = (-values[2]+8*values[1]-8*values[-1]+values[-2])/(12*_d(h))
        curvature = (-values[2]+16*values[1]-30*values[0]+16*values[-1]-values[-2])/(12*_d(h)**2)
    _contains(sum(response.generalized_gradient_bounds_n[1:4], I(0)), derivative)
    _contains(sum((response.generalized_hessian_bounds_n_m[i][j] for i in (1, 2, 3) for j in (1, 2, 3)), I(0)), curvature)
    exported = _api().native_common_cold_strain_energy_to_dict(response)
    assert exported["raw_integrals"]["integral_snapshot_sha256"] == response.raw_integrals.integral_snapshot_sha256
    assert exported["common_body_binding"]["original_body"]["angular_rate_per_physical_volume_depth_m"] == [[0, 1]]*3
    assert exported["body"]["angular_rate_role"] == "NEW_computational_embedding_not_original_saved128_source"


def test_actual_nonzero_source_prestress_on_changed_donor_keeps_exact_reference_energy_and_tip_force():
    body, material = _body("octant:4"), _material("octant:4")
    response = _api().evaluate_native_common_cold_strain_energy(body, (F(0),)*7,
        material=material, precision=64, taylor_order=0)
    assert body.accepted_tip_displacement_m != 0
    assert body.source_old_tip_stiffness_n_m > 0
    assert response.energy_bounds_j == I(body.source_old_tip_stiffness_n_m*body.accepted_tip_displacement_m**2/2)
    assert response.generalized_gradient_bounds_n[0] == I(body.source_old_tip_stiffness_n_m*body.accepted_tip_displacement_m)
    raw = response.raw_integrals
    expected_h00 = I(body.source_old_tip_stiffness_n_m)+response.reference_tip_force_n/raw.reference_tip_strain_stiffness_n_m*raw.projection_hessian_n_m2[0][0]
    assert response.generalized_hessian_bounds_n_m[0][0] == expected_h00
    assert raw.active_angular_rate_per_physical_volume_depth_m != raw.historical_angular_rate_per_physical_volume_depth_m
    assert raw.reference_tip_strain_stiffness_n_m.lo > 0
    assert raw.remainder_strain_inner_product_j == I(0)
