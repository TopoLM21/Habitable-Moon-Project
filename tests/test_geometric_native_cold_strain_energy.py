"""Independent analytic quarter energy for supported, prestressed cold fields.

The oracle uses orthonormal strain invariants, polynomial cosine moments and
exact radial integrals.  It does not reuse the production angular polynomial
algebra, Taylor integrator, or calibrated-energy evaluator.
"""
from dataclasses import replace
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
from functools import lru_cache

import pytest

from tectonics.geometric_native_pose import FractionInterval as I


ZERO = F(0)


def _api():
    from tectonics import geometric_native_cold_strain_energy as module
    return module


@lru_cache(maxsize=1)
def _source():
    from tectonics.geometric_native_normal_support import load_native_normal_support_source
    return load_native_normal_support_source()


@lru_cache(maxsize=8)
def _body(parent):
    from tectonics.geometric_native_cold_body import bind_native_cold_body
    bank = next(bank for bank in _source().banks
        if bank.material_kind == "original_live" and bank.original_parent_id == parent)
    return bind_native_cold_body(bank.component_id, source=_source(), precision=96)


@lru_cache(maxsize=8)
def _material(parent):
    return _api().build_native_cold_strain_material(_body(parent))


def _decimal(value):
    return D(value.numerator)/D(value.denominator)


def _atan_small(value):
    total, term = value, value
    for index in range(1, 230):
        term *= -value*value
        total += term/D(2*index+1)
    return total


def _sin_cos(value):
    sine, cosine = value, D(1)
    first, second = value, D(1)
    for index in range(1, 150):
        first *= -value*value/D(2*index*(2*index+1))
        second *= -value*value/D((2*index-1)*2*index)
        sine += first
        cosine += second
    return sine, cosine


def _add(*polynomials):
    size = max(map(len, polynomials))
    return tuple(sum((p[index] if index < len(p) else ZERO for p in polynomials), ZERO)
        for index in range(size))


def _scale(polynomial, coefficient):
    return tuple(coefficient*value for value in polynomial)


def _multiply(first, second):
    result = [ZERO]*(len(first)+len(second)-1)
    for i, a in enumerate(first):
        for j, b in enumerate(second):
            result[i+j] += a*b
    return tuple(result)


def _cosine_integral(polynomial, beta_cut):
    """Integrate a polynomial in t=beta/b against cos(beta), independently."""
    b = _decimal(beta_cut)
    sine, cosine = _sin_cos(b)
    moments = [sine, b*sine+cosine-1]
    for degree in range(2, len(polynomial)):
        moments.append(b**degree*sine+degree*b**(degree-1)*cosine
            -degree*(degree-1)*moments[degree-2])
    return sum((_decimal(value)*moments[degree]/b**degree
        for degree, value in enumerate(polynomial)), D(0))


def _quarter_oracle(body, material, amplitude):
    """Return calibrated U, raw Green U, projection p and raw tip metric."""
    assert not any(body.angular_rate_per_physical_volume_depth_m)
    assert body.archive_area_bounds_sr == I(ZERO)
    with localcontext() as context:
        context.prec = 110
        shape = (F(1), F(-3, 2), ZERO, F(1, 2))
        slope = (F(-3, 2)/body.beta_cut_radians, ZERO,
            F(3, 2)/body.beta_cut_radians)
        coefficient = amplitude/body.radius_m
        dilation = _add((F(1),), _scale(shape, coefficient))
        shear_strain = _scale(slope, coefficient)
        dilation_squared = _multiply(dilation, dilation)
        shear_squared = _multiply(shear_strain, shear_strain)
        green_diagonal = _scale(_add(dilation_squared, (F(-1),)), F(1, 2))
        green_squared = _multiply(green_diagonal, green_diagonal)
        trace = _add(_scale(green_diagonal, F(3)), _scale(shear_squared, F(1, 2)))
        tensor_norm = _add(_scale(green_squared, F(3)),
            _multiply(green_diagonal, shear_squared),
            _scale(_multiply(shear_squared, shear_squared), F(1, 4)),
            _scale(_multiply(dilation_squared, shear_squared), F(1, 2)))
        radius = _decimal(body.radius_m)
        top = _decimal(body.radius_m+body.roof_radial_displacement_m)
        elastic_bottom = top-_decimal(body.elastic_thickness_m)
        bottom = (top**3-3*radius**2*_decimal(body.areal_material_thickness_m))**(D(1)/D(3))
        upper_young, upper_nu = material.upper_young_modulus_pa, material.upper_poisson_ratio
        lower_young, lower_shear = material.lower_young_modulus_pa, material.lower_shear_modulus_pa
        lower_nu = lower_young/(2*lower_shear)-1
        layers = ((upper_young/(2*(1+upper_nu)),
            upper_young*upper_nu/((1+upper_nu)*(1-2*upper_nu)),
            body.elastic_layer_areal_depth_m, top, elastic_bottom),
            (lower_shear, 2*lower_shear*lower_nu/(1-2*lower_nu),
            body.areal_material_thickness_m-body.elastic_layer_areal_depth_m, elastic_bottom, bottom))
        quarter_angle = (16*_atan_small(D(1)/5)-4*_atan_small(D(1)/239))/2
        raw_green, projection, reference_metric = D(0), D(0), D(0)
        for shear, lame, depth, upper_radius, lower_radius in layers:
            energy_polynomial = _add(_scale(tensor_norm, shear),
                _scale(_multiply(trace, trace), lame/2))
            projection_polynomial = _add(
                _scale(_multiply(shape, _add(_scale(green_diagonal, F(2)),
                    _scale(shear_squared, F(1, 2)))), 2*shear),
                _scale(_multiply(slope, _multiply(dilation, shear_strain)), shear),
                _scale(_multiply(shape, trace), 2*lame))
            metric_polynomial = _add(_scale(_multiply(shape, shape), 4*(shear+lame)),
                _scale(_multiply(slope, slope), shear))
            raw_green += quarter_angle*radius**2*_decimal(depth)*_cosine_integral(energy_polynomial, body.beta_cut_radians)
            projection += quarter_angle*(upper_radius**2-lower_radius**2)/2*_cosine_integral(projection_polynomial, body.beta_cut_radians)
            reference_metric += quarter_angle*(upper_radius-lower_radius)*_cosine_integral(metric_polynomial, body.beta_cut_radians)
        initial_tip = _decimal(body.accepted_tip_displacement_m)
        old_stiffness = _decimal(body.source_old_tip_stiffness_n_m)
        calibrated = raw_green+initial_tip*projection+reference_metric*initial_tip**2/2
        calibrated += (old_stiffness-reference_metric)*(initial_tip+projection/reference_metric)**2/2
        return calibrated, raw_green, projection, reference_metric


def _contains(interval, expected):
    expected = F(expected) if isinstance(expected, D) else expected
    assert interval.lo <= expected <= interval.hi


@lru_cache(maxsize=2)
def _finite_case(parent):
    body, material = _body(parent), _material(parent)
    amplitude = F(17, 10)
    coordinates = (ZERO, amplitude, amplitude, amplitude, ZERO, ZERO, ZERO)
    response = _api().evaluate_native_cold_strain_energy(body, coordinates,
        material=material, precision=64, taylor_order=1, beta_cell_count=1, depth_cell_count=1)
    return body, material, amplitude, response


def test_all_source_cold_materials_keep_the_upper_tensor_and_declare_the_new_lower_projection():
    for parent in ("octant:"+str(index) for index in range(8)):
        body, material = _body(parent), _material(parent)
        assert material.upper_young_modulus_pa == body.young_modulus_pa
        assert material.upper_poisson_ratio == body.poisson_ratio
        assert material.lower_young_modulus_pa == F(10**9)
        assert material.lower_shear_modulus_pa == F(4*10**8)
        assert material.nonlocal_constitutive_calibration_declared
        assert not material.historical_full_three_dimensional_prestress_recovered
        assert not material.historical_local_three_dimensional_tensor_inherited
        assert not material.source_elastic_layer_thickness_used_as_areal_depth


@pytest.mark.parametrize("parent", ["octant:0", "octant:4"])
def test_actual_layered_quarter_energy_projection_and_raw_metric_match_independent_invariants(parent):
    body, material, amplitude, response = _finite_case(parent)
    energy, raw_green, projection, metric = _quarter_oracle(body, material, amplitude)
    _contains(response.energy_bounds_j, energy)
    _contains(response.raw_green_energy_bounds_j, raw_green)
    with localcontext() as context:
        context.prec = 100
        _contains(response.projection_increment_m, projection/metric)
    _contains(response.raw_integrals.reference_tip_strain_stiffness_n_m, metric)
    _contains(response.raw_integrals.projection_n, projection)
    assert response.reference_energy_j == body.source_old_tip_stiffness_n_m*body.accepted_tip_displacement_m**2/2
    assert response.reference_tip_force_n == body.source_old_tip_stiffness_n_m*body.accepted_tip_displacement_m
    assert response.reference_energy_j > 0
    assert response.reference_tip_force_n != 0


@pytest.mark.parametrize("parent", ["octant:0", "octant:4"])
def test_finite_calibrated_forces_and_hessian_match_independent_quartic_energy_variations(parent):
    body, material, amplitude, response = _finite_case(parent)
    with localcontext() as context:
        context.prec = 100
        step = F(7, 11)
        energies = {index: _quarter_oracle(body, material, amplitude+index*step)[0]
            for index in (-2, -1, 0, 1, 2)}
        h = _decimal(step)
        derivative = (-energies[2]+8*energies[1]-8*energies[-1]+energies[-2])/(12*h)
        curvature = (-energies[2]+16*energies[1]-30*energies[0]
            +16*energies[-1]-energies[-2])/(12*h*h)
    _contains(sum(response.generalized_gradient_bounds_n[1:4], I(ZERO)), derivative)
    _contains(sum((response.generalized_hessian_bounds_n_m[i][j]
        for i in (1, 2, 3) for j in (1, 2, 3)), I(ZERO)), curvature)


def test_actual_prestress_calibration_is_not_an_extra_old_beam_energy_charge():
    body, material, amplitude, response = _finite_case("octant:0")
    calibrated, raw_green, projection, metric = _quarter_oracle(body, material, amplitude)
    with localcontext() as context:
        context.prec = 100
        old = _decimal(body.source_old_tip_stiffness_n_m)
        initial = _decimal(body.accepted_tip_displacement_m)
        projected = initial+projection/metric
        complement = raw_green-projection**2/(2*metric)
        expected = old*projected**2/2+complement
        assert abs(expected-calibrated) < D("1e-70")*max(D(1), abs(calibrated))
        assert complement >= 0
    _contains(response.energy_bounds_j, expected)


def test_forged_material_cannot_change_the_original_upper_tensor_or_initial_strain_model():
    material = _material("octant:0")
    forged = replace(material, upper_young_modulus_pa=2*material.upper_young_modulus_pa)
    with pytest.raises((ValueError, TypeError), match="altered|material|declaration|source"):
        _api().validate_native_cold_strain_material(forged, body=_body("octant:0"))


@lru_cache(maxsize=1)
def _zero_case():
    body, material = _body("octant:0"), _material("octant:0")
    response = _api().evaluate_native_cold_strain_energy(body, (ZERO,)*7,
        material=material, precision=64, taylor_order=1, beta_cell_count=1, depth_cell_count=1)
    return body, material, response


def _tip_geometric_prestress_oracle(body, material):
    """Integral H0:C:(Gtip.T Gtip) from orthonormal radial-tip invariants."""
    with localcontext() as context:
        context.prec = 110
        shape = (F(1), F(-3, 2), ZERO, F(1, 2))
        slope = (F(-3, 2)/body.beta_cut_radians, ZERO, F(3, 2)/body.beta_cut_radians)
        radial_polynomial = _multiply(shape, _add(
            _scale(_multiply(shape, shape), F(2)), _multiply(slope, slope)))
        radius = _decimal(body.radius_m)
        top = _decimal(body.radius_m+body.roof_radial_displacement_m)
        middle = top-_decimal(body.elastic_thickness_m)
        bottom = (top**3-3*radius**2*_decimal(body.areal_material_thickness_m))**(D(1)/D(3))
        upper_nu = material.upper_poisson_ratio
        upper_shear = material.upper_young_modulus_pa/(2*(1+upper_nu))
        upper_lame = material.upper_young_modulus_pa*upper_nu/((1+upper_nu)*(1-2*upper_nu))
        lower_shear = material.lower_shear_modulus_pa
        lower_nu = material.lower_young_modulus_pa/(2*lower_shear)-1
        lower_lame = 2*lower_shear*lower_nu/(1-2*lower_nu)
        quarter_angle = (16*_atan_small(D(1)/5)-4*_atan_small(D(1)/239))/2
        integral = _cosine_integral(radial_polynomial, body.beta_cut_radians)
        return quarter_angle*integral*(2*_decimal(upper_shear+upper_lame)*(top/middle).ln()
            +2*_decimal(lower_shear+lower_lame)*(middle/bottom).ln())


def test_zero_increment_preserves_literal_source_energy_force_and_geometric_prestress_hessian():
    body, material, response = _zero_case()
    source_energy = body.source_old_tip_stiffness_n_m*body.accepted_tip_displacement_m**2/2
    source_force = body.source_old_tip_stiffness_n_m*body.accepted_tip_displacement_m
    assert response.energy_bounds_j == I(source_energy)
    assert response.generalized_gradient_bounds_n[0] == I(source_force)
    assert response.raw_green_energy_bounds_j == I(ZERO)
    assert response.projection_increment_m == I(ZERO)
    _, _, _, raw_metric = _quarter_oracle(body, material, ZERO)
    geometric = _tip_geometric_prestress_oracle(body, material)
    with localcontext() as context:
        context.prec = 100
        correction = _decimal(source_force)*geometric/raw_metric
        expected_hessian = _decimal(body.source_old_tip_stiffness_n_m)+correction
        assert correction != 0
    _contains(response.raw_integrals.projection_hessian_n_m2[0][0], geometric)
    _contains(response.generalized_hessian_bounds_n_m[0][0], expected_hessian)
    assert response.accepted_global_beam_reference_energy_preserved
    assert response.accepted_global_beam_reference_tip_force_preserved
    assert not response.old_beam_stiffness_added_as_second_energy
    assert not response.historical_full_three_dimensional_prestress_recovered
    assert not response.historical_archived_strain_energy_density_recovered
    assert not response.complete_three_dimensional_strain_energy_certified
    assert not response.mixed_force_equilibrium_certified
    assert not response.material_admission_authorized
    assert not response.original_point_1_completed


def _point_isotropic_invariants(body, material, amplitude):
    """Actual upper-layer point densities integrated over the quarter angle.

    In a radial orthonormal frame the isotropic supported field gives
    F=[[d,s,0],[0,d,0],[0,0,d]]. Its invariants are independent of theta.
    """
    with localcontext() as context:
        context.prec = 110
        radius0 = _decimal(body.radius_m)
        beta = _decimal(body.beta_cut_radians)/3
        depth = _decimal(body.elastic_layer_areal_depth_m)/2
        top = radius0+_decimal(body.roof_radial_displacement_m)
        radius = (top**3-3*radius0**2*depth)**(D(1)/D(3))
        scaled = D(1)/3
        shape = 1-D(3)*scaled/2+scaled**3/2
        slope = D(3)*(scaled**2-1)/(2*_decimal(body.beta_cut_radians))
        d = 1+_decimal(amplitude)*shape/radius0
        s = _decimal(amplitude)*slope/radius0
        e = (d*d-1)/2
        trace = 3*e+s*s/2
        norm_squared = 3*e*e+e*s*s+s**4/4+d*d*s*s/2
        young, nu = material.upper_young_modulus_pa, material.upper_poisson_ratio
        shear = _decimal(young/(2*(1+nu)))
        lame = _decimal(young*nu/((1+nu)*(1-2*nu)))
        pi = 16*_atan_small(D(1)/5)-4*_atan_small(D(1)/239)
        weight = pi/2*radius0**2*_sin_cos(beta)[1]
        metric = (4*(shear+lame)*shape*shape+shear*slope*slope)/radius**2
        projection = (2*shear*shape*(2*e+s*s/2)
            +shear*slope*d*s+2*lame*shape*trace)/radius
        green_energy = shear*norm_squared+lame*trace*trace/2
        return weight*metric, weight*projection, weight*green_energy


def test_actual_quarter_point_tensor_signs_and_variations_have_tight_independent_oracles():
    from tectonics.geometric_native_cold_tensor_integration import cold_tensor_angular_kernel
    from tectonics.geometric_native_curved_certified_integration import TaylorAlgebra
    body, material = _body("octant:0"), _material("octant:0")
    amplitude = F(17, 10)
    coordinates = (ZERO, amplitude, amplitude, amplitude, ZERO, ZERO, ZERO)
    scalar = TaylorAlgebra(order=0, precision=64)
    young, nu = material.upper_young_modulus_pa, material.upper_poisson_ratio
    shear = young/(2*(1+nu))
    lame = young*nu/((1+nu)*(1-2*nu))
    values = tuple(value.constant_bounds for value in cold_tensor_angular_kernel(body,
        coordinates, scalar.constant(body.beta_cut_radians/3),
        scalar.constant(body.elastic_layer_areal_depth_m/2), scalar,
        lame_pa=lame, shear_pa=shear))
    assert len(values) == 73
    metric, projection, energy = _point_isotropic_invariants(body, material, amplitude)
    checks = [(values[0], metric), (values[1], projection), (values[37]/2, energy)]
    with localcontext() as context:
        context.prec = 100
        step = F(7, 11)
        energies = {index: _point_isotropic_invariants(body, material, amplitude+index*step)[2]
            for index in (-2, -1, 0, 1, 2)}
        h = _decimal(step)
        derivative = (-energies[2]+8*energies[1]-8*energies[-1]+energies[-2])/(12*h)
        curvature = (-energies[2]+16*energies[1]-30*energies[0]
            +16*energies[-1]-energies[-2])/(12*h*h)
    # Kernel gradients are derivatives of U_G; its stored norm is 2 U_G.
    checks.append((sum((values[38+i] for i in (1, 2, 3)), I(ZERO)), derivative))
    pairs = [(i, j) for i in range(7) for j in range(i, 7)]
    isotropic_curvature = sum((values[45+pairs.index((min(i, j), max(i, j)))]
        for i in (1, 2, 3) for j in (1, 2, 3)), I(ZERO))
    checks.append((isotropic_curvature, curvature))
    for interval, expected in checks:
        _contains(interval, expected)
        # This point regression has no beta/depth tail. Whole-mask interval
        # widths are intentionally not presented as a force certificate.
        assert interval.hi-interval.lo < F(1, 10**30)*max(F(1), abs(F(expected)))
