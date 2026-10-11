"""Independent, small line goldens with no Genesis restore or producer import.

The source chart is declared explicitly.  High-precision expectations use
its literal binary64 inputs and Cartesian derivatives, without generating
an audit record through the polynomial producer under review.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math

import mpmath as mp
import pytest

from analysis.genesis_volume_exact_lambda_validation import independent_line_audit as audit


X = [[0., 0., 2.], [1., 0., 2.], [0., 1., 2.]]
N = (.5, .25, .25)


def _identity(n):
    return [[float(i == j) for j in range(n)] for i in range(n)]


def _source(center=(.125, .125, 1.), radius=.5, normal=(0., 0., 1.), offset=1.):
    return dict(geometry_source=dict(triangle_m=deepcopy(X),
        displacement_A_m=[[0.]*3 for _ in range(3)],
        slope_P_m=[[0.]*3 for _ in range(3)], lambda_bounds=[.2, .9],
        normal_xyz=list(normal), offset_m=offset, mass_kg=1., volume_m3=1.),
        seed_center_world_m=list(center), seed_radius_m=radius,
        C=_identity(6), Q=_identity(3),
        e0=[[.01, -.02, .03, -.04, .05, -.06],
            [.02, -.03, .04, -.05, .06, -.07],
            [.03, -.04, .05, -.06, .07, -.08]],
        e1=[[.001, -.002, .003, -.004, .005, -.006],
            [.002, -.003, .004, -.005, .006, -.007],
            [.003, -.004, .005, -.006, .007, -.008]])


def _close(actual, expected, relative="1e-65"):
    actual, expected = mp.mpf(actual), mp.mpf(expected)
    if expected == 0:
        assert actual == 0
    else:
        assert abs(actual-expected) <= mp.mpf(relative)*abs(expected)


def _intervals(result):
    return [(mp.mpf(row["lambda_low"]), mp.mpf(row["lambda_high"]), row["side"])
            for row in result["intervals"]]


def _matrix_close(actual, expected, relative="1e-65"):
    assert (actual.rows, actual.cols) == (expected.rows, expected.cols)
    for i in range(actual.rows):
        for j in range(actual.cols):
            _close(actual[i, j], expected[i, j], relative)


def _global_matrix(value):
    return mp.matrix([[mp.mpf(value[i, j]) for j in range(value.cols)] for i in range(value.rows)])


def _sym(vector, gradient):
    x, y, z = (mp.mpf(v) for v in vector)
    gx, gy, gz = (mp.mpf(v) for v in gradient)
    return mp.matrix((x*gx, y*gy, z*gz, x*gy+y*gx, x*gz+z*gx, y*gz+z*gy))


def _material_gradients(N, lam):
    # x=λN1, y=λN2, z=2λ; differentiate these Cartesian identities.
    return (mp.matrix((-1/lam, -1/lam, (N[1]+N[2])/(2*lam))),
            mp.matrix((1/lam, 0, -N[1]/(2*lam))),
            mp.matrix((0, 1/lam, -N[2]/(2*lam))))


def test_fixed_ray_sphere_and_gamma_roots_have_independent_analytic_goldens():
    with mp.workdps(100):
        result = audit.independent_line_geometry(_source(), N, precision=80)
        radius_along_lambda = mp.mpf(.5)/mp.sqrt(mp.mpf(4)+mp.mpf(1)/8)
        expected = ((mp.mpf(.5)-radius_along_lambda, mp.mpf(.5), -.5),
                    (mp.mpf(.5), mp.mpf(.5)+radius_along_lambda, .5))
        actual = _intervals(result)
        assert len(actual) == 2
        for interval, golden in zip(actual, expected):
            _close(interval[0], golden[0])
            _close(interval[1], golden[1])
            assert interval[2] == golden[2]
        assert mp.mpf(result["sphere_margin_times_direction_squared"]) > 0


def test_cohort_radial_bound_clips_entry_before_gamma_split():
    source = _source(center=(0., 0., .6), offset=.6)
    with mp.workdps(100):
        actual = _intervals(audit.independent_line_geometry(source, (1., 0., 0.), precision=80))
        center_lambda = mp.mpf(.6)/2
        expected = ((mp.mpf(.2), center_lambda, -.5),
                    (center_lambda, center_lambda+mp.mpf(.25), .5))
        assert len(actual) == 2
        for interval, golden in zip(actual, expected):
            _close(interval[0], golden[0])
            _close(interval[1], golden[1])
            assert interval[2] == golden[2]


def test_plane_miss_has_one_literal_side_without_manufactured_split():
    result = audit.independent_line_geometry(_source(offset=-1.), N)
    assert len(result["intervals"]) == 1
    assert result["intervals"][0]["side"] == .5


def test_exact_tangency_and_one_ulp_miss_are_empty_but_one_ulp_penetration_is_retained():
    source = _source(center=(1., 0., 1.), radius=1.)
    direction = (1., 0., 0.)
    tangent = audit.independent_line_geometry(source, direction)
    assert tangent["intervals"] == []
    assert mp.mpf(tangent["sphere_margin_times_direction_squared"]) == 0
    missed = deepcopy(source)
    missed["seed_radius_m"] = math.nextafter(1., 0.)
    assert audit.independent_line_geometry(missed, direction)["intervals"] == []
    entered = deepcopy(source)
    entered["seed_radius_m"] = math.nextafter(1., math.inf)
    with mp.workdps(100):
        result = audit.independent_line_geometry(entered, direction, precision=80)
        intervals = _intervals(result)
        assert len(intervals) == 2
        assert all(high > low for low, high, _ in intervals)
        total_width = sum(high-low for low, high, _ in intervals)
        _close(total_width, mp.sqrt(mp.mpf(entered["seed_radius_m"])**2-1))
        assert mp.mpf(result["sphere_margin_times_direction_squared"]) > 0


def test_positive_support_that_exceeds_precision_resolution_refuses_instead_of_becoming_empty():
    source = _source(center=(0., 0., 1.), radius=1e-100)
    with pytest.raises(audit.IndependentLineRefusal):
        audit.independent_line_geometry(source, (1., 0., 0.), precision=80)
    with mp.workdps(140):
        resolved = audit.independent_line_geometry(source, (1., 0., 0.), precision=120)
        assert len(resolved["intervals"]) == 2
        assert all(high > low for low, high, _ in _intervals(resolved))
        assert mp.mpf(resolved["sphere_margin_times_direction_squared"]) > 0


def test_coplanar_gamma_with_positive_line_support_is_refused():
    source = _source(center=(0., 0., 1.), normal=(1., 0., 0.), offset=0.)
    with pytest.raises(audit.IndependentLineRefusal):
        audit.independent_line_geometry(source, (1., 0., 0.))


def test_source_C_and_D_are_literal_float64_sums_before_high_precision_geometry():
    source = _source(center=(1.5, 0., 1.))
    section = source["geometry_source"]
    section["displacement_A_m"][1][0] = 1.
    section["slope_P_m"][1][0] = 2.**-54
    baseline = deepcopy(source)
    baseline["geometry_source"]["slope_P_m"][1][0] = 0.
    # Both archived binary64 sums are exactly 1.  Recomputing their source
    # addition/subtraction in MP would invent a different material chart.
    assert float(1.+2.**-54) == 1.
    assert float(1.-2.**-54) == 1.
    with mp.workdps(100):
        actual = audit.independent_line_geometry(source, (0., 1., 0.), precision=80)
        expected = audit.independent_line_geometry(baseline, (0., 1., 0.), precision=80)
        assert _intervals(actual) == _intervals(expected)
        assert mp.mpf(1.)+mp.mpf(2.**-54) != mp.mpf(1.)
        assert mp.mpf(1.)-mp.mpf(2.**-54) != mp.mpf(1.)


def test_geometry_does_not_mutate_declared_source_or_global_mp_context():
    source = _source()
    original = deepcopy(source)
    with mp.workdps(17):
        before = mp.mp.prec
        audit.independent_line_geometry(source, N, precision=80)
        assert mp.mp.prec == before
    assert source == original


def test_archived_native_N_exact_ratios_preserve_the_same_material_line():
    exact = tuple(f"{value.as_integer_ratio()[0]}/{value.as_integer_ratio()[1]}" for value in N)
    fractions = tuple(Fraction.from_float(value) for value in N)
    original = audit.independent_line_geometry(_source(), N)
    assert audit.independent_line_geometry(_source(), exact)["intervals"] == original["intervals"]
    assert audit.independent_line_geometry(_source(), fractions)["intervals"] == original["intervals"]


def test_accepted_native_N_roundoff_is_preserved_without_normalizing_the_line():
    nodes = (math.nextafter(.5, math.inf), .25, .25)
    ratios = tuple(Fraction.from_float(value) for value in nodes)
    assert sum(ratios) != 1
    with mp.workdps(100):
        direction = mp.matrix((mp.mpf(nodes[1]), mp.mpf(nodes[2]),
            2*sum(mp.mpf(v) for v in nodes)))
        center = mp.matrix((mp.mpf(.125), mp.mpf(.125), 1))
        norm2 = sum(v*v for v in direction)
        projection = sum(a*b for a, b in zip(direction, center))
        sphere_margin = mp.mpf(.25)*norm2-sum(v*v for v in center)*norm2+projection**2
        midpoint, halfwidth = projection/norm2, mp.sqrt(sphere_margin)/norm2
        root = 1/direction[2]
        expected = ((midpoint-halfwidth, root, -.5), (root, midpoint+halfwidth, .5))
        result = audit.independent_line_geometry(_source(), nodes, precision=80)
        for interval, golden in zip(_intervals(result), expected):
            _close(interval[0], golden[0])
            _close(interval[1], golden[1])
            assert interval[2] == golden[2]
        literal = tuple(f"{v.numerator}/{v.denominator}" for v in ratios)
        assert audit.independent_line_geometry(_source(), literal, precision=80)["intervals"] == result["intervals"]
        normalized = tuple(v/sum(ratios) for v in ratios)
        assert audit.independent_line_geometry(_source(), normalized, precision=80)["intervals"] != result["intervals"]


def test_full_original_A_P_strain_matches_cartesian_derivatives_for_simple_chart():
    with mp.workdps(100):
        lam = mp.mpf(.5)
        nodes = tuple(mp.mpf(value) for value in N)
        fields = audit.evaluate_source_fields(_source(), N, .5, .5, precision=80)
        gradients = _material_gradients(nodes, lam)
        grad_lambda = mp.matrix((0, 0, mp.mpf(.5)))
        expected, trace = mp.zeros(6, 18), mp.zeros(3, 18)
        for vertex in range(3):
            slope_gradient = (lam-1)*gradients[vertex]+nodes[vertex]*grad_lambda
            for axis in range(3):
                direction = mp.eye(3)[:, axis]
                expected[:, 6*vertex+axis] = _sym(direction, gradients[vertex])
                expected[:, 6*vertex+3+axis] = _sym(direction, slope_gradient)
                trace[axis, 6*vertex+axis] = nodes[vertex]
                trace[axis, 6*vertex+3+axis] = (lam-1)*nodes[vertex]
        _matrix_close(fields["Bhat"], lam*expected)
        _matrix_close(fields["direct_derivative_B"], expected)
        _matrix_close(fields["literal_vs_direct_derivative_difference"], mp.zeros(6, 18))
        _matrix_close(fields["T"], trace)


def test_full_A_P_strain_uses_retained_proper_source_basis():
    source = _source()
    # Proper 90-degree source-frame rotation has exact binary64 entries.
    source["Q"] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    with mp.workdps(100):
        lam = mp.mpf(.5)
        nodes = tuple(mp.mpf(value) for value in N)
        Q = mp.matrix(source["Q"])
        gradients = _material_gradients(nodes, lam)
        grad_lambda = mp.matrix((0, 0, mp.mpf(.5)))
        expected = mp.zeros(6, 18)
        for vertex in range(3):
            slope_gradient = (lam-1)*gradients[vertex]+nodes[vertex]*grad_lambda
            for axis in range(3):
                direction = Q.T[:, axis]
                expected[:, 6*vertex+axis] = _sym(direction, Q.T*gradients[vertex])
                expected[:, 6*vertex+3+axis] = _sym(direction, Q.T*slope_gradient)
        fields = audit.evaluate_source_fields(source, N, .5, .5)
        _matrix_close(fields["Bhat"], lam*expected)
        _matrix_close(fields["direct_derivative_B"], expected)
        _matrix_close(fields["literal_vs_direct_derivative_difference"], mp.zeros(6, 18))


def test_bubble_and_its_cartesian_gradient_match_independent_center_golden():
    with mp.workdps(100):
        source = _source()
        fields = audit.evaluate_source_fields(source, N, .5, .5, precision=80)
        nodes, lam = tuple(mp.mpf(v) for v in N), mp.mpf(.5)
        b, a = (mp.mpf(v) for v in source["geometry_source"]["lambda_bounds"])
        angular = 27*nodes[0]*nodes[1]*nodes[2]
        radial = 4*(lam-b)*(a-lam)/(a-b)**2
        gradients = _material_gradients(nodes, lam)
        angular_gradient = 27*(nodes[1]*nodes[2]*gradients[0]
            +nodes[0]*nodes[2]*gradients[1]+nodes[0]*nodes[1]*gradients[2])
        radial_derivative = 4*(a+b-2*lam)/(a-b)**2
        # At the actual ball center, the sphere factor is one and its
        # Cartesian derivative is zero.  Angular/radial gradients remain.
        gradient = radial*angular_gradient+angular*radial_derivative*mp.matrix((0, 0, mp.mpf(.5)))
        _close(fields["w"], angular*radial)
        _matrix_close(fields["L"], lam*gradient)
        original_term = angular*radial*_global_matrix(fields["Bhat"])
        expected = original_term.copy()
        own = mp.matrix([[abs(original_term[i, j]) for j in range(18)] for i in range(6)])
        for column in range(18):
            gradient_term = lam*_sym(fields["T"][:, column], gradient)
            expected[:, column] += gradient_term
            for row in range(6):
                own[row, column] += abs(gradient_term[row])
        for row in range(6):
            for column in range(18):
                # Cancellation is resolved against this entry's two actual
                # operands, without borrowing another component's scale.
                assert abs(mp.mpf(fields["Ghat_without_H"][row, column])-expected[row, column]) <= mp.mpf("1e-65")*own[row, column]
        _matrix_close(fields["E_without_H"], fields["w"]*fields["T"])


def test_off_center_bubble_gradient_uses_full_A_P_birth_map_and_retained_reference_chart():
    source = _source()
    A, P = (.03125, -.015625, .0625), (.125, 0., -.0625)
    source["geometry_source"]["displacement_A_m"] = [list(A) for _ in range(3)]
    source["geometry_source"]["slope_P_m"] = [list(P) for _ in range(3)]
    original = deepcopy(source)
    with mp.workdps(100):
        lam = mp.mpf(.5)
        nodes = tuple(mp.mpf(value) for value in N)
        b, a = (mp.mpf(value) for value in source["geometry_source"]["lambda_bounds"])
        material = mp.matrix((lam*nodes[1], lam*nodes[2], 2*lam))
        birth = material+mp.matrix(A)+(lam-1)*mp.matrix(P)
        offset = birth-mp.matrix(source["seed_center_world_m"])
        radius2 = mp.mpf(source["seed_radius_m"])**2
        sphere_base = 1-sum(value*value for value in offset)/radius2
        assert 0 < sphere_base < 1
        sphere = sphere_base**2
        grad_lambda = mp.matrix((0, 0, mp.mpf(.5)))
        # Uniform A/P gives ∂birth/∂reference = I + P ⊗ ∇λ.
        J_birth = mp.eye(3)+mp.matrix(P)*grad_lambda.T
        sphere_gradient = J_birth.T*(-4*sphere_base*offset/radius2)
        assert any(value != 0 for value in sphere_gradient)
        angular = 27*nodes[0]*nodes[1]*nodes[2]
        gradients = _material_gradients(nodes, lam)
        angular_gradient = 27*(nodes[1]*nodes[2]*gradients[0]
            +nodes[0]*nodes[2]*gradients[1]+nodes[0]*nodes[1]*gradients[2])
        radial = 4*(lam-b)*(a-lam)/(a-b)**2
        radial_gradient = 4*(a+b-2*lam)/(a-b)**2*grad_lambda
        expected_w = angular*radial*sphere
        gradient = radial*sphere*angular_gradient+angular*sphere*radial_gradient+angular*radial*sphere_gradient
        fields = audit.evaluate_source_fields(source, N, .5, .5, precision=80)
        _close(fields["w"], expected_w)
        _matrix_close(fields["L"], lam*gradient)
        first = expected_w*_global_matrix(fields["Bhat"])
        for column in range(18):
            second = _sym(fields["T"][:, column], lam*gradient)
            for row in range(6):
                own = abs(first[row, column])+abs(second[row])
                actual = mp.mpf(fields["Ghat_without_H"][row, column])
                assert abs(actual-first[row, column]-second[row]) <= mp.mpf("1e-65")*own
    assert source == original


def test_two_sides_share_original_memory_and_retain_full_nonzero_radial_memory():
    with mp.workdps(100):
        raw = _source()
        original = deepcopy(raw)
        source = audit.read_line_source(raw, precision=80)
        positive = audit.evaluate_source_fields(source, N, .5, .5)
        negative = audit.evaluate_source_fields(source, N, .5, -.5)
        angular = mp.matrix([list(N)])*_global_matrix(source.angular_sample_inverse)
        e0, e1 = mp.matrix(raw["e0"]), mp.matrix(raw["e1"])
        expected = (mp.mpf(.5)*angular*e0+angular*e1).T
        _matrix_close(positive["memoryhat"], expected)
        _matrix_close(negative["memoryhat"], expected)
        assert any(expected[i] != (mp.mpf(.5)*angular*e0).T[i] for i in range(6))
        _matrix_close(positive["E"]-negative["E"], positive["E_without_H"])
        _matrix_close(positive["G"]+negative["G"], mp.zeros(6, 18))
        assert raw == original


def _multiply(a, b):
    result = [Fraction(0)]*(len(a)+len(b)-1)
    for i, left in enumerate(a):
        for j, right in enumerate(b):
            result[i+j] += left*right
    return result


def _integral(coefficients, low, high):
    return sum((mp.mpf(coefficient.numerator)/coefficient.denominator)
        *(high**(index+1)-low**(index+1))/(index+1)
        for index, coefficient in enumerate(coefficients))


def _analytic_mass_goldens():
    # Closed polynomial antiderivatives of this specific declared bubble.
    # All coefficients are exact rationals in the source float64 bits.
    b, a = (Fraction.from_float(v) for v in (.2, .9))
    radial = [-a*b, a+b, Fraction(-1)]
    sphere = [Fraction(-25, 8), Fraction(33, 2), Fraction(-33, 2)]
    factor = Fraction(27, 8)/(a-b)**2
    bubble = [factor*value for value in _multiply(radial, _multiply(sphere, sphere))]
    midpoint, half = mp.mpf(.5), mp.sqrt(mp.mpf(2)/33)
    low, high = midpoint-half, midpoint+half
    weighted = [Fraction(0), Fraction(0), *bubble]
    cross = (_integral(weighted, midpoint, high)-_integral(weighted, low, midpoint))/4
    self_mass = _integral([Fraction(0), Fraction(0), *_multiply(bubble, bubble)], low, high)/8
    return cross, self_mass


@pytest.fixture(scope="module")
def direct_line_result():
    basis, probes = _integration_inputs()
    return audit.integrate_source_line(_source(), N, parent_basis=basis,
        added_basis=basis, probes=probes, precision=80, quadrature_orders=(12, 20))


def _integration_inputs():
    basis = [[float(i == 0)] for i in range(18)]
    weak_q, weak_a = [0.]*18, [0.]*18
    weak_q[0], weak_a[0] = 1e-100, -3e-120
    probes = (dict(name="weak", parent_increment_m=weak_q, enrichment_increment_m=weak_a),
              dict(name="zero", parent_increment_m=[0.]*18, enrichment_increment_m=[0.]*18))
    return basis, probes


def test_direct_lambda_quadrature_matches_exact_signed_mass_antiderivatives(direct_line_result):
    with mp.workdps(100):
        values = dict(zip(direct_line_result["component_names"], direct_line_result["flattened_values"]))
        cross, self_mass = _analytic_mass_goldens()
        assert cross > 0 and self_mass > 0
        _close(values["projected.mass_cross[0,0]"], cross)
        _close(values["projected.mass_enriched[0,0]"], self_mass)
        assert direct_line_result["parent_work_scope"] == "whole_original_cohort_parent_work_is_not_a_clipped_line_term"
        for key in ("angular_coverage_certified", "general_contraction_accuracy_certified",
                    "physical_contraction_certified", "physical_creation_admitted",
                    "physical_release_admitted", "available_for_creation_funding"):
            assert direct_line_result[key] is False


def test_weak_signed_integral_keeps_its_own_scale_without_a_positive_floor(direct_line_result):
    with mp.workdps(100):
        names = direct_line_result["component_names"]
        values = dict(zip(names, direct_line_result["flattened_values"]))
        scales = dict(zip(names, direct_line_result["absolute_operand_scales"]))
        budgets = dict(zip(names, direct_line_result["numerical_error_bounds"]))
        cross, self_mass = _analytic_mass_goldens()
        q, a = mp.mpf(1e-100), mp.mpf(-3e-120)
        cross_name = "probe.weak.mass_cross_bilinear_kg_m2"
        self_name = "probe.weak.mass_self_bilinear_kg_m2"
        _close(values[cross_name], q*a*cross)
        _close(values[self_name], a*a*self_mass)
        assert values[cross_name] < 0 and values[self_name] > 0
        _close(scales[cross_name], abs(q*a)*scales["projected.mass_cross[0,0]"])
        assert 0 < budgets[cross_name] < mp.mpf("1e-60")*abs(values[cross_name])
        for term in audit.TERMS:
            name = "probe.zero."+term
            assert values[name] == scales[name] == budgets[name] == 0


@pytest.mark.parametrize("Nbad", ((.5, .5), (.5, .5, .5), (-.1, .5, .6),
    (.5, math.nan, .5), (True, 0, 0), (object(), .25, .25)))
def test_nonfinite_or_exterior_material_barycentrics_are_refused(Nbad):
    with pytest.raises(audit.IndependentLineRefusal):
        audit.independent_line_geometry(_source(), Nbad)


@pytest.mark.parametrize("mutation", ("zero_radius", "boolean_radius", "nonfinite_center", "zero_normal", "singular_chart"))
def test_invalid_source_geometry_is_refused(mutation):
    source = _source()
    if mutation == "zero_radius":
        source["seed_radius_m"] = 0.
    elif mutation == "boolean_radius":
        source["seed_radius_m"] = True
    elif mutation == "nonfinite_center":
        source["seed_center_world_m"][0] = math.inf
    elif mutation == "zero_normal":
        source["geometry_source"]["normal_xyz"] = [0., 0., 0.]
    else:
        source["geometry_source"]["triangle_m"] = [[0., 0., 2.], [1., 0., 2.], [2., 0., 2.]]
    with pytest.raises(audit.IndependentLineRefusal):
        audit.independent_line_geometry(source, N)


@pytest.mark.parametrize("precision", (True, 0, 80.5))
def test_precision_must_be_an_explicit_supported_integer(precision):
    with pytest.raises(audit.IndependentLineRefusal):
        audit.independent_line_geometry(_source(), N, precision=precision)


@pytest.mark.parametrize("mutation", ("reflected_basis", "scaled_basis", "nonsymmetric_C",
    "negative_C", "nonfinite_memory", "missing_radial_memory", "wrong_memory_shape"))
def test_invalid_retained_constitutive_or_memory_source_is_refused(mutation):
    source = _source()
    if mutation == "reflected_basis":
        source["Q"][0][0] = -1.
    elif mutation == "scaled_basis":
        source["Q"][0][0] = 2.
    elif mutation == "nonsymmetric_C":
        source["C"][0][1] = .5
    elif mutation == "negative_C":
        source["C"][0][0] = -1.
    elif mutation == "nonfinite_memory":
        source["e1"][0][0] = math.nan
    elif mutation == "missing_radial_memory":
        del source["e1"]
    else:
        source["e0"] = source["e0"][:2]
    with pytest.raises(audit.IndependentLineRefusal):
        audit.read_line_source(source)


def test_explicit_long_source_aliases_retain_identical_fields():
    raw = _source()
    aliased = deepcopy(raw)
    aliased["section"] = aliased.pop("geometry_source")
    for short, long in (("C", "elasticity_pa"), ("Q", "basis_xyz"),
                        ("e0", "memory0"), ("e1", "memory1")):
        aliased[long] = aliased.pop(short)
    before = deepcopy(aliased)
    direct = audit.evaluate_source_fields(raw, N, .5, .5)
    renamed = audit.evaluate_source_fields(aliased, N, .5, .5)
    with mp.workdps(100):
        _close(direct["w"], renamed["w"])
        for key in ("L", "Bhat", "Ghat_without_H", "T", "E_without_H", "memoryhat"):
            _matrix_close(direct[key], renamed[key])
    assert aliased == before


@pytest.mark.parametrize("lam,side", ((0., .5), (.1, .5), (.95, .5),
    (.5, 0.), (.5, 1.), (.5, True), (math.nan, .5)))
def test_field_lambda_and_side_must_retain_literal_support(lam, side):
    with pytest.raises(audit.IndependentLineRefusal):
        audit.evaluate_source_fields(_source(), N, lam, side)


def test_field_outside_sphere_is_refused_even_inside_original_radial_cohort():
    with pytest.raises(audit.IndependentLineRefusal):
        audit.evaluate_source_fields(_source(), N, .2, -.5)


@pytest.mark.parametrize("orders", ((12, 12), (20, 12), (7, 20), (12, 49), (True, 20), (12., 20)))
def test_direct_integral_requires_bounded_increasing_integer_orders(orders):
    with pytest.raises(audit.IndependentLineRefusal):
        audit.integrate_source_line(_source(), N, quadrature_orders=orders)


@pytest.mark.parametrize("mutation", ("missing_mass", "missing_volume", "zero_mass", "boolean_volume"))
def test_direct_integral_requires_explicit_positive_retained_density(mutation):
    source = _source()
    section = source["geometry_source"]
    if mutation == "missing_mass":
        del section["mass_kg"]
    elif mutation == "missing_volume":
        del section["volume_m3"]
    elif mutation == "zero_mass":
        section["mass_kg"] = 0.
    else:
        section["volume_m3"] = True
    with pytest.raises(audit.IndependentLineRefusal):
        audit.integrate_source_line(source, N)


def test_empty_material_line_keeps_schema_and_exact_zero_operands():
    source = _source(center=(10., 10., 10.))
    basis = [[float(i == 0)] for i in range(18)]
    probe = dict(name="absent", parent_increment_m=[0.]*18,
                 enrichment_increment_m=[0.]*18)
    result = audit.integrate_source_line(source, N, parent_basis=basis,
        added_basis=basis, probes=(probe,))
    assert result["geometry"]["intervals"] == []
    names = result["component_names"]
    assert len(names) == len(set(names))
    assert "full.stiffness_cross[17,17]" in names
    assert "projected.mass_enriched[0,0]" in names
    assert "probe.absent.mass_self_bilinear_kg_m2" in names
    for key in ("flattened_values", "absolute_operand_scales", "numerical_error_bounds"):
        assert len(result[key]) == len(names)
        assert all(value == 0 for value in result[key])


@pytest.mark.parametrize("empty", (False, True))
@pytest.mark.parametrize("mutation", ("missing_coordinate", "wrong_coordinate_shape", "wrong_memory_shape"))
def test_probe_source_validation_is_not_bypassed_by_empty_support(empty, mutation):
    source = _source(center=(10., 10., 10.)) if empty else _source()
    probe = dict(name="invalid", parent_increment_m=[0.]*18,
                 enrichment_increment_m=[0.]*18)
    if mutation == "missing_coordinate":
        del probe["enrichment_increment_m"]
    elif mutation == "wrong_coordinate_shape":
        probe["parent_increment_m"] = [0.]*17
    else:
        probe["mean_e1"] = [[0.]*6]*2
    with pytest.raises(audit.IndependentLineRefusal):
        audit.integrate_source_line(source, N, probes=(probe,))


def test_compact_decimal_archive_has_reproducible_self_seal_and_limited_scope(direct_line_result):
    archived = audit.archive_line_result(direct_line_result)
    payload = deepcopy(archived)
    digest = payload.pop("sha256")
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode()
    assert digest == hashlib.sha256(raw).hexdigest()
    assert all(isinstance(value, str) for value in archived["flattened_values"])
    assert archived["scope"] == "independent_fixed_N_high_precision_checks_no_angular_or_physical_certificate"
    assert archived["angular_coverage_certified"] is False
    assert archived["physical_creation_admitted"] is False


def _native_record(values, shape):
    pairs = []
    for value in values:
        if hasattr(value, "_mpf_"):
            sign, mantissa, exponent, _ = value._mpf_
            fraction = Fraction((-1 if sign else 1)*mantissa)*Fraction(2)**exponent
        else:
            fraction = Fraction(value)
        pairs.append([str(fraction.numerator), str(fraction.denominator)])
    return dict(shape=list(shape), numerator_denominator=pairs)


def _line_witness(oracle, native=True):
    # These values come from the independent direct route, never from the
    # polynomial producer under review.  The mass entries also have the
    # separate exact antiderivative assertions above.
    with mp.workdps(100):
        geometry = oracle["geometry"]
        witness = dict(component_names=list(oracle["component_names"]),
            values=[mp.nstr(v, 90) for v in oracle["flattened_values"]],
            roundoff_component_bounds=[0.]*len(oracle["component_names"]),
            interval={key: geometry[key] for key in
                ("lambda_mid", "lambda_halfwidth", "gamma_root_xi")})
        if native:
            witness["native_exact_binary_ratio"] = {
                "values": _native_record(oracle["flattened_values"], (len(oracle["component_names"]),))}
    return witness


def _audit_witness(oracle, witness, **changes):
    basis, probes = _integration_inputs()
    operands = dict(raw=_source(), N=N, added_basis=basis,
        parent_basis=basis, probes=probes, precision=80)
    operands.update(changes)
    return audit.audit_line_witness(witness=witness, oracle=oracle, **operands)


def test_independent_native_line_witness_preserves_small_signed_components(direct_line_result):
    witness = _line_witness(direct_line_result)
    report = _audit_witness(direct_line_result, witness)
    assert report["source_direct_quadrature_status"] == "PASS"
    assert report["native_logical_values_used"] is True
    check = report["all_component_checks"]["probe.weak.mass_cross_bilinear_kg_m2"]
    assert check["passed"] is True
    assert check["positive_scale_floor_used"] is False
    assert check["binary64_terminal_conversion_used"] is False
    assert mp.mpf(check["actual"]) < 0
    for key in ("angular_accuracy_certified", "general_contraction_accuracy_certified",
                "physical_contraction_certified", "physical_creation_admitted",
                "physical_release_admitted", "available_for_creation_funding"):
        assert report[key] is False


@pytest.mark.parametrize("mutation", ("source", "N", "parent_basis", "added_basis", "probe", "precision"))
def test_cached_direct_oracle_cannot_be_reused_for_changed_operands(direct_line_result, mutation):
    changes = {}
    basis, probes = _integration_inputs()
    if mutation == "source":
        changes["raw"] = _source()
        changes["raw"]["e1"][0][0] = math.nextafter(.001, math.inf)
    elif mutation == "N":
        changes["N"] = (math.nextafter(.5, math.inf), .25, .25)
    elif mutation in ("parent_basis", "added_basis"):
        basis[0][0] = math.nextafter(1., math.inf)
        changes[mutation] = basis
    elif mutation == "probe":
        probes[0]["parent_increment_m"][0] = math.nextafter(1e-100, math.inf)
        changes["probes"] = probes
    else:
        changes["precision"] = 81
    with pytest.raises(audit.IndependentLineRefusal, match="binding differs"):
        _audit_witness(direct_line_result, _line_witness(direct_line_result), **changes)


def test_native_weak_component_erasure_is_reported_without_laundering_by_other_scales(direct_line_result):
    witness = _line_witness(direct_line_result)
    name = "probe.weak.mass_cross_bilinear_kg_m2"
    position = witness["component_names"].index(name)
    witness["native_exact_binary_ratio"]["values"]["numerator_denominator"][position] = ["0", "1"]
    report = _audit_witness(direct_line_result, witness, raise_on_mismatch=False)
    assert report["source_direct_quadrature_status"] == "OPEN"
    assert report["mismatching_components"] == [name]
    check = report["all_component_checks"][name]
    with mp.workdps(100):
        assert mp.mpf(check["absolute_difference"]) > mp.mpf(check["arithmetic_bound"])
        assert mp.mpf(check["own_operand_scale"]) < mp.mpf("1e-210")
    assert check["positive_scale_floor_used"] is False
    with pytest.raises(audit.IndependentLineRefusal):
        _audit_witness(direct_line_result, witness)


def test_saved_budget_cannot_create_scale_for_exact_zero_component(direct_line_result):
    witness = _line_witness(direct_line_result)
    name = "probe.zero.mass_self_bilinear_kg_m2"
    position = witness["component_names"].index(name)
    witness["roundoff_component_bounds"][position] = 1e-100
    report = _audit_witness(direct_line_result, witness, raise_on_mismatch=False)
    assert report["source_direct_quadrature_status"] == "OPEN"
    assert report["mismatching_components"] == [name]
    assert report["all_component_checks"][name]["saved_arithmetic_budget_within_independent_operands"] is False


def test_negative_saved_line_budget_is_refused_before_oracle_error_is_added(direct_line_result):
    witness = _line_witness(direct_line_result)
    position = witness["component_names"].index("full.stiffness_enriched[0,0]")
    witness["roundoff_component_bounds"][position] = -1e-100
    with pytest.raises(audit.IndependentLineRefusal):
        _audit_witness(direct_line_result, witness)


@pytest.mark.parametrize("mutation", ("endpoint", "gamma_split"))
def test_corrupted_line_geometry_is_open_even_when_component_values_match(direct_line_result, mutation):
    witness = _line_witness(direct_line_result)
    key = "lambda_mid" if mutation == "endpoint" else "gamma_root_xi"
    witness["interval"][key] = .125
    report = _audit_witness(direct_line_result, witness, raise_on_mismatch=False)
    assert report["mismatching_components"] == []
    assert report["geometry_status"] == "OPEN"
    assert report["source_direct_quadrature_status"] == "OPEN"


def test_narrow_support_geometry_cannot_borrow_its_midpoint_as_a_width_scale():
    source = _source(center=(0., 0., 1.), radius=1e-20)
    true = audit.independent_line_geometry(source, (1., 0., 0.), precision=80)
    witness = dict(interval={key: true[key] for key in
        ("lambda_mid", "lambda_halfwidth", "gamma_root_xi")})
    valid = audit.audit_geometry_witness(source, (1., 0., 0.), witness, precision=80)
    assert valid["geometry_status"] == "PASS"
    witness["interval"]["lambda_halfwidth"] = 1e-18
    valid = audit.audit_geometry_witness(source, (1., 0., 0.), witness, precision=80)
    assert valid["geometry_status"] == "OPEN"
    assert valid["geometry_checks"]["halfwidth_matches_own_scale"] is False


@pytest.mark.parametrize("mutation", ("missing_component", "duplicate_component", "invalid_native_denominator"))
def test_malformed_saved_line_schema_or_native_ratio_is_refused(direct_line_result, mutation):
    witness = _line_witness(direct_line_result)
    if mutation == "missing_component":
        witness["component_names"].pop()
    elif mutation == "duplicate_component":
        witness["component_names"][-1] = witness["component_names"][0]
    else:
        witness["native_exact_binary_ratio"]["values"]["numerator_denominator"][0][1] = "0"
    with pytest.raises(audit.IndependentLineRefusal):
        _audit_witness(direct_line_result, witness)


def _zero_coefficient_witness(bound=0.):
    # All-zero fields are an explicitly wrong representation of the positive
    # interior bubble, nonzero A/P derivatives and inherited source memory.
    shapes = dict(lam=(), w=(), L=(3,), Bhat=(6, 18), Ghat_without_H=(6, 18),
                  T=(3, 18), E_without_H=(3, 18), memoryhat=(6,))
    def fill(shape, value):
        return value if not shape else [fill(shape[1:], value) for _ in range(shape[0])]
    geometry = audit.independent_line_geometry(_source(), N)
    return dict(interval={key: geometry[key] for key in
        ("lambda_mid", "lambda_halfwidth", "gamma_root_xi")},
        coefficients={key: [fill(shape, 0.)] for key, shape in shapes.items()},
        coefficient_roundoff_bounds={key: [fill(shape, bound)] for key, shape in shapes.items()})


def test_wrong_source_coefficients_are_open_at_both_independent_field_probes():
    report = audit.audit_coefficient_fields(_source(), N, _zero_coefficient_witness(),
                                            raise_on_mismatch=False)
    assert report["coefficient_source_status"] == "OPEN"
    assert any(name.startswith("xi=0.0.w") for name in report["mismatching_components"])
    assert any(name.startswith("xi=0.42857143.w") for name in report["mismatching_components"])
    assert report["physical_creation_admitted"] is False
    with pytest.raises(audit.IndependentLineRefusal):
        audit.audit_coefficient_fields(_source(), N, _zero_coefficient_witness())


def test_negative_saved_coefficient_bounds_are_refused():
    with pytest.raises(audit.IndependentLineRefusal):
        audit.audit_coefficient_fields(_source(), N, _zero_coefficient_witness(-1e9))


def test_inflated_coefficient_budget_cannot_hide_wrong_source_fields():
    report = audit.audit_coefficient_fields(_source(), N, _zero_coefficient_witness(1e9),
                                            raise_on_mismatch=False)
    assert report["coefficient_source_status"] == "OPEN"


def _poly_product(left, right):
    zero = left[0]*right[0]*0
    values = [zero.copy() if hasattr(zero, "rows") else zero for _ in range(len(left)+len(right)-1)]
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            values[i+j] += a*b
    return values


def _poly_sum(*polynomials):
    sample = polynomials[0][0]*0
    values = [sample.copy() if hasattr(sample, "rows") else sample
              for _ in range(max(len(value) for value in polynomials))]
    for polynomial in polynomials:
        for i, value in enumerate(polynomial):
            values[i] += value
    return values


def _center_polynomial(polynomial, mid, half):
    result = []
    for power in range(len(polynomial)):
        zero = polynomial[0]*0
        value = zero.copy() if hasattr(zero, "rows") else zero
        for old_power in range(power, len(polynomial)):
            value += polynomial[old_power]*math.comb(old_power, power)*mid**(old_power-power)*half**power
        result.append(value)
    return result


def _decimal_polynomial(polynomial, scale=1):
    output = []
    for coefficient in polynomial:
        if hasattr(coefficient, "rows"):
            output.append([[mp.nstr(scale*coefficient[i, j], 95) for j in range(coefficient.cols)]
                           for i in range(coefficient.rows)])
        else:
            output.append(mp.nstr(scale*coefficient, 95))
    return output


@pytest.fixture(scope="module")
def analytic_coefficient_case():
    """Degree-first fields derived from explicit Cartesian scalar products."""
    raw = _source()
    raw["geometry_source"]["lambda_bounds"] = [.25, .75]
    raw["e0"] = raw["e1"] = [[0.]*6 for _ in range(3)]
    return raw, _analytic_coefficient_witness(raw)


def _analytic_coefficient_witness(raw):
    with mp.workdps(100):
        nodes = tuple(mp.mpf(value) for value in N)
        b, a = (mp.mpf(value) for value in raw["geometry_source"]["lambda_bounds"])
        mid, half = mp.mpf(.5), mp.sqrt(mp.mpf(2)/33)
        angular = 27*nodes[0]*nodes[1]*nodes[2]
        sphere_base = [mp.mpf(-25)/8, mp.mpf(33)/2, -mp.mpf(33)/2]
        sphere = _poly_product(sphere_base, sphere_base)
        radial = [4*value/(a-b)**2 for value in (-a*b, a+b, -1)]
        bubble = [angular*value for value in _poly_product(radial, sphere)]
        grad_lambda = mp.matrix((0, 0, mp.mpf(.5)))
        lam_grad_nodes = (mp.matrix((-1, -1, (nodes[1]+nodes[2])/2)),
            mp.matrix((1, 0, -nodes[1]/2)), mp.matrix((0, 1, -nodes[2]/2)))
        lam_grad_angular = 27*(nodes[1]*nodes[2]*lam_grad_nodes[0]
            +nodes[0]*nodes[2]*lam_grad_nodes[1]+nodes[0]*nodes[1]*lam_grad_nodes[2])
        birth_offset = [mp.matrix((-.125, -.125, -1)), mp.matrix((.25, .25, 2))]
        lam_grad_sphere = [(-4/mp.mpf(.25))*value
            for value in _poly_product([0, 1], _poly_product(sphere_base, birth_offset))]
        lam_grad_radial = _poly_product([0, 1],
            [4*(a+b)/(a-b)**2*grad_lambda, -8/(a-b)**2*grad_lambda])
        L = _poly_sum(_poly_product(_poly_product(radial, sphere), [lam_grad_angular]),
            [angular*value for value in _poly_product(radial, lam_grad_sphere)],
            [angular*value for value in _poly_product(sphere, lam_grad_radial)])
        Bhat, trace = [mp.zeros(6, 18), mp.zeros(6, 18)], [mp.zeros(3, 18), mp.zeros(3, 18)]
        for vertex in range(3):
            for axis in range(3):
                direction = mp.eye(3)[:, axis]
                column = 6*vertex+axis
                Bhat[0][:, column] = _sym(direction, lam_grad_nodes[vertex])
                Bhat[0][:, column+3] = -_sym(direction, lam_grad_nodes[vertex])
                Bhat[1][:, column+3] = _sym(direction, lam_grad_nodes[vertex]+nodes[vertex]*grad_lambda)
                trace[0][axis, column] = nodes[vertex]
                trace[0][axis, column+3] = -nodes[vertex]
                trace[1][axis, column+3] = nodes[vertex]
        Ghat = _poly_product(bubble, Bhat)
        gradient_term = [mp.zeros(6, 18) for _ in range(len(trace)+len(L)-1)]
        for i, T in enumerate(trace):
            for j, gradient in enumerate(L):
                for column in range(18):
                    gradient_term[i+j][:, column] += _sym(T[:, column], gradient)
        Ghat = _poly_sum(Ghat, gradient_term)
        fields = dict(lam=[0, 1], w=bubble, L=L, Bhat=Bhat, Ghat_without_H=Ghat,
            T=trace, E_without_H=_poly_product(bubble, trace), memoryhat=[mp.zeros(6, 1)])
        centered = {key: _center_polynomial(value, mid, half) for key, value in fields.items()}
        bounds = {key: [mp.matrix([[abs(v[i, j]) for j in range(v.cols)]
            for i in range(v.rows)]) if hasattr(v, "rows") else abs(v) for v in polynomial]
            for key, polynomial in centered.items()}
        witness = dict(interval=dict(lambda_mid="0.5", lambda_halfwidth=mp.nstr(half, 95), gamma_root_xi="0",
            literal_phi_coefficients=["0", mp.nstr(2*half, 95)], reference_determinant_m3="2"),
            coefficients={key: _decimal_polynomial(value) for key, value in centered.items()},
            coefficient_roundoff_bounds={key: _decimal_polynomial(value, mp.mpf("1e-60"))
                for key, value in bounds.items()})
    return witness


def test_analytic_cartesian_coefficient_fields_pass_without_a_producer_record(analytic_coefficient_case):
    raw, witness = analytic_coefficient_case
    before = deepcopy((raw, witness))
    report = audit.audit_coefficient_fields(raw, N, witness, precision=80)
    assert report["coefficient_source_status"] == "PASS"
    assert report["checked_component_count"] == 670
    assert report["angular_accuracy_certified"] is False
    assert report["physical_creation_admitted"] is False
    assert (raw, witness) == before


def test_analytic_bubble_constant_corruption_is_identified_by_source_field_audit(analytic_coefficient_case):
    raw, original = analytic_coefficient_case
    witness = deepcopy(original)
    witness["coefficients"]["w"][0] = "0"
    report = audit.audit_coefficient_fields(raw, N, witness, precision=80, raise_on_mismatch=False)
    assert report["coefficient_source_status"] == "OPEN"
    assert len(report["mismatching_components"]) == 2
    assert all(".w" in name for name in report["mismatching_components"])


def test_direct_stiffness_matches_cartesian_polynomial_antiderivatives(direct_line_result):
    witness = _analytic_coefficient_witness(_source())
    with mp.workdps(100):
        B = [mp.matrix(value)[:, 0] for value in witness["coefficients"]["Bhat"]]
        G = [mp.matrix(value)[:, 0] for value in witness["coefficients"]["Ghat_without_H"]]
        def dot_coefficients(left, right):
            result = [mp.mpf(0)]*(len(left)+len(right)-1)
            for i, a in enumerate(left):
                for j, b in enumerate(right):
                    result[i+j] += sum(a[row]*b[row] for row in range(6))
            return result
        def primitive(coefficients, low, high):
            return sum(value*(high**(power+1)-low**(power+1))/(power+1)
                       for power, value in enumerate(coefficients))
        half = mp.mpf(witness["interval"]["lambda_halfwidth"])
        cross_poly, self_poly = dot_coefficients(B, G), dot_coefficients(G, G)
        # detX=2 and H=±1/2.  The reference λ² cancels the two
        # explicit 1/λ factors of B and G before integration.
        cross = half*(primitive(cross_poly, 0, 1)-primitive(cross_poly, -1, 0))
        self_stiffness = half/2*primitive(self_poly, -1, 1)
        values = dict(zip(direct_line_result["component_names"], direct_line_result["flattened_values"]))
        _close(values["projected.stiffness_cross[0,0]"], cross)
        _close(values["projected.stiffness_enriched[0,0]"], self_stiffness)
        assert self_stiffness > 0


def _primitive_witness(root="0", slope=1, widths=None):
    interval = dict(lambda_mid="0.5", lambda_halfwidth="0.25",
                    gamma_root_xi=root, literal_phi_coefficients=[0, slope])
    if widths is not None:
        interval.update(gamma_left_width_xi=widths[0], gamma_right_width_xi=widths[1])
    return dict(interval=interval)


@pytest.mark.parametrize("degree", range(7))
@pytest.mark.parametrize("side_power", (0, 1, 2))
def test_centered_monomial_primitive_has_exact_signed_bank_goldens(degree, side_power):
    coefficients = [0]*degree+[1]
    result = audit.independent_polynomial_integral(_primitive_witness(), coefficients,
                                                  side_power=side_power, precision=80)
    with mp.workdps(100):
        if side_power == 1:
            golden = mp.mpf(1)/(4*(degree+1)) if degree % 2 else mp.mpf(0)
        else:
            golden = mp.mpf(1)/(2*(degree+1))*mp.mpf(.5)**side_power if not degree % 2 else mp.mpf(0)
        _close(result["value"].item(), golden)
        assert result["absolute_operand_scale"].item() > 0
        assert result["high_precision_arithmetic_bound"].item() > 0
    assert result["scope"] == "represented_polynomial_antiderivative_no_source_geometry_or_angular_certificate"
    assert result["source_geometry_accuracy_certified"] is False
    assert result["angular_accuracy_certified"] is False
    assert result["physical_contraction_certified"] is False


def test_gamma_orientation_reverses_signed_primitive_without_changing_self_integral():
    negative = _primitive_witness(slope=-1)
    with mp.workdps(100):
        signed = audit.independent_polynomial_integral(negative, [0, 1], side_power=1)
        self_integral = audit.independent_polynomial_integral(negative, [1], side_power=2)
        _close(signed["value"].item(), -mp.mpf(1)/8)
        _close(self_integral["value"].item(), mp.mpf(1)/8)


def test_no_gamma_split_uses_the_declared_constant_side():
    witness = dict(interval=dict(lambda_mid=.5, lambda_halfwidth=.25,
                                gamma_root_xi=None, constant_side=-.5))
    result = audit.independent_polynomial_integral(witness, [1, 0, 1], side_power=1)
    with mp.workdps(100):
        _close(result["value"].item(), -mp.mpf(1)/3)


def test_separately_archived_tiny_gamma_bank_survives_rounded_endpoint_root():
    with mp.workdps(140):
        width = mp.mpf("1e-40")
        right = 2-width
        witness = _primitive_witness(root=-1., widths=("1e-40", mp.nstr(right, 135)))
        witness["interval"]["literal_phi_coefficients"] = [mp.nstr(1-width, 135), 1]
        # Public endpoint root rounds to -1 in binary64.  Independent archived
        # side widths preserve the small positive bank instead of erasing it.
        result = audit.independent_polynomial_integral(witness, [0, 1],
            side_power=1, precision=120)
        expected = mp.mpf(.25)*(width-width**2/2)
        assert result["value"].item() > 0
        _close(result["value"].item(), expected)


def test_zero_polynomial_primitive_has_exact_zero_own_scale_and_bound():
    result = audit.independent_polynomial_integral(_primitive_witness(), [[0, 0], [0, 0]])
    for key in ("value", "absolute_operand_scale", "high_precision_arithmetic_bound"):
        assert result[key].shape == (2,)
        assert all(value == 0 for value in result[key])


@pytest.mark.parametrize("coefficient", (mp.inf, mp.nan))
def test_nonfinite_MP_polynomial_operand_is_refused(coefficient):
    with pytest.raises(audit.IndependentLineRefusal):
        audit.independent_polynomial_integral(_primitive_witness(), [coefficient])


@pytest.mark.parametrize("side_power", (-1, 3, True, 1.))
def test_primitive_refuses_invalid_side_power(side_power):
    with pytest.raises(audit.IndependentLineRefusal):
        audit.independent_polynomial_integral(_primitive_witness(), [1], side_power=side_power)


@pytest.mark.parametrize("widths", ((0, 2), (-.1, 2.1), (1, 2), ("1e-100", "2")))
def test_invalid_or_unresolved_represented_gamma_widths_are_refused(widths):
    with pytest.raises(audit.IndependentLineRefusal):
        audit.independent_polynomial_integral(_primitive_witness(root=-1., widths=widths), [1], precision=80)


def test_represented_primitive_refuses_field_degree_above_declared_contraction_limit(analytic_coefficient_case):
    raw, original = analytic_coefficient_case
    witness = deepcopy(original)
    witness["coefficients"]["Ghat_without_H"].append([[0]*18 for _ in range(6)])
    with pytest.raises(audit.IndependentLineRefusal, match="degree exceeds"):
        audit.audit_represented_primitive(raw, N, witness)


@pytest.fixture(scope="module")
def represented_primitive_case(analytic_coefficient_case):
    raw, coefficients = analytic_coefficient_case
    basis, probes = _integration_inputs()
    direct = audit.integrate_source_line(raw, N, parent_basis=basis, added_basis=basis,
                                        probes=probes, precision=80)
    witness = _line_witness(direct)
    witness["interval"].update(coefficients["interval"])
    witness["coefficients"] = deepcopy(coefficients["coefficients"])
    witness["coefficient_roundoff_bounds"] = deepcopy(coefficients["coefficient_roundoff_bounds"])
    return raw, witness


def test_full_projected_and_probe_represented_primitive_accepts_independent_source_values(represented_primitive_case):
    raw, witness = represented_primitive_case
    before = deepcopy((raw, witness))
    basis, probes = _integration_inputs()
    report = audit.audit_represented_primitive(raw, N, witness, parent_basis=basis,
                                               added_basis=basis, probes=probes)
    assert report["represented_polynomial_primitive_status"] == "PASS"
    assert report["checked_component_count"] == len(witness["component_names"])
    assert report["declared_contraction_maximum_degree"] == 16
    assert report["arithmetic_budget_operand_authentication_certified"] is False
    assert report["source_geometry_accuracy_certified"] is False
    assert report["physical_creation_admitted"] is False
    checks = report["component_checks"]
    with mp.workdps(100):
        weak = checks["probe.weak.mass_cross_bilinear_kg_m2"]
        assert mp.mpf(weak["actual"]) != 0
        assert 0 < mp.mpf(weak["own_operand_scale"]) < mp.mpf("1e-210")
        assert weak["positive_scale_floor_used"] is False
        for term in audit.TERMS:
            check = checks["probe.zero."+term]
            assert mp.mpf(check["actual"]) == mp.mpf(check["own_operand_scale"]) == mp.mpf(check["arithmetic_bound"]) == 0
    assert (raw, witness) == before


def test_represented_primitive_identifies_single_corrupted_full_stiffness_component(represented_primitive_case):
    raw, original = represented_primitive_case
    witness = deepcopy(original)
    name = "full.stiffness_enriched[0,0]"
    position = witness["component_names"].index(name)
    witness["native_exact_binary_ratio"]["values"]["numerator_denominator"][position] = ["0", "1"]
    basis, probes = _integration_inputs()
    report = audit.audit_represented_primitive(raw, N, witness, parent_basis=basis,
        added_basis=basis, probes=probes, raise_on_mismatch=False)
    assert report["represented_polynomial_primitive_status"] == "OPEN"
    assert report["mismatching_components"] == [name]
    check = report["component_checks"][name]
    with mp.workdps(100):
        assert mp.mpf(check["expected"]) > 0
        assert mp.mpf(check["absolute_difference"]) > mp.mpf(check["arithmetic_bound"])
    assert check["positive_scale_floor_used"] is False


def test_negative_saved_represented_budget_is_refused_before_MP_error_is_added(represented_primitive_case):
    raw, original = represented_primitive_case
    witness = deepcopy(original)
    position = witness["component_names"].index("full.stiffness_enriched[0,0]")
    witness["roundoff_component_bounds"][position] = -1e-100
    basis, probes = _integration_inputs()
    with pytest.raises(audit.IndependentLineRefusal):
        audit.audit_represented_primitive(raw, N, witness, parent_basis=basis,
                                         added_basis=basis, probes=probes)


@pytest.fixture(scope="module")
def bounded_quadrature_refusal_case():
    # Eight Gauss nodes integrate through degree15, while the full original
    # P/P enriched mass contains a genuine degree16 term.  Twelve nodes do
    # integrate this declared polynomial.  The two supported orders therefore
    # must refuse the unchanged own-arithmetic agreement criterion.
    raw = _source()
    before = deepcopy(raw)
    basis, probes = _integration_inputs()
    with pytest.raises(audit.IndependentLineRefusal, match="did not agree") as raised:
        audit.integrate_source_line(raw, N, parent_basis=basis, added_basis=basis,
            probes=probes, precision=80, quadrature_orders=(8, 12))
    assert raw == before
    return raised.value.diagnostics


def _assert_refusal_observations(diagnostics, orders):
    assert isinstance(diagnostics, dict)
    assert diagnostics["status"] == "BOUNDED_HIGH_PRECISION_DIRECT_QUADRATURE_REFUSED"
    assert diagnostics["precision_decimal_digits"] == 80
    assert diagnostics["quadrature_orders"] == orders
    names = diagnostics["component_names"]
    assert diagnostics["checked_component_count"] == len(names) == len(set(names))
    assert diagnostics["failed_component_count"] == len(diagnostics["failed_component_checks"]) > 0
    assert len(diagnostics["oracle_input_binding_sha256"]) == 64
    for key in ("coarse_observed_values", "fine_observed_values",
                "coarse_absolute_operand_scales", "fine_absolute_operand_scales"):
        assert len(diagnostics[key]) == len(names)
    assert "flattened_values" not in diagnostics
    assert "numerical_error_bounds" not in diagnostics
    for key in ("trusted_oracle_available", "observed_fine_values_certified",
                "source_geometry_accuracy_certified", "angular_accuracy_certified",
                "general_contraction_accuracy_certified", "physical_contraction_certified",
                "physical_creation_admitted", "physical_release_admitted", "available_for_creation_funding"):
        assert diagnostics[key] is False
    with mp.workdps(100):
        for name, check in diagnostics["failed_component_checks"].items():
            position = names.index(name)
            coarse = mp.mpf(diagnostics["coarse_observed_values"][position])
            fine = mp.mpf(diagnostics["fine_observed_values"][position])
            coarse_own = mp.mpf(diagnostics["coarse_absolute_operand_scales"][position])
            fine_own = mp.mpf(diagnostics["fine_absolute_operand_scales"][position])
            own = max(coarse_own, fine_own)
            _close(check["coarse_observed_value"], coarse)
            _close(check["fine_observed_value"], fine)
            _close(check["absolute_difference"], abs(coarse-fine))
            _close(check["own_operand_scale"], own)
            budget = audit._ctx(80).eps*262144*own
            _close(check["arithmetic_bound"], budget)
            assert mp.mpf(check["absolute_difference"]) > mp.mpf(check["arithmetic_bound"])
            assert check["positive_scale_floor_used"] is False
            if budget == 0:
                assert check["ratio"] is None
            else:
                _close(check["ratio"], abs(coarse-fine)/budget, "1e-25")
        for term in audit.TERMS:
            name = "probe.zero."+term
            position = names.index(name)
            assert name not in diagnostics["failed_component_checks"]
            assert all(diagnostics[key][position] == 0 for key in
                ("coarse_observed_values", "fine_observed_values",
                 "coarse_absolute_operand_scales", "fine_absolute_operand_scales"))


def test_real_degree16_mass_quadrature_refusal_retains_structured_own_observations(bounded_quadrature_refusal_case):
    diagnostics = bounded_quadrature_refusal_case
    _assert_refusal_observations(diagnostics, (8, 12))
    name = "full.mass_enriched[3,3]"
    assert name in diagnostics["failed_component_checks"]
    # Independent polynomial antiderivative identifies the fine observation
    # for this one small golden without promoting all observations to oracle.
    with mp.workdps(100):
        b, a = (Fraction.from_float(value) for value in (.2, .9))
        radial = [-a*b, a+b, Fraction(-1)]
        sphere = [Fraction(-25, 8), Fraction(33, 2), Fraction(-33, 2)]
        bubble = [Fraction(27, 8)/(a-b)**2*value
                  for value in _multiply(radial, _multiply(sphere, sphere))]
        mass_coefficients = _multiply([Fraction(0), Fraction(0), Fraction(1), Fraction(-2), Fraction(1)],
                                      _multiply(bubble, bubble))
        half = mp.sqrt(mp.mpf(2)/33)
        golden = _integral(mass_coefficients, mp.mpf(.5)-half, mp.mpf(.5)+half)/8
        position = diagnostics["component_names"].index(name)
        _close(diagnostics["fine_observed_values"][position], golden)
    assert diagnostics["observed_fine_values_certified"] is False


def test_refused_observations_archive_preserves_refusal_and_reproducible_seal(bounded_quadrature_refusal_case):
    archived = audit.archive_line_result(bounded_quadrature_refusal_case)
    assert archived["format"] == "independent-source-fixed-N-line-refusal-0.1"
    assert archived["status"] == "BOUNDED_HIGH_PRECISION_DIRECT_QUADRATURE_REFUSED"
    assert archived["trusted_oracle_available"] is False
    assert archived["observed_fine_values_certified"] is False
    assert all(isinstance(value, str) for value in archived["fine_observed_values"])
    payload = deepcopy(archived)
    digest = payload.pop("sha256")
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    assert digest == hashlib.sha256(raw).hexdigest()


def test_perturbed_positive_fine_weights_refuse_default12_20_without_changing_criterion(monkeypatch):
    original_gauss = audit._gauss
    calls = []
    def perturbed_positive_gauss(precision, order):
        calls.append((precision, order))
        nodes, weights = original_gauss(precision, order)
        assert all(weight > 0 for weight in weights)
        if order == 20:
            weights = tuple(weight*(1+2.**-20) for weight in weights)
        assert all(weight > 0 for weight in weights)
        return nodes, weights
    monkeypatch.setattr(audit, "_gauss", perturbed_positive_gauss)
    raw = _source()
    before = deepcopy(raw)
    basis, probes = _integration_inputs()
    with pytest.raises(audit.IndependentLineRefusal, match="did not agree") as raised:
        audit.integrate_source_line(raw, N, parent_basis=basis, added_basis=basis, probes=probes)
    assert calls == [(80, 12), (80, 20)]
    assert raw == before
    diagnostics = raised.value.diagnostics
    _assert_refusal_observations(diagnostics, (12, 20))
    weak_name = "probe.weak.mass_cross_bilinear_kg_m2"
    assert weak_name in diagnostics["failed_component_checks"]
    with mp.workdps(100):
        weak = diagnostics["failed_component_checks"][weak_name]
        assert mp.mpf(weak["fine_observed_value"]) < 0
        assert 0 < mp.mpf(weak["own_operand_scale"]) < mp.mpf("1e-210")


def test_public_geometry_observation_does_not_need_a_direct_quadrature_or_constitutive_source(monkeypatch):
    raw = _source()
    for key in ("C", "Q", "e0", "e1"):
        del raw[key]
    for key in ("mass_kg", "volume_m3"):
        del raw["geometry_source"][key]
    true = audit.independent_line_geometry(raw, N)
    witness = dict(interval={key: true[key] for key in
        ("lambda_mid", "lambda_halfwidth", "gamma_root_xi")})
    before = deepcopy((raw, witness))
    def forbidden_integral(*args, **kwargs):
        pytest.fail("Geometry observation must not depend on quadrature agreement")
    monkeypatch.setattr(audit, "integrate_source_line", forbidden_integral)
    report = audit.audit_geometry_witness(raw, N, witness)
    assert report["geometry_status"] == "PASS"
    assert report["geometry_checks"]["Gamma_split_matches_literal_source"] is True
    for key in ("source_geometry_accuracy_certified", "angular_accuracy_certified",
                "general_contraction_accuracy_certified", "physical_contraction_certified",
                "physical_creation_admitted", "physical_release_admitted", "available_for_creation_funding"):
        assert report[key] is False
    assert (raw, witness) == before


@pytest.mark.parametrize("mutation", ("false_empty", "gamma_split", "halfwidth"))
def test_public_geometry_wrapper_reports_source_disagreement_separately_from_component_checks(mutation):
    raw = _source()
    geometry = audit.independent_line_geometry(raw, N)
    witness = dict(interval={key: geometry[key] for key in
        ("lambda_mid", "lambda_halfwidth", "gamma_root_xi")})
    if mutation == "false_empty":
        witness["interval"] = None
    elif mutation == "gamma_split":
        witness["interval"]["gamma_root_xi"] = .125
    else:
        witness["interval"]["lambda_halfwidth"] = .125
    report = audit.audit_geometry_witness(raw, N, witness)
    assert report["geometry_status"] == "OPEN"
    assert report["source_geometry_accuracy_certified"] is False
    assert "source_direct_quadrature_status" not in report


@pytest.mark.parametrize("forge_success_status", (False, True))
def test_refusal_observations_cannot_be_laundered_into_a_cached_trusted_oracle(bounded_quadrature_refusal_case, forge_success_status):
    observations = deepcopy(bounded_quadrature_refusal_case)
    observations["flattened_values"] = observations["fine_observed_values"]
    observations["absolute_operand_scales"] = tuple(max(a, b) for a, b in zip(
        observations["coarse_absolute_operand_scales"], observations["fine_absolute_operand_scales"]))
    observations["numerical_error_bounds"] = (0,)*len(observations["component_names"])
    if forge_success_status:
        observations["status"] = "HIGH_PRECISION_FIXED_LINE_QUADRATURE_AGREEMENT"
    # Copying the observed numbers under successful-cache keys, or changing
    # the headline status, cannot remove their explicit untrusted provenance.
    assert observations["trusted_oracle_available"] is False
    assert observations["observed_fine_values_certified"] is False
    with pytest.raises(audit.IndependentLineRefusal):
        _audit_witness(observations, _line_witness(observations))


def test_translated_narrow_support_cannot_borrow_absolute_midpoint_scale():
    raw = _source(center=(0., 0., 1.), radius=1e-20, offset=2.)
    geometry = audit.independent_line_geometry(raw, (1., 0., 0.), precision=80)
    witness = dict(interval=dict(lambda_mid=geometry["lambda_mid"],
        lambda_halfwidth=geometry["lambda_halfwidth"], gamma_root_xi=None, constant_side=-.5))
    report = audit.audit_geometry_witness(raw, (1., 0., 0.), witness, precision=80)
    assert report["geometry_status"] == "PASS"
    # This exact decimal moves the midpoint by 100 full support widths,
    # while retaining the width and literal constant negative Gamma side.
    witness["interval"]["lambda_mid"] = "0.500000000000000001"
    report = audit.audit_geometry_witness(raw, (1., 0., 0.), witness, precision=80)
    assert report["geometry_status"] == "OPEN"
    assert report["geometry_checks"]["halfwidth_matches_own_scale"] is True
    assert report["source_geometry_accuracy_certified"] is False


def test_exact_local_lambda_distance_preserves_P_trace_and_birth_offset_at_unchanged_MP80():
    raw = _source(center=(.25, .25, 2.), radius=1e-90, offset=2.)
    raw["geometry_source"]["lambda_bounds"] = [.9, 1.1]
    before = deepcopy(raw)
    delta = Fraction(1, 10**100)
    lam = 1+delta
    context = audit._ctx(80)
    assert context.mpf(lam.numerator)/lam.denominator == 1
    fields = audit.evaluate_source_fields(raw, N, lam, .5, precision=80)
    with mp.workdps(150):
        exact_delta = mp.mpf(delta.numerator)/delta.denominator
        for vertex, node in enumerate(N):
            for axis in range(3):
                _close(fields["T"][axis, 6*vertex+3+axis], mp.mpf(node)*exact_delta)
        radius2 = mp.mpf(raw["seed_radius_m"])**2
        offset_squared = exact_delta**2*(mp.mpf(4)+mp.mpf(1)/8)
        sphere = (1-offset_squared/radius2)**2
        b, a = (Fraction.from_float(value) for value in raw["geometry_source"]["lambda_bounds"])
        radial_exact = 4*(lam-b)*(a-lam)/(a-b)**2
        radial = mp.mpf(radial_exact.numerator)/radial_exact.denominator
        center_bubble = mp.mpf(27)/32*radial
        _close(fields["w"], center_bubble*sphere)
        assert mp.mpf(fields["w"]) < center_bubble
    assert raw == before


def test_exact_lambda_outside_original_radial_support_refuses_even_when_MP80_rounds_to_bound():
    raw = _source(center=(.25, .25, 2.), radius=1e-90, offset=2.)
    raw["geometry_source"]["lambda_bounds"] = [.9, 1.]
    lam = 1+Fraction(1, 10**100)
    context = audit._ctx(80)
    assert context.mpf(lam.numerator)/lam.denominator == 1
    with pytest.raises(audit.IndependentLineRefusal):
        audit.evaluate_source_fields(raw, N, lam, .5, precision=80)
