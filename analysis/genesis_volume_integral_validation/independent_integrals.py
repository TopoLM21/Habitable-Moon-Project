"""Exact rational source-line integrals, including displaced sphere endpoints.

The source coefficients and polynomial contractions are independent of the
mechanical producer.  Integer-square-root enclosures describe the original
sphere/cohort intersection.  Endpoint antiderivatives are translated at the
represented chart endpoints, so a tiny displacement is integrated as a tiny
displacement.  Literal source Gamma cuts are retained even in endpoint slivers.

This is a selected-radial-line certificate.  It supplies neither an angular
quadrature certificate nor mechanical admission, and never advances Genesis.
"""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np

from analysis.genesis_volume_operand_validation import independent_contractions as algebra
from analysis.genesis_volume_operand_validation.independent_coefficients import exact_source_coefficients
from analysis.genesis_volume_operand_validation.independent_geometry_bounds import source_line, _source_interval


FORMAT = "genesis-true-source-radial-integral-certificate-0.1"
ZERO = Fraction(0)
ONE = Fraction(1)
VOLUME_NAMES = ("reference_volume_negative_m3", "reference_volume_positive_m3")


class SourceIntegralRefusal(ValueError):
    """The source-line truth or a claimed enclosure is inconsistent."""


def _require(condition, message):
    if not condition:
        raise SourceIntegralRefusal(message)


def _q(value):
    return algebra._fraction(value)


def _interval(value):
    _require(isinstance(value, (tuple, list)) and len(value) == 2, "Interval shape differs")
    result = tuple(map(_q, value))
    _require(result[0] <= result[1], "Reversed rational interval")
    return result


def _add(left, right):
    return left[0]+right[0], left[1]+right[1]


def _subtract(left, right):
    return left[0]-right[1], left[1]-right[0]


def _multiply(left, right):
    products = tuple(a*b for a in left for b in right)
    return min(products), max(products)


def _scale(value, scalar):
    scalar = _q(scalar)
    a, b = value[0]*scalar, value[1]*scalar
    return min(a, b), max(a, b)


def _union(left, right):
    return min(left[0], right[0]), max(left[1], right[1])


def translated_antiderivative_interval(polynomial, anchor, endpoint):
    """Enclose integral(anchor, x) p(x) dx for all x in endpoint, exactly.

    Polynomial coefficients use ascending powers.  The endpoint displacement
    is the Horner variable; no rounded subtraction of absolute primitives is
    involved.  Interval arithmetic here is rational, with no rounding credit.
    """
    p = tuple(map(_q, polynomial))
    anchor, endpoint = _q(anchor), _interval(endpoint)
    _require(1 <= len(p) <= 17, "Source polynomial exceeds radial degree 16")
    translated = tuple(sum((p[k]*math.comb(k, j)*anchor**(k-j)
                            for k in range(j, len(p))), ZERO)
                       for j in range(len(p)))
    delta = endpoint[0]-anchor, endpoint[1]-anchor
    result = (translated[-1]/len(p),)*2
    for j in range(len(p)-2, -1, -1):
        result = _add(_multiply(result, delta), (translated[j]/(j+1),)*2)
    return _multiply(result, delta)


def _literal_side(phi0, phi1, x):
    value = phi0+phi1*x
    return Fraction(1, 2) if value > 0 else Fraction(-1, 2) if value < 0 else ZERO


def signed_endpoint_integral(polynomial, anchor, endpoint, side_power, phi0, phi1):
    """Rational enclosure of a signed endpoint correction with literal Gamma.

    For H, intervals that straddle the rational Gamma root are split before
    integration.  H² is exactly 1/4 almost everywhere on a positive line.
    Endpoint direction may be inward or outward; signs remain explicit.
    """
    anchor, endpoint = _q(anchor), _interval(endpoint)
    phi0, phi1 = _q(phi0), _q(phi1)
    _require(side_power in (0, 1, 2), "Gamma power differs")
    _require(phi0 or phi1, "Positive source line lies identically in Gamma")
    if side_power in (0, 2):
        return _scale(translated_antiderivative_interval(polynomial, anchor, endpoint),
                      ONE if side_power == 0 else Fraction(1, 4))
    if not phi1:
        return _scale(translated_antiderivative_interval(polynomial, anchor, endpoint),
                      _literal_side(phi0, phi1, anchor))
    root = -phi0/phi1
    orientation = ONE if phi1 > 0 else -ONE
    # A continuous antiderivative anchored at Gamma gives each bank a simple
    # exact expression.  Translating at anchor still makes small slivers small.
    at_root = translated_antiderivative_interval(polynomial, anchor, (root, root))[0]
    anchor_bank = Fraction(1, 2)*orientation if anchor >= root else Fraction(-1, 2)*orientation
    to_root = anchor_bank*at_root
    def bank(low, high, bank_side):
        plain = translated_antiderivative_interval(polynomial, anchor, (low, high))
        return _add((to_root, to_root), _scale(_subtract(plain, (at_root, at_root)), bank_side))
    left_side, right_side = -orientation/2, orientation/2
    if endpoint[1] <= root:
        return bank(*endpoint, left_side)
    if endpoint[0] >= root:
        return bank(*endpoint, right_side)
    return _union(bank(endpoint[0], root, left_side), bank(root, endpoint[1], right_side))


def bank_endpoint_integral(polynomial, anchor, endpoint, bank_index, phi0, phi1):
    """Integrate one exact negative/positive Gamma indicator in an endpoint sliver."""
    anchor, endpoint = _q(anchor), _interval(endpoint)
    phi0, phi1 = _q(phi0), _q(phi1)
    _require(bank_index in (0, 1), "Reference bank index differs")
    _require(phi0 or phi1, "Positive source line lies identically in Gamma")
    if not phi1:
        active = int((phi0 > 0) == (bank_index == 1))
        return _scale(translated_antiderivative_interval(polynomial, anchor, endpoint), active)
    root = -phi0/phi1
    right_active = int((phi1 > 0) == (bank_index == 1))
    left_active = 1-right_active
    at_root = translated_antiderivative_interval(polynomial, anchor, (root, root))[0]
    anchor_active = right_active if anchor >= root else left_active
    to_root = anchor_active*at_root
    def bank(low, high, active):
        if not active:
            return (to_root, to_root)
        plain = translated_antiderivative_interval(polynomial, anchor, (low, high))
        return _add((to_root, to_root), _subtract(plain, (at_root, at_root)))
    if endpoint[1] <= root:
        return bank(*endpoint, left_active)
    if endpoint[0] >= root:
        return bank(*endpoint, right_active)
    return _union(bank(endpoint[0], root, left_active), bank(root, endpoint[1], right_active))


def source_moment_enclosures(raw, N, witness, *, binary_places=256):
    """Return exact base moments and interval endpoint corrections in native ξ."""
    native_N = algebra.decode_native_array(witness, "interval.barycentric")
    _require(native_N.shape == (3,) and np.array_equal(native_N, algebra._array(N)),
             "Captured native material N differs")
    midpoint = algebra._scalar(witness, "lambda_mid")
    halfwidth = algebra._scalar(witness, "lambda_halfwidth")
    _require(halfwidth > 0, "Captured radial chart is not positive")
    line = source_line(raw, N)
    truth = _source_interval(line, binary_places)
    _require(truth is not None, "Original source sphere has no positive support")
    lower = tuple((q-midpoint)/halfwidth for q in truth["low"])
    upper = tuple((q-midpoint)/halfwidth for q in truth["high"])
    _require(lower[1] < upper[0], "Source radial interval remains unresolved")
    phi0 = line["slope"]*midpoint+line["intercept"]
    phi1 = line["slope"]*halfwidth
    _require(phi0 or phi1, "Positive source line lies identically in Gamma")
    base, corrections, moments = {}, {}, {}
    for side in (0, 1, 2):
        base[side] = algebra.exact_moments(witness, 16, side, gamma_mode="literal_source", raw=raw, N=N)
        corrections[side], moments[side] = [], []
        for degree in range(17):
            p = (ZERO,)*degree+(ONE,)
            left = signed_endpoint_integral(p, -ONE, lower, side, phi0, phi1)
            right = signed_endpoint_integral(p, ONE, upper, side, phi0, phi1)
            delta = _scale(_subtract(right, left), halfwidth)
            corrections[side].append(delta)
            moments[side].append(_add((base[side][degree],)*2, delta))
        corrections[side] = tuple(corrections[side])
        moments[side] = tuple(moments[side])
    # Separate indicator antiderivatives retain exact inactive-bank zeros.
    # Subtracting independently enclosed ordinary/H moments would lose their
    # algebraic dependence at a bank whose exact measure is zero.
    for bank_index, sign in ((0, -ONE), (1, ONE)):
        side = 3+bank_index
        base[side] = tuple(a/2+sign*b for a, b in zip(base[0], base[1]))
        corrections[side], moments[side] = [], []
        for degree in range(17):
            p = (ZERO,)*degree+(ONE,)
            left = bank_endpoint_integral(p, -ONE, lower, bank_index, phi0, phi1)
            right = bank_endpoint_integral(p, ONE, upper, bank_index, phi0, phi1)
            delta = _scale(_subtract(right, left), halfwidth)
            corrections[side].append(delta)
            moments[side].append(_add((base[side][degree],)*2, delta))
        corrections[side] = tuple(corrections[side])
        moments[side] = tuple(moments[side])
    return dict(base=base, endpoint_corrections=corrections, true_moments=moments,
                source_interval=truth, source_lower_xi=lower, source_upper_xi=upper,
                represented_lambda_mid=midpoint, represented_lambda_halfwidth=halfwidth,
                literal_gamma_phi=(phi0, phi1), binary_places=binary_places)


def source_component_polynomials(raw, N, witness, *, added_basis=None, parent_basis=None, probes=()):
    """Rebuild each component as exact source polynomial terms (H power, p)."""
    f = exact_source_coefficients(raw, N, algebra._scalar(witness, "lambda_mid"),
                                  algebra._scalar(witness, "lambda_halfwidth"))
    C = algebra._array(algebra._alias(raw, "elasticity_pa", "C"))
    _require(C.shape == (6, 6), "Elasticity shape differs")
    det, mass_factor = algebra._weights(raw, witness, "literal_source")
    lam2 = algebra.polynomial_product(f["lam"], f["lam"])
    result = {}
    def append(prefix, key, polynomial, side, factor, *, squeeze=False):
        polynomial = algebra._array(polynomial)*factor
        if squeeze:
            polynomial = polynomial[:, :, 0]
        _require(len(polynomial) <= 17, "Source component degree exceeds 16")
        for index in np.ndindex(polynomial.shape[1:]):
            suffix = "["+",".join(map(str, index))+"]" if index else ""
            name = prefix+"."+key+suffix if prefix else key+suffix
            result.setdefault(name, []).append((side, tuple(polynomial[(slice(None),)+index])))
    # Old angular vector begins with the two original reference-measure banks.
    for bank_index, name in enumerate(VOLUME_NAMES):
        result[name] = [(3+bank_index, tuple(det*q for q in lam2))]
    lam2_matrix = lam2.reshape((len(lam2), 1, 1))
    def blocks(prefix, V, W):
        B = algebra.polynomial_project(f["Bhat"], V)
        G = algebra.polynomial_project(f["Ghat_without_H"], W)
        T = algebra.polynomial_project(f["T"], V)
        E = algebra.polynomial_project(f["E_without_H"], W)
        for name, left, middle, right, side, factor, mass in (
            ("stiffness_cross", B, C, G, 1, det, False),
            ("stiffness_enriched", G, C, G, 2, det, False),
            ("mass_cross", T, algebra._identity(3), E, 1, mass_factor, True),
            ("mass_enriched", E, algebra._identity(3), E, 2, mass_factor, True),
            ("enrichment_force_n", G, C, f["memoryhat"][:, :, None], 1, det, False)):
            p = algebra.polynomial_bilinear(left, middle, right)
            if mass:
                p = algebra.polynomial_product(lam2_matrix, p)
            append(prefix, name, p, side, factor, squeeze=name == "enrichment_force_n")
    blocks("full", algebra._identity(18), algebra._identity(18))
    if added_basis is not None:
        V = algebra._identity(18) if parent_basis is None else algebra._projector(parent_basis, "Parent basis")
        blocks("projected", V, algebra._projector(added_basis, "Added basis"))
    _require(len(probes) <= 16 and len({p.get("name") for p in probes}) == len(probes),
             "Declared probe names/count differ")
    for probe in probes:
        _require(isinstance(probe.get("name"), str) and probe["name"], "Probe name differs")
        prefix = "probe."+probe["name"]
        q = algebra._probe_vector(probe, "parent_increment_m", "q")
        a = algebra._probe_vector(probe, "enrichment_increment_m", "a")
        z = algebra._probe_vector(probe, "mean_z_m", "mean_z", np.zeros(18))
        Bq, Ga, Gz = (algebra.polynomial_project(f[key], basis) for key, basis in
                       (("Bhat", q), ("Ghat_without_H", a), ("Ghat_without_H", z)))
        Tq, Ea = algebra.polynomial_project(f["T"], q), algebra.polynomial_project(f["E_without_H"], a)
        e0 = probe.get("mean_e0", algebra._alias(raw, "memory0", "e0"))
        e1 = probe.get("mean_e1", algebra._alias(raw, "memory1", "e1"))
        mean = algebra._memory_coefficients(raw, N, f["lam"], e0, e1)[:, :, None]
        for name, left, middle, right, side, factor, mass in (
            ("cross_parent_work_j", Bq, C, Gz, 1, det, False),
            ("cross_added_work_j", Ga, C, mean, 1, det, False),
            ("self_work_j", Ga, C, Gz, 2, det, False),
            ("stiffness_cross_bilinear_j", Bq, C, Ga, 1, det, False),
            ("stiffness_self_bilinear_j", Ga, C, Ga, 2, det, False),
            ("mass_cross_bilinear_kg_m2", Tq, algebra._identity(3), Ea, 1, mass_factor, True),
            ("mass_self_bilinear_kg_m2", Ea, algebra._identity(3), Ea, 2, mass_factor, True)):
            p = algebra.polynomial_bilinear(left, middle, right)
            if mass:
                p = algebra.polynomial_product(lam2_matrix, p)
            append(prefix, name, p[:, 0, 0], side, factor)
        append(prefix, "added_parent_force_n", algebra.polynomial_bilinear(f["Bhat"], C, Gz),
               1, det, squeeze=True)
        append(prefix, "enrichment_force_n", algebra.polynomial_bilinear(f["Ghat_without_H"], C, mean),
               1, det, squeeze=True)
        append(prefix, "enrichment_force_n", algebra.polynomial_bilinear(f["Ghat_without_H"], C, Gz),
               2, det, squeeze=True)
    return result


def _component_enclosures(polynomials, moment_data):
    result, represented, corrections = {}, {}, {}
    for name, terms in polynomials.items():
        base, delta = ZERO, (ZERO, ZERO)
        for side, p in terms:
            base += sum((a*b for a, b in zip(p, moment_data["base"][side])), ZERO)
            for coefficient, correction in zip(p, moment_data["endpoint_corrections"][side]):
                delta = _add(delta, _scale(correction, coefficient))
        represented[name] = base
        corrections[name] = delta
        result[name] = _add((base, base), delta)
    return result, represented, corrections


def true_source_integral_enclosures(raw, N, witness, *, added_basis=None, parent_basis=None,
                                     probes=(), binary_places=256):
    """Return all exact rational source-target intervals, volumes first."""
    moments = source_moment_enclosures(raw, N, witness, binary_places=binary_places)
    polynomials = source_component_polynomials(raw, N, witness, added_basis=added_basis,
                                              parent_basis=parent_basis, probes=probes)
    return _component_enclosures(polynomials, moments)[0]


def _encode_interval(value):
    return dict(lower=algebra._ratio(value[0]), upper=algebra._ratio(value[1]))


def _encode_moment_data(data):
    return dict(source_lower_xi=_encode_interval(data["source_lower_xi"]),
                source_upper_xi=_encode_interval(data["source_upper_xi"]),
                source_lambda_lower=_encode_interval(data["source_interval"]["low"]),
                source_lambda_upper=_encode_interval(data["source_interval"]["high"]),
                source_sqrt_halfwidth=_encode_interval(data["source_interval"]["sqrt"]),
                represented_lambda_mid=algebra._ratio(data["represented_lambda_mid"]),
                represented_lambda_halfwidth=algebra._ratio(data["represented_lambda_halfwidth"]),
                literal_gamma_phi=[algebra._ratio(q) for q in data["literal_gamma_phi"]],
                sqrt_enclosure_binary_places=data["binary_places"])


def build_true_source_integral_certificate(raw, N, witness, *, added_basis=None, parent_basis=None,
                                           probes=(), binary_places=256):
    """Create one canonical source truth certificate; claimed widening is refused."""
    options = dict(added_basis=added_basis, parent_basis=parent_basis, probes=probes)
    moment_data = source_moment_enclosures(raw, N, witness, binary_places=binary_places)
    polynomials = source_component_polynomials(raw, N, witness, **options)
    targets, represented, corrections = _component_enclosures(polynomials, moment_data)
    names = witness.get("component_names")
    values = algebra.decode_native_array(witness, "values")
    budgets = algebra.decode_native_array(witness, "budgets")
    _require(type(names) is list and len(names) == len(set(names)) and
             values.shape == budgets.shape == (len(names),) and list(targets) == list(VOLUME_NAMES)+names,
             "Saved source component schema differs")
    saved = dict(zip(names, zip(values, budgets)))
    components, failures = {}, []
    for name, target in targets.items():
        item = dict(true_source_integral_enclosure=_encode_interval(target),
                    endpoint_migration_integral_enclosure=_encode_interval(corrections[name]),
                    exact_literal_source_on_represented_interval=algebra._ratio(represented[name]))
        if name in saved:
            value, budget = saved[name]
            _require(budget >= 0, "Historical saved budget is negative")
            residual = _subtract((value, value), target)
            upper = max(map(abs, residual))
            sufficient = upper <= budget
            if not sufficient:
                failures.append(name)
            item.update(saved_native_minus_true_source_enclosure=_encode_interval(residual),
                        saved_native_true_source_absolute_error_upper=algebra._ratio(upper),
                        historical_saved_budget_sufficient_for_source_integral=sufficient)
        components[name] = item
    body = dict(format=FORMAT, status="PASS", input_binding_sha256=algebra._digest(
                    dict(raw=raw, N=N, witness=witness, binary_places=binary_places, **options)),
                component_names=list(targets), checked_component_count=len(targets),
                checked_saved_component_count=len(names), source_geometry=_encode_moment_data(moment_data),
                components=components, canonical_exact_source_integral_certificate=True,
                arbitrary_error_bound_inflation_accepted=False,
                source_endpoint_displacement_integrated=True,
                source_gamma_cut_in_endpoint_slivers_integrated=True,
                historical_saved_budget_source_integral_sufficiency_status="PASS" if not failures else "OPEN",
                historical_saved_budget_source_integral_sufficiency_failures=failures,
                historical_saved_bound_operand_authentication_certified=False,
                selected_line_uniform_forward_bound_certified=False,
                angular_node_uniform_error_certified=False, angular_accuracy_certified=False,
                general_contraction_accuracy_certified=False, physical_contraction_certified=False,
                physical_creation_admitted=False, physical_release_admitted=False,
                available_for_creation_funding=False, source_history_reinitialized=False,
                source_history_advanced=False, automated_goal_enabled=False)
    body["certificate_sha256"] = algebra._digest(body)
    return body


def verify_true_source_integral_certificate(raw, N, witness, certificate, *, added_basis=None,
                                            parent_basis=None, probes=(), binary_places=256):
    """Recompute canonical bounds, rejecting even consistently resealed widening."""
    _require(type(certificate) is dict, "Source integral certificate must be an object")
    expected = build_true_source_integral_certificate(raw, N, witness, added_basis=added_basis,
                    parent_basis=parent_basis, probes=probes, binary_places=binary_places)
    _require(certificate == expected, "Canonical source integral certificate differs")
    return dict(status="PASS", checked_component_count=expected["checked_component_count"],
                certificate_sha256=expected["certificate_sha256"],
                source_endpoint_displacement_integrated=True, angular_accuracy_certified=False,
                physical_contraction_certified=False, available_for_creation_funding=False)


def verify_runtime_enclosures(targets, runtime_lower, runtime_upper, component_names):
    """Check a forward enclosure against independently derived rational truth."""
    _require(list(targets) == list(component_names), "Runtime component schema differs")
    lower, upper = algebra._array(runtime_lower), algebra._array(runtime_upper)
    _require(lower.shape == upper.shape == (len(targets),), "Runtime enclosure shape differs")
    failures = []
    for name, lo, hi in zip(component_names, lower, upper):
        _require(lo <= hi, "Runtime interval is reversed")
        source = _interval(targets[name])
        if not (lo <= source[0] <= source[1] <= hi):
            failures.append(name)
    return dict(status="PASS" if not failures else "OPEN", checked_component_count=len(targets),
                failures=failures, source_endpoint_displacement_integrated=True,
                runtime_interval_contains_complete_independent_source_enclosure=not failures,
                runtime_forward_bound_operand_authentication_certified=False,
                scope="Selected source-target containment only; runtime construction authentication is separate.",
                angular_accuracy_certified=False, physical_contraction_certified=False,
                available_for_creation_funding=False)
