"""Independent whole first-entry envelope and material-ledger audit.

The standard library alone reconstructs the original matrix exponential
coefficients, complete envelope inequalities and every whole-area quadrature
cell.  No production certificate reader or material query is imported.  The
previous independent point audit supplies its separately written exact
matrix/Bernstein and Decimal Rodrigues kernels.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from decimal import Decimal, localcontext
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
import importlib.util
import json
from math import comb, factorial
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
KERNEL_PATH = ROOT / "analysis/geometric_native_full_domain_validation/independent_domain_audit.py"
_spec = importlib.util.spec_from_file_location("independent_native_domain_kernel", KERNEL_PATH)
kernel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kernel)
F = Fraction
ZERO, ONE = F(0), F(1)
encoded, digest, rational, interval = kernel.encoded, kernel.digest, kernel.rational, kernel.interval
fdto, idto = kernel.fdto, kernel.idto


def add(a, b):
    return a[0] + b[0], a[1] + b[1]


def neg(a):
    return -a[1], -a[0]


def sub(a, b):
    return add(a, neg(b))


def mul(a, b):
    values = tuple(x * y for x in a for y in b)
    return min(values), max(values)


def div(a, b):
    if b[0] <= 0 <= b[1]:
        raise ValueError("Independent whole-envelope ratio has an unresolved denominator")
    return mul(a, (ONE / b[1], ONE / b[0]))


def sqr(a):
    if a[0] <= 0 <= a[1]:
        return ZERO, max(a[0] ** 2, a[1] ** 2)
    return min(a[0] ** 2, a[1] ** 2), max(a[0] ** 2, a[1] ** 2)


def sqrt(a):
    return kernel._sqrt_rational(a[0], bits=1536)[0], kernel._sqrt_rational(a[1], bits=1536)[1]


def enclosing(saved, independent, name):
    low, high = interval(saved)
    if low > independent[0] or high < independent[1]:
        raise ValueError("Saved " + name + " fails the independent whole-domain enclosure")


def pder(values, degree=1):
    return tuple(value * F(factorial(k), factorial(k - degree))
                 for k, value in enumerate(values) if k >= degree)


def padd(a, b):
    return tuple((a[k] if k < len(a) else ZERO) + (b[k] if k < len(b) else ZERO)
                 for k in range(max(len(a), len(b))))


def pneg(a):
    return tuple(-x for x in a)


def psub(a, b):
    return padd(a, pneg(b))


def pmul(a, b):
    return kernel.polynomial_multiply(a, b)


@lru_cache(maxsize=16)
def make_series(donor, receiver, order):
    """Full-vector independent series of R_d^-1 R_r (-X)."""
    matrices = kernel.matrix_series(tuple(donor), tuple(receiver), order + 4)
    rows = tuple(tuple(-matrix[j][0] for j in range(3)) for matrix in matrices)
    c = tuple(-row[0] for row in rows)
    y = tuple(row[1] for row in rows)
    z = tuple(row[2] for row in rows)
    cp, yp, zp = pder(c), pder(y), pder(z)
    app = psub(pmul(yp, c), pmul(y, cp))
    bpp = psub(pmul(zp, c), pmul(z, cp))
    w = psub(pmul(y, zp), pmul(yp, z))
    g = psub(pmul(pder(bpp), app), pmul(bpp, pder(app)))
    rate = sum(map(abs, donor), ZERO) + sum(map(abs, receiver), ZERO)
    return dict(C=c[:order + 1], Y=y[:order + 1], Z=z[:order + 1],
                A1=app[:order + 1], B1=bpp[:order + 1],
                W=w[:order + 1], G=g[:order + 1], rate=rate, order=order)


def series_bounds(series, name, low, high, derivative=0, factor=0):
    """Independent exact Bernstein enclosure plus analytic exponential tail.

    Each C/Y/Z scalar has coefficient majorant L^n/n!.  A1/B1 and W
    contain two products with one differentiated factor; G contains eight
    products of four factors with three total derivatives.  These give the
    conservative majorants below without depending on a production bound.
    """
    rate, order = series["rate"], series["order"]
    values = series[name]
    if name in ("C", "Y", "Z"):
        mass, major_rate = ONE, rate
    elif name in ("A1", "B1", "W"):
        mass, major_rate = 2 * rate, 2 * rate
    elif name == "G":
        mass, major_rate = 8 * rate ** 3, 4 * rate
    else:
        raise ValueError("Unknown independent envelope expression")
    if any(values[k] for k in range(factor)):
        raise ValueError("Envelope expression lost its exact initial multiplicity")
    poly = pder(values[factor:], derivative)
    lo, hi = kernel.power_bounds(poly, low, high)
    tail = kernel.factor_tail(major_rate, mass, order, factor, high, derivative)
    return lo - tail, hi + tail


def endpoint_octant_areas(a, b, precision=256):
    """Independent analytic solid angles, preserving all transverse terms."""
    q = div(b, a)
    unit = (ONE, ONE)
    qa, aa, ba = sqrt(add(unit, sqr(q))), sqrt(add(unit, sqr(a))), sqrt(add(unit, sqr(b)))
    t5 = div(mul(b, q), mul(add(qa, unit), add(ba, unit)))
    t6 = div(a, mul(add(aa, unit), add(qa, q)))
    def atan(bounds):
        bounds = rounded(bounds, precision + 40)
        count = max(16, (precision + 80) // 12 + 1)
        return kernel._atan_rational(bounds[0], count)[0], kernel._atan_rational(bounds[1], count)[1]
    area5, area6 = mul((F(2), F(2)), atan(t5)), mul((F(2), F(2)), atan(t6))
    total = mul((F(2), F(2)), atan(sqrt(add(sqr(a), sqr(b)))))
    area7 = sub(sub(total, area5), area6)
    return {"octant:4": (ZERO, ZERO), "octant:5": area5,
            "octant:6": area6, "octant:7": area7}, total


def _self_seal(raw, seal):
    if raw.get(seal) != digest({key: value for key, value in raw.items() if key != seal}):
        raise ValueError("Independent audit rejected inconsistent " + seal)


def _source_hashes(profile_only=True):
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(ROOT.rglob("*.py"))
            if not any(part in {".venv", ".git", "__pycache__"} for part in path.parts)
            and (not profile_only or path.relative_to(ROOT).parts[0] in {"tectonics", "tests", "scripts", "analysis"})}


@dataclass(frozen=True)
class Jet:
    coefficients: tuple
    tails: tuple
    order: int
    precision: int = 256

    def __post_init__(self):
        centers, errors = [], []
        for coefficient in self.coefficients:
            low, high = rounded((coefficient, coefficient), self.precision + 24)
            center = (low + high) / 2
            centers.append(center)
            errors.append(abs(center - coefficient))
        tails = []
        for degree, tail in enumerate(self.tails):
            residual = sum((error * F(factorial(k), factorial(k - degree))
                            for k, error in enumerate(errors) if k >= degree), ZERO)
            tails.append(rounded((tail + residual, tail + residual), self.precision + 24)[1])
        object.__setattr__(self, "coefficients", tuple(centers))
        object.__setattr__(self, "tails", tuple(tails))

    def derivative(self):
        return Jet(pder(self.coefficients), self.tails[1:], self.order, self.precision)

    def __neg__(self):
        return Jet(pneg(self.coefficients), self.tails, self.order, self.precision)

    def __sub__(self, other):
        return Jet(psub(self.coefficients, other.coefficients),
                   tuple(x + y for x, y in zip(self.tails, other.tails)), self.order, self.precision)

    def __mul__(self, other):
        complete = pmul(self.coefficients, other.coefficients)
        tails = []
        for degree in range(min(len(self.tails), len(other.tails))):
            omitted = sum((abs(value) * F(factorial(k), factorial(k - degree))
                           for k, value in enumerate(complete)
                           if k > self.order and k >= degree), ZERO)
            residual = ZERO
            for left_degree in range(degree + 1):
                right_degree = degree - left_degree
                left_size = sum(map(abs, pder(self.coefficients, left_degree)), ZERO)
                right_size = sum(map(abs, pder(other.coefficients, right_degree)), ZERO)
                residual += comb(degree, left_degree) * (
                    self.tails[left_degree] * right_size + other.tails[right_degree] * left_size
                    + self.tails[left_degree] * other.tails[right_degree])
            tails.append(omitted + residual)
        return Jet(complete[:self.order + 1], tuple(tails), self.order, self.precision)

    def bounds(self, domain, derivative=0):
        value = (ZERO, ZERO)
        for coefficient in reversed(pder(self.coefficients, derivative)):
            value = add(mul(value, domain), (coefficient, coefficient))
        return value[0] - self.tails[derivative], value[1] + self.tails[derivative]


def rounded(value, precision):
    maximum = max(map(abs, value))
    if not maximum:
        return ZERO, ZERO
    exponent = maximum.numerator.bit_length() - maximum.denominator.bit_length()
    power = F(1 << exponent) if exponent >= 0 else F(1, 1 << -exponent)
    if power > maximum:
        exponent -= 1
    bits = (3322 * precision + 999) // 1000 + 8
    shift = bits - exponent
    scale = F(1 << shift) if shift >= 0 else F(1, 1 << -shift)
    low, high = value[0] * scale, value[1] * scale
    return F(low.numerator // low.denominator) / scale, F(-((-high.numerator) // high.denominator)) / scale


@lru_cache(maxsize=16)
def independent_jets(donor, receiver, order, precision=256):
    matrices = kernel.matrix_series(tuple(donor), tuple(receiver), order)
    rows = tuple(tuple(-matrix[j][0] for j in range(3)) for matrix in matrices)
    rate = sum(map(abs, donor), ZERO) + sum(map(abs, receiver), ZERO)
    tails = tuple(rate ** (order + 1) / factorial(order + 1 - degree)
                  / (1 - rate / (order + 2 - degree)) for degree in range(7))
    c = Jet(tuple(-row[0] for row in rows), tails, order, precision)
    y = Jet(tuple(row[1] for row in rows), tails, order, precision)
    z = Jet(tuple(row[2] for row in rows), tails, order, precision)
    ap = y.derivative() * c - y * c.derivative()
    bp = z.derivative() * c - z * c.derivative()
    curvature = bp.derivative() * ap - bp * ap.derivative()
    height = z * y.derivative() - y * z.derivative()
    area_numerator = -height * curvature
    return dict(C=c, Y=y, Z=z, A1=ap, B1=bp,
                curvature_numerator=curvature, height_numerator=height,
                area_integrand_numerator=area_numerator), rate


def _native_endpoint_identity(raw, parent, donor_motion, receiver_motion):
    """Authenticate inward support words of the exact original halfspace clip."""
    if raw is None:
        return
    source_keys = {kernel._plane_key(plane) for plane in parent["geometry"]["supports"]}
    endpoint_key = kernel._plane_key(dict(normal=[-1, 0, 0], rotations=[receiver_motion,
        dict(omega_rad_per_myr=[-x for x in donor_motion["omega_rad_per_myr"]],
             dt_myr=donor_motion["dt_myr"])], label=""))
    if raw.get("pose"):
        raise ValueError("Saved endpoint original material domain gained a pose")
    keys = {kernel._plane_key(plane) for plane in raw["supports"]}
    if endpoint_key not in keys or not keys <= source_keys | {endpoint_key}:
        raise ValueError("Saved endpoint original material clip changed its full inverse lineage")
    with localcontext() as context:
        context.prec = kernel.PRECISION
        normals, vertices, _ = kernel._polygon_world(raw)
        if len(vertices) != len(normals) or len(vertices) < 3:
            raise ValueError("Saved endpoint native cone omitted its positive area boundary")
        for vertex in vertices:
            if any(kernel.ddot(normal, vertex) < -(Decimal(10) ** -430) for normal in normals):
                raise ValueError("Saved endpoint cone has an independently inconsistent inward support")
        original = parent["geometry"]
        endpoint_plane = dict(normal=[-1, 0, 0], rotations=[receiver_motion,
            dict(omega_rad_per_myr=[-x for x in donor_motion["omega_rad_per_myr"]], dt_myr=donor_motion["dt_myr"])], label="")
        expected_planes = tuple(original["supports"]) + (endpoint_plane,)
        numerical_planes = tuple(kernel._plane_world(plane) for plane in expected_planes)
        candidates = list(original["vertices"])
        for support in original["supports"]:
            for sign in (-1, 1):
                candidates.append(dict(kind="cross", planes=[support, endpoint_plane], sign=sign,
                                       reference=[], rotations=[]))
        expected_rays = set()
        for candidate in candidates:
            vertex = kernel._ray_world(candidate)
            if all(kernel.ddot(normal, vertex) >= -(Decimal(10) ** -430) * max(map(abs, vertex))
                   for normal in numerical_planes):
                expected_rays.add(kernel._ray_key(candidate))
        if {kernel._ray_key(vertex) for vertex in raw["vertices"]} != expected_rays:
            raise ValueError("Saved endpoint clip lost an exact original/intersection ray generator")


def audit_sweep(raw, binding, journal):
    """Prove the entire saved owner envelope, area integral and endpoint image."""
    _self_seal(raw, "payload_sha256")
    if (raw.get("format") != "native-whole-selected-owner-first-entry-domains-1"
            or raw.get("model") != "complete_axis_hemisphere_endpoint_and_curved_owner_envelope"
            or (raw.get("donor_owner"), raw.get("receiver_owner")) != (1, 0)
            or encoded(raw["journal"]) != encoded(journal)):
        raise ValueError("Whole first-entry sweep changed its selected full original epoch")
    order, count = raw["order"], raw["integration_cells"]
    if (type(order) is not int or not 16 <= order <= 48
            or type(count) is not int or not 8 <= count <= 8192
            or type(raw["precision"]) is not int or not 96 <= raw["precision"] <= 1024):
        raise ValueError("Whole envelope proof omitted its explicit bounded numerical contract")
    parents, omega = binding["parents"], binding["omega"]
    for identity, parent in parents.items():
        expected_signs = tuple(1 if int(identity.split(":")[-1]) & (4 >> j) else -1 for j in range(3))
        expected_normals = {tuple(expected_signs[j] if i == j else 0 for i in range(3)) for j in range(3)}
        if ({tuple(plane["normal"]) for plane in parent["geometry"]["supports"]} != expected_normals
                or parent["geometry"]["pose"] or any(plane["rotations"] for plane in parent["geometry"]["supports"])
                or parent["parcel"]["plate"] != (1 if expected_signs[0] > 0 else 0)):
            raise ValueError("Whole owner proof omitted or counterfeited a primitive original octant")
    dt = rational(journal["dt_myr"])
    donor, receiver = (tuple(rational(value) * dt for value in omega[owner]) for owner in (1, 0))
    jets, rate = independent_jets(donor, receiver, order, raw["precision"])
    if rational(raw["normal_rate_bound"]) != rate or rate >= F(1, 4):
        raise ValueError("Whole first-entry normal series changed an original small full-vector trajectory")
    saved_jets = {row["name"]: row for row in raw["analytic_polynomials"]}
    if set(saved_jets) != set(jets) or len(saved_jets) != len(raw["analytic_polynomials"]):
        raise ValueError("Whole envelope analytic proof omitted a complete original expression")
    for name, jet in jets.items():
        row = saved_jets[name]
        if (tuple(map(rational, row["coefficients"])) != jet.coefficients
                or tuple(map(rational, row["derivative_tails"])) != jet.tails):
            raise ValueError("Whole envelope coefficient or analytic derivative tail was forged")
    whole = (ZERO, ONE)
    proof = raw["proof_bounds"]
    qm_window = interval(raw["envelope"]["maximum_ratio_time_fraction"])
    if not 0 < qm_window[0] < qm_window[1] < 1:
        raise ValueError("Whole envelope maximum root lost its complete strict interval")
    # The exact u^2 factor is retained rather than dividing a bounded remainder
    # by arbitrarily small time.  Its separate exponential majorant is valid
    # on the entire [0,1], including the original common interface at zero.
    hs = pneg(make_series(donor, receiver, order)["W"])
    if hs[:2] != (ZERO, ZERO):
        raise ValueError("Whole ratio-turning numerator lost exact original incidence")
    factored = hs[2:]
    def height_factor(domain, derivative=0):
        lo, hi = kernel.power_bounds(pder(factored, derivative), *domain)
        tail = kernel.factor_tail(2 * rate, 2 * rate, order, 2, domain[1], derivative)
        return lo - tail, hi + tail
    if (height_factor(whole, 1)[0] <= 0
            or height_factor((qm_window[0], qm_window[0]))[1] >= 0
            or height_factor((qm_window[1], qm_window[1]))[0] <= 0):
        raise ValueError("Saved root is not the unique complete maximum of the owner cone ratio")
    independent_proof = dict(c_positive=jets["C"].bounds(whole),
        a_derivative_numerator_positive=jets["A1"].bounds(whole),
        b_derivative_numerator_positive=jets["B1"].bounds(whole),
        curvature_derivative_negative=jets["curvature_numerator"].bounds(whole, 1),
        curvature_at_maximum_ratio_negative=jets["curvature_numerator"].bounds((qm_window[0], qm_window[0])),
        curvature_after_maximum_ratio_negative=jets["curvature_numerator"].bounds((qm_window[0], ONE)))
    original_series = make_series(donor, receiver, order)
    def factored_native(name, domain, derivative=0):
        values = original_series[name][1:]
        lo, hi = kernel.power_bounds(pder(values, derivative), *domain)
        remainder = rate ** (order + 1) / factorial(order + 1)
        remainder *= F(factorial(order), factorial(order - derivative))
        remainder /= 1 - rate / (order + 1 - derivative)
        return lo - remainder, hi + remainder
    def turning(domain, derivative=0):
        aa = factored_native("Y", domain)
        bb = neg(factored_native("Z", domain))
        if derivative:
            return sub(mul(aa, neg(factored_native("Z", domain, 2))),
                       mul(factored_native("Y", domain, 2), bb))
        return sub(mul(aa, neg(factored_native("Z", domain, 1))),
                   mul(factored_native("Y", domain, 1), bb))
    initial_ratio = div(factored_native("Z", (ZERO, ZERO)), factored_native("Y", (ZERO, ZERO)))
    exact_terminal_ratio = div(factored_native("Z", (ONE, ONE)), factored_native("Y", (ONE, ONE)))
    independent_proof.update(
        y_over_u_positive=factored_native("Y", whole),
        z_over_u_positive=factored_native("Z", whole),
        maximum_ratio_turning_derivative_positive=turning(whole, 1),
        maximum_ratio_initial_turning_negative=turning((ZERO, ZERO)),
        maximum_ratio_terminal_turning_positive=turning((ONE, ONE)),
        initial_ratio_above_terminal=sub(initial_ratio, exact_terminal_ratio))
    if set(proof) != set(independent_proof):
        raise ValueError("Whole envelope orientation proof lost a required inequality")
    for name, bounds in independent_proof.items():
        enclosing(proof[name], bounds, name)
        if ("positive" in name and bounds[0] <= 0) or ("negative" in name and bounds[1] >= 0):
            raise ValueError("Independent complete envelope orientation is unresolved")
    terminal = (ONE, ONE)
    qm = div(jets["Z"].bounds(qm_window), jets["Y"].bounds(qm_window))
    q1 = div(jets["Z"].bounds(terminal), jets["Y"].bounds(terminal))
    k1 = div(jets["B1"].bounds(terminal), jets["A1"].bounds(terminal))
    h1 = div(jets["height_numerator"].bounds(terminal), jets["A1"].bounds(terminal))
    a1 = div(jets["Y"].bounds(terminal), jets["C"].bounds(terminal))
    b1 = div(jets["Z"].bounds(terminal), jets["C"].bounds(terminal))
    for name, value in (("maximum_ratio_bounds", qm), ("terminal_ratio_bounds", q1),
                        ("terminal_slope_bounds", k1), ("terminal_height_bounds", h1)):
        enclosing(raw["envelope"][name], value, name)
    if not (qm[0] > q1[1] and q1[0] > k1[1] and k1[0] > 0) or h1[0] <= 0:
        raise ValueError("Whole curved owner envelope lacks its positive ordered branch")
    envelope = raw["envelope"]
    if (envelope["represented_as_great_circle_polygon"] is not False
            or envelope["negative_z_branch"] != "whole swept equals endpoint"
            or envelope["positive_y_positive_z_branch"] != "whole swept equals endpoint"
            or envelope["negative_y_negative_z_branch"] != "empty"
            or envelope["implicit_domain"] != "x>=0 and max_{0<u<=1}[-C(u)*x+Y(u)*y+Z(u)*z]>0"):
        raise ValueError("Whole implicit owner envelope was replaced by an endpoint-only claim")
    integral = (ZERO, ZERO)
    if len(raw["quadrature"]) != count:
        raise ValueError("Whole envelope area proof omitted a quadrature cell")
    start = qm_window[1]
    for index, cell in enumerate(raw["quadrature"]):
        expected = (start + (1 - start) * F(index, count), start + (1 - start) * F(index + 1, count))
        if interval(cell["time_fraction"]) != expected:
            raise ValueError("Whole envelope area quadrature has a gap or a repeated interval")
        ap = jets["A1"].bounds(expected)
        h = div(jets["height_numerator"].bounds(expected), ap)
        h = max(ZERO, h[0]), max(ZERO, h[1])
        minus_kprime = div(neg(jets["curvature_numerator"].bounds(expected)), mul(ap, ap))
        value = mul(h, minus_kprime)
        numerator = jets["area_integrand_numerator"]
        central = sum((coefficient * (expected[1] ** (k + 1) - expected[0] ** (k + 1)) / (k + 1)
                       for k, coefficient in enumerate(numerator.coefficients)), ZERO)
        tail = numerator.tails[0] * (expected[1] - expected[0])
        numerator_integral = max(ZERO, central - tail), max(ZERO, central + tail)
        amount = div(numerator_integral, mul(mul(ap, ap), ap))
        enclosing(cell["integrand_bounds"], value, "whole envelope integrand cell")
        enclosing(cell["integral_bounds"], amount, "whole envelope integral cell")
        integral = add(integral, amount)
    ap = jets["A1"].bounds(qm_window)
    initial_h = div(jets["height_numerator"].bounds(qm_window), ap)
    initial_kprime = div(neg(jets["curvature_numerator"].bounds(qm_window)), mul(ap, ap))
    initial = (ZERO, max(ZERO, initial_h[1]) * max(ZERO, initial_kprime[1]) * (qm_window[1] - qm_window[0]))
    total_integral = add(integral, initial)
    triangle = mul(mul(a1, sqr(sub(q1, k1))), (F(1, 2), F(1, 2)))
    excess = sub(total_integral, triangle)
    if excess[0] <= 0:
        raise ValueError("Independent whole excess measure is not strictly positive")
    density_base = (ONE, ONE + qm[1] ** 2 + h1[1] ** 2)
    density = div((ONE, ONE), mul(density_base, sqrt(density_base)))
    extra = mul(excess, density)
    integration = raw["integration"]
    for name, value in (("uncertain_initial_integral_bounds", initial),
        ("curved_planar_integral_bounds", total_integral), ("endpoint_planar_triangle_bounds", triangle),
        ("excess_planar_area_bounds", excess), ("excess_density_bounds", density),
        ("extra_swept_area_bounds_sr", extra)):
        enclosing(integration[name], value, name)
    endpoint_order = max(48, (raw["precision"] + 40) // 7 + 8)
    endpoint_matrices = kernel.matrix_series(donor, receiver, endpoint_order)
    endpoint_normal = tuple(sum((-matrix[j][0] for matrix in endpoint_matrices), ZERO) for j in range(3))
    endpoint_tail = rate ** (endpoint_order + 1) / factorial(endpoint_order + 1) / (1 - rate / (endpoint_order + 2))
    endpoint_c = rounded((-endpoint_normal[0] - endpoint_tail, -endpoint_normal[0] + endpoint_tail), raw["precision"] + 40)
    endpoint_a = rounded(div(rounded((endpoint_normal[1] - endpoint_tail, endpoint_normal[1] + endpoint_tail), raw["precision"] + 40), endpoint_c), raw["precision"] + 40)
    endpoint_b = rounded(div(rounded((endpoint_normal[2] - endpoint_tail, endpoint_normal[2] + endpoint_tail), raw["precision"] + 40), endpoint_c), raw["precision"] + 40)
    areas, endpoint_total = endpoint_octant_areas(endpoint_a, endpoint_b, raw["precision"])
    domains = {row["fragment_id"]: row for row in raw["donor_domains"]}
    if set(domains) != set(areas) or len(domains) != len(raw["donor_domains"]):
        raise ValueError("Whole owner archive omitted an original donor domain")
    total = (ZERO, ZERO)
    donor_motion = dict(omega_rad_per_myr=omega[1], dt_myr=journal["dt_myr"])
    receiver_motion = dict(omega_rad_per_myr=omega[0], dt_myr=journal["dt_myr"])
    for identity, endpoint_area in areas.items():
        row, parent = domains[identity], parents[identity]
        if encoded(row["source_geometry"]) != encoded(parent["geometry"]):
            raise ValueError("Whole donor area source is counterfeited")
        enclosing(row["parent_area_bounds_sr"], kernel._octant_area_bounds(), "whole original donor octant area")
        enclosing(row["endpoint_area_bounds_sr"], endpoint_area, "independent endpoint hemisphere clip solid angle")
        expected_extra = interval(integration["extra_swept_area_bounds_sr"]) if identity == "octant:5" else (ZERO, ZERO)
        if interval(row["extra_swept_area_bounds_sr"]) != expected_extra:
            raise ValueError("Whole owner domain omitted or duplicated its curved-only material")
        expected_case = "empty" if identity == "octant:4" else "endpoint_plus_curved_swept_extra" if identity == "octant:5" else "endpoint_equals_whole_swept"
        if row["domain_case"] != expected_case or (row["endpoint_domain"] is None) != (identity == "octant:4"):
            raise ValueError("Whole selected owner octant case was changed")
        _native_endpoint_identity(row["endpoint_domain"], parent, donor_motion, receiver_motion)
        enclosing(row["swept_area_bounds_sr"], add(interval(row["endpoint_area_bounds_sr"]), expected_extra), "whole selected donor swept area")
        total = add(total, interval(row["swept_area_bounds_sr"]))
    enclosing(raw["swept_area_bounds_sr"], total, "whole selected owner swept measure")
    if raw["extra_endpoint_fixed_parent_id"] != "octant:5":
        raise ValueError("Whole curved-only world vacancy was assigned to a counterfeit fixed parent")
    qm_saved = interval(envelope["maximum_ratio_bounds"])
    k_saved = interval(envelope["terminal_slope_bounds"])
    h_saved = interval(envelope["terminal_height_bounds"])
    chart = ((ZERO, h_saved[1]), (-qm_saved[1], -k_saved[0]), (ONE, ONE))
    matrix = kernel.matrix_series((ZERO, ZERO, ZERO), donor, endpoint_order)
    donor_rate = sum(map(abs, donor), ZERO)
    remainder = donor_rate ** (endpoint_order + 1) / factorial(endpoint_order + 1) / (1 - donor_rate / (endpoint_order + 2))
    moved = []
    for i in range(3):
        value = (ZERO, ZERO)
        for j in range(3):
            scalar = sum((term[i][j] for term in matrix), ZERO)
            value = add(value, mul((scalar - remainder, scalar + remainder), chart[j]))
        moved.append(value)
    support_rows = {tuple(row["normal"]): row for row in raw["extra_endpoint_parent_support_bounds"]}
    expected_normals = {tuple(plane["normal"]) for plane in parents["octant:5"]["geometry"]["supports"]}
    if set(support_rows) != expected_normals or len(support_rows) != len(raw["extra_endpoint_parent_support_bounds"]):
        raise ValueError("Whole swept-only world image lost an original fixed parent support")
    for normal, row in support_rows.items():
        value = (ZERO, ZERO)
        for coefficient, component in zip(normal, moved):
            value = add(value, mul((F(coefficient), F(coefficient)), component))
        enclosing(row["bounds"], value, "whole curved-only endpoint fixed parent support")
        if value[0] <= 0:
            raise ValueError("Complete curved-only world vacancy left its asserted original footprint")
    for flag in ("physical_direction_authenticated", "physical_archive_adopted", "native_force_step_accepted", "global_inventory_updated"):
        if raw[flag] is not False:
            raise ValueError("Whole geometry proof was promoted to physical force or inventory adoption")
    result = dict(whole_selected_owner_envelope_independently_proved=True,
        whole_curved_excess_measure_independently_enclosed=True,
        quadrature_cell_count=count, complete_donor_count=len(domains),
        unique_ratio_maximum_independently_proved=True,
        all_native_endpoint_clip_solid_angles_independently_proved=True,
        curved_only_endpoint_fixed_parent_id="octant:5",
        extra_swept_area_bounds_sr=integration["extra_swept_area_bounds_sr"],
        swept_area_bounds_sr=raw["swept_area_bounds_sr"],
        endpoint_owner_overlap_area_bounds_sr=idto(*rounded(endpoint_total, raw["precision"] + 30)),
        certificate_sha256=raw["payload_sha256"], source_journal_sha256=journal["journal_sha256"],
        donor_domain_area_bounds_sr={identity: row["swept_area_bounds_sr"] for identity, row in domains.items()},
        physical_force_or_global_inventory_adopted=False)
    _VERIFIED_SWEEPS[raw["payload_sha256"]] = (encoded(raw), encoded(journal), deepcopy(result))
    return result


_VERIFIED_SWEEPS = {}


def audit_source_binding(journal, original, initial, candidate, support, *, initial_checkpoint_sha256):
    """Authenticate the added full seed bindings against the real checkpoint."""
    _self_seal(journal, "journal_sha256")
    before = journal["memory_reference"]["before_payload"]
    expected_mechanics = initial["native_underthrust_motion"]["initial_mechanics"]
    if (encoded(before.get("original_mechanics_payload")) != encoded(expected_mechanics)
            or encoded(before.get("original_surface_payload")) != encoded(original)
            or before.get("original_force_checkpoint_sha256") != initial_checkpoint_sha256):
        raise ValueError("Material epoch bindings changed the authentic complete source or mechanical seed")
    core = deepcopy(journal)
    for name in ("original_mechanics_payload", "original_surface_payload", "original_force_checkpoint_sha256"):
        del core["memory_reference"]["before_payload"][name]
    core["memory_reference"]["before_sha256"] = digest(core["memory_reference"]["before_payload"])
    core["journal_sha256"] = digest({key: value for key, value in core.items() if key != "journal_sha256"})
    binding = kernel.audit_source_binding(core, original, initial, candidate, support)
    if (journal["memory_reference"]["before_sha256"] != digest(before)
            or journal["memory_reference"]["candidate_after_sha256"] != digest(journal["memory_reference"]["candidate_after_payload"])):
        raise ValueError("Full material source mechanical references no longer authenticate their immutable payloads")
    binding["original_force_checkpoint_file_sha256"] = initial_checkpoint_sha256
    return binding


def vdot(a, b):
    result = (ZERO, ZERO)
    for x, y in zip(a, b):
        result = add(result, mul(x, y))
    return result


def vcross(a, b):
    return tuple(sub(mul(a[j], b[k]), mul(a[k], b[j]))
                 for j, k in ((1, 2), (2, 0), (0, 1)))


@lru_cache(maxsize=32)
def rotation_matrix(axis, duration, precision=256):
    axis = tuple(value * duration for value in axis)
    rate = sum(map(abs, axis), ZERO)
    order = max(48, (precision + 40) // 7 + 8)
    matrices = kernel.matrix_series((ZERO, ZERO, ZERO), axis, order)
    tail = rate ** (order + 1) / factorial(order + 1) / (1 - rate / (order + 2))
    return tuple(tuple(rounded((sum((matrix[i][j] for matrix in matrices), ZERO) - tail,
        sum((matrix[i][j] for matrix in matrices), ZERO) + tail), precision + 40)
        for j in range(3)) for i in range(3))


def rotate_intervals(vector, history, precision=256):
    for step in history:
        matrix = rotation_matrix(tuple(rational(x) for x in step["omega_rad_per_myr"]),
                                 rational(step["dt_myr"]), precision)
        vector = tuple(rounded(vdot(row, vector), precision + 40) for row in matrix)
    return vector


def plane_intervals(raw, pose=(), precision=256):
    vector = tuple((value, value) for value in kernel.normalize_scale(raw["normal"]))
    return rotate_intervals(vector, tuple(raw["rotations"]) + tuple(pose), precision)


def ray_intervals(raw, pose=(), precision=256):
    if raw["kind"] == "seed":
        vector = tuple((value, value) for value in kernel.normalize_scale(raw["reference"]))
        return rotate_intervals(vector, tuple(raw["rotations"]) + tuple(pose), precision)
    if raw["kind"] != "cross" or len(raw["planes"]) != 2 or raw["sign"] not in (-1, 1):
        raise ValueError("Independent area ray lost its exact original generator")
    first, second = (plane_intervals(plane, pose, precision) for plane in raw["planes"])
    return tuple(mul((F(raw["sign"]), F(raw["sign"])), value) for value in vcross(first, second))


@lru_cache(maxsize=64)
def polygon_area_bounds(raw_bytes, precision=256):
    """Independent rigorous native-ray solid angle; no tiny-area cutoff."""
    raw = json.loads(raw_bytes)
    vertices = tuple(ray_intervals(vertex, raw["pose"], precision) for vertex in raw["vertices"])
    normalized = []
    for vector in vertices:
        norm_square = (ZERO, ZERO)
        for component in vector:
            norm_square = add(norm_square, sqr(component))
        norm = sqrt(norm_square)
        if norm[0] <= 0:
            raise ValueError("Whole raw vacancy ray norm was not independently resolved")
        normalized.append(tuple(rounded(div(component, norm), precision + 40) for component in vector))
    area = (ZERO, ZERO)
    for b, c in zip(normalized[1:-1], normalized[2:]):
        a = normalized[0]
        determinant = vdot(a, vcross(b, c))
        denominator = add(add(add((ONE, ONE), vdot(a, b)), vdot(b, c)), vdot(c, a))
        if determinant[0] <= 0 or denominator[0] <= 0:
            raise ValueError("Whole raw vacancy triangle lacks positive independent orientation")
        tangent = rounded(div(determinant, denominator), precision + 40)
        if not 0 < tangent[0] <= tangent[1] < F(1, 100):
            raise ValueError("Raw vacancy triangle exceeded this bounded small-area analytic audit")
        count = max(24, (precision + 80) // 4 + 1)
        angle = kernel._atan_rational(tangent[0], count)[0], kernel._atan_rational(tangent[1], count)[1]
        area = rounded(add(area, mul((F(2), F(2)), angle)), precision + 30)
    if area[0] <= 0:
        raise ValueError("Whole positive native vacancy area was not independently enclosed")
    return area


def audit_material(raw, sweep_receipt, binding, journal, initial, candidate):
    """Audit actual once-only symbolic allocations and hot material vacancies."""
    _self_seal(raw, "state_sha256")
    if (raw.get("format") != "native-full-first-entry-material-state-1"
            or raw.get("model") != "explicit_original_seed_direction_once_only_full_first_entry_hot_endpoint"
            or encoded(raw["journal"]) != encoded(journal)):
        raise ValueError("First-entry material law changed its complete original epoch")
    sweep = raw["sweep_certificate"]
    _self_seal(sweep, "payload_sha256")
    evidence = _VERIFIED_SWEEPS.get(sweep["payload_sha256"])
    if evidence is None or evidence[:2] != (encoded(sweep), encoded(journal)):
        expected_sweep_receipt = audit_sweep(sweep, binding, journal)
    else:
        expected_sweep_receipt = evidence[2]
    if encoded(sweep_receipt) != encoded(expected_sweep_receipt):
        raise ValueError("Material law supplied an unauthenticated independent whole-domain receipt")
    original = initial["native_underthrust_motion"]["initial_mechanics"]
    before = journal["memory_reference"]["before_payload"]
    checkpoint_sha = binding.get("original_force_checkpoint_file_sha256")
    if checkpoint_sha is None:
        # This existing checkpoint format has canonical compact JSON plus one
        # terminal newline.  CLI additionally compares the actual file bytes.
        checkpoint_sha = hashlib.sha256(encoded(initial) + b"\n").hexdigest()
    if (encoded(raw["original_mechanics_payload"]) != encoded(original)
            or raw["original_mechanics_sha256"] != digest(original)
            or encoded(before.get("original_mechanics_payload")) != encoded(original)
            or encoded(before.get("original_surface_payload")) != encoded(initial["current_surface"])
            or before.get("original_force_checkpoint_sha256") != checkpoint_sha
            or encoded(raw["memory_reference"]) != encoded(journal["memory_reference"])):
        raise ValueError("First-entry material law counterfeited original full mechanical or material evidence")
    parents = binding["parents"]
    trace_rows = []
    traces = original["fault_traces"]["traces"]
    if len(traces) != 4:
        raise ValueError("First-entry material direction lacks all four original physical weak planes")
    for trace in traces:
        if (trace["subducting_plate"], trace["overriding_plate"]) != (1, 0):
            raise ValueError("Original physical weak planes do not declare the selected donor direction")
        endpoints = tuple(tuple(rational(value) for value in trace[name]) for name in ("start", "end"))
        if kernel.cross(*endpoints) != (F(-1), ZERO, ZERO):
            raise ValueError("Original physical bank plane changed its strict orientation")
        matches = []
        for index in range(4):
            donor_parent, receiver_parent = parents["octant:" + str(index + 4)], parents["octant:" + str(index)]
            donor_rays = {tuple(rational(value) for value in vertex["reference"]) for vertex in donor_parent["geometry"]["vertices"]}
            receiver_rays = {tuple(rational(value) for value in vertex["reference"]) for vertex in receiver_parent["geometry"]["vertices"]}
            if frozenset(endpoints) == donor_rays & receiver_rays:
                matches.append((donor_parent, receiver_parent))
        if len(matches) != 1:
            raise ValueError("Original physical weak plane changed its exact original material banks")
        donor_parent, receiver_parent = matches[0]
        trace_rows.append(dict(trace_id=trace["trace_id"], donor_fragment_id=donor_parent["fragment_id"],
            receiver_fragment_id=receiver_parent["fragment_id"], donor_material_id=donor_parent["parcel"]["material_id"],
            receiver_material_id=receiver_parent["parcel"]["material_id"], subducting_plate=1, overriding_plate=0,
            trace_payload_sha256=digest(trace)))
    trace_rows.sort(key=lambda row: row["trace_id"])
    if encoded(raw["trace_bindings"]) != encoded(trace_rows):
        raise ValueError("Material direction changed authentic original physical-pair references")
    birth = raw["birth_parameters"]
    if (set(birth) != {"crust_thickness_km", "damage", "water", "provenance"}
            or any(type(birth[name]) is not float for name in ("crust_thickness_km", "damage", "water"))
            or not birth["crust_thickness_km"] > 0 or not 0 <= birth["damage"] <= 1
            or not 0 <= birth["water"] <= 1 or not isinstance(birth["provenance"], str) or not birth["provenance"].strip()
            or not isinstance(raw["provenance"], str) or not raw["provenance"].strip()):
        raise ValueError("Material endpoint birth lost its explicit canonical parameter law")
    if encoded(birth) != encoded(initial["native_underthrust_motion"]["birth_parameters"]):
        raise ValueError("Material endpoint hot fill changed the actual declared original source parameters")
    context = digest(dict(journal_sha256=journal["journal_sha256"], original_mechanics_sha256=digest(original),
        birth_parameters=birth, provenance=raw["provenance"], model=raw["model"]))
    domains = {row["fragment_id"]: row for row in sweep["donor_domains"]}
    allocations = {row["fragment_id"]: row for row in raw["parent_allocations"]}
    if set(allocations) != set(parents) or len(allocations) != len(raw["parent_allocations"]):
        raise ValueError("Once-only material ledger omitted an entire original parent")
    fields = ("area_km2", "oceanic_volume_km3", "cold_mantle_volume_km3", "density_excess_mass_kg")
    cohorts = []
    for identity, parent in parents.items():
        allocation, parcel = allocations[identity], parent["parcel"]
        archive_id = "native-first-entry-archive-domain:" + digest(dict(context=context, original_parent_id=identity))
        live_id = "native-first-entry-live-domain:" + digest(dict(archive_domain_id=archive_id, original_parent_id=identity))
        measure = "native_area(" + archive_id + ") / native_area(" + identity + ")"
        archived = (ZERO, ZERO) if identity not in domains else interval(domains[identity]["swept_area_bounds_sr"])
        parent_area = interval(allocation["parent_area_bounds_sr"])
        enclosing(allocation["parent_area_bounds_sr"], kernel._octant_area_bounds(), "whole original material parent area")
        ratio = div(archived, parent_area)
        if (allocation["material_id"] != parcel["material_id"] or allocation["owner"] != parcel["plate"]
                or allocation["parent_payload_sha256"] != digest(parent)
                or allocation["archive_domain_id"] != archive_id or allocation["live_domain_id"] != live_id
                or allocation["measure_reference"] != measure or interval(allocation["archived_area_bounds_sr"]) != archived
                or interval(allocation["archive_ratio_bounds"]) != ratio
                or rational(allocation["original_age_myr"]) != rational(parcel["age_myr"])
                or rational(allocation["material_age_at_endpoint_myr"]) != rational(parcel["age_myr"]) + rational(journal["dt_myr"])
                or encoded(allocation["material_fields"]) != encoded(parcel["material_fields"])
                or encoded(allocation["specific_properties"]) != encoded(parcel["specific_properties"])
                or allocation["receiver_internal_boundary_allocates_new_cohort"] is not False
                or allocation["exact_partition_identity"] != "live_domain disjoint_union archive_domain = original_parent"):
            raise ValueError("Once-only original material partition changed its source, domain symbol or age")
        if not 0 <= ratio[0] <= ratio[1] < 1:
            raise ValueError("Whole archived fraction is not a proper original material subset")
        rows = {row["reservoir"]: row for row in allocation["reservoirs"]}
        if set(rows) != set(fields) or len(rows) != len(allocation["reservoirs"]):
            raise ValueError("Once-only material ledger omitted an original extensive reservoir")
        for field, row in rows.items():
            amount = rational(parcel[field])
            if (rational(row["parent_amount"]) != amount
                    or interval(row["archived_amount_bounds"]) != mul(ratio, (amount, amount))
                    or interval(row["live_amount_bounds"]) != mul(sub((ONE, ONE), ratio), (amount, amount))
                    or row["measure_reference"] != measure
                    or row["archived_expression"] != "original_parent_amount * " + measure
                    or row["live_expression"] != "original_parent_amount - archived_amount"
                    or row["conservation_identity"] != "live_amount + archived_amount = original_parent_amount"
                    or row["interval_midpoint_used_as_debit"] is not False):
                raise ValueError("Original reservoir conservation changed authentic amounts or used inconsistent geometric measures")
        cohort_id = None
        if archived[1] > 0:
            if archived[0] <= 0:
                raise ValueError("A complete original material cohort lacks positive proved area")
            cohort_id = "native-first-entry-cohort:" + digest(dict(context=context, original_parent_id=identity, archive_domain_id=archive_id))
            cohorts.append(dict(cohort_id=cohort_id, original_parent_id=identity, material_id=parcel["material_id"],
                owner=parcel["plate"], receiver_owner=0, archive_domain_id=archive_id,
                material_time_function="first positive owner entry of each original point",
                receiver_ancestry_function="all original receiving fragments incident at that first entry",
                archive_law="once_only_passive_archive; exit and reentry never allocate twice"))
        if allocation["cohort_id"] != cohort_id:
            raise ValueError("Original material was duplicated or detached from its once-only cohort")
    if encoded(raw["archive_cohorts"]) != encoded(cohorts):
        raise ValueError("Once-only material archive invented or omitted a complete parent cohort")
    vacancies = {row["fixed_original_parent_id"]: row for row in raw["endpoint_vacancies"]}
    if set(vacancies) != set(parents) or len(vacancies) != len(raw["endpoint_vacancies"]):
        raise ValueError("Endpoint material closure omitted a fixed original world domain")
    endpoint = rational(journal["source_surface"]["time_myr"]) + rational(journal["dt_myr"])
    radius = rational(journal["source_surface"]["radius_km"])
    total_vacancy, total_raw, birth_ids, birth_count = (ZERO, ZERO), (ZERO, ZERO), set(), 0
    for identity, vacancy in vacancies.items():
        parent, parcel = parents[identity], parents[identity]["parcel"]
        raw_gaps = [row for row in journal["raw_vacancies"] if row["fixed_original_parent_id"] == identity]
        extra_domains = [row for row in domains.values() if interval(row["extra_swept_area_bounds_sr"])[1] > 0
                         and sweep["extra_endpoint_fixed_parent_id"] == identity]
        area = (ZERO, ZERO)
        for row in raw_gaps:
            enclosing(row["area_bounds_sr"], polygon_area_bounds(encoded(row["geometry"])),
                      "complete original raw vacancy solid angle")
            area = add(area, interval(row["area_bounds_sr"]))
            total_raw = add(total_raw, interval(row["area_bounds_sr"]))
        for row in extra_domains:
            area = add(area, interval(row["extra_swept_area_bounds_sr"]))
        vacancy_id = "native-first-entry-actual-vacancy:" + digest(dict(context=context, fixed_original_parent_id=identity))
        measure = "native_area(" + vacancy_id + ")"
        if (vacancy["vacancy_domain_id"] != vacancy_id or vacancy["measure_reference"] != measure
                or interval(vacancy["area_bounds_sr"]) != area
                or vacancy["raw_vacancy_ids"] != [row["vacancy_sha256"] for row in raw_gaps]
                or vacancy["curved_extra_donor_ids"] != [row["fragment_id"] for row in extra_domains]
                or vacancy["domain"] != "fixed_original_parent minus union_of_ALL_moved_live_original_domains"
                or vacancy["exact_disjoint_decomposition"] != "raw_geometric_gap disjoint_union moved_archived_swept_only_extra"
                or vacancy["native_polygon_proxy_used"] is not False):
            raise ValueError("Actual material vacancy omitted or duplicated once-archived transient material")
        newborn = vacancy["newborn"]
        if area[1] == 0:
            if newborn is not None:
                raise ValueError("Endpoint birth assigned material to an empty world vacancy")
        else:
            if area[0] <= 0 or newborn is None:
                raise ValueError("A positive actual material vacancy lost its complete hot fill")
            identity_new = "native-first-entry-hot-birth:" + digest(dict(context=context, vacancy_domain_id=vacancy_id))
            actual_area = mul(area, (radius ** 2, radius ** 2))
            volume = mul(actual_area, (rational(birth["crust_thickness_km"]), rational(birth["crust_thickness_km"])))
            if (newborn["material_id"] != identity_new or identity_new in birth_ids
                    or identity_new in journal["source_surface"]["known_material_ids"]
                    or newborn["fixed_original_parent_id"] != identity or newborn["owner"] != parcel["plate"]
                    or newborn["cell"] != parcel["cell"] or newborn["vacancy_domain_id"] != vacancy_id
                    or rational(newborn["birth_time_myr"]) != endpoint
                    or interval(newborn["area_km2_bounds"]) != actual_area
                    or interval(newborn["oceanic_volume_km3_bounds"]) != volume
                    or any(rational(newborn[field]) != 0 for field in ("cold_mantle_volume_km3", "density_excess_mass_kg", "age_myr"))
                    or encoded(newborn["material_fields"]) != encoded([["damage", birth["damage"]], ["water", birth["water"]]])
                    or tuple(map(rational, newborn["specific_properties"])) != (rational(birth["crust_thickness_km"]), ZERO, ZERO)
                    or newborn["area_expression"] != measure + " * radius_km^2"
                    or newborn["oceanic_volume_expression"] != "area_km2 * declared_hot_crust_thickness_km"
                    or newborn["birth_law"] != "hot_endpoint_fill_of_actual_material_vacancy"
                    or newborn["continuous_birth_or_newborn_motion_defined"] is not False
                    or newborn["newborn_physical_pair_admission_authorized"] is not False):
                raise ValueError("Endpoint hot birth changed its actual vacancy, identity, source law or clock")
            birth_ids.add(identity_new)
            birth_count += 1
        total_vacancy = add(total_vacancy, area)
    endpoint_lens = interval(sweep_receipt["endpoint_owner_overlap_area_bounds_sr"])
    enclosing(idto(*total_raw), endpoint_lens, "complete raw hemisphere-gap measure")
    enclosing(idto(*total_vacancy), add(endpoint_lens,
        interval(sweep["integration"]["extra_swept_area_bounds_sr"])),
        "complete actual material vacancy equals selected archived swept measure")
    for flag in ("material_closure_applied", "material_law_declared", "original_seed_direction_used", "source_intensives_and_memory_preserved"):
        if raw[flag] is not True:
            raise ValueError("Full original first-entry material closure was left conditional")
    for flag in ("seed_direction_is_physical_force_admission", "physical_pair_admission_authorized", "candidate_mechanics_adopted",
                 "endpoint_polygon_registry_compatible", "mechanical_work_assigned_to_material_volume"):
        if raw[flag] is not False:
            raise ValueError("Material closure was misrepresented as constitutive, work or force acceptance")
    if raw["native_force_intervals_accepted"] != 0:
        raise ValueError("Material first-entry closure adopted an unaccepted native force interval")
    return dict(full_original_parent_count=len(allocations), once_only_archive_cohort_count=len(cohorts),
        extensive_reservoir_count=len(fields), original_extensive_reservoir_count=len(allocations) * len(fields),
        all_original_reservoir_complements_authenticated=True,
        common_exact_geometric_measure_symbols_preserved=True, endpoint_vacancy_parent_count=len(vacancies),
        newborn_material_count=birth_count, whole_curved_only_actual_vacancy_retained=True,
        original_seed_trace_direction_authenticated=True, endpoint_hot_birth_law_applied=True,
        all_raw_vacancy_solid_angles_independently_enclosed=True,
        geometric_vacancy_equals_archived_swept_measure=True,
        carried_float_source_area_budget_equals_native_hot_area_claimed=False,
        accepted_loaded_native_intervals=0, candidate_memory_adopted=False,
        actual_vacancy_area_bounds_sr=idto(*total_vacancy), source_journal_sha256=journal["journal_sha256"],
        material_state_sha256=raw["state_sha256"])


def audit(control):
    started = time.perf_counter()
    control = Path(control)
    before_sources = _source_hashes()
    before_all_sources = _source_hashes(profile_only=False)
    def read(name):
        return json.loads((control / name).read_bytes())
    original, journal = read("original_source_surface.json"), read("full_domain_journal.json")
    initial, candidate, support = (read("original_force/" + name + ".json")
        for name in ("initial", "force_candidate", "constitutive_time_support"))
    initial_file = control / "original_force/initial.json"
    checkpoint_sha = hashlib.sha256(initial_file.read_bytes()).hexdigest()
    if initial_file.read_bytes() != (control / "original_force/after_refusal.json").read_bytes():
        raise ValueError("First-entry material control changed the refused physical checkpoint")
    binding = audit_source_binding(journal, original, initial, candidate, support,
                                   initial_checkpoint_sha256=checkpoint_sha)
    geometry = kernel.audit_endpoint_geometry(journal, binding, candidate)
    sweep_raw, material_raw = read("first_entry_sweep.json"), read("material_state.json")
    sweep = audit_sweep(sweep_raw, binding, journal)
    material = audit_material(material_raw, sweep, binding, journal, initial, candidate)
    helper_path = Path(__file__).with_name("independent_first_entry_point_audit.py")
    spec = importlib.util.spec_from_file_location("independent_native_first_entry_point_audit", helper_path)
    point_audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(point_audit)
    points = point_audit.audit_first_entry_point_artifacts(control, journal, sweep_raw, material_raw, binding)
    report = read("report.json")
    _self_seal(report, "report_sha256")
    for name in ("status", "original_force_re_solved", "original_candidate_all_fields_identical",
                 "whole_selected_owner_swept_domain_certified", "full_curved_extra_integrated",
                 "complete_standalone_material_closure_applied", "actual_material_vacancies_computed",
                 "endpoint_hot_fill_applied", "original_live_archive_reservoirs_symbolically_conserved",
                 "original_seed_direction_authenticated", "original_mechanical_work_preserved"):
        if report[name] != ("PASS" if name == "status" else True):
            raise ValueError("First-entry control report misstates its verified material closure")
    for name in ("candidate_mechanics_adopted", "physical_pair_admission_authorized", "global_native_inventory_updated"):
        if report[name] is not False:
            raise ValueError("First-entry report adopted refused physical mechanics or a global native inventory")
    for name in ("accepted_loaded_native_intervals", "accepted_source_work_j"):
        if report[name] != 0:
            raise ValueError("First-entry report adopted a native force step or assigned its work to material")
    expected_counts = dict(source_count=len(before_sources), source_parent_count=8,
        donor_domain_count=sweep["complete_donor_count"],
        positive_archive_cohort_count=material["once_only_archive_cohort_count"],
        endpoint_vacancy_domain_count=material["endpoint_vacancy_parent_count"],
        endpoint_newborn_count=material["newborn_material_count"],
        point_certificate_count=points["point_certificate_count"],
        material_query_count=points["material_point_query_count"],
        endpoint_query_count=points["endpoint_material_query_count"])
    if any(type(report[name]) is not int or report[name] != value for name, value in expected_counts.items()):
        raise ValueError("First-entry control report changed its independently counted complete material evidence")
    for relative, expected_sha in report["artifacts_sha256"].items():
        if hashlib.sha256((control / relative).read_bytes()).hexdigest() != expected_sha:
            raise ValueError("First-entry artifact changed after the immutable control receipt")
    for first, second in report["byte_pairs"]:
        if (control / first).read_bytes() != (control / second).read_bytes():
            raise ValueError("First-entry source, ledger or queries changed during restart replay")
    saved_before, saved_after = read("sources_before.json"), read("sources_after.json")
    after_sources = _source_hashes()
    if (before_sources != after_sources or before_sources != saved_before or saved_before != saved_after
            or before_all_sources != _source_hashes(profile_only=False)):
        raise ValueError("Independent first-entry source code changed after or during the control")
    artifacts = {str(path.relative_to(control)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted(control.rglob("*.json"))}
    return dict(format="independent-native-full-first-entry-material-audit-1", status="PASS",
        elapsed_seconds=time.perf_counter() - started, source_count=len(before_sources),
        auditor_all_python_source_count=len(before_all_sources),
        auditor_source_unchanged=True, all_python_sources_unchanged=True,
        full_original_owner_vectors_preserved=True, all_original_parcel_payloads_authenticated=True,
        original_full_mechanical_seed_and_checkpoint_authenticated=True,
        selected_owner_first_entry_domain=sweep, actual_standalone_material_closure=material,
        executable_material_points=points, native_endpoint_geometry=geometry,
        complete_selected_owner_swept_area_integrated=True,
        actual_material_vacancies_computed=True, endpoint_hot_birth_law_applied=True,
        all_original_reservoirs_symbolically_conserved=True,
        independent_force_solution_recomputed=False, force_acceptance_inferred_from_seed_direction=False,
        accepted_loaded_native_intervals=0, accepted_source_work_j=0,
        candidate_memory_adopted=False, physical_pair_admission_authorized=False,
        global_native_inventory_updated=False,
        continuous_birth_law_defined=False, newborn_material_motion_defined=False,
        input_artifact_sha256=artifacts,
        auditor_sha256=before_sources[str(Path(__file__).relative_to(ROOT))],
        independent_kernel_sha256=before_sources[str(KERNEL_PATH.relative_to(ROOT))],
        independent_point_auditor_sha256=before_sources[str(helper_path.relative_to(ROOT))],
        limitation="Whole swept-domain area and once-only material closure are proved for the complete original selected owner 1 into owner 0 in this bounded constant full-vector epoch. General endpoint spherical Boolean partitions are independently checked numerically at 500 digits; every saved raw vacancy polygon area has a separate rigorous native-ray solid-angle enclosure. Original seed direction does not authenticate force/work or a later physical material pair. No continuous hot birth, newborn motion, full slab evolution or force admission is inferred.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Independent first-entry receipt requires a fresh output path")
    result = audit(args.control)
    result["receipt_sha256"] = digest(result)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "elapsed_seconds")}, indent=2))
