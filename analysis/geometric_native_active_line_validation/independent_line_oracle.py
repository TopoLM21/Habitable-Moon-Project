"""Independent exact matrix-series and Decimal rotation cross-checks.

No tectonics modules are imported.  Rational Taylor bounds cover every clock
in [0,1]; optional Decimal observations check a particular clock only.
"""
from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction as F
from math import factorial


ZERO = ((F(0), F(0), F(0)),)*3
IDENTITY = tuple(tuple(F(i == j) for j in range(3)) for i in range(3))


def add(first, second):
    return tuple(tuple(a+b for a, b in zip(row_a, row_b))
        for row_a, row_b in zip(first, second))


def scale(matrix, factor):
    return tuple(tuple(value*factor for value in row) for row in matrix)


def multiply(first, second):
    return tuple(tuple(sum((first[i][k]*second[k][j] for k in range(3)), F(0))
        for j in range(3)) for i in range(3))


def vector(matrix, value):
    return tuple(sum((row[j]*value[j] for j in range(3)), F(0)) for row in matrix)


def cross_matrix(axis):
    x, y, z = axis
    return ((F(0), -z, y), (z, F(0), -x), (-y, x, F(0)))


def axes_from_source(source):
    duration = F.from_float(source["literal_dt_myr"])
    records = source["actual_candidate"]["omega_rad_per_myr"]
    axes = {owner: tuple(F.from_float(value)*duration for value in omega)
        for owner, omega in records}
    banks = source["material_state"]["trace_bindings"]
    owner_pairs = {(row["subducting_plate"], row["overriding_plate"]) for row in banks}
    if len(owner_pairs) != 1:
        raise ValueError("Independent original line oracle requires its saved single owner pair")
    donor, receiver = owner_pairs.pop()
    return axes[donor], axes[receiver]


def exponential_coefficients(matrix, degree):
    result = [IDENTITY]
    for n in range(1, degree+1):
        result.append(scale(multiply(result[-1], matrix), F(1, n)))
    return tuple(result)


def joint_pullback_coefficients(source, degree=18):
    """M(u)=exp(-u[ad]x) exp(u[ar]x), coefficients in one common clock."""
    donor, receiver = axes_from_source(source)
    first = exponential_coefficients(cross_matrix(tuple(-x for x in donor)), degree)
    second = exponential_coefficients(cross_matrix(receiver), degree)
    result = tuple(sum_matrices(multiply(first[k], second[n-k]) for k in range(n+1))
        for n in range(degree+1))
    norm = sum(map(abs, donor), F(0))+sum(map(abs, receiver), F(0))
    if norm > F(1, 2):
        raise ValueError("The original tiny-step geometric tail bound needs norm <=1/2")
    return result, norm


def sum_matrices(values):
    result = ZERO
    for item in values:
        result = add(result, item)
    return result


def derivative_tail(norm, degree, derivative=0):
    if not 0 <= derivative <= degree:
        raise ValueError("A derivative tail needs 0 <= derivative <= degree")
    return 2*norm**(degree+1)/factorial(degree+1-derivative)


def polynomial_bounds(coefficients, derivative=0):
    """Power-basis interval bound on the entire original clock [0,1]."""
    values = tuple(value*F(factorial(n), factorial(n-derivative))
        for n, value in enumerate(coefficients) if n >= derivative)
    if not values:
        return F(0), F(0)
    constant, *rest = values
    return constant+sum((min(F(0), x) for x in rest), F(0)), \
        constant+sum((max(F(0), x) for x in rest), F(0))


def polynomial_at(coefficients, clock, derivative=0):
    rows = tuple(value*F(factorial(n), factorial(n-derivative))
        for n, value in enumerate(coefficients) if n >= derivative)
    result = F(0)
    for value in reversed(rows):
        result = result*clock+value
    return result


def polynomial_product(first, second):
    result = [F(0)]*(len(first)+len(second)-1)
    for i, a in enumerate(first):
        for j, b in enumerate(second):
            result[i+j] += a*b
    return tuple(result)


def polynomial_derivative(coefficients):
    return tuple(n*value for n, value in enumerate(coefficients) if n)


def turning_numerator_coefficients(source, axis, degree=18):
    rows, norm = joint_pullback_coefficients(source, degree)
    seam = 3-axis
    a = tuple(row[seam][seam] for row in rows)
    b = tuple(row[seam][axis] for row in rows)
    first = polynomial_product(polynomial_derivative(b), a)
    second = polynomial_product(b, polynomial_derivative(a))
    result = tuple(x-y for x, y in zip(first, second))
    # Each exact matrix entry <=exp(s)<2 and each derivative <=s exp(s)<1;
    # the two product errors use both truncated endpoint and derivative tails.
    tail = 16*norm**(degree+1)/factorial(degree)
    return result, tail


def whole_clock_matrix_bounds(source, degree=18, derivative=0):
    coefficients, norm = joint_pullback_coefficients(source, degree)
    tail = derivative_tail(norm, degree, derivative)
    return tuple(tuple((lo-tail, hi+tail)
        for lo, hi in (polynomial_bounds(tuple(row[i][j] for row in coefficients), derivative)
            for j in range(3))) for i in range(3))


def initial_depth_divided_clock_bounds(source, degree=18):
    """Bound qX(u)/u on all positive clocks, retaining exact qX(0)=0."""
    coefficients, norm = joint_pullback_coefficients(source, degree)
    tail = derivative_tail(norm, degree)
    result = []
    for axis in (1, 2):
        if coefficients[0][0][axis] != 0:
            raise ValueError("Original receiving line must be on the exact zero-clock interface")
        lo, hi = polynomial_bounds(tuple(row[0][axis] for row in coefficients[1:]))
        result.append((lo-tail, hi+tail))
    return tuple(result)


def fresh_depth_whole_two_clock_bounds(source, degree=10):
    """Bound the fresh-owner derivative for every event u and owner clock v.

    F(u,v)=-eX^T exp(-v Ar) (Ad-Ar) exp(v Ad) M(u).
    This is a total-degree bivariate Taylor enclosure, not observed clocks.
    """
    donor, receiver = axes_from_source(source)
    ad, ar = cross_matrix(donor), cross_matrix(receiver)
    delta = add(ad, scale(ar, F(-1)))
    left = exponential_coefficients(scale(ar, F(-1)), degree)
    right = exponential_coefficients(ad, degree)
    pullback, norm = joint_pullback_coefficients(source, degree)
    if 2*norm > F(1, 2):
        raise ValueError("The independent two-clock tail needs 2*norm<=1/2")
    fresh_v = tuple(scale(sum_matrices(multiply(multiply(left[k], delta), right[n-k])
        for k in range(n+1)), F(-1)) for n in range(degree+1))
    delta_norm = sum((abs(a-b) for a, b in zip(donor, receiver)), F(0))
    tail = 2*delta_norm*(2*norm)**(degree+1)/factorial(degree+1)
    result = []
    for axis in (1, 2):
        constant = fresh_v[0][0][axis]
        rest = [multiply(fresh_v[j], pullback[i])[0][axis]
            for i in range(degree+1) for j in range(degree+1-i) if i+j]
        result.append((constant+sum((min(F(0), x) for x in rest), F(0))-tail,
            constant+sum((max(F(0), x) for x in rest), F(0))+tail))
    return tuple(result)


def quarter_columns(source):
    traces = source["contact_time_path"]["initial_mechanics"]["fault_traces"]["traces"]
    by_id = {row["trace_id"]: row for row in traces}
    return tuple((bank["receiver_fragment_id"], bank["trace_id"],
        tuple(F(value) for value in by_id[bank["trace_id"]]["start"]),
        tuple(F(value) for value in by_id[bank["trace_id"]]["end"]))
        for bank in source["material_state"]["trace_bindings"])


def decimal(value):
    value = F(value)
    return Decimal(value.numerator)/Decimal(value.denominator)


def cross(first, second):
    return tuple(first[j]*second[k]-first[k]*second[j]
        for j, k in ((1, 2), (2, 0), (0, 1)))


def dot(first, second):
    return sum((a*b for a, b in zip(first, second)), Decimal(0))


def sine_cosine(angle, epsilon=Decimal("1e-210")):
    sine, cosine = angle, Decimal(1)
    first, second = angle, Decimal(1)
    for n in range(1, 2000):
        first *= -angle*angle/Decimal((2*n)*(2*n+1))
        second *= -angle*angle/Decimal((2*n-1)*(2*n))
        sine += first
        cosine += second
        if max(abs(first), abs(second)) < epsilon:
            return sine, cosine
    raise AssertionError("Independent Decimal rotation series did not converge")


def rotate(point, axis, clock):
    size = dot(axis, axis).sqrt()
    if not size:
        return point
    sine, cosine = sine_cosine(size*clock)
    first = cross(axis, point)
    second = cross(axis, first)
    return tuple(p+sine/size*a+(1-cosine)/size**2*b
        for p, a, b in zip(point, first, second))


def decimal_pullback(source, exact_point, clock):
    with localcontext() as context:
        context.prec = 240
        donor, receiver = (tuple(map(decimal, axis)) for axis in axes_from_source(source))
        point = tuple(map(decimal, exact_point))
        return rotate(rotate(point, receiver, decimal(clock)), donor, -decimal(clock))


def pi_decimal():
    def atan_small(value):
        term, total = value, value
        for n in range(1, 2000):
            term *= -value*value
            addition = term/Decimal(2*n+1)
            total += addition
            if abs(addition) < Decimal("1e-220"):
                return total
        raise AssertionError("Independent Machin pi series did not converge")
    return 16*atan_small(Decimal(1)/5)-4*atan_small(Decimal(1)/239)


def original_measure_and_coefficients(source, trace_id):
    traces = source["contact_time_path"]["initial_mechanics"]["fault_traces"]["traces"]
    trace = next(row for row in traces if row["trace_id"] == trace_id)
    node = next(row for row in source["quadrature_histories"] if row["trace_id"] == trace_id)
    radius = F.from_float(source["origin_surface"]["radius_km"])*1000
    coefficient = radius*F.from_float(trace["contact_height_m"])/F.from_float(node["original_frame"][0])
    columns = trace["columns"]
    kb_float = 1/(columns["sub_length_m"]/columns["sub_normal_modulus_pa"]
        +columns["over_length_m"]/columns["over_normal_modulus_pa"])
    kc_float = trace["law"]["normal_stiffness_pa_m"]
    kb, kc = F.from_float(kb_float), F.from_float(kc_float)
    kn = F.from_float(1/(1/kb_float+1/kc_float))
    return coefficient, (kn, kb*(kn/kc-1)**2, kn*kn/kc)


def independent_frozen_potential_integral(source, trace_id, low_computational, high_computational, extent):
    """Independent real-harmonic integral for the original frozen material.

    This is a numerical cross-check of the unchanged core or a proved
    negative-entry strip; it does not choose a path in a switched strip.
    """
    with localcontext() as context:
        context.prec = 240
        donor, receiver = (tuple(map(decimal, axis)) for axis in axes_from_source(source))
        trace = next(row for row in source["contact_time_path"]["initial_mechanics"]["fault_traces"]["traces"]
            if row["trace_id"] == trace_id)
        node = next(row for row in source["quadrature_histories"] if row["trace_id"] == trace_id)
        start, end = (tuple(decimal(F.from_float(value)) for value in trace[name])
            for name in ("start", "end"))
        horizontal = tuple(decimal(F.from_float(value)) for value in node["original_point"]["horizontal"])
        sine, cosine = (decimal(F.from_float(value)) for value in node["original_frame"])
        radius = decimal(F.from_float(source["origin_surface"]["radius_km"]))*1000
        old = next(row for row in source["accepted_before"]["snapshot"]["transverse_displacement_m"]
            if row[0] == trace_id)
        new = next(row for row in source["actual_candidate"]["transverse_displacement_m"]
            if row[0] == trace_id)
        ws0, wo0, ws1, wo1 = (decimal(F.from_float(value)) for value in (*old[1:], *new[1:]))
        fields = next(row for row in source["contact_time_path"]["previous_anchor_fields"] if row[0] == trace_id)
        if any(fields[1]["coefficients"][1:]):
            raise ValueError("Independent source oracle expects the unchanged constant initial gap")
        g0 = decimal(F.from_float(fields[1]["coefficients"][0]))
        ds, de = (tuple(v-p for v, p in zip(rotate(rotate(point, donor, Decimal(1)), receiver, Decimal(-1)), point))
            for point in (start, end))
        aa, bb = dot(start, ds), dot(end, de)
        ab = dot(start, de)+dot(end, ds)
        a = g0+cosine*((wo1-wo0)-(ws1-ws0))-(radius+ws1)*cosine*(aa+bb)/2
        b, c = (-(radius+ws1)*sine*dot(horizontal, change) for change in (ds, de))
        d, e = -(radius+ws1)*cosine*(aa-bb)/2, -(radius+ws1)*cosine*ab/2
        constant = a*a+(b*b+c*c+d*d+e*e)/2-g0*g0
        cosines = (2*a*b+b*d+c*e, 2*a*d+(b*b-c*c)/2, b*d-c*e, (d*d-e*e)/2)
        sines = (2*a*c-c*d+b*e, 2*a*e+b*c, b*e+c*d, d*e)
        native_scale = pi_decimal()/2/decimal(extent)
        low, high = decimal(low_computational)*native_scale, decimal(high_computational)*native_scale
        total = constant*(high-low)
        for frequency, (cosine_coefficient, sine_coefficient) in enumerate(zip(cosines, sines), 1):
            sl, cl = sine_cosine(frequency*low)
            sh, ch = sine_cosine(frequency*high)
            total += (cosine_coefficient*(sh-sl)+sine_coefficient*(cl-ch))/frequency
        coefficient, _ = original_measure_and_coefficients(source, trace_id)
        return total/2*decimal(coefficient)/native_scale
