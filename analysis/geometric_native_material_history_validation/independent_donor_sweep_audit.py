"""Independent full-vector audit of a genuine two-dimensional donor sweep.

No production module is imported. Exact Fraction matrix series prove a complete
three-root schedule over the entire material rectangle. Separate high precision
Rodrigues evaluations locate the roots at five original material points. This
receipt audits geometry; it does not adopt a material or force transaction.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
from math import comb, factorial
from pathlib import Path


F = Fraction
ZERO = F(0)
ORDER = 20
DECIMAL_PRECISION = 500


@dataclass(frozen=True)
class Bounds:
    """Local exact interval implementation, independent of the native kernel."""

    lo: Fraction
    hi: Fraction | None = None

    def __post_init__(self):
        lo = F(self.lo)
        hi = lo if self.hi is None else F(self.hi)
        if lo > hi:
            raise ValueError("Independent interval endpoints are reversed")
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)

    def __add__(self, other):
        other = bounds(other)
        return Bounds(self.lo+other.lo, self.hi+other.hi)

    __radd__ = __add__

    def __neg__(self):
        return Bounds(-self.hi, -self.lo)

    def __sub__(self, other):
        return self+-bounds(other)

    def __mul__(self, other):
        other = bounds(other)
        values = tuple(x*y for x in (self.lo, self.hi) for y in (other.lo, other.hi))
        return Bounds(min(values), max(values))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = bounds(other)
        if other.lo <= 0 <= other.hi:
            raise ValueError("Independent interval divisor contains zero")
        return self*Bounds(1/other.hi, 1/other.lo)

    @property
    def midpoint(self):
        return (self.lo+self.hi)/2


def bounds(value):
    return value if type(value) is Bounds else Bounds(value)


def rational(value):
    if type(value) is Fraction:
        return value
    if type(value) is int:
        return F(value)
    if type(value) is float:
        return F.from_float(value)
    if (type(value) in (tuple, list) and len(value) == 2
            and all(type(item) is int for item in value) and value[1] > 0):
        return F(*value)
    if type(value) is dict and set(value) == {"numerator", "denominator"}:
        return rational([value["numerator"], value["denominator"]])
    raise ValueError("Independent scalar must preserve an exact rational input")


def interval(value):
    if type(value) is Bounds:
        return value
    if type(value) is dict and set(value) == {"lo", "hi"}:
        return Bounds(rational(value["lo"]), rational(value["hi"]))
    if type(value) is dict and set(value) == {"lower", "upper"}:
        return Bounds(rational(value["lower"]), rational(value["upper"]))
    raise ValueError("Independent interval must contain both original endpoints")


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def fraction_dto(value):
    return [value.numerator, value.denominator]


def _outward_report_interval(value):
    """Bound report size without rounding any proof or original coordinate."""
    maximum = max(abs(value.lo), abs(value.hi))
    if not maximum:
        return value
    exponent = maximum.numerator.bit_length()-maximum.denominator.bit_length()
    power = F(2**exponent) if exponent >= 0 else F(1, 2**-exponent)
    if maximum < power:
        exponent -= 1
    exponent -= 400
    unit = F(2**exponent) if exponent >= 0 else F(1, 2**-exponent)
    lower, upper = value.lo/unit, value.hi/unit
    return Bounds((lower.numerator//lower.denominator)*unit,
                  -((-upper.numerator)//upper.denominator)*unit)


def interval_dto(value, *, exact=False):
    value = value if exact else _outward_report_interval(value)
    return dict(lo=fraction_dto(value.lo), hi=fraction_dto(value.hi))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity():
    return tuple(tuple(F(i == j) for j in range(3)) for i in range(3))


def matmul(left, right):
    return tuple(tuple(sum((left[i][k]*right[k][j] for k in range(3)), ZERO)
                       for j in range(3)) for i in range(3))


def skew(axis):
    x, y, z = axis
    return ((ZERO, -z, y), (z, ZERO, -x), (-y, x, ZERO))


def relative_matrix_series(donor_rate, receiver_rate, order=ORDER):
    """Direct matrix exponential product: exp(-[d]u) exp([r]u)."""
    left, right = skew(tuple(-value for value in donor_rate)), skew(receiver_rate)
    left_powers, right_powers = [identity()], [identity()]
    for _ in range(order):
        left_powers.append(matmul(left_powers[-1], left))
        right_powers.append(matmul(right_powers[-1], right))
    result = []
    for degree in range(order+1):
        terms = tuple((matmul(left_powers[i], right_powers[degree-i]),
                       F(1, factorial(i)*factorial(degree-i)))
                      for i in range(degree+1))
        result.append(tuple(tuple(sum((matrix[row][column]*weight
                                       for matrix, weight in terms), ZERO)
                                  for column in range(3)) for row in range(3)))
    return tuple(result)


def plane_series(matrices, normal):
    return tuple(tuple(sum((matrix[i][j]*normal[j] for j in range(3)), ZERO)
                       for i in range(3)) for matrix in matrices)


def scalar_coefficients(series, patch):
    return tuple(sum((coefficient*coordinate
                      for coefficient, coordinate in zip(vector, patch)), Bounds(0))
                 for vector in series)


def remainder(rate, derivative=0, upper=F(1), order=ORDER):
    ratio = rate*upper/F(order+2-derivative)
    if not 0 <= derivative <= order or ratio >= 1 or upper < 0:
        raise ValueError("Independent series exceeds its declared derivative domain")
    return rate**(order+1)*upper**(order+1-derivative) / (
        factorial(order+1-derivative)*(1-ratio))


def point_bounds(coefficients, instant, tail):
    value = Bounds(0)
    for coefficient in reversed(coefficients):
        value = value*instant+coefficient
    return value+Bounds(-tail, tail)


def global_bounds(coefficients, tail, derivative=0):
    differentiated = tuple(value*F(factorial(k), factorial(k-derivative))
                           for k, value in enumerate(coefficients) if k >= derivative)
    degree = len(differentiated)-1
    bernstein = tuple(sum((differentiated[k]*F(comb(i, k), comb(degree, k))
                           for k in range(i+1)), Bounds(0)) for i in range(degree+1))
    return Bounds(min(value.lo for value in bernstein)-tail,
                  max(value.hi for value in bernstein)+tail)


def actual_owners(probe):
    assessments = [item for item in probe["admission"]["actual_overlap_assessments"]
                   if (item["fragment_a"], item["fragment_b"]) == ("octant:2", "octant:6")]
    if len(assessments) != 1 or len(assessments[0]["receiver_arcs"]) != 1:
        raise ValueError("Independent donor audit requires the actual original octant:2/6")
    row = assessments[0]["receiver_arcs"][0]
    if (row["overriding_plate"], row["subducting_plate"]) != (0, 1):
        raise ValueError("Independent donor audit requires original receiver/donor owners")
    if probe["accepted_loaded_native_intervals"] != 0 or not probe["input_unchanged"]:
        raise ValueError("Independent donor audit requires the unchanged unaccepted candidate")
    duration = rational(probe["admission"]["dt_myr"])
    if duration <= 0:
        raise ValueError("Independent donor audit requires the original positive duration")
    owner = {int(key): tuple(rational(value) for value in omega)
             for key, omega in probe["omega"].items()}
    if set(owner) != {0, 1} or any(len(value) != 3 for value in owner.values()):
        raise ValueError("Independent donor audit requires both complete Euler triples")
    support_histories = assessments[0]["native_overlap"]["supports"]
    wanted = {(-1, 0, 0): 0, (1, 0, 0): 1, (0, 0, -1): 1}
    if len(support_histories) != 3:
        raise ValueError("Independent donor audit lost actual endpoint overlap supports")
    for support in support_histories:
        support_owner = wanted.get(tuple(support["normal"]))
        if support_owner is None or len(support["rotations"]) != 1:
            raise ValueError("Independent donor audit changed actual support lineages")
        pose = support["rotations"][0]
        if (rational(pose["dt_myr"]) != duration
                or tuple(rational(x) for x in pose["omega_rad_per_myr"]) != owner[support_owner]):
            raise ValueError("Independent donor audit changed an original native rotation")
    return owner, duration


def certify_actual_rectangle(probe):
    """Independent exact proof for every point of a positive material area."""
    owners, duration = actual_owners(probe)
    rates = {owner: tuple(value*duration for value in omega)
             for owner, omega in owners.items()}
    matrices = relative_matrix_series(rates[1], rates[0])
    planes = {name: plane_series(matrices, tuple(F(x) for x in normal))
              for name, normal in (("receiver_west", (-1, 0, 0)),
                                   ("receiver_north", (0, 1, 0)),
                                   ("receiver_south", (0, 0, -1)))}
    west = planes["receiver_west"]
    q = []
    for k in range(3):
        q.append((west[k+1][2]-sum((q[j]*west[k+1-j][1] for j in range(k)), ZERO))
                 / west[1][1])
    reference = -q[1]/(2*q[2])
    height = -q[1]**2/(4*q[2])
    scale = west[1][1]*reference*height
    if min(q[0], reference, height, scale) <= 0 or 4*reference >= 1:
        raise ValueError("Independent actual material rectangle has no strict domain")
    patch = (Bounds(scale/1000, scale/500),
             Bounds(q[0]+height/3, q[0]+2*height/3), Bounds(-1))
    material_norm = sum((max(abs(v.lo), abs(v.hi)) for v in patch), ZERO)
    rate = sum((sum(map(abs, value), ZERO) for value in rates.values()), ZERO)
    coefficients = {name: scalar_coefficients(series, patch)
                    for name, series in planes.items()}
    probes = (ZERO, reference/16, reference, 4*reference, F(1))
    signs = tuple(point_bounds(coefficients["receiver_west"], instant,
                  remainder(rate, upper=instant)*material_norm) for instant in probes)
    expected = (-1, 1, -1, 1, 1)
    if any(not (value.lo > 0 if sign == 1 else value.hi < 0)
           for value, sign in zip(signs, expected)):
        raise ValueError("Independent entire material rectangle lost its alternating signs")
    third = global_bounds(coefficients["receiver_west"],
        remainder(rate, derivative=3)*material_norm, derivative=3)
    others = {name: global_bounds(coefficients[name], remainder(rate)*material_norm)
              for name in ("receiver_north", "receiver_south")}
    if third.lo <= 0 or any(value.lo <= 0 for value in others.values()):
        raise ValueError("Independent whole-time material schedule is not certified")
    # The chart normalization Jacobian is (1+x²+y²)^(-3/2). For the whole
    # explicit positive rectangle, norm²<2, hence Jacobian>1/4 strictly.
    norm_squared = 1+patch[0].hi**2+patch[1].hi**2
    if norm_squared >= 2:
        raise ValueError("Independent spherical material chart exceeds its area bound")
    chart_area = (patch[0].hi-patch[0].lo)*(patch[1].hi-patch[1].lo)
    area = Bounds(chart_area/4, chart_area)
    return dict(owners=owners, duration=duration, rates=rates, patch=patch,
        reference=reference, height=height, scale=scale, coefficients=coefficients,
        rate_bound=rate, signs=signs, probes=probes, third=third, other_supports=others,
        area=area, three_roots_complete=True,
        root_count_proof="Three strict sign changes give at least three roots; strictly positive third derivative and Rolle's theorem give at most three roots for each material point.")


def spherical_rectangle_area(patch):
    """Integrate the chart Jacobian with a proved binomial-series remainder."""
    x, y = patch[:2]
    measure = (x.hi-x.lo)*(y.hi-y.lo)
    integral = ZERO
    for degree, coefficient in enumerate((F(1), F(-3, 2), F(15, 8))):
        moment = sum((F(comb(degree, k))
            *(x.hi**(2*k+1)-x.lo**(2*k+1))/F(2*k+1)
            *(y.hi**(2*(degree-k)+1)-y.lo**(2*(degree-k)+1))/F(2*(degree-k)+1)
            for k in range(degree+1)), ZERO)
        integral += coefficient*moment
    maximum_square = x.hi**2+y.hi**2
    if maximum_square >= F(1, 2):
        raise ValueError("Independent Jacobian remainder is outside its small chart domain")
    # For s <= maximum_square < 1/2, the alternating binomial terms decrease.
    tail = F(35, 16)*maximum_square**3*measure
    return Bounds(integral-tail, integral)


def half_pi_bounds():
    """Machin's identity with independently bounded alternating arctangents."""
    def arctangent(denominator):
        count = 128
        total = sum((F((-1)**k, (2*k+1)*denominator**(2*k+1))
                     for k in range(count)), ZERO)
        next_term = F((-1)**count, (2*count+1)*denominator**(2*count+1))
        return Bounds(min(total, total+next_term), max(total, total+next_term))
    return 8*arctangent(5)-2*arctangent(239)


def decimal(value):
    value = rational(value)
    return Decimal(value.numerator)/Decimal(value.denominator)


def decimal_dot(left, right):
    return sum((x*y for x, y in zip(left, right)), Decimal(0))


def decimal_cross(left, right):
    return (left[1]*right[2]-left[2]*right[1],
            left[2]*right[0]-left[0]*right[2],
            left[0]*right[1]-left[1]*right[0])


def decimal_rotation(point, rate, instant):
    norm = decimal_dot(rate, rate).sqrt()
    if not norm or not instant:
        return point
    angle, axis = norm*instant, tuple(value/norm for value in rate)
    if abs(angle) > 1:
        raise ValueError("Independent Rodrigues rotation exceeds its audit domain")
    sine = sine_term = angle
    cosine = cosine_term = Decimal(1)
    threshold = Decimal(10)**-(DECIMAL_PRECISION+10)
    k = 0
    while True:
        sine_term *= -angle**2/((2*k+2)*(2*k+3))
        cosine_term *= -angle**2/((2*k+1)*(2*k+2))
        sine += sine_term
        cosine += cosine_term
        if max(abs(sine_term), abs(cosine_term)) < threshold:
            break
        k += 1
    cross = decimal_cross(axis, point)
    along = decimal_dot(axis, point)
    return tuple(x*cosine+y*sine+z*along*(1-cosine)
                 for x, y, z in zip(point, cross, axis))


def numerical_point_roots(evidence, point):
    """Rodrigues replay of a fully proved point, independent of matrix series."""
    rates = {owner: tuple(decimal(x) for x in value)
             for owner, value in evidence["rates"].items()}
    coordinates = tuple(decimal(x) for x in point)
    normal = (Decimal(-1), Decimal(0), Decimal(0))

    def gap(instant):
        moving_normal = decimal_rotation(normal, rates[0], instant)
        moving_point = decimal_rotation(coordinates, rates[1], instant)
        return decimal_dot(moving_normal, moving_point)

    brackets = tuple((decimal(low), decimal(high))
                     for low, high in zip(evidence["probes"][:3], evidence["probes"][1:4]))
    result = []
    for kind, (low, high) in zip(("entry", "exit", "entry"), brackets):
        first, last = gap(low), gap(high)
        if not first or not last or (first > 0) == (last > 0):
            raise ValueError("Independent numerical root lost an already proved bracket")
        threshold = Decimal(10)**-400
        while high-low > threshold:
            middle = (low+high)/2
            value = gap(middle)
            if not value:
                low = high = middle
            elif (value > 0) == (first > 0):
                low, first = middle, value
            else:
                high = middle
        root = (low+high)/2
        residual = gap(root)
        if abs(residual) >= Decimal("1e-430"):
            raise ValueError("Independent numerical root has an excessive residual")
        result.append(dict(kind=kind, fraction=str(root), time_myr=str(root*decimal(evidence["duration"])),
                           residual=str(residual), numerical_only=True))
    return result


def audit_actual(probe):
    exact = certify_actual_rectangle(probe)
    patch = exact["patch"]
    points = (("midpoint", tuple(value.midpoint for value in patch)),
              *((f"corner:{i}:{j}", (x, y, F(-1)))
                for i, x in enumerate((patch[0].lo, patch[0].hi))
                for j, y in enumerate((patch[1].lo, patch[1].hi))))
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        numerical = [dict(original_material_point=[fraction_dto(x) for x in point],
                          label=label, events=numerical_point_roots(exact, point))
                     for label, point in points]
    return exact, dict(format="independent-actual-donor-sweep-audit-1",
        source_material_id="octant-material:6", source_fragment_id="octant:6",
        receiver_material_id="octant-material:2", receiver_fragment_id="octant:2",
        chart="original donor p=normalize(x,y,-1); no world projection",
        original_donor_patch=[interval_dto(value, exact=True) for value in patch],
        rational_reference_fraction=fraction_dto(exact["reference"]),
        reference_is_event_time=False,
        all_patch_state_signs=[dict(time_fraction=fraction_dto(t), receiver_west_gap=interval_dto(g))
                               for t, g in zip(exact["probes"], exact["signs"])],
        whole_time_third_gap_derivative=interval_dto(exact["third"]),
        whole_time_other_receiver_supports={name: interval_dto(value)
                                           for name, value in exact["other_supports"].items()},
        positive_two_dimensional_area_sr=interval_dto(exact["area"]),
        certified_roots_per_material_point=3,
        complete_root_count_proof=exact["root_count_proof"],
        point_events_independent_rodrigues=numerical,
        decimal_precision=DECIMAL_PRECISION, original_complete_owner_vectors_preserved=True,
        admission_authorized=False,
        limitation="Independent geometry proof does not advance material, memory, work or a force receipt.")


def _original_octant(raw, expected):
    if (type(raw) is not dict or raw.get("version") != "geometric-native-polygons-1"
            or raw.get("pose") != [] or len(raw.get("supports", [])) != 3
            or len(raw.get("vertices", [])) != 3):
        raise ValueError("Production certificate lost its original native octant")
    support_axes = {tuple(item["normal"]) for item in raw["supports"]}
    ray_axes = {tuple(item["reference"]) for item in raw["vertices"]}
    if support_axes != set(expected) or ray_axes != set(expected):
        raise ValueError("Production certificate changed original material domains")
    if any(item.get("rotations") != [] for item in raw["supports"]):
        raise ValueError("Original native support contains an invented pose")
    if any(item.get("kind") != "seed" or item.get("rotations") != []
           or item.get("planes") != [] or item.get("sign") != 1
           for item in raw["vertices"]):
        raise ValueError("Original native vertices contain an invented lineage")


def _patch_geometry(raw, evidence):
    if (type(raw) is not dict or raw.get("version") != "geometric-native-polygons-1"
            or raw.get("pose") != [] or len(raw.get("vertices", [])) != 4
            or len(raw.get("supports", [])) != 4):
        raise ValueError("Production material patch must retain four native corners")
    patch = evidence["patch"]
    wanted = {(x, y, F(-1)) for x in (patch[0].lo, patch[0].hi)
              for y in (patch[1].lo, patch[1].hi)}
    actual = set()
    for ray in raw["vertices"]:
        if (ray.get("kind") != "seed" or ray.get("rotations") != []
                or ray.get("planes") != [] or ray.get("sign") != 1):
            raise ValueError("Production material patch changed a primitive corner")
        reference = tuple(rational(x) for x in ray["reference"])
        if len(reference) != 3 or reference[2] >= 0:
            raise ValueError("Production material patch left the original donor chart")
        actual.add(tuple(value/-reference[2] for value in reference))
    if actual != wanted:
        raise ValueError("Production native patch and exact donor rectangle differ")
    for support in raw["supports"]:
        if support.get("rotations") != []:
            raise ValueError("Production patch support changed its original donor epoch")
        normal = tuple(rational(x) for x in support["normal"])
        if len(normal) != 3 or all(not value for value in normal):
            raise ValueError("Production patch support is not a native halfspace")
        if any(sum((a*b for a, b in zip(normal, point)), ZERO) < 0 for point in wanted):
            raise ValueError("Production patch support rejects its actual rectangle")


def bind_production_certificate(raw, evidence, report):
    """Bind actual native domains, entire patch and complete event windows."""
    if (raw.get("format") != "native-original-donor-patch-sweep-1"
            or raw.get("model") != "complete_three_event_original_donor_patch"
            or raw.get("admission_authorized") is not False):
        raise ValueError("Independent audit requires the explicit diagnostic native sweep format")
    body = dict(raw)
    seal = body.pop("payload_sha256", None)
    if seal != hashlib.sha256(encoded(body)).hexdigest():
        raise ValueError("Production donor certificate seal differs from its exact body")
    trajectory = raw["trajectory"]
    _original_octant(trajectory["donor_polygon"], ((1, 0, 0), (0, 1, 0), (0, 0, -1)))
    _original_octant(trajectory["receiver_polygon"], ((-1, 0, 0), (0, 1, 0), (0, 0, -1)))
    for key, owner in (("donor_omega", 1), ("receiver_omega", 0)):
        if tuple(rational(x) for x in trajectory[key]) != evidence["owners"][owner]:
            raise ValueError("Production certificate altered an original Euler component")
    if rational(trajectory["dt_myr"]) != evidence["duration"]:
        raise ValueError("Production certificate altered the actual force duration")
    recipe = raw["recipe"]
    independent_recipe = dict(q0=evidence["patch"][1].lo-evidence["height"]/3,
        U=evidence["reference"], H=evidence["height"], B=evidence["scale"])
    for key, value in independent_recipe.items():
        if rational(recipe[key]) != value:
            raise ValueError("Production patch recipe differs from independent exact reconstruction")
    # q1/q2 are checked by their exact relationships with U/H, without
    # reusing production coefficients or a production interval routine.
    q1, q2 = rational(recipe["q1"]), rational(recipe["q2"])
    if (-q1/(2*q2) != evidence["reference"]
            or -q1**2/(4*q2) != evidence["height"]):
        raise ValueError("Production material patch turning recipe is inconsistent")
    certificate = raw["certificate"]
    for key, value in (("patch_x_bounds", evidence["patch"][0]),
                       ("patch_y_bounds", evidence["patch"][1])):
        if interval(certificate[key]) != value:
            raise ValueError("Production material patch coordinates differ")
    if tuple(rational(x) for x in certificate["sample_fractions"]) != evidence["probes"]:
        raise ValueError("Production whole-patch witness times differ")
    first_negative = rational(certificate["first_entry_negative_fraction"])
    if first_negative != evidence["reference"]/4096:
        raise ValueError("Production first-entry lower bound changed its exact recipe")
    material_norm = sum((max(abs(x.lo), abs(x.hi)) for x in evidence["patch"]), ZERO)
    first_negative_gap = point_bounds(evidence["coefficients"]["receiver_west"], first_negative,
        remainder(evidence["rate_bound"], upper=first_negative)*material_norm)
    declared_first_gap = interval(certificate["first_entry_negative_gap_bounds"])
    if (first_negative_gap.hi >= 0 or declared_first_gap.hi >= 0
            or declared_first_gap.hi < first_negative_gap.lo
            or declared_first_gap.lo > first_negative_gap.hi):
        raise ValueError("Production first material entry lacks a strictly positive time bound")
    declared_samples = tuple(interval(value) for value in certificate["gap_sample_bounds"])
    if (len(declared_samples) != 5 or any(
            saved.hi < independent.lo or saved.lo > independent.hi
            for saved, independent in zip(declared_samples, evidence["signs"]))):
        raise ValueError("Production gap samples disagree with independent entire-patch signs")
    _patch_geometry(raw["patch_geometry"], evidence)
    area = interval(certificate["patch_area_bounds"])
    if area.lo <= 0 or area.hi < evidence["area"].lo or area.lo > evidence["area"].hi:
        raise ValueError("Production positive area disagrees with independent spherical chart")
    third = interval(certificate["gap_third_derivative_bounds"])
    if third.lo <= 0 or third.hi < evidence["third"].lo or third.lo > evidence["third"].hi:
        raise ValueError("Production third derivative disagrees with independent whole-time proof")
    positive = certificate["receiver_positive_bounds"]
    independent_positive = tuple(evidence["other_supports"].values())
    if (len(positive) != 2 or any(interval(value).lo <= 0 for value in positive)
            or any(interval(saved).hi < independent.lo
                   or interval(saved).lo > independent.hi
                   for saved, independent in zip(positive, independent_positive))):
        raise ValueError("Production native receiver lacks all its whole-time supports")
    windows = certificate["event_windows"]
    if len(windows) != 3:
        raise ValueError("Production material sweep lost a complete crossing window")
    result = []
    previous = F(-1)
    norm = sum((max(abs(x.lo), abs(x.hi)) for x in evidence["patch"]), ZERO)
    for event, kind in zip(windows, ("entry", "exit", "entry")):
        time = interval(event["time_fraction"])
        physical = interval(event["time_myr"])
        if (event["kind"] != kind or not 0 <= time.lo < time.hi <= 1
                or time.lo < previous or physical.lo != time.lo*evidence["duration"]
                or physical.hi != time.hi*evidence["duration"]):
            raise ValueError("Production complete crossing windows are not original exact times")
        if not result and time.lo != first_negative:
            raise ValueError("Production first-entry window includes an unproved initial epoch")
        lower = point_bounds(evidence["coefficients"]["receiver_west"], time.lo,
                             remainder(evidence["rate_bound"], upper=time.lo)*norm)
        upper = point_bounds(evidence["coefficients"]["receiver_west"], time.hi,
                             remainder(evidence["rate_bound"], upper=time.hi)*norm)
        if (not ((lower.hi < 0 and upper.lo > 0) if kind == "entry"
                else (lower.lo > 0 and upper.hi < 0))):
            raise ValueError("Production window does not bracket every original material point")
        with localcontext() as context:
            context.prec = DECIMAL_PRECISION
            low, high = decimal(time.lo), decimal(time.hi)
            for point in report["point_events_independent_rodrigues"]:
                root = Decimal(point["events"][len(result)]["fraction"])
                if not 0 < root < 1 or not low <= root <= high:
                    raise ValueError("Production window omits an independent material root")
        previous = time.hi
        result.append(dict(kind=kind, time_fraction=interval_dto(time, exact=True),
            independently_certified_all_patch_lower_gap=interval_dto(lower),
            independently_certified_all_patch_upper_gap=interval_dto(upper),
            every_patch_point_bracketed=True, independent_midpoint_and_corner_roots_inside=True))
    report["production_windows_independently_verified"] = result
    report["production_payload_sha256"] = seal
    report["production_patch_exactly_matches_independent_rectangle"] = True
    report["production_complete_native_domains_match_original_force_candidate"] = True
    report["first_material_entry_has_strict_positive_time_lower_bound"] = True


def bind_material_history(history, initial, candidate, support, sweep, evidence, report):
    """Audit local first-entry conservation and immutable mechanical references.

    The original force solve and constitutive certificates are external
    evidence. This function checks their exact preservation, rather than
    claiming to have independently solved that mechanical problem.
    """
    if (history.get("format") != "native-original-donor-patch-material-history-1"
            or history.get("model") != "once_only_passive_archive_first_entry"
            or history.get("admission_authorized") is not False
            or history.get("global_inventory_updated") is not False
            or history.get("archive_policy") != "one_original_donor_cohort; contact_exit_does_not_unbury"):
        raise ValueError("Independent history audit requires the explicit local passive archive closure")
    body = dict(history)
    seal = body.pop("history_sha256", None)
    if seal != hashlib.sha256(encoded(body)).hexdigest():
        raise ValueError("Local material history seal differs from its complete body")
    fragments = {item["fragment_id"]: item for item in initial["current_surface"]["fragments"]}
    parent, receiver = fragments["octant:6"], fragments["octant:2"]
    if history["parent"] != parent or history["receiver_parent"] != receiver:
        raise ValueError("Local chronology reset an original parent parcel or native geometry")
    if (parent["parcel"]["material_id"], parent["parcel"]["plate"],
            receiver["parcel"]["material_id"], receiver["parcel"]["plate"]) != (
                "octant-material:6", 1, "octant-material:2", 0):
        raise ValueError("Local chronology changed original donor/receiver ownership")
    parent_hash, receiver_hash = (hashlib.sha256(encoded(value)).hexdigest()
                                  for value in (parent, receiver))
    patch_hash = hashlib.sha256(encoded(sweep["patch_geometry"])).hexdigest()
    sweep_hash = hashlib.sha256(encoded(sweep)).hexdigest()
    denominator_hash = hashlib.sha256(encoded(parent["geometry"])).hexdigest()
    if (history["parent_payload_sha256"] != parent_hash
            or history["receiver_payload_sha256"] != receiver_hash
            or history["patch_geometry_sha256"] != patch_hash
            or history["sweep_sha256"] != sweep_hash
            or history["sweep_certificate"] != sweep):
        raise ValueError("Local material ancestry references differ from actual native sources")
    cohort = "native-passive-donor-patch:"+hashlib.sha256(encoded(dict(
        parent=parent_hash, receiver=receiver_hash, patch=patch_hash, sweep=sweep_hash))).hexdigest()
    if history["cohort_id"] != cohort:
        raise ValueError("Local material archive cohort lost its original ancestry")
    epoch = rational(initial["current_surface"]["time_myr"])
    if rational(history["material_epoch_myr"]) != epoch:
        raise ValueError("Local material history changed the original clock epoch")
    dt = evidence["duration"]
    events = history["events"]
    if len(events) != 3:
        raise ValueError("Local material history lost a genuine complete crossing")
    operations = ("live_to_passive_archive_once", "contact_exit_keep_archive", "contact_reentry_reuse_archive")
    for index, (event, geometry, operation) in enumerate(zip(events, sweep["certificate"]["event_windows"], operations), 1):
        time = interval(event["time_fraction"])
        relative, absolute = interval(event["relative_time_myr"]), interval(event["absolute_time_myr"])
        if (event["sequence"] != index or event["kind"] != geometry["kind"]
                or event["time_fraction"] != geometry["time_fraction"]
                or relative != time*dt or absolute != relative+epoch
                or event["operation"] != operation or event["cohort_id"] != cohort
                or event["material_time_function"] != "unique_sweep_root_for_each_original_donor_point"):
            raise ValueError("Local material timeline reset an epoch, duplicated a transfer or changed a crossing")
    if interval(events[0]["time_fraction"]).lo <= 0:
        raise ValueError("Local material archive falsely initializes first entry at epoch zero")
    ratios = []
    independent_area = spherical_rectangle_area(evidence["patch"])
    independent_ratio = independent_area/half_pi_bounds()
    fields = ("area_km2", "oceanic_volume_km3", "cold_mantle_volume_km3", "density_excess_mass_kg")
    shares, partitions = history["shares"], history["parent_reservoir_partitions"]
    if tuple(value["reservoir"] for value in shares) != fields or len(partitions) != len(fields):
        raise ValueError("Local material history omitted an original extensive reservoir")
    for share, partition, field in zip(shares, partitions, fields):
        amount = rational(parent["parcel"][field])
        ratio = interval(share["native_area_ratio_bounds"])
        quantity = interval(share["amount_bounds"])
        if (rational(share["parent_amount"]) != amount or quantity != ratio*amount
                or not 0 < ratio.lo <= ratio.hi < 1
                or ratio.hi < independent_ratio.lo or ratio.lo > independent_ratio.hi
                or share["numerator_geometry_sha256"] != patch_hash
                or share["denominator_geometry_sha256"] != denominator_hash
                or share["expression"] != "parent_amount * native_area(patch) / native_area(parent)"):
            raise ValueError("Local material allocation differs from its true original positive area share")
        if (partition["reservoir"] != field or rational(partition["parent_amount"]) != amount
                or partition["patch_share"] != share
                or interval(partition["unmodelled_parent_complement_bounds"]) != (Bounds(1)-ratio)*amount
                or partition["conservation_identity"] != "patch_share + unmodelled_parent_complement = parent_amount"):
            raise ValueError("Local original-parent reservoir complement is not conserved")
        ratios.append(ratio)
    if any(value != ratios[0] for value in ratios):
        raise ValueError("Local material reservoirs use different implicit original subsets")
    memory = history["memory_reference"]
    old = initial["native_underthrust_motion"]["snapshot"]
    before_keys = ("states", "spatial_memory", "free_anchor_fields", "elapsed_time_s",
        "cumulative_source_work_j", "cumulative_basal_dissipation_j", "cumulative_bending_numerical_dissipation_j")
    expected_before = {key: old[key] for key in before_keys}
    expected_after = dict(states=candidate["states"], spatial_memory=candidate["spatial_step"]["memory"],
        free_anchor_fields=candidate["spatial_step"]["anchor_fields"], dt_s=candidate["dt_s"],
        candidate_diagnostics=candidate["diagnostics"],
        candidate_sha256=hashlib.sha256(encoded(candidate)).hexdigest(),
        whole_time_support=support, candidate_adopted=False)
    if (memory["before_payload"] != expected_before or memory["candidate_after_payload"] != expected_after
            or memory["before_sha256"] != hashlib.sha256(encoded(expected_before)).hexdigest()
            or memory["candidate_after_sha256"] != hashlib.sha256(encoded(expected_after)).hexdigest()
            or memory["constitutive_path_sha256"] != support["path_sha256"]):
        raise ValueError("Local material chronology changed an original mechanical memory or work reference")
    if (candidate["admission"]["dt_myr"] != float(dt)
            or {int(owner): tuple(rational(x) for x in value)
                for owner, value in candidate["omega_rad_per_myr"]} != evidence["owners"]):
        raise ValueError("Local mechanical references differ from the actual full-vector material solve")
    pressure = support["certificates"]
    if (support.get("candidate_memory_adopted") is not False
            or support.get("geometry_permission") is not False or len(pressure) != 4
            or any(value.get("path_sha256") != support["path_sha256"]
                   or value.get("admissible") is not True
                   or interval(value["pressure_bounds_pa"]).lo <= 0
                   or interval(value["plastic_rate_bounds_m_s"]).hi >= 0
                   or value["time_fraction_bounds"] != [[0, 1], [1, 1]] for value in pressure)):
        raise ValueError("Local mechanical reference lost the original whole-time constitutive evidence")
    report["local_history_independently_bound"] = dict(history_sha256=seal,
        original_parent_and_receiver_payloads_identical=True, original_material_intensives_and_age_preserved=True,
        four_original_extensive_reservoirs_conserved_symbolically=True,
        independent_native_area_ratio_bounds=interval_dto(independent_ratio),
        complete_original_crossings_reused=True, one_archive_cohort_for_all_three_events=True,
        archive_transfer_count=1, contact_exit_does_not_resurrect_material=True,
        contact_reentry_does_not_allocate_again=True, first_archive_entry_strictly_after_initial_epoch=True,
        mechanical_memory_and_work_are_exact_external_references=True,
        independent_force_or_constitutive_solve_performed=False,
        global_inventory_updated=False, admission_authorized=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--certificate", type=Path)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--support", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    probe_bytes = arguments.probe.read_bytes()
    probe = json.loads(probe_bytes)
    exact, report = audit_actual(probe)
    if arguments.certificate is not None:
        certificate_bytes = arguments.certificate.read_bytes()
        bind_production_certificate(json.loads(certificate_bytes), exact, report)
        if arguments.certificate.read_bytes() != certificate_bytes:
            raise ValueError("Production certificate changed during independent audit")
        report["source_certificate_sha256"] = hashlib.sha256(certificate_bytes).hexdigest()
    history_paths = (arguments.history, arguments.initial, arguments.candidate, arguments.support)
    if any(path is not None for path in history_paths):
        if arguments.certificate is None or any(path is None for path in history_paths):
            raise ValueError("A local history audit needs certificate/history/initial/candidate/support inputs")
        frozen = tuple(path.read_bytes() for path in history_paths)
        bind_material_history(*(json.loads(value) for value in frozen),
                              json.loads(certificate_bytes), exact, report)
        if any(path.read_bytes() != value for path, value in zip(history_paths, frozen)):
            raise ValueError("An original local-history input changed during independent audit")
        report["history_source_files_sha256"] = {name: hashlib.sha256(value).hexdigest()
            for name, value in zip(("history", "initial", "candidate", "support"), frozen)}
    if arguments.probe.read_bytes() != probe_bytes:
        raise ValueError("Independent input changed during the audit")
    report["source_probe_sha256"] = hashlib.sha256(probe_bytes).hexdigest()
    report["audit_source_sha256"] = sha256(__file__)
    report["receipt_sha256"] = hashlib.sha256(encoded(report)).hexdigest()
    with arguments.out.open("xb") as stream:
        stream.write(encoded(report)+b"\n")
    print(json.dumps(dict(out=str(arguments.out), receipt_sha256=report["receipt_sha256"],
                         certified_roots_per_material_point=3, admission_authorized=False)))


if __name__ == "__main__":
    main()
