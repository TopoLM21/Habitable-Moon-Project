"""Independent exact point chronology and full-source domain evidence audit.

Only the standard library is used. Exact rational matrix exponential series,
Bernstein bounds and analytic remainder bounds verify every saved point proof
leaf; separate Decimal(500) Rodrigues calculations replay roots and geometry.
The endpoint native geometry checks are independent numerical checks, not a
replacement exact spherical set construction or a material/force adoption.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
from math import comb, factorial, isqrt
from pathlib import Path
import time

F = Fraction
PRECISION = 500
ZERO = F(0)
ONE = F(1)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def rational(value):
    if type(value) is F:
        return value
    if type(value) is int:
        return F(value)
    if type(value) is float:
        return F.from_float(value)
    if (type(value) is list and len(value) == 2
            and all(type(item) is int for item in value) and value[1] > 0):
        return F(*value)
    raise ValueError("Independent audit requires original finite rational scalar inputs")


def interval(value):
    if type(value) is not dict or set(value) != {"lo", "hi"}:
        raise ValueError("Missing complete exact interval")
    low, high = rational(value["lo"]), rational(value["hi"])
    if low > high:
        raise ValueError("Reversed exact interval")
    return low, high


def fdto(value):
    return [value.numerator, value.denominator]


def idto(low, high):
    return dict(lo=fdto(low), hi=fdto(high))


def dot(a, b):
    return sum((x*y for x, y in zip(a, b)), ZERO)


def cross(a, b):
    return tuple(a[j]*b[k]-a[k]*b[j] for j, k in ((1, 2), (2, 0), (0, 1)))


def normalize_scale(vector):
    vector = tuple(rational(x) for x in vector)
    if len(vector) != 3 or not any(vector):
        raise ValueError("A full nonzero three-dimensional vector is required")
    scale = max(map(abs, vector))
    return tuple(x/scale for x in vector)


def eye():
    return tuple(tuple(F(i == j) for j in range(3)) for i in range(3))


def skew(a):
    x, y, z = a
    return ((ZERO, -z, y), (z, ZERO, -x), (-y, x, ZERO))


def matmul(a, b):
    return tuple(tuple(sum((a[i][k]*b[k][j] for k in range(3)), ZERO)
                       for j in range(3)) for i in range(3))


@lru_cache(maxsize=64)
def matrix_series(donor, receiver, order):
    left, right = skew(tuple(-x for x in donor)), skew(receiver)
    a, b = [eye()], [eye()]
    for k in range(1, order+1):
        a.append(tuple(tuple(x/F(k) for x in row) for row in matmul(a[-1], left)))
        b.append(tuple(tuple(x/F(k) for x in row) for row in matmul(b[-1], right)))
    result = []
    for k in range(order+1):
        terms = tuple(matmul(a[j], b[k-j]) for j in range(k+1))
        result.append(tuple(tuple(sum((term[i][m] for term in terms), ZERO)
                                 for m in range(3)) for i in range(3)))
    return tuple(result)


def polynomial_multiply(a, b):
    result = [ZERO]*(len(a)+len(b)-1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            result[i+j] += x*y
    return tuple(result)


def generator_characteristic(donor, receiver):
    """Nine-dimensional Kronecker generator spectrum, with full axis norms."""
    a, b = dot(donor, donor), dot(receiver, receiver)
    value = (ZERO, ONE)
    for factor in ((a, ZERO, ONE), (b, ZERO, ONE),
                   ((a-b)**2, ZERO, 2*(a+b), ZERO, ONE)):
        value = polynomial_multiply(value, factor)
    return value


@lru_cache(maxsize=512)
def coefficients(donor, receiver, normal, point, order):
    normal, point = normalize_scale(normal), normalize_scale(point)
    matrices = matrix_series(donor, receiver, order)
    values = tuple(dot(point, tuple(dot(row, normal) for row in matrix))
                   for matrix in matrices)
    mass = sum(map(abs, normal), ZERO)*sum(map(abs, point), ZERO)
    return values, mass


def factor_tail(rate, mass, order, initial_order, upper, derivative=0):
    degree = order+1-initial_order
    if upper < 0 or upper > 1 or derivative > degree:
        raise ValueError("Unsupported independent factored-series derivative domain")
    ratio = rate*upper*F(degree+1, (order+2)*(degree+1-derivative))
    if ratio >= 1:
        raise ValueError("Independent Taylor remainder lacks geometric domination")
    first = mass*rate**(order+1)/factorial(order+1)
    first *= F(factorial(degree), factorial(degree-derivative))*upper**(degree-derivative)
    return first/(1-ratio)


def power_bounds(coefficients, low, high):
    """Exact Bernstein enclosure after affine restriction to an entire leaf."""
    degree, width = len(coefficients)-1, high-low
    shifted = tuple(sum((coefficients[k]*comb(k, j)*low**(k-j)*width**j
                         for k in range(j, degree+1)), ZERO)
                    for j in range(degree+1))
    bernstein = tuple(sum((shifted[k]*F(comb(i, k), comb(degree, k))
                          for k in range(i+1)), ZERO) for i in range(degree+1))
    return min(bernstein), max(bernstein)


def factored_bounds(values, rate, mass, initial_order, low, high, derivative=0):
    original_order = len(values)-1
    factored = values[initial_order:]
    derived = tuple(x*F(factorial(k), factorial(k-derivative))
                    for k, x in enumerate(factored) if k >= derivative)
    low_value, high_value = power_bounds(derived, low, high)
    tail = factor_tail(rate, mass, original_order, initial_order, high, derivative)
    return low_value-tail, high_value+tail


def strict_sign(bounds):
    low, high = bounds
    return 1 if low > 0 else -1 if high < 0 else 0


def encloses(saved, independent, name):
    low, high = interval(saved)
    if low > independent[0] or high < independent[1]:
        raise ValueError("Saved " + name + " does not enclose its independent exact Taylor proof")


def decimal(value):
    value = rational(value)
    return Decimal(value.numerator)/Decimal(value.denominator)


def ddot(a, b):
    return sum((x*y for x, y in zip(a, b)), Decimal(0))


def rotate(point, rate, instant=Decimal(1)):
    """Independent analytic Rodrigues evaluation at 500 decimal digits."""
    norm = ddot(rate, rate).sqrt()
    if not norm or not instant:
        return point
    angle, axis = norm*instant, tuple(x/norm for x in rate)
    if abs(angle) > 1:
        raise ValueError("Independent Rodrigues audit is restricted to small analytic increments")
    sine = sine_term = angle
    cosine = cosine_term = Decimal(1)
    threshold = Decimal(10)**-(PRECISION+10)
    for k in range(10000):
        sine_term *= -angle**2/((2*k+2)*(2*k+3))
        cosine_term *= -angle**2/((2*k+1)*(2*k+2))
        sine += sine_term
        cosine += cosine_term
        if max(abs(sine_term), abs(cosine_term)) < threshold:
            break
    else:
        raise ValueError("Independent Rodrigues series did not finish")
    return tuple(x*cosine+y*sine+z*ddot(axis, point)*(1-cosine)
                 for x, y, z in zip(point, cross(axis, point), axis))


def _gap_decimal(normal, point, donor, receiver, instant):
    return ddot(rotate(tuple(decimal(x) for x in normal),
                       tuple(decimal(x) for x in receiver), decimal(instant)),
                rotate(tuple(decimal(x) for x in point),
                       tuple(decimal(x) for x in donor), decimal(instant)))


def _plain_first_epoch_polygon(raw):
    if (type(raw) is not dict or raw.get("version") != "geometric-native-polygons-1"
            or raw.get("pose") != [] or not raw.get("supports")
            or any(s.get("rotations") for s in raw["supports"])
            or any(v.get("kind") != "seed" or v.get("rotations") for v in raw["vertices"])):
        raise ValueError("Independent point proof requires literal first-epoch native planes and rays")
    return raw


def audit_certificate(raw, *, expected_trajectory=None):
    """Verify every exact proof leaf without consulting production code or seals."""
    if raw.get("format") != "native-full-point-chronology-1":
        raise ValueError("Unsupported complete point chronology format")
    if raw.get("payload_sha256") != digest({key: value for key, value in raw.items() if key != "payload_sha256"}):
        raise ValueError("Saved point proof integrity seal is inconsistent")
    trajectory = raw["trajectory"]
    if expected_trajectory is not None and encoded(trajectory) != encoded(expected_trajectory):
        raise ValueError("Point chronology replaced an authentic full source trajectory")
    receiver_polygon = _plain_first_epoch_polygon(trajectory["receiver_polygon"])
    donor_polygon = trajectory["donor_polygon"]
    if donor_polygon is not None:
        _plain_first_epoch_polygon(donor_polygon)
    point = tuple(rational(x) for x in trajectory["point"])
    duration = rational(trajectory["dt_myr"])
    donor = tuple(rational(x)*duration for x in trajectory["donor_omega"])
    receiver = tuple(rational(x)*duration for x in trajectory["receiver_omega"])
    if len(donor) != 3 or len(receiver) != 3 or duration <= 0:
        raise ValueError("Point chronology lost the full original Euler triples")
    characteristic = tuple(rational(x) for x in raw["generator_characteristic_polynomial"])
    if characteristic != generator_characteristic(donor, receiver):
        raise ValueError("Full-axis nine-dimensional generator polynomial changed")
    if donor_polygon is not None and any(dot(point, tuple(rational(x) for x in s["normal"])) < 0
                                         for s in donor_polygon["supports"]):
        raise ValueError("Original material point is outside its literal donor domain")
    order = raw["order"]
    if type(order) is not int or order < 8:
        raise ValueError("Point certificate cannot support the CH9 identity proof")
    rate = sum(map(abs, donor), ZERO)+sum(map(abs, receiver), ZERO)
    planes = raw["plane_chronologies"]
    if len(planes) != len(receiver_polygon["supports"]):
        raise ValueError("Point chronology omitted an authoritative receiver halfspace")
    roots, leaves, identities, initial_boundaries = [], 0, 0, []
    by_support = {}
    with localcontext() as context:
        context.prec = PRECISION
        for index, (proof, normal_dto) in enumerate(zip(planes, receiver_polygon["supports"])):
            if proof["support_index"] != index or encoded(proof["support"]) != encoded(normal_dto):
                raise ValueError("Point chronology changed literal receiver support lineage")
            normal = tuple(rational(x) for x in normal_dto["normal"])
            values, mass = coefficients(donor, receiver, normal, point, order)
            nonzero = next((i for i, value in enumerate(values[:9]) if value), None)
            coincident = nonzero is None
            if proof["coincident"] is not coincident or proof["initial_zero_order"] != nonzero:
                raise ValueError("Point initial zero multiplicity/CH9 identity is false")
            cells = proof["cells"]
            if not cells or interval(cells[0]["time_fraction"])[0] != 0:
                raise ValueError("Point proof omitted the beginning of its complete leaf partition")
            cursor = ZERO
            support_roots = []
            for cell in cells:
                low, high = interval(cell["time_fraction"])
                if low != cursor or high <= low or high > 1:
                    raise ValueError("Point proof leaves do not partition the complete original interval")
                cursor, leaves = high, leaves+1
                if coincident:
                    if (len(cells) != 1 or cell["status"] != "identically_zero" or cell["sign"] != 0
                            or interval(cell["gap_bounds"]) != (ZERO, ZERO)):
                        raise ValueError("Point CH9 identity was not retained over the whole interval")
                    identities += 1
                    continue
                bound = factored_bounds(values, rate, mass, nonzero, low, high)
                encloses(cell["gap_bounds"], bound, "factored gap bounds")
                if cell["status"] == "no_root":
                    sign = strict_sign(bound)
                    if not sign:
                        derivative = factored_bounds(values, rate, mass, nonzero, low, high, 1)
                        left = factored_bounds(values, rate, mass, nonzero, low, low)
                        right = factored_bounds(values, rate, mass, nonzero, high, high)
                        first, last, slope = strict_sign(left), strict_sign(right), strict_sign(derivative)
                        if slope and first == last and first:
                            sign = first
                    if not sign or cell["sign"] != sign:
                        raise ValueError("Independent complete leaf does not exclude a hidden point root")
                elif cell["status"] == "unique_root":
                    derivative = factored_bounds(values, rate, mass, nonzero, low, high, 1)
                    left = factored_bounds(values, rate, mass, nonzero, low, low)
                    right = factored_bounds(values, rate, mass, nonzero, high, high)
                    first, last, slope = strict_sign(left), strict_sign(right), strict_sign(derivative)
                    if not slope or not first or not last or first == last:
                        raise ValueError("Independent point root lacks strict bracket and oriented derivative proof")
                    encloses(cell["derivative_bounds"], derivative, "root derivative bounds")
                    encloses(cell["left_gap_bounds"], left, "left root gap bounds")
                    encloses(cell["right_gap_bounds"], right, "right root gap bounds")
                    a, b = interval(cell["root_fraction"])
                    if not low <= a < b <= high:
                        raise ValueError("Point root window is outside its complete proof leaf")
                    ra = factored_bounds(values, rate, mass, nonzero, a, a)
                    rb = factored_bounds(values, rate, mass, nonzero, b, b)
                    if strict_sign(ra) != first or strict_sign(rb) != last:
                        raise ValueError("Refined point root window is not an independently strict bracket")
                    numerical_a = _gap_decimal(normal, point, donor, receiver, a)
                    numerical_b = _gap_decimal(normal, point, donor, receiver, b)
                    if (numerical_a > 0) != (first > 0) or (numerical_b > 0) != (last > 0):
                        raise ValueError("Independent Rodrigues root bracket disagrees with exact rational proof")
                    root = dict(support_index=index, kind="entry" if first < last else "exit",
                                low=a, high=b, numerical_gap_at_left=str(numerical_a),
                                numerical_gap_at_right=str(numerical_b))
                    roots.append(root); support_roots.append(root)
                else:
                    raise ValueError("Point proof contains an unresolved or fabricated leaf status")
            if cursor != 1:
                raise ValueError("Point proof omitted the end of the complete original interval")
            if nonzero and not coincident:
                initial_boundaries.append(dict(support_index=index, kind="entry" if values[nonzero] > 0 else "exit",
                                               low=ZERO, high=ZERO))
            by_support[index] = dict(values=values, mass=mass, order=nonzero,
                                      roots=support_roots, coincident=coincident)
    roots.sort(key=lambda item: (item["low"], item["support_index"]))
    for first, second in zip(roots, roots[1:]):
        if first["high"] >= second["low"]:
            raise ValueError("Unproved nonzero simultaneous/overlapping point roots were silently resolved")
    all_events = sorted(initial_boundaries+roots, key=lambda r: (r["low"], r["support_index"]))
    event_dtos = []
    for sequence, root in enumerate(all_events, 1):
        root["sequence"] = sequence
        event_dtos.append(dict(sequence=sequence, kind=root["kind"], support_index=root["support_index"],
            time_fraction=idto(root["low"], root["high"]),
            time_myr=idto(root["low"]*duration, root["high"]*duration)))
    if encoded(raw["plane_events"]) != encoded(event_dtos):
        raise ValueError("Saved plane events dropped, reordered or invented a raw crossing")
    for index, proof in enumerate(planes):
        if encoded(proof["events"]) != encoded([r for r in event_dtos if r["support_index"] == index]):
            raise ValueError("Per-support event journal disagrees with independently proved roots")
    initial_sides = [dot(point, tuple(rational(x) for x in normal["normal"]))
                     for normal in receiver_polygon["supports"]]
    initial = "inside" if all(x >= 0 for x in initial_sides) else "outside"
    boundaries = [i for i, x in enumerate(initial_sides) if not x]
    if raw["initial_occupancy"] != initial or raw["initial_boundary_supports"] != boundaries:
        raise ValueError("Original point boundary occupancy changed its literal closed halfspaces")
    if not leaves <= raw["processed_cells"] <= raw["max_cells"]:
        raise ValueError("Point proof leaf budget is inconsistent")
    germs = {i: (0 if info["coincident"] else (1 if info["values"][info["order"]] > 0 else -1))
             for i, info in by_support.items()}
    occupancy = []
    zero_events = [r for r in all_events if r["high"] == 0]
    if zero_events and initial == "inside":
        kind = "entry" if all(sign >= 0 for sign in germs.values()) else "exit"
        selected = next((r for r in zero_events if r["kind"] == kind), None)
        if selected is not None:
            body = deepcopy(event_dtos[selected["sequence"]-1])
            body.update(sequence=1, plane_event_sequence=selected["sequence"],
                support_indices=[r["support_index"] for r in zero_events],
                plane_event_sequences=[r["sequence"] for r in zero_events])
            occupancy.append(body)
    for root in all_events:
        if root["high"] == 0:
            continue
        others = dict(germs)
        for previous in all_events:
            if previous["high"] < root["low"]:
                others[previous["support_index"]] = 1 if previous["kind"] == "entry" else -1
        if all(sign >= 0 for support_index, sign in others.items() if support_index != root["support_index"]):
            body = deepcopy(event_dtos[root["sequence"]-1])
            body.update(sequence=len(occupancy)+1, plane_event_sequence=root["sequence"],
                support_indices=[root["support_index"]], plane_event_sequences=[root["sequence"]])
            occupancy.append(body)
    if encoded(raw["occupancy_events"]) != encoded(occupancy):
        raise ValueError("Saved material occupancy ignored a masking receiver support crossing")
    if raw["admission_authorized"] is not False or raw["global_inventory_updated"] is not False:
        raise ValueError("A point proof was misrepresented as an adopted material step")
    return dict(trajectory=trajectory, donor_rate=donor, receiver_rate=receiver, point=point,
                by_support=by_support, roots=roots, initial_boundaries=initial_boundaries,
                all_events=all_events, occupancy_events=occupancy, germs=germs,
                leaf_count=leaves, exact_coincident_support_count=identities,
                whole_interval_root_count_complete=True)


def _memory_payloads(initial, candidate, support):
    old = initial["native_underthrust_motion"]["snapshot"]
    before = {key: old[key] for key in ("states", "spatial_memory", "free_anchor_fields",
        "elapsed_time_s", "cumulative_source_work_j", "cumulative_basal_dissipation_j",
        "cumulative_bending_numerical_dissipation_j")}
    after = dict(states=candidate["states"], spatial_memory=candidate["spatial_step"]["memory"],
        free_anchor_fields=candidate["spatial_step"]["anchor_fields"], dt_s=candidate["dt_s"],
        candidate_diagnostics=candidate["diagnostics"], candidate_sha256=digest(candidate),
        whole_time_support=deepcopy(support), candidate_adopted=False)
    return before, after


def audit_query(query, instant, proof, certificate):
    """Preserve certified unknown windows instead of granting sampled occupancy."""
    instant = rational(instant)
    if not 0 <= instant <= 1:
        raise ValueError("Point query left its original complete time interval")
    signs, pending = [], []
    for index, info in proof["by_support"].items():
        sign = proof["germs"][index]
        for event in proof["all_events"]:
            if event["support_index"] != index:
                continue
            if event["low"] <= instant <= event["high"]:
                sign = None; pending.append(event["sequence"]); break
            if event["high"] < instant:
                sign = 1 if event["kind"] == "entry" else -1
        signs.append(sign)
    occupancy = "outside" if -1 in signs else "event_window" if None in signs else "inside"
    expected = dict(format="native-full-point-chronology-query-1",
        model="complete_all_receiver_halfspace_point_history",
        certificate_sha256=digest({key: value for key, value in certificate.items() if key != "payload_sha256"}),
        time_fraction=fdto(instant), time_myr=fdto(instant*rational(proof["trajectory"]["dt_myr"])),
        occupancy=occupancy, plane_signs=signs,
        active_support_indices=[index for index, sign in enumerate(signs) if sign in (0, None)],
        coincident_support_indices=[index for index, info in proof["by_support"].items() if info["coincident"]],
        event_sequence=sorted(pending), admission_authorized=False, global_inventory_updated=False)
    if encoded(query) != encoded(expected):
        raise ValueError("Saved point query changed certified occupancy or promoted an unknown event window")
    with localcontext() as context:
        context.prec = PRECISION
        for index, sign in enumerate(signs):
            if sign in (None, 0):
                continue
            normal = proof["trajectory"]["receiver_polygon"]["supports"][index]["normal"]
            value = _gap_decimal(tuple(rational(x) for x in normal), proof["point"],
                                  proof["donor_rate"], proof["receiver_rate"], instant)
            if (value > 0) != (sign > 0) or value == 0:
                raise ValueError("Independent Rodrigues point query contradicts exact complete chronology")
    return occupancy


def _ibinary(a, b, operation):
    if operation == "add":
        return a[0]+b[0], a[1]+b[1]
    if operation == "mul":
        values = tuple(x*y for x in a for y in b)
        return min(values), max(values)
    if operation == "div":
        if b[0] <= 0 <= b[1]:
            raise ValueError("Swept cone ratio lost its nonzero denominator")
        return _ibinary(a, (1/b[1], 1/b[0]), "mul")
    raise ValueError("Unknown independent interval operation")


def _ineg(a):
    return -a[1], -a[0]


def _isub(a, b):
    return _ibinary(a, _ineg(b), "add")


def _sqrt_rational(value, bits=1024):
    if value < 0:
        raise ValueError("Independent solid-angle square root has a negative radicand")
    scale = 1 << bits
    root = isqrt((value.numerator << (2*bits))//value.denominator)
    lower = F(root, scale)
    return lower, lower if lower*lower == value else F(root+1, scale)


def _sqrt_interval(value):
    return _sqrt_rational(value[0])[0], _sqrt_rational(value[1])[1]


def _atan_rational(value, count=64):
    if not 0 <= value < 1:
        raise ValueError("Independent alternating arctangent exceeded its proven domain")
    result = sum((F((-1)**k, 2*k+1)*value**(2*k+1) for k in range(count)), ZERO)
    next_term = F((-1)**count, 2*count+1)*value**(2*count+1)
    return min(result, result+next_term), max(result, result+next_term)


def _octant_area_bounds():
    # π/2 = 8 atan(1/5) - 2 atan(1/239); each alternating tail is bounded.
    return _isub(_ibinary((F(8), F(8)), _atan_rational(F(1, 5), 256), "mul"),
                 _ibinary((F(2), F(2)), _atan_rational(F(1, 239), 128), "mul"))


def _cone_area_bounds(p, q):
    # Exact Van Oosterom--Strackee solid angle for rays (0,q,-1),(0,1,0),(1,p,0).
    # tan(area/2)=1/[(sqrt(1+q²)+q)(sqrt(1+p²)+p)]. All scalars are positive.
    factors = []
    for value in (p, q):
        square = _ibinary(value, value, "mul")
        root = _sqrt_interval(_ibinary(square, (ONE, ONE), "add"))
        factors.append(_ibinary(root, value, "add"))
    denominator = _ibinary(*factors, "mul")
    tangent = _ibinary((ONE, ONE), denominator, "div")
    lower = _atan_rational(tangent[0])[0]
    upper = _atan_rational(tangent[1])[1]
    return 2*lower, 2*upper


def audit_material_partition(raw, swept_receipt, swept_raw, binding):
    parent_id = swept_receipt["donor_fragment_id"]
    parent = binding["parents"][parent_id]
    if raw["source_parent_id"] != parent_id or raw["material_id"] != parent["parcel"]["material_id"]:
        raise ValueError("Full swept material partition changed original material identity")
    source_area, area = interval(raw["source_parent_area_sr_bounds"]), interval(raw["whole_positive_area_swept_domain_sr_bounds"])
    encloses(raw["source_parent_area_sr_bounds"], _octant_area_bounds(), "literal full octant area")
    if area != interval(swept_raw["certificate"]["endpoint_area_bounds_sr"]):
        raise ValueError("Full swept material shares changed the proved native cone area")
    declared = swept_raw["certificate"]
    independent_area = _cone_area_bounds(interval(declared["terminal_p_bounds"]), interval(declared["terminal_q_bounds"]))
    encloses(raw["whole_positive_area_swept_domain_sr_bounds"], independent_area, "independent cone solid angle")
    fraction = _ibinary(area, source_area, "div")
    if interval(raw["potential_fraction_bounds"]) != fraction or not 0 < fraction[0] <= fraction[1] < 1:
        raise ValueError("Full swept material fraction is inconsistent with complete original domain areas")
    rows = {row["reservoir"]: row for row in raw["reservoirs"]}
    fields = ("area_km2", "oceanic_volume_km3", "cold_mantle_volume_km3", "density_excess_mass_kg")
    if set(rows) != set(fields) or len(raw["reservoirs"]) != 4:
        raise ValueError("Full swept material partition omitted an extensive reservoir")
    for name in fields:
        row, amount = rows[name], rational(parent["parcel"][name])
        archive = _ibinary(fraction, (amount, amount), "mul")
        live = _ibinary(_isub((ONE, ONE), fraction), (amount, amount), "mul")
        if (rational(row["original_amount"]) != amount or interval(row["potential_archived_amount_bounds"]) != archive
                or interval(row["potential_live_amount_bounds"]) != live or row["physical_allocation_adopted"] is not False):
            raise ValueError("Whole-domain reservoir complement lost authentic mass or adopted conditional allocation")
    if (raw["mechanical_work_allocated_to_2d_mass"] is not False or raw["material_allocation_adopted"] is not False
            or raw["first_entry_times_require_individual_temporal_certificates"] is not True
            or raw["physical_pairs_and_direction_require_separate_authentication"] is not True):
        raise ValueError("Potential swept mass was promoted to a once-only physical archive law")
    return dict(complete_original_parent_area_independently_proved=True,
        full_native_terminal_cone_solid_angle_independently_proved=True,
        four_extensive_reservoir_complements_authenticated=True,
        physical_mass_adopted=False)


def audit_transient_patch(raw, binding, journal):
    """Exact positive chart rectangle enters a receiver absent at the endpoint."""
    donor = binding["parents"][raw["donor_fragment_id"]]
    receiver = binding["parents"][raw["receiver_fragment_id"]]
    if (encoded(raw["donor_geometry"]) != encoded(donor["geometry"])
            or encoded(raw["receiver_geometry"]) != encoded(receiver["geometry"])
            or raw["donor_omega"] != binding["omega"][donor["parcel"]["plate"]]
            or raw["receiver_omega"] != binding["omega"][receiver["parcel"]["plate"]]
            or raw["dt_myr"] != journal["dt_myr"]):
        raise ValueError("Transient material patch changed authentic original native domains or full trajectories")
    duration, order = rational(raw["dt_myr"]), raw["order"]
    d = tuple(rational(x)*duration for x in raw["donor_omega"])
    r = tuple(rational(x)*duration for x in raw["receiver_omega"])
    matrices = matrix_series(d, r, order)
    xb, zb = interval(raw["x_bounds"]), interval(raw["z_bounds"])
    if not 0 < xb[0] < xb[1] or not 0 < zb[0] < zb[1]:
        raise ValueError("Transient material witness lost positive original two-dimensional measure")
    chart = (xb, (ONE, ONE), _ineg(zb))
    corners = {(x, ONE, -z) for x in xb for z in zb}
    actual_corners = set()
    for ray in raw["original_patch"]["vertices"]:
        if ray["kind"] != "seed" or ray["rotations"]:
            raise ValueError("Transient material witness replaced original exact chart corner rays")
        vector = tuple(rational(x) for x in ray["reference"])
        actual_corners.add(tuple(x/vector[1] for x in vector))
    if actual_corners != corners:
        raise ValueError("Transient material patch changed a full rectangle corner")
    if any(dot(corner, tuple(rational(x) for x in plane["normal"])) < 0
           for corner in corners for plane in donor["geometry"]["supports"]):
        raise ValueError("Transient material witness is outside its original full donor domain")
    instants = tuple(rational(x) for x in raw["sample_time_fractions"])
    if len(instants) != 4 or instants[0] != 0 or instants[-1] != 1 or not 0 < instants[1] < instants[2] < 1:
        raise ValueError("Transient witness chronology replaced its exact physical fractions")
    rate = sum(map(abs, d), ZERO)+sum(map(abs, r), ZERO)
    mass = xb[1]+ONE+zb[1]
    proofs = raw["full_receiver_plane_proofs"]
    if len(proofs) != len(receiver["geometry"]["supports"]):
        raise ValueError("Transient whole rectangle proof omitted an actual receiver halfspace")
    all_bounds = []
    for record, plane in zip(proofs, receiver["geometry"]["supports"]):
        if record["receiver_normal"] != plane["normal"]:
            raise ValueError("Transient patch replaced a literal receiver plane")
        normal = tuple(rational(x) for x in plane["normal"])
        coeff = []
        for matrix in matrices:
            vector = tuple(dot(row, normal) for row in matrix)
            value = (ZERO, ZERO)
            for component, coordinates in zip(vector, chart):
                value = _ibinary(value, _ibinary((component, component), coordinates, "mul"), "add")
            coeff.append(value)
        values = []
        for instant, saved in zip(instants, record["gap_bounds"]):
            value = (ZERO, ZERO)
            for component in reversed(coeff):
                value = _ibinary(_ibinary(value, (instant, instant), "mul"), component, "add")
            tail = factor_tail(rate, mass, order, 0, instant)
            value = value[0]-tail, value[1]+tail
            encloses(saved, value, "whole transient material patch gap")
            values.append(value)
        if values[1][0] <= 0:
            raise ValueError("Transient entire positive-area original patch is not inside all receiver sides")
        all_bounds.append(values)
    if not any(values[2][1] < 0 for values in all_bounds):
        raise ValueError("Transient material rectangle did not leave its receiving fragment")
    pair_ids = {raw["donor_fragment_id"], raw["receiver_fragment_id"]}
    if any({p["fragment_a"], p["fragment_b"]} == pair_ids for p in journal["endpoint_pairs"]):
        raise ValueError("Claimed transient pair is actually present in the endpoint overlap table")
    if raw["material_allocation_adopted"] is not False or raw["geometry_admission_authorized"] is not False or raw["physical_pair_ancestry_authenticated"] is not False:
        raise ValueError("Transient domain evidence inferred material ancestry or force adoption")
    area = (xb[1]-xb[0])*(zb[1]-zb[0])
    if ONE+xb[1]**2+zb[1]**2 >= 2 or interval(raw["original_patch_area_sr_bounds"])[0] <= 0:
        raise ValueError("Transient spherical chart lost its independently positive area bound")
    return dict(whole_original_rectangle_all_receiver_sides_independently_proved=True,
        positive_spherical_area_lower_bound_sr=fdto(area/4),
        positive_material_pair_absent_at_endpoint=True,
        receiver_fragment_switch_is_not_an_authenticated_new_physical_fault=True,
        physical_material_allocation_adopted=False)


def _interval_polynomial_bounds(coefficients, low, high):
    degree, width = len(coefficients)-1, high-low
    shifted = []
    for j in range(degree+1):
        value = (ZERO, ZERO)
        for k in range(j, degree+1):
            factor = comb(k, j)*low**(k-j)*width**j
            value = _ibinary(value, _ibinary(coefficients[k], (factor, factor), "mul"), "add")
        shifted.append(value)
    bernstein = []
    for i in range(degree+1):
        value = (ZERO, ZERO)
        for k in range(i+1):
            factor = F(comb(i, k), comb(degree, k))
            value = _ibinary(value, _ibinary(shifted[k], (factor, factor), "mul"), "add")
        bernstein.append(value)
    return min(x[0] for x in bernstein), max(x[1] for x in bernstein)


def audit_swept_only_material_patch(raw, binding, journal):
    """Authenticate a positive 2D owner entry/exit missing from endpoint loss."""
    donor_id, receiver_id = raw["donor_fragment_id"], raw["receiver_fragment_id"]
    donor, receiver = binding["parents"][donor_id], binding["parents"][receiver_id]
    if (encoded(raw["donor_geometry"]) != encoded(donor["geometry"])
            or encoded(raw["receiver_geometry"]) != encoded(receiver["geometry"])
            or raw["donor_omega"] != binding["omega"][donor["parcel"]["plate"]]
            or raw["receiver_omega"] != binding["omega"][receiver["parcel"]["plate"]]
            or raw["dt_myr"] != journal["dt_myr"]):
        raise ValueError("Swept-only material patch changed an authentic original full owner trajectory")
    duration, order = rational(raw["dt_myr"]), raw["order"]
    d = tuple(rational(x)*duration for x in raw["donor_omega"])
    r = tuple(rational(x)*duration for x in raw["receiver_omega"])
    matrices = matrix_series(d, r, order)
    xb, yb = interval(raw["x_bounds"]), interval(raw["y_bounds"])
    if not 0 < xb[0] < xb[1] or not 0 < yb[0] < yb[1]:
        raise ValueError("Swept-only owner entry/exit rectangle has no positive material area")
    chart = (xb, _ineg(yb), (ONE, ONE))
    corners = {(x, -y, ONE) for x in xb for y in yb}
    actual = set()
    for ray in raw["original_patch"]["vertices"]:
        if ray["kind"] != "seed" or ray["rotations"]:
            raise ValueError("Swept-only material witness replaced a literal original corner")
        vector = tuple(rational(x) for x in ray["reference"])
        actual.add(tuple(x/vector[2] for x in vector))
    if actual != corners or any(dot(corner, tuple(rational(x) for x in plane["normal"])) < 0
                                for corner in corners for plane in donor["geometry"]["supports"]):
        raise ValueError("Swept-only material rectangle lost complete original donor membership")
    instants = tuple(rational(x) for x in raw["sample_time_fractions"])
    if instants != (ZERO, F(1, 4), F(1, 2), F(3, 4), ONE):
        raise ValueError("Swept-only material witness replaced its actual crossing time probes")
    rate, mass = sum(map(abs, d), ZERO)+sum(map(abs, r), ZERO), xb[1]+yb[1]+ONE
    records = raw["full_receiver_plane_proofs"]
    if len(records) != len(receiver["geometry"]["supports"]):
        raise ValueError("Swept-only material witness omitted a full original receiver support")
    main = None
    for record, plane in zip(records, receiver["geometry"]["supports"]):
        if record["receiver_normal"] != plane["normal"]:
            raise ValueError("Swept-only witness replaced the true receiver plane lineage")
        normal = tuple(rational(x) for x in plane["normal"])
        coeff = []
        for matrix in matrices:
            vector = tuple(dot(row, normal) for row in matrix)
            value = (ZERO, ZERO)
            for component, coordinates in zip(vector, chart):
                value = _ibinary(value, _ibinary((component, component), coordinates, "mul"), "add")
            coeff.append(value)
        bounds = []
        for instant, saved in zip(instants, record["gap_bounds"]):
            value = _interval_polynomial_bounds(coeff, instant, instant)
            tail = factor_tail(rate, mass, order, 0, instant)
            value = value[0]-tail, value[1]+tail
            encloses(saved, value, "swept-only whole material rectangle gap")
            bounds.append(value)
        if bounds[2][0] <= 0:
            raise ValueError("Entire swept-only material rectangle did not enter the true receiver")
        whole = _interval_polynomial_bounds(coeff, ZERO, ONE)
        tail = factor_tail(rate, mass, order, 0, ONE)
        whole = whole[0]-tail, whole[1]+tail
        encloses(record["whole_interval_bounds"], whole, "swept-only whole time receiver support")
        if normal == (-ONE, ZERO, ZERO):
            if bounds[0][1] >= 0 or bounds[-1][1] >= 0:
                raise ValueError("Entire swept-only material rectangle did not leave the receiver by the endpoint")
            main = bounds[-1]
        elif whole[0] <= 0:
            raise ValueError("Another receiver halfspace obstructs the complete swept-only owner crossing")
    if main is None:
        raise ValueError("Swept-only owner exclusion lacks its authentic main receiver plane")
    receiving_owner = receiver["parcel"]["plate"]
    for identity, parent in binding["parents"].items():
        if parent["parcel"]["plate"] != receiving_owner:
            continue
        if not any(tuple(p["normal"]) == (-1, 0, 0) and p["rotations"] == [] for p in parent["geometry"]["supports"]):
            raise ValueError("Swept-only final owner exclusion did not cover every receiver fragment")
    pair = next((p for p in journal["endpoint_pairs"] if {p["fragment_a"], p["fragment_b"]} == {donor_id, receiver_id}), None)
    domain_key = "donor_a_original_domain" if pair and pair["fragment_a"] == donor_id else "donor_b_original_domain"
    if pair is None or encoded(raw["actual_endpoint_donor_domain"]) != encoded(pair[domain_key]):
        raise ValueError("Swept-only material counterexample replaced the actual native endpoint loss domain")
    if (raw["global_endpoint_loss_equals_complete_swept_loss"] is not False
            or raw["material_allocation_adopted"] is not False or raw["global_inventory_updated"] is not False
            or raw["geometry_admission_authorized"] is not False
            or raw["physical_first_entry_admission_requires_separate_authentication"] is not True):
        raise ValueError("Swept-only counterexample was promoted to an adopted global archive step")
    area = (xb[1]-xb[0])*(yb[1]-yb[0])
    if ONE+xb[1]**2+yb[1]**2 >= 2 or interval(raw["original_patch_area_sr_bounds"])[0] <= 0:
        raise ValueError("Swept-only spherical material witness lost positive native area")
    return dict(whole_positive_original_material_rectangle_owner_entry_then_exit_independently_proved=True,
        final_exclusion_from_every_original_receiver_owner_fragment_independently_proved=True,
        full_rectangle_disjoint_from_actual_endpoint_material_loss=True,
        raw_endpoint_donor_geometric_coverage_is_not_once_only_archive_material_coverage=True,
        positive_spherical_area_lower_bound_sr=fdto(area/4),
        global_endpoint_loss_equals_complete_swept_loss=False,
        true_once_only_archive_material_vacancy_computed=False,
        physical_first_entry_admission_authenticated=False)


def audit_swept_domain(raw, binding, journal):
    """Re-prove the whole positive-time donor cone coefficient inequalities."""
    trajectory = raw["trajectory"]
    donor_id = next((key for key, parent in binding["parents"].items()
                     if encoded(parent["geometry"]) == encoded(trajectory["donor_polygon"])), None)
    receiver_id = next((key for key, parent in binding["parents"].items()
                        if encoded(parent["geometry"]) == encoded(trajectory["receiver_polygon"])), None)
    if donor_id is None or receiver_id is None:
        raise ValueError("Full swept cone replaced its authentic original material domains")
    donor_parent, receiver_parent = binding["parents"][donor_id], binding["parents"][receiver_id]
    if (trajectory["donor_omega"] != binding["omega"][donor_parent["parcel"]["plate"]]
            or trajectory["receiver_omega"] != binding["omega"][receiver_parent["parcel"]["plate"]]
            or trajectory["dt_myr"] != journal["dt_myr"]):
        raise ValueError("Full swept cone projected an original full owner trajectory")
    duration, order = rational(trajectory["dt_myr"]), raw["order"]
    donor = tuple(rational(x)*duration for x in trajectory["donor_omega"])
    receiver = tuple(rational(x)*duration for x in trajectory["receiver_omega"])
    matrix = matrix_series(donor, receiver, order)
    vectors = tuple(tuple(-row[0] for row in m) for m in matrix)
    c = tuple(-v[0] for v in vectors)
    a, b = tuple(v[1] for v in vectors), tuple(-v[2] for v in vectors)
    if c[0] != 1 or c[1] != 0 or a[0] or b[0]:
        raise ValueError("Full swept cone lost its exact original axis incidence")
    rate = sum(map(abs, donor), ZERO)+sum(map(abs, receiver), ZERO)
    def bound(values, factor, low=ZERO, high=ONE, derivative=0):
        return factored_bounds(values, rate, ONE, factor, low, high, derivative)
    cb, csecond, yprime = bound(c, 0), bound(c, 0, derivative=2), bound(a, 0, derivative=1)
    ab, bb = bound(a, 1), bound(b, 1)
    if cb[0] <= 0 or csecond[1] >= 0 or yprime[0] <= 0 or ab[0] <= 0 or bb[1] >= 0:
        raise ValueError("Independent full swept cone coefficient signs are not proved")
    derivative = _isub(_ibinary(ab, bound(b, 1, derivative=2), "mul"),
                      _ibinary(bound(a, 1, derivative=2), bb, "mul"))
    def turning(low, high):
        return _isub(_ibinary(bound(a, 1, low, high), bound(b, 1, low, high, 1), "mul"),
                      _ibinary(bound(a, 1, low, high, 1), bound(b, 1, low, high), "mul"))
    first, last = turning(ZERO, ZERO), turning(ONE, ONE)
    if derivative[0] <= 0 or first[1] >= 0 or last[0] <= 0:
        raise ValueError("Independent whole swept cone ratio lacks its unique maximum proof")
    q0 = _ibinary(_ineg(bound(b, 1, ZERO, ZERO)), bound(a, 1, ZERO, ZERO), "div")
    q1 = _ibinary(_ineg(bound(b, 1, ONE, ONE)), bound(a, 1, ONE, ONE), "div")
    delta = _isub(q0, q1)
    p1 = _ibinary(bound(c, 0, ONE, ONE), bound(a, 1, ONE, ONE), "div")
    if delta[0] <= 0 or q1[0] <= 0 or p1[0] <= 0:
        raise ValueError("Independent endpoint is not the complete swept cone minimum")
    declared = raw["certificate"]
    for name, value in (("c_bounds", cb), ("c_second_derivative_bounds", csecond),
                        ("main_y_first_derivative_bounds", yprime), ("a_over_u_bounds", ab),
                        ("b_over_u_bounds", bb), ("turning_derivative_bounds", derivative),
                        ("initial_turning_bounds", first), ("terminal_turning_bounds", last),
                        ("initial_q_bounds", q0), ("terminal_q_bounds", q1),
                        ("initial_minus_terminal_q_bounds", delta), ("terminal_p_bounds", p1)):
        encloses(declared[name], value, "full swept cone " + name)
    low, high = interval(declared["ratio_turning_fraction"])
    if not 0 < low < high < 1 or turning(low, low)[1] >= 0 or turning(high, high)[0] <= 0:
        raise ValueError("Swept cone ratio turning window lacks strict independent bracketing")
    matched = next((pair for pair in journal["endpoint_pairs"]
                    if {pair["fragment_a"], pair["fragment_b"]} == {donor_id, receiver_id}), None)
    if matched is None or encoded(raw["endpoint_world_overlap"]) != encoded(matched["overlap_geometry"]):
        raise ValueError("Full swept cone does not bind the actual complete endpoint overlap")
    pullback_key = "donor_a_original_domain" if matched["fragment_a"] == donor_id else "donor_b_original_domain"
    if encoded(raw["endpoint_donor_domain"]) != encoded(matched[pullback_key]):
        raise ValueError("Full swept cone lost its original donor pullback")
    cone, endpoint = raw["terminal_native_cone"], raw["endpoint_donor_domain"]
    expected_planes = {
        ((ONE, ZERO, ZERO), ()), ((ZERO, ZERO, -ONE), ()),
        ((-ONE, ZERO, ZERO), _effective_word([
            dict(omega_rad_per_myr=trajectory["receiver_omega"], dt_myr=trajectory["dt_myr"]),
            dict(omega_rad_per_myr=[-x for x in trajectory["donor_omega"]], dt_myr=trajectory["dt_myr"])],
            (-ONE, ZERO, ZERO)))}
    if ({_plane_key(p, cone["pose"]) for p in cone["supports"]} != expected_planes
            or {_plane_key(p, endpoint["pose"]) for p in endpoint["supports"]} != expected_planes
            or {_ray_key(v, cone["pose"]) for v in cone["vertices"]} !=
            {_ray_key(v, endpoint["pose"]) for v in endpoint["vertices"]}):
        raise ValueError("Independent exact full-axis plane/ray lineage did not establish native cone equality")
    if (raw["initial_boundary"]["separate_from_positive_time_swept_domain"] is not True
            or raw["initial_boundary"]["discarded"] is not False
            or raw["physical_archive_allocation_adopted"] is not False
            or raw["global_inventory_updated"] is not False or raw["admission_authorized"] is not False):
        raise ValueError("A swept footprint discarded zero-time boundary or adopted passive archive mass")
    with localcontext() as context:
        context.prec = PRECISION
        _, vertices, cone_point = _polygon_world(raw["terminal_native_cone"])
        endpoint_normals, _, _ = _polygon_world(raw["endpoint_donor_domain"])
        if any(ddot(cone_point, n) <= 0 for n in endpoint_normals):
            raise ValueError("Independent terminal cone is outside its actual native endpoint")
        for side in declared["terminal_other_halfspace_vertex_bounds"]:
            normal = dict(normal=side["normal"], rotations=[
                dict(omega_rad_per_myr=trajectory["receiver_omega"], dt_myr=trajectory["dt_myr"]),
                dict(omega_rad_per_myr=[-x for x in trajectory["donor_omega"]], dt_myr=trajectory["dt_myr"])])
            moving = _plane_world(normal)
            if len(side["vertices"]) != len(vertices) or any(interval(saved)[0] <= 0 for saved in side["vertices"]):
                raise ValueError("Terminal cone omitted strictly containing receiver supports")
            if any(ddot(vertex, moving) <= 0 for vertex in vertices):
                raise ValueError("Another original receiver halfspace excludes the full terminal cone")
    return dict(donor_fragment_id=donor_id, receiver_fragment_id=receiver_id,
        complete_positive_time_swept_coefficient_inequalities_independently_proved=True,
        analytic_terminal_cone_containment_independently_proved=True,
        native_cone_set_identity_independently_exact_recomputed=True,
        native_endpoint_lineage_authenticated_and_numerical_containment_replayed=True,
        initial_one_dimensional_boundary_retained=True,
        physical_archive_mass_adopted=False)


def audit_source_binding(journal, original, initial, candidate, support):
    """Authenticate all original material and unadopted mechanical references."""
    if journal.get("journal_sha256") != digest({key: value for key, value in journal.items() if key != "journal_sha256"}):
        raise ValueError("Saved whole-domain journal integrity seal is inconsistent")
    if encoded(original) != encoded(initial["current_surface"]) or encoded(journal["source_surface"]) != encoded(original):
        raise ValueError("Full-domain journal counterfeited the authentic original material payload")
    if journal["source_surface_sha256"] != digest(original):
        raise ValueError("Whole-domain source digest does not bind the original payload")
    if encoded(journal["omega_rad_per_myr"]) != encoded(candidate["omega_rad_per_myr"]):
        raise ValueError("Whole-domain journal projected or changed an original owner Euler vector")
    if journal["dt_myr"] != candidate["admission"]["dt_myr"] or rational(journal["exact_interval_myr"]) != rational(journal["dt_myr"]):
        raise ValueError("Whole-domain interval changed the unaccepted physical candidate duration")
    before, after = _memory_payloads(initial, candidate, support)
    expected_memory = dict(before_payload=before, candidate_after_payload=after,
        before_sha256=digest(before), candidate_after_sha256=digest(after),
        constitutive_path_sha256=support["path_sha256"])
    if encoded(journal["memory_reference"]) != encoded(expected_memory):
        raise ValueError("Whole-domain journal counterfeited the authentic external mechanical memory")
    for name in ("candidate_mechanics_adopted", "global_inventory_updated", "admission_authorized",
                 "original_material_allocation_adopted", "material_loss_decided",
                 "endpoint_pair_direction_resolved", "newborn_material_created",
                 "raw_gap_equals_passive_archive_material_vacancy_proved"):
        if journal[name] is not False:
            raise ValueError("Unaccepted domain evidence was misrepresented as an adopted force/material law")
    parents = {row["fragment_id"]: row for row in original["fragments"]}
    omega = dict(candidate["omega_rad_per_myr"])
    if len(parents) != len(original["fragments"]) or set(omega) != {p["parcel"]["plate"] for p in parents.values()}:
        raise ValueError("Whole-domain source lost an original fragment or owner")
    moved = {row["fragment_id"]: row for row in journal["moved_fragments"]}
    domains = {row["fragment_id"]: row for row in journal["whole_original_parent_domains"]}
    if set(moved) != set(parents) or len(moved) != len(journal["moved_fragments"]) or set(domains) != set(parents):
        raise ValueError("Whole-domain journal omitted an entire original material domain")
    reservoirs = ("area_km2", "oceanic_volume_km3", "cold_mantle_volume_km3", "density_excess_mass_kg")
    for identity, parent in parents.items():
        row, domain, parcel = moved[identity], domains[identity], parent["parcel"]
        expected_geometry = deepcopy(parent["geometry"])
        expected_geometry["pose"].append(dict(omega_rad_per_myr=omega[parcel["plate"]], dt_myr=journal["dt_myr"]))
        if encoded(row["geometry"]) != encoded(expected_geometry) or row["owner"] != parcel["plate"] or row["material_id"] != parcel["material_id"]:
            raise ValueError("Moved domain changed literal geometry, full owner rotation, or material identity")
        if row["parent_payload_sha256"] != digest(parent) or domain["parent_payload_sha256"] != digest(parent):
            raise ValueError("Domain lineage no longer binds every original parcel field")
        if rational(row["material_age_at_endpoint_myr"]) != rational(parcel["age_myr"])+rational(journal["dt_myr"]):
            raise ValueError("Original material age no longer follows the physical interval")
        if (domain["original_geometry_sha256"] != digest(parent["geometry"])
                or domain["owner"] != parcel["plate"] or domain["material_id"] != parcel["material_id"]
                or rational(domain["original_age_myr"]) != rational(parcel["age_myr"])
                or encoded(domain["material_fields"]) != encoded(parcel["material_fields"])
                or encoded(domain["specific_properties"]) != encoded(parcel["specific_properties"])):
            raise ValueError("Conditional original-domain definitions counterfeited material history")
        amounts = {item["reservoir"]: item for item in domain["original_reservoirs"]}
        if set(amounts) != set(reservoirs):
            raise ValueError("An original extensive reservoir was omitted")
        for field in reservoirs:
            item, amount = amounts[field], rational(parcel[field])
            if (rational(item["original_amount"]) != amount
                    or interval(item["each_conditional_amount_bounds"]) != (ZERO, amount)
                    or item["material_allocation_adopted"] is not False):
                raise ValueError("Conditional reservoir conservation changed its authentic original amount")
        expected_receivers = {key for key, p in parents.items() if p["parcel"]["plate"] != parcel["plate"]}
        contacts = domain["potential_contact_domains"]
        if {c["receiver_fragment_id"] for c in contacts} != expected_receivers or len(contacts) != len(expected_receivers):
            raise ValueError("Whole-domain definition omitted a possible original receiver domain")
        for contact in contacts:
            receiver = parents[contact["receiver_fragment_id"]]
            if contact["receiver_payload_sha256"] != digest(receiver) or contact["receiver_owner"] != receiver["parcel"]["plate"] or contact["physical_pair_and_direction_authenticated"] is not False:
                raise ValueError("Conditional contact reference counterfeited receiver material/direction")
    return dict(parents=parents, moved=moved, omega=omega, domains=domains)


def _decimal_vector(raw):
    vector = tuple(decimal(x) for x in raw)
    scale = max(map(abs, vector))
    if not scale:
        raise ValueError("Native analytic geometry contains a zero direction")
    return tuple(x/scale for x in vector)


def _history_vector(vector, history):
    for pose in history:
        rate = tuple(decimal(x)*decimal(pose["dt_myr"]) for x in pose["omega_rad_per_myr"])
        vector = rotate(vector, rate)
    return vector


def _plane_world(raw, pose=()):
    return _history_vector(_history_vector(_decimal_vector(raw["normal"]), raw["rotations"]), pose)


def _ray_world(raw, pose=()):
    if raw["kind"] == "seed":
        vector = _history_vector(_decimal_vector(raw["reference"]), raw["rotations"])
    elif raw["kind"] == "cross":
        vector = tuple(Decimal(raw["sign"])*x for x in cross(*tuple(_plane_world(p) for p in raw["planes"])))
    else:
        raise ValueError("Unsupported native ray lineage")
    return _history_vector(vector, pose)


def _polygon_world(raw):
    normals = tuple(_plane_world(s, raw["pose"]) for s in raw["supports"])
    vertices = tuple(_ray_world(v, raw["pose"]) for v in raw["vertices"])
    if len(vertices) < 3 or len(vertices) != len(normals):
        raise ValueError("Native domain is missing positive cyclic geometry")
    interior = tuple(sum((v[i]/max(map(abs, v)) for v in vertices), Decimal(0)) for i in range(3))
    if any(ddot(interior, normal) <= 0 for normal in normals):
        raise ValueError("Independent native endpoint interior is not strictly positive")
    tolerance = Decimal("1e-430")
    if any(ddot(vertex, normal) < -tolerance*max(map(abs, vertex))*max(map(abs, normal))
           for vertex in vertices for normal in normals):
        raise ValueError("Native endpoint ray lies outside an authoritative inherited support")
    return normals, vertices, interior


def _dnorm(vector):
    scale = max(map(abs, vector))
    if not scale:
        raise ValueError("Independent numerical polygon produced a zero ray")
    return tuple(x/scale for x in vector)


def _dclip(vertices, normal):
    """Independent numerical great-circle clipping; no saved cut rays used."""
    tolerance = Decimal("1e-430")
    if not vertices:
        return ()
    vertices, normal = tuple(_dnorm(v) for v in vertices), _dnorm(normal)
    def side(vertex):
        value = ddot(vertex, normal)
        return 1 if value > tolerance else -1 if value < -tolerance else 0
    result = []
    previous, previous_side = vertices[-1], side(vertices[-1])
    for current in vertices:
        current_side = side(current)
        if previous_side*current_side == -1:
            edge = cross(previous, current)
            point = cross(edge, normal)
            if ddot(point, tuple(x+y for x, y in zip(previous, current))) < 0:
                point = tuple(-x for x in point)
            result.append(_dnorm(point))
        if current_side >= 0:
            result.append(current)
        previous, previous_side = current, current_side
    unique = []
    for point in result:
        if not unique or max(abs(x-y) for x, y in zip(point, unique[-1])) > tolerance:
            unique.append(point)
    if len(unique) > 1 and max(abs(x-y) for x, y in zip(unique[0], unique[-1])) <= tolerance:
        unique.pop()
    return tuple(unique) if len(unique) >= 3 else ()


def _dsubtract(vertices, normals):
    parts, remainder = [], vertices
    for normal in normals:
        outside = _dclip(remainder, tuple(-x for x in normal))
        if outside:
            parts.append(outside)
        remainder = _dclip(remainder, normal)
        if not remainder:
            break
    return tuple(parts)


def _same_numerical_vertices(first, second):
    tolerance = Decimal("1e-420")
    first, second = tuple(_dnorm(v) for v in first), tuple(_dnorm(v) for v in second)
    return (len(first) == len(second) and all(any(max(abs(x-y) for x, y in zip(a, b)) < tolerance
                                                for b in second) for a in first))


def _effective_word(history, reference):
    """Only exact inverse cancellation and exact axis invariance are reduced."""
    word = []
    for step in history:
        axis = tuple(rational(x) for x in step["omega_rad_per_myr"])
        duration = rational(step["dt_myr"])
        if not any(axis) or not duration:
            continue
        value = (axis, duration)
        if word and word[-1] == (tuple(-x for x in axis), duration):
            word.pop()
        else:
            word.append(value)
    if all(not any(cross(reference, axis)) for axis, _ in word):
        return ()
    return tuple(word)


def _plane_key(plane, pose=()):
    reference = normalize_scale(plane["normal"])
    return reference, _effective_word(tuple(plane["rotations"])+tuple(pose), reference)


def _ray_key(ray, pose=()):
    if ray["kind"] == "seed":
        reference = normalize_scale(ray["reference"])
        return "seed", reference, _effective_word(tuple(ray["rotations"])+tuple(pose), reference)
    first, second = tuple(_plane_key(plane, pose) for plane in ray["planes"])
    if first[1] == second[1]:
        reference = normalize_scale(tuple(ray["sign"]*x for x in cross(first[0], second[0])))
        return "seed", reference, _effective_word([
            dict(omega_rad_per_myr=list(axis), dt_myr=duration) for axis, duration in first[1]], reference)
    polarity, keys = ray["sign"], []
    for normal, word in (first, second):
        sign = next(1 if x > 0 else -1 for x in normal if x)
        keys.append((tuple(sign*x for x in normal), word))
        polarity *= sign
    if keys[0] > keys[1]:
        keys.reverse(); polarity *= -1
    return "cross", tuple(keys), polarity


def audit_endpoint_geometry(journal, binding, candidate):
    """Independent Rodrigues geometry checks with explicit numerical scope."""
    with localcontext() as context:
        context.prec = PRECISION
        moved = {key: _polygon_world(row["geometry"])[0] for key, row in binding["moved"].items()}
        independently_detected_pairs = set()
        moved_items = list(binding["moved"].items())
        for index, (identity, first) in enumerate(moved_items):
            for other_id, second in moved_items[index+1:]:
                if first["owner"] == second["owner"]:
                    continue
                vertices = _polygon_world(first["geometry"])[1]
                for normal in moved[other_id]:
                    vertices = _dclip(vertices, normal)
                    if not vertices:
                        break
                if vertices:
                    independently_detected_pairs.add((identity, other_id))
        pairs = journal["endpoint_pairs"]
        expected = {(row["fragment_a"], row["fragment_b"]): row
                    for row in candidate["admission"]["actual_overlap_assessments"]}
        if {(row["fragment_a"], row["fragment_b"]) for row in pairs} != set(expected) or len(pairs) != len(expected):
            raise ValueError("Whole-domain journal omitted an original native endpoint overlap")
        if independently_detected_pairs != set(expected):
            raise ValueError("Independent complete-source numerical endpoint intersection found a missing pair")
        for pair in pairs:
            key = pair["fragment_a"], pair["fragment_b"]
            low, high = interval(pair["area_bounds_sr"])
            if not 0 < low <= high or pair["direction_status"] != "unresolved_both_potential_original_material_directions":
                raise ValueError("Endpoint overlap erased positive material area or invented its direction")
            geometry = pair["overlap_geometry"]
            if encoded(geometry) != encoded(expected[key]["native_overlap"]):
                raise ValueError("Endpoint overlap replaced the authentic complete force source geometry")
            _, vertices, interior = _polygon_world(geometry)
            for identity, owner_name, pullback_name in ((key[0], "owner_a", "donor_a_original_domain"),
                                                       (key[1], "owner_b", "donor_b_original_domain")):
                owner = binding["parents"][identity]["parcel"]["plate"]
                if pair[owner_name] != owner or any(ddot(interior, n) <= 0 for n in moved[identity]):
                    raise ValueError("Endpoint overlap lost an original owning footprint")
                pullback = deepcopy(geometry)
                pullback["pose"].append(dict(omega_rad_per_myr=[-x for x in binding["omega"][owner]], dt_myr=journal["dt_myr"]))
                if encoded(pair[pullback_name]) != encoded(pullback):
                    raise ValueError("Endpoint donor pullback discarded the full inverse owner rotation")
                parent_normals = _polygon_world(binding["parents"][identity]["geometry"])[0]
                _, _, material_point = _polygon_world(pair[pullback_name])
                if any(ddot(material_point, n) <= 0 for n in parent_normals):
                    raise ValueError("Potential donor domain is outside its original material footprint")
        reservations = {row["vacancy_sha256"]: row for row in journal["endpoint_hot_birth_reservations"]}
        counts = {identity: 0 for identity in binding["parents"]}
        numeric_vacancies = {}
        for identity, parent in binding["parents"].items():
            pieces = (_polygon_world(parent["geometry"])[1],)
            for normals in moved.values():
                pieces = tuple(part for piece in pieces for part in _dsubtract(piece, normals))
            numeric_vacancies[identity] = pieces
        saved_vertices = {identity: [] for identity in counts}
        for vacancy in journal["raw_vacancies"]:
            identity = vacancy["fixed_original_parent_id"]
            if identity not in counts or vacancy["vacancy_kind"] != "raw_kinematic_gap_not_passive_archive_material_vacancy":
                raise ValueError("Raw vacancy dropped its fixed original footprint or became an archive vacancy")
            if interval(vacancy["area_bounds_sr"])[0] <= 0:
                raise ValueError("A positive native vacancy was discarded or fabricated")
            _, vertices, interior = _polygon_world(vacancy["geometry"])
            saved_vertices[identity].append(vertices)
            fixed_normals = _polygon_world(binding["parents"][identity]["geometry"])[0]
            if any(ddot(interior, n) <= 0 for n in fixed_normals):
                raise ValueError("Raw vacancy moved its original fixed world footprint")
            if any(all(ddot(interior, n) >= 0 for n in normals) for normals in moved.values()):
                raise ValueError("Raw vacancy is still covered by an original moved domain")
            counts[identity] += 1
            reservation = reservations.get(vacancy["vacancy_sha256"])
            if reservation is None or reservation["actual_retained_surface_available"] is not False or reservation["newborn_parcel_created"] is not False or reservation["continuous_birth_or_newborn_motion_defined"] is not False:
                raise ValueError("A raw endpoint gap was presented as an adopted continuous birth law")
        if len(reservations) != len(journal["raw_vacancies"]):
            raise ValueError("Raw endpoint vacancy reservations omitted domain evidence")
        for identity, actual in numeric_vacancies.items():
            persisted = saved_vertices[identity]
            if len(actual) != len(persisted) or not all(any(_same_numerical_vertices(a, b) for b in persisted) for a in actual):
                raise ValueError("Independent all-source numerical subtraction found missing raw vacancy coverage")
    return dict(endpoint_pair_count=len(pairs), raw_vacancy_count=sum(counts.values()),
        raw_vacancy_counts_by_fixed_parent=counts,
        native_positive_interiors_independently_replayed=True,
        authentic_overlap_payloads_and_full_inverse_pullbacks_bound=True,
        all_endpoint_pairs_numerically_recomputed_from_all_original_domains=True,
        all_raw_vacancy_domains_numerically_reconstructed_against_all_original_moved_footprints=True,
        exact_native_set_partition_independently_recomputed=False,
        numerical_geometry_precision_digits=PRECISION,
        numerical_geometry_limit="Native endpoint interiors/rays and all raw vacancy polygons independently reconstructed with 500-digit Rodrigues and great-circle clipping, using a 1e-430 numerical incidence threshold and 1e-420 vertex comparison. Saved tiny positive rational area and authentic support payloads retained. This is numerical spherical Boolean set checking, not independent exact area/partition certification.")


def audit(control):
    started = time.perf_counter()
    auditor_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    control = Path(control)
    def read(name):
        return json.loads((control/name).read_bytes())
    original, journal = read("original_source_surface.json"), read("full_domain_journal.json")
    initial, candidate, support = (read("original_force/"+name+".json")
                                   for name in ("initial", "force_candidate", "constitutive_time_support"))
    binding = audit_source_binding(journal, original, initial, candidate, support)
    geometry = audit_endpoint_geometry(journal, binding, candidate)
    certificates = read("point_certificates.json")
    labels, results, raw_by_label = set(), {}, {}
    for record in certificates:
        label = record["label"]
        if label in labels:
            raise ValueError("Complete point evidence repeats a query label")
        labels.add(label)
        receiver = binding["parents"][record["receiver_fragment_id"]]
        donor_id = record["donor_fragment_id"]
        donor = binding["parents"][donor_id] if donor_id is not None else None
        expected = dict(donor_polygon=None if donor is None else donor["geometry"],
            receiver_polygon=receiver["geometry"], donor_omega=[0., 0., 0.] if donor is None else binding["omega"][donor["parcel"]["plate"]],
            receiver_omega=binding["omega"][receiver["parcel"]["plate"]],
            dt_myr=journal["dt_myr"], point=record["certificate"]["trajectory"]["point"])
        if record["kind"] != ("fixed_world" if donor is None else "donor"):
            raise ValueError("Point evidence changed original material/world trajectory kind")
        results[label] = audit_certificate(record["certificate"], expected_trajectory=expected)
        raw_by_label[label] = record["certificate"]
    queries = read("point_queries.json")
    occupancy_counts = {"inside": 0, "outside": 0, "event_window": 0}
    for record in queries:
        label = record["label"]
        if label not in results:
            raise ValueError("Point query refers to an unauthenticated trajectory")
        occupancy = audit_query(record["query"], record["time_fraction"], results[label], raw_by_label[label])
        occupancy_counts[occupancy] += 1
        if "original_material_age_myr" in record:
            donor_id = next(row["donor_fragment_id"] for row in certificates if row["label"] == label)
            parent = binding["parents"][donor_id]
            age = rational(parent["parcel"]["age_myr"])+rational(record["time_fraction"])*rational(journal["dt_myr"])
            if rational(record["original_material_age_myr"]) != age:
                raise ValueError("Point query altered original material age or reassigned a birth clock")
    if {r["label"] for r in queries} != labels:
        raise ValueError("A saved full-point executable is missing its actual queries")
    swept_raw = read("full_swept_domain.json")
    swept = audit_swept_domain(swept_raw, binding, journal)
    partition = audit_material_partition(read("full_donor_material_partition.json"), swept, swept_raw, binding)
    transient = audit_transient_patch(read("transient_receiver_patch.json"), binding, journal)
    swept_only = audit_swept_only_material_patch(read("swept_only_material_patch.json"), binding, journal)
    if auditor_sha256 != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("Independent auditor source changed during proof replay")
    return dict(format="independent-native-full-source-domain-audit-1", status="PASS",
        elapsed_seconds=time.perf_counter()-started, source_parent_count=len(binding["parents"]),
        full_original_owner_vectors_preserved=True, all_original_parcel_payloads_authenticated=True,
        original_unaccepted_mechanical_memory_references_authenticated=True,
        point_certificate_count=len(results), independently_verified_proof_leaf_count=sum(r["leaf_count"] for r in results.values()),
        certified_point_root_count=sum(len(r["roots"]) for r in results.values()),
        exact_coincident_support_count=sum(r["exact_coincident_support_count"] for r in results.values()),
        point_query_count=len(queries), point_query_occupancy_counts=occupancy_counts,
        point_chronology_summary=[dict(label=label, proof_leaf_count=proof["leaf_count"],
            nonzero_plane_root_count=len(proof["roots"]),
            initial_boundary_event_count=len(proof["initial_boundaries"]),
            true_receiver_occupancy_event_count=len(proof["occupancy_events"]),
            masked_plane_event_count=len(proof["all_events"])-len({sequence
                for event in proof["occupancy_events"] for sequence in event["plane_event_sequences"]}),
            exact_coincident_support_count=proof["exact_coincident_support_count"])
            for label, proof in results.items()],
        event_windows_preserved_without_sampled_occupancy_grants=True,
        masked_raw_support_roots_independently_resolved=True,
        every_saved_point_full_interval_root_count_complete=True,
        complete_all_original_material_points_certified=False,
        force_solution_independently_recomputed=False, accepted_loaded_native_intervals=0,
        candidate_memory_adopted=False, admission_authorized=False,
        native_endpoint_geometry=geometry,
        full_blocking_pair_positive_time_swept_footprint=swept,
        full_blocking_pair_conditional_material_partition=partition,
        positive_area_transient_receiver_patch=transient,
        positive_area_owner_entry_exit_material_missing_endpoint_loss=swept_only,
        limitation="Independent exact proofs concern the executable saved individual point queries. They do not certify every original material point, integrate swept archive loss or adopt a force, memory, material direction or continuous hot-birth law.",
        input_artifact_sha256={str(path.relative_to(control)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(control.rglob("*.json"))},
        auditor_sha256=auditor_sha256, auditor_source_unchanged=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Independent domain receipt requires a new output path")
    result = audit(args.control)
    result["receipt_sha256"] = digest(result)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: result[key] for key in ("status", "elapsed_seconds", "source_parent_count",
        "point_certificate_count", "independently_verified_proof_leaf_count", "certified_point_root_count")}, indent=2))
