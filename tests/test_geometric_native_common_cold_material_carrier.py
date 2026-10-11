"""Independent checks of the whole cold carrier and its original Cartesian chart.

All authority data come from the saved full packets.  The oracle below writes
the roof, physical-depth rotation, material clamp and modal derivatives in the
original Cartesian chart directly; it does not call an old force solver or
reuse the new field derivative implementation.
"""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as F
from functools import lru_cache
from math import factorial
import hashlib
import json

import pytest

from tectonics.geometric_native_pose import FractionInterval as I, interval_sqrt, interval_atan
from tectonics.geometric_native_spherical_solid import interval_cbrt


@lru_cache(maxsize=1)
def _api():
    from tectonics import geometric_native_common_cold_material_carrier as api
    return api


@lru_cache(maxsize=1)
def _native():
    from analysis.geometric_native_common_cold_material_carrier_validation import native_control
    return native_control


def _fraction(value):
    assert type(value) is list and len(value) == 2 and all(type(x) is int for x in value)
    return F(*value)


def _interval(value):
    assert set(value) == {"lo", "hi"}
    return I(_fraction(value["lo"]), _fraction(value["hi"]))


def _decode(value):
    if type(value) is dict and set(value) == {"lo", "hi"}:
        return _interval(value)
    if type(value) is list:
        return tuple(_decode(x) for x in value)
    return value


def _dot(a, b):
    return sum((x*y for x, y in zip(a, b)), I(0))


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _mv(a, b):
    return tuple(_dot(row, b) for row in a)


def _mm(a, b):
    return tuple(tuple(_dot(row, col) for col in zip(*b)) for row in a)


def _outer(a, b):
    return tuple(tuple(x*y for y in b) for x in a)


IDENTITY = tuple(tuple(I(int(i == j)) for j in range(3)) for i in range(3))
GENERATORS = (
    ((1, 0, 0), (0, 0, 0), (0, 0, 0)),
    ((0, 0, 0), (0, 1, 0), (0, 0, 0)),
    ((0, 0, 0), (0, 0, 0), (0, 0, 1)),
    ((0, 1, 0), (1, 0, 0), (0, 0, 0)),
    ((0, 0, 1), (0, 0, 0), (1, 0, 0)),
    ((0, 0, 0), (0, 0, 1), (0, 1, 0)),
)


def _overlap(a, b):
    if type(a) is I:
        b = b if type(b) is I else I(b)
        assert max(a.lo, b.lo) <= min(a.hi, b.hi), (a, b)
    else:
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _overlap(x, y)


def _contains_zero(value):
    if type(value) is I:
        assert value.lo <= 0 <= value.hi, value
    else:
        for child in value:
            _contains_zero(child)


def _rounded(value, bits=190):
    denominator = 1 << bits
    return I(F((value.lo*denominator).__floor__(), denominator),
             F((value.hi*denominator).__ceil__(), denominator))


def _rotation(rate, depth):
    """Independent interval exponential with a uniform matrix-series tail."""
    a = tuple(_rounded(I(x)*depth) for x in rate)
    skew = ((I(0), -a[2], a[1]), (a[2], I(0), -a[0]), (-a[1], a[0], I(0)))
    term = IDENTITY
    total = IDENTITY
    for n in range(1, 37):
        term = tuple(tuple(_rounded(x/n) for x in row) for row in _mm(skew, term))
        total = tuple(tuple(_rounded(x+y) for x, y in zip(row, add))
                      for row, add in zip(total, term))
    magnitude = sum((max(abs(x.lo), abs(x.hi)) for x in a), F(0))
    assert magnitude < F(1, 4)
    remainder = 2*magnitude**37/F(factorial(37))
    return tuple(tuple(x+I(-remainder, remainder) for x in row) for row in total)


@lru_cache(maxsize=1)
def _authorities():
    transport, source_work = _native().source_packets()
    bodies = {b["original_body"]["original_parent_id"]: b for b in transport["original_cold_body_contexts"]}
    assert set(bodies) == {"octant:"+str(i) for i in range(8)}
    system = source_work["carrier_source_work_system"]
    values = tuple(_fraction(x) for x in system["bound_coordinate_increments"])
    order = tuple(tuple(x) for x in system["coordinate_order"])
    coordinates = {
        parent: tuple(values[order.index(("body", b["original_body"]["component_id"], name))]
                      for name in b["original_body"]["coordinate_order"])
        for parent, b in bodies.items()
    }
    return transport, source_work, bodies, coordinates


def _bank_oracle(parent, ray, depth):
    """Seven exact declared modes and full derivatives d/dX_original."""
    _, _, bodies, coefficients = _authorities()
    context = bodies[parent]
    body, bank = context["original_body"], context["embedding_bank"]
    radius = _fraction(bank["radius_m"])
    roof = _fraction(bank["roof_radial_displacement_m"])
    rate = tuple(_fraction(x) for x in bank["angular_rate_per_physical_volume_depth_m"])
    Q = _decode(body["owner_rotation_matrix_bounds"])
    norm = interval_sqrt(I(sum((x*x for x in ray), F(0))), 115)
    q = tuple(I(x)/norm for x in ray)
    r0 = interval_cbrt(I(radius**3)-3*radius**2*depth, 115)
    old_cubes = I((radius+roof)**3)-3*radius**2*depth
    rold = interval_cbrt(old_cubes, 115)
    tau = (I(radius**3)-old_cubes)/(3*radius**2)
    S = _rotation(rate, tau)
    SQ = _mm(S, Q)
    e = _mv(SQ, q)
    X = tuple(rold*x for x in e)
    C = tuple(tuple(rold/r0*(IDENTITY[i][j]-q[i]*q[j])
                    +r0*r0/(rold*rold)*q[i]*q[j] for j in range(3)) for i in range(3))
    posed_q = _mv(Q, q)
    shear = _cross(tuple(I(x) for x in rate), posed_q)
    rank = tuple(tuple(IDENTITY[i][j]-old_cubes/radius**2*shear[i]*posed_q[j]
                       for j in range(3)) for i in range(3))
    B = _mm(S, rank)
    Dphi = _mm(_mm(B, Q), C)
    axes = tuple(tuple(_fraction(x) for x in body[name])
                 for name in ("original_quarter_start", "original_quarter_end", "original_quarter_inward"))
    labels = tuple(_dot(axis, q) for axis in axes)
    horizontal = interval_sqrt(labels[0]*labels[0]+labels[1]*labels[1], 115)
    cut = _fraction(body["beta_cut_radians"])
    if horizontal.hi == 0:
        active, beta = False, None
    else:
        beta = interval_atan(labels[2]/horizontal, 115)
        assert not beta.lo < cut < beta.hi
        active = beta.hi < cut
    if not active:
        return dict(position=X, reference_gradient=Dphi, roof=C, modes=((I(0),)*3,)*7,
                    gradients=(((I(0),)*3,)*3,)*7, displacement=(I(0),)*3)
    t = beta/cut
    chi = (I(1)-t)*(I(1)-t)*(t+2)/2
    slope = -F(3, 2)*(I(1)-t)*(I(1)+t)/cut
    dbeta = tuple((I(axis)-labels[2]*qi)/(r0*horizontal)
                  for axis, qi in zip(axes[2], q))
    dchi = tuple(slope*x for x in dbeta)
    projected = tuple(tuple(IDENTITY[i][j]-e[i]*e[j] for j in range(3)) for i in range(3))
    de = tuple(tuple(x/rold for x in row) for row in _mm(projected, Dphi))
    modes = [tuple(chi*x for x in e)]
    gradients = [tuple(tuple(chi*de[i][j]+e[i]*dchi[j] for j in range(3)) for i in range(3))]
    for E in GENERATORS:
        direction = tuple(x/radius for x in _mv(E, X))
        Ed = _mm(E, Dphi)
        modes.append(tuple(chi*x for x in direction))
        gradients.append(tuple(tuple(chi*Ed[i][j]/radius+direction[i]*dchi[j]
                                     for j in range(3)) for i in range(3)))
    displacement = tuple(sum((a*m[i] for a, m in zip(coefficients[parent], modes)), I(0)) for i in range(3))
    return dict(position=X, reference_gradient=Dphi, roof=C, modes=tuple(modes),
                gradients=tuple(gradients), displacement=displacement)


def _quintic(theta, delta):
    if theta <= -delta:
        return F(0), F(0)
    if theta >= delta:
        return F(1), F(0)
    t = (theta+delta)/(2*delta)
    return 6*t**5-15*t**4+10*t**3, 30*t*t*(1-t)*(1-t)/(2*delta)


def _signed_angle(value):
    return interval_atan(I(value), 115) if value >= 0 else -interval_atan(I(-value), 115)


def _partition_axis(ray, numerator, denominator, r0, delta):
    y, z = ray[numerator], abs(ray[denominator])
    if not z:
        return (I(F(1, 2)) if not y else I(int(y > 0))), (I(0),)*3
    theta = _signed_angle(y/z)
    if theta.hi <= -delta:
        return I(0), (I(0),)*3
    if theta.lo >= delta:
        return I(1), (I(0),)*3
    t = (theta+delta)/(2*delta)
    weight = 6*t*t*t*t*t-15*t*t*t*t+10*t*t*t
    derivative = 30*t*t*(I(1)-t)*(I(1)-t)/(2*delta)
    norm = interval_sqrt(I(sum((x*x for x in ray), F(0))), 115)
    q = tuple(I(x)/norm for x in ray)
    denom = q[1]*q[1]+q[2]*q[2]
    gradient = [I(0)]*3
    gradient[numerator] = I(abs(ray[denominator]))/norm/(r0*denom)
    gradient[denominator] = -I((1 if ray[denominator] > 0 else -1)*ray[numerator])/norm/(r0*denom)
    return weight, tuple(derivative*x for x in gradient)


def _blend_oracle(owner, ray, depth, order):
    _, _, bodies, coordinates = _authorities()
    rows = {parent: b for parent, b in bodies.items() if b["original_body"]["owner"] == owner}
    radius = _fraction(next(iter(rows.values()))["embedding_bank"]["radius_m"])
    r0 = interval_cbrt(I(radius**3)-3*radius**2*depth, 115)
    wy, gy = _partition_axis(ray, 1, 2, r0, F(1, 100))
    wz, gz = _partition_axis(ray, 2, 1, r0, F(1, 100))
    columns = [(I(0),)*3]*56
    gradients = [((I(0),)*3,)*3]*56
    weights, dweights = {}, {}
    for parent, context in rows.items():
        body = context["original_body"]
        axes = [tuple(_fraction(x) for x in body[key])
                for key in ("original_quarter_start", "original_quarter_end")]
        signs = tuple(sum((a[i] for a in axes), F(0)) for i in (1, 2))
        fy, fz = (wy if signs[0] > 0 else I(1)-wy), (wz if signs[1] > 0 else I(1)-wz)
        dy = gy if signs[0] > 0 else tuple(-x for x in gy)
        dz = gz if signs[1] > 0 else tuple(-x for x in gz)
        w = fy*fz
        dw = tuple(x*fz+fy*y for x, y in zip(dy, dz))
        weights[parent], dweights[parent] = w, dw
        if w.hi == 0:
            continue
        field = _bank_oracle(parent, ray, depth)
        for name, psi, G in zip(body["coordinate_order"], field["modes"], field["gradients"]):
            index = order.index(("body", body["component_id"], name))
            columns[index] = tuple(w*x for x in psi)
            gradients[index] = tuple(tuple(w*G[i][j]+psi[i]*dw[j] for j in range(3)) for i in range(3))
    coeff = {}
    for parent, context in bodies.items():
        coeff.update({("body", context["original_body"]["component_id"], name): value
                      for name, value in zip(context["original_body"]["coordinate_order"], coordinates[parent])})
    displacement = tuple(sum((coeff[row]*column[i] for row, column in zip(order, columns)), I(0))
                         for i in range(3))
    derivative = tuple(tuple(sum((coeff[row]*G[i][j] for row, G in zip(order, gradients)), I(0))
                            for j in range(3)) for i in range(3))
    return dict(weights=weights, weight_gradients=dweights, modes=tuple(columns),
                gradients=tuple(gradients), displacement=displacement, gradient=derivative)


def test_saved_authorities_contain_all_56_authentic_nonzero_cold_coordinates():
    transport, source_work, bodies, coordinates = _authorities()
    assert len(bodies) == 8 and sum(map(len, coordinates.values())) == 56
    assert all(type(x) is F and x for values in coordinates.values() for x in values)
    for packet in (transport, source_work):
        body = {k: v for k, v in packet.items() if k != "packet_sha256"}
        assert hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False).encode()).hexdigest() == packet["packet_sha256"]


def test_quintic_partition_has_exact_complement_and_zero_endpoint_derivatives():
    delta = F(1, 100)
    for theta in (-2*delta, -delta, -delta/2, F(0), delta/3, delta, 2*delta):
        value, derivative = _api()._quintic_partition(I(theta), delta)
        expected_value, expected_derivative = _quintic(theta, delta)
        _overlap(value, expected_value)
        _overlap(derivative, expected_derivative)
        other, _ = _api()._quintic_partition(I(-theta), delta)
        _overlap(value+other, F(1))
        assert derivative.lo >= 0 and derivative.hi <= F(15, 16)/delta
    for theta in (-delta, delta):
        assert _quintic(theta, delta)[1] == 0


def test_roof_chart_correction_has_distinct_radial_and_tangential_derivatives():
    q = (I(F(3, 5)), I(F(4, 5)), I(0))
    original, old = I(F(90000)), I(F(90001))
    correction = _api()._roof_chart_correction(q, original, old, precision=96)
    _overlap(_mv(correction, q), tuple(original*original/(old*old)*x for x in q))
    tangent = (I(-F(4, 5)), I(F(3, 5)), I(0))
    _overlap(_mv(correction, tangent), tuple(old/original*x for x in tangent))
    assert any(not correction[i][i].lo <= 1 <= correction[i][i].hi for i in (0, 1))


def test_actual_saved_bank_modes_and_full_original_gradients_match_independent_chart():
    system = _native().system()
    for parent, ray, depth in (("octant:1", (F(-1), F(-2), F(100)), F(5000)),
                               ("octant:6", (F(1), F(100), F(-100)), F(7000))):
        expected = _bank_oracle(parent, ray, I(depth))
        actual = _api()._bank_field(system, parent, ray, I(depth), 96)
        _overlap(actual.reference_position_bounds_m, expected["position"])
        _overlap(actual.reference_original_gradient_bounds, expected["reference_gradient"])
        _overlap(actual.roof_chart_correction_bounds, expected["roof"])
        _overlap(actual.mode_displacement_columns, expected["modes"])
        _overlap(actual.mode_original_gradient_columns_per_m, expected["gradients"])
        _overlap(actual.displacement_bounds_m, expected["displacement"])


def test_original_quarter_fields_have_a_genuine_negative_seam_jump():
    # The two literal banks were independently registered; neither their
    # nearly equal roofs nor their seven coefficients authorize gluing.
    ray, depth = (F(-1), F(0), F(100)), I(5000)
    left = _bank_oracle("octant:1", ray, depth)
    right = _bank_oracle("octant:3", ray, depth)
    body = _authorities()[2]["octant:1"]["original_body"]
    Q = _decode(body["owner_rotation_matrix_bounds"])
    seam_normal = _mv(Q, (I(0), I(1), I(0)))
    jump = _dot(seam_normal, tuple(y-x for x, y in zip(left["displacement"], right["displacement"])))
    assert jump.hi < 0
    assert -F(35, 10**7) < jump.lo <= jump.hi < -F(34, 10**7)


def test_all_32_saved_source_labels_are_inside_single_bank_partition_interiors():
    transport = _authorities()[0]
    labels = tuple(row["response"] for row in transport["original_material_queries"]
                   if row["response"]["material_depth_bounds_m"] == {"lo": [0, 1], "hi": [0, 1]})
    assert len(labels) == 32
    for row in labels:
        ray = tuple(_fraction(x) for x in row["original_direction_literal"])
        # Homogeneous ratios are evaluated before normalization; no interval
        # rounding manufactures a sector boundary.
        y, z = map(abs, ray[1:])
        assert y and z
        angle = interval_atan(I(min(y, z)/max(y, z)), 96)
        assert angle.lo > F(1, 100)


def test_inner_partition_preserves_authentic_879_positions_and_seven_mode_values():
    transport = _authorities()[0]
    system = _native().system()
    # Both a live receiver and a donor source at a nonzero source depth are
    # actual saved queries.  This is value compatibility; the old derivative
    # input chart omitted the roof correction and is checked separately.
    selected = []
    for owner in (0, 1):
        selected.append(next(row["response"] for row in transport["original_material_queries"]
            if row["response"]["owner"] == owner
            and row["response"]["material_depth_bounds_m"] != {"lo": [0, 1], "hi": [0, 1]}))
    for row in selected:
        ray = tuple(_fraction(x) for x in row["original_direction_literal"])
        actual = _api()._bank_field(system, row["original_parent_id"], ray,
                                   _interval(row["material_depth_bounds_m"]), 96)
        _overlap(actual.reference_position_bounds_m, _decode(row["reference_position_bounds_m"]))
        _overlap(actual.mode_displacement_columns, _decode(row["mode_displacement_columns"]))
        _overlap(actual.displacement_bounds_m, _decode(row["displacement_increment_bounds_m"]))


def test_roof_correction_exposes_the_actual_saved_bq_chart_alias():
    transport = _authorities()[0]
    row = next(row["response"] for row in transport["original_material_queries"]
               if row["response"]["material_depth_bounds_m"] == {"lo": [0, 1], "hi": [0, 1]})
    ray = tuple(_fraction(x) for x in row["original_direction_literal"])
    actual = _api()._bank_field(_native().system(), row["original_parent_id"], ray, I(0), 96)
    old_bq = _mm(_decode(row["reference_gradient_bounds"]), _decode(row["material_owner_pose_bounds"]))
    _overlap(actual.reference_original_gradient_bounds, _mm(old_bq, actual.roof_chart_correction_bounds))
    assert any(max(a.lo, b.lo) > min(a.hi, b.hi)
               for arow, brow in zip(actual.reference_original_gradient_bounds, old_bq)
               for a, b in zip(arow, brow))


def test_full_56_column_blend_and_original_gradients_match_independent_seam_oracle():
    system = _native().system()
    ray, depth = (F(-1), F(1), F(10000)), I(5000)
    point = _api().evaluate_native_common_cold_material_point(system, "octant:3", ray, depth)
    expected = _blend_oracle(0, ray, depth, system.coordinate_order)
    _overlap(point.mode_displacement_columns, expected["modes"])
    _overlap(point.mode_original_gradient_columns_per_m, expected["gradients"])
    _overlap(point.displacement_bounds_m, expected["displacement"])
    for parent, weight in point.seam_weights:
        _overlap(weight, expected["weights"][parent])
    for parent, gradient in point.seam_weight_original_gradients:
        _overlap(gradient, expected["weight_gradients"][parent])
    saved_archive = json.loads(point.source_archive_point_json)
    archive_gradient = _decode(saved_archive["response"]["deformation_gradient_bounds"])
    total = tuple(tuple(point.fixed_radial_scale*archive_gradient[i][j]+expected["gradient"][i][j]
                        for j in range(3)) for i in range(3))
    _overlap(point.original_cartesian_gradient_bounds, total)
    _overlap(sum((weight for _, weight in point.seam_weights), I(0)), F(1))
    _contains_zero(tuple(sum((gradient[i] for _, gradient in point.seam_weight_original_gradients), I(0))
                         for i in range(3)))


def test_same_owner_parent_seams_share_position_and_full_cartesian_gradient():
    system = _native().system()
    for parents, ray in ((("octant:1", "octant:3"), (F(-1), F(0), F(100))),
                         (("octant:0", "octant:1"), (F(-1), F(-100), F(0)))):
        left, right = (_api().evaluate_native_common_cold_material_point(system, parent, ray, I(5000))
                       for parent in parents)
        assert left.original_material_id != right.original_material_id
        _overlap(left.position_bounds_m, right.position_bounds_m)
        _overlap(left.original_cartesian_gradient_bounds, right.original_cartesian_gradient_bounds)
        _overlap(left.mode_displacement_columns, right.mode_displacement_columns)
        assert sum(weight.lo > 0 for _, weight in left.seam_weights) == 2


def test_inactive_owner_poles_are_regular_without_a_singular_beta_or_seam_chart():
    system = _native().system()
    for parent, sign in (("octant:0", -1), ("octant:4", 1)):
        point = _api().evaluate_native_common_cold_material_point(
            system, parent, (F(sign), F(0), F(0)), I(7000))
        assert point.volume_jacobian_bounds.lo > 0
        _contains_zero(point.displacement_bounds_m)
        _contains_zero(point.mode_displacement_columns)
        _contains_zero(point.mode_original_gradient_columns_per_m)
        assert all(weight.lo == weight.hi == F(1, 4) for _, weight in point.seam_weights)


def test_whole_support_certificate_covers_all_parents_and_archives_with_a_uniform_margin():
    system = _native().system()
    certificate = system.whole_domain_certificate
    assert certificate.original_parent_count == 8
    assert set(certificate.original_parent_ids) == {"octant:"+str(i) for i in range(8)}
    assert certificate.nonempty_archive_cohort_count == len(certificate.complete_archive_domain_ids) == 3
    assert certificate.complete_live_and_archive_material_retained
    assert certificate.within_owner_parent_corner_gluing_certified
    assert certificate.whole_owner_material_injectivity_certified
    assert certificate.whole_cold_cross_owner_support_separation_certified
    for owner in (certificate.donor, certificate.receiver):
        assert len(owner.parent_ids) == 4 and owner.global_injectivity_certified
        # A derivative bound over the complete hemisphere shell, together
        # with its quasiconvex factor two, proves global injectivity.  These
        # identities are checked independently of any finite point list.
        assert owner.total_gradient_deviation_bound < F(1, 2)
        assert owner.local_minimum_singular_value == 1-owner.total_gradient_deviation_bound
        assert owner.global_colipschitz_margin == 1-2*owner.total_gradient_deviation_bound > 0
        assert owner.volume_jacobian_bounds.lo > 0
        assert owner.volume_jacobian_bounds.lo <= owner.local_minimum_singular_value**3
        assert owner.volume_jacobian_bounds.hi >= (1+owner.total_gradient_deviation_bound)**3
    donor, receiver = certificate.donor, certificate.receiver
    radius = _native().archive_carrier().radius_m
    assert donor.fixed_radial_scale == 1-2*donor.maximum_displacement_m/radius
    assert receiver.fixed_radial_scale == 1+2*receiver.maximum_displacement_m/radius
    assert certificate.donor_radial_upper_bound_m == donor.fixed_radial_scale*radius+donor.maximum_displacement_m
    assert certificate.receiver_radial_lower_bound_m == receiver.fixed_radial_scale*radius-receiver.maximum_displacement_m
    assert certificate.cold_owner_radial_gap_lower_bound_m == (
        certificate.receiver_radial_lower_bound_m-certificate.donor_radial_upper_bound_m) > 0
    assert not certificate.physical_material_admission_authorized
    assert not certificate.whole_clock_all_pair_admission_authorized


def test_unbuffered_inner_value_compatibility_does_not_claim_cross_owner_separation():
    transport, work = _native().source_packets()
    unbuffered = _api().bind_native_common_cold_material_carrier(transport, work,
        _native().archive_carrier(), provenance="Independent unbuffered comparison only",
        radial_support_guard=False)
    certificate = unbuffered.whole_domain_certificate
    assert certificate.donor.fixed_radial_scale == certificate.receiver.fixed_radial_scale == 1
    assert certificate.cold_owner_radial_gap_lower_bound_m < 0
    assert not certificate.whole_cold_cross_owner_support_separation_certified
    system = _native().system()
    ray = (F(-1), F(-100), F(100))
    bare = _api().evaluate_native_common_cold_material_point(unbuffered, "octant:1", ray, I(5000))
    buffered = _api().evaluate_native_common_cold_material_point(system, "octant:1", ray, I(5000))
    _overlap(bare.displacement_bounds_m, buffered.displacement_bounds_m)
    assert any(max(x.lo, y.lo) > min(x.hi, y.hi) for x, y in zip(bare.position_bounds_m, buffered.position_bounds_m))


def test_endpoint_queries_keep_material_time_and_source_history_unchanged():
    system = _native().system()
    assert (system.source_geometry_elapsed_time_s, system.contact_geometry_elapsed_time_s,
            system.memory_elapsed_time_s, system.material_geometry_elapsed_time_s) == (F(0), F(2000000), F(1000000), F(1000000))
    assert system.accepted_original_native_intervals == 1 and system.accepted_new_native_intervals == 0
    assert not system.source_state_or_work_advanced
    assert not system.physical_source_force_prestress_work_handover_certified
    assert not system.hot_constitutive_law_defined and not system.finite_mean_chart_defined
    assert not system.new_mixed_force_equilibrium_certified
    assert not system.whole_clock_all_pair_admission_authorized
    assert not system.full_system_temporal_path_work_certified
    for parent, ray, depth, time in (("octant:1", (F(-1), F(-100), F(100)), I(10001), F(1)),
                                    ("octant:1", (F(-1), F(100), F(100)), I(5000), F(1)),
                                    ("octant:1", (F(-1), F(-100), F(100)), I(5000), F(1, 2)),
                                    ("HOT", (F(-1), F(-100), F(100)), I(5000), F(1))):
        with pytest.raises(ValueError):
            _api().evaluate_native_common_cold_material_point(system, parent, ray, depth, time_fraction=time)


def test_coordinate_envelope_keeps_fixed_reference_and_actual_56_parameter_variations():
    system = _native().system()
    assert system.coordinate_envelope_abs_m == tuple(2*abs(x) for x in system.bound_coordinate_increments)
    parent, ray, depth = "octant:3", (F(-1), F(1), F(10000)), I(5000)
    zero = (F(0),)*56
    opposite = tuple(-x for x in system.bound_coordinate_increments)
    base = _api().evaluate_native_common_cold_material_point(system, parent, ray, depth)
    absent = _api().evaluate_native_common_cold_material_point(system, parent, ray, depth,
                                                               coordinate_increments=zero)
    reversed_point = _api().evaluate_native_common_cold_material_point(system, parent, ray, depth,
                                                                       coordinate_increments=opposite)
    assert base.fixed_radial_scale == absent.fixed_radial_scale == reversed_point.fixed_radial_scale
    assert absent.bound_coordinate_increments == zero and reversed_point.bound_coordinate_increments == opposite
    _contains_zero(absent.displacement_bounds_m)
    _overlap(base.displacement_bounds_m, tuple(-x for x in reversed_point.displacement_bounds_m))
    _overlap(base.mode_displacement_columns, reversed_point.mode_displacement_columns)
    _overlap(base.mode_original_gradient_columns_per_m, reversed_point.mode_original_gradient_columns_per_m)
    assert all(point.volume_jacobian_bounds.lo > 0 for point in (base, absent, reversed_point))
    outside = list(system.bound_coordinate_increments)
    outside[0] = 3*system.coordinate_envelope_abs_m[0]
    for invalid in (tuple(outside), (0,)*56, tuple(system.bound_coordinate_increments[:-1])):
        with pytest.raises(ValueError):
            _api().evaluate_native_common_cold_material_point(system, parent, ray, depth,
                                                              coordinate_increments=invalid)


def test_typed_guard_rejects_altered_clocks_roofs_rates_and_parent_context():
    system = _native().system()
    bank = system.banks[0]
    changed_banks = (
        replace(bank, roof_radial_displacement_m=bank.roof_radial_displacement_m+1),
        replace(bank, angular_rate_per_physical_volume_depth_m=(F(0), F(0), F(0))),
        replace(bank, original_parent_id="octant:forged"),
    )
    changes = [replace(system, banks=(changed,)+system.banks[1:]) for changed in changed_banks]
    changes.append(replace(system, contact_geometry_elapsed_time_s=system.contact_geometry_elapsed_time_s+1))
    changes.append(replace(system, source_packet_seals=("forged",)))
    for changed in changes:
        with pytest.raises(ValueError):
            _api().native_common_cold_material_carrier_to_dict(changed)


def test_joint_restart_requires_exact_complete_authorities_and_point_query_context():
    api, native = _api(), _native()
    system = native.system()
    transport, source_work = native.source_packets()
    dto = api.native_common_cold_material_carrier_to_dict(system)
    restored = api.native_common_cold_material_carrier_from_dict(dto,
        material_transport_packet=transport, source_work_packet=source_work,
        archive_carrier=native.archive_carrier())
    assert api.native_common_cold_material_carrier_to_dict(restored) == dto
    parent, ray, depth = "octant:1", (F(-1), F(0), F(100)), I(5000)
    point = api.evaluate_native_common_cold_material_point(system, parent, ray, depth)
    point_dto = api.native_common_cold_material_point_to_dict(point)
    expected = dict(system=restored, expected_original_parent_id=parent,
        expected_original_ray=ray, expected_material_depth_bounds_m=depth,
        expected_coordinate_increments=system.bound_coordinate_increments)
    restarted = api.native_common_cold_material_point_from_dict(point_dto, **expected)
    assert api.native_common_cold_material_point_to_dict(restarted) == point_dto
    wrong = dict(expected, expected_original_parent_id="octant:3")
    with pytest.raises(ValueError):
        api.native_common_cold_material_point_from_dict(point_dto, **wrong)
    altered = deepcopy(point_dto)
    altered["material_depth_bounds_m"] = {"lo": [5001, 1], "hi": [5001, 1]}
    raw_body = {k: v for k, v in altered.items() if k not in ("format", "binding_sha256")}
    altered["binding_sha256"] = hashlib.sha256(json.dumps(raw_body, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    with pytest.raises(ValueError):
        api.native_common_cold_material_point_from_dict(altered, **expected)
    with pytest.raises(ValueError):
        api.native_common_cold_material_point_to_dict(replace(point, archive_debits=point.archive_debits+1))
    forged_packet = deepcopy(source_work)
    forged_packet["clocks_advanced"] = True
    with pytest.raises(ValueError):
        api.bind_native_common_cold_material_carrier(transport, forged_packet,
            native.archive_carrier(), provenance="Untrusted clock-altered source")
