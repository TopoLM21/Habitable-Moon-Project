"""Physical archive carriers retain entire original domains and material depth.

Fixtures restore completed source DTOs only; no historical test fixture,
force solve or old tensor quadrature runs. Independent radius and rotation
oracles check the explicitly declared new paired physical reference.
"""
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import pytest

from tectonics.geometric_native_pose import FractionInterval as I


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PACKET = ROOT / (
    "analysis/geometric_native_common_material_transfer_validation/"
    "release_validation_01/native_control/source_packet.json")
SOURCE_PACKET_SHA = "a9fcc7874c0e68940e20ded658748300f55d9f1d296db5c9d67b59269ac8b3ed"
FIRST_ENTRY = ROOT / "analysis/geometric_native_first_entry_validation/control_release_02"
POINTS_SHA = "c57801a6e80051dffec2250321f6e812bfe36d10f19993d2a9c4172bb6e459b0"
PROVENANCE = "Declared paired areal-volume reference and irreversible cubic activation"


@lru_cache(maxsize=1)
def _api():
    from tectonics import geometric_native_archive_material_carrier as api
    return api


@lru_cache(maxsize=1)
def _native():
    from analysis.geometric_native_archive_material_carrier_validation import native_control
    return native_control


def _encoded(raw):
    return json.dumps(raw, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _decimal(value):
    return Decimal(value.numerator) / Decimal(value.denominator)


def _encloses(bounds, value):
    if type(bounds) is I:
        value = value if type(value) is I else I(F(value))
        assert bounds.lo <= value.lo <= value.hi <= bounds.hi
    else:
        assert len(bounds) == len(value)
        for a, b in zip(bounds, value):
            _encloses(a, b)


def _overlaps(first, second):
    if type(first) is I:
        second = second if type(second) is I else I(F(second))
        assert max(first.lo, second.lo) <= min(first.hi, second.hi)
    else:
        assert len(first) == len(second)
        for a, b in zip(first, second):
            _overlaps(a, b)


def _cube_root(value):
    """Decimal Newton oracle independent of the production dyadic root grid."""
    with localcontext() as context:
        context.prec = 230
        number = _decimal(value)
        if number == 0:
            return F(0)
        guess = Decimal(10) ** (number.adjusted() // 3)
        for _ in range(1000):
            after = (2 * guess + number / guess**2) / 3
            if abs(after - guess) <= Decimal("1e-225") * max(Decimal(1), abs(after)):
                return F(after)
            guess = after
        raise AssertionError("Independent radius iteration did not converge")


def _dot(first, second):
    return sum((a*b for a, b in zip(first, second)), F(0))


def _cross(first, second):
    return (first[1]*second[2]-first[2]*second[1],
            first[2]*second[0]-first[0]*second[2],
            first[0]*second[1]-first[1]*second[0])


def _det(matrix):
    return _dot(matrix[0], _cross(matrix[1], matrix[2]))


@pytest.fixture(scope="module")
def actual():
    packet_bytes = SOURCE_PACKET.read_bytes()
    points_path = FIRST_ENTRY / "point_certificates.json"
    points_bytes = points_path.read_bytes()
    assert hashlib.sha256(packet_bytes).hexdigest() == SOURCE_PACKET_SHA
    assert hashlib.sha256(points_bytes).hexdigest() == POINTS_SHA
    packet = json.loads(packet_bytes)
    assert packet["packet_sha256"] == hashlib.sha256(_encoded({
        k: v for k, v in packet.items() if k != "packet_sha256"})).hexdigest()
    material_raw = json.loads(packet["material_transfer"]["complete_original_material_state_json"])
    state = _native().source_state()
    assert _encoded(material_raw) == _encoded(_native().source_documents()["material_state.json"])
    points = {row["label"]: dict(
        parent=row["certificate"]["original_parent_id"],
        ray=tuple(F(*pair) for pair in row["certificate"]["original_point"]))
        for row in json.loads(points_bytes)}
    yield dict(state=state, points=points, material_raw=material_raw)
    assert SOURCE_PACKET.read_bytes() == packet_bytes
    assert points_path.read_bytes() == points_bytes


@pytest.fixture(scope="module")
def carrier(actual):
    return _native().system()


def _point(carrier, actual, label, *, depth=F(0), time=F(1), precision=96):
    row = actual["points"][label]
    return _api().evaluate_native_archive_material_point(carrier,
        row["parent"], row["ray"], material_depth_bounds_m=I(depth),
        time_fraction=time, precision=precision)


def test_areal_volume_radius_preserves_whole_depth_and_declared_paired_reference():
    api = _api()
    radius, thickness = F(100), F(2)
    for depth in (F(0), F(1, 10**100), F(1, 3), F(4, 3), thickness):
        donor = api.areal_volume_radius(radius, I(depth), precision=128)
        upper = api.areal_volume_radius(radius, I(depth),
            upper_thickness_m=thickness, precision=128)
        _encloses(donor, _cube_root(radius**3 - 3*radius**2*depth))
        _encloses(upper, _cube_root(radius**3 + 3*radius**2*(thickness-depth)))
    donor = api.areal_volume_radius(radius, I(0, thickness), precision=128)
    upper = api.areal_volume_radius(radius, I(0, thickness),
        upper_thickness_m=thickness, precision=128)
    assert donor.hi == upper.lo == radius
    assert donor.lo < radius-thickness
    assert upper.hi > radius
    # R^2 H is the actual source amount per steradian, not the Euclidean shell.
    assert (radius**3-(radius**3-3*radius**2*thickness))/3 == radius**2*thickness
    assert (radius**3+3*radius**2*thickness-radius**3)/3 == radius**2*thickness
    assert (radius**3-(radius-thickness)**3)/3 != radius**2*thickness


def test_transcendental_bounds_retain_exact_values_and_sub_float_increment():
    api = _api()
    assert api.interval_exp(I(0), precision=128) == I(1)
    assert api.interval_cbrt(I(F(1, 8)), precision=128) == I(F(1, 2))
    increment = F(1, 10**150)
    bound = api.interval_exp(I(increment), precision=180)
    with localcontext() as context:
        context.prec = 230
        _encloses(bound, F(_decimal(increment).exp()))
    assert bound.lo > 1 and float(bound.lo) == 1.0


def _sin_cos(value):
    """Independent Decimal recurrence at small angles, with excess precision."""
    sine, cosine = value, Decimal(1)
    sine_term, cosine_term = value, Decimal(1)
    for k in range(1, 180):
        sine_term *= -value*value / Decimal((2*k)*(2*k+1))
        cosine_term *= -value*value / Decimal((2*k-1)*(2*k))
        sine += sine_term
        cosine += cosine_term
    return sine, cosine


def test_cubic_history_matches_closed_integral_and_tangent_derivative():
    # For a pure relative Z rotation, g(s) = sin(a*s). This elementary
    # integral is independent of production positive-time cell integration.
    api = _api()
    a, t = F(1, 8), F(3, 4)
    activation = api.smooth_activation_history((F(0), F(1), F(0)),
        (F(0), F(0), a), (F(0),)*3, t, order=24, cells=32, precision=128)
    with localcontext() as context:
        context.prec = 230
        angle, axis = _decimal(a*t), _decimal(a)
        sine, cosine = _sin_cos(angle)
        beta = (Decimal(2)/3-cosine+cosine**3/3)/(6*axis**3)
        tangent_derivative = -sine**3/(6*axis**3)
        rate = sine**3/(6*axis**2)
        _encloses(activation.beta_bounds, F(beta))
        _encloses(activation.gradient_bounds[0], F(tangent_derivative))
        _encloses(activation.beta_rate_per_fraction_bounds, F(rate))
    assert activation.beta_bounds.lo > 0
    _encloses(activation.gradient_bounds[2],F(0))
    assert activation.gamma == a
    assert activation.beta_bounds.hi <= a/6


def test_activation_before_entry_and_at_zero_motion_has_no_pole_singularity():
    api = _api()
    zero = (F(0),)*3
    # A pole and a ray far inside the original donor never enter for this
    # finite history. They retain the exact unmodified owner motion.
    for ray, donor in (((F(1),F(0),F(0)), zero),
                       ((F(1),F(1),F(0)), (F(0),F(0),F(1,8)))):
        activation = api.smooth_activation_history(ray, donor, zero, F(1),
            order=24, cells=32, precision=96)
        assert activation.beta_bounds == I(0)
        assert activation.gradient_bounds == (I(0),)*3
        assert activation.beta_rate_per_fraction_bounds == I(0)


def test_all_three_original_cohorts_and_correlated_parent_reservoirs_survive(actual, carrier):
    state = actual["state"]
    assert len(state.parent_allocations) == 8
    assert {cohort.original_parent_id for cohort in state.archive_cohorts} == {
        "octant:5", "octant:6", "octant:7"}
    assert len({cohort.cohort_id for cohort in state.archive_cohorts}) == 3
    assert all((cohort.owner,cohort.receiver_owner) == (1,0)
               for cohort in state.archive_cohorts)
    raw = _api().native_archive_material_carrier_to_dict(carrier)
    assert raw["carrier"]["source_material_state_sha256"] == state.state_sha256
    originals = {parent.fragment_id: parent for parent in state.journal.source_surface.fragments}
    for allocation in state.parent_allocations:
        parent = originals[allocation.fragment_id]
        assert allocation.material_id == parent.parcel.material_id
        assert allocation.owner == parent.parcel.plate
        assert allocation.material_fields == parent.parcel.material_fields
        assert allocation.specific_properties == parent.parcel.specific_properties
        for reservoir in allocation.reservoirs:
            original = F.from_float(getattr(parent.parcel,reservoir.reservoir))
            assert reservoir.parent_amount == original
            assert reservoir.live_amount_bounds == I(original)-reservoir.archived_amount_bounds
            assert reservoir.live_amount_bounds.lo+reservoir.archived_amount_bounds.hi == original
            assert reservoir.live_amount_bounds.hi+reservoir.archived_amount_bounds.lo == original
    domain = next(domain for domain in state.sweep_certificate.donor_domains
                  if domain.fragment_id == "octant:5")
    assert domain.extra_swept_area_bounds_sr.lo > 0
    assert domain.swept_area_bounds_sr.lo > domain.endpoint_area_bounds_sr.hi


def test_receiver_full_depth_gradient_matches_independent_radial_deformation(actual, carrier):
    radius = F.from_float(actual["state"].journal.source_surface.radius_km)*1000
    height = F(10000)
    omega = tuple(F.from_float(value)/F(31557600000000)
                  for value in dict(actual["state"].journal.omega_rad_per_myr)[0])
    for depth in (F(0),height/3,2*height/3,height):
        response = _point(carrier,actual,"retained_overriding_material",depth=depth,time=F(0))
        r0 = _cube_root(radius**3-3*radius**2*depth)
        upper = _cube_root(radius**3+3*radius**2*(height-depth))
        with localcontext() as context:
            context.prec = 220
            ratio = _decimal(upper)/_decimal(r0)
            normal = ratio**-2
            expected = tuple(tuple(F((ratio if i == j else Decimal(0))
                +(normal-ratio)/3) for j in range(3)) for i in range(3))
            position = tuple(F(-_decimal(upper)/Decimal(3).sqrt()) for _ in range(3))
        _encloses(response.deformation_gradient_bounds, expected)
        _encloses(response.volume_jacobian_bounds,F(1))
        _overlaps(_det(response.deformation_gradient_bounds),response.volume_jacobian_bounds)
        assert response.areal_material_thickness_m == height
        _encloses(response.position_bounds_m,position)
        _encloses(response.velocity_bounds_m_per_s,_cross(omega,position))
        if depth < height:
            assert response.radial_support_bounds_m.lo > radius
        else:
            assert response.radial_support_bounds_m == I(radius)


def test_entry_activation_glues_full_donor_thickness_and_velocity(actual, carrier):
    api = _api()
    # This genuine original axis label lies on the initial owner interface;
    # activation vanishes at entry itself. It has no inferred old spring ID.
    radius = F.from_float(actual["state"].journal.source_surface.radius_km)*1000
    omega = tuple(F.from_float(value)/F(31557600000000)
                  for value in dict(actual["state"].journal.omega_rad_per_myr)[1])
    ray = (F(0),F(1),F(0))
    identity = tuple(tuple(F(i == j) for j in range(3)) for i in range(3))
    for depth in (F(0),F(10000,3),F(20000,3),F(10000)):
        response = api.evaluate_native_archive_material_point(carrier,
            "octant:6",ray,material_depth_bounds_m=I(depth),time_fraction=F(0),precision=128)
        reference_radius = _cube_root(radius**3-3*radius**2*depth)
        position = (F(0),reference_radius,F(0))
        assert response.beta_bounds == I(0)
        assert response.beta_rate_per_fraction_bounds == I(0)
        assert response.beta_original_q_gradient_bounds == (I(0),)*3
        _encloses(response.position_bounds_m,position)
        _encloses(response.velocity_bounds_m_per_s,_cross(omega,position))
        _encloses(response.deformation_gradient_bounds,identity)
        _encloses(response.volume_jacobian_bounds,F(1))


def _observer_matrix(history):
    """Independent full ordered Decimal Rodrigues, not a pose DTO replay."""
    with localcontext() as context:
        context.prec = 230
        matrix = tuple(tuple(Decimal(i == j) for j in range(3)) for i in range(3))
        for step in history:
            axis = tuple(_decimal(F.from_float(x))*_decimal(F.from_float(step.dt_myr))
                         for x in step.omega_rad_per_myr)
            theta = sum((x*x for x in axis),Decimal(0)).sqrt()
            sine, cosine = _sin_cos(theta)
            skew = ((Decimal(0),-axis[2],axis[1]),
                    (axis[2],Decimal(0),-axis[0]),
                    (-axis[1],axis[0],Decimal(0)))
            increment = tuple(tuple(cosine*Decimal(i == j)+sine/theta*skew[i][j]
                +(1-cosine)/theta**2*axis[i]*axis[j] for j in range(3)) for i in range(3))
            matrix = tuple(tuple(sum((increment[i][k]*matrix[k][j]
                for k in range(3)),Decimal(0)) for j in range(3)) for i in range(3))
        # At these small literal axes 180 alternating-series terms contribute
        # less than 1e-1000; this bound also covers all 230-digit operations.
        error = F(1,10**200)
        return tuple(tuple(I(F(x)-error,F(x)+error) for x in row) for row in matrix)


def _mv(matrix,vector):
    return tuple(_dot(row,vector) for row in matrix)


def _mm(first,second):
    return tuple(tuple(_dot(row,column) for column in zip(*second)) for row in first)


def test_complete_ordered_observer_pose_transforms_position_velocity_and_gradient(actual,carrier):
    from tectonics.geometric_intrinsic_geometry import RotationIncrement
    api = _api()
    history = (RotationIncrement((.01,.02,-.01),1.),
               RotationIncrement((-.015,.005,.02),1.))
    translation = (F(7,3),F(-9,7),F(11,5))
    row = actual["points"]["ordinary_entry"]
    arguments = dict(material_depth_bounds_m=I(F(10000,3)),time_fraction=F(1),precision=96)
    original = api.evaluate_native_archive_material_point(carrier,row["parent"],row["ray"],**arguments)
    transformed = api.evaluate_native_archive_material_point(carrier,row["parent"],row["ray"],
        **arguments,world_history=history,world_translation_m=translation)
    rotation = _observer_matrix(history)
    expected_position = tuple(a+b for a,b in zip(_mv(rotation,original.position_bounds_m),translation))
    _overlaps(transformed.position_bounds_m,expected_position)
    _overlaps(transformed.velocity_bounds_m_per_s,_mv(rotation,original.velocity_bounds_m_per_s))
    _overlaps(transformed.deformation_gradient_bounds,_mm(rotation,original.deformation_gradient_bounds))
    assert transformed.volume_jacobian_bounds == original.volume_jacobian_bounds
    assert transformed.beta_bounds == original.beta_bounds
    assert transformed.cohort_id == original.cohort_id
    # Reversing noncommuting words must change the actual spatial placement.
    reversed_pose = api.evaluate_native_archive_material_point(carrier,row["parent"],row["ray"],
        **arguments,world_history=tuple(reversed(history)),world_translation_m=translation)
    assert any(a.hi < b.lo or b.hi < a.lo for a,b in
               zip(transformed.position_bounds_m,reversed_pose.position_bounds_m))


def test_swept_only_material_stays_buried_after_exit_and_reentry_archives_once(actual,carrier):
    swept = _point(carrier,actual,"swept_only_owner_entry_exit",depth=F(10000,3),precision=256)
    assert swept.allocation == "archived"
    assert swept.contact_occupancy == "outside"
    assert swept.archive_debits == 1
    assert swept.first_entry_receiver_fragment_ids == ("octant:1",)
    assert swept.cohort_id == next(cohort.cohort_id for cohort in actual["state"].archive_cohorts
        if cohort.original_parent_id == "octant:5")
    assert swept.beta_bounds.lo > 0
    assert swept.beta_rate_per_fraction_bounds == I(0)
    assert swept.radial_support_bounds_m.hi < F(100000)
    assert swept.volume_jacobian_bounds.lo > 0
    reentry = _point(carrier,actual,"main_plane_reentry",depth=F(10000,3),precision=256)
    assert reentry.allocation == "archived" and reentry.archive_debits == 1
    assert reentry.beta_bounds.lo > 0 and reentry.volume_jacobian_bounds.lo > 0


def test_first_receiving_material_ancestry_is_preserved_after_internal_switch(actual,carrier):
    from tectonics.geometric_native_domain_journal import evaluate_native_domain_journal_owner_point
    response = _point(carrier,actual,"transverse_masked_main_entry",depth=F(10000,3))
    assert response.allocation == "archived" and response.archive_debits == 1
    assert response.first_entry_receiver_fragment_ids == ("octant:3",)
    context = next(row["chronology"] for row in _native().source_point_rows()
                   if row["label"] == "transverse_masked_main_entry")
    saved = _native().source_documents()["material_point_queries.json"]
    times = tuple(F(*row["query"]["time_fraction"]) for row in saved
                  if row["label"] == "transverse_masked_main_entry")
    current_receivers = []
    for time in times:
        current = evaluate_native_domain_journal_owner_point(context.receiver_chronology,time)
        current_receivers.append(tuple(identity for identity,query in current.receiver_queries
                                      if query.occupancy == "inside"))
    assert ("octant:2",) in current_receivers and ("octant:3",) in current_receivers


def test_whole_donor_depth_enclosures_have_positive_jacobian_and_upper_support_is_disjoint(actual,carrier):
    api = _api()
    rows = (("octant:5",actual["points"]["swept_only_owner_entry_exit"]["ray"]),
            ("octant:6",actual["points"]["ordinary_entry"]["ray"]),
            ("octant:7",(F(1,10**9),F(1),F(1))))
    upper = _point(carrier,actual,"retained_overriding_material",depth=F(10000,2))
    for parent,ray in rows:
        whole = api.evaluate_native_archive_material_point(carrier,parent,ray,
            material_depth_bounds_m=I(0,10000),time_fraction=F(1),precision=128)
        assert whole.cohort_id == next(cohort.cohort_id for cohort in actual["state"].archive_cohorts
                                       if cohort.original_parent_id == parent)
        assert whole.volume_jacobian_bounds.lo > 0
        assert whole.radial_support_bounds_m.hi <= 100000
        assert whole.radial_support_bounds_m.hi < upper.radial_support_bounds_m.lo
        _overlaps(_det(whole.deformation_gradient_bounds),whole.volume_jacobian_bounds)
        for depth in (F(10000,3),F(20000,3)):
            inner = api.evaluate_native_archive_material_point(carrier,parent,ray,
                material_depth_bounds_m=I(depth),time_fraction=F(1),precision=128)
            assert whole.radial_support_bounds_m.lo <= inner.radial_support_bounds_m.lo
            assert inner.radial_support_bounds_m.hi <= whole.radial_support_bounds_m.hi
            assert whole.volume_jacobian_bounds.lo <= inner.volume_jacobian_bounds.lo
            assert inner.volume_jacobian_bounds.hi <= whole.volume_jacobian_bounds.hi


def test_genuine_no_entry_points_keep_live_owner_motion_and_zero_archive_debit(actual,carrier):
    for label in ("no_entry","empty_donor_four"):
        response = _point(carrier,actual,label,depth=F(10000,3))
        assert response.allocation == "live" and response.archive_debits == 0
        assert response.first_entry_time_fraction is None
        assert response.first_entry_receiver_fragment_ids == ()
        assert response.beta_bounds == I(0)
        assert response.beta_rate_per_fraction_bounds == I(0)
        assert response.volume_jacobian_bounds == I(1)
        assert response.radial_support_bounds_m == response.reference_radius_bounds_m


def test_added_down_dip_velocity_has_literal_radial_dip_and_density_conserves_mass(actual,carrier):
    response = _point(carrier,actual,"ordinary_entry",depth=F(10000,3),precision=128)
    assert response.beta_rate_per_fraction_bounds.lo > 0
    with localcontext() as context:
        context.prec = 230
        delta = _decimal(carrier.common_dip_radians)
        sine,cosine = _sin_cos(delta)
        tangent = F(sine/cosine)
    # All source/observer rotations preserve radius. Their velocity has
    # zero radial power, so only the declared extra settling contributes.
    expected = -tangent*response.beta_rate_per_fraction_bounds \
        *response.radial_support_bounds_m*response.radial_support_bounds_m/carrier.duration_s
    radial_power = _dot(response.velocity_bounds_m_per_s,response.position_bounds_m)
    _overlaps(radial_power,expected)
    assert radial_power.hi < 0
    _encloses(response.volume_jacobian_bounds*response.density_pullback_multiplier_bounds,F(1))


def test_global_certificate_uses_whole_domains_and_keeps_force_admission_separate(actual,carrier):
    certificate = carrier.whole_domain_certificate
    assert certificate.original_parent_count == 8 and certificate.nonempty_cohort_count == 3
    assert certificate.cohort_original_parent_ids == ("octant:5","octant:6","octant:7")
    assert set(certificate.complete_archive_domain_ids) == {
        cohort.archive_domain_id for cohort in actual["state"].archive_cohorts}
    assert certificate.maximum_beta_bounds == I(0,certificate.gamma/6)
    assert certificate.minimum_latitude_derivative == F(1,2)
    assert certificate.minimum_volume_jacobian_bounds.lo > 0
    assert certificate.donor_radius_upper_bound_m == certificate.receiving_base_radius_m == F(100000)
    assert dict(certificate.original_parent_thicknesses_m) == {f"octant:{i}":F(10000) for i in range(8)}
    assert certificate.swept_only_curved_domain_retained is True
    raw = _api().native_archive_material_carrier_to_dict(carrier)
    for flag in ("former_receiver_roof_or_prestress_recovered",
                 "former_source_forces_or_work_handed_over",
                 "cold_modes_globally_attached_to_new_reference",
                 "hot_constitutive_law_defined","hot_point_membership_authenticated",
                 "whole_clock_all_pair_admission_authorized",
                 "physical_material_admission_authorized",
                 "new_mixed_force_equilibrium_certified",
                 "original_point_1_completed","source_history_advanced"):
        assert raw[flag] is False
    assert raw["accepted_new_native_intervals"] == 0
    # The pure radius law does not authenticate a HOT point or its trajectory.
    with pytest.raises(ValueError,match="HOT point membership"):
        _api().evaluate_native_archive_hot_point(carrier,"invented-hot-point",
            (F(1),F(1),F(1)),I(F(1)),precision=96)


def test_restart_replays_exact_complete_model_point_depth_epoch_and_ancestry(actual,carrier):
    api = _api()
    model_raw = api.native_archive_material_carrier_to_dict(carrier)
    before = _encoded(model_raw)
    rebuilt = api.native_archive_material_carrier_from_dict(deepcopy(model_raw),
        expected_source_material_state=actual["state"])
    assert rebuilt == carrier
    assert _encoded(api.native_archive_material_carrier_to_dict(rebuilt)) == before
    response = _point(carrier,actual,"ordinary_entry",depth=F(10000,3))
    point_raw = api.native_archive_material_point_to_dict(response,system=carrier)
    row = actual["points"]["ordinary_entry"]
    replay = api.native_archive_material_point_from_dict(deepcopy(point_raw),system=rebuilt,
        expected_original_parent_id=row["parent"],expected_original_ray=row["ray"],
        expected_depth=I(F(10000,3)),expected_time_fraction=F(1),precision=96)
    assert replay == response
    assert replay.journal_time_myr == F.from_float(actual["state"].journal.source_surface.time_myr) \
        + F.from_float(actual["state"].journal.dt_myr)
    assert replay.first_entry_receiver_fragment_ids == response.first_entry_receiver_fragment_ids
    assert _encoded(api.native_archive_material_point_to_dict(replay,system=rebuilt)) == _encoded(point_raw)


def _reseal(raw,member,seal):
    raw[member][seal] = hashlib.sha256(_encoded({k:v for k,v in raw[member].items() if k != seal})).hexdigest()
    return raw


def test_resealed_models_cannot_replace_dip_mask_history_or_numeric_types(actual,carrier):
    api = _api()
    raw = api.native_archive_material_carrier_to_dict(carrier)
    changes = (
        lambda r:r["carrier"]["donor_axis"].__setitem__(0,[1,1]),
        lambda r:r["carrier"].__setitem__("common_dip_radians",[1,2]),
        lambda r:r["carrier"]["whole_domain_certificate"]["complete_archive_domain_ids"].pop(),
        lambda r:r["carrier"]["radius_m"].__setitem__(1,True),
        lambda r:r["carrier"].__setitem__("precision",96.0),
        lambda r:r.__setitem__("former_source_forces_or_work_handed_over",True),
    )
    for change in changes:
        damaged = deepcopy(raw)
        change(damaged)
        with pytest.raises(ValueError):
            api.native_archive_material_carrier_from_dict(
                _reseal(damaged,"carrier","model_sha256"),
                expected_source_material_state=actual["state"])
    # Preserve the original seal here: Python JSON equality would otherwise
    # accept True == 1 and bypass a comparison against the authentic payload.
    alias = deepcopy(raw)
    alias["carrier"]["radius_m"][1] = True
    with pytest.raises(ValueError):
        api.native_archive_material_carrier_from_dict(alias,
            expected_source_material_state=actual["state"])
    # Matching hashes or a cache hit cannot authenticate a mutated typed model.
    with pytest.raises(ValueError):
        api.native_archive_material_carrier_to_dict(replace(carrier,radius_m=F(100001)))
    damaged_source = replace(actual["state"],archive_cohorts=actual["state"].archive_cohorts[:-1])
    with pytest.raises(ValueError):
        api.native_archive_material_carrier_from_dict(raw,expected_source_material_state=damaged_source)


def test_cached_point_dto_rejects_changed_query_numeric_alias_and_resealed_ancestry(actual,carrier):
    api = _api()
    row = actual["points"]["ordinary_entry"]
    response = _point(carrier,actual,"ordinary_entry",depth=F(10000,3))
    raw = api.native_archive_material_point_to_dict(response,system=carrier)
    expected = dict(system=carrier,expected_original_parent_id=row["parent"],
        expected_original_ray=row["ray"],expected_depth=I(F(10000,3)),
        expected_time_fraction=F(1),precision=96)
    for change in (
        lambda r:r["response"]["first_entry_receiver_fragment_ids"].__setitem__(0,"octant:3"),
        lambda r:r["response"]["time_fraction"].__setitem__(1,True),
        lambda r:r["response"].__setitem__("archive_debits",0),
    ):
        damaged = deepcopy(raw)
        change(damaged)
        with pytest.raises(ValueError):
            api.native_archive_material_point_from_dict(
                _reseal(damaged,"response","response_sha256"),**expected)
    alias = deepcopy(raw)
    alias["response"]["time_fraction"][1] = True
    with pytest.raises(ValueError):
        api.native_archive_material_point_from_dict(alias,**expected)
    for changes in (dict(expected_depth=I(F(10000,2))),dict(expected_time_fraction=F(3,4)),
                    dict(world_translation_m=(F(1),F(0),F(0)))):
        with pytest.raises(ValueError):
            api.native_archive_material_point_from_dict(raw,**dict(expected,**changes))
    with pytest.raises(ValueError):
        api.native_archive_material_point_to_dict(replace(response,archive_debits=0),system=carrier)


def test_invalid_complete_depth_clock_and_parent_are_rejected_before_cached_geometry(actual,carrier):
    api = _api()
    row = actual["points"]["ordinary_entry"]
    base = dict(material_depth_bounds_m=I(F(1)),time_fraction=F(1),precision=96)
    for changes in (dict(material_depth_bounds_m=I(-1,1)),
                    dict(material_depth_bounds_m=I(0,10001)),
                    dict(material_depth_bounds_m=F(1)),dict(time_fraction=True),
                    dict(time_fraction=1.0),dict(time_fraction=F(11,10)),
                    dict(precision=96.0),dict(world_history=[])):
        with pytest.raises(ValueError):
            api.evaluate_native_archive_material_point(carrier,row["parent"],row["ray"],
                **dict(base,**changes))
    for ray in ((F(0),)*3,(True,F(1),F(1)),(1.0,F(1),F(1))):
        with pytest.raises(ValueError):
            api.evaluate_native_archive_material_point(carrier,row["parent"],ray,**base)
    with pytest.raises(ValueError):
        api.evaluate_native_archive_material_point(carrier,"octant:0",row["ray"],**base)
