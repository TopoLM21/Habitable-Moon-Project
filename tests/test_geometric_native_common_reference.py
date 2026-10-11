"""Independent kinematics and actual-source boundary checks for the NEW map."""
from dataclasses import replace
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
from functools import lru_cache
import json

import pytest

from tectonics import geometric_native_common_reference as api
from tectonics.geometric_native_pose import FractionInterval as I


@lru_cache(maxsize=1)
def reference():
    return api.bind_native_common_reference()


def decimal(value):
    value = value.midpoint if type(value) is I else value
    value = F(value)
    return D(value.numerator)/D(value.denominator)


def sin_cos(x):
    sine, cosine, s, c = x, D(1), x, D(1)
    for n in range(1, 180):
        s *= -x*x/D(2*n*(2*n+1))
        c *= -x*x/D((2*n-1)*2*n)
        sine += s
        cosine += c
    return sine, cosine


def dot(a, b):
    return sum((x*y for x, y in zip(a, b)), D(0))


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def rotate_vector(x, axis, time):
    """Independent Decimal Rodrigues action on vectors, no production Q matrix."""
    norm = dot(axis, axis).sqrt()
    sine, cosine = sin_cos(norm*time)
    ax = cross(axis, x)
    aax = cross(axis, ax)
    return tuple(v+sine/norm*w+(1-cosine)/(norm*norm)*z for v, w, z in zip(x, ax, aax))


def direct_map(x, radius, axis, inverse=False):
    r = dot(x, x).sqrt()
    tau = (radius**3-r**3)/(3*radius**2)
    return rotate_vector(x, axis, -tau if inverse else tau)


def close_to_interval(value, bounds, tolerance=D("1e-65")):
    assert decimal(bounds.lo)-tolerance <= value <= decimal(bounds.hi)+tolerance


def test_complete_actual_source_bindings_and_historical_groups_remain_literal():
    r = reference()
    source = api.load_native_normal_support_source()
    assert r.complete_source_dto_json == source.complete_source_dto_json
    assert r.saved_field_json == source.saved_field_json
    assert len(r.banks) == 13
    assert sum(b.material_kind == "original_live" for b in r.banks) == 8
    assert sum(b.material_kind == "hot_birth" for b in r.banks) == 5
    original = {b.component_id: b for b in source.banks}
    assert len({b.bank_snapshot_sha256 for b in r.banks}) == 13
    assert sum(any(b.historical_angular_rate_per_physical_volume_depth_m) for b in r.banks) == 6
    assert all(b.angular_rate_per_physical_volume_depth_m == r.angular_rate_per_physical_volume_depth_m for b in r.banks)
    for b in r.banks:
        assert b.complete_mask_source_json == original[b.component_id].complete_mask_source_json
        assert b.areal_material_thickness_m == original[b.component_id].areal_material_thickness_m
        assert b.roof_radial_displacement_m == original[b.component_id].roof_radial_displacement_m
        assert b.saved_field_json == source.saved_field_json
        assert b.reference_volume_bounds_m3 == b.actual_area_bounds_sr*b.radius_m**2*b.areal_material_thickness_m
    replay = r.source_partition_replay
    assert replay.strict_triple_count == 84
    assert replay.strict_feasible_cell_count == 74
    assert replay.source_component_count == 13
    assert replay.curved_extra_area_bounds_sr.lo > 0
    assert replay.curved_extra_area_bounds_sr.hi < F(1, 10**70)
    assert r.initial_all13_positive_volume_exclusion_certified
    assert not r.elapsed_physical_time_added
    assert not r.whole_clock_all_pairs_admission_certified
    assert not r.physical_material_admission_authorized


def test_roundtrip_replays_complete_independent_source_and_refuses_resealed_forgery():
    r = reference()
    raw = api.native_common_reference_to_dict(r)
    assert api.native_common_reference_from_dict(raw) == r
    changed = json.loads(json.dumps(raw))
    changed["reference"]["banks"][0]["roof_radial_displacement_m"] = [1, 1]
    with pytest.raises(ValueError, match="source replay"):
        api.native_common_reference_from_dict(changed)
    bank = r.banks[0]
    forged = replace(bank, angular_rate_per_physical_volume_depth_m=(F(0),)*3)
    forged = replace(forged, bank_snapshot_sha256=api._bank_seal(forged))
    with pytest.raises(ValueError, match="altered common bank"):
        api.validate_native_common_reference_bank(forged)
    forged_ref = replace(r, saved_field_json="{}")
    forged_ref = replace(forged_ref, reference_snapshot_sha256=api._reference_seal(forged_ref))
    with pytest.raises(ValueError, match="complete source authentication"):
        api.validate_native_common_reference(forged_ref)


def test_all13_full_depth_points_use_tau_top_plus_z_with_independent_rodrigues_oracle():
    r = reference()
    ray = (I(F(3, 5)), I(F(4, 5)), I(0))
    with localcontext() as context:
        context.prec = 120
        axis = tuple(decimal(a) for a in r.angular_rate_per_physical_volume_depth_m)
        R = decimal(r.radius_m)
        for bank in r.banks:
            z = bank.areal_material_thickness_m/F(3)
            point = api.bound_native_common_reference_bank_point(bank, ray, I(z), precision=128, order=28)
            top = R+decimal(bank.roof_radial_displacement_m)
            radius = (top**3-3*R**2*decimal(z))**(D(1)/D(3))
            tau = (R**3-top**3)/(3*R**2)+decimal(z)
            expected = rotate_vector(tuple(radius*decimal(u) for u in ray), axis, tau)
            for a, bounds in zip(expected, point.world_position_bounds_m):
                close_to_interval(a, bounds)
            close_to_interval(tau, point.physical_volume_depth_bounds_m)
            assert point.determinant_exact == 1
            assert point.signed_material_chart_jacobian_exact == -r.radius_m**2
            assert not point.material_membership_authenticated
            assert not point.physical_material_admission_authorized


def test_cartesian_inverse_and_differential_match_independent_numerical_derivatives():
    r = reference()
    x = (F(-57000), F(76000), F(1900))
    image = api.bound_native_common_reference_point(r, tuple(I(v) for v in x), precision=128, order=64)
    inverse = api.bound_native_common_reference_inverse(r, image.world_position_bounds_m, precision=128, order=64)
    assert all(bounds.lo <= value <= bounds.hi for value, bounds in zip(x, inverse.world_position_bounds_m))
    with localcontext() as context:
        context.prec = 160
        original = tuple(decimal(v) for v in x)
        R = decimal(r.radius_m)
        axis = tuple(decimal(a) for a in r.angular_rate_per_physical_volume_depth_m)
        expected = direct_map(original, R, axis)
        for value, bounds in zip(expected, image.world_position_bounds_m):
            close_to_interval(value, bounds, D("1e-115"))
        h = D("1e-35")
        for j in range(3):
            plus = tuple(v+(h if k == j else 0) for k, v in enumerate(original))
            minus = tuple(v-(h if k == j else 0) for k, v in enumerate(original))
            upper, lower = direct_map(plus, R, axis), direct_map(minus, R, axis)
            for i in range(3):
                numerical = (upper[i]-lower[i])/(2*h)
                assert abs(numerical-decimal(image.reference_gradient_bounds[i][j])) < D("1e-65")
        # r=R has tau=0 and is fixed, even though its radial differential has shear.
        on_sphere = api.bound_native_common_reference_point(r, (I(r.radius_m), I(0), I(0)), precision=128, order=64)
        assert all(bounds.lo <= value <= bounds.hi for value, bounds in zip((r.radius_m, F(0), F(0)), on_sphere.world_position_bounds_m))
        assert on_sphere.physical_volume_depth_bounds_m.lo <= 0 <= on_sphere.physical_volume_depth_bounds_m.hi


def test_different_roofs_match_at_common_radius_but_not_same_material_depth():
    r = reference()
    cold = next(b for b in r.banks if b.material_kind == "original_live" and b.roof_radial_displacement_m > 0)
    hot = next(b for b in r.banks if b.material_kind == "hot_birth")
    ray = (I(1), I(0), I(0))
    cubes = F(99500)**3
    zd = ((r.radius_m+cold.roof_radial_displacement_m)**3-cubes)/(3*r.radius_m**2)
    zr = ((r.radius_m+hot.roof_radial_displacement_m)**3-cubes)/(3*r.radius_m**2)
    assert zd != zr
    p = api.bound_native_common_reference_bank_point(cold, ray, I(zd), precision=128, order=28)
    q = api.bound_native_common_reference_bank_point(hot, ray, I(zr), precision=128, order=28)
    assert p.world_position_bounds_m == q.world_position_bounds_m
    assert p.physical_volume_depth_bounds_m == q.physical_volume_depth_bounds_m
    wrong = api.bound_native_common_reference_bank_point(hot, ray, I(zd), precision=128, order=28)
    assert wrong.radius_bounds_m.hi < p.radius_bounds_m.lo


@pytest.mark.parametrize("family_id", ["born-shared-0", "born-shared-3"])
def test_whole_born_faces_use_true_shared_boundaries_and_physical_radius(family_id):
    r = reference()
    bounds = api.bound_native_common_reference_face(r, family_id, precision=128, order=28)
    assert bounds.face.true_source_boundary_of_both_actual_masks_certified
    assert bounds.whole_common_reference_face_enclosed
    assert bounds.reference_faces_match_at_every_common_physical_radius
    donor = api.bind_native_common_reference_bank(r, bounds.face.donor_component_id)
    receiver = api.bind_native_common_reference_bank(r, bounds.face.receiver_component_id)
    assert (bounds.donor_material_depth_bounds_m == bounds.receiver_material_depth_bounds_m) == (
        donor.roof_radial_displacement_m == receiver.roof_radial_displacement_m)
    assert bounds.area_density_bounds_m2_per_radian_per_material_m.lo > 0
    assert bounds.common_physical_radius_cubed_bounds_m3.width > 0
    assert bounds.source_declared_birth_gap_m == F.from_float(-0.01)
    assert not bounds.geometric_gluing_supplies_constitutive_rest_gap
    assert not bounds.source_constant_dip_reproduced_on_whole_face
    assert not bounds.every_point_of_parameter_enclosure_is_material
    assert not bounds.physical_material_admission_authorized
    with pytest.raises(ValueError, match="genuine boundary authentication"):
        api.validate_native_common_reference_face(replace(bounds.face, donor_component_id=bounds.face.receiver_component_id))


@pytest.mark.parametrize("family_id", ["born-shared-0", "born-shared-3"])
def test_face_area_and_normal_match_independent_parametric_cross_product(family_id):
    r = reference()
    face = api.bind_native_common_reference_face(r, family_id, precision=128)
    donor = api.bind_native_common_reference_bank(r, face.donor_component_id)
    receiver = api.bind_native_common_reference_bank(r, face.receiver_component_id)
    low = max(donor.layer.layer.bottom_radius_cubed_m3, receiver.layer.layer.bottom_radius_cubed_m3)
    high = min(donor.layer.layer.top_radius_m**3, receiver.layer.layer.top_radius_m**3)
    cubes = (low+high)/2
    theta = face.native_angle_end_bounds_radians.lo/2
    observed = api.bound_native_common_reference_face(r, face, angle_bounds_radians=I(theta),
        physical_radius_cubed_bounds_m3=I(cubes), precision=128, order=64)
    with localcontext() as context:
        context.prec = 120
        R, radius = decimal(r.radius_m), decimal(cubes)**(D(1)/D(3))
        A = tuple(decimal(a) for a in r.angular_rate_per_physical_volume_depth_m)
        st, ct = sin_cos(decimal(theta))
        S = tuple(decimal(v) for v in face.start_unit_bounds)
        T0 = tuple(decimal(v) for v in face.tangent_unit_bounds)
        U = tuple(ct*s+st*t for s, t in zip(S, T0))
        T = tuple(-st*s+ct*t for s, t in zip(S, T0))
        tau = (R**3-decimal(cubes))/(3*R**2)
        xtheta = rotate_vector(tuple(radius*t for t in T), A, tau)
        xz = rotate_vector(tuple(-R**2/radius**2*u+radius*v for u, v in zip(U, cross(A, U))), A, tau)
        oriented = tuple(face.receiving_normal_orientation*v for v in cross(xtheta, xz))
        area = dot(oriented, oriented).sqrt()
        close_to_interval(area, observed.area_density_bounds_m2_per_radian_per_material_m, D("1e-105"))
        for actual, bounds in zip(oriented, observed.receiver_inward_unit_normal_bounds):
            close_to_interval(actual/area, bounds, D("1e-105"))
        for actual, bounds in zip(xtheta, observed.angular_tangent_bounds_m):
            close_to_interval(actual, bounds, D("1e-105"))
        for actual, bounds in zip(xz, observed.material_depth_tangent_bounds):
            close_to_interval(actual, bounds, D("1e-105"))


def test_all_eight_clamps_are_true_live_material_and_map_full_depth_not_only_surface():
    r = reference()
    source_before = json.loads(dict(r.complete_source_dto_json)["before"])
    journal = source_before["snapshot"]["current_material_epoch"]["source_material_state"]["journal"]
    owner_axes = dict(journal["omega_rad_per_myr"])
    for bank in r.banks:
        if bank.material_kind != "original_live":
            continue
        clamp = api.bound_native_common_reference_clamp(r, bank.component_id, precision=128, order=28)
        assert clamp.genuine_live_material_clamp_support_certified
        assert clamp.transported_source_material_clamp
        assert not clamp.source_world_clamp_kept_fixed
        assert clamp.material_depth_bounds_m == I(0, bank.areal_material_thickness_m)
        assert clamp.common_physical_volume_depth_bounds_m.width == bank.areal_material_thickness_m
        assert clamp.all_seven_modal_fields_and_gradients_zero
        assert clamp.common_physical_volume_depth_bounds_m.hi > 9000
        if clamp.source_bank == "sub":
            assert clamp.source_live_support_inequality_upper < 0
        assert not clamp.physical_material_admission_authorized
        # Independent true material clamp points through all depths validate
        # the broad interval rotation, including negative tau at positive roofs.
        with localcontext() as context:
            context.prec = 110
            theta = D(1)/3
            st, ct = sin_cos(theta)
            sb, cb = sin_cos(decimal(clamp.beta_cut_radians))
            trace = json.loads(clamp.complete_source_clamp_row_json)["trace_id"]
            material = source_before["snapshot"]["current_material_epoch"]["source_material_state"]
            trace_source = next(t for t in material["original_mechanics_payload"]["fault_traces"]["traces"] if t["trace_id"] == trace)
            S, T = (tuple(D.from_float(v) for v in trace_source[k]) for k in ("start", "end"))
            N = tuple((-1 if clamp.source_bank == "sub" else 1)*v for v in cross(S, T))
            original = tuple(cb*(ct*s+st*t)+sb*n for s, t, n in zip(S, T, N))
            source_direction = rotate_vector(original, tuple(D.from_float(v) for v in owner_axes[bank.owner]),
                                            D.from_float(journal["dt_myr"]))
            R = decimal(r.radius_m)
            top = R+decimal(bank.roof_radial_displacement_m)
            A = tuple(decimal(a) for a in r.angular_rate_per_physical_volume_depth_m)
            for z in (F(0), bank.areal_material_thickness_m/2, bank.areal_material_thickness_m):
                cube = top**3-3*R**2*decimal(z)
                radius = cube**(D(1)/D(3))
                tau = (R**3-cube)/(3*R**2)
                actual = rotate_vector(tuple(radius*x for x in source_direction), A, tau)
                for value, bound in zip(actual, clamp.mapped_world_position_bounds_m):
                    close_to_interval(value, bound, D("1e-90"))
    hot = next(b for b in r.banks if b.material_kind == "hot_birth")
    with pytest.raises(ValueError, match="HOT material cannot replace"):
        api.bound_native_common_reference_clamp(r, hot.component_id)


def test_carried_original_sheets_preserve_histories_and_are_not_silently_glued():
    r = reference()
    before = json.loads(dict(r.complete_source_dto_json)["before"])
    material = before["snapshot"]["current_material_epoch"]["source_material_state"]
    trace = material["trace_bindings"][0]["trace_id"]
    # One physical radius and identical original theta remain different under
    # the genuine two owner histories; invertible common Phi cannot erase it.
    kwargs = dict(angle_bounds_radians=I(F(1, 3)), physical_radius_cubed_bounds_m3=I(F(99500)**3),
                  precision=128, order=28)
    donor = api.bound_native_common_reference_carried_original_face(r, trace, "sub", **kwargs)
    receiver = api.bound_native_common_reference_carried_original_face(r, trace, "over", **kwargs)
    assert any(a.hi < b.lo or b.hi < a.lo for a, b in zip(donor.world_position_bounds_m, receiver.world_position_bounds_m))
    for row in (donor, receiver):
        assert json.loads(row.complete_original_mechanical_history_json) == before["original_physical_history"]
        assert row.old_sixteen_constitutive_histories_preserved
        assert row.carried_original_material_sheet_enclosed
        assert not row.whole_sheet_is_retained_live_contact_interface
        assert not row.donor_and_receiver_historical_sheets_glued
        assert not row.physical_material_admission_authorized


def test_invalid_zero_radius_and_source_extent_queries_are_refused():
    r = reference()
    with pytest.raises(ValueError, match="positive-radius box"):
        api.bound_native_common_reference_inverse(r, (I(0), I(0), I(0)))
    bank = r.banks[0]
    with pytest.raises(ValueError, match="full literal source depth"):
        api.bound_native_common_reference_bank_point(bank, (I(1), I(0), I(0)), I(bank.areal_material_thickness_m+1))
    with pytest.raises(ValueError, match="genuine source component"):
        api.bind_native_common_reference_bank(r, "made-up-bank")
