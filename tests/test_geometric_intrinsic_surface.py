"""Independent controls for the explicit persisted common-motion geometry.

These controls do not claim differential cuts or physical material rematching.
The Decimal oracle evaluates the original rays and each complete requested
Rodrigues action independently of the production intrinsic representation.
"""
from dataclasses import asdict, replace
from decimal import Decimal, getcontext, localcontext
import hashlib
import json
import math

import numpy as np
import pytest

from tectonics.fractional_surface import EXTENSIVE_FIELDS
from tectonics.geometric_intrinsic_surface import (
    advance_intrinsic_surface,
    enroll_intrinsic_surface,
    IntrinsicSurfaceState,
    intrinsic_surface_to_dict,
    intrinsic_surface_from_dict,
    load_intrinsic_checkpoint,
    save_intrinsic_checkpoint,
)
from tectonics.geometric_surface import (GeometricSurfaceState, rotate_surface,
                                        save_geometric_checkpoint)
from test_geometric_precision import NEXT_DT_MYR, NEXT_OMEGA, thin_partition


PRECISION = 160
ACTUAL = NEXT_OMEGA[0][1]
STEPS = (
    (ACTUAL, NEXT_DT_MYR),
    ((.023, -.037, .049), .71),
    (NEXT_OMEGA[1][0], NEXT_DT_MYR),
    ((-.16, .09, .13), .29),
)


def _decimal(value):
    return Decimal.from_float(float(value))


def _dot(first, second):
    return sum((a*b for a, b in zip(first, second)), Decimal(0))


def _cross(first, second):
    return tuple(first[j]*second[k]-first[k]*second[j]
                 for j, k in ((1, 2), (2, 0), (0, 1)))


def _unit(ray):
    length = _dot(ray, ray).sqrt()
    return tuple(value/length for value in ray)


def _sine_cosine(angle):
    square = angle*angle
    sine_term, cosine_term = angle, Decimal(1)
    sine, cosine = sine_term, cosine_term
    threshold = Decimal(10)**(-(getcontext().prec+5))
    for index in range(1, 1000):
        sine_term *= -square/Decimal((2*index)*(2*index+1))
        cosine_term *= -square/Decimal((2*index-1)*(2*index))
        sine += sine_term
        cosine += cosine_term
        if abs(sine_term) < threshold and abs(cosine_term) < threshold:
            return sine, cosine
    raise AssertionError("Independent trigonometric oracle did not converge")


def _atan(value):
    if value < 0:
        return -_atan(-value)
    multiplier = Decimal(1)
    # Half-angle reduction also handles the octants' atan(1) efficiently.
    while value > Decimal(".125"):
        value /= Decimal(1)+(Decimal(1)+value*value).sqrt()
        multiplier *= 2
    term = value
    total = term
    square = value*value
    threshold = Decimal(10)**(-(getcontext().prec+5))
    for index in range(1, 1000):
        term *= -square
        addition = term/Decimal(2*index+1)
        total += addition
        if abs(addition) < threshold:
            return multiplier*total
    raise AssertionError("Independent area oracle did not converge")


def _area(rays):
    total = Decimal(0)
    anchor = rays[0]
    for first, second in zip(rays[1:-1], rays[2:]):
        numerator = _dot(anchor, _cross(first, second))
        denominator = (Decimal(1)+_dot(anchor, first)+_dot(first, second)
                       +_dot(second, anchor))
        assert denominator > 0
        total += Decimal(2)*_atan(numerator/denominator)
    return abs(total)


def _oracle(polygon, steps=()):
    with localcontext() as context:
        context.prec = PRECISION+15
        rays = tuple(_unit(tuple(_decimal(value) for value in ray))
                     for ray in polygon)
        body_area = _area(rays)
        for vector, duration in steps:
            omega = tuple(_decimal(value) for value in vector)
            magnitude = _dot(omega, omega).sqrt()
            if magnitude == 0:
                continue
            axis = tuple(value/magnitude for value in omega)
            sine, cosine = _sine_cosine(magnitude*_decimal(duration))
            transformed = []
            for ray in rays:
                cross = _cross(axis, ray)
                dot = _dot(axis, ray)
                transformed.append(_unit(tuple(
                    ray[index]*cosine+cross[index]*sine+
                    axis[index]*dot*(Decimal(1)-cosine) for index in range(3))))
            rays = tuple(transformed)
        return rays, body_area, _area(rays)


def _snapshot(state):
    payload = intrinsic_surface_to_dict(state) if type(state) is IntrinsicSurfaceState else asdict(state)
    return json.dumps(payload, sort_keys=True, allow_nan=False)


def _common(state, vector):
    return {fragment.parcel.plate: vector for fragment in state.fragments}


def _advance(state, steps=STEPS):
    for vector, duration in steps:
        state = advance_intrinsic_surface(state, _common(state, vector), duration)
    return state


def _material_ledger(fragments):
    ledger = {}
    for fragment in fragments:
        for field in EXTENSIVE_FIELDS:
            ledger.setdefault((fragment.parcel.material_id, field), []).append(
                getattr(fragment.parcel, field))
    return {key: math.fsum(values) for key, values in ledger.items()}


def _rewrite_payload(path, payload):
    payload.pop("payload_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    payload["payload_sha256"] = hashlib.sha256(encoded).hexdigest()
    path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False)+"\n",
                    encoding="utf-8")


def test_enrollment_keeps_every_original_shape_and_independent_area():
    source = thin_partition()
    before = _snapshot(source)
    state = enroll_intrinsic_surface(source)
    assert _snapshot(source) == before
    assert state.reference_surface == source
    assert state.phase == "partition"
    assert state.time_myr == source.time_myr
    assert state.radius_km == source.radius_km
    assert state.known_material_ids == source.known_material_ids
    assert _material_ledger(state.fragments) == _material_ledger(source.fragments)
    assert len(state.fragments) == len(source.fragments) == 9
    for old, current in zip(source.fragments, state.fragments):
        assert current.fragment_id == old.fragment_id
        assert current.parent_fragment_id == old.parent_fragment_id
        assert current.geometry.body_polygon == old.polygon
        assert current.geometry.rotations == ()
        _, expected_area, _ = _oracle(old.polygon)
        assert current.geometry.area_sr == pytest.approx(float(expected_area),
                                                        rel=2e-13, abs=0.)
    thin = next(fragment for fragment in state.fragments if fragment.fragment_id == "thin")
    assert 0. < thin.parcel.area_km2 < 1e-30
    assert not hasattr(thin, "polygon")  # No implicit legacy cutting projection.


def test_four_accepted_full_rotations_preserve_material_and_true_geometry():
    source = thin_partition()
    initial = enroll_intrinsic_surface(source)
    before = _snapshot(initial)
    current = initial
    old_by_id = {fragment.fragment_id: fragment for fragment in source.fragments}
    history = []
    for vector, duration in STEPS:
        history.append((vector, duration))
        current = advance_intrinsic_surface(current, _common(current, vector), duration)
        assert len(current.rotations) == len(history)
        assert tuple(current.rotations[-1].omega_rad_per_myr) == tuple(vector)
        assert current.rotations[-1].dt_myr == duration
        assert _material_ledger(current.fragments) == _material_ledger(source.fragments)
        assert current.time_myr == pytest.approx(math.fsum(dt for _, dt in history),
                                                rel=2e-15, abs=0.)
        for fragment in current.fragments:
            old = old_by_id[fragment.fragment_id]
            assert fragment.geometry.body_polygon == old.polygon
            assert fragment.parent_fragment_id == old.parent_fragment_id
            assert fragment.parcel.material_fields == old.parcel.material_fields
            assert fragment.parcel.age_myr == pytest.approx(
                old.parcel.age_myr+math.fsum(dt for _, dt in history), rel=2e-15)
            assert fragment.geometry.area_sr == initial.fragments[
                list(old_by_id).index(fragment.fragment_id)].geometry.area_sr
            expected, body_area, world_area = _oracle(old.polygon, history)
            # This checks the geometry, rather than reading a saved area receipt.
            with localcontext() as context:
                context.prec = PRECISION
                assert abs(world_area/body_area-1) < Decimal("1e-125")
            assert fragment.geometry.area_sr == pytest.approx(float(world_area),
                                                            rel=2e-13, abs=0.)
            # Independently measure the actual native world rays, so returning
            # only an unchanged body-area scalar cannot satisfy this control.
            native_rays = fragment.geometry.decimal_world_rays()
            with localcontext() as context:
                context.prec = PRECISION+15
                native_area = _area(native_rays)
                assert abs(native_area/body_area-1) < Decimal("1e-60")
            actual = np.asarray(fragment.geometry.world_polygon())
            assert actual.shape == (len(old.polygon), 3)
            assert np.isfinite(actual).all()
            for measured, oracle_ray in zip(actual, expected):
                for coordinate, exact in zip(measured, oracle_ray):
                    nearest = float(exact)
                    spacing = abs(np.spacing(nearest))
                    assert abs(coordinate-nearest) <= 4.*spacing
    assert _snapshot(initial) == before


def test_off_axis_motion_is_retained_and_world_projection_does_not_authorize_legacy_state():
    source = thin_partition()
    state = enroll_intrinsic_surface(source)
    moved = advance_intrinsic_surface(state, _common(state, ACTUAL), NEXT_DT_MYR)
    thin = next(fragment for fragment in moved.fragments if fragment.fragment_id == "thin")
    assert ACTUAL[0] != 0. and ACTUAL[1] != 0.
    assert tuple(moved.rotations[0].omega_rad_per_myr) == ACTUAL
    expected, _, _ = _oracle(thin.geometry.body_polygon, ((ACTUAL, NEXT_DT_MYR),))
    actual = np.asarray(thin.geometry.world_polygon())
    assert abs(actual[0, 2]) > 1e-26
    assert abs(actual[0, 2]-float(expected[0][2])) <= 4.*abs(np.spacing(float(expected[0][2])))
    # Ordinary geometry remains subject to its existing strict area contract.
    common = {owner: ACTUAL for owner in (0, 1)}
    with pytest.raises(ValueError, match="footprint disagree"):
        rotate_surface(source, common, NEXT_DT_MYR)
    with pytest.raises(ValueError, match="footprint disagree"):
        GeometricSurfaceState(source.time_myr, source.radius_km,
            tuple(replace(fragment,
                polygon=tuple(map(tuple, counterpart.geometry.world_polygon())))
                for fragment, counterpart in zip(source.fragments, moved.fragments)),
            source.known_material_ids)


def test_native_checkpoint_bytes_and_complete_resumed_trajectory_are_exact(tmp_path):
    initial = enroll_intrinsic_surface(thin_partition())
    partial = _advance(initial, STEPS[:2])
    provenance = {"control": "четыре общих поворота", "physical_rematching": False}
    saved = save_intrinsic_checkpoint(tmp_path/"partial.json", partial, provenance=provenance)
    restored, restored_provenance = load_intrinsic_checkpoint(saved)
    assert restored_provenance == provenance
    assert _snapshot(restored) == _snapshot(partial)
    resaved = save_intrinsic_checkpoint(tmp_path/"partial_resaved.json", restored,
                                       provenance=restored_provenance)
    assert saved.read_bytes() == resaved.read_bytes()
    direct = _advance(initial)
    resumed = _advance(restored, STEPS[2:])
    assert _snapshot(direct) == _snapshot(resumed)
    direct_path = save_intrinsic_checkpoint(tmp_path/"direct.json", direct, provenance=provenance)
    resumed_path = save_intrinsic_checkpoint(tmp_path/"resumed.json", resumed, provenance=provenance)
    assert direct_path.read_bytes() == resumed_path.read_bytes()
    final, final_provenance = load_intrinsic_checkpoint(direct_path)
    final_path = save_intrinsic_checkpoint(tmp_path/"final_resaved.json", final,
                                          provenance=final_provenance)
    assert direct_path.read_bytes() == final_path.read_bytes()


def test_native_dictionary_roundtrip_is_json_native_and_complete():
    state = _advance(enroll_intrinsic_surface(thin_partition()), STEPS[:2])
    payload = intrinsic_surface_to_dict(state)
    assert payload == json.loads(json.dumps(payload, allow_nan=False))
    restored = intrinsic_surface_from_dict(payload)
    assert restored == state
    assert _snapshot(restored) == _snapshot(state)


@pytest.mark.parametrize("corruption", ["body_width", "material_area", "negative_pose_dt",
                                        "boolean_pose_omega", "extra_field", "wrong_version"])
def test_rehashed_native_checkpoint_still_validates_geometry_material_and_pose(tmp_path, corruption):
    state = _advance(enroll_intrinsic_surface(thin_partition()), STEPS[:1])
    path = save_intrinsic_checkpoint(tmp_path/"original.json", state)
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload["state"]
    thin = next(fragment for fragment in raw["reference_surface"]["fragments"]
                if fragment["fragment_id"] == "thin")
    if corruption == "body_width":
        thin["polygon"][0][0] *= 2.
    elif corruption == "material_area":
        thin["parcel"]["area_km2"] *= 2.
    elif corruption == "negative_pose_dt":
        raw["rotations"][0]["dt_myr"] = -1.
    elif corruption == "boolean_pose_omega":
        raw["rotations"][0]["omega_rad_per_myr"][0] = True
    elif corruption == "extra_field":
        raw["rotations"][0]["unvalidated_pose"] = True
    else:
        raw["version"] = "unsupported-intrinsic-surface"
    corrupt_path = tmp_path/"rehashed_corrupt.json"
    _rewrite_payload(corrupt_path, payload)
    original_bytes = path.read_bytes()
    with pytest.raises(ValueError):
        load_intrinsic_checkpoint(corrupt_path)
    assert path.read_bytes() == original_bytes


def test_differential_motion_is_an_explicit_atomic_boundary_of_this_format(tmp_path):
    state = enroll_intrinsic_surface(thin_partition())
    before = _snapshot(state)
    path = save_intrinsic_checkpoint(tmp_path/"accepted.json", state)
    bytes_before = path.read_bytes()
    with pytest.raises(ValueError):
        advance_intrinsic_surface(state, NEXT_OMEGA[0], NEXT_DT_MYR)
    assert _snapshot(state) == before
    assert path.read_bytes() == bytes_before


@pytest.mark.parametrize("duration", [0., -1., True, float("nan"), float("inf"), "1"])
def test_invalid_native_timestep_does_not_change_accepted_state(duration):
    state = enroll_intrinsic_surface(thin_partition())
    before = _snapshot(state)
    with pytest.raises(ValueError):
        advance_intrinsic_surface(state, _common(state, ACTUAL), duration)
    assert _snapshot(state) == before


def test_native_state_cannot_be_written_with_legacy_format_or_mutated(tmp_path):
    state = enroll_intrinsic_surface(thin_partition())
    parent = tmp_path/"not_created"
    with pytest.raises(TypeError):
        save_geometric_checkpoint(parent/"bad_legacy.json", state)
    assert not parent.exists()
    with pytest.raises(AttributeError, match="immutable"):
        state.rotations = ()
    with pytest.raises(AttributeError, match="immutable"):
        del state.rotations
    # The explicit writer also checks derived fields before creating a path.
    object.__setattr__(state, "_time_myr", state.time_myr+1.)
    with pytest.raises(ValueError, match="derived material state"):
        advance_intrinsic_surface(state, _common(state, ACTUAL), NEXT_DT_MYR)
    with pytest.raises(ValueError, match="derived material state"):
        save_intrinsic_checkpoint(parent/"bad_native.json", state)
    assert not parent.exists()
