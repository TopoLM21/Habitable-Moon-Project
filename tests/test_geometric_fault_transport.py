"""Independent finite-rotation, conservation and hostile restart checks."""
from dataclasses import replace
from functools import lru_cache
import copy
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tests.test_geometric_distributed_contact_invariants import world, traces
from tectonics.geometric_bending import ElasticBendingBankParameters, FaultBendingParameters
from tectonics.geometric_boundary import surface_digest
from tectonics.geometric_fault_traces import (TraceQuadrature, TracePolarityEvidence,
    declare_fault_trace_set)
from tectonics.geometric_fault_transport import (
    CARRIER, STATUS, initialize_fault_material_atlas, advance_fault_material_atlas,
    material_atlas_surface, transported_fault_geometry, validate_fault_material_atlas,
    fault_material_atlas_to_dict, fault_material_atlas_from_dict, _hash, _body,
)
from tectonics.geometric_polarity import PolarityEvidence
from tectonics.geometric_surface import totals, audit_partition


@lru_cache(maxsize=1)
def origin():
    _, surface, _ = world()
    trace_set = declare_fault_trace_set(surface, traces(), TraceQuadrature(1, 2))
    sub = ElasticBendingBankParameters(10000., 2000., 7e10, .25, "Declared sub clamp")
    over = ElasticBendingBankParameters(15000., 3000., 8e10, .3, "Declared over clamp")
    parameters = tuple(FaultBendingParameters(trace.trace_id, sub, over)
                       for trace in trace_set.traces)
    trace = trace_set.traces[0]
    history = (TracePolarityEvidence(PolarityEvidence("old:event", "old:material:cut",
        trace.subducting_plate, trace.overriding_plate, "Retained original acceptance"),
        trace.trace_id, trace.start, trace.end, surface_digest(surface)),)
    return surface, initialize_fault_material_atlas(surface, trace_set, parameters, history)


def moved(atlas=None, vectors=((0, (.2, -.1, .3)), (1, (-.3, .15, .12))), dt=.01):
    surface, initial = origin()
    return advance_fault_material_atlas(surface, initial if atlas is None else atlas,
        vectors, dt, provenance="Explicit synthetic finite material rotation")


def _rehash(raw):
    raw["payload_sha256"] = _hash({key: value for key, value in raw.items()
                                   if key != "payload_sha256"})
    return raw


def test_origin_atlas_preserves_labels_clamps_history_and_original_coordinates():
    surface, atlas = origin()
    assert atlas.status == STATUS and atlas.carrier_convention == CARRIER
    assert atlas.plate_ids == (0, 1)
    assert atlas.origin_surface_sha256 == atlas.current_surface_sha256 == surface_digest(surface)
    assert material_atlas_surface(surface, atlas) == surface
    geometry = transported_fault_geometry(surface, atlas)
    assert geometry.points == atlas.reference_points
    assert geometry.history == atlas.polarity_history
    for point, reference in zip(geometry.bank_points, atlas.reference_bank_clamps):
        assert point.sub_edge == point.over_edge == reference.edge
        assert point.sub_clamp == reference.sub_clamp
        assert point.over_clamp == reference.over_clamp
        assert point.bank_separation_m == 0.
        for clamp, width in ((reference.sub_clamp, reference.sub_width_m),
                             (reference.over_clamp, reference.over_width_m)):
            angle = math.atan2(np.linalg.norm(np.cross(reference.edge, clamp)),
                               np.asarray(reference.edge)@clamp)
            assert angle*surface.radius_km*1000. == pytest.approx(width, rel=2e-15)


def test_noncommuting_finite_rotations_compose_in_chronological_order():
    surface, initial = origin()
    first = moved(vectors=((0, (.5, 0., 0.)), (1, (0., .7, 0.))))
    second = moved(first, ((0, (0., -.8, 0.)), (1, (.4, 0., .2))), .03)
    expected = {
        0: Rotation.from_rotvec((0., -.8, 0.)).as_matrix()@Rotation.from_rotvec((.5, 0., 0.)).as_matrix(),
        1: Rotation.from_rotvec((.4, 0., .2)).as_matrix()@Rotation.from_rotvec((0., .7, 0.)).as_matrix(),
    }
    for owner, matrix in second.owner_rotations:
        np.testing.assert_array_equal(matrix, expected[owner])
    summed = Rotation.from_rotvec((.5, -.8, 0.)).as_matrix()
    assert np.linalg.norm(expected[0]-summed) > .2
    assert second.elapsed_time_myr == .04
    assert second.time_myr == surface.time_myr+.04
    assert initial.motion_receipts == ()
    assert initial.owner_rotations != second.owner_rotations


def test_each_bank_label_and_clamp_use_its_own_rotation_plane_uses_overriding():
    surface, initial = origin()
    atlas = moved()
    geometry = transported_fault_geometry(surface, atlas)
    maps = dict(atlas.owner_rotations)
    reference = {point.point_id: point for point in initial.reference_points}
    for bank, point, clamp in zip(geometry.bank_points, geometry.points, initial.reference_bank_clamps):
        np.testing.assert_allclose(bank.sub_edge, np.asarray(maps[bank.subducting_plate])@clamp.edge, atol=2e-16)
        np.testing.assert_allclose(bank.over_edge, np.asarray(maps[bank.overriding_plate])@clamp.edge, atol=2e-16)
        assert point.position == bank.over_edge
        for name in ("horizontal", "normal", "down_dip"):
            np.testing.assert_allclose(getattr(point, name),
                np.asarray(maps[bank.overriding_plate])@getattr(reference[point.point_id], name), atol=2e-16)
        assert bank.sub_edge != bank.over_edge
        assert bank.sub_radial == bank.sub_edge and bank.over_radial == bank.over_edge
        assert point.area_m2 == reference[point.point_id].area_m2
    for old, new in zip(initial.polarity_history, geometry.history):
        assert old.origin == new.origin
        assert old.origin_surface_sha256 == new.origin_surface_sha256
        assert old.start != new.start


def test_material_reservoirs_ids_properties_preserved_without_overlap_resolution():
    surface, initial = origin()
    candidate = material_atlas_surface(surface, moved())
    assert candidate.phase == "advected_candidates"
    assert totals(candidate.fragments) == totals(surface.fragments)
    assert candidate.known_material_ids == surface.known_material_ids
    for old, new in zip(surface.fragments, candidate.fragments):
        assert old.fragment_id == new.fragment_id and old.parent_fragment_id == new.parent_fragment_id
        assert replace(new.parcel, age_myr=old.parcel.age_myr) == old.parcel
        assert new.parcel.age_myr == old.parcel.age_myr+.01
    with pytest.raises(ValueError, match="not resolved"):
        audit_partition(candidate)
    assert material_atlas_surface(surface, initial) == surface


def test_common_rotation_is_resolved_partition_and_retains_zero_pair_mismatch():
    surface, _ = origin()
    atlas = moved(vectors=((0, (.7, .4, -.3)), (1, (.7, .4, -.3))))
    candidate = material_atlas_surface(surface, atlas)
    assert candidate.phase == "partition"
    audit_partition(candidate)
    geometry = transported_fault_geometry(surface, atlas)
    assert all(item.bank_maps_identical and item.maximum_bank_separation_m == 0.
               for item in geometry.trace_mismatches)
    assert all(item.sub_edge == item.over_edge for item in geometry.bank_points)


def test_sixty_exact_common_increments_replay_geometry_without_so3_projection():
    surface, initial = origin()
    vectors = ((0, (.2, -.1, .3)), (1, (.2, -.1, .3)))
    increment = Rotation.from_rotvec(vectors[0][1]).as_matrix()
    expected = np.eye(3)
    atlas = initial
    for _ in range(60):
        expected = increment@expected
        atlas = advance_fault_material_atlas(surface, atlas, vectors, .001,
            provenance="Explicit repeated finite common material rotation")
    assert len(atlas.motion_receipts) == 60
    # The actual chronological products retain their accumulating last bits.
    assert abs(float(np.linalg.det(expected))-1.) > 64.*np.finfo(float).eps
    for _, matrix in atlas.owner_rotations:
        np.testing.assert_array_equal(matrix, expected)
    geometry = transported_fault_geometry(surface, atlas)
    assert all(item.bank_maps_identical and item.maximum_bank_separation_m == 0.
               for item in geometry.trace_mismatches)
    assert all(item.sub_edge == item.over_edge for item in geometry.bank_points)
    for original, moved_point in zip(initial.reference_points, geometry.points):
        np.testing.assert_array_equal(moved_point.position, expected@original.position)
    restored = fault_material_atlas_from_dict(fault_material_atlas_to_dict(surface, atlas), surface)
    assert restored == atlas
    assert transported_fault_geometry(surface, restored) == geometry


def test_continuous_pair_mismatch_includes_interior_maximum_missed_by_endpoints():
    surface, _ = origin()
    axis = np.array((1., -1., 0.))/math.sqrt(2.)
    atlas = moved(vectors=((0, tuple(.6*axis)), (1, (0., 0., 0.))))
    geometry = transported_fault_geometry(surface, atlas)
    trace = atlas.reference_traces.traces[0]
    mismatch = geometry.trace_mismatches[0]
    # The first arc midpoint is parallel to this axis and has zero mismatch;
    # the neighboring arc has an interior perpendicular direction and maximum.
    neighbor = geometry.trace_mismatches[1]
    assert neighbor.maximum_bank_separation_m > max(neighbor.endpoint_separation_m)*1.3
    expected = 2.*surface.radius_km*1000.*math.sin(.3)
    assert neighbor.maximum_bank_separation_m == pytest.approx(expected, rel=2e-15)
    maps = dict(atlas.owner_rotations)
    for declared, result in zip(atlas.reference_traces.traces, geometry.trace_mismatches):
        start = np.asarray(declared.start)
        tangent = np.cross(declared.normal_sub_to_over, start)
        theta = math.atan2(np.linalg.norm(np.cross(declared.start, declared.end)),
                           start@declared.end)
        samples = np.linspace(0., theta, 4001)
        positions = np.outer(np.cos(samples), start)+np.outer(np.sin(samples), tangent)
        independent = np.linalg.norm(positions@(np.asarray(maps[0])-maps[1]).T, axis=1)
        assert result.maximum_bank_separation_m >= max(independent)*surface.radius_km*1000.*(1.-2e-15)
        assert result.maximum_bank_separation_m == pytest.approx(max(independent)*surface.radius_km*1000., rel=1e-7)


def test_restart_replays_original_geometry_without_last_bit_renormalization():
    surface, _ = origin()
    middle = moved()
    raw = fault_material_atlas_to_dict(surface, middle)
    restored = fault_material_atlas_from_dict(copy.deepcopy(raw), surface)
    assert restored == middle
    assert fault_material_atlas_to_dict(surface, restored) == raw
    vectors = ((0, (.1, -.2, .07)), (1, (-.05, .21, .13)))
    direct, resumed = moved(middle, vectors, .02), moved(restored, vectors, .02)
    assert resumed == direct
    assert material_atlas_surface(surface, resumed) == material_atlas_surface(surface, direct)
    assert transported_fault_geometry(surface, resumed) == transported_fault_geometry(surface, direct)


@pytest.mark.parametrize("vectors", [
    ((0, (0., 0., 0.)),),
    ((1, (0., 0., 0.)), (0, (0., 0., 0.))),
    ((0, (0., 0., 0.)), (0, (0., 0., 0.))),
    ((False, (0., 0., 0.)), (1, (0., 0., 0.))),
    ((0, (True, 0., 0.)), (1, (0., 0., 0.))),
    ((0, (float("nan"), 0., 0.)), (1, (0., 0., 0.))),
])
def test_invalid_or_incomplete_compact_rotation_maps_rejected_atomically(vectors):
    surface, atlas = origin()
    raw = fault_material_atlas_to_dict(surface, atlas)
    with pytest.raises(ValueError):
        moved(vectors=vectors)
    assert fault_material_atlas_to_dict(surface, atlas) == raw


@pytest.mark.parametrize("dt", [0., -.01, True, float("nan")])
def test_transport_requires_positive_real_advancing_clock(dt):
    with pytest.raises(ValueError):
        moved(dt=dt)


@pytest.mark.parametrize("mutate,match", [
    (lambda raw: raw.update(carrier_convention="subducting_bank"), "carrier"),
    (lambda raw: raw.update(status="accepted"), "status"),
    (lambda raw: raw.update(elapsed_time_myr=.02), "elapsed"),
    (lambda raw: raw.update(time_myr=True), "clock"),
    (lambda raw: raw["owner_rotations"][0][1][0].__setitem__(0, -1.), "SO3"),
    (lambda raw: raw["owner_rotations"][0][1].__setitem__(0, [1., 0., 0.]), "SO3"),
    (lambda raw: raw["reference_bank_clamps"][0].update(sub_width_m=20000.), "clamps"),
    (lambda raw: raw["reference_points"][0].update(point_id="invented"), "labels"),
    (lambda raw: raw["reference_points"][0].update(panel_index=False), "indices"),
    (lambda raw: raw["reference_bank_clamps"][0].update(subducting_plate=False), "owner"),
    (lambda raw: raw.update(current_surface_sha256="0"*64), "current surface"),
    (lambda raw: raw["motion_receipts"][0].update(start_time_myr=True), "clock"),
    (lambda raw: raw["motion_receipts"][0].update(previous_motion_sha256="0"*64), "chain"),
])
def test_rehashed_corruption_rejected_by_semantic_replay(mutate, match):
    surface, _ = origin()
    raw = fault_material_atlas_to_dict(surface, moved())
    mutate(raw)
    with pytest.raises(ValueError, match=match):
        fault_material_atlas_from_dict(_rehash(raw), surface)


def test_rehashed_valid_rotation_matrix_not_matching_motion_receipt_rejected():
    surface, _ = origin()
    raw = fault_material_atlas_to_dict(surface, moved())
    raw["owner_rotations"][0][1] = Rotation.from_rotvec((.7, .4, .3)).as_matrix().tolist()
    with pytest.raises(ValueError, match="chronological"):
        fault_material_atlas_from_dict(_rehash(raw), surface)


def test_sparse_owner_ids_do_not_allocate_up_to_maximum_owner():
    surface, original = origin()
    owner_map = {0: 7, 1: 10**9}
    sparse = replace(surface, fragments=tuple(replace(fragment,
        parcel=replace(fragment.parcel, plate=owner_map[fragment.parcel.plate]))
        for fragment in surface.fragments))
    declared = tuple(replace(trace, subducting_plate=owner_map[trace.subducting_plate],
        overriding_plate=owner_map[trace.overriding_plate]) for trace in original.reference_traces.traces)
    trace_set = declare_fault_trace_set(sparse, declared, original.reference_traces.quadrature)
    atlas = initialize_fault_material_atlas(sparse, trace_set, original.bending_parameters)
    atlas = advance_fault_material_atlas(sparse, atlas,
        ((7, (.1, 0., 0.)), (10**9, (0., .2, 0.))), .001, provenance="Sparse declared owners")
    assert len(atlas.owner_rotations) == 2
    assert atlas.plate_ids == (7, 10**9)
    assert material_atlas_surface(sparse, atlas).phase == "advected_candidates"
