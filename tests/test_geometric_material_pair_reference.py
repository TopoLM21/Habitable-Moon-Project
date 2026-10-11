"""Independent objective derivatives and exact analytic material-pair fields."""
from dataclasses import FrozenInstanceError, asdict, replace
from functools import lru_cache
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tests.test_geometric_bending_contact_dynamics import prepared
from tests.test_geometric_bending_contact_dynamics import BOUNDS
from tests.test_geometric_distributed_contact_invariants import PARAMETERS, PROVENANCE, world
from tectonics.geometric_basal_loading import assemble_geometric_basal_load
from tectonics.geometric_bending_contact_dynamics import initialize_bending_contact_dynamics
from tectonics.geometric_bending_reference import current_bending_reference
from tectonics.geometric_boundary import initialize_boundary
from tectonics.geometric_fault_traces import _arc_frame, declare_fault_trace_set
from tectonics.geometric_material_pair_reference import (
    _material_pair_network, canonical_material_pair_poses,
    current_material_pair_reference, evaluate_material_pair_field,
    material_pair_exact_anchor_fields, material_pair_increment_fields,
    material_pair_point_reactions,
)


@lru_cache(maxsize=1)
def fixture():
    origin, _, _, seed = prepared(mixed=True)
    poses = tuple((owner, tuple(map(tuple, Rotation.from_rotvec(vector).as_matrix())))
        for owner, vector in zip(seed.plate_ids, ((.003, -.001, .002), (-.002, .004, .001))))
    radial = tuple((trace.trace_id, -2.-index, 3.+.5*index)
                   for index, trace in enumerate(seed.fault_traces.traces))
    return origin, seed, poses, radial, _material_pair_network(origin, seed, poses, radial)


def _pose_value(point, trace, poses, radial, radius):
    sub, over = (np.asarray(poses[owner]) for owner in (trace.subducting_plate, trace.overriding_plate))
    sub_position = (radius+radial[0])*(sub@point.position)
    over_position = (radius+radial[1])*(over@point.position)
    return np.asarray([float((over@axis)@(over_position-sub_position))
                       for axis in (point.normal, point.down_dip)])


def test_objective_operator_matches_independent_finite_pose_derivatives():
    origin, seed, poses, radial, network = fixture()
    original, _, _, _, _, operators = network
    matrices = {owner: np.asarray(matrix) for owner, matrix in poses}
    trace_indices = {trace.trace_id: index for index, trace in enumerate(seed.fault_traces.traces)}
    radius, count = origin.radius_km*1000., 3*len(seed.plate_ids)
    operator = np.asarray(operators)
    epsilon = 2e-6
    for index, (point, trace) in enumerate(zip(original.points, original.materials)):
        pair = radial[trace_indices[point.trace_id]][1:]
        for owner_index, owner in enumerate(seed.plate_ids):
            for component in range(3):
                axis = np.eye(3)[component]
                plus, minus = dict(matrices), dict(matrices)
                plus[owner] = Rotation.from_rotvec(epsilon*axis).as_matrix()@matrices[owner]
                minus[owner] = Rotation.from_rotvec(-epsilon*axis).as_matrix()@matrices[owner]
                numerical = (_pose_value(point, trace, plus, pair, radius)
                    -_pose_value(point, trace, minus, pair, radius))/(2.*epsilon)
                np.testing.assert_allclose(operator[index, :, 3*owner_index+component],
                                           numerical, rtol=3e-7, atol=1e-5)
        for side in range(2):
            plus, minus = np.asarray(pair).copy(), np.asarray(pair).copy()
            plus[side] += epsilon*radius
            minus[side] -= epsilon*radius
            numerical = (_pose_value(point, trace, matrices, plus, radius)
                -_pose_value(point, trace, matrices, minus, radius))/(2.*epsilon)
            np.testing.assert_allclose(operator[index, :, count+2*trace_indices[point.trace_id]+side],
                                       numerical, rtol=3e-7, atol=1e-5)


def test_each_angular_operator_retains_an_exact_common_rotation_nullspace():
    _, seed, _, _, network = fixture()
    operators = np.asarray(network[-1])
    for point in operators:
        for row in point:
            for component in range(3):
                assert math.fsum(row[3*owner+component] for owner in range(len(seed.plate_ids))) == 0.


def test_reactions_require_the_carrier_couple_to_reproduce_virtual_work():
    origin, seed, _, _, network = fixture()
    original, _, _, ports, _, operators = network
    count, radius = 3*len(seed.plate_ids), origin.radius_km*1000.
    coordinates = np.linspace(-2e-5, 3e-5, len(original.stiffness))
    positions = {owner: index for index, owner in enumerate(seed.plate_ids)}
    trace_indices = {trace.trace_id: index for index, trace in enumerate(seed.fault_traces.traces)}
    for port, area, operator in zip(ports, original.areas, operators):
        traction = (-2e5, -7e4)
        reactions = material_pair_point_reactions(port, area, traction)
        np.testing.assert_array_equal(np.asarray(reactions["torque_on_sub_nm"])
                                      +np.asarray(reactions["torque_on_over_nm"]), np.zeros(3))
        force_moment, couple, body_moment = (np.asarray(reactions[name]) for name in
            ("force_torque_on_over_nm", "carrier_couple_on_over_nm", "torque_on_over_nm"))
        rounding = 64*np.finfo(float).eps*(np.linalg.norm(force_moment)
            +np.linalg.norm(couple)+np.linalg.norm(body_moment))
        assert np.linalg.norm(force_moment+couple-body_moment) <= rounding
        assert np.linalg.norm(reactions["carrier_couple_on_over_nm"]) > 0.
        physical_work = math.fsum((
            np.dot(reactions["torque_on_sub_nm"], coordinates[3*positions[port.subducting_plate]:3*positions[port.subducting_plate]+3]),
            np.dot(reactions["torque_on_over_nm"], coordinates[3*positions[port.overriding_plate]:3*positions[port.overriding_plate]+3]),
            *(force*radius*coordinates[count+2*trace_indices[port.trace_id]+side]
              for side, force in enumerate(reactions["radial_force_n"]))))
        generalized_work = -area*np.asarray(traction)@(np.asarray(operator)@coordinates)
        assert physical_work == pytest.approx(generalized_work, rel=4e-15)


def test_port_view_preserves_original_material_measure_stiffness_and_history():
    origin, seed, _, _, network = fixture()
    before = asdict(origin), asdict(seed)
    original, _, points, ports, _, _ = network
    independent = current_bending_reference(origin, seed)
    assert original.areas == independent.areas
    assert original.stiffness == independent.stiffness
    assert original.materials == independent.materials
    assert tuple(point.point_id for point in points) == tuple(point.point_id for point in independent.points)
    assert tuple(point.area_m2 for point in points) == tuple(point.area_m2 for point in independent.points)
    assert tuple(point.line_weight_m for point in points) == tuple(point.line_weight_m for point in independent.points)
    assert any(port.sub_radial != port.over_radial for port in ports)
    assert (asdict(origin), asdict(seed)) == before
    with pytest.raises(FrozenInstanceError):
        ports[0].sub_radial = (0., 0., 0.)


def test_five_harmonic_increment_fields_match_each_objective_operator():
    origin, seed, poses, before, network = fixture()
    original, _, _, _, _, operators = network
    increments = ((3e-6, -2e-6, 7e-6), (-1e-6, 4e-6, -6e-6))
    after = tuple((identity, sub+.02, over-.03) for identity, sub, over in before)
    fields = dict((identity, (gap, jump)) for identity, gap, jump
        in material_pair_increment_fields(seed, poses, before, increments, after))
    trace_map = {trace.trace_id: trace for trace in seed.fault_traces.traces}
    radius = origin.radius_km*1000.
    coordinates = np.r_[np.asarray(increments).reshape(-1),
        (np.asarray([row[1:] for row in after])-np.asarray([row[1:] for row in before])).reshape(-1)/radius]
    for point, operator in zip(original.points, operators):
        start, _, tangent, _ = _arc_frame(trace_map[point.trace_id].start, trace_map[point.trace_id].end)
        angle = math.atan2(float(np.asarray(point.position)@tangent), float(np.asarray(point.position)@start))
        analytic = [evaluate_material_pair_field(field, angle) for field in fields[point.trace_id]]
        np.testing.assert_allclose(analytic, np.asarray(operator)@coordinates, rtol=3e-14, atol=3e-14)
    assert any(abs(field[3])+abs(field[4]) > 1e-8 for pair in fields.values() for field in pair)


def test_finite_pose_anchor_change_matches_independent_geometry_between_quadrature_nodes():
    origin, seed, poses, radial, _ = fixture()
    matrices = dict(poses)
    fields = material_pair_exact_anchor_fields(seed, poses, radial)
    radius = origin.radius_km*1000.
    for trace, pair, (_, gap, jump) in zip(seed.fault_traces.traces, radial, fields):
        start, horizontal, tangent, extent = _arc_frame(trace.start, trace.end)
        sine, cosine = math.sin(trace.dip_radians), math.cos(trace.dip_radians)
        for angle in np.linspace(0., extent, 19):
            point = start*math.cos(angle)+tangent*math.sin(angle)
            sub, over = np.asarray(matrices[trace.subducting_plate]), np.asarray(matrices[trace.overriding_plate])
            # Independent direct global-vector geometry, including frame rotation.
            difference = (radius+pair[2])*(over@point)-(radius+pair[1])*(sub@point)
            exact = [float((over@axis)@difference) for axis in
                (sine*horizontal+cosine*point, cosine*horizontal-sine*point)]
            baseline = dict((identity, (ws, wo)) for identity, ws, wo in seed.transverse_displacement_m)[trace.trace_id]
            exact[0] -= cosine*(baseline[1]-baseline[0])
            exact[1] += sine*(baseline[1]-baseline[0])
            np.testing.assert_allclose([evaluate_material_pair_field(gap, angle),
                evaluate_material_pair_field(jump, angle)], exact, rtol=5e-13, atol=3e-11)


def test_exact_pose_change_is_zero_at_original_seed_and_does_not_rebase_prepared_anchors():
    _, seed, _, _, _ = fixture()
    identity = tuple((owner, tuple(map(tuple, np.eye(3)))) for owner in seed.plate_ids)
    changes = material_pair_exact_anchor_fields(seed, identity, seed.transverse_displacement_m)
    assert all(gap == jump == (0., 0., 0., 0., 0.) for _, gap, jump in changes)
    assert any(local.gap_free_m != 0. for _, local in seed.states)


def test_global_frame_covariance_preserves_each_port_increment_and_radial_beam_energy():
    origin, seed, poses, radial, first = fixture()
    rotation = Rotation.from_rotvec((.2, -.3, .4)).as_matrix()
    moved = tuple((owner, tuple(map(tuple, rotation@np.asarray(matrix)))) for owner, matrix in poses)
    second = _material_pair_network(origin, seed, moved, radial)
    coordinates = np.linspace(-1e-6, 2e-6, len(first[0].stiffness))
    transformed = coordinates.copy()
    for index in range(len(seed.plate_ids)):
        transformed[3*index:3*index+3] = rotation@coordinates[3*index:3*index+3]
    np.testing.assert_allclose(np.asarray(first[-1])@coordinates,
                              np.asarray(second[-1])@transformed, rtol=7e-14, atol=2e-14)
    assert first[0].areas == second[0].areas and first[0].stiffness == second[0].stiffness


@pytest.mark.parametrize("mutate", [
    lambda poses: poses[:-1],
    lambda poses: poses+poses[-1:],
    lambda poses: tuple(reversed(poses)),
    lambda poses: ((True, poses[0][1]), poses[1]),
    lambda poses: ((8, poses[0][1]), poses[1]),
    lambda poses: ((poses[0][0], np.diag((1., 1., -1.))), poses[1]),
    lambda poses: ((poses[0][0], np.diag((1., 1.01, 1.))), poses[1]),
    lambda poses: ((poses[0][0], [[True, 0., 0.], [0., 1., 0.], [0., 0., 1.]]), poses[1]),
    lambda poses: ((poses[0][0], None), poses[1]),
])
def test_pose_records_require_exact_actual_owner_set_and_real_proper_rotations(mutate):
    _, seed, poses, _, _ = fixture()
    with pytest.raises(ValueError):
        canonical_material_pair_poses(seed, mutate(poses))


@pytest.mark.parametrize("increments", [((0., 0., 0.),), ((True, 0., 0.), (0., 0., 0.)),
    ((float("inf"), 0., 0.), (0., 0., 0.)), (("0", "0", "0"), ("0", "0", "0"))])
def test_increment_field_rejects_aliases_and_nonfinite_compact_vectors(increments):
    _, seed, poses, radial, _ = fixture()
    with pytest.raises(ValueError):
        material_pair_increment_fields(seed, poses, radial, increments, radial)


def test_exact_anchor_fields_reject_overflow_from_finite_radial_coordinates():
    _, seed, poses, radial, _ = fixture()
    extreme = tuple((identity, 1.7e308, -1.7e308) for identity, _, _ in radial)
    with pytest.raises(ValueError, match="overflow"):
        material_pair_exact_anchor_fields(seed, poses, extreme)


@pytest.mark.parametrize("coefficients,angle", [((0., 0., 0.), 0.),
    ((True, 0., 0., 0., 0.), 0.), ((0., 0., 0., 0., float("nan")), 0.),
    ((0., 0., 0., 0., 0.), True)])
def test_field_evaluator_rejects_invalid_explicit_numeric_basis(coefficients, angle):
    with pytest.raises(ValueError):
        evaluate_material_pair_field(coefficients, angle)


def test_public_reference_uses_fresh_support_proof_without_advancing_history_or_clock():
    from tectonics.geometric_buried_bank_support import initial_buried_bank_registry
    origin, seed, _, _, _ = fixture()
    inventory = initialize_boundary(origin)
    registry = initial_buried_bank_registry(origin, inventory)
    poses = tuple((owner, tuple(map(tuple, np.eye(3)))) for owner in seed.plate_ids)
    before = asdict(origin), asdict(seed), asdict(inventory), asdict(registry)
    view = current_material_pair_reference(origin, seed, origin, poses,
        seed.transverse_displacement_m, inventory=inventory, buried_registry=registry)
    assert view.physical_dt_s == 0.
    assert view.surface is origin
    assert view.reference_surface_sha256 == view.current_surface_sha256
    assert view.support_receipt is not None
    assert view.areas == current_bending_reference(origin, seed).areas
    assert (asdict(origin), asdict(seed), asdict(inventory), asdict(registry)) == before
    again = current_material_pair_reference(origin, seed, origin, dict(poses),
        seed.transverse_displacement_m, inventory=inventory, buried_registry=registry,
        support_receipt=view.support_receipt)
    assert again == view
    with pytest.raises(ValueError, match="receipt"):
        current_material_pair_reference(origin, seed, origin, poses,
            seed.transverse_displacement_m, inventory=inventory, buried_registry=registry,
            support_receipt=object())


def test_sparse_actual_owners_keep_compact_port_blocks_and_authentic_bank_support():
    from tectonics.geometric_buried_bank_support import initial_buried_bank_registry
    _, old_seed, _, _, _ = fixture()
    mesh, original, lid = world()
    owners = {0: 7, 1: 10**12}
    origin = replace(original, fragments=tuple(replace(fragment,
        parcel=replace(fragment.parcel, plate=owners[fragment.parcel.plate]))
        for fragment in original.fragments))
    declared = tuple(replace(trace, subducting_plate=owners[trace.subducting_plate],
        overriding_plate=owners[trace.overriding_plate]) for trace in old_seed.fault_traces.traces)
    trace_set = declare_fault_trace_set(origin, declared, old_seed.fault_traces.quadrature)
    load = assemble_geometric_basal_load(mesh, origin, PARAMETERS, lid, PROVENANCE)
    seed = initialize_bending_contact_dynamics(origin, trace_set, load,
        old_seed.bending_parameters, **BOUNDS)
    inventory = initialize_boundary(origin)
    registry = initial_buried_bank_registry(origin, inventory)
    poses = {owner: np.eye(3) for owner in owners.values()}
    view = current_material_pair_reference(origin, seed, origin, poses,
        seed.transverse_displacement_m, inventory=inventory, buried_registry=registry)
    assert view.plate_ids == (7, 10**12)
    assert np.asarray(view.operators).shape == (8, 2, 14)
    assert len(view.stiffness) == 14
    assert {port.subducting_plate for port in view.pair_ports} == {7}
    assert {port.overriding_plate for port in view.pair_ports} == {10**12}
