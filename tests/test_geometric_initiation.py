"""Analytic local-fault checks, independent of any material winner heuristic."""
from dataclasses import asdict, replace
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tectonics.geometric_contacts import contacts_between
from tectonics.geometric_initiation import OrientedFault, evaluate_fault


RADIUS = 1000.
OMEGA = np.array([[0., 0., 0.], [0., -.01, 0.]])


def contact():
    def fragment(name, plate, points):
        polygon = np.array([(x, y, 1.) for x, y in points])
        polygon /= np.linalg.norm(polygon, axis=1)[:, None]
        return SimpleNamespace(fragment_id=name, polygon=polygon,
            parcel=SimpleNamespace(plate=plate, material_id=name))
    a = fragment('a', 0, [(-1., -1.), (0., -1.), (0., 1.), (-1., 1.)])
    b = fragment('b', 1, [(0., -1.), (1., -1.), (1., 1.), (0., 1.)])
    result, = contacts_between(a, b, RADIUS, OMEGA)
    return result


def fault(boundary=None, **changes):
    boundary = boundary or contact()
    values = dict(fault_id='declared-weak-fault', contact_id=boundary.contact_id,
        subducting_plate=0, overriding_plate=1, dip_radians=math.pi/6,
        effective_stress_pa=((160e6, 0., 10e6), (0., 100e6, 0.), (10e6, 0., 100e6)),
        cohesion_pa=2e6, friction=.2, provenance='explicit analytic test loading')
    values.update(changes)
    return OrientedFault(**values)


def assess(**changes):
    boundary = contact()
    return evaluate_fault(fault(boundary, **changes), boundary, OMEGA, RADIUS)


def test_resolved_stress_matches_independent_cross_section_equations():
    result = assess()
    expected_normal = (100+60/4+10*math.sqrt(3)/2)*1e6
    expected_shear = (60*math.sqrt(3)/4+10/2)*1e6
    assert result.normal_stress_pa == pytest.approx(expected_normal, rel=2e-15)
    assert result.shear_stress_pa == pytest.approx(expected_shear, rel=2e-15)
    assert result.strength_pa == pytest.approx(2e6+.2*expected_normal, rel=2e-15)
    assert result.margin_pa == pytest.approx(expected_shear-2e6-.2*expected_normal, rel=5e-15)
    assert result.status == 'forced_underthrust_admissible'
    assert result.normal_velocity_min_km_per_myr == pytest.approx(-10.)
    assert result.normal_velocity_max_km_per_myr == pytest.approx(-10/math.sqrt(2))


def test_signed_radial_shear_distinguishes_mirror_faults_without_plate_id_preference():
    direct = assess()
    opposite = assess(subducting_plate=1, overriding_plate=0)
    assert direct.margin_pa == pytest.approx(4.248711305964282e6)
    assert opposite.margin_pa == pytest.approx(-2.2871870788979623e6)
    assert opposite.status == 'locked'


def test_symmetric_stress_does_not_choose_between_mirror_candidates():
    tensor = np.diag([200e6, 100e6, 100e6])
    left = assess(effective_stress_pa=tensor)
    right = assess(effective_stress_pa=tensor, subducting_plate=1, overriding_plate=0)
    assert left.status == right.status == 'forced_underthrust_admissible'
    for name in ('normal_stress_pa', 'shear_stress_pa', 'strength_pa', 'margin_pa'):
        assert getattr(left, name) == getattr(right, name)


def test_hydrostatic_pressure_has_no_down_dip_drive():
    result = assess(effective_stress_pa=np.eye(3)*100e6)
    assert abs(result.shear_stress_pa) < result.stress_tolerance_pa
    assert result.status == 'locked'


def test_pore_pressure_is_subtracted_once_in_explicit_effective_input():
    total = np.diag([150e6, 100e6, 100e6])
    dry = assess(effective_stress_pa=total)
    wet = assess(effective_stress_pa=total-50e6*np.eye(3))
    assert wet.shear_stress_pa == pytest.approx(dry.shear_stress_pa)
    assert wet.normal_stress_pa == pytest.approx(dry.normal_stress_pa-50e6)
    assert wet.strength_pa == pytest.approx(dry.strength_pa-10e6)
    assert dry.status == 'locked'
    assert wet.status == 'forced_underthrust_admissible'


def test_reverse_shear_cannot_reactivate_down_dip_even_when_magnitude_exceeds_strength():
    result = assess(effective_stress_pa=np.diag([10e6, 100e6, 100e6]), cohesion_pa=0., friction=0.)
    assert result.shear_stress_pa < -30e6
    assert result.status == 'locked'


def test_effective_tension_is_separate_from_frictionless_admission():
    result = assess(effective_stress_pa=np.diag([0., 0., -100e6]), cohesion_pa=0., friction=0.)
    assert result.shear_stress_pa > 0
    assert result.normal_stress_pa < 0
    assert result.status == 'effective_tension'


def test_exact_coulomb_yield_is_locked_with_only_arithmetic_tolerance():
    trial = assess()
    result = assess(cohesion_pa=trial.shear_stress_pa-.2*trial.normal_stress_pa)
    assert abs(result.margin_pa) <= result.stress_tolerance_pa
    assert result.status == 'locked'
    beyond = assess(cohesion_pa=trial.shear_stress_pa-.2*trial.normal_stress_pa-1.)
    assert beyond.status == 'forced_underthrust_admissible'


def test_missing_stress_stays_missing():
    result = assess(effective_stress_pa=None)
    assert result.status == 'missing_stress'
    assert result.normal_stress_pa is result.shear_stress_pa is result.margin_pa is None


@pytest.mark.parametrize('omega', [np.zeros((2, 3)), -OMEGA, [[0, 0, 0], [-.1, 0, 0]]])
def test_stationary_divergent_and_transform_contacts_do_not_admit(omega):
    boundary = contact()
    assert evaluate_fault(fault(boundary), boundary, omega, RADIUS).status == 'nonconvergent'


def test_full_arc_convergence_rejects_negative_midpoint_but_positive_endpoint():
    boundary = contact()
    omega = np.array([[0., 0., 0.], [0., -.005, .01]])
    assert RADIUS*np.dot(np.cross(omega[1], boundary.midpoint), boundary.normal_a_to_b) < 0
    result = evaluate_fault(fault(boundary), boundary, omega, RADIUS)
    assert result.status == 'mixed_convergence'
    assert result.normal_velocity_min_km_per_myr == pytest.approx(-15/math.sqrt(2))
    assert result.normal_velocity_max_km_per_myr == pytest.approx(5/math.sqrt(2))


def test_full_arc_extremum_includes_interior_stationary_point():
    boundary = contact()
    omega = np.array([[0., 0., 0.], [0., -.01, .002]])
    result = evaluate_fault(fault(boundary), boundary, omega, RADIUS)
    assert result.normal_velocity_min_km_per_myr == pytest.approx(-math.sqrt(104))
    assert result.normal_velocity_max_km_per_myr == pytest.approx(-8/math.sqrt(2))


def test_isolated_stagnant_endpoint_preserves_finite_convergent_arc():
    boundary = contact()
    omega = np.array([[0., 0., 0.], [0., -.01, .01]])
    result = evaluate_fault(fault(boundary), boundary, omega, RADIUS)
    assert abs(result.normal_velocity_max_km_per_myr) <= result.velocity_tolerance_km_per_myr
    assert result.normal_velocity_min_km_per_myr == pytest.approx(-math.sqrt(200))
    assert result.status == 'forced_underthrust_admissible'


def test_whole_arc_check_is_independent_of_cached_contact_velocities():
    boundary = replace(contact(), normal_velocity_km_per_myr=1e99,
        normal_area_rate_km2_per_myr=1e99)
    assert evaluate_fault(fault(boundary), boundary, OMEGA, RADIUS).status == 'forced_underthrust_admissible'


def test_objectivity_under_simultaneous_geometry_stress_and_motion_rotation():
    boundary = contact()
    condition = fault(boundary)
    direct = evaluate_fault(condition, boundary, OMEGA, RADIUS)
    q = Rotation.from_rotvec([.4, -.7, 1.2]).as_matrix()
    rotated = replace(boundary, **{name: tuple(q@np.asarray(getattr(boundary, name)))
        for name in ('start', 'end', 'midpoint', 'normal_a_to_b', 'tangent',
                     'integrated_position_unit_km', 'moment_arm_cross_normal_km2')})
    rotated_fault = replace(condition, effective_stress_pa=q@np.asarray(condition.effective_stress_pa)@q.T)
    moved = evaluate_fault(rotated_fault, rotated, OMEGA@q.T, RADIUS)
    assert moved.status == direct.status
    for name in ('normal_stress_pa', 'shear_stress_pa', 'strength_pa', 'margin_pa',
                 'normal_velocity_min_km_per_myr', 'normal_velocity_max_km_per_myr'):
        assert getattr(moved, name) == pytest.approx(getattr(direct, name), rel=3e-14)


def test_plate_relabeling_preserves_physical_assessment():
    boundary = contact()
    changed = replace(boundary, plate_a=7, plate_b=3)
    omega = np.zeros((8, 3)); omega[3] = OMEGA[1]
    result = evaluate_fault(fault(changed, subducting_plate=7, overriding_plate=3), changed, omega, RADIUS)
    assert result.status == assess().status
    assert result.margin_pa == assess().margin_pa


def test_declared_parallel_transport_keeps_local_tractions_constant_along_arc():
    boundary = contact()
    half_length = .05
    short = replace(boundary, start=(0., -math.sin(half_length), math.cos(half_length)),
        end=(0., math.sin(half_length), math.cos(half_length)), length_km=2*half_length*RADIUS)
    condition = fault(short)
    reference = evaluate_fault(condition, short, OMEGA, RADIUS)
    for position in (-.6, -.3, .3, .6):
        # Move a short neighborhood to another point of the original contact.
        # Omega stays fixed: this is the declared along-arc field, not a global
        # change of coordinates or a common rotation of the whole experiment.
        q = Rotation.from_rotvec(position*np.asarray(short.normal_a_to_b)).as_matrix()
        shifted = replace(short, **{name: tuple(q@np.asarray(getattr(short, name)))
            for name in ('start', 'end', 'midpoint')})
        transported = replace(condition,
            effective_stress_pa=q@np.asarray(condition.effective_stress_pa)@q.T)
        result = evaluate_fault(transported, shifted, OMEGA, RADIUS)
        for name in ('normal_stress_pa', 'shear_stress_pa', 'strength_pa', 'margin_pa'):
            assert getattr(result, name) == pytest.approx(getattr(reference, name), rel=2e-14)


def test_effective_tensor_is_an_immutable_canonical_copy():
    tensor = np.diag([200e6, 100e6, 100e6])
    condition = fault(effective_stress_pa=tensor)
    tensor[:] = 0
    assert condition.effective_stress_pa[0][0] == 200e6
    assert isinstance(condition.effective_stress_pa, tuple)
    assert asdict(condition)['provenance']


def test_numpy_scalar_inputs_are_canonical_and_json_serializable():
    condition = fault(subducting_plate=np.int64(0), overriding_plate=np.int32(1),
        dip_radians=np.float64(math.pi/6), cohesion_pa=np.float32(2e6), friction=np.float64(.2))
    encoded = json.dumps(asdict(condition), allow_nan=False)
    assert json.loads(encoded)['subducting_plate'] == 0
    assert type(condition.subducting_plate) is int
    assert type(condition.cohesion_pa) is float


@pytest.mark.parametrize('changes', [
    {'fault_id': ''}, {'contact_id': ' '}, {'provenance': ''},
    {'subducting_plate': True}, {'subducting_plate': -1}, {'overriding_plate': 0},
    {'overriding_plate': 1.5}, {'dip_radians': False}, {'dip_radians': 0},
    {'dip_radians': math.pi/2}, {'dip_radians': math.nan}, {'cohesion_pa': -1},
    {'cohesion_pa': True}, {'friction': -1}, {'friction': math.inf},
    {'effective_stress_pa': np.ones((2, 2))},
    {'effective_stress_pa': [[1, 2, 0], [0, 1, 0], [0, 0, 1]]},
    {'effective_stress_pa': [[True, 0, 0], [0, 1, 0], [0, 0, 1]]},
    {'effective_stress_pa': np.diag([math.inf]*3)},
    {'effective_stress_pa': np.eye(3, dtype=complex)},
])
def test_invalid_explicit_fault_fields_are_rejected(changes):
    with pytest.raises(ValueError):
        fault(**changes)


@pytest.mark.parametrize('changes', [
    {'contact_id': 'remote-contact'}, {'plate_b': 9},
    {'midpoint': (1., 0., 0.)}, {'normal_a_to_b': (0., 0., 1.)},
    {'start': (0., 0., 0.)}, {'end': (0., math.nan, 1.)}, {'length_km': 3.},
])
def test_invalid_contact_geometry_and_ownership_are_rejected(changes):
    boundary = contact()
    with pytest.raises(ValueError):
        evaluate_fault(fault(boundary), replace(boundary, **changes), OMEGA, RADIUS)


@pytest.mark.parametrize('omega,radius', [([[0., 0., 0.]], RADIUS),
    (np.zeros((2, 2)), RADIUS), ([[0, 0, 0], [0, math.inf, 0]], RADIUS),
    (OMEGA, 0), (OMEGA, True), (OMEGA, math.nan)])
def test_invalid_motion_and_radius_are_rejected(omega, radius):
    boundary = contact()
    with pytest.raises(ValueError):
        evaluate_fault(fault(boundary), boundary, omega, radius)
