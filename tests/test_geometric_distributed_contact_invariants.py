"""Independent continuum-arc and material-cut checks of frozen contact dynamics."""
from dataclasses import asdict, replace
from itertools import product
import json
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import tectonics.fractional_dynamics as fractional_dynamics
from tectonics.basal_coupling import PrescribedBasalParameters
from tectonics.fractional_surface import SurfaceParcel, split_parcel
from tectonics.genesis_contact_law import ContactLawParameters
from tectonics.geometric_basal_loading import assemble_geometric_basal_load
from tectonics.geometric_boundary import (
    consume_geometric_transaction, initialize_boundary, polarity_evidence,
)
from tectonics.geometric_boundary_io import load_boundary_checkpoint, save_boundary_checkpoint
from tectonics.geometric_contact_mechanics import ElasticColumnParameters
from tectonics.geometric_contact_dynamics import UnresolvedContactDynamics
from tectonics.geometric_fault_traces import (
    PhysicalFaultTrace, TraceQuadrature, declare_fault_trace_set, fault_trace_points,
)
from tectonics.geometric_distributed_contact_dynamics import (
    advance_distributed_contact_dynamics, distributed_contact_dynamics_from_dict,
    distributed_contact_dynamics_to_dict, initialize_distributed_contact_dynamics,
)
from tectonics.geometric_contacts import extract_contacts
from tectonics.geometric_polarity import PolarityEvidence
from tectonics.geometric_surface import GeometricFragment, GeometricSurfaceState, totals
from tectonics.geometric_transport import advance_geometric_surface
from tectonics.mesh import build_icosphere
from tectonics.spherical_polygons import normalize_polygon, polygon_area


DT = 1e10
PARAMETERS = PrescribedBasalParameters(17, 80000., 1e14, 2.)
PROVENANCE = "Explicit fixed mantle source and inherited nonuniform thermal lid"


def world():
    radius = 100.
    fragments = []
    for index, signs in enumerate(product((-1., 1.), repeat=3)):
        polygon = normalize_polygon(np.diag(signs))
        area = polygon_area(polygon)*radius**2
        owner = int(signs[2] < 0.)
        parcel = SurfaceParcel(index, owner, f"material:{index}", area, 2.*area, 3.*area,
            area*(2e11 if owner == 0 else 1e11), 10. if owner == 0 else 5.,
            (("damage", .2),))
        fragments.append(GeometricFragment(f"octant:{index}", tuple(map(tuple, polygon)), parcel))
    surface = GeometricSurfaceState(0., radius, tuple(fragments))
    lid = {item.fragment_id: .2+1.3*index for index, item in enumerate(surface.fragments)}
    return build_icosphere(0), surface, lid


def traces(*, sub=0, over=1, offset=0., cohesion=1e7, height_variation=True, gap=-1.):
    law = ContactLawParameters(normal_stiffness_pa_m=1e6, tangential_stiffness_pa_m=1e6,
        tensile_strength_pa=2e6, fracture_energy_j_m2=2e7, cohesion_pa=cohesion,
        friction_dry=.4, friction_wet=.2, wet_cohesion_fraction=.5, viscosity_pa_s_m=1e8)
    columns = ElasticColumnParameters(1000., 1e9, 4e8, 1000., 1e9, 4e8)
    result = []
    vertices = ([(1., 0., 0.), (0., -1., 0.), (-1., 0., 0.), (0., 1., 0.)]
                if offset == 0. else [(math.cos(offset-index*math.pi/2.),
                                      math.sin(offset-index*math.pi/2.), 0.) for index in range(4)])
    for index in range(4):
        # Clockwise support gives h=-z, from northern sub to southern over.
        start, end = vertices[index], vertices[(index+1) % 4]
        result.append(PhysicalFaultTrace(f"physical:{index}", start, end, sub, over,
            math.pi/4., columns, law, 1000.*(1.+.2*index if height_variation else 1.),
            1., 0., gap, 0., "Declared physical weak-plane support, fixed independent of material cuts"))
    return tuple(result)


def prepare(mesh, surface, lid, *, declared=None, panels=1, order=6, parameters=PARAMETERS, history=()):
    load = assemble_geometric_basal_load(mesh, surface, parameters, lid, PROVENANCE)
    trace_set = declare_fault_trace_set(surface, traces() if declared is None else declared,
                                      TraceQuadrature(panels, order))
    initial = initialize_distributed_contact_dynamics(surface, trace_set, load,
        initial_loading_dt_s=1e6, max_rotation_radians=.01, history=history)
    return load, trace_set, initial


def exact_arc_moments(trace, radius_m):
    """Closed form, derived independently of quadrature/node implementation."""
    start, end = np.asarray(trace.start), np.asarray(trace.end)
    theta = math.atan2(np.linalg.norm(np.cross(start, end)), float(start@end))
    normal = np.cross(start, end)
    normal /= np.linalg.norm(normal)
    midpoint = (start+end)/np.linalg.norm(start+end)
    tangent = np.cross(normal, midpoint)
    beta, gamma = np.cross(midpoint, normal), np.cross(tangent, normal)
    second = radius_m**3*(.5*(theta+math.sin(theta))*np.outer(beta, beta)
                         +.5*(theta-math.sin(theta))*np.outer(gamma, gamma))
    first = 2.*radius_m**2*math.sin(.5*theta)*beta
    return theta, first, second


def exact_locked_solution(surface, declared, load):
    ids = load.plate_ids
    matrix = np.asarray(load.drag_nm_s)/DT
    initial_force = np.zeros(3*len(ids))
    for trace in declared:
        _, first, second = exact_arc_moments(trace, surface.radius_km*1000.)
        sine, cosine = math.sin(trace.dip_radians), math.cos(trace.dip_radians)
        normal = 1./(1./trace.columns.normal_stiffness_pa_m+1./trace.law.normal_stiffness_pa_m)
        shear = 1./(1./trace.columns.shear_stiffness_pa_m+1./trace.law.tangential_stiffness_pa_m)
        width = trace.contact_height_m/sine
        stiffness = width*(normal*sine*sine+shear*cosine*cosine)*second
        a, b = ids.index(trace.subducting_plate), ids.index(trace.overriding_plate)
        for row, row_sign in ((a, -1.), (b, 1.)):
            initial_force[3*row:3*row+3] += row_sign*width*(
                sine*normal*trace.initial_gap_m+cosine*shear*trace.initial_jump_m)*first
            for column, column_sign in ((a, -1.), (b, 1.)):
                matrix[3*row:3*row+3, 3*column:3*column+3] += row_sign*column_sign*stiffness
    return np.linalg.solve(matrix, np.asarray(load.torque_nm)-initial_force).reshape((-1, 3))


def test_pointwise_second_moment_resolves_both_arc_modes_and_converges_to_exact_tensor():
    _, surface, _ = world()
    declared = traces()
    theta, _, exact = exact_arc_moments(declared[0], surface.radius_km*1000.)
    errors = []
    for order in (2, 4, 8):
        trace_set = declare_fault_trace_set(surface, declared, TraceQuadrature(1, order))
        points = [item for item in fault_trace_points(surface, trace_set)
                  if item.trace_id == declared[0].trace_id]
        tensor = sum((point.line_weight_m*np.outer(
            surface.radius_km*1000.*np.cross(point.position, point.horizontal),
            surface.radius_km*1000.*np.cross(point.position, point.horizontal)) for point in points),
            start=np.zeros((3, 3)))
        errors.append(np.linalg.norm(tensor-exact)/np.linalg.norm(exact))
        assert np.trace(tensor) == pytest.approx(surface.radius_km**3*1e9*theta, rel=1e-14)
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 2e-14
    assert np.linalg.eigvalsh(exact)[1] > 0.
    _, first, _ = exact_arc_moments(declared[0], surface.radius_km*1000.)
    old_mean = np.outer(first, first)/(surface.radius_km*1000.*theta)
    assert np.trace(old_mean)/np.trace(exact) == pytest.approx(8./math.pi**2, rel=1e-14)


def test_locked_global_solution_converges_with_gauss_order_and_panels_to_continuum_oracle():
    mesh, surface, lid = world()
    declared = traces()
    errors = []
    for panels, order in ((1, 2), (1, 4), (1, 8), (4, 2)):
        load, _, initial = prepare(mesh, surface, lid, declared=declared, panels=panels, order=order)
        step = advance_distributed_contact_dynamics(surface, initial, load, DT)
        exact = exact_locked_solution(surface, declared, load)
        errors.append(np.linalg.norm(np.asarray(step.state.rotation_radians)-exact)/np.linalg.norm(exact))
        assert all(item.step.slip_increment_m == 0. for item in step.points)
        assert step.minimum_certified_eigenvalue > 0.
        assert step.incremental_energy_balance_relative_error < 2e-9
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 2e-10
    assert errors[3] < errors[0]


def split_material_cuts(surface, lid):
    fragments, new_lid = [], {}
    for original in surface.fragments:
        polygon = np.asarray(original.polygon)
        pole = next(vertex for vertex in polygon if abs(vertex[2]) > .5)
        equator = [vertex for vertex in polygon if abs(vertex[2]) < .5]
        middle = (equator[0]+equator[1])/np.linalg.norm(equator[0]+equator[1])
        for index, support in enumerate(((pole, equator[0], middle), (pole, middle, equator[1]))):
            child_polygon = normalize_polygon(support)
            fraction = polygon_area(child_polygon)*surface.radius_km**2/original.parcel.area_km2
            child = GeometricFragment(f"cut:{original.fragment_id}:{index}", tuple(map(tuple, child_polygon)),
                split_parcel(original.parcel, fraction), original.fragment_id)
            fragments.append(child)
            new_lid[child.fragment_id] = lid[original.fragment_id]
    return replace(surface, fragments=tuple(reversed(fragments))), new_lid


def test_material_boundary_cuts_preserve_physical_points_and_evolving_history():
    mesh, surface, lid = world()
    cut_surface, cut_lid = split_material_cuts(surface, lid)
    declared = traces(cohesion=10000., height_variation=False)
    parameters = replace(PARAMETERS, convective_traction_pa=100000.)
    load, trace_set, initial = prepare(mesh, surface, lid, declared=declared, parameters=parameters)
    cut_load, cut_trace_set, cut_initial = prepare(mesh, cut_surface, cut_lid, declared=declared,
                                                parameters=parameters)
    assert trace_set.trace_sha256 == cut_trace_set.trace_sha256
    assert fault_trace_points(surface, trace_set) == fault_trace_points(cut_surface, cut_trace_set)
    # Symmetry makes several entries theoretically zero, with reduction noise
    # at the scale of the diagonal; compare the physical operator norm.
    drag = np.asarray(load.drag_nm_s)
    assert np.linalg.norm(np.asarray(cut_load.drag_nm_s)-drag) < 2e-13*np.linalg.norm(drag)
    np.testing.assert_allclose(cut_load.torque_nm, load.torque_nm, rtol=2e-12,
                               atol=np.linalg.norm(load.torque_nm)*1e-13)
    assert totals(cut_surface.fragments) == pytest.approx(totals(surface.fragments), rel=2e-14)
    assert cut_surface.known_material_ids == surface.known_material_ids
    for _ in range(2):
        step = advance_distributed_contact_dynamics(surface, initial, load, DT)
        cut_step = advance_distributed_contact_dynamics(cut_surface, cut_initial, cut_load, DT)
        np.testing.assert_allclose(cut_step.angular_velocity_rad_s, step.angular_velocity_rad_s,
                                   rtol=2e-10, atol=1e-27)
        assert [point.point_id for point in cut_step.points] == [point.point_id for point in step.points]
        for original, cut in zip(step.points, cut_step.points):
            assert cut.step.traction_pa == pytest.approx(original.step.traction_pa, rel=2e-10)
            assert cut.step.state.plastic_slip_m == pytest.approx(original.step.state.plastic_slip_m, rel=2e-10)
        assert cut_step.source_work_j == pytest.approx(step.source_work_j, rel=2e-10)
        initial, cut_initial = step.state, cut_step.state
    assert any(abs(local.plastic_slip_m) > 1e-5 for _, local in initial.states)


def test_sparse_owner_permutation_preserves_distributed_solution_and_point_history():
    mesh, surface, lid = world()
    declared = traces()
    load, _, initial = prepare(mesh, surface, lid, declared=declared)
    step = advance_distributed_contact_dynamics(surface, initial, load, DT)
    labels = {0: 10**12, 1: 7}
    changed_surface = replace(surface, fragments=tuple(replace(item,
        parcel=replace(item.parcel, plate=labels[item.parcel.plate])) for item in reversed(surface.fragments)))
    changed_traces = tuple(replace(item, subducting_plate=labels[item.subducting_plate],
                                  overriding_plate=labels[item.overriding_plate]) for item in declared)
    changed_load, _, changed_initial = prepare(mesh, changed_surface, lid, declared=changed_traces)
    changed = advance_distributed_contact_dynamics(changed_surface, changed_initial, changed_load, DT)
    assert changed_load.plate_ids == (7, 10**12)
    assert np.asarray(changed_load.drag_nm_s).shape == (6, 6)
    for index, owner in enumerate(load.plate_ids):
        target = changed_load.plate_ids.index(labels[owner])
        np.testing.assert_allclose(changed.angular_velocity_rad_s[target], step.angular_velocity_rad_s[index],
                                   rtol=2e-10, atol=1e-27)
    assert [item.point_id for item in changed.points] == [item.point_id for item in step.points]
    for original, permuted in zip(step.points, changed.points):
        assert permuted.step.traction_pa == pytest.approx(original.step.traction_pa, rel=2e-10)
    assert changed.source_work_j == pytest.approx(step.source_work_j, rel=2e-10)


def test_passive_rotation_of_source_geometry_and_physical_traces_is_covariant(monkeypatch):
    mesh, surface, lid = world()
    declared = traces()
    load, _, initial = prepare(mesh, surface, lid, declared=declared)
    step = advance_distributed_contact_dynamics(surface, initial, load, DT)
    rotation = Rotation.from_rotvec([.31, -.17, .24]).as_matrix()
    rotated_mesh = replace(mesh, vertices=mesh.vertices@rotation.T, centroids=mesh.centroids@rotation.T)
    rotated_surface = replace(surface, fragments=tuple(replace(item,
        polygon=tuple(map(tuple, np.asarray(item.polygon)@rotation.T))) for item in surface.fragments))
    rotated_traces = tuple(replace(item, start=tuple(rotation@item.start), end=tuple(rotation@item.end))
                           for item in declared)
    source = fractional_dynamics.prescribed_vertex_traction

    def rotate_physical_source(current_mesh, **kwargs):
        assert current_mesh is rotated_mesh
        return source(mesh, **kwargs)@rotation.T

    monkeypatch.setattr(fractional_dynamics, "prescribed_vertex_traction", rotate_physical_source)
    rotated_load, _, rotated_initial = prepare(rotated_mesh, rotated_surface, lid, declared=rotated_traces)
    changed = advance_distributed_contact_dynamics(rotated_surface, rotated_initial, rotated_load, DT)
    np.testing.assert_allclose(changed.angular_velocity_rad_s,
        np.asarray(step.angular_velocity_rad_s)@rotation.T, rtol=3e-10, atol=1e-27)
    assert [item.point_id for item in changed.points] == [item.point_id for item in step.points]
    for original, rotated in zip(step.points, changed.points):
        assert rotated.step.traction_pa == pytest.approx(original.step.traction_pa, rel=3e-10)
    assert changed.source_work_j == pytest.approx(step.source_work_j, rel=3e-10)


def test_one_physical_arc_carries_mixed_locked_and_sliding_histories():
    mesh, surface, lid = world()
    declared = traces(cohesion=10000., height_variation=False)
    parameters = replace(PARAMETERS, convective_traction_pa=100000.)
    load, trace_set, initial = prepare(mesh, surface, lid, declared=declared, parameters=parameters, order=8)
    step = advance_distributed_contact_dynamics(surface, initial, load, DT)
    relative = np.asarray(step.state.rotation_radians)[1]-np.asarray(step.state.rotation_radians)[0]
    by_point = {item.point_id: item for item in step.points}
    by_trace = {trace.trace_id: [] for trace in declared}
    assert all(local.plastic_slip_m == 0. for _, local in initial.states)
    for point in fault_trace_points(surface, trace_set):
        trace = next(item for item in declared if item.trace_id == point.trace_id)
        displacement = surface.radius_km*1000.*np.cross(point.position, point.horizontal)@relative
        gap = trace.initial_gap_m+math.sin(trace.dip_radians)*displacement
        jump = trace.initial_jump_m+math.cos(trace.dip_radians)*displacement
        normal = 1./(1./trace.columns.normal_stiffness_pa_m+1./trace.law.normal_stiffness_pa_m)
        shear = 1./(1./trace.columns.shear_stiffness_pa_m+1./trace.law.tangential_stiffness_pa_m)
        strength = trace.law.cohesion_pa*trace.law.wet_cohesion_fraction+trace.law.friction_wet*(-normal*gap)
        # Independent series-column Coulomb/viscous formula, holding old slip=0.
        expected = math.copysign(max(abs(shear*jump)-strength, 0.)/(
            shear+trace.law.viscosity_pa_s_m/DT), jump)
        result = by_point[point.point_id]
        assert result.step.slip_increment_m == pytest.approx(expected, rel=2e-9, abs=2e-12)
        assert result.step.normal_pressure_pa == pytest.approx(-normal*gap, rel=2e-10)
        by_trace[point.trace_id].append(result.step.slip_increment_m)
    assert any(any(value == 0. for value in values) and any(abs(value) > 1e-5 for value in values)
               for values in by_trace.values())
    assert step.friction_work_j > 0.
    assert step.viscous_work_j > 0.
    assert step.incremental_energy_balance_relative_error < 2e-9


def test_opening_between_gauss_nodes_rejects_even_when_all_sampled_gaps_are_compressed():
    mesh, surface, lid = world()
    load = assemble_geometric_basal_load(mesh, surface, PARAMETERS, lid, PROVENANCE)
    declared = traces(height_variation=False)
    q = exact_locked_solution(surface, declared, load)
    relative = q[1]-q[0]
    # Align the interior peak with the physical support midpoint, while the
    # two Gauss nodes are away from it. The full-circle elastic operator and
    # constant-preload resultant do not depend on this support partition.
    peak = math.atan2(-relative[0], relative[1])
    maximum = surface.radius_km*1000.*math.sin(math.pi/4.)*np.linalg.norm(relative[:2])
    assert maximum > 0.
    declared = traces(offset=peak+math.pi/4., gap=-.95*maximum, height_variation=False)
    load, trace_set, initial = prepare(mesh, surface, lid, declared=declared, order=2)
    expected = exact_locked_solution(surface, declared, load)
    np.testing.assert_allclose(expected, q, rtol=2e-12, atol=1e-18)
    for point in fault_trace_points(surface, trace_set):
        trace = next(item for item in declared if item.trace_id == point.trace_id)
        horizontal = surface.radius_km*1000.*np.cross(point.position, point.horizontal)@(q[1]-q[0])
        assert trace.initial_gap_m+math.sin(trace.dip_radians)*horizontal < 0.
    midpoint = (np.asarray(declared[0].start)+declared[0].end)
    midpoint /= np.linalg.norm(midpoint)
    gap = declared[0].initial_gap_m+math.sin(declared[0].dip_radians)*(
        surface.radius_km*1000.*np.cross(midpoint, declared[0].normal_sub_to_over)@(q[1]-q[0]))
    assert gap > 0.
    before = json.dumps(distributed_contact_dynamics_to_dict(surface, initial, load), sort_keys=True)
    with pytest.raises(UnresolvedContactDynamics, match="[Cc]losed|opening|whole.arc"):
        advance_distributed_contact_dynamics(surface, initial, load, DT)
    assert json.dumps(distributed_contact_dynamics_to_dict(surface, initial, load), sort_keys=True) == before


def archive_world():
    mesh, surface, _ = world()

    def birth(cell, owner, area, time, serial):
        return SurfaceParcel(cell, owner, f"birth:{time.hex()}:{serial}", area, 2.*area,
                             0., 0., 0., (("damage", 0.),))

    motion = np.array([[0., 0., 0.], [0., .01, 0.]])
    transported = advance_geometric_surface(mesh, surface, motion, .001, birth_factory=birth)
    archive = consume_geometric_transaction(surface, transported, initialize_boundary(surface), motion, .001)
    surface = transported.state
    template = traces()[0]
    declared = []
    for index, contact in enumerate(extract_contacts(surface.fragments, surface.radius_km)):
        expected = np.asarray(contact.normal_a_to_b)*(1. if contact.plate_b == 1 else -1.)
        start, end = contact.start, contact.end
        if np.cross(start, end)@expected < 0.:
            start, end = end, start
        declared.append(replace(template, trace_id=f"archive-physical:{index}", start=start, end=end))
    lid = {item.fragment_id: .2+1.3*index for index, item in enumerate(surface.fragments)}
    history = tuple(PolarityEvidence(**item) for item in polarity_evidence(archive))
    return mesh, surface, lid, tuple(declared), archive, history


def test_exact_joint_restart_preserves_physical_point_histories_and_nonempty_archive(tmp_path):
    mesh, surface, lid, declared, archive, history = archive_world()
    assert archive.cohorts
    before_surface, before_archive = asdict(surface), asdict(archive)
    load, _, initial = prepare(mesh, surface, lid, declared=declared, history=history)
    first = advance_distributed_contact_dynamics(surface, initial, load, 1e8)
    direct = advance_distributed_contact_dynamics(surface, first.state, load, 1e8)
    path = tmp_path/"joint_distributed.json"
    raw = distributed_contact_dynamics_to_dict(surface, first.state, load)
    save_boundary_checkpoint(path, surface, archive, provenance={"distributed_contact": raw})
    restored_surface, restored_archive, metadata = load_boundary_checkpoint(path)
    restored = distributed_contact_dynamics_from_dict(metadata["distributed_contact"], restored_surface, load)
    resumed = advance_distributed_contact_dynamics(restored_surface, restored, load, 1e8)
    assert restored == first.state
    assert resumed == direct
    assert restored.polarity_history
    assert resumed.state.preparation_work_j == initial.preparation_work_j > 0.
    assert asdict(restored_surface) == before_surface == asdict(surface)
    assert asdict(restored_archive) == before_archive == asdict(archive)
