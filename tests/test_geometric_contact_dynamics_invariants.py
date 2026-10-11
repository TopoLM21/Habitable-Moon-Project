"""Independent physical invariants of the frozen global contact experiment."""
from dataclasses import asdict, replace
from itertools import product
import json
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import tectonics.fractional_dynamics as fractional_dynamics
from tectonics.basal_coupling import PrescribedBasalParameters
from tectonics.fractional_surface import SurfaceParcel
from tectonics.genesis_contact_law import ContactLawParameters
from tectonics.geometric_basal_loading import assemble_geometric_basal_load
from tectonics.geometric_boundary import (
    consume_geometric_transaction, initialize_boundary, polarity_evidence,
)
from tectonics.geometric_boundary_io import load_boundary_checkpoint, save_boundary_checkpoint
from tectonics.geometric_contact_dynamics import (
    advance_contact_dynamics, contact_dynamics_from_dict, contact_dynamics_to_dict,
    initialize_contact_dynamics,
)
from tectonics.geometric_contact_mechanics import ElasticColumnParameters
from tectonics.geometric_contact_mechanics_adapter import ContactExperiment
from tectonics.geometric_contacts import extract_contacts
from tectonics.geometric_initiation import OrientedFault
from tectonics.geometric_polarity import PolarityEvidence
from tectonics.geometric_surface import GeometricFragment, GeometricSurfaceState, totals
from tectonics.geometric_transport import advance_geometric_surface
from tectonics.mesh import build_icosphere
from tectonics.spherical_polygons import normalize_polygon, polygon_area


PARAMETERS = PrescribedBasalParameters(17, 80000., 1e14, 2.)
DT = 1e10
PROVENANCE = "Declared Genesis source and nonuniform synthetic thermal lid on a fixed surface"


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


def experiments(surface, *, sub=0, over=1):
    law = ContactLawParameters(normal_stiffness_pa_m=1e6, tangential_stiffness_pa_m=1e6,
        tensile_strength_pa=2e6, fracture_energy_j_m2=2e7, cohesion_pa=1e7,
        friction_dry=.4, friction_wet=.2, wet_cohesion_fraction=.5, viscosity_pa_s_m=1e8)
    columns = ElasticColumnParameters(1000., 1e9, 4e8, 1000., 1e9, 4e8)
    return tuple(ContactExperiment(OrientedFault(f"fault:{contact.contact_id}", contact.contact_id,
        sub, over, math.pi/4., None, 5e6, .2, "Explicit pre-existing inclined plane"),
        columns, law, 1000., 1., 0., "Declared elastic columns, wet friction and initial compression")
        for contact in extract_contacts(surface.fragments, surface.radius_km))


def prepare(mesh, surface, lid, *, declared=None, history=()):
    load = assemble_geometric_basal_load(mesh, surface, PARAMETERS, lid, PROVENANCE)
    declared = experiments(surface) if declared is None else declared
    offsets = {item.fault.fault_id: (-1., 0.) for item in declared}
    state = initialize_contact_dynamics(surface, declared, offsets, load,
        initial_loading_dt_s=1e6, max_rotation_radians=.01, history=history)
    return load, state


def test_sparse_owner_permutation_preserves_velocities_tractions_and_work():
    mesh, surface, lid = world()
    load, initial = prepare(mesh, surface, lid)
    original = advance_contact_dynamics(surface, initial, load, DT)
    labels = {0: 10**12, 1: 7}
    relabeled = replace(surface, fragments=tuple(replace(item,
        parcel=replace(item.parcel, plate=labels[item.parcel.plate]))
        for item in reversed(surface.fragments)))
    declared = experiments(relabeled, sub=labels[0], over=labels[1])
    new_load, new_initial = prepare(mesh, relabeled, lid, declared=declared)
    changed = advance_contact_dynamics(relabeled, new_initial, new_load, DT)
    assert new_load.plate_ids == (7, 10**12)
    assert np.asarray(new_load.drag_nm_s).shape == (6, 6)
    for old_index, identity in enumerate(load.plate_ids):
        new_index = new_load.plate_ids.index(labels[identity])
        np.testing.assert_allclose(changed.angular_velocity_rad_s[new_index],
            original.angular_velocity_rad_s[old_index], rtol=2e-11, atol=1e-27)
    before = {item.fault_id: item for item in original.contacts}
    for item in changed.contacts:
        assert item.step.traction_pa == pytest.approx(before[item.fault_id].step.traction_pa, rel=2e-11)
    assert changed.source_work_j == pytest.approx(original.source_work_j, rel=2e-11)
    assert changed.basal_dissipation_j == pytest.approx(original.basal_dissipation_j, rel=2e-11)


def test_passive_coordinate_change_of_geometry_and_physical_source_rotates_global_solution(monkeypatch):
    mesh, surface, lid = world()
    load, initial = prepare(mesh, surface, lid)
    original = advance_contact_dynamics(surface, initial, load, DT)
    rotation = Rotation.from_rotvec([.31, -.17, .24]).as_matrix()
    rotated_mesh = replace(mesh, vertices=mesh.vertices@rotation.T, centroids=mesh.centroids@rotation.T)
    rotated_surface = replace(surface, fragments=tuple(replace(item,
        polygon=tuple(map(tuple, np.asarray(item.polygon)@rotation.T))) for item in surface.fragments))
    original_source = fractional_dynamics.prescribed_vertex_traction

    def rotated_physical_source(current_mesh, **kwargs):
        # A passive coordinate change also rotates the actual prescribed field.
        # Recreating the seed matrix in new axes would specify another load.
        assert current_mesh is rotated_mesh
        return original_source(mesh, **kwargs)@rotation.T

    monkeypatch.setattr(fractional_dynamics, "prescribed_vertex_traction", rotated_physical_source)
    rotated_load, rotated_initial = prepare(rotated_mesh, rotated_surface, lid)
    changed = advance_contact_dynamics(rotated_surface, rotated_initial, rotated_load, DT)
    np.testing.assert_allclose(changed.angular_velocity_rad_s,
        np.asarray(original.angular_velocity_rad_s)@rotation.T, rtol=3e-11, atol=1e-27)
    np.testing.assert_allclose(changed.contact_torque_nm,
        np.asarray(original.contact_torque_nm)@rotation.T, rtol=3e-10,
        atol=np.linalg.norm(original.contact_torque_nm)*3e-12)
    assert changed.source_work_j == pytest.approx(original.source_work_j, rel=3e-11)
    assert changed.basal_dissipation_j == pytest.approx(original.basal_dissipation_j, rel=3e-11)


def test_restart_preserves_preparation_and_all_mechanical_history_exactly():
    mesh, surface, lid = world()
    load, initial = prepare(mesh, surface, lid)
    first = advance_contact_dynamics(surface, initial, load, DT)
    direct = advance_contact_dynamics(surface, first.state, load, DT)
    raw = json.loads(json.dumps(contact_dynamics_to_dict(surface, first.state, load),
                               sort_keys=True, allow_nan=False))
    restored = contact_dynamics_from_dict(raw, surface, load)
    resumed = advance_contact_dynamics(surface, restored, load, DT)
    assert restored == first.state
    assert resumed == direct
    assert resumed.state.preparation_work_j == initial.preparation_work_j
    assert initial.preparation_work_j > 0.


def test_frozen_global_step_and_joint_restart_preserve_nonempty_slab_archive(tmp_path):
    mesh, surface, _ = world()

    def birth(cell, owner, area, time, serial):
        return SurfaceParcel(cell, owner, f"birth:{time.hex()}:{serial}", area, 2.*area,
                             0., 0., 0., (("damage", 0.),))

    motion = np.array([[0., 0., 0.], [0., .01, 0.]])
    transported = advance_geometric_surface(mesh, surface, motion, .001, birth_factory=birth)
    archive = consume_geometric_transaction(surface, transported, initialize_boundary(surface), motion, .001)
    surface = transported.state
    assert archive.cohorts
    before_surface, before_archive = asdict(surface), asdict(archive)
    material_before = totals(surface.fragments)
    buried_before = {field: math.fsum(getattr(item.parcel, field) for item in archive.cohorts)
                    for field in material_before}
    lid = {item.fragment_id: .2+1.3*index for index, item in enumerate(surface.fragments)}
    history = tuple(PolarityEvidence(**item) for item in polarity_evidence(archive))
    load, initial = prepare(mesh, surface, lid, history=history)
    step = advance_contact_dynamics(surface, initial, load, 1e8)
    path = tmp_path/"frozen_joint.json"
    save_boundary_checkpoint(path, surface, archive,
        provenance={"global_contact": contact_dynamics_to_dict(surface, step.state, load)})
    loaded_surface, loaded_archive, metadata = load_boundary_checkpoint(path)
    restored = contact_dynamics_from_dict(metadata["global_contact"], loaded_surface, load)
    assert restored == step.state
    assert asdict(loaded_surface) == before_surface == asdict(surface)
    assert asdict(loaded_archive) == before_archive == asdict(archive)
    assert totals(surface.fragments) == material_before
    assert {field: math.fsum(getattr(item.parcel, field) for item in loaded_archive.cohorts)
            for field in buried_before} == buried_before


def test_declared_arc_mean_stiffness_has_the_expected_subdivision_convergence():
    # Arc means are a fixed Galerkin approximation, not pointwise contact.
    # Splitting physical arcs changes this approximation; no invariance is claimed.
    radius = 100.
    ratios = []
    for count in (4, 8, 16):
        equator = [np.array([math.cos(2*math.pi*i/count), math.sin(2*math.pi*i/count), 0.])
                   for i in range(count)]
        fragments = []
        for owner, pole in enumerate((np.array([0., 0., 1.]), np.array([0., 0., -1.]))):
            for index in range(count):
                polygon = normalize_polygon([pole, equator[index], equator[(index+1) % count]])
                area = polygon_area(polygon)*radius**2
                parcel = SurfaceParcel(index, owner, f"fan-material:{owner}:{index}",
                                       area, area, area, area*1e11, 1.)
                fragments.append(GeometricFragment(f"fan:{owner}:{index}", tuple(map(tuple, polygon)), parcel))
        surface = GeometricSurfaceState(0., radius, tuple(fragments))
        contacts = extract_contacts(surface.fragments, radius)
        assert len(contacts) == count
        # Compare Hessian traces. A single arc mean has rank one and misses an
        # orthogonal displacement mode, so this is not a directional claim.
        discrete = math.fsum(contact.length_km*1000.*float(np.sum((
            np.asarray(contact.moment_arm_cross_normal_km2)/contact.length_km*1000.)**2))
            for contact in contacts)
        exact = 2.*math.pi*(radius*1000.)**3
        ratio = discrete/exact
        assert ratio == pytest.approx((math.sin(math.pi/count)/(math.pi/count))**2, rel=3e-15)
        ratios.append(ratio)
    assert ratios[0] == pytest.approx(8/math.pi**2, rel=3e-15)
    assert 0. < ratios[0] < ratios[1] < ratios[2] < 1.
