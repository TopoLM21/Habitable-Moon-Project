"""Analytical time-decay and endpoint balance controls for embedded mechanics."""
import numpy as np
import pytest
from scipy import sparse
from scipy.linalg import eigh

from tectonics.genesis_crack_path import CrackInterval, ReferenceCrackPath
from tectonics.genesis_path_basis import EmbeddedPathBasis
from tectonics.genesis_path_dynamics import PathMechanics
from tectonics.genesis_path_mesh import insert_crack_path
from tectonics.mesh import build_icosphere


SECONDS_PER_YEAR = 365.25*86400.


def _fixture():
    parent = build_icosphere(1)
    points = np.array([[.6, .2, .2], [.2, .5, .3], [.25, .2, .55]])
    points = points@parent.vertices[parent.faces[0]]
    points /= np.linalg.norm(points, axis=1)[:, None]
    path = ReferenceCrackPath(points, 5300.)
    insertion = insert_crack_path(parent, path,
        front_coordinates_m=path.length_m*np.array([.2, .8]))
    basis = EmbeddedPathBasis(parent, insertion, 5.3e6, .25)
    model = PathMechanics(basis, np.full(basis.topology.mesh.cell_count, 1e4))
    volume = basis.subdivision.mesh.areas_unit_sphere*basis.radius_m**2*1e4
    elasticity = np.broadcast_to(60e9*basis.membrane.d, (len(volume), 3, 3)).copy()
    return model, volume, elasticity


def _loading(model, volume, elasticity, state, dt, eta=1e21):
    return model.isothermal_loading(state, dt, volume, elasticity,
        np.full(len(volume), eta), 60e9, np.zeros(model.basis.ndof), np.zeros(len(volume)))


def test_combined_maxwell_and_drag_converges_to_independent_continuous_eigenmode():
    model, volume, elasticity = _fixture()
    basis, eta = model.basis, 1e21
    membrane, parent = basis.parent_membrane, basis.subdivision.parent_mesh
    parent_volume = parent.areas_unit_sphere*basis.radius_m**2*1e4
    # Construct the original parent stiffness and drag independently of the
    # hierarchical bulk()/drag assembly under test.
    local = np.einsum("fai,ab,fbj,f->fij", membrane.b, 60e9*membrane.d,
                      membrane.b, parent_volume/basis.radius_m**2)
    stiffness = sparse.coo_matrix((local.ravel(), (membrane.rr, membrane.cc)),
                                  shape=(membrane.ndof, membrane.ndof)).toarray()
    nodal_area = np.zeros(parent.vertex_count)
    np.add.at(nodal_area, parent.faces.ravel(), np.repeat(parent.areas_unit_sphere/3, 3))
    drag = np.repeat(nodal_area*basis.radius_m**2, 2)*model.parameters.basal_drag_pa_s_m
    # The common radial DOF has no drag: eliminate its force equation before
    # constructing a tangent generalized eigenmode K_eff v = lambda D v.
    effective = stiffness[:-1, :-1]-np.outer(stiffness[:-1, -1], stiffness[-1, :-1])/stiffness[-1, -1]
    rates, vectors = eigh(effective, np.diag(drag))
    selected = np.flatnonzero(rates > rates[-1]*1e-6)[-8]
    mode = np.zeros(basis.ndof)
    mode[:basis.nparent-1] = vectors[:, selected]
    mode[basis.nparent-1] = -stiffness[-1, :-1]@vectors[:, selected]/stiffness[-1, -1]
    mode *= 1e-5/np.max(np.abs(basis.strain(mode)))
    initial = model.initial(basis.strain(mode))
    duration = 100.
    # Maxwell relaxation and drag-driven unloading independently contribute
    # to the decay rate. This oracle does not use discrete Maxwell factors.
    exact = np.exp(-(60e9/eta+rates[selected])*duration*SECONDS_PER_YEAR)
    errors = []
    for step in (20., 10., 5., 2.5):
        final = model.advance(initial, duration,
            lambda state, dt: _loading(model, volume, elasticity, state, dt, eta),
            max_step_years=step)
        assert final.stopped_reason is None and final.elapsed_years == duration
        assert final.rejected_steps == 0
        ratio = float(np.sum(final.elastic_strain*initial.elastic_strain)
                      /np.sum(initial.elastic_strain**2))
        # Check that the full field stays in this mode, not only its amplitude.
        np.testing.assert_allclose(final.elastic_strain, ratio*initial.elastic_strain,
                                   rtol=1e-8, atol=2e-15)
        errors.append(abs(ratio-exact))
    assert errors[-1] < 3e-5
    assert all(.45 < finer/coarser < .55 for coarser, finer in zip(errors, errors[1:]))


def test_endpoint_forces_close_with_held_reactions_and_reactions_do_no_work():
    model, volume, elasticity = _fixture()
    basis = model.basis
    initial = model.initial(np.random.default_rng(121).normal(size=(len(volume), 3))*1e-4)
    tied = model.trial(initial, _loading(model, volume, elasticity, initial, 10.))
    interval = CrackInterval(.2*basis.insertion.path.length_m, .8*basis.insertion.path.length_m)
    before = model.release(tied, interval)
    loading = _loading(model, volume, elasticity, before, 10.)
    after = model.trial(before, loading)
    delta = after.displacement_m-before.displacement_m
    stress = np.einsum("fij,fj->fi", elasticity, after.elastic_strain)
    # Reconstruct endpoint bulk force from final stress, not K*du plus memory.
    bulk = basis.strain_operator.T@(volume[:, None]*stress).ravel()/basis.radius_m
    traces = np.zeros((len(model.trace_depth_m), 2))
    np.add.at(traces, after.cohorts.trace_index,
              after.cohorts.area_ref_m2[:, None]*after.cohorts.traction_pa)
    contact = model.jump_operator.T@traces.ravel()
    drag = basis.drag_area_m2*model.parameters.basal_drag_pa_s_m*delta/(10.*SECONDS_PER_YEAR)
    residual = bulk+contact+drag-loading.external_force
    scale = max(np.linalg.norm(bulk), np.linalg.norm(contact), np.linalg.norm(drag))
    assert np.linalg.norm(residual+after.constraint_reaction_n)/scale < 2e-8
    assert np.linalg.norm(residual)/scale > 1e-4
    assert np.linalg.norm(after.displacement_m[basis.nparent:]) > 0
    free = basis.free_dofs(interval)
    held = np.setdiff1d(np.arange(basis.ndof), free)
    assert len(held) > 0
    np.testing.assert_array_equal(delta[held], 0.)
    assert after.constraint_reaction_n@delta == 0.
    endpoint_work = float((bulk+contact+drag)@delta)
    work_scale = max(abs(float(bulk@delta)), abs(float(contact@delta)), abs(float(drag@delta)))
    assert abs(endpoint_work-float(loading.external_force@delta))/work_scale < 2e-8
    assert after.drag_work_j-before.drag_work_j == pytest.approx(float(drag@delta), rel=2e-14)
    # The bulk ledger uses trapezoidal stress, so its difference from endpoint
    # work is exactly the half-increment term, not heat or fracture energy.
    strain_increment = basis.strain(delta)
    stress_increment = np.einsum("fij,fj->fi", elasticity,
                                loading.effective_b[:, None]*strain_increment)
    correction = .5*float(np.sum(volume[:, None]*stress_increment*strain_increment))
    bulk_ledger_increment = after.bulk_work_j-before.bulk_work_j
    assert float(bulk@delta)-bulk_ledger_increment == pytest.approx(correction, rel=2e-8)
