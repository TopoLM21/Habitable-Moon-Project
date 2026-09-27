"""Independent physical-time controls for a prescribed enriched crack path."""
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
import json

import numpy as np
import pytest

from tectonics.genesis_contact import ContactParameters
from tectonics.genesis_coupled import CoupledModel
from tectonics.genesis_coupled_thermal import advance_thermal_loading
from tectonics.genesis_crack_path import CrackInterval, ReferenceCrackPath
from tectonics.genesis_path_basis import EmbeddedPathBasis
from tectonics.genesis_path_dynamics import PathLoading, PathMechanics, _PathRetry
from tectonics.genesis_path_mesh import insert_crack_path
from tectonics.genesis_shell import maxwell_factors
from tectonics.mesh import build_icosphere
from test_genesis_contact import _source_bytes


def _same(first, second, *, omit=()):
    for item in fields(first):
        if item.name in omit:
            continue
        left, right = getattr(first, item.name), getattr(second, item.name)
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right, err_msg=item.name)
        elif is_dataclass(left):
            _same(left, right)
        else:
            assert left == right, item.name


def _basis(mesh=None, radius_m=5.3e6, poisson=.25):
    mesh = build_icosphere(1) if mesh is None else mesh
    points = np.array([[.6, .2, .2], [.2, .5, .3], [.25, .2, .55]])@mesh.vertices[mesh.faces[0]]
    points /= np.linalg.norm(points, axis=1)[:, None]
    path = ReferenceCrackPath(points, radius_m/1000.)
    insertion = insert_crack_path(mesh, path)
    return EmbeddedPathBasis(mesh, insertion, radius_m, poisson)


def _model(parameters=None):
    basis = _basis()
    depth = np.full(basis.topology.mesh.cell_count, 1e4)
    return PathMechanics(basis, depth, parameters=parameters)


def _material(model):
    basis = model.basis
    volume = basis.subdivision.mesh.areas_unit_sphere*basis.radius_m**2*1e4
    elasticity = np.broadcast_to(60e9*basis.membrane.d, (len(volume), 3, 3)).copy()
    return volume, elasticity


def _isothermal(model, state, dt, *, viscosity=1e22, external=None):
    volume, elasticity = _material(model)
    if external is None:
        external = np.zeros(model.basis.ndof)
    return model.isothermal_loading(state, dt, volume, elasticity,
        np.full(len(volume), viscosity), 60e9, external, np.zeros(len(volume)))


def test_uniform_thermal_predictor_has_analytic_radial_relaxation():
    model = _model()
    n = model.basis.topology.mesh.cell_count
    before = model.initial(np.zeros((n, 3)))
    volume, elasticity = _material(model)
    eigen, beta = 1e-5, .8
    memory = np.tile([eigen, eigen, 0.], (n, 1))
    loading = PathLoading(10., volume, elasticity, memory, np.full(n, beta),
        np.zeros(model.basis.ndof), np.zeros(n))
    after = model.trial(before, loading)
    expected = np.zeros(model.basis.ndof)
    expected[model.basis.nparent-1] = -model.basis.radius_m*eigen/beta
    np.testing.assert_allclose(after.displacement_m, expected, rtol=2e-12, atol=2e-10)
    np.testing.assert_allclose(after.elastic_strain, 0., rtol=0., atol=1e-17)
    assert after.elapsed_years == 10.
    assert after.equilibrium_residual <= model.parameters.equilibrium_tolerance
    assert after.external_work_j == 0.


def test_isothermal_predictor_retains_maxwell_memory_exactly():
    model = _model()
    n = model.basis.topology.mesh.cell_count
    elastic = np.random.default_rng(130).normal(size=(n, 3))*1e-6
    before = model.initial(elastic)
    original = deepcopy(before)
    eta, years = 4e19, 12.
    loading = _isothermal(model, before, years, viscosity=eta)
    retention, increment = maxwell_factors(years*365.25*86400., np.full(n, eta/60e9))
    np.testing.assert_array_equal(loading.memory, retention[:, None]*elastic)
    np.testing.assert_array_equal(loading.effective_b, increment)
    _same(before, original)


def test_tied_inserted_path_matches_existing_coupled_mechanics_over_shared_thermal_steps(tmp_path):
    data, *_ = _source_bytes(tmp_path, elastic=(2e-6, -1e-6, 3e-6), active=False)
    coupled = CoupledModel(data)
    original = coupled.initial()
    assert len(original.cut_edges) == 0
    source_p = coupled.source_model.p
    basis = _basis(coupled.original_mesh, coupled.radius_m, source_p.poisson_ratio)
    subdivision = basis.subdivision
    model = PathMechanics(basis, subdivision.intensive(coupled.source.depth_m),
                          parameters=coupled.contact_parameters)
    state = model.initial(subdivision.tensor(original.elastic_strain, engineering=True))
    thermal, orbit = coupled.source.thermal_state, coupled.source.orbit
    for years in (1., 7., 13.):
        mesh, radius, _, _, _, _ = coupled._phase_fields(original, thermal)
        loading = advance_thermal_loading(coupled.source_model, mesh=mesh, radius_km=radius,
            layer_mass_kg=coupled.layer_mass_kg, column_enthalpy=original.column_enthalpy,
            elastic_strain=original.elastic_strain, damage=original.damage,
            water_access=original.water_access, boundary_energy_j=original.boundary_energy_j,
            thermal_state=thermal, orbit=orbit, target_myr=thermal.time_myr+years/1e6)
        volume = coupled.layer_mass_kg.sum(axis=1)*loading.fraction/source_p.density_kg_m3
        degradation = source_p.residual_stiffness+(1-source_p.residual_stiffness)*(1-loading.damage0)**2
        elasticity = (source_p.young_modulus_pa*degradation)[:, None, None]*coupled.original_membrane.d
        external = coupled._external(coupled.source, loading.depth_km)
        path_loading = PathLoading(loading.dt_myr*1e6, subdivision.extensive(volume),
            subdivision.intensive(elasticity), subdivision.tensor(loading.memory, engineering=True),
            subdivision.intensive(loading.effective_b),
            basis.external(external, np.zeros(basis.membrane.ndof)),
            subdivision.intensive(loading.water_access))
        history, elastic, _, work, _, cohorts = coupled._solve(original, loading, loading.damage0)
        state = model.trial(state, path_loading)
        np.testing.assert_allclose(state.displacement_m[:basis.nparent], history.displacement_m,
                                   rtol=2e-8, atol=2e-9)
        np.testing.assert_array_equal(state.displacement_m[basis.nparent:], 0.)
        np.testing.assert_allclose(state.elastic_strain, subdivision.tensor(elastic, engineering=True),
                                   rtol=2e-8, atol=2e-16)
        assert state.external_work_j == pytest.approx(history.external_work_j, rel=2e-8)
        assert state.drag_work_j == pytest.approx(history.drag_work_j, rel=2e-8)
        assert state.equilibrium_residual <= model.parameters.equilibrium_tolerance
        assert state.active_interval is None
        np.testing.assert_array_equal(state.cohorts.damage, 0.)
        np.testing.assert_array_equal(state.cohorts.traction_pa, 0.)
        original = replace(original, contact=history, elastic_strain=elastic, cohorts=cohorts,
            time_myr=loading.thermal_state.time_myr, column_enthalpy=loading.column_enthalpy,
            damage=loading.damage0, water_access=loading.water_access,
            boundary_energy_j=loading.boundary_energy_j)
        thermal, orbit = loading.thermal_state, loading.orbit


def test_prescribed_release_creates_fresh_contact_without_rewriting_bulk_history():
    model = _model()
    n = model.basis.topology.mesh.cell_count
    elastic = np.random.default_rng(11).normal(size=(n, 3))*1e-6
    before = model.initial(elastic)
    original = deepcopy(before)
    interval = CrackInterval(0., model.basis.insertion.path.length_m)
    after = model.release(before, interval)
    _same(before, original)
    np.testing.assert_array_equal(after.displacement_m, before.displacement_m)
    np.testing.assert_array_equal(after.elastic_strain, before.elastic_strain)
    assert after.elapsed_years == before.elapsed_years
    assert after.active_interval == interval
    assert not after.released_reaction_measured
    assert len(after.cohorts.trace_index) > 0
    for name in ("plastic_slip_m", "cumulative_slip_m", "max_opening_m", "damage",
                 "traction_pa", "friction_work_j", "viscous_work_j", "fracture_work_j"):
        np.testing.assert_array_equal(getattr(after.cohorts, name), 0.)
    assert np.all(after.cohorts.bonded)


def test_zero_load_released_contact_cannot_invent_motion_or_dissipation():
    model = _model()
    n = model.basis.topology.mesh.cell_count
    before = model.release(model.initial(np.zeros((n, 3))),
        CrackInterval(0., model.basis.insertion.path.length_m))
    original = deepcopy(before)
    after = model.trial(before, _isothermal(model, before, 100.))
    _same(before, original)
    np.testing.assert_array_equal(after.displacement_m, 0.)
    np.testing.assert_array_equal(after.elastic_strain, 0.)
    np.testing.assert_array_equal(after.cohorts.traction_pa, 0.)
    assert after.elapsed_years == 100.
    assert after.drag_work_j == after.external_work_j == after.bulk_work_j == 0.
    assert after.mechanical_remainder_j == 0.


def test_release_reports_removed_tie_reaction_without_cancelling_it_by_an_invented_force():
    model = _model()
    count = model.basis.topology.mesh.cell_count
    elastic = np.random.default_rng(121).normal(size=(count, 3))*1e-4
    initial = model.initial(elastic)
    tied = model.trial(initial, _isothermal(model, initial, 10.))
    expected = np.linalg.norm(tied.constraint_reaction_n[model.basis.nparent:])
    assert expected > 0
    released = model.release(tied, CrackInterval(0., model.basis.insertion.path.length_m))
    assert released.released_reaction_norm_n == expected
    assert released.released_reaction_measured
    np.testing.assert_array_equal(released.constraint_reaction_n, 0.)
    np.testing.assert_array_equal(released.displacement_m, tied.displacement_m)
    assert released.external_work_j == tied.external_work_j
    after = model.trial(released, _isothermal(model, released, 10.))
    assert np.linalg.norm(after.displacement_m[model.basis.nparent:]) > 0
    assert after.equilibrium_residual <= model.parameters.equilibrium_tolerance


def test_invalid_or_stopped_release_cannot_heal_or_reset_reference_limits():
    model = _model()
    state = _nonlinear_initial(model)
    with pytest.raises(ValueError):
        model.release(state, None)
    with pytest.raises(ValueError):
        model.release(state, CrackInterval(0., model.basis.insertion.path.length_m))
    tied = model.initial(np.zeros_like(state.elastic_strain))
    stopped = replace(tied, stopped_reason="path_reference_strain_limit")
    with pytest.raises(ValueError):
        model.release(stopped, CrackInterval(0., model.basis.insertion.path.length_m))


def test_geometric_rejection_preserves_input_state():
    model = _model()
    n = model.basis.topology.mesh.cell_count
    before = model.initial(np.zeros((n, 3)))
    original = deepcopy(before)
    volume, elasticity = _material(model)
    memory = np.tile([.1, .1, 0.], (n, 1))
    loading = PathLoading(1., volume, elasticity, memory, np.ones(n),
                          np.zeros(model.basis.ndof), np.zeros(n))
    with pytest.raises(_PathRetry):
        model.trial(before, loading)
    _same(before, original)


def _nonlinear_initial(model, amplitude=1e-4):
    count = model.basis.topology.mesh.cell_count
    elastic = np.random.default_rng(121).normal(size=(count, 3))*amplitude
    return model.release(model.initial(elastic),
        CrackInterval(0., model.basis.insertion.path.length_m))


def test_nonlinear_contact_restart_replays_all_constitutive_history_exactly(tmp_path):
    model = _model()
    before = _nonlinear_initial(model)
    first = model.advance(before, 40., lambda state, dt: _isothermal(model, state, dt), max_step_years=10.)
    assert np.max(first.cohorts.damage) > 0
    assert first.cohorts.fracture_work_j.sum() > 0
    assert first.cohorts.friction_work_j.sum() > 0
    path = tmp_path/"mechanics.npz"
    model.save_state(path, first)
    loaded = model.load_state(path)
    _same(first, loaded)
    direct = model.trial(first, _isothermal(model, first, 20.))
    resumed = model.trial(loaded, _isothermal(model, loaded, 20.))
    _same(direct, resumed)
    changed = PathMechanics(model.basis, model.depth_m,
        parameters=replace(model.parameters, basal_drag_pa_s_m=2*model.parameters.basal_drag_pa_s_m))
    with pytest.raises(ValueError, match="different"):
        changed.load_state(path)


def test_adaptive_rejections_leave_exact_accepted_history_uncontaminated():
    model = _model()
    before = _nonlinear_initial(model, amplitude=1e-3)
    untouched = deepcopy(before)
    calls = []

    def loading_factory(state, dt):
        calls.append((deepcopy(state), dt))
        return _isothermal(model, state, dt)

    after = model.advance(before, 10., loading_factory, max_step_years=10.)
    _same(before, untouched)
    assert after.stopped_reason is None and after.elapsed_years == 10.
    assert after.rejected_steps > 0 and after.accepted_steps > 1
    # Replay only successful mechanics increments from the same state. The
    # rejected larger predictors must contribute no slip, relaxation or work.
    reference = before
    rejections = 0
    for recorded, dt in calls:
        _same(recorded, reference, omit=("rejected_steps",))
        try:
            reference = model.trial(reference, _isothermal(model, reference, dt))
        except _PathRetry:
            rejections += 1
    assert rejections == after.rejected_steps
    _same(reference, after, omit=("rejected_steps",))


def test_nonlinear_time_refinement_reduces_displacement_and_fracture_work_differences():
    model = _model()
    before = _nonlinear_initial(model)
    finals = [model.advance(before, 40., lambda state, dt: _isothermal(model, state, dt),
        max_step_years=dt) for dt in (10., 5., 2.5, 1.25)]
    assert all(s.elapsed_years == 40. and s.stopped_reason is None for s in finals)
    displacement_error = [np.linalg.norm(a.displacement_m-b.displacement_m)
                          for a, b in zip(finals, finals[1:])]
    fracture_error = [abs(a.cohorts.fracture_work_j.sum()-b.cohorts.fracture_work_j.sum())
                      for a, b in zip(finals, finals[1:])]
    for errors in (displacement_error, fracture_error):
        assert 0 < errors[2] < .7*errors[1] < .49*errors[0]
    assert finals[-1].cohorts.damage.max() > 0
    assert finals[-1].cohorts.friction_work_j.sum() > 0


@pytest.mark.parametrize("corruption", ["traction", "future_birth", "negative_counter",
                                         "fractional_counter", "nonfinite_work", "free_reaction"])
def test_restart_rejects_inconsistent_constitutive_and_clock_history(tmp_path, corruption):
    model = _model()
    state = _nonlinear_initial(model)
    state = model.trial(state, _isothermal(model, state, 10.))
    path = tmp_path/"original.npz"
    model.save_state(path, state)
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key].copy() for key in archive.files}
    metadata = json.loads(str(values["metadata"]))
    if corruption == "traction":
        values["cohort_traction_pa"][0, 0] += 100.
    elif corruption == "future_birth":
        values["cohort_birth_time_myr"][0] = state.elapsed_years/1e6+1.
    elif corruption == "negative_counter":
        metadata["accepted_steps"] = -1
    elif corruption == "fractional_counter":
        metadata["rejected_steps"] = .25
    elif corruption == "nonfinite_work":
        metadata["external_work_j"] = float("nan")
    elif corruption == "free_reaction":
        values["constraint_reaction_n"][0] = 1.
    values["metadata"] = np.array(json.dumps(metadata))
    damaged = tmp_path/"damaged.npz"
    np.savez(damaged, **values)
    with pytest.raises(ValueError):
        model.load_state(damaged)


def test_checkpoint_fingerprint_includes_path_order_and_poisson_basis(tmp_path):
    model = _model()
    state = _nonlinear_initial(model)
    path = tmp_path/"mechanics.npz"
    model.save_state(path, state)
    basis = deepcopy(model.basis)
    insertion = basis.insertion
    reverse_path = ReferenceCrackPath(insertion.path.points_xyz[::-1], insertion.path.radius_km)
    basis.insertion = replace(insertion, path=reverse_path,
        path_vertex_ids=insertion.path_vertex_ids[::-1],
        path_arclength_m=insertion.path.length_m-insertion.path_arclength_m[::-1])
    reversed_model = PathMechanics(basis, model.depth_m)
    assert reversed_model.fingerprint != model.fingerprint
    with pytest.raises(ValueError, match="different"):
        reversed_model.load_state(path)
    changed_basis = EmbeddedPathBasis(model.basis.subdivision.parent_mesh,
        insertion, model.basis.radius_m, .3)
    changed_model = PathMechanics(changed_basis, model.depth_m)
    assert changed_model.fingerprint != model.fingerprint
    with pytest.raises(ValueError, match="different"):
        changed_model.load_state(path)


@pytest.mark.parametrize("corruption", ["missing_array", "missing_metadata", "extra_array",
    "metadata_shape", "metadata_list", "unknown_version", "malformed_interval", "bad_json"])
def test_checkpoint_schema_rejects_partial_or_malformed_archives(tmp_path, corruption):
    model = _model()
    state = _nonlinear_initial(model)
    original = tmp_path/"original.npz"
    model.save_state(original, state)
    with np.load(original, allow_pickle=False) as archive:
        values = {key: archive[key].copy() for key in archive.files}
    metadata = json.loads(str(values["metadata"]))
    if corruption == "missing_array":
        values.pop("elastic_strain")
    elif corruption == "missing_metadata":
        metadata.pop("fingerprint")
    elif corruption == "extra_array":
        values["unused"] = np.zeros(1)
    elif corruption == "metadata_shape":
        values["metadata"] = np.array([str(values["metadata"])])
    elif corruption == "metadata_list":
        metadata = []
    elif corruption == "unknown_version":
        metadata["version"] = "genesis-path-mechanics-unknown"
    elif corruption == "malformed_interval":
        metadata["active_interval"] = "full path"
    elif corruption == "bad_json":
        values["metadata"] = np.array("not json")
    if corruption not in {"metadata_shape", "bad_json"}:
        values["metadata"] = np.array(json.dumps(metadata))
    damaged = tmp_path/"damaged.npz"
    np.savez(damaged, **values)
    with pytest.raises(ValueError):
        model.load_state(damaged)


def test_checkpoint_load_rechecks_reference_geometry(tmp_path):
    model = _model()
    state = model.initial(np.zeros((model.basis.topology.mesh.cell_count, 3)))
    state = model.trial(state, _isothermal(model, state, 1.))
    original = tmp_path/"original.npz"
    model.save_state(original, state)
    with np.load(original, allow_pickle=False) as archive:
        values = {key: archive[key].copy() for key in archive.files}
    # A radial shift leaves all bank jumps and local contact history unchanged,
    # so only the global reference-geometry guard can reject this corruption.
    values["displacement_m"][model.basis.nparent-1] = .1*model.basis.radius_m
    damaged = tmp_path/"damaged.npz"
    np.savez(damaged, **values)
    with pytest.raises(ValueError, match="inadmissible"):
        model.load_state(damaged)
