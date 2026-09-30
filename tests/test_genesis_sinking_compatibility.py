"""Independent version, rheology, and saved-state contracts for slab sinking."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math

import numpy as np
import pytest
import yaml

from tectonics.dynamics import DynamicsParameters, angular_velocity_vectors, update_plate_dynamics
from tectonics.genesis_starter_continuation import YoungWorldCoupling, build_starter_continuation
from tectonics.genesis_starter_continuation import mechanics_limitations, NEW_MECHANICS_LIMITATIONS, LIMITATIONS
from tectonics.genesis_young_mechanics import (BASAL_RIDGE_MECHANICS, UNIFORM_SINKING_MECHANICS,
    transmitted_mantle_flow)
from tectonics.genesis_continuation_remesh import remesh_continuation_state
from tectonics.genesis_starter_fracture import YoungShellFracture
from tectonics.kinematics import classify_boundaries
from tectonics.mantle_convection import mantle_viscosity_pa_s
from tectonics.subduction_memory import (SubductionMemoryParameters, memory_from_json, memory_to_json,
    advance_subduction_memory)
from tectonics.young_boundary import (SlabAcceptance, accept_slab_material, contact_key,
    accepted_slab_force_sections, synchronize_young_zones, slab_thermal_deficit_fraction)
from test_genesis_starter_continuation_independent import source
from test_genesis_starter_continuation_thermal import isolated_runner


def imported(source, version=None):
    model, state, original = source
    config = deepcopy(original)
    if version is not None:
        config['young_shell'] = {'mechanics_model_version': version}
    bundle, config, _ = build_starter_continuation(model, state, config)
    return model, state, bundle.checkpoint, config


def force_call(model, cp, cfg, coupling, parameters, step):
    material = deepcopy(cp.state)
    coupling.mechanical_fields(material)
    thickness = coupling.local_mechanical_diagnostics['local_total_lid_thickness_km']
    flow = transmitted_mantle_flow(model, cp.mantle_flow, thickness)
    trace = {}
    result, _, _, _ = update_plate_dynamics(model.mesh, material, cp.system, cp.baseline,
        model.thermal.radius_km, step, **cfg['classification'], params=parameters,
        mantle_flow=flow, subduction_memory=cp.subduction_memory,
        subduction_memory_params=SubductionMemoryParameters(**cfg['subduction_memory']),
        young_slab_strength_pa=coupling.fracture.memory.strength_pa.copy(),
        trace=trace)
    return result, trace


def test_archived_basal_ridge_configuration_retains_relaxation_and_fixed_contacts(source):
    model, state, cp, cfg = imported(source, BASAL_RIDGE_MECHANICS)
    # These fields did not exist in the saved 0.3 schema.
    cfg['plate_dynamics'].pop('young_velocity_response_model')
    cfg['subduction_memory'].pop('young_slab_connectivity_model')
    serialized = memory_to_json(cp.subduction_memory)
    assert 'connectivity_model' not in serialized['young_boundary_state']
    cp.subduction_memory = memory_from_json(serialized)
    before = deepcopy(cfg)
    coupling = YoungWorldCoupling(model, state, cfg, cp)
    params = DynamicsParameters(**cfg['plate_dynamics'])
    assert coupling.dynamics_parameters(params) is params
    assert params.young_velocity_response_model == 'relaxed'
    assert params.young_slab_force_model == 'disabled_pending_closure'
    assert cp.subduction_memory.young_boundary_state.connectivity_model == 'legacy_fixed_contacts'
    _, trace = force_call(model, cp, cfg, coupling, params, .7)
    alpha = -math.expm1(-.7/params.velocity_relaxation_myr)
    assert trace['alpha'] == alpha
    np.testing.assert_array_equal(trace['final_omega'],
        trace['current_omega']+alpha*(trace['target_omega']-trace['current_omega']))
    # New rheology knobs have no effect on the archived basal/ridge equation.
    altered = replace(params, young_slab_viscosity_contrast=1e9,
        young_slab_bend_radius_thickness_ratio=1000.,
        young_slab_mantle_shear_length_fraction=100.,
        young_slab_mantle_viscosity_pa_s=1e30,
        young_slab_mantle_depth_km=1e9)
    _, alternate = force_call(model, cp, cfg, coupling, altered, .7)
    for key in ('target_omega', 'final_omega', 'basal_driving_torque_nm', 'ridge_torque_nm'):
        np.testing.assert_array_equal(trace[key], alternate[key])
    assert cfg == before
    assert memory_to_json(cp.subduction_memory) == serialized


@pytest.mark.parametrize('positional', [False, True])
def test_runtime_hook_uses_current_genesis_viscosity_without_editing_saved_parameters(source, positional):
    model, state, cp, cfg = imported(source)
    coupling = YoungWorldCoupling(model, state, cfg, cp)
    runner = isolated_runner()
    seen = {}
    def capture(*args, **kwargs):
        seen['params'] = args[8] if len(args) > 8 else kwargs['params']
        seen['strength'] = kwargs['young_slab_strength_pa']
        return args[2], None, [], None
    runner.base.update_plate_dynamics = capture
    coupling.install(runner)
    params = DynamicsParameters(**cfg['plate_dynamics'])
    initial_viscosity = params.young_slab_mantle_viscosity_pa_s
    source_energy = state.thermal_context.thermal.energy.copy()
    coupling.advance_heat(cp.thermal, 1.)
    before_call = coupling.source_state.thermal_context.thermal.energy.copy()
    sample = model.loading.sample(coupling.source_state.thermal_context)
    expected = mantle_viscosity_pa_s(sample.thermal['mantle_temperature_k'], model.thermal)
    assert expected != initial_viscosity
    args = (model.mesh, cp.state, cp.system, cp.baseline, model.thermal.radius_km, 1., 4., 1.)
    if positional:
        runner.base.update_plate_dynamics(*args, params, mantle_flow=cp.mantle_flow)
    else:
        runner.base.update_plate_dynamics(*args, params=params, mantle_flow=cp.mantle_flow)
    assert seen['params'].young_slab_mantle_viscosity_pa_s == expected
    assert seen['params'].young_slab_mantle_depth_km == (
        model.thermal.radius_km*model.thermal.mantle_depth_fraction_radius)
    assert params.young_slab_mantle_viscosity_pa_s == initial_viscosity
    assert cfg['plate_dynamics']['young_slab_mantle_viscosity_pa_s'] == initial_viscosity
    np.testing.assert_array_equal(seen['strength'], coupling.fracture.memory.strength_pa)
    assert not np.shares_memory(seen['strength'], coupling.fracture.memory.strength_pa)
    np.testing.assert_array_equal(state.thermal_context.thermal.energy, source_energy)
    np.testing.assert_array_equal(coupling.source_state.thermal_context.thermal.energy, before_call)


def test_new_quasistatic_response_does_not_inherit_45_myr_velocity_lag(source):
    model, state, cp, cfg = imported(source)
    coupling = YoungWorldCoupling(model, state, cfg, cp)
    params = coupling.dynamics_parameters(DynamicsParameters(**cfg['plate_dynamics']))
    small, a = force_call(model, cp, cfg, coupling, params, .1)
    large, b = force_call(model, cp, cfg, coupling, params, 5.)
    assert a['alpha'] == b['alpha'] == 1.
    np.testing.assert_allclose(a['final_omega'], a['target_omega'], rtol=2e-15, atol=1e-18)
    np.testing.assert_array_equal(angular_velocity_vectors(small), angular_velocity_vectors(large))
    torque = np.linalg.norm(a['basal_driving_torque_nm']+a['ridge_torque_nm']+a['slab_torque_nm'])
    assert np.linalg.norm(a['transient_torque_residual_nm']) < 2e-14*torque


def test_saved_connectivity_mismatch_is_rejected_before_evolution(source):
    model, state, cp, cfg = imported(source)
    cp.subduction_memory.young_boundary_state.connectivity_model = 'legacy_fixed_contacts'
    with pytest.raises(ValueError, match='connectivity'):
        YoungWorldCoupling(model, state, cfg, cp)


def test_sinking_cannot_be_enabled_by_relabeling_a_03_configuration(source):
    model, state, cp, cfg = imported(source, BASAL_RIDGE_MECHANICS)
    cfg['plate_dynamics']['young_slab_force_model'] = 'viscous_sinking_v1'
    with pytest.raises(ValueError, match='version 0.4'):
        YoungWorldCoupling(model, state, cfg, cp)


def test_sinking_rejects_legacy_connectivity_even_when_checkpoint_and_configuration_agree(source):
    model, state, cp, cfg = imported(source)
    cfg['subduction_memory']['young_slab_connectivity_model'] = 'legacy_fixed_contacts'
    cp.subduction_memory.young_boundary_state.connectivity_model = 'legacy_fixed_contacts'
    with pytest.raises(ValueError, match='connectivity|local_edge_transfer'):
        YoungWorldCoupling(model, state, cfg, cp)


def add_connected_material(model, cp, cfg):
    """Controlled material fixture, with real current mesh contact geometry."""
    params = SubductionMemoryParameters(**cfg['subduction_memory'])
    boundaries = classify_boundaries(model.mesh, cp.system, model.thermal.radius_km, 0., 0.)
    advance_subduction_memory(model.mesh, cp.state, boundaries, model.thermal.radius_km,
        1., cp.subduction_memory, params)
    inv = cp.subduction_memory.young_boundary_state
    c = inv.contacts[contact_key(boundaries[0])]
    e = SlabAcceptance(c.face_a, c.face_b, c.plate_a, c.plate_b,
        10., 70., 300., 1.8e13, c.key, c.trench_length_km,
        c.midpoint, c.torque_direction_ab)
    accept_slab_material(model.mesh, inv, [e], cp.state.time_myr)
    synchronize_young_zones(cp.subduction_memory, cp.state, params)
    return params


def test_whole_checkpoint_refinement_preserves_new_connected_inventory_and_valid_source(source):
    model, state, cp, cfg = imported(source)
    params = add_connected_material(model, cp, cfg)
    before_memory = memory_to_json(cp.subduction_memory)
    before_cfg = deepcopy(cfg)
    (coarse,) = accepted_slab_force_sections(cp.subduction_memory, params=params)
    remeshed = remesh_continuation_state(model, state, YoungShellFracture(model, state), cp, cfg, 2)
    YoungWorldCoupling(remeshed.model, remeshed.starter_state, remeshed.config,
        remeshed.checkpoint, fracture=remeshed.fracture)
    sections = accepted_slab_force_sections(remeshed.checkpoint.subduction_memory, params=params)
    assert len(sections) == 2
    assert remeshed.checkpoint.subduction_memory.young_boundary_state.connectivity_model == 'local_edge_transfer_v1'
    for field in ('accepted_area_km2', 'oceanic_volume_km3', 'cold_mantle_volume_km3',
                  'density_excess_mass_kg', 'area_thickness_cubed_km5'):
        assert sum(getattr(s, field) for s in sections) == pytest.approx(getattr(coarse, field), rel=2e-14)
    assert memory_to_json(cp.subduction_memory) == before_memory
    assert cfg == before_cfg


@pytest.mark.parametrize('version', [None, BASAL_RIDGE_MECHANICS])
def test_populated_frozen_diagnostic_matches_installed_runtime_and_leaves_files_unchanged(source, tmp_path, version, monkeypatch):
    from tectonics.checkpoint import save_checkpoint
    from tectonics.plate_velocity_diagnostics import diagnose_checkpoint
    model, state, cp, cfg = imported(source, version)
    sub_params = add_connected_material(model, cp, cfg)
    coupling = YoungWorldCoupling(model, state, cfg, cp)
    coupling.fracture.memory.strength_pa *= 1.75
    runner = isolated_runner()
    runner.base.update_plate_dynamics = update_plate_dynamics
    coupling.install(runner)
    young = tmp_path/'young_context'
    young.mkdir()
    model.save_state(young/'starter_checkpoint.npz', state)
    coupling.fracture.save(young/'fracture_memory.npz')
    (young/'parameters.json').write_text(json.dumps({
        'format': 'genesis-starter-run-0.1', **model.configuration}), encoding='utf-8')
    (tmp_path/'mature_config.yaml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
    save_checkpoint(tmp_path/'mature_checkpoint', cp)
    digests = {str(path.relative_to(tmp_path)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in tmp_path.rglob('*') if path.is_file()}
    (tmp_path/'continuation.json').write_text(json.dumps({
        'checkpoint_sha256': digests, 'checks': {}}), encoding='utf-8')
    trace = {}
    runner.base.update_plate_dynamics(model.mesh, cp.state, cp.system, cp.baseline,
        model.thermal.radius_km, .5, **cfg['classification'],
        params=DynamicsParameters(**cfg['plate_dynamics']), mantle_flow=cp.mantle_flow,
        subduction_memory=cp.subduction_memory, subduction_memory_params=sub_params, trace=trace)
    captured = {}
    def audit_force(*args, **kwargs):
        captured.update(kwargs)
        return update_plate_dynamics(*args, **kwargs)
    monkeypatch.setattr('tectonics.dynamics.update_plate_dynamics', audit_force)
    report = diagnose_checkpoint(tmp_path, step_myr=.5)
    for key in ('target_omega', 'final_omega', 'slab_torque_nm', 'slab_bending_drag_tensor_nm_s',
                'slab_mantle_drag_tensor_nm_s', 'total_dissipation_w'):
        np.testing.assert_array_equal(report['trace'][key], trace[key])
    component_sum = sum((trace[key] for key in ('common_mantle', 'ridge_drive_normalized',
        'slab_drive_normalized', 'slab_constraint_omega')), np.zeros_like(trace['target_omega']))
    assert np.linalg.norm(component_sum-trace['target_omega']) <= (
        5e-12*np.linalg.norm(trace['target_omega'])+1e-18)
    if version is None:
        np.testing.assert_array_equal(captured['young_slab_strength_pa'],
            coupling.fracture.memory.strength_pa)
        assert not np.array_equal(captured['young_slab_strength_pa'], state.strength_pa)
        assert np.linalg.norm(trace['slab_torque_nm']) > 0.
        assert len(trace['slab_sections']) == 1
    else:
        assert 'young_slab_strength_pa' not in captured
        assert not np.any(trace['slab_torque_nm'])
    for relative, digest in digests.items():
        assert hashlib.sha256((tmp_path/relative).read_bytes()).hexdigest() == digest


def test_runtime_commits_solver_failure_even_without_user_trace(source):
    model, state, cp, cfg = imported(source)
    sub_params = add_connected_material(model, cp, cfg)
    coupling = YoungWorldCoupling(model, state, cfg, cp)
    section, = accepted_slab_force_sections(cp.subduction_memory, params=sub_params)
    failure = dict(contact_key=section.contact_key, subducting_plate=section.subducting_plate,
        overriding_plate=section.overriding_plate, tension_n=100., capacity_n=10., iteration=0)
    runner = isolated_runner()
    def solved(*args, **kwargs):
        assert kwargs['trace'] is not None
        kwargs['trace']['slab_neck_failures'] = [failure]
        return args[2], None, [], None
    runner.base.update_plate_dynamics = solved
    coupling.install(runner)
    arguments = (model.mesh, cp.state, cp.system, cp.baseline,
        model.thermal.radius_km, .5)
    options = dict(**cfg['classification'], params=DynamicsParameters(**cfg['plate_dynamics']),
        mantle_flow=cp.mantle_flow, subduction_memory=cp.subduction_memory,
        subduction_memory_params=sub_params)
    runner.base.update_plate_dynamics(*arguments, **options)
    assert accepted_slab_force_sections(cp.subduction_memory, params=sub_params) == ()
    history = cp.subduction_memory.young_boundary_state.mechanical_detachments
    assert len(history) == 1
    assert history[0]['retained_oceanic_volume_km3'] == 70.
    assert not cp.subduction_memory.zones[(section.subducting_plate, section.overriding_plate)].active
    runner.base.update_plate_dynamics(*arguments, **options)
    assert len(history) == 1


@pytest.mark.parametrize('version', [None, UNIFORM_SINKING_MECHANICS, BASAL_RIDGE_MECHANICS])
@pytest.mark.parametrize('positional', [False, True])
def test_slab_thermal_clock_uses_accepted_heat_endpoint_for_sinking_versions(source, version, positional):
    model, state, cp, cfg = imported(source, version)
    sub_params = add_connected_material(model, cp, cfg)
    coupling = YoungWorldCoupling(model, state, cfg, cp)
    runner = isolated_runner()
    runner.base.advance_subduction_memory = advance_subduction_memory
    coupling.install(runner)
    before_time = cp.state.time_myr
    before_ages = cp.state.crust_age_myr.copy()
    before_energy = state.thermal_context.thermal.energy.copy()
    boundaries = classify_boundaries(model.mesh, cp.system, model.thermal.radius_km, 0., 0.)
    inventory = cp.subduction_memory.young_boundary_state
    before_closure = sum(c.cumulative_closure_area_km2 for c in inventory.contacts.values())
    coupling.advance_heat(cp.thermal, 1.)
    if positional:
        runner.base.advance_subduction_memory(model.mesh, cp.state, boundaries,
            model.thermal.radius_km, 1., cp.subduction_memory, sub_params)
    else:
        runner.base.advance_subduction_memory(mesh=model.mesh, state=cp.state, boundaries=boundaries,
            radius_km=model.thermal.radius_km, dt_myr=1., memory=cp.subduction_memory, params=sub_params)
    expected_age = 0. if version == BASAL_RIDGE_MECHANICS else 1.
    assert cp.subduction_memory.time_myr == before_time+expected_age
    section, = accepted_slab_force_sections(cp.subduction_memory, params=sub_params)
    expected_mass = 1.8e13*slab_thermal_deficit_fraction(expected_age, 30.,
        sub_params.young_slab_thermal_diffusivity_m2_s)
    assert section.density_excess_mass_kg == pytest.approx(expected_mass, rel=2e-14)
    expected_dip = sub_params.initial_dip_deg+(sub_params.mature_dip_deg-sub_params.initial_dip_deg)*(
        1.-math.exp(-expected_age/sub_params.dip_maturation_myr))
    assert section.dip_deg == pytest.approx(expected_dip, rel=2e-14)
    after_closure = sum(c.cumulative_closure_area_km2 for c in inventory.contacts.values())
    expected_increment = sum(c.trench_length_km*max(-c.normal_rate_km_per_myr, 0.)
                             for c in inventory.contacts.values() if c.present)
    assert after_closure-before_closure == pytest.approx(expected_increment, rel=2e-14)
    assert cp.state.time_myr == before_time
    np.testing.assert_array_equal(cp.state.crust_age_myr, before_ages)
    np.testing.assert_array_equal(state.thermal_context.thermal.energy, before_energy)


@pytest.mark.parametrize('version', [None, UNIFORM_SINKING_MECHANICS])
def test_generated_limitations_match_enabled_sinking_and_preserve_prior_versions(source, version):
    _, _, _, cfg = imported(source, version)
    text = '\n'.join(mechanics_limitations(cfg))
    assert 'viscous bending and mantle drag' in text
    assert 'default slab force is disabled' not in text
    assert 'upper-bound experiment' not in text
    assert 'finite live tensile strength' in text
    assert 'compositional crust buoyancy' in text
    assert 'frequent detachment' in text
    cfg['plate_dynamics']['young_slab_force_model'] = 'disabled_pending_closure'
    version_number = cfg['young_shell']['mechanics_model_version'].split('-')[-1]
    assert f'explicitly disabled in this {version_number} control' in '\n'.join(mechanics_limitations(cfg))
    cfg['young_shell']['mechanics_model_version'] = BASAL_RIDGE_MECHANICS
    assert mechanics_limitations(cfg) == NEW_MECHANICS_LIMITATIONS
    cfg['young_shell']['mechanics_model_version'] = 'legacy-young-0.2'
    assert mechanics_limitations(cfg) == LIMITATIONS


def test_saved_04_without_new_buoyancy_fields_preserves_uniform_force_law(source):
    model, state, cp, cfg = imported(source, UNIFORM_SINKING_MECHANICS)
    add_connected_material(model, cp, cfg)
    cfg['plate_dynamics'].pop('young_slab_buoyancy_model')
    cfg['subduction_memory'].pop('young_slab_buoyancy_model')
    saved = memory_to_json(cp.subduction_memory)
    assert 'buoyancy_geometry_model' not in saved['young_boundary_state']
    cp.subduction_memory = memory_from_json(saved)
    before = deepcopy(cfg)
    coupling = YoungWorldCoupling(model, state, cfg, cp)
    params = coupling.dynamics_parameters(DynamicsParameters(**cfg['plate_dynamics']))
    assert params.young_slab_buoyancy_model == 'uniform_thermal_mass_v1'
    _, trace = force_call(model, cp, cfg, coupling, params, 1.)
    assert trace['young_slab_buoyancy_model'] == 'uniform_thermal_mass_v1'
    assert cfg == before
    assert memory_to_json(cp.subduction_memory) == saved


@pytest.mark.parametrize('location', ['force', 'memory', 'inventory'])
def test_mismatched_saved_buoyancy_models_are_rejected(source, location):
    model, state, cp, cfg = imported(source)
    if location == 'force':
        cfg['plate_dynamics']['young_slab_buoyancy_model'] = 'uniform_thermal_mass_v1'
    elif location == 'memory':
        cfg['subduction_memory']['young_slab_buoyancy_model'] = 'uniform_thermal_mass_v1'
    else:
        cp.subduction_memory.young_boundary_state.buoyancy_geometry_model = 'uniform_thermal_mass_v1'
    with pytest.raises(ValueError, match='buoyancy geometry'):
        YoungWorldCoupling(model, state, cfg, cp)


def test_ordered_model_cannot_be_silently_enabled_in_saved_04(source):
    model, state, cp, cfg = imported(source, UNIFORM_SINKING_MECHANICS)
    cfg['plate_dynamics']['young_slab_buoyancy_model'] = 'ordered_thermal_cohorts_v1'
    with pytest.raises(ValueError, match='version 0.5'):
        YoungWorldCoupling(model, state, cfg, cp)
