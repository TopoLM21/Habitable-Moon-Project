"""Versioned source meaning and actual continuation coupling contracts."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from tectonics.genesis_starter_continuation import build_starter_continuation, YoungWorldCoupling
from tectonics.genesis_starter_material import independent_mantle_omega, independent_mantle_source_omega
from tectonics.genesis_young_mechanics import (MECHANICS_MODEL_VERSION, LEGACY_MECHANICS,
    mechanics_version, advance_prescribed_source)
from test_genesis_starter_continuation_independent import source
from test_genesis_starter_continuation_thermal import isolated_runner


def test_fresh_import_records_uncoupled_source_and_sinking_closure(source):
    model, state, cfg = source
    bundle, config, report = build_starter_continuation(model, state, cfg)
    assert mechanics_version(config) == MECHANICS_MODEL_VERSION
    assert config['young_shell']['origin_time_myr'] == state.time_myr
    assert config['plate_dynamics']['young_slab_force_model'] == 'viscous_sinking_v1'
    assert config['plate_dynamics']['young_velocity_response_model'] == 'quasistatic'
    assert config['plate_dynamics']['young_slab_buoyancy_model'] == 'ordered_thermal_cohorts_v1'
    assert config['subduction_memory']['young_slab_buoyancy_model'] == 'ordered_thermal_cohorts_v1'
    assert config['subduction_memory']['young_slab_connectivity_model'] == 'local_edge_transfer_v1'
    assert report['mechanics_model_version'] == MECHANICS_MODEL_VERSION
    np.testing.assert_array_equal(bundle.checkpoint.mantle_flow.cell_omega_rad_per_myr,
                                  independent_mantle_source_omega(model))
    assert bundle.checkpoint.subduction_memory.young_boundary_state is not None
    assert bundle.checkpoint.subduction_memory.young_boundary_state.connectivity_model == 'local_edge_transfer_v1'
    assert bundle.checkpoint.subduction_memory.young_boundary_state.buoyancy_geometry_model == 'ordered_thermal_cohorts_v1'
    # The archived first-partition kinematics are never rescaled on import.
    from tectonics.dynamics import angular_velocity_vectors
    np.testing.assert_array_equal(angular_velocity_vectors(bundle.checkpoint.system),
                                  angular_velocity_vectors(state.system))


def test_legacy_import_and_missing_version_retain_original_source_meaning(source):
    model, state, cfg = source
    config = deepcopy(cfg)
    config['young_shell'] = {'mechanics_model_version': LEGACY_MECHANICS}
    bundle, config, _ = build_starter_continuation(model, state, config)
    np.testing.assert_array_equal(bundle.checkpoint.mantle_flow.cell_omega_rad_per_myr,
                                  independent_mantle_omega(model, state))
    config['young_shell'].pop('mechanics_model_version')
    assert mechanics_version(config) == LEGACY_MECHANICS
    assert not YoungWorldCoupling(model, state, config, bundle.checkpoint).new_mechanics


def test_unknown_or_changed_source_provenance_is_rejected(source):
    model, state, cfg = source
    bundle, cfg, _ = build_starter_continuation(model, state, cfg)
    cfg = deepcopy(cfg)
    cfg['young_shell']['basal_source']['seed'] += 1
    with pytest.raises(ValueError, match='provenance'):
        YoungWorldCoupling(model, state, cfg, bundle.checkpoint)
    cfg['young_shell']['mechanics_model_version'] = 'future'
    with pytest.raises(ValueError, match='Unsupported'):
        YoungWorldCoupling(model, state, cfg, bundle.checkpoint)


def test_prescribed_source_has_no_partition_clock_or_heat_flux_memory(source):
    model, state, cfg = source
    bundle, _, _ = build_starter_continuation(model, state, cfg)
    initial = bundle.checkpoint.mantle_flow
    a, _ = advance_prescribed_source(model.mesh, initial, 10., .01)
    b, _ = advance_prescribed_source(model.mesh, initial, 4., 100.)
    b, _ = advance_prescribed_source(model.mesh, b, 6., 0.)
    np.testing.assert_array_equal(a.cell_omega_rad_per_myr, b.cell_omega_rad_per_myr)
    np.testing.assert_array_equal(a.cell_omega_rad_per_myr, initial.cell_omega_rad_per_myr)
    assert a.time_myr == b.time_myr
    assert not np.shares_memory(a.cell_omega_rad_per_myr, initial.cell_omega_rad_per_myr)


def test_installed_dynamics_uses_current_lid_and_endpoint_material_age(source):
    model, state, cfg = source
    bundle, cfg, _ = build_starter_continuation(model, state, cfg)
    cp = bundle.checkpoint
    coupling = YoungWorldCoupling(model, state, cfg, cp)
    runner = isolated_runner()
    seen = {}
    def capture(*args, **kwargs):
        seen.update(material=args[1], flow=kwargs['mantle_flow'])
        return args[2], None, [], None
    runner.base.update_plate_dynamics = capture
    coupling.install(runner)
    before = deepcopy(cp.state)
    coupling.advance_heat(cp.thermal, 1.)
    runner.base.update_plate_dynamics(model.mesh, cp.state, cp.system, cp.baseline,
        model.thermal.radius_km, 1., 4., 1., None, mantle_flow=cp.mantle_flow)
    expected = independent_mantle_omega(model, coupling.source_state)
    np.testing.assert_allclose(seen['flow'].cell_omega_rad_per_myr, expected, rtol=2e-15, atol=1e-18)
    assert np.linalg.norm(expected) > np.linalg.norm(independent_mantle_omega(model, state))
    np.testing.assert_array_equal(cp.state.crust_age_myr, before.crust_age_myr)
    np.testing.assert_array_equal(cp.state.mantle_lithosphere_thickness_km, before.mantle_lithosphere_thickness_km)
    np.testing.assert_array_equal(cp.mantle_flow.cell_omega_rad_per_myr, independent_mantle_source_omega(model))
