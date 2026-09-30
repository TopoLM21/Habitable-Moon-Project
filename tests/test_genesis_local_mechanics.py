"""Material birth/cooling must survive the young global-column adapter."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import erfinv

from tectonics.checkpoint import save_checkpoint
from tectonics.genesis import GenesisParameters, SECONDS_PER_MYR
from tectonics.genesis_local_mechanics import refresh_young_material_mechanics
from tectonics.genesis_shell import ShellParameters, rock_enthalpy
from tectonics.genesis_starter import StarterModel
from tectonics.genesis_starter_continuation import (
    YoungWorldCoupling, build_starter_continuation, _load_cp)
from tectonics.genesis_starter_fracture import YoungShellFracture
from tectonics.genesis_tides import TidalParameters
from tectonics.mesh import build_icosphere
from tectonics.simulation import load_config
from tectonics.genesis_continuation_remesh import remesh_continuation_state

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", params=["young-mechanics-0.3", "young-mechanics-0.4", "young-mechanics-0.5"])
def real_source(request):
    model = StarterModel(build_icosphere(1), GenesisParameters(), TidalParameters(enabled=False),
        ShellParameters(subdivisions=1, convective_traction_pa=50000.))
    young = model.advance(model.initial_state(), 2.)
    assert young.stopped_reason == "first_partition"
    config = load_config(ROOT/"configs/canonical_moon.yaml")
    config["young_shell"] = {"mechanics_model_version": request.param}
    bundle, cfg, _ = build_starter_continuation(model, young, config)
    return model, young, bundle.checkpoint, cfg


def late_controlled_sample(model):
    """Independent finite conduction column with an exact linear solidus depth."""
    ts, tm = 282., 1575.
    depth = model.shell.column_depth_km
    z = (np.arange(model.shell.column_layers)+.5)*depth/model.shell.column_layers
    h = depth*(model.thermal.solidus_k-ts)/(tm-ts)
    return SimpleNamespace(thermal={"mantle_temperature_k": tm, "surface_temperature_k": ts},
        lid_thickness_km=h, mean_lid_temperature_k=.5*(ts+model.thermal.solidus_k),
        state=SimpleNamespace(column_enthalpy=rock_enthalpy(ts+(tm-ts)*z/depth,model.thermal)))


def late_state(real_source):
    model, young, checkpoint, cfg = real_source
    state = deepcopy(checkpoint.state)
    state.time_myr = young.time_myr+405.
    state.crust_age_myr[:] = 405.
    state.crust_thickness_km[:] = 1.
    return model, state, late_controlled_sample(model), young.time_myr


def test_primordial_column_preserved_at_entry_and_at_late_ages(real_source):
    model, young, checkpoint, _ = real_source
    state = deepcopy(checkpoint.state)
    expected_h = state.mantle_lithosphere_thickness_km.copy()
    expected_rho = state.mantle_lithosphere_density_anomaly_kg_m3.copy()
    diagnostic = refresh_young_material_mechanics(state, model.loading.sample(young.thermal_context),
        model, origin_time_myr=young.time_myr)
    np.testing.assert_array_equal(state.mantle_lithosphere_thickness_km, expected_h)
    np.testing.assert_array_equal(state.mantle_lithosphere_density_anomaly_kg_m3, expected_rho)
    assert diagnostic["rejuvenated_cell_count"] == 0
    model, state, sample, origin = late_state(real_source)
    refresh_young_material_mechanics(state, sample, model, origin_time_myr=origin)
    np.testing.assert_array_equal(state.mantle_lithosphere_thickness_km, sample.lid_thickness_km-1.)
    np.testing.assert_array_equal(state.mantle_lithosphere_density_anomaly_kg_m3,
        model.shell.density_kg_m3*3e-5*(1575.-sample.mean_lid_temperature_k))


def test_actual_material_age_controls_newborn_and_young_support(real_source):
    model, state, sample, origin = late_state(real_source)
    state.crust_age_myr[:5] = [0., .1, .4, 3., 300.]
    source_energy = sample.state.column_enthalpy.copy()
    original_material = state.oceanic_volume_km3.copy()
    diagnostic = refresh_young_material_mechanics(state, sample, model, origin_time_myr=origin)
    h, rho = state.mantle_lithosphere_thickness_km, state.mantle_lithosphere_density_anomaly_kg_m3
    assert h[0] == rho[0] == 0.
    assert 0 < h[1] < h[2] < h[3] < h[5]
    assert rho[3] != pytest.approx(rho[5])
    exact_h = 2*np.sqrt(1e-6*3.*SECONDS_PER_MYR)/1000.*erfinv((1400.-282.)/(1575.-282.))
    assert h[3]+1. == pytest.approx(exact_h, rel=1e-14)
    assert (h[2]+1.)/(h[1]+1.) == pytest.approx(2.)
    assert h[4] == h[5]
    assert rho[4] == pytest.approx(rho[5], rel=1e-14)
    assert diagnostic["rejuvenated_heat_content_deficit_j_m2"][0] == 0.
    assert diagnostic["rejuvenated_heat_content_deficit_j_m2"][3] > 0.
    np.testing.assert_array_equal(sample.state.column_enthalpy, source_energy)
    np.testing.assert_array_equal(state.oceanic_volume_km3, original_material)


def test_donor_age_permutation_and_zero_time_calls_have_no_hidden_history(real_source):
    model, state, sample, origin = late_state(real_source)
    state.crust_age_myr[:5] = [0., .1, 1., 3., 7.]
    refresh_young_material_mechanics(state, sample, model, origin_time_myr=origin)
    expected = deepcopy(state)
    for _ in range(3):
        refresh_young_material_mechanics(state, sample, model, origin_time_myr=origin)
    np.testing.assert_array_equal(state.mantle_lithosphere_thickness_km, expected.mantle_lithosphere_thickness_km)
    np.testing.assert_array_equal(state.mantle_lithosphere_density_anomaly_kg_m3, expected.mantle_lithosphere_density_anomaly_kg_m3)
    permutation = np.arange(len(state.crust_age_myr))[::-1]
    state.crust_age_myr = state.crust_age_myr[permutation]
    state.crust_thickness_km = state.crust_thickness_km[permutation]
    refresh_young_material_mechanics(state, sample, model, origin_time_myr=origin)
    np.testing.assert_array_equal(state.mantle_lithosphere_thickness_km, expected.mantle_lithosphere_thickness_km[permutation])
    np.testing.assert_array_equal(state.mantle_lithosphere_density_anomaly_kg_m3, expected.mantle_lithosphere_density_anomaly_kg_m3[permutation])


def test_material_adapter_is_explicit_and_legacy_retains_global_fields(real_source,monkeypatch):
    model, young, checkpoint, cfg = real_source
    model, state, sample, origin = late_state(real_source)
    state.crust_age_myr[0] = 0.
    monkeypatch.setattr(model.loading,"sample",lambda context: sample)
    legacy_cfg=deepcopy(cfg)
    legacy_cfg["young_shell"].pop("mechanics_model_version")
    legacy=YoungWorldCoupling(model,young,legacy_cfg,deepcopy(checkpoint))
    legacy.mechanical_fields(state,0.)
    assert state.mantle_lithosphere_thickness_km[0] == state.mantle_lithosphere_thickness_km[1]
    modern=YoungWorldCoupling(model,young,deepcopy(cfg),deepcopy(checkpoint))
    modern.mechanical_fields(state,0.)
    assert state.mantle_lithosphere_thickness_km[0] == 0.
    assert state.mantle_lithosphere_thickness_km[1] > 0.


def test_real_heat_checkpoint_and_refinement_preserve_local_age_closure(real_source,tmp_path):
    model, young, checkpoint, cfg = real_source
    cp, cfg = deepcopy(checkpoint), deepcopy(cfg)
    coupling=YoungWorldCoupling(model,young,cfg,cp)
    thermal,_=coupling.advance_heat(cp.thermal,10.)
    cp.thermal=thermal
    cp.state.time_myr=thermal.time_myr
    cp.state.crust_age_myr[:]=10.
    cp.state.crust_age_myr[:3]=[0.,1.,3.]
    cp.state.tidal_damage=coupling.fracture.damage.copy()
    young=coupling.source_state
    heat=deepcopy(young.thermal_context)
    coupling.mechanical_fields(cp.state,0.)
    assert cp.state.mantle_lithosphere_thickness_km[0] == 0.
    assert 0 < cp.state.mantle_lithosphere_thickness_km[1] < cp.state.mantle_lithosphere_thickness_km[3]
    save_checkpoint(tmp_path/"cp",cp)
    restored=_load_cp(tmp_path/"cp",cfg)
    reloaded=YoungWorldCoupling(model,deepcopy(young),deepcopy(cfg),restored,fracture=deepcopy(coupling.fracture))
    reloaded.mechanical_fields(restored.state,0.)
    np.testing.assert_array_equal(restored.state.mantle_lithosphere_thickness_km,cp.state.mantle_lithosphere_thickness_km)
    np.testing.assert_array_equal(restored.state.mantle_lithosphere_density_anomaly_kg_m3,cp.state.mantle_lithosphere_density_anomaly_kg_m3)
    refined=remesh_continuation_state(model,young,coupling.fracture,cp,cfg,2)
    fine=YoungWorldCoupling(refined.model,refined.starter_state,refined.config,refined.checkpoint,fracture=refined.fracture)
    fine.mechanical_fields(refined.checkpoint.state,0.)
    np.testing.assert_array_equal(refined.checkpoint.state.crust_age_myr,np.repeat(cp.state.crust_age_myr,4))
    np.testing.assert_array_equal(refined.checkpoint.state.mantle_lithosphere_thickness_km,np.repeat(cp.state.mantle_lithosphere_thickness_km,4))
    np.testing.assert_array_equal(refined.checkpoint.state.mantle_lithosphere_density_anomaly_kg_m3,np.repeat(cp.state.mantle_lithosphere_density_anomaly_kg_m3,4))
    np.testing.assert_array_equal(coupling.source_state.thermal_context.column_enthalpy,heat.column_enthalpy)
    np.testing.assert_array_equal(coupling.source_state.thermal_context.thermal.energy,heat.thermal.energy)


def test_transitioned_source_diagnostic_retains_subcrust_cooling_depth(real_source):
    model, young, checkpoint, cfg = real_source
    cp,cfg=deepcopy(checkpoint),deepcopy(cfg)
    # Controlled exhausted-column branch: source-coupling must receive the
    # actual cold-layer depth even before it extends below chemical crust.
    cfg["young_shell"]["mechanical_transition"] = {
        "time_myr":young.time_myr,"equivalent_cooling_age_myr":12.,
        "thermal_diffusivity_m2_s":1e-6,"cooling_coefficient":2.,
        "max_total_thickness_km":155.,"mean_temperature_fraction":.5}
    cp.state.crust_age_myr[:3]=[0.,.001,3.]
    coupling=YoungWorldCoupling(model,young,cfg,cp)
    coupling.mechanical_fields(cp.state,0.)
    total=coupling.local_mechanical_diagnostics["local_total_lid_thickness_km"]
    assert total[0] == 0.
    assert 0 < total[1] < cp.state.crust_thickness_km[1]
    assert total[1] == pytest.approx(2*np.sqrt(1e-6*.001*SECONDS_PER_MYR)/1000.)
    assert cp.state.mantle_lithosphere_thickness_km[1] == 0.


@pytest.mark.parametrize("field,value",[("origin_time_myr",-1.),("origin_time_myr",1000.),("thermal_diffusivity_m2_s",0.)])
def test_invalid_closure_clock_and_diffusivity_are_rejected(real_source,field,value):
    model,state,sample,origin=late_state(real_source)
    options=dict(origin_time_myr=origin)
    options[field]=value
    with pytest.raises(ValueError,match="origin, clock, and diffusivity"):
        refresh_young_material_mechanics(state,sample,model,**options)
