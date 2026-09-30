"""Independent continuous-birth and event-time cooling oracles.

The constant-flux cold-volume oracle integrates the half-space solidus depth
analytically, rather than comparing the implementation with itself. Real
Genesis fixtures check the physical event clock, heat ownership and restart.
"""
from dataclasses import asdict, replace
import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import erfinv

from tectonics.fractional_surface import FractionalSurfaceState, SurfaceParcel
from tectonics.fractional_transport import advance_fractional_transport
from tectonics.fractional_thermal import refresh_fractional_mechanics
from tectonics.genesis import GenesisParameters, SECONDS_PER_MYR
from tectonics.genesis_shell import ShellParameters, rock_enthalpy
from tectonics.mesh import build_icosphere

from test_fractional_coupling import source, factory, snapshot


def gauss_rule(count):
    nodes, weights = np.polynomial.legendre.leggauss(count)
    return tuple(zip((nodes+1.)/2., weights/2.))


def transport_fixture(crust=1e-6):
    mesh = build_icosphere(0)
    radius = 100.
    areas = tuple(mesh.physical_cell_areas_km2(radius))
    parcels = tuple(SurfaceParcel(i, int(mesh.centroids[i, 0] > 0.), f"source:{i}",
        area, area*crust, area*10., area*1e12, 50., (("damage", .25),))
        for i, area in enumerate(areas))
    state = FractionalSurfaceState(405., areas, parcels)
    created_at = {}
    def birth(cell, plate, area, time, serial):
        identity = f"birth:{time.hex()}:{serial}"
        created_at[identity] = time
        return SurfaceParcel(cell, plate, identity, area, area*crust, 0., 0., 0., (("damage", 0.),))
    omega = np.array([[0., 0., .05], [0., 0., -.05]])
    return mesh, radius, state, birth, omega, created_at


def fixed_thermal(time):
    model = SimpleNamespace(thermal=GenesisParameters(), shell=ShellParameters())
    ts, tm = 282., 1575.
    depth = model.shell.column_depth_km
    z = (np.arange(model.shell.column_layers)+.5)*depth/model.shell.column_layers
    sample = SimpleNamespace(time_myr=time,
        thermal={"mantle_temperature_k": tm, "surface_temperature_k": ts},
        lid_thickness_km=depth*(model.thermal.solidus_k-ts)/(tm-ts),
        mean_lid_temperature_k=.5*(ts+model.thermal.solidus_k),
        column_depth_limit_reached=False,
        state=SimpleNamespace(column_enthalpy=rock_enthalpy(ts+(tm-ts)*z/depth, model.thermal)))
    return model, sample


def born_parcels(result):
    ids = {p.material_id for p in result.births}
    return tuple(p for p in result.state.parcels if p.material_id in ids)


@pytest.mark.parametrize("count", [2, 4, 8])
def test_constant_flux_births_preserve_uniform_age_moments_and_birth_clock(count):
    mesh, radius, state, birth, omega, created_at = transport_fixture()
    dt = .75
    result = advance_fractional_transport(mesh, state, omega, radius, dt,
        birth_factory=birth, event_time_quadrature=gauss_rule(count))
    assert result.diagnostics["substeps"] == 1
    born = born_parcels(result)
    area = math.fsum(p.area_km2 for p in born)
    assert area > 0.
    assert len({p.age_myr for p in born}) == count
    assert math.fsum(p.area_km2*p.age_myr for p in born)/area == pytest.approx(dt/2., rel=3e-13)
    assert math.fsum(p.area_km2*p.age_myr**2 for p in born)/area == pytest.approx(dt**2/3., rel=5e-13)
    assert all(p.age_myr == pytest.approx(result.state.time_myr-created_at[p.material_id], abs=3e-14)
               for p in born)
    assert all(p.age_myr == p.cold_mantle_volume_km3 == p.density_excess_mass_kg == 0. for p in result.births)
    assert all(state.time_myr < t < result.state.time_myr for t in created_at.values())


def test_birth_cooling_converges_to_analytic_continuous_creation_integral():
    crust, dt = 1e-6, .75
    values = []
    for count in (2, 4, 8):
        mesh, radius, state, birth, omega, _ = transport_fixture(crust)
        result = advance_fractional_transport(mesh, state, omega, radius, dt,
            birth_factory=birth, event_time_quadrature=gauss_rule(count))
        model, sample = fixed_thermal(result.state.time_myr)
        cooled, _ = refresh_fractional_mechanics(result.state, sample, model, origin_time_myr=.75)
        identities = {p.material_id for p in result.births}
        newborn = tuple(p for p in cooled.parcels if p.material_id in identities)
        area = math.fsum(p.area_km2 for p in newborn)
        observed = math.fsum(p.cold_mantle_volume_km3 for p in newborn)/area
        inverse = float(erfinv((model.thermal.solidus_k-282.)/(1575.-282.)))
        coefficient = 2.*math.sqrt(1e-6*SECONDS_PER_MYR)/1000.*inverse
        assert coefficient*math.sqrt(dt) < sample.lid_thickness_km
        a0 = (crust/coefficient)**2
        # Constant creation rate: retained newborn ages are uniform on [0,dt].
        expected = ((2./3.)*coefficient*(dt**1.5-a0**1.5)-crust*(dt-a0))/dt
        values.append(abs(observed/expected-1.))
    assert values[1] < values[0]/4.
    assert values[2] < values[1]/4.
    assert values[0] < .012
    assert values[2] < .0003


def test_event_quadrature_preserves_each_chemical_and_mechanical_inventory():
    mesh, radius, state, birth, omega, _ = transport_fixture()
    result = advance_fractional_transport(mesh, state, omega, radius, .75,
        birth_factory=birth, event_time_quadrature=gauss_rule(4))
    assert result.losses and result.births
    for field in ("area_km2", "oceanic_volume_km3", "cold_mantle_volume_km3", "density_excess_mass_kg"):
        old = math.fsum(getattr(p, field) for p in state.parcels)
        retained = math.fsum(getattr(p, field) for p in result.state.parcels)
        removed = math.fsum(getattr(loss.parcel, field) for loss in result.losses)
        born = math.fsum(getattr(p, field) for p in result.births)
        assert math.fsum((retained, removed, -born, -old))/old == pytest.approx(0., abs=4e-14)
    for loss in result.losses:
        assert state.time_myr < loss.time_myr < result.state.time_myr
        assert loss.parcel.age_myr == pytest.approx(50.+loss.time_myr-state.time_myr, abs=1e-13)
        assert loss.parcel.material_fields == (("damage", .25),)


def test_omitted_transport_rule_retains_explicit_legacy_endpoint_result():
    mesh, radius, state, birth, omega, _ = transport_fixture()
    old = advance_fractional_transport(mesh, state, omega, radius, .75, birth_factory=birth)
    explicit = advance_fractional_transport(mesh, state, omega, radius, .75,
        birth_factory=birth, event_time_quadrature=None)
    assert asdict(old) == asdict(explicit)
    assert all(p.age_myr == 0. for p in born_parcels(old))
    assert all(loss.time_myr == old.state.time_myr for loss in old.losses)


def test_removed_primordial_material_is_cooled_at_its_physical_event_time(source):
    from tectonics.fractional_coupling import advance_coupling, ledger_diagnostics
    model, _, _, state = source
    final = advance_coupling(model.mesh, state, model, .25, birth_factory=factory(source))
    assert final.removed_material
    contexts = {}
    for loss in final.removed_material:
        assert state.surface.time_myr < loss.time_myr < final.surface.time_myr
        if loss.time_myr not in contexts:
            context, _ = model.loading.advance(state.thermal_context, loss.time_myr,
                max_sample_myr=model.parameters.max_loading_interval_myr)
            contexts[loss.time_myr] = model.loading.sample(context)
        sample = contexts[loss.time_myr]
        thickness = max(sample.lid_thickness_km-loss.parcel.specific_properties[0], 0.)
        density = model.shell.density_kg_m3*3e-5*max(sample.thermal["mantle_temperature_k"]-sample.mean_lid_temperature_k, 0.)
        assert loss.parcel.age_myr == pytest.approx(loss.time_myr-state.provenance["origin_time_myr"], abs=2e-14)
        # This deliberately independent integration starts from the outer
        # interval, whereas production starts at a preceding accepted sample.
        # Different adaptive endpoints agree to the Genesis solver's rtol
        # (1e-7), not necessarily to floating-point bit equality.
        assert loss.parcel.cold_mantle_volume_km3 == pytest.approx(loss.parcel.area_km2*thickness, rel=1e-7)
        assert loss.parcel.density_excess_mass_kg == pytest.approx(loss.parcel.area_km2*thickness*density*1e9, rel=1e-7)
    assert any(loss.parcel.cold_mantle_volume_km3 > 0. for loss in final.removed_material)
    assert max(map(abs, ledger_diagnostics(final)["relative_residuals"].values())) < 5e-14
    following = advance_coupling(model.mesh, final, model, .25, birth_factory=factory(source))
    assert following.removed_material[:len(final.removed_material)] == final.removed_material


def test_corrected_event_timing_restarts_all_clocks_cohorts_and_cumulative_sources(source, tmp_path):
    from tectonics.fractional_coupling import advance_coupling, load_coupled_checkpoint, save_coupled_checkpoint
    model, _, _, state = source
    first = advance_coupling(model.mesh, state, model, .25, birth_factory=factory(source))
    path = tmp_path/"events.json"
    save_coupled_checkpoint(path, model.mesh, first)
    loaded = load_coupled_checkpoint(path, model.mesh, model, first.provenance)
    assert snapshot(first) == snapshot(loaded)
    direct = advance_coupling(model.mesh, first, model, .25, birth_factory=factory(source))
    resumed = advance_coupling(model.mesh, loaded, model, .25, birth_factory=factory(source))
    assert snapshot(direct) == snapshot(resumed)
