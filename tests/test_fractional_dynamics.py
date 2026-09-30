"""Parcel-resolved force quadrature, independent torque and power oracles."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tectonics.basal_coupling import (
    PrescribedBasalParameters, prescribed_cell_basal_state, velocity_to_local_omega,
)
from tectonics.fractional_dynamics import (
    fractional_prescribed_tractions, solve_fractional_basal_dynamics,
)
from tectonics.fractional_surface import FractionalSurfaceState, SurfaceParcel, split_parcel
from tectonics.genesis import SECONDS_PER_MYR
from tectonics.mesh import build_icosphere
from tectonics.young_plate_dynamics import basal_torque_system


RADIUS = 5287.
PARAMETERS = PrescribedBasalParameters(17, 80000., 1e14, 2.)


def world(subdivision=1):
    mesh = build_icosphere(subdivision)
    areas = mesh.physical_cell_areas_km2(RADIUS)
    owner = (mesh.centroids[:, 0] > 0).astype(int)+2*(mesh.centroids[:, 1] > 0).astype(int)
    h = .1+8.*(mesh.centroids[:, 2]+1.)/2.
    parcels = tuple(SurfaceParcel(i, int(owner[i]), f"origin:{i}", float(a),
        float(7.*a), float(20.*a), float(60e9*20.*a), 30.) for i, a in enumerate(areas))
    return mesh, FractionalSurfaceState(30., tuple(areas), parcels), h


def solve(mesh, state, h, **kwargs):
    return solve_fractional_basal_dynamics(mesh, state, RADIUS, PARAMETERS, h, **kwargs)


@pytest.mark.parametrize("uniform", [False, True])
def test_one_component_per_cell_matches_existing_basal_si_solver(uniform):
    mesh, state, h = world(2)
    if uniform:
        h[:] = .216162257
    result = solve(mesh, state, h)
    old = prescribed_cell_basal_state(mesh, PARAMETERS, h)
    owner = np.asarray([p.plate for p in state.parcels])
    flow = SimpleNamespace(cell_omega_rad_per_myr=velocity_to_local_omega(
        mesh.centroids, old.equilibrium_velocity_m_s, RADIUS))
    drag, torque, _ = basal_torque_system(mesh, owner, 4, RADIUS, flow, PARAMETERS.basal_drag_pa_s_m)
    oracle = np.stack([np.linalg.lstsq(d/np.linalg.norm(d), t/np.linalg.norm(d), rcond=None)[0]
                       for d, t in zip(drag, torque)])*SECONDS_PER_MYR
    np.testing.assert_allclose(result.parcel_transmitted_traction_pa, old.transmitted_traction_pa,
                               rtol=1e-13, atol=2e-11)
    np.testing.assert_array_equal(result.basal_drag_tensor_nm_s, drag)
    np.testing.assert_allclose(result.basal_driving_torque_nm, torque, rtol=2e-13,
                               atol=np.linalg.norm(torque)*2e-15)
    np.testing.assert_allclose(result.omega_rad_per_myr, oracle, rtol=2e-13, atol=1e-18)


def test_mixed_lid_quadrature_matches_conditional_legacy_realizations_not_mean_age():
    mesh, state, h = world()
    cell, fraction = 9, .23
    p = state.parcels[cell]
    thin = split_parcel(p, fraction)
    thick = replace(split_parcel(p, 1.-fraction), plate=4, material_id="minority")
    mixed = replace(state, parcels=state.parcels[:cell]+(thin, thick)+state.parcels[cell+1:],
                    known_material_ids=())
    mixed_h = np.insert(h, cell+1, 25.)
    mixed_h[cell] = 0.
    source, traction = fractional_prescribed_tractions(mesh, mixed, PARAMETERS, mixed_h)
    h_thin, h_thick = h.copy(), h.copy()
    h_thin[cell], h_thick[cell] = 0., 25.
    a = prescribed_cell_basal_state(mesh, PARAMETERS, h_thin)
    b = prescribed_cell_basal_state(mesh, PARAMETERS, h_thick)
    expected = fraction*a.transmitted_traction_pa+(1.-fraction)*b.transmitted_traction_pa
    expected = np.insert(expected, cell+1, b.transmitted_traction_pa[cell], axis=0)
    expected[cell] = a.transmitted_traction_pa[cell]
    np.testing.assert_allclose(traction, expected, rtol=3e-14, atol=2e-11)
    np.testing.assert_array_equal(source[cell], source[cell+1])
    assert np.linalg.norm(traction[cell+1]-traction[cell]) > 100.
    wrong_h = h.copy()
    wrong_h[cell] = fraction*0.+(1.-fraction)*25.
    wrong = prescribed_cell_basal_state(mesh, PARAMETERS, wrong_h)
    assert np.linalg.norm(wrong.transmitted_traction_pa[cell]-traction[cell]) > 100.


def test_minority_owners_move_and_contribute_without_a_raster_majority():
    mesh, state, h = world()
    parcels, lids = [], []
    for p, lid in zip(state.parcels, h):
        parcels.extend((split_parcel(p, .99), replace(split_parcel(p, .01),
                       plate=p.plate+4, material_id=f"minor:{p.material_id}")))
        lids.extend((lid, .2*lid))
    mixed = replace(state, parcels=tuple(parcels), known_material_ids=())
    result = solve(mesh, mixed, lids)
    assert result.omega_rad_per_myr.shape == (8, 3)
    assert np.all(np.linalg.norm(result.omega_rad_per_myr[4:], axis=1) > 1e-6)
    np.testing.assert_allclose(result.plate_areas_km2[4:], result.plate_areas_km2[:4]/99., rtol=3e-15)
    assert np.linalg.norm(result.omega_rad_per_myr[4:]-result.omega_rad_per_myr[:4]) > 1e-5


def test_identical_component_splitting_and_order_do_not_change_motion_or_power():
    mesh, state, h = world()
    before = solve(mesh, state, h)
    parcels, lids = [], []
    for p, lid in zip(state.parcels, h):
        for fraction in (.07, .13, .31, .49):
            parcels.append(split_parcel(p, fraction))
            lids.append(lid)
    order = np.random.default_rng(39).permutation(len(parcels))
    changed = replace(state, parcels=tuple(parcels[i] for i in order))
    after = solve(mesh, changed, np.asarray(lids)[order])
    np.testing.assert_allclose(after.omega_rad_per_myr, before.omega_rad_per_myr, rtol=3e-14, atol=1e-18)
    np.testing.assert_allclose(after.basal_driving_torque_nm, before.basal_driving_torque_nm,
                               rtol=3e-14, atol=np.linalg.norm(before.basal_driving_torque_nm)*2e-15)
    assert after.basal_source_power_w == pytest.approx(before.basal_source_power_w, rel=4e-15)
    assert after.basal_drag_dissipation_w == pytest.approx(before.basal_drag_dissipation_w, rel=4e-15)
    assert after.mean_speed_mm_per_year == pytest.approx(before.mean_speed_mm_per_year, rel=4e-15)


def test_area_weighted_component_force_equals_legacy_mean_transmission_quadrature():
    mesh, state, h = world()
    parcels, lids = [], []
    fractions = (.17, .29, .54)
    for p, lid in zip(state.parcels, h):
        for index, fraction in enumerate(fractions):
            parcels.append(replace(split_parcel(p, fraction), plate=p.plate+4*index,
                                   material_id=f"{p.material_id}:{index}"))
            lids.append(float(lid*(index+.1)))
    mixed = replace(state, parcels=tuple(parcels), known_material_ids=())
    source, transmitted = fractional_prescribed_tractions(mesh, mixed, PARAMETERS, lids)
    area = np.asarray([p.area_km2 for p in mixed.parcels]).reshape(-1, 3)
    averaged = np.sum(transmitted.reshape(-1, 3, 3)*area[:, :, None], axis=1)
    averaged /= np.asarray(state.cell_areas_km2)[:, None]
    # Invert c only for this independent legacy oracle, after averaging c(H).
    # The implementation itself never constructs a mean thickness.
    component_c = -np.expm1(-np.asarray(lids).reshape(-1, 3)/PARAMETERS.traction_coupling_depth_km)
    mean_c = component_c@np.asarray(fractions)
    effective_h = -PARAMETERS.traction_coupling_depth_km*np.log1p(-mean_c)
    oracle = prescribed_cell_basal_state(mesh, PARAMETERS, effective_h)
    np.testing.assert_allclose(averaged, oracle.transmitted_traction_pa, rtol=5e-14, atol=2e-11)
    np.testing.assert_allclose(source.reshape(-1, 3, 3)[:, 0], oracle.source_traction_pa,
                               rtol=2e-15, atol=2e-11)


def test_plate_relabeling_and_empty_labels_preserve_physical_solution():
    mesh, state, h = world()
    before = solve(mesh, state, h)
    labels = np.array([7, 2, 5, 0])
    changed = replace(state, parcels=tuple(replace(p, plate=int(labels[p.plate])) for p in state.parcels))
    after = solve(mesh, changed, h, plate_count=10)
    np.testing.assert_array_equal(after.omega_rad_per_myr[labels], before.omega_rad_per_myr)
    np.testing.assert_array_equal(after.omega_rad_per_myr[[1, 3, 4, 6, 8, 9]], 0.)
    assert after.mean_speed_mm_per_year == before.mean_speed_mm_per_year


def test_quasistatic_solution_matches_independent_velocity_fit_and_power_integral():
    mesh, state, h = world()
    result = solve(mesh, state, h)
    for pid in range(4):
        mask = np.array([p.plate == pid for p in state.parcels])
        x = mesh.centroids[mask]
        weights = np.sqrt(np.asarray(state.cell_areas_km2)[mask])
        design = np.stack([np.cross(axis, x)*RADIUS*1000. for axis in np.eye(3)], axis=2)
        target = result.parcel_transmitted_traction_pa[mask]/PARAMETERS.basal_drag_pa_s_m
        oracle = np.linalg.lstsq((design*weights[:, None, None]).reshape(-1, 3),
            (target*weights[:, None]).ravel(), rcond=None)[0]*SECONDS_PER_MYR
        np.testing.assert_allclose(result.omega_rad_per_myr[pid], oracle, rtol=4e-14, atol=1e-18)
    v, area = result.parcel_velocity_m_s, np.asarray(state.cell_areas_km2)*1e6
    independent_power = sum(float(a*np.dot(t, u)) for a, t, u in zip(area,
                            result.parcel_transmitted_traction_pa, v))
    independent_drag = sum(float(a*PARAMETERS.basal_drag_pa_s_m*np.dot(u, u)) for a, u in zip(area, v))
    assert result.basal_source_power_w == pytest.approx(independent_power, rel=2e-15)
    assert result.basal_drag_dissipation_w == pytest.approx(independent_drag, rel=2e-15)
    assert independent_drag > 0.
    assert result.torque_relative_residual < 3e-15
    assert result.power_relative_residual < 3e-15
    residual_work = np.sum(result.torque_residual_nm*result.omega_rad_per_myr)/SECONDS_PER_MYR
    assert abs(result.power_residual_w-residual_work) < 5e-15*independent_power


@pytest.mark.parametrize("zero", ["source", "lid"])
def test_zero_forcing_has_zero_motion_power_and_finite_residuals(zero):
    mesh, state, h = world()
    parameters = replace(PARAMETERS, convective_traction_pa=0.) if zero == "source" else PARAMETERS
    if zero == "lid":
        h[:] = 0.
    result = solve_fractional_basal_dynamics(mesh, state, RADIUS, parameters, h)
    np.testing.assert_array_equal(result.omega_rad_per_myr, 0.)
    assert result.mean_speed_mm_per_year == result.max_speed_mm_per_year == 0.
    assert result.basal_source_power_w == result.basal_drag_dissipation_w == 0.
    assert result.torque_relative_residual == result.power_relative_residual == 0.
    assert np.linalg.norm(result.basal_drag_tensor_nm_s) > 0.


def test_existing_source_and_drag_parameters_keep_their_independent_si_scalings():
    mesh, state, h = world()
    base = solve(mesh, state, h)
    stronger = solve_fractional_basal_dynamics(mesh, state, RADIUS,
        replace(PARAMETERS, convective_traction_pa=2.*PARAMETERS.convective_traction_pa), h)
    slower = solve_fractional_basal_dynamics(mesh, state, RADIUS,
        replace(PARAMETERS, basal_drag_pa_s_m=2.*PARAMETERS.basal_drag_pa_s_m), h)
    np.testing.assert_array_equal(stronger.basal_drag_tensor_nm_s, base.basal_drag_tensor_nm_s)
    np.testing.assert_array_equal(slower.basal_driving_torque_nm, base.basal_driving_torque_nm)
    np.testing.assert_allclose(stronger.omega_rad_per_myr, 2.*base.omega_rad_per_myr, rtol=2e-15)
    np.testing.assert_allclose(slower.omega_rad_per_myr, .5*base.omega_rad_per_myr, rtol=2e-15)
    assert stronger.basal_source_power_w == pytest.approx(4.*base.basal_source_power_w, rel=2e-15)
    assert slower.basal_drag_dissipation_w == pytest.approx(.5*base.basal_drag_dissipation_w, rel=2e-15)


def test_single_cell_owner_uses_minimum_norm_without_an_artificial_speed_floor():
    mesh, state, h = world()
    changed = replace(state, parcels=tuple(replace(p, plate=4) if p.cell == 8 else p for p in state.parcels))
    result = solve(mesh, changed, h)
    omega = result.omega_rad_per_myr[4]
    assert np.linalg.norm(omega) > 0.
    assert abs(np.dot(omega, mesh.centroids[8])) < np.linalg.norm(omega)*2e-15
    np.testing.assert_allclose(result.parcel_velocity_m_s[8],
        result.parcel_transmitted_traction_pa[8]/PARAMETERS.basal_drag_pa_s_m, rtol=2e-14, atol=1e-24)


def test_speed_diagnostics_have_mm_per_year_units_and_include_minority_area():
    mesh, state, h = world()
    result = solve(mesh, state, h)
    owner = np.array([p.plate for p in state.parcels])
    # km/Myr equals mm/year, independently avoiding SI conversions here.
    speeds = np.linalg.norm(np.cross(result.omega_rad_per_myr[owner], mesh.centroids), axis=1)*RADIUS
    expected = np.average(speeds, weights=state.cell_areas_km2)
    assert result.mean_speed_mm_per_year == pytest.approx(expected, rel=2e-15)
    assert result.max_speed_mm_per_year == pytest.approx(float(speeds.max()), rel=2e-15)


def test_explicit_thermal_field_fallback_does_not_use_chemical_plus_mantle_thickness():
    mesh, state, h = world()
    changed = replace(state, parcels=tuple(replace(p,
        material_fields=(("thermal_total_lid_thickness_km", float(lid)),)) for p, lid in zip(state.parcels, h)))
    result = solve(mesh, changed, None)
    np.testing.assert_array_equal(result.omega_rad_per_myr, solve(mesh, state, h).omega_rad_per_myr)
    with pytest.raises(ValueError, match="Explicit per-parcel"):
        solve(mesh, state, None)


@pytest.mark.parametrize("kwargs", [{"source_frame": "plate_mean"}, {"source_frame": "inertial"},
                                    {"remove_net_rotation": True}])
def test_frame_changes_cannot_silently_change_prescribed_source_slip(kwargs):
    mesh, state, h = world()
    with pytest.raises(ValueError, match="shared fixed_mesh"):
        solve(mesh, state, h, **kwargs)
    result = solve(mesh, state, h)
    assert np.linalg.norm(result.area_weighted_mean_omega_rad_per_myr) > 1e-6


@pytest.mark.parametrize("h", [1., [1.], np.nan, -1., True])
def test_invalid_or_cell_averaged_thickness_rejected(h):
    mesh, state, _ = world()
    with pytest.raises(ValueError, match="per parcel"):
        solve(mesh, state, h)


@pytest.mark.parametrize("count", [0, 3, -1, 1.5, True])
def test_plate_count_cannot_drop_a_material_owner(count):
    mesh, state, h = world()
    with pytest.raises(ValueError, match="Plate count"):
        solve(mesh, state, h, plate_count=count)


def test_mismatched_geometry_rejected_before_a_physical_solve():
    mesh, state, h = world()
    with pytest.raises(ValueError, match="capacities"):
        solve_fractional_basal_dynamics(mesh, state, RADIUS*1.01, PARAMETERS, h)
