import numpy as np

from tectonics.dynamics import DynamicsParameters, plate_ridge_push_factors, mantle_lithosphere_ridge_gpe_proxy
from tectonics.lithosphere import CrustType, LithosphereState
from tectonics.mesh import build_icosphere


def _ocean_state(mesh, split=True):
    n = mesh.cell_count
    pids = np.zeros(n, dtype=np.int32)
    if split:
        pids[n // 2:] = 1
    return LithosphereState(
        time_myr=0.0,
        cell_plate=pids,
        crust_type=np.full(n, int(CrustType.OCEANIC), dtype=np.int8),
        crust_age_myr=np.full(n, 80.0),
        crust_thickness_km=np.full(n, 7.0),
        tidal_damage=np.zeros(n, dtype=float),
        continental_fraction=np.zeros(n, dtype=float),
        continental_volume_km3=np.zeros(n, dtype=float),
    )


def test_ridge_gpe_proxy_scales_as_density_times_thickness_squared():
    mesh = build_icosphere(1)
    st = _ocean_state(mesh, split=False)
    n = mesh.cell_count
    st.mantle_lithosphere_thickness_km = np.linspace(10.0, 20.0, n)
    st.mantle_lithosphere_density_anomaly_kg_m3 = np.full(n, 60.0)
    p = mantle_lithosphere_ridge_gpe_proxy(st)
    assert np.isclose(p[-1] / p[0], 4.0)


def test_older_thicker_oceanic_plate_gets_stronger_ridge_push():
    mesh = build_icosphere(1)
    st = _ocean_state(mesh, split=True)
    n = mesh.cell_count
    st.mantle_lithosphere_thickness_km = np.where(st.cell_plate == 0, 45.0, 105.0)
    st.mantle_lithosphere_density_anomaly_kg_m3 = np.full(n, 64.0)
    p = DynamicsParameters(ridge_gpe_min_factor=0.01, ridge_gpe_max_factor=10.0)
    f = plate_ridge_push_factors(mesh, st, 5287.0, 2, p)
    assert f[1] > f[0] > 0.0


def test_ridge_push_reference_normalization_is_order_unity_for_reference_flank():
    mesh = build_icosphere(1)
    st = _ocean_state(mesh, split=False)
    n = mesh.cell_count
    # A linearly growing flank has mean H^2 about half the end-member H^2;
    # the implementation multiplies the mean GPE by two.
    x = np.linspace(0.0, 1.0, n)
    href = 93.5
    st.mantle_lithosphere_thickness_km = href * np.sqrt(x)
    st.mantle_lithosphere_density_anomaly_kg_m3 = np.full(n, 64.0)
    p = DynamicsParameters(ridge_gpe_exponent=1.0, ridge_gpe_calibration_gain=1.0,
                           ridge_gpe_min_factor=0.01, ridge_gpe_max_factor=10.0)
    f = plate_ridge_push_factors(mesh, st, 5287.0, 1, p)
    assert 0.95 < f[0] < 1.05


def test_legacy_state_without_explicit_mantle_lithosphere_keeps_old_ridge_push_factor():
    mesh = build_icosphere(1)
    st = _ocean_state(mesh, split=True)
    p = DynamicsParameters()
    f = plate_ridge_push_factors(mesh, st, 5287.0, 2, p)
    assert np.array_equal(f, np.ones(2))


def test_area_integrated_ridge_push_factor_is_resolution_stable():
    vals=[]
    p=DynamicsParameters(ridge_gpe_saturation_ratio=0.2,ridge_gpe_calibration_gain=1.0,
                         ridge_gpe_min_factor=0.01,ridge_gpe_max_factor=10.0)
    for sub in (2,3,4):
        mesh=build_icosphere(sub)
        st=_ocean_state(mesh, split=False)
        # Smooth physical field sampled on each mesh; area-weighted integration
        # should converge instead of scaling with cell count.
        z=np.asarray(mesh.centroids[:,2],dtype=float)
        st.mantle_lithosphere_thickness_km=75.0+18.0*z
        st.mantle_lithosphere_density_anomaly_kg_m3=60.0+3.0*z
        vals.append(float(plate_ridge_push_factors(mesh,st,5287.0,1,p)[0]))
    assert max(vals)-min(vals) < 0.01
