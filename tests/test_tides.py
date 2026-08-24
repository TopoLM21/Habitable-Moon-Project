import numpy as np

from tectonics.tides import (
    constant_eccentricity,
    radial_displacement_amplitude_m,
    semi_major_axis_from_period,
    tidal_strain_amplitude,
)


def test_constant_eccentricity_history() -> None:
    h = constant_eccentricity(0.00047)
    assert h.at(0.0) == 0.00047
    assert h.at(123.0) == 0.00047


def test_zero_eccentricity_gives_zero_cyclic_strain() -> None:
    pts = np.asarray([[1.0,0.0,0.0],[0.0,1.0,0.0]], dtype=float)
    strain = tidal_strain_amplitude(pts, 0.0, 5287.0, 7.12, 47.0)
    assert np.all(strain == 0.0)


def test_tidal_strain_scales_linearly_with_eccentricity() -> None:
    pts = np.asarray([[1.0,0.0,0.0],[0.0,1.0,0.0]], dtype=float)
    a = tidal_strain_amplitude(pts, 0.0002, 5287.0, 7.12, 47.0)
    b = tidal_strain_amplitude(pts, 0.0004, 5287.0, 7.12, 47.0)
    assert np.allclose(b, 2.0*a)


def test_canonical_radial_displacement_is_meter_scale() -> None:
    pts = np.asarray([[1.0,0.0,0.0]], dtype=float)
    strain = tidal_strain_amplitude(pts, 0.00047, 5287.0, 7.12, 47.0, love_h2=0.6)
    disp = radial_displacement_amplitude_m(strain, 5287.0)
    assert 1.0 < float(disp[0]) < 10.0


def test_fixed_period_tidal_strain_is_nearly_primary_mass_independent():
    from tectonics.tides import tidal_strain_amplitude
    pts=np.asarray([[1.0,0.0,0.0],[0.0,1.0,0.0]],dtype=float)
    a=tidal_strain_amplitude(pts,0.00047,5287.0,7.12,47.0,primary_mass_jupiter=1.0,love_h2=0.6)
    b=tidal_strain_amplitude(pts,0.00047,5287.0,7.12,47.0,primary_mass_jupiter=5.0,love_h2=0.6)
    assert np.allclose(a,b,rtol=1e-12,atol=0.0)


def test_five_jupiter_mass_primary_has_larger_semimajor_axis_at_same_period():
    from tectonics.tides import semi_major_axis_from_period, M_JUPITER
    a1=semi_major_axis_from_period(1.0*M_JUPITER,47.0)
    a5=semi_major_axis_from_period(5.0*M_JUPITER,47.0)
    assert a5 > a1
    assert np.isclose(a5/a1,5.0**(1.0/3.0),rtol=1e-12)
