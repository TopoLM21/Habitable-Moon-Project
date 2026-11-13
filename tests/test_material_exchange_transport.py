from copy import deepcopy
from dataclasses import replace
import math

import numpy as np
import pytest

from tectonics.mesh import build_icosphere
from tectonics.plates import Plate, PlateSystem
from tectonics.lithosphere import LithosphereState, advance_lithosphere
from tectonics.transport import initialize_transport_state
from tectonics.tides import constant_eccentricity
from tectonics.continental import ContinentalCycleParameters, initialize_continental_cycle, advance_continental_cycle


def ocean(sub=1):
    mesh = build_icosphere(sub); n = mesh.cell_count
    a = mesh.physical_cell_areas_km2(5287.)
    s = LithosphereState(0., np.zeros(n, np.int32), np.zeros(n, np.int8),
                         np.ones(n), np.linspace(5., 9., n), np.zeros(n))
    s.oceanic_volume_km3 = a * s.crust_thickness_km
    return mesh, s


def system(s, speed=0.):
    return PlateSystem(s.cell_plate.copy(), (Plate(0, 0, np.array([0.,0.,1.]), speed),))


def lith_step(mesh, s, dt, speed=0., memory=None):
    return advance_lithosphere(mesh, system(s, speed), s, dt, 5287., 7.12, 47.,
                              constant_eccentricity(0.),
                              transport_state=initialize_transport_state(1) if memory is None else memory)


def test_stationary_nonuniform_ocean_is_not_reset_to_reference_thickness():
    mesh, s = ocean()
    n, _, _, d = lith_step(mesh, s, 4.)
    assert np.array_equal(n.oceanic_volume_km3, s.oceanic_volume_km3)
    assert np.allclose(n.crust_thickness_km, s.crust_thickness_km, rtol=2e-16)
    assert d.oceanic_created_volume_km3 == d.oceanic_recycled_volume_km3 == 0.


def test_whole_sphere_rotation_preserves_all_basalt_on_unequal_cells():
    mesh, s = ocean(2)
    n, _, _, d = lith_step(mesh, s, 20., speed=.04)
    assert sorted(n.oceanic_volume_km3) == sorted(s.oceanic_volume_km3)
    assert d.oceanic_created_volume_km3 == d.oceanic_recycled_volume_km3 == 0.


def test_rigid_mixed_coast_rotation_cannot_create_basalt_from_raster_area_difference():
    mesh,s=ocean(2);a=mesh.physical_cell_areas_km2(5287.)
    continent=mesh.centroids[:,0]>.2
    s.crust_type=continent.astype(np.int8)
    s.crust_thickness_km=np.where(continent,35.,7.)
    s.continental_fraction=continent.astype(float)
    s.continental_volume_km3=a*s.continental_fraction*s.crust_thickness_km
    s.oceanic_volume_km3=a*(1-s.continental_fraction)*7
    n,_,_,d=lith_step(mesh,s,20.,speed=.04)
    assert math.fsum(n.oceanic_volume_km3)==math.fsum(s.oceanic_volume_km3)
    assert d.oceanic_created_volume_km3==d.oceanic_recycled_volume_km3==0.
    # The raster's imperfect coastal footprint is exposed, not filled for free.
    assert d.oceanic_unfilled_footprint_km2>0.


def test_moving_two_plates_have_explicit_created_and_returned_volumes():
    mesh, s = ocean(2)
    s.cell_plate[mesh.centroids[:,0] < 0] = 1
    sys = PlateSystem(s.cell_plate.copy(), (Plate(0,0,np.array([0.,0.,1.]),.03),
                                         Plate(1,1,np.array([0.,0.,1.]),-.03)))
    n, _, _, d = advance_lithosphere(mesh, sys, s, 20., 5287., 7.12, 47., constant_eccentricity(0.),
                                    transport_state=initialize_transport_state(2))
    assert d.oceanic_created_volume_km3 > 0
    assert d.oceanic_recycled_volume_km3 > 0
    residual = math.fsum([*n.oceanic_volume_km3, *(-s.oceanic_volume_km3),
                         -d.oceanic_created_volume_km3, d.oceanic_recycled_volume_km3])
    assert abs(residual) < 1e-6


def test_frozen_maturation_matches_exact_solution_for_all_substeps():
    mesh, s = ocean()
    forcing = np.full(mesh.cell_count, .5)
    p = ContinentalCycleParameters(analytic_maturation=True)
    expected = .5*p.arc_maturation_rate_per_myr*p.arc_potential_decay_myr*(-np.expm1(-40/p.arc_potential_decay_myr))
    for dt in (4., 2., 1.):
        st = deepcopy(s); cycle = initialize_continental_cycle(mesh)
        for _ in range(int(40/dt)):
            st.time_myr += dt
            st, cycle, _ = advance_continental_cycle(mesh, st, [], cycle, dt, 5287., p, volcanic_arc_forcing=forcing)
        assert cycle.felsic_potential == pytest.approx(expected, rel=1e-14)


def test_juvenile_event_time_and_post_event_thickening_are_step_independent():
    mesh, s = ocean()
    p = ContinentalCycleParameters(analytic_maturation=True)
    rate, tau = p.arc_maturation_rate_per_myr, p.arc_potential_decay_myr
    event = -tau*math.log((rate*tau-p.juvenile_threshold)/(rate*tau-.95))
    expected_h = p.juvenile_continental_thickness_km + p.continental_arc_thickening_km_per_myr*(8-event)
    forcing = np.ones(mesh.cell_count)
    for dt in (4., 2., 1.):
        st=deepcopy(s); cycle=initialize_continental_cycle(mesh); cycle.felsic_potential[:]=.95
        events=[]; ocean_return=0.
        for _ in range(int(8/dt)):
            st.time_myr += dt
            st, cycle, d = advance_continental_cycle(mesh, st, [], cycle, dt, 5287., p, volcanic_arc_forcing=forcing)
            if d.first_juvenile_event_time_myr is not None: events.append(d.first_juvenile_event_time_myr)
            ocean_return += d.oceanic_recycled_volume_km3
        assert events == pytest.approx([event], rel=1e-13)
        assert st.crust_thickness_km == pytest.approx(expected_h, rel=1e-14)
        assert np.all(st.oceanic_volume_km3 == 0)
        assert ocean_return == pytest.approx(float(np.sum(s.oceanic_volume_km3)), rel=1e-15)


def test_analytic_mode_rejects_large_increment_instead_of_clipping_production():
    mesh, s = ocean()
    s.crust_type[:] = 1; s.crust_thickness_km[:] = 35.; s.oceanic_volume_km3[:] = 0
    with pytest.raises(ValueError, match='reduce timestep'):
        advance_continental_cycle(mesh, s, [], initialize_continental_cycle(mesh), 40., 5287.,
            ContinentalCycleParameters(analytic_maturation=True), volcanic_arc_forcing=np.ones(mesh.cell_count))


def test_explicit_inventory_cannot_use_nonconservative_remap():
    mesh,s = ocean()
    with pytest.raises(ValueError, match='requires conservative'):
        advance_lithosphere(mesh, system(s), s, 1., 5287., 7.12, 47., constant_eccentricity(0.))
