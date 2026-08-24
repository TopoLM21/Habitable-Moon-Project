import numpy as np

from tectonics.lithosphere import CrustType, advance_lithosphere, initialize_lithosphere
from tectonics.mesh import build_icosphere, connected_components
from tectonics.plates import Plate, PlateSystem, random_plate_system
from tectonics.tides import constant_eccentricity
from tectonics.topology import PlateTopologyParameters, _attempt_split


def test_tidal_damage_without_real_extension_cannot_break_continent():
    mesh = build_icosphere(2)
    system = random_plate_system(mesh, 1, 1234, 0.0, 0.3, 0.3)
    state = initialize_lithosphere(mesh, system, continental_fraction=0.45, continental_nuclei=1)
    state.tidal_damage[:] = 1.0
    before = state.crust_type.copy()
    out, _, _, diag = advance_lithosphere(
        mesh, system, state, 4.0, 5287.0, 7.12, 47.0, constant_eccentricity(0.00047),
        primary_mass_jupiter=5.0,
        continental_extension_min_duration_myr=4.0,
        continental_rift_extension_threshold=0.01,
        continental_min_breakup_thickness_km=34.9,
        continental_thinning_km_per_myr=10.0,
    )
    assert np.array_equal(out.crust_type, before)
    assert diag.tidally_rifted_continental_area_km2 == 0.0
    assert np.allclose(out.rift_extension, 0.0)
    assert np.allclose(out.extension_age_myr, 0.0)


def test_topology_split_uses_postrift_extension_marker():
    mesh = build_icosphere(3)
    owner = np.zeros(mesh.cell_count, dtype=np.int32)
    system = PlateSystem(owner.copy(), (Plate(0, 0, np.array([0., 0., 1.]), np.deg2rad(0.25)),))
    state = initialize_lithosphere(mesh, system, continental_fraction=0.8, continental_nuclei=1)
    # Closed young oceanic band across a previously continental parent. Damage is
    # zero: v0.9.2 should recognize the explicit extension memory instead.
    cut = np.abs(mesh.centroids[:, 0]) < 0.12
    state.crust_type[cut] = int(CrustType.OCEANIC)
    state.crust_age_myr[cut] = 2.0
    state.crust_thickness_km[cut] = 7.0
    state.tidal_damage[cut] = 0.0
    state.rift_extension[cut] = 1.0
    params = PlateTopologyParameters(
        split_min_rift_cells=6,
        split_min_child_cells=100,
        split_differential_speed_deg_per_myr=0.05,
        split_min_postrift_extension=0.7,
    )
    out, event = _attempt_split(mesh, state, system, params, 5287.0)
    assert out is not None and event is not None
    assert event.kind == 'split'
    assert len(out.plates) == 2
    for pid in range(2):
        cells = np.flatnonzero(out.cell_plate == pid)
        assert len(cells) >= 100
        assert len(connected_components(cells, mesh.neighbors)) == 1


def test_passive_margin_gap_does_not_keep_eating_single_continental_flank():
    """A gap beside only one continental plate is oceanic spreading, not a new continental rift."""
    mesh = build_icosphere(2)
    # Two moving plates; make only plate 0 continental near the contact and plate 1 oceanic.
    owner = (mesh.centroids[:, 0] >= 0.0).astype(np.int32)
    plates = (
        Plate(0, 0, np.array([0., 0., 1.]), np.deg2rad(-0.5)),
        Plate(1, 1, np.array([0., 0., 1.]), np.deg2rad(0.5)),
    )
    system = PlateSystem(owner.copy(), plates)
    state = initialize_lithosphere(mesh, system, continental_fraction=0.0, continental_nuclei=1)
    state.cell_plate[:] = owner
    state.crust_type[:] = int(CrustType.OCEANIC)
    state.crust_thickness_km[:] = 7.0
    # Continentalize only a narrow band on plate 0 side of the contact.
    band = (owner == 0) & (np.abs(mesh.centroids[:, 0]) < 0.35)
    state.crust_type[band] = int(CrustType.CONTINENTAL)
    state.crust_thickness_km[band] = 35.0
    before = state.crust_thickness_km.copy()
    out, _, _, diag = advance_lithosphere(
        mesh, system, state, 4.0, 5287.0, 7.12, 47.0, constant_eccentricity(0.00047),
        primary_mass_jupiter=5.0,
        continental_extension_min_duration_myr=4.0,
        continental_rift_extension_threshold=0.01,
        continental_min_breakup_thickness_km=34.9,
        continental_thinning_km_per_myr=10.0,
        continental_extension_requires_two_plate_flanks=True,
    )
    # If numerical advection produced opening gaps, none may drive a unilateral
    # passive-margin breakup front into this single continental flank.
    assert diag.tidally_rifted_continental_area_km2 == 0.0
    assert np.all(out.crust_thickness_km[band] >= before[band] - 1e-12)
