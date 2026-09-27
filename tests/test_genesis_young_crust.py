"""Young basalt survives mature transport, including mixed material cells."""
from contextlib import nullcontext
import json

import numpy as np
import pytest

from tectonics.continental import (
    ContinentalCycleParameters, advance_continental_cycle,
    initialize_continental_cycle,
)
from tectonics.cpu_runtime import CpuExecution
from tectonics.kinematics import BoundaryRecord, BoundaryType
from tectonics.lithosphere import (
    CrustType, advance_lithosphere, effective_oceanic_thickness_km,
    initialize_lithosphere,
)
from tectonics.mesh import build_icosphere
from tectonics.plates import PlateSystem, random_plate_system
from tectonics.tides import constant_eccentricity
from tectonics.transport import (
    SubgridTransportParameters, TransportDiagnostics, TransportMap,
    initialize_transport_state,
)

RADIUS = 5287.0


def _setup(subdivisions=1, plate_count=1):
    mesh = build_icosphere(subdivisions)
    system = random_plate_system(mesh, plate_count, 947, 0.0, 20.0, 20.0)
    state = initialize_lithosphere(mesh, system, 0.0, 0, radius_km=RADIUS)
    areas = mesh.physical_cell_areas_km2(RADIUS)
    state.crust_thickness_km[:] = np.linspace(2.0, 6.0, mesh.cell_count)
    state.oceanic_volume_km3 = areas * state.crust_thickness_km
    return mesh, system, state, areas


def _step(mesh, system, state, **kwargs):
    kwargs.setdefault("transport_state", initialize_transport_state(len(system.plates)))
    kwargs.setdefault("transport_parameters", SubgridTransportParameters(
        min_changed_fraction=0.0, min_p75_cell_spacing_fraction=0.0,
        forced_min_changed_fraction=0.0,
    ))
    return advance_lithosphere(
        mesh, system, state, 1.0, RADIUS, 7.12, 47.0,
        constant_eccentricity(0.0), **kwargs,
    )


def _fixed_map(monkeypatch, mesh, system, state, targets):
    """Each original parcel occurs once; tests prescribe convergence explicitly."""
    covered = np.zeros((len(system.plates), mesh.cell_count), dtype=bool)
    source = np.full(covered.shape, -1, dtype=np.int32)
    per_plate = []
    for pid in range(len(system.plates)):
        src = np.flatnonzero(state.cell_plate == pid)
        dst = np.asarray(targets, dtype=np.int32)[src]
        assert len(set(dst)) == len(dst)
        covered[pid, dst] = True
        source[pid, dst] = src
        per_plate.append(dst)

    def mapped(_mesh, _system, _state, _dt, transport_state, _params):
        return TransportMap(covered, source, tuple(per_plate), transport_state,
                            TransportDiagnostics(1, 1, 0., 0., 0., 1., 1.))

    monkeypatch.setattr("tectonics.transport.build_transport_map", mapped)


@pytest.mark.parametrize("threaded", [False, True])
def test_rigid_ocean_transport_preserves_imported_volume_on_unequal_area_mesh(threaded):
    mesh, system, state, areas = _setup()
    context = CpuExecution(cell_workers=2) if threaded else nullcontext()
    with context:
        out, _, _, diag = _step(mesh, system, state)
    src = diag.material_source_index
    assert np.any(src != np.arange(mesh.cell_count))
    assert np.all(src >= 0)
    assert np.array_equal(out.oceanic_volume_km3, state.oceanic_volume_km3[src])
    np.testing.assert_allclose(areas * out.crust_thickness_km, out.oceanic_volume_km3)
    assert np.sum(out.oceanic_volume_km3) == pytest.approx(np.sum(state.oceanic_volume_km3))
    assert abs(diag.oceanic_volume_balance_error_km3) < 1e-5
    assert diag.oceanic_created_volume_km3 == diag.oceanic_subducted_volume_km3 == 0.0


def test_legacy_none_reservoir_keeps_fixed_basalt_thickness():
    mesh, system, state, _ = _setup()
    state.oceanic_volume_km3 = None
    out, _, _, diag = _step(mesh, system, state)
    assert out.oceanic_volume_km3 is None
    np.testing.assert_array_equal(out.crust_thickness_km, 7.0)
    assert diag.oceanic_volume_balance_error_km3 == 0.0


def test_tracked_volume_requires_conservative_transport():
    mesh, system, state, _ = _setup()
    with pytest.raises(ValueError, match="requires conservative transport"):
        _step(mesh, system, state, transport_state=None)


@pytest.mark.parametrize("winner_fraction", [0.0, 0.35, 0.65])
def test_convergence_loses_actual_old_basalt_and_ridge_creates_configured_basalt(monkeypatch, winner_fraction):
    mesh, system, state, areas = _setup(0, 2)
    state.cell_plate[:] = 1
    state.cell_plate[0] = 0
    system = PlateSystem(state.cell_plate.copy(), system.plates)
    state.crust_age_myr[1] = 90.0
    state.continental_fraction[0] = winner_fraction
    state.continental_volume_km3[0] = areas[0] * winner_fraction * 32.0
    state.crust_type[0] = int(winner_fraction >= 0.5)
    state.oceanic_volume_km3[0] = areas[0] * (1.0 - winner_fraction) * 3.5
    state.crust_thickness_km[0] = 32.0 if winner_fraction >= 0.5 else 3.5
    state.oceanic_volume_km3[1] = areas[1] * 12.0
    state.crust_thickness_km[1] = 12.0
    targets = np.arange(mesh.cell_count)
    targets[1] = 0
    _fixed_map(monkeypatch, mesh, system, state, targets)
    out, _, _, diag = _step(mesh, system, state, oceanic_thickness_km=8.0)
    assert out.oceanic_volume_km3[0] == state.oceanic_volume_km3[0]
    assert out.oceanic_volume_km3[1] == areas[1] * 8.0
    assert out.crust_thickness_km[1] == 8.0
    assert diag.oceanic_subducted_volume_km3 == areas[1] * 12.0
    assert diag.oceanic_created_volume_km3 == areas[1] * 8.0
    assert abs(diag.oceanic_volume_balance_error_km3) < 1e-5


def test_mixed_parcels_keep_both_materials_when_visible_type_changes(monkeypatch):
    mesh, system, state, areas = _setup()
    small, large = int(np.argmin(areas)), int(np.argmax(areas))
    state.continental_fraction[small] = 0.51
    state.continental_fraction[large] = 0.49
    f = state.continental_fraction
    state.continental_volume_km3[:] = areas * f * 35.0
    state.oceanic_volume_km3[:] = areas * (1.0 - f) * 3.0
    state.crust_type[:] = (f >= 0.5).astype(np.int8)
    state.crust_thickness_km[:] = np.where(f >= 0.5, 35.0, 3.0)
    targets = np.arange(mesh.cell_count)
    targets[small], targets[large] = large, small
    _fixed_map(monkeypatch, mesh, system, state, targets)
    out, _, _, diag = _step(mesh, system, state)
    assert out.crust_type[large] == int(CrustType.OCEANIC)
    assert out.crust_type[small] == int(CrustType.CONTINENTAL)
    assert out.oceanic_volume_km3[small] == state.oceanic_volume_km3[large]
    assert out.oceanic_volume_km3[large] == state.oceanic_volume_km3[small]
    assert out.continental_volume_km3[small] == state.continental_volume_km3[large]
    assert out.crust_thickness_km[large] == pytest.approx(
        out.oceanic_volume_km3[large] / (areas[large] * (1.0 - out.continental_fraction[large])))
    assert out.crust_thickness_km[small] == pytest.approx(35.0)
    assert abs(diag.oceanic_volume_balance_error_km3) < 1e-5


@pytest.mark.parametrize("volume,fraction", [(1.0, 1.0), (-1.0, 0.0), (np.nan, 0.0)])
def test_unrepresentable_ocean_volume_is_rejected(volume, fraction):
    with pytest.raises(ValueError):
        effective_oceanic_thickness_km(np.array([fraction]), np.array([volume]), np.array([1.0]))


def test_continental_spreading_cannot_silently_delete_remaining_basalt(monkeypatch):
    mesh, system, state, areas = _setup(0, 2)
    state.cell_plate[:] = 1
    state.cell_plate[0] = 0
    system = PlateSystem(state.cell_plate.copy(), system.plates)
    state.continental_fraction[:2] = 0.65
    state.continental_volume_km3[:2] = areas[:2] * 0.65 * 35.0
    state.oceanic_volume_km3[:2] = areas[:2] * 0.35 * 3.0
    state.crust_type[:2] = int(CrustType.CONTINENTAL)
    state.crust_thickness_km[:2] = 35.0
    targets = np.arange(mesh.cell_count)
    targets[1] = 0
    _fixed_map(monkeypatch, mesh, system, state, targets)
    with pytest.raises(ValueError, match="no remaining oceanic footprint"):
        _step(mesh, system, state)


def test_rift_replacement_records_old_mixed_basalt_and_new_full_column(monkeypatch):
    mesh, system, state, areas = _setup()
    state.crust_type[1] = int(CrustType.CONTINENTAL)
    state.crust_thickness_km[1] = 19.0
    state.continental_fraction[1] = 0.65
    state.continental_volume_km3[1] = areas[1] * 0.65 * 19.0
    state.oceanic_volume_km3[1] = areas[1] * 0.35 * 3.0
    state.rift_extension[1] = 1.0
    state.extension_age_myr[1] = 100.0
    _fixed_map(monkeypatch, mesh, system, state, np.arange(mesh.cell_count))
    forcing = np.zeros(mesh.cell_count)
    forcing[1] = 1.0
    out, _, _, diag = _step(mesh, system, state,
                           continental_extension_external_forcing=forcing,
                           oceanic_thickness_km=8.0)
    assert out.crust_type[1] == int(CrustType.OCEANIC)
    assert out.continental_volume_km3[1] == out.continental_fraction[1] == 0.0
    assert out.oceanic_volume_km3[1] == areas[1] * 8.0
    assert diag.oceanic_rift_recycled_volume_km3 == state.oceanic_volume_km3[1]
    assert diag.oceanic_created_volume_km3 == areas[1] * 8.0
    assert abs(diag.oceanic_volume_balance_error_km3) < 1e-5


def test_checkpoint_preserves_optional_basalt_and_rejects_invalid_payload(tmp_path):
    from tectonics.checkpoint import RunCheckpoint, load_checkpoint, save_checkpoint
    from tectonics.thermal import ThermalParameters, initialize_thermal_state
    from tectonics.topography import TopographyParameters, initialize_topography
    from tectonics.topology import PlateTopologyManager, PlateTopologyParameters

    mesh, system, state, _ = _setup()
    manager = PlateTopologyManager(PlateTopologyParameters())
    cp = RunCheckpoint(
        state, initialize_continental_cycle(mesh),
        initialize_thermal_state(0.5, RADIUS, 7.12, ThermalParameters()),
        initialize_topography(mesh, state, [], TopographyParameters()),
        system, system, manager, 0.0, 0.0, [], [], [], [], [], [],
    )
    folder = tmp_path / "young"
    save_checkpoint(folder, cp)
    assert json.loads((folder / "meta.json").read_text())["version"] == "0.31-oceanic-volume"
    loaded = load_checkpoint(folder, manager)
    np.testing.assert_array_equal(loaded.state.oceanic_volume_km3, state.oceanic_volume_km3)
    with np.load(folder / "state.npz", allow_pickle=False) as data:
        fields = {key: data[key].copy() for key in data.files}
    fields["oceanic_volume_km3"][0] = -1.0
    np.savez_compressed(folder / "state.npz", **fields)
    with pytest.raises(ValueError, match="invalid checkpoint oceanic_volume"):
        load_checkpoint(folder, manager)
    del fields["oceanic_volume_km3"]
    np.savez_compressed(folder / "state.npz", **fields)
    with pytest.raises(ValueError, match="missing oceanic_volume_km3"):
        load_checkpoint(folder, manager)
    state.oceanic_volume_km3 = None
    save_checkpoint(folder, cp)
    assert json.loads((folder / "meta.json").read_text())["version"] == "0.11-material"
    assert load_checkpoint(folder, manager).state.oceanic_volume_km3 is None


def _boundary(mesh, a, b):
    midpoint = mesh.centroids[a] + mesh.centroids[b]
    midpoint /= np.linalg.norm(midpoint)
    return BoundaryRecord(a, b, 0, 1, 0, 1, midpoint, -50.0, 0.0, 50.0, BoundaryType.CONVERGENT)


def test_juvenile_continent_replacement_accounts_for_actual_basalt():
    mesh, _, state, _ = _setup()
    cycle = initialize_continental_cycle(mesh)
    cycle.felsic_potential[1] = 2.0
    before = state.oceanic_volume_km3.copy()
    out, _, diag = advance_continental_cycle(
        mesh, state, [], cycle, 1.0, RADIUS, ContinentalCycleParameters())
    assert out.crust_type[1] == int(CrustType.CONTINENTAL)
    assert out.oceanic_volume_km3[1] == 0.0
    assert diag.oceanic_replaced_by_juvenile_volume_km3 == before[1]
    assert abs(diag.oceanic_volume_balance_error_km3) < 1e-5
    np.testing.assert_array_equal(state.oceanic_volume_km3, before)
    np.testing.assert_array_equal(out.oceanic_volume_km3[2:], before[2:])


def test_mixed_continent_erosion_replacement_accounts_for_both_basalt_fluxes():
    mesh, _, state, areas = _setup()
    state.crust_type[1] = int(CrustType.CONTINENTAL)
    state.crust_thickness_km[1] = 21.2
    state.continental_fraction[1] = 0.65
    state.continental_volume_km3[1] = areas[1] * 0.65 * 21.2
    state.oceanic_volume_km3[1] = areas[1] * 0.35 * 3.0
    state.tidal_damage[1] = 0.9
    params = ContinentalCycleParameters(
        continental_arc_thickening_km_per_myr=0.0,
        subduction_erosion_km_per_myr=1.0,
        subduction_erosion_damage_threshold=0.1,
    )
    out, _, diag = advance_continental_cycle(
        mesh, state, [_boundary(mesh, 0, 1)], initialize_continental_cycle(mesh),
        4.0, RADIUS, params, oceanic_thickness_km=8.0)
    assert out.crust_type[1] == int(CrustType.OCEANIC)
    assert diag.oceanic_recycled_volume_km3 == state.oceanic_volume_km3[1]
    assert diag.oceanic_generated_volume_km3 == areas[1] * 8.0
    assert out.oceanic_volume_km3[1] == areas[1] * 8.0
    assert diag.subduction_erosion_volume_km3 == pytest.approx(state.continental_volume_km3[1])
    assert abs(diag.oceanic_volume_balance_error_km3) < 1e-5


def test_continental_delamination_preserves_hidden_basalt():
    mesh, _, state, areas = _setup()
    state.crust_type[1] = int(CrustType.CONTINENTAL)
    state.crust_thickness_km[1] = 70.0
    state.continental_fraction[1] = 0.65
    state.continental_volume_km3[1] = areas[1] * 0.65 * 70.0
    state.oceanic_volume_km3[1] = areas[1] * 0.35 * 3.0
    out, _, diag = advance_continental_cycle(
        mesh, state, [], initialize_continental_cycle(mesh),
        4.0, RADIUS, ContinentalCycleParameters())
    np.testing.assert_array_equal(out.oceanic_volume_km3, state.oceanic_volume_km3)
    assert diag.delaminated_volume_km3 > 0.0
    assert np.sum(state.continental_volume_km3 - out.continental_volume_km3) == pytest.approx(diag.delaminated_volume_km3)
    assert diag.oceanic_volume_balance_error_km3 == 0.0
