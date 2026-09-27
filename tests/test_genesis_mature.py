"""Explicit-state imports must not regenerate young-moon geology on resume."""
import json

import numpy as np
import pytest

from tectonics.genesis_mature import (
    build_experimental_genesis_mature_import,
    load_experimental_genesis_mature_import,
    save_experimental_genesis_mature_import,
)
from tectonics.mesh import build_icosphere
from tectonics.plates import Plate, PlateSystem
from tectonics.thermal import ThermalState
from tectonics.topology import PlateTopologyManager, PlateTopologyParameters


def _inputs():
    mesh = build_icosphere(1)
    n = mesh.cell_count
    owner = (mesh.centroids[:, 2] < 0).astype(np.int32)
    plates = tuple(
        Plate(index, int(np.flatnonzero(owner == index)[0]), np.array([0., 0., 1.]), speed)
        for index, speed in enumerate((0.001, -0.002))
    )
    fields = dict(
        radius_km=5287.0,
        system=PlateSystem(owner, plates),
        crust_age_myr=np.linspace(0.0, 0.2, n),
        crust_thickness_km=np.linspace(3.0, 6.0, n),
        tidal_damage=np.linspace(0.0, 0.4, n),
        mantle_lithosphere_thickness_km=np.linspace(10.0, 20.0, n),
        mantle_lithosphere_density_anomaly_kg_m3=np.linspace(0.0, 30.0, n),
        thermal=ThermalState(2.0, 2.0, 1550.0, 0.12, 1.0, 35.0),
        elevation_m=np.linspace(-800.0, -100.0, n),
        water_volume_km3=1126719620.5563111,
        # An independent field, deliberately unlike either plate velocity.
        mantle_cell_omega_rad_per_myr=np.tile([0.0003, -0.0005, 0.0001], (n, 1)),
        source_mass_kg=np.linspace(1e15, 2e15, n),
        source_enthalpy_j=np.linspace(1e22, 3e22, n),
        next_plume_birth_time_myr=12.0,
        source_metadata={"origin": "admissible explicit-state fixture", "coupled_clock_myr": 2.0},
    )
    return mesh, fields


def test_import_retains_explicit_fields_without_initialization():
    mesh, fields = _inputs()
    bundle = build_experimental_genesis_mature_import(mesh, **fields)
    cp = bundle.checkpoint
    for name in ("crust_age_myr", "crust_thickness_km", "tidal_damage", "mantle_lithosphere_thickness_km", "mantle_lithosphere_density_anomaly_kg_m3"):
        np.testing.assert_array_equal(getattr(cp.state, name), fields[name])
    np.testing.assert_array_equal(cp.topo.elevation_m, fields["elevation_m"])
    np.testing.assert_array_equal(cp.mantle_flow.cell_omega_rad_per_myr, fields["mantle_cell_omega_rad_per_myr"])
    np.testing.assert_array_equal(cp.system.cell_plate, fields["system"].cell_plate)
    assert not cp.state.continental_fraction.any()
    assert not cp.state.continental_volume_km3.any()
    assert not cp.state.craton_strength.any()
    assert not cp.cycle.felsic_potential.any()
    assert cp.hydrosphere.water_volume_km3 == fields["water_volume_km3"]
    assert cp.plume_state.ages_myr.size == 0
    assert cp.plume_state.next_birth_time_myr == 12.0
    for state in (cp.state, cp.cycle, cp.topo, cp.thermal, cp.mantle_flow, cp.hydrosphere,
                  cp.plume_state, cp.plume_rifting_state, cp.hotspot_track_state):
        assert state.time_myr == 2.0
    assert bundle.report["physical_handoff_certified"] is False
    fields["crust_age_myr"][:] = 99.0
    fields["system"].cell_plate[:] = 0
    fields["source_metadata"]["origin"] = "modified"
    assert cp.state.crust_age_myr.max() == 0.2
    assert len(np.unique(cp.system.cell_plate)) == 2
    assert bundle.report["source_metadata"]["origin"] != "modified"


def test_archive_roundtrip_retains_unrepresented_reservoirs(tmp_path):
    mesh, fields = _inputs()
    bundle = build_experimental_genesis_mature_import(mesh, **fields)
    root = save_experimental_genesis_mature_import(tmp_path / "import", bundle)
    loaded = load_experimental_genesis_mature_import(root)
    for name in ("source_mass_kg", "source_enthalpy_j"):
        np.testing.assert_array_equal(loaded.reservoir_arrays[name], fields[name])
    np.testing.assert_array_equal(loaded.checkpoint.state.crust_age_myr, fields["crust_age_myr"])
    assert loaded.checkpoint.hydrosphere.water_volume_km3 == fields["water_volume_km3"]
    meta = json.loads((root / "mature_checkpoint/meta.json").read_text())
    assert meta["version"] == "0.31-oceanic-volume"
    assert "not evolved" in loaded.report["unsupported_state"][0]
    with pytest.raises(FileExistsError):
        save_experimental_genesis_mature_import(root, bundle)


@pytest.mark.parametrize("target", ["source_reservoirs.npz", "mature_checkpoint/state.npz", "mature_checkpoint/meta.json"])
def test_archive_detects_corruption_before_load(tmp_path, target):
    mesh, fields = _inputs()
    root = save_experimental_genesis_mature_import(tmp_path / "import", build_experimental_genesis_mature_import(mesh, **fields))
    path = root / target
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_experimental_genesis_mature_import(root)


def test_archive_rejects_manifest_reservoir_mismatch(tmp_path):
    mesh, fields = _inputs()
    root = save_experimental_genesis_mature_import(tmp_path / "import", build_experimental_genesis_mature_import(mesh, **fields))
    path = root / "handoff.json"
    manifest = json.loads(path.read_text())
    manifest["archived_source_mass_kg"] *= 2.0
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest mismatch: archived_source_mass"):
        load_experimental_genesis_mature_import(root)


def test_full_v131_resume_wrapper_does_not_seed_plumes_or_change_ages(tmp_path, monkeypatch):
    import run_long_evolution_v131 as runner

    mesh, fields = _inputs()
    bundle = build_experimental_genesis_mature_import(mesh, **fields)
    root = save_experimental_genesis_mature_import(tmp_path / "import", bundle)
    monkeypatch.setattr(runner.v124, "_mesh", mesh)
    monkeypatch.setattr(runner.v124, "_radius_km", fields["radius_km"])

    def forbidden(*args, **kwargs):
        raise AssertionError("resume attempted to fabricate the missing initial state")

    monkeypatch.setattr(runner.v125, "initialize_mantle_plumes", forbidden)
    monkeypatch.setattr(runner.base, "initialize_mantle_flow", forbidden)
    monkeypatch.setattr(runner.base, "initialize_lithosphere", forbidden)
    monkeypatch.setattr(runner.base, "initialize_oceanic_crust_ages", forbidden)
    cp = runner._load_checkpoint_v131(root / "mature_checkpoint", PlateTopologyManager(PlateTopologyParameters()))
    assert runner.v125._plume_state.ages_myr.size == 0
    assert runner.v125._plume_state.next_birth_time_myr == 12.0
    assert runner._mantle_flow is cp.mantle_flow
    np.testing.assert_array_equal(cp.state.crust_age_myr, fields["crust_age_myr"])
    np.testing.assert_array_equal(cp.state.crust_thickness_km, fields["crust_thickness_km"])
    np.testing.assert_array_equal(cp.mantle_flow.cell_omega_rad_per_myr, fields["mantle_cell_omega_rad_per_myr"])
    assert cp.hydrosphere.water_volume_km3 == fields["water_volume_km3"]


def test_explicit_continent_material_is_retained():
    mesh, fields = _inputs()
    fraction = np.zeros(mesh.cell_count)
    fraction[:4] = 0.6
    volume = mesh.physical_cell_areas_km2(fields["radius_km"]) * fields["crust_thickness_km"] * fraction
    cp = build_experimental_genesis_mature_import(mesh, **fields, continental_fraction=fraction, continental_volume_km3=volume).checkpoint
    np.testing.assert_array_equal(cp.state.continental_fraction, fraction)
    np.testing.assert_array_equal(cp.state.continental_volume_km3, volume)
    assert cp.initial_continental_volume_km3 == volume.sum()
    assert not cp.cycle.felsic_potential.any()


def test_mixed_crust_converts_total_thickness_to_visible_endmember(tmp_path):
    mesh, fields = _inputs()
    n = mesh.cell_count
    areas = mesh.physical_cell_areas_km2(fields["radius_km"])
    fields["crust_thickness_km"] = np.full(n, 20.0)
    fraction = np.full(n, 0.6)
    # 60% footprint at 30 km felsic thickness + 40% at 5 km basalt = 20 km mean.
    volume = areas * 0.6 * 30.0
    bundle = build_experimental_genesis_mature_import(mesh, **fields, continental_fraction=fraction, continental_volume_km3=volume)
    cp = bundle.checkpoint
    np.testing.assert_allclose(cp.state.crust_thickness_km, 30.0, rtol=1e-15)
    np.testing.assert_allclose(cp.state.oceanic_volume_km3, areas * 0.4 * 5.0, rtol=3e-15)
    np.testing.assert_allclose(cp.state.oceanic_volume_km3 + cp.state.continental_volume_km3, areas * 20.0, rtol=1e-15)
    root = save_experimental_genesis_mature_import(tmp_path / "mixed", bundle)
    loaded = load_experimental_genesis_mature_import(root)
    np.testing.assert_array_equal(loaded.checkpoint.state.oceanic_volume_km3, cp.state.oceanic_volume_km3)


def test_first_mature_step_keeps_imported_young_oceanic_crust(tmp_path):
    from tectonics.lithosphere import advance_lithosphere
    from tectonics.tides import constant_eccentricity
    from tectonics.transport import SubgridTransportParameters

    mesh, fields = _inputs()
    root = save_experimental_genesis_mature_import(tmp_path / "import", build_experimental_genesis_mature_import(mesh, **fields))
    cp = load_experimental_genesis_mature_import(root).checkpoint
    dt = 1e-4
    state, _, _, diagnostic = advance_lithosphere(
        mesh, cp.system, cp.state, dt, fields["radius_km"], 7.12, 47.136,
        constant_eccentricity(0.0002), oceanic_thickness_km=7.0,
        transport_state=cp.transport_state, transport_parameters=SubgridTransportParameters(),
    )
    assert diagnostic.created_oceanic_area_km2 == 0.0
    assert diagnostic.subducted_oceanic_area_km2 == 0.0
    np.testing.assert_array_equal(state.oceanic_volume_km3, cp.state.oceanic_volume_km3)
    np.testing.assert_allclose(state.crust_thickness_km, fields["crust_thickness_km"], rtol=2e-15)
    np.testing.assert_allclose(state.crust_age_myr, fields["crust_age_myr"] + dt, rtol=2e-15)


@pytest.mark.parametrize("field,value", [
    ("crust_age_myr", -1.0), ("crust_age_myr", 3.0),
    ("crust_thickness_km", 0.0), ("tidal_damage", 1.1),
    ("mantle_lithosphere_thickness_km", -1.0), ("source_mass_kg", -1.0),
    ("source_enthalpy_j", float("nan")),
])
def test_invalid_imported_fields_rejected(field, value):
    mesh, fields = _inputs()
    fields[field][0] = value
    with pytest.raises(ValueError):
        build_experimental_genesis_mature_import(mesh, **fields)


def test_moved_or_open_contact_geometry_is_rejected():
    mesh, fields = _inputs()
    mesh.vertices[0, 0] += 1e-10
    with pytest.raises(ValueError, match="canonical icosphere"):
        build_experimental_genesis_mature_import(mesh, **fields)
    mesh, fields = _inputs()
    mesh.shared_edges = mesh.shared_edges[:-1]
    with pytest.raises(ValueError, match="closed canonical"):
        build_experimental_genesis_mature_import(mesh, **fields)


def test_disconnected_plate_and_noncompact_labels_rejected():
    mesh, fields = _inputs()
    owner = np.ones(mesh.cell_count, dtype=np.int32)
    owner[0] = 0
    other = int(np.argmin(mesh.centroids @ mesh.centroids[0]))
    owner[other] = 0
    fields["system"].cell_plate = owner
    fields["system"].plates[0].seed_cell = 0
    with pytest.raises(ValueError, match="connected"):
        build_experimental_genesis_mature_import(mesh, **fields)
    fields["system"].cell_plate += 1
    with pytest.raises(ValueError, match="compact"):
        build_experimental_genesis_mature_import(mesh, **fields)


def test_incomplete_continental_material_rejected():
    mesh, fields = _inputs()
    with pytest.raises(ValueError, match="supplied together"):
        build_experimental_genesis_mature_import(mesh, **fields, continental_fraction=np.ones(mesh.cell_count))
    with pytest.raises(ValueError, match="consistently present"):
        build_experimental_genesis_mature_import(mesh, **fields, continental_fraction=np.ones(mesh.cell_count), continental_volume_km3=np.zeros(mesh.cell_count))
