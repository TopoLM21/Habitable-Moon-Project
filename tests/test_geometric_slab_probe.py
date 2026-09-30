"""Joint surface/slab runner contracts on a small genuine spherical geometry."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

import run_geometric_slab_probe as runner
from tectonics.fractional_surface import EXTENSIVE_FIELDS, FractionalSurfaceState, SurfaceParcel
from tectonics.geometric_boundary_io import load_boundary_checkpoint
from tectonics.geometric_contacts import extract_contacts
from tectonics.geometric_surface import from_fractional_surface, totals
from tectonics.mesh import build_icosphere
from tectonics.young_boundary import (
    SlabThermalCohort, YoungBoundaryState, YoungContact, YoungSlabSegment,
)


@pytest.fixture
def source(monkeypatch, tmp_path):
    mesh = build_icosphere(0)
    areas = mesh.physical_cell_areas_km2(100.)
    owners = (mesh.centroids[:, 0] > 0.).astype(int)
    parcels = tuple(SurfaceParcel(i, int(owner), f"material:{i}", float(area),
        2.*float(area), 10.*float(area), (1e12+i*1e10)*float(area), 5.+i,
        (("damage", i/20.),)) for i, (owner, area) in enumerate(zip(owners, areas)))
    fractional = FractionalSurfaceState(2., tuple(areas), parcels)
    source_file = tmp_path/"original.source"
    source_file.write_bytes(b"Immutable source used by the geometric slab probe")
    provenance = dict(source=str(source_file), source_sha256={str(source_file): runner.digest(source_file)},
        source_mechanics_version="young-mechanics-0.5", source_time_myr=2., origin_time_myr=2.,
        subdivisions=0, radius_km=100., newborn_crust_thickness_km=2.)
    system = SimpleNamespace(plates=tuple(SimpleNamespace(euler_axis=np.array([0., 0., 1.]),
        angular_speed_rad_per_myr=speed) for speed in (.01, -.01)))
    checkpoint = SimpleNamespace(system=system, state=object(),
        subduction_memory=SimpleNamespace(young_boundary_state=YoungBoundaryState()))
    model = SimpleNamespace(shell=SimpleNamespace(tensile_strength_pa=1.), strength_factor=np.ones(mesh.cell_count))
    config = SimpleNamespace(path=source_file, mesh=mesh, fractional=fractional,
        checkpoint=checkpoint, model=model, provenance=provenance)
    monkeypatch.setattr(runner, "load_probe_source", lambda path: (
        config.mesh, config.checkpoint, SimpleNamespace(memory=object()), config.model, dict(config.provenance)))
    monkeypatch.setattr(runner, "surface_from_lithosphere", lambda *args, **kwargs: config.fractional)
    return config


def make_tied(source):
    source.fractional = replace(source.fractional, parcels=tuple(
        replace(parcel, density_excess_mass_kg=parcel.area_km2*1e12,
                age_myr=5., specific_properties=()) for parcel in source.fractional.parcels))


def add_legacy_slab(source):
    surface = from_fractional_surface(source.mesh, source.fractional, 100.)
    geometric = extract_contacts(surface.fragments, 100.)[0]
    by_id = {fragment.fragment_id: fragment for fragment in surface.fragments}
    a, b = by_id[geometric.fragment_a].parcel.cell, by_id[geometric.fragment_b].parcel.cell
    shared = sorted(set(source.mesh.faces[a]) & set(source.mesh.faces[b]))
    key = f"{shared[0]}:{shared[1]}"
    contact = YoungContact(key, a, b, geometric.plate_a, geometric.plate_b,
        geometric.length_km, list(geometric.midpoint), [0., 0., 1.])
    segment = YoungSlabSegment("historical", key, geometric.plate_a, geometric.plate_b,
        source.mesh.centroids[a].tolist(), source.mesh.centroids[b].tolist(),
        list(geometric.midpoint), geometric.length_km,
        accepted_area_km2=10., oceanic_volume_km3=20., cold_mantle_volume_km3=100.,
        density_excess_mass_kg=1e12,
        thermal_cohorts=[SlabThermalCohort(1., 10., 20., 100., 1e12, [0., 0., 1e12], 10.)])
    inventory = YoungBoundaryState(contacts={key: contact}, segments={segment.key: segment},
        cumulative_accepted_area_km2=10., cumulative_accepted_oceanic_volume_km3=20.)
    source.checkpoint.subduction_memory.young_boundary_state = inventory
    return inventory


def test_common_motion_creates_no_cohorts_or_source_sink_and_saves_joint_state(source, tmp_path):
    output = tmp_path/"common"
    original = asdict(source.fractional)
    report = runner.execute_probe(source.path, output, .25, 1, common_omega=(.03, -.02, .01))
    assert report["status"] == "complete" and report["completed_steps"] == 1
    assert report["source_unchanged"]
    assert report["passive_boundary"]["cohort_count"] == 0
    assert report["passive_boundary"]["forces_active"] is False
    assert all(value == 0. for value in report["cumulative_losses"].values())
    assert all(value == 0. for value in report["cumulative_births"].values())
    surface, inventory, saved = load_boundary_checkpoint(output/report["checkpoint_file"])
    assert surface.time_myr == inventory.time_myr == 2.25
    assert not inventory.cohorts
    assert len(inventory.transactions) == 1
    assert saved["history"] == report["history"]
    assert asdict(source.fractional) == original
    assert runner.digest(source.path) == source.provenance["source_sha256"][str(source.path)]


def test_differential_motion_accepts_actual_losses_into_passive_cohorts_with_four_balances(source, tmp_path):
    output = tmp_path/"differential"
    report = runner.execute_probe(source.path, output, .25, 1)
    assert report["status"] == "complete"
    assert report["passive_boundary"]["cohort_count"] > 0
    assert report["cumulative_births"]["area_km2"] > 0.
    surface, inventory, saved = load_boundary_checkpoint(output/report["checkpoint_file"])
    assert len({cohort.event_id for cohort in inventory.cohorts}) == len(inventory.cohorts)
    archived = {name: math.fsum(getattr(cohort.parcel, name) for cohort in inventory.cohorts)
                for name in EXTENSIVE_FIELDS}
    remaining = totals(surface.fragments)
    for name in EXTENSIVE_FIELDS:
        assert archived[name] == pytest.approx(report["cumulative_losses"][name], rel=5e-13)
        assert remaining[name]+archived[name] == pytest.approx(
            report["initial_totals"][name]+report["cumulative_births"][name], rel=5e-12)
    assert max(abs(value) for value in report["history"][-1]["relative_residuals"].values()) < 5e-12
    assert report["passive_boundary"]["forces_active"] is False
    assert all(cohort.acceptance_time_myr == 2.25 for cohort in inventory.cohorts)


@pytest.mark.parametrize("differential", [False, True])
def test_two_step_resume_exactly_reproduces_surface_inventory_registry_and_history(source, tmp_path, differential):
    options = {} if differential else dict(common_omega=(.03, -.02, .01))
    full = runner.execute_probe(source.path, tmp_path/"full", .25, 2, **options)
    part = runner.execute_probe(source.path, tmp_path/"part", .25, 1, **options)
    resumed = runner.execute_probe(source.path, tmp_path/"resumed", .25, 1,
        resume=tmp_path/"part"/part["checkpoint_file"], **options)
    assert full["status"] == resumed["status"] == "complete"
    full_surface, full_inventory, full_saved = load_boundary_checkpoint(tmp_path/"full"/full["checkpoint_file"])
    resumed_surface, resumed_inventory, resumed_saved = load_boundary_checkpoint(tmp_path/"resumed"/resumed["checkpoint_file"])
    assert asdict(full_surface) == asdict(resumed_surface)
    assert asdict(full_inventory) == asdict(resumed_inventory)
    assert full_saved == resumed_saved
    assert full["history"] == resumed["history"]
    assert (tmp_path/"full"/full["checkpoint_file"]).read_bytes() == (tmp_path/"resumed"/resumed["checkpoint_file"]).read_bytes()


def test_physically_tied_differential_motion_stays_unresolved_without_invalid_checkpoint(source, tmp_path):
    make_tied(source)
    report = runner.execute_probe(source.path, tmp_path/"unresolved", .25, 2)
    assert report["status"] == "unresolved_polarity"
    assert report["completed_steps"] == 0
    assert report["unresolved"]["overlaps"]
    assert report["unresolved"]["last_valid_time_myr"] == 2.
    assert report["checkpoint_file"] is report["checkpoint_sha256"] is None
    assert not list((tmp_path/"unresolved").glob("*checkpoint*"))
    assert report["passive_boundary"]["cohort_count"] == 0
    assert json.loads((tmp_path/"unresolved/report.json").read_text())["status"] == "unresolved_polarity"


@pytest.mark.parametrize("change", ["dt", "motion", "source", "polarity"])
def test_resume_rejects_changed_source_motion_step_or_polarity_flag_before_output(source, tmp_path, change):
    options = dict(common_omega=(.03, -.02, .01))
    first = runner.execute_probe(source.path, tmp_path/"part", .25, 1, **options)
    if change == "motion":
        options["common_omega"] = (.02, -.02, .01)
    elif change == "source":
        source.provenance["source"] = str(tmp_path/"another.source")
    elif change == "polarity":
        options["inherit_legacy_polarity"] = True
    with pytest.raises(ValueError, match="identical source, motion, time step and polarity"):
        runner.execute_probe(source.path, tmp_path/"rejected", .5 if change == "dt" else .25, 1,
            resume=tmp_path/"part"/first["checkpoint_file"], **options)
    assert not (tmp_path/"rejected").exists()


def test_legacy_direction_requires_explicit_opt_in_and_imports_no_old_slab_material(source, tmp_path):
    inventory = add_legacy_slab(source)
    before = deepcopy(asdict(inventory))
    options = dict(common_omega=(.03, -.02, .01))
    default = runner.execute_probe(source.path, tmp_path/"default", .25, 1, **options)
    inherited = runner.execute_probe(source.path, tmp_path/"inherited", .25, 1,
                                    inherit_legacy_polarity=True, **options)
    assert default["experiment"]["inherit_legacy_polarity"] is False
    assert default["initial_polarity"] == []
    assert inherited["initial_polarity"]
    assert inherited["polarity_initialization"]["counts"]["geometry_matched_segments"] == 1
    assert inherited["polarity_initialization"]["imported_material"] is False
    assert inherited["polarity_initialization"]["imported_force"] is False
    for report in (default, inherited):
        assert report["passive_boundary"]["cohort_count"] == 0
        assert all(value == 0. for value in report["passive_boundary"]["accepted"].values())
    assert asdict(inventory) == before


def test_existing_output_never_overwrites_user_files(source, tmp_path):
    output = tmp_path/"existing"
    output.mkdir()
    (output/"sentinel").write_text("retain", encoding="utf-8")
    with pytest.raises(ValueError, match="new directory"):
        runner.execute_probe(source.path, output, .25, 1)
    assert (output/"sentinel").read_text() == "retain"


def test_cli_uses_requested_common_motion_and_reports_passive_scope(source, tmp_path, capsys):
    assert runner.main(["--source", str(source.path), "--out", str(tmp_path/"cli"),
        "--steps", "1", "--dt-myr", ".25", "--common-omega", ".03", "-.02", ".01"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "complete"
    assert report["passive_boundary"]["forces_active"] is False
