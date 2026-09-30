"""Independent geometry CLI/report/resume contracts with a small real mesh."""
from dataclasses import asdict
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

import run_geometric_contact_probe as runner
from tectonics.fractional_surface import FractionalSurfaceState, SurfaceParcel
from tectonics.geometric_surface import load_geometric_checkpoint
from tectonics.mesh import build_icosphere


@pytest.fixture
def source(monkeypatch, tmp_path):
    mesh = build_icosphere(0)
    radius = 100.
    areas = mesh.physical_cell_areas_km2(radius)
    owners = (mesh.centroids[:, 0] > 0.).astype(int)
    parcels = tuple(SurfaceParcel(i, int(owner), f"material:{i}", float(area), 2.*float(area),
        10.*float(area), 2e11*float(area), 10., ()) for i, (owner, area) in enumerate(zip(owners, areas)))
    surface = FractionalSurfaceState(2., tuple(areas), parcels)
    source_file = tmp_path/"original.source"
    source_file.write_bytes(b"The original source must remain untouched")
    provenance = dict(source=str(source_file), source_sha256={str(source_file): runner.digest(source_file)},
                      source_mechanics_version="young-mechanics-0.5", source_time_myr=2., origin_time_myr=2.,
                      subdivisions=0, radius_km=radius, newborn_crust_thickness_km=2.)
    system = SimpleNamespace(plates=tuple(SimpleNamespace(euler_axis=np.asarray([1., 0., 0.]),
                                            angular_speed_rad_per_myr=0.) for _ in range(2)))
    checkpoint = SimpleNamespace(system=system, state=object())
    model = SimpleNamespace(shell=SimpleNamespace(tensile_strength_pa=1.), strength_factor=np.ones(mesh.cell_count))
    monkeypatch.setattr(runner, "load_probe_source", lambda source: (
        mesh, checkpoint, SimpleNamespace(memory=object()), model, dict(provenance)))
    import tectonics.fractional_surface_io as surface_io
    monkeypatch.setattr(surface_io, "surface_from_lithosphere", lambda *args, **kwargs: surface)
    return source_file, mesh, provenance


def test_common_rotation_report_has_true_mixed_cells_contacts_coverage_and_intact_source(source, tmp_path):
    path, mesh, before = source
    output = tmp_path/"run"
    report = runner.execute_probe(path, output, .25, 1, common_omega=(.03, -.02, .01))
    assert report["status"] == "complete" and report["source_unchanged"]
    assert report["completed_steps"] == 1
    assert report["projection"]["mixed_plate_cells"] > 0
    assert report["projection"]["maximum_cell_relative_coverage_error"] < 2e-11
    assert report["projection"]["uncovered_cells"] == 0
    assert report["contact_count"] > 0
    assert report["total_contact_length_km"] > 0.
    assert all(value == 0. for value in report["cumulative_losses"].values())
    assert all(value == 0. for value in report["cumulative_births"].values())
    assert max(abs(value) for value in report["history"][-1]["relative_residuals"].values()) < 2e-14
    assert all(contact["normal_area_rate_km2_per_myr"] == 0. for contact in report["contacts"])
    state, saved = load_geometric_checkpoint(output/"geometric_checkpoint.json")
    assert len(state.fragments) == mesh.cell_count
    assert state.time_myr == 2.25
    assert saved["contacts"] == json.loads(json.dumps(report["contacts"]))
    assert runner.digest(path) == before["source_sha256"][str(path)]
    assert json.loads((output/"report.json").read_text(encoding="utf-8"))["status"] == "complete"


def test_geometric_resume_reproduces_uninterrupted_geometry_contacts_and_ledger(source, tmp_path):
    path, _, _ = source
    motion = (.03, -.02, .01)
    full = runner.execute_probe(path, tmp_path/"full", .25, 2, common_omega=motion)
    runner.execute_probe(path, tmp_path/"part", .25, 1, common_omega=motion)
    resumed = runner.execute_probe(path, tmp_path/"resumed", .25, 1, common_omega=motion,
                                   resume=tmp_path/"part/geometric_checkpoint.json")
    a, provenance_a = load_geometric_checkpoint(tmp_path/"full/geometric_checkpoint.json")
    b, provenance_b = load_geometric_checkpoint(tmp_path/"resumed/geometric_checkpoint.json")
    assert asdict(a) == asdict(b)
    assert provenance_a == provenance_b
    assert full["history"] == resumed["history"]
    assert full["contacts"] == resumed["contacts"]
    assert full["projection"] == resumed["projection"]


@pytest.mark.parametrize("change", ["motion", "dt", "mode"])
def test_resume_rejects_parameter_changes(source, tmp_path, change):
    path, _, _ = source
    motion = (.03, -.02, .01)
    runner.execute_probe(path, tmp_path/"part", .25, 1, common_omega=motion)
    with pytest.raises(ValueError, match="identical source, motion, timestep"):
        runner.execute_probe(path, tmp_path/"bad", .5 if change == "dt" else .25, 1,
            common_omega=None if change == "mode" else ((.04, -.02, .01) if change == "motion" else motion),
            resume=tmp_path/"part/geometric_checkpoint.json")
    assert not (tmp_path/"bad").exists()


def test_default_uses_source_plate_motion_and_transaction_path(source, tmp_path, monkeypatch):
    import tectonics.geometric_surface as geometry
    original = geometry.advance_geometric_surface
    calls = []
    def advance(*args, **kwargs):
        calls.append(np.asarray(args[2]).copy())
        return original(*args, **kwargs)
    monkeypatch.setattr(geometry, "advance_geometric_surface", advance)
    report = runner.execute_probe(source[0], tmp_path/"default", .25, 1)
    assert report["status"] == "complete"
    assert report["experiment"]["motion_mode"] == "source_plate_rotations"
    np.testing.assert_array_equal(calls, np.zeros((1, 2, 3)))


def test_unresolved_overlap_is_saved_explicitly_without_invalid_final_checkpoint(source, tmp_path, monkeypatch):
    import tectonics.geometric_surface as geometry
    from tectonics.geometric_transport import GeometricOverlap, UnresolvedPolarityError
    overlap = GeometricOverlap("a", "b", 0, 1, tuple(map(tuple, source[1].vertices[source[1].faces[0]])),
                               1., "equal physical properties")
    def unresolved(*args, **kwargs):
        raise UnresolvedPolarityError((overlap,))
    monkeypatch.setattr(geometry, "advance_geometric_surface", unresolved)
    report = runner.execute_probe(source[0], tmp_path/"unresolved", .25, 3)
    assert report["status"] == "unresolved_polarity"
    assert report["completed_steps"] == 0
    assert report["unresolved"]["last_valid_time_myr"] == 2.
    assert report["unresolved"]["overlaps"] == [asdict(overlap)]
    assert report["checkpoint_sha256"] is None
    assert (tmp_path/"unresolved/report.json").exists()
    assert not (tmp_path/"unresolved/geometric_checkpoint.json").exists()


def test_unresolved_second_step_saves_only_previous_valid_partition(source, tmp_path, monkeypatch):
    import tectonics.geometric_surface as geometry
    from tectonics.geometric_transport import GeometricOverlap, UnresolvedPolarityError
    original = geometry.advance_geometric_surface
    count = 0
    overlap = GeometricOverlap("a", "b", 0, 1, tuple(map(tuple, source[1].vertices[source[1].faces[0]])),
                               1., "equal physical properties")
    def second_fails(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise UnresolvedPolarityError((overlap,))
        return original(*args, **kwargs)
    monkeypatch.setattr(geometry, "advance_geometric_surface", second_fails)
    report = runner.execute_probe(source[0], tmp_path/"partial", .25, 3)
    assert report["status"] == "unresolved_polarity" and report["completed_steps"] == 1
    assert not (tmp_path/"partial/geometric_checkpoint.json").exists()
    state, saved = load_geometric_checkpoint(tmp_path/"partial/last_valid_geometric_checkpoint.json")
    assert state.time_myr == 2.25
    assert len(saved["history"]) == 1
    assert report["last_valid_checkpoint_sha256"]


def test_invalid_projection_is_rejected_before_checkpoint_save(source, tmp_path, monkeypatch):
    import tectonics.geometric_surface as geometry
    original = geometry.project_to_mesh
    def double_piece(*args, **kwargs):
        result = original(*args, **kwargs)
        return result+(result[0],)
    monkeypatch.setattr(geometry, "project_to_mesh", double_piece)
    with pytest.raises(ValueError, match="projection coverage or material budget"):
        runner.execute_probe(source[0], tmp_path/"bad_view", .25, 1, common_omega=(0., 0., 0.))
    assert not (tmp_path/"bad_view/geometric_checkpoint.json").exists()


@pytest.mark.parametrize("dt,steps", [(0., 1), (-1., 1), (math.nan, 1), (math.inf, 1),
                                     (.25, 0), (.25, -1), (.25, 1.5), (.25, True)])
def test_invalid_step_arguments_fail_before_output_creation(source, tmp_path, dt, steps):
    with pytest.raises(ValueError, match="positive"):
        runner.execute_probe(source[0], tmp_path/"invalid", dt, steps)
    assert not (tmp_path/"invalid").exists()


def test_existing_output_is_never_overwritten(source, tmp_path):
    output = tmp_path/"existing"
    output.mkdir()
    sentinel = output/"keep"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="never overwritten"):
        runner.execute_probe(source[0], output, .25, 1)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_cli_reports_success_and_uses_explicit_common_rotation(source, tmp_path, capsys):
    result = runner.main(["--source", str(source[0]), "--out", str(tmp_path/"cli"),
                          "--steps", "1", "--dt-myr", ".25", "--common-omega", ".03", "-.02", ".01"])
    assert result == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "complete"
    assert printed["projection"]["mixed_plate_cells"] > 0
