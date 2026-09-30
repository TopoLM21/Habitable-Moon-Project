"""One-step fault-loading runner: atomicity, conservation and fresh snapshots."""
from dataclasses import asdict, replace
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

import run_geometric_initiation_probe as runner
from tectonics.fractional_surface import EXTENSIVE_FIELDS, FractionalSurfaceState, SurfaceParcel
from tectonics.geometric_boundary_io import load_boundary_checkpoint, save_boundary_checkpoint
from tectonics.geometric_contacts import extract_contacts
from tectonics.geometric_initiation import OrientedFault
from tectonics.geometric_initiation_adapter import fault_snapshot_to_dict
from tectonics.geometric_surface import from_fractional_surface, rotate_surface, totals
from tectonics.mesh import SphereMesh, build_icosphere
from tectonics.young_boundary import YoungBoundaryState


@pytest.fixture
def source(monkeypatch, tmp_path):
    mesh = build_icosphere(0)
    areas = mesh.physical_cell_areas_km2(100.)
    owners = (mesh.centroids[:, 0] > 0.).astype(int)
    fractional = FractionalSurfaceState(2., tuple(areas), tuple(
        SurfaceParcel(i, int(owner), f"material:{i}", float(area), 2.*float(area),
            10.*float(area), (1e12+i*1e10)*float(area), 5.+i, (("damage", i/20.),))
        for i, (owner, area) in enumerate(zip(owners, areas))))
    path = tmp_path/"immutable.source"
    path.write_bytes(b"Forced-underthrust runner source fixture")
    provenance = dict(source=str(path), source_sha256={str(path): runner.digest(path)},
        source_mechanics_version="young-mechanics-0.5", source_time_myr=2., origin_time_myr=2.,
        subdivisions=0, radius_km=100., newborn_crust_thickness_km=2.)
    system = SimpleNamespace(plates=tuple(SimpleNamespace(euler_axis=np.array([0., 0., 1.]),
        angular_speed_rad_per_myr=speed) for speed in (.01, -.01)))
    checkpoint = SimpleNamespace(system=system, state=object(),
        subduction_memory=SimpleNamespace(young_boundary_state=YoungBoundaryState()))
    model = SimpleNamespace(shell=SimpleNamespace(tensile_strength_pa=1.), strength_factor=np.ones(mesh.cell_count))
    config = SimpleNamespace(path=path, mesh=mesh, fractional=fractional, checkpoint=checkpoint,
                             model=model, provenance=provenance)
    monkeypatch.setattr(runner, "load_probe_source", lambda path: (
        config.mesh, config.checkpoint, SimpleNamespace(memory=object()), config.model, dict(config.provenance)))
    monkeypatch.setattr(runner, "surface_from_lithosphere", lambda *args, **kwargs: config.fractional)
    return config


def write_faults(source, path, *, state=None, strong=False, both_directions=False):
    state = state or from_fractional_surface(source.mesh, source.fractional, 100.)
    faults = []
    for contact in extract_contacts(state.fragments, 100.):
        horizontal = np.asarray(contact.normal_a_to_b)
        stress = 1e7*np.eye(3)+1e8*np.outer(horizontal, horizontal)
        directions = (0, 1) if both_directions else (0,)
        for subducting in directions:
            faults.append(OrientedFault(f"fault:{contact.contact_id}:{subducting}", contact.contact_id,
                subducting, 1-subducting, math.pi/6, tuple(map(tuple, stress)),
                1e9 if strong else 1e5, .1, "synthetic prescribed weak fault and compressive load"))
    path.write_text(json.dumps(fault_snapshot_to_dict(state, faults)), encoding="utf-8")
    return path


def meridional_contacts_source(source):
    """Eight octants; contacts converge/diverge except for isolated poles."""
    vertices = np.array([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.],
                         [0., -1., 0.], [0., 0., 1.], [0., 0., -1.]])
    faces = []
    for x in (0, 1):
        for y in (2, 3):
            for z in (4, 5):
                face = [x, y, z]
                if np.linalg.det(vertices[face]) < 0.:
                    face.reverse()
                faces.append(face)
    faces = np.asarray(faces)
    centroids = vertices[faces].mean(axis=1)
    centroids /= np.linalg.norm(centroids, axis=1)[:, None]
    mesh = SphereMesh(vertices, faces, centroids, np.full(8, math.pi/2), (), ())
    areas = mesh.physical_cell_areas_km2(100.)
    source.mesh = mesh
    source.fractional = FractionalSurfaceState(2., tuple(areas), tuple(
        SurfaceParcel(i, int(centroids[i, 0] > 0), f"octant:{i}", area,
                      2.*area, 10.*area, 1e12*area, 5., ()) for i, area in enumerate(areas)))
    source.model.strength_factor = np.ones(8)
    source.checkpoint.system.plates[0].angular_speed_rad_per_myr = 0.
    source.checkpoint.system.plates[1].euler_axis = np.array([0., 0., 1.])
    source.checkpoint.system.plates[1].angular_speed_rad_per_myr = .02


def test_loaded_weak_fault_accepts_material_without_using_buoyancy_order(source, tmp_path):
    meridional_contacts_source(source)
    # Identical density and age deliberately supply no scalar ordering.
    source.fractional = replace(source.fractional, parcels=tuple(replace(parcel,
        density_excess_mass_kg=1e12*parcel.area_km2, age_myr=5., specific_properties=())
        for parcel in source.fractional.parcels))
    snapshot = write_faults(source, tmp_path/"weak.json")
    before_source, before_snapshot = source.path.read_bytes(), snapshot.read_bytes()
    report = runner.execute_probe(source.path, tmp_path/"accepted", .1, fault_snapshot=snapshot)
    assert report["status"] == "complete" and report["completed_steps"] == 1
    state, inventory, saved = load_boundary_checkpoint(tmp_path/"accepted"/report["checkpoint_file"])
    assert inventory.cohorts and report["cumulative_losses"]["area_km2"] > 0.
    assert all(cohort.subducting_plate == 0 and cohort.overriding_plate == 1 for cohort in inventory.cohorts)
    assert saved["history"][-1]["initiation"]["used_fault_evidence_ids"]
    assert len({cohort.event_id for cohort in inventory.cohorts}) == len(inventory.cohorts)
    retained = totals(state.fragments)
    for key in EXTENSIVE_FIELDS:
        archived = math.fsum(getattr(cohort.parcel, key) for cohort in inventory.cohorts)
        assert archived == pytest.approx(report["cumulative_losses"][key], rel=5e-13)
        assert retained[key]+archived == pytest.approx(
            report["initial_totals"][key]+report["cumulative_births"][key], rel=5e-12)
    assert source.path.read_bytes() == before_source and snapshot.read_bytes() == before_snapshot
    assert not report["passive_boundary"]["forces_active"]
    # Accepted directional memory cannot replace a fresh load on the next step.
    resumed = runner.execute_probe(source.path, tmp_path/"missing_next_load", .1,
        resume=tmp_path/"accepted"/report["checkpoint_file"])
    assert resumed["status"] == "unresolved_initiation"
    assert resumed["completed_steps"] == 0 and resumed["checkpoint_file"] is None
    assert resumed["cumulative_losses"] == report["cumulative_losses"]


def test_two_admissible_opposite_dips_remain_ambiguous(source, tmp_path):
    meridional_contacts_source(source)
    snapshot = write_faults(source, tmp_path/"opposite.json", both_directions=True)
    report = runner.execute_probe(source.path, tmp_path/"ambiguous", .1, fault_snapshot=snapshot)
    assert report["status"] == "unresolved_initiation"
    assert report["completed_steps"] == 0 and report["checkpoint_file"] is None
    assert report["unresolved"]["overlaps"]
    assert not report["passive_boundary"]["cohort_count"]


def test_default_blocks_differential_motion_without_using_age_or_buoyancy(source, tmp_path):
    before = asdict(source.fractional)
    report = runner.execute_probe(source.path, tmp_path/"blocked")
    assert report["status"] == "unresolved_initiation"
    assert report["completed_steps"] == 0 and report["requested_steps"] == 1
    assert report["unresolved"]["overlaps"]
    assert report["checkpoint_file"] is report["checkpoint_sha256"] is None
    assert not list((tmp_path/"blocked").glob("*checkpoint*"))
    assert report["passive_boundary"]["cohort_count"] == 0
    assert report["source_unchanged"] and not report["code_changed_during_run"]
    assert asdict(source.fractional) == before
    saved = json.loads((tmp_path/"blocked/report.json").read_text(encoding="utf-8"))
    assert saved["initiation"] == report["initiation"]


def test_common_rotation_needs_no_loading_and_keeps_all_four_quantities(source, tmp_path):
    report = runner.execute_probe(source.path, tmp_path/"common", common_omega=(.03, -.02, .01))
    assert report["status"] == "complete" and report["completed_steps"] == 1
    state, inventory, saved = load_boundary_checkpoint(tmp_path/"common"/report["checkpoint_file"])
    assert state.time_myr == inventory.time_myr == 2.25
    assert len(inventory.transactions) == 1 and not inventory.cohorts
    assert not report["passive_boundary"]["forces_active"]
    assert report["fault_snapshot"] is None
    assert saved["external_polarity_evidence"] == []
    for key in EXTENSIVE_FIELDS:
        assert totals(state.fragments)[key] == pytest.approx(report["initial_totals"][key], rel=5e-13)
        assert report["cumulative_losses"][key] == report["cumulative_births"][key] == 0.


def test_snapshot_loading_is_not_reused_or_serialized_as_future_stress(source, tmp_path):
    snapshot = write_faults(source, tmp_path/"faults.json")
    options = dict(common_omega=(.03, -.02, .01))
    first = runner.execute_probe(source.path, tmp_path/"first", fault_snapshot=snapshot, **options)
    path = tmp_path/"first"/first["checkpoint_file"]
    payload = path.read_text(encoding="utf-8")
    assert "effective_stress_pa" not in payload and "dip_radians" not in payload
    resumed = runner.execute_probe(source.path, tmp_path/"resumed", resume=path, **options)
    assert resumed["fault_snapshot"] is None
    assert resumed["history"][-1]["initiation"]["fault_snapshot"] is None
    with pytest.raises(ValueError, match="(?i)(snapshot|surface|time|state)"):
        runner.execute_probe(source.path, tmp_path/"stale", resume=path, fault_snapshot=snapshot, **options)
    assert not (tmp_path/"stale").exists()


def test_common_rotation_resume_is_exact_with_equivalent_fresh_snapshots(source, tmp_path):
    options = dict(common_omega=(.03, -.02, .01))
    first = runner.execute_probe(source.path, tmp_path/"first", **options)
    checkpoint = tmp_path/"first"/first["checkpoint_file"]
    state, _, _ = load_boundary_checkpoint(checkpoint)
    a = write_faults(source, tmp_path/"fresh_a.json", state=state)
    b = write_faults(source, tmp_path/"fresh_b.json", state=state)
    one = runner.execute_probe(source.path, tmp_path/"one", resume=checkpoint, fault_snapshot=a, **options)
    two = runner.execute_probe(source.path, tmp_path/"two", resume=checkpoint, fault_snapshot=b, **options)
    assert one["status"] == two["status"] == "complete"
    assert (tmp_path/"one"/one["checkpoint_file"]).read_bytes() == (tmp_path/"two"/two["checkpoint_file"]).read_bytes()


def test_existing_passive_slab_checkpoint_can_enter_strict_admission_mode(source, tmp_path, monkeypatch):
    import run_geometric_slab_probe as passive
    monkeypatch.setattr(passive, "load_probe_source", runner.load_probe_source)
    monkeypatch.setattr(passive, "surface_from_lithosphere", runner.surface_from_lithosphere)
    options = dict(common_omega=(.03, -.02, .01))
    first = passive.execute_probe(source.path, tmp_path/"passive", .25, 1, **options)
    resumed = runner.execute_probe(source.path, tmp_path/"strict",
        resume=tmp_path/"passive"/first["checkpoint_file"], **options)
    assert resumed["status"] == "complete" and resumed["completed_steps"] == 1
    state, inventory, saved = load_boundary_checkpoint(tmp_path/"strict"/resumed["checkpoint_file"])
    assert state.time_myr == 2.5 and len(inventory.transactions) == 2
    assert saved["experiment"]["format"] == runner.FORMAT
    assert saved["history"][0] == first["history"][0]


def test_input_mutation_is_detected_before_publishing_any_checkpoint(source, tmp_path, monkeypatch):
    snapshot = write_faults(source, tmp_path/"faults.json")
    advance = runner.advance_geometric_surface
    def changed_input(*args, **kwargs):
        result = advance(*args, **kwargs)
        snapshot.write_text(snapshot.read_text(encoding="utf-8")+"\n", encoding="utf-8")
        return result
    monkeypatch.setattr(runner, "advance_geometric_surface", changed_input)
    with pytest.raises(RuntimeError, match="changed during the probe"):
        runner.execute_probe(source.path, tmp_path/"changed", fault_snapshot=snapshot,
                             common_omega=(.03, -.02, .01))
    assert not list((tmp_path/"changed").glob("*checkpoint*"))


@pytest.mark.parametrize("change", ["source", "dt", "motion", "polarity"])
def test_resume_rejects_changed_experiment_before_creating_output(source, tmp_path, change):
    options = dict(common_omega=(.03, -.02, .01))
    first = runner.execute_probe(source.path, tmp_path/"first", **options)
    if change == "source":
        source.provenance["source"] = str(tmp_path/"different.source")
    elif change == "motion":
        options["common_omega"] = (.02, -.02, .01)
    elif change == "polarity":
        options["inherit_legacy_polarity"] = True
    with pytest.raises(ValueError, match="identical source, motion, time step and polarity"):
        runner.execute_probe(source.path, tmp_path/"rejected", .5 if change == "dt" else .25,
            resume=tmp_path/"first"/first["checkpoint_file"], **options)
    assert not (tmp_path/"rejected").exists()


@pytest.mark.parametrize("change", ["history", "ledger"])
def test_rehashed_but_inconsistent_resume_provenance_is_rejected(source, tmp_path, change):
    options = dict(common_omega=(.03, -.02, .01))
    first = runner.execute_probe(source.path, tmp_path/"first", **options)
    state, inventory, saved = load_boundary_checkpoint(tmp_path/"first"/first["checkpoint_file"])
    if change == "history":
        saved["history"][-1]["time_myr"] += .1
    else:
        saved["cumulative_losses"]["area_km2"] += 1.
    altered = tmp_path/"altered.json"
    save_boundary_checkpoint(altered, state, inventory, provenance=saved)
    with pytest.raises(ValueError, match="(?i)(chronology|ledger)"):
        runner.execute_probe(source.path, tmp_path/"rejected", resume=altered, **options)
    assert not (tmp_path/"rejected").exists()


@pytest.mark.parametrize("with_transaction", [False, True])
def test_internally_valid_checkpoint_cannot_claim_another_initial_geometry(source, tmp_path, with_transaction):
    options = dict(common_omega=(.03, -.02, .01))
    first = runner.execute_probe(source.path, tmp_path/"first", **options)
    _, _, saved = load_boundary_checkpoint(tmp_path/"first"/first["checkpoint_file"])
    original = from_fractional_surface(source.mesh, source.fractional, 100.)
    omega = np.tile(options["common_omega"], (2, 1))
    # Same IDs, amounts and clock, but the supposedly initial surface is rotated.
    altered = replace(rotate_surface(original, omega, .13), time_myr=original.time_myr)
    inventory = runner.initialize_boundary(altered)
    if with_transaction:
        factory = runner.make_birth_factory(source.fractional, source.model, saved["experiment"])
        result = runner.advance_geometric_surface(source.mesh, altered, omega, .25, birth_factory=factory)
        inventory = runner.consume_geometric_transaction(altered, result, inventory, omega, .25)
        altered = result.state
    else:
        saved["history"] = []
    checkpoint = tmp_path/"wrong_origin.json"
    save_boundary_checkpoint(checkpoint, altered, inventory, provenance=saved)
    # All internal state/checksum/transaction checks pass. Only the independent
    # binding to the actual immutable source can reveal this substituted origin.
    load_boundary_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="immutable source geometry"):
        runner.execute_probe(source.path, tmp_path/"rejected", resume=checkpoint, **options)
    assert not (tmp_path/"rejected").exists()


def test_locked_faults_do_not_remove_material_or_publish_checkpoint(source, tmp_path):
    snapshot = write_faults(source, tmp_path/"strong.json", strong=True)
    report = runner.execute_probe(source.path, tmp_path/"locked", fault_snapshot=snapshot)
    assert report["status"] == "unresolved_initiation"
    assert report["completed_steps"] == 0 and report["checkpoint_file"] is None
    assert all(value == 0. for value in report["cumulative_losses"].values())


def test_existing_output_is_never_overwritten(source, tmp_path):
    output = tmp_path/"existing"
    output.mkdir()
    (output/"sentinel").write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="new directory"):
        runner.execute_probe(source.path, output)
    assert (output/"sentinel").read_text(encoding="utf-8") == "preserve"


def test_cli_is_single_step_and_returns_blocked_exit_code(source, tmp_path, capsys):
    assert runner.main(["--source", str(source.path), "--out", str(tmp_path/"cli")]) == 2
    assert json.loads(capsys.readouterr().out)["completed_steps"] == 0
    with pytest.raises(SystemExit):
        runner.main(["--out", str(tmp_path/"many"), "--steps", "2"])
