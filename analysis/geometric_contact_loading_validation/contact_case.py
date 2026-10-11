"""Reproduce solved contact loading, locked shortening and exact restart.

The spherical material fixture remains frozen. A declared plane and column
clamps are explicit synthetic inputs. Diagnostic Euler velocities test the
existing admission interface independently; they are not a solved prediction.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from analysis.geometric_initiation_validation.admission_case import fixture, snapshot
from tectonics.genesis_contact_law import ContactLawParameters
from tectonics.geometric_boundary import initialize_boundary
from tectonics.geometric_boundary_io import load_boundary_checkpoint, save_boundary_checkpoint
from tectonics.geometric_contact_mechanics import ElasticColumnParameters
from tectonics.geometric_contact_mechanics_adapter import (
    ContactExperiment, mechanics_from_dict, mechanics_to_dict,
    solve_contact_mechanics, solved_fault_snapshot,
)
from tectonics.geometric_contacts import extract_contacts
from tectonics.geometric_initiation import OrientedFault, evaluate_fault
from tectonics.geometric_initiation_adapter import make_forced_underthrust_resolver
from tectonics.geometric_polarity import PolarityEvidence

DT_S = 1e10


def _hashes():
    files = ["tectonics/"+name+".py" for name in (
        "geometric_contact_mechanics", "geometric_contact_mechanics_adapter",
        "genesis_contact_law", "geometric_initiation", "geometric_initiation_adapter",
        "geometric_boundary", "geometric_boundary_io", "geometric_contacts",
        "geometric_surface", "spherical_polygons", "mesh")]
    files += ["analysis/geometric_initiation_validation/admission_case.py",
              str(Path(__file__).resolve().relative_to(ROOT))]
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}


def _declared(surface):
    contact = extract_contacts(surface.fragments, surface.radius_km)[0]
    fault = OrientedFault("synthetic-column-fault", contact.contact_id, 1, 0,
        math.pi/6., None, 2e6, .2, "explicit pre-existing dip30deg weak plane, sub bank1")
    columns = ElasticColumnParameters(5e4, 5e10, 2e10, 5e4, 5e10, 2e10)
    law = ContactLawParameters(normal_stiffness_pa_m=1e7, tangential_stiffness_pa_m=1e7,
        tensile_strength_pa=2e6, fracture_energy_j_m2=2e7, cohesion_pa=2e6,
        friction_dry=.2, friction_wet=.1, wet_cohesion_fraction=.5, viscosity_pa_s_m=1e15)
    experiment = ContactExperiment(fault, columns, law, 1e3, 0., 0.,
        "paired fault-normal columns with prescribed midpoint clamps; all values synthetic SI inputs")
    h = np.asarray(contact.normal_a_to_b)*(1 if fault.overriding_plate == contact.plate_b else -1)
    return contact, experiment, h


def _anchors(horizontal, amount):
    values = np.zeros((2, 3))
    values[1] = amount*horizontal
    return values


def _diagnostic_omega(contact):
    values = np.zeros((2, 3))
    values[contact.plate_b] = -.01*np.cross(contact.midpoint, contact.normal_a_to_b)
    return values


def _admission(surface, contact, results, *, history=()):
    _, audit = make_forced_underthrust_resolver(surface, _diagnostic_omega(contact), .1,
        faults=tuple(item.solved_fault for item in results), history=history)
    return next(row for row in audit["contacts"] if row["contact_id"] == contact.contact_id)


def execute(output):
    started, hashes = time.perf_counter(), _hashes()
    output = Path(output)
    if output.exists():
        raise ValueError("Validation output must be new")
    _, surface = fixture()
    archive = initialize_boundary(surface)
    initial = snapshot(surface), snapshot(archive)
    contact, experiment, h = _declared(surface)
    load_path_m = (10., 40., 30., 45.)
    current, rows = None, []
    output.mkdir(parents=True)
    for index, amount in enumerate(load_path_m):
        current, (result,) = solve_contact_mechanics(surface, (experiment,), _anchors(h, amount), DT_S,
                                                    previous=current)
        rows.append(dict(anchor_horizontal_displacement_m=amount, result=asdict(result),
                         conditional_admission=_admission(surface, contact, (result,))))
        if index == 0:
            assert result.down_dip_plastic_increment_m == 0.
            assert result.step.state.bulk_energy_j > 0.
            assert result.step.state.contact_energy_j > 0.
            # Independent series-elastic result, before any yielding.
            kn, kt = experiment.columns.normal_stiffness_pa_m, experiment.columns.shear_stiffness_pa_m
            expected_gap = -amount*math.sin(experiment.fault.dip_radians)*kn/(kn+experiment.law.normal_stiffness_pa_m)
            expected_jump = -amount*math.cos(experiment.fault.dip_radians)*kt/(kt+experiment.law.tangential_stiffness_pa_m)
            assert math.isclose(result.step.state.gap_m, expected_gap, rel_tol=2e-13)
            assert math.isclose(result.step.state.jump_m, expected_jump, rel_tol=2e-13)
            assert rows[-1]["conditional_admission"]["status"] == "locked"
            save_boundary_checkpoint(output/"first.json", surface, archive,
                provenance={"local_contact": mechanics_to_dict(surface, current)})
        if index == 1:
            assert result.down_dip_plastic_increment_m > 0.
            assert rows[-1]["conditional_admission"]["status"] == "forced_underthrust_admissible"
            (output/"solved_fault_input.json").write_text(json.dumps(
                solved_fault_snapshot(surface, (result,)), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    save_boundary_checkpoint(output/"direct_final.json", surface, archive,
        provenance={"local_contact": mechanics_to_dict(surface, current)})
    loaded, inventory, provenance = load_boundary_checkpoint(output/"first.json")
    resumed = mechanics_from_dict(provenance["local_contact"], loaded)
    for amount in load_path_m[1:]:
        resumed, _ = solve_contact_mechanics(loaded, (experiment,), _anchors(h, amount), DT_S, previous=resumed)
    assert resumed == current
    save_boundary_checkpoint(output/"resumed_final.json", loaded, inventory,
        provenance={"local_contact": mechanics_to_dict(loaded, resumed)})
    assert (output/"direct_final.json").read_bytes() == (output/"resumed_final.json").read_bytes()

    mirror = replace(experiment, fault=replace(experiment.fault, fault_id="opposite-declared-plane",
        subducting_plate=0, overriding_plate=1))
    _, alternatives = solve_contact_mechanics(surface, (experiment, mirror), _anchors(h, 40.), DT_S)
    opposed = _admission(surface, contact, alternatives)
    assert opposed["status"] == "multiple_admissible_directions"
    old = tuple(PolarityEvidence(f"opposed-old-material:{sub}", contact.contact_id, sub, 1-sub,
        "synthetic opposed material histories; no inventory deletion") for sub in (0, 1))
    conflict = _admission(surface, contact, alternatives, history=old)
    assert conflict["status"] == "conflicting_local_history"
    dry_rate_independent = replace(experiment, law=replace(experiment.law, viscosity_pa_s_m=0.))
    _, (at_yield,) = solve_contact_mechanics(surface, (dry_rate_independent,), _anchors(h, 40.), DT_S)
    assert at_yield.down_dip_plastic_increment_m > 0.
    static_gate = evaluate_fault(at_yield.solved_fault, contact, _diagnostic_omega(contact), surface.radius_km)
    assert static_gate.status == "locked"
    assert (snapshot(surface), snapshot(archive)) == initial
    assert snapshot(loaded) == snapshot(surface) and snapshot(inventory) == snapshot(archive)
    source_hashes = {name: hashlib.sha256((ROOT/"handoff"/name).read_bytes()).hexdigest()
                     for name in ("cloud_sources.zip", "cloud_sources_manifest.json")}
    report = dict(format="geometric-local-contact-loading-validation-1", status="complete",
        elapsed_seconds=time.perf_counter()-started,
        runtime=dict(python=platform.python_version(), platform=platform.platform(),
                     packages={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "PyYAML", "pytest")}),
        source_model="eight octants, equal-age equal-buoyancy material fixture, frozen geometry",
        inputs=dict(experiment=asdict(experiment), anchor_horizontal_displacements_m=load_path_m,
            step_duration_s=DT_S, diagnostic_omega_rad_per_myr=_diagnostic_omega(contact).tolist(),
            diagnostic_motion_source="independent prescribed convergence, not solved plate dynamics"),
        load_steps=rows, final_mechanical_history=mechanics_to_dict(surface, current),
        controls=dict(mirrored_alternatives=opposed, conflicting_history=conflict,
            rate_independent_sliding_with_strict_static_gate=asdict(static_gate)),
        maximum_force_relative_residual=max(row["result"]["step"]["force_relative_residual"] for row in rows),
        maximum_energy_balance_relative_error=max(row["result"]["step"]["energy_balance_relative_error"] for row in rows),
        source_surface_preserved=True, passive_material_archive_preserved=True,
        transferred_material_area_km2=0., new_slab_cohorts=0,
        restart_exact=True, checkpoint_bytes_identical=True,
        checkpoint_sha256={name: hashlib.sha256((output/name).read_bytes()).hexdigest()
                          for name in ("first.json", "direct_final.json", "resumed_final.json")},
        cloud_source_bundle_sha256=source_hashes, code_sha256=hashes,
        code_unchanged=hashes == _hashes(),
        scope="Solved local contact on declared inclined planes with prescribed elastic clamps; "
              "no spontaneous fault formation, bending, Euler velocity solve or spherical material transfer.")
    assert report["code_unchanged"]
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = execute(args.out)
    print(json.dumps({key: result[key] for key in (
        "status", "elapsed_seconds", "maximum_force_relative_residual",
        "maximum_energy_balance_relative_error", "source_surface_preserved",
        "passive_material_archive_preserved", "restart_exact", "checkpoint_bytes_identical")}, indent=2))
