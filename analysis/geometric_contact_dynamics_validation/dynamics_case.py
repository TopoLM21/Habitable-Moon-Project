"""Reproduce basal/contact Euler equilibrium on a frozen synthetic sphere.

The octant lid, planes and precompression are declared inputs. The existing
prescribed mantle traction supplies every driving moment. No target speed,
material transfer, cooling, bending or spontaneous fault selection is used.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
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
from tectonics.basal_coupling import PrescribedBasalParameters
from tectonics.genesis import SECONDS_PER_MYR
from tectonics.genesis_contact_law import ContactLawParameters
from tectonics.geometric_basal_loading import (
    assemble_geometric_basal_load, basal_load_from_dict, basal_load_to_dict,
)
from tectonics.geometric_boundary import initialize_boundary
from tectonics.geometric_boundary_io import load_boundary_checkpoint, save_boundary_checkpoint
from tectonics.geometric_contact_dynamics import (
    advance_contact_dynamics, contact_dynamics_from_dict, contact_dynamics_to_dict,
    initialize_contact_dynamics,
)
from tectonics.geometric_contact_mechanics import ElasticColumnParameters
from tectonics.geometric_contact_mechanics_adapter import ContactExperiment
from tectonics.geometric_contacts import extract_contacts
from tectonics.geometric_initiation import OrientedFault
from tectonics.geometric_initiation_adapter import make_forced_underthrust_resolver

DT_S = 1e11
STEP_COUNT = 8
MAX_ROTATION_RAD = .005
SOURCE_PROVENANCE = (
    "Synthetic octant control: existing prescribed quadratic mantle traction, "
    "explicit full thermal lid H[octant:i]=20+5*i km, independent basal drag; "
    "fixed Cartesian source, no fitted plate speed"
)


def _hashes():
    paths = sorted((ROOT/"tectonics").glob("*.py"))
    paths += [ROOT/"analysis/geometric_initiation_validation/admission_case.py", Path(__file__).resolve()]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _inputs(surface):
    parameters = PrescribedBasalParameters(seed=31, convective_traction_pa=1e6,
        basal_drag_pa_s_m=1e15, traction_coupling_depth_km=50.)
    lid = {f"octant:{i}": 20.+5.*i for i in range(8)}
    columns = ElasticColumnParameters(5e4, 5e10, 2e10, 5e4, 5e10, 2e10)
    law = ContactLawParameters(normal_stiffness_pa_m=1e7, tangential_stiffness_pa_m=1e7,
        tensile_strength_pa=2e6, fracture_energy_j_m2=2e7, cohesion_pa=1e4,
        friction_dry=.05, friction_wet=.025, wet_cohesion_fraction=.5, viscosity_pa_s_m=1e15)
    experiments = tuple(ContactExperiment(
        OrientedFault(f"declared:{c.contact_id}", c.contact_id, 1, 0, math.pi/6.,
            None, law.cohesion_pa, law.friction_dry,
            "Synthetic initial dip30deg plane, bank1 below bank0; no stress prescribed"),
        columns, law, 1e3, 0., 0.,
        "Synthetic paired fault-normal elastic columns; explicitly dry, zero pore pressure")
        for c in extract_contacts(surface.fragments, surface.radius_km))
    offsets = {item.fault.fault_id: (-20., 0.) for item in experiments}
    return parameters, lid, experiments, offsets


def _prepare(surface, experiments, offsets, load):
    return initialize_contact_dynamics(surface, experiments, offsets, load,
        initial_loading_dt_s=DT_S, max_rotation_radians=MAX_ROTATION_RAD)


def _locked_oracle(surface, experiments, offsets, load):
    """Independent linear series-spring matrix, before any contact slides."""
    n = 3*len(load.plate_ids)
    stiffness, preload = np.zeros((n, n)), np.zeros(n)
    contacts = {c.contact_id: c for c in extract_contacts(surface.fragments, surface.radius_km)}
    positions = {pid: i for i, pid in enumerate(load.plate_ids)}
    for experiment in experiments:
        f = experiment.fault
        c = contacts[f.contact_id]
        sign = 1. if f.overriding_plate == c.plate_b else -1.
        mean = sign*np.asarray(c.moment_arm_cross_normal_km2)*1000./c.length_km
        b = np.zeros((len(load.plate_ids), 3))
        b[positions[f.subducting_plate]] = -mean
        b[positions[f.overriding_plate]] = mean
        b = b.ravel()
        sine, cosine = math.sin(f.dip_radians), math.cos(f.dip_radians)
        kn, kt = experiment.columns.normal_stiffness_pa_m, experiment.columns.shear_stiffness_pa_m
        alpha = 1./(1./kn+1./experiment.law.normal_stiffness_pa_m)
        shear = 1./(1./kt+1./experiment.law.tangential_stiffness_pa_m)
        area = c.length_km*1000.*experiment.contact_height_m/sine
        stiffness += area*(alpha*sine*sine+shear*cosine*cosine)*np.outer(b, b)
        gap, jump = offsets[f.fault_id]
        preload += area*(alpha*sine*gap+shear*cosine*jump)*b
    dt = 1e10
    result = advance_contact_dynamics(surface, _prepare(surface, experiments, offsets, load), load, dt)
    assert all(item.step.slip_increment_m == 0. for item in result.contacts)
    expected = np.linalg.solve(np.asarray(load.drag_nm_s)+dt*stiffness,
                              np.asarray(load.torque_nm)-preload)
    computed = np.asarray(result.angular_velocity_rad_s).ravel()
    relative = float(np.linalg.norm(computed-expected)/np.linalg.norm(expected))
    assert relative < 2e-11
    return dict(step_duration_s=dt, expected_angular_velocity_rad_s=expected.reshape((-1, 3)).tolist(),
        computed_angular_velocity_rad_s=result.angular_velocity_rad_s, relative_error=relative,
        all_contacts_locked=True, initial_preload_torque_nm=preload.tolist())


def _metadata(surface, state, load):
    return dict(basal_source=basal_load_to_dict(surface, load),
                global_contact=contact_dynamics_to_dict(surface, state, load))


def execute(output):
    started, hashes = time.perf_counter(), _hashes()
    output = Path(output)
    if output.exists():
        raise ValueError("Validation output must be new")
    mesh, surface = fixture()
    archive = initialize_boundary(surface)
    original = snapshot(surface), snapshot(archive)
    parameters, lid, experiments, offsets = _inputs(surface)
    load = assemble_geometric_basal_load(mesh, surface, parameters, lid, SOURCE_PROVENANCE)
    oracle = _locked_oracle(surface, experiments, offsets, load)
    current = _prepare(surface, experiments, offsets, load)
    preparation = asdict(current)
    rows, steps = [], []
    output.mkdir(parents=True)
    for index in range(STEP_COUNT):
        step = advance_contact_dynamics(surface, current, load, DT_S)
        current = step.state
        _, admission = make_forced_underthrust_resolver(surface,
            np.asarray(step.angular_velocity_rad_s)*SECONDS_PER_MYR, DT_S/SECONDS_PER_MYR,
            faults=tuple(item.solved_fault for item in step.contacts))
        torque_scale = max(np.linalg.norm(load.torque_nm), np.linalg.norm(step.basal_drag_torque_nm),
                           np.linalg.norm(step.contact_torque_nm), np.finfo(float).tiny)
        work_scale = max(abs(step.source_work_j), abs(step.basal_dissipation_j)
            +abs(step.stored_energy_increment_j)+step.friction_work_j
            +step.viscous_work_j+step.numerical_dissipation_j, np.finfo(float).tiny)
        torque_error = float(np.linalg.norm(step.torque_residual_nm)/torque_scale)
        work_error = abs(step.energy_balance_residual_j)/work_scale
        assert torque_error < 2e-10 and work_error < 2e-10
        rows.append(dict(step=index+1, mechanics=asdict(step), conditional_admission=admission,
                         resultant_torque_relative_residual=torque_error,
                         incremental_work_relative_residual=work_error))
        steps.append(step)
        if index == 2:
            save_boundary_checkpoint(output/"middle.json", surface, archive,
                                     provenance=_metadata(surface, current, load))
    save_boundary_checkpoint(output/"direct_final.json", surface, archive,
                             provenance=_metadata(surface, current, load))
    loaded, inventory, provenance = load_boundary_checkpoint(output/"middle.json")
    resumed_load = basal_load_from_dict(provenance["basal_source"], loaded, mesh=mesh,
        parameters=parameters, thermal_lid_by_fragment=lid, provenance=SOURCE_PROVENANCE)
    resumed = contact_dynamics_from_dict(provenance["global_contact"], loaded, resumed_load)
    for _ in range(3, STEP_COUNT):
        resumed = advance_contact_dynamics(loaded, resumed, resumed_load, DT_S).state
    assert resumed == current and resumed_load == load
    save_boundary_checkpoint(output/"resumed_final.json", loaded, inventory,
                             provenance=_metadata(loaded, resumed, resumed_load))
    assert (output/"direct_final.json").read_bytes() == (output/"resumed_final.json").read_bytes()
    half = _prepare(surface, experiments, offsets, load)
    for _ in range(2*STEP_COUNT):
        half = advance_contact_dynamics(surface, half, load, DT_S/2.).state
    difference = np.linalg.norm(np.asarray(current.rotation_radians)-np.asarray(half.rotation_radians))
    assert (snapshot(surface), snapshot(archive)) == original
    assert snapshot(loaded) == original[0] and snapshot(inventory) == original[1]
    report = dict(format="geometric-frozen-euler-contact-validation-1", status="complete",
        elapsed_seconds=time.perf_counter()-started,
        runtime=dict(python=platform.python_version(), platform=platform.platform(),
            packages={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "PyYAML", "pytest")}),
        source_model="eight equal-age equal-buoyancy octants, two hemispheric plates, frozen geometry",
        inputs=dict(basal_source=basal_load_to_dict(surface, load),
                    step_duration_s=DT_S, step_count=STEP_COUNT,
                    max_small_rotation_rad=MAX_ROTATION_RAD, preparation=preparation),
        independent_locked_control=oracle, load_steps=rows,
        final_mechanical_history=contact_dynamics_to_dict(surface, current, load),
        maximum_resultant_torque_relative_residual=max(r["resultant_torque_relative_residual"] for r in rows),
        maximum_incremental_work_relative_residual=max(r["incremental_work_relative_residual"] for r in rows),
        friction_work_j=math.fsum(s.friction_work_j for s in steps),
        viscous_work_j=math.fsum(s.viscous_work_j for s in steps),
        numerical_dissipation_j=math.fsum(s.numerical_dissipation_j for s in steps),
        sliding_contact_steps=sum(c.step.slip_increment_m != 0. for s in steps for c in s.contacts),
        time_step_sensitivity=dict(comparison_step_duration_s=DT_S/2.,
            final_small_rotation_difference_rad=float(difference),
            final_small_rotation_relative_difference=float(difference/np.linalg.norm(current.rotation_radians)),
            interpretation="backward-Euler timestep sensitivity, not an assertion of convergence"),
        source_surface_preserved=True, passive_material_archive_preserved=True,
        transferred_material_area_km2=0., new_slab_cohorts=0,
        restart_exact=True, checkpoint_bytes_identical=True,
        checkpoint_sha256={name: hashlib.sha256((output/name).read_bytes()).hexdigest()
            for name in ("middle.json", "direct_final.json", "resumed_final.json")},
        cloud_source_bundle_sha256={name: hashlib.sha256((ROOT/"handoff"/name).read_bytes()).hexdigest()
            for name in ("cloud_sources.zip", "cloud_sources_manifest.json")},
        code_sha256=hashes, code_unchanged=hashes == _hashes(),
        scope="Solved SI Euler/contact balance on a frozen reference with declared preload; "
              "arc-mean Galerkin contact strain, fixed source and lid; no finite surface motion, "
              "material transfer, cooling, bending, slab pull, ridges or generated fault planes.")
    assert report["code_unchanged"]
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    result = execute(parser.parse_args().out)
    print(json.dumps({key: result[key] for key in (
        "status", "elapsed_seconds", "maximum_resultant_torque_relative_residual",
        "maximum_incremental_work_relative_residual", "sliding_contact_steps",
        "source_surface_preserved", "passive_material_archive_preserved",
        "restart_exact", "checkpoint_bytes_identical", "time_step_sensitivity")}, indent=2))
