"""Audit measured tied reactions and unchanged strengths on the real 5120-cell source.

No cohesive activation, stress rescaling, interface energy or source mutation.
The one-year predictor is isothermal mechanics, not an orbital/climate advance.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields, is_dataclass
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.genesis_path_dynamics_validation import SOURCE, TRACE, _external
from tectonics.genesis_coupled import CoupledModel
from tectonics.genesis_crack_path import ReferenceCrackPath
from tectonics.genesis_path_basis import EmbeddedPathBasis
from tectonics.genesis_path_birth import recover_tied_tractions, classify_tied_onset, tied_reaction_capacity
from tectonics.genesis_path_dynamics import PathMechanics
from tectonics.genesis_path_mesh import insert_crack_path


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _state_hash(state):
    digest = hashlib.sha256()
    def add(value):
        if is_dataclass(value):
            for item in fields(value):
                digest.update(item.name.encode())
                add(getattr(value, item.name))
        elif isinstance(value, np.ndarray):
            digest.update(str((value.dtype.str, value.shape)).encode())
            digest.update(value.tobytes())
        else:
            digest.update(repr(value).encode())
    add(state)
    return digest.hexdigest()


def source_case():
    """Return the fixed-source model, pure loading factory and fixed elasticity."""
    coupled = CoupledModel(SOURCE.read_bytes(), source_path=str(SOURCE))
    source, p = coupled.source.source_state, coupled.source_model.p
    with np.load(TRACE, allow_pickle=False) as data:
        path = ReferenceCrackPath(data["points_xyz"], source.radius_km)
    inserted = insert_crack_path(coupled.original_mesh, path,
        front_coordinates_m=path.length_m*np.array([.25, .5, .75]))
    basis = EmbeddedPathBasis(coupled.original_mesh, inserted, coupled.radius_m, p.poisson_ratio)
    projection = basis.subdivision
    model = PathMechanics(basis, projection.intensive(coupled.source.depth_m),
        coupled.contact_parameters, coupled.law_parameters)
    degradation = p.residual_stiffness+(1-p.residual_stiffness)*(1-source.damage)**2
    elasticity = projection.intensive((p.young_modulus_pa*degradation)[:, None, None]*coupled.original_membrane.d)
    temperature = coupled.source.source_fields["temperature_k"]
    viscosity = np.clip(p.viscosity_reference_pa_s*np.exp(np.clip(p.activation_energy_j_mol/8.314462618
        *(1/np.maximum(temperature, 1)-1/p.viscosity_reference_temperature_k), -60, 60)),
        p.viscosity_min_pa_s, p.viscosity_max_pa_s)
    viscosity, water = projection.intensive(viscosity), projection.intensive(source.water_access)
    _, external = _external(coupled, basis, coupled.source.depth_m/1000.)

    def factory(state, years):
        return model.isothermal_loading(state, years, model.reference_volume_m3,
            elasticity, viscosity, p.young_modulus_pa, external, water)

    return coupled, model, factory, elasticity, water


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Validation output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), ROOT/"tectonics/genesis_path_birth.py",
        ROOT/"tectonics/genesis_path_dynamics.py", ROOT/"tectonics/genesis_path_basis.py",
        ROOT/"tectonics/genesis_contact_geometry.py", ROOT/"tectonics/genesis_contact_law.py"]
    code_hashes = {str(path.relative_to(ROOT)): _hash(path) for path in paths}
    source_hashes = {str(path.relative_to(ROOT)): _hash(path) for path in (SOURCE, TRACE)}
    started = perf_counter()
    coupled, model, factory, elasticity, water = source_case()
    source, basis = coupled.source.source_state, model.basis
    initial = model.initial(basis.subdivision.tensor(source.elastic_strain, engineering=True))
    state = model.trial(initial, factory(initial, 1.))
    source_state_before, tied_before = _state_hash(source), _state_hash(state)
    stress = np.einsum("fij,fj->fi", elasticity, state.elastic_strain)
    recovery = recover_tied_tractions(model, state, stress)
    trace_water = water[basis.topology.seam_faces].mean(axis=1).repeat(2)
    law_before = asdict(model.law_parameters)
    onset = classify_tied_onset(recovery, trace_water, model.law_parameters)
    capacity = tied_reaction_capacity(model, state, trace_water, model.law_parameters)
    # Independent virtual work using a deterministic displacement vector.
    virtual = np.random.default_rng(2036).normal(size=basis.ndof)
    reaction_work = float(state.constraint_reaction_n@virtual)
    interface_work = float(np.sum(model.geometry.interface_area_m2[:, None]*recovery.traction_pa
        *(model.jump_operator@virtual).reshape(-1, 2)))
    virtual_error = abs(reaction_work-interface_work)/max(abs(reaction_work), abs(interface_work), 1.)
    np.savez_compressed(output/"tied_traction_recovery.npz",
        **{item.name: getattr(recovery, item.name) for item in fields(recovery)
           if isinstance(getattr(recovery, item.name), np.ndarray)},
        **{"capacity_"+item.name: getattr(capacity, item.name) for item in fields(capacity)},
        stress_pa=stress, normal_ratio=onset.normal_ratio, shear_ratio=onset.shear_ratio,
        shear_strength_pa=onset.shear_strength_pa, trace_water=trace_water,
        interface_area_m2=model.geometry.interface_area_m2,
        cut_edges=basis.topology.cut_edges, seam_faces=basis.topology.seam_faces,
        metadata=np.array(json.dumps({"format": "tied-traction-diagnostic-0.2",
            "source_sha256": source_hashes, "elapsed_mechanical_years": state.elapsed_years,
            "source_thermal_orbit_age_myr": source.time_myr,
            "physical_birth_performed": False}, allow_nan=False)))
    observed = recovery.trace_observed
    report = {
        "scope": "reaction reconstruction and local-strength diagnostic; no activation or natural-onset claim",
        "source_cells": coupled.original_mesh.cell_count,
        "source_age_myr": source.time_myr, "elapsed_isothermal_mechanical_years": state.elapsed_years,
        "path_length_km": basis.insertion.path.length_m/1000.,
        "reaction_norm_n": float(np.linalg.norm(recovery.target_reaction_n)),
        "trace_count": len(observed), "observed_trace_count": int(observed.sum()),
        "reaction_rank": recovery.reaction_rank, "traction_nullity": recovery.traction_nullity,
        "reaction_relative_residual": recovery.relative_residual,
        "stress_prior_relative_residual": recovery.prior_relative_residual,
        "area_weighted_relative_correction": recovery.relative_area_weighted_correction,
        "virtual_work_relative_error": virtual_error,
        "all_trace_normal_mpa_quantiles": (np.quantile(recovery.traction_pa[:, 0], [0, .25, .5, .75, 1])/1e6).tolist(),
        "observed_max_abs_shear_mpa": float(np.max(np.abs(recovery.traction_pa[observed, 1]))/1e6),
        "observed_tensile_exceeded_count": int(onset.tensile_exceeded.sum()),
        "observed_shear_exceeded_count": int(onset.shear_exceeded.sum()),
        "maximum_observed_strength_ratio": onset.maximum_observed_ratio,
        "capacity_certificate": {
            "scope": "necessary support-function bound for the unchanged tensile/cohesive-Coulomb onset envelope; independent of reconstructed traction nullspace; not a finite-rate viscous law",
            "vertex_count": len(capacity.required_n),
            "finite_capacity_count": int(capacity.finite_capacity.sum()),
            "unbounded_capacity_count": int(capacity.unbounded_capacity.sum()),
            "indeterminate_capacity_count": int(capacity.indeterminate_capacity.sum()),
            "violation_count": int(capacity.violations.sum()),
            "required_to_capacity_ratio_quantiles": np.quantile(capacity.ratio, [0, .25, .5, .75, 1]).tolist(),
            "an_admissible_reconstruction_is_ruled_out": bool(capacity.violations.any()),
            "ratio_below_one_would_not_prove_admissibility": True,
        },
        "status": onset.status, "physical_birth_ready": onset.physical_birth_ready,
        "law_parameters": law_before, "source_sha256": source_hashes,
        "code_sha256": code_hashes, "wall_seconds": perf_counter()-started,
        "interpretation": "The late source is above the unchanged interface strength. A prior-independent necessary capacity bound rules out any traction reconstruction inside the configured tensile/cohesive-Coulomb onset envelope. An earlier physical bracket must be sought rather than resetting peak strength or relabeling prescribed release as nucleation; this does not prove such a bracket exists under a fully coupled run.",
        "references": ["https://arxiv.org/abs/cond-mat/0106318",
            "https://www.sandia.gov/files/sierra/SM_Development_5_26/main/cohesive/ext.html"],
    }
    report["checks"] = {
        "reaction_balanced": recovery.relative_residual < 1e-12,
        "virtual_work_balanced": virtual_error < 1e-12,
        "observability_explicit": recovery.traction_nullity > 0 and int((~observed).sum()) == 2,
        "stress_prior_close": recovery.relative_area_weighted_correction < .01,
        "source_is_supercritical": onset.status == "strength_exceeded" and bool(onset.tensile_exceeded[observed].all()),
        "capacity_directions_are_finite": bool(capacity.finite_capacity.all()),
        "capacity_exceeded_independently_of_prior": bool(capacity.violations.all()),
        "no_strength_reset": asdict(model.law_parameters) == law_before,
        "no_physical_birth": not onset.physical_birth_ready and state.active_interval is None,
        "no_interface_energy_or_work": all(np.all(getattr(state.cohorts, name) == 0) for name in
            ("traction_pa", "damage", "fracture_work_j", "friction_work_j", "viscous_work_j")),
        "states_unchanged_by_recovery": _state_hash(source) == source_state_before and _state_hash(state) == tied_before,
        "source_files_unchanged": all(_hash(ROOT/name) == digest for name, digest in source_hashes.items()),
        "code_unchanged_during_run": all(_hash(ROOT/name) == digest for name, digest in code_hashes.items()),
    }
    report["artifact_sha256"] = {"tied_traction_recovery.npz": _hash(output/"tied_traction_recovery.npz")}
    (output/"validation.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, allow_nan=False))
    if not all(report["checks"].values()):
        raise RuntimeError("Tied traction validation failed")


if __name__ == "__main__":
    main()
