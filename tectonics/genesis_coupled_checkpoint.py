"""Exact restart of coupled cooling with independently ageing contact cohorts.

The 0.1 format predates material-growth histories. It cannot be promoted to
this model by assigning an invented birth date to its aggregate contacts.
"""
from __future__ import annotations

from dataclasses import asdict, fields
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from .genesis import GenesisState
from .genesis_contact import ContactParameters, ContactState
from .genesis_contact_growth import CohortState, aggregate_cohorts, validate_cohorts
from .genesis_contact_law import ContactLawParameters
from .genesis_mobile import _RetryStep
from .genesis_shell import maximum_total_strain
from .genesis_tides import TidalOrbitState, validate_tidal_orbit


def save_coupled_checkpoint(path, model, state, thermal, orbit):
    from .genesis_coupled import COUPLED_VERSION

    arrays = {f.name: getattr(state, f.name) for f in fields(state)
              if isinstance(getattr(state, f.name), np.ndarray)}
    scalars = {f.name: getattr(state, f.name) for f in fields(state)
               if f.name not in arrays and f.name not in {"contact", "cohorts"}}
    contact_arrays = {f.name: getattr(state.contact, f.name) for f in fields(state.contact)
                      if isinstance(getattr(state.contact, f.name), np.ndarray)}
    contact_scalars = {f.name: getattr(state.contact, f.name) for f in fields(state.contact)
                       if f.name not in contact_arrays}
    parameters = {"coupled": asdict(model.parameters), "contact": asdict(model.contact_parameters),
                  "law": asdict(model.law_parameters)}
    meta = {"format": COUPLED_VERSION, "source_time_myr": model.source_time_myr,
            "source_path": model.source_path, "source_hash": model.source_hash,
            "parameters": parameters,
            "parameter_hash": hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest(),
            "state": scalars, "contact_state": contact_scalars,
            "thermal_state": asdict(thermal), "orbit": asdict(orbit)}
    target = Path(path)
    temporary = Path(str(target)+".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, metadata=np.asarray(json.dumps(meta, allow_nan=False)),
            source_checkpoint=np.frombuffer(model.source_bytes, dtype=np.uint8), **arrays,
            **{"contact__"+key: value for key, value in contact_arrays.items()},
            **{"cohort__"+f.name: getattr(state.cohorts, f.name) for f in fields(state.cohorts)})
    temporary.replace(target)


def load_coupled_checkpoint(path):
    from .genesis_coupled import COUPLED_VERSION, CoupledModel, CoupledParameters, CoupledState

    try:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            if meta["format"] == "genesis-coupled-0.1":
                raise ValueError("Coupled 0.1 checkpoints have no independent contact growth histories; "
                                 "restart from the original fault checkpoint instead of inventing material history")
            parameters, source = meta["parameters"], data["source_checkpoint"]
            if (meta["format"] != COUPLED_VERSION or source.dtype != np.uint8 or source.ndim != 1
                    or hashlib.sha256(source.tobytes()).hexdigest() != meta["source_hash"]
                    or hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest() != meta["parameter_hash"]):
                raise ValueError("Coupled checkpoint version or hash mismatch")
            model = CoupledModel(source.tobytes(), CoupledParameters(**parameters["coupled"]),
                ContactParameters(**parameters["contact"]), ContactLawParameters(**parameters["law"]), meta["source_path"])
            contact_arrays = {key[9:]: data[key].copy() for key in data.files if key.startswith("contact__")}
            cohort_arrays = {key[8:]: data[key].copy() for key in data.files if key.startswith("cohort__")}
            state_arrays = {
                key: data[key].copy() for key in data.files
                if key not in {"metadata", "source_checkpoint"} and not key.startswith(("contact__", "cohort__"))}
            # Dataclass defaults are for new states, never substitutes for
            # missing restart ledgers or counters in a supposedly exact save.
            if (set(meta["state"]) | set(state_arrays) != {f.name for f in fields(CoupledState)}-{"contact", "cohorts"}
                    or set(meta["contact_state"]) | set(contact_arrays) != {f.name for f in fields(ContactState)}
                    or set(cohort_arrays) != {f.name for f in fields(CohortState)}):
                raise ValueError("Coupled checkpoint is missing required state history")
            contact = ContactState(**meta["contact_state"], **contact_arrays)
            cohorts = CohortState(**cohort_arrays)
            state = CoupledState(**meta["state"], contact=contact, cohorts=cohorts, **state_arrays)
        thermal, orbit = GenesisState(**meta["thermal_state"]), TidalOrbitState(**meta["orbit"])
        _validate_coupled(model, state, thermal, orbit)
        if meta["source_time_myr"] != model.source_time_myr:
            raise ValueError("Coupled source clock mismatch")
        return model, state, thermal, orbit
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Malformed coupled checkpoint") from exc


def _validate_coupled(model, state, thermal, orbit):
    initial = model.initial()
    n = model.original_mesh.cell_count
    if (state.cut_edges.ndim != 2 or state.cut_edges.shape[1] != 2 or state.cut_edges.dtype.kind not in "iu"
            or not np.array_equal(np.unique(state.cut_edges, axis=0), state.cut_edges)):
        raise ValueError("Invalid coupled cut identities")
    if not set(map(tuple, initial.cut_edges)).issubset(set(map(tuple, state.cut_edges))):
        raise ValueError("Coupled checkpoint removed inherited cuts")
    geometry = model._geometry(state.cut_edges)
    ntraces = 2*len(state.cut_edges)
    shapes = {"cut_edges": (len(state.cut_edges), 2), "interface_birth_area_m2": (ntraces,),
              "interface_depth_ref_m": (ntraces,), "column_enthalpy": model.layer_mass_kg.shape,
              "elastic_strain": (n, 3), "plane_normal": (n, 2), "velocity_km_myr": (n, 3)}
    for f in fields(state):
        value = getattr(state, f.name)
        if isinstance(getattr(initial, f.name), np.ndarray):
            if (not isinstance(value, np.ndarray) or value.dtype.kind not in "bifu"
                    or value.shape != shapes.get(f.name, (n,)) or not np.isfinite(value).all()):
                raise ValueError(f"Invalid coupled field {f.name}")
        elif f.name not in {"contact", "cohorts", "stopped_reason"}:
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"Invalid coupled scalar {f.name}")
    if (state.time_myr != thermal.time_myr or state.time_myr != orbit.time_myr or state.time_myr < model.source_time_myr
            or abs(state.contact.elapsed_years-(state.time_myr-model.source_time_myr)*1e6) > 1e-7):
        raise ValueError("Coupled physical clocks disagree")
    if (any(not isinstance(getattr(state, key), Integral) or getattr(state, key) < 0
            for key in ("accepted_steps", "rejected_steps", "new_seam_count"))
            or state.new_seam_count != len(state.cut_edges)-len(initial.cut_edges)
            or state.last_step_years < 0 or state.last_step_years > state.contact.elapsed_years+1e-7
            or state.tidal_heat_received_j < 0 or state.maxwell_relaxation_release_j < 0
            or state.interface_birth_energy_j < 0
            or state.stopped_reason is not None and not isinstance(state.stopped_reason, str)):
        raise ValueError("Invalid coupled counters or energy history")
    if (state.fault_active.dtype != bool or np.any(state.interface_birth_area_m2 <= 0)
            or np.any(state.interface_depth_ref_m <= 0) or np.any(state.column_enthalpy <= 0)
            or maximum_total_strain(state.elastic_strain) > model.source_model.mobile_p.max_elastic_strain
            or np.any((state.damage < 0)|(state.damage > 1)) or np.any((state.water_access < 0)|(state.water_access > 1))
            or np.any(state.fault_candidate_age_myr < 0)
            or not np.allclose(np.linalg.norm(state.plane_normal[state.fault_active], axis=1), 1., atol=1e-10, rtol=0)
            or np.any(state.plane_normal[~state.fault_active] != 0)):
        raise ValueError("Invalid coupled material history")
    contact, expected_contact = state.contact, geometry.initial()
    for f in fields(contact):
        value, expected = getattr(contact, f.name), getattr(expected_contact, f.name)
        if isinstance(expected, np.ndarray):
            if (not isinstance(value, np.ndarray) or value.dtype.kind not in "fiu"
                    or value.shape != expected.shape or not np.isfinite(value).all()):
                raise ValueError(f"Invalid coupled contact field {f.name}")
        elif f.name != "stopped_reason" and (isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value)):
            raise ValueError(f"Invalid coupled contact scalar {f.name}")
    if (contact.accepted_steps != state.accepted_steps or contact.last_step_years != state.last_step_years
            or contact.drag_work_j < 0 or contact.equilibrium_residual < 0
            or contact.equilibrium_residual > model.contact_parameters.equilibrium_tolerance
            or any(np.any(getattr(contact, name) < 0) for name in
                   ("friction_work_cell_j", "viscous_work_cell_j", "fracture_work_cell_j", "shear_remainder_cell_j"))):
        raise ValueError("Invalid coupled contact work or accepted-step history")
    _validate_cohort_geometry(model, state, geometry)
    validate_tidal_orbit(orbit, model.source_model.tides_p)
    energy = np.asarray(thermal.energy)
    if (energy.shape != (4,) or not np.isfinite(energy).all() or np.any(energy[:2] <= 0)
            or thermal.initial_total_energy != model.source.thermal_state.initial_total_energy):
        raise ValueError("Invalid coupled thermal energy")
    report = model.diagnostics(state, thermal, orbit)
    if (abs(report["relative_global_energy_residual"]) > 1e-5
            or abs(report["relative_column_energy_residual"]) > 1e-8
            or abs(report["orbit_heat_transfer_relative_residual"]) > 1e-8):
        raise ValueError("Coupled heat ledgers are inconsistent")
    _, _, fraction, _, _, depth = model._phase_fields(state, thermal)
    if np.any(fraction*depth < .25*model.source_model.p.min_load_bearing_thickness_km):
        raise ValueError("Coupled checkpoint has lost load-bearing support")
    front = model._reference_front(geometry, fraction)
    tolerance = np.maximum(1e-7, np.abs(front)*1e-11)
    pending = front-state.interface_depth_ref_m
    if np.any(pending < -tolerance) or np.any(pending >= model.parameters.growth_layer_m+tolerance):
        raise ValueError("Coupled represented contact front is inconsistent with solid material")
    area_change = model._interface_area_change(geometry, fraction, depth)
    if (area_change > model.parameters.max_interface_area_change_fraction*(1+1e-12)
            or not np.isclose(state.interface_area_change_fraction, area_change, rtol=1e-8, atol=1e-12)):
        raise ValueError("Coupled interface reference area is inconsistent")
    try:
        model._check_geometry(geometry, contact)
    except _RetryStep as exc:
        raise ValueError("Coupled checkpoint exceeds geometry limits") from exc


def _validate_cohort_geometry(model, state, geometry):
    cohorts, law = state.cohorts, model.law_parameters
    ntraces = 2*len(state.cut_edges)
    validate_cohorts(cohorts, law, ntraces)
    if len(cohorts.trace_index) > model.parameters.max_cohort_count:
        raise ValueError("Coupled cohort count exceeds the configured limit")
    if (np.any(cohorts.birth_time_myr < model.source_time_myr)
            or np.any(cohorts.birth_time_myr > state.time_myr)
            or not np.array_equal(cohorts.bonded, cohorts.birth_gap_m <= model.parameters.bonding_gap_tolerance_m)):
        raise ValueError("Coupled cohort birth history is inconsistent")
    lengths = np.repeat(geometry.edge_length_m, 2)
    expected_area = lengths[cohorts.trace_index]*(cohorts.z_hi_ref_m-cohorts.z_lo_ref_m)/2
    if not np.allclose(cohorts.area_ref_m2, expected_area, rtol=1e-12, atol=1e-6):
        raise ValueError("Coupled cohort reference area disagrees with material interval")
    represented_area = np.bincount(cohorts.trace_index, weights=cohorts.area_ref_m2, minlength=ntraces)
    if not np.allclose(state.interface_birth_area_m2, represented_area, rtol=1e-12, atol=1e-6):
        raise ValueError("Coupled represented area disagrees with contact cohorts")
    for trace in range(ntraces):
        indices = np.flatnonzero(cohorts.trace_index == trace)
        if not len(indices):
            raise ValueError("Coupled contact trace has no material cohort")
        indices = indices[np.argsort(cohorts.z_lo_ref_m[indices], kind="stable")]
        lo, hi = cohorts.z_lo_ref_m[indices], cohorts.z_hi_ref_m[indices]
        if (lo[0] != 0 or not np.array_equal(lo[1:], hi[:-1])
                or hi[-1] != state.interface_depth_ref_m[trace]
                or np.any(np.diff(cohorts.birth_time_myr[indices]) < 0)):
            raise ValueError("Coupled cohort intervals must cover the represented front without gaps or overlaps")
    gap, jump = (geometry.jump_operator@state.contact.displacement_m).reshape(-1, 2).T
    local_gap = gap[cohorts.trace_index]
    relative_opening = np.maximum(local_gap-cohorts.birth_gap_m, 0)
    traction = np.column_stack((
        law.normal_stiffness_pa_m*(np.minimum(local_gap, 0)+(1-cohorts.damage)*relative_opening),
        law.tangential_stiffness_pa_m*(jump[cohorts.trace_index]-cohorts.birth_jump_m-cohorts.plastic_slip_m)))
    if (np.any(cohorts.max_opening_m[cohorts.bonded]+1e-9 < relative_opening[cohorts.bonded])
            or not np.allclose(cohorts.traction_pa, traction, rtol=1e-10, atol=1e-6)):
        raise ValueError("Coupled cohort tractions or opening history disagree with displacement")
    aggregate = aggregate_cohorts(cohorts, ntraces)
    for name, expected in aggregate.items():
        if not np.allclose(getattr(state.contact, name), expected, rtol=1e-12, atol=1e-8):
            raise ValueError(f"Coupled contact summary {name} disagrees with independent cohort histories")
