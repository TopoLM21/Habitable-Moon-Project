"""Persistent, finite-width frictional shear bands in the moving shell.

These material weak planes accommodate irreversible, non-dilatant shear.
Shared mesh vertices still enforce continuous displacement: this is not a
free-surface split, a contact search, or a mature plate handoff. Dissipation
is recorded separately and does not enter the global/column heat ledgers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from .genesis import GenesisParameters, GenesisState
from .genesis_checkpoint_compat import MODEL_VERSION, require_thermal_model_version
from .genesis_shell import ShellParameters, Membrane
from .genesis_onset import OnsetParameters
from .genesis_tides import TidalParameters, TidalOrbitState
from .genesis_mobile import MobileModel, MobileState, MobileParameters, _RetryStep, _validate_loaded
from .genesis_material import face_deformation, polar_increment, rotate_tensor
from .genesis_fault_law import WeakPlaneParameters, return_map, select_plane

FAULT_VERSION = "genesis-faults-0.1"


def transport_plane_normals(plane_normal, deformation):
    """Advect material plane covectors by inverse transpose, then normalize.

    The deformation maps components from the previous face frame to the new
    face frame. Zero normals mark inactive planes and remain zero. Using only
    the polar rotation would miss plane reorientation under finite stretch.
    """
    normal = np.asarray(plane_normal, dtype=float)
    deformation = np.asarray(deformation, dtype=float)
    if (normal.ndim != 2 or normal.shape[1] != 2
            or deformation.shape != (len(normal), 2, 2)
            or not np.isfinite(normal).all() or not np.isfinite(deformation).all()):
        raise ValueError("Plane transport requires finite [face, 2] normals and [face, 2, 2] deformation")
    norm = np.linalg.norm(normal, axis=1)
    active = norm > 0
    if not np.allclose(norm[active], 1., rtol=0, atol=1e-10):
        raise ValueError("Transported material plane normals must be unit vectors or zero")
    determinant = np.linalg.det(deformation)
    singular = np.linalg.svd(deformation, compute_uv=False)
    if (np.any(determinant <= 0) or not np.isfinite(determinant).all()
            or np.any(singular[:, -1] <= 1e-12 * singular[:, 0])):
        raise ValueError("Material plane deformation must preserve orientation and remain nonsingular")
    transported = np.linalg.solve(deformation.transpose(0, 2, 1), normal[..., None])[..., 0]
    norm = np.linalg.norm(transported, axis=1)
    if not np.isfinite(norm).all() or np.any(norm[active] <= 0):
        raise ValueError("Material plane transport produced invalid normals")
    return np.divide(transported, norm[:, None], out=np.zeros_like(transported),
                     where=active[:, None])


@dataclass
class FaultState(MobileState):
    fault_active: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    plane_normal: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    activation_time_myr: np.ndarray = field(default_factory=lambda: np.empty(0))
    fault_candidate_age_myr: np.ndarray = field(default_factory=lambda: np.empty(0))
    cumulative_shear: np.ndarray = field(default_factory=lambda: np.empty(0))
    signed_shear: np.ndarray = field(default_factory=lambda: np.empty(0))
    last_shear_increment: np.ndarray = field(default_factory=lambda: np.empty(0))
    shear_stress_pa: np.ndarray = field(default_factory=lambda: np.empty(0))
    normal_stress_pa: np.ndarray = field(default_factory=lambda: np.empty(0))
    shear_strength_pa: np.ndarray = field(default_factory=lambda: np.empty(0))
    friction_work_cell_j: np.ndarray = field(default_factory=lambda: np.empty(0))
    viscous_work_cell_j: np.ndarray = field(default_factory=lambda: np.empty(0))
    friction_work_j: float = 0.
    viscous_fault_work_j: float = 0.


class FaultModel(MobileModel):
    def __init__(self, shell_p, thermal, onset_p, tides_p, mobile_p=None, fault_p=None):
        super().__init__(shell_p, thermal, onset_p, tides_p, mobile_p)
        self.fault_p = fault_p or WeakPlaneParameters()
        self.fault_p.validate()

    def initial(self):
        state, thermal, orbit = super().initial()
        n = self.mesh.cell_count
        args = {f.name: getattr(state, f.name) for f in fields(state)}
        return FaultState(**args, fault_active=np.zeros(n, dtype=bool), plane_normal=np.zeros((n, 2)),
            activation_time_myr=np.full(n, -1.), cumulative_shear=np.zeros(n), signed_shear=np.zeros(n),
            fault_candidate_age_myr=np.zeros(n), friction_work_cell_j=np.zeros(n), viscous_work_cell_j=np.zeros(n),
            last_shear_increment=np.zeros(n), shear_stress_pa=np.zeros(n), normal_stress_pa=np.zeros(n),
            shear_strength_pa=np.zeros(n)), thermal, orbit

    def _prepare_trial_state(self, state):
        if not self.fault_p.enabled or not state.membrane_established:
            return state
        activate = (~state.fault_active & (state.damage >= self.fault_p.activation_damage)
                    & (state.fault_candidate_age_myr >= self.fault_p.activation_persistence_myr))
        if not np.any(activate):
            return state
        mesh = self.mesh_for(state)
        membrane = Membrane(mesh, self.p.poisson_ratio)
        degradation = self.p.residual_stiffness+(1-self.p.residual_stiffness)*(1-state.damage)**2
        stress = (state.elastic_strain@membrane.d.T)*(self.p.young_modulus_pa*degradation[:, None])
        friction = self.fault_p.friction_dry+(self.fault_p.friction_wet-self.fault_p.friction_dry)*state.water_access
        normals = state.plane_normal.copy()
        normals[activate] = select_plane(stress[activate], friction[activate])
        times = state.activation_time_myr.copy()
        times[activate] = state.time_myr
        return replace(state, plane_normal=normals, fault_active=state.fault_active | activate,
                       activation_time_myr=times)

    def _return(self, state, deformation, elastic_trial, effective_b, damage, dt):
        normals = transport_plane_normals(state.plane_normal, deformation)
        # Water availability is lagged from the accepted step start, just as
        # plane history is. It is never interpreted as a pore-pressure field.
        return return_map(elastic_trial, normals, state.fault_active, damage, state.water_access,
            effective_b, dt, self.p.young_modulus_pa, self.p.poisson_ratio, self.fault_p,
            residual_stiffness=self.p.residual_stiffness)

    def _mechanical_response(self, state, rotation, elastic_trial, effective_b, damage, membrane,
                             dt_myr, *, deformation=None):
        if not self.fault_p.enabled or not np.any(state.fault_active):
            return super()._mechanical_response(state, rotation, elastic_trial, effective_b, damage,
                                                membrane, dt_myr, deformation=deformation)
        result = self._return(state, deformation, elastic_trial, effective_b, damage, dt_myr)
        if not np.any(result["shear_increment"]):
            return super()._mechanical_response(state, rotation, elastic_trial, effective_b, damage,
                                                membrane, dt_myr, deformation=deformation)
        return result["elastic_strain"], result["stress_pa"], result["tangent_pa"]

    def _finish_trial_state(self, before, after, old_mesh, memory, effective_b, fraction, dt_myr):
        if not self.fault_p.enabled:
            return after
        age = np.where(after.membrane_established & (after.damage>=self.fault_p.activation_damage),
                       before.fault_candidate_age_myr+dt_myr, 0.)
        after = replace(after, fault_candidate_age_myr=age)
        if not np.any(before.fault_active):
            return after
        mesh = self.mesh_for(after)
        deformation = face_deformation(old_mesh, mesh, before.radius_km, after.radius_km)
        rotation, inc = polar_increment(deformation)
        elastic_trial = rotate_tensor(memory+effective_b[:, None]*inc, rotation, engineering=True)
        result = self._return(before, deformation, elastic_trial, effective_b, after.damage, dt_myr)
        gamma = result["shear_increment"]
        if float(np.max(np.abs(gamma))) > self.fault_p.max_shear_increment:
            raise _RetryStep("fault_shear_step_limit")
        # Fixed material masses make this effective solid volume independent
        # of purely geometric area changes. These are backward-Euler work
        # estimates, separate from the conserved thermal enthalpy ledgers.
        volume = before.layer_mass_kg.sum(axis=1)*fraction/self.p.density_kg_m3
        friction_work = before.friction_work_cell_j+volume*result["friction_work_density_j_m3"]
        viscous_work = before.viscous_work_cell_j+volume*result["viscous_work_density_j_m3"]
        return replace(after, plane_normal=transport_plane_normals(before.plane_normal, deformation),
            cumulative_shear=before.cumulative_shear+np.abs(gamma), signed_shear=before.signed_shear+gamma,
            last_shear_increment=gamma, shear_stress_pa=result["shear_stress_pa"],
            normal_stress_pa=result["normal_stress_pa"], shear_strength_pa=result["yield_strength_pa"],
            friction_work_cell_j=friction_work, viscous_work_cell_j=viscous_work,
            friction_work_j=float(friction_work.sum()), viscous_fault_work_j=float(viscous_work.sum()))

    def fields(self, state, thermal):
        data = super().fields(state, thermal)
        rate = (np.abs(state.last_shear_increment)*self.fault_p.band_width_km/state.last_step_myr*.1
                if state.last_step_myr > 0 else np.zeros_like(state.cumulative_shear))
        data.update(fault_active=state.fault_active, plane_normal=state.plane_normal,
            cumulative_shear=state.cumulative_shear, signed_shear=state.signed_shear,
            equivalent_slip_km=state.cumulative_shear*self.fault_p.band_width_km,
            slip_rate_cm_yr=rate, shear_stress_mpa=np.abs(state.shear_stress_pa)/1e6,
            shear_strength_mpa=state.shear_strength_pa/1e6)
        return data

    def diagnostics(self, state, thermal, orbit):
        g, s, o, orbital = super().diagnostics(state, thermal, orbit)
        area = self.mesh_for(state).areas_unit_sphere
        equivalent = state.cumulative_shear*self.fault_p.band_width_km
        s.update(fault_active_area_fraction=float(np.average(state.fault_active, weights=area)),
            fault_slipping_area_fraction=float(np.average(np.abs(state.last_shear_increment)>0, weights=area)),
            max_equivalent_slip_km=float(equivalent.max()),
            mean_equivalent_slip_km=float(np.average(equivalent, weights=area)),
            max_shear_increment=float(np.abs(state.last_shear_increment).max()))
        o.update(friction_work_j=state.friction_work_j, viscous_fault_work_j=state.viscous_fault_work_j,
                 fault_dissipation_j=state.friction_work_j+state.viscous_fault_work_j)
        return g, s, o, orbital

    def checkpoint_array_shapes(self):
        return {"plane_normal": (self.mesh.cell_count, 2)}


def save_fault_checkpoint(path, model, state, thermal, orbit, controls, provenance):
    arrays = {f.name: getattr(state, f.name) for f in fields(state) if isinstance(getattr(state, f.name), np.ndarray)}
    scalars = {f.name: getattr(state, f.name) for f in fields(state) if f.name not in arrays}
    parameters = {"shell": asdict(model.p), "thermal": asdict(model.thermal), "onset": asdict(model.onset_p),
                  "tides": asdict(model.tides_p), "mobile": asdict(model.mobile_p), "faults": asdict(model.fault_p)}
    meta = {"format": FAULT_VERSION, "thermal_model_version": MODEL_VERSION, "parameters": parameters,
        "parameter_hash": hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest(),
        "state": scalars, "thermal_state": asdict(thermal), "orbit": asdict(orbit),
        "controls": controls, "provenance": provenance}
    target = Path(path)
    temporary = Path(str(target)+".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, metadata=np.array(json.dumps(meta, allow_nan=False)), **arrays)
    temporary.replace(target)


def load_fault_checkpoint(path):
    try:
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata"]))
            p = meta["parameters"]
            if meta["format"] != FAULT_VERSION or hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest() != meta["parameter_hash"]:
                raise ValueError("Fault checkpoint version/parameter hash mismatch")
            require_thermal_model_version(meta)
            model = FaultModel(ShellParameters(**p["shell"]), GenesisParameters(**p["thermal"]),
                OnsetParameters(**p["onset"]), TidalParameters(**p["tides"]), MobileParameters(**p["mobile"]),
                WeakPlaneParameters(**p["faults"]))
            state = FaultState(**meta["state"], **{k: archive[k].copy() for k in archive.files if k != "metadata"})
        thermal, orbit = GenesisState(**meta["thermal_state"]), TidalOrbitState(**meta["orbit"])
        _validate_loaded(model, state, thermal, orbit)
        _validate_fault_history(model, state)
        return model, state, thermal, orbit, meta
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Malformed fault checkpoint") from exc


def _validate_fault_history(model, state):
    if state.fault_active.dtype != np.bool_:
        raise ValueError("Fault activation flags must be boolean")
    active = state.fault_active
    if (not np.allclose(np.linalg.norm(state.plane_normal[active], axis=1), 1, atol=1e-10, rtol=0)
            or np.any(state.plane_normal[~active] != 0)
            or np.any(state.activation_time_myr[~active] != -1)
            or np.any((state.activation_time_myr[active]<0)|(state.activation_time_myr[active]>state.time_myr))
            or np.any(state.activation_time_myr[active]+1e-12 < model.fault_p.activation_persistence_myr)
            or np.any(state.cumulative_shear<0) or np.any(state.shear_strength_pa<0)
            or np.any((state.fault_candidate_age_myr<0)|(state.fault_candidate_age_myr>state.time_myr+1e-12))
            or np.any(state.friction_work_cell_j<0) or np.any(state.viscous_work_cell_j<0)
            or np.any(np.abs(state.signed_shear)>state.cumulative_shear+1e-12)
            or np.any(np.abs(state.last_shear_increment)>state.cumulative_shear+1e-12)
            or np.any(np.abs(state.last_shear_increment)>model.fault_p.max_shear_increment)
            or np.any(state.cumulative_shear[~active] != 0)
            or state.friction_work_j < 0 or state.viscous_fault_work_j < 0
            or (np.any(active) and not state.membrane_established)
            or (not model.fault_p.enabled and (np.any(active) or state.friction_work_j != 0 or state.viscous_fault_work_j != 0))):
        raise ValueError("Invalid fault history, orientation or work ledger")
    for cells, total in ((state.friction_work_cell_j, state.friction_work_j),
                         (state.viscous_work_cell_j, state.viscous_fault_work_j)):
        if not np.isclose(float(cells.sum()), total, rtol=1e-12, atol=1e-12) or np.any(cells[~active] != 0):
            raise ValueError("Fault work ledger does not match material cells")
