"""One source-bound BE transition. Separate state/format, no production hook."""
from dataclasses import dataclass,replace,asdict
from types import SimpleNamespace
import hashlib,json
import numpy as np
from tectonics.genesis import SECONDS_PER_MYR,temperatures,diagnose
from tectonics.genesis_bulk_birth import bulk_birth_to_dict
from tectonics.genesis_volume_dynamics import radial_moments
from tectonics.genesis_volume_loading import VolumeMechanicsState,VolumeCohortMemory,prepare_volume_loading
from tectonics.genesis_volume_thermal import prepare_volume_thermal_step,thermal_to_dict
from tectonics.genesis_volume_fault_dynamics import retained_live_fraction
from tectonics.genesis_volume_fault_evolution import evaluate_volume_fault_evolution,volume_fault_history_to_dict
from tectonics.genesis_volume_slip_dynamics import _vertices
from tectonics.genesis_volume_operator import evaluate_genesis_volume_operator
from tectonics.genesis_volume_temporal import cone_homothetic_to_dict
from tectonics.genesis_volume_checkpoint import _geometry_claims,_jsonable
from tectonics.genesis_shell import principal_tensile
from analysis.genesis_connected_step_validation.run import gate,PARENT_SHA
from .solver import solve_trial

FORMAT='genesis-be-isolated-candidate-0.1'


@dataclass(frozen=True)
class BEMemory:
    source_identity:str
    time_myr:float
    accepted_steps:int
    cohorts:tuple
    elastic_energy_j:float


@dataclass(frozen=True)
class BEState:
    geometry:object
    thermal:object
    history:object
    mechanics:BEMemory
    fault_history:object
    displacement_m:np.ndarray
    velocity_m_s:np.ndarray
    last_certificate:object
    interval_ledger:dict
    be_accumulated_ledger:dict
    inherited_parent_sha256:str=PARENT_SHA
    scheme:str='backward_euler_velocity_and_maxwell_endpoint'
    @property
    def time_myr(self):return self.mechanics.time_myr


def preparation_view(memory):
    # Ephemeral compatibility view ONLY for prepare_volume_loading. No old
    # work ledger is replaced or serialized as this artificial energy origin.
    return VolumeMechanicsState(memory.source_identity,memory.time_myr,memory.accepted_steps,
        memory.cohorts,memory.elastic_energy_j,memory.elastic_energy_j)


def no_remelt(before,after):
    rows={c.id:c for c in after.cohorts}
    for c in before.cohorts:
        if c.id not in rows or any(getattr(rows[c.id],k)!=getattr(c,k) for k in ['q_lo','live_q_hi','mass_kg','material_id','owner_id','face_index']):
            raise ValueError('STOP: remelt or changed old material unsupported')


def trial(model,parent,thermal,damage,history,certify):
    before,basal0=model._assembly(parent.geometry,parent.fault_history.damage)
    after,basal=model._assembly(thermal.geometry,damage);no_remelt(parent.geometry,thermal.geometry)
    loading=prepare_volume_loading(preparation_view(parent.mechanics),after_ids=after.cohort_ids,
        after_temperature_k=thermal.after_temperature_k,birth_temperature_k=thermal.birth_temperature_k,
        radial_moments=thermal.radial_moments,before_radial_moments=thermal.before_radial_moments,
        removed_radial_moments=thermal.removed_radial_moments,before_damage=parent.fault_history.damage[before.face_indices],
        after_damage=damage[after.face_indices],parameters=model.shell_parameters,after_time_myr=thermal.state.time_myr,
        before_temperature_k=thermal.before_temperature_k)
    if loading.remelt_energy_removed_j!=0:raise ValueError('STOP: remelt unsupported')
    dt=(thermal.state.time_myr-parent.time_myr)*SECONDS_PER_MYR
    du,v,e0,e1,L=solve_trial(basal.mass_matrix.toarray(),basal.basal_drag_matrix.toarray(),basal.external_force_n,
        parent.velocity_m_s,dt,after.dofs,after.B0,after.B1,loading)
    phase,birth,removed=model._phase_kinetic_flux(before,after,parent.velocity_m_s)
    if removed!=0:raise ValueError('STOP: kinetic remelt unsupported')
    K0=float(.5*parent.velocity_m_s@basal0.mass_matrix@parent.velocity_m_s)
    L.update(kinetic_before_j=K0,elastic_before_j=parent.mechanics.elastic_energy_j,birth_kinetic_inflow_j=birth,
             damage_energy_change_j=loading.damage_energy_change_j,remelt_kinetic_outflow_j=removed,remelt_elastic_outflow_j=0.)
    L['balance_residual_j']=L['kinetic_after_j']+L['elastic_after_j']-K0-L['elastic_before_j']-birth-L['damage_energy_change_j']-L['external_work_j']-L['thermal_work_j']+sum(L[k] for k in ['physical_drag_j','physical_maxwell_j','numerical_kinetic_j','numerical_memory_j'])
    endpoint=evaluate_genesis_volume_operator(after,np.zeros(after.ndof),elastic_constant=e0,elastic_inverse_scale=e1)
    if endpoint.maximum_elastic_principal_strain>model.elastic_strain_limit:raise ValueError('Inherited elastic strain limit')
    residual=basal.mass_matrix@(v-parent.velocity_m_s)/dt+basal.basal_drag_matrix@v+endpoint.force-basal.external_force_n
    relative=float(np.linalg.norm(residual)/max(np.linalg.norm(endpoint.force),np.linalg.norm(basal.external_force_n),1.))
    if relative>model.contact_parameters.equilibrium_tolerance:raise ValueError('Inherited momentum tolerance')
    memory=BEMemory(parent.mechanics.source_identity,thermal.state.time_myr,parent.mechanics.accepted_steps+1,
        tuple(VolumeCohortMemory(int(cid),loading.temperatures_k[i],e0[i],e1[i]) for i,cid in enumerate(after.cohort_ids)),L['elastic_after_j'])
    certificate=model._certificate(parent,after,parent.displacement_m+du) if certify else None
    state=BEState(thermal.geometry,thermal.state,thermal.history,memory,history,parent.displacement_m+du,v,certificate,L,dict(L))
    inputs=model._audit_input(parent,after,loading,du,basal)
    return state,dict(inputs=inputs,momentum_relative_residual=relative,maximum_elastic_principal_strain=endpoint.maximum_elastic_principal_strain)


def transition(model,parent):
    # This adapter is intentionally single-step and refuses BE continuation.
    if isinstance(parent,BEState):raise ValueError('Single candidate adapter cannot extend BE history')
    model.validate_state(parent);gate(parent.fault_history,model.fault_parameters)
    thermal=prepare_volume_thermal_step(parent.thermal,parent.history,parent.geometry,.0001,displacement_m=parent.displacement_m)
    no_remelt(parent.geometry,thermal.geometry)
    retained=retained_live_fraction(parent.geometry,thermal.geometry);d0=parent.fault_history.damage*retained
    h0=replace(parent.fault_history,time_myr=thermal.state.time_myr,accepted_steps=parent.fault_history.accepted_steps+1,damage=d0)
    predictor,pred_report=trial(model,parent,thermal,d0,h0,False)
    driver=model.face_response(predictor);vertices=_vertices(predictor)
    _,surface=temperatures(np.asarray(thermal.state.global_state.energy),thermal.state.source_boundary_parameters.thermal)
    ocean=diagnose(thermal.state.global_state,thermal.state.source_boundary_parameters.thermal)['ocean_fraction']
    faults,evolution=evaluate_volume_fault_evolution(parent.fault_history,model.source_packet,
        after_time_myr=thermal.state.time_myr,physical_vertices_m=vertices,face_stress_pa=driver['stress_pa'],
        face_mean_temperature_k=driver['mean_temperature_k'],retained_fraction=retained,active_material=driver['active_material'],
        surface_temperature_k=surface,ocean_fraction=ocean,orbit_before=parent.thermal.orbit,orbit_after=thermal.state.orbit,
        weak_plane_slip_available=True)
    gate(faults,model.fault_parameters) # stop before any activated/final solve
    final,report=trial(model,parent,thermal,faults.damage,faults,True)
    tensile=principal_tensile(model.face_response(final)['stress_pa'])
    final=replace(final,fault_history=replace(faults,peak_tensile_pa=np.maximum(faults.peak_tensile_pa,tensile)))
    gate(final.fault_history,model.fault_parameters)
    driver_payload=dict(physical_vertices_m=vertices,stress_pa=driver['stress_pa'],mean_temperature_k=driver['mean_temperature_k'],
        retained_fraction=retained,active_material=driver['active_material'],surface_temperature_k=float(surface),ocean_fraction=float(ocean))
    return final,dict(final=report,predictor=pred_report,predictor_state=state_payload(predictor),
        predictor_driver=_jsonable(driver_payload),fault_evolution=_jsonable(evolution),final_face_tensile_pa=tensile.tolist())


def state_payload(s):
    return _jsonable(dict(geometry=_geometry_claims(s.geometry),thermal=thermal_to_dict(s.thermal,include_source=False),
        history=bulk_birth_to_dict(s.history),mechanics=asdict(s.mechanics),fault_history=volume_fault_history_to_dict(s.fault_history),
        displacement_m=s.displacement_m,velocity_m_s=s.velocity_m_s,last_certificate=None if s.last_certificate is None else cone_homothetic_to_dict(s.last_certificate),
        interval_ledger=s.interval_ledger,be_accumulated_ledger=s.be_accumulated_ledger,inherited_parent_sha256=s.inherited_parent_sha256,scheme=s.scheme))


def encode(model,state,report):
    return dict(format=FORMAT,parent_sha256=PARENT_SHA,source_packet=model.source_packet,state=state_payload(state),report=report)


def dumps(packet):return json.dumps(packet,sort_keys=True,indent=2,allow_nan=False)+'\n'
