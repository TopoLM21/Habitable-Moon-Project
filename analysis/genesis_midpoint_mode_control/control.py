"""Manufactured pure-drag mode through UNMODIFIED production solve and update.

Only spatial/constitutive, thermal, and geometry interfaces are fixtures. The
matrix equation, equilibration, scipy solve, displacement/velocity update,
force residual, kinetic energy, drag work and balance run in VolumeModel.
No Genesis model is initialized, stepped, loaded or replayed.
"""
from contextlib import ExitStack
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
from scipy.sparse import csr_matrix
import tectonics.genesis_volume_dynamics as production


def ledger(time):
    return NS(time_myr=time,elastic_energy_j=0.,initial_energy_j=0.,thermal_work_j=0.,
              viscous_dissipation_j=0.,damage_energy_change_j=0.,remelt_energy_removed_j=0.)


class DragFixture(production.VolumeModel):
    def __init__(self,gamma):
        self.assembly=NS(ndof=1,cohort_ids=(),face_indices=np.array([],int),
                         dofs=np.empty((0,1),int),B0=np.empty((0,3,6,1)),B1=np.empty((0,3,6,1)))
        self.basal=NS(mass_matrix=csr_matrix([[1.]]),basal_drag_matrix=csr_matrix([[gamma]]),
                      external_force_n=np.zeros(1),total_live_mass_kg=1.)
        self.shell_parameters=None
        self.contact_parameters=NS(equilibrium_tolerance=1e-8)
        self.contact_parameters_json='{}';self.elastic_strain_limit=.03
    def _assembly(self,*args):return self.assembly,self.basal
    def _phase_kinetic_flux(self,*args):return 0.,0.,0.
    def _certificate(self,*args):return None
    def _audit_input(self,*args):return {}
    def validate_state(self,state):return state


def run_mode(gamma,dt_seconds,count=4,initial_velocity=1e-8):
    if gamma<=0 or dt_seconds<=0:raise ValueError('Positive drag and duration required')
    model=DragFixture(gamma);K0=.5*initial_velocity**2
    state=production.VolumeState(None,None,None,ledger(0.),np.zeros(1),np.array([initial_velocity]),'{}',K0,K0)
    zero=np.zeros((0,3,6));loading=NS(old0=zero)
    response=NS(strain0=zero,strain1=zero,mechanical_work_j=0.,thermal_work_j=0.,viscous_dissipation_j=0.)
    operator=NS(force=np.zeros(1),maximum_elastic_principal_strain=0.)
    # No patch to _mechanical_trial, _advance_trial, linalg.solve, M, D or update.
    replacements=dict(prepare_volume_loading=lambda *a,**k:loading,
        average_volume_loading_coefficients=lambda *a,**k:(zero,zero),
        average_volume_loading_stiffness_factors=lambda *a,**k:np.empty(0),
        evaluate_genesis_volume_operator=lambda *a,**k:operator,
        algorithmic_stiffness=lambda *a,**k:csr_matrix((1,1)),
        evaluate_volume_loading=lambda *a,**k:response)
    rows=[]
    with ExitStack() as stack:
        for name,value in replacements.items():stack.enter_context(patch.object(production,name,value))
        for i in range(1,count+1):
            t=i*dt_seconds/production.SECONDS_PER_MYR
            thermal=NS(geometry=None,state=NS(time_myr=t),history=None,
                       after_temperature_k=[],birth_temperature_k=[],radial_moments=None,
                       before_radial_moments=None,removed_radial_moments=None,before_temperature_k=[])
            with patch.object(production,'accept_volume_loading',return_value=ledger(t)):
                after=model._advance_trial(state,thermal,np.array([]),np.array([]),
                                           requested_dt_myr=dt_seconds/production.SECONDS_PER_MYR)
            dt=(t-state.time_myr)*production.SECONDS_PER_MYR
            v=float(after.velocity_m_s[0]);u=float(after.displacement_m[0]);z=gamma*dt
            expected_r=(2-z)/(2+z)
            mean=(u-float(state.displacement_m[0]))/dt
            rows.append(dict(step=i,time_seconds=t*production.SECONDS_PER_MYR,z=z,
                endpoint_velocity_over_initial=v/initial_velocity,
                analytical_velocity_over_initial=float(np.exp(-gamma*t*production.SECONDS_PER_MYR)),
                midpoint_amplification=expected_r,
                update_error_over_initial=(v-expected_r*state.velocity_m_s[0])/initial_velocity,
                displacement_m=u,exact_displacement_m=float(initial_velocity*(-np.expm1(-gamma*t*production.SECONDS_PER_MYR))/gamma),
                mean_velocity_over_initial=mean/initial_velocity,
                kinetic_over_initial=after.kinetic_energy_j/K0,
                cumulative_drag_over_initial=after.basal_dissipation_j/K0,
                normalized_balance_residual=after.global_balance_residual_j/K0,
                momentum_residual=after.last_report['momentum_equation_relative_residual']))
            state=after
    return dict(gamma_per_second=gamma,dt_seconds=dt_seconds,rows=rows)
