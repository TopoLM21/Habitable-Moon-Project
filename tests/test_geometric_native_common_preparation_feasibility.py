"""Whole-support relaxation, literal offsets and conditional work controls."""
from dataclasses import replace
from fractions import Fraction as F
import json
from pathlib import Path

import pytest

from tectonics import geometric_native_common_basal_work_bound as power
from tectonics import geometric_native_common_preparation_feasibility as api

ROOT = Path(__file__).resolve().parents[1]
PATHS = (
    'analysis/geometric_native_common_general_force_validation/development_native03/native_control.json',
    'analysis/geometric_native_common_cold_continuum_energy_validation/release_validation_01/native_control/source_packet.json',
    'analysis/geometric_native_common_cold_preparation_energy_validation/development_native01/native_control.json',
    'analysis/geometric_native_common_hot_mean_validation/release_validation_02/native_control/source_packet.json',
)


@pytest.fixture(scope='module')
def sources():
    return tuple(json.loads((ROOT/name).read_bytes()) for name in PATHS)


@pytest.fixture(scope='module')
def context(sources):
    return power.bind_native_common_preparation_work_source_context(*sources,
        provenance='Full four-source global relaxation theorem test')


@pytest.fixture(scope='module')
def power_system(context):
    return power.bind_native_common_basal_work_bound(context,
        provenance='Original68 necessary NET-work capacity')


@pytest.fixture(scope='module')
def system(context, power_system):
    return api.bind_native_common_preparation_feasibility(context, power_system,
        provenance='Global100-coordinate relaxed declared preparation floor')


def test_full_source_complement_and_projection_premises_are_authenticated(context, sources):
    evidence = power.native_common_preparation_work_source_context_evidence(context)
    e, t = evidence['energy_evidence'], evidence['theorem_evidence']
    assert len(e['original_parent_ids']) == len(e['internal_beams']) == 8
    assert len(t['complement_cell_receipts']) == 192
    assert len(t['original_internal_attachment_bindings']) == 32
    assert len(e['hot_bodies']) == 5 and len(e['contacts16']) == 16
    assert t['gram_inverse_certificate']['positive_definite_certified'] is True
    assert t['gram_inverse_certificate']['global_eight_tip_overlap_retained'] is True
    assert t['global_bound_relaxes_all_100_mechanical_coordinates'] is True
    assert t['donor_collar_excluded_from_positive_floor_but_retained_in_nonnegative_remainder'] is True
    assert t['frozen_owner_load_potential_used'] is False
    assert e['prepared_zero_hot_energy_j'] == [0,1]
    assert len({x['original_parent_id'] for x in t['authentic_clamp_support']}) == 8
    assert evidence['source_clocks_s'] == [[0,1],[2000000,1],[1000000,1],[1000000,1]]


def test_actual_literal_floor_caps_and_offsets_cannot_be_rounded_into_old_energies(context, system):
    e = power.native_common_preparation_work_source_context_evidence(context)['energy_evidence']
    literal_beams = sum((F(*v) for v in e['literal_reference_energies_j']),F(0))
    literal_contacts = sum((F(*a['literal_source_energy_j']) for a in e['contacts16']),F(0))
    assert system.prepared_zero_potential_j == literal_beams+literal_contacts
    assert system.prepared_zero_potential_j == F(4944229710564471659,16777216)
    # They happen to agree in printed decimals. Their actual binary64
    # ledger residuals still enter the work baseline and must remain intact.
    assert literal_beams != system.eight_beam_relaxation_cap_j
    assert literal_contacts != system.sixteen_contact_relaxation_cap_j
    response = api.evaluate_native_common_preparation_feasibility(system, duration_s=F(0))
    expected = F(59126726146071543873642448447068649967723382248352787074751633580314679518165164646098541053658769365439170459819661370325739479797,
        3177051841726052353433775962765353844787036973428597847065971776463557774879554006902544847656189952000000)
    assert response.global_relaxed_potential_increase_lower_j == expected
    assert F(18610563847125)*10**12 < expected < F(18610563847126)*10**12


def test_projection_removes_active_strain_but_cannot_remove_the_zero_modal_complement():
    # True two-dimensional projection: the first direction carries a tip;
    # the complementary direction has identically zero tip support.
    complement = F(17,3)
    for active in (F(0),F(3),F(-91),F(10**12)):
        norm = (active**2+complement**2)/2
        residual = norm-active**2/2
        assert residual == complement**2/2
        # A floor based on the unprojected WHOLE norm would be invalid:
        # all active strain can be removed by an internal tip coordinate.
        if active:
            assert residual < norm
    # If a nonzero mode is wrongly allowed on the complement, it can
    # cancel that strain. This is why the zero ψ/G/H premise is essential.
    assert (complement-complement)**2/2 < complement**2/2


def test_full_original_internal_cross_term_is_needed_to_complete_the_beam_square(context):
    e = power.native_common_preparation_work_source_context_evidence(context)['energy_evidence']
    for E0,f,K in zip(e['literal_reference_energies_j'], e['literal_reference_forces_n'],e['tip_stiffnesses_n_m']):
        E0,f,K = F(*E0),F(*f),F(*K)
        for p in (F(0), F(7,13), F(-111), F(10**9)):
            r = -f/K-p
            whole = E0+f*p+K*p*p/2+f*r+K*p*r+K*r*r/2
            assert whole == E0-f*f/(2*K)
            assert whole == E0+f*(p+r)+K*(p+r)**2/2
            if p:
                without_cross = E0+f*p+K*p*p/2+f*r+K*r*r/2
                assert without_cross != whole


def test_original_contact_energy_offset_is_retained_even_when_contact_potential_relaxes(context):
    e = power.native_common_preparation_work_source_context_evidence(context)['energy_evidence']
    for a in e['contacts16']:
        area,tau,K,literal = (F(*a[n]) for n in ('original_area_m2','literal_source_traction_pa',
            'series_stiffness_exact_pa_m','literal_source_energy_j'))
        anchor = tau/K
        initial_phi = K*anchor*anchor/2
        offset = literal-area*initial_phi
        assert offset+area*initial_phi == literal
        relaxed = offset  # Φ(0)=0, the maximal independent relaxation.
        assert relaxed-literal == -area*tau*tau/(2*K)
        if offset:
            assert relaxed != 0


def test_all_series_branches_are_nonnegative_including_compression_and_full_failure(context):
    e = power.native_common_preparation_work_source_context_evidence(context)['energy_evidence']
    for a in e['contacts16']:
        K,T,gamma = (F(*a[n]) for n in ('series_stiffness_exact_pa_m','tensile_strength_pa','fracture_energy_j_m2'))
        onset, failure = T/K,2*gamma/T
        assert onset > 0 and failure > onset
        for gap in (-1000,-1,0,onset/2,onset):
            assert K*gap*gap/2 >= 0
        for delta in (F(0),(failure-onset)/3,(failure-onset)/2,failure-onset):
            phi = T*onset/2+T*delta-T*delta*delta/(2*(failure-onset))
            assert phi >= 0
        assert T*onset/2+T*(failure-onset)/2 == gamma


def test_lambda_zero_relaxed_floor_stays_negative_and_duration_zero_is_only_necessary(system):
    r = api.evaluate_native_common_preparation_feasibility(system,F(0),duration_s=F(0))
    assert r.global_relaxed_potential_increase_lower_j == -system.eight_beam_relaxation_cap_j-system.sixteen_contact_relaxation_cap_j
    assert r.global_relaxed_potential_increase_lower_j < 0
    assert r.necessary_net_external_work_lower_j < 0
    assert r.necessary_minimum_duration_s == 0
    assert r.duration_satisfies_necessary_bound is True
    assert r.negative_lower_bound_replaced_by_zero_work_claim is False
    assert r.necessary_condition_is_sufficient is False


def test_parametric_minimum_duration_and_deficit_do_not_pay_preparation(system):
    initial = api.evaluate_native_common_preparation_feasibility(system,duration_s=F(0))
    minimum = initial.necessary_minimum_duration_s
    assert minimum == initial.global_relaxed_potential_increase_lower_j/system.maximum_net_power_w
    assert F(298037)*10**9 < minimum < F(298038)*10**9
    for T in (F(0),F(10**6),F(2*10**6),minimum/2,minimum,2*minimum):
        r = api.evaluate_native_common_preparation_feasibility(system,duration_s=T)
        assert r.maximum_net_basal_work_j == system.maximum_net_power_w*T
        assert r.necessary_work_deficit_lower_j == max(initial.global_relaxed_potential_increase_lower_j-r.maximum_net_basal_work_j,F(0))
        assert r.duration_satisfies_necessary_bound == (T >= minimum)
        assert r.full_system_preparation_work_certified is False
        assert r.reference_preparation_work_paid is False
        assert r.actual_equilibrium_or_preparation_path_computed is False
        assert r.clocks_advanced is False and r.accepted_new_native_intervals == 0


def test_hypothetical_initial_kinetic_budget_is_an_explicit_assumption_not_source_history(system):
    zero = api.evaluate_native_common_preparation_feasibility(system,duration_s=F(0))
    B = zero.global_relaxed_potential_increase_lower_j/2
    budget = api.evaluate_native_common_preparation_feasibility(system,duration_s=F(0),initial_kinetic_budget_j=B)
    assert budget.global_relaxed_potential_increase_lower_j == zero.global_relaxed_potential_increase_lower_j
    assert budget.necessary_net_external_work_lower_j == zero.global_relaxed_potential_increase_lower_j-B
    assert budget.necessary_minimum_duration_s == zero.necessary_minimum_duration_s/2
    assert budget.no_net_initial_kinetic_energy_release_required is False
    assert zero.no_net_initial_kinetic_energy_release_required is True
    assert budget.actual_initial_kinetic_energy_budget_known is False
    assert budget.initial_kinetic_budget_is_user_supplied_conditional_assumption is True
    assert budget.physical_inertial_extension_certified is False
    assert budget.supplied_initial_kinetic_release_budget_must_not_be_exceeded is True


@pytest.mark.parametrize('lam,T,B',[(True,F(1),F(0)),(F(-1),F(1),F(0)),(F(2),F(1),F(0)),
    (F(1),None,F(0)),(F(1),1,F(0)),(F(1),F(-1),F(0)),(F(1),F(1),True),(F(1),F(1),F(-1))])
def test_undefined_or_out_of_domain_parameter_types_fail_closed(system,lam,T,B):
    with pytest.raises(ValueError):
        api.evaluate_native_common_preparation_feasibility(system,lam,duration_s=T,initial_kinetic_budget_j=B)


def test_feasibility_and_response_restarts_reject_result_scope_and_bool_int_forgery(context,power_system,system):
    raw = api.native_common_preparation_feasibility_to_dict(system)
    assert api.native_common_preparation_feasibility_from_dict(raw,context,power_system) == system
    for field,value in [('inactive_complement_S2_lower_j',[0,1]),('mechanical_coordinate_count',True),
                        ('source_raw_pins',[]),('source_clocks_s',[[0,1]]*4)]:
        broken = dict(raw);broken[field]=value
        with pytest.raises(ValueError):
            api.native_common_preparation_feasibility_from_dict(broken,context,power_system)
    r = api.evaluate_native_common_preparation_feasibility(system,F(1,2),duration_s=F(10**6))
    dto = api.native_common_preparation_feasibility_response_to_dict(r)
    assert api.native_common_preparation_feasibility_response_from_dict(dto,system=system,
        expected_preparation_fraction=F(1,2),expected_duration_s=F(10**6)) == r
    for field,value in [('global_relaxed_potential_increase_lower_j',[0,1]),
                        ('actual_initial_kinetic_energy_budget_known',True),('accepted_new_native_intervals',False),
                        ('new_mixed_force_equilibrium_certified',True),('duration_s',[0,1])]:
        broken=dict(dto);broken[field]=value
        with pytest.raises(ValueError):
            api.native_common_preparation_feasibility_response_from_dict(broken,system=system,
                expected_preparation_fraction=F(1,2),expected_duration_s=F(10**6))
    forged = replace(r,new_mixed_force_equilibrium_certified=True)
    forged = replace(forged,binding_sha256=api._seal(forged))
    with pytest.raises(ValueError):
        api.native_common_preparation_feasibility_response_to_dict(forged)


@pytest.mark.parametrize('mode',['gram','stiffness','projection','original_internal_binding','contact_offset','hot_birth'])
def test_extractor_fails_closed_when_a_global_theorem_premise_is_removed(sources,mode):
    s935,s899,s923,s903 = sources
    # Shallow outer copies plus replacement of the one affected source path
    # avoid copying hundreds of MB and never mutate the actual frozen source.
    if mode == 'gram':
        s899 = dict(s899);qs=list(s899['evaluation_queries']);q=dict(qs[0]);r=dict(q['response'])
        g=dict(r['gram_inverse_certificate']);g['positive_definite_certified']=False
        r['gram_inverse_certificate']=g;q['response']=r;qs[0]=q;s899['evaluation_queries']=qs
    elif mode == 'stiffness':
        s935=dict(s935);c=dict(s935['cold_energy_system']);ks=list(c['tip_stiffnesses_n_m']);ks[0]=[0,1]
        c['tip_stiffnesses_n_m']=ks;s935['cold_energy_system']=c
    elif mode == 'hot_birth':
        s903=dict(s903);h=dict(s903['hot_mean_system']);bs=list(h['bodies']);b=dict(bs[0]);b['source_birth_radial_displacement_m']=[1,1]
        bs[0]=b;h['bodies']=bs;s903['hot_mean_system']=h
    else:
        s935=dict(s935);qs=list(s935['queries']);q=dict(qs[0]);r=dict(q['response'])
        if mode == 'projection':
            cold=json.loads(r['cold_response_json']);cell=next(x for x in cold['cell_receipts'] if x['kind']=='zero_modal_complement_preparation')
            cell['active_coordinate_indices']=[0];r['cold_response_json']=json.dumps(cold)
        elif mode == 'original_internal_binding':
            beams=list(r['internal_beam_receipts_json']);b=json.loads(beams[0]);b['global_internal_coordinate_index']=92
            beams[0]=json.dumps(b);r['internal_beam_receipts_json']=beams
        else:
            atoms=list(r['contact_atoms_json']);a=json.loads(atoms[0]);a['literal_source_energy_j']=[0,1]
            atoms[0]=json.dumps(a);r['contact_atoms_json']=atoms
        q['response']=r;qs[0]=q;s935['queries']=qs
    with pytest.raises(ValueError):
        api._extract_preparation_energy_evidence(s935,s899,s923,s903)
