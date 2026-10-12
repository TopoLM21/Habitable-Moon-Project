from copy import deepcopy
import numpy as np
import pytest
from analysis.genesis_backward_euler_prototype.core import Memory,step
from analysis.genesis_be_time_control.chain import accumulate,seal
from analysis.genesis_be_time_control.audit import verify_sum,verify_link
from analysis.genesis_connected_step_validation.run import PARENT_SHA


def manufactured_chain():
    v=np.array([.3]);u=np.zeros(1);mass=np.ones((1,1));memory=np.array([.2]);previous=None;packets=[];audits=[]
    root={'runtime_state':{'kinetic_energy_j':.045,'mechanics':{'state':{'elastic_energy_j':.04}}}}
    for i in range(2):
        old=Memory(np.ones((1,1)),np.array([[2.]]),memory,np.zeros(1),.7)
        birth=np.array([[.2]])
        newborn=Memory(np.ones((1,1)),np.array([[.4]]),np.zeros(1),np.zeros(1),0.,True)
        rows=[old] if i==0 else [old,newborn]
        r=step(mass,[[.2]],[.1],u,v,.1,rows,birth if i else None)
        L={**r['ledger'],'damage_energy_change_j':0.,'remelt_kinetic_outflow_j':0.,'remelt_elastic_outflow_j':0.}
        c=accumulate(previous,L);packets.append({'state':{'be_accumulated_ledger':c}});audits.append({'final':{'ledger':L}})
        previous=c;u=r['displacement'];v=r['velocity'];memory=r['memory'][0]
    return root,packets,audits


def test_two_real_solves_accumulate_work_not_endpoint_energy():
    root,packets,audits=manufactured_chain();result=verify_sum(root,packets,audits)
    assert result['status']=='PASS'
    c=packets[-1]['state']['be_accumulated_ledger']
    assert c['elastic_before_j']==pytest.approx(.04)
    assert c['elastic_after_j']==audits[-1]['final']['ledger']['elastic_after_j']
    assert c['numerical_memory_j']==sum(a['final']['ledger']['numerical_memory_j'] for a in audits)


@pytest.mark.parametrize('key',['numerical_memory_j','birth_kinetic_inflow_j','elastic_after_j'])
def test_double_counted_or_lost_terms_fail(key):
    root,packets,audits=manufactured_chain();packets[-1]['state']['be_accumulated_ledger'][key]+=1
    with pytest.raises(ValueError):verify_sum(root,packets,audits)


def linked():
    root={'source_packet':{},'runtime_state':{'thermal':{'time_myr':.8684}}}
    p=seal(dict(format='genesis-be-chain-0.1',parent_sha256=PARENT_SHA,predecessor_sha256=PARENT_SHA,step_sizes_myr=[5e-5],source_packet={},
        state={'inherited_parent_sha256':PARENT_SHA,'thermal':{'time_myr':.86845}},report={}))
    return root,p


def test_wrong_prefix_resealed_still_fails():
    root,p=linked();assert verify_link(root,None,p)==5e-5
    p.pop('sha256');p['predecessor_sha256']='0'*64
    with pytest.raises(ValueError):verify_link(root,None,seal(p))


def test_100_plus_100_interval_rejected():
    root,p=linked();p.pop('sha256');p['step_sizes_myr']=[.0001,.0001]
    with pytest.raises(ValueError):verify_link(root,None,seal(p))


def test_restart_checks_intermediate_journal_even_when_final_is_correct():
    from analysis.genesis_be_time_control.audit import verify_prefixes
    root,packets,audits=manufactured_chain()
    verify_prefixes(root,packets,audits)
    packets[0]['state']['be_accumulated_ledger']['numerical_memory_j']+=1
    verify_sum(root,packets,audits) # endpoint alone cannot detect this
    with pytest.raises(ValueError):verify_prefixes(root,packets,audits)
