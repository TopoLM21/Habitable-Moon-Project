"""Saved-result audit: direct spatial quadrature, cellwise heat and total budget."""
import math
import numpy as np
from analysis.genesis_be_candidate.audit import block,close
from analysis.genesis_volume_continuation_validation import independent_audit as q
from analysis.genesis_volume_fault_validation import independent_audit as fault


def audit_feedback(root,thermal,packet):
    before=root['runtime_state'];s=packet['state'];g=s['geometry'];inputs=packet['inputs'];source=root['source_packet'];shell=source['source']['parameters']['shell'];tp=source['source']['parameters']['thermal']
    X=np.array(g['vertices_m']);faces=np.array(g['faces']);mass=np.array(g['column_mass_kg']);rho=g['density_kg_m3']
    old={x['id']:x for x in before['mechanics']['state']['cohorts']};new={x['id']:x for x in s['mechanics']['cohorts']}
    assert len(old)==len(new)==800 and old.keys()==new.keys() and g==before['geometry']
    entries={x['material_id']:x for x in packet['exchanges']};assert len(entries)==len(packet['exchanges'])==800
    du=np.array(s['displacement_m'])-before['displacement_m'];totalQ=[];totalW=[]
    for i,c in enumerate(g['cohorts']):
        cid=c['id'];f=c['face_index'];triangle=X[faces[f]];frame=fault.physical_face_geometry(X,faces[f][None,:])[0][0]
        C=q.isotropic_matrix(shell['young_modulus_pa'],shell['poisson_ratio'],s['fault_history']['damage'][f],shell['residual_stiffness'])
        dofs=(6*faces[f,:,None]+np.arange(6)).ravel()
        values,_,_=block(triangle,frame,mass[f],rho,c,C,old[cid]['strain0'],old[cid]['strain1'],new[cid]['strain0'],new[cid]['strain1'],np.array(inputs['thermal0'][i]),inputs['maxwell_ratio'][i],du[dofs])
        e=entries[cid];cell=e['cutcell'];assert thermal['fixed_cut_geometry'][cell]['material_id']==cid
        close(thermal['mass_kg'][cell],c['mass_kg'],'material mass',rtol=3e-12,atol=0)
        assert e['thermal_entry_j']==-e['mechanical_entry_j']
        close(e['maxwell_heat_j'],values['physical_maxwell_j'],'local Maxwell',rtol=3e-8,atol=1e3)
        close(e['thermal_work_to_mechanics_j'],values['thermal_work_j'],'local Wthermal',rtol=3e-8,atol=1e3)
        assert e['thermal_entry_j']==e['maxwell_heat_j']-e['thermal_work_to_mechanics_j']
        totalQ.append(e['maxwell_heat_j']);totalW.append(e['thermal_work_to_mechanics_j'])
        expected=np.zeros((3,6));expected[:,:3]=shell['linear_expansion_per_k']*(new[cid]['temperature_k']-old[cid]['temperature_k'])
        close(inputs['thermal0'][i],expected,'thermal strain from material temperature',rtol=2e-7,atol=2e-13)
    L=s['interval_ledger'];Q=math.fsum(totalQ);W=math.fsum(totalW)
    close(Q,L['physical_maxwell_j'],'Maxwell sum',rtol=3e-12,atol=1e3);close(W,L['thermal_work_j'],'thermal work sum',rtol=3e-12,atol=1e3)
    base=np.array(thermal['energy_j']);E=np.array(packet['thermal']['energy_j']);m=np.array(packet['thermal']['mass_kg']);T=np.array(packet['thermal']['temperature_k'])
    assert np.array_equal(m,thermal['mass_kg']);delta=E.astype(np.longdouble)-base.astype(np.longdouble)
    desired=np.zeros(len(E),dtype=np.longdouble)
    for e in entries.values():desired[e['cutcell']]=np.longdouble(e['maxwell_heat_j'])-np.longdouble(e['thermal_work_to_mechanics_j'])
    allowance=4*(np.spacing(E)+np.spacing(base));assert np.all(np.abs(delta-desired)<=allowance)
    # Unowned material, deep and water are not mechanical heat recipients.
    untouched=[i for i in range(len(E)) if i not in {e['cutcell'] for e in entries.values()}]
    assert np.array_equal(E[untouched],base[untouched])
    phi=np.clip((T-tp['solidus_k'])/(tp['liquidus_k']-tp['solidus_k']),0,1)
    h=tp['silicate_heat_capacity_j_kg_k']*T+tp['silicate_latent_heat_j_kg']*phi
    h[-1]=tp['water_heat_capacity_j_kg_k']*T[-1]+tp['water_latent_heat_j_kg']
    assert np.isfinite(E).all() and np.all(E>=0)
    assert np.array_equal(phi[:-1],packet['thermal']['phase_fraction'])
    assert T.min()>0 and T[-1]>647.096
    assert np.all(np.abs(m*h-E)<=8*np.spacing(E))
    for cid,e in entries.items():
        i=e['cutcell'];assert phi[i]==0
        assert abs(T[i]-new[cid]['temperature_k'])<=1e-8
    storage=packet['storage'];assert storage['interface_drag_energy_j']==L['physical_drag_j']>=0
    assert storage['damage_energy_j']==-L['damage_energy_change_j']>=0
    assert storage['numerical_sink_j']==L['numerical_kinetic_j']+L['numerical_memory_j']>=0
    # Sum changes, not enormous thermal endpoint totals. External work is open.
    terms=[L['kinetic_after_j'],-L['kinetic_before_j'],L['elastic_after_j'],-L['elastic_before_j'],storage['interface_drag_energy_j'],storage['damage_energy_j'],storage['numerical_sink_j'],-L['external_work_j']]
    residual=float(delta.sum()+sum(np.longdouble(v) for v in terms))
    touched=[e['cutcell'] for e in entries.values()]
    budget=float(allowance[touched].sum())+5e-10*max(abs(v) for v in L.values())
    assert abs(residual)<=budget
    cert=q.geometry_binding(s['last_certificate'],g['vertices_m'],g['faces'],before['displacement_m'],s['displacement_m'],before['geometry']['cohorts']+g['cohorts'],g['column_mass_kg'],g['density_kg_m3'],before['thermal']['time_myr'],s['thermal']['time_myr'])
    return dict(status='PASS',cohort_quadratures=800,maxwell_j=Q,thermal_work_j=W,heat_increment_j=float(delta.sum()),local_heat_max_residual_j=float(np.max(np.abs(delta-desired))),combined_balance_residual_j=residual,combined_allowance_j=budget,geometry=cert)
