"""Opt-in shared nodal Maxwell moments on the inherited material chart.

One connected body only. A real partition requires an explicit future handoff;
this module neither cuts shared nodes nor develops a contact solver.
"""
from copy import deepcopy
from dataclasses import asdict,replace
import numpy as np
from analysis.genesis_aggregate_coupled.core import step
from analysis.genesis_volume_continuation_validation import independent_audit as q
from .genesis_volume_basal import local_consistent_mass_matrix
from .genesis import GenesisParameters,SECONDS_PER_MYR,vapor_column_kg_m2
from .genesis_tides import TidalParameters,TidalOrbitState,tidal_strain_cycle
from .genesis_thermal_ports import apply_ports
from .genesis_implicit_enthalpy import solve
from .genesis_shell import principal_tensile
from .genesis_starter_topology import select_starter_cut
from .plates import Plate,PlateSystem
from .experimental_edge_weakening import scale as effective_scale
from .simplified_genesis_phase import promote


def validate(m):
    c,s=m.config,m.state;n=6*m.mesh.vertex_count
    for k in ('nodal_velocity','nodal_displacement'):
        x=np.asarray(s[k])
        if x.shape!=(n,) or not np.isfinite(x).all():raise ValueError('NODAL_STATE_SHAPE')
    if len(s['omega'])!=1 or s['events'] or any(s['consumed']):raise ValueError('NODAL_SINGLE_BODY_ONLY')
    if np.asarray(c['nodal']['M']).shape!=(n,n):raise ValueError('NODAL_MASS_SHAPE')
    if c['passive_kinetic_projection_j']!=0:raise ValueError('NODAL_KINETIC_DOUBLE_COUNT')
    if 'experimental_edge_fracture_reserve_j' in s:
        reserve=s['experimental_edge_fracture_reserve_j']
        if not np.isfinite(reserve) or reserve<0:raise ValueError('EDGE_RESERVE_INVALID')
    if c.get('experimental_geological_edge_fatigue'):
        from .experimental_edge_weakening import face_damage
        face_damage(m.mesh,s.get('experimental_edge_fatigue',{}).get('damage',np.zeros(len(m.mesh.shared_edges))))


def operators(m):
    if hasattr(m,'_nodal_operators'):return m._nodal_operators
    c=m.config['nodal'];r={k:np.asarray(c[k]) for k in ('M','D','F','X','frames','L','face_D')}
    r['dofs']=(m.mesh.faces[:,:,None]*6+np.arange(6)).reshape(m.nf,18)
    r['births']={};m._nodal_operators=r;return r


def birth_operator(m,cell):
    o=operators(m);key=cell['id']
    if key not in o['births']:
        f=cell['face'];tri=o['X'][m.mesh.faces[f]];c=m.config['nodal']
        N,wa=q.angular_rule();rad,wr=q.radial_rule(cell['q_lo'],cell['q_hi'])
        lam,_=q.conical_scales(tri,c['column_mass'][f],c['density'],rad)
        B0,B1=q.independent_strain_operators(tri,o['frames'][f],N)
        L=np.einsum('a,aij->ij',wa,B0)+np.einsum('a,aij->ij',wa,B1)*float(wr@(1/lam))
        bounds,_=q.conical_scales(tri,c['column_mass'][f],c['density'],np.array([cell['q_lo'],cell['q_hi']]))
        M=local_consistent_mass_matrix(cell['mass_kg'],float(bounds[0]),float(bounds[1]))
        o['births'][key]=(L,M)
    return o['births'][key]


def material_mass(m):
    o=operators(m);M=o['M'].copy();catalog={x['id']:x for x in m.config['phase_cells']}
    for x in m.state.get('born_material',[]):
        _,local=birth_operator(m,catalog[x['cell_id']]);ids=o['dofs'][x['face']]
        M[np.ix_(ids,ids)]+=local
    return M


def rate(m,T):
    p=m.shell
    eta=np.clip(p['viscosity_reference_pa_s']*np.exp(np.clip(p['activation_energy_j_mol']/8.314462618*(1/T-1/p['viscosity_reference_temperature_k']),-60,60)),p['viscosity_min_pa_s'],p['viscosity_max_pa_s'])
    return p['young_modulus_pa']/eta


def blocks(m,T,D,oldT):
    o=operators(m);out=[];unit=np.array([1.,1.,1.,0.,0.,0.]);alpha=m.shell['linear_expansion_per_k']
    for f in range(m.nf):
        ids=m.solid_nodes[f];P=m.P[f];w=m.weights[f]
        R=np.linalg.solve(m.G[f],P.T@((w*rate(m,T[ids]))[:,None]*P))
        theta=np.linalg.solve(P,T[ids]-oldT[ids])[:,None]*alpha*unit
        out.append(dict(H=np.kron(m.G[f],m.C[f]*effective_scale(m,D)[f]),L=o['L'][f],R=np.kron(R,np.eye(6)),old=np.asarray(m.state['memory'][f]).ravel(),theta=theta.ravel(),dofs=o['dofs'][f]))
    catalog={x['id']:x for x in m.config['phase_cells']}
    for x in m.state.get('born_material',[]):
        f=x['face'];i=x['node'];L,_=birth_operator(m,catalog[x['cell_id']])
        out.append(dict(H=x['volume_m3']*m.C[f]*effective_scale(m,D)[f],L=L,R=np.eye(6)*rate(m,T[i]),old=np.asarray(x['memory']),theta=alpha*(T[i]-oldT[i])*unit,dofs=o['dofs'][f]))
    return out


def mechanical(m,T,D,oldT,dt,M,basal=None):
    o=operators(m);b=blocks(m,T,D,oldT)
    if basal is None and m.config['nodal'].get('moving_basal'):
        from .simplified_genesis_basal import assemble
        basal=assemble(m)
    r=step(M,o['D'] if basal is None else basal['D'],o['F'] if basal is None else basal['F'],np.asarray(m.state['nodal_velocity']),dt,b)
    if r['ledger']['momentum_relative']>1e-8 or r['ledger']['balance_relative']>5e-10:raise ValueError('NODAL_OPERATOR_BALANCE')
    sigma=np.zeros((m.nf,3));vol=m.G[:,0,0].copy();heat=np.zeros(len(T));dU=0.;exchange=np.zeros(2)
    for f in range(m.nf):
        ids=m.solid_nodes[f];e=m.P[f]@r['memory'][f].reshape(2,6);C=m.C[f]*effective_scale(m,D)[f];stress=e@C
        Q=dt*m.weights[f]*rate(m,T[ids])*np.einsum('ni,ni->n',stress,e)
        W=-m.weights[f]*m.shell['linear_expansion_per_k']*(T[ids]-oldT[ids])*stress[:,:3].sum(axis=1)
        old=m.P[f]@np.asarray(m.state['memory'][f]);delta=.5*m.weights[f]*np.einsum('ni,ij,nj->n',old,m.C[f]*(effective_scale(m,D)[f]-effective_scale(m,m.state['damage'])[f]),old)
        heat[ids]+=Q-W-delta;exchange+=[Q.sum(),W.sum()];dU+=delta.sum()
        sigma[f]=np.einsum('n,ni->i',m.weights[f],stress)[[0,1,3]]
    born=deepcopy(m.state.get('born_material',[]))
    for j,x in enumerate(born,m.nf):
        f=x['face'];i=x['node'];old=np.asarray(x['memory']);new=r['memory'][j];C=m.C[f]*effective_scale(m,D)[f]
        delta=.5*x['volume_m3']*old@(m.C[f]*(effective_scale(m,D)[f]-effective_scale(m,m.state['damage'])[f]))@old
        row=r['rows'][j];heat[i]+=row['maxwell']-row['thermal_work']-delta
        exchange+=[row['maxwell'],row['thermal_work']];dU+=delta
        sigma[f]+=x['volume_m3']*(C@new)[[0,1,3]];vol[f]+=x['volume_m3'];x['memory']=new.tolist()
    for val,key in zip(exchange,('maxwell','thermal_work')):
        if abs(val-r['ledger'][key])>1e3+3e-8*abs(val):raise ValueError('NODAL_NODE_WORK_IDENTITY')
    # Basal drag heat goes to the same live thermal supports as the inherited model.
    if basal is None:
        for f,ids in enumerate(o['dofs']):
            heat[m.solid_nodes[f]]+=dt*(r['velocity'][ids]@o['face_D'][f]@r['velocity'][ids])/2
    else:
        from .simplified_genesis_basal import budget
        exchange=budget(basal,r['velocity'],dt)
        np.add.at(heat,basal['geometry']['thermal_nodes'],[x['heat_j'] for x in exchange['rows']])
        r['basal_budget']={k:v for k,v in exchange.items() if k!='rows'}
    r.update(stress=sigma/vol[:,None],heat=heat,damage_energy_change=float(dU),born=born,blocks=b)
    return r


def capture(M,newM,v):
    """Quiescent melt capture: conserve full generalized momentum, dissipate K."""
    new=np.linalg.solve(newM,M@v);K0=.5*v@M@v;K1=.5*new@newM@new
    loss=float(K0-K1)
    if loss< -1e-12*max(abs(K0),1.):raise ValueError('NODAL_CAPTURE_ENERGY')
    error=float(np.linalg.norm(newM@new-M@v)/max(np.linalg.norm(M@v),1.))
    if error>1e-10:raise ValueError('NODAL_CAPTURE_MOMENTUM')
    return new,max(loss,0.),error


def advance(m,years):
    s=m.state;c=m.config;o=operators(m);nexts=deepcopy(s);M=material_mass(m)
    if s.get('experimental_edge_fatigue',{}).get('partition_owner') is not None:
        raise ValueError('EXPERIMENTAL_GEOLOGICAL_HANDOFF_PENDING')
    target=s['time_myr']+years/1e6;dt=(target-s['time_myr'])*SECONDS_PER_MYR
    E0=np.asarray(s['energy']);oldT=m.temperature(E0);oldD=np.asarray(s['damage'])
    basal=None
    if c['nodal'].get('moving_basal'):
        from .simplified_genesis_basal import assemble
        basal=assemble(m)
    sourced,orbit,ports=apply_ports(dict(time_myr=s['time_myr'],mass_kg=m.mass.tolist(),energy_j=E0.tolist()),GenesisParameters(**c['thermal_parameters']),TidalParameters(**c['tidal_parameters']),TidalOrbitState(**s['orbit']),target,allow_condensation=m.water_parameters is not None)
    thermal=solve(m.mass,np.asarray(sourced['energy_j'])/m.mass,m.cp,m.latent,np.full(len(E0),c['solidus']),np.full(len(E0),c['liquidus']),m.offset,c['edges'],dt,water_parameters=m.water_parameters)
    base=m.mass*thermal['specific_h'];T=m.temperature(base)
    tp=replace(TidalParameters(**c['tidal_parameters']),eccentricity=orbit.eccentricity,semimajor_axis_km=orbit.semimajor_axis_km)
    nu=m.shell['poisson_ratio'];plane=np.array([[1.,nu,0.],[nu,1.,0.],[0.,0.,(1-nu)/2]])/(1-nu**2)
    tide=np.einsum('ij,pfj->pfi',plane,tidal_strain_cycle(m.mesh,tp))*m.shell['young_modulus_pa']*effective_scale(m,oldD)[None,:,None]
    for iteration in range(8):
        pred=mechanical(m,T,oldD,oldT,dt,M,basal)
        tensile=np.array([principal_tensile(pred['stress']+t) for t in tide])
        load=np.mean(np.maximum(tensile/m.shell['tensile_strength_pa']-1,0)**2,axis=0)/m.shell['damage_timescale_myr']
        load=m.smoother.apply(load,np.ones(m.nf,bool));hot=np.clip((T[m.solid_nodes].mean(axis=1)-700)/650,0,1)
        heal=1/m.shell['cold_healing_timescale_myr']+hot**4/m.shell['hot_healing_timescale_myr'];rr=load+heal
        D=load/rr+(oldD-load/rr)*np.exp(-rr*years/1e6)
        r=mechanical(m,T,D,oldT,dt,M,basal);E=base+r['heat'];Tnew=m.temperature(E)
        if np.max(abs(Tnew-T))<1e-8:break
        T=Tnew
    else:raise ValueError('NODAL_THERMAL_CORRECTOR_LIMIT')
    m.phase_guard(Tnew)
    age=np.where(D>=m.faults['activation_damage'],np.asarray(s['candidate_age'])+years/1e6,0.)
    eligible=(D>=m.faults['activation_damage'])&(age>=m.faults['activation_persistence_myr'])
    if not m.faults['enabled'] or not s['membrane_established']:eligible[:]=False
    owner=np.asarray(s['owner']);system=PlateSystem(owner,(Plate(0,0,np.array([0.,0.,1.]),0.),))
    # The explicit option replaces only the onset selector. Existing mechanical
    # and thermal guards remain active; a certified event stops this nodal mode.
    cut=None if c.get('experimental_geological_edge_fatigue') else select_starter_cut(m.mesh,system,eligible,D,**c['selector'])
    if cut is not None:
        m.validate_cut_localization(owner,D,cut)
        raise ValueError('NODAL_PARTITION_REQUIRES_HANDOFF')
    u=np.asarray(s['nodal_displacement'])+r['displacement_increment'];v=r['velocity']
    du_total=u-np.asarray(c['nodal']['initial_displacement'])
    strains=[block['L']@du_total[block['dofs']] for block in r['blocks']]
    if max(float(np.max(abs(e))) for e in strains)>m.shell['max_total_strain']:
        raise ValueError('NODAL_SMALL_STRAIN_LIMIT')
    outer=du_total.reshape(-1,6)[:,:3];normal=o['X']/np.linalg.norm(o['X'],axis=1)[:,None]
    tangent=outer-normal*np.einsum('ni,ni->n',outer,normal)[:,None]
    if np.max(np.linalg.norm(tangent,axis=1))/(c['radius_km']*1000)>c['max_rotation_rad']:
        raise ValueError('NODAL_REFERENCE_CHART_LIMIT')
    nexts.update(time_myr=target,born_material=r['born'])
    # Reuse phase ID/mass/enthalpy accounting, with zero angular capture;
    # actual capture is performed below in all shared nodal degrees of freedom.
    _,phase_report,_=promote(c,s,nexts,oldT,Tnew,E,m.material_inertia(),np.zeros((1,3)))
    oldids={x['cell_id'] for x in s.get('born_material',[])};newborn=[x for x in nexts['born_material'] if x['cell_id'] not in oldids]
    oldvol=m.G[:,0,0].copy();newvol=np.zeros(m.nf)
    for x in s.get('born_material',[]):oldvol[x['face']]+=x['volume_m3']
    for x in newborn:newvol[x['face']]+=x['volume_m3']
    r['stress']*= (oldvol/(oldvol+newvol))[:,None]
    newM=M.copy();catalog={x['id']:x for x in c['phase_cells']}
    for x in newborn:
        _,local=birth_operator(m,catalog[x['cell_id']]);idx=o['dofs'][x['face']];newM[np.ix_(idx,idx)]+=local
    capture_heat=0.;capture_error=0.
    if newborn:
        v,capture_heat,capture_error=capture(M,newM,v);total=sum(x['mass_kg'] for x in newborn)
        for x in newborn:E[x['node']]+=capture_heat*x['mass_kg']/total
    phase_report.update(capture_heat_j=capture_heat,capture_nodal_momentum_relative=capture_error,load_bearing_mass_total_kg=sum(c['face_mass_kg'])+sum(x['mass_kg'] for x in nexts['born_material']),thermal_silicate_mass_kg=float(m.mass[:-2].sum()))
    phase_report['not_load_bearing_mass_kg']=phase_report['thermal_silicate_mass_kg']-phase_report['load_bearing_mass_total_kg']
    if phase_report['not_load_bearing_mass_kg']<0:raise ValueError('NODAL_MASS_DOUBLE_COUNT')
    finalT=m.temperature(E);m.phase_guard(finalT)
    if any(finalT[x['node']]>c['solidus'] for x in newborn):raise ValueError('CAPTURE_REHEATS_NEW_SOLID')
    L=deepcopy(r['ledger']);L['elastic_before']-=r['damage_energy_change'];L['kinetic_after']-=capture_heat
    if basal is not None:
        L['nodal_core_external_work_j']=L['external_work']
        L['external_work']=r['basal_budget']['physical_external_work_j']
        L['basal']=r['basal_budget']
    change=float(np.sum(E.astype(np.longdouble)-E0.astype(np.longdouble)))
    balance=change+L['elastic_after']-L['elastic_before']+L['kinetic_after']-L['kinetic_before']-ports['net_thermal_input_j']-L['external_work']+L['numerical_memory']+L['numerical_kinetic']
    allowance=float(np.sum(32*(np.spacing(E)+np.spacing(E0))))+1e-8*max(abs(L['thermal_work']),L['maxwell'],abs(L['external_work']),1.)
    if abs(balance)>allowance:raise ValueError('NODAL_WHOLE_ENERGY_BALANCE')
    L.update(phase=phase_report,whole_balance_j=balance,whole_allowance_j=allowance,moment_balance_relative=r['ledger']['balance_relative'],damage_energy_change=r['damage_energy_change'],source_energy=ports['net_thermal_input_j'],fracture_heat=0.,corrector_iterations=iteration+1)
    if m.water_parameters is not None:
        p=m.water_parameters
        vapor0=m.mass[-1]*vapor_column_kg_m2(oldT[-1],p)/p.water_column_kg_m2
        vapor1=m.mass[-1]*vapor_column_kg_m2(finalT[-1],p)/p.water_column_kg_m2
        sensible=m.mass[-1]*p.water_heat_capacity_j_kg_k*(finalT[-1]-oldT[-1]);latent=p.water_latent_heat_j_kg*(vapor1-vapor0)
        residual=float(E[-1]-E0[-1]-sensible-latent);wa=64*(abs(np.spacing(E[-1]))+abs(np.spacing(E0[-1])))
        if abs(residual)>wa:raise ValueError('WATER_ENTHALPY_IDENTITY')
        L['water']=dict(total_mass_kg=float(m.mass[-1]),vapor_mass_kg=float(vapor1),liquid_mass_kg=float(m.mass[-1]-vapor1),residual_j=residual,roundoff_allowance_j=float(wa),added_latent_heat_j=0.)
    # Domain omega remains a diagnostic rigid projection, not the strain driver.
    X=o['X'];J=np.zeros((len(v),3))
    for i,x in enumerate(X):
        cross=np.array([[0.,-x[2],x[1]],[x[2],0.,-x[0]],[-x[1],x[0],0.]])
        J[6*i:6*i+3]=-cross;J[6*i+3:6*i+6]=-cross
    omega=np.linalg.solve(J.T@newM@J,J.T@newM@v)
    nexts.update(energy=E.tolist(),memory=[e.reshape(2,6).tolist() for e in r['memory'][:m.nf]],damage=D.tolist(),candidate_age=age.tolist(),nodal_displacement=u.tolist(),nodal_velocity=v.tolist(),omega=[omega.tolist()],orbit=asdict(orbit),stress_pa=r['stress'].tolist(),last_ledger=L,steps=s['steps']+1)
    m.state=nexts
    try:
        if c.get('experimental_geological_edge_fatigue'):
            from .experimental_genesis_edge_rule import update
            update(m,s,years)
        m.validate()
    except Exception:m.state=s;raise
    return deepcopy(nexts)
