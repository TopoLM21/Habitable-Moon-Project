"""One instantaneous saved-state quadrature, no evolution or Starter initialization."""
import argparse,json,time
from pathlib import Path
import numpy as np
from analysis.genesis_aggregate_projection.projection import moments,thermal_bins
from analysis.genesis_volume_continuation_validation import independent_audit as q
from analysis.genesis_volume_fault_validation.independent_audit import physical_face_geometry
from tectonics.genesis import GenesisParameters
from tectonics.genesis_shell import rock_temperature
from tectonics.genesis_starter import StarterParameters
from tectonics.genesis_transfer_bridge import digest,encoded

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    start=time.perf_counter()
    paths=[Path('/workspace/genesis-exploratory-100-01/continuation/initial.json'),Path('/workspace/genesis-exploratory-100-01/continuation/half2.json'),ROOT/'analysis/genesis_volume_creation_validation/evidence/actual_continuation/direct_after_second.json']
    pins=['4c0427d81bcb4b2a1dd8df2021e5ae1f6fedf983a9b8b2c165d69128a716bc95','20be57d6bae5255b3d440ea77a9da086bb32ecac06c38ada97d337ff1b0e59b9','62579a0bad128830a66e2f44579cf2b1f5d3835b691225b8d431102f3e4486d7']
    raw=[path.read_bytes() for path in paths]
    for data,pin in zip(raw,pins): assert digest(data)==pin
    packet=json.loads(raw[0])['packet'];cuts=json.loads(raw[1])['packet']['thermal_packet']['fixed_cut_geometry']
    source=json.loads(raw[2])['source_packet'];params=source['source']['parameters'];shell=params['shell']
    s=packet['state'];g=s['geometry'];X=np.array(g['vertices_m']);faces=np.array(g['faces']);nf=len(faces)
    frames,areas_km2=physical_face_geometry(X,faces);area=areas_km2*1e6
    rows={c['id']:c for c in s['mechanics']['cohorts']};N,wa=q.angular_rule();interp=N@np.linalg.inv(q.SOURCE_ANGULAR_POINTS)
    results=[];F=np.zeros(X.size*2);Fp=F.copy();constant_energy=0.
    for f in range(nf):
        pieces=[];C=q.isotropic_matrix(shell['young_modulus_pa'],shell['poisson_ratio'],s['fault_history']['damage'][f],shell['residual_stiffness'])
        B0,B1=q.independent_strain_operators(X[faces[f]],frames[f],N)
        height=float(X[faces[f,0]]@frames[f,:,2])
        for c in g['cohorts']:
            if c['face_index']!=f: continue
            rad,wr=q.radial_rule(c['q_lo'],c['live_q_hi']);lam,_=q.conical_scales(X[faces[f]],g['column_mass_kg'][f],g['density_kg_m3'],rad)
            e=rows[c['id']];strain=(interp@np.array(e['strain0']))[:,None,:]+(interp@np.array(e['strain1']))[:,None,:]/lam[None,:,None]
            w=(c['mass_kg']/g['density_kg_m3']*wa[:,None]*wr[None,:]).ravel()
            z=np.broadcast_to(height*(lam-1),(len(wa),len(wr))).ravel()
            B=B0[:,None,:,:]+B1[:,None,:,:]/lam[None,:,None,None]
            pieces.append((w,z,strain.reshape(-1,6),B.reshape(-1,6,18)))
        w=np.concatenate([r[0] for r in pieces]);z=np.concatenate([r[1] for r in pieces]);e=np.concatenate([r[2] for r in pieces]);B=np.concatenate([r[3] for r in pieces])
        r=moments(w,z,e,C);stress=e@C;fitstress=r['fit']@C
        local=np.einsum('n,nij,ni->j',w,B,stress);localp=np.einsum('n,nij,ni->j',w,B,fitstress)
        dofs=(faces[f,:,None]*6+np.arange(6)).ravel();np.add.at(F,dofs,local);np.add.at(Fp,dofs,localp)
        meanstress=C@r['mean'];membrane=w.sum()/area[f]*meanstress
        bending=np.einsum('n,n,ni->i',w,z-r['center'],stress)/area[f]
        meanstrain=np.sum(w[:,None]*e,axis=0)/w.sum();constant_energy+=float(.5*w.sum()*meanstrain@C@meanstrain)
        results.append(dict(face=f,reference_area_m2=area[f],solid_effective_thickness_m=float(w.sum()/area[f]),
            centroid_depth_m=-r['center'],mean_stress_pa=meanstress.tolist(),stress_slope_pa_per_m=(C@r['slope']).tolist(),
            generalized_membrane_n_per_m=membrane.tolist(),generalized_bending_n=bending.tolist(),
            elastic_source_j=r['energy_before'],elastic_retained_j=r['energy_retained'],elastic_residual_j=r['energy_residual'],
            projection_energy_balance_j=r['energy_balance'],
            virtual_constant_work_residual_j=float(np.linalg.norm(C@r['zeroth_error'])),
            virtual_linear_work_residual_j_m=float(np.linalg.norm(C@r['first_error']))))
    m=np.array(packet['thermal']['mass_kg']);H=np.array(packet['thermal']['energy_j']);nc=len(cuts)
    groups=np.array([c['layer'] for c in cuts]);thermal=thermal_bins(m[:nc],H[:nc],groups,32)
    thermo=GenesisParameters(**params['thermal']);T=rock_temperature(H[:nc]/m[:nc],thermo);Tc=rock_temperature(thermal['specific_enthalpy'],thermo)
    protected_labels=[(c['face'],c['layer'],c['material_id'] is not None) for c in cuts]
    unique=sorted(set(protected_labels));lookup={k:i for i,k in enumerate(unique)};pg=np.array([lookup[k] for k in protected_labels]);protected=thermal_bins(m[:nc],H[:nc],pg,len(unique))
    Tp=rock_temperature(protected['specific_enthalpy'],thermo)
    sourceU=sum(r['elastic_source_j'] for r in results);retained=sum(r['elastic_retained_j'] for r in results);residual=sum(r['elastic_residual_j'] for r in results)
    starter=StarterParameters()
    result=dict(status='PHYSICAL_REDUCTION_DECISION_NOT_ACTIVE_ADAPTER',new_time_steps=0,
        input_sha256={str(path):pin for path,pin in zip(paths,pins)},source_time_myr=s['thermal']['time_myr'],
        stress_projection=dict(elastic_source_quadrature_j=sourceU,elastic_source_stored_j=s['mechanics']['elastic_energy_j'],
            elastic_retained_j=retained,elastic_unrepresented_j=residual,unrepresented_fraction=residual/sourceU,
            membrane_only_unrepresented_fraction=1-constant_energy/sourceU,
            quadrature_vs_stored_relative=abs(sourceU-s['mechanics']['elastic_energy_j'])/sourceU,
            assembled_internal_force_relative_l2=float(np.linalg.norm(F-Fp)/np.linalg.norm(F)),
            energy_split_relative=abs(sourceU-retained-residual)/sourceU,faces=results),
        thermal_projection=dict(original_cut_count=nc,single_column_bins=32,phase_protected_face_bins=len(unique),
            column_mass_kg=thermal['mass'].tolist(),column_enthalpy_j=thermal['enthalpy'].tolist(),column_temperature_k=Tc.tolist(),
            global_column_mass_relative_error=thermal['mass_residual']/m[:nc].sum(),global_column_energy_relative_error=thermal['energy_residual']/abs(H[:nc].sum()),
            single_column_mass_rms_temperature_error_k=float(np.sqrt(np.sum(m[:nc]*(T-Tc[groups])**2)/m[:nc].sum())),
            protected_mass_rms_temperature_error_k=float(np.sqrt(np.sum(m[:nc]*(T-Tp[pg])**2)/m[:nc].sum())),
            source_mass_below_solidus_kg=float(m[:nc][T<=thermo.solidus_k].sum()),
            single_column_mass_below_solidus_kg=float(m[:nc][Tc[groups]<=thermo.solidus_k].sum()),
            protected_mass_below_solidus_kg=float(m[:nc][Tp[pg]<=thermo.solidus_k].sum()),
            protected_mass_relative_error=protected['mass_residual']/m[:nc].sum(),protected_energy_relative_error=protected['energy_residual']/abs(H[:nc].sum()),
            protected_labels=unique,protected_mass_kg=protected['mass'].tolist(),protected_enthalpy_j=protected['enthalpy'].tolist(),
            separate_deep_water_mass_kg=m[nc:].tolist(),separate_deep_water_enthalpy_j=H[nc:].tolist()),
        parameters=dict(source_shell=shell,source_onset=params['onset'],source_faults=params['faults'],
            starter_only=dict(strength_variation_fraction=starter.strength_variation_fraction,hot_strength_fraction=starter.hot_strength_fraction,
                damage_strength_reduction=starter.damage_strength_reduction,regularization_km=starter.regularization_km,
                rupture_damage=starter.rupture_damage,cooling_contrast_fraction=starter.cooling_contrast_fraction),
            traction_is_not_strength=True,adopt_50kPa=False),seconds=time.perf_counter()-start)
    assert all(path.read_bytes()==data for path,data in zip(paths,raw))
    with a.output.open('xb') as f:f.write(encoded(result))
    print(json.dumps({k:v for k,v in result.items() if k not in ['stress_projection','thermal_projection','parameters','input_sha256']},indent=2))
    print(json.dumps({k:v for k,v in result['stress_projection'].items() if k!='faces'},indent=2))
    print(json.dumps({k:v for k,v in result['thermal_projection'].items() if not isinstance(v,list)},indent=2))


if __name__=='__main__':main()
