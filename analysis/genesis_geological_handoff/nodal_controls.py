"""Frozen-geometry operator controls, no planetary time advancement."""
from copy import deepcopy
import json,time
from pathlib import Path
import numpy as np
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.simplified_genesis_nodal import blocks,operators,material_mass,mechanical,capture
from analysis.genesis_aggregate_coupled.core import step


def run(path):
    start=time.monotonic();m=SimplifiedGenesis.load(path);o=operators(m);M=material_mass(m)
    T=m.temperature(m.state['energy']);D=np.array(m.state['damage']);dt=25*31557600.
    a=mechanical(m,T,D,T,dt,M)
    face=int(D.argmax());control=D.copy();control[face]=D.mean()
    b=mechanical(m,T,control,T,dt,M)
    strain=lambda r:np.array([x['L']@r['displacement_increment'][x['dofs']] for x in r['blocks']])
    delta=np.linalg.norm(strain(a)-strain(b),axis=1);neighbors=np.array(m.mesh.neighbors[face])
    assert np.max(delta[neighbors])>1e-12
    # Free body: no external traction or drag, rest, initially unstressed;
    # inertia retained. Compare with every displacement DOF clamped (L=0).
    cold=np.full(len(T),500.);previous=cold+1.;bb=blocks(m,cold,D,previous)
    for block in bb:block['old'][:]=0.
    free=step(M,o['D']*0,o['F']*0,np.zeros(len(M)),1e6,bb)
    fixed=deepcopy(bb)
    for block in fixed:block['L'][:]=0.
    clamped=step(M,o['D']*0,o['F']*0,np.zeros(len(M)),1e6,fixed)
    reaction=np.zeros(len(M))
    for block,e in zip(bb,clamped['memory']):
        np.add.at(reaction,block['dofs'],block['L'].T@block['H']@e)
    assert np.linalg.norm(reaction)>0
    assert reaction@clamped['displacement_increment']==0
    energy_ratio=free['ledger']['elastic_after']/clamped['ledger']['elastic_after']
    assert energy_ratio<1e-4
    radial=lambda du:float(np.mean(np.einsum('vi,vi->v',du.reshape(-1,6)[:,:3],o['X'])/np.sum(o['X']**2,axis=1)))
    contraction=radial(free['displacement_increment'])
    assert abs(contraction+m.shell['linear_expansion_per_k'])<1e-3*m.shell['linear_expansion_per_k']
    assert np.max(abs(clamped['displacement_increment']))==0.
    dragged=step(M,o['D'],o['F']*0,np.zeros(len(M)),1e6,bb)
    # Capture independent of heat/phase: incoming material at rest.
    v,heat,error=capture(M,1.1*M,a['velocity'])
    assert heat>=0 and error<1e-10
    return dict(status='PASS',seconds=time.monotonic()-start,source=path,existing_damage_range=[float(D.min()),float(D.max())],changed_face_control_only=face,
        neighbor_strain_response_max=float(delta[neighbors].max()),far_strain_response_max=float(np.max(np.delete(delta,[face,*neighbors]))),
        real_operator_momentum_relative=a['ledger']['momentum_relative'],real_operator_energy_relative=a['ledger']['balance_relative'],
        free_cooling_radial_strain=contraction,expected_free_strain=-m.shell['linear_expansion_per_k'],free_to_clamped_elastic_energy=energy_ratio,
        free_energy_relative=free['ledger']['balance_relative'],free_momentum_relative=free['ledger']['momentum_relative'],clamped_energy_relative=clamped['ledger']['balance_relative'],clamped_support_reaction_norm_n=float(np.linalg.norm(reaction)),clamped_support_work_j=0.,capture_heat_j=heat,capture_momentum_relative=error,
        same_uniform_cooling_with_inherited_basal_drag=dict(radial_strain=radial(dragged['displacement_increment']),traction_zero=True,drag_unchanged=True,interpretation='Inherited basal drag strongly retards the uniform cooling response on this frozen 1e6-second control; not a new evolution or chosen parameter.'),
        limitation='Frozen operator controls; no evolved damage or guaranteed localization; free control removes traction/drag, clamp suppresses all displacements.')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    result=run(args.source);Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
