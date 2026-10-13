"""Artificial stress belt for positive-branch testing; never a real source input."""
import numpy as np
from tectonics.mesh import build_icosphere
from tectonics.simplified_genesis import SimplifiedGenesis,VERSION

def synthetic(mesh=None,axis=0,width=.12):
    mesh=build_icosphere(3) if mesh is None else mesh;n=mesh.cell_count;nodes=np.arange(2*n).reshape(n,2)
    phi=np.array([1.,-1.])/np.sqrt(3);P=np.column_stack((np.ones(2),phi));T=np.r_[np.full(2*n,1000.),990.,990.]
    mass=np.r_[np.full(2*n,1.5e15),1e20,1e20];G=np.diag([1e12,1e12/3]);weak=abs(mesh.centroids[:,axis])<width
    shell=dict(young_modulus_pa=1e9,poisson_ratio=.25,residual_stiffness=.02,linear_expansion_per_k=1e-5,viscosity_reference_pa_s=1e25,viscosity_reference_temperature_k=1000.,activation_energy_j_mol=0.,viscosity_min_pa_s=1e20,viscosity_max_pa_s=1e28,tensile_strength_pa=1e6,damage_timescale_myr=.01,cold_healing_timescale_myr=100.,hot_healing_timescale_myr=.01)
    edges=[(int(i),int(j),1e9) for i,j in nodes]+[(int(i),2*n,1e9) for i in nodes[:,0]]
    cfg=dict(mesh=dict(vertices=mesh.vertices.tolist(),faces=mesh.faces.tolist(),centroids=mesh.centroids.tolist(),areas=mesh.areas_unit_sphere.tolist(),neighbors=mesh.neighbors,shared_edges=mesh.shared_edges),mass=mass.tolist(),cp=np.full(len(T),1200.).tolist(),latent=np.zeros(len(T)).tolist(),offset=np.zeros(len(T)).tolist(),edges=edges,G=np.tile(G,(n,1,1)).tolist(),P=np.tile(P,(n,1,1)).tolist(),weights=np.full((n,2),5e11).tolist(),C=np.tile(np.eye(6)*1e9,(n,1,1)).tolist(),solid_nodes=nodes.tolist(),profile_limits=np.tile([[1.,1.],[1.,-1.]],(n,1,1)).tolist(),face_inertia=np.tile(np.eye(3)*1e29,(n,1,1)).tolist(),face_drag=np.tile(np.eye(3)*1e32,(n,1,1)).tolist(),face_torque=np.column_stack((np.zeros(n),np.zeros(n),1e18*mesh.centroids[:,axis])).tolist(),face_mass_kg=np.full(n,3e15).tolist(),shell=shell,faults=dict(enabled=True,activation_damage=.65,activation_persistence_myr=.000025),radius_km=5287.,regularization_km=0.,solidus=1400.,liquidus=2000.,source_mode='synthetic_prescribed_power',synthetic_power_w=np.zeros(len(T)).tolist(),selector=dict(radius_km=5287.,min_child_area_km2=3.1e6,min_band_span_km=1000.),max_step_years=25.,max_rotation_rad=1e-3,passive_reserve_j=12345.)
    memory=np.zeros((n,2,6));memory[weak,0,:2]=.03
    s=dict(format=VERSION,time_myr=0.,energy=(mass*1200*T).tolist(),memory=memory.tolist(),damage=np.zeros(n).tolist(),candidate_age=np.zeros(n).tolist(),consumed=np.zeros(n,bool).tolist(),owner=np.zeros(n,int).tolist(),omega=[[0.,0.,0.]],rotation=[np.eye(3).tolist()],membrane_established=True,passive_reserve_j=12345.,events=[],steps=0)
    return SimplifiedGenesis(cfg,s)



def hemisphere_synthetic():
    """Manufactured equatorial stress belt on an octahedral grid; one owner.

    No daughter map is supplied. Damage and the unchanged selector create it.
    This geometry admits exact frictionless sliding unlike the old jagged cut.
    """
    from tectonics.mesh import _subdivide
    from analysis.genesis_starter_selector_bridge.adapter import mesh_view
    v=np.array([[1,0,0],[0,1,0],[-1,0,0],[0,-1,0],[0,0,1],[0,0,-1]],float)
    f=np.array([[4,i,(i+1)%4] for i in range(4)]+[[5,(i+1)%4,i] for i in range(4)])
    for _ in range(3):v,f=_subdivide(v,f)
    return synthetic(mesh_view(v,f),axis=2,width=.2)
