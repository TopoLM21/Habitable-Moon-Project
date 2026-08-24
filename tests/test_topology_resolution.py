import numpy as np
from tectonics.lithosphere import CrustType,LithosphereState,initialize_lithosphere
from tectonics.mesh import build_icosphere
from tectonics.plates import Plate,PlateSystem
from tectonics.topology import PlateTopologyParameters,_attempt_disconnected_split,_attempt_split,_smallest_doomed_plate
R=5287.0
def sys(owner):
 p=[]
 for pid in range(int(owner.max())+1): p.append(Plate(pid,int(np.flatnonzero(owner==pid)[0]),np.array([0.,0.,1.]),np.deg2rad(.2+.03*pid)))
 return PlateSystem(owner.copy(),tuple(p))
def state(owner):
 n=len(owner);return LithosphereState(100.,owner.copy(),np.ones(n,dtype=np.int8),np.ones(n)*500,np.ones(n)*35,np.zeros(n),np.zeros(n),np.zeros(n))
def test_physical_microplate_area_threshold_is_resolution_independent():
 for s in (3,4,5):
  m=build_icosphere(s);o=np.zeros(m.cell_count,dtype=np.int32);o[m.centroids[:,2]>.975]=1
  assert _smallest_doomed_plate(m,sys(o),PlateTopologyParameters(min_plate_area_km2=5e6,min_plate_cells=1),R)==1
def test_physical_disconnect_area_threshold_is_resolution_independent():
 for s in (3,4,5):
  m=build_icosphere(s);o=np.ones(m.cell_count,dtype=np.int32);o[np.abs(m.centroids[:,2])>.94]=0
  out,e=_attempt_disconnected_split(m,state(o),sys(o),PlateTopologyParameters(disconnect_min_child_area_km2=8e6,disconnect_min_child_cells=10**7),R)
  assert out is not None and e is not None and len(out.plates)==3
def test_physical_rift_split_across_resolutions():
 for s in (3,4,5):
  m=build_icosphere(s);o=np.zeros(m.cell_count,dtype=np.int32);sy=sys(o);st=initialize_lithosphere(m,sy,continental_fraction=.8,continental_nuclei=1,radius_km=R);cut=np.abs(m.centroids[:,0])<.12
  st.crust_type[cut]=int(CrustType.OCEANIC);st.crust_age_myr[cut]=2.;st.crust_thickness_km[cut]=7.;st.rift_extension[cut]=1.
  out,e=_attempt_split(m,st,sy,PlateTopologyParameters(split_min_postrift_extension=.7,split_min_rift_span_km=1000,split_min_child_area_km2=3.1e6,split_min_rift_cells=10**7,split_min_child_cells=10**7),R)
  assert out is not None and e is not None and len(out.plates)==2

def test_physical_collision_weld_clock_is_resolution_independent():
    from tectonics.kinematics import BoundaryRecord,BoundaryType
    from tectonics.topology import PlateTopologyManager
    for s in (3,4,5):
        m=build_icosphere(s);o=(m.centroids[:,0]>0).astype(np.int32);sy=sys(o);st=state(o)
        def records(speed,normal,kind):
            out=[]
            for fa,fb,u,v in m.shared_edges:
                pa=int(o[fa]);pb=int(o[fb])
                if pa==pb:continue
                mid=m.vertices[u]+m.vertices[v];mid/=np.linalg.norm(mid)
                out.append(BoundaryRecord(fa,fb,u,v,pa,pb,mid,float(normal),0.0,float(speed),kind))
            return out
        conv=records(30.0,-20.0,BoundaryType.CONVERGENT)
        quiet=records(2.0,0.0,BoundaryType.INACTIVE)
        mgr=PlateTopologyManager(PlateTopologyParameters(split_enabled=False,min_plate_area_km2=1.0,merge_min_continental_boundary_km=900.0,collision_coupling_start_myr=1e9,weld_min_collision_age_myr=40.0,weld_quiet_persistence_myr=20.0,weld_max_relative_speed_km_per_myr=5.0,weld_max_normal_divergence_km_per_myr=1.0))
        for _ in range(10): sy,_,ev=mgr.update(m,st,sy,conv,R,4.0); assert not any(e.kind=='merge' for e in ev)
        for _ in range(4): sy,_,ev=mgr.update(m,st,sy,quiet,R,4.0); assert not any(e.kind=='merge' for e in ev)
        sy,_,ev=mgr.update(m,st,sy,quiet,R,4.0)
        assert len(sy.plates)==1 and any(e.kind=='merge' for e in ev)
