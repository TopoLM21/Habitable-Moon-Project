import math
from dataclasses import replace
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from tectonics.boundary_exchange import frozen_exchange,exchange_on_boundaries
from tectonics.kinematics import BoundaryRecord,BoundaryType
from tectonics.mesh import build_icosphere
from tectonics.plates import Plate,PlateSystem
from tectonics.lithosphere import LithosphereState,advance_lithosphere
from tectonics.transport import initialize_transport_state
from tectonics.tides import constant_eccentricity


@pytest.mark.parametrize('k',[0.,1e-12,.02,1.,100.])
def test_finite_donor_and_age_moment_against_independent_ode(k):
    v,m=30.,600.;s=8.;dt=4.
    final,aged,born,returned=frozen_exchange([v],[m],[s],[k],dt)
    ref=solve_ivp(lambda t,y:[s-k*y[0],y[0]-k*y[1]],(0,dt),[v,m],method='Radau',rtol=1e-11,atol=1e-11)
    assert final[0]==pytest.approx(ref.y[0,-1],rel=2e-9,abs=1e-10)
    assert aged[0]==pytest.approx(ref.y[1,-1],rel=2e-9,abs=1e-10)
    assert final[0]>=0 and aged[0]>=0 and 0<=returned[0]<=v+born[0]
    assert final[0]+returned[0]==pytest.approx(v+born[0],rel=2e-15)
    a,b=np.array([v]),np.array([m]);out=0.
    for _ in range(4):
        a,b,_,loss=frozen_exchange(a,b,[s],[k],1.);out+=loss[0]
    assert a==pytest.approx(final,rel=2e-15)
    assert b==pytest.approx(aged,rel=2e-15)
    assert out==pytest.approx(returned[0],rel=3e-15,abs=1e-25)


def ocean(mesh):
    n=mesh.cell_count;a=mesh.physical_cell_areas_km2(5287.)
    s=LithosphereState(0.,np.zeros(n,np.int32),np.zeros(n,np.int8),np.full(n,100.),np.full(n,7.),np.zeros(n))
    s.oceanic_volume_km3=a*7;s.oceanic_age_moment_km3_myr=a*700
    return s


def test_line_opening_flux_is_exact_on_same_physical_nested_boundary():
    base=build_icosphere(1);owner=(base.centroids[:,0]>0).astype(np.int32)
    values=[]
    for sub in (1,2,3):
        mesh=build_icosphere(sub);s=ocean(mesh);s.cell_plate=np.repeat(owner,4**(sub-1))
        edges=[];total_length=0.
        for a,b,u,v in mesh.shared_edges:
            if s.cell_plate[a]==s.cell_plate[b]:continue
            mid=mesh.vertices[u]+mesh.vertices[v];mid/=np.linalg.norm(mid)
            edges.append(BoundaryRecord(a,b,u,v,int(s.cell_plate[a]),int(s.cell_plate[b]),mid,10.,0.,10.,BoundaryType.DIVERGENT))
            total_length+=5287.*math.atan2(np.linalg.norm(np.cross(mesh.vertices[u],mesh.vertices[v])),mesh.vertices[u]@mesh.vertices[v])
        final,aged,born,returned=exchange_on_boundaries(mesh,s,edges,4.,5287.,7.)
        assert sum(born)==pytest.approx(total_length*10*7*4,rel=2e-14)
        assert np.all(returned==0)
        values.append(math.fsum(born))
    assert values==pytest.approx([values[0]]*3,rel=1e-14)


def test_raster_overlap_alone_cannot_recycle_or_create_basalt():
    mesh=build_icosphere(2);s=ocean(mesh);s.cell_plate[mesh.centroids[:,0]<0]=1
    system=PlateSystem(s.cell_plate.copy(),(Plate(0,0,np.array([0.,0.,1.]),.03),Plate(1,1,np.array([0.,0.,1.]),-.03)))
    n,_,_,d=advance_lithosphere(mesh,system,s,20.,5287.,7.12,47.,constant_eccentricity(0.),
        transport_state=initialize_transport_state(2),ocean_flux_boundaries=[])
    assert d.oceanic_created_volume_km3==d.oceanic_recycled_volume_km3==0.
    assert math.fsum(n.oceanic_volume_km3)==pytest.approx(math.fsum(s.oceanic_volume_km3),rel=1e-15)
    assert math.fsum(n.oceanic_age_moment_km3_myr)==pytest.approx(math.fsum(s.oceanic_age_moment_km3_myr)+20*math.fsum(s.oceanic_volume_km3),rel=1e-15)


def test_native_restart_keeps_integral_law_and_age_memory(tmp_path):
    from tectonics.checkpoint import RunCheckpoint,save_checkpoint,load_checkpoint
    from tectonics.continental import initialize_continental_cycle
    from tectonics.material_exchange import MaterialExchangeBudget
    from tectonics.topography import initialize_topography,TopographyParameters
    from tectonics.topology import PlateTopologyManager,PlateTopologyParameters
    from tectonics.thermal import initialize_thermal_state,ThermalParameters
    mesh=build_icosphere(1);s=ocean(mesh)
    # This fixture starts a NEW inventory; initialize() correctly refuses to
    # overwrite an already-budgeted ocean() fixture's extensive fields.
    s.oceanic_volume_km3=None;s.oceanic_age_moment_km3_myr=None
    s.continental_fraction=np.zeros(mesh.cell_count);s.continental_volume_km3=np.zeros(mesh.cell_count)
    system=PlateSystem(s.cell_plate.copy(),(Plate(0,0,np.array([0.,0.,1.]),.03),))
    tp=ThermalParameters();thermal=initialize_thermal_state(.5,5287.,7.12,tp)
    budget,s,thermal=MaterialExchangeBudget.initialize(mesh,s,thermal,.5,5287.,7.12,tp)
    s.oceanic_age_moment_km3_myr=s.oceanic_volume_km3*s.crust_age_myr
    manager=PlateTopologyManager(PlateTopologyParameters())
    cp=RunCheckpoint(s,initialize_continental_cycle(mesh),thermal,
        initialize_topography(mesh,s,[],TopographyParameters()),system,system,manager,0.,0.,[],[],[],[],[],[])
    cp.material_exchange=budget.checkpoint()
    cp.arc_integration={'version':1,'integral_ocean_fluxes':True,'initialization':'explicit synthetic age field'}
    save_checkpoint(tmp_path/'native',cp)
    restored=load_checkpoint(tmp_path/'native',PlateTopologyManager(PlateTopologyParameters()))
    assert np.array_equal(restored.state.oceanic_age_moment_km3_myr,s.oceanic_age_moment_km3_myr)
    assert restored.arc_integration==cp.arc_integration and restored.material_exchange==cp.material_exchange
    # A damaged copy must not silently reconstruct lost history from display age.
    with np.load(tmp_path/'native/state.npz') as z:arrays={k:z[k] for k in z.files if k!='oceanic_age_moment_km3_myr'}
    broken=tmp_path/'broken';broken.mkdir()
    np.savez_compressed(broken/'state.npz',**arrays)
    (broken/'meta.json').write_text((tmp_path/'native/meta.json').read_text())
    with pytest.raises(ValueError,match='missing material age moment'):
        load_checkpoint(broken,PlateTopologyManager(PlateTopologyParameters()))
    cp.state.oceanic_age_moment_km3_myr=None
    with pytest.raises(ValueError,match='requires material age moment'):save_checkpoint(tmp_path/'invalid',cp)
    assert not (tmp_path/'invalid').exists()
