import numpy as np

from tectonics.lithosphere import CrustType, LithosphereState
from tectonics.mesh import build_icosphere, connected_components
from tectonics.plates import Plate, PlateSystem
from tectonics.topology import PlateTopologyManager, PlateTopologyParameters, _attempt_split, _merge_pair


def _state(owner):
    n=len(owner)
    return LithosphereState(
        time_myr=50.0,
        cell_plate=np.asarray(owner,dtype=np.int32).copy(),
        crust_type=np.full(n,int(CrustType.CONTINENTAL),dtype=np.int8),
        crust_age_myr=np.full(n,500.0),
        crust_thickness_km=np.full(n,35.0),
        tidal_damage=np.zeros(n),
    )


def test_split_of_global_plate_across_weakened_great_circle_band():
    mesh=build_icosphere(3)
    owner=np.zeros(mesh.cell_count,dtype=np.int32)
    system=PlateSystem(owner.copy(),(Plate(0,0,np.array([0.,0.,1.]),np.deg2rad(0.25)),))
    state=_state(owner)
    # A young damaged oceanic band around x=0 forms a closed great-circle cut.
    cut=np.abs(mesh.centroids[:,0])<0.12
    state.crust_type[cut]=int(CrustType.OCEANIC)
    state.crust_age_myr[cut]=2.0
    state.crust_thickness_km[cut]=7.0
    state.tidal_damage[cut]=0.08
    params=PlateTopologyParameters(split_min_rift_cells=6,split_min_child_cells=100,split_differential_speed_deg_per_myr=0.05)
    out,event=_attempt_split(mesh,state,system,params,5287.0)
    assert out is not None and event is not None
    assert len(out.plates)==2
    assert event.kind=='split'
    for pid in range(2):
        cells=np.flatnonzero(out.cell_plate==pid)
        assert len(cells)>=100
        assert len(connected_components(cells,mesh.neighbors))==1


def test_merge_compacts_ids_and_area_weights_velocity():
    mesh=build_icosphere(2)
    owner=(mesh.centroids[:,0]>0).astype(np.int32)
    system=PlateSystem(owner.copy(),(
        Plate(0,int(np.flatnonzero(owner==0)[0]),np.array([0.,0.,1.]),np.deg2rad(0.2)),
        Plate(1,int(np.flatnonzero(owner==1)[0]),np.array([0.,0.,1.]),np.deg2rad(0.6)),
    ))
    state=_state(owner)
    out,event=_merge_pair(mesh,state,system,0,1,5287.0,'merge','test')
    assert len(out.plates)==1
    assert np.all(out.cell_plate==0)
    speed=np.rad2deg(out.plates[0].angular_speed_rad_per_myr)
    assert 0.2 < speed < 0.6
    assert event.parents==(0,1)


def test_tiny_plate_is_absorbed_by_manager():
    mesh=build_icosphere(2)
    owner=np.zeros(mesh.cell_count,dtype=np.int32)
    owner[0]=1
    system=PlateSystem(owner.copy(),(
        Plate(0,1,np.array([0.,0.,1.]),np.deg2rad(0.2)),
        Plate(1,0,np.array([0.,1.,0.]),np.deg2rad(0.3)),
    ))
    state=_state(owner)
    manager=PlateTopologyManager(PlateTopologyParameters(split_enabled=False,merge_enabled=False,min_plate_cells=5,max_events_per_step=2))
    out,diag,events=manager.update(mesh,state,system,[],5287.0,4.0)
    assert len(out.plates)==1
    assert diag.absorbed_small_plates==1
    assert diag.topology_changed
    assert events[0].kind=='absorb'
    assert np.all(state.cell_plate==out.cell_plate)




def test_zero_area_plate_is_explicitly_compacted_before_coupling():
    mesh=build_icosphere(2)
    # IDs 0 and 2 own cells; ID 1 has vanished completely.
    owner=np.where(mesh.centroids[:,0]>0,2,0).astype(np.int32)
    system=PlateSystem(owner.copy(),(
        Plate(0,int(np.flatnonzero(owner==0)[0]),np.array([0.,0.,1.]),np.deg2rad(0.2)),
        Plate(1,0,np.array([0.,1.,0.]),np.deg2rad(0.3)),
        Plate(2,int(np.flatnonzero(owner==2)[0]),np.array([1.,0.,0.]),np.deg2rad(0.4)),
    ))
    state=_state(owner)
    manager=PlateTopologyManager(PlateTopologyParameters(
        split_enabled=False,merge_enabled=False,min_plate_cells=1,
        min_plate_persistence_myr=20.0,max_events_per_step=2))
    out,diag,events=manager.update(mesh,state,system,[],5287.0,4.0)
    assert events and events[0].kind=='vanish'
    assert events[0].parents==(1,)
    assert len(out.plates)==2
    assert np.array_equal(np.unique(out.cell_plate),np.array([0,1],dtype=np.int32))
    assert np.array_equal(state.cell_plate,out.cell_plate)


def test_collision_coupling_never_changes_owner_ids_or_plate_count():
    mesh=build_icosphere(2)
    owner=(mesh.centroids[:,0]>0).astype(np.int32)
    system=PlateSystem(owner.copy(),(
        Plate(0,int(np.flatnonzero(owner==0)[0]),np.array([0.,0.,1.]),np.deg2rad(0.2)),
        Plate(1,int(np.flatnonzero(owner==1)[0]),np.array([0.,1.,0.]),np.deg2rad(0.3)),
    ))
    manager=PlateTopologyManager(PlateTopologyParameters())
    manager.collision_age_myr[(0,1)]=80.0
    out=manager._apply_collision_coupling(mesh,system,{(0,1):(1200.0,20.0,-10.0)},5287.0,4.0)
    assert len(out.plates)==len(system.plates)
    assert np.array_equal(out.cell_plate,system.cell_plate)

def test_tiny_plate_requires_persistence_before_absorption():
    mesh=build_icosphere(2)
    owner=np.zeros(mesh.cell_count,dtype=np.int32)
    owner[0]=1
    system=PlateSystem(owner.copy(),(
        Plate(0,1,np.array([0.,0.,1.]),np.deg2rad(0.2)),
        Plate(1,0,np.array([0.,1.,0.]),np.deg2rad(0.3)),
    ))
    state=_state(owner)
    manager=PlateTopologyManager(PlateTopologyParameters(
        split_enabled=False,merge_enabled=False,min_plate_cells=5,
        min_plate_persistence_myr=12.0,max_events_per_step=2))
    current=system
    # Two 4-Myr steps below the threshold are not enough.
    for _ in range(2):
        current,diag,events=manager.update(mesh,state,current,[],5287.0,4.0)
        assert not events
        assert len(current.plates)==2
        state.time_myr += 4.0
    # The third continuous step reaches 12 Myr and permits cleanup.
    current,diag,events=manager.update(mesh,state,current,[],5287.0,4.0)
    assert events and events[0].kind=='absorb'
    assert len(current.plates)==1


def test_microplate_persistence_resets_if_plate_recovers_area():
    mesh=build_icosphere(2)
    owner=np.zeros(mesh.cell_count,dtype=np.int32); owner[0]=1
    system=PlateSystem(owner.copy(),(
        Plate(0,1,np.array([0.,0.,1.]),np.deg2rad(0.2)),
        Plate(1,0,np.array([0.,1.,0.]),np.deg2rad(0.3)),
    ))
    state=_state(owner)
    manager=PlateTopologyManager(PlateTopologyParameters(
        split_enabled=False,merge_enabled=False,min_plate_cells=5,
        min_plate_persistence_myr=12.0,max_events_per_step=2))
    current,_,events=manager.update(mesh,state,system,[],5287.0,4.0)
    assert not events and manager.small_plate_age_myr.get(1)==4.0
    # Give plate 1 enough cells to rise above threshold for one step.
    recovered=current.cell_plate.copy(); recovered[:8]=1
    current=PlateSystem(recovered,current.plates); state.cell_plate=recovered.copy(); state.time_myr+=4.0
    current,_,events=manager.update(mesh,state,current,[],5287.0,4.0)
    assert not events and 1 not in manager.small_plate_age_myr

def test_no_event_keeps_topology_unchanged():
    mesh=build_icosphere(2)
    owner=(mesh.centroids[:,0]>0).astype(np.int32)
    system=PlateSystem(owner.copy(),(
        Plate(0,int(np.flatnonzero(owner==0)[0]),np.array([0.,0.,1.]),np.deg2rad(0.2)),
        Plate(1,int(np.flatnonzero(owner==1)[0]),np.array([0.,1.,0.]),np.deg2rad(0.3)),
    ))
    state=_state(owner)
    manager=PlateTopologyManager(PlateTopologyParameters(split_enabled=False,merge_enabled=False,min_plate_cells=1))
    out,diag,events=manager.update(mesh,state,system,[],5287.0,4.0)
    assert len(events)==0
    assert not diag.topology_changed
    assert np.array_equal(out.cell_plate,owner)


def test_macroscopic_disconnected_component_becomes_separate_plate():
    from tectonics.topology import _attempt_disconnected_split
    mesh = build_icosphere(3)
    # Plate 0 occupies two large caps separated by plate 1's equatorial belt.
    owner = np.ones(mesh.cell_count, dtype=np.int32)
    owner[np.abs(mesh.centroids[:, 2]) > 0.28] = 0
    system = PlateSystem(owner.copy(), (
        Plate(0, int(np.flatnonzero(owner == 0)[0]), np.array([0., 0., 1.]), np.deg2rad(0.2)),
        Plate(1, int(np.flatnonzero(owner == 1)[0]), np.array([0., 1., 0.]), np.deg2rad(0.3)),
    ))
    state = _state(owner)
    params = PlateTopologyParameters(disconnect_min_child_cells=100)
    out, event = _attempt_disconnected_split(mesh, state, system, params, 5287.0)
    assert out is not None and event is not None
    assert event.kind == 'disconnect_split'
    assert len(out.plates) == 3


def _continental_interface_records(mesh, owner, relative_speed=30.0, normal_rate=-20.0, kind=None):
    from tectonics.kinematics import BoundaryRecord, BoundaryType
    if kind is None:
        kind = BoundaryType.CONVERGENT
    records=[]
    for fa,fb,u,v in mesh.shared_edges:
        pa=int(owner[fa]); pb=int(owner[fb])
        if pa==pb:
            continue
        midpoint=mesh.vertices[u]+mesh.vertices[v]
        midpoint=midpoint/np.linalg.norm(midpoint)
        records.append(BoundaryRecord(
            face_a=fa,face_b=fb,vertex_u=u,vertex_v=v,
            plate_a=pa,plate_b=pb,midpoint=midpoint,
            normal_rate_km_per_myr=float(normal_rate),
            tangential_rate_km_per_myr=0.0,
            relative_speed_km_per_myr=float(relative_speed),
            boundary_type=kind,
        ))
    return records


def _two_hemisphere_system(mesh):
    owner=(mesh.centroids[:,0]>0).astype(np.int32)
    system=PlateSystem(owner.copy(),(
        Plate(0,int(np.flatnonzero(owner==0)[0]),np.array([0.,0.,1.]),np.deg2rad(0.2)),
        Plate(1,int(np.flatnonzero(owner==1)[0]),np.array([0.,0.,1.]),np.deg2rad(0.6)),
    ))
    return owner,system


def test_v095_collision_does_not_weld_after_legacy_twenty_myr():
    mesh=build_icosphere(2)
    owner,system=_two_hemisphere_system(mesh)
    state=_state(owner)
    records=_continental_interface_records(mesh,owner,relative_speed=30.0,normal_rate=-20.0)
    params=PlateTopologyParameters(
        split_enabled=False,min_plate_cells=1,
        merge_min_continental_boundary_km=100.0,
        weld_min_collision_age_myr=160.0,weld_quiet_persistence_myr=80.0,
    )
    manager=PlateTopologyManager(params)
    events=[]
    for _ in range(5):
        system,_,ev=manager.update(mesh,state,system,records,5287.0,4.0)
        events.extend(ev)
    assert len(system.plates)==2
    assert not any(e.kind=='merge' for e in events)
    assert max(manager.collision_age_myr.values())==20.0


def test_v095_weld_requires_separate_quiet_phase_after_mature_collision():
    from tectonics.kinematics import BoundaryType
    mesh=build_icosphere(2)
    owner,system=_two_hemisphere_system(mesh)
    state=_state(owner)
    convergent=_continental_interface_records(mesh,owner,relative_speed=30.0,normal_rate=-20.0)
    quiet=_continental_interface_records(mesh,owner,relative_speed=2.0,normal_rate=0.0,kind=BoundaryType.INACTIVE)
    params=PlateTopologyParameters(
        split_enabled=False,min_plate_cells=1,
        merge_min_continental_boundary_km=100.0,
        collision_coupling_start_myr=1e9,
        weld_min_collision_age_myr=40.0,
        weld_quiet_persistence_myr=20.0,
        weld_max_relative_speed_km_per_myr=5.0,
        weld_max_normal_divergence_km_per_myr=1.0,
    )
    manager=PlateTopologyManager(params)
    for _ in range(10):
        system,_,ev=manager.update(mesh,state,system,convergent,5287.0,4.0)
        assert not any(e.kind=='merge' for e in ev)
    # Four quiet steps = 16 Myr: still two plates.
    for _ in range(4):
        system,_,ev=manager.update(mesh,state,system,quiet,5287.0,4.0)
        assert not any(e.kind=='merge' for e in ev)
        assert len(system.plates)==2
    # Fifth quiet step completes the independent 20 Myr weld phase.
    system,_,ev=manager.update(mesh,state,system,quiet,5287.0,4.0)
    assert len(system.plates)==1
    assert any(e.kind=='merge' for e in ev)


def test_v095_collision_coupling_reduces_relative_euler_motion_without_merge():
    mesh=build_icosphere(2)
    owner,system=_two_hemisphere_system(mesh)
    state=_state(owner)
    records=_continental_interface_records(mesh,owner,relative_speed=30.0,normal_rate=-20.0)
    params=PlateTopologyParameters(
        split_enabled=False,min_plate_cells=1,
        merge_min_continental_boundary_km=100.0,
        collision_coupling_start_myr=0.0,
        collision_coupling_timescale_myr=20.0,
        collision_coupling_max_step_fraction=0.2,
        weld_min_collision_age_myr=1e9,
    )
    manager=PlateTopologyManager(params)
    before=abs(np.rad2deg(system.plates[1].angular_speed_rad_per_myr-system.plates[0].angular_speed_rad_per_myr))
    for _ in range(5):
        system,_,ev=manager.update(mesh,state,system,records,5287.0,4.0)
        assert not ev
        assert len(system.plates)==2
    after=abs(np.rad2deg(system.plates[1].angular_speed_rad_per_myr-system.plates[0].angular_speed_rad_per_myr))
    assert after < before


def test_v095_collision_memory_survives_unrelated_plate_id_compaction():
    mesh=build_icosphere(2)
    # Three longitudinal-ish regions; plate 2 will be absorbed into plate 0,
    # while collision memory between 0 and 1 should survive renumbering.
    owner=np.zeros(mesh.cell_count,dtype=np.int32)
    owner[mesh.centroids[:,0]>0.2]=1
    tiny=np.flatnonzero(owner==0)[:2]
    owner[tiny]=2
    system=PlateSystem(owner.copy(),(
        Plate(0,int(np.flatnonzero(owner==0)[0]),np.array([0.,0.,1.]),np.deg2rad(0.2)),
        Plate(1,int(np.flatnonzero(owner==1)[0]),np.array([0.,1.,0.]),np.deg2rad(0.3)),
        Plate(2,int(tiny[0]),np.array([1.,0.,0.]),np.deg2rad(0.1)),
    ))
    state=_state(owner)
    from tectonics.kinematics import BoundaryType
    manager=PlateTopologyManager(PlateTopologyParameters(split_enabled=False,merge_enabled=False,min_plate_cells=5,max_events_per_step=1,weld_min_collision_age_myr=100.0))
    manager.collision_age_myr[(0,1)]=120.0
    manager.quiet_weld_age_myr[(0,1)]=12.0
    quiet=_continental_interface_records(mesh,owner,relative_speed=2.0,normal_rate=0.0,kind=BoundaryType.INACTIVE)
    out,diag,events=manager.update(mesh,state,system,quiet,5287.0,4.0)
    assert events and events[0].kind=='absorb'
    assert len(out.plates)==2
    assert manager.collision_age_myr=={(0,1):124.0}
    assert manager.quiet_weld_age_myr=={(0,1):16.0}

def test_v096_collision_zone_builds_local_extension_suppression():
    mesh=build_icosphere(2)
    owner,system=_two_hemisphere_system(mesh)
    state=_state(owner)
    records=_continental_interface_records(mesh,owner,relative_speed=25.0,normal_rate=-15.0)
    params=PlateTopologyParameters(
        split_enabled=False,min_plate_cells=1,
        merge_min_continental_boundary_km=100.0,
        collision_rift_suppression_start_myr=8.0,
        collision_rift_suppression_max=0.8,
        weld_min_collision_age_myr=40.0,
    )
    manager=PlateTopologyManager(params)
    # Establish 12 Myr of persistent collision memory.
    for _ in range(3):
        system,_,_=manager.update(mesh,state,system,records,5287.0,4.0)
    field=manager.extension_suppression_field(mesh,state,records)
    assert float(np.max(field)) > 0.0
    boundary_cells={int(b.face_a) for b in records}|{int(b.face_b) for b in records}
    assert any(field[c] > 0.0 for c in boundary_cells)
    assert np.all((field >= 0.0) & (field <= 1.0))
