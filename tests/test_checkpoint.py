from pathlib import Path
import numpy as np
from tectonics.checkpoint import RunCheckpoint,save_checkpoint,load_checkpoint
from tectonics.continental import initialize_continental_cycle
from tectonics.lithosphere import initialize_lithosphere,refresh_mechanical_lithosphere
from tectonics.mesh import build_icosphere
from tectonics.plates import random_plate_system,PlateSystem
from tectonics.thermal import ThermalParameters,initialize_thermal_state
from tectonics.topography import TopographyParameters,initialize_topography
from tectonics.lithosphere import boundary_records_for_state
from tectonics.topology import PlateTopologyManager,PlateTopologyParameters
from tectonics.cratons import CratonParameters,initialize_craton_memory
from tectonics.plumes import MantlePlumeParameters,initialize_mantle_plumes
from tectonics.plume_rifting import initialize_plume_rifting
from tectonics.plume_dynamic_topography import initialize_plume_dynamic_topography


def test_checkpoint_roundtrip(tmp_path: Path):
    mesh=build_icosphere(2)
    plates=random_plate_system(mesh,6,123,0.2,0.1,0.4)
    state=initialize_lithosphere(mesh,plates,0.2,3)
    cycle=initialize_continental_cycle(mesh)
    thermal=initialize_thermal_state(0.5,5287.0,7.12,ThermalParameters())
    system=PlateSystem(cell_plate=state.cell_plate.copy(),plates=plates.plates)
    bounds=boundary_records_for_state(mesh,state,system,5287.0,4.0,1.0)
    topo=initialize_topography(mesh,state,bounds,TopographyParameters())
    manager=PlateTopologyManager(PlateTopologyParameters())
    manager.collision_age_myr[(1,2)]=12.0;manager.quiet_weld_age_myr[(1,2)]=4.0;manager.small_plate_age_myr[2]=8.0;manager.last_split_time_myr=4.0
    cp=RunCheckpoint(state,cycle,thermal,topo,system,system,manager,0.2,123.0,[{'time_myr':4.0}],[],[],[],[],[{'kind':'x'}])
    cp.arc_rows=[{'time_myr':4.0,'active_arc_zones':3,'mean_trench_arc_distance_km':110.0}]
    save_checkpoint(tmp_path/'cp',cp)
    manager2=PlateTopologyManager(PlateTopologyParameters())
    got=load_checkpoint(tmp_path/'cp',manager2)
    assert np.array_equal(got.state.cell_plate,state.cell_plate)
    assert np.allclose(got.state.rift_extension,state.rift_extension)
    assert np.allclose(got.state.extension_age_myr,state.extension_age_myr)
    assert np.allclose(got.topo.elevation_m,topo.elevation_m)
    assert np.allclose(got.cycle.felsic_potential,cycle.felsic_potential)
    assert got.thermal.mantle_temperature_k==thermal.mantle_temperature_k
    assert len(got.system.plates)==len(system.plates)
    assert np.allclose(got.system.plates[0].euler_axis,system.plates[0].euler_axis)
    assert manager2.collision_age_myr=={(1,2):12.0}
    assert manager2.quiet_weld_age_myr=={(1,2):4.0}
    assert manager2.small_plate_age_myr=={2:8.0}
    assert manager2.last_split_time_myr==4.0
    assert got.events==[{'kind':'x'}]
    assert got.arc_rows==cp.arc_rows


def test_checkpoint_roundtrip_reconstructed_transport_and_mantle(tmp_path: Path):
    from tectonics.mantle import initialize_mantle_flow
    from tectonics.transport import initialize_transport_state, quaternion_from_axis_angle
    mesh=build_icosphere(1)
    plates=random_plate_system(mesh,4,321,0.2,0.1,0.4)
    state=initialize_lithosphere(mesh,plates,0.25,2)
    initialize_craton_memory(mesh,state,5287.0,CratonParameters())
    refresh_mechanical_lithosphere(state,0.0)
    cycle=initialize_continental_cycle(mesh)
    thermal=initialize_thermal_state(0.5,5287.0,7.12,ThermalParameters())
    system=PlateSystem(cell_plate=state.cell_plate.copy(),plates=plates.plates)
    bounds=boundary_records_for_state(mesh,state,system,5287.0,4.0,1.0)
    topo=initialize_topography(mesh,state,bounds,TopographyParameters())
    manager=PlateTopologyManager(PlateTopologyParameters())
    mantle=initialize_mantle_flow(mesh,system)
    transport=initialize_transport_state(len(system.plates))
    transport.residual_quaternions[0]=quaternion_from_axis_angle(np.array([0.,0.,1.]),np.deg2rad(0.4))
    transport.hold_age_myr[0]=12.0; transport.cumulative_commit_count=7; transport.max_hold_age_myr=16.0
    cp=RunCheckpoint(state,cycle,thermal,topo,system,system,manager,0.25,123.0,[],[],[],[],[],[],[],mantle,transport)
    cp.craton_rows=[{'time_myr':0.0,'mean_craton_strength':float(np.mean(state.craton_strength))}]
    cp.plume_state=initialize_mantle_plumes(mesh,0.0,MantlePlumeParameters(seed=322,initial_plume_count=2))
    cp.plume_state.ages_myr[:]=12.0
    cp.plume_state.last_flux[:]=np.linspace(0.0,1.0,mesh.cell_count)
    cp.plume_state.cumulative_exposure_myr[:]=2.0*cp.plume_state.last_flux
    cp.plume_state.cumulative_root_erosion_km[:]=0.5*cp.plume_state.last_flux
    cp.plume_rows=[{'time_myr':0.0,'active_plume_count':2}]
    cp.plume_rifting_state=initialize_plume_rifting(mesh,0.0)
    cp.plume_rifting_state.last_extension_forcing[:]=np.linspace(0.0,0.8,mesh.cell_count)
    cp.plume_rifting_state.cumulative_extension_impulse_myr[:]=12.0*cp.plume_rifting_state.last_extension_forcing
    cp.plume_rifting_state.last_dynamic_uplift_m[:]=1800.0*cp.plume_rifting_state.last_extension_forcing
    cp.plume_rifting_state.last_magmatic_productivity[:]=cp.plume_rifting_state.last_extension_forcing**1.4
    cp.plume_rifting_rows=[{'time_myr':0.0,'max_surface_extension_forcing':0.8}]
    cp.plume_dynamic_topography_state=initialize_plume_dynamic_topography(mesh,0.0)
    cp.plume_dynamic_topography_state.target_dynamic_topography_m[:]=np.linspace(-40.0,900.0,mesh.cell_count)
    cp.plume_dynamic_topography_state.realized_dynamic_topography_m[:]=0.7*cp.plume_dynamic_topography_state.target_dynamic_topography_m
    cp.plume_dynamic_topography_state.cumulative_positive_support_m_myr[:]=20.0*np.maximum(cp.plume_dynamic_topography_state.realized_dynamic_topography_m,0.0)
    cp.plume_dynamic_topography_rows=[{'time_myr':0.0,'maximum_realized_uplift_m':630.0}]
    save_checkpoint(tmp_path/'cp2',cp)
    got=load_checkpoint(tmp_path/'cp2',PlateTopologyManager(PlateTopologyParameters()))
    assert got.mantle_flow is not None and got.transport_state is not None
    assert np.allclose(got.mantle_flow.cell_omega_rad_per_myr,mantle.cell_omega_rad_per_myr)
    assert np.allclose(got.transport_state.residual_quaternions,transport.residual_quaternions)
    assert np.allclose(got.transport_state.hold_age_myr,transport.hold_age_myr)
    assert got.transport_state.cumulative_commit_count==7
    assert got.state.continental_fraction is not None
    assert got.state.continental_volume_km3 is not None
    assert np.allclose(got.state.continental_fraction,state.continental_fraction)
    assert np.allclose(got.state.continental_volume_km3,state.continental_volume_km3)
    assert got.state.mantle_lithosphere_thickness_km is not None
    assert got.state.mantle_lithosphere_density_anomaly_kg_m3 is not None
    assert np.array_equal(got.state.mantle_lithosphere_thickness_km,state.mantle_lithosphere_thickness_km)
    assert np.array_equal(got.state.mantle_lithosphere_density_anomaly_kg_m3,state.mantle_lithosphere_density_anomaly_kg_m3)
    assert got.state.continental_lithosphere_age_myr is not None
    assert got.state.mantle_depletion_fraction is not None
    assert got.state.craton_strength is not None
    assert np.array_equal(got.state.continental_lithosphere_age_myr,state.continental_lithosphere_age_myr)
    assert np.array_equal(got.state.mantle_depletion_fraction,state.mantle_depletion_fraction)
    assert np.array_equal(got.state.craton_strength,state.craton_strength)
    assert got.craton_rows==cp.craton_rows
    assert got.plume_state is not None
    assert np.array_equal(got.plume_state.centers_unit,cp.plume_state.centers_unit)
    assert np.array_equal(got.plume_state.ages_myr,cp.plume_state.ages_myr)
    assert np.array_equal(got.plume_state.last_flux,cp.plume_state.last_flux)
    assert np.array_equal(got.plume_state.cumulative_exposure_myr,cp.plume_state.cumulative_exposure_myr)
    assert np.array_equal(got.plume_state.cumulative_root_erosion_km,cp.plume_state.cumulative_root_erosion_km)
    assert got.plume_state.next_plume_id==cp.plume_state.next_plume_id
    assert got.plume_state.next_birth_time_myr==cp.plume_state.next_birth_time_myr
    assert got.plume_rows==cp.plume_rows
    assert got.plume_rifting_state is not None
    assert np.array_equal(got.plume_rifting_state.last_extension_forcing,cp.plume_rifting_state.last_extension_forcing)
    assert np.array_equal(got.plume_rifting_state.cumulative_extension_impulse_myr,cp.plume_rifting_state.cumulative_extension_impulse_myr)
    assert np.array_equal(got.plume_rifting_state.last_dynamic_uplift_m,cp.plume_rifting_state.last_dynamic_uplift_m)
    assert np.array_equal(got.plume_rifting_state.last_magmatic_productivity,cp.plume_rifting_state.last_magmatic_productivity)
    assert got.plume_rifting_rows==cp.plume_rifting_rows
    assert got.plume_dynamic_topography_state is not None
    assert np.array_equal(got.plume_dynamic_topography_state.target_dynamic_topography_m,cp.plume_dynamic_topography_state.target_dynamic_topography_m)
    assert np.array_equal(got.plume_dynamic_topography_state.realized_dynamic_topography_m,cp.plume_dynamic_topography_state.realized_dynamic_topography_m)
    assert np.array_equal(got.plume_dynamic_topography_state.cumulative_positive_support_m_myr,cp.plume_dynamic_topography_state.cumulative_positive_support_m_myr)
    assert got.plume_dynamic_topography_rows==cp.plume_dynamic_topography_rows
