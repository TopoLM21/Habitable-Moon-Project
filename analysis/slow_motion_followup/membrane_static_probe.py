"""Static diagnostic only: current-lid basal shear on saved plate domains."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tectonics.genesis_starter_continuation import _load_cp,load_starter_source
from tectonics.genesis_local_mechanics import refresh_young_material_mechanics
from tectonics.genesis_plate_membrane_diagnostic import solve_plate_basal_membrane
from tectonics.genesis_starter_fracture import YoungShellFracture
from tectonics.genesis import SECONDS_PER_MYR
from tectonics.genesis_shell import principal_tensile
from tectonics.basal_coupling import prescribed_vertex_traction
from tectonics.simulation import load_config

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('checkpoint',type=Path)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
cfg=load_config(args.checkpoint/'mature_config.yaml')
model,source,_=load_starter_source(args.checkpoint/'young_context/starter_checkpoint.npz')
cp=_load_cp(args.checkpoint/'mature_checkpoint',cfg)
saved=json.loads((args.checkpoint/'continuation.json').read_text(encoding='utf8'))
sample=model.loading.sample(source.thermal_context)
local=refresh_young_material_mechanics(cp.state,sample,model,origin_time_myr=saved['import']['origin_time_myr'])
depth=local['local_total_lid_thickness_km']
source_traction=prescribed_vertex_traction(model.mesh,seed=model.parameters.seed,
    convective_traction_pa=model.shell.convective_traction_pa,thickness_km=depth,
    traction_coupling_depth_km=model.shell.traction_coupling_depth_km)
omega=np.asarray([p.euler_axis*p.angular_speed_rad_per_myr for p in cp.system.plates])
velocity=np.cross(omega[cp.state.cell_plate,None,:],model.mesh.vertices[model.mesh.faces])*(model.thermal.radius_km*1000./SECONDS_PER_MYR)
traction=source_traction[model.mesh.faces]-model.parameters.basal_drag_pa_s_m*velocity
traction[depth==0.]=0.
result=solve_plate_basal_membrane(model.mesh,cp.state.cell_plate,traction,depth,
    radius_km=model.thermal.radius_km,young_modulus_pa=model.shell.young_modulus_pa,
    poisson_ratio=model.shell.poisson_ratio,small_strain_limit=model.shell.max_total_strain)
# Test the separation with saved actual velocities and the same source at zero
# rigid speed: their deforming stress should agree after torque projection.
unmoved=solve_plate_basal_membrane(model.mesh,cp.state.cell_plate,source_traction[model.mesh.faces],depth,
    radius_km=model.thermal.radius_km,young_modulus_pa=model.shell.young_modulus_pa,
    poisson_ratio=model.shell.poisson_ratio,small_strain_limit=model.shell.max_total_strain)
tensile=principal_tensile(result.stress_pa)
fracture=YoungShellFracture.load(model,args.checkpoint/'young_context/fracture_memory.npz')
area=model.areas
summary=dict(checkpoint=str(args.checkpoint.resolve()),time_myr=cp.state.time_myr,
    purpose='Static basal-only free-edge diagnostic at current lid, not a live Maxwell/damage result',
    stress_basis='face e1=vertex0-to-vertex1, e2=normal-cross-e1; sigma11,sigma22,sigma12 in Pa',
    excluded='Maxwell history, plasticity, ridge/slab/contact edge loads, transported tensor memory and damage evolution',
    max_tensile_mpa=float(tensile.max()/1e6),mean_tensile_mpa=float(area@tensile/area.sum()/1e6),
    area_above_current_tensile_strength=float(area@(tensile>fracture.memory.strength_pa)/area.sum()),
    max_principal_strain=result.max_principal_strain,within_small_strain_limit=result.within_small_strain_limit,
    elastic_energy_j=result.elastic_energy_j,balanced_external_work_j=result.balanced_external_work_j,
    work_balance_relative_error=float(abs(result.balanced_external_work_j-2*result.elastic_energy_j)/max(2*result.elastic_energy_j,1.)),
    stress_change_when_rigid_plate_speed_is_removed_pa=float(np.max(np.abs(result.stress_pa-unmoved.stress_pa))),
    components=result.components)
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
print(json.dumps(summary,indent=2))
