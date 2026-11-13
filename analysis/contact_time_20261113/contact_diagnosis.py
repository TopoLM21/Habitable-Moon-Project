"""Independent material-bank geometry and coverage from existing native states."""
import json,math,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tectonics.mesh import build_icosphere
from tectonics.checkpoint import load_checkpoint
from tectonics.topology import PlateTopologyManager,PlateTopologyParameters
from tectonics.regional_area import AreaMaterialInventory,ActiveAreaParameters
from tectonics.regional_material import RegionalMaterialParameters
HERE=Path(__file__).resolve().parent;OUT=ROOT/'outputs_v131_contact_time';R=5287.


def load(name):
    cp=load_checkpoint(OUT/name/'restart',PlateTopologyManager(PlateTopologyParameters()))
    d=cp.regional_material
    inv=AreaMaterialInventory.restore(d,RegionalMaterialParameters(**d['parameters']),ActiveAreaParameters(**d['area_parameters']))
    return cp,inv


def coverage(inv,sub):
    mesh=build_icosphere(sub);count=np.zeros(mesh.cell_count,int)
    for owner,q in enumerate(inv.quaternions):
        points=Rotation.from_quat(q).inv().apply(mesh.centroids)
        root=inv.receiver.locate(points);count+=inv.coarse.arrays['owner'][root]==owner
    area=mesh.physical_cell_areas_km2(R);total=math.fsum(area)
    return dict(samples=mesh.cell_count,gap_fraction=math.fsum(area[count==0])/total,
                multiply_covered_area_fraction=math.fsum(area[count>1])/total,
                excess_coverage_fraction=math.fsum(area*np.maximum(count-1,0))/total,
                material_measure_sampling_residual=math.fsum(area*(count-1))/total),mesh,count


if __name__=='__main__':
    rows=[]
    for name in ('gate8','gate20','gate20_dt2','gate20_dt1'):
        cp,inv=load(name);reg=inv.material_boundary_registry()
        _,geometry=reg.evaluate(inv.quaternions,R)
        identity=Rotation.from_rotvec(np.zeros((len(inv.quaternions),3))).as_quat()
        _,initial=reg.require_shared_support(identity,R)
        samples=[coverage(inv,s)[0] for s in (2,3,4)]
        rows.append(dict(case=name,time_myr=cp.state.time_myr,initial_geometry=initial,geometry=geometry,
                         coverage_quadrature=samples,scope='coverage is independent quadrature, not an exact union or a mass debit'))
    result=dict(cases=rows,scope='bank contact registry is independent of receiver labels; geometric closure, pressure and gap matter remain missing')
    with (HERE/'contact_diagnosis.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
