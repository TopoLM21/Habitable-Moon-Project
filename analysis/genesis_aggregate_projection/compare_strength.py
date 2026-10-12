"""Instantaneous law comparison; no damage update, initialization or new step."""
import json
from pathlib import Path
import numpy as np
from analysis.genesis_be_candidate.audit import face_driver
from tectonics.genesis_starter import StarterParameters
from tectonics.genesis_shell import smooth_anomaly
from tectonics.mesh import build_icosphere
from tectonics.genesis_transfer_bridge import digest,encoded

ROOT=Path(__file__).resolve().parents[2]


def main():
    path=Path('/workspace/genesis-exploratory-100-01/continuation/initial.json');raw=path.read_bytes()
    assert digest(raw)=='4c0427d81bcb4b2a1dd8df2021e5ae1f6fedf983a9b8b2c165d69128a716bc95'
    source_raw=(ROOT/'analysis/genesis_volume_creation_validation/evidence/actual_continuation/direct_after_second.json').read_bytes()
    assert digest(source_raw)=='62579a0bad128830a66e2f44579cf2b1f5d3835b691225b8d431102f3e4486d7'
    s=json.loads(raw)['packet']['state'];source=json.loads(source_raw)['source_packet'];params=source['source']['parameters']
    driver=face_driver(s,source);stress=driver['stress_pa'];T=driver['mean_temperature_k'];history=s['fault_history'];d=np.array(history['damage']);water=np.array(history['water_access'])
    p=StarterParameters();strength=params['shell']['tensile_strength_pa']
    source_strength=strength*(1-(1-params['onset']['wet_strength_fraction'])*water)
    factor=1+p.strength_variation_fraction*smooth_anomaly(build_icosphere(params['shell']['subdivisions']),p.seed+1)
    hot=np.clip((T-700)/650,0,1)
    starter_strength=strength*factor*(1-(1-p.wet_strength_fraction)*water)*(1-(1-p.hot_strength_fraction)*hot)*(1-p.damage_strength_reduction*d)
    tensile=lambda a:np.maximum(0,.5*(a[:,0]+a[:,1])+np.sqrt((.5*(a[:,0]-a[:,1]))**2+a[:,2]**2))
    mean=.5*(stress[:,0]+stress[:,1]);scalar=np.column_stack((mean,mean,np.zeros(len(mean))))
    result=dict(source_input_sha256=digest(raw),no_time_update=True,
        comparison_scope='Same source constitutive stress and water; instantaneous tensile-only comparison, excluding tides/shear/smoothing and any future evolution.',
        source_strength_min_max_pa=[float(source_strength.min()),float(source_strength.max())],
        starter_strength_at_source_temperature_min_max_pa=[float(starter_strength.min()),float(starter_strength.max())],
        starter_to_source_strength_min_max=[float((starter_strength/source_strength).min()),float((starter_strength/source_strength).max())],
        source_tensile_exceeded_faces=np.flatnonzero(tensile(stress)>=source_strength).tolist(),
        starter_law_tensile_exceeded_faces=np.flatnonzero(tensile(stress)>=starter_strength).tolist(),
        isotropic_scalar_source_law_exceeded_faces=np.flatnonzero(tensile(scalar)>=source_strength).tolist(),
        tensor_to_isotropic_scalar_relative_frobenius=float(np.linalg.norm(np.column_stack((stress[:,:2]-scalar[:,:2],np.sqrt(2)*stress[:,2])))/np.linalg.norm(np.column_stack((stress[:,:2],np.sqrt(2)*stress[:,2])))),
        historical_50kPa_is_basal_traction_not_strength=True,recommended_keep_source_traction_pa=params['shell']['convective_traction_pa'])
    out=ROOT/'analysis/genesis_aggregate_projection/strength_comparison.json'
    with out.open('xb') as f:f.write(encoded(result))
    print(json.dumps({k:v for k,v in result.items() if 'faces' not in k},indent=2))
    print({k:len(v) for k,v in result.items() if 'faces' in k})


if __name__=='__main__':main()
