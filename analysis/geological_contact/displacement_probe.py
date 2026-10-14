"""Read actual native transfer arrays for the paired hypothesis control."""
import json,math
from pathlib import Path
import numpy as np
from tectonics.unified_geological_genesis import UnifiedGeologicalGenesis as U
import tectonics.genesis_geological_handoff as handoff
import tectonics.lithosphere as lithosphere
import tectonics.geological_material as material


def probe():
    evidence={};original_step=handoff.advance_lithosphere;original_thickness=lithosphere.effective_oceanic_thickness_km
    original_transfer=material.transport_cycle
    def thickness(frac,volume,areas,*args,**kwargs):
        try:return original_thickness(frac,volume,areas,*args,**kwargs)
        except ValueError:
            bad=(frac>=1)&(volume>0)
            evidence.setdefault('invalid_columns',[]).append(dict(cells=np.flatnonzero(bad).tolist(),basalt_volume_km3=float(volume[bad].sum())))
            raise
    def step(mesh,system,state,*args,**kwargs):
        result=original_step(mesh,system,state,*args,**kwargs)
        trace=kwargs['material_trace'];src=trace['winner_source'];valid=src>=0
        areas=mesh.physical_cell_areas_km2(args[1]);after=result[0]
        old_cont_area=np.zeros(mesh.cell_count);old_cont_area[valid]=state.continental_fraction[src[valid]]*areas[src[valid]]
        added=after.continental_fraction*areas-old_cont_area
        lost=trace['basalt_before_displacement']-trace['basalt_after_displacement']
        only_remap=(added<=1e-7)&(lost>0)
        evidence['native_displacement']=dict(volume_km3=float(lost.sum()),affected_cells=np.flatnonzero(lost>0).tolist(),
            no_net_incoming_continental_area_volume_km3=float(lost[only_remap].sum()),
            no_net_incoming_continental_area_cells=np.flatnonzero(only_remap).tolist(),
            formula='V_basalt * max(0, 1-(1-f_new)/(1-f_source)); fractions are on unequal raster cells')
        evidence.setdefault('native_displacement_steps',[]).append(dict(time_myr=result[0].time_myr,**evidence['native_displacement']))
        return result
    def transfer(parcel,mass,energy,old,lith,final,trace,cycle,time):
        src=trace['winner_source'];lost=trace['basalt_before_displacement']-trace['basalt_after_displacement']
        H=0.;M=0.
        for target in np.flatnonzero(lost>0):
            origin=int(src[target]);dm=float(lost[target])*material.CONVERSION
            if origin>=0:
                parts=[p for node in parcel['nodes'][origin] for p in parcel['node_parts'][str(node)] if p.get('material_phase','mafic')=='mafic']
                h=math.fsum(p['energy_j'] for p in parts)/math.fsum(p['mass_kg'] for p in parts)
            else:h=energy[-2]/mass[-2]
            M+=dm;H+=dm*h
        result=original_transfer(parcel,mass,energy,old,lith,final,trace,cycle,time)
        evidence['predicted_displaced_basalt']=dict(mass_kg=M,actual_source_enthalpy_j=H,
            destination='well_mixed_deep_silicate; original mafic provenance retained in recycled_history, not separate active chemical inventory')
        evidence['all_material_transfers']={k:v for k,v in result.items()}
        return result
    handoff.advance_lithosphere=step;lithosphere.effective_oceanic_thickness_km=thickness;material.transport_cycle=transfer
    try:
        source='/workspace/geological-continuation-01/dt1/last_accepted'
        for mode in ('off','on'):
            e=U.load(source);e.record['continental_displacement']=None if mode=='off' else 'retire_displaced_basalt'
            try:e.step(1.)
            except ValueError as error:evidence[mode+'_error']=str(error)
        evidence['first_contact']={k:v for k,v in evidence.items() if k!='native_displacement_steps'}
        evidence['native_displacement_steps']=[]
        e=U.load('/workspace/geological-continuation-03/dt1/at160')
        for _ in range(40):e.step(1.)
    finally:
        handoff.advance_lithosphere=original_step;lithosphere.effective_oceanic_thickness_km=original_thickness;material.transport_cycle=original_transfer
    return evidence

if __name__=='__main__':
    result=probe();Path('analysis/geological_contact/displacement_probe.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
