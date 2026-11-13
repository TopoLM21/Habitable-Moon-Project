"""Independent source, threshold and boundary-geometry diagnostics, no tuning."""
import json,math
from pathlib import Path
import sys
import numpy as np
import yaml
from scipy.integrate import quad
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tectonics.integral_arcs import ArcSegment,integrate_sources
from tectonics.mesh import build_icosphere
from tectonics.lithosphere import LithosphereState
from tectonics.continental import ContinentalCycleParameters,initialize_continental_cycle,advance_continental_cycle
from tectonics.checkpoint import load_checkpoint
from tectonics.topology import PlateTopologyManager,PlateTopologyParameters
from tectonics.kinematics import angular_velocity_vectors,classify_boundaries,BoundaryType
HERE=Path(__file__).resolve().parent;OUT=ROOT/'outputs_v131_integral_arcs';R=5287.


def run():
    source=ArcSegment(np.array([1.,0.,0.]),np.array([math.cos(1.),math.sin(1.),0.]),np.array([0.,0.,1.]),0,.7,105.)
    exact=.7*R*quad(lambda d:math.cos(d/R)*math.exp(-.5*((d-105.)/140.)**2),0,425.,epsabs=1e-9)[0]
    rows=[]
    for sub in (1,2,3,4):
        mesh=build_icosphere(sub);n=mesh.cell_count;a=mesh.physical_cell_areas_km2(R)
        fields={order:integrate_sources(mesh,np.zeros(n,int),[source],R,140.,320.,order=order)[0] for order in (8,16,32)}
        params=ContinentalCycleParameters(**yaml.safe_load((HERE/'control.yaml').read_text())['continental_cycle'])
        for dt in (2.,1.):
            s=LithosphereState(0.,np.zeros(n,np.int32),np.zeros(n,np.int8),np.ones(n)*100,np.ones(n)*7,np.zeros(n),
                continental_fraction=np.zeros(n),continental_volume_km3=np.zeros(n))
            cycle=initialize_continental_cycle(mesh);first=None
            for i in range(round(100/dt)):
                s.time_myr+=dt
                s,cycle,diag=advance_continental_cycle(mesh,s,[],cycle,dt,R,params,volcanic_arc_forcing=fields[32]/a,integral_arc_forcing=True)
                if first is None and diag.juvenile_arc_area_created_km2>0:first=s.time_myr
            rows.append({'cells':n,'dt_myr':dt,'source_integral_km2':math.fsum(fields[32]),
                'default_integral_relative_error':math.fsum(fields[8])/exact-1,
                'default_vs_order32_deposition_l1_relative':math.fsum(abs(fields[8]-fields[32]))/exact,
                'order16_vs_order32_deposition_l1_relative':math.fsum(abs(fields[16]-fields[32]))/exact,
                'mean_field_max':float(max(fields[32]/a)),'juvenile_area_km2':cycle.cumulative_generated_area_km2,
                'continental_generated_km3':cycle.cumulative_generated_volume_km3,'first_step_with_event_myr':first})
    geom=[]
    for sub in (1,2,3):
        mesh=build_icosphere(sub);cp=load_checkpoint(OUT/f'initial_same_g{sub}/restart',PlateTopologyManager(PlateTopologyParameters()))
        bs=classify_boundaries(mesh,cp.system,R,4.,1.);omega=angular_velocity_vectors(cp.system)
        oldpos=oldneg=truepos=trueneg=tilt=lengthsum=0.
        for b in bs:
            u,v=mesh.vertices[b.vertex_u],mesh.vertices[b.vertex_v];cross=np.cross(u,v)
            n=cross/np.linalg.norm(cross)
            if n@(mesh.centroids[b.face_b]-mesh.centroids[b.face_a])<0:n=-n
            mid=b.midpoint;true=float(np.cross(omega[b.plate_b]-omega[b.plate_a],mid)@n)*R
            length=R*math.atan2(np.linalg.norm(cross),u@v)
            # Compare midpoint laws only, retaining each method's same 4 km/Myr threshold.
            oldpos+=length*max(b.normal_rate_km_per_myr,0.)*(b.normal_rate_km_per_myr>4)
            oldneg+=length*max(-b.normal_rate_km_per_myr,0.)*(b.normal_rate_km_per_myr<-4)
            truepos+=length*max(true,0.)*(true>4);trueneg+=length*max(-true,0.)*(true<-4)
            d=mesh.centroids[b.face_b]-mesh.centroids[b.face_a];d-=mid*(d@mid);d/=np.linalg.norm(d)
            tilt+=length*math.acos(np.clip(n@d,-1,1));lengthsum+=length
        geom.append({'cells':mesh.cell_count,'mean_centroid_normal_tilt_deg':math.degrees(tilt/lengthsum),
            'legacy_normal_opening_km2_per_myr':oldpos,'geodesic_normal_opening_km2_per_myr':truepos,
            'legacy_normal_closing_km2_per_myr':oldneg,'geodesic_normal_closing_km2_per_myr':trueneg})
    max_p=.7*params.arc_maturation_rate_per_myr*params.arc_potential_decay_myr*(-math.expm1(-100/params.arc_potential_decay_myr))
    half_width=140*math.sqrt(2*math.log(max_p/params.juvenile_threshold)) if max_p>params.juvenile_threshold else 0.
    pointwise_area=R*R*(math.sin(min(425.,105+half_width)/R)-math.sin(max(0.,105-half_width)/R)) if half_width else 0.
    result={'exact_strip_source_km2':exact,'continuum_pointwise_mature_area_at100_km2':pointwise_area,
            'fixed_source_threshold_tests':rows,'frozen_initial_boundary_normals':geom,
            'scope':'diagnosis; does not change the runtime classifier or physical thresholds'}
    with (HERE/'operator_diagnostics_control.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result),flush=True)

if __name__=='__main__':run()
