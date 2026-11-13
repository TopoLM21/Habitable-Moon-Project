"""Read-only evidence extraction, analytical comparison and compact figures."""
import csv,importlib.util,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent
from tectonics.conforming_surface import ConformingSurface,SurfaceParameters,SurfaceControl
from tectonics.integral_arcs import SphereReceiver
from tectonics.mesh import build_icosphere,connected_components
from tectonics.thermal import SECONDS_PER_MYR
s=importlib.util.spec_from_file_location('harness',HERE/'verify.py');v=importlib.util.module_from_spec(s);s.loader.exec_module(v)


def extract():
    rows=[]
    for path in sorted(HERE.glob('*.json')):
        d=json.loads(path.read_text())
        if not isinstance(d,dict) or 'final' not in d or 'case' not in d:continue
        f=d['final'];obj=ConformingSurface.load(HERE/'states'/d['case']);a=obj.arrays
        components=sum(len(connected_components(np.flatnonzero(a['owner']==owner),obj.neighbors)) for owner in np.unique(a['owner']))
        row=dict(case=d['case'],status=d.get('status','validity_limit' if d.get('failure') else 'complete'),
            end_myr=f['time_myr'],cells=f['cells'],fixed_dt_myr=d.get('fixed_step_myr'),
            heat_j=f['heat_generated_j'],external_work_j=f['external_work_j'],energy_residual_j=f['energy_residual_j'],
            energy_error_exchange=f['relative_energy_error_on_exchange'],max_height_km=f['max_height_m']/1000,
            heat_increment_k=f['crust_heat_increment_k'],height_l2_m=f['height_l2_m'],min_area_fraction=f['min_area_fraction'],
            area_error=f['area_relative_error'],owners=f['material_regions'],material_components=components,
            wall_s=d['wall_s'],integration_wall_s=d.get('integration_wall_s'),process_peak_rss_mib=d.get('process_peak_rss_mib'),
            steps=f['accepted_steps'],rejected_trials=d.get('rejected_trials',f['rejected_steps']))
        rows.append(row)
    with (HERE/'comparison.json').open('x') as f:json.dump(rows,f,indent=2)
    with (HERE/'comparison.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    original=ConformingSurface.load(HERE/'states/native100');fresh=ConformingSurface.load(HERE/'states/native100_fresh')
    assert original.checkpoint()==fresh.checkpoint()
    with (HERE/'fresh_equivalence.json').open('x') as f:json.dump(dict(all_arrays_metadata_equal=True),f)
    return rows


def analytical():
    rows=[];p=SurfaceParameters();h0=20000.;epsilon=1e-4
    J0=h0+h0*h0/p.radius_m+h0**3/(3*p.radius_m**2)
    expected=p.basalt_density_kg_m3*p.gravity_m_s2*J0**2/(p.basal_drag_pa_s_m*(1+h0/p.radius_m)**2)*6/p.radius_m**2
    for sub in (1,2,3):
        mesh=build_icosphere(sub);area=mesh.areas_unit_sphere*p.radius_m**2
        y=(3*mesh.centroids[:,2]**2-1)/2;h=h0*(1+epsilon*y);J=h+h*h/p.radius_m+h**3/(3*p.radius_m**2)
        zero=np.zeros(len(h));vol=area*J
        a=dict(owner=zero.astype(int),felsic_fraction=zero,felsic_volume_m3=zero,basalt_volume_m3=vol,
            felsic_age_moment_m3_myr=zero,basalt_age_moment_m3_myr=zero,potential=zero)
        obj=ConformingSurface(mesh,a);ev=obj.evaluate(obj.vertices);area,grad=obj.geometry(obj.vertices)
        adot=np.sum(grad*ev['velocity_per_s'][obj.faces],axis=(1,2));hdot=-J/(1+h/p.radius_m)**2*adot/area
        measured=-np.sum(area*y*hdot)/np.sum(area*y*(h-h0))
        rows.append(dict(cells=len(h),expected_per_myr=expected*SECONDS_PER_MYR,measured_per_myr=measured*SECONDS_PER_MYR,relative_error=measured/expected-1))
    with (HERE/'analytic_damping.json').open('x') as f:json.dump(dict(amplitude=epsilon,degree=2,rows=rows),f,indent=2)
    return rows


def figures(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    end=ConformingSurface.load(HERE/'states/native100');start=v.native(end.params,end.control)[0]
    lon=np.linspace(-math.pi,math.pi,181);lat=np.linspace(-math.pi/2,math.pi/2,91)
    ll,pp=np.meshgrid((lon[1:]+lon[:-1])/2,(lat[1:]+lat[:-1])/2)
    points=np.c_[np.cos(pp).ravel()*np.cos(ll).ravel(),np.cos(pp).ravel()*np.sin(ll).ravel(),np.sin(pp).ravel()]
    fig=plt.figure(figsize=(12,8),layout='constrained');grid=fig.add_gridspec(2,2)
    for i,obj in enumerate((start,end)):
        ax=fig.add_subplot(grid[0,i],projection='mollweide');ev=obj.evaluate(obj.vertices)
        height=(ev['height_m'][:,0]*obj.arrays['felsic_fraction']+ev['height_m'][:,1]*(1-obj.arrays['felsic_fraction']))/1000
        cell=SphereReceiver(obj.mesh()).locate(points)
        im=ax.pcolormesh(lon,lat,height[cell].reshape(ll.shape),vmin=0,vmax=40,cmap='terrain',shading='flat',rasterized=True)
        lines=[]
        for fa,fb,u,w in obj.edges:
            if obj.arrays['owner'][fa]==obj.arrays['owner'][fb]:continue
            t=np.linspace(0,1,5);x=(1-t[:,None])*obj.vertices[u]+t[:,None]*obj.vertices[w];x/=np.linalg.norm(x,axis=1)[:,None]
            xy=np.c_[np.arctan2(x[:,1],x[:,0]),np.arcsin(x[:,2])]
            lines.extend(np.array([a,b]) for a,b in zip(xy[:-1],xy[1:]) if abs(a[0]-b[0])<math.pi)
        ax.add_collection(LineCollection(lines,colors='k',linewidths=.35));ax.grid(alpha=.2)
        ax.set_title('Initial material partition' if i==0 else '100 Myr: deforming shared boundaries')
        fig.colorbar(im,ax=ax,orientation='horizontal',shrink=.75,label='Area-mean column height, km')
    history=end.data['history'];t=[0.,*(h['time_myr'] for h in history)]
    u=[end.data['initial_potential_j'],*(h['potential_j'] for h in history)]
    q=np.r_[0,np.cumsum([h['heat_j'] for h in history])]
    ax=fig.add_subplot(grid[1,0]);ax.plot(t,(u[0]-np.array(u))/1e26,label='Potential released');ax.plot(t,q/1e26,'--',label='Drag heat')
    ax.set(xlabel='Time, Myr',ylabel='Energy, 10²⁶ J');ax.grid(alpha=.25);ax.legend()
    ax.text(.03,.82,'Residual / heat = 2.09 × 10⁻⁶\nResidual is not deposited as heat',transform=ax.transAxes)
    ax=fig.add_subplot(grid[1,1]);ax.plot(t,[start.budget()['max_height_m']/1000,*(h['max_height_m']/1000 for h in history)])
    ax.set(xlabel='Time, Myr',ylabel='Maximum column height, km');ax.grid(alpha=.25)
    fig.suptitle('Bounded gravity/drag prototype: finite material, welded regions, fixed base\n12 prescribed regions remain 12; no cooling, crust creation/recycling, slip or fracture',fontsize=12)
    fig.savefig(HERE/'shared_surface100.png',dpi=140);plt.close(fig)
    by={r['case']:r for r in rows};fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for prefix in ('smooth','oblique'):
        sel=[by[f'smooth_g{i}_dt1' if prefix=='smooth' else f'oblique_g{i}'] for i in (1,2,3)]
        axs[0].plot([r['cells'] for r in sel],[r['heat_j']/sel[-1]['heat_j'] for r in sel],'o-',label=prefix)
    axs[0].set(xlabel='Cells',ylabel='Heat / 1280-cell reference');axs[0].legend();axs[0].set_title('Smooth physical initial fields')
    sel=[by[n] for n in ('native_g2_100','native100','native_g4_100')]
    axs[1].plot([r['cells'] for r in sel],[r['heat_j']/1e26 for r in sel],'o-')
    axs[1].set(xlabel='Cells',ylabel='100 Myr heat, 10²⁶ J',title='Native discontinuous columns')
    sel=[by[f'smooth_g3_dt{dt}'] for dt in ('2','1','0.5')]
    axs[2].loglog([2,1,.5],[r['energy_error_exchange'] for r in sel],'o-',label='smooth 1280')
    axs[2].loglog([1,.5],[by[n]['energy_error_exchange'] for n in ('native100','native100_dt_half')],'s-',label='native 1280')
    axs[2].set(xlabel='Maximum/fixed step, Myr',ylabel='|Energy residual| / heat',title='Work error is kept explicit');axs[2].legend()
    for ax in axs:ax.grid(alpha=.25)
    fig.suptitle('Temporal improvement does not establish spatial convergence of sharp initial profiles')
    fig.savefig(HERE/'convergence.png',dpi=140);plt.close(fig)
    kin=ConformingSurface.load(HERE/'states/prescribed_native20');report=json.loads((HERE/'prescribed_native20.json').read_text())
    _,points=v.sampled_coverage(kin);_,cp,inv=v.native(kin.params,kin.control)
    from scipy.spatial.transform import Rotation
    count=np.zeros(len(points),int)
    for owner,w in enumerate(np.array(report['input_omega_rad_myr'])):
        root=inv.receiver.locate(Rotation.from_rotvec(w*kin.data['time_myr']).inv().apply(points))
        count+=inv.coarse.arrays['owner'][root]==owner
    from matplotlib.colors import ListedColormap,BoundaryNorm
    cmap=ListedColormap(['#d86b48','#d8e5df','#526eb5']);norm=BoundaryNorm([-.5,.5,1.5,2.5],3)
    fig,axs=plt.subplots(1,2,figsize=(12,4.6),subplot_kw={'projection':'mollweide'},layout='constrained')
    for ax,c,title in [(axs[0],np.where(count==0,0,np.where(count==1,1,2)),'Independent rigid footprints'),(axs[1],np.ones(len(points)),'Shared welded material surface')]:
        sc=ax.scatter(np.arctan2(points[:,1],points[:,0]),np.arcsin(points[:,2]),c=c,s=7,cmap=cmap,norm=norm);ax.grid(alpha=.2);ax.set_title(title)
    bar=fig.colorbar(sc,ax=axs,orientation='horizontal',shrink=.65,ticks=[0,1,2]);bar.ax.set_xticklabels(['Uncovered','Covered once','Overlap'])
    fig.suptitle(f'Same imposed regional rotations, achieved {kin.data["time_myr"]:.4f} Myr\nShared geometry closes gaps, but imposed motion reaches the declared column-height limit',fontsize=12)
    fig.savefig(HERE/'contact_comparison.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    rows=extract();analytic=analytical();figures(rows)
    print(json.dumps(dict(rows=rows,analytical=analytic),indent=2))
