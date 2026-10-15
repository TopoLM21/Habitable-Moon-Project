"""Render actual saved cell relief, not an invented continuous coastline."""
import json,sys,subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from tectonics.spherical_polygons import clip_hemisphere
from matplotlib.patches import Patch
from tectonics.unified_geological_genesis import UnifiedGeologicalGenesis as U
from analysis.geological_continuation.run import snap

def render(saved,output):
    e=U.load(saved);s=snap(e);m=e.mesh;cp=e.checkpoint
    sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    result=json.loads((Path(saved).parent/'result.json').read_text())
    height=cp.topo.elevation_m-s['sea_level_m'];land=height>=0
    colors=plt.colormaps['Blues'](.38+.55*np.clip(-height/8000,0,1))
    colors[land]=plt.colormaps['YlGn'](.4+.5*np.clip(height[land]/4000,0,1))
    fig=plt.figure(figsize=(13,7.5),facecolor='#eef4f6')
    for index,angle in enumerate((-65,115)):
        ax=fig.add_subplot(1,2,index+1);ax.set_facecolor('#eef4f6')
        az=np.deg2rad(angle);el=np.deg2rad(12 if index==0 else -12);view=np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
        east=np.array([-np.sin(az),np.cos(az),0.]);north=np.cross(view,east)
        polygons=[];facecolors=[]
        for i,face in enumerate(m.faces):
            clipped=clip_hemisphere(m.vertices[face],view)
            if len(clipped)<3:continue
            arcs=[]
            for u,v in zip(clipped,np.roll(clipped,-1,axis=0)):
                points=np.linspace(u,v,12,endpoint=False);points/=np.linalg.norm(points,axis=1)[:,None];arcs.extend(points)
            points=np.array(arcs);polygons.append(np.c_[points@east,points@north]);facecolors.append(colors[i])
        ax.add_collection(PolyCollection(polygons,facecolors=facecolors,edgecolors='none'))
        for a,b,u,v in m.shared_edges:
            if cp.system.cell_plate[a]!=cp.system.cell_plate[b]:
                points=np.linspace(m.vertices[u],m.vertices[v],40);points/=np.linalg.norm(points,axis=1)[:,None]
                visible=points@view>=0;xy=np.c_[points@east,points@north];xy[~visible]=np.nan
                ax.plot(xy[:,0],xy[:,1],color='#f8e6a0',lw=1.15)
        ax.set_xlim(-1.06,1.06);ax.set_ylim(-1.06,1.06);ax.set_aspect('equal');ax.set_axis_off()
        ax.set_title(('Полушарие A','Противоположное полушарие')[index],fontsize=12,color='#34505e',pad=-4)
    fig.suptitle(f'Habitable Moon · поверхность на {s["time_myr"]:.4f} млн лет',fontsize=20,color='#173445',y=.94)
    fig.text(.5,.88,f'Суша по модели: {100*s["land_area_fraction"]:.3f}%  ·  {s["plates"]} кинематических групп  ·  уровень моря {s["sea_level_m"]:.1f} м',ha='center',fontsize=11)
    fig.legend(handles=[Patch(color='#2369a5',label='Океан: темнее — глубже'),Patch(color='#74b560',label='Суша'),Patch(color='#f8e6a0',label='Граница доминирующих плит')],loc='lower center',bbox_to_anchor=(.5,.13),ncol=3,frameon=False)
    fig.text(.5,.085,f'{m.cell_count} тепловых ячеек · берег показан по средней высоте ячейки; доля суши учитывает внутрячеечный рельеф.',ha='center',fontsize=9,color='#34505e')
    fig.text(.5,.058,f'Экспериментальная дробная геометрия · dt = {result["dt"]} млн лет · карта сохранённого состояния',ha='center',fontsize=9,color='#34505e')
    fig.text(.5,.035,f'Checkpoint: {Path(saved).absolute()}',ha='center',fontsize=8,color='#34505e')
    fig.text(.5,.014,f'SHA кода: {sha}',ha='center',fontsize=8,color='#34505e')
    fig.subplots_adjust(left=.015,right=.985,bottom=.18,top=.84,wspace=-.1)
    fig.savefig(output,dpi=190,facecolor=fig.get_facecolor());plt.close(fig)
    print(json.dumps(dict(saved=saved,output=output,metrics=s,topology_parameters=vars(cp.manager.params) if hasattr(cp.manager.params,'__dict__') else str(cp.manager.params)),ensure_ascii=False))
if __name__=='__main__':render(*sys.argv[1:])
