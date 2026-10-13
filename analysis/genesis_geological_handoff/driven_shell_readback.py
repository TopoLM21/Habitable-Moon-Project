"""Read-only comparisons of the fixed driven-shell experiment."""
import json,hashlib
from pathlib import Path
import numpy as np
from tectonics.mesh import build_icosphere,connected_components
from tectonics.genesis_onset_support import NonlocalLoading

ROOT=Path(__file__).parent


def zones(mesh,values):
    a=mesh.areas_unit_sphere;ar=a/a.sum();flux=ar*values
    order=np.argsort(values)[::-1];count=np.searchsorted(np.cumsum(flux[order]),.5*flux.sum())+1
    selected=np.zeros(len(values),bool);selected[order[:count]]=True
    components=connected_components(np.flatnonzero(selected),mesh.neighbors)
    perimeter=0.;R=5287.
    for f,g,u,v in mesh.shared_edges:
        if selected[f]!=selected[g]:perimeter+=R*np.arctan2(np.linalg.norm(np.cross(mesh.vertices[u],mesh.vertices[v])),mesh.vertices[u]@mesh.vertices[v])
    area=float(a[selected].sum()*R**2)
    return dict(selected_area_fraction=float(ar[selected].sum()),plastic_flow_share=float(flux[selected].sum()/flux.sum()),
                components=len(components),area_km2=area,boundary_km=perimeter,
                area_perimeter_width_km=2*area/perimeter if perimeter else None,
                note='Whole cells carrying at least half current plastic flow; 2A/P is a geometric proxy, not a demonstrated belt width.')


def main():
    cases=[]
    for name in ('2_0_0.05','2_1_0.05','3_0_0.05','3_1_0.05','2_0_0.025','2_1_0.025'):
        p=ROOT/f'driven_{name}_run.json'
        if not p.exists():continue
        packet=json.loads(p.read_text());r=packet['result']
        if r['status']!='PASS':cases.append(dict(case=name,status=r['status'],error=r.get('error')));continue
        rows=r['rows'];ledger=r['ledger'];mesh=build_icosphere(r['level']);tail=rows[-len(rows)//4:]
        balance=sum(ledger.values())-2*ledger['external']
        flow=np.asarray(rows[-1]['flow']);a=mesh.areas_unit_sphere;a=a/a.sum()
        bar=NonlocalLoading(mesh,5287.,4500.).apply(np.array(r['kappa']),np.ones(mesh.cell_count,bool))
        cases.append(dict(case=name,status='PASS',seconds=r['seconds'],faces=r['faces'],unknowns=r['unknowns'],restart=r['restart'],
            kappa_metrics=r['kappa_metrics'],last_flow=rows[-1]['metrics'],tail_cv_range=[min(x['metrics']['cv'] for x in tail),max(x['metrics']['cv'] for x in tail)],
            tail_log_cv_slope=float(np.log(tail[-1]['metrics']['cv']/tail[0]['metrics']['cv'])/(tail[-1]['time']-tail[0]['time'])),
            zones=zones(mesh,flow),narrow_criterion=r['criterion'],
            max_momentum=max(x['momentum'] for x in rows),max_energy=max(x['energy'] for x in rows),
            max_constitutive=max(x['closure'] for x in rows),max_isochoric=max(x['isochoric'] for x in rows),
            max_newton=max(x['newton'] for x in rows),max_outer=max(x['outer'] for x in rows),
            external_j=ledger['external'],plastic_j=ledger['plastic'],numerical_memory_fraction=ledger['numerical_memory']/ledger['external'],
            whole_energy_relative=abs(balance)/ledger['external'],ell_over_max_cell=r['ell_over_max_cell_size'],
            fraction_at_residual=float(a[bar>=packet['J2']['equivalent_critical_strain']].sum()) if r['weak'] else None))
    lookup={c['case']:c for c in cases};comparisons={}
    for label,x,y in [('mesh_A','2_0_0.05','3_0_0.05'),('mesh_B','2_1_0.05','3_1_0.05'),
                      ('dt_A','2_0_0.05','2_0_0.025'),('dt_B','2_1_0.05','2_1_0.025')]:
        if x in lookup and y in lookup and lookup[x]['status']==lookup[y]['status']=='PASS':
            a,b=lookup[x],lookup[y];comparisons[label]={k:b['last_flow'][k]/a['last_flow'][k]-1 for k in ('mean','cv','participation_area','area_for_half_flow')}
            old=json.loads((ROOT/f'driven_{x}_run.json').read_text())['result']
            fine=json.loads((ROOT/f'driven_{y}_run.json').read_text())['result']
            oldflow=np.array(old['rows'][-1]['flow']);fineflow=np.array(fine['rows'][-1]['flow'])
            weights=build_icosphere(old['level']).areas_unit_sphere;weights/=weights.sum()
            if label.startswith('mesh'):
                fa=build_icosphere(fine['level']).areas_unit_sphere.reshape(-1,4)
                fineflow=(fineflow.reshape(-1,4)*fa).sum(axis=1)/fa.sum(axis=1)
            comparisons[label]['profile_relative_L2']=float(np.sqrt(weights@(fineflow-oldflow)**2/(weights@oldflow**2)))
    protected={}
    for p in (Path('/workspace/genesis-moving-basal-source-01.json'),Path('/workspace/genesis-moving-basal-pilot-01/checkpoint.json')):
        protected[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    output=dict(cases=cases,relative_differences=comparisons,protected_hashes=protected)
    (ROOT/'driven_summary.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
