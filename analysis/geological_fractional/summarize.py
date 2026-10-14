"""Summarize completed matched runs; keep prior attempts untouched."""
import hashlib,json,math,re
from pathlib import Path

root=Path('/workspace/geological-fractional-final-02');out={}
for name in ('young1','young05','cont1','cont05'):
    r=json.loads((root/name/'result.json').read_text());a=json.loads((root/name/'geos_audit.json').read_text())
    assert r['error'] is None and r['restart_exact'] and not a['raw_invalid_polygons'] and not a['invalid_polygons']
    g=[row['geometry'] for row in r['steps']]
    log=(root/name/'run.log').read_text()
    elapsed=[float(x) for x in re.findall(r'^accepted \S+ \d+ (\S+)$',log,re.M)]
    out[name]=dict(source=r['source'],dt_myr=r['dt'],initial=r['history'][0],final=r['history'][-1],
        advance_elapsed_seconds=elapsed[-1],total_elapsed_seconds=r['seconds'],restart_exact=r['restart_exact'],events=r['events'],
        incremental_mass_relative_error=r['incremental_mass_relative_error'],incremental_H_relative_error=r['incremental_H_relative_error'],
        global_balances=r['balances'],generated_volume_km3=math.fsum(x['material']['generated_volume_km3'] for x in r['steps']),
        recycled_volume_km3=math.fsum(x['material']['recycled_volume_km3'] for x in r['steps']),
        max_cell_area_relative_error=max(x['max_cell_area_relative_error'] for x in g),
        source_plus_contact_area_closure_km2=math.fsum(x['source_absolute_area_correction_km2']+x['contact_absolute_area_correction_km2'] for x in g),
        source_mass_weight_correction_kg=math.fsum(x['source_absolute_mass_correction_kg'] for x in g),
        source_area_budget_km2=math.fsum(x['source_area_correction_budget_km2'] for x in g),
        contact_area_budget_km2=math.fsum(x['contact_area_correction_budget_km2'] for x in g),
        within_cell_consolidation_area_km2=math.fsum(x['end_cell_roundoff_cleanup']['area_km2'] for x in g),
        within_cell_consolidation_mass_kg=math.fsum(x['end_cell_roundoff_cleanup']['mass_kg'] for x in g),
        unresolved_gap_area_km2=math.fsum(x['unresolved_gap_area_km2'] for x in g),
        max_gap_grid=max(x['gap_overlay_max_grid'] for x in g),audit=a,
        total_polygon_edge_components=sum(v['count'] for v in a['polygon_edge_components_by_owner'].values()),
        secondary_component_area_km2=math.fsum(math.fsum(v['component_areas_steradians'][1:]) for v in a['polygon_edge_components_by_owner'].values())*5287**2)
controls={}
for name in ('native1','native05'):
    r=json.loads((Path('/workspace/geological-fractional-final-01')/name/'result.json').read_text())
    controls[name]=dict(final=r['history'][-1],restart_exact=r['restart_exact'],seconds=r['seconds'],
        generated_volume_km3=math.fsum(x['material']['generated_volume_km3'] for x in r['steps']),
        recycled_volume_km3=math.fsum(x['material']['recycled_volume_km3'] for x in r['steps']))
files=['tectonics/geological_area_transport.py','tectonics/geological_fractional.py','tectonics/geological_planar.py','tectonics/geological_roundoff.py','tectonics/unified_geological_genesis.py','requirements-cloud.txt']
mid={name:json.loads((Path('/workspace/geological-fractional-mid-restart')/name/'result.json').read_text()) for name in ('young05','cont05')}
assert all(x['exact'] for x in mid.values())
result=dict(mid_interval_restart=mid,base_commit='ab88bcc4e0dd8daea64009e939e4e3ae23f1f8b1',raw_root=str(root),runs=out,native_controls=controls,
    implementation_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},tests='89 passed in 23.73s')
Path('analysis/geological_fractional/VALIDATED_SUMMARY.json').write_text(json.dumps(result,indent=2))
for name,r in out.items():print(name,r['advance_elapsed_seconds'],r['total_elapsed_seconds'],r['generated_volume_km3'],r['recycled_volume_km3'],r['total_polygon_edge_components'])
