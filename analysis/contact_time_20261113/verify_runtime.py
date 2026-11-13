"""Bounded localized-source checks; previous states and physics stay intact."""
import importlib.util,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=ROOT/'outputs_v131_contact_time'
spec=importlib.util.spec_from_file_location('harness',ROOT/'analysis/reconstruction_exchange_20261113/verify_runtime.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h);h.HERE=HERE;h.OUT=OUT
h.BASE=h.yaml.safe_load((ROOT/'analysis/regional_area_20261113/area20.yaml').read_text())
h.BASE['regional_material']['active_area']['source_time_rule']='localized'
CASES={'gate8':(2,4.,8.,'initial_same_g2'),'gate20':(2,4.,20.,'initial_same_g2'),
       'gate_restart20':(2,4.,20.,'gate8'),'gate20_dt2':(2,2.,20.,'initial_same_g2'),
       'gate20_dt1':(2,1.,20.,'initial_same_g2'),'gate20_g1':(1,4.,20.,'initial_same_g1'),
       'gate20_g3':(3,4.,20.,'initial_same_g3')}
if __name__=='__main__':
    name=sys.argv[1]
    if name=='prepare':
        for sub in (1,2,3):
            source=ROOT/f'analysis/regional_inventory_20261113/saved_states/initial_same_g{sub}'
            target=OUT/f'initial_same_g{sub}/restart';target.mkdir(parents=True,exist_ok=False)
            for n in ('meta.json','state.npz'):shutil.copy2(source/n,target/n)
    elif name=='compare':h.compare(OUT/'gate20/restart',OUT/'gate_restart20/restart','gate_restart_comparison')
    elif name=='old_area':
        h.BASE['regional_material']['active_area'].pop('source_time_rule')
        h.run_case('old_area20',2,4.,20.,'initial_same_g2')
        h.compare(OUT/'old_area20/restart',ROOT/'analysis/regional_area_20261113/saved_states/area20','old_area_comparison')
    elif name in ('audit4','strict4'):
        import hashlib
        h.BASE['regional_material']['active_area']['contact_policy']='audit_material_banks' if name=='audit4' else 'require_shared_support'
        initial=OUT/'initial_same_g2/restart'
        before={n:hashlib.sha256((initial/n).read_bytes()).hexdigest() for n in ('meta.json','state.npz')}
        try:h.run_case(name,2,4.,4.,'initial_same_g2')
        except RuntimeError:
            if name!='strict4' or 'Material contacts not geometrically closed' not in (HERE/f'{name}.log').read_text():raise
            after={n:hashlib.sha256((initial/n).read_bytes()).hexdigest() for n in before}
            assert before==after and not (OUT/name/'restart/meta.json').exists()
            (HERE/'strict4_expected_failure.json').write_text(json.dumps(dict(expected_rejection=True,initial_hashes=before,
                  initial_unchanged=True,no_completed_checkpoint=True,scope='physical shared-bank geometry unavailable; no tolerance relaxation'),indent=2)+'\n')
        else:
            if name=='strict4':raise AssertionError('Separated banks were accepted as a closed contact geometry')
    else:h.run_case(name,*CASES[name])
