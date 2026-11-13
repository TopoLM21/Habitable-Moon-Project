"""Bounded coupled checks, using the same physical initial state as a6e7894."""
from copy import deepcopy
import importlib.util,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent;OUT=ROOT/'outputs_v131_regional_area'
spec=importlib.util.spec_from_file_location('previous_checks',ROOT/'analysis/reconstruction_exchange_20261113/verify_runtime.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h);h.HERE=HERE;h.OUT=OUT
h.BASE=h.yaml.safe_load((ROOT/'analysis/regional_inventory_20261113/control.yaml').read_text())
h.BASE['regional_material']['active_area']=dict(enabled=True,max_edge_km=125.,quadrature_order=16,quadrature_spacing_km=100.,strip_rule='coarea')
CASES={
    'area12':(125.,2,4.,12.,'initial_same_g2'),
    'area20':(125.,2,4.,20.,'initial_same_g2'),
    'area_restart20':(125.,2,4.,20.,'area12'),
    'area20_edge250':(250.,2,4.,20.,'initial_same_g2'),
    'area20_edge62':(62.5,2,4.,20.,'initial_same_g2'),
    'area20_dt2':(125.,2,2.,20.,'initial_same_g2'),
    'area20_dt1':(125.,2,1.,20.,'initial_same_g2'),
    'area20_g1':(125.,1,4.,20.,'initial_same_g1'),
    'area20_g3':(125.,3,4.,20.,'initial_same_g3'),
}
if __name__=='__main__':
    arg=sys.argv[1]
    if arg=='prepare':
        for sub in (1,2,3):
            source=ROOT/f'analysis/regional_inventory_20261113/saved_states/initial_same_g{sub}'
            target=OUT/f'initial_same_g{sub}/restart';target.mkdir(parents=True,exist_ok=False)
            for name in ('state.npz','meta.json'):shutil.copy2(source/name,target/name)
    elif arg=='compare':h.compare(OUT/'area_restart20/restart',OUT/'area20/restart','native_restart_comparison')
    else:
        edge,sub,dt,end,resume=CASES[arg];h.BASE['regional_material']['active_area']['max_edge_km']=edge
        h.run_case(arg,sub,dt,end,resume)
