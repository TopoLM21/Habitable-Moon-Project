"""Sequential bounded controls. Never overwrite a run or retry a failed case."""
import importlib.util,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent;OUT=ROOT/'outputs_v131_regional'
spec=importlib.util.spec_from_file_location('previous_checks',ROOT/'analysis/reconstruction_exchange_20261113/verify_runtime.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h);h.HERE=HERE;h.OUT=OUT
h.BASE=h.yaml.safe_load((HERE/'control.yaml').read_text())

def prepare():
    for sub in (1,2,3):
        source=ROOT/f'analysis/integral_arcs_20261113/saved_states/initial_same_g{sub}'
        target=OUT/f'initial_same_g{sub}/restart';target.mkdir(parents=True,exist_ok=False)
        for name in ('meta.json','state.npz'):shutil.copy2(source/name,target/name)

CASES={
    'm3_first48':(3,2,4.,48.,'initial_same_g2'),
    'm3_restart100':(3,2,4.,100.,'m3_first48'),
    'm3_g2_dt4':(3,2,4.,100.,'initial_same_g2'),
    'm3_g2_dt2':(3,2,2.,100.,'initial_same_g2'),
    'm3_g2_dt1':(3,2,1.,100.,'initial_same_g2'),
    'm2_g2_dt4':(2,2,4.,100.,'initial_same_g2'),
    'm4_g2_dt4':(4,2,4.,100.,'initial_same_g2'),
    'm3_g1_dt4':(3,1,4.,100.,'initial_same_g1'),
    'm3_g3_dt4':(3,3,4.,100.,'initial_same_g3'),
}
SHORT_CASES={
    'short_m3_g2_dt4':(3,2,4.,20.,'initial_same_g2'),
    'short_m3_g2_dt2':(3,2,2.,20.,'initial_same_g2'),
    'short_m3_g2_dt1':(3,2,1.,20.,'initial_same_g2'),
    'short_m2_g2_dt4':(2,2,4.,20.,'initial_same_g2'),
    'short_m4_g2_dt4':(4,2,4.,20.,'initial_same_g2'),
    'short_m3_g1_dt4':(3,1,4.,20.,'initial_same_g1'),
    'short_m3_g3_dt4':(3,3,4.,20.,'initial_same_g3'),
    'fixed_restart100':(3,2,4.,100.,'m3_first48'),
}

def run(name):
    material,sub,dt,end,resume=(CASES|SHORT_CASES)[name]
    h.BASE['regional_material']['material_subdivisions']=material
    h.run_case(name,sub,dt,end,resume)

if __name__=='__main__':
    arg=sys.argv[1]
    if arg=='prepare':prepare()
    elif arg in ('suite','short_suite'):
        failures=[]
        for name in (CASES if arg=='suite' else SHORT_CASES):
            try:run(name)
            except (RuntimeError,TimeoutError) as exc:
                failures.append({'case':name,'failure':str(exc)})
                print(json.dumps(failures[-1]),flush=True)
        with (HERE/f'{arg}_failures.json').open('x') as f:json.dump(failures,f,indent=2)
        if arg=='short_suite' and (OUT/'fixed_restart100/restart').exists():
            h.compare(OUT/'fixed_restart100/restart',OUT/'m3_g2_dt4/restart','fixed_native_restart_comparison')
    else:run(arg)
