"""Real legacy projection/restart only; no real-planet advance."""
import argparse,hashlib,json,resource,signal,time
from pathlib import Path
from tectonics.genesis_reservoir_thermal import migrate,save,load,total,temperature
from tectonics.genesis import GenesisParameters
from analysis.genesis_energy_inventory.audit import ROOT,ROOT_SHA


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
    def stop(*args):raise TimeoutError('60-second budget')
    signal.signal(signal.SIGALRM,stop);signal.alarm(60);start=time.monotonic()
    raw=ROOT.read_bytes();s=migrate(raw,ROOT_SHA);save(s,a.output/'projected.json');restored=load(a.output/'projected.json',legacy_path=ROOT);save(restored,a.output/'restart.json')
    if (a.output/'projected.json').read_bytes()!=(a.output/'restart.json').read_bytes():raise ValueError('Restart differs')
    if ROOT.read_bytes()!=raw:raise ValueError('Canonical changed')
    origin=json.loads(s.origin_json);p=GenesisParameters(**json.loads(s.parameters_json));legacy=json.loads(raw)['runtime_state']['thermal']
    for i,row in enumerate(legacy['layer_mass_kg']):
        for j,m in enumerate(row):
            c=next(c for c in s.cells if c.identity==f'column:{i}:{j}')
            if c.mass_kg!=m or c.energy_j!=m*legacy['column_enthalpy'][i][j]:raise ValueError('Column changed')
    result=dict(status='PASS',new_Genesis_steps=0,real_thermal_advance=False,canonical_sha256=ROOT_SHA,
      reservoirs=len(s.cells),origin=origin,thermal_total_j=total(s),
      projection_residual_j=total(s)-origin['legacy_thermal_total_j'],restart_byte_identical=True,
      column_profiles_preserved=True,historical_opening_balance=s.historical_opening_balance,
      water_temperature_k=temperature(s.cells[-1],p),seconds=time.monotonic()-start,
      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      code_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ['tectonics/genesis_reservoir_thermal.py',__file__]},
      artifacts={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in a.output.iterdir()})
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));signal.alarm(0)

if __name__=='__main__':main()
