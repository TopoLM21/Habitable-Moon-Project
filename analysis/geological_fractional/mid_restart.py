"""Compare a true mid-interval restart against the accepted uninterrupted pair."""
import json,sys,time
from pathlib import Path
from tectonics.unified_geological_genesis import UnifiedGeologicalGenesis as U
from analysis.genesis_geological_handoff.experiment import exact_saved_state

def run(source,output):
    source=Path(source);output=Path(output);output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    e=U.load(source/'initial');e.step(.5);e.save(output/'mid');del e
    resumed=U.load(output/'mid');resumed.step(.5);path=resumed.save(output/'resumed')
    exact_saved_state(source/'accepted',path)
    result=dict(source=str(source),exact=True,seconds=time.monotonic()-start,mid_time_myr=resumed.checkpoint.state.time_myr-.5)
    (output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':run(*sys.argv[1:])
