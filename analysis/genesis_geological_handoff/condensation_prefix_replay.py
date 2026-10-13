"""Deterministic prefix verification, preserving the rejected split experiment.

No new trajectory parameters; skip duplicate expensive per-step diagnostics,
compare every recorded endpoint to the original prefix, retain last valid state.
"""
import json
from pathlib import Path
import time
from analysis.genesis_adaptive_control.controller import time_limit
from analysis.genesis_geological_handoff.real_progression import observe,checkpoint,sha,independent_phase_audit
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.genesis_transfer_bridge import encoded


def run():
    source=Path('/workspace/genesis-real-progression-01/checkpoint.json')
    original=Path('/workspace/genesis-real-condensation-01')
    root=Path('/workspace/genesis-real-condensation-verified');root.mkdir(exist_ok=False)
    rows=[json.loads(s) for s in (original/'history.jsonl').read_text().splitlines()]
    start=time.monotonic();model=SimplifiedGenesis.load(source).with_condensation();count=0;reason=None
    excluded={'wall_seconds','step_seconds','independent_phase'};initial=observe(model,0.)
    last_before=None;matched=0
    try:
        with time_limit(110):
            for row in rows[1:]:
                before=model.state
                try:model.advance(25.)
                except BaseException:
                    model.state=before
                    raise
                last_before=before;count+=1
                # Observables are cheap; the original run already checked all phase ledgers.
                actual=observe(model,0.)
                if any(actual[k]!=v for k,v in row.items() if k not in excluded):
                    raise AssertionError('REPLAY_PREFIX_OBSERVABLE_MISMATCH')
                matched+=1
                if count%500==0:print(json.dumps({'verified_prefix_steps':count,'seconds':time.monotonic()-start}),flush=True)
    except (ValueError,TimeoutError) as e:reason=type(e).__name__+': '+str(e)
    final_audit=independent_phase_audit(model,last_before) if last_before is not None else None
    checkpoint(model,root)
    report=dict(source_file=str(source),source_sha256=sha(source),original_candidate_run=str(original),
        status='STOPPED_BEFORE_NONLOCALIZED_SPLIT' if reason=='ValueError: DIFFUSE_DAMAGE_NOT_A_LOCALIZED_BELT' else 'REPLAY_INCOMPLETE',
        stop_reason=reason,initial=initial,final=observe(model,time.monotonic()-start),
        prefix_steps=count,exact_endpoint_observables_matched=matched,final_phase_audit=final_audit,
        final_checkpoint_sha256=sha(root/'checkpoint.json'),elapsed_seconds=time.monotonic()-start,
        physical_parameters_changed=False,selector_new_numeric_thresholds=False)
    (root/'report.json').write_bytes(encoded(report));print(json.dumps(report,indent=2),flush=True)
    return report

if __name__=='__main__':run()
