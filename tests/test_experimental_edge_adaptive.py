from pathlib import Path
import pytest
from tectonics.simplified_genesis import SimplifiedGenesis
from tectonics.experimental_edge_adaptive import attempt,integrate
from tectonics.experimental_edge_weakening import MODE
from tectonics.genesis_transfer_bridge import encoded
from analysis.genesis_geological_handoff.nodal_retirement_check import OPTION

@pytest.fixture
def source():
    p=Path('/workspace/genesis-moving-basal-source-01.json')
    if not p.exists():pytest.skip('Local source only')
    m=SimplifiedGenesis.load(p);m.config['experimental_geological_edge_fatigue']={**OPTION,'mode':MODE};return m


def test_error_rejection_and_fine_acceptance_are_transactional(source):
    before=encoded(source.state);trial,r=attempt(source,.25)
    assert trial is None and r['error_ratio']>1 and encoded(source.state)==before
    trial,r=attempt(source,.125)
    assert trial is not None and r['error_ratio']<=1 and encoded(source.state)==before
    assert trial.state['steps']==source.state['steps']+2


def test_guard_rejection_does_not_mutate_source(source,monkeypatch):
    before=encoded(source.state)
    def fail(*args,**kwargs):raise ValueError('manufactured safety guard')
    monkeypatch.setattr(SimplifiedGenesis,'advance',fail)
    trial,r=attempt(source,1.)
    assert trial is None and r['guard']=='manufactured safety guard' and encoded(source.state)==before


def test_min_step_stop_and_invalid_horizon(source):
    before=encoded(source.state)
    m,r=integrate(source,1.,{'min_years':.25})
    assert r['status']=='MIN_STEP_REJECTED' and encoded(m.state)==before
    with pytest.raises(ValueError,match='HORIZON'):integrate(source,0.)


def test_saved_predictive_run_grows_only_after_two_slow_checked_steps():
    import json
    path=Path(__file__).resolve().parents[1]/'analysis/genesis_geological_handoff/edge_adaptive_predictive_result.json'
    result=json.loads(path.read_text());streak=0;allow_growth=False;prior=None
    for row in result['attempts']:
        if prior is not None and row['dt_years']>prior*(1+1e-8):assert allow_growth
        if row['accepted']:
            assert row['error_ratio']<=1 and row['event_agreement']
            streak=streak+1 if row['slow'] and row['predicted_doubled_error_ratio']<.8 else 0
            allow_growth=streak>=2
            if allow_growth:streak=0
        else:streak=0;allow_growth=False
        prior=row['dt_years']
    assert result['status']=='COMPLETED' and result['rejection_count']>=1
    assert all(c['max_traction']<.05 and c['max_damage']<.02 for c in result['comparisons'])
