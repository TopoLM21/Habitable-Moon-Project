import json
from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest
import run_genesis_volume as cli
from tectonics.genesis_volume_runtime import metadata,validate_bundle,seal,VolumeRuntime
from tectonics.genesis_volume_slip_dynamics import VolumeSlipModel


def test_default_raw_model_step_is_not_replaced():
    assert VolumeSlipModel.step.__module__=='tectonics.genesis_volume_slip_dynamics'
    assert VolumeSlipModel.configured_runtime.__kwdefaults__['integrator']=='midpoint'
    assert metadata('midpoint')['role']=='legacy_default'
    assert metadata('be50')['role']=='engineering_candidate'
    assert metadata('be25')['role']=='control_mode'


@pytest.mark.parametrize('mode',['midpoint','be25'])
def test_wrong_mode_refuses_be_journal_before_io(mode):
    p={'format':'genesis-volume-integrator-runtime-0.1','integrator_metadata':metadata('be50'),'root_checkpoint_utf8':'','chain':[]}
    with pytest.raises(ValueError,match='mismatch'):validate_bundle(seal(p),mode)


def test_metadata_cannot_enable_numerical_heating():
    for mode in ['midpoint','be50','be25']:
        assert metadata(mode)['numerical_dissipation_destination']=='diagnostic_only_no_thermal_feedback'
        assert metadata(mode)['physical_accuracy_certified'] is False
    p=dict(format='genesis-volume-integrator-runtime-0.1',integrator_metadata={**metadata('be50'),'numerical_dissipation_destination':'heat'},root_checkpoint_utf8='',chain=[])
    with pytest.raises(ValueError):validate_bundle(seal(p),'be50')


@pytest.fixture
def fake_cli(monkeypatch):
    import tectonics.genesis_volume_runtime as module
    runtime=Mock();runtime.describe.return_value={'mode':'fake'}
    load=Mock(return_value=runtime);monkeypatch.setattr(module,'load_runtime',load)
    monkeypatch.setattr(cli.resource,'setrlimit',lambda *a:None)
    return load,runtime


def test_inspect_default_never_advances(fake_cli):
    load,runtime=fake_cli
    assert cli.main(['inspect','--input','legacy.json'])==0
    load.assert_called_once_with(cli.Path('legacy.json'),integrator='midpoint')
    runtime.advance.assert_not_called()


def test_config_routes_explicit_be_only_on_advance(fake_cli,tmp_path):
    load,runtime=fake_cli;c=tmp_path/'c.json';c.write_text('{"integrator":"be50","steps":2}')
    assert cli.main(['advance','--input','x','--config',str(c),'--output',str(tmp_path/'o')])==0
    assert load.call_args.kwargs['integrator']=='be50' and runtime.advance.call_count==2


@pytest.mark.parametrize('config',[
    {'integrator':'unknown'},{'steps':0},{'steps':True},{'steps':5},{'wall_seconds':float('nan')},
    {'wall_seconds':601},{'young_modulus_pa':1},{'numerical_dissipation_to_heat':True}])
def test_bad_config_refused_before_loading(fake_cli,tmp_path,config):
    load,_=fake_cli;p=tmp_path/'c';p.write_text(json.dumps(config))
    assert cli.main(['inspect','--input','x','--config',str(p)])==1
    load.assert_not_called()


def test_existing_output_is_protected(fake_cli,tmp_path):
    load,_=fake_cli;p=tmp_path/'x';p.write_text('frozen')
    assert cli.main(['advance','--input','x','--output',str(p)])==1
    assert p.read_text()=='frozen';load.assert_not_called()


def test_gate_failure_leaves_runtime_state_unchanged(monkeypatch):
    import tectonics.genesis_volume_runtime as module
    r=VolumeRuntime();r.state=object();r.model=NS(fault_parameters=None);r.mode='be50'
    state=NS(fault_history=None);r.state=state
    def stop(*a):raise ValueError('gate stop')
    monkeypatch.setattr(module,'gate',stop)
    with pytest.raises(ValueError,match='gate'):r.advance()
    assert r.state is state


def test_wrong_dt_cannot_silently_switch_be_mode():
    r=VolumeRuntime();r.mode='be50'
    with pytest.raises(ValueError,match='dt'):r.advance(dt_myr=2.5e-5)


def test_failed_independent_audit_does_not_commit_candidate(monkeypatch):
    import tectonics.genesis_volume_runtime as module
    r=VolumeRuntime();r.mode='be50';r.state=NS(fault_history=None)
    r.model=NS(fault_parameters=None);r.chain=[];r.root={}
    original=r.state
    monkeypatch.setattr(module,'gate',lambda *a:None)
    transition=Mock(return_value=(NS(),{'candidate':True}))
    monkeypatch.setattr(module,'transition',transition)
    def reject(*a,**k):raise ValueError('independent audit failed')
    monkeypatch.setattr(module,'audit_step',reject)
    with pytest.raises(ValueError,match='independent audit'):r.advance()
    assert r.state is original and r.chain==[]
    assert transition.call_args.kwargs['max_elapsed_myr'] is None


def test_nonobject_checkpoint_is_rejected(tmp_path):
    from tectonics.genesis_volume_runtime import load_runtime
    p=tmp_path/'x.json';p.write_text('[]')
    with pytest.raises(ValueError,match='object'):load_runtime(p)


@pytest.mark.parametrize('content',['{"a":1,"a":2}','{"a":NaN}'])
def test_ambiguous_checkpoint_json_is_rejected(tmp_path,content):
    from tectonics.genesis_volume_runtime import load_runtime
    p=tmp_path/'x.json';p.write_text(content)
    with pytest.raises(ValueError):load_runtime(p)
