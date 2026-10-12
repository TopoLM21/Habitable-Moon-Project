from dataclasses import replace

from tectonics.genesis import GenesisParameters
from tectonics.genesis_shell import ShellParameters
from tectonics.genesis_starter import StarterModel
from tectonics.genesis_tides import TidalParameters
from tectonics.genesis_transfer_bridge import assess_starter_entry
from tectonics.mesh import build_icosphere


def test_single_shell_is_valid_starter_input_without_partition_or_seeding():
    m = StarterModel(build_icosphere(1), GenesisParameters(), TidalParameters(), ShellParameters(subdivisions=1))
    state = m.initial_state()
    report = assess_starter_entry(m,state)
    assert report['admitted'] and report['owner_count'] == 1
    assert not report['single_owner_is_blocker']
    assert state.events == [] and state.time_myr == 0
    stopped = assess_starter_entry(m,replace(state,stopped_reason='thermal_stop'))
    assert not stopped['admitted'] and stopped['code'] == 'NATIVE_STARTER_STOPPED'


def test_volume_packet_is_blocked_for_memory_not_owner_count():
    r = assess_starter_entry(None,dict(source_owner_ids=['one-body']))
    assert not r['admitted'] and not r['single_owner_is_blocker']
    assert r['code'] == 'VOLUME_TO_STARTER_MEMORY_CLOSURE_REQUIRED'
