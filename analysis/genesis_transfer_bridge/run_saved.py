"""One pinned saved-state bridge scenario; no solver or time advancement."""
import argparse
import json
import time
from pathlib import Path

from tectonics.genesis_transfer_bridge import digest, encoded, write_bundle, require_admission

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    paths = dict(packet=Path('/workspace/genesis-exploratory-100-01/continuation/initial.json'),
                 cuts=Path('/workspace/genesis-exploratory-100-01/continuation/half2.json'),
                 source=ROOT/'analysis/genesis_volume_creation_validation/evidence/actual_continuation/direct_after_second.json')
    pins = dict(packet='4c0427d81bcb4b2a1dd8df2021e5ae1f6fedf983a9b8b2c165d69128a716bc95',
                cuts='20be57d6bae5255b3d440ea77a9da086bb32ecac06c38ada97d337ff1b0e59b9',
                source='62579a0bad128830a66e2f44579cf2b1f5d3835b691225b8d431102f3e4486d7')
    manifest,t = write_bundle(args.output, packet_path=paths['packet'], cuts_path=paths['cuts'], source_path=paths['source'], expected_sha256=pins)
    try:
        require_admission(t)
    except ValueError as exc:
        gate = str(exc)
    else:
        raise AssertionError('unexpected physical admission')
    result = dict(status='DATA_BRIDGE_VERIFIED_PHYSICAL_HANDOFF_BLOCKED', new_time_steps=0,
        time_myr=t['time_myr'], cohorts=len(t['source_cohort_ids']), checks=t['checks'],
        input_sha256=pins, bundle_manifest=manifest, physical_gate=gate,
        inventory_totals={k:v for k,v in t['inventory'].items() if k.startswith('total_') or 'residual' in k},
        velocity_relative_mass_rms=t['motion']['relative_mass_rms'],
        memory_projection_errors=t['history']['projection_error'],
        preserved_elastic_energy_j=t['history']['elastic_energy_j'],
        bytes=sum(p.stat().st_size for p in args.output.iterdir()), seconds=time.perf_counter()-start,
        code_sha256={str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in [Path(__file__), ROOT/'tectonics/genesis_transfer_bridge.py']})
    with (args.output/'scenario.json').open('xb') as f:
        f.write(encoded(result))
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
