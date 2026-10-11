"""One fixed-budget tangent-volume experiment; no radial source evaluations."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
import math
import resource
import time
from analysis.genesis_volume_bernstein_validation.control import run


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('Fresh output required')
    source=Path('/workspace/genesis-residual-tangent-Gamma-01/report.json')
    saved=json.loads(source.read_bytes());assert saved['case']=='tangent_Gamma'
    last=saved['trace'][-1]
    # JSON saved the old long-double allowances as binary64. One downward
    # binary64 step gives a conservative budget relative to that rounding.
    allowance=[F(math.nextafter(a,-math.inf)) if a else F() for a in last['allowances']]
    reference=list(map(F,last['values']))
    clock=time.monotonic();report=run(saved['source_geometry'],reference,allowance)
    report['controller_seconds']=time.monotonic()-clock
    report['peak_process_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    truth=[tuple(F(*map(int,v)) for v in pair) for pair in saved['analytic_truth_bounds']]
    report.update(analytic_truth_bounds=[[str(v) for v in row] for row in truth],
                  contains_analytic_truth_interval=all(F(l)<=a<=b<=F(u) for (l,u),(a,b) in zip(report['integral_bounds'],truth)),
                  saved_reference_format='exact value of serialized binary64 Q55; not unrecorded long-double sum',
                  allowance_policy='one binary64 step downward from saved original allowance; zero stays zero',
                  input_pin=dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
                  prior_legacy_gate_passed=saved['embedded_gate_passed'],base_commit='1017aafd0c494b01f36efe65b63510c10839da80',
                  source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                    (Path(__file__),Path('analysis/genesis_volume_bernstein_validation/control.py'))})
    with args.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    summary={k:report[k] for k in ('reason','processed_blocks','polynomial_transforms','stored_bernstein_coefficients',
          'maximum_depth','radial_evaluations','certificate_gate_passed','contains_analytic_truth_interval','controller_seconds','peak_process_rss_kib')}
    summary['bounds_float']=[[float(F(v)) for v in row] for row in report['integral_bounds']]
    summary['error_to_allowance']=[float(F(e)/a) if a else 0 for e,a in zip(report['error_bounds'],allowance)]
    print(json.dumps(summary))


if __name__=='__main__':main()
