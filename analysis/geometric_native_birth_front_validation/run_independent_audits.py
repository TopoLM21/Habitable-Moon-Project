"""Run independent arithmetic in a fresh stdlib-only auditor process."""
from __future__ import annotations

from pathlib import Path
import argparse
import sys
import time

from .validation_support import ROOT,guard_sources,digest,read,write
from .independent_continuum_audit import audit_continuum
from .independent_pair_polynomial_audit import audit_pair_polynomials
from .independent_curved_pair_audit import audit_curved_pair


def run_audits(control_directory,out):
    control_directory,out=Path(control_directory),Path(out)
    out.mkdir(exist_ok=False)
    started=time.monotonic()
    before,_=guard_sources(require_frozen=True)
    if any(name.startswith("tectonics.") for name in sys.modules):
        raise ValueError("Independent arithmetic must run in a fresh process without production mechanics imports")
    candidate=read(control_directory/"candidate_packet.json")
    source=candidate["before"]["snapshot"]["current_material_epoch"]
    network=write(out/"full_continuum_network.json",audit_continuum(control_directory/"candidate_packet.json"),"receipt_sha256")
    pair_paths=[path for path in sorted((control_directory/"pair_exclusions").glob("pair_*.json"))
        if not path.stem.endswith("_replayed")]
    pair_paths.extend(control_directory/"curved_pairs"/(name+".json") for name in ("hot5_to_cold0","hot5_to_cold1"))
    polynomial=write(out/"all_scalar_polynomials.json",audit_pair_polynomials(pair_paths,expected_time_cells=1),"receipt_sha256")
    curves=[]
    for name in ("hot5_to_cold0","hot5_to_cold1"):
        target=out/(name+"_signed_curve.json")
        receipt=audit_curved_pair(control_directory/"curved_pairs"/(name+".json"),
            source_epoch_payload=source,candidate_payload=candidate,output_path=target)
        if receipt["whole_signed_curved_component_exclusion_independently_proved"]is not True:
            raise ValueError("The independent signed curved-component proof failed")
        curves.append(dict(path=str(target.relative_to(ROOT)),sha256=digest(target)))
    after,_=guard_sources(require_frozen=True)
    if before!=after or any(name.startswith("tectonics.") for name in sys.modules):
        raise ValueError("The independent arithmetic process imported production modules or changed source")
    audits=[dict(path=str((out/name).relative_to(ROOT)),sha256=digest(out/name))
        for name in ("full_continuum_network.json","all_scalar_polynomials.json")]+curves
    return write(out/"report.json",dict(format="native-birth-front-independent-audit-suite-1",status="PASS",
        elapsed_s=time.monotonic()-started,source_count=len(before),fresh_process_without_production_modules=True,
        input_candidate=dict(path=str((control_directory/"candidate_packet.json").relative_to(ROOT)),
            sha256=digest(control_directory/"candidate_packet.json")),
        auditor_sources={name:before[name] for name in before if name.startswith(
            "analysis/geometric_native_birth_front_validation/independent_")},
        independent_audits=audits,audit_count=len(audits),scalar_vertex_count=polynomial["vertex_count"],
        ideal_force_relative_error=network["maximum_per_coordinate_relative_error"],
        stored_reaction_relative_error=network["maximum_actual_per_coordinate_relative_error"],
        complete_work_relative_error=network["complete_work_balance_relative_error"],
        force_equations_resolved_here=False,whole_native_interval_admitted=False,whole_point_1_completed=False),"report_sha256")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--control",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    result=run_audits(args.control.resolve(),args.out.resolve())
    print({name:result[name] for name in ("status","elapsed_s","audit_count","scalar_vertex_count")})
