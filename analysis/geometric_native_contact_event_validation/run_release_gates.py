"""Focused swept-event, constitutive-time and native geometry regression gate."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
TESTS=("native_pose","native_polygons","native_surface","native_spatial_underthrust",
       "native_underthrust_motion","native_contact_events","native_contact_time_support",
       "native_contact_event_oracle")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot():
    return {str(p.relative_to(ROOT)):digest(p) for directory in ("tectonics","tests","scripts","analysis")
            for p in sorted((ROOT/directory).rglob("*.py"))}


def write(path,value):
    with path.open("x") as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write("\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,"-m","pytest","-q",*("tests/test_geometric_"+name+".py" for name in TESTS)]
    before,started=snapshot(),time.monotonic()
    write(args.out/"sources_before.json",before)
    start_utc=datetime.now(timezone.utc).isoformat()
    with (args.out/"pytest.log").open("x") as log:
        result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    after=snapshot()
    write(args.out/"sources_after.json",after)
    changed=sorted(p for p in set(before)|set(after) if before.get(p)!=after.get(p))
    lines=(args.out/"pytest.log").read_text().splitlines()
    summary=next((line for line in reversed(lines) if re.search(r"\d+ (passed|failed|error)",line)),"")
    receipt=dict(status="PASS" if result.returncode==0 and not changed else "FAIL",command=command,
        started_utc=start_utc,finished_utc=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-started,
        exit_code=result.returncode,is_full_profile_execution=False,pytest_summary=summary,
        counts={label:int(count) for count,label in re.findall(r"(\d+) (passed|failed|errors?|skipped)",summary)},
        source_count=len(before),changed_paths=changed,runner_sha256=digest(Path(__file__)),
        sources_before_sha256=digest(args.out/"sources_before.json"),
        sources_after_sha256=digest(args.out/"sources_after.json"),log_sha256=digest(args.out/"pytest.log"))
    write(args.out/"receipt.json",receipt)
    print(json.dumps(receipt),flush=True)
    if receipt["status"]!="PASS":
        raise SystemExit(result.returncode or 1)


if __name__=="__main__":
    main()
