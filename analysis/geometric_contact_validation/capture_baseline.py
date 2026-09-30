"""Read-only provenance capture for the geometric contact experiment."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "results/genesis_runs/starter_20260928_192306_547551/starter_checkpoint.npz"
LATE = ROOT / "analysis/slab_sinking_followup/runs/ordered_sub4_dt1/elapsed_0050"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture():
    production = list((ROOT / "tectonics").glob("*.py"))
    production += list((ROOT / "configs").glob("*.yaml"))
    production += list(ROOT.glob("run_*.py"))
    sources = [SOURCE] + sorted(p for p in LATE.rglob("*") if p.is_file())
    record = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Before new geometric-contact modules; existing modified work is retained.",
        "production_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(production)},
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
    }
    target = OUT / "baseline.json"
    with target.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    print(json.dumps({"baseline": str(target), "production_files": len(production), "source_files": len(sources)}))


if __name__ == "__main__":
    capture()
