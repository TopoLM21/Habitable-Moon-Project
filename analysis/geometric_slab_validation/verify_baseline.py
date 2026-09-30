"""Compare preserved files with the pre-stage independent manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def verify(root: Path):
    directory = root / "analysis/geometric_slab_validation"
    baseline = json.loads((directory / "baseline.json").read_text(encoding="utf-8"))
    result = {}
    for category in ("production_sha256", "source_sha256", "previous_geometry_sha256",
                     "previous_geometry_tests_sha256"):
        unchanged, changed, missing = [], {}, []
        for relative, before in baseline[category].items():
            path = root / relative
            if not path.is_file():
                missing.append(relative)
                continue
            after = hashlib.sha256(path.read_bytes()).hexdigest()
            if after == before:
                unchanged.append(relative)
            else:
                changed[relative] = {"before": before, "after": after}
        result[category] = {"unchanged_count": len(unchanged), "changed": changed,
                            "missing": missing}
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    result = verify(root)
    output = root / "analysis/geometric_slab_validation/baseline_comparison.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
