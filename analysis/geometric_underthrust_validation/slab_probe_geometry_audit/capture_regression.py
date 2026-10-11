"""Capture the existing slab probe regression without changing its algorithms."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import pytest

import tests.test_geometric_slab_probe as regression
import tectonics.geometric_transport as transport
from tectonics.geometric_surface import candidate_pairs
from tectonics.spherical_polygons import intersect_convex, polygon_area


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    options = parser.parse_args()
    options.out.mkdir(parents=True, exist_ok=False)
    monkeypatch = pytest.MonkeyPatch()
    source = regression.source.__wrapped__(monkeypatch, options.out)
    original_audit = transport.audit_partition
    captured = []

    def capture(state, **kwargs):
        try:
            return original_audit(state, **kwargs)
        except ValueError as error:
            if str(error) != "Geometric partition contains overlapping footprints":
                raise
            overlaps = []
            for i, j in candidate_pairs(state.fragments):
                overlap = intersect_convex(state.fragments[i].polygon, state.fragments[j].polygon)
                if len(overlap):
                    overlaps.append(dict(i=i, j=j, fragment_a=asdict(state.fragments[i]),
                        fragment_b=asdict(state.fragments[j]), overlap=overlap.tolist(),
                        area_sr=polygon_area(overlap)))
            captured.append(dict(state=asdict(state), overlaps=overlaps))
            raise

    monkeypatch.setattr(transport, "audit_partition", capture)
    try:
        regression.runner.execute_probe(source.path, options.out / "full", .25, 2)
    except ValueError as error:
        if not captured:
            raise
        captured[-1]["exception"] = str(error)
        (options.out / "capture.json").write_text(json.dumps(captured[-1], indent=2) + "\n")
        print(json.dumps(dict(status="captured_existing_regression", output=str(options.out),
            time_myr=captured[-1]["state"]["time_myr"], pairs=len(captured[-1]["overlaps"]))))
    else:
        raise RuntimeError("The recorded regression was not reproduced on this source version")
    finally:
        monkeypatch.undo()


if __name__ == "__main__":
    main()
