"""Reproduce opt-in legacy direction translation without advancing sources."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from run_fractional_transport_probe import DEFAULT_SOURCE, digest, load_probe_source
from tectonics.fractional_surface_io import surface_from_lithosphere
from tectonics.geometric_polarity_source import legacy_polarity_evidence
from tectonics.geometric_surface import from_fractional_surface


def main():
    report = {}
    sources = dict(starter=DEFAULT_SOURCE,
        saved50=ROOT/"analysis/slab_sinking_followup/runs/ordered_sub4_dt1/elapsed_0050")
    for name, source in sources.items():
        mesh, checkpoint, fracture, model, provenance = load_probe_source(source)
        fractional = surface_from_lithosphere(mesh, checkpoint.state, provenance["radius_km"],
                                              fracture_memory=fracture.memory)
        surface = from_fractional_surface(mesh, fractional, provenance["radius_km"])
        evidence, audit = legacy_polarity_evidence(mesh, surface,
            checkpoint.subduction_memory.young_boundary_state)
        if any(digest(path) != expected for path, expected in provenance["source_sha256"].items()):
            raise RuntimeError("Source hashes changed during legacy polarity translation")
        report[name] = dict(provenance=provenance, audit=audit, source_hashes_unchanged=True)
    output = Path(__file__).with_name("source_polarity_mapping.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({name: row["audit"]["counts"] for name, row in report.items()}))


if __name__ == "__main__":
    main()
