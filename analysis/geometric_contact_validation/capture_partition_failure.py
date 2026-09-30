"""Capture the exact common-rotation pair rejected by partition validation."""
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tectonics.fractional_surface import SurfaceParcel
from tectonics.geometric_surface import GeometricFragment, candidate_pairs
from tectonics.spherical_polygons import GeometryDiagnostics, intersect_convex, polygon_area

source = Path(__file__).with_name("starter_common_sub4")/"geometric_checkpoint.json"
raw = json.loads(source.read_text(encoding="utf-8"))["state"]
fragments = tuple(GeometricFragment(f["fragment_id"], tuple(map(tuple, f["polygon"])),
    SurfaceParcel(**f["parcel"]), f.get("parent_fragment_id")) for f in raw["fragments"])
for i, j in candidate_pairs(fragments):
    diag = GeometryDiagnostics()
    overlap = intersect_convex(fragments[i].polygon, fragments[j].polygon, diagnostics=diag)
    if len(overlap):
        report = {"source": str(source), "indices": [i, j], "first": asdict(fragments[i]),
                  "second": asdict(fragments[j]), "overlap": overlap.tolist(),
                  "overlap_area_steradians": polygon_area(overlap), "diagnostics": asdict(diag)}
        target = source.parent/"first_false_overlap.json"
        with target.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(json.dumps({"target": str(target), "indices": [i, j],
                          "overlap_area_steradians": report["overlap_area_steradians"], "diagnostics": asdict(diag)}))
        break
else:
    print("No overlap on current geometry code")
