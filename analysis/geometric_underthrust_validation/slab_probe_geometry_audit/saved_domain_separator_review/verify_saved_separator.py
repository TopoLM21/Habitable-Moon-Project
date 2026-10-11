"""Independent read-only exact-sign and public-validation audit.

Run with a new --out JSON path.  This checks the frozen saved-domain empty
proof; it does not claim an exact rational implementation of every clip.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[3]
sys.path.insert(0, str(REPOSITORY))
import tectonics.spherical_polygons as polygons


def exact_determinant(rows):
    a, b, vertex = (tuple(Fraction(float(x)) for x in row) for row in rows)
    return sum(a[i]*(b[j]*vertex[k]-b[k]*vertex[j])
               for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)))


def verify():
    receipt = {
        "kernel_sha256": hashlib.sha256(Path(polygons.__file__).read_bytes()).hexdigest(),
        "method": "Fraction exact binary64 determinant oracle; isolated LD64 and extended arithmetic; public validation checks",
        "cases": [],
    }
    original_precision = polygons._LD
    try:
        for dtype in (np.longdouble, np.float64):
            polygons._LD = dtype
            for name in ("underflow_false_sign.json", "underflow_rounded_zero.json"):
                case = json.loads((HERE/name).read_text())
                rows = [np.asarray(case[key], dtype=np.float64)
                        for key in ("a", "b", "vertex")]
                for swapped in (False, True):
                    ordered = (rows[1], rows[0], rows[2]) if swapped else rows
                    exact = exact_determinant(ordered)
                    expected = (exact > 0)-(exact < 0)
                    assert expected == (1 if swapped else -1)
                    side = polygons._saved_ray_side(*ordered, {})
                    assert side == expected
                    value, bound = polygons._determinant_value_and_error(*ordered)
                    receipt["cases"].append({
                        "kind": "exact_saved_side", "dtype": str(dtype),
                        "case": name, "swapped": swapped, "side": int(side),
                        "floating_value": str(value), "floating_bound": str(bound),
                        "exact_numerator": str(exact.numerator),
                        "exact_denominator": str(exact.denominator),
                    })
        polygons._LD = original_precision
        valid = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
        invalids = {
            "two_vertices": [[0., -1., 0.], [0., 0., -1.]],
            "zero_area": [[1., 0., 0.], [1., 1., 0.], [1., 2., 0.]],
            "nonfinite": [[1., 0., 0.], [0., 1., 0.], [0., 0., float("nan")]],
            "antipodal": [[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.]],
            "hemisphere_spanning": [[1., 0., 0.], [0., 1., 0.], [-1., 0., 0.], [0., -1., 0.]],
        }
        for label, invalid in invalids.items():
            try:
                polygons.intersect_convex(valid, invalid)
            except ValueError as error:
                receipt["cases"].append({"kind": "public_invalid_rejection",
                    "case": label, "error": str(error)})
            else:
                raise AssertionError("Invalid public input accepted: "+label)
        assert not polygons._saved_binary64_rays(np.asarray(valid, dtype=np.longdouble))
        assert not polygons._saved_binary64_rays(
            [[1., 2**60+1, 0.], [0., 1., 0.], [0., 0., 1.]])
        assert not polygons._saved_binary64_rays(polygons.normalize_polygon(valid))
        receipt["excluded_ordinary_precisions"] = [
            "longdouble", "mixed_large_integer", "inherited_supported_polygon"]
        receipt["passed"] = len(receipt["cases"]) == 13
        return receipt
    finally:
        polygons._LD = original_precision


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("Refusing to replace an existing audit receipt")
    result = verify()
    args.out.write_text(json.dumps(result, indent=2)+"\n")
    print("13 cases passed; kernel "+result["kernel_sha256"])
