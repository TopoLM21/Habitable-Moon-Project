"""Compare checkpoint arrays and metadata exactly and with numeric tolerances."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def compare_metadata(left, right, *, rtol: float, atol: float, path: str = "meta", differences=None):
    if differences is None:
        differences = []
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            differences.append({"path": path, "kind": "keys"})
            return differences
        for key in left:
            compare_metadata(
                left[key], right[key], rtol=rtol, atol=atol,
                path=f"{path}.{key}", differences=differences,
            )
        return differences
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            differences.append({"path": path, "kind": "length", "left": len(left), "right": len(right)})
            return differences
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            compare_metadata(
                a, b, rtol=rtol, atol=atol,
                path=f"{path}[{index}]", differences=differences,
            )
        return differences
    if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)) and not isinstance(right, bool):
        if not math.isclose(float(left), float(right), rel_tol=rtol, abs_tol=atol):
            differences.append({
                "path": path,
                "kind": "number",
                "left": left,
                "right": right,
                "absolute_difference": abs(float(right) - float(left)),
            })
        return differences
    if left != right:
        differences.append({"path": path, "kind": "value", "left": left, "right": right})
    return differences


def compare_checkpoints(left: Path, right: Path, *, rtol: float, atol: float) -> dict:
    array_rows = []
    arrays_exact = True
    arrays_within_tolerance = True
    with np.load(left / "state.npz", allow_pickle=False) as a, np.load(right / "state.npz", allow_pickle=False) as b:
        if a.files != b.files:
            return {
                "exact": False,
                "within_tolerance": False,
                "array_keys_equal": False,
                "left_keys": a.files,
                "right_keys": b.files,
            }
        for name in a.files:
            expected = a[name]
            actual = b[name]
            exact = expected.dtype == actual.dtype and expected.shape == actual.shape and expected.tobytes() == actual.tobytes()
            arrays_exact &= exact
            if exact:
                continue
            if expected.dtype.kind not in "fc" or expected.shape != actual.shape:
                within = False
                row = {"name": name, "exact": False, "within_tolerance": False}
            else:
                difference = np.abs(actual - expected)
                within = bool(np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True))
                row = {
                    "name": name,
                    "exact": False,
                    "within_tolerance": within,
                    "different_values": int(np.count_nonzero(actual != expected)),
                    "max_absolute_difference": float(np.nanmax(difference)),
                    "rms_difference": float(np.sqrt(np.nanmean(difference * difference))),
                }
            arrays_within_tolerance &= within
            array_rows.append(row)
    left_meta = json.loads((left / "meta.json").read_text(encoding="utf-8"))
    right_meta = json.loads((right / "meta.json").read_text(encoding="utf-8"))
    metadata_exact = left_meta == right_meta
    metadata_differences = compare_metadata(left_meta, right_meta, rtol=rtol, atol=atol)
    return {
        "exact": bool(arrays_exact and metadata_exact),
        "within_tolerance": bool(arrays_within_tolerance and not metadata_differences),
        "rtol": rtol,
        "atol": atol,
        "arrays_compared": len(a.files),
        "array_differences": array_rows,
        "metadata_exact": metadata_exact,
        "metadata_differences": metadata_differences,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--rtol", type=float, default=1.0e-12)
    parser.add_argument("--atol", type=float, default=1.0e-10)
    args = parser.parse_args()
    print(json.dumps(compare_checkpoints(
        args.left.resolve(), args.right.resolve(), rtol=args.rtol, atol=args.atol
    ), indent=2))


if __name__ == "__main__":
    main()
