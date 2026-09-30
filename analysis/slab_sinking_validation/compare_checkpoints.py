"""Compare scientific state arrays and metadata, excluding output/log paths."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def diff(first, second, prefix=""):
    if isinstance(first, dict) and isinstance(second, dict):
        return [item for key in sorted(first.keys() | second.keys())
            for item in diff(first.get(key), second.get(key), f"{prefix}.{key}" if prefix else key)]
    if isinstance(first, list) and isinstance(second, list) and len(first) == len(second):
        return [item for index, (a, b) in enumerate(zip(first, second)) for item in diff(a, b, f"{prefix}[{index}]")]
    return [] if first == second else [dict(field=prefix, first=first, second=second)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arrays = {}
    for name in ("mature_checkpoint/state.npz", "young_context/starter_checkpoint.npz", "young_context/fracture_memory.npz"):
        with np.load(args.first / name, allow_pickle=False) as first, np.load(args.second / name, allow_pickle=False) as second:
            shared = first.files == second.files
            different = []
            for key in first.files if shared else []:
                numeric = first[key].dtype.kind in "biufc"
                if not np.array_equal(first[key], second[key], equal_nan=numeric):
                    record = dict(array=key)
                    if numeric and first[key].shape == second[key].shape:
                        a, b = first[key].astype(float), second[key].astype(float)
                        record["maximum_absolute_difference"] = float(np.nanmax(np.abs(a-b)))
                        record["relative_norm_difference"] = float(np.linalg.norm(a-b)/max(np.linalg.norm(a), np.linalg.norm(b), 1.))
                    different.append(record)
            arrays[name] = dict(same_array_names=shared, bitwise_identical=shared and not different, differences=different)
    metadata_differences = diff(read(args.first / "mature_checkpoint/meta.json"), read(args.second / "mature_checkpoint/meta.json"))
    first, second = read(args.first / "continuation.json"), read(args.second / "continuation.json")
    selected = ("checks", "material_ledger", "accepted_slab_inventory", "history", "young_fracture", "young_fracture_events")
    report_differences = diff({key:first.get(key) for key in selected}, {key:second.get(key) for key in selected})
    result = dict(first=str(args.first.resolve()), second=str(args.second.resolve()),
        execution=[first["execution"], second["execution"]], arrays=arrays,
        scientific_metadata_equal=not metadata_differences, scientific_metadata_differences=metadata_differences,
        physical_report_equal=not report_differences, physical_report_differences=report_differences,
        all_checks_passed=all(first["checks"].values()) and all(second["checks"].values()))
    result["scientific_state_bitwise_identical"] = all(value["bitwise_identical"] for value in arrays.values()) and not metadata_differences and not report_differences
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key:result[key] for key in ("scientific_state_bitwise_identical", "all_checks_passed")}))
    if not result["scientific_state_bitwise_identical"] or not result["all_checks_passed"]:
        raise SystemExit("Scientific checkpoint comparison was not bitwise identical")


if __name__ == "__main__":
    main()
