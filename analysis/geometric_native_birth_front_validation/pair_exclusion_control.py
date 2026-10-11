"""Exclusive evidence for complete current pair separating-support proofs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

from tectonics.geometric_native_birth_pair_exclusion import (
    UnresolvedNativePairExclusion, certify_native_birth_pair_exclusion,
    native_birth_pair_exclusion_to_dict, native_birth_pair_exclusion_from_dict,
)
from tectonics.geometric_native_implicit_epoch_admission import _portable_fingerprint_sha256
from tectonics.geometric_native_implicit_material_epoch import validate_native_implicit_material_epoch


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()


def _write(path, value):
    with path.open("xb") as stream:
        stream.write(_encoded(value)+b"\n")


def run_pair_exclusion_control(out, *, epoch, omega_rad_per_myr, dt_myr,
        pair_inventory, precision=256, order=28, time_cells=1, provenance):
    """Prove each selected unresolved pair or retain an explicit refusal.

    This control must receive the independently authenticated genuine force
    vectors and inventory.  Reclassification here is strictly geometrical;
    the positive material pairs still require their own physical time proofs.
    """
    epoch = validate_native_implicit_material_epoch(epoch)
    if (pair_inventory["source_epoch_sha256"] != epoch.epoch_sha256
            or pair_inventory["complete_source_epoch_fingerprint_sha256"]
                != _portable_fingerprint_sha256(epoch)
            or _encoded(pair_inventory["omega_rad_per_myr"])
                != _encoded([[owner, list(vector)] for owner, vector in omega_rad_per_myr])
            or type(pair_inventory["dt_myr"]) is not float
            or pair_inventory["dt_myr"].hex() != dt_myr.hex()):
        raise ValueError("Pair inventory differs from the independent complete current source and force motion")
    out = Path(out)
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    rows, byte_pairs = [], []
    unresolved = tuple(row for row in pair_inventory["pairs"] if row["status"] == "unresolved_upper_only")
    for index, row in enumerate(unresolved, 1):
        key = row["donor_component_id"], row["receiver_component_id"]
        arguments = dict(precision=precision, order=order, time_cells=time_cells,
            provenance=provenance)
        observation = {name: row[name] for name in ("donor_component_id", "receiver_component_id",
            "donor_material_kind", "receiver_material_kind", "donor_original_parent_id",
            "receiver_original_parent_id", "involves_hot_material")}
        try:
            certificate = certify_native_birth_pair_exclusion(epoch, *key,
                omega_rad_per_myr, dt_myr, **arguments)
        except UnresolvedNativePairExclusion as error:
            rows.append(dict(observation, status="unresolved", message=str(error)))
            continue
        raw = native_birth_pair_exclusion_to_dict(certificate)
        path = out / f"pair_{index:02d}.json"
        _write(path, raw)
        decoded = json.loads(path.read_bytes())
        replay = native_birth_pair_exclusion_from_dict(decoded, expected_epoch=epoch,
            expected_donor_component_id=key[0], expected_receiver_component_id=key[1],
            expected_omega_rad_per_myr=omega_rad_per_myr, expected_dt_myr=dt_myr,
            expected_precision=precision, expected_order=order, expected_time_cells=time_cells,
            expected_provenance=provenance)
        replayed_path = out / f"pair_{index:02d}_replayed.json"
        _write(replayed_path, native_birth_pair_exclusion_to_dict(replay))
        if path.read_bytes() != replayed_path.read_bytes():
            raise AssertionError("Independent complete source replay changed the pair proof")
        byte_pairs.append([path.name, replayed_path.name])
        rows.append(dict(observation, status="excluded_for_whole_interval",
            proof_path=path.name, proof_sha256=raw["proof_sha256"],
            upper_union_piece_pair_count=len(certificate.piece_proofs),
            vertex_sign_proof_count=sum(len(piece.vertex_proofs) for piece in certificate.piece_proofs),
            exact_birth_incidence_division_count=sum(vertex.divided_by_birth_clock
                for piece in certificate.piece_proofs for vertex in piece.vertex_proofs)))
    complete = sum(row["status"] == "excluded_for_whole_interval" for row in rows)
    previous_zero = pair_inventory["classification_counts"]["excluded_for_whole_interval"]
    report = dict(format="native-correlated-current-pair-exclusion-control-1",
        status="PASS", seconds=time.monotonic()-started,
        source_epoch_sha256=epoch.epoch_sha256,
        complete_source_epoch_fingerprint_sha256=_portable_fingerprint_sha256(epoch),
        actual_euler_vectors=[[owner, list(vector)] for owner, vector in omega_rad_per_myr],
        dt_myr=dt_myr, precision=precision, order=order, time_cells=time_cells,
        provenance=provenance, previous_unresolved_pair_count=len(unresolved),
        newly_complete_zero_area_pair_count=complete,
        remaining_unresolved_pair_count=len(unresolved)-complete,
        inherited_whole_interval_exclusion_count=previous_zero,
        total_whole_interval_exclusion_count=previous_zero+complete,
        positive_material_pair_count=pair_inventory["classification_counts"]["positive_material_subset"],
        rows=rows, byte_pairs=byte_pairs,
        positive_area_tolerance_used=False, actual_euler_vectors_changed=False,
        signed_material_corrections_discarded=False,
        material_area_debited=False, force_geometry_admission_authorized=False,
        whole_native_interval_admitted=False)
    report["report_sha256"] = hashlib.sha256(_encoded(report)).hexdigest()
    _write(out / "report.json", report)
    return report
