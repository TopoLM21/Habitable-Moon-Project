"""Authenticated native signed-curve exclusion and interior cold-front control."""
from __future__ import annotations

from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time

from tectonics.geometric_native_birth_curved_pair_support import (
    certify_native_curved_pair_support, certify_native_curved_related_cold_front,
    native_curved_pair_support_from_dict, native_curved_pair_support_to_dict,
)


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()


def _write(path, value):
    payload = _encoded(value)+b"\n"
    with Path(path).open("xb") as stream:
        stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def run_curved_pair_control(epoch, omega_rad_per_myr, dt_myr, output_dir):
    """Use independently supplied typed source and a literal actual candidate.

    This run owns only new evidence files.  It reports complete two-pair
    geometric exclusion and an actual interior cold subset; it does not
    invent a constitutive carrier or claim force acceptance.
    """
    began = time.monotonic()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    components = {(row.material_kind, row.original_parent_id): row.component_id for row in epoch.components}
    rows, byte_pairs, refusals = [], [], []
    for parent in ("octant:0", "octant:1"):
        value = certify_native_curved_pair_support(epoch, omega_rad_per_myr,
            dt_myr, components[("hot_birth", "octant:5")], components[("original_live", parent)],
            precision=256, order=28,
            provenance="Current genuine mixed source; exact signed correction with full scalar new-clock proof")
        raw = native_curved_pair_support_to_dict(value)
        restored = native_curved_pair_support_from_dict(raw, expected_epoch=epoch)
        replayed = native_curved_pair_support_to_dict(restored)
        name = "hot5_to_cold"+parent.split(":")[-1]
        hashes = [_write(destination/(name+suffix+".json"), payload)
            for suffix, payload in (("", raw), ("_replayed", replayed))]
        if hashes[0] != hashes[1]:
            raise AssertionError("Semantic restart differs in canonical native signed-curve proof bytes")
        byte_pairs.append(dict(pair=name, sha256=hashes[0], identical_bytes=True))
        bad = json.loads(_encoded(raw))
        bad["curved_pieces"][0]["initial_support_constraint"] = "caller_supplied_positive_chart_sign"
        bad.pop("proof_sha256")
        bad["proof_sha256"] = hashlib.sha256(_encoded(bad)).hexdigest()
        try:
            native_curved_pair_support_from_dict(bad, expected_epoch=epoch)
        except ValueError as exc:
            refusals.append(dict(pair=name, tamper="changed true signed mask assumption; resealed root only",
                refused=True, reason=str(exc)))
        else:
            raise AssertionError("A resealed but false curved-mask assumption was accepted")
        rows.append(dict(pair=name, proof_sha256=value.proof_sha256,
            proxy_piece_count=len(value.proxy_piece_proofs), curved_piece_count=len(value.curved_piece_proofs),
            curve_methods=[row.method for row in value.curved_piece_proofs],
            entire_pair_excluded=True, force_admission_authorized=False))
    cold = certify_native_curved_related_cold_front(epoch, omega_rad_per_myr,
        dt_myr, sample_fraction=F(1, 100000), precision=256,
        provenance="Interior current physical clock; certified cold5 lower masks retain negative curve")
    cold_sha = _write(destination/"cold5_to_cold3_positive_interior.json", cold)
    cold_again = certify_native_curved_related_cold_front(epoch, omega_rad_per_myr,
        dt_myr, sample_fraction=F(1, 100000), precision=256,
        provenance=cold["provenance"])
    repeated_sha = _write(destination/"cold5_to_cold3_positive_interior_replayed.json", cold_again)
    if cold_sha != repeated_sha:
        raise AssertionError("Repeated exact native interior cold-front source differs in bytes")
    byte_pairs.append(dict(pair="cold5_to_cold3_positive_interior", sha256=cold_sha, identical_bytes=True))
    body = dict(format="native-curved-birth-front-control-1",
        source_epoch_sha256=epoch.epoch_sha256,
        omega_rad_per_myr=[[owner, list(vector)] for owner, vector in omega_rad_per_myr],
        dt_myr=dt_myr, exclusions=rows, semantic_refusals=refusals, byte_pairs=byte_pairs,
        cold_front_witness_sha256=cold["proof_sha256"],
        cold_front_starts_at_arbitrarily_small_positive_time=cold["initial_contact_starts_at_arbitrarily_small_positive_time"],
        interior_cold_positive_subset_count=len(cold["witnesses"]),
        interior_cold_subset_absent_at_endpoint=all(not row["full_interval_endpoint_has_positive_inner_subset"] for row in cold["witnesses"]),
        terminal_cold_contact_clock_proved=False,
        signed_curved_correction_replaced_by_polygon=False,
        exact_masks_preserved=True, constitutive_time_admission_proved=False,
        candidate_mechanics_adopted=False, force_admission_authorized=False)
    body["report_sha256"] = hashlib.sha256(_encoded(body)).hexdigest()
    _write(destination/"report.json", body)
    return dict(report=body, elapsed_s=time.monotonic()-began)
