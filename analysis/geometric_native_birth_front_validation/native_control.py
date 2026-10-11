"""Frozen genuine continuum, complete pair inventory and explicit admission limits."""
from __future__ import annotations

from fractions import Fraction as F
from pathlib import Path
import time
import traceback

from .validation_support import ROOT,guard_sources,digest,encoded,read,write,value_sha,sealed


def _same_motion(packet,epoch,omega,duration):
    if (packet["source_epoch_sha256"]!=epoch.epoch_sha256
            or encoded(packet["omega_rad_per_myr"])!=encoded(omega)
            or type(packet["dt_myr"])is not float or packet["dt_myr"].hex()!=duration.hex()):
        raise ValueError("The geometry proof differs from the complete genuine force motion")


def _reference(path):
    return dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=digest(path))


def run_native_control(out):
    """A complete source experiment; failed physical adoption remains explicit."""
    out=Path(out)
    out.mkdir(exist_ok=False)
    started=time.monotonic()
    before,original=guard_sources(require_frozen=True)
    write(out/"sources_before.json",before)
    stages={}
    def timed(name,function):
        tick=time.monotonic()
        print("FRONT control starting",name,flush=True)
        value=function()
        stages[name]=time.monotonic()-tick
        print("FRONT control finished",name,round(stages[name],3),flush=True)
        return value
    body=dict(format="native-birth-front-genuine-complete-control-1",status="REFUSED",
        inherited_source_count=751,source_count=len(before),accepted_original_native_intervals=1,
        accepted_new_native_intervals=0,whole_point_1_completed=False)
    try:
        from .native_continuum_case import build_native_continuum_case,solve_native_continuum_case,export_native_continuum_case
        from tectonics.thermal import SECONDS_PER_MYR
        from tectonics.geometric_native_implicit_underthrust_motion import native_implicit_underthrust_motion_to_dict
        from tectonics.geometric_native_birth_finite_pose import native_birth_candidate_finite_pose_to_dict
        from tectonics.geometric_native_pose import FractionInterval as I
        from analysis.geometric_native_birth_contact_validation.coverage_inventory import build_coverage_inventory,write_inventory
        from .pair_exclusion_control import run_pair_exclusion_control
        from .curved_pair_control import run_curved_pair_control
        from .run_front_control import run_front_control
        from tectonics.geometric_native_birth_curved_pair_support import (
            certify_native_curved_related_cold_front,certify_native_cold5_receiver1_immediate_slice)
        case=timed("canonical_genuine_source",build_native_continuum_case)
        candidate=timed("actual_full_continuum_force",lambda:solve_native_continuum_case(case,
            provenance="Frozen genuine native full-arc force and complete current-pair control"))
        packet=timed("strict_full_source_and_force_replay",lambda:export_native_continuum_case(case,candidate))
        write(out/"candidate_packet.json",packet)
        saved=read(ROOT/"analysis/geometric_native_active_line_validation/control_release_01/motion/accepted_motion_before.json")
        if encoded(saved)!=encoded(native_implicit_underthrust_motion_to_dict(case["previous"])):
            raise ValueError("The complete first accepted original history changed")
        epoch=case["previous"].current_surface
        omega=candidate.omega_rad_per_myr
        duration=candidate.dt_s/SECONDS_PER_MYR
        inventory=timed("fresh_all_42_pair_inventory",lambda:build_coverage_inventory(epoch,omega,duration,
            time_cells=1,precision=128,order=20,
            provenance="Frozen actual full-arc continuum force: complete native pair inventory, no adoption authority",
            progress=lambda row:print("FRONT pair",row["donor_original_parent_id"],row["receiver_original_parent_id"],
                row["status"],flush=True)))
        write_inventory(out/"pair_inventory.json",inventory)
        pair_report=timed("whole_clock_correlated_pair_exclusions",lambda:run_pair_exclusion_control(out/"pair_exclusions",
            epoch=epoch,omega_rad_per_myr=omega,dt_myr=duration,pair_inventory=inventory["inventory"],
            precision=256,order=28,time_cells=1,
            provenance="Frozen actual full-arc continuum force: source-bound whole-clock pair exclusions"))
        curve_report=timed("signed_curved_pair_exclusions_and_cold_transient",lambda:run_curved_pair_control(
            epoch,omega,duration,out/"curved_pairs"))
        immediate=timed("exact_cold5_receiver1_immediate_slice",lambda:certify_native_cold5_receiver1_immediate_slice(
            epoch,omega,duration,precision=256,
            provenance="Frozen actual full-arc continuum force: canonical first-entry receiving-history slice"))
        _same_motion(immediate,epoch,omega,duration)
        write(out/"cold5_receiver1_immediate_slice.json",immediate)
        # The read-only physical review was made on the archived development
        # case. Authenticate that complete source and every computed field,
        # rather than attaching its old checksum to a new force declaration.
        old_candidate_path=ROOT/"analysis/geometric_native_birth_front_validation/development_native_continuum_01/candidate_packet.json"
        old_slice_path=ROOT/"analysis/geometric_native_birth_front_validation/development_immediate_slice_03/cold5_receiver1_immediate_slice.json"
        review_path=ROOT/"analysis/geometric_native_birth_front_validation/mechanical_admission_review.json"
        review=sealed(review_path,"review_sha256")
        old_candidate,old_slice=read(old_candidate_path),read(old_slice_path)
        meaningful_candidate=lambda value:{name:item for name,item in value.items()
            if name not in ("provenance","candidate_sha256")}
        meaningful_slice=lambda value:{name:item for name,item in value.items()
            if name not in ("provenance","proof_sha256")}
        if (encoded(meaningful_candidate(old_candidate))!=encoded(meaningful_candidate(packet))
                or encoded(meaningful_slice(old_slice))!=encoded(meaningful_slice(immediate))):
            raise ValueError("The preserved mechanical review's complete physical case differs from this final force and entry slice")
        review_source=review["source"]
        for key,path in (("force_candidate",old_candidate_path),("immediate_slice",old_slice_path)):
            reference=review_source[key]
            if reference.get("file_sha256")!=digest(path) or reference["path"]!=str(path.relative_to(ROOT)):
                raise ValueError("The preserved mechanical review's raw source reference differs")
        write(out/"mechanical_review_authority_equivalence.json",dict(
            format="native-birth-front-physical-review-authority-equivalence-1",status="PASS",
            final_force_candidate_sha256=packet["candidate_sha256"],
            preserved_development_candidate_sha256=old_candidate["candidate_sha256"],
            entire_candidate_physical_payload_identical=True,
            entire_first_entry_slice_physical_payload_identical=True,
            candidate_fields_excluded_only=["candidate_sha256","provenance"],
            slice_fields_excluded_only=["proof_sha256","provenance"],
            complete_candidate_physical_payload_sha256=value_sha(meaningful_candidate(packet)),
            complete_slice_physical_payload_sha256=value_sha(meaningful_slice(immediate)),
            preserved_development_force=_reference(old_candidate_path),
            preserved_development_slice=_reference(old_slice_path),preserved_physical_review=_reference(review_path),
            actual_cold_incoming_up_dip_counterexample_preserved=True,
            native_mechanical_admission_authorized=False),"equivalence_sha256")
        witness=timed("cold7_receiver2_interior_positive_subset",lambda:certify_native_curved_related_cold_front(
            epoch,omega,duration,donor_original_parent_id="octant:7",receiver_original_parent_id="octant:2",
            sample_fraction=F(1,100000),precision=256,
            provenance="Frozen actual full-arc continuum force: retained cold7/cold2 interior transient"))
        _same_motion(witness,epoch,omega,duration)
        repeated=timed("cold7_receiver2_positive_subset_replay",lambda:certify_native_curved_related_cold_front(
            epoch,omega,duration,donor_original_parent_id="octant:7",receiver_original_parent_id="octant:2",
            sample_fraction=F(1,100000),precision=256,
            provenance="Frozen actual full-arc continuum force: retained cold7/cold2 interior transient"))
        write(out/"cold7_receiver2_positive_interior.json",witness)
        write(out/"cold7_receiver2_positive_interior_replayed.json",repeated)
        if encoded(witness)!=encoded(repeated):
            raise ValueError("Complete genuine source replay changed the interior positive material subset")
        front_report=timed("full_native_frames_and_causal_hot_front",lambda:run_front_control(out/"causal_front",
            mesh=case["mesh"],previous=case["previous"],load=case["load"],support=case["support"],
            candidate=candidate,expected_candidate_packet=packet,require_frozen=True))
        endpoint=timed("actual_full_SO3_endpoint_anchor_fields",lambda:native_birth_candidate_finite_pose_to_dict(candidate,
            mesh=case["mesh"],previous=case["previous"],load=case["load"],support=case["support"],
            time_fraction_bounds=I(1),precision=256,order=28))
        write(out/"finite_pose_endpoint_fields.json",endpoint)
        wholeclock=timed("actual_full_SO3_whole_clock_anchor_fields",lambda:native_birth_candidate_finite_pose_to_dict(candidate,
            mesh=case["mesh"],previous=case["previous"],load=case["load"],support=case["support"],
            time_fraction_bounds=I(0,1),precision=256,order=28))
        write(out/"finite_pose_whole_clock_fields.json",wholeclock)
        for fields in (endpoint,wholeclock):
            if fields["force_candidate_sha256"]!=packet["candidate_sha256"]:
                raise ValueError("The finite-pose field uses a different genuine force candidate")
        cold5witness=read(out/"curved_pairs/cold5_to_cold3_positive_interior.json")
        _same_motion(cold5witness,epoch,omega,duration)
        curve_zero=read(out/"curved_pairs/hot5_to_cold1.json")
        _same_motion(curve_zero,epoch,omega,duration)
        pair_rows={(row["donor_component_id"],row["receiver_component_id"]):row for row in pair_report["rows"]}
        curve_key=(curve_zero["donor_component_id"],curve_zero["receiver_component_id"])
        transient_keys={(row["donor_component_id"],row["receiver_component_id"]):row for row in (cold5witness,witness)}
        coverage=[]
        for row in inventory["inventory"]["pairs"]:
            key=(row["donor_component_id"],row["receiver_component_id"])
            status=row["status"]
            if status=="unresolved_upper_only":
                if pair_rows.get(key,{}).get("status")=="excluded_for_whole_interval":
                    status="excluded_for_whole_interval"
                elif key==curve_key:
                    status="excluded_for_whole_interval"
                elif key in transient_keys:
                    status="positive_interior_material_subset"
                else:
                    raise ValueError("A complete current material pair remains geometrically unclassified")
            coverage.append(dict(donor_component_id=key[0],receiver_component_id=key[1],
                donor_original_parent_id=row["donor_original_parent_id"],receiver_original_parent_id=row["receiver_original_parent_id"],
                involves_hot_material=row["involves_hot_material"],status=status))
        if len(coverage)!=42 or len({(row["donor_component_id"],row["receiver_component_id"]) for row in coverage})!=42:
            raise ValueError("The canonical current geometry inventory is not exhaustive")
        counts={name:sum(row["status"]==name for row in coverage) for name in sorted({row["status"] for row in coverage})}
        hot_counts={name:sum(row["status"]==name and row["involves_hot_material"] for row in coverage) for name in counts}
        write(out/"complete_geometric_pair_classification.json",dict(
            format="native-birth-front-complete-geometric-pair-classification-1",status="PASS",
            source_epoch_sha256=epoch.epoch_sha256,force_candidate_sha256=packet["candidate_sha256"],
            omega_rad_per_myr=omega,dt_myr=duration,pair_count=42,classification_counts=counts,
            hot_pair_count=sum(row["involves_hot_material"] for row in coverage),hot_classification_counts=hot_counts,
            pairs=coverage,positive_area_tolerance_used=False,
            complete_positive_pair_contact_time_domains_proved=False,native_mechanical_admission_authorized=False),"classification_sha256")
        body.update(status="PASS",candidate_sha256=packet["candidate_sha256"],
            source_epoch_sha256=epoch.epoch_sha256,before_state_sha256=case["previous"].state_sha256,
            actual_omega_rad_per_myr=omega,dt_s=candidate.dt_s,dt_myr=duration,
            original_point_count=len(candidate.original_point_states),full_continuum_arc_count=2,born_midpoint_atoms=0,
            foundation_patch_count=5,coordinate_count=len(candidate.network_step.coordinates_decimal),
            diagnostics=dict(candidate.network_step.diagnostics),
            complete_first_accepted_history_unchanged=True,complete_geometric_pair_count=42,
            complete_geometric_pair_classification=counts,hot_geometric_pair_classification=hot_counts,
            causal_hot_front_whole_clock_proved=True,
            original_contact_up_dip_and_finite_pose_3d_placement_admission_proved=False,
            preserved_development_mechanical_review_complete_physical_case_reauthenticated=True,
            all_positive_pair_contact_time_domains_proved=False,
            born_finite_pose_force_equilibrium_proved=False,
            native_admission_status="REFUSED",whole_native_interval_admitted=False,
            remaining_original_point_1_work_block_count=1,
            remaining_physical_obligations=[
                "Physical constitutive/history permission over every positive pair's complete contact time domain",
                "Full finite-pose normal force branch and solid placement before an accepted native restart"],
            scope="Actual source full-arc affine equilibrium, exhaustive geometry and causal HOT front; no accepted continuation")
    except Exception as error:
        body.update(refusal_type=type(error).__name__,refusal=str(error),traceback=traceback.format_exc())
        print("FRONT control refusal",type(error).__name__,str(error),flush=True)
        after,_=guard_sources(require_frozen=True)
        write(out/"sources_after.json",after)
        body.update(stage_elapsed_s=stages,elapsed_s=time.monotonic()-started)
        write(out/"report.json",body,"report_sha256")
        raise
    after,_=guard_sources(require_frozen=True)
    if before!=after:
        raise ValueError("A frozen source changed during the complete genuine control")
    write(out/"sources_after.json",after)
    artifacts={str(path.relative_to(out)):_reference(path) for path in sorted(out.rglob("*")) if path.is_file()}
    body.update(stage_elapsed_s=stages,elapsed_s=time.monotonic()-started,
        all_751_inherited_sources_preserved=True,frozen_source_before_after_unchanged=True,artifacts=artifacts)
    return write(out/"report.json",body,"report_sha256")
