"""Read-only forensic comparison of the archived mechanics-0.4 sensitivity runs."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
RUNS = ROOT / "analysis/slab_sinking_validation/runs"
CASES = {"sub4_dt1": "validated_sinking_sub4_dt1", "sub4_dt0p5": "validated_sinking_sub4_dt0p5",
         "sub5_dt1": "validated_sinking_sub5_dt1"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stats(values):
    values = np.asarray(list(values), dtype=float)
    if not values.size:
        return dict(count=0)
    return dict(count=int(values.size), min=float(values.min()), median=float(np.median(values)),
        mean=float(values.mean()), p90=float(np.quantile(values,.9)), max=float(values.max()))


def rounded(value):
    return round(float(value), 8)


def case_data(label, folder):
    from tectonics.young_boundary import slab_thermal_deficit_fraction
    path = RUNS / folder / "elapsed_0050"
    report = read(path / "continuation.json")
    origin = report["import"]["origin_time_myr"]
    metadata = read(path / "mature_checkpoint/meta.json")
    inventory = metadata["subduction_memory"]["young_boundary_state"]
    dt = report["step_myr"]
    history = {rounded(row["time_myr"]-origin):row for row in report["history"]}
    trace = [json.loads(line) for line in path.with_suffix(".dynamics.jsonl").read_text(encoding="utf-8").splitlines()]
    parcels, acceptance = [], defaultdict(lambda:dict(count=0,area_km2=0.,oceanic_volume_km3=0.,cold_volume_km3=0.,initial_excess_mass_kg=0.))
    for segment in inventory["segments"]:
        for cohort in segment["thermal_cohorts"]:
            age = rounded(cohort["acceptance_time_myr"]-origin)
            event = acceptance[age]
            event["count"] += 1
            for target, source in (("area_km2","accepted_area_km2"),("oceanic_volume_km3","oceanic_volume_km3"),
                    ("cold_volume_km3","cold_mantle_volume_km3"),("initial_excess_mass_kg","initial_density_excess_mass_kg")):
                event[target] += cohort[source]
            h = cohort["initial_thickness_km"]
            width = segment["trench_length_km"]
            parcels.append(dict(acceptance_elapsed_myr=age, attached_at50=segment["attached"],
                area_km2=cohort["accepted_area_km2"], oceanic_volume_km3=cohort["oceanic_volume_km3"],
                initial_thickness_km=h, initial_excess_mass_kg=cohort["initial_density_excess_mass_kg"],
                width_km=width, length_proxy_km=cohort["accepted_area_km2"]/width if width>0 else None,
                cold_fraction_one_step_after_acceptance=slab_thermal_deficit_fraction(dt,h,inventory["thermal_diffusivity_m2_s"]),
                cold_fraction_after50=slab_thermal_deficit_fraction(report["final_time_myr"]-cohort["acceptance_time_myr"],h,inventory["thermal_diffusivity_m2_s"])))
    # Final segment widths can follow later edges: parcel length_proxy is a
    # diagnostic using current/preserved width, not a recovered original cell.
    rows, flat_sections = [], []
    for row in trace:
        age = rounded(row["state_time_myr"]-origin)
        sections = row["trace"].get("slab_sections", [])
        failures = row["trace"].get("slab_neck_failures", [])
        sink = row["trace"].get("slab_sinking_diagnostics", {})
        previous_age = rounded(age-dt)
        acc = acceptance.get(age,{})
        row_data = dict(elapsed_myr=age, dt_myr=dt,
            mean_speed_mm_yr=row["final_omega_surface_speed_mm_yr"]["mean"],
            post_transport_mean_speed_mm_yr=history[age]["mean_surface_speed_km_myr"],
            transport_commits=history[age]["transport_commits"],
            new_transport_commits=history[age]["transport_commits"]-history.get(previous_age,{"transport_commits":0})["transport_commits"],
            accepted_now=acc, accepted_previous=acceptance.get(previous_age,{}),
            neck_failure_count=len(failures), failure_ratios=[f["tension_n"]/max(f["capacity_n"],1.) for f in failures],
            section_count=len(sections), active_constraints=sink.get("no_eduction_active_count",0),
            basal_power_w=row["metrics"]["basal_source_power_w"],
            gravity_power_w=row["metrics"].get("slab_power_w",0.),
            bending_power_w=row["metrics"].get("slab_bending_dissipation_w",0.),
            mantle_power_w=row["metrics"].get("slab_mantle_dissipation_w",0.),
            attached_volume_at_trace_km3=row["accepted_slab_inventory"]["attached_oceanic_volume_km3"],
            initial_excess_mass_at_trace_kg=row["accepted_slab_inventory"]["initial_negative_buoyancy_mass_kg"],
            current_excess_mass_at_trace_kg=row["accepted_slab_inventory"]["current_negative_buoyancy_mass_kg"])
        for section in sections:
            reduced = {key:section[key] for key in ("contact_key","subducting_plate","overriding_plate", "accepted_area_km2", "slab_length_km", "trench_length_km",
                "cold_hinge_thickness_km","bend_length_km","tip_dip_deg","effective_sine","thickness_origin", "thermal_excess_mass_kg","gravitational_feed_force_n", "bending_coefficient_n_s_m", "neck_strength_pa", "neck_capacity_n", "neck_tension_n", "feed_m_s")}
            reduced.update(elapsed_myr=age, length_over_bend_length=section["slab_length_km"]/section["bend_length_km"],
                capacity_ratio=section["neck_tension_n"]/max(section["neck_capacity_n"],1.))
            flat_sections.append(reduced)
        row_data["section_length_over_bend_length"] = stats(s["slab_length_km"]/s["bend_length_km"] for s in sections)
        row_data["section_strength_pa"] = stats(s["neck_strength_pa"] for s in sections)
        row_data["surviving_capacity_ratio"] = stats(s["neck_tension_n"]/max(s["neck_capacity_n"],1.) for s in sections)
        rows.append(row_data)
    failures = inventory["mechanical_detachments"]
    timeline = [dict(elapsed_myr=rounded(f["time_myr"]-origin), **{k:v for k,v in f.items() if k!="time_myr"},
        tension_capacity_ratio=f["tension_n"]/max(f["capacity_n"],1.)) for f in failures]
    first_failure = next((row for row in rows if row["neck_failure_count"]),None)
    first_sections = next((row for row in rows if row["section_count"]),None)
    summary = dict(path=str(path), origin_time_myr=origin, dt_myr=dt,
        subdivisions=4 if label.startswith("sub4") else 5,
        mean_speed50_mm_yr=report["final_mean_surface_speed_km_myr"],
        max_speed50_mm_yr=report["final_max_surface_speed_km_myr"], transport_commits=report["transport_commits"],
        fracture_times=[e["time_myr"]-origin for e in report["young_fracture_events"]],
        first_acceptance_elapsed_myr=min(acceptance,default=None),
        first_force_elapsed_myr=None if first_sections is None else first_sections["elapsed_myr"],
        first_neck_failure=first_failure,
        accepted_event_times=len(acceptance), parcels=len(parcels), mechanical_detachments=len(failures),
        cumulative_accepted_oceanic_volume_km3=inventory["cumulative_accepted_oceanic_volume_km3"],
        parcel_volume_sum_km3=sum(p["oceanic_volume_km3"] for p in parcels),
        acceptance_burst_volume_km3=stats(v["oceanic_volume_km3"] for v in acceptance.values()),
        parcel_statistics={key:stats(p[key] for p in parcels if p[key] is not None) for key in
            ("area_km2","oceanic_volume_km3","initial_thickness_km","length_proxy_km","cold_fraction_one_step_after_acceptance","cold_fraction_after50")},
        surviving_section_statistics={key:stats(s[key] for s in flat_sections) for key in
            ("slab_length_km","trench_length_km","cold_hinge_thickness_km","length_over_bend_length","tip_dip_deg","neck_strength_pa","capacity_ratio")},
        retained_thickness_fallback_count=sum(s["thickness_origin"]!="incoming_cold_mantle" for s in flat_sections),
        short_section_fraction=sum(s["length_over_bend_length"]<1 for s in flat_sections)/max(len(flat_sections),1),
        very_short_section_fraction=sum(s["length_over_bend_length"]<.1 for s in flat_sections)/max(len(flat_sections),1),
        failure_ratio_statistics=stats(f["tension_capacity_ratio"] for f in timeline),
        failure_steps_with_previous_acceptance=sum(bool(r["accepted_previous"]) for r in rows if r["neck_failure_count"]),
        failure_steps_without_previous_acceptance=sum(not r["accepted_previous"] for r in rows if r["neck_failure_count"]),
        failed_capacity_n=stats(f["capacity_n"] for f in timeline),
        failed_tension_n=stats(f["tension_n"] for f in timeline),
        trace_sha256=hashlib.sha256(path.with_suffix(".dynamics.jsonl").read_bytes()).hexdigest(),
        report_sha256=hashlib.sha256((path/"continuation.json").read_bytes()).hexdigest())
    return dict(summary=summary,rows=rows,failures=timeline,parcels=parcels,sections=flat_sections)


def main():
    cases={label:case_data(label,folder) for label,folder in CASES.items()}
    same_times=[]
    base={row["elapsed_myr"]:row for row in cases["sub4_dt1"]["rows"]}
    for label in ("sub4_dt0p5","sub5_dt1"):
        aligned=[]
        for row in cases[label]["rows"]:
            if row["elapsed_myr"] not in base:
                continue
            reference=base[row["elapsed_myr"]]
            aligned.append(dict(elapsed_myr=row["elapsed_myr"],
                reference_speed_mm_yr=reference["mean_speed_mm_yr"],case_speed_mm_yr=row["mean_speed_mm_yr"],
                relative_speed_difference=row["mean_speed_mm_yr"]/max(reference["mean_speed_mm_yr"],1e-30)-1.,
                reference_failures=reference["neck_failure_count"],case_failures=row["neck_failure_count"],
                reference_commits=reference["transport_commits"],case_commits=row["transport_commits"],
                reference_sections=reference["section_count"],case_sections=row["section_count"]))
        onset = min(cases[label]["summary"]["first_force_elapsed_myr"],cases["sub4_dt1"]["summary"]["first_force_elapsed_myr"])
        same_times.append(dict(case=label, first_over_one_percent=next((r for r in aligned if abs(r["relative_speed_difference"])>.01),None),
            maximum_absolute_relative_speed_difference_before_either_slab_force=max(abs(r["relative_speed_difference"]) for r in aligned if r["elapsed_myr"]<onset), aligned=aligned))
    result=dict(scope="Read-only forensic comparison of existing0.4 runs; no new simulations or production edits.",
        cases=cases,aligned_comparison=same_times,
        cautions=["Saved force traces include only surviving sections, so their strength/shape distribution excludes failed necks.",
            "The per-cohort length proxy uses final/preserved segment width; actual force-section lengths in traces are authoritative.",
            "Initial accepted properties are sampled by raster_acceptance_events from step-start material, then assigned the step-end acceptance timestamp.",
            "Contact cohorts can first exert force on the following force evaluation. The one-step thermal fraction is a potential weight, not proof every cohort remained attached and exerted force.",
            "The inventory diagnostics at trace evaluation include detached cohorts; their total cold mass is not the force-active cold mass."])
    HERE.mkdir(parents=True,exist_ok=True)
    (HERE/"validation_baseline.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    compact={label:case["summary"] for label,case in cases.items()}
    (HERE/"validation_summary.json").write_text(json.dumps(dict(cases=compact,
        first_difference={row["case"]:row["first_over_one_percent"] for row in same_times},
        maximum_pre_slab_relative_difference={row["case"]:row["maximum_absolute_relative_speed_difference_before_either_slab_force"] for row in same_times}),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({label:{key:value["summary"][key] for key in ("mean_speed50_mm_yr","first_acceptance_elapsed_myr","first_force_elapsed_myr","mechanical_detachments","short_section_fraction")} for label,value in cases.items()},indent=2))


if __name__=="__main__":
    main()
