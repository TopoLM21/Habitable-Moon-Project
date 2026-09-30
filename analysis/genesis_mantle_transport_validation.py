"""Reproduce old/new canonical thermal histories without fitting parameters.

Run from the repository root, e.g.:
    python analysis/genesis_mantle_transport_validation.py --include-orbit

The legacy counterfactual patches only the mantle exchange law in a scoped
context. Enthalpy, OLR, latent heat, radiogenic power and solver stay identical.
The optional orbit run uses the same orbit/global-thermal coupling as Starter,
but does not solve a passive mechanical column or any plate mechanics.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tectonics import genesis
from tectonics.genesis_onset import advance_orbit_thermal
from tectonics.genesis_tides import (
    SYNCHRONOUS_SPIN, TidalParameters, initial_tidal_orbit,
    tidal_heat_flux_w_m2, validate_tidal_orbit,
)

TIMES_MYR = (0.0, 1.0, 10.0, 50.0, 100.0, 200.0, 400.0, 640.0, 1000.0, 4500.0)
EARLY_TIMES_MYR = (0.0, .001, .01, .1, .5, 1.0)
CURRENT_TRANSPORT = genesis.mantle_transport
SOLID_FIELDS = (
    "viscosity_pa_s", "rayleigh_number", "nusselt_number",
    "conductive_heat_flux_w_m2", "solid_convective_heat_flux_w_m2",
    "thermal_boundary_layer_thickness_km",
)


def legacy_transport(tm, ts, p):
    """Exact pre-change log blend; shared-law diagnostics are counterfactual."""
    shared = CURRENT_TRANSPORT(tm, ts, p)
    phi = genesis.melt_fraction(tm, p)
    transition = min(1.0, max(0.0, (phi-p.rheology_transition_low_melt) /
                            (p.rheology_transition_high_melt-p.rheology_transition_low_melt)))
    weight = transition**2 * (3-2*transition)
    transfer = math.exp((1-weight)*math.log(p.solid_transfer_w_m2_k)
                        + weight*math.log(p.magma_transfer_w_m2_k))
    return {
        "mantle_to_surface_flux_w_m2": transfer*(tm-ts),
        "effective_heat_transfer_w_m2_k": transfer,
        "magma_transport_weight": weight,
        "legacy_solid_flux_w_m2": p.solid_transfer_w_m2_k*(tm-ts),
        "legacy_equivalent_resistance_length_km": p.thermal_conductivity_w_m_k / p.solid_transfer_w_m2_k / 1000,
        **{f"counterfactual_{key}": shared[key] for key in SOLID_FIELDS},
    }


def enriched_row(row, state, name, old, orbit=None, tidal_energy_per_area=0.0):
    result = {"run": name, "transport_law": "legacy_constant" if old else "shared_convection",
              "solid_diagnostics_role": "counterfactual_only" if old else "active_solid_branch", **row}
    result.update(
        mantle_energy_j_m2=state.energy[0]*genesis.ENERGY_SCALE,
        surface_energy_j_m2=state.energy[1]*genesis.ENERGY_SCALE,
        cumulative_input_j_m2=state.energy[2]*genesis.ENERGY_SCALE,
        cumulative_olr_j_m2=state.energy[3]*genesis.ENERGY_SCALE,
        relative_water_mass_residual=(row["vapor_mass_kg"]+row["ocean_mass_kg"]-row["total_water_mass_kg"])/row["total_water_mass_kg"],
    )
    if orbit is not None:
        result.update(
            eccentricity=orbit.eccentricity, semimajor_axis_km=orbit.semimajor_axis_km,
            orbit_dissipated_energy_j=orbit.dissipated_energy_j,
            cumulative_tidal_input_j_m2=tidal_energy_per_area,
        )
    return result


def run_history(p, *, old=False, max_step=1.0, segmented=False, orbital=False,
                sample_times=TIMES_MYR, label_suffix=""):
    name = f"{'old' if old else 'new'}_{'orbit' if orbital else 'zero_tide'}_step{max_step:g}"
    if segmented:
        name += "_segmented"
    name += label_suffix
    context = patch.object(genesis, "mantle_transport", legacy_transport) if old else nullcontext()
    started = perf_counter()
    tides = TidalParameters(enabled=True, spin_state=SYNCHRONOUS_SPIN) if orbital else None
    orbit = initial_tidal_orbit(tides) if orbital else None
    cumulative_tidal = 0.0
    with context:
        state = genesis.initial_state(p)
        initial_p = replace(p, tidal_heat_flux_w_m2=tidal_heat_flux_w_m2(tides)) if orbital else p
        rows = [enriched_row(genesis.diagnose(state, initial_p), state, name, old, orbit)]
        events = []
        all_residuals = [rows[0]["relative_energy_residual"]]
        for target in sample_times[1:]:
            outer_start = state.time_myr
            # The external segmentation comparison changes call boundaries
            # only, leaving the internal adaptive maximum step unchanged.
            endpoints = [(outer_start+target)/2, target] if segmented else [target]
            for endpoint in endpoints:
                while state.time_myr < endpoint-1e-12 and not state.stopped_reason:
                    previous_time = state.time_myr
                    # Resolve early cooling with 0.01 Myr; late max_step is
                    # 1.0 or 0.5 Myr. Orbital forcing is averaged over <=1 Myr,
                    # with <=0.01 Myr in the initial hot phase.
                    early = previous_time < 1.0-1e-12
                    step = min(max_step, .01*max_step) if early else max_step
                    stop = min(endpoint, 1.0) if early else endpoint
                    if orbital:
                        stop = min(stop, previous_time+(.01 if early else 1.0))
                        state, orbit, heating, accepted = advance_orbit_thermal(
                            state, orbit, p, tides, stop, max_step_myr=step)
                        cumulative_tidal += heating*(state.time_myr-previous_time)*genesis.SECONDS_PER_MYR
                    else:
                        state, accepted = genesis.advance(state, p, stop, max_step_myr=step)
                    all_residuals.extend(row["relative_energy_residual"] for row in accepted)
                    events.extend({"run": name, **row} for row in accepted[:-1])
                if state.stopped_reason:
                    break
            rows.append(enriched_row(accepted[-1], state, name, old, orbit, cumulative_tidal))
            if state.stopped_reason:
                break
        orbit_error = None
        if orbit is not None:
            validate_tidal_orbit(orbit, tides)
            orbit_error = ((cumulative_tidal*p.area_m2-orbit.dissipated_energy_j)
                           / max(orbit.dissipated_energy_j, 1.0))
        summary = {
            "run": name, "max_thermal_step_myr": max_step,
            "early_max_thermal_step_myr": .01*max_step,
            "external_segmentation": segmented, "orbital": orbital,
            "events_myr": state.events, "stopped_reason": state.stopped_reason,
            "final_time_myr": state.time_myr,
            "max_abs_relative_energy_residual": max(map(abs, all_residuals)),
            "max_abs_relative_water_mass_residual": max(abs(row["relative_water_mass_residual"]) for row in rows),
            "orbit_heat_transfer_relative_residual": orbit_error,
            "wall_seconds": perf_counter()-started,
        }
    print(json.dumps(summary), flush=True)
    return rows, events, summary


def compare(reference, candidate, event_a, event_b):
    a = {row["time_myr"]: row for row in reference}
    b = {row["time_myr"]: row for row in candidate}
    shared_times = sorted(set(a) & set(b))
    fields = ("mantle_temperature_k", "surface_temperature_k", "mantle_melt_fraction",
              "mantle_to_surface_flux_w_m2", "net_mantle_flux_w_m2")
    return {
        "reference": reference[0]["run"], "candidate": candidate[0]["run"],
        "sample_count": len(shared_times),
        "max_absolute_differences": {key: max(abs(a[t][key]-b[t][key]) for t in shared_times) for key in fields},
        "event_time_absolute_differences_myr": {key: abs(event_a[key]-event_b[key]) for key in set(event_a) & set(event_b)},
    }


def write_csv(path, rows):
    names = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def plot(histories, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for label, data, color in (("Old constant solid conductance", histories["old_zero_tide_step1"], "#b45309"),
                               ("Shared mantle convection", histories["new_zero_tide_step1"], "#0369a1")):
        data = [row for row in data if row["time_myr"] > 0]
        times = [row["time_myr"] for row in data]
        for ax, key in zip(axes.flat, ("mantle_temperature_k", "mantle_melt_fraction", "mantle_to_surface_flux_w_m2", "net_mantle_flux_w_m2")):
            ax.plot(times, [row[key] for row in data], "o-", color=color, label=label, markersize=4)
            ax.set_xscale("log")
            ax.set_xlabel("Time since molten initial state (Myr)")
            ax.grid(alpha=.2)
    axes[0, 0].set_ylabel("Mantle temperature (K)")
    axes[0, 1].set_ylabel("Mantle melt fraction")
    axes[1, 0].set_ylabel("Mantle-to-surface exchange (W/m²)")
    axes[1, 0].set_yscale("log")
    reference = [row for row in histories["new_zero_tide_step1"] if row["time_myr"] > 0]
    axes[1, 0].plot([row["time_myr"] for row in reference],
                    [row["radiogenic_flux_w_m2"] for row in reference],
                    "--", color="#475569", label="Radiogenic heating")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].set_ylabel("Net mantle heating (W/m²)")
    axes[1, 1].set_yscale("symlog", linthresh=.001)
    axes[1, 1].axhline(0, color="black", linewidth=.7)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Canonical Genesis: only the mantle exchange law changes\nNo parameter fitting; zero prescribed tides")
    fig.savefig(output, dpi=140)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"analysis/thermal_transport_validation")
    parser.add_argument("--include-orbit", action="store_true")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    p = genesis.GenesisParameters()
    histories, summaries, all_rows, all_events = {}, {}, [], []
    for old in (True, False):
        for maximum, segmented in ((1.0, False), (.5, False), (1.0, True)):
            rows, events, summary = run_history(p, old=old, max_step=maximum, segmented=segmented)
            histories[summary["run"]], summaries[summary["run"]] = rows, summary
            all_rows.extend(rows)
            all_events.extend(events)
        if args.include_orbit:
            rows, events, summary = run_history(p, old=old, orbital=True)
            histories[summary["run"]], summaries[summary["run"]] = rows, summary
            all_rows.extend(rows)
            all_events.extend(events)
    comparisons = []
    for law in ("old", "new"):
        baseline = f"{law}_zero_tide_step1"
        for alternative in (f"{law}_zero_tide_step0.5", f"{law}_zero_tide_step1_segmented"):
            comparisons.append(compare(histories[baseline], histories[alternative],
                summaries[baseline]["events_myr"], summaries[alternative]["events_myr"]))
    early_rows, early_summaries = [], []
    for old in (True, False):
        rows, _, summary = run_history(p, old=old, sample_times=EARLY_TIMES_MYR, label_suffix="_early")
        early_rows.extend(rows)
        early_summaries.append(summary)
    probe = {"mantle_temperature_k": 1560.0, "surface_temperature_k": 282.0,
             "old": legacy_transport(1560.0, 282.0, p), "new": CURRENT_TRANSPORT(1560.0, 282.0, p)}
    probe["budget_at_system_age_10_myr"] = {
        "radiogenic_flux_w_m2": p.silicate_column_kg_m2*p.radiogenic_specific_power_w_kg*2**(-10/p.radiogenic_half_life_myr),
        "tidal_flux_w_m2": 0.0,
    }
    for law in ("old", "new"):
        probe["budget_at_system_age_10_myr"][f"{law}_net_mantle_flux_w_m2"] = (
            probe["budget_at_system_age_10_myr"]["radiogenic_flux_w_m2"]-probe[law]["mantle_to_surface_flux_w_m2"])
    hot_window = [{"surface_temperature_k": t,
                   "outgoing_longwave_w_m2": genesis.outgoing_longwave_w_m2(t, p),
                   "absorbed_stellar_flux_w_m2": p.absorbed_stellar_flux_w_m2,
                   "hot_window_w_m2": genesis.SIGMA*max(t-p.hot_window_temperature_k, 0)**4}
                  for t in (2300.0, 2000.0, 1800.0, 1600.0)]
    checks = {
        "all_histories_reach_4500_myr": all(row["final_time_myr"] == 4500 for row in summaries.values()),
        "energy_ledger_relative_error_below_1e-7": all(row["max_abs_relative_energy_residual"] < 1e-7 for row in summaries.values()),
        "water_mass_relative_error_below_1e-14": all(row["max_abs_relative_water_mass_residual"] < 1e-14 for row in summaries.values()),
        "step_and_segmentation_temperature_differences_below_0_01_k": all(max(row["max_absolute_differences"][key] for key in ("mantle_temperature_k", "surface_temperature_k")) < .01 for row in comparisons),
        "orbit_heat_transfer_relative_error_below_1e-12": all(row["orbit_heat_transfer_relative_residual"] is None or abs(row["orbit_heat_transfer_relative_residual"]) < 1e-12 for row in summaries.values()),
    }
    sources = ("tectonics/genesis.py", "tectonics/mantle_convection.py", "tectonics/genesis_onset.py",
               "tectonics/genesis_tides.py", "analysis/genesis_mantle_transport_validation.py")
    report = {
        "parameters": asdict(p), "parameter_hash": genesis.parameter_hash(p),
        "tidal_parameters": asdict(TidalParameters(enabled=True, spin_state=SYNCHRONOUS_SPIN)) if args.include_orbit else None,
        "requested_times_myr": TIMES_MYR, "runs": summaries, "convergence": comparisons,
        "early_times_myr": EARLY_TIMES_MYR, "early_runs": early_summaries,
        "same_state_probe": probe, "hot_window_probe": hot_window, "checks": checks,
        "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
        "limitations": [
            "Legacy histories replace only mantle exchange with the exact previous log blend; all other current Genesis laws are shared.",
            "Legacy eta/Ra/Nu/solid-convection fields are explicitly counterfactual diagnostics, not terms in the old energy balance.",
            "The optional orbit path is the Starter global heat/orbit solver with isolated eccentricity damping; it excludes passive columns and mechanics.",
            "Partially molten trajectories remain governed by the inherited empirical melt interval and geometric conductance blend, not a resolved melt-flow model.",
            "The hot-window OLR is unchanged and is an approximate atmosphere parameterization, not a radiative-transfer calculation.",
            "No stagnant/mobile-lid regime classification or tectonic onset is inferred from the heat-transport scaling.",
        ],
    }
    write_csv(args.output/"histories.csv", all_rows)
    write_csv(args.output/"early_history.csv", early_rows)
    write_csv(args.output/"events.csv", all_events)
    (args.output/"metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    plot(histories, args.output/"old_vs_new.png")
    print(json.dumps({"checks": checks, "output": str(args.output)}), flush=True)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
