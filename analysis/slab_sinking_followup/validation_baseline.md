# Archived mechanics 0.4: timing and grid baseline

This is a read-only analysis of the three existing, independently evolved 50 Myr arms from the same Starter. No checkpoint or archived result was changed. `validation_baseline.py` regenerates `validation_baseline.json` and `validation_summary.json`, including source trace/report hashes. Times below are elapsed since the shared origin, 0.8781219482421875 Myr; force values and post-transport history are aligned at the same endpoint.

| Quantity | sub4, dt=1 Myr | sub4, dt=0.5 Myr | sub5, dt=1 Myr |
|---|---:|---:|---:|
| Mean speed at 50 Myr, mm/yr | 4.48305 | 3.88669 | 3.08441 |
| First accepted material, Myr | 16 | 15.5 | 11 |
| First surviving slab force, Myr | 17 | 16 | 12 |
| First neck failure, Myr | 17 | 16.5 | 21 |
| Accepted parcels / accepting timesteps | 129 / 25 | 180 / 59 | 461 / 37 |
| Cumulative accepted ocean volume, million km³ | 8.84162 | 12.16467 | 7.95844 |
| Mechanical detachments | 54 | 89 | 134 |
| Median parcel area, km² | 65,312 | 65,053 | 16,325 |
| Median parcel length proxy, km | 176.8 | 170.8 | 88.3 |
| Median initial cold thickness, km | 32.945 | 32.906 | 32.930 |
| Median potential cold fraction one step after acceptance | 0.61519 | 0.72757 | 0.61502 |
| Median surviving-section length / bend length | 1.40168 | 1.38032 | 0.73353 |
| Fraction of surviving sections shorter than the bend | 0% | 0% | 64.56% |
| Median surviving live neck strength, MPa | 4.337 | 4.333 | 4.298 |
| Median failed neck capacity, N | 5.582e16 | 5.474e16 | 2.754e16 |

The parcel length proxy uses the final/preserved segment width. The actual force-section length ratios are measured directly in saved traces. Section statistics pool timestep observations; failed sections are absent from these surviving-section distributions. Potential thermal weights do not assert that every accepted parcel survived until its first possible force evaluation.

## Findings supported by the saved data

1. **Acceptance and force geometry cause the first divergence before failure.** The first >1% dt discrepancy is at elapsed 16 Myr: 6.88659 versus 10.61039 mm/yr, 14 versus zero force sections, and zero neck failures in either arm. The first grid discrepancy is at 12 Myr: 7.88820 versus 10.61037 mm/yr, 16 versus zero sections, again without failures. At 13 Myr, sub5 is already at 2.91303 mm/yr with 58 sections; its first failure is only at 21 Myr. Before either arm has slab force, maximum relative speed discrepancies are 1.26e-6 for dt refinement and 0.002513 for grid refinement.

2. **A cell-sized acceptance jump changes the regime of the prescribed bend.** The finer grid quarters accepted parcel area and approximately halves its length and trench width, while cold thickness remains almost identical. Coarse sections have already completed the bend; most fine sections lie inside it. Their median tip angle falls from 37.57° to 32.53°. Thus this is not merely a more accurate quadrature of the same attached slab: the evolving attached geometry itself changes. There are no observed sections with L/bend_length below 0.1, so the saved data do not demonstrate a floating-point instability at vanishing length. They demonstrate unresolved geometry at first acceptance.

3. **Failure is strongly synchronized with newly accepted material.** Every timestep containing failure immediately follows an accepting timestep: 15/15, 39/39 and 16/16 respectively. There are no failure timesteps without previous-step acceptance in these arms. This is consistent with an acceptance burst introducing a new feed constraint/geometry, then triggering the tensile test. It does not independently identify which force term overloaded a particular neck.

4. **A discontinuous fallback thickness is not responsible here.** All 5,502 surviving section observations use `incoming_cold_mantle`; none use retained cubic-mean fallback. Typical live strength differs by less than 1% between arms. The failed capacity approximately halves with trench width on the finer grid. The initial divergence therefore cannot be attributed to a global change in tensile strength. Local source-face changes and failed-section geometry remain possible contributors to later bursts.

5. **The time split has a measurable thermal consequence.** Accepted H and density come from step-start material, but acceptance receives the step-end timestamp; the next dynamics evaluation is the first possible force evaluation. For these actual parcels, median cold fraction at that evaluation is about 0.615 at dt=1 versus 0.728 at dt=0.5. The heat solution itself was previously found equal to numerical precision between the arms. This combination is a discrete material/force coupling effect, not faster bulk cooling in one arm. It warrants a later substep or time-centred acceptance study; these three runs alone do not establish a correction formula.

6. **Old failure logs are insufficient to explain extreme late tensions.** Maximum recorded tension/capacity ratios are 2,751, 47,642 and 4,321. The old records contain total tension and capacity but omit gravity, bending, mantle drag and no-eduction reaction for the removed sections. Attributing the extremes to slab pull, bending, or reaction alone would be unsupported. Mechanics 0.5 must retain their decomposition before removal.

## Comparisons required after the 0.5 production freeze

The ordered thermal-cohort gravity correction can improve the location of buoyancy without resolving discrete acceptance, brittle instantaneous detachment or grid convergence. Report its actual effect even if speeds change little.

- Frozen **saved 0.4** baselines at elapsed 50, 100 and 400 Myr, retaining saved mechanics and inventory semantics.
- Fresh 0.5 from the original Starter at 50/100/200/400 Myr, plus dt=0.5 and sub5 arms to 50 Myr. Record full failure decomposition, gravity integrals, cohort ordering and proportional equal-age deep transfer.
- Saved populated 0.4 at 50→51 Myr to confirm compatibility; populated 0.5 resume parity after its first 50 Myr state exists.
- Check torque balance, nonnegative dissipation, zero-work constraints, finite neck capacity, material inventories, source hashes and all continuation checks. Compare endpoint snapshots and time-window statistics; a single post-detachment high speed is not a steady velocity.

`validation_case.py` defaults fresh imports to `young-mechanics-0.5`. The shared historical harness now explicitly defaults fresh imports to 0.4, applies physical overrides before constructing the initial inventory, and rejects changing a saved version on resume or a frozen probe.
