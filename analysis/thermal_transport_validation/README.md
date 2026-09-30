# Canonical mantle transport comparison

From the repository root, regenerate the CSVs, metrics and figure with:

```console
python analysis/genesis_mantle_transport_validation.py --include-orbit
```

`GenesisParameters()` supplies all physical parameters unchanged. The scoped
legacy comparison restores only the previous logarithmic blend between
`solid_transfer_w_m2_k=0.0005` and the magma coefficient. Both histories use the
same enthalpy, latent heat, surface/water reservoir, OLR and radiogenic inventory.
No parameters were fitted to an endpoint temperature.

Files:

- `histories.csv`: old/new values at 0, 1, 10, 50, 100, 200, 400, 640, 1000 and
  4500 Myr; 1 and 0.5 Myr maximum thermal steps; an independently segmented
  external schedule; and enabled canonical isolated orbital damping.
- `early_history.csv`: additional zero-tide samples at 0, 0.001, 0.01, 0.1, 0.5
  and 1 Myr to inspect hot-window cooling.
- `events.csv`: resolved event diagnostics from the long histories.
- `metrics.json`: exact parameters and source hashes, event times, conservation
  and convergence checks, the same-state exchange comparison and OLR probes.
- `old_vs_new.png`: temperature, melt fraction, exchange/heating and net budget.
- `tests.json`: final regression command, environment and result: 1970 passed.

Late thermal integration uses maximum steps of 1 or 0.5 Myr. Before 1 Myr the
corresponding bounds are 0.01 or 0.005 Myr. The external segmentation comparison
bisects every requested output interval while preserving the thermal step bound.
The optional orbit case uses the same `advance_orbit_thermal` coupling as Starter
with forcing intervals bounded by 0.01 Myr before 1 Myr and 1 Myr thereafter.
Its tides are the canonical enabled synchronous, isolated eccentricity-damping
model; there is no sustained eccentricity pumping. This diagnostic does not run
passive columns, shell mechanics or plate dynamics.

At Tm=1560 K and Ts=282 K, the bare old solid coefficient gives 0.639 W/m².
The complete old law gives 0.71247 W/m² because the state is still inside the
inherited partial-melt transition. The shared solid branch gives 0.011721 W/m²,
and the complete new blend gives 0.013543 W/m². At age 10 Myr, radiogenic heating
is 0.028820 W/m², so this same state cools under the old law and heats under the
new law. The sign follows the energy balance without a temperature clamp.

The new canonical trajectory remains partially molten: its melt fraction is
0.3345 at 10 Myr and 0.2507 at 4500 Myr. Thus this long history is **not a
validation of a fully solid mantle**: it continues to depend on the inherited
empirical melting interval and geometric conductance blend. The two mantle
temperatures at 4500 Myr, about 300 K old and 1550 K new, are outcomes, not
calibration targets. Near the new endpoint the exchange (0.01106 W/m²) still
exceeds radiogenic input (0.007880 W/m²), so the mantle continues cooling.

The unchanged hot-window law gives OLR of approximately 13896.6, 1733.6, 372.7
and 282.0 W/m² at 2300, 2000, 1800 and 1600 K, respectively. These are values of
the explicit parameterization, not validation of an atmosphere model. The
rapid early cooling follows this large hot-state radiative loss. The solid
transport change neither modifies nor calibrates the OLR law.

Legacy eta/Ra/Nu, solid convection and depth/Nu columns are prefixed
`counterfactual_`: those values were never used in the old energy balance.
The new depth/Nu is a thermal boundary-layer scale, not chemical crust or the
passive column's mechanical lid. No mobile/stagnant-lid classification is inferred.
