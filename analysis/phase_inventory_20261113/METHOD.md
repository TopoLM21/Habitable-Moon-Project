# Bounded finite phase supply, reconstruction 2026-11-13

Base: `ea58c142b8e5f38ed6158e1c8184ff23049bee48`, working branch
`reconstruction/regional-20261113`. This is a new experimental initialization
and new evidence, not a recovered result from the lost environment. This stage
ends with bounded tests and publication; fracture mechanics and long runs are
not part of it.

## What changes

`PhaseStrainExchange` reuses the existing shared spherical geometry, dilation
mechanics, finite compliant pressure store, geometric exchange quotas, material
ages and compensated work/heat ledgers. A separate material subclass replaces
the sensible-only thermal owner. Old checkpoints and old classes remain valid;
old loaders reject the new version. The application default is unchanged.

An explicit fraction chi of **existing** mantle mass is tagged as eligible
feedstock. The rest is refractory in this bounded model. Eligible material and
existing basalt are divided into equally weighted effective constituents with
specified melting temperatures. These are finite material stocks, not extra
mass, a reconstructed deep-mantle composition or a declaration that the entire
hot mantle is molten. Each constituent retains its initial-mantle/basalt/felsic
origin in the source and crust, including after extraction and return.

For each thermal box, at fixed geometric volumes and reference phase envelope:

```
H = cp M T + L sum_j m_liquid,j
S = cp M ln(T) + sum_j L m_liquid,j / T_melt,j
```

Below its own melting point a constituent is solid; above it, liquid. At the
melting point its latent inventory supplies a Stefan plateau. This is a model
of effective immiscible constituents, not chemical phase equilibrium. Initial
H is unchanged, so adding this constitutive description slightly lowers the
diagnosed temperature instead of adding latent energy. Both boxes use the
same caloric reference. Mechanical/hydraulic heat enters H once.

With total eligible mass F, liquid mass Ml and retained liquid fraction r, the
maximum extractable mass is `max(0,(Ml-r F)/(1-r))`. This keeps the remaining
liquid fraction at least r; successive calls do not reset the depleted stock.
Extraction carries `cp T + L` J/kg, using actual liquid component proportions.
Basalt return carries its current component mix and `cp T + L phi` J/kg.
Cold returned rock must acquire its latent enthalpy from the existing pool to
remelt. Provenance and energy use the same actual transferred amounts.

Positive geometric quotas are scaled proportionally if their combined demand
exceeds mobile mass. The previous passive pressure projection then decides
whether finite pressure work can pay emplacement. This is a coarse allocation
rule, not a conduit solution; pressure-blocked sites can leave stock unused.
Birth precedes return within this endpoint exchange. Same-step cold return is
not pre-counted as liquid supply. The old sensible-only mode retains its
previous return-then-birth ordering. Material remains well mixed within the
thermal/composition pools; local phase chemistry is not resolved.

## Thermal integration and validity

The old exact sensible-heat matrix cannot be reused with latent heat. The new
thermal owner uses monotone backward-Euler enthalpy exchange, a scalar bracketed
root, positive frozen conductances, and step doubling. The same signed basal
heat is removed from one box and added to the other. Root accuracy is scaled
to the exchanged heat, not the whole planet's enthalpy. Error control includes
temperature, integrated flux and liquid fraction. The method and endpoint
material splitting are first order. This is error control, not a proof that
every possible short-lived phase excursion is localized exactly.

Radiogenic and prescribed orbital inputs remain explicit; surface loss is
explicit. The pressure store remains the mechanical source, never silently
drawn from total heat. Phase thresholds are fixed reference parameters; there
is no pressure-dependent EOS or pressure-driven melting, and no thermal
contraction in the mechanical constitutive law. The mechanical model rejects
any liquid crust rather than advancing it as a welded solid. The existing
thermal-layer and mechanical geometry applicability guards also remain.

The explicit test values (1700–2200 K envelope, L=400 kJ/kg, chi=0, 1e-7 or
1e-5, r=0.02, 4/16/64 constituents) are **synthetic sensitivity controls**.
They are not measured lunar solidus, available melt fraction, water content,
redox, permeability or mantle chemistry. A broad fixed envelope is deliberate;
fractionation, thermochemical differentiation and buoyant melt segregation are
outside this stage. In particular the pressure reference volume is not an
extra fusible or thermal stock.

Real melting laws require pressure, composition and volatile constraints. The
primary reference [Katz et al. (2003)](https://doi.org/10.1029/2002GC000433)
describes a parameterized peridotite melting model; its full law is **not**
implemented here, and its parameters are not substituted for unknown lunar
properties. The present tests establish thermodynamic bookkeeping of a stated
coarse approximation, not petrological validation.

## Completion criteria

Independent Stefan and implicit two-box limits; no mass/enthalpy addition at
initialization; selective depletion with full donor enthalpy; positive entropy
in isolated heat exchange; no supply merely from hot temperature or pressure;
phase-component/provenance conservation; complete checkpoint/restart equality;
short linked grid/time/phase-resolution comparisons and preserved regressions.
No new plates, spontaneous fractures or lunar-history realism are claimed.
