# One BE50 cut-cell mechanical/thermal feedback trial

Base: `435725080eb762868726024de703059fc7adb0e1`. Restore CLOSED.
This is an isolated 50-year interval from the accepted 0.8684 Myr mechanical
state and the already saved implicit conduction endpoint. No new conduction
solve, mesh refinement, canonical/default advance or publication is performed.

## Conservative scheme and limits

Hold the previously computed conduction endpoint fixed. For each bounded
corrector, prepare the existing 800 material temperatures, solve BE predictor,
evaluate the inherited damage law, stop at its gate before activation, then
solve final BE from the **same original memory** with the predicted damage.
Compute, for each retained cohort, with the actual radial moments and stiffness:

- `Q_i = (dt/tau_i) integral(e1:C:e1) dV >= 0`, in joules: Maxwell loss.
- `W_i = -integral(e1:C:alpha*DeltaT) dV`, signed joules: thermal work **into
  mechanics**. Its thermal partner is `-W_i`, not `+W_i`.
- Set `E_i = E_conduction_i + Q_i - W_i` in its unique material cut-cell.
  Mechanical BE already includes `-Q_i + W_i`; never apply another debit.

Invert the same sensible-plus-latent enthalpy law to obtain actual new
reservoir temperatures. Correctors always start from the original conduction
energy; they do not accumulate repeated heat. Maximum 6 correctors, with both
maximum temperature mismatch <=1e-8 K (existing loading tolerance) and local
exchange changes <=1000 J + 3e-8*abs(exchange) (existing quadrature audit budget).
A stop on nonconvergence, negative energy or mixed old material retains the
failure evidence and does not admit a state. Predictor journals are never added
to final energy. This is one time interval, not a series of physical steps.

No numerical, tidal or extra latent heat is added. Mechanical numerical
kinetic/memory loss has its own sink. Drag loss enters a distinct interface
energy store, and `-DeltaU_damage` a separate damage energy store. These are
explicit energy accounting reservoirs with **unknown thermal feedback**, not
a chosen heat split or a closed physical model. A positive damage energy change
requiring an unspecified donor is rejected.

External prescribed traction work remains an external input with no dynamical
mantle donor. Orbital parameters are frozen; a timestamped endpoint boundary
snapshot allows the existing damage law to evaluate its inherited cyclic
strain driver. There is no orbital evolution or tidal heat entry. Surface
water remains on its hot-vapor branch; ocean fraction is zero.

Conduction is not recomputed after the feedback. Its splitting error and the
initial cut-cell closure sensitivity remain open. The geometric reference
chart and its material mass intervals are fixed, while mechanical displacement
and elastic memory evolve. Unowned solid material is not assigned new mechanical
IDs in this control. The original four spatial controls remain open.

## Independent audit and restart

The saved predictor and final states use the existing producer-independent
spatial quadrature audit for BE memory, source elasticity/relaxation, work,
impulse and energy. An additional audit independently integrates **each**
cohort's Maxwell and thermal work, verifies recipient identity/mass, the actual
cell energy increments, phase inversion and the combined budget including
interface/damage storage and the numerical sink. Geometry is checked against
the interval certificate. The inherited damage law is separately recomputed.

Restart seals and dependency hashes authenticate the custom isolated packet;
all 800 memories are decoded as typed `VolumeCohortMemory` objects and reencoded
with byte-identical JSON. The feedback audit is repeated on that decoded packet.
No second mechanical interval is run. This is not a production checkpoint or
permission to continue the canonical history.

`run.py` is the bounded producer (300 seconds / 4 GiB). Do not rerun it just to
inspect the saved result. `audit.py` reads saved states without a mechanical solve.
The first attempt (`/workspace/genesis-cutcell-mechanical-01`) stopped before
final BE because the frozen orbital boundary had its initial timestamp; that
failure is retained. The corrected attempt represents the same single interval.

## Confirmed result

**PASS** in `/workspace/genesis-cutcell-mechanical-02/completion.json`.
Four correctors (eight predictor/final solves of the same interval) converged;
last temperature mismatch is 0 K in stored arithmetic and exchange-change /
allowance is 0.0005587. No history is advanced per corrector. The corrected
producer used 91.12 seconds / 496924 KiB peak RSS. Its final in-memory equality
check compared tuples to JSON lists after the numerical audits passed; the
saved candidate was then completed through read-only `finish.py`, without
another solve. Original failure records remain intact.

- **800 existing IDs/masses retained, zero births or retirements**, all old
  material remains below solidus. Memory evolves by the BE law from the
  authenticated parent's original strain coefficients; it is not reset.
- Located Maxwell heat: **9.6395956211e17 J**.
- Signed thermal work into mechanics: **−3.3498198491e16 J**; its thermal
  partner is positive. Net thermal gain **9.9745776096e17 J**.
- Maximum feedback temperature change **4.4282001909e-5 K**.
- Interface drag store **9.8461846625e15 J**; damage store
  **8.2086256405e15 J**; numerical sink **8.2071453880e15 J**. None heats rock.
- Independent mechanical momentum residual **3.82e-15** (relative), balance
  **−11648 J / 3.38e-16 relative**, memory error **1.97e-15**.
- Independent local quadrature covers all 800 cohorts. Maximum cellwise heat
  increment roundoff is **6.653e7 J**, checked against that cell's endpoint
  ulps. The combined audit uses only *touched* thermal cell ulps plus the
  existing mechanical audit budget; unchanged deep/water energies supply no
  tolerance. Exact budget and residual are recorded in `summary.json`.
- Damage law independently reproduced; maximum damage **0.0401571**, gate
  threshold **0.65**. No active or persistent candidate; geometry certificate
  independently verified over the interval.
- Typed memory restart and full packet bytes agree. Candidate SHA256:
  `5a0cfcd3edfd57f4f05b59e8b408b09c25ac2a704068609e22029ab865230f2e`.
- **66 tests PASS**, including analytical Maxwell/thermal work, signed transfer,
  nonaccumulating correctors, invalid recipients/negative heat, enthalpy and
  existing mechanical regressions.

The defined thermal→mechanical→thermal path works. Next blocker is **physical
closure**, not transfer arithmetic: specify drag/damage thermal destinations
and the external traction donor, then assess conduction/feedback splitting
and initial-profile sensitivity before any longer coupled evolution. This
control does not close those questions or any original physical fracture stage.
