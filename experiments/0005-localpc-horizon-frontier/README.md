# 0005 — The unroll-horizon frontier: where the hypergradient operationally fails and Local-PC does not

**Status:** PRE-REGISTERED + runnable first cut. Registration fixed here
before results. Discharges the exp-0002 deferred claim ("the crash is an
extrapolation, not a result"). Idea:
[ideas/localpc-horizon-frontier.md](../../ideas/localpc-horizon-frontier.md).

## Question

Is there a **reachable** unroll horizon H\* where backprop-through-the-inner-
loop (global hypergradient) operationally fails — non-finite, OR peak memory
> budget, OR per-step wall-time > 50× small-H — while Local-PC (O(1)-in-H,
per-step credit) stays finite and within ε of its own small-H quality?

## Design

exp-0002 nested optimizer-memory (K-level cascade on θ∈R^d). Two pre-registered
failure axes: **(M) memory** — grid over (H, d, K) so global's O(H·K·d) graph
crosses a fixed RSS budget; **(D) divergence** — deep nonlinear inner
recurrence + long H so the H-Jacobian product → non-finite. Per seed/H record:
finite?, peak-RSS, per-step wall-time, final meta-loss, exact autograd node
count (O(H·K) vs O(1) anchor).

## Pre-registered predictions & decision rule

- **Guard (must pass first):** global actually fails within the feasible
  grid (≥80% seeds at some H\*). If not → operational payoff UNREACHED;
  report straight as "exp-0002 honest gap still open." No spin.
- **H-frontier:** at H\*, Local-PC finite AND final meta-loss ≤ 1.25× its
  own small-H meta-loss on ≥80% seeds.
- **Quality parity where both run:** at the largest H both survive, paired
  (CRN) Local-PC−global meta-loss is **NOT a loss** for Local-PC (verbatim
  exp-0002 three-way: must be NOT-separated or SEP-in-Local-PC's-favour;
  Local-PC worse ⇒ claim fails). Quality *superiority is not claimed*.
- **Falsified if:** Local-PC also fails by H\*, or is quality-worse where
  both run.

Paired CRN, divergence guard (non-finite or >1e3), exact sign-test p as
corroboration; n=10 then n=60 if the parity test is AMBIGUOUS.

## Escalation

1. `--smoke` pipeline + node-count sanity (global O(H), local O(1)).
2. `--memory` axis sweep (the OOM/wall-time frontier).
3. `--diverge` axis sweep (the non-finite frontier).
4. parity check at max-common-H; n=60 if AMBIGUOUS.

## Results (n=10, RESOLVED 2026-05-17) — held to the pre-registered standard

Structural law (uncontested, = what exp-0002 proved): both axes global graph
O(H) (488→30728 ; 448→56328), local-PC **flat 31, O(1)** every H.

```
DIVERGE (nonlinear, the STRONG channel)            global_fail  localpc_fail
 H=8/32/128                                          0/10         0/10
 H=512  (H*)                                        10/10         0/10
 parity@512: localpc−global Δq +0.0006 p=1.000 -> PARITY OK
 -> FRONTIER DEMONSTRATED (operational), BOTH gates pass.

MEMORY (d=256, the bytes-cheap channel)            global_fail  localpc_fail
 H=8/64/256                                          0/10         0/10
 H=1024 (H*)                                         10/10         0/10
 parity@1024: localpc consistently worse by 0.0002 -> PARITY GATE FAILS
 -> PARTIAL: no OOM (56k nodes ≪4GB; wall-time only), parity gate fails.
```

**Honest verdict, not the auto-line (which ignored its own parity gate —
fixed in run.py):**
- **Divergence axis = the win.** H*≈512, global fails 10/10, local-PC 0/10,
  quality parity holds (p=1.000). Local-PC's structural advantage becomes
  **operationally necessary at extreme depth, at parity quality.** Qualifier:
  the harness emits aggregate fail count only, **not the failure-mode
  breakdown**; parity computing at H=512 implies some global runs were
  finite-but-wall-time-failed → "operational (wall-time-inclusive) failure,"
  **not confirmed pure mathematical non-finite.** "Necessary" earned at the
  operational level; the strongest (NaN/OOM) form needs instrumentation —
  NOT claimed.
- **Memory axis = honest gap restated, as pre-flagged.** No OOM at d=256
  (bytes cheap — exactly exp-0002's disclosed gap); failure is wall-time
  only; AND the pre-registered parity gate FAILS (local-PC consistently
  worse by a negligible 0.0002 — the recurring significance≠effect-size
  artifact). Per pre-registration the memory-axis frontier does **not**
  cleanly pass.
- **H\* is at the grid edge** (one point: (128,512] / (256,1024]) — the
  frontier is real but coarsely located, not finely resolved.

**Net (answer to "does the structural advantage become NECESSARY at extreme
depth?"):** YES on the **divergence channel**, operationally, at parity
quality — the one honest home consistent with all prior evidence. The memory
channel does not bite at feasible scale (predicted) and trips the parity gate.
Capability claim only ("enables a regime the hypergradient can't survive"),
not a value claim.

**Immediate follow-ups (pre-named, before any upgrade):**
1. Instrument the failure mode (non-finite vs RSS vs wall-time) at H* — the
   only thing that can upgrade "operational" → "mathematical" necessity.
2. Bisect H* between the bracketing grid points (locate it, don't just bracket).
3. Real-d memory run (if ever hardware-feasible) — else leave exp-0002's
   memory gap formally open, as here.

## Honest caveats baked in

Toy nested-optimizer (existence of an operational frontier, not LM-scale).
Wall-time-blowup is a softer failure than OOM — reported separately, never
blended. The only new claim over exp-0002 is the *survival frontier*;
quality is parity (already established), stated as such.
