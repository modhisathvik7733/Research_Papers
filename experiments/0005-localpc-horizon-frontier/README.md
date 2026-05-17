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

## Honest caveats baked in

Toy nested-optimizer (existence of an operational frontier, not LM-scale).
Wall-time-blowup is a softer failure than OOM — reported separately, never
blended. The only new claim over exp-0002 is the *survival frontier*;
quality is parity (already established), stated as such.
