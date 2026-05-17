# Finding (proposed): Local-PC's Only Defensible Home — the Unroll Horizon Where Backprop-Through-Optimization Operationally Fails and Local-PC Does Not

- **Date:** 2026-05-17
- **Source:** Extension of exp-0002 (the O(1)-vs-O(H) structural law +
  the explicitly-deferred "OOM/crash is an extrapolation, not a result")
  after the CL-quality angle was closed (exp-0002/0004 negatives).
- **Status:** RUN (2026-05-17). **Divergence channel: FRONTIER DEMONSTRATED**
  (H*≈512, global fails 10/10, local-PC 0/10, quality parity p=1.000) —
  operational necessity at parity quality, the one honest home. Qualifiers:
  failure-mode not disaggregated (operational, not yet confirmed-mathematical);
  memory channel does NOT bite at d=256 (no OOM, wall-time only, parity gate
  fails on a negligible 0.0002 — exp-0002's gap restated as pre-flagged).
  Capability claim only. Results + follow-ups in
  [experiments/0005-localpc-horizon-frontier/](../experiments/0005-localpc-horizon-frontier/).
- **Intended use:** give Local-PC the one identity the evidence supports —
  *the credit rule that makes deep/long nested optimization feasible at all* —
  by converting the proven asymptotic law into a demonstrated operational
  failure of the hypergradient that Local-PC survives at equal quality.

## 1. One-sentence contribution

Backprop-through-optimization has an O(H·K·d) backward graph and is a product
of H Jacobians; Local-PC credit is O(1) in H and per-step. We claim there is a
**reachable** unroll horizon H\* at which the hypergradient *operationally
fails* (out-of-memory, wall-time blowup, or divergence to non-finite) with
probability →1, while Local-PC remains finite and **within ε of its own
small-H quality** — i.e. Local-PC is *necessary, not merely cheaper*, beyond
H\*.

## 2. Why this is the honest extension

exp-0002 proved the graph-node law (global O(H) → 155k @H2048; local-PC flat
at 107) but explicitly disclosed: *"the asymptotic memory law is proven; the
hardware crash is an extrapolation, not a result."* exp-0002/0004 then closed
the continual-learning-quality angle (degenerate benchmark; ties; anti-stacks
with replay). What survives untouched is the structural/robustness axis. This
experiment discharges exactly the deferred claim — no new theory, just the
operational demonstration the prior work said it had not done.

## 3. Falsifiable claim (pre-registration)

> **H-frontier.** Sweeping inner unroll horizon H (and, on a second axis,
> parameter dimension d and inner nonlinearity), there exists H\* within a
> hardware-feasible range where: (a) the **global** hypergradient fails —
> non-finite loss, OR peak memory > budget, OR per-step wall-time > 50×
> its small-H value — on ≥ 80% of seeds; (b) **Local-PC** stays finite and
> within ε (pre-registered ε = 1.25× its own small-H meta-loss) at the same
> H; (c) where **both** still run, Local-PC is a **paired tie** on quality
> (NOT worse — verbatim exp-0002 three-way rule). Local-PC must *tie* on
> quality and *win only by surviving*; a quality win is not claimed.
>
> **Non-degeneracy guard (must pass first).** The global hypergradient must
> actually fail within the feasible range. If it never fails up to the
> largest H we can run, the operational payoff is **unreached** — reported
> straight as "the exp-0002 honest gap remains open," not spun.
>
> **Falsified if:** Local-PC also fails by H\* (no operational payoff
> anywhere — a decisive bound on the whole idea), or is quality-worse where
> both run.

## 4. Design (cheap, self-contained)

Reuse the exp-0002 nested optimizer-memory. Two failure axes, pre-registered
grids: (i) **memory** — raise d and K so global's O(H·K·d) graph crosses a
fixed RSS budget; (ii) **divergence** — deep nonlinear inner recurrence + long
H so the H-Jacobian product blows up. Metrics per seed: finite? peak RSS,
per-step wall-time, final meta-loss; the exact autograd-graph node count
(O(H·K) vs O(1)) as the structural anchor. Paired CRN, divergence guard,
verbatim three-way on the quality gap at max-common-H.

## 5. Honest limitations

Toy nested-optimizer (existence of the operational frontier, not LM-scale).
"Failure" includes wall-time blowup, a softer criterion than OOM — reported
separately, not blended. Equal-quality-where-both-run is a *parity* claim;
exp-0002 already showed parity, so the only new thing is the survival
frontier — stated as such.

## 6. Score

- Novelty: 4 (discharges a named, deferred claim; reframes the contribution
  onto the one axis the evidence supports)
- Testable cheaply: 4 (exp-0002 harness + a memory/divergence sweep; ≤1 day)
- Informative if it fails: 5 (if global never fails feasibly, or Local-PC
  fails too, the structural law has *no operational payoff* — that bound is
  the most important thing we could learn about the whole idea)
- **Total = 13 → promote to experiment 0005.**
