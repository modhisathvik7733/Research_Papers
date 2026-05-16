# 0002 — Local predictive-coding coupling vs global backprop-through-nesting

**Status:** RESOLVED. Structural claim robust (100% of seeds, unconditional).
Quality "parity" **deflated by the pre-registered adversarial control P1,
which was run and returned DEGENERATE** — a single tuned scalar matches both
local-PC and the global hypergradient on §4.3/Eq. 45. See §"P1 resolution".
**Hardware:** CPU, ~5 s/seed on an M1 Max. `python3 run.py` (single seed,
plots) or `python3 run.py --seeds 10` (robustness sweep).

## Hypothesis (the single falsifiable post-HOPE claim)

A K-level nested optimizer-memory whose meta-parameters are credited by **local
adjacent-level prediction-error** (everything detached across levels and steps)
should:

1. **(equivalence)** reach a continual-learning solution comparable to a
   **global** optimizer credited by backprop through the unrolled inner loop, and
2. **(scaling)** keep its backward graph free of the unroll-depth (H) multiplier
   that global backprop-through-nesting carries — i.e. remove the scaling wall
   unique to Nested Learning.

A flat momentum-SGD baseline establishes that catastrophic forgetting exists.

## Design

- **Task family = paper Eq. 45.** Shared, *non-learned* random feature map
  `φ(x)=tanh(Fx)`; student is a linear readout `θ∈R^24`; task *t* targets an
  **orthonormal** direction `r_t`. Sequential stream of 10 tasks, 120 steps each.
  Momentum drags the shared readout across orthogonal directions ⇒ provable
  catastrophic forgetting (this is the §4.3 mechanism, not generic task drift).
- **Nested optimizer-memory.** K-level momentum cascade
  `m_k = σ(a_k)m_k + g_k m_{k-1}`, `Δθ = Σ c_k m_k`. Meta-params `{a,g,c}`.
- **`global`** — `{a,g,c}` trained by backprop through an H=8-step unrolled
  inner loop; retention meta-loss = current + anchor task (the §4.3 objective).
- **`localpc`** — `{a,g,c}` trained only by local errors: each slower level
  predicts the slow-EMA of the faster level; the combiner predicts the §4.3
  retentive (anchor-orthogonalised) update. All detached across levels/steps.
- **Metric for the scaling claim** is the *exact autograd-graph node count*
  behind the loss that is differentiated — not wall time (wall time is
  confounded by Python/optimizer-step overhead and was rejected in v2).

## Pre-registered pass criteria

1. `localpc` final avg loss ≤ 1.25× `global` **and** ≤ flat baseline.
2. Real forgetting exists (flat forgetting > 1e-3) **and** `localpc` ≤ flat.
3. `global` backward-graph grows ≥1.8× over K1→K8 **and** `global` nodes >
   4× `localpc` nodes at K8 (i.e. global carries the ×H unroll factor, local
   does not).

## Result (10 seeds, mean ± std, K=8)

```
flat momentum-SGD   final 0.644±0.114   forget 0.716±0.126
global              final 0.617±0.114   nodes 582 (every seed)
local-PC            final 0.609±0.104   forget 0.666±0.116   nodes 101 (every seed)
gate pass-rate: structural=100%  quality=80%  helps=80%  all=80%
node ratio g8/l8 = 5.8x, deterministic across all seeds
```

- **Structural gate: 100% of seeds** — local-PC's differentiated-loss graph
  carries no ×H unroll factor; 5.8× smaller than global at K=8, every seed.
- **Quality/helps: 80%.** Means favour local-PC over global and beat flat, but
  within 1 std → "competitive, not superior." The 2 failing seeds are the same
  low-interference draws where flat barely forgets and the nested memory is
  correctly ~neutral (not harmful).

## Honest caveats (do not over-read)

- **Toy.** 24-d linear-in-features readout, 10 orthogonal tasks, single seed.
  Supports the *mechanism*, not scale or deep nonlinear memories.
- **Forgetting reduction is modest** (~11% rel.). Consistent with the paper's
  own claim that forgetting is conserved under finite capacity — the nested
  memory helps at the margin, it does not "solve" forgetting.
- **`localpc` > `global` is not proof local is superior.** Truncated-unroll
  hypergradients are hard to optimise at this scale (classic L2O pathology);
  the honest claim is *competitive and structurally cheaper*, not *better*.
- **`localpc` graph still grows ~linearly in K** (16→101): the surviving claim
  is specifically "no ×H unroll multiplier," which is the actual NL wall.

## Escalation chain — complete

| step | command | result |
|---|---|---|
| multi-seed (10) | `--seeds 10` | structural 100%, quality/helps 80% |
| target-design ablation (10) | `--ablation 10` | de-tuned ≡ orig ⇒ gain is **coupling**, not targets |
| nonlinear MLP memory (10) | `--nonlinear 10` | quality 100%, structural 100% (5.8×) |
| scale-to-failure (H) | `--scale` | graph: global O(H) 624→155,664; local-PC **flat 107 ∀H** (1455× @H2048, unbounded). Wall-clock crash NOT shown at toy d (honest). |
| deep nonlinear + fairness (10) | `--deep 10` | global\*(best-of-LR-grid) 0.618 vs local-PC(untuned) 0.615 vs flat 0.644; **quality 100%, structural 100%** — "global undertuned" critique dead |
| theory test F3: λ-sweep | `--lambda 8` / `--lambda-fair 8` | **FALSIFIED** — gap *grows* with λ (slope +0.55, wrong sign) even with global LR controlled; Θ(1/λ) premise wrong |
| theory test F4: interleave | `--interleave 8` | **FALSIFIED as stated** — advantage non-monotone (peaks mid-interleaving). Parity \|g−l\| ≈0.01 across ALL regimes (post-P1: this is the degeneracy showing through, not coupling strength) |
| **P1 adversarial control** | `--p1 10` | **DEGENERATE** — single tuned scalar ties both local-PC and global (see below) |
| **C-scale** (pre-registered) | `--p1-scale 8` | predicted DEGENERATE → **CONFIRMED**: same geometry 10×/5×/5×/5× larger still degenerate (sc 0.751 / lpc 0.750 / gl 0.748, margin +0.004, 0/8 div). Degeneracy is geometry, not size. |
| **C-hetero** (pre-registered) | `--p2 10` | predicted "NESTING MATTERS" → **NOT cleanly confirmed**: verbatim rule still DEGENERATE on 9 stable seeds (margin +0.048 ≤ pooled 0.065). BUT degeneracy breaking (monotone gl 0.704 < lpc 0.729 < sc 0.748; exploratory paired gl<sc 8/9) + **global diverged 1/10** (L2O blow-up, never on canonical geometry). |

## P1 resolution (the pre-registered control — RUN, DEGENERATE)

We pre-committed to running P1 and reporting it whichever way it fell. We ran
it. Linear, K=8, 10 seeds:

```
best single tuned scalar (just β)   final 0.617 ± 0.113
local-PC (K=8 nested)               final 0.611 ± 0.099
global  (K=8 hypergradient)         final 0.617 ± 0.115
gaps ≈0.01, pooled std ≈0.107  →  all three statistically identical
best-β picks mostly 0.0 or 0.5 (plain / lightly-momented SGD)
P1 VERDICT: DEGENERATE
```

**What it means, reported straight:**

1. **H-deg confirmed.** The continual-optimal inner optimizer on §4.3/Eq. 45 is
   effectively ~1-dimensional. The nested cascade buys nothing here.
2. **Contribution B (quality parity) deflated.** "local-PC matches global" is
   true but uninformative — a single scalar matches both. The benchmark cannot
   distinguish credit-assignment methods at all.
3. **This indicts the hypergradient equally.** HOPE/Titans-style
   backprop-through-nesting also buys nothing over a tuned scalar here. Not
   "we lose" — **NL's own canonical forgetting benchmark is optimizer-
   degenerate** for arguing about optimizers/nesting. Real, defensible, more
   interesting result — a fundamentally different paper.
4. **Contribution A (O(1)-vs-O(H) structural law) untouched but motivation
   weakened** — a memory-cheaper way to do something that doesn't matter on
   this benchmark needs a non-degenerate benchmark to be worth anything.

**Defensible claim (post-P1):** the *only* surviving positive claim is the
**unconditional structural law** — local PC coupling makes the backward graph
**O(1) in unroll horizon vs O(H)** (1455× @H2048, unbounded). The
credit-assignment "parity" is reported as **deflated by our own control**, not
as a contribution. Never frame as "matches/beats global" without immediately
stating a scalar also matches both.

## Scale & heterogeneity controls — RESOLVED (2026-05-17)

Both pre-registered before running; **P1 decision rule reused verbatim** (no
new goalposts). Added a **divergence guard** to `p1` (a run is diverged if
loss non-finite or >1e3; counted, never silently averaged) — backward-compatible:
original `--p1` has 0 divergences and is numerically unchanged (regression-
checked: 0.617/0.611/0.617 → DEGENERATE, as before).

- **C-scale (`--p1-scale 8`): prediction CONFIRMED.** Same orthonormal/
  shared-Σ geometry, D_in 64 / D_feat 256 / N_tasks 50 / stream 600. scalar
  0.751±0.065, local-PC 0.750±0.061, global 0.748±0.061; margin +0.004 ≪
  pooled 0.061; **0/8 divergences**; DEGENERATE. Scale alone does NOT break
  degeneracy → H-deg's "geometry not size" premise survives a falsification
  test; the "just scale up" objection is answered.
- **C-hetero (`--p2 10`): prediction NOT cleanly confirmed; degeneracy
  breaking; instability finding.** Per-task input covariance (heterogeneous
  Σ_t) + non-orthogonal correlated r_t, toy size. Verbatim rule on 9 stable
  seeds still **DEGENERATE** (sc 0.748 / lpc 0.729 / gl 0.704; margin +0.048 ≤
  pooled 0.065) — my prediction of clean separation was **wrong, reported as
  wrong**. But: (i) monotone ordering global < local-PC < scalar emerges
  (absent on toy where all ≈0.617); (ii) *exploratory, non-pre-registered*
  paired sign test: global < scalar on **8/9** stable seeds (≈p0.02);
  (iii) **global hypergradient diverged 1/10** (1.08e23 — L2O pathology),
  while local-PC and scalar never diverged.

**Synthesis:** §4.3/Eq.45 degeneracy is **scale-robust** and only **partially
lifted by strong heterogeneity** — a deep property, not a small-scale artifact.
The one regime where credit assignment starts to matter is exactly where the
expensive hypergradient is unstable and the cheap O(1) local rule is not →
Contribution A re-motivated on a **second axis (stability)**; Contribution B
stays deflated by the governing verdict.

## Remaining (now the critical path)

1. **P2-stronger** — raise heterogeneity strength until the pre-registered
   separation bar is *cleanly* cleared (or bound where it cannot be). C-hetero
   shows the direction; this pins the regime.
2. **P3** — bias–variance crossover where genuine long-horizon credit is
   required (global should overtake scalar there).
3. **Divergence-rate confirmation** — the 1/10 global blow-up is first-class
   but n=10; confirm the rate at higher n.
4. Real-d demonstration converting the proven graph-memory law into an actual
   OOM/timeout (extrapolation only so far).
5. Deep-MLP *per level* (currently tanh-per-level + MLP combiner).
