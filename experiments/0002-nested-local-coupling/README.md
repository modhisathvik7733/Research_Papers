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
| **C-stronger** (pre-registered) | `--p2-strength 10` | predicted monotone margin (a) + flip at h\*≤4 (b) + div↑h (c) → **(a) confirmed, (b)(c) NOT**. Margin grows 0.027→0.169 monotone but pooled inflates in lockstep; rule DEGENERATE at every h∈{0..4}; **bounding clause triggered → §4.3/Eq.45 family ROBUSTLY degenerate**. |
| **P3** out-of-family (pre-registered) | `--p3 10` | predicted P3-A non-degenerate → run head-to-head P3-B. **P3-A DEGENERATE, prediction WRONG (reported wrong)**: cyclic reactivation, scalar 0.419 vs best-nested 0.387, margin +0.041 ≤ pooled 0.060; nested wins 9/10 seeds but never clears seed variance. P3-B gated off (as committed). local-PC≈global (tie, qgap −0.006). |

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

**C-stronger sweep (`--p2-strength 10`, h∈{0,1,2,3,4}):**

```
   h   scalar  nest_best   margin   pooled  div_gl     verdict
 0.0   0.6097     0.5833  +0.0265   0.0988    0/10  DEGENERATE   (sanity ✓)
 1.0   0.7483     0.6998  +0.0484   0.0653    1/10  DEGENERATE
 2.0   0.9412     0.8727  +0.0685   0.0931    0/10  DEGENERATE
 3.0   1.1564     1.0288  +0.1276   0.1491    0/10  DEGENERATE
 4.0   1.3425     1.1731  +0.1694   0.1860    0/10  DEGENERATE
margin monotone non-decreasing in h: True   |   h=0 sanity: DEGENERATE
margin/pooled ratio: 0.27 → 0.74 → 0.74 → 0.86 → 0.91  (→ never crosses 1)
```

(a) monotone margin **confirmed**; (b) flip at h\*≤4 **NOT** (variance
inflates in lockstep — rule DEGENERATE at every h); (c) div↑h **NOT** (1/10
at h=1 only). Bounding clause triggered. **Not extrapolated past the
pre-registered grid** (at h=4 per-task condition spread ≈2000:1; further is
pathological).

**Synthesis:** §4.3/Eq.45 degeneracy is **scale-robust** and **family-level
robust** — it survives P1, scale-up, heterogeneous Hessians + non-orthogonal
directions, AND a pre-registered curvature-spread sweep to ~2000:1. The mean
nested advantage grows but never clears seed noise within the family.
Contribution A moves from *contingent* to **explicitly bounded**: no
demonstrated home anywhere in this family; it retains only a stability edge
over the sometimes-diverging hypergradient. Contribution B deflated,
family-level. Next benchmark must **leave the family** (P3).

## P3 — out-of-family decisive benchmark (RESOLVED 2026-05-17)

```
P3 cyclic reactivation (N=3, B=30, stream=600, 10 seeds, K=8)
  best scalar : 0.4194 ± 0.064
  local-PC    : 0.3931 ± 0.064   graph 101  (O(1) in H)
  global      : 0.3874 ± 0.056   graph 582  (O(H·K))
  scalar − best-nested = +0.041  (pooled 0.060)   div: 0/0/0
P3-A: DEGENERATE (margin ≤ pooled) — prediction WRONG, reported wrong
P3-B: GATED OFF (pre-registered: only runs if P3-A holds)
```
nested beats scalar 9/10 seeds but never clears seed variance — the
C-stronger obstruction reproduced on the temporal axis. local-PC vs global =
statistical tie (qgap −0.006 ≪ pooled), as on every prior benchmark.

## Paired analysis (§7.6.7–8) — pre-registered NEW, correct variance

`--paired 10`. Methods already share the per-seed task draw (common random
numbers) ⇒ paired design; the verbatim rule used the wrong (between-seed)
variance. Originals reprinted UNCHANGED; paired alongside:

```
construction   original(between-seed,UNCHANGED)   paired Δ(scalar−nested)
canonical-P1   DEGENERATE                          m+0.014 t2.26 8/10 p.109 -> AMBIGUOUS
C-hetero       DEGENERATE                          m+0.048 t4.25 8/9  p.039 -> NON-DEGEN
P3-cyclic      DEGENERATE                          m+0.041 t3.77 9/10 p.021 -> NON-DEGEN
head-to-head global vs local-PC: paired TIE on both (raw lean mildly to global)
```

**Higher-n confirmation (`--paired 60`, §7.6.9–10) — resolves it:**

```
construction   n=10 paired      n=60 paired                     reading
canonical-P1   AMBIGUOUS        DEGENERATE m+.003 t0.95 40/60    n=10 = noise; HEADLINE RESTORED
C-hetero       NON-DEGEN        AMBIGUOUS  m+.030 t4.68 48/59    p<.001 but ~0.6σ (modest)
P3-cyclic      NON-DEGEN        NON-DEGEN  m+.038 t8.85 55/60    p<.001, 1.14σ — DECISIVE
head-to-head (P3): paired TIE m+.006 33/60 -> local-PC by dominance
```

Resolved, reported straight: **canonical is genuinely DEGENERATE** (n=10
ambiguity was underpowered noise — the §7.6.8 withdrawal is *reinstated* at
n=60, audit trail kept). **P3: nesting decisively beats the tuned scalar**
(t=8.85, p<.001, pre-registered effect-size gate cleared) — the real positive.
**C-hetero: real but modest** (highly significant by sign/t, sub-1σ effect).
Head-to-head: **paired tie ⇒ local-PC better only by dominance**, NOT quality.
Scorecard: H-n1 half right (P3 yes, C-hetero no — significant≠large); H-n2
verdict right / mechanism wrong (effect collapsed to ≈0, not "small
consistent"); H-n3 confirmed.

## Bottom line — IS THE LOCAL IDEA WORKING? (resolved at n=60)

- **Structural/efficiency primitive: YES, unconditionally.** O(1)-in-H
  (101 vs 582 nodes), 0 divergences.
- **Canonical §4.3/Eq.45: genuinely DEGENERATE** (n=60, gap≈0, t=0.95). The
  central deflationary headline holds on solid footing.
- **Does nesting/credit-assignment matter? YES — off the canonical geometry,
  decisively on P3** (n=60 t=8.85, p<.001, pre-registered gate cleared);
  real-but-modest on C-hetero (t=4.68, p<.001, ~0.6σ).
- **local-PC vs global: paired quality TIE** (n=60 lean a coin-flip) ⇒
  **local-PC the better method by strict dominance** (equal quality, O(1) vs
  O(H), better stability — global diverged 1/60 on C-hetero, local-PC never).
  NOT a quality win; an efficiency+stability win at equal quality.
- **Remaining:** C-scale-paired (untested half of P-1); formal effect-size
  account of C-hetero. Neither changes the resolved picture.

## Remaining

1. C-scale under the paired analysis (untested half of prediction P-1).
2. Formal effect-size characterisation of C-hetero (significant but ~0.6σ).
3. Divergence-rate confirmation (global blow-up sparse: 1/60 on C-hetero).
4. Real-d demonstration converting the proven graph-memory law into an actual
   OOM/timeout (extrapolation only so far).
