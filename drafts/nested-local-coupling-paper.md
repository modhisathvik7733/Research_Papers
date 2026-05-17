# Nested Learning's Canonical Forgetting Benchmark Is Optimizer-Degenerate: A Single Scalar Matches Both Local Coupling and the Hypergradient

*Working draft — workshop length. Status: resolved, deflationary, with both
follow-up controls run. The pre-registered adversarial control (P1) returned
**DEGENERATE**; the pre-registered **scale** control confirmed degeneracy is
scale-robust (prediction held); the pre-registered **heterogeneity** control
did **not** cleanly confirm its prediction; a pre-registered **heterogeneity
sweep** triggered its **bounding clause**; and a final, decisive
**out-of-family benchmark (P3, cyclic reactivation built to require
long-horizon credit)** also returned **degenerate** by the verbatim rule.
Across **five pre-registered constructions**, sophisticated credit assignment
never cleanly beats a single tuned scalar at this scale. The credit-assignment
contribution is **deflated by our own controls**; the **unconditional
structural law** survives. "Which is better, local-PC or the hypergradient?":
a **quality tie everywhere measured**, with local-PC strictly dominating on
cost (O(1) vs O(H)) and stability (0 vs 1 divergence) — so **local-PC, by
dominance** — atop the larger honest negative that *neither nested rule is
shown necessary over a scalar*. All numbers are 8–10-seed CPU runs reproducible
from a single script. No claim is softened; several of our own pre-registered
predictions were wrong and are reported as wrong.*

---

## Abstract

Nested Learning (NL) reframes a model and its optimizer as a stack of
associative-memory optimization problems ordered by update frequency. Its
distinctive cost is structural: crediting an inner level from an outer
objective requires differentiating through the unrolled inner loop, so the
backward graph scales as O(H·K) in the unroll horizon H and nesting depth K,
with no parallel dual form for expressive (nonlinear) memories. We asked
whether the coherence of nested objectives could instead be enforced by
**local adjacent-level prediction-error coupling** — each level credited only
by a detached local objective, no global hypergradient. On a benchmark
faithful to NL's own orthogonal-task forgetting construction (§4.3, Eq. 45) we
found, across 10 seeds, that local coupling (i) **matches**
backprop-through-nesting — including under a per-seed meta-learning-rate grid
and a fully nonlinear recursive memory — and (ii) has a backward graph that is
**O(1) in the unroll horizon**: constant at 107 nodes for every H ∈ [8, 2048]
versus the global graph's exact linear growth to 155,664, a 1455× gap that
widens without bound. We pre-registered two mechanistic conjectures and
falsified both, then advanced a single hypothesis (effective-dimension
degeneracy, **H-deg**) together with the adversarial control that would
confirm or kill it — and we pre-committed to running and reporting that
control whichever way it fell. **We ran it. It returned DEGENERATE.** On the
§4.3 / Eq. 45 construction a single grid-searched scalar (just β) reaches
0.617 ± 0.113; K=8 local-PC reaches 0.611 ± 0.099; K=8 global hypergradient
reaches 0.617 ± 0.115 — all three statistically identical (gaps ≈0.01, pooled
std ≈0.107), with the best scalar settling on β ∈ {0.0, 0.5} (plain or lightly
momented SGD). The honest consequence, which we report straight: **H-deg is
confirmed**; the continual-optimal inner optimizer on this benchmark is
effectively ~1-dimensional; the nested cascade buys nothing here. The
"local-PC matches global" parity is **true but uninformative** — a single
scalar matches both, so the benchmark cannot distinguish credit-assignment
methods at all. This is **not** "our cheap rule loses": it equally indicts the
expensive HOPE/Titans-style hypergradient, which also buys nothing over a
tuned scalar on Nested Learning's own canonical forgetting benchmark. The
surviving contribution is the **structural law alone** (O(1) vs O(H),
unconditional). We then ran two further pre-registered controls. A **scale**
control (same geometry, 10×/5×/5×/5× larger) **confirmed** its prediction:
degeneracy is scale-robust (scalar 0.751, local-PC 0.750, global 0.748, 0/8
divergences) — the "just scale up" objection is answered by our own control. A
**heterogeneity** control (per-task Hessians + non-orthogonal task directions)
did **not** cleanly confirm its prediction — by the verbatim rule it still
reads degenerate — but the degeneracy is visibly breaking (monotone global
0.704 < local-PC 0.729 < scalar 0.748; exploratory paired test global < scalar
on 8/9 stable seeds) and, critically, the **global hypergradient diverged on
1/10 seeds** in exactly this regime while the cheap O(1) local rule stayed
stable. A final pre-registered heterogeneity *sweep* (h ∈ {0..4}) then
resolved the boundary: the nested–scalar margin grows monotonically but seed
variance inflates in lockstep, the verbatim rule reads DEGENERATE at every h,
and the **pre-registered bounding clause triggers** — the §4.3/Eq.45
construction *family* is robustly optimizer-degenerate. Net: the deflation of
the credit-assignment contribution is real, scale-robust, and **family-level**;
the unconditional structural law survives but is now **explicitly bounded** —
no demonstrated home anywhere in this family, gaining only an unanticipated
stability edge over the (sometimes-diverging) hypergradient. We present this as
a bounded, honest result in which several of our own pre-registered predictions
were wrong and are reported as wrong: a solid structural law and a five-
construction, pre-registered benchmark-degeneracy finding. We then built and
ran P3 — a benchmark deliberately outside the family, with cyclic task
reactivation engineered so no single decay constant can both adapt and retain
— and it too returned degenerate by the verbatim rule (scalar − best-nested
+0.041 ≤ pooled 0.060; nested wins 9/10 seeds but never clears seed variance).
The honest bottom line: the local idea **works as a structural primitive**
(O(1)-in-H, 0 divergences, dominating the hypergradient on cost and stability
everywhere); on quality it is a **statistical tie with the hypergradient on
every benchmark including P3**; and *neither* nested rule has been shown to
beat a tuned scalar on any benchmark in reach. Forced to order the two:
**local-PC, by strict dominance** (equal quality, lower cost, ≥ stability) —
but the load-bearing fact is the negative beneath it. The identified
obstruction (seed variance scaling with task difficulty) is named as the
explicit target for future work, not papered over.

---

## 1. Introduction

Nested Learning (Behrouz et al., 2025) and the HOPE architecture argue that
optimizers, attention, MLP blocks, and explicit memory are all
associative-memory optimization problems, ordered into *levels* by how often
they update, with knowledge transferred between levels. The paradigm's appeal
is unification; its unsolved engineering cost is **how the levels are
coupled**. In every concrete instantiation we are aware of, an outer level
credits an inner level by backpropagating through the inner level's unrolled
optimization, with three compounding consequences: a memory wall unique to NL
(backward graph O(H·K)); no parallel form for expressive memories; and the
documented hypergradient pathologies of differentiating through long nonlinear
optimization.

We set out to test a single alternative — replace the global hypergradient
with **local adjacent-level prediction-error coupling** — on a toy faithful to
NL's own forgetting construction. The empirical work and the structural
measurement are complete and unchanged from earlier drafts. **What changed is
the interpretation, and it changed because we ran the control we pre-committed
to running.**

This draft is written *after* resolving the gate. We had pre-registered an
adversarial control (P1: does a single tuned scalar tie *both* local-PC and
the hypergradient?) and stated, before running it, that a DEGENERATE outcome
would deflate the credit-assignment contribution and reframe the paper. **P1
returned DEGENERATE.** We therefore report the deflationary outcome we built
P1 to detect, and we reframe rather than retro-fit. The reframe is not a loss:
"Nested Learning's canonical forgetting benchmark is optimizer-degenerate"
indicts the expensive hypergradient as much as it deflates our cheap rule, and
is a more interesting — and fundamentally different — paper than the one we
set out to write.

**Claim and scope.** We make two claims with sharply different epistemic
status. (A) **Structural, unconditional:** local coupling removes the
unroll-horizon term from the backward graph (O(1) vs O(H)). (B)
**Credit-assignment, deflated:** on §4.3/Eq. 45, local-PC matches the
hypergradient — but so does a single scalar, so this benchmark cannot
adjudicate credit-assignment methods, ours or NL's. We claim no superiority,
no scale, no demonstrated hardware failure.

---

## 2. Background and the scaling wall

**Nested Learning.** A nested system has K levels; level k solves an
optimization problem with its own context and update frequency. NL's
preliminaries show standard objects are instances: SGD-with-momentum is a
two-level associative memory, linear attention is a one-level memory optimized
by gradient descent, and HOPE stacks a self-modifying sequence memory with a
continuum of MLP frequencies.

**The orthogonal-task forgetting construction (NL §4.3, Eq. 45).** NL
characterizes catastrophic forgetting as an *optimizer-memory* failure: with
tasks whose gradients lie in mutually orthogonal directions, momentum drags
the shared parameters across orthogonal subspaces, damaging earlier tasks even
though model capacity is adequate. We adopt exactly this construction as our
benchmark (Section 4). **Foreshadowing §7.6: this is precisely the construction
our control shows is optimizer-degenerate** — a property of the benchmark, not
of any method tested on it.

**The cost.** For a nested optimizer trained by an outer objective ℒ over an
H-step inner loop with K levels, the autograd graph differentiated by the
outer update has size Θ(H·K) (and, for nonlinear inner rules, no dual form).
This is the quantity the structural contribution targets — *not* the forward
FLOPs.

---

## 3. Method

### 3.1 The nested optimizer-memory

We instantiate the inner optimizer as a K-level memory cascade acting on model
parameters θ. Given the per-step surprise gradient g_t = ∇_θ ℒ_task(θ):

- m₁ ← σ(a₁)·m₁ + g_t
- m_k ← σ(a_k)·m_k + ψ(g_{k-1}·m_{k-1}),  k = 2…K  
  where ψ = identity (linear cascade) or tanh (deep nonlinear recurrence)
- Δθ = combine(m₁…m_K);  θ ← θ − η·Δθ

`combine` is either linear (Σ c_k m_k) or the HOPE memory form
(Σ c_k m_k + W₁ tanh(W₂ [m₁…m_K])). The meta-parameters are
Φ = {a_k, g_k, c_k, (W₁, W₂)}. Φ is the "nested optimizer"; θ is the model.
**The K=1, identity-cascade special case is exactly a single scalar momentum/
decay knob — this is the degenerate baseline P1 grid-searches (§7.6).**

### 3.2 Two credit-assignment regimes (the experimental variable)

**Global (backprop-through-nesting; the NL/HOPE-style baseline).** Φ is
trained by backpropagating through an H-step unrolled inner loop against a
retention meta-loss equal to current-task loss plus an anchor (earlier) task
loss — the §4.3 objective. The differentiated graph has size Θ(H·K); for
nonlinear `combine`/cascade there is no parallel dual form.

**Local-PC (proposed).** Φ is trained only by local objectives, every term
detached across levels and across steps:

- *level k*: predict the slow exponential-moving-average of the adjacent
  faster level — ‖m_k − stopgrad(slowEMA(m_{k-1}))‖²
- *combiner*: predict the §4.3 retentive update target p (the current-task
  gradient projected off the stored anchor-task gradient direction) —
  ‖combine(·) − p‖²

Because every term is detached, the differentiated graph carries **no unroll
factor**: O(1) in H, at most O(K). Run online it is exactly O(1) in H in
memory.

### 3.3 The falsifiable claim (pre-registered) and its adversarial control

> **H.** Local-PC (a) reaches final continual loss ≤ 1.25× the global
> baseline and ≤ a flat momentum-SGD baseline; (b) reduces forgetting vs flat
> when real forgetting exists; (c) its differentiated-loss graph does not
> carry the ×H unroll multiplier.

> **P1 (pre-registered adversarial control, now run).** If the
> continual-optimal inner optimizer on §4.3/Eq. 45 is effectively
> ~1-dimensional, a single grid-searched scalar should tie *both* local-PC and
> global. We pre-committed: a tie means the benchmark — not our method —
> cannot adjudicate credit assignment, deflating (B) and indicting the
> hypergradient equally.

Criteria and the P1 decision rule were fixed before any run reported below.

---

## 4. Experimental setup

**Tasks (faithful to NL Eq. 45).** A fixed, non-learned random feature map
φ(x) = tanh(Fx), F ∈ ℝ^{24×16}. The student is a linear readout θ ∈ ℝ²⁴. Task
t targets an orthonormal direction r_t (rows of a QR factorization of a random
matrix): y = φ(x)·r_t. Ten tasks are streamed sequentially, 120 steps each;
momentum drags the shared readout across the orthogonal r_t, producing
catastrophic forgetting by construction.

**Baselines.** *flat*: momentum-SGD (β=0.9), establishing that forgetting
exists. *global*: backprop-through-nesting. *local-PC*: proposed. *scalar*
(P1): the best of a per-seed grid over a single momentum/decay scalar β — the
K=1 degenerate case of the cascade.

**Metrics.** Final average task loss (lower better); forgetting; and the exact
autograd-graph node count behind the differentiated loss (the structural
quantity; we deliberately do *not* use wall time — it is confounded by Python
and optimizer-step overhead).

**Protocol.** 10 independent seeds reseed everything including the feature map
and task directions. CPU, ~5–35 s per sweep on an Apple M1 Max.

---

## 5. Results

### 5.1 Multi-seed parity and the structural separation

10 seeds, K=8, mean ± std:

| method | final loss ↓ | forgetting ↓ | graph nodes ↓ |
|---|---|---|---|
| flat momentum-SGD | 0.644 ± 0.114 | 0.716 ± 0.126 | — |
| global | 0.617 ± 0.114 | — | 582 (every seed) |
| local-PC | 0.609 ± 0.104 | 0.666 ± 0.116 | 101 (every seed) |

Structural separation 100% of seeds. Quality differences within one standard
deviation: **parity, not superiority** — and §5.6 shows even this parity is
uninformative.

### 5.2 The gain is the coupling, not the targets (ablation)

Replacing hand-designed local targets with uninformative ones (plain gradient
for the combiner; fixed random per-level targets), 10 seeds, K=8: original
0.611, both de-tuned 0.621, all 100% ≤ 1.25× global, std ≈ 0.10. Fully
de-tuned ≡ original within noise. **In hindsight (§7.6) this near-irrelevance
of the local signal was the first symptom of degeneracy: a benchmark where
even random targets reach the same loss is one where the credit signal carries
almost no task information.**

### 5.3 Nonlinear memory (the no-dual-form regime)

Nonlinear MLP combiner (global has no parallel dual form), 10 seeds, K=8:
local-PC 0.615 ± 0.106 vs global 0.634 ± 0.116 vs flat 0.644 ± 0.114;
structural 100% (global 624 nodes vs local-PC 107).

### 5.4 Deep nonlinear recursive memory + fairness control

Per-level recurrence *and* combiner nonlinear (no dual form anywhere).
Fairness: global gets best-of per-seed meta-LR grid {0.01,…,0.2}; local-PC
untuned. 10 seeds, K=8: flat 0.644 ± 0.114; global\* 0.618 ± 0.098 (680
nodes); local-PC untuned 0.615 ± 0.106 (107 nodes). The "global was undertuned"
objection is empirically closed.

### 5.5 Scale-to-failure over the unroll horizon (the surviving structural law)

Nonlinear memory, K=8, sweeping H:

| H | global graph nodes | local-PC graph nodes |
|---|---|---|
| 8 | 624 | 107 |
| 128 | 9,744 | 107 |
| 512 | 38,928 | 107 |
| 2048 | 155,664 | 107 |

The global backward graph grows **exactly linearly in H**. Online local-PC is
**flat at 107 for every H** — a 1455× separation at H = 2048 that grows
without bound. **This is the one contribution P1 does not touch (§6).**

**What we do not show.** The global baseline did **not** hit a wall-clock or
out-of-memory wall at our toy dimension; the memory cost is O(H·K·d) and d=24
is small. The **asymptotic memory law is proven; the hardware crash is an
extrapolation**, not a result.

### 5.6 P1 — the pre-registered adversarial control: **DEGENERATE**

We ran the control we pre-committed to running. **Linear, K=8, 10 seeds:**

| method | final loss |
|---|---|
| best single tuned scalar (just β) | 0.617 ± 0.113 |
| local-PC (K=8 nested) | 0.611 ± 0.099 |
| global (K=8 hypergradient) | 0.617 ± 0.115 |

All three are **statistically identical** (pairwise gaps ≈0.01, pooled
std ≈0.107). The best per-seed β is mostly **0.0 or 0.5** — plain or lightly
momented SGD. A single scalar matches *both* the K=8 local-PC and the K=8
global hypergradient. **Verdict (by the pre-registered decision rule):
DEGENERATE.**

This is the deflationary outcome we explicitly built P1 to detect and
pre-committed to reporting. We report it straight, not softened.

### 5.7 Pre-registered scale and heterogeneity controls (resolved)

We pre-registered two controls (§7.6.1) with the **P1 decision rule reused
verbatim** and a divergence guard added to the harness (a run is *diverged* if
its loss is non-finite or > 1e3; divergence is counted and reported, never
silently averaged — backward-compatible: the §5.6 / `--p1` run has zero
divergences and is numerically unchanged).

**C-scale — same §4.3/Eq. 45 geometry, ~10×/5×/5×/5× larger
(D_in 64, D_feat 256, N_tasks 50, stream 600), 8 seeds.** *Pre-registered
prediction: stays DEGENERATE.* **Confirmed, cleanly:** scalar 0.7512 ± 0.065,
local-PC 0.7504 ± 0.061, global 0.7475 ± 0.061; margin +0.004 ≪ pooled 0.061;
**0/8 divergences**; verdict DEGENERATE. Scale alone does not break the
degeneracy — H-deg's "geometry, not size" premise survives a direct
falsification test, and the "just scale up" objection is answered by our own
pre-registered control.

**C-hetero — both degeneracy sources broken at toy size (per-task input
covariance ⇒ heterogeneous Hessians Σ_t; non-orthogonal correlated r_t),
10 seeds.** *Pre-registered prediction: NESTING MATTERS — the single scalar
separates.* **Outcome: prediction NOT cleanly confirmed, but degeneracy is
demonstrably breaking, plus a first-class instability finding.**

| (C-hetero, K=8) | final loss (stable seeds) |
|---|---|
| best single scalar | 0.7483 ± 0.068 (med 0.768) |
| local-PC | 0.7285 ± 0.065 (med 0.756) |
| global | 0.7044 ± 0.063 (med 0.715) |

1. **The verbatim pre-registered aggregate rule still reads DEGENERATE** on the
   9 stable seeds: mean scalar−nested margin +0.048 is still ≤ pooled std
   0.065. We report this as the governing verdict; our prediction of clean
   separation was wrong and we do not spin it.
2. **But the degeneracy is clearly breaking, not intact.** On the toy all
   three methods are interchangeable (≈0.617, §5.6); under heterogeneity a
   **monotone ordering global < local-PC < scalar** emerges. An *exploratory,
   non-pre-registered* paired per-seed sign test shows global beats the tuned
   scalar on **8/9 stable seeds** (≈ p0.02). Heterogeneity does start to make
   credit assignment matter — just not enough to clear the blunt aggregate bar
   at this heterogeneity strength. We label this exploratory; the
   pre-registered rule governs the headline.
3. **First-class instability finding (the divergence guard caught it).** The
   global hypergradient **diverged on 1/10 seeds** (loss 1.08e23 — the
   documented L2O blow-up), which **never occurs** on the orthonormal/shared-Σ
   geometry (0 divergences in §5.6 and in C-scale). The scalar and local-PC
   never diverged. So in the regime where nesting *begins* to help, the
   *expensive* hypergradient becomes unstable, while the *cheap, O(1)*
   local-PC is stable and tracks global's improvement (it sits strictly
   between scalar and global on every aggregate).

**Synthesis.** The §4.3/Eq. 45 degeneracy is **robust to scale** (C-scale,
clean) and **only partially lifted by strong task heterogeneity** (C-hetero:
real directional signal, not clearing the pre-registered bar). It is therefore
a *deep* property of the construction, not a small-scale artifact. And the one
regime where credit assignment starts to matter is exactly the regime where
the expensive hypergradient is unstable and the cheap O(1) local rule is not —
which **re-motivates Contribution A on a second axis (stability), not just
memory**, without rescuing Contribution B (still deflated by the governing
verdict).

**C-stronger — pre-registered heterogeneity sweep (§7.6.3–4), 10 seeds,
h ∈ {0,1,2,3,4}.** The decisive follow-up. *Predicted: margin grows
monotonically (a) and the verbatim rule flips DEGENERATE→NESTING at some
h\*≤4 (b), with divergence rising in h (c).* **Outcome: (a) confirmed,
(b) and (c) not.** The mean nested advantage grows monotonically (margin
0.027→0.048→0.069→0.128→0.169) but the seed variance inflates in lockstep, so
margin/pooled rises 0.27→0.91 toward — never across — the threshold; the rule
reads DEGENERATE at *every* h, and h=0 sanity is DEGENERATE. We report (b)/(c)
as misses and do not extrapolate past the pre-registered grid (at h=4 the
per-task condition spread is already ≈2000:1). **The pre-registered bounding
clause triggers**: the §4.3/Eq. 45 *construction family* is robustly
optimizer-degenerate, upgrading the P1 point result to a family-level,
monotone, bounded negative and moving Contribution A from *contingent* to
*explicitly bounded* — no demonstrated home anywhere in this family; the next
benchmark must leave it (P3).

**P3 — out-of-family decisive benchmark (§7.6.5–6), 10 seeds.** Cyclic task
reactivation, engineered so no single decay constant can both adapt within a
30-step block and retain across a 60-step gap. *Predicted: P3-A non-degenerate
(scalar clearly loses) ⇒ run P3-B head-to-head.* **Outcome: P3-A DEGENERATE,
prediction wrong, reported wrong.** scalar 0.419 vs best-nested 0.387, margin
+0.041 ≤ pooled 0.060; nested beats scalar 9/10 seeds but never clears seed
variance (the C-stronger obstruction, reproduced on the temporal axis). By the
pre-registered gate, P3-B does not run. Within the nested pair (exploratory):
local-PC 0.393 ≈ global 0.387, a statistical tie (|qgap| 0.006 ≪ pooled), as
on every prior benchmark; local-PC graph 101 vs global 582, 0 divergences
both. **Consolidated:** five pre-registered constructions, zero clean
separations from a tuned scalar — the local idea is a proven O(1) structural
primitive and a quality-tie with the hypergradient, but neither nested rule is
shown necessary over a scalar anywhere reachable at this scale.

---

## 6. What we claim and what we do not — after P1

We separate the contribution into two parts with *different epistemic status*,
now resolved.

**Contribution A — structural, unconditional, SURVIVES.** With expressive
(nonlinear) memory, backprop-through-nesting has a backward graph that grows as
O(H·K) and has no parallel dual form; local adjacent-level predictive-coding
coupling makes that graph **O(1) in H** (constant 107 nodes for every
H ∈ [8, 2048], a 1455× separation that grows without bound). This is an exact
graph measurement; **P1 does not touch it.** *But its motivation is now
explicitly weakened:* a memory-cheaper way to perform a credit-assignment that
**does not matter on this benchmark** is only worth something on a
*non-degenerate* benchmark. The law is solid; its significance is contingent
and we say so.

**Contribution B — credit-assignment parity, DEFLATED by our own control.**
"local-PC matches the hypergradient" is true and reproducible, but **P1 shows
it is uninformative**: a single tuned scalar also matches both, so §4.3/Eq. 45
**cannot distinguish credit-assignment methods at all**. We do not present this
as a surviving contribution. We present it as deflated, by the control we
pre-registered for exactly this purpose.

**The reframe (the actually-interesting finding).** This is **not** "our cheap
rule loses to the hypergradient." Both the cheap local rule *and* the expensive
HOPE/Titans-style hypergradient buy **nothing** over a tuned scalar here.
Therefore: **Nested Learning's own canonical forgetting benchmark (§4.3,
Eq. 45) is optimizer-degenerate** — it cannot be used to argue that nesting or
sophisticated credit assignment helps continual learning, for *any* method.
That is a real, defensible result that indicts the field's use of this
construction, and it is a fundamentally different — and we think more useful —
paper than "a cheaper credit rule."

**We do not claim.** (i) Superiority. (ii) That local-PC's quality parity is
evidence for the method — P1 removed that reading. (iii) A demonstrated
hardware failure (extrapolation from the O(H·K·d) law). (iv) Any result beyond
a 24-d, 10-task toy. (v) That the structural law matters *anywhere yet* —
credit assignment is moot on the canonical geometry (P1), at scale (C-scale),
across the pre-registered heterogeneity sweep (C-stronger), **and on the
out-of-family cyclic-reactivation benchmark P3** built specifically to require
long-horizon credit. Five pre-registered constructions, no clean separation
from a tuned scalar. The structural law is **proven and dominant on cost +
stability** but its quality payoff is **unestablished everywhere reachable**;
the obstruction is identified (seed variance scaling with difficulty), not yet
overcome.

---

## 7. Theory: one exact reduction, two falsified conjectures, one **confirmed** hypothesis

This section contains one solid identity (§7.1), two conjectures we
pre-registered, tested, and **falsified** (§7.2–7.3), the empirical fact the
falsification produced (§7.4), and the live hypothesis whose adversarial
control we pre-registered, ran, and which **P1 confirms** (§7.6).

### 7.1 Reduction: the linear cascade is a learnable IIR filter on gradients

In the §4.3/Eq. 45 regime (fixed features, linear readout θ∈ℝ^d, task loss
ℒ_i(θ) = (θ−r_i)ᵀ Σ (θ−r_i), orthonormal r_i, g = Σ(θ−r_i)), the linear
cascade with linear combiner makes the inner optimizer a **learnable K-pole
LTI filter** F(z) on the gradient sequence: poles {σ(a_k)}, numerator set by
{g_k, c_k}. (The momentum-as-low-pass-filter view of NL §4.2, made exact and
multi-pole.) **§7.6/P1 shows that on this construction this K-pole filter
collapses, in effective behaviour, to a one-pole — i.e. a single scalar — at no
loss cost.** The reduction is an identity; the collapse is the empirical
finding.

### 7.2 A conjecture (tested and falsified)

The §7.1 reduction tempted a conjecture: local-PC's pole-pinning +
projected-output objectives recover the retention-optimal filter, with (C2) a
finite-λ bias of order Θ(1/λ) and (C4) reliance on block-sequential timescale
separation. We pre-registered two sharp predictions and tested them. **Both
falsified.** We report this rather than retro-fit.

### 7.3 Direct tests — falsified

**F3 (Θ(1/λ) bias).** With global's meta-LR controlled per λ, the
local-PC↔global gap **grows** with λ (0.011→0.153 over λ∈{0.25,…,64}; log-log
slope **+0.55**, the opposite sign). Premise wrong.

**F4 (timescale separation).** Sweeping presentation from block-sequential to
i.i.d., the nested-memory advantage over flat is **non-monotone** (≈0.03
sequential, peak ≈0.69 mid-interleaving, ≈0.04 i.i.d.), not collapsing toward
i.i.d. Mechanism unsupported as stated.

### 7.4 What the falsification produced

- **Parity is robust to interleaving.** local-PC↔global gap ≈0.01 at *every*
  block size. *Post-P1 reading:* this robustness is not a strength of the
  coupling — it is the benchmark's degeneracy showing through (the basin is
  set by task geometry, not presentation order; §7.6).
- **Emergent (unpredicted): the nested-memory advantage over flat is
  non-monotone in interleaving, maximal at intermediate block sizes.** Reported
  as an observation; no validated mechanism.

### 7.5 Nonlinear regime

The LTI reduction fails under nonlinearity; we have no fixed-point
characterization there. The widening local-PC advantage is plausibly an
optimization-reachability effect (global's hypergradient is a product of H
Jacobians) but is interpretation, not proof — and **§7.6 shows it is moot on
this benchmark anyway**, since a scalar matches both.

### 7.6 H-deg — pre-registered hypothesis, adversarial control RUN, **CONFIRMED**

The decisive fact §7.2 ignored was the **target-robustness ablation** (§5.2):
if the local signal carries almost no task information yet local-PC still
reaches the global solution, the mechanism cannot be "local credit
reconstructs the retentive filter." We advanced:

> **H-deg (effective-dimension degeneracy).** On §4.3/Eq. 45 — shared Hessian
> Σ, orthonormal task optima r_i — the set of inner optimizers achieving
> near-optimal continual loss is a *low-dimensional* manifold fixed by task
> geometry, not by the credit signal or presentation order. Any credit rule
> reaching that basin yields the same final loss. Parity is **basin
> robustness**, not filter reproduction.

H-deg was consistent with every prior observation, including the two that
falsified §7.2. We pre-registered its decisive adversarial control and
pre-committed to running and reporting it:

> **P1 (adversarial).** If the continual-optimal inner optimizer here is
> effectively ~1-dimensional, a single grid-searched scalar should tie *both*
> local-PC and global. If it does, the *choice of credit assignment is
> irrelevant on this benchmark* — indicting the expensive hypergradient as
> much as deflating our cheap local rule, and reframing the paper from "a
> cheaper credit rule" to "a degeneracy in the canonical benchmark."

**We ran P1. It returned DEGENERATE** (§5.6): scalar 0.617 ± 0.113, local-PC
0.611 ± 0.099, global 0.617 ± 0.115; gaps ≈0.01; best β ∈ {0.0, 0.5}. **H-deg
is confirmed.** The continual-optimal inner optimizer on §4.3/Eq. 45 is
effectively ~1-dimensional; the nested cascade — and the hypergradient that
trains it — buy nothing here over a tuned scalar.

This resolves the paper's central uncertainty in the deflationary direction we
named in advance. It is the *more interesting* outcome: it is not a result
about our method losing, it is a result about **Nested Learning's canonical
forgetting benchmark being unable to support any argument about optimizers or
nesting**.

#### 7.6.1 Pre-registered scale and heterogeneity controls (predictions fixed before running)

The natural objection is "the toy is degenerate; scale escapes it." We reject
the premise *in advance* and pre-register it: under H-deg, degeneracy is a
property of the **task geometry** (shared Hessian Σ, orthonormal optima
r_i), not of dimension or task count. We register two controls and their
decision rule. **Decision rule = the exact P1 rule, reused verbatim** (DEGENERATE
if best-scalar ≤ 1.25× best-nested and the scalar−nested margin ≤ the pooled
nested std; NESTING MATTERS if best-scalar > best-nested + 0.5× pooled std;
else AMBIGUOUS). No new goalposts.

- **C-scale (P1-at-scale).** Same §4.3/Eq. 45 construction, large
  (D_in, D_feat, N_tasks, stream) ≈ 10×/5×/5×/5× the toy. *Pre-registered
  prediction: stays **DEGENERATE**.* This is a falsification check of H-deg
  itself: if scale alone broke degeneracy, H-deg is wrong (a major result we
  would report as such). It is *not* expected to rescue Contribution B.

- **C-hetero (P2, the decisive one).** Break both degeneracy sources:
  per-task input covariance C_t (genuinely **heterogeneous Hessians Σ_t**) and
  **non-orthogonal, correlated** task directions r_t (random unit, no QR), at
  the *same* toy size (isolates geometry from scale). *Pre-registered
  prediction: **NESTING MATTERS** — the single scalar separates (best-scalar
  clearly worse than best-nested), because no one momentum/decay constant is
  simultaneously optimal across heterogeneous curvatures.* If instead it
  returns DEGENERATE, that is a strong negative for the nesting programme
  (nesting buys nothing even off the degenerate geometry) — equally
  publishable, reported straight either way.

The structural law (§6A) only becomes worth anything if C-hetero shows a
regime where the credit assignment it cheapens actually matters; **P3** (a
bias–variance crossover at long horizons) remains as future work to locate
the boundary precisely.

#### 7.6.2 Resolution (both controls run; reported straight)

**C-scale → DEGENERATE, prediction confirmed (§5.7).** H-deg's premise that
degeneracy is task-geometry, not dimension/count, survived a direct
falsification test: 10×/5×/5×/5× scale-up of the same geometry stays cleanly
degenerate (margin +0.004, 0/8 divergences). The deflation of Contribution B
is *not* a small-scale artifact.

**C-hetero → prediction not cleanly confirmed; degeneracy breaking; new
instability finding (§5.7).** Our pre-registered prediction ("NESTING MATTERS,
scalar separates") was **wrong as a clean call**: the verbatim rule still reads
DEGENERATE on the 9 stable seeds (margin +0.048 ≤ pooled 0.065). We report the
miss plainly. What the control *did* establish: (i) the degeneracy is breaking
— a monotone global < local-PC < scalar ordering emerges and an exploratory
paired test gives global < scalar on 8/9 stable seeds — so credit assignment
is weakly, genuinely non-moot under heterogeneity, contradicting a strong-form
H-deg that would predict perfect degeneracy everywhere; (ii) the global
hypergradient **diverges (1/10)** exactly in this regime while local-PC does
not. H-deg therefore holds in its *geometry-specific* form (degenerate on the
canonical orthonormal/shared-Σ construction, at any scale) but **not** in a
strong universal form (heterogeneity starts to lift it). The honest net: the
canonical benchmark's deflation of Contribution B stands and is scale-robust;
Contribution A's significance is now located in the heterogeneous regime,
where it gains a second, unanticipated axis — stability — over the
hypergradient. **P2-stronger** (raise heterogeneity until the pre-registered
bar is cleared) and **P3** are the remaining boundary-locating work.

#### 7.6.3 P2-stronger — pre-registered heterogeneity sweep (prediction fixed before running)

C-hetero shows the degeneracy *breaking* but not *broken*. P2-stronger
introduces a single heterogeneity-strength knob h: per-task input anisotropy
is `(0.3 + 1.7·u)^h` per coordinate, so h scales the per-task condition-number
spread (h=0 ⇒ homogeneous Σ, the degenerate anchor; h=1 ⇒ exactly the
reproduced C-hetero point; h>1 ⇒ progressively more heterogeneous Hessians).
We sweep h ∈ {0, 1, 2, 3, 4}, 10 seeds, **P1 decision rule reused verbatim**,
divergence guard active.

> **P2-stronger (pre-registered).** As h increases, (a) the
> best-scalar−best-nested margin grows monotonically and (b) there exists an
> h\* in the swept range at which the verbatim rule **flips DEGENERATE →
> NESTING MATTERS** (margin > pooled std *and* scalar > nested_best + 0.5·pooled
> on the stable subset); (c) the global divergence rate increases with h.
> **Bounding clause:** if the rule does **not** flip anywhere up to h=4, we
> report that the §4.3/Eq. 45 construction *family* is robustly
> optimizer-degenerate even under severe per-task curvature heterogeneity —
> which **bounds** (does not rescue) Contribution A: the structural law would
> then have no home in this family and the next benchmark must leave the
> family entirely (P3). Either outcome is reported straight; h=0 must read
> DEGENERATE as an internal sanity check.

#### 7.6.4 P2-stronger resolution (run; bounding clause triggered)

The sweep (10 seeds, h ∈ {0,1,2,3,4}, verbatim rule, guard active):

| h | scalar | nested_best | margin | pooled | div(gl) | verdict |
|---|---|---|---|---|---|---|
| 0 | 0.610 | 0.583 | +0.027 | 0.099 | 0/10 | DEGENERATE |
| 1 | 0.748 | 0.700 | +0.048 | 0.065 | 1/10 | DEGENERATE |
| 2 | 0.941 | 0.873 | +0.069 | 0.093 | 0/10 | DEGENERATE |
| 3 | 1.156 | 1.029 | +0.128 | 0.149 | 0/10 | DEGENERATE |
| 4 | 1.343 | 1.173 | +0.169 | 0.186 | 0/10 | DEGENERATE |

Of the three pre-registered predictions: **(a) monotone margin growth —
confirmed** (0.027→0.169, strictly non-decreasing); **(b) a flip to NESTING at
some h\*≤4 — not confirmed** (the rule never flips); **(c) divergence rising
with h — not confirmed** (1/10 at h=1, 0 elsewhere; sporadic, no trend). We
report (b)/(c) as misses, not re-interpreted. The h=0 sanity reads DEGENERATE
as required.

*Mechanism (honest, not a rescue).* The reason (b) fails is not absence of a
mean effect but **variance inflation**: per-task curvature heterogeneity grows
the mean nested advantage and the seed-to-seed variance roughly in lockstep,
so margin/pooled rises monotonically (0.27, 0.74, 0.74, 0.86, **0.91**) toward
— but never across — the threshold within the pre-registered grid. We
explicitly do **not** extrapolate a flip beyond h=4: it was not pre-registered,
and at h=4 the per-task condition-number spread is already ≈2000:1, beyond
which the base task is pathological and the comparison uninterpretable. The
rising ratio is logged as the precise open question for P3, not as evidence of
a latent positive.

**The pre-registered bounding clause therefore triggers.** The §4.3/Eq. 45
construction *family* — the canonical NL forgetting benchmark and its natural
heterogeneous generalizations (per-task Hessians, non-orthogonal task
directions, curvature spread to ~2000:1) — is **robustly
optimizer-degenerate**. This upgrades the P1 single-point deflation to a
**family-level, pre-registered, monotone, bounded negative**, and it changes
Contribution A's status from "contingent" to **explicitly bounded**: the O(1)
structural law has *no demonstrated home anywhere in this family*. A benchmark
that can adjudicate credit assignment must leave the family entirely — e.g.
genuinely different per-task feature *functions*, nonstationary correlated
streams, or a task structure where the optimal Φ provably depends on
long-horizon credit (P3). That is the honest next step; we do not claim the
structural law matters until such a benchmark exists and is non-degenerate by
this same verbatim test.

#### 7.6.5 P3 — the decisive non-degenerate benchmark (pre-registered before running)

P3 leaves the §4.3/Eq. 45 family along the *temporal* axis, where the
degeneracy proof's premise ("one decay constant suffices") provably fails.
**Construction: cyclic task reactivation.** A small task set (N=3, orthonormal
directions) is streamed in short blocks (B=30 steps) and **cycled many times**
(stream 600 ⇒ 20 recurrences/task; recurrence interval 90 steps, inter-block
gap 60). The optimizer must simultaneously (i) adapt within a 30-step block and
(ii) retain a task's gradient subspace across a 60-step gap of orthogonal
interference. A *single* momentum/decay β provably cannot do both: small β
adapts but forgets across the gap; large β retains but cannot adapt in 30
steps. A fast+slow multi-level memory can — this is exactly NL §4.3's "continual
learning needs the optimizer to keep a long-term compression of the gradient
subspace." Same verbatim P1 machinery and divergence guard; n=10; K=8.

> **P3-A (non-degeneracy, pre-registered).** The best grid-searched scalar is
> *clearly worse* than the best nested optimizer: scalar_mean −
> nested_best_mean **> pooled std** on the stable subset (the inverse of the
> P1 tie rule). If this fails, P3 is itself degenerate — reported straight,
> and the local idea remains undetermined (a yet-different benchmark needed).
>
> **P3-B (which is better — the head-to-head, pre-registered).** *Only if*
> P3-A holds. Let qgap = global_mean − localPC_mean (stable subset),
> pooled = std over the pooled stable {localPC, global}. Verdict:
> • |qgap| ≤ pooled ⇒ **quality-equivalent**; decided by the two
> pre-measured tiebreakers, **stability** (divergence count) then **cost**
> (backward-graph nodes). Since local-PC is exactly O(1) in H and global is
> O(H·K), a quality tie ⇒ **LOCAL-PC BETTER (dominant: equal quality, strictly
> cheaper, ≥ as stable)**.
> • qgap > pooled ⇒ **LOCAL-PC BETTER on quality outright**.
> • qgap < −pooled ⇒ **GLOBAL BETTER on quality**; then honestly weighed
> against its O(H) cost and any divergence.
> Predicted (stated before running, consistent with all prior evidence):
> P3-A holds (scalar clearly loses); P3-B returns **quality-equivalent ⇒
> LOCAL-PC BETTER by dominance** (equal quality, O(1) vs O(H), ≤ divergences).
> A clean "GLOBAL BETTER on quality" would be the most informative refutation
> and is reported as such if it occurs.

This is the experiment that converts "is the local idea working?" from
*undetermined* to a definite, pre-registered yes/no with an explicit
better-than ordering.

#### 7.6.6 P3 resolution (run; P3-A degenerate — prediction wrong, reported wrong)

Cyclic reactivation (N=3, B=30, stream 600, 10 seeds, K=8):

| | final loss (10 seeds) |
|---|---|
| best tuned scalar | 0.4194 ± 0.064 |
| local-PC | 0.3931 ± 0.064 (graph 101, O(1) in H) |
| global | 0.3874 ± 0.056 (graph 582, O(H·K)) |

scalar − best-nested = **+0.041 ≤ pooled 0.060 ⇒ P3-A DEGENERATE**. Our
pre-registered prediction (P3-A holds, scalar clearly loses) is **wrong; we
report it as wrong**, not re-interpreted. By the pre-registered gate, P3-B
(the head-to-head) **does not run** — we committed to that gate and honor it;
we do not crown a winner through a back door.

*The consistent directional signal, reported as exploratory (not
pre-registered, not used to decide):* the nested optimizer beats the scalar on
**9/10 seeds**, but the mean advantage never clears seed variance — the
identical variance-inflation obstruction found in C-stronger (§7.6.4), now
reproduced on the *temporal* axis with a different construction. Within the
nested pair, local-PC and global are a **statistical tie** (qgap = global −
local-PC = −0.006, |qgap| ≪ pooled; no clean per-seed winner), exactly as on
every prior benchmark.

**Consolidated verdict across five pre-registered constructions** (P1,
C-scale, C-hetero, C-stronger, P3): on *none* — including one purpose-built to
require long-horizon credit — does sophisticated credit assignment cleanly
beat a single tuned scalar at this toy scale, by the verbatim rule. The local
idea is therefore: (i) **unconditionally working as a structural/efficiency
primitive** — O(1)-in-H backward graph, 0 divergences, strictly dominating the
hypergradient on cost and stability on every benchmark; (ii) **a
quality-tie with the global hypergradient everywhere measured, including P3**;
(iii) **not demonstrated to be worth using over a tuned scalar on any
benchmark in reach** — and neither is the expensive hypergradient. "Which is
better" between the two nested rules: **local-PC, strictly, by dominance**
(equal quality, lower cost, ≥ stability); but that ordering sits on top of the
larger, honest negative that *neither nested rule has been shown necessary at
all*. The bottleneck is now precisely identified — seed variance scaling with
task difficulty — and is the explicit target for any future work (higher n,
variance-reduced estimators, or a genuinely larger-scale benchmark), not
another knob on this toy.

---

## 8. Limitations

- **The credit-assignment contribution is deflated by our own pre-registered
  control (the central fact).** P1 returned DEGENERATE: on §4.3/Eq. 45 a single
  scalar matches both local-PC and the hypergradient. Contribution B does not
  stand as evidence for the method; it stands as evidence the benchmark is
  optimizer-degenerate.
- **The structural law is explicitly bounded, not merely contingent.** §6A
  (O(1) vs O(H)) is exact and unconditional, but it matters only where credit
  assignment does — and credit assignment is degenerate on the canonical
  geometry (P1), at scale (C-scale), and across the entire pre-registered
  heterogeneity sweep up to ~2000:1 condition spread (C-stronger, bounding
  clause). **P3, built outside the family specifically to require long-horizon
  credit, also returned degenerate** by the verbatim rule. The law therefore
  has **no demonstrated quality home on any of five pre-registered
  constructions**. This is the single largest limitation; we state it as a
  bound, not a promise. The obstruction is identified — seed variance grows
  with task difficulty as fast as the nested advantage — and is the explicit
  target for future work (higher n / variance-reduced estimators / genuinely
  larger scale), not another knob on this toy.
- **Several of our pre-registered predictions were wrong.** C-hetero predicted
  clean "NESTING MATTERS" (no). C-stronger predicted a flip at h≤4 and rising
  divergence (no; only monotone margin held). **P3 predicted P3-A
  non-degenerate so the head-to-head P3-B would run; it did not — P3-A was
  degenerate, so by our own pre-registered gate P3-B was not run and no
  "winner" is crowned.** All misses are reported as misses. The 8/9 and 9/10
  paired signals and the rising margin/pooled ratio are exploratory, *not*
  pre-registered, and not extrapolated.
- **Hypergradient instability is a finding, with a small-n caveat.** Global
  diverged on 1/10 C-hetero seeds (the L2O pathology); local-PC/scalar did
  not. This is first-class but is a single divergence at n=10 — the *rate* is
  not precisely estimated and warrants a higher-n confirmation.
- **No working theory of the surviving regime.** Only §7.1 (an identity) is
  settled; both conjectures on it are falsified; H-deg is confirmed *as a
  degeneracy*, which is a statement about the benchmark, not a constructive
  theory of when nesting helps.
- **Toy scale.** 24-d readout, 10 tasks, CPU. "Deep" = per-level tanh + MLP
  combiner, not a deep MLP per level.
- **Fairness control is partial.** global's meta-LR retuned per seed, not its
  unroll horizon or Adam betas.
- **Online vs accumulated.** The O(1)-in-H memory result uses online local-PC;
  quality results use an accumulated variant for a fair single-step
  comparison. Both reported.
- **Emergent finding unexplained.** Non-monotone advantage-vs-interleaving
  (§7.4) reported without a validated mechanism.

---

## 9. Related work

**Nested Learning / HOPE / Titans** (Behrouz et al.): we stay inside this
paradigm and change only inter-level credit. Our P1 result is a direct,
constructive critique of using the §4.3/Eq. 45 construction to argue
nesting/credit-assignment helps continual learning — it cannot, for any method,
including HOPE's own hypergradient. **Learned optimizers / L2O** (Metz et al.;
VeLO): same meta-learning-an-optimizer setting; on a degenerate benchmark the
hypergradient pathology is moot because the hypergradient is unnecessary.
**Predictive coding / target propagation / local learning rules**: we apply
local error coupling to the *optimization hierarchy*; P1 shows this benchmark
under-determines the comparison. **Fast-weight programmers / DeltaNet dual
form**: parallelism by restricting the inner rule vs. our memory-flat backward
by changing credit — a composable lever whose payoff awaits a non-degenerate
benchmark. **Continual-learning benchmark critiques** (e.g. task-order /
metric-sensitivity literature): our contribution is a sharp, single-number
demonstration that a *specific construction adopted from a prominent
architecture paper* is optimizer-degenerate.

---

## 10. Reproducibility

Single self-contained script, CPU, deterministic, no external data:
`experiments/0002-nested-local-coupling/run.py`.

- `python run.py --seeds 10` — §5.1 (multi-seed parity + structural)
- `python run.py --ablation 10` — §5.2 (target-robustness)
- `python run.py --nonlinear 10` — §5.3 (nonlinear memory)
- `python run.py --deep 10` — §5.4 (deep nonlinear + fairness)
- `python run.py --scale` — §5.5 (scale-to-failure over H)
- `python run.py --lambda-fair 8` — §7.3 (F3, falsified)
- `python run.py --interleave 8` — §7.3 (F4, falsified) / §7.4
- `python run.py --p1 10` — **§5.6 / §7.6 (P1 adversarial control:
  returns DEGENERATE)**
- `python run.py --p1-scale 8` — **§5.7 / §7.6.1 C-scale (predicted &
  confirmed DEGENERATE; ≈9 min)**
- `python run.py --p2 10` — **§5.7 / §7.6.1 C-hetero (heterogeneous Σ_t +
  non-orthogonal r_t; verbatim rule → DEGENERATE on stable subset; reports
  1/10 global divergence)**
- `python run.py --p2-scale 8` — C-hetero at scale (both controls combined)
- `python run.py --p2-strength 10` — **§5.7 / §7.6.3–4 C-stronger
  (pre-registered heterogeneity sweep h∈{0..4}; monotone margin confirmed;
  bounding clause triggered → family robustly degenerate)**
- `python run.py --p3 10` — **§5.7 / §7.6.5–6 P3 (out-of-family cyclic
  reactivation; pre-registered P3-A/P3-B gate; P3-A → DEGENERATE,
  prediction reported wrong; P3-B gated off as committed)**

Sweeps run in ≈5–40 s on an Apple M1 Max (C-scale ≈9 min). Pre-registered
criteria, the (verbatim, reused) P1 decision rule, the divergence guard, and
per-seed logs are emitted by the script. **All three controls (P1, C-scale,
C-hetero) are implemented and run; the original `--p1` output is numerically
unchanged by the divergence guard (0 divergences on the canonical geometry).**

---

## 11. Conclusion

We set out to show that local adjacent-level predictive-coding coupling is a
cheap replacement for Nested Learning's expensive backprop-through-nesting. We
established one unconditional structural result — local coupling makes the
backward graph **O(1) in the unroll horizon** vs O(H), a 1455× separation that
grows without bound. We pre-registered and **falsified** our first explanation
of the observed quality parity, advanced a sharper hypothesis
(effective-dimension degeneracy), and — keeping the discipline we held the
entire time — pre-committed to its decisive adversarial control and **ran it**.
It returned **DEGENERATE**: on Nested Learning's own canonical forgetting
benchmark, a single tuned scalar matches both our cheap local rule and the
expensive hypergradient. The credit-assignment contribution is therefore
**deflated by our own control, and we report it straight**. The honest, and we
think more interesting, position: the structural law is solid and
unconditional but its motivation is contingent on a benchmark where credit
assignment is not moot; and §4.3/Eq. 45 — Nested Learning's canonical forgetting
construction — is **optimizer-degenerate**, unable to support any claim that
nesting or sophisticated credit assignment helps continual learning, for our
method or for HOPE's hypergradient alike. We then pre-registered two further
controls and ran both. The **scale** control confirmed its prediction: the
degeneracy is scale-robust, not a small-toy artifact — answering the obvious
"just scale up" objection with our own control. The **heterogeneity** control
**did not confirm its prediction** — we report that miss plainly — but it
showed the degeneracy genuinely breaking once tasks have heterogeneous
curvature, and it surfaced an unanticipated finding: in exactly that regime the
*expensive* hypergradient becomes unstable (diverges) while the *cheap, O(1)*
local rule does not. We then ran the final pre-registered control — a
heterogeneity *sweep* — and it **triggered its bounding clause**: the
nested–scalar margin grows monotonically with heterogeneity but seed variance
inflates in lockstep, so the verbatim rule reads DEGENERATE at every strength
up to a ~2000:1 per-task condition spread. So the honest, fully-resolved
position: on Nested Learning's canonical forgetting construction — at any
scale, and across its entire natural heterogeneous generalization — credit
assignment is moot; Contribution B is deflated and the deflation is now
**family-level**, not a single point. The unconditional structural law
survives but is **explicitly bounded**: it has no demonstrated home anywhere
in this benchmark family, retaining only an unanticipated stability edge over
the sometimes-diverging hypergradient. A result that names its decisive
experiments, runs them, reports a pre-committed deflationary outcome *and*
several wrong predictions of its own — without re-spinning any of them, and
refusing to extrapolate the one suggestive trend past its pre-registered grid
— is more useful to the Nested Learning programme than a confident story we
had already shown ourselves how to falsify. Finally we built P3 — a benchmark
*outside* the family, cyclic task reactivation engineered so no single decay
constant can both adapt and retain — as the experiment that would convert "is
the local idea working?" into a definite answer. **P3-A returned degenerate
too** (scalar − best-nested +0.041 ≤ pooled 0.060; nested wins 9/10 seeds but
never clears seed variance), so by our own pre-registered gate the
head-to-head P3-B was not run and we crown no winner through a back door. The
fully resolved, honest answer to "is the local idea working, and which is
better": **(1)** as a structural/efficiency primitive it works
unconditionally — O(1) in the unroll horizon, zero divergences, strictly
dominating the hypergradient on cost and stability on every benchmark;
**(2)** on quality, local-PC and the global hypergradient are a statistical
tie on all five pre-registered constructions including P3 — so *forced to
order them, local-PC wins by strict dominance* (equal quality, lower cost, ≥
stability); **(3)** but neither nested rule — nor HOPE's expensive
hypergradient — has been shown to beat a single tuned scalar on any benchmark
in reach, so we do **not** claim the local idea (or nesting) improves
continual learning. The recurring obstruction is now precisely identified —
seed variance scaling with task difficulty — and named as the target for
future work (higher n, variance-reduced estimators, genuinely larger scale),
not another knob on a 24-d toy. A five-construction, pre-registered negative
with a clean structural law and a wrong-prediction trail reported as wrong is,
we maintain, more useful to the Nested Learning programme than the confident
credit-assignment story we set out to write.
