# Finding: Local Predictive-Coding Coupling Removes the Unroll-Depth Wall in Nested Learning

- **Date:** 2026-05-16
- **Source:** Nested Learning / HOPE paper notes ([papers/nested-learning-hope.md](../papers/nested-learning-hope.md)); critical analysis turns; experiment [0002](../experiments/0002-nested-local-coupling/)
- **Status:** RESOLVED (deflationary). Structural claim robust (100%,
  unconditional). Pre-registered control **P1 RUN → DEGENERATE**: a single
  tuned scalar matches both local-PC and the global hypergradient on §4.3/
  Eq. 45 → the credit-assignment contribution is deflated by our own control;
  the benchmark is optimizer-degenerate. Reframed paper: "NL's canonical
  forgetting benchmark is optimizer-degenerate" + the surviving structural law.
- **Intended use:** seed for a paper on a Nested Learning extension ("post-HOPE")
- **Draft manuscript:** [drafts/nested-local-coupling-paper.md](../drafts/nested-local-coupling-paper.md) (workshop length, written 2026-05-16)

> This file is the durable record of the contribution + evidence so it can be
> lifted into a paper draft. Numbers are from a single-seed toy; treat as
> existence-of-mechanism, not a benchmark claim.

---

## 1. One-sentence contribution

Nested Learning's unique scaling wall — every added level is another layer of
backprop-through-optimization, whose backward graph scales with the inner-loop
unroll length H × number of levels K — can be removed by crediting each level
with a **local adjacent-level prediction-error objective** instead of a global
hypergradient: this makes the backward graph **O(1) in H** (unconditional,
exact). *Resolved caveat:* on NL's own §4.3/Eq.45 benchmark this cheaper credit
buys no quality difference — but neither does the expensive hypergradient over
a single tuned scalar (P1 → DEGENERATE), so the headline finding is that
**that benchmark is optimizer-degenerate**, and the surviving positive claim is
the structural law alone (significance contingent on a non-degenerate
benchmark, P2/P3).

## 2. The gap in NL/HOPE this addresses

From the paper notes, three things NL's own logic says should be learned but
HOPE hardcodes, and one structural cost:

- Update **frequency / level count / inter-level transfer** are hand-set.
- HOPE's self-modification is depth-2 and its parallel form survives **only**
  for DeltaNet-class inner rules; the expressive nonlinear case has no scalable
  gradient.
- **Unique NL scaling bottleneck:** standard nets scale by width/depth (cheap,
  parallel); NL scales by *nesting depth*, which is sequential and
  backward-memory-heavy because differentiating through K stacked optimizers
  unrolled H steps costs O(H·K) graph.

Key observation (from the analysis turns): the coherence problem (nested local
objectives can fight — the §4.3 orthogonal-task forgetting *is* two NL levels
optimising incoherently), the escape-from-Transformer problem, and the scaling
wall **have the same fix** — replace global backprop-through-nesting with local
adjacent-level prediction-error coupling. That convergence is the thesis.

## 3. Proposed mechanism

A K-level memory cascade is the nested optimizer:
`m_k = σ(a_k) m_k + g_k m_{k-1}`, update `Δθ = Σ_k c_k m_k`, meta-params
`Φ = {a_k, g_k, c_k}`.

- **Global (baseline / HOPE-style credit):** train Φ by backprop through an
  H-step unrolled inner loop against a §4.3 retention meta-loss. Backward graph
  ∝ H·K.
- **Local-PC (proposed):** train Φ only by local errors, fully detached across
  levels and steps — each slower level predicts the slow-EMA (stable component)
  of the faster level; the combiner predicts the §4.3 retentive
  (anchor-orthogonalised) update target. Backward graph carries **no H factor**;
  grows at most ~linearly in K.

This is a predictive-coding hierarchy in the *optimization* domain: the
inter-level prediction error *is* the knowledge-transfer message, replacing NL's
four hand-picked transfer types with one learned local rule.

## 4. Falsifiable claim (as tested)

> **H:** Local-PC coupling (a) reaches final continual loss ≤ 1.25× the global
> nested optimizer and ≤ a flat momentum-SGD baseline, (b) reduces forgetting
> vs flat when real forgetting exists, and (c) its differentiated-loss autograd
> graph does not carry the ×H unroll multiplier (global nodes > 4× local at
> K=8, global graph grows ≥1.8× over K1→K8).

Pre-registered before the run that produced the reported numbers.

## 5. Evidence (toy, single seed, faithful to paper Eq. 45)

Testbed = paper Eq. 45 exactly: shared fixed random features, linear readout
`θ∈R²⁴`, 10 tasks each on an **orthonormal** direction, streamed 120 steps
each; momentum drags the shared readout across orthogonal directions ⇒ the
§4.3 catastrophic-forgetting mechanism (not generic drift). CPU, ~5 s, M1 Max.

**10 independent seeds** (`run.py --seeds 10`), K=8, mean ± std:

| method | final avg loss ↓ | forgetting ↓ | backward-graph nodes ↓ |
|---|---|---|---|
| flat momentum-SGD | 0.644 ± 0.114 | 0.716 ± 0.126 | — |
| global | 0.617 ± 0.114 | – | 582 (every seed) |
| **local-PC** | **0.609 ± 0.104** | **0.666 ± 0.116** | **101 (every seed)** |

Gate pass-rates over 10 seeds: **structural (no ×H factor) = 100%**,
quality = 80%, helps-vs-flat = 80%, all-three = 80%. The 2 failing seeds are
the *same* low-interference draws where flat momentum barely forgets — there
the nested memory is correctly ~neutral, not harmful.

Scaling (exact autograd-node count, not wall time): global 133→582 over
K1→K8, already 133 at K=1 because it carries the ×8 unroll factor; local-PC
16→101, **no unroll factor**, **5.8× smaller at K=8 deterministically across
all seeds**. Full script + per-seed log: experiment 0002.

**Headline for a paper (post-P1, resolved):** local prediction-error coupling
**deterministically removes** the unroll-depth (×H) term that is Nested
Learning's distinctive scaling cost (unconditional structural law). The
quality "parity" with the hypergradient on the orthogonal-task benchmark is
**deflated by the pre-registered control P1 (DEGENERATE)**: a single tuned
scalar matches both local-PC and the hypergradient, so §4.3/Eq.45 is
optimizer-degenerate and cannot adjudicate credit assignment for any method.
Do *not* claim "competitive with / better than global" as a contribution —
report it as deflated, and lead with the benchmark-degeneracy finding +
the surviving structural law.

## 6. Honest limitations / threats to validity (must appear in any paper)

1. **Toy & single seed.** 24-d, linear-in-features memory, 10 tasks. Mechanism
   existence only — *not* scale, *not* deep nonlinear memory. Multi-seed + CIs
   are a prerequisite for any claim.
2. **`local-PC > global` is not "local is better."** Truncated-unroll
   hypergradients optimise poorly at small scale (known L2O pathology). The
   defensible claim is *competitive and structurally cheaper*, not superior.
   Must show this is not just a global-baseline-weakness artifact.
3. **Forgetting reduction is modest (~11% rel.).** Consistent with the paper's
   own "forgetting is conserved under finite capacity." The mechanism helps at
   the margin; do not frame as solving forgetting.
4. **local-PC graph still grows ~linearly in K.** The surviving claim is
   precisely "no ×H unroll multiplier" — the actual NL wall — not "O(1) in K."
   Be exact about this in the writeup or a reviewer will (correctly) attack it.
5. ~~The local targets are hand-designed; need an ablation.~~ **RESOLVED
   (10-seed ablation, `run.py --ablation 10`).** De-tuning both targets
   (plain-grad combiner + random per-level targets) leaves quality
   statistically unchanged: orig 0.611 vs fully-de-tuned 0.621 (within ~0.10
   seed std), both 100% ≤1.25×global. ⇒ the contribution is the **coupling
   structure**, not target hand-tuning. *Nuance:* per-level targets are
   near-irrelevant in the linear toy (only the combiner matters, ~0.01) — the
   linear regime under-exercises deep nesting, motivating the nonlinear test.

## 7. Positioning vs related work (for the related-work section)

- vs **HOPE / Titans / Nested Learning** (Behrouz et al.): same paradigm; we
  remove the unroll-depth credit, not the architecture. We do not collapse to
  retrieval (deliberate — staying inside NL).
- vs **predictive coding / target propagation / local learning rules**: we
  apply local error coupling to the *optimization hierarchy*, not the
  forward-pass layer stack — a different axis.
- vs **learned optimizers (L2O, VeLO)**: same meta-learning-an-optimizer
  setting; our contribution is the *credit-assignment locality* that makes the
  nesting cheap, and an honest note that L2O hypergradient pathologies are why
  the global baseline underperforms.
- vs **fast-weight programmers / DeltaNet dual form**: those get parallelism by
  restricting the inner rule; we get cheap nesting by changing how the meta-rule
  is credited — orthogonal lever, composable with them.

## 8. Paper skeleton (if promoted)

1. NL recap + the unique scaling wall (O(H·K) backward through nesting).
2. Thesis: coherence + escape-Transformer + scaling have one fix = local
   adjacent-level predictive coupling.
3. Mechanism + formalization (the cascade + the two credit regimes).
4. Theory sketch: conditions under which local-PC fixed point ≈ hypergradient
   solution (steelman; this is the risky theoretical claim).
5. Experiments: Eq. 45 toy (done) → multi-seed → nonlinear MLP memory → scale
   until global is unroll-bound while local-PC stays flat → an M-scale task.
6. Honest limitations (§6 above verbatim).

## 9. Escalation chain (status)

1. **Multi-seed (10) — DONE.** Structural gate 100% (deterministic 5.8×),
   quality 80%, helps 80%. Quality refined to "competitive, not superior."
2. **Target-design ablation (10) — DONE, PASSED.** Fully de-tuned local targets
   ≡ original (0.621 vs 0.611, within std), 100% ≤1.25×global ⇒ the gain is
   the **coupling**, not target hand-tuning. (`run.py --ablation 10`)
3. **Nonlinear 2-layer-MLP memory (10) — DONE, STRONGEST.**
   (`run.py --nonlinear 10`) local-PC 0.615±0.106 vs global 0.634±0.116 vs
   flat 0.644±0.114; **quality 100%**, helps 90%, **structural 100%** (global
   624 vs local 107 nodes, 5.8×). Quality *improves* vs the linear case:
   backprop-through-nonlinear-nesting is unstable, so global degrades exactly
   where the memory is expressive — the thesis's whole point.
4. **Scale-to-failure over unroll horizon H — DONE, SPLIT RESULT.**
   (`run.py --scale`) Memory/graph law **decisively proven**: global backward
   graph grows *exactly linearly* in H (624@H8 → 155,664@H2048); online
   local-PC **flat at 107 nodes for all H** (1455× smaller @H2048, unbounded).
   BUT global did not hit a wall-clock/OOM wall at toy d (155k nodes is huge in
   *count*, cheap in *bytes* at d=24) and online local-PC paid a per-step
   Python tax (~2× slower wall-time at small d). Honest claim: **the O(1)-vs-
   O(H) backward-memory law is proven; the crash/timeout is an extrapolation
   to real d (memory is O(H·K·d)), not a demonstrated result.** Do not claim
   wall-clock superiority at small d; claim asymptotic memory + the law.
5. **Deep nonlinear recursive memory + global fairness retune — DONE,
   PASSED.** (`run.py --deep 10`) Per-level AND combiner nonlinear (no dual
   form anywhere); global gets best-of-5 meta-LR grid *per seed*, local-PC
   untuned. global* 0.618±0.098 vs local-PC 0.615±0.106 vs flat 0.644;
   quality 100%, structural 100% (680 vs 107). Tuning global closed most of
   the nonlinear gap (0.634→0.618) — confirming the fairness caveat — yet
   untuned local-PC still **ties** best-tuned global at a constant graph.
   The "global undertuned" critique is dead.
6. **Theory predictions tested — BOTH FALSIFIED (honest negative).**
   (`run.py --lambda-fair 8`, `--interleave 8`) F3: gap *grows* with λ
   (log-log slope +0.55, wrong sign) even with global LR controlled — the
   Θ(1/λ) premise (C2) is wrong. F4: advantage over flat is *non-monotone*
   (peaks at intermediate interleaving), does not collapse at i.i.d. — the
   timescale-separation mechanism is unsupported. **Strengthened positive:**
   local-PC↔global parity ≈0.01 across *all* interleaving regimes — the
   equivalence is robust and does NOT need the theory's crutch.
7. M-scale continual-stream test from the experimental program. (not started)

### Bottom line after the full chain (post P1 — RESOLVED)
The credit-assignment story is **deflated by our own pre-registered control**.
The *only* surviving positive claim is the **unconditional structural law**:
local adjacent-level predictive-coding coupling makes the backward graph
**O(1) in the unroll horizon** vs global's O(H) (1455× @H2048, unbounded) —
P1 does not touch this. The quality "parity" with the hypergradient is true
but **uninformative** (a single scalar matches both), so it is reported as
deflated, not as a contribution. Never frame as "matches/beats global"
without immediately stating a scalar also matches both. Honesty: toy scale;
"deep"=tanh+MLP combiner; fairness=meta-LR only; wall-clock crash
extrapolated; structural law's *significance* is contingent on a
non-degenerate benchmark (P2/P3).

### Live hypothesis RESOLVED: effective-dimension degeneracy CONFIRMED
v1 (Θ(1/λ) + timescale) falsified. Replacement **H-deg**: on §4.3/Eq.45 the
continual-optimal inner optimizer lies on a low-dim manifold fixed by task
geometry; any credit rule reaching that basin gives the same loss → parity =
basin-robustness, not filter reproduction. We pre-registered the decisive
adversarial control and **pre-committed to running + reporting it whichever
way it fell. We ran it (`run.py --p1 10`): DEGENERATE.** Linear, K=8, 10
seeds: scalar 0.617±0.113, local-PC 0.611±0.099, global 0.617±0.115 (gaps
≈0.01, pooled std ≈0.107; best-β mostly 0.0/0.5). **H-deg confirmed.** A single
scalar matches BOTH nested optimizers ⇒ §4.3/Eq.45 cannot distinguish
credit-assignment methods at all ⇒ Contribution B deflated **and HOPE's
hypergradient indicted equally** (it too buys nothing over a scalar here).
Not "we lose" — **NL's own canonical forgetting benchmark is
optimizer-degenerate**: a real, defensible, fundamentally different (and more
interesting) paper. **P2/P3 are now the critical path** — build the
non-degenerate benchmark (task heterogeneity / per-task Hessians; bias–variance
crossover at long horizons) where credit assignment is NOT moot, so the
surviving structural law has something to matter for. Resolved in draft §5.6
+ §6 + §7.6 + §11.

### Scale & heterogeneity controls RESOLVED (2026-05-17)
Both pre-registered (P1 rule reused verbatim) + a backward-compatible
divergence guard added to `p1` (orig `--p1` numerically unchanged).
- **C-scale (`--p1-scale 8`) — prediction CONFIRMED, clean DEGENERATE.** Same
  geometry 10×/5×/5×/5× larger: sc 0.751 / lpc 0.750 / gl 0.748, margin
  +0.004, 0/8 div. **Degeneracy is geometry, not size** — H-deg premise
  survives a falsification test; "just scale up" objection dead.
- **C-hetero (`--p2 10`) — prediction NOT cleanly confirmed (reported as a
  miss); degeneracy breaking + instability finding.** Verbatim rule still
  DEGENERATE on 9 stable seeds (sc 0.748 / lpc 0.729 / gl 0.704; margin +0.048
  ≤ pooled 0.065). But monotone ordering gl<lpc<sc emerges (absent on toy);
  exploratory paired gl<sc on 8/9; **global diverged 1/10** (L2O blow-up) while
  local-PC/scalar never did.
- **Synthesis:** canonical degeneracy is *scale-robust* and only *partially*
  lifted by strong heterogeneity (deep, not artifact). The regime where credit
  assignment starts to matter is exactly where the expensive hypergradient is
  unstable and the cheap O(1) local rule is not ⇒ Contribution A re-motivated
  on a **second axis (stability)**; Contribution B stays deflated.
  Critical path now: **P2-stronger** (raise heterogeneity until the
  pre-registered bar is cleanly cleared or bounded) + divergence-rate
  confirmation at higher n + P3.
- **C-stronger (`--p2-strength 10`, h∈{0..4}) — RESOLVED, bounding clause
  triggered.** Pre-registered (a) monotone margin, (b) flip at h\*≤4, (c) div↑h.
  **(a) confirmed; (b),(c) wrong, reported as wrong.** Margin grows
  0.027→0.169 monotone but pooled inflates in lockstep (margin/pooled
  0.27→0.91, never crosses 1); DEGENERATE at every h; h=0 sanity DEGENERATE.
  Not extrapolated past pre-registered grid (~2000:1 cond spread at h=4).
  ⇒ **§4.3/Eq.45 construction family is robustly optimizer-degenerate**
  (family-level, not a point). Contribution A: contingent → **explicitly
  bounded** (no demonstrated home in this family; only a stability edge over
  the sometimes-diverging hypergradient). **Critical path now: P3 — leave the
  family entirely**; non-degenerate by the same verbatim test is the
  precondition for any "structural law matters" claim.
- **P3 (`--p3 10`) out-of-family decisive benchmark — RESOLVED, P3-A
  DEGENERATE.** Cyclic task reactivation (N=3, B=30, stream 600; recur/90,
  gap 60), engineered so no single decay constant adapts AND retains.
  Pre-registered P3-A (scalar clearly loses) → **WRONG, reported wrong**:
  scalar 0.419 vs best-nested 0.387, margin +0.041 ≤ pooled 0.060; nested
  beats scalar 9/10 seeds but never clears seed variance (same obstruction as
  C-stronger, temporal axis). P3-B gated off as pre-committed. local-PC 0.393
  ≈ global 0.387 (tie, qgap −0.006), as everywhere.
- **Paired analysis (`--paired 10`, §7.6.7–8) — pre-registered NEW, correct
  variance, originals preserved. SUPERSEDES the pre-paired "five-construction
  negative".** Methods share per-seed task draw ⇒ paired design; verbatim rule
  used wrong (between-seed) variance. Result, cuts both ways:
  - **P-2 CONFIRMED:** nesting **significantly beats** the tuned scalar off
    the canonical geometry — C-hetero p=.039 (t4.25, 8/9), P3 p=.021 (t3.77,
    9/10). Real effect, was masked. The genuine "better result" (correct
    stats, not spin).
  - **P-1 WRONG for canonical:** AMBIGUOUS at n=10 (p=.109), NOT cleanly
    degenerate → strong "canonical is optimizer-degenerate" claim **withdrawn,
    downgraded to underdetermined**, reported as pre-committed. C-scale half
    untested.
  - **P-3 CONFIRMED w/ twist:** local-PC vs global = paired tie, raw lean
    mildly to GLOBAL ⇒ local-PC better **only by dominance** (equal quality,
    O(1) cost, global diverged 1/9 & local-PC never), NOT a quality win.
- **REVISED FINAL ANSWER.** (1) Structural primitive: works unconditionally.
  (2) Nesting **does** matter — off the canonical geometry, paired-significant.
  (3) local-PC = the better method by dominance (equal quality, cheaper,
  stabler), not by quality. (4) Canonical degeneracy: undetermined at n=10.
  Contribution = clean O(1) structural law + a genuine pre-registered
  off-degenerate-geometry positive for nesting + an honest retraction of the
  over-strong canonical claim. Next: **higher n** (resolve canonical, tighten
  significances) — cheap, decisive.

### Remaining honest caveats after the chain
- Still a toy (24-d readout, 10 tasks). "Nonlinear" = MLP **combiner/readout**;
  per-level recurrence is still linear — deep per-level nonlinear memory
  untested.
- Global's nonlinear-regime gap may be *partly* hypergradient optimisation
  difficulty (global's meta-LR not separately retuned for the harder
  landscape). That difficulty is itself the motivation, but the writeup must
  say "competitive-to-better AND cheaper" and report this tuning caveat.

## Score

- Novelty: 4 (specific, not obviously answered in NL literature)
- Testable cheaply: 5 (toy already runs in 5 s; escalations are ≤1 day)
- Informative if it fails: 5 (a clean negative kills the post-HOPE direction)
- **Total ≥ 11 and cheap experiment < 1 day → promoted.**
