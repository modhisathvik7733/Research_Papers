# 0008 — Gated consolidation: the smallest CMS component fix for global rigidity

**Status:** PRE-REGISTERED + runnable first cut. Registration fixed here
before results. Idea:
[ideas/nl-gated-consolidation.md](../../ideas/nl-gated-consolidation.md).

## The change (one component, parameter-free)

Vanilla CMS / multi-timescale optimizer: `e_k ← β_k e_k + (1−β_k) g` every
step, `Δ = Σ e_k/K` — globally rigid (exp-0004: class-IL ≈ chance; anti-stacks
with replay). Gated CMS: slow levels (k≥1) use agreement
`γ_k = relu(cos(g, e_k))` — `e_k ← β_k e_k + γ_k(1−β_k) g`,
`Δ = e₀/K + Σ_{k≥1} γ_k e_k/K`; fast level k=0 unchanged (full plasticity).
No learned params, no meta-objective, O(1)-in-H preserved.

## Decisive regime

exp-0004 **class-IL Split-MNIST** — the protocol where vanilla CMS provably
fails and anti-stacks with replay. Single-shared 10-way head, no task ID at
test. Field-standard Chaudhry ACC + Forgetting, paired common-seed.

## Arms

`naive` (Adam) · `replay` (Adam+rehearsal, the bar) · `rpc` (replay+vanilla
CMS = the exp-0004 failure, sanity) · **`rgc`** (replay+Gated CMS, the fix) ·
`van` (vanilla CMS alone) · `gat` (Gated CMS alone). CMS arms keep a
pre-registered LR fairness grid {3e-3, 1e-2} (bounded for runtime; the
exp-0004 best-of-grid control).

## Pre-registered predictions & decision rule

- **Sanity (must hold first):** `replay − rpc` ACC worse on ≥8/10 →
  anti-stacking reproduced. Else baseline broken; stop, report straight.
- **G-1 (fix removes conflict):** `rgc` **not worse than `replay`** on
  *both* Forgetting and ACC — verbatim exp-0002 three-way: SEP-favourable
  OR NOT-separated with mean ≥ 0.
- **G-2 (plasticity restored):** `gat` alone ACC − `van` alone ACC is
  SEP-positive (vanilla ≈ chance 0.22; gated learns).
- **Honest pre-stated prior:** G-1 failing ≥ as likely as succeeding;
  a clean G-1-no = "optimizer-side consolidation can't match rehearsal even
  gated" (hardened law). Testing a fix, not promoting it.
- Decision rule: verbatim exp-0002 paired three-way + effect-size gate
  (`m > s_pair` AND sign ≥ ⌈0.8n⌉) + practical δ = 0.02; divergence guard;
  exact sign-test p as corroboration; n=60 if any headline AMBIGUOUS.
- O(1)-in-H must hold (graph-node check) for any positive to count.

## Escalation

1. `--smoke` pipeline + sanity (rpc must anti-stack).
2. `--seeds 10` class-IL — sanity → G-1 → G-2.
3. n=60 confirmation on any AMBIGUOUS headline.
4. (only if G-1/G-2 positive) richer agreement gates as a separate idea.

## Results (class-IL, n=10, RESOLVED 2026-05-17) — clean double negative

```
 naive  ACC 0.195 Forget 0.992
 replay ACC 0.895 Forget 0.110   (the bar)
 rpc    ACC 0.758 Forget 0.218   replay+vanilla CMS (anti-stacks; SANITY ✓)
 rgc    ACC 0.347 Forget 0.579   replay+Gated CMS  (THE FIX)
 van    ACC 0.189 Forget 0.971   vanilla alone
 gat    ACC 0.177 Forget 0.471   gated alone
 SANITY  replay−rpc ACC m+0.136 10/10 p=.002 SEP  (anti-stacking reproduced)
 G-1     replay−rgc Forget m-0.469 0/10 ; rgc−replay ACC m-0.548 0/10  -> FAIL
 G-2     gat−van ACC m-0.012 3/10 NOT                                  -> FAIL
```

**Verdict: gated consolidation FAILS both pre-registered tests, decisively,
every seed.** Not ambiguous, not partial — large negative effects.

- **G-1 fails hard.** `rgc` is catastrophically worse than replay on both
  axes (ACC −0.55, Forget +0.47, 0/10) AND worse than *vanilla* (`rpc`
  0.758 → `rgc` 0.347): the gate deepened the conflict, did not fix it.
  Pre-registered conclusion (the honest prior fired): **optimizer-side
  consolidation cannot match or productively combine with rehearsal even
  when gated** — a hardened, mechanism-backed negative across exp-0004+0008.
- **G-2 fails AND falsifies our own exp-0004 root-cause story.** gat≈van
  (≈chance). If removing rigidity does not restore plasticity, **rigidity
  is not the (sole) cause** of vanilla CMS's class-IL failure. The deeper
  honest read: a fixed multi-timescale momentum is an inadequate optimiser
  for a real class-IL net regardless of gating (no adaptive
  preconditioning; the LR grid could not rescue it). Reported as a
  self-correction, not buried.
- **Mechanism hypothesis (labelled, untested):** gating BOTH consolidation
  and contribution by cos-agreement plausibly causes *update starvation*
  under replay (mixed new+replayed gradients rarely align with a slow EMA →
  updates broadly throttled → rgc worse than vanilla). Offered as the likely
  "why", not asserted.

**Net for the NL programme:** the smallest, most principled CMS component
fix to the measured failure does not work — and in failing, corrected our
mechanism model. Consolidated across exp-0002/0004/0008: the
"stack/consolidate timescales in the optimiser to resist forgetting" lever
is dominated by rehearsal and is likely the wrong lever once replay is
available. A clean, decisive, useful negative.

## Honest caveats baked in

Class-IL Split-MNIST scale. γ=relu(cos) is the simplest gate by design
(smallest change); a clean G-1-no here is already a strong negative. No
learned params (deliberate). Any G-1 positive = conflict-removed parity with
replay, not superiority — stated as such, never inflated.
