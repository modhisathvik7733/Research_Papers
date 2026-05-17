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

## Honest caveats baked in

Class-IL Split-MNIST scale. γ=relu(cos) is the simplest gate by design
(smallest change); a clean G-1-no here is already a strong negative. No
learned params (deliberate). Any G-1 positive = conflict-removed parity with
replay, not superiority — stated as such, never inflated.
