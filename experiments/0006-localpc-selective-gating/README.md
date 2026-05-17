# 0006 — Selective (gradient-gated) Local-PC: does off-task retention stop it fighting replay?

**Status:** PRE-REGISTERED + runnable first cut. Tests the *mechanism* behind
the exp-0004 anti-stacking and a principled fix. Idea:
[ideas/localpc-selective-gating.md](../../ideas/localpc-selective-gating.md).

## Question

exp-0004 class-IL: replay+plain-LocalPC (`rpc`) is strictly worse than
`replay` on both Forgetting (0.110→0.218) and ACC (0.895→0.758), 0/10 —
*anti-stacking*. Hypothesis: cause = the slow momentum is **globally rigid**,
freezing current-task directions replay needs. Fix: confine retention to the
**orthogonal complement of the current-task gradient direction** (fast level
free in-task). Does that remove the conflict, at O(1)-in-H?

## Arms (class-IL Split-MNIST, the decisive regime)

`naive` · `replay` · `rpc` (replay + plain Local-PC) · **`rspc`** (replay +
SelectiveLocalPC). Local-PC arms keep the pre-registered LR grid; Selective
gating projects the *slow* levels off an EMA of the current-task gradient
(first cut r=1; r∈{4,8,16} true-subspace is the pre-registered escalation).

## Pre-registered predictions & decision rule

- **(a) Sanity (must pass first):** plain `rpc` reproduces anti-stacking —
  `replay − rpc` ACC worse on ≥8/10 seeds. Else baseline broken → stop.
- **(b) Fix works iff:** `rspc` is **not worse than `replay`** on *both*
  Forgetting and ACC — verbatim exp-0002 three-way: SEP-in-rspc-favour or
  NOT-separated with mean ≥ 0. Plus O(1)-in-H preserved (graph-node check).
- **(c) Fix fails iff:** `rspc` still worse than `replay` on either axis →
  conclusion (hardened, mechanism-backed): optimizer-side retention is the
  wrong lever under rehearsal *even when selective*. Reported as a law, not
  a disappointment.
- Pre-registered honest prior: **(c) ≥ as likely as (b)**; we test a
  mechanism, not promote a method. Paired CRN, divergence guard, exact
  sign-test p; n=60 if a headline comparison is AMBIGUOUS.

## Escalation

1. `--smoke` pipeline sanity (1 seed).
2. `--seeds 10` class-IL 2×2+rspc — (a) sanity then (b)/(c).
3. r∈{4,8,16} true-subspace grid if r=1 is (b)-promising or clearly (c).
4. n=60 confirmation on any AMBIGUOUS headline.

## Honest caveats baked in

Class-IL Split-MNIST scale. r=1 first cut is the weakest form of selectivity
(EMA direction, not a true subspace) — a clean (c) at r=1 is *suggestive*;
the r-grid escalation is required before declaring (c) a law. Selective gating
costs O(r·d)/step — cheaper than the replay buffer, must be stated; any (b)
is parity-at-low-extra, not superiority.
