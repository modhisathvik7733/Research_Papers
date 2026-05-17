# Finding (proposed): Selective (Gradient-Gated) Local-PC — Does Confining Retention to the Off-Task Subspace Stop It Fighting Replay?

- **Date:** 2026-05-17
- **Source:** Root-cause analysis of the exp-0004 stage-1c result (plain
  Local-PC *anti-stacks* with replay in class-IL: replay 0.110/0.895 →
  rpc 0.218/0.758, strictly worse on both axes, 0/10).
- **Status:** PROPOSED — pre-registered, not yet run. Skeleton:
  [experiments/0006-localpc-selective-gating/](../experiments/0006-localpc-selective-gating/).
- **Intended use:** test the *mechanism* behind the anti-stacking and a
  principled fix — informative either way (a clean failure hardens the
  negative into a mechanism-backed law).

## 1. One-sentence contribution

exp-0004 showed plain Local-PC degrades replay because its multi-timescale
momentum is **globally rigid** — it resists *all* weight change, including the
directions the new task (and replay) must move. We test a fix: apply the
slow-timescale retention **only in the orthogonal complement of the current
task's gradient subspace** (fast timescale free in-task), and ask whether this
removes the conflict (recovers ≥ replay on the decisive class-IL 2×2) while
preserving the O(1)-in-H property.

## 2. The mechanism hypothesis (what we're actually testing)

Anti-stacking cause (pre-registered): *non-selective* slow momentum freezes
prior-task directions **and** current-task directions indiscriminately; replay
needs the current-task directions free to relearn old classes. If the cause is
selectivity, gating retention off the current-task subspace should restore
non-harm. If selective gating *still* harms replay, the cause is deeper:
optimizer-side weight retention is the wrong lever whenever rehearsal is
present — a strong, mechanism-backed negative worth stating as a law.

## 3. Method

`SelectiveLocalPC`: keep the K-level multi-timescale momentum; estimate the
current task's top-r gradient directions online (running PCA / GPM-style
basis P_t); apply the **slow** levels' contribution through (I − P_tP_tᵀ)
(off-task only); the **fast** level applies unprojected (in-task plasticity
preserved). Per-step, no unroll → O(1)-in-H preserved (asserted/measured).

## 4. Falsifiable claim (pre-registration)

> **H-sel.** On the exp-0004 **class-IL 2×2** (decisive regime), arms
> `replay`, `rpc`(replay+plain-LocalPC), **`rspc`(replay+SelectiveLocalPC)**,
> paired CRN:
> (a) **Sanity:** plain `rpc` reproduces the established anti-stacking
> (`replay − rpc` worse on ACC, ≥8/10) — else the baseline is broken, stop.
> (b) **Fix works iff:** `rspc` is **not worse than `replay`** on *both*
> Forgetting and ACC (verbatim three-way: SEP-positive or NOT-separated with
> non-negative mean), i.e. selective gating removes the conflict.
> (c) **Fix fails iff:** `rspc` still worse than `replay` on either axis →
> conclusion: optimizer-side retention is wrong-lever-under-replay *even when
> selective* (hardened negative, reported as such).
> O(1)-in-H must be preserved (graph-node check) for any positive to count.
>
> Pre-registered honest prior: given exp-0002/0004, **(c) is at least as
> likely as (b)** — we are testing a mechanism, not promoting a method.

## 5. Honest limitations

Class-IL Split-MNIST scale; r (subspace rank) is a knob — a small
pre-registered grid {4,8,16}, best reported with the others shown (no
silent tuning). Selective gating adds O(r·d) per step — cheaper than replay's
buffer, must be stated. If (b) holds, it is parity-with-replay-at-lower-extra
not superiority — claimed as such.

## 6. Score

- Novelty: 4 (mechanism test + principled fix targeting a *measured* failure)
- Testable cheaply: 5 (exp-0004 class-IL 2×2 harness + one optimizer; ≤1 day)
- Informative if it fails: 5 (a clean (c) converts "anti-stacks" into a
  mechanism-backed law: optimizer-side retention is the wrong lever under
  rehearsal, selectivity notwithstanding)
- **Total = 14 → promote to experiment 0006.**
