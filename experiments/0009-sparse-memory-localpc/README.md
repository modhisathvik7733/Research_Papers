# 0009 — Sparse explicit memory (retention) + Local-PC (stability): do the two survivors coexist?

**Status:** PRE-REGISTERED + runnable first cut. Registration fixed here
before results. Idea:
[ideas/nl-sparse-memory-localpc.md](../../ideas/nl-sparse-memory-localpc.md).

## The claim

The two mechanisms that survived falsification — sparse explicit key-value
memory (retention; storage beat compression in exp-0002→0008 and SMF) and
Local-PC (O(1) deep-optimization stability; exp-0005) — act on *different
axes* and therefore should **coexist without the anti-stacking** that killed
vanilla CMS + every retention mechanism (exp-0004/0008).

## Setup

exp-0004 **class-IL Split-MNIST** (the regime vanilla CMS catastrophically
failed: ≈chance, anti-stacked with replay). Dense FFN → minimal **sparse
key-value memory layer**: M=4096 slots, top-k=8 cosine access, output =
softmax(top-k) · values; updates are sparse by construction (only accessed
slots get gradient — the SMF low-compute / low-interference property).

## Design-time clarification (logged before any run)

On a single real classifier, Local-PC's *deployable* form **is** the
gain-normalised multi-timescale-momentum optimiser (= exp-0008 "vanilla
CMS"); its distinctive O(1)-*credit* differs from the hypergradient only in
the *nested/unrolled* regime (exp-0005 owns that, S-3 does not re-claim it).
So the honest experimental variable here is **moving retention into an
explicit sparse-memory architecture** (vs replay/momentum) and asking whether
the NL-style optimiser then **coexists** with it — the direct contrast to
exp-0008, where that same optimiser **anti-stacked** with replay-based
retention. `mem-cms` is therefore dropped (identical object); arms simplified.

## Arms

`dense-naive` (dense+Adam, no replay — exp-0004 failure ref ≈0.99 forget) ·
`replay` (dense+Adam+replay — exp-0004 bar 0.110/0.895) · `mem-adam`
(sparse-memory+Adam, **no replay**) · `mem-localpc` (sparse-memory + the
deployable multi-timescale Local-PC optimiser, **no replay**). Local-PC arm
keeps the pre-registered LR fairness grid {3e-3, 1e-2}.

## Pre-registered predictions & decision rule

- **S-1 (memory → retention) — HARDENED (smoke-found, logged before the
  decisive run):** low Forget is *vacuous if ACC ≈ chance* (nothing learned
  to forget). S-1 requires **BOTH** `mem-adam` ACC > 0.40 (≫ 0.10 chance)
  **AND** Forget ≪ `dense-naive` (≈0.99). If `mem-adam` ≈ chance ⇒ S-1
  FAIL = *testbed-inadequate* (minimal KV layer too weak), reported as
  such — **not** a synthesis verdict. S-2 is **not interpretable** unless
  S-1 holds (non-conflict is vacuous if the model didn't learn).
- **S-2 (THE test — non-conflict):** `mem-localpc` **not worse than
  `mem-adam`** on *both* Forget and ACC (verbatim three-way: SEP-favourable
  OR NOT with mean ≥ 0). Honest prior: should HOLD (different axes). Clean
  fail ⇒ synthesis false (Local-PC conflicts with retention generally) —
  hardened negative, reported straight.
- **S-3 (scope honesty):** `mem-localpc` shows **no quality advantage** over
  `mem-adam` — Local-PC's value is the exp-0005 deep-unroll regime, NOT
  tested here and NOT re-claimed. The win (if S-1+S-2) = coexistence +
  memory retention, not a quality win.
- Rule: verbatim exp-0002 paired-CRN three-way + effect-size gate
  (`m>s_pair` AND sign≥⌈0.8n⌉) + δ=0.02; divergence guard; exact sign-test
  p corroboration; n=60 if any headline AMBIGUOUS.

## Escalation

1. `--smoke` pipeline + S-1 sanity (mem-adam must beat dense-naive).
2. `--seeds 10` class-IL — S-1 → S-2 (decisive) → S-3 → contrast.
3. n=60 confirmation on any AMBIGUOUS headline.

## Honest caveats baked in

Minimal KV-memory (not SMF's TF-IDF ranking) — coexistence existence-test,
not an SMF reproduction. Class-IL Split-MNIST scale (for comparability to
exp-0004/0008). Deep-unroll stability NOT tested here (exp-0005 owns it);
S-3 prevents mis-selling coexistence as a quality win. Any S-1+S-2 success
is "the one coherent NL system left", not "beats replay".
