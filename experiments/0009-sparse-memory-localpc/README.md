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

## Results (class-IL, n=10, RESOLVED 2026-05-17) — S-1 FAIL: testbed-inadequate, synthesis UNTESTED

```
 dense-naive ACC 0.195 Forget 0.992
 replay      ACC 0.895 Forget 0.110  (bar)
 mem-adam    ACC 0.104 Forget 0.086  <- ACC ≈ chance (0.10)
 mem-localpc ACC 0.103 Forget 0.293  <- ACC ≈ chance
 S-1 forget gap dense−mem-adam +0.906 10/10 p=.002  BUT mem-adam ACC≈chance
 S-2 gated OFF (not interpretable at chance accuracy)
```

**S-1: FAIL — the minimal KV-memory layer is too weak to learn class-IL
(ACC ≈ chance). Its low Forget is the not-learning artifact, not
retention.** The pre-registered, smoke-hardened accuracy gate fired
correctly and **prevented a fabricated success**: the raw forgetting
collapse (0.99→0.09, 10/10, p=.002, "SEP") would, unguarded, have read as
"sparse memory delivers retention — synthesis confirmed." It is not; the
model learned nothing. **S-2 not interpretable; the synthesis is neither
confirmed nor refuted — UNTESTED.**

Mechanism (hypothesis, labelled): zero-init values + fixed-ish random keys
+ hard top-k, replacing (not augmenting) the FFN, starves gradient flow →
representation can't form. Real SMF has a learned query, TF-IDF slot
ranking, and augments rather than replaces capacity; the minimal version
stripped too much. This is an implementation-strength issue, not evidence
about the synthesis.

**Methodological win (the actual takeaway):** the only thing that worked
here was the discipline — the pre-registered hardened gate caught a
spurious "it works" that the headline numbers would otherwise have
produced. 9 experiments in, the guardrails are still earning their keep.

**Status:** synthesis UNTESTED (testbed inadequate). To test it honestly
needs a real memory layer (learned query, better init, augment-not-replace,
ideally the actual SMF code), which is a larger build than a toy reimpl.

## Honest caveats baked in

Minimal KV-memory (not SMF's TF-IDF ranking) — coexistence existence-test,
not an SMF reproduction. Class-IL Split-MNIST scale (for comparability to
exp-0004/0008). Deep-unroll stability NOT tested here (exp-0005 owns it);
S-3 prevents mis-selling coexistence as a quality win. Any S-1+S-2 success
is "the one coherent NL system left", not "beats replay".
