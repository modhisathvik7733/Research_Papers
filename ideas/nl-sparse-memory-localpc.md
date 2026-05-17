# Finding (proposed): Sparse Explicit Memory (retention) + Local-PC (deep-optimization stability) — the only NL synthesis consistent with all our evidence

- **Date:** 2026-05-17
- **Source:** Convergence of (a) our 8-experiment arc — storage beats
  compressed traces; replay won; NL/CMS optimizer-memory anti-stacks with
  retention (exp-0004/0008); Local-PC's only survivor is structural O(1)
  deep-unroll stability (exp-0005) — and (b) external SOTA
  [Sparse Memory Finetuning, arXiv 2510.15103] + [Memory Is Not the
  Bottleneck, arXiv 2502.07274].
- **Status:** RUN (2026-05-17, n=10 class-IL). **S-1 FAIL: testbed
  inadequate — synthesis UNTESTED.** Minimal KV-memory layer collapses to
  chance ACC (0.104 ≈ 0.10); its low Forget is the not-learning artifact.
  The pre-registered hardened accuracy gate fired and **blocked a fabricated
  "synthesis confirmed"** (raw forget 0.99→0.09 looked spectacular, was
  vacuous). S-2 not interpretable. Synthesis neither confirmed nor refuted;
  needs a real memory layer (learned query / better init / augment-not-
  replace / actual SMF code) — a larger build. Results:
  [experiments/0009-sparse-memory-localpc/](../experiments/0009-sparse-memory-localpc/).
- **Intended use:** the culminating test — do the two *surviving* mechanisms
  coexist without the anti-stacking that killed every prior NL variant?

## 1. One-sentence contribution

We pair the two mechanisms that *survived* falsification — **sparse explicit
key-value memory** for retention (Family B: store, don't compress) and
**Local-PC** for O(1) credit assignment — and test the precise claim that,
because they act on *different axes* (what is kept vs how credit flows), they
do **not** conflict, unlike vanilla CMS/optimizer-memory which anti-stacked
with every retention mechanism (exp-0004/0008).

## 2. Why this is the only evidence-consistent NL synthesis

Every prior NL variant failed for one reason: it used a *compressed
parametric trace* to do retention, and that loses to / conflicts with
explicit storage (exp-0002→0008; SMF independently). The two things that
never failed: explicit storage (replay/KV) for retention, and Local-PC's
O(1)-in-H structural property for deep optimization (exp-0005). This proposal
is exactly their composition and nothing else — retention is handed to the
mechanism that won it; Local-PC is confined to the axis where it survived.

## 3. The falsifiable claims (pre-registration)

Decisive regime = exp-0004 **class-IL Split-MNIST**, where vanilla CMS
catastrophically failed (≈chance, anti-stacked with replay). The dense
FFN is replaced by a minimal **sparse key-value memory layer** (M slots,
top-k access, sparse-by-construction updates).

> - **S-1 (memory delivers retention):** `mem-adam` (sparse-memory net,
>   Adam, **no replay buffer**) has Forgetting ≪ dense-naive (exp-0004
>   ≈0.99) — approaching replay-level retention with *no* buffer. If not,
>   the SMF mechanism didn't transfer to our minimal setting; synthesis
>   premise weakens (reported straight).
> - **S-2 (THE test — non-conflict):** `mem-localpc` is **not worse than
>   `mem-adam`** on *both* Forgetting and ACC (verbatim exp-0002 three-way:
>   SEP-favourable OR NOT-separated with mean ≥ 0). I.e. Local-PC and sparse
>   memory coexist — in sharp contrast to exp-0008 where CMS+replay
>   anti-stacked. **Honest prior: this should HOLD (different axes).** A
>   clean failure ⇒ the synthesis is false: Local-PC conflicts with *any*
>   retention mechanism, not just rigidity-based ones — a stronger, hardened
>   negative, reported as such.
> - **S-3 (scope honesty, pre-registered):** `mem-localpc` shows **no
>   quality *advantage*** over `mem-adam` here. Local-PC's value is the
>   deep-unroll stability regime (exp-0005), which this shallow benchmark
>   does **not** test and we do **not** re-claim. The win claimed if S-1+S-2
>   hold is *coherent coexistence + memory-driven retention*, with Local-PC's
>   structural value established separately — not a quality win here.
> - **Contrast:** `mem-cms` (vanilla CMS on the memory net) — does the
>   rigidity conflict persist even on a memory architecture? Either result
>   localises whether it is specifically Local-PC's O(1) form that coexists.

Decision rule = verbatim exp-0002 paired-CRN three-way + effect-size gate
(`m>s_pair`, sign≥⌈0.8n⌉) + practical δ=0.02; divergence guard; exact
sign-test p as corroboration; n=60 if any headline AMBIGUOUS.

## 3a. Design-time clarification (logged before any run)

On a single real classifier, Local-PC's *deployable* form **is** the
multi-timescale-momentum optimiser (= exp-0008 vanilla CMS); Local-PC's
distinctive O(1)-*credit* only differs from the hypergradient in the
*nested/unrolled* regime (exp-0005). So 0009's honest experimental variable
is **moving retention into an explicit sparse-memory architecture** and
asking whether the NL-style optimiser then coexists with it — the direct
contrast to exp-0008, where the same optimiser anti-stacked with
replay-based retention. The distinctive deep-unroll O(1) value of Local-PC
is owned by exp-0005 and is **not** re-claimed here (S-3).

## 4. Honest limitations

Minimal KV-memory (not SMF's TF-IDF slot ranking) — existence test of
coexistence, not an SMF reproduction. Class-IL Split-MNIST scale (chosen for
direct comparability to exp-0004/0008). Local-PC = the deployable O(1)
multi-timescale form. **Deep-unroll stability is NOT tested here** (exp-0005
owns that); S-3 makes that explicit so coexistence is not mis-sold as a
quality win.

## 5. Score

- Novelty: 4 (composition of the two falsification-survivors; tests the
  precise non-conflict claim, not a vibe)
- Testable cheaply: 5 (exp-0008 class-IL harness + a KV-memory layer; ≤1 day)
- Informative if it fails: 5 (clean S-2 failure = Local-PC conflicts with
  retention *generally* — hardens the negative into a law and closes the NL
  line cleanly; clean S-1+S-2 success = the one coherent NL system left)
- **Total = 14 → promote to experiment 0009.**
