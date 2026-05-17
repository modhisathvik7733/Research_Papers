# Finding (proposed): A Continual Benchmark That Certifies Its Own Non-Degeneracy — Fixing Nested Learning's Broken Evaluation Substrate

- **Date:** 2026-05-17
- **Source:** Direct consequence of the exp-0002 result we proved
  (§4.3/Eq.45 is optimizer-degenerate at every scale & heterogeneity: a
  tuned scalar matches everything) + the exp-0005 lesson
  (significance ≠ effect size; require power, not just a p-value).
- **Status:** PROPOSED — pre-registered, not yet run. Skeleton:
  [experiments/0007-nl-nondegenerate-benchmark/](../experiments/0007-nl-nondegenerate-benchmark/).
- **Intended use:** the field-facing artifact — a continual benchmark NL (or
  anyone) can argue method claims on *honestly*, because it refuses to be
  used when it cannot discriminate.

## 1. One-sentence contribution

We proved Nested Learning's canonical forgetting benchmark cannot tell a
sophisticated method from a tuned scalar; we therefore ship a benchmark with a
**mandatory degeneracy certificate** — a pre-registered single-scalar control
run as a pass/fail gate baked into the benchmark — plus a minimal instance
that *passes* it, so method comparisons on it are meaningful by construction.

## 2. The gap (and why it is uniquely ours to fill)

exp-0002→0005 established, pre-registered and paired: §4.3/Eq.45 is degenerate
(P1 DEGENERATE, scale-robust C-scale, family-robust C-stronger); off that
geometry (C-hetero, P3) real methods *can* be distinguished. **NL has no
admissible substrate for its central claims.** No one else has the degeneracy
result, so no one else can build the certificate. The contribution is not a
new model — it is making every future NL/optimizer/CL claim *falsifiable* by
refusing degenerate evaluation.

## 3. The certificate (pre-registered, method-agnostic, NL-neutral)

**Design-time scoping correction (caught before any run, logged honestly).**
The certificate certifies non-degeneracy *on the optimizer / credit-assignment
axis* — the axis NL's claims and the P1 result live on. The non-trivial
reference must therefore be the *best non-trivial optimizer*, **not replay**:
replay exploits a different (data-rehearsal) axis and would trivially "admit"
§4.3/Eq.45 (rehearsal fixes orthonormal-task forgetting), breaking the C-1
sanity and the framing. Degeneracy is **axis-specific**; this certificate is
explicitly the *optimizer-axis* certificate. To stay NL-neutral the reference
is the **establishment strong optimizer — the global hypergradient (the
expensive standard HOPE itself uses), not our Local-PC**. This reproduces P1
exactly (§4.3 REJECT; P3/C-hetero ADMIT) and cannot be accused of favouring
our own method.

A benchmark instance is **ADMISSIBLE** iff, paired common-random-number over
n certified seeds, the *best non-trivial optimizer* (global-hypergradient
nested optimizer) beats the **best trivial baseline** by the verbatim
exp-0002 three-way rule **with the effect-size gate** (`m > s_pair`, not a
bare p) **and** the gap exceeds a pre-registered minimal practical effect δ
(so we never certify on a negligible-but-consistent margin — the exp-0005
lesson). Else **REJECTED**, with reason logged: *degenerate* (no exploitable
headroom) or *underpowered* (headroom exists but seed variance swamps it at
feasible n — the C-stronger pathology).

Trivial baseline set (the "a constant suffices" family the benchmark must
defeat): plain SGD, best-grid single-β momentum (the P1 scalar), and a
fixed multi-β EMA (so multi-timescale-ness *alone*, with no learned
structure, cannot pass it either).

> The certificate is reported **alongside** any method result and a method
> claim on a REJECTED instance is, by construction, inadmissible.

## 4. Falsifiable claims (pre-registration)

> **C-1 (sanity, must hold).** The §4.3/Eq.45 instance (orthonormal
> directions, shared Hessian, single-pass) is **REJECTED — degenerate**
> (reproduces P1).
> **C-2.** A cyclic-reactivation + heterogeneous instance (P3-lineage:
> recurrence gap > momentum horizon, per-task Hessians, non-orthogonal
> directions) is **ADMITTED** (replay clearly beats best-trivial, effect
> size > δ, clears variance at certified n).
> **C-3.** Sweeping the three degeneracy knobs (orthogonality, Hessian
> heterogeneity, recurrence gap) the certificate flips REJECT→ADMIT at a
> **locatable boundary** — reported as the benchmark's operating manual.
> **Honest pre-stated risk:** if *no* cheaply-constructible instance passes
> BOTH clauses at a feasible seed budget, we report straight that
> toy-scale CL evaluation is fundamentally underpowered — a profound,
> publishable negative, not a failure to hide.

## 5. Honest limitations

Synthetic task family (the certificate logic is the contribution, not
photo-realism). δ and certified-n are pre-registered knobs, reported with
the boundary, never tuned post hoc. "Non-trivial reference = replay" is a
choice; a benchmark admissible only because *replay* exploits it may still be
degenerate for some other axis — stated explicitly; the certificate certifies
*discriminability*, not universal informativeness.

## 6. Score

- Novelty: 5 (a benchmark that certifies its own admissibility is, to our
  knowledge, not done in CL; only possible because we proved the degeneracy)
- Testable cheaply: 5 (reuses exp-0002 construction lineage + the paired
  machinery; ≤1 day)
- Informative if it fails: 4 (a "nothing passes at feasible n" outcome is a
  major statement about CL evaluation; a clean boundary is the deliverable)
- **Total = 14 → promote to experiment 0007.**
