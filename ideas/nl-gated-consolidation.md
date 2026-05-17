# Finding (proposed): Gated Consolidation — a Signal-Driven, Parameter-Free Fix to NL/CMS Global Rigidity

- **Date:** 2026-05-17
- **Source:** Root-cause fix for the measured failure in exp-0004 (vanilla
  multi-timescale CMS fails class-IL ≈ chance AND anti-stacks with replay:
  replay 0.110/0.895 → rpc 0.218/0.758, strictly worse 0/10).
- **Status:** RUN (2026-05-17, n=10 class-IL). **DECISIVE DOUBLE NEGATIVE.**
  Sanity ✓; **G-1 FAIL** (rgc ACC 0.347 vs replay 0.895, 0/10, and worse
  than vanilla 0.758 — gate deepened the conflict); **G-2 FAIL** (gat≈van≈
  chance — and this *falsifies the exp-0004 "rigidity is the cause" story*:
  removing rigidity didn't restore plasticity). Pre-registered honest prior
  fired: optimizer-side consolidation can't match/combine-with rehearsal
  even gated — hardened law. Self-correction logged. Results:
  [experiments/0008-nl-gated-consolidation/](../experiments/0008-nl-gated-consolidation/).
- **Intended use:** the smallest component change to HOPE/CMS that targets
  the *measured* root cause; informative either way.

## 1. One-sentence contribution

Vanilla CMS consolidates every level on every step and is therefore globally
rigid (resists *all* weight change, including what new tasks need); we replace
that with **gated consolidation** — each slow level absorbs the current
gradient, and contributes to the update, only in proportion to its *agreement*
with that gradient (`γ_k = relu(cos(g, e_k))`), so it locks in stable
knowledge but neither absorbs nor fights conflicting (new-task) signal; the
fast level stays fully plastic.

## 2. Why this is the right change (measured, not speculative)

exp-0004 (real Split-MNIST, pre-registered, paired): vanilla multi-timescale
CMS as an optimizer barely learns class-IL (ACC ≈ chance 0.22) and *degrades
replay* (anti-stacking, 0/10). Mechanism established there: global rigidity —
slow momentum freezes the directions the new task and replay must move.
Gated consolidation attacks exactly that, with **no learned parameters and no
meta-objective** (we showed meta-training is fragile/degenerate), preserving
the proven O(1)-in-H property.

## 3. The component (precise)

Per step, flat gradient g; per level k an EMA e_k with fixed pole β_k.
- Fast level k=0: γ₀ = 1 (full plasticity, unchanged).
- Slow levels k≥1: aₖ = cos(g, e_k); **γ_k = relu(aₖ) ∈ [0,1]**.
- Consolidation: `e_k ← β_k e_k + γ_k (1−β_k) g`  (don't absorb conflict).
- Update: `Δ = e₀/K + Σ_{k≥1} γ_k e_k / K`  (don't let stale memory fight
  current learning).
Parameter-free, signal-driven, O(1)-in-H. A drop-in replacement for the CMS
block / the exp-0004 multi-timescale optimizer.

## 4. Falsifiable claims (pre-registration)

> Decisive regime = exp-0004 **class-IL Split-MNIST** (where vanilla CMS
> provably fails). Arms: `naive`, `replay`, `rpc`(replay+vanilla CMS),
> **`rgc`(replay+Gated CMS)**, plus `van`(vanilla alone), `gat`(gated alone).
> Paired CRN, verbatim exp-0002 three-way + effect-size gate + δ.
>
> - **Sanity (must hold):** `rpc` reproduces anti-stacking (`replay − rpc`
>   ACC worse ≥8/10). Else baseline broken — stop.
> - **G-1 (fix removes the conflict):** `rgc` is **not worse than `replay`**
>   on *both* Forgetting and ACC (SEP-favourable or NOT with mean ≥ 0).
> - **G-2 (gating restores plasticity):** `gat` alone class-IL ACC clearly
>   **>** `van` alone (vanilla ≈ chance 0.22; gated should learn).
> - **Honest pre-stated prior:** given the consistent finding that
>   optimizer-side memory is dominated by rehearsal, **G-1 failing is ≥ as
>   likely as succeeding**. A clean failure hardens "optimizer-side
>   consolidation, even gated, cannot match rehearsal" into a
>   mechanism-backed law. We test a fix, we do not promote it.
> O(1)-in-H must be preserved (graph-node check) for any positive to count.

## 5. Honest limitations

Class-IL Split-MNIST scale. γ = relu(cos) is the simplest agreement gate; a
clean G-2-yes / G-1-no result motivates richer gates but a clean G-1-no at
this gate is already a strong negative. Parameter-free by design (no
meta-objective) — a deliberate constraint, stated. Any G-1 positive is
parity-with-replay (conflict removed), not superiority — claimed as such.

## 6. Score

- Novelty: 4 (parameter-free signal-gated consolidation as a CMS redesign
  targeting a *measured* failure)
- Testable cheaply: 5 (exp-0004 class-IL harness + one optimizer; ≤1 day)
- Informative if it fails: 5 (clean G-1-no = optimizer-side consolidation
  cannot match rehearsal even when gated — a hardened, general law)
- **Total = 14 → promote to experiment 0008.**
