# 0003 — Plasticity loss vs catastrophic forgetting (cost-asymmetry)

**Status:** PRE-REGISTERED, not yet run. This README is the registration;
predictions and the decision rule are fixed here *before* any result.
**Hardware:** CPU, M1 Max. `python3 run.py --sanity` then `--seeds 10`.
Idea record: [ideas/plasticity-vs-forgetting.md](../../ideas/plasticity-vs-forgetting.md).

## Question

At the small-capacity regime where cheap continual learning is feasible, is
continual degradation dominated by **loss of plasticity** (can't fit *new*
tasks) rather than **catastrophic forgetting** (loss on *old* tasks)? And are
the two **doubly dissociable** by interventions of asymmetric cost — an O(1)
plasticity tweak vs an O(buffer) replay?

## Construction

- Shared fixed random feature map `φ(x)=tanh(Fx)`, `F∈R^{D_FEAT×D_IN}`,
  reseeded per seed. Student = 1-hidden-layer net `relu` over φ.
- T tasks streamed sequentially, B steps each. Task t = regress to a **fresh
  random teacher** `y = u_tᵀ relu(V_t φ(x))` (every task equally hard for a
  *fresh* student by construction — so any rising difficulty is the student's
  lost plasticity, not the task).

## Metrics (per seed, common random numbers across arms)

- **F (forgetting)** = mean_{i<T} [ loss_i(final) − loss_i(just after task i) ].
- **P (plasticity loss)** = mean_t [ trainloss_t(continual, end of block t) −
  trainloss_t(**fresh net**, same arch, same B steps, same task) ].
- Total degradation ≈ F + P (residual reported, not assumed zero).

## Arms

| arm | cost | targets |
|---|---|---|
| `plain` | — | reference (Adam) |
| `plast` | **O(1)** | shrink-perturb at block boundaries: θ ← α·θ + σ·ε |
| `replay` | **O(buffer)** | mix k% replayed past samples each step |

## Pre-registered predictions (fixed before running)

- **H-diss (double dissociation, the core claim).** Paired CRN, n seeds:
  1. `plast` vs `plain`: ΔP **SEPARATED-negative** (P reduced) **AND** ΔF
     **NOT-separated** (F unchanged).
  2. `replay` vs `plain`: ΔF **SEPARATED-negative** (F reduced) **AND** ΔP
     **NOT-separated** (P unchanged).
  Claim holds **iff all four** hold (clean cross). Any three-of-four ⇒ reported
  as *partial / not a clean dissociation*, not spun into a win.
- **H-cap (capacity flip).** Plain model: P-dominant at small width →
  F-dominant at large width, crossing at some c\* in the swept range. Direction
  pre-registered; "no flip / F always dominates" reported straight (kills the
  cost claim).
- **H-cost (payoff).** In the small-width regime, fraction of total
  degradation removed by the O(1) `plast` ≥ fraction removed by O(buffer)
  `replay`. Quantified table, not asserted.

## Decision rule (verbatim from exp-0002; no new goalposts)

Per comparison, on the stable subset (divergence guard: loss non-finite or
>1e3 ⇒ diverged, counted, never averaged), paired per-seed difference d,
m=mean(d), s=std(d, paired). **SEPARATED** iff `m > s` and sign ≥ ⌈0.8·n⌉;
**NOT** iff `m ≤ 0.5·s` or sign < ⌈0.6·n⌉; else **AMBIGUOUS**. Exact
two-sided sign-test p reported as corroboration, not the gate. Higher-n (n=60)
confirmation pre-committed if n=10 lands AMBIGUOUS (the exp-0002 lesson).

## Non-degeneracy guard (MUST pass before interpreting H-diss)

`--sanity`: in the `plain` model, per-task plasticity P_t must have a
**significantly positive slope in t** (the net genuinely loses new-task
fitting ability over the stream). If slope ≈ 0, the construction cannot
exhibit the phenomenon → experiment degenerate for this question, reported
straight (the P1 lesson, carried forward). No H-diss claim is valid if the
guard fails.

## Escalation chain (planned, not yet run)

1. `--sanity` — plasticity-exists slope. MUST pass.
2. `--seeds 10` — paired double dissociation (H-diss).
3. `--capacity` — the P→F flip (H-cap).
4. second O(1) fix arm (L2-to-init / dormant reset) — not intervention-specific.
5. `--seeds 60` — higher-n confirmation if any n=10 verdict is AMBIGUOUS.
6. cost-fraction table (H-cost) — the headline.

## Honest caveats baked in

Toy/CPU; synthetic teachers; one concrete O(1) intervention; F and P treated
as approximately additive (residual reported). Existence-of-mechanism + cost
asymmetry, not scale.
