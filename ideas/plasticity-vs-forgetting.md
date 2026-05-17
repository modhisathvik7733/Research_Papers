# Finding (proposed): At Toy/Cheap Scale, Continual Degradation Is Plasticity Loss, Not Catastrophic Forgetting — and the Fix Is O(1), Not O(buffer)

- **Date:** 2026-05-17
- **Source:** Idea-generation turn after exp-0002 closed; carries forward the
  cost thesis ([lab_notebook/2026-05-16.md](../lab_notebook/2026-05-16.md)) and
  the pre-registration / paired-CRN discipline proven in
  [experiments/0002](../experiments/0002-nested-local-coupling/).
- **Status:** PROPOSED — pre-registered, not yet run. Experiment skeleton at
  [experiments/0003-plasticity-vs-forgetting/](../experiments/0003-plasticity-vs-forgetting/).
- **Intended use:** seed for a short, honest empirical paper: *"the cheap CL
  win is a 3-line plasticity fix, not a replay buffer — and the field measures
  the wrong failure at small scale."*

> Durable record of the contribution + the falsifiable claim so it can be
> lifted into a draft. No numbers yet — this file is the pre-registration.

---

## 1. One-sentence contribution

We claim that at the small-capacity regime where cheap continual learning is
actually feasible, the dominant cause of continual degradation is **loss of
plasticity** (the network progressively loses the ability to fit *new* tasks),
**not** catastrophic forgetting (loss on *old* tasks) — and that the two are
**doubly dissociable** by interventions of asymmetric cost: an **O(1)**,
≤3-line plasticity intervention removes the plasticity term without touching
forgetting, while an **O(buffer)** replay intervention removes forgetting
without touching plasticity. If true, the cheapest large win in continual
learning is an optimizer tweak, not data storage.

## 2. The gap this addresses (and why it's the cost thesis)

The program's thesis: the $100k retrain exists because we discard the model
and retrain from scratch; continual learning *is* the cost-reduction
mechanism. The CL literature overwhelmingly attributes continual failure to
catastrophic forgetting and "fixes" it with replay/regularisation that cost
memory or compute proportional to history. Recent work (Dohare 2024; Lyle
2023; Abbas 2023) argues a different failure — *loss of plasticity* — may
dominate, and its fixes are O(1) (shrink-perturb, L2-to-init, dormant-unit
reset). **Nobody has cleanly disentangled the two at the tiny scale where a
solo researcher gets 30+ runs/week**, nor mapped which one dominates as a
function of capacity, nor stated it as a cost asymmetry. That disentanglement,
done with pre-registration + paired statistics, is the contribution.

## 3. Definitions (pre-registered, crisp)

Sequential stream of T tasks, B steps each; each task = regress to a fresh
random teacher over a shared fixed feature map (every task equally hard for a
*fresh* net by construction).

- **Forgetting F** = mean over i<T of [ loss_i(final continual net) −
  loss_i(continual net immediately after task i's block) ]. (Old-task damage.)
- **Plasticity loss P** = mean over t of [ trainloss_t(continual net at end of
  task t's block) − trainloss_t(a freshly-initialised net of identical
  architecture, same B-step budget, same task) ]. (New-task-fitting ability
  lost, measured against the fair fresh-net reference.)
- **Total continual degradation** decomposed as F (old) + P (new). Both are
  per-seed, common-random-number paired across arms.

## 4. Falsifiable claim (the pre-registration)

> **H-diss (double dissociation — the core claim).** Over paired seeds:
> (a) the O(1) plasticity intervention reduces **P** (paired effect-size gate
> cleared) **and** does **not** reduce **F** (NOT-separated); (b) the
> O(buffer) replay intervention reduces **F** (gate cleared) **and** does
> **not** reduce **P**. The claim is **falsified unless all four hold** — a
> clean cross, not a single arm.
>
> **H-cap (capacity flip).** The dominant term in plain continual training
> flips from **P-dominant at small capacity** to **F-dominant at large
> capacity** at some width c\*. Direction pre-registered; either outcome
> reported. (If P never dominates anywhere, the cost claim is dead — reported
> straight.)
>
> **H-cost (the payoff).** In the small-capacity regime relevant to cheap CL,
> the fraction of total degradation removable by the **O(1)** fix ≥ the
> fraction removable by the **O(buffer)** fix. Quantified, not asserted.
>
> **Decision rule** = the verbatim paired (CRN) three-way rule proven in
> exp-0002 (`m > s_pair` effect-size gate AND sign ≥ ⌈0.8n⌉ ⇒ SEPARATED;
> `m ≤ 0.5 s_pair` or sign < ⌈0.6n⌉ ⇒ NOT; else AMBIGUOUS), divergence guard
> active, exact two-sided sign-test p reported as corroboration only.

## 5. Non-degeneracy guard (the exp-0002 lesson, carried forward)

Pre-registered sanity that must pass *before* any H-diss interpretation:
**plasticity loss must actually exist in the plain model** — P measured per
task must have a significantly positive slope in t (the net genuinely gets
worse at fitting new tasks over the stream). If P's slope ≈ 0, the construction
cannot exhibit the phenomenon and the experiment is degenerate for this
question — reported straight, no spin, exactly as P1 was. This is the analogue
of the §4.3/Eq.45 single-scalar control: prove the testbed can show the effect
before claiming anything about it.

## 6. Honest limitations (must appear in any paper)

1. Toy/CPU scale, synthetic teachers — existence-of-mechanism and the cost
   asymmetry, *not* ImageNet-scale plasticity claims.
2. "Plasticity fix" is one concrete O(1) intervention (shrink-perturb);
   generality across the family (L2-to-init, reset) is a follow-up arm.
3. The dissociation could be intervention-specific rather than fundamental —
   the capacity-flip (H-cap) and a second fix arm are the controls for that.
4. Forgetting and plasticity may *interact* (not purely additive); the
   decomposition is reported as an approximation with the residual shown.

## 7. Positioning vs related work

- vs **Dohare 2024 / Lyle 2023 / Abbas 2023 (loss of plasticity):** we add the
  *disentanglement from forgetting* via a pre-registered double dissociation
  and the *cost-asymmetry* framing (O(1) vs O(buffer)), at a scale that is
  cheaply reproducible.
- vs **replay / EWC / GPM (forgetting):** we test, pre-registered, whether
  these fix the *wrong* term in the regime that matters for cheap CL.
- vs **exp-0002 / Nested Learning:** same harness lineage, same discipline;
  this asks "is the expensive nesting machinery even attacking the dominant
  failure?" — if plasticity dominates cheaply, much of the nesting/optimizer
  literature is solving the rarer problem.

## 8. Escalation chain (planned; mirrors 0002)

1. `--sanity` — plasticity-exists slope test (non-degeneracy guard). MUST pass.
2. `--seeds 10` paired CRN — the double dissociation (H-diss).
3. `--capacity` sweep — the P→F flip (H-cap), pre-registered direction.
4. second fix arm (L2-to-init / reset) — dissociation not intervention-specific.
5. higher n (n=60) paired confirmation (the exp-0002 lesson on variance).
6. cost-fraction table (H-cost) — the headline number.

## 9. Score

- Novelty: 4 (disentanglement + cost-asymmetry framing + pre-registration is
  not in the toy-scale literature)
- Testable cheaply: 5 (one self-contained CPU script; ≤1 day; reuses 0002's
  paired/guard machinery)
- Informative if it fails: 5 (a clean *no dissociation* or *F always
  dominates* result refocuses the entire cost program — equally publishable)
- **Total = 14; cheap experiment < 1 day → promote to experiment 0003.**
