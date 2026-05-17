# 0004 — Real-benchmark forgetting: the nested optimizer on Split-MNIST (then Permuted, then text)

**Status:** PRE-REGISTERED + runnable. This README is the registration;
predictions and metric/decision rules are fixed here before reading results.
**Hardware:** M1 Max (CPU/MPS), torchvision MNIST auto-downloads. Minutes/run.
Order (user-agreed, scientifically clean): **Split-MNIST → Permuted-MNIST →
real text streams.** This dir implements stage 1 (Split-MNIST) first.

## Why this exists

Everything in exp-0001/0002/0003 is a 24-d synthetic toy. There is **no
real-benchmark forgetting number** for the nested local-PC system. This
experiment produces one, using the metrics real continual-learning papers use,
so "how much does my system forget in real scenarios" gets a defensible answer
instead of an extrapolation.

## Benchmark (stage 1)

**Split-MNIST, task-incremental, multi-head** (the canonical setup where the
Chaudhry et al. 2018 / GEM-lineage *Forgetting* metric is well-defined): 5
sequential tasks — digit pairs (0,1)(2,3)(4,5)(6,7)(8,9), each relabelled
{0,1}; shared MLP trunk 784-256-256, one 2-way linear head per task. Real
MNIST data, real test split.

## Metrics (exactly the field-standard ones)

Accuracy matrix R[t,i] = test acc on task i after training through task t.
- **ACC** = mean_i R[T-1, i]  (final average accuracy; higher better).
- **Forgetting** = mean_{i<T-1} ( max_{t≤T-1} R[t,i] − R[T-1,i] )  (Chaudhry
  et al.; the headline "how much did it forget" number; lower better).
- Also reported raw: task-1 accuracy right after task 1 vs at the very end.

## Arms

| arm | what it is | cost |
|---|---|---|
| `naive` | Adam, sequential, no mitigation | — (the raw forgetting number) |
| `localpc` | **your system**: K-level momentum-cascade optimizer credited by local adjacent-level prediction-error (faithful port of exp-0002, applied as the optimizer on the real net) | O(1)-in-H |
| `replay` | small per-task reservoir, mixed each step | O(buffer) |
| `ewc` | elastic weight consolidation (Fisher penalty) | O(params) |

## Pre-registered predictions (fixed before running)

Honesty note: exp-0002 established local-PC ties a *tuned scalar* on quality
off-degenerate-geometry and wins by *dominance* (cost/stability), **not** by
being a strong anti-forgetting method per se. We therefore do **not** predict
local-PC beats replay/EWC at forgetting. Pre-registered:

- **H-real-1 (naive forgets a lot).** `naive` Forgetting is large
  (pre-registered threshold: > 15 accuracy points; literature says far more
  for class-IL, less for task-IL — we fix 15 pts as "non-trivial").
- **H-real-2 (ordering).** Forgetting: `replay` < `ewc` < `naive`, and
  `localpc` is **not better than `replay`** (pre-registered: we expect
  localpc ≈ naive ± , i.e. an optimizer cascade is not a strong CL fix). A
  surprise (localpc clearly < naive, approaching replay) is reported as a
  genuine positive; localpc ≥ naive is reported straight as "the nested
  optimizer does not mitigate real forgetting."
- **H-real-3 (the honest headline).** The deliverable is the **number**:
  naive Split-MNIST task-IL Forgetting and ACC with CIs, and where localpc
  lands between naive and replay. No spin; whatever it is, it is.

## Fairness control (pre-registered after smoke, before the seeded run)

The smoke run (1 seed) revealed the first naive port of the exp-0002 cascade
chains momentum levels → effective LR ≈250× → `localpc` collapsed to chance.
Reporting that would repeat the exp-0002 "undertuned baseline" error. Fix,
pre-registered here before any seeded run: (a) the cascade is reimplemented as
a **gain-normalised parallel multi-timescale momentum** (per-level proper EMA,
unit DC gain, mean over K — the faithful deployable form of the local-PC
structural law); (b) `localpc` receives a **per-seed LR fairness grid**
{3e-4, 1e-3, 3e-3, 1e-2}, best-final-ACC kept (exactly the exp-0002
best-of-grid fairness control); reported as `localpc*`. Adam baselines stay at
the canonical 1e-3 (already adaptive; standard). This sequence — smoke →
found-unfair → fix+pre-register → seeded run — is logged so it is auditable.

## Decision rule

Paired across seeds (common data order per seed across arms), divergence
guard (acc non-finite ⇒ drop, counted), verbatim exp-0002 three-way rule on
paired per-seed metric differences (`m>s_pair` & sign≥⌈0.8n⌉ ⇒ SEP; `m≤0.5
s_pair` or sign<⌈0.6n⌉ ⇒ NOT; else AMB); exact sign-test p as corroboration;
n=60 confirmation pre-committed if a headline comparison is AMBIGUOUS.

## Escalation chain

1. `--smoke` — 1 seed, reduced epochs: pipeline + metric sanity.
2. `--seeds 10` Split-MNIST — the real numbers (ACC, Forgetting, all arms).
3. Permuted-MNIST (10 tasks) — stage 2 (separate flag, after stage 1 holds).
4. Real text streams (math→science domain-incremental) — stage 3.
5. `--seeds 60` confirmation on any AMBIGUOUS headline comparison.

## Honest caveats baked in

Task-incremental multi-head is the *easier* protocol (test-time task ID
known) — class-incremental forgetting is far worse and is a pre-registered
stage-1b flag, reported separately, not blended. MNIST is "real data" but
small; it is the field-standard CL benchmark, not a claim about LLM-scale.
`localpc` is the exp-0002 optimizer ported faithfully but is an *optimizer*,
not a dedicated CL algorithm — measured honestly alongside, not promoted.
