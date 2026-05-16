# Experiment 0001 — Reproduce (and measure) catastrophic forgetting

**You do not run this until you've read papers 1 and 2.** You need GEM's metric
definitions (Average Accuracy, Backward Transfer) to even define "result" here.
This file exists now to lock the hypothesis *before* reading biases you.

## Why this is experiment 0001
Per methodology: baseline before method. Here the "baseline" is the *phenomenon
itself*. You cannot claim any CL method works until you can produce, measure,
and quantify the forgetting it's supposed to fix. Most people skip this and
never actually know how bad forgetting is on their setup. We won't.

## Hypothesis (written BEFORE running, BEFORE reading the papers)
> H1 (forgetting exists, quantified): A small MLP/CNN trained *sequentially* on
> the 5 tasks of Split-MNIST (class-incremental: {0,1}→{2,3}→…→{8,9}) will, after
> task 5, show accuracy on task 1 **drop by ≥ 40 absolute points** vs. its
> accuracy right after learning task 1.
>
> H2 (the cost premise of the whole project): The "no-forgetting" reference —
> retraining from scratch on the union of all tasks (joint training) — costs
> **≥ 3× the wall-clock** of one sequential pass, and this multiple grows with
> the number of tasks.
>
> Decision rule: if H1's drop is < 40 pts, forgetting is mild on this setup →
> the toy benchmark is too easy, pick a harder one before proceeding. If H2's
> multiple is < 1, the cost premise is wrong on this setup → rethink the thesis.

## Pre-registered prediction
Write your honest guess here, with a number, BEFORE running:
- Task-1 accuracy right after task 1: ____%
- Task-1 accuracy after task 5: ____%
- Joint-training wall-clock ÷ sequential wall-clock: ____×
(Being wrong here is the point. It calibrates how badly you currently
understand the phenomenon.)

## Setup (fill after reading papers 1–2)
- Baseline (no-forgetting reference): joint training on all tasks at once.
- The one variable: training *order* — sequential vs. joint. Nothing else.
- Held fixed: architecture, optimizer, epochs/task, seed, data.
- Metrics: Average Accuracy + Backward Transfer (GEM defs) + wall-clock seconds
  and parameter-update count for BOTH regimes.
- Config: `config.yaml`
- Seed / commit / env: filled by the run.

## Run
```bash
# from repo root, one-time:
uv sync --extra experiments            # see docs/setup.md
# then:
uv run jupyter lab experiments/0001-reproduce-forgetting/run.ipynb
```
The notebook (`run.ipynb`) refuses to run until you fill the pre-registration
gate. It tests H1 (task-1 drop ≥ 40 pts) and H2 (joint ≥ 3× sequential cost)
using Avalanche's Split-MNIST + Naive/JointTraining so the baseline is a known,
reproducible one — not hand-rolled.

## Results (raw)
_pending_

## Conclusion
_pending — must explicitly state H1/H2 supported/rejected with the numbers._
