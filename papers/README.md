# Reading list

One row per paper. Triage ruthlessly. Promote a paper to its own note
(`<short-name>.md` from `TEMPLATE.md`) only when you've done Pass 2.

Queue is ordered. Read top-down. Don't skip ahead — each unlocks the next.

| # | Paper | Why it's here (re: charter) | Status | Note |
|---|-------|-----------------------------|--------|------|
| 1 | **Kirkpatrick et al. 2017 — "Overcoming catastrophic forgetting in neural networks" (EWC)** | The canonical statement of the problem + the regularization approach. You must understand forgetting before any method. | queued | — |
| 2 | **Lopez-Paz & Ranzato 2017 — "Gradient Episodic Memory" (GEM)** | Introduces the standard CL metrics you will use forever: Average Accuracy, **Backward Transfer (forgetting)**, Forward Transfer. Read it for the metrics as much as the method. | queued | — |
| 3 | **Prabhu et al. 2020 — "GDumb: A Simple Approach that Questions Our Progress in Continual Learning"** | A deliberately dumb buffer+retrain baseline that beats many sophisticated CL methods. Read this early so you stay skeptical of complexity — directly serves the "cost vs. naive retrain" framing. | queued | — |
| 4 | **Finn et al. 2017 — "Model-Agnostic Meta-Learning" (MAML)** | The *acquisition* face: learning to learn a new task from few examples. The other half of your thesis — read after you understand forgetting. | queued | — |
| 5 | **A recent continual-learning survey (e.g. De Lange et al. 2021 / Wang et al. 2024)** | Map the method families (regularization / replay / parameter-isolation / meta-CL) so you know what's been tried and where the cost+generative angle is unexplored. | queued | — |

## Skipped (with reason)

- _(none yet)_

## Reading rule

For papers 1–4 you must reach **Pass 3** (could-I-reproduce-it). They are short
and foundational. The survey (#5) is Pass 2 only — it's a map, not a method.
