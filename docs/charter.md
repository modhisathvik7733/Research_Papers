# Research charter

The one document every idea, paper, and experiment gets checked against. If a
proposed activity doesn't serve this, it's a distraction.

## The thesis

> Today an LLM is *thrown away and retrained from scratch* whenever the data
> changes. A 7B retrain costs ~$100k and the energy that implies. If a model
> could **keep learning new data cheaply without forgetting old data**, that
> retrain cost largely disappears.
>
> **Continual learning without catastrophic forgetting is therefore not a
> separate goal from cost reduction — it IS the cost-reduction mechanism.**

This project develops and understands that mechanism.

## What makes this defensible for a solo researcher

Catastrophic forgetting and training-cost reduction are each heavily attacked by
large labs. A solo researcher with no GPU farm cannot win at 7B from-scratch
training. The contribution surface that *is* open:

1. **Cost is a first-class metric.** Most continual-learning (CL) papers report
   only accuracy/forgetting and ignore compute. We measure, in every
   experiment, the wall-clock / FLOPs / $ of the CL method **vs. the from-scratch
   retrain baseline it claims to replace.** That comparison is under-reported and
   is exactly the thing that matters for the thesis.
2. **Mechanism at tiny scale.** The *why* of forgetting (representation drift,
   capacity, task interference) is studied at sub-125M params where the loop is
   minutes, not weeks. Mechanisms transfer up; specific big runs don't.

## Locked scope (from scoping Q&A, 2026-05-16)

| Decision        | Choice |
|-----------------|--------|
| Thrust          | Single: cheap incremental learning w/o forgetting. **Training cost measured explicitly in every experiment.** |
| Scale           | Tiny first (sub-125M / toy), fast loop, 30+ runs/week. Scale up only after a mechanism works small. |
| Compute         | M1 Max 32GB local for all mechanism work. Rent 1×A100 only for a later transfer-validation run. |
| Dev domain      | Cheap non-LLM toy CL benchmark first (Split-MNIST class-incremental). Transfer mechanism to small LM only after it works. |
| Time            | Full-time (30+ hrs/wk). |
| 3-month success | A *working system*: a model that incrementally learns a sequence of tasks with measured low forgetting AND measured lower cost than from-scratch retrain. |

## Definition of done (3 months)

A runnable pipeline where: given tasks T1…Tk arriving in sequence, the model
ends with **average accuracy within X of joint training**, **forgetting below
Y**, at **< Z% of the compute of retraining from scratch each time** — with
X, Y, Z measured, not asserted, against a reproduced baseline.

## Non-goals (say no to these)

- Beating SOTA LLMs on anything.
- Training anything ≥1B before a mechanism works at tiny scale.
- Inference speed / quantization / serving (revisit only after the thesis holds).
- New architectures for their own sake. Architecture changes must be justified
  by a forgetting/cost result, not aesthetics.
