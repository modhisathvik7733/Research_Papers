# Research charter

The one document every idea, paper, and experiment gets checked against. If a
proposed activity doesn't serve this, it's a distraction.

## The thesis

> Today an LLM is *thrown away and retrained from scratch* whenever the data
> changes, and it needs vast repetition to learn anything. A 7B retrain costs
> ~$100k and the energy that implies. The root cause is one disease:
> **the model extracts too little structure per example, so it brute-forces
> generalization with scale and repetition.**
>
> That disease has two faces:
> - **Temporal — catastrophic forgetting:** it destroys knowledge it already
>   paid for when it learns something new.
> - **Acquisition — sample inefficiency:** it needs ~100 examples to grasp a
>   pattern a human gets in one or two.
>
> **Learning efficiency** = maximum generalization per unit of data and
> compute = the single thing this project attacks. Training cost is the *bill*
> for inefficiency; reducing it is the consequence, measured every experiment.

This project develops and understands that mechanism — and it must end in a
model that can **generate coherent text, not just classify**.

> Most of this capability is ~80–90% already implemented (closest: Google's
> Nested Learning/Hope). That is normal, not fatal. The defensible residual —
> cost-honest, reproducible evidence at verifiable scale — is analyzed in
> [prior_art.md](prior_art.md). Read it with this charter.

## Quality bar (non-negotiable — this is what stops self-deception)

Any efficiency win is only real if quality holds. Therefore:

1. **The end artifact is generative.** "Able to talk, not just understand." Toy
   classification benchmarks are allowed for *mechanism development only*; the
   thesis is not validated until the mechanism is shown on a **small generative
   language model that produces coherent text for its scale**.
2. **Parity is relative, never absolute.** Quality is judged against a *fair,
   same-scale, same-data baseline* trained normally — **never against GPT**.
   Absolute GPT-level fluency is a scale phenomenon and is explicitly out of
   reach on this hardware; claiming it would be dishonest.
3. **The bar:** the efficient method must be ≥ the same-scale baseline on
   fluency and generalization *while* being cheaper / forgetting less. A
   cheaper-but-dumber model is a failed experiment, not a result.

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
| Thrust          | Single: **learning efficiency** (forgetting + sample efficiency are two faces of it). **Training cost measured explicitly in every experiment.** |
| Target regime   | Sequential tasks *with few examples each* ("both at once") is the destination. But experiments isolate ONE face at a time first, then combine. |
| Scale           | Tiny first (sub-125M / toy), fast loop, 30+ runs/week. Scale up only after a mechanism works small. |
| Compute         | M1 Max 32GB local for all mechanism work. Rent 1×A100 only for the month-3 generative-LM transfer-validation run. |
| Dev domain      | Toy classification (Split-MNIST) for mechanism dev. **Final validation is mandatory: a small *generative* LM that produces coherent text.** |
| Time            | Full-time (30+ hrs/wk). |
| 3-month success | A *working system*: a small **generative** LM that learns a sequence of few-shot tasks with measured low forgetting, measured lower cost than from-scratch retrain, AND text coherence ≥ a same-scale baseline. |

## Definition of done (3 months)

A runnable pipeline where: a **small generative LM** is taught tasks T1…Tk
arriving in sequence with **few examples per task**, and ends with
**generalization within X of joint training**, **forgetting below Y**,
**text coherence ≥ a same-scale normally-trained baseline**, at **< Z% of the
compute of retraining from scratch each time** — with X, Y, Z and the coherence
comparison measured, not asserted, against reproduced baselines. Mechanism may
be developed on toy classification; it is not "done" until shown generative.

## Non-goals (say no to these)

- Beating SOTA LLMs on anything.
- Training anything ≥1B before a mechanism works at tiny scale.
- Inference speed / quantization / serving (revisit only after the thesis holds).
- New architectures for their own sake. Architecture changes must be justified
  by a forgetting/cost result, not aesthetics.
