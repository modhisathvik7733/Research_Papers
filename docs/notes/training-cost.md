# What actually determines training cost (intuitive)

*Triggered 2026-05-16 by the belief "training cost is mostly the optimizer."
It mostly isn't. Here's the mental model, no jargon.*

## The one equation, in plain words

> **Total cost = (cost of one step) × (how many steps you need).**

That's it. Every cost lever is really pulling on one of those two things.
Hold this picture: it's a **road trip**.

```
Total fuel  =  (fuel per mile)  ×  (number of miles)
```

## The road-trip model

| Road trip thing | Training thing | How big a lever |
|---|---|---|
| **Length of the route** (how far you must drive) | **Model size × amount of data** — bigger model, more data = far longer trip | 🟥 **Huge.** Orders of magnitude. The trip itself. |
| **Taking a smarter, shorter route** | **Data quality / efficiency** — good data reaches the destination in fewer miles | 🟧 **Big.** Can cut miles by a lot (e.g. "drop 90% of data, still arrive"). |
| **Re-driving the *entire* trip every time the map updates** | **Retraining from scratch** whenever data changes | 🟧 **Big & repeated.** Drive it 10 times = 10× fuel. |
| **How good the driver is at not taking wrong turns** | **The optimizer** (SGD vs Adam vs Muon…) | 🟨 **Medium-small.** A good driver saves maybe ~2× the miles. Real, but it's *driver skill, not route length*. |
| **Energy to turn the steering wheel vs. to move the car** | **The optimizer's own compute** (Adam's per-parameter math) | 🟩 **Tiny.** Far under ~1% of a step. The matmuls (moving the car) dominate. |

## Why the optimizer *feels* like the main thing (the trap)

In a training loop, the optimizer is the **most visible knob** — you literally
type `SGD(lr=0.05)`. The route length (model size × data) and "am I re-driving
the whole trip" are invisible in the code. **The thing you can see is almost
never the thing that dominates the cost.** Watch for this everywhere.

## Two facts that make it concrete

1. **One training step:** forward pass + backward pass (backward ≈ 2× forward)
   are almost all the FLOPs. The optimizer's update (a few operations per
   parameter) is a rounding error next to the big matrix multiplies. So the
   optimizer barely changes *cost per step* — at most it changes *how many
   steps* you need, and only by a constant factor (~1.5–3×).
2. **The famous rule of thumb:** compute ≈ `6 × (parameters) × (tokens)`.
   Notice what's in it: **size and data.** The optimizer isn't even in the
   formula — it hides inside "how many tokens until good enough," as a
   multiplier, not the base.

## The toy-scale warning (matters for this whole project)

On a **10-minute drive**, the time spent adjusting mirrors and the driver's
habits is a *big fraction* of the trip. On a **cross-country drive**, it's
noise. Same with training: at MNIST toy scale the optimizer looms large; at LLM
scale it's a constant factor swamped by route length.

➡️ **A cost intuition formed at toy scale can be flat wrong at real scale.**
This is exactly why the charter says *mechanism transfers, specific numbers
don't.* Develop mechanisms small; never trust a *cost ratio* measured small.

## Why this matters for the thesis (the payoff)

The project deliberately attacks the **route-length terms**:

- **Continual learning** = *stop re-driving the whole trip* — only drive the
  new part of the map. (Kills the 🟧 "re-driving" row.)
- **Sample efficiency** = *need fewer miles to learn the route* — extract the
  pattern from few examples. (Shrinks the 🟥 "route length" row.)

We are **not** chasing the optimizer (🟨). That's polishing the doorknob while
the house is the cost. The belief "cost is mostly the optimizer" is wrong, but
the *instinct* behind it — "cost has a dominant cause, find it" — is exactly
right. You just had the wrong cause. That instinct is the whole project.
